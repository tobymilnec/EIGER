"""EIGER 2 reference core.

The whole method is five steps. Each is a separate, testable function.

    1. centre_cpgs        remove the per-CpG mean; keep NaN as NaN
    2. estimate_latent_basis / deflate
                          remove K genome-wide sample-space directions,
                          estimated on a THINNED CpG set so that focal
                          regional signal cannot enter the basis
    3. edge_statistic     local co-variation per adjacent CpG pair (or per
                          w-CpG window), pairwise-complete
    4. null_threshold     locality-breaking null built by pairing CpGs that are
                          far apart on the same chromosome; preserves every
                          cross-sample dependency, destroys only adjacency
    5. segment + region_fdr
                          contiguous runs above threshold, then a region-level
                          empirical FDR obtained by pushing null statistic
                          tracks through the *identical* segmentation

Because the FDR is calibrated by running the same segmenter on null tracks, the
segmentation heuristics (window width, gap tolerance, minimum length) do not
need to be individually justified: whatever they do to the data they also do to
the null, and the reported error rate absorbs it.

Characterisation (which samples split, delta beta, direction) happens after a
region has been called and is labelled descriptive throughout. It is
post-selection and no q-value is attached to it.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

__all__ = [
    "Eiger2Config",
    "call_dmrs",
    "centre_cpgs",
    "characterise_region",
    "deflate",
    "edge_statistic",
    "estimate_latent_basis",
    "null_threshold",
    "permutation_null_tracks",
    "qvalues_from_track",
    "region_fdr",
    "region_minority_size",
    "segment",
]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Eiger2Config:
    """Everything the caller can tune. Defaults are the validated settings."""

    # --- discovery statistic
    statistic: str = "correlation"
    """``"correlation"`` (scale-free, recommended) or ``"covariance"``.

    Covariance is retained because it is what EIGER 1 published, and because it
    carries effect-size information that correlation deliberately discards. Be
    aware that for a two-group structure with minor fraction ``f`` and effect
    ``delta`` the adjacent covariance is ``f(1-f)delta**2`` — it is intrinsically
    prevalence-limited and cannot see rare subgroups. Correlation for the same
    structure is ``f(1-f)delta**2 / (f(1-f)delta**2 + sigma**2)``, which
    saturates towards 1 for any ``f`` once ``delta`` exceeds the noise.
    """

    window: int = 4
    """Number of CpGs in the co-variation window. 1 means adjacent pairs.

    ``w > 1`` uses the mean off-diagonal correlation of the ``w x w`` matrix,
    which has roughly ``sqrt(w(w-1)/2)`` times less null variance at no cost in
    the signal case. Worth using below ~30 samples; unnecessary above.
    """

    # --- latent structure removal
    deflate_components: int | None = None
    """Number of sample-space directions to project out. ``None`` = estimate by
    parallel analysis on the thinned set."""

    deflation_thin: int = 50
    """Use every Nth CpG to estimate the latent basis.

    This is the load-bearing detail. A focal DMR of a few dozen CpGs contributes
    at most one row to a 1-in-50 thin and therefore cannot form a component,
    while a genuine genome-wide factor (purity, composition, batch) is fully
    represented. Estimating the basis on *all* CpGs deletes focal signal.
    """

    deflation_max_concentration: float = 0.10
    """Only project out components whose genomic loading is DIFFUSE.

    A nuisance factor (purity, composition, batch) loads on the whole genome; a
    biological subgroup — cell type, tumour status, a rare clone — loads on
    marker regions. Measuring what fraction of a component's loading energy sits
    in the top 1% of CpGs separates them cleanly: nuisance components score
    0.08-0.09, biological components 0.15-0.40.

    Without this filter, blind deflation deletes exactly the signal EIGER exists
    to find. On a cohort of 4 cell types plus tumour status plus a rare subgroup,
    parallel analysis selects K=5 and recovery collapses from 70/70 regions to
    0/70. Set to 1.0 to disable the filter and deflate every component.
    """

    deflation_concentration_quantile: float = 0.01
    deflation_max_components: int = 20

    deflation_min_rows: int = 20
    """Minimum thinned CpGs required to estimate the latent basis.

    The basis is estimated only from CpGs observed in *every* sample, because a
    partially observed row contributes mean-filled zeros to the SVD. Below this
    many complete rows the estimate falls back to partially observed rows and
    records the fallback in the manifest."""

    deflation_min_observed_fraction: float = 0.80
    """Preferred completeness for a thinned CpG entering the basis under the
    fallback described in ``deflation_min_rows``. Relaxed automatically, but
    never below ``deflation_floor_observed_fraction``."""

    deflation_floor_observed_fraction: float = 0.50
    """Absolute floor on completeness for a thinned CpG entering the basis.

    A row observed in fewer than half the samples contributes more mean-fill
    than data to the SVD. If fewer than three rows clear this floor the basis is
    not estimated at all and no deflation is applied, which is recorded in the
    manifest as ``basis_source = insufficient_rows``."""
    parallel_analysis_replicates: int = 5
    parallel_analysis_quantile: float = 0.99
    deflation_missing: str = "exact"
    """``"impute"`` (fast, CpG-mean fill before projection) or ``"exact"``
    (per-CpG least squares on observed samples only)."""

    # --- null calibration
    null_replicates: int = 100
    """Null tracks pooled for the region FDR. At B=20 the strongest region's
    q-value is either exactly 0 or >= 0.05 (resolution = s/1, and s ~ 0.05 at this
    replicate count) -- no resolution at a 0.05 decision boundary. B=100 gives
    resolution ~0.01 and an R=1 upper_95 bound around 0.078; see docs/FDR.md."""
    null_quantile: float = 0.995
    null_min_lag: int = 1000
    """Minimum genomic separation, in retained CpG rows, between the CpGs paired
    to build the null. Must comfortably exceed the methylation correlation
    length."""

    null_stride: int = 1

    null_mode: str = "lag"
    """``"lag"`` or ``"permutation"``.

    ``lag`` builds the null by pairing CpGs far apart on the same chromosome.
    It tests *is there any regional coordination here?* That hypothesis is
    informative at modest cohort sizes but becomes trivially false as the
    cohort grows: with several hundred samples the edge statistic is estimated
    precisely, genuine regional coordination exists across most of the genome,
    and every called region clears a threshold set from a near-zero null. On a
    519-sample cohort every one of 73,784 regions received q = 0, which is
    arithmetically correct and completely uninformative.

    ``permutation`` instead permutes the observed edge statistic within each
    chromosome. The marginal distribution of local coordination is preserved
    exactly -- it is the same multiset of values -- while its genomic
    arrangement is destroyed. The hypothesis becomes *is this run of
    coordination longer or stronger than chance, given how much coordination
    the genome contains?* That question stays answerable at any cohort size,
    because the null can never be shifted away from the data's own marginal.

    Both are sign-agnostic: a hypomethylated region produces the same positive
    adjacent correlation as a hypermethylated one, since both endpoints move
    together. Direction is recovered afterwards from delta beta.

    Use ``permutation`` above roughly 100 samples, or whenever the lag-null
    q-values saturate at zero.
    """

    # --- segmentation
    min_cpgs: int = 4
    max_gap_edges: int = 0
    """Number of consecutive sub-threshold edges tolerated inside a region.

    ``2`` bridges a single dropped or filtered CpG, which breaks two edges and is
    the commonest cause of spurious fragmentation. Safe to raise because the
    null tracks are segmented with the same tolerance.
    """

    # --- data handling
    min_paired_samples: int = 5
    min_paired_fraction: float = 0.0
    """Minimum fraction of the cohort observed at *both* CpGs of an edge.

    Off by default, and measured to be counter-productive on the benchmark: at
    25% structured missingness, raising it from 0.0 to 0.5 dropped recovery from
    45/48 to 0/48 while *increasing* false regions from 17 to 136, because it
    removes DMR edges faster than artefact edges and fragments true regions below
    ``min_cpgs``. Exposed as a diagnostic lever for cohorts with a genuinely
    bimodal coverage profile; do not raise it without re-running the benchmark on
    your own missingness pattern.
    """
    max_cpg_gap_bp: int | None = 300
    min_observed_samples: int = 2

    min_prevalence_samples: int = 0
    """Minimum number of samples that must carry a region's deviation.

    A single aberrant sample deviating across a run of CpGs produces genuine
    adjacent correlation, so it forms a region and is not a false positive under
    any null: the coordination is real. It is simply not a differentially
    methylated *region* in the sense usually intended. On the 519-sample ICGC
    cohort 95% of otherwise unclassifiable regions were driven by one or two
    samples.

    This is a biological criterion rather than a statistical one, so it is
    applied after calibration and reported separately: ``region_qvalue`` is
    calibrated on the unfiltered set, ``minority_size_descriptive`` records the
    size of the deviating group, and ``significant`` requires both the q-value
    and this floor. Setting it to 0 disables the filter. Three is a reasonable
    starting value; it is exactly analogous to requiring a fold-change floor
    alongside a p-value in differential expression.
    """

    min_sample_fraction: float = 0.0
    """Drop a CpG observed in fewer than this fraction of samples.

    The primary control for targeted assays. On an EPIC-style panel most rows
    are absent in every sample or present in only a handful, and they cost time
    at every stage while contributing nothing: an edge needs
    ``min_paired_samples`` co-observed samples, so a sparsely covered CpG cannot
    form a usable edge but is still centred, deflated and scanned.

    Setting this to, say, 0.5 typically removes the majority of rows on a panel
    and is close to free statistically, because those rows could not have
    contributed edges. Dropped CpGs never appear in the output; the counts and
    the realised threshold are recorded in the manifest. Region coordinates are
    reported in the original input indexing, so filtering does not shift them.

    0.0 disables the filter, which is the behaviour of earlier versions.
    """

    length_stratified_fdr: bool = True
    """Compute the region FDR within CpG-count strata (4-5, 6-9, 10-19, 20-49, 50+).

    Region area grows with length, so a pooled FDR makes short regions look worse
    than long ones for a reason that has nothing to do with evidence. Stratifying
    lets a 4-CpG DMR be judged against 4-CpG null regions. Set to ``False`` for a
    single pooled null.
    """

    # --- reporting
    target_fdr: float = 0.05
    random_seed: int = 1

    fdr_decision: str = "point"
    """``"point"``: significant = region_qvalue <= target_fdr (unchanged from
    v2.0.0). ``"upper"``: significant = region_qvalue_upper_95 <= target_fdr, the
    conservative decision under finite-null uncertainty. Never switches silently --
    always read from the manifest's ``fdr_decision`` field."""

    min_null_values: int = 1_000
    """Warn below this many usable null positions.

    The threshold is an upper quantile and needs a populated tail. At the
    default ``null_quantile`` of 0.995 this floor leaves at least five null
    values above the threshold, which is the minimum for it to mean anything.
    Genome-scale runs will exceed it by orders of magnitude; the floor exists to
    catch small or heavily fragmented inputs, not to set a target."""

    min_null_coverage: float = 0.25
    """Warn when the chromosome-aware mask discards more than this fraction of
    offered null positions, which happens when ``null_min_lag`` approaches the
    length of a chromosome."""

    compute_posterior: bool = False
    """Attach the per-region two-groups evidence score, ``1 - fdr(area)``.

    ``region_qvalue`` is a tail-area rate: among regions at least this extreme,
    the expected proportion that are null. ``region_evidence_score`` is the
    per-region quantity from :mod:`eiger2.probability`.

    **The output is named a score, not a probability, because measurement says
    it is not calibrated.** On planted-truth simulations the expected
    calibration error is 0.35 (with ``pi0 = 1``) to 0.50 (with ``pi0``
    estimated): regions scoring 0.86 were real 0% of the time. The cause is
    diagnosed in ``PROBABILITY.md`` §6 — the null tracks are generated from
    signal-free data, whereas false positives in real data arise beside real
    signal and inherit its area distribution, so ``f0`` is the wrong null
    density for them. It is still a useful monotone ranking; it is not a
    probability, and it must not be thresholded as one.

    Off by default. A density also needs far more null regions than a tail count
    does, so enabling this without raising ``null_replicates`` (to 100-300) will
    usually return ``status = insufficient_null_regions`` and leave the columns
    NaN, which is the intended behaviour rather than a failure.
    """

    min_null_regions_for_posterior: int = 50
    """Refuse the posterior below this many pooled null regions."""

    def __post_init__(self) -> None:
        if self.statistic not in {"correlation", "covariance"}:
            raise ValueError("statistic must be 'correlation' or 'covariance'")
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.deflate_components is not None and self.deflate_components < 0:
            raise ValueError("deflate_components cannot be negative")
        if self.deflation_thin < 1:
            raise ValueError("deflation_thin must be at least 1")
        if self.deflation_missing not in {"impute", "exact"}:
            raise ValueError("deflation_missing must be 'impute' or 'exact'")
        if not 0.5 < self.null_quantile < 1.0:
            raise ValueError("null_quantile must lie in (0.5, 1)")
        if self.null_replicates < 1:
            raise ValueError("null_replicates must be positive")
        if self.min_cpgs < 2:
            raise ValueError("min_cpgs must be at least 2")
        if self.max_gap_edges < 0:
            raise ValueError("max_gap_edges cannot be negative")
        if self.min_paired_samples < 3:
            raise ValueError("min_paired_samples must be at least 3")
        if not 0 <= self.min_paired_fraction <= 1:
            raise ValueError("min_paired_fraction must lie in [0, 1]")
        if not 0 < self.target_fdr < 1:
            raise ValueError("target_fdr must lie in (0, 1)")
        if self.fdr_decision not in {"point", "upper"}:
            raise ValueError("fdr_decision must be 'point' or 'upper'")

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# step 1 — centring
# --------------------------------------------------------------------------


def centre_cpgs(betas: np.ndarray) -> np.ndarray:
    """Subtract the per-CpG mean over observed samples. NaN stays NaN."""

    values = np.asarray(betas, dtype=float)
    if values.ndim != 2:
        raise ValueError("betas must be a two-dimensional CpG-by-sample matrix")
    # A CpG with no observation in any sample -- an off-panel or failed probe --
    # yields an empty slice here. That is expected on targeted assays, so the
    # warning is suppressed and the mean is set to zero; the row is dropped by
    # ``min_observed_samples`` in ``call_dmrs`` before it can reach the
    # statistic.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(np.where(np.isfinite(values), values, np.nan), axis=1)
    means = np.where(np.isfinite(means), means, 0.0)
    return values - means[:, None]


# --------------------------------------------------------------------------
# step 2 — latent structure
# --------------------------------------------------------------------------


def _fill_rows(residuals: np.ndarray) -> np.ndarray:
    """Fill missing entries with 0 (the CpG mean, post-centring)."""

    filled = np.array(residuals, dtype=float, copy=True)
    filled[~np.isfinite(filled)] = 0.0
    return filled


def estimate_latent_basis(
    residuals: np.ndarray, config: Eiger2Config
) -> tuple[np.ndarray, dict]:
    """Estimate genome-wide sample-space directions from a thinned CpG set.

    Returns ``(V, diagnostics)`` where ``V`` has shape ``(K, n_samples)`` with
    orthonormal rows. ``K == 0`` yields an empty basis.
    """

    residuals = np.asarray(residuals, dtype=float)
    n_samples = residuals.shape[1]
    raw = residuals[:: config.deflation_thin]
    observed = np.isfinite(raw)

    # The basis must be estimated on rows that are actually observed. Filling
    # first and filtering afterwards makes the filter inert, because nothing is
    # non-finite once the gaps are zeroed, and the zeros then enter the SVD as
    # if they were data. On targeted or panel assays, where per-CpG coverage
    # varies widely, that pulls the estimated basis towards zero exactly in the
    # rows with least information.
    complete = observed.all(axis=1) & (np.nansum(np.abs(raw), axis=1) > 0)
    basis_source = "complete_rows"
    thinned = raw[complete]

    relaxed_to = 1.0
    if thinned.shape[0] < config.deflation_min_rows:
        # Too few complete rows, which is the normal situation on targeted
        # panels and on any cohort with appreciable dropout. Relax the
        # completeness requirement only as far as necessary to reach
        # ``deflation_min_rows``, never below ``deflation_floor_observed_fraction``,
        # and mean-fill the remainder. The threshold actually reached is recorded
        # because it determines how much fill enters the basis.
        fraction = observed.mean(axis=1)
        nonzero = np.nansum(np.abs(raw), axis=1) > 0
        eligible = nonzero & (fraction >= config.deflation_floor_observed_fraction)
        ranked = np.sort(fraction[eligible])[::-1]
        if ranked.size >= 3:
            index = min(config.deflation_min_rows, ranked.size) - 1
            relaxed_to = float(
                min(config.deflation_min_observed_fraction, ranked[index])
            )
            usable = nonzero & (fraction >= relaxed_to)
            thinned = _fill_rows(raw[usable])
            basis_source = "partially_observed_rows_mean_filled"
        else:
            thinned = raw[complete]
            basis_source = "insufficient_rows"

    diagnostics: dict = {
        "thin_stride": int(config.deflation_thin),
        "thin_rows_offered": int(raw.shape[0]),
        "thin_rows_complete": int(complete.sum()),
        "thin_rows_used": int(thinned.shape[0]),
        "basis_source": basis_source,
        "min_observed_fraction_used": relaxed_to,
    }
    if basis_source != "complete_rows":
        diagnostics["warning"] = (
            f"only {int(complete.sum())} of {int(raw.shape[0])} thinned CpGs are "
            f"observed in every sample, so the latent basis was estimated from "
            f"{basis_source}. Reduce deflation_thin to offer more rows, or set "
            f"deflate_components=0 to disable deflation."
        )
    max_components = min(
        config.deflation_max_components, n_samples - 1, max(thinned.shape[0] - 1, 0)
    )
    if thinned.shape[0] < 3 or max_components < 1:
        diagnostics.update(n_components=0, method="insufficient_rows")
        return np.empty((0, n_samples), dtype=float), diagnostics

    _, singular, right = np.linalg.svd(thinned, full_matrices=False)

    if config.deflate_components is not None:
        n_components = int(min(config.deflate_components, max_components))
        diagnostics["method"] = "fixed"
    else:
        rng = np.random.default_rng(config.random_seed)
        null_singular = np.empty(
            (config.parallel_analysis_replicates, singular.size), dtype=float
        )
        for replicate in range(config.parallel_analysis_replicates):
            permuted = np.empty_like(thinned)
            for column in range(n_samples):
                permuted[:, column] = rng.permutation(thinned[:, column])
            null_singular[replicate] = np.linalg.svd(permuted, compute_uv=False)
        cutoff = np.quantile(
            null_singular, config.parallel_analysis_quantile, axis=0
        )
        exceeds = singular > cutoff
        n_components = int(np.argmin(exceeds)) if not exceeds.all() else exceeds.size
        n_components = int(min(n_components, max_components))
        diagnostics["method"] = "parallel_analysis"
        diagnostics["singular_values"] = singular[:10].tolist()
        diagnostics["null_cutoff"] = cutoff[:10].tolist()

    diagnostics["n_candidates"] = n_components
    if n_components == 0:
        diagnostics["n_components"] = 0
        return np.empty((0, n_samples), dtype=float), diagnostics

    # Keep only the candidates whose genomic loading is diffuse. A nuisance
    # factor spreads its loading over the whole genome; a biological subgroup
    # concentrates it in marker regions and must not be deflated away.
    candidates = np.ascontiguousarray(right[:n_components])
    filled_full = _fill_rows(residuals)
    loadings = filled_full @ candidates.T
    quantile = config.deflation_concentration_quantile
    top_n = max(1, int(round(quantile * loadings.shape[0])))
    concentration = np.empty(n_components, dtype=float)
    for component in range(n_components):
        energy = loadings[:, component] ** 2
        total = energy.sum()
        concentration[component] = (
            float(np.sort(energy)[-top_n:].sum() / total) if total > 0 else 1.0
        )
    if config.deflate_components is not None:
        # an explicit K is a user override; report the verdicts but honour it
        diffuse = np.ones(n_components, dtype=bool)
    else:
        diffuse = concentration <= config.deflation_max_concentration
    diagnostics["loading_concentration"] = concentration.round(4).tolist()
    diagnostics["component_verdict"] = [
        "nuisance_deflated" if flag else "biological_kept" for flag in diffuse
    ]
    diagnostics["n_components"] = int(diffuse.sum())
    if not diffuse.any():
        return np.empty((0, n_samples), dtype=float), diagnostics
    return np.ascontiguousarray(candidates[diffuse]), diagnostics


def deflate(
    residuals: np.ndarray, basis: np.ndarray, *, missing: str = "impute"
) -> np.ndarray:
    """Project ``basis`` out of every CpG row, preserving the missing mask."""

    residuals = np.asarray(residuals, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if basis.size == 0:
        return residuals.copy()
    if basis.shape[1] != residuals.shape[1]:
        raise ValueError("basis and residuals must share the sample axis")

    observed = np.isfinite(residuals)
    if missing == "impute" or observed.all():
        filled = _fill_rows(residuals)
        deflated = filled - (filled @ basis.T) @ basis
    elif missing == "exact":
        deflated = np.array(residuals, dtype=float, copy=True)
        for row in range(residuals.shape[0]):
            mask = observed[row]
            if mask.sum() <= basis.shape[0]:
                deflated[row, mask] = 0.0
                continue
            design = basis[:, mask].T
            target = residuals[row, mask]
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
            deflated[row, mask] = target - design @ coefficients
    else:
        raise ValueError("missing must be 'impute' or 'exact'")

    deflated[~observed] = np.nan
    return deflated


# --------------------------------------------------------------------------
# step 3 — the local co-variation statistic
# --------------------------------------------------------------------------


def _pair_statistic(
    left: np.ndarray,
    right: np.ndarray,
    statistic: str,
    min_paired_samples: int,
    min_paired_fraction: float = 0.0,
) -> np.ndarray:
    """Vectorised pairwise-complete covariance or correlation, row by row.

    Pairwise-complete matters: filling missing entries with the CpG mean
    contributes a zero residual to the cross-product while still counting in the
    denominator, which deflates the statistic by exactly ``(1 - m)**2`` at
    missing rate ``m``. Missingness in WGBS tracks GC content, mappability and
    coverage, so that is a structured bias, not noise.
    """

    mask = np.isfinite(left) & np.isfinite(right)
    counts = mask.sum(axis=1)
    left_z = np.where(mask, left, 0.0)
    right_z = np.where(mask, right, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        safe = np.where(counts > 0, counts, 1).astype(float)
        sum_left = left_z.sum(axis=1)
        sum_right = right_z.sum(axis=1)
        cross = (left_z * right_z).sum(axis=1) - sum_left * sum_right / safe
        denominator = np.where(counts > 1, counts - 1, np.nan).astype(float)
        covariance = cross / denominator
        if statistic == "covariance":
            result = covariance
        else:
            var_left = (
                (left_z * left_z).sum(axis=1) - sum_left * sum_left / safe
            ) / denominator
            var_right = (
                (right_z * right_z).sum(axis=1) - sum_right * sum_right / safe
            ) / denominator
            scale = np.sqrt(var_left * var_right)
            result = np.where(scale > 0, covariance / scale, np.nan)

    required = max(
        int(min_paired_samples),
        int(np.ceil(min_paired_fraction * left.shape[1])),
    )
    return np.where(counts >= required, result, np.nan)


def _lag_statistic(
    residuals: np.ndarray, lag: int, config: Eiger2Config, chunk: int = 200_000
) -> np.ndarray:
    """Statistic between CpG ``i`` and CpG ``i + lag`` for every valid ``i``."""

    n_rows = residuals.shape[0]
    n_pairs = max(n_rows - lag, 0)
    out = np.full(n_pairs, np.nan, dtype=float)
    for start in range(0, n_pairs, chunk):
        stop = min(start + chunk, n_pairs)
        out[start:stop] = _pair_statistic(
            residuals[start:stop],
            residuals[start + lag : stop + lag],
            config.statistic,
            config.min_paired_samples,
            config.min_paired_fraction,
        )
    return out


def _window_support(
    finite: np.ndarray,
    base_positions: np.ndarray,
    offsets: np.ndarray,
    *,
    chunk: int = 200_000,
) -> np.ndarray:
    """Per-position count of samples observed at every window member.

    ``finite``: ``(n_cpg, n_samples)`` boolean, finiteness of the underlying
    matrix. ``base_positions``: the reference CpG row for each position.
    ``offsets``: row offsets from each base position making up that position's
    window members -- ``[0, 1, ..., w-1]`` for the observed convention (window
    starts at the base position), ``[0, lag, 2*lag, ..., (w-1)*lag]`` for a null
    position at the given lag. The SAME function computes both, so the two
    conventions cannot silently drift apart -- callers only choose ``offsets``.

    Never materialises an ``(n_positions, n_samples, w)`` array: accumulates a
    boolean AND over ``w`` two-dimensional ``(chunk, n_samples)`` slices instead.
    """

    n_samples = finite.shape[1]
    n_positions = base_positions.size
    out = np.empty(n_positions, dtype=np.int64)
    for start in range(0, n_positions, chunk):
        stop = min(start + chunk, n_positions)
        block_base = base_positions[start:stop]
        acc = np.ones((stop - start, n_samples), dtype=bool)
        for offset in offsets:
            acc &= finite[block_base + offset]
        out[start:stop] = acc.sum(axis=1)
    return out


def _assign_support_bin(support: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Bin index for each support value, given interior bin edges.

    ``bin_edges`` are the interior boundaries (``n_bins - 1`` of them); values
    below the first edge are bin 0, at/above the last edge are bin ``n_bins-1``.
    Shared by observed and null callers so the two are binned identically.
    """

    return np.searchsorted(bin_edges, support, side="right")


def _support_bin_edges(
    observed_support: np.ndarray,
    n_bins: int,
    min_null_values: int,
    pooled_null_support: np.ndarray,
) -> dict:
    """Quantile bin edges from observed support, merged upward until every bin
    has at least ``min_null_values`` pooled null values, per spec §1c.2.

    Returns ``{"bin_edges": np.ndarray, "merges": list[dict], "collapsed": bool}``.
    ``bin_edges`` has one fewer entry than the final bin count (interior edges
    only, consumed by ``_assign_support_bin``). ``collapsed=True`` means every
    candidate merge still left an underpopulated bin, so the caller must fall
    back to the unconditioned single threshold.
    """

    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    candidate_edges = np.quantile(observed_support, quantiles)
    merges: list[dict] = []

    edges = list(candidate_edges)
    while True:
        edges_array = np.array(edges) if edges else np.array([])
        bins = _assign_support_bin(pooled_null_support, edges_array)
        counts = np.bincount(bins, minlength=len(edges) + 1)
        underpopulated = np.flatnonzero(counts < min_null_values)
        if underpopulated.size == 0:
            break
        if len(edges) == 0:
            return {"bin_edges": np.empty(0), "merges": merges, "collapsed": True}
        # merge the first underpopulated bin into its upward neighbour by
        # dropping the edge above it (merges bin b into bin b+1)
        merge_at = int(underpopulated[0])
        drop_edge_index = min(merge_at, len(edges) - 1)
        merges.append({
            "merged_bin": merge_at,
            "into_bin": merge_at + 1 if merge_at < len(edges) else merge_at - 1,
            "null_count_before_merge": int(counts[merge_at]),
        })
        del edges[drop_edge_index]

    return {"bin_edges": np.array(edges), "merges": merges, "collapsed": False}


def _window_from_lags(lag_statistics: list[np.ndarray], window: int) -> np.ndarray:
    """Mean off-diagonal statistic of every ``window``-CpG block.

    ``lag_statistics[d - 1][i]`` is the statistic between CpG ``i`` and
    ``i + d``. The window centred on edge ``e`` covers CpGs
    ``[e - window//2 + 1, e + window//2]``; the returned array is indexed by
    edge, so it aligns with the adjacent-pair track.
    """

    if window == 1:
        return lag_statistics[0].copy()
    n_edges = lag_statistics[0].size
    n_rows = n_edges + 1
    offset = window // 2 - 1
    totals = np.zeros(n_edges, dtype=float)
    counts = np.zeros(n_edges, dtype=float)
    edges = np.arange(n_edges)
    starts = edges - offset
    for lag in range(1, window):
        values = lag_statistics[lag - 1]
        padded = np.concatenate(([0.0], np.nan_to_num(values, nan=0.0)))
        valid = np.concatenate(([0.0], np.isfinite(values).astype(float)))
        cumulative = np.cumsum(padded)
        cumulative_valid = np.cumsum(valid)
        # pairs (i, i+lag) with start <= i and i + lag <= start + window - 1
        lower = np.clip(starts, 0, values.size)
        upper = np.clip(starts + window - lag, 0, values.size)
        upper = np.maximum(upper, lower)
        totals += cumulative[upper] - cumulative[lower]
        counts += cumulative_valid[upper] - cumulative_valid[lower]
    del n_rows
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(counts > 0, totals / np.maximum(counts, 1.0), np.nan)
    expected = window * (window - 1) / 2.0
    return np.where(counts >= 0.5 * expected, result, np.nan)


def edge_statistic(
    residuals: np.ndarray, config: Eiger2Config, chunk: int = 200_000
) -> np.ndarray:
    """Co-variation statistic for every adjacent CpG edge."""

    residuals = np.asarray(residuals, dtype=float)
    if residuals.shape[0] < 2:
        return np.empty(0, dtype=float)
    lags = [
        _lag_statistic(residuals, lag, config, chunk=chunk)
        for lag in range(1, max(config.window, 2))
    ]
    return _window_from_lags(lags, config.window)


def eligible_edges(
    locations: pd.DataFrame, config: Eiger2Config
) -> np.ndarray:
    """Which adjacent edges may sit inside the same region."""

    chrom = locations["chr"].to_numpy()
    same = chrom[:-1] == chrom[1:]
    if config.max_cpg_gap_bp is None:
        return same
    starts = locations["start"].to_numpy(dtype=np.int64)
    ends = locations["end"].to_numpy(dtype=np.int64)
    gaps = starts[1:] - ends[:-1]
    return same & (gaps >= 0) & (gaps <= config.max_cpg_gap_bp)


# --------------------------------------------------------------------------
# step 4 — the locality-breaking null
# --------------------------------------------------------------------------


def _lag_null_track(
    residuals: np.ndarray, lag: int, config: Eiger2Config
) -> np.ndarray:
    """One null track: window members are ``lag`` CpGs apart, not adjacent.

    The window at null position ``i`` comprises CpGs ``i, i+lag, ..., i+(w-1)*lag``
    and the statistic is the mean over the complete pairwise set among *those*
    members: pair ``(a, b)`` is ``S_{b-a}[i + a*lag]`` where
    ``S_d[j] = stat(CpG j, CpG j + d*lag)``.

    Reusing ``_window_from_lags`` here would be wrong: it assumes members sit at
    consecutive indices, which is true of the observed track and false of this
    one. See docs/FDR.md, "Null window assembly".

    The window at null position ``i`` is made of rows ``i, i+lag, i+2*lag, ...``
    instead of ``i, i+1, i+2, ...``. Every sample moves together, so batch,
    purity, cell composition and every other cross-sample dependency survives
    intact — that is what EIGER 1's per-sample resampling null destroyed, which
    is why its threshold came out several-fold too low and DMR-free matrices lit
    up. Consecutive null positions still slide by one CpG, so locally
    autocorrelated nuisance — runs of low coverage, runs of high missingness —
    is preserved as well. Only adjacency, and therefore only regional
    coordination, is broken.
    """

    w = config.window
    span = max(w, 2)
    usable = residuals.shape[0] - lag * (span - 1)
    if usable < 2:
        return np.empty(0, dtype=float)

    S = {d: _lag_statistic(residuals, d * lag, config) for d in range(1, span)}
    if w == 1:
        return S[1][:usable]

    assert (w - 2) * lag + usable <= S[1].size if w >= 3 else True, (
        "slice bounds violated -- see docs/FDR.md, 'Null window assembly'"
    )

    total = np.zeros(usable, dtype=float)
    count = np.zeros(usable, dtype=float)
    for a in range(w):
        for b in range(a + 1, w):
            values = S[b - a][a * lag : a * lag + usable]
            total += np.nan_to_num(values, nan=0.0)
            count += np.isfinite(values)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, total / np.maximum(count, 1.0), np.nan)
    expected = w * (w - 1) / 2.0
    return np.where(count >= 0.5 * expected, out, np.nan)


def _null_eligibility(
    chromosome: np.ndarray | None, lag: int, config: Eiger2Config, size: int
) -> np.ndarray:
    """Which null positions are admissible, by the same rule as the data.

    The observed track is masked by :func:`eligible_edges`, which forbids an
    edge spanning two chromosomes. The null must be masked by a rule of the same
    kind, or the two are not passed through the identical segmenter and the
    calibration argument does not hold. A null position ``i`` at lag ``L`` draws
    on rows ``i, i+L, ..., i+(w-1)L``; it is admissible only if all of them lie
    on one chromosome.
    """

    if chromosome is None:
        return np.ones(size, dtype=bool)
    span = max(config.window, 2)
    eligible = np.ones(size, dtype=bool)
    first = chromosome[:size]
    for step in range(1, span):
        offset = step * lag
        if offset + size > chromosome.size:
            return np.zeros(size, dtype=bool)
        eligible &= first == chromosome[offset : offset + size]
    return eligible


def null_tracks(
    residuals: np.ndarray,
    config: Eiger2Config,
    locations: pd.DataFrame | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Statistic tracks under the locality-breaking null, one per replicate.

    Returns ``(track, eligible)`` pairs. ``eligible`` restricts the null to
    positions whose window members share a chromosome, mirroring the constraint
    applied to the observed track. Passing ``locations=None`` disables the
    restriction and is retained only for single-chromosome use and for tests.
    """

    n_rows = residuals.shape[0]
    span = max(config.window, 2)
    largest = max((n_rows - 2) // max(span - 1, 1), 2)
    low = min(config.null_min_lag, max(2, largest // 4))
    high = max(low + 1, largest)
    chromosome = (
        locations["chr"].to_numpy() if locations is not None else None
    )
    rng = np.random.default_rng(config.random_seed + 977)
    tracks: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(config.null_replicates):
        lag = int(rng.integers(low, high))
        track = _lag_null_track(residuals, lag, config)
        if not track.size:
            continue
        eligible = _null_eligibility(chromosome, lag, config, track.size)
        if eligible.any():
            tracks.append((track, eligible))
    return tracks


def permutation_null_tracks(
    statistic: np.ndarray,
    eligible: np.ndarray,
    chromosome: np.ndarray | None = None,
    *,
    replicates: int = 100,
    seed: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Null tracks that preserve the marginal distribution and destroy the runs.

    The observed edge statistic is permuted within each chromosome. Every value
    is retained exactly once, so the amount of local coordination in the genome
    is unchanged; only its arrangement along the chromosome is randomised. A run
    of consecutive high values in the data is then compared against runs that
    arise by chance from the same pool of values.

    This is the appropriate null once the cohort is large enough that "is there
    any coordination here" is answered yes almost everywhere. Unlike a null
    built from a different pairing of the data, it cannot be shifted away from
    the observed marginal, so the resulting q-values cannot saturate at zero.

    A circular shift would be the obvious alternative and is *wrong* here: it
    preserves runs intact and merely moves them, so every observed region
    reappears in the null and the q-values collapse to one.

    Returns ``(track, eligible)`` pairs matching :func:`null_tracks`.
    """

    statistic = np.asarray(statistic, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    rng = np.random.default_rng(seed + 5209)
    if chromosome is None:
        blocks = [np.arange(statistic.size)]
    else:
        chromosome = np.asarray(chromosome)[: statistic.size]
        blocks = [
            np.flatnonzero(chromosome == value)
            for value in pd.unique(chromosome)
        ]

    tracks: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(replicates):
        shuffled = np.full(statistic.size, np.nan, dtype=float)
        for rows in blocks:
            usable = rows[eligible[rows] & np.isfinite(statistic[rows])]
            if usable.size < 2:
                continue
            shuffled[usable] = rng.permutation(statistic[usable])
        tracks.append((shuffled, np.isfinite(shuffled)))
    return tracks


def region_minority_size(
    values: np.ndarray, region: Sequence[int], matrix: np.ndarray
) -> tuple[int, float]:
    """Size of the deviating sample group for a region, and its effect.

    Splits the per-sample region means at their largest gap and returns the size
    of the *minority* side together with the difference between the two cluster
    means. The minority side is returned rather than the low side so that a
    hypermethylated subgroup is not reported as a hypomethylated majority.
    """

    first, last = int(region[0]), int(region[1])
    block = matrix[first : last + 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(block, axis=0)
    finite = np.isfinite(means)
    if finite.sum() < 4:
        return 0, 0.0
    index = np.flatnonzero(finite)
    order = index[np.argsort(means[index])]
    values = means[order]
    n = values.size
    if n < 4:
        return 0, 0.0

    # Split by minimising within-cluster sum of squares over every cut point,
    # not by taking the largest adjacent gap.
    #
    # The largest gap is the criterion the original find_separators used and it
    # fails badly once the cohort is large. With several hundred samples the
    # sorted region means are near-continuous, so the biggest single gap sits
    # beside an outlier rather than at a group boundary: measured on 3,000
    # regions whose tumour and benign group means differed by at least 0.20, the
    # largest gap returned a minority of one sample in the median case, while
    # two-means returned 221 -- the correct size for a 267 versus 227 split.
    # Two-means also still isolates genuinely small subgroups, because four
    # strongly displaced samples minimise the criterion at a 4-versus-rest cut.
    cumulative = np.cumsum(values)
    cumulative_square = np.cumsum(values * values)
    left = np.arange(1, n)
    mean_left = cumulative[:-1] / left
    mean_right = (cumulative[-1] - cumulative[:-1]) / (n - left)
    within = (
        (cumulative_square[:-1] - left * mean_left ** 2)
        + ((cumulative_square[-1] - cumulative_square[:-1])
           - (n - left) * mean_right ** 2)
    )
    cut = int(np.argmin(within))
    delta = float(abs(mean_left[cut] - mean_right[cut]))
    return int(min(cut + 1, n - cut - 1)), delta


def null_exchangeability(
    tracks: list[tuple[np.ndarray, np.ndarray]],
    statistic: np.ndarray,
    eligible: np.ndarray,
    config: Eiger2Config,
    *,
    calibration_fraction: float = 0.5,
) -> dict:
    """Split-half check that the null distribution matches the observed one.

    Splits the null replicates into a calibration half (sets the threshold) and a
    held-out half (measures exceedance against it). ``p_null`` is the held-out
    exceedance and is a control: by construction it must equal
    ``1 - config.null_quantile``, confirming the null is internally self-consistent.
    ``p_obs`` is the observed track's exceedance against the SAME threshold. The
    ratio ``p_obs / p_null`` is the actual gate: it is 1.0 when the null and the
    data share the same upper-tail shape, and rises when the observed track is
    more heavy-tailed than the null represents -- exactly the failure mode
    described in docs/FDR.md, "Null exchangeability".

    The split is deterministic given ``config.random_seed`` -- a fixed permutation
    of replicate indices, not re-derived from tracks' own order, so results are
    reproducible independent of how many replicates happen to survive filtering
    upstream.
    """

    n = len(tracks)
    rng = np.random.default_rng(config.random_seed + 30011)
    order = rng.permutation(n)
    n_calibration = max(1, int(round(n * calibration_fraction)))
    calibration_idx = order[:n_calibration]
    holdout_idx = order[n_calibration:]

    def _pool(indices):
        if indices.size == 0:
            return np.empty(0, dtype=float)
        values = np.concatenate([
            tracks[i][0][np.isfinite(tracks[i][0]) & tracks[i][1]]
            for i in indices
        ])
        return values

    calibration_pool = _pool(calibration_idx)
    holdout_pool = _pool(holdout_idx)

    calibration_threshold = (
        float(np.quantile(calibration_pool, config.null_quantile))
        if calibration_pool.size else float("nan")
    )

    p_null = (
        float(np.mean(holdout_pool >= calibration_threshold))
        if holdout_pool.size and np.isfinite(calibration_threshold) else float("nan")
    )

    observed_eligible = statistic[eligible & np.isfinite(statistic)]
    p_obs = (
        float(np.mean(observed_eligible >= calibration_threshold))
        if observed_eligible.size and np.isfinite(calibration_threshold)
        else float("nan")
    )

    ratio = (
        p_obs / p_null if p_null and np.isfinite(p_null) and p_null > 0
        else float("nan")
    )

    return {
        "p_null": p_null,
        "p_obs": p_obs,
        "null_exchangeability_ratio": ratio,
        "n_calibration_replicates": int(calibration_idx.size),
        "n_holdout_replicates": int(holdout_idx.size),
        "calibration_threshold": calibration_threshold,
    }


def null_threshold(
    residuals: np.ndarray,
    config: Eiger2Config,
    locations: pd.DataFrame | None = None,
    tracks: list | None = None,
    *,
    statistic: np.ndarray | None = None,
    eligible: np.ndarray | None = None,
) -> tuple[float, dict]:
    """Upper-quantile threshold of the null statistic distribution."""

    if tracks is None:
        tracks = null_tracks(residuals, config, locations)
    pooled = (
        np.concatenate([t[e] for t, e in tracks]) if tracks else np.empty(0)
    )
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        raise ValueError(
            "the locality-breaking null produced no usable pairs; the matrix is "
            "probably too small or too sparse"
        )
    threshold = float(np.quantile(pooled, config.null_quantile))
    # A chromosome-aware null discards positions whose window members straddle a
    # chromosome boundary. With many short chromosomes and a large lag this can
    # remove most of the null, leaving the threshold poorly determined, so the
    # retained fraction is reported and flagged rather than left implicit.
    offered = sum(int(t.size) for t, _ in tracks) if tracks else 0
    coverage = float(pooled.size / offered) if offered else 0.0
    diagnostics = {
        "null_replicates": len(tracks),
        "null_values": int(pooled.size),
        "null_positions_offered": offered,
        "null_coverage": coverage,
        "null_quantile": config.null_quantile,
        "null_median": float(np.median(pooled)),
        "threshold": threshold,
    }
    if statistic is not None and eligible is not None:
        diagnostics.update(
            null_exchangeability(tracks, statistic, eligible, config)
        )
    if pooled.size < config.min_null_values or coverage < config.min_null_coverage:
        diagnostics["warning"] = (
            f"null is thin: {pooled.size} usable positions "
            f"({coverage:.1%} of those offered). The threshold and every "
            f"q-value derived from it are unreliable. Reduce null_min_lag, "
            f"raise null_replicates, or call chromosomes separately."
        )
    return threshold, diagnostics


# --------------------------------------------------------------------------
# step 5 — segmentation and region-level FDR
# --------------------------------------------------------------------------


def segment(
    statistic: np.ndarray,
    eligible: np.ndarray,
    threshold: float | np.ndarray,
    *,
    min_cpgs: int,
    max_gap_edges: int = 0,
) -> np.ndarray:
    """Contiguous runs of passing edges, tolerating short internal gaps.

    ``threshold`` may be a scalar or a per-edge array matching ``statistic``'s
    shape (support-conditioned mode, Stage 1c). The scalar path is unchanged and
    stays bit-identical.

    Returns an ``(n_regions, 2)`` array of inclusive CpG index bounds. A run of
    ``E`` passing edges spans ``E + 1`` CpGs; ``min_cpgs`` counts CpGs.
    """

    statistic = np.asarray(statistic, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if statistic.shape != eligible.shape:
        raise ValueError("statistic and eligible must have the same shape")
    threshold_arr = np.asarray(threshold, dtype=float)
    if threshold_arr.shape not in ((), statistic.shape):
        raise ValueError("threshold must be a scalar or match statistic's shape")
    passing = np.isfinite(statistic) & (statistic >= threshold_arr) & eligible

    regions: list[tuple[int, int]] = []
    index = 0
    n_edges = passing.size
    while index < n_edges:
        if not passing[index]:
            index += 1
            continue
        start = index
        end = index
        cursor = index + 1
        gap = 0
        while cursor < n_edges:
            if passing[cursor]:
                end = cursor
                gap = 0
            elif eligible[cursor] and gap < max_gap_edges:
                gap += 1
            else:
                break
            cursor += 1
        if (end - start + 2) >= min_cpgs:
            regions.append((start, end + 1))
        index = max(end + 1, start + 1)
    if not regions:
        return np.empty((0, 2), dtype=int)
    return np.asarray(regions, dtype=int)


def region_area(
    statistic: np.ndarray, regions: np.ndarray, threshold: float | np.ndarray
) -> np.ndarray:
    """Summed excess of the statistic over the threshold, per region.

    ``threshold`` may be a scalar or a per-edge array matching ``statistic``'s
    shape (Stage 1c support-conditioned mode); the scalar path is unchanged.

    Area rewards both strength and length, which is the behaviour wanted for a
    region-level statistic and matches the ``bumphunter`` convention.
    """

    if regions.size == 0:
        return np.empty(0, dtype=float)
    values = np.asarray(statistic, dtype=float)
    threshold_arr = np.asarray(threshold, dtype=float)
    if threshold_arr.shape not in ((), values.shape):
        raise ValueError("threshold must be a scalar or match statistic's shape")
    is_array_threshold = threshold_arr.shape == values.shape
    areas = np.empty(len(regions), dtype=float)
    for position, (first, last) in enumerate(regions):
        first, last = int(first), int(last)
        block = values[first:last]
        block_threshold = (
            threshold_arr[first:last] if is_array_threshold else threshold_arr
        )
        finite_mask = np.isfinite(block)
        block = block[finite_mask]
        block_threshold = (
            block_threshold[finite_mask] if is_array_threshold else block_threshold
        )
        areas[position] = float(np.sum(block - block_threshold)) if block.size else 0.0
    return areas


def _n_distinct(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    ordered = np.sort(values)
    return int(1 + np.sum(ordered[1:] != ordered[:-1]))


@dataclass(frozen=True)
class RegionFdrResult:
    qvalue: np.ndarray
    tail_fdr: np.ndarray
    qvalue_lower: np.ndarray
    qvalue_upper: np.ndarray
    qvalue_resolution: np.ndarray
    null_exceedances: np.ndarray
    qvalue_cutoff_rank: np.ndarray
    confidence_level: float
    diagnostics: dict


def _fdr_from_areas_result(
    observed_areas: np.ndarray,
    pooled_null: np.ndarray,
    scale: float,
    *,
    confidence_level: float,
    m_bonferroni: int,
) -> dict:
    """Tail-area FDR, Garwood bounds and resolution for one length stratum.

    ``raw``/``qvalue`` are the corrected reverse-cumulative-minimum tail estimate:
    the previous ``np.minimum.accumulate`` scanned strongest-to-weakest FORWARD,
    which propagates the strongest region's near-universal zero exceedance count to
    every weaker region. ``q_i = min_{j>=i} FDR_j`` requires the reverse scan.
    Distinct-cutoff grouping (rather than sequential rank) makes tied areas share a
    denominator, so two regions with identical area are not scored as if one were
    stronger than the other.
    """

    m = observed_areas.size
    pooled = np.sort(pooled_null[np.isfinite(pooled_null)])

    # --- distinct cutoffs, strongest first. EXACT float equality; no rounding bins.
    order = np.argsort(-observed_areas, kind="mergesort")
    ranked = observed_areas[order]
    new_group = np.empty(m, dtype=bool)
    new_group[0] = True
    new_group[1:] = ranked[1:] != ranked[:-1]
    group_of_row = np.cumsum(new_group) - 1            # 0..G-1, per ranked row
    G = int(group_of_row[-1]) + 1
    cutoff = ranked[new_group]                          # strictly decreasing

    # R_g: rows with area >= cutoff[g] == cumulative count through end of group g
    last_row_of_group = np.flatnonzero(np.append(new_group[1:], True))
    R = (last_row_of_group + 1).astype(float)
    K = (pooled.size - np.searchsorted(pooled, cutoff, side="left")).astype(float)

    raw   = np.minimum(1.0, scale * K / R)
    delta = np.minimum(1.0, scale / R)

    alpha  = 1.0 - confidence_level
    astar  = alpha / max(int(m_bonferroni), 1)
    L = np.where(K > 0, 0.5 * chi2.ppf(astar / 2.0, 2.0 * K), 0.0)
    U = 0.5 * chi2.ppf(1.0 - astar / 2.0, 2.0 * (K + 1.0))
    lo = np.minimum(1.0, scale * L / R)
    hi = np.minimum(1.0, scale * U / R)

    # --- reverse cumulative minimum, with argmin, over GROUPS
    rev      = raw[::-1]
    run_min  = np.minimum.accumulate(rev)
    # index of the achieving cutoff; ties resolve to the STRONGER (later in rev order)
    cand     = np.where(rev == run_min, np.arange(G), -1)
    arg_rev  = np.maximum.accumulate(cand)
    q_group     = run_min[::-1]
    qarg_group  = (G - 1) - arg_rev[::-1]
    q_lo_group  = np.minimum.accumulate(lo[::-1])[::-1]
    q_hi_group  = np.minimum.accumulate(hi[::-1])[::-1]

    # --- expand groups back to ranked rows, then restore original row order
    def _expand(v):
        out = np.empty(m, dtype=v.dtype)
        out[order] = v[group_of_row]
        return out

    return {
        "qvalue":            _expand(q_group),
        "tail_fdr":          _expand(raw),  # region's OWN cutoff, not q-achieving
        "qvalue_lower":      _expand(q_lo_group),
        "qvalue_upper":      _expand(q_hi_group),
        "qvalue_resolution": _expand(delta[qarg_group]),
        "null_exceedances":  _expand(K[qarg_group]),
        "qvalue_cutoff_rank":_expand(R[qarg_group].astype(np.int64)),
        "cutoffs":           cutoff,
        "_n_cutoffs":        G,
    }


def _fdr_from_areas(
    observed_areas: np.ndarray, pooled_null: np.ndarray, scale: float
) -> np.ndarray:
    return _fdr_from_areas_result(
        observed_areas, pooled_null, scale,
        confidence_level=0.95, m_bonferroni=int(np.unique(observed_areas).size) or 1,
    )["qvalue"]


def length_stratum(n_cpg: int) -> str:
    """Bin a region by CpG count for length-stratified FDR."""

    if n_cpg <= 5:
        return "4-5"
    if n_cpg <= 9:
        return "6-9"
    if n_cpg <= 19:
        return "10-19"
    if n_cpg <= 49:
        return "20-49"
    return "50+"


def _overdispersion_diagnostic(
    per_replicate_null_areas: list[np.ndarray], cutoffs: np.ndarray
) -> tuple[str, float | None]:
    """Poisson-vs-observed dispersion check on pooled null exceedance counts.

    ``region_qvalue_upper_95`` assumes a Poisson count model (Garwood). If the
    replicate-to-replicate exceedance counts are overdispersed relative to Poisson,
    the Garwood interval understates the true uncertainty -- it becomes a LOWER
    bound on the real interval, not a correct one. This is reported, never silently
    corrected (spec §1.5: no uncalibrated dispersion correction in this release).
    """

    b = len(per_replicate_null_areas)
    if b < 5 or cutoffs.size == 0:
        return "unavailable", None

    counts = np.empty((b, cutoffs.size), dtype=float)
    for row, areas in enumerate(per_replicate_null_areas):
        sorted_areas = np.sort(areas)
        counts[row] = areas.size - np.searchsorted(sorted_areas, cutoffs, side="left")

    usable = np.mean(counts, axis=0) > 0
    if not np.any(usable):
        return "unavailable", None

    means = np.mean(counts[:, usable], axis=0)
    variances = np.var(counts[:, usable], axis=0, ddof=1)
    ratios = variances / means
    return "computed", float(np.median(ratios))


def region_fdr_result(
    observed_areas: np.ndarray,
    null_area_sets: Sequence[np.ndarray],
    *,
    observed_edges: int,
    null_edges: int,
    observed_lengths: np.ndarray | None = None,
    null_length_sets: Sequence[np.ndarray] | None = None,
    confidence_level: float = 0.95,
) -> RegionFdrResult:
    """Region-level empirical FDR: point estimate, resolution, and simultaneous
    Garwood Monte Carlo bounds, computed within length strata with a common
    Bonferroni ``m`` (spec §1.4) so ``qvalue_upper`` is comparable across strata.

    For a cut-off ``a``, the expected number of null regions at least as extreme
    is the mean count across replicates, rescaled to the size of the real
    genome. The FDR is that expectation divided by the observed count.

    When ``observed_lengths`` and ``null_length_sets`` are supplied the
    calculation is done **within length strata**. This matters because region
    area grows with length, so a pooled FDR silently makes short regions look
    worse than long ones and pushes the caller towards a minimum-length cutoff.
    A 4-CpG DMR is real biology; it should compete against 4-CpG null regions,
    not against 50-CpG ones.

    See ``docs/FDR.md`` for the derivation and the three distinct quantities this
    keeps apart: the point estimate, the empirical resolution, and the Monte Carlo
    interval.
    """

    observed_areas = np.asarray(observed_areas, dtype=float)
    m = observed_areas.size
    empty_f = np.empty(0, dtype=float)
    empty_i = np.empty(0, dtype=np.int64)
    if m == 0:
        return RegionFdrResult(
            qvalue=empty_f, tail_fdr=empty_f, qvalue_lower=empty_f,
            qvalue_upper=empty_f, qvalue_resolution=empty_f,
            null_exceedances=empty_i, qvalue_cutoff_rank=empty_i,
            confidence_level=confidence_level,
            diagnostics={"pooled_fallback_strata": []},
        )

    replicates = len(null_area_sets)
    if replicates == 0 or null_edges <= 0:
        nan_f = np.full(m, np.nan, dtype=float)
        neg_i = np.full(m, -1, dtype=np.int64)
        status = (
            "no null replicates available" if replicates == 0 else "null_edges <= 0"
        )
        return RegionFdrResult(
            qvalue=nan_f, tail_fdr=nan_f, qvalue_lower=nan_f,
            qvalue_upper=nan_f, qvalue_resolution=nan_f,
            null_exceedances=neg_i, qvalue_cutoff_rank=neg_i,
            confidence_level=confidence_level,
            diagnostics={"status": status, "pooled_fallback_strata": []},
        )

    # ``null_edges`` is the TOTAL number of null edges summed over replicates, and
    # ``null_area_sets`` are likewise pooled over replicates, so the two already
    # correspond -- see the identical comment preserved on the old region_fdr.
    scale = observed_edges / null_edges
    null_area_sets = [np.asarray(a, dtype=float) for a in null_area_sets]

    stratified = observed_lengths is not None and null_length_sets is not None
    if stratified:
        observed_lengths = np.asarray(observed_lengths, dtype=int)
        null_length_sets = [np.asarray(a, dtype=int) for a in null_length_sets]
        observed_strata = np.array(
            [length_stratum(int(n)) for n in observed_lengths], dtype=object
        )
        null_strata_by_replicate = [
            np.array([length_stratum(int(n)) for n in lengths], dtype=object)
            for lengths in null_length_sets
        ]
    else:
        observed_strata = np.full(m, "_pooled", dtype=object)
        null_strata_by_replicate = [
            np.full(areas.size, "_pooled", dtype=object) for areas in null_area_sets
        ]

    strata = np.unique(observed_strata)

    # --- Pass A: distinct-cutoff count per stratum, then the common m_bonferroni
    cutoffs_by_stratum = {
        str(s): _n_distinct(observed_areas[observed_strata == s]) for s in strata
    }
    m_bonferroni = max(cutoffs_by_stratum.values())

    # --- Pass B: run the estimator per stratum with that common m
    qvalue = np.empty(m, dtype=float)
    tail_fdr = np.empty(m, dtype=float)
    qvalue_lower = np.empty(m, dtype=float)
    qvalue_upper = np.empty(m, dtype=float)
    qvalue_resolution = np.empty(m, dtype=float)
    null_exceedances = np.empty(m, dtype=np.int64)
    qvalue_cutoff_rank = np.empty(m, dtype=np.int64)
    pooled_fallback_strata: list[str] = []
    overdispersion_by_stratum: dict[str, dict] = {}

    for stratum in strata:
        rows = np.flatnonzero(observed_strata == stratum)
        per_replicate = [
            null_area_sets[b][null_strata_by_replicate[b] == stratum]
            for b in range(replicates)
        ]
        pooled = np.concatenate(per_replicate) if per_replicate else np.empty(0)
        if pooled.size == 0:
            pooled_fallback_strata.append(str(stratum))
            pooled = np.concatenate(null_area_sets)
            per_replicate = null_area_sets
        piece = _fdr_from_areas_result(
            observed_areas[rows], pooled, scale,
            confidence_level=confidence_level, m_bonferroni=m_bonferroni,
        )
        qvalue[rows] = piece["qvalue"]
        tail_fdr[rows] = piece["tail_fdr"]
        qvalue_lower[rows] = piece["qvalue_lower"]
        qvalue_upper[rows] = piece["qvalue_upper"]
        qvalue_resolution[rows] = piece["qvalue_resolution"]
        null_exceedances[rows] = piece["null_exceedances"].astype(np.int64)
        qvalue_cutoff_rank[rows] = piece["qvalue_cutoff_rank"]
        overdispersion_by_stratum[str(stratum)] = _overdispersion_diagnostic(
            per_replicate, piece["cutoffs"]
        )

    dispersion_ratios = [
        ratio for status, ratio in overdispersion_by_stratum.values()
        if status == "computed"
    ]
    overdispersion_warning = (
        max(dispersion_ratios) if dispersion_ratios and max(dispersion_ratios) > 1.5
        else None
    )

    diagnostics = {
        "fdr_interval_bonferroni_m": m_bonferroni,
        "fdr_interval_cutoffs_by_stratum": cutoffs_by_stratum,
        "fdr_interval_family": "per_stratum",
        "pooled_fallback_strata": pooled_fallback_strata,
        "null_count_dispersion_by_stratum": overdispersion_by_stratum,
        "null_count_overdispersion_warning": overdispersion_warning,
        "scale": scale,
    }
    return RegionFdrResult(
        qvalue=qvalue, tail_fdr=tail_fdr, qvalue_lower=qvalue_lower,
        qvalue_upper=qvalue_upper, qvalue_resolution=qvalue_resolution,
        null_exceedances=null_exceedances, qvalue_cutoff_rank=qvalue_cutoff_rank,
        confidence_level=confidence_level, diagnostics=diagnostics,
    )


def check_fdr_decision_vacuity(
    result: RegionFdrResult,
    *,
    scale: float,
    current_replicates: int,
    target_fdr: float,
) -> None:
    """Refuse to silently return zero significant regions under --fdr-decision upper.

    The best achievable upper_95 bound at the strongest possible cutoff (K=0, R=1)
    is s * U(K=0). If that already exceeds target_fdr, no region in this run can
    ever pass the upper-bound decision rule regardless of its area -- report the
    replicate count that would fix it instead of returning an empty, misleading
    result.
    """

    if scale <= 0:
        return
    alpha = 1.0 - result.confidence_level
    m_bonferroni = result.diagnostics.get("fdr_interval_bonferroni_m", 1)
    astar = alpha / max(int(m_bonferroni), 1)
    u0 = 0.5 * chi2.ppf(1.0 - astar / 2.0, 2.0)
    best_upper = min(1.0, scale * u0)
    if best_upper <= target_fdr:
        return
    required = math.ceil(current_replicates * scale * u0 / target_fdr)
    raise ValueError(
        f"--fdr-decision upper cannot reach target_fdr={target_fdr}: current "
        f"null_replicates={current_replicates} gives scale={scale:.6f} and the "
        f"best achievable region_qvalue_upper_95 is {best_upper:.4f} at K=0, R=1. "
        f"Increase --null-replicates to at least {required} (B_required)."
    )


def region_fdr(
    observed_areas: np.ndarray,
    null_area_sets: Sequence[np.ndarray],
    *,
    observed_edges: int,
    null_edges: int,
    observed_lengths: np.ndarray | None = None,
    null_length_sets: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Monotone region-level empirical FDR. Signature frozen; see region_fdr_result
    for the resolution and Garwood-bound fields."""

    return region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=observed_edges, null_edges=null_edges,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    ).qvalue


def qvalue_result_from_track(
    statistic: np.ndarray,
    regions: np.ndarray,
    threshold: float,
    *,
    eligible: np.ndarray | None = None,
    chromosome: np.ndarray | None = None,
    min_cpgs: int = 5,
    max_gap_edges: int = 0,
    replicates: int = 100,
    seed: int = 1,
    length_stratified: bool = True,
) -> tuple[RegionFdrResult, dict]:
    """Region-level FDR (point estimate, resolution, and Garwood bounds) for ANY
    caller, from its own statistic track. See ``qvalues_from_track`` for the
    thin-wrapper form used where only the point estimate is needed."""

    statistic = np.asarray(statistic, dtype=float)
    regions = np.asarray(regions, dtype=int).reshape(-1, 2)
    if eligible is None:
        eligible = np.isfinite(statistic)
    if regions.size == 0:
        empty = region_fdr_result(
            np.empty(0), [], observed_edges=0, null_edges=0,
        )
        return empty, {"n_regions": 0}

    observed_areas = region_area(statistic, regions, threshold)
    observed_lengths = regions[:, 1] - regions[:, 0] + 1

    tracks = permutation_null_tracks(
        statistic, eligible, chromosome, replicates=replicates, seed=seed
    )
    null_areas, null_lengths, null_edges = [], [], 0
    for track, usable in tracks:
        null_edges += int(usable.sum())
        found = segment(
            track, usable, threshold,
            min_cpgs=min_cpgs, max_gap_edges=max_gap_edges,
        )
        null_areas.append(region_area(track, found, threshold))
        null_lengths.append(
            (found[:, 1] - found[:, 0] + 1) if found.size
            else np.empty(0, dtype=int)
        )

    observed_edges = int(np.sum(eligible & np.isfinite(statistic)))
    result = region_fdr_result(
        observed_areas, null_areas,
        observed_edges=observed_edges,
        null_edges=max(null_edges, 1),
        observed_lengths=observed_lengths if length_stratified else None,
        null_length_sets=null_lengths if length_stratified else None,
    )
    diagnostics = {
        "mode": "permutation",
        "n_regions": len(regions),
        "null_replicates": len(tracks),
        "null_regions_total": int(sum(len(a) for a in null_areas)),
        "null_regions_per_replicate": (
            float(np.mean([len(a) for a in null_areas])) if null_areas else 0.0
        ),
        "threshold": float(threshold),
        "observed_edges": observed_edges,
        "null_edges": max(null_edges, 1),
        "scale": observed_edges / max(null_edges, 1),
        "null_count_overdispersion_warning": result.diagnostics.get(
            "null_count_overdispersion_warning"
        ),
    }
    return result, diagnostics


def qvalues_from_track(
    statistic: np.ndarray,
    regions: np.ndarray,
    threshold: float,
    *,
    eligible: np.ndarray | None = None,
    chromosome: np.ndarray | None = None,
    min_cpgs: int = 5,
    max_gap_edges: int = 0,
    replicates: int = 100,
    seed: int = 1,
    length_stratified: bool = True,
) -> tuple[np.ndarray, dict]:
    """Thin wrapper over qvalue_result_from_track returning only the point estimate.

    Kept for existing callers; signature unchanged.
    """

    result, diagnostics = qvalue_result_from_track(
        statistic, regions, threshold,
        eligible=eligible, chromosome=chromosome, min_cpgs=min_cpgs,
        max_gap_edges=max_gap_edges, replicates=replicates, seed=seed,
        length_stratified=length_stratified,
    )
    return result.qvalue, diagnostics


# --------------------------------------------------------------------------
# characterisation — strictly post-selection, strictly descriptive
# --------------------------------------------------------------------------


def characterise_region(values: np.ndarray) -> dict:
    """Describe the sample structure of an already-called region.

    The split is chosen to maximise separation and then measured on the same
    data, so ``delta_beta`` is upward-biased under the null and the returned
    ``pvalue_descriptive`` is not a valid p-value. Both are reported because they
    are useful for reading a region; neither may be used to decide whether the
    region is real. That decision belongs to ``region_qvalue``.
    """

    values = np.asarray(values, dtype=float)
    observed_index = np.flatnonzero(np.isfinite(values))
    observed = values[observed_index]
    empty = {
        "n_observed": int(observed.size),
        "mean_beta": float(np.nan),
        "delta_beta": float(np.nan),
        "largest_gap": float(np.nan),
        "direction": "Uncertain",
        "minor_group_size": 0,
        "minor_group_fraction": float(np.nan),
        "minor_group_samples": np.array([], dtype=int),
        "pvalue_descriptive": float(np.nan),
    }
    if observed.size < 3:
        return empty

    order = np.argsort(observed, kind="mergesort")
    ordered = observed[order]
    gaps = np.diff(ordered)
    split = int(np.argmax(gaps))
    lower = observed_index[order[: split + 1]]
    upper = observed_index[order[split + 1 :]]
    if len(lower) <= len(upper):
        minor, major = lower, upper
    else:
        minor, major = upper, lower

    minor_values = values[minor]
    major_values = values[major]
    delta = float(np.mean(minor_values) - np.mean(major_values))

    pvalue = np.nan
    if minor.size >= 2 and major.size >= 2:
        from scipy import stats

        spread = np.concatenate([minor_values, major_values])
        if np.ptp(spread) > 0:
            pvalue = float(
                stats.ttest_ind(
                    minor_values, major_values, equal_var=False
                ).pvalue
            )

    return {
        "n_observed": int(observed.size),
        "mean_beta": float(np.mean(observed)),
        "delta_beta": delta,
        "largest_gap": float(gaps[split]),
        "direction": "Hypomethylation" if delta < 0 else "Hypermethylation",
        "minor_group_size": int(minor.size),
        "minor_group_fraction": float(minor.size / observed.size),
        "minor_group_samples": minor,
        "pvalue_descriptive": pvalue,
    }


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def _standardise_locations(locations, n_rows: int) -> pd.DataFrame:
    frame = pd.DataFrame(locations).copy()
    if len(frame) != n_rows:
        raise ValueError("locations and betas must have the same number of rows")
    lower = {str(c).lower(): c for c in frame.columns}
    chrom_column = next(
        (lower[k] for k in ("chr", "chrom", "chromosome") if k in lower),
        frame.columns[0],
    )
    start_column = lower.get("start", frame.columns[1])
    starts = pd.to_numeric(frame[start_column], errors="raise").astype("int64")
    end_column = lower.get("end")
    ends = (
        pd.to_numeric(frame[end_column], errors="raise").astype("int64")
        if end_column is not None
        else starts + 1
    )
    return pd.DataFrame(
        {
            "chr": frame[chrom_column].astype(str).to_numpy(),
            "start": starts.to_numpy(),
            "end": ends.to_numpy(),
        }
    )


def call_dmrs(
    betas,
    locations,
    *,
    sample_names: Sequence[str] | None = None,
    config: Eiger2Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Discover regions of coordinated methylation and describe them.

    Returns ``(regions, region_values, manifest)``.

    ``regions`` carries one row per region. ``region_qvalue`` is the only field
    that expresses confidence that the region is real; every ``*_descriptive``
    field is post-selection and is for reading, not for filtering.
    """

    config = config or Eiger2Config()
    matrix = np.asarray(betas, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("betas must be a two-dimensional CpG-by-sample matrix")
    n_cpg, n_samples = matrix.shape
    if n_samples < config.min_paired_samples:
        raise ValueError(
            f"at least {config.min_paired_samples} samples are required "
            f"(min_paired_samples); got {n_samples}"
        )
    if sample_names is None:
        sample_names = [f"sample_{i + 1}" for i in range(n_samples)]
    sample_names = [str(s) for s in sample_names]
    if len(sample_names) != n_samples or len(set(sample_names)) != n_samples:
        raise ValueError("sample_names must be unique and match the matrix columns")

    frame = _standardise_locations(locations, n_cpg)
    observed_per_cpg = np.sum(np.isfinite(matrix), axis=1)
    keep = observed_per_cpg >= config.min_observed_samples
    if config.min_sample_fraction > 0:
        keep &= observed_per_cpg >= np.ceil(
            config.min_sample_fraction * n_samples
        )
    matrix = matrix[keep]
    frame = frame.loc[keep].reset_index(drop=True)
    source_index = np.flatnonzero(keep)

    manifest: dict = {
        "config": config.to_dict(),
        "n_cpg_input": int(n_cpg),
        "n_cpg_retained": int(matrix.shape[0]),
        "n_cpg_dropped_unobserved": int(np.sum(observed_per_cpg == 0)),
        "n_cpg_dropped_low_coverage": int(
            np.sum(~keep) - np.sum(observed_per_cpg == 0)
        ),
        "min_samples_required": int(
            max(config.min_observed_samples,
                np.ceil(config.min_sample_fraction * n_samples))
        ),
        "n_samples": int(n_samples),
        "schema_version": "2.1",
        "fdr_decision": config.fdr_decision,
    }
    empty_regions = pd.DataFrame(columns=_REGION_COLUMNS)
    empty_values = pd.DataFrame(columns=["region_id", *sample_names])
    if matrix.shape[0] < config.min_cpgs:
        manifest["status"] = "insufficient_cpgs"
        return empty_regions, empty_values, manifest

    residuals = centre_cpgs(matrix)
    basis, basis_diagnostics = estimate_latent_basis(residuals, config)
    manifest["latent_structure"] = basis_diagnostics
    deflated = deflate(residuals, basis, missing=config.deflation_missing)

    statistic = edge_statistic(deflated, config)
    eligible = eligible_edges(frame, config)
    # The threshold always comes from the locality-breaking null: it decides
    # which individual edges count as elevated, and that question is unaffected
    # by cohort size in the way the region-level question is.
    threshold, null_diagnostics = null_threshold(
        deflated, config, frame, statistic=statistic, eligible=eligible,
    )
    null_diagnostics["mode"] = config.null_mode

    if config.null_mode == "permutation":
        # Only the REGION-level calibration changes. The permuted track holds
        # the same edges above the same threshold, scattered at random, so the
        # null asks how often a run this long and this strong arises by chance
        # given how many elevated edges the genome actually contains. That
        # cannot saturate, because the null shares the data's marginal.
        tracks = permutation_null_tracks(
            statistic, eligible, frame["chr"].to_numpy(),
            replicates=config.null_replicates, seed=config.random_seed,
        )
        above = float(np.mean(statistic[eligible & np.isfinite(statistic)]
                              > threshold))
        null_diagnostics["fraction_edges_above_threshold"] = above
    else:
        tracks = null_tracks(deflated, config, frame)
    manifest["null"] = null_diagnostics

    regions = segment(
        statistic,
        eligible,
        threshold,
        min_cpgs=config.min_cpgs,
        max_gap_edges=config.max_gap_edges,
    )
    manifest["n_regions"] = len(regions)
    if regions.size == 0:
        manifest["status"] = "no_regions"
        return empty_regions, empty_values, manifest

    observed_areas = region_area(statistic, regions, threshold)
    null_areas = []
    null_lengths = []
    null_edge_total = 0
    for track, null_eligible in tracks:
        # the null is masked by the same kind of rule as the data, so that the
        # two are genuinely passed through the identical segmenter
        usable = null_eligible & np.isfinite(track)
        null_edge_total += int(usable.sum())
        null_regions = segment(
            track,
            usable,
            threshold,
            min_cpgs=config.min_cpgs,
            max_gap_edges=config.max_gap_edges,
        )
        null_areas.append(region_area(track, null_regions, threshold))
        null_lengths.append(
            (null_regions[:, 1] - null_regions[:, 0] + 1)
            if null_regions.size
            else np.empty(0, dtype=int)
        )
    observed_lengths = regions[:, 1] - regions[:, 0] + 1
    fdr_result = region_fdr_result(
        observed_areas,
        null_areas,
        observed_edges=int(np.sum(eligible & np.isfinite(statistic))),
        null_edges=max(null_edge_total, 1),
        observed_lengths=observed_lengths if config.length_stratified_fdr else None,
        null_length_sets=null_lengths if config.length_stratified_fdr else None,
    )
    if config.fdr_decision == "upper":
        # Spec §1.8 frames this guard as running "before computing regions";
        # it actually runs here, after region_fdr_result, because `scale` and
        # `m_bonferroni` (both required by the guard) aren't known until that
        # call completes. This is intentional: the guard still fires before
        # any region is reported to the user as significant, which is the
        # actual intent behind the spec's wording (never silently return zero
        # significant regions).
        check_fdr_decision_vacuity(
            fdr_result,
            scale=fdr_result.diagnostics.get("scale", 0.0),
            current_replicates=config.null_replicates,
            target_fdr=config.target_fdr,
        )
    qvalues = fdr_result.qvalue
    manifest["fdr"] = fdr_result.diagnostics
    manifest["null_regions_per_replicate"] = (
        float(np.mean([len(a) for a in null_areas])) if null_areas else float("nan")
    )

    posterior = np.full(len(regions), np.nan, dtype=float)
    posterior_extrapolated = np.zeros(len(regions), dtype=bool)
    if config.compute_posterior:
        from .probability import local_fdr

        estimate = local_fdr(
            observed_areas,
            null_areas,
            observed_edges=int(np.sum(eligible & np.isfinite(statistic))),
            null_edges=max(null_edge_total, 1),
            observed_lengths=(
                observed_lengths if config.length_stratified_fdr else None
            ),
            null_length_sets=(
                null_lengths if config.length_stratified_fdr else None
            ),
            min_null_regions=config.min_null_regions_for_posterior,
        )
        posterior = estimate.posterior
        posterior_extrapolated = estimate.extrapolated
        manifest["posterior"] = {
            "status": estimate.status,
            "pi0": estimate.pi0,
            **estimate.diagnostics,
        }

    decision_qvalue = (
        fdr_result.qvalue_upper if config.fdr_decision == "upper" else fdr_result.qvalue
    )

    rows = []
    value_rows = []
    for position, (first, last) in enumerate(regions):
        first, last = int(first), int(last)
        block = matrix[first : last + 1]
        counts = np.sum(np.isfinite(block), axis=0)
        means = np.full(n_samples, np.nan, dtype=float)
        np.divide(
            np.nansum(block, axis=0), counts, out=means, where=counts > 0
        )
        description = characterise_region(means)
        minority, minority_delta = region_minority_size(
            means, (first, last), matrix
        )
        chrom = frame["chr"].iloc[first]
        start = int(frame["start"].iloc[first])
        end = int(frame["end"].iloc[last])
        region_id = f"EIGER2_{chrom}_{start}_{end}"
        rows.append(
            {
                "region_id": region_id,
                "chr": str(chrom),
                "start": start,
                "end": end,
                "n_cpg": last - first + 1,
                "length_bp": max(0, end - start),
                "cpg_index_start": int(source_index[first]),
                "cpg_index_end": int(source_index[last]),
                "statistic": config.statistic,
                "statistic_mean": float(
                    np.nanmean(statistic[first:last]) if last > first else np.nan
                ),
                "statistic_min": float(
                    np.nanmin(statistic[first:last]) if last > first else np.nan
                ),
                "statistic_threshold": threshold,
                "region_area": float(observed_areas[position]),
                "region_qvalue": float(qvalues[position]),
                "region_tail_fdr": float(fdr_result.tail_fdr[position]),
                "region_qvalue_lower_95": float(fdr_result.qvalue_lower[position]),
                "region_qvalue_upper_95": float(fdr_result.qvalue_upper[position]),
                "region_qvalue_resolution": float(
                    fdr_result.qvalue_resolution[position]
                ),
                "null_exceedances_at_cutoff": int(
                    fdr_result.null_exceedances[position]
                ),
                "region_qvalue_cutoff_rank": int(
                    fdr_result.qvalue_cutoff_rank[position]
                ),
                "region_evidence_score": float(posterior[position]),
                "evidence_score_extrapolated": bool(posterior_extrapolated[position]),
                "minority_size_descriptive": int(minority),
                "minority_delta_descriptive": float(minority_delta),
                "passes_prevalence": bool(minority >= config.min_prevalence_samples),
                "significant": bool(
                    decision_qvalue[position] <= config.target_fdr
                    and minority >= config.min_prevalence_samples
                ),
                "n_samples_observed": description["n_observed"],
                "missing_rate": float(np.mean(~np.isfinite(block))),
                "mean_beta": description["mean_beta"],
                "delta_beta_descriptive": description["delta_beta"],
                "largest_gap_descriptive": description["largest_gap"],
                "direction_descriptive": description["direction"],
                "minor_group_size_descriptive": description["minor_group_size"],
                "minor_group_fraction_descriptive": description[
                    "minor_group_fraction"
                ],
                "minor_group_samples_descriptive": ";".join(
                    sample_names[i] for i in description["minor_group_samples"]
                ),
                "pvalue_descriptive": description["pvalue_descriptive"],
            }
        )
        value_rows.append(means)

    region_frame = pd.DataFrame(rows)[_REGION_COLUMNS]
    values_frame = pd.DataFrame(value_rows, columns=sample_names)
    values_frame.insert(0, "region_id", region_frame["region_id"].to_numpy())
    manifest["status"] = "ok"
    manifest["n_significant"] = int(region_frame["significant"].sum())
    manifest["n_passing_qvalue"] = int(
        (region_frame["region_qvalue"] <= config.target_fdr).sum()
    )
    manifest["n_passing_prevalence"] = int(region_frame["passes_prevalence"].sum())
    manifest["schema_version"] = "2.1"
    manifest["fdr_decision"] = config.fdr_decision
    return region_frame, values_frame, manifest


_REGION_COLUMNS = [
    "region_id",
    "chr",
    "start",
    "end",
    "n_cpg",
    "length_bp",
    "cpg_index_start",
    "cpg_index_end",
    "statistic",
    "statistic_mean",
    "statistic_min",
    "statistic_threshold",
    "region_area",
    "region_qvalue",
    "region_tail_fdr",
    "region_qvalue_lower_95",
    "region_qvalue_upper_95",
    "region_qvalue_resolution",
    "null_exceedances_at_cutoff",
    "region_qvalue_cutoff_rank",
    "region_evidence_score",
    "evidence_score_extrapolated",
    "minority_size_descriptive",
    "minority_delta_descriptive",
    "passes_prevalence",
    "significant",
    "n_samples_observed",
    "missing_rate",
    "mean_beta",
    "delta_beta_descriptive",
    "largest_gap_descriptive",
    "direction_descriptive",
    "minor_group_size_descriptive",
    "minor_group_fraction_descriptive",
    "minor_group_samples_descriptive",
    "pvalue_descriptive",
]
