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
import os
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


_DEFAULT_SUPPORT_DISPERSION_CUT = 0.5
# Measured across standard benchmark scenarios: with no missingness, support
# dispersion is structurally 0 (every eligible window has full support, so the
# 90th/10th percentile spread is zero); under 25% structured missingness it
# measured 1.04-1.13. 0.5 sits with comfortable margin on both sides of that
# gap and reliably separates the two regimes.


@dataclass(frozen=True)
class Eiger2Config:
    """Everything the caller can tune. Defaults are the validated settings."""

    # --- discovery statistic
    statistic: str = "correlation"
    """``"correlation"`` (scale-free, recommended) or ``"covariance"`` (carries
    effect-size information but is intrinsically prevalence-limited: for a
    two-group structure with minor fraction ``f`` and effect ``delta``, the
    adjacent covariance is ``f(1-f)delta**2`` and cannot see rare subgroups,
    whereas correlation, ``f(1-f)delta**2 / (f(1-f)delta**2 + sigma**2)``,
    saturates towards 1 for any ``f`` once ``delta`` exceeds the noise).
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

    A nuisance factor (purity, composition, batch) loads on the whole genome;
    a biological subgroup (cell type, tumour status, a rare clone) loads on
    marker regions. The fraction of a component's loading energy in the top
    1% of CpGs separates them cleanly: nuisance components score 0.08-0.09,
    biological components 0.15-0.40. Without this filter, blind deflation
    deletes exactly the signal EIGER exists to find -- on a cohort of 4 cell
    types plus tumour status plus a rare subgroup, parallel analysis selects
    K=5 and recovery collapses from 70/70 regions to 0/70. Set to 1.0 to
    disable the filter and deflate every component.
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
    """``"lag"`` is the only supported value for the edge-level null: it builds
    the null by pairing CpGs far apart on the same chromosome, testing *is
    there any regional coordination here?*

    ``"permutation"`` (edge-wise permutation of the observed statistic within
    each chromosome) is no longer available: it destroys the same run-length
    autocorrelation the region-level null depends on, measured at zero usable
    null regions. Requesting it raises immediately, naming the replacement.
    The large-cohort saturation problem it was written to fix (a null that
    becomes trivially uninformative once genuine coordination is estimated
    precisely across most of the genome) is addressed by ``region_null_mode``
    below instead.

    Sign-agnostic: a hypomethylated region produces the same positive adjacent
    correlation as a hypermethylated one. Direction is recovered afterwards
    from delta beta.
    """

    region_null_mode: str = "auto"
    """``"auto"`` | ``"member_step"`` | ``"lag"`` | ``"off"``. The REGION-level
    null construction, distinct from ``null_mode`` above (which is edge-level:
    which edges count as elevated). This axis decides which *runs* of elevated
    edges are significant.

    ``"auto"`` (default): use ``"member_step"`` (below) when the input clears
    its genome-scale minimum-size floor (``member_step_min_n_cpg(config)``);
    otherwise fall back to ``"lag"`` automatically, with a warning naming the
    input size, the required size, and that ``"lag"``'s q-values are not
    calibrated (``call_dmrs`` records ``manifest["gates"]
    ["region_null_mode_fallback"]``). An **explicitly** requested
    ``"member_step"`` (below) does NOT fall back: a caller who asked for it by
    name is told it cannot be done on this input, not silently given
    something else.

    ``"member_step"``: consecutive null "positions" step by one LAG (not one
    CpG), so they share ``window - 1`` of ``window`` members exactly as
    consecutive OBSERVED windows do -- reproducing the observed track's
    autocorrelation by construction, which ``"lag"`` (below) does not (see
    ``member_step_lag_null_ensemble``). Passes both region-level gates on real
    layouts at genome scale with recall identical to ``"lag"``; see
    ``docs/FDR.md``. Raises below its minimum-size floor rather than
    degrading silently -- use ``"auto"`` (default) unless you specifically
    want that raise.

    ``"lag"``: the region-level null built the same way as the edge-level one
    -- consecutive positions slide by one CpG rather than sharing members the
    way the observed track does. **Fails region-level exchangeability on real
    layouts** -- do not request it directly for region-level inference; kept
    for small inputs (``"auto"``'s fallback) and reproducibility. Still the
    edge-level null every ``null_mode`` uses; only the REGION-level use of it
    is not recommended.

    ``"off"``: no region null is built at all. ``region_qvalue`` and every
    inferential column are refused (``NaN`` / ``-1`` / ``significant=False``)
    via the same degenerate path ``region_fdr_result`` already takes for
    ``replicates == 0`` -- an explicit, cheap "candidates only, no region FDR"
    mode, not a new refusal mechanism.
    """

    n_lag_draws: int = 75
    """``region_null_mode="member_step"`` only: number of independent lag draws
    pooled into the null, each treated as one "replicate" -- directly
    comparable to ``null_replicates`` for the ``"lag"`` construction.

    Derived from a resolution target, not convenience: finite-null resolution
    is ``Delta = s / R`` where ``s = observed_eligible_edges /
    null_eligible_edges`` -- SMALLER ``s`` is better, since it means MORE null
    volume. Each lag draw contributes about one observed track's worth of null
    positions, so ``s ~ 1.3-1.45 / n_lag_draws``.

    A sensitivity sweep (``n_lag_draws`` in {40, 60, 75, 100, 150}, both real
    layouts, 3 seeds each, genome scale) measured `s = 0.0187-0.0195` at
    `n_lag_draws=75` -- comfortably resolving decisions at a 0.05 target -- with
    both region-level gates passing, an adequate pooled null, and recall/realised
    FDR unchanged from `150` (1.0 / 0.0 throughout) on every cell. `150` was
    chosen to match `s <= 0.01`, the resolution `null_replicates=100` was
    originally picked for; that target predates the corrected realised-FDR
    measurement (a 95% upper bound of 1.6% against the 5% nominal target, i.e.
    substantially conservative), which the extra resolution beyond `75` was not
    found to be buying. `75` cuts measured wall-clock at genome scale from
    ~219s to ~160s per cell (~27%, and closer to a 2x cut in the null
    construction's own share of that time) with no measured cost to any gate.
    Set `n_lag_draws=150` explicitly to restore the previous default if
    resolution finer than 0.02 is specifically wanted.
    """

    min_subtrack_length: int = 100
    """``region_null_mode="member_step"`` only: the shortest sub-track length a
    drawn lag is allowed to produce. Bounds the maximum lag drawable:
    ``L_max = n_cpg // (min_subtrack_length + window - 1)``; lags are drawn
    uniformly from ``[null_min_lag, L_max]``. Raises (naming ``n_cpg``,
    ``null_min_lag``, and the implied minimum input size) rather than silently
    degrading when ``L_max < null_min_lag`` -- a too-small floor produces
    short sub-tracks and an inadequate null silently. Use
    ``region_null_mode="lag"`` on inputs too small for ``"member_step"``, not
    a smaller ``min_subtrack_length``: a shorter floor reintroduces the
    segmentation edge effects it exists to prevent.
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

    Off by default, and measured to be counter-productive on the benchmark:
    at 25% structured missingness, raising it from 0.0 to 0.5 dropped
    recovery from 45/48 to 0/48 while *increasing* false regions from 17 to
    136 -- it removes DMR edges faster than artefact edges and fragments true
    regions below ``min_cpgs``. Exposed as a diagnostic lever for cohorts
    with a genuinely bimodal coverage profile; do not raise it without
    re-running the benchmark on your own missingness pattern.
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

    The primary control for targeted assays: on an EPIC-style panel most rows
    are absent or nearly so, and cost time at every stage while contributing
    nothing (a sparsely covered CpG cannot form a usable edge, needing
    ``min_paired_samples`` co-observed samples, but is still centred,
    deflated and scanned). Setting this to, say, 0.5 typically removes most
    rows on a panel and is close to free statistically, since those rows
    could not have contributed edges. Dropped CpGs never appear in the
    output; region coordinates are reported in the original input indexing,
    so filtering does not shift them. 0.0 (default) disables the filter.
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
    always read from the manifest's ``fdr_decision`` field.

    Depends on ``min_null_regions_for_fdr`` being an aggregate (not
    per-stratum) check: a stratum whose own null pool is empty still gets a
    finite ``K=0`` Garwood upper bound rather than being refused, which is
    what makes this mode usable rather than rejecting almost everything -- see
    ``docs/FDR.md``.
    """

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
    ``region_qvalue`` is a tail-area rate; ``region_evidence_score`` is the
    per-region quantity from :mod:`eiger2.probability`.

    **Named a score, not a probability, because measurement says it is not
    calibrated.** On planted-truth simulations the expected calibration error
    is 0.35-0.50: regions scoring 0.86 were real 0% of the time. Cause
    (``PROBABILITY.md``): null tracks come from signal-free data, but false
    positives in real data arise beside real signal and inherit its area
    distribution, so ``f0`` is the wrong null density for them. Still a
    useful monotone ranking; not a probability, must not be thresholded as
    one.

    Off by default. A density also needs far more null regions than a tail
    count does, so enabling this without raising ``null_replicates`` (to
    100-300) will usually return ``status = insufficient_null_regions`` and
    leave the columns NaN -- intended, not a failure.
    """

    min_null_regions_for_posterior: int = 50
    """Refuse the posterior below this many pooled null regions."""

    min_null_regions_for_fdr: int = 50
    """Refuse ``region_qvalue`` for the WHOLE run when the pooled null-region
    count, summed across every stratum and every replicate, falls below this
    floor. Below the floor, every inferential column (``region_qvalue`` and
    friends) becomes ``NaN``, every count column becomes ``-1``,
    ``significant`` is forced ``False``, and
    ``manifest["fdr"]["pooled_null_regions"]`` records the total against the
    floor.

    **This is an AGGREGATE, run-level guard, not a per-stratum one.** A
    stratum whose OWN pooled null count is thin, or even exactly zero, is not
    refused once the run as a whole is adequate -- it is let through the
    estimator like any other stratum, and the Garwood interval it already
    computes is what represents that thinness (a narrow pool gives a wide
    ``region_qvalue_upper_95``, not an outright ``NaN``). A stratum whose OWN
    pool is exactly empty gets ``K=0`` at every cutoff, which is
    arithmetically well-defined (point estimate ``q=0``, a finite Garwood
    upper bound) and is recorded, not refused, in
    ``manifest["fdr"]["zero_exceedance_strata"]`` -- the null having millions
    of opportunities and never producing a region that long is decisive
    evidence, not missing information; see ``docs/FDR.md`` and
    ``region_fdr_result``'s docstring for the arithmetic that justifies it.
    """

    support_binned_threshold: str = "auto"
    """``"auto"`` | ``"on"`` | ``"off"``. When enabled, the null-quantile
    threshold is computed per support bin (samples observed at every CpG in the
    window) instead of once genome-wide, correcting region-level FDR inflation
    under structured missingness. ``"auto"`` enables it only when observed
    support varies enough to matter (``support_dispersion >
    support_dispersion_cut``). See ``docs/FDR.md``, "Support conditioning"."""

    support_bins: int = 4
    """Quantile bins for support-conditioned thresholding. Adjacent bins merge
    upward if the pooled null has fewer than ``min_null_values`` values in a
    bin; collapses to the unconditioned single threshold if merging cannot
    satisfy every bin."""

    support_dispersion_cut: float | None = None
    """Auto-enable threshold for ``support_binned_threshold="auto"``. ``None``
    uses the calibrated default, ``_DEFAULT_SUPPORT_DISPERSION_CUT = 0.5``
    (see that constant's own comment for the measurement behind it)."""

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
        if self.min_null_regions_for_fdr < 0:
            raise ValueError("min_null_regions_for_fdr cannot be negative")
        if self.null_mode not in {"lag", "permutation"}:
            raise ValueError("null_mode must be 'lag' or 'permutation'")
        if self.null_mode == "permutation":
            raise ValueError(
                "null_mode='permutation' is broken and no longer supported: "
                "measured at zero null regions across 400 replicates "
                "(edge-wise permutation destroys the same autocorrelation "
                "the corrected 'lag' null exists to preserve). Use the "
                "default null_mode='lag' for the edge-level null; for the "
                "region-level null, see region_null_mode='member_step' (or "
                "leave it at its default 'auto', which resolves to "
                "'member_step' automatically on input large enough to "
                "support it)."
            )
        if self.region_null_mode not in {"auto", "member_step", "lag", "off"}:
            raise ValueError(
                "region_null_mode must be 'auto', 'member_step', 'lag', or 'off'"
            )
        if self.n_lag_draws < 1:
            raise ValueError("n_lag_draws must be positive")
        if self.min_subtrack_length < 1:
            raise ValueError("min_subtrack_length must be positive")
        if not 0 <= self.min_paired_fraction <= 1:
            raise ValueError("min_paired_fraction must lie in [0, 1]")
        if not 0 < self.target_fdr < 1:
            raise ValueError("target_fdr must lie in (0, 1)")
        if self.fdr_decision not in {"point", "upper"}:
            raise ValueError("fdr_decision must be 'point' or 'upper'")
        if self.support_binned_threshold not in {"auto", "on", "off"}:
            raise ValueError("support_binned_threshold must be 'auto', 'on', or 'off'")
        if self.support_bins < 2:
            raise ValueError("support_bins must be at least 2")

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

# Optional accelerated kernel (the `fast` extra, `pip install eiger[fast]`).
# numba is NOT a hard dependency: every path below works identically, if more
# slowly, without it. Import failure of any kind (not installed, incompatible
# version, no working C compiler for the JIT) falls back to the pure-numpy
# path silently -- this must never be the reason a user's analysis fails.
#
# Even when numba IS installed, this kernel is opt-in, not automatic: it
# computes the identical quantity via a different summation order (a scalar
# accumulation loop, not numpy's vectorised reduction), which differs from
# the numpy path at the ~1e-15 floating-point noise floor (see
# test_pair_statistic_numba_matches_numpy_nan_pattern_exactly for the
# measured bound). That is invisible at any statistical scale this method
# operates at, but it is enough to break exact-equality invariant fixtures
# elsewhere in the test suite, so a caller must ask for it explicitly --
# `EIGER2_USE_NUMBA=1` in the environment (the parallel paper-grid sweep this
# kernel exists for sets it in every worker) -- rather than every user with
# numba installed for an unrelated reason (e.g. the `legacy` extra) silently
# getting numerically different results from the same input.
try:
    from numba import njit, prange

    @njit(cache=True, parallel=True)
    def _pair_statistic_numba(
        left: np.ndarray,
        right: np.ndarray,
        want_correlation: bool,
        required: int,
    ) -> np.ndarray:
        # fastmath=True MUST NOT be added to this decorator. It was measured
        # to produce an apparent 60x speedup that was entirely fake:
        # fastmath permits the compiler to assume no NaNs and elides the
        # `l == l` finiteness check below, so the kernel silently returned
        # ALL-NaN. Every downstream gate happened to report "no data" rather
        # than a wrong number that time, but a different code path could as
        # easily return a plausible wrong answer instead. See
        # test_pair_statistic_numba_matches_numpy_nan_pattern_exactly, which
        # would catch a regression here.
        n_pairs, n_samples = left.shape
        out = np.empty(n_pairs, dtype=np.float64)
        for i in prange(n_pairs):
            count = 0
            sum_l = 0.0
            sum_r = 0.0
            sum_lr = 0.0
            sum_ll = 0.0
            sum_rr = 0.0
            for j in range(n_samples):
                lv = left[i, j]
                rv = right[i, j]
                if lv == lv and rv == rv:  # finite check; NaN != NaN
                    count += 1
                    sum_l += lv
                    sum_r += rv
                    sum_lr += lv * rv
                    if want_correlation:
                        sum_ll += lv * lv
                        sum_rr += rv * rv
            if count < 2 or count < required:
                out[i] = np.nan
                continue
            denom = count - 1
            cross = sum_lr - sum_l * sum_r / count
            covariance = cross / denom
            if want_correlation:
                var_l = (sum_ll - sum_l * sum_l / count) / denom
                var_r = (sum_rr - sum_r * sum_r / count) / denom
                scale = np.sqrt(var_l * var_r)
                out[i] = covariance / scale if scale > 0 else np.nan
            else:
                out[i] = covariance
        return out

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _pair_statistic_numpy(
    left: np.ndarray,
    right: np.ndarray,
    statistic: str,
    min_paired_samples: int,
    min_paired_fraction: float = 0.0,
) -> np.ndarray:
    """Pairwise-complete covariance or correlation, row by row -- the
    reference numpy implementation, always available and always correct.

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


def _pair_statistic(
    left: np.ndarray,
    right: np.ndarray,
    statistic: str,
    min_paired_samples: int,
    min_paired_fraction: float = 0.0,
) -> np.ndarray:
    """Pairwise-complete covariance or correlation, row by row.

    Uses the fused numba kernel above when the optional ``fast`` extra is
    installed AND ``EIGER2_USE_NUMBA=1`` is set in the environment
    (bit-for-bit NaN pattern, max abs difference <= 1e-12 against
    ``_pair_statistic_numpy`` -- see the acceptance test); otherwise defers
    to ``_pair_statistic_numpy`` directly. See the module-level comment above
    the kernel for why this is opt-in rather than automatic.
    """

    if not (_HAS_NUMBA and os.environ.get("EIGER2_USE_NUMBA") == "1"):
        return _pair_statistic_numpy(
            left, right, statistic, min_paired_samples, min_paired_fraction,
        )
    required = max(
        int(min_paired_samples),
        int(np.ceil(min_paired_fraction * left.shape[1])),
    )
    return _pair_statistic_numba(
        np.ascontiguousarray(left, dtype=np.float64),
        np.ascontiguousarray(right, dtype=np.float64),
        statistic != "covariance",
        required,
    )


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
    has at least ``min_null_values`` pooled null values.

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
    (rows ``i, i+lag, i+2*lag, ...`` instead of ``i, i+1, i+2, ...``) and the
    statistic is the mean over the complete pairwise set among *those*
    members: pair ``(a, b)`` is ``S_{b-a}[i + a*lag]`` where
    ``S_d[j] = stat(CpG j, CpG j + d*lag)``. Reusing ``_window_from_lags``
    here would be wrong: it assumes members sit at consecutive indices, which
    is true of the observed track and false of this one. See docs/FDR.md,
    "Null window assembly".

    Every sample moves together, so batch, purity, cell composition and every
    other cross-sample dependency survives intact -- that is what EIGER 1's
    per-sample resampling null destroyed, which is why its threshold came out
    several-fold too low and DMR-free matrices lit up. Consecutive null
    positions still slide by one CpG, so locally autocorrelated nuisance
    (runs of low coverage or high missingness) is preserved as well. Only
    adjacency, and therefore only regional coordination, is broken.
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
    chromosome: np.ndarray | None,
    lag: int,
    config: Eiger2Config,
    size: int,
    observed_eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Which null positions are admissible, by the same rule as the data.

    The observed track is masked by :func:`eligible_edges`, which forbids an
    edge spanning two chromosomes OR whose bp gap exceeds ``max_cpg_gap_bp``.
    The null must be masked the same way, or the two are not passed through
    the identical segmenter and the calibration argument does not hold. A
    null position ``i`` at lag ``L`` draws on rows ``i, i+L, ..., i+(w-1)L``;
    the chromosome check below requires all of them lie on one chromosome.

    ``observed_eligible`` is the SAME ``eligible_edges()`` array the observed
    track is held to, additionally requiring the LOCAL adjacent edge at EVERY
    member offset (not just the chromosome loop's ``w-1`` pairs) to itself be
    gap-eligible -- each null member must sit in as ordinary a local
    neighbourhood as any observed CpG does (16-50% of null positions violate
    this on realistic layouts if left unchecked). Deliberately NOT applied to
    the long lag ``L`` *between* members -- that gap is the intentional
    locality-breaking part of the construction. ``observed_eligible=None``
    (the default) skips the check and is a no-op on ``dense`` regardless.
    """

    if chromosome is None:
        eligible = np.ones(size, dtype=bool)
    else:
        span = max(config.window, 2)
        eligible = np.ones(size, dtype=bool)
        first = chromosome[:size]
        for step in range(1, span):
            offset = step * lag
            if offset + size > chromosome.size:
                return np.zeros(size, dtype=bool)
            eligible &= first == chromosome[offset : offset + size]

    if observed_eligible is not None:
        # `observed_eligible` is edge-indexed (n_cpg - 1 entries: entry k is the
        # edge AFTER CpG k), one shorter than the CpG-indexed `chromosome` array
        # the loop above bounds against. A member sitting at the very last CpG
        # has no following edge to check -- not a violation, just nothing to
        # check -- so pad one trailing `True` (vacuously eligible) rather than
        # bounds-failing the whole null position over a member that has no edge
        # to have failed.
        padded = np.concatenate([observed_eligible, [True]])
        w = max(config.window, 1)
        for a in range(w):
            offset = a * lag
            if offset + size > padded.size:
                return np.zeros(size, dtype=bool)
            eligible &= padded[offset : offset + size]
    return eligible


def null_tracks(
    residuals: np.ndarray,
    config: Eiger2Config,
    locations: pd.DataFrame | None = None,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Statistic tracks under the locality-breaking null, one per replicate.

    Returns ``(track, eligible, lag)`` triples. ``eligible`` restricts the null to
    positions whose window members (a) share a chromosome and (b) each sit at a
    locally gap-eligible position -- the same ``eligible_edges()`` rule the
    observed track is held to, applied to every member except across the
    intentional long lag itself. ``lag`` is that replicate's own draw, needed
    by the support-conditioned threshold to reconstruct the replicate's own
    window-member set (mirroring ``_lag_null_track``'s construction) when
    computing that replicate's own support. Passing ``locations=None``
    disables BOTH restrictions and is retained only for single-chromosome use
    and for tests.
    """

    n_rows = residuals.shape[0]
    span = max(config.window, 2)
    largest = max((n_rows - 2) // max(span - 1, 1), 2)
    low = min(config.null_min_lag, max(2, largest // 4))
    high = max(low + 1, largest)
    chromosome = (
        locations["chr"].to_numpy() if locations is not None else None
    )
    observed_eligible = (
        eligible_edges(locations, config) if locations is not None else None
    )
    rng = np.random.default_rng(config.random_seed + 977)
    tracks: list[tuple[np.ndarray, np.ndarray, int]] = []
    for _ in range(config.null_replicates):
        lag = int(rng.integers(low, high))
        track = _lag_null_track(residuals, lag, config)
        if not track.size:
            continue
        eligible = _null_eligibility(
            chromosome, lag, config, track.size,
            observed_eligible=observed_eligible,
        )
        if eligible.any():
            tracks.append((track, eligible, lag))
    return tracks


def member_step_lag_null_tracks(
    residuals: np.ndarray,
    lag: int,
    config: Eiger2Config,
    locations: pd.DataFrame | None = None,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """The member-step lag null: one sub-track per starting offset at a fixed lag.

    ``null_tracks()``'s construction steps its position index ``p`` by one
    CpG: position ``p``'s members are ``{p, p+lag, ..., p+(w-1)*lag}`` and
    position ``p+1``'s are ``{p+1, p+1+lag, ...}`` -- DISJOINT sets, unlike
    consecutive OBSERVED windows, which share ``w-1`` of ``w`` members. That
    mismatch is why that null's autocorrelation does not match the observed
    track's.

    This construction re-parameterises: fix a starting offset ``i`` and step a
    SECOND index ``j``, so the null "position" is ``p = i + j*lag`` and
    consecutive positions share ``w-1`` of ``w`` members by construction --
    reproducing the autocorrelation directly, not tuning toward it.

    Implementation: ``_lag_null_track(residuals, lag, config)`` already
    computes the per-``p`` value for every ``p``; re-slicing its output at
    stride ``lag`` from each offset ``i`` in ``[0, lag)`` recovers exactly the
    ``j``-indexed sub-track for that offset, with no new pair-assembly code.
    Eligibility likewise re-slices ``_null_eligibility``'s existing output.

    Returns one ``(sub_track, sub_eligible, offset)`` triple per offset ``i``
    with at least one eligible position. ``offset`` takes the third-element
    slot ``null_tracks()`` uses for ``lag``, so both are drop-in compatible
    with ``acf_gate``'s ``(track, eligible, *_rest)`` unpacking.
    """

    full = _lag_null_track(residuals, lag, config)
    if not full.size:
        return []
    chromosome = locations["chr"].to_numpy() if locations is not None else None
    observed_eligible = (
        eligible_edges(locations, config) if locations is not None else None
    )
    full_eligible = _null_eligibility(
        chromosome, lag, config, full.size, observed_eligible=observed_eligible,
    )
    sub_tracks: list[tuple[np.ndarray, np.ndarray, int]] = []
    for i in range(min(lag, full.size)):
        sub_track = full[i::lag]
        sub_eligible = full_eligible[i::lag]
        if sub_track.size and sub_eligible.any():
            sub_tracks.append((sub_track, sub_eligible, i))
    return sub_tracks


def member_step_min_n_cpg(config: Eiger2Config) -> int:
    """Minimum ``n_cpg`` for ``region_null_mode="member_step"`` to be buildable
    at all: the smallest input at which ``L_max = n_cpg //
    (min_subtrack_length + window - 1)`` still clears ``null_min_lag``. Shared
    by ``member_step_lag_null_ensemble``'s raise and ``call_dmrs``'s
    ``region_null_mode="auto"`` fallback check, so the two cannot drift apart.
    """

    span = config.min_subtrack_length + max(config.window, 1) - 1
    return config.null_min_lag * span


def member_step_lag_null_ensemble(
    residuals: np.ndarray,
    config: Eiger2Config,
    locations: pd.DataFrame | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], list[int]]:
    """The region-level null pool: ``config.n_lag_draws`` independent lag draws,
    one ``(track, eligible, lag)`` triple per draw -- directly interchangeable
    with ``null_tracks()``'s output everywhere it is consumed (``acf_gate``,
    ``null_exchangeability``, ``_overdispersion_diagnostic``), with each lag
    draw standing in for one "replicate" exactly as one ``"lag"``-construction
    replicate does.

    Each draw's members-share-``window - 1`` structure comes from
    :func:`member_step_lag_null_tracks`, one sub-track per starting offset
    ``i in [0, lag)``; concatenated into ONE array per lag, breaking the
    eligible mask at every sub-track boundary so ``segment()`` can never
    merge two unrelated offsets into one spurious region -- the same device
    :func:`eligible_edges` uses to stop a region crossing a chromosome
    boundary.

    **Lag range**: drawn uniformly from ``[null_min_lag, L_max]`` where
    ``L_max = n_cpg // (min_subtrack_length + window - 1)``. A narrow jitter
    band near ``null_min_lag`` correlates nearby draws (lag 1006 and 1008
    pair CpG ``i`` with ``i+1006`` and ``i+1008``, nearly the same
    construction) and was rejected for that reason. At genome scale ``L_max``
    is comfortably above ``null_min_lag``, so this wide range gives genuinely
    different constructions per draw. Raises when ``L_max < null_min_lag``
    rather than silently degrading to short sub-tracks.

    Returns ``(replicates, lags_used)`` -- ``lags_used`` is recorded in
    ``manifest["gates"]`` on every run.
    """

    n_rows = residuals.shape[0]
    w = max(config.window, 1)
    span = config.min_subtrack_length + w - 1
    l_max = n_rows // span
    if l_max < config.null_min_lag:
        minimum_n_cpg = member_step_min_n_cpg(config)
        raise ValueError(
            f"region_null_mode='member_step' needs at least one lag able to "
            f"produce sub-tracks of min_subtrack_length={config.min_subtrack_length} "
            f"at null_min_lag={config.null_min_lag}, i.e. n_cpg >= "
            f"null_min_lag * (min_subtrack_length + window - 1) = {minimum_n_cpg}; "
            f"got n_cpg={n_rows} (L_max={l_max}). Use region_null_mode='lag' on "
            f"input this small, or lower min_subtrack_length/null_min_lag "
            f"knowingly (min_subtrack_length trades off segmentation edge "
            f"effects; null_min_lag trades off the methylation correlation-"
            f"length margin)."
        )

    rng = np.random.default_rng(config.random_seed + 61_093)
    lags = [
        int(rng.integers(config.null_min_lag, l_max + 1))
        for _ in range(config.n_lag_draws)
    ]

    replicates: list[tuple[np.ndarray, np.ndarray, int]] = []
    for lag in lags:
        sub_tracks = member_step_lag_null_tracks(residuals, lag, config, locations)
        if not sub_tracks:
            continue
        track_pieces, eligible_pieces = [], []
        for track, sub_eligible, _offset in sub_tracks:
            track_pieces.append(track)
            boundary = sub_eligible.copy()
            if boundary.size:
                boundary[-1] = False  # never merge into the next offset's piece
            eligible_pieces.append(boundary)
        replicates.append((
            np.concatenate(track_pieces), np.concatenate(eligible_pieces), lag,
        ))
    return replicates, lags


def permutation_null_tracks(
    statistic: np.ndarray,
    eligible: np.ndarray,
    chromosome: np.ndarray | None = None,
    *,
    replicates: int = 100,
    seed: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Retired: edge-wise permutation (block length 1) of the observed statistic
    within each chromosome, measured at zero usable null regions across 400
    replicates. Always raises; kept only so old call sites get a clear error
    naming the replacement rather than an import failure."""

    raise ValueError(
        "permutation_null_tracks is edge-wise permutation (block length 1), "
        "measured at zero null regions across 400 replicates, and is no "
        "longer usable. For the region-level null, use "
        "region_null_mode='member_step' (or leave it at its default 'auto', "
        "which resolves to 'member_step' automatically on input large enough "
        "to support it); for the edge-level null, null_mode='lag' is the "
        "only supported value."
    )


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

    Splits the null replicates into a calibration half (sets the threshold)
    and a held-out half (measures exceedance against it). ``p_null`` is the
    held-out exceedance and is a control: by construction it must equal
    ``1 - config.null_quantile``, confirming the null is internally
    self-consistent. ``p_obs`` is the observed track's exceedance against the
    SAME threshold. The ratio ``p_obs / p_null`` is the actual gate: 1.0 when
    the null and data share the same upper-tail shape, rising when the
    observed track is more heavy-tailed than the null represents -- see
    docs/FDR.md, "Null exchangeability".

    The split is deterministic given ``config.random_seed`` (a fixed
    permutation of replicate indices, not re-derived from tracks' own order),
    so results are reproducible independent of how many replicates happen to
    survive filtering upstream.
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


def track_autocorrelation(
    values: np.ndarray, eligible: np.ndarray, lags: Sequence[int]
) -> dict[int, float]:
    """Lag-``k`` autocorrelation of ``values`` at every ``k`` in ``lags``.

    Pairs ``(values[i], values[i+k])`` are used where both positions are
    eligible and finite; the correlation is the Pearson coefficient over that
    pooled pair set. ``NaN`` for a lag with fewer than 2 usable pairs or a
    constant series (undefined correlation), not raised.
    """

    values = np.asarray(values, dtype=float)
    usable = np.asarray(eligible, dtype=bool) & np.isfinite(values)
    out: dict[int, float] = {}
    for k in lags:
        if k <= 0 or k >= values.size:
            out[k] = float("nan")
            continue
        paired = usable[:-k] & usable[k:]
        x = values[:-k][paired]
        y = values[k:][paired]
        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            out[k] = float("nan")
        else:
            out[k] = float(np.corrcoef(x, y)[0, 1])
    return out


def acf_gate(
    observed_statistic: np.ndarray,
    eligible: np.ndarray,
    null_tracks_: Sequence[tuple],
    window: int,
    *,
    tolerance: float = 0.05,
) -> dict:
    """Gate 1: null autocorrelation must match the observed track's within
    ``tolerance`` at lags 1..``window``, on DMR-free data. A sliding-window
    statistic manufactures autocorrelation by construction (windows at edges
    ``e`` and ``e+k`` share CpG pairs whenever ``|k| < w``), so validating a
    null's marginal and variance is not enough -- its dependence structure has
    to match too, or a run of elevated edges that is pure geometry looks like
    signal.

    ``null_tracks_`` pools every replicate's usable pairs at each lag BEFORE computing
    the correlation (not an average of per-replicate correlations), matching how the
    null is pooled everywhere else in this codebase. Accepts either
    ``null_tracks()``'s ``(track, eligible, lag)`` triples or
    ``permutation_null_tracks()``'s ``(track, eligible)`` pairs -- only the first two
    elements are used, mirroring the ``*_rest`` unpacking ``call_dmrs`` already uses
    for the same reason. Passes only when every lag has a finite, in-tolerance
    difference; a lag that could not be measured (too few usable pairs) fails the gate
    rather than being silently skipped, since an unmeasurable lag is not evidence of a
    match.
    """

    lags = list(range(1, max(int(window), 1) + 1))
    observed_acf = track_autocorrelation(observed_statistic, eligible, lags)

    null_acf: dict[int, float] = {}
    for k in lags:
        xs, ys = [], []
        for track, track_eligible, *_rest in null_tracks_:
            track = np.asarray(track, dtype=float)
            usable = np.asarray(track_eligible, dtype=bool) & np.isfinite(track)
            if k <= 0 or k >= track.size:
                continue
            paired = usable[:-k] & usable[k:]
            xs.append(track[:-k][paired])
            ys.append(track[k:][paired])
        x = np.concatenate(xs) if xs else np.empty(0)
        y = np.concatenate(ys) if ys else np.empty(0)
        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            null_acf[k] = float("nan")
        else:
            null_acf[k] = float(np.corrcoef(x, y)[0, 1])

    diff: dict[int, float] = {}
    for k in lags:
        o, n = observed_acf[k], null_acf[k]
        diff[k] = (n - o) if (np.isfinite(o) and np.isfinite(n)) else float("nan")

    passed = bool(lags) and all(
        np.isfinite(diff[k]) and abs(diff[k]) <= tolerance for k in lags
    )
    return {
        "lags": lags,
        "observed_acf": observed_acf,
        "null_acf": null_acf,
        "diff": diff,
        "tolerance": tolerance,
        "passed": passed,
    }


def region_exchangeability_gate(
    *,
    observed_regions: int,
    observed_eligible_edges: int,
    null_regions_total: int,
    null_eligible_edges: int,
    min_null_regions_for_fdr: int,
) -> dict:
    """Gate 2: region-level exchangeability, on DMR-free data.

    ``observed_rate``/``null_rate`` are candidate REGIONS (pre-FDR, i.e. segmenter
    output) per 10,000 eligible edges. ``region_exchangeability_ratio`` is their
    ratio; ``inf`` is reported EXPLICITLY (not coerced to ``NaN`` or dropped) when
    ``null_rate`` is zero and ``observed_rate`` is positive -- that must be visible,
    not hidden. ``NaN`` only when BOTH rates are zero (nothing happened on either
    side; the ratio is genuinely undefined, not infinite). ``null_count_adequate``
    records whether the pooled null-region count clears
    ``min_null_regions_for_fdr``, i.e. whether the ratio is "estimated from
    something" -- reported alongside the ratio, not a refusal gate like the FDR
    estimator's own guard; the ratio is still returned either way, for the same
    reason ``inf`` is reported rather than hidden.
    """

    observed_rate = (
        10_000 * observed_regions / observed_eligible_edges
        if observed_eligible_edges > 0 else float("nan")
    )
    null_rate = (
        10_000 * null_regions_total / null_eligible_edges
        if null_eligible_edges > 0 else float("nan")
    )
    if not (np.isfinite(observed_rate) and np.isfinite(null_rate)):
        ratio = float("nan")
    elif null_rate > 0:
        ratio = observed_rate / null_rate
    elif observed_rate > 0:
        ratio = float("inf")
    else:
        ratio = float("nan")

    return {
        "observed_regions": int(observed_regions),
        "observed_eligible_edges": int(observed_eligible_edges),
        "observed_rate": observed_rate,
        "null_regions_total": int(null_regions_total),
        "null_eligible_edges": int(null_eligible_edges),
        "null_rate": null_rate,
        "region_exchangeability_ratio": ratio,
        "null_count_adequate": null_regions_total >= min_null_regions_for_fdr,
        "min_null_regions_for_fdr": int(min_null_regions_for_fdr),
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
        np.concatenate([t[e] for t, e, _ in tracks]) if tracks else np.empty(0)
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
    offered = sum(int(t.size) for t, _, _lag in tracks) if tracks else 0
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
    shape (support-conditioned mode). The scalar path is unchanged and stays
    bit-identical.

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
    shape (support-conditioned mode); the scalar path is unchanged.

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


# Minimum pooled null-region count for a joint (length x support-bin) stratum
# to be estimated on its own. Below this the stratum falls back to
# length-only pooling. Not user-configurable.
_MIN_NULL_REGIONS_PER_JOINT_STRATUM = 50


def _overdispersion_diagnostic(
    per_replicate_null_areas: list[np.ndarray], cutoffs: np.ndarray
) -> tuple[str, float | None]:
    """Poisson-vs-observed dispersion check on pooled null exceedance counts.

    ``region_qvalue_upper_95`` assumes a Poisson count model (Garwood). If the
    replicate-to-replicate exceedance counts are overdispersed relative to Poisson,
    the Garwood interval understates the true uncertainty -- it becomes a LOWER
    bound on the real interval, not a correct one. This is reported, never
    silently corrected: no uncalibrated dispersion correction is applied.
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
    observed_support_bins: np.ndarray | None = None,
    null_support_bin_sets: Sequence[np.ndarray] | None = None,
    confidence_level: float = 0.95,
    min_null_regions_for_fdr: int = 50,
) -> RegionFdrResult:
    """Region-level empirical FDR: point estimate, resolution, and simultaneous
    Garwood Monte Carlo bounds, computed within length strata with a common
    Bonferroni ``m`` so ``qvalue_upper`` is comparable across strata. See
    ``docs/FDR.md`` for the full derivation and the three distinct quantities
    this keeps apart (point estimate, empirical resolution, Monte Carlo
    interval).

    For a cut-off ``a``, the expected number of null regions at least as
    extreme is the mean count across replicates, rescaled to the size of the
    real genome. The FDR is that expectation divided by the observed count.

    When ``observed_lengths``/``null_length_sets`` are supplied, the
    calculation is done **within length strata**: region area grows with
    length, so a pooled FDR silently makes short regions look worse than long
    ones. A 4-CpG DMR should compete against 4-CpG null regions, not 50-CpG
    ones.

    When ``observed_support_bins``/``null_support_bin_sets`` are ALSO
    supplied, the null pool is stratified jointly, by
    ``"<length_stratum>|sbin<b>"`` -- equalising the per-edge threshold across
    support bins does not equalise the rate regions FORM per bin. A joint
    stratum whose pooled null count is below
    ``_MIN_NULL_REGIONS_PER_JOINT_STRATUM`` falls back to the length-only
    pool, listed in ``diagnostics["support_stratification_fallback_strata"]``.

    ``min_null_regions_for_fdr`` guards null adequacy as a single AGGREGATE
    check summed across every stratum and replicate -- not a per-stratum one.
    A per-stratum floor conflates "merely small" (arithmetically well-defined,
    refusing it hides real evidence as certainty) with "genuinely, decisively
    empty" -- but with the run as a whole adequately backed by null volume,
    ``K=0`` at a stratum's own cutoff is the strongest evidence the estimator
    can produce (point estimate ``q=0``, a finite Garwood ``qvalue_upper_95``).
    Below the aggregate floor, every inferential field is ``NaN``, every count
    field is ``-1``. At or above it, every stratum is let through, including
    empty ones, named with their exposure in
    ``diagnostics["zero_exceedance_strata"]`` -- reported, not refused.
    ``diagnostics["insufficient_null_strata"]`` is kept for schema
    compatibility and is now always ``[]``.
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
            diagnostics={
                "pooled_fallback_strata": [], "insufficient_null_strata": [],
                "zero_exceedance_strata": [],
            },
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
            diagnostics={
                "status": status, "pooled_fallback_strata": [],
                "zero_exceedance_strata": [],
                "insufficient_null_strata": [],
            },
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
        observed_length_labels = np.array(
            [length_stratum(int(n)) for n in observed_lengths], dtype=object
        )
        null_length_labels_by_replicate = [
            np.array([length_stratum(int(n)) for n in lengths], dtype=object)
            for lengths in null_length_sets
        ]
    else:
        observed_length_labels = np.full(m, "_pooled", dtype=object)
        null_length_labels_by_replicate = [
            np.full(areas.size, "_pooled", dtype=object) for areas in null_area_sets
        ]

    # Spec §1c.6 escalation: joint (length x support-bin) null pooling, layered on
    # top of the length strata above. Never activates unless BOTH support arrays
    # are supplied AND length stratification is already on, so every pre-existing
    # caller (which passes neither) keeps bit-identical behaviour.
    support_stratified = (
        stratified
        and observed_support_bins is not None
        and null_support_bin_sets is not None
    )
    support_fallback_joint_strata: list[str] = []
    null_joint_labels_by_replicate = null_length_labels_by_replicate
    if support_stratified:
        observed_support_bins = np.asarray(observed_support_bins, dtype=int)
        null_support_bin_sets = [
            np.asarray(a, dtype=int) for a in null_support_bin_sets
        ]
        observed_joint_labels = np.array(
            [
                f"{length_stratum(int(n))}|sbin{int(b)}"
                for n, b in zip(observed_lengths, observed_support_bins, strict=True)
            ],
            dtype=object,
        )
        null_joint_labels_by_replicate = [
            np.array(
                [
                    f"{length_stratum(int(n))}|sbin{int(b)}"
                    for n, b in zip(lengths, bins, strict=True)
                ],
                dtype=object,
            )
            for lengths, bins in zip(
                null_length_sets, null_support_bin_sets, strict=True
            )
        ]
        joint_null_counts: dict[str, int] = {}
        for labels in null_joint_labels_by_replicate:
            if labels.size == 0:
                continue
            values, counts = np.unique(labels, return_counts=True)
            for value, count in zip(values, counts, strict=True):
                joint_null_counts[str(value)] = (
                    joint_null_counts.get(str(value), 0) + int(count)
                )

        observed_strata = np.empty(m, dtype=object)
        for i in range(m):
            joint_label = str(observed_joint_labels[i])
            if (
                joint_null_counts.get(joint_label, 0)
                >= _MIN_NULL_REGIONS_PER_JOINT_STRATUM
            ):
                observed_strata[i] = joint_label
            else:
                observed_strata[i] = observed_length_labels[i]
                support_fallback_joint_strata.append(joint_label)
    else:
        observed_strata = observed_length_labels

    strata = np.unique(observed_strata)

    # --- Pass A: distinct-cutoff count per stratum, then the common m_bonferroni
    cutoffs_by_stratum = {
        str(s): _n_distinct(observed_areas[observed_strata == s]) for s in strata
    }
    m_bonferroni = max(cutoffs_by_stratum.values())

    # The null-adequacy floor is a single AGGREGATE check on the run as a
    # whole, not a per-stratum one -- see this function's docstring. Run AFTER
    # stratification (and its validation, e.g. the strict support-bin zips
    # above) and Pass A, so a caller bug still raises regardless of null
    # adequacy and the diagnostics keys stay present either way; a stratum
    # whose own pool happens to be thin, or exactly empty, is not refused once
    # the run clears this bar -- handled in the per-stratum loop below
    # (``zero_exceedance_strata``), not here.
    pooled_null_regions = int(sum(a.size for a in null_area_sets))
    if pooled_null_regions < min_null_regions_for_fdr:
        nan_f = np.full(m, np.nan, dtype=float)
        neg_i = np.full(m, -1, dtype=np.int64)
        rate = pooled_null_regions / replicates if replicates else 0.0
        return RegionFdrResult(
            qvalue=nan_f, tail_fdr=nan_f, qvalue_lower=nan_f,
            qvalue_upper=nan_f, qvalue_resolution=nan_f,
            null_exceedances=neg_i, qvalue_cutoff_rank=neg_i,
            confidence_level=confidence_level,
            diagnostics={
                "status": "aggregate null inadequate",
                "pooled_null_regions": pooled_null_regions,
                "replicates": replicates,
                "null_regions_per_replicate": rate,
                "min_null_regions_for_fdr": int(min_null_regions_for_fdr),
                "replicates_required": (
                    math.ceil(min_null_regions_for_fdr / rate) if rate > 0 else None
                ),
                "scale": scale,
                "fdr_interval_bonferroni_m": m_bonferroni,
                "fdr_interval_cutoffs_by_stratum": cutoffs_by_stratum,
                "fdr_interval_family": "per_stratum",
                "pooled_fallback_strata": [],
                "insufficient_null_strata": [],
                "zero_exceedance_strata": [],
                "support_stratified": support_stratified,
                "support_stratification_fallback_strata": sorted(
                    set(support_fallback_joint_strata)
                ),
                "min_null_regions_per_joint_stratum": _MIN_NULL_REGIONS_PER_JOINT_STRATUM,
            },
        )

    # --- Pass B: run the estimator per stratum with that common m
    qvalue = np.empty(m, dtype=float)
    tail_fdr = np.empty(m, dtype=float)
    qvalue_lower = np.empty(m, dtype=float)
    qvalue_upper = np.empty(m, dtype=float)
    qvalue_resolution = np.empty(m, dtype=float)
    null_exceedances = np.empty(m, dtype=np.int64)
    qvalue_cutoff_rank = np.empty(m, dtype=np.int64)
    # DEPRECATED: always [] now. Kept for schema compatibility with older
    # manifest parsers; scheduled for removal at the next schema bump.
    pooled_fallback_strata: list[str] = []
    # DEPRECATED: the per-stratum refusal that used to populate this list was
    # retracted -- see this function's docstring. Always [] now, kept for
    # schema compatibility alongside `pooled_fallback_strata` above.
    insufficient_null_strata: list[dict] = []
    zero_exceedance_strata: list[dict] = []
    overdispersion_by_stratum: dict[str, dict] = {}

    for stratum in strata:
        rows = np.flatnonzero(observed_strata == stratum)
        # A joint label selects null regions by the joint key; a length-only
        # label (either because support stratification is off, or because this
        # stratum fell back) selects by length alone, i.e. the union over every
        # support bin of that length stratum.
        is_joint_stratum = support_stratified and "|sbin" in str(stratum)
        labels_by_replicate = (
            null_joint_labels_by_replicate
            if is_joint_stratum
            else null_length_labels_by_replicate
        )
        per_replicate = [
            null_area_sets[b][labels_by_replicate[b] == stratum]
            for b in range(replicates)
        ]
        pooled = np.concatenate(per_replicate) if per_replicate else np.empty(0)
        # A stratum whose own pool is completely empty is not refused -- the
        # run as a whole already cleared the aggregate floor above, so an
        # empty own pool means the null, given real opportunity, essentially
        # never produces a region this long: `K=0` at every cutoff,
        # arithmetically well-defined below (point estimate `q=0`, a finite
        # Garwood upper bound), not an absence of information. Recorded, not
        # refused, so a reader can see the evidence was real.
        if pooled.size == 0:
            zero_exceedance_strata.append({
                "stratum": str(stratum),
                "null_regions": 0,
                "replicates": replicates,
                "null_regions_per_replicate": 0.0,
            })

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
        "insufficient_null_strata": insufficient_null_strata,
        "zero_exceedance_strata": zero_exceedance_strata,
        "pooled_null_regions": pooled_null_regions,
        "null_count_dispersion_by_stratum": overdispersion_by_stratum,
        "null_count_overdispersion_warning": overdispersion_warning,
        "scale": scale,
        "support_stratified": support_stratified,
        "support_stratification_fallback_strata": sorted(
            set(support_fallback_joint_strata)
        ),
        "min_null_regions_per_joint_stratum": _MIN_NULL_REGIONS_PER_JOINT_STRATUM,
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
        "schema_version": "2.2",
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

    # Deferred import: eiger2.layouts imports FROM this module, so importing it
    # back here (module already fully initialised) avoids a circular import at
    # module load time.
    from .layouts import LENGTH_STRATA, describe_layout, realised_length_strata

    layout_diagnostics = describe_layout(frame, config.max_cpg_gap_bp)
    realised = realised_length_strata(layout_diagnostics["layout_max_detectable_region_cpgs"])
    layout_diagnostics["realised_length_strata"] = list(realised)
    layout_diagnostics["collapsed_length_strata"] = [
        s for s in LENGTH_STRATA if s not in realised
    ]
    manifest["layout"] = layout_diagnostics
    # The threshold always comes from the locality-breaking null: it decides
    # which individual edges count as elevated, and that question is unaffected
    # by cohort size in the way the region-level question is.
    threshold, null_diagnostics = null_threshold(
        deflated, config, frame, statistic=statistic, eligible=eligible,
    )
    null_diagnostics["mode"] = config.null_mode
    # null_mode="permutation" is rejected at Eiger2Config construction, so
    # this is the only edge-level null reachable here.
    tracks = null_tracks(deflated, config, frame)
    manifest["null"] = null_diagnostics

    # The REGION-level null, a separate axis from the edge-level `tracks`
    # above. `"lag"` reuses `tracks` directly; `"member_step"` builds an
    # independent ensemble reproducing the observed track's autocorrelation;
    # `"off"` uses `region_fdr_result`'s existing degenerate (NaN/-1) path for
    # `replicates == 0`. `"auto"` (the default) resolves to `"member_step"`
    # when the input clears its minimum-size floor, else falls back to
    # `"lag"` and records why in `region_null_mode_fallback` below (`cli.py`
    # turns that into a stderr warning; this function does no I/O). An
    # EXPLICIT `"member_step"` request does not fall back and still raises
    # below the floor. Every branch from here on keys on
    # `resolved_region_null_mode`, not `config.region_null_mode`.
    region_null_mode_fallback: dict | None = None
    resolved_region_null_mode = config.region_null_mode
    if config.region_null_mode == "auto":
        n_cpg_here = deflated.shape[0]
        min_n_cpg = member_step_min_n_cpg(config)
        if n_cpg_here < min_n_cpg:
            resolved_region_null_mode = "lag"
            region_null_mode_fallback = {
                "requested": "auto",
                "used": "lag",
                "reason": "input below member_step minimum",
                "n_cpg": int(n_cpg_here),
                "n_cpg_required": int(min_n_cpg),
            }
        else:
            resolved_region_null_mode = "member_step"

    if resolved_region_null_mode == "lag":
        region_tracks = tracks
        region_lags = [lag for _, _, lag in tracks]
    elif resolved_region_null_mode == "off":
        region_tracks = []
        region_lags = []
    else:
        region_tracks, region_lags = member_step_lag_null_ensemble(
            deflated, config, frame,
        )
    manifest["gates"] = {
        "region_null_mode": resolved_region_null_mode,
        "n_lag_draws": len(region_lags),
        "lags_used": region_lags,
    }
    if region_null_mode_fallback is not None:
        manifest["gates"]["region_null_mode_fallback"] = region_null_mode_fallback
    if resolved_region_null_mode == "member_step" and region_lags:
        w_for_subtrack = max(config.window, 1)
        n_for_subtrack = deflated.shape[0]
        manifest["gates"]["implied_subtrack_length_range"] = [
            int(n_for_subtrack / lag - w_for_subtrack + 1)
            for lag in (max(region_lags), min(region_lags))
        ]

    # Gate 1 (ACF match): computed unconditionally, here, because it needs
    # only the observed track / eligibility / region null already built above
    # -- unlike Gate 2 (region exchangeability, below), it does not depend on
    # any region ever being called, so it is available even on the
    # "no_regions" early-return path. Informative regardless of whether real
    # signal is present; only a red flag on genuinely DMR-free data. Uses
    # `region_tracks`, not the edge-level `tracks` -- Gate 1 validates
    # whichever null actually forms regions.
    manifest["gates"]["acf"] = acf_gate(
        statistic, eligible, region_tracks, config.window,
    )

    # --- support-conditioned threshold ---
    n_cpg_total = deflated.shape[0]
    w = max(config.window, 1)
    finite_mask = np.isfinite(deflated)

    def _member_offsets(window: int, lag: int = 1) -> np.ndarray:
        """Window-member offsets from a base position: the adjacent pair at
        w=1 (offsets [0, lag]), else [0, lag, 2*lag, ..., (w-1)*lag].
        `lag=1` gives the observed-track convention; any other `lag` gives a null
        position's member set, matching `_lag_null_track`'s own construction."""
        if window == 1:
            return np.array([0, lag])
        return np.arange(window) * lag

    obs_offsets = _member_offsets(w)
    obs_offset0 = 0 if w == 1 else w // 2 - 1
    obs_base = np.arange(statistic.size) - obs_offset0
    valid_obs = (obs_base >= 0) & (obs_base + obs_offsets.max() < n_cpg_total)
    observed_support = np.zeros(statistic.size, dtype=np.int64)
    observed_support[valid_obs] = _window_support(
        finite_mask, obs_base[valid_obs], obs_offsets,
    )

    support_dispersion = float("nan")
    usable_mask = eligible & np.isfinite(statistic)
    if np.any(usable_mask):
        usable_support = observed_support[usable_mask]
        median = np.median(usable_support)
        if median > 0:
            support_dispersion = float(
                (np.quantile(usable_support, 0.9) - np.quantile(usable_support, 0.1))
                / median
            )

    dispersion_cut = (
        config.support_dispersion_cut if config.support_dispersion_cut is not None
        else _DEFAULT_SUPPORT_DISPERSION_CUT
    )

    if config.support_binned_threshold == "off":
        support_binning = "forced_off"
    elif config.support_binned_threshold == "on":
        support_binning = "forced_on"
    elif np.isfinite(support_dispersion) and support_dispersion > dispersion_cut:
        support_binning = "auto_on"
    else:
        support_binning = "auto_off"

    support_binning_skipped_reason: str | None = None
    if resolved_region_null_mode != "lag" and support_binning != "forced_off":
        # Support-conditioned thresholding's per-replicate reconstruction
        # assumes each null track's per-position member set follows
        # _lag_null_track's single-lag formula on a CONTIGUOUS track -- true
        # of "lag"'s tracks, false of "member_step"'s (several offset
        # sub-tracks concatenated, so the formula misdescribes positions near
        # a boundary) and vacuous for "off". Not extended to either -- a
        # known, documented limitation, not a silent miscalibration -- so
        # both fall back to the plain scalar threshold (see
        # `region_track_thresholds` below).
        support_binning_skipped_reason = "region_null_mode_not_lag"
        support_binning = "forced_off"

    support_bin_edges: np.ndarray = np.empty(0)
    support_bin_merges: list = []
    support_thresholds: list = []
    # `null_full_support`/`pooled_null_support` are only ever assigned inside the
    # `if binning_requested and ...:` block below, but diagnostic code later
    # references `null_full_support` unconditionally under a
    # `support_binning in ("auto_on", "forced_on")` gate that is today always
    # equivalent to "that block ran". There's no structural guarantee of that
    # equivalence holding forever, so initialize defensively here to turn a
    # future NameError into an empty (inert) list instead.
    null_full_support: list[np.ndarray] = []
    per_edge_threshold: float | np.ndarray = threshold
    # Per-replicate threshold arrays fed to segment()/region_area() for each
    # null track, aligned 1:1 with `tracks`. Defaults to the scalar threshold;
    # replaced below when binning succeeds. Each null track's threshold must
    # come from THAT replicate's own support
    # geometry (own lag), never from slicing the observed per_edge_threshold
    # array, since null tracks are not the same length as the observed track
    # and do not share its position-to-bin mapping.
    null_edge_thresholds: list[float | np.ndarray] = [threshold] * len(tracks)

    binning_requested = support_binning in ("forced_on", "auto_on")
    if binning_requested:
        # Per null replicate, compute that replicate's OWN support (using its own
        # lag, mirroring _lag_null_track's member set exactly) and pool this
        # replicate's STATISTIC VALUES -- not its support values -- into the bin
        # each position's support falls into. threshold[b] is then the
        # null_quantile of all pooled statistic values across every replicate
        # whose position landed in bin b.
        null_support_parts = []
        # (support_array, value_array) per replicate, usable-filtered
        null_value_by_replicate = []
        null_full_support = []  # full-length (track.size) support array per replicate
        for track, track_eligible, lag in tracks:
            null_offsets = _member_offsets(w, lag)
            null_base = np.arange(track.size)
            valid_null = null_base + null_offsets.max() < n_cpg_total
            support = np.zeros(track.size, dtype=np.int64)
            support[valid_null] = _window_support(
                finite_mask, null_base[valid_null], null_offsets,
            )
            null_full_support.append(support)
            usable = track_eligible & np.isfinite(track) & valid_null
            null_support_parts.append(support[usable])
            null_value_by_replicate.append((support[usable], track[usable]))

        pooled_null_support = (
            np.concatenate(null_support_parts) if null_support_parts else np.empty(0)
        )
        bin_result = _support_bin_edges(
            observed_support[usable_mask], config.support_bins,
            config.min_null_values, pooled_null_support,
        )
        if bin_result["collapsed"]:
            support_binning = "collapsed_to_single"
        else:
            support_bin_edges = bin_result["bin_edges"]
            support_bin_merges = bin_result["merges"]
            n_final_bins = support_bin_edges.size + 1
            observed_bins = _assign_support_bin(observed_support, support_bin_edges)

            per_bin_threshold = np.full(n_final_bins, np.nan)
            for b in range(n_final_bins):
                bin_values = np.concatenate([
                    values[_assign_support_bin(support, support_bin_edges) == b]
                    for support, values in null_value_by_replicate
                ]) if null_value_by_replicate else np.empty(0)
                if bin_values.size:
                    per_bin_threshold[b] = float(
                        np.quantile(bin_values, config.null_quantile)
                    )
            # a bin with zero pooled null values (shouldn't happen after merging
            # satisfied min_null_values, but guard defensively) falls back to the
            # unconditioned scalar threshold for that bin only
            per_bin_threshold = np.where(
                np.isfinite(per_bin_threshold), per_bin_threshold, threshold,
            )
            per_edge_threshold = per_bin_threshold[observed_bins]
            support_thresholds = per_bin_threshold.tolist()
            # each null track's own threshold array, from its OWN support's bin
            # assignment -- never the observed track's per_edge_threshold
            null_edge_thresholds = [
                per_bin_threshold[_assign_support_bin(support, support_bin_edges)]
                for support in null_full_support
            ]

    manifest["support_dispersion"] = support_dispersion
    manifest["support_dispersion_cut"] = dispersion_cut
    manifest["support_binning"] = support_binning
    manifest["support_bin_edges"] = support_bin_edges.tolist()
    manifest["support_bin_merges"] = support_bin_merges
    manifest["support_thresholds"] = support_thresholds
    if support_binning_skipped_reason is not None:
        manifest["support_binning_skipped_reason"] = support_binning_skipped_reason

    region_threshold_mode_value = (
        "support_binned" if support_binning in ("auto_on", "forced_on") else "single"
    )

    # `null_edge_thresholds` is 1:1 with `tracks` (the edge null); when
    # `region_null_mode="lag"`, `region_tracks IS tracks`, so it applies
    # directly. Otherwise support-conditioning is forced off above (this
    # `region_threshold_mode_value` will read "single"), so every region-null
    # entry gets the same plain scalar `threshold` the observed track itself
    # gets in that case -- observed and null stay on the identical segmenter.
    region_track_thresholds: list[float | np.ndarray] = (
        null_edge_thresholds if resolved_region_null_mode == "lag"
        else [threshold] * len(region_tracks)
    )

    regions = segment(
        statistic,
        eligible,
        per_edge_threshold,
        min_cpgs=config.min_cpgs,
        max_gap_edges=config.max_gap_edges,
    )
    manifest["n_regions"] = len(regions)
    if regions.size == 0:
        manifest["status"] = "no_regions"
        return empty_regions, empty_values, manifest

    # Computed once, in one place, so both the row-building loop below and the
    # region-level exposure diagnostic can read it by position rather than
    # recomputing it.
    manifest["region_support_median_descriptive"] = [
        float(np.median(observed_support[first:last])) for first, last in regions
    ]

    observed_areas = region_area(statistic, regions, per_edge_threshold)
    null_areas = []
    null_lengths = []
    null_edge_total = 0
    # Reused (rather than re-segmenting) for `region_rate_by_support_bin`:
    # each entry is the `segment()` output for that replicate, already
    # computed against that replicate's OWN per-bin threshold array
    # (`null_track_threshold` below), exactly mirroring the observed-region
    # convention. Populated for every replicate regardless of whether support
    # binning is active, since it's cheap to keep and the gating on
    # `support_binning in ("auto_on", "forced_on")` happens once, below,
    # before it's used.
    null_regions_by_replicate: list[np.ndarray] = []
    for (track, null_eligible, *_rest), null_track_threshold in zip(
        region_tracks, region_track_thresholds, strict=True
    ):
        # the null is masked by the same kind of rule as the data, so that the
        # two are genuinely passed through the identical segmenter. Each track's
        # threshold comes from ITS OWN support-conditioned bin assignment (see
        # `null_edge_thresholds` above), exactly mirroring the existing scalar
        # flow -- previously every null shared the single scalar `threshold`;
        # now every null shares the same per-bin threshold VALUES but each
        # applies them through its own support geometry.
        usable = null_eligible & np.isfinite(track)
        null_edge_total += int(usable.sum())
        null_regions = segment(
            track,
            usable,
            null_track_threshold,
            min_cpgs=config.min_cpgs,
            max_gap_edges=config.max_gap_edges,
        )
        null_areas.append(region_area(track, null_regions, null_track_threshold))
        null_lengths.append(
            (null_regions[:, 1] - null_regions[:, 0] + 1)
            if null_regions.size
            else np.empty(0, dtype=int)
        )
        null_regions_by_replicate.append(null_regions)
    observed_lengths = regions[:, 1] - regions[:, 0] + 1

    # Gate 2 (region-level exchangeability). Unlike Gate 1 above, this needs
    # `null_regions_by_replicate` -- the null tracks actually segmented into
    # candidate regions -- which is only built on this (regions.size > 0)
    # path, matching the region_rate_by_support_bin diagnostic's own scope
    # below: both are silent on the "no_regions" early return, same as
    # `manifest["fdr"]`.
    manifest["gates"]["region_exchangeability"] = region_exchangeability_gate(
        observed_regions=len(regions),
        observed_eligible_edges=int(np.sum(eligible & np.isfinite(statistic))),
        null_regions_total=int(sum(len(r) for r in null_regions_by_replicate)),
        null_eligible_edges=null_edge_total,
        min_null_regions_for_fdr=config.min_null_regions_for_fdr,
    )
    # `s = observed_eligible_edges / null_eligible_edges` is
    # `region_fdr_result`'s finite-null resolution scale
    # (`region_qvalue_resolution = s / R`) -- SMALLER is better (more null
    # volume). Recorded explicitly, the right way round, so a user can see
    # what their q-values are quantised at without re-deriving it from
    # `region_exchangeability`'s edge counts.
    manifest["gates"]["s"] = (
        int(np.sum(eligible & np.isfinite(statistic))) / max(null_edge_total, 1)
    )
    manifest["gates"]["implied_region_qvalue_resolution_at_R1"] = manifest["gates"]["s"]

    # --- region-level exposure diagnostic ---
    # Compares the observed rate of CALLED REGIONS per support bin against the
    # null's rate (both per 10,000 eligible edges): equalising the per-edge
    # threshold does not guarantee regions FORM at the same rate per bin --
    # clustering could still be support-driven even with fair edges. Null
    # regions reuse `null_regions_by_replicate` (already segmented against
    # each replicate's own per-bin threshold) rather than re-segmenting, so
    # this cannot drift from how the null was actually segmented; each null
    # region's support-median comes from that replicate's own full-length
    # support array. Eligible-edge counts per bin reuse
    # `observed_support[usable_mask]`/`pooled_null_support` from the binning
    # setup above. These same per-region bin assignments also stratify the
    # region-FDR null pool below; default None is length-only pooling, which
    # is what every run with support binning off gets.
    observed_support_bins_for_fdr = None
    null_support_bin_sets_for_fdr = None
    if support_binning in ("auto_on", "forced_on"):
        n_final_bins = support_bin_edges.size + 1
        region_bins = _assign_support_bin(
            np.asarray(manifest["region_support_median_descriptive"], dtype=float),
            support_bin_edges,
        )
        observed_regions_by_bin = np.bincount(region_bins, minlength=n_final_bins)

        null_bin_hits = []
        for null_regions, support in zip(
            null_regions_by_replicate, null_full_support, strict=True
        ):
            for first, last in null_regions:
                support_median = float(np.median(support[first:last]))
                null_bin_hits.append(support_median)
        if null_bin_hits:
            null_region_bins = _assign_support_bin(
                np.asarray(null_bin_hits, dtype=float), support_bin_edges
            )
            null_regions_by_bin = np.bincount(
                null_region_bins, minlength=n_final_bins
            )
        else:
            null_regions_by_bin = np.zeros(n_final_bins, dtype=int)

        observed_edges_by_bin = np.bincount(
            _assign_support_bin(observed_support[usable_mask], support_bin_edges),
            minlength=n_final_bins,
        )
        null_edges_by_bin = np.bincount(
            _assign_support_bin(pooled_null_support, support_bin_edges),
            minlength=n_final_bins,
        )

        region_rate_by_support_bin: dict = {}
        for b in range(n_final_bins):
            obs_edges = int(observed_edges_by_bin[b])
            n_null_edges = int(null_edges_by_bin[b])
            obs_regions = int(observed_regions_by_bin[b])
            n_null_regions = int(null_regions_by_bin[b])
            observed_rate = (
                10_000 * obs_regions / obs_edges if obs_edges > 0 else float("nan")
            )
            null_rate = (
                10_000 * n_null_regions / n_null_edges
                if n_null_edges > 0
                else float("nan")
            )
            entry = {
                "observed_regions": obs_regions,
                "observed_eligible_edges": obs_edges,
                "observed_rate": observed_rate,
                "null_regions": n_null_regions,
                "null_eligible_edges": n_null_edges,
                "null_rate": null_rate,
            }
            if n_null_edges == 0:
                entry["ratio"] = None
                entry["note"] = (
                    "null had zero eligible edges in this bin; ratio undefined"
                )
            elif null_rate == 0.0:
                entry["ratio"] = None
                entry["note"] = (
                    "null had zero regions in this bin; ratio undefined"
                    if observed_rate > 0
                    else "both observed and null had zero regions in this bin"
                )
            else:
                entry["ratio"] = observed_rate / null_rate
            region_rate_by_support_bin[int(b)] = entry
        manifest["region_rate_by_support_bin"] = region_rate_by_support_bin

        # Joint (length x support-bin) FDR stratification is defined ON TOP of
        # the length strata, so it only activates when those are on too. Each
        # null region's bin comes from its OWN replicate's full-length support
        # array via the same median-then-_assign_support_bin recipe used for the
        # observed regions above, so observed and null are binned identically.
        if config.length_stratified_fdr:
            observed_support_bins_for_fdr = region_bins
            null_region_support_medians_by_replicate = [
                (
                    np.array(
                        [
                            float(np.median(support[first:last]))
                            for first, last in null_regions
                        ],
                        dtype=float,
                    )
                    if null_regions.size
                    else np.empty(0, dtype=float)
                )
                for null_regions, support in zip(
                    null_regions_by_replicate, null_full_support, strict=True
                )
            ]
            null_support_bin_sets_for_fdr = [
                (
                    _assign_support_bin(medians, support_bin_edges)
                    if medians.size
                    else np.empty(0, dtype=int)
                )
                for medians in null_region_support_medians_by_replicate
            ]
    fdr_result = region_fdr_result(
        observed_areas,
        null_areas,
        observed_edges=int(np.sum(eligible & np.isfinite(statistic))),
        null_edges=max(null_edge_total, 1),
        observed_lengths=observed_lengths if config.length_stratified_fdr else None,
        null_length_sets=null_lengths if config.length_stratified_fdr else None,
        observed_support_bins=observed_support_bins_for_fdr,
        null_support_bin_sets=null_support_bin_sets_for_fdr,
        min_null_regions_for_fdr=config.min_null_regions_for_fdr,
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
        # `local_fdr` above is only ever given `observed_lengths`/`null_length_sets`
        # (never support-bin arguments), so this posterior path is length-stratified
        # at most, never support-stratified -- unlike `region_fdr_result` above.
        # Recorded here so a consumer of `manifest["posterior"]` doesn't assume
        # parity with the region-level q-values.
        manifest["posterior"]["support_stratified"] = False

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
                "statistic_threshold": (
                    float(np.mean(per_edge_threshold[first:last]))
                    if region_threshold_mode_value == "support_binned"
                    else threshold
                ),
                "region_support_median": manifest[
                    "region_support_median_descriptive"
                ][position],
                "region_threshold_mode": region_threshold_mode_value,
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
    manifest["schema_version"] = "2.2"
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
    "region_support_median",
    "region_threshold_mode",
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
