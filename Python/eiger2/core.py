"""Scan primitives behind :mod:`eiger2.coordination` and :mod:`eiger2.axis`.

    centre_cpgs           remove the per-CpG mean; NaN stays NaN
    edge_statistic        pairwise-complete co-variation of every adjacent CpG
                          pair, or the mean over a w-CpG window
    null_threshold        a quantile of the same statistic computed on windows
                          whose members sit a distant lag apart on the same
                          chromosome (the candidate threshold)
    eligible_edges        same chromosome and a bp gap within the limit
    segment, region_area  runs of passing edges and their summed excess
    latent_axes           genome-wide sample axes from a thinned CpG set, used
                          only by the secondary axis annotations

The region-level error rate is not computed here: it comes from complete
permutation rescans in :func:`eiger2.coordination.call_coordinated_regions`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

__all__ = [
    "CANONICAL_CHROMOSOMES",
    "Eiger2Config",
    "LatentAxes",
    "canonical_chromosome_order",
    "centre_cpgs",
    "chromosome_lag_feasibility",
    "chromosome_sort_permutation",
    "edge_statistic",
    "eligible_edges",
    "latent_axes",
    "null_threshold",
    "null_tracks",
    "region_area",
    "segment",
]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

#: Components whose loading concentration lies within this distance of the
#: boundary are listed in the latent-axis diagnostics (a note, not a gate).
DEFLATION_NEAR_CUTOFF_MARGIN = 0.02


@dataclass(frozen=True)
class Eiger2Config:
    """Settings read by the scan and latent-axis functions in this module.

    Internal: :class:`eiger2.coordination.CallerProfile` and
    :func:`eiger2.axis.annotate_axes` build it. The defaults of the fields they
    do not set are the reviewed values.
    """

    # --- discovery statistic
    statistic: str = "correlation"
    """``"correlation"`` or ``"covariance"``, pairwise-complete."""

    window: int = 4
    """CpGs per co-variation window; 1 means adjacent pairs."""

    # --- latent axes (used by the secondary axis annotations)
    deflate_components: int | None = None
    """Number of sample-space components; ``None`` = parallel analysis."""

    deflation_thin: int = 50
    """Every Nth CpG row is used to estimate the axes, so a focal region of a
    few dozen CpGs contributes at most one row."""

    deflation_max_concentration: float = 0.12
    """Loading-concentration boundary between a diffuse (nuisance) and a
    concentrated (biological) component under the ``"absolute"`` rule."""

    deflation_rule: str = "absolute"
    """``"absolute"`` (the constant above) or ``"relative"`` (a quantile of the
    concentration of permutation-null components)."""

    deflation_null_quantile: float = 0.95
    """Quantile used by the ``"relative"`` rule."""

    deflation_concentration_quantile: float = 0.01
    """Top fraction of CpGs whose loading energy defines the concentration."""

    deflation_max_components: int = 20
    """Upper limit on the number of components considered."""

    deflation_min_rows: int = 20
    """Fewer complete thinned rows than this triggers the mean-filled relaxation
    path, or, where relaxation is refused, the ``insufficient_rows_no_relaxation``
    label (the SVD then still runs on the complete rows, if at least 3)."""

    deflation_min_observed_fraction: float = 0.80
    """Observed fraction a thinned row needs on the relaxation path."""

    deflation_floor_observed_fraction: float = 0.50
    """Lowest observed fraction the relaxation path may fall back to."""

    parallel_analysis_replicates: int = 5
    """Column-permutation replicates behind parallel analysis."""

    parallel_analysis_quantile: float = 0.99
    """Singular-value quantile a component must exceed to be retained."""

    # --- distant-lag candidate threshold
    null_replicates: int = 100
    """Distant-lag draws pooled into the threshold (the profiles set 8)."""

    null_quantile: float = 0.995
    """Quantile of the pooled distant-lag statistic used as the threshold."""

    null_min_lag: int = 1000
    """Smallest lag, in CpG rows, between the members of a null window."""

    min_null_values: int = 1_000
    """Below this many pooled null values the threshold carries a warning."""

    min_null_coverage: float = 0.25
    """Below this retained fraction of null positions, likewise."""

    # --- pairing and geometry
    min_paired_samples: int = 5
    """Fewest samples observed at both CpGs for a pair statistic."""

    min_paired_fraction: float = 0.0
    """Fewest paired samples as a fraction of all samples."""

    max_cpg_gap_bp: int | None = 300
    """Largest ``next.start - previous.end`` gap inside a region."""

    random_seed: int = 1
    """Seed of the distant-lag draws and of parallel analysis."""

    def __post_init__(self) -> None:
        if self.statistic not in {"correlation", "covariance"}:
            raise ValueError("statistic must be 'correlation' or 'covariance'")
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.deflate_components is not None and self.deflate_components < 0:
            raise ValueError("deflate_components cannot be negative")
        if self.deflation_thin < 1:
            raise ValueError("deflation_thin must be at least 1")
        if self.deflation_rule not in {"absolute", "relative"}:
            raise ValueError("deflation_rule must be 'absolute' or 'relative'")
        if not 0.0 < self.deflation_null_quantile <= 1.0:
            raise ValueError("deflation_null_quantile must lie in (0, 1]")
        if not 0.5 < self.null_quantile < 1.0:
            raise ValueError("null_quantile must lie in (0.5, 1)")
        if self.null_replicates < 1:
            raise ValueError("null_replicates must be positive")
        if self.min_paired_samples < 3:
            raise ValueError("min_paired_samples must be at least 3")
        if not 0 <= self.min_paired_fraction <= 1:
            raise ValueError("min_paired_fraction must lie in [0, 1]")

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
    # warning is suppressed and the mean is set to zero; the row stays all-NaN,
    # so it contributes no pair statistic.
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


def _loading_concentration(
    filled: np.ndarray, directions: np.ndarray, quantile: float
) -> np.ndarray:
    """Share of each direction's genomic loading energy in its top CpGs.

    ``filled`` is the mean-filled residual matrix, ``directions`` a ``(K,
    n_samples)`` block of orthonormal sample-space rows. A nuisance factor
    (purity, composition, batch) spreads its loading over the whole genome and
    scores low; a biological subgroup concentrates it in marker regions and
    scores high. See ``Eiger2Config.deflation_max_concentration`` for the
    measured separation.

    The relative rule scores its permutation-null directions with this same
    routine on the same matrix, so candidates and null are comparable.
    """

    loadings = filled @ directions.T
    top_n = max(1, int(round(quantile * loadings.shape[0])))
    concentration = np.empty(directions.shape[0], dtype=float)
    for component in range(directions.shape[0]):
        energy = loadings[:, component] ** 2
        total = energy.sum()
        concentration[component] = (
            float(np.sort(energy)[-top_n:].sum() / total) if total > 0 else 1.0
        )
    return concentration


@dataclass(frozen=True)
class LatentAxes:
    """Everything the latent-structure step knows, not only what it deflates.

    ``candidates`` holds EVERY component parallel analysis retained, one
    orthonormal row per component in sample space, with its ``singular``
    value, its ``concentration`` and its ``verdict``. ``deflation_mask`` says
    which of them the deflation step then projects out -- the
    ``nuisance_deflated`` ones -- and ``basis`` is that subset.

    The secondary axis annotations (:mod:`eiger2.axis`) use the
    ``biological_kept`` candidates. The supported workflow never deflates, so
    ``basis`` is informational there.
    """

    candidates: np.ndarray
    singular: np.ndarray
    concentration: np.ndarray
    verdicts: tuple[str, ...]
    deflation_mask: np.ndarray
    basis: np.ndarray
    diagnostics: dict

    def kept(self) -> np.ndarray:
        """The ``biological_kept`` candidate axes, in candidate-index order."""

        if self.candidates.size == 0:
            return self.candidates
        keep = np.array(
            [verdict == "biological_kept" for verdict in self.verdicts], dtype=bool
        )
        return np.ascontiguousarray(self.candidates[keep])

    def kept_indices(self) -> list[int]:
        return [
            index for index, verdict in enumerate(self.verdicts)
            if verdict == "biological_kept"
        ]


def _empty_latent_axes(n_samples: int, diagnostics: dict) -> LatentAxes:
    empty_rows = np.empty((0, n_samples), dtype=float)
    return LatentAxes(
        candidates=empty_rows,
        singular=np.empty(0, dtype=float),
        concentration=np.empty(0, dtype=float),
        verdicts=(),
        deflation_mask=np.empty(0, dtype=bool),
        basis=empty_rows,
        diagnostics=diagnostics,
    )


def latent_axes(
    residuals: np.ndarray,
    config: Eiger2Config,
    *,
    row_mask: np.ndarray | None = None,
    allow_relaxation: bool = True,
    overwrite_residuals: bool = False,
) -> LatentAxes:
    """:class:`LatentAxes` for ``residuals``: candidates AND the deflation subset.

    ``row_mask`` is a boolean over ``residuals``' rows, applied AFTER thinning:
    a thinned row whose mask entry is ``False`` takes no part in the SVD, in
    parallel analysis or in the verdicts. The axis annotations use it to drop
    every thinned row falling inside a candidate region, which removes the
    tested regions from the axes exactly.

    ``allow_relaxation=False`` refuses the mean-filled fallback that the
    deflation path uses when too few rows are complete. The axis annotations ask for
    that: filling a masked row's gaps with zeros would put the very rows the
    mask removed back into the axis, in a weaker form.

    With ``row_mask=None`` and ``allow_relaxation=True`` this is the shipped
    computation, unchanged to the bit.

    ``overwrite_residuals=True`` lets the loading-concentration step fill the
    missing entries of ``residuals`` with zeros in place instead of on a copy.
    The values used are identical; pass it only for a private copy the caller
    will not read again (it saves one matrix-sized allocation).
    """

    residuals = np.asarray(residuals, dtype=float)
    n_samples = residuals.shape[1]
    raw = residuals[:: config.deflation_thin]
    n_offered_before_mask = int(raw.shape[0])
    n_masked_out = 0
    if row_mask is not None:
        mask = np.asarray(row_mask, dtype=bool)[:: config.deflation_thin]
        if mask.size != raw.shape[0]:
            raise ValueError("row_mask must have one entry per residual row")
        n_masked_out = int((~mask).sum())
        raw = raw[mask]
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
    if thinned.shape[0] < config.deflation_min_rows and not allow_relaxation:
        # The axis annotations' masked path: no mean-filled fallback, which
        # would reintroduce the masked rows. The SVD below still runs on the
        # complete rows that remain (at least 3).
        basis_source = "insufficient_rows_no_relaxation"
    elif thinned.shape[0] < config.deflation_min_rows:
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
    if row_mask is not None:
        diagnostics["thin_rows_offered_before_mask"] = n_offered_before_mask
        diagnostics["thin_rows_masked_out"] = n_masked_out
        diagnostics["relaxation_allowed"] = bool(allow_relaxation)
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
        diagnostics.update(n_components=0, n_candidates=0, method="insufficient_rows")
        return _empty_latent_axes(n_samples, diagnostics)

    _, singular, right = np.linalg.svd(thinned, full_matrices=False)

    # Under deflation_rule="relative" the boundary is read
    # off the SAME permutation ensemble the rank test already draws, so the null
    # components' directions have to be kept as well as their singular values.
    # Kept only when that rule is in force: under "absolute" the SVD call, its
    # LAPACK driver and therefore every committed number are untouched, and the
    # rng is consumed identically either way (the permutations are drawn before
    # the branch, in the same order, whatever is done with them afterwards).
    relative = config.deflation_rule == "relative"
    null_directions: list[np.ndarray] = []
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
            if relative:
                _, null_values, null_right = np.linalg.svd(
                    permuted, full_matrices=False
                )
                null_singular[replicate] = null_values
                null_directions.append(null_right)
            else:
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
        return _empty_latent_axes(n_samples, diagnostics)

    # Keep only the candidates whose genomic loading is diffuse. A nuisance
    # factor spreads its loading over the whole genome; a biological subgroup
    # concentrates it in marker regions and must not be deflated away.
    candidates = np.ascontiguousarray(right[:n_components])
    if overwrite_residuals:
        residuals[~np.isfinite(residuals)] = 0.0
        filled_full = residuals
    else:
        filled_full = _fill_rows(residuals)
    quantile = config.deflation_concentration_quantile
    concentration = _loading_concentration(filled_full, candidates, quantile)

    # The boundary the concentrations are read against. Under "absolute" it is
    # the shipped constant, so this whole block reduces to the plain
    # comparison; under "relative" it is a quantile of the concentration of
    # components fitted where locality is broken -- the same directions the rank
    # test permuted, scored through the same routine on the same residual
    # matrix, so candidate and null differ in exactly one thing: whether the
    # sample-space direction was fitted to real structure or to permuted noise.
    boundary = float(config.deflation_max_concentration)
    boundary_source = "deflation_max_concentration"
    null_concentration = np.array([], dtype=float)
    if relative and null_directions:
        null_concentration = np.concatenate([
            _loading_concentration(
                filled_full,
                np.ascontiguousarray(null_right[:n_components]),
                quantile,
            )
            for null_right in null_directions
        ])
        boundary = float(
            np.quantile(null_concentration, config.deflation_null_quantile)
        )
        boundary_source = "permutation_null_quantile"

    if config.deflate_components is not None:
        # an explicit K is a user override; report the verdicts but honour it
        diffuse = np.ones(n_components, dtype=bool)
    else:
        diffuse = concentration <= boundary
    diagnostics["loading_concentration"] = concentration.round(4).tolist()
    diagnostics["component_verdict"] = [
        "nuisance_deflated" if flag else "biological_kept" for flag in diffuse
    ]
    diagnostics["n_components"] = int(diffuse.sum())
    # Every candidate sitting within
    # DEFLATION_NEAR_CUTOFF_MARGIN of the cutoff, on either side, named with its
    # concentration and the cutoff it is near. Recorded on every run that gets as
    # far as scoring candidates, empty list included, so a consumer can tell
    # "none near the cutoff" from "not computed". The verdict travels with each
    # entry because the note is about the decision being marginal, not about
    # which way it went.
    diagnostics["deflation_max_concentration"] = float(
        config.deflation_max_concentration
    )
    diagnostics["deflation_near_cutoff_margin"] = DEFLATION_NEAR_CUTOFF_MARGIN
    # The realised boundary is recorded exactly as the
    # absolute path records its cutoff, plus -- under the relative rule -- the
    # null band it was read off, so a consumer can re-read the decision at
    # another quantile without re-running. Under "absolute" `deflation_boundary`
    # equals `deflation_max_concentration` by construction and the null band is
    # absent, so the two paths' manifests differ only where they genuinely do.
    diagnostics["deflation_rule"] = config.deflation_rule
    diagnostics["deflation_boundary"] = float(boundary)
    diagnostics["deflation_boundary_source"] = boundary_source
    if null_concentration.size:
        diagnostics["deflation_null_quantile"] = float(
            config.deflation_null_quantile
        )
        diagnostics["deflation_null_concentration"] = {
            "n": int(null_concentration.size),
            "n_replicates": int(len(null_directions)),
            "min": float(null_concentration.min()),
            "median": float(np.median(null_concentration)),
            "max": float(null_concentration.max()),
        }
    diagnostics["near_cutoff_components"] = [
        {
            "component": int(index),
            "loading_concentration": float(round(value, 4)),
            "verdict": (
                "nuisance_deflated" if diffuse[index] else "biological_kept"
            ),
        }
        for index, value in enumerate(concentration)
        if abs(float(value) - boundary) <= DEFLATION_NEAR_CUTOFF_MARGIN
    ]
    return LatentAxes(
        candidates=candidates,
        singular=np.ascontiguousarray(singular[:n_components]),
        concentration=concentration,
        verdicts=tuple(diagnostics["component_verdict"]),
        deflation_mask=diffuse,
        basis=(
            np.ascontiguousarray(candidates[diffuse])
            if diffuse.any()
            else np.empty((0, n_samples), dtype=float)
        ),
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------
# step 3 — the local co-variation statistic
# --------------------------------------------------------------------------


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
    """Pairwise-complete covariance or correlation, row by row."""

    return _pair_statistic_numpy(
        left, right, statistic, min_paired_samples, min_paired_fraction,
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
    is true of the observed track and false of this one.

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
        "slice bounds violated in the distant-lag window assembly"
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


def _chromosome_blocks(chromosome: np.ndarray) -> list[tuple[str, int, int]]:
    """``(name, start, end)`` for each contiguous chromosome block in
    ``chromosome`` (``end`` exclusive), in the order the blocks appear.

    Assumes rows are already grouped by chromosome -- true after
    :func:`chromosome_sort_permutation` (spec §2b prerequisite), which every
    caller of this function runs first. A chromosome appearing twice,
    non-contiguously, would silently split into two blocks here; nothing
    re-validates that upstream guarantee at this point.
    """

    if chromosome.size == 0:
        return []
    boundaries = np.flatnonzero(chromosome[1:] != chromosome[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [chromosome.size]))
    return [
        (str(chromosome[s]), int(s), int(e))
        for s, e in zip(starts, ends, strict=True)
    ]


def chromosome_lag_feasibility(
    locations: pd.DataFrame,
    config: Eiger2Config,
    *,
    observed_eligible: np.ndarray | None = None,
) -> dict[str, dict]:
    """Per chromosome: whether a feasible locality-breaking lag exists, and
    that chromosome's own share of eligible observed edges (spec §2b.1/§2b.3).

    A chromosome admits a feasible lag only if ``l_max >= config.null_min_lag``,
    where ``l_max = (n_c - 2) // (span - 1)`` -- the largest lag at which
    :func:`_lag_null_track` still has at least 2 usable positions
    (``usable = n_c - lag * (span - 1) >= 2``; matching the pre-§2b
    genome-wide code's own ``largest = (n_rows - 2) // (span - 1)``
    convention, not just ``n_c // (span - 1)``, which overestimates ``l_max``
    by enough that a lag drawn at the reported boundary can still leave
    ``usable < 2`` and silently produce an empty piece). The largest
    chromosome(s) dominate a single genome-wide lag draw (spec §2b.1); this
    measures feasibility PER chromosome instead, the basis both for
    :func:`null_tracks`'s per-chromosome construction (§2b.2) and for the
    missing-exposure diagnostic an infeasible chromosome is reported under
    (§2b.3), rather than excluding it behind a fixed, arbitrary threshold.

    ``observed_eligible`` lets a caller that already computed
    ``eligible_edges(locations, config)`` pass it straight through instead
    of this function recomputing it.
    """

    chromosome = locations["chr"].to_numpy()
    if observed_eligible is None:
        observed_eligible = eligible_edges(locations, config)
    span = max(config.window, 2)
    result: dict[str, dict] = {}
    for name, start, end in _chromosome_blocks(chromosome):
        n_c = end - start
        l_max = max(n_c - 2, 0) // max(span - 1, 1)
        # observed_eligible is edge-indexed (one shorter than CpG-indexed):
        # this chromosome's OWN internal edges are entries start..end-2; the
        # boundary edge at end-1 (into the NEXT chromosome, if any) belongs
        # to neither block and is correctly excluded by the end-1 stop.
        edges = int(observed_eligible[start : max(end - 1, start)].sum())
        result[name] = {
            "n_cpg": int(n_c),
            "l_max": int(l_max),
            "feasible": bool(l_max >= config.null_min_lag),
            "eligible_observed_edges": edges,
        }
    return result


def null_tracks(
    residuals: np.ndarray,
    config: Eiger2Config,
    locations: pd.DataFrame | None = None,
) -> list[tuple[np.ndarray, np.ndarray, list[tuple[str, int, int, int]]]]:
    """Statistic tracks under the locality-breaking null, one per replicate.

    Chromosome-aware (spec §2b.2): each replicate draws an INDEPENDENT
    feasible lag per chromosome -- not one genome-wide lag applied
    everywhere -- builds that chromosome's null segment from its own rows,
    and concatenates every feasible chromosome's segment into one
    genome-wide realisation, with the eligibility mask broken at every
    concatenation seam so a window can never straddle two chromosomes'
    segments. This preserves chromosome composition in every replicate:
    each chromosome contributes null volume roughly in proportion to its own
    share of eligible observed edges. A single genome-wide lag draw does not
    -- it survives almost entirely on the largest chromosome(s), since a
    lag large enough to be locality-breaking on the whole genome is rarely
    small enough to fit inside a smaller one.

    Chromosomes with no feasible lag (``n_c // (span - 1) < null_min_lag``)
    contribute nothing to the null in any replicate; their exposure is
    measured separately by :func:`chromosome_lag_feasibility` (spec §2b.3),
    not this function's concern.

    Returns ``(track, eligible, segments)`` triples. ``segments`` is
    ``[(chromosome, lag, start, length), ...]`` in concatenation order,
    describing which slice of ``track``/``eligible`` came from which
    chromosome, at what lag, and at what genome-wide row offset (``start``)
    -- needed by the support-conditioned threshold to reconstruct each
    segment's own window-member set (mirroring ``_lag_null_track``'s
    construction) against ``deflated``'s genome-wide row indexing when
    computing that segment's own support; a single scalar lag can no longer
    describe a whole replicate now that one replicate spans multiple
    chromosomes' own lags.

    Passing ``locations=None`` treats every row as one chromosome (the
    single-chromosome / genome-wide construction, unchanged) and is retained
    for single-chromosome use and for tests that do not exercise chromosome
    awareness at all.
    """

    span = max(config.window, 2)
    if locations is None:
        chromosome = np.full(residuals.shape[0], "", dtype=object)
        l_max = max(residuals.shape[0] - 2, 0) // max(span - 1, 1)
        feasibility = {
            "": {
                "n_cpg": residuals.shape[0],
                "l_max": l_max,
                "feasible": l_max >= config.null_min_lag,
            }
        }
        observed_eligible = None
    else:
        chromosome = locations["chr"].to_numpy()
        observed_eligible = eligible_edges(locations, config)
        feasibility = chromosome_lag_feasibility(
            locations, config, observed_eligible=observed_eligible,
        )

    blocks = [
        (name, start, end) for name, start, end in _chromosome_blocks(chromosome)
        if feasibility[name]["feasible"]
    ]
    rng = np.random.default_rng(config.random_seed + 977)
    tracks: list[tuple[np.ndarray, np.ndarray, list[tuple[str, int, int, int]]]] = []
    for _ in range(config.null_replicates):
        track_pieces: list[np.ndarray] = []
        eligible_pieces: list[np.ndarray] = []
        segments: list[tuple[str, int, int, int]] = []
        for name, start, end in blocks:
            l_max = feasibility[name]["l_max"]
            lag = int(rng.integers(config.null_min_lag, l_max + 1))
            piece = _lag_null_track(residuals[start:end], lag, config)
            if not piece.size:
                continue
            observed_eligible_c = (
                observed_eligible[start : max(end - 1, start)]
                if observed_eligible is not None else None
            )
            piece_eligible = _null_eligibility(
                None, lag, config, piece.size,
                observed_eligible=observed_eligible_c,
            )
            if not piece_eligible.any():
                continue
            # Broken at the boundary, not left as `eligible_edges` would
            # naturally leave it: a run must never cross from one
            # chromosome's segment into the next's, even though both are
            # individually eligible right up to their own last position --
            # exactly `member_step_lag_null_ensemble`'s existing device for
            # the same reason (concatenated offset sub-tracks).
            piece_eligible = piece_eligible.copy()
            piece_eligible[-1] = False
            track_pieces.append(piece)
            eligible_pieces.append(piece_eligible)
            segments.append((name, lag, int(start), int(piece.size)))
        if not track_pieces:
            continue
        tracks.append((
            np.concatenate(track_pieces), np.concatenate(eligible_pieces), segments,
        ))
    return tracks


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
    SAME threshold. The ratio ``p_obs / p_null`` is a descriptive diagnostic:
    1.0 when the null and data share the same upper-tail shape, rising when
    the observed track is more heavy-tailed than the null represents.

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


def _null_quantile_by_chromosome(
    tracks: list[tuple[np.ndarray, np.ndarray, list[tuple[str, int, int, int]]]],
    quantile: float,
) -> dict[str, float]:
    """The SAME ``null_quantile`` computed separately per chromosome, from
    the pooled null values :func:`null_tracks` already produced for it
    (``segments`` metadata) -- no recomputation of anything upstream, just
    grouping and re-quantiling data already in hand (spec §2b.3's
    sensitivity report). Shows whether pooling every feasible chromosome's
    null together masks one whose own null behaves differently, at the cost
    of one extra quantile per chromosome, not a second run.
    """

    pools: dict[str, list[np.ndarray]] = {}
    for track, eligible, segments in tracks:
        position = 0
        for name, _lag, _start, length in segments:
            piece = track[position : position + length]
            usable = eligible[position : position + length] & np.isfinite(piece)
            if usable.any():
                pools.setdefault(name, []).append(piece[usable])
            position += length
    return {
        name: float(np.quantile(np.concatenate(values), quantile))
        for name, values in pools.items()
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
    threshold_by_chromosome = _null_quantile_by_chromosome(
        tracks, config.null_quantile,
    )
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
        # spec §2b.3's sensitivity report: the same quantile recomputed per
        # feasible chromosome (never a second run -- see
        # _null_quantile_by_chromosome), alongside the pooled "threshold"
        # above it is a comparison for.
        "threshold_by_chromosome": threshold_by_chromosome,
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
# step 5 — segmentation and region scores
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


# --------------------------------------------------------------------------
# locations and chromosome order
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


# Human reference chromosomes in their natural biological order -- not
# lexicographic ("chr10" < "chr2" as strings, which a prior version's
# reject-on-mismatch input check enforced, rejecting valid natural-order
# input over it -- spec §2b prerequisite). Any other chromosome name
# (unplaced/alt/random contigs) sorts after this set, lexicographically
# among themselves: deterministic and independent of the order they
# happened to be presented in, which is what makes the canonical order
# reproducible regardless of input order.
CANONICAL_CHROMOSOMES: tuple[str, ...] = tuple(
    f"chr{i}" for i in range(1, 23)
) + ("chrX", "chrY", "chrM")
_CANONICAL_CHROMOSOME_RANK = {
    name: rank for rank, name in enumerate(CANONICAL_CHROMOSOMES)
}


def canonical_chromosome_order(names: Sequence[str]) -> list[str]:
    """The canonical presentation order for a SET of chromosome names.

    A pure function of the SET of distinct names present -- not of the order
    they were given in -- so the same chromosomes always canonicalise to the
    same order regardless of whether the input happened to be presented in
    natural order, lexicographic order, or shuffled (spec §2b prerequisite).
    ``chr1``..``chr22``, ``chrX``, ``chrY``, ``chrM`` sort in that biological
    order; anything else (unplaced, alt, or random contigs) sorts after them,
    lexicographically among themselves.
    """

    unique = set(names)
    known = sorted(
        (n for n in unique if n in _CANONICAL_CHROMOSOME_RANK),
        key=lambda n: _CANONICAL_CHROMOSOME_RANK[n],
    )
    unplaced = sorted(n for n in unique if n not in _CANONICAL_CHROMOSOME_RANK)
    return known + unplaced


def chromosome_sort_permutation(
    locations: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Permutation that groups ``locations`` by chromosome in canonical
    order and sorts positions ascending within each chromosome, plus the
    canonical order of the chromosomes actually present -- recorded in the
    manifest (``chromosome_order``) so a run is reproducible regardless of
    input order (spec §2b prerequisite). Apply the SAME permutation to the
    corresponding matrix rows to keep them aligned with ``locations``.

    Every downstream step that infers genomic proximity from row-index
    proximity (:func:`eligible_edges`, :func:`_null_eligibility`, and
    therefore the null construction §2b builds on) depends on this -- a
    chromosome split across two non-adjacent row blocks, or positions not
    ascending within a chromosome, corrupts the null it builds without
    raising, since nothing here currently checks for the CONDITION this
    permutation guarantees, only relies on it.
    """

    chrom = locations["chr"].to_numpy()
    starts = locations["start"].to_numpy()
    order = canonical_chromosome_order(chrom)
    rank = {name: index for index, name in enumerate(order)}
    chrom_rank = np.array([rank[c] for c in chrom], dtype=np.int64)
    # lexsort's LAST key is primary: chromosome rank first, start ascending
    # within each chromosome second. Stable, so rows already in canonical
    # order (including duplicate positions) keep their relative order.
    permutation = np.lexsort((starts, chrom_rank))
    return permutation, order
