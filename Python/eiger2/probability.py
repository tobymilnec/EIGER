"""Per-region posterior probabilities from the pipeline null.

``core.region_fdr`` answers a set-level question: *among the regions at least
this extreme, what proportion are expected to be null?* That is a tail-area
rate. It is routinely misread as a statement about the individual region it is
attached to, and it is not one.

This module supplies the per-region quantity, using Efron's two-groups model.
Writing ``a`` for region area, the observed regions are a mixture of null and
non-null regions,

    f(a) = pi0 * f0(a) + (1 - pi0) * f1(a)

and the local false discovery rate is

    fdr(a) = pi0 * f0(a) / f(a),      P(region is real | a) = 1 - fdr(a)

Every term is available without new machinery:

``f0``   the null area density, estimated from the null tracks pushed through
         the *identical* segmenter. ``core.call_dmrs`` already computes these
         areas; they were previously used only for a tail count.
``f``    the observed area density, from the called regions.
``pi0``  the null proportion, obtained directly by rescaling the pooled null
         region count to the size of the observed genome. Setting ``pi0 = 1``
         recovers the conservative behaviour of the tail-area rate, so the
         model is strictly nested inside what EIGER 2 already did.

Why this is not a repeat of EIGER 1's probability fields
--------------------------------------------------------
EIGER 1 exposed ``sample_model_probability``, ``effect_probability``,
``call_probability`` and ``region_probability``. None was calibrated against
anything. ``region_probability`` was >= 0.5 by construction, because regions
were *defined* as runs whose posterior exceeded the calling threshold, so the
field could never flag a weak region. The BIC transform ``expit(0.5 * dBIC)``
is the Schwarz approximation, whose regularity conditions fail for mixtures.

The quantity here differs in three respects that matter:

1. it is a posterior for a stated model rather than a rescaled test statistic;
2. it cannot be bounded below by 0.5 by construction, because ``f0`` is
   estimated from regions the segmenter actually produced on null data rather
   than assumed;
3. it is checkable. :func:`reliability_table` bins regions by predicted
   probability and reports the observed fraction that are genuinely real.

The third point is the operative one. **No probability from this module should
be reported without its reliability diagram.** If the curve does not sit on the
diagonal the output is a score, and must be named one.

Limitations, stated plainly
---------------------------
* Density estimation is least reliable in the tail, which is exactly where the
  interesting regions are. Areas beyond the support of the pooled null are
  flagged ``extrapolated`` and given the conservative tail-area rate instead of
  an extrapolated density ratio.
* A density needs far more null regions than a tail count does.
  ``min_null_regions`` (default 50) is enforced, and the estimate is refused
  rather than fudged when the null is too thin. In practice this means raising
  ``Eiger2Config.null_replicates`` from 20 to 100-200 when posteriors are
  wanted.
* ``pi0`` is unstable when the non-null fraction is large. Both the estimated
  and the conservative ``pi0 = 1`` results are returned.
* Length stratification multiplies the data requirement by the number of
  strata. Strata below ``min_null_regions`` fall back to the pooled null and
  are flagged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .core import length_stratum

__all__ = [
    "LocalFdrResult",
    "calibration_error",
    "estimate_pi0",
    "local_fdr",
    "reliability_table",
]


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------


@dataclass
class LocalFdrResult:
    """Per-region local false discovery rate and posterior probability."""

    fdr: np.ndarray
    """Local false discovery rate, ``P(region is null | area)``."""

    posterior: np.ndarray
    """``1 - fdr``. The probability the region is real, given its area."""

    fdr_conservative: np.ndarray
    """The same quantity computed with ``pi0 = 1``. Always >= ``fdr``."""

    extrapolated: np.ndarray
    """True where the area lies beyond the support of the pooled null, so the
    density ratio would have been extrapolated. These entries carry the
    tail-area rate instead."""

    pi0: float
    """Estimated proportion of called regions that are null."""

    status: str
    """``"ok"``, or the reason the estimate was refused."""

    diagnostics: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "local_fdr": self.fdr,
                "posterior_real": self.posterior,
                "local_fdr_conservative": self.fdr_conservative,
                "posterior_extrapolated": self.extrapolated,
            }
        )


# --------------------------------------------------------------------------
# density estimation
# --------------------------------------------------------------------------


def _log_area(areas: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Region areas are positive and heavy-tailed; work on the log scale."""

    values = np.asarray(areas, dtype=float)
    return np.log(np.clip(values, floor, None))


def _silverman_bandwidth(values: np.ndarray) -> float:
    """Silverman's rule, guarded against zero spread."""

    n = values.size
    if n < 2:
        return 1.0
    sd = float(np.std(values, ddof=1))
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    if not np.isfinite(spread) or spread <= 0:
        spread = sd if sd > 0 else 1.0
    return float(0.9 * spread * n ** (-0.2))


def _gaussian_kde(sample: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """Plain Gaussian KDE evaluated on ``grid``.

    Written out rather than taken from scipy so the bandwidth is shared
    explicitly between the null and observed densities. Using a different
    bandwidth for each creates structure in the ratio that is an artefact of
    the smoother rather than of the data.
    """

    if sample.size == 0 or bandwidth <= 0:
        return np.zeros_like(grid)
    z = (grid[:, None] - sample[None, :]) / bandwidth
    kernel = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return kernel.sum(axis=1) / (sample.size * bandwidth)


# --------------------------------------------------------------------------
# pi0
# --------------------------------------------------------------------------


def estimate_pi0(
    n_observed: int,
    n_pooled_null: int,
    *,
    observed_edges: int,
    null_edges: int,
) -> float:
    """Proportion of called regions expected to be null.

    The pipeline null gives this directly. ``n_pooled_null`` regions were
    produced across null tracks covering ``null_edges`` edges in total, so the
    expected number of null regions in a genome of ``observed_edges`` edges is
    ``n_pooled_null * observed_edges / null_edges``. Dividing by the number of
    regions actually called gives ``pi0``.

    This is preferred over a Storey-style lambda estimator because the pipeline
    supplies a direct null count rather than requiring one to be inferred from
    the shape of the observed distribution. Clipped to ``[0, 1]``.
    """

    if n_observed <= 0 or null_edges <= 0:
        return 1.0
    expected_null = n_pooled_null * (observed_edges / null_edges)
    return float(np.clip(expected_null / n_observed, 0.0, 1.0))


# --------------------------------------------------------------------------
# tail-area rate, used as the conservative fallback in the extrapolated tail
# --------------------------------------------------------------------------


def _tail_area_fdr(
    observed_areas: np.ndarray, pooled_null: np.ndarray, scale: float
) -> np.ndarray:
    pooled = np.sort(pooled_null[np.isfinite(pooled_null)])
    order = np.argsort(-observed_areas, kind="mergesort")
    ranked = observed_areas[order]
    expected = (pooled.size - np.searchsorted(pooled, ranked, side="left")) * scale
    counts = np.arange(1, ranked.size + 1, dtype=float)
    raw = np.clip(expected / counts, 0.0, 1.0)
    monotone = np.minimum.accumulate(raw)
    out = np.empty_like(monotone)
    out[order] = monotone
    return out


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------


def _local_fdr_one_stratum(
    observed_areas: np.ndarray,
    pooled_null: np.ndarray,
    *,
    pi0: float,
    scale: float,
    grid_size: int,
    bandwidth_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Local fdr for one homogeneous set of regions.

    Returns ``(fdr, extrapolated, diagnostics)``.
    """

    obs_log = _log_area(observed_areas)
    null_log = _log_area(pooled_null)

    # One bandwidth, taken from the larger sample, used for both densities.
    bandwidth = bandwidth_scale * _silverman_bandwidth(
        null_log if null_log.size >= obs_log.size else obs_log
    )
    low = min(obs_log.min(), null_log.min()) - 3 * bandwidth
    high = max(obs_log.max(), null_log.max()) + 3 * bandwidth
    grid = np.linspace(low, high, grid_size)

    f_null = _gaussian_kde(null_log, grid, bandwidth)
    f_obs = _gaussian_kde(obs_log, grid, bandwidth)

    # fdr on the grid, then interpolated back to the regions. Both densities can
    # underflow to zero far from their support, so the ratio is only formed
    # where the observed density is meaningfully positive; elsewhere it is left
    # undefined and filled by the monotone step below.
    floor = 1e-300
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.where(f_obs > floor, pi0 * f_null / np.maximum(f_obs, floor), np.nan)
    ratio = np.clip(np.nan_to_num(ratio, nan=np.nan, posinf=1.0), 0.0, 1.0)

    # The statistic is one-sided: a larger area is never weaker evidence. Impose
    # that, which also stabilises the sparse tail.
    finite = np.isfinite(ratio)
    if finite.any():
        filled = pd.Series(ratio).ffill().bfill().to_numpy()
    else:
        filled = np.ones_like(ratio)
    monotone = np.minimum.accumulate(filled)  # grid ascends in area

    fdr = np.interp(obs_log, grid, monotone)

    # Beyond the support of the pooled null the density ratio is extrapolation,
    # not estimation. Fall back to the conservative tail-area rate there.
    null_max = float(null_log.max())
    extrapolated = obs_log > null_max
    if extrapolated.any():
        tail = _tail_area_fdr(observed_areas, pooled_null, scale)
        fdr = np.where(extrapolated, tail, fdr)

    diagnostics = {
        "bandwidth_log_area": float(bandwidth),
        "n_observed": int(observed_areas.size),
        "n_pooled_null": int(pooled_null.size),
        "null_area_max": float(np.exp(null_max)),
        "n_extrapolated": int(extrapolated.sum()),
    }
    return np.clip(fdr, 0.0, 1.0), extrapolated, diagnostics


def local_fdr(
    observed_areas: np.ndarray,
    null_area_sets: Sequence[np.ndarray],
    *,
    observed_edges: int,
    null_edges: int,
    observed_lengths: np.ndarray | None = None,
    null_length_sets: Sequence[np.ndarray] | None = None,
    min_null_regions: int = 50,
    grid_size: int = 512,
    bandwidth_scale: float = 1.0,
    pi0: float | None = None,
) -> LocalFdrResult:
    """Two-groups local false discovery rate for called regions.

    Parameters mirror :func:`core.region_fdr` so the two can be computed from
    the same inputs. ``null_edges`` is the total across replicates.

    ``min_null_regions`` guards the estimate: below it the null area density
    cannot be estimated and the result is refused, with ``status`` explaining
    why, rather than a number being produced that looks usable and is not.
    """

    observed_areas = np.asarray(observed_areas, dtype=float)
    n_obs = observed_areas.size
    nan = np.full(n_obs, np.nan, dtype=float)
    false = np.zeros(n_obs, dtype=bool)

    if n_obs == 0:
        return LocalFdrResult(nan, nan, nan, false, 1.0, "no_regions")
    if not null_area_sets or null_edges <= 0:
        return LocalFdrResult(nan, nan, nan, false, 1.0, "no_null")

    pooled_null = np.concatenate(
        [np.asarray(a, dtype=float) for a in null_area_sets]
    )
    pooled_null = pooled_null[np.isfinite(pooled_null) & (pooled_null > 0)]
    scale = observed_edges / null_edges

    if pooled_null.size < min_null_regions:
        return LocalFdrResult(
            nan,
            nan,
            nan,
            false,
            1.0,
            f"insufficient_null_regions({pooled_null.size}<{min_null_regions})",
            {
                "hint": "raise Eiger2Config.null_replicates; a density needs "
                "far more null regions than a tail count does",
                "n_pooled_null": int(pooled_null.size),
            },
        )

    if pi0 is None:
        pi0 = estimate_pi0(
            n_obs,
            pooled_null.size,
            observed_edges=observed_edges,
            null_edges=null_edges,
        )

    diagnostics: dict = {"pi0": float(pi0), "scale": float(scale), "strata": {}}

    stratified = observed_lengths is not None and null_length_sets is not None
    fdr = np.empty(n_obs, dtype=float)
    extrapolated = np.zeros(n_obs, dtype=bool)

    if not stratified:
        fdr, extrapolated, stratum_diag = _local_fdr_one_stratum(
            observed_areas,
            pooled_null,
            pi0=pi0,
            scale=scale,
            grid_size=grid_size,
            bandwidth_scale=bandwidth_scale,
        )
        diagnostics["strata"]["pooled"] = stratum_diag
    else:
        obs_len = np.asarray(observed_lengths, dtype=int)
        null_len = np.concatenate(
            [np.asarray(a, dtype=int) for a in null_length_sets]
        )
        null_areas_all = np.concatenate(
            [np.asarray(a, dtype=float) for a in null_area_sets]
        )
        keep = np.isfinite(null_areas_all) & (null_areas_all > 0)
        null_areas_all = null_areas_all[keep]
        null_len = null_len[keep]

        obs_strata = np.array([length_stratum(int(v)) for v in obs_len], dtype=object)
        null_strata = np.array(
            [length_stratum(int(v)) for v in null_len], dtype=object
        )
        for name in np.unique(obs_strata):
            rows = np.flatnonzero(obs_strata == name)
            pool = null_areas_all[null_strata == name]
            fell_back = False
            if pool.size < min_null_regions:
                pool = pooled_null
                fell_back = True
            part_fdr, part_extra, stratum_diag = _local_fdr_one_stratum(
                observed_areas[rows],
                pool,
                pi0=pi0,
                scale=scale,
                grid_size=grid_size,
                bandwidth_scale=bandwidth_scale,
            )
            fdr[rows] = part_fdr
            extrapolated[rows] = part_extra
            stratum_diag["fell_back_to_pooled_null"] = bool(fell_back)
            diagnostics["strata"][str(name)] = stratum_diag

    conservative, _, _ = _local_fdr_one_stratum(
        observed_areas,
        pooled_null,
        pi0=1.0,
        scale=scale,
        grid_size=grid_size,
        bandwidth_scale=bandwidth_scale,
    )

    # Consistency with the tail-area rate. The average local fdr among regions
    # at least as extreme as a cut-off should track the tail-area rate at that
    # cut-off. A large discrepancy means the density estimate is unreliable.
    tail = _tail_area_fdr(observed_areas, pooled_null, scale)
    order = np.argsort(-observed_areas, kind="mergesort")
    running_mean = np.cumsum(fdr[order]) / np.arange(1, n_obs + 1)
    diagnostics["max_abs_tail_area_discrepancy"] = float(
        np.nanmax(np.abs(running_mean - tail[order]))
    )
    diagnostics["n_extrapolated"] = int(extrapolated.sum())

    return LocalFdrResult(
        fdr=fdr,
        posterior=1.0 - fdr,
        fdr_conservative=np.maximum(conservative, fdr),
        extrapolated=extrapolated,
        pi0=float(pi0),
        status="ok",
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------
# calibration — the thing that makes the number reportable
# --------------------------------------------------------------------------


def reliability_table(
    posterior: np.ndarray,
    is_true: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Observed against predicted probability, binned.

    ``is_true`` is the ground-truth indicator for each region, available only in
    simulation. A calibrated posterior puts ``observed_fraction_real`` on the
    diagonal against ``mean_predicted``.
    """

    posterior = np.asarray(posterior, dtype=float)
    is_true = np.asarray(is_true, dtype=bool)
    keep = np.isfinite(posterior)
    posterior, is_true = posterior[keep], is_true[keep]
    if posterior.size == 0:
        return pd.DataFrame(
            columns=[
                "bin_low",
                "bin_high",
                "n_regions",
                "mean_predicted",
                "observed_fraction_real",
            ]
        )

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(posterior, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = which == b
        if not sel.any():
            continue
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "n_regions": int(sel.sum()),
                "mean_predicted": float(posterior[sel].mean()),
                "observed_fraction_real": float(is_true[sel].mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_error(table: pd.DataFrame) -> dict:
    """Expected and maximum calibration error from a reliability table."""

    if table.empty:
        return {"expected_calibration_error": float("nan"),
                "max_calibration_error": float("nan"),
                "n_regions": 0}
    weight = table["n_regions"].to_numpy(dtype=float)
    gap = np.abs(
        table["mean_predicted"].to_numpy() - table["observed_fraction_real"].to_numpy()
    )
    return {
        "expected_calibration_error": float(np.sum(weight * gap) / weight.sum()),
        "max_calibration_error": float(gap.max()),
        "n_regions": int(weight.sum()),
    }
