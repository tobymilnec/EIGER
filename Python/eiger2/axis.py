"""Axis effects: secondary annotations of candidate regions.

These annotations describe how a candidate region's per-sample mean
methylation varies along genome-wide sample axes. They are strictly secondary:
they never add, remove, rescore or reorder a candidate, never enter a
permutation scan, and never change ``coordination_qvalue`` or
``coordination_selected``.

**Axis family.** Every pre-FDR candidate region of the run's observed scan (the
whole analysed scope, one caller profile). The axes are estimated once per run
from the analysed matrix with every CpG row inside any candidate region masked
out; each candidate is then fitted against the same axes; only afterwards are
coordination-selected regions read off the annotated table. Axes are never
refitted, and no multiple-testing pass is redone, on the selected subset.

**Axes.** Per-CpG centred residuals, thinned (``Eiger2Config.deflation_thin``),
candidate-masked, complete rows only, with no mean-filled relaxation
(:func:`eiger2.core.latent_axes` with ``row_mask`` and
``allow_relaxation=False``); the number of components comes from parallel
analysis and only components with verdict ``biological_kept`` are used; each is
oriented by a label-free sign rule (:func:`orient_axes`). The p10/p90 of each
axis's sample scores are fixed once per run (:func:`axis_reference_points`).

**Effect.** For each candidate, the logit of each tested sample's region mean
is regressed on an intercept plus the K axes (joint least squares). The axis
effect is the exact beta-scale contrast ``expit(b0 + b_k p90_k) - expit(b0 +
b_k p10_k)`` for the reported axis, with a delta-method standard error from
the HC3 covariance and a ``2K``-Bonferroni interval. The reported axis is the
one with the smallest corrected minimum-effect p-value (the reviewed selection
rule). The standard error and interval describe the contrast against the axis
this run estimated, with that axis and its p10/p90 treated as fixed; they are
not an interval for a population axis contrast. This effect is not the
historical two-component ``delta_hat``.

**Inference status.** Default annotations are descriptive: effect, standard
error, interval, axis index, tested-sample count and status. The
minimum-effect test (``|Delta_k| <= 0.10`` null, two directions per axis,
``2K`` Bonferroni, Benjamini-Hochberg across the candidate family) has been
checked only in simulations with a single retained axis and no missing values,
which excludes typical real data, so its p-value, q-value and
``axis_supported`` flag are left missing by default. They are computed only in
the explicitly experimental mode, and then only for regions with at least
:data:`AXIS_MIN_SAMPLES_FOR_INFERENCE` tested samples; below that floor, or
with no usable axis or fit, inference is missing, never ``False`` or
``q = 1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .core import Eiger2Config, centre_cpgs, latent_axes

__all__ = [
    "AXIS_COLUMNS",
    "AXIS_INFERENCE_DEFAULT",
    "AXIS_INTERVAL_ALPHA",
    "AXIS_MIN_SAMPLES_FOR_INFERENCE",
    "BETA_CLAMP",
    "DELTA_MIN",
    "MIN_CONFIRM_SAMPLES",
    "annotate_axes",
    "axis_margin_test",
    "axis_reference_points",
    "benjamini_hochberg",
    "empty_axis_columns",
    "orient_axes",
    "region_sample_means",
]

BETA_CLAMP = (0.005, 0.995)
"""Betas are clamped here before the logit, so a 0 or a 1 stays finite."""

MIN_CONFIRM_SAMPLES = 10
"""Fewer tested samples than this and a region is not fitted at all.

The regression also needs ``K + 10`` tested samples, because it spends K
degrees of freedom before it estimates anything.
"""

DELTA_MIN = 0.10
"""Margin of the experimental minimum-effect null, on the beta scale."""

AXIS_INTERVAL_ALPHA = 0.05
"""Family-wise level of ``axis_effect_lower``/``axis_effect_upper``.

The interval carries the ``2K`` Bonferroni multiplicity of the test, so each
tail is ``AXIS_INTERVAL_ALPHA / (2K)``.
"""

AXIS_MIN_SAMPLES_FOR_INFERENCE = 200
"""Fewest tested samples in a region for experimental axis inference.

A floor on the region's own tested-sample count, not on the cohort. Below it
the effect and interval may still be reported; the p-value, q-value and
``axis_supported`` are missing.
"""

AXIS_INFERENCE_DEFAULT = "unavailable_pending_validation"
"""Default disposition of axis p-values, q-values and ``axis_supported``.

A targeted simulation study supported the test with one retained axis and
complete data only. Settings with several retained axes (where the per-axis
contrast depends on the basis) and with missing values are not yet validated,
so default runs leave inferential axis values missing.
"""

_MIN_OBSERVED_FRACTION = 0.5
"""A sample must observe at least half a region's CpGs to contribute a mean."""

_AXIS_LOW_QUANTILE = 0.10
_AXIS_HIGH_QUANTILE = 0.90

AXIS_COLUMNS = [
    "axis_status",
    "axis_n_axes",
    "axis_n_samples_tested",
    "axis_index",
    "axis_effect",
    "axis_effect_se",
    "axis_effect_lower",
    "axis_effect_upper",
    "axis_pvalue",
    "axis_qvalue",
    "axis_supported",
]

#: Per-region ``axis_status`` values.
AXIS_STATUSES = {
    "not_computed": "axis annotations were switched off for this run",
    "no_axes": "no usable candidate-masked cohort axis was estimated",
    "insufficient_samples": "too few tested samples to fit the regression",
    "fit_failed": "the regression could not be fitted (singular design)",
    "estimated": "effect and interval estimated; inference not requested",
    "below_inference_floor": (
        "effect estimated; experimental inference withheld below the "
        f"{AXIS_MIN_SAMPLES_FOR_INFERENCE}-tested-sample floor"
    ),
    "tested": "effect estimated and experimental inference computed",
}


# --------------------------------------------------------------------------
# carried over unchanged from the reviewed implementation (e050f8f)
# --------------------------------------------------------------------------


def region_sample_means(
    betas: np.ndarray, first: int, last: int
) -> tuple[np.ndarray, np.ndarray]:
    """Logit of each sample's mean beta over the region CpGs it observes.

    ``betas`` is the ``(n_cpg, n_samples)`` analysed matrix;
    ``first`` and ``last`` are inclusive row bounds. Returns the logit means of
    the tested samples and the boolean mask saying which samples those were.
    """

    block = np.asarray(betas, dtype=float)[first : last + 1]
    finite = np.isfinite(block)
    observed = finite.sum(axis=0)
    n_cpg = block.shape[0]
    tested = observed >= _MIN_OBSERVED_FRACTION * n_cpg
    tested &= observed > 0
    if not tested.any():
        return np.empty(0, dtype=float), tested
    totals = np.where(finite, block, 0.0).sum(axis=0)
    means = totals[tested] / observed[tested]
    clamped = np.clip(means, *BETA_CLAMP)
    return np.log(clamped / (1.0 - clamped)), tested


def orient_axes(axes: np.ndarray) -> np.ndarray:
    """Fix each axis's sign by a label-free rule, so ``axis_effect`` is stable.

    An SVD's right singular vector is only defined up to sign, so the sign of
    a slope -- and therefore of ``axis_effect`` -- would otherwise depend on
    LAPACK. The rule: flip each row so its largest-magnitude loading is
    positive, ties going to the lowest sample index. It reads no label, no
    phenotype and no ordering of the samples beyond the matrix's own columns.
    """

    axes = np.asarray(axes, dtype=float)
    if axes.size == 0:
        return axes if axes.ndim == 2 else np.empty((0, 0), dtype=float)
    axes = np.atleast_2d(axes)
    out = np.array(axes, dtype=float, copy=True)
    for row in range(out.shape[0]):
        values = out[row]
        pivot = int(np.argmax(np.abs(values)))
        if values[pivot] < 0:
            out[row] = -values
    return out


def axis_reference_points(axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The p10 and p90 of each axis's sample scores, over the whole cohort.

    Computed once per run, over every sample, so ``axis_effect`` is the same
    contrast for every region regardless of which samples a given region
    happened to test. Returned rather than recomputed per region for exactly
    that reason.
    """

    axes = np.asarray(axes, dtype=float)
    if axes.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    axes = np.atleast_2d(axes)
    low = np.quantile(axes, _AXIS_LOW_QUANTILE, axis=1)
    high = np.quantile(axes, _AXIS_HIGH_QUANTILE, axis=1)
    return low, high


def _hc3_covariance(design: np.ndarray, residuals: np.ndarray, inverse: np.ndarray):
    """HC3 sandwich covariance of the least-squares coefficients.

    ``(X'X)^-1 X' diag(e_i^2 / (1 - h_i)^2) X (X'X)^-1``. HC3 rather than the
    homoscedastic ``s^2 (X'X)^-1`` because a region's per-sample means are
    heteroscedastic almost by construction: a sample observing half the
    region's CpGs has a noisier mean than one observing all of them, and the
    logit transform makes the variance depend on the level as well.
    """

    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    denominator = np.clip((1.0 - leverage) ** 2, 1e-12, None)
    weights = residuals**2 / denominator
    middle = (design * weights[:, None]).T @ design
    return inverse @ middle @ inverse


def axis_margin_test(
    values: np.ndarray,
    axis_scores: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    delta_min: float = DELTA_MIN,
    alpha: float = AXIS_INTERVAL_ALPHA,
) -> dict:
    """Minimum-effect test of the region's logit means against K cohort axes.

    ``values`` are the tested samples' logit region means, ``axis_scores`` the
    ``(K, n_tested)`` block of oriented axis scores for the SAME samples in the
    same order, and ``low``/``high`` the run-level p10/p90 of each axis
    (:func:`axis_reference_points`).

    The region's logit mean is regressed on an intercept plus the K axes. For
    axis ``k`` the reported effect is the EXACT contrast

    ``Delta_k = expit(b0 + b_k * high_k) - expit(b0 + b_k * low_k)``

    with the other axes held at zero -- not the ``m(1-m)`` linearisation, which
    is wrong by an amount that grows with the effect it is measuring. Its
    standard error comes from the delta method on the HC3 covariance.

    The null is ``|Delta_k| <= delta_min``, tested in both directions by a
    one-sided t on ``n - K - 1`` degrees of freedom and Bonferroni-corrected
    across the ``2K`` tests. The reported axis is the one with the smallest
    corrected p-value. Returns ``None`` when the region cannot be tested.
    """

    values = np.asarray(values, dtype=float)
    scores = np.atleast_2d(np.asarray(axis_scores, dtype=float))
    n_axes, n = scores.shape
    if n_axes == 0 or values.size != n:
        return None
    if n < max(MIN_CONFIRM_SAMPLES, n_axes + 10):
        return None
    degrees = n - n_axes - 1
    if degrees <= 0:
        return None

    design = np.column_stack([np.ones(n), scores.T])
    gram = design.T @ design
    try:
        inverse = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(inverse)):
        return None
    coefficients = inverse @ (design.T @ values)
    residuals = values - design @ coefficients
    covariance = _hc3_covariance(design, residuals, inverse)

    family = 2 * n_axes
    best: dict | None = None
    for axis in range(n_axes):
        eta_high = coefficients[0] + coefficients[1 + axis] * high[axis]
        eta_low = coefficients[0] + coefficients[1 + axis] * low[axis]
        beta_high = 1.0 / (1.0 + np.exp(-eta_high))
        beta_low = 1.0 / (1.0 + np.exp(-eta_low))
        effect = float(beta_high - beta_low)
        slope_high = beta_high * (1.0 - beta_high)
        slope_low = beta_low * (1.0 - beta_low)
        gradient = np.zeros(n_axes + 1, dtype=float)
        gradient[0] = slope_high - slope_low
        gradient[1 + axis] = high[axis] * slope_high - low[axis] * slope_low
        variance = float(gradient @ covariance @ gradient)
        error = float(np.sqrt(variance)) if variance > 0 else 0.0
        for sign in (1.0, -1.0):
            if error > 0 and np.isfinite(error):
                statistic = (sign * effect - delta_min) / error
                raw = float(stats.t.sf(statistic, degrees))
            else:
                # No usable standard error is no evidence against the margin.
                statistic, raw = float("nan"), 1.0
            corrected = float(min(1.0, family * raw))
            if best is None or corrected < best["axis_pvalue"]:
                best = {
                    "axis_index": axis,
                    "axis_direction": int(sign),
                    "axis_effect": effect,
                    "axis_effect_se": error,
                    "axis_statistic": statistic,
                    "axis_pvalue_raw": raw,
                    "axis_pvalue": corrected,
                }
    if best is None:  # pragma: no cover - n_axes >= 1 guarantees one pass
        return None
    critical = (
        float(stats.t.ppf(1.0 - alpha / family, degrees))
        if best["axis_effect_se"] > 0 else float("nan")
    )
    best["axis_effect_lower"] = best["axis_effect"] - critical * best["axis_effect_se"]
    best["axis_effect_upper"] = best["axis_effect"] + critical * best["axis_effect_se"]
    best["axis_n_axes"] = int(n_axes)
    best["axis_degrees_of_freedom"] = int(degrees)
    best["axis_interval_tail_alpha"] = float(alpha / family)
    return best


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values. ``NaN`` entries stay out of the family entirely.

    A region that was not tested is not a test that failed, so it neither
    receives a q-value nor enlarges anyone else's denominator.
    """

    pvalues = np.asarray(pvalues, dtype=float)
    result = np.full(pvalues.size, np.nan, dtype=float)
    rows = np.flatnonzero(np.isfinite(pvalues))
    if rows.size == 0:
        return result
    values = pvalues[rows]
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    n = ranked.size
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    result[rows] = out
    return result



# --------------------------------------------------------------------------
# run-level annotation of the candidate family
# --------------------------------------------------------------------------


def _missing_columns(n: int, status: str) -> dict:
    return {
        "axis_status": np.full(n, status, dtype=object),
        "axis_n_axes": pd.array([pd.NA] * n, dtype="Int64"),
        "axis_n_samples_tested": pd.array([pd.NA] * n, dtype="Int64"),
        "axis_index": pd.array([pd.NA] * n, dtype="Int64"),
        "axis_effect": np.full(n, np.nan),
        "axis_effect_se": np.full(n, np.nan),
        "axis_effect_lower": np.full(n, np.nan),
        "axis_effect_upper": np.full(n, np.nan),
        "axis_pvalue": np.full(n, np.nan),
        "axis_qvalue": np.full(n, np.nan),
        "axis_supported": pd.array([pd.NA] * n, dtype="boolean"),
    }


def _base_manifest(mode: str, n: int) -> dict:
    experimental = mode == "experimental_inference"
    return {
        "mode": mode,
        "inference_default": AXIS_INFERENCE_DEFAULT,
        "inference_reported": experimental,
        "inference_label": (
            "EXPERIMENTAL: axis_pvalue, axis_qvalue and axis_supported were "
            "checked only in simulations with one retained axis and no missing "
            "values; they are not an established error rate for other data"
            if experimental else
            "not reported: axis_pvalue, axis_qvalue and axis_supported are "
            "missing; axis inference is not validated for several retained "
            "axes or missing values"
        ),
        "family": {
            "definition": (
                "every pre-FDR candidate region of this run's observed scan; "
                "axes estimated once with all candidate rows masked; BH (in the "
                "experimental mode) across this family, never across the "
                "coordination-selected subset"
            ),
            "n_regions": int(n),
        },
        "affects_primary_calls": False,
        "delta_min": DELTA_MIN,
        "interval_alpha": AXIS_INTERVAL_ALPHA,
        "inference_floor_tested_samples": AXIS_MIN_SAMPLES_FOR_INFERENCE,
        "status_meanings": dict(AXIS_STATUSES),
    }


def empty_axis_columns(n: int, *, mode: str) -> tuple[dict, dict]:
    """Axis columns for a run that computes no axes (switched off, or empty)."""

    status = "not_computed" if mode == "off" else "no_axes"
    manifest = _base_manifest(mode, n)
    manifest["n_axes"] = None
    manifest["status_counts"] = {status: int(n)} if n else {}
    if mode != "off":
        manifest["note"] = "no candidate regions, so no axes were estimated"
    return _missing_columns(n, status), manifest


def annotate_axes(
    betas: np.ndarray,
    regions: np.ndarray,
    *,
    mode: str = "descriptive",
    random_seed: int = 1,
    target_fdr: float = 0.05,
    axis_config: Eiger2Config | None = None,
) -> tuple[dict, dict]:
    """Axis columns for every candidate of one run, plus their manifest block.

    ``betas`` is the analysed ``(n_cpg, n_samples)`` matrix (sorted, filtered,
    NaN for missing); ``regions`` the ``(n, 2)`` inclusive row bounds of the
    run's candidates -- the whole pre-FDR family. ``mode`` is
    ``"descriptive"`` (default) or ``"experimental_inference"``. Returns a
    column dict aligned with ``regions`` (keys :data:`AXIS_COLUMNS`) and a
    JSON-ready manifest block. Axis estimation uses its own seeded generator
    (``axis_config.random_seed``, parallel analysis); it consumes no draw from
    the permutation generator.

    Memory: axis estimation holds one float64 centred copy of the matrix
    (zero-filled in place for the loading-concentration verdict).
    """

    from .coordination import _plain

    if mode not in {"descriptive", "experimental_inference"}:
        raise ValueError("mode must be 'descriptive' or 'experimental_inference'")
    regions = np.asarray(regions, dtype=int).reshape(-1, 2)
    n = len(regions)
    matrix = np.asarray(betas)
    experimental = mode == "experimental_inference"

    candidate_rows = np.zeros(matrix.shape[0], dtype=bool)
    for first, last in regions:
        candidate_rows[int(first) : int(last) + 1] = True
    config = axis_config or Eiger2Config(random_seed=random_seed)
    # Centring is row-local, so it is done in row chunks into one float64
    # array: a float32 matrix is never converted whole, and latent_axes may
    # zero-fill this private copy in place.
    residuals = np.empty(matrix.shape, dtype=float)
    for start in range(0, matrix.shape[0], 100_000):
        stop = min(start + 100_000, matrix.shape[0])
        residuals[start:stop] = centre_cpgs(matrix[start:stop])
    fitted = latent_axes(
        residuals, config, row_mask=~candidate_rows, allow_relaxation=False,
        overwrite_residuals=True,
    )
    del residuals
    kept = fitted.kept()
    axes = orient_axes(kept) if kept.size else np.empty((0, matrix.shape[1]))
    n_axes = int(axes.shape[0])
    low, high = axis_reference_points(axes) if n_axes else (None, None)

    columns = _missing_columns(n, "no_axes")
    tested_counts = np.zeros(n, dtype=np.int64)
    raw_pvalue = np.full(n, np.nan)
    axis_index = np.full(n, -1, dtype=np.int64)
    status = columns["axis_status"]
    for position, (first, last) in enumerate(regions):
        first, last = int(first), int(last)
        # A row slice, so a float32 matrix is never converted whole per region.
        values, tested = region_sample_means(matrix[first : last + 1], 0, last - first)
        tested_counts[position] = int(values.size)
        if n_axes == 0:
            continue
        if values.size < max(MIN_CONFIRM_SAMPLES, n_axes + 10):
            status[position] = "insufficient_samples"
            continue
        margin = axis_margin_test(values, axes[:, tested], low, high)
        if margin is None:
            status[position] = "fit_failed"
            continue
        status[position] = "estimated"
        axis_index[position] = int(margin["axis_index"])
        columns["axis_effect"][position] = margin["axis_effect"]
        columns["axis_effect_se"][position] = margin["axis_effect_se"]
        columns["axis_effect_lower"][position] = margin["axis_effect_lower"]
        columns["axis_effect_upper"][position] = margin["axis_effect_upper"]
        raw_pvalue[position] = margin["axis_pvalue"]

    columns["axis_n_axes"] = pd.array([n_axes] * n, dtype="Int64")
    columns["axis_n_samples_tested"] = pd.array(tested_counts, dtype="Int64")
    columns["axis_index"] = pd.array(
        [int(i) if i >= 0 else pd.NA for i in axis_index], dtype="Int64"
    )

    manifest = _base_manifest(mode, n)
    n_tested = 0
    n_supported = None
    if experimental:
        fitted_ok = np.isfinite(raw_pvalue)
        above_floor = tested_counts >= AXIS_MIN_SAMPLES_FOR_INFERENCE
        eligible = fitted_ok & above_floor
        status[fitted_ok & ~above_floor] = "below_inference_floor"
        status[eligible] = "tested"
        pvalue = np.where(eligible, raw_pvalue, np.nan)
        qvalue = benjamini_hochberg(pvalue)
        columns["axis_pvalue"] = pvalue
        columns["axis_qvalue"] = qvalue
        columns["axis_supported"] = pd.array(
            [bool(q <= target_fdr) if np.isfinite(q) else pd.NA for q in qvalue],
            dtype="boolean",
        )
        n_tested = int(eligible.sum())
        n_supported = int(np.sum(np.isfinite(qvalue) & (qvalue <= target_fdr)))
        manifest["axis_target_fdr"] = float(target_fdr)
    manifest.update({
        "n_axes": n_axes,
        "axis_estimation": {
            "method": (
                "per-CpG centred residuals, thinned, candidate rows masked, "
                "complete rows only (no relaxation); parallel analysis for K; "
                "biological_kept components only; label-free sign orientation"
            ),
            "random_seed": int(config.random_seed),
            "n_candidate_components": int(fitted.candidates.shape[0]),
            "kept_component_indices": [int(i) for i in fitted.kept_indices()],
            "diagnostics": _plain(fitted.diagnostics),
        },
        "reference_points": {
            "p10": [float(v) for v in low] if n_axes else [],
            "p90": [float(v) for v in high] if n_axes else [],
        },
        "status_counts": {
            str(k): int(v) for k, v in pd.Series(status).value_counts().items()
        },
        "n_inference_tested": n_tested,
        "n_supported": n_supported,
    })
    return columns, manifest
