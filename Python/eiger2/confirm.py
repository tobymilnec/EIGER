"""Tier two: confirming a candidate region from its own per-sample means.

Tier one asks whether a run of CpGs co-varies more than this genome's own
edges do, and answers it with a rank. It cannot say how often such a run
arises on a DMR-free genome, because no self-calibrated background null
survived measurement (``docs/FDR.md`` section 18).

Tier two asks a different question, of the region alone: **do this region's
per-sample mean methylation values come from one population?** The null is a
property of the region's own values, not of the genome background, so it needs
no reference panel. Hartigan's dip test answers it, Benjamini-Hochberg across
a run's candidates controls the FDR against that hypothesis, and a
two-component fit supplies the effect size, prevalence and direction of
whatever separation is there.

The test and its calibration are documented, with the measurements that chose
them, in ``docs/CONFIRM.md``.

Pure functions: no I/O, no global state, nothing here reads a file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "BETA_CLAMP",
    "MIN_CONFIRM_SAMPLES",
    "benjamini_hochberg",
    "confirm_regions",
    "region_sample_means",
    "two_component_fit",
]

BETA_CLAMP = (0.005, 0.995)
"""Betas are clamped here before the logit, so a 0 or a 1 stays finite.

0.005 is below anything a real coverage can resolve at the depths WGBS runs
at, so the clamp moves no value that carries information.
"""

MIN_CONFIRM_SAMPLES = 10
"""Fewer tested samples than this and the region is not tested at all.

Hartigan's dip has no usable power on a handful of points, and a p-value from
one would be a number without a meaning. Such regions get ``NaN`` and
``pd.NA``, never a `False`: "not tested" is not "tested, not separated". Fixed
rather than configurable -- a run that lowers it does not get a more sensitive
test, only a less honest one.
"""

_MIN_OBSERVED_FRACTION = 0.5
"""A sample must observe at least half a region's CpGs to contribute a mean.

Below that the "region mean" is a mean of a different, shorter region.
"""


def region_sample_means(
    betas: np.ndarray, first: int, last: int
) -> tuple[np.ndarray, np.ndarray]:
    """Logit of each sample's mean beta over the region CpGs it observes.

    ``betas`` is the ``(n_cpg, n_samples)`` matrix ``call_dmrs`` works on;
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


def _dip_pvalue(values: np.ndarray) -> tuple[float, float]:
    """Hartigan's dip statistic and its interpolated-table p-value."""

    try:
        import diptest
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError(
            "tier-two confirmation needs the `diptest` package. Install eiger "
            "with its declared dependencies, or pass confirm_regions=False "
            "(--no-confirm) to report candidates without confirmation."
        ) from exc
    statistic, pvalue = diptest.diptest(np.asarray(values, dtype=float))
    return float(statistic), float(pvalue)


def _one_dimensional_em(
    values: np.ndarray, start: float, iterations: int = 200, tolerance: float = 1e-8
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Two-component one-dimensional Gaussian EM from one split point.

    Returns ``(loglikelihood, weights, means, variances)``-ish: the weights,
    the means and the shared-shape variances the run converged to. Deliberately
    tiny and dependency-free -- this is a descriptive fit for the effect size,
    not an inference.
    """

    n = values.size
    lower, upper = values <= start, values > start
    if not lower.any() or not upper.any():
        lower = np.arange(n) < max(1, n // 2)
        upper = ~lower
    weights = np.array([lower.mean(), upper.mean()], dtype=float)
    means = np.array([values[lower].mean(), values[upper].mean()], dtype=float)
    floor = 1e-6 * max(float(np.var(values)), 1e-12)
    variances = np.array(
        [max(float(np.var(values[lower])), floor),
         max(float(np.var(values[upper])), floor)],
        dtype=float,
    )
    previous = -np.inf
    loglikelihood = -np.inf
    for _ in range(iterations):
        density = (
            weights[None, :]
            * np.exp(
                -0.5 * (values[:, None] - means[None, :]) ** 2
                / variances[None, :]
            )
            / np.sqrt(2.0 * np.pi * variances[None, :])
        )
        total = density.sum(axis=1)
        if not np.all(np.isfinite(total)) or np.any(total <= 0):
            break
        loglikelihood = float(np.log(total).sum())
        responsibility = density / total[:, None]
        mass = responsibility.sum(axis=0)
        if np.any(mass <= 0):
            break
        weights = mass / n
        means = (responsibility * values[:, None]).sum(axis=0) / mass
        variances = np.maximum(
            (responsibility * (values[:, None] - means[None, :]) ** 2).sum(axis=0)
            / mass,
            floor,
        )
        if loglikelihood - previous <= tolerance * abs(loglikelihood):
            break
        previous = loglikelihood
    return loglikelihood, weights, means, variances


def two_component_fit(values: np.ndarray) -> dict:
    """Effect size of the best two-component split, on the beta scale.

    ``delta_hat`` is the difference of the component means mapped back through
    the logistic, ``f_hat`` the weight of the smaller component, ``direction``
    the sign of the smaller component's shift relative to the larger. Starts
    are the quartiles of the data, so the fit is deterministic: no seed, and
    the same input always gives the same answer.
    """

    values = np.asarray(values, dtype=float)
    if values.size < 4 or float(np.var(values)) <= 0:
        return {"delta_hat": np.nan, "f_hat": np.nan, "direction": 0}
    best = None
    for quantile in (0.2, 0.35, 0.5, 0.65, 0.8):
        candidate = _one_dimensional_em(values, float(np.quantile(values, quantile)))
        if best is None or candidate[0] > best[0]:
            best = candidate
    _loglikelihood, weights, means, _variances = best
    minor = int(np.argmin(weights))
    major = 1 - minor
    beta_means = 1.0 / (1.0 + np.exp(-means))
    delta = float(beta_means[minor] - beta_means[major])
    return {
        "delta_hat": delta,
        "f_hat": float(weights[minor]),
        "direction": int(np.sign(delta)),
    }


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


def confirm_regions(
    betas: np.ndarray, regions: np.ndarray, *, target_fdr: float = 0.05
) -> dict:
    """Confirm every region of one run, together.

    ``regions`` is the ``(n_regions, 2)`` array of inclusive CpG-row bounds
    that :func:`eiger2.core.segment` produces. BH runs across the tested
    regions of THIS run, which is the family the question is asked of.

    Returns arrays aligned with ``regions``: ``confirm_pvalue``,
    ``confirm_qvalue``, ``confirmed``, ``delta_hat``, ``f_hat``, ``direction``,
    ``n_samples_tested`` and the dip statistic itself.
    """

    regions = np.asarray(regions, dtype=int).reshape(-1, 2)
    n = len(regions)
    pvalues = np.full(n, np.nan, dtype=float)
    statistics = np.full(n, np.nan, dtype=float)
    delta_hat = np.full(n, np.nan, dtype=float)
    f_hat = np.full(n, np.nan, dtype=float)
    direction = np.full(n, np.nan, dtype=float)
    tested_counts = np.zeros(n, dtype=int)

    for position, (first, last) in enumerate(regions):
        values, tested = region_sample_means(betas, int(first), int(last))
        tested_counts[position] = int(values.size)
        if values.size < MIN_CONFIRM_SAMPLES:
            continue
        statistics[position], pvalues[position] = _dip_pvalue(values)
        fit = two_component_fit(values)
        delta_hat[position] = fit["delta_hat"]
        f_hat[position] = fit["f_hat"]
        direction[position] = fit["direction"]

    qvalues = benjamini_hochberg(pvalues)
    confirmed = np.array(
        [
            pd.NA if not np.isfinite(q) else bool(q <= target_fdr)
            for q in qvalues
        ],
        dtype=object,
    )
    return {
        "confirm_pvalue": pvalues,
        "confirm_qvalue": qvalues,
        "confirmed": confirmed,
        "confirm_statistic": statistics,
        "delta_hat": delta_hat,
        "f_hat": f_hat,
        "direction": direction,
        "n_samples_tested": tested_counts,
        "n_tested_regions": int(np.isfinite(pvalues).sum()),
    }
