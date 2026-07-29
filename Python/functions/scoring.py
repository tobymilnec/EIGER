"""Transparent scoring helpers shared by EIGER calling and annotation."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


WEIGHT_METHODS = ("none", "gaussian", "logistic", "prevalence_variance")


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values.

    Non-finite input values remain ``NaN`` and are excluded from the number of
    tested hypotheses.
    """

    p = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return adjusted

    pv = np.clip(p[valid], 0.0, 1.0)
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    m = ranked.size
    raw = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    restored = np.empty_like(monotone)
    restored[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[valid] = restored
    return adjusted


def descriptive_score(delta: float, stdev: float, p_value: float) -> float:
    """Combine standardised effect and descriptive evidence for ranking."""

    if not np.isfinite(delta) or not np.isfinite(stdev) or stdev <= 0:
        return 0.0
    if not np.isfinite(p_value):
        return 0.0
    p = float(np.clip(p_value, np.finfo(float).tiny, 1.0))
    return float(abs(delta) / stdev * -math.log10(p))


def gaussian_weight(p_obs: float, p_target: float, sigma: float = 0.05) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(np.exp(-((p_obs - p_target) ** 2) / (2.0 * sigma**2)))


def logistic_weight(p_obs: float, p_target: float, tau: float = 0.02) -> float:
    if tau <= 0:
        raise ValueError("tau must be positive")
    return float(1.0 / (1.0 + np.exp((p_obs - p_target) / tau)))


def prevalence_variance_weight(
    p_obs: float,
    p_target: float,
    *,
    epsilon: float = 1e-6,
    minimum: float = 0.25,
    maximum: float = 4.0,
) -> float:
    """Bounded Bernoulli prevalence-variance weight.

    The unclipped form is
    ``sqrt(p_target * (1-p_target) / (p_obs * (1-p_obs)))``.
    """

    if not 0 < minimum <= maximum:
        raise ValueError("weight bounds must satisfy 0 < minimum <= maximum")
    p = float(np.clip(p_obs, epsilon, 1.0 - epsilon))
    target = float(np.clip(p_target, epsilon, 1.0 - epsilon))
    raw = math.sqrt(target * (1.0 - target) / (p * (1.0 - p)))
    return float(np.clip(raw, minimum, maximum))


def score_weight(
    method: str,
    p_obs: float,
    p_target: float,
    *,
    sigma: float = 0.05,
    tau: float = 0.02,
    minimum: float = 0.25,
    maximum: float = 4.0,
) -> float:
    if method == "none":
        return 1.0
    if method == "gaussian":
        return gaussian_weight(p_obs, p_target, sigma)
    if method == "logistic":
        return logistic_weight(p_obs, p_target, tau)
    if method == "prevalence_variance":
        return prevalence_variance_weight(
            p_obs, p_target, minimum=minimum, maximum=maximum
        )
    raise ValueError(f"Unknown weighting method: {method!r}")


def prospective_power(
    n_target: int,
    n_reference: int,
    stdev: float,
    minimum_effect: float,
    *,
    alpha: float = 0.05,
) -> float:
    """Normal-approximation power for a pre-specified methylation effect."""

    if n_target < 1 or n_reference < 1 or not np.isfinite(stdev) or stdev <= 0:
        return 0.0
    if minimum_effect < 0 or not 0 < alpha < 1:
        raise ValueError("minimum_effect must be non-negative and alpha in (0, 1)")
    n_eff = n_target * n_reference / (n_target + n_reference)
    noncentrality = minimum_effect / stdev * math.sqrt(n_eff)
    critical = stats.norm.ppf(1.0 - alpha / 2.0)
    power = stats.norm.cdf(-critical - noncentrality)
    power += 1.0 - stats.norm.cdf(critical - noncentrality)
    return float(np.clip(power, 0.0, 1.0))


def fixed_strata(
    minor_group_size: int, n_cpg: int, missing_rate: float
) -> tuple[str, str, str]:
    """Return deterministic bins for non-parametric rank-within-strata."""

    if minor_group_size <= 1:
        group_bin = "n=1"
    elif minor_group_size == 2:
        group_bin = "n=2"
    elif minor_group_size <= 4:
        group_bin = "n=3-4"
    elif minor_group_size <= 9:
        group_bin = "n=5-9"
    else:
        group_bin = "n>=10"

    if n_cpg <= 5:
        cpg_bin = "cpg<=5"
    elif n_cpg <= 9:
        cpg_bin = "cpg=6-9"
    elif n_cpg <= 24:
        cpg_bin = "cpg=10-24"
    else:
        cpg_bin = "cpg>=25"

    if missing_rate < 0.05:
        missing_bin = "missing<0.05"
    elif missing_rate <= 0.20:
        missing_bin = "missing=0.05-0.20"
    else:
        missing_bin = "missing>0.20"
    return group_bin, cpg_bin, missing_bin


def add_rank_within_strata(
    frame: pd.DataFrame,
    *,
    score_column: str,
    strata_columns: Sequence[str],
    output_column: str = "rank_percentile",
) -> pd.DataFrame:
    """Rank a score within comparable strata, with higher values ranked best."""

    out = frame.copy()
    if out.empty:
        out[output_column] = pd.Series(dtype=float)
        return out
    out[output_column] = (
        out.groupby(list(strata_columns), dropna=False)[score_column]
        .rank(method="average", pct=True, ascending=True)
        .astype(float)
    )
    return out
