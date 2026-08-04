"""Reusable EIGER matrix API with compact core and optional annotations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, ndtr
from sklearn.mixture import GaussianMixture

try:
    from .functions.processing import (
        compute_laplacian_score,
        covariance_blocks,
        covariance_standard_errors,
        covariance_standard_errors_with_measurement,
        extract_blocks,
    )
    from .functions.scoring import (
        WEIGHT_METHODS,
        add_rank_within_strata,
        benjamini_hochberg,
        descriptive_score,
        fixed_strata,
        prospective_power,
        score_weight,
    )
except ImportError:  # direct execution from the Python directory
    from functions.processing import (
        compute_laplacian_score,
        covariance_blocks,
        covariance_standard_errors,
        covariance_standard_errors_with_measurement,
        extract_blocks,
    )
    from functions.scoring import (
        WEIGHT_METHODS,
        add_rank_within_strata,
        benjamini_hochberg,
        descriptive_score,
        fixed_strata,
        prospective_power,
        score_weight,
    )


CORE_COLUMNS = [
    "dmr_id",
    "chr",
    "start",
    "end",
    "cpg_start",
    "cpg_end",
    "n_cpg",
    "length_bp",
    "n_samples",
    "n_samples_observed",
    "missing_rate",
    "mean_beta",
    "stdev",
    "covariance_score",
    "coverage_uncertainty_used",
    "mean_read_coverage",
    "segmentation_method",
    "edge_support_mean",
    "edge_support_min",
    "region_probability",
    "covariance_threshold_method",
    "covariance_threshold_used",
    "laplacian_score",
    "laplacian_rank_percentile",
    "stability_score",
    "delta_beta",
    "delta_minmax",
    "direction",
    "sample_cluster_size",
    "sample_cluster_fraction",
    "minor_group_size",
    "passes_delta_threshold",
    "sample_pvalue_descriptive",
    "sample_qvalue_descriptive",
    "sample_model_probability",
    "effect_probability",
    "call_probability",
    "sample_score_raw",
    "prevalence_weight",
    "power_support",
    "sample_score_weighted",
    "sample_rank_percentile",
    "call_status",
    "comparison",
    "group_annotation_status",
]

COMPACT_COLUMNS = [
    "dmr_id",
    "chr",
    "start",
    "end",
    "cpg_start",
    "cpg_end",
    "n_cpg",
    "length_bp",
    "mean_beta",
    "delta_beta",
    "direction",
    "sample_cluster_size",
    "sample_cluster_fraction",
    "passes_delta_threshold",
    "sample_pvalue_descriptive",
    "sample_qvalue_descriptive",
    "covariance_score",
    "covariance_threshold_method",
    "covariance_threshold_used",
    "stability_score",
    "region_probability",
    "call_probability",
    "sample_rank_percentile",
    "call_status",
    "comparison",
]

DIAGNOSTIC_COLUMNS = [
    "dmr_id",
    *[column for column in CORE_COLUMNS if column not in COMPACT_COLUMNS],
]

ANNOTATION_COLUMNS = [
    "dmr_id",
    "annotation_source",
    "grouping_variable",
    "target_groups",
    "reference_groups",
    "target_n",
    "reference_n",
    "minor_group_fraction",
    "target_mean_beta",
    "reference_mean_beta",
    "delta_beta",
    "direction",
    "delta_group_gap",
    "passes_delta_threshold",
    "annotation_status",
    "pvalue_descriptive",
    "qvalue_descriptive",
    "pvalue_empirical",
    "qvalue_empirical",
    "score_raw",
    "prevalence_weight",
    "power_support",
    "score_weighted",
    "rank_percentile",
    "group_size_stratum",
    "cpg_stratum",
    "missingness_stratum",
]


@dataclass(frozen=True)
class EigerConfig:
    min_cpgs: int = 4
    covariance_threshold: float = 0.012
    covariance_threshold_method: str = "fixed"
    covariance_null_quantile: float = 0.995
    covariance_null_permutations: int = 20
    covariance_null_max_pairs: int = 50000
    delta_threshold: float = 0.2
    target_group_size: int = 3
    prevalence_weight_method: str = "none"
    prevalence_weight_min: float = 0.25
    prevalence_weight_max: float = 4.0
    gaussian_sigma: float = 0.05
    logistic_tau: float = 0.02
    power_minimum_effect: float = 0.2
    power_alpha: float = 0.05
    laplacian_neighbors: int = 5
    min_observed_samples: int = 2
    pairwise_covariance: bool = False
    min_paired_samples: int = 5
    max_cpg_gap_bp: Optional[int] = 300
    annotation_permutations: int = 0
    random_seed: int = 1
    genome_build: str = "unknown"
    calling_method: str = "probabilistic"
    hmm_signal_prior: float = 0.05
    hmm_signal_persistence: float = 0.90
    hmm_posterior_threshold: float = 0.50
    sample_model_prior: float = 0.50
    sample_probability_threshold: float = 0.80
    min_sample_cluster_size: int = 2
    stability_bootstraps: int = 0
    stability_min_jaccard: float = 0.50

    def __post_init__(self):
        if self.min_cpgs < 2:
            raise ValueError("min_cpgs must be at least 2")
        if not np.isfinite(self.covariance_threshold):
            raise ValueError("covariance_threshold must be finite")
        if self.covariance_threshold_method not in {"fixed", "auto"}:
            raise ValueError(
                "covariance_threshold_method must be 'fixed' or 'auto'"
            )
        if not 0.5 < self.covariance_null_quantile < 1.0:
            raise ValueError("covariance_null_quantile must be between 0.5 and 1")
        if self.covariance_null_permutations < 1:
            raise ValueError("covariance_null_permutations must be positive")
        if self.covariance_null_max_pairs < 1:
            raise ValueError("covariance_null_max_pairs must be positive")
        if not 0 <= self.delta_threshold <= 1:
            raise ValueError("delta_threshold must be between 0 and 1")
        if self.target_group_size < 1:
            raise ValueError("target_group_size must be positive")
        if self.max_cpg_gap_bp is not None and self.max_cpg_gap_bp < 1:
            raise ValueError("max_cpg_gap_bp must be positive when provided")
        if self.prevalence_weight_method not in WEIGHT_METHODS:
            raise ValueError(
                "prevalence_weight_method must be one of "
                + ", ".join(WEIGHT_METHODS)
            )
        if self.calling_method not in {"legacy", "probabilistic"}:
            raise ValueError("calling_method must be 'legacy' or 'probabilistic'")
        for name, value in (
            ("hmm_signal_prior", self.hmm_signal_prior),
            ("hmm_signal_persistence", self.hmm_signal_persistence),
            ("hmm_posterior_threshold", self.hmm_posterior_threshold),
            ("sample_model_prior", self.sample_model_prior),
            ("sample_probability_threshold", self.sample_probability_threshold),
            ("stability_min_jaccard", self.stability_min_jaccard),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.min_sample_cluster_size < 1:
            raise ValueError("min_sample_cluster_size must be positive")
        if self.stability_bootstraps < 0:
            raise ValueError("stability_bootstraps cannot be negative")

    def to_dict(self) -> dict:
        return asdict(self)


def impute_for_covariance(betas: np.ndarray) -> np.ndarray:
    """Mean-impute within CpG for covariance calculations only."""

    array = np.asarray(betas)
    dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else float
    values = np.array(array, dtype=dtype, copy=True)
    with np.errstate(invalid="ignore"):
        row_means = np.nanmean(values, axis=1)
    global_mean = np.nanmean(values)
    if not np.isfinite(global_mean):
        global_mean = 0.5
    row_means = np.where(np.isfinite(row_means), row_means, global_mean)
    for sample_index in range(values.shape[1]):
        missing = ~np.isfinite(values[:, sample_index])
        values[missing, sample_index] = row_means[missing]
    return values


def beta_measurement_variance(
    betas: np.ndarray,
    total_counts: np.ndarray,
) -> np.ndarray:
    """Jeffreys-Beta posterior variance for observed methylation proportions."""

    values = np.asarray(betas, dtype=float)
    totals = np.asarray(total_counts, dtype=float)
    if values.shape != totals.shape:
        raise ValueError("betas and total_counts must have the same shape")
    if np.any(~np.isfinite(totals)) or np.any(totals < 0):
        raise ValueError("total_counts must be finite and non-negative")
    methylated = np.clip(values, 0.0, 1.0) * totals
    alpha = methylated + 0.5
    beta = totals - methylated + 0.5
    denominator = (alpha + beta) ** 2 * (alpha + beta + 1.0)
    variance = np.zeros(values.shape, dtype=float)
    observed = np.isfinite(values) & (totals > 0)
    variance[observed] = (alpha[observed] * beta[observed]) / denominator[observed]
    return variance


@dataclass(frozen=True)
class CovarianceThresholdEstimate:
    threshold: float
    null_pairs_sampled: int


def estimate_covariance_threshold(
    covariance_input: np.ndarray,
    eligible_pairs: np.ndarray,
    *,
    quantile: float = 0.995,
    permutations: int = 20,
    max_pairs: int = 50000,
    seed: int = 1,
) -> CovarianceThresholdEstimate:
    """Estimate an adjacent-covariance threshold from a label-free null.

    CpG means are removed, then each sample independently draws eligible
    adjacent genomic pairs. This preserves residual methylation variation and
    ordinary adjacent structure while breaking a coordinated regional signal.
    """

    matrix = np.asarray(covariance_input, dtype=float)
    eligible = np.asarray(eligible_pairs, dtype=bool).reshape(-1)
    if matrix.ndim != 2 or eligible.size != max(matrix.shape[0] - 1, 0):
        raise ValueError("eligible_pairs must align with covariance_input rows")
    if matrix.shape[1] < 2:
        raise ValueError("at least two samples are required")
    pair_indices = np.flatnonzero(eligible)
    if pair_indices.size == 0:
        raise ValueError("no eligible adjacent CpG pairs are available")

    rng = np.random.default_rng(seed)
    n_null_pairs = min(pair_indices.size, max_pairs)
    residuals = matrix - np.mean(matrix, axis=1, keepdims=True)
    sample_indices = np.arange(matrix.shape[1])
    denominator = matrix.shape[1] - 1
    null_covariance = np.empty(
        (permutations, n_null_pairs), dtype=float
    )
    for permutation_index in range(permutations):
        sampled_pairs = rng.choice(
            pair_indices,
            size=(n_null_pairs, matrix.shape[1]),
            replace=True,
        )
        left = residuals[sampled_pairs, sample_indices]
        right = residuals[sampled_pairs + 1, sample_indices]
        left -= np.mean(left, axis=1, keepdims=True)
        right -= np.mean(right, axis=1, keepdims=True)
        null_covariance[permutation_index] = np.sum(
            left * right, axis=1
        ) / denominator
    return CovarianceThresholdEstimate(
        threshold=float(np.quantile(null_covariance, quantile)),
        null_pairs_sampled=permutations * n_null_pairs,
    )


def estimate_covariance_threshold_from_pairs(
    left_residuals: np.ndarray,
    right_residuals: np.ndarray,
    *,
    quantile: float = 0.995,
    permutations: int = 20,
    max_pairs: int = 50000,
    seed: int = 1,
    batch_size: int = 5000,
) -> CovarianceThresholdEstimate:
    """Estimate the auto threshold from a uniform reservoir of adjacent pairs."""

    left = np.asarray(left_residuals)
    right = np.asarray(right_residuals)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("left and right residual pairs must have matching matrices")
    if left.shape[0] == 0 or left.shape[1] < 2:
        raise ValueError("at least one pair and two samples are required")

    rng = np.random.default_rng(seed)
    n_null_pairs = min(left.shape[0], max_pairs)
    n_samples = left.shape[1]
    sample_indices = np.arange(n_samples)
    denominator = n_samples - 1
    null_covariance = np.empty(permutations * n_null_pairs, dtype=float)
    output_offset = 0

    for _ in range(permutations):
        for batch_start in range(0, n_null_pairs, batch_size):
            batch_count = min(batch_size, n_null_pairs - batch_start)
            sampled_pairs = rng.integers(
                0,
                left.shape[0],
                size=(batch_count, n_samples),
            )
            sampled_left = left[sampled_pairs, sample_indices].astype(
                float, copy=False
            )
            sampled_right = right[sampled_pairs, sample_indices].astype(
                float, copy=False
            )
            sampled_left = sampled_left - np.mean(
                sampled_left, axis=1, keepdims=True
            )
            sampled_right = sampled_right - np.mean(
                sampled_right, axis=1, keepdims=True
            )
            null_covariance[
                output_offset : output_offset + batch_count
            ] = np.sum(sampled_left * sampled_right, axis=1) / denominator
            output_offset += batch_count

    return CovarianceThresholdEstimate(
        threshold=float(np.quantile(null_covariance, quantile)),
        null_pairs_sampled=int(null_covariance.size),
    )


def _standardise_locations(
    locations,
    n_rows: int,
    source_cpg_offset: int = 0,
) -> pd.DataFrame:
    loc = pd.DataFrame(locations).copy()
    if len(loc) != n_rows:
        raise ValueError("locations and beta matrix must have the same number of rows")

    lower = {str(c).lower(): c for c in loc.columns}
    chrom_col = next(
        (lower[k] for k in ("chr", "chrom", "chromosome") if k in lower),
        loc.columns[0],
    )
    start_col = lower.get("start", loc.columns[1] if loc.shape[1] > 1 else None)
    if start_col is None:
        raise ValueError("locations must contain chromosome and start columns")
    starts = pd.to_numeric(loc[start_col], errors="raise").astype("int64")
    end_col = lower.get("end")
    if end_col is None:
        ends = starts + 1
    else:
        ends = pd.to_numeric(loc[end_col], errors="raise").astype("int64")
        if (ends <= starts).any():
            raise ValueError(
                "location end coordinates must be greater than start coordinates"
            )
    return pd.DataFrame(
        {
            "chr": loc[chrom_col].astype(str),
            "start": starts,
            "end": ends,
            "source_cpg_index": (
                np.arange(n_rows, dtype=int) + int(source_cpg_offset)
            ),
        }
    )


def _safe_dmr_id(
    chrom: str, start: int, end: int, cpg_start: int, cpg_end: int
) -> str:
    safe_chrom = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(chrom))
    return f"EIGER_{safe_chrom}_{start}_{end}_{cpg_start}_{cpg_end}"


def _welch_pvalue(target: np.ndarray, reference: np.ndarray) -> float:
    if target.size < 2 or reference.size < 2:
        return 1.0
    if np.allclose(target, target[0]) and np.allclose(reference, reference[0]):
        return 1.0 if np.isclose(target[0], reference[0]) else 0.0
    result = stats.ttest_ind(target, reference, equal_var=False, nan_policy="omit")
    return float(result.pvalue) if np.isfinite(result.pvalue) else 1.0


def covariance_edge_support(
    covariance: np.ndarray,
    standard_error: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return approximate P(covariance > threshold) for each adjacent edge."""

    covariance = np.asarray(covariance, dtype=float)
    standard_error = np.asarray(standard_error, dtype=float)
    if covariance.shape != standard_error.shape:
        raise ValueError("covariance and standard_error must have the same shape")
    support = np.full(covariance.shape, 0.5, dtype=float)
    regular = np.isfinite(standard_error) & (standard_error > 0)
    support[regular] = ndtr(
        (covariance[regular] - threshold) / standard_error[regular]
    )
    degenerate = np.isfinite(covariance) & ~regular
    support[degenerate & (covariance > threshold)] = 1.0
    support[degenerate & (covariance < threshold)] = 0.0
    return np.clip(support, 1e-9, 1.0 - 1e-9)


def _smooth_support_segment(
    support: np.ndarray,
    *,
    signal_prior: float,
    signal_persistence: float,
) -> np.ndarray:
    """Two-state forward/backward smoothing of edge-level support."""

    observed = np.clip(np.asarray(support, dtype=float), 1e-9, 1.0 - 1e-9)
    if observed.size == 0:
        return observed.copy()
    transition_11 = signal_persistence
    transition_10 = 1.0 - transition_11
    transition_01 = min(
        signal_prior * transition_10 / (1.0 - signal_prior),
        1.0 - 1e-9,
    )
    transition_00 = 1.0 - transition_01

    filtered = np.empty(observed.size, dtype=float)
    predicted_signal = signal_prior
    for index, emission_signal in enumerate(observed):
        numerator = predicted_signal * emission_signal
        denominator = numerator + (1.0 - predicted_signal) * (
            1.0 - emission_signal
        )
        filtered[index] = numerator / denominator if denominator else signal_prior
        predicted_signal = (
            filtered[index] * transition_11
            + (1.0 - filtered[index]) * transition_01
        )

    posterior = np.empty(observed.size, dtype=float)
    backward_0 = 1.0
    backward_1 = 1.0
    for index in range(observed.size - 1, -1, -1):
        alpha_1 = filtered[index]
        denominator = (1.0 - alpha_1) * backward_0 + alpha_1 * backward_1
        posterior[index] = (
            alpha_1 * backward_1 / denominator if denominator else alpha_1
        )
        if index:
            next_signal = observed[index]
            next_background = 1.0 - next_signal
            new_backward_0 = (
                transition_00 * next_background * backward_0
                + transition_01 * next_signal * backward_1
            )
            new_backward_1 = (
                transition_10 * next_background * backward_0
                + transition_11 * next_signal * backward_1
            )
            scale = new_backward_0 + new_backward_1
            if scale:
                new_backward_0 /= scale
                new_backward_1 /= scale
            backward_0, backward_1 = new_backward_0, new_backward_1
    return np.clip(posterior, 0.0, 1.0)


def smooth_edge_support(
    support: np.ndarray,
    eligible_pairs: np.ndarray,
    *,
    signal_prior: float = 0.05,
    signal_persistence: float = 0.90,
) -> np.ndarray:
    """Smooth eligible edge support without crossing genomic discontinuities."""

    support = np.asarray(support, dtype=float)
    eligible = np.asarray(eligible_pairs, dtype=bool)
    if support.shape != eligible.shape:
        raise ValueError("support and eligible_pairs must have the same shape")
    posterior = np.zeros(support.shape, dtype=float)
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return posterior
    split_points = np.flatnonzero(np.diff(indices) != 1) + 1
    for segment in np.split(indices, split_points):
        posterior[segment] = _smooth_support_segment(
            support[segment],
            signal_prior=signal_prior,
            signal_persistence=signal_persistence,
        )
    return posterior


def _effect_probability(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: float,
    target_measurement_variance: Optional[np.ndarray] = None,
    reference_measurement_variance: Optional[np.ndarray] = None,
) -> float:
    """Approximate posterior probability of a biologically meaningful effect."""

    if target.size < 2 or reference.size < 2:
        return 0.0
    delta = float(np.mean(target) - np.mean(reference))
    if target_measurement_variance is None:
        target_variance = np.var(target, ddof=1) / target.size
        reference_variance = np.var(reference, ddof=1) / reference.size
    else:
        target_measurement_variance = np.asarray(
            target_measurement_variance, dtype=float
        )
        reference_measurement_variance = np.asarray(
            reference_measurement_variance, dtype=float
        )
        target_biological = max(
            np.var(target, ddof=1) - np.mean(target_measurement_variance),
            0.0,
        )
        reference_biological = max(
            np.var(reference, ddof=1) - np.mean(reference_measurement_variance),
            0.0,
        )
        target_variance = target_biological / target.size
        target_variance += np.sum(target_measurement_variance) / target.size**2
        reference_variance = reference_biological / reference.size
        reference_variance += (
            np.sum(reference_measurement_variance) / reference.size**2
        )
    standard_error = np.sqrt(target_variance + reference_variance)
    if not np.isfinite(standard_error) or standard_error <= 0:
        return float(abs(delta) >= threshold)
    probability = ndtr((-threshold - delta) / standard_error)
    probability += 1.0 - ndtr((threshold - delta) / standard_error)
    return float(np.clip(probability, 0.0, 1.0))


def _separation_from_values(values: np.ndarray, config: EigerConfig) -> dict:
    observed_indices = np.flatnonzero(np.isfinite(values))
    observed = values[observed_indices]
    if observed.size < 2:
        return {
            "target_indices": np.array([], dtype=int),
            "reference_indices": np.array([], dtype=int),
            "target_n": 0,
            "reference_n": 0,
            "minor_n": 0,
            "minor_fraction": np.nan,
            "target_mean": np.nan,
            "reference_mean": np.nan,
            "delta": np.nan,
            "gap": 0.0,
            "direction": "Uncertain",
            "passes": False,
            "call_supported": False,
            "pvalue": np.nan,
            "score_raw": 0.0,
            "weight": 1.0,
            "power": 0.0,
            "score_weighted": 0.0,
            "model_probability": np.nan,
            "effect_probability": np.nan,
            "call_probability": np.nan,
        }

    order = np.argsort(observed, kind="mergesort")
    sorted_values = observed[order]
    gaps = np.diff(sorted_values)
    split = int(np.argmax(gaps))
    gap = float(gaps[split])
    lower = observed_indices[order[: split + 1]]
    upper = observed_indices[order[split + 1 :]]

    if len(upper) < len(lower):
        target_indices, reference_indices = upper, lower
    else:
        target_indices, reference_indices = lower, upper

    target = values[target_indices]
    reference = values[reference_indices]
    target_mean = float(np.nanmean(target))
    reference_mean = float(np.nanmean(reference))
    delta = target_mean - reference_mean
    stdev = float(np.nanstd(observed))
    pvalue = _welch_pvalue(target, reference)
    score_raw = descriptive_score(delta, stdev, pvalue)
    minor_n = min(len(target), len(reference))
    minor_fraction = minor_n / observed.size
    target_prevalence = min(config.target_group_size / observed.size, 0.5)
    weight = score_weight(
        config.prevalence_weight_method,
        minor_fraction,
        target_prevalence,
        sigma=config.gaussian_sigma,
        tau=config.logistic_tau,
        minimum=config.prevalence_weight_min,
        maximum=config.prevalence_weight_max,
    )
    power = prospective_power(
        len(target),
        len(reference),
        stdev,
        config.power_minimum_effect,
        alpha=config.power_alpha,
    )
    return {
        "target_indices": target_indices,
        "reference_indices": reference_indices,
        "target_n": len(target),
        "reference_n": len(reference),
        "minor_n": minor_n,
        "minor_fraction": minor_fraction,
        "target_mean": target_mean,
        "reference_mean": reference_mean,
        "delta": delta,
        "gap": gap,
        "direction": "Hypomethylation" if delta < 0 else "Hypermethylation",
        "passes": abs(delta) >= config.delta_threshold,
        "call_supported": abs(delta) >= config.delta_threshold,
        "pvalue": pvalue,
        "score_raw": score_raw,
        "weight": weight,
        "power": power,
        "score_weighted": score_raw * weight,
        "model_probability": np.nan,
        "effect_probability": np.nan,
        "call_probability": np.nan,
    }


def _probabilistic_separation_from_values(
    values: np.ndarray,
    config: EigerConfig,
    measurement_variance: Optional[np.ndarray] = None,
) -> dict:
    """Infer a two-component sample partition with BIC posterior support."""

    if measurement_variance is not None:
        measurement_variance = np.asarray(measurement_variance, dtype=float)
        if measurement_variance.shape != values.shape:
            raise ValueError("measurement_variance must align with values")
    observed_indices = np.flatnonzero(np.isfinite(values))
    observed = np.asarray(values[observed_indices], dtype=float)
    if observed.size < max(4, 2 * config.min_sample_cluster_size):
        result = _separation_from_values(values, config)
        result.update(
            passes=False,
            call_supported=False,
            model_probability=0.0,
            effect_probability=0.0,
            call_probability=0.0,
        )
        return result

    clipped = np.clip(observed, 1e-4, 1.0 - 1e-4)
    model_values = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    try:
        one_component = GaussianMixture(
            n_components=1,
            covariance_type="full",
            reg_covar=1e-5,
            n_init=3,
            random_state=config.random_seed,
        ).fit(model_values)
        two_component = GaussianMixture(
            n_components=2,
            covariance_type="full",
            reg_covar=1e-5,
            n_init=10,
            random_state=config.random_seed,
        ).fit(model_values)
        labels = two_component.predict(model_values)
        bic_difference = one_component.bic(model_values) - two_component.bic(
            model_values
        )
        prior_log_odds = np.log(config.sample_model_prior) - np.log1p(
            -config.sample_model_prior
        )
        model_probability = float(expit(0.5 * bic_difference + prior_log_odds))
    except (ValueError, FloatingPointError):
        result = _separation_from_values(values, config)
        result.update(
            passes=False,
            call_supported=False,
            model_probability=0.0,
            effect_probability=0.0,
            call_probability=0.0,
        )
        return result

    label_zero = observed_indices[labels == 0]
    label_one = observed_indices[labels == 1]
    if len(label_one) < len(label_zero):
        target_indices, reference_indices = label_one, label_zero
    else:
        target_indices, reference_indices = label_zero, label_one
    target = values[target_indices]
    reference = values[reference_indices]
    target_mean = float(np.mean(target))
    reference_mean = float(np.mean(reference))
    delta = target_mean - reference_mean
    order = np.argsort(observed, kind="mergesort")
    gaps = np.diff(observed[order])
    gap = float(np.max(gaps)) if len(gaps) else 0.0
    stdev = float(np.std(observed))
    pvalue = _welch_pvalue(target, reference)
    score_raw = descriptive_score(delta, stdev, pvalue)
    minor_n = min(len(target), len(reference))
    minor_fraction = minor_n / observed.size
    target_prevalence = min(config.target_group_size / observed.size, 0.5)
    weight = score_weight(
        config.prevalence_weight_method,
        minor_fraction,
        target_prevalence,
        sigma=config.gaussian_sigma,
        tau=config.logistic_tau,
        minimum=config.prevalence_weight_min,
        maximum=config.prevalence_weight_max,
    )
    power = prospective_power(
        len(target),
        len(reference),
        stdev,
        config.power_minimum_effect,
        alpha=config.power_alpha,
    )
    effect_probability = _effect_probability(
        target,
        reference,
        config.delta_threshold,
        None if measurement_variance is None else measurement_variance[target_indices],
        None
        if measurement_variance is None
        else measurement_variance[reference_indices],
    )
    call_probability = model_probability * effect_probability
    sufficiently_large = minor_n >= config.min_sample_cluster_size
    passes_delta = abs(delta) >= config.delta_threshold
    call_supported = (
        sufficiently_large
        and passes_delta
        and call_probability >= config.sample_probability_threshold
    )
    return {
        "target_indices": target_indices,
        "reference_indices": reference_indices,
        "target_n": len(target),
        "reference_n": len(reference),
        "minor_n": minor_n,
        "minor_fraction": minor_fraction,
        "target_mean": target_mean,
        "reference_mean": reference_mean,
        "delta": delta,
        "gap": gap,
        "direction": "Hypomethylation" if delta < 0 else "Hypermethylation",
        "passes": passes_delta,
        "call_supported": call_supported,
        "pvalue": pvalue,
        "score_raw": score_raw,
        "weight": weight,
        "power": power,
        "score_weighted": score_raw * weight,
        "model_probability": model_probability,
        "effect_probability": effect_probability,
        "call_probability": call_probability,
    }


def infer_sample_partition(
    values: Iterable[float], config: Optional[EigerConfig] = None
) -> np.ndarray:
    """Return binary inferred sample membership for one DMR.

    ``1`` marks the inferred target/minor side, ``0`` the reference side, and
    ``-1`` a missing sample. Cluster labels are intentionally label-invariant
    for downstream ARI/NMI benchmarking.
    """

    array = np.asarray(list(values), dtype=float)
    resolved_config = config or EigerConfig()
    result = (
        _separation_from_values(array, resolved_config)
        if resolved_config.calling_method == "legacy"
        else _probabilistic_separation_from_values(array, resolved_config)
    )
    labels = np.full(array.shape, -1, dtype=int)
    labels[result["reference_indices"]] = 0
    labels[result["target_indices"]] = 1
    return labels


def _empty_outputs(
    sample_names: Sequence[str],
    *,
    covariance_threshold_used: float,
    covariance_threshold_method: str,
    covariance_threshold_components: Optional[dict] = None,
):
    core = pd.DataFrame(columns=CORE_COLUMNS)
    core.attrs["covariance_threshold_used"] = covariance_threshold_used
    core.attrs["covariance_threshold_method"] = covariance_threshold_method
    core.attrs["covariance_threshold_components"] = (
        covariance_threshold_components or {}
    )
    values = pd.DataFrame(columns=["dmr_id", *sample_names])
    annotations = pd.DataFrame(columns=ANNOTATION_COLUMNS)
    return core, values, annotations


def call_dmrs_from_matrix(
    betas,
    locations,
    *,
    total_counts=None,
    sample_names: Optional[Sequence[str]] = None,
    metadata: Optional[pd.DataFrame] = None,
    sample_column: Optional[str] = None,
    group_columns: Optional[Sequence[str]] = None,
    config: Optional[EigerConfig] = None,
    source_cpg_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Call EIGER DMRs from a CpG-by-sample methylation matrix."""

    config = config or EigerConfig()
    matrix = np.asarray(betas)
    if not np.issubdtype(matrix.dtype, np.floating):
        matrix = matrix.astype(float)
    if matrix.ndim != 2:
        raise ValueError("betas must be a two-dimensional CpG-by-sample matrix")
    coverage = None
    if total_counts is not None:
        coverage = np.asarray(total_counts)
        if coverage.shape != matrix.shape:
            raise ValueError("total_counts must match the beta matrix shape")
        if np.any(~np.isfinite(coverage)) or np.any(coverage < 0):
            raise ValueError("total_counts must be finite and non-negative")
    n_cpg_total, n_samples = matrix.shape
    if n_samples < 2:
        raise ValueError("EIGER requires at least two samples")
    if sample_names is None:
        sample_names = [f"sample_{i + 1}" for i in range(n_samples)]
    sample_names = [str(s) for s in sample_names]
    if len(sample_names) != n_samples or len(set(sample_names)) != n_samples:
        raise ValueError("sample_names must be unique and match the matrix columns")

    loc = _standardise_locations(
        locations,
        n_cpg_total,
        source_cpg_offset=source_cpg_offset,
    )
    valid = np.sum(np.isfinite(matrix), axis=1) >= config.min_observed_samples
    if np.sum(valid) < config.min_cpgs:
        return _empty_outputs(
            sample_names,
            covariance_threshold_used=config.covariance_threshold,
            covariance_threshold_method=config.covariance_threshold_method,
        )

    valid_matrix = matrix[valid]
    valid_coverage = coverage[valid] if coverage is not None else None
    valid_locations = loc.loc[valid].reset_index(drop=True)
    covariance_input = impute_for_covariance(valid_matrix)
    measurement_variance = (
        beta_measurement_variance(valid_matrix, valid_coverage)
        if valid_coverage is not None
        else None
    )
    # Pairwise-complete covariance uses the matrix WITH its missing values so
    # that unobserved samples are dropped from both the cross-product and the
    # denominator. The imputed copy is still used for the null, the Laplacian and
    # the standard errors, which need a complete matrix.
    adjacent_covariance = np.asarray(
        covariance_blocks(
            valid_matrix if config.pairwise_covariance else covariance_input,
            pairwise=config.pairwise_covariance,
            min_paired_samples=config.min_paired_samples,
        ),
        dtype=float,
    )
    if config.pairwise_covariance:
        adjacent_covariance = np.where(
            np.isfinite(adjacent_covariance), adjacent_covariance, -np.inf
        )
    eligible_pairs = (
        valid_locations["chr"].iloc[:-1].to_numpy()
        == valid_locations["chr"].iloc[1:].to_numpy()
    )
    if config.max_cpg_gap_bp is None:
        eligible_pairs &= (
            np.diff(valid_locations["source_cpg_index"].to_numpy()) == 1
        )
    else:
        gaps = (
            valid_locations["start"].iloc[1:].to_numpy()
            - valid_locations["end"].iloc[:-1].to_numpy()
        )
        eligible_pairs &= (gaps >= 0) & (gaps <= config.max_cpg_gap_bp)
    covariance_threshold_used = config.covariance_threshold
    covariance_threshold_components = {}
    if config.covariance_threshold_method == "auto":
        if np.any(eligible_pairs):
            estimate = estimate_covariance_threshold(
                covariance_input,
                eligible_pairs,
                quantile=config.covariance_null_quantile,
                permutations=config.covariance_null_permutations,
                max_pairs=config.covariance_null_max_pairs,
                seed=config.random_seed,
            )
            covariance_threshold_used = estimate.threshold
            covariance_threshold_components = {
                "resampled_null_threshold": estimate.threshold,
                "null_pairs_sampled": estimate.null_pairs_sampled,
            }
    if config.calling_method == "legacy":
        edge_support = (adjacent_covariance >= covariance_threshold_used).astype(
            float
        )
        edge_posterior = edge_support
        segmentation_values = adjacent_covariance
        segmentation_threshold = covariance_threshold_used
    else:
        covariance_se = np.asarray(
            covariance_standard_errors(covariance_input)
            if measurement_variance is None
            else covariance_standard_errors_with_measurement(
                covariance_input, measurement_variance
            ),
            dtype=float,
        )
        edge_support = covariance_edge_support(
            adjacent_covariance,
            covariance_se,
            covariance_threshold_used,
        )
        edge_posterior = smooth_edge_support(
            edge_support,
            eligible_pairs,
            signal_prior=config.hmm_signal_prior,
            signal_persistence=config.hmm_signal_persistence,
        )
        segmentation_values = edge_posterior
        segmentation_threshold = config.hmm_posterior_threshold
    adjacent_covariance[~eligible_pairs] = -np.inf
    edge_support[~eligible_pairs] = 0.0
    edge_posterior[~eligible_pairs] = 0.0
    segmentation_values[~eligible_pairs] = -np.inf
    blocks = extract_blocks(
        segmentation_values,
        DMR_length_threshold=config.min_cpgs,
        cov_threshold=segmentation_threshold,
    )
    if blocks.size == 0:
        return _empty_outputs(
            sample_names,
            covariance_threshold_used=covariance_threshold_used,
            covariance_threshold_method=config.covariance_threshold_method,
            covariance_threshold_components=covariance_threshold_components,
        )

    block_values = []
    block_missing = []
    block_measurement_variance = []
    for first, last in blocks:
        region = valid_matrix[int(first) : int(last) + 1]
        observed_counts = np.sum(np.isfinite(region), axis=0)
        sample_means = np.full(region.shape[1], np.nan, dtype=float)
        np.divide(
            np.nansum(region, axis=0),
            observed_counts,
            out=sample_means,
            where=observed_counts > 0,
        )
        block_values.append(sample_means)
        block_missing.append(float(np.mean(~np.isfinite(region))))
        if measurement_variance is not None:
            region_variance = measurement_variance[int(first) : int(last) + 1]
            mean_variance = np.zeros(region.shape[1], dtype=float)
            np.divide(
                np.sum(region_variance, axis=0),
                observed_counts**2,
                out=mean_variance,
                where=observed_counts > 0,
            )
            block_measurement_variance.append(mean_variance)
    block_values_matrix = np.asarray(block_values, dtype=float)
    laplacian_input = impute_for_covariance(block_values_matrix)
    laplacian = compute_laplacian_score(
        laplacian_input.T, neighbor_size=config.laplacian_neighbors
    )

    rows = []
    for block_index, (first, last) in enumerate(blocks):
        first = int(first)
        last = int(last)
        first_loc = valid_locations.iloc[first]
        last_loc = valid_locations.iloc[last]
        cpg_start = int(first_loc["source_cpg_index"])
        cpg_end = int(last_loc["source_cpg_index"])
        start = int(first_loc["start"])
        end = int(last_loc["end"])
        values = block_values_matrix[block_index]
        separation = (
            _separation_from_values(values, config)
            if config.calling_method == "legacy"
            else _probabilistic_separation_from_values(
                values,
                config,
                None
                if measurement_variance is None
                else block_measurement_variance[block_index],
            )
        )
        n_observed = int(np.sum(np.isfinite(values)))
        n_cpg = last - first + 1
        covariance_score = float(np.nanmean(adjacent_covariance[first:last]))
        region_edge_support = edge_support[first:last]
        region_edge_posterior = edge_posterior[first:last]
        rows.append(
            {
                "dmr_id": _safe_dmr_id(
                    first_loc["chr"], start, end, cpg_start, cpg_end
                ),
                "chr": str(first_loc["chr"]),
                "start": start,
                "end": end,
                "cpg_start": cpg_start,
                "cpg_end": cpg_end,
                "n_cpg": n_cpg,
                "length_bp": max(0, end - start),
                "n_samples": n_samples,
                "n_samples_observed": n_observed,
                "missing_rate": block_missing[block_index],
                "mean_beta": float(np.nanmean(values)),
                "stdev": float(np.nanstd(values)),
                "covariance_score": covariance_score,
                "coverage_uncertainty_used": measurement_variance is not None,
                "mean_read_coverage": (
                    float(
                        np.mean(
                            valid_coverage[first : last + 1][
                                valid_coverage[first : last + 1] > 0
                            ]
                        )
                    )
                    if valid_coverage is not None
                    and np.any(valid_coverage[first : last + 1] > 0)
                    else np.nan
                ),
                "segmentation_method": config.calling_method,
                "edge_support_mean": float(np.mean(region_edge_support)),
                "edge_support_min": float(np.min(region_edge_support)),
                "region_probability": float(np.mean(region_edge_posterior)),
                "covariance_threshold_method": (
                    config.covariance_threshold_method
                ),
                "covariance_threshold_used": covariance_threshold_used,
                "laplacian_score": float(laplacian[block_index]),
                "delta_beta": separation["delta"],
                "delta_minmax": separation["gap"],
                "direction": separation["direction"],
                "sample_cluster_size": separation["target_n"],
                "sample_cluster_fraction": separation["target_n"] / n_observed
                if n_observed
                else np.nan,
                "minor_group_size": separation["minor_n"],
                "passes_delta_threshold": separation["passes"],
                "sample_pvalue_descriptive": separation["pvalue"],
                "sample_model_probability": separation["model_probability"],
                "effect_probability": separation["effect_probability"],
                "call_probability": separation["call_probability"],
                "sample_score_raw": separation["score_raw"],
                "prevalence_weight": separation["weight"],
                "power_support": separation["power"],
                "sample_score_weighted": separation["score_weighted"],
                "call_status": "sample_structure"
                if separation["call_supported"]
                else "candidate_region",
                "comparison": "unsupervised_sample_structure",
                "group_annotation_status": "requested"
                if group_columns
                else "not_requested",
            }
        )

    core = pd.DataFrame(rows)
    core.attrs["covariance_threshold_used"] = covariance_threshold_used
    core.attrs["covariance_threshold_method"] = (
        config.covariance_threshold_method
    )
    core.attrs["covariance_threshold_components"] = (
        covariance_threshold_components
    )
    core["sample_qvalue_descriptive"] = benjamini_hochberg(
        core["sample_pvalue_descriptive"]
    )
    core["laplacian_rank_percentile"] = core["laplacian_score"].rank(
        method="average", pct=True, ascending=False
    )
    if config.calling_method == "legacy":
        core["stability_score"] = core["laplacian_rank_percentile"]
    else:
        core["stability_score"] = np.nan
    strata = core.apply(
        lambda row: fixed_strata(
            int(row["minor_group_size"]),
            int(row["n_cpg"]),
            float(row["missing_rate"]),
        ),
        axis=1,
        result_type="expand",
    )
    strata.columns = ["group_size_stratum", "cpg_stratum", "missingness_stratum"]
    core = pd.concat([core, strata], axis=1)
    core = add_rank_within_strata(
        core,
        score_column="sample_score_weighted",
        strata_columns=[
            "group_size_stratum",
            "cpg_stratum",
            "missingness_stratum",
        ],
        output_column="sample_rank_percentile",
    )
    if config.calling_method == "probabilistic" and config.stability_bootstraps:
        core["stability_score"] = _bootstrap_call_stability(
            matrix,
            loc,
            core,
            total_counts=coverage,
            config=config,
            covariance_threshold_used=covariance_threshold_used,
            source_cpg_offset=source_cpg_offset,
        )
    core = core[[c for c in CORE_COLUMNS if c in core.columns]]

    values_frame = pd.DataFrame(block_values_matrix, columns=sample_names)
    values_frame.insert(0, "dmr_id", core["dmr_id"].to_numpy())
    annotations = annotate_groups(
        core,
        values_frame,
        metadata,
        sample_column=sample_column,
        group_columns=group_columns or [],
        config=config,
    )
    core.attrs["covariance_threshold_used"] = covariance_threshold_used
    core.attrs["covariance_threshold_method"] = (
        config.covariance_threshold_method
    )
    core.attrs["covariance_threshold_components"] = (
        covariance_threshold_components
    )
    return core, values_frame, annotations


def _bootstrap_call_stability(
    matrix: np.ndarray,
    locations: pd.DataFrame,
    core: pd.DataFrame,
    *,
    total_counts: Optional[np.ndarray],
    config: EigerConfig,
    covariance_threshold_used: float,
    source_cpg_offset: int,
) -> np.ndarray:
    """Conditional sample-bootstrap recurrence of complete probabilistic calls."""

    if core.empty or config.stability_bootstraps <= 0:
        return np.full(len(core), np.nan, dtype=float)
    rng = np.random.default_rng(config.random_seed)
    counts = np.zeros(len(core), dtype=float)
    bootstrap_config = replace(
        config,
        covariance_threshold=float(covariance_threshold_used),
        covariance_threshold_method="fixed",
        stability_bootstraps=0,
    )
    for bootstrap_index in range(config.stability_bootstraps):
        sampled = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        bootstrap_core, _, _ = call_dmrs_from_matrix(
            matrix[:, sampled],
            locations,
            total_counts=(
                None if total_counts is None else total_counts[:, sampled]
            ),
            sample_names=[
                f"bootstrap_{bootstrap_index}_{sample_index}"
                for sample_index in range(matrix.shape[1])
            ],
            config=bootstrap_config,
            source_cpg_offset=source_cpg_offset,
        )
        bootstrap_core = bootstrap_core.loc[
            bootstrap_core["call_status"] == "sample_structure"
        ]
        if bootstrap_core.empty:
            continue
        for row_index, row in core.reset_index(drop=True).iterrows():
            candidates = bootstrap_core.loc[
                (bootstrap_core["chr"] == row["chr"])
                & (bootstrap_core["direction"] == row["direction"])
            ]
            if candidates.empty:
                continue
            left = np.maximum(candidates["cpg_start"], row["cpg_start"])
            right = np.minimum(candidates["cpg_end"], row["cpg_end"])
            intersection = np.maximum(0, right - left + 1)
            union = (
                candidates["cpg_end"]
                - candidates["cpg_start"]
                + 1
                + row["cpg_end"]
                - row["cpg_start"]
                + 1
                - intersection
            )
            if np.any(intersection / union >= config.stability_min_jaccard):
                counts[row_index] += 1.0
    return counts / config.stability_bootstraps


def _group_split(
    values: np.ndarray,
    labels: np.ndarray,
    config: EigerConfig,
    *,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    observed = np.isfinite(values) & pd.notna(labels)
    values = values[observed]
    labels = labels[observed].astype(str)
    groups = np.unique(labels)
    if values.size < 2 or groups.size < 2:
        return {"status": "insufficient_groups"}

    means = {group: float(np.mean(values[labels == group])) for group in groups}
    ordered = sorted(means, key=means.get)
    ordered_means = np.array([means[group] for group in ordered], dtype=float)
    split = int(np.argmax(np.diff(ordered_means)))
    gap = float(np.diff(ordered_means)[split])
    lower_groups = set(ordered[: split + 1])
    upper_groups = set(ordered[split + 1 :])
    lower_mask = np.array([label in lower_groups for label in labels])
    upper_mask = ~lower_mask
    if upper_mask.sum() < lower_mask.sum():
        target_mask, reference_mask = upper_mask, lower_mask
        target_groups, reference_groups = upper_groups, lower_groups
    else:
        target_mask, reference_mask = lower_mask, upper_mask
        target_groups, reference_groups = lower_groups, upper_groups

    target = values[target_mask]
    reference = values[reference_mask]
    target_mean = float(np.mean(target))
    reference_mean = float(np.mean(reference))
    delta = target_mean - reference_mean
    stdev = float(np.std(values))
    pvalue = _welch_pvalue(target, reference)
    score_raw = descriptive_score(delta, stdev, pvalue)
    minor_fraction = min(len(target), len(reference)) / len(values)
    target_prevalence = min(config.target_group_size / len(values), 0.5)
    weight = score_weight(
        config.prevalence_weight_method,
        minor_fraction,
        target_prevalence,
        sigma=config.gaussian_sigma,
        tau=config.logistic_tau,
        minimum=config.prevalence_weight_min,
        maximum=config.prevalence_weight_max,
    )
    power = prospective_power(
        len(target),
        len(reference),
        stdev,
        config.power_minimum_effect,
        alpha=config.power_alpha,
    )

    empirical = np.nan
    if config.annotation_permutations > 0:
        rng = rng or np.random.default_rng(config.random_seed)
        null_gaps = np.empty(config.annotation_permutations, dtype=float)
        for i in range(config.annotation_permutations):
            permuted = rng.permutation(labels)
            perm_means = sorted(
                float(np.mean(values[permuted == group])) for group in groups
            )
            null_gaps[i] = float(np.max(np.diff(perm_means)))
        empirical = (1.0 + np.sum(null_gaps >= gap)) / (
            config.annotation_permutations + 1.0
        )

    return {
        "status": (
            "separation"
            if abs(delta) >= config.delta_threshold
            else "no_separation"
        ),
        "target_groups": ";".join(sorted(target_groups)),
        "reference_groups": ";".join(sorted(reference_groups)),
        "target_n": len(target),
        "reference_n": len(reference),
        "minor_fraction": minor_fraction,
        "target_mean": target_mean,
        "reference_mean": reference_mean,
        "delta": delta,
        "direction": "Hypomethylation" if delta < 0 else "Hypermethylation",
        "gap": gap,
        "passes": abs(delta) >= config.delta_threshold,
        "pvalue": pvalue,
        "empirical": empirical,
        "score_raw": score_raw,
        "weight": weight,
        "power": power,
        "score_weighted": score_raw * weight,
    }


def annotate_groups(
    core: pd.DataFrame,
    values: pd.DataFrame,
    metadata: Optional[pd.DataFrame],
    *,
    sample_column: Optional[str],
    group_columns: Sequence[str],
    config: EigerConfig,
) -> pd.DataFrame:
    """Create optional long-form metadata annotations keyed by ``dmr_id``."""

    if not group_columns:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    if metadata is None:
        raise ValueError("metadata is required when group_columns are requested")
    meta = metadata.copy()
    sample_column = sample_column or str(meta.columns[0])
    required = {sample_column, *group_columns}
    missing = required.difference(meta.columns)
    if missing:
        raise KeyError(f"metadata lacks requested columns: {sorted(missing)}")
    if meta[sample_column].astype(str).duplicated().any():
        raise ValueError("metadata sample identifiers must be unique")

    meta[sample_column] = meta[sample_column].astype(str)
    meta = meta.set_index(sample_column)
    sample_columns = [c for c in values.columns if c != "dmr_id" and c in meta.index]
    if len(sample_columns) < 2:
        raise ValueError("fewer than two value columns matched metadata samples")

    rng = np.random.default_rng(config.random_seed)
    core_by_id = core.set_index("dmr_id")
    rows = []
    indexed_values = values.set_index("dmr_id")
    for dmr_id, value_row in indexed_values[sample_columns].iterrows():
        dmr_id = str(dmr_id)
        sample_values = value_row.to_numpy(dtype=float)
        dmr = core_by_id.loc[dmr_id]
        for group_column in group_columns:
            labels = meta.loc[sample_columns, group_column].to_numpy()
            result = _group_split(sample_values, labels, config, rng=rng)
            if result["status"] == "insufficient_groups":
                rows.append(
                    {
                        "dmr_id": dmr_id,
                        "annotation_source": "eiger_group",
                        "grouping_variable": group_column,
                        "annotation_status": "insufficient_groups",
                    }
                )
                continue
            group_bin, cpg_bin, missing_bin = fixed_strata(
                min(result["target_n"], result["reference_n"]),
                int(dmr["n_cpg"]),
                float(dmr["missing_rate"]),
            )
            rows.append(
                {
                    "dmr_id": dmr_id,
                    "annotation_source": "eiger_group",
                    "grouping_variable": group_column,
                    "target_groups": result["target_groups"],
                    "reference_groups": result["reference_groups"],
                    "target_n": result["target_n"],
                    "reference_n": result["reference_n"],
                    "minor_group_fraction": result["minor_fraction"],
                    "target_mean_beta": result["target_mean"],
                    "reference_mean_beta": result["reference_mean"],
                    "delta_beta": result["delta"],
                    "direction": result["direction"],
                    "delta_group_gap": result["gap"],
                    "passes_delta_threshold": result["passes"],
                    "annotation_status": result["status"],
                    "pvalue_descriptive": result["pvalue"],
                    "pvalue_empirical": result["empirical"],
                    "score_raw": result["score_raw"],
                    "prevalence_weight": result["weight"],
                    "power_support": result["power"],
                    "score_weighted": result["score_weighted"],
                    "group_size_stratum": group_bin,
                    "cpg_stratum": cpg_bin,
                    "missingness_stratum": missing_bin,
                }
            )

    annotations = pd.DataFrame(rows)
    if annotations.empty:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    for column in (
        "pvalue_descriptive",
        "pvalue_empirical",
        "score_weighted",
        "group_size_stratum",
        "cpg_stratum",
        "missingness_stratum",
    ):
        if column not in annotations:
            annotations[column] = np.nan
    annotations["qvalue_descriptive"] = np.nan
    annotations["qvalue_empirical"] = np.nan
    for _, idx in annotations.groupby("grouping_variable").groups.items():
        annotations.loc[idx, "qvalue_descriptive"] = benjamini_hochberg(
            annotations.loc[idx, "pvalue_descriptive"]
        )
        annotations.loc[idx, "qvalue_empirical"] = benjamini_hochberg(
            annotations.loc[idx, "pvalue_empirical"]
        )
    annotations = add_rank_within_strata(
        annotations,
        score_column="score_weighted",
        strata_columns=[
            "grouping_variable",
            "group_size_stratum",
            "cpg_stratum",
            "missingness_stratum",
        ],
        output_column="rank_percentile",
    )
    for column in ANNOTATION_COLUMNS:
        if column not in annotations:
            annotations[column] = np.nan
    return annotations[ANNOTATION_COLUMNS]


def combine_dmr_results(
    core_frames: Sequence[pd.DataFrame],
    value_frames: Sequence[pd.DataFrame],
    *,
    sample_names: Sequence[str],
    metadata: Optional[pd.DataFrame] = None,
    sample_column: Optional[str] = None,
    group_columns: Optional[Sequence[str]] = None,
    config: Optional[EigerConfig] = None,
    covariance_threshold_used: Optional[float] = None,
    covariance_threshold_method: Optional[str] = None,
    covariance_threshold_components: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine chromosome-level discoveries and apply genome-wide statistics."""

    config = config or EigerConfig()
    sample_names = [str(sample) for sample in sample_names]
    threshold = (
        config.covariance_threshold
        if covariance_threshold_used is None
        else float(covariance_threshold_used)
    )
    threshold_method = (
        config.covariance_threshold_method
        if covariance_threshold_method is None
        else str(covariance_threshold_method)
    )
    components = covariance_threshold_components or {}

    nonempty_cores = [frame for frame in core_frames if not frame.empty]
    nonempty_values = [frame for frame in value_frames if not frame.empty]
    if not nonempty_cores:
        return _empty_outputs(
            sample_names,
            covariance_threshold_used=threshold,
            covariance_threshold_method=threshold_method,
            covariance_threshold_components=components,
        )

    core = pd.concat(nonempty_cores, ignore_index=True)
    values = pd.concat(nonempty_values, ignore_index=True)
    if core["dmr_id"].duplicated().any() or values["dmr_id"].duplicated().any():
        raise ValueError("Chromosome-level DMR identifiers must be unique")
    if core["dmr_id"].tolist() != values["dmr_id"].tolist():
        raise ValueError("Chromosome-level core and value rows are misaligned")

    core["covariance_threshold_method"] = threshold_method
    core["covariance_threshold_used"] = threshold
    block_values = values[sample_names].to_numpy(dtype=float)
    laplacian_input = impute_for_covariance(block_values)
    core["laplacian_score"] = compute_laplacian_score(
        laplacian_input.T,
        neighbor_size=config.laplacian_neighbors,
    )
    core["sample_qvalue_descriptive"] = benjamini_hochberg(
        core["sample_pvalue_descriptive"]
    )
    core["laplacian_rank_percentile"] = core["laplacian_score"].rank(
        method="average",
        pct=True,
        ascending=False,
    )
    if config.calling_method == "legacy":
        core["stability_score"] = core["laplacian_rank_percentile"]
    elif "stability_score" not in core:
        core["stability_score"] = np.nan

    strata = core.apply(
        lambda row: fixed_strata(
            int(row["minor_group_size"]),
            int(row["n_cpg"]),
            float(row["missing_rate"]),
        ),
        axis=1,
        result_type="expand",
    )
    strata.columns = [
        "group_size_stratum",
        "cpg_stratum",
        "missingness_stratum",
    ]
    core = core.drop(columns=list(strata.columns), errors="ignore")
    core = pd.concat([core, strata], axis=1)
    core = add_rank_within_strata(
        core,
        score_column="sample_score_weighted",
        strata_columns=[
            "group_size_stratum",
            "cpg_stratum",
            "missingness_stratum",
        ],
        output_column="sample_rank_percentile",
    )
    core = core[[column for column in CORE_COLUMNS if column in core]]

    annotations = annotate_groups(
        core,
        values,
        metadata,
        sample_column=sample_column,
        group_columns=group_columns or [],
        config=config,
    )
    core.attrs["covariance_threshold_used"] = threshold
    core.attrs["covariance_threshold_method"] = threshold_method
    core.attrs["covariance_threshold_components"] = components
    return core, values, annotations
