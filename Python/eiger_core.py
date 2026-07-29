"""Reusable EIGER matrix API with compact core and optional annotations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    from .functions.processing import (
        compute_laplacian_score,
        covariance_blocks,
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
    max_cpg_gap_bp: Optional[int] = None
    annotation_permutations: int = 0
    random_seed: int = 1
    genome_build: str = "unknown"

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

    def to_dict(self) -> dict:
        return asdict(self)


def impute_for_covariance(betas: np.ndarray) -> np.ndarray:
    """Mean-impute within CpG for covariance calculations only."""

    values = np.asarray(betas, dtype=float).copy()
    with np.errstate(invalid="ignore"):
        row_means = np.nanmean(values, axis=1)
    global_mean = np.nanmean(values)
    if not np.isfinite(global_mean):
        global_mean = 0.5
    row_means = np.where(np.isfinite(row_means), row_means, global_mean)
    missing = np.where(~np.isfinite(values))
    values[missing] = row_means[missing[0]]
    return values


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


def _standardise_locations(locations, n_rows: int) -> pd.DataFrame:
    loc = pd.DataFrame(locations).copy()
    if len(loc) != n_rows:
        raise ValueError("locations and beta matrix must have the same number of rows")

    lower = {str(c).lower(): c for c in loc.columns}
    chrom_col = next(
        (lower[k] for k in ("chr", "chrom", "chromosome") if k in lower),
        loc.columns[0],
    )
    start_col = lower.get("start", loc.columns[1] if loc.shape[1] > 1 else None)
    end_col = lower.get("end", loc.columns[2] if loc.shape[1] > 2 else start_col)
    if start_col is None:
        raise ValueError("locations must contain chromosome and start columns")
    return pd.DataFrame(
        {
            "chr": loc[chrom_col].astype(str),
            "start": pd.to_numeric(loc[start_col], errors="raise").astype("int64"),
            "end": pd.to_numeric(loc[end_col], errors="raise").astype("int64"),
            "source_cpg_index": np.arange(n_rows, dtype=int),
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
            "pvalue": np.nan,
            "score_raw": 0.0,
            "weight": 1.0,
            "power": 0.0,
            "score_weighted": 0.0,
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
        "pvalue": pvalue,
        "score_raw": score_raw,
        "weight": weight,
        "power": power,
        "score_weighted": score_raw * weight,
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
    result = _separation_from_values(array, config or EigerConfig())
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
    sample_names: Optional[Sequence[str]] = None,
    metadata: Optional[pd.DataFrame] = None,
    sample_column: Optional[str] = None,
    group_columns: Optional[Sequence[str]] = None,
    config: Optional[EigerConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Call EIGER DMRs from a CpG-by-sample methylation matrix."""

    config = config or EigerConfig()
    matrix = np.asarray(betas, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("betas must be a two-dimensional CpG-by-sample matrix")
    n_cpg_total, n_samples = matrix.shape
    if n_samples < 2:
        raise ValueError("EIGER requires at least two samples")
    if sample_names is None:
        sample_names = [f"sample_{i + 1}" for i in range(n_samples)]
    sample_names = [str(s) for s in sample_names]
    if len(sample_names) != n_samples or len(set(sample_names)) != n_samples:
        raise ValueError("sample_names must be unique and match the matrix columns")

    loc = _standardise_locations(locations, n_cpg_total)
    valid = np.sum(np.isfinite(matrix), axis=1) >= config.min_observed_samples
    if np.sum(valid) < config.min_cpgs:
        return _empty_outputs(
            sample_names,
            covariance_threshold_used=config.covariance_threshold,
            covariance_threshold_method=config.covariance_threshold_method,
        )

    valid_matrix = matrix[valid]
    valid_locations = loc.loc[valid].reset_index(drop=True)
    covariance_input = impute_for_covariance(valid_matrix)
    adjacent_covariance = np.asarray(covariance_blocks(covariance_input), dtype=float)
    eligible_pairs = (
        valid_locations["chr"].iloc[:-1].to_numpy()
        == valid_locations["chr"].iloc[1:].to_numpy()
    )
    eligible_pairs &= (
        np.diff(valid_locations["source_cpg_index"].to_numpy()) == 1
    )
    if config.max_cpg_gap_bp is not None:
        gaps = (
            valid_locations["start"].iloc[1:].to_numpy()
            - valid_locations["end"].iloc[:-1].to_numpy()
        )
        eligible_pairs &= gaps <= config.max_cpg_gap_bp
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
    adjacent_covariance[~eligible_pairs] = -np.inf
    blocks = extract_blocks(
        adjacent_covariance,
        DMR_length_threshold=config.min_cpgs,
        cov_threshold=covariance_threshold_used,
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
        separation = _separation_from_values(values, config)
        n_observed = int(np.sum(np.isfinite(values)))
        n_cpg = last - first + 1
        covariance_score = float(np.nanmean(adjacent_covariance[first:last]))
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
                "sample_score_raw": separation["score_raw"],
                "prevalence_weight": separation["weight"],
                "power_support": separation["power"],
                "sample_score_weighted": separation["score_weighted"],
                "call_status": "sample_structure"
                if separation["passes"]
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
    core["stability_score"] = core["laplacian_rank_percentile"]
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
