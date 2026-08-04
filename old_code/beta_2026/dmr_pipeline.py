#!/usr/bin/env python3
"""Command-line EIGER DMR discovery with optional downstream annotation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from eiger_core import (
    COMPACT_COLUMNS,
    DIAGNOSTIC_COLUMNS,
    EigerConfig,
    call_dmrs_from_matrix,
    combine_dmr_results,
    estimate_covariance_threshold_from_pairs,
)
from functions.load_functions import (
    beta_site_count,
    infer_methylation_file_type,
    load_bismark_coverage_betas,
    load_beta_values,
)
from functions.scoring import WEIGHT_METHODS


def _sidecar(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}{suffix}")


def _run_id(
    config: EigerConfig,
    sample_names: list[str],
    processing_mode: str,
) -> str:
    payload = json.dumps(
        {
            "config": config.to_dict(),
            "samples": sample_names,
            "processing_mode": processing_mode,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _covariance_settings(value: str) -> tuple[float, str]:
    if str(value).strip().lower() == "auto":
        return 0.012, "auto"
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--cov-threshold must be a number or 'auto'"
        ) from error
    return threshold, "fixed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover EIGER DMRs from methylation alone, then optionally "
            "annotate them with one or more metadata groupings."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--meta", required=True, help="Sample metadata CSV")
    parser.add_argument(
        "--input-dir",
        "--beta-dir",
        dest="input_dir",
        required=True,
        help="Directory containing beta/lbeta or Bismark cov/cov.gz files",
    )
    parser.add_argument(
        "--input-type",
        choices=("auto", "beta", "cov"),
        default="auto",
        help="Methylation input type; auto infers it from all file extensions",
    )
    parser.add_argument(
        "--location-file",
        required=True,
        help=(
            "Gzipped CpG BED used to map beta rows or align coverage "
            "coordinates"
        ),
    )
    parser.add_argument(
        "--location-third-column",
        choices=("auto", "cpg-index", "end"),
        default="auto",
        help=(
            "Meaning of a third column in a headerless CpG location file. "
            "Auto distinguishes wgbstools CpG indices from BED end coordinates"
        ),
    )
    parser.add_argument(
        "--output",
        default="eiger_dmrs.csv",
        help="Compact one-row-per-DMR core output",
    )
    parser.add_argument(
        "--values-output",
        default=None,
        help="Per-DMR sample methylation matrix; defaults to <output>_values.csv",
    )
    parser.add_argument(
        "--diagnostics-output",
        default=None,
        help="Detailed DMR scores and diagnostics; defaults to <output>_diagnostics.csv",
    )
    parser.add_argument(
        "--annotations-output",
        default=None,
        help="Long-form group annotations; defaults to <output>_annotations.csv",
    )
    parser.add_argument(
        "--run-metadata",
        default=None,
        help="Run manifest JSON; defaults to <output>_run.json",
    )
    parser.add_argument(
        "--legacy-wide-output",
        default=None,
        help="Optional core-plus-sample wide CSV for compatibility",
    )
    parser.add_argument(
        "--sample-column",
        default=None,
        help="Metadata sample identifier column; defaults to the first column",
    )
    parser.add_argument(
        "--file-column",
        "--beta-file-column",
        dest="file_column",
        default=None,
        help="Metadata input filename column; defaults to the first column",
    )
    parser.add_argument(
        "--group-column",
        action="append",
        default=[],
        help=(
            "Optional metadata grouping to annotate, such as celltype, tissue "
            "or phenotype. Repeat for multiple groupings."
        ),
    )
    parser.add_argument("--genome-build", default="unknown")
    parser.add_argument("--min-cpgs", type=int, default=4)
    parser.add_argument(
        "--calling-method",
        choices=("probabilistic", "legacy"),
        default="probabilistic",
        help=(
            "Probabilistic covariance-edge segmentation and mixture-supported "
            "sample calls, or the preserved legacy hard-threshold method"
        ),
    )
    parser.add_argument(
        "--cov-threshold",
        default="0.012",
        help=(
            "Fixed adjacent covariance threshold, or 'auto' for a label-free "
            "null/background estimate"
        ),
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.2,
        help="Minimum absolute difference between inferred cluster mean betas",
    )
    parser.add_argument("--cov-null-quantile", type=float, default=0.995)
    parser.add_argument("--cov-null-permutations", type=int, default=20)
    parser.add_argument("--cov-null-max-pairs", type=int, default=50000)
    parser.add_argument(
        "--cov-coordinate-system",
        choices=("auto", "one-based", "zero-based"),
        default="auto",
        help="Coordinate convention for Bismark coverage input",
    )
    parser.add_argument(
        "--cov-chunk-size",
        type=int,
        default=1_000_000,
        help="Rows read at once from each Bismark coverage file",
    )
    parser.add_argument(
        "--cov-min-match-rate",
        type=float,
        default=0.5,
        help="Minimum fraction of coverage rows that must match the CpG reference",
    )
    parser.add_argument(
        "--processing-mode",
        choices=("auto", "in-memory", "chromosome"),
        default="auto",
        help=(
            "Matrix processing mode; auto streams large beta cohorts by "
            "chromosome"
        ),
    )
    parser.add_argument(
        "--max-in-memory-gb",
        type=float,
        default=8.0,
        help=(
            "Estimated input-matrix size above which auto mode switches to "
            "chromosome streaming (includes read-depth confidence in "
            "probabilistic mode)"
        ),
    )
    parser.add_argument("--target-group-size", type=int, default=3)
    parser.add_argument(
        "--weight-method",
        choices=WEIGHT_METHODS,
        default="none",
        help="Optional prevalence correction; rank-within-strata is always reported",
    )
    parser.add_argument("--weight-min", type=float, default=0.25)
    parser.add_argument("--weight-max", type=float, default=4.0)
    parser.add_argument("--gaussian-sigma", type=float, default=0.05)
    parser.add_argument("--logistic-tau", type=float, default=0.02)
    parser.add_argument(
        "--power-min-effect",
        type=float,
        default=0.2,
        help="Pre-specified delta beta for prospective power support",
    )
    parser.add_argument("--power-alpha", type=float, default=0.05)
    parser.add_argument("--laplacian-neighbors", type=int, default=5)
    parser.add_argument("--min-observed-samples", type=int, default=2)
    parser.add_argument(
        "--pairwise-covariance",
        action="store_true",
        help=(
            "Use pairwise-complete adjacent covariance: each edge is estimated "
            "from the samples observed at BOTH CpGs, with the means taken over "
            "that same subset. Off by default, which reproduces the published "
            "complete-case behaviour where any missing value makes the edge NaN "
            "and breaks the region. Mean imputation is not an alternative: it "
            "deflates the covariance by (1 - missing_rate)**2."
        ),
    )
    parser.add_argument(
        "--min-paired-samples",
        type=int,
        default=5,
        help=(
            "With --pairwise-covariance, the minimum number of co-observed "
            "samples for an edge to be used at all."
        ),
    )
    parser.add_argument(
        "--max-cpg-gap-bp",
        type=int,
        default=300,
        help=(
            "Maximum genomic gap between retained CpGs in one DMR; retained "
            "CpGs may bridge coverage-filtered sites within this distance"
        ),
    )
    parser.add_argument(
        "--strict-cpg-adjacency",
        action="store_true",
        help=(
            "Require consecutive CpG reference rows, reproducing the previous "
            "strict behaviour and overriding --max-cpg-gap-bp"
        ),
    )
    parser.add_argument(
        "--annotation-permutations",
        type=int,
        default=0,
        help="Label permutations per DMR/group; 0 disables empirical annotation p-values",
    )
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--hmm-signal-prior", type=float, default=0.05)
    parser.add_argument("--hmm-signal-persistence", type=float, default=0.90)
    parser.add_argument("--hmm-posterior-threshold", type=float, default=0.50)
    parser.add_argument("--sample-model-prior", type=float, default=0.50)
    parser.add_argument("--sample-probability-threshold", type=float, default=0.80)
    parser.add_argument("--min-sample-cluster-size", type=int, default=2)
    parser.add_argument(
        "--stability-bootstraps",
        type=int,
        default=0,
        help=(
            "Optional sample bootstraps for conditional whole-caller recurrence; "
            "0 disables this computationally expensive calculation"
        ),
    )
    parser.add_argument("--stability-min-jaccard", type=float, default=0.50)
    return parser


def _chromosome_slices(locations: pd.DataFrame) -> list[tuple[str, int, int]]:
    chromosomes = locations.iloc[:, 0].astype(str).to_numpy()
    if len(chromosomes) == 0:
        return []
    boundaries = np.empty(len(chromosomes), dtype=bool)
    boundaries[0] = True
    boundaries[1:] = chromosomes[1:] != chromosomes[:-1]
    starts = np.flatnonzero(boundaries)
    ends = np.r_[starts[1:], len(chromosomes)]
    slices = []
    seen = set()
    for start, end in zip(starts, ends):
        chromosome = str(chromosomes[start])
        if chromosome in seen:
            raise ValueError(
                f"CpG locations for chromosome {chromosome!r} are not contiguous"
            )
        seen.add(chromosome)
        slices.append((chromosome, int(start), int(end)))
    return slices


def _normalise_location_reference(
    locations: pd.DataFrame,
    third_column: str = "auto",
) -> tuple[pd.DataFrame, str]:
    """Name CpG-reference columns and create BED-style half-open intervals."""
    if locations.shape[1] < 2:
        raise ValueError("CpG location files require chromosome and start columns")

    loc = locations
    chrom = loc.iloc[:, 0].astype(str)
    starts = pd.to_numeric(loc.iloc[:, 1], errors="raise").astype("int64")
    resolved = "absent"
    ends = starts + 1

    if loc.shape[1] >= 3:
        third = pd.to_numeric(loc.iloc[:, 2], errors="raise").astype("int64")
        resolved = third_column
        if third_column == "auto":
            deltas = third - starts
            looks_like_end = float(deltas.between(1, 10).mean()) >= 0.99
            sequential = third.groupby(chrom, sort=False).diff().dropna()
            looks_like_index = (
                len(sequential) > 0
                and float(sequential.eq(1).mean()) >= 0.99
            )
            if looks_like_end:
                resolved = "end"
            elif looks_like_index:
                resolved = "cpg-index"
            else:
                raise ValueError(
                    "Could not determine whether the third location column is "
                    "a BED end or CpG index. Set --location-third-column to "
                    "'end' or 'cpg-index'."
                )
        if resolved == "end":
            ends = third
            if (ends <= starts).any():
                raise ValueError(
                    "Location end coordinates must be greater than starts"
                )
        elif resolved != "cpg-index":
            raise ValueError(f"Unknown third-column interpretation: {resolved}")

    normalised = pd.DataFrame(
        {"chr": chrom, "start": starts, "end": ends}
    )
    if loc.shape[1] >= 3 and resolved == "cpg-index":
        normalised["reference_cpg_index"] = pd.to_numeric(
            loc.iloc[:, 2], errors="raise"
        ).astype("int64")
    for column_index in range(3, loc.shape[1]):
        normalised[f"location_extra_{column_index + 1}"] = loc.iloc[
            :, column_index
        ].to_numpy()
    return normalised, resolved


def _validate_beta_inputs(input_paths: list[str], expected_sites: int) -> None:
    for path in input_paths:
        observed_sites = beta_site_count(path)
        if observed_sites != expected_sites:
            raise ValueError(
                f"{path} contains {observed_sites:,} CpG rows, but the "
                f"location file contains {expected_sites:,}. Use the CpG "
                "reference from the same genome configuration."
            )


def _pair_quotas(
    chromosome_slices: list[tuple[str, int, int]],
    maximum_pairs: int,
) -> np.ndarray:
    pair_counts = np.array(
        [max(0, end - start - 1) for _, start, end in chromosome_slices],
        dtype=float,
    )
    if pair_counts.sum() == 0:
        return np.zeros(len(pair_counts), dtype=int)
    exact = maximum_pairs * pair_counts / pair_counts.sum()
    quotas = np.floor(exact).astype(int)
    remainder = maximum_pairs - int(quotas.sum())
    if remainder:
        order = np.argsort(exact - quotas)[::-1]
        quotas[order[:remainder]] += 1
    return quotas


def _eligible_pair_mask(
    valid_indices: np.ndarray,
    chromosome_locations: pd.DataFrame,
    config: EigerConfig,
) -> np.ndarray:
    if config.max_cpg_gap_bp is None:
        return np.diff(valid_indices) == 1

    lower = {str(column).lower(): column for column in chromosome_locations}
    start_column = lower.get("start", chromosome_locations.columns[1])
    end_column = lower.get("end")
    starts = chromosome_locations[start_column].to_numpy(dtype=np.int64)
    ends = (
        chromosome_locations[end_column].to_numpy(dtype=np.int64)
        if end_column is not None
        else starts + 1
    )
    gaps = starts[valid_indices[1:]] - ends[valid_indices[:-1]]
    return (gaps >= 0) & (gaps <= config.max_cpg_gap_bp)


def _residual_pair_reservoir(
    input_paths: list[str],
    locations: pd.DataFrame,
    chromosome_slices: list[tuple[str, int, int]],
    config: EigerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.random_seed)
    quotas = _pair_quotas(
        chromosome_slices,
        config.covariance_null_max_pairs,
    )
    left_pairs = []
    right_pairs = []

    print("Estimating automatic covariance threshold (streaming pass 1/2)...")
    for chromosome_index, (chromosome, start, end) in enumerate(
        chromosome_slices
    ):
        quota = int(quotas[chromosome_index])
        if quota == 0:
            continue
        print(
            f"  [{chromosome_index + 1}/{len(chromosome_slices)}] "
            f"{chromosome}: sampling background pairs",
            flush=True,
        )
        chromosome_betas = load_beta_values(
            input_paths,
            slice_start=start,
            slice_end=end,
            dtype=np.float32,
        )
        valid = (
            np.sum(np.isfinite(chromosome_betas), axis=1)
            >= config.min_observed_samples
        )
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) < 2:
            del chromosome_betas
            continue
        eligible = _eligible_pair_mask(
            valid_indices,
            locations.iloc[start:end],
            config,
        )
        pair_indices = np.flatnonzero(eligible)
        if len(pair_indices) == 0:
            del chromosome_betas
            continue

        residuals = np.array(chromosome_betas[valid], dtype=np.float32, copy=True)
        del chromosome_betas
        with np.errstate(invalid="ignore"):
            row_means = np.nanmean(residuals, axis=1)
        for sample_index in range(residuals.shape[1]):
            missing = ~np.isfinite(residuals[:, sample_index])
            residuals[missing, sample_index] = row_means[missing]
        residuals -= row_means[:, None]

        selected = rng.choice(
            pair_indices,
            size=min(quota, len(pair_indices)),
            replace=False,
        )
        left_pairs.append(np.array(residuals[selected], copy=True))
        right_pairs.append(np.array(residuals[selected + 1], copy=True))
        del residuals
        gc.collect()

    if not left_pairs:
        raise ValueError(
            "No eligible adjacent CpG pairs were available for the automatic "
            "covariance threshold"
        )
    return np.concatenate(left_pairs), np.concatenate(right_pairs)


def _call_beta_by_chromosome(
    input_paths: list[str],
    locations: pd.DataFrame,
    *,
    sample_names: list[str],
    metadata: pd.DataFrame,
    sample_column: str,
    group_columns: list[str],
    config: EigerConfig,
):
    chromosome_slices = _chromosome_slices(locations)
    covariance_threshold_used = config.covariance_threshold
    covariance_threshold_components = {}
    call_config = config

    if config.covariance_threshold_method == "auto":
        left_pairs, right_pairs = _residual_pair_reservoir(
            input_paths,
            locations,
            chromosome_slices,
            config,
        )
        estimate = estimate_covariance_threshold_from_pairs(
            left_pairs,
            right_pairs,
            quantile=config.covariance_null_quantile,
            permutations=config.covariance_null_permutations,
            max_pairs=config.covariance_null_max_pairs,
            seed=config.random_seed,
        )
        covariance_threshold_used = estimate.threshold
        covariance_threshold_components = {
            "resampled_null_threshold": estimate.threshold,
            "null_pairs_sampled": estimate.null_pairs_sampled,
            "pair_reservoir_size": int(len(left_pairs)),
            "streaming_method": "chromosome_stratified_pair_reservoir",
        }
        call_config = replace(
            config,
            covariance_threshold=estimate.threshold,
            covariance_threshold_method="fixed",
        )
        del left_pairs, right_pairs
        gc.collect()
        print(
            f"Automatic covariance threshold: {estimate.threshold:.6g}",
            flush=True,
        )

    pass_label = "2/2" if config.covariance_threshold_method == "auto" else "1/1"
    print(f"Calling DMRs by chromosome (streaming pass {pass_label})...")
    core_frames = []
    value_frames = []
    for chromosome_index, (chromosome, start, end) in enumerate(
        chromosome_slices
    ):
        print(
            f"  [{chromosome_index + 1}/{len(chromosome_slices)}] "
            f"{chromosome}: {end - start:,} CpGs",
            flush=True,
        )
        loaded = load_beta_values(
            input_paths,
            slice_start=start,
            slice_end=end,
            dtype=np.float32,
            return_total_counts=config.calling_method == "probabilistic",
        )
        if config.calling_method == "probabilistic":
            chromosome_betas, chromosome_coverage = loaded
        else:
            chromosome_betas, chromosome_coverage = loaded, None
        chromosome_core, chromosome_values, _ = call_dmrs_from_matrix(
            chromosome_betas,
            locations.iloc[start:end],
            total_counts=chromosome_coverage,
            sample_names=sample_names,
            group_columns=[],
            config=call_config,
            source_cpg_offset=start,
        )
        core_frames.append(chromosome_core)
        value_frames.append(chromosome_values)
        del chromosome_betas, chromosome_coverage
        gc.collect()

    return combine_dmr_results(
        core_frames,
        value_frames,
        sample_names=sample_names,
        metadata=metadata,
        sample_column=sample_column,
        group_columns=group_columns,
        config=config,
        covariance_threshold_used=covariance_threshold_used,
        covariance_threshold_method=config.covariance_threshold_method,
        covariance_threshold_components=covariance_threshold_components,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    covariance_threshold, covariance_threshold_method = _covariance_settings(
        args.cov_threshold
    )
    output = Path(args.output)
    values_output = (
        Path(args.values_output)
        if args.values_output
        else _sidecar(output, "_values.csv")
    )
    diagnostics_output = (
        Path(args.diagnostics_output)
        if args.diagnostics_output
        else _sidecar(output, "_diagnostics.csv")
    )
    annotations_output = (
        Path(args.annotations_output)
        if args.annotations_output
        else _sidecar(output, "_annotations.csv")
    )
    manifest_output = (
        Path(args.run_metadata)
        if args.run_metadata
        else _sidecar(output, "_run.json")
    )
    for path in (
        output,
        values_output,
        diagnostics_output,
        annotations_output,
        manifest_output,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading sample metadata...")
    metadata = pd.read_csv(args.meta)
    sample_column = args.sample_column or str(metadata.columns[0])
    file_column = args.file_column or str(metadata.columns[0])
    required = {sample_column, file_column, *args.group_column}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata lacks requested columns: {sorted(missing)}")
    sample_names = metadata[sample_column].astype(str).tolist()
    if len(sample_names) != len(set(sample_names)):
        raise ValueError("Metadata sample identifiers must be unique")

    print("Reading CpG locations...")
    raw_locations = pd.read_csv(
        args.location_file,
        compression="gzip",
        header=None,
        sep="\t",
        quotechar='"',
    )
    locations, location_third_column = _normalise_location_reference(
        raw_locations,
        args.location_third_column,
    )
    del raw_locations
    gc.collect()
    print(
        "CpG location interpretation: "
        f"third column={location_third_column}; intervals are 0-based half-open"
    )
    chromosome_column = locations.columns[0]
    locations[chromosome_column] = locations[chromosome_column].astype("category")
    input_files = metadata[file_column].astype(str).tolist()
    input_paths = [str(Path(args.input_dir) / name) for name in input_files]
    inferred_type = infer_methylation_file_type(input_paths)
    input_type = (
        inferred_type
        if args.input_type == "auto"
        else args.input_type
    )
    if input_type != inferred_type:
        raise ValueError(
            f"--input-type {input_type!r} does not match the supplied "
            f"{inferred_type!r} files"
        )

    config = EigerConfig(
        min_cpgs=args.min_cpgs,
        covariance_threshold=covariance_threshold,
        covariance_threshold_method=covariance_threshold_method,
        covariance_null_quantile=args.cov_null_quantile,
        covariance_null_permutations=args.cov_null_permutations,
        covariance_null_max_pairs=args.cov_null_max_pairs,
        delta_threshold=args.delta_threshold,
        target_group_size=args.target_group_size,
        prevalence_weight_method=args.weight_method,
        prevalence_weight_min=args.weight_min,
        prevalence_weight_max=args.weight_max,
        gaussian_sigma=args.gaussian_sigma,
        logistic_tau=args.logistic_tau,
        power_minimum_effect=args.power_min_effect,
        power_alpha=args.power_alpha,
        laplacian_neighbors=args.laplacian_neighbors,
        min_observed_samples=args.min_observed_samples,
        pairwise_covariance=args.pairwise_covariance,
        min_paired_samples=args.min_paired_samples,
        max_cpg_gap_bp=(
            None if args.strict_cpg_adjacency else args.max_cpg_gap_bp
        ),
        annotation_permutations=args.annotation_permutations,
        random_seed=args.random_seed,
        genome_build=args.genome_build,
        calling_method=args.calling_method,
        hmm_signal_prior=args.hmm_signal_prior,
        hmm_signal_persistence=args.hmm_signal_persistence,
        hmm_posterior_threshold=args.hmm_posterior_threshold,
        sample_model_prior=args.sample_model_prior,
        sample_probability_threshold=args.sample_probability_threshold,
        min_sample_cluster_size=args.min_sample_cluster_size,
        stability_bootstraps=args.stability_bootstraps,
        stability_min_jaccard=args.stability_min_jaccard,
    )

    if args.max_in_memory_gb <= 0:
        raise ValueError("--max-in-memory-gb must be positive")
    matrix_gb = len(locations) * len(sample_names) * 8 / 1_000_000_000
    coverage_matrix_gb = (
        len(locations) * len(sample_names) * 4 / 1_000_000_000
        if config.calling_method == "probabilistic"
        else 0.0
    )
    input_matrices_gb = matrix_gb + coverage_matrix_gb
    processing_mode = args.processing_mode
    if processing_mode == "auto":
        processing_mode = (
            "chromosome"
            if input_type == "beta" and input_matrices_gb > args.max_in_memory_gb
            else "in-memory"
        )
    if processing_mode == "chromosome" and input_type != "beta":
        raise ValueError(
            "Chromosome processing currently requires beta/lbeta input; use "
            "--processing-mode in-memory for coverage files"
        )

    print(
        f"Whole-genome float64 matrix estimate: {matrix_gb:.1f} GB "
        "(before working copies)"
    )
    if coverage_matrix_gb:
        print(
            "Read-depth confidence matrix estimate: "
            f"{coverage_matrix_gb:.1f} GB (before working copies)"
        )
    print(f"Processing mode: {processing_mode}")
    coverage_summaries = []
    if input_type == "beta":
        _validate_beta_inputs(input_paths, len(locations))

    if processing_mode == "chromosome":
        core, values, annotations = _call_beta_by_chromosome(
            input_paths,
            locations,
            sample_names=sample_names,
            metadata=metadata,
            sample_column=sample_column,
            group_columns=args.group_column,
            config=config,
        )
    else:
        print(f"Loading {len(sample_names)} {input_type} files...")
        if input_type == "beta":
            loaded = load_beta_values(
                input_paths,
                dtype=float,
                return_total_counts=config.calling_method == "probabilistic",
            )
            if config.calling_method == "probabilistic":
                betas, total_counts = loaded
            else:
                betas, total_counts = loaded, None
        else:
            loaded = load_bismark_coverage_betas(
                input_paths,
                locations,
                coordinate_system=args.cov_coordinate_system,
                chunk_size=args.cov_chunk_size,
                minimum_match_rate=args.cov_min_match_rate,
                progress=True,
                return_total_counts=config.calling_method == "probabilistic",
            )
            if config.calling_method == "probabilistic":
                betas, coverage_summaries, total_counts = loaded
            else:
                betas, coverage_summaries = loaded
                total_counts = None
        print("Calling unsupervised covariance blocks...")
        core, values, annotations = call_dmrs_from_matrix(
            betas,
            locations,
            total_counts=total_counts,
            sample_names=sample_names,
            metadata=metadata,
            sample_column=sample_column,
            group_columns=args.group_column,
            config=config,
        )

    compact = core[[column for column in COMPACT_COLUMNS if column in core]]
    diagnostics = core[
        [column for column in DIAGNOSTIC_COLUMNS if column in core]
    ]
    compact.to_csv(output, index=False)
    diagnostics.to_csv(diagnostics_output, index=False)
    values.to_csv(values_output, index=False)
    annotations.to_csv(annotations_output, index=False)
    if args.legacy_wide_output:
        legacy = core.merge(values, on="dmr_id", how="left", validate="one_to_one")
        legacy_path = Path(args.legacy_wide_output)
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy.to_csv(legacy_path, index=False)

    manifest = {
        "schema_version": "3.0",
        "run_id": _run_id(config, sample_names, processing_mode),
        "genome_build": args.genome_build,
        "inputs": {
            "metadata": str(Path(args.meta).resolve()),
            "input_directory": str(Path(args.input_dir).resolve()),
            "input_type": input_type,
            "processing_mode": processing_mode,
            "whole_matrix_estimate_gb": matrix_gb,
            "coverage_matrix_estimate_gb": coverage_matrix_gb,
            "input_matrices_estimate_gb": input_matrices_gb,
            "beta_directory": (
                str(Path(args.input_dir).resolve())
                if input_type == "beta"
                else None
            ),
            "location_file": str(Path(args.location_file).resolve()),
            "location_third_column": location_third_column,
            "coverage_alignment": coverage_summaries,
        },
        "outputs": {
            "dmrs": str(output.resolve()),
            "diagnostics": str(diagnostics_output.resolve()),
            "values": str(values_output.resolve()),
            "annotations": str(annotations_output.resolve()),
            "legacy_wide": str(Path(args.legacy_wide_output).resolve())
            if args.legacy_wide_output
            else None,
        },
        "sample_column": sample_column,
        "file_column": file_column,
        "beta_file_column": file_column if input_type == "beta" else None,
        "group_columns": args.group_column,
        "n_samples": len(sample_names),
        "n_dmrs": len(compact),
        "n_annotations": len(annotations),
        "parameters": config.to_dict(),
        "covariance_threshold_used": core.attrs.get(
            "covariance_threshold_used", covariance_threshold
        ),
        "covariance_threshold_components": core.attrs.get(
            "covariance_threshold_components", {}
        ),
        "statistical_notes": {
            "sample_pvalues": (
                "Descriptive only: the split and test use the same observations."
            ),
            "group_pvalues": (
                "Descriptive unless annotation_permutations is positive; empirical "
                "annotation p-values condition on the discovered DMR boundaries."
            ),
            "power_support": (
                "Prospective normal-approximation power at power_minimum_effect; "
                "reported as support, not multiplied into the score."
            ),
            "automatic_covariance_threshold": (
                "When enabled, uses a label-free genome-resampled residual "
                "adjacent-covariance null."
            ),
            "delta_threshold": (
                "Applied to the absolute difference between inferred cluster "
                "mean betas; delta_minmax remains an adjacent-gap diagnostic."
            ),
            "probabilistic_calling": (
                "Uses the original adjacent covariance and threshold. Edge "
                "uncertainty is smoothed within eligible genomic runs; sample "
                "calls additionally require one-versus-two-component model support."
            ),
            "stability_score": (
                "In probabilistic mode this is conditional sample-bootstrap call "
                "recurrence when stability_bootstraps is positive; otherwise NaN. "
                "Legacy mode retains the historical Laplacian percentile value."
            ),
            "coverage_uncertainty": (
                "Probabilistic mode propagates Jeffreys-Beta posterior variance "
                "from methylated-read totals into edge and effect uncertainty; "
                "the covariance values themselves are unchanged."
            ),
        },
    }
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(compact):,} compact DMRs to {output}")
    print(
        "Covariance threshold used: "
        f"{core.attrs.get('covariance_threshold_used', covariance_threshold):.6g} "
        f"({covariance_threshold_method})"
    )
    print(f"Wrote detailed diagnostics to {diagnostics_output}")
    print(f"Wrote sample values to {values_output}")
    print(f"Wrote {len(annotations):,} annotations to {annotations_output}")
    print(f"Wrote run metadata to {manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
