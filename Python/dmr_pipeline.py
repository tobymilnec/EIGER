#!/usr/bin/env python3
"""Command-line EIGER DMR discovery with optional downstream annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eiger_core import (
    COMPACT_COLUMNS,
    DIAGNOSTIC_COLUMNS,
    EigerConfig,
    call_dmrs_from_matrix,
)
from functions.load_functions import load_methylation_frequencies
from functions.scoring import WEIGHT_METHODS


def _sidecar(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}{suffix}")


def _run_id(config: EigerConfig, sample_names: list[str]) -> str:
    payload = json.dumps(
        {"config": config.to_dict(), "samples": sample_names},
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
    parser.add_argument("--beta-dir", required=True, help="Directory of beta files")
    parser.add_argument(
        "--location-file",
        required=True,
        help="Gzipped CpG BED used to map beta rows to genomic coordinates",
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
        "--beta-file-column",
        default=None,
        help="Metadata beta filename column; defaults to the first column",
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
        "--max-cpg-gap-bp",
        type=int,
        default=None,
        help="Optional maximum genomic gap between adjacent CpGs in one DMR",
    )
    parser.add_argument(
        "--annotation-permutations",
        type=int,
        default=0,
        help="Label permutations per DMR/group; 0 disables empirical annotation p-values",
    )
    parser.add_argument("--random-seed", type=int, default=1)
    return parser


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
    beta_file_column = args.beta_file_column or str(metadata.columns[0])
    required = {sample_column, beta_file_column, *args.group_column}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata lacks requested columns: {sorted(missing)}")
    sample_names = metadata[sample_column].astype(str).tolist()
    if len(sample_names) != len(set(sample_names)):
        raise ValueError("Metadata sample identifiers must be unique")

    print(f"Loading {len(sample_names)} beta files...")
    beta_files = metadata[beta_file_column].astype(str).tolist()
    beta_paths = [str(Path(args.beta_dir) / name) for name in beta_files]
    beta_counts = load_methylation_frequencies(beta_paths, "beta")
    with np.errstate(divide="ignore", invalid="ignore"):
        betas = (
            beta_counts[:, :, 0].astype(float)
            / beta_counts[:, :, 1].astype(float)
        )

    print("Reading CpG locations...")
    locations = pd.read_csv(
        args.location_file,
        compression="gzip",
        header=None,
        sep="\t",
        quotechar='"',
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
        max_cpg_gap_bp=args.max_cpg_gap_bp,
        annotation_permutations=args.annotation_permutations,
        random_seed=args.random_seed,
        genome_build=args.genome_build,
    )

    print("Calling unsupervised covariance blocks...")
    core, values, annotations = call_dmrs_from_matrix(
        betas,
        locations,
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
        "schema_version": "2.1",
        "run_id": _run_id(config, sample_names),
        "genome_build": args.genome_build,
        "inputs": {
            "metadata": str(Path(args.meta).resolve()),
            "beta_directory": str(Path(args.beta_dir).resolve()),
            "location_file": str(Path(args.location_file).resolve()),
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
        "beta_file_column": beta_file_column,
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
