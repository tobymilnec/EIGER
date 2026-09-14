"""Command-line interface for EIGER.

Two callers are exposed from one entry point so that the published method and
the revised method can be run on identical inputs and their outputs compared
directly.

``eiger2 call``
    The revised caller: pairwise-complete estimation, correlation or covariance,
    a locality-breaking null, and a region-level empirical false discovery rate.

``eiger2 call --legacy``
    The method as originally published, invoked from
    ``Python/functions/processing.py`` rather than reimplemented:
    complete-case adjacent covariance against a fixed threshold, contiguous runs
    of at least ``--min-cpgs`` CpGs, and the sample partition and effect-size
    criterion from ``find_separators``. No false discovery rate is available in
    this mode, because the published method does not define one; the
    ``region_qvalue`` column is written as NA so that the two output schemas
    remain comparable.

Examples
--------
    # revised caller on a beta matrix
    eiger2 call --betas betas.tsv --locations cpgs.bed --out results/

    # the published method, at its published threshold
    eiger2 call --betas betas.tsv --locations cpgs.bed --out results_legacy/ \\
        --legacy --cov-threshold 0.012 --delta-threshold 0.20

    # both, for a side-by-side comparison
    eiger2 compare --betas betas.tsv --locations cpgs.bed --out comparison/

Inputs
------
``--betas``      CpG-by-sample matrix, TSV/CSV, one row per CpG. May contain
                 missing values as empty fields, ``NA`` or ``NaN``.
``--locations``  BED-like table with ``chr``, ``start``, ``end``, one row per
                 CpG, in the same order as ``--betas``.

Every run writes a JSON manifest recording the resolved configuration, package
versions, input checksums and the realised null diagnostics, so that a result
can be traced to the code and data that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .core import (
    BOTTOM_STRATUM_FDR_NOTE,
    CALIBRATION_DIAGNOSTIC_NOTE,
    LOW_PREVALENCE_NOTE,
    LOW_PREVALENCE_SAMPLE_CEILING,
    NEAR_BOUNDARY_CONCENTRATION_NOTE,
    NEAR_CUTOFF_CONCENTRATION_NOTE,
    NO_CALIBRATED_NULL_NOTE,
    Eiger2Config,
    call_dmrs,
    chromosome_sort_permutation,
    main_region_table,
    member_step_min_n_cpg,
)
from .manifest_utils import build_manifest_envelope

# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------


def _read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read a delimited table with pandas' C parser.

    The separator is taken from the file extension, and gzip is handled by
    extension too. ``engine="python"`` with ``sep=None`` would sniff the
    delimiter automatically but is roughly eleven-fold slower, which on a
    three-million-row panel matrix is the difference between five seconds and a
    minute, so the delimiter is inferred from the name instead.
    """

    name = path.name.lower()
    name = name.removesuffix(".gz")
    separator = "," if name.endswith(".csv") else "\t"
    return pd.read_csv(path, sep=separator, **kwargs)


def _checksum(path: Path) -> str:
    """SHA-256 of the whole file, truncated to 16 hex digits for display.

    A prior version stopped after the first 64MB, so two inputs sharing that
    prefix collided; every genome-scale beta matrix exceeds it. Hashing the
    whole file is the honest option for a tool whose case rests on
    reproducible manifests -- see spec §2a.6.
    """

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _git_commit() -> str:
    """Commit-only helper retained for lightweight benchmark manifests."""

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _git_provenance() -> dict[str, str | bool | None]:
    """Identify both HEAD and the exact dirty worktree used for a run.

    ``git_commit`` alone is misleading when tracked source changes have not
    been committed. The fingerprint covers Git's porcelain status (including
    untracked path names) and the binary-safe tracked diff from HEAD. It does
    not read untracked file contents, avoiding accidentally hashing large
    generated cohort outputs while still making their presence visible in the
    dirty state/fingerprint.
    """

    repository = Path(__file__).resolve().parents[2]
    try:
        commit = _git_commit()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=repository, capture_output=True, check=True,
        ).stdout
        dirty = bool(status)
        fingerprint = None
        if dirty:
            tracked_diff = subprocess.run(
                ["git", "diff", "--binary", "HEAD", "--", "."],
                cwd=repository, capture_output=True, check=True,
            ).stdout
            digest = hashlib.sha256()
            digest.update(status)
            digest.update(b"\0EIGER_TRACKED_DIFF\0")
            digest.update(tracked_diff)
            fingerprint = digest.hexdigest()
        return {
            "git_commit": commit,
            "git_worktree_dirty": dirty,
            "git_worktree_fingerprint_sha256": fingerprint,
        }
    except Exception:
        return {
            "git_commit": "unknown",
            "git_worktree_dirty": None,
            "git_worktree_fingerprint_sha256": None,
        }


def _peak_memory_mb() -> float:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


#: Column names recognised as coordinates rather than samples, in any case.
COORDINATE_NAMES = {
    "chr", "chrom", "chromosome", "seqnames",
    "start", "pos", "position", "end", "stop",
    "strand", "cpg_index", "cpgindex", "index",
}


def _standardise_locations(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Coerce any of the accepted coordinate layouts to ``chr/start/end``.

    Accepted:

    * named columns ``chr``/``chrom``/``chromosome``/``seqnames``,
      ``start``/``pos``/``position``, and optionally ``end``/``stop``;
    * a headerless three-column BED, either ``chr start end`` or the
      wgbstools ``chr position CpG_index`` layout.

    Where no end coordinate is supplied the CpG is represented as a one-base
    interval, which is what every downstream step assumes.
    """

    lower = {str(c).lower(): c for c in frame.columns}
    chrom = next((lower[c] for c in ("chr", "chrom", "chromosome", "seqnames")
                  if c in lower), None)
    start = next((lower[c] for c in ("start", "pos", "position")
                  if c in lower), None)
    end = next((lower[c] for c in ("end", "stop") if c in lower), None)

    if chrom is None or start is None:
        raise SystemExit(
            f"{source} needs a chromosome column (chr / chrom / chromosome / "
            f"seqnames) and a position column (start / pos / position).\n"
            f"Columns found: {list(frame.columns)}\n"
            f"If the file is headerless, this is expected: pandas has taken "
            f"the first data row as the header. Pass --locations-no-header."
        )

    out = pd.DataFrame({
        "chr": frame[chrom].astype(str),
        "start": pd.to_numeric(frame[start], errors="coerce").astype("int64"),
    })
    if end is not None:
        candidate = pd.to_numeric(frame[end], errors="coerce")
        # A wgbstools CpG index is a running counter, not a coordinate; it is
        # detected by being smaller than the start it supposedly ends.
        if (candidate >= out["start"]).mean() > 0.99:
            out["end"] = candidate.astype("int64")
        else:
            out["end"] = out["start"] + 1
    else:
        out["end"] = out["start"] + 1
    return out


def load_matrix_file(
    path: Path, *, no_header: bool = False
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Read a single table holding coordinates AND samples.

    This is the methrix / bsseq layout: the leading columns are coordinates
    (``chr``, ``start``, and often ``strand``) and every remaining column is a
    sample. It avoids having to split the data into two files that must then be
    kept in the same order.
    """

    frame = _read_table(path, header=None if no_header else "infer")
    if no_header:
        frame.columns = (
            ["chr", "start", "end"][: min(3, frame.shape[1])]
            + [f"S{i}" for i in range(max(0, frame.shape[1] - 3))]
        )
    locations = _standardise_locations(frame, str(path))
    sample_columns = [
        c for c in frame.columns if str(c).lower() not in COORDINATE_NAMES
    ]
    if not sample_columns:
        raise SystemExit(
            f"{path} contains only coordinate columns; no sample columns were "
            f"found. Columns: {list(frame.columns)}"
        )
    matrix = frame[sample_columns].to_numpy(dtype=np.float64)
    return matrix, locations, [
        str(c) for c in sample_columns
    ]


def load_inputs(
    betas_path: Path,
    locations_path: Path,
    *,
    locations_no_header: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Read the matrix and coordinates, validating that they correspond."""

    betas = _read_table(betas_path)
    raw = _read_table(
        locations_path, header=None if locations_no_header else "infer"
    )
    if locations_no_header:
        raw.columns = ["chr", "start", "end"][: raw.shape[1]] + list(
            raw.columns[3:]
        )
    locations = _standardise_locations(raw, "--locations")

    # drop any coordinate columns that leaked into the matrix
    drop = [c for c in betas.columns if str(c).lower() in COORDINATE_NAMES]
    if drop:
        betas = betas.drop(columns=drop)

    if len(betas) != len(locations):
        raise SystemExit(
            f"--betas has {len(betas)} rows and --locations has "
            f"{len(locations)}; they must correspond row for row."
        )

    matrix = betas.to_numpy(dtype=np.float64)
    return _validate(matrix, locations, [str(c) for c in betas.columns])


def _validate(
    matrix: np.ndarray, locations: pd.DataFrame, samples: list[str]
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    if np.nanmin(matrix) < -1e-9 or np.nanmax(matrix) > 1 + 1e-9:
        raise SystemExit(
            "--betas must contain methylation proportions in [0, 1]; values "
            f"range from {np.nanmin(matrix):.3f} to {np.nanmax(matrix):.3f}. "
            "Percentages should be divided by 100."
        )
    # No longer rejects input that isn't pre-sorted (spec §2b prerequisite):
    # the previous check sorted by chromosome as a STRING
    # (np.lexsort(...chr...)), so natural order (chr1, chr2, ..., chr10)
    # failed it -- confirmed on the repo's own real hg38 fixture
    # (chr1, chr7, chr19). Both callers (call_dmrs, run_legacy) now group by
    # chromosome in canonical order and sort ascending within each
    # chromosome themselves, so any input order is accepted here.
    return matrix, locations[["chr", "start", "end"]], samples


# --------------------------------------------------------------------------
# the published method, driven from processing.py
# --------------------------------------------------------------------------


def run_legacy(
    matrix: np.ndarray,
    locations: pd.DataFrame,
    sample_names: list[str],
    *,
    cov_threshold: float,
    delta_threshold: float,
    min_cpgs: int,
    impute: bool,
    fdr_replicates: int = 0,
    target_fdr: float = 0.05,
    min_prevalence: int = 0,
    seed: int = 1,
    fdr_decision: str = "point",
) -> tuple[pd.DataFrame, dict]:
    """The originally published caller, invoked from ``processing.py``.

    ``fdr_replicates > 0`` is not supported: the permutation null it would
    annotate with (``qvalue_result_from_track``) always raises, since it
    depends on ``permutation_null_tracks``, which was retired. Fails fast,
    before running discovery, rather than letting the run fail partway
    through on data-dependent timing (whether any block happened to be
    found). Discovery itself (``fdr_replicates=0``, the default) is
    untouched: the regions are exactly those the published method reports.
    """

    if fdr_replicates > 0:
        print(
            "[eiger2] error: --legacy --fdr-replicates is not supported -- "
            "the permutation null it depended on was retired (see "
            "qvalue_result_from_track's docstring for the reason). Drop "
            "--legacy for a calibrated region_qvalue via the revised "
            "caller, or omit --fdr-replicates to run --legacy for "
            "discovery only.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # Spec §2b prerequisite: group by chromosome in canonical order, sort
    # ascending within each chromosome, regardless of the order the caller
    # presented rows in -- the published method (below) infers genomic
    # proximity from row-index proximity the same way the revised caller's
    # own null construction does, so it depends on this too.
    permutation, chromosome_order = chromosome_sort_permutation(locations)
    matrix = matrix[permutation]
    locations = locations.iloc[permutation].reset_index(drop=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from functions import processing as original

    work = matrix
    if impute:
        work = np.array(matrix, dtype=float, copy=True)
        with np.errstate(invalid="ignore"):
            row_means = np.nanmean(work, axis=1)
        row_means = np.where(np.isfinite(row_means), row_means, 0.5)
        gaps = ~np.isfinite(work)
        work[gaps] = np.take(row_means, np.nonzero(gaps)[0])

    statistic = original.covariance_blocks(work, pairwise=False)
    blocks = original.extract_blocks(
        statistic, DMR_length_threshold=min_cpgs, cov_threshold=cov_threshold
    )

    # fdr_replicates > 0 already exited above; region_qvalue is always
    # unavailable through this caller now.
    fdr_result, fdr_diagnostics = None, None
    qvalues = None

    meta = np.full((matrix.shape[1], 5), "NA", dtype=object)
    rows = []
    kept_block = []
    for block_index, (start, stop) in enumerate(blocks):
        span = np.arange(int(start), int(stop) + 1)
        if span.size < 2:
            continue
        kept_block.append(block_index)
        result = original.find_separators(
            work, span, meta, thres=delta_threshold
        )
        means = np.asarray(result.get("averages"), dtype=float)
        order = np.argsort(means)
        gaps = np.diff(means[order])
        cut = int(np.argmax(gaps)) if gaps.size else 0
        low, high = order[: cut + 1], order[cut + 1 :]
        cluster_delta = (
            float(abs(means[low].mean() - means[high].mean()))
            if low.size and high.size else 0.0
        )
        minor = low if low.size <= high.size else high
        first, last = int(span[0]), int(span[-1])
        rows.append({
            "region_id": (
                f"EIGER1_{locations['chr'].iloc[first]}_"
                f"{int(locations['start'].iloc[first])}_"
                f"{int(locations['end'].iloc[last])}"
            ),
            "chr": str(locations["chr"].iloc[first]),
            "start": int(locations["start"].iloc[first]),
            "end": int(locations["end"].iloc[last]),
            "n_cpg": last - first + 1,
            "cpg_index_start": first,
            "cpg_index_end": last,
            "statistic": "covariance",
            "statistic_mean": float(np.nanmean(statistic[first:last]))
            if last > first else np.nan,
            "statistic_threshold": cov_threshold,
            "region_qvalue": (
                float(qvalues[block_index]) if qvalues is not None else np.nan
            ),
            "region_tail_fdr": (
                float(fdr_result.tail_fdr[block_index])
                if fdr_result is not None else np.nan
            ),
            "region_qvalue_lower_95": (
                float(fdr_result.qvalue_lower[block_index])
                if fdr_result is not None else np.nan
            ),
            "region_qvalue_upper_95": (
                float(fdr_result.qvalue_upper[block_index])
                if fdr_result is not None else np.nan
            ),
            "region_qvalue_resolution": (
                float(fdr_result.qvalue_resolution[block_index])
                if fdr_result is not None else np.nan
            ),
            "null_exceedances_at_cutoff": (
                int(fdr_result.null_exceedances[block_index])
                if fdr_result is not None else -1
            ),
            "region_qvalue_cutoff_rank": (
                int(fdr_result.qvalue_cutoff_rank[block_index])
                if fdr_result is not None else -1
            ),
            "minority_size_descriptive": int(minor.size),
            "delta_consecutive_gap_descriptive": float(
                result.get("delta_minmax", 0.0) or 0.0
            ),
            "delta_cluster_mean_descriptive": cluster_delta,
            "direction_descriptive": result.get("direction"),
            "classification_descriptive": result.get("classification"),
            "minor_group_size_descriptive": int(minor.size),
            "minor_group_samples_descriptive": ";".join(
                sample_names[i] for i in sorted(minor.tolist())
            ),
            "pvalue_descriptive": float(result.get("pvalue_sample", np.nan)),
        })

    frame = pd.DataFrame(rows)
    if len(frame):
        frame["passes_prevalence"] = (
            frame["minority_size_descriptive"] >= min_prevalence
        )
        # No "significant" column: qvalues is always None through this
        # caller now (fdr_replicates > 0 already exited above), so there is
        # never a decision to report one from.
    manifest = {
        "caller": "legacy",
        "source": "Python/functions/processing.py",
        "cov_threshold": cov_threshold,
        "delta_threshold": delta_threshold,
        "min_cpgs": min_cpgs,
        "mean_imputed": impute,
        "n_blocks_before_effect_gate": len(blocks),
        "n_regions": len(frame),
        "fdr": fdr_diagnostics,
        "chromosome_order": chromosome_order,
        "schema_version": "2.3",
        "fdr_decision": fdr_decision,
        "note": (
            "Discovery is exactly the published method. Where --fdr-replicates "
            "is set, region_qvalue is an annotation obtained by permuting the "
            "caller's own covariance track within chromosome and segmenting the "
            "result identically; no region is added or removed by it. Both "
            "delta criteria remain descriptive and neither filters."
        ),
    }
    return frame, manifest


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _config_from_args(args: argparse.Namespace) -> Eiger2Config:
    names = {f.name for f in fields(Eiger2Config)}
    supplied = {
        k: v for k, v in vars(args).items() if k in names and v is not None
    }
    return Eiger2Config(**supplied)


def _require_default_confidence_level(regions_confidence_level: float | None) -> None:
    """Refuse to write ``regions.csv`` at a non-default confidence level.

    ``confidence_level`` is a ``region_fdr_result`` keyword-only parameter, not an
    ``Eiger2Config`` field, so no CLI flag can currently set it away from 0.95 --
    this guard exists so that a future stage adding such a flag cannot silently
    ship a CSV whose ``*_95`` suffixed columns lie about their own confidence level.
    """

    if regions_confidence_level is not None and regions_confidence_level != 0.95:
        raise ValueError(
            f"regions.csv export is fixed at the 95% confidence level; the "
            f"requested level {regions_confidence_level} is a Python-API-only "
            f"research setting and cannot be written to CSV. Generalised suffixes "
            f"belong to a later schema version."
        )


def _json_safe(value):
    """Replace NaN/Inf floats with None, recursively.

    ``null_exchangeability`` can return NaN for p_null/ratio/calibration_threshold
    when a null replicate's holdout half is empty (only reachable at the
    edge case null_replicates=1); ``json.dumps(..., default=str)`` otherwise
    writes those out as a bare, non-standard ``NaN`` token, which is not valid
    strict JSON and breaks any downstream `json.load`.
    """

    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_manifest(out: Path, payload: dict) -> None:
    (out / "run_manifest.json").write_text(
        json.dumps(_json_safe(payload), indent=2, default=str)
    )


def _resolve_inputs(args: argparse.Namespace):
    """Accept either ``--matrix`` or ``--betas`` + ``--locations``."""

    if args.matrix:
        if args.betas or args.locations:
            raise SystemExit(
                "--matrix supplies both coordinates and samples; do not also "
                "pass --betas or --locations."
            )
        path = Path(args.matrix)
        matrix, locations, samples = load_matrix_file(
            path, no_header=args.locations_no_header
        )
        return (*_validate(matrix, locations, samples), [path])
    if not (args.betas and args.locations):
        raise SystemExit(
            "Supply either --matrix (coordinates and samples in one file) or "
            "both --betas and --locations."
        )
    betas_path, locations_path = Path(args.betas), Path(args.locations)
    return (
        *load_inputs(betas_path, locations_path,
                     locations_no_header=args.locations_no_header),
        [betas_path, locations_path],
    )


def _format_structural_gate(name: str, gate: dict) -> str:
    """One human-readable line per spec §2a.2 structural gate, naming its
    measured value -- used by both the blocked CLI summary and its tests."""

    if name == "surviving_null_tracks":
        return (
            f"surviving_null_tracks: {gate['surviving']}/{gate['null_replicates']} "
            f"({gate['fraction']:.1%}) below the {gate['floor_fraction']:.0%} floor"
        )
    if name == "null_coverage":
        return (
            f"null_coverage: {gate['value']:.1%} below the {gate['floor']:.1%} floor"
        )
    if name == "null_construction":
        return (
            f"null_construction: {gate['value']} pooled null values below the "
            f"{gate['floor']} floor"
        )
    if name == "numerical_failure":
        return f"numerical_failure: {gate.get('error')}"
    if name in {"chromosome_exposure", "region_null_chromosome_exposure"}:
        label = (
            "member-step region-null chromosome exposure"
            if name == "region_null_chromosome_exposure"
            else "edge-null chromosome exposure"
        )
        return (
            f"{label}: {gate['missing_exposure']:.1%} of eligible "
            f"edges have no feasible null lag (chromosomes: "
            f"{', '.join(gate['infeasible_chromosomes'])}), above the "
            f"{gate['floor']:.1%} floor"
        )
    return f"{name}: {gate}"


def command_call(args: argparse.Namespace) -> int:
    # Capture provenance before creating/writing the output directory so an
    # in-repository output path cannot make an otherwise clean implementation
    # appear dirty, or contaminate the source-diff fingerprint with this run's
    # own generated files.
    git_provenance = _git_provenance()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    matrix, locations, samples, input_paths = _resolve_inputs(args)

    # The coverage filter is applied here rather than inside the callers so that
    # it governs BOTH. It matters most for the legacy path: mean imputation
    # replaces a CpG unobserved in every sample with the fallback value 0.5,
    # inventing a constant row that then breaks regions. Filtering first means
    # nothing is imputed where nothing was measured.
    kept = np.arange(matrix.shape[0])
    if args.min_sample_fraction:
        observed = np.sum(np.isfinite(matrix), axis=1)
        mask = observed >= np.ceil(args.min_sample_fraction * matrix.shape[1])
        kept = np.flatnonzero(mask)
        dropped_total = int(matrix.shape[0] - kept.size)
        matrix = matrix[mask]
        locations = locations.loc[mask].reset_index(drop=True)
        print(f"[eiger2] filtered   : {dropped_total:,} of "
              f"{dropped_total + kept.size:,} CpGs below "
              f"--min-sample-fraction {args.min_sample_fraction}")

    started = time.perf_counter()
    if args.legacy:
        regions, caller_manifest = run_legacy(
            matrix, locations, samples,
            cov_threshold=args.cov_threshold,
            delta_threshold=args.delta_threshold,
            min_cpgs=args.min_cpgs if args.min_cpgs is not None else 5,
            impute=not args.no_impute,
            fdr_replicates=args.fdr_replicates,
            target_fdr=args.target_fdr,
            min_prevalence=args.min_prevalence_samples or 0,
            seed=args.random_seed,
            fdr_decision=args.fdr_decision,
        )
        values = pd.DataFrame()
    else:
        config = _config_from_args(args)
        # Stage 1f Task 1c (pre-release, master spec pre-release): report the
        # region_null_mode="member_step" minimum-size floor before the
        # pipeline runs, not only after a fallback/raise -- so the constraint
        # is visible before a long run, not buried in output at the end.
        min_n_cpg = member_step_min_n_cpg(config)
        print(
            f"[eiger2] region null: region_null_mode={config.region_null_mode!r}, "
            f"input has {matrix.shape[0]:,} CpGs, member_step needs >= "
            f"{min_n_cpg:,}"
        )
        if config.region_null_mode != "off":
            print(
                f"[eiger2] WARNING: region_null_mode={config.region_null_mode!r} "
                "calibrates region_qvalue against the locality-breaking null, "
                "which is misspecified on WGBS backgrounds -- see "
                "docs/FDR.md section 18. The default is 'off': ranked "
                "candidates, no genome-background FDR.",
                file=sys.stderr,
            )
        regions, values, caller_manifest = call_dmrs(
            matrix, locations, sample_names=samples, config=config
        )
    elapsed = time.perf_counter() - started

    # Region coordinates are genomic and therefore already correct, but the CpG
    # indices refer to the filtered matrix; map them back to the input rows so
    # that outputs from filtered and unfiltered runs are directly comparable.
    if args.min_sample_fraction and len(regions):
        for column in ("cpg_index_start", "cpg_index_end"):
            if column in regions.columns:
                regions[column] = kept[regions[column].to_numpy(dtype=int)]

    _require_default_confidence_level(None if args.legacy else 0.95)
    if args.legacy:
        regions_for_output = regions
    else:
        support_diagnostics = regions.attrs.get("region_support_diagnostics")
        regions_for_output = main_region_table(regions)
        if support_diagnostics is not None:
            support_diagnostics.to_csv(
                out / "region_support_diagnostics.csv", index=False
            )
    regions_for_output.to_csv(out / "regions.csv", index=False)
    if len(values):
        values.to_csv(out / "region_sample_means.csv", index=False)

    warnings_list: list[str] = []
    fdr_diag = caller_manifest.get("fdr") or {}
    # Spec §2c: null-opportunity adequacy is a per-LENGTH-stratum check now,
    # not a single aggregate one -- a refused stratum's rows are suppressed,
    # every other stratum reports normally (run status `partial_inference`,
    # not `inference_unreliable`). A per-stratum empty-but-ADEQUATE pool is a
    # separate, non-refusing case reported in zero_exceedance_strata below.
    for entry in fdr_diag.get("insufficient_null_opportunity_strata") or []:
        required = entry.get("required_null_opportunity")
        region_null_mode = (caller_manifest.get("gates") or {}).get(
            "region_null_mode"
        )
        replicate_option = (
            "--n-lag-draws"
            if region_null_mode == "member_step" else "--null-replicates"
        )
        required_note = (
            f"required {required:.1f}; increase {replicate_option} to add "
            f"opportunity without weakening m_opportunity"
            if required is not None
            else "observed opportunity is zero, so no null volume clears it"
        )
        warnings_list.append(
            f"stratum {entry['stratum']!r}: region_qvalue refused "
            f"({entry['observed_regions']} observed regions, null opportunity "
            f"{entry['null_opportunity']:.1f} vs. {required_note}, "
            f"m_opportunity={entry['m_opportunity']})"
        )
    zero_exceedance_strata = fdr_diag.get("zero_exceedance_strata")
    if zero_exceedance_strata:
        for entry in zero_exceedance_strata:
            warnings_list.append(
                f"stratum {entry['stratum']!r}: null pool is empty (0 pooled "
                f"regions across {entry['replicates']} replicates) -- "
                f"region_qvalue is reported with a finite positive K=0 "
                f"replicate-cluster upper bound, "
                f"not refused"
            )
    for entry in fdr_diag.get("insufficient_cluster_replicates_strata") or []:
        warnings_list.append(
            f"stratum {entry['stratum']!r}: region_qvalue refused because "
            f"{entry['clusters']} independent null clusters are below the "
            f"fixed method minimum of {entry['minimum_clusters']}"
        )
    for entry in fdr_diag.get("insufficient_scale_opportunity_strata") or []:
        warnings_list.append(
            f"stratum {entry['stratum']!r}: region_qvalue refused because "
            f"its observed/null opportunity scale is unavailable"
        )
    fallback_scale_strata = fdr_diag.get("fdr_scale_fallback_strata") or []
    if fallback_scale_strata:
        warnings_list.append(
            "global edge scaling retained for FDR stratum/strata without "
            f"measured matching opportunity: {fallback_scale_strata}"
        )
    collapsed_strata = caller_manifest.get("layout", {}).get(
        "collapsed_length_strata"
    )
    if collapsed_strata:
        ceiling = caller_manifest["layout"]["layout_max_detectable_region_cpgs"]
        warnings_list.append(
            f"layout_max_detectable_region_cpgs={ceiling} cannot populate length "
            f"stratum/strata {collapsed_strata} -- no eligible run in this input is "
            f"long enough, regardless of what was called"
        )
    acf_gate_result = caller_manifest.get("gates", {}).get("acf")
    if acf_gate_result is not None and not acf_gate_result.get("passed", True):
        bad_lags = [
            lag for lag, d in acf_gate_result["diff"].items()
            if not (np.isfinite(d) and abs(d) <= acf_gate_result["tolerance"])
        ]
        warnings_list.append(
            f"ACF calibration diagnostic differed at lag(s) {bad_lags} "
            f"of {acf_gate_result['lags']}: null autocorrelation does not match "
            f"observed within {acf_gate_result['tolerance']}; this is not a "
            f"hard gate on a signal-rich cohort"
        )
    region_exch = caller_manifest.get("gates", {}).get("region_exchangeability")
    if region_exch is not None and not region_exch.get("skipped"):
        ratio = region_exch["region_exchangeability_ratio"]
        if not math.isfinite(ratio):
            warnings_list.append(
                f"Gate 2 (region-level exchangeability, master spec §6.2): "
                f"null_rate=0 (region_exchangeability_ratio is inf or undefined) "
                f"-- {region_exch['null_regions_total']} pooled null regions "
                f"across this call, null_count_adequate="
                f"{region_exch['null_count_adequate']}"
            )

    # Spec §0.3.4: record every derived seed. core.py derives its null-track
    # RNGs from `random_seed` via fixed, documented additive offsets rather
    # than by drawing a fresh independent seed per replicate: `null_tracks`
    # (the fixed-window "lag" null, core.py) uses `random_seed + 977`, and
    # `permutation_null_tracks` uses `seed + 5209`. What is recorded below is
    # that offset CONSTANT actually applied on this run -- not an enumerated
    # list of per-replicate seeds, since replicates share one RNG stream
    # seeded once from the offset rather than each having their own derived
    # seed. The legacy caller no longer touches any RNG at all:
    # `fdr_replicates > 0`, the only path that used to reach
    # `permutation_null_tracks` via `qvalue_result_from_track`, now exits
    # before discovery runs -- see `run_legacy`'s docstring.
    if args.legacy:
        derived_seeds = {}
    elif _config_from_args(args).null_mode == "permutation":
        derived_seeds = {"permutation_null_track_offset": 5209}
    else:
        derived_seeds = {"lag_null_track_offset": 977}
        if (caller_manifest.get("gates") or {}).get(
            "region_null_mode"
        ) == "member_step":
            derived_seeds["member_step_lag_draw_offset"] = 61_093

    envelope = build_manifest_envelope(
        config_dict=(
            _config_from_args(args).to_dict() if not args.legacy else {}
        ),
        git_commit=str(git_provenance["git_commit"]),
        git_worktree_dirty=git_provenance["git_worktree_dirty"],
        git_worktree_fingerprint_sha256=git_provenance[
            "git_worktree_fingerprint_sha256"
        ],
        input_checksums={
            str(p): _checksum(p) for p in input_paths
        },
        random_seed=args.random_seed,
        derived_seeds=derived_seeds,
        runtime_seconds=elapsed,
        peak_memory_mb=_peak_memory_mb(),
        warnings=warnings_list,
        fdr_decision=args.fdr_decision,
    )
    manifest = {
        **envelope,
        "eiger_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "caller": "legacy" if args.legacy else "eiger2",
        "inputs": {
            "files": [
                {"path": str(p), "sha256_16": _checksum(p)} for p in input_paths
            ],
            "n_cpg": int(matrix.shape[0]),
            "n_samples": int(matrix.shape[1]),
            "missing_fraction": float(np.mean(~np.isfinite(matrix))),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "caller_manifest": caller_manifest,
    }
    _write_manifest(out, manifest)

    significant = (
        int(regions["significant"].sum())
        if "significant" in regions.columns and len(regions) else None
    )
    dropped = caller_manifest.get("n_cpg_dropped_low_coverage")
    if dropped:
        print(f"[eiger2] filtered   : {dropped:,} CpGs below "
              f"--min-sample-fraction, "
              f"{caller_manifest.get('n_cpg_dropped_unobserved', 0):,} "
              f"unobserved in every sample")
    print(f"[eiger2] caller     : {'legacy' if args.legacy else 'eiger2'}")
    print(f"[eiger2] regions    : {len(regions)}")
    run_status = caller_manifest.get("status")
    if run_status == "discovery_only":
        # "0 significant" would be a measurement; nothing was measured.
        print(
            "[eiger2] significant: not computed (candidates ranked by "
            "region_area; region FDR needs matched controls)"
        )
    elif significant is not None:
        print(f"[eiger2] significant: {significant} at q <= {args.target_fdr}")
    else:
        print("[eiger2] significant: n/a (legacy defines no error rate)")
    if run_status:
        print(f"[eiger2] status      : {run_status}")
    if run_status == "ok_with_warnings":
        print(
            f"[eiger2] NOTE: {CALIBRATION_DIAGNOSTIC_NOTE}\n"
            f"         See the individual gate warnings below for the "
            f"measured values.",
            file=sys.stderr,
        )
    elif run_status == "discovery_only":
        print(
            f"[eiger2] NOTE: {NO_CALIBRATED_NULL_NOTE}\n"
            f"         The regions above are candidates, ranked by "
            f"region_area (column `rank`).",
            file=sys.stderr,
        )
    elif run_status == "partial_inference":
        print(
            "[eiger2] NOTE: one or more length strata lacked enough null "
            "opportunity to report region_qvalue (spec §2c). Those strata's "
            "rows are refused (NaN / pd.NA); every other stratum reports "
            "normally -- the null is otherwise sound. See the WARNING(s) "
            "below for which stratum/strata and why.",
            file=sys.stderr,
        )
    # Spec §2c: per-LENGTH-stratum null-opportunity refusal -- that stratum's
    # rows only, not the whole run (see region_fdr_result's docstring).
    fdr_diag = caller_manifest.get("fdr") or {}
    # FIX_SPEC §WP2: exactly ONE run-level note when any significant region
    # lands in the bottom length stratum -- the measured miscalibration is a
    # property of that regime, not of an individual row, so it is never
    # emitted per region. Informational only: no column, and no run status,
    # depends on it (see BOTTOM_STRATUM_FDR_NOTE in core.py).
    n_bottom_stratum = fdr_diag.get("n_significant_bottom_stratum") or 0
    if n_bottom_stratum:
        print(
            f"[eiger2] NOTE: {BOTTOM_STRATUM_FDR_NOTE}\n"
            f"         {n_bottom_stratum} of {significant} significant "
            f"region(s) in this run fall in that 4-5 CpG stratum.",
            file=sys.stderr,
        )
    # FIX_SPEC §WP4: exactly ONE run-level note naming how many significant
    # regions are carried by one or two samples. The measured alternative --
    # shipping min_prevalence_samples=3 -- cost up to 80 recall points in the
    # two-carrier-floor signal_grid cells, so the count is reported and the
    # filter left to the user (see LOW_PREVALENCE_NOTE in core.py, and
    # Eiger2Config.min_prevalence_samples' docstring for the measurement that
    # decided it). Informational only: no column and no run status depends on
    # it.
    n_low_prevalence = caller_manifest.get("n_significant_low_prevalence") or 0
    if n_low_prevalence:
        print(
            f"[eiger2] NOTE: {LOW_PREVALENCE_NOTE}\n"
            f"         {n_low_prevalence} of {significant} significant "
            f"region(s) in this run are carried by "
            f"1-{LOW_PREVALENCE_SAMPLE_CEILING} sample(s).",
            file=sys.stderr,
        )
    # FIX_SPEC §WP6b item 2 (WP6 disposition option E): exactly ONE run-level
    # note when any candidate component's loading concentration lands within
    # DEFLATION_NEAR_CUTOFF_MARGIN of deflation_max_concentration. Read off the
    # latent_structure diagnostics the run already wrote, so it fires whether or
    # not the run went on to call anything -- the deflation decision is upstream
    # of every region. Informational only: no column, no verdict and no run
    # status depends on it (see NEAR_CUTOFF_CONCENTRATION_NOTE in core.py).
    latent_diag = caller_manifest.get("latent_structure") or {}
    near_cutoff = latent_diag.get("near_cutoff_components") or []
    if near_cutoff:
        listed = "; ".join(
            f"component {entry['component']} at "
            f"{entry['loading_concentration']:.4f} ({entry['verdict']})"
            for entry in near_cutoff
        )
        # FIX_SPEC §WP7a item 2: the same note under either deflation rule,
        # worded for the boundary the run actually used. Under the shipped
        # "absolute" rule `deflation_boundary` IS `deflation_max_concentration`,
        # so this line's output is unchanged; under "relative" the boundary is
        # the run's own permutation-null quantile and the absolute wording
        # would misdescribe it.
        relative = latent_diag.get("deflation_rule") == "relative"
        boundary = latent_diag.get(
            "deflation_boundary", latent_diag.get("deflation_max_concentration")
        )
        note = (
            NEAR_BOUNDARY_CONCENTRATION_NOTE if relative
            else NEAR_CUTOFF_CONCENTRATION_NOTE
        )
        print(
            f"[eiger2] NOTE: {note}\n"
            f"         {len(near_cutoff)} of "
            f"{latent_diag.get('n_candidates', len(near_cutoff))} candidate "
            f"component(s) sit within "
            f"{latent_diag.get('deflation_near_cutoff_margin')} of the "
            f"{boundary} {'boundary' if relative else 'cutoff'}: "
            f"{listed}.",
            file=sys.stderr,
        )
    for entry in fdr_diag.get("insufficient_null_opportunity_strata") or []:
        required = entry.get("required_null_opportunity")
        region_null_mode = (caller_manifest.get("gates") or {}).get(
            "region_null_mode"
        )
        replicate_option = (
            "--n-lag-draws"
            if region_null_mode == "member_step" else "--null-replicates"
        )
        required_note = (
            f"required {required:.1f} at m_opportunity={entry['m_opportunity']}; "
            f"increase {replicate_option} to add null opportunity without "
            f"weakening the adequacy criterion."
            if required is not None
            else "observed opportunity for this stratum is zero, so no null "
                 "volume clears it."
        )
        print(
            f"[eiger2] WARNING: stratum {entry['stratum']!r} lacked enough null "
            f"opportunity to report region_qvalue\n"
            f"         ({entry['observed_regions']} observed regions, null "
            f"opportunity {entry['null_opportunity']:.1f}).\n"
            f"         region_qvalue, its bounds, and significant are refused "
            f"(NaN / pd.NA) for this stratum only -- every other stratum "
            f"reports normally.\n"
            f"         {required_note}",
            file=sys.stderr,
        )
    for entry in fdr_diag.get("zero_exceedance_strata") or []:
        print(
            f"[eiger2] NOTE: stratum {entry['stratum']!r} has an empty pooled "
            f"null (0 across {entry['replicates']} replicates)\n"
            f"         -- region_qvalue point estimate is zero, while the "
            f"replicate-cluster upper bound remains finite and positive.\n"
            f"         The null had real opportunity and never produced a "
            f"region this long; zero is not treated as zero uncertainty.",
            file=sys.stderr,
        )
    for entry in fdr_diag.get("insufficient_cluster_replicates_strata") or []:
        print(
            f"[eiger2] WARNING: stratum {entry['stratum']!r} has only "
            f"{entry['clusters']} independent null realisations\n"
            f"         (minimum {entry['minimum_clusters']}) -- region_qvalue, "
            f"its bounds, and significant are refused for this stratum.",
            file=sys.stderr,
        )
    for entry in fdr_diag.get("insufficient_scale_opportunity_strata") or []:
        print(
            f"[eiger2] WARNING: stratum {entry['stratum']!r} has no valid "
            f"observed/null opportunity scale\n"
            f"         -- region_qvalue, its bounds, and significant are "
            f"refused for this stratum.",
            file=sys.stderr,
        )
    fallback_scale_strata = fdr_diag.get("fdr_scale_fallback_strata") or []
    if fallback_scale_strata:
        print(
            "[eiger2] WARNING: matching opportunity was unavailable for FDR "
            f"stratum/strata {fallback_scale_strata}\n"
            "         -- the explicitly recorded global-edge fallback was used; "
            "joint support-bin strata are not claimed to be opportunity-scaled.",
            file=sys.stderr,
        )
    collapsed_strata = (caller_manifest.get("layout") or {}).get(
        "collapsed_length_strata"
    )
    if collapsed_strata:
        ceiling = caller_manifest["layout"]["layout_max_detectable_region_cpgs"]
        print(
            f"[eiger2] WARNING: this layout's longest possible region is "
            f"{ceiling} CpGs\n"
            f"         (layout_max_detectable_region_cpgs) -- length stratum/strata "
            f"{collapsed_strata} can\n"
            f"         never be populated here, however this input is called; "
            f"see manifest['layout'].",
            file=sys.stderr,
        )
    acf_gate_result = (caller_manifest.get("gates") or {}).get("acf")
    if acf_gate_result is not None and not acf_gate_result.get("passed", True):
        tolerance = acf_gate_result["tolerance"]
        bad_lags = [
            lag for lag in acf_gate_result["lags"]
            if not (
                np.isfinite(acf_gate_result["diff"][lag])
                and abs(acf_gate_result["diff"][lag]) <= tolerance
            )
        ]
        print(
            f"[eiger2] WARNING: ACF calibration diagnostic differs\n"
            f"         null autocorrelation does not match the observed track's "
            f"within {tolerance} at lag(s) {bad_lags} of {acf_gate_result['lags']}.\n"
            f"         This is descriptive on a signal-rich cohort, where real "
            f"coordination raises observed ACF; establish calibration on DMR-free "
            f"controls/simulations. See manifest['gates']['acf'].",
            file=sys.stderr,
        )
    region_exch = (caller_manifest.get("gates") or {}).get("region_exchangeability")
    if (
        region_exch is not None
        and not region_exch.get("skipped")
        and not math.isfinite(region_exch["region_exchangeability_ratio"])
    ):
        print(
            f"[eiger2] WARNING: Gate 2 (region-level exchangeability, master spec "
            f"§6.2): null_rate is zero\n"
            f"         ({region_exch['null_regions_total']} pooled null regions; "
            f"null_count_adequate={region_exch['null_count_adequate']}) -- "
            f"region_exchangeability_ratio is inf/undefined, reported explicitly "
            f"rather than hidden.\n"
            f"         Expected on current code at every layout (master spec §6.3 "
            f"baseline); see manifest['gates']['region_exchangeability'].",
            file=sys.stderr,
        )
    exchangeability_ratio = (caller_manifest.get("null") or {}).get(
        "null_exchangeability_ratio"
    )
    if exchangeability_ratio is not None and exchangeability_ratio > 1.25:
        print(
            f"[eiger2] WARNING: observed edges exceed the null threshold "
            f"{exchangeability_ratio:.2f}x more often than the null itself "
            f"does (nominal 1.0x).\n"
            f"         Expected on data with real regional signal; only a "
            f"red flag on genuinely\n"
            f"         DMR-free data, where it means the null doesn't match "
            f"the data's tail shape.\n"
            f"         Check calibration with the paper repository's "
            f"DMR-free benchmark suite\n"
            f"         on DMR-free synthetic data, not by reading this "
            f"single-run number alone.",
            file=sys.stderr,
        )
    warning = caller_manifest.get("null", {}).get("warning") \
        if isinstance(caller_manifest.get("null"), dict) else None
    if warning:
        print(f"[eiger2] WARNING    : {warning}")

    # Spec §2a.4: fail closed on a global structural gate failure -- named,
    # with its measured value, not just a status string buried in the
    # manifest.
    status = caller_manifest.get("status")
    if status == "inference_unreliable":
        structural = (caller_manifest.get("gates") or {}).get("structural") or {}
        failed = {
            name: g for name, g in structural.items() if not g.get("passed", True)
        }
        print(
            "[eiger2] BLOCKED    : inference_unreliable -- the null is invalid "
            "as a whole.\n"
            "         region_qvalue, its bounds, and significant are refused "
            "(NaN / pd.NA) for the whole run;\n"
            "         candidates and descriptive columns are still written.",
            file=sys.stderr,
        )
        for name, gate in failed.items():
            print(f"         - {_format_structural_gate(name, gate)}", file=sys.stderr)
        if not failed:
            # region_fdr_result's own global refusal (e.g. no null replicates
            # available at all), not one of the named structural gates above.
            print(
                f"         - {fdr_diag.get('status', 'null construction invalid')} "
                f"(region_fdr_result diagnostics; see manifest['fdr'])",
                file=sys.stderr,
            )

    print(f"[eiger2] written to : {out}")
    return 1 if status == "inference_unreliable" else 0


def _overlap_summary(
    a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str
) -> pd.DataFrame:
    """Base-pair overlap between two region sets, computed on coordinates.

    Interval-based rather than index-based, so it works on outputs produced by
    separate runs and does not require the two to share a CpG indexing.
    """

    def intervals(frame: pd.DataFrame) -> dict:
        by_chrom: dict = {}
        for chrom, block in frame.groupby("chr"):
            spans = np.sort(
                block[["start", "end"]].to_numpy(dtype=np.int64), axis=0
            )
            order = np.argsort(block["start"].to_numpy())
            spans = block[["start", "end"]].to_numpy(dtype=np.int64)[order]
            merged = []
            for start, end in spans:
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            by_chrom[chrom] = np.array(merged, dtype=np.int64)
        return by_chrom

    ia, ib = intervals(a), intervals(b)
    covered_a = sum(int((v[:, 1] - v[:, 0]).sum()) for v in ia.values())
    covered_b = sum(int((v[:, 1] - v[:, 0]).sum()) for v in ib.values())

    shared = 0
    for chrom, va in ia.items():
        vb = ib.get(chrom)
        if vb is None or not len(va) or not len(vb):
            continue
        i = j = 0
        while i < len(va) and j < len(vb):
            lo = max(va[i, 0], vb[j, 0])
            hi = min(va[i, 1], vb[j, 1])
            if hi > lo:
                shared += int(hi - lo)
            if va[i, 1] < vb[j, 1]:
                i += 1
            else:
                j += 1

    union = covered_a + covered_b - shared
    return pd.DataFrame([{
        f"{label_a}_regions": len(a),
        f"{label_b}_regions": len(b),
        f"{label_a}_bp": covered_a,
        f"{label_b}_bp": covered_b,
        "shared_bp": shared,
        "jaccard_bp": shared / union if union else float("nan"),
        f"{label_b}_bp_recovered_by_{label_a}": (
            shared / covered_b if covered_b else float("nan")
        ),
        f"{label_a}_bp_recovered_by_{label_b}": (
            shared / covered_a if covered_a else float("nan")
        ),
    }])


def command_compare(args: argparse.Namespace) -> int:
    """Run both callers on the same input and tabulate the overlap.

    With ``--regions-a`` and ``--regions-b`` the callers are not re-run and two
    existing outputs are compared directly, which matters when a run takes
    minutes on a genome-scale matrix.
    """

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.regions_a or args.regions_b:
        if not (args.regions_a and args.regions_b):
            raise SystemExit("supply both --regions-a and --regions-b")
        a = pd.read_csv(args.regions_a)
        b = pd.read_csv(args.regions_b)
        # Not a `.notna().any()` guard: that only checked whether ANY row
        # was available, not every row, so a frame mixing available and
        # unavailable rows (e.g. a partial_inference run refusing one length
        # stratum while reporting the rest -- see docs/FDR.md §14) passed
        # the guard and then raised on `.astype(bool)`, which cannot convert
        # the unavailable rows' `pd.NA`. Latent today (spec §2a's
        # suppression is always whole-run, never a mix within one CSV) but
        # reachable as soon as a per-stratum refusal exists.
        # `.astype("boolean")` first because a plain `pd.read_csv` reads the
        # nullable-boolean column back as `object` (mixed Python
        # True/False/NaN, not pandas `BooleanDtype` -- the same round-trip
        # point docs/FDR.md §14.4 documents), and `.fillna` on `object`
        # dtype warns about an implicit downcast. Rows with inference
        # unavailable are excluded from the "significant" subset, the same
        # as `False` -- neither compared caller ever reported them as a hit.
        if "significant" in a.columns:
            a = a[a["significant"].astype("boolean").fillna(False).astype(bool)]
        if "significant" in b.columns:
            b = b[b["significant"].astype("boolean").fillna(False).astype(bool)]
        summary = _overlap_summary(a, b, args.label_a, args.label_b)
        summary.to_csv(out / "comparison_summary.csv", index=False)
        print(summary.round(4).to_string(index=False))
        print(f"[eiger2] written to : {out}")
        return 0
    matrix, locations, samples, _ = _resolve_inputs(args)
    if args.min_sample_fraction:
        observed = np.sum(np.isfinite(matrix), axis=1)
        mask = observed >= np.ceil(args.min_sample_fraction * matrix.shape[1])
        matrix, locations = matrix[mask], locations.loc[mask].reset_index(drop=True)

    config = _config_from_args(args)
    revised, _, revised_manifest = call_dmrs(
        matrix, locations, sample_names=samples, config=config
    )
    legacy, legacy_manifest = run_legacy(
        matrix, locations, samples,
        cov_threshold=args.cov_threshold,
        delta_threshold=args.delta_threshold,
        min_cpgs=args.min_cpgs if args.min_cpgs is not None else 5,
        impute=not args.no_impute,
        fdr_replicates=args.fdr_replicates,
        target_fdr=args.target_fdr,
        min_prevalence=args.min_prevalence_samples or 0,
        seed=args.random_seed,
    )
    revised.to_csv(out / "regions_eiger2.csv", index=False)
    legacy.to_csv(out / "regions_legacy.csv", index=False)

    def mask(frame: pd.DataFrame) -> np.ndarray:
        covered = np.zeros(matrix.shape[0], dtype=bool)
        for _, row in frame.iterrows():
            covered[int(row["cpg_index_start"]) : int(row["cpg_index_end"]) + 1] = True
        return covered

    called = revised[revised["significant"]] if len(revised) else revised
    a, b = mask(called), mask(legacy)
    summary = pd.DataFrame([{
        "eiger2_regions": len(called),
        "legacy_regions": len(legacy),
        "eiger2_cpgs": int(a.sum()),
        "legacy_cpgs": int(b.sum()),
        "shared_cpgs": int((a & b).sum()),
        "jaccard": float((a & b).sum() / max((a | b).sum(), 1)),
        "legacy_cpgs_recovered_by_eiger2": float(
            (a & b).sum() / b.sum()) if b.any() else np.nan,
    }])
    summary.to_csv(out / "comparison_summary.csv", index=False)
    _write_manifest(out, {
        "eiger_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "eiger2": revised_manifest,
        "legacy": legacy_manifest,
    })
    print(summary.to_string(index=False))
    print(f"[eiger2] written to : {out}")
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eiger2",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version=f"eiger {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--matrix",
                       help="single table with coordinate columns AND samples "
                            "(methrix / bsseq layout: chr, start[, strand], "
                            "then one column per sample). Use this instead of "
                            "--betas/--locations.")
        p.add_argument("--betas", help="CpG-by-sample matrix, no coordinates")
        p.add_argument("--locations",
                       help="coordinates for --betas: chr/start[/end], or a "
                            "headerless 3-column BED")
        p.add_argument("--locations-no-header", dest="locations_no_header",
                       action="store_true",
                       help="the coordinate file has no header row (e.g. a "
                            "wgbstools CpG.bed.gz)")
        p.add_argument("--out", required=True, help="output directory")
        # revised caller
        p.add_argument("--statistic", choices=["correlation", "covariance"],
                       default="correlation")
        p.add_argument("--window", type=int, default=None,
                       help="CpGs per co-variation window (default 4; use 1 "
                            "above ~30 samples)")
        p.add_argument("--min-cpgs", dest="min_cpgs", type=int, default=None,
                       help="minimum region length in CpGs")
        p.add_argument("--min-sample-fraction", dest="min_sample_fraction",
                       type=float, default=None,
                       help="drop CpGs observed in fewer than this fraction of "
                            "samples. Recommended for targeted or panel assays, "
                            "where most rows cannot form an edge anyway; try "
                            "0.5. Default 0.0 (keep everything).")
        p.add_argument("--max-gap-edges", dest="max_gap_edges", type=int,
                       default=None)
        p.add_argument("--null-replicates", dest="null_replicates", type=int,
                       default=None, help="null tracks (default 100; use "
                                          "200-300 with --compute-posterior)")
        p.add_argument("--n-lag-draws", dest="n_lag_draws", type=int,
                       default=None,
                       help="member-step region-null replicates (default 75); "
                            "increase this when a member-step run reports "
                            "insufficient per-stratum null opportunity")
        p.add_argument("--null-min-lag", dest="null_min_lag", type=int,
                       default=None)
        p.add_argument("--target-fdr", dest="target_fdr", type=float,
                       default=0.05)
        p.add_argument("--null-mode", dest="null_mode",
                       choices=["lag", "permutation"], default=None,
                       help="edge-level null; 'lag' is the only supported "
                            "value. 'permutation' is accepted here (so the "
                            "informative error names its replacement) but "
                            "raises immediately: it was removed in v2.1.0, "
                            "measured at zero usable null regions -- see "
                            "--region-null-mode for large-cohort saturation, "
                            "which no longer needs a workaround.")
        p.add_argument("--region-null-mode", dest="region_null_mode",
                       choices=["member_step", "lag", "off"], default=None,
                       help="region-level null behind region_qvalue (distinct "
                            "from --null-mode above). Default (unset here) is "
                            "'off': no region null is built, region_qvalue is "
                            "refused with fdr_status='no_calibrated_null', "
                            "and the output is the ranked candidate list. "
                            "'member_step' and 'lag' are calibrated against "
                            "the locality-breaking null, misspecified on WGBS "
                            "backgrounds (docs/FDR.md section 18), and warn "
                            "when requested; 'member_step' raises below its "
                            "minimum size rather than falling back; "
                            "'lag' for small inputs or reproducing historical "
                            "comparisons (fails both region gates, not "
                            "recommended for inference); 'off' to skip the "
                            "region-level null entirely (candidates only, no "
                            "region_qvalue).")
        p.add_argument("--support-binned-threshold", dest="support_binned_threshold",
                       choices=["auto", "on", "off"], default=None,
                       help="condition the null-quantile threshold on local "
                            "measurement support (samples observed at every "
                            "CpG in the window). 'auto' (default) enables it "
                            "only when support varies enough to matter.")
        p.add_argument("--support-bins", dest="support_bins", type=int, default=None)
        p.add_argument("--fdr-decision", dest="fdr_decision",
                       choices=["point", "upper"], default="point",
                       help="'point' (default): significant iff region_qvalue <= "
                            "target-fdr. 'upper': significant iff "
                            "region_qvalue_upper_95 <= target-fdr -- the "
                            "conservative decision under finite-null uncertainty.")
        p.add_argument(
            "--biological-support-policy",
            dest="biological_support_policy",
            choices=["off", "five_class_v1"],
            default=None,
            help="optional post-selection five-class biological support layer. "
                 "Default 'off' preserves discovery runtime; 'five_class_v1' "
                 "writes one biological_class plus a separate diagnostics "
                 "CSV and never changes the FDR.",
        )
        p.add_argument("--min-prevalence-samples", dest="min_prevalence_samples",
                       type=int, default=None,
                       help="require at least this many samples to carry the "
                            "region's deviation. A single aberrant sample "
                            "produces real coordination but is not a DMR. "
                            "Default 0 (off), measured: a floor of 3 is free "
                            "in 10 of 14 signal_grid cells but costs 91-94 "
                            "recall points where the planted subgroup is two "
                            "samples, and raises pooled per-run FDP from "
                            "0.062 to 0.117 -- it removes true calls, not "
                            "false ones. Set 3 only when subgroup-defining "
                            "regions are the estimand.")
        p.add_argument("--fdr-replicates", dest="fdr_replicates", type=int,
                       default=0,
                       help="legacy only: NOT CURRENTLY SUPPORTED (any value "
                            "> 0 exits with an error naming the reason). Was "
                            "meant to attach a permutation q-value; the "
                            "permutation null it depended on was retired.")
        p.add_argument("--random-seed", dest="random_seed", type=int, default=1)
        p.add_argument("--compute-posterior", dest="compute_posterior",
                       action="store_true",
                       help="attach the per-region evidence score; this is a "
                            "ranking, not a calibrated probability")
        # legacy caller
        p.add_argument("--cov-threshold", dest="cov_threshold", type=float,
                       default=0.012,
                       help="legacy fixed covariance threshold (default 0.012)")
        p.add_argument("--delta-threshold", dest="delta_threshold", type=float,
                       default=0.20,
                       help="legacy effect-size threshold, reported only")
        p.add_argument("--no-impute", dest="no_impute", action="store_true",
                       help="legacy: use complete-case covariance, which at "
                            "realistic missingness reports almost nothing")

    call = sub.add_parser("call", help="run one caller")
    add_common(call)
    call.add_argument("--legacy", action="store_true",
                      help="run the originally published method instead")
    call.set_defaults(func=command_call)

    compare = sub.add_parser(
        "compare",
        help="run both callers and compare, or compare two existing outputs",
    )
    add_common(compare)
    compare.add_argument("--regions-a", dest="regions_a",
                         help="an existing regions.csv, to compare without "
                              "re-running either caller")
    compare.add_argument("--regions-b", dest="regions_b",
                         help="a second existing regions.csv")
    compare.add_argument("--label-a", dest="label_a", default="A")
    compare.add_argument("--label-b", dest="label_b", default="B")
    compare.set_defaults(func=command_compare, legacy=False)
    return parser


#: Characters that word processors and some terminals substitute for "--".
_DASH_LOOKALIKES = ("\u2014", "\u2013", "\u2012", "\u2212")


def _normalise_dashes(argv: list[str]) -> list[str]:
    """Rewrite em-dash and en-dash option prefixes to a plain double hyphen.

    Copying a command out of a document or chat window frequently converts
    ``--matrix`` to ``—matrix``. argparse then reports it as an unrecognised
    positional argument, which is an unhelpful way to learn about a typographic
    substitution.
    """

    fixed, substituted = [], []
    for token in argv:
        original = token
        for dash in _DASH_LOOKALIKES:
            if token.startswith(dash):
                token = "--" + token[len(dash):]
                break
        if token != original:
            substituted.append((original, token))
        fixed.append(token)
    if substituted:
        for before, after in substituted:
            print(f"[eiger2] note: read '{before}' as '{after}' "
                  f"(the leading character was an en/em dash, not two hyphens)")
    return fixed


def main(argv: Iterable[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(_normalise_dashes(raw))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
