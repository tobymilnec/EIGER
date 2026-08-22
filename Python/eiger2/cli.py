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
    Eiger2Config,
    call_dmrs,
    check_fdr_decision_vacuity,
    member_step_min_n_cpg,
    qvalue_result_from_track,
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


def _checksum(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        read = 0
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            read += len(chunk)
            if read >= limit:
                digest.update(b"<truncated>")
                break
    return digest.hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


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
    matrix = frame[sample_columns].to_numpy(dtype=np.float32)
    return matrix.astype(np.float64, copy=False), locations, [
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

    matrix = betas.to_numpy(dtype=np.float32).astype(np.float64, copy=False)
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
    order = np.lexsort(
        (locations["start"].to_numpy(), locations["chr"].to_numpy())
    )
    if not np.array_equal(order, np.arange(len(locations))):
        raise SystemExit(
            "Coordinates must be sorted by chromosome then start position; "
            "EIGER segments along the row order of the matrix. Sort both the "
            "matrix and the coordinates together before running."
        )
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

    With ``fdr_replicates > 0`` a region-level empirical false discovery rate is
    attached by permuting the caller's own covariance track. Discovery is
    untouched: the regions are exactly those the published method reports, and
    the q-value is an annotation on them.
    """

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

    fdr_result, fdr_diagnostics = None, None
    if fdr_replicates > 0 and len(blocks):
        chrom = locations["chr"].to_numpy()[: statistic.size]
        fdr_result, fdr_diagnostics = qvalue_result_from_track(
            statistic, np.asarray(blocks, dtype=int), cov_threshold,
            eligible=np.isfinite(statistic), chromosome=chrom,
            min_cpgs=min_cpgs, max_gap_edges=0,
            replicates=fdr_replicates, seed=seed,
        )
        if fdr_decision == "upper":
            # Spec §1.8 frames this guard as running "before computing
            # regions"; it actually runs here, after qvalue_result_from_track,
            # because `scale` and `m_bonferroni` (both required by the guard)
            # aren't known until that call completes. This is intentional:
            # the guard still fires before any region is reported to the
            # user as significant, which is the actual intent behind the
            # spec's wording (never silently return zero significant
            # regions).
            check_fdr_decision_vacuity(
                fdr_result,
                scale=fdr_diagnostics.get("scale", 0.0),
                current_replicates=fdr_replicates,
                target_fdr=target_fdr,
            )
    qvalues = fdr_result.qvalue if fdr_result is not None else None

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
        if qvalues is not None:
            decision_series = (
                frame["region_qvalue_upper_95"] if fdr_decision == "upper"
                else frame["region_qvalue"]
            )
            frame["significant"] = (
                (decision_series <= target_fdr) & frame["passes_prevalence"]
            )
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
        "schema_version": "2.1",
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


def command_call(args: argparse.Namespace) -> int:
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
    regions.to_csv(out / "regions.csv", index=False)
    if len(values):
        values.to_csv(out / "region_sample_means.csv", index=False)

    warnings_list: list[str] = []
    # Stage 1f Task 1b (pre-release, master spec pre-release): region_null_mode
    # left at its default "auto" fell back to "lag" because the input was
    # below member_step's minimum size. Loud and immediate (printed to stderr
    # below, not only recorded in the manifest), naming the input size, the
    # required size, and that "lag" fails both region gates and is not
    # calibrated -- the whole point of the fallback is that this is visible in
    # the output, not merely non-fatal.
    fallback = caller_manifest.get("gates", {}).get("region_null_mode_fallback")
    if fallback:
        warnings_list.append(
            f"region_null_mode='auto' fell back to 'lag': input has "
            f"{fallback['n_cpg']:,} CpGs, below member_step's minimum of "
            f"{fallback['n_cpg_required']:,}. 'lag' fails both region-level "
            f"exchangeability gates on real layouts and its region_qvalue is "
            f"not calibrated -- treat it as candidates-only. Pass "
            f"--region-null-mode member_step to raise instead of falling "
            f"back, or --region-null-mode off to skip region_qvalue entirely."
        )
    fdr_diag = caller_manifest.get("fdr") or {}
    if fdr_diag.get("null_count_overdispersion_warning"):
        ratio = fdr_diag["null_count_overdispersion_warning"]
        warnings_list.append(
            f"pooled null region counts are overdispersed (variance/mean="
            f"{ratio:.2f})"
        )
    # Stage 1e Task 1 (master spec §8): the null-adequacy floor is now a single
    # AGGREGATE check on the run as a whole (see region_fdr_result's
    # docstring); a per-stratum empty pool is reported in
    # zero_exceedance_strata, not refused, so it is an informational note here,
    # not a "refused" warning.
    fdr_status = fdr_diag.get("status")
    if fdr_status == "aggregate null inadequate":
        required = fdr_diag.get("replicates_required")
        required_note = (
            f"; ~{required} replicates needed to reach the floor at the "
            f"observed rate"
            if required is not None
            else "; observed rate is zero, so no replicate count reaches "
                 "the floor"
        )
        warnings_list.append(
            f"region_qvalue refused for the whole run: "
            f"{fdr_diag.get('pooled_null_regions')} pooled null regions across "
            f"{fdr_diag.get('replicates')} replicates, below "
            f"min_null_regions_for_fdr{required_note}"
        )
    zero_exceedance_strata = fdr_diag.get("zero_exceedance_strata")
    if zero_exceedance_strata:
        for entry in zero_exceedance_strata:
            warnings_list.append(
                f"stratum {entry['stratum']!r}: null pool is empty (0 pooled "
                f"regions across {entry['replicates']} replicates) -- "
                f"region_qvalue is reported at the finite K=0 Garwood bound, "
                f"not refused"
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
    if acf_gate_result is not None and not acf_gate_result["passed"]:
        bad_lags = [
            lag for lag, d in acf_gate_result["diff"].items()
            if not (np.isfinite(d) and abs(d) <= acf_gate_result["tolerance"])
        ]
        warnings_list.append(
            f"Gate 1 (ACF match, master spec §6.2) failed at lag(s) {bad_lags} "
            f"of {acf_gate_result['lags']}: null autocorrelation does not match "
            f"observed within {acf_gate_result['tolerance']}"
        )
    region_exch = caller_manifest.get("gates", {}).get("region_exchangeability")
    if region_exch is not None:
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
    # `permutation_null_tracks` (used by both the "permutation" null_mode and
    # unconditionally by the legacy caller's `qvalue_result_from_track`) uses
    # `seed + 5209`. What is recorded below is that offset CONSTANT actually
    # applied on this run -- not an enumerated list of per-replicate seeds,
    # since replicates share one RNG stream seeded once from the offset
    # rather than each having their own derived seed.
    if args.legacy or _config_from_args(args).null_mode == "permutation":
        derived_seeds = {"permutation_null_track_offset": 5209}
    else:
        derived_seeds = {"lag_null_track_offset": 977}

    envelope = build_manifest_envelope(
        config_dict=(
            _config_from_args(args).to_dict() if not args.legacy else {}
        ),
        git_commit=_git_commit(),
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
    if significant is not None:
        print(f"[eiger2] significant: {significant} at q <= {args.target_fdr}")
    else:
        print("[eiger2] significant: n/a (legacy defines no error rate)")
    # Stage 1f Task 1b (pre-release, master spec pre-release): region_null_mode
    # "auto" fell back to "lag" -- loud and immediate, not buried only in the
    # manifest (STOP 1e §5 flagged the previous behaviour, a hard uncatchable
    # error with no CLI way out, as broken for a public tool).
    fallback = (caller_manifest.get("gates") or {}).get("region_null_mode_fallback")
    if fallback:
        print(
            f"[eiger2] WARNING: region_null_mode='auto' fell back to 'lag'\n"
            f"         (input has {fallback['n_cpg']:,} CpGs, below "
            f"member_step's minimum of {fallback['n_cpg_required']:,}).\n"
            f"         'lag' fails both region-level exchangeability gates on "
            f"real layouts and its region_qvalue is not calibrated -- treat "
            f"it as candidates-only.\n"
            f"         Pass --region-null-mode member_step to raise instead "
            f"of falling back, or --region-null-mode off to skip "
            f"region_qvalue entirely.",
            file=sys.stderr,
        )
    overdispersion_ratio = (caller_manifest.get("fdr") or {}).get(
        "null_count_overdispersion_warning"
    )
    if overdispersion_ratio is not None:
        print(
            f"[eiger2] WARNING: pooled null region counts are overdispersed\n"
            f"         (variance/mean = {overdispersion_ratio:.2f}, Poisson assumes "
            f"1.0).\n"
            f"         region_qvalue_upper_95 is a LOWER BOUND on the true "
            f"uncertainty.",
            file=sys.stderr,
        )
    # Stage 1e Task 1 (master spec §8): a single AGGREGATE null-adequacy
    # refusal for the whole run, not a per-stratum one -- see
    # region_fdr_result's docstring.
    fdr_diag = caller_manifest.get("fdr") or {}
    if fdr_diag.get("status") == "aggregate null inadequate":
        required = fdr_diag.get("replicates_required")
        required_note = (
            f"raise --null-replicates to at least {required} to clear the "
            f"floor at the observed rate."
            if required is not None
            else "the observed rate is zero, so no replicate count reaches "
                 "the floor at these parameters."
        )
        print(
            f"[eiger2] WARNING: too few pooled null regions to report "
            f"region_qvalue\n"
            f"         ({fdr_diag.get('pooled_null_regions')} across "
            f"{fdr_diag.get('replicates')} replicates, below "
            f"min_null_regions_for_fdr).\n"
            f"         region_qvalue, its bounds, and significant are refused "
            f"(NaN / False) for the whole run, not a fabricated zero.\n"
            f"         {required_note}",
            file=sys.stderr,
        )
    for entry in fdr_diag.get("zero_exceedance_strata") or []:
        print(
            f"[eiger2] NOTE: stratum {entry['stratum']!r} has an empty pooled "
            f"null (0 across {entry['replicates']} replicates)\n"
            f"         -- region_qvalue is reported at the finite K=0 Garwood "
            f"bound for this stratum, not refused: the null had real "
            f"opportunity and never produced a region this long.",
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
    if acf_gate_result is not None and not acf_gate_result["passed"]:
        tolerance = acf_gate_result["tolerance"]
        bad_lags = [
            lag for lag in acf_gate_result["lags"]
            if not (
                np.isfinite(acf_gate_result["diff"][lag])
                and abs(acf_gate_result["diff"][lag]) <= tolerance
            )
        ]
        print(
            f"[eiger2] WARNING: Gate 1 (ACF match, master spec §6.2) failed\n"
            f"         null autocorrelation does not match the observed track's "
            f"within {tolerance} at lag(s) {bad_lags} of {acf_gate_result['lags']}.\n"
            f"         Expected on current code at every layout (master spec §6.3 "
            f"baseline); see manifest['gates']['acf'].",
            file=sys.stderr,
        )
    region_exch = (caller_manifest.get("gates") or {}).get("region_exchangeability")
    if region_exch is not None and not math.isfinite(
        region_exch["region_exchangeability_ratio"]
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
            f"         Check calibration properly with "
            f"`benchmarking/benchmark_suite.py exchangeability`\n"
            f"         on DMR-free synthetic data, not by reading this "
            f"single-run number alone.",
            file=sys.stderr,
        )
    warning = caller_manifest.get("null", {}).get("warning") \
        if isinstance(caller_manifest.get("null"), dict) else None
    if warning:
        print(f"[eiger2] WARNING    : {warning}")
    print(f"[eiger2] written to : {out}")
    return 0


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
        if "significant" in a.columns and a["significant"].notna().any():
            a = a[a["significant"].astype(bool)]
        if "significant" in b.columns and b["significant"].notna().any():
            b = b[b["significant"].astype(bool)]
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
                            "'auto': use 'member_step' when the input clears "
                            "its genome-scale minimum size (about 103,000 "
                            "CpGs at defaults), else fall back to 'lag' "
                            "automatically with a warning. Pass 'member_step' "
                            "explicitly to raise instead of falling back; "
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
        p.add_argument("--min-prevalence-samples", dest="min_prevalence_samples",
                       type=int, default=None,
                       help="require at least this many samples to carry the "
                            "region's deviation. A single aberrant sample "
                            "produces real coordination but is not a DMR; try 3.")
        p.add_argument("--fdr-replicates", dest="fdr_replicates", type=int,
                       default=0,
                       help="legacy only: attach a permutation q-value using "
                            "this many null replicates (e.g. 50). Discovery is "
                            "unchanged.")
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
