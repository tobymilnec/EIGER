"""Command-line interface for EIGER.

``eiger2 call``
    Find regions of coordinated methylation and estimate their coordination
    FDR. Every candidate region of the observed scan is written with its
    ``region_score`` and ``coordination_qvalue``; regions with
    ``coordination_qvalue <= --target-fdr`` are selected. The q-values come
    from ``--fdr-replicates`` (default 99) complete rescans of the matrix with
    sample values permuted independently within every CpG row.

``eiger2 call --legacy``
    The same FDR wrapper around the original covariance statistic
    (``functions/processing.py``), with row-mean imputation and a fixed
    threshold of ``--cov-threshold`` (default 0.01).

``eiger2 compare``
    Base-pair overlap between two existing ``regions.csv`` files.

Examples
--------
    eiger2 call --betas betas.tsv --locations cpgs.bed --out results/
    eiger2 call --betas betas.tsv --locations cpgs.bed --out legacy/ --legacy
    eiger2 compare --regions-a results/regions.csv \\
        --regions-b legacy/regions.csv --out comparison/

Inputs
------
``--betas``      CpG-by-sample matrix, TSV/CSV, one row per CpG, proportions in
                 [0, 1]; missing values as empty fields, ``NA`` or ``NaN``.
``--locations``  ``chr``, ``start`` (and optionally ``end``) per CpG, in the
                 same order as ``--betas``; or ``--matrix`` for one table with
                 coordinate columns and sample columns.

Outputs
-------
``regions.csv``        every candidate region: coordinates, CpG count,
                       ``region_score``, ``coordination_qvalue``,
                       ``coordination_selected`` and secondary axis columns.
``discoveries.csv``    the selected subset (same columns).
``permutations.csv``   one row per completed permutation scan.
``run_manifest.json``  resolved settings, seeds, input checksums, counts and
                       per-permutation thresholds.

Axis columns are secondary annotations and never change a primary call. By
default they are descriptive (``axis_effect`` and its uncertainty);
``axis_pvalue``, ``axis_qvalue`` and ``axis_supported`` stay empty unless
``--axis-annotations experimental-inference`` is requested.
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
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .coordination import (
    DEFAULT_FDR_REPLICATES,
    LEGACY,
    NEWER,
    CandidateThresholdError,
    PermutationScanError,
    RegionFdrConfig,
    call_coordinated_regions,
)

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


def _source_checkout() -> Path | None:
    """The EIGER checkout this module runs from, or ``None`` when installed.

    Only a directory whose ``Python/eiger2/cli.py`` is this very file counts.
    An installed package can sit inside some other Git repository (a virtual
    environment created in an analysis project, say); that repository's
    commit and dirty state say nothing about the EIGER code that ran.
    """

    here = Path(__file__).resolve()
    repository = here.parents[2]
    candidate = repository / "Python" / "eiger2" / "cli.py"
    if candidate.is_file() and candidate.resolve() == here:
        return repository
    return None


def _git_commit() -> str:
    """The checkout's HEAD commit, or ``"unknown"`` outside a checkout."""

    repository = _source_checkout()
    if repository is None:
        return "unknown"
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
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

    repository = _source_checkout()
    try:
        if repository is None:
            raise LookupError("not running from an EIGER source checkout")
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
    # Any row order is accepted: the pipeline groups rows by chromosome in
    # canonical order and sorts positions within each chromosome itself.
    return matrix, locations[["chr", "start", "end"]], samples


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# eiger2 call
# --------------------------------------------------------------------------

#: Options of the newer profile, and of the legacy profile, with their defaults.
_NEWER_ONLY = {
    "statistic": NEWER.statistic,
    "window": NEWER.window,
    "null_quantile": NEWER.null_quantile,
    "null_replicates": NEWER.null_replicates,
    "null_min_lag": NEWER.null_min_lag,
}
_LEGACY_ONLY = {"cov_threshold": LEGACY.cov_threshold, "no_impute": False}

_AXIS_MODES = {
    "descriptive": "descriptive",
    "off": "off",
    "experimental-inference": "experimental_inference",
}


def _config_from_args(args: argparse.Namespace) -> RegionFdrConfig:
    """Resolve parsed options to a :class:`RegionFdrConfig`, or exit with a reason."""

    if args.legacy:
        given = [k for k in _NEWER_ONLY if getattr(args, k) is not None]
        if given:
            raise SystemExit(
                "[eiger2] error: "
                + ", ".join("--" + k.replace("_", "-") for k in given)
                + (" applies" if len(given) == 1 else " apply")
                + " to the newer profile only; --legacy uses the covariance "
                "statistic at --cov-threshold"
            )
        profile = LEGACY
        if args.cov_threshold is not None:
            profile = replace(profile, cov_threshold=args.cov_threshold)
        if args.no_impute:
            profile = replace(profile, impute_row_means=False)
    else:
        given = [k for k in _LEGACY_ONLY if getattr(args, k) not in (None, False)]
        if given:
            raise SystemExit(
                "[eiger2] error: "
                + ", ".join("--" + k.replace("_", "-") for k in given)
                + (" applies" if len(given) == 1 else " apply")
                + " to --legacy only"
            )
        overrides = {k: getattr(args, k) for k in _NEWER_ONLY
                     if getattr(args, k) is not None}
        profile = replace(NEWER, **overrides)
    shared = {}
    if args.min_cpgs is not None:
        shared["min_cpgs"] = args.min_cpgs
    if args.max_cpg_gap_bp is not None:
        shared["max_cpg_gap_bp"] = args.max_cpg_gap_bp
    if shared:
        profile = replace(profile, **shared)
    config = RegionFdrConfig(
        profile=profile,
        target_fdr=args.target_fdr,
        fdr_replicates=args.fdr_replicates,
        random_seed=args.random_seed,
        permutation_seed=args.permutation_seed,
        mask_mode=args.mask_mode.replace("-", "_"),
        scan_mode=args.scan_mode,
        storage_dtype=args.storage_dtype,
        min_sample_fraction=args.min_sample_fraction or 0.0,
        discovery_only=args.discovery_only,
        axis_mode=_AXIS_MODES[args.axis_annotations],
        axis_target_fdr=args.target_fdr,
    )
    try:
        config.validate()
    except ValueError as exc:
        raise SystemExit(f"[eiger2] error: {exc}") from None
    return config


def _environment() -> dict:
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, na_rep="NA")


def command_call(args: argparse.Namespace) -> int:
    # Provenance before the output directory exists, so an in-repository output
    # path cannot make the source tree look dirty.
    git_provenance = _git_provenance()
    config = _config_from_args(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    matrix, locations, samples, input_paths = _resolve_inputs(args)
    n_rows, n_samples = matrix.shape
    if config.storage_dtype == "float32":
        # Convert before the call so the parsed float64 copy can be freed.
        matrix = matrix.astype(np.float32)
    say = (lambda _m: None) if args.quiet else (
        lambda m: print(f"[eiger2] {m}", file=sys.stderr, flush=True)
    )

    started = time.perf_counter()
    try:
        run = call_coordinated_regions(
            matrix, locations, sample_names=samples, config=config, progress=say,
        )
    except (CandidateThresholdError, PermutationScanError, ValueError,
            ImportError) as exc:
        print(f"[eiger2] error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    _write_csv(run.regions, out / "regions.csv")
    written = ["regions.csv"]
    if run.manifest["results"]["qvalues_available"] and not args.no_discoveries_table:
        _write_csv(run.discoveries, out / "discoveries.csv")
        written.append("discoveries.csv")
    _write_csv(run.permutations, out / "permutations.csv")
    written.append("permutations.csv")
    if args.save_null_scores and run.null_scores:
        np.savez_compressed(
            out / "null_region_scores.npz",
            **{f"replicate_{i + 1:04d}": s for i, s in enumerate(run.null_scores)},
        )
        written.append("null_region_scores.npz")
    written.append("run_manifest.json")

    manifest = {
        "run_schema": run.manifest["run_schema"],
        "software_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "source": {
            **git_provenance,
            "package_path": str(Path(__file__).resolve().parent),
        },
        "inputs": {
            "files": [{"path": str(p), "sha256": _sha256(p)} for p in input_paths],
            "n_rows": int(n_rows),
            "n_samples": int(n_samples),
        },
        "resolved_configuration": config.to_dict(),
        "environment": _environment(),
        "outputs": written,
        "runtime_seconds": elapsed,
        **{k: v for k, v in run.manifest.items()
           if k not in ("run_schema", "software_version")},
    }
    _write_manifest(out, manifest)

    results = run.manifest["results"]
    status = run.manifest["status"]
    print(f"[eiger2] caller     : {config.profile.name}")
    print(f"[eiger2] candidates : {results['n_candidates']}")
    if status == "discovery_only":
        print("[eiger2] selected   : not computed (--discovery-only: no "
              "permutation scans, coordination_qvalue is empty)")
    else:
        print(f"[eiger2] selected   : {results['n_selected']} at "
              f"coordination_qvalue <= {config.target_fdr} "
              f"({run.manifest['permutations']['completed']} permutation scans)")
    print(f"[eiger2] status     : {status}")
    axis = run.manifest["axis"]
    if config.axis_mode != "off":
        print(f"[eiger2] axes       : {axis.get('n_axes')} "
              f"({axis['inference_label']})")
    print(f"[eiger2] written to : {out}")
    return 0


# --------------------------------------------------------------------------
# eiger2 compare
# --------------------------------------------------------------------------


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


def _selected_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows a caller reported as selected; unavailable counts as not selected."""

    for column in ("coordination_selected", "significant"):
        if column in frame.columns:
            flags = frame[column].astype("boolean").fillna(False).astype(bool)
            return frame[flags.to_numpy()]
    return frame


def command_compare(args: argparse.Namespace) -> int:
    """Overlap of the selected regions of two existing ``regions.csv`` files."""

    if not (args.regions_a and args.regions_b):
        raise SystemExit(
            "[eiger2] error: compare needs --regions-a and --regions-b (two "
            "existing regions.csv files). To compare the callers, run "
            "`eiger2 call` once with and once without --legacy."
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(args.regions_a)
    b = pd.read_csv(args.regions_b)
    if not args.all_candidates:
        a, b = _selected_rows(a), _selected_rows(b)
    summary = _overlap_summary(a, b, args.label_a, args.label_b)
    summary.to_csv(out / "comparison_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"[eiger2] written to : {out}")
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

#: Historical options and the reason each is no longer accepted.
RETIRED_OPTIONS: dict[str, str] = {
    "--null-mode": "there is one candidate-threshold null (the distant-lag "
                   "quantile); remove --null-mode",
    "--region-null-mode": "the member-step/lag region nulls were replaced by the "
                          "per-CpG permutation FDR; set its size with "
                          "--fdr-replicates (default 99)",
    "--n-lag-draws": "the member-step region null was removed; the permutation "
                     "FDR is sized with --fdr-replicates",
    "--edge-threshold": "the newer profile's candidate threshold is always the "
                        "distant-lag quantile; observed-quantile thresholds were "
                        "removed",
    "--support-binned-threshold": "support-binned candidate thresholds were "
                                  "removed from the supported method",
    "--support-bins": "support-binned candidate thresholds were removed from the "
                      "supported method",
    "--fdr-decision": "the only decision rule is coordination_qvalue <= "
                      "--target-fdr; there is no upper-bound decision",
    "--no-confirm": "the dip/split confirmation tier is not part of the "
                    "supported method; use --axis-annotations off to skip the "
                    "secondary axis columns",
    "--biological-support-policy": "the five-class biological support layer is "
                                   "not part of the supported method",
    "--min-prevalence-samples": "prevalence filters no longer change primary "
                                "calls; filter regions.csv downstream if needed",
    "--compute-posterior": "posterior/local-FDR scores were removed; use "
                           "coordination_qvalue",
    "--delta-threshold": "the legacy delta/effect criteria are no longer "
                         "computed; --legacy runs the covariance statistic at "
                         "--cov-threshold inside the permutation FDR",
    "--max-gap-edges": "failing-edge bridges are not part of the supported "
                       "method (always 0)",
}


class _RetiredOption(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(
            f"{option_string} is no longer supported: "
            f"{RETIRED_OPTIONS[option_string]}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eiger2",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version=f"eiger {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    call = sub.add_parser(
        "call", help="find coordinated regions and their coordination FDR",
        description="Find regions of coordinated methylation and estimate their "
                    "coordination FDR with complete per-CpG permutation scans.",
    )
    inputs = call.add_argument_group("inputs")
    inputs.add_argument("--matrix",
                        help="single table with coordinate columns AND samples "
                             "(chr, start[, end/strand], then one column per "
                             "sample). Use instead of --betas/--locations.")
    inputs.add_argument("--betas", help="CpG-by-sample matrix, no coordinates")
    inputs.add_argument("--locations",
                        help="coordinates for --betas: chr/start[/end], or a "
                             "headerless 3-column BED")
    inputs.add_argument("--locations-no-header", dest="locations_no_header",
                        action="store_true",
                        help="the coordinate file has no header row")
    inputs.add_argument("--out", required=True, help="output directory")
    inputs.add_argument("--min-sample-fraction", dest="min_sample_fraction",
                        type=float, default=None,
                        help="drop CpGs observed in fewer than this fraction of "
                             "samples before analysis (default: keep all)")

    fdr = call.add_argument_group("coordination FDR")
    fdr.add_argument("--legacy", action="store_true",
                     help="use the original covariance statistic (fixed "
                          "--cov-threshold, row-mean imputation) inside the same "
                          "permutation FDR")
    fdr.add_argument("--target-fdr", dest="target_fdr", type=float, default=0.05,
                     help="select regions with coordination_qvalue <= this "
                          "(default 0.05)")
    fdr.add_argument("--fdr-replicates", dest="fdr_replicates", type=int,
                     default=DEFAULT_FDR_REPLICATES,
                     help="B, the number of complete permutation scans behind "
                          "coordination_qvalue (default 99)")
    fdr.add_argument("--permutation-seed", dest="permutation_seed", type=int,
                     default=1,
                     help="seed of the generator that permutes samples within "
                          "each CpG row (default 1)")
    fdr.add_argument("--random-seed", dest="random_seed", type=int, default=1,
                     help="internal scan seed: the observed scan uses it, "
                          "permutation scan b uses it + b (default 1)")
    fdr.add_argument("--mask-mode", dest="mask_mode",
                     choices=["values-with-mask", "mask-fixed"],
                     default="values-with-mask",
                     help="'values-with-mask' (default) permutes each row's "
                          "values and missingness together; 'mask-fixed' keeps "
                          "missing positions and permutes observed values only "
                          "(assumes exchangeability given the observation mask)")
    fdr.add_argument("--discovery-only", dest="discovery_only",
                     action="store_true",
                     help="observed scan only: candidates and scores, no "
                          "permutation scans, empty coordination_qvalue")

    caller = call.add_argument_group("caller settings (validated at defaults)")
    caller.add_argument("--min-cpgs", dest="min_cpgs", type=int, default=None,
                        help="minimum region length in CpGs (default 4)")
    caller.add_argument("--max-cpg-gap-bp", dest="max_cpg_gap_bp", type=int,
                        default=None,
                        help="largest gap in bp between adjacent CpGs of one "
                             "region (default 300)")
    caller.add_argument("--statistic", choices=["correlation", "covariance"],
                        default=None, help="newer profile statistic "
                                           "(default correlation)")
    caller.add_argument("--window", type=int, default=None,
                        help="newer profile: CpGs per co-variation window "
                             "(default 4)")
    caller.add_argument("--null-quantile", dest="null_quantile", type=float,
                        default=None,
                        help="newer profile: quantile of the distant-lag null "
                             "used as the scan threshold (default 0.995)")
    caller.add_argument("--null-replicates", dest="null_replicates", type=int,
                        default=None,
                        help="newer profile: internal distant-lag draws per scan "
                             "that set the scan threshold (default 8); not the "
                             "permutation count, which is --fdr-replicates")
    caller.add_argument("--null-min-lag", dest="null_min_lag", type=int,
                        default=None,
                        help="newer profile: minimum distant lag in CpG rows "
                             "(default 1000)")
    caller.add_argument("--cov-threshold", dest="cov_threshold", type=float,
                        default=None,
                        help="--legacy: fixed covariance threshold (default 0.01)")
    caller.add_argument("--no-impute", dest="no_impute", action="store_true",
                        help="--legacy: complete-case covariance without "
                             "row-mean imputation (non-default)")

    axis = call.add_argument_group("secondary axis annotations")
    axis.add_argument("--axis-annotations", dest="axis_annotations",
                      choices=list(_AXIS_MODES), default="descriptive",
                      help="'descriptive' (default): axis effect and its "
                           "uncertainty; 'off'; 'experimental-inference': also "
                           "axis_pvalue/axis_qvalue/axis_supported for regions "
                           "with >= 200 tested samples -- EXPERIMENTAL, checked "
                           "only with one retained axis and complete data. "
                           "Never changes primary calls.")

    run = call.add_argument_group("resources and output")
    run.add_argument("--scan-mode", dest="scan_mode",
                     choices=["chromosome", "whole"], default="chromosome",
                     help="'chromosome' (default) computes each scan one "
                          "chromosome at a time to bound memory; 'whole' scans "
                          "the whole matrix at once")
    run.add_argument("--storage-dtype", dest="storage_dtype",
                     choices=["float64", "float32"], default="float64",
                     help="in-memory precision of the matrix and its permuted "
                          "copy; peak-memory savings vary (default float64)")
    run.add_argument("--no-discoveries-table", dest="no_discoveries_table",
                     action="store_true",
                     help="do not write discoveries.csv")
    run.add_argument("--save-null-scores", dest="save_null_scores",
                     action="store_true",
                     help="also write every permutation scan's region scores "
                          "to null_region_scores.npz")
    run.add_argument("--quiet", action="store_true",
                     help="no progress messages on stderr")

    retired = call.add_argument_group("retired options (rejected with a reason)")
    for option in RETIRED_OPTIONS:
        retired.add_argument(option, action=_RetiredOption, nargs="?",
                             help=argparse.SUPPRESS)
    call.set_defaults(func=command_call)

    compare = sub.add_parser(
        "compare", help="overlap of two existing regions.csv files",
    )
    compare.add_argument("--regions-a", dest="regions_a", required=False,
                         help="an existing regions.csv")
    compare.add_argument("--regions-b", dest="regions_b", required=False,
                         help="a second existing regions.csv")
    compare.add_argument("--label-a", dest="label_a", default="A")
    compare.add_argument("--label-b", dest="label_b", default="B")
    compare.add_argument("--all-candidates", dest="all_candidates",
                         action="store_true",
                         help="compare every candidate, not only selected rows")
    compare.add_argument("--out", required=True, help="output directory")
    compare.set_defaults(func=command_compare)
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
