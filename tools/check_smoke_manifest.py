#!/usr/bin/env python3
"""Check a smoke run's outputs against what ``eiger2 call`` documents.

    python tools/check_smoke_manifest.py OUT --exit-code 0 \\
        [--version 2.2.0] [--profile newer|legacy] [--axis-mode descriptive]

``OUT`` is the output directory of one ``eiger2 call``.  Used by the smoke
targets in the root ``Makefile``, so that a green smoke target means "the run
completed the documented method and its files agree with each other", never
merely "the process exited".  Standard library only.

What fails the check:

* a run schema, status or software version other than the documented ones;
* a nonzero exit code, or a status other than ``ok`` (the example is built to
  produce candidates and q-values);
* permutation scans that were requested but not all completed, or
  per-permutation records that do not match the completed count;
* an FDR formula, pseudocount or deflation setting other than the supported
  method's;
* ``regions.csv`` or ``discoveries.csv`` disagreeing with the manifest's
  counts, a selected region above the target FDR, or an unselected region at
  or below it;
* axis annotations recorded as affecting primary calls, or inferential axis
  columns filled in the default descriptive mode;
* an input file recorded without a full SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

RUN_SCHEMA = "eiger.region_fdr_run/1"
DOCUMENTED_STATUSES = ("ok", "no_candidates", "discovery_only")
FDR_FORMULA_VERSION = "fdr_plus_cummin/1"
PRIMARY_COLUMNS = (
    "region_id", "chr", "start", "end", "n_cpg", "cpg_index_start",
    "cpg_index_end", "caller", "region_score", "statistic_mean",
    "scan_threshold", "coordination_qvalue", "coordination_selected",
)
AXIS_COLUMNS = (
    "axis_status", "axis_n_axes", "axis_n_samples_tested", "axis_index",
    "axis_effect", "axis_effect_se", "axis_effect_lower", "axis_effect_upper",
    "axis_pvalue", "axis_qvalue", "axis_supported",
)
INFERENTIAL_AXIS_COLUMNS = ("axis_pvalue", "axis_qvalue", "axis_supported")
MISSING = ("", "NA")
SHA256 = re.compile(r"[0-9a-f]{64}")


def fail(message: str) -> int:
    print(f"[smoke] FAIL: {message}", file=sys.stderr)
    return 1


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def check(args: argparse.Namespace) -> int:
    out: Path = args.out
    manifest_path = out / "run_manifest.json"
    if args.exit_code != 0:
        return fail(f"eiger2 call exited {args.exit_code}; the smoke run must "
                    "exit 0")
    try:
        manifest = json.loads(manifest_path.read_text())
    except OSError as exc:
        return fail(f"cannot read {manifest_path}: {exc}")
    except json.JSONDecodeError as exc:
        return fail(f"{manifest_path} is not valid JSON: {exc}")
    if not isinstance(manifest, dict):
        return fail(f"{manifest_path} is not a manifest object")

    if manifest.get("run_schema") != RUN_SCHEMA:
        return fail(f"run_schema is {manifest.get('run_schema')!r}, "
                    f"expected {RUN_SCHEMA!r}")
    if args.version and manifest.get("software_version") != args.version:
        return fail(f"software_version is {manifest.get('software_version')!r}, "
                    f"expected {args.version!r}")
    status = manifest.get("status")
    if status not in DOCUMENTED_STATUSES:
        return fail(f"undocumented run status {status!r}")
    if status != "ok":
        return fail(f"status is {status!r}; the example must complete with "
                    "candidates and q-values ('ok')")
    profile = (manifest.get("profile") or {}).get("name")
    if args.profile and profile != args.profile:
        return fail(f"caller profile is {profile!r}, expected {args.profile!r}")

    files = (manifest.get("inputs") or {}).get("files") or []
    if not files or not all(
        isinstance(f, dict) and SHA256.fullmatch(str(f.get("sha256", "")))
        for f in files
    ):
        return fail("every input file needs a full SHA-256 in inputs.files")

    method = manifest.get("method") or {}
    if method.get("fdr_formula_version") != FDR_FORMULA_VERSION:
        return fail(f"fdr_formula_version is "
                    f"{method.get('fdr_formula_version')!r}")
    if method.get("pseudocount") != 1:
        return fail(f"pseudocount is {method.get('pseudocount')!r}, expected 1")
    if method.get("automatic_deflation") is not False:
        return fail("automatic_deflation must be recorded as false")

    permutations = manifest.get("permutations") or {}
    requested = (manifest.get("settings") or {}).get("fdr_replicates_requested")
    completed = permutations.get("completed")
    if not (isinstance(requested, int) and requested >= 1):
        return fail(f"fdr_replicates_requested is {requested!r}")
    if permutations.get("requested") != requested or completed != requested:
        return fail(f"{completed!r} of {requested!r} permutation scans "
                    "completed; a partial set is not a valid FDR")
    for key in ("null_regions_per_permutation", "scan_thresholds"):
        if len(permutations.get(key) or []) != completed:
            return fail(f"permutations.{key} has "
                        f"{len(permutations.get(key) or [])} entries for "
                        f"{completed} completed scans")
    _header, perm_rows = _read_rows(out / "permutations.csv")
    if len(perm_rows) != completed:
        return fail(f"permutations.csv has {len(perm_rows)} rows for "
                    f"{completed} completed scans")

    results = manifest.get("results") or {}
    n_candidates, n_selected = results.get("n_candidates"), results.get("n_selected")
    if not (isinstance(n_candidates, int) and n_candidates >= 1):
        return fail(f"n_candidates is {n_candidates!r}; the example must "
                    "produce candidate regions")
    target = (manifest.get("settings") or {}).get("target_fdr")
    header, rows = _read_rows(out / "regions.csv")
    missing = [c for c in PRIMARY_COLUMNS + AXIS_COLUMNS if c not in header]
    if missing:
        return fail(f"regions.csv lacks documented columns {missing}")
    if len(rows) != n_candidates:
        return fail(f"regions.csv has {len(rows)} rows, manifest says "
                    f"{n_candidates} candidates")
    selected = 0
    for row in rows:
        q = float(row["coordination_qvalue"])
        flag = row["coordination_selected"]
        if flag not in ("True", "False"):
            return fail(f"coordination_selected {flag!r} in {row['region_id']}")
        if (flag == "True") != (q <= target):
            return fail(f"{row['region_id']}: coordination_selected={flag} but "
                        f"coordination_qvalue={q} against target {target}")
        selected += flag == "True"
    if selected != n_selected:
        return fail(f"{selected} selected rows in regions.csv, manifest says "
                    f"{n_selected}")
    _dheader, discoveries = _read_rows(out / "discoveries.csv")
    if [r["region_id"] for r in discoveries] != [
        r["region_id"] for r in rows if r["coordination_selected"] == "True"
    ]:
        return fail("discoveries.csv is not the selected subset of regions.csv")

    axis = manifest.get("axis") or {}
    if axis.get("affects_primary_calls") is not False:
        return fail("axis annotations must be recorded as not affecting "
                    "primary calls")
    if args.axis_mode and axis.get("mode") != args.axis_mode:
        return fail(f"axis mode is {axis.get('mode')!r}, expected "
                    f"{args.axis_mode!r}")
    if axis.get("mode") == "descriptive":
        if axis.get("inference_reported") is not False:
            return fail("descriptive axis mode must not report inference")
        filled = [c for c in INFERENTIAL_AXIS_COLUMNS
                  if any(r[c] not in MISSING for r in rows)]
        if filled:
            return fail(f"descriptive axis mode filled {filled}")

    ratio = (manifest.get("threshold_ratio") or {}).get("observed_over_mean_permuted")
    print(f"[smoke] {manifest_path}: schema {RUN_SCHEMA}, version "
          f"{manifest.get('software_version')}, profile {profile}")
    print(f"[smoke] status ok: {n_candidates} candidates, {n_selected} selected "
          f"at q <= {target}, {completed}/{requested} permutation scans")
    print(f"[smoke] axis: mode {axis.get('mode')}, {axis.get('n_axes')} axes, "
          f"status counts {axis.get('status_counts')}")
    print(f"[smoke] threshold ratio (descriptive): {ratio}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("out", type=Path, help="the eiger2 call output directory")
    parser.add_argument("--exit-code", type=int, required=True,
                        help="the exit code eiger2 call returned")
    parser.add_argument("--version", default=None,
                        help="require this software_version")
    parser.add_argument("--profile", choices=["newer", "legacy"], default=None,
                        help="require this caller profile")
    parser.add_argument("--axis-mode", default=None,
                        help="require this axis annotation mode")
    args = parser.parse_args(argv)
    try:
        return check(args)
    except (OSError, KeyError, ValueError) as exc:
        return fail(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
