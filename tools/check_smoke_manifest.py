#!/usr/bin/env python3
"""Inspect a smoke run's ``run_manifest.json`` and report what it actually says.

    python tools/check_smoke_manifest.py OUT/run_manifest.json --exit-code 0

Used by the ``smoke`` targets in the root ``Makefile``.  It exists so that a
green smoke target means "the run reached a documented state that was read off
the manifest", never merely "the process exited".  A refusal is always printed
as a refusal, and a refused run is never described as a completed inference.

What fails the check:

* a failed structural gate (spec 2a.2 -- the null is not usable at all);
* ``inference_unreliable`` -- the whole run's inferential columns are refused;
* ``no_regions`` -- the example fixture is expected to produce regions;
* a run status eiger2 does not document, which is a stop-and-escalate signal;
* a status that contradicts the caller's exit code, or the refusal lists.

What is reported but does not fail: ``partial_inference``.  It is a documented
stratum-local state (``Python/eiger2/core.py``, spec 2c/2a.1): the null
construction is sound and every other length stratum reports normally.  The
refused strata and their opportunity arithmetic are printed in full.  Pass
``--expect-no-refusals`` to demand that no stratum is refused; that is not the
default because the shipped example currently refuses stratum '20-49' -- see
the note in the repository's development documentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Run statuses eiger2 documents (Python/eiger2/core.py, spec 2a.1).  Anything
# else is undocumented and must be escalated rather than interpreted here.
COMPLETED_STATUSES = ("ok", "ok_with_warnings")
DOCUMENTED_STATUSES = COMPLETED_STATUSES + (
    "partial_inference", "inference_unreliable", "no_regions",
)

# Per-stratum refusals recorded by region_fdr_result.
REFUSAL_KEYS = (
    "insufficient_null_opportunity_strata",
    "insufficient_cluster_replicates_strata",
    "insufficient_scale_opportunity_strata",
)


def fail(message: str) -> int:
    print(f"[smoke] FAIL: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--exit-code", type=int, required=True,
        help="the exit code the revised caller returned, cross-checked here",
    )
    parser.add_argument(
        "--expect-no-refusals", action="store_true",
        help="also require that no length stratum was refused",
    )
    args = parser.parse_args()

    try:
        manifest = json.loads(args.manifest.read_text())
    except OSError as exc:
        return fail(f"cannot read {args.manifest}: {exc}")
    except json.JSONDecodeError as exc:
        return fail(f"{args.manifest} is not valid JSON: {exc}")
    if not isinstance(manifest, dict):
        return fail(f"{args.manifest} is not a manifest object")
    caller = manifest.get("caller_manifest")
    if not isinstance(caller, dict):
        return fail(f"{args.manifest} has no caller_manifest block")

    # Structural gates first: they decide whether the null is usable at all,
    # so nothing else in the manifest means anything until they pass.
    gates = caller.get("gates") or {}
    structural = (gates.get("structural") if isinstance(gates, dict) else None) or {}
    if not isinstance(structural, dict) or not structural:
        return fail("no structural gate results in the manifest")
    # A gate entry that does not say `passed: true` counts as failed, so a
    # truncated or malformed manifest cannot read as a pass.
    failed = {
        name: gate
        for name, gate in structural.items()
        if not (isinstance(gate, dict) and gate.get("passed") is True)
    }
    if failed:
        return fail(f"structural gate(s) failed: {failed}")
    print(f"[smoke] structural gates: {len(structural)} present, all passed")

    # Refused length strata, inspected and printed whether or not any exist.
    fdr = caller.get("fdr") or {}
    if not isinstance(fdr, dict):
        return fail("the manifest's fdr block is not an object")
    refused = {key: (fdr.get(key) or []) for key in REFUSAL_KEYS}
    for key, entries in refused.items():
        print(f"[smoke] {key}: {entries}")
    n_refused = sum(len(entries) for entries in refused.values())

    status = caller.get("status")
    print(f"[smoke] final inference status: {status}")

    if status not in DOCUMENTED_STATUSES:
        return fail(
            f"undocumented run status {status!r} -- stop and escalate rather "
            "than treating this run as a pass"
        )
    if status == "inference_unreliable":
        return fail(
            "run status 'inference_unreliable': the whole run's inferential "
            "columns are refused. This is a refusal, not an inference"
        )
    if status == "no_regions":
        return fail(
            "run status 'no_regions': the example is expected to produce "
            "regions, so this is a regression rather than a pass"
        )

    # The status and the refusal lists must agree about what happened.
    if status == "partial_inference" and n_refused == 0:
        return fail(
            "status 'partial_inference' but no refused stratum is recorded; "
            "the manifest contradicts itself"
        )
    if status in COMPLETED_STATUSES and n_refused:
        return fail(
            f"status {status!r} reports a completed inference while "
            f"{n_refused} stratum/strata are recorded as refused; "
            "the manifest contradicts itself"
        )

    # The CLI returns 1 only for inference_unreliable, which already failed
    # above; any other nonzero code here means exit and manifest disagree.
    if args.exit_code != 0:
        return fail(
            f"caller exited {args.exit_code} but the manifest reports "
            f"{status!r}; exit code and manifest disagree"
        )

    if args.expect_no_refusals and n_refused:
        return fail(
            f"--expect-no-refusals was requested but {n_refused} "
            "stratum/strata were refused"
        )

    n_significant = caller.get("n_significant")
    if status == "partial_inference":
        strata = [
            entry.get("stratum")
            for entries in refused.values() for entry in entries
        ]
        print(
            f"[smoke] revised caller REFUSED inference for length stratum/strata "
            f"{strata}: the printed entries give the null opportunity available "
            f"against the opportunity required. Every other stratum reported "
            f"normally ({n_significant} significant regions); the run as a "
            f"whole did not complete inference"
        )
    else:
        print(
            f"[smoke] revised caller completed inference for every stratum "
            f"(status {status}, {n_significant} significant regions)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
