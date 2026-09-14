"""Tests for ``tools/check_smoke_manifest.py``, the smoke targets' verdict.

The helper decides whether ``make smoke-revised`` may report success, so its
own failure modes need covering: a refusal must never read as a completed
inference, and a manifest that is malformed, self-contradictory or in an
undocumented state must fail rather than be interpreted.

It is exercised as the command line the Makefile actually runs, through
``sys.executable``. It imports nothing from eiger2, so these tests neither
need the package installed nor can pick it up from a sibling checkout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parent.parent / "tools" / "check_smoke_manifest.py"


def make_manifest(
    status="ok_with_warnings",
    structural=None,
    refused=None,
    n_significant=11,
    fdr_extra=None,
):
    """A run manifest of the shape ``eiger2 call`` writes."""
    if structural is None:
        structural = {
            "null_track_survival": {"passed": True},
            "region_exchangeability": {"passed": True},
        }
    fdr = {"insufficient_null_opportunity_strata": list(refused or [])}
    if fdr_extra:
        fdr.update(fdr_extra)
    return {
        "caller_manifest": {
            "status": status,
            "gates": {"structural": structural},
            "fdr": fdr,
            "n_significant": n_significant,
        }
    }


REFUSAL = {
    "stratum": "20-49",
    "observed_regions": 1,
    "null_opportunity": 282134.0,
    "required_null_opportunity": 499810.0,
}


def run_checker(tmp_path, manifest, exit_code=0, extra=()):
    path = tmp_path / "run_manifest.json"
    if isinstance(manifest, str):
        path.write_text(manifest)
    else:
        path.write_text(json.dumps(manifest))
    command = [sys.executable, str(CHECKER), str(path)]
    command += ["--exit-code", str(exit_code), *extra]
    return subprocess.run(command, capture_output=True, text=True)


@pytest.mark.parametrize("status", ["ok", "ok_with_warnings"])
def test_a_completed_run_with_no_refusal_passes(tmp_path, status):
    result = run_checker(tmp_path, make_manifest(status=status))
    assert result.returncode == 0, result.stderr
    assert f"final inference status: {status}" in result.stdout
    assert "completed inference for every stratum" in result.stdout


def test_partial_inference_with_a_recorded_refusal_passes_and_reports_it(tmp_path):
    result = run_checker(
        tmp_path, make_manifest(status="partial_inference", refused=[REFUSAL])
    )
    assert result.returncode == 0, result.stderr
    # The refusal is printed as a refusal, with the stratum named ...
    assert "REFUSED inference for length stratum/strata ['20-49']" in result.stdout
    assert "did not complete inference" in result.stdout
    # ... and the opportunity arithmetic that justifies it is shown.
    assert "499810.0" in result.stdout
    # It is never described as a completed inference.
    assert "completed inference for every stratum" not in result.stdout


def test_partial_inference_without_a_refusal_fails(tmp_path):
    result = run_checker(tmp_path, make_manifest(status="partial_inference"))
    assert result.returncode == 1
    assert "contradicts itself" in result.stderr


def test_a_completed_status_carrying_a_refusal_fails(tmp_path):
    result = run_checker(tmp_path, make_manifest(status="ok", refused=[REFUSAL]))
    assert result.returncode == 1
    assert "contradicts itself" in result.stderr


def test_inference_unreliable_fails(tmp_path):
    result = run_checker(
        tmp_path, make_manifest(status="inference_unreliable"), exit_code=1
    )
    assert result.returncode == 1
    assert "refusal, not an inference" in result.stderr


def test_no_regions_fails(tmp_path):
    result = run_checker(tmp_path, make_manifest(status="no_regions"))
    assert result.returncode == 1
    assert "no_regions" in result.stderr


def test_a_failed_structural_gate_fails(tmp_path):
    result = run_checker(
        tmp_path,
        make_manifest(
            structural={
                "null_track_survival": {"passed": False, "observed": 0.31},
                "region_exchangeability": {"passed": True},
            }
        ),
    )
    assert result.returncode == 1
    assert "structural gate(s) failed" in result.stderr


def test_a_gate_entry_without_a_verdict_is_not_read_as_a_pass(tmp_path):
    result = run_checker(
        tmp_path, make_manifest(structural={"null_track_survival": {}})
    )
    assert result.returncode == 1
    assert "structural gate(s) failed" in result.stderr


def test_an_undocumented_status_fails(tmp_path):
    result = run_checker(tmp_path, make_manifest(status="mostly_fine"))
    assert result.returncode == 1
    assert "undocumented run status" in result.stderr
    assert "escalate" in result.stderr


def test_an_exit_code_the_manifest_does_not_explain_fails(tmp_path):
    result = run_checker(tmp_path, make_manifest(status="ok"), exit_code=1)
    assert result.returncode == 1
    assert "exit code and manifest disagree" in result.stderr


def test_expect_no_refusals_rejects_a_run_that_refused_a_stratum(tmp_path):
    manifest = make_manifest(status="partial_inference", refused=[REFUSAL])
    assert run_checker(tmp_path, manifest).returncode == 0
    strict = run_checker(tmp_path, manifest, extra=["--expect-no-refusals"])
    assert strict.returncode == 1
    assert "--expect-no-refusals" in strict.stderr


def test_the_other_refusal_kinds_are_inspected_too(tmp_path):
    """Cluster-replicate and scale-opportunity refusals count as refusals."""
    manifest = make_manifest(
        status="partial_inference",
        fdr_extra={"insufficient_cluster_replicates_strata": [{"stratum": "6-9"}]},
    )
    result = run_checker(tmp_path, manifest)
    assert result.returncode == 0, result.stderr
    printed = "insufficient_cluster_replicates_strata: [{'stratum': '6-9'}]"
    assert printed in result.stdout
    assert "REFUSED inference for length stratum/strata ['6-9']" in result.stdout


def test_a_manifest_without_a_caller_block_fails_cleanly(tmp_path):
    result = run_checker(tmp_path, {"schema_version": 3})
    assert result.returncode == 1
    assert "no caller_manifest block" in result.stderr
    assert "Traceback" not in result.stderr


def test_malformed_json_fails_cleanly(tmp_path):
    result = run_checker(tmp_path, "{not json")
    assert result.returncode == 1
    assert "not valid JSON" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_missing_manifest_fails_cleanly(tmp_path):
    command = [sys.executable, str(CHECKER), str(tmp_path / "absent.json")]
    result = subprocess.run(
        command + ["--exit-code", "0"], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "cannot read" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_manifest_with_no_structural_gates_fails(tmp_path):
    result = run_checker(tmp_path, make_manifest(structural={}))
    assert result.returncode == 1
    assert "no structural gate results" in result.stderr


def test_discovery_only_passes_with_no_structural_gates(tmp_path):
    # v2.3's default path builds no null, so it reports no null-construction
    # gates. That is the correct shape for that status, not a truncated
    # manifest, and it is a pass.
    manifest = make_manifest(status="discovery_only", structural={})
    result = run_checker(tmp_path, manifest)
    assert result.returncode == 0, result.stderr
    assert "none built (discovery_only)" in result.stdout


def test_discovery_only_with_structural_gates_is_self_contradictory(tmp_path):
    # Gates on a null that was never built mean one of the two statements is
    # false; the checker must not pick which.
    manifest = make_manifest(status="discovery_only")
    result = run_checker(tmp_path, manifest)
    assert result.returncode == 1
    assert "contradicts itself" in result.stderr
