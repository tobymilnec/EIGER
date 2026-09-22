"""Tests for ``tools/check_smoke_manifest.py``, the smoke targets' verdict.

Each test writes a small, internally consistent ``eiger2 call`` output
directory, breaks one documented property, and requires the checker to name
it. The checker is standard-library only and runs as a script.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parent.parent / "tools" / "check_smoke_manifest.py"
HEADER = [
    "region_id", "chr", "start", "end", "n_cpg", "cpg_index_start",
    "cpg_index_end", "caller", "region_score", "statistic_mean",
    "scan_threshold", "coordination_qvalue", "coordination_selected",
    "axis_status", "axis_n_axes", "axis_n_samples_tested", "axis_index",
    "axis_effect", "axis_effect_se", "axis_effect_lower", "axis_effect_upper",
    "axis_pvalue", "axis_qvalue", "axis_supported",
]
MANIFEST = {
    "run_schema": "eiger.region_fdr_run/1",
    "software_version": "9.9.9",
    "status": "ok",
    "inputs": {"files": [{"path": "betas.tsv", "sha256": "a" * 64}]},
    "method": {"fdr_formula_version": "fdr_plus_cummin/1", "pseudocount": 1,
               "automatic_deflation": False},
    "profile": {"name": "newer"},
    "settings": {"target_fdr": 0.05, "fdr_replicates_requested": 3},
    "permutations": {"requested": 3, "completed": 3,
                     "null_regions_per_permutation": [0, 1, 0],
                     "scan_thresholds": [0.1, 0.1, 0.1]},
    "results": {"n_candidates": 2, "n_selected": 1},
    "axis": {"mode": "descriptive", "affects_primary_calls": False,
             "inference_reported": False, "n_axes": 1,
             "status_counts": {"estimated": 2}},
    "threshold_ratio": {"observed_over_mean_permuted": 1.1},
}


def _region(name, q, selected):
    row = dict.fromkeys(HEADER, "NA")
    row.update(region_id=name, chr="chr1", start="1", end="9", n_cpg="4",
               caller="newer", region_score="1.0", coordination_qvalue=str(q),
               coordination_selected=selected, axis_status="estimated",
               axis_effect="0.2")
    return row


def load_checker():
    spec = importlib.util.spec_from_file_location("check_smoke_manifest", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(out: Path, manifest=None, regions=None, discoveries=None,
              n_permutation_rows=3):
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_manifest.json").write_text(json.dumps(manifest or MANIFEST))
    regions = regions if regions is not None else [
        _region("r1", 0.02, "True"), _region("r2", 0.5, "False"),
    ]
    if discoveries is None:
        discoveries = [r for r in regions if r["coordination_selected"] == "True"]
    for name, rows in (("regions.csv", regions), ("discoveries.csv", discoveries)):
        with (out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(rows)
    with (out / "permutations.csv").open("w", newline="") as handle:
        handle.write("replicate,n_regions\n")
        for b in range(n_permutation_rows):
            handle.write(f"{b + 1},0\n")
    return out


def run(out, *extra, exit_code=0):
    return load_checker().main([str(out), "--exit-code", str(exit_code), *extra])


def test_a_consistent_run_passes_and_is_summarised(tmp_path, capsys):
    out = write_run(tmp_path / "run")
    assert run(out, "--version", "9.9.9", "--profile", "newer",
               "--axis-mode", "descriptive") == 0
    printed = capsys.readouterr().out
    assert "2 candidates, 1 selected" in printed
    assert "3/3 permutation scans" in printed


def _mutated(**changes):
    manifest = copy.deepcopy(MANIFEST)
    for path, value in changes.items():
        target = manifest
        keys = path.split("__")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return manifest


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"run_schema": "eiger.region_fdr_run/0"}, "run_schema"),
        ({"status": "partial_inference"}, "undocumented"),
        ({"status": "no_candidates"}, "status"),
        ({"permutations__completed": 2}, "permutation scans"),
        ({"permutations__scan_thresholds": [0.1]}, "scan_thresholds"),
        ({"method__fdr_formula_version": "bh/1"}, "fdr_formula_version"),
        ({"method__pseudocount": 0}, "pseudocount"),
        ({"method__automatic_deflation": True}, "automatic_deflation"),
        ({"results__n_candidates": 3}, "regions.csv has 2 rows"),
        ({"results__n_selected": 2}, "selected rows"),
        ({"axis__affects_primary_calls": True}, "primary calls"),
        ({"inputs__files": [{"path": "betas.tsv", "sha256": "abc"}]}, "SHA-256"),
    ],
)
def test_each_documented_property_is_enforced(tmp_path, capsys, changes, message):
    out = write_run(tmp_path / "run", manifest=_mutated(**changes))
    assert run(out) == 1
    assert message in capsys.readouterr().err


def test_a_nonzero_exit_code_fails(tmp_path, capsys):
    out = write_run(tmp_path / "run")
    assert run(out, exit_code=1) == 1
    assert "exited 1" in capsys.readouterr().err


def test_a_selection_flag_contradicting_the_qvalue_fails(tmp_path, capsys):
    regions = [_region("r1", 0.02, "True"), _region("r2", 0.04, "False")]
    out = write_run(tmp_path / "run", regions=regions,
                    discoveries=regions[:1])
    assert run(out) == 1
    assert "r2" in capsys.readouterr().err


def test_discoveries_must_be_the_selected_subset(tmp_path, capsys):
    out = write_run(tmp_path / "run", discoveries=[])
    assert run(out) == 1
    assert "discoveries.csv" in capsys.readouterr().err


def test_inferential_axis_columns_must_be_empty_in_descriptive_mode(
    tmp_path, capsys
):
    regions = [_region("r1", 0.02, "True"), _region("r2", 0.5, "False")]
    regions[0]["axis_qvalue"] = "0.01"
    out = write_run(tmp_path / "run", regions=regions)
    assert run(out) == 1
    assert "axis_qvalue" in capsys.readouterr().err


def test_a_permutation_table_of_the_wrong_length_fails(tmp_path, capsys):
    out = write_run(tmp_path / "run", n_permutation_rows=2)
    assert run(out) == 1
    assert "permutations.csv" in capsys.readouterr().err


def test_version_and_profile_expectations_are_enforced(tmp_path, capsys):
    out = write_run(tmp_path / "run")
    assert run(out, "--version", "2.2.0rc1") == 1
    assert "software_version" in capsys.readouterr().err
    assert run(out, "--profile", "legacy") == 1
    assert "profile" in capsys.readouterr().err


def test_a_missing_or_invalid_manifest_fails_cleanly(tmp_path, capsys):
    assert run(tmp_path / "absent") == 1
    assert "cannot read" in capsys.readouterr().err
    out = write_run(tmp_path / "run")
    (out / "run_manifest.json").write_text("{not json")
    assert run(out) == 1
    assert "not valid JSON" in capsys.readouterr().err
