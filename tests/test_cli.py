"""Tests for eiger2.cli's argument parsing (Stage 1f Task 1a, master spec pre-release).

parse_args()/`_config_from_args` do not touch the filesystem, so these tests exercise
argument round-tripping directly without needing real input files.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

import eiger2.cli as cli  # noqa: E402
from eiger2.cli import _config_from_args, build_parser  # noqa: E402

_BASE_ARGS = [
    "call", "--betas", "betas.tsv", "--locations", "cpgs.bed", "--out", "out/",
]


def test_git_provenance_records_dirty_state_and_deterministic_fingerprint(
    monkeypatch,
):
    status = b" M Python/eiger2/core.py\0?? local-note.txt\0"
    diff = b"diff --git a/core.py b/core.py\n+fixed\n"

    def fake_run(command, **_kwargs):
        if command[1:3] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="abc123\n")
        if command[1] == "status":
            return SimpleNamespace(stdout=status)
        if command[1] == "diff":
            return SimpleNamespace(stdout=diff)
        raise AssertionError(command)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    result = cli._git_provenance()
    digest = hashlib.sha256()
    digest.update(status)
    digest.update(b"\0EIGER_TRACKED_DIFF\0")
    digest.update(diff)

    assert result == {
        "git_commit": "abc123",
        "git_worktree_dirty": True,
        "git_worktree_fingerprint_sha256": digest.hexdigest(),
    }


def test_region_null_mode_cli_flag_round_trips_into_config():
    for value in ("member_step", "lag", "off"):
        args = build_parser().parse_args(
            [*_BASE_ARGS, "--region-null-mode", value]
        )
        config = _config_from_args(args)
        assert config.region_null_mode == value


def test_region_null_mode_unset_leaves_config_at_its_own_default():
    # Stage 1f Task 1b: the CLI flag choices are member_step/lag/off only --
    # "auto" is not CLI-selectable, it is what happens when the flag is left
    # unset entirely (matching every other --null-* flag's default=None
    # pattern, filtered out of _config_from_args's kwargs by the `is not
    # None` check, so Eiger2Config's own field default applies untouched).
    args = build_parser().parse_args(_BASE_ARGS)
    config = _config_from_args(args)
    assert config.region_null_mode == "auto"


def test_n_lag_draws_cli_flag_round_trips_into_config():
    args = build_parser().parse_args([*_BASE_ARGS, "--n-lag-draws", "123"])
    assert _config_from_args(args).n_lag_draws == 123


@pytest.mark.parametrize(
    ("region_null_mode", "expected_option", "wrong_option"),
    [
        ("member_step", "--n-lag-draws", "--null-replicates"),
        ("lag", "--null-replicates", "--n-lag-draws"),
    ],
)
def test_insufficient_opportunity_advice_names_mode_effective_volume_control(
    monkeypatch, tmp_path, capsys, region_null_mode, expected_option, wrong_option,
):
    matrix = np.full((5, 5), 0.5)
    locations = pd.DataFrame(
        {"chr": ["chr1"] * 5, "start": range(5), "end": range(1, 6)}
    )
    caller_manifest = {
        "status": "partial_inference",
        "gates": {
            "region_null_mode": region_null_mode,
            "structural": {},
        },
        "fdr": {
            "insufficient_null_opportunity_strata": [{
                "stratum": "50+", "observed_regions": 1,
                "null_opportunity": 10.0, "required_null_opportunity": 100.0,
                "m_opportunity": 10.0,
            }],
        },
        "layout": {},
        "null": {},
    }
    monkeypatch.setattr(
        cli, "_resolve_inputs",
        lambda _args: (matrix, locations, [f"S{i}" for i in range(5)], []),
    )
    monkeypatch.setattr(
        cli, "call_dmrs",
        lambda *_args, **_kwargs: (
            pd.DataFrame(), pd.DataFrame(), caller_manifest,
        ),
    )

    out = tmp_path / region_null_mode
    assert cli.main(["call", "--matrix", "unused.tsv", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert expected_option in captured.err
    assert wrong_option not in captured.err


def test_legacy_call_accepts_none_fdr_diagnostics(monkeypatch, tmp_path):
    """The legacy manifest carries ``fdr=None`` when no FDR was requested."""

    matrix = np.full((5, 5), 0.5)
    locations = pd.DataFrame(
        {"chr": ["chr1"] * 5, "start": range(5), "end": range(1, 6)}
    )
    monkeypatch.setattr(
        cli,
        "_resolve_inputs",
        lambda _args: (matrix, locations, [f"S{i}" for i in range(5)], []),
    )
    monkeypatch.setattr(
        cli,
        "run_legacy",
        lambda *_args, **_kwargs: (
            pd.DataFrame(),
            {"caller": "legacy", "fdr": None, "n_regions": 0},
        ),
    )
    monkeypatch.setattr(
        cli,
        "_git_provenance",
        lambda: {
            "git_commit": "dirty-head",
            "git_worktree_dirty": True,
            "git_worktree_fingerprint_sha256": "a" * 64,
        },
    )

    out = tmp_path / "legacy"
    argv = ["call", "--matrix", "unused.tsv", "--out", str(out), "--legacy"]
    assert cli.main(argv) == 0
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["caller_manifest"]["fdr"] is None
    assert manifest["git_commit"] == "dirty-head"
    assert manifest["git_worktree_dirty"] is True
    assert manifest["git_worktree_fingerprint_sha256"] == "a" * 64


def test_cli_input_parsing_matches_python_api_dtype_bit_for_bit(tmp_path):
    """Spec §2a.6: the CLI used to parse text through float32 before upcasting
    to float64, while the Python API kept float64 throughout -- the same data
    through the two entry points then produced different numbers. Values here
    are not exactly representable in float32, so a reintroduced float32 step
    would flip bits this test can see.
    """

    from eiger2.core import Eiger2Config, edge_statistic

    rng = np.random.default_rng(0)
    values = 0.1 + 0.8 * rng.random((6, 3)) + 1e-7 * rng.random((6, 3))
    betas_path = tmp_path / "betas.tsv"
    locations_path = tmp_path / "cpgs.bed"
    pd.DataFrame(values, columns=["S1", "S2", "S3"]).to_csv(
        betas_path, sep="\t", index=False
    )
    pd.DataFrame(
        {
            "chr": ["chr1"] * 6,
            "start": [0, 10, 20, 30, 40, 50],
            "end": [1, 11, 21, 31, 41, 51],
        }
    ).to_csv(locations_path, sep="\t", index=False)

    cli_matrix, _locations, _samples = cli.load_inputs(betas_path, locations_path)

    # The Python API's own entry point (call_dmrs) forces float64 on entry
    # regardless of what a caller passes in; a caller reading the same
    # serialised input directly with pandas (the natural thing to do before
    # calling the API) gets float64 by default already -- this is exactly
    # that path, independent of the CLI's own parsing.
    api_matrix = pd.read_csv(betas_path, sep="\t").to_numpy(dtype=np.float64)

    assert cli_matrix.dtype == np.float64
    assert api_matrix.dtype == np.float64
    assert np.array_equal(cli_matrix, api_matrix)

    config = Eiger2Config(min_paired_samples=3)
    cli_statistic = edge_statistic(cli_matrix, config)
    api_statistic = edge_statistic(api_matrix, config)
    assert np.array_equal(cli_statistic, api_statistic, equal_nan=True)
    assert np.isfinite(cli_statistic).any()


def test_command_call_exits_non_zero_and_prints_blocked_summary(
    monkeypatch, tmp_path, capsys,
):
    """Spec §2a.4: command_call exits non-zero on inference_unreliable and
    names every failed structural gate with its measured value."""

    matrix = np.full((5, 5), 0.5)
    locations = pd.DataFrame(
        {"chr": ["chr1"] * 5, "start": range(5), "end": range(1, 6)}
    )
    fake_manifest = {
        "status": "inference_unreliable",
        "gates": {
            "structural": {
                "surviving_null_tracks": {
                    "surviving": 7, "null_replicates": 100,
                    "fraction": 0.07, "floor_fraction": 0.5, "passed": False,
                },
                "null_coverage": {
                    "value": 0.213, "floor": 0.25, "passed": False,
                },
                "null_construction": {
                    "value": 50_000, "floor": 1_000, "passed": True,
                },
            },
        },
        "fdr": {"status": "structural_gate_failure"},
        "null": {},
        "layout": {},
    }
    monkeypatch.setattr(
        cli,
        "_resolve_inputs",
        lambda _args: (matrix, locations, [f"S{i}" for i in range(5)], []),
    )
    monkeypatch.setattr(
        cli,
        "call_dmrs",
        lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame(), fake_manifest),
    )

    out = tmp_path / "blocked"
    argv = [
        "call", "--betas", "unused.tsv", "--locations", "unused.bed",
        "--out", str(out),
    ]
    exit_code = cli.main(argv)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "BLOCKED" in captured.err
    assert "surviving_null_tracks" in captured.err
    assert "7/100" in captured.err
    assert "null_coverage" in captured.err
    assert "21.3%" in captured.err
    assert "null_construction" not in captured.err


def test_manifest_envelope_and_caller_manifest_schema_versions_agree():
    """Spec §2a.6: the manifest envelope used to report schema "2.1" while the
    caller manifest reported "2.2"/"2.3" -- reconciled to one literal,
    pinned here across both real caller manifests (not the mocked fixture
    other tests use, which would hide a literal drifting out of sync)."""

    from eiger2.cli import run_legacy
    from eiger2.core import call_dmrs
    from eiger2.manifest_utils import build_manifest_envelope

    envelope = build_manifest_envelope(
        config_dict={}, git_commit="deadbeef", input_checksums={},
        random_seed=1, derived_seeds={}, runtime_seconds=0.1, peak_memory_mb=1.0,
        warnings=[], fdr_decision="point",
    )

    # eiger2 caller: the fast "insufficient_cpgs" early return still sets
    # schema_version, without needing a real region-calling run.
    from eiger2.core import Eiger2Config

    _regions, _values, eiger2_manifest = call_dmrs(
        np.full((2, 5), 0.5), pd.DataFrame(
            {"chr": ["chr1", "chr1"], "start": [0, 10], "end": [1, 11]}
        ),
        config=Eiger2Config(min_cpgs=5),
    )
    assert eiger2_manifest["status"] == "insufficient_cpgs"

    # legacy caller: fdr_replicates=0 skips the (separately, deliberately
    # unsupported -- see test_run_legacy_fdr_replicates_fails_cleanly_not_a_traceback)
    # FDR annotation path, exercising only discovery.
    matrix = np.random.default_rng(0).uniform(0.1, 0.9, size=(30, 20))
    locations = pd.DataFrame(
        {"chr": ["chr1"] * 30, "start": range(30), "end": range(1, 31)}
    )
    _frame, legacy_manifest = run_legacy(
        matrix, locations, [f"S{i}" for i in range(20)],
        cov_threshold=0.012, delta_threshold=0.2, min_cpgs=5, impute=True,
        fdr_replicates=0,
    )

    assert envelope["schema_version"] == eiger2_manifest["schema_version"]
    assert envelope["schema_version"] == legacy_manifest["schema_version"]
    assert envelope["schema_version"] == "2.3"


def test_command_compare_handles_a_regions_csv_mixing_available_and_unavailable(
    tmp_path,
):
    """A `.notna().any()` guard only checks that SOME row is available, not
    every row, so a frame mixing available and unavailable `significant`
    values (e.g. a future partial_inference run refusing one length stratum
    while reporting the rest -- see docs/FDR.md §14) passed the guard and
    then raised on `.astype(bool)`, which cannot convert the unavailable
    rows' NA. `.fillna(False)` handles every mix -- all available, all
    unavailable, or, as here, both at once -- treating unavailable rows the
    same as `False` (excluded from the "significant" subset)."""

    regions_a = pd.DataFrame({
        "chr": ["chr1", "chr1", "chr1"],
        "start": [0, 100, 200],
        "end": [10, 110, 210],
        "significant": pd.array([True, False, pd.NA], dtype="boolean"),
    })
    regions_b = pd.DataFrame({
        "chr": ["chr1", "chr1"],
        "start": [0, 300],
        "end": [10, 310],
        "significant": pd.array([True, pd.NA], dtype="boolean"),
    })
    path_a = tmp_path / "regions_a.csv"
    path_b = tmp_path / "regions_b.csv"
    regions_a.to_csv(path_a, index=False)
    regions_b.to_csv(path_b, index=False)

    out = tmp_path / "compare"
    argv = [
        "compare", "--regions-a", str(path_a), "--regions-b", str(path_b),
        "--out", str(out),
    ]
    assert cli.main(argv) == 0

    summary = pd.read_csv(out / "comparison_summary.csv")
    # Only the one genuinely True row from each side entered the overlap --
    # the False row and both pd.NA rows are excluded, not counted as hits.
    assert summary["A_regions"].iloc[0] == 1
    assert summary["B_regions"].iloc[0] == 1
