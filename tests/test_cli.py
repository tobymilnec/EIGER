"""Tests for the ``eiger2`` command line.

The supported command is ``eiger2 call`` (coordination FDR, optionally
``--legacy``) and ``eiger2 compare`` on two existing outputs. Historical
options are rejected with a migration message rather than silently changing
meaning.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from region_fdr_data import make_dataset  # noqa: E402

import eiger2.cli as cli  # noqa: E402
from eiger2.cli import RETIRED_OPTIONS, build_parser  # noqa: E402
from eiger2.coordination import (  # noqa: E402
    LEGACY,
    RegionFdrConfig,
    call_coordinated_regions,
)

_LEGACY_DATA = {
    "seed": 12,
    "n_samples": 10,
    "noise_sd": 0.2,
    "n_blocks": 20,
    "block_strength": 0.12,
    "missing_fraction": 0.03,
    "chromosomes": (("chr1", 5_000), ("chr2", 4_000)),
    "shuffle_rows": True,
}


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli_inputs")
    betas, locations = make_dataset(**_LEGACY_DATA)
    samples = [f"S{i}" for i in range(betas.shape[1])]
    pd.DataFrame(betas, columns=samples).to_csv(
        root / "betas.tsv", sep="\t", index=False, na_rep="NA"
    )
    locations.to_csv(root / "cpgs.bed", sep="\t", index=False)
    return root, betas, locations, samples


def _call(inputs, out, *extra):
    root = inputs[0]
    return cli.main([
        "call", "--betas", str(root / "betas.tsv"), "--locations",
        str(root / "cpgs.bed"), "--out", str(out), "--quiet", *extra,
    ])


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


def test_git_provenance_ignores_a_repository_that_is_not_this_checkout(
    monkeypatch, tmp_path,
):
    """An installed package inside someone else's Git repository (a virtual
    environment created in an analysis project) must not report that
    repository's commit or dirty state as EIGER's source."""

    fake_module = tmp_path / "venv" / "lib" / "site-packages" / "eiger2" / "cli.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("")

    def must_not_run(command, **_kwargs):
        raise AssertionError(f"git was run for an installed package: {command}")

    monkeypatch.setattr(cli, "__file__", str(fake_module))
    monkeypatch.setattr(cli.subprocess, "run", must_not_run)
    assert cli._source_checkout() is None
    assert cli._git_provenance() == {
        "git_commit": "unknown",
        "git_worktree_dirty": None,
        "git_worktree_fingerprint_sha256": None,
    }


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


def test_command_compare_handles_a_regions_csv_mixing_available_and_unavailable(
    tmp_path,
):
    """A `.notna().any()` guard only checks that SOME row is available, not
    every row, so a frame mixing available and unavailable `significant`
    values (e.g. a future partial_inference run refusing one length stratum
    while reporting the rest) passed the guard and
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


def test_legacy_call_writes_the_output_contract(inputs, tmp_path):
    out = tmp_path / "legacy"
    assert _call(inputs, out, "--legacy", "--fdr-replicates", "9") == 0
    for name in ("regions.csv", "discoveries.csv", "permutations.csv",
                 "run_manifest.json"):
        assert (out / name).exists(), name
    regions = pd.read_csv(out / "regions.csv")
    assert list(regions.columns[:13]) == [
        "region_id", "chr", "start", "end", "n_cpg", "cpg_index_start",
        "cpg_index_end", "caller", "region_score", "statistic_mean",
        "scan_threshold", "coordination_qvalue", "coordination_selected",
    ]
    assert "region_qvalue" not in regions.columns
    assert "significant" not in regions.columns
    discoveries = pd.read_csv(out / "discoveries.csv")
    assert len(discoveries) == int(regions["coordination_selected"].sum())
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["run_schema"] == "eiger.region_fdr_run/1"
    assert manifest["permutations"]["completed"] == 9
    assert manifest["settings"]["fdr_replicates_requested"] == 9
    assert manifest["profile"]["cov_threshold"] == 0.01
    assert manifest["method"]["fdr_formula_version"] == "fdr_plus_cummin/1"
    assert len(manifest["permutations"]["scan_thresholds"]) == 9
    assert manifest["threshold_ratio"]["observed_over_mean_permuted"] == 1.0
    assert all(len(f["sha256"]) == 64 for f in manifest["inputs"]["files"])
    assert manifest["axis"]["inference_reported"] is False


def test_cli_and_api_agree(inputs, tmp_path):
    root = inputs[0]
    out = tmp_path / "agree"
    assert _call(inputs, out, "--legacy", "--fdr-replicates", "9") == 0
    written = pd.read_csv(out / "regions.csv", float_precision="round_trip")
    # the API on exactly the matrix the CLI parsed from the same files
    betas, locations, samples = cli.load_inputs(root / "betas.tsv", root / "cpgs.bed")
    api = call_coordinated_regions(
        betas, locations, sample_names=samples,
        config=RegionFdrConfig(profile=LEGACY, fdr_replicates=9),
    ).regions
    np.testing.assert_array_equal(written["region_id"], api["region_id"])
    np.testing.assert_array_equal(written["region_score"], api["region_score"])
    np.testing.assert_array_equal(
        written["coordination_qvalue"], api["coordination_qvalue"]
    )
    np.testing.assert_array_equal(
        written["coordination_selected"].astype(bool),
        api["coordination_selected"].astype(bool),
    )


def test_discovery_only_writes_empty_qvalues_and_no_discoveries_table(inputs, tmp_path):
    out = tmp_path / "discovery"
    assert _call(inputs, out, "--legacy", "--discovery-only") == 0
    regions = pd.read_csv(out / "regions.csv")
    assert regions["coordination_qvalue"].isna().all()
    assert regions["coordination_selected"].isna().all()
    assert not (out / "discoveries.csv").exists()


def test_lag_infeasible_input_fails_with_a_reason(inputs, tmp_path, capsys):
    # 5,000 CpGs on the longest chromosome cannot host a 5,000-row lag
    assert _call(inputs, tmp_path / "newer", "--fdr-replicates", "3",
                 "--null-min-lag", "5000") == 1
    assert "distant-lag candidate threshold" in capsys.readouterr().err


@pytest.mark.parametrize("option", sorted(RETIRED_OPTIONS))
def test_retired_options_fail_with_a_migration_message(option, capsys):
    argv = ["call", "--betas", "b.tsv", "--locations", "c.bed", "--out", "o", option]
    if option not in {"--no-confirm", "--compute-posterior"}:
        argv.append("1")
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(argv)
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert f"{option} is no longer supported" in err


def test_profile_specific_options_are_not_silently_ignored():
    args = build_parser().parse_args(
        ["call", "--matrix", "m.tsv", "--out", "o", "--legacy", "--window", "3"]
    )
    with pytest.raises(SystemExit, match="newer profile only"):
        cli._config_from_args(args)
    args = build_parser().parse_args(
        ["call", "--matrix", "m.tsv", "--out", "o", "--cov-threshold", "0.02"]
    )
    with pytest.raises(SystemExit, match="--legacy only"):
        cli._config_from_args(args)


def test_defaults_resolve_to_the_frozen_profiles():
    args = build_parser().parse_args(["call", "--matrix", "m.tsv", "--out", "o"])
    config = cli._config_from_args(args)
    assert config.fdr_replicates == 99
    assert config.target_fdr == 0.05
    assert config.profile.is_reference_default()
    assert config.axis_mode == "descriptive"
    assert config.mask_mode == "values_with_mask"
    args = build_parser().parse_args(
        ["call", "--matrix", "m.tsv", "--out", "o", "--legacy",
         "--axis-annotations", "experimental-inference", "--fdr-replicates", "0"]
    )
    with pytest.raises(SystemExit, match="fdr_replicates"):
        cli._config_from_args(args)


def test_compare_requires_existing_outputs(tmp_path):
    with pytest.raises(SystemExit, match="--regions-a and --regions-b"):
        cli.main(["compare", "--out", str(tmp_path)])


def test_legacy_without_its_dependencies_exits_1_with_an_install_hint(
    inputs, tmp_path, monkeypatch, capsys
):
    """A default install has no numba or scikit-learn. --legacy must stop with
    a message naming them, not a traceback and not a fallback to another
    method. The hint names the packages, never a bare ``pip install eiger``:
    that distribution name on the public index belongs to another project."""

    import functions

    # an earlier test may have imported it; hide both routes to the module
    monkeypatch.delattr(functions, "processing", raising=False)
    monkeypatch.setitem(sys.modules, "functions.processing", None)
    assert _call(inputs, tmp_path / "out", "--legacy") == 1
    err = capsys.readouterr().err
    assert "pip install numba scikit-learn" in err
    assert "pip install 'eiger" not in err
    assert not (tmp_path / "out" / "regions.csv").exists()
