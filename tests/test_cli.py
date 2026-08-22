"""Tests for eiger2.cli's argument parsing (Stage 1f Task 1a, master spec pre-release).

parse_args()/`_config_from_args` do not touch the filesystem, so these tests exercise
argument round-tripping directly without needing real input files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

import eiger2.cli as cli  # noqa: E402
from eiger2.cli import _config_from_args, build_parser  # noqa: E402

_BASE_ARGS = [
    "call", "--betas", "betas.tsv", "--locations", "cpgs.bed", "--out", "out/",
]


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

    out = tmp_path / "legacy"
    argv = ["call", "--matrix", "unused.tsv", "--out", str(out), "--legacy"]
    assert cli.main(argv) == 0
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["caller_manifest"]["fdr"] is None
