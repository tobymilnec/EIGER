"""Tests for eiger2.cli's argument parsing (Stage 1f Task 1a, master spec pre-release).

parse_args()/`_config_from_args` do not touch the filesystem, so these tests exercise
argument round-tripping directly without needing real input files.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

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
