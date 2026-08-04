from __future__ import annotations

import sys
import gzip
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO
sys.path.insert(0, str(ROOT / "old_code" / "beta_2026"))

from dmr_pipeline import _normalise_location_reference  # noqa: E402
from repair_eiger_coordinates import (  # noqa: E402
    _reference_positions,
    _repair_frame,
)


def test_wgbstools_cpg_index_is_auto_detected():
    source = pd.DataFrame(
        [
            ["chr1", 10469, 1],
            ["chr1", 10471, 2],
            ["chr1", 10484, 3],
            ["chr2", 10000, 4],
            ["chr2", 10008, 5],
        ]
    )
    locations, interpretation = _normalise_location_reference(source)

    assert interpretation == "cpg-index"
    assert locations["end"].tolist() == [10470, 10472, 10485, 10001, 10009]
    assert locations["reference_cpg_index"].tolist() == [1, 2, 3, 4, 5]


def test_bed_end_is_auto_detected():
    source = pd.DataFrame(
        [
            ["chr1", 100, 101],
            ["chr1", 110, 111],
            ["chr1", 125, 126],
        ]
    )
    locations, interpretation = _normalise_location_reference(source)

    assert interpretation == "end"
    assert locations["end"].tolist() == [101, 111, 126]


def test_ambiguous_third_column_requires_override():
    source = pd.DataFrame(
        [
            ["chr1", 100, 50],
            ["chr1", 110, 75],
            ["chr1", 125, 110],
        ]
    )
    with pytest.raises(ValueError, match="--location-third-column"):
        _normalise_location_reference(source)


def test_existing_dmr_coordinates_can_be_repaired(tmp_path):
    reference = tmp_path / "CpG.bed.gz"
    with gzip.open(reference, "wt") as handle:
        handle.write("chr1\t100\t1\nchr1\t110\t2\nchr1\t125\t3\n")
    frame = pd.DataFrame(
        {
            "dmr_id": ["stable_id"],
            "chr": ["chr1"],
            "start": [100],
            "end": [3],
            "cpg_start": [0],
            "cpg_end": [2],
            "length_bp": [0],
        }
    )
    positions = _reference_positions(reference, {0, 2})
    repaired = _repair_frame(frame, positions, Path("dmrs.csv"))

    assert repaired.loc[0, "dmr_id"] == "stable_id"
    assert repaired.loc[0, "start"] == 100
    assert repaired.loc[0, "end"] == 126
    assert repaired.loc[0, "length_bp"] == 26
