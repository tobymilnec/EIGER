"""Tests for eiger2.layouts (master spec §6.1, Stage 1d-A)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from eiger2.core import Eiger2Config, eligible_edges, length_stratum  # noqa: E402
from eiger2.layouts import (  # noqa: E402
    _LENGTH_STRATUM_LOWER_BOUND,
    describe_layout,
    eligible_run_lengths,
    eligible_run_spans,
    plantability_fraction,
    realised_length_strata,
    window_eligibility_fraction,
)


def _locations(starts: list[int], chrom: str | list[str] = "chr1") -> pd.DataFrame:
    starts = np.asarray(starts, dtype=np.int64)
    chrom = [chrom] * len(starts) if isinstance(chrom, str) else chrom
    return pd.DataFrame({"chr": chrom, "start": starts, "end": starts + 1})


def test_eligible_run_lengths_one_contiguous_run():
    # 5 CpGs, 2 bp apart -- one gap-eligible run of 4 edges under a generous cap
    locs = _locations([0, 2, 4, 6, 8])
    runs = eligible_run_lengths(locs, max_cpg_gap_bp=300)
    assert runs.tolist() == [4]


def test_eligible_run_lengths_splits_on_a_large_gap():
    # gap of 1000 bp between CpG 2 and 3 exceeds the 300 bp cap -> two runs
    locs = _locations([0, 2, 4, 1004, 1006, 1008])
    runs = eligible_run_lengths(locs, max_cpg_gap_bp=300)
    assert sorted(runs.tolist()) == [2, 2]


def test_eligible_run_lengths_splits_on_chromosome_boundary():
    locs = _locations([0, 2, 4, 100, 102], chrom=["chr1"] * 3 + ["chr2"] * 2)
    runs = eligible_run_lengths(locs, max_cpg_gap_bp=300)
    assert sorted(runs.tolist()) == [1, 2]


def test_eligible_run_lengths_isolated_cpg_is_a_zero_length_run():
    locs = _locations([0, 2, 5000, 5002, 5004])
    eligible_run_lengths(locs, max_cpg_gap_bp=300)
    # [0,2] is one eligible edge (run len 1); the far CpG at 5000 is isolated on
    # its left (gap too big) but eligible on its right -> [5000,5002,5004] is a
    # run of length 2. Nothing here is isolated on both sides in this fixture,
    # so exercise that case explicitly with a genuinely solitary CpG:
    solo = _locations([0, 2, 10_000, 20_000, 20_002])
    solo_runs = eligible_run_lengths(solo, max_cpg_gap_bp=300)
    assert sorted(solo_runs.tolist()) == [0, 1, 1]


def test_eligible_run_lengths_matches_eligible_edges_exactly():
    # Consistency, not reimplementation: sum of run lengths (in edges) must equal
    # the number of eligible edges eligible_edges() itself reports.
    rng = np.random.default_rng(7)
    starts = np.cumsum(rng.integers(1, 500, size=500)).astype(np.int64)
    locs = _locations(starts.tolist())
    eligible = eligible_edges(locs, Eiger2Config(max_cpg_gap_bp=300))
    runs = eligible_run_lengths(locs, max_cpg_gap_bp=300)
    assert runs.sum() == eligible.sum()


def test_eligible_run_spans_partitions_every_cpg_index_exactly_once():
    locs = _locations([0, 2, 4, 1004, 1006, 1008, 1010])
    spans = eligible_run_spans(locs, max_cpg_gap_bp=300)
    covered = []
    for start, end in spans:
        covered.extend(range(start, end + 1))
    assert covered == list(range(len(locs)))  # every index, in order, exactly once


def test_eligible_run_spans_cpg_counts_match_run_lengths_plus_one():
    locs = _locations([0, 2, 4, 6, 1006, 1008, 20_000, 20_002, 20_004, 20_006, 20_008])
    lengths = eligible_run_lengths(locs, max_cpg_gap_bp=300)
    spans = eligible_run_spans(locs, max_cpg_gap_bp=300)
    assert [end - start + 1 for start, end in spans] == (lengths + 1).tolist()


def test_eligible_run_spans_isolated_cpg_is_a_one_cpg_span():
    locs = _locations([0, 2, 10_000, 20_000, 20_002])
    spans = eligible_run_spans(locs, max_cpg_gap_bp=300)
    assert (2, 2) in spans  # the lone CpG at index 2 is its own one-CpG span


def test_plantability_fraction_near_zero_on_one_giant_run():
    # Only the last (dmr_length - 1) start positions near the very end of the
    # one giant run cannot fit a dmr_length-CpG span -- boundary effect, not a
    # bug: 39 of 2000 positions here.
    locs = _locations((np.arange(2000) * 2).tolist())
    assert plantability_fraction(locs, max_cpg_gap_bp=300, dmr_length=40) == (
        pytest.approx(39 / 2000)
    )


def test_plantability_fraction_hand_derived():
    # two runs: 5 CpGs [0..4], 2 CpGs [5000..5001] (far apart)
    locs = _locations([0, 2, 4, 6, 8, 5000, 5002])
    # dmr_length=4: run of 5 admits 2 positions (5-4+1), run of 2 admits 0.
    # 2 plantable / 7 total candidates -> 1 - 2/7 unplantable
    assert plantability_fraction(locs, max_cpg_gap_bp=300, dmr_length=4) == (
        pytest.approx(1.0 - 2 / 7)
    )
    # dmr_length longer than every run: nothing is plantable anywhere
    assert plantability_fraction(locs, max_cpg_gap_bp=300, dmr_length=6) == 1.0


def test_plantability_fraction_increases_with_dmr_length():
    rng = np.random.default_rng(11)
    locs = _locations(np.cumsum(rng.integers(1, 400, size=3000)).tolist())
    fractions = [
        plantability_fraction(locs, max_cpg_gap_bp=300, dmr_length=L)
        for L in (4, 10, 50)
    ]
    assert fractions == sorted(fractions)


def test_plantability_fraction_empty_input_is_nan():
    assert np.isnan(
        plantability_fraction(_locations([]), max_cpg_gap_bp=300, dmr_length=4)
    )


def test_window_eligibility_fraction_hand_derived():
    # two runs: 5 CpGs [0..4] (edges 0,1,2,3 eligible), 2 CpGs [5,6] (edge 5
    # eligible); edge 4 (between the runs) ineligible. eligible edges (5
    # total, by index): 0,1,2,3,5.
    locs = _locations([0, 2, 4, 6, 8, 5000, 5002])

    # w=2: offset0 = 2//2-1 = 0, span=1 -- window at edge i just needs edge i
    # itself eligible, which is true of every edge in the denominator by
    # construction -> all 5.
    assert window_eligibility_fraction(locs, max_cpg_gap_bp=300, window=2) == 1.0

    # w=4: offset0 = 4//2-1 = 1, span=3, base = i-1. In-bounds i in {1,2,3,4}.
    # i=1 (base=0): edges[0:3]=T,T,T -> ok. i=2 (base=1): edges[1:4]=T,T,T -> ok.
    # i=3 (base=2): edges[2:5]=T,T,F -> not ok (crosses the ineligible edge 4).
    # i=4 (base=3): edges[3:6]=T,F,T -> not ok. i=0,5 out of bounds -> not ok.
    # 2 of 5 eligible edges admit an aligned 4-CpG window.
    assert window_eligibility_fraction(locs, max_cpg_gap_bp=300, window=4) == (
        pytest.approx(2 / 5)
    )

    # w=6: neither run has enough CpGs at any alignment -> 0
    assert window_eligibility_fraction(locs, max_cpg_gap_bp=300, window=6) == 0.0


def test_window_eligibility_fraction_one_at_window_one():
    # w=1 is trivially eligible at every eligible edge (any run has >=1 CpG)
    locs = _locations([0, 2, 4, 5000, 5002, 5004])
    assert window_eligibility_fraction(locs, max_cpg_gap_bp=300, window=1) == 1.0


def test_window_eligibility_fraction_decreases_with_window():
    rng = np.random.default_rng(13)
    locs = _locations(np.cumsum(rng.integers(1, 400, size=2000)).tolist())
    fractions = [
        window_eligibility_fraction(locs, max_cpg_gap_bp=300, window=w)
        for w in (1, 2, 4, 6, 8, 12)
    ]
    assert fractions == sorted(fractions, reverse=True)


def test_window_eligibility_fraction_empty_input_is_nan():
    assert np.isnan(
        window_eligibility_fraction(_locations([]), max_cpg_gap_bp=300, window=4)
    )


def test_window_eligibility_fraction_matches_master_spec_7_corrected_numbers():
    # Regression pin for the corrected master spec §7 / docs/MULTISCALE_BLOCKED.md
    # figures, on the real vendored fixture (max_cpg_gap_bp=300). The original
    # ("shelved") version of this module used a looser "run has enough CpGs
    # somewhere" criterion that overstated eligibility (chr1 w=4: 97.2%); this
    # pins the alignment-aware numbers that superseded it.
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "benchmarking" / "layouts")
    )
    from layouts import load_wgbs_layout

    fixture = str(
        Path(__file__).resolve().parents[1]
        / "benchmarking" / "layouts" / "wgbs_layout_hg38.npz"
    )
    chr1, _ = load_wgbs_layout(fixture, chrom="chr1")
    chr19, _ = load_wgbs_layout(fixture, chrom="chr19")

    expected_chr1 = {4: 0.875, 8: 0.713, 12: 0.610}
    expected_chr19 = {4: 0.964, 8: 0.907, 12: 0.863}
    for w, expected in expected_chr1.items():
        assert window_eligibility_fraction(chr1, 300, w) == pytest.approx(
            expected, abs=5e-4
        )
    for w, expected in expected_chr19.items():
        assert window_eligibility_fraction(chr19, 300, w) == pytest.approx(
            expected, abs=5e-4
        )


def test_describe_layout_matches_hand_derived_numbers():
    # two runs of eligible length 3 and 1 edge (chr1: 0-2-4-6 is 3 edges; chr1
    # after a big gap: 5000-5002 is 1 edge), max_cpg_gap_bp=300
    locs = _locations([0, 2, 4, 6, 5000, 5002])
    out = describe_layout(locs, max_cpg_gap_bp=300)
    assert out["layout_n_cpg"] == 6
    assert out["layout_n_chromosomes"] == 1
    assert out["layout_eligible_run_lengths_edges"]["n_runs"] == 2
    assert out["layout_eligible_run_lengths_edges"]["max"] == 3
    # longest run (3 edges) + 1 CpG
    assert out["layout_max_detectable_region_cpgs"] == 4


def test_describe_layout_reports_depth_when_given():
    locs = _locations([0, 2, 4, 6])
    depth = np.array([2, 8, 20, 3])  # sorted: 2, 3, 8, 20 -> median (3+8)/2
    out = describe_layout(locs, max_cpg_gap_bp=300, depth=depth)
    assert out["layout_depth"]["median"] == pytest.approx(5.5)
    assert out["layout_depth"]["fraction_ge5"] == pytest.approx(0.5)


def test_describe_layout_classifies_contiguous_vs_fragmented():
    dense = _locations((np.arange(2000) * 2).tolist())
    fragmented = _locations(np.cumsum(
        np.random.default_rng(3).integers(1, 400, size=2000)
    ).tolist())
    assert describe_layout(dense, max_cpg_gap_bp=300)["layout_class"] == "contiguous"
    assert describe_layout(fragmented, max_cpg_gap_bp=300)["layout_class"] in {
        "fragmented", "highly_fragmented",
    }


def test_describe_layout_empty_input_does_not_crash():
    # Inherited from the vendored, already-validated fixture algorithm
    # unchanged: with zero CpGs there are zero edges, but the run-boundary
    # scaffolding still emits one trivial zero-length "run" via its sentinel
    # boundaries. Real inputs always have CpGs, so this never fires in
    # practice; the assertion here is "does not crash / stays well-formed",
    # not a claim that n_runs is semantically meaningful at n_cpg=0.
    locs = _locations([])
    out = describe_layout(locs, max_cpg_gap_bp=300)
    assert out["layout_n_cpg"] == 0
    assert out["layout_eligible_run_lengths_edges"]["max"] == 0


# --------------------------------------------------------------------------
# realised_length_strata()
# --------------------------------------------------------------------------


def test_length_stratum_lower_bounds_match_core_length_stratum_exactly():
    # _LENGTH_STRATUM_LOWER_BOUND is a deliberate, independent copy of
    # core.length_stratum()'s bin boundaries (see that constant's docstring
    # note); this pins the two together so they cannot silently drift.
    for stratum, lower in _LENGTH_STRATUM_LOWER_BOUND.items():
        assert length_stratum(lower) == stratum
        if lower > 4:
            assert length_stratum(lower - 1) != stratum


def test_realised_length_strata_all_five_when_ceiling_is_generous():
    assert realised_length_strata(1000) == (
        "4-5", "6-9", "10-19", "20-49", "50+",
    )


def test_realised_length_strata_collapses_upward_when_ceiling_is_low():
    # a layout whose longest possible region is 15 CpGs can never populate
    # 20-49 or 50+ -- they collapse away, "10-19" becomes the effective top
    assert realised_length_strata(15) == ("4-5", "6-9", "10-19")


def test_realised_length_strata_exactly_at_a_boundary_includes_it():
    assert realised_length_strata(50) == ("4-5", "6-9", "10-19", "20-49", "50+")
    assert realised_length_strata(49) == ("4-5", "6-9", "10-19", "20-49")


def test_realised_length_strata_empty_when_ceiling_below_smallest_stratum():
    assert realised_length_strata(3) == ()
    assert realised_length_strata(0) == ()


# --------------------------------------------------------------------------
# Stage 1d-C §2 — C.1 block permutation: real-layout run-length confirmation
# --------------------------------------------------------------------------


def test_c1_block_permutation_b11_applies_to_roughly_the_top_decile_of_runs():
    # STAGE1D_C_COMPRESSED_SPEC.md §2: at the measured rho=0.503 and
    # tolerance=0.05, Gate 1 needs b >~ 11, i.e. runs of >= 22 edges (two
    # blocks). Pin that this is "roughly the top decile" on the real WGBS
    # fixture, not a looser or tighter fraction.
    import sys
    from pathlib import Path
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "benchmarking" / "layouts")
    )
    from layouts import load_wgbs_layout

    fixture = str(
        Path(__file__).resolve().parents[1]
        / "benchmarking" / "layouts" / "wgbs_layout_hg38.npz"
    )
    chr1, _ = load_wgbs_layout(fixture, chrom="chr1")
    lengths = eligible_run_lengths(chr1, max_cpg_gap_bp=300)
    fraction_ge_22 = float(np.mean(lengths >= 22))
    assert 0.08 <= fraction_ge_22 <= 0.13  # "roughly the top decile"
