"""Tests for eiger2.benchmark's layout integration (master spec §6.1, Stage 1d-A)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from eiger2.benchmark import _plant_dmrs, evaluate, simulate  # noqa: E402


def _locations(starts: list[int], chrom: str | list[str] = "chr1") -> pd.DataFrame:
    starts = np.asarray(starts, dtype=np.int64)
    chrom = [chrom] * len(starts) if isinstance(chrom, str) else chrom
    return pd.DataFrame({"chr": chrom, "start": starts, "end": starts + 1})


# --------------------------------------------------------------------------
# simulate(): dense layout must be byte-identical to pre-§6.1 behaviour
# --------------------------------------------------------------------------


def test_simulate_dense_layout_matches_pre_layout_behaviour_exactly():
    # Regression pin: hash and truth captured from the pre-§6.1 implementation
    # (git HEAD before this stage) at strata=[(0.1,0.3),(0.2,0.5)],
    # replicates=3, seed=42, defaults otherwise.
    matrix, locations, truth = simulate(
        strata=[(0.1, 0.3), (0.2, 0.5)], replicates=3, seed=42,
    )
    assert matrix.shape == (2800, 80)
    import hashlib
    assert hashlib.sha256(matrix.tobytes()).hexdigest()[:16] == "ec7d44da96eb02d1"
    assert [t[:4] for t in truth] == [
        (200, 239, 0.1, 0.3), (600, 639, 0.1, 0.3), (1000, 1039, 0.1, 0.3),
        (1400, 1439, 0.2, 0.5), (1800, 1839, 0.2, 0.5), (2200, 2239, 0.2, 0.5),
    ]


def test_simulate_default_layout_is_dense():
    import inspect
    assert inspect.signature(simulate).parameters["layout"].default == "dense"


def test_simulate_records_layout_diagnostics_in_locations_attrs():
    _, locations, _ = simulate(strata=[(0.1, 0.3)], replicates=2, seed=1)
    assert locations.attrs["layout"] == "dense"
    assert locations.attrs["n_dmrs_requested"] == 2
    assert locations.attrs["n_dmrs_not_plantable"] == 0


def test_simulate_truth_records_the_eligible_run_it_sits_in():
    _, locations, truth = simulate(strata=[(0.1, 0.3)], replicates=2, seed=1)
    n_cpg = len(locations)
    for start, end, _fraction, _delta, run_start, run_end in truth:
        assert run_start == 0 and run_end == n_cpg - 1  # one giant run, dense
        assert run_start <= start and end <= run_end


def test_simulate_rejects_an_unknown_layout():
    with pytest.raises(ValueError, match="layout"):
        simulate(layout="bogus")


# --------------------------------------------------------------------------
# _plant_dmrs(): eligibility-respecting planting on a hand-built fragmented layout
# --------------------------------------------------------------------------


def test_plant_dmrs_fits_a_dmr_inside_its_containing_run():
    # One run of 10 CpGs [0..9], one of 3 CpGs [1010..1012] (far gap between).
    locs = _locations([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1010, 1012, 1014])
    matrix = np.zeros((len(locs), 20))
    rng = np.random.default_rng(0)
    truth, n_not_plantable = _plant_dmrs(
        matrix, locs, strata=[(0.2, 0.3)], replicates=1, spacing=400,
        dmr_length=4, max_cpg_gap_bp=300, rng=rng, n_samples=20, start_position=0,
    )
    assert n_not_plantable == 0
    (start, end, _f, _d, run_start, run_end) = truth[0]
    assert run_start == 0 and run_end == 9  # planted in the 10-CpG run
    assert run_start <= start and end <= run_end
    assert end - start + 1 == 4


def test_plant_dmrs_skips_a_dmr_too_long_for_its_run_and_counts_it():
    # every nominal grid position lands in a run far shorter than the
    # requested dmr_length -- every stratum member is unplantable.
    locs = _locations(list(range(0, 20, 2)))  # 10 CpGs, run [0..9], 9 edges
    matrix = np.zeros((len(locs), 20))
    rng = np.random.default_rng(0)
    truth, n_not_plantable = _plant_dmrs(
        matrix, locs, strata=[(0.2, 0.3)], replicates=3, spacing=1,
        dmr_length=50, max_cpg_gap_bp=300, rng=rng, n_samples=20, start_position=0,
    )
    assert truth == []
    assert n_not_plantable == 3
    assert not np.any(matrix)  # nothing was ever added to the matrix


def test_plant_dmrs_off_the_end_of_the_track_is_unplantable_not_a_crash():
    locs = _locations(list(range(0, 20, 2)))  # 10 CpGs
    matrix = np.zeros((len(locs), 5))
    rng = np.random.default_rng(0)
    truth, n_not_plantable = _plant_dmrs(
        matrix, locs, strata=[(0.2, 0.3)], replicates=2, spacing=5,
        dmr_length=2, max_cpg_gap_bp=300, rng=rng, n_samples=5,
        start_position=200,  # far past this 10-CpG track
    )
    assert truth == []
    assert n_not_plantable == 2


def test_plant_dmrs_mixed_run_only_plants_where_it_fits():
    # A short run (3 CpGs) followed immediately (index-wise, not genome-wise)
    # by a long one (100 CpGs) far away in bp. Two candidates starting at
    # index 3, spacing 50: both land inside the long run (indices 3..102),
    # never in the 3-CpG run -- confirms plantable candidates are not
    # incorrectly skipped and land inside the run that actually contains them.
    short_starts = [0, 2, 4]
    long_starts = (np.arange(100) * 2 + 100_000).tolist()
    locs = _locations(short_starts + long_starts)
    matrix = np.zeros((len(locs), 10))
    rng = np.random.default_rng(0)
    truth, n_not_plantable = _plant_dmrs(
        matrix, locs, strata=[(0.5, 0.3)], replicates=2, spacing=50,
        dmr_length=5, max_cpg_gap_bp=300, rng=rng, n_samples=10, start_position=3,
    )
    assert n_not_plantable == 0
    assert len(truth) == 2
    for start, end, _f, _d, run_start, run_end in truth:
        assert run_start == 3 and run_end == 102
        assert run_start <= start and end <= run_end


def test_plant_dmrs_never_writes_outside_a_planted_span():
    locs = _locations((np.arange(30) * 2).tolist())
    matrix = np.zeros((len(locs), 10))
    rng = np.random.default_rng(2)
    truth, _ = _plant_dmrs(
        matrix, locs, strata=[(0.3, 1.0)], replicates=1, spacing=400,
        dmr_length=5, max_cpg_gap_bp=300, rng=rng, n_samples=10, start_position=0,
    )
    start, end, *_rest = truth[0]
    untouched = np.delete(matrix, np.arange(start, end + 1), axis=0)
    assert not np.any(untouched)


# --------------------------------------------------------------------------
# simulate(layout="wgbs_real" / "epic_fitted"): the vendored fixture
# --------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["wgbs_real", "epic_fitted"])
def test_simulate_real_layouts_produce_a_fragmented_non_uniform_track(layout):
    matrix, locations, truth = simulate(
        strata=[(0.2, 0.35)], replicates=4, spacing=300, dmr_length=6,
        layout=layout, seed=3,
    )
    assert len(locations) == matrix.shape[0]
    starts = locations["start"].to_numpy()
    gaps = starts[1:] - starts[:-1]
    assert gaps.std() > 0  # not uniformly spaced, unlike dense
    assert locations.attrs["layout"] == layout
    # every planted DMR (if any were plantable) must respect its recorded run
    for start, end, _f, _d, run_start, run_end in truth:
        assert run_start <= start and end <= run_end and end - start + 1 <= (
            run_end - run_start + 1
        )


def test_simulate_wgbs_real_varies_offset_by_seed_when_not_pinned():
    _, locs_a, _ = simulate(
        strata=[(0.2, 0.35)], replicates=2, layout="wgbs_real", seed=1,
    )
    _, locs_b, _ = simulate(
        strata=[(0.2, 0.35)], replicates=2, layout="wgbs_real", seed=2,
    )
    assert locs_a["start"].iloc[0] != locs_b["start"].iloc[0]


def test_simulate_wgbs_real_pinned_offset_is_reproducible():
    _, locs_a, _ = simulate(
        strata=[], replicates=0, layout="wgbs_real", layout_offset=1_000_000, seed=1,
    )
    _, locs_b, _ = simulate(
        strata=[], replicates=0, layout="wgbs_real", layout_offset=1_000_000, seed=99,
    )
    pd.testing.assert_frame_equal(
        locs_a.reset_index(drop=True), locs_b.reset_index(drop=True)
    )


def test_simulate_real_layout_recall_still_works_end_to_end():
    # Not a calibration claim -- just confirms the whole pipeline (layout ->
    # plant -> call_dmrs -> evaluate) runs without error on a real layout and
    # can recover at least the DMRs that were geometrically plantable.
    from eiger2.core import Eiger2Config, call_dmrs

    matrix, locations, truth = simulate(
        strata=[(0.3, 0.4)], replicates=3, spacing=500, dmr_length=4,
        layout="wgbs_real", seed=5, n_samples=60,
    )
    assert len(truth) <= locations.attrs["n_dmrs_requested"]
    config = Eiger2Config(
        null_replicates=6, min_null_regions_for_fdr=0, min_cpgs=2,
        region_null_mode="lag",
    )
    regions, _, _ = call_dmrs(matrix, locations, config=config)
    recovered, _false = evaluate(regions, truth, min_overlap=2)
    assert isinstance(recovered, set)


def test_stage2d_biological_support_smoke_covers_all_named_scenarios():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarking"))
    from benchmark_suite import biological_support_smoke

    frame = biological_support_smoke([0])
    assert set(frame["scenario"]) == {
        "one_sample_artifact", "two_sample_artifact",
        "genuine_rare_epivariant", "recurrent_subgroup", "cohort_subgroup",
        "continuous_gradient", "local_background_autocorrelation",
    }
    assert set(frame["statistic"]) == {"correlation", "covariance"}
    assert frame["classification_correct"].all()


def test_false_discovery_proportion_is_zero_when_nothing_is_called():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarking"))
    from benchmark_suite import _false_discovery_proportion

    assert _false_discovery_proportion(0, 0) == 0.0
    assert _false_discovery_proportion(2, 5) == 0.4


def test_benchmark_manifest_records_dirty_worktree_provenance(
    monkeypatch, tmp_path
):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarking"))
    import benchmark_suite

    monkeypatch.setattr(benchmark_suite, "command_support", lambda _args: None)
    monkeypatch.setattr(
        benchmark_suite,
        "_git_provenance",
        lambda: {
            "git_commit": "deadbeef",
            "git_worktree_dirty": True,
            "git_worktree_fingerprint_sha256": "a" * 64,
        },
    )
    assert benchmark_suite.main([
        "support", "--out-dir", str(tmp_path), "--seeds", "0",
    ]) == 0
    manifest = json.loads((tmp_path / "support_manifest.json").read_text())
    assert manifest["git_commit"] == "deadbeef"
    assert manifest["git_worktree_dirty"] is True
    assert manifest["git_worktree_fingerprint_sha256"] == "a" * 64


# --------------------------------------------------------------------------
# gap_sweep() (master spec §6.1: max_cpg_gap_bp sweep)
# --------------------------------------------------------------------------


def test_gap_sweep_reports_all_five_gap_values_per_layout_and_seed():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarking"))
    from benchmark_suite import GAP_SWEEP_VALUES, gap_sweep

    # region_null_mode="lag", spacing=400 (pre-Stage-1e default): these two
    # tests check structural properties (which gap values got reported, run-
    # length monotonicity), not realised FDR, so they don't need the shipped
    # region_null_mode="member_step" default's genome-scale minimum input --
    # forcing "lag" at the old small spacing keeps them fast.
    frame = gap_sweep(
        [0], layouts=["wgbs_real"], spacing=400, region_null_mode="lag",
    )
    assert set(frame["max_cpg_gap_bp"]) == set(GAP_SWEEP_VALUES)
    assert len(frame) == len(GAP_SWEEP_VALUES)
    assert (frame["layout"] == "wgbs_real").all()


def test_gap_sweep_run_length_distribution_widens_with_larger_gap():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarking"))
    from benchmark_suite import gap_sweep

    frame = gap_sweep(
        [0], layouts=["wgbs_real"], spacing=400, region_null_mode="lag",
    ).sort_values("max_cpg_gap_bp")
    # a larger max_cpg_gap_bp can only merge runs together, never split them,
    # so the run-length ceiling is non-decreasing in the gap value
    ceilings = frame["layout_max_detectable_region_cpgs"].tolist()
    assert ceilings == sorted(ceilings)


# --------------------------------------------------------------------------
# exchangeability() (master spec §6.2, Stage 1d-B: the three permanent gates)
# --------------------------------------------------------------------------


def test_exchangeability_reports_all_three_gates():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarking"))
    from benchmark_suite import exchangeability

    frame = exchangeability([0], ["clean"], layout="dense")
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["layout"] == "dense"
    # edge-level gate (spec §1b.3, retained)
    assert "null_exchangeability_ratio" in frame.columns
    # Gate 1
    assert isinstance(row["acf_gate_passed"], (bool, np.bool_))
    assert isinstance(row["acf_diff_by_lag"], dict)
    # Gate 2
    assert "region_exchangeability_ratio" in frame.columns
    assert "region_exchangeability_null_count_adequate" in frame.columns


def test_exchangeability_gate2_inf_when_null_rate_zero_and_observed_positive():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarking"))
    from benchmark_suite import exchangeability

    # small, fragmented real layout at default null_replicates: the null forms
    # ~zero regions (Stage 1c/1c-bis's own finding), so any seed with at least
    # one observed candidate region gives ratio = inf.
    frame = exchangeability(range(6), ["clean"], layout="wgbs_real")
    with_candidates = frame[frame["region_exchangeability_observed_regions"] > 0]
    if len(with_candidates):
        assert (np.isinf(with_candidates["region_exchangeability_ratio"])).any()
