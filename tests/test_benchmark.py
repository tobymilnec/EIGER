"""Core tests for the simulator bundled with the EIGER software package.

Paper-only layouts, fitted marginals, comparison harnesses, and their fixtures live in
the separate ``eiger_paper`` repository. These tests deliberately exercise only the
self-contained dense/unimodal simulator shipped in the minimal software release.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from eiger2.benchmark import _plant_dmrs, simulate  # noqa: E402


def _locations(starts: list[int], chrom: str | list[str] = "chr1") -> pd.DataFrame:
    starts = np.asarray(starts, dtype=np.int64)
    chrom = [chrom] * len(starts) if isinstance(chrom, str) else chrom
    return pd.DataFrame({"chr": chrom, "start": starts, "end": starts + 1})


def test_simulate_dense_layout_matches_pre_layout_behaviour_exactly():
    matrix, _locations_frame, truth = simulate(
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
    for start, end, _fraction, _delta, run_start, run_end, _effective in truth:
        assert run_start == 0 and run_end == n_cpg - 1
        assert run_start <= start and end <= run_end


def test_simulate_rejects_an_unknown_layout():
    with pytest.raises(ValueError, match="layout"):
        simulate(layout="bogus")


def test_plant_dmrs_fits_a_dmr_inside_its_containing_run():
    locs = _locations([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1010, 1012, 1014])
    matrix = np.zeros((len(locs), 20))
    truth, n_not_plantable, _plants = _plant_dmrs(
        matrix, locs, strata=[(0.2, 0.3)], replicates=1, spacing=400,
        dmr_length=4, max_cpg_gap_bp=300, rng=np.random.default_rng(0),
        n_samples=20, baseline_mu=np.full(len(locs), 0.5), start_position=0,
    )
    assert n_not_plantable == 0
    start, end, _fraction, _delta, run_start, run_end = truth[0]
    assert run_start == 0 and run_end == 9
    assert run_start <= start and end <= run_end
    assert end - start + 1 == 4


def test_plant_dmrs_skips_a_dmr_too_long_for_its_run_and_counts_it():
    locs = _locations(list(range(0, 20, 2)))
    matrix = np.zeros((len(locs), 20))
    truth, n_not_plantable, _plants = _plant_dmrs(
        matrix, locs, strata=[(0.2, 0.3)], replicates=3, spacing=1,
        dmr_length=50, max_cpg_gap_bp=300, rng=np.random.default_rng(0),
        n_samples=20, baseline_mu=np.full(len(locs), 0.5), start_position=0,
    )
    assert truth == []
    assert n_not_plantable == 3
    assert not np.any(matrix)


def test_plant_dmrs_off_the_end_of_the_track_is_unplantable_not_a_crash():
    locs = _locations(list(range(0, 20, 2)))
    matrix = np.zeros((len(locs), 5))
    truth, n_not_plantable, _plants = _plant_dmrs(
        matrix, locs, strata=[(0.2, 0.3)], replicates=2, spacing=5,
        dmr_length=2, max_cpg_gap_bp=300, rng=np.random.default_rng(0),
        n_samples=5, baseline_mu=np.full(len(locs), 0.5), start_position=200,
    )
    assert truth == []
    assert n_not_plantable == 2


def test_plant_dmrs_mixed_run_only_plants_where_it_fits():
    locs = _locations([0, 2, 4] + (np.arange(100) * 2 + 100_000).tolist())
    matrix = np.zeros((len(locs), 10))
    truth, n_not_plantable, _plants = _plant_dmrs(
        matrix, locs, strata=[(0.5, 0.3)], replicates=2, spacing=50,
        dmr_length=5, max_cpg_gap_bp=300, rng=np.random.default_rng(0),
        n_samples=10, baseline_mu=np.full(len(locs), 0.5), start_position=3,
    )
    assert n_not_plantable == 0
    assert len(truth) == 2
    for start, end, _fraction, _delta, run_start, run_end in truth:
        assert run_start == 3 and run_end == 102
        assert run_start <= start and end <= run_end


def test_plant_dmrs_never_writes_outside_a_planted_span():
    locs = _locations((np.arange(30) * 2).tolist())
    matrix = np.zeros((len(locs), 10))
    truth, _, _plants = _plant_dmrs(
        matrix, locs, strata=[(0.3, 1.0)], replicates=1, spacing=400,
        dmr_length=5, max_cpg_gap_bp=300, rng=np.random.default_rng(2),
        n_samples=10, baseline_mu=np.full(len(locs), 0.5), start_position=0,
    )
    start, end, *_rest = truth[0]
    untouched = np.delete(matrix, np.arange(start, end + 1), axis=0)
    assert not np.any(untouched)


PRE_WP6_DENSE_MATRIX_HASHES: tuple[tuple[dict, str], ...] = (
    ({"strata": [(0.1, 0.3), (0.2, 0.5)], "replicates": 3, "seed": 42},
     "ec7d44da96eb02d1"),
    ({"strata": [], "seed": 7, "spacing": 1000}, "5bfaaada0c391c3f"),
    ({"strata": [(0.25, 0.25)], "replicates": 2, "seed": 3,
      "n_samples": 20, "n_latent": 3, "noise_sd": 0.06,
      "latent_loading_sd": 0.05, "missing_rate": 0.2},
     "1404d774590e27cc"),
)


@pytest.mark.parametrize("kwargs,digest", PRE_WP6_DENSE_MATRIX_HASHES)
def test_simulate_unimodal_marginal_is_byte_identical_to_pre_wp6(kwargs, digest):
    import hashlib

    matrix, _locations_frame, _truth = simulate(**kwargs)
    assert hashlib.sha256(matrix.tobytes()).hexdigest()[:16] == digest


def test_simulate_defaults_to_the_unimodal_marginal():
    import inspect

    assert inspect.signature(simulate).parameters["marginal"].default == "unimodal"
    _matrix, locations, _truth = simulate(strata=[(0.1, 0.3)], replicates=1, seed=2)
    assert locations.attrs["marginal"] == "unimodal"


def test_simulate_rejects_an_unknown_marginal():
    with pytest.raises(ValueError, match="marginal must be one of"):
        simulate(strata=[], seed=0, marginal="lognormal")


def test_simulate_truth_records_the_realised_effective_delta():
    _matrix, _locations_frame, truth = simulate(
        strata=[(0.25, 0.30)], replicates=3, seed=4,
    )
    for entry in truth:
        assert len(entry) == 7
        _start, _end, _fraction, delta, _run_start, _run_end, effective = entry
        assert effective == pytest.approx(delta, rel=0.05)
        assert effective <= delta
