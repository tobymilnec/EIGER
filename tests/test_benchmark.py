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


# --- The eiger_paper boundary (plan Fix S1) -------------------------------------
#
# Real coordinate layouts and marginals fitted to real cohorts are maintained in
# eiger_paper, not shipped in the wheel. The installed package must reject their
# mode names with one actionable message instead of searching for a benchmarking/
# directory that an installed package does not have, and must offer the injection
# points paper code uses instead.


@pytest.mark.parametrize("layout", ["wgbs_real", "epic_fitted"])
def test_simulate_rejects_a_paper_only_layout_by_name(layout):
    with pytest.raises(ValueError) as excinfo:
        simulate(strata=[], seed=0, layout=layout)
    message = str(excinfo.value)
    assert "eiger_paper" in message
    assert "locations" in message
    assert len(message.splitlines()) == 1


@pytest.mark.parametrize("marginal", ["bimodal_fitted", "bimodal_atlas"])
def test_simulate_rejects_a_paper_only_marginal_by_name(marginal):
    with pytest.raises(ValueError) as excinfo:
        simulate(strata=[], seed=0, marginal=marginal)
    message = str(excinfo.value)
    assert "eiger_paper" in message
    assert "marginal_sampler" in message
    assert len(message.splitlines()) == 1


def test_simulate_reads_no_files_and_needs_no_source_checkout():
    """No shipped mode may touch the filesystem: the wheel carries no fixtures."""

    import builtins

    real_open = builtins.open
    opened: list[str] = []

    def recording_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(builtins, "open", recording_open)
        simulate(strata=[(0.1, 0.3)], replicates=1, seed=11)
    finally:
        monkey.undo()
    assert opened == []


def test_advertised_layouts_and_marginals_are_the_self_contained_ones():
    from eiger2 import benchmark

    assert benchmark.LAYOUTS == ("dense",)
    assert benchmark.MARGINALS == ("unimodal",)


def test_cli_help_describes_only_installed_package_behaviour(capsys):
    from eiger2 import benchmark

    with pytest.raises(SystemExit) as excinfo:
        benchmark.main(["--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "dense" in help_text and "unimodal" in help_text
    # a paper-only mode may be named only inside the scope note that says it is
    # not shipped here, never in the scenario list the help advertises
    advertised, _, scope = help_text.partition("Scope of the installed package")
    assert "eiger_paper" in scope
    for mode in benchmark.PAPER_ONLY_LAYOUTS + benchmark.PAPER_ONLY_MARGINALS:
        assert mode not in advertised


def test_supplied_locations_replace_the_generated_layout():
    locations = _locations(
        list(range(0, 4000, 2)) + list(range(100_000, 102_000, 2)),
        chrom=["chr1"] * 2000 + ["chr2"] * 1000,
    )
    matrix, used, truth = simulate(
        strata=[(0.25, 0.3)], replicates=2, seed=5, locations=locations,
        layout="paper_layout",
    )
    assert matrix.shape == (len(locations), 80)
    assert used["chr"].tolist() == locations["chr"].tolist()
    assert used.attrs["layout"] == "paper_layout"
    # planting still respects the layout's eligible runs
    for start, end, _f, _d, run_start, run_end, _eff in truth:
        assert run_start <= start and end <= run_end


def test_supplied_marginal_sampler_hits_the_requested_per_cpg_total_sd():
    baseline = None
    target = None

    def sampler(rng, n_cpg):
        nonlocal baseline, target
        baseline = rng.uniform(0.05, 0.95, n_cpg)
        target = np.full(n_cpg, 0.02)
        return baseline, target

    matrix, locations, _truth = simulate(
        strata=[], seed=6, spacing=1000, marginal="paper_marginal",
        marginal_sampler=sampler,
    )
    assert locations.attrs["marginal"] == "paper_marginal"
    interior = (baseline > 0.2) & (baseline < 0.8)  # away from the BETA_CLIP rails
    realised = matrix[interior].std(axis=1, ddof=1)
    assert realised.mean() == pytest.approx(target[0], rel=0.05)
    assert np.corrcoef(matrix[interior].mean(axis=1), baseline[interior])[0, 1] > 0.99


def test_supplied_marginal_sampler_does_not_disturb_the_unimodal_draw_order():
    """The sampler branch is separate, so no shipped seed's matrix can move."""

    import hashlib

    def sampler(rng, n_cpg):
        return np.full(n_cpg, 0.5), np.full(n_cpg, 0.04)

    kwargs = {"strata": [(0.1, 0.3)], "replicates": 2, "seed": 42}
    shipped, _locations_frame, _truth = simulate(**kwargs)
    injected, _locations_frame, _truth = simulate(**kwargs, marginal_sampler=sampler)
    # identical draw counts in identical order, but the sampler branch scales the
    # noise and loading rows, so the matrices differ while both stay reproducible
    assert injected.shape == shipped.shape
    assert (
        hashlib.sha256(injected.tobytes()).hexdigest()
        == hashlib.sha256(
            simulate(**kwargs, marginal_sampler=sampler)[0].tobytes()
        ).hexdigest()
    )


# --- Advertised --scenario modes must run from an installed wheel ----------------
#
# call_dmrs returns rather than raises when a structural gate fails, and an
# aborted run carries no manifest["null"] block and no decision column. The
# scenario printers must report that refusal, not crash on a key the run never
# earned -- exit code zero is not complete inference.


def _aborted_manifest():
    return {
        "status": "inference_unreliable",
        "latent_structure": {"n_components": 2},
        "gates": {"structural": {
            "numerical_failure": {"passed": False, "error": "no usable pairs"},
        }},
    }


def test_summary_reports_a_refused_run_instead_of_raising():
    from eiger2.benchmark import _summary

    summary = _summary(_aborted_manifest())
    assert "threshold=refused" in summary
    assert "status=inference_unreliable" in summary
    assert "K=2" in summary


def test_summary_reports_the_threshold_of_a_completed_run():
    from eiger2.benchmark import _summary

    manifest = _aborted_manifest() | {
        "status": "partial_inference", "null": {"threshold": 0.1234},
    }
    assert "threshold=0.1234" in _summary(manifest)
    assert "status=partial_inference" in _summary(manifest)


def test_refusal_reason_names_the_failed_structural_gate():
    from eiger2.benchmark import _refusal_reason

    assert _refusal_reason(_aborted_manifest()) == (
        "numerical_failure: no usable pairs"
    )


def test_scenario_null_completes_and_reports_refusals(capsys):
    """The historical 800-CpG null scenario aborts on the structural gates now.

    It must say so per run rather than raise: the numbers in docs/EIGER2_DESIGN.md
    were produced before the current gates, and rescaling the scenario to clear
    them would move every one of them -- a results decision, not a bug fix.
    """

    from eiger2 import benchmark

    assert benchmark.main(["--scenario", "null", "--null-replicates", "2"]) == 0
    out = capsys.readouterr().out
    assert "significant at q<=0.05" in out
    assert "status=" in out


def test_refused_strata_are_reported_so_a_refusal_is_not_read_as_a_miss():
    """evaluate() counts a refused region as not recovered; the scenario must say so."""

    from eiger2.benchmark import _refused_strata

    manifest = {"fdr": {"insufficient_null_opportunity_strata": [
        {"stratum": "20-49", "null_opportunity": 0.0, "observed_regions": 48},
    ]}}
    assert _refused_strata(manifest) == ["20-49"]
    assert _refused_strata({}) == []
