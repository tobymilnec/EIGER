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
# mode names with one actionable message instead of searching for a paper fixture
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


# --- Planted-direction coherence (Fix S7) ---------------------------------------
#
# ``_plant_dmrs`` used to choose the shift direction independently per CpG. Under a
# supplied marginal whose adjacent baselines straddle 0.5 that makes a planted
# "DMR" locally anti-correlated rather than a coordinated same-direction region, so
# such an arm is not an ordinary coherent-DMR power simulation at all. The default
# is now one scalar direction per planted region; the per-CpG rule is retained by
# name for reproduction and for a deliberate heterogeneous-direction stress test.


def _alternating_marginal(rng, n_cpg):
    """Adjacent baselines alternate below and above 0.5. Draws nothing."""

    baseline = np.where(np.arange(n_cpg) % 2 == 0, 0.2, 0.8)
    return baseline, np.full(n_cpg, 0.03)


def _bimodal_marginal(rng, n_cpg):
    """A two-component bimodal marginal. Consumes exactly one draw."""

    hyper = rng.random(n_cpg) < 0.5
    return np.where(hyper, 0.85, 0.15), np.full(n_cpg, 0.05)


def _split_marginal(rng, n_cpg):
    """Hypomethylated first half, hypermethylated second half. Draws nothing.

    Blocks planted at CpG 200 and CpG 600 with ``spacing=400`` therefore sit in
    opposite headroom regimes, so the two must choose opposite directions.
    """

    baseline = np.where(np.arange(n_cpg) < 500, 0.10, 0.90)
    return baseline, np.full(n_cpg, 0.02)


_COHERENCE_KWARGS = {
    "strata": [(0.25, 0.30), (0.10, 0.45)],
    "replicates": 2,
    "spacing": 400,
    "dmr_length": 4,
    "marginal": "injected_marginal",
}


def _planted_shift(sampler, *, seed, **kwargs):
    """``(shift, truth, baseline)`` for one planted run against its own null run.

    The second run plants nothing but draws the marginal, noise, latent and
    loading arrays in the same order from the same seed, so subtracting it leaves
    exactly the post-clip planted shift. ``locations`` is passed to the second run
    because ``n_cpg`` is otherwise derived from ``strata``.
    """

    planted, locations, truth = simulate(
        seed=seed, marginal_sampler=sampler, **kwargs
    )
    unplanted, _locations_frame, _empty = simulate(
        **{**kwargs, "strata": []}, seed=seed, marginal_sampler=sampler,
        locations=locations,
    )
    baseline, _sd = sampler(np.random.default_rng(seed), len(locations))
    return planted - unplanted, truth, baseline


def _block_signs(shift, start, end):
    """The distinct signs of the non-zero planted shifts inside one block."""

    block = shift[start : end + 1]
    return set(np.unique(np.sign(block[block != 0])).tolist())


def _simulate_capturing_plants(**kwargs):
    """``(matrix, locations, truth, plants)`` for one run.

    ``plants`` is ``_plant_dmrs``'s own record, one
    ``(start, end, carriers, pre_block)`` per planted region: the carrier mask
    comes from the simulator's ``rng.choice`` group draw and the block's values
    from before the shift was added. Both are therefore known independently of the
    clipped matrix a test is checking, so a carrier whose shift the clip erased is
    still a carrier here.
    """

    from eiger2 import benchmark

    captured: list[tuple] = []
    original = benchmark._plant_dmrs

    def recording_plant_dmrs(*args, **kwargs):
        truth, n_not_plantable, plants = original(*args, **kwargs)
        captured.extend(plants)
        return truth, n_not_plantable, plants

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(benchmark, "_plant_dmrs", recording_plant_dmrs)
        matrix, locations, truth = benchmark.simulate(**kwargs)
    finally:
        monkey.undo()
    return matrix, locations, truth, captured


def _rail_pinned_marginal(rng, n_cpg):
    """Every fourth CpG is pinned above the upper clip rail. Draws nothing.

    With ``dmr_length=4`` and blocks starting at a multiple of four, each planted
    block is three hypomethylated CpGs and one sitting at 0.9995 -- above
    ``BETA_CLIP``'s 0.999. The block's aggregate headroom is upward, so the pinned
    CpG's carriers are shifted from a value the clip has already pulled to 0.999
    to a value the clip pulls to 0.999 again: a fully clipped, exactly-zero
    contribution. The near-zero total sd keeps that CpG above the rail in every
    replicate rather than only usually.
    """

    baseline = np.where(np.arange(n_cpg) % 4 == 3, 0.9995, 0.05)
    return baseline, np.full(n_cpg, 1e-5)


# 1./2. the new default is region-consistent under an alternating marginal.


@pytest.mark.parametrize("sampler", [_alternating_marginal, _bimodal_marginal])
def test_region_headroom_gives_one_planted_region_one_direction(sampler):
    shift, truth, _baseline = _planted_shift(sampler, seed=4, **_COHERENCE_KWARGS)
    assert len(truth) == 4
    for start, end, *_rest in truth:
        signs = _block_signs(shift, start, end)
        assert len(signs) == 1, f"block {start}-{end} carries signs {signs}"


@pytest.mark.parametrize("sampler", [_alternating_marginal, _bimodal_marginal])
def test_region_headroom_never_gives_adjacent_cpgs_opposite_effects(sampler):
    shift, truth, _baseline = _planted_shift(sampler, seed=4, **_COHERENCE_KWARGS)
    for start, end, *_rest in truth:
        block = shift[start : end + 1]
        # the per-CpG signed mean over the carriers of this block
        per_cpg = np.sign(block.sum(axis=1))
        moved = per_cpg[per_cpg != 0]
        assert np.all(moved[:-1] * moved[1:] > 0)


def _expected_region_direction(block_mu, delta):
    """The documented rule, written out here rather than called from the module.

    Independent oracle for ``_region_direction``: mean attainable ``+delta`` shift
    against mean attainable ``-delta`` shift under ``BETA_CLIP``, with the
    documented ``HEADROOM_TIE_ATOL`` tie band resolving to ``+delta``. It restates
    the rule from the documentation, so a change to the implementation that also
    changes the rule shows up as a disagreement rather than as two matching
    mistakes.
    """

    from eiger2.benchmark import BETA_CLIP, HEADROOM_TIE_ATOL

    low, high = BETA_CLIP
    block_mu = np.asarray(block_mu, dtype=float)
    at = np.clip(block_mu, low, high)
    up = float((np.clip(block_mu + delta, low, high) - at).mean())
    down = float((at - np.clip(block_mu - delta, low, high)).mean())
    if up > down:
        return 1.0
    if abs(up - down) <= HEADROOM_TIE_ATOL:
        return 1.0
    return -1.0


def test_region_headroom_picks_the_region_wide_headroom_optimum():
    """The chosen sign is the one with the larger mean attainable beta shift."""

    shift, truth, baseline = _planted_shift(
        _bimodal_marginal, seed=42, **_COHERENCE_KWARGS
    )
    for start, end, _fraction, delta, *_run in truth:
        expected = _expected_region_direction(baseline[start : end + 1], delta)
        assert _block_signs(shift, start, end) == {expected}


def test_region_headroom_tie_breaks_to_plus_delta_deterministically():
    """A block symmetric about 0.5 has equal headroom either way; +delta wins."""

    from eiger2.benchmark import _region_direction

    symmetric = np.array([0.2, 0.8, 0.2, 0.8])
    assert _region_direction(symmetric, 0.30) == 1.0
    assert _region_direction(np.full(4, 0.5), 0.30) == 1.0
    assert _expected_region_direction(symmetric, 0.30) == 1.0
    # and it is a decision, not a coin: repeated calls agree, and so does the
    # planted matrix built from them
    assert {_region_direction(symmetric, 0.30) for _ in range(8)} == {1.0}
    shift, truth, _baseline = _planted_shift(
        _alternating_marginal, seed=4, **_COHERENCE_KWARGS
    )
    for start, end, *_rest in truth:
        assert _block_signs(shift, start, end) == {1.0}


def test_a_near_tie_inside_the_tolerance_still_takes_the_plus_delta_break():
    """``down`` strictly beats ``up``, by less than ``HEADROOM_TIE_ATOL``.

    This is the case the tolerance exists for and is not hypothetical: baselines
    0.2/0.8 at ``delta=0.30`` have identical headroom in exact arithmetic, but
    ``0.999 - 0.8`` and ``0.2 - 0.001`` do not round identically, so ``down``
    exceeds ``up`` by ~5.6e-17. Without the tolerance the block would be planted
    downward on a last-bit rounding difference rather than by the documented rule.
    """

    from eiger2.benchmark import BETA_CLIP, HEADROOM_TIE_ATOL, _region_direction

    low, high = BETA_CLIP
    block_mu = np.array([0.2, 0.8])
    delta = 0.30
    at = np.clip(block_mu, low, high)
    up = float((np.clip(block_mu + delta, low, high) - at).mean())
    down = float((at - np.clip(block_mu - delta, low, high)).mean())
    assert down > up                                    # a real, strict difference
    assert 0 < down - up < HEADROOM_TIE_ATOL            # but inside the tie band
    assert _expected_region_direction(block_mu, delta) == 1.0
    assert _region_direction(block_mu, delta) == 1.0


def test_a_difference_outside_the_tolerance_is_not_treated_as_a_tie():
    """The tie band is narrow: a real headroom deficit still flips the sign."""

    from eiger2.benchmark import HEADROOM_TIE_ATOL, _region_direction

    # every baseline is hypermethylated, so downward has strictly more headroom
    block_mu = np.full(4, 0.90)
    delta = 0.30
    assert _expected_region_direction(block_mu, delta) == -1.0
    assert _region_direction(block_mu, delta) == -1.0
    assert HEADROOM_TIE_ATOL < 1e-6                     # a rounding band, not a rule


def _independent_effective_delta(pre_block, carriers, block_mu, delta):
    """What ``effective_delta`` must be, computed from the plant record alone.

    The carrier mask is the simulator's group draw, not a set of entries observed
    to have moved, and the direction comes from the documented rule rather than
    from the planted matrix. Every carrier entry is in the denominator, including
    one the clip flattened to no change at all.
    """

    from eiger2.benchmark import BETA_CLIP

    sign = _expected_region_direction(block_mu, delta)
    before = pre_block[:, carriers]
    after = np.clip(before + sign * delta, *BETA_CLIP)
    return float(np.abs(after - np.clip(before, *BETA_CLIP)).mean())


def test_effective_delta_matches_the_realised_post_clipping_change():
    """truth's effective_delta is the mean |post-clip shift| over the carriers."""

    _matrix, _locations, truth, plants = _simulate_capturing_plants(
        seed=4, marginal_sampler=_bimodal_marginal, **_COHERENCE_KWARGS
    )
    # the sampler is the run's first consumer of the generator, so a fresh one at
    # the same seed and size reproduces exactly the baselines simulate planted on
    baseline, _sd = _bimodal_marginal(np.random.default_rng(4), len(_matrix))
    assert len(plants) == len(truth)
    for entry, plant in zip(truth, plants, strict=True):
        start, end, _fraction, delta, _run_start, _run_end, effective = entry
        plant_start, plant_end, carriers, pre_block = plant
        assert (plant_start, plant_end) == (start, end)
        assert carriers.sum() == max(1, round(_fraction * 80))
        expected = _independent_effective_delta(
            pre_block, carriers, baseline[start : end + 1], delta
        )
        assert effective == pytest.approx(expected, abs=1e-12)
        # coherence is paid for here rather than silently: one direction cannot
        # have full headroom at every CpG of a mixed-baseline block
        assert 0 < effective <= delta


def test_effective_delta_counts_a_fully_clipped_carrier_entry_as_zero():
    """A carrier the clip flattened stays in the denominator, contributing 0.

    It is the difference between "the shift we planted" and "the shift the data
    actually carries": dropping the flattened entries would report the effect the
    unclipped CpGs got and call it the region's effect.
    """

    from eiger2.benchmark import BETA_CLIP

    kwargs = {**_COHERENCE_KWARGS, "strata": [(0.25, 0.30)], "dmr_length": 4}
    matrix, _locations, truth, plants = _simulate_capturing_plants(
        seed=4, marginal_sampler=_rail_pinned_marginal, **kwargs
    )
    baseline, _sd = _rail_pinned_marginal(None, len(matrix))
    assert plants
    saw_a_flattened_entry = False
    for entry, (start, end, carriers, pre_block) in zip(truth, plants, strict=True):
        _s, _e, _fraction, delta, _run_start, _run_end, effective = entry
        block_mu = baseline[start : end + 1]
        sign = _expected_region_direction(block_mu, delta)
        assert sign == 1.0                       # three hypomethylated CpGs win
        before = pre_block[:, carriers]
        realised = np.abs(
            np.clip(before + sign * delta, *BETA_CLIP)
            - np.clip(before, *BETA_CLIP)
        )
        # the pinned CpG is the last row of each block and is fully clipped
        pinned = block_mu > BETA_CLIP[1]
        assert pinned.sum() == 1
        assert np.all(realised[pinned] == 0.0)
        saw_a_flattened_entry = True
        # every carrier entry counts, the flattened ones included
        assert realised.size == 4 * carriers.sum()
        assert effective == pytest.approx(float(realised.mean()), abs=1e-12)
        # and that is strictly less than the average over only the entries that
        # did move, which is what dropping them would have reported
        moved = realised[realised > 0]
        assert moved.size < realised.size
        assert effective < float(moved.mean())
        assert effective == pytest.approx(delta * 3 / 4, rel=1e-3)
    assert saw_a_flattened_entry


# 3. the retained per-CpG mode reproduces the historical behaviour exactly.


def test_per_cpg_headroom_keeps_the_historical_alternating_signs():
    shift, truth, baseline = _planted_shift(
        _alternating_marginal, seed=4,
        **{**_COHERENCE_KWARGS, "planting_direction": "per_cpg_headroom"},
    )
    for start, end, *_rest in truth:
        assert _block_signs(shift, start, end) == {1.0, -1.0}
        per_cpg = np.sign(shift[start : end + 1].sum(axis=1))
        expected = np.where(baseline[start : end + 1] > 0.5, -1.0, 1.0)
        assert np.array_equal(per_cpg, expected)
        # adjacent CpGs of one "DMR" move in opposite directions: the defect this
        # mode is retained to reproduce, not a mode to run a power curve in
        assert np.all(per_cpg[:-1] * per_cpg[1:] < 0)


#: Supplied-marginal matrices produced by the pre-fix per-CpG rule. They pin the
#: retained mode to the draws and signs it had before ``planting_direction``
#: existed, so a legacy reproduction stays byte-exact.
LEGACY_SUPPLIED_MARGINAL_HASHES: tuple[tuple[str, int, str], ...] = (
    ("alternating", 4, "fe8d99caa364a532"),
    ("alternating", 42, "5b63d1c40aaddada"),
    ("bimodal", 4, "6c9fc42271519d4f"),
    ("bimodal", 42, "a443ab12aad6e28e"),
)


@pytest.mark.parametrize("name,seed,digest", LEGACY_SUPPLIED_MARGINAL_HASHES)
def test_per_cpg_headroom_reproduces_the_pre_fix_supplied_marginal_matrix(
    name, seed, digest
):
    import hashlib

    sampler = {
        "alternating": _alternating_marginal, "bimodal": _bimodal_marginal,
    }[name]
    matrix, _locations_frame, _truth = simulate(
        seed=seed, marginal_sampler=sampler, planting_direction="per_cpg_headroom",
        **_COHERENCE_KWARGS,
    )
    assert hashlib.sha256(matrix.tobytes()).hexdigest()[:16] == digest


#: The same configurations under the new default. A coherent-mode digest a paper
#: consumer can pin its own rerun against.
COHERENT_SUPPLIED_MARGINAL_HASHES: tuple[tuple[str, int, str], ...] = (
    ("alternating", 4, "ef3157b97174115d"),
    ("alternating", 42, "f800dd4b9f42147b"),
    ("bimodal", 4, "7f532257874fe345"),
    ("bimodal", 42, "1290c340765c4a73"),
)


@pytest.mark.parametrize("name,seed,digest", COHERENT_SUPPLIED_MARGINAL_HASHES)
def test_region_headroom_supplied_marginal_matrix_is_pinned(name, seed, digest):
    import hashlib

    sampler = {
        "alternating": _alternating_marginal, "bimodal": _bimodal_marginal,
    }[name]
    matrix, _locations_frame, _truth = simulate(
        seed=seed, marginal_sampler=sampler, **_COHERENCE_KWARGS
    )
    assert hashlib.sha256(matrix.tobytes()).hexdigest()[:16] == digest


# 4. the shipped dense/unimodal path is untouched by either mode.


@pytest.mark.parametrize("kwargs,digest", PRE_WP6_DENSE_MATRIX_HASHES)
@pytest.mark.parametrize(
    "planting_direction", ["region_headroom", "per_cpg_headroom"]
)
def test_dense_unimodal_hashes_are_unchanged_in_both_directions(
    kwargs, digest, planting_direction
):
    """Every unimodal baseline is 0.5, so both rules give +1 at every CpG."""

    import hashlib

    matrix, _locations_frame, _truth = simulate(
        **kwargs, planting_direction=planting_direction
    )
    assert hashlib.sha256(matrix.tobytes()).hexdigest()[:16] == digest


# 5./6. the modes differ only in the signs, never in what was drawn.


def test_the_two_modes_agree_on_carriers_truth_and_placement():
    locs = _locations((np.arange(1800) * 2).tolist())
    baseline, _sd = _alternating_marginal(None, len(locs))
    planted = {}
    for mode in ("region_headroom", "per_cpg_headroom"):
        matrix = np.zeros((len(locs), 40))
        truth, n_not_plantable, plants = _plant_dmrs(
            matrix, locs, strata=[(0.25, 0.3), (0.10, 0.45)], replicates=2,
            spacing=400, dmr_length=6, max_cpg_gap_bp=300,
            rng=np.random.default_rng(9), n_samples=40, baseline_mu=baseline,
            planting_direction=mode, start_position=200,
        )
        planted[mode] = (truth, n_not_plantable, [p[2].copy() for p in plants])
    region, per_cpg = planted["region_headroom"], planted["per_cpg_headroom"]
    assert region[0] == per_cpg[0]          # truth coordinates and placement
    assert region[1] == per_cpg[1]          # not-plantable accounting
    assert len(region[2]) == 4
    for a, b in zip(region[2], per_cpg[2], strict=True):
        assert np.array_equal(a, b)         # carrier identities


def test_changing_the_mode_consumes_no_extra_random_numbers():
    """The dropout mask is drawn after planting, so it moves if a draw is added."""

    kwargs = {
        **_COHERENCE_KWARGS, "seed": 4, "missing_rate": 0.3,
        "marginal_sampler": _bimodal_marginal,
    }
    region = simulate(**kwargs, planting_direction="region_headroom")
    per_cpg = simulate(**kwargs, planting_direction="per_cpg_headroom")
    assert np.array_equal(np.isnan(region[0]), np.isnan(per_cpg[0]))
    assert [t[:6] for t in region[2]] == [t[:6] for t in per_cpg[2]]


# 7. several planted regions each get their own single direction.


def test_each_planted_region_gets_its_own_one_scalar_direction():
    shift, truth, _baseline = _planted_shift(
        _split_marginal, seed=4, **{**_COHERENCE_KWARGS, "dmr_length": 8}
    )
    directions = []
    for start, end, *_rest in truth:
        signs = _block_signs(shift, start, end)
        assert len(signs) == 1
        directions.append(signs.pop())
    # hypomethylated first half plants upward, hypermethylated second half down
    assert directions[0] == 1.0
    assert -1.0 in directions
    assert set(directions) == {1.0, -1.0}


# 8./9. the argument itself.


@pytest.mark.parametrize("mode", ["", "region", "per_cpg", "coherent", None])
def test_simulate_rejects_an_unknown_planting_direction(mode):
    with pytest.raises(ValueError) as excinfo:
        simulate(strata=[], seed=0, planting_direction=mode)
    message = str(excinfo.value)
    assert "planting_direction must be one of" in message
    assert "region_headroom" in message and "per_cpg_headroom" in message
    assert len(message.splitlines()) == 1


def test_plant_dmrs_rejects_an_unknown_planting_direction():
    locs = _locations((np.arange(30) * 2).tolist())
    with pytest.raises(ValueError, match="planting_direction must be one of"):
        _plant_dmrs(
            np.zeros((len(locs), 10)), locs, strata=[(0.3, 0.3)], replicates=1,
            spacing=400, dmr_length=5, max_cpg_gap_bp=300,
            rng=np.random.default_rng(2), n_samples=10,
            baseline_mu=np.full(len(locs), 0.5), planting_direction="bogus",
        )


def test_an_invalid_planting_direction_is_rejected_before_anything_is_drawn():
    """Validation is cheap and up front: no partial simulation is built first."""

    import hashlib

    with pytest.raises(ValueError):
        simulate(strata=[(0.1, 0.3)], replicates=1, seed=42,
                 planting_direction="bogus")
    # the valid run at the same seed is unaffected by the rejected one
    matrix, _locations_frame, _truth = simulate(
        strata=[(0.1, 0.3), (0.2, 0.5)], replicates=3, seed=42
    )
    assert hashlib.sha256(matrix.tobytes()).hexdigest()[:16] == "ec7d44da96eb02d1"


def test_simulate_defaults_to_region_consistent_planting_and_records_it():
    import inspect

    from eiger2 import benchmark

    parameter = inspect.signature(simulate).parameters["planting_direction"]
    assert parameter.default == "region_headroom"
    assert parameter.annotation == "str"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert benchmark.PLANTING_DIRECTIONS == (
        "region_headroom", "per_cpg_headroom"
    )
    _matrix, locations, _truth = simulate(strata=[(0.1, 0.3)], replicates=1, seed=2)
    assert locations.attrs["planting_direction"] == "region_headroom"
    _matrix, locations, _truth = simulate(
        strata=[(0.1, 0.3)], replicates=1, seed=2,
        planting_direction="per_cpg_headroom",
    )
    assert locations.attrs["planting_direction"] == "per_cpg_headroom"
    # portable metadata only: attrs must never carry a filesystem path
    assert all(
        not str(value).startswith("/") for value in locations.attrs.values()
    )


def test_simulate_docstring_documents_both_planting_directions():
    for mode in ("region_headroom", "per_cpg_headroom"):
        assert mode in simulate.__doc__
    assert "effective_delta" in simulate.__doc__
    assert _plant_dmrs.__doc__ is not None
    for mode in ("region_headroom", "per_cpg_headroom"):
        assert mode in _plant_dmrs.__doc__


def test_planting_direction_does_not_change_the_truth_tuple_schema():
    for mode in ("region_headroom", "per_cpg_headroom"):
        _matrix, _locations_frame, truth = simulate(
            strata=[(0.25, 0.30)], replicates=2, seed=4, planting_direction=mode,
        )
        for entry in truth:
            assert len(entry) == 7
            start, end, fraction, delta, run_start, run_end, effective = entry
            assert isinstance(start, int) and isinstance(end, int)
            assert (fraction, delta) == (0.25, 0.30)
            assert run_start <= start and end <= run_end
            assert isinstance(effective, float)
