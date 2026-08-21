"""Tests for the region-level FDR scaling and the two-groups posterior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from eiger2.benchmark import simulate  # noqa: E402
from eiger2.core import (  # noqa: E402
    Eiger2Config,
    call_dmrs,
    region_fdr,
    region_fdr_result,
)
from eiger2.probability import (  # noqa: E402
    _tail_area_fdr,
    calibration_error,
    estimate_pi0,
    local_fdr,
    reliability_table,
)

# --------------------------------------------------------------------------
# the scaling regression
# --------------------------------------------------------------------------


@pytest.mark.parametrize("replicates", [1, 5, 20, 50])
def test_region_fdr_scale_is_not_double_discounted(replicates):
    """``null_edges`` is the total over replicates, so the expectation must use
    only the edge ratio.

    ``null_area_sets`` are pooled over replicates and ``null_edges`` is summed
    over them, so the two already correspond. An earlier implementation divided
    by ``replicates`` a second time, which made every q-value exactly
    ``replicates``-fold too small — 20-fold at the default configuration.
    """

    observed_edges = 10_000
    per_replicate = 4
    null_sets = [np.full(per_replicate, 2.0) for _ in range(replicates)]
    # every observed region is less extreme than every null region, so the
    # expected null count is the full per-replicate rate at every rank
    observed = np.linspace(1.9, 1.0, 100)

    # region_fdr()'s signature is frozen, so it always applies the spec §5
    # default min_null_regions_for_fdr=50; at replicates in {1, 5} the pooled
    # null (replicates * per_replicate = 4 or 20) is deliberately far below
    # that. Called through region_fdr_result directly with the guard disabled
    # to isolate the scale arithmetic this test targets from null adequacy.
    q = region_fdr_result(
        observed,
        null_sets,
        observed_edges=observed_edges,
        null_edges=replicates * observed_edges,
        min_null_regions_for_fdr=0,
    ).qvalue
    # expected null regions in a genome of observed_edges edges = per_replicate
    assert q[-1] == pytest.approx(per_replicate / observed.size, rel=1e-9)


def test_region_fdr_is_monotone_in_area():
    rng = np.random.default_rng(0)
    observed = rng.gamma(3.0, 1.0, 200)
    null_sets = [rng.gamma(2.0, 1.0, 40) for _ in range(10)]
    q = region_fdr(observed, null_sets, observed_edges=5000, null_edges=50_000)
    # ``order`` lists regions strongest-area-first; q-values must be
    # non-decreasing along that ordering, i.e. non-decreasing as area weakens.
    order = np.argsort(-observed)
    assert np.all(np.diff(q[order]) >= -1e-12)


# --------------------------------------------------------------------------
# tail-area fallback (extrapolated tail of local_fdr)
# --------------------------------------------------------------------------


def test_tail_area_fdr_four_distinct_areas():
    observed = np.array([10.0, 8.0, 6.0, 4.0])
    pooled = np.array([9.0, 7.0, 5.0, 3.0])
    result = _tail_area_fdr(observed, pooled, scale=1.0)
    assert result == pytest.approx([0.0, 0.5, 2 / 3, 0.75])


def test_tail_area_fdr_tie_group():
    observed = np.array([10.0, 8.0, 8.0, 4.0])
    pooled = np.array([9.0, 7.5, 5.0, 3.0])
    result = _tail_area_fdr(observed, pooled, scale=1.0)
    assert result == pytest.approx([0.0, 1 / 3, 1 / 3, 0.75])


# --------------------------------------------------------------------------
# pi0
# --------------------------------------------------------------------------


def test_estimate_pi0_recovers_a_known_null_fraction():
    # 200 pooled null regions over 20x the observed genome -> 10 expected null
    # regions among 100 called -> pi0 = 0.1
    assert estimate_pi0(
        100, 200, observed_edges=1_000, null_edges=20_000
    ) == pytest.approx(0.1)


def test_estimate_pi0_is_clipped_to_one():
    assert estimate_pi0(10, 10_000, observed_edges=1_000, null_edges=1_000) == 1.0


# --------------------------------------------------------------------------
# local fdr behaviour
# --------------------------------------------------------------------------


def test_local_fdr_refuses_a_thin_null():
    result = local_fdr(
        np.array([1.0, 2.0, 3.0]),
        [np.array([0.5]), np.array([0.6])],
        observed_edges=1000,
        null_edges=2000,
        min_null_regions=50,
    )
    assert result.status.startswith("insufficient_null_regions")
    assert np.all(np.isnan(result.posterior))


def test_local_fdr_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    observed = np.concatenate([rng.gamma(2.0, 1.0, 150), rng.gamma(8.0, 1.0, 150)])
    null_sets = [rng.gamma(2.0, 1.0, 60) for _ in range(20)]
    result = local_fdr(
        observed, null_sets, observed_edges=10_000, null_edges=200_000
    )
    assert result.status == "ok"
    assert np.all((result.fdr >= 0) & (result.fdr <= 1))
    order = np.argsort(observed)
    # larger area is never weaker evidence
    assert np.all(np.diff(result.fdr[order]) <= 1e-9)
    assert result.posterior == pytest.approx(1.0 - result.fdr)


def test_local_fdr_separates_a_clear_mixture():
    """With a strongly separated non-null component the posterior should be
    high for the large-area regions and low for the null-like ones."""

    rng = np.random.default_rng(2)
    null_like = rng.gamma(2.0, 1.0, 200)
    signal = rng.gamma(30.0, 1.0, 200)
    observed = np.concatenate([null_like, signal])
    null_sets = [rng.gamma(2.0, 1.0, 100) for _ in range(20)]
    result = local_fdr(
        observed, null_sets, observed_edges=10_000, null_edges=200_000
    )
    assert result.status == "ok"
    assert result.posterior[200:].mean() > 0.9
    assert result.posterior[:200].mean() < result.posterior[200:].mean()


def test_conservative_variant_is_never_smaller():
    rng = np.random.default_rng(3)
    observed = rng.gamma(4.0, 1.0, 200)
    null_sets = [rng.gamma(2.0, 1.0, 60) for _ in range(20)]
    result = local_fdr(
        observed, null_sets, observed_edges=10_000, null_edges=200_000
    )
    assert np.all(result.fdr_conservative >= result.fdr - 1e-12)


def test_extrapolated_regions_are_flagged():
    rng = np.random.default_rng(4)
    null_sets = [rng.gamma(2.0, 1.0, 60) for _ in range(20)]
    pooled_max = max(a.max() for a in null_sets)
    observed = np.concatenate([rng.gamma(2.0, 1.0, 100), [pooled_max * 10]])
    result = local_fdr(
        observed, null_sets, observed_edges=10_000, null_edges=200_000
    )
    assert result.extrapolated[-1]
    assert not result.extrapolated[:100].all()


# --------------------------------------------------------------------------
# calibration helpers
# --------------------------------------------------------------------------


def test_reliability_table_of_a_perfect_predictor():
    posterior = np.concatenate([np.full(100, 0.05), np.full(100, 0.95)])
    truth = np.concatenate([np.zeros(100, bool), np.ones(100, bool)])
    table = reliability_table(posterior, truth, n_bins=10)
    error = calibration_error(table)
    assert error["max_calibration_error"] < 0.06


def test_reliability_table_detects_overconfidence():
    posterior = np.full(200, 0.95)
    truth = np.zeros(200, dtype=bool)  # nothing is real
    table = reliability_table(posterior, truth, n_bins=10)
    error = calibration_error(table)
    assert error["expected_calibration_error"] > 0.9


# --------------------------------------------------------------------------
# end-to-end wiring
# --------------------------------------------------------------------------


def test_call_dmrs_emits_posterior_columns():
    matrix, locations, _ = simulate(
        strata=[(0.25, 0.30)], replicates=2, seed=11, n_samples=40, spacing=300
    )
    regions, _, manifest = call_dmrs(
        matrix, locations,
        config=Eiger2Config(compute_posterior=True, region_null_mode="lag"),
    )
    assert "region_evidence_score" in regions.columns
    assert "evidence_score_extrapolated" in regions.columns
    assert "posterior" in manifest


def test_posterior_is_off_by_default():
    matrix, locations, _ = simulate(
        strata=[(0.25, 0.30)], replicates=2, seed=12, n_samples=40, spacing=300
    )
    regions, _, manifest = call_dmrs(
        matrix, locations, config=Eiger2Config(region_null_mode="lag")
    )
    assert "posterior" not in manifest
    if len(regions):
        assert regions["region_evidence_score"].isna().all()


# --------------------------------------------------------------------------
# permutation null, prevalence floor, and the legacy q-value
# --------------------------------------------------------------------------


def test_permutation_null_tracks_raises_naming_the_replacement():
    # Stage 1d-D Task 6.1 (master spec §6.4): permutation_null_tracks is
    # edge-wise permutation (block length 1) -- measured at zero null regions
    # across 400 replicates and, before the Stage 1c-bis null-adequacy guard
    # existed to catch it, 3/3 significant on DMR-free clean data
    # (STAGE1D_C_COMPRESSED_SPEC.md §0; docs/MULTISCALE_BLOCKED.md). The
    # "richer null than the lag null" narrative this test used to pin
    # (replaced below) was itself part of the problem -- more candidate
    # regions from a null this broken is anti-conservative, not an
    # improvement. It must not be reachable, silently broken, any more.
    from eiger2.core import permutation_null_tracks

    rng = np.random.default_rng(20)
    statistic = np.concatenate([
        rng.normal(0, 0.05, 400), np.full(40, 0.9), rng.normal(0, 0.05, 400)
    ])
    eligible = np.ones(statistic.size, dtype=bool)
    chrom = np.array(["chr1"] * statistic.size)
    with pytest.raises(ValueError, match="region_null_mode='member_step'"):
        permutation_null_tracks(statistic, eligible, chrom, replicates=5, seed=1)


def test_null_mode_permutation_rejected_at_config_construction():
    # Fails as early as possible, not deep inside call_dmrs after potentially
    # expensive computation.
    with pytest.raises(ValueError, match="region_null_mode='member_step'"):
        Eiger2Config(null_mode="permutation")


def test_qvalue_result_from_track_legacy_path_raises_not_silently_broken():
    # Stage 1d-D Task 6.3: qvalue_result_from_track (the --legacy
    # --fdr-replicates CLI path, cli.py:286) calls permutation_null_tracks
    # directly and unconditionally. This is the "explicit test" flagged as
    # missing across three consecutive specs -- confirms the Task 6.1 fix
    # actually reaches this path rather than leaving the legacy annotation
    # silently using the broken null.
    from eiger2.core import qvalue_result_from_track, qvalues_from_track

    rng = np.random.default_rng(20)
    statistic = rng.normal(0, 0.05, 800)
    statistic[300:340] = 0.8
    regions = np.array([[300, 339]])
    with pytest.raises(ValueError, match="region_null_mode='member_step'"):
        qvalue_result_from_track(
            statistic, regions, threshold=0.25,
            chromosome=np.array(["chr1"] * statistic.size),
        )
    with pytest.raises(ValueError, match="region_null_mode='member_step'"):
        qvalues_from_track(
            statistic, regions, threshold=0.25,
            chromosome=np.array(["chr1"] * statistic.size),
        )


def test_prevalence_floor_demotes_single_sample_regions():
    rng = np.random.default_rng(21)
    matrix, locations, _ = simulate(
        strata=[(0.25, 0.35)], replicates=3, seed=3, n_samples=60, spacing=300
    )
    for start in (1200, 1500):
        matrix[start : start + 30, rng.integers(0, 60)] += 0.45
    matrix = np.clip(matrix, 0.01, 0.99)

    loose, _, _ = call_dmrs(
        matrix, locations,
        config=Eiger2Config(min_prevalence_samples=0, region_null_mode="lag"),
    )
    strict, _, _ = call_dmrs(
        matrix, locations,
        config=Eiger2Config(min_prevalence_samples=3, region_null_mode="lag"),
    )
    # the regions themselves are unchanged; only their significance differs
    assert len(loose) == len(strict)
    single = strict["minority_size_descriptive"] <= 1
    if single.any():
        assert not strict.loc[single, "significant"].any()
    assert strict["significant"].sum() <= loose["significant"].sum()


# test_qvalues_from_track_annotates_an_external_caller retired (Stage 1d-D,
# master spec §6.4): it pinned qvalue_result_from_track's behaviour when its
# internal permutation_null_tracks call returned zero null regions. That call
# now raises unconditionally (Task 6.1) -- see
# test_qvalue_result_from_track_legacy_path_raises_not_silently_broken in
# test_probability.py's permutation-null section, which is what this test's
# own comment said should replace it once §6.4 landed a decision (raise, not
# repair).
