"""Tests for the region-level FDR scaling and the two-groups posterior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from eiger2.benchmark import simulate  # noqa: E402
from eiger2.core import Eiger2Config, call_dmrs, region_fdr  # noqa: E402
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

    q = region_fdr(
        observed,
        null_sets,
        observed_edges=observed_edges,
        null_edges=replicates * observed_edges,
    )
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
        matrix, locations, config=Eiger2Config(compute_posterior=True)
    )
    assert "region_evidence_score" in regions.columns
    assert "evidence_score_extrapolated" in regions.columns
    assert "posterior" in manifest


def test_posterior_is_off_by_default():
    matrix, locations, _ = simulate(
        strata=[(0.25, 0.30)], replicates=2, seed=12, n_samples=40, spacing=300
    )
    regions, _, manifest = call_dmrs(matrix, locations, config=Eiger2Config())
    assert "posterior" not in manifest
    if len(regions):
        assert regions["region_evidence_score"].isna().all()


# --------------------------------------------------------------------------
# permutation null, prevalence floor, and the legacy q-value
# --------------------------------------------------------------------------


def test_permutation_null_preserves_the_marginal_and_breaks_the_runs():
    from eiger2.core import permutation_null_tracks

    rng = np.random.default_rng(20)
    statistic = np.concatenate([
        rng.normal(0, 0.05, 400), np.full(40, 0.9), rng.normal(0, 0.05, 400)
    ])
    eligible = np.ones(statistic.size, dtype=bool)
    chrom = np.array(["chr1"] * statistic.size)
    tracks = permutation_null_tracks(
        statistic, eligible, chrom, replicates=5, seed=1
    )
    for track, usable in tracks:
        # same multiset of values: the amount of coordination is unchanged
        assert np.allclose(np.sort(track[usable]), np.sort(statistic[eligible]))
        # but the long run of high values no longer survives intact
        high = track[usable] > 0.5
        longest = 0
        run = 0
        for value in high:
            run = run + 1 if value else 0
            longest = max(longest, run)
        assert longest < 40


def test_permutation_null_yields_a_richer_null_than_the_lag_null():
    """The point of the change: the lag null can produce no regions at all,
    which forces every q-value to zero regardless of the data."""

    from eiger2.core import (
        centre_cpgs,
        deflate,
        edge_statistic,
        eligible_edges,
        estimate_latent_basis,
        null_threshold,
        null_tracks,
        permutation_null_tracks,
        segment,
    )

    matrix, locations, _ = simulate(
        strata=[(0.25, 0.35)], replicates=6, seed=5, n_samples=120, spacing=250
    )
    config = Eiger2Config(null_replicates=40)
    residuals = centre_cpgs(matrix)
    basis, _ = estimate_latent_basis(residuals, config)
    deflated = deflate(residuals, basis, missing=config.deflation_missing)
    statistic = edge_statistic(deflated, config)
    eligible = eligible_edges(locations, config)
    threshold, _ = null_threshold(deflated, config, locations)

    def count(tracks):
        total = 0
        for track, usable in tracks:
            found = segment(
                track, usable & np.isfinite(track), threshold,
                min_cpgs=config.min_cpgs, max_gap_edges=config.max_gap_edges,
            )
            total += len(found)
        return total

    lag = count(null_tracks(deflated, config, locations))
    perm = count(permutation_null_tracks(
        statistic, eligible, locations["chr"].to_numpy(),
        replicates=config.null_replicates, seed=config.random_seed,
    ))
    assert perm > lag


def test_prevalence_floor_demotes_single_sample_regions():
    rng = np.random.default_rng(21)
    matrix, locations, _ = simulate(
        strata=[(0.25, 0.35)], replicates=3, seed=3, n_samples=60, spacing=300
    )
    for start in (1200, 1500):
        matrix[start : start + 30, rng.integers(0, 60)] += 0.45
    matrix = np.clip(matrix, 0.01, 0.99)

    loose, _, _ = call_dmrs(
        matrix, locations, config=Eiger2Config(min_prevalence_samples=0)
    )
    strict, _, _ = call_dmrs(
        matrix, locations, config=Eiger2Config(min_prevalence_samples=3)
    )
    # the regions themselves are unchanged; only their significance differs
    assert len(loose) == len(strict)
    single = strict["minority_size_descriptive"] <= 1
    if single.any():
        assert not strict.loc[single, "significant"].any()
    assert strict["significant"].sum() <= loose["significant"].sum()


def test_qvalues_from_track_annotates_an_external_caller():
    """The legacy caller defines no error rate; this supplies one without
    changing which regions it reports."""

    from eiger2.core import qvalues_from_track

    rng = np.random.default_rng(22)
    statistic = rng.normal(0, 0.05, 3000)
    statistic[500:540] = 0.8          # one strong planted run
    statistic[1500:1506] = 0.35       # one marginal run
    regions = np.array([[500, 539], [1500, 1505]])
    q, diagnostics = qvalues_from_track(
        statistic, regions, threshold=0.25,
        chromosome=np.array(["chr1"] * statistic.size),
        min_cpgs=5, replicates=40, seed=1,
    )
    assert q.shape == (2,)
    assert np.all((q >= 0) & (q <= 1))
    assert q[0] <= q[1]               # the stronger region is never worse
    assert diagnostics["n_regions"] == 2
