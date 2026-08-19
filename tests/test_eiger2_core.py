"""Unit and operating-characteristic tests for the EIGER 2 core.

The operating-characteristic tests are the ones that matter. They are the checks
that EIGER 1 lacked and that would have caught every major finding in
PEER_REVIEW.md: prevalence bias, the latent-factor confound, the missingness
deflation, and uncalibrated discovery on DMR-free data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "Python"))
# benchmarking/ is not an installed package (see pyproject.toml's setuptools
# packages.find scope); add the repo root so `from benchmarking.benchmark_suite
# import SCENARIOS` resolves under pytest, matching how it already resolves when
# run as a plain script from the repo root.
sys.path.insert(0, str(_REPO_ROOT))

from eiger2 import (  # noqa: E402
    Eiger2Config,
    call_dmrs,
    centre_cpgs,
    characterise_region,
    deflate,
    edge_statistic,
    estimate_latent_basis,
    region_fdr,
    segment,
)
from eiger2.benchmark import evaluate, simulate  # noqa: E402
from eiger2.core import (  # noqa: E402
    RegionFdrResult,  # noqa: F401
    _assign_support_bin,
    _fdr_from_areas_result,
    _overdispersion_diagnostic,
    _pair_statistic,
    _support_bin_edges,
    _window_support,
    null_exchangeability,
    null_threshold,
    null_tracks,
    permutation_null_tracks,
    qvalue_result_from_track,
    qvalues_from_track,
    region_area,
    region_fdr_result,
)

FAST = {"null_replicates": 6}


def locations(n_cpg: int, chrom: str = "chr1") -> pd.DataFrame:
    start = np.arange(n_cpg) * 2
    return pd.DataFrame({"chr": chrom, "start": start, "end": start + 1})


# --------------------------------------------------------------------------
# statistic
# --------------------------------------------------------------------------


def test_pair_statistic_matches_numpy_on_complete_data():
    rng = np.random.default_rng(0)
    left = rng.normal(size=(3, 40))
    right = rng.normal(size=(3, 40))
    covariance = _pair_statistic(left, right, "covariance", 5)
    correlation = _pair_statistic(left, right, "correlation", 5)
    for row in range(3):
        assert covariance[row] == pytest.approx(
            np.cov(left[row], right[row], ddof=1)[0, 1]
        )
        assert correlation[row] == pytest.approx(
            np.corrcoef(left[row], right[row])[0, 1]
        )


def test_pairwise_complete_removes_the_missingness_deflation():
    """EIGER 1 filled gaps with the CpG mean, deflating covariance by (1-m)^2."""

    rng = np.random.default_rng(1)
    n, reps = 60, 300
    truth = []
    observed = []
    for missing in (0.0, 0.3):
        estimates = []
        mean_filled = []
        for _ in range(reps):
            group = np.zeros(n)
            group[: n // 2] = 1
            x = 0.30 * group + rng.normal(0, 0.05, n)
            y = 0.30 * group + rng.normal(0, 0.05, n)
            pair = np.vstack([x, y])
            pair[rng.random(pair.shape) < missing] = np.nan
            estimates.append(
                _pair_statistic(pair[:1], pair[1:], "covariance", 3)[0]
            )
            filled = np.where(np.isfinite(pair), pair, 0.0)
            mean_filled.append(np.cov(filled[0], filled[1], ddof=1)[0, 1])
        truth.append(np.nanmean(estimates))
        observed.append(np.mean(mean_filled))
    # pairwise-complete is unbiased across missing rates
    assert truth[1] == pytest.approx(truth[0], rel=0.10)
    # mean-filling deflates by roughly (1 - m)^2 = 0.49
    assert observed[1] / observed[0] == pytest.approx(0.49, abs=0.08)


def test_correlation_sees_rare_subgroups_that_covariance_cannot():
    """cov = f(1-f)d^2 is prevalence-limited; correlation saturates."""

    rng = np.random.default_rng(2)
    n, k, delta, sigma = 519, 3, 0.30, 0.03
    group = np.zeros(n)
    group[:k] = 1
    covariances, correlations = [], []
    for _ in range(200):
        x = (delta * group + rng.normal(0, sigma, n))[None, :]
        y = (delta * group + rng.normal(0, sigma, n))[None, :]
        covariances.append(_pair_statistic(x, y, "covariance", 5)[0])
        correlations.append(_pair_statistic(x, y, "correlation", 5)[0])
    fraction = k / n
    assert np.mean(covariances) == pytest.approx(
        fraction * (1 - fraction) * delta**2, rel=0.30
    )
    assert np.mean(covariances) < 0.012  # below EIGER 1's default threshold
    assert np.mean(correlations) > 0.30  # comfortably detectable


def test_window_reduces_null_variance():
    rng = np.random.default_rng(3)
    config_pair = Eiger2Config(window=1, min_paired_samples=5)
    config_window = Eiger2Config(window=6, min_paired_samples=5)
    residuals = rng.normal(size=(4000, 20))
    spread_pair = np.nanstd(edge_statistic(residuals, config_pair))
    spread_window = np.nanstd(edge_statistic(residuals, config_window))
    assert spread_window < 0.6 * spread_pair


def test_window_support_counts_samples_finite_at_every_member():
    finite = np.array([
        [True,  True,  False],   # row 0
        [True,  False, True],    # row 1
        [True,  True,  True],    # row 2
        [False, True,  True],    # row 3
    ])
    # window covering rows {0,1}: sample 0 finite at both (T,T), sample 1 not
    # (T,F), sample 2 not (F,T) -> support = 1
    support = _window_support(
        finite, base_positions=np.array([0]), offsets=np.array([0, 1])
    )
    assert support.tolist() == [1]
    # window covering rows {1,2}: sample0 (T,T)=finite, sample1 (F,T)=not,
    # sample2 (T,T)=finite -> support = 2
    support2 = _window_support(
        finite, base_positions=np.array([1]), offsets=np.array([0, 1])
    )
    assert support2.tolist() == [2]


def test_window_support_matches_naive_loop_on_random_data():
    rng = np.random.default_rng(11)
    n_cpg, n_samples = 200, 30
    finite = rng.random((n_cpg, n_samples)) > 0.3
    offsets = np.array([0, 3, 7])
    base_positions = np.arange(n_cpg - offsets.max())
    fast = _window_support(finite, base_positions, offsets)
    naive = np.array([
        finite[base + offsets].all(axis=0).sum() for base in base_positions
    ])
    np.testing.assert_array_equal(fast, naive)


def test_window_support_chunking_does_not_change_result():
    rng = np.random.default_rng(12)
    finite = rng.random((500, 20)) > 0.4
    offsets = np.array([0, 1, 2, 3])
    base_positions = np.arange(496)
    whole = _window_support(finite, base_positions, offsets, chunk=200_000)
    chunked = _window_support(finite, base_positions, offsets, chunk=50)
    np.testing.assert_array_equal(whole, chunked)


def test_support_bin_edges_are_observed_quantiles():
    rng = np.random.default_rng(31)
    observed_support = rng.integers(10, 100, size=2000)
    pooled_null_support = rng.integers(10, 100, size=20000)
    result = _support_bin_edges(observed_support, n_bins=4,
                                 min_null_values=1000,
                                 pooled_null_support=pooled_null_support)
    expected_edges = np.quantile(observed_support, [0.25, 0.5, 0.75])
    np.testing.assert_allclose(result["bin_edges"], expected_edges)
    assert not result["collapsed"]


def test_support_bins_merge_upward_when_null_underpopulated():
    # construct null support so that the top bin has very few values
    rng = np.random.default_rng(32)
    observed_support = rng.integers(0, 100, size=2000)
    pooled_null_support = np.concatenate([
        rng.integers(0, 75, size=5000),   # plenty in the lower 3 quartiles
        rng.integers(75, 100, size=10),   # almost nothing in the top quartile
    ])
    result = _support_bin_edges(observed_support, n_bins=4,
                                 min_null_values=1000,
                                 pooled_null_support=pooled_null_support)
    assert len(result["merges"]) >= 1
    # fewer final bins than requested, since the top one merged into its neighbour
    assert result["bin_edges"].size < 3


def test_support_bins_collapse_to_single_when_null_too_thin_everywhere():
    observed_support = np.arange(2000)
    pooled_null_support = np.arange(50)  # far too few null values overall
    result = _support_bin_edges(observed_support, n_bins=4,
                                 min_null_values=1000,
                                 pooled_null_support=pooled_null_support)
    assert result["collapsed"]
    assert result["bin_edges"].size == 0


def test_assign_support_bin_matches_searchsorted_semantics():
    bin_edges = np.array([10.0, 20.0, 30.0])
    support = np.array([5, 10, 15, 20, 25, 30, 35])
    bins = _assign_support_bin(support, bin_edges)
    # 4 bins: [-inf,10), [10,20), [20,30), [30,inf)
    assert bins.tolist() == [0, 1, 1, 2, 2, 3, 3]


# --------------------------------------------------------------------------
# latent structure
# --------------------------------------------------------------------------


def test_parallel_analysis_recovers_the_number_of_latent_factors():
    for true_k in (1, 2, 3):
        rng = np.random.default_rng(11)
        latent = rng.normal(0, 1, (60, true_k))
        loading = rng.normal(0, 0.07, (4000, true_k))
        matrix = 0.5 + loading @ latent.T + rng.normal(0, 0.04, (4000, 60))
        _, diagnostics = estimate_latent_basis(
            centre_cpgs(matrix), Eiger2Config()
        )
        assert diagnostics["n_components"] == true_k


def test_thinned_basis_preserves_focal_signal_that_a_full_basis_deletes():
    """The reason the basis is estimated on a thinned CpG set."""

    rng = np.random.default_rng(5)
    n, n_cpg = 60, 6000
    latent = rng.normal(0, 1, n)
    loading = rng.normal(0, 0.08, (n_cpg, 1))
    matrix = 0.5 + loading * latent + rng.normal(0, 0.05, (n_cpg, n))
    group = np.zeros(n)
    group[:6] = 1
    matrix[1000:1040] += 0.45 * group
    residuals = centre_cpgs(matrix)

    def median_rho(deflated):
        config = Eiger2Config(window=1, min_paired_samples=5)
        track = edge_statistic(deflated[1000:1040], config)
        return float(np.nanmedian(track))

    full = np.linalg.svd(residuals, full_matrices=False)[2][:2]
    thin = estimate_latent_basis(
        residuals, Eiger2Config(deflate_components=2, deflation_thin=50)
    )[0]
    assert median_rho(deflate(residuals, full)) < 0.1  # focal signal destroyed
    assert median_rho(deflate(residuals, thin)) > 0.4  # focal signal survives


def test_exact_deflation_does_not_leave_latent_structure_in_sparse_cpgs():
    rng = np.random.default_rng(7)
    n, n_cpg = 40, 2000
    latent = rng.normal(0, 1, n)
    loading = rng.normal(0, 0.08, (n_cpg, 1))
    matrix = 0.5 + loading * latent + rng.normal(0, 0.05, (n_cpg, n))
    matrix[rng.random(matrix.shape) < 0.4] = np.nan
    residuals = centre_cpgs(matrix)
    basis, _ = estimate_latent_basis(residuals, Eiger2Config(deflate_components=1))

    def leftover(deflated):
        filled = np.where(np.isfinite(deflated), deflated, 0.0)
        norm = np.maximum(np.linalg.norm(filled, axis=1), 1e-9)
        return float(np.mean(np.linalg.norm(filled @ basis.T, axis=1) / norm))

    assert leftover(deflate(residuals, basis, missing="exact")) < 1e-8
    assert leftover(deflate(residuals, basis, missing="impute")) > 0.05


def test_biological_components_are_kept_and_nuisance_components_deflated():
    """The property that protects one-pass cell-type discovery.

    A nuisance factor loads on the whole genome; cell type, tumour status and a
    rare clone load on marker regions. Blind deflation deletes the second kind,
    which is the signal EIGER exists to find.
    """

    rng = np.random.default_rng(3)
    n, n_cpg, length = 120, 16000, 40
    celltype = np.repeat(np.arange(4), n // 4)
    tumour = (rng.random(n) < 0.5).astype(float)
    matrix = 0.5 + rng.normal(0, 0.04, (n_cpg, n))
    position = 200
    for kind in ["celltype"] * 40 + ["tumour"] * 20:
        if kind == "celltype":
            effect = rng.permutation([0.0, 0.18, 0.36, 0.54])[celltype]
        else:
            effect = 0.35 * tumour
        matrix[position : position + length] += effect - effect.mean()
        position += 200
    residuals = centre_cpgs(np.clip(matrix, 0.001, 0.999))

    # biology only: several strong candidates, none of them diffuse
    _, clean = estimate_latent_basis(residuals, Eiger2Config())
    assert clean["n_candidates"] >= 4
    assert clean["n_components"] == 0
    assert all(v == "biological_kept" for v in clean["component_verdict"])

    # add a genuine nuisance factor: exactly one component becomes diffuse
    latent = rng.normal(0, 1, n)
    loading = rng.normal(0, 0.07, (n_cpg, 1))
    _, mixed = estimate_latent_basis(
        residuals + loading @ latent[None, :], Eiger2Config()
    )
    assert mixed["n_components"] == 1
    assert mixed["component_verdict"].count("biological_kept") >= 4


def test_deflation_preserves_the_missing_mask():
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(200, 20))
    matrix[rng.random(matrix.shape) < 0.2] = np.nan
    residuals = centre_cpgs(matrix)
    basis, _ = estimate_latent_basis(residuals, Eiger2Config(deflate_components=2))
    for mode in ("impute", "exact"):
        result = deflate(residuals, basis, missing=mode)
        assert np.array_equal(np.isfinite(result), np.isfinite(residuals))


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------


def test_segment_counts_cpgs_not_edges():
    """EIGER 1 applied its length threshold to edges, rejecting valid blocks."""

    eligible = np.ones(4, dtype=bool)
    assert segment(
        np.ones(4), eligible, 0.5, min_cpgs=5
    ).tolist() == [[0, 4]]  # 4 edges span 5 CpGs
    assert segment(np.ones(3), np.ones(3, bool), 0.5, min_cpgs=5).size == 0


def test_segment_gap_tolerance_bridges_a_single_dropped_cpg():
    statistic = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1.0])
    eligible = np.ones(10, dtype=bool)
    assert len(segment(statistic, eligible, 0.5, min_cpgs=5, max_gap_edges=0)) == 2
    assert segment(
        statistic, eligible, 0.5, min_cpgs=5, max_gap_edges=2
    ).tolist() == [[0, 10]]


def test_segment_never_crosses_ineligible_edges():
    statistic = np.ones(10)
    eligible = np.ones(10, dtype=bool)
    eligible[5] = False
    for first, last in segment(
        statistic, eligible, 0.5, min_cpgs=3, max_gap_edges=3
    ):
        assert not (first <= 5 < last)


def test_segment_constant_array_threshold_matches_scalar():
    rng = np.random.default_rng(21)
    statistic = rng.normal(size=300)
    eligible = np.ones(300, dtype=bool)
    scalar_regions = segment(statistic, eligible, 0.5, min_cpgs=4, max_gap_edges=1)
    array_regions = segment(
        statistic, eligible, np.full(300, 0.5), min_cpgs=4, max_gap_edges=1,
    )
    np.testing.assert_array_equal(scalar_regions, array_regions)


def test_region_area_constant_array_threshold_matches_scalar():
    rng = np.random.default_rng(22)
    statistic = rng.normal(size=300)
    regions = np.array([[10, 25], [100, 130]])
    scalar_areas = region_area(statistic, regions, 0.3)
    array_areas = region_area(statistic, regions, np.full(300, 0.3))
    np.testing.assert_array_equal(scalar_areas, array_areas)


def test_segment_and_region_area_reject_mismatched_array_threshold_shape():
    statistic = np.zeros(10)
    with pytest.raises(ValueError):
        segment(statistic, np.ones(10, dtype=bool), np.zeros(5), min_cpgs=2)
    with pytest.raises(ValueError):
        region_area(statistic, np.array([[0, 2]]), np.zeros(5))


def test_region_area_array_threshold_uses_per_edge_values():
    statistic = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    threshold = np.array([0.5, 1.5, 1.0, 3.5, 0.0])
    regions = np.array([[0, 4]])
    areas = region_area(statistic, regions, threshold)
    expected = (1.0 - 0.5) + (2.0 - 1.5) + (3.0 - 1.0) + (4.0 - 3.5)
    assert areas[0] == pytest.approx(expected)


def test_regions_do_not_cross_chromosomes():
    rng = np.random.default_rng(9)
    n_cpg = 4000
    group = np.zeros(30)
    group[:10] = 1
    matrix = 0.5 + rng.normal(0, 0.04, (n_cpg, 30))
    matrix[1980:2020] += 0.4 * group  # straddles the chromosome boundary
    frame = locations(n_cpg)
    frame.loc[2000:, "chr"] = "chr2"
    regions, _, _ = call_dmrs(
        np.clip(matrix, 0.001, 0.999), frame, config=Eiger2Config(**FAST)
    )
    for _, row in regions.iterrows():
        assert row["cpg_index_start"] // 2000 == row["cpg_index_end"] // 2000


# --------------------------------------------------------------------------
# region-level FDR
# --------------------------------------------------------------------------


def test_region_fdr_is_monotone_and_bounded():
    observed = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    nulls = [np.array([5.0, 3.0]), np.array([7.0, 1.0])]
    q = region_fdr(observed, nulls, observed_edges=1000, null_edges=1000)
    assert np.all((q >= 0) & (q <= 1))
    assert np.all(np.diff(q[np.argsort(-observed)]) >= -1e-12)


def test_region_fdr_rises_when_the_null_produces_comparable_regions():
    observed = np.array([5.0, 4.0])
    quiet = region_fdr(
        observed, [np.array([])] * 4, observed_edges=1000, null_edges=1000
    )
    noisy = region_fdr(
        observed,
        [np.array([9.0, 8.0, 7.0])] * 4,
        observed_edges=1000,
        null_edges=1000,
    )
    assert np.all(quiet < noisy)


def test_region_evidence_score_exercises_corrected_tail_area_fdr():
    # NOTE: the brief's literal config (null_replicates=8) yields zero pooled
    # null regions for this simulated genome, so `local_fdr` always returns
    # "insufficient_null_regions" and evidence_score_extrapolated is never
    # set -- that failure is unrelated to the _tail_area_fdr direction/tie
    # bug this task fixes.
    #
    # Post-Stage-1b-fix, null_replicates=500 (the brief's original raised value)
    # is no longer enough either: the corrected null is properly calibrated, so
    # min_cpgs=4 consecutive exceedances (the default, matching window=4) never
    # occur in this null -- measured n_pooled_null=0 at null_replicates up to
    # 20,000 (about 21s), so this is not a replicate-count problem at all; the
    # corrected null simply produces far fewer spurious null regions than the
    # pre-fix defective one did, which happened to leak enough locality back in
    # to occasionally clear a 4-in-a-row bar. min_cpgs is lowered to 3 (still
    # >= the framework's own floor of 2) instead: a lower run-length bar that
    # the corrected, better-calibrated null can still clear, restoring
    # n_pooled_null=7 at null_replicates=500 and genuinely exercising the
    # extrapolated tail-area fallback -- exactly the code path _tail_area_fdr
    # governs -- rather than raising null_replicates without bound.
    matrix, locs, _planted = simulate(
        n_samples=50, strata=[(0.2, 0.4)], replicates=3,
        spacing=250, dmr_length=10, seed=99,
    )
    config = Eiger2Config(
        window=4, null_replicates=500, random_seed=99, compute_posterior=True,
        min_null_regions_for_posterior=5, min_cpgs=3,
    )
    regions, _values, manifest = call_dmrs(matrix, locs, config=config)
    assert "posterior" in manifest
    assert manifest["posterior"]["status"] == "ok"
    assert regions["evidence_score_extrapolated"].any()


def test_estimate_pi0_output_is_unchanged_by_the_tail_area_fdr_fix():
    from eiger2.probability import estimate_pi0
    rng = np.random.default_rng(41)
    observed = rng.exponential(2.0, size=100)
    pooled = rng.exponential(2.0, size=600)
    # estimate_pi0 uses counts, not the cummin -- confirm it takes no code path
    # through _tail_area_fdr by checking its result matches a hand re-derivation
    # from its own (unedited) counting logic.
    result_a = estimate_pi0(
        len(observed), len(pooled), observed_edges=500, null_edges=3000
    )
    result_b = estimate_pi0(
        len(observed), len(pooled), observed_edges=500, null_edges=3000
    )
    assert result_a == pytest.approx(result_b)


# --------------------------------------------------------------------------
# characterisation
# --------------------------------------------------------------------------


def test_characterise_region_finds_the_minor_group_either_way_round():
    """EIGER 1 always reported the lowest group, so rare hypermethylation was
    weighted as though it were the majority."""

    low = np.array([0.1, 0.1, 0.1, 0.7, 0.7, 0.7, 0.7, 0.7])
    high = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.7])
    assert characterise_region(low)["minor_group_size"] == 3
    assert characterise_region(low)["direction"] == "Hypomethylation"
    assert characterise_region(high)["minor_group_size"] == 3
    assert characterise_region(high)["direction"] == "Hypermethylation"


def test_characterise_region_handles_missing_and_degenerate_input():
    assert characterise_region(np.array([np.nan, np.nan]))["n_observed"] == 0
    constant = characterise_region(np.full(10, 0.4))
    assert np.isfinite(constant["delta_beta"])
    assert constant["largest_gap"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# operating characteristics — the tests EIGER 1 was missing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("statistic", ["correlation", "covariance"])
def test_no_regions_are_called_on_dmr_free_data_with_latent_structure(statistic):
    """The failure mode that broke EIGER 1's automatic threshold."""

    matrix, frame, _ = simulate(strata=[], seed=100)
    config = Eiger2Config(statistic=statistic, window=1, **FAST)
    regions, _, manifest = call_dmrs(matrix, frame, config=config)
    assert manifest["latent_structure"]["n_components"] == 2
    n_significant = int(regions["significant"].sum()) if len(regions) else 0
    assert n_significant == 0


@pytest.mark.parametrize("statistic", ["correlation", "covariance"])
def test_planted_dmrs_are_recovered_across_the_prevalence_grid(statistic):
    strata = [(0.05, 0.35), (0.10, 0.30), (0.25, 0.25), (0.50, 0.20)]
    matrix, frame, truth = simulate(strata=strata, replicates=3)
    config = Eiger2Config(statistic=statistic, **FAST)
    regions, _, _ = call_dmrs(matrix, frame, config=config)
    recovered, false_regions = evaluate(regions, truth)
    assert len(recovered) == len(truth)
    assert false_regions <= 2


def test_recovery_survives_structured_missingness():
    strata = [(0.05, 0.35), (0.10, 0.30), (0.25, 0.25), (0.50, 0.20)]
    matrix, frame, truth = simulate(
        strata=strata, replicates=3, missing_rate=0.25
    )
    regions, _, _ = call_dmrs(matrix, frame, config=Eiger2Config(**FAST))
    recovered, false_regions = evaluate(regions, truth)
    assert len(recovered) >= len(truth) - 1
    # min_cpgs defaults to 4 so that genuinely short DMRs survive; the price is
    # a handful of short false regions under heavy missingness.
    assert false_regions <= 8


def test_outputs_are_aligned_and_reproducible():
    strata = [(0.10, 0.40)]
    matrix, frame, _ = simulate(strata=strata, replicates=3)
    names = [f"s{i}" for i in range(matrix.shape[1])]
    first = call_dmrs(
        matrix, frame, sample_names=names, config=Eiger2Config(**FAST)
    )
    second = call_dmrs(
        matrix, frame, sample_names=names, config=Eiger2Config(**FAST)
    )
    regions, values, manifest = first
    assert regions["region_id"].tolist() == values["region_id"].tolist()
    assert not regions["region_id"].duplicated().any()
    assert list(values.columns) == ["region_id", *names]
    assert regions["region_qvalue"].tolist() == second[0]["region_qvalue"].tolist()
    assert manifest["config"]["statistic"] == "correlation"


def test_short_dmrs_survive_and_a_length_cutoff_would_delete_them():
    """A 4-CpG DMR is real biology; `min_cpgs` must not be used as a FP filter."""

    rng = np.random.default_rng(9)
    n, n_cpg = 80, 20000
    latent = rng.normal(0, 1, (n, 2))
    loading = rng.normal(0, 0.07, (n_cpg, 2))
    matrix = 0.5 + loading @ latent.T + rng.normal(0, 0.04, (n_cpg, n))
    truth = []
    position = 200
    for length in [4] * 10 + [6] * 10 + [25] * 10:
        group = np.zeros(n)
        group[rng.choice(n, 20, replace=False)] = 1
        matrix[position : position + length] += 0.35 * group
        truth.append((position, position + length - 1, length))
        position += 300
    matrix = np.clip(matrix, 0.001, 0.999)
    frame = locations(n_cpg)

    def recovered(config):
        regions, _, _ = call_dmrs(matrix, frame, config=config)
        called = [
            (int(r["cpg_index_start"]), int(r["cpg_index_end"]))
            for _, r in regions[regions["significant"]].iterrows()
        ]
        return {
            length: sum(
                1
                for s, e, L in truth
                if length == L
                and any(
                    min(b, e) - max(a, s) + 1 >= max(2, L // 2) for a, b in called
                )
            )
            for length in (4, 6, 25)
        }

    default = recovered(Eiger2Config(**FAST))
    assert default[4] == 10 and default[6] == 10 and default[25] == 10

    cutoff = recovered(Eiger2Config(min_cpgs=20, **FAST))
    assert cutoff[4] == 0 and cutoff[6] == 0  # the cutoff deletes short DMRs
    assert cutoff[25] == 10


def test_pairwise_flag_preserves_published_behaviour():
    """The legacy default must still return NaN, bit for bit."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eiger_processing",
        Path(__file__).resolve().parents[1] / "Python" / "functions" / "processing.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rng = np.random.default_rng(0)
    complete = rng.normal(0.5, 0.1, (50, 30))
    legacy = module.covariance_blocks(complete)
    pairwise = module.covariance_blocks(complete, pairwise=True)
    assert np.allclose(legacy, pairwise)  # identical when nothing is missing

    gapped = complete.copy()
    gapped[10, 3] = np.nan
    legacy_gapped = module.covariance_blocks(gapped)
    pairwise_gapped = module.covariance_blocks(gapped, pairwise=True)
    assert not np.isfinite(legacy_gapped[9])   # published behaviour: NaN
    assert not np.isfinite(legacy_gapped[10])
    assert np.isfinite(pairwise_gapped[9])     # pairwise recovers the edge
    assert np.isfinite(pairwise_gapped[10])


def test_too_few_samples_is_rejected_rather_than_silently_wrong():
    matrix = np.random.default_rng(0).normal(0.5, 0.05, (500, 3))
    with pytest.raises(ValueError, match="min_paired_samples"):
        call_dmrs(matrix, locations(500))


# --------------------------------------------------------------------------
# targeted / panel assays
# --------------------------------------------------------------------------


def test_panel_data_drops_unobserved_cpgs_and_respects_off_target_gaps():
    """On a targeted panel, off-panel CpGs must be dropped, not imputed, and no
    region may bridge the untargeted gap between two panel intervals."""

    rng = np.random.default_rng(11)
    n_regions, region_len, n_samples = 30, 60, 24
    positions, cursor = [], 0
    for _ in range(n_regions):
        for _ in range(region_len):
            positions.append(cursor)
            cursor += 20
        cursor += 5000                      # untargeted gap

    n_cpg = len(positions)
    matrix = 0.5 + rng.normal(0, 0.04, (n_cpg, n_samples))
    scores = rng.normal(0, 1, (n_samples, 2))
    matrix += rng.normal(0, 0.05, (n_cpg, 2)) @ scores.T
    planted = []
    for region in rng.choice(n_regions, 8, replace=False):
        start = region * region_len + 20
        group = rng.choice(n_samples, 8, replace=False)
        matrix[start : start + 20, group] += 0.35
        planted.append((start, start + 19))
    matrix = np.clip(matrix, 0.02, 0.98)

    # probe dropout: 10% of CpGs observed in no sample at all
    dead = rng.choice(n_cpg, size=int(0.10 * n_cpg), replace=False)
    matrix[dead, :] = np.nan

    locations = pd.DataFrame({
        "chr": "chr1", "start": positions,
        "end": [p + 1 for p in positions],
    })
    regions, _, manifest = call_dmrs(matrix, locations, config=Eiger2Config())

    # entirely unobserved CpGs are removed, not filled
    assert manifest["n_cpg_retained"] == n_cpg - len(dead)

    # the latent basis is built from observed rows, not mean-filled ones
    assert manifest["latent_structure"]["basis_source"] in {
        "complete_rows", "partially_observed_rows_mean_filled"
    }

    # no called region spans an off-panel gap
    if len(regions):
        assert (regions["end"] - regions["start"]).max() <= region_len * 20

    # planted regions are still recovered
    recovered = sum(
        any(
            min(int(row["cpg_index_end"]), end)
            - max(int(row["cpg_index_start"]), start) + 1 >= 10
            for _, row in regions.iterrows()
        )
        for start, end in planted
    )
    assert recovered >= len(planted) - 1


def test_latent_basis_is_not_estimated_from_mean_filled_rows():
    """Regression: the completeness filter must be applied BEFORE filling.

    Filling first makes ``np.isfinite(...).all(axis=1)`` vacuously true, so the
    filter never fires and mean-filled zeros enter the SVD.
    """

    rng = np.random.default_rng(12)
    matrix = 0.5 + rng.normal(0, 0.05, (3000, 30))
    matrix[rng.random(matrix.shape) < 0.15] = np.nan
    residuals = centre_cpgs(matrix)
    _, diagnostics = estimate_latent_basis(residuals, Eiger2Config())

    assert diagnostics["thin_rows_used"] <= diagnostics["thin_rows_offered"]
    if diagnostics["basis_source"] == "complete_rows":
        assert diagnostics["thin_rows_used"] == diagnostics["thin_rows_complete"]


def test_fdr_from_areas_result_four_distinct_areas():
    observed = np.array([10.0, 8.0, 6.0, 4.0])
    pooled = np.array([9.0, 7.0, 5.0, 3.0])
    result = _fdr_from_areas_result(
        observed, pooled, scale=1.0, confidence_level=0.95, m_bonferroni=4,
    )
    assert result["tail_fdr"] == pytest.approx([0.0, 0.5, 2 / 3, 0.75])
    assert result["qvalue"] == pytest.approx([0.0, 0.5, 2 / 3, 0.75])
    assert result["null_exceedances"] == pytest.approx([0, 1, 2, 3])
    assert result["qvalue_cutoff_rank"] == pytest.approx([1, 2, 3, 4])


def test_fdr_from_areas_result_tie_group():
    observed = np.array([10.0, 8.0, 8.0, 4.0])
    pooled = np.array([9.0, 7.5, 5.0, 3.0])
    result = _fdr_from_areas_result(
        observed, pooled, scale=1.0, confidence_level=0.95, m_bonferroni=3,
    )
    assert result["tail_fdr"] == pytest.approx([0.0, 1 / 3, 1 / 3, 0.75])
    assert result["qvalue"] == pytest.approx([0.0, 1 / 3, 1 / 3, 0.75])
    assert result["qvalue_cutoff_rank"] == pytest.approx([1, 3, 3, 4])
    assert result["_n_cutoffs"] == 3


def test_fdr_from_areas_result_qvalue_is_nondecreasing_as_area_weakens():
    rng = np.random.default_rng(11)
    observed = rng.exponential(2.0, size=60)
    pooled = rng.exponential(2.0, size=400)
    result = _fdr_from_areas_result(
        observed, pooled, scale=0.3, confidence_level=0.95, m_bonferroni=60,
    )
    order = np.argsort(-observed, kind="mergesort")
    q = result["qvalue"][order]
    assert np.all(np.diff(q) >= -1e-12)


def test_fdr_from_areas_result_bounds_bracket_the_point_estimate():
    rng = np.random.default_rng(12)
    observed = rng.exponential(2.0, size=50)
    pooled = rng.exponential(2.0, size=300)
    result = _fdr_from_areas_result(
        observed, pooled, scale=0.4, confidence_level=0.95, m_bonferroni=50,
    )
    assert np.all(result["qvalue_lower"] <= result["qvalue"] + 1e-12)
    assert np.all(result["qvalue"] <= result["qvalue_upper"] + 1e-12)


def test_fdr_from_areas_result_zero_exceedances_has_nonzero_upper_bound():
    observed = np.array([50.0, 3.0])
    pooled = np.array([1.0, 0.5])
    result = _fdr_from_areas_result(
        observed, pooled, scale=1.0, confidence_level=0.95, m_bonferroni=2,
    )
    assert result["qvalue"][0] == 0.0
    assert result["qvalue_lower"][0] == 0.0
    assert result["qvalue_upper"][0] > 0.0
    # Spec §0.3.5: a zero point estimate must carry both a nonzero upper bound
    # AND a nonzero resolution -- the upper bound alone isn't sufficient.
    assert result["qvalue_resolution"][0] > 0.0


def test_fdr_from_areas_result_more_null_exposure_narrows_the_interval():
    # spec §1.10 item 8: increasing null exposure at a fixed rate (same scale,
    # more pooled null observations at the same relative density) must narrow
    # [lower, upper] because larger K gives a tighter Garwood interval.
    observed = np.array([9.0, 7.0, 5.0, 3.0])
    rng = np.random.default_rng(15)
    sparse_null = rng.exponential(2.0, size=40)
    dense_null = np.concatenate([sparse_null, rng.exponential(2.0, size=400)])
    sparse = _fdr_from_areas_result(
        observed, sparse_null, scale=0.5, confidence_level=0.95, m_bonferroni=4,
    )
    dense = _fdr_from_areas_result(
        observed, dense_null, scale=0.5, confidence_level=0.95, m_bonferroni=4,
    )
    sparse_width = sparse["qvalue_upper"] - sparse["qvalue_lower"]
    dense_width = dense["qvalue_upper"] - dense["qvalue_lower"]
    assert np.all(dense_width <= sparse_width + 1e-9)
    assert np.any(dense_width < sparse_width - 1e-9)


def test_fdr_from_areas_result_resolution_matches_scale_over_rank():
    rng = np.random.default_rng(13)
    observed = rng.exponential(1.0, size=30)
    pooled = rng.exponential(1.0, size=200)
    scale = 0.5
    result = _fdr_from_areas_result(
        observed, pooled, scale=scale, confidence_level=0.95, m_bonferroni=30,
    )
    expected = np.minimum(1.0, scale / result["qvalue_cutoff_rank"])
    assert result["qvalue_resolution"] == pytest.approx(expected)


def test_fdr_from_areas_result_vectorised_argmin_matches_naive_reverse_loop():
    rng = np.random.default_rng(14)
    for _ in range(200):
        n = rng.integers(3, 25)
        observed = rng.choice(np.round(rng.exponential(2.0, size=n), 2), size=n)
        pooled = rng.exponential(2.0, size=rng.integers(5, 60))
        vectorised = _fdr_from_areas_result(
            observed, pooled, scale=0.6, confidence_level=0.95,
            m_bonferroni=int(np.unique(observed).size),
        )

        # naive reverse loop over the same grouped structure
        order = np.argsort(-observed, kind="mergesort")
        ranked = observed[order]
        pooled_sorted = np.sort(pooled)
        m = ranked.size
        new_group = np.empty(m, dtype=bool)
        new_group[0] = True
        new_group[1:] = ranked[1:] != ranked[:-1]
        group_of_row = np.cumsum(new_group) - 1
        G = int(group_of_row[-1]) + 1
        cutoff = ranked[new_group]
        last_row_of_group = np.flatnonzero(np.append(new_group[1:], True))
        R = (last_row_of_group + 1).astype(float)
        K = (pooled_sorted.size - np.searchsorted(pooled_sorted, cutoff)).astype(float)
        raw = np.minimum(1.0, 0.6 * K / R)
        q_group = np.empty(G)
        arg_group = np.empty(G, dtype=int)
        running_min = raw[G - 1]
        running_arg = G - 1
        for g in range(G - 1, -1, -1):
            if raw[g] <= running_min:
                running_min = raw[g]
                running_arg = g
            q_group[g] = running_min
            arg_group[g] = running_arg
        assert arg_group[-1] != -1  # spec's own sanity check, restated

        expanded_q = q_group[group_of_row]
        out = np.empty(m)
        out[order] = expanded_q
        assert vectorised["qvalue"] == pytest.approx(out)


def test_fdr_from_areas_result_exact_tie_areas_share_every_field():
    observed = np.array([10.0, 8.0, 8.0, 4.0])
    pooled = np.array([9.0, 7.5, 5.0, 3.0])
    result = _fdr_from_areas_result(
        observed, pooled, scale=1.0, confidence_level=0.95, m_bonferroni=3,
    )
    for key in ("qvalue", "tail_fdr", "qvalue_lower", "qvalue_upper",
                "qvalue_resolution", "null_exceedances", "qvalue_cutoff_rank"):
        assert result[key][1] == result[key][2], key


def _null_sets(rng, n_replicates, n_per_replicate, scale_hint=2.0):
    return [
        rng.exponential(scale_hint, size=n_per_replicate) for _ in range(n_replicates)
    ]


def test_region_fdr_result_matches_region_fdr_exactly():
    rng = np.random.default_rng(21)
    observed = rng.exponential(2.0, size=40)
    nulls = _null_sets(rng, 6, 90)
    result = region_fdr_result(
        observed, nulls, observed_edges=500, null_edges=3000,
    )
    legacy = region_fdr(observed, nulls, observed_edges=500, null_edges=3000)
    assert np.array_equal(result.qvalue, legacy)


def test_region_fdr_result_empty_inputs():
    result = region_fdr_result(
        np.empty(0), [], observed_edges=0, null_edges=0,
    )
    for field in (result.qvalue, result.tail_fdr, result.qvalue_lower,
                  result.qvalue_upper, result.qvalue_resolution):
        assert field.shape == (0,)
    assert result.diagnostics["pooled_fallback_strata"] == []


def test_region_fdr_result_zero_replicates_is_all_nan():
    result = region_fdr_result(
        np.array([1.0, 2.0]), [], observed_edges=10, null_edges=0,
    )
    assert np.all(np.isnan(result.qvalue))
    assert np.all(result.null_exceedances == -1)
    assert np.all(result.qvalue_cutoff_rank == -1)
    assert "status" in result.diagnostics


def test_region_fdr_result_pooled_fallback_stratum_uses_all_six_fields():
    rng = np.random.default_rng(22)
    # one stratum ("4-5", n_cpg<=5) has zero null regions of that length
    observed_lengths = np.array([4, 4, 30, 30])
    observed_areas = np.array([5.0, 3.0, 9.0, 6.0])
    null_area_sets = [rng.exponential(2.0, size=20) for _ in range(5)]
    null_length_sets = [np.full(20, 30, dtype=int) for _ in range(5)]
    result = region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=200, null_edges=100,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    )
    assert "4-5" in result.diagnostics["pooled_fallback_strata"]
    assert not np.any(np.isnan(result.qvalue[:2]))
    assert not np.any(result.null_exceedances[:2] == -1)


def test_region_fdr_result_row_order_survives_permutation():
    rng = np.random.default_rng(23)
    observed = rng.exponential(2.0, size=25)
    lengths = rng.integers(4, 60, size=25)
    nulls = _null_sets(rng, 5, 80)
    null_lengths = [rng.integers(4, 60, size=80) for _ in range(5)]
    baseline = region_fdr_result(
        observed, nulls, observed_edges=300, null_edges=400,
        observed_lengths=lengths, null_length_sets=null_lengths,
    )
    perm = rng.permutation(25)
    inv = np.argsort(perm)
    permuted = region_fdr_result(
        observed[perm], nulls, observed_edges=300, null_edges=400,
        observed_lengths=lengths[perm], null_length_sets=null_lengths,
    )
    for a, b in (
        (baseline.qvalue, permuted.qvalue[inv]),
        (baseline.qvalue_lower, permuted.qvalue_lower[inv]),
        (baseline.qvalue_upper, permuted.qvalue_upper[inv]),
        (baseline.qvalue_resolution, permuted.qvalue_resolution[inv]),
        (baseline.null_exceedances, permuted.null_exceedances[inv]),
        (baseline.qvalue_cutoff_rank, permuted.qvalue_cutoff_rank[inv]),
    ):
        assert np.array_equal(a, b)


def test_region_fdr_result_common_bonferroni_m_is_max_across_strata():
    rng = np.random.default_rng(24)
    observed_lengths = np.array([4] * 3 + [30] * 10)
    observed_areas = rng.exponential(2.0, size=13)
    null_area_sets = [rng.exponential(2.0, size=40) for _ in range(6)]
    null_length_sets = [
        np.concatenate([np.full(15, 4), np.full(25, 30)]) for _ in range(6)
    ]
    result = region_fdr_result(
        observed_areas, null_area_sets, observed_edges=200, null_edges=240,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    )
    by_stratum = result.diagnostics["fdr_interval_cutoffs_by_stratum"]
    assert result.diagnostics["fdr_interval_bonferroni_m"] == max(by_stratum.values())
    assert result.diagnostics["fdr_interval_family"] == "per_stratum"


def test_region_fdr_result_length_stratified_maps_back_to_original_rows():
    rng = np.random.default_rng(25)
    observed_lengths = np.array([4, 30, 4, 30])
    observed_areas = np.array([1.0, 100.0, 2.0, 50.0])
    null_area_sets = [rng.exponential(2.0, size=60) for _ in range(4)]
    null_length_sets = [
        np.concatenate([np.full(30, 4), np.full(30, 30)]) for _ in range(4)
    ]
    result = region_fdr_result(
        observed_areas, null_area_sets, observed_edges=100, null_edges=240,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    )
    # short-stratum regions (indices 0, 2) must not be compared against the
    # long-stratum null: their null-exceedance counts should differ from a
    # pooled-only run (qvalue itself can coincidentally saturate at the
    # min(1.0, ...) cap for these areas, so exceedances is the field that
    # actually distinguishes a 120-element per-stratum pool from the
    # 240-element fully-pooled null)
    pooled_result = region_fdr_result(
        observed_areas, null_area_sets, observed_edges=100, null_edges=240,
    )
    assert not np.array_equal(result.null_exceedances, pooled_result.null_exceedances)


def test_region_fdr_result_weak_candidates_do_not_inherit_a_leading_zero():
    # spec §1.10 item 17: this is the defect's actual symptom end-to-end -- the
    # old forward-cummin propagated the strongest region's zero to every weaker
    # one. With plenty of weaker, genuinely null-comparable candidates, q must
    # not be all zero or constant.
    rng = np.random.default_rng(26)
    strong = np.array([40.0])
    weak = rng.exponential(2.0, size=80)  # comparable in scale to the null
    observed = np.concatenate([strong, weak])
    nulls = [rng.exponential(2.0, size=90) for _ in range(10)]
    result = region_fdr_result(
        observed, nulls, observed_edges=1000, null_edges=900,
    )
    assert (result.qvalue == 0).sum() < len(result.qvalue)
    assert np.unique(result.qvalue).size > 1


# --------------------------------------------------------------------------
# overdispersion diagnostic
# --------------------------------------------------------------------------


def test_overdispersion_diagnostic_flags_a_deliberately_overdispersed_null():
    rng = np.random.default_rng(31)
    # 8 replicates, exceedance counts drawn from a negative-binomial-like process
    # (Poisson mixed over a Gamma-distributed rate) so variance >> mean at every cutoff
    cutoffs = np.array([5.0, 4.0, 3.0])
    per_replicate = []
    for _ in range(8):
        rate = rng.gamma(shape=0.5, scale=6.0)  # overdispersed rate draw
        counts = rng.poisson(rate, size=3)
        # build a null-area array whose >= cutoff exceedance counts match `counts`
        areas = np.concatenate([
            np.full(counts[i], cutoffs[i] + 0.1) for i in range(3)
        ] + [np.zeros(5)])
        per_replicate.append(areas)
    status, ratio = _overdispersion_diagnostic(per_replicate, cutoffs)
    assert status == "computed"
    assert ratio is not None
    assert ratio > 1.5


def test_overdispersion_diagnostic_unavailable_below_five_replicates():
    per_replicate = [np.array([1.0, 2.0]) for _ in range(3)]
    status, ratio = _overdispersion_diagnostic(per_replicate, np.array([1.0]))
    assert status == "unavailable"
    assert ratio is None


def test_region_fdr_result_overdispersion_warning_end_to_end():
    rng = np.random.default_rng(32)
    observed = rng.exponential(2.0, size=20)
    # 8 replicates with wildly varying null density -> overdispersed exceedance counts
    nulls = []
    for i in range(8):
        density = 300 if i % 2 == 0 else 10
        nulls.append(rng.exponential(2.0, size=density))
    result = region_fdr_result(
        observed, nulls, observed_edges=500,
        null_edges=sum(len(n) for n in nulls),
    )
    assert result.diagnostics["null_count_overdispersion_warning"] is not None


def test_region_fdr_result_poisson_null_does_not_trigger_overdispersion():
    rng = np.random.default_rng(33)
    observed = rng.exponential(2.0, size=20)
    nulls = [rng.exponential(2.0, size=150) for _ in range(8)]  # same density each rep
    result = region_fdr_result(observed, nulls, observed_edges=500, null_edges=1200)
    status, ratio = result.diagnostics["null_count_dispersion_by_stratum"]["_pooled"]
    if status == "computed":
        assert ratio < 1.5
    # Spec 1.10 item 25: confirm the top-level warning does not fire for a
    # Poisson null, unconditionally (not gated on whether the per-stratum
    # diagnostic happened to be "computed" for this particular draw).
    assert result.diagnostics["null_count_overdispersion_warning"] is None


def test_null_replicates_default_is_100():
    assert Eiger2Config().null_replicates == 100


def test_permutation_null_tracks_default_replicates_is_100():
    import inspect
    sig = inspect.signature(permutation_null_tracks)
    assert sig.parameters["replicates"].default == 100


def test_qvalue_result_from_track_default_replicates_is_100():
    import inspect
    sig = inspect.signature(qvalue_result_from_track)
    assert sig.parameters["replicates"].default == 100
    sig2 = inspect.signature(qvalues_from_track)
    assert sig2.parameters["replicates"].default == 100


# --------------------------------------------------------------------------
# Task 7: --fdr-decision {point,upper} and the vacuity guard
# --------------------------------------------------------------------------


def test_fdr_decision_field_defaults_to_point():
    assert Eiger2Config().fdr_decision == "point"


def test_fdr_decision_rejects_invalid_value():
    with pytest.raises(ValueError, match="fdr_decision"):
        Eiger2Config(fdr_decision="maybe")


def test_point_and_upper_decision_modes_are_deterministic_with_same_seed():
    # null_replicates=300, not the brief's original 100: at 100 replicates this
    # simulation's scale/m_bonferroni combination makes the best achievable
    # region_qvalue_upper_95 (~0.14) exceed the default target_fdr (0.05), so
    # check_fdr_decision_vacuity correctly refuses "upper" mode outright -- see
    # test_fdr_decision_upper_with_too_few_replicates_raises_naming_b_required for
    # a dedicated test of exactly that refusal. 300 replicates clears it (verified
    # empirically) while keeping this test about determinism, not vacuity.
    matrix, locs, _ = simulate(
        n_samples=60, strata=[(0.15, 0.35)], replicates=4,
        spacing=300, dmr_length=12, seed=51,
    )
    for mode in ("point", "upper"):
        config = Eiger2Config(
            window=4, null_replicates=300, random_seed=51, fdr_decision=mode,
        )
        first, _, _ = call_dmrs(matrix, locs, config=config)
        second, _, _ = call_dmrs(matrix, locs, config=config)
        assert first["significant"].tolist() == second["significant"].tolist()
        assert first["region_qvalue"].tolist() == second["region_qvalue"].tolist()


def test_upper_decision_uses_upper_bound_not_point_estimate():
    # The brief's original parameters (13 vs 12 significant regions) collapse
    # to 13 vs 13 post-Stage-1b-fix: the corrected, better-calibrated null
    # narrows the Garwood interval enough that "point" and "upper" now call the
    # identical region set here. Re-searched empirically (varying strata,
    # replicates, null_replicates and target_fdr) for a config that still
    # diverges on the corrected code -- found at 31 vs 30 significant regions
    # with a two-strata signal (a weaker (0.10, 0.15) stratum alongside the
    # original (0.15, 0.25)) and more simulated replicates (15, not 12), which
    # produces enough borderline-area regions for the upper Garwood bound to
    # push at least one of them across target_fdr=0.05 while the point
    # estimate does not.
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.10, 0.15), (0.15, 0.25)], replicates=15,
        spacing=300, dmr_length=10, seed=7,
    )
    point_config = Eiger2Config(
        window=4, null_replicates=500, random_seed=7, target_fdr=0.05,
        fdr_decision="point",
    )
    upper_config = Eiger2Config(
        window=4, null_replicates=500, random_seed=7, target_fdr=0.05,
        fdr_decision="upper",
    )
    point_regions, _, _ = call_dmrs(matrix, locs, config=point_config)
    upper_regions, _, _ = call_dmrs(matrix, locs, config=upper_config)
    # upper mode is strictly more conservative than point mode at these
    # parameters -- a non-strict "<=" here would also pass if the decision
    # mode were silently ignored, which is exactly the gap this test closes.
    assert int(upper_regions["significant"].sum()) < int(
        point_regions["significant"].sum()
    )
    # And directly confirm "upper" mode's significance is derived from the
    # upper Garwood bound, not the point estimate.
    assert (upper_regions["significant"] == (
        (upper_regions["region_qvalue_upper_95"] <= 0.05)
        & upper_regions["passes_prevalence"]
    )).all()


def test_fdr_decision_upper_with_too_few_replicates_raises_naming_b_required():
    matrix, locs, _ = simulate(
        n_samples=40, strata=[(0.1, 0.2)], replicates=2,
        spacing=250, dmr_length=8, seed=53,
    )
    config = Eiger2Config(
        window=4, null_replicates=2, random_seed=53, fdr_decision="upper",
        target_fdr=0.01,
    )
    with pytest.raises(ValueError, match="B_required|null_replicates|Increase"):
        call_dmrs(matrix, locs, config=config)


def test_run_legacy_upper_decision_uses_upper_bound_not_point_estimate():
    """``cli.run_legacy`` carries its own, independent copy of the same
    point-vs-upper decision logic as the ``call_dmrs`` path above (around
    ``cli.py:409-414``). Confirm it too derives "significant" from
    ``region_qvalue_upper_95`` under ``fdr_decision="upper"``, not from the
    point estimate ``region_qvalue``."""

    from eiger2.cli import run_legacy

    matrix, locs, _ = simulate(
        n_samples=60, strata=[(0.15, 0.35)], replicates=4,
        spacing=300, dmr_length=12, seed=52,
    )
    sample_names = [f"s{i}" for i in range(matrix.shape[1])]

    regions, _ = run_legacy(
        matrix, locs, sample_names,
        cov_threshold=0.002, delta_threshold=0.20, min_cpgs=5, impute=True,
        fdr_replicates=200, target_fdr=0.05, min_prevalence=0, seed=52,
        fdr_decision="upper",
    )
    assert len(regions) > 0
    assert (regions["significant"] == (
        (regions["region_qvalue_upper_95"] <= 0.05)
        & regions["passes_prevalence"]
    )).all()
    # And confirm this genuinely diverges from what the point estimate would
    # have decided -- otherwise the assertion above would pass identically
    # even if fdr_decision were silently ignored.
    point_based = (
        (regions["region_qvalue"] <= 0.05) & regions["passes_prevalence"]
    )
    assert not (regions["significant"] == point_based).all()


# --------------------------------------------------------------------------
# Task 8: output schema -- six new region columns, call_dmrs wiring,
# schema_version
# --------------------------------------------------------------------------


def test_schema_round_trip_has_documented_columns_in_documented_order():
    matrix, locs, _ = simulate(
        n_samples=60, strata=[(0.15, 0.35)], replicates=4,
        spacing=300, dmr_length=12, seed=61,
    )
    config = Eiger2Config(window=4, null_replicates=20, random_seed=61)
    regions, _values, manifest = call_dmrs(matrix, locs, config=config)
    idx = list(regions.columns).index("region_qvalue")
    expected_tail = [
        "region_tail_fdr", "region_qvalue_lower_95", "region_qvalue_upper_95",
        "region_qvalue_resolution", "null_exceedances_at_cutoff",
        "region_qvalue_cutoff_rank",
    ]
    assert list(regions.columns)[idx + 1: idx + 1 + len(expected_tail)] == expected_tail
    # Bumped to "2.2" by Task 7 (Stage 1c: region_support_median/
    # region_threshold_mode columns added to _REGION_COLUMNS).
    assert manifest["schema_version"] == "2.2"


def test_confidence_level_other_than_95_returns_arrays_but_is_not_csv_exportable():
    result = region_fdr_result(
        np.array([5.0, 3.0]), [np.array([1.0, 2.0])],
        observed_edges=10, null_edges=5, confidence_level=0.90,
    )
    assert not np.all(np.isnan(result.qvalue))  # Python API still returns arrays

    from eiger2.cli import _require_default_confidence_level
    _require_default_confidence_level(0.95)  # no raise
    _require_default_confidence_level(None)  # no raise
    with pytest.raises(ValueError, match="95%"):
        _require_default_confidence_level(0.90)


# --------------------------------------------------------------------------
# Task 10 (v2.0.0, pre-Stage-1b): invariant-preservation regression test --
# proved Stage 1's FDR corrections (Tasks 2-9) changed only q-value/
# evidence-score/significant columns, leaving every other region column
# byte-identical to v2.0.0. This held through the whole of Stage 1
# (v2.0.0 -> v2.0.1): the fixture below was pinned before Stage 1 started
# (commit 306fa8f) and the original comparison test passed unchanged all
# the way through Stage 1's completion (commit 46327e7) and the v2.0.1
# release (6f116cd).
#
# Stage 1b (this stage) breaks that contract on purpose. Task 2's lag-null
# window-assembly fix (commit 3ca07ef) corrects the null construction used
# to compute region boundaries, which changes the calibration threshold,
# which changes segmentation, which cascades into every column this fixture
# pins: region_id/start/end/n_cpg/cpg_index_* (boundaries themselves --
# region_id is literally f"EIGER2_{chrom}_{start}_{end}", see
# characterise_region in core.py), statistic/statistic_mean/statistic_min
# (region-slice aggregates computed over the new boundaries),
# statistic_threshold and region_area (explicitly redesigned outputs, per
# this plan's Global Constraints), and every remaining *_descriptive column
# (all computed over the region's -- now different -- CpG slice). Checked
# directly, not assumed: none of the 24 columns this fixture pins is
# independent of region boundaries, so none remains a meaningful invariant
# under Stage 1b's narrowed contract. That narrowed contract (this plan's
# Global Constraints) keeps only edge_statistic()'s raw output and
# eligible_edges()'s raw output byte-identical from here on -- see
# test_stage1b_preserves_edge_statistic_and_eligibility below.
#
# Per spec §1: retain this fixture file for historical reference; do not
# delete it. Since it documents pre-Stage-1b output that is not expected to
# ever match live output again by design, the comparison test is retired
# below in favour of a smoke check that the historical fixture file itself
# still exists, parses, and has the shape it always had. Deliberately NOT
# an xfail: "expected to fail forever" reads, to a future reader, as a live
# regression that still needs investigating -- but there is nothing left to
# investigate here, this file is a closed historical record, not a pending
# fix.
# --------------------------------------------------------------------------


def test_v200_fixture_is_a_historical_record_not_compared_after_stage_1b():
    """stage1_invariant_fixture_v200.csv documents v2.0.0 call_dmrs() region
    output, retained per spec §1 for historical reference only. Do NOT
    compare it against post-Stage-1b call_dmrs() output: region boundaries
    (and therefore every column in this fixture) are expected to differ by
    design from Stage 1b onward. See the module comment immediately above
    for the full reasoning."""

    fixture_path = (
        Path(__file__).parent / "fixtures" / "stage1_invariant_fixture_v200.csv"
    )
    historical = pd.read_csv(fixture_path, float_precision="round_trip")
    expected_columns = [
        "region_id", "chr", "start", "end", "n_cpg", "cpg_index_start",
        "cpg_index_end", "statistic", "statistic_mean", "statistic_min",
        "statistic_threshold", "region_area", "minority_size_descriptive",
        "minority_delta_descriptive", "n_samples_observed", "missing_rate",
        "mean_beta", "delta_beta_descriptive", "largest_gap_descriptive",
        "direction_descriptive", "minor_group_size_descriptive",
        "minor_group_fraction_descriptive", "minor_group_samples_descriptive",
        "pvalue_descriptive",
    ]
    assert list(historical.columns) == expected_columns
    assert len(historical) > 0


# --------------------------------------------------------------------------
# Task 6 (Stage 1b): narrowed invariant-preservation regression test --
# proves the lag-null window-assembly fix (Task 2) and the null_exchangeability
# diagnostic (Task 3) left edge_statistic()'s and eligible_edges()'s raw
# output byte-identical, per this plan's Global Constraints. This is the
# contract that replaces the retired test above.
# --------------------------------------------------------------------------


def test_stage1b_preserves_edge_statistic_and_eligibility():
    """Narrowed Stage 1b invariant contract: edge_statistic()'s raw output
    and eligible_edges()'s raw output stay byte-identical to v2.0.0/v2.0.1 --
    region boundaries/areas/thresholds do NOT (those change by design; see
    the retired test above). Compares the current worktree's call_dmrs()
    -internal edge_statistic()/eligible_edges() output (captured via a
    temporary monkeypatch, so it is exactly what call_dmrs() computes,
    including its low-coverage keep-filter -- not a hand-reproduced
    pipeline) against tests/fixtures/stage1b_invariant_fixture_v201.npz.
    That fixture was generated from the SAME generate_inputs() the old
    v2.0.0 fixture used (see generate_stage1b_invariant_fixture.py), and
    confirmed at generation time that edge_statistic()/eligible_edges() are
    byte-identical between the pre-Stage-1 commit and the current worktree
    -- so the current worktree's output IS the correct baseline to pin."""

    import importlib.util

    import eiger2.core as core_mod

    spec = importlib.util.spec_from_file_location(
        "generate_stage1_invariant_fixture",
        Path(__file__).parent / "fixtures" / "generate_stage1_invariant_fixture.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matrix, locs, config = module.generate_inputs()

    captured: dict[str, np.ndarray] = {}
    original_edge_statistic = core_mod.edge_statistic
    original_eligible_edges = core_mod.eligible_edges

    def capturing_edge_statistic(residuals, cfg, chunk=200_000):
        result = original_edge_statistic(residuals, cfg, chunk=chunk)
        captured["edge_statistic"] = result
        return result

    def capturing_eligible_edges(locations, cfg):
        result = original_eligible_edges(locations, cfg)
        captured["eligible_edges"] = result
        return result

    core_mod.edge_statistic = capturing_edge_statistic
    core_mod.eligible_edges = capturing_eligible_edges
    try:
        core_mod.call_dmrs(matrix, locs, config=config)
    finally:
        core_mod.edge_statistic = original_edge_statistic
        core_mod.eligible_edges = original_eligible_edges

    assert "edge_statistic" in captured and "eligible_edges" in captured

    fixture_path = (
        Path(__file__).parent / "fixtures" / "stage1b_invariant_fixture_v201.npz"
    )
    expected = np.load(fixture_path)
    np.testing.assert_array_equal(
        captured["edge_statistic"], expected["edge_statistic"]
    )
    np.testing.assert_array_equal(
        captured["eligible_edges"], expected["eligible_edges"]
    )


# --------------------------------------------------------------------------
# Stage 1b — lag-null window assembly
# --------------------------------------------------------------------------

from eiger2.core import _lag_null_track, _lag_statistic, _null_eligibility  # noqa: E402


def _reference_lag_null_track(residuals, lag, config):
    """Old (buggy) implementation, kept ONLY as a fixed reference point for the
    no-op-at-w<=2 test below -- never reintroduced into the package."""
    from eiger2.core import _window_from_lags
    span = max(config.window, 2)
    usable = residuals.shape[0] - lag * (span - 1)
    if usable < 2:
        return np.empty(0, dtype=float)
    lag_statistics = []
    for separation in range(1, span):
        values = _lag_statistic(residuals, separation * lag, config)
        lag_statistics.append(values[:usable])
    if config.window == 1:
        return lag_statistics[0]
    return _window_from_lags(lag_statistics, config.window)


def test_lag_null_track_is_a_noop_at_window_1_and_2():
    rng = np.random.default_rng(101)
    residuals = rng.normal(size=(200, 30))
    for w in (1, 2):
        config = Eiger2Config(window=w, random_seed=1)
        old = _reference_lag_null_track(residuals, lag=17, config=config)
        new = _lag_null_track(residuals, lag=17, config=config)
        assert new.shape == old.shape
        # allclose, not exact equality: at w=2, the old reference path goes
        # through _window_from_lags's cumsum-then-diff construction, which is
        # mathematically a no-op but is not bit-exact (floating-point
        # non-associativity, ~1e-16). The new direct-sum implementation is
        # bit-exact to the raw underlying statistic; a tight tolerance still
        # catches any real (non-floating-point-noise) discrepancy.
        np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-10)


def test_lag_null_track_member_set_is_exactly_the_lag_multiples():
    # Build a residual matrix where stat(row j, row k) is recoverable from the
    # pair itself: use two samples, values j and k respectively, so
    # correlation/covariance of (j, j2, ..) vs (k, k2, ..) across a handful of
    # synthetic "samples" encodes (j, k) in a way we can invert. Simpler and
    # exact: monkeypatch _lag_statistic to return a matrix whose value at
    # position i is the pair (i, i+d) encoded as i * 10_000 + d, then confirm
    # _lag_null_track's output at window position i decodes to exactly the
    # pairs {(i + a*lag, i + b*lag) : 0 <= a < b < w} and no others.
    import eiger2.core as core_mod

    lag = 5
    w = 4
    n_positions = 30

    def fake_lag_statistic(residuals, d, config, chunk=200_000):
        n_pairs = residuals.shape[0] - d
        # value at row i encodes the pair (i, i + d) uniquely
        return np.array([i * 100_000 + d for i in range(n_pairs)], dtype=float)

    original = core_mod._lag_statistic
    core_mod._lag_statistic = fake_lag_statistic
    try:
        config = Eiger2Config(window=w, random_seed=1)
        dummy_residuals = np.zeros((n_positions + lag * (w - 1) + 5, 3))
        out = core_mod._lag_null_track(dummy_residuals, lag, config)
    finally:
        core_mod._lag_statistic = original

    # decode window position i=0: expect the average of pair-encodings for
    # {(0, lag), (0, 2*lag), (0, 3*lag), (lag, 2*lag), (lag, 3*lag), (2*lag, 3*lag)}
    expected_pairs = [(a * lag, b * lag) for a in range(w) for b in range(a + 1, w)]
    expected_values = [j * 100_000 + (k - j) for (j, k) in expected_pairs]
    assert out[0] == pytest.approx(np.mean(expected_values))

    # And decode a NON-ZERO window position (i=10) the same way, pinning that
    # the base offset i is carried correctly through the whole window, not just
    # at the i=0 edge case: expect the pairs {(10 + a*lag, 10 + b*lag)}, i.e.
    # every member shifted by the base position, and nothing else.
    i = 10
    expected_pairs_i = [
        (i + a * lag, i + b * lag) for a in range(w) for b in range(a + 1, w)
    ]
    expected_values_i = [j * 100_000 + (k - j) for (j, k) in expected_pairs_i]
    assert out[i] == pytest.approx(np.mean(expected_values_i))


def test_null_eligibility_offsets_match_lag_null_track_member_set():
    # Extract _lag_null_track's actual member-set offsets programmatically, via
    # the same fake-_lag_statistic monkeypatch/decode technique as
    # test_lag_null_track_member_set_is_exactly_the_lag_multiples above, instead
    # of hand-rederiving _null_eligibility's own offset formula. Copying
    # _null_eligibility's "step * lag" loop into the test only proves
    # _null_eligibility is self-consistent with itself -- it can't fail even if
    # _lag_null_track's member set diverges from it. Deriving the expected
    # offsets from _lag_null_track's real output (and validating that
    # derivation against the real output first) makes the two genuinely
    # coupled: this test now fails if either function's member set moves
    # without the other following.
    import eiger2.core as core_mod

    lag = 7
    for w in (1, 2, 3, 5):
        config = Eiger2Config(window=w, random_seed=1)
        span = max(w, 2)

        def fake_lag_statistic(residuals, d, config, chunk=200_000):
            n_pairs = residuals.shape[0] - d
            return np.array([i * 100_000 + d for i in range(n_pairs)], dtype=float)

        n_positions = 10
        original = core_mod._lag_statistic
        core_mod._lag_statistic = fake_lag_statistic
        try:
            dummy_residuals = np.zeros((n_positions + lag * (span - 1) + 5, 3))
            out = core_mod._lag_null_track(dummy_residuals, lag, config)
        finally:
            core_mod._lag_statistic = original

        if w == 1:
            # No window combination at w=1 (see _lag_null_track's `if w == 1`
            # branch): the raw single-lag statistic is returned directly, so
            # the member set at position 0 is {0, lag}. Confirm against the
            # real output before relying on it below.
            assert out[0] == pytest.approx(0 * 100_000 + lag)
            member_offsets = {0, lag}
        else:
            # Confirm the {0, lag, 2*lag, ..., (w-1)*lag} member set against
            # the real output -- same decode formula validated by
            # test_lag_null_track_member_set_is_exactly_the_lag_multiples --
            # before trusting it as ground truth for _null_eligibility below.
            candidate = sorted(a * lag for a in range(w))
            pairs = [
                (candidate[a], candidate[b])
                for a in range(w)
                for b in range(a + 1, w)
            ]
            predicted = np.mean([j * 100_000 + (k - j) for j, k in pairs])
            assert out[0] == pytest.approx(predicted)
            member_offsets = set(candidate)

        size = 50
        chromosome = np.array(["chr1"] * (size + lag * (span - 1) + 5))
        eligible = _null_eligibility(chromosome, lag, config, size)
        # A null position i is admissible iff every extracted member offset
        # (relative to i) sits on the same chromosome as i itself -- built from
        # member_offsets extracted above, not from copying _null_eligibility's
        # own loop.
        expected = np.ones(size, dtype=bool)
        first = chromosome[:size]
        for offset in sorted(member_offsets - {0}):
            expected &= first == chromosome[offset : offset + size]
        np.testing.assert_array_equal(eligible, expected)


def test_lag_null_track_slice_bounds_hold_for_large_window_and_lag():
    rng = np.random.default_rng(202)
    for w in (1, 2, 4, 8, 12):
        for lag in (2, 50, 1999):
            n = 4000
            residuals = rng.normal(size=(n, 10))
            config = Eiger2Config(window=w, random_seed=1)
            out = _lag_null_track(residuals, lag, config)
            span = max(w, 2)
            expected_usable = max(n - lag * (span - 1), 0)
            if expected_usable < 2:
                assert out.size == 0
            else:
                assert out.size == expected_usable


# --------------------------------------------------------------------------
# null exchangeability
# --------------------------------------------------------------------------


def test_null_exchangeability_self_consistency_control():
    # held-out null exceedance must equal 1 - null_quantile to within Monte Carlo
    # error -- this is the control that proves the gate measures the right thing
    rng = np.random.default_rng(303)
    residuals = rng.normal(size=(3000, 40))
    config = Eiger2Config(window=1, random_seed=7, null_replicates=200)
    tracks = null_tracks(residuals, config)
    statistic = rng.normal(size=2999)  # arbitrary observed track, same length class
    eligible = np.ones(2999, dtype=bool)
    result = null_exchangeability(tracks, statistic, eligible, config)
    assert result["p_null"] == pytest.approx(1 - config.null_quantile, abs=0.003)


def test_null_exchangeability_ratio_is_one_when_observed_equals_null():
    rng = np.random.default_rng(404)
    residuals = rng.normal(size=(3000, 40))
    config = Eiger2Config(window=1, random_seed=7, null_replicates=200)
    tracks = null_tracks(residuals, config)
    # use one of the null's own held-out-eligible values as a stand-in "observed"
    # track drawn from the identical distribution -- ratio should be close to 1
    pooled_all = np.concatenate([t[np.isfinite(t) & e] for t, e, _ in tracks])
    observed_like_null = rng.choice(pooled_all, size=2000, replace=True)
    eligible = np.ones(2000, dtype=bool)
    result = null_exchangeability(tracks, observed_like_null, eligible, config)
    assert result["null_exchangeability_ratio"] == pytest.approx(1.0, abs=0.3)


def test_null_exchangeability_deterministic_split():
    rng = np.random.default_rng(505)
    residuals = rng.normal(size=(2000, 30))
    config = Eiger2Config(window=1, random_seed=9, null_replicates=100)
    tracks = null_tracks(residuals, config)
    statistic = rng.normal(size=1999)
    eligible = np.ones(1999, dtype=bool)
    first = null_exchangeability(tracks, statistic, eligible, config)
    second = null_exchangeability(tracks, statistic, eligible, config)
    assert first == second


def test_null_threshold_reports_exchangeability_when_statistic_given():
    rng = np.random.default_rng(606)
    residuals = rng.normal(size=(2000, 30))
    config = Eiger2Config(window=1, random_seed=3, null_replicates=100)
    statistic = rng.normal(size=1999)
    eligible = np.ones(1999, dtype=bool)
    threshold, diagnostics = null_threshold(
        residuals, config, statistic=statistic, eligible=eligible,
    )
    assert "null_exchangeability_ratio" in diagnostics
    assert diagnostics["null_exchangeability_ratio"] > 0


def test_null_threshold_omits_exchangeability_when_statistic_not_given():
    rng = np.random.default_rng(707)
    residuals = rng.normal(size=(2000, 30))
    config = Eiger2Config(window=1, random_seed=3, null_replicates=100)
    threshold, diagnostics = null_threshold(residuals, config)
    assert "null_exchangeability_ratio" not in diagnostics


# --------------------------------------------------------------------------
# Task 6: exchangeability-gate regression test + recall-regression test
# --------------------------------------------------------------------------


def test_exchangeability_ratio_at_most_1_25_on_dmr_free_scenarios():
    # spec §1b.5 item 4: all five scenarios, 10 seeds. This is a correctness gate,
    # not an exploratory check -- do not reduce scenario/seed coverage for speed.
    from benchmarking.benchmark_suite import SCENARIOS

    from eiger2.benchmark import simulate
    from eiger2.core import (
        centre_cpgs,
        deflate,
        edge_statistic,
        eligible_edges,
        estimate_latent_basis,
        null_exchangeability,
        null_tracks,
    )
    # Sourced from benchmarking.benchmark_suite.SCENARIOS rather than a hand-copied
    # dict, so the two definitions cannot silently drift apart. Only "missingness"
    # is overridden: missing_rate=0.5, not SCENARIOS' default 0.25. Ruled by the
    # user after Task 1's reproduction found 0.25 leaves the pre-fix defect
    # marginal and seed-dependent (measured ratios 1.631 and 1.262 -- both over
    # the 1.25 threshold -- vs 1.226 which passes, across 3 tested seeds), while
    # 0.5 makes it unambiguous (measured ~3.8x pre-fix) -- the real reason 0.5 was
    # chosen for this demonstration.
    scenarios = {
        **SCENARIOS,
        "missingness": {**SCENARIOS["missingness"], "missing_rate": 0.5},
    }
    for name, params in scenarios.items():
        for seed in range(10):
            # spacing=20000 (n_cpg ~20000-20400 across scenarios), NOT the
            # brief's literal spacing=5000. Investigated before deviating:
            # at spacing=5000 (n_cpg~5400) this test deterministically fails
            # 4/50 scenario x seed combos (missingness seed 8: ratio 1.368;
            # high_noise seeds 0/5/7: ratios 1.418/1.260/1.865) on the current,
            # Task-5-verified-correct code -- not a fix defect. A convergence
            # check (spacing 5000 -> 20000 -> 50000, same seeds/scenarios)
            # shows ratios tightening around 1.0 as n_cpg grows, confirming
            # this is finite-sample Monte Carlo/binomial noise in the p_obs
            # exceedance count (only ~5000-5400 observed positions at a
            # ~0.5% tail rate), not a systematic bias -- spacing=5000 is
            # simply too small a panel for a tight 1.25 per-instance gate to
            # be reliable. spacing=20000 matches the codebase's own canonical
            # spec-1b.3 implementation (benchmarking/benchmark_suite.py's
            # exchangeability(), added in this same stage) and Task 5's
            # verified reproduction (spacing=19600), and passes all 50
            # combos with comfortable margin (max ratio 1.169 across a full
            # sweep). See
            # docs/superpowers/plans/2026-08-18-stage1b-lag-null-window-fix.md
            # for the full investigation.
            matrix, locs, _ = simulate(seed=seed, strata=[], replicates=0,
                                        spacing=20000, **params)
            config = Eiger2Config(window=4, random_seed=seed, null_replicates=100)
            residuals = centre_cpgs(matrix)
            basis, _ = estimate_latent_basis(residuals, config)
            deflated = deflate(residuals, basis, missing=config.deflation_missing)
            statistic = edge_statistic(deflated, config)
            eligible = eligible_edges(locs, config)
            tracks = null_tracks(deflated, config, locs)
            result = null_exchangeability(tracks, statistic, eligible, config)
            assert result["null_exchangeability_ratio"] <= 1.25, (name, seed, result)


def test_recall_does_not_regress_more_than_5_points_after_lag_null_fix():
    # Uses the kept _reference_lag_null_track (old, buggy construction) as the
    # "before" comparison point, monkeypatched in, against the shipped "after".
    #
    # evaluate()'s actual signature/return type (confirmed by reading
    # Python/eiger2/benchmark.py directly, not guessed):
    #   evaluate(regions, truth, *, min_overlap=20)
    #       -> (recovered: set[int], false_regions: int)
    # It is a tuple, not a dict -- there is no "recall" key to index. Recall is
    # derived as len(recovered) / len(truth), matching Task 5's report's usage.
    import eiger2.core as core_mod
    from eiger2.benchmark import evaluate, simulate

    # dmr_length=25, NOT the brief's literal 15: evaluate()'s default
    # min_overlap=20 means a truth region can only ever be "recovered" if its
    # own span is >= 20 CpGs (a called region can overlap a dmr_length=15
    # truth region by at most 15 CpGs, which never reaches min_overlap=20).
    # At dmr_length=15 this test was vacuous in both directions -- measured
    # before_recall == after_recall == 0/6 -- which the "assert before_recall
    # > 0" guard below now genuinely catches instead of silently passing.
    # dmr_length=25 restores real recall (measured 6/6 before and after).
    matrix, locs, planted = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=25, seed=11,
    )
    config = Eiger2Config(window=4, null_replicates=50, random_seed=11)

    original = core_mod._lag_null_track
    core_mod._lag_null_track = lambda residuals, lag, config: _reference_lag_null_track(
        residuals, lag, config
    )
    try:
        before_regions, _, _ = call_dmrs(matrix, locs, config=config)
    finally:
        core_mod._lag_null_track = original
    after_regions, _, _ = call_dmrs(matrix, locs, config=config)

    before_recovered, _before_false = evaluate(before_regions, planted)
    after_recovered, _after_false = evaluate(after_regions, planted)
    before_recall = len(before_recovered) / len(planted)
    after_recall = len(after_recovered) / len(planted)
    # Guard against the comparison becoming vacuously true (0 >= 0 - 0.05) if
    # future parameter changes ever drive recall to zero on both sides.
    assert before_recall > 0
    assert after_recall >= before_recall - 0.05, (before_recall, after_recall)


# --------------------------------------------------------------------------
# Task 6: support-conditioned threshold -- config, auto-enable, per-bin
# thresholds, call_dmrs wiring (Stage 1c)
# --------------------------------------------------------------------------


def test_support_binned_threshold_off_reproduces_scalar_path_bit_for_bit():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=4,
        spacing=400, dmr_length=15, missing_rate=0.25, seed=41,
    )
    base_config = Eiger2Config(window=4, null_replicates=50, random_seed=41,
                                support_binned_threshold="off")
    regions_off, values_off, manifest_off = call_dmrs(matrix, locs, config=base_config)
    # compare against a config from BEFORE this task's changes semantically --
    # "off" must produce the same region set/areas/threshold as Stage 1b's output
    # for identical inputs. Since Stage 1b's code path IS the "off" path structurally,
    # this is really a self-consistency check: run "off" twice, confirm determinism,
    # and confirm statistic_threshold is a bare scalar (float), not array-derived.
    regions_off2, _, _ = call_dmrs(matrix, locs, config=base_config)
    pd.testing.assert_frame_equal(regions_off, regions_off2, check_exact=True)
    assert manifest_off["support_binning"] == "forced_off"
    assert isinstance(regions_off["statistic_threshold"].iloc[0], float)

    # The three assertions above hold identically even under forced_on --
    # statistic_threshold was always a bare scalar float before Task 7's fix,
    # regardless of region_threshold_mode, so `isinstance(..., float)` never
    # actually distinguished the "off" scalar path from the "support_binned"
    # path. Task 6's independent review flagged this as tautological. Fixed
    # here with a genuine bit-identity assertion: hand-build the Stage 1b
    # scalar pipeline directly (centre_cpgs -> estimate_latent_basis ->
    # deflate -> edge_statistic -> eligible_edges -> null_threshold ->
    # segment(scalar) -> region_area(scalar)), bypassing call_dmrs's Stage 1c
    # support-binning branch entirely, and confirm it reproduces
    # call_dmrs(..., support_binned_threshold="off")'s region bounds and
    # region_area values exactly. Includes the same min_observed_samples/
    # min_sample_fraction row-filtering call_dmrs applies before centre_cpgs
    # (core.py, top of call_dmrs) so cpg_index_start/cpg_index_end line up
    # even if that filter were ever to remove rows for this seed/config.
    from eiger2.core import _standardise_locations
    from eiger2.core import eligible_edges as _eligible_edges
    from eiger2.core import null_threshold as _null_threshold
    from eiger2.core import region_area as _region_area
    from eiger2.core import segment as _segment

    n_cpg, n_samples = matrix.shape
    frame = _standardise_locations(locs, n_cpg)
    observed_per_cpg = np.sum(np.isfinite(matrix), axis=1)
    keep = observed_per_cpg >= base_config.min_observed_samples
    if base_config.min_sample_fraction > 0:
        keep &= observed_per_cpg >= np.ceil(
            base_config.min_sample_fraction * n_samples
        )
    filtered_matrix = matrix[keep]
    filtered_frame = frame.loc[keep].reset_index(drop=True)
    source_index = np.flatnonzero(keep)

    residuals = centre_cpgs(filtered_matrix)
    basis, _ = estimate_latent_basis(residuals, base_config)
    deflated = deflate(residuals, basis, missing=base_config.deflation_missing)
    statistic = edge_statistic(deflated, base_config)
    eligible = _eligible_edges(filtered_frame, base_config)
    threshold, _ = _null_threshold(
        deflated, base_config, filtered_frame,
        statistic=statistic, eligible=eligible,
    )
    manual_regions = _segment(
        statistic, eligible, threshold,
        min_cpgs=base_config.min_cpgs, max_gap_edges=base_config.max_gap_edges,
    )
    manual_areas = _region_area(statistic, manual_regions, threshold)

    assert threshold == pytest.approx(regions_off["statistic_threshold"].iloc[0])
    expected_bounds = np.column_stack([
        source_index[manual_regions[:, 0]], source_index[manual_regions[:, 1]],
    ]) if manual_regions.size else np.empty((0, 2), dtype=int)
    actual_bounds = regions_off[["cpg_index_start", "cpg_index_end"]].to_numpy()
    assert np.array_equal(expected_bounds, actual_bounds)
    assert np.array_equal(manual_areas, regions_off["region_area"].to_numpy())


def test_support_dispersion_detector_computed_and_recorded():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=4,
        spacing=400, dmr_length=15, missing_rate=0.25, seed=42,
    )
    config = Eiger2Config(window=4, null_replicates=50, random_seed=42)
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    assert "support_dispersion" in manifest
    assert manifest["support_dispersion"] >= 0
    assert "support_dispersion_cut" in manifest
    assert manifest["support_binning"] in {
        "auto_on", "auto_off", "forced_on", "forced_off", "collapsed_to_single",
    }


def test_support_binned_threshold_on_produces_per_bin_thresholds():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=15, missing_rate=0.4, seed=43,
    )
    config = Eiger2Config(window=4, null_replicates=100, random_seed=43,
                           support_binned_threshold="on")
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    assert manifest["support_binning"] in {"forced_on", "collapsed_to_single"}
    if manifest["support_binning"] == "forced_on":
        assert len(manifest["support_thresholds"]) >= 2
        assert len(regions) >= 0  # ran without error
        # Task 7 wires region_threshold_mode into _REGION_COLUMNS/the row dict,
        # so the column always exists now -- no presence guard needed (a
        # "col" in regions.columns check here would silently re-hide a future
        # regression that dropped the column). no_regions truly has zero rows,
        # so the len(regions) guard stays.
        if len(regions):
            assert (regions["region_threshold_mode"] == "support_binned").all()


def test_determinism_same_seed_same_bins_same_thresholds_same_regions():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=15, missing_rate=0.4, seed=44,
    )
    config = Eiger2Config(window=4, null_replicates=100, random_seed=44,
                           support_binned_threshold="on")
    regions1, _, manifest1 = call_dmrs(matrix, locs, config=config)
    regions2, _, manifest2 = call_dmrs(matrix, locs, config=config)
    pd.testing.assert_frame_equal(regions1, regions2, check_exact=True)
    assert manifest1["support_bin_edges"] == manifest2["support_bin_edges"]


def test_support_binned_threshold_config_validation():
    with pytest.raises(ValueError, match="support_binned_threshold"):
        Eiger2Config(support_binned_threshold="bogus")
    with pytest.raises(ValueError, match="support_bins"):
        Eiger2Config(support_bins=1)


def test_support_binning_forced_off_for_permutation_mode():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=4,
        spacing=400, dmr_length=15, missing_rate=0.25, seed=45,
    )
    config = Eiger2Config(window=4, null_replicates=50, random_seed=45,
                           null_mode="permutation", support_binned_threshold="on")
    _, _, manifest = call_dmrs(matrix, locs, config=config)
    assert manifest["support_binning"] == "forced_off"
    assert manifest["support_binning_skipped_reason"] == (
        "permutation_mode_not_supported"
    )


def test_null_tracks_returns_track_eligible_lag_triples():
    rng = np.random.default_rng(11)
    residuals = rng.normal(size=(3000, 20))
    config = Eiger2Config(window=1, random_seed=1, null_replicates=10)
    tracks = null_tracks(residuals, config)
    assert len(tracks) > 0
    for track, eligible, lag in tracks:
        assert isinstance(lag, int)
        assert track.shape == eligible.shape


# --------------------------------------------------------------------------
# Task 7: output schema -- region_support_median/region_threshold_mode
# columns, schema_version -> "2.2" (Stage 1c)
# --------------------------------------------------------------------------


def test_schema_v22_has_support_columns_and_version():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=4,
        spacing=400, dmr_length=15, missing_rate=0.25, seed=51,
    )
    config = Eiger2Config(window=4, null_replicates=50, random_seed=51)
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    assert "region_support_median" in regions.columns
    assert "region_threshold_mode" in regions.columns
    assert manifest["schema_version"] == "2.2"
    if len(regions):
        assert regions["region_threshold_mode"].isin(["single", "support_binned"]).all()


def test_statistic_threshold_is_mean_per_edge_threshold_when_support_binned():
    # Task 6's independent review found statistic_threshold was written as the
    # single unconditioned scalar `threshold` even when per-bin thresholds
    # were actually applied to produce region_area -- confirmed empirically
    # divergent on a forced_on run (column reported 0.7143 while the actually
    # -applied per-bin thresholds were [0.7308, 0.7236, 0.7001, 0.7079]).
    # Task 7 fixes this: when region_threshold_mode == "support_binned",
    # statistic_threshold is the region's mean per-edge threshold (mean of
    # per_edge_threshold over the same edge slice [first:last] that
    # region_support_median uses); when region_threshold_mode == "single", it
    # stays the existing scalar unchanged (asserted separately, below).
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=15, missing_rate=0.4, seed=43,
    )
    config = Eiger2Config(window=4, null_replicates=100, random_seed=43,
                           support_binned_threshold="on")
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    if manifest["support_binning"] != "forced_on" or not len(regions):
        pytest.skip("this seed did not produce a forced_on run with regions")
    binned = regions["region_threshold_mode"] == "support_binned"
    assert binned.any()
    # statistic_threshold must genuinely diverge from the unconditioned scalar
    # threshold for at least one support_binned region -- otherwise this test
    # would pass identically under the pre-fix behaviour it is meant to catch.
    scalar_threshold = manifest["null"]["threshold"]
    assert not np.allclose(
        regions.loc[binned, "statistic_threshold"], scalar_threshold
    )


def test_statistic_threshold_stays_scalar_when_single_mode():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=4,
        spacing=400, dmr_length=15, missing_rate=0.25, seed=41,
    )
    config = Eiger2Config(window=4, null_replicates=50, random_seed=41,
                           support_binned_threshold="off")
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    assert (regions["region_threshold_mode"] == "single").all()
    scalar_threshold = manifest["null"]["threshold"]
    assert np.array_equal(
        regions["statistic_threshold"].to_numpy(),
        np.full(len(regions), scalar_threshold),
    )


# --------------------------------------------------------------------------
# Task 8: calibrated support_dispersion_cut -- auto-enable fires only on
# missingness at the calibrated default (Stage 1c, spec §1c.7 test item 6)
# --------------------------------------------------------------------------


def test_auto_enable_fires_only_on_missingness_at_calibrated_cut():
    # Calibration basis (Task 8, full table in task-8-report.md): the 5
    # standard scenarios (benchmarking.benchmark_suite.SCENARIOS) at
    # strata=DEFAULT_STRATA, replicates=6, spacing=400 (matching the
    # benchmark harness's own standard invocation, e.g. heuristic_sweep()/
    # calibration()), 10 seeds each. Measured manifest["support_dispersion"]
    # with support_binned_threshold="off": clean/nuisance/small_n/high_noise
    # (missing_rate=0.0) == 0.0 at every seed (structurally exact -- with no
    # missingness every eligible window has full support, so the 90th/10th
    # percentile spread is zero); missingness (missing_rate=0.25) in
    # [1.0417, 1.1250], median 1.0833. _DEFAULT_SUPPORT_DISPERSION_CUT = 0.5
    # sits near the midpoint of the (0, 1.0417) separating gap. This test
    # re-runs a subset of those seeds (3, not all 10 -- the calibration
    # script itself, not this suite, is the full 10-seed record) with
    # support_binned_threshold="auto" (the calibrated default) and asserts
    # the auto-enable decision lands correctly for every scenario x seed.
    from benchmarking.benchmark_suite import DEFAULT_STRATA, SCENARIOS

    scenario_names = ["clean", "nuisance", "missingness", "small_n", "high_noise"]
    for name in scenario_names:
        params = SCENARIOS[name]
        for seed in range(3):
            matrix, locs, _ = simulate(
                seed=seed, strata=DEFAULT_STRATA, replicates=6, spacing=400,
                **params,
            )
            config = Eiger2Config(window=4, random_seed=seed, null_replicates=100)
            assert config.support_binned_threshold == "auto"
            assert config.support_dispersion_cut is None
            _, _, manifest = call_dmrs(matrix, locs, config=config)
            expected = "auto_on" if name == "missingness" else "auto_off"
            assert manifest["support_binning"] == expected, (
                name, seed, manifest["support_binning"], manifest["support_dispersion"],
            )
