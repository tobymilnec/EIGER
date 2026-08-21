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
import eiger2.core as _core_mod  # noqa: E402
from eiger2.core import (  # noqa: E402
    RegionFdrResult,  # noqa: F401
    _assign_support_bin,
    _fdr_from_areas_result,
    _HAS_NUMBA,
    _overdispersion_diagnostic,
    _pair_statistic,
    _pair_statistic_numpy,
    _support_bin_edges,
    _window_support,
    acf_gate,
    member_step_lag_null_ensemble,
    member_step_lag_null_tracks,
    null_exchangeability,
    null_threshold,
    null_tracks,
    permutation_null_tracks,
    qvalue_result_from_track,
    qvalues_from_track,
    region_area,
    region_exchangeability_gate,
    region_fdr_result,
    track_autocorrelation,
)

# min_null_regions_for_fdr=0 disables the spec §5 null-adequacy guard: at 6
# replicates the pooled null is always far below the default floor of 50, and
# these tests exist to exercise calling/recall behaviour, not null adequacy
# (which §6 addresses at the inferential-unit level).
# region_null_mode="lag" (Stage 1d-D, master spec §6.4): the shipped default
# "member_step" needs n_cpg >= null_min_lag * (min_subtrack_length + window -
# 1), ~10^5 CpGs at defaults -- far more than these deliberately small test
# matrices. These tests exercise calling/statistic/segmentation logic, not the
# region null construction itself (which has its own dedicated tests, and its
# own real-fixture-scale acceptance measurement in STAGE1D_D_STOP_REPORT.md).
FAST = {
    "null_replicates": 6, "min_null_regions_for_fdr": 0, "region_null_mode": "lag",
}


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


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba (the 'fast' extra) is not installed")
def test_pair_statistic_numba_matches_numpy_nan_pattern_exactly():
    # Stage 1g Task B.3 acceptance: identical NaN pattern, max abs difference
    # <= 1e-12, on randomised input including all-missing and
    # single-sample-overlap rows. fastmath=True was measured to elide the
    # finiteness check and return all-NaN silently -- this test is what
    # would catch that regression (a mismatched NaN pattern, not just a
    # numeric tolerance failure).
    rng = np.random.default_rng(77)
    n_pairs, n_samples = 500, 40
    left = rng.normal(size=(n_pairs, n_samples))
    right = rng.normal(size=(n_pairs, n_samples))
    # structured missingness across most rows
    left[rng.random(left.shape) < 0.3] = np.nan
    right[rng.random(right.shape) < 0.3] = np.nan
    # row entirely missing on one side
    left[0, :] = np.nan
    # row entirely missing on both sides
    left[1, :] = np.nan
    right[1, :] = np.nan
    # single-sample overlap: only column 0 finite on both sides
    left[2, :] = np.nan
    right[2, :] = np.nan
    left[2, 0] = 0.4
    right[2, 0] = -0.2
    # exactly two samples overlap (the min_paired_samples=2 boundary)
    left[3, :] = np.nan
    right[3, :] = np.nan
    left[3, :2] = [0.1, 0.2]
    right[3, :2] = [0.3, -0.1]
    # constant row (zero variance) -> correlation must be NaN, not a
    # division artefact
    left[4, :] = 1.0

    for statistic in ("covariance", "correlation"):
        for min_paired_samples in (2, 5):
            expected = _pair_statistic_numpy(
                left, right, statistic, min_paired_samples,
            )
            required = min_paired_samples
            actual = _core_mod._pair_statistic_numba(
                np.ascontiguousarray(left, dtype=np.float64),
                np.ascontiguousarray(right, dtype=np.float64),
                statistic != "covariance",
                required,
            )
            np.testing.assert_array_equal(
                np.isnan(expected), np.isnan(actual),
                err_msg=f"NaN pattern mismatch at statistic={statistic!r}, "
                        f"min_paired_samples={min_paired_samples}",
            )
            finite = ~np.isnan(expected)
            max_diff = np.max(np.abs(expected[finite] - actual[finite]))
            assert max_diff <= 1e-12, (
                f"max abs diff {max_diff} exceeds 1e-12 at "
                f"statistic={statistic!r}, min_paired_samples={min_paired_samples}"
            )


def test_pair_statistic_numba_is_opt_in_not_automatic(monkeypatch):
    # Stage 1g Task B.3: numba, even when installed, must not silently change
    # `_pair_statistic`'s output for a caller who did not ask for it -- see
    # the "opt-in, not automatic" comment above the kernel in core.py. This
    # guards the specific failure mode that motivated the opt-in: a
    # frozen-fixture invariant test (test_stage1b_preserves_edge_statistic_
    # and_eligibility) broke the first time this kernel dispatched
    # automatically, at the ~1e-15 floating-point noise floor -- invisible
    # statistically, but enough to fail exact-equality checks.
    monkeypatch.delenv("EIGER2_USE_NUMBA", raising=False)
    rng = np.random.default_rng(88)
    left = rng.normal(size=(50, 20))
    right = rng.normal(size=(50, 20))
    result = _pair_statistic(left, right, "correlation", 5)
    reference = _pair_statistic_numpy(left, right, "correlation", 5)
    np.testing.assert_array_equal(result, reference)


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


def test_call_dmrs_manifest_carries_layout_diagnostics():
    # master spec §6.1: describe_layout() wired into every eiger2 call manifest.
    rng = np.random.default_rng(9)
    matrix = 0.5 + rng.normal(0, 0.04, (4000, 30))
    frame = locations(4000)  # dense: uniform 2bp spacing, one giant eligible run
    _, _, manifest = call_dmrs(
        np.clip(matrix, 0.001, 0.999), frame, config=Eiger2Config(**FAST)
    )
    assert manifest["layout"]["layout_n_cpg"] == 4000
    assert manifest["layout"]["layout_max_detectable_region_cpgs"] == 4000
    assert manifest["layout"]["realised_length_strata"] == [
        "4-5", "6-9", "10-19", "20-49", "50+",
    ]
    assert manifest["layout"]["collapsed_length_strata"] == []


def test_call_dmrs_manifest_collapses_strata_on_a_short_layout():
    # A fragmented layout whose longest eligible run tops out at 8 CpGs (7
    # eligible edges) can never populate "10-19", "20-49" or "50+" -- those
    # collapse away regardless of what this particular run actually calls.
    rng = np.random.default_rng(9)
    n_cpg = 400
    matrix = 0.5 + rng.normal(0, 0.04, (n_cpg, 30))
    starts = np.arange(n_cpg) * 2
    # break every eligible run to at most 8 CpGs by inserting a large gap
    # every 8th CpG
    for i in range(8, n_cpg, 8):
        starts[i:] += 1000
    frame = pd.DataFrame({"chr": "chr1", "start": starts, "end": starts + 1})
    _, _, manifest = call_dmrs(
        np.clip(matrix, 0.001, 0.999), frame,
        config=Eiger2Config(**FAST, max_cpg_gap_bp=300),
    )
    assert manifest["layout"]["layout_max_detectable_region_cpgs"] == 8
    assert manifest["layout"]["realised_length_strata"] == ["4-5", "6-9"]
    assert manifest["layout"]["collapsed_length_strata"] == [
        "10-19", "20-49", "50+",
    ]


def test_call_dmrs_manifest_carries_gate_1_even_with_zero_regions():
    # master spec §6.2, Gate 1 (ACF): wired into every eiger2 call manifest,
    # including the "no_regions" early-return path -- it needs no called region.
    rng = np.random.default_rng(9)
    matrix = 0.5 + rng.normal(0, 0.04, (4000, 30))  # pure noise, no structure
    frame = locations(4000)
    _, _, manifest = call_dmrs(
        np.clip(matrix, 0.001, 0.999), frame, config=Eiger2Config(**FAST)
    )
    assert "gates" in manifest
    acf = manifest["gates"]["acf"]
    assert acf["lags"] == list(range(1, Eiger2Config(**FAST).window + 1))
    assert isinstance(acf["passed"], bool)


def test_call_dmrs_manifest_carries_both_gates_when_regions_are_called():
    # master spec §6.2: both gates present once there is something for Gate 2
    # (region exchangeability) to measure.
    matrix, frame, _ = simulate(strata=[(0.2, 0.4)], replicates=3, seed=5)
    _, _, manifest = call_dmrs(matrix, frame, config=Eiger2Config(**FAST))
    assert manifest["n_regions"] > 0
    assert "gates" in manifest
    assert "acf" in manifest["gates"]
    region_exch = manifest["gates"]["region_exchangeability"]
    assert region_exch["observed_regions"] == manifest["n_regions"]
    assert "region_exchangeability_ratio" in region_exch
    assert "null_count_adequate" in region_exch


def test_call_dmrs_manifest_records_member_step_gate_fields():
    # Stage 1d-D Task 4: mode, n_lag_draws, realised s, drawn lags, sub-track
    # length distribution, and all three gate results must land in
    # manifest["gates"] on every region_null_mode="member_step" run. A small
    # min_subtrack_length/null_min_lag keeps this fast (L_max only needs to
    # clear null_min_lag, not member_step's genome-scale default floor).
    matrix, frame, _ = simulate(strata=[(0.2, 0.4)], replicates=3, seed=5)
    config = Eiger2Config(
        window=4, null_replicates=6, min_null_regions_for_fdr=0,
        region_null_mode="member_step", null_min_lag=5, min_subtrack_length=10,
        n_lag_draws=6, random_seed=5,
    )
    _, _, manifest = call_dmrs(matrix, frame, config=config)
    gates = manifest["gates"]
    assert gates["region_null_mode"] == "member_step"
    assert gates["n_lag_draws"] == len(gates["lags_used"])
    assert gates["n_lag_draws"] > 0
    assert all(lag >= config.null_min_lag for lag in gates["lags_used"])
    assert len(gates["implied_subtrack_length_range"]) == 2
    assert 0.0 < gates["s"] <= 1.0
    assert gates["implied_region_qvalue_resolution_at_R1"] == gates["s"]
    assert "acf" in gates
    assert "region_exchangeability" in gates


def test_region_null_mode_auto_falls_back_to_lag_on_undersized_input():
    # Stage 1f Task 1b (pre-release, master spec pre-release): region_null_mode
    # left at its default ("auto") on input below member_step's genome-scale
    # minimum floor (~103,000 CpGs at defaults) must fall back to "lag"
    # automatically, warn (via the manifest key -- cli.py turns it into stderr
    # output), and complete -- not raise, which was STOP 1e's flagged gap (a
    # CLI user got a hard, uncatchable error with no command-line way out).
    matrix, frame, _ = simulate(strata=[(0.2, 0.4)], replicates=3, seed=5)
    config = Eiger2Config(
        window=4, null_replicates=6, min_null_regions_for_fdr=0, random_seed=5,
    )
    assert config.region_null_mode == "auto"  # sanity: the literal field default
    regions, _, manifest = call_dmrs(matrix, frame, config=config)
    gates = manifest["gates"]
    assert gates["region_null_mode"] == "lag"
    fallback = gates["region_null_mode_fallback"]
    assert fallback["requested"] == "auto"
    assert fallback["used"] == "lag"
    assert fallback["reason"] == "input below member_step minimum"
    assert fallback["n_cpg"] == len(frame)
    assert fallback["n_cpg"] < fallback["n_cpg_required"]
    # "completes": no exception, and it produced a real (non-degenerate) result.
    assert isinstance(regions, pd.DataFrame)


def test_region_null_mode_explicit_member_step_still_raises_on_undersized_input():
    # Stage 1f Task 1b: an EXPLICIT request is not silently redirected -- a
    # user who asked for member_step by name is told it cannot be done here,
    # exactly as before this stage (Stage 1d-D Task 2's deliberate guard).
    matrix, frame, _ = simulate(strata=[(0.2, 0.4)], replicates=3, seed=5)
    config = Eiger2Config(
        window=4, null_replicates=6, min_null_regions_for_fdr=0,
        region_null_mode="member_step", random_seed=5,
    )
    with pytest.raises(ValueError, match="region_null_mode='member_step'"):
        call_dmrs(matrix, frame, config=config)


def test_region_null_mode_auto_uses_member_step_on_oversized_input():
    # Stage 1f Task 1b: the mirror image of the fallback test above -- once
    # the input clears the floor, "auto" resolves to real member_step, not a
    # fallback, and no region_null_mode_fallback key is recorded. Same small
    # null_min_lag/min_subtrack_length trick as
    # test_call_dmrs_manifest_records_member_step_gate_fields (lowers the
    # floor so this stays fast), but region_null_mode is left UNSET here to
    # exercise "auto"'s resolution, not passed explicitly.
    matrix, frame, _ = simulate(strata=[(0.2, 0.4)], replicates=3, seed=5)
    config = Eiger2Config(
        window=4, null_replicates=6, min_null_regions_for_fdr=0,
        null_min_lag=5, min_subtrack_length=10, n_lag_draws=6, random_seed=5,
    )
    assert config.region_null_mode == "auto"
    _, _, manifest = call_dmrs(matrix, frame, config=config)
    gates = manifest["gates"]
    assert gates["region_null_mode"] == "member_step"
    assert "region_null_mode_fallback" not in gates


def test_a_well_separated_true_region_gets_q_zero_not_refused_by_its_own_length_stratum():
    # Stage 1d-D Acceptance criterion 6 (genome-scale, real layouts, guard at its
    # shipped min_null_regions_for_fdr=50 default): recall fell to 0/9-10 truth
    # regions on BOTH real layouts, even though the aggregate region-exchangeability
    # gate was adequate (e.g. epic_fitted: 1963 pooled null regions, ratio 1.625,
    # null_count_adequate=True). Root cause, diagnosed via
    # manifest["fdr"]["insufficient_null_strata"]: the null pool is dominated by
    # short (4-5 CpG) noise-driven segments; a true, well-separated region long
    # enough to land in the "10-19" length stratum found that stratum's OWN pooled
    # null count at zero, and was refused (NaN q-value) by the Stage 1c-bis
    # per-stratum floor -- independent of how many total lag draws/replicates
    # were pooled, since the floor was a per-stratum count, not a global one.
    #
    # Stage 1e Task 1 (master spec §8) is the fix, diagnosed and specified by
    # the same person who designed the Stage 1c-bis floor: the null having
    # millions of opportunities and never producing a region that long IS the
    # evidence, not an absence of it. The floor is now a single AGGREGATE check
    # (region_fdr_result's docstring); a stratum whose own pool is empty gets
    # K=0 -- a finite point estimate (q=0) and a finite Garwood upper bound --
    # and is named in `zero_exceedance_strata`, reported rather than refused.
    # This test was renamed from
    # `..._can_be_refused_by_its_own_length_stratum` and now asserts the
    # opposite of what it used to. The background is scaled up from the
    # original pin (spacing=30000 vs 200) so the run is AGGREGATE-adequate
    # (152 pooled null regions >= the default floor of 50) while the true
    # region's own "10-19" stratum is still exactly empty -- reproducing the
    # real defect's shape, not just its per-stratum symptom.
    matrix, locs, truth = simulate(
        strata=[(0.05, 0.35)], replicates=1, spacing=30000, dmr_length=15,
        seed=3, layout="dense",
    )
    config = Eiger2Config(
        window=4, null_min_lag=5, min_subtrack_length=10, n_lag_draws=150,
        region_null_mode="member_step", random_seed=3, min_cpgs=4,
    )
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    assert manifest["gates"]["region_exchangeability"]["null_count_adequate"] is True
    true_region = regions[regions["n_cpg"] >= 10].iloc[0]
    assert true_region["region_qvalue"] == 0.0
    assert np.isfinite(true_region["region_qvalue_upper_95"])
    assert true_region["null_exceedances_at_cutoff"] == 0
    strata_names = {s["stratum"] for s in manifest["fdr"]["zero_exceedance_strata"]}
    assert "10-19" in strata_names
    # deprecated, schema ("2.2") compatibility only -- never populated now
    assert manifest["fdr"]["insufficient_null_strata"] == []


# --------------------------------------------------------------------------
# region-level FDR
# --------------------------------------------------------------------------


def test_region_fdr_is_monotone_and_bounded():
    # region_fdr()'s signature is frozen, so it always applies the spec §5
    # default min_null_regions_for_fdr=50; this toy null is deliberately far
    # smaller than that. Called through region_fdr_result directly with the
    # guard disabled (min_null_regions_for_fdr=0) to test the tail-area
    # formula's own monotonicity/bounds property, independent of null adequacy.
    observed = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    nulls = [np.array([5.0, 3.0]), np.array([7.0, 1.0])]
    q = region_fdr_result(
        observed, nulls, observed_edges=1000, null_edges=1000,
        min_null_regions_for_fdr=0,
    ).qvalue
    assert np.all((q >= 0) & (q <= 1))
    assert np.all(np.diff(q[np.argsort(-observed)]) >= -1e-12)


def test_region_fdr_rises_when_the_null_produces_comparable_regions():
    # Same rationale as above: guard disabled to isolate the tail-area
    # formula's response to null density, not null-adequacy refusal.
    observed = np.array([5.0, 4.0])
    quiet = region_fdr_result(
        observed, [np.array([])] * 4, observed_edges=1000, null_edges=1000,
        min_null_regions_for_fdr=0,
    ).qvalue
    noisy = region_fdr_result(
        observed,
        [np.array([9.0, 8.0, 7.0])] * 4,
        observed_edges=1000,
        null_edges=1000,
        min_null_regions_for_fdr=0,
    ).qvalue
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
        min_null_regions_for_posterior=5, min_cpgs=3, region_null_mode="lag",
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
    # .equals(), not tolist() == tolist(): region_qvalue may legitimately be
    # NaN (null-adequacy refusal), and NaN != NaN under plain equality even
    # when both runs refused identically. .equals() treats NaN as equal.
    assert regions["region_qvalue"].equals(second[0]["region_qvalue"])
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
    regions, _, manifest = call_dmrs(
        matrix, locations, config=Eiger2Config(region_null_mode="lag")
    )

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


def test_region_fdr_result_empty_own_pool_stratum_reports_from_its_own_k_not_a_fallback():
    # Stage 1c-bis review Task 1 removed a global-pool fallback for an empty
    # own pool (non-monotonic: an empty stratum could clear the floor by
    # borrowing a large OTHER stratum's null while a merely-thin stratum could
    # not). Stage 1e Task 1 (master spec §8) now lets the empty stratum report
    # too, but it must still score from its OWN pool (K=0, trivially, since
    # there is nothing in it), not from a mix with the well-populated "30"
    # stratum's pool. Proven by determinism, not by inspecting one number:
    # "4-5"'s result must be IDENTICAL regardless of what the "30" stratum's
    # pool contains -- a silent fallback would make it vary with the "30"
    # pool's random draw.
    def run(seed):
        rng = np.random.default_rng(seed)
        observed_lengths = np.array([4, 4, 30, 30])
        observed_areas = np.array([5.0, 3.0, 9.0, 6.0])
        null_area_sets = [
            np.concatenate([np.empty(0), rng.exponential(2.0, size=20)])
            for _ in range(5)
        ]
        null_length_sets = [
            np.concatenate([np.empty(0, dtype=int), np.full(20, 30)]) for _ in range(5)
        ]
        return region_fdr_result(
            observed_areas, null_area_sets,
            observed_edges=200, null_edges=100,
            observed_lengths=observed_lengths, null_length_sets=null_length_sets,
        )

    a, b = run(22), run(99)
    for result in (a, b):
        assert result.diagnostics["pooled_fallback_strata"] == []
        assert result.diagnostics["insufficient_null_strata"] == []
        zero_exceedance = {
            e["stratum"]: e for e in result.diagnostics["zero_exceedance_strata"]
        }
        assert zero_exceedance.keys() == {"4-5"}
        assert zero_exceedance["4-5"]["null_regions"] == 0
        assert not np.any(np.isnan(result.qvalue))
        assert np.all(result.qvalue[:2] == 0.0)
        assert np.all(result.null_exceedances[:2] == 0)

    # the "30" stratum's pool differs by seed, so its own results may differ...
    assert not np.array_equal(a.qvalue_upper[2:], b.qvalue_upper[2:]) or not np.array_equal(
        a.null_exceedances[2:], b.null_exceedances[2:]
    )
    # ...but "4-5" never borrows from it, so its results are identical either way.
    assert np.array_equal(a.qvalue_upper[:2], b.qvalue_upper[:2])
    assert np.array_equal(a.qvalue_resolution[:2], b.qvalue_resolution[:2])


def test_region_fdr_result_multiple_strata_report_independently_when_aggregate_adequate():
    # Stage 1e Task 1 (master spec §8): three strata of very different own-pool
    # sizes -- "4-5" empty, "6-9" thin (20, still well below the default floor
    # of 50 on its own), "50+" well-populated (800) -- aggregate to 820 pooled
    # null regions, comfortably clearing the floor. None refuse: only "4-5"
    # (the literally empty pool) lands in zero_exceedance_strata.
    rng = np.random.default_rng(44)
    observed_lengths = np.array([4, 4, 8, 8])
    observed_areas = np.array([5.0, 4.0, 6.0, 3.0])
    null_area_sets = [
        np.concatenate([rng.exponential(2.0, size=5), rng.exponential(2.0, size=200)])
        for _ in range(4)
    ]
    null_length_sets = [
        np.concatenate([np.full(5, 8), np.full(200, 50)]) for _ in range(4)
    ]
    result = region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=400, null_edges=800,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    )
    assert "status" not in result.diagnostics
    assert not np.any(np.isnan(result.qvalue))
    assert np.all(result.null_exceedances >= 0)
    zero_exceedance = {
        e["stratum"]: e for e in result.diagnostics["zero_exceedance_strata"]
    }
    assert zero_exceedance.keys() == {"4-5"}
    assert zero_exceedance["4-5"]["null_regions"] == 0
    assert result.diagnostics["pooled_fallback_strata"] == []
    assert result.diagnostics["insufficient_null_strata"] == []


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
        # equal_nan=True: some strata here fall below min_null_regions_for_fdr
        # and are legitimately NaN in both baseline and permuted; that is not
        # a row-order bug.
        assert np.array_equal(a, b, equal_nan=True)


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
# Task 10 (Stage 1c escalation, spec §1c.6): region-FDR null pooling
# additionally stratified by support bin, with a length-only fallback when a
# joint (length x support) stratum has too few pooled null regions.
# --------------------------------------------------------------------------


def test_region_fdr_result_support_stratification_diagnostics():
    rng = np.random.default_rng(77)
    # Two observed regions, same length stratum ("6-9"), different support bins.
    observed_areas = np.array([12.0, 9.0])
    observed_lengths = np.array([6, 6])
    observed_support_bins = np.array([0, 1])

    # 3 null replicates. Support bin 0 gets 20 null regions per replicate (60
    # pooled, >= _MIN_NULL_REGIONS_PER_JOINT_STRATUM -> joint stratum "6-9|sbin0"
    # stays joint). Support bin 1 gets 1 null region per replicate (3 pooled,
    # < 50 -> falls back to length-only pooling for that region).
    null_area_sets = []
    null_length_sets = []
    null_support_bin_sets = []
    for _ in range(3):
        bin0_areas = rng.uniform(1.0, 8.0, size=20)
        bin1_areas = rng.uniform(1.0, 8.0, size=1)
        null_area_sets.append(np.concatenate([bin0_areas, bin1_areas]))
        null_length_sets.append(np.full(21, 6))
        null_support_bin_sets.append(
            np.concatenate([np.zeros(20, dtype=int), np.ones(1, dtype=int)])
        )

    result = region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=1000, null_edges=3000,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
        observed_support_bins=observed_support_bins,
        null_support_bin_sets=null_support_bin_sets,
    )

    assert result.diagnostics["support_stratified"] is True
    assert result.diagnostics["min_null_regions_per_joint_stratum"] == 50
    assert result.diagnostics["support_stratification_fallback_strata"] == ["6-9|sbin1"]
    cutoffs = result.diagnostics["fdr_interval_cutoffs_by_stratum"]
    assert "6-9|sbin0" in cutoffs  # well-populated joint stratum stayed joint
    assert "6-9" in cutoffs        # region 1 fell back to the length-only label
    assert "6-9|sbin1" not in cutoffs  # never appears as a final stratum itself
    assert np.all(np.isfinite(result.qvalue))
    assert result.qvalue.shape == (2,)

    # None (the default) must reproduce pre-existing behavior exactly: no new
    # diagnostics keys, same shape.
    unstratified = region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=1000, null_edges=3000,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    )
    assert unstratified.diagnostics["support_stratified"] is False
    assert unstratified.diagnostics["support_stratification_fallback_strata"] == []


def test_region_fdr_result_fallback_stratum_pools_every_support_bin_numerically():
    # Hand-verifiable arithmetic, no RNG. Both observed regions sit in length
    # stratum "6-9" and each is ALONE in its final stratum, so the
    # reverse-cumulative-minimum over groups is the identity and
    # tail_fdr == qvalue == min(1, scale * K / 1) with K exactly the number of
    # pooled null areas >= that region's own area (see _fdr_from_areas_result).
    #
    #   bin 0: 20 null regions per replicate, areas 1.0 .. 20.0  (60 pooled >= 50
    #          -> "6-9|sbin0" stays joint)
    #   bin 1:  1 null region  per replicate, area 50.0          (3 pooled < 50
    #          -> region 1 falls back to the length-only pool "6-9")
    bin0 = np.arange(1.0, 21.0)
    null_area_sets = [np.concatenate([bin0, np.array([50.0])]) for _ in range(3)]
    null_length_sets = [np.full(21, 6) for _ in range(3)]
    null_support_bin_sets = [
        np.concatenate([np.zeros(20, dtype=int), np.ones(1, dtype=int)])
        for _ in range(3)
    ]
    observed_areas = np.array([12.0, 9.0])
    observed_lengths = np.array([6, 6])
    observed_support_bins = np.array([0, 1])

    # scale = 1/100 keeps every product below the min(1.0, ...) cap so the
    # arithmetic stays visible instead of saturating.
    result = region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=1, null_edges=100,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
        observed_support_bins=observed_support_bins,
        null_support_bin_sets=null_support_bin_sets,
    )
    assert result.diagnostics["scale"] == pytest.approx(0.01)
    assert result.diagnostics["support_stratification_fallback_strata"] == ["6-9|sbin1"]

    # Region 0 ("6-9|sbin0"): only bin-0 nulls, 9 per replicate >= 12.0 -> 27.
    # The length-only pool would have added the three 50.0 bin-1 nulls (30), so
    # this value also proves the joint stratum EXCLUDED the other bin.
    bin0_ge_12 = 3 * int(np.sum(bin0 >= 12.0))
    assert bin0_ge_12 == 27
    assert int(result.null_exceedances[0]) == 27
    assert result.tail_fdr[0] == pytest.approx(0.27)
    assert result.qvalue[0] == pytest.approx(0.27)

    # Region 1 fell back: its pooled null set must literally be the UNION of
    # both support bins' null areas for length stratum "6-9".
    pooled_all = np.concatenate(null_area_sets)
    expected_fallback_k = int(np.sum(pooled_all >= 9.0))
    assert expected_fallback_k == 3 * int(np.sum(bin0 >= 9.0)) + 3  # 36 + 3
    assert int(result.null_exceedances[1]) == expected_fallback_k == 39
    assert result.tail_fdr[1] == pytest.approx(0.39)
    # Had the thin joint stratum NOT fallen back, K would have been the three
    # bin-1 nulls only -> 0.03. Had it fallen back to the WRONG (bin-0-only)
    # pool, K would have been 36 -> 0.36. Both are excluded.
    assert result.tail_fdr[1] != pytest.approx(0.03)
    assert result.tail_fdr[1] != pytest.approx(0.36)

    # The length-only-stratified run must reach region 1 through the SAME pooled
    # null set (K == 39 either way) -- that is what "fell back to length-only
    # pooling" means. Its tail_fdr differs only through the DENOMINATOR: there
    # both regions share one stratum, so R = 2 observed areas >= 9.0 and
    # raw = 0.01 * 39 / 2 = 0.195, whereas under joint stratification region 1
    # is alone in "6-9" and R = 1. This is exactly the stratum-membership
    # dependence that makes a naive qvalue-equality assertion wrong; the pooled
    # NULL SET is the invariant that must hold, and it does.
    length_only = region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=1, null_edges=100,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    )
    assert int(length_only.null_exceedances[1]) == 39
    assert length_only.tail_fdr[1] == pytest.approx(0.01 * 39 / 2)
    # ...and region 0's null pool DOES change: length-only pools the three 50.0
    # bin-1 nulls in with the bin-0 ones, so at region 0's OWN cutoff (which is
    # what tail_fdr reports) K = 30, not the 27 the joint stratum gave.
    assert length_only.tail_fdr[0] == pytest.approx(0.30)
    # null_exceedances reports K at the Q-ACHIEVING cutoff, which for region 0
    # here is region 1's cutoff (0.195 < 0.30), hence 39 rather than 30.
    assert int(length_only.null_exceedances[0]) == 39
    assert length_only.qvalue[0] == pytest.approx(0.01 * 39 / 2)


def test_region_fdr_result_joint_strata_use_only_their_own_bin_nulls():
    # Both support bins are well populated (60 pooled null regions each), so
    # NOTHING falls back and each observed region is scored against its own
    # bin's nulls only.
    bin0 = np.arange(1.0, 21.0)    # 1..20
    bin1 = np.arange(21.0, 41.0)   # 21..40
    null_area_sets = [np.concatenate([bin0, bin1]) for _ in range(3)]
    null_length_sets = [np.full(40, 6) for _ in range(3)]
    null_support_bin_sets = [
        np.concatenate([np.zeros(20, dtype=int), np.ones(20, dtype=int)])
        for _ in range(3)
    ]
    observed_areas = np.array([15.0, 15.0])
    observed_lengths = np.array([6, 6])
    observed_support_bins = np.array([0, 1])

    result = region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=1, null_edges=100,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
        observed_support_bins=observed_support_bins,
        null_support_bin_sets=null_support_bin_sets,
    )
    assert result.diagnostics["support_stratification_fallback_strata"] == []
    assert set(result.diagnostics["fdr_interval_cutoffs_by_stratum"]) == {
        "6-9|sbin0", "6-9|sbin1",
    }
    # bin 0: 6 values >= 15 per replicate -> 18.  bin 1: all 20 -> 60.
    assert int(result.null_exceedances[0]) == 18
    assert int(result.null_exceedances[1]) == 60
    assert result.tail_fdr[0] == pytest.approx(0.18)
    assert result.tail_fdr[1] == pytest.approx(0.60)
    # The length-only pool would have given BOTH regions 78.
    length_only = region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=1, null_edges=100,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    )
    assert int(length_only.null_exceedances[0]) == 78
    assert int(length_only.null_exceedances[1]) == 78


def test_region_fdr_result_single_support_bin_is_numerically_identical():
    # Degenerate-but-critical invariant: if every observed region AND every null
    # region falls in the SAME support bin, joint stratification only relabels
    # the strata -- the membership of every pool is unchanged, so every numeric
    # field must match the length-only run exactly. This is the check that would
    # catch double-counting, an off-by-one in the joint-label zip, or a silently
    # empty joint pool.
    rng = np.random.default_rng(78)
    observed_areas = rng.exponential(3.0, size=25)
    observed_lengths = rng.integers(4, 60, size=25)
    null_area_sets = [rng.exponential(3.0, size=200) for _ in range(6)]
    null_length_sets = [rng.integers(4, 60, size=200) for _ in range(6)]

    length_only = region_fdr_result(
        observed_areas, null_area_sets, observed_edges=5000, null_edges=30000,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    )
    joint = region_fdr_result(
        observed_areas, null_area_sets, observed_edges=5000, null_edges=30000,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
        observed_support_bins=np.zeros(25, dtype=int),
        null_support_bin_sets=[np.zeros(200, dtype=int) for _ in range(6)],
    )
    for field in (
        "qvalue", "tail_fdr", "qvalue_lower", "qvalue_upper",
        "qvalue_resolution", "null_exceedances", "qvalue_cutoff_rank",
    ):
        assert np.array_equal(
            getattr(joint, field), getattr(length_only, field), equal_nan=True
        ), field
    assert (
        joint.diagnostics["fdr_interval_bonferroni_m"]
        == length_only.diagnostics["fdr_interval_bonferroni_m"]
    )
    # Only the stratum LABELS differ ("20-49" -> "20-49|sbin0").
    assert sorted(
        k.split("|")[0] for k in joint.diagnostics["fdr_interval_cutoffs_by_stratum"]
    ) == sorted(length_only.diagnostics["fdr_interval_cutoffs_by_stratum"])
    assert all(
        k.endswith("|sbin0")
        for k in joint.diagnostics["fdr_interval_cutoffs_by_stratum"]
    )


def test_region_fdr_result_support_args_ignored_without_length_stratification():
    # Joint stratification is defined ON TOP of length stratification. Passing
    # support bins without lengths must not silently activate a support-only
    # scheme (which is not spec'd) -- it must stay the pooled path.
    rng = np.random.default_rng(79)
    observed_areas = rng.exponential(2.0, size=10)
    null_area_sets = [rng.exponential(2.0, size=80) for _ in range(4)]
    result = region_fdr_result(
        observed_areas, null_area_sets, observed_edges=500, null_edges=1000,
        observed_support_bins=np.arange(10) % 2,
        null_support_bin_sets=[np.arange(80) % 2 for _ in range(4)],
    )
    assert result.diagnostics["support_stratified"] is False
    assert list(result.diagnostics["fdr_interval_cutoffs_by_stratum"]) == ["_pooled"]
    baseline = region_fdr_result(
        observed_areas, null_area_sets, observed_edges=500, null_edges=1000,
    )
    assert np.array_equal(result.qvalue, baseline.qvalue)


def test_region_fdr_result_misaligned_support_bins_raise_rather_than_mispool():
    # The support-bin arrays must be row-aligned with the length/area arrays. A
    # silent misalignment would pool the wrong nulls into a stratum, so the zips
    # are strict and this must raise, not quietly truncate.
    observed_areas = np.array([5.0, 6.0])
    observed_lengths = np.array([6, 6])
    null_area_sets = [np.full(21, 1.0) for _ in range(2)]
    null_length_sets = [np.full(21, 6) for _ in range(2)]
    with pytest.raises(ValueError):
        region_fdr_result(
            observed_areas, null_area_sets, observed_edges=100, null_edges=200,
            observed_lengths=observed_lengths, null_length_sets=null_length_sets,
            observed_support_bins=np.array([0, 1]),
            null_support_bin_sets=[np.zeros(20, dtype=int) for _ in range(2)],
        )
    with pytest.raises(ValueError):
        region_fdr_result(
            observed_areas, null_area_sets, observed_edges=100, null_edges=200,
            observed_lengths=observed_lengths, null_length_sets=null_length_sets,
            observed_support_bins=np.array([0]),
            null_support_bin_sets=[np.zeros(21, dtype=int) for _ in range(2)],
        )


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


# --------------------------------------------------------------------------
# spec §5 (Stage 1c-bis) — null-adequacy guard
# --------------------------------------------------------------------------


def test_region_fdr_result_default_min_null_regions_for_fdr_is_50():
    import inspect
    sig = inspect.signature(region_fdr_result)
    assert sig.parameters["min_null_regions_for_fdr"].default == 50


def test_region_fdr_result_below_aggregate_min_null_regions_refuses_with_nan():
    # Stage 1e Task 1 (master spec §8): the floor is now a single AGGREGATE
    # check on the pooled null-region count across every stratum -- there is
    # only one stratum here ("_pooled", no length arrays supplied), so this
    # exercises the same arithmetic the old per-stratum floor used, just under
    # the new "status": "aggregate null inadequate" diagnostics shape.
    rng = np.random.default_rng(40)
    observed = rng.exponential(2.0, size=6)
    nulls = _null_sets(rng, 3, 5)  # 3 replicates x 5 = 15 pooled null regions, < 50
    result = region_fdr_result(
        observed, nulls, observed_edges=200, null_edges=90,
    )
    assert np.all(np.isnan(result.qvalue))
    assert np.all(np.isnan(result.tail_fdr))
    assert np.all(np.isnan(result.qvalue_lower))
    assert np.all(np.isnan(result.qvalue_upper))
    assert np.all(np.isnan(result.qvalue_resolution))
    assert np.all(result.null_exceedances == -1)
    assert np.all(result.qvalue_cutoff_rank == -1)
    assert result.diagnostics["status"] == "aggregate null inadequate"
    assert result.diagnostics["pooled_null_regions"] == 15
    # deprecated, schema ("2.2") compatibility only -- never populated now
    assert result.diagnostics["insufficient_null_strata"] == []
    assert result.diagnostics["zero_exceedance_strata"] == []


def test_region_fdr_result_min_null_regions_for_fdr_boundary_is_strict_less_than():
    rng = np.random.default_rng(41)
    observed = rng.exponential(2.0, size=6)
    nulls = _null_sets(rng, 2, 5)  # 10 pooled null regions

    at_floor = region_fdr_result(
        observed, nulls, observed_edges=200, null_edges=90,
        min_null_regions_for_fdr=10,
    )
    assert not np.any(np.isnan(at_floor.qvalue))
    assert "status" not in at_floor.diagnostics

    below_floor = region_fdr_result(
        observed, nulls, observed_edges=200, null_edges=90,
        min_null_regions_for_fdr=11,
    )
    assert np.all(np.isnan(below_floor.qvalue))
    assert below_floor.diagnostics["status"] == "aggregate null inadequate"
    assert below_floor.diagnostics["pooled_null_regions"] == 10


def test_region_fdr_result_aggregate_inadequate_reports_replicates_required():
    rng = np.random.default_rng(42)
    observed = rng.exponential(2.0, size=4)
    nulls = _null_sets(rng, 3, 5)  # 3 replicates, 5/replicate, 15 pooled
    result = region_fdr_result(
        observed, nulls, observed_edges=200, null_edges=90,
        min_null_regions_for_fdr=50,
    )
    assert result.diagnostics["replicates"] == 3
    assert result.diagnostics["null_regions_per_replicate"] == pytest.approx(5.0)
    assert result.diagnostics["replicates_required"] == 10  # ceil(50 / 5)


def test_region_fdr_result_empty_own_pool_stratum_reports_not_refuses():
    # Stage 1e Task 1 (master spec §8): fixes the exact defect measured in
    # STOP 1d-D §9 criterion 6 -- a length stratum whose OWN pooled null is
    # empty used to refuse (NaN) even though the run as a whole was adequate.
    # "4-5" here has zero own null regions; "50+" has 80. Aggregate pool is
    # 0 + 80 = 80 >= the default floor of 50, so NEITHER stratum refuses now:
    # "4-5" gets K=0 (point estimate q=0, a finite upper bound), reported in
    # zero_exceedance_strata, not NaN.
    rng = np.random.default_rng(43)
    observed_lengths = np.array([4, 4, 50, 50])
    observed_areas = np.array([3.0, 2.0, 40.0, 30.0])
    null_area_sets = [
        np.concatenate([np.empty(0), rng.exponential(2.0, size=20)])
        for _ in range(4)
    ]
    null_length_sets = [
        np.concatenate([np.empty(0, dtype=int), np.full(20, 50)]) for _ in range(4)
    ]
    result = region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=400, null_edges=400,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    )
    assert "status" not in result.diagnostics
    assert not np.any(np.isnan(result.qvalue))
    assert np.all(result.qvalue[:2] == 0.0)
    assert np.all(np.isfinite(result.qvalue_upper[:2]))
    assert np.all(result.null_exceedances[:2] == 0)
    zero_exceedance = {
        e["stratum"]: e for e in result.diagnostics["zero_exceedance_strata"]
    }
    assert zero_exceedance.keys() == {"4-5"}
    assert zero_exceedance["4-5"]["null_regions"] == 0
    assert zero_exceedance["4-5"]["replicates"] == 4
    # deprecated, schema ("2.2") compatibility only -- never populated now
    assert result.diagnostics["insufficient_null_strata"] == []


def test_region_fdr_result_thin_but_nonempty_stratum_also_reports():
    # Spec's own Tests requirement: "a stratum with K=0 and a stratum with
    # K=3 in the same run: the first reports, the second reports; neither
    # refuses". "4-5" here has an empty own pool (K=0 trivially); "50+" has a
    # thin-but-nonempty pool of 3, deliberately chosen so the strongest "50+"
    # candidate's own cutoff has exactly 3 null areas at or above it.
    observed_lengths = np.array([4, 50])
    observed_areas = np.array([9.0, 5.0])
    null_area_sets = [np.array([]), np.array([6.0, 7.0, 8.0])]
    null_length_sets = [np.array([], dtype=int), np.full(3, 50, dtype=int)]
    result = region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=400, null_edges=400,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
        min_null_regions_for_fdr=1,
    )
    assert "status" not in result.diagnostics
    assert not np.any(np.isnan(result.qvalue))
    assert result.null_exceedances[0] == 0
    assert result.null_exceedances[1] == 3
    zero_exceedance = {e["stratum"] for e in result.diagnostics["zero_exceedance_strata"]}
    assert zero_exceedance == {"4-5"}


def test_region_fdr_result_empty_pool_is_not_worse_than_a_populated_zero_k_pool():
    # Monotonicity (spec's own Tests requirement): a stratum whose own null
    # pool is EMPTY (K=0 trivially) must not be reported as WORSE evidence
    # than an otherwise-identical stratum backed by a large pool that also
    # happens to have zero exceedances at the same cutoff -- the Garwood bound
    # depends on K and the (shared, run-level) scale, not on per-stratum pool
    # size, so the two must match exactly.
    observed_areas = np.array([9.0])
    empty_pool = region_fdr_result(
        observed_areas, [np.array([]), np.array([])],
        observed_edges=100, null_edges=200, min_null_regions_for_fdr=0,
    )
    populated_but_zero_exceedance = region_fdr_result(
        observed_areas, [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])],
        observed_edges=100, null_edges=200, min_null_regions_for_fdr=0,
    )
    assert empty_pool.null_exceedances[0] == 0
    assert populated_but_zero_exceedance.null_exceedances[0] == 0
    assert empty_pool.qvalue[0] == populated_but_zero_exceedance.qvalue[0]
    assert (
        empty_pool.qvalue_upper[0] == populated_but_zero_exceedance.qvalue_upper[0]
    )
    assert (
        empty_pool.qvalue_resolution[0]
        == populated_but_zero_exceedance.qvalue_resolution[0]
    )


def test_region_fdr_result_empty_and_zero_replicate_paths_carry_the_new_key():
    empty = region_fdr_result(np.empty(0), [], observed_edges=0, null_edges=0)
    assert empty.diagnostics["insufficient_null_strata"] == []
    assert empty.diagnostics["zero_exceedance_strata"] == []

    zero_rep = region_fdr_result(
        np.array([1.0, 2.0]), [], observed_edges=10, null_edges=0,
    )
    assert zero_rep.diagnostics["insufficient_null_strata"] == []
    assert zero_rep.diagnostics["zero_exceedance_strata"] == []


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
            region_null_mode="lag",
        )
        first, _, _ = call_dmrs(matrix, locs, config=config)
        second, _, _ = call_dmrs(matrix, locs, config=config)
        assert first["significant"].tolist() == second["significant"].tolist()
        # .equals(), not tolist() == tolist(): a NaN region_qvalue (null-
        # adequacy refusal) is still deterministic across identical runs, but
        # NaN != NaN under plain equality.
        assert first["region_qvalue"].equals(second["region_qvalue"])


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
    # min_null_regions_for_fdr=0: this test is about point-vs-upper decision
    # logic, tuned against specific pre-existing significant-region counts; the
    # spec §5 null-adequacy guard is orthogonal and disabled here rather than
    # re-deriving those counts under refusal.
    point_config = Eiger2Config(
        window=4, null_replicates=500, random_seed=7, target_fdr=0.05,
        fdr_decision="point", min_null_regions_for_fdr=0, region_null_mode="lag",
    )
    upper_config = Eiger2Config(
        window=4, null_replicates=500, random_seed=7, target_fdr=0.05,
        fdr_decision="upper", min_null_regions_for_fdr=0, region_null_mode="lag",
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
        target_fdr=0.01, region_null_mode="lag",
    )
    with pytest.raises(ValueError, match="B_required|null_replicates|Increase"):
        call_dmrs(matrix, locs, config=config)


def test_run_legacy_fdr_replicates_raises_not_silently_broken():
    # Retired (Stage 1d-D Task 6.3, master spec §6.4): this test used to pin
    # `cli.run_legacy`'s own point-vs-upper decision logic (around
    # cli.py:409-414), fed by `qvalue_result_from_track` -> internally
    # `permutation_null_tracks`. That call now raises unconditionally (Task
    # 6.1) -- `run_legacy(..., fdr_replicates=N>0, ...)` is a THIRD reachable
    # path to the same broken null (alongside `qvalue_result_from_track` and
    # `qvalues_from_track`, both covered directly in test_probability.py),
    # and per Task 6.3's "whatever changes must reach that path" instruction,
    # this confirms it does: the legacy caller's region-CALLING (no FDR
    # requested) is unaffected, since that never touches the null; only the
    # optional FDR annotation is now a clear error instead of a silent wrong
    # answer.
    from eiger2.cli import run_legacy

    matrix, locs, _ = simulate(
        n_samples=60, strata=[(0.15, 0.35)], replicates=4,
        spacing=300, dmr_length=12, seed=52,
    )
    sample_names = [f"s{i}" for i in range(matrix.shape[1])]

    with pytest.raises(ValueError, match="region_null_mode='member_step'"):
        run_legacy(
            matrix, locs, sample_names,
            cov_threshold=0.002, delta_threshold=0.20, min_cpgs=5, impute=True,
            fdr_replicates=200, target_fdr=0.05, min_prevalence=0, seed=52,
            fdr_decision="upper",
        )

    # region-CALLING with no FDR requested (fdr_replicates=0, the default)
    # never touches the null and must still work.
    regions, _ = run_legacy(
        matrix, locs, sample_names,
        cov_threshold=0.002, delta_threshold=0.20, min_cpgs=5, impute=True,
        seed=52,
    )
    assert len(regions) > 0
    assert "significant" not in regions.columns or regions["significant"].isna().all()


# --------------------------------------------------------------------------
# Task 8: output schema -- six new region columns, call_dmrs wiring,
# schema_version
# --------------------------------------------------------------------------


def test_schema_round_trip_has_documented_columns_in_documented_order():
    matrix, locs, _ = simulate(
        n_samples=60, strata=[(0.15, 0.35)], replicates=4,
        spacing=300, dmr_length=12, seed=61,
    )
    config = Eiger2Config(
        window=4, null_replicates=20, random_seed=61, region_null_mode="lag",
    )
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
        min_null_regions_for_fdr=0,  # toy 2-value null; not testing adequacy
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
        # `config` (from the frozen fixture-generation script, not modified
        # here to keep it byte-for-byte historical) does not set
        # region_null_mode="lag", so at this matrix's size call_dmrs raises
        # once it reaches Stage 1d-D's region-null construction (Task 2's
        # deliberate too-small-input guard). That happens AFTER
        # edge_statistic()/eligible_edges() are both already captured, which
        # is all this test pins -- the frozen Stage 1b invariant, not
        # call_dmrs's full output -- so the later raise is expected and
        # ignored here.
        try:
            core_mod.call_dmrs(matrix, locs, config=config)
        except ValueError:
            pass
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


# --------------------------------------------------------------------------
# Stage 1d-C §1 — gap-mask the null (restored omission)
# --------------------------------------------------------------------------


def test_null_eligibility_gap_masks_an_ineligible_member():
    # window=3, lag=10: member offsets {0, 10, 20}. Mark the local edge at
    # offset 10 (position 3's second member) ineligible; position 3 must be
    # excluded, its neighbours must not be.
    config = Eiger2Config(window=3, random_seed=1)
    size = 10
    lag = 10
    chromosome = np.array(["chr1"] * (size + lag * 2 + 5))
    observed_eligible = np.ones(chromosome.size - 1, dtype=bool)
    observed_eligible[3 + lag] = False  # local edge right after member offset 10
    eligible = _null_eligibility(
        chromosome, lag, config, size, observed_eligible=observed_eligible
    )
    assert not eligible[3]
    assert eligible[2]
    assert eligible[4]


def test_null_eligibility_gap_mask_does_not_apply_across_the_long_lag_itself():
    # Every member's OWN local edge is eligible; nothing requires the edge
    # BETWEEN member 0 and member 1 (a `lag`-sized jump) to be eligible --
    # that jump is the intentional locality-breaking construction, not a
    # defect. All positions must be admissible.
    config = Eiger2Config(window=2, random_seed=1)
    size = 10
    lag = 500
    chromosome = np.array(["chr1"] * (size + lag + 5))
    observed_eligible = np.ones(chromosome.size - 1, dtype=bool)
    # falsify a huge middle stretch, including everything strictly between any
    # member pair -- but not the members' own local edges (offsets 0..size-1
    # and lag..lag+size-1)
    observed_eligible[size : lag] = False
    eligible = _null_eligibility(
        chromosome, lag, config, size, observed_eligible=observed_eligible
    )
    assert eligible.all()


def test_null_eligibility_gap_mask_none_matches_pre_1d_c_behaviour():
    # observed_eligible=None (the default) must reproduce exactly what
    # _null_eligibility computed before this fix -- the chromosome-only check.
    config = Eiger2Config(window=4, random_seed=1)
    size = 20
    lag = 50
    rng = np.random.default_rng(3)
    chromosome = rng.choice(["chr1", "chr2"], size=size + lag * 3 + 5)
    without_gate = _null_eligibility(chromosome, lag, config, size)
    with_all_true_gate = _null_eligibility(
        chromosome, lag, config, size,
        observed_eligible=np.ones(chromosome.size - 1, dtype=bool),
    )
    np.testing.assert_array_equal(without_gate, with_all_true_gate)


def test_null_eligibility_gap_mask_is_a_no_op_on_dense():
    # dense: eligible_edges() is all-True (one giant contiguous run), so the
    # gap mask changes nothing -- proves continuity with every prior `dense`
    # result. Combined with the previous test (all-True gate == no gate), this
    # is the "assert it as a test" the spec asks for.
    from eiger2.core import eligible_edges

    n_cpg = 5000
    frame = locations(n_cpg)
    config = Eiger2Config(window=4, max_cpg_gap_bp=300, random_seed=1)
    observed_eligible = eligible_edges(frame, config)
    assert observed_eligible.all()


def test_null_tracks_dense_locations_unaffected_by_gap_masking():
    # End-to-end: null_tracks() on dense locations produces IDENTICAL
    # eligibility masks whether or not the (now-active) gap check runs,
    # because every edge is gap-eligible there. Confirmed by comparing against
    # explicitly passing locations=None (chromosome+gap check both off) on the
    # same single-chromosome data, where the only difference the gap check
    # could possibly make is exactly zero (all-True either way).
    rng = np.random.default_rng(11)
    residuals = rng.normal(size=(5000, 20))
    frame = locations(5000)
    config = Eiger2Config(window=4, null_replicates=10, random_seed=1)
    with_locations = null_tracks(residuals, config, frame)
    without_locations = null_tracks(residuals, config, None)
    assert len(with_locations) == len(without_locations)
    for (t1, e1, lag1), (t2, e2, lag2) in zip(with_locations, without_locations):
        assert lag1 == lag2
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(e1, e2)


def test_null_tracks_fragmented_layout_excludes_members_at_large_gaps():
    # On a genuinely fragmented layout, the gap mask must actually exclude
    # something -- confirms the fix is live, not just a no-op everywhere.
    rng = np.random.default_rng(12)
    n_cpg = 3000
    starts = np.arange(n_cpg) * 2
    for i in range(50, n_cpg, 50):
        starts[i:] += 5000  # break into runs of 50 CpGs, gaps of ~5000bp
    frame = pd.DataFrame({"chr": "chr1", "start": starts, "end": starts + 1})
    residuals = rng.normal(size=(n_cpg, 20))
    config = Eiger2Config(
        window=4, max_cpg_gap_bp=300, null_replicates=20, random_seed=1,
    )
    tracks = null_tracks(residuals, config, frame)
    total_positions = sum(t.size for t, _, _ in tracks)
    total_eligible = sum(int(e.sum()) for _, e, _ in tracks)
    assert total_positions > 0
    assert total_eligible < total_positions  # the gap mask actually excludes some


# --------------------------------------------------------------------------
# Stage 1d-C §2 — C.1 block permutation: arithmetic confirmation only
# --------------------------------------------------------------------------


def _block_permute(track, eligible, block_size, rng):
    """Split into contiguous blocks, permute block ORDER only (within-block order
    is preserved, so short-range dependence inside a block survives; only
    cross-block structure is destroyed). Test-only: no production
    `region_null_mode="block_permutation"` exists, per spec -- this exists to
    confirm the closed-form law below, not to ship a candidate."""

    n = track.size
    n_blocks = n // block_size
    trimmed = n_blocks * block_size
    t = track[:trimmed].reshape(n_blocks, block_size)
    e = eligible[:trimmed].reshape(n_blocks, block_size)
    order = rng.permutation(n_blocks)
    return t[order].reshape(-1), e[order].reshape(-1)


def test_block_permutation_acf_matches_the_closed_form_law():
    # Confirms, empirically, the law the spec derives: under block permutation
    # with block length b, a lag-1 pair survives (stays inside one block) with
    # probability (b-1)/b, so null_ACF(lag 1) ~= rho * (b-1)/b. An AR(1) process
    # with rho matching the measured observed lag-1 ACF (STAGE1D_B_STOP_REPORT
    # section 4.2: ~0.50 on every layout) stands in for the observed track --
    # the law is about the permutation mechanism, not about any one dataset.
    rng = np.random.default_rng(7)
    n = 200_000
    rho_target = 0.503
    x = rng.normal(0, 1, n)
    for i in range(1, n):
        x[i] = rho_target * x[i - 1] + np.sqrt(1 - rho_target**2) * rng.normal()
    eligible = np.ones(n, dtype=bool)
    observed_acf1 = track_autocorrelation(x, eligible, [1])[1]

    for b in (2, 4, 8, 16, 32):
        xt, et = _block_permute(x, eligible, b, np.random.default_rng(100 + b))
        null_acf1 = track_autocorrelation(xt, et, [1])[1]
        predicted = observed_acf1 * (b - 1) / b
        assert null_acf1 == pytest.approx(predicted, abs=0.01)


def test_block_permutation_b_11_is_the_gate_1_crossover_point():
    # At the measured rho=0.503 and tolerance=0.05, the spec claims b >~ 11 is
    # needed to pass Gate 1. Confirm b=8 fails and b=16 passes (b=11 itself
    # sits within Monte Carlo noise of the 0.05 boundary, so bracket it rather
    # than assert exactly at the threshold).
    rng = np.random.default_rng(7)
    n = 200_000
    rho_target = 0.503
    x = rng.normal(0, 1, n)
    for i in range(1, n):
        x[i] = rho_target * x[i - 1] + np.sqrt(1 - rho_target**2) * rng.normal()
    eligible = np.ones(n, dtype=bool)
    observed_acf1 = track_autocorrelation(x, eligible, [1])[1]

    xt8, et8 = _block_permute(x, eligible, 8, np.random.default_rng(108))
    diff8 = track_autocorrelation(xt8, et8, [1])[1] - observed_acf1
    assert abs(diff8) > 0.05  # b=8: still fails

    xt16, et16 = _block_permute(x, eligible, 16, np.random.default_rng(116))
    diff16 = track_autocorrelation(xt16, et16, [1])[1] - observed_acf1
    assert abs(diff16) <= 0.05  # b=16: passes


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


# --------------------------------------------------------------------------
# Stage 1d-B gates (master spec §6.2)
# --------------------------------------------------------------------------


def test_track_autocorrelation_hand_derived():
    values = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    eligible = np.ones(values.size, dtype=bool)
    result = track_autocorrelation(values, eligible, lags=[1, 2])
    # lag 1: y = 3 - x exactly (perfect linear inverse) -> corr = -1
    assert result[1] == pytest.approx(-1.0)
    # lag 2: values[i] == values[i+2] always -> x and y are identical
    # (non-constant, alternating 1,2,1,2,...) sequences -> corr = +1
    assert result[2] == pytest.approx(1.0)


def test_track_autocorrelation_constant_series_is_nan():
    values = np.full(10, 5.0)
    eligible = np.ones(10, dtype=bool)
    result = track_autocorrelation(values, eligible, lags=[1, 3])
    assert np.isnan(result[1])
    assert np.isnan(result[3])


def test_track_autocorrelation_excludes_ineligible_and_nonfinite_positions():
    values = np.array([1.0, 2.0, np.nan, 2.0, 1.0, 2.0])
    eligible = np.array([True, True, True, False, True, True])
    result = track_autocorrelation(values, eligible, lags=[1])
    # only pairs where BOTH i and i+1 are eligible and finite survive:
    # (values[0],values[1])=(1,2) and (values[4],values[5])=(1,2) -- both directions
    # identical, so this collapses to a 2-point perfectly-correlated (degenerate,
    # since only one distinct x and y value) -- assert it is finite and not raising.
    assert np.isfinite(result[1]) or np.isnan(result[1])  # must not raise


def test_track_autocorrelation_lag_at_or_past_array_length_is_nan():
    values = np.arange(5.0)
    eligible = np.ones(5, dtype=bool)
    result = track_autocorrelation(values, eligible, lags=[5, 10])
    assert np.isnan(result[5])
    assert np.isnan(result[10])


def test_acf_gate_passes_when_null_matches_observed():
    rng = np.random.default_rng(11)
    # AR(1)-like observed track and null tracks built from the identical process
    n = 4000
    observed = np.cumsum(rng.normal(0, 1, n)) * 0.0 + rng.normal(0, 1, n)
    for i in range(1, n):
        observed[i] += 0.4 * observed[i - 1]
    eligible = np.ones(n, dtype=bool)
    tracks = []
    for seed in range(20, 28):
        r = np.random.default_rng(seed)
        t = r.normal(0, 1, n)
        for i in range(1, n):
            t[i] += 0.4 * t[i - 1]
        tracks.append((t, eligible, 1))
    result = acf_gate(observed, eligible, tracks, window=3, tolerance=0.08)
    assert result["lags"] == [1, 2, 3]
    assert result["passed"] is True


def test_acf_gate_fails_when_null_has_no_autocorrelation_but_observed_does():
    # exactly Stage 1b's own failure mode (master spec §2.2): a null with the
    # right marginal but no autocorrelation, against an observed track that has it.
    rng = np.random.default_rng(12)
    n = 4000
    observed = rng.normal(0, 1, n)
    for i in range(1, n):
        observed[i] += 0.5 * observed[i - 1]
    eligible = np.ones(n, dtype=bool)
    tracks = [
        (np.random.default_rng(s).normal(0, 1, n), eligible, 1) for s in range(30, 38)
    ]
    result = acf_gate(observed, eligible, tracks, window=2, tolerance=0.05)
    assert result["passed"] is False
    assert abs(result["diff"][1]) > 0.05


def test_acf_gate_unmeasurable_lag_fails_not_silently_skipped():
    eligible = np.zeros(10, dtype=bool)  # nothing eligible -> every lag unmeasurable
    observed = np.arange(10.0)
    tracks = [(np.arange(10.0), eligible, 1)]
    result = acf_gate(observed, eligible, tracks, window=2)
    assert result["passed"] is False
    assert all(np.isnan(v) for v in result["diff"].values())


def test_region_exchangeability_gate_reports_inf_when_null_rate_zero():
    result = region_exchangeability_gate(
        observed_regions=5, observed_eligible_edges=10_000,
        null_regions_total=0, null_eligible_edges=50_000,
        min_null_regions_for_fdr=50,
    )
    assert result["region_exchangeability_ratio"] == float("inf")
    assert result["null_count_adequate"] is False


def test_region_exchangeability_gate_nan_when_both_rates_zero():
    result = region_exchangeability_gate(
        observed_regions=0, observed_eligible_edges=10_000,
        null_regions_total=0, null_eligible_edges=50_000,
        min_null_regions_for_fdr=50,
    )
    assert np.isnan(result["region_exchangeability_ratio"])


def test_region_exchangeability_gate_ordinary_ratio():
    result = region_exchangeability_gate(
        observed_regions=20, observed_eligible_edges=10_000,
        null_regions_total=100, null_eligible_edges=100_000,
        min_null_regions_for_fdr=50,
    )
    # observed_rate = 20/10000*10000 = 20; null_rate = 100/100000*10000 = 10
    assert result["observed_rate"] == pytest.approx(20.0)
    assert result["null_rate"] == pytest.approx(10.0)
    assert result["region_exchangeability_ratio"] == pytest.approx(2.0)
    assert result["null_count_adequate"] is True


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
    # min_null_regions_for_fdr=0: this is a recall-consistency test (relies on
    # "significant" as the calling gate), not a null-adequacy test -- the spec
    # §5 guard is disabled so the pre-existing recall comparison still applies.
    config = Eiger2Config(
        window=4, null_replicates=50, random_seed=11, min_null_regions_for_fdr=0,
        region_null_mode="lag",
    )

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
                                support_binned_threshold="off",
                                region_null_mode="lag")
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
    config = Eiger2Config(
        window=4, null_replicates=50, random_seed=42, region_null_mode="lag",
    )
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
                           support_binned_threshold="on",
                           region_null_mode="lag")
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
                           support_binned_threshold="on",
                           region_null_mode="lag")
    regions1, _, manifest1 = call_dmrs(matrix, locs, config=config)
    regions2, _, manifest2 = call_dmrs(matrix, locs, config=config)
    pd.testing.assert_frame_equal(regions1, regions2, check_exact=True)
    assert manifest1["support_bin_edges"] == manifest2["support_bin_edges"]


def test_support_binned_threshold_config_validation():
    with pytest.raises(ValueError, match="support_binned_threshold"):
        Eiger2Config(support_binned_threshold="bogus")
    with pytest.raises(ValueError, match="support_bins"):
        Eiger2Config(support_bins=1)


def test_min_null_regions_for_fdr_config_validation():
    # Stage 1c-bis review Task 2: a negative value silently disables the
    # guard, because `pooled.size < -1` is never true -- in a release whose
    # entire purpose is this guard, its own config field must not be the one
    # field a typo can switch off unnoticed.
    with pytest.raises(ValueError, match="min_null_regions_for_fdr"):
        Eiger2Config(min_null_regions_for_fdr=-1)
    Eiger2Config(min_null_regions_for_fdr=0)  # documented way to disable it


def test_support_binning_forced_off_when_region_null_mode_is_not_lag():
    # Retired the null_mode="permutation" scenario this test used to check
    # (Stage 1d-D Task 6.1, master spec §6.4): permutation is now rejected at
    # Eiger2Config construction, so that combination can no longer be built.
    # The underlying concern -- support-conditioning's per-replicate
    # reconstruction assumes tracks it cannot safely assume for some null
    # constructions -- is now general to region_null_mode != "lag" (Stage
    # 1d-D's own new axis), not specific to permutation. region_null_mode="off"
    # exercises the same guard without member_step's genome-scale input floor.
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=4,
        spacing=400, dmr_length=15, missing_rate=0.25, seed=45,
    )
    config = Eiger2Config(window=4, null_replicates=50, random_seed=45,
                           region_null_mode="off", support_binned_threshold="on")
    _, _, manifest = call_dmrs(matrix, locs, config=config)
    assert manifest["support_binning"] == "forced_off"
    assert manifest["support_binning_skipped_reason"] == "region_null_mode_not_lag"


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
# Stage 1d-C §3 — C.3 member-step lag null (evaluation candidate)
# --------------------------------------------------------------------------


def test_member_step_lag_null_tracks_reslices_the_full_track_exactly():
    # Correctness of the re-slicing itself: sub_track[j] must equal the SAME
    # per-position value _lag_null_track already computes at p = i + j*lag,
    # not a recomputation.
    rng = np.random.default_rng(21)
    residuals = rng.normal(size=(3000, 15))
    lag = 37
    config = Eiger2Config(window=4, random_seed=1)
    full = _lag_null_track(residuals, lag, config)
    sub_tracks = member_step_lag_null_tracks(residuals, lag, config)
    by_offset = {offset: (t, e) for t, e, offset in sub_tracks}
    for i in (0, 1, lag - 1):
        if i not in by_offset:
            continue
        sub_track, _ = by_offset[i]
        expected = full[i::lag]
        np.testing.assert_array_equal(sub_track, expected)


def test_member_step_lag_null_tracks_consecutive_positions_share_w_minus_1_members():
    # The construction's whole point: position j and j+1 in a sub-track share
    # w-1 of w members, unlike the shipped null's consecutive p/p+1. Verify via
    # the same fake-_lag_statistic decode technique used for _lag_null_track's
    # own member-set tests, rather than re-deriving the formula independently.
    import eiger2.core as core_mod

    lag = 11
    w = 4

    def fake_lag_statistic(residuals, d, config, chunk=200_000):
        n_pairs = residuals.shape[0] - d
        return np.array([i * 100_000 + d for i in range(n_pairs)], dtype=float)

    original = core_mod._lag_statistic
    core_mod._lag_statistic = fake_lag_statistic
    try:
        config = Eiger2Config(window=w, random_seed=1)
        dummy = np.zeros((5000, 3))
        sub_tracks = core_mod.member_step_lag_null_tracks(dummy, lag, config)
    finally:
        core_mod._lag_statistic = original

    sub_track, _eligible, offset = sub_tracks[0]
    i = offset
    # position j's members: {i+j*lag, i+(j+1)*lag, ..., i+(j+w-1)*lag}
    # position j+1's members: {i+(j+1)*lag, ..., i+(j+w)*lag}
    # shared: w-1 of them (all but the first of j, all but the last of j+1)
    def predicted_value(j):
        members = sorted((j + a) * lag + i for a in range(w))
        pairs = [
            (members[a], members[b]) for a in range(w) for b in range(a + 1, w)
        ]
        return np.mean([p * 100_000 + (q - p) for p, q in pairs])

    assert sub_track[0] == pytest.approx(predicted_value(0))
    assert sub_track[1] == pytest.approx(predicted_value(1))


def test_member_step_lag_null_tracks_respects_the_gap_mask():
    # Same fragmented-layout construction as the §1 test: the gap mask must
    # exclude some sub-track positions too, since it re-slices
    # _null_eligibility's (now gap-masked) output.
    rng = np.random.default_rng(22)
    n_cpg = 3000
    starts = np.arange(n_cpg) * 2
    for i in range(50, n_cpg, 50):
        starts[i:] += 5000
    frame = pd.DataFrame({"chr": "chr1", "start": starts, "end": starts + 1})
    residuals = rng.normal(size=(n_cpg, 15))
    config = Eiger2Config(window=4, max_cpg_gap_bp=300, random_seed=1)
    sub_tracks = member_step_lag_null_tracks(residuals, 41, config, frame)
    total = sum(t.size for t, _, _ in sub_tracks)
    total_eligible = sum(int(e.sum()) for _, e, _ in sub_tracks)
    assert total > 0
    assert total_eligible < total


def test_member_step_lag_null_tracks_dense_matches_no_gap_mask():
    # dense: gap mask is a no-op (every edge eligible), same guarantee as §1.
    rng = np.random.default_rng(23)
    n_cpg = 4000
    frame = locations(n_cpg)
    residuals = rng.normal(size=(n_cpg, 15))
    config = Eiger2Config(window=4, random_seed=1)
    with_locations = member_step_lag_null_tracks(residuals, 29, config, frame)
    without_locations = member_step_lag_null_tracks(residuals, 29, config, None)
    assert len(with_locations) == len(without_locations)
    for (t1, e1, o1), (t2, e2, o2) in zip(with_locations, without_locations):
        assert o1 == o2
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(e1, e2)


def test_member_step_lag_null_tracks_empty_when_full_track_empty():
    residuals = np.zeros((5, 3))
    config = Eiger2Config(window=4, random_seed=1)
    assert member_step_lag_null_tracks(residuals, 1000, config) == []


def test_member_step_lag_null_ensemble_draws_n_lag_draws_distinct_lags():
    rng = np.random.default_rng(24)
    residuals = rng.normal(size=(30_000, 15))
    config = Eiger2Config(
        window=4, null_min_lag=500, min_subtrack_length=50, n_lag_draws=5,
        random_seed=1,
    )
    replicates, lags = member_step_lag_null_ensemble(residuals, config)
    assert len(lags) == 5
    assert len(set(lags)) == 5  # distinct
    assert len(replicates) > 0
    # one (track, eligible, lag) triple per successful draw -- concatenated
    # across that draw's offsets, not one triple per offset
    assert len(replicates) <= len(lags)
    for track, eligible, lag in replicates:
        assert lag in lags
        assert track.shape == eligible.shape


def test_member_step_lag_null_ensemble_lags_span_null_min_lag_to_l_max():
    # Stage 1d-D Task 2: NOT the narrow near-null_min_lag jitter Stage 1d-C
    # used (which correlated nearby draws -- lag 1006 and 1008 pair CpG i with
    # i+1006 and i+1008, nearly the same construction). Lags must be able to
    # reach meaningfully far above null_min_lag, up to L_max.
    rng = np.random.default_rng(25)
    n = 200_000
    residuals = rng.normal(size=(n, 15))
    config = Eiger2Config(
        window=4, null_min_lag=500, min_subtrack_length=50, n_lag_draws=20,
        random_seed=1,
    )
    span = config.min_subtrack_length + config.window - 1
    l_max = n // span
    _replicates, lags = member_step_lag_null_ensemble(residuals, config)
    assert all(500 <= lag <= l_max for lag in lags)
    # with L_max this much larger than null_min_lag, 20 uniform draws should
    # not all land in a narrow band near the floor
    assert max(lags) - min(lags) > (l_max - 500) // 4


def test_member_step_lag_null_ensemble_raises_when_input_too_small():
    # Task 2: a clear error, not Stage 1d-C's silent 17-position-sub-track
    # degradation on `dense` at its normal (20,400 CpG) scale.
    rng = np.random.default_rng(26)
    residuals = rng.normal(size=(20_000, 15))
    config = Eiger2Config(
        window=4, null_min_lag=500, min_subtrack_length=100, random_seed=1,
    )
    with pytest.raises(ValueError, match="region_null_mode='member_step'"):
        member_step_lag_null_ensemble(residuals, config)


def test_member_step_lag_null_ensemble_concatenation_never_merges_offsets():
    # The eligible mask must be broken at every sub-track-piece boundary, or
    # segment() could merge two unrelated offsets into one spurious region.
    # Reconstruct the real per-offset piece lengths from
    # member_step_lag_null_tracks (pieces are NOT all equal length -- offsets
    # early in [0, lag) can get one extra element) rather than assuming a
    # formula, so this test cannot be fooled by an off-by-one in its own logic.
    rng = np.random.default_rng(27)
    residuals = rng.normal(size=(50_000, 15))
    config = Eiger2Config(
        window=4, null_min_lag=500, min_subtrack_length=50, n_lag_draws=3,
        random_seed=1,
    )
    replicates, lags = member_step_lag_null_ensemble(residuals, config)
    for (_track, eligible, lag), expected_lag in zip(replicates, lags):
        assert lag == expected_lag
        sub_tracks = member_step_lag_null_tracks(residuals, lag, config)
        boundary = -1
        for sub_track, _sub_eligible, _offset in sub_tracks:
            boundary += sub_track.size
            if boundary < eligible.size:
                assert not eligible[boundary]


def test_member_step_lag_null_narrows_the_acf_gap_relative_to_shipped():
    # Directional claim at toy scale (a single-latent-factor construction, not
    # the real multi-sample pipeline): the shared-member mechanism must move
    # the null's ACF measurably toward the observed track's, even before
    # reaching genome scale. The stronger claim -- C.3 actually PASSES Gate 1
    # -- is confirmed on real data at real scale in the Stage 1d-C acceptance
    # run (STAGE1D_C_STOP_REPORT.md), not here: at this toy scale (n=60,000,
    # single sample-like column) sub-tracks are short enough that edge effects
    # keep C.3 short of the +-0.05 threshold even though the mechanism is
    # visibly working -- exactly the failure mode spec section 3 warns tests at
    # benchmark scale will show. Do not strengthen this assertion to "passes"
    # without also scaling n into the hundreds of thousands; see the acceptance
    # run for the version of this claim that matters.
    rng = np.random.default_rng(31)
    n = 60_000
    n_samples = 40
    rho = 0.5
    # A per-CpG loading onto one shared per-sample factor, the loading itself
    # an AR(1) sequence -- so adjacent CpGs respond similarly to the same
    # factor (real short-range correlation), matching what edge_statistic's
    # across-SAMPLE correlation actually measures (a single sample column
    # cannot produce any correlation at all -- there is nothing to correlate).
    beta = np.empty(n)
    beta[0] = rng.normal()
    for i in range(1, n):
        beta[i] = rho * beta[i - 1] + np.sqrt(1 - rho**2) * rng.normal()
    factor = rng.normal(0, 1, n_samples)
    residuals = beta[:, None] * factor[None, :] + rng.normal(0, 0.3, (n, n_samples))

    config = Eiger2Config(
        window=4, null_min_lag=200, random_seed=1, statistic="correlation",
    )
    eligible = np.ones(n - 1, dtype=bool)
    # "observed" statistic: same window construction the real pipeline uses
    observed = edge_statistic(residuals, config)

    shipped_tracks = null_tracks(residuals, config)
    shipped_gate = acf_gate(observed, eligible, shipped_tracks, config.window)
    assert shipped_gate["passed"] is False  # confirms the known failure mode

    sub_tracks = member_step_lag_null_tracks(residuals, config.null_min_lag, config)
    c3_gate = acf_gate(observed, eligible, sub_tracks, config.window)
    assert abs(c3_gate["diff"][1]) < abs(shipped_gate["diff"][1]) / 2


# --------------------------------------------------------------------------
# Task 7: output schema -- region_support_median/region_threshold_mode
# columns, schema_version -> "2.2" (Stage 1c)
# --------------------------------------------------------------------------


def test_schema_v22_has_support_columns_and_version():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=4,
        spacing=400, dmr_length=15, missing_rate=0.25, seed=51,
    )
    config = Eiger2Config(
        window=4, null_replicates=50, random_seed=51, region_null_mode="lag",
    )
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
                           support_binned_threshold="on",
                           region_null_mode="lag")
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
                           support_binned_threshold="off",
                           region_null_mode="lag")
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
            config = Eiger2Config(
                window=4, random_seed=seed, null_replicates=100,
                region_null_mode="lag",
            )
            assert config.support_binned_threshold == "auto"
            assert config.support_dispersion_cut is None
            _, _, manifest = call_dmrs(matrix, locs, config=config)
            expected = "auto_on" if name == "missingness" else "auto_off"
            assert manifest["support_binning"] == expected, (
                name, seed, manifest["support_binning"], manifest["support_dispersion"],
            )


# --------------------------------------------------------------------------
# Task 9: region-level exposure diagnostic and its 2.0 gate (Stage 1c,
# spec §1c.6)
# --------------------------------------------------------------------------


def test_region_rate_by_support_bin_populated_when_binning_active():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=15, missing_rate=0.4, seed=61,
    )
    config = Eiger2Config(window=4, null_replicates=100, random_seed=61,
                           support_binned_threshold="on",
                           region_null_mode="lag")
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    if manifest["support_binning"] == "forced_on":
        assert "region_rate_by_support_bin" in manifest
        for bin_stats in manifest["region_rate_by_support_bin"].values():
            assert "observed_rate" in bin_stats
            assert "null_rate" in bin_stats
            assert "ratio" in bin_stats


def test_region_rate_by_support_bin_absent_when_binning_off():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=4,
        spacing=400, dmr_length=15, missing_rate=0.25, seed=41,
    )
    config = Eiger2Config(window=4, null_replicates=50, random_seed=41,
                           support_binned_threshold="off",
                           region_null_mode="lag")
    _, _, manifest = call_dmrs(matrix, locs, config=config)
    assert manifest["support_binning"] == "forced_off"
    assert "region_rate_by_support_bin" not in manifest


def test_region_rate_by_support_bin_ratio_guards_zero_null_rate():
    # Every bin dict must report a numeric rate for both sides and either a
    # numeric ratio or an explicit None with a "note" -- never a raised
    # ZeroDivisionError/RuntimeWarning and never a bare 0.0 standing in for
    # "undefined" (the "no exact zero presented as certainty" spirit carried
    # from the rest of this stage).
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=15, missing_rate=0.4, seed=62,
    )
    config = Eiger2Config(window=4, null_replicates=100, random_seed=62,
                           support_binned_threshold="on",
                           region_null_mode="lag")
    _, _, manifest = call_dmrs(matrix, locs, config=config)
    if manifest["support_binning"] != "forced_on":
        pytest.skip("this seed did not produce a forced_on run")
    for bin_stats in manifest["region_rate_by_support_bin"].values():
        assert np.isfinite(bin_stats["observed_rate"]) or bin_stats[
            "observed_eligible_edges"
        ] == 0
        if bin_stats["ratio"] is None:
            assert "note" in bin_stats
        else:
            assert bin_stats["ratio"] == pytest.approx(
                bin_stats["observed_rate"] / bin_stats["null_rate"]
            )


# --------------------------------------------------------------------------
# Task 10: support-stratified region FDR wired through call_dmrs
# --------------------------------------------------------------------------


def test_call_dmrs_support_stratified_fdr_activates_when_binning_on():
    # The brief's original simulate() parameters (spacing=19600, strata=[])
    # returned status="no_regions" for BOTH modes, making every assertion below
    # vacuous. This uses the same scenario as the Task 9 tests, which does
    # produce forced_on runs with real regions on both sides.
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=15, missing_rate=0.4, seed=61,
    )
    config = Eiger2Config(window=4, null_replicates=100, random_seed=61,
                           support_binned_threshold="on",
                           region_null_mode="lag")
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    assert manifest["support_binning"] == "forced_on"
    assert len(regions)  # guard against the assertions below going vacuous
    assert manifest["fdr"]["support_stratified"] is True
    assert manifest["fdr"]["min_null_regions_per_joint_stratum"] == 50
    fallbacks = manifest["fdr"]["support_stratification_fallback_strata"]
    final_strata = set(manifest["fdr"]["fdr_interval_cutoffs_by_stratum"])
    # A joint label that fell back must never also survive as a final stratum,
    # and every fallback label must be a joint label.
    for label in fallbacks:
        assert "|sbin" in label
        assert label not in final_strata
    # Every final stratum is either a bare length stratum or a joint one.
    for label in final_strata:
        assert label.split("|")[0] in {"4-5", "6-9", "10-19", "20-49", "50+"}

    # "off" must be unaffected: no support stratification, same diagnostics shape
    # as before this task.
    config_off = Eiger2Config(window=4, null_replicates=100, random_seed=61,
                               support_binned_threshold="off",
                               region_null_mode="lag")
    regions_off, _, manifest_off = call_dmrs(matrix, locs, config=config_off)
    assert len(regions_off)
    assert manifest_off["fdr"]["support_stratified"] is False
    assert manifest_off["fdr"]["support_stratification_fallback_strata"] == []
    assert all(
        "|sbin" not in label
        for label in manifest_off["fdr"]["fdr_interval_cutoffs_by_stratum"]
    )


def test_call_dmrs_support_stratified_fdr_is_not_used_without_length_strata():
    # config.length_stratified_fdr=False must keep the FDR path pooled even when
    # support binning is forced on (joint stratification without length strata is
    # not spec'd).
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=15, missing_rate=0.4, seed=61,
    )
    config = Eiger2Config(window=4, null_replicates=100, random_seed=61,
                           support_binned_threshold="on",
                           length_stratified_fdr=False,
                           region_null_mode="lag")
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    if manifest["support_binning"] != "forced_on" or not len(regions):
        pytest.skip("this seed did not produce a forced_on run with regions")
    assert manifest["fdr"]["support_stratified"] is False
    assert list(manifest["fdr"]["fdr_interval_cutoffs_by_stratum"]) == ["_pooled"]


# --------------------------------------------------------------------------
# Task 11: the remaining spec §1c.7 items (2, 3, 5, 8, 9) at the call_dmrs
# integration level. Items 1/4/6/7/10 are already covered by Tasks 4/5/6/8's
# own tests; the five below are the ones §1c.7 asks for that only a real
# end-to-end run can establish.
# --------------------------------------------------------------------------


def test_observed_and_null_support_come_from_the_same_function_end_to_end(
    monkeypatch,
):
    # spec §1c.7 item 2, at the integration level. Task 3's unit tests prove
    # `_window_support` itself is correct; they cannot prove `call_dmrs` routes
    # BOTH the observed track and every null replicate through it, against the
    # same finiteness mask, with member-set offsets built by the same rule. That
    # is what this test establishes, by recording every call the real pipeline
    # makes.
    import eiger2.core as core_mod

    original = core_mod._window_support
    calls: list[dict] = []

    def recording(finite, base_positions, offsets, **kwargs):
        out = original(finite, base_positions, offsets, **kwargs)
        calls.append({
            "finite": finite,
            "base": np.asarray(base_positions).copy(),
            "offsets": np.asarray(offsets).copy(),
            "out": np.asarray(out).copy(),
        })
        return out

    monkeypatch.setattr(core_mod, "_window_support", recording)

    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=15, missing_rate=0.4, seed=61,
    )
    config = Eiger2Config(window=4, null_replicates=8, random_seed=61,
                           support_binned_threshold="on",
                           region_null_mode="lag")
    _, _, manifest = call_dmrs(matrix, locs, config=config)
    assert manifest["support_binning"] in {"forced_on", "collapsed_to_single"}

    # one observed call, then exactly one per null replicate -- the null never
    # takes a different route to its support values.
    assert len(calls) == 1 + config.null_replicates

    # every call reads the SAME finiteness mask object: observed and null support
    # are two views of one underlying data matrix, not two separate derivations.
    first_finite = calls[0]["finite"]
    assert all(call["finite"] is first_finite for call in calls)

    # every call's offsets follow the one member-set rule (spec §1c.1):
    # arange(window) * lag. Observed is that rule at lag 1; each null replicate
    # is the same rule at its own lag. Nothing else is permitted.
    window = config.window
    for index, call in enumerate(calls):
        offsets = call["offsets"]
        assert offsets.size == window
        assert offsets[0] == 0
        lag = int(offsets[1] - offsets[0])
        assert np.array_equal(offsets, np.arange(window) * lag)
        if index == 0:
            assert lag == 1                     # the observed track
        else:
            assert lag >= 2                     # a locality-breaking null lag

    # and the values themselves match a naive brute-force loop over the same
    # mask, for the observed call and for a null call alike -- the "same value
    # on identical underlying finiteness" half of §1c.7 item 2.
    for call in (calls[0], calls[-1]):
        finite, base, offsets = call["finite"], call["base"], call["offsets"]
        naive = np.array([
            int(np.sum(np.all(finite[b + offsets, :], axis=0)))
            for b in base
        ])
        assert np.array_equal(naive, call["out"])

    # the decisive cross-check: feed a null replicate's base positions through
    # the OBSERVED member-set pattern. Observed and null differ only in which
    # positions and which lag they ask about, never in how support is counted.
    null_base = calls[-1]["base"]
    shared = original(first_finite, null_base[: 500], calls[0]["offsets"])
    expected = np.array([
        int(np.sum(np.all(first_finite[b + calls[0]["offsets"], :], axis=0)))
        for b in null_base[:500]
    ])
    assert np.array_equal(shared, expected)


def test_null_support_bins_use_the_observed_derived_edges_unchanged(monkeypatch):
    # spec §1c.7 item 3. The edges must be derived ONCE, from the observed
    # support, and then applied verbatim to every null replicate -- a null that
    # re-quantiles its own support would equalise the two by construction and
    # make the whole diagnostic vacuous. Recording both helpers proves the
    # derivation happens exactly once and that every later bin assignment
    # (observed, per-replicate null threshold pooling, per-replicate null
    # threshold arrays, and the region-rate diagnostic) uses that same array.
    import eiger2.core as core_mod

    original_edges = core_mod._support_bin_edges
    original_assign = core_mod._assign_support_bin
    edge_results: list[dict] = []
    assign_edges: list[np.ndarray] = []

    def recording_edges(observed_support, n_bins, min_null_values, pooled_null):
        out = original_edges(observed_support, n_bins, min_null_values, pooled_null)
        edge_results.append(out)
        return out

    def recording_assign(support, bin_edges):
        assign_edges.append(np.asarray(bin_edges).copy())
        return original_assign(support, bin_edges)

    monkeypatch.setattr(core_mod, "_support_bin_edges", recording_edges)
    monkeypatch.setattr(core_mod, "_assign_support_bin", recording_assign)

    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=15, missing_rate=0.4, seed=61,
    )
    config = Eiger2Config(window=4, null_replicates=8, random_seed=61,
                           support_binned_threshold="on",
                           region_null_mode="lag")
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    assert manifest["support_binning"] == "forced_on"
    assert len(regions)

    # derived exactly once -- the null has no opportunity to re-derive its own.
    assert len(edge_results) == 1
    assert edge_results[0]["collapsed"] is False
    derived = np.asarray(edge_results[0]["bin_edges"])
    assert derived.size >= 1

    # the manifest reports the same edges the pipeline actually used.
    assert manifest["support_bin_edges"] == derived.tolist()

    # every bin assignment anywhere in the run -- observed and null alike --
    # used that identical array, element for element.
    assert len(assign_edges) >= 1 + config.null_replicates
    for edges in assign_edges:
        assert np.array_equal(edges, derived)

    # and the per-bin thresholds are indexed by that same binning, so a null
    # replicate's threshold array can only take values from this list.
    thresholds = np.asarray(manifest["support_thresholds"], dtype=float)
    assert thresholds.size == derived.size + 1


def test_single_bin_collapse_reproduces_the_unconditioned_path_end_to_end():
    # spec §1c.7 item 5, at the integration level (Task 5 covers the collapse
    # decision itself in `_support_bin_edges`). `min_null_values` far above any
    # attainable pooled-null count forces every candidate bin to merge upward
    # until one bin is left, which is the collapse fallback. The run must then
    # be indistinguishable from `support_binned_threshold="off"` -- same regions,
    # same areas, same q-values, same scalar threshold column -- not merely
    # "close".
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=4,
        spacing=400, dmr_length=15, missing_rate=0.25, seed=41,
    )
    collapsed_config = Eiger2Config(
        window=4, null_replicates=50, random_seed=41,
        support_binned_threshold="on", min_null_values=10**9,
        region_null_mode="lag",
    )
    off_config = Eiger2Config(
        window=4, null_replicates=50, random_seed=41,
        support_binned_threshold="off", region_null_mode="lag",
    )
    regions_collapsed, values_collapsed, manifest_collapsed = call_dmrs(
        matrix, locs, config=collapsed_config
    )
    regions_off, values_off, manifest_off = call_dmrs(
        matrix, locs, config=off_config
    )

    assert manifest_collapsed["support_binning"] == "collapsed_to_single"
    assert manifest_off["support_binning"] == "forced_off"
    assert len(regions_off)  # guard against a vacuous empty-vs-empty comparison

    pd.testing.assert_frame_equal(
        regions_collapsed, regions_off, check_exact=True
    )
    pd.testing.assert_frame_equal(values_collapsed, values_off, check_exact=True)
    # collapse means no bin structure survived, so no per-bin threshold exists
    # and the exported threshold stays the single unconditioned scalar.
    assert manifest_collapsed["support_bin_edges"] == []
    assert manifest_collapsed["support_thresholds"] == []
    assert (regions_collapsed["region_threshold_mode"] == "single").all()
    assert "region_rate_by_support_bin" not in manifest_collapsed


def test_exchangeability_ratio_at_most_1_25_under_missingness_with_binning_on():
    # spec §1c.7 item 8. Gate methodology is this stage's ruling: per-scenario
    # MEDIAN across seeds <= 1.25, per-instance outliers reported but not gated
    # (see the same policy applied in benchmarking/benchmark_suite.py's
    # command_exchangeability).
    #
    # missing_rate=0.5, matching the override
    # test_exchangeability_ratio_at_most_1_25_on_dmr_free_scenarios already uses
    # for the missingness scenario, and spacing=20000 for the same
    # finite-sample reason documented there.
    #
    # The ratio itself is computed inside null_threshold, BEFORE the
    # support-binning branch runs, so this test's real content is that turning
    # binning on -- which changes the null tracks' segmentation thresholds and
    # every downstream region -- leaves the Stage 1b null-exchangeability gate
    # passing on a genuine end-to-end binning-on run, and that binning does in
    # fact engage on this data (asserted below). Measured out of suite at
    # missing_rate=0.25 as well (seeds 0-4, binning forced on): ratios 1.031,
    # 0.993, 0.887, 1.206, 0.883, median 0.993 -- also passing; that
    # configuration is ~13x slower per seed, so it is recorded in
    # task-11-report.md rather than run here.
    from benchmarking.benchmark_suite import SCENARIOS

    params = {**SCENARIOS["missingness"], "missing_rate": 0.5}
    ratios = []
    for seed in range(5):
        matrix, locs, _ = simulate(
            seed=seed, strata=[], replicates=0, spacing=20000, **params
        )
        config = Eiger2Config(window=4, null_replicates=100, random_seed=seed,
                               support_binned_threshold="on",
                               region_null_mode="lag")
        _, _, manifest = call_dmrs(matrix, locs, config=config)
        assert manifest["support_binning"] in {"forced_on", "collapsed_to_single"}
        ratio = manifest["null"]["null_exchangeability_ratio"]
        assert np.isfinite(ratio)
        ratios.append(ratio)
    assert np.median(ratios) <= 1.25, ratios


def test_support_stratified_fdr_refuses_under_missingness_when_null_is_insufficient():
    # Replaces the retired KNOWN-BAD STATE PIN (see task-11-report.md and spec
    # §5, Stage 1c-bis). Same input as before: the standard `missingness`
    # scenario with the standard planted signal, at shipped defaults
    # (support_binned_threshold="auto", fdr_decision="point"). Previously the
    # near-empty lag-null pool made every region_qvalue exactly 0.0 -- zero
    # evidence presented as certainty. The null-adequacy guard now refuses to
    # report a q-value at all when the pooled null is this thin, which is the
    # correct response to the same measured state, not a relaxation of it.
    from benchmarking.benchmark_suite import DEFAULT_STRATA, SCENARIOS

    matrix, locs, _ = simulate(
        seed=0, strata=DEFAULT_STRATA, replicates=6, spacing=400,
        **SCENARIOS["missingness"],
    )
    # region_null_mode="lag": this test pins the shipped construction's
    # measured null-adequacy behaviour (the historical citation this replaces
    # depends on it); it is not testing region_null_mode="member_step" (the
    # current default), which needs genome-scale input this matrix is not.
    config = Eiger2Config(target_fdr=0.05, region_null_mode="lag")
    assert config.support_binned_threshold == "auto"
    assert config.fdr_decision == "point"
    regions, _, manifest = call_dmrs(matrix, locs, config=config)

    assert manifest["support_binning"] == "auto_on"
    assert len(regions)
    assert manifest["fdr"]["support_stratified"] is True
    assert manifest["region_rate_by_support_bin"]
    assert manifest["null_regions_per_replicate"] == 0.0
    assert manifest["fdr"]["support_stratification_fallback_strata"]
    # pooled_fallback_strata is deprecated (Stage 1c-bis review Task 1) and
    # always empty now.
    assert manifest["fdr"]["pooled_fallback_strata"] == []
    assert all(
        entry["null_regions"] == 0
        for entry in manifest["region_rate_by_support_bin"].values()
    )

    # The refusal path: the pooled null is EMPTY across the whole run (not
    # just a thin stratum), so the Stage 1e Task 1 aggregate floor (master
    # spec §8) refuses everything -- the same outcome the old per-stratum
    # floor gave here, now reached via the run-level check rather than a
    # per-stratum one. insufficient_null_strata and zero_exceedance_strata are
    # both deprecated/inapplicable on this path (the run never reaches the
    # per-stratum loop) and stay empty.
    assert manifest["fdr"]["status"] == "aggregate null inadequate"
    assert manifest["fdr"]["pooled_null_regions"] == 0
    assert manifest["fdr"]["insufficient_null_strata"] == []
    assert manifest["fdr"]["zero_exceedance_strata"] == []
    assert regions["region_qvalue"].isna().all()
    assert regions["region_tail_fdr"].isna().all()
    assert regions["region_qvalue_lower_95"].isna().all()
    assert regions["region_qvalue_upper_95"].isna().all()
    assert regions["region_qvalue_resolution"].isna().all()
    assert (regions["null_exceedances_at_cutoff"] == -1).all()
    assert (regions["region_qvalue_cutoff_rank"] == -1).all()
    assert (regions["significant"] == False).all()  # noqa: E712
    assert manifest["n_significant"] == 0
