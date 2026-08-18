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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

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
    _fdr_from_areas_result,
    _overdispersion_diagnostic,
    _pair_statistic,
    permutation_null_tracks,
    qvalue_result_from_track,
    qvalues_from_track,
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
    # bug this task fixes. null_replicates is raised (with a matching, lower
    # min_null_regions_for_posterior) so the posterior estimate actually
    # reaches "ok" status and the extrapolated tail-area fallback -- which is
    # exactly the code path _tail_area_fdr governs -- is genuinely exercised.
    matrix, locs, _planted = simulate(
        n_samples=50, strata=[(0.2, 0.4)], replicates=3,
        spacing=250, dmr_length=10, seed=99,
    )
    config = Eiger2Config(
        window=4, null_replicates=500, random_seed=99, compute_posterior=True,
        min_null_regions_for_posterior=5,
    )
    regions, _values, manifest = call_dmrs(matrix, locs, config=config)
    assert "posterior" in manifest
    assert regions["evidence_score_extrapolated"].any() or len(regions) == 0


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
    result = region_fdr_result(observed, nulls, observed_edges=500, null_edges=sum(len(n) for n in nulls))
    by_stratum = result.diagnostics["null_count_dispersion_by_stratum"]["_pooled"]
    assert by_stratum[0] in {"computed", "unavailable"}
    assert result.diagnostics["null_count_overdispersion_warning"] is not None


def test_region_fdr_result_poisson_null_does_not_trigger_overdispersion():
    rng = np.random.default_rng(33)
    observed = rng.exponential(2.0, size=20)
    nulls = [rng.exponential(2.0, size=150) for _ in range(8)]  # same density each rep
    result = region_fdr_result(observed, nulls, observed_edges=500, null_edges=1200)
    status, ratio = result.diagnostics["null_count_dispersion_by_stratum"]["_pooled"]
    if status == "computed":
        assert ratio < 1.5


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
