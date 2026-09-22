"""Behaviour of the supported coordination-FDR workflow (eiger2.coordination).

Independent expected values for the estimator, the empty/zero/tie cases,
missingness, chromosome and gap geometry, seeds, chunking and streaming, the
single decision family, failure handling, and the independence of primary
calls from the secondary axis annotations.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Python"))
sys.path.insert(0, str(HERE))

from region_fdr_data import make_dataset  # noqa: E402

import eiger2.coordination as coordination  # noqa: E402
from eiger2.axis import AXIS_MIN_SAMPLES_FOR_INFERENCE  # noqa: E402
from eiger2.coordination import (  # noqa: E402
    LEGACY,
    NEWER,
    PRIMARY_COLUMNS,
    CandidateThresholdError,
    PermutationScanError,
    RegionFdrConfig,
    _mean_impute,
    call_coordinated_regions,
    coordination_fdr,
    permute_rows,
    permute_rows_into,
    scan,
    scan_by_chromosome,
)

# A small mixed case (legacy: no lag-feasibility requirement, many null
# regions, q-values spread over (0, 1]).
LEGACY_MIXED = {
    "seed": 12,
    "n_samples": 10,
    "noise_sd": 0.2,
    "n_blocks": 20,
    "block_strength": 0.12,
    "missing_fraction": 0.03,
    "chromosomes": (("chr1", 5_000), ("chr2", 4_000)),
}
# The newer profile needs >= 3,002 CpGs on some chromosome for its lag null.
NEWER_MIXED = {
    "seed": 11,
    "n_samples": 12,
    "n_blocks": 25,
    "block_strength": 0.05,
    "missing_fraction": 0.05,
    "chromosomes": (("chr1", 6_000), ("chr2", 3_500), ("chr3", 800)),
}
AXIS_DATA = {
    "seed": 31,
    "n_samples": 220,
    "axis_strength": 0.25,
    "axis_fraction": 0.03,
    "blocks_follow_axis": True,
    "n_blocks": 12,
    "block_strength": 0.1,
    "missing_fraction": 0.0,
    "chromosomes": (("chr1", 3_400), ("chr2", 3_100)),
}


def _config(profile=NEWER, **kwargs):
    kwargs.setdefault("fdr_replicates", 9)
    kwargs.setdefault("axis_mode", "off")
    return RegionFdrConfig(profile=profile, **kwargs)


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------


def _brute_force_q(observed, nulls):
    """The definition, evaluated literally at every distinct observed score."""

    b = len(nulls)
    pooled = np.concatenate([np.asarray(a, float) for a in nulls]) if nulls else []
    pooled = np.asarray(pooled, float)
    thresholds = sorted({float(x) for x in observed})
    fdr = {}
    for t in thresholds:
        r = sum(1 for x in observed if x >= t)
        n = int(np.sum(pooled >= t))
        fdr[t] = min(1.0, (1 + n) / (b * max(r, 1)))
    return np.array([min(fdr[t] for t in thresholds if t <= s) for s in observed])


def test_fdr_plus_worked_example_with_ties():
    observed = np.array([5.0, 3.0, 3.0, 1.0])
    nulls = [np.array([4.0, 1.0]), np.array([2.0])]
    # R(1)=4, R(3)=3, R(5)=1; sum N(1)=3, sum N(3)=1, sum N(5)=0; B=2
    # FDR+(1)=4/8, FDR+(3)=2/6, FDR+(5)=1/2; q = cumulative minimum upwards.
    result = coordination_fdr(observed, nulls)
    np.testing.assert_array_equal(result.thresholds, [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(result.r_at, [4, 3, 1])
    np.testing.assert_array_equal(result.null_sum_at, [3, 1, 0])
    np.testing.assert_allclose(result.fdr_plus_at, [0.5, 1 / 3, 0.5])
    np.testing.assert_allclose(result.qvalue, [1 / 3, 1 / 3, 1 / 3, 0.5])
    assert result.qvalue[1] == result.qvalue[2]  # ties share the q-value


def test_fdr_plus_matches_the_literal_definition_on_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(200):
        b = int(rng.integers(1, 12))
        observed = rng.choice([0.5, 1.0, 2.0, 3.5, 7.0], size=int(rng.integers(1, 25)))
        nulls = [
            rng.choice([0.5, 1.0, 2.5, 4.0], size=int(rng.integers(0, 5)))
            for _ in range(b)
        ]
        result = coordination_fdr(observed, nulls)
        np.testing.assert_allclose(result.qvalue, _brute_force_q(observed, nulls))


def test_zero_permutation_regions_give_the_added_count_estimate_not_zero():
    result = coordination_fdr(np.array([2.0, 1.0]), [np.empty(0)] * 99)
    # FDR+(1) = 1 / (99 * 2), FDR+(2) = 1 / 99; q = running minimum upwards.
    np.testing.assert_allclose(result.qvalue, [1 / 198, 1 / 198])
    assert np.all(result.qvalue > 0)


def test_qvalues_are_monotone_in_score_and_capped_at_one():
    rng = np.random.default_rng(1)
    observed = rng.exponential(1.0, 60)
    nulls = [rng.exponential(1.5, 40) for _ in range(19)]
    q = coordination_fdr(observed, nulls).qvalue
    order = np.argsort(observed)
    assert np.all(np.diff(q[order]) <= 1e-15)
    assert q.max() <= 1.0


def test_estimator_rejects_incomplete_or_empty_permutation_sets():
    with pytest.raises(ValueError, match="at least one"):
        coordination_fdr(np.array([1.0]), [])
    with pytest.raises(ValueError, match="complete"):
        coordination_fdr(np.array([1.0]), [np.empty(0)] * 3, n_permutations=4)
    with pytest.raises(ValueError, match="finite"):
        coordination_fdr(np.array([np.nan]), [np.empty(0)])
    empty = coordination_fdr(np.empty(0), [np.array([1.0])])
    assert empty.qvalue.size == 0


# --------------------------------------------------------------------------
# empty, discovery-only and failing runs
# --------------------------------------------------------------------------


def test_no_candidates_is_a_valid_empty_result():
    betas, locations = make_dataset(
        seed=20260920,
        n_samples=40,
        n_blocks=0,
        missing_fraction=0.0,
        chromosomes=(("chr1", 3_300), ("chr2", 3_050)),
    )
    run = call_coordinated_regions(betas, locations, config=_config())
    assert run.manifest["status"] == "no_candidates"
    assert len(run.regions) == 0
    assert list(run.regions.columns[: len(PRIMARY_COLUMNS)]) == PRIMARY_COLUMNS
    assert run.manifest["permutations"]["completed"] == 0
    assert (
        run.manifest["permutations"]["not_run_reason"]
        == "no observed candidate regions"
    )
    assert run.manifest["results"]["n_selected"] == 0
    assert len(run.discoveries) == 0


def test_discovery_only_leaves_qvalues_missing_not_false():
    betas, locations = make_dataset(**LEGACY_MIXED)
    run = call_coordinated_regions(
        betas, locations, config=_config(LEGACY, discovery_only=True)
    )
    assert run.manifest["status"] == "discovery_only"
    assert len(run.regions) > 0
    assert run.regions["coordination_qvalue"].isna().all()
    assert run.regions["coordination_selected"].isna().all()
    assert run.manifest["permutations"]["completed"] == 0
    assert run.manifest["results"]["n_selected"] is None


def test_a_failed_permutation_scan_aborts_the_run(monkeypatch):
    betas, locations = make_dataset(**LEGACY_MIXED)
    real = coordination.scan_by_chromosome
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:  # observed scan + 2 permutations succeed
            raise MemoryError("simulated")
        return real(*args, **kwargs)

    monkeypatch.setattr(coordination, "scan_by_chromosome", flaky)
    with pytest.raises(PermutationScanError, match="permutation scan 3 of 9"):
        call_coordinated_regions(betas, locations, config=_config(LEGACY))


def test_lag_infeasible_input_is_an_error_with_a_reason_not_a_fallback():
    betas, locations = make_dataset(
        seed=5,
        n_samples=12,
        chromosomes=(("chr1", 2_000), ("chr2", 1_500)),
    )
    with pytest.raises(
        CandidateThresholdError, match="distant-lag candidate threshold"
    ):
        call_coordinated_regions(betas, locations, config=_config())
    # the legacy fixed threshold needs no lag null
    run = call_coordinated_regions(betas, locations, config=_config(LEGACY))
    assert run.manifest["status"] in {"ok", "no_candidates"}


def test_invalid_settings_are_rejected():
    for bad in (
        {"fdr_replicates": 0},
        {"target_fdr": 0.0},
        {"mask_mode": "rows"},
        {"scan_mode": "stream"},
        {"axis_mode": "on"},
    ):
        with pytest.raises(ValueError):
            _config(**bad).validate()
    with pytest.raises(ValueError, match="bridges"):
        replace(NEWER, max_gap_edges=1).validate()
    with pytest.raises(ValueError, match="deflation"):
        replace(NEWER, deflate_components=None).validate()
    assert NEWER.is_reference_default() and LEGACY.is_reference_default()
    assert not replace(NEWER, window=3).is_reference_default()
    assert LEGACY.cov_threshold == 0.01


# --------------------------------------------------------------------------
# seeds, chunking, streaming
# --------------------------------------------------------------------------


def test_seed_schedule_and_determinism():
    betas, locations = make_dataset(**LEGACY_MIXED)
    first = call_coordinated_regions(betas, locations, config=_config(LEGACY))
    again = call_coordinated_regions(betas, locations, config=_config(LEGACY))
    pd.testing.assert_frame_equal(first.regions, again.regions)
    pd.testing.assert_frame_equal(first.permutations, again.permutations)
    assert first.permutations["internal_seed"].tolist() == list(range(2, 11))
    other = call_coordinated_regions(
        betas, locations, config=_config(LEGACY, permutation_seed=2)
    )
    np.testing.assert_array_equal(
        other.regions["region_score"], first.regions["region_score"]
    )
    assert not np.array_equal(
        np.concatenate(other.null_scores), np.concatenate(first.null_scores)
    )


def test_newer_internal_seeds_follow_the_reference_schedule():
    betas, locations = make_dataset(**NEWER_MIXED)
    run = call_coordinated_regions(betas, locations, config=_config(fdr_replicates=3))
    rng = np.random.default_rng(1)
    for b, row in enumerate(run.permutations.itertuples(), start=1):
        permuted = permute_rows(betas, rng)
        expected = scan(permuted, locations, NEWER, seed=1 + b)
        assert row.internal_seed == 1 + b
        assert row.scan_threshold == expected.threshold


@pytest.mark.parametrize("mask_mode", ["values_with_mask", "mask_fixed"])
def test_chunked_permutation_is_draw_for_draw_the_whole_matrix_permutation(mask_mode):
    rng = np.random.default_rng(3)
    x = rng.random((1_003, 11))
    x[rng.random(x.shape) < 0.2] = np.nan
    x[:250] = rng.random((250, 11))  # a fully observed first chunk
    whole = [permute_rows(x, g, mask_mode) for g in [np.random.default_rng(7)] * 3]
    for chunk in (1, 97, 250, 5_000):
        g = np.random.default_rng(7)
        out = np.empty_like(x)
        for expected in whole:
            permute_rows_into(x, out, g, mask_mode, chunk_rows=chunk)
            np.testing.assert_array_equal(out, expected)


def test_permutation_preserves_each_rows_values_and_mask_rule():
    rng = np.random.default_rng(4)
    x = rng.random((50, 9))
    x[rng.random(x.shape) < 0.3] = np.nan
    vwm = permute_rows(x, np.random.default_rng(1), "values_with_mask")
    fixed = permute_rows(x, np.random.default_rng(1), "mask_fixed")
    for row in range(x.shape[0]):
        for out in (vwm, fixed):
            np.testing.assert_array_equal(
                np.sort(out[row][np.isfinite(out[row])]),
                np.sort(x[row][np.isfinite(x[row])]),
            )
    np.testing.assert_array_equal(np.isnan(fixed), np.isnan(x))


def test_pipeline_is_invariant_to_permutation_chunk_size():
    betas, locations = make_dataset(**LEGACY_MIXED)
    runs = [
        call_coordinated_regions(
            betas, locations, config=_config(LEGACY, permutation_chunk_rows=rows)
        )
        for rows in (13, 1_000, 1_000_000)
    ]
    for other in runs[1:]:
        pd.testing.assert_frame_equal(runs[0].regions, other.regions)
        np.testing.assert_array_equal(
            np.concatenate(runs[0].null_scores), np.concatenate(other.null_scores)
        )


@pytest.mark.parametrize("profile", [NEWER, LEGACY], ids=["newer", "legacy"])
def test_chromosome_streaming_matches_the_whole_matrix_scan(profile):
    betas, locations = make_dataset(**NEWER_MIXED)
    whole = scan(betas, locations, profile, seed=1, keep_statistic=True)
    streamed = scan_by_chromosome(
        lambda a, b: np.asarray(betas[a:b], float),
        locations,
        profile,
        seed=1,
        keep_statistic=True,
    )
    assert streamed.threshold == whole.threshold
    np.testing.assert_array_equal(streamed.regions, whole.regions)
    np.testing.assert_allclose(streamed.areas, whole.areas, rtol=0, atol=1e-10)
    same = locations["chr"].to_numpy()[:-1] == locations["chr"].to_numpy()[1:]
    np.testing.assert_allclose(
        streamed.statistic[same], whole.statistic[same], rtol=0, atol=1e-10
    )
    # the inter-chromosome edge is never part of a region
    assert np.all(np.isnan(streamed.statistic[~same]))


def test_float32_storage_uses_the_same_permutations():
    betas, locations = make_dataset(**LEGACY_MIXED)
    a = call_coordinated_regions(betas, locations, config=_config(LEGACY))
    b = call_coordinated_regions(
        betas, locations, config=_config(LEGACY, storage_dtype="float32")
    )
    assert b.manifest["input"]["storage_dtype"] == "float32"
    # float32 rounding may move scores slightly; the candidates stay put here
    np.testing.assert_array_equal(
        a.regions[["cpg_index_start", "cpg_index_end"]],
        b.regions[["cpg_index_start", "cpg_index_end"]],
    )


# --------------------------------------------------------------------------
# one decision family; geometry and ordering
# --------------------------------------------------------------------------


def test_one_genome_wide_decision_family():
    betas, locations = make_dataset(**LEGACY_MIXED)
    run = call_coordinated_regions(betas, locations, config=_config(LEGACY))
    pooled = coordination_fdr(run.regions["region_score"].to_numpy(), run.null_scores)
    np.testing.assert_array_equal(run.regions["coordination_qvalue"], pooled.qvalue)
    assert set(run.regions["chr"]) == {"chr1", "chr2"}
    np.testing.assert_array_equal(
        run.regions["coordination_selected"].to_numpy(dtype=bool),
        pooled.qvalue <= 0.05,
    )


def test_row_order_of_the_input_does_not_change_the_result():
    betas, locations = make_dataset(**LEGACY_MIXED)
    shuffled_betas, shuffled_locations = make_dataset(**LEGACY_MIXED, shuffle_rows=True)
    a = call_coordinated_regions(betas, locations, config=_config(LEGACY))
    b = call_coordinated_regions(
        shuffled_betas, shuffled_locations, config=_config(LEGACY)
    )
    pd.testing.assert_frame_equal(a.regions, b.regions)
    assert b.manifest["input"]["input_order_changed_by_sort"] is True


def _block_matrix(n_samples=12, seed=0):
    """Two chromosomes of 30 CpGs; rows 5-24 of each share one latent value."""

    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, n_samples)
    betas = 0.5 + 0.02 * rng.normal(0, 1, (60, n_samples))
    betas[5:25] += 0.2 * latent
    betas[35:55] += 0.2 * latent
    return np.clip(betas, 0, 1)


def _locations(steps):
    starts = np.concatenate([[1_000], 1_000 + np.cumsum(steps)])
    chrom = ["chr1"] * 30 + ["chr2"] * 30
    return pd.DataFrame({"chr": chrom, "start": starts, "end": starts + 1})


def test_gap_and_chromosome_boundaries_split_regions():
    betas = _block_matrix()
    steps = np.full(59, 50)
    steps[14] = 300 + 1  # gap between CpG 14 and 15 is exactly 300 bp
    steps[44] = 301 + 1  # gap between CpG 44 and 45 is 301 bp
    regions = scan(betas, _locations(steps), LEGACY).regions.tolist()
    assert [5, 24] in regions  # 300 bp gap is eligible
    assert [35, 44] in regions and [45, 54] in regions  # 301 bp gap splits
    # a block continued across the chromosome boundary is split there
    betas2 = _block_matrix()
    betas2[25:35] = betas2[5:15]
    regions2 = scan(betas2, _locations(np.full(59, 50)), LEGACY).regions
    assert not np.any((regions2[:, 0] <= 29) & (regions2[:, 1] >= 30))


def test_overlapping_coordinates_are_not_eligible_edges():
    betas = _block_matrix()
    steps = np.full(59, 50)
    steps[9] = 0  # next.start == previous.start: gap of -1 bp
    regions = scan(betas, _locations(steps), LEGACY).regions.tolist()
    assert [5, 9] in regions and [10, 24] in regions


# --------------------------------------------------------------------------
# missingness
# --------------------------------------------------------------------------


def test_legacy_imputation_uses_row_means_and_one_half_for_empty_rows():
    x = np.array([[0.2, np.nan, 0.4], [np.nan, np.nan, np.nan]])
    np.testing.assert_allclose(_mean_impute(x), [[0.2, 0.3, 0.4], [0.5, 0.5, 0.5]])


def test_all_missing_rows_are_kept_and_counted_not_dropped():
    betas, locations = make_dataset(**LEGACY_MIXED)
    betas[100:103] = np.nan
    run = call_coordinated_regions(betas, locations, config=_config(LEGACY))
    assert run.manifest["input"]["n_cpg_analysed"] == betas.shape[0]
    assert run.manifest["input"]["n_cpg_all_missing_analysed"] == 3
    assert "0.5" in run.manifest["missingness"]["statistic"]


def test_newer_edges_need_five_co_observed_samples():
    rng = np.random.default_rng(2)
    betas = rng.random((6_200, 8))
    locations = pd.DataFrame(
        {
            "chr": ["chr1"] * 6_200,
            "start": np.arange(6_200) * 20,
            "end": np.arange(6_200) * 20 + 1,
        }
    )
    betas[3_000, :4] = np.nan  # 4 observed samples on this row
    result = scan(betas, locations, replace(NEWER, window=1), keep_statistic=True)
    assert np.isnan(result.statistic[2_999]) and np.isnan(result.statistic[3_000])
    assert np.isfinite(result.statistic[2_998])


def test_min_sample_fraction_filter_keeps_indices_relative_to_sorted_input():
    betas, locations = make_dataset(**LEGACY_MIXED)
    betas[0:50, :6] = np.nan
    run = call_coordinated_regions(
        betas, locations, config=_config(LEGACY, min_sample_fraction=0.5)
    )
    assert run.manifest["input"]["n_cpg_dropped_min_sample_fraction"] == 50
    assert run.regions["cpg_index_start"].min() >= 50


# --------------------------------------------------------------------------
# axis annotations are secondary
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def axis_runs():
    betas, locations = make_dataset(**AXIS_DATA)
    return {
        mode: call_coordinated_regions(
            betas, locations, config=_config(fdr_replicates=5, axis_mode=mode)
        )
        for mode in ("off", "descriptive", "experimental_inference")
    }


def test_axis_annotations_never_change_primary_calls(axis_runs):
    base = axis_runs["off"]
    for mode in ("descriptive", "experimental_inference"):
        run = axis_runs[mode]
        pd.testing.assert_frame_equal(
            run.regions[PRIMARY_COLUMNS], base.regions[PRIMARY_COLUMNS]
        )
        pd.testing.assert_frame_equal(run.permutations, base.permutations)
        np.testing.assert_array_equal(
            np.concatenate(run.null_scores), np.concatenate(base.null_scores)
        )
        for key in ("observed_scan", "permutations", "results", "threshold_ratio"):
            assert run.manifest[key] == base.manifest[key]


def test_default_axis_annotations_are_descriptive_and_leave_inference_missing(
    axis_runs,
):
    regions = axis_runs["descriptive"].regions
    assert (regions["axis_status"] == "estimated").all()
    assert regions["axis_effect"].notna().all()
    assert regions["axis_n_samples_tested"].min() >= AXIS_MIN_SAMPLES_FOR_INFERENCE
    assert regions["axis_pvalue"].isna().all()
    assert regions["axis_qvalue"].isna().all()
    assert regions["axis_supported"].isna().all()  # missing, never False
    manifest = axis_runs["descriptive"].manifest["axis"]
    assert manifest["inference_reported"] is False
    assert manifest["affects_primary_calls"] is False


def test_experimental_axis_inference_is_labelled_and_uses_the_candidate_family(
    axis_runs,
):
    run = axis_runs["experimental_inference"]
    regions = run.regions
    manifest = run.manifest["axis"]
    assert manifest["inference_label"].startswith("EXPERIMENTAL")
    assert manifest["family"]["n_regions"] == len(regions)
    assert (regions["axis_status"] == "tested").all()
    from eiger2.axis import benjamini_hochberg

    np.testing.assert_array_equal(
        regions["axis_qvalue"], benjamini_hochberg(regions["axis_pvalue"].to_numpy())
    )
    # effects agree with the descriptive run: one fit, two dispositions
    np.testing.assert_array_equal(
        regions["axis_effect"], axis_runs["descriptive"].regions["axis_effect"]
    )


def test_experimental_inference_is_missing_below_the_tested_sample_floor():
    betas, locations = make_dataset(**{**AXIS_DATA, "n_samples": 150, "seed": 32})
    run = call_coordinated_regions(
        betas,
        locations,
        config=_config(fdr_replicates=3, axis_mode="experimental_inference"),
    )
    regions = run.regions
    assert (regions["axis_status"] == "below_inference_floor").all()
    assert regions["axis_effect"].notna().all()
    assert regions["axis_pvalue"].isna().all() and regions["axis_qvalue"].isna().all()
    assert regions["axis_supported"].isna().all()


def test_no_usable_axis_is_reported_as_missing_not_false():
    betas, locations = make_dataset(**LEGACY_MIXED)
    run = call_coordinated_regions(
        betas,
        locations,
        config=_config(LEGACY, axis_mode="experimental_inference"),
    )
    regions = run.regions
    assert (regions["axis_status"] == "no_axes").all()
    assert regions["axis_effect"].isna().all()
    assert regions["axis_supported"].isna().all()
    off = call_coordinated_regions(betas, locations, config=_config(LEGACY))
    assert (off.regions["axis_status"] == "not_computed").all()


def test_the_callers_matrix_is_never_modified():
    # canonical order and no filter: the pipeline reads the caller's array
    # directly instead of copying it, so nothing may write to it
    betas, locations = make_dataset(**LEGACY_MIXED)
    before = betas.copy()
    call_coordinated_regions(
        betas, locations, config=_config(LEGACY, axis_mode="descriptive")
    )
    np.testing.assert_array_equal(betas, before)
