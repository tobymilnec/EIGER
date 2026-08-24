"""Post-selection five-class biological-support layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eiger2.core import (
    Eiger2Config,
    biological_support_diagnostics,
    call_dmrs,
    characterise_region,
    classify_biological_support,
    main_region_table,
    region_minority_size,
)


def _config(statistic: str = "correlation") -> Eiger2Config:
    return Eiger2Config(
        statistic=statistic, deflate_components=0, min_paired_samples=3
    )


def _support(block: np.ndarray, statistic: str = "correlation") -> dict:
    means = np.nanmean(block, axis=0)
    centred = block - np.nanmean(block, axis=1, keepdims=True)
    return biological_support_diagnostics(
        block, centred, means, _config(statistic)
    )


def test_support_policy_is_explicit_opt_in_and_validated():
    assert Eiger2Config().biological_support_policy == "off"
    assert Eiger2Config(
        biological_support_policy="five_class_v1"
    ).biological_support_policy == "five_class_v1"
    with pytest.raises(ValueError, match="biological_support_policy"):
        Eiger2Config(biological_support_policy="unknown")


def test_leave_top_one_out_score_matches_direct_brute_force_with_missingness():
    rng = np.random.default_rng(10)
    block = rng.normal(size=(7, 18))
    block[:, 3] += 3.0
    block[1, 5] = np.nan
    block[4, 8] = np.nan
    for statistic in ("correlation", "covariance"):
        diagnostics = _support(block, statistic)
        top = diagnostics["furthest_mean_sample_index"]
        reduced = np.delete(block, top, axis=1)
        direct = _support(reduced, statistic)["adjacent_support_score"]
        assert np.isclose(
            diagnostics["adjacent_support_score_without_furthest1"], direct,
            rtol=1e-12, atol=1e-12, equal_nan=True,
        )
        means = np.nanmean(block, axis=0)
        ranked = np.argsort(-np.abs(means - np.nanmedian(means)), kind="mergesort")
        reduced_two = np.delete(block, ranked[:2], axis=1)
        direct_two = _support(reduced_two, statistic)["adjacent_support_score"]
        assert np.isclose(
            diagnostics["adjacent_support_score_without_furthest2"], direct_two,
            rtol=1e-12, atol=1e-12, equal_nan=True,
        )


def test_five_classes_are_exclusive_and_non_calls_are_missing():
    labels = {
        "cohort_supported", "recurrent_subgroup", "rare_epivariant",
        "outlier_sensitive", "diffuse_coordination",
    }
    base = {
        "minority_size": 10, "minority_fraction": 0.20,
        "minority_delta": 0.20, "boundary_gap_fraction_delta": 0.5,
        "minority_direction_coherence": 1.0,
        "furthest1_adjacent_score_influence_fraction": 0.0,
        "winsorised_score_retention": 1.0,
    }
    assert classify_biological_support(
        base, coordination_significant=False, n_observed=50
    ) is None
    observed = set()
    observed.add(classify_biological_support(
        base, coordination_significant=True, n_observed=50
    ))
    observed.add(classify_biological_support(
        {**base, "minority_size": 3, "minority_fraction": 0.06},
        coordination_significant=True, n_observed=50,
    ))
    observed.add(classify_biological_support(
        {**base, "minority_size": 1, "minority_fraction": 0.02},
        coordination_significant=True, n_observed=50,
    ))
    observed.add(classify_biological_support(
        {**base, "minority_size": 1, "minority_fraction": 0.02,
         "minority_direction_coherence": 0.5,
         "furthest1_adjacent_score_influence_fraction": 0.9,
         "winsorised_score_retention": 0.1},
        coordination_significant=True, n_observed=50,
    ))
    observed.add(classify_biological_support(
        {**base, "boundary_gap_fraction_delta": 0.01},
        coordination_significant=True, n_observed=50,
    ))
    assert observed == labels


def test_rare_fraction_is_a_strict_maximum():
    diagnostics = {
        "minority_size": 3,
        "minority_fraction": 3 / 260,
        "minority_delta": 0.20,
        "boundary_gap_fraction_delta": 0.5,
        "minority_direction_coherence": 1.0,
        "furthest1_adjacent_score_influence_fraction": 0.0,
        "winsorised_score_retention": 1.0,
    }
    assert classify_biological_support(
        diagnostics, coordination_significant=True, n_observed=260
    ) == "recurrent_subgroup"


def test_rare_subgroup_and_continuous_gradient_have_expected_descriptions():
    rng = np.random.default_rng(1)
    rare = 0.4 + rng.normal(0, 0.01, (6, 40))
    rare[:, 0] += 0.4
    rare_diag = _support(rare)
    assert classify_biological_support(
        rare_diag, coordination_significant=True, n_observed=40
    ) == "rare_epivariant"
    assert rare_diag["furthest1_adjacent_score_influence_fraction"] > 0.5
    assert rare_diag["winsorised_score_retention"] < 0.5

    subgroup = 0.4 + rng.normal(0, 0.01, (6, 100))
    subgroup[:, :5] += 0.3
    subgroup_diag = _support(subgroup)
    assert classify_biological_support(
        subgroup_diag, coordination_significant=True, n_observed=100
    ) == "recurrent_subgroup"

    gradient = np.tile(np.linspace(0.2, 0.8, 40), (6, 1))
    gradient += rng.normal(0, 0.005, gradient.shape)
    gradient_diag = _support(gradient)
    assert classify_biological_support(
        gradient_diag, coordination_significant=True, n_observed=40
    ) == "diffuse_coordination"


def test_cohort_and_outlier_classes_have_expected_descriptions():
    rng = np.random.default_rng(2)

    # Stable 20% subgroup with a robust delta beta.
    cohort = 0.4 + rng.normal(0, 0.01, (20, 100))
    cohort[:, :20] += 0.25
    cohort_diag = _support(cohort)
    assert classify_biological_support(
        cohort_diag, coordination_significant=True, n_observed=100
    ) == "cohort_supported"
    assert cohort_diag["minority_fraction"] >= 0.10

    # A sign-changing single-sample shift should be outlier-sensitive.
    outlier = 0.4 + rng.normal(0, 0.01, (8, 80))
    outlier[:, 0] += np.array([.5, .5, .5, .5, -.1, -.1, -.1, -.1])
    outlier_diag = _support(outlier)
    assert classify_biological_support(
        outlier_diag, coordination_significant=True, n_observed=80
    ) == "outlier_sensitive"
    assert outlier_diag["minority_direction_coherence"] < 0.80
    assert outlier_diag["furthest1_adjacent_score_influence_fraction"] >= 0.50


def test_largest_gap_tail_is_not_authoritative_two_means_prevalence():
    rng = np.random.default_rng(7)
    lower = rng.normal(0.30, 0.02, 200)
    upper = rng.normal(0.60, 0.02, 200)
    values = np.r_[lower, upper, 0.95]
    matrix = np.tile(values, (5, 1))
    largest_gap = characterise_region(values)
    minority_size, _ = region_minority_size(values, (0, 4), matrix)
    assert largest_gap["minor_group_size"] == 1
    assert minority_size >= 190


def test_main_table_has_only_two_new_support_facing_columns():
    frame = pd.DataFrame({
        "region_id": ["r1"], "region_qvalue": [0.01],
        "coordination_significant": pd.array([True], dtype="boolean"),
        "biological_class": ["cohort_supported"],
        "passes_prevalence": [True],
        "minority_size_descriptive": [10], "minor_group_size_descriptive": [1],
    })
    main = main_region_table(frame)
    assert list(main.columns) == [
        "region_id", "region_qvalue", "coordination_significant",
        "biological_class", "passes_prevalence",
    ]
    assert not any("minor" in column for column in main.columns)


def test_opt_in_empty_run_exposes_header_only_diagnostics():
    matrix = np.full((2, 5), 0.5)
    locations = pd.DataFrame({
        "chr": ["chr1", "chr1"], "start": [0, 1], "end": [1, 2],
    })
    regions, _, manifest = call_dmrs(
        matrix,
        locations,
        config=Eiger2Config(
            min_cpgs=5, biological_support_policy="five_class_v1"
        ),
    )
    diagnostics = regions.attrs["region_support_diagnostics"]
    assert diagnostics.empty
    assert "diagnostic_status" in diagnostics
    assert manifest["biological_support"]["diagnostics_file"] \
        == "region_support_diagnostics.csv"


def test_nullable_coordination_call_survives_csv_round_trip(tmp_path):
    path = tmp_path / "regions.csv"
    frame = pd.DataFrame({
        "coordination_significant": pd.array([True, False, pd.NA], dtype="boolean")
    })
    frame.to_csv(path, index=False)
    restored = pd.read_csv(path, dtype={"coordination_significant": "boolean"})
    assert restored["coordination_significant"].tolist() == [True, False, pd.NA]
