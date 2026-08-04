from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO
for _p in (REPO, REPO / "old_code", REPO / "old_code" / "beta_2026", REPO / "Python"):
    sys.path.insert(0, str(_p))

from beta_2026.eiger_core import (  # noqa: E402
    COMPACT_COLUMNS,
    DIAGNOSTIC_COLUMNS,
    EigerConfig,
    call_dmrs_from_matrix,
    combine_dmr_results,
    estimate_covariance_threshold,
    estimate_covariance_threshold_from_pairs,
    covariance_edge_support,
    smooth_edge_support,
    impute_for_covariance,
    _probabilistic_separation_from_values,
    _standardise_locations,
)
from beta_2026.legacy.eiger_hard_threshold_v1 import (  # noqa: E402
    LEGACY_ALGORITHM_VERSION,
    call_dmrs_from_matrix as call_dmrs_legacy,
)
from functions.load_functions import load_beta_frequencies  # noqa: E402
from functions.scoring import (  # noqa: E402
    benjamini_hochberg,
    prevalence_variance_weight,
)


def _example():
    base = np.array([0.10, 0.12, 0.11, 0.78, 0.80, np.nan])
    betas = np.vstack([base + offset for offset in np.linspace(0.0, 0.05, 6)])
    locations = pd.DataFrame(
        {
            "chr": ["chr1"] * 6,
            "start": np.arange(100, 160, 10),
            "end": np.arange(101, 161, 10),
        }
    )
    samples = [f"s{i}" for i in range(1, 7)]
    config = EigerConfig(
        min_cpgs=4,
        covariance_threshold=0.01,
        delta_threshold=0.2,
        laplacian_neighbors=2,
    )
    return betas, locations, samples, config


def test_core_is_compact_and_annotation_is_optional():
    betas, locations, samples, config = _example()
    core, values, annotations = call_dmrs_from_matrix(
        betas, locations, sample_names=samples, config=config
    )

    assert len(core) == 1
    assert core["dmr_id"].is_unique
    assert annotations.empty
    assert "celltype" not in core.columns
    assert list(values.columns) == ["dmr_id", *samples]
    assert np.isnan(values.loc[0, "s6"])
    assert core.loc[0, "group_annotation_status"] == "not_requested"
    assert len(COMPACT_COLUMNS) < len(core.columns)
    assert set(COMPACT_COLUMNS).issubset(core.columns)
    assert set(DIAGNOSTIC_COLUMNS).issubset(core.columns)


def test_any_metadata_group_can_be_annotated_long_form():
    betas, locations, samples, config = _example()
    metadata = pd.DataFrame(
        {
            "sample_id": samples,
            "celltype": ["epithelial"] * 3 + ["stromal"] * 3,
            "batch": ["A", "A", "B", "A", "B", "B"],
        }
    )
    core, _, annotations = call_dmrs_from_matrix(
        betas,
        locations,
        sample_names=samples,
        metadata=metadata,
        sample_column="sample_id",
        group_columns=["celltype", "batch"],
        config=config,
    )

    assert set(annotations["grouping_variable"]) == {"celltype", "batch"}
    assert set(annotations["dmr_id"]) == set(core["dmr_id"])
    assert "celltype" not in core.columns
    celltype = annotations.loc[
        annotations["grouping_variable"] == "celltype"
    ].iloc[0]
    assert celltype["passes_delta_threshold"]
    assert {
        celltype["target_groups"],
        celltype["reference_groups"],
    } == {"epithelial", "stromal"}


def test_blocks_do_not_cross_chromosomes():
    betas, locations, samples, config = _example()
    betas = np.vstack([betas[:4], betas[:4]])
    locations = pd.DataFrame(
        {
            "chr": ["chr1"] * 4 + ["chr2"] * 4,
            "start": [100, 110, 120, 130, 100, 110, 120, 130],
            "end": [101, 111, 121, 131, 101, 111, 121, 131],
        }
    )
    core, _, _ = call_dmrs_from_matrix(
        betas, locations, sample_names=samples, config=config
    )

    assert len(core) == 2
    assert core.groupby("dmr_id")["chr"].nunique().eq(1).all()
    assert set(core["chr"]) == {"chr1", "chr2"}


def test_genomic_gap_can_bridge_a_coverage_filtered_cpg():
    values = np.array([0.05, 0.12, 0.20, 0.72, 0.80, 0.88])
    betas = np.tile(values, (5, 1))
    betas[2, :] = np.nan
    locations = pd.DataFrame(
        {
            "chr": ["chr1"] * 5,
            "start": [100, 110, 120, 130, 140],
            "end": [101, 111, 121, 131, 141],
        }
    )
    common = {
        "min_cpgs": 4,
        "covariance_threshold": 0.001,
        "delta_threshold": 0.2,
        "min_observed_samples": 6,
        "laplacian_neighbors": 2,
    }

    bridged, _, _ = call_dmrs_from_matrix(
        betas,
        locations,
        config=EigerConfig(**common, max_cpg_gap_bp=20),
    )
    short_gap, _, _ = call_dmrs_from_matrix(
        betas,
        locations,
        config=EigerConfig(**common, max_cpg_gap_bp=10),
    )
    strict, _, _ = call_dmrs_from_matrix(
        betas,
        locations,
        config=EigerConfig(**common, max_cpg_gap_bp=None),
    )

    assert len(bridged) == 1
    assert bridged.loc[0, "n_cpg"] == 4
    assert bridged.loc[0, "cpg_start"] == 0
    assert bridged.loc[0, "cpg_end"] == 4
    assert short_gap.empty
    assert strict.empty


def test_headerless_third_location_column_is_not_treated_as_end():
    locations = pd.DataFrame(
        [
            ["chr1", 10469, 1],
            ["chr1", 10471, 2],
        ]
    )
    standardised = _standardise_locations(locations, n_rows=2)

    assert standardised["start"].tolist() == [10469, 10471]
    assert standardised["end"].tolist() == [10470, 10472]


def test_named_location_end_is_respected():
    locations = pd.DataFrame(
        {
            "chr": ["chr1", "chr1"],
            "start": [100, 110],
            "end": [102, 112],
        }
    )
    standardised = _standardise_locations(locations, n_rows=2)

    assert standardised["end"].tolist() == [102, 112]


def test_automatic_covariance_threshold_is_label_free_and_reproducible():
    betas, locations, samples, _ = _example()
    covariance_input = impute_for_covariance(betas)
    eligible = np.ones(len(betas) - 1, dtype=bool)
    first = estimate_covariance_threshold(
        covariance_input,
        eligible,
        quantile=0.95,
        permutations=8,
        seed=7,
    )
    second = estimate_covariance_threshold(
        covariance_input,
        eligible,
        quantile=0.95,
        permutations=8,
        seed=7,
    )
    assert np.isfinite(first.threshold)
    assert first == second

    config = EigerConfig(
        min_cpgs=4,
        covariance_threshold_method="auto",
        covariance_null_quantile=0.95,
        covariance_null_permutations=8,
        delta_threshold=0.2,
        random_seed=7,
    )
    core, _, _ = call_dmrs_from_matrix(
        betas, locations, sample_names=samples, config=config
    )
    assert core.attrs["covariance_threshold_method"] == "auto"
    assert core.attrs["covariance_threshold_used"] == first.threshold
    if not core.empty:
        assert core["covariance_threshold_used"].eq(first.threshold).all()


def test_pair_reservoir_threshold_is_reproducible():
    rng = np.random.default_rng(3)
    left = rng.normal(size=(20, 6)).astype(np.float32)
    right = rng.normal(size=(20, 6)).astype(np.float32)
    first = estimate_covariance_threshold_from_pairs(
        left,
        right,
        quantile=0.95,
        permutations=4,
        max_pairs=10,
        seed=7,
        batch_size=3,
    )
    second = estimate_covariance_threshold_from_pairs(
        left,
        right,
        quantile=0.95,
        permutations=4,
        max_pairs=10,
        seed=7,
        batch_size=3,
    )
    assert first == second
    assert first.null_pairs_sampled == 40


def test_chromosome_results_reproduce_genome_wide_finalisation():
    betas, _, samples, config = _example()
    betas = np.vstack([betas, betas])
    locations = pd.DataFrame(
        {
            "chr": ["chr1"] * 6 + ["chr2"] * 6,
            "start": list(np.arange(100, 160, 10)) * 2,
            "end": list(np.arange(101, 161, 10)) * 2,
        }
    )
    full_core, full_values, _ = call_dmrs_from_matrix(
        betas,
        locations,
        sample_names=samples,
        config=config,
    )

    core_frames = []
    value_frames = []
    for start, end in ((0, 6), (6, 12)):
        core, values, _ = call_dmrs_from_matrix(
            betas[start:end],
            locations.iloc[start:end],
            sample_names=samples,
            config=config,
            source_cpg_offset=start,
        )
        core_frames.append(core)
        value_frames.append(values)
    combined_core, combined_values, _ = combine_dmr_results(
        core_frames,
        value_frames,
        sample_names=samples,
        config=config,
    )

    pd.testing.assert_frame_equal(
        full_core.reset_index(drop=True),
        combined_core.reset_index(drop=True),
        check_dtype=False,
        rtol=1e-10,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(full_values, combined_values)


def test_delta_threshold_uses_cluster_mean_difference():
    values = np.array([0.10, 0.20, 0.30, 0.45, 0.55, 0.65])
    betas = np.tile(values, (6, 1))
    locations = pd.DataFrame(
        {
            "chr": ["chr1"] * 6,
            "start": np.arange(100, 160, 10),
            "end": np.arange(101, 161, 10),
        }
    )
    core, _, _ = call_dmrs_from_matrix(
        betas,
        locations,
        config=EigerConfig(
            min_cpgs=4,
            covariance_threshold=0.001,
            delta_threshold=0.2,
        ),
    )

    assert len(core) == 1
    assert core.loc[0, "delta_minmax"] < 0.2
    assert abs(core.loc[0, "delta_beta"]) >= 0.2
    assert core.loc[0, "passes_delta_threshold"]


def test_bh_and_bounded_prevalence_variance_weight():
    adjusted = benjamini_hochberg([0.04, 0.01, np.nan, 0.03])
    np.testing.assert_allclose(adjusted[[0, 1, 3]], [0.04, 0.03, 0.04])
    assert np.isnan(adjusted[2])

    expected = np.sqrt((0.25 * 0.75) / (0.10 * 0.90))
    assert np.isclose(
        prevalence_variance_weight(0.10, 0.25), expected
    )
    assert prevalence_variance_weight(
        1e-12, 0.25, minimum=0.25, maximum=2.0
    ) == 2.0


def test_lbeta_loader_preserves_16_bit_counts():
    expected = np.array([[300, 500], [1024, 2048]], dtype=np.uint16)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.lbeta"
        expected.tofile(path)
        observed = load_beta_frequencies(path)
        sliced = load_beta_frequencies(path, slice_start=1, slice_end=2)

    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(sliced, expected[1:2])


def test_legacy_snapshot_matches_explicit_legacy_mode():
    betas, locations, samples, config = _example()
    explicit = EigerConfig(**{**config.to_dict(), "calling_method": "legacy"})
    expected_core, expected_values, _ = call_dmrs_from_matrix(
        betas,
        locations,
        sample_names=samples,
        config=explicit,
    )
    snapshot_core, snapshot_values, _ = call_dmrs_legacy(
        betas,
        locations,
        sample_names=samples,
        config=config,
    )

    assert LEGACY_ALGORITHM_VERSION == "eiger-hard-threshold-v1"
    pd.testing.assert_frame_equal(expected_core, snapshot_core)
    pd.testing.assert_frame_equal(expected_values, snapshot_values)
    assert snapshot_core["segmentation_method"].eq("legacy").all()
    frozen = snapshot_core.iloc[0]
    assert (frozen["chr"], frozen["start"], frozen["end"]) == (
        "chr1",
        100,
        151,
    )
    assert (frozen["cpg_start"], frozen["cpg_end"], frozen["n_cpg"]) == (
        0,
        5,
        6,
    )
    assert np.isclose(frozen["covariance_score"], 0.111056)
    assert np.isclose(frozen["delta_beta"], 0.68)
    assert frozen["sample_cluster_size"] == 2
    assert frozen["call_status"] == "sample_structure"


def test_probabilistic_edge_support_retains_covariance_threshold_semantics():
    covariance = np.array([0.020, 0.011, 0.021])
    standard_error = np.array([0.002, 0.003, 0.002])
    support = covariance_edge_support(covariance, standard_error, 0.012)
    posterior = smooth_edge_support(
        support,
        np.ones(3, dtype=bool),
        signal_prior=0.05,
        signal_persistence=0.95,
    )

    assert support[0] > 0.99
    assert support[1] < 0.5
    assert support[2] > 0.99
    assert posterior[1] > support[1]


def test_probabilistic_and_legacy_modes_use_the_same_covariance_signal():
    betas, locations, samples, config = _example()
    probabilistic_core, _, _ = call_dmrs_from_matrix(
        betas,
        locations,
        sample_names=samples,
        config=EigerConfig(**{**config.to_dict(), "calling_method": "probabilistic"}),
    )
    legacy_core, _, _ = call_dmrs_from_matrix(
        betas,
        locations,
        sample_names=samples,
        config=EigerConfig(**{**config.to_dict(), "calling_method": "legacy"}),
    )

    assert probabilistic_core[["chr", "start", "end"]].equals(
        legacy_core[["chr", "start", "end"]]
    )
    np.testing.assert_allclose(
        probabilistic_core["covariance_score"],
        legacy_core["covariance_score"],
    )
    np.testing.assert_allclose(
        probabilistic_core["covariance_threshold_used"],
        legacy_core["covariance_threshold_used"],
    )


def test_read_coverage_changes_confidence_but_not_covariance():
    betas, locations, samples, config = _example()
    observed = np.isfinite(betas)
    low_coverage = np.where(observed, 2, 0)
    high_coverage = np.where(observed, 200, 0)
    low_core, _, _ = call_dmrs_from_matrix(
        betas,
        locations,
        total_counts=low_coverage,
        sample_names=samples,
        config=config,
    )
    high_core, _, _ = call_dmrs_from_matrix(
        betas,
        locations,
        total_counts=high_coverage,
        sample_names=samples,
        config=config,
    )

    np.testing.assert_allclose(
        low_core["covariance_score"], high_core["covariance_score"]
    )
    assert low_core.loc[0, "edge_support_mean"] <= high_core.loc[
        0, "edge_support_mean"
    ]
    assert low_core.loc[0, "coverage_uncertainty_used"]
    assert low_core.loc[0, "mean_read_coverage"] == 2


def test_probabilistic_sample_model_rejects_gradient_and_singleton():
    config = EigerConfig(
        sample_probability_threshold=0.8,
        min_sample_cluster_size=2,
    )
    gradient = _probabilistic_separation_from_values(
        np.linspace(0.0, 1.0, 20), config
    )
    singleton = _probabilistic_separation_from_values(
        np.array([0.10, 0.11, 0.12, 0.13, 0.14, 0.75]), config
    )
    clusters = _probabilistic_separation_from_values(
        np.array([0.10, 0.11, 0.12, 0.75, 0.76, 0.77]), config
    )

    assert not gradient["call_supported"]
    assert not singleton["call_supported"]
    assert singleton["minor_n"] == 1
    assert clusters["call_supported"]
    assert clusters["call_probability"] > 0.99


def test_probabilistic_bootstrap_stability_is_a_recurrence_probability():
    betas, locations, samples, config = _example()
    config = EigerConfig(
        **{
            **config.to_dict(),
            "calling_method": "probabilistic",
            "stability_bootstraps": 3,
            "sample_probability_threshold": 0.5,
        }
    )
    core, _, _ = call_dmrs_from_matrix(
        betas,
        locations,
        sample_names=samples,
        config=config,
    )

    assert core["stability_score"].between(0.0, 1.0).all()
