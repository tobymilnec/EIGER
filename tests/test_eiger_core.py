from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Python.eiger_core import (  # noqa: E402
    COMPACT_COLUMNS,
    DIAGNOSTIC_COLUMNS,
    EigerConfig,
    call_dmrs_from_matrix,
    estimate_covariance_threshold,
    impute_for_covariance,
)
from Python.functions.load_functions import load_beta_frequencies  # noqa: E402
from Python.functions.scoring import (  # noqa: E402
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
