from __future__ import annotations

import gzip
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO
for _p in (REPO, REPO / "old_code", REPO / "old_code" / "beta_2026", REPO / "Python"):
    sys.path.insert(0, str(_p))

from functions.load_functions import (  # noqa: E402
    beta_site_count,
    infer_methylation_file_type,
    load_beta_values,
    load_bismark_coverage_betas,
)


def _locations():
    return pd.DataFrame(
        [
            ["chr1", 99, 1],
            ["chr1", 109, 2],
            ["chr2", 199, 3],
            ["chrM", 49, 4],
        ]
    )


def _write_coverage(path: Path, rows: list[str]):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as handle:
        handle.write("\n".join(rows) + "\n")


def test_input_type_inference_handles_compressed_coverage():
    assert infer_methylation_file_type(["a.beta", "b.lbeta"]) == "beta"
    assert infer_methylation_file_type(["a.cov", "b.cov.gz"]) == "cov"

    with pytest.raises(ValueError, match="same input type"):
        infer_methylation_file_type(["a.beta", "b.cov.gz"])


def test_beta_value_loader_supports_memory_efficient_slices():
    with tempfile.TemporaryDirectory() as directory:
        first_path = Path(directory) / "first.beta"
        second_path = Path(directory) / "second.beta"
        np.array([[1, 2], [0, 0], [3, 4]], dtype=np.uint8).tofile(first_path)
        np.array([[2, 2], [1, 4], [0, 5]], dtype=np.uint8).tofile(second_path)

        values = load_beta_values(
            [first_path, second_path],
            slice_start=1,
            slice_end=3,
        )
        values_with_counts, total_counts = load_beta_values(
            [first_path, second_path],
            slice_start=1,
            slice_end=3,
            return_total_counts=True,
        )
        site_count = beta_site_count(first_path)

    assert site_count == 3
    assert values.dtype == np.float32
    np.testing.assert_allclose(values[1], [0.75, 0.0])
    assert np.isnan(values[0, 0])
    assert values[0, 1] == 0.25
    np.testing.assert_allclose(values_with_counts, values, equal_nan=True)
    np.testing.assert_array_equal(total_counts, [[0, 4], [4, 5]])


def test_bismark_cov_gz_loading_aligns_and_combines_counts():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.cov.gz"
        _write_coverage(
            path,
            [
                "1\t100\t100\t75\t3\t1",
                "chr1\t110\t110\t0\t0\t4",
                "2\t200\t200\t100\t2\t0",
                "MT\t50\t50\t25\t1\t3",
                "1\t100\t100\t100\t1\t0",
            ],
        )
        betas, summaries = load_bismark_coverage_betas(
            [path],
            _locations(),
            chunk_size=2,
        )
        betas_with_counts, _, total_counts = load_bismark_coverage_betas(
            [path],
            _locations(),
            chunk_size=2,
            return_total_counts=True,
        )

    np.testing.assert_allclose(betas[:, 0], [0.8, 0.0, 1.0, 0.25])
    assert summaries[0]["coordinate_system"] == "one-based"
    assert summaries[0]["match_rate"] == 1.0
    assert summaries[0]["observed_cpgs"] == 4
    np.testing.assert_allclose(betas_with_counts, betas)
    np.testing.assert_array_equal(total_counts[:, 0], [5, 4, 2, 4])


def test_zero_based_coverage_is_auto_detected():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.cov"
        _write_coverage(
            path,
            [
                "chr1\t98\t99\t50\t1\t1",
                "chr1\t108\t109\t25\t1\t3",
            ],
        )
        betas, summaries = load_bismark_coverage_betas([path], _locations())

    np.testing.assert_allclose(betas[:2, 0], [0.5, 0.25])
    assert np.isnan(betas[2:, 0]).all()
    assert summaries[0]["coordinate_system"] == "zero-based"


def test_reference_mismatch_fails_early():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.cov.gz"
        _write_coverage(path, ["chr9\t999\t999\t50\t1\t1"])

        with pytest.raises(ValueError, match="genome build"):
            load_bismark_coverage_betas(
                [path],
                _locations(),
                minimum_match_rate=0.5,
            )
