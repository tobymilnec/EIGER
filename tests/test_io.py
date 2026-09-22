"""Bismark ``.cov`` conversion in ``eiger2.io``: union grid, counts, betas."""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from eiger2.io import (  # noqa: E402
    CpGReference,
    build_cpg_reference,
    convert_cov_files,
    load_matrix,
)


def _write_cov(path: Path, rows: list[tuple]) -> Path:
    text = "".join(
        f"{c}\t{p}\t{p}\t{100 * m / (m + u):.1f}\t{m}\t{u}\n" for c, p, m, u in rows
    )
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)
    return path


def test_cov_files_round_trip_to_a_beta_matrix(tmp_path):
    a = _write_cov(tmp_path / "A.cov.gz", [
        ("chr1", 100, 3, 1), ("chr1", 150, 0, 4), ("chr2", 10, 5, 5),
    ])
    b = _write_cov(tmp_path / "B.cov", [
        ("1", 150, 2, 2), ("chr1", 200, 1, 0), ("chrM", 5, 1, 1),
    ])
    reference = build_cpg_reference([a, b], progress=False)
    assert len(reference) == 4  # chrM is outside the default chromosome set
    assert reference.chrom_names == sorted(reference.chrom_names)

    saved = tmp_path / "grid.npz"
    reference.save(saved)
    np.testing.assert_array_equal(CpGReference.load(saved).positions,
                                  reference.positions)

    cache = convert_cov_files([a, b], reference, tmp_path / "cache", progress=False)
    assert [p.name for p in cache] == ["A.counts.npy", "B.counts.npy"]
    beta, total, locations = load_matrix(cache, reference, chromosome="1",
                                         min_observed_samples=1)
    assert locations[["chr", "start", "end"]].values.tolist() == [
        ["1", 99, 100], ["1", 149, 150], ["1", 199, 200],
    ]
    expected = np.array([[0.75, np.nan], [0.0, 0.5], [np.nan, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(beta, expected)
    np.testing.assert_array_equal(total, [[4, 0], [4, 4], [0, 1]])

    # a row observed in fewer samples than required is dropped
    beta2, _total2, locations2 = load_matrix(cache, reference, chromosome="1",
                                             min_observed_samples=2)
    assert locations2["start"].tolist() == [149]
    np.testing.assert_array_equal(beta2, [[0.0, 0.5]])
