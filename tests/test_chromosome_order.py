"""Spec §2b prerequisite: canonical chromosome ordering.

Natural ordering (``chr1, chr2, chr10``) conflicted with the lexicographic
validation a prior version applied (``np.lexsort`` on ``chr`` as a string
sorts ``"chr10"`` before ``"chr2"``), confirmed on the repo's own real hg38
fixture (``chr1, chr7, chr19`` -- already not lexicographic). Both callers now
group rows by chromosome in a canonical order and sort ascending within each
chromosome themselves, regardless of the order a caller presents rows in.
These tests pin exactly what the spec asks for: ``chr1``..``chr22``, ``chrX``,
``chrY``, ``chrM`` plus unplaced contigs, presented in natural order,
lexicographic order and shuffled, producing identical internal state and
identical results.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from eiger2.core import (  # noqa: E402
    CANONICAL_CHROMOSOMES,
    Eiger2Config,
    call_dmrs,
    canonical_chromosome_order,
    chromosome_sort_permutation,
)

_KNOWN = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
_UNPLACED = ["chrUn_KI270302v1", "chr1_KI270706v1_random", "chrUn_GL000220v1"]
_ALL_NAMES = _KNOWN + _UNPLACED


def test_canonical_chromosomes_constant_is_the_expected_biological_order():
    assert tuple(_KNOWN) == CANONICAL_CHROMOSOMES


def test_canonical_chromosome_order_is_independent_of_presentation_order():
    natural = list(_ALL_NAMES)
    lexicographic = sorted(_ALL_NAMES)
    shuffled = list(_ALL_NAMES)
    random.Random(7).shuffle(shuffled)
    assert natural != lexicographic  # the fixture actually exercises both
    assert shuffled not in (natural, lexicographic)

    orders = [canonical_chromosome_order(p) for p in (natural, lexicographic, shuffled)]
    assert orders[0] == orders[1] == orders[2]
    # known chromosomes precede unplaced ones, in biological (not
    # lexicographic) order; unplaced ones sort lexicographically among
    # themselves.
    assert orders[0][: len(_KNOWN)] == _KNOWN
    assert orders[0][len(_KNOWN) :] == sorted(_UNPLACED)


def test_canonical_chromosome_order_ignores_chromosomes_not_present():
    # A pure function of the SET of names given, not of CANONICAL_CHROMOSOMES
    # itself -- a run touching only chr2 and chr10 does not invent chr1.
    assert canonical_chromosome_order(["chr10", "chr2"]) == ["chr2", "chr10"]


def _multi_chromosome_layout(
    order: list[str], *, n_per_chrom: int, n_samples: int, seed: int
):
    """One block of ``n_per_chrom`` CpGs per chromosome in ``order``, each
    block already ascending-sorted internally -- ``order`` controls only
    which chromosome's block comes first in the row array, not intra-block
    order, isolating exactly the thing spec §2b prerequisite requires."""

    rng = np.random.default_rng(seed)
    chrom_col, start_col, rows = [], [], []
    for name in order:
        starts = np.arange(n_per_chrom) * 50
        values = np.clip(
            0.5 + 0.05 * rng.standard_normal((n_per_chrom, n_samples)), 0.02, 0.98
        )
        chrom_col.extend([name] * n_per_chrom)
        start_col.extend(starts.tolist())
        rows.append(values)
    matrix = np.concatenate(rows, axis=0)
    locations = pd.DataFrame({
        "chr": chrom_col, "start": start_col,
        "end": [s + 1 for s in start_col],
    })
    return matrix, locations


def test_chromosome_sort_permutation_groups_and_sorts_regardless_of_input_order():
    natural = list(_KNOWN)
    lexicographic = sorted(_KNOWN)
    shuffled = list(_KNOWN)
    random.Random(13).shuffle(shuffled)

    canonical_frames = []
    for order in (natural, lexicographic, shuffled):
        _matrix, locations = _multi_chromosome_layout(
            order, n_per_chrom=5, n_samples=3, seed=1
        )
        permutation, chromosome_order = chromosome_sort_permutation(locations)
        assert chromosome_order == _KNOWN
        reordered = locations.iloc[permutation].reset_index(drop=True)
        canonical_frames.append(reordered)

    base = canonical_frames[0]
    for frame in canonical_frames[1:]:
        pd.testing.assert_frame_equal(frame, base)
    # grouped by chromosome in canonical order, ascending start within each
    assert list(base["chr"]) == [c for c in _KNOWN for _ in range(5)]
    for _chrom, block in base.groupby("chr", sort=False):
        assert list(block["start"]) == sorted(block["start"])


def test_call_dmrs_results_are_independent_of_chromosome_presentation_order():
    # A representative subset, not all 25 canonical chromosomes: natural and
    # lexicographic order genuinely diverge here (chr9 vs chr10), and this
    # stage's structural gates (spec §2a.2) need chromosome blocks large
    # relative to null_replicates to reach a non-degenerate `n_regions` --
    # the null construction is not yet chromosome-aware (that is §2b's own
    # body, not this prerequisite), so with all 25 real chromosomes even a
    # working construction here would be testing that pre-existing
    # limitation, not order-invariance.
    chroms = ["chr1", "chr2", "chr9", "chr10", "chrX"]
    natural = list(chroms)
    lexicographic = sorted(chroms)
    shuffled = list(chroms)
    random.Random(23).shuffle(shuffled)
    assert natural != lexicographic

    config = Eiger2Config(
        window=4, null_min_lag=3, null_replicates=100, random_seed=5,
        region_null_mode="lag",
    )
    results = []
    for order in (natural, lexicographic, shuffled):
        matrix, locations = _multi_chromosome_layout(
            order, n_per_chrom=100, n_samples=10, seed=42,
        )
        regions, values, manifest = call_dmrs(matrix, locations, config=config)
        results.append((regions, values, manifest))

    base_regions, base_values, base_manifest = results[0]
    # chr1, chr2, chr9, chr10, chrX is already the canonical (biological)
    # order for this subset -- the point being tested is that ALL THREE
    # presentations resolve to it, not just the one already in that order.
    assert base_manifest["chromosome_order"] == [
        "chr1", "chr2", "chr9", "chr10", "chrX",
    ]
    for regions, values, manifest in results[1:]:
        pd.testing.assert_frame_equal(regions, base_regions)
        pd.testing.assert_frame_equal(values, base_values)
        assert manifest["chromosome_order"] == base_manifest["chromosome_order"]
        assert manifest["status"] == base_manifest["status"]
        assert manifest["n_regions"] == base_manifest["n_regions"]
        assert manifest["n_cpg_retained"] == base_manifest["n_cpg_retained"]


def test_run_legacy_results_are_independent_of_chromosome_presentation_order():
    from eiger2.cli import run_legacy

    natural = list(_KNOWN)
    lexicographic = sorted(_KNOWN)
    shuffled = list(_KNOWN)
    random.Random(29).shuffle(shuffled)

    results = []
    for order in (natural, lexicographic, shuffled):
        matrix, locations = _multi_chromosome_layout(
            order, n_per_chrom=20, n_samples=10, seed=17,
        )
        sample_names = [f"s{i}" for i in range(matrix.shape[1])]
        frame, manifest = run_legacy(
            matrix, locations, sample_names,
            cov_threshold=0.01, delta_threshold=0.2, min_cpgs=5, impute=True,
        )
        results.append((frame, manifest))

    base_frame, base_manifest = results[0]
    assert base_manifest["chromosome_order"] == _KNOWN
    for frame, manifest in results[1:]:
        pd.testing.assert_frame_equal(frame, base_frame)
        assert manifest["chromosome_order"] == base_manifest["chromosome_order"]
        assert manifest["n_regions"] == base_manifest["n_regions"]


def test_cli_validate_no_longer_rejects_natural_chromosome_order():
    # Confirmed defect: a prior version's np.lexsort-on-string check put
    # "chr10" before "chr2", so natural order raised SystemExit here.
    import eiger2.cli as cli

    matrix, locations = _multi_chromosome_layout(
        ["chr1", "chr2", "chr10"], n_per_chrom=3, n_samples=4, seed=3,
    )
    samples = [f"s{i}" for i in range(matrix.shape[1])]
    out_matrix, out_locations, out_samples = cli._validate(matrix, locations, samples)
    assert out_samples == samples
    np.testing.assert_array_equal(out_matrix, matrix)
    pd.testing.assert_frame_equal(
        out_locations.reset_index(drop=True), locations[["chr", "start", "end"]]
    )
