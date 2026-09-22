"""Chromosome handling: canonical order and per-chromosome lag feasibility.

The callers group rows by chromosome in a canonical order (``chr1``..``chr22``,
``chrX``, ``chrY``, ``chrM``, then other contigs lexicographically) and sort by
position within each chromosome, whatever order the rows arrive in. Natural,
lexicographic and shuffled presentations must give identical results.
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
    canonical_chromosome_order,
    chromosome_lag_feasibility,
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


def test_legacy_region_fdr_is_independent_of_chromosome_presentation_order():
    """The supported pipeline (here its --legacy profile, which needs no lag
    null) sorts rows itself: natural, lexicographic and shuffled chromosome
    order give identical regions, q-values and IDs. Each chromosome's values
    are generated from its own name, so only the presentation order varies."""

    from eiger2.coordination import (
        LEGACY,
        RegionFdrConfig,
        call_coordinated_regions,
    )

    def layout(order):
        chrom_col, start_col, rows = [], [], []
        for name in order:
            rng = np.random.default_rng(_KNOWN.index(name) + 100)
            values = 0.5 + 0.2 * rng.standard_normal((40, 10))
            values[10:22] += 0.25 * rng.standard_normal(10)
            chrom_col.extend([name] * 40)
            start_col.extend((np.arange(40) * 50).tolist())
            rows.append(np.clip(values, 0.0, 1.0))
        locations = pd.DataFrame({
            "chr": chrom_col, "start": start_col,
            "end": [s + 1 for s in start_col],
        })
        return np.concatenate(rows, axis=0), locations

    natural = list(_KNOWN)
    lexicographic = sorted(_KNOWN)
    shuffled = list(_KNOWN)
    random.Random(29).shuffle(shuffled)
    config = RegionFdrConfig(profile=LEGACY, fdr_replicates=5, axis_mode="off")
    runs = [
        call_coordinated_regions(*layout(order), config=config)
        for order in (natural, lexicographic, shuffled)
    ]
    base = runs[0]
    assert len(base.regions) > 0
    assert base.manifest["input"]["chromosome_order"] == _KNOWN
    for run in runs[1:]:
        pd.testing.assert_frame_equal(run.regions, base.regions)
        pd.testing.assert_frame_equal(run.permutations, base.permutations)


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


def test_chromosome_lag_feasibility_identifies_infeasible_chromosomes():
    starts = np.concatenate([np.arange(2000), np.arange(40)]) * 20
    locations = pd.DataFrame({
        "chr": ["chrBig"] * 2000 + ["chrTiny"] * 40,
        "start": starts,
        "end": starts + 1,
    })
    config = Eiger2Config(window=4, null_min_lag=50)
    feasibility = chromosome_lag_feasibility(locations, config)

    # l_max = (n_c - 2) // (span - 1) with span = max(window, 2) = 4: the
    # largest lag that still leaves _lag_null_track two usable positions.
    assert feasibility["chrBig"]["feasible"] is True
    assert feasibility["chrBig"]["n_cpg"] == 2000
    assert feasibility["chrBig"]["l_max"] == (2000 - 2) // 3

    assert feasibility["chrTiny"]["feasible"] is False
    assert feasibility["chrTiny"]["n_cpg"] == 40
    assert feasibility["chrTiny"]["l_max"] == (40 - 2) // 3

    # every edge here is gap-eligible: n_cpg - 1 internal edges per chromosome
    assert feasibility["chrBig"]["eligible_observed_edges"] == 1999
    assert feasibility["chrTiny"]["eligible_observed_edges"] == 39
