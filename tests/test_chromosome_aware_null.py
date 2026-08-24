"""Spec §2b.5: mandatory multi-chromosome regression test.

Every benchmark ever run was single-chromosome: ``dense`` is one by
construction, and the real hg38 layout fixture (``wgbs_layout_hg38.npz``,
holding chr1, chr7 and chr19) was always sliced to one chromosome
(``chrom="chr1"``, etc.) by every existing caller. That is why the
genome-wide-lag defect (spec §2b.1/§2b.2) survived seven review stages: no
existing test ever built a null across more than one chromosome.

This test uses the real fixture with all three chromosomes present at once,
at a scale down from the full ~5.1M-row fixture (a 1/25 proportional slice of
each chromosome, preserving each chromosome's real relative size) so the
suite stays fast while still exercising real, unequal chromosome sizes --
chr1 is the largest at ~46.5% of the total, chr19 the smallest at ~21.7%
-- exactly the size disparity that made the old genome-wide lag draw survive
almost entirely on chr1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "benchmarking" / "layouts")
)

from eiger2.core import (  # noqa: E402
    Eiger2Config,
    call_dmrs,
    centre_cpgs,
    chromosome_lag_feasibility,
    null_tracks,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "benchmarking" / "layouts" / "wgbs_layout_hg38.npz"
)
# Real full-fixture chromosome sizes: chr1=2,375,159, chr7=1,622,825,
# chr19=1,105,620 (confirmed against the vendored .npz directly). Scaled by
# 1/25, preserving each chromosome's real share of the whole (46.5% /
# 31.8% / 21.7%) rather than three equal-sized slices, which would not
# exercise the size disparity the original defect depended on.
_SLICE_SIZES = {"chr1": 95_006, "chr7": 64_913, "chr19": 44_224}


def _real_multi_chromosome_layout(*, n_samples: int, seed: int):
    from layouts import load_wgbs_layout

    frames = []
    for name, n in _SLICE_SIZES.items():
        locs, _depth = load_wgbs_layout(str(_FIXTURE), chrom=name, n_cpg=n, offset=0)
        frames.append(locs)
    locations = pd.concat(frames, ignore_index=True)

    rng = np.random.default_rng(seed)
    n_total = len(locations)
    innovation = rng.normal(0, 0.08, n_total)
    baseline = np.empty(n_total)
    baseline[0] = innovation[0]
    for i in range(1, n_total):
        baseline[i] = 0.65 * baseline[i - 1] + innovation[i]
    matrix = np.clip(
        0.5 + baseline[:, None] + rng.normal(0, 0.03, (n_total, n_samples)),
        0.02, 0.98,
    )
    return matrix, locations


def test_real_multi_chromosome_layout_clears_structural_gates():
    matrix, locations = _real_multi_chromosome_layout(n_samples=20, seed=0)
    config = Eiger2Config(
        window=4, null_min_lag=1000, null_replicates=100, random_seed=5,
        region_null_mode="lag",
    )
    regions, _values, manifest = call_dmrs(matrix, locations, config=config)
    structural = manifest["gates"]["structural"]

    # spec §2b.5's three named bounds, checked directly rather than only via
    # the pass/fail booleans, since those booleans gate at 50%/min_null_coverage
    # -- weaker floors than this regression test's own, stricter requirement.
    assert structural["surviving_null_tracks"]["fraction"] >= 0.9
    assert (
        structural["null_coverage"]["value"] >= config.min_null_coverage
    )
    # "structural gates all pass" -- the fourth (chromosome_exposure, added
    # by this same stage) included, though it is not separately bounded by
    # the spec's own three named checks above.
    assert all(gate["passed"] for gate in structural.values())


def test_real_multi_chromosome_layout_null_composition_matches_exposure():
    # "each chromosome's share of pooled null positions within a factor of
    # 1.2 of its share of eligible observed edges -- the composition check,
    # which a cap-only fix would fail." A fix that merely capped the lag
    # range (rather than drawing independently per chromosome) could still
    # pass the survival/coverage bounds above while remaining skewed toward
    # the largest chromosome; this is the check that catches that.
    matrix, locations = _real_multi_chromosome_layout(n_samples=20, seed=0)
    config = Eiger2Config(
        window=4, null_min_lag=1000, null_replicates=100, random_seed=5,
    )
    residuals = centre_cpgs(matrix)
    tracks = null_tracks(residuals, config, locations)
    assert len(tracks) / config.null_replicates >= 0.9

    null_positions_by_chromosome: dict[str, int] = {}
    for _track, eligible, segments in tracks:
        position = 0
        for name, _lag, _start, length in segments:
            segment_eligible = eligible[position : position + length]
            null_positions_by_chromosome[name] = (
                null_positions_by_chromosome.get(name, 0)
                + int(segment_eligible.sum())
            )
            position += length

    feasibility = chromosome_lag_feasibility(locations, config)
    assert all(v["feasible"] for v in feasibility.values())
    total_edges = sum(v["eligible_observed_edges"] for v in feasibility.values())
    total_null = sum(null_positions_by_chromosome.values())
    assert set(null_positions_by_chromosome) == set(_SLICE_SIZES)

    for name in _SLICE_SIZES:
        edge_share = feasibility[name]["eligible_observed_edges"] / total_edges
        null_share = null_positions_by_chromosome[name] / total_null
        ratio = null_share / edge_share
        assert 1 / 1.2 <= ratio <= 1.2, (
            f"{name}: null share {null_share:.4f} vs edge share "
            f"{edge_share:.4f} (ratio {ratio:.3f}) outside the 1.2x factor"
        )


def test_real_multi_chromosome_layout_chromosome_order_is_canonical():
    # chr1, chr7, chr19 is the fixture's own storage order AND the
    # canonical (biological) order for this subset -- unlike chr1, chr19,
    # chr7 (lexicographic). Loaded via three separate chrom= calls and
    # concatenated in dict order (chr1, chr7, chr19) here, so this also
    # confirms chromosome_sort_permutation is a no-op on already-canonical
    # input, not merely that it fixes a deliberately-shuffled one (spec
    # §2b prerequisite, tests/test_chromosome_order.py covers the shuffled
    # case directly).
    matrix, locations = _real_multi_chromosome_layout(n_samples=5, seed=1)
    config = Eiger2Config(
        window=4, null_min_lag=1000, null_replicates=10, random_seed=2,
        region_null_mode="lag", min_cpgs=4,
    )
    _regions, _values, manifest = call_dmrs(matrix, locations, config=config)
    assert manifest["chromosome_order"] == ["chr1", "chr7", "chr19"]
