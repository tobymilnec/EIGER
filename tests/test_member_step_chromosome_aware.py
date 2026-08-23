"""Spec §2b.4: the same chromosome-aware audit for
``member_step_lag_null_ensemble``.

It derived ``L_max = n_cpg // (min_subtrack_length + w - 1)`` from the
genome-wide ``n_cpg`` -- the same class of error as ``null_tracks()``
(spec §2b.2). Each replicate now draws an independent lag per feasible
chromosome and concatenates every feasible chromosome's own sub-tracks into
one genome-wide realisation. Raises only when NO chromosome individually
clears the floor -- not when the genome-wide total looks large enough but
every individual chromosome is too small.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from eiger2.core import (  # noqa: E402
    Eiger2Config,
    member_step_lag_null_ensemble,
    member_step_lag_null_tracks,
    segment,
)


def _locations(sizes: dict[str, int]) -> pd.DataFrame:
    chrom_col: list[str] = []
    start_col: list[int] = []
    for name, n in sizes.items():
        chrom_col.extend([name] * n)
        start_col.extend((np.arange(n) * 20).tolist())
    return pd.DataFrame({
        "chr": chrom_col, "start": start_col,
        "end": [s + 1 for s in start_col],
    })


def test_each_replicate_draws_an_independent_lag_per_chromosome():
    rng = np.random.default_rng(1)
    sizes = {"chr1": 6000, "chr2": 3000}
    residuals = rng.normal(size=(sum(sizes.values()), 15))
    locations = _locations(sizes)
    config = Eiger2Config(
        window=4, null_min_lag=10, min_subtrack_length=20, n_lag_draws=8,
        random_seed=3,
    )
    replicates, lags = member_step_lag_null_ensemble(residuals, config, locations)

    assert len(replicates) == config.n_lag_draws
    assert len(lags) == config.n_lag_draws
    for segments in lags:
        chromosomes = [name for name, _lag, _n in segments]
        assert chromosomes == ["chr1", "chr2"]
    # genuinely independent draws, not the same lag reused across
    # chromosomes within a replicate, nor identical across replicates
    chr1_lags = {segments[0][1] for segments in lags}
    chr2_lags = {segments[1][1] for segments in lags}
    assert len(chr1_lags) > 1
    assert len(chr2_lags) > 1
    assert chr1_lags != chr2_lags


def test_infeasible_chromosome_is_silently_excluded_not_fatal():
    # chrSmall is too small to admit a feasible member_step lag at this
    # null_min_lag/min_subtrack_length; the ensemble still builds normally
    # from chrBig alone, rather than raising because ONE chromosome failed.
    rng = np.random.default_rng(1)
    sizes = {"chrBig": 6000, "chrSmall": 100}
    residuals = rng.normal(size=(sum(sizes.values()), 15))
    locations = _locations(sizes)
    config = Eiger2Config(
        window=4, null_min_lag=30, min_subtrack_length=20, n_lag_draws=5,
        random_seed=3,
    )
    replicates, lags = member_step_lag_null_ensemble(residuals, config, locations)

    assert len(replicates) == config.n_lag_draws
    for segments in lags:
        assert [name for name, _lag, _n in segments] == ["chrBig"]


def test_raises_only_when_no_chromosome_individually_clears_the_floor():
    # Both chromosomes are individually too small, even though their
    # combined (genome-wide) total might look large enough under the OLD
    # genome-wide-derived L_max -- the per-chromosome check is what
    # actually governs now.
    rng = np.random.default_rng(1)
    sizes = {"chr1": 100, "chr2": 80}
    residuals = rng.normal(size=(sum(sizes.values()), 15))
    locations = _locations(sizes)
    config = Eiger2Config(
        window=4, null_min_lag=30, min_subtrack_length=20, n_lag_draws=5,
        random_seed=3,
    )
    with pytest.raises(ValueError, match="at least one CHROMOSOME"):
        member_step_lag_null_ensemble(residuals, config, locations)


def test_raise_message_names_the_largest_chromosomes_own_n_cpg_and_l_max():
    rng = np.random.default_rng(1)
    sizes = {"chr1": 100, "chr2": 80}
    residuals = rng.normal(size=(sum(sizes.values()), 15))
    locations = _locations(sizes)
    config = Eiger2Config(
        window=4, null_min_lag=30, min_subtrack_length=20, n_lag_draws=5,
        random_seed=3,
    )
    with pytest.raises(ValueError, match=r"n_cpg=100 \(L_max=4\)"):
        member_step_lag_null_ensemble(residuals, config, locations)


def test_chromosome_ensemble_preserves_large_gap_eligibility():
    """The chromosome wrapper must pass its location slice to the underlying
    member-step builder; otherwise max_cpg_gap_bp silently disappears."""

    rng = np.random.default_rng(11)
    n_cpg = 120
    residuals = rng.normal(size=(n_cpg, 15))
    starts = np.arange(n_cpg) * 20
    starts[40:] += 10_000
    starts[80:] += 10_000
    fragmented = pd.DataFrame({
        "chr": ["chr1"] * n_cpg,
        "start": starts,
        "end": starts + 1,
    })
    config = Eiger2Config(
        window=4, null_min_lag=2, min_subtrack_length=10,
        n_lag_draws=1, max_cpg_gap_bp=300, random_seed=3,
    )

    replicates, lags = member_step_lag_null_ensemble(
        residuals, config, fragmented,
    )
    track, ensemble_eligible, _segments = replicates[0]
    lag = lags[0][0][1]

    direct = member_step_lag_null_tracks(
        residuals, lag, config, fragmented,
    )
    direct_without_locations = member_step_lag_null_tracks(
        residuals, lag, config, None,
    )

    def concatenate(items):
        tracks, eligibility = [], []
        for sub_track, sub_eligible, _offset in items:
            boundary = sub_eligible.copy()
            boundary[-1] = False
            tracks.append(sub_track)
            eligibility.append(boundary)
        return np.concatenate(tracks), np.concatenate(eligibility)

    expected_track, expected_eligible = concatenate(direct)
    _ungapped_track, ungapped_eligible = concatenate(direct_without_locations)
    assert np.array_equal(track, expected_track, equal_nan=True)
    assert np.array_equal(ensemble_eligible, expected_eligible)
    # At least one position is excluded specifically because a lagged window
    # would use members on both sides of a >300 bp observed gap.
    assert np.any(ungapped_eligible & ~ensemble_eligible)

    # With every finite edge above an effectively unrestrictive threshold,
    # segmentation may form regions only inside contiguous eligible runs;
    # none can include an eligibility break introduced by either genomic gap.
    regions = segment(
        track, ensemble_eligible, -np.inf,
        min_cpgs=2, max_gap_edges=0,
    )
    for first, last in regions:
        assert ensemble_eligible[first:last].all()
