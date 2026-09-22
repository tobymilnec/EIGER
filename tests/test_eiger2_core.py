"""Unit tests for the scan primitives in ``eiger2.core``.

Pairwise-complete statistics, windows, the distant-lag null and its
eligibility rules, segmentation and region scores. End-to-end behaviour of the
supported workflow is tested in ``test_region_fdr.py`` and pinned against the
frozen reference in ``test_region_fdr_reference.py``.
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
    _pair_statistic,
    edge_statistic,
    null_exchangeability,
    null_threshold,
    null_tracks,
    region_area,
    segment,
)


def locations(n_cpg: int, chrom: str = "chr1") -> pd.DataFrame:
    start = np.arange(n_cpg) * 2
    return pd.DataFrame({"chr": chrom, "start": start, "end": start + 1})


# --------------------------------------------------------------------------
# pair and window statistics
# --------------------------------------------------------------------------


def test_pair_statistic_matches_numpy_on_complete_data():
    rng = np.random.default_rng(0)
    left = rng.normal(size=(3, 40))
    right = rng.normal(size=(3, 40))
    covariance = _pair_statistic(left, right, "covariance", 5)
    correlation = _pair_statistic(left, right, "correlation", 5)
    for row in range(3):
        assert covariance[row] == pytest.approx(
            np.cov(left[row], right[row], ddof=1)[0, 1]
        )
        assert correlation[row] == pytest.approx(
            np.corrcoef(left[row], right[row])[0, 1]
        )


def test_pairwise_complete_removes_the_missingness_deflation():
    """EIGER 1 filled gaps with the CpG mean, deflating covariance by (1-m)^2."""

    rng = np.random.default_rng(1)
    n, reps = 60, 300
    truth = []
    observed = []
    for missing in (0.0, 0.3):
        estimates = []
        mean_filled = []
        for _ in range(reps):
            group = np.zeros(n)
            group[: n // 2] = 1
            x = 0.30 * group + rng.normal(0, 0.05, n)
            y = 0.30 * group + rng.normal(0, 0.05, n)
            pair = np.vstack([x, y])
            pair[rng.random(pair.shape) < missing] = np.nan
            estimates.append(
                _pair_statistic(pair[:1], pair[1:], "covariance", 3)[0]
            )
            filled = np.where(np.isfinite(pair), pair, 0.0)
            mean_filled.append(np.cov(filled[0], filled[1], ddof=1)[0, 1])
        truth.append(np.nanmean(estimates))
        observed.append(np.mean(mean_filled))
    # pairwise-complete is unbiased across missing rates
    assert truth[1] == pytest.approx(truth[0], rel=0.10)
    # mean-filling deflates by roughly (1 - m)^2 = 0.49
    assert observed[1] / observed[0] == pytest.approx(0.49, abs=0.08)


def test_correlation_sees_rare_subgroups_that_covariance_cannot():
    """cov = f(1-f)d^2 is prevalence-limited; correlation saturates."""

    rng = np.random.default_rng(2)
    n, k, delta, sigma = 519, 3, 0.30, 0.03
    group = np.zeros(n)
    group[:k] = 1
    covariances, correlations = [], []
    for _ in range(200):
        x = (delta * group + rng.normal(0, sigma, n))[None, :]
        y = (delta * group + rng.normal(0, sigma, n))[None, :]
        covariances.append(_pair_statistic(x, y, "covariance", 5)[0])
        correlations.append(_pair_statistic(x, y, "correlation", 5)[0])
    fraction = k / n
    assert np.mean(covariances) == pytest.approx(
        fraction * (1 - fraction) * delta**2, rel=0.30
    )
    assert np.mean(covariances) < 0.012  # below EIGER 1's default threshold
    assert np.mean(correlations) > 0.30  # comfortably detectable


def test_window_reduces_null_variance():
    rng = np.random.default_rng(3)
    config_pair = Eiger2Config(window=1, min_paired_samples=5)
    config_window = Eiger2Config(window=6, min_paired_samples=5)
    residuals = rng.normal(size=(4000, 20))
    spread_pair = np.nanstd(edge_statistic(residuals, config_pair))
    spread_window = np.nanstd(edge_statistic(residuals, config_window))
    assert spread_window < 0.6 * spread_pair


# --------------------------------------------------------------------------
# latent-axis settings
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"deflation_rule": "self_referenced"},
    {"deflation_null_quantile": 0.0},
    {"deflation_null_quantile": 1.5},
])
def test_the_relative_rule_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        Eiger2Config(**kwargs)


# --------------------------------------------------------------------------
# segmentation and region scores
# --------------------------------------------------------------------------


def test_segment_counts_cpgs_not_edges():
    """EIGER 1 applied its length threshold to edges, rejecting valid blocks."""

    eligible = np.ones(4, dtype=bool)
    assert segment(
        np.ones(4), eligible, 0.5, min_cpgs=5
    ).tolist() == [[0, 4]]  # 4 edges span 5 CpGs
    assert segment(np.ones(3), np.ones(3, bool), 0.5, min_cpgs=5).size == 0


def test_segment_gap_tolerance_bridges_a_single_dropped_cpg():
    statistic = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1.0])
    eligible = np.ones(10, dtype=bool)
    assert len(segment(statistic, eligible, 0.5, min_cpgs=5, max_gap_edges=0)) == 2
    assert segment(
        statistic, eligible, 0.5, min_cpgs=5, max_gap_edges=2
    ).tolist() == [[0, 10]]


def test_segment_never_crosses_ineligible_edges():
    statistic = np.ones(10)
    eligible = np.ones(10, dtype=bool)
    eligible[5] = False
    for first, last in segment(
        statistic, eligible, 0.5, min_cpgs=3, max_gap_edges=3
    ):
        assert not (first <= 5 < last)


def test_segment_constant_array_threshold_matches_scalar():
    rng = np.random.default_rng(21)
    statistic = rng.normal(size=300)
    eligible = np.ones(300, dtype=bool)
    scalar_regions = segment(statistic, eligible, 0.5, min_cpgs=4, max_gap_edges=1)
    array_regions = segment(
        statistic, eligible, np.full(300, 0.5), min_cpgs=4, max_gap_edges=1,
    )
    np.testing.assert_array_equal(scalar_regions, array_regions)


def test_region_area_constant_array_threshold_matches_scalar():
    rng = np.random.default_rng(22)
    statistic = rng.normal(size=300)
    regions = np.array([[10, 25], [100, 130]])
    scalar_areas = region_area(statistic, regions, 0.3)
    array_areas = region_area(statistic, regions, np.full(300, 0.3))
    np.testing.assert_array_equal(scalar_areas, array_areas)


def test_segment_and_region_area_reject_mismatched_array_threshold_shape():
    statistic = np.zeros(10)
    with pytest.raises(ValueError):
        segment(statistic, np.ones(10, dtype=bool), np.zeros(5), min_cpgs=2)
    with pytest.raises(ValueError):
        region_area(statistic, np.array([[0, 2]]), np.zeros(5))


def test_region_area_array_threshold_uses_per_edge_values():
    statistic = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    threshold = np.array([0.5, 1.5, 1.0, 3.5, 0.0])
    regions = np.array([[0, 4]])
    areas = region_area(statistic, regions, threshold)
    expected = (1.0 - 0.5) + (2.0 - 1.5) + (3.0 - 1.0) + (4.0 - 3.5)
    assert areas[0] == pytest.approx(expected)


# --------------------------------------------------------------------------
# statistic properties
# --------------------------------------------------------------------------


def test_pairwise_flag_preserves_published_behaviour():
    """The legacy default must still return NaN, bit for bit."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eiger_processing",
        Path(__file__).resolve().parents[1] / "Python" / "functions" / "processing.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rng = np.random.default_rng(0)
    complete = rng.normal(0.5, 0.1, (50, 30))
    legacy = module.covariance_blocks(complete)
    pairwise = module.covariance_blocks(complete, pairwise=True)
    assert np.allclose(legacy, pairwise)  # identical when nothing is missing

    gapped = complete.copy()
    gapped[10, 3] = np.nan
    legacy_gapped = module.covariance_blocks(gapped)
    pairwise_gapped = module.covariance_blocks(gapped, pairwise=True)
    assert not np.isfinite(legacy_gapped[9])   # published behaviour: NaN
    assert not np.isfinite(legacy_gapped[10])
    assert np.isfinite(pairwise_gapped[9])     # pairwise recovers the edge
    assert np.isfinite(pairwise_gapped[10])


# --------------------------------------------------------------------------
# distant-lag null: defaults
# --------------------------------------------------------------------------


def test_null_replicates_default_is_100():
    assert Eiger2Config().null_replicates == 100


# --------------------------------------------------------------------------
# distant-lag null: window assembly
# --------------------------------------------------------------------------

from eiger2.core import _lag_null_track, _lag_statistic, _null_eligibility  # noqa: E402


def _reference_lag_null_track(residuals, lag, config):
    """Old (buggy) implementation, kept ONLY as a fixed reference point for the
    no-op-at-w<=2 test below -- never reintroduced into the package."""
    from eiger2.core import _window_from_lags
    span = max(config.window, 2)
    usable = residuals.shape[0] - lag * (span - 1)
    if usable < 2:
        return np.empty(0, dtype=float)
    lag_statistics = []
    for separation in range(1, span):
        values = _lag_statistic(residuals, separation * lag, config)
        lag_statistics.append(values[:usable])
    if config.window == 1:
        return lag_statistics[0]
    return _window_from_lags(lag_statistics, config.window)


def test_lag_null_track_is_a_noop_at_window_1_and_2():
    rng = np.random.default_rng(101)
    residuals = rng.normal(size=(200, 30))
    for w in (1, 2):
        config = Eiger2Config(window=w, random_seed=1)
        old = _reference_lag_null_track(residuals, lag=17, config=config)
        new = _lag_null_track(residuals, lag=17, config=config)
        assert new.shape == old.shape
        # allclose, not exact equality: at w=2, the old reference path goes
        # through _window_from_lags's cumsum-then-diff construction, which is
        # mathematically a no-op but is not bit-exact (floating-point
        # non-associativity, ~1e-16). The new direct-sum implementation is
        # bit-exact to the raw underlying statistic; a tight tolerance still
        # catches any real (non-floating-point-noise) discrepancy.
        np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-10)


def test_lag_null_track_member_set_is_exactly_the_lag_multiples():
    # Build a residual matrix where stat(row j, row k) is recoverable from the
    # pair itself: use two samples, values j and k respectively, so
    # correlation/covariance of (j, j2, ..) vs (k, k2, ..) across a handful of
    # synthetic "samples" encodes (j, k) in a way we can invert. Simpler and
    # exact: monkeypatch _lag_statistic to return a matrix whose value at
    # position i is the pair (i, i+d) encoded as i * 10_000 + d, then confirm
    # _lag_null_track's output at window position i decodes to exactly the
    # pairs {(i + a*lag, i + b*lag) : 0 <= a < b < w} and no others.
    import eiger2.core as core_mod

    lag = 5
    w = 4
    n_positions = 30

    def fake_lag_statistic(residuals, d, config, chunk=200_000):
        n_pairs = residuals.shape[0] - d
        # value at row i encodes the pair (i, i + d) uniquely
        return np.array([i * 100_000 + d for i in range(n_pairs)], dtype=float)

    original = core_mod._lag_statistic
    core_mod._lag_statistic = fake_lag_statistic
    try:
        config = Eiger2Config(window=w, random_seed=1)
        dummy_residuals = np.zeros((n_positions + lag * (w - 1) + 5, 3))
        out = core_mod._lag_null_track(dummy_residuals, lag, config)
    finally:
        core_mod._lag_statistic = original

    # decode window position i=0: expect the average of pair-encodings for
    # {(0, lag), (0, 2*lag), (0, 3*lag), (lag, 2*lag), (lag, 3*lag), (2*lag, 3*lag)}
    expected_pairs = [(a * lag, b * lag) for a in range(w) for b in range(a + 1, w)]
    expected_values = [j * 100_000 + (k - j) for (j, k) in expected_pairs]
    assert out[0] == pytest.approx(np.mean(expected_values))

    # And decode a NON-ZERO window position (i=10) the same way, pinning that
    # the base offset i is carried correctly through the whole window, not just
    # at the i=0 edge case: expect the pairs {(10 + a*lag, 10 + b*lag)}, i.e.
    # every member shifted by the base position, and nothing else.
    i = 10
    expected_pairs_i = [
        (i + a * lag, i + b * lag) for a in range(w) for b in range(a + 1, w)
    ]
    expected_values_i = [j * 100_000 + (k - j) for (j, k) in expected_pairs_i]
    assert out[i] == pytest.approx(np.mean(expected_values_i))


def test_null_eligibility_offsets_match_lag_null_track_member_set():
    # Extract _lag_null_track's actual member-set offsets programmatically, via
    # the same fake-_lag_statistic monkeypatch/decode technique as
    # test_lag_null_track_member_set_is_exactly_the_lag_multiples above, instead
    # of hand-rederiving _null_eligibility's own offset formula. Copying
    # _null_eligibility's "step * lag" loop into the test only proves
    # _null_eligibility is self-consistent with itself -- it can't fail even if
    # _lag_null_track's member set diverges from it. Deriving the expected
    # offsets from _lag_null_track's real output (and validating that
    # derivation against the real output first) makes the two genuinely
    # coupled: this test now fails if either function's member set moves
    # without the other following.
    import eiger2.core as core_mod

    lag = 7
    for w in (1, 2, 3, 5):
        config = Eiger2Config(window=w, random_seed=1)
        span = max(w, 2)

        def fake_lag_statistic(residuals, d, config, chunk=200_000):
            n_pairs = residuals.shape[0] - d
            return np.array([i * 100_000 + d for i in range(n_pairs)], dtype=float)

        n_positions = 10
        original = core_mod._lag_statistic
        core_mod._lag_statistic = fake_lag_statistic
        try:
            dummy_residuals = np.zeros((n_positions + lag * (span - 1) + 5, 3))
            out = core_mod._lag_null_track(dummy_residuals, lag, config)
        finally:
            core_mod._lag_statistic = original

        if w == 1:
            # No window combination at w=1 (see _lag_null_track's `if w == 1`
            # branch): the raw single-lag statistic is returned directly, so
            # the member set at position 0 is {0, lag}. Confirm against the
            # real output before relying on it below.
            assert out[0] == pytest.approx(0 * 100_000 + lag)
            member_offsets = {0, lag}
        else:
            # Confirm the {0, lag, 2*lag, ..., (w-1)*lag} member set against
            # the real output -- same decode formula validated by
            # test_lag_null_track_member_set_is_exactly_the_lag_multiples --
            # before trusting it as ground truth for _null_eligibility below.
            candidate = sorted(a * lag for a in range(w))
            pairs = [
                (candidate[a], candidate[b])
                for a in range(w)
                for b in range(a + 1, w)
            ]
            predicted = np.mean([j * 100_000 + (k - j) for j, k in pairs])
            assert out[0] == pytest.approx(predicted)
            member_offsets = set(candidate)

        size = 50
        chromosome = np.array(["chr1"] * (size + lag * (span - 1) + 5))
        eligible = _null_eligibility(chromosome, lag, config, size)
        # A null position i is admissible iff every extracted member offset
        # (relative to i) sits on the same chromosome as i itself -- built from
        # member_offsets extracted above, not from copying _null_eligibility's
        # own loop.
        expected = np.ones(size, dtype=bool)
        first = chromosome[:size]
        for offset in sorted(member_offsets - {0}):
            expected &= first == chromosome[offset : offset + size]
        np.testing.assert_array_equal(eligible, expected)


# --------------------------------------------------------------------------
# distant-lag null: gap eligibility of members
# --------------------------------------------------------------------------


def test_null_eligibility_gap_masks_an_ineligible_member():
    # window=3, lag=10: member offsets {0, 10, 20}. Mark the local edge at
    # offset 10 (position 3's second member) ineligible; position 3 must be
    # excluded, its neighbours must not be.
    config = Eiger2Config(window=3, random_seed=1)
    size = 10
    lag = 10
    chromosome = np.array(["chr1"] * (size + lag * 2 + 5))
    observed_eligible = np.ones(chromosome.size - 1, dtype=bool)
    observed_eligible[3 + lag] = False  # local edge right after member offset 10
    eligible = _null_eligibility(
        chromosome, lag, config, size, observed_eligible=observed_eligible
    )
    assert not eligible[3]
    assert eligible[2]
    assert eligible[4]


def test_null_eligibility_gap_mask_does_not_apply_across_the_long_lag_itself():
    # Every member's OWN local edge is eligible; nothing requires the edge
    # BETWEEN member 0 and member 1 (a `lag`-sized jump) to be eligible --
    # that jump is the intentional locality-breaking construction, not a
    # defect. All positions must be admissible.
    config = Eiger2Config(window=2, random_seed=1)
    size = 10
    lag = 500
    chromosome = np.array(["chr1"] * (size + lag + 5))
    observed_eligible = np.ones(chromosome.size - 1, dtype=bool)
    # falsify a huge middle stretch, including everything strictly between any
    # member pair -- but not the members' own local edges (offsets 0..size-1
    # and lag..lag+size-1)
    observed_eligible[size : lag] = False
    eligible = _null_eligibility(
        chromosome, lag, config, size, observed_eligible=observed_eligible
    )
    assert eligible.all()


def test_null_eligibility_gap_mask_none_matches_pre_1d_c_behaviour():
    # observed_eligible=None (the default) must reproduce exactly what
    # _null_eligibility computed before this fix -- the chromosome-only check.
    config = Eiger2Config(window=4, random_seed=1)
    size = 20
    lag = 50
    rng = np.random.default_rng(3)
    chromosome = rng.choice(["chr1", "chr2"], size=size + lag * 3 + 5)
    without_gate = _null_eligibility(chromosome, lag, config, size)
    with_all_true_gate = _null_eligibility(
        chromosome, lag, config, size,
        observed_eligible=np.ones(chromosome.size - 1, dtype=bool),
    )
    np.testing.assert_array_equal(without_gate, with_all_true_gate)


def test_null_eligibility_gap_mask_is_a_no_op_on_dense():
    # dense: eligible_edges() is all-True (one giant contiguous run), so the
    # gap mask changes nothing -- proves continuity with every prior `dense`
    # result. Combined with the previous test (all-True gate == no gate), this
    # is the "assert it as a test" the spec asks for.
    from eiger2.core import eligible_edges

    n_cpg = 5000
    frame = locations(n_cpg)
    config = Eiger2Config(window=4, max_cpg_gap_bp=300, random_seed=1)
    observed_eligible = eligible_edges(frame, config)
    assert observed_eligible.all()


def test_null_tracks_dense_locations_unaffected_by_gap_masking():
    # End-to-end: null_tracks() on dense locations produces IDENTICAL
    # eligibility masks whether or not the (now-active) gap check runs,
    # because every edge is gap-eligible there. Confirmed by comparing against
    # explicitly passing locations=None (chromosome+gap check both off) on the
    # same single-chromosome data, where the only difference the gap check
    # could possibly make is exactly zero (all-True either way).
    rng = np.random.default_rng(11)
    residuals = rng.normal(size=(5000, 20))
    frame = locations(5000)
    config = Eiger2Config(window=4, null_replicates=10, random_seed=1)
    with_locations = null_tracks(residuals, config, frame)
    without_locations = null_tracks(residuals, config, None)
    assert len(with_locations) == len(without_locations)
    for (t1, e1, seg1), (t2, e2, seg2) in zip(
        with_locations, without_locations, strict=True
    ):
        # Segments carry the chromosome name too (spec §2b.2), which
        # genuinely differs between the two calls ("chr1" vs. "" for
        # locations=None) even though both describe the same single
        # contiguous block -- compare the lags actually drawn, not the name.
        assert [lag for _name, lag, _start, _length in seg1] == [
            lag for _name, lag, _start, _length in seg2
        ]
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(e1, e2)


def test_null_tracks_fragmented_layout_excludes_members_at_large_gaps():
    # On a genuinely fragmented layout, the gap mask must actually exclude
    # something -- confirms the fix is live, not just a no-op everywhere.
    rng = np.random.default_rng(12)
    n_cpg = 3000
    starts = np.arange(n_cpg) * 2
    for i in range(50, n_cpg, 50):
        starts[i:] += 5000  # break into runs of 50 CpGs, gaps of ~5000bp
    frame = pd.DataFrame({"chr": "chr1", "start": starts, "end": starts + 1})
    residuals = rng.normal(size=(n_cpg, 20))
    config = Eiger2Config(
        window=4, max_cpg_gap_bp=300, null_min_lag=5, null_replicates=20,
        random_seed=1,
    )
    tracks = null_tracks(residuals, config, frame)
    total_positions = sum(t.size for t, _, _ in tracks)
    total_eligible = sum(int(e.sum()) for _, e, _ in tracks)
    assert total_positions > 0
    assert total_eligible < total_positions  # the gap mask actually excludes some


# --------------------------------------------------------------------------
# distant-lag null: slice bounds
# --------------------------------------------------------------------------


def test_lag_null_track_slice_bounds_hold_for_large_window_and_lag():
    rng = np.random.default_rng(202)
    for w in (1, 2, 4, 8, 12):
        for lag in (2, 50, 1999):
            n = 4000
            residuals = rng.normal(size=(n, 10))
            config = Eiger2Config(window=w, random_seed=1)
            out = _lag_null_track(residuals, lag, config)
            span = max(w, 2)
            expected_usable = max(n - lag * (span - 1), 0)
            if expected_usable < 2:
                assert out.size == 0
            else:
                assert out.size == expected_usable


# --------------------------------------------------------------------------
# null exchangeability diagnostic
# --------------------------------------------------------------------------


def test_null_exchangeability_self_consistency_control():
    # held-out null exceedance must equal 1 - null_quantile to within Monte Carlo
    # error -- this is the control that proves the gate measures the right thing
    rng = np.random.default_rng(303)
    residuals = rng.normal(size=(3000, 40))
    config = Eiger2Config(window=1, random_seed=7, null_replicates=200)
    tracks = null_tracks(residuals, config)
    statistic = rng.normal(size=2999)  # arbitrary observed track, same length class
    eligible = np.ones(2999, dtype=bool)
    result = null_exchangeability(tracks, statistic, eligible, config)
    assert result["p_null"] == pytest.approx(1 - config.null_quantile, abs=0.003)


def test_null_exchangeability_ratio_is_one_when_observed_equals_null():
    rng = np.random.default_rng(404)
    residuals = rng.normal(size=(3000, 40))
    config = Eiger2Config(
        window=1, null_min_lag=5, random_seed=7, null_replicates=200,
    )
    tracks = null_tracks(residuals, config)
    # use one of the null's own held-out-eligible values as a stand-in "observed"
    # track drawn from the identical distribution -- ratio should be close to 1
    pooled_all = np.concatenate([t[np.isfinite(t) & e] for t, e, _ in tracks])
    observed_like_null = rng.choice(pooled_all, size=2000, replace=True)
    eligible = np.ones(2000, dtype=bool)
    result = null_exchangeability(tracks, observed_like_null, eligible, config)
    assert result["null_exchangeability_ratio"] == pytest.approx(1.0, abs=0.3)


def test_null_exchangeability_deterministic_split():
    rng = np.random.default_rng(505)
    residuals = rng.normal(size=(2000, 30))
    config = Eiger2Config(window=1, random_seed=9, null_replicates=100)
    tracks = null_tracks(residuals, config)
    statistic = rng.normal(size=1999)
    eligible = np.ones(1999, dtype=bool)
    first = null_exchangeability(tracks, statistic, eligible, config)
    second = null_exchangeability(tracks, statistic, eligible, config)
    assert first == second


# --------------------------------------------------------------------------
# distant-lag threshold diagnostics
# --------------------------------------------------------------------------


def test_null_threshold_reports_exchangeability_when_statistic_given():
    rng = np.random.default_rng(606)
    residuals = rng.normal(size=(2000, 30))
    config = Eiger2Config(window=1, random_seed=3, null_replicates=100)
    statistic = rng.normal(size=1999)
    eligible = np.ones(1999, dtype=bool)
    threshold, diagnostics = null_threshold(
        residuals, config, statistic=statistic, eligible=eligible,
    )
    assert "null_exchangeability_ratio" in diagnostics
    assert diagnostics["null_exchangeability_ratio"] > 0


def test_null_threshold_omits_exchangeability_when_statistic_not_given():
    rng = np.random.default_rng(707)
    residuals = rng.normal(size=(2000, 30))
    config = Eiger2Config(window=1, random_seed=3, null_replicates=100)
    threshold, diagnostics = null_threshold(residuals, config)
    assert "null_exchangeability_ratio" not in diagnostics


# --------------------------------------------------------------------------
# distant-lag null tracks
# --------------------------------------------------------------------------


def test_null_tracks_returns_track_eligible_segments_triples():
    # Spec §2b.2: the third element used to be a single scalar lag, one per
    # replicate. One replicate can now span several chromosomes, each with
    # its own independently-drawn lag, so it is a list of
    # (chromosome, lag, start, length) segments describing which slice of
    # `track`/`eligible` came from which chromosome, at what lag, and at
    # what genome-wide row offset -- self-describing rather than a single
    # int that could no longer capture a multi-chromosome replicate.
    rng = np.random.default_rng(11)
    residuals = rng.normal(size=(3000, 20))
    config = Eiger2Config(window=1, random_seed=1, null_replicates=10)
    tracks = null_tracks(residuals, config)
    assert len(tracks) > 0
    for track, eligible, segments in tracks:
        assert track.shape == eligible.shape
        assert segments  # at least one chromosome contributed
        assert sum(length for _name, _lag, _start, length in segments) == track.size
        for name, lag, start, length in segments:
            assert isinstance(name, str)
            assert isinstance(lag, int)
            assert isinstance(start, int)
            assert isinstance(length, int)
            assert length > 0
