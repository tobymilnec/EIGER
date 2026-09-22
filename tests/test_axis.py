"""Axis-effect building blocks (eiger2.axis, eiger2.core.latent_axes).

Carried over from EIGER-clean fdr-closeout-v2 tests/test_confirm_axis.py
(e050f8f): the tests of the pure functions that the release keeps unchanged --
candidate axes versus the deflation basis, candidate masking, orientation, the
exact expit contrast, the two-direction 2K family, the interval, the
minimum-effect margin, the K + 10 requirement and BH. Imports now point at
eiger2.axis; tests of the retired dip/split/composite layer were not carried
over.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from eiger2.axis import (  # noqa: E402
    DELTA_MIN,
    MIN_CONFIRM_SAMPLES,
    axis_margin_test,
    axis_reference_points,
    benjamini_hochberg,
    orient_axes,
)
from eiger2.core import (  # noqa: E402
    Eiger2Config,
    latent_axes,
)


def _orthonormal_axes(n_axes: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n_samples, n_axes))
    q, _ = np.linalg.qr(raw)
    return np.ascontiguousarray(q.T)


def _concentrated_residuals(
    axes: np.ndarray, n_cpg: int, seed: int, hot_fraction: float = 0.04
) -> np.ndarray:
    """Residuals whose axis loading is concentrated in marker rows.

    A nuisance factor spreads its loading over the whole genome and is
    deflated; a biological subgroup concentrates it and is kept. This builds
    the second kind, so the components come back ``biological_kept`` -- the
    ICGC shape -- while still loading every row a little, so that masking the
    candidates out does not delete the axis.
    """

    rng = np.random.default_rng(seed)
    n_axes, n_samples = axes.shape
    residuals = 0.01 * rng.normal(size=(n_cpg, n_samples))
    hot = rng.choice(n_cpg, size=max(1, int(hot_fraction * n_cpg)), replace=False)
    for index in range(n_axes):
        amplitude = np.full(n_cpg, 0.01)
        amplitude[hot] = 1.0
        rng.shuffle(amplitude)
        # Equal amplitude on every axis: parallel analysis must retain all of
        # them, which is the ICGC shape this fixture exists to reproduce.
        residuals += np.outer(amplitude, axes[index]) * 0.6
    return residuals


def _graded_levels(axis: np.ndarray, centre: float, contrast: float) -> np.ndarray:
    """Per-sample beta levels whose p10-to-p90 span across ``axis`` is ``contrast``."""

    low, high = np.quantile(axis, [0.10, 0.90])
    slope = contrast / (high - low)
    return np.clip(centre + slope * (axis - np.median(axis)), 0.01, 0.99)


def test_candidate_axes_and_the_deflation_basis_are_separate_objects():
    axes = _orthonormal_axes(3, 50, seed=1)
    residuals = _concentrated_residuals(axes, n_cpg=600, seed=2)
    config = Eiger2Config(deflation_thin=1)
    result = latent_axes(residuals, config)

    assert result.candidates.shape[0] == len(result.verdicts)
    assert result.candidates.shape[0] >= 1
    assert set(result.verdicts) <= {"biological_kept", "nuisance_deflated"}
    # the basis is exactly the deflated subset of the candidates
    assert result.basis.shape[0] == int(result.deflation_mask.sum())
    assert np.array_equal(
        result.basis, result.candidates[result.deflation_mask]
    ) or result.basis.size == 0
    # the kept axes are the complement, and the two never overlap
    kept = set(result.kept_indices())
    deflated = {i for i, flag in enumerate(result.deflation_mask) if flag}
    assert kept.isdisjoint(deflated)
    assert kept | deflated == set(range(result.candidates.shape[0]))


def test_an_icgc_shaped_cohort_keeps_sixteen_axes_and_deflates_none():
    """All 16 candidates `biological_kept`: K = 16 for confirmation, 0 for deflation.

    This is the ICGC shape (`provenance/eiger_fdr_closeout_v2/BASELINE.md`
    section 4), and the case the pre-refactor code could not express: the
    deflation basis is empty, so "the basis Eiger already has" is nothing,
    while sixteen candidate axes describe the cohort.
    """

    axes = _orthonormal_axes(16, 120, seed=5)
    residuals = _concentrated_residuals(axes, n_cpg=3000, seed=6, hot_fraction=0.02)
    config = Eiger2Config(deflation_thin=1, deflation_max_components=16)
    result = latent_axes(residuals, config)
    assert result.candidates.shape[0] == 16
    assert list(result.verdicts) == ["biological_kept"] * 16
    assert result.basis.shape[0] == 0
    assert result.diagnostics["n_candidates"] == 16
    assert result.diagnostics["n_components"] == 0
    assert result.kept().shape == (16, 120)


def test_orientation_is_deterministic_and_label_free():
    axes = _orthonormal_axes(4, 30, seed=7)
    flipped = axes * np.array([1.0, -1.0, 1.0, -1.0])[:, None]
    assert np.allclose(orient_axes(axes), orient_axes(flipped))
    for row in orient_axes(np.vstack([axes, flipped])):
        assert row[int(np.argmax(np.abs(row)))] > 0
    # a permutation of the SAMPLES permutes the scores and nothing else
    order = np.random.default_rng(8).permutation(30)
    assert np.allclose(
        orient_axes(axes)[:, order], orient_axes(axes[:, order])
    )


def test_candidate_rows_are_masked_out_of_the_confirmation_axes_and_counted():
    axes = _orthonormal_axes(2, 40, seed=9)
    residuals = _concentrated_residuals(axes, n_cpg=800, seed=10)
    config = Eiger2Config(deflation_thin=4)
    mask = np.ones(800, dtype=bool)
    mask[100:300] = False  # 200 rows "inside a candidate"
    masked = latent_axes(residuals, config, row_mask=mask, allow_relaxation=False)
    offered = masked.diagnostics["thin_rows_offered_before_mask"]
    assert offered == len(range(0, 800, 4))
    assert masked.diagnostics["thin_rows_masked_out"] == 50  # 200 / 4
    assert masked.diagnostics["thin_rows_offered"] == offered - 50
    assert masked.diagnostics["relaxation_allowed"] is False
    # and the masked rows genuinely took no part: a full-data run differs
    full = latent_axes(residuals, config)
    assert not np.allclose(full.candidates[0], masked.candidates[0])


def test_too_few_masked_rows_gives_k_zero_with_a_note_not_a_filled_basis():
    axes = _orthonormal_axes(2, 40, seed=11)
    residuals = _concentrated_residuals(axes, n_cpg=400, seed=12)
    config = Eiger2Config(deflation_thin=50)  # 8 thinned rows offered
    mask = np.ones(400, dtype=bool)
    mask[:350] = False
    masked = latent_axes(residuals, config, row_mask=mask, allow_relaxation=False)
    assert masked.diagnostics["basis_source"] == "insufficient_rows_no_relaxation"
    assert masked.diagnostics["n_components"] == 0
    assert masked.kept().shape[0] == 0
    assert "warning" in masked.diagnostics


def test_the_contrast_is_the_exact_expit_difference_not_the_m_one_minus_m_form():
    """Near the beta boundaries the linearisation is wrong by a large factor.

    ``m(1-m) * slope * (p90 - p10)`` is the derivative of expit at the region's
    own mean, so it under- or over-states the contrast whenever the region sits
    away from beta 0.5 or the effect is large. This pins the exact form by
    computing the same contrast by hand.
    """

    rng = np.random.default_rng(13)
    n = 80
    axis = np.sort(rng.normal(size=n))
    axes = axis[None, :]
    low, high = axis_reference_points(axes)
    # a region near beta 0.02: expit is very flat there
    values = -4.0 + 1.5 * axis + 1e-9 * rng.normal(size=n)
    result = axis_margin_test(values, axes, low, high, delta_min=DELTA_MIN)
    expected = (
        1.0 / (1.0 + np.exp(-(-4.0 + 1.5 * high[0])))
        - 1.0 / (1.0 + np.exp(-(-4.0 + 1.5 * low[0])))
    )
    assert result["axis_effect"] == pytest.approx(expected, rel=1e-6)
    mean_beta = 1.0 / (1.0 + np.exp(-values.mean()))
    approximation = mean_beta * (1.0 - mean_beta) * 1.5 * (high[0] - low[0])
    # the approximation is not merely different, it is far out at this level
    assert abs(approximation - expected) > 0.2 * abs(expected)


def test_both_directions_are_tested_and_the_family_is_two_k():
    rng = np.random.default_rng(14)
    n, n_axes = 90, 3
    axes = _orthonormal_axes(n_axes, n, seed=15)
    low, high = axis_reference_points(axes)
    for sign in (1.0, -1.0):
        values = sign * 40.0 * axes[1] + 0.001 * rng.normal(size=n)
        result = axis_margin_test(values, axes, low, high, delta_min=DELTA_MIN)
        assert result["axis_index"] == 1
        assert np.sign(result["axis_effect"]) == np.sign(sign * high[1])
        # the reported direction is the one the effect actually has
        assert result["axis_direction"] == int(np.sign(result["axis_effect"]))
        assert result["axis_n_axes"] == n_axes
        assert result["axis_pvalue"] <= 1.0
        assert result["axis_degrees_of_freedom"] == n - n_axes - 1
        # Bonferroni over 2K, exactly
        assert result["axis_pvalue"] == pytest.approx(
            min(1.0, 2 * n_axes * result["axis_pvalue_raw"])
        )
        assert result["axis_interval_tail_alpha"] == pytest.approx(
            0.05 / (2 * n_axes)
        )


def test_the_interval_carries_the_same_multiplicity_and_brackets_the_effect():
    rng = np.random.default_rng(16)
    n, n_axes = 100, 2
    axes = _orthonormal_axes(n_axes, n, seed=17)
    low, high = axis_reference_points(axes)
    values = 12.0 * axes[0] + 0.2 * rng.normal(size=n)
    result = axis_margin_test(values, axes, low, high, delta_min=DELTA_MIN)
    lower, upper = result["axis_effect_lower"], result["axis_effect_upper"]
    assert lower < result["axis_effect"] < upper
    # the interval excludes delta_min exactly when the one-sided test rejects
    rejects = result["axis_pvalue"] <= 0.05
    excludes = min(
        abs(result["axis_effect_lower"]), abs(result["axis_effect_upper"])
    ) > DELTA_MIN
    assert rejects == excludes


def test_an_effect_of_five_hundredths_does_not_confirm_at_delta_min_a_tenth():
    """The minimum-effect null is the point of the test.

    A 0.05 contrast estimated to arbitrary precision still must not reject
    ``|Delta| <= 0.10``: the null being tested is not ``Delta == 0``.
    """

    rng = np.random.default_rng(18)
    n = 400
    axis = rng.normal(size=n)
    axes = orient_axes(axis[None, :])
    low, high = axis_reference_points(axes)
    levels = _graded_levels(axes[0], centre=0.5, contrast=0.05)
    values = np.log(levels / (1 - levels)) + 1e-6 * rng.normal(size=n)
    result = axis_margin_test(values, axes, low, high, delta_min=0.10)
    assert abs(result["axis_effect"]) == pytest.approx(0.05, abs=0.01)
    assert result["axis_pvalue"] > 0.05
    # and the same effect DOES reject the weaker margin
    weaker = axis_margin_test(values, axes, low, high, delta_min=0.01)
    assert weaker["axis_pvalue"] < 0.05


def test_the_axis_test_needs_k_plus_ten_samples_not_just_the_floor():
    axes = _orthonormal_axes(8, 17, seed=19)
    low, high = axis_reference_points(axes)
    values = np.linspace(-1, 1, 17)
    assert MIN_CONFIRM_SAMPLES <= 17
    assert axis_margin_test(values, axes, low, high) is None  # 17 < 8 + 10
    axes18 = _orthonormal_axes(8, 18, seed=19)
    low18, high18 = axis_reference_points(axes18)
    assert axis_margin_test(np.linspace(-1, 1, 18), axes18, low18, high18) is not None


def test_benjamini_hochberg_handles_an_all_nan_family():
    result = benjamini_hochberg(np.array([np.nan, np.nan]))
    assert np.isnan(result).all()
