"""Tests for tier two: confirming a region from its own per-sample means.

The question under test is not "is this region real against the genome
background" -- no self-calibrated null answers that (docs/FDR.md section 18) --
but "do this region's per-sample mean betas come from one population". These
tests pin the four properties that makes it usable: a planted two-population
region confirms, a unimodal region carrying drift does not, the BH step is
monotone, and the reported effect size recovers what was planted.

Calibration on realistic data is measured outside this repository, in the
eiger-benchmark provenance directories named in docs/CONFIRM.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eiger2.confirm import (
    MIN_CONFIRM_SAMPLES,
    benjamini_hochberg,
    confirm_regions,
    region_sample_means,
    two_component_fit,
)


def _region(
    values_by_sample: np.ndarray, n_cpg: int, noise: float, seed: int
) -> np.ndarray:
    """An ``(n_cpg, n_samples)`` beta block around each sample's own level."""

    rng = np.random.default_rng(seed)
    block = values_by_sample[None, :] + rng.normal(
        0.0, noise, size=(n_cpg, values_by_sample.size)
    )
    return np.clip(block, 0.01, 0.99)


def test_a_planted_two_population_region_confirms():
    levels = np.concatenate([np.full(30, 0.30), np.full(10, 0.55)])
    betas = _region(levels, n_cpg=12, noise=0.01, seed=1)
    result = confirm_regions(betas, np.array([[0, 11]]))
    assert result["n_samples_tested"][0] == 40
    assert result["confirmed"][0] is True
    assert result["confirm_qvalue"][0] <= 0.05


def test_a_unimodal_region_carrying_drift_does_not_confirm():
    # One population, wide: a spread of sample levels is not two populations,
    # and a region-level caller that confirmed this would confirm everything.
    rng = np.random.default_rng(2)
    levels = np.clip(rng.normal(0.45, 0.08, size=40), 0.05, 0.95)
    betas = _region(levels, n_cpg=12, noise=0.01, seed=2)
    result = confirm_regions(betas, np.array([[0, 11]]))
    assert result["confirmed"][0] is False
    assert result["confirm_qvalue"][0] > 0.05


def test_delta_hat_recovers_a_planted_two_tenths():
    levels = np.concatenate([np.full(20, 0.35), np.full(20, 0.55)])
    betas = _region(levels, n_cpg=20, noise=0.01, seed=3)
    result = confirm_regions(betas, np.array([[0, 19]]))
    assert abs(abs(float(result["delta_hat"][0])) - 0.20) <= 0.03
    assert abs(float(result["f_hat"][0]) - 0.5) <= 0.05
    assert float(result["direction"][0]) in (-1.0, 1.0)


def test_two_component_fit_is_deterministic_and_reports_the_minor_component():
    values = np.concatenate([np.full(34, -1.0), np.full(6, 1.0)])
    first = two_component_fit(values)
    second = two_component_fit(values)
    assert first == second
    assert first["f_hat"] == pytest.approx(6 / 40, abs=0.02)
    assert first["direction"] == 1


def test_benjamini_hochberg_is_monotone_and_ignores_untested_regions():
    pvalues = np.array([0.001, 0.02, np.nan, 0.04, 0.9])
    qvalues = benjamini_hochberg(pvalues)
    tested = np.isfinite(pvalues)
    # order preserved, never below the p-value, never above one
    assert np.all(np.diff(qvalues[tested][np.argsort(pvalues[tested])]) >= 0)
    assert np.all(qvalues[tested] >= pvalues[tested] - 1e-12)
    assert np.all(qvalues[tested] <= 1.0)
    # an untested region neither gets a q-value nor enlarges the family
    assert np.isnan(qvalues[2])
    assert qvalues[0] == pytest.approx(0.001 * 4 / 1)


def test_a_region_too_few_samples_observe_is_not_tested():
    levels = np.concatenate([np.full(30, 0.30), np.full(10, 0.55)])
    betas = _region(levels, n_cpg=12, noise=0.01, seed=4)
    # only 8 samples observe enough of the region to contribute a mean
    betas[:, 8:] = np.nan
    result = confirm_regions(betas, np.array([[0, 11]]))
    assert result["n_samples_tested"][0] < MIN_CONFIRM_SAMPLES
    assert result["confirmed"][0] is pd.NA
    assert np.isnan(result["confirm_pvalue"][0])
    assert np.isnan(result["confirm_qvalue"][0])


def test_a_sample_observing_under_half_the_region_is_excluded():
    levels = np.full(40, 0.4)
    betas = _region(levels, n_cpg=10, noise=0.01, seed=5)
    betas[:6, 0] = np.nan  # sample 0 observes 4 of 10
    values, tested = region_sample_means(betas, 0, 9)
    assert not tested[0]
    assert values.size == 39


def test_call_dmrs_carries_the_confirmation_columns_and_the_flag_removes_them():
    from eiger2 import Eiger2Config, call_dmrs
    from eiger2.benchmark import simulate

    matrix, locations, _truth = simulate(
        n_samples=60, strata=[(0.25, 0.4)], replicates=6, spacing=300,
        dmr_length=12, seed=11,
    )
    columns = [
        "confirm_pvalue", "confirm_qvalue", "confirmed", "delta_hat", "f_hat",
        "direction", "n_samples_tested",
    ]
    # The fixture is far too small for the default lag null's 1,000-row
    # minimum lag, so this test exercises tier two through the cheap
    # observed-quantile threshold; the columns are the same on both paths.
    config = Eiger2Config(
        window=4, random_seed=11,
        edge_threshold="observed_quantile", null_quantile=0.99,
    )
    regions, _values, manifest = call_dmrs(matrix, locations, config=config)
    assert len(regions)
    for column in columns:
        assert column in regions
    assert regions["confirmed"].dtype.name == "boolean"
    assert manifest["confirmation"]["test"] == "hartigan_dip_interpolated_table"
    assert manifest["n_confirmed"] == int(regions["confirmed"].fillna(False).sum())

    off, _values, off_manifest = call_dmrs(
        matrix, locations,
        config=Eiger2Config(
            window=4, random_seed=11, confirm_regions=False,
            edge_threshold="observed_quantile", null_quantile=0.99,
        ),
    )
    for column in columns:
        assert column not in off
    assert off_manifest["confirmation"]["test"] is None
    # turning tier two off changes nothing about tier one
    assert off["region_id"].tolist() == regions["region_id"].tolist()
    assert off["rank"].tolist() == regions["rank"].tolist()
