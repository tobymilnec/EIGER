"""Spec §2b.3: infeasible chromosomes are measured, not silently excluded.

A chromosome with `n_c // (span - 1) < null_min_lag` admits no feasible
locality-breaking lag and contributes nothing to the null in any replicate
(`null_tracks()`, spec §2b.2). Rather than dropping it behind a fixed,
arbitrary warning threshold, its share of the run's own eligible observed
edges is measured (`chromosome_lag_feasibility`) and escalates to a
structural gate (`manifest["gates"]["structural"]["chromosome_exposure"]`,
spec §2a.2) when it exceeds the explicit, independently tunable
`max_missing_null_exposure` config field (default 0.05) -- not derived from
`min_null_coverage`, a different quantity (see docs/FDR.md §15.3).
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
    _member_step_lag_limits,
    call_dmrs,
    chromosome_lag_feasibility,
    member_step_chromosome_feasibility,
)


def _build(sizes: dict[str, int], *, seed: int = 0, n_samples: int = 10):
    rng = np.random.default_rng(seed)
    chrom_col: list[str] = []
    start_col: list[int] = []
    for name, n in sizes.items():
        chrom_col.extend([name] * n)
        start_col.extend((np.arange(n) * 20).tolist())
    n_total = len(chrom_col)
    matrix = np.clip(
        0.5 + 0.05 * rng.standard_normal((n_total, n_samples)), 0.02, 0.98
    )
    locations = pd.DataFrame({
        "chr": chrom_col, "start": start_col,
        "end": [s + 1 for s in start_col],
    })
    return matrix, locations


def test_chromosome_lag_feasibility_identifies_infeasible_chromosomes():
    _matrix, locations = _build({"chrBig": 2000, "chrTiny": 40})
    config = Eiger2Config(window=4, null_min_lag=50)
    feasibility = chromosome_lag_feasibility(locations, config)

    # l_max = (n_c - 2) // (span - 1), span = max(window, 2) = 4: the largest
    # lag at which _lag_null_track still has >= 2 usable positions, not just
    # n_c // (span - 1) (which overestimates by enough that a lag drawn at
    # the reported boundary can still leave < 2 usable -- the bug this
    # formula fixes).
    assert feasibility["chrBig"]["feasible"] is True
    assert feasibility["chrBig"]["n_cpg"] == 2000
    assert feasibility["chrBig"]["l_max"] == (2000 - 2) // 3

    assert feasibility["chrTiny"]["feasible"] is False
    assert feasibility["chrTiny"]["n_cpg"] == 40
    assert feasibility["chrTiny"]["l_max"] == (40 - 2) // 3

    # eligible_observed_edges: one fewer than n_cpg per chromosome (n_cpg-1
    # internal edges), since every position here is gap-eligible.
    assert feasibility["chrBig"]["eligible_observed_edges"] == 1999
    assert feasibility["chrTiny"]["eligible_observed_edges"] == 39


def test_chromosome_exposure_gate_passes_when_missing_exposure_is_small():
    matrix, locations = _build({"chrBig": 2000, "chrTiny": 40})
    config = Eiger2Config(
        window=4, null_min_lag=50, null_replicates=20,
        max_missing_null_exposure=0.5, region_null_mode="lag",
    )
    _regions, _values, manifest = call_dmrs(matrix, locations, config=config)
    gate = manifest["gates"]["structural"]["chromosome_exposure"]

    assert gate["infeasible_chromosomes"] == ["chrTiny"]
    assert gate["floor"] == 0.5
    assert gate["missing_exposure"] < gate["floor"]
    assert gate["passed"] is True
    # a small, isolated infeasible chromosome must not fail the RUN closed
    assert manifest["status"] != "inference_unreliable"


def test_chromosome_exposure_gate_escalates_when_missing_exposure_is_material():
    # Two infeasible chromosomes together outweigh the one feasible one --
    # note feasibility is monotone in size for a fixed null_min_lag/span,
    # so an infeasible chromosome is always individually SMALLER than a
    # feasible one; material missing exposure has to come from several
    # small ones together; here it is engineered directly, not left to
    # chance.
    matrix, locations = _build({"chrBig": 200, "chrTiny1": 140, "chrTiny2": 140})
    config = Eiger2Config(
        window=4, null_min_lag=50, null_replicates=20,
        max_missing_null_exposure=0.5, region_null_mode="lag",
    )
    _regions, _values, manifest = call_dmrs(matrix, locations, config=config)
    structural = manifest["gates"]["structural"]
    gate = structural["chromosome_exposure"]

    assert set(gate["infeasible_chromosomes"]) == {"chrTiny1", "chrTiny2"}
    assert gate["floor"] == 0.5
    assert gate["missing_exposure"] > gate["floor"]
    assert gate["passed"] is False
    # every OTHER structural gate can independently still pass -- isolates
    # this gate as the one actually doing the failing, not a side effect of
    # something else
    assert structural["surviving_null_tracks"]["passed"] is True
    assert manifest["status"] == "inference_unreliable"


def test_max_missing_null_exposure_defaults_to_a_strict_explicit_five_percent():
    # Not derived from min_null_coverage (which defaults to 0.25, and would
    # imply a permissive 75% floor): an explicit, documented, independently
    # tunable field, defaulting to a much stricter 5%.
    assert Eiger2Config().max_missing_null_exposure == 0.05


def test_member_step_feasibility_reports_preferred_fallback_and_infeasible():
    _matrix, locations = _build({"chrBig": 200, "chrFallback": 60, "chrTiny": 30})
    config = Eiger2Config(
        window=4, null_min_lag=5, min_subtrack_length=10,
    )
    edge = chromosome_lag_feasibility(locations, config)
    member = member_step_chromosome_feasibility(locations, config)

    assert member["chrBig"]["construction"] == "preferred"
    assert member["chrFallback"]["construction"] == "fallback"
    assert member["chrFallback"]["feasible"] is True
    assert member["chrFallback"]["preferred_l_max"] == 60 // (10 + 4 - 1)
    assert member["chrFallback"]["fallback_l_max"] == 60 // (4 + 4 - 1)
    assert edge["chrTiny"]["feasible"] is True
    assert member["chrTiny"]["construction"] == "infeasible"
    assert member["chrTiny"]["eligible_observed_edges"] == 29


def test_member_step_region_exposure_gate_passes_when_missing_share_is_small():
    matrix, locations = _build({"chrBig": 200, "chrSmall": 30})
    config = Eiger2Config(
        window=4, null_min_lag=5, min_subtrack_length=10,
        null_replicates=4, n_lag_draws=2, region_null_mode="member_step",
        max_missing_null_exposure=0.2, min_null_values=1,
        min_null_coverage=0.0, m_opportunity=0,
    )
    _regions, _values, manifest = call_dmrs(matrix, locations, config=config)
    structural = manifest["gates"]["structural"]
    edge_gate = structural["chromosome_exposure"]
    region_gate = structural["region_null_chromosome_exposure"]

    assert edge_gate["infeasible_chromosomes"] == []
    assert region_gate["infeasible_chromosomes"] == ["chrSmall"]
    assert region_gate["missing_exposure"] == pytest.approx(29 / (199 + 29))
    assert region_gate["passed"] is True
    assert manifest["status"] != "inference_unreliable"


def test_member_step_region_exposure_gate_fails_when_missing_share_is_material():
    matrix, locations = _build({
        "chrBig": 100,
        "chrSmall1": 30,
        "chrSmall2": 30,
        "chrSmall3": 30,
        "chrSmall4": 30,
    })
    config = Eiger2Config(
        window=4, null_min_lag=5, min_subtrack_length=10,
        null_replicates=4, n_lag_draws=2, region_null_mode="member_step",
        max_missing_null_exposure=0.5, min_null_values=1,
        min_null_coverage=0.0, m_opportunity=0,
    )
    regions, _values, manifest = call_dmrs(matrix, locations, config=config)
    structural = manifest["gates"]["structural"]
    edge_gate = structural["chromosome_exposure"]
    region_gate = structural["region_null_chromosome_exposure"]

    assert edge_gate["passed"] is True
    assert edge_gate["infeasible_chromosomes"] == []
    assert set(region_gate["infeasible_chromosomes"]) == {
        "chrSmall1", "chrSmall2", "chrSmall3", "chrSmall4",
    }
    assert region_gate["missing_exposure"] == pytest.approx(
        (29 * 4) / (99 + 29 * 4)
    )
    assert region_gate["passed"] is False
    assert manifest["status"] == "inference_unreliable"
    if len(regions):
        assert (regions["fdr_status"] == "unavailable_run").all()


def test_icgc_shaped_counts_leave_only_chry_and_chrm_truly_infeasible():
    """Regression against the chromosome counts recorded by the Stage 2c run.

    This is count-only: it exercises the exact feasibility algebra without
    allocating the 12 GB cohort matrix or a 2.3-million-row locations frame.
    """

    n_cpg = {
        "chr1": 210482, "chr2": 167359, "chr3": 121833, "chr4": 96032,
        "chr5": 114058, "chr6": 109008, "chr7": 128230, "chr8": 91353,
        "chr9": 98145, "chr10": 113083, "chr11": 122088, "chr12": 110939,
        "chr13": 54509, "chr14": 73097, "chr15": 71175, "chr16": 106795,
        "chr17": 124620, "chr18": 45754, "chr19": 143622, "chr20": 73129,
        "chr21": 29563, "chr22": 59262, "chrX": 29759, "chrY": 1874,
        "chrM": 222,
    }
    eligible_edges = {
        "chr1": 179322, "chr2": 140941, "chr3": 101150, "chr4": 80995,
        "chr5": 95957, "chr6": 91527, "chr7": 110681, "chr8": 76539,
        "chr9": 84264, "chr10": 96403, "chr11": 104918, "chr12": 94263,
        "chr13": 46065, "chr14": 62333, "chr15": 60585, "chr16": 94595,
        "chr17": 110542, "chr18": 38363, "chr19": 131011, "chr20": 63179,
        "chr21": 25468, "chr22": 52087, "chrX": 22034, "chrY": 1596,
        "chrM": 212,
    }
    config = Eiger2Config(window=4, min_cpgs=4, null_min_lag=1000)
    construction = {
        chrom: _member_step_lag_limits(count, config)["construction"]
        for chrom, count in n_cpg.items()
    }
    assert {c for c, mode in construction.items() if mode == "infeasible"} == {
        "chrY", "chrM"
    }
    assert construction["chr1"] == "preferred"
    assert construction["chr4"] == "fallback"
    missing = sum(
        eligible_edges[c] for c, mode in construction.items()
        if mode == "infeasible"
    ) / sum(eligible_edges.values())
    assert missing == pytest.approx(0.00092009, abs=1e-7)
    assert missing < config.max_missing_null_exposure


def test_threshold_by_chromosome_reports_the_sensitivity_half_of_2b3():
    # spec §2b.3's "report the sensitivity... by recomputing on the retained
    # chromosomes only" -- the cheap half: the same null_quantile, per
    # feasible chromosome, alongside the pooled "threshold" it compares
    # against. No second run: both come from the null null_tracks() already
    # built.
    matrix, locations = _build({"chr1": 2000, "chr2": 1500})
    config = Eiger2Config(
        window=4, null_min_lag=50, null_replicates=20, region_null_mode="lag",
    )
    _regions, _values, manifest = call_dmrs(matrix, locations, config=config)
    null_diagnostics = manifest["null"]

    assert set(null_diagnostics["threshold_by_chromosome"]) == {"chr1", "chr2"}
    for chromosome_threshold in null_diagnostics["threshold_by_chromosome"].values():
        assert np.isfinite(chromosome_threshold)
        # each chromosome's own quantile should be in the same ballpark as
        # the pooled one -- both drawn from the same i.i.d. noise process
        # here, so a wildly different value would indicate a real bug, not
        # just sampling noise
        assert chromosome_threshold == pytest.approx(
            null_diagnostics["threshold"], abs=0.2
        )
