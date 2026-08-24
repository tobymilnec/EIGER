"""Operating-characteristic suite: EIGER 1 versus EIGER 2.

Six experiments, each answering a question the head-to-head accuracy tables in
``BENCHMARK.md`` do not:

``negative``   No planted signal anywhere. How many regions does each caller
               report per megabase, and does EIGER 2's stated error rate hold
               when the truth is that there is nothing to find? This is the
               first thing a reviewer checks.

``sweep``      The heuristic-insensitivity claim. EIGER 2 argues that ``window``,
               ``max_gap_edges`` and ``min_cpgs`` need no individual
               justification, because the null tracks pass through the same
               segmenter and the reported error rate absorbs whatever the
               heuristics do. That is currently an assertion. Sweeping each
               parameter over a wide range and watching the realised FDR while
               the raw region count moves several-fold is the test of it.

``scaling``    Runtime and peak memory against sample count and CpG count, for
               both callers on identical matrices.

``calibration`` Reliability of the two-groups posterior: bin regions by
               predicted probability, report the observed fraction that are
               genuinely planted.

``exchangeability`` Split-half self-consistency check of the null itself (spec
               §1b.3). The null is split into a calibration half, which sets
               a threshold, and a held-out half, whose exceedance against that
               threshold is a control that must equal ``1 - null_quantile`` by
               construction. The actual gate is the observed track's
               exceedance against the same threshold: the ratio of the two is
               1.0 when the null and the data share the same upper-tail shape,
               and rises when the null is thinner-tailed than the data --
               exactly the failure mode described in docs/FDR.md, "Null
               exchangeability".

``region_rate``     Region-level exposure diagnostic and its gate (spec
               §1c.6). Per-edge support-conditioned thresholding equalises the
               marginal within a support bin but does not by itself guarantee
               observed and null REGIONS form at the same rate per bin; this
               compares the two directly, DMR-free, per bin, and flags any
               bin whose ratio exceeds 2.0.

``support``    Compact deterministic smoke grid for the five post-selection
               biological-support labels. It is a regression screen, not a
               substitute for the confirmatory full-pipeline grid.

Run everything::

    python -m eiger2.benchmark_suite all --out-dir benchmark_v2 --seeds 10

Each command writes tidy CSVs plus a manifest recording the exact parameters,
so any number can be traced back to the run that produced it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time
import tracemalloc
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from compare import run_original

from eiger2.benchmark import simulate
from eiger2.cli import _git_provenance, _peak_memory_mb
from eiger2.core import (
    Eiger2Config,
    biological_support_diagnostics,
    call_dmrs,
    classify_biological_support,
)
from eiger2.manifest_utils import build_manifest_envelope
from eiger2.probability import calibration_error, reliability_table

# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------

#: Simulator settings per scenario. ``strata`` is filled in per experiment so
#: the same nuisance structure can be run with and without planted signal.
SCENARIOS: dict[str, dict] = {
    "clean": dict(n_samples=80, n_latent=0, noise_sd=0.04, missing_rate=0.0),
    "nuisance": dict(n_samples=80, n_latent=2, latent_loading_sd=0.07,
                     noise_sd=0.04, missing_rate=0.0),
    "missingness": dict(n_samples=80, n_latent=2, latent_loading_sd=0.07,
                        noise_sd=0.04, missing_rate=0.25),
    "small_n": dict(n_samples=12, n_latent=1, latent_loading_sd=0.07,
                    noise_sd=0.04, missing_rate=0.0),
    "high_noise": dict(n_samples=40, n_latent=2, latent_loading_sd=0.10,
                       noise_sd=0.08, missing_rate=0.0),
}

#: Planted subgroup fraction and effect size used wherever signal is required.
DEFAULT_STRATA: list[tuple[float, float]] = [
    (0.05, 0.35), (0.10, 0.30), (0.25, 0.25), (0.50, 0.20)
]

#: Covariance thresholds swept for EIGER 1, which has no automatic threshold.
V1_THRESHOLDS: tuple[float, ...] = (0.002, 0.004, 0.012)


def _simulate(scenario: str, seed: int, *, strata, replicates: int, spacing: int):
    return simulate(seed=seed, strata=strata, replicates=replicates,
                    spacing=spacing, **SCENARIOS[scenario])


def _bp_per_cpg(locations: pd.DataFrame) -> float:
    """Genome span covered, in base pairs, from the simulated coordinates."""

    return float(locations["end"].iloc[-1] - locations["start"].iloc[0] + 1)


# --------------------------------------------------------------------------
# region matching
# --------------------------------------------------------------------------


def _classify(
    called: Sequence[tuple[int, int]],
    truth: Sequence[tuple[int, int, float, float]],
    *,
    min_overlap: int,
) -> tuple[int, int, set[int]]:
    """Return ``(n_true_positive, n_false_positive, recovered_truth_indices)``.

    A called region counts as a true positive if it overlaps any planted region
    by at least ``min_overlap`` CpGs. Several calls may match one planted
    region; each is a true positive, and the planted region is recovered once.
    """

    true_positive = 0
    false_positive = 0
    recovered: set[int] = set()
    for start, end in called:
        matched = False
        for index, (t_start, t_end, _, _, *_run) in enumerate(truth):
            overlap = min(end, t_end) - max(start, t_start) + 1
            if overlap >= min_overlap:
                recovered.add(index)
                matched = True
        true_positive += int(matched)
        false_positive += int(not matched)
    return true_positive, false_positive, recovered


def _false_discovery_proportion(false_positive: int, n_called: int) -> float:
    """Return V/max(R, 1), the per-run quantity averaged to estimate FDR."""

    return false_positive / n_called if n_called else 0.0


def _v2_called(regions: pd.DataFrame, *, significant_only: bool = True):
    if len(regions) == 0:
        return []
    frame = regions[regions["significant"]] if significant_only else regions
    return list(
        zip(
            frame["cpg_index_start"].astype(int),
            frame["cpg_index_end"].astype(int),
        )
    )


def _v2_called_by_qvalue(
    regions: pd.DataFrame, column: str, target_fdr: float,
    *, min_prevalence_samples: int = 0,
) -> list[tuple[int, int]]:
    """Evaluate a q-value decision with the configured prevalence floor."""

    if len(regions) == 0:
        return []
    decision = pd.to_numeric(regions[column], errors="coerce") <= target_fdr
    if min_prevalence_samples > 0:
        decision &= regions["minority_size_descriptive"] >= min_prevalence_samples
    frame = regions[decision.fillna(False)]
    return list(zip(
        frame["cpg_index_start"].astype(int),
        frame["cpg_index_end"].astype(int),
    ))


# --------------------------------------------------------------------------
# experiment 1 — no-signal negative control
# --------------------------------------------------------------------------


def negative_control(
    seeds: Sequence[int],
    scenarios: Sequence[str],
    *,
    statistics: Sequence[str] = ("correlation", "covariance"),
    v1_thresholds: Sequence[float] = V1_THRESHOLDS,
    n_cpg_target: int = 20000,
) -> pd.DataFrame:
    """Matrices with no planted DMRs. Everything reported is a false call."""

    rows: list[dict] = []
    for scenario in scenarios:
        for seed in seeds:
            # ``simulate`` sizes the matrix as ``spacing * max(n_dmr, 1) + 400``;
            # with no planted DMRs that is ``spacing + 400``.
            matrix, locations, _ = simulate(
                seed=seed, strata=[], replicates=0,
                spacing=max(400, n_cpg_target - 400), **SCENARIOS[scenario]
            )
            n_cpg = int(matrix.shape[0])
            per_10k = 10000.0 / n_cpg
            megabases = _bp_per_cpg(locations) / 1e6

            def _row(method: str, count: int, **extra) -> dict:
                return {
                    "scenario": scenario, "seed": seed, "method": method,
                    "n_cpg": n_cpg, "n_samples": int(matrix.shape[1]),
                    "false_regions": count,
                    "false_regions_per_10k_cpg": count * per_10k,
                    "false_calls_per_mb": count / megabases,
                    "any_call": bool(count),
                    "megabases": megabases,
                    **extra,
                }

            for threshold in v1_thresholds:
                blocks = run_original(matrix, threshold, 5, impute=True)
                rows.append(_row(f"EIGER 1 @ {threshold:g}", len(blocks)))

            for statistic in statistics:
                config = Eiger2Config(statistic=statistic)
                regions, _, manifest = call_dmrs(matrix, locations, config=config)
                point = _v2_called_by_qvalue(
                    regions, "region_qvalue", config.target_fdr,
                    min_prevalence_samples=config.min_prevalence_samples,
                )
                upper = _v2_called_by_qvalue(
                    regions, "region_qvalue_upper_95", config.target_fdr,
                    min_prevalence_samples=config.min_prevalence_samples,
                )
                rows.append(_row(
                    f"EIGER 2 {statistic}", len(point),
                    point_false_regions=len(point),
                    upper_false_regions=len(upper),
                    point_any_call=bool(point), upper_any_call=bool(upper),
                    candidate_regions=int(manifest.get("n_regions", 0)),
                    threshold=manifest.get("null", {}).get("threshold", np.nan),
                    n_latent_removed=manifest.get(
                        "latent_structure", {}
                    ).get("n_components", np.nan),
                ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# experiment 1b — null exchangeability (spec §1b.3)
# --------------------------------------------------------------------------


def exchangeability(
    seeds: Sequence[int],
    scenarios: Sequence[str],
    *,
    statistic: str = "correlation",
    layout: str = "dense",
    max_cpg_gap_bp: int | None = 300,
) -> pd.DataFrame:
    """The three permanent null-construction gates, per DMR-free scenario/seed/layout.

    Edge-level (spec §1b.3, unchanged): split null replicates into calibration/
    held-out halves, threshold on the calibration half, measure exceedance on the
    held-out half (the control -- must equal ``1 - null_quantile``) and on the
    observed track (the actual gate; median across seeds must be <= 1.25).

    ACF match and region-level exchangeability are computed from the same tracks,
    not a second simulation. They are calibration criteria here because these
    inputs are known to be DMR-free; the same differences are descriptive, not
    hard structural failures, on a signal-containing cohort.
    """

    from eiger2.core import (
        Eiger2Config,
        acf_gate,
        centre_cpgs,
        deflate,
        edge_statistic,
        eligible_edges,
        estimate_latent_basis,
        null_exchangeability,
        null_threshold,
        null_tracks,
        region_exchangeability_gate,
        segment,
    )

    rows: list[dict] = []
    for scenario_name in scenarios:
        for seed in seeds:
            matrix, locations, _ = simulate(
                seed=seed, strata=[], replicates=0, spacing=20000,
                layout=layout, max_cpg_gap_bp=max_cpg_gap_bp,
                **SCENARIOS[scenario_name],
            )
            config = Eiger2Config(
                statistic=statistic, random_seed=seed, max_cpg_gap_bp=max_cpg_gap_bp,
            )
            residuals = centre_cpgs(matrix)
            basis, _ = estimate_latent_basis(residuals, config)
            deflated = deflate(residuals, basis, missing=config.deflation_missing)
            statistic_track = edge_statistic(deflated, config)
            eligible = eligible_edges(locations, config)
            tracks = null_tracks(deflated, config, locations)
            edge_gate = null_exchangeability(tracks, statistic_track, eligible, config)
            acf_result = acf_gate(statistic_track, eligible, tracks, config.window)

            threshold, _ = null_threshold(
                deflated, config, locations, tracks=tracks,
                statistic=statistic_track, eligible=eligible,
            )
            observed_regions = segment(
                statistic_track, eligible, threshold,
                min_cpgs=config.min_cpgs, max_gap_edges=config.max_gap_edges,
            )
            null_regions_total = 0
            null_edge_total = 0
            for track, null_eligible, _lag in tracks:
                usable = null_eligible & np.isfinite(track)
                null_edge_total += int(usable.sum())
                found = segment(
                    track, usable, threshold,
                    min_cpgs=config.min_cpgs, max_gap_edges=config.max_gap_edges,
                )
                null_regions_total += len(found)
            region_gate = region_exchangeability_gate(
                observed_regions=len(observed_regions),
                observed_eligible_edges=int(
                    np.sum(eligible & np.isfinite(statistic_track))
                ),
                null_regions_total=null_regions_total,
                null_eligible_edges=null_edge_total,
                min_null_regions_for_fdr=config.min_null_regions_for_fdr,
            )
            rows.append({
                "layout": layout, "scenario": scenario_name, "seed": seed,
                "n_cpg": int(matrix.shape[0]), "n_samples": int(matrix.shape[1]),
                **edge_gate,
                "acf_gate_passed": acf_result["passed"],
                "acf_diff_by_lag": dict(acf_result["diff"]),
                "region_exchangeability_ratio": region_gate[
                    "region_exchangeability_ratio"
                ],
                "region_exchangeability_null_count_adequate": region_gate[
                    "null_count_adequate"
                ],
                "region_exchangeability_null_regions_total": region_gate[
                    "null_regions_total"
                ],
                "region_exchangeability_observed_regions": region_gate[
                    "observed_regions"
                ],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# experiment 2 — heuristic-insensitivity sweep
# --------------------------------------------------------------------------


#: Parameters swept, and the values each takes. The sweep is one-at-a-time
#: around the default so the effect of each is separable.
SWEEP_GRID: dict[str, tuple] = {
    "window": (1, 2, 4, 6, 8, 12),
    "max_gap_edges": (0, 1, 2, 4, 8),
    "min_cpgs": (2, 3, 4, 6, 10, 20),
}


def heuristic_sweep(
    seeds: Sequence[int],
    scenarios: Sequence[str],
    *,
    statistic: str = "correlation",
    strata: list | None = None,
    replicates: int = 6,
    spacing: int = 400,
    min_overlap: int = 20,
    target_fdr: float = 0.05,
) -> pd.DataFrame:
    """Sweep each segmentation heuristic; watch region count against realised FDR.

    The claim under test: because the null tracks are segmented with the same
    parameters as the data, the realised FDR should stay near ``target_fdr``
    across the sweep even where the raw number of candidate regions changes
    several-fold.
    """

    strata = strata if strata is not None else DEFAULT_STRATA
    rows: list[dict] = []
    for scenario in scenarios:
        for seed in seeds:
            matrix, locations, truth = _simulate(
                scenario, seed, strata=strata, replicates=replicates,
                spacing=spacing
            )
            for parameter, values in SWEEP_GRID.items():
                for value in values:
                    kwargs = {parameter: value, "statistic": statistic,
                              "target_fdr": target_fdr}
                    config = Eiger2Config(**kwargs)
                    started = time.perf_counter()
                    regions, _, manifest = call_dmrs(
                        matrix, locations, config=config
                    )
                    elapsed = time.perf_counter() - started

                    candidates = int(manifest.get("n_regions", 0))
                    called = _v2_called_by_qvalue(
                        regions, "region_qvalue", target_fdr,
                        min_prevalence_samples=config.min_prevalence_samples,
                    )
                    upper_called = _v2_called_by_qvalue(
                        regions, "region_qvalue_upper_95", target_fdr,
                        min_prevalence_samples=config.min_prevalence_samples,
                    )
                    true_positive, false_positive, recovered = _classify(
                        called, truth, min_overlap=min_overlap
                    )
                    upper_tp, upper_fp, upper_recovered = _classify(
                        upper_called, truth, min_overlap=min_overlap
                    )
                    n_called = true_positive + false_positive
                    rows.append({
                        "scenario": scenario, "seed": seed,
                        "parameter": parameter, "value": value,
                        "candidate_regions": candidates,
                        "significant_regions": n_called,
                        "true_positive": true_positive,
                        "false_positive": false_positive,
                        "realised_fdr": _false_discovery_proportion(
                            false_positive, n_called
                        ),
                        "recall": len(recovered) / len(truth) if truth else np.nan,
                        "upper_significant_regions": upper_tp + upper_fp,
                        "upper_true_positive": upper_tp,
                        "upper_false_positive": upper_fp,
                        "upper_realised_fdr": _false_discovery_proportion(
                            upper_fp, upper_tp + upper_fp
                        ),
                        "upper_recall": (
                            len(upper_recovered) / len(truth) if truth else np.nan
                        ),
                        "n_truth": len(truth),
                        "target_fdr": target_fdr,
                        "runtime_seconds": elapsed,
                    })
    return pd.DataFrame(rows)


#: max_cpg_gap_bp values swept (master spec §6.1). q90 of chr1's all-CpG
#: inter-gap distribution is ~256bp, so the 300bp default sits on the shoulder
#: of the gap distribution; this has never been examined despite getting the
#: same scrutiny window/min_cpgs get.
GAP_SWEEP_VALUES: tuple[int, ...] = (100, 200, 300, 500, 1000)


def gap_sweep(
    seeds: Sequence[int],
    *,
    layouts: Sequence[str] = ("wgbs_real", "epic_fitted"),
    strata: list | None = None,
    replicates: int = 4,
    spacing: int = 7000,
    dmr_length: int = 8,
    n_samples: int = 60,
    min_overlap: int = 8,
    target_fdr: float = 0.05,
    region_null_mode: str = "member_step",
) -> pd.DataFrame:
    """Sweep ``max_cpg_gap_bp`` on real layouts (master spec §6.1).

    Reports, per layout x seed x gap value: the run-length distribution and
    ``layout_max_detectable_region_cpgs`` (via ``manifest["layout"]``, i.e.
    ``describe_layout``), the candidate region count, and the realised FDR
    (matching ``heuristic_sweep``'s convention -- a single planted-signal run's
    empirical false-positive rate, not a DMR-free calibration estimate).
    Requires a full repo checkout (the vendored fixture); see
    ``eiger2.benchmark._fixture_dir``.

    ``spacing=7000`` (default): with ``replicates=4`` and the 4-tuple
    ``DEFAULT_STRATA``, this gives ``n_cpg = spacing * 16 + 400 ~= 112,400`` --
    comfortably above ``region_null_mode="member_step"``'s genome-scale
    minimum-input floor (``null_min_lag * (min_subtrack_length + window - 1)``
    = 103,000 at defaults). Below Stage 1e Task 1 (master spec §8), which
    fixed the null-adequacy floor from a per-stratum to an aggregate check,
    ``realised_fdr`` was structurally ``NaN`` at every gap value on both real
    layouts even at genome scale (`results_gap_sweep/PROVISIONAL.md`) -- the
    fix, not a scale change, is what makes this column populate now; the
    scale change (400 -> 7000) is what makes ``region_null_mode="member_step"``
    (the shipped default, used here rather than overridden to ``"lag"``)
    reachable at all on this sweep's matrix size.
    """

    strata = strata if strata is not None else DEFAULT_STRATA
    rows: list[dict] = []
    for layout in layouts:
        for seed in seeds:
            for gap in GAP_SWEEP_VALUES:
                matrix, locations, truth = simulate(
                    n_samples=n_samples, strata=strata, replicates=replicates,
                    spacing=spacing, dmr_length=dmr_length, layout=layout,
                    max_cpg_gap_bp=gap, seed=seed,
                )
                config = Eiger2Config(
                    max_cpg_gap_bp=gap, target_fdr=target_fdr,
                    region_null_mode=region_null_mode,
                )
                regions, _, manifest = call_dmrs(matrix, locations, config=config)
                run_lengths = manifest["layout"]["layout_eligible_run_lengths_edges"]
                candidates = int(manifest.get("n_regions", 0))
                called = _v2_called(regions)
                true_positive, false_positive, recovered = _classify(
                    called, truth, min_overlap=min_overlap
                )
                n_called = true_positive + false_positive
                rows.append({
                    "layout": layout, "seed": seed, "max_cpg_gap_bp": gap,
                    "layout_eligible_fraction": manifest["layout"][
                        "layout_eligible_fraction"
                    ],
                    "layout_run_q50_edges": run_lengths["q50"],
                    "layout_run_q90_edges": run_lengths["q90"],
                    "layout_run_max_edges": run_lengths["max"],
                    "layout_max_detectable_region_cpgs": manifest["layout"][
                        "layout_max_detectable_region_cpgs"
                    ],
                    "n_dmrs_requested": locations.attrs.get("n_dmrs_requested"),
                    "n_dmrs_not_plantable": locations.attrs.get(
                        "n_dmrs_not_plantable"
                    ),
                    "candidate_regions": candidates,
                    "significant_regions": n_called,
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "realised_fdr": _false_discovery_proportion(
                        false_positive, n_called
                    ),
                    "recall": len(recovered) / len(truth) if truth else np.nan,
                    "n_truth": len(truth),
                    "target_fdr": target_fdr,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# experiment 3 — runtime and memory scaling, EIGER 1 vs EIGER 2
# --------------------------------------------------------------------------


def _measure(function, *args, **kwargs) -> tuple[object, float, float]:
    """Run ``function`` and return ``(result, seconds, peak_MiB)``.

    ``tracemalloc`` measures Python-level allocation, which is the comparable
    quantity across the two callers; it does not capture allocator overhead.
    """

    tracemalloc.start()
    started = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


def runtime_scaling(
    seeds: Sequence[int],
    *,
    sample_counts: Sequence[int] = (10, 20, 40, 80, 160),
    cpg_counts: Sequence[int] = (5000, 10000, 20000, 40000),
    base_samples: int = 40,
    base_cpgs: int = 10000,
    statistic: str = "correlation",
    v1_threshold: float = 0.012,
) -> pd.DataFrame:
    """One axis varied at a time, both callers on the identical matrix."""

    rows: list[dict] = []

    def _one(n_samples: int, n_cpg: int, seed: int, axis: str) -> None:
        # ``simulate`` sizes the matrix as ``spacing * n_dmr + 400``; solve for
        # the spacing that yields at least the requested number of CpGs, then
        # trim so the axis is exact.
        n_dmr = max(len(DEFAULT_STRATA), 1)
        spacing = max(200, -(-(n_cpg - 400) // n_dmr))
        matrix, locations, _ = simulate(
            n_samples=n_samples, n_latent=2, latent_loading_sd=0.07,
            noise_sd=0.04, strata=DEFAULT_STRATA, replicates=1,
            spacing=spacing, seed=seed,
        )
        matrix = matrix[:n_cpg]
        locations = locations.iloc[:n_cpg].reset_index(drop=True)

        _, seconds, peak = _measure(
            run_original, matrix, v1_threshold, 5, False, True
        )
        rows.append({
            "axis": axis, "seed": seed, "method": "EIGER 1",
            "n_samples": n_samples, "n_cpg": int(matrix.shape[0]),
            "runtime_seconds": seconds, "peak_memory_mib": peak,
        })

        config = Eiger2Config(statistic=statistic)
        (regions, _, manifest), seconds, peak = _measure(
            call_dmrs, matrix, locations, config=config
        )
        rows.append({
            "axis": axis, "seed": seed, "method": "EIGER 2",
            "n_samples": n_samples, "n_cpg": int(matrix.shape[0]),
            "runtime_seconds": seconds, "peak_memory_mib": peak,
            "null_replicates": config.null_replicates,
            "regions": int(manifest.get("n_regions", 0)),
        })

    for seed in seeds:
        for n_samples in sample_counts:
            _one(n_samples, base_cpgs, seed, "samples")
        for n_cpg in cpg_counts:
            _one(base_samples, n_cpg, seed, "cpgs")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# experiment 4 — posterior calibration
# --------------------------------------------------------------------------


def calibration(
    seeds: Sequence[int],
    scenarios: Sequence[str],
    *,
    statistic: str = "correlation",
    null_replicates: int = 200,
    strata: list | None = None,
    replicates: int = 6,
    spacing: int = 400,
    min_overlap: int = 20,
    n_bins: int = 10,
    min_null_regions: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reliability of ``posterior_real``.

    Returns ``(region_level, reliability_table)``. Regions are pooled across
    seeds and scenarios before binning, because a single run rarely produces
    enough regions to populate ten bins.
    """

    strata = strata if strata is not None else DEFAULT_STRATA
    region_rows: list[dict] = []
    for scenario in scenarios:
        for seed in seeds:
            matrix, locations, truth = _simulate(
                scenario, seed, strata=strata, replicates=replicates,
                spacing=spacing
            )
            config = Eiger2Config(
                statistic=statistic,
                null_replicates=null_replicates,
                compute_posterior=True,
                min_null_regions_for_posterior=min_null_regions,
            )
            regions, _, manifest = call_dmrs(matrix, locations, config=config)
            status = manifest.get("posterior", {}).get("status", "not_computed")
            if len(regions) == 0:
                continue
            for _, row in regions.iterrows():
                start = int(row["cpg_index_start"])
                end = int(row["cpg_index_end"])
                is_true = any(
                    min(end, t_end) - max(start, t_start) + 1 >= min_overlap
                    for t_start, t_end, _, _, *_run in truth
                )
                region_rows.append({
                    "scenario": scenario, "seed": seed,
                    "n_cpg": int(row["n_cpg"]),
                    "region_area": float(row["region_area"]),
                    "region_qvalue": float(row["region_qvalue"]),
                    "posterior_real": float(row["region_evidence_score"]),
                    "posterior_extrapolated": bool(row["evidence_score_extrapolated"]),
                    "is_true": bool(is_true),
                    "posterior_status": status,
                    "pi0": manifest.get("posterior", {}).get("pi0", np.nan),
                })

    region_level = pd.DataFrame(region_rows)
    if region_level.empty or region_level["posterior_real"].isna().all():
        return region_level, pd.DataFrame()

    usable = region_level.dropna(subset=["posterior_real"])
    table = reliability_table(
        usable["posterior_real"].to_numpy(),
        usable["is_true"].to_numpy(),
        n_bins=n_bins,
    )
    return region_level, table


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# experiment — region-rate-by-support-bin diagnostic gate (spec §1c.6)
# --------------------------------------------------------------------------


def region_rate_by_bin(
    seeds: Sequence[int],
    scenarios: Sequence[str],
    *,
    statistic: str = "correlation",
    null_replicates: int = 200,
    n_cpg_target: int = 20000,
) -> pd.DataFrame:
    """Per-support-bin observed/null region rate, DMR-free (spec §1c.6).

    Runs with no planted signal (``strata=[]``, ``replicates=0``) so every
    called region is a false call by construction -- the same convention as
    ``negative_control``, including ``n_cpg_target``: at ``spacing=400`` (the
    other experiments' default matrix size) both observed and null region
    counts are so close to zero DMR-free that most bins are structurally
    undefined (0/0) and the gate cannot be evaluated at all -- the genome
    needs to be large enough for a small per-bin false-region rate to be
    visible above zero. ``support_binned_threshold="on"`` forces support
    binning regardless of the ``auto`` dispersion gate, so the diagnostic
    populates on every scenario/seed rather than only the ones that would
    trigger binning on their own; ``call_dmrs`` still reports
    ``support_binning="collapsed_to_single"`` (and no
    ``region_rate_by_support_bin``) on a run whose null is too thin to
    support any bin, which this function skips.

    The gate itself (any bin's ``ratio`` > 2.0) is read off this table, not
    computed here -- see ``command_region_rate``.
    """

    rows: list[dict] = []
    for scenario in scenarios:
        for seed in seeds:
            matrix, locations, _ = simulate(
                seed=seed, strata=[], replicates=0,
                spacing=max(400, n_cpg_target - 400), **SCENARIOS[scenario],
            )
            config = Eiger2Config(
                statistic=statistic,
                null_replicates=null_replicates,
                random_seed=seed,
                support_binned_threshold="on",
            )
            _, _, manifest = call_dmrs(matrix, locations, config=config)
            per_bin = manifest.get("region_rate_by_support_bin", {})
            for bin_index, stats in per_bin.items():
                rows.append({
                    "scenario": scenario, "seed": seed, "bin": bin_index,
                    "support_binning": manifest["support_binning"],
                    **stats,
                })
    return pd.DataFrame(rows)


def _summarise(frame: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    present = [m for m in metrics if m in frame.columns]
    grouped = frame.groupby(by, dropna=False)[present].agg(["mean", "std", "count"])
    grouped.columns = [f"{a}__{b}" for a, b in grouped.columns]
    return grouped.reset_index()


def _manifest(args: argparse.Namespace, extra: dict | None = None) -> dict:
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": args.command,
        "arguments": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in vars(args).items()
            if not callable(v)
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "default_config": asdict(Eiger2Config()),
    }
    if extra:
        manifest.update(extra)
    return manifest


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(frame)} rows)")


# --------------------------------------------------------------------------
# compact Stage 2D biological-support smoke grid
# --------------------------------------------------------------------------


def biological_support_smoke(
    seeds: Sequence[int], *, statistics: Sequence[str] = ("correlation", "covariance")
) -> pd.DataFrame:
    """Run a deterministic smoke test of the five-class support layer."""

    rows: list[dict] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        scenarios: list[tuple[str, np.ndarray, str, int]] = []

        rare = 0.4 + rng.normal(0, 0.01, (8, 80))
        rare[:, 0] += 0.40
        scenarios.append(("genuine_rare_epivariant", rare, "rare_epivariant", 1))

        one_artifact = 0.4 + rng.normal(0, 0.01, (8, 80))
        one_artifact[:, 0] += np.array([.5, .5, .5, .5, -.1, -.1, -.1, -.1])
        scenarios.append(("one_sample_artifact", one_artifact, "outlier_sensitive", 1))

        two_artifact = 0.4 + rng.normal(0, 0.01, (8, 80))
        two_artifact[:, :2] += np.array([.5, .5, .5, .5, -.1, -.1, -.1, -.1])[:, None]
        scenarios.append(("two_sample_artifact", two_artifact, "outlier_sensitive", 2))

        recurrent = 0.4 + rng.normal(0, 0.01, (10, 100))
        recurrent[:, :5] += 0.30
        scenarios.append(("recurrent_subgroup", recurrent, "recurrent_subgroup", 5))

        cohort = 0.4 + rng.normal(0, 0.01, (20, 100))
        cohort[:, :20] += 0.25
        scenarios.append(("cohort_subgroup", cohort, "cohort_supported", 20))

        gradient = np.tile(np.linspace(0.2, 0.8, 80), (12, 1))
        gradient += rng.normal(0, 0.005, gradient.shape)
        scenarios.append(("continuous_gradient", gradient, "diffuse_coordination", 40))

        # Ordinary local background correlation without a discrete sample split.
        background = rng.normal(0, 0.04, (12, 80))
        for row in range(1, background.shape[0]):
            background[row] = 0.8 * background[row - 1] + 0.6 * background[row]
        background += 0.5
        scenarios.append(("local_background_autocorrelation", background,
                          "diffuse_coordination", 0))

        for statistic in statistics:
            config = Eiger2Config(
                statistic=statistic, deflate_components=0, min_paired_samples=3
            )
            for scenario, block, expected, prevalence in scenarios:
                # Exercise pairwise-complete handling under structured missingness.
                evaluated = block.copy()
                evaluated[::3, -1] = np.nan
                means = np.nanmean(evaluated, axis=0)
                centred = evaluated - np.nanmean(
                    evaluated, axis=1, keepdims=True
                )
                diagnostics = biological_support_diagnostics(
                    evaluated, centred, means, config
                )
                predicted = classify_biological_support(
                    diagnostics, coordination_significant=True,
                    n_observed=int(np.isfinite(means).sum()),
                )
                rows.append({
                    "seed": seed, "scenario": scenario, "statistic": statistic,
                    "n_samples": evaluated.shape[1], "length_cpg": evaluated.shape[0],
                    "planted_prevalence_samples": prevalence,
                    "expected_class": expected, "predicted_class": predicted,
                    "classification_correct": predicted == expected,
                    "furthest1_adjacent_score_influence_fraction": diagnostics[
                        "furthest1_adjacent_score_influence_fraction"
                    ],
                    "furthest2_adjacent_score_influence_fraction": diagnostics[
                        "furthest2_adjacent_score_influence_fraction"
                    ],
                    "influence_flag": max(
                        diagnostics["furthest1_adjacent_score_influence_fraction"],
                        diagnostics["furthest2_adjacent_score_influence_fraction"],
                    ) >= 0.5,
                    "winsorised_score_retention": diagnostics[
                        "winsorised_score_retention"
                    ],
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def command_negative(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    seeds = list(range(args.seeds))
    frame = negative_control(seeds, args.scenarios)
    _write(frame, out / "negative_control.csv")
    summary = _summarise(
        frame, ["scenario", "method"],
        ["false_regions", "false_regions_per_10k_cpg", "any_call",
         "candidate_regions", "point_false_regions", "upper_false_regions",
         "point_any_call", "upper_any_call"],
    )
    _write(summary, out / "negative_control_summary.csv")
    compact = summary[[
        "scenario", "method", "false_regions__mean",
        "false_regions_per_10k_cpg__mean", "any_call__mean",
    ]].rename(columns={
        "false_regions__mean": "false_regions",
        "false_regions_per_10k_cpg__mean": "per_10k_cpg",
        "any_call__mean": "frac_runs_with_a_call",
    })
    print(compact.round(3).to_string(index=False))


def command_support(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    frame = biological_support_smoke(list(range(args.seeds)))
    _write(frame, out / "biological_support_smoke.csv")
    summary = _summarise(
        frame, ["scenario", "statistic"],
        ["classification_correct", "influence_flag",
         "furthest1_adjacent_score_influence_fraction",
         "winsorised_score_retention"],
    )
    _write(summary, out / "biological_support_smoke_summary.csv")
    print(summary.to_string(index=False))


def command_sweep(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    seeds = list(range(args.seeds))
    frame = heuristic_sweep(seeds, args.scenarios, statistic=args.statistic)
    _write(frame, out / "heuristic_sweep.csv")
    summary = _summarise(
        frame, ["scenario", "parameter", "value"],
        ["candidate_regions", "significant_regions", "realised_fdr", "recall",
         "upper_significant_regions", "upper_realised_fdr", "upper_recall"],
    )
    _write(summary, out / "heuristic_sweep_summary.csv")
    print(summary.to_string(index=False))


def command_gap_sweep(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    seeds = list(range(args.seeds))
    # "dense" has one giant eligible run regardless of max_cpg_gap_bp -- nothing
    # for this sweep to show -- so it is dropped even if present in --layouts.
    real_layouts = [layout for layout in args.layouts if layout != "dense"]
    frame = gap_sweep(seeds, layouts=real_layouts or ["wgbs_real", "epic_fitted"])
    _write(frame, out / "gap_sweep.csv")
    summary = _summarise(
        frame, ["layout", "max_cpg_gap_bp"],
        ["layout_eligible_fraction", "layout_run_q50_edges",
         "layout_run_q90_edges", "layout_max_detectable_region_cpgs",
         "candidate_regions", "significant_regions", "realised_fdr", "recall",
         "n_dmrs_not_plantable"],
    )
    _write(summary, out / "gap_sweep_summary.csv")
    print(summary.to_string(index=False))


def command_scaling(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    seeds = list(range(args.scaling_seeds))
    frame = runtime_scaling(seeds)
    _write(frame, out / "runtime_scaling.csv")
    summary = _summarise(
        frame, ["axis", "method", "n_samples", "n_cpg"],
        ["runtime_seconds", "peak_memory_mib"],
    )
    _write(summary, out / "runtime_scaling_summary.csv")
    print(summary.to_string(index=False))


def command_calibration(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    seeds = list(range(args.seeds))
    regions, table = calibration(
        seeds, args.scenarios, statistic=args.statistic,
        null_replicates=args.null_replicates,
    )
    _write(regions, out / "calibration_regions.csv")
    if table.empty:
        print("  posterior not estimable on this configuration; "
              "see posterior_status in calibration_regions.csv")
        return
    _write(table, out / "calibration_reliability.csv")
    error = calibration_error(table)
    (out / "calibration_error.json").write_text(json.dumps(error, indent=2))
    print(table.to_string(index=False))
    print(f"  expected calibration error {error['expected_calibration_error']:.4f}")
    print(f"  maximum calibration error  {error['max_calibration_error']:.4f}")


def command_exchangeability(args: argparse.Namespace) -> None:
    """DMR-free calibration diagnostics across every layout and scenario."""

    out = Path(args.out_dir)
    seeds = list(range(args.seeds))
    frame = pd.concat(
        [
            exchangeability(seeds, args.scenarios, statistic=args.statistic, layout=layout)
            for layout in args.layouts
        ],
        ignore_index=True,
    )
    _write(frame, out / "exchangeability.csv")

    print("--- Edge-level gate (spec §1b.3, retained unchanged; median <= 1.25) ---")
    per_cell = frame.groupby(["layout", "scenario"])["null_exchangeability_ratio"]
    print(per_cell.mean().round(4).to_string())
    medians = per_cell.median()
    print("\nper layout x scenario median (the gate):")
    print(medians.round(4).to_string())
    outliers = frame[frame["null_exchangeability_ratio"] > 1.25]
    if len(outliers):
        print(f"\n{len(outliers)} of {len(frame)} per-instance outlier(s) above "
              f"1.25 (reported, not gated)")
    worst = float(medians.max())
    print(f"worst layout x scenario median: {worst:.3f} "
          f"({'PASS' if worst <= 1.25 else 'FAIL'} at gate 1.25)")

    print("\n--- Gate 1: ACF match (master spec §6.2; every lag within +-0.05) ---")
    pass_rate = frame.groupby(["layout", "scenario"])["acf_gate_passed"].mean()
    print(pass_rate.round(4).to_string())
    overall = bool(frame["acf_gate_passed"].all())
    print(f"every instance must pass: {'PASS' if overall else 'FAIL'} "
          f"({int(frame['acf_gate_passed'].sum())} of {len(frame)} instances passed)")

    print("\n--- Gate 2: region-level exchangeability (master spec §6.2; "
          "median <= 3.0) ---")
    n_inf = int(np.isinf(frame["region_exchangeability_ratio"]).sum())
    print(f"{n_inf} of {len(frame)} instances: ratio = inf (null_rate is zero, "
          f"reported explicitly, not hidden)")
    n_inadequate = int((~frame["region_exchangeability_null_count_adequate"]).sum())
    print(f"{n_inadequate} of {len(frame)} instances: pooled null-region count "
          f"below min_null_regions_for_fdr (ratio not estimated from an adequate "
          f"null)")
    region_medians = frame.groupby(["layout", "scenario"])[
        "region_exchangeability_ratio"
    ].median()
    print(region_medians.round(4).to_string())
    worst_region = float(region_medians.max())
    print(f"worst layout x scenario median: {worst_region:.3f} "
          f"({'PASS' if worst_region <= 3.0 else 'FAIL'} at gate 3.0)")


def command_region_rate(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    seeds = list(range(args.seeds))
    frame = region_rate_by_bin(
        seeds, args.scenarios, statistic=args.statistic,
        null_replicates=args.null_replicates,
    )
    _write(frame, out / "region_rate_by_bin.csv")
    if frame.empty:
        print("  no bins populated (support binning did not activate, or "
              "collapsed to single, on every run)")
        return
    summary = frame.groupby(["scenario", "bin"])[
        ["observed_rate", "null_rate", "ratio"]
    ].mean()
    print(summary.round(4).to_string())
    ratios = frame["ratio"].dropna()
    n_undefined = int(frame["ratio"].isna().sum())
    if n_undefined:
        print(f"\n  {n_undefined} bin-instance(s) had an undefined ratio "
              f"(null had zero eligible edges or zero regions -- see the "
              f"'note' column in region_rate_by_bin.csv)")
    if len(ratios):
        worst = float(ratios.max())
        print(f"\nworst bin ratio: {worst:.3f} "
              f"({'PASS' if worst <= 2.0 else 'FAIL'} at gate 2.0)")
    else:
        print("\n  every bin-instance had an undefined ratio; gate not "
              "evaluable from this run")


def command_all(args: argparse.Namespace) -> None:
    for name, function in (
        ("negative", command_negative),
        ("sweep", command_sweep),
        ("scaling", command_scaling),
        ("calibration", command_calibration),
        ("exchangeability", command_exchangeability),
        ("region_rate", command_region_rate),
        ("support", command_support),
    ):
        print("=" * 78)
        print(name.upper())
        print("=" * 78)
        args.command = name
        function(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "negative", "sweep", "scaling", "calibration", "exchangeability",
        "region_rate", "gap_sweep", "support", "all",
    ])
    parser.add_argument(
        "--layouts", nargs="+", default=["dense", "wgbs_real", "epic_fitted"],
        choices=["dense", "wgbs_real", "epic_fitted"],
        help="gap_sweep/exchangeability: layouts to sweep ('dense' is fixture-free; "
             "wgbs_real/epic_fitted need a repo checkout). gap_sweep silently drops "
             "'dense' -- it has no gap sensitivity to show.",
    )
    parser.add_argument("--out-dir", default="benchmark_v2")
    parser.add_argument("--seeds", type=int, default=10,
                        help="number of simulation seeds (0..seeds-1)")
    parser.add_argument("--scaling-seeds", type=int, default=3,
                        help="seeds for the scaling experiment, which is slower")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                        choices=list(SCENARIOS))
    parser.add_argument("--statistic", default="correlation",
                        choices=["correlation", "covariance"])
    parser.add_argument("--null-replicates", type=int, default=200,
                        help="null tracks for the posterior; a density needs "
                             "many more than a tail count does")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    git_provenance = _git_provenance()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "negative": command_negative,
        "sweep": command_sweep,
        "scaling": command_scaling,
        "calibration": command_calibration,
        "exchangeability": command_exchangeability,
        "region_rate": command_region_rate,
        "gap_sweep": command_gap_sweep,
        "support": command_support,
        "all": command_all,
    }
    started = time.perf_counter()
    dispatch[args.command](args)
    elapsed = time.perf_counter() - started

    # ``args.seeds`` and ``args.scaling_seeds`` are both always present (they
    # are parser-level defaults), regardless of which sub-command ran; every
    # experiment enumerates its own seeds as range(0, N), so the base seed is
    # always 0 and the two ranges together are the full derived-seed set.
    envelope = build_manifest_envelope(
        config_dict={
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in vars(args).items()
            if not callable(v)
        },
        git_commit=str(git_provenance["git_commit"]),
        git_worktree_dirty=git_provenance["git_worktree_dirty"],
        git_worktree_fingerprint_sha256=git_provenance[
            "git_worktree_fingerprint_sha256"
        ],
        # Every input is simulated in-process; nothing is read from disk.
        input_checksums={},
        random_seed=0,
        derived_seeds={
            "seeds": list(range(args.seeds)),
            "scaling_seeds": list(range(args.scaling_seeds)),
        },
        runtime_seconds=elapsed,
        peak_memory_mb=_peak_memory_mb(),
        warnings=[],
        # Full-pipeline experiments derive both decisions from the same region
        # q-value/upper-bound fit; no duplicate stochastic run is required.
        fdr_decision="point_and_upper_from_same_run",
    )
    (out / f"{args.command}_manifest.json").write_text(
        json.dumps(_manifest(args, extra=envelope), indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
