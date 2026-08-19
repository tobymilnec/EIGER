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
from eiger2.cli import _git_commit, _peak_memory_mb
from eiger2.core import Eiger2Config, call_dmrs
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
        for index, (t_start, t_end, _, _) in enumerate(truth):
            overlap = min(end, t_end) - max(start, t_start) + 1
            if overlap >= min_overlap:
                recovered.add(index)
                matched = True
        true_positive += int(matched)
        false_positive += int(not matched)
    return true_positive, false_positive, recovered


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
                significant = _v2_called(regions)
                rows.append(_row(
                    f"EIGER 2 {statistic}", len(significant),
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
) -> pd.DataFrame:
    """Split-half null self-consistency check, per spec §1b.3.

    For each DMR-free scenario/seed: generate null replicates, split into
    calibration/held-out halves, threshold on the calibration half, measure
    exceedance on the held-out half (the control -- must equal
    ``1 - null_quantile``) and on the observed track (the actual gate).
    """

    from eiger2.core import (
        Eiger2Config,
        centre_cpgs,
        deflate,
        edge_statistic,
        eligible_edges,
        estimate_latent_basis,
        null_exchangeability,
        null_tracks,
    )

    rows: list[dict] = []
    for scenario_name in scenarios:
        for seed in seeds:
            matrix, locations, _ = simulate(
                seed=seed, strata=[], replicates=0,
                spacing=20000, **SCENARIOS[scenario_name],
            )
            config = Eiger2Config(statistic=statistic, random_seed=seed)
            residuals = centre_cpgs(matrix)
            basis, _ = estimate_latent_basis(residuals, config)
            deflated = deflate(residuals, basis, missing=config.deflation_missing)
            statistic_track = edge_statistic(deflated, config)
            eligible = eligible_edges(locations, config)
            tracks = null_tracks(deflated, config, locations)
            result = null_exchangeability(tracks, statistic_track, eligible, config)
            rows.append({
                "scenario": scenario_name, "seed": seed,
                "n_cpg": int(matrix.shape[0]), "n_samples": int(matrix.shape[1]),
                **result,
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
                    called = _v2_called(regions)
                    true_positive, false_positive, recovered = _classify(
                        called, truth, min_overlap=min_overlap
                    )
                    n_called = true_positive + false_positive
                    rows.append({
                        "scenario": scenario, "seed": seed,
                        "parameter": parameter, "value": value,
                        "candidate_regions": candidates,
                        "significant_regions": n_called,
                        "true_positive": true_positive,
                        "false_positive": false_positive,
                        "realised_fdr": (
                            false_positive / n_called if n_called else np.nan
                        ),
                        "recall": len(recovered) / len(truth) if truth else np.nan,
                        "n_truth": len(truth),
                        "target_fdr": target_fdr,
                        "runtime_seconds": elapsed,
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
                    for t_start, t_end, _, _ in truth
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
         "candidate_regions"],
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


def command_sweep(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    seeds = list(range(args.seeds))
    frame = heuristic_sweep(seeds, args.scenarios, statistic=args.statistic)
    _write(frame, out / "heuristic_sweep.csv")
    summary = _summarise(
        frame, ["scenario", "parameter", "value"],
        ["candidate_regions", "significant_regions", "realised_fdr", "recall"],
    )
    _write(summary, out / "heuristic_sweep_summary.csv")
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
    out = Path(args.out_dir)
    seeds = list(range(args.seeds))
    frame = exchangeability(seeds, args.scenarios, statistic=args.statistic)
    _write(frame, out / "exchangeability.csv")
    summary = frame.groupby("scenario")[
        ["p_null", "p_obs", "null_exchangeability_ratio"]
    ].mean()
    print(summary.round(4).to_string())
    per_scenario = frame.groupby("scenario")["null_exchangeability_ratio"].median()
    print("\nper-scenario median (the gate):")
    print(per_scenario.round(4).to_string())
    outliers = frame[frame["null_exchangeability_ratio"] > 1.25]
    if len(outliers):
        print(f"\n{len(outliers)} per-instance outlier(s) above 1.25 (reported, "
              f"not gated):")
        print(outliers[["scenario", "seed", "null_exchangeability_ratio"]]
              .to_string(index=False))
    worst_median = float(per_scenario.max())
    print(f"\nworst per-scenario median: {worst_median:.3f} "
          f"({'PASS' if worst_median <= 1.25 else 'FAIL'} at gate 1.25)")


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
        "region_rate", "all",
    ])
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
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "negative": command_negative,
        "sweep": command_sweep,
        "scaling": command_scaling,
        "calibration": command_calibration,
        "exchangeability": command_exchangeability,
        "region_rate": command_region_rate,
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
        git_commit=_git_commit(),
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
        # Every Eiger2Config built by this suite leaves fdr_decision at its
        # 'point' default; none of the five experiments override it.
        fdr_decision="point",
    )
    (out / f"{args.command}_manifest.json").write_text(
        json.dumps(_manifest(args, extra=envelope), indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
