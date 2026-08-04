"""Operating-characteristic benchmark for EIGER 2.

Run with ``python -m eiger2.benchmark`` from the ``Python`` directory. Every
number quoted in EIGER2_DESIGN.md is produced here, so the claims can be
re-checked whenever the core changes.

Scenarios
    null            DMR-free matrix with latent structure -> false-positive rate
    prevalence      planted DMRs across a grid of subgroup fraction and effect
    missingness     the same grid with structured dropout
    statistic       correlation vs covariance under a hard, low-SNR setting
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .core import Eiger2Config, call_dmrs


def simulate(
    *,
    n_samples: int = 80,
    n_latent: int = 2,
    latent_loading_sd: float = 0.07,
    noise_sd: float = 0.04,
    strata: list[tuple[float, float]] | None = None,
    replicates: int = 6,
    spacing: int = 400,
    dmr_length: int = 40,
    missing_rate: float = 0.0,
    seed: int = 4,
) -> tuple[np.ndarray, pd.DataFrame, list[tuple[int, int, float, float]]]:
    """A CpG-by-sample matrix with latent structure and planted DMRs."""

    rng = np.random.default_rng(seed)
    strata = strata if strata is not None else []
    n_dmr = len(strata) * replicates
    n_cpg = spacing * max(n_dmr, 1) + 400

    matrix = 0.5 + rng.normal(0, noise_sd, (n_cpg, n_samples))
    if n_latent:
        latent = rng.normal(0, 1, (n_samples, n_latent))
        loading = rng.normal(0, latent_loading_sd, (n_cpg, n_latent))
        matrix += loading @ latent.T

    truth: list[tuple[int, int, float, float]] = []
    position = 200
    for fraction, delta in strata:
        for _ in range(replicates):
            size = max(1, int(round(fraction * n_samples)))
            group = np.zeros(n_samples)
            group[rng.choice(n_samples, size, replace=False)] = 1
            matrix[position : position + dmr_length] += delta * group
            truth.append((position, position + dmr_length - 1, fraction, delta))
            position += spacing

    matrix = np.clip(matrix, 0.001, 0.999)
    if missing_rate > 0:
        # structured dropout: a per-CpG propensity, so missingness clusters
        propensity = np.clip(
            rng.beta(1.2, 1.2, (n_cpg, 1)) * 2 * missing_rate, 0, 0.95
        )
        matrix[rng.random(matrix.shape) < propensity] = np.nan

    locations = pd.DataFrame(
        {
            "chr": "chr1",
            "start": np.arange(n_cpg) * 2,
            "end": np.arange(n_cpg) * 2 + 1,
        }
    )
    return matrix, locations, truth


def evaluate(
    regions: pd.DataFrame,
    truth: list[tuple[int, int, float, float]],
    *,
    min_overlap: int = 20,
) -> tuple[set[int], int]:
    """Return the set of recovered truth indices and the false-region count."""

    called = regions[regions["significant"]] if len(regions) else regions
    recovered: set[int] = set()
    false_regions = 0
    for _, row in called.iterrows():
        matched = False
        for index, (start, end, _, _) in enumerate(truth):
            overlap = (
                min(int(row["cpg_index_end"]), end)
                - max(int(row["cpg_index_start"]), start)
                + 1
            )
            if overlap >= min_overlap:
                recovered.add(index)
                matched = True
        if not matched:
            false_regions += 1
    return recovered, false_regions


def scenario_null(config_kwargs: dict) -> None:
    print("=" * 78)
    print("NULL: no DMRs, two latent factors. Nothing should be called.")
    print("=" * 78)
    for seed in range(4):
        matrix, locations, _ = simulate(strata=[], seed=100 + seed)
        for statistic in ("correlation", "covariance"):
            config = Eiger2Config(statistic=statistic, **config_kwargs)
            regions, _, manifest = call_dmrs(matrix, locations, config=config)
            significant = int(regions["significant"].sum()) if len(regions) else 0
            print(
                f"  seed={100 + seed} {statistic:>11}: "
                f"K={manifest['latent_structure']['n_components']} "
                f"threshold={manifest['null']['threshold']:.4f} "
                f"regions={manifest.get('n_regions', 0):>3} "
                f"significant at q<=0.05 = {significant}"
            )


def scenario_prevalence(config_kwargs: dict, missing_rate: float = 0.0) -> None:
    label = "PREVALENCE" + (
        f" WITH {missing_rate:.0%} STRUCTURED MISSINGNESS" if missing_rate else ""
    )
    print("=" * 78)
    print(f"{label}: 48 planted DMRs across subgroup fraction x effect size")
    print("=" * 78)
    strata = [
        (0.05, 0.35),
        (0.05, 0.55),
        (0.10, 0.30),
        (0.10, 0.50),
        (0.25, 0.25),
        (0.25, 0.40),
        (0.50, 0.20),
        (0.50, 0.35),
    ]
    matrix, locations, truth = simulate(strata=strata, missing_rate=missing_rate)
    results = {}
    for statistic in ("correlation", "covariance"):
        config = Eiger2Config(statistic=statistic, **config_kwargs)
        regions, _, manifest = call_dmrs(matrix, locations, config=config)
        recovered, false_regions = evaluate(regions, truth)
        results[statistic] = recovered
        print(
            f"  {statistic:>11}: K={manifest['latent_structure']['n_components']} "
            f"threshold={manifest['null']['threshold']:.4f} -> "
            f"{len(recovered):>2}/{len(truth)} recovered, {false_regions} false regions"
        )
    print(f"    {'f':>6} {'delta':>6} | {'correlation':>12} {'covariance':>11}")
    for fraction, delta in strata:
        index = [
            i
            for i, (_, _, f, d) in enumerate(truth)
            if f == fraction and d == delta
        ]
        counts = [
            len([i for i in index if i in results[s]])
            for s in ("correlation", "covariance")
        ]
        print(
            f"    {fraction:6.2f} {delta:6.2f} | {counts[0]:>9}/{len(index)} "
            f"{counts[1]:>8}/{len(index)}"
        )


def scenario_statistic(config_kwargs: dict) -> None:
    print("=" * 78)
    print("HARD SETTING: small cohort, rare subgroups, low SNR")
    print("=" * 78)
    strata = [(0.05, 0.25), (0.08, 0.25), (0.10, 0.20), (0.15, 0.20)]
    for n_samples in (20, 40):
        matrix, locations, truth = simulate(
            n_samples=n_samples,
            strata=strata,
            replicates=6,
            noise_sd=0.06,
            seed=17,
        )
        for statistic in ("correlation", "covariance"):
            for window in (1, 6):
                config = Eiger2Config(
                    statistic=statistic, window=window, **config_kwargs
                )
                regions, _, _ = call_dmrs(matrix, locations, config=config)
                recovered, false_regions = evaluate(regions, truth)
                print(
                    f"  n={n_samples:>3} {statistic:>11} window={window}: "
                    f"{len(recovered):>2}/{len(truth)} recovered, "
                    f"{false_regions} false regions"
                )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", "null", "prevalence", "missingness", "statistic"],
    )
    parser.add_argument("--null-replicates", type=int, default=8)
    args = parser.parse_args(argv)
    config_kwargs = {"null_replicates": args.null_replicates}

    if args.scenario in ("all", "null"):
        scenario_null(config_kwargs)
        print()
    if args.scenario in ("all", "prevalence"):
        scenario_prevalence(config_kwargs)
        print()
    if args.scenario in ("all", "missingness"):
        scenario_prevalence(config_kwargs, missing_rate=0.25)
        print()
    if args.scenario in ("all", "statistic"):
        scenario_statistic(config_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
