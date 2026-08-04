"""Head-to-head benchmark: original EIGER vs EIGER 2.

The original method is given every advantage. Its covariance threshold is swept
over three orders of magnitude and its best operating point is selected *with
knowledge of the ground truth* — an oracle it would not have in practice. EIGER 2
runs once, at its automatic operating point, with no tuning.

Two metric families are reported because they answer different questions.

    CpG-level    Unambiguous. Every retained CpG is a TP/FP/FN/TN against the
                 planted regions. Precision, recall, F1, MCC, Jaccard. This is
                 the primary metric: it has no matching rule and cannot be
                 gamed by fragmenting or merging regions.

    Region-level Interpretable. A called region is a true positive if it
                 overlaps a planted region by at least `min_overlap` CpGs.
                 Region recall counts planted regions recovered; region
                 precision counts called regions that hit something real.
                 Fragmentation = called regions per recovered truth.

Run:  python -m eiger2.compare            (writes CSV + markdown next to the repo)
      python -m eiger2.compare --quick    (2 seeds, fewer thresholds)
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from eiger2.benchmark import simulate
from eiger2.core import Eiger2Config, call_dmrs

# --------------------------------------------------------------------------
# the original method, implemented faithfully from the 2023 processing.py
# --------------------------------------------------------------------------


def original_covariance(matrix: np.ndarray) -> np.ndarray:
    """`covariance_blocks`: np.cov(ddof=1) on consecutive rows, NaN propagates."""

    left = matrix[:-1]
    right = matrix[1:]
    n = matrix.shape[1]
    left_mean = left.mean(axis=1, keepdims=True)
    right_mean = right.mean(axis=1, keepdims=True)
    return ((left - left_mean) * (right - right_mean)).sum(axis=1) / (n - 1)


def original_extract_blocks(
    statistic: np.ndarray,
    cov_threshold: float,
    dmr_length_threshold: int = 5,
    fix_length_bug: bool = False,
) -> list[tuple[int, int]]:
    """`extract_blocks` + `contiguous_regions`.

    The original tests `stop - start >= DMR_length_threshold`, where both index
    the covariance array. A run of E passing edges spans E + 1 CpGs, so the
    threshold is applied to edges, not CpGs. `fix_length_bug=True` uses
    `stop - start + 1`, which is what the docstring describes.
    """

    passing = np.asarray(statistic, dtype=float) >= cov_threshold
    passing = np.where(np.isfinite(statistic), passing, False)
    blocks: list[tuple[int, int]] = []
    index = 0
    while index < passing.size:
        if not passing[index]:
            index += 1
            continue
        start = index
        while index + 1 < passing.size and passing[index + 1]:
            index += 1
        stop = index + 1
        span = (stop - start + 1) if fix_length_bug else (stop - start)
        if span >= dmr_length_threshold:
            blocks.append((start, stop))
        index += 1
    return blocks


def mean_impute(matrix: np.ndarray) -> np.ndarray:
    """CpG-mean imputation, as the 2026 beta does before computing covariance."""

    filled = np.array(matrix, dtype=float, copy=True)
    with np.errstate(invalid="ignore"):
        row_means = np.nanmean(filled, axis=1)
    row_means = np.where(np.isfinite(row_means), row_means, 0.5)
    missing = ~np.isfinite(filled)
    filled[missing] = np.take(row_means, np.nonzero(missing)[0])
    return filled


def run_original(
    matrix: np.ndarray,
    cov_threshold: float,
    dmr_length_threshold: int = 5,
    fix_length_bug: bool = False,
    impute: bool = False,
) -> list[tuple[int, int]]:
    statistic = original_covariance(mean_impute(matrix) if impute else matrix)
    return original_extract_blocks(
        statistic, cov_threshold, dmr_length_threshold, fix_length_bug
    )


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def _mask(intervals, n_cpg: int) -> np.ndarray:
    mask = np.zeros(n_cpg, dtype=bool)
    for first, last in intervals:
        mask[int(first) : int(last) + 1] = True
    return mask


def cpg_metrics(called, truth, n_cpg: int) -> dict:
    predicted = _mask(called, n_cpg)
    actual = _mask([(s, e) for s, e, *_ in truth], n_cpg)
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = np.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = ((tp * tn - fp * fn) / denominator) if denominator > 0 else 0.0
    jaccard = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "cpg_tp": tp,
        "cpg_fp": fp,
        "cpg_fn": fn,
        "cpg_precision": precision,
        "cpg_recall": recall,
        "cpg_f1": f1,
        "cpg_mcc": mcc,
        "cpg_jaccard": jaccard,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def region_metrics(called, truth, min_overlap: int = 20) -> dict:
    recovered = set()
    matched_calls = 0
    for first, last in called:
        hit = False
        for index, (start, end, *_) in enumerate(truth):
            if min(int(last), end) - max(int(first), start) + 1 >= min_overlap:
                recovered.add(index)
                hit = True
        matched_calls += int(hit)
    n_called = len(called)
    precision = matched_calls / n_called if n_called else 0.0
    recall = len(recovered) / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "regions_called": n_called,
        "regions_recovered": len(recovered),
        "regions_false": n_called - matched_calls,
        "region_precision": precision,
        "region_recall": recall,
        "region_f1": f1,
        "fragmentation": (matched_calls / len(recovered)) if recovered else np.nan,
    }


def stratum_recall(called, truth, min_overlap: int = 20) -> dict:
    out: dict = {}
    for fraction, delta in sorted({(f, d) for _, _, f, d in truth}):
        index = [
            i for i, (_, _, f, d) in enumerate(truth) if f == fraction and d == delta
        ]
        got = 0
        for i in index:
            start, end, *_ = truth[i]
            if any(
                min(int(last), end) - max(int(first), start) + 1 >= min_overlap
                for first, last in called
            ):
                got += 1
        out[f"f={fraction:g},d={delta:g}"] = got / len(index)
    return out


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------

PREVALENCE_GRID = [
    (0.05, 0.35),
    (0.05, 0.55),
    (0.10, 0.30),
    (0.10, 0.50),
    (0.25, 0.25),
    (0.25, 0.40),
    (0.50, 0.20),
    (0.50, 0.35),
]


def make_celltype_cohort(seed: int, nuisance: bool):
    """4 cell types + tumour/benign + a rare 5/120 subgroup."""

    rng = np.random.default_rng(seed)
    n, n_cpg, length = 120, 16000, 40
    celltype = np.repeat(np.arange(4), n // 4)
    tumour = (rng.random(n) < 0.5).astype(float)
    rare = np.zeros(n)
    rare[rng.choice(n, 5, replace=False)] = 1
    matrix = 0.5 + rng.normal(0, 0.04, (n_cpg, n))
    truth = []
    position = 200
    for kind in ["celltype"] * 40 + ["tumour"] * 20 + ["rare"] * 10:
        if kind == "celltype":
            effect = rng.permutation([0.0, 0.18, 0.36, 0.54])[celltype]
            marker = (4.0, 0.18)
        elif kind == "tumour":
            effect = 0.35 * tumour
            marker = (0.5, 0.35)
        else:
            effect = 0.45 * rare
            marker = (0.042, 0.45)
        matrix[position : position + length] += effect - effect.mean()
        truth.append((position, position + length - 1, marker[0], marker[1]))
        position += 200
    if nuisance:
        latent = rng.normal(0, 1, n)
        loading = rng.normal(0, 0.07, (n_cpg, 1))
        matrix = matrix + loading @ latent[None, :]
    matrix = np.clip(matrix, 0.001, 0.999)
    locations = pd.DataFrame(
        {
            "chr": "chr1",
            "start": np.arange(n_cpg) * 2,
            "end": np.arange(n_cpg) * 2 + 1,
        }
    )
    return matrix, locations, truth


def build_scenario(name: str, seed: int):
    if name == "clean":
        return simulate(strata=PREVALENCE_GRID, seed=seed)
    if name == "nuisance":
        return simulate(
            strata=PREVALENCE_GRID, n_latent=3, latent_loading_sd=0.10, seed=seed
        )
    if name == "missingness":
        return simulate(strata=PREVALENCE_GRID, missing_rate=0.25, seed=seed)
    if name == "small_n":
        return simulate(
            strata=[(0.05, 0.25), (0.08, 0.25), (0.10, 0.20), (0.15, 0.20)],
            n_samples=20,
            noise_sd=0.06,
            seed=seed,
        )
    if name == "celltype":
        return make_celltype_cohort(seed, nuisance=False)
    if name == "celltype_nuisance":
        return make_celltype_cohort(seed, nuisance=True)
    raise ValueError(name)


SCENARIOS = [
    "clean",
    "nuisance",
    "missingness",
    "small_n",
    "celltype",
    "celltype_nuisance",
]

THRESHOLDS = [
    0.0005,
    0.001,
    0.002,
    0.004,
    0.008,
    0.012,
    0.020,
    0.030,
    0.050,
    0.080,
    0.120,
]


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def collect(seeds: list[int], thresholds: list[float], scenarios: list[str]):
    rows = []
    strata_rows = []
    for scenario in scenarios:
        for seed in seeds:
            matrix, locations, truth = build_scenario(scenario, seed)
            n_cpg = matrix.shape[0]

            variants = [
                ("original (as written)", dict(fix_length_bug=False, impute=False)),
                ("original (length fixed)", dict(fix_length_bug=True, impute=False)),
                ("original + mean imputation", dict(fix_length_bug=False, impute=True)),
            ]
            for threshold in thresholds:
                for name, options in variants:
                    called = run_original(matrix, threshold, 5, **options)
                    record = {
                        "scenario": scenario,
                        "seed": seed,
                        "method": name,
                        "threshold": threshold,
                        "n_samples": matrix.shape[1],
                    }
                    record.update(cpg_metrics(called, truth, n_cpg))
                    record.update(region_metrics(called, truth))
                    rows.append(record)
                    if name == "original (as written)":
                        strata_rows.append(
                            {
                                "scenario": scenario,
                                "seed": seed,
                                "method": f"original @ {threshold:g}",
                                **stratum_recall(called, truth),
                            }
                        )

            for label, kwargs in [
                ("EIGER 2 correlation", {"statistic": "correlation"}),
                ("EIGER 2 covariance", {"statistic": "covariance"}),
            ]:
                config = Eiger2Config(null_replicates=6, **kwargs)
                regions, _, manifest = call_dmrs(matrix, locations, config=config)
                significant = (
                    regions[regions["significant"]] if len(regions) else regions
                )
                called = [
                    (int(r["cpg_index_start"]), int(r["cpg_index_end"]))
                    for _, r in significant.iterrows()
                ]
                record = {
                    "scenario": scenario,
                    "seed": seed,
                    "method": label,
                    "threshold": manifest.get("null", {}).get("threshold", np.nan),
                    "n_samples": matrix.shape[1],
                    "k_deflated": manifest["latent_structure"]["n_components"],
                    "k_candidates": manifest["latent_structure"].get(
                        "n_candidates", 0
                    ),
                }
                record.update(cpg_metrics(called, truth, n_cpg))
                record.update(region_metrics(called, truth))
                rows.append(record)
                strata_rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": label,
                        **stratum_recall(called, truth),
                    }
                )
            print(f"  done: {scenario} seed={seed}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(strata_rows)


def _fmt(frame: pd.DataFrame) -> str:
    frame = frame.copy()
    if "threshold" in frame.columns:
        frame["threshold"] = frame["threshold"].map(lambda v: f"{v:.4g}")
    return frame.to_markdown(index=False, floatfmt=".3f")


def report(results: pd.DataFrame, strata: pd.DataFrame) -> str:
    out: list[str] = []
    numeric = results.select_dtypes(include=[np.number]).columns

    out.append("# EIGER benchmark: original vs EIGER 2\n")
    out.append(
        f"Seeds per scenario: {results['seed'].nunique()}. "
        f"All values are means across seeds. The original method's covariance "
        f"threshold is swept; EIGER 2 runs once at its automatic operating "
        f"point.\n"
    )

    out.append("\n## Table 1 — original method, covariance threshold sweep")
    out.append("\nCpG-level metrics, `original (as written)`.\n")
    for scenario in results["scenario"].unique():
        variant = (
            "original + mean imputation"
            if scenario == "missingness"
            else "original (as written)"
        )
        subset = results[
            (results["scenario"] == scenario) & (results["method"] == variant)
        ]
        columns = [c for c in numeric if c != "threshold"]
        table = (
            subset.groupby("threshold")[columns]
            .mean()
            .reset_index()[
                [
                    "threshold",
                    "cpg_precision",
                    "cpg_recall",
                    "cpg_f1",
                    "cpg_mcc",
                    "cpg_jaccard",
                    "regions_called",
                    "regions_recovered",
                    "regions_false",
                ]
            ]
        )
        out.append(f"\n**{scenario}** — `{variant}`\n")
        out.append(_fmt(table))

    out.append("\n\n## Table 2 — head to head at the original's oracle-best threshold")
    out.append(
        "\nThe original is scored at the threshold that maximises its own CpG F1 "
        "on each scenario, chosen using the ground truth. EIGER 2 is untuned.\n"
    )
    head = []
    for scenario in results["scenario"].unique():
        subset = results[results["scenario"] == scenario]
        variant = (
            "original + mean imputation"
            if scenario == "missingness"
            else "original (as written)"
        )
        original = (
            subset[subset["method"] == variant]
            .groupby("threshold")[[c for c in numeric if c != "threshold"]]
            .mean()
        )
        best = original["cpg_f1"].idxmax()
        head.append(
            {
                "scenario": scenario,
                "method": f"{variant} @ {best:.4g} (oracle)",
                **{
                    k: original.loc[best, k]
                    for k in (
                        "cpg_precision",
                        "cpg_recall",
                        "cpg_f1",
                        "cpg_mcc",
                        "region_precision",
                        "region_recall",
                        "region_f1",
                    )
                },
            }
        )
        head.append(
            {
                "scenario": scenario,
                "method": f"{variant} @ 0.012 (published default)",
                **{
                    k: original.loc[0.012, k]
                    for k in (
                        "cpg_precision",
                        "cpg_recall",
                        "cpg_f1",
                        "cpg_mcc",
                        "region_precision",
                        "region_recall",
                        "region_f1",
                    )
                },
            }
        )
        for label in ("EIGER 2 correlation", "EIGER 2 covariance"):
            row = subset[subset["method"] == label][list(numeric)].mean()
            head.append(
                {
                    "scenario": scenario,
                    "method": label,
                    **{
                        k: row[k]
                        for k in (
                            "cpg_precision",
                            "cpg_recall",
                            "cpg_f1",
                            "cpg_mcc",
                            "region_precision",
                            "region_recall",
                            "region_f1",
                        )
                    },
                }
            )
    out.append(_fmt(pd.DataFrame(head)))

    out.append("\n\n## Table 3 — region-level detail at the oracle threshold\n")
    detail = []
    for scenario in results["scenario"].unique():
        subset = results[results["scenario"] == scenario]
        variant = (
            "original + mean imputation"
            if scenario == "missingness"
            else "original (as written)"
        )
        original = (
            subset[subset["method"] == variant]
            .groupby("threshold")[[c for c in numeric if c != "threshold"]]
            .mean()
        )
        best = original["cpg_f1"].idxmax()
        detail.append(
            {
                "scenario": scenario,
                "method": f"{variant} @ {best:.4g}",
                **{
                    k: original.loc[best, k]
                    for k in (
                        "regions_called",
                        "regions_recovered",
                        "regions_false",
                        "fragmentation",
                        "fpr",
                    )
                },
            }
        )
        for label in ("EIGER 2 correlation", "EIGER 2 covariance"):
            row = subset[subset["method"] == label][list(numeric)].mean()
            detail.append(
                {
                    "scenario": scenario,
                    "method": label,
                    **{
                        k: row[k]
                        for k in (
                            "regions_called",
                            "regions_recovered",
                            "regions_false",
                            "fragmentation",
                            "fpr",
                        )
                    },
                }
            )
    out.append(_fmt(pd.DataFrame(detail)))

    out.append("\n\n## Table 4 — recall by subgroup prevalence and effect size\n")
    for scenario in strata["scenario"].unique():
        subset = strata[strata["scenario"] == scenario]
        keep = [
            m
            for m in subset["method"].unique()
            if m.startswith("EIGER")
            or m in ("original @ 0.012", "original @ 0.004", "original @ 0.002")
        ]
        table = (
            subset[subset["method"].isin(keep)]
            .drop(columns=["seed"])
            .groupby("method")
            .mean(numeric_only=True)
            .dropna(axis=1, how="all")
            .reset_index()
        )
        out.append(f"\n**{scenario}**\n")
        out.append(_fmt(table))

    out.append("\n\n## Table 5 — effect of the length-threshold bug\n")
    bug = (
        results[results["method"].str.startswith("original")]
        .groupby(["scenario", "method"])[list(numeric)]
        .mean()
        .reset_index()[
            ["scenario", "method", "cpg_recall", "cpg_precision", "cpg_f1"]
        ]
    )
    out.append(_fmt(bug))
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    seeds = [4, 5] if args.quick else [4, 5, 6]
    thresholds = sorted(set(THRESHOLDS[::2] + [0.012])) if args.quick else THRESHOLDS
    scenarios = SCENARIOS[:3] if args.quick else SCENARIOS

    print(f"scenarios={scenarios} seeds={seeds} thresholds={len(thresholds)}")
    results, strata = collect(seeds, thresholds, scenarios)

    root = Path(args.out) if args.out else Path(__file__).resolve().parents[2]
    results.to_csv(root / "benchmark_results.csv", index=False)
    text = report(results, strata)
    (root / "BENCHMARK.md").write_text(text)
    print(f"\nwrote {root/'BENCHMARK.md'} and {root/'benchmark_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
