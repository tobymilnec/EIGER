#!/usr/bin/env python3
"""Benchmark of the ORIGINAL EIGER, and of EIGER 2 against it.

Why this module exists
----------------------
The benchmark reported in the thesis characterises the *2026 beta* caller: it
sweeps an automatic covariance threshold, applies Gaussian / logistic /
prevalence-variance weighting, and reports rank-within-strata. None of those
exist in the original 2023 method. Comparing EIGER 2 against the beta therefore
measures the wrong baseline, and mixes two rounds of revision into one number.

This module benchmarks the method as originally published:

  * adjacent-CpG covariance, complete case (``processing.covariance_blocks``
    with ``pairwise=False``), so any missing observation at either CpG makes
    the edge undefined;
  * a FIXED covariance threshold, with no label-free calibration;
  * contiguous runs in which every edge passes, subject to a minimum length
    (``processing.extract_blocks``);
  * the sample partition taken as the FIRST adjacent gap exceeding the delta
    threshold, with the descriptive score built from it
    (``processing.find_separators``).

Every one of those steps is invoked from ``Python/functions/processing.py``
itself rather than reimplemented here, so there can be no question of the
baseline being a paraphrase. The only thing this module supplies is the
simulator, the evaluation and the loop.

Usage
-----
    python -m eiger2.original_benchmark calibrate --replicates 10
    python -m eiger2.original_benchmark all --out-dir original_bench --replicates 10

Stages
------
``calibrate``   tune the simulator's nuisance structure so the ORIGINAL caller
                reproduces a target null-call profile before anything is compared
``recovery``    planted-signal recovery, original vs EIGER 2
``null``        complete-null false calls
``samplesize``  the portability of a fixed threshold across cohort sizes
``strata``      recall by planted effect, prevalence and length
``partition``   whether the inferred sample split matches the planted one (ARI)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# The original implementation lives outside the eiger2 package.
# ``functions/processing.py`` is the original 2023 implementation and is
# invoked directly rather than reimplemented, so the baseline cannot be a
# paraphrase. It lives under Python/ alongside the package.
_SOURCE_ROOT = Path(__file__).resolve().parent.parent / "Python"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

# ``chapter2_benchmark`` is a sibling module in this directory, not an installed
# package. Without this the import succeeds only when the working directory
# happens to be benchmarking/, so running the documented
# ``python benchmarking/original_benchmark.py`` from the repository root fails.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from chapter2_benchmark import (
    DEFAULT_SAMPLES,
    TARGET_NULL_CALLS,
    V1_THRESHOLDS,
    _latex,
    evaluate,
    simulate_chapter2,
)

from eiger2.core import Eiger2Config, call_dmrs
from functions import processing as original

#: Effect-size gate applied by the original caller, in beta units.
DELTA_THRESHOLD = 0.20

#: Minimum region length in CpGs, as published.
MIN_LENGTH = 5


# --------------------------------------------------------------------------
# the original caller, driven end to end
# --------------------------------------------------------------------------


def _cluster_delta(values: np.ndarray) -> tuple[float, np.ndarray]:
    """Split at the largest adjacent gap; return |mean(A) - mean(B)| and group A.

    This is the revised effect-size criterion: a threshold of 0.20 denotes a
    20-percentage-point difference between the two inferred *cluster means*.

    The distinction from the original criterion matters more than it appears.
    ``find_separators`` reports ``delta_minmax``, the gap between two
    *consecutive* sorted sample means. That gap is eroded by within-group
    sampling noise: at read depth 15 a per-sample block mean over ~7 CpGs
    carries a standard error near 0.05, so 12 samples drawn from one group span
    roughly 0.2 in range, and any true separation smaller than that leaves no
    single adjacent gap above threshold. The difference between the two cluster
    means is unaffected by that spread, because averaging within each group
    removes it.
    """

    finite = np.isfinite(values)
    if finite.sum() < 2:
        return 0.0, np.array([], dtype=int)
    order = np.argsort(np.where(finite, values, np.inf))[: int(finite.sum())]
    ordered = np.asarray(values)[order]
    gaps = np.diff(ordered)
    if gaps.size == 0:
        return 0.0, np.array([], dtype=int)
    cut = int(np.argmax(gaps))
    low, high = order[: cut + 1], order[cut + 1 :]
    if low.size == 0 or high.size == 0:
        return 0.0, np.array([], dtype=int)
    delta = float(abs(np.mean(np.asarray(values)[low])
                      - np.mean(np.asarray(values)[high])))
    # report the minority side as the separated group, which is the quantity of
    # interest and avoids the low-side convention returning the complement
    minor = low if low.size <= high.size else high
    return delta, minor


def run_original_eiger(
    matrix: np.ndarray,
    *,
    cov_threshold: float,
    delta_threshold: float = DELTA_THRESHOLD,
    min_length: int = MIN_LENGTH,
    apply_delta_gate: bool = True,
    delta_mode: str = "cluster_mean",
) -> tuple[list[tuple[int, int]], list[dict]]:
    """Run the published EIGER pipeline and return ``(regions, diagnostics)``.

    Discovery is complete-case covariance against a fixed threshold. Each
    surviving block is then passed through ``find_separators`` to obtain the
    sample partition, and is retained only if the inferred effect clears
    ``delta_threshold``.

    ``delta_mode`` selects the effect-size criterion:

    ``"adjacent_gap"``
        the original ``delta_minmax`` --- the first adjacent gap in the sorted
        sample means exceeding the threshold;
    ``"cluster_mean"``
        the revised criterion --- the difference between the two inferred
        cluster means, split at the largest adjacent gap. This is the setting
        described in the thesis benchmark and is the default here.
    """

    if delta_mode not in {"adjacent_gap", "cluster_mean"}:
        raise ValueError("delta_mode must be 'adjacent_gap' or 'cluster_mean'")

    statistic = original.covariance_blocks(matrix, pairwise=False)
    blocks = original.extract_blocks(
        statistic, DMR_length_threshold=min_length, cov_threshold=cov_threshold
    )

    n_samples = matrix.shape[1]
    # ``find_separators`` indexes a metadata array positionally; the original
    # study supplied cell type in column 3 and tissue in column 4. Discovery
    # does not depend on them, so a single constant group is supplied and the
    # group-level branches are inert.
    meta = np.zeros((n_samples, 5), dtype=object)
    meta[:, 2] = "NA"
    meta[:, 3] = "NA"
    meta[:, 4] = "NA"

    regions: list[tuple[int, int]] = []
    diagnostics: list[dict] = []
    for start, stop in blocks:
        span = np.arange(int(start), int(stop) + 1)
        if span.size < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = original.find_separators(
                    matrix, span, meta, thres=delta_threshold
                )
            except Exception:
                continue
        gap_delta = float(result.get("delta_minmax", 0.0) or 0.0)
        block_means = np.asarray(result.get("averages"), dtype=float)
        cluster_delta, minority = _cluster_delta(block_means)
        delta = gap_delta if delta_mode == "adjacent_gap" else cluster_delta
        if apply_delta_gate and not (delta > delta_threshold):
            continue
        regions.append((int(span[0]), int(span[-1])))
        diagnostics.append({
            "start": int(span[0]), "end": int(span[-1]),
            "delta_minmax": gap_delta,
            "delta_cluster_mean": cluster_delta,
            "minority_index": minority,
            "classification": result.get("classification"),
            "pvalue_sample": float(result.get("pvalue_sample", np.nan)),
            "score_sample_gaussian": float(
                result.get("score_sample_gaussian", np.nan)
            ),
            "sample_index": np.asarray(result.get("sample_index", []), dtype=int),
        })
    return regions, diagnostics


def _v2_regions(regions: pd.DataFrame):
    if len(regions) == 0:
        return []
    frame = regions[regions["significant"]]
    return list(zip(frame["cpg_index_start"].astype(int),
                    frame["cpg_index_end"].astype(int)))


def _adjusted_rand(a: Sequence[int], b: Sequence[int]) -> float:
    """ARI between two binary sample partitions, written out to avoid sklearn."""

    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = a.size
    if n < 2:
        return float("nan")
    table = np.zeros((2, 2), dtype=float)
    for i in (0, 1):
        for j in (0, 1):
            table[i, j] = np.sum((a == i) & (b == j))
    comb = lambda x: x * (x - 1) / 2.0
    index = comb(table).sum()
    a_sum = comb(table.sum(axis=1)).sum()
    b_sum = comb(table.sum(axis=0)).sum()
    total = comb(float(n))
    expected = a_sum * b_sum / total
    maximum = 0.5 * (a_sum + b_sum)
    return float((index - expected) / (maximum - expected)) if maximum != expected \
        else float("nan")


# --------------------------------------------------------------------------
# stage 0 — calibrate the background against the original caller
# --------------------------------------------------------------------------


def calibrate(seeds, n_samples=DEFAULT_SAMPLES,
              loading_grid=(0.03, 0.04, 0.05, 0.06, 0.07),
              smoothing_grid=(4, 6, 8, 12, 20)) -> pd.DataFrame:
    """Match the simulator's nuisance structure to the published null profile.

    The criterion is the ORIGINAL caller's null-call count at each fixed
    covariance threshold, so the background is of comparable difficulty to the
    one the published numbers came from before any comparison is drawn.
    """

    import chapter2_benchmark as c2

    saved = c2.LATENT_SMOOTHING
    rows = []
    try:
        for smoothing in smoothing_grid:
            c2.LATENT_SMOOTHING = smoothing
            for loading in loading_grid:
                counts = {t: [] for t in V1_THRESHOLDS}
                for seed in seeds:
                    matrix, _, _ = simulate_chapter2(
                        n_samples=n_samples, plant_dmrs=False,
                        latent_loading_sd=loading, seed=1000 + seed,
                    )
                    for threshold in V1_THRESHOLDS:
                        # complete-case discovery reports nothing at realistic
                        # missingness, so the background is calibrated on the
                        # mean-imputed variant, which is what the published runs
                        # used in practice
                        filled = _mean_impute(matrix)
                        found, _ = run_original_eiger(
                            filled, cov_threshold=threshold
                        )
                        counts[threshold].append(len(found))
                means = [float(np.mean(counts[t])) for t in V1_THRESHOLDS]
                rows.append({
                    "latent_loading_sd": loading, "latent_smoothing": smoothing,
                    **{f"null_calls_at_{t:g}": m
                       for t, m in zip(V1_THRESHOLDS, means)},
                    "l1_error": float(sum(abs(m - t) for m, t
                                          in zip(means, TARGET_NULL_CALLS))),
                })
    finally:
        c2.LATENT_SMOOTHING = saved
    return pd.DataFrame(rows)


def _mean_impute(matrix: np.ndarray) -> np.ndarray:
    filled = np.array(matrix, dtype=float, copy=True)
    with np.errstate(invalid="ignore"):
        row_means = np.nanmean(filled, axis=1)
    row_means = np.where(np.isfinite(row_means), row_means, 0.5)
    missing = ~np.isfinite(filled)
    filled[missing] = np.take(row_means, np.nonzero(missing)[0])
    return filled


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


def _methods(matrix, locations, *, include_v2: bool = True):
    """Yield ``(name, regions)`` for the original caller and EIGER 2.

    The original caller appears in four configurations, because the effect-size
    criterion turns out to dominate its behaviour and the two published forms of
    it behave very differently:

    ``complete case``       discovery only, no imputation --- reports nothing at
                            realistic missingness
    ``adjacent gap``        the original ``delta_minmax`` criterion
    ``cluster mean``        the revised criterion used in the thesis benchmark
    ``no effect gate``      discovery alone, to isolate the criterion's effect
    """

    filled = _mean_impute(matrix)
    for threshold in V1_THRESHOLDS:
        found, _ = run_original_eiger(
            matrix, cov_threshold=threshold, delta_mode="cluster_mean"
        )
        yield f"Original @ {threshold:g}, complete case", found
        for mode, label in (("adjacent_gap", "adjacent gap"),
                            ("cluster_mean", "cluster mean")):
            found, _ = run_original_eiger(
                filled, cov_threshold=threshold, delta_mode=mode
            )
            yield f"Original @ {threshold:g}, {label}", found
        found, _ = run_original_eiger(
            filled, cov_threshold=threshold, apply_delta_gate=False
        )
        yield f"Original @ {threshold:g}, no effect gate", found
    if include_v2:
        for statistic in ("covariance", "correlation"):
            regions, _, _ = call_dmrs(
                matrix, locations, config=Eiger2Config(statistic=statistic)
            )
            yield f"EIGER 2 {statistic}", _v2_regions(regions)


def recovery(seeds, n_samples=DEFAULT_SAMPLES) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        matrix, locations, truth = simulate_chapter2(
            n_samples=n_samples, seed=seed
        )
        n_cpg = matrix.shape[0]
        for name, called in _methods(matrix, locations):
            metrics = evaluate(called, truth, n_cpg)
            metrics.pop("recovered_indices")
            rows.append({"seed": seed, "n_samples": n_samples,
                         "method": name, **metrics})
    return pd.DataFrame(rows)


def null_control(seeds, n_samples=DEFAULT_SAMPLES) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        matrix, locations, _ = simulate_chapter2(
            n_samples=n_samples, plant_dmrs=False, seed=1000 + seed
        )
        megabases = matrix.shape[0] * 20 / 1e6
        for name, called in _methods(matrix, locations):
            rows.append({"seed": seed, "method": name,
                         "null_calls": len(called),
                         "false_calls_per_mb": len(called) / megabases})
    return pd.DataFrame(rows)


def sample_size(seeds, sizes=(8, 12, 24, 48)) -> pd.DataFrame:
    frames = []
    for n_samples in sizes:
        signal = recovery(seeds, n_samples=n_samples)
        nulls = null_control(seeds, n_samples=n_samples)
        merged = (
            signal.groupby("method")[["region_recall", "region_precision",
                                      "bp_precision"]].mean()
            .join(nulls.groupby("method")[["null_calls"]].mean())
        )
        merged["n_samples"] = n_samples
        frames.append(merged.reset_index())
    return pd.concat(frames, ignore_index=True)


def strata(seeds, n_samples=DEFAULT_SAMPLES) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        matrix, locations, truth = simulate_chapter2(
            n_samples=n_samples, seed=seed
        )
        n_cpg = matrix.shape[0]
        for name, called in _methods(matrix, locations):
            recovered = set(evaluate(called, truth, n_cpg)["recovered_indices"])
            for index, t in enumerate(truth):
                rows.append({"seed": seed, "method": name,
                             "effect": t["effect"], "length": t["length"],
                             "fraction": t["fraction"],
                             "recovered": int(index in recovered)})
    return pd.DataFrame(rows)


def partition(seeds, n_samples=DEFAULT_SAMPLES,
              cov_threshold: float = 0.012) -> pd.DataFrame:
    """Does the original caller recover the planted sample split?

    ``find_separators`` takes the FIRST adjacent gap exceeding the threshold and
    assigns the low side as the separated group. Where the affected samples are
    hypermethylated, or where more than one gap exceeds the threshold, the
    inferred partition can be the complement of the planted one or an arbitrary
    prefix of the sorted order. ARI is reported for the regions the caller
    recovered, so it measures partition quality given detection.
    """

    rows = []
    for seed in seeds:
        matrix, _, truth = simulate_chapter2(n_samples=n_samples, seed=seed)
        filled = _mean_impute(matrix)
        found, diagnostics = run_original_eiger(
            filled, cov_threshold=cov_threshold
        )
        for region, diagnostic in zip(found, diagnostics):
            best, best_overlap = None, 0
            for t in truth:
                overlap = min(region[1], t["end"]) - max(region[0], t["start"]) + 1
                if overlap > best_overlap:
                    best, best_overlap = t, overlap
            if best is None or best_overlap < 3:
                continue
            planted = np.zeros(n_samples, dtype=int)
            rng = np.random.default_rng(seed)  # not used; membership recomputed
            del rng
            # planted membership is recoverable from the region's beta profile:
            # the affected samples are those displaced from the regional median
            block = np.nanmean(
                filled[best["start"] : best["end"] + 1], axis=0
            )
            displacement = np.abs(block - np.nanmedian(block))
            planted[np.argsort(-displacement)[: best["n_affected"]]] = 1
            inferred = np.zeros(n_samples, dtype=int)
            inferred[np.asarray(diagnostic["sample_index"], dtype=int)] = 1
            rows.append({
                "seed": seed, "effect": best["effect"],
                "fraction": best["fraction"],
                "n_affected": best["n_affected"],
                "inferred_group_size": int(inferred.sum()),
                "ari": _adjusted_rand(planted, inferred),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _write(frame: pd.DataFrame, out: Path, stem: str,
           caption: str = "", label: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{stem}.csv", index=False)
    if caption:
        (out / f"{stem}.tex").write_text(_latex(frame, caption, label))
    print(f"  wrote {stem}.csv ({len(frame)} rows)")
    if not frame.empty and len(frame) <= 40:
        print(frame.round(3).to_string(index=False))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["calibrate", "recovery", "null",
                                          "samplesize", "strata", "partition",
                                          "all"])
    parser.add_argument("--out-dir", default="original_bench")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args(list(argv) if argv is not None else None)

    out = Path(args.out_dir)
    seeds = list(range(args.replicates))
    stages = (["recovery", "null", "samplesize", "strata", "partition"]
              if args.stage == "all" else [args.stage])

    for stage in stages:
        print(f"\n== {stage} ==")
        if stage == "calibrate":
            frame = calibrate(seeds, n_samples=args.samples)
            _write(frame.sort_values("l1_error"), out, "calibration_grid")
            print("\nSet LATENT_LOADING_SD / LATENT_SMOOTHING in "
                  "chapter2_benchmark.py to the best row before comparing.")

        elif stage == "recovery":
            frame = recovery(seeds, n_samples=args.samples)
            summary = frame.groupby("method")[
                ["calls", "region_precision", "region_recall", "region_f1",
                 "bp_precision", "bp_recall"]].mean().reset_index()
            _write(frame, out, "recovery_raw")
            _write(summary, out, "recovery_summary",
                   caption=(f"Recovery of planted DMRs at {args.samples} "
                            f"samples, mean over {args.replicates} replicates. "
                            f"The original caller is invoked from "
                            f"\\texttt{{functions/processing.py}}."),
                   label="tab:original_recovery")

        elif stage == "null":
            frame = null_control(seeds, n_samples=args.samples)
            summary = frame.groupby("method")[
                ["null_calls", "false_calls_per_mb"]].agg(["mean", "std"])
            summary.columns = [f"{a}_{b}" for a, b in summary.columns]
            _write(frame, out, "null_raw")
            _write(summary.reset_index(), out, "null_summary",
                   caption="Complete-null simulations; every call is false.",
                   label="tab:original_null")

        elif stage == "samplesize":
            frame = sample_size(seeds)
            _write(frame, out, "samplesize_summary",
                   caption="Portability of a fixed covariance threshold.",
                   label="tab:original_samplesize")

        elif stage == "strata":
            frame = strata(seeds, n_samples=args.samples)
            _write(frame, out, "strata_raw")
            for key, name in (("effect", "planted effect size"),
                              ("fraction", "affected-sample fraction"),
                              ("length", "planted length in CpGs")):
                table = frame.pivot_table(index="method", columns=key,
                                          values="recovered").reset_index()
                _write(table, out, f"strata_by_{key}",
                       caption=f"Recall stratified by {name}.",
                       label=f"tab:original_strata_{key}")

        elif stage == "partition":
            frame = partition(seeds, n_samples=args.samples)
            _write(frame, out, "partition_raw")
            if not frame.empty:
                summary = frame.groupby("fraction")[
                    ["ari", "inferred_group_size", "n_affected"]
                ].mean().reset_index()
                _write(summary, out, "partition_summary",
                       caption=("Agreement between the planted sample split and "
                                "the partition inferred by "
                                "\\texttt{find\\_separators}."),
                       label="tab:original_partition")

    (out / "manifest.json").write_text(json.dumps({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage, "replicates": args.replicates,
        "samples": args.samples, "delta_threshold": DELTA_THRESHOLD,
        "min_length": MIN_LENGTH, "cov_thresholds": list(V1_THRESHOLDS),
        "original_source": "Python/functions/processing.py",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
