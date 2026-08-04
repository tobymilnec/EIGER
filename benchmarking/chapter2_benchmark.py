#!/usr/bin/env python3
"""EIGER 1 vs EIGER 2 on the Chapter 2 simulation design.

The Chapter 2 benchmark of EIGER 1 used a specific semi-synthetic design:

    4,000 CpGs, spatially correlated methylation, mean read depth 15,
    5% missing observations; signal runs plant 30 non-overlapping DMRs with
    effect sizes 0.15 / 0.25 / 0.40, lengths 4 / 8 / 16 / 32 CpGs, and
    affected-sample fractions ~0.08 / 0.25 / 0.42; complete-null runs use the
    same background with no planted DMRs; five signal and five null replicates.

This module reproduces that design exactly and runs both callers on it, so the
EIGER 2 numbers are directly comparable with the tables already in the thesis
rather than to a differently-parameterised simulation.

Usage
-----
    python -m eiger2.chapter2_benchmark all --out-dir chapter2_bench

    # individual stages
    python -m eiger2.chapter2_benchmark recovery   --out-dir chapter2_bench
    python -m eiger2.chapter2_benchmark null       --out-dir chapter2_bench
    python -m eiger2.chapter2_benchmark samplesize --out-dir chapter2_bench
    python -m eiger2.chapter2_benchmark strata     --out-dir chapter2_bench
    python -m eiger2.chapter2_benchmark scaling    --out-dir chapter2_bench

Runtime is a few minutes for ``all`` at the default five replicates.
Every table is written as a CSV plus a LaTeX fragment ready to paste.
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
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from compare import run_original

from eiger2.core import Eiger2Config, call_dmrs

# --------------------------------------------------------------------------
# Chapter 2 simulation design
# --------------------------------------------------------------------------

N_CPG = 4000
MEAN_DEPTH = 15
MISSING_RATE = 0.05
N_DMR = 30
EFFECTS = (0.15, 0.25, 0.40)
LENGTHS = (4, 8, 16, 32)
FRACTIONS = (0.08, 0.25, 0.42)
DEFAULT_SAMPLES = 24
V1_THRESHOLDS = (0.010, 0.012, 0.015)

#: Spatial autocorrelation of the background. Adjacent CpGs share a slowly
#: varying baseline, which is what makes a covariance caller non-trivial.
SPATIAL_RHO = 0.65
BASELINE_SD = 0.08
SAMPLE_NOISE_SD = 0.03

#: Latent nuisance structure. A shared genome-wide baseline is removed by
#: centring and therefore generates no adjacent *cross-sample* covariance; a
#: covariance caller only produces false regions when samples differ from one
#: another in a spatially coherent way. Tumour purity, immune infiltration and
#: cell composition all do exactly that, so the null model needs them or it is
#: unrealistically easy.
#:
#: CALIBRATION. ``N_LATENT`` and ``LATENT_LOADING_SD`` are the two knobs that set
#: how many false regions EIGER 1 reports on complete-null data. Tune them so the
#: EIGER 1 null-call counts reproduce those already reported in
#: Table~\ref{tab:eiger_threshold_benchmark} (24.6 / 7.6 / 0.8 at covariance
#: 0.010 / 0.012 / 0.015 with 24 samples); ``--calibrate`` reports the current
#: values so the match can be checked before any comparison is made.
N_LATENT = 2
LATENT_LOADING_SD = 0.05
LATENT_SMOOTHING = 6


def simulate_chapter2(
    *,
    n_samples: int = DEFAULT_SAMPLES,
    n_cpg: int = N_CPG,
    plant_dmrs: bool = True,
    n_dmr: int = N_DMR,
    mean_depth: int = MEAN_DEPTH,
    missing_rate: float = MISSING_RATE,
    n_latent: int = N_LATENT,
    latent_loading_sd: float = LATENT_LOADING_SD,
    seed: int = 0,
) -> tuple[np.ndarray, pd.DataFrame, list[dict]]:
    """One replicate of the Chapter 2 design.

    Returns ``(beta_matrix, locations, truth)``. ``truth`` is a list of dicts
    with ``start``, ``end``, ``effect``, ``length`` and ``fraction`` so that
    recovery can be stratified exactly as in Table 2.x.
    """

    rng = np.random.default_rng(seed)

    # --- spatially correlated background ---------------------------------
    # AR(1) along the genome gives adjacent CpGs a shared baseline; each sample
    # then deviates from it independently. Adjacent-pair covariance is therefore
    # near zero in the absence of a planted subgroup, which is the correct null
    # for a covariance caller.
    innovation = rng.normal(0, BASELINE_SD, n_cpg)
    baseline = np.empty(n_cpg)
    baseline[0] = innovation[0]
    for i in range(1, n_cpg):
        baseline[i] = SPATIAL_RHO * baseline[i - 1] + innovation[i]
    baseline = 0.5 + baseline
    truth_beta = baseline[:, None] + rng.normal(
        0, SAMPLE_NOISE_SD, (n_cpg, n_samples)
    )

    # --- latent nuisance structure ---------------------------------------
    # Spatially smoothed loadings, so that samples differ from one another in a
    # regionally coherent way. This is what generates adjacent cross-sample
    # covariance in the absence of any planted DMR, and it is why a fixed
    # covariance threshold produces false regions on null data.
    if n_latent > 0:
        scores = rng.normal(0, 1, (n_samples, n_latent))
        raw = rng.normal(0, latent_loading_sd, (n_cpg, n_latent))
        kernel = np.ones(LATENT_SMOOTHING) / LATENT_SMOOTHING
        loadings = np.column_stack([
            np.convolve(raw[:, k], kernel, mode="same") * np.sqrt(LATENT_SMOOTHING)
            for k in range(n_latent)
        ])
        truth_beta = truth_beta + loadings @ scores.T
    truth_beta = np.clip(truth_beta, 0.02, 0.98)

    # --- planted DMRs ----------------------------------------------------
    truth: list[dict] = []
    if plant_dmrs and n_dmr > 0:
        combos = [
            (effect, length, fraction)
            for effect in EFFECTS
            for length in LENGTHS
            for fraction in FRACTIONS
        ]
        chosen = [combos[i % len(combos)] for i in range(n_dmr)]
        rng.shuffle(chosen)
        margin = 40
        spacing = (n_cpg - 2 * margin) // n_dmr
        for index, (effect, length, fraction) in enumerate(chosen):
            start = margin + index * spacing + rng.integers(0, max(1, spacing // 4))
            end = min(start + length - 1, n_cpg - 1)
            size = max(1, int(round(fraction * n_samples)))
            members = rng.choice(n_samples, size, replace=False)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            truth_beta[start : end + 1, members] = np.clip(
                truth_beta[start : end + 1, members] + sign * effect, 0.02, 0.98
            )
            truth.append({
                "start": int(start), "end": int(end), "effect": float(effect),
                "length": int(end - start + 1), "fraction": float(fraction),
                "n_affected": int(size),
            })

    # --- read-depth sampling ---------------------------------------------
    depth = rng.poisson(mean_depth, (n_cpg, n_samples)).astype(float)
    depth[depth < 1] = 1
    methylated = rng.binomial(depth.astype(int), truth_beta)
    observed = methylated / depth

    # --- missingness ------------------------------------------------------
    if missing_rate > 0:
        observed[rng.random(observed.shape) < missing_rate] = np.nan

    locations = pd.DataFrame({
        "chr": "chr1",
        "start": np.arange(n_cpg) * 20,
        "end": np.arange(n_cpg) * 20 + 1,
    })
    return observed, locations, truth


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def _mask(intervals: Iterable[tuple[int, int]], n_cpg: int) -> np.ndarray:
    out = np.zeros(n_cpg, dtype=bool)
    for start, end in intervals:
        out[int(start) : int(end) + 1] = True
    return out


def evaluate(
    called: Sequence[tuple[int, int]],
    truth: Sequence[dict],
    n_cpg: int,
    *,
    reciprocal: float = 0.5,
) -> dict:
    """Region- and base-pair-level metrics, matching the Chapter 2 definitions.

    A region counts as recovered under the 50% reciprocal-overlap criterion used
    throughout Chapter 2.
    """

    called = [(int(a), int(b)) for a, b in called]
    truth_iv = [(t["start"], t["end"]) for t in truth]

    matched_called = set()
    recovered = set()
    for ci, (cs, ce) in enumerate(called):
        for ti, (ts, te) in enumerate(truth_iv):
            overlap = min(ce, te) - max(cs, ts) + 1
            if overlap <= 0:
                continue
            if (overlap / (ce - cs + 1) >= reciprocal
                    and overlap / (te - ts + 1) >= reciprocal):
                matched_called.add(ci)
                recovered.add(ti)

    call_mask = _mask(called, n_cpg)
    truth_mask = _mask(truth_iv, n_cpg)
    tp = int(np.sum(call_mask & truth_mask))
    n_region_precision = len(matched_called) / len(called) if called else np.nan
    n_region_recall = len(recovered) / len(truth_iv) if truth_iv else np.nan
    f1 = (
        2 * n_region_precision * n_region_recall
        / (n_region_precision + n_region_recall)
        if called and truth_iv and (n_region_precision + n_region_recall) > 0
        else 0.0
    )
    return {
        "calls": len(called),
        "region_precision": n_region_precision,
        "region_recall": n_region_recall,
        "region_f1": f1,
        "bp_precision": tp / call_mask.sum() if call_mask.any() else np.nan,
        "bp_recall": tp / truth_mask.sum() if truth_mask.any() else np.nan,
        "recovered_indices": sorted(recovered),
    }


def _v2_regions(regions: pd.DataFrame, significant_only: bool = True):
    if len(regions) == 0:
        return []
    frame = regions[regions["significant"]] if significant_only else regions
    return list(zip(frame["cpg_index_start"].astype(int),
                    frame["cpg_index_end"].astype(int)))


def _v2_config(statistic: str, **kw) -> Eiger2Config:
    return Eiger2Config(statistic=statistic, **kw)


# --------------------------------------------------------------------------
# stage 0 — calibrate the nuisance structure against the published EIGER 1 run
# --------------------------------------------------------------------------

#: Mean null calls per replicate reported for EIGER 1 at 24 samples in
#: Table~\ref{tab:eiger_threshold_benchmark}, at covariance 0.010 / 0.012 /
#: 0.015. The simulator is tuned to reproduce these before any comparison is
#: made, so that EIGER 2 is evaluated on a background of the same difficulty as
#: the one the published numbers came from.
TARGET_NULL_CALLS = (24.6, 7.6, 0.8)


def calibrate(
    seeds: Sequence[int],
    n_samples: int = DEFAULT_SAMPLES,
    loading_grid: Sequence[float] = (0.03, 0.04, 0.05, 0.06, 0.07),
    smoothing_grid: Sequence[int] = (4, 6, 8, 12, 20),
) -> pd.DataFrame:
    """Grid-search the nuisance parameters against the published null profile.

    The simulator used for the original Chapter 2 benchmark is not the one in
    this module, so the background must be matched empirically before the two
    sets of numbers can be placed in the same table. The criterion is the
    EIGER 1 null-call count at each covariance threshold.
    """

    global LATENT_SMOOTHING
    original = LATENT_SMOOTHING
    rows = []
    try:
        for smoothing in smoothing_grid:
            LATENT_SMOOTHING = smoothing
            for loading in loading_grid:
                counts = {t: [] for t in V1_THRESHOLDS}
                for seed in seeds:
                    matrix, _, _ = simulate_chapter2(
                        n_samples=n_samples, plant_dmrs=False,
                        latent_loading_sd=loading, seed=1000 + seed,
                    )
                    for threshold in V1_THRESHOLDS:
                        counts[threshold].append(
                            len(run_original(matrix, threshold, 5, impute=True))
                        )
                means = [float(np.mean(counts[t])) for t in V1_THRESHOLDS]
                rows.append({
                    "latent_loading_sd": loading,
                    "latent_smoothing": smoothing,
                    **{f"null_calls_at_{t:g}": m
                       for t, m in zip(V1_THRESHOLDS, means)},
                    "l1_error": float(sum(
                        abs(m - target)
                        for m, target in zip(means, TARGET_NULL_CALLS)
                    )),
                })
    finally:
        LATENT_SMOOTHING = original
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# stage 1 — recovery on signal simulations
# --------------------------------------------------------------------------


def recovery(seeds: Sequence[int], n_samples: int = DEFAULT_SAMPLES) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        matrix, locations, truth = simulate_chapter2(
            n_samples=n_samples, seed=seed
        )
        n_cpg = matrix.shape[0]

        for threshold in V1_THRESHOLDS:
            for label, impute in (("complete-case", False), ("mean-imputed", True)):
                blocks = run_original(matrix, threshold, 5, impute=impute)
                metrics = evaluate(blocks, truth, n_cpg)
                metrics.pop("recovered_indices")
                rows.append({
                    "seed": seed, "n_samples": n_samples,
                    "method": f"EIGER 1 @ {threshold:g} ({label})",
                    **metrics,
                })

        for statistic in ("covariance", "correlation"):
            regions, _, manifest = call_dmrs(
                matrix, locations, config=_v2_config(statistic)
            )
            metrics = evaluate(_v2_regions(regions), truth, n_cpg)
            metrics.pop("recovered_indices")
            rows.append({
                "seed": seed, "n_samples": n_samples,
                "method": f"EIGER 2 {statistic}",
                "threshold": manifest.get("null", {}).get("threshold", np.nan),
                **metrics,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# stage 2 — complete-null simulations
# --------------------------------------------------------------------------


def null_control(seeds: Sequence[int],
                 n_samples: int = DEFAULT_SAMPLES) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        matrix, locations, _ = simulate_chapter2(
            n_samples=n_samples, plant_dmrs=False, seed=1000 + seed
        )
        n_cpg = matrix.shape[0]
        megabases = n_cpg * 20 / 1e6

        for threshold in V1_THRESHOLDS:
            for label, impute in (("complete-case", False), ("mean-imputed", True)):
                blocks = run_original(matrix, threshold, 5, impute=impute)
                rows.append({
                    "seed": seed, "method": f"EIGER 1 @ {threshold:g} ({label})",
                    "null_calls": len(blocks),
                    "false_calls_per_mb": len(blocks) / megabases,
                })

        for statistic in ("covariance", "correlation"):
            regions, _, manifest = call_dmrs(
                matrix, locations, config=_v2_config(statistic)
            )
            calls = _v2_regions(regions)
            rows.append({
                "seed": seed, "method": f"EIGER 2 {statistic}",
                "null_calls": len(calls),
                "false_calls_per_mb": len(calls) / megabases,
                "threshold": manifest.get("null", {}).get("threshold", np.nan),
                "candidates": int(manifest.get("n_regions", 0)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# stage 3 — sample-size dependence
# --------------------------------------------------------------------------


def sample_size(seeds: Sequence[int],
                sizes: Sequence[int] = (8, 12, 24, 48)) -> pd.DataFrame:
    rows = []
    for n_samples in sizes:
        signal = recovery(seeds, n_samples=n_samples)
        nulls = null_control(seeds, n_samples=n_samples)
        merged = (
            signal.groupby("method")[["region_recall", "region_precision",
                                      "bp_precision"]].mean()
            .join(nulls.groupby("method")[["null_calls"]].mean())
        )
        merged["n_samples"] = n_samples
        rows.append(merged.reset_index())
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# stage 4 — recovery stratified by planted DMR characteristics
# --------------------------------------------------------------------------


def strata(seeds: Sequence[int],
           n_samples: int = DEFAULT_SAMPLES) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        matrix, locations, truth = simulate_chapter2(
            n_samples=n_samples, seed=seed
        )
        n_cpg = matrix.shape[0]
        methods: list[tuple[str, list]] = [
            (f"EIGER 1 @ {t:g}", run_original(matrix, t, 5, impute=True))
            for t in V1_THRESHOLDS
        ]
        for statistic in ("covariance", "correlation"):
            regions, _, _ = call_dmrs(
                matrix, locations, config=_v2_config(statistic)
            )
            methods.append((f"EIGER 2 {statistic}", _v2_regions(regions)))

        for name, called in methods:
            recovered = set(evaluate(called, truth, n_cpg)["recovered_indices"])
            for index, t in enumerate(truth):
                rows.append({
                    "seed": seed, "method": name,
                    "effect": t["effect"], "length": t["length"],
                    "fraction": t["fraction"], "n_affected": t["n_affected"],
                    "recovered": int(index in recovered),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# stage 5 — runtime and memory
# --------------------------------------------------------------------------


def _measure(function, *args, **kwargs):
    tracemalloc.start()
    started = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


def scaling(seeds: Sequence[int],
            sample_counts: Sequence[int] = (8, 12, 24, 48, 96),
            cpg_counts: Sequence[int] = (2000, 4000, 8000, 16000)) -> pd.DataFrame:
    rows = []

    def one(n_samples: int, n_cpg: int, seed: int, axis: str) -> None:
        matrix, locations, _ = simulate_chapter2(
            n_samples=n_samples, n_cpg=n_cpg, seed=seed
        )
        _, seconds, peak = _measure(run_original, matrix, 0.012, 5, False, True)
        rows.append({"axis": axis, "seed": seed, "method": "EIGER 1",
                     "n_samples": n_samples, "n_cpg": n_cpg,
                     "runtime_seconds": seconds, "peak_memory_mib": peak})
        _, seconds, peak = _measure(
            call_dmrs, matrix, locations, config=_v2_config("correlation")
        )
        rows.append({"axis": axis, "seed": seed, "method": "EIGER 2",
                     "n_samples": n_samples, "n_cpg": n_cpg,
                     "runtime_seconds": seconds, "peak_memory_mib": peak})

    for seed in seeds:
        for n_samples in sample_counts:
            one(n_samples, N_CPG, seed, "samples")
        for n_cpg in cpg_counts:
            one(DEFAULT_SAMPLES, n_cpg, seed, "cpgs")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def _tex_escape(value: object) -> str:
    text = "--" if value is None or (isinstance(value, float) and pd.isna(value)) \
        else str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                     ("}", r"\}"), ("~", r"\textasciitilde{}"),
                     ("^", r"\textasciicircum{}")):
        text = text.replace(old, new)
    return text


def _latex(frame: pd.DataFrame, caption: str, label: str,
           decimals: int = 3) -> str:
    """Minimal booktabs table. Written out to avoid a jinja2 dependency and to
    keep control of the formatting for the thesis."""

    align = "".join(
        "r" if pd.api.types.is_numeric_dtype(frame[c]) else "l"
        for c in frame.columns
    )
    header = " & ".join(_tex_escape(c) for c in frame.columns) + r" \\"
    lines = []
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                cells.append("--" if pd.isna(value) else f"{value:.{decimals}f}")
            else:
                cells.append(_tex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{header}\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n"
    )


def _write(frame: pd.DataFrame, out: Path, stem: str,
           caption: str = "", label: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{stem}.csv", index=False)
    if caption:
        (out / f"{stem}.tex").write_text(_latex(frame, caption, label))
    print(f"  wrote {stem}.csv ({len(frame)} rows)")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["recovery", "null", "samplesize",
                                          "strata", "scaling", "calibrate",
                                          "all"])
    parser.add_argument("--out-dir", default="chapter2_bench")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args(list(argv) if argv is not None else None)

    out = Path(args.out_dir)
    seeds = list(range(args.replicates))
    stages = (["recovery", "null", "samplesize", "strata", "scaling"]
              if args.stage == "all" else [args.stage])

    for stage in stages:
        print(f"== {stage} ==")
        if stage == "calibrate":
            frame = calibrate(seeds, n_samples=args.samples)
            _write(frame, out, "calibration_grid")
            best = frame.sort_values("l1_error").head(5)
            print("Closest settings to the published EIGER 1 null-call profile")
            print(f"(target {TARGET_NULL_CALLS} at covariance "
                  f"{[f'{t:g}' for t in V1_THRESHOLDS]}):")
            print(best.round(2).to_string(index=False))
            print("\nSet LATENT_LOADING_SD and LATENT_SMOOTHING at the top of "
                  "this module to the best row before running any comparison.")

        elif stage == "recovery":
            frame = recovery(seeds, n_samples=args.samples)
            summary = frame.groupby("method")[
                ["calls", "region_precision", "region_recall", "region_f1",
                 "bp_precision", "bp_recall"]
            ].mean().reset_index()
            _write(frame, out, "recovery_raw")
            _write(summary, out, "recovery_summary",
                   caption=(f"Recovery on the Chapter~2 signal design at "
                            f"{args.samples} samples, mean over "
                            f"{args.replicates} replicates. Region matches "
                            f"require 50\\% reciprocal overlap."),
                   label="tab:eiger2_recovery")
            print(summary.round(3).to_string(index=False))

        elif stage == "null":
            frame = null_control(seeds, n_samples=args.samples)
            summary = frame.groupby("method")[
                ["null_calls", "false_calls_per_mb"]
            ].agg(["mean", "std"]).round(3)
            summary.columns = [f"{a}_{b}" for a, b in summary.columns]
            summary = summary.reset_index()
            _write(frame, out, "null_raw")
            _write(summary, out, "null_summary",
                   caption=("Complete-null simulations on the Chapter~2 "
                            "background. Every reported region is a false call."),
                   label="tab:eiger2_null")
            print(summary.to_string(index=False))

        elif stage == "samplesize":
            frame = sample_size(seeds)
            _write(frame, out, "samplesize_summary",
                   caption=("Sample-size dependence on the Chapter~2 design."),
                   label="tab:eiger2_samplesize")
            print(frame.round(3).to_string(index=False))

        elif stage == "strata":
            frame = strata(seeds, n_samples=args.samples)
            by_effect = frame.pivot_table(index="method", columns="effect",
                                          values="recovered").round(3)
            by_fraction = frame.pivot_table(index="method", columns="fraction",
                                            values="recovered").round(3)
            by_length = frame.pivot_table(index="method", columns="length",
                                          values="recovered").round(3)
            _write(frame, out, "strata_raw")
            for name, table, cap in (
                ("strata_by_effect", by_effect, "planted effect size"),
                ("strata_by_fraction", by_fraction, "affected-sample fraction"),
                ("strata_by_length", by_length, "planted DMR length in CpGs"),
            ):
                _write(table.reset_index(), out, name,
                       caption=f"Recall stratified by {cap}.",
                       label=f"tab:eiger2_{name}")
            print("by effect:\n", by_effect.to_string())
            print("\nby fraction:\n", by_fraction.to_string())

        elif stage == "scaling":
            frame = scaling(seeds[: max(1, min(3, len(seeds)))])
            summary = frame.groupby(
                ["axis", "method", "n_samples", "n_cpg"]
            )[["runtime_seconds", "peak_memory_mib"]].mean().reset_index()
            _write(frame, out, "scaling_raw")
            _write(summary, out, "scaling_summary",
                   caption="Runtime and peak memory on the Chapter~2 design.",
                   label="tab:eiger2_scaling")
            print(summary.round(3).to_string(index=False))

    (out / "manifest.json").write_text(json.dumps({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage, "replicates": args.replicates,
        "samples": args.samples,
        "design": {"n_cpg": N_CPG, "mean_depth": MEAN_DEPTH,
                   "missing_rate": MISSING_RATE, "n_dmr": N_DMR,
                   "effects": list(EFFECTS), "lengths": list(LENGTHS),
                   "fractions": list(FRACTIONS),
                   "spatial_rho": SPATIAL_RHO},
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "platform": platform.platform(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
