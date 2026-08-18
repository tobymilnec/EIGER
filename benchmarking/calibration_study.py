#!/usr/bin/env python3
"""Is the region evidence score calibratable? A four-step study.

Background. The two-groups local FDR in :mod:`eiger2.probability` failed its
reliability check (expected calibration error 0.35-0.50). But the diagnostic
that should have been run first shows AUC = 1.000 against planted truth, so the
score *ranks* perfectly and only its scale is wrong. That earlier experiment was
also confounded: it used ``min_cpgs = 2`` to manufacture false positives, which
produced a mixture with no overlap at all -- 40-CpG planted regions against
2-CpG fragments. With perfectly separated components the true local FDR is a
step function, and a kernel density estimate necessarily smooths it into a ramp.
The intermediate scores may therefore be a bandwidth artefact rather than a
failure of the model.

This module runs the four steps that settle it, in order.

``overlap``    Plant regions at effect sizes near the detection limit so that
               true and false regions have genuinely overlapping area
               distributions. Report AUC and calibration error. **If the score
               is calibrated here, the remaining steps are unnecessary.**

``bandwidth``  Sweep the KDE bandwidth multiplier. If over-smoothing is the
               cause, calibration error should fall as the bandwidth narrows.

``pi0``        Compare the pipeline-null estimator against pi0 = 1 and against
               a Storey-type estimator on the area distribution.

``isotonic``   Fit an isotonic regression from score to observed truth on
               training seeds and evaluate on held-out seeds. Given a
               well-ranking score this is guaranteed to calibrate in-range; the
               question is whether it transfers across cohort sizes and effect
               sizes, which is what the held-out split measures.

Usage
-----
    python -m eiger2.calibration_study all --out-dir calibration_study

Each stage writes a CSV. ``all`` runs the four in sequence and prints a verdict.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from eiger2.cli import _git_commit, _peak_memory_mb
from eiger2.core import (
    Eiger2Config,
    centre_cpgs,
    deflate,
    edge_statistic,
    eligible_edges,
    estimate_latent_basis,
    null_threshold,
    null_tracks,
    region_area,
    region_fdr,
    segment,
)
from eiger2.manifest_utils import build_manifest_envelope
from eiger2.probability import calibration_error, local_fdr, reliability_table

# --------------------------------------------------------------------------
# regimes
# --------------------------------------------------------------------------

#: Effect sizes producing a genuinely overlapping mixture. The point of the
#: ``overlap`` regime is that true regions should be hard to tell from false
#: ones on area alone; planting only strong, long regions guarantees separation
#: and makes the calibration question unanswerable.
OVERLAP_EFFECTS = (0.10, 0.14, 0.18)
OVERLAP_LENGTHS = (4, 6, 8)

SEPARATED_EFFECTS = (0.30, 0.40)
SEPARATED_LENGTHS = (24, 32)


def _simulate(regime: str, seed: int, n_samples: int):
    """Return ``(matrix, locations, truth)`` for a named regime."""

    import chapter2_benchmark as c2

    saved = (c2.EFFECTS, c2.LENGTHS)
    try:
        if regime == "overlap":
            c2.EFFECTS, c2.LENGTHS = OVERLAP_EFFECTS, OVERLAP_LENGTHS
        elif regime == "separated":
            c2.EFFECTS, c2.LENGTHS = SEPARATED_EFFECTS, SEPARATED_LENGTHS
        return c2.simulate_chapter2(
            n_samples=n_samples, n_dmr=40, seed=seed
        )
    finally:
        c2.EFFECTS, c2.LENGTHS = saved


# --------------------------------------------------------------------------
# one run: segment, score, label
# --------------------------------------------------------------------------


def score_one(
    matrix,
    locations,
    truth,
    *,
    config: Eiger2Config,
    bandwidth_scale: float = 1.0,
    pi0: float | None = None,
    min_overlap: int = 3,
) -> pd.DataFrame:
    """Segment, compute the evidence score, and label each region by truth."""

    residuals = centre_cpgs(matrix)
    basis, _ = estimate_latent_basis(residuals, config)
    deflated = deflate(residuals, basis, missing=config.deflation_missing)
    statistic = edge_statistic(deflated, config)
    eligible = eligible_edges(locations, config)
    threshold, _ = null_threshold(deflated, config)

    regions = segment(statistic, eligible, threshold,
                      min_cpgs=config.min_cpgs,
                      max_gap_edges=config.max_gap_edges)
    if regions.size == 0:
        return pd.DataFrame()

    areas = region_area(statistic, regions, threshold)
    null_areas, null_edges = [], 0
    for track, null_eligible in null_tracks(deflated, config):
        finite = null_eligible & np.isfinite(track)
        null_edges += int(finite.sum())
        null_regions = segment(track, finite, threshold,
                               min_cpgs=config.min_cpgs,
                               max_gap_edges=config.max_gap_edges)
        null_areas.append(region_area(track, null_regions, threshold))

    observed_edges = int(np.sum(eligible & np.isfinite(statistic)))
    estimate = local_fdr(
        areas, null_areas,
        observed_edges=observed_edges, null_edges=max(null_edges, 1),
        bandwidth_scale=bandwidth_scale, pi0=pi0,
        min_null_regions=config.min_null_regions_for_posterior,
    )
    qvalues = region_fdr(
        areas, null_areas,
        observed_edges=observed_edges, null_edges=max(null_edges, 1),
    )

    rows = []
    for index, (first, last) in enumerate(regions):
        is_true = any(
            min(int(last), t["end"]) - max(int(first), t["start"]) + 1 >= min_overlap
            for t in truth
        )
        rows.append({
            "n_cpg": int(last) - int(first) + 1,
            "area": float(areas[index]),
            "qvalue": float(qvalues[index]),
            "score": float(estimate.posterior[index]),
            "score_conservative": float(1.0 - estimate.fdr_conservative[index]),
            "extrapolated": bool(estimate.extrapolated[index]),
            "is_true": bool(is_true),
            "pi0": estimate.pi0,
            "status": estimate.status,
        })
    return pd.DataFrame(rows)


def _auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    keep = np.isfinite(scores)
    labels, scores = labels[keep], scores[keep]
    order = np.argsort(scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[j + 1] == scores[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = labels.sum()
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _report(frame: pd.DataFrame, column: str = "score", n_bins: int = 10) -> dict:
    usable = frame.dropna(subset=[column])
    if usable.empty or usable["is_true"].nunique() < 2:
        return {"n": len(usable), "true_fraction": np.nan, "auc": np.nan,
                "ece": np.nan, "max_ce": np.nan}
    table = reliability_table(usable[column].to_numpy(),
                              usable["is_true"].to_numpy(), n_bins=n_bins)
    error = calibration_error(table)
    return {
        "n": len(usable),
        "true_fraction": float(usable["is_true"].mean()),
        "auc": _auc(usable["is_true"], usable[column]),
        "ece": error["expected_calibration_error"],
        "max_ce": error["max_calibration_error"],
    }


def _collect(regime: str, seeds: Sequence[int], n_samples: int,
             *, bandwidth_scale: float = 1.0, pi0: float | None = None,
             null_replicates: int = 300, min_cpgs: int = 3) -> pd.DataFrame:
    config = Eiger2Config(
        null_replicates=null_replicates, min_cpgs=min_cpgs,
        compute_posterior=True, min_null_regions_for_posterior=50,
    )
    frames = []
    for seed in seeds:
        matrix, locations, truth = _simulate(regime, seed, n_samples)
        frame = score_one(matrix, locations, truth, config=config,
                          bandwidth_scale=bandwidth_scale, pi0=pi0)
        if not frame.empty:
            frame["seed"] = seed
            frame["regime"] = regime
            frame["n_samples"] = n_samples
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
# stage 1 — overlapping mixture
# --------------------------------------------------------------------------


def stage_overlap(seeds, sample_sizes=(24, 48)) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, raw = [], []
    for regime in ("separated", "overlap"):
        for n_samples in sample_sizes:
            frame = _collect(regime, seeds, n_samples)
            if frame.empty:
                continue
            raw.append(frame)
            for column in ("score", "score_conservative"):
                rows.append({"regime": regime, "n_samples": n_samples,
                             "variant": column, **_report(frame, column)})
            rows.append({"regime": regime, "n_samples": n_samples,
                         "variant": "area (ranking only)",
                         **_report(frame, "area"), "ece": np.nan,
                            "max_ce": np.nan})
    return pd.DataFrame(rows), (pd.concat(raw, ignore_index=True)
                                if raw else pd.DataFrame())


# --------------------------------------------------------------------------
# stage 2 — bandwidth
# --------------------------------------------------------------------------


def stage_bandwidth(seeds, n_samples=24,
                    scales=(0.25, 0.4, 0.6, 0.8, 1.0, 1.5)) -> pd.DataFrame:
    rows = []
    for regime in ("overlap", "separated"):
        for scale in scales:
            frame = _collect(regime, seeds, n_samples, bandwidth_scale=scale)
            if frame.empty:
                continue
            rows.append({"regime": regime, "bandwidth_scale": scale,
                         **_report(frame)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# stage 3 — pi0
# --------------------------------------------------------------------------


def _storey_pi0(areas: np.ndarray, null_areas: np.ndarray,
                lam: float = 0.5) -> float:
    """Storey-type estimator on the area scale.

    The fraction of observed regions below the ``lam`` quantile of the null,
    divided by ``lam``, estimates the null proportion.
    """

    cut = float(np.quantile(null_areas, lam))
    below = float(np.mean(np.asarray(areas) <= cut))
    return float(np.clip(below / lam, 0.0, 1.0))


def stage_pi0(seeds, n_samples=24) -> pd.DataFrame:
    rows = []
    for regime in ("overlap", "separated"):
        base = _collect(regime, seeds, n_samples)
        if base.empty:
            continue
        truth_pi0 = 1.0 - base["is_true"].mean()
        rows.append({"regime": regime, "estimator": "pipeline null",
                     "pi0": base["pi0"].mean(), "true_pi0": truth_pi0,
                     **_report(base, "score")})
        rows.append({"regime": regime, "estimator": "pi0 = 1",
                     "pi0": 1.0, "true_pi0": truth_pi0,
                     **_report(base, "score_conservative")})
        oracle = _collect(regime, seeds, n_samples, pi0=float(truth_pi0))
        if not oracle.empty:
            rows.append({"regime": regime, "estimator": "oracle pi0",
                         "pi0": truth_pi0, "true_pi0": truth_pi0,
                         **_report(oracle, "score")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# stage 4 — isotonic recalibration, with a held-out split
# --------------------------------------------------------------------------


def _isotonic_fit(scores: np.ndarray, labels: np.ndarray):
    """Pool-adjacent-violators isotonic regression. Returns a step function.

    Written out rather than taken from scikit-learn so the module has no extra
    dependency and the fitted map can be serialised transparently.
    """

    order = np.argsort(scores, kind="mergesort")
    x = np.asarray(scores, dtype=float)[order]
    y = np.asarray(labels, dtype=float)[order]
    weight = np.ones_like(y)
    values, weights, bounds = [], [], []
    for value, w in zip(y, weight):
        values.append(value)
        weights.append(w)
        bounds.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2, b2 = values.pop(), weights.pop(), bounds.pop()
            v1, w1, b1 = values.pop(), weights.pop(), bounds.pop()
            values.append((v1 * w1 + v2 * w2) / (w1 + w2))
            weights.append(w1 + w2)
            bounds.append(b1 + b2)
    fitted = np.repeat(values, bounds)
    return x, fitted


def _isotonic_apply(fit, scores):
    x, y = fit
    return np.interp(np.asarray(scores, dtype=float), x, y)


def stage_isotonic(seeds, n_samples_train=24, n_samples_test=(24, 48),
                   regime="overlap") -> pd.DataFrame:
    seeds = list(seeds)
    split = max(1, len(seeds) // 2)
    train_seeds, test_seeds = seeds[:split], seeds[split:]
    if not test_seeds:
        raise ValueError("need at least two seeds to hold one out")

    train = _collect(regime, train_seeds, n_samples_train)
    if train.empty:
        return pd.DataFrame()
    usable = train.dropna(subset=["score"])
    fit = _isotonic_fit(usable["score"].to_numpy(), usable["is_true"].to_numpy())

    rows = [{"split": "train", "n_samples": n_samples_train,
             "variant": "raw", **_report(train, "score")}]
    train = train.copy()
    train["calibrated"] = _isotonic_apply(fit, train["score"])
    rows.append({"split": "train", "n_samples": n_samples_train,
                 "variant": "isotonic", **_report(train, "calibrated")})

    for n_samples in n_samples_test:
        test = _collect(regime, test_seeds, n_samples)
        if test.empty:
            continue
        rows.append({"split": "held-out", "n_samples": n_samples,
                     "variant": "raw", **_report(test, "score")})
        test = test.copy()
        test["calibrated"] = _isotonic_apply(fit, test["score"])
        rows.append({"split": "held-out", "n_samples": n_samples,
                     "variant": "isotonic", **_report(test, "calibrated")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _write(frame: pd.DataFrame, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{stem}.csv", index=False)
    print(f"  wrote {stem}.csv ({len(frame)} rows)")
    if not frame.empty:
        print(frame.round(3).to_string(index=False))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["overlap", "bandwidth", "pi0",
                                          "isotonic", "all"])
    parser.add_argument("--out-dir", default="calibration_study")
    parser.add_argument("--replicates", type=int, default=6)
    args = parser.parse_args(list(argv) if argv is not None else None)

    out = Path(args.out_dir)
    seeds = list(range(args.replicates))
    stages = (["overlap", "bandwidth", "pi0", "isotonic"]
              if args.stage == "all" else [args.stage])

    started = time.perf_counter()
    verdict = {}
    for stage in stages:
        print(f"\n== {stage} ==")
        if stage == "overlap":
            summary, raw = stage_overlap(seeds)
            _write(summary, out, "overlap_summary")
            if not raw.empty:
                _write(raw, out, "overlap_regions")
            hit = summary[(summary.regime == "overlap")
                          & (summary.variant == "score")]
            if not hit.empty:
                verdict["overlap_ece"] = float(hit["ece"].mean())
                verdict["overlap_auc"] = float(hit["auc"].mean())
        elif stage == "bandwidth":
            _write(stage_bandwidth(seeds), out, "bandwidth_summary")
        elif stage == "pi0":
            _write(stage_pi0(seeds), out, "pi0_summary")
        elif stage == "isotonic":
            _write(stage_isotonic(seeds), out, "isotonic_summary")

    if verdict:
        ece = verdict.get("overlap_ece", float("nan"))
        print("\n" + "=" * 70)
        if np.isfinite(ece) and ece < 0.10:
            print(f"VERDICT: calibrated in the overlapping regime (ECE {ece:.3f}).")
            print("Report it as a probability, with the reliability diagram and")
            print("the range of effect sizes and cohort sizes tested.")
        else:
            print(f"VERDICT: not calibrated in the overlapping regime "
                  f"(ECE {ece:.3f}).")
            print("Keep the name 'evidence score'. Use the bandwidth, pi0 and")
            print("isotonic tables to decide which remedy to pursue.")
        print("=" * 70)

    elapsed = time.perf_counter() - started
    resolved_configuration = {
        "stage": args.stage, "replicates": args.replicates,
        "overlap_effects": list(OVERLAP_EFFECTS),
        "overlap_lengths": list(OVERLAP_LENGTHS),
        "separated_effects": list(SEPARATED_EFFECTS),
        "separated_lengths": list(SEPARATED_LENGTHS),
    }
    envelope = build_manifest_envelope(
        config_dict=resolved_configuration,
        git_commit=_git_commit(),
        # This study simulates every matrix it scores; nothing is read from disk.
        input_checksums={},
        random_seed=seeds[0] if seeds else 0,
        derived_seeds={"replicate_seeds": seeds},
        runtime_seconds=elapsed,
        peak_memory_mb=_peak_memory_mb(),
        warnings=[],
        # Every Eiger2Config built by this study leaves fdr_decision at its
        # 'point' default; the study characterises the evidence score itself
        # rather than making a point/upper significance call.
        fdr_decision="point",
    )
    (out / "manifest.json").write_text(json.dumps({
        **envelope,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **resolved_configuration,
        "verdict": verdict,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
