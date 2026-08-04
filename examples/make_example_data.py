#!/usr/bin/env python3
"""Write a small example dataset so EIGER can be run without real WGBS data.

    python examples/make_example_data.py

Produces, in ``examples/data/``:

    betas.tsv       5,000 CpGs x 24 samples, methylation proportions in [0, 1]
    cpgs.bed        matching chr / start / end, two chromosomes
    truth.tsv       the planted regions, for checking what was recovered

The simulation carries the features that make DMR calling non-trivial:
spatially autocorrelated background methylation, two genome-wide latent factors
(standing in for purity, infiltration or batch), binomial read-depth noise at a
mean depth of 15, and 5% missing observations. Twenty planted regions span a
grid of effect sizes, lengths and affected-sample fractions, including cases
below the detection limit of the covariance statistic.

The data are simulated, not real, and are provided only so that the software can
be exercised end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Python"))

N_CPG = 5000
N_SAMPLES = 24
MEAN_DEPTH = 15
MISSING_RATE = 0.05
SEED = 0

EFFECTS = (0.15, 0.25, 0.40)
LENGTHS = (6, 12, 24)
FRACTIONS = (0.08, 0.25, 0.42)
N_PLANTED = 20

SPATIAL_RHO = 0.65
BASELINE_SD = 0.08
SAMPLE_NOISE_SD = 0.03
N_LATENT = 2
LATENT_LOADING_SD = 0.05
LATENT_SMOOTHING = 6


def main() -> int:
    rng = np.random.default_rng(SEED)

    # spatially autocorrelated baseline, shared across samples
    innovation = rng.normal(0, BASELINE_SD, N_CPG)
    baseline = np.empty(N_CPG)
    baseline[0] = innovation[0]
    for i in range(1, N_CPG):
        baseline[i] = SPATIAL_RHO * baseline[i - 1] + innovation[i]
    beta = 0.5 + baseline[:, None] + rng.normal(
        0, SAMPLE_NOISE_SD, (N_CPG, N_SAMPLES)
    )

    # latent nuisance: samples differ from one another in a regionally
    # coherent way, as purity and cell composition do
    scores = rng.normal(0, 1, (N_SAMPLES, N_LATENT))
    raw = rng.normal(0, LATENT_LOADING_SD, (N_CPG, N_LATENT))
    kernel = np.ones(LATENT_SMOOTHING) / LATENT_SMOOTHING
    loadings = np.column_stack([
        np.convolve(raw[:, k], kernel, mode="same") * np.sqrt(LATENT_SMOOTHING)
        for k in range(N_LATENT)
    ])
    beta = beta + loadings @ scores.T

    # planted regions
    combos = [
        (effect, length, fraction)
        for effect in EFFECTS
        for length in LENGTHS
        for fraction in FRACTIONS
    ]
    chosen = [combos[i % len(combos)] for i in range(N_PLANTED)]
    rng.shuffle(chosen)
    truth = []
    margin, spacing = 100, (N_CPG - 200) // N_PLANTED
    for index, (effect, length, fraction) in enumerate(chosen):
        start = margin + index * spacing + int(rng.integers(0, spacing // 4))
        end = min(start + length - 1, N_CPG - 1)
        size = max(1, int(round(fraction * N_SAMPLES)))
        members = rng.choice(N_SAMPLES, size, replace=False)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        beta[start : end + 1, members] += sign * effect
        truth.append({
            "cpg_index_start": start, "cpg_index_end": end,
            "effect": effect, "length": end - start + 1,
            "fraction": fraction, "n_affected": size,
            "direction": "hyper" if sign > 0 else "hypo",
            "samples": ";".join(f"S{i:02d}" for i in sorted(members)),
        })
    beta = np.clip(beta, 0.02, 0.98)

    # read-depth sampling, then dropout
    depth = np.maximum(rng.poisson(MEAN_DEPTH, beta.shape), 1)
    observed = rng.binomial(depth, beta) / depth
    observed[rng.random(observed.shape) < MISSING_RATE] = np.nan

    # two chromosomes, so the chromosome-aware null is exercised
    half = N_CPG // 2
    chrom = np.where(np.arange(N_CPG) < half, "chr1", "chr2")
    position = np.where(
        np.arange(N_CPG) < half,
        np.arange(N_CPG) * 20,
        (np.arange(N_CPG) - half) * 20,
    )

    out = Path(__file__).resolve().parent / "data"
    out.mkdir(parents=True, exist_ok=True)
    samples = [f"S{i:02d}" for i in range(N_SAMPLES)]
    pd.DataFrame(observed, columns=samples).to_csv(
        out / "betas.tsv", sep="\t", index=False, float_format="%.4f",
        na_rep="NA",
    )
    pd.DataFrame({"chr": chrom, "start": position, "end": position + 1}).to_csv(
        out / "cpgs.bed", sep="\t", index=False
    )
    pd.DataFrame(truth).to_csv(out / "truth.tsv", sep="\t", index=False)

    print(f"wrote {out}/betas.tsv   {N_CPG} CpGs x {N_SAMPLES} samples")
    print(f"wrote {out}/cpgs.bed    two chromosomes")
    print(f"wrote {out}/truth.tsv   {len(truth)} planted regions")
    print(f"missing fraction        {np.mean(~np.isfinite(observed)):.3f}")
    print()
    print("Now try:")
    print("  eiger2 call    --betas examples/data/betas.tsv "
          "--locations examples/data/cpgs.bed --out examples/out_eiger2")
    print("  eiger2 call    --betas examples/data/betas.tsv "
          "--locations examples/data/cpgs.bed --out examples/out_legacy --legacy")
    print("  eiger2 compare --betas examples/data/betas.tsv "
          "--locations examples/data/cpgs.bed --out examples/out_compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
