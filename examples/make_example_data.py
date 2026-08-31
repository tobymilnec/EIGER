#!/usr/bin/env python3
"""Write a small example dataset so EIGER can be run without real WGBS data.

    python examples/make_example_data.py

Produces, in ``examples/data/``:

    betas.tsv       50,000 CpGs x 24 samples, methylation proportions in [0, 1]
    cpgs.bed        matching chr / start / end, one chromosome
    truth.tsv       the planted regions, for checking what was recovered

The simulation carries the features that make DMR calling non-trivial:
spatially autocorrelated background methylation, two genome-wide latent factors
(standing in for purity, infiltration or batch), binomial read-depth noise at a
mean depth of 15, and 5% missing observations. Twenty planted regions span a
grid of effect sizes, lengths and affected-sample fractions, including cases
below the detection limit of the covariance statistic.

50,000 CpGs is below ``region_null_mode="member_step"``'s default
~103,000-CpG floor (see ``README.md``'s "Minimum input size" section), so
``eiger2 call``'s default ``region_null_mode="auto"`` falls back to ``"lag"``
here -- deliberately: this example demonstrates that fallback, not the
member-step construction. One chromosome, not two: the current (pre-§2b)
null draws its lag from a genome-wide range rather than per chromosome, so a
two-chromosome layout caps the fraction of null tracks that can survive at
~50% *regardless of total CpG count* (confirmed empirically up to 100,000
CpGs split across two chromosomes) -- below the spec §2a.2 structural gate's
50% floor. That gate is exactly the defect a future chromosome-aware null
construction (spec §2b) fixes; one chromosome sidesteps it here so this
small illustrative fixture clears it.

Clearing that structural gate is not the same as the run reporting `ok`,
though, and this fixture does not reach `ok`: "lag" reuses the edge-level
null directly as the region-level one, and that null's autocorrelation does
not reproduce the observed sliding-window statistic's (``README.md``'s "The
region-level null" section), so it essentially never forms a multi-CpG run by
chance -- every length stratum's own pooled null is genuinely empty (`K=0`).
Before spec §2c, `region_fdr_result`'s aggregate-adequacy floor
(``min_null_regions_for_fdr``) checked the wrong quantity here and refused
the whole run over that emptiness, reporting `inference_unreliable`. §2c
replaced it with a per-stratum null-OPPORTUNITY check, and that is where this
fixture now sits: emptiness alone no longer refuses anything, but one
stratum's null did not have enough opportunity for its `K=0` to count as
evidence.

**What this fixture actually does, as of 2.2.0.** `eiger2 call` on it exits
`0`, all five structural gates pass, 18 regions are reported and 11 are
significant, and the run status is **`partial_inference`**, not
`ok_with_warnings`:

* length stratum `20-49` is REFUSED for insufficient null opportunity: it
  holds 1 observed region over 49,981 of observed opportunity, while the null
  supplied 282,134 against the 499,810 the adequacy criterion requires at
  `m_opportunity=10`. `region_qvalue`, its bounds and `significant` are
  `NaN`/`pd.NA` for that stratum alone. Raising `--n-lag-draws` adds null
  opportunity without weakening the criterion.
* the `6-9` and `10-19` strata have empty pooled nulls that DID have real
  opportunity (`K=0` backed by opportunity gives a finite Garwood upper
  bound, spec §2c.3), so they report genuine `region_qvalue`s and are not
  refused. Every stratum other than `20-49` reports normally.

A refused stratum is a stratum-local outcome, not a broken null: the null
construction is sound and the rest of the run is real inference. It is still
a refusal, and must be read as one -- `partial_inference` is not a completed
inference for that stratum.

The end-to-end smoke tests check exactly this shape rather than a status
string: the structural gates pass, the refused strata are inspected and
printed, and the run writes real output. ``.github/workflows/tests.yml``
prints the refused strata and rejects exit codes above 1;
``make smoke``/``tools/check_smoke_manifest.py`` additionally reads the
status off ``run_manifest.json``, reports the `20-49` refusal explicitly, and
fails on `inference_unreliable`, on `no_regions`, on an undocumented status
or on a status that contradicts the exit code. Neither treats reaching `ok`
as the expectation: "lag" mode's calibration diagnostics keep `ok` out of
reach on any layout.

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

N_CPG = 50_000
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

    # One chromosome, not two: see the module docstring -- the current
    # (pre-§2b) null draws its lag from a genome-wide range rather than per
    # chromosome, so a two-chromosome layout caps null-track survival at
    # ~50% regardless of total CpG count, below spec §2a.2's structural gate
    # floor.
    chrom = np.array(["chr1"] * N_CPG)
    position = np.arange(N_CPG) * 20

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
    print(f"wrote {out}/cpgs.bed    one chromosome")
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
