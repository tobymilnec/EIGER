#!/usr/bin/env python3
"""Write a small synthetic dataset for trying ``eiger2 call``.

    python make_example_data.py [--out example_data]

Writes, into ``--out`` (default ``example_data/`` in the current directory):

    betas.tsv    10,000 CpGs x 60 samples, methylation proportions in [0, 1],
                 2% missing values written as NA
    cpgs.tsv     chr / start / end for each row of betas.tsv
    matrix.tsv   the same data as one table: chr, start, end, then samples
    truth.tsv    the planted regions, for checking what was recovered

The data are simulated. Two chromosomes of 5,000 CpGs each are long enough for
the default profile's distant-lag threshold (at least 3,002 CpGs per
chromosome at the default minimum lag of 1,000 rows and window of 4). Twelve
regions of 8-16 CpGs share a per-sample value across their CpGs, which is what
makes neighbouring CpGs co-vary. Six of them follow a genome-wide sample axis
that also shifts 3% of scattered single CpGs; the other six follow their own
local sample pattern. The output is identical on every run: numpy's PCG64
generator is seeded with a fixed value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260919
N_SAMPLES = 60
CHROMOSOMES = (("chr1", 5_000), ("chr2", 5_000))
N_REGIONS = 12
REGION_CPGS = (8, 16)
REGION_STRENGTH = 0.15
AXIS_CPG_FRACTION = 0.03
AXIS_STRENGTH = 0.15
NOISE_SD = 0.07
MISSING_FRACTION = 0.02
LONG_GAP_FRACTION = 0.03


def simulate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)

    names: list[str] = []
    starts: list[np.ndarray] = []
    for name, n in CHROMOSOMES:
        steps = rng.integers(10, 150, size=n)
        long = rng.random(n) < LONG_GAP_FRACTION
        steps[long] = rng.integers(400, 2_000, size=int(long.sum()))
        starts.append(10_000 + np.cumsum(steps))
        names.extend([name] * n)
    start = np.concatenate(starts).astype(np.int64)
    n_cpg = start.size

    level = rng.beta(2.0, 2.0, size=n_cpg)
    betas = level[:, None] + rng.normal(0.0, NOISE_SD, size=(n_cpg, N_SAMPLES))

    # a genome-wide sample axis carried by scattered single CpGs
    axis_score = rng.normal(0.0, 1.0, size=N_SAMPLES)
    axis_rows = rng.random(n_cpg) < AXIS_CPG_FRACTION
    betas[axis_rows] += AXIS_STRENGTH * axis_score[None, :]

    # planted regions, spread evenly and away from chromosome ends
    truth = []
    offsets = np.cumsum([0] + [n for _, n in CHROMOSOMES])
    per_chromosome = N_REGIONS // len(CHROMOSOMES)
    for index in range(N_REGIONS):
        chrom = index % len(CHROMOSOMES)
        slot = index // len(CHROMOSOMES)
        n_chrom = CHROMOSOMES[chrom][1]
        spacing = (n_chrom - 400) // per_chromosome
        length = int(rng.integers(REGION_CPGS[0], REGION_CPGS[1] + 1))
        first = int(offsets[chrom] + 200 + slot * spacing
                    + rng.integers(0, spacing - REGION_CPGS[1]))
        last = first + length - 1
        follows_axis = index % 2 == 0
        latent = axis_score if follows_axis else rng.normal(0.0, 1.0, N_SAMPLES)
        betas[first : last + 1] += REGION_STRENGTH * latent[None, :]
        truth.append({
            "chr": names[first],
            "start": int(start[first]),
            "end": int(start[last]) + 1,
            "n_cpg": length,
            "pattern": "genome_wide_axis" if follows_axis else "local",
        })

    betas = np.clip(betas, 0.0, 1.0)
    betas[rng.random(betas.shape) < MISSING_FRACTION] = np.nan

    samples = [f"S{i:02d}" for i in range(1, N_SAMPLES + 1)]
    beta_frame = pd.DataFrame(betas, columns=samples)
    locations = pd.DataFrame({"chr": names, "start": start, "end": start + 1})
    return beta_frame, locations, pd.DataFrame(truth)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="example_data",
                        help="output directory (default: example_data)")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    betas, locations, truth = simulate()
    options = {"sep": "\t", "index": False, "na_rep": "NA"}
    betas.to_csv(out / "betas.tsv", float_format="%.4f", **options)
    locations.to_csv(out / "cpgs.tsv", **options)
    pd.concat([locations, betas], axis=1).to_csv(
        out / "matrix.tsv", float_format="%.4f", **options
    )
    truth.to_csv(out / "truth.tsv", **options)

    print(f"wrote {out}/betas.tsv, cpgs.tsv, matrix.tsv, truth.tsv "
          f"({len(locations):,} CpGs x {betas.shape[1]} samples, "
          f"{len(truth)} planted regions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
