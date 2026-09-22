"""Deterministic synthetic inputs for the coordination-FDR tests.

Small, fully synthetic CpG-by-sample matrices: no cohort data. Chromosomes are
long enough (>= 3,002 CpGs) for the newer profile's distant-lag threshold at
its frozen minimum lag of 1,000 rows, except where a test deliberately wants a
short, lag-infeasible chromosome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CHROMOSOMES = (("chr1", 3_400), ("chr2", 3_100), ("chr3", 900))


def make_dataset(
    seed: int = 20260918,
    *,
    n_samples: int = 20,
    chromosomes: tuple[tuple[str, int], ...] = DEFAULT_CHROMOSOMES,
    n_blocks: int = 10,
    block_cpgs: tuple[int, int] = (6, 16),
    block_strength: float = 0.12,
    missing_fraction: float = 0.04,
    long_gap_fraction: float = 0.03,
    axis_strength: float = 0.0,
    axis_fraction: float = 0.0,
    shuffle_rows: bool = False,
    noise_sd: float = 0.07,
    blocks_follow_axis: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Betas in [0, 1] with NaN gaps, and ``chr/start/end`` locations.

    Background: an independent per-CpG level plus sample noise. Planted blocks
    share one per-sample latent value across their CpGs, which is what makes
    adjacent CpGs co-vary. ``long_gap_fraction`` of inter-CpG gaps exceed 300 bp,
    breaking edge eligibility. An optional genome-wide sample axis shifts a
    random ``axis_fraction`` of CpGs by ``axis_strength`` times a shared
    per-sample score; with ``blocks_follow_axis`` the planted blocks use that
    same score as their latent value.
    """

    rng = np.random.default_rng(seed)
    names: list[str] = []
    starts: list[np.ndarray] = []
    for name, n in chromosomes:
        steps = rng.integers(8, 140, size=n)
        long = rng.random(n) < long_gap_fraction
        steps[long] = rng.integers(400, 2_000, size=int(long.sum()))
        starts.append(10_000 + np.cumsum(steps))
        names.extend([name] * n)
    start = np.concatenate(starts).astype(np.int64)
    n_cpg = start.size

    level = rng.beta(2.0, 2.0, size=n_cpg)
    betas = level[:, None] + rng.normal(0.0, noise_sd, size=(n_cpg, n_samples))
    # Drawn only when an axis is requested, so datasets without one keep the
    # generator stream the frozen-reference comparison was produced from.
    use_axis = bool(axis_strength and axis_fraction) or blocks_follow_axis
    score = rng.normal(0.0, 1.0, size=n_samples) if use_axis else None
    if axis_strength and axis_fraction:
        rows = rng.random(n_cpg) < axis_fraction
        betas[rows] += axis_strength * score[None, :]
    offsets = np.cumsum([0] + [n for _, n in chromosomes])
    for _ in range(n_blocks):
        chrom = int(rng.integers(0, len(chromosomes)))
        length = int(rng.integers(block_cpgs[0], block_cpgs[1] + 1))
        lo = offsets[chrom] + int(rng.integers(0, chromosomes[chrom][1] - length))
        latent = rng.normal(0.0, 1.0, size=n_samples)
        if blocks_follow_axis:
            latent = score
        betas[lo : lo + length] += block_strength * latent[None, :]
    betas = np.clip(betas, 0.0, 1.0)
    if missing_fraction:
        betas[rng.random(betas.shape) < missing_fraction] = np.nan

    locations = pd.DataFrame({"chr": names, "start": start, "end": start + 1})
    if shuffle_rows:
        order = rng.permutation(n_cpg)
        betas = betas[order]
        locations = locations.iloc[order].reset_index(drop=True)
    return betas, locations
