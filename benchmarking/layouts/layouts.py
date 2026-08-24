"""Real and fitted CpG layouts for the EIGER benchmark.

Every calibration number EIGER has produced to date came from a simulated layout of
uniformly spaced CpGs on one chromosome -- a single eligible run of ~20,000 edges.
Real data, on both WGBS and targeted capture, is ~150,000 runs of median length 3-5
edges. This module supplies layouts that are not fiction:

``load_wgbs_layout``     real hg38 CpG coordinates and real per-CpG depth, from the
                         committed fixture built by ``extract_wgbs_layout.py``.
``generate_epic_layout`` a seeded generator fitted to a targeted panel's empirical gap,
                         run-length and depth structure by ``fit_epic_layout.py``.

``describe_layout`` computes the diagnostics the run manifest is required to carry, and
is the same function the benchmark and the caller should both use.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "load_wgbs_layout",
    "wgbs_chromosome_sizes",
    "generate_epic_layout",
    "describe_layout",
    "eligible_run_lengths",
]


# --------------------------------------------------------------------------
# real WGBS layout
# --------------------------------------------------------------------------


def load_wgbs_layout(
    fixture: str | Path,
    *,
    chrom: str | None = None,
    n_cpg: int | None = None,
    offset: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Real hg38 CpG positions and total depth.

    ``chrom=None`` returns every chromosome in the fixture, concatenated in fixture
    order. ``n_cpg`` takes a contiguous slice starting at ``offset`` -- contiguous,
    because taking a random subset would destroy the run structure that is the whole
    point of using a real layout.

    Returns ``(locations, depth)`` where ``locations`` has the ``chr``/``start``/``end``
    columns EIGER expects.
    """

    with np.load(Path(fixture), allow_pickle=False) as data:
        names = [str(x) for x in data["chrom_names"]]
        starts = data["chrom_start"]
        firsts = data["chrom_first"]
        delta = data["pos_delta"]
        depth = data["depth"]

    bounds = list(starts) + [delta.size]
    frames, depths = [], []
    for i, name in enumerate(names):
        if chrom is not None and name != chrom:
            continue
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        block = delta[lo:hi].astype(np.int64).copy()
        block[0] = 0
        position = np.cumsum(block) + int(firsts[i])
        frames.append(pd.DataFrame({"chr": name, "start": position, "end": position + 1}))
        depths.append(depth[lo:hi])
    if not frames:
        raise KeyError(f"{chrom!r} is not in the fixture; available: {names}")

    locations = pd.concat(frames, ignore_index=True)
    out_depth = np.concatenate(depths).astype(np.int32)
    if n_cpg is not None:
        stop = offset + int(n_cpg)
        if stop > len(locations):
            raise ValueError(
                f"requested CpGs {offset}:{stop} but the selection holds {len(locations)}"
            )
        locations = locations.iloc[offset:stop].reset_index(drop=True)
        out_depth = out_depth[offset:stop]
    return locations, out_depth


def wgbs_chromosome_sizes(fixture: str | Path) -> dict[str, int]:
    """Chromosome name -> CpG count in the vendored fixture, in storage
    order, without materialising the full position/depth arrays
    ``load_wgbs_layout`` returns -- just the small per-chromosome boundary
    metadata already stored in the ``.npz``. Used to build a real
    multi-chromosome layout proportioned to the fixture's actual chromosome
    sizes (spec §2b.5 -- see ``paper_grid.py``'s ``multichrom`` cell).
    """

    with np.load(Path(fixture), allow_pickle=False) as data:
        names = [str(x) for x in data["chrom_names"]]
        starts = list(data["chrom_start"]) + [int(data["pos_delta"].size)]
    return {name: int(starts[i + 1] - starts[i]) for i, name in enumerate(names)}


# --------------------------------------------------------------------------
# fitted targeted layout
# --------------------------------------------------------------------------


def _sample_qtable(table, size, rng) -> np.ndarray:
    """Inverse-CDF sample from a stored quantile table, linearly interpolated."""

    table = np.asarray(table, dtype=float)
    u = rng.random(size) * (table.size - 1)
    lo = np.floor(u).astype(int)
    hi = np.minimum(lo + 1, table.size - 1)
    frac = u - lo
    return table[lo] * (1.0 - frac) + table[hi] * frac


def generate_epic_layout(
    params: str | Path | dict,
    n_cpg: int,
    *,
    seed: int = 0,
    profile: str = "chr1",
    chrom: str = "chr1",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a targeted-panel layout with the fitted fragmentation and depth structure.

    ``profile`` selects which fitted source chromosome to imitate. Chromosomes differ
    enormously in CpG density -- chr19 is far denser than chr1 -- so there is no single
    "targeted layout"; pick the profile whose density regime you want to test against.

    Built run by run, because run length is the property that matters and sampling gaps
    independently would not reproduce it: a run length is drawn from the exact empirical
    distribution, then that many within-run gaps, then one between-run gap.

    Depth is generated in two steps. A latent field is built as a per-run level plus an
    AR(1) residual, which gives the right spatial correlation (most of the variance in
    log depth sits between capture targets, not within them). That field is then
    rank-mapped onto the empirical per-CpG depth distribution, so the marginal is exact
    rather than approximated through a log/exp round trip.
    """

    if not isinstance(params, dict):
        params = json.loads(Path(params).read_text())
    if "profiles" not in params:
        raise KeyError("params predate per-chromosome profiles; re-run fit_epic_layout.py")
    try:
        fit = params["profiles"][profile]
    except KeyError:
        raise KeyError(
            f"{profile!r} not fitted; available: {sorted(params['profiles'])}"
        ) from None

    rng = np.random.default_rng(seed)
    run_values = np.asarray(fit["run_length_values"], dtype=int)
    counts = np.asarray(fit["run_length_counts"], dtype=float)
    run_probs = counts / counts.sum()
    sd = float(fit["latent_within_run_sd"])
    rho = float(fit["latent_within_run_rho"])
    step = sd * np.sqrt(max(1.0 - rho ** 2, 0.0))

    gaps = np.empty(n_cpg, dtype=np.int64)
    latent = np.empty(n_cpg, dtype=float)
    written = 0
    while written < n_cpg:
        take = min(int(rng.choice(run_values, p=run_probs)) + 1, n_cpg - written)

        within = np.rint(_sample_qtable(fit["gap_within_run_quantiles"], take, rng))
        gaps[written : written + take] = np.maximum(within, 1).astype(np.int64)
        if written == 0:
            gaps[0] = 0
        else:
            gaps[written] = max(
                int(round(float(_sample_qtable(fit["gap_between_run_quantiles"], 1, rng)[0]))),
                int(params["max_cpg_gap_bp"]) + 1,
            )

        level = float(_sample_qtable(fit["latent_level_quantiles"], 1, rng)[0])
        noise = np.empty(take)
        noise[0] = rng.normal(0.0, sd)
        for i in range(1, take):
            noise[i] = rho * noise[i - 1] + rng.normal(0.0, step)
        latent[written : written + take] = level + noise
        written += take

    # rank-map the latent field onto the empirical per-CpG depth distribution
    table = np.asarray(fit["depth_quantiles"], dtype=float)
    ranks = np.empty(n_cpg, dtype=float)
    ranks[np.argsort(latent, kind="mergesort")] = (np.arange(n_cpg) + 0.5) / n_cpg
    depth = np.rint(np.interp(ranks, np.linspace(0.0, 1.0, table.size), table))
    depth = np.maximum(depth, 0).astype(np.int32)

    position = np.cumsum(gaps) + 10_000
    locations = pd.DataFrame({"chr": chrom, "start": position, "end": position + 1})
    return locations, depth


# --------------------------------------------------------------------------
# diagnostics -- the manifest fields required by the Stage 1d spec, section A.2
# --------------------------------------------------------------------------
#
# Stage 1d-A (master spec §6.1): re-exported from eiger2.layouts rather than
# defined here. describe_layout() is required in every `eiger2 call` manifest
# for whatever coordinates a caller supplies, not just these benchmark
# fixtures, so it now lives in the installed eiger2 package -- see that
# module's docstring. This keeps the promise this module's own original
# docstring made ("the same function the benchmark and the caller should both
# use") as a fact about the code, not just a comment: there is exactly one
# implementation, cross-checked against this one before the re-export landed
# (identical output on the committed hg38 fixture, chr1, 50k CpGs).

from eiger2.layouts import (  # noqa: E402,F401
    describe_layout,
    eligible_run_lengths,
    eligible_run_spans,
    plantability_fraction,
    realised_length_strata,
    window_eligibility_fraction,
)
