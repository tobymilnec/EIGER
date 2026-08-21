#!/usr/bin/env python3
"""Fit a seeded generator to a targeted (EPIC-seq / methyl-capture) CpG layout.

No patient-derived coordinates are written. The output is a set of empirical
distributions -- inter-CpG gaps, eligible-run lengths, and per-run depth levels --
sufficient to regenerate a layout with the same fragmentation and coverage structure,
plus a validation table recording the source figures the generator must reproduce.

Methylation values are never read: only column 0 (chrom), column 1 (position),
and columns 4+5 (counts, summed to total depth) of the Bismark .cov are used.

Usage:
    python fit_epic_layout.py --cov SAMPLE.bismark.cov.gz --chroms 1 7 19 \
        --max-cpg-gap-bp 300 --out epic_layout_params.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

N_QUANTILES = 1001


def _qtable(values: np.ndarray) -> list[float]:
    q = np.linspace(0.0, 1.0, N_QUANTILES)
    return [float(v) for v in np.quantile(np.asarray(values, dtype=float), q)]


def read_cov(path: Path, chroms: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pos: dict[str, list[int]] = {c: [] for c in chroms}
    dep: dict[str, list[int]] = {c: [] for c in chroms}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            f = line.split("\t")
            c = f[0]
            if c in chroms:
                pos[c].append(int(f[1]))
                dep[c].append(int(f[4]) + int(f[5]))
    return {c: (np.asarray(pos[c], dtype=np.int64), np.asarray(dep[c], dtype=np.float64))
            for c in chroms if pos[c]}


def runs_of(eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end CpG indices of each maximal eligible run (end inclusive)."""

    breaks = np.flatnonzero(~eligible)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [eligible.size]))
    return starts, ends


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cov", required=True, type=Path)
    ap.add_argument("--chroms", nargs="+", default=["1", "7", "19"])
    ap.add_argument("--max-cpg-gap-bp", type=int, default=300)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    data = read_cov(args.cov, set(args.chroms))
    if not data:
        raise SystemExit("no rows matched the requested chromosomes")
    validation: dict[str, dict] = {}

    # Fit ONE PROFILE PER CHROMOSOME. Chromosomes differ enormously in CpG density --
    # chr19 is far denser than chr1 or chr7 -- so a pooled fit reproduces none of them.
    profiles: dict[str, dict] = {}
    for chrom, (pos, depth) in sorted(data.items()):
        order = np.argsort(pos, kind="mergesort")
        pos, depth = pos[order], depth[order]
        gap = np.diff(pos)
        eligible = gap <= args.max_cpg_gap_bp
        starts, ends = runs_of(eligible)
        run_len = ends - starts                       # in EDGES; 0 is a lone CpG
        run_size = run_len + 1

        log_depth = np.log1p(depth)
        levels = np.asarray([log_depth[s : e + 1].mean() for s, e in zip(starts, ends)])
        resid = np.concatenate(
            [log_depth[s : e + 1] - levels[i] for i, (s, e) in enumerate(zip(starts, ends))
             if e > s]
        ) if (run_len > 0).any() else np.zeros(1)

        adjacent = np.hstack(
            [np.vstack([log_depth[s:e], log_depth[s + 1 : e + 1]])
             for s, e in zip(starts, ends) if e > s]
        )
        observed_adjacent = float(np.corrcoef(adjacent[0], adjacent[1])[0, 1])
        weighted_levels = np.repeat(levels, run_size)
        share_between = float(weighted_levels.var() /
                              max(weighted_levels.var() + resid.var(), 1e-12))
        rho = float(np.clip((observed_adjacent - share_between) /
                            max(1.0 - share_between, 1e-9), 0.0, 0.95))
        values, counts = np.unique(run_len, return_counts=True)

        profiles[f"chr{chrom}"] = {
            "gap_within_run_quantiles": _qtable(gap[eligible]),
            "gap_between_run_quantiles": _qtable(gap[~eligible]),
            "run_length_values": [int(v) for v in values],
            "run_length_counts": [int(c) for c in counts],
            # The latent depth field is generated for its CORRELATION structure and then
            # rank-mapped onto this table, so the per-CpG marginal is exact by
            # construction rather than approximated through a log/exp round trip.
            "depth_quantiles": _qtable(depth),
            "latent_level_quantiles": _qtable(weighted_levels),
            "latent_within_run_sd": float(np.sqrt(resid.var())),
            "latent_within_run_rho": rho,
            "between_run_variance_share": share_between,
            "observed_adjacent_log_depth_corr": observed_adjacent,
        }
        validation[f"chr{chrom}"] = {
            "n_cpg": int(pos.size),
            "eligible_fraction": float(eligible.mean()),
            "n_runs": int(run_len.size),
            "run_len_median": float(np.median(run_len)),
            "run_len_q90": float(np.quantile(run_len, 0.9)),
            "run_len_q99": float(np.quantile(run_len, 0.99)),
            "run_len_max": int(run_len.max()),
            "frac_runs_under_3_edges": float(np.mean(run_len < 3)),
            "depth_median": float(np.median(depth)),
            "frac_depth_ge5": float(np.mean(depth >= 5)),
            "frac_depth_ge10": float(np.mean(depth >= 10)),
            "adjacent_log_depth_corr": observed_adjacent,
        }

    params = {
        "_comment": "Fitted layout generator for a targeted bisulfite panel. Empirical "
                    "distributions only: no source coordinates, no methylation values.",
        "max_cpg_gap_bp": args.max_cpg_gap_bp,
        "source_chromosomes": args.chroms,
        "n_source_cpg": int(sum(len(v[0]) for v in data.values())),
        "profiles": profiles,
        "validation_targets": validation,
    }
    args.out.write_text(json.dumps(params, indent=1))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")
    for name, v in validation.items():
        print(f"  {name}: {v['n_cpg']:>9,} CpGs  eligible={v['eligible_fraction']:.1%}  "
              f"run q50={v['run_len_median']:.0f} q90={v['run_len_q90']:.0f}  "
              f"depth q50={v['depth_median']:.0f}  adj-corr={v['adjacent_log_depth_corr']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
