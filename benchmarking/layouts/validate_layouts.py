#!/usr/bin/env python3
"""Validate the fitted EPIC generator against the source figures it must reproduce.

Run after ``fit_epic_layout.py``. Writes VALIDATION.md and exits non-zero if any
metric misses the 10% tolerance the Stage 1d spec sets.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import layouts

TOL = 0.10
params = json.loads(Path("epic_layout_params.json").read_text())
lines = ["# Layout fixture validation", ""]
worst = 0.0
fail = False

lines += ["## EPIC-seq fitted generator vs source", "",
          "Source figures are computed from the patient sample and recorded in",
          "`epic_layout_params.json`; no source coordinates are committed. The generator",
          "is seeded and must reproduce each figure within 10%.", ""]
for chrom in ("chr1", "chr7", "chr19"):
    tg = params["validation_targets"][chrom]
    loc, dep = layouts.generate_epic_layout(params, n_cpg=tg["n_cpg"], seed=0, profile=chrom)
    got = layouts.describe_layout(loc, params["max_cpg_gap_bp"], dep)
    runs, depth = got["layout_eligible_run_lengths_edges"], got["layout_depth"]
    rows = [
        ("eligible fraction", tg["eligible_fraction"], got["layout_eligible_fraction"]),
        ("run median (edges)", tg["run_len_median"], runs["q50"]),
        ("run q90 (edges)", tg["run_len_q90"], runs["q90"]),
        ("frac runs <3 edges", tg["frac_runs_under_3_edges"], runs["fraction_under_3_edges"]),
        ("depth median", tg["depth_median"], depth["median"]),
        ("frac depth>=5", tg["frac_depth_ge5"], depth["fraction_ge5"]),
        ("frac depth>=10", tg["frac_depth_ge10"], depth["fraction_ge10"]),
    ]
    lines += [f"### {chrom} (n_cpg = {tg['n_cpg']:,}, seed 0)", "",
              "| metric | source | generated | rel. error | |", "|---|---|---|---|---|"]
    for name, a, b in rows:
        err = abs(b - a) / abs(a) if a else 0.0
        worst = max(worst, err)
        ok = err <= TOL
        fail |= not ok
        lines.append(f"| {name} | {a:.4f} | {b:.4f} | {err:.1%} | {'OK' if ok else '**FAIL**'} |")
    lines.append("")

lines += [f"Worst relative error across all chromosomes and metrics: **{worst:.1%}** "
          f"(tolerance {TOL:.0%}).", ""]

lines += ["## Real WGBS layout (committed fixture)", "",
          "Read back from `wgbs_layout_hg38.npz`. These are measurements, not targets --",
          "the fixture holds real hg38 coordinates and real depth.", "",
          "| chromosome | CpGs | eligible % | runs | run q50 | run q90 | max run | runs <3 | depth q50 | depth>=10 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
for chrom in ("chr1", "chr7", "chr19"):
    loc, dep = layouts.load_wgbs_layout("wgbs_layout_hg38.npz", chrom=chrom)
    d = layouts.describe_layout(loc, 300, dep)
    r = d["layout_eligible_run_lengths_edges"]
    lines.append(
        f"| {chrom} | {d['layout_n_cpg']:,} | {d['layout_eligible_fraction']:.1%} | "
        f"{r['n_runs']:,} | {r['q50']:.0f} | {r['q90']:.0f} | {r['max']:,} | "
        f"{r['fraction_under_3_edges']:.0%} | {d['layout_depth']['median']:.0f} | "
        f"{d['layout_depth']['fraction_ge10']:.1%} |")

lines += ["", "## For contrast: the layout every prior EIGER benchmark used", "",
          "| layout | CpGs | eligible % | runs | run q50 | max run |", "|---|---|---|---|---|---|",
          "| `dense` (uniform 2 bp, one chromosome) | 20,000 | 100% | **1** | **19,999** | 19,999 |", ""]

Path("VALIDATION.md").write_text("\n".join(lines))
print("\n".join(lines))
sys.exit(1 if fail else 0)
