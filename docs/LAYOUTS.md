# Layouts: the fixture, the real measurements, and why `dense` is retained

v2.1.0 (master spec §8; originally scheduled new by master spec §8.3). Full detail on
the fixture itself — provenance, extraction/fitting scripts, rebuild instructions,
what is and is not in the committed files — lives in the separate `eiger_paper`
repository, at
[`benchmarking/layouts/README.md`](https://github.com/tobymilnec/eiger_paper/blob/d83f61decfa624cbc079c38f38e3dd480fee322b/benchmarking/layouts/README.md); the fixture
itself is not part of this repository or of the `eiger2` wheel. This document is
the release-facing summary: what the real measurements showed, why `dense` is kept
despite them, and what that means for interpreting `max_cpg_gap_bp`.

## 1. Why a layout fixture exists at all

Every EIGER calibration number before Stage 1d was produced on `dense`: uniformly
spaced CpGs on one synthetic chromosome, one eligible run spanning the whole
simulation. Real WGBS and targeted-capture data look nothing like that. The fixture
(the `benchmarking/layouts/` directory of `eiger_paper`, vendored per master spec
§6.1) provides real hg38
coordinates (WGBS) and a fitted targeted-panel generator (EPIC-seq) so calibration
claims can be checked against the fragmentation structure EIGER actually runs on.

## 2. The real measurements

From [`benchmarking/layouts/VALIDATION.md`](https://github.com/tobymilnec/eiger_paper/blob/d83f61decfa624cbc079c38f38e3dd480fee322b/benchmarking/layouts/VALIDATION.md)
in `eiger_paper`:

| layout | eligible % | run median (edges) | run q90 | max run |
|---|---|---|---|---|
| WGBS chr1 | 92.2% | 3 | 24 | 35,404 |
| WGBS chr7 | 92.1% | 3 | 22 | 15,446 |
| WGBS chr19 | 97.9% | 9 | 103 | 33,051 |
| EPIC-seq chr1 profile | 88.9% | 3 | 15 | — |
| EPIC-seq chr19 profile | 93.6% | 5 | 33 | — |
| `dense` (for contrast) | 100% | **19,999** | 19,999 | 19,999 |

**The median eligible run on real data is 3–5 edges; `dense` is a single run of
19,999.** That gap — three to four orders of magnitude in the quantity that governs
region-level inference — is why the region-level null was never validated against
anything realistic before Stage 1d, and it is the appendix finding the Stage 1e
specification — an unpublished development record retained outside this repository
— explicitly flags as worth writing up as a general result: real
WGBS and real targeted capture have a median eligible run of 3 edges under a standard
300 bp gap rule, against the single 19,999-edge run a uniformly-spaced simulated
layout gives.

## 3. Why `dense` is retained — for continuity only, never for a calibration claim

`dense` is not deleted: it is the layout every pre-Stage-1d benchmark number used, so
keeping it lets a statistic-level regression (does `edge_statistic` still produce the
same values after a refactor?) be checked by direct comparison. Stage 1d-A first
recorded that this must never be extended to a calibration claim; Stage 1d-D §5 made
it a formal decision after measuring why: at `dense`'s normal 20,400-CpG scale, the
region-level null (`region_null_mode="member_step"`) cannot form sub-tracks long
enough to be adequate (`min_subtrack_length` needs far more CpGs than a single
19,999-edge run provides at real `null_min_lag`), so Gate 2 fails on `dense` by
construction, regardless of anything about the null's correctness. Scaling `dense`
up to meet the floor (~10⁵ CpGs) would break the one property that makes it useful —
byte-for-byte continuity with every prior benchmark number.

**Decision (Stage 1d-D §5, unchanged by this release): `dense` is excluded from every
FDR-facing acceptance or release criterion.** It is used for statistic-level
continuity only. A `dense` failure on Gate 1, Gate 2, the edge gate, `s`/resolution,
the overdispersion ratio, or recall is expected and does not block a release; a
`dense` pass on those is not evidence of anything either, because it is never run at
genome scale. The v2.1.0 paper-grid result files state this explicitly per cell,
not only in this document; they are retained with the v2.1.0 paper outputs rather
than committed in `eiger_paper`.

## 4. `max_cpg_gap_bp` sensitivity

The fixture README's own caveat: all-CpG inter-gap on chr1 is q50 = 40
bp, q90 = 256 bp, q99 = 722 bp, so the shipped `max_cpg_gap_bp = 300` default sits
directly on the shoulder of the distribution — run structure is acutely sensitive to
it. Stage 1d-A's `gap_sweep` first measured the run-length side of this
(its provisional write-up, an unpublished development record: median run 0.6 → 378
edges across 100→1000 bp on `wgbs_real`) but could not populate the realised-FDR
column, because
`min_null_regions_for_fdr` refused every candidate at every gap value on both real
layouts — not a defect in the sweep, but the same defect Stage 1e Task 1 fixes.
`gap_sweep()`, in `eiger_paper`'s
[`benchmarking/benchmark_suite.py`](https://github.com/tobymilnec/eiger_paper/blob/d83f61decfa624cbc079c38f38e3dd480fee322b/benchmarking/benchmark_suite.py), now runs at
genome scale with `region_null_mode="member_step"` (the shipped default) by
default; its gap-sweep CSV and the Stage 1e stop report §3 carry the populated
realised-FDR column this release adds. Both are retained with the v2.1.0 paper
outputs and development records rather than committed in `eiger_paper`.

## 5. Caveats that still apply

Unchanged from the fixture README: a contiguous slice is not a
representative sample of a chromosome (take varied offsets or take large slices); the
fixture is one WGBS sample and one panel sample, not population-representative; and
run-length figures count singleton (0-edge) runs, so ad-hoc measurements that drop
them will report a higher median than the figures here.

**New for `region_null_mode="member_step"`:** the layout must additionally clear
`null_min_lag * (min_subtrack_length + window - 1)` CpGs (103,000 at shipped
defaults) for the region-level null to build at all; below that,
`region_null_mode="lag"` is the documented fallback for small inputs (`Eiger2Config.
region_null_mode`'s docstring), not a smaller `min_subtrack_length`.
