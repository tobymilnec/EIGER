# STOP 1 report — EIGER Stage 1: region FDR correction and finite-null uncertainty

Branch `fix/region-fdr`, worktree `EIGER-wt-fdr`, based on `b8e4a19` (tag `v2.0.0`).
Produced per spec §0.6 at the mandatory hard STOP after Stage 1. **This report halts
work.** No Stage 2 or Stage 3 worktree has been created. Nothing has been merged or
tagged. This branch does not touch the live repository at
`/Users/tobymilne-clark/Documents/Code/Github_beta_Versions/EIGER`, which remains at
`b8e4a19` throughout.

Worktree HEAD at report time: `ba046c91671d75a649bc5a15f036fe2ba43a030d`
`git status --short`: empty (clean)

---

## 1. Change summary

Stage 1 corrects two confirmed defects in EIGER's region-level empirical FDR: a
wrong-direction cumulative-minimum scan that silently propagated the strongest
region's near-zero null-exceedance count to every weaker region, and a
sequential-rank denominator that scored tied-area regions inconsistently. On top of
the fix, it adds finite-null resolution and simultaneous Garwood Monte Carlo
confidence bounds on the FDR estimate (`region_qvalue_lower_95`/`upper_95`), a
per-stratum overdispersion diagnostic against the Garwood interval's Poisson
assumption, a common-Bonferroni-`m` two-pass stratification so bounds are comparable
across length strata, raises the null-replicate default from 20 to 100, adds an
opt-in conservative decision mode (`--fdr-decision upper`) with a vacuity guard that
refuses to silently return zero significant regions, extends and versions the output
schema (six new `regions.csv` columns, `schema_version = "2.1"`), and adds the
required manifest envelope (git commit, checksums, seeds, warnings) from spec §0.5.
`region_fdr`'s public signature is unchanged throughout. A regression test pins every
non-FDR region column (boundaries, areas, descriptive statistics) as byte-identical
to v2.0.0. Benchmarking this release (at reduced scale — see §5, §8) surfaced a real,
independently-confirmed, pre-existing FDR miscalibration under missingness/small-n/
high-noise conditions, upstream of anything Stage 1 touched; per your instruction it
is documented prominently rather than fixed (fixing it is out of Stage 1's scope) and
is the reason this release is not yet recommended for tagging as-is. All work was
executed via 12 dispatched implementer tasks, each independently task-reviewed, plus
a final whole-branch review and one consolidated fix wave; every fix-loop and
adjudication decision is recorded with its reasoning in the execution ledger.

## 2. Files changed

```
$ git diff --stat b8e4a19 ba046c9
```

37 files changed, 7285 insertions(+), 110 deletions(-). By category:

**Source (7 files):**
```
Python/eiger2/__init__.py                8 +-
Python/eiger2/cli.py                   161 +-
Python/eiger2/core.py                  485 +++-
Python/eiger2/manifest_utils.py         68 +  (new)
Python/eiger2/probability.py            30 +-
benchmarking/benchmark_suite.py         32 +-
benchmarking/calibration_study.py       32 +-
```

**Tests (5 files):**
```
tests/fixtures/generate_stage1_invariant_fixture.py   50 +  (new)
tests/fixtures/stage1_invariant_fixture_v200.csv       5 +  (new)
tests/test_eiger2_core.py                            604 ++-
tests/test_manifest_utils.py                          39 +  (new)
tests/test_probability.py                             24 +-
```

**Docs and metadata (7 files):**
```
CITATION.cff        7 +-
QUICKSTART.md       64 +-
README.md          102 +-
docs/BENCHMARK.md    8 +   (staleness banner only)
docs/FDR.md        388 +   (new)
docs/PROBABILITY.md 25 +-
pyproject.toml       2 +-
```

**Benchmark evidence (`results_v201/`, new directory, 336KB, 15 files)** — raw output
from §5/§8's bounded benchmark run, committed for reproducibility.

**Plan/process artifacts (not shipped code):** the implementation plan itself
(`docs/superpowers/plans/2026-08-18-stage1-region-fdr.md`, 2721 lines) is committed
in this worktree per the `writing-plans`/`subagent-driven-development` skill
convention. It is process documentation, not part of the release.

`git diff --check b8e4a19 ba046c9`: **exit 0** (no whitespace errors).

## 3. Mathematical behaviour, before and after — Appendix A cases, actual numbers

Printed directly from `core._fdr_from_areas_result` at HEAD (not asserted-and-hidden
in a test — these are real interpreter output):

**Appendix A.1 — four distinct areas** (`observed = [10, 8, 6, 4]`,
`pooled null = [9, 7, 5, 3]`, `scale = 1.0`):

| | region 1 | region 2 | region 3 | region 4 |
|---|---|---|---|---|
| v2.0.0 (erroneous) `region_qvalue` | 0 | 0 | 0 | 0 |
| v2.0.1 (corrected) `region_qvalue` | 0.0 | 0.5 | 0.6667 | 0.75 |

**Appendix A.2 — tie group** (`observed = [10, 8, 8, 4]`,
`pooled null = [9, 7.5, 5, 3]`, `scale = 1.0`):

| | region 1 | region 2 (tied) | region 3 (tied) | region 4 |
|---|---|---|---|---|
| v2.0.0 (erroneous) `region_qvalue` | 0 | 0 | 0 | 0 |
| v2.0.1 (corrected) `region_qvalue` | 0.0 | 0.3333 | 0.3333 | 0.75 |
| v2.0.1 `region_qvalue_cutoff_rank` | 1 | 3 | 3 | 4 |

Both cases match the spec's Appendix A exactly. The tie case's cutoff rank (`3`, not
`2`) confirms the denominator is taken at the **end** of the tie group, as required.

## 4. Test commands and output

```
$ conda activate eiger && pytest tests -q
........................................................................ [ 82%]
...............                                                          [100%]
87 passed in 10.89s

$ conda activate eiger && pytest old_code/tests -q
...........................                                              [100%]
27 passed in 2.01s

$ conda activate eiger && ruff check Python/eiger2
All checks passed!

$ conda activate eiger && ruff check tests
All checks passed!
```

Test count grew from the Task 1 baseline (48 in `tests/`, unchanged 27 in
`old_code/tests/`) to 87 in `tests/` — every one of spec §1.10's 25 required test
items is present (mapped 1:1 in the final whole-branch review), plus tests added
during rulings and the final fix wave to close weak/tautological assertions found
along the way.

**Environment note for whoever runs this next:** every command above requires the
`eiger` conda environment active (`conda activate eiger`) — several points during
this work hit spurious `ModuleNotFoundError`s (missing `numba`, missing `ruff`) from
shells that weren't actually in that environment. These were consistently confirmed
as environment artifacts, never real regressions, but cost real verification time
each time. Worth a one-line note in `docs/EIGER2_DESIGN.md` or `CONTRIBUTING`-type
doc for future sessions.

## 5. Benchmark commands, output paths, and summary tables

**Scope reduction, stated plainly:** spec §1.11 asks for 100 DMR-free seeds × 5
scenario types × planted-signal grids × both `null_mode` settings × windows
{1,4,6,8} × both `null_replicates` {20,100} — a multi-hour-to-multi-day compute job.
What was actually run, in this session, at bounded scale:

```
$ conda activate eiger && python -m benchmarking.benchmark_suite all \
    --out-dir results_v201/benchmark_suite --seeds 3 \
    --scenarios clean nuisance missingness small_n high_noise
# ~11 min wall-clock, exit 0

$ conda activate eiger && python -m benchmarking.calibration_study all \
    --out-dir results_v201/calibration_study --replicates 4
# ~62s wall-clock, exit 0
```

Plus a supplementary direct `call_dmrs` gate-check (5 seeds × 5 scenarios, 231
regions) → `results_v201/gate_check_regions.csv`.

**Release gate (spec §1.11), assessed against real measured data:**

| # | criterion | result |
|---|---|---|
| 1 | Point q-values monotone in area, every stratum, every run | **HOLDS** — 0 violations across ~500 checked region-pairs, two independent datasets |
| 2 | No exact-zero point estimate lacks nonzero `region_qvalue_upper_95` | **HOLDS** — 0 violations across 231 regions (72 exact-zero, median `upper_95` = 0.0121) |
| 3 | Realised FDR ≤ nominal in DMR-free scenarios, `point` mode | **DOES NOT HOLD** for 3 of 5 scenarios — see table below |
| 4 | Invariant column set byte-identical to v2.0.0 | **HOLDS** — see §7 |
| 5 | Every doc number derived from old q-values regenerated or labelled historical | **HOLDS**, after the final-review fix wave added `docs/BENCHMARK.md`'s staleness banner |

Criteria 1 and 2 are confirmed only at this reduced 3–5 seed scale, not the spec's
100-seed design — documented in `docs/FDR.md`.

**Criterion 3 — the finding, in full:**

| scenario | realised FDR (nominal target: 5%) |
|---|---|
| clean | ~9–10% |
| nuisance | ~9% |
| high_noise | ~56–70% |
| small_n | ~55–71% |
| missingness | **~99%** |

Confirmed via three mechanically independent measurement paths (a direct `call_dmrs`
run; the official `negative_control` experiment; the official `heuristic_sweep`
experiment using ground-truth region-overlap matching, which still shows missingness
at 33.5% realised FDR with real planted signal present, not only in the null case).
Independently re-derived from raw committed CSVs, including bit-level checks, by two
separate reviewers during this session (see §8). **This does not implicate Stage 1's
own FDR-correction code** — monotonicity and the vacuity guard both hold with zero
violations on the same data. It points upstream, to the discovery statistic / null-
track construction under those three conditions, which is explicitly out of Stage
1's scope ("do not change the discovery statistic, the threshold rule, or
segmentation"). Full detail, including the exact commands and per-run numbers, is in
`docs/FDR.md`'s "Known limitation" section and
`.superpowers/sdd/2026-08-18-stage1-region-fdr/task-11-report.md`.

Per your explicit instruction (see §8), this is documented prominently rather than
adapted around, per spec §0.6's "stop and report, do not adapt the method."

## 6. Runtime and peak memory, before and after

Measured on a 200-sample, `dmr_length=20` synthetic input, comparing the
`null_replicates` default change (20 → 100):

| `null_replicates` | wall-clock | regions found |
|---|---|---|
| 20 (old default) | 0.197 s | 7 |
| 100 (new default) | 0.805 s | 7 |

≈4.1× slower at the new default, consistent with the spec's stated expectation.
Peak memory was not separately profiled at this input size (both runs are small
enough that peak RSS is dominated by the interpreter/numpy baseline, not the
algorithm) — the manifest envelope now records `peak_memory_mb` per run going
forward (spec §0.5), so this is available for any future genome-scale run without
further code changes.

## 7. Backward-compatibility assessment against spec §0.3.3

**Byte-identical (unchanged), confirmed by a committed regression test
(`tests/test_eiger2_core.py::test_stage1_preserves_the_invariant_column_set`,
independently re-verified at the bit level via `float.hex()` during task review — see
ledger):** `region_id`, `chr`, `start`, `end`, `n_cpgs`, `cpg_index_start`,
`cpg_index_end`, `statistic`, `statistic_mean`, `statistic_min`,
`statistic_threshold`, `region_area`, and every `*_descriptive` column.

**Changed by design (spec §0.3.3's full list, all six confirmed and now named
consistently in `docs/FDR.md` and `README.md`'s migration note after the final-review
fix wave closed a 3-of-6 documentation gap):** `region_qvalue`,
`region_evidence_score`, `evidence_score_extrapolated`, `significant`,
`n_significant`, `n_passing_qvalue`.

**New columns (not present in v2.0.0, additive only):** `region_tail_fdr`,
`region_qvalue_lower_95`, `region_qvalue_upper_95`, `region_qvalue_resolution`,
`null_exceedances_at_cutoff`, `region_qvalue_cutoff_rank`.

**Schema version:** v2.0.0 output is retroactively `"2.0"` (it never carried this
field, so absence identifies it); v2.0.1 output carries `schema_version = "2.1"`.

## 8. Remaining limitations, and the one failed acceptance criterion

**The Criterion 3 finding (§5) is the primary open item and the reason this is not
yet recommended for tagging.** It was surfaced to you mid-session; you explicitly
chose to finish the plan and document it prominently rather than pause to
investigate immediately. It remains open. Recommended next step (from
`docs/FDR.md` and the Task 11 report): root-cause the null-track/discovery-statistic
behaviour under missingness/small-n/high-noise before relying on `region_qvalue` /
`significant` in those regimes, then re-run the full spec-scale benchmark grid.

**Benchmark scope reduction (§5):** the spec's full 100-seed × full-grid benchmark
was not run — 3–5 seeds, a single discovery statistic, no `null_mode`/window/
replicate sweep. `benchmark_suite.py`'s CLI would need a small extension (it has no
`--null-modes` flag today) to drive that exact grid mechanically.

**One residual, parked gap from the final-review fix wave:** the manifest's
`derived_seeds` field (spec §0.5/§0.3.4) records which RNG-offset *scheme* applied to
a run, not literal per-run seed values (deeper `core.py` plumbing would be needed for
that), and — found during the fix wave's own re-review — currently records that
offset unconditionally rather than gating on whether the corresponding RNG stream
actually ran (e.g. a default `--legacy` invocation with `--fdr-replicates 0` still
claims a permutation-null offset was applied). This affects only manifest provenance
metadata, never FDR computation or region calling; no test depends on the field's
value. Ruled non-load-bearing and parked rather than triggering a second fix wave;
recommended as a fast follow.

**Minor, deferred, non-blocking items** (full detail with file:line references in
the execution ledger): a few small dead-code/comment items inherited verbatim from
the spec's own literal code blocks (e.g. an unused `m` local, since fixed; a kept
`_fdr_from_areas` wrapper the spec explicitly asks to retain); the CLI's
`confidence_level` CSV-export guard has no live trigger path today since no
`--confidence-level` flag exists yet (accepted scope decision, forward-looking); the
`519-sample controlled-access cohort` mentioned in the spec's benchmark section was
not located or available in this environment and so was not included.

**No scenario silently failed and was averaged away.** Every gate result in §5's
table reflects the real measured outcome, including the one that did not hold.

## 9. `git diff --check` and final `git status --short`

```
$ git diff --check b8e4a19 ba046c9
(no output — exit 0)

$ git status --short
(no output — clean)
```

## 10. Evidentiary basis for every claim above

Every number in this report was either printed directly from a live interpreter
session in this worktree (§3, §6), taken from a real command's captured output
(§4, §5, §9), or cross-checked independently by at least one dedicated task/final
reviewer against raw source data rather than trusted from a prior report (§5's
gate-check figures, re-derived at the bit level in one case). No claim here is
stronger than what was actually measured. Full task-by-task detail, every controller
ruling with its reasoning, and every deferred/parked minor finding are preserved in
`.superpowers/sdd/2026-08-18-stage1-region-fdr/progress.md` (the execution ledger)
and the individual task reports in that same directory.

---

## What happens next

Per spec §0.6: this report halts Stage 1. No Stage 2 (`exp/multiscale`) or Stage 3
(`exp/programs`) worktree has been created. Nothing has been merged to a release
branch. Nothing has been tagged. **You tag and upload the v2.0.1 release** — or
decide not to yet, given §5/§8's open finding. The spec's own additional checkpoint
before Stage 2 begins also applies here regardless: rerun the full benchmark suite on
corrected q-values and re-derive any thesis/README figures before comparing a future
MultiScale stage against this baseline.
