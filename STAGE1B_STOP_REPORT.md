# STOP 1b report — EIGER Stage 1b: lag-null window-assembly fix

Branch `fix/region-fdr`, worktree `EIGER-wt-fdr`, continuing directly from Stage 1's
STOP report commit (`6965fd7`). Produced per `EIGER_WP1-3_IMPLEMENTATION_SPEC.md` §0.6,
as directed by `EIGER_STAGE1BCD_IMPLEMENTATION_SPEC.md`'s STOP 1b instruction. **This
report halts work.** Stage 1c has not begun. Nothing has been merged or tagged. The
live repository at `/Users/tobymilne-clark/Documents/Code/Github_beta_Versions/EIGER`
remains untouched throughout, at `b8e4a19`.

Worktree HEAD at report time: `8840078b101e2379ff5873191cc19a71f2e3c23a`
`git status --short`: empty (clean)
`git diff --check 6965fd7 HEAD`: exit 0 (no whitespace errors)

---

## 1. Change summary

Stage 1's own benchmarking (documented in `STAGE1_STOP_REPORT.md`) found that realised
FDR badly exceeds nominal under missingness/small-n/high-noise conditions, even though
the FDR-correction machinery itself was verified correct. A follow-on diagnosis
(`EIGER_NULL_CALIBRATION_DIAGNOSIS.md`) root-caused this to a single indexing defect in
`core._lag_null_track`: the null window was assembled from 9 distinct CpGs at 3 base
positions instead of the 6 pairwise statistics among one true set of `w` CpGs, so the
null's upper-tail shape did not match the observed data's — most severely under
structured missingness, where per-pair measurement precision varies most.

Stage 1b fixes that defect with the exact replacement construction specified, adds a
permanent "null exchangeability" regression gate (a split-half self-consistency check,
both as a manifest field on every real `eiger2 call` and as a standalone benchmark
command) that would have caught the original defect, and reproduces the diagnosis's
before/after tables with fresh measurements in this environment. The fix is a no-op at
`window ∈ {1, 2}` (independently verified to the ULP), touches nothing about the
discovery statistic, threshold rule, or segmentation, and correctly changes
`statistic_threshold`/`region_area`/the region set by design — the revised, narrower
backward-compatibility contract this stage introduces (spec §1) is now pinned by a new
regression test alongside the retained (but no-longer-comparable) Stage 1 fixture.

A go/no-go gate at the start of this stage found the diagnosis's implied
`missing_rate=0.25` did not reproduce the dramatic pre-fix effect as strongly in this
environment; you ruled to use `missing_rate=0.5` for this stage's own severity
demonstrations, leaving the shared benchmark scenario definition untouched. A later
finding showed the fix works cleanly at the milder 0.25 too (see §5). All work was
executed via 6 dispatched tasks, each individually reviewed with unusually rigorous
independent numeric verification, plus a final whole-branch review that found and a
consolidated fix wave that resolved 4 real (non-Critical) issues — a lint regression,
two tests that had quietly lost real feature coverage, and a misleading CLI warning —
without touching any of the core fix's logic.

## 2. Files changed

```
$ git diff --stat 6965fd7 8840078
```

7 files changed, 1922 insertions(+), 42 deletions(-):

```
Python/eiger2/cli.py                                        42 +-
Python/eiger2/core.py                                       127 ++-
benchmarking/benchmark_suite.py                              86 +-
docs/superpowers/plans/2026-08-18-stage1b-lag-null-window-fix.md  1066 ++ (plan, process)
tests/fixtures/generate_stage1b_invariant_fixture.py         102 ++ (new)
tests/fixtures/stage1b_invariant_fixture_v201.npz            new, 14933 bytes
tests/test_eiger2_core.py                                    541 ++-
```

`Python/eiger2/core.py`'s 127 lines cover exactly four things: `_lag_null_track`'s
corrected body, the new `null_exchangeability` function, `null_threshold`'s
backward-compatible keyword-only extension, and the `call_dmrs` call-site change.
`_window_from_lags`, `_pair_statistic`, `_null_eligibility`, `edge_statistic()`, and
`eligible_edges()` are untouched — confirmed both per-task and again in the final
whole-branch review via direct diff inspection.

## 3. Mathematical behaviour, before and after — real numbers

Printed directly from live runs against this worktree (Task 5, independently
spot-checked during this session — not asserted-and-hidden in a test):

**Per-edge exceedance ratio (`p_obs/p_null`, gate ≤ 1.25), all 5 scenarios,
`window=4, correlation, null_replicates=100, seed=0, 20000 CpG, DMR-free`
(missingness at `missing_rate=0.5` per the ruling, demonstrating the defect at its
clearest):**

| scenario | ratio before fix | ratio after fix |
|---|---|---|
| clean | 1.64–1.70 | 1.15 |
| nuisance | 1.63–1.69 | 0.98 |
| **missingness (mr=0.5)** | **~6.19 (diagnosis) / 3.80 (this env., seed 0)** | **1.16** |
| small_n | 2.53–2.66 | 1.22 |
| high_noise | 1.78–2.01 | 1.12 |

**A fresh, additional 5-seed exchangeability run at the milder, shared
`missing_rate=0.25` default (never overridden), run for this report:**

```
$ python -m benchmarking.benchmark_suite exchangeability --seeds 5 \
    --scenarios clean nuisance missingness small_n high_noise
             p_null   p_obs  null_exchangeability_ratio
scenario
clean        0.0050  0.0048                      0.9644
high_noise   0.0051  0.0052                      1.0190
missingness  0.0049  0.0049                      0.9999
nuisance     0.0050  0.0047                      0.9251
small_n      0.0050  0.0048                      0.9697
  worst null_exchangeability_ratio: 1.206 (PASS at gate 1.25)
```

All five scenarios pass comfortably at the shared default too — the fix is not
dependent on the specific `missing_rate=0.5` demonstration value.

**DMR-free significant-region counts, seeds 0/1/2, missingness at mr=0.5:**

| scenario | before | after |
|---|---|---|
| clean | 0,0,0 | 1,0,1 |
| nuisance | 0,0,0 | 0,1,0 |
| **missingness** | **51,56,0** | **2,1,1** |
| small_n | 2,3,1 | 0,2,1 |
| high_noise | 4,1,2 | 3,1,2 |

**Planted-signal FDR/recall, 24 DMRs × 3 seeds pooled (72 total), before vs. after:**

At `missing_rate=0.25` (the value that gives an interpretable comparison — at 0.5 the
signal used for this specific benchmark collapses to recall 0 in *both* code versions
due to a genuine, pre-existing detection-power ceiling for this effect-size mix under
heavy dropout, confirmed via a `missing_rate` sweep showing the cliff between 0.25 and
0.4, unrelated to this fix — see §8):

| scenario | FDR before → after | recall before → after |
|---|---|---|
| clean / nuisance | 0.000 → 0.000 | 1.000 → 1.000 |
| **missingness (mr=0.25)** | **0.614 → 0.250** | **0.708 → 0.667** (−4.1pp, within 5pp tolerance) |
| small_n | 0.053 → 0.000 | 1.000 → 1.000 |
| high_noise | 0.014 → 0.014 | 1.000 → 1.000 |

**The strongest single piece of evidence** for the fix, produced during the final
whole-branch review: the shipped `null_exchangeability` gate, run against the
pre-fix `_lag_null_track` via monkeypatch, independently confirmed to correctly FAIL
the 1.25 gate on all three scenarios tested — clean (1.642 pre-fix → 1.118 post-fix),
missingness at mr=0.5 (3.803 → 0.975), and small_n (2.602 → 1.077) — proof the new
regression gate is real, not ceremonial.

## 4. Test commands and output

```
$ conda activate eiger && pytest tests -q
........................................................................ [ 72%]
...........................                                              [100%]
99 passed in 67.81s

$ conda activate eiger && pytest old_code/tests -q
...........................                                              [100%]
27 passed in 1.85s

$ conda activate eiger && ruff check Python/eiger2
All checks passed!

$ conda activate eiger && ruff check tests
All checks passed!

$ conda activate eiger && ruff check benchmarking
All checks passed!
```

All 99 tests genuinely pass — including, after the final-review fix wave, the two
tests that had quietly lost real feature coverage during this stage's own work
(`test_region_evidence_score_exercises_corrected_tail_area_fdr`, restored by
verifying the `_tail_area_fdr` extrapolation path genuinely fires again at
`min_cpgs=3`; `test_upper_decision_uses_upper_bound_not_point_estimate`, restored
with a freshly-found parameter set where `point`/`upper` decision modes genuinely
diverge, 31 vs 30 significant regions). All of spec §1b.5's 7 required test items map
to real, independently-verified tests: no-op at `w∈{1,2}`, member-set correctness,
`_null_eligibility` offset coupling (initially self-referential, caught and fixed by
the final review to genuinely couple to `_lag_null_track`'s own behaviour), the
exchangeability gate at full 5-scenario × 10-seed coverage, the held-out-null
self-consistency control, the recall-regression test (which itself caught and fixed a
genuine latent bug — see §8), and the slice-bounds assertion up to `w=12`.

## 5. Benchmark commands, output, and the exchangeability gate

The standalone `benchmarking/benchmark_suite.py exchangeability` command (new this
stage, spec §1b.3) was run multiple times during this stage — see §3 for the fresh
5-seed, 5-scenario run produced for this report. `command_all` now runs five
experiments (`negative`, `sweep`, `scaling`, `calibration`, `exchangeability`), all
registered in the parser, dispatch table, and module docstring.

Every real `eiger2 call` run now also carries `null_exchangeability_ratio` in its
manifest, and `command_call` prints a stderr warning above 1.25 with wording corrected
during the final review to accurately state that an elevated ratio is *expected* on
data with genuine regional signal and is only a red flag on DMR-free data — the
original wording was found, and fixed, before this report, to falsely claim results
were unreliable on a realistic sparse-signal run that was in fact 12/13 correct.

## 6. Runtime and peak memory

Not separately re-profiled in this stage — the fix changes null-track construction
arithmetic, not its asymptotic cost (same number of `_lag_statistic` calls per
replicate, same replicate count), so no material runtime/memory change is expected
relative to Stage 1's own measurements (0.197s at 20 replicates, 0.805s at 100, on a
200-sample/`dmr_length=20` input). The manifest envelope (Stage 1, spec §0.5) records
`peak_memory_mb`/`runtime_seconds` on every real run going forward, so this remains
directly measurable without further code changes if a future stage needs it.

## 7. Backward-compatibility assessment against the revised (§1) contract

**Byte-identical to v2.0.0/v2.0.1, confirmed by a new regression test
(`tests/test_eiger2_core.py::test_stage1b_preserves_edge_statistic_and_eligibility`,
pinned against `tests/fixtures/stage1b_invariant_fixture_v201.npz`, independently
verified during the final review via direct diff against the pre-Stage-1 baseline —
zero-line diff for both functions):** the raw `edge_statistic()` and `eligible_edges()`
output.

**Changed by design, per spec §1, and confirmed to have changed (as required):**
`statistic_threshold`, `region_area`, the region set and its boundaries, and
everything derived from them — `region_qvalue` and every FDR column,
`region_evidence_score`, `significant`, and all manifest counts.

**The old Stage 1 invariant fixture
(`tests/fixtures/stage1_invariant_fixture_v200.csv`) is retained, unmodified, per
spec §1's explicit "do not delete" instruction.** Its comparison test is retired to a
smoke check (`test_v200_fixture_is_a_historical_record_not_compared_after_stage_1b`)
that validates the historical file is still readable and structurally intact, rather
than comparing it to live output — since none of its 24 columns survive as a
meaningful invariant under the narrowed contract (every one is either a region
boundary itself, a value this stage explicitly redesigns, or an aggregate computed
over the region's now-different CpG slice). This was a deliberate choice over
`xfail`, reasoned in the implementation: `xfail` would misrepresent a closed
historical record as an open, pending-fix issue.

**`schema_version` stays `"2.1"` throughout this stage** — confirmed at all four
places it's written. Bumping to `"2.2"` is explicitly Stage 1c's job (it adds
columns and changes `statistic_threshold`'s meaning further), not this stage's.

## 8. Remaining limitations, and items to carry into Stage 1c/1e

**From the diagnosis, not yet addressed (expected — this is Stage 1c's job):**
missingness still shows elevated realised FDR with planted signal (0.250 at
`mr=0.25`, against 0.05 nominal) even after this fix. The per-edge marginal is now
correct (ratio ≈1.0), so the residual is a region-level clustering effect under
structured missingness — exactly what Stage 1c's support-conditioned threshold is
designed to address.

**A genuine, pre-existing detection-power ceiling, unrelated to this fix:** at
`missing_rate=0.5`, a specific planted-signal benchmark configuration (24 DMRs at
`DEFAULT_STRATA`'s effect sizes) loses all recoverable signal in *both* the pre-fix
and post-fix code — confirmed via a `missing_rate` sweep showing recall collapses
between 0.25 and 0.4, well before reaching 0.5. Not a fix regression; a real limit on
what this particular signal-strength mix can survive under that much structured
dropout.

**Recommendations from the final whole-branch review, worth carrying forward
explicitly:**
1. **The 1.25 exchangeability gate has less headroom than it first appears.**
   Post-fix `clean` measured as high as 1.169–1.206 across different runs in this
   session. At Stage 1e's full 100-seed × 5-scenario grid (500 draws), a strict
   per-instance 1.25 gate will likely trip on tail seeds from Monte Carlo noise
   alone, even with a correct fix. Decide before that grid runs whether spec
   §1e.2 criterion 1 means a per-instance or a per-scenario-aggregate (mean/median,
   or a bootstrap CI) pass — so it doesn't look like moving goalposts afterward.
2. **The corrected null yields far fewer null regions than the pre-fix defective
   one.** One test needed `min_cpgs` lowered from 4 to 3 (a test-local override, not
   a production default change) before the posterior-extrapolation path would fire
   at all, even at very high `null_replicates`. Stage 1c's support binning
   (requiring ≥`min_null_values` per bin) and its §1c.6 per-bin null-region-rate gate
   should budget `null_replicates` with this in mind — the "fewer than 50 null
   regions per joint stratum" fallback in §1c.6 may fire more often than Stage 1's
   experience would suggest.
3. **`docs/EIGER2_DESIGN.md` contains a now-stale measured claim** ("the lag null
   restores calibration: 0.4–0.6% of DMR-free edges pass") that this stage's fix
   changes. Not touched in this stage (correctly deferred — Stage 1e regenerates
   docs), but should be added explicitly to Stage 1e's §1e.3 documentation list so
   it isn't missed alongside `docs/FDR.md`/`docs/PROBABILITY.md`/`docs/BENCHMARK.md`.
4. **Process:** `.github/workflows/tests.yml` currently lints only `Python/eiger2`
   and has no failure guard on the `pytest` step. The one defect this stage's final
   review caught that no individual task review did was a lint regression whose
   *cause* spanned commits — exactly the class of finding a per-task review scope
   cannot see. Extending CI's lint scope to `tests`/`benchmarking` and removing any
   `|| true` guard would make this class of finding free going forward. Not applied
   in this stage (explicitly out of scope for the fix wave — a CI/process decision,
   not an in-branch code fix).

**Full task-by-task detail, every controller ruling with its reasoning, and every
finding from both the per-task reviews and the final whole-branch review are
preserved in `.superpowers/sdd/2026-08-18-stage1b-lag-null-window-fix/progress.md`
(the execution ledger) and the individual task/fix-wave reports in that same
directory.**

## 9. `git diff --check` and final `git status --short`

```
$ git diff --check 6965fd7 8840078
(no output — exit 0)

$ git status --short
(no output — clean)
```

## 10. Evidentiary basis for every claim above

Every number in this report was either printed directly from a live run in this
worktree during this session (§3, §5), taken from a real command's captured output
(§4, §9), or independently re-verified by at least one dedicated task reviewer or the
final whole-branch reviewer against live code execution rather than trusted from a
prior report — including, notably, the final review's own novel check (monkeypatching
the pre-fix code through the shipped gate to confirm it correctly fails) and the
scoped re-review's independent reproduction of every fix-wave numeric claim to exact
matching values. No claim here is stronger than what was actually measured.

---

## What happens next

Per spec: this report halts Stage 1b. Stage 1c has not begun. Nothing has been merged
to a release branch. Nothing has been tagged. All work continues on `fix/region-fdr`
in this same worktree, per the spec's explicit "do not open new worktrees" instruction
— Stage 1c, when you're ready, continues here.
