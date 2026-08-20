# STOP 1c report — EIGER Stage 1c: support-conditioned threshold

Branch `fix/region-fdr`, worktree `EIGER-wt-fdr`, continuing directly from Stage 1b's
STOP report commit (`15b3bd7`). Produced per `EIGER_WP1-3_IMPLEMENTATION_SPEC.md` §0.6,
as directed by `EIGER_STAGE1BCD_IMPLEMENTATION_SPEC.md`'s STOP 1c instruction. **This
report halts work. Stage 1d has not begun.** Nothing has been merged or tagged. The
live repository at `/Users/tobymilne-clark/Documents/Code/Github_beta_Versions/EIGER`
remains untouched throughout, at `b8e4a19`.

Worktree HEAD at report time: `b7ae9cad260b16d3d65083351bf07611eb489afc`
`git status --short`: empty (clean)
`git diff --check 15b3bd7 HEAD`: exit 0 (no whitespace errors)

**Headline result, stated plainly up front because it is the single most important
fact in this report: Stage 1c's own target (spec §1c.8 — realised FDR under
missingness falling from 0.250 to ≤0.05) was NOT met.** Realised FDR did not fall; it
rose slightly (0.250 → 0.273 on the spec's own 3-seed pool; 0.421 → 0.425 pooled over
10 seeds). Recall was unchanged between before/after (0.667 / 0.608 in both arms), so
this is explicitly **not** the "target met only via recall collapse" failure mode spec
names — it is a plain, clean miss on the FDR criterion itself. Every task in this
stage was built correctly, reviewed independently (in several cases to a level of
scrutiny — hand-arithmetic verification, 400-trial differential fuzzing, byte-level
A/B against pre-change code — well beyond Stage 1/1b's already-high bar), and the
final measurement is itself independently reproduced to the fifth decimal place. The
negative result is real, not a measurement or implementation defect. §8 gives the
root cause and the evidence for it in full.

---

## 1. Change summary

Stage 1b's fix (the lag-null window-assembly correction) fixed the per-*edge* marginal
exceedance rate but left realised FDR under structured missingness at 0.250 against a
0.05 target — the diagnosis attributed this to a region-level clustering effect: under
uneven local measurement support, observed exceedances cluster more tightly than null
ones. Stage 1c's stated fix was a threshold conditioned on local support (samples
observed at every CpG in the window), computed identically for observed and null data
so the two stay exchangeable, auto-enabled when support varies enough to matter, with
a mandatory region-level exposure diagnostic and a documented escalation path if
per-edge conditioning alone didn't fix region-level clustering.

All of that was built, and built correctly:

- A shared `_window_support` function computes support identically for observed
  windows and null member-sets from one underlying finite-mask array (Task 3).
- `segment()`/`region_area()` were generalized to accept a per-edge threshold array,
  with the scalar path kept bit-identical — independently verified at every
  subsequent task, and finally verified end-to-end by a hand-reconstructed Stage-1b
  scalar pipeline compared bit-for-bit against `support_binned_threshold="off"`
  output (Task 4, strengthened in Task 11).
- Quantile bin edges are derived from observed data only, with an upward-merge rule
  for underpopulated bins and a collapse fallback to the unconditioned path (Task 5).
- The full per-bin threshold pipeline was wired into `call_dmrs`, including a
  self-caught, genuinely serious design correction: each null replicate gets its own
  support-derived threshold array (not the observed track's), because null replicates
  have different lengths/geometries per replicate — verified by brute-forcing every
  replicate's support against `_lag_null_track`'s documented member set (Task 6).
- The auto-enable dispersion cut was calibrated from 50 real runs across the 5
  standard scenarios, not guessed (Task 8) — it stayed at its provisional value of
  `0.5`, now with a measured, non-overlapping separating gap of width `1.04` behind
  it (§3).
- The mandatory region-level exposure diagnostic (`region_rate_by_support_bin`) was
  built and run for real on DMR-free negative-control data (Task 9). **It fired**:
  every evaluable support bin showed an observed/null region-formation ratio of
  283×–811×, one bin infinite, confirmed not a small-count artefact (§3).
- Per your explicit ruling to implement the spec-prescribed response, the §1c.6
  escalation — stratifying `region_fdr_result`'s null-region pooling by support bin,
  with a fallback to length-only pooling when a joint stratum has fewer than 50
  pooled null regions — was built as a new task inserted mid-stage (Task 10),
  correctly and with unusually extensive independent verification (a 400-trial
  differential fuzz test against an independently-written reference implementation,
  zero mismatches; §3, §8).
- The real §1c.8 target validation (Task 11) then measured, directly, whether all of
  the above actually closed the gap. **It did not** — see the headline above and §3/§8
  for the full evidence and root cause.

Two significant, uncertain findings were surfaced to you mid-stage rather than
resolved unilaterally, matching the practice established in Stages 1 and 1b: (1)
whether to implement the spec's escalation once Task 9's gate fired (you ruled: yes,
implement it as specified); (2) whether to proceed to the real validation once Task
10's implementer flagged that the escalation was likely inert on the triggering data
(you ruled: proceed, let the real numbers decide). Both rulings are reflected in the
execution above.

All work was executed via 10 dispatched implementation tasks (plus one inserted
mid-stage) matching the plan's own decomposition, each individually task-reviewed —
several on the most capable available model given the statistical stakes — plus a
final whole-branch review across all 12 commits and a consolidated fix wave that
resolved 4 Important and 3 Minor documentation/consistency findings, none of them
correctness defects.

## 2. Files changed

```
$ git diff --stat 15b3bd7 b7ae9ca
```

9 files changed, 3,461 insertions(+), 41 deletions(-):

```
Python/eiger2/cli.py                                                  7 +
Python/eiger2/core.py                                               650 +++++++-
QUICKSTART.md                                                       14 +
benchmarking/benchmark_suite.py                                    118 +-
benchmarking/calibration_study.py                                    2 +-
docs/FDR.md                                                          13 +-
docs/superpowers/plans/2026-08-19-stage1c-...-threshold.md        1612 ++++++++++++++++++++ (plan, process)
tests/test_eiger2_core.py                                         1081 ++++++++++++-
tests/test_probability.py                                            5 +-
```

`Python/eiger2/core.py`'s 650 lines cover: the shared support function and its two
helpers (bin-edge derivation, bin assignment); `segment()`/`region_area()`'s
generalization to array thresholds; three new `Eiger2Config` fields and their
validation; `null_tracks`'s 2-tuple→3-tuple signature change (lag threading) and
every call site fix that required (6 sites total, 2 more than the pre-flight scan
found); the full per-bin threshold pipeline in `call_dmrs`; three new `_REGION_COLUMNS`
entries/behavior changes; the `region_rate_by_support_bin` diagnostic; and
`region_fdr_result`'s joint (length × support) stratification with its fallback rule.

## 3. Mathematical behaviour — real numbers

### 3a. Support-dispersion calibration (Task 8)

50 real runs (5 scenarios × 10 seeds), `support_binned_threshold="off"` so the
detector runs without binning activating:

| scenario | min | median | max |
|---|---|---|---|
| clean | 0.000 | 0.000 | 0.000 |
| nuisance | 0.000 | 0.000 | 0.000 |
| **missingness** | **1.042** | **1.083** | **1.125** |
| small_n | 0.000 | 0.000 | 0.000 |
| high_noise | 0.000 | 0.000 | 0.000 |

Clean, non-overlapping separation — the four non-missingness scenarios give
`support_dispersion == 0.0` at every one of 40 seeds tested (structurally exact:
`missing_rate=0.0` means every window has full support, so the 90th/10th percentile
spread is identically zero). The cut stayed at `0.5`, its Task-6 provisional value,
now empirically grounded near the midpoint of a `(0, 1.042)` gap. Independently
reproduced by the task reviewer at unseen seeds (100–104).

### 3b. The region-level exposure diagnostic — the gate that fired (Task 9)

DMR-free negative-control data, `missingness`/`small_n` scenarios, 20 seeds,
`null_replicates=500`, `support_binned_threshold="on"`, pooled across seeds:

| scenario | bin | obs regions | obs eligible edges | null regions | null eligible edges | obs rate (/10k) | null rate (/10k) | ratio |
|---|---|---|---|---|---|---|---|---|
| missingness | 0 | 5 | 52,392 | 3 | 11,200,409 | 0.954 | 0.003 | **356.3×** |
| missingness | 1 | 9 | 53,814 | 0 | 11,494,060 | 1.672 | 0.000 | **∞** |
| missingness | 2 | 4 | 55,880 | 3 | 11,867,659 | 0.716 | 0.003 | **283.2×** |
| missingness | 3 | 3 | 57,903 | 1 | 12,409,828 | 0.518 | 0.001 | **643.0×** |
| small_n | 0 | 19 | 239,988 | 5 | 51,236,016 | 0.792 | 0.001 | **811.3×** |

Every evaluable bin's ratio is far past the 2.0 gate — the smallest, 283×, is two
orders of magnitude past it. Not a small-count artefact: pooled null exposure is
11–51 million eligible edges per bin, against which 0–5 null regions formed.

### 3c. The escalation's own numeric correctness (Task 10)

Independently verified three ways (hand-trace, an independently-written reference
implementation, and a 400-trial differential fuzz test against the committed code —
zero mismatches). Worked example: length stratum `"6-9"`, bin 0 well-populated (60
pooled null regions, stays joint), bin 1 thin (3 pooled, falls back). The joint
stratum's null count excludes bin 1 exactly (`27`, not the length-only `30`); the
fallback stratum's null count equals the true union of both bins exactly (`39`,
excluding both plausible wrong answers `36` and `3`). Off-mode bit-identity confirmed
end-to-end via a byte-verified A/B against pre-change code across 6 cases (3
scenarios × {off, auto} × `compute_posterior=True`) — every region column, the full
`fdr` diagnostics dict, and the posterior dict identical.

### 3d. The §1c.8 target validation — the headline result (Task 11, independently reproduced to the fifth decimal place)

Pooled over Stage 1b's own 3-seed comparison, `null_replicates=100`,
`fdr_decision="point"` (shipped default), planted signal:

| arm | FDR | recall |
|---|---|---|
| `off` (reproduces Stage 1b exactly) | 0.250 | 0.667 |
| **`auto` (Stage 1c shipped default)** | **0.273** | **0.667** |
| target | ≤ 0.05 | ≥ 0.617 |

10-seed pool: `off` 0.421/0.608, `auto` 0.425/0.608. FDR did not fall in either
pooling — it rose slightly. Recall is identical between arms in every pool, ruling
out the recall-collapse failure mode explicitly.

**Root cause, confirmed independently by both the implementer and a skeptical
reviewer who reproduced every number from scratch:** at shipped defaults, the
lag-null forms **zero regions per replicate** in effectively every validation run
(measured `null_regions_per_replicate == 0.0` in all 20 runs at `null_replicates=100`,
and 0–2 regions *total* across 400 replicates when quadrupled — more replicates buy
essentially nothing). Consequently `region_qvalue` comes out **exactly `0.0`** for
**100% of called regions** in every run, in both `fdr_decision` modes — `upper` does
**not** protect once ≥2 regions are called (measured range `[0.00191, 0.04581]`, never
1.0, correcting an earlier assumption). Every candidate region is accepted; Task 10's
escalation, though correct, cannot manufacture null regions that don't exist by
redistributing an already-empty pool.

**This zero-null-region count is the arithmetically expected value, not a symptom of
a bug**, independently re-derived by the Task 11 reviewer: at `null_quantile=0.995`
(per-edge exceedance probability `p=0.005`) and `min_cpgs=4` (3 consecutive
above-threshold edges required) over ~10⁴ eligible edges per replicate,
`E[null regions/replicate] ≈ 10⁴ × 0.005³ ≈ 0.00125` — a near-exact match to what was
measured. **The lag-null (Stage 1b's own fix) is working exactly as designed, and
that is the problem**: it restores independence between adjacent edges; independent
edges essentially never form runs of three at a 0.995 quantile. The observed track
forms regions because methylation is spatially autocorrelated. No achievable
`null_replicates` fixes this — reaching the escalation's own 50-null-region joint
threshold would need `B ≈ 40,000`, and the FDR estimator's `scale ∝ 1/B` shrinks
proportionally, so the point estimate never becomes informative at any replicate
count a real user would run.

Supplementary finding: the benchmark is strongly bimodal across seeds — 4 of 10 seeds
show near-zero recall with per-bin thresholds 3–5× higher than the 6 working seeds (a
discrete jump, not a gradient). Stage 1b's own headline 0.250 figure was produced
entirely by one such seed.

## 4. Test commands and output

```
$ conda activate eiger && pytest tests -q
........................................................................ [ 52%]
.................................................................        [100%]
137 passed in 104.76s

$ conda activate eiger && ruff check Python/eiger2
All checks passed!

$ conda activate eiger && ruff check tests
All checks passed!

$ conda activate eiger && ruff check benchmarking
All checks passed!
```

137/137 passing throughout (99 at Stage 1b's close → 137 here: 38 new tests across
this stage). One test — `test_support_stratified_fdr_pins_the_known_zero_qvalue_state_
under_missingness` — is a **deliberate pin of the pathological state described in
§3d**, not a normal correctness assertion; it is clearly marked
(`# KNOWN-BAD STATE PIN — see task-11-report.md`) and is expected, by design, to go
red the day the underlying null-region-formation problem is actually fixed. Whoever
picks up that fix should update it then, not relax it now.

## 5. Benchmark commands and output

`benchmarking/benchmark_suite.py` gained a sixth experiment this stage,
`region_rate` (`command_region_rate`), producing the Task 9 table in §3b and writing
`region_rate_by_bin.csv`; it is wired into `build_parser`, `main`'s dispatch table,
and `command_all`. `command_exchangeability`'s summary was changed per your explicit
ruling to gate on the per-scenario median across seeds (≤1.25) with per-instance
outliers reported, not gated (Task 1) — applied to already-shipped Stage 1b code as a
small prerequisite update, and re-used by this stage's own multi-seed exchangeability
checks.

## 6. Runtime and peak memory

Not separately re-profiled as a dedicated exercise. One material note the final
review flagged: `region_rate`'s default scale (`n_cpg_target=20,000`, i.e. 25× the
matrix size most other experiments use, required to get any measurable null-region
signal at all — see §3b/§8) means `command_all` is now dominated by this one
experiment (50 full `call_dmrs` runs at that scale). Not fixed in this stage (a
one-sentence documentation note was the recommended fix, not applied — carrying it
into the minor-findings list below).

## 7. Backward-compatibility assessment

**`schema_version` bumped to `"2.2"` this stage, at both write sites in `core.py`
(confirmed) — this was this stage's job per spec, correctly not done in Stage 1b.**

**`statistic_threshold`'s meaning changed by design, per spec §1c.5**: it is now the
region's mean per-edge threshold when `region_threshold_mode == "support_binned"`,
and the unchanged scalar otherwise. This was caught as an inconsistency mid-stage (an
independent reviewer found the column diverged from the actually-applied per-bin
thresholds on a real run, `0.7143` reported vs. `[0.7308, 0.7236, 0.7001, 0.7079]`
applied) and fixed before this stage closed; `docs/FDR.md`'s claim that this column
was bit-for-bit invariant has now been corrected to match (final-review fix wave).

**Two genuinely new columns added to `_REGION_COLUMNS`**: `region_support_median`
(median per-edge support across the region's edges, populated always, independent of
whether binning fired) and `region_threshold_mode` (`"single"` or `"support_binned"`).

**The scalar-threshold path stays bit-identical when `support_binned_threshold="off"`
— the single most heavily re-tested invariant across this entire stage.** Verified at
every task that could plausibly disturb it (Tasks 4, 6, 7, 10, 11), including a final
hand-built Stage-1b-pipeline reconstruction compared bit-for-bit against real
`call_dmrs` output (replacing an earlier, weaker tautological assertion an
independent reviewer caught), and a byte-level A/B against pre-Task-10 code across 6
scenario/mode combinations.

**`region_fdr_result` gained two new optional parameters** (`observed_support_bins`,
`null_support_bin_sets`), both defaulting to `None`. Every pre-existing caller
(`region_fdr`, `qvalue_result_from_track`'s two call sites) passes neither and is
therefore behaviourally unchanged — confirmed by grep and by direct reading of all
four production call sites, twice, by two independent reviewers.

## 8. Remaining limitations, known deferred findings, and items to carry forward

**The central limitation is the headline result itself (§3d): Stage 1c's target was
not met, and the root cause is now well understood** — the lag-null construction
(Stage 1b's fix) restores edge-level independence so effectively that it essentially
never forms the multi-edge "regions" needed to calibrate a *region-level* FDR
estimator, at any achievable replicate count. This is not fixable by anything within
Stage 1c's own scope (per-edge threshold conditioning, or FDR-layer stratification of
an empty null pool) — it requires either:

1. A null construction that preserves spatial autocorrelation while still breaking
   the specific local coordination that made the original (pre-Stage-1b) null
   over-permissive — plausibly Stage 1d's territory (permutation-null
   anti-conservatism at `window > 1` is the next spec stage; whether it shares this
   root cause is an open question, not yet diagnosed), or
2. `region_fdr_result`'s empty-null-pool path being changed to refuse a `qvalue=0.0`
   result when it has zero evidence (i.e., report "cannot conclude" rather than
   "certainly significant") — this is a defensible, narrower fix that wouldn't
   require touching the null construction at all, but was explicitly out of scope
   for every task in this stage (each implementer was instructed not to redesign it
   speculatively) and needs its own diagnosis/plan.

Recommend **not** proceeding directly into Stage 1d without first deciding which of
these two paths (or both) to pursue, since Stage 1d's own premise (permutation-null
anti-conservatism) may or may not be entangled with this same underlying issue.

**Three further findings from Task 10's independent review, all confirmed still
present and unaddressed (all pre-existing spec-level design choices or accepted
scope limits, not implementation bugs), carried forward as known limitations:**

1. `region_fdr_result`'s exposure `scale` is global, not per-support-bin, even under
   joint stratification — **a literal deviation from spec §1c.6's own wording**
   ("run `region_fdr_result` per support bin **with that bin's own `observed_edges`
   and `null_edges`**"), not merely an unimplemented refinement. Measured up to ~8%
   anti-conservative bias on a real run where the joint path was reachable. Currently
   low-risk because the joint path is essentially inert at this stage's validation
   scale (§3d) — but this reasoning is scale-dependent: a real dataset, orders of
   magnitude larger, would plausibly reach the joint path and re-expose this bias.
2. Joint stratification shrinks `m_bonferroni` (splitting a length stratum into up
   to `n_bins` pieces reduces the per-stratum-max distinct-cutoff count), loosening
   `qvalue_upper`. Verified end-to-end on a real run (`m_bonferroni` 22→12, all 32
   regions' upper bounds changed) — net effect was more conservative overall on that
   run (pooling dominated), but the two effects point opposite ways and only one is
   documented anywhere in the code.
3. Fallback strata mix an all-bins-pooled null count against a fallen-back-only
   observed count — proven safe (always conservative, never anti-conservative:
   `K_union/R_fallback ≥ K_union/R_all`) but a real, undocumented magnitude effect
   (2× in a worked example).

**A residual test-instrumentation gap** (spec §1c.7 item 8): the shipped
exchangeability-ratio test measures the ratio computed *before* the support-binning
branch runs, so it establishes that a real binning-on run still passes the Stage 1b
gate, but not whether the binned per-bin thresholds are themselves exchangeable — no
per-bin exchangeability instrumentation exists yet to measure that directly. Spec
item 8 is literally satisfied; the arguably-intended finer-grained quantity isn't
measurable with current instrumentation.

**Bounded validation scale, as in every prior stage's precedent**: this stage's
calibration (Task 8) and validation (Task 9, Task 11) all ran at bounded scale (tens
of seeds, not the full spec-mandated 100-seed × 5-scenario grid), which is Stage 1e's
job. Given the headline result, re-running the full grid on the current code is
unlikely to change the qualitative conclusion (the root-cause arithmetic in §3d does
not depend on seed count), but Stage 1e's grid remains the authoritative source for
any final calibration numbers once a fix to the root cause is decided and built.

**8 of 11 minor findings from the final whole-branch review were deliberately left
unfixed** (the review itself framed these as optional, not blocking): a couple of
edge-case notes (boundary-edge support sentinel, a diagnostic-label edge case), a
missing CLI `--help` string (carried verbatim from the original spec's brief), a
docstring pointing at a not-yet-written `docs/FDR.md` section, a coverage gap around
non-default `support_bins` values, an imprecise docstring about which direction bin
merging happens, the `command_all` runtime note (§6), and a latent test-fragility
note about a monkeypatch's positional-argument naming. None are correctness risks;
full detail is in `.superpowers/sdd/2026-08-19-stage1c-support-conditioned-threshold/
final-review.md`.

**Full task-by-task detail, every controller ruling with its reasoning, and every
finding from every per-task review, the final whole-branch review, and the scoped
re-review are preserved in
`.superpowers/sdd/2026-08-19-stage1c-support-conditioned-threshold/progress.md`
(the execution ledger) and the individual task/review reports in that same
directory.** In particular, the ledger records two places where the controller's own
preflight work was independently found to be wrong and corrected during execution —
a mis-assessed "safe" call site in the `null_tracks` signature-change preflight scan,
and an incomplete pipeline design in the Task 6 brief (reusing the observed track's
threshold array for every null replicate, which the implementer caught and fixed) —
both left in the ledger un-sanitized, per this project's practice of recording real
mistakes rather than only clean outcomes.

## 9. `git diff --check` and final `git status --short`

```
$ git diff --check 15b3bd7 b7ae9ca
(no output — exit 0)

$ git status --short
(no output — clean)
```

## 10. Evidentiary basis for every claim above

Every number in this report was either printed directly from a live run during this
stage (§3), taken from a real command's captured output (§4), or independently
re-verified — in most cases by more than one reviewer, and for the headline §3d
result, by a reviewer who rebuilt the entire benchmark methodology from scratch
without reference to the implementer's own script and reproduced every reported
number, including several to five decimal places, exactly. The single most
significant finding in this report (§3d/§8's root-cause diagnosis) was independently
re-derived arithmetically — not just empirically observed — by a reviewer who was
explicitly instructed to be skeptical of a dramatic, surprising, negative result
rather than accept it because it was well-argued. No claim in this report is stronger
than what was actually measured.

---

## What happens next

Per spec: this report halts Stage 1c. **Stage 1d has not begun.** Nothing has been
merged to a release branch. Nothing has been tagged. All work continues on
`fix/region-fdr` in this same worktree, per the spec's explicit "do not open new
worktrees" instruction.

The target this stage set out to hit was not met, and the reason is now understood in
enough depth to make a real decision rather than another guess: the region-level FDR
apparatus this whole effort has been built around depends on the null forming
"regions" the same way the observed data does, and at present it structurally does
not, by the same design choice (lag-null independence) that fixed Stage 1b's original
defect. Whether Stage 1d's separate premise (permutation-null anti-conservatism)
shares this root cause, and whether the right fix is a different null construction, a
change to how an empty null pool is reported, or both, is a decision for you before
further implementation work begins.
