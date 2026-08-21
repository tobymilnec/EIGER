# Region-level FDR: estimator, bounds, and a known calibration limitation

`Python/eiger2/core.py` (`_fdr_from_areas_result`, `region_fdr_result`, `region_fdr`),
mirrored in `Python/eiger2/probability.py` (`_tail_area_fdr`, documented in
[`docs/PROBABILITY.md`](PROBABILITY.md)).

This document covers the v2.0.1 region-level FDR estimator: its derivation, the two
defects corrected from v2.0.0, the three distinct quantities it reports, the Garwood
Monte Carlo interval and its assumptions, and the output schema change. It also states,
prominently and first, a real calibration problem found while benchmarking this release.

---

## Known limitation: FDR calibration under missingness, small-n, and high-noise

**This is a release-blocking finding, not a footnote.** The corrected q-value machinery
documented below is verified correct — see "What this finding does and does not
implicate" — but under three of five tested simulation regimes, the **realised**
false discovery rate at the default `target_fdr = 0.05` and `--fdr-decision point`
substantially exceeds the nominal 5% target, in genuinely null (DMR-free) data:

| scenario | realised FDR (nominal target: 5%) |
|---|---|
| clean | ~9–10% |
| nuisance | ~9% |
| high_noise | ~56–70% |
| small_n | ~55–71% |
| missingness | **~99%** |

These numbers come from three mechanically independent measurement paths, all run on
this release's code (branch `fix/region-fdr`, commit `46327e7`):

1. A direct `call_dmrs` run over 5 DMR-free seeds x 5 scenarios (231 candidate regions
   total), using the pipeline's own `significant` column exactly as `eiger2 call`
   computes it.
2. The official `benchmarking/benchmark_suite.py` `negative_control` experiment
   (3 seeds, DMR-free by construction) — an independently generated dataset that
   reproduces the same pattern: clean/nuisance near nominal, high_noise 70%, small_n
   55%, missingness 99%.
3. The official `benchmarking/benchmark_suite.py` `heuristic_sweep` experiment, which
   plants **real** ground-truth DMRs and matches calls against them by region overlap
   rather than by construction being null — critically, this shows `missingness` still
   realises **33.5%** FDR at the pipeline's own default hyperparameters, even with real
   signal present, so the miscalibration is not an artefact of a fully-null test case.

Full data, commands, and per-run numbers are in
`.superpowers/sdd/2026-08-18-stage1-region-fdr/task-11-report.md`.

### What this finding does and does not implicate

**It does not implicate the corrected q-value machinery this document describes.**
The two invariants Stage 1 exists to guarantee both hold with zero violations across
every check run against real data:

- **Monotonicity** (`region_qvalue` non-decreasing as area weakens, within each
  stratum): 0 violations across ~500 checked region-pairs on two independently
  generated datasets.
- **The zero-q vacuity guard** (no exact-zero point estimate without a nonzero
  `region_qvalue_upper_95`): 0 violations across 231 checked regions, 31.2% of which
  had an exact-zero point q-value and a median `region_qvalue_upper_95` of 0.0121.

What the corrected estimator did is **correctly reveal** a pre-existing problem that
was previously invisible: under v2.0.0's forward-cummin defect (see "Why the monotone
step is a reverse scan" below), nearly every candidate region in every scenario was
assigned q ≈ 0 regardless of how weak its evidence actually was, so the null-track
construction's poor behaviour under missingness/small-n/high-noise never surfaced as
an excess of false calls — it was already there, just hidden by a defect that called
almost everything significant unconditionally.

The miscalibration instead points **upstream**, to the discovery statistic's and
null-track construction's behaviour under missingness, small sample size, and high
noise — something Stage 1 is explicitly scoped not to touch ("do not change the
discovery statistic, the threshold rule, or segmentation"). Plausible mechanisms
(not diagnosed further within this scope) include the locality-breaking null failing
to reproduce the observed data's missingness/noise structure closely enough, or the
edge statistic behaving non-exchangeably between observed and null tracks under those
conditions. This is consistent with — and appears to sharpen — a pre-existing weakness
already visible in [`docs/BENCHMARK.md`](BENCHMARK.md)'s v2.0.0-era planted-signal
tables, where `missingness` `region_precision` (0.514–0.684) was already the weakest
scenario tested, before Stage 1's FDR machinery existed at all.

### What to do about it, until this is investigated further

- **Do not rely on `region_qvalue` / `significant` at face value** for data with
  substantial missingness, small sample size (`small_n`-like regimes), or high
  measurement noise. Treat calls from those regimes as a ranked candidate list, not
  as FDR-controlled.
- `clean` and `nuisance` scenarios (~9–10% realised vs. 5% nominal) are closer to
  calibrated and plausibly within simulation noise at the tested scale, but were only
  checked at 3–5 seeds; they have not been confirmed at the spec's intended 100-seed
  scale either.
- `--fdr-decision upper` (deciding on `region_qvalue_upper_95` instead of the point
  estimate) is *more* conservative than `point`, but has not itself been benchmarked
  against these same scenarios as part of this release — it is not presented here as
  a fix, only as an existing, more conservative option.
- Root-causing the null-track/discovery-statistic behaviour under these three regimes,
  and re-running the full spec-scale benchmark grid (100 seeds x 5 scenarios x both
  `null_mode`s x windows {1,4,6,8} x `null_replicates` {20,100}), is the recommended
  next step before relying on this release's FDR control in these regimes.
- This release's benchmarks ran at reduced scale — 3-5 seeds, a single discovery
  statistic, no sweep over `null_mode`, `window`, or `null_replicates` (see
  `results_v201/`) — so it is not only the miscalibration (criterion 3) that is
  confirmed only at this reduced scale: the monotonicity and zero-q vacuity guard
  checks reported above as "0 violations" (gate criteria 1 and 2) are likewise
  confirmed only at the 3-5 seed scale actually run, not at the spec's full
  100-seed design.

---

## 1. Three quantities that must stay distinct

EIGER 2 reports three different things about each region's FDR, and the naming is
deliberately kept apart because conflating them is the most common way to
misrepresent this kind of estimate:

1. **Point estimate** (`region_qvalue`) — expected number of false regions divided by
   observed called regions, at the q-achieving cutoff. May be exactly zero.
2. **Empirical resolution** (`region_qvalue_resolution`) — the change in the point
   estimate caused by one additional pooled-null exceedance. This is nonzero and
   assumption-free: it says nothing about a confidence level, only about how coarse
   the point estimate's grid is at that many null replicates.
3. **Monte Carlo interval** (`region_qvalue_lower_95` / `region_qvalue_upper_95`) —
   uncertainty in the expected null count arising from having only a *finite* number
   of null tracks. This requires a count model (Poisson, via Garwood — see below) and
   the model's assumption must be stated, not left implicit.

A q-value is not a posterior probability. A Monte Carlo interval quantifies finite-null
estimation uncertainty, not biological truth. Neither of the two new bound columns
should be read as "the region is real with probability X."

## 2. Reference tail estimator

Within one length stratum, `s = observed_edges / null_edges` (`null_edges` is the total
summed over replicates; `null_area_sets` are pooled over replicates the same way, so
the two already correspond — this scaling is unchanged from v2.0.0).

For the `j`-th **distinct** observed area cutoff `a_j`, ordered strongest to weakest:

- `R_j` = count of observed regions with area >= `a_j` (the complete tie group at that
  cutoff is included — see "The tie-aware denominator" below).
- `K_j` = count of pooled null regions with area >= `a_j`.
- Point tail FDR: `FDR_j = min(1, s * K_j / R_j)`.
- Resolution: `Δ_j = min(1, s / R_j)` — the FDR's own step size at that denominator.
- Garwood bounds, with `α* = α / m_bonf` (see "The Bonferroni-for-simultaneity
  argument" below):
  - `L_j = ½ · χ²_{2K_j, α*/2}` (defined as `0` when `K_j = 0`)
  - `U_j = ½ · χ²_{2(K_j+1), 1−α*/2}`
  - `FDR^lo_j = min(1, s·L_j/R_j)`, `FDR^hi_j = min(1, s·U_j/R_j)`
- q-values, the reverse cumulative minimum over `j` (see next section):
  - `q_i = min_{j>=i} FDR_j`
  - `q^lo_i = min_{j>=i} FDR^lo_j`
  - `q^hi_i = min_{j>=i} FDR^hi_j`

`region_tail_fdr` in `regions.csv` is `FDR_j` evaluated at the region's **own** cutoff
— not the q-achieving cutoff. `region_qvalue`, both bound columns,
`region_qvalue_resolution`, `null_exceedances_at_cutoff`, and
`region_qvalue_cutoff_rank` are all evaluated at the q-achieving cutoff `j*` instead.
This asymmetry is intentional: `region_tail_fdr` tells you what a threshold set exactly
at this region's area would imply, while the other five columns tell you what q-value
this region actually receives once the reverse scan has run.

## 3. Why the monotone step is a reverse scan

v2.0.0 (`core.py:995`, `probability.py:217`) applied `np.minimum.accumulate(raw)` to
`raw` sorted **strongest-to-weakest**. That is a *forward* cumulative minimum. The
correct q-value requires `q_i = min_{j>=i} FDR_j` — the minimum of the tail FDR over
this cutoff **and every weaker one**, i.e. a *reverse* cumulative minimum:

```python
q = np.minimum.accumulate(raw[::-1])[::-1]
```

Because the strongest region in a stratum almost always has zero null exceedances,
`raw[0] == 0`, and a forward scan propagates that zero to every region weaker than it
— not because those weaker regions have any evidence of their own, but purely because
they come after a strong one in sort order. The reverse scan instead only lets a
*weaker* cutoff's own low FDR propagate backward to stronger cutoffs (which is the
statistically correct direction: if a looser threshold still controls FDR well, a
stricter one controls it at least as well).

**Worked example (spec Appendix A.1):**

```
observed area, descending:   10, 8, 6, 4
pooled null area:             9, 7, 5, 3
observed_edges / null_edges:  1
null exceedance counts K:     0, 1, 2, 3
R:                            1, 2, 3, 4
raw tail FDR:                 0, 1/2, 2/3, 3/4
```

- v2.0.0 (forward cummin) q-values: `[0, 0, 0, 0]` — every region including the two
  weakest reads as a perfect call, because the strongest region's zero exceedance
  count is carried forward to all of them.
- v2.0.1 (reverse cummin) q-values: `[0, 0.5, 2/3, 0.75]` — each region's q-value
  reflects the tail FDR at its own cutoff or a weaker one, correctly increasing as the
  evidence weakens.

This is pinned exactly by `tests/test_eiger2_core.py::test_fdr_from_areas_result_four_distinct_areas`.

## 4. The tie-aware denominator

The second defect: v2.0.0's denominator was the sequential rank `1, 2, 3, ...`, so two
regions with numerically identical area received *different* tail estimates depending
on which one happened to sort first. A threshold placed at area `a` calls **all**
observed regions with area `>= a`, so the estimator must be computed once per
**distinct** observed area, with `R(a) = count(observed_area >= a)` including the
entire tie group, then expanded back to individual rows so every tied region shares
identical output.

**Worked example (spec Appendix A.2):**

```
observed area, descending:   10, 8, 8, 4
pooled null area:             9, 7.5, 5, 3
distinct cutoffs:            10, 8, 4
R (tie group complete):       1, 3, 4
K:                            0, 1, 3
raw tail FDR:                 0, 1/3, 3/4
```

- v2.0.0 (sequential rank, forward cummin) q-values: `[0, 0, 0, 0]`.
- v2.0.1 (tie-aware `R`, reverse cummin) q-values by row: `[0, 1/3, 1/3, 3/4]` — both
  regions with area 8 share the *same* denominator (`R=3`, the size of the complete
  tie group) and therefore the same q-value, resolution, bounds, exceedance count, and
  cutoff rank.

Grouping uses **exact float equality** on the stored area values — no `np.isclose` or
rounding bins — since areas being compared come from the same computation and any
rounding would introduce an arbitrary tie-breaking rule of its own.

## 5. Why Phipson–Smyth's `(b+1)/(B+1)` is declined here

Phipson & Smyth's small-sample correction adds one to both the numerator and
denominator of a permutation **p-value**, to avoid ever reporting an exact zero from a
finite number of permutations. It is a well-known and appropriate correction for that
specific quantity.

EIGER's `region_qvalue` is not a p-value. It is a **scaled expected-null count**:
`s * K_j / R_j`, where `K_j` is a count of null region exceedances and `s` rescales the
null's edge count to the observed genome's edge count. Adding one to `K_j` (the
Phipson–Smyth-style correction) would not correct a small-sample p-value bias here; it
would silently distort the actual estimated count. Instead, the estimation uncertainty
in a small `K_j` is handled honestly and separately, via the Garwood interval (below)
— which quantifies the finite-null uncertainty explicitly, as a confidence interval
around the count, rather than folding an ad hoc correction into the point estimate
itself.

## 6. The Garwood interval and its Poisson assumption

`region_qvalue_lower_95` / `region_qvalue_upper_95` are Garwood exact confidence
bounds for a Poisson count `K_j`, mapped through the same scale-and-normalise
transform as the point estimate:

```
L_j = ½ · χ²_{2K_j, α*/2}          (0 when K_j = 0)
U_j = ½ · χ²_{2(K_j+1), 1−α*/2}
```

This is the standard exact (Clopper–Pearson-style) confidence interval for a Poisson
rate given an observed count, expressed via the chi-squared distribution. **It assumes
the null exceedance count `K_j` is Poisson-distributed** — i.e. that
replicate-to-replicate variance in the exceedance count equals its mean. This is the reason a
zero point estimate (`K_j = 0`) still has a strictly positive upper bound: a Poisson
process with rate zero would never produce any variability, but a Poisson process with
a small positive rate can easily produce zero counts in a finite sample, and the
Garwood upper bound reflects that possibility rather than treating an observed zero as
certainty.

This Poisson assumption is exactly what the overdispersion diagnostic (below) checks,
and it is exactly what can fail — see the top-of-document limitation section for what
happens to calibration in that case.

## 7. The Bonferroni-for-simultaneity argument

The reverse cumulative minimum is applied not only to the point estimate but to the
upper bound: `q^hi_i = min_{j>=i} FDR^hi_j`. This step is only valid on the event that
**all** `U_j` simultaneously bound their respective `FDR_j` — i.e. `U_j >= FDR_j` for
every `j` at once, not merely for each `j` individually at its own confidence level.
Only under that joint event does `min_j U_j >= min_j FDR_j` hold, which is what
licenses reporting `q^hi_i` as a bound on `q_i`.

A single per-cutoff Garwood interval at level `1 − α` only guarantees marginal
coverage for that one cutoff; it says nothing about simultaneous coverage across all
`m` cutoffs a stratum considers. The Bonferroni correction — testing each interval at
`α* = α / m_bonferroni` instead of `α` — is what converts `m` marginal `1 − α`
guarantees into one joint `1 − α` guarantee across all of them, via the union bound.

**This correction must not be removed as an "optimisation."** Doing so would make the
reverse-scan step over the bounds statistically invalid: `q^hi_i` would no longer be a
valid upper bound on `q_i`, even though it would still look like one in the output.

## 8. Per-stratum vs. joint coverage

`m_bonferroni` is computed as the **maximum** number of distinct observed cutoffs
across all length strata (`m_bonferroni = max_s G_s`, where `G_s` is the distinct-cutoff
count within stratum `s`), and the same `m_bonferroni` is then used for every stratum's
Garwood calculation. This is what makes `region_qvalue_upper_95` comparable in
magnitude across strata.

State this explicitly: **coverage is >= 1 − α within each stratum; joint coverage
across `S` strata is only >= 1 − S·α.** The bounds reported here are **not** described
as genome-wide simultaneous across strata — only within-stratum simultaneity across
that stratum's own cutoffs is guaranteed by construction. If genome-wide (all-strata)
simultaneity were required, `α*` would need to be divided by `S` again on top of the
per-stratum Bonferroni correction; this release does not do that, and the diagnostics
recorded (`fdr_interval_bonferroni_m`, `fdr_interval_cutoffs_by_stratum`,
`fdr_interval_family = "per_stratum"`) exist specifically so a reader can see which
scope of simultaneity was actually used.

## 9. The overdispersion caveat

**Under overdispersion, the Garwood interval understates uncertainty.** Plainly: the
Garwood interval's entire validity rests on the Poisson assumption stated in section 6.
If the null exceedance count varies replicate-to-replicate by more than a Poisson
process would predict (`variance > mean`), the true sampling uncertainty is wider than
what the Garwood formula reports, and `region_qvalue_upper_95` becomes an
**underestimate** of the real upper bound — not a conservative one.

`region_fdr_result` checks this directly: for each length stratum, at each distinct
cutoff with positive mean exceedance count across replicates, it computes
`ratio_j = var(K_{.,j}, ddof=1) / mean(K_{.,j})` and takes `median_ratio` across usable
cutoffs (`_overdispersion_diagnostic` in `core.py`, requiring >= 5 replicates and >= 1
usable cutoff; otherwise it reports `"unavailable"` rather than guessing). If the
**largest** `median_ratio` across strata exceeds `1.5`,
`null_count_overdispersion_warning` is appended to
`manifest["warnings"]` and printed to stderr by `eiger2 call`:

```
[eiger2] WARNING: pooled null region counts are overdispersed
         (variance/mean = {ratio:.2f}, Poisson assumes 1.0).
         region_qvalue_upper_95 is a LOWER BOUND on the true uncertainty.
```

**This release does not attempt to widen or narrow the interval using an uncalibrated
dispersion correction.** The diagnostic is reported, not silently corrected for, so
that a user with an overdispersed null knows to treat `region_qvalue_upper_95` as
optimistic rather than trust it as-is.

## 10. Schema: v2.0.0 to v2.0.1

`schema_version` did not exist as a manifest field before this release.
**`schema_version = "2.0"` is recorded retrospectively here** as the name for the
implicit v2.0.0 output format — it was never written to any manifest at the time, so a
v2.0.0-era `regions.csv`/`run_manifest.json` is identified by the **absence** of a
`schema_version` key, not by that key being present with value `"2.0"`. From this
release onward, `manifest["schema_version"]` is written explicitly by every run
(currently `"2.2"`, bumped from `"2.1"` in Stage 1c).

`regions.csv` gains six columns, inserted immediately after `region_qvalue`:

| v2.0.0 column | v2.0.1 column | Notes |
|---|---|---|
| `region_qvalue` | `region_qvalue` | value changes; corrected direction + tie-aware denominator |
| — | `region_tail_fdr` | new: point FDR at the region's own area cutoff |
| — | `region_qvalue_lower_95` | new: simultaneous Garwood lower bound |
| — | `region_qvalue_upper_95` | new: simultaneous Garwood upper bound; nonzero whenever the point estimate is zero |
| — | `region_qvalue_resolution` | new: one scaled null exceedance's worth of change at the q-achieving cutoff |
| — | `null_exceedances_at_cutoff` | new: pooled exceedance count `K` at the q-achieving cutoff (`-1` if unavailable) |
| — | `region_qvalue_cutoff_rank` | new: `R` at the q-achieving cutoff (`-1` if unavailable) |
| `region_evidence_score` | `region_evidence_score` | value changes for extrapolated-tail regions (uses the corrected `_tail_area_fdr`, see `docs/PROBABILITY.md`) |
| `evidence_score_extrapolated` | `evidence_score_extrapolated` | value changes for extrapolated-tail regions, same reason as `region_evidence_score` above |
| `significant` | `significant` | value changes (derived from the corrected q-value; conditionally from the upper bound under `--fdr-decision upper`) |

Spec §0.3.3 names six values that change by design in total; the six rows above cover
four of them (`region_qvalue`, `region_evidence_score`, `evidence_score_extrapolated`,
`significant`). The remaining two, `n_significant` and `n_passing_qvalue`, are not
`regions.csv` columns — they are manifest-level counts (`manifest["n_significant"]`,
`manifest["n_passing_qvalue"]` in `core.py`) — but they change by design for the same
reason: they are derived from the corrected `significant` / `region_qvalue` values.

The following are bit-for-bit invariant between v2.0.0 and v2.0.1 for a default
fixed-window run and are **not** in the table above because they do not change:
`region_id`, `chr`, `start`, `end`, `n_cpgs`, `cpg_index_start`, `cpg_index_end`,
`statistic`, `statistic_mean`, `statistic_min`, `region_area`,
and every `*_descriptive` column. This is pinned by
`tests/test_eiger2_core.py::test_stage1_preserves_the_invariant_column_set` against a
committed v2.0.0 fixture.

As of schema 2.2, `statistic_threshold` is **not** in this invariant set: under
`region_threshold_mode == "support_binned"` (reachable via the shipped default
`support_binned_threshold="auto"` when there is sufficient support dispersion),
`statistic_threshold` becomes the region's mean per-edge threshold rather than a
single fixed scalar, and its value is not comparable across runs with different
`support_binning`.

`confidence_level` (default `0.95`) remains available in the Python API for research
use. If a caller requests a value other than `0.95`, `eiger2 call`'s CSV export refuses
with an explicit error naming the requested level, rather than silently writing bound
columns whose confidence level the fixed schema doesn't record.

---

## 11. The region null: the member-step construction (`region_null_mode`)

### 11.1 Why the observed track has autocorrelation, and why that is geometry, not biology

The observed track's lag-1 autocorrelation is close to +0.5 on every real layout
measured (`STAGE1D_B_STOP_REPORT.md` §4.2: dense/wgbs_real/epic_fitted all ≈ −0.50 in
the *diff from a flat null*, i.e. the observed value itself is ≈ +0.50). This is not a
biological claim about regional coordination — it follows mechanically from how
`edge_statistic()` builds a window: the statistic at edge `e` is the mean over pairs
among CpGs `{e - offset0, ..., e - offset0 + window - 1}`, and the statistic at edge
`e+1` shares `window - 1` of those `window` members. Two overlapping sample means are
correlated because they are built from mostly the same underlying values, regardless of
whether the biology is coordinated at all. **A null construction that does not
reproduce this member overlap is not testing "is there real coordination here" — it is
comparing the data to a null that lacks a structural property the data has for reasons
having nothing to do with the hypothesis being tested.**

### 11.2 Why the shipped (Stage 1b) null lacks it

`null_tracks()`'s per-position value at null "position" `p` averages pairs among
members `{p, p+L, ..., p+(w-1)L}` (`L` the drawn lag, deliberately far larger than the
methylation correlation length). Stepping `p -> p+1` shifts *every* member by one CpG
simultaneously, so position `p`'s member set and position `p+1`'s are disjoint — no
overlap, hence (correctly, by construction) no autocorrelation. This is exactly right
for Stage 1b's purpose (an edge-level marginal that does not inherit the far-apart
pairing's own irrelevant structure) and exactly wrong for a REGION-level null, which
needs the SAME member-overlap property the observed track has in order to be compared
against fairly. Measured: Gate 1 fails on 0/75 instances at every layout
(`STAGE1D_B_STOP_REPORT.md` §4.2); Gate 2 correspondingly finds an inadequate,
near-empty null pool on 75/75 (§4.3) — a null with no autocorrelation forms almost no
runs, so there is almost nothing to compare the observed region-formation rate against.

### 11.3 The member-step construction (C.3)

Fix a lag `L` and a starting offset `i in [0, L)`. Walk a second index `j`: null
"position" `j`'s members are `{i + jL, i + (j+1)L, ..., i + (j+w-1)L}`. Stepping
`j -> j+1` shifts the member set by ONE LAG, not by one CpG, so positions `j` and `j+1`
share `w - 1` of `w` members — the same overlap fraction consecutive OBSERVED windows
share, reproducing the autocorrelation **by construction, not by tuning**. The pair
formula itself is unchanged from `_lag_null_track` (`member_step_lag_null_tracks`
re-slices its existing per-position output at stride `L`; no new pair-assembly code).
Running every offset `i in [0, L)` as an independent sub-track recovers exposure (a
single offset alone gives a short sub-track); the null pool is one array per DRAWN
LAG, sub-tracks concatenated with the eligible mask broken at each boundary so
`segment()` cannot merge two unrelated offsets (`member_step_lag_null_ensemble`).

Measured on the real fixture at genome scale (600,000 CpGs, `STAGE1D_C_STOP_REPORT.md`
§5, `STAGE1D_D_STOP_REPORT.md` §1): Gate 1 passes 45/45; Gate 2 median ratio
1.10–1.12 with an adequate pooled null count on 15/15 instances on both real layouts.

### 11.4 The two gates

**Gate 1 (ACF match).** Null autocorrelation must be within ±0.05 of the observed
track's at lags `1..window`, on DMR-free data (`acf_gate`, `track_autocorrelation`).
This is the gate Stage 1b lacked — it validated the null's marginal and variance but
never re-validated the dependence structure its own fix had just changed
(`EIGER_NULL_CALIBRATION_DIAGNOSIS.md`, corrected banner; master spec §2.2, §10).

**Gate 2 (region-level exchangeability).** `observed_rate` / `null_rate`, both regions
per 10,000 eligible edges; the ratio's median across seeds must be ≤ 3.0, **and** the
pooled null-region count must clear `min_null_regions_for_fdr` (Stage 1c-bis, corrected
to an aggregate check in §12) so the ratio is estimated from something
(`region_exchangeability_gate`). `inf` is
reported explicitly — never coerced to `NaN` or hidden — when the null rate is zero and
the observed rate is not; that was the state on every layout before this stage
(`STAGE1D_B_STOP_REPORT.md` §4.3) and had to stay visible while it was.

Both run alongside the pre-existing edge-level gate (`null_exchangeability`, Stage 1b,
median ≤ 1.25, unchanged) and are recorded, together with `region_null_mode`,
`n_lag_draws`, the drawn lags, and the implied sub-track length range, in
`manifest["gates"]` on every `eiger2 call` run.

### 11.5 `s` and resolution — stated the right way round

`region_fdr_result`'s finite-null resolution is `Delta = s / R`, where
`s = observed_eligible_edges / null_eligible_edges` and `R` is a region's rank among
distinct cutoffs. **Smaller `s` is better** — it means more null volume relative to the
observed track, which is a finer-grained empirical null and therefore a smaller
quantisation step in the reported q-value. Raising the null's volume (more replicates
for `"lag"`; more lag draws, `n_lag_draws`, for `"member_step"`) *improves* resolution;
its cost is runtime, not resolution.

The shipped `"lag"` construction at its `null_replicates=100` default gives
`s ≈ 0.01`. `member_step`'s `n_lag_draws` was originally set to `150` by measuring `s`
at genome scale on both real layouts and raising the draw count until `s ≤ 0.01` held
on the tighter of the two (EPIC-seq): `s = 0.0090` (WGBS chr1), `s = 0.0097` (EPIC-seq)
at `n_lag_draws = 150`. That target predates the realised-FDR measurement described
below (§13): once realised FDR was shown to be substantially conservative relative to
the 5% nominal target, a resolution finer than what a 5% decision needs stopped being
load-bearing. A sensitivity sweep (`n_lag_draws` in {40, 60, 75, 100, 150}, both real
layouts, 3 seeds, genome scale) found `n_lag_draws = 75` gives `s ≈ 0.019` with both
region-level gates still passing and recall/realised FDR unchanged from `150` on every
cell measured, at ~27% lower wall-clock per cell (closer to a 2x cut in the null
construction's own share of that time) — the current default. `Eiger2Config.n_lag_draws`'s
docstring records the full sweep; pass `n_lag_draws=150` explicitly to restore the
finer resolution.

`manifest["gates"]["s"]` and `manifest["gates"]["implied_region_qvalue_resolution_at_R1"]`
(identical values; the latter names what `s` actually means for the top-ranked region)
are recorded on every run so a user can see what their q-values are quantised at
without re-deriving it from the region-exchangeability edge counts.

### 11.6 `region_null_mode="lag"` — retained, not recommended for inference

Kept so the Stage 1d-B/1d-C measurements stay reproducible and the historical
comparison can be regenerated. **Fails both region-level gates on every layout
measured** (§11.2 above) — do not use it for region-level inference. It remains the
only construction Stage 1c's support-conditioned threshold extends to
(`region_null_mode="member_step"`/`"off"` force support binning off — the
per-replicate support reconstruction assumes a per-position member-set formula that
does not describe `"member_step"`'s concatenated sub-tracks near a boundary; a
documented limitation, not a silent miscalibration, per `Eiger2Config.region_null_mode`'s
docstring).

## 12. The adequacy-floor correction: `K=0` under adequate exposure is evidence, not
absence (Stage 1e Task 1, master spec §8)

STOP 1d-D §9 measured recall = 0.000 on both real layouts at genome scale, with the
guard (`min_null_regions_for_fdr`) at its shipped default. The cause was diagnosed
precisely: the member-step null pooled 1,963 regions and was aggregate-adequate
(`region_exchangeability_ratio = 1.625`, `null_count_adequate = True`), but **zero** of
them fell in the `10-19` CpG length stratum — exactly where every planted true region
sat. Stage 1c-bis's per-stratum floor (§8 above discusses the related but distinct
`m_bonferroni` scoping; this is the null-adequacy guard, `min_null_regions_for_fdr`)
refused every one of them.

**Read that plainly: the null, given millions of opportunities, essentially never
produces a region that long. Observing ten of them is about as strong as evidence
gets. The guard responded by refusing to answer.**

Stage 1c-bis's original reasoning conflated two different situations:

- **Thin but populated** — a handful of null regions in a stratum, so a count-based
  ratio estimated from it would be noisy.
- **Empty because the null cannot produce them** — not missing information. Decisive
  information, given the run as a whole is adequately backed by null volume.

The finite-null uncertainty layer this document builds in §1–§7 already handles the
second case correctly and was never the problem: `K=0` at a cutoff gives a point
estimate `q = s·0/R = 0` and a **finite** Garwood upper bound `s·U(K=0)/R` (§6). Neither
depends on how many null regions happened to land in that particular stratum — the
Garwood interval is a function of `K`, `m_bonferroni`, and `scale` (itself a function of
`observed_edges`/`null_edges`, run-level quantities), not of a per-stratum pool size. A
stratum backed by an empty pool and a stratum backed by a huge pool that also happens to
have zero exceedances at the same cutoff get *identical* `qvalue_upper_95` — as they
should, since both report the same `K=0`. The Stage 1c-bis floor short-circuited this
machinery before it could run, discarding a correct, already-implemented answer in favor
of `NaN`.

**The fix moves the adequacy floor off per-stratum realised region counts and onto
aggregate adequacy.** `min_null_regions_for_fdr` (`Eiger2Config`, `region_fdr_result`) is
now checked **once, globally**: pooled null regions summed across every stratum and
every replicate. If the run as a whole fails this, everything is refused exactly as
before — that Stage 1c-bis behaviour is unchanged. If the run passes, **every** stratum
is let through the tail-area estimator, including ones whose own pool is exactly empty.
`diagnostics["insufficient_null_strata"]` is retained for schema (`"2.2"`) compatibility
and is now always `[]`; a stratum whose own pool is empty is instead named, with its
exposure (pooled count, replicate count), in
`diagnostics["zero_exceedance_strata"]` — reported, not refused, so a reader can see the
evidence was real rather than absent.

Two alternatives were considered and rejected:

- **Length-neighbour pooling for an empty stratum.** This reintroduces exactly the
  length bias stratification exists to remove (§1c.6 above), and a cruder version of it
  was removed in the Stage 1c-bis review for causing a non-monotonic refusal ladder
  (§1c-bis's own review, cited in `core.py`). Not needed: `K=0` under adequate aggregate
  exposure is already informative on its own.
- **Relabelling the refusal as "indeterminate" rather than `NaN`.** Better wording for
  the wrong answer. The region should get a q-value, not a more politely-worded refusal.

**This is the third correction recorded in this file** (after the direction/tie-aware
denominator fix in §3–§4 and the region-null autocorrelation fix in §11), and the
pattern is worth naming plainly: **every defect in this sequence came from validating a
null on one moment and not the one the estimator actually depends on.** §3–§4's fix
validated the q-value's monotonicity assumption, not just its point value. §11's fix
validated the region-level null's autocorrelation, not just its marginal distribution.
This fix validated the adequacy guard's unit of analysis — the run as a whole, which is
what `scale` and the Garwood interval are actually computed against — not a per-stratum
count that the estimator never needed subdivided that finely to begin with.

## 13. Calibrated, or conservative? Quantifying the realised-FDR claim (Stage 1f Task 2,
master spec pre-release)

STOP 1e reported realised FDR of exactly `0.0000` in 25 of 26 measured planted-signal
cells (`benchmarking/results_v210/gap_sweep.csv`, 20 cells, plus the dedicated
`missingness`-scenario measurement, 6 cells; `STAGE1D_D_STOP_REPORT.md` §6.4 and
`STAGE1E_STOP_REPORT.md` §2). **That is not, on its own, evidence of calibration at
5%.** With 6–16 significant calls per cell, both "controls FDR at 0.05" and
"substantially conservative" predict roughly zero observed false positives — the two
hypotheses are not distinguishable cell by cell. The question has to be asked of the
**pooled** count, not read off individual cells.

**Pooled across all 26 cells: 293 significant calls, 1 false positive.** (One extra
candidate at `wgbs_real`, seed 1, `max_cpg_gap_bp=100` — §11.4's small-`n` caveat,
not smoothed over.) The realised-FDR point estimate is `1/293 = 0.0034` (0.34%). The
one-sided 95% Clopper–Pearson upper bound on the true FDR given this count is:

```
upper_95 = Beta.ppf(0.95, false_positives + 1, significant_calls - false_positives)
         = Beta.ppf(0.95, 2, 292)
         = 0.0161
```

**The 95% upper bound is 1.6%, comfortably below the 5% nominal target.** For scale:
if the true FDR were exactly 5%, the expected false-positive count at 293 calls would
be `14.65`, and the probability of observing 1 or fewer is `4.9 x 10^-6`
(`scipy.stats.binom.cdf(1, 293, 0.05)`) — essentially impossible. The pooled data
reject "calibrated at exactly 5%" and are consistent with "substantially
conservative", the same direction STOP 1e's own DMR-free calibration numbers already
pointed (`benchmarking/results_v210/calibration_member_step.csv`: Gate 2 ratios
0.6–1.6, comfortably under the 3.0 gate, on every completed cell).

**Say "conservative", not "calibrated".** This is the honest reading of the evidence,
it is the safe direction for a discovery tool (a conservative FDR estimator
under-calls before it over-calls), and it names where power is being left on the
table — a reader comparing this method to one that reports realised FDR nearer 5%
should not conclude EIGER is "better calibrated"; the fair comparison is at matched
realised FDR, which is not what 293-vs-1 data can establish on its own.
Overclaiming calibration on evidence that cannot distinguish "controls FDR" from
"substantially conservative" would be the one thing to undermine an otherwise
careful record — this section exists so the claim actually matches the evidence
behind it. Quantifying exactly how conservative (a power curve against a
"realised FDR near nominal" boundary, not the comfortably-detectable signal grid
`DEFAULT_STRATA` provides) is Stage 1f Task 3's paper-grid work, not resolved here.

---

See also: [`docs/PROBABILITY.md`](PROBABILITY.md) for the identical direction/tie fix
applied to `probability._tail_area_fdr` and its effect on `region_evidence_score`.
