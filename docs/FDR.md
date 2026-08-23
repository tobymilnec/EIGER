# Region-level FDR: estimator, bounds, and a known calibration limitation

`Python/eiger2/core.py` (`_fdr_from_areas_result`, `region_fdr_result`, `region_fdr`),
mirrored in `Python/eiger2/probability.py` (`_tail_area_fdr`, documented in
[`docs/PROBABILITY.md`](PROBABILITY.md)).

This document covers the empirical region-FDR estimator, the member-step region null,
the per-length opportunity guard, and the current replicate-cluster uncertainty band.
Sections describing earlier Poisson/Garwood-only releases are historical where noted.

> **Current uncertainty method (Stage 2d).** The empirical pooled exceedance count and
> monotone q-value construction are retained, with an opportunity-specific scale for
> each ordinary length stratum. The former Poisson-only Garwood interval is superseded by one fixed
> replicate-cluster quasi-Poisson interval. One independently drawn genome-wide
> member-step realisation is one cluster; regions inside it are never treated as
> independent. Exposure-adjusted Pearson dispersion widens a scaled-count chi-square
> interval, and Bonferroni correction makes the reported cutoff curve simultaneous
> within a stratum. This is an approximate, model-based 95% band, conditional on
> independent null realisations and `Var(K_b)=phi*E(K_b)`. It does not correct a
> systematically non-exchangeable null. Negative-binomial simulation coverage and
> DMR-free/signal decision regressions are permanent tests; broader publication-scale
> calibration remains required.

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
- **The zero-q uncertainty invariant** (no exact-zero point estimate without a nonzero
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
  confirmed only at this reduced scale: the monotonicity and zero-q uncertainty
  invariant
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
3. **Replicate-cluster interval** (`region_qvalue_lower_95` / `region_qvalue_upper_95`) —
   uncertainty in the expected null count arising from having only a *finite* number
   of null tracks. This uses a quasi-Poisson count model over independent genome-wide
   null realisations; its assumptions are stated below.

A q-value is not a posterior probability. A Monte Carlo interval quantifies finite-null
estimation uncertainty, not biological truth. Neither of the two new bound columns
should be read as "the region is real with probability X."

## 2. Reference tail estimator

Within length stratum `s`, let `O_s` and `N_s` be the observed and pooled-null
opportunities defined by the same minimum-length run convention as the adequacy guard.
The publication member-step path uses `scale_s = O_s/N_s`. This matters once fallback
chromosomes contribute short-region opportunity without equal long-region opportunity.
An unstratified caller without measured opportunities retains the historical
`observed_edges/null_edges` scale and records it as a fallback. A true joint
length×support-bin stratum also retains a prominently labelled global-edge fallback:
joint opportunity is not currently measured, so it is not claimed to be
opportunity-scaled.

For the `j`-th **distinct** observed area cutoff `a_j`, ordered strongest to weakest:

- `R_j` = count of observed regions with area >= `a_j` (the complete tie group at that
  cutoff is included — see "The tie-aware denominator" below).
- `K_j` = count of pooled null regions with area >= `a_j`.
- Point tail FDR: `FDR_j = min(1, scale_s * K_j / R_j)`.
- Resolution: `Δ_j = min(1, scale_s / R_j)` — the FDR's own step size at that denominator.
- replicate-cluster quasi-Poisson bounds, with stratum dispersion `phi`, effective
  count `Keff_j = K_j/phi`, and `α* = α / m_bonf` (see "The
  Bonferroni-for-simultaneity argument" below):
  - `L_j = phi/2 · χ²_{2Keff_j, α*/2}` (defined as `0` when `K_j = 0`)
  - `U_j = phi/2 · χ²_{2(Keff_j+1), 1−α*/2}`
  - `FDR^lo_j = min(1, scale_s·L_j/R_j)`,
    `FDR^hi_j = min(1, scale_s·U_j/R_j)`
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
`scale_s * K_j / R_j`, where `K_j` is a count of null region exceedances and
`scale_s` maps the null opportunity to observed opportunity. Adding one to `K_j` (the
Phipson–Smyth-style correction) would not correct a small-sample p-value bias here; it
would silently distort the actual estimated count. Instead, the estimation uncertainty
in a small `K_j` is handled honestly and separately, via the cluster interval (below)
— which quantifies the finite-null uncertainty explicitly, as a confidence interval
around the count, rather than folding an ad hoc correction into the point estimate
itself.

## 6. Replicate-cluster quasi-Poisson interval

For cutoff `j` in ordinary length stratum `s`, the point estimate is

```
scale_s     = O_s / N_s
FDR_hat_j   = scale_s * sum_b K_bj / R_j
```

Here `N_s=sum_b E_sb`, and `E_sb` is replicate `b`'s actual opportunity for length
stratum `s`. For an explicitly diagnosed unstratified or joint-support fallback, the
historical global edge scale is used instead. This distinction matters for
short-subtrack fallback chromosomes, which may offer short-region opportunity but no
long-region opportunity. Dispersion is estimated from one count per cluster at the
weakest reported cutoff in the stratum, where count information is greatest. With
`mu_b = E_sb * sum(K_b)/sum(E_sb)`, the exposure-adjusted Pearson estimate is

```
phi_hat = max(1, sum_b (K_b - mu_b)^2 / mu_b / (B - 1)).
```

The same `phi_hat` is used across that stratum's nested cutoff curve. For a completely
empty curve, where dispersion cannot be estimated, the method borrows the largest
estimable dispersion from another reported stratum in the same run; if none is
estimable it records and uses the quasi-Poisson floor `phi=1`. This is a finite-null
safeguard, not a claim that an unseen long-region process is known to be Poisson.

For pooled count `K_j`, the scaled-count endpoints are

```
K_eff = K_j / phi
L_j = phi/2 * chi2_(2*K_eff, alpha*/2)            (0 when K_j=0)
U_j = phi/2 * chi2_(2*(K_eff+1), 1-alpha*/2).
```

They reduce to Garwood at `phi=1` and widen under measured cluster overdispersion.
At `K=0`, `U_j` is finite and strictly positive, so geometric opportunity plus zero
events never becomes zero-uncertainty significance. These are approximate,
model-based bounds: validity depends on independent genome-wide null draws and a
quasi-Poisson mean-variance relationship. They quantify conditional finite-null
uncertainty, not systematic null misspecification.

## 7. The Bonferroni-for-simultaneity argument

The reverse cumulative minimum is applied not only to the point estimate but to the
upper bound: `q^hi_i = min_{j>=i} FDR^hi_j`. This step is only valid on the event that
**all** `U_j` simultaneously bound their respective `FDR_j` — i.e. `U_j >= FDR_j` for
every `j` at once, not merely for each `j` individually at its own confidence level.
Only under that joint event does `min_j U_j >= min_j FDR_j` hold, which is what
licenses reporting `q^hi_i` as a bound on `q_i`.

A single per-cutoff cluster interval at level `1 − α` targets marginal
coverage for that one cutoff under the quasi-Poisson model; it does not by itself
address simultaneous coverage across all
`m` cutoffs a stratum considers. The Bonferroni correction — testing each interval at
`α* = α / m_bonferroni` instead of `α` — is what converts the `m` model-based
marginal statements into one model-based simultaneous statement via the union bound.

**This correction must not be removed as an "optimisation."** Doing so would make the
reverse-scan step over the bounds statistically invalid: `q^hi_i` would no longer be a
valid upper bound on `q_i`, even though it would still look like one in the output.

## 8. Per-stratum vs. joint coverage

`m_bonferroni` is computed as the **maximum** number of distinct observed cutoffs
across all length strata (`m_bonferroni = max_s G_s`, where `G_s` is the distinct-cutoff
count within stratum `s`), and the same `m_bonferroni` is then used for every stratum's
cluster-interval calculation. This is what makes `region_qvalue_upper_95` comparable in
magnitude across strata.

State this explicitly: **the nominal family is one complete cutoff curve within a
stratum.** Under the stated quasi-Poisson approximation its target coverage is
`1 − α`; the corresponding nominal lower bound across `S` strata is `1 − S·α`.
The bounds are **not** described as genome-wide simultaneous across strata. If
genome-wide (all-strata)
simultaneity were required, `α*` would need to be divided by `S` again on top of the
per-stratum Bonferroni correction; this release does not do that, and the diagnostics
recorded (`fdr_interval_bonferroni_m`, `fdr_interval_cutoffs_by_stratum`,
`fdr_interval_family = "per_stratum"`) exist specifically so a reader can see which
scope of simultaneity was actually used.

## 9. Dispersion diagnostics and limitations

The old `null_count_overdispersion_warning` is superseded once the cluster method is
used: measured overdispersion now widens the interval. The manifest records
`fdr_uncertainty_method`, cluster count, minimum cluster count, simultaneity scope,
exposure source, assumptions, zero-count safeguard, and, for each stratum, raw Pearson
dispersion, dispersion used, and whether it came from that stratum, another estimable
stratum, or the `phi=1` floor.

At least five positive-exposure independent genome-wide null realisations are required
for reported dispersion/upper inference. Two are algebraically estimable, but one
residual degree of freedom is not defensible for an upper claim. Below five, the affected stratum is refused through
the normal nullable-inference path and named in
`insufficient_cluster_replicates_strata`. The default member-step construction uses
75 clusters.

The interval does **not** prove the locality-breaking null is biologically
exchangeable. ACF and observed/null region-rate comparisons remain diagnostics; on a
signal-rich cohort they are expected to differ and cannot be hard gates. Calibration
must be established on DMR-free controls and simulations. The permanent tests include
deterministic Poisson-like, negative-binomial/overdispersed and all-zero count cases,
a nested-cutoff negative-binomial simultaneous-coverage check, and both count-level
and end-to-end DMR-free/strong-signal decision regressions.

## 10. Schema: v2.0.0 to v2.0.1

`schema_version` did not exist as a manifest field before this release.
**`schema_version = "2.0"` is recorded retrospectively here** as the name for the
implicit v2.0.0 output format — it was never written to any manifest at the time, so a
v2.0.0-era `regions.csv`/`run_manifest.json` is identified by the **absence** of a
`schema_version` key, not by that key being present with value `"2.0"`. From this
release onward, `manifest["schema_version"]` is written explicitly by every run
(currently `"2.3"`, bumped from `"2.2"` in Stage 2a for the nullable `significant`
column -- see §14).

`regions.csv` gains six columns, inserted immediately after `region_qvalue`:

| v2.0.0 column | v2.0.1 column | Notes |
|---|---|---|
| `region_qvalue` | `region_qvalue` | value changes; corrected direction + tie-aware denominator |
| — | `region_tail_fdr` | new: point FDR at the region's own area cutoff |
| — | `region_qvalue_lower_95` | simultaneous replicate-cluster lower bound |
| — | `region_qvalue_upper_95` | simultaneous replicate-cluster upper bound; nonzero whenever the point estimate is zero |
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

The finite-null uncertainty layer handles the second case explicitly: `K=0` at a
cutoff gives a point estimate `q = scale_s·0/R = 0` and a **finite positive**
replicate-cluster upper bound (§6). Its width depends on `m_bonferroni`, `scale`, `R`
and replicate dispersion. A wholly empty curve borrows the largest estimable
dispersion in the run, or records the `phi=1` floor when none is estimable; it is never
reported as zero uncertainty. The Stage 1c-bis floor short-circuited this
machinery before it could run, discarding a correct, already-implemented answer in favor
of `NaN`.

**Historical Stage 1e fix.** This moved the adequacy floor off per-stratum realised
region counts and onto aggregate adequacy. `min_null_regions_for_fdr`
(`Eiger2Config`, `region_fdr_result`) was checked **once, globally**: pooled null regions summed across every stratum and
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
Stage 2c subsequently added the per-length opportunity guard, and Stage 2d uses that
same opportunity at the estimator's scale. The historical global-only account above
does not describe the current member-step publication path.

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

## 14. Structural gates, run status, and the nullable `significant` (Stage 2a, spec §2a)

### 14.1 Structural gates are signal-independent; calibration diagnostics are not

An earlier draft of this stage would have made the ACF-match and region-exchangeability
gates (§11, and the top of this document's "Two gates" description in `README.md`) into
hard gates that suppress inference. That is wrong: both ratios were defined and validated
on DMR-free data, where observed and null rates should agree. On a cohort containing
genuine coordinated regions, the observed elevated-edge and region-formation rates are
*supposed* to exceed the null's -- that is what detection means. A hard gate on either
ratio would have declared a strongly positive dataset `inference_unreliable` **because it
contained signal**, which is worse than the miscalibration problem it would have been
written to fix.

The gates therefore split into two kinds:

- **Structural gates** (`manifest["gates"]["structural"]`) -- conditions broken regardless
  of whether signal exists, checked purely on how much of the null actually survived
  construction. These hard-fail the run.
- **Calibration diagnostics** (ACF match, region exchangeability, edge exchangeability
  `null_exchangeability_ratio`, and the overdispersion ratio) -- confounded with signal on
  real data. Every one is reported prominently, in the manifest and on `eiger2 call`'s
  stderr, with the label:

  > Calibration diagnostic. Valid as a hard criterion only on data known to be null. On a
  > cohort containing genuine coordinated regions these ratios are expected to exceed 1,
  > and a high value is not by itself evidence of a broken null.

  They change only which status label a run reports (`ok` vs. `ok_with_warnings`); they do
  not suppress `region_qvalue` or `significant`. They may become hard gates once a future
  release establishes a false-rejection rate for a proposed tolerance -- not before.

### 14.2 The four run states

`manifest["status"]` is one of:

| status | when | effect |
|---|---|---|
| `ok` | no structural failure, no calibration-diagnostic concern | unchanged |
| `ok_with_warnings` | a calibration diagnostic (§14.1) is out of its expected range | `region_qvalue`/`significant` reported; diagnostics prominent |
| `partial_inference` | one or more length strata refused for insufficient null opportunity (§16) | those strata's rows suppressed; every other stratum reports normally |
| `inference_unreliable` | a structural gate failed | all inferential columns suppressed; candidates and descriptive columns still written |

At the time this section was written, `inference_unreliable` also fired when
`region_fdr_result`'s own aggregate null-adequacy check refused the run
(`fdr.status == "aggregate null inadequate"`, §8/§12) -- a second, independent way for
inference to become unavailable, orthogonal to the structural gates below. §16 (spec §2c)
retired that aggregate check in favour of the per-stratum opportunity mechanism behind the
`partial_inference` row above; it is no longer a way `inference_unreliable` can fire.

### 14.3 The structural gates

Recorded with their measured values in `manifest["gates"]["structural"]`:

- **`surviving_null_tracks`** -- fewer than half of `null_replicates` produced a usable
  track (every position ineligible, or the track empty). This is exactly what the audited
  ICGC run showed: 7 of 100 tracks surviving.
- **`null_coverage`** -- the fraction of offered null positions that survived eligibility
  filtering is below `min_null_coverage` (default 0.25). The audited run measured 21.3%.
- **`null_construction`** -- fewer than `min_null_values` (default 1,000) pooled null
  values in total. Promoted from `null_threshold`'s pre-existing non-fatal warning to a
  hard gate. The spec also names "no feasible lag on a chromosome carrying a material
  share of eligible edges" under this heading; that check needs the per-chromosome null
  construction a future release adds and is not measurable before it -- the only pre-that
  proxy, zero surviving tracks, is unreachable here because `null_threshold` already
  raises in that case (caught separately, below) before this gate is ever evaluated.
- **`numerical_failure`** -- the null pipeline raised (e.g. `null_threshold`'s "no usable
  pairs" `ValueError`, previously an unhandled exception with no manifest and no output at
  all). Caught and reported the same way as the other three.

Each failed individually still produces `inference_unreliable`; a clean run's status is
unaffected. The most important test of this stage is the opposite case: a high-signal run
with large observed/null ratios but sound structure returns `ok` or `ok_with_warnings`,
never `inference_unreliable` -- the regression test for §14.1
(`tests/test_eiger2_core.py::test_high_signal_sound_structure_run_is_never_inference_unreliable`).

### 14.4 The nullable `significant`, and `fdr_status`

`False` in `regions.csv`'s `significant` column means "tested, not significant" -- a
different statement from "not tested because inference was unavailable for this row".
Encoding the second as the first (`NaN <= target_fdr` evaluates `False`) is the same
category error as reporting `q = 0` from an empty null. `significant` is now a pandas
`BooleanDtype` column:

| value | meaning |
|---|---|
| `True` | tested, significant |
| `False` | tested, not significant |
| `pd.NA` | inference unavailable |

`regions.csv` also carries `fdr_status`: `available` when this row was tested,
`unavailable_stratum` when §16 refused only its length stratum for insufficient null
opportunity, and `unavailable_run` when a global structural failure suppressed every row.
The reason is therefore explicit rather than inferred from a blank.

`pd.NA` serialises to an empty CSV field -- a consumer reading `significant` back with a
plain `pd.read_csv` gets `object` dtype (`True`/`False`/`NaN` as Python objects), losing
the nullable-boolean distinction. Read it back with `dtype={"significant": "boolean"}` to
recover the three states exactly, as
`tests/test_eiger2_core.py::test_significant_columns_three_states_survive_a_csv_round_trip`
does. This is a schema change: `schema_version` is `"2.3"` from this release.

## 15. Chromosome-aware null construction (Stage 2b, spec §2b)

### 15.1 The defect

`null_tracks()` and `member_step_lag_null_ensemble()` each drew ONE lag per replicate from
a range computed on the genome-wide row count, then filtered out window members that
crossed a chromosome boundary after the fact. A lag long enough to be locality-breaking
genome-wide is rarely also small enough to fit inside a shorter chromosome, so almost every
draw survived only on the largest chromosome(s) -- on the audited ICGC run, 7 of 100 edge
tracks survived, drawn almost entirely from the largest chromosomes, even though the
observed track spanned all of them. This also inflated the overdispersion ratio that
motivated §2d's interval work, since that measurement was taken on this same defective
ensemble -- §2d re-measures after this stage, not before it.

### 15.2 The fix

Both constructions now iterate every chromosome per replicate, drawing an INDEPENDENT
feasible lag for each (`lag_c ~ U[null_min_lag, n_c // (span - 1)]`, `n_c` that
chromosome's own row count), building that chromosome's segment from its own rows, and
concatenating every feasible chromosome's segment into one genome-wide realisation with
the eligibility mask broken at every concatenation seam. This preserves chromosome
composition in every replicate and makes the replicate a clean cluster unit (relevant to
any future interval construction over replicates).

`null_tracks()` returns `(track, eligible, segments)` triples, `segments` =
`[(chromosome, lag, start, length), ...]` in concatenation order -- a single scalar lag can
no longer describe a whole replicate now that one replicate spans multiple chromosomes'
own lags. The support-conditioned threshold (`docs/FDR.md`, Stage 1c) reconstructs each
segment's own window-member set from its own lag and genome-wide start offset, not a single
lag applied to the whole track.

### 15.3 Infeasible chromosomes: measured, not silently excluded

A chromosome with `n_c // (span - 1) < null_min_lag` admits no feasible lag and
contributes nothing to the null in any replicate. Rather than a fixed, arbitrary exclusion
threshold, its share of the run's own eligible observed edges is measured
(`chromosome_lag_feasibility()`) and reported as `missing_exposure` in
`manifest["gates"]["structural"]["chromosome_exposure"]`, escalating to a structural gate
(§14.3) -- `inference_unreliable` -- when `missing_exposure` exceeds
`config.max_missing_null_exposure` (default `0.05`).

This edge-null measurement remains in `structural.chromosome_exposure`. When the resolved
region null is `member_step`, the manifest separately records and gates
`structural.region_null_chromosome_exposure`. `min_subtrack_length` is a preferred
construction: if its ceiling is below `null_min_lag`, the chromosome may use a short
fallback ceiling derived from `min_cpgs`. Only a chromosome below
`null_min_lag * (min_cpgs + window - 1)` is absent from the region null. This exact
`+ window - 1` convention includes the forced ineligible seam at the end of every
subtrack: `min_cpgs` statistic positions leave `min_cpgs - 1` usable passing edges,
which span exactly `min_cpgs` CpGs under `segment()`.

**Explicit and independently tunable, not derived from `min_null_coverage`.** An earlier
version of this gate tied its floor to `1 - min_null_coverage` (0.75 at the default
`min_null_coverage=0.25`) -- permitting three-quarters of observed exposure to sit outside
the null entirely, and relating two different quantities: `min_null_coverage` is about how
much of the OFFERED null was usable, `max_missing_null_exposure` about how much of the
OBSERVED genome has no null representation at all. "Explicit and tunable" is the honest
form of "not an arbitrary percentage" (the spec's own phrase) -- sweep it in the §2f grid
the same as any other floor here, rather than trusting the shipped default.

**The cheap half of the literal "recompute and compare" ask.** The spec text also asks to
"report the sensitivity of the threshold and the resulting FDR to that exclusion, by
recomputing both on the retained chromosomes only and comparing" -- a full second run
restricted to feasible chromosomes only, compared against the actual one. That remains
undone (expensive, near-doubling runtime, and its exact comparison methodology is still
underspecified -- compare which summary statistics, at what tolerance?). What IS
implemented is a genuinely free-riding half-measure: `manifest["null"]
["threshold_by_chromosome"]` reports the same `null_quantile` computed separately on each
feasible chromosome's own pooled null values (grouped via the `segments` metadata
`null_tracks()` already records), alongside the pooled `threshold` it is a comparison for
-- one extra quantile per feasible chromosome, not a second run, since both come from the
null `null_tracks()` already built. A wide spread across chromosomes is a signal the pooled
threshold may not represent every contributing chromosome well; a full FDR-level
sensitivity comparison is still the place `chromosome_lag_feasibility()` and this
per-chromosome quantile leave for future work, not something either claims to be.

### 15.4 `member_step_lag_null_ensemble`

The same audit and the same fix, with a preferred
`L_max_c = n_c // (min_subtrack_length + window - 1)` and fallback
`L_max_c = n_c // (min_cpgs + window - 1)`. Raises only when NO chromosome
individually clears even the fallback floor, naming the largest chromosome's own
`n_cpg`/fallback `L_max` rather than the genome-wide total.
`member_step_min_n_cpg()` stays genome-wide -- a cheap, approximate, conservative
pre-check for `region_null_mode="auto"`'s fallback decision, run before any per-chromosome
breakdown is available; the authoritative check is `member_step_lag_null_ensemble`'s own.
Its preferred/fallback/infeasible chromosome lists, missing eligible-edge share,
realised per-chromosome subtrack-length range, and forced-boundary fraction are recorded
by `structural.region_null_chromosome_exposure`; only truly infeasible exposure is
compared with `max_missing_null_exposure`, independently of the edge-null exposure gate.

**The ACF failure at every lag in the audited run is not yet explained by this stage.**
Diagnosing whether the concatenated sub-tracks' eligibility mask (false at every piece
boundary and every chromosome transition) independently depresses the ACF measurement,
separately from the sampling bias this stage fixes, is unresolved -- flagged per the
spec's own instruction not to fold this in silently. Measuring the region-null track's
autocorrelation directly on the audited cohort, with and against this fix, is follow-up
work this stage does not include.

### 15.5 Canonical chromosome ordering (prerequisite)

Both constructions depend on rows being grouped by chromosome and sorted ascending
within each chromosome -- guaranteed by `chromosome_sort_permutation()`, applied by both
`call_dmrs` and the legacy caller before any other processing, regardless of input order
(natural, lexicographic, or shuffled). See `tests/test_chromosome_order.py`.

---

## 16. Per-stratum null opportunity (Stage 2c, spec §2c)

### 16.1 The defect

§14 moved `region_qvalue`'s null-adequacy floor from a per-stratum region-COUNT check
(Stage 1c-bis) to a single AGGREGATE count summed across every stratum and replicate
(Stage 1e), arguing that once the run as a whole is adequately backed, `K=0` at a
stratum's own cutoff is decisive evidence, not missing information. That argument stands.
The implementation, though, checked aggregate *count* -- the wrong quantity. A run could
hold thousands of null regions in the short strata while the 20-49 and 50+ strata never
had the *chance* to grow a region that long at all, and every long observed region then
received `q=0` from a null that could never have contradicted it. That is what the audit
found, and it is a category error one level down from §14's own fix: "the null pool is
small" and "the null pool never had the opportunity to be anything else" look identical
under a count, and only the second one licenses `q=0` as evidence.

### 16.2 Opportunity, not count

For length stratum `s`, with `min_edges(s)` the minimum edge count a region in that
stratum requires (`s`'s lower bound: 4, 6, 10, 20, 50 for "4-5" through "50+"):

```
O_s = sum over OBSERVED eligible runs of max(0, run_len - min_edges(s) + 1)   # observed opportunity
N_s = sum over NULL     eligible runs of max(0, run_len - min_edges(s) + 1)   # null opportunity, pooled
R_s = observed regions in stratum s
K_s = pooled null regions in stratum s
```

An eligible *run* here is a maximal contiguous span of usable edges (`eligible &
isfinite(statistic)` for the observed track, `null_eligible & isfinite(track)` per null
replicate) -- a property of coverage/layout structure, independent of whether the
statistic ever crossed threshold inside it. `max(0, run_len - min_edges(s) + 1)` counts
the distinct positions a stratum-`s`-length region COULD have started at within that run.
Because `segment()` spans `E + 1` CpGs with `E` passing edges, the minimum edge counts for
the CpG strata `4-5`, `6-9`, `10-19`, `20-49`, and `50+` are respectively
`3`, `5`, `9`, `19`, and `49`.

Adequacy asks: had the observed regions all been false, would the null have been likely to
show it? At the observed rate `r_s = R_s / O_s`, the null would be expected to produce
`r_s * N_s` regions. Requiring that to clear `m_opportunity` (default 10, a starting value
per spec §2c.2, not yet validated -- sweep it in the §2f grid) gives

```
N_s >= m_opportunity * O_s / R_s
```

It is assessed once per length stratum, not at every area cutoff (opportunity does not
depend on the cutoff, only `K` does). Stage 2d now uses the same `O_s/N_s` ratio to
scale the point estimate and bounds for ordinary length strata; the earlier
global-edge-only statement is historical.

### 16.3 Outcomes, and the two lists

- **Adequate, `K_s = 0`** -- reported with a zero empirical point estimate and finite,
  positive replicate-cluster upper
  bound, named in `manifest["fdr"]["zero_exceedance_strata"]`.
- **Inadequate** -- that length stratum's rows are refused (`region_qvalue` etc. `NaN`,
  count fields `-1`, `significant` `pd.NA`), named in
  `manifest["fdr"]["insufficient_null_opportunity_strata"]` with `O_s`, `N_s`, `R_s`, and
  the opportunity required. Run status becomes `partial_inference` (§14.2's table), not
  `inference_unreliable`: the null is otherwise sound, it simply never had the chance to
  speak to regions this long. Every other stratum reports normally in the same run.

The two lists are mutually exclusive by construction -- a stratum is either adequate (and
may or may not turn out empty) or refused; never both, never neither when it holds any
observed regions at all.

Refusal is per LENGTH stratum specifically, not per joint (length x support-bin) stratum:
opportunity is a property of run-length structure, independent of support-bin membership.
A refused length stratum's rows are refused regardless of which support bin they fell into
under §1c.6's joint stratification.

### 16.4 What this retired

`region_fdr_result`'s aggregate pooled-count floor (`min_null_regions_for_fdr`, §14.2's
"a second, independent way for inference to become unavailable") is gone; that parameter
no longer exists on `region_fdr_result` (though `Eiger2Config.min_null_regions_for_fdr`
survives, now backing only `region_exchangeability_gate`, a different, still-aggregate
structural diagnostic §14.3 doesn't cover). `"aggregate null inadequate"` can no longer
appear as `manifest["fdr"]["status"]`.

One incidental fix alongside this stage: `Eiger2Config.min_null_values` was, before this
stage, dual-purposed as both the `null_construction` structural gate's pooled floor (§14.3)
*and* the per-support-bin collapse trigger inside `_support_bin_edges` -- two different
quantities (one pooled and run-level, one per-bin), conflated under one name. Forcing bin
collapse by raising it meant the exact same value then failed the whole run as an
inadequate pooled null, with no value able to do one without the other. Split into
`min_null_values` (pooled floor, unchanged default 1,000) and
`min_null_values_per_support_bin` (per-bin collapse trigger, same default, independently
tunable) -- unrelated to opportunity, found while exercising this stage's full test suite.

---

See also: [`docs/PROBABILITY.md`](PROBABILITY.md) for the identical direction/tie fix
applied to `probability._tail_area_fdr` and its effect on `region_evidence_score`.
