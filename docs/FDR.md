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
the null exceedance count `K_j` is Poisson-distributed** — i.e. that replicate-to-
replicate variance in the exceedance count equals its mean. This is the reason a
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
usable cutoff; otherwise it reports `"unavailable"` rather than guessing). If
`median_ratio > 1.5`, `null_count_overdispersion_warning` is appended to
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
release onward, `manifest["schema_version"] = "2.1"` is written explicitly by every
run.

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
| `significant` | `significant` | value changes (derived from the corrected q-value; conditionally from the upper bound under `--fdr-decision upper`) |

The following are bit-for-bit invariant between v2.0.0 and v2.0.1 for a default
fixed-window run and are **not** in the table above because they do not change:
`region_id`, `chr`, `start`, `end`, `n_cpgs`, `cpg_index_start`, `cpg_index_end`,
`statistic`, `statistic_mean`, `statistic_min`, `statistic_threshold`, `region_area`,
and every `*_descriptive` column. This is pinned by
`tests/test_eiger2_core.py::test_stage1_preserves_the_invariant_column_set` against a
committed v2.0.0 fixture.

`confidence_level` (default `0.95`) remains available in the Python API for research
use. If a caller requests a value other than `0.95`, `eiger2 call`'s CSV export refuses
with an explicit error naming the requested level, rather than silently writing bound
columns whose confidence level the fixed schema doesn't record.

---

See also: [`docs/PROBABILITY.md`](PROBABILITY.md) for the identical direction/tie fix
applied to `probability._tail_area_fdr` and its effect on `region_evidence_score`.
