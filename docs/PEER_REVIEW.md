# Independent peer review — EIGER covariance DMR caller

**Scope reviewed:** `Python/eiger_core.py`, `Python/functions/{processing,scoring,load_functions}.py`,
`Python/dmr_pipeline.py`, `Python/legacy/eiger_hard_threshold_v1.py`, `tests/`, `README.md`,
`LEGACY_METHOD.md` (working tree, uncommitted).

**Recommendation:** major revision. The engineering is careful — schema separation, streaming,
reproducible seeds, coordinate-system detection and a frozen legacy profile are all above the norm
for a methods repository. The concerns are statistical. Several headline quantities named
`*_probability` are not calibrated probabilities, the label-free automatic threshold fails under
exactly the confounder that dominates real WGBS cohorts, and the discovery statistic has a strong
built-in prevalence bias that the downstream rare-group weighting cannot repair.

Every claim below was checked by simulation against the actual code, not read off the source.
Scripts and outputs are in the review artefacts; each finding cites the experiment that produced it.

---

## Major issues

### M1. The automatic covariance threshold assumes no shared cross-sample factor, and collapses when there is one

`estimate_covariance_threshold` builds its null by letting **each sample independently draw a
different genomic adjacent pair**. Under that null, sample *j*'s residual pair and sample *k*'s
residual pair come from unrelated loci, so the across-sample covariance has expectation zero. The
null therefore represents "no shared latent factor anywhere in the genome", not "no *regional*
coordination". Real cohorts always violate this: tumour purity, immune infiltration, cell
composition, global hypomethylation and batch all induce genome-wide covariance between *any* pair
of CpGs.

Simulated a 60-sample, 5,000-CpG matrix with a single latent factor and CpG-specific loadings and
**no DMRs at all** (EXP 5 / EXP 14):

| loading SD | true 99.5th pct of adjacent cov | EIGER auto threshold | false blocks called |
|---|---|---|---|
| 0.04 | 0.00458 | 0.00142 | 6 |
| 0.08 | 0.01725 | 0.00351 | 36 |
| 0.15 | 0.05198 | 0.00928 | 58 |

At a realistic loading SD of 0.08 the estimated threshold is **5× too low** and 15.8% of null edges
pass it. The fixed 0.012 threshold called **zero** false blocks in the first two rows — i.e. the
"improvement" is strictly worse than the default it replaces under the most common real confounder.

On genuinely i.i.d. data the null *is* well calibrated (EXP 12: auto threshold 0.00256 vs true
99.5th percentile 0.00262, 0.63% of edges passing against a nominal 0.5%). So the estimator is
correct for the model it assumes; the model is the problem.

**Fix.** Use a null that preserves the cross-sample dependence structure and breaks only locality —
a distance-matched (lag-*K*) pairing on the same chromosome. Pairing CpG *i* with CpG *i+K* keeps
every latent factor intact but destroys regional coordination. Tested on the same matrix (EXP 15):

| null | threshold | % of DMR-free adjacent edges passing (nominal 0.5%) |
|---|---|---|
| EIGER per-sample resampling | 0.00351 | 15.8% |
| lag *K*=500 | 0.01819 | 0.4% |
| lag *K*=1000 | 0.01614 | 0.6% |
| lag *K*=2000 | 0.01855 | 0.4% |

This is a small change — swap the resampling in `estimate_covariance_threshold` and
`_residual_pair_reservoir` for a circular lag shift — and it restores calibration. *K* should be
chosen well beyond the methylation correlation length (a few kb) and the resulting empirical
autocorrelation-vs-lag curve should be reported in the manifest so the choice is auditable.

Until this is fixed, `--cov-threshold auto` should carry a warning in the README, not be presented
as the adaptive default.

### M2. The discovery statistic is prevalence-limited by construction; downstream rare-group weighting cannot recover it

For a two-group structure with minor fraction *f* and effect δ, adjacent covariance is
`f(1-f)δ²`. Simulation confirms the identity exactly (EXP 1). The consequence, at the default
threshold of 0.012:

| minor fraction *f* | minimum detectable δ |
|---|---|
| 0.50 | 0.219 |
| 0.25 | 0.253 |
| 0.10 | 0.365 |
| 0.05 | 0.503 |
| 0.03 | 0.642 |
| 0.01 | 1.101 (impossible) |

For a 519-sample cohort, a 3-sample subgroup with a 30-point methylation difference produces an
adjacent covariance of 0.00051 — **23× below threshold**. It is never discovered, so
`--weight-method prevalence_variance`, `target_group_size` and `sample_rank_percentile` operate only
on regions that already passed a filter biased against exactly the rare subgroups they are meant to
up-weight. The rare-group machinery is applied at the wrong stage.

This is the most important scientific point in the review, because "find rare epigenetic
subpopulations without labels" is EIGER's distinguishing claim.

**Fixes, in increasing order of work:**

1. **State the operating characteristic.** Publish the `δ_min(f)` curve above and make the caller
   emit it for the run's *n*. Users need to know what EIGER cannot see.
2. **Prevalence-explicit thresholding.** Let the user declare a target prevalence `f*` and set
   `threshold = f*(1-f*)δ_min²` rather than an absolute covariance. This makes the tuning knob
   interpretable and directly ties discovery sensitivity to the biology of interest.
3. **Replace the statistic for the rare regime.** Covariance is the wrong sufficient statistic for
   sparse subgroups. A robust/trimmed alternative — max over *k*-subset mean gaps, a top-*k*
   co-outlier count between adjacent CpGs, or a sparse-loading PC on the CpG × sample residual
   block — scales as δ² rather than *f*δ² and would detect the 3-in-519 case. Worth a supplementary
   comparison even if not adopted as the default.

### M3. `sample_model_probability`, `effect_probability` and `call_probability` are not calibrated, and they gate `call_status`

Three separate problems compound:

**(a) Post-selection.** The GMM chooses the split that maximises separation, then `delta`,
`_welch_pvalue` and `_effect_probability` are computed on that same split. The README labels the
p-value `descriptive`, but `effect_probability` and `call_probability` are not flagged, and
`call_probability` is what gates `call_status == "sample_structure"` — the primary output field.

**(b) BIC is not a posterior for a mixture.** `expit(0.5 * ΔBIC + prior_log_odds)` is the Schwarz
approximation, whose regularity conditions fail for mixtures (boundary and non-identifiability
problems). It is a heuristic for choosing *k*, not a calibrated posterior.

**(c) Negative-variance clipping.** In `_effect_probability`,
`target_biological = max(var(target) - mean(measurement_variance), 0)`. When the subtraction goes
negative the SE collapses to the measurement term alone and `effect_probability → 1`. Tight clusters
therefore get maximum confidence.

Measured on data drawn from a **single** Gaussian (no structure at all), *n*=6, 1,500 replicates
(EXP 9 / rev6):

- `P(model_probability > 0.8)` = **0.597** — the model says two components are supported 60% of the
  time when there is one;
- mean post-selection `|delta|` = **0.213** against a true δ of 0 — i.e. the selected split exceeds
  the 0.20 delta threshold on average under the null;
- `P(call_supported)` = **0.031** per region.

Three percent per region is not obviously fatal, but it is an *uncalibrated* 3%, it is untested
above *n*=6, and it is applied without any region-level multiplicity control.

**Fixes.**
- Calibrate `call_probability` against the same resampled null used for the threshold: run the
  complete caller on null matrices, record the `call_probability` distribution, and report an
  empirical region-level FDR alongside it. This is the standard `dmrseq`/`bumphunter` pattern and it
  is the single highest-value addition to the method.
- Guard (c): floor the biological variance at a small positive value, or better, fit a proper
  hierarchical model (`var_total = var_bio + var_meas`) rather than subtracting.
- Rename `sample_model_probability` → `sample_model_bic_support` unless and until it is calibrated.
  The current name will be read as a posterior probability by every downstream user.

### M4. `sample_qvalue_descriptive` is a BH q-value on a post-selection p-value and controls nothing

`benjamini_hochberg` is itself correct (monotone, NaN-preserving, excludes non-finite from *m* —
verified). The problem is the input. The p-values come from a Welch test between clusters chosen to
maximise the separation in the same data, so they are not uniform under the null and BH provides no
FDR guarantee. The field then appears in `eiger_dmrs.csv`, the compact output, where it will be
thresholded at 0.05.

The BH independence/PRDS assumption is also violated: adjacent and overlapping blocks share CpGs.

**Fix.** Either remove `sample_qvalue_descriptive` from the compact output and keep it in
diagnostics only, or replace it with a permutation-calibrated region-level q-value from M3's null.
Renaming it `sample_pvalue_uncalibrated` would at least stop it from being used as an FDR.

### M5. Shared sequencing reads generate large false adjacent covariance at low depth, and the code assumes independent errors

Adjacent CpGs within a read/fragment are supported by the *same* reads in WGBS. Their measurement
errors are strongly positively correlated, which produces genuine positive adjacent covariance from
pure sequencing noise. `covariance_standard_errors_with_measurement` models errors as independent
(`vx*vy` term, no cross-covariance), and no correction is applied to the covariance estimate itself.

Simulated with **zero biological variation** (all true β = 0.5), 40 samples, varying fragment
sharing (EXP 7 / EXP 16):

| depth | shared read fraction | mean adjacent cov | % edges ≥ 0.012 |
|---|---|---|---|
| 5 | 0.0 | 0.00013 | 6% |
| 5 | 0.6 | 0.03004 | **99%** |
| 10 | 0.0 | 0.00011 | 0.2% |
| 10 | 0.6 | 0.01508 | **74%** |
| 30 | 0.6 | 0.00503 | 0% |
| 30 | 0.9 | 0.00751 | 1% |

Below ~10× the effect swamps the threshold. Above ~30× it is negligible. Since EIGER is being
applied to cfDNA/liquid-biopsy data (per the citation), this matters: shallow libraries will produce
dense false DMR blocks concentrated in the highest-sharing (shortest-fragment, densest-CpG) regions
— which are CpG islands, i.e. exactly where the biology is expected.

**Fixes.**
- Document a minimum recommended per-sample depth (~20–30×) for covariance mode and warn when
  `mean_read_coverage` falls below it.
- Where read-level data is available, estimate β at the two CpGs from **disjoint read halves**
  (even/odd read index), which removes the shared-read covariance entirely at the cost of √2 noise.
- Failing that, add the shared-read covariance term to the SE and, more importantly, subtract an
  estimate of it from the covariance, using fragment-length and CpG-spacing to estimate the sharing
  fraction.

### M6. Mean imputation deflates covariance by (1−m)², biasing discovery against high-missingness regions

`impute_for_covariance` fills missing values with the CpG mean, i.e. with residual exactly zero.
Those samples then contribute nothing to the cross-product while remaining in the *n*−1 denominator.
Measured (EXP 11), 40 samples, true covariance 0.0225:

| missing rate *m* | observed covariance | ratio | (1−*m*)² |
|---|---|---|---|
| 0.0 | 0.02313 | 1.03 | 1.00 |
| 0.1 | 0.01878 | 0.83 | 0.81 |
| 0.2 | 0.01478 | 0.66 | 0.64 |
| 0.4 | 0.00812 | 0.36 | 0.36 |

The deflation is exactly (1−*m*)². At 20% missingness a true DMR loses a third of its signal and
falls below the fixed threshold. Missingness in WGBS is not random — it tracks GC content, mappability,
repeats and coverage — so this is a systematic, structured bias, not noise. `missing_rate` is
reported and used for stratified *ranking*, but the discovery step itself is uncorrected.

**Fix.** Use pairwise-complete covariance: compute over samples observed at *both* CpGs and divide by
`n_both − 1` rather than `n − 1`. This is a two-line change in `_cov2`/`covariance_blocks` and it
removes the bias exactly, at the cost of a variable effective *n* which should then be carried into
the SE. Keep the imputed path only as a fallback for the Laplacian step.

---

## Moderate issues

### N1. The probabilistic segmentation delivers a narrower benefit than claimed, and its parameters are nearly inert

The README motivates smoothing as preventing "a single borderline edge" from splitting a
well-supported region. Tested directly with a 40-CpG DMR containing one attenuated CpG (EXP 17,
25 replicates):

| CpG 170 retains … | legacy: fragments / largest block | probabilistic: fragments / largest block |
|---|---|---|
| 100% of effect | 1.00 / 40.0 | 1.00 / 40.0 |
| 60% | 1.60 / 28.2 | **1.08 / 38.4** |
| 40% | 2.00 / 20.0 | 1.96 / 20.8 |
| 20% | 2.00 / 20.0 | 2.00 / 20.0 |
| 0% (dropout) | 2.00 / 20.0 | 2.00 / 20.0 |

The benefit is real but confined to a narrow window around 50–70% attenuation. It cannot bridge a
full dropout, because one bad CpG breaks *two* consecutive edges and the two-state chain with
`persistence = 0.90` will not cross two failures. Since a single dropped/filtered CpG is the most
common real cause of fragmentation, the mechanism misses its main use case. Consider making the
smoother edge-*pair* aware, or allowing a bounded number of consecutive sub-threshold edges (a
run-length rule) which is simpler and does the same job transparently.

Parameter sensitivity is also odd (EXP 18): varying `hmm_signal_prior` from 0.01 to 0.8 changed
nothing at all; only `persistence = 0.99` altered the result. The prior is exposed as a tunable but
is effectively inert because the emission likelihood ratios saturate. Either remove it or document
that it does nothing over its useful range.

Separately, the emission model is not principled: `covariance_edge_support` returns
`Φ((cov − t)/SE)`, a frequentist tail area, and `_smooth_support_segment` then uses it as
`P(o | S=1) = p` and `P(o | S=0) = 1 − p`. That is only a valid likelihood under a 50/50 prior, and
it is being combined with a separate 0.05 prior in the same model — the prior is applied twice. The
output should be described as a smoothed score, not a posterior.

### N2. Edge-support standard errors are biased low at small *n* and the small-sample correction is dimensionally wrong

`covariance_standard_errors` applies `correction = n/(n-1)` to the **standard error**, i.e. outside
the square root, where a variance-scale correction would be `sqrt(n/(n-1))`. More consequentially,
the estimator is still biased low where it matters (EXP 8a, 3,000 replicates):

| *n* | true SD(cov) | mean reported SE | ratio |
|---|---|---|---|
| 5 | 0.01488 | 0.01021 | **0.69** |
| 8 | 0.01128 | 0.00874 | 0.77 |
| 12 | 0.00879 | 0.00755 | 0.86 |
| 20 | 0.00679 | 0.00609 | 0.90 |
| 50 | 0.00416 | 0.00402 | 0.97 |
| 150 | 0.00240 | 0.00238 | 0.99 |

The Normal approximation is also poor because centred cross-products are strongly right-skewed:
`P(support > 0.975 | cov = threshold)` is 0.009 at *n*=6 and 0.011 at *n*=200 against a nominal
0.025, and the median support at the threshold is 0.41–0.49 rather than 0.50 (EXP 8c). So the
statistic is simultaneously over-confident in scale and mis-centred in shape. It should not be
called a probability.

**Fix.** Fisher-*z* on the correlation with a covariance rescaling, or a fast bootstrap/jackknife
over samples for the edge SE. At minimum, apply the correction as `sqrt(n/(n-1))` and document that
edge support is uninterpretable below *n* ≈ 20.

### N3. Sequencing-depth variance is double-counted in the SE

`covariance_standard_errors_with_measurement` computes the empirical variance of the observed cross
products — which **already contains** sequencing noise, since the observed β values are the noisy
ones — and then adds an analytic measurement term on top. Measured at *n*=30, depth 10 (EXP 8b): the
SE inflates from 0.005202 to 0.008650, a 1.66× over-statement of uncertainty.

`test_read_coverage_changes_confidence_but_not_covariance` confirms the SE changes, but does not
check the *direction* or magnitude is right. The fix is to subtract the measurement component from
the empirical product variance rather than add it, i.e. estimate the biological cross-product
variance as `max(empirical − measurement, 0)`.

### N4. The bootstrap stability estimate is biased downward by (n−1)/n

`_bootstrap_call_stability` resamples samples with replacement, but `_cov2` divides by *n*−1 on the
resample. For an i.i.d. bootstrap `E*[cov] = ((n−1)/n) · cov_obs`. Confirmed exactly (EXP 6,
20,000 resamples):

| *n* | observed cov | bootstrap mean | ratio | (n−1)/n |
|---|---|---|---|---|
| 8 | 0.01782 | 0.01562 | 0.877 | 0.875 |
| 20 | 0.01519 | 0.01444 | 0.951 | 0.950 |
| 50 | 0.01243 | 0.01218 | 0.980 | 0.980 |

Because the bootstrap runs against a **fixed** threshold taken from the full data, this 2–12%
deflation pushes borderline edges below threshold and systematically understates recurrence — most
severely in the small cohorts where a stability estimate is most needed. Multiply the bootstrap
covariance by `n/(n−1)`, or recompute the threshold within each bootstrap.

Two further problems with the same function: the Jaccard matching is an O(B × |DMR| × |DMR_boot|)
Python loop over `DataFrame.iterrows`, which will not run at genome scale, and matching requires
`direction` equality — a region whose minor cluster flips sign under resampling (common when
clusters are near-balanced) scores zero recurrence for a reason unrelated to stability.

### N5. `descriptive_score` measures group imbalance, not effect size

`descriptive_score = |δ| / stdev × −log₁₀(p)` uses the SD of the *pooled* per-sample means, which is
itself dominated by the group difference being measured. For a two-group split with small
within-group variance, `stdev ≈ sqrt(f(1-f))·|δ|`, so the standardised term converges to
`1/sqrt(f(1-f))` — a function of prevalence only. Measured (EXP 3, *n*=100):

| *f* | δ=0.2 | δ=0.4 | δ=0.8 | 1/√(f(1−f)) |
|---|---|---|---|---|
| 0.03 | 5.129 | 5.629 | 5.816 | 5.862 |
| 0.10 | 3.174 | 3.284 | 3.321 | 3.333 |
| 0.30 | 2.124 | 2.170 | 2.179 | 2.182 |
| 0.50 | 1.966 | 1.988 | 1.997 | 2.000 |

A four-fold change in effect size moves the score by under 14%. `sample_score_raw` and
`sample_score_weighted` therefore rank primarily by group imbalance and *p*-value, then get
multiplied by a prevalence weight that adjusts for imbalance *again*. Use a pooled **within-group**
SD (Cohen's *d*) instead; the term then behaves as intended and the prevalence weight stops
double-correcting.

`prospective_power` has the same defect: it takes the pooled `stdev`, which is inflated by the
between-group difference, so power is systematically understated for exactly the real DMRs.

### N6. The "preserved legacy" profile does not reproduce the published legacy behaviour

`LEGACY_METHOD.md` describes "the largest adjacent sample gap as the inferred partition", and
`_separation_from_values` implements `argmax(gaps)`. The actual historical function still in the
repository, `processing.find_separators`, uses the **first** gap exceeding `thres` (line 258:
`break_idx = np.argmax(sorted_diffs > thres)`), and additionally performs cell-type and tissue-level
separation that the new profile drops entirely.

`test_legacy_snapshot_matches_explicit_legacy_mode` compares the legacy wrapper against
`calling_method="legacy"` — i.e. it checks the new code against itself, not against the old code.
If results in the SSRN manuscript were produced with `find_separators`, the "frozen behavioural
definition" will not regenerate them. Either add a regression test that pins the new legacy profile
against `find_separators` output on a fixed matrix, or state plainly in `LEGACY_METHOD.md` that this
is a re-implementation with documented differences.

---

## Minor issues and code-level observations

1. **`extract_blocks` default mismatch.** `EigerConfig.min_cpgs = 4` but
   `extract_blocks(DMR_length_threshold=5)`. Harmless (the config always passes explicitly) but
   invites a future bug.

2. **Aliased arrays in `call_dmrs_from_matrix` (lines 970–1002).** In probabilistic mode
   `segmentation_values = edge_posterior` is an alias, so `edge_posterior[~eligible] = 0.0` is then
   overwritten to `-inf` by `segmentation_values[~eligible] = -np.inf`. It is currently harmless
   because blocks only contain eligible edges, but `edge_posterior` no longer holds what its name
   says. Use `.copy()`.

3. **`covariance_score` can fall below `covariance_threshold_used`** in probabilistic mode (that is
   the point of smoothing), which reads as a contradiction in the CSV. Worth a note in the README.

4. **`region_probability` is ≥ 0.5 by construction** — blocks are *defined* as runs with posterior
   ≥ `hmm_posterior_threshold`, so the field can never flag a weak region. It is a selected
   statistic and should be documented as such.

5. **`_welch_pvalue` returns 0.0** for constant-but-different groups; `descriptive_score` then clips
   to `np.finfo(float).tiny`, giving `−log₁₀(p) ≈ 308` and a score three orders of magnitude above
   everything else. These rows will dominate any ranking. Clip to a documented floor (e.g. 1e-16).

6. **Laplacian score is cohort-dependent and undiscriminating at small *n*.** `kneighbors_graph` uses
   binary `connectivity` rather than the heat kernel of He et al.; with `laplacian_neighbors = 5` and
   fewer than ~8 samples the graph is near-complete and all scores collapse toward 1. Add a guard, and
   note that `laplacian_rank_percentile` is not comparable across runs. (The rank direction is
   correct — high percentile = low score = more informative — but this is worth a docstring.)

7. **`GaussianMixture` cost.** `_probabilistic_separation_from_values` fits 13 EM runs per DMR:
   7.7 ms/DMR at *n*=10 and 18.3 ms at *n*=60, versus 0.14–0.64 ms for legacy — a 12–134× slowdown,
   or roughly 13–31 minutes per 100,000 DMRs, multiplied again by `--stability-bootstraps`. Reduce
   `n_init` for the 1-component fit (it is deterministic; `n_init=3` is wasted), and consider a
   closed-form 2-means split with a likelihood-ratio statistic instead.

8. **`smooth_edge_support` is a pure-Python loop.** 1.22 s per 1M edges, ~34 s genome-wide, doubled
   in auto mode. Not fatal, but it is the one obvious `@njit` candidate left in the hot path.

9. **`covariance_blocks` returns a reflected list**, not an array — deprecated in Numba and ~900 MB
   transiently at 28M edges before `np.asarray`. Preallocate `np.empty(M)` and fill.

10. **`extract_blocks` uses `np.append` in a loop**, which is O(k²) in the number of blocks. Collect
    into a list and concatenate once.

11. **Streaming and in-memory modes are not numerically equivalent.** Chromosome mode loads
    `float32` and samples the null reservoir *without* replacement within chromosome; in-memory mode
    uses `float64` and samples *with* replacement genome-wide. The two paths will not give identical
    thresholds or identical calls on the same data. `test_chromosome_results_reproduce_genome_wide_finalisation`
    tests the finalisation step, not end-to-end equivalence. Either make them equivalent or document
    the difference prominently — users switching modes for memory reasons will assume equivalence.

12. **`.beta` totals are `uint8` and saturate at 255.** `beta_measurement_variance` will therefore
    *overstate* precision at high depth. Minor and in the conservative direction, but note it.

13. **`_group_split` tests a different quantity from the one it reports.** `pvalue_empirical` is a
    permutation test on the group-mean *gap*, while `annotation_status` and `passes` use `delta`.
    A region can be `no_separation` with a highly significant empirical p-value.

14. **`impute_for_covariance` global-mean fallback of 0.5** silently changes behaviour in
    chromosome-streaming mode, where the "global" mean is per chromosome.

15. **Test coverage gaps.** The suite (27 tests) is good on I/O, coordinates and schema, but there is
    no test that: (a) checks false-positive rate on null data, (b) checks recovery of a planted DMR at
    known δ and *f*, (c) checks the auto threshold under a latent factor, or (d) checks that
    probabilistic mode beats legacy on any metric. A small `tests/test_operating_characteristics.py`
    with 3–4 fixed-seed simulations would catch every M-level regression above.

---

## Prioritised recommendations

**Must fix before publication**

1. Replace the resampling null with a distance-matched (lag-*K*) null — M1. Validated: restores
   0.4–0.6% edge pass rate against a nominal 0.5% under latent structure.
2. Calibrate `call_probability` on null simulations and report an empirical region-level FDR; stop
   using an uncalibrated BIC-derived quantity to gate `call_status` — M3.
3. Move or rename `sample_qvalue_descriptive` so it is not read as an FDR — M4.
4. Switch to pairwise-complete covariance to remove the (1−*m*)² missingness bias — M6.
5. Publish the `δ_min(f)` operating characteristic and state EIGER's rare-subgroup detection limit
   honestly — M2.

**Should fix**

6. Document a minimum depth and add a shared-read caveat or correction — M5.
7. Stop double-counting sequencing variance in the SE — N3.
8. Correct the bootstrap `n/(n−1)` deflation and vectorise the Jaccard matching — N4.
9. Use a within-group pooled SD in `descriptive_score` and `prospective_power` — N5.
10. Either fix the legacy profile to match `find_separators` or document the divergence — N6.

**Worth doing**

11. Allow the smoother to bridge two consecutive weak edges, or replace it with an explicit
    run-length rule — N1.
12. Add operating-characteristic tests — minor 15.
13. Numba/vectorise `smooth_edge_support`; preallocate in `covariance_blocks` and `extract_blocks`;
    reduce GMM restarts — minors 7–10.

## What is good

Worth saying explicitly, because none of the above should obscure it: the compact/diagnostic/values/
annotation schema separation is the right design and rare in this space; the stable `dmr_id` and the
non-destructive `repair_eiger_coordinates.py` path show real care about downstream users; the
coordinate-system auto-detection with a match-rate floor and manifest logging is better than most
published DMR callers; the run manifest with realized thresholds and null sizes is exactly what
reproducibility needs; and separating discovery from annotation so that group labels cannot influence
DMR boundaries is a genuinely good methodological decision that many competing tools get wrong.
The fixed-threshold caller is also demonstrably conservative — it produced zero false blocks on
i.i.d. null data and on two of three latent-factor scenarios, which is more than can be said for the
automatic mode that was added to replace it.
