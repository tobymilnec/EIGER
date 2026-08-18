# Per-region evidence scores (two-groups local FDR)

`Python/eiger2/probability.py`

---

## 1. The quantity that was missing

`region_qvalue` answers a question about a *set*:

> among the regions at least this extreme, what proportion do we expect to be null?

That is a tail-area rate. It is what Benjamini–Hochberg and the `bumphunter`
family report, and it is the right quantity for deciding where to cut a ranked
list. It is not a statement about the region it is printed next to, and it is
almost universally read as one.

The per-region quantity is the **local** false discovery rate. Writing `a` for
region area, model the called regions as a mixture of null and non-null:

```
f(a) = pi0 * f0(a) + (1 - pi0) * f1(a)

fdr(a) = pi0 * f0(a) / f(a)          P(region is real | a) = 1 - fdr(a)
```

Every term is already available in EIGER 2, which is the reason this is a small
addition rather than a new inferential apparatus:

| term | source |
|---|---|
| `f0` | density of null region areas, from the null tracks pushed through the **identical** segmenter. `call_dmrs` already computes these; they were previously used only for a tail count. |
| `f` | density of observed region areas, from the called regions. |
| `pi0` | pooled null region count rescaled to the size of the observed genome, divided by the number of regions called. |

Setting `pi0 = 1` recovers the conservative tail-area behaviour, so the model is
strictly nested inside what EIGER 2 already did. Nothing is lost by enabling it.

## 2. Why this is not a repeat of EIGER 1

EIGER 1 exposed four fields with `probability` in the name. A reviewer's
objection to all four was that none was calibrated against anything:

- `region_probability` was **≥ 0.5 by construction**, because regions were
  *defined* as runs whose posterior exceeded the calling threshold. The field
  could never flag a weak region.
- `sample_model_probability` applied `expit(0.5 * ΔBIC + prior)`. The Schwarz
  approximation's regularity conditions fail for mixtures, so this is a
  heuristic for choosing the number of components, not a posterior.
- `effect_probability` clipped a negative variance estimate to zero, so tight
  clusters received maximum confidence.
- The edge smoother used a frequentist tail area `Φ((cov − t)/SE)` simultaneously
  as `P(o | S=1)` and `1 − P(o | S=0)`, valid only under a 50/50 prior, while a
  separate 0.05 prior was applied in the same model.

Three things differ here:

1. **It is a posterior for a stated model**, not a rescaled test statistic.
2. **It cannot be bounded below by 0.5 by construction**, because `f0` is
   estimated from regions the segmenter actually produced on null data rather
   than assumed from the calling rule.
3. **It is checkable.** `reliability_table` bins regions by predicted
   probability and reports the observed fraction that are genuinely real.

Point 3 is the operative one.

> **No probability from this module is reported without its reliability diagram.**
> If the curve does not sit on the diagonal, the output is a score and must be
> named one.

That rule was applied, the curve did not sit on the diagonal, and the output is
accordingly named `region_evidence_score`. Section 5 reports the measurement.

## 3. Usage

```python
from eiger2 import Eiger2Config, call_dmrs

config = Eiger2Config(
    compute_posterior=True,
    null_replicates=200,          # a density needs far more null regions than
                                  # a tail count does; 20 will be refused
)
regions, values, manifest = call_dmrs(betas, locations, config=config)

regions[["region_id", "region_area", "region_qvalue",
         "region_evidence_score", "evidence_score_extrapolated"]]
manifest["posterior"]     # status, pi0, per-stratum diagnostics
```

Standalone, on areas you already have:

```python
from eiger2 import local_fdr

result = local_fdr(
    observed_areas, null_area_sets,
    observed_edges=n_edges_observed,
    null_edges=n_edges_null_total,        # summed over replicates
    observed_lengths=lengths,             # optional, for length stratification
    null_length_sets=null_lengths,
)
result.posterior          # 1 - fdr
result.fdr_conservative   # the pi0 = 1 variant, always >= fdr
result.extrapolated       # areas beyond the null's support
result.pi0
result.status
```

## 4. Implementation notes

**Log scale.** Areas are positive and heavy-tailed, so all density estimation is
on `log(area)`.

**One bandwidth, not two.** `f` and `f0` are estimated with a Gaussian KDE using
a *single* shared bandwidth, taken by Silverman's rule from the larger of the two
samples. Estimating each with its own bandwidth puts structure into the ratio
that is an artefact of the smoother rather than of the data.

**Monotonicity is imposed.** The statistic is one-sided — a larger area is never
weaker evidence — so `fdr` is made non-increasing in area by a cumulative
minimum along the grid. This is also what stabilises the sparse upper tail.

**The tail is not extrapolated.** Beyond `max(pooled null area)` the density
ratio would be extrapolation rather than estimation. Those regions are flagged
`evidence_score_extrapolated` and given the conservative tail-area rate instead.

**Length stratification.** Region area grows with length, so `f0`, `f` and the
comparison are computed within the same CpG-count strata used by
`region_fdr` (4–5, 6–9, 10–19, 20–49, 50+). A stratum with fewer than
`min_null_regions` pooled nulls falls back to the pooled null and is flagged in
`manifest["posterior"]["strata"][name]["fell_back_to_pooled_null"]`.

**A consistency diagnostic.** The mean local fdr among regions at least as
extreme as a cut-off should track the tail-area rate at that cut-off.
`manifest["posterior"]["max_abs_tail_area_discrepancy"]` reports the largest
disagreement. A large value means the density estimate is not to be trusted.

## 5. Measured calibration: it does not calibrate, and is therefore a score

The rule in Section 2 was applied to this module's own output. It failed.

Planted-truth simulation, regions pooled over scenarios and seeds, `min_cpgs=2`
to generate a genuine mixture of true and false calls, `null_replicates=300`:

| Regime | n regions | true fraction | estimated `pi0` | true `pi0` | ECE | max CE |
|---|---:|---:|---:|---:|---:|---:|
| `clean` + `nuisance` (null generator valid) | 626 | 0.307 | 0.553 | 0.693 | **0.498** | 0.687 |
| same, with `pi0 = 1` | 626 | 0.307 | 1.0 | 0.693 | **0.347** | 0.709 |
| `1 - region_qvalue`, for comparison | 626 | 0.307 | — | — | **0.693** | 0.693 |
| `missingness` + `high_noise` + `small_n` | 1130 | 0.172 | 0.351 | 0.828 | **0.714** | 0.864 |

Reliability, `clean` + `nuisance`:

| predicted | n | observed fraction real |
|---|---:|---:|
| 0.4-0.6 | 19 | 0.000 |
| 0.6-0.8 | 201 | 0.000 |
| 0.8-1.0 | 406 | 0.473 |

The score is grossly overconfident. Regions scoring 0.72 were real 0% of the
time. **The output column is therefore named `region_evidence_score`, not
`posterior_real`, and must not be thresholded as a probability.**

Two things are worth stating alongside that.

First, `1 - region_qvalue` is *worse* (ECE 0.693) than either variant of the
score. The tail-area q-value is an even poorer per-region probability than the
thing built to replace it, which is the original argument for not reading it as
one.

Second, the failure is diagnosable and is not an error in the estimator. The
unit tests confirm the estimator recovers the right answer when `f0` really is
the density of the null component. The problem is that on EIGER 2's regions it
is not. **The null tracks are generated from signal-free data, whereas false
positives in real data arise beside real signal** — fragments flanking a true
DMR, edges inflated by leakage from an adjacent region — and they inherit an
area distribution much closer to the signal than to the signal-free null. So
`f0` is the wrong null density for exactly the regions the score gets wrong,
`pi0` is underestimated (0.553 against a true 0.693), and the score is inflated.

This is the same condition that governs the q-value: the null generator must
reproduce the structure of the observed data other than the signal being
tested. Under structured missingness it fails more severely still, which is
visible independently in the realised FDR (0.24-0.35 at every parameter
setting) and in the negative control.

### What would fix it

In rough order of expected benefit:

1. **A signal-retaining null.** Generate null tracks that preserve real regions
   elsewhere in the genome while destroying local coordination at the position
   under test — a leave-one-region-out or block-permuted null rather than a
   global lag shift. This targets the diagnosed cause directly.
2. **Estimate `f0` from an empirical null** fitted to the central bulk of the
   observed area distribution (Efron's empirical null), rather than from the
   simulated null tracks. This is the standard remedy when the theoretical null
   is known to be misspecified, and it needs no new simulation.
3. **Isotonic recalibration** against planted truth, reported with the range of
   region sizes and cohort sizes over which it was fitted. Pragmatic, and
   honest provided the fitting range is stated.

Until one of these is done the field is a ranking, and the documentation, the
column name and the manifest all say so.

## 6. Limitations, stated plainly

- **A density needs many more null regions than a tail count does.** At the
  default `null_replicates = 20` the pooled null frequently contains fewer than
  ten regions, which is nowhere near enough. The estimate is **refused** below
  `min_null_regions` (default 50), returning `status =
  insufficient_null_regions(n<50)` and leaving the columns NaN. That is the
  intended behaviour, not a failure. Raise `null_replicates` to 100–200 when
  posteriors are wanted; it is the dominant cost of enabling them.
- **The tail is where the estimate is worst and the interest is highest.** Hence
  the extrapolation flag. Treat flagged regions as ranked, not as scored.
- **`pi0` is unstable when the non-null fraction is large.** Both the estimated
  and the `pi0 = 1` conservative results are returned; report both if they
  disagree materially.
- **Stratification multiplies the data requirement** by the number of strata.
  Watch the fallback flags.
- **The posterior inherits every assumption of the null generator.** If the
  locality-breaking null fails to preserve the cohort's nuisance dependence, the
  posterior is wrong in the same direction and by the same amount as the
  q-value. It is not an independent check.

## 7. The v2.0.1 `_tail_area_fdr` direction/tie correction

`_tail_area_fdr` (`probability.py`) is the fallback used for the **extrapolated tail**
of `local_fdr` — i.e. it governs the regions with the strongest area, beyond the
support of the pooled null, where the density-ratio estimate in Sections 1-4 cannot be
evaluated. Through v2.0.0 it carried the identical defect as `core._fdr_from_areas`:
a forward (rather than reverse) cumulative minimum over strongest-to-weakest areas,
and a sequential-rank (rather than tie-aware) denominator.

v2.0.1 applies the identical fix: `q = np.minimum.accumulate(raw[::-1])[::-1]` in
place of a forward `np.minimum.accumulate(raw)`, and a distinct-cutoff `R` (tie group
included) in place of sequential rank `1, 2, 3, ...`. The corrected function returns a
**point estimate only** — no Garwood bounds are attached on this path, since it is a
one-shot fallback rather than the primary per-region estimator. See
[`docs/FDR.md`](FDR.md) for the full derivation, the worked hand-calculation examples,
and why the direction and tie-aware denominator matter; it is not re-derived here.

**Consequence for this module's output:** `region_evidence_score` and
`evidence_score_extrapolated` change for regions that fall on the extrapolated-tail
path, because they are computed from the corrected `_tail_area_fdr` from this release
onward. This is listed in the v2.0.0-to-v2.0.1 invariant/non-invariant column table in
`docs/FDR.md` and in the release migration note.

## 8. A bug fixed at the same time

`core.region_fdr` computed its scale as `(observed_edges / null_edges) /
replicates`, while `call_dmrs` passes `null_edges` **summed over replicates**.
Because `null_area_sets` are also pooled over replicates, the two already
correspond, and dividing by `replicates` a second time discounted the expected
null count again.

The effect was that **every q-value was exactly `null_replicates`-fold too
small** — a factor of 20 at the default configuration — in the anti-conservative
direction. Fixed, with a parametrised regression test at
`tests/test_probability.py::test_region_fdr_scale_is_not_double_discounted`.

Any q-value produced by EIGER 2 before this fix should be multiplied by the
`null_replicates` used for that run.
