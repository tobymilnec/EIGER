# Tier two: confirming a candidate from its own per-sample means

Tier one ranks candidates and claims nothing about how often such a region
arises on a DMR-free genome; `docs/FDR.md` section 18 says why no
self-calibrated background null supports that claim. Tier two asks a different
question, which one cohort *can* answer.

## The hypothesis

For a candidate region, take each sample's mean beta over the CpGs of the
region it observes. The null hypothesis is **"these samples are one
population"** — one distribution of region means, however wide. The
alternative is that the region means separate into more than one group.

That null is a property of the region's own values. It needs no reference
panel, no matched controls and no model of the genome background, which is
exactly why it survives where the background null does not.

## The test

- **Statistic:** Hartigan's dip statistic, the largest departure of the
  empirical distribution from the closest unimodal one.
  Hartigan, J. A. and Hartigan, P. M. (1985), "The dip test of unimodality",
  *Annals of Statistics* 13(1), 70–84. Computed by the `diptest` package, a
  runtime dependency.
- **p-value:** the package's interpolated table.
- **Inputs:** each sample's mean beta over the region CpGs it observes,
  requiring at least half the region's CpGs observed, clamped to
  [0.005, 0.995] and mapped to the logit scale.
- **Floor:** a region with fewer than 10 tested samples is not tested at all.
  Its `confirm_pvalue`, `confirm_qvalue`, `delta_hat` and `f_hat` are `NaN` and
  `confirmed` is `pd.NA` — "not tested" is a different statement from "tested,
  not separated", and it is never reported as `False`.
- **Decision:** Benjamini–Hochberg across the tested candidates of the run, at
  `target_fdr` (default 0.05). `confirmed` is `confirm_qvalue <= target_fdr`.
- **Effect size:** a two-component one-dimensional Gaussian fit of the same
  logit means, mapped back through the logistic. `delta_hat` is the difference
  of the component means on the beta scale, `f_hat` the weight of the smaller
  component, `direction` the sign of the smaller component's shift **relative
  to the larger one** -- in a cohort that is mostly tumour, the minor component
  is usually benign, so a hypermethylated tumour region reads as `direction =
  -1`; orient against a label before comparing signs (on ICGC, 33.7% raw
  agreement with the published tumour-minus-benign sign became 95.6% once
  oriented). Starts are
  data quantiles, so the fit is deterministic and needs no seed. This is
  descriptive: it is computed after the region was selected and it changes no
  decision.

Turn the whole tier off with `--no-confirm` (`confirm_regions=False`); the
seven columns are then absent rather than empty.

## What it does not do — three limitations, stated

1. **A DMR present in every sample is invisible to it.** If every sample
   carries the same methylation state across the region, the per-sample means
   are one population and the region will not confirm, however real and however
   large the effect. Tier two tests *between-sample* separation, not the
   existence of a region.
2. **A continuously graded effect stays a candidate.** An effect that varies
   smoothly across samples — a dose, an age gradient, tumour purity — produces
   one wide population of means, not two. Such regions rank but do not confirm.
3. **A rare subset has little power.** With 40 samples and a subset of two or
   three, the dip has almost nothing to see. Confirmation is a statement about
   regions where enough samples differ; absence of confirmation in a small
   cohort is close to uninformative.

A `confirmed` region is therefore a region with evidence of *sample structure*.
An unconfirmed candidate is not a rejected region.

## Calibration

Measured on the benchmark's frozen development blocks: 20 correlated-null and
20 paired delta-0.20 target matrices, 40 samples, seeds 21–40. Every figure
below is in the eiger-benchmark repository at the path named beside it. None is
a held-out benchmark result.

**Null arm** (`provenance/eiger_confirm_spike_v2/summary.json`,
`aggregate.dip.correlated_null`), 1,591 tested candidates over 20 DMR-free
matrices:

| quantity | value |
|---|---:|
| candidates confirmed at BH 5% | 0 |
| null matrices with any confirmed call | 0 of 20 |
| fraction of p-values at or below 0.05 | 0.0063 |
| fraction of p-values at or below 0.01 | 0.0013 |
| one-sided KS p against uniform (alternative: p stochastically small) | 0.978 |

The test is **conservative** on this background, and deliberately left so.

**Target arm** (`provenance/eiger_confirm_spike_v2/summary.json`,
`aggregate.dip.target_positive`), 408 planted regions with delta 0.20 at
prevalence 0.5:

| quantity | value |
|---|---:|
| planted regions confirmed | 148 of 408 (0.363) |
| by truth length, 10–19 CpGs | 51 of 239 (0.213) |
| by truth length, 20–49 CpGs | 72 of 131 (0.550) |
| by truth length, 50+ CpGs | 25 of 38 (0.658) |
| confirmed calls wholly off target | 2 |
| median `\|delta_hat\|` (planted 0.20) | 0.2040 |
| median `f_hat` (planted 0.5) | 0.50000 |

**Why this test and not a better-calibrated one.** Three alternatives were
predeclared and measured on the same matrices
(`provenance/eiger_confirm_spike_v2/REPORT.md`): Silverman's critical-bandwidth
test with the Hall and York (2001) calibration, the dip statistic with a
smoothed-bootstrap p-value, and a bootstrap likelihood ratio against a fitted
skew-normal. The Silverman test bought 17.9 points of recall and lost validity
doing it (8.99% of null p-values at or below 0.05, and the only one of the four
to call a null matrix). The other two were valid and less powerful than the
tabled dip — 0.338 and 0.326 against 0.363 — while costing 200 resamples per
region. An earlier Gaussian-null likelihood ratio
(`provenance/eiger_confirm_spike_v1/REPORT.md`) was anticonservative at 8.30%,
which the skew-normal null halved to 4.15%, confirming that its excess was
misspecification rather than bimodality.

Hall, P. and York, M. (2001), "On the calibration of Silverman's test for
multimodality", *Statistica Sinica* 11(2), 515–536.
