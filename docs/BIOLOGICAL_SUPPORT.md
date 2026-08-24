# Biological support after coordination discovery

EIGER's empirical region FDR answers one deliberately narrow question:

> Is local CpG coordination stronger than expected after genomic locality is
> broken in this dataset?

It does **not** by itself say that a region is cohort-prevalent, disease-specific
or differentially methylated between pre-specified groups. Version 2.2 therefore
keeps the Stage 2D FDR unchanged and adds a post-selection, descriptive support
layer.

It is intentionally opt-in while its thresholds remain provisional and because
bounded-influence summaries over tens of thousands of regions have a material
runtime cost:

```bash
eiger2 call ... --biological-support-policy five_class_v1
```

With the default `off`, `coordination_significant` is still reported,
`biological_class` is missing, and no support-diagnostics table is constructed or
written. Historical verbose columns remain available from the Python API for
backward compatibility.

## Why a local-background null is harder

The current null is dataset-adaptive: it preserves the samples, methylation
values, missingness, nuisance handling, chromosomes and statistic, but compares
nearby CpGs with CpGs at least the configured lag apart. Ordinary local biology
is removed and is consequently part of the signal.

A local-background-matched null would instead ask whether a candidate is more
structured than *ordinary nearby CpGs*. That requires an explicit background
model: what must be matched (distance, CpG density, island/enhancer state,
variance, coverage, cell mixture, copy number, chromatin domain), what is allowed
to vary, and whether the estimand is exceptional coordination, differential
methylation, or disease specificity. Re-sampling nearby regions can also sample
real biology into the null and erase the signal being sought.

A biologically "blank" stem cell does not solve this. It is a different cell
state with its own methylome, chromatin domains and local correlation. It cannot
provide the counterfactual local background for a heterogeneous tumour cohort.
Matched normal tissue can answer a valuable *differential* question, but that is
again a different estimand and requires a group-aware analysis.

## Two decisions, kept separate

`coordination_significant` is nullable and uses only the configured FDR decision:

```text
point: region_qvalue <= target_fdr
upper: region_qvalue_upper_95 <= target_fdr
```

Prevalence and biological support never change this column or any q-value.
`significant` is retained for compatibility and still combines the configured
FDR decision with the historical `min_prevalence_samples` filter.

When `five_class_v1` is enabled, each coordination-significant region's
`biological_class` contains exactly one
of five provisional labels. It is missing for non-calls or unavailable FDR:

1. `rare_epivariant`: the two-means minority is at most the larger of two
   samples or 1% of observed samples, and its direction is coherent across at
   least 80% of CpGs.
2. `outlier_sensitive`: removing the one or two samples whose regional means
   are furthest from the regional median removes at least 50% of the adjacent
   support score and winsorised confirmation retains
   less than 50%; coherent rare events were assigned above and are not called
   artefacts merely for being rare.
3. `cohort_supported`: at least 10% of observed samples form a discrete group,
   delta beta is at least 0.10, the boundary gap is at least 10% of the group
   delta, and bounded/influence confirmation supports the split.
4. `recurrent_subgroup`: the same discrete-group criteria hold below 10% but
   above the rare range.
5. `diffuse_coordination`: significant coordination without a stable discrete
   two-group interpretation, including continuous gradients.

The numbering above is also the precedence order the implementation checks in:
`rare_epivariant` is decided before `outlier_sensitive`, which is decided before
the `cohort_supported`/`recurrent_subgroup` discrete-group test, which falls
through to `diffuse_coordination`. This matters whenever a region could satisfy
more than one class's criteria at once -- concretely, a region with a stable
discrete subgroup (satisfying `cohort_supported`) that is also driven by one
highly influential sample (satisfying `outlier_sensitive`) is labelled
`outlier_sensitive`, not `cohort_supported`: influence sensitivity is checked,
and wins, before the discrete-group test runs. A coherent one/few-sample event
is the one deliberate exception in the other direction -- it is still
`rare_epivariant` even when a bounded estimator attenuates it (see the
`classify_biological_support` docstring in `core.py`), so that true private
biology is not relabelled as artefact merely because one patient drives it.

Below roughly *n* = 20 observed samples, `cohort_supported`/`recurrent_subgroup`
labels are unreliable: a 1-2 sample minority can simultaneously satisfy the
rare-count rule (`rare_max_fraction`, floored at 2 samples) and the 10%
cohort-fraction rule, and an incoherent-direction split that skips
`rare_epivariant` can land in `cohort_supported` off that same 1-2 sample
split. This is a literal, consistent application of the stated percentage
thresholds, not a logic error -- it is a reminder that these two labels are
percentage-based and need a cohort large enough for a percentage to mean
anything.

The hierarchy, thresholds and its status
`provisional_descriptive_pending_validation` are recorded under
`caller_manifest.biological_support.policy`. They are starting definitions for
simulation and replication, not thresholds selected to reduce a particular
cohort's call count.
`caller_manifest.biological_support.diagnostics_file` is `null` when the policy is
off and names the compact CSV only when enabled.

## Diagnostics and output size

`regions.csv` remains lean. It adds only `coordination_significant` and
`biological_class`. When `five_class_v1` is enabled, compact prevalence,
largest-gap tail isolation and new support diagnostics are written to
`region_support_diagnostics.csv`, keyed by `region_id`. Bulk per-region sample
membership lists are deliberately omitted; `region_sample_means.csv` contains the
authoritative values needed to reproduce them.

`region_support_diagnostics.csv` carries every candidate region, not only
coordination-significant ones, and its `diagnostic_status` column names why:
`computed` (the real diagnostics ran), `not_selected` (tested, not
coordination-significant), or `unavailable` (FDR inference was unavailable for
this region, so whether it would have been significant is unknown). The last
two are deliberately kept apart -- "not significant" and "inference
unavailable" are different statements, the same distinction `significant`/
`fdr_status` make in `regions.csv` (`docs/FDR.md` §14.4) -- even though this
file is purely descriptive and this distinction does not feed back into any
q-value.

The authoritative prevalence fields are prefixed `two_means_`. Historical
`minor_group_*_descriptive` API columns used the largest adjacent gap and often
isolate one tail sample in a large continuous cohort. They remain in the Python
API for compatibility but are renamed `largest_gap_tail_*_legacy` in the
diagnostics CSV and must not be interpreted as authoritative prevalence.

The influence scores exactly recompute the pairwise-complete **adjacent**
correlation or covariance after removing the one and two samples furthest from the regional
median. It is explicitly an **adjacent support score**, not a refitted discovery
area: exact leave-one-out discovery area would require re-estimating the
genome-wide nuisance model for every candidate and patient. The robust check is
row-wise 5% winsorised Pearson correlation or covariance, matching the selected
statistic and missingness rules. It is descriptive. It would become inferential
only if both observed and null pipelines were rebuilt with the same transform.

The furthest-mean samples are a fast screening heuristic, not an exhaustive
search for the mathematically maximally influential samples. No within-region
label can distinguish a coherent private biological event from a coherent
block-wide technical artefact. Accordingly, `rare_epivariant` means a
**candidate** rare epivariant, not validated private biology. Coverage/CNV diagnostics, independent
replication and the planned simulation grid remain necessary.
