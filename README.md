<img width="1408" height="768" alt="EIGER banner" src="https://github.com/user-attachments/assets/32576634-76bb-40e6-9a16-bf4ae769dd5c" />

# EIGER: Estimating Interdependent Genome-wide Epigenetic Regions

**Unsupervised discovery of coordinated DNA methylation regions, with region-level empirical false discovery rate estimation.**

EIGER identifies genomic regions in which neighbouring CpGs show coordinated methylation patterns across samples. Unlike conventional DMR callers, it does not require predefined groups or a specified comparison before regions are identified.

This means that the underlying methylation structure is discovered first. Sample labels can then be applied afterwards to determine whether the resulting regions distinguish cell types, tumour and benign tissue, molecular subtypes, or previously unrecognised sample groups.

The repository contains both the original EIGER implementation and the revised EIGER2 method. Both can be run through the same command-line interface, allowing their outputs to be compared directly using identical input data.

```bash
conda env create -f environment.yml
conda activate eiger
pip install -e . --no-deps

python examples/make_example_data.py

eiger2 call \
    --betas examples/data/betas.tsv \
    --locations examples/data/cpgs.bed \
    --out results/
```

Full installation instructions, input formats, and command-line options are provided in **[QUICKSTART.md](QUICKSTART.md)**.

---

## What changed in EIGER2?

The original EIGER method identifies coordinated regions by thresholding the covariance between adjacent CpGs. This works well when the underlying methylation pattern is present in a substantial proportion of samples, but covariance depends on both the magnitude of the methylation difference and the prevalence of the subgroup carrying it.

For a two-group structure with minor-group fraction *f* and methylation difference δ, the covariance is:

*f*(1−*f*)δ²

The practical consequence is that the same biological effect becomes progressively harder to detect as the subgroup becomes rarer. At the published covariance threshold, the smallest detectable effect is approximately 0.22 when *f* = 0.50, but approximately 0.50 when *f* = 0.05. Rare but potentially meaningful methylation states can therefore be excluded before downstream prevalence-based analyses are applied.

EIGER2 was developed to address this limitation and to provide a clearer statistical interpretation of the regions it reports.

| Change | Rationale |
|---|---|
| **Pairwise-complete estimation** | Avoids the covariance reduction introduced by mean imputation. Under 5% missingness in a 24-sample simulation, complete-case estimation detected no regions, meaning that the original result depended strongly on how missing values were handled. |
| **Correlation as the default statistic** | Correlation normalises covariance by the variance of the CpGs and is therefore less directly constrained by subgroup prevalence once the methylation difference exceeds the background noise. |
| **A locality-breaking null model** | CpGs are paired with distant sites on the same chromosome. This preserves cross-sample structure while removing local genomic adjacency, directly testing whether neighbouring CpGs are more coordinated than expected from the wider methylation structure. |
| **Region-level empirical FDR estimation** | Null coordination tracks are processed through the same segmentation procedure as the observed data. Region areas are then compared within CpG-count strata, providing an empirical region-level false discovery rate. |

The validation analyses, assumptions, and commands underlying these changes are documented in **[docs/](docs/)** and **[analysis/](analysis/README.md)**.

---

## Important limitations

EIGER2 improves the interpretation of coordinated methylation regions, but it does not remove every limitation of unsupervised regional calling.

### Structured missingness, small samples, and high noise — FDR miscalibration (v2.0.1, superseded)

> **Superseded by v2.1.0.** The table below was measured against v2.0.1's
> region-level null (`region_null_mode`, the axis this release adds, did not exist
> yet -- this is what is now called `region_null_mode="lag"`), which is now known to
> fail both region-level exchangeability gates on real layouts and is no longer the
> default. Re-measured under the shipped default (`region_null_mode="member_step"`,
> Stage 1e, master spec §8) below the table.

**Known limitation (v2.0.1):** with v2.0.1's corrected region-level q-values (see below and
[docs/FDR.md](docs/FDR.md)), the realised false discovery rate at the default
`target_fdr = 0.05` and `--fdr-decision point` has been measured — on genuinely
null (DMR-free) simulated data, three independent ways — to substantially exceed the
5% nominal target in three of five tested scenarios:

| scenario | realised FDR (nominal target: 5%) |
|---|---|
| clean | ~9–10% |
| nuisance | ~9% |
| high_noise | ~56–70% |
| small_n | ~55–71% |
| missingness | **~99%** |

This supersedes the earlier structured-missingness figure previously reported here
(0.24–0.35 under 25% dropout, produced under the old, defective forward-cummin q-value
machinery). The corrected estimator did not introduce this problem — it revealed it:
under v2.0.0's defect, nearly every candidate region in every scenario was assigned
q ≈ 0 regardless of its actual evidence, which hid the underlying miscalibration
entirely. The monotonicity invariant and the zero-q vacuity guard that the corrected
estimator is responsible for both hold with zero violations across every check run
against real data — this is not a defect in the FDR-correction code itself. It points
upstream, to the discovery statistic's and null-track construction's behaviour under
missingness, small sample size, and high noise, which is out of scope for this release
(v2.0.1 explicitly does not change the discovery statistic, threshold rule, or
segmentation) and is flagged here for follow-up investigation.

**Treat `region_qvalue` and `significant` with caution in these regimes** — high
missingness, small sample size, or high measurement noise — until this is
investigated further. `clean` and `nuisance` scenarios are close to nominal at the
scale tested, but have not been confirmed at a larger simulation scale either.

Full data, three independent measurement paths, and the release-gate assessment are
recorded in [docs/FDR.md](docs/FDR.md#known-limitation-fdr-calibration-under-missingness-small-n-and-high-noise).
Datasets with substantial structured missingness, small sample size, or high noise
should be interpreted cautiously and assessed using the diagnostics written to the run
manifest.

### Per-region evidence score

The per-region evidence score is intended for ranking regions. It is **not** calibrated as a posterior probability or as the probability that an individual region is biologically real.

The analyses used to assess this behaviour are described in **[docs/PROBABILITY.md](docs/PROBABILITY.md)**.

---

## Large cohorts

**As of v2.1.0, no special handling is needed.** Earlier releases saw q-values
saturate at exactly 0 for almost every region in large cohorts (519 samples: q = 0 for
all 73,784 reported regions) and recommended `--null-mode permutation` as a
workaround. That saturation turned out to be the same defect the region-level null's
two other fixes in this release address, not a large-cohort-specific problem: it was
the SHIPPED region-level null pool being close to empty, which produces q ≈ 0 for
almost anything regardless of cohort size. With the corrected region-level null
(`region_null_mode="member_step"`, the default) and the corrected null-adequacy floor
(Stage 1e), a simulated 500-sample cohort now produces finite, non-saturating q-values
(minimum measured 0.27, no exact zeros) instead. See `docs/FDR.md` §11–§12 and
`STAGE1D_D_STOP_REPORT.md` §6.4 for the measurement.

**`--null-mode permutation` no longer exists.** It was edge-wise permutation (block
length 1), which destroys the same run-length autocorrelation the region-level null
depends on and was measured to produce zero usable null regions; `Eiger2Config` now
raises immediately if it is requested, naming `region_null_mode="member_step"` as the
replacement. There is no `--null-mode` flag to reach for in a large cohort any more --
the default handles it.

---

## Running the original EIGER method

The `--legacy` option invokes the original implementation directly from `Python/functions/processing.py`. It does not reproduce or reinterpret the original algorithm in new code.

```bash
eiger2 call \
    --betas betas.tsv \
    --locations cpgs.bed \
    --out legacy/ \
    --legacy

eiger2 compare \
    --betas betas.tsv \
    --locations cpgs.bed \
    --out both/
```

Comparison with the published prostate cancer analysis recovered all 8,915 published regions across every tested size category:

- ≤200 bp
- 200 bp–10 kb
- >10 kb

The legacy command reports additional regions because the published analysis included further downstream filtering. No region in the published set was lost.

**`--legacy --fdr-replicates` no longer runs.** It attached an empirical FDR to the
legacy caller's regions via the same edge-wise permutation null removed above
(`--null-mode permutation`) -- it now raises, naming `region_null_mode="member_step"`
as the replacement, rather than silently returning q-values from a null measured to
contain zero usable regions. `--legacy` without `--fdr-replicates` is unaffected: it
still reports the original method's regions with `region_qvalue` written as `NA`, since
the published method defines no FDR of its own.

---

## Repository structure

```text
Python/eiger2/        Installable EIGER2 package, including the core caller,
                      command-line interface, input/output functions,
                      probability utilities, and validation simulator

Python/functions/     Original EIGER implementation used by --legacy

benchmarking/         Benchmarking and validation framework, together with
                      generated benchmark outputs

old_code/             Superseded EIGER2 beta implementation, retained with its
                      tests and review scripts for transparency

scripts/              Cohort-specific analysis scripts that are not part of
                      the installable package

analysis/             Exact commands and analyses underlying the reported
                      validation results

examples/             Example dataset generator and runnable input files

tests/                Tests for the current installable package

docs/                 Method design, validation results, and statistical notes
```

Only `Python/eiger2` and `Python/functions` are installed as package code. The benchmarking framework, superseded beta implementation, and cohort-specific analyses are retained for review and reproducibility but are not shipped as part of the library.

```bash
pytest tests -q
pytest old_code/tests -q
```

At the time of writing (v2.0.1), the current package contains 87 tests and the superseded beta implementation contains 27 tests.

---

## Outputs

> **Migration note (v2.0.1):** corrected q-values differ from v2.0.0 even though
> candidate region boundaries are unchanged. Results generated with v2.0.0 must be
> regenerated before use — the direction of the region-level FDR's monotone step was
> reversed, which under-called false discoveries in v2.0.0 (see
> [docs/FDR.md](docs/FDR.md)). Some scenarios remain poorly calibrated even after this
> correction — see the "New in v2.1.0" section below, which supersedes this.

> **Migration note (v2.1.0): every `region_qvalue` and every FDR-derived field from
> v2.0.0 and every interim (v2.0.1/v2.0.2) result is void; re-run before trusting a
> `significant` call.** The region-level null construction changed
> (`region_null_mode="member_step"`, replacing what is now called
> `region_null_mode="lag"`) and the null-adequacy floor changed from a per-stratum to
> an aggregate check — both change `region_qvalue` and everything derived from it
> (`region_tail_fdr`, the Garwood bounds, `significant`, `n_significant`). Candidate
> region boundaries themselves are unaffected (segmentation is driven by the
> edge-level null, `null_mode`, which this release does not change). `--null-mode
> permutation` and `--legacy --fdr-replicates` no longer run at all (see "Large
> cohorts" below). This supersedes the miscalibration figures previously reported in
> this section — see "New in v2.1.0" below.

The principal region-level output is `regions.csv`, with one row per detected region.

**`region_qvalue` is the principal field used to assess region-level statistical support.**

Other fields, including methylation difference, direction, and inferred sample grouping, are descriptive. These values are calculated after region selection using the same data and should not be treated as independent statistical evidence or used as additional significance filters without further validation.

### New in v2.0.1: FDR resolution and Monte Carlo bounds

Six columns were added to `regions.csv`, inserted immediately after `region_qvalue`.
Full derivation and worked examples are in [docs/FDR.md](docs/FDR.md); summary:

| Column | Meaning |
|---|---|
| `region_tail_fdr` | point FDR at the region's own area cutoff |
| `region_qvalue_lower_95` | simultaneous Garwood lower Monte Carlo bound |
| `region_qvalue_upper_95` | simultaneous Garwood upper bound; nonzero whenever the point estimate is zero |
| `region_qvalue_resolution` | one scaled null exceedance's worth of change at the q-achieving cutoff |
| `null_exceedances_at_cutoff` | pooled null exceedance count at the q-achieving cutoff (`-1` if unavailable) |
| `region_qvalue_cutoff_rank` | number of observed regions at least as strong as the q-achieving cutoff (`-1` if unavailable) |

Six values change from v2.0.0 by design (spec §0.3.3; see the migration note above):
the `regions.csv` columns `region_qvalue`, `region_evidence_score`,
`evidence_score_extrapolated`, and `significant`, plus the manifest-level counts
`n_significant` and `n_passing_qvalue`.

### `--fdr-decision {point,upper}`

```bash
eiger2 call --betas betas.tsv --locations cpgs.bed --out results/ \
    --fdr-decision upper
```

`point` (default) calls a region significant when `region_qvalue <= target_fdr` —
unchanged in spirit from v2.0.0. `upper` instead requires
`region_qvalue_upper_95 <= target_fdr`: the conservative decision under finite-null
estimation uncertainty, useful when few null replicates are available and a point
estimate of exactly zero should not, on its own, be trusted as a call. `upper` refuses
to run (rather than silently returning zero significant regions) if the replicate
count can't achieve the target even in the best case; the error names the required
replicate count.

### Null replicate default: 20 → 100

`Eiger2Config.null_replicates` now defaults to **100** (previously 20), giving a finer
resolution at the decision boundary and a materially narrower `region_qvalue_upper_95`
at zero exceedances. This costs runtime roughly linearly in replicate count: measured
on a 200-sample, `dmr_length=20` synthetic input, `call_dmrs` took **0.197s at 20
replicates** and **0.805s at 100 replicates** (~4.1x). Pass `--null-replicates 20` to
restore the old default if runtime matters more than resolution for a given run.

### New in v2.1.0: the region-level null, and the adequacy-floor correction

Two independent defects in the region-level null (the construction behind
`region_qvalue`, distinct from the edge-level `null_mode` null above) are fixed:

1. **`region_null_mode` resolves to `"member_step"` by default.** The previous
   region-level null (`region_null_mode="lag"`, still available for small inputs or
   reproducing historical comparisons) failed both region-level exchangeability
   gates on real layouts at genome scale -- it did not reproduce the observed
   statistic's autocorrelation, so it under-populated the null pool and either
   refused to report `region_qvalue` or, at large cohort sizes, saturated it at
   exactly `q = 0`. The new construction re-indexes by a random lag instead of by
   one CpG, sharing `window - 1` of `window` members between consecutive positions
   exactly as the observed track does, and passes both gates on real WGBS and
   targeted-capture layouts. See [docs/FDR.md](docs/FDR.md) §11.
2. **The null-adequacy floor (`min_null_regions_for_fdr`) is now an aggregate check,
   not a per-stratum one.** A length stratum whose own pooled null happened to be
   empty used to be refused (`region_qvalue = NaN`) even when the run as a whole was
   comfortably backed by null volume -- discarding decisive evidence (`K = 0`: the
   null essentially never produces a region that long) as if it were missing
   information. It is now reported as a finite point estimate (`q = 0`) with a
   finite `region_qvalue_upper_95`, named in the run manifest
   (`manifest["fdr"]["zero_exceedance_strata"]`) rather than silently discarded. See
   [docs/FDR.md](docs/FDR.md) §12.
3. **`--region-null-mode` is now a CLI flag, and small input no longer hard-errors
   with no way out.** `Eiger2Config.region_null_mode`'s own default is `"auto"`: on
   input below `member_step`'s genome-scale minimum size, it falls back to `"lag"`
   automatically with a loud warning, instead of the uncatchable `ValueError` a CLI
   user previously got with no command-line workaround. `--region-null-mode
   member_step` (passed explicitly) still raises on undersized input rather than
   falling back -- a caller who names the mode is told it cannot be done, not
   silently given something else. `eiger2 call` reports the computed floor against
   the input's CpG count in its startup output, before the run, not only after a
   failure.

**`--null-mode permutation` and `--legacy --fdr-replicates` are removed, not
deprecated.** Both used edge-wise permutation (block length 1) as their null, which
was measured to produce zero usable null regions and destroys the same
autocorrelation the fix above depends on; both now raise immediately, naming
`region_null_mode="member_step"` as the replacement, rather than silently returning
q-values from a null known not to work. See "Large cohorts" above.

**The realised-FDR claim, quantified: conservative, not calibrated at exactly 5%.**
Pooled across every measured planted-signal cell (293 significant calls, 1 false
positive), the one-sided 95% Clopper-Pearson upper bound on the true realised FDR is
**1.6%**, comfortably under the 5% nominal target -- and observing only 1 false
positive at 293 calls would be a roughly 1-in-200,000 event if the true FDR were
actually 5%. Read this as "substantially conservative", not "calibrated": both
predictions look the same cell by cell (near-zero observed false positives either
way), and only the pooled count distinguishes them. A conservative estimator is the
safe direction for a discovery tool, but it also means power is being left on the
table; see [docs/FDR.md](docs/FDR.md) §13 for the full data and reasoning.

Each run also produces `run_manifest.json`, which records:

- the resolved analysis configuration
- input file checksums
- software and package versions
- random seeds
- null-model diagnostics
- the parameters used for segmentation and FDR estimation

This allows each result to be traced back to the code, inputs, and settings that generated it.

---

## Licence and citation

EIGER is released under the MIT licence. See **[LICENSE](LICENSE)** for details.

Citation metadata are provided in **[CITATION.cff](CITATION.cff)**.

