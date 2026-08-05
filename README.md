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

### Structured missingness

The locality-breaking null assumes that missing data do not contain strong genomic structure. When dropout is spatially or biologically structured, this assumption can fail.

In simulations with 25% structured dropout, the realised false discovery rate was approximately 0.24–0.35 at a nominal threshold of 0.05. This remained largely unchanged across segmentation settings, suggesting a systematic bias rather than a parameter-tuning problem.

Datasets with substantial structured missingness should therefore be interpreted cautiously and assessed using the diagnostics written to the run manifest.

### Per-region evidence score

The per-region evidence score is intended for ranking regions. It is **not** calibrated as a posterior probability or as the probability that an individual region is biologically real.

The analyses used to assess this behaviour are described in **[docs/PROBABILITY.md](docs/PROBABILITY.md)**.

---

## Large cohorts

In large cohorts, the default locality-breaking null can become overly permissive. With enough samples, coordinated methylation can be detected across a large proportion of the genome, making the comparison against distant CpG pairs relatively easy to satisfy. In this setting, empirical q-values may saturate close to zero.

For cohorts above approximately 100 samples, the permutation null is generally more informative:

```bash
eiger2 call \
    --betas betas.tsv \
    --locations cpgs.bed \
    --out results/ \
    --null-mode permutation \
    --min-prevalence-samples 3
```

The permutation null preserves the observed marginal distribution of coordination values and randomises only their genomic arrangement.

In a cohort of 519 samples, the default null assigned q = 0 to all 73,784 reported regions. Using the permutation null produced a more informative separation, with 41,636 regions at q ≤ 0.05 and 32,224 above this threshold.

The exact commands, outputs, and explanation for differences in the called region sets are provided in **[analysis/](analysis/README.md)**.

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

An empirical FDR can also be estimated for the legacy caller without changing the regions produced by the original algorithm:

```bash
eiger2 call \
    --betas betas.tsv \
    --locations cpgs.bed \
    --out legacy/ \
    --legacy \
    --fdr-replicates 50
```

This applies permutation-based FDR estimation to the covariance track generated by the original method.

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

At the time of writing, the current package contains 48 tests and the superseded beta implementation contains 27 tests.

---

## Outputs

The principal region-level output is `regions.csv`, with one row per detected region.

**`region_qvalue` is the principal field used to assess region-level statistical support.**

Other fields, including methylation difference, direction, and inferred sample grouping, are descriptive. These values are calculated after region selection using the same data and should not be treated as independent statistical evidence or used as additional significance filters without further validation.

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

