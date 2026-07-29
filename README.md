<img width="1408" height="768" alt="EIGER banner" src="https://github.com/user-attachments/assets/32576634-76bb-40e6-9a16-bf4ae769dd5c" />

# EIGER: Estimating Interdependent Genome-wide Epigenetic Regions

Covariance-based DMR calling for whole-genome bisulfite sequencing data.

EIGER calls differentially methylated regions by computing covariance between
adjacent CpG sites across samples and grouping sites into contiguous
high-covariance blocks. Discovery is methylation-only. Cell type, tissue,
phenotype, batch or any other sample grouping can be added afterwards as an
optional annotation without changing DMR boundaries.

## Features

- Covariance-based block segmentation of CpG sites (Numba-accelerated)
- Compact, one-row-per-DMR core output with stable `dmr_id` values
- Separate per-sample methylation and long-form annotation tables
- Optional annotation of any metadata grouping, not only cell type or tissue
- Gaussian, logistic and bounded prevalence-variance weighting
- Prospective power support and non-parametric rank-within-strata
- Laplacian structural ranking of DMR blocks
- Benjamini-Hochberg FDR correction
- Reads wgbstools `.beta` / `.lbeta` files, with optional `.bed`, `.bedgraph` and `.cov` support through `wgbstools`
- Directly consumable by JORAS V2

## Requirements

- Python >= 3.10
- [wgbstools](https://github.com/nloyfer/wgbs_tools) on `PATH` (only needed for `.bed`/`.cov` input)

See [requirements.txt](requirements.txt) for the Python dependencies.

## Installation

```bash
git clone https://github.com/tobymilnec/EIGER.git
cd EIGER
pip install -r requirements.txt
```

## Usage

```bash
python Python/dmr_pipeline.py \
    --meta          samples.csv \
    --beta-dir      /path/to/beta/files \
    --location-file cpg_sites.bed.gz \
    --sample-column sample_id \
    --beta-file-column beta_file \
    --output        eiger_dmrs.csv
```

To add annotations, repeat `--group-column` for each metadata field:

```bash
python Python/dmr_pipeline.py \
    --meta samples.csv \
    --beta-dir /path/to/beta/files \
    --location-file cpg_sites.bed.gz \
    --sample-column sample_id \
    --beta-file-column beta_file \
    --group-column celltype \
    --group-column tissue \
    --weight-method prevalence_variance \
    --cov-threshold 0.012 \
    --delta-threshold 0.20 \
    --output eiger_dmrs.csv
```

The fixed defaults are covariance `0.012` and delta beta `0.20`. To estimate
the covariance threshold without labels:

```bash
python Python/dmr_pipeline.py \
    --meta samples.csv \
    --beta-dir /path/to/beta/files \
    --location-file cpg_sites.bed.gz \
    --sample-column sample_id \
    --beta-file-column beta_file \
    --cov-threshold auto \
    --delta-threshold 0.20 \
    --output eiger_dmrs.csv
```

Automatic mode removes CpG means and constructs a null by independently
resampling eligible adjacent genomic pairs for each sample. This preserves
residual methylation variation and ordinary adjacent structure while breaking
a coordinated regional signal. It adapts to sample size without using cell
type, phenotype or other group labels. Configure it with
`--cov-null-quantile`, `--cov-null-permutations` and
`--cov-null-max-pairs`. The realized threshold and number of sampled null pairs
are stored in the run manifest.

The delta threshold is applied to the absolute difference between the two
inferred cluster means. This gives `0.20` its usual interpretation as a
20-percentage-point methylation difference. `delta_minmax` is retained as a
diagnostic for the adjacent sample gap used to form the unsupervised split.

### Meta file format

Column names are explicit rather than position-dependent:

| Column | Content |
|---|---|
| `sample_id` | Unique sample identifier |
| `beta_file` | Beta filename relative to `--beta-dir` |
| Any optional column | A grouping passed with `--group-column` |

### Output

Each run writes five linked files:

| File | Shape and purpose |
|---|---|
| `eiger_dmrs.csv` | Compact, one row per DMR; coordinates, effects, primary ranks, thresholds and stable ID |
| `eiger_dmrs_diagnostics.csv` | Detailed p-values, raw scores, weights, power, missingness and Laplacian fields |
| `eiger_dmrs_values.csv` | One row per DMR with per-sample mean methylation |
| `eiger_dmrs_annotations.csv` | Long-form optional group annotations keyed by `dmr_id` |
| `eiger_dmrs_run.json` | Parameters, provenance, schema version and statistical notes |

Use `--legacy-wide-output` only when an older workflow requires the core and
all sample values in one wide table.

### Scoring

`--weight-method prevalence_variance` implements
`sqrt(p_target(1-p_target) / (p_observed(1-p_observed)))` with configurable
bounds. Gaussian and logistic prevalence weights remain available. The raw and
weighted scores are both retained, power is reported separately, and
`sample_rank_percentile` provides a distribution-free comparison within fixed
group-size, CpG-count and missingness strata.

Blocks never bridge chromosomes or CpGs removed for insufficient coverage.
`--max-cpg-gap-bp` can additionally prevent a block from spanning a large
genomic gap.

Sample split p-values are labelled `descriptive`: the samples used to identify
the split are also used in the test. Group annotations can use
`--annotation-permutations` for empirical label-permutation p-values conditional
on the discovered DMR boundaries.

## Repository structure

```
EIGER/
├── Python/
│   ├── dmr_pipeline.py          # Main entry point
│   ├── eiger_core.py            # Reusable matrix API and optional annotation
│   └── functions/
│       ├── load_functions.py    # Beta file loading utilities
│       ├── processing.py        # Covariance and Laplacian primitives
│       └── scoring.py           # Weights, power, FDR and stratified ranks
├── data/
│   └── ref/                     # Reference files
├── requirements.txt
└── README.md
```

## Citation

If you use EIGER, please cite the associated paper:

> Milne-Clark T, Lee Yu H, Nazlamova L, et al. *Epigenetic Drivers of Localised Prostate Cancers Reveal Neuronal and Microenvironmental Reprogramming and inform Liquid Biopsy Biomarkers*. SSRN (2026). doi:10.2139/ssrn.6121766. https://ssrn.com/abstract=6121766

You may also cite the software directly:

> Milne-Clark T. *EIGER: Estimating Interdependent Genome-wide Epigenetic Regions*. GitHub (2024). https://github.com/tobymilnec/EIGER

## Planned features

- Methrix object input support
- Multi-omic integration (e.g. scRNA-seq overlay)
- Network propagation-based DMR refinement

## Licence

Released under the MIT Licence. See [LICENSE](LICENSE).
