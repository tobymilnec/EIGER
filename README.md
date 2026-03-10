<img width="1408" height="768" alt="EIGER banner" src="https://github.com/user-attachments/assets/32576634-76bb-40e6-9a16-bf4ae769dd5c" />

# EIGER: Estimating Interdependent Genome-wide Epigenetic Regions

**Covariance-based differentially methylated region (DMR) calling for whole-genome bisulfite sequencing data.**

EIGER identifies DMRs by computing the covariance between adjacent CpG sites across samples, grouping sites into contiguous high-covariance blocks, and scoring each block for sample-, cell-type-, and tissue-level methylation separation. Blocks are ranked using a Laplacian score and corrected for rare-group prevalence.

<img width="1360" height="1010" alt="Example DMR plot (GSTP1)" src="https://github.com/user-attachments/assets/b733b5a6-230c-454d-9a79-5037cefa1803" />

---

## Features

- Covariance-based block segmentation of CpG sites (Numba-accelerated)
- Sample-level, cell-type-level, and tissue-level separation scoring
- Rare-group prevalence correction (Gaussian and logistic)
- Laplacian score ranking of DMR blocks
- Benjamini-Hochberg FDR correction
- Parallel processing via `joblib`
- Input: wgbstools `.beta` / `.lbeta` files, with optional `.bed`, `.bedgraph`, and `.cov` support via `wgbstools`
- Output: annotated DMR table with genomic coordinates as a CSV

---

## Requirements

- Python ≥ 3.10
- [wgbstools](https://github.com/nloyfer/wgbs_tools) (must be on `PATH` for `.bed`/`.cov` input)
- R packages: `ggplot2`, `dplyr`, `tidyr`, `grid`, `gtable` (for the plotting script)

See [requirements.txt](requirements.txt) for the full Python dependency list.

---

## Installation

```bash
git clone https://github.com/your-username/EIGER.git
cd EIGER
pip install -r requirements.txt
```

---

## Usage

```bash
python Python/dmr_pipeline.py \
    --meta        samples.csv \
    --beta-dir    /path/to/beta/files \
    --location-file cpg_sites.bed.gz \
    --output      DMRs_with_coords.csv
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--meta` | Yes | CSV file; first column must contain beta file names (relative to `--beta-dir`) |
| `--beta-dir` | Yes | Directory containing `.beta` / `.lbeta` files |
| `--location-file` | Yes | Gzipped BED file mapping CpG index to genomic coordinates |
| `--output` | No | Output CSV file name (default: `DMRs_with_coords.csv`) |

### Meta file format

The meta CSV must have at least the following columns (0-indexed):

| Column index | Content |
|---|---|
| 0 | Beta file name |
| 2 | Cell type label |
| 3 | Tissue label |

### Output

The output CSV contains one row per DMR with columns including:

- `chr`, `start`, `end` — genomic coordinates
- `CpG_start`, `CpG_end` — CpG index range
- `classification` — separation class (e.g. `"grouped celltype separation"`)
- `direction` — `Hypomethylation` or `Hypermethylation`
- `delta_minmax`, `delta_celltype_next`, `delta_tissue_next` — methylation gap metrics
- `pvalue_sample`, `pvalue_cell`, `pvalue_tissue` — t-test p-values
- `score_*_gaussian`, `score_*_logistic` — prevalence-corrected scores
- `laplacian_score` — Laplacian feature score
- `fdr` — Benjamini-Hochberg corrected p-value (sample level)
- One column per sample containing the per-block average methylation

---

## Repository structure

```
EIGER/
├── Python/
│   ├── dmr_pipeline.py          # Main entry point
│   └── functions/
│       ├── load_functions.py    # Beta file loading utilities
│       ├── processing.py        # DMR calling and scoring functions
│       ├── plotting.py          # Python DMR plotting (methylation + tracks + covariance)
│       └── DMR_plotting_TMC.R   # R equivalent of plotting.py
├── data/
│   └── ref/                     # Reference files
├── requirements.txt
└── README.md
```

---

## Citation

If you use EIGER in your work, please cite:

> Milne-Clark T. *EIGER: Estimating Interdependent Genome-wide Epigenetic Regions*. GitHub (2024). https://github.com/your-username/EIGER

---

## Planned features

- Methrix object input support
- rpy2 integration for inline R annotation
- Multi-omic integration (e.g. scRNA-seq overlay)
- Network propagation-based DMR refinement

---

## Licence

This project is licensed under the MIT Licence — see [LICENSE](LICENSE) for details.
