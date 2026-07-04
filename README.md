<img width="1408" height="768" alt="EIGER banner" src="https://github.com/user-attachments/assets/32576634-76bb-40e6-9a16-bf4ae769dd5c" />

# EIGER: Estimating Interdependent Genome-wide Epigenetic Regions

Covariance-based DMR calling for whole-genome bisulfite sequencing data.

EIGER calls differentially methylated regions by computing the covariance between adjacent CpG sites across samples, grouping sites into contiguous high-covariance blocks, and scoring each block for sample-, cell-type- and tissue-level methylation separation. Blocks are ranked by a Laplacian score and corrected for rare-group prevalence.

## Features

- Covariance-based block segmentation of CpG sites (Numba-accelerated)
- Sample-, cell-type- and tissue-level separation scoring
- Rare-group prevalence correction (Gaussian and logistic)
- Laplacian score ranking of DMR blocks
- Benjamini-Hochberg FDR correction
- Parallel processing via `joblib`
- Reads wgbstools `.beta` / `.lbeta` files, with optional `.bed`, `.bedgraph` and `.cov` support through `wgbstools`
- Writes an annotated DMR table with genomic coordinates as a CSV

## Requirements

- Python >= 3.10
- [wgbstools](https://github.com/nloyfer/wgbs_tools) on `PATH` (only needed for `.bed`/`.cov` input)

See [requirements.txt](requirements.txt) for the Python dependencies.

## Installation

```bash
git clone https://github.com/your-username/EIGER.git
cd EIGER
pip install -r requirements.txt
```

## Usage

```bash
python Python/dmr_pipeline.py \
    --meta          samples.csv \
    --beta-dir      /path/to/beta/files \
    --location-file cpg_sites.bed.gz \
    --output        DMRs_with_coords.csv
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--meta` | Yes | CSV file; first column must contain beta file names (relative to `--beta-dir`) |
| `--beta-dir` | Yes | Directory containing `.beta` / `.lbeta` files |
| `--location-file` | Yes | Gzipped BED file mapping CpG index to genomic coordinates |
| `--output` | No | Output CSV file name (default: `DMRs_with_coords.csv`) |

### Meta file format

The meta CSV needs at least these columns (0-indexed):

| Column index | Content |
|---|---|
| 0 | Beta file name |
| 2 | Cell type label |
| 3 | Tissue label |

### Output

The output CSV has one row per DMR, with columns including:

- `chr`, `start`, `end` - genomic coordinates
- `CpG_start`, `CpG_end` - CpG index range
- `classification` - separation class (e.g. `"grouped celltype separation"`)
- `direction` - `Hypomethylation` or `Hypermethylation`
- `delta_minmax`, `delta_celltype_next`, `delta_tissue_next` - methylation gap metrics
- `pvalue_sample`, `pvalue_cell`, `pvalue_tissue` - t-test p-values
- `score_*_gaussian`, `score_*_logistic` - prevalence-corrected scores
- `laplacian_score` - Laplacian feature score
- `fdr` - Benjamini-Hochberg corrected sample-level p-value
- one column per sample with the per-block average methylation

## Repository structure

```
EIGER/
├── Python/
│   ├── dmr_pipeline.py          # Main entry point
│   └── functions/
│       ├── load_functions.py    # Beta file loading utilities
│       └── processing.py        # DMR calling and scoring functions
├── data/
│   └── ref/                     # Reference files
├── requirements.txt
└── README.md
```

## Citation

If you use EIGER, please cite:

> Milne-Clark T. *EIGER: Estimating Interdependent Genome-wide Epigenetic Regions*. GitHub (2024). https://github.com/your-username/EIGER

## Planned features

- Methrix object input support
- Multi-omic integration (e.g. scRNA-seq overlay)
- Network propagation-based DMR refinement

## Licence

Released under the MIT Licence. See [LICENSE](LICENSE).
