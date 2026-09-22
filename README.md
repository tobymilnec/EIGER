# EIGER

EIGER finds regions of **coordinated DNA methylation** without predefined
sample groups. A region is a run of neighbouring CpGs whose methylation rises
and falls together across samples. Every candidate region gets a
**coordination q-value**, an estimate of the false discovery rate from complete
permutation rescans of the data. It also gets optional, descriptive
annotations of how the region follows genome-wide sample axes.

EIGER reports coordination, not its cause. A coordinated region may reflect a
sample subgroup, cell composition, genotype, batch or a local regulatory
event. The axis annotations help separate regions that follow a genome-wide
sample axis from local ones. They do not establish biological origin.

## Install

EIGER needs Python 3.10 or later. It depends on numpy, pandas and scipy.

```sh
pip install eiger-2.2.0-py3-none-any.whl          # a built wheel
pip install .                                        # or from a source checkout
pip install "./eiger-2.2.0-py3-none-any.whl[legacy]" # adds the --legacy caller
```

The `legacy` extra adds numba and scikit-learn. Only `--legacy` needs them.

## Quick start

`examples/make_example_data.py` writes a small, deterministic synthetic
dataset: 10,000 CpGs × 60 samples, with twelve planted regions.

```sh
python examples/make_example_data.py --out example_data
eiger2 call --betas example_data/betas.tsv --locations example_data/cpgs.tsv \
    --out results
```

This takes a few seconds and writes to `results/`:

| file | contents |
|---|---|
| `regions.csv` | every candidate region: coordinates, CpG count, `region_score`, `coordination_qvalue`, `coordination_selected` and the axis columns |
| `discoveries.csv` | the selected regions only |
| `permutations.csv` | one row per permutation scan |
| `run_manifest.json` | version, input checksums, resolved settings, seeds and per-scan counts |

Input can be two files (a sample matrix plus coordinates in the same row
order) or one table with coordinate columns followed by sample columns:

```sh
eiger2 call --matrix example_data/matrix.tsv --out results_matrix
```

Values are methylation proportions in [0, 1]. Missing values are allowed.

## The three evidence columns

| column | meaning | status |
|---|---|---|
| `coordination_qvalue` | estimated FDR of calling this region coordinated. `coordination_selected` is `coordination_qvalue <= --target-fdr` (default 0.05). | primary result |
| `axis_effect` | beta-scale change in the region's mean methylation between the 10th and 90th percentiles of a genome-wide sample axis estimated from the data, with a standard error and an interval | descriptive, on by default |
| `axis_qvalue` (with `axis_pvalue`, `axis_supported`) | test that the axis effect exceeds 0.10 | **experimental**: empty unless `--axis-annotations experimental-inference` is given |

Axis columns never change which regions are candidates or selected. A missing
axis value means "not computed" (see `axis_status`), never "no effect".

Axis inference has been checked only in simulations with one retained axis
and no missing values. Typical real data fall outside those settings, so it
stays off by default.

## Legacy caller

`eiger2 call --legacy` runs the originally published covariance statistic,
with row-mean imputation and a fixed threshold of 0.01 (`--cov-threshold`),
inside the same permutation FDR. Its regions differ from those of pre-2.2
`--legacy` output. See the [migration notes](docs/USER_GUIDE.md#migrating-from-earlier-versions).

## Runtime

Each run scans the data once, then rescans `--fdr-replicates` (default 99)
permuted copies. Runtime scales roughly with CpGs × samples × 100. The example
takes about 5 seconds. For scale, 400,000 CpGs × 100 samples took about
3.5 seconds per permutation scan in an API benchmark. Peak memory was about
1.4 GB with axis annotations off and 1.6 GB with descriptive annotations
(float64, chromosome mode). A whole-genome cohort of several million CpGs and
hundreds of samples takes hours. Peak memory can be several times the matrix
size. `--storage-dtype float32` reduces matrix storage, but does not guarantee
lower peak memory. Progress, with an estimated time remaining, is printed to
stderr.

## Documentation

[docs/USER_GUIDE.md](docs/USER_GUIDE.md) covers input formats, every option,
all output columns and manifest fields, the method, its assumptions, the
Python API and migration from earlier versions.

## Citation and license

If you use EIGER, please cite the software (see [CITATION.cff](CITATION.cff)).
MIT licensed: see [LICENSE](LICENSE).
