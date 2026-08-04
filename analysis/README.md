# Reproducing the published and thesis results

Every number quoted in the README, the thesis benchmarking chapter and the
reviewer response was produced by one of the four commands below. Each was run
against a single input file whose SHA-256 prefix is recorded, so a reviewer can
confirm that all four analyses saw identical data.

## Input

| | |
|---|---|
| File | `methylation_matrix_10x_90perc.tsv` |
| SHA-256 (first 16) | `c23795152fd6254f` |
| CpGs | 2,295,991 |
| Samples | 519 |
| Missing fraction | 0.0025 |

The matrix is the CRUK-ICGC prostate cohort at 10x coverage in at least 90% of
samples. It is **controlled-access and cannot be redistributed here**; it is
obtained from the ICGC data portal under the relevant data access agreement.
`examples/make_example_data.py` generates a small synthetic substitute so that
every command below can be exercised without it.

The matrix carries coordinates in its first columns, so these runs use
`--matrix`. For the two-file layout (a sample matrix plus a separate BED) use
`--betas` and `--locations` instead; the two forms are mutually exclusive.

## The four runs

Environment for all four: Python 3.12.13, numpy 1.26.4, pandas 2.2.2,
macOS 26.5.1 arm64. Runtimes are from that machine.

### 1. The method as published (`legacy/`)

```bash
eiger2 call --matrix methylation_matrix_10x_90perc.tsv \
    --out legacy/ --min-sample-fraction 0.5 --legacy
```

Discovery is `Python/functions/processing.py` invoked directly, at the published
settings: `cov_threshold = 0.012`, `delta_threshold = 0.2`, `min_cpgs = 5`,
mean imputation on. **32,878 regions, 37 s.** `region_qvalue` is NA by
construction — the published method defines no error rate.

Concordance: all 8,915 regions in the published call set are recovered, at every
region size (100% at ≤200 bp, at 200 bp–10 kb and at >10 kb). This run reports
more regions than the published set because the published analysis applied
further filtering downstream; no published call is lost.

### 2. The published method with an FDR attached (`legacy_fdr/`)

```bash
eiger2 call --matrix methylation_matrix_10x_90perc.tsv \
    --out legacy_fdr/ --legacy --fdr-replicates 50 --min-prevalence-samples 3
```

**The same 32,878 regions, 46 s.** Discovery is untouched; the q-value is an
annotation obtained by permuting the caller's own covariance track (50
replicates, 153,478 null regions, 3,070 per replicate). This is the run behind
the reviewer response on statistical significance: an error rate can be supplied
for the published method without changing which regions it reports.

### 3. EIGER 2, default (locality-breaking) null (`results/`)

```bash
eiger2 call --matrix methylation_matrix_10x_90perc.tsv \
    --out results/ --min-sample-fraction 0.5
```

Correlation statistic, window 4, `min_cpgs = 4`, 20 null replicates.
**73,784 regions in 386 s, and every one of them received q = 0.**

That is arithmetically correct and analytically useless, and it is the result
that motivates run 4. With 519 samples the edge statistic is estimated
precisely and genuine regional coordination exists across most of the genome, so
the hypothesis the lag null tests — *is there any regional coordination here?* —
is trivially false everywhere. The null is not broken; the question is.

### 4. EIGER 2, permutation null (`results_perm/`)

```bash
eiger2 call --matrix methylation_matrix_10x_90perc.tsv \
    --out results_perm/ --null-mode permutation \
    --min-prevalence-samples 3 --null-replicates 50
```

**73,860 regions in 908 s, split 41,636 at q ≤ 0.05 against 32,224 above it.**

The permutation null preserves the observed marginal distribution of local
coordination exactly — it is the same multiset of values — and randomises only
its arrangement along the chromosome. The hypothesis becomes *is this run of
coordination longer or stronger than chance, given how much coordination the
genome contains?*, which stays answerable at any cohort size.

Note that runs 3 and 4 do not call the identical region set (73,784 against
73,860): run 4 also applies `--min-prevalence-samples 3` and leaves
`--min-sample-fraction` at its default. The comparison to draw is between the
two q-value *distributions*, not between individual regions.

## Post-selection annotation

Group classification is deliberately outside the package and carries no error
rate, because it describes regions that discovery has already chosen, using the
same data:

```bash
python scripts/annotate_regions_by_group.py \
    --regions results_perm/regions.csv \
    --sample-means results_perm/region_sample_means.csv \
    --out results_perm/region_groups.csv
```

## Benchmarks

The simulation studies behind the thesis chapter are in `benchmarking/`, with
their generated outputs committed alongside them in `benchmarking/results_*`:

```bash
python benchmarking/chapter2_benchmark.py    # revised method, simulation sweep
python benchmarking/original_benchmark.py    # published method, same simulations
python benchmarking/calibration_study.py     # posterior calibration
python benchmarking/benchmark_suite.py       # heuristic sweep, seeds, controls
```

Each writes a manifest recording its resolved configuration and seeds.
