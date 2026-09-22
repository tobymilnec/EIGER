# EIGER user guide

This guide covers EIGER 2.2: installation, inputs, `eiger2 call` and its
options, the output files, how to read the results, the Python API and
migration from earlier versions. The [README](../README.md) is the short
version.

## Installation

```sh
pip install eiger-2.2.0-py3-none-any.whl                # default: numpy, pandas, scipy
pip install "./eiger-2.2.0-py3-none-any.whl[legacy]"    # also numba, scikit-learn
pip install ".[legacy,dev]"                                # from a checkout, with tests
```

Python 3.10 or later. The default install is enough for everything except
`--legacy`. Without the extra, `--legacy` exits with an install hint.
`environment.yml` describes an equivalent conda environment.

## Inputs

Both forms carry one row per CpG and one column per sample. Values are
methylation proportions in [0, 1]. A missing value is an empty field, `NA` or
`NaN`. The separator follows the file extension: `.csv` is comma-separated and
anything else is tab-separated, with `.gz` allowed.

**Two files.** `--betas` holds the sample columns only; its header row gives
the sample names. `--locations` gives the coordinates in the same row order,
with columns `chr`/`chrom`/`chromosome`/`seqnames`, then `start`/`pos`/`position`,
and optionally `end`/`stop`. For a headerless three-column BED, add
`--locations-no-header`.

```sh
eiger2 call --betas betas.tsv --locations cpgs.tsv --out results
```

**One table.** `--matrix` holds the coordinate columns followed by the sample
columns, as exported by methrix or bsseq. Columns named `strand` or
`cpg_index` are recognised as coordinates, not samples.

```sh
eiger2 call --matrix matrix.tsv --out results
```

Rows may be in any order. EIGER sorts them by chromosome (`chr1`…`chr22`,
`chrX`, `chrY`, `chrM`, then other contigs alphabetically) and by position.
The `cpg_index_*` output columns refer to this sorted order.

**Size requirement.** The default profile sets its threshold from CpG pairs at
least 1,000 rows apart on the same chromosome. At least one chromosome must
therefore have 3,002 or more CpGs; otherwise the run stops with an
explanation. Shorter chromosomes are still scanned, using the threshold from
the longer ones. Real WGBS or panel data meet this easily. A toy test file may
not.

**Bismark coverage files.** `eiger2.io` converts Bismark `.cov`/`.cov.gz`
files into a matrix:

```python
from eiger2.io import build_cpg_reference, convert_cov_files, load_matrix
reference = build_cpg_reference(cov_files)            # union CpG grid
cache = convert_cov_files(cov_files, reference, "cache/")
betas, depth, locations = load_matrix(cache, reference, min_depth=5)
```

## What `eiger2 call` does

1. **Scan.** For each pair of adjacent CpGs, compute a co-variation statistic
   across samples. The default is Pearson correlation averaged over a window of
   4 CpGs, using samples observed at both CpGs, with at least 5 such samples.
   An edge is eligible when both CpGs lie on the same chromosome and
   `next.start - previous.end` is between 0 and 300 bp.
2. **Candidate threshold.** The same statistic is computed on windows whose
   members lie a distant lag (at least 1,000 CpG rows) apart. Eight random
   lags are drawn per chromosome. The 0.995 quantile of these values becomes
   the scan threshold.
3. **Candidates.** Runs of eligible edges at or above the threshold, spanning
   at least 4 CpGs, are candidate regions. A region's score is
   `S = sum over its edges of (statistic − threshold)`.
4. **Permutation scans.** In each of B = 99 replicates, the sample values are
   permuted independently within every CpG row. The permuted matrix goes
   through steps 1–3 unchanged, including a new threshold.
5. **Coordination q-value.** At each distinct observed score *t*, let
   `R(t)` be the number of observed regions scoring ≥ *t* and `N_b(t)` the number
   in permutation *b*. Then

   ```text
   FDR_plus(t) = min(1, (1 + Σ_b N_b(t)) / (B · max(R(t), 1)))
   coordination_qvalue(S) = min of FDR_plus(t) over observed scores t ≤ S
   ```

   Tied scores share a q-value. The smallest possible q-value is
   `1 / (B · R)`. The added 1 prevents an estimate of zero when no permuted
   region is that strong. All candidates of one run form one decision family.
6. **Axis annotations** (secondary). See [below](#axis-columns).

Seeds make runs exactly reproducible. The observed scan uses `--random-seed`
(default 1); permutation *b* uses `--random-seed` + *b*. `--permutation-seed`
(default 1) drives the row permutations. Scanning one chromosome at a time
(the default, to bound memory) gives the same regions and selections as
scanning the whole matrix, with scores equal to floating-point rounding.

## Outputs

### `regions.csv`

One row per candidate region, in genomic order. `NA` marks a missing value.

| column | meaning |
|---|---|
| `region_id` | `EIGER2_<chr>_<start>_<end>` (`EIGER1_…` for `--legacy`); depends only on the profile and coordinates |
| `chr`, `start`, `end` | region coordinates: start of the first CpG, end of the last |
| `n_cpg` | CpGs in the region |
| `cpg_index_start`, `cpg_index_end` | first and last row in the sorted input |
| `caller` | `newer` or `legacy` |
| `region_score` | `S`, the summed excess of the statistic over the threshold |
| `statistic_mean` | mean edge statistic inside the region |
| `scan_threshold` | the observed scan's threshold |
| `coordination_qvalue` | the estimated FDR (above); `NA` with `--discovery-only` |
| `coordination_selected` | `coordination_qvalue <= --target-fdr`; `NA` with `--discovery-only` |
| `axis_status` | how far the axis analysis got for this region (table below) |
| `axis_n_axes` | genome-wide sample axes used |
| `axis_n_samples_tested` | samples observing at least half the region's CpGs |
| `axis_index` | which axis `axis_effect` refers to (0-based) |
| `axis_effect` | beta-scale p10→p90 contrast along that axis |
| `axis_effect_se` | its standard error (HC3, delta method) |
| `axis_effect_lower`, `axis_effect_upper` | its interval (95%, Bonferroni-corrected over 2 × `axis_n_axes`) |
| `axis_pvalue`, `axis_qvalue`, `axis_supported` | experimental axis test; `NA` unless requested |

### Other files

* `discoveries.csv`: the rows of `regions.csv` with `coordination_selected`
  true. Not written with `--discovery-only` or `--no-discoveries-table`.
* `permutations.csv`: one row per completed permutation scan: `replicate`,
  `internal_seed`, `scan_threshold`, `n_candidate_edges`, `n_regions` and
  `null_values` (usable distant-lag values behind that scan's threshold).
* `null_region_scores.npz`: every permutation scan's region scores. Written
  only with `--save-null-scores`.
* `run_manifest.json` (schema `eiger.region_fdr_run/1`):
  * `software_version`, `source` (git commit and dirty state when run from a
    checkout, `unknown` when installed) and `command`;
  * `inputs`: every input file with its SHA-256;
  * `resolved_configuration`, `profile`, `settings`: every setting and seed;
  * `status`: `ok`, `no_candidates` or `discovery_only`;
  * `method`: the FDR formula and version (`fdr_plus_cummin/1`);
  * `observed_scan` and `permutations`: thresholds and counts, requested and
    completed scans;
  * `threshold_ratio`: the observed threshold divided by the mean permuted
    threshold, recorded for description only;
  * `results`: candidates, selected, the retained fraction and the smallest
    q-value;
  * `axis`: mode, number of axes, status counts, reference points;
  * `timing` and `peak_rss_mb`.

A run with no candidates is a valid result: it writes a header-only
`regions.csv` and status `no_candidates`.

## Reading the results

### Coordination q-value

The permutation scans keep each CpG's own distribution of values, including
its missing values, but break every link between CpGs. The q-value therefore
estimates the FDR of calling a region coordinated when neighbouring CpGs do
not co-vary. Keep in mind:

* It concerns coordination, not cause. Genome-wide sample structure, such as
  cell composition, purity, batch or a sample subgroup, makes neighbouring
  CpGs co-vary too. Regions it produces are genuinely coordinated and can be
  selected. The axis columns help tell them apart from local events.
* Samples are treated as exchangeable within each CpG. With the default
  `--mask-mode values-with-mask`, a CpG's missing values are permuted
  together with its observed values. `mask-fixed` keeps each missing
  position in place and permutes only the observed values; it assumes
  exchangeability given the pattern of observations. The default is the
  validated setting.
* With R candidates, no q-value can be smaller than `1 / (B · R)`. When
  permuted scans produce few regions, many candidates share values near this
  floor. Increasing `--fdr-replicates` resolves small q-values more finely.
  Do not reduce it for reported results.
* The threshold, window, minimum length and gap limit were validated at their
  defaults. The options that change them are for exploration.

### Axis columns

EIGER estimates genome-wide sample axes once per run. It uses a thinned set of
CpGs, excludes every candidate region's CpGs and keeps only complete rows.
Parallel analysis chooses how many axes to keep, and only axes concentrated on
a subset of CpGs are kept. For each candidate, the logit of each sample's
region mean is regressed on the axes. `axis_effect` is the fitted change in
beta between the 10th and 90th percentile of the reported axis.

| `axis_status` | meaning |
|---|---|
| `estimated` | effect and interval computed (the default mode) |
| `no_axes` | no usable axis. Common with scattered missing values, because only complete rows are used. |
| `insufficient_samples` | fewer than max(10, axes + 10) samples observe the region |
| `fit_failed` | the regression could not be fitted |
| `not_computed` | `--axis-annotations off` |
| `below_inference_floor` | experimental mode: effect computed, test withheld below 200 tested samples |
| `tested` | experimental mode: effect and test computed |

The standard error and interval describe the contrast along the axis
estimated in this run, treating that axis and its percentiles as fixed. They
are not an interval for a population-level axis.

`--axis-annotations experimental-inference` adds `axis_pvalue`, `axis_qvalue`
and `axis_supported`. The test is of |effect| > 0.10 across both directions
of every axis, with Benjamini–Hochberg over all candidates. It runs only for
regions with at least 200 tested samples. It has been checked only in
simulations with a single retained axis and no missing values. Treat it as
experimental.

## Options

| option | default | meaning |
|---|---|---|
| `--target-fdr` | 0.05 | selection level for `coordination_qvalue` |
| `--fdr-replicates` | 99 | B, the number of permutation scans |
| `--random-seed`, `--permutation-seed` | 1, 1 | internal scan seed; row-permutation seed |
| `--mask-mode` | `values-with-mask` | or `mask-fixed` (above) |
| `--discovery-only` | off | candidates and scores only; no permutation scans, no q-values |
| `--min-sample-fraction` | keep all | drop CpGs observed in fewer than this fraction of samples |
| `--min-cpgs` | 4 | minimum region length |
| `--max-cpg-gap-bp` | 300 | largest gap between adjacent CpGs of a region |
| `--statistic`, `--window` | correlation, 4 | default profile only |
| `--null-quantile`, `--null-replicates`, `--null-min-lag` | 0.995, 8, 1000 | default profile's threshold: quantile, internal lag draws, minimum lag in rows |
| `--legacy`, `--cov-threshold`, `--no-impute` | off, 0.01, off | legacy profile and its settings |
| `--axis-annotations` | `descriptive` | or `off`, `experimental-inference` |
| `--scan-mode` | `chromosome` | or `whole` (same results, more memory) |
| `--storage-dtype` | `float64` | `float32` reduces matrix storage; peak-memory savings vary |
| `--no-discoveries-table`, `--save-null-scores`, `--quiet` | off | output and progress controls |

A profile-specific option used with the other profile is an error. Exit codes
are 0 for success (including no candidates), 1 for an input or method error
(explained on stderr) and 2 for a command-line usage error.

`eiger2 compare --regions-a A/regions.csv --regions-b B/regions.csv --out DIR`
writes the base-pair overlap of the selected regions of two runs to
`comparison_summary.csv`. Add `--all-candidates` to compare every candidate.

## Runtime and memory

A run costs B + 1 = 100 scans. The example (10,000 CpGs × 60 samples) takes
about 5 seconds. In an API benchmark, 400,000 CpGs × 100 samples took about
3.5 seconds per scan in chromosome mode with float64 storage. Peak memory was
about 1.4 GB with axis annotations off and 1.6 GB with descriptive annotations.
These measurements are workload- and machine-specific.

Peak memory can be several times the input matrix size, including parsing,
temporary arrays and axis estimation. A separate CLI benchmark at 200,000
CpGs × 100 samples with descriptive annotations peaked at about 921 MB for
float64 and 813 MB for float32. `--storage-dtype float32` reduces storage for
the analysed matrix and permutation buffer, but parsing and axis estimation
still allocate float64 arrays; lower peak memory is not guaranteed. Measure a
representative subset before sizing a whole-genome job. Several million CpGs
and hundreds of samples take hours. Permutation scans run serially. Progress
and an estimate of the remaining time go to stderr.

## Python API

```python
from eiger2 import call_coordinated_regions, RegionFdrConfig, LEGACY

run = call_coordinated_regions(betas, locations, sample_names=names,
                               config=RegionFdrConfig())
run.regions        # the regions.csv table
run.discoveries    # the selected subset
run.permutations   # the permutations.csv table
run.manifest       # the run's manifest (without the CLI's file checksums)

legacy = call_coordinated_regions(betas, locations,
                                  config=RegionFdrConfig(profile=LEGACY))
```

`betas` is a CpG-by-sample array and `locations` a table with `chr`, `start`
and `end`. `RegionFdrConfig` takes the same settings as the command line.
`coordination_fdr(observed_scores, null_scores_per_permutation)` computes
q-values from region scores, and `annotate_axes(betas, regions)` annotates a
fixed set of regions given as inclusive CpG row ranges.

## Legacy caller

`--legacy` computes the covariance of adjacent CpGs with the original 2023
code, after filling missing values with each CpG's mean (0.5 for a CpG with
no observations). It uses a fixed threshold of 0.01 (`--cov-threshold`). The
region geometry, score and permutation FDR are the same as the default
profile's. A fixed covariance threshold depends on the scale of the data;
the default profile's threshold does not.

## Migrating from earlier versions

EIGER 2.2 replaces the earlier discovery pipeline (`call_dmrs`, region-level
lag/member-step nulls and dip-test confirmation) with the permutation FDR
above. To reproduce earlier results, install the earlier release (for
example, tag `v2.1.1`).

* **Output columns.** `coordination_qvalue` and `coordination_selected`
  replace `region_qvalue` and `significant`, which are no longer written.
  `eiger2 compare` still reads `significant` from old files. Confirmation,
  biological-support, posterior, delta and prevalence columns are gone. The
  axis columns are new.
* **Options.** These now fail with a message saying why:
  `--null-mode`, `--region-null-mode`, `--n-lag-draws`, `--edge-threshold`,
  `--support-binned-threshold`, `--support-bins`, `--fdr-decision`,
  `--no-confirm`, `--biological-support-policy`, `--min-prevalence-samples`,
  `--compute-posterior`, `--delta-threshold` and `--max-gap-edges`.
* **Changed meaning.** `--fdr-replicates` is now the number of permutation
  scans for both profiles (default 99). `--null-replicates` defaults to 8.
  `--legacy` now uses threshold 0.01, a 4-CpG minimum and the permutation FDR;
  it no longer applies the old delta or prevalence criteria.
* **Behaviour.** No automatic deflation of latent factors. No
  length-stratified FDR. The manifest schema is `eiger.region_fdr_run/1`.
* **Python API.** `eiger2` now exports only the names listed above. The
  modules `eiger2.benchmark`, `eiger2.probability`, `eiger2.confirm`,
  `eiger2.layouts` and `eiger2.manifest_utils` are removed. So are `call_dmrs`,
  `region_fdr` and `deflate`.
* **Dependencies.** `diptest` is no longer required.
