# Quickstart

Five minutes from a clean machine to a called region set.

---

## 1. Install

### Recommended: conda for the dependencies, pip for the package

`environment.yml` pins every dependency, including the ones only the legacy
caller needs. Create the environment first, then install EIGER into it. The
`pip install -e .` step adds no new dependencies — everything is already
present — it only registers the package and the `eiger2` command.

```bash
git clone https://github.com/<user>/EIGER.git
cd EIGER

conda env create -f environment.yml     # or: mamba env create -f environment.yml
conda activate eiger

pip install -e . --no-deps              # --no-deps: conda already supplied them
```

Check it:

```bash
eiger2 --version
python -c "import eiger2; print(eiger2.__version__)"
pytest -q                               # 86 tests, ~9 s
```

### Alternative: pip only

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[legacy,dev]"
```

`.[legacy]` adds `numba` and `scikit-learn`, which **only the `--legacy` caller
needs**. The revised caller depends on `numpy`, `pandas` and `scipy` alone, so
`pip install -e .` is sufficient if you do not need to run the original method.

### Without installing at all

Everything works from `Python/` without installation:

```bash
cd Python
python -m eiger2.cli call --betas ... --locations ... --out ...
```

The only thing you lose is the short `eiger2` command.

---

## 2. Make some example data

```bash
python examples/make_example_data.py
```

Writes `examples/data/`: a 50,000 CpG × 24 sample matrix on one chromosome,
with spatially autocorrelated background, two latent nuisance factors, read
depth 15, 5% missing observations, and 20 planted regions spanning a grid of
effect sizes and prevalences. `truth.tsv` records what was planted. 50,000
CpGs is below `region_null_mode="member_step"`'s ~103,000-CpG floor, so
`eiger2 call` on this data demonstrates the automatic `"lag"` fallback (see
the README's quick-start note and `examples/make_example_data.py`'s
docstring for why the run still reports `inference_unreliable`).

---

## 3. Run it

```bash
# the revised caller
eiger2 call --betas examples/data/betas.tsv \
            --locations examples/data/cpgs.bed \
            --out examples/out_eiger2

# the method as originally published
eiger2 call --betas examples/data/betas.tsv \
            --locations examples/data/cpgs.bed \
            --out examples/out_legacy --legacy

# both, with an overlap summary
eiger2 compare --betas examples/data/betas.tsv \
               --locations examples/data/cpgs.bed \
               --out examples/out_compare
```

Expected output for the first command:

```
[eiger2] region null: region_null_mode='auto', input has 50,000 CpGs, member_step needs >= 103,000
[eiger2] caller     : eiger2
[eiger2] regions    : 18
[eiger2] significant: 0 at q <= 0.05
[eiger2] status      : inference_unreliable
[eiger2] written to : examples/out_eiger2
```

`regions` lists 18 candidates, but `status: inference_unreliable` -- expected on
this fixture, and explained by the warnings the run also prints (see the
quick-start note above and `examples/make_example_data.py`'s docstring):
`"lag"`'s region-null essentially never forms a multi-CpG run on autocorrelated
data, so `region_qvalue` and `significant` are withheld for the whole run rather
than a fabricated q-value being reported. `eiger2 call` exits 1 in this
situation; the exit code is not something to work around for this fixture.

---

## 4. Inputs

There are two ways to supply data. Both accept TSV (default), CSV (`.csv`) and
gzip (`.gz`).

### One file: coordinates and samples together (recommended)

The methrix / bsseq layout. Leading columns are coordinates, every other column
is a sample:

```
chr      start     strand   Sample1   Sample2   Sample3
chr1     10469     +        0.812     0.774     NA
chr1     10471     +        0.795     0.801     0.788
```

```bash
eiger2 call --matrix methrix_betas.tsv --out results/
```

Recognised coordinate columns, in any case: `chr` / `chrom` / `chromosome` /
`seqnames`; `start` / `pos` / `position`; optionally `end` / `stop`; and
`strand`, `cpg_index`, `index`, which are read and discarded. **Every remaining
column is treated as a sample**, so remove any annotation columns first. Where
no end coordinate is given each CpG becomes a one-base interval.

### Two files: matrix and coordinates separately

```bash
eiger2 call --betas betas.tsv --locations cpgs.bed --out results/
```

`--locations` accepts named columns as above, or a **headerless three-column
BED** with `--locations-no-header`. Both the `chr start end` and the wgbstools
`chr position CpG_index` layouts are handled; a CpG index is detected by being
smaller than the position it would otherwise end, and is discarded.

```bash
# wgbstools CpG.bed.gz, which has no header row
eiger2 call --betas betas.tsv --locations CpG.bed.gz --locations-no-header \
            --out results/
```

### Requirements common to both

- Methylation as **proportions in [0, 1]**, not percentages.
- Missing values as empty fields, `NA` or `NaN`.
- Rows sorted by chromosome then position, and the matrix in the same order.

The CLI checks the row counts, the value range and the sort order, and refuses
with an explanatory message rather than producing wrong output.

### Targeted panels and other sparse assays

**Supply only the CpGs you have.** EIGER does not need, and should not be given,
a whole-genome scaffold padded with missing rows. Segmentation runs along row
order and `--max-cpg-gap-bp` breaks a region wherever consecutive coordinates
jump further apart than that, so untargeted stretches of genome are handled by
their absence. Padding to the whole genome multiplies the work with rows that
cannot contribute an edge.

Two settings matter on panel data:

```bash
# drop CpGs observed in under half the samples, and set the gap that separates
# one target interval from the next to about twice your within-target spacing
eiger2 call --matrix panel.tsv --out results/ \
    --min-sample-fraction 0.5 \
    --max-cpg-gap-bp 500
```

`--min-sample-fraction` is both a speed and a correctness control. On a
panel-like matrix it cut runtime 3.3-fold *and raised the number of regions
found from 8 to 80*, because sparsely covered CpGs sit between well-covered ones
and fragment genuine regions below `--min-cpgs`. Nothing is imputed where
nothing was measured: dropped CpGs are removed before either caller runs, and
region coordinates are reported against the original input rows.

### Other formats

`.beta` / `.lbeta` and Bismark `.cov` / `.cov.gz` conversion lives in
`eiger2.io` (`build_cpg_reference`, `convert_cov_files`, `load_matrix`). A
first-class `--cov` loader that builds the matrix directly from per-sample
coverage files is not yet wired into the CLI.

---

## 5. Outputs

> **Migration note (v2.0.1):** corrected q-values differ from v2.0.0 even though
> candidate region boundaries are unchanged. Results generated with v2.0.0 must be
> regenerated before use — the direction of the region-level FDR's monotone step was
> reversed, which under-called false discoveries in v2.0.0 (see
> [docs/FDR.md](docs/FDR.md)).
>
> **Known limitation:** realised FDR control has been found to be substantially
> miscalibrated under high-missingness, small-sample, and high-noise conditions, even
> with the corrected q-values above (see [docs/FDR.md](docs/FDR.md) and the README's
> "Important limitations" section). Treat `region_qvalue`/`significant` with caution
> in these regimes until this is investigated further.

| File | Contents |
|---|---|
| `regions.csv` | one row per region: coordinates, `n_cpg`, `region_area`, `region_qvalue` and six related FDR columns (new in v2.0.1, see below), `significant`, and `*_descriptive` characterisation fields |
| `region_sample_means.csv` | per-region mean methylation for each sample |
| `run_manifest.json` | resolved configuration, input checksums, package versions, runtime, `schema_version`, `fdr_decision`, and the realised null diagnostics |

**`region_qvalue` is the principal field that expresses confidence that a region is
real.** Every `*_descriptive` field — delta beta, direction, the inferred sample
split — is computed *after* the region was selected, on the same data, and is
for reading rather than filtering.

In `--legacy` mode `region_qvalue` is `NA` by construction: the published method
defines no false discovery rate. That is the point of the comparison.

### New in v2.0.1: six columns alongside `region_qvalue`

| Column | Meaning |
|---|---|
| `region_tail_fdr` | point FDR at the region's own area cutoff |
| `region_qvalue_lower_95` | simultaneous Garwood lower Monte Carlo bound |
| `region_qvalue_upper_95` | simultaneous Garwood upper bound; nonzero whenever the point estimate is zero |
| `region_qvalue_resolution` | one scaled null exceedance's worth of change at the q-achieving cutoff |
| `null_exceedances_at_cutoff` | pooled null exceedance count at the q-achieving cutoff (`-1` if unavailable) |
| `region_qvalue_cutoff_rank` | number of observed regions at least as strong as the q-achieving cutoff (`-1` if unavailable) |

See [docs/FDR.md](docs/FDR.md) for the full derivation, the Garwood interval's Poisson
assumption, and the coverage guarantees (per-stratum, not genome-wide simultaneous).

### New in v2.2: two columns alongside `statistic_threshold`

| Column | Meaning |
|---|---|
| `region_support_median` | median per-edge measurement support (samples observed at every CpG) across the region's edges |
| `region_threshold_mode` | `"single"` or `"support_binned"`, depending on whether `--support-binned-threshold` was active for this region |

---

## 6. Options worth knowing

```bash
--null-mode lag                      # edge-level null; the only supported value
                                      # (see below -- large-cohort saturation no
                                      # longer needs a different null-mode)
--region-null-mode member_step|lag|off  # region-level null (see below); if
                                      # unset (the default), resolves
                                      # automatically to member_step when the
                                      # input is large enough, else falls back
                                      # to lag with a warning
--min-prevalence-samples 3           # a region must be carried by >= 3 samples
--statistic correlation|covariance   # default correlation; see README
--window 4                           # CpGs per window; use 1 above ~30 samples
--min-cpgs 4                         # minimum region length
--null-replicates 100                # default since v2.0.1 (was 20); raise further
                                      # for --compute-posterior (100-300)
--fdr-decision point|upper           # default point; upper decides on
                                      # region_qvalue_upper_95 instead (conservative)
--support-binned-threshold auto|on|off  # condition the null-quantile threshold on
                                      # local measurement support; default auto
                                      # enables it only when support varies enough
                                      # to matter
--support-bins N                     # number of support quantile bins when
                                      # support-binned thresholding is active
                                      # (default 4, minimum 2)
--target-fdr 0.05
--compute-posterior                  # per-region evidence SCORE, not a probability
```

### `--null-replicates` default change (v2.0.1) and its runtime cost

`--null-replicates` now defaults to **100** (previously 20). This gives a finer
resolution at the significance boundary and a materially narrower
`region_qvalue_upper_95` when the point estimate is zero (see
[docs/FDR.md](docs/FDR.md)). It costs runtime roughly linearly in replicate count.
Measured on a 200-sample, `dmr_length=20` synthetic input:

- **20 replicates: 0.197 s**
- **100 replicates: 0.805 s** (~4.1x)

Pass `--null-replicates 20` explicitly to restore the old default when runtime matters
more than resolution.

### `--fdr-decision point|upper`

`point` (default) calls a region significant when `region_qvalue <= target-fdr`,
unchanged in spirit from v2.0.0. `upper` instead requires
`region_qvalue_upper_95 <= target-fdr` — the conservative decision under finite-null
estimation uncertainty. Use `upper` when few null replicates are available and a point
estimate of exactly zero should not, on its own, be trusted as a call. If the current
replicate count cannot achieve the target even in the best case, `upper` refuses to
run rather than silently returning zero significant regions; the error names the
replicate count required.

### Choosing a null: `--null-mode` and `--region-null-mode`

`--null-mode lag` (the only supported value; `permutation` was removed in v2.1.0,
below) pairs CpGs far apart on the same chromosome for the **edge-level** null --
which edges count as elevated. Region-level FDR (which *runs* of elevated edges are
significant) is a separate null, `--region-null-mode` (Stage 1f, pre-release).

**Default: leave it unset.** With no `--region-null-mode` flag, the resolved
behaviour is `"auto"`: `member_step` (below) is used when the input clears its
genome-scale minimum size, and the CLI falls back to `lag` automatically --
printing a loud warning naming the input size, the required size, and that `lag`'s
q-values are not calibrated -- when it does not. This is what a CLI user hitting a
small test file, a targeted panel, or a single-chromosome subset now gets instead
of a hard, uncatchable error (the gap flagged in an earlier release note, now
fixed). `eiger2 call`'s startup output always reports the computed floor and the
input's CpG count before the run, e.g. `region_null_mode='auto', input has 50,000
CpGs, member_step needs >= 103,000`.

`member_step` (the resolved default on large-enough input) is what actually fixed
large-cohort saturation: on a 519-sample cohort, the pre-v2.1.0 default assigned
q = 0 to all 73,784 reported regions -- arithmetically correct given a near-empty
null, but useless. A simulated 500-sample cohort under the current default
produces finite, non-saturating q-values instead (measured minimum 0.27, zero
exact-zeros; see `docs/FDR.md` §11-§12).

`--region-null-mode member_step` (passed explicitly) skips the fallback: on input
below the floor it raises, naming the required size, rather than silently using
`lag` -- for a caller who wants to know for certain which construction ran.
`--region-null-mode lag` (reproduces the pre-v2.1.0 region-level null, including
the saturation above) is for small inputs below the floor
(`null_min_lag * (min_subtrack_length + window - 1)`, 103,000 CpGs at shipped
defaults) where you specifically want `lag`'s behaviour rather than the automatic
fallback, or for reproducing historical comparisons -- not recommended for
inference on real-sized data. `--region-null-mode off` skips the region-level null
entirely (candidates only, no `region_qvalue`).

**`--null-mode permutation` was removed in v2.1.0.** It was edge-wise permutation
(block length 1), which destroys the same run-length autocorrelation region-level FDR
depends on and was measured to produce zero usable null regions; requesting it now
raises immediately, naming `region_null_mode="member_step"` as the replacement.

Both edge-level nulls are sign-agnostic: hypo- and hypermethylated regions produce the
same positive adjacent correlation, and direction is recovered afterwards from delta
beta.

### A prevalence floor: `--min-prevalence-samples`

A single aberrant sample deviating across a run of CpGs produces genuine
coordination, so it forms a region and no null will reject it. It is simply not
a differentially methylated *region* in the usual sense. On the 519-sample ICGC
cohort, 95% of otherwise unclassifiable regions were driven by one or two
samples.

This is a biological criterion, not a statistical one, so it is applied after
calibration and reported separately: `region_qvalue` stays calibrated on the
unfiltered set, `minority_size_descriptive` records how many samples carry the
deviation, and `significant` requires both. It is exactly analogous to requiring
a fold-change floor alongside a p-value.

### `--legacy --fdr-replicates` no longer runs (removed in v2.1.0)

This used to attach a region-level empirical FDR to the originally published caller
by permuting **its own** covariance track and segmenting the result identically --
the same edge-wise permutation null removed from `--null-mode` above. It now raises
immediately, naming `region_null_mode="member_step"` as the replacement, rather than
silently returning q-values from a null measured to contain zero usable regions.
`eiger2.core.qvalues_from_track`/`qvalue_result_from_track` raise the same way for any
other caller that reaches them. `--legacy` without `--fdr-replicates` is unaffected --
it still reports the published method's regions with `region_qvalue` written as `NA`.

---

## 7. Reproducibility

Every run writes `run_manifest.json` containing the exact resolved
configuration, SHA-256 prefixes of both input files, package versions and
platform. Results are deterministic given `--random-seed`. Quote the manifest
when reporting a result.
