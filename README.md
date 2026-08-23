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

The generated example is 50,000 CpGs on one chromosome. It is below the preferred ~103,000-CpG member-step size but above the exact minimum detectable-region floor, so it exercises the measured short-subtrack fallback while retaining the configured locality-breaking lag. Inspect the manifest's preferred/fallback diagnostics and per-length null opportunity before interpreting its q-values.

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
| **A locality-breaking edge threshold** | CpGs are paired with distant sites on the same chromosome to decide which individual edges count as elevated. This preserves cross-sample structure while removing local genomic adjacency. |
| **A separate region-level null** | Deciding whether a *run* of elevated edges is longer or stronger than chance requires a different null from the one that sets the edge threshold. See "The region-level null" below. |
| **Region-level empirical FDR estimation** | Null coordination tracks are processed through the same segmentation procedure as the observed data. Region areas are compared within CpG-count strata, giving an empirical region-level false discovery rate with explicit finite-null uncertainty bounds. |

The validation analyses, assumptions, and commands underlying these changes are documented in **[docs/](docs/)** and **[analysis/](analysis/README.md)**.

---

## The region-level null

The edge threshold and the region-level FDR answer different questions and need different nulls. Getting this wrong was the substance of the v2.1.0 release.

A sliding-window co-variation statistic **manufactures autocorrelation**. The window at edge *e* and the window at edge *e+k* share CpG pairs whenever `|k| < w`, so the observed statistic track carries a lag-1 autocorrelation of about **+0.5 under a complete null with no biology at all** — in the same way that a moving average of white noise is autocorrelated.

Any null used to calibrate *runs* of elevated edges must reproduce that autocorrelation, because run lengths depend on it. A null that does not will essentially never form multi-edge regions, and every observed run will then look impossible.

EIGER2 uses a **member-step lag null** (`region_null_mode="member_step"`, the default). Consecutive null positions share *w*−1 of their *w* member CpGs, matching the observed window-overlap mechanism while the members themselves remain at least `null_min_lag` CpGs apart. Observed-versus-null ACF is retained as a calibration diagnostic: real coordinated regions can raise the observed ACF, so it is not a hard gate on a signal-rich cohort.

### Chromosome-aware null construction

Both the edge-level ("lag") and region-level ("member_step") nulls draw their locality-breaking lag **independently per chromosome**, not once genome-wide. A single genome-wide lag draw is far more likely to survive on the largest chromosome(s) than a smaller one -- a lag long enough to break locality genome-wide rarely also fits inside a short chromosome -- so the null ends up drawn almost entirely from the largest chromosomes even when the observed data spans every one of them. Each replicate now iterates every chromosome, draws that chromosome's own feasible lag (`U[null_min_lag, n_c // (span - 1)]`, where `n_c` is that chromosome's own CpG count), and concatenates every chromosome's own segment into one genome-wide realisation, with the eligibility mask broken at every chromosome boundary. This preserves chromosome composition in every replicate, and makes each replicate a clean cluster unit.

For the region null, `min_subtrack_length=100` is preferred rather than an all-or-nothing chromosome floor. A chromosome that cannot support 100-position subtracks may use a measured fallback down to the configured minimum detectable region length (`min_cpgs`), without changing the `null_min_lag=1000` locality-breaking floor. The per-length opportunity guard decides which length strata that shorter chromosome can support. Only chromosomes unable to support even `min_cpgs` at the minimum lag are absent and counted in `region_null_chromosome_exposure`. The manifest distinguishes preferred, fallback and truly infeasible chromosomes, realised subtrack lengths and the fraction of null positions used as forced boundaries.

Two gates run automatically on every call and are recorded in `run_manifest.json`:

- **autocorrelation match** — the null's autocorrelation function must lie within ±0.05 of the observed track's at lags 1…*w*;
- **region-level exchangeability** — observed and null region-formation rates per unit exposure must be comparable.

Both are reported in `manifest["gates"]` whether they pass or fail. **These are calibration
diagnostics, not gates**: they are confounded with genuine signal on real data (a cohort
containing real coordinated regions is *expected* to show elevated observed/null ratios --
that is what detection means), so they change only which status label a run reports
(`ok` vs. `ok_with_warnings`), never whether `region_qvalue`/`significant` are suppressed.
Full derivation is in **[docs/FDR.md](docs/FDR.md)**.

### Structural gates and run status

A separate set of **structural gates** -- signal-independent conditions on the null's
construction, checked regardless of whether the observed data contains any signal at all --
can fail closed. `manifest["status"]` is one of four values:

| status | meaning |
|---|---|
| `ok` | no structural failure, no calibration-diagnostic concern |
| `ok_with_warnings` | a calibration diagnostic above is out of its expected range; `region_qvalue`/`significant` unaffected |
| `partial_inference` | one or more length strata lacked enough null OPPORTUNITY to report `region_qvalue` (below `m_opportunity`, spec §2c); those strata's rows are refused (`NaN`/`pd.NA`), every other stratum reports normally -- the null is otherwise sound |
| `inference_unreliable` | a structural gate failed -- `region_qvalue`, its bounds, and `significant` are refused (`NaN`/`pd.NA`) for the whole run; candidates and descriptive columns are still written |

The structural gates, recorded in `manifest["gates"]["structural"]` with their measured
values: surviving null-track fraction (below 50% of `null_replicates`), null coverage
(below `min_null_coverage`), pooled null volume (below `min_null_values`), chromosome
exposure with no feasible null lag (above `max_missing_null_exposure`, default 5%), and numerical
failure in the null pipeline. `eiger2 call` exits non-zero, with a summary naming every
failed gate on stderr, exactly when `status == "inference_unreliable"`. A per-stratum
`partial_inference` refusal is reported (`manifest["fdr"]["insufficient_null_opportunity_strata"]`,
each entry naming its observed/null opportunity and the opportunity required) but does not
affect the exit code -- every other stratum's `region_qvalue` is unaffected and real.

### Minimum input size, and automatic fallback

The preferred 100-position member-step construction needs about **103,000 CpGs on a chromosome** at default settings. Shorter chromosomes now use the fallback above. The absolute construction floor is derived from the minimum detectable region: `null_min_lag × (min_cpgs + window − 1)`, **7,000 CpGs** at the defaults. Below that, `region_null_mode="auto"` falls back to the older `"lag"` construction and prints a warning.

Pass `--region-null-mode member_step` to raise an error instead of falling back, or `--region-null-mode off` to skip `region_qvalue` entirely and report candidate regions only.

---

## Error rate: conservative, not calibrated

Pooled across every measured planted-signal benchmark cell — **293 significant calls, 1 false positive** — the realised false discovery rate is:

| | value |
|---|---|
| point estimate | 0.34% |
| one-sided 95% Clopper–Pearson upper bound | **1.6%** |
| P(observing ≤ 1 false positive if the true FDR were 5%) | 4.9 × 10⁻⁶ |

At a nominal `target_fdr = 0.05`, EIGER2 v2.1.0 is therefore **conservative rather than calibrated**: the measured error rate sits well below the target, which means some detection power is being left unused. This is stated as measured rather than claimed as calibration — the benchmark's planted signal is comfortably detectable, so these cells cannot distinguish "controls FDR at exactly 5%" from "controls it at well under 5%".

The earlier v2.0.1 finding of substantial FDR miscalibration under structured missingness, small sample size, and high noise **does not survive the corrected region-level null**. Re-measured at genome scale on real hg38 CpG layouts under 25% structured dropout: realised FDR 0.0000 and recall 1.000 on 6 of 6 instances. That miscalibration was downstream of a region-level null that formed essentially no regions, not an independent problem.

### Per-region evidence score

The per-region evidence score is intended for ranking regions. It is **not** calibrated as a posterior probability or as the probability that an individual region is biologically real.

The analyses used to assess this behaviour are described in **[docs/PROBABILITY.md](docs/PROBABILITY.md)**.

---

## Large cohorts

Earlier versions recommended `--null-mode permutation` above roughly 100 samples, because the default null's q-values saturated at zero on large cohorts — in a 519-sample cohort, all 73,784 reported regions received q = 0.

**That recommendation has been withdrawn and `--null-mode permutation` now raises an error.** The saturation was a symptom of the same region-level null defect described above, not a property of large cohorts, and the permutation null was independently found to be anti-conservative: it permutes edge statistics individually, destroying the window-induced autocorrelation, so it forms no null regions at all (measured: zero across 400 replicates) and every observed run appears impossible.

Re-examined under the member-step null at *n* = 500, q-values are finite and non-degenerate (0.27–0.54 on nuisance-only data, none at exactly zero). No cohort-size-specific handling is required; the default is appropriate at any scale above the minimum input size.

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

> **`--legacy --fdr-replicates` is no longer available in v2.1.0.** It applied permutation-based FDR estimation to the legacy covariance track, and the permutation null it depended on has been removed for the reasons given under "Large cohorts". The flag now raises an error naming the replacement. The legacy caller itself is unaffected and still runs exactly as published.

---

## Repository structure

```text
Python/eiger2/        Installable EIGER2 package, including the core caller,
                      command-line interface, input/output functions,
                      probability utilities, layout diagnostics, and the
                      validation simulator

Python/functions/     Original EIGER implementation used by --legacy

benchmarking/         Benchmarking and validation framework, the real hg38 CpG
                      layout fixture, and generated benchmark outputs

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

At the time of writing (v2.1.0), the current package contains 234 tests and the superseded beta implementation contains 27 tests.

---

## Outputs

> **Migration note (v2.1.0):** results from v2.0.0 and v2.0.1 are **void and must be regenerated.** The region-level null was replaced, which changes the edge threshold, the set of candidate regions, their areas, and every FDR column. Region *identifiers* are therefore not comparable across versions either. The edge statistic itself, the eligibility mask, and the latent-structure removal are unchanged.

The principal region-level output is `regions.csv`, with one row per detected region.

**`region_qvalue` is the principal field used to assess region-level statistical support.**

Other fields, including methylation difference, direction, and inferred sample grouping, are descriptive. These values are calculated after region selection using the same data and should not be treated as independent statistical evidence or used as additional significance filters without further validation.

### FDR resolution and Monte Carlo bounds

Six columns sit immediately after `region_qvalue`. Full derivation and worked examples are in [docs/FDR.md](docs/FDR.md); summary:

| Column | Meaning |
|---|---|
| `region_tail_fdr` | point FDR at the region's own area cutoff |
| `region_qvalue_lower_95` | simultaneous replicate-cluster quasi-Poisson lower bound |
| `region_qvalue_upper_95` | simultaneous replicate-cluster quasi-Poisson upper bound; finite and positive when the point estimate is zero |
| `region_qvalue_resolution` | one scaled null exceedance's worth of change at the q-achieving cutoff |
| `null_exceedances_at_cutoff` | pooled null exceedance count at the q-achieving cutoff (`-1` if unavailable) |
| `region_qvalue_cutoff_rank` | number of observed regions at least as strong as the q-achieving cutoff (`-1` if unavailable) |

A `region_qvalue` of exactly zero means the null produced no region at least as extreme, not that the region is certain. Read it together with `region_qvalue_upper_95` and `region_qvalue_resolution`, which say how finely the available null can resolve that statement.

`schema_version` in the run manifest is `"2.3"`. `significant` is a nullable boolean
(pandas `BooleanDtype`): `pd.NA` means inference was unavailable for that row, a
different statement from `False` ("tested, not significant") -- see `fdr_status` and
`docs/FDR.md` §14.

### `--fdr-decision {point,upper}`

```bash
eiger2 call --betas betas.tsv --locations cpgs.bed --out results/ \
    --fdr-decision upper
```

`point` (default) calls a region significant when `region_qvalue <= target_fdr`. `upper` instead requires `region_qvalue_upper_95 <= target_fdr`. The upper bound uses each independently drawn genome-wide null realisation as one cluster, estimates exposure-adjusted replicate dispersion, and applies a Bonferroni band across the complete cutoff curve within each stratum. It is model-based (quasi-Poisson), not an assumption-free guarantee; its assumptions and simulation coverage checks are documented in `docs/FDR.md`. A valid zero-rejection result is returned normally and described in `manifest["fdr"]["upper_decision_diagnostic"]`; strata are refused only when the null opportunity or independent-cluster count is inadequate.

### Null adequacy

If a length stratum has inadequate null opportunity, or there are fewer than five independent region-null realisations with positive exposure, EIGER2 **refuses to report a q-value**. The affected regions receive `NaN` in every inferential column, `-1` in every count column, nullable `significant = pd.NA`, and an entry in `manifest["fdr"]` naming the shortfall. Ordinary length strata scale expected null-region counts by their own observed/null opportunity ratio; joint length×support strata currently retain a prominently diagnosed global-edge fallback because matching joint opportunity is not yet measured.

A stratum in which the null produced *no* region, while the run as a whole is adequate, is not a shortfall — it is strong evidence, and it is reported as `q = 0` with a finite upper bound.

### Runtime controls

| Setting | Default | Effect |
|---|---|---|
| `--null-replicates` | 100 | Edge-threshold null replicates. |
| `--n-lag-draws` / `n_lag_draws` | 75 | Member-step region-null draws. Runtime scales roughly linearly; resolution and per-stratum opportunity improve as it rises. At 75, `s ≈ 0.019`; at 150, `s ≈ 0.010`. Every region gate passes across 40–150 in benchmarking, so 75 is the runtime/resolution default rather than the finest available. |
| `EIGER2_USE_NUMBA=1` | unset | Opt-in fused numba kernel for the pair statistic, available with `pip install eiger[fast]`. Off by default because it changes results in the last floating-point digits. |

Each run also produces `run_manifest.json`, which records:

- the resolved analysis configuration
- input file checksums
- software and package versions
- random seeds
- null-model diagnostics and both region-level gate results
- layout diagnostics, including the longest detectable region on the given CpG layout
- the parameters used for segmentation and FDR estimation
- `chromosome_order`: the canonical order (`chr1`...`chr22`, `chrX`, `chrY`, `chrM`, then
  any other contig, lexicographically) the run's chromosomes were grouped and sorted
  into internally -- input rows are accepted in any order (natural, lexicographic, or
  shuffled) and always canonicalise to this same order, so a run is reproducible
  regardless of how the input happened to be presented

This allows each result to be traced back to the code, inputs, and settings that generated it.

---

## Licence and citation

EIGER is released under the MIT licence. See **[LICENSE](LICENSE)** for details.

Citation metadata are provided in **[CITATION.cff](CITATION.cff)**.
