<img width="1408" height="768" alt="EIGER banner" src="https://github.com/user-attachments/assets/32576634-76bb-40e6-9a16-bf4ae769dd5c" />

# EIGER: Estimating Interdependent Genome-wide Epigenetic Regions

**Unsupervised discovery of coordinated DNA methylation regions, reported as a ranked candidate list.**

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

The generated example is 50,000 CpGs on one chromosome. The default run is a
**discovery** run: it reports candidate regions ranked by `region_area`, and no
q-value.

```
rank,region_id,chr,start,end,n_cpg,...,region_area,region_qvalue,...,fdr_status
1,EIGER2_chr1_511820_512241,chr1,511820,512241,22,...,6.195,,...,no_calibrated_null
2,EIGER2_chr1_650200_650421,chr1,650200,650421,12,...,3.348,,...,no_calibrated_null
```

`rank` orders regions by area within chromosome order; `region_qvalue` and
`significant` are empty, and `fdr_status` says why. Alongside them,
`confirmed`, `confirm_qvalue`, `delta_hat`, `f_hat` and `direction` report
tier two: whether that region's own per-sample means separate into more than
one population, and by how much ([docs/CONFIRM.md](docs/CONFIRM.md)). **Region FDR against the
genome background requires matched control matrices**, and the path that
consumes them is specified but not implemented in this release -- see
[docs/FDR.md](docs/FDR.md) section 18 for the measurements behind that default.

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
| **A null-anchored edge threshold** | An edge counts as elevated when it exceeds the 99.5th percentile of the locality-breaking null: the same statistic between CpGs at least 1,000 rows apart, so the threshold is anchored to this cohort's cross-sample structure and makes no assumption that signal is sparse (`--edge-threshold lag_null`, the v2.2 rule). The alternative `observed_quantile` keeps a fixed fraction of the genome's own edges within gap and support bins; it is cheap but on a cohort with pervasive signal it discards most of it (on the 519-sample ICGC cohort it kept 1,915 of 81,979 candidates). See [docs/FDR.md](docs/FDR.md) section 18. |
| **Ranked candidates by default** | Runs of elevated edges become regions, and regions are ranked by `region_area`. Region FDR against the genome background is available only when matched control matrices are supplied: every self-calibrated background null measured on WGBS-realistic data failed. |
| **Per-region confirmation (tier two)** | Each candidate is separately tested for whether its own per-sample mean betas come from one population (Hartigan's dip test, BH across the run's candidates), with the effect size, prevalence and direction of the split. That null is a property of the region, not of the genome background, so it needs no controls. See [docs/CONFIRM.md](docs/CONFIRM.md). |

The assumptions and commands underlying these changes are documented in **[docs/](docs/)**. The validation analyses themselves live in the separate `eiger_paper` repository, which is not part of this package: see its **[analysis/README.md](https://github.com/tobymilnec/eiger_paper/blob/d83f61decfa624cbc079c38f38e3dd480fee322b/analysis/README.md)**.

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
can fail closed. `manifest["status"]` is one of five values:

| status | meaning |
|---|---|
| `discovery_only` | **the default.** No region null was built, so `region_qvalue`, its bounds and `significant` are refused (`NaN`/`pd.NA`) with `fdr_status = "no_calibrated_null"`, and the regions are ranked candidates. Nothing failed; nothing was claimed |
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

The preferred 100-position member-step construction needs about **103,000 CpGs on a chromosome** at default settings. Shorter chromosomes now use the fallback above. The absolute construction floor is derived from the minimum detectable region: `null_min_lag × (min_cpgs + window − 1)`, **7,000 CpGs** at the defaults. Both matter only when a region null is requested at all.

Since v2.3 `--region-null-mode` defaults to `off`, and `"auto"` has been removed: it
resolved silently to a null now known to be misspecified on WGBS backgrounds.
`--region-null-mode member_step` and `--region-null-mode lag` remain available by
explicit request, both with a warning naming [docs/FDR.md](docs/FDR.md) section 18.

---

## Error rate varies by regime

The earlier easy-signal grid produced 293 calls with one false positive (0.34%
pooled false calls), but it did not probe the detection boundary.

Stage 2e extended the test to ~600,000-CpG genomes. Null-construction diagnostics
passed in all 1,000 calibration runs. In the planted-signal grid, the mean per-run
false discovery proportion was 0.8-4.1% in the four 10-CpG cells, but rose to 6.9%,
15.4%, and 11.3% in the weak-effect 4-CpG cells. Across all 280 runs it averaged
6.2%. The often-quoted 4.4% value is the call-weighted ratio (90 false / 2,051
calls), not the FDR estimator, and it masks the short-region failure mode.

These results are a substantial improvement over the earlier null, but they do not
establish uniform 5% FDR control. Treat weak regions near the `min_cpgs` floor with
caution, especially in small cohorts. Full results and methodology are in
**[docs/FDR.md](docs/FDR.md)**.

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

examples/             Example dataset generator and runnable input files

tests/                Tests for the current installable package

docs/                 Method design and statistical notes

tools/                Small helper scripts used by the `make` targets

Makefile              Software development task interface (`make help`)
```

Only `Python/eiger2` and `Python/functions` are installed as package code. The
paper analyses, benchmark harnesses, large fixtures, and generated evidence are
maintained separately in the `eiger_paper` repository.

```bash
pytest tests -q
```

The CI workflow runs this software-only suite on every supported Python version.

### Development

The root `Makefile` collects the local software tasks. `make` on its own only
prints help; it installs nothing and writes nothing.

```bash
make install-dev    # pip install -e ".[legacy,dev]"
make check          # the complete local software gate
```

`make check` runs, in order:

| Step | What it does |
|---|---|
| `lint` | Ruff over `Python/eiger2` and `tests` |
| `test` | the software test suite in `tests` |
| `wheel-check` | builds an sdist and wheel in isolation, audits the wheel archive, installs it into a fresh virtual environment, and checks the version, `pip check`, the `eiger2` entry point and the `eiger2`/`functions` imports from outside the checkout |
| `smoke` | generates the synthetic example once, then runs the revised caller (its exit code is inspected against the run manifest, never accepted blindly) and the legacy caller (exit 0 required) |

The revised smoke call reads its verdict off `run_manifest.json` rather than
off the exit code: `tools/check_smoke_manifest.py` requires the structural
gates to pass, prints every refused length stratum, and fails on
`inference_unreliable`, on `no_regions`, on a status eiger2 does not document,
or on a status that contradicts the exit code. The synthetic example currently
reaches `partial_inference` -- length stratum `20-49` is refused for
insufficient null opportunity -- which is a documented stratum-local state and
is reported as a refusal, never as a completed inference. Add
`--expect-no-refusals` to that script to require a refusal-free run.

Every build, virtual environment and smoke input or output `make check` creates
lives in a `mktemp` directory outside the repository, which is removed on exit.
No target deletes or overwrites `dist/`, `examples/data/` or `examples/out_*`.
`make build` is the one target that writes into the repository, into the
ignored `dist/` directory, and it deletes nothing there.

Individual targets (`make lint`, `make test`, `make build`, `make wheel-check`,
`make wheel-gate`, `make smoke-revised`, `make smoke-legacy`) can be run on
their own, and `PYTHON` selects the interpreter: `make check PYTHON=python3.12`.
The direct `pip` and `pytest` commands above remain valid and are what the
targets run. Nothing here fetches data or touches the paper workspace.

#### The installed-wheel gate

`make wheel-gate` is the release-facing form of `wheel-check` and is what CI
runs. It builds one sdist and wheel in isolation, requires exactly one wheel,
audits the archive, installs `<wheel>[legacy]` into a fresh virtual environment
outside the repository, and then runs **both** smoke calls through that
environment's own `eiger2` executable. It never installs the checkout, editably
or otherwise, so a file missing from the wheel cannot be satisfied by the
source tree:

```bash
make wheel-gate                 # the exact target the CI wheel job runs
```

`tools/check_wheel.py` holds its verdict, in three modes with their own tests
in `tests/test_check_wheel.py`. `environment` runs before anything is built and
refuses to proceed if `PYTHONPATH` reaches into the checkout -- which would not
only shadow an import but make pip read the editable install's `eiger.egg-info`
and skip installing the wheel altogether. `contents` reads the archive and requires the
`eiger-2.2.0-py3-none-any.whl` filename, a top level of exactly `eiger2`,
`functions` and the distribution metadata, `LICENSE` under that metadata, and
the declared `eiger2` console entry point; any benchmarking, paper, analysis,
provenance, script, result or fixture material fails it. `installed` runs in
the temporary environment, from a temporary working directory, and requires
that `PYTHONPATH`, `sys.path`, the working directory and the imported module
itself all stay out of the checkout, that `eiger2.__version__` is `2.2.0`, that
the installed `direct_url.json` names the wheel just built, and that
`eiger2.benchmark` imports with no paper fixture present.

`make check` runs the cheaper `wheel-check` -- the same build, audit and import
isolation without the legacy extra or the smoke calls, which `make smoke`
already covers against the editable install.

CI runs these same targets: `make lint`, `make test` and `make smoke` on Python
3.10, 3.11 and 3.12 against an editable install, and `make wheel-gate` once on
Python 3.12. One version is enough for the wheel job because the artefact is a
pure-Python `py3-none-any` wheel -- the same file each version installs -- and
`wheel-gate` fails if that platform tag ever changes.

---

## Outputs

> **Note (v2.2.0):** `deflation_max_concentration` moved 0.10 → 0.12, so a component scoring in (0.10, 0.12] is now treated as nuisance and projected out. Results are unchanged on both real cohorts the method has been run on (leading components 0.1258 and 0.1335) and on every synthetic grid the docs cite; what changes is behaviour on bimodal marginals, where the filter previously half-fired. No output column, schema or file changed — `schema_version` is unchanged. See [`deflation_max_concentration`](#deflation_max_concentration) below.

> **Migration note (v2.1.0):** results from v2.0.0 and v2.0.1 are **void and must be regenerated.** The region-level null was replaced, which changes the edge threshold, the set of candidate regions, their areas, and every FDR column. Region *identifiers* are therefore not comparable across versions either. The edge statistic itself, the eligibility mask, and the latent-structure removal are unchanged.

The principal region-level output is the deliberately lean `regions.csv`, with one
row per detected region. When the support policy is enabled, compact post-selection
prevalence, largest-gap tail and influence/robustness details live in
`region_support_diagnostics.csv`, keyed by `region_id`; see
**[docs/BIOLOGICAL_SUPPORT.md](docs/BIOLOGICAL_SUPPORT.md)**.

**`rank` is the principal field on the default path: regions ordered by
`region_area` within chromosome order.** `region_qvalue` and the six columns below
it are populated only when a region null was requested and built; on the default
path they are `NaN` and `fdr_status` reads `no_calibrated_null`.

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

`coordination_significant` is the prevalence-independent, nullable result of the
configured point/upper FDR decision. `biological_class` is missing for non-calls and
otherwise uses exactly five provisional descriptive labels: `cohort_supported`,
`recurrent_subgroup`, `rare_epivariant`, `outlier_sensitive`, or
`diffuse_coordination`. Neither field changes the Stage 2D q-values. Historical
largest-gap `minor_group_*` API fields are not authoritative prevalence measures.
The provisional classification is opt-in with
`--biological-support-policy five_class_v1`; the default `off` leaves
`biological_class` missing, constructs no diagnostics table, writes no diagnostics
file and preserves discovery runtime. See the support document for the candidate-label
and performance limitations.

### `--min-prevalence-samples`

```bash
eiger2 call --betas betas.tsv --locations cpgs.bed --out results/ \
    --min-prevalence-samples 3
```

Minimum number of samples that must carry a region's deviation for it to be
`significant`. **The default is 0 (off), and that is a measured decision, not an
omission.** A one- or two-sample deviation across a run of CpGs is genuinely
coordinated — it is not a false positive under any null — it is simply not a
subgroup-defining DMR. Raising the floor to 3 was measured on the full 280-run
`signal_grid` and on the 519-sample ICGC cohort by the paper-side sweep
[`benchmarking/prevalence_default.py`](https://github.com/tobymilnec/eiger_paper/blob/d83f61decfa624cbc079c38f38e3dd480fee322b/benchmarking/prevalence_default.py),
which lives in the separate `eiger_paper` repository and is not part of this
package:

- in 10 of the 14 grid cells it costs nothing at all — mean recall identical to
  the last decimal, because their planted subgroups are 4 samples or larger;
- in the four cells whose planted subgroup is exactly two samples it costs
  **91.4–94.3 recall points** (mean recall 0.99–1.00 → 0.05–0.09);
- pooled over all 280 runs it removes 641 of 2,051 significant calls but only 3
  of the 90 false ones, so mean per-run FDP **rises** from 6.2% to 11.7%;
- on ICGC it drops 1,980 of 81,979 significant regions (2.4%), of which 1,973
  are already labelled `rare_epivariant` by
  `--biological-support-policy five_class_v1`.

So the floor stays off and `eiger2 call` instead prints one note per run naming
how many significant regions are carried by one or two samples. Set it to 3 when
the analysis genuinely concerns subgroup-defining regions; do not set it on a
cohort whose minority groups may be at the two-carrier floor. `region_qvalue` is
calibrated on the unfiltered set either way, so the setting changes no q-value.

### `deflation_max_concentration`

| Setting | Default | Effect |
|---|---|---|
| `deflation_max_concentration` | `0.12` | Only components whose loading energy is DIFFUSE are projected out. A component's *loading concentration* is the fraction of its loading energy in the top 1% of CpGs; at or below the cutoff it is `nuisance_deflated`, above it `biological_kept`. Set to `1.0` to deflate every component, or set `deflate_components` explicitly to bypass the filter. |
| `DEFLATION_NEAR_CUTOFF_MARGIN` | `0.02` | Not a setting: a note. Any candidate within 0.02 of the cutoff — the band (0.10, 0.14) — is named in `manifest["latent_structure"]["near_cutoff_components"]` and in one stderr line. It changes no verdict and no q-value. |

**0.12 is measured on both sides** (raised from 0.10 in v2.2.0). Below it, the
simulated nuisance bands the filter must cover: 98.6% of candidates on a
leukocyte-fitted bimodal marginal, 100% on an ICGC-fitted one, 100% on the
synthetic one, and small-cohort recall recovering from 0.46–0.56 to 0.72–0.83.
Above it, the real cohorts it must not touch: the immune atlas scores its
leading component at 0.1258 genome-wide and ICGC at 0.1335, both kept, while a
cutoff of 0.15 deflates both and churns the ICGC significant set. Full evidence
in **[docs/EIGER2_DESIGN.md](docs/EIGER2_DESIGN.md)**.

**Deflation verdicts have the scope of the basis estimation.** The same
nuisance factor scored one chromosome at a time scatters about ±0.05 around its
genome-wide value (immune atlas: 0.1074–0.1802 per chromosome, 0.1258 pooled).
Estimate the basis genome-wide where memory allows; per-chromosome runs are a
workaround whose deflation verdicts carry that scatter. Always check
`manifest["latent_structure"]["component_verdict"]`.

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
