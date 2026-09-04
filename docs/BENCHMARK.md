# EIGER benchmark: original vs EIGER 2

## The simulator: how a planted DMR is planted

This section documents `eiger2.benchmark.simulate`, which **is** shipped in the
wheel, as distinct from the harness and tables below, which are not.

### Direction is chosen per region, not per CpG

A planted DMR is a *coordinated* region: one methylation direction, shared by
every CpG and every carrier sample in the block. `simulate(planting_direction=…)`
selects how that direction is chosen.

| value | meaning |
|---|---|
| `"region_headroom"` (**default**) | One scalar direction for the whole planted region, chosen from the region's aggregate beta headroom. Every CpG and every carrier in that region moves the same way; adjacent CpGs can never receive opposite-signed effects. |
| `"per_cpg_headroom"` | The direction is chosen independently for each CpG, from that CpG's own baseline (`-1` above 0.5, `+1` otherwise). |

**Region-consistent direction is the default, and it is the only mode that
simulates an ordinary coherent-DMR power scenario.** `"per_cpg_headroom"` is
retained for two purposes only: reproducing a run made before
`planting_direction` existed, and an explicit heterogeneous-direction stress
test. Under a bimodal or otherwise non-unimodal supplied marginal it can shift
adjacent CpGs of one block in opposite directions, which makes the block locally
*anti*-correlated rather than a coordinated region — the opposite of what the
planted truth claims. Do not use it for a recall-vs-effect curve.

Under `marginal="unimodal"` — the only marginal the package draws on its own —
every baseline is exactly 0.5, so both modes plant `+delta` at every CpG and are
byte-identical. The choice only matters when a marginal is supplied through
`marginal_sampler`.

### The aggregate-headroom rule

For a planted block with baseline means `mu` and nominal effect `delta`, and the
same `BETA_CLIP = (0.001, 0.999)` bounds the finished matrix is clipped to:

```
up_i   = clip(mu_i + delta) - clip(mu_i)      # attainable upward shift
down_i = clip(mu_i) - clip(mu_i - delta)      # attainable downward shift
```

The region is planted upward when `mean(up) > mean(down)`, downward otherwise,
and upward on a tie. `HEADROOM_TIE_ATOL` makes that tie-break reachable: a block
whose baselines are symmetric about 0.5 ties in exact arithmetic but not always
in floating point, and the documented rule should decide it rather than a
last-bit rounding difference.

The rule reads the block's baselines, `delta` and the clip bounds and nothing
else — no biological label, no called region, no q-value, no benchmark outcome —
and it consumes no random number. Choosing a `planting_direction` therefore does
not move the random stream: carrier identities, truth coordinates and layout
placement are identical between the two modes at one seed, and only the sign
pattern inside a planted block differs.

### `effective_delta` is still the realised beta-scale effect

Each truth entry's `effective_delta` is the mean absolute *post-clip* shift
actually realised over the planted block's carrier entries, exactly as before.
It is unchanged in meaning, and it is what a recall-vs-effect curve must be
plotted against whenever a marginal is supplied.

Region-consistent planting makes that number do visible work. One direction
cannot have full headroom at every CpG of a block whose baselines straddle 0.5,
so on such blocks `effective_delta` sits below the nominal `delta` — where the
per-CpG rule reported the full `delta` by shifting each CpG whichever way suited
it, and so reported a coherent effect it had not planted. The shortfall is
recorded per region rather than hidden, and no block is filtered, relocated or
rejected on account of it.

### What a supplied marginal does and does not reproduce

`marginal_sampler` supplies a per-CpG `(baseline_mu, total_sd)`. That reproduces
the *marginal* distribution of methylation values — it does not reproduce the
*spatial autocorrelation* of real data, because each CpG's baseline is supplied
independently of its neighbours' and the only shared structure in the matrix is
the latent factors. A fitted-marginal arm is therefore a marginal-realism
simulation, not a full replication of a real track's correlation structure, and
its results should be described as such.

### Which earlier simulated results the direction change invalidates

> **Historical — every supplied-marginal *signal* simulation produced before the
> commit introducing `region_headroom` used the equivalent of
> `planting_direction="per_cpg_headroom"`.** In those runs adjacent CpGs of one
> planted block could receive opposite-signed shifts, so the block was not a
> coherent same-direction region. Such runs are **not** coherent-DMR recall or
> power evidence and must not be presented as such. They remain valid as
> heterogeneous-direction (alternating-direction) stress tests, provided they are
> labelled that way. Results produced under `planting_direction="region_headroom"`
> are the coherent-DMR evidence.

The exact API this turns on:

```python
simulate(..., planting_direction="region_headroom")    # coherent DMR; the default
simulate(..., planting_direction="per_cpg_headroom")   # historical / stress only
```

**Unaffected — no rerun, no relabelling.**

- `call_dmrs` and every real-data EIGER analysis. The change is confined to the
  simulator; no caller, inference default or output semantic moved.
- DMR-free null and any-call calibration, **including under supplied marginals**:
  no signal is planted, so the planting code path is never entered.
- Dense/unimodal simulations. Every baseline is exactly 0.5, so the two modes are
  byte-identical and every pinned digest is unchanged.
- Coordinate-layout estimation and marginal fitting themselves. The fits are
  inputs to the simulator, not products of it.
- Software inference defaults and output semantics, including the truth-tuple
  schema and `effective_delta`'s meaning.

**Rerun required, under `planting_direction="region_headroom"`.** Relabelling is
not sufficient: the simulated matrix itself differs.

- Supplied-marginal `signal_grid` and `power_curve` arms.
- `bimodal_marginal` recall/FDP evidence, and the inputs to Table S6.
- Realistic-marginal beta-versus-M signal results.
- Simulated supervised and unsupervised method-comparison arms run on those
  marginals.
- M1 `bimodal_atlas` and `bimodal_fitted` weak-signal arms.
- M1 `delta=0.20` positive controls and the paired amplitude diagnostics.
- Figures 3 and 4, and every supplementary table or claim consuming the above.

**Historical / stress only.**

- Anything generated under `per_cpg_headroom`, whether or not it was named at the
  time.
- Retain it as heterogeneous-direction or alternating-direction stress evidence
  if that is what is being argued.
- Never present it as ordinary same-direction DMR power or recall evidence.

**Consumers in `eiger_paper` that need an explicit update.** These are named for
the reader's orientation; nothing here is a dependency of this repository, and
the wheel neither imports nor ships any of it.

- `benchmarking/simulate_support.py`
- `benchmarking/bimodal_marginal.py`
- `benchmarking/beta_vs_m.py`
- the comparator runners that simulate under a supplied marginal
- the uncommitted M1 scripts and protocols

The paper-side simulation helper must pass `planting_direction` **explicitly**
rather than relying on the default, and must record the value it passed in the
run manifest, so a stored result states which mode produced it instead of leaving
it to be inferred from a date. `simulate` reports the mode it used at
`locations.attrs["planting_direction"]`; that value is a plain string and is the
one to record.

---

## Stage 2D screen and confirm workflow

> **This workflow belongs to `eiger_paper`, not to this repository.** The
> benchmark harness `benchmark_suite.py` is maintained in the separate
> `eiger_paper` repository and is **not** installed by the `eiger2` wheel, so
> none of it can be run from a checkout or an installation of this repository.
> It is recorded here as the design of the measurement the numbers in this
> document come from.

Do not rerun the historical grid unchanged. The current suite derives both point
and upper-bound decisions from each fitted run and records both, so stochastic null
realisations are shared. It also includes a compact regression screen for the
provisional biological-support layer. Screen stage, against
[`benchmarking/benchmark_suite.py`](https://github.com/tobymilnec/eiger_paper/blob/d83f61decfa624cbc079c38f38e3dd480fee322b/benchmarking/benchmark_suite.py) in
`eiger_paper`: the `support` command at 10 seeds, then `negative` and `sweep` at
20 seeds, all three writing to one shared screen output directory.

After the code, null definition and thresholds are frozen, the confirmation stage
reruns `negative`, `sweep` and `support` at 100 seeds each into a **fresh** output
directory, so no screen-stage result is overwritten or silently reused.

The 2026-08-23 confirmatory manifests record commit `a4faddb`, but their dirty
worktree fields are `null`; the harness now records the dirty state and source
fingerprint. Re-run publication-grade results from a clean tagged commit. The
confirmatory CSVs committed in `eiger_paper` remain useful audit evidence, but do
not provide byte-level source provenance for that run.

The confirmatory design must report coordination FDR/FDP, probability of any call
under the complete null, point and upper power/recall by planted prevalence and
region length, five-class accuracy, and one/two-sample influence-flag performance.
Required signal types are private coherent epivariants, recurrent and cohort
subgroups, continuous gradients, and realistic ordinary local background
autocorrelation; required stresses are nuisance structure, small cohorts, high noise
and random/structured missingness. The compact `support` command exercises those
classification mechanics, but a publication claim still requires them inside full
genomic simulations with null construction and segmentation active.

> **Scale caveat, found running the 100-seed confirmatory grid above (Stage 2e):**
> `negative`/`sweep` simulate ~10,000-20,000-CpG genomes and leave `null_min_lag`
> at its genome-scale default of 1000. At that combination, `member_step`'s null
> subtracks realise only 4-7 edges long, so the null has **zero** opportunity for
> length strata at and above "10-19" CpGs -- including the 40-CpG stratum
> `DEFAULT_STRATA` plants signal in. `sweep`'s point/upper realised-FDR and recall
> columns are therefore not a valid real-signal calibration measurement at this
> scale: true regions get correctly segmented and then refused from inference
> (zero opportunity), not wrongly scored, which is the opportunity guard (§16 of
> `docs/FDR.md`) working as designed, not a defect -- but it means this command
> cannot be used to confirm or refute real-signal FDR control. `negative`'s
> DMR-free `any_call` measurement is unaffected by this (it does not depend on the
> null having opportunity for long strata to be meaningful) and is the one
> confirmed quantity from this workflow; see `docs/FDR.md`'s "DMR-free
> re-measurement (Stage 2e)". Real-signal calibration needed
> [`benchmarking/paper_grid.py`](https://github.com/tobymilnec/eiger_paper/blob/d83f61decfa624cbc079c38f38e3dd480fee322b/benchmarking/paper_grid.py)'s ~600,000-CpG
> `calibration`/`signal_grid`/`power_curve` cells in `eiger_paper` instead -- run
> 2026-08-23. Mean per-run FDP was 6.2%
> overall and 6.9-15.4% in the weak-effect 4-CpG cells; the 4.4% pooled
> false-call ratio is call-weighted and masks this boundary failure. See
> `docs/FDR.md`'s "Real-signal re-measurement at genome scale (Stage 2e,
> `paper_grid.py`)" for the per-cell results. `sweep`'s own scale bug still blocks this
> command specifically for that purpose -- see `docs/FDR.md`'s "The
> `heuristic_sweep` benchmark-scale bug, found en route (Stage 2e)".

> **Historical — produced with EIGER v2.0.0 (commit `b8e4a19`).** All EIGER 2
> region-level metrics below derive from the pre-v2.0.1 q-value calculation, which
> had a confirmed direction/tie defect (see [docs/FDR.md](FDR.md)) and understated
> the realised false discovery rate. See
> [docs/FDR.md's "Known limitation" section](FDR.md#known-limitation-fdr-calibration-under-missingness-small-n-and-high-noise)
> for the v2.0.1 corrected, measured figures. Do not use these numbers to assess
> v2.0.1's calibration, and note that even those v2.0.1 figures are themselves now
> superseded: they used what is now called `region_null_mode="lag"`, which is known
> to fail both region-level exchangeability gates on real layouts and is no longer
> the default. Current (v2.1.0, `region_null_mode="member_step"`) figures are
> reproduced in [docs/FDR.md](FDR.md); they were produced by the v2.1.0 paper-grid
> run and recorded in the Stage 1e stop report, an unpublished development record
> retained outside this repository.

Seeds per scenario: 3. All values are means across seeds. The original method's covariance threshold is swept; EIGER 2 runs once at its automatic operating point.


## Table 1 — original method, covariance threshold sweep

CpG-level metrics, `original (as written)`.


**clean** — `original (as written)`

|   threshold |   cpg_precision |   cpg_recall |   cpg_f1 |   cpg_mcc |   cpg_jaccard |   regions_called |   regions_recovered |   regions_false |
|------------:|----------------:|-------------:|---------:|----------:|--------------:|-----------------:|--------------------:|----------------:|
|       0.001 |           0.568 |        0.920 |    0.702 |     0.687 |         0.541 |          273.000 |              35.333 |         237.667 |
|       0.001 |           0.664 |        0.905 |    0.766 |     0.747 |         0.620 |          211.667 |              34.333 |         177.333 |
|       0.002 |           0.806 |        0.872 |    0.837 |     0.820 |         0.720 |          146.000 |              31.333 |         114.667 |
|       0.004 |           0.937 |        0.788 |    0.856 |     0.846 |         0.748 |          100.667 |              26.000 |          74.667 |
|       0.008 |           0.988 |        0.557 |    0.712 |     0.723 |         0.553 |           65.333 |              18.000 |          47.333 |
|       0.012 |           0.992 |        0.371 |    0.540 |     0.587 |         0.370 |           38.000 |              15.000 |          23.000 |
|       0.020 |           1.000 |        0.234 |    0.380 |     0.465 |         0.234 |           29.333 |               8.000 |          21.333 |
|       0.030 |           1.000 |        0.026 |    0.051 |     0.150 |         0.026 |            6.667 |               0.000 |           6.667 |
|       0.050 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.080 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.120 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |

**nuisance** — `original (as written)`

|   threshold |   cpg_precision |   cpg_recall |   cpg_f1 |   cpg_mcc |   cpg_jaccard |   regions_called |   regions_recovered |   regions_false |
|------------:|----------------:|-------------:|---------:|----------:|--------------:|-----------------:|--------------------:|----------------:|
|       0.001 |           0.440 |        0.679 |    0.534 |     0.486 |         0.364 |          344.667 |              14.000 |         330.667 |
|       0.001 |           0.472 |        0.660 |    0.551 |     0.502 |         0.380 |          309.667 |              13.667 |         296.000 |
|       0.002 |           0.535 |        0.616 |    0.572 |     0.524 |         0.401 |          251.000 |              12.333 |         238.667 |
|       0.004 |           0.636 |        0.540 |    0.584 |     0.545 |         0.412 |          179.000 |              11.000 |         168.000 |
|       0.008 |           0.817 |        0.393 |    0.531 |     0.538 |         0.361 |           95.667 |               7.000 |          88.667 |
|       0.012 |           0.937 |        0.277 |    0.427 |     0.487 |         0.272 |           57.000 |               4.000 |          53.000 |
|       0.020 |           0.998 |        0.122 |    0.218 |     0.334 |         0.122 |           28.000 |               0.333 |          27.667 |
|       0.030 |           0.667 |        0.010 |    0.019 |     0.076 |         0.010 |            2.667 |               0.000 |           2.667 |
|       0.050 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.080 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.120 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |

**missingness** — `original + mean imputation`

|   threshold |   cpg_precision |   cpg_recall |   cpg_f1 |   cpg_mcc |   cpg_jaccard |   regions_called |   regions_recovered |   regions_false |
|------------:|----------------:|-------------:|---------:|----------:|--------------:|-----------------:|--------------------:|----------------:|
|       0.001 |           0.659 |        0.884 |    0.755 |     0.734 |         0.607 |          215.333 |              32.000 |         183.333 |
|       0.001 |           0.796 |        0.847 |    0.820 |     0.801 |         0.695 |          154.000 |              27.667 |         126.333 |
|       0.002 |           0.928 |        0.766 |    0.839 |     0.828 |         0.723 |          108.000 |              23.667 |          84.333 |
|       0.004 |           0.979 |        0.546 |    0.701 |     0.713 |         0.540 |           77.667 |              14.667 |          63.000 |
|       0.008 |           0.993 |        0.298 |    0.458 |     0.524 |         0.297 |           41.667 |              10.333 |          31.333 |
|       0.012 |           0.999 |        0.155 |    0.269 |     0.377 |         0.155 |           30.000 |               1.333 |          28.667 |
|       0.020 |           0.667 |        0.002 |    0.004 |     0.037 |         0.002 |            0.667 |               0.000 |           0.667 |
|       0.030 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.050 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.080 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.120 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |

**small_n** — `original (as written)`

|   threshold |   cpg_precision |   cpg_recall |   cpg_f1 |   cpg_mcc |   cpg_jaccard |   regions_called |   regions_recovered |   regions_false |
|------------:|----------------:|-------------:|---------:|----------:|--------------:|-----------------:|--------------------:|----------------:|
|       0.001 |           0.444 |        0.568 |    0.498 |     0.442 |         0.333 |          155.333 |               1.667 |         153.667 |
|       0.001 |           0.504 |        0.499 |    0.502 |     0.449 |         0.338 |          122.000 |               1.000 |         121.000 |
|       0.002 |           0.622 |        0.382 |    0.473 |     0.446 |         0.312 |           77.333 |               0.333 |          77.000 |
|       0.004 |           0.699 |        0.152 |    0.249 |     0.297 |         0.142 |           31.333 |               0.000 |          31.333 |
|       0.008 |           0.766 |        0.014 |    0.027 |     0.094 |         0.014 |            3.000 |               0.000 |           3.000 |
|       0.012 |           0.333 |        0.002 |    0.004 |     0.025 |         0.002 |            0.333 |               0.000 |           0.333 |
|       0.020 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.030 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.050 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.080 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.120 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |

**celltype** — `original (as written)`

|   threshold |   cpg_precision |   cpg_recall |   cpg_f1 |   cpg_mcc |   cpg_jaccard |   regions_called |   regions_recovered |   regions_false |
|------------:|----------------:|-------------:|---------:|----------:|--------------:|-----------------:|--------------------:|----------------:|
|       0.001 |           0.988 |        1.000 |    0.994 |     0.993 |         0.988 |           70.000 |              70.000 |           0.000 |
|       0.001 |           0.995 |        1.000 |    0.998 |     0.997 |         0.995 |           70.000 |              70.000 |           0.000 |
|       0.002 |           1.000 |        1.000 |    1.000 |     1.000 |         1.000 |           70.000 |              70.000 |           0.000 |
|       0.004 |           1.000 |        1.000 |    1.000 |     1.000 |         1.000 |           70.000 |              70.000 |           0.000 |
|       0.008 |           1.000 |        0.903 |    0.949 |     0.941 |         0.903 |           76.333 |              60.000 |          16.333 |
|       0.012 |           1.000 |        0.857 |    0.923 |     0.912 |         0.857 |           60.000 |              60.000 |           0.000 |
|       0.020 |           1.000 |        0.857 |    0.923 |     0.912 |         0.857 |           60.000 |              60.000 |           0.000 |
|       0.030 |           1.000 |        0.775 |    0.873 |     0.860 |         0.775 |           94.667 |              42.667 |          52.000 |
|       0.050 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.080 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.120 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |

**celltype_nuisance** — `original (as written)`

|   threshold |   cpg_precision |   cpg_recall |   cpg_f1 |   cpg_mcc |   cpg_jaccard |   regions_called |   regions_recovered |   regions_false |
|------------:|----------------:|-------------:|---------:|----------:|--------------:|-----------------:|--------------------:|----------------:|
|       0.001 |           0.792 |        0.988 |    0.879 |     0.858 |         0.785 |          176.333 |              66.000 |         110.333 |
|       0.001 |           0.873 |        0.987 |    0.926 |     0.912 |         0.863 |          132.000 |              66.000 |          66.000 |
|       0.002 |           0.942 |        0.984 |    0.962 |     0.955 |         0.928 |          102.667 |              64.667 |          38.000 |
|       0.004 |           0.979 |        0.966 |    0.973 |     0.967 |         0.947 |           91.000 |              62.000 |          29.000 |
|       0.008 |           0.996 |        0.872 |    0.930 |     0.919 |         0.869 |           68.333 |              60.000 |           8.333 |
|       0.012 |           0.999 |        0.855 |    0.921 |     0.910 |         0.854 |           63.667 |              60.000 |           3.667 |
|       0.020 |           1.000 |        0.849 |    0.918 |     0.907 |         0.848 |           74.333 |              58.000 |          16.333 |
|       0.030 |           1.000 |        0.633 |    0.775 |     0.767 |         0.633 |           91.333 |              36.000 |          55.000 |
|       0.050 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.080 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |
|       0.120 |           0.000 |        0.000 |    0.000 |     0.000 |         0.000 |            0.000 |               0.000 |           0.000 |


## Table 2 — head to head at the original's oracle-best threshold

The original is scored at the threshold that maximises its own CpG F1 on each scenario, chosen using the ground truth. EIGER 2 is untuned.

| scenario          | method                                                 |   cpg_precision |   cpg_recall |   cpg_f1 |   cpg_mcc |   region_precision |   region_recall |   region_f1 |
|:------------------|:-------------------------------------------------------|----------------:|-------------:|---------:|----------:|-------------------:|----------------:|------------:|
| clean             | original (as written) @ 0.004 (oracle)                 |           0.937 |        0.788 |    0.856 |     0.846 |              0.258 |           0.542 |       0.350 |
| clean             | original (as written) @ 0.012 (published default)      |           0.992 |        0.371 |    0.540 |     0.587 |              0.398 |           0.312 |       0.350 |
| clean             | EIGER 2 correlation                                    |           0.963 |        1.000 |    0.981 |     0.979 |              0.993 |           1.000 |       0.997 |
| clean             | EIGER 2 covariance                                     |           0.952 |        1.000 |    0.975 |     0.973 |              1.000 |           1.000 |       1.000 |
| nuisance          | original (as written) @ 0.004 (oracle)                 |           0.636 |        0.540 |    0.584 |     0.545 |              0.061 |           0.229 |       0.097 |
| nuisance          | original (as written) @ 0.012 (published default)      |           0.937 |        0.277 |    0.427 |     0.487 |              0.070 |           0.083 |       0.076 |
| nuisance          | EIGER 2 correlation                                    |           0.968 |        1.000 |    0.984 |     0.982 |              1.000 |           1.000 |       1.000 |
| nuisance          | EIGER 2 covariance                                     |           0.953 |        1.000 |    0.976 |     0.973 |              1.000 |           1.000 |       1.000 |
| missingness       | original + mean imputation @ 0.002 (oracle)            |           0.928 |        0.766 |    0.839 |     0.828 |              0.220 |           0.493 |       0.304 |
| missingness       | original + mean imputation @ 0.012 (published default) |           0.999 |        0.155 |    0.269 |     0.377 |              0.046 |           0.028 |       0.035 |
| missingness       | EIGER 2 correlation                                    |           0.947 |        0.885 |    0.913 |     0.906 |              0.514 |           0.757 |       0.587 |
| missingness       | EIGER 2 covariance                                     |           0.966 |        0.928 |    0.946 |     0.941 |              0.684 |           0.868 |       0.757 |
| small_n           | original (as written) @ 0.001 (oracle)                 |           0.504 |        0.499 |    0.502 |     0.449 |              0.008 |           0.042 |       0.014 |
| small_n           | original (as written) @ 0.012 (published default)      |           0.333 |        0.002 |    0.004 |     0.025 |              0.000 |           0.000 |       0.000 |
| small_n           | EIGER 2 correlation                                    |           0.992 |        0.976 |    0.984 |     0.982 |              0.871 |           0.986 |       0.923 |
| small_n           | EIGER 2 covariance                                     |           0.983 |        0.993 |    0.988 |     0.987 |              0.961 |           1.000 |       0.980 |
| celltype          | original (as written) @ 0.004 (oracle)                 |           1.000 |        1.000 |    1.000 |     1.000 |              1.000 |           1.000 |       1.000 |
| celltype          | original (as written) @ 0.012 (published default)      |           1.000 |        0.857 |    0.923 |     0.912 |              1.000 |           0.857 |       0.923 |
| celltype          | EIGER 2 correlation                                    |           0.993 |        1.000 |    0.996 |     0.995 |              1.000 |           1.000 |       1.000 |
| celltype          | EIGER 2 covariance                                     |           0.991 |        0.902 |    0.943 |     0.934 |              1.000 |           0.905 |       0.949 |
| celltype_nuisance | original (as written) @ 0.004 (oracle)                 |           0.979 |        0.966 |    0.973 |     0.967 |              0.686 |           0.886 |       0.772 |
| celltype_nuisance | original (as written) @ 0.012 (published default)      |           0.999 |        0.855 |    0.921 |     0.910 |              0.945 |           0.857 |       0.898 |
| celltype_nuisance | EIGER 2 correlation                                    |           0.995 |        1.000 |    0.997 |     0.997 |              1.000 |           1.000 |       1.000 |
| celltype_nuisance | EIGER 2 covariance                                     |           0.995 |        0.902 |    0.945 |     0.937 |              1.000 |           0.905 |       0.949 |


## Table 3 — region-level detail at the oracle threshold

| scenario          | method                             |   regions_called |   regions_recovered |   regions_false |   fragmentation |   fpr |
|:------------------|:-----------------------------------|-----------------:|--------------------:|----------------:|----------------:|------:|
| clean             | original (as written) @ 0.004      |          100.667 |              26.000 |          74.667 |           1.000 | 0.006 |
| clean             | EIGER 2 correlation                |           48.333 |              48.000 |           0.333 |           1.000 | 0.004 |
| clean             | EIGER 2 covariance                 |           48.000 |              48.000 |           0.000 |           1.000 | 0.005 |
| nuisance          | original (as written) @ 0.004      |          179.000 |              11.000 |         168.000 |           1.000 | 0.034 |
| nuisance          | EIGER 2 correlation                |           48.000 |              48.000 |           0.000 |           1.000 | 0.004 |
| nuisance          | EIGER 2 covariance                 |           48.000 |              48.000 |           0.000 |           1.000 | 0.005 |
| missingness       | original + mean imputation @ 0.002 |          108.000 |              23.667 |          84.333 |           1.000 | 0.006 |
| missingness       | EIGER 2 correlation                |           91.000 |              36.333 |          54.667 |           1.000 | 0.005 |
| missingness       | EIGER 2 covariance                 |           65.000 |              41.667 |          23.333 |           1.000 | 0.004 |
| small_n           | original (as written) @ 0.001      |          122.000 |               1.000 |         121.000 |           1.000 | 0.052 |
| small_n           | EIGER 2 correlation                |           27.333 |              23.667 |           3.667 |           1.000 | 0.001 |
| small_n           | EIGER 2 covariance                 |           25.000 |              24.000 |           1.000 |           1.000 | 0.002 |
| celltype          | original (as written) @ 0.004      |           70.000 |              70.000 |           0.000 |           1.000 | 0.000 |
| celltype          | EIGER 2 correlation                |           70.000 |              70.000 |           0.000 |           1.000 | 0.002 |
| celltype          | EIGER 2 covariance                 |           63.333 |              63.333 |           0.000 |           1.000 | 0.002 |
| celltype_nuisance | original (as written) @ 0.004      |           91.000 |              62.000 |          29.000 |           1.000 | 0.004 |
| celltype_nuisance | EIGER 2 correlation                |           70.000 |              70.000 |           0.000 |           1.000 | 0.001 |
| celltype_nuisance | EIGER 2 covariance                 |           63.333 |              63.333 |           0.000 |           1.000 | 0.001 |


## Table 4 — recall by subgroup prevalence and effect size


**clean**

| method              |   f=0.05,d=0.35 |   f=0.05,d=0.55 |   f=0.1,d=0.3 |   f=0.1,d=0.5 |   f=0.25,d=0.25 |   f=0.25,d=0.4 |   f=0.5,d=0.2 |   f=0.5,d=0.35 |
|:--------------------|----------------:|----------------:|--------------:|--------------:|----------------:|---------------:|--------------:|---------------:|
| EIGER 2 correlation |           1.000 |           1.000 |         1.000 |         1.000 |           1.000 |          1.000 |         1.000 |          1.000 |
| EIGER 2 covariance  |           1.000 |           1.000 |         1.000 |         1.000 |           1.000 |          1.000 |         1.000 |          1.000 |
| original @ 0.002    |           0.111 |           0.556 |         0.389 |         1.000 |           0.667 |          1.000 |         0.500 |          1.000 |
| original @ 0.004    |           0.000 |           0.389 |         0.167 |         1.000 |           0.500 |          1.000 |         0.278 |          1.000 |
| original @ 0.012    |           0.000 |           0.000 |         0.000 |         0.500 |           0.000 |          1.000 |         0.000 |          1.000 |

**nuisance**

| method              |   f=0.05,d=0.35 |   f=0.05,d=0.55 |   f=0.1,d=0.3 |   f=0.1,d=0.5 |   f=0.25,d=0.25 |   f=0.25,d=0.4 |   f=0.5,d=0.2 |   f=0.5,d=0.35 |
|:--------------------|----------------:|----------------:|--------------:|--------------:|----------------:|---------------:|--------------:|---------------:|
| EIGER 2 correlation |           1.000 |           1.000 |         1.000 |         1.000 |           1.000 |          1.000 |         1.000 |          1.000 |
| EIGER 2 covariance  |           1.000 |           1.000 |         1.000 |         1.000 |           1.000 |          1.000 |         1.000 |          1.000 |
| original @ 0.002    |           0.000 |           0.000 |         0.000 |         0.389 |           0.056 |          0.722 |         0.000 |          0.889 |
| original @ 0.004    |           0.000 |           0.000 |         0.000 |         0.333 |           0.056 |          0.667 |         0.000 |          0.778 |
| original @ 0.012    |           0.000 |           0.000 |         0.000 |         0.000 |           0.000 |          0.222 |         0.000 |          0.444 |

**missingness**

| method              |   f=0.05,d=0.35 |   f=0.05,d=0.55 |   f=0.1,d=0.3 |   f=0.1,d=0.5 |   f=0.25,d=0.25 |   f=0.25,d=0.4 |   f=0.5,d=0.2 |   f=0.5,d=0.35 |
|:--------------------|----------------:|----------------:|--------------:|--------------:|----------------:|---------------:|--------------:|---------------:|
| EIGER 2 correlation |           0.333 |           0.611 |         0.500 |         1.000 |           0.778 |          1.000 |         0.833 |          1.000 |
| EIGER 2 covariance  |           0.333 |           0.889 |         0.722 |         1.000 |           1.000 |          1.000 |         1.000 |          1.000 |
| original @ 0.002    |           0.000 |           0.000 |         0.000 |         0.000 |           0.000 |          0.000 |         0.000 |          0.000 |
| original @ 0.004    |           0.000 |           0.000 |         0.000 |         0.000 |           0.000 |          0.000 |         0.000 |          0.000 |
| original @ 0.012    |           0.000 |           0.000 |         0.000 |         0.000 |           0.000 |          0.000 |         0.000 |          0.000 |

**small_n**

| method              |   f=0.05,d=0.25 |   f=0.08,d=0.25 |   f=0.1,d=0.2 |   f=0.15,d=0.2 |
|:--------------------|----------------:|----------------:|--------------:|---------------:|
| EIGER 2 correlation |           0.944 |           1.000 |         1.000 |          1.000 |
| EIGER 2 covariance  |           1.000 |           1.000 |         1.000 |          1.000 |
| original @ 0.002    |           0.000 |           0.056 |         0.000 |          0.000 |
| original @ 0.004    |           0.000 |           0.000 |         0.000 |          0.000 |
| original @ 0.012    |           0.000 |           0.000 |         0.000 |          0.000 |

**celltype**

| method              |   f=0.5,d=0.35 |   f=0.042,d=0.45 |   f=4,d=0.18 |
|:--------------------|---------------:|-----------------:|-------------:|
| EIGER 2 correlation |          1.000 |            1.000 |        1.000 |
| EIGER 2 covariance  |          1.000 |            0.333 |        1.000 |
| original @ 0.002    |          1.000 |            1.000 |        1.000 |
| original @ 0.004    |          1.000 |            1.000 |        1.000 |
| original @ 0.012    |          1.000 |            0.000 |        1.000 |

**celltype_nuisance**

| method              |   f=0.5,d=0.35 |   f=0.042,d=0.45 |   f=4,d=0.18 |
|:--------------------|---------------:|-----------------:|-------------:|
| EIGER 2 correlation |          1.000 |            1.000 |        1.000 |
| EIGER 2 covariance  |          1.000 |            0.333 |        1.000 |
| original @ 0.002    |          1.000 |            0.467 |        1.000 |
| original @ 0.004    |          1.000 |            0.200 |        1.000 |
| original @ 0.012    |          1.000 |            0.000 |        1.000 |


## Table 5 — effect of the length-threshold bug

| scenario          | method                     |   cpg_recall |   cpg_precision |   cpg_f1 |
|:------------------|:---------------------------|-------------:|----------------:|---------:|
| celltype          | original (as written)      |        0.672 |           0.726 |    0.696 |
| celltype          | original (length fixed)    |        0.675 |           0.726 |    0.698 |
| celltype          | original + mean imputation |        0.672 |           0.726 |    0.696 |
| celltype_nuisance | original (as written)      |        0.649 |           0.689 |    0.662 |
| celltype_nuisance | original (length fixed)    |        0.655 |           0.692 |    0.650 |
| celltype_nuisance | original + mean imputation |        0.649 |           0.689 |    0.662 |
| clean             | original (as written)      |        0.425 |           0.632 |    0.440 |
| clean             | original (length fixed)    |        0.449 |           0.577 |    0.422 |
| clean             | original + mean imputation |        0.425 |           0.632 |    0.440 |
| missingness       | original (as written)      |        0.000 |           0.000 |    0.000 |
| missingness       | original (length fixed)    |        0.000 |           0.000 |    0.000 |
| missingness       | original + mean imputation |        0.318 |           0.547 |    0.350 |
| nuisance          | original (as written)      |        0.300 |           0.500 |    0.312 |
| nuisance          | original (length fixed)    |        0.338 |           0.451 |    0.296 |
| nuisance          | original + mean imputation |        0.300 |           0.500 |    0.312 |
| small_n           | original (as written)      |        0.147 |           0.306 |    0.159 |
| small_n           | original (length fixed)    |        0.185 |           0.301 |    0.166 |
| small_n           | original + mean imputation |        0.147 |           0.306 |    0.159 |
