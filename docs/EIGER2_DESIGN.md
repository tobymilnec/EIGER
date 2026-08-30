# EIGER 2 — stripped back and rebuilt

A design specification and a working reference core. Every number in this
document is produced by `Python/eiger2/benchmark.py` and locked by
`tests/test_eiger2_core.py`, so the claims can be re-checked whenever the code
changes.

---

## 0. The three versions at a glance

| | **Original (2023)** `processing.py` | **Beta (2026)** `eiger_core.py` | **EIGER 2** `eiger2/` |
|---|---|---|---|
| Discovery signal | adjacent covariance | adjacent covariance | correlation *or* covariance |
| Threshold | fixed 0.012 | fixed, or auto from a per-sample resampling null | from a locality-breaking null that keeps sample structure |
| Nuisance handling | none | none | deflate diffuse components only |
| Missing data | NaN breaks the region silently | mean-imputed, deflates signal by (1−m)² | pairwise-complete |
| Segmentation | run of passing edges | run, plus a 2-state smoother with hand-set parameters | run, plus a gap tolerance calibrated on the null |
| Error rate | none | BH on a post-selection p-value | region-level empirical FDR |
| Subgroup model | **none — any number of groups** | binary via largest gap, then a 2-component GMM | **none for discovery**; binary split only in the descriptive fields |
| Cell-type detection | one pass, free | one pass, free | one pass, free |
| Rare subgroups | limited by `f(1−f)δ²` | same limit | correlation removes the limit |
| Ranking systems | 3 | 7 | 1 |
| Lines | 466 | ~2,700 across 4 files | ~950 |

The beta version changed the machinery around the statistic without changing the
statistic, and added six ranking systems. EIGER 2 changes the statistic and the
calibration, and removes the ranking systems.

---

## 1. The original idea, and what is worth keeping

The 2023 `processing.py` does one distinctive thing:

> find contiguous genomic blocks where neighbouring CpGs **co-vary across
> samples**, using no labels at all.

That is a good idea and it is not what `bsseq`, `dmrseq`, `metilene` or
`bumphunter` do — they all need a grouping variable. Label-free discovery of
regional coordination is EIGER's reason to exist, and it survives the rebuild
unchanged.

What has to go is everything that grew around it. The original 466-line file
mixes the discovery statistic with cell-type annotation, tissue annotation,
three ranking scores, two prevalence corrections and a feature-selection score,
all inside one 260-line function with hard-coded metadata column indices. The
2026 version added a further five ranking systems on top. None of that is the
idea; all of it is scaffolding that obscured a statistic which, it turns out,
could not detect the thing the paper is about.

**Kept:** label-free discovery; adjacent co-variation as the signal; contiguous
blocks; the discovery/annotation separation the 2026 version introduced (good
decision — group labels must not move region boundaries); the compact-versus-
diagnostic output split; the run manifest.

**Dropped:** `find_separators` in its entirety; cell-type and tissue branches;
`f_rare_gaussian` / `f_rare_logistic`; `descriptive_score`; `prospective_power`;
the Laplacian score; stratified rank percentiles; the BIC "model probability";
the bootstrap stability estimate; the two-state smoother with hand-set
parameters. That is seven competing ranking systems replaced by two numbers: a
region-level q-value for *is it real*, and a delta beta for *how big*.

---

## 2. Stringent review of the original `processing.py`

Ten findings, each verified against the uploaded file.

### 2.1 The discovery statistic cannot see rare subgroups — the central problem

For a two-group structure with minor fraction `f`, effect `δ` and within-group
noise `σ`, the adjacent covariance is exactly

```
cov = f(1-f)·δ²
```

Simulation matches the identity to three significant figures. At the default
threshold of 0.012 this fixes a hard detection floor:

| minor fraction `f` | minimum detectable `δ` |
|---|---|
| 0.50 | 0.219 |
| 0.25 | 0.253 |
| 0.10 | 0.365 |
| 0.05 | 0.503 |
| 0.03 | 0.642 |
| 0.01 | 1.101 — impossible |

In a 519-sample cohort a 3-sample subgroup with a 30-point methylation
difference produces an adjacent covariance of **0.00051**, twenty-three-fold
below threshold. It is never discovered. The prevalence corrections
(`f_rare_gaussian`, `f_rare_logistic`, `N_target=3`) exist to up-weight exactly
those subgroups, but they run on regions that have already passed the filter
that excluded them. The rare-group machinery is at the wrong end of the
pipeline.

The fix is to change the statistic. Correlation for the same structure is

```
ρ = f(1-f)·δ² / (f(1-f)·δ² + σ²)
```

which **saturates towards 1 for any `f`** once `δ` clears the noise. Measured at
`n = 519`, `σ = 0.03`:

| `f` | `δ` | covariance | vs 0.012 | correlation |
|---|---|---|---|---|
| 0.0058 | 0.30 | 0.00051 | fail | **0.363** |
| 0.0058 | 0.60 | 0.00208 | fail | **0.699** |
| 0.019 | 0.30 | 0.00170 | fail | **0.654** |
| 0.050 | 0.30 | 0.00429 | fail | **0.827** |
| 0.50 | 0.30 | 0.02257 | pass | 0.962 |

### 2.2 `extract_blocks` applies its length threshold to edges, not CpGs

```python
if (stop - start >= DMR_length_threshold):
```

`start` and `stop` index the covariance array. A run of `E` passing edges spans
`E + 1` CpGs, so `DMR_length_threshold=5` actually requires six CpGs. Verified:
a genuine five-CpG block (four passing edges) returns `[]`. The returned
`[start, stop]` pair is then ambiguous downstream — `stop` is a CpG index if the
consumer uses `arange(start, stop+1)` and an exclusive bound if it uses
`arange(start, stop)`.

### 2.3 Group-level separation can only ever find the *lowest* group

```python
gap_ct = sorted_ct[1][1] - sorted_ct[0][1]
separated_celltype = sorted_ct[0][0]
```

Only the gap between the two lowest group means is examined, and the separated
group is always the lowest. A hypermethylated cell type or tissue is
structurally undetectable, regardless of effect size. The same asymmetry hits
the sample level: when the rare group sits at the *top* of the sorted order,
`sample_values` is assigned the large group, so `p_obs_sample` is the majority
fraction and the prevalence correction penalises the region as though it were
common.

### 2.4 The `p == 0` clamp is non-monotone in evidence

```python
if pval_sample == 0:
    pval_sample = 1e-4
```

| p | `-log10(p)` used |
|---|---|
| 0.0 | **4.00** |
| 1e-12 | 12.00 |
| 1e-06 | 6.00 |
| 1e-03 | 3.00 |

The strongest possible evidence scores *below* moderate evidence. Every region
whose t-test underflows is silently demoted.

### 2.5 The Laplacian graph uses distance where similarity is required

```python
S = kneighbors_graph(X, neighbor_size + 1, mode='distance', metric='euclidean')
```

In He, Cai and Niyogi's Laplacian Score, `S` must be a *similarity* — the heat
kernel `exp(-‖xᵢ-xⱼ‖²/t)`, so that near neighbours get large weights. With
`mode='distance'` the nearest neighbour in each neighbourhood receives a weight
of 0.0058 and the farthest retained neighbour receives 0.4218: the graph encodes
dissimilarity, and the quadratic form `fᵀLf` no longer measures local
smoothness. `S` is also never symmetrised, so `L = D − S` is not symmetric.

### 2.6 Student's t with wildly unbalanced groups

`stats.ttest_ind` is called without `equal_var=False` throughout. With group
sizes of 3 versus 500 and unequal variances the pooled-variance test is
strongly anti-conservative. Welch is required here, not optional.

### 2.7 The score double-counts prevalence and barely responds to effect size

`base_score = (|delta_minmax| / sd) · -log10(p)`, where `sd` is the standard
deviation of the *pooled* sample means. For a two-group split with small
within-group variance, `sd ≈ sqrt(f(1-f))·|δ|`, so the standardised term
collapses to `1/sqrt(f(1-f))` — a function of prevalence alone. Measured at
`n = 100`:

| `f` | δ=0.2 | δ=0.4 | δ=0.8 | `1/√(f(1−f))` |
|---|---|---|---|---|
| 0.03 | 5.129 | 5.629 | 5.816 | 5.862 |
| 0.10 | 3.174 | 3.284 | 3.321 | 3.333 |
| 0.50 | 1.966 | 1.988 | 1.997 | 2.000 |

A four-fold change in effect size moves the score by under 14%. The score is
then multiplied by a prevalence correction, correcting for prevalence a second
time. Also note that `delta_minmax` is the *gap* between adjacent sorted samples
while the p-value tests the difference in *means* — two different quantities
multiplied together.

### 2.8 No missing-data handling anywhere

`np.cov` propagates NaN, `NaN >= threshold` is `False`, so a single uncovered
sample at one CpG silently splits a region. There is no imputation, no
pairwise-complete option and no record that it happened. (The 2026 version added
mean imputation, which is worse in a subtler way — it deflates covariance by
exactly `(1-m)²` at missing rate `m`, verified: 0.83× at 10% missing, 0.66× at
20%, 0.36× at 40%.)

### 2.9 Hard-coded metadata column indices

`tissue_col=4, celltype_col=3, subtype_col=2` positional indices, with
`subtype_col` accepted and never used. Any change to the metadata file silently
produces wrong annotations rather than an error.

### 2.10 Minor

`itertools`, `matplotlib`, `seaborn` imported and unused. `contiguous_regions`
requires a 2-D boolean array because of `idx, _ = d.nonzero()`, which is why
`extract_blocks` reshapes to `(-1, 1)` — undocumented coupling.
`expand_averages_column` indexes `avg_list[0]` without checking for an empty
frame. `correlation_blocks` is dead code. `compute_laplacian_score` has no guard
for `n_samples <= neighbor_size`, where the graph becomes near-complete and all
scores collapse to 1.

---

## 3. What EIGER 2 does

Five steps, each a separate function in `Python/eiger2/core.py`.

```
betas ──▶ centre ──▶ deflate ──▶ local statistic ──▶ segment ──▶ region FDR
                        ▲              ▲                 ▲            ▲
                  thinned basis   pairwise-complete   same segmenter  │
                                                      on null tracks ─┘
```

### Step 1 — centre

Subtract the per-CpG mean over observed samples. NaN stays NaN; nothing is
imputed.

### Step 2 — remove genome-wide latent structure

Estimate `K` orthonormal directions in sample space and project them out of
every CpG. This is the step that makes a label-free threshold possible at all.

**Why it is needed.** Tumour purity, immune infiltration, cell composition and
batch induce covariance between *any* two CpGs in the genome, adjacent or not.
On a DMR-free matrix with a single latent factor, every adjacent correlation is
lifted to ≈0.89, and a planted DMR at ρ=0.766 sits *below* the null threshold.
Correlation alone is not enough; deflation is not optional.

**Only diffuse components are removed.** This is the correction that keeps
EIGER's original vision intact, and the first draft of this document got it
wrong.

Cell type *is* a genome-wide latent factor. So is tumour status. Deflating "the
latent structure" therefore deletes precisely the biology EIGER exists to find.
Measured on a cohort of 4 cell types (40 marker regions) plus tumour/benign
(20 regions) plus a rare 5-in-120 subgroup (10 regions):

| | cell type | tumour | rare | false |
|---|---|---|---|---|
| blind deflation, K by parallel analysis (K=5) | **0/40** | **0/20** | 0/10 | 43 |
| no deflation (K=0) | 40/40 | 20/20 | 10/10 | 0 |

Blind deflation is catastrophic, and parallel analysis walks straight into it
because five components genuinely *are* present — they are just all signal.

The two kinds of component can be separated without labels. A nuisance factor
loads on the whole genome; a biological subgroup loads on marker regions. The
fraction of a component's loading energy sitting in the top 1% of CpGs
separates them cleanly:

| scenario | component | loading concentration | verdict |
|---|---|---|---|
| nuisance factor + focal DMRs | 0 | 0.081 | diffuse → deflate |
| | 1 | 0.146 | concentrated → keep |
| 4 cell types + tumour + rare | 0–4 | 0.180–0.400 | concentrated → keep |
| both at once | 0 | 0.081 | diffuse → deflate |
| | 1–5 | 0.181–0.400 | concentrated → keep |

`deflation_max_concentration = 0.12` implements the cut. Results with it:

| cohort | method | cell type | tumour | rare | false | K |
|---|---|---|---|---|---|---|
| no nuisance | original, cov ≥ 0.012 | 40/40 | 20/20 | **0/10** | 0 | – |
| | EIGER 2 correlation | 40/40 | 20/20 | **10/10** | 0 | 0 of 5 |
| | EIGER 2 covariance | 40/40 | 20/20 | 0/10 | 0 | 0 of 5 |
| plus nuisance | original, cov ≥ 0.012 | 40/40 | 20/20 | **0/10** | 1 | – |
| | EIGER 2 correlation | 40/40 | 20/20 | **10/10** | 0 | 1 of 5 |

With no nuisance present, nothing is deflated and EIGER 2 reduces to the
original's behaviour plus the rare-subgroup gain. With a nuisance factor
present, exactly one of five candidates is removed. Note also that the rare
subgroup is recovered only by the *correlation* statistic — covariance misses it
either way, which is §2.1 showing up in a realistic cohort.

**Where 0.12 comes from, and what the cut is sensitive to.** The table above is
measured on the synthetic marginal, where the two bands are far apart and the
cutoff can sit anywhere between them. Real beta values are bimodal, and on a
realistic marginal the *nuisance* band rises: fitted to purified leukocyte
cell types it runs to 0.145 with a 95th percentile of 0.1146, so the original
0.10 kept a nuisance component in 158 of 280 simulated runs and the deflation
stage was half switched off on exactly the data type the method is for. 0.12 is
set from both sides and nothing else (`specs/publication/FIX_SPEC.md` §WP7b):

| side | measurement |
|---|---|
| nuisance bands it must cover | 98.6% of atlas-marginal candidates, 100% of ICGC-fitted (max 0.1095), 100% of synthetic (max 0.0850) |
| recall it recovers | small-cohort (n = 12) recall 0.46–0.56 → 0.72–0.83 |
| real cohorts it must not touch | immune atlas 0.1258 genome-wide (26.3 M CpGs × 36 samples), ICGC 0.1335 — both kept |
| ruled out from above | 0.15 deflates both, and churns the ICGC significant set (Jaccard 0.328) |

Two real cohorts at 0.1258 and 0.1335 is the evidence there is. It is a
measurement on two datasets, not a guarantee about WGBS in general, and the
margin at 0.12 is 0.0058 on the tighter of the two.

**Deflation verdicts have the scope of the basis estimation.** The
concentration is the top-1% energy share of a component fitted to the matrix in
front of it, so the same nuisance factor scored one chromosome at a time
scatters about ±0.05 around its genome-wide value — the immune atlas scores
0.1074–0.1802 per chromosome against 0.1258 pooled. Estimate the basis
genome-wide where you can; per-chromosome runs are a memory workaround, and
their deflation verdicts carry that scatter. Either way the run tells you when
it was close: a candidate within `DEFLATION_NEAR_CUTOFF_MARGIN` (0.02) of the
cutoff is named in `manifest["latent_structure"]["near_cutoff_components"]` and
in one stderr note. At 0.12 that band is (0.10, 0.14), which both real cohorts
above sit inside — they are kept, and kept by a margin another cohort could
close.

**Why the basis is estimated on a thinned CpG set.** A second, complementary
protection. The concentration filter stops *many* focal regions from being
deflated collectively; thinning stops a *single* focal DMR from entering the
basis at all. Measured, one 40-CpG DMR (0.7% of the matrix) with one latent
factor:

| basis estimated on | median ρ inside the DMR after deflation | background 99.5th pct |
|---|---|---|
| all CpGs | **−0.063** — signal destroyed | 0.324 |
| every 50th CpG | **+0.591** — signal survives | 0.332 |
| every 200th CpG | +0.534 | 0.336 |

A 40-CpG region contributes at most one row to a 1-in-50 thin and cannot form a
component; a genuine genome-wide factor is fully represented. `K` is chosen by
parallel analysis on the thinned set (permute within CpG, take the 99th
percentile of null singular values), which recovers the true `K` exactly for
`K = 1, 2, 3`.

**Missing data must be handled exactly here.** Filling gaps with the CpG mean
before projecting leaves **44%** of the latent factor in high-missingness CpGs
versus 0% for per-CpG least squares on observed samples only. Those
under-deflated CpGs then correlate with each other, and runs of them become
false regions. At 25% structured missingness this alone is the difference
between 45/48 recovered with 17 false regions (exact) and 34/48 with 60 false
(imputed). `deflation_missing="exact"` is the default and costs 0.3 s on a
19,600 × 80 matrix.

### Step 3 — the local co-variation statistic

Pairwise-complete covariance or correlation, computed over the samples observed
at *both* CpGs and divided by `n_both − 1`. This removes the `(1-m)²` deflation
exactly rather than approximately.

`window = w > 1` replaces the single adjacent pair with the mean off-diagonal
statistic of the `w × w` block, which cuts the null standard deviation roughly
`sqrt(w(w-1)/2)`-fold at no cost in the signal case:

| `n` | true ρ | single pair | 6-CpG window |
|---|---|---|---|
| 10 | 0.0 | −0.007 ± 0.335 | −0.004 ± **0.081** |
| 30 | 0.0 | −0.003 ± 0.188 | +0.003 ± **0.050** |
| 30 | 0.5 | +0.496 ± 0.141 | +0.495 ± 0.082 |

Mean off-diagonal correlation is preferred over the leading eigenvalue
`(λ₁−1)/(w−1)`, which carries a positive bias under the null (+0.249 at `n`=10).

### Step 4 — the locality-breaking null

The window at null position `i` is built from rows `i, i+L, i+2L, …` instead of
`i, i+1, i+2, …`, for a random large lag `L`. **Every sample moves together**, so
batch, purity, composition and every other cross-sample dependency survives
intact; consecutive null positions still slide by one CpG, so locally
autocorrelated nuisance survives too. Only adjacency — and therefore only
regional coordination — is broken.

This is precisely where EIGER 1's automatic threshold failed. Its null resampled
each *sample* independently, which also destroys the shared latent factor:

| latent loading SD | true 99.5th pct of adjacent cov | EIGER 1 auto threshold | false blocks on DMR-free data |
|---|---|---|---|
| 0.04 | 0.00458 | 0.00142 | 6 |
| 0.08 | 0.01725 | 0.00351 | 36 |
| 0.15 | 0.05198 | 0.00928 | 58 |

The fixed 0.012 threshold called **zero** false blocks in the first two rows, so
the adaptive threshold was strictly worse than the constant it replaced. The
lag null restores **edge-level** calibration: 0.4–0.6% of DMR-free edges pass,
against a nominal 0.5%. This number is unchanged by v2.1.0 and still describes
`null_mode="lag"`, the only edge-level null.

**As of v2.1.0 this is only half the calibration story.** Region-level FDR
(the next step) is backed by a *separate* null construction,
`region_null_mode`, not directly by the edge-level pass rate above — an
edge-level null passing this check says nothing about whether the SAME null,
segmented the same way as the data, forms multi-edge regions at the right
rate, which is what region-level FDR actually depends on. That distinction —
and the two defects (missing autocorrelation, then a per-stratum adequacy
floor that refused decisive `K=0` evidence) found and fixed by validating the
region-level null on its own terms rather than assuming the edge-level result
above covered it — is `docs/FDR.md` §11–§12's subject, not this document's.

### Step 5 — segmentation and region-level FDR

Contiguous runs of passing edges, with `max_gap_edges` tolerating short internal
gaps. `max_gap_edges=2` bridges a single dropped CpG, which breaks *two*
consecutive edges and is the commonest cause of spurious fragmentation — the
case EIGER 1's two-state smoother could not handle by construction.

Region strength is the area above threshold, `Σ(statistic − threshold)`, which
rewards both strength and length. The q-value comes from pushing the null tracks
through the **identical segmenter** and comparing area distributions.

This is the structural pay-off of the design: because the heuristics are applied
to the null as well as the data, `window`, `max_gap_edges` and `min_cpgs` do not
each need a separate justification. Whatever they do to the data they also do to
the null, and the reported error rate absorbs it.

### Characterisation is downstream and descriptive

`delta_beta_descriptive`, `direction_descriptive`, `minor_group_*` and
`pvalue_descriptive` are computed after a region is called, on the split that
maximises separation in the same data. They are named `*_descriptive` and no
q-value is attached to any of them. Only `region_qvalue` expresses confidence
that the region is real.

This matters because the post-selection bias is large: on data drawn from a
single Gaussian with no structure at all, EIGER 1's selected split produced a
mean `|delta|` of **0.213** against a true value of 0 — above its own 0.20
calling threshold — and its BIC-derived `model_probability` exceeded 0.8 in
**60%** of replicates.

---

## 4. Measured behaviour

`python -m eiger2.benchmark` from the `Python` directory. 19,600 CpGs, 80
samples, two latent factors, 48 planted 40-CpG DMRs.

**Clean data**

| method | recovered | false regions |
|---|---|---|
| original: adjacent covariance ≥ 0.012 | 12/48 | 45 |
| EIGER 2, covariance | **48/48** | **0** |
| EIGER 2, correlation | **48/48** | **0** |

The original recovers nothing below `f = 0.25`; EIGER 2 recovers 6/6 in every
stratum down to `f = 0.05, δ = 0.35`.

**Note this carefully:** with deflation and a calibrated threshold, *covariance
also reaches 48/48*. In this scenario the fixed 0.012 threshold was a bigger
problem than the choice of statistic. Correlation's advantage is specific to the
rare-and-strong corner (§2.1), where covariance is intrinsically blind.

**25% structured missingness**

| statistic | window | recovered | false regions |
|---|---|---|---|
| correlation | 1 | 45/48 | 17 |
| correlation | **4** | **48/48** | **1** |
| covariance | 1 | 45/48 | 18 |
| covariance | **4** | **48/48** | **0** |

**Small cohort, rare subgroups, low SNR** (`σ = 0.06`, `f` ∈ 0.05–0.15,
`δ` ∈ 0.20–0.25)

| `n` | statistic | window=1 | window=6 |
|---|---|---|---|
| 20 | correlation | 0/24, 32 false | **24/24, 1 false** |
| 20 | covariance | 9/24, 42 false | **24/24, 0 false** |
| 40 | correlation | 6/24, 60 false | **24/24, 0 false** |
| 40 | covariance | 21/24, 21 false | **24/24, 0 false** |

**DMR-free data, 12 runs** (the honest cost of windowing)

| window | correlation | covariance |
|---|---|---|
| 1 | 0 false regions | 0 |
| 4 | 4 | 4 |
| 6 | 11 | 8 |

`window=4` is the default: it buys the missingness and small-`n` robustness
above at the price of a few false regions per twelve DMR-free genomes. Use
`window=1` when specificity matters more than sensitivity.

---

## 5. Honest limitations

**Broad differential methylation is not identifiable.** If a subgroup differs
across a large fraction of the genome, that difference *is* a latent factor and
deflation removes it. Measured, with the basis on a 1-in-50 thin:

| DMR spans | median ρ inside after deflation |
|---|---|
| 40 CpGs (0.7%) | +0.591 — detected |
| 400 CpGs (6.7%) | +0.160 — below threshold |
| 1200 CpGs (20%) | +0.059 — invisible |

EIGER 2 is a **focal** DMR caller. Global hypomethylation, large-scale PMD
changes and whole-arm effects need a different method and external labels.
This is a statement about identifiability, not an implementation gap.

**The null does not cover locally-structured technical artefacts.** No
locality-breaking null can distinguish "locally coordinated because of biology"
from "locally coordinated because of an artefact that is itself local". Runs of
low-coverage CpGs are the main example. That is what exact deflation and the
`window` parameter are for; they are nuisance controls, not statistical ones.

**`min_paired_fraction` is off by default and is a trap.** Raising it from 0.0
to 0.5 at 25% missingness dropped recovery from 45/48 to 0/48 while *increasing*
false regions from 17 to 136 — it removes DMR edges faster than artefact edges
and fragments true regions below `min_cpgs`. Do not raise it without re-running
the benchmark on your own missingness pattern.

**`K` is the single most dangerous parameter.** Under-deflating leaves the
latent factor and everything correlates; over-deflating deletes signal. The
concentration filter and the thinned basis together make the automatic choice
safe for focal and moderately widespread regions, but not for broad ones. Always
check `manifest["latent_structure"]["component_verdict"]` and confirm that what
was deflated is what you expected. If a component you believe is biological was
classed `nuisance_deflated`, raise `deflation_max_concentration` or set
`deflate_components=0`.

**Not yet addressed.** Shared sequencing reads between adjacent CpGs generate
large false covariance below ~10× depth (mean adjacent covariance 0.015–0.031
from zero biology at 60% fragment sharing). EIGER 2 inherits this. The proper
fix is to estimate the two CpGs from disjoint read halves; until then, document
a depth floor.

---

## 6. Files

| File | Contents |
|---|---|
| `Python/eiger2/core.py` | The five steps, ~700 lines, no dependencies beyond numpy/pandas/scipy |
| `Python/eiger2/benchmark.py` | Every number in §4; `python -m eiger2.benchmark` |
| `tests/test_eiger2_core.py` | 23 tests, including operating-characteristic tests that would have caught each §2 finding |
| `PEER_REVIEW.md` | The full review of the 2026 version |

Not yet built: chromosome streaming, the file loaders, the CLI. The core takes a
matrix and locations; wiring it to the existing `load_functions.py` and
`dmr_pipeline.py` is the next step, and deliberately separate from getting the
statistics right.

```python
from eiger2 import Eiger2Config, call_dmrs

regions, values, manifest = call_dmrs(
    betas, locations,
    sample_names=samples,
    config=Eiger2Config(statistic="correlation", window=4),
)
significant = regions[regions["significant"]]
```
