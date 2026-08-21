# EIGER layout fixture

Real and fitted CpG layouts for the EIGER benchmark, built for Stage 1d §A of
`EIGER_STAGE1D_REGION_NULL_SPEC.md`. Vendored into `benchmarking/layouts/` (this
directory) per master spec §6.1 — delivered and validated; not rebuilt as part of that
integration.

**`describe_layout()` and `eligible_run_lengths()` now live in the installed
`eiger2.layouts` module**, not here — every `eiger2 call` needs them on a user's own
coordinates, not just these fixtures, and only `eiger2*`/`functions*` ship
(`pyproject.toml`). `layouts.py` in this directory re-exports both, so `from layouts
import describe_layout` below still works and there remains exactly one implementation.

Every calibration number EIGER has produced came from a simulated layout of uniformly
spaced CpGs on a single chromosome — one eligible run of 19,999 edges, 100% of edges
eligible, no chromosome boundary ever crossed. Real data is roughly 150,000 runs per
chromosome with a median length of 3–5 edges. That gap is why the region-level null was
never validated against anything realistic.

## Contents

| file | what it is |
|---|---|
| `wgbs_layout_hg38.npz` | **real** hg38 CpG coordinates + real per-CpG depth, chr1/chr7/chr19 (5.1 M CpGs, 9.8 MB) |
| `wgbs_layout_hg38.provenance.json` | source accession, input checksums, what the fixture does and does not contain |
| `epic_layout_params.json` | fitted empirical distributions for a targeted panel, one profile per source chromosome |
| `layouts.py` | loader, generator, and `describe_layout()` — the manifest diagnostics |
| `extract_wgbs_layout.py` | rebuilds the WGBS fixture from the public sources |
| `fit_epic_layout.py` | refits the EPIC profiles from a Bismark `.cov.gz` |
| `validate_layouts.py` | regenerates `VALIDATION.md`; exits non-zero on any miss |
| `VALIDATION.md` | generator-vs-source table, and the real-WGBS measurements |

## What is in these files, and what is not

**WGBS fixture — public sources, coordinates and depth only.** Built from
`GSM5652215` (Loyfer et al. cell-type atlas, GEO) joined to the wgbstools hg38 CpG index
`CpG.bed.gz`. The `.beta` format is `uint8` pairs `(methylated, total)`; the extractor
reads column 1 and **discards column 0**, so the fixture carries no methylome — only
where CpGs are and how deeply they were sequenced. Positions are delta-encoded, which is
why 5.1 M CpGs fit in 9.8 MB.

**EPIC-seq — fitted distributions, no patient data.** The source is a patient tumour
sample and **no coordinates from it are committed**. `epic_layout_params.json` holds
only empirical distributions: within-run and between-run gap quantiles, the exact
run-length PMF, a per-CpG depth quantile table, and the latent depth-field parameters.
Methylation values are never read by the fitter — it uses columns 0, 1, 4 and 5 of the
`.cov` and sums 4+5 to a total depth.

Both extraction scripts are committed so either artefact can be rebuilt from its source.

## Use

```python
from layouts import load_wgbs_layout, generate_epic_layout, describe_layout

# real hg38 coordinates and depth
locations, depth = load_wgbs_layout("wgbs_layout_hg38.npz", chrom="chr1",
                                    n_cpg=20_000, offset=1_000_000)

# fitted targeted panel; `profile` picks the density regime to imitate
locations, depth = generate_epic_layout("epic_layout_params.json", n_cpg=20_000,
                                        seed=0, profile="chr1")

describe_layout(locations, max_cpg_gap_bp=300, depth=depth)
```

`locations` is the `chr`/`start`/`end` frame `eiger2.core.eligible_edges` expects.
`describe_layout` returns the manifest fields Stage 1d §A.2 requires, including
`layout_max_detectable_region_cpgs` — the hard ceiling on region length imposed by the
layout, since a region cannot span an ineligible edge.

Rebuild:

```bash
python extract_wgbs_layout.py --cpg-index CpG.bed.gz --beta SAMPLE.hg38.beta \
    --chroms chr1 chr7 chr19 --out wgbs_layout_hg38.npz
python fit_epic_layout.py --cov SAMPLE.bismark.cov.gz --chroms 1 7 19 \
    --out epic_layout_params.json
python validate_layouts.py
```

## Design notes

**Layouts are fitted per chromosome, not pooled.** chr19 is far denser than chr1 or
chr7 — 93.6% eligible and median run 5 against 88.8% and median run 3. A pooled fit
reproduced none of the three. There is no single "targeted layout"; pick the profile
whose density regime you want to test against.

**The generator builds run by run.** Run length is the property that matters, and
sampling gaps independently does not reproduce it. A run length is drawn from the exact
empirical PMF, then that many within-run gaps, then one between-run gap.

**Depth is generated in two steps.** A latent field is built as a per-run level plus an
AR(1) residual, which reproduces the spatial structure — 56–73% of the variance in log
depth sits *between* capture targets rather than within them, and adjacent log-depth
correlation is ~+0.76. That field is then rank-mapped onto the empirical per-CpG depth
quantile table, so the marginal is exact rather than approximated through a log/exp
round trip. An earlier version that sampled depth directly from a per-run level
distribution missed the median by 40%.

## Distribution: committed, deliberately

`wgbs_layout_hg38.npz` (9.6 MB) is kept committed to the repository rather than
attached as a release asset fetched on demand. It is derived entirely from public
sources (extraction script alongside it, above), and a benchmark that cannot be
reproduced without a separate download is a benchmark people will not reproduce. 9.6
MB is unremarkable against that trade-off. If this ever needs revisiting (repository
size becomes a real constraint), the alternative is a GitHub release asset with
`layouts.py` fetching or erroring — that adds a network dependency to every benchmark
run, which is the reason it was not chosen here.

## Caveats

**A contiguous slice is not a representative sample of a chromosome.** The first 20,000
CpGs of chr1 happen to fall in a dense region: 97.3% eligible with a maximum run of
4,749 edges, against 92.2% and a median run of 3 for the chromosome as a whole. Slices
must be taken at varied offsets, or taken large. Do not take a random subset of CpGs —
that destroys the run structure the fixture exists to provide.

**`max_cpg_gap_bp = 300` is load-bearing.** All-CpG inter-gap on chr1 is q50 = 40 bp,
q90 = 256 bp, q99 = 722 bp, so the default cutoff sits directly on the shoulder of the
distribution and run structure is acutely sensitive to it. Sweep it.

**One WGBS sample, one panel sample.** The fixture is a real layout, not a
population-representative one. Depth in particular is sample-specific.

**Run-length figures count singleton runs.** A lone CpG with no eligible neighbour is a
run of 0 edges. Ad-hoc measurements that drop empty runs report a higher median (4
rather than 3 on EPIC chr1) — the figures here include them, because the generator has
to reproduce lone CpGs too.
