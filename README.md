<img width="1408" height="768" alt="EIGER banner" src="https://github.com/user-attachments/assets/32576634-76bb-40e6-9a16-bf4ae769dd5c" />

# EIGER: Estimating Interdependent Genome-wide Epigenetic Regions

**Label-free discovery of regionally coordinated DNA methylation, with a
region-level empirical false discovery rate.**

EIGER segments the genome into blocks in which neighbouring CpGs co-vary across
samples, without requiring predefined sample groups. Conventional DMR callers
test a specified contrast between known classes; EIGER discovers the structure
first and lets labels be applied afterwards, so cell types, tumour status and
unanticipated subgroups can be found in one pass without any of them being
specified in advance.

Both the method as originally published and the revised method are in this
repository and run from one command, so their outputs can be compared directly
on identical inputs.

```bash
conda env create -f environment.yml && conda activate eiger
pip install -e . --no-deps

python examples/make_example_data.py
eiger2 call --betas examples/data/betas.tsv \
            --locations examples/data/cpgs.bed --out results/
```

Installation, input formats and every option: **[QUICKSTART.md](QUICKSTART.md)**.

---

## What changed in version 2, and why

The original caller thresholds the adjacent-CpG **covariance**. For a two-group
structure with minor fraction *f* and effect δ that quantity is exactly
*f*(1−*f*)δ², so what is thresholded is not the effect size but the effect size
scaled by group imbalance. At the published threshold the smallest detectable
effect is 0.22 at *f* = 0.50 and 0.50 at *f* = 0.05: rare subgroups are
undetectable by construction, and the prevalence corrections applied downstream
operate only on regions that the filter has already excluded.

Four changes follow from that diagnosis.

| Change | Reason |
|---|---|
| **Pairwise-complete estimation** | Removes the (1−*m*)² covariance deflation caused by mean imputation. At 5% missingness with 24 samples, complete-case estimation reports *no regions at all*, so every earlier result depends on imputation. |
| **Correlation as the default statistic** | ρ = *f*(1−*f*)δ² / (*f*(1−*f*)δ² + σ²) saturates for any *f* once δ clears the noise, removing the prevalence ceiling. |
| **A locality-breaking null** | Pairs CpGs far apart on the same chromosome, so every cross-sample dependency survives and only adjacency is destroyed. That is the hypothesis the caller actually tests. |
| **A region-level empirical FDR** | Null tracks are pushed through the *identical* segmenter and region areas compared within CpG-count strata. The published method defines no error rate; this supplies one. |

Measured effects, and the conditions under which each claim holds, are in
**[docs/](docs/)**. Two limitations bound the method and are stated here because
they matter as much as the improvements:

- **Structured missingness defeats the locality-breaking null.** Under 25%
  dropout the realised FDR is 0.24–0.35 against a nominal 0.05 and is
  insensitive to every segmentation parameter — a constant bias, not a tuning
  problem.
- **The per-region evidence score is not calibrated.** It is reported as a
  ranking and explicitly not as a probability. `docs/PROBABILITY.md` reports the
  measurement that established this.

### Large cohorts

Above roughly 100 samples the default null asks a question that has become
trivially true — regional coordination exists nearly everywhere — and q-values
saturate at zero. Use the permutation null, which preserves the observed
marginal distribution of coordination exactly and randomises only its
arrangement along the chromosome:

```bash
eiger2 call --betas betas.tsv --locations cpgs.bed --out results/ \
    --null-mode permutation --min-prevalence-samples 3
```

On a 519-sample cohort the default null gave q = 0 for all 73,784 regions it
called; the permutation null gave a genuine split, 41,636 at q ≤ 0.05 against
32,224 above it. The exact commands, and the reason the two runs do not call
quite the same region set, are in [analysis/](analysis/README.md).

---

## Running the method as originally published

`--legacy` invokes `Python/functions/processing.py` directly rather than
reimplementing it, so the baseline cannot be a paraphrase.

```bash
eiger2 call    --betas betas.tsv --locations cpgs.bed --out legacy/ --legacy
eiger2 compare --betas betas.tsv --locations cpgs.bed --out both/
```

**Concordance with the published prostate cohort calls: all 8,915 published
regions are recovered, at every region size** — 100% at ≤200 bp, at 200 bp–10 kb
and at >10 kb. `--legacy` reports more regions than the published set because
the published analysis applied further filtering downstream; no published call
is lost.

An FDR can be attached to the published method without changing which regions it
reports, by permuting its own covariance track:

```bash
eiger2 call --betas betas.tsv --locations cpgs.bed --out legacy/ \
    --legacy --fdr-replicates 50
```

---

## Repository layout

```
Python/eiger2/        the installable package: core, cli, io, probability,
                      and the simulator used for validation
Python/functions/     the original 2023 implementation, invoked by --legacy
benchmarking/         benchmark and validation harness, plus generated results
old_code/             the superseded 2026 beta, its tests, and review scripts
scripts/              cohort-specific analysis, kept out of the package
analysis/             the exact commands behind every published number
examples/             a runnable example dataset generator
tests/                tests for the shipped package
docs/                 method design, benchmarks, and the probability layer
```

Only `Python/eiger2` and `Python/functions` are installed. The benchmark
harness, the superseded beta and the analysis scripts are present for review and
reproducibility but are deliberately not shipped as library code.

```bash
pytest tests -q             # 48 tests, the shipped package
pytest old_code/tests -q    # 27 tests, the superseded beta
```

---

## Outputs

`regions.csv` carries one row per region. **`region_qvalue` is the only field
that expresses confidence that a region is real.** Every `*_descriptive`
field — delta beta, direction, the inferred sample split — is computed *after*
the region was selected, on the same data, and is for reading rather than
filtering. `run_manifest.json` records the resolved configuration, input
checksums, package versions and the realised null diagnostics for every run, so
a result can always be traced to the code and data that produced it.

## Licence and citation

MIT. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
