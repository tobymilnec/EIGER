# Layout fixture validation

## EPIC-seq fitted generator vs source

Source figures are computed from the patient sample and recorded in
`epic_layout_params.json`; no source coordinates are committed. The generator
is seeded and must reproduce each figure within 10%.

### chr1 (n_cpg = 816,866, seed 0)

| metric | source | generated | rel. error | |
|---|---|---|---|---|
| eligible fraction | 0.8885 | 0.8887 | 0.0% | OK |
| run median (edges) | 3.0000 | 3.0000 | 0.0% | OK |
| run q90 (edges) | 15.0000 | 15.0000 | 0.0% | OK |
| frac runs <3 edges | 0.4600 | 0.4625 | 0.5% | OK |
| depth median | 5.0000 | 5.0000 | 0.0% | OK |
| frac depth>=5 | 0.5134 | 0.5135 | 0.0% | OK |
| frac depth>=10 | 0.4088 | 0.4085 | 0.1% | OK |

### chr7 (n_cpg = 535,653, seed 0)

| metric | source | generated | rel. error | |
|---|---|---|---|---|
| eligible fraction | 0.8925 | 0.8946 | 0.2% | OK |
| run median (edges) | 3.0000 | 3.0000 | 0.0% | OK |
| run q90 (edges) | 16.0000 | 16.0000 | 0.0% | OK |
| frac runs <3 edges | 0.4646 | 0.4667 | 0.5% | OK |
| depth median | 5.0000 | 5.0000 | 0.0% | OK |
| frac depth>=5 | 0.5003 | 0.5005 | 0.0% | OK |
| frac depth>=10 | 0.3923 | 0.3925 | 0.0% | OK |

### chr19 (n_cpg = 532,829, seed 0)

| metric | source | generated | rel. error | |
|---|---|---|---|---|
| eligible fraction | 0.9364 | 0.9369 | 0.1% | OK |
| run median (edges) | 5.0000 | 5.0000 | 0.0% | OK |
| run q90 (edges) | 32.0000 | 33.0000 | 3.1% | OK |
| frac runs <3 edges | 0.3031 | 0.3005 | 0.9% | OK |
| depth median | 6.0000 | 6.0000 | 0.0% | OK |
| frac depth>=5 | 0.5485 | 0.5485 | 0.0% | OK |
| frac depth>=10 | 0.4418 | 0.4415 | 0.1% | OK |

Worst relative error across all chromosomes and metrics: **3.1%** (tolerance 10%).

## Real WGBS layout (committed fixture)

Read back from `wgbs_layout_hg38.npz`. These are measurements, not targets --
the fixture holds real hg38 coordinates and real depth.

| chromosome | CpGs | eligible % | runs | run q50 | run q90 | max run | runs <3 | depth q50 | depth>=10 |
|---|---|---|---|---|---|---|---|---|---|
| chr1 | 2,375,159 | 92.2% | 185,697 | 3 | 24 | 35,404 | 43% | 9 | 47.1% |
| chr7 | 1,622,825 | 92.1% | 128,323 | 3 | 22 | 15,446 | 44% | 9 | 46.8% |
| chr19 | 1,105,620 | 97.9% | 23,334 | 9 | 103 | 33,051 | 27% | 9 | 44.4% |

## For contrast: the layout every prior EIGER benchmark used

| layout | CpGs | eligible % | runs | run q50 | max run |
|---|---|---|---|---|---|
| `dense` (uniform 2 bp, one chromosome) | 20,000 | 100% | **1** | **19,999** | 19,999 |
