# WP2 MultiScale — blocked on the region null, not on layout

**Status update (Stage 1e, master spec §8): the "revisit after Stage 1d-D" point has
arrived, and the answer is "still blocked, but for a smaller reason."** §4 below listed
two independent branches that were both unsound. The first -- "the evaluation lag-null
tracks inherit the zero-null-regions problem... a null built the way Stage 1b's fix
builds it forms almost no multi-edge structure at real layouts" -- is resolved for
`region_null_mode="member_step"` (Stage 1d-D/1e's C.3 null reproduces the observed
track's autocorrelation by construction and passes both region-level gates on real
layouts at genome scale; see `docs/FDR.md` §11). The second branch -- edge-wise
permutation as a fallback -- is not just unresolved, it no longer exists as a runnable
path at all (`null_mode="permutation"` now raises at `Eiger2Config` construction, Stage
1d-D Task 6.1). Neither branch was ever a layout objection, so §1-§3's correction below
still stands unchanged. A full re-evaluation of WP2 against the working region null
(block length `max(windows)` in place of the single `window`, R-3 Šidák normalisation
per §5) is real design work this release does not do -- it is a revive-and-redesign
task, not a documentation update, and is out of scope for Stage 1e's release-gate work.
This status line records that the blocking premise has narrowed, not that WP2 is ready.

**Original status: blocked pending master spec §6, revisit after Stage 1d-D.** This is the
master spec §7 measurement and decision record, produced in Stage 1d-A per that
section's instruction. Originally written as `docs/MULTISCALE_SHELVED.md` with status
"shelved" and layout cited as the reason; **that reasoning was wrong and has been
corrected** (Stage 1d-A review, 20 Aug 2026) — see §3 below for what was wrong and why.
The file was renamed accordingly. A recorded negative result is citable; a
contradicted one left standing is worse than not writing the document at all.

## 1. The claim originally under test

A window of `w` CpGs needs `w` consecutive gap-eligible CpGs. WP2 MultiScale's premise
is that widening `w` (1, 2, 4, 6, 8, 12, ...) gives a family of statistics at different
scales, trading resolution for power. The original version of this document argued that
premise fails because real layouts are fragmented (master spec §2.4: median eligible
run 3–4 edges on both WGBS and EPIC-seq) — reasoning that most **runs** are too short
for a wide window. **That reasoning does not survive being measured per edge**, which is
the unit EIGER actually processes; see §3.

## 2. Measurement

`eiger2.layouts.window_eligibility_fraction(locations, max_cpg_gap_bp, window)`,
committed and tested (`tests/test_layouts.py`, including a regression pin against the
exact numbers below). For eligible edge `i`, the window's base CpG index is `i - (0 if
window == 1 else window // 2 - 1)` — mirroring `core.call_dmrs`'s own `obs_offset0`
exactly, not a looser existence check — and the window is geometrically eligible at
edge `i` iff every one of the `window - 1` sub-edges it spans, starting at that base,
is itself gap-eligible: the whole window must lie inside one eligible run, at the
*specific* alignment EIGER actually computes.

Two views are reported, because the divergence between them is the substantive finding:

- **Edge-weighted** (`window_eligibility_fraction`, master spec §7's literal wording:
  "the fraction of eligible **edges**", and the *operative* one — see §3): what
  fraction of the genome's eligible edges admit an aligned `w`-CpG window there.
- **Run-count-weighted** (for contrast only): what fraction of the eligible runs
  themselves, by count — median 3–4 edges — are individually long enough for `w`
  CpGs at all, regardless of alignment.

`max_cpg_gap_bp = 300` throughout, matching the default. WGBS: full committed
chromosome (no subsampling). EPIC-seq: 200,000 CpGs generated from the fitted profile,
seed 0.

### 2.1 Edge-weighted, alignment-aware (the operative measure)

| layout | w=1 | w=2 | w=4 | w=6 | w=8 | w=12 |
|---|---|---|---|---|---|---|
| WGBS chr1 | 1.000 | 1.000 | **0.875** | 0.784 | **0.713** | 0.610 |
| WGBS chr19 | 1.000 | 1.000 | 0.964 | 0.938 | 0.907 | 0.863 |
| EPIC-seq (chr1 profile) | 1.000 | 1.000 | 0.819 | 0.696 | 0.610 | 0.498 |
| EPIC-seq (chr19 profile) | 1.000 | 1.000 | 0.884 | 0.794 | 0.723 | 0.615 |

(chr1's w=6 cell, 0.784, is measured but not separately cited in the master spec §7
table, which reports w=4/8/12 only; included here for the full swept grid.)

`w = 8` is available at 61–91% of eligible edges across these four layouts; `w = 4` at
82–96%. **This does not support a layout objection to MultiScale.**

### 2.2 Run-count-weighted (context only — not the operative measure)

| layout | w=1 | w=2 | w=4 | w=6 | w=8 | w=12 |
|---|---|---|---|---|---|---|
| WGBS chr1 | 1.000 | 0.807 | 0.572 | 0.438 | 0.352 | 0.242 |
| EPIC-seq (chr1 profile) | 1.000 | 0.809 | 0.539 | 0.378 | 0.276 | 0.162 |

## 3. Why the run-count view is the wrong one, and why this document originally used it

`w = 8` fits only 35% of runs by count (§2.2), but 71% of eligible **edges** (§2.1).
Both are correct measurements of different things: most runs are short, but most edges
live inside the rare long ones. **EIGER's window statistic is computed per edge, not
per run** — `edge_statistic()` evaluates a window at every eligible edge independently,
so the quantity that determines how often a wide window is usable is the edge-weighted
fraction, not the run-count fraction. Reporting the run-count number as if it answered
"how often can EIGER use `w = 8`" was reasoning from a per-run summary about a per-edge
question — the same class of error as the depth-fit issue caught two stages earlier in
this project (fitting depth from a per-run level distribution when depth is a per-CpG
quantity; see `benchmarking/layouts/README.md`, "Depth is generated in two steps").
Whenever a distribution is this heavy-tailed (chr1's eligible-run lengths: median 3
edges, max 35,404 — master spec §2.4), the unweighted view and the size-weighted view
diverge sharply, and the operative one is whichever matches the unit the method
actually processes. Both views are kept in this document (§2) so the divergence stays
visible for future readers, rather than only reporting the one that happened to support
a conclusion.

## 4. The actual reason WP2 is blocked

Independently of layout, both branches of the WP2 §2.5 region-null are unsound:

- The evaluation lag-null tracks inherit the zero-null-regions problem (master spec
  §2.3, §5) — a null built the way Stage 1b's fix builds it forms almost no multi-edge
  structure at real layouts, at any window scale.
- The "permute the combined score track" fallback is edge-wise permutation (block
  length 1), the anti-conservative case — measured at **zero null regions across 400
  replicates** (master spec §6.4; `tests/test_probability.py`,
  `test_qvalues_from_track_annotates_an_external_caller`, pinned in Stage 1c-bis).

That is sufficient to block on its own, and it is the only reason recorded here.

## 5. Decision

**Blocked pending master spec §6, revisit after Stage 1d-D** — not "shelved because the
data cannot support it". When §6 produces a working region null, MultiScale inherits it
with block length `max(windows)` in place of `window`, and the case should be
re-examined on its merits at that point, not treated as closed.

**R-3 (Šidák normalisation for varying `|W_i|`) is live again if WP2 is revived.** It
was previously called moot for the same wrong (layout) reason this document originally
gave. §2.1 shows the eligible-scale count genuinely varies per edge, and varies across
a wide range (50–100 percentage points between `w=1` and `w=12`) rather than collapsing
to a single scale — R-3 is required, not optional, in any revived WP2 spec.

**R-5 (support-conditioned per-scale transform) remains delivered** — it is Stage 1c,
unaffected by this correction.
