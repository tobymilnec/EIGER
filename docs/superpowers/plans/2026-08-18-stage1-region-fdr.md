# Stage 1 — Region FDR correction and finite-null uncertainty (v2.0.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the two confirmed defects in EIGER's region-level FDR (wrong-direction
monotone step, tied-cutoff denominator), add finite-null resolution and simultaneous
Garwood Monte Carlo bounds, raise the null replicate default, add an opt-in conservative
decision mode, extend and version the output schema, and ship v2.0.1 without changing the
discovery statistic, threshold rule, or segmentation.

**Architecture:** All numerics live in a new `core._fdr_from_areas_result` (per-stratum
tail-FDR + Garwood bounds, distinct-cutoff grouping) orchestrated by a new
`core.region_fdr_result` (two-pass common-Bonferroni-`m`, per-stratum overdispersion
diagnostic, degenerate-input handling). `region_fdr` becomes a one-line wrapper so its
signature is frozen. The identical fix is mirrored in `probability._tail_area_fdr`
(point estimate only — the extrapolated-tail fallback does not carry bounds) and in a new
`core.qvalue_result_from_track` used by both `eiger2 call` and `eiger2 call --legacy`.

**Tech Stack:** Python 3.10+, NumPy, pandas, SciPy (`scipy.stats.chi2`), pytest, ruff.

**Spec:** `EIGER_WP1-3_IMPLEMENTATION_SPEC.md` (repo root of
`Github_beta_Versions`), §0 (contract) and STAGE 1 (§1.0–§1.12, Appendix A cases,
Appendix B cross-reference).

## Global Constraints

- Work happens **only** in worktree `../EIGER-wt-fdr` on branch `fix/region-fdr`,
  branched from `b8e4a19` (tag `v2.0.0`). The live repo
  `/Users/tobymilne-clark/Documents/Code/Github_beta_Versions/EIGER` must never be
  checked out, stashed, or committed to.
- Never edit `Python/functions/**` or `old_code/**`. Never write into
  `analysis/`, `data/`, or any existing `results*/` directory — new benchmark output
  goes in a new `results_v201/` directory.
- Discovery/annotation separation: nothing added in this stage may read labels,
  outcomes, expression, clinical variables, or programme assignments — none exist on
  this code path, so this is a standing check, not new work.
- Mirror every adaptive operation in the null (spec §0.3.2) — any new statistic (Garwood
  bound, overdispersion ratio) must be computed identically for observed and null data;
  all quantities here are pooled-null-derived by construction, so this is satisfied by
  construction as long as no new bespoke observed-only shortcut is introduced.
- Bit-for-bit invariant set (spec §0.3.3) must reproduce v2.0.0 exactly:
  `region_id, chr, start, end, n_cpgs, cpg_index_start, cpg_index_end, statistic,
  statistic_mean, statistic_min, statistic_threshold, region_area`, and every
  `*_descriptive` column. `region_qvalue, region_evidence_score,
  evidence_score_extrapolated, significant, n_significant, n_passing_qvalue` change by
  design and must be named in the release note.
- Determinism: every derived seed must come from `(config.random_seed, stage_id, ...)`,
  never call order. This stage does not add new randomness (Garwood bounds and the
  overdispersion ratio are deterministic transforms of already-derived null tracks), so
  no new seed derivation is required — confirm this holds when reviewing the diff.
- No exact zero without a nonzero `region_qvalue_upper_95` and nonzero
  `region_qvalue_resolution`.
- No statistical claim without a falsifying simulation — Task 11 benchmarks are release
  gates, not optional polish.
- Naming discipline: q-value ≠ posterior; Monte Carlo interval ≠ biological truth;
  `region_evidence_score` stays descriptive language throughout new docs.
- Docs, CLI help, README, manifest and tests ship in the **same commit** as the code
  that needs them — do not defer documentation to a later task's commit.
- `region_fdr(observed_areas, null_area_sets, *, observed_edges, null_edges,
  observed_lengths=None, null_length_sets=None)` is frozen. Any new argument on any
  touched function is keyword-only with a default equal to current behaviour.
- Use exact float equality for area-based cutoff grouping — never `np.isclose` or
  rounding bins (spec explicitly forbids this).

---

## Verified baseline (read 2026-08-18, do not re-derive)

- Live repo HEAD = `b8e4a1916ec18b8024dc9bfd328a89ca68851ba6`, tag `v2.0.0`, clean tree.
- `core.py:995` (`_fdr_from_areas`) and `probability.py:217` (`_tail_area_fdr`) both do
  `raw = np.clip(expected/counts, 0, 1); monotone = np.minimum.accumulate(raw)` on
  areas sorted strongest→weakest — the confirmed forward-cummin defect, and the
  denominator is `np.arange(1, n+1)` (sequential rank, not tie-aware `R`).
  `_tail_area_fdr` in `probability.py` is bit-identical in structure.
- `Eiger2Config` is a frozen dataclass in `core.py`, fields start at line 61,
  `__post_init__` validation block at line 298–323, `null_replicates: int = 20` at
  line 145, `target_fdr: float = 0.05` at line 255, `length_stratified_fdr: bool =
  True` at line 245.
- `region_fdr` (core.py:1015) loops over `np.unique(observed_strata)`, has the pooled
  fallback `if pooled.size == 0: pooled = null_areas` — this exact fallback must be
  preserved.
- `call_dmrs` (core.py:1260) calls `region_fdr(...)` at line 1388, builds `null_areas`/
  `null_lengths` per replicate in a loop first (line ~1367–1383), then builds the row
  dict per region at line ~1426–1486 using `_REGION_COLUMNS` (defined at line
  1503–1533) to select column order — **new columns not added to `_REGION_COLUMNS` are
  silently dropped.**
- `qvalues_from_track` (core.py:1081) returns `(qvalues, diagnostics)`, calls
  `permutation_null_tracks` (default `replicates=20`, core.py:761) and `region_fdr`
  internally. Used by `cli.run_legacy` (cli.py:244) at the call `qvalues_from_track(...)`
  around cli.py:286.
- `cli.command_call` (cli.py:423) builds the outer manifest dict at line ~478–498 and
  computes `significant`/prints it at line ~502–515. `cli.run_legacy` builds its own
  `significant` column at line ~354–357.
- `__init__.py` exports `Eiger2Config, call_dmrs, centre_cpgs, characterise_region,
  deflate, edge_statistic, estimate_latent_basis, null_threshold, region_fdr, segment`
  from `.core` and `LocalFdrResult, calibration_error, estimate_pi0, local_fdr,
  reliability_table` from `.probability`, `__all__` alphabetised (capitalised names
  first, then lowercase).
- `pyproject.toml`: `dependencies = ["numpy>=1.23", "pandas>=1.5", "scipy>=1.9"]` —
  `scipy.stats.chi2` needs no new dependency. `version = "2.0.0"`; bump to `2.0.1` in
  Task 12.
- `tests/test_eiger2_core.py` imports `from eiger2.benchmark import evaluate, simulate`
  and defines a `locations(n_cpg, chrom="chr1")` helper and `FAST = {"null_replicates":
  6}` config-override dict — reuse both in new tests.
- No `docs/FDR.md` exists yet. `docs/` currently has `BENCHMARK.md`, `EIGER2_DESIGN.md`,
  `LEGACY_METHOD.md`, `PROBABILITY.md`.
- `benchmarking/benchmark_suite.py` (`command_all`) and
  `benchmarking/calibration_study.py` (calls `region_fdr` at its line 151, `local_fdr`
  at 145) both write `manifest.json` without the §0.5 required fields.

---

### Task 1: Worktree, baseline verification, and invariant fixture

**Files:**
- Create (outside repo tracking, one level up): `../EIGER-wt-fdr/` (new git worktree)
- Create: `tests/fixtures/stage1_invariant_fixture_v200.csv` (committed fixture)
- Create: `tests/fixtures/generate_stage1_invariant_fixture.py` (fixture generator,
  run once against v2.0.0 code, then committed alongside its output)

**Interfaces:**
- Produces: a committed `regions.csv` slice (invariant columns only) generated by
  **unmodified v2.0.0 code**, for Task 10's regression test to diff against.

- [ ] **Step 1: Create the worktree**

```bash
cd /Users/tobymilne-clark/Documents/Code/Github_beta_Versions/EIGER
git worktree add ../EIGER-wt-fdr -b fix/region-fdr b8e4a19
cd ../EIGER-wt-fdr
git rev-parse HEAD
git status --short
```

Record both outputs verbatim in the STOP 1 report (Task 13). Expect
`b8e4a1916ec18b8024dc9bfd328a89ca68851ba6` and an empty status.

- [ ] **Step 2: Baseline verification**

```bash
conda env create -f environment.yml   # only if the env is absent
conda activate eiger
pip install -e ".[legacy,dev]"
pytest tests -q
pytest old_code/tests -q
ruff check Python/eiger2
```

Record the exact pass/fail counts. Expected: baseline green except a documented
`numba`-only environmental note if `[legacy,dev]` was already satisfied; if genuinely
red, stop and report before touching anything further. **Do not** "fix"
`tests/test_probability.py::test_region_fdr_is_monotone_in_area` here — it is corrected
in Task 2/Task 5's test rewrite (spec §1.10 item 16), not before.

- [ ] **Step 3: Generate the invariant fixture from unmodified v2.0.0 code**

Write `tests/fixtures/generate_stage1_invariant_fixture.py`:

```python
"""Regenerate tests/fixtures/stage1_invariant_fixture_v200.csv.

Run ONLY against unmodified v2.0.0 code (git tag v2.0.0 / commit b8e4a19). Do not
rerun this after Stage 1 changes land -- the fixture is the pre-correction baseline
that tests/test_eiger2_core.py::test_stage1_preserves_the_invariant_column_set diffs
against.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Python"))

from eiger2 import Eiger2Config, call_dmrs
from eiger2.benchmark import locations as _unused  # noqa: F401  (import-shape check)
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_eiger2_core import locations  # reuse the shared synthetic-locations helper
from eiger2.benchmark import simulate

INVARIANT_COLUMNS = [
    "region_id", "chr", "start", "end", "n_cpg", "cpg_index_start", "cpg_index_end",
    "statistic", "statistic_mean", "statistic_min", "statistic_threshold",
    "region_area", "minority_size_descriptive", "minority_delta_descriptive",
    "n_samples_observed", "missing_rate", "mean_beta", "delta_beta_descriptive",
    "largest_gap_descriptive", "direction_descriptive",
    "minor_group_size_descriptive", "minor_group_fraction_descriptive",
    "minor_group_samples_descriptive", "pvalue_descriptive",
]


def main() -> None:
    matrix, locs, planted = simulate(
        n_samples=60, strata=[(0.15, 0.35)], replicates=4,
        spacing=300, dmr_length=12, missing_rate=0.02, seed=1701,
    )
    config = Eiger2Config(window=4, null_replicates=8, random_seed=1701)
    regions, _values, _manifest = call_dmrs(matrix, locs, config=config)
    out = Path(__file__).parent / "stage1_invariant_fixture_v200.csv"
    regions[INVARIANT_COLUMNS].to_csv(out, index=False)
    print(f"wrote {len(regions)} rows to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it against the untouched worktree and commit the fixture**

```bash
cd /Users/tobymilne-clark/Documents/Code/Github_beta_Versions/EIGER-wt-fdr
python tests/fixtures/generate_stage1_invariant_fixture.py
git add tests/fixtures/generate_stage1_invariant_fixture.py \
        tests/fixtures/stage1_invariant_fixture_v200.csv
git commit -m "test: commit v2.0.0 invariant-column fixture before Stage 1 changes"
```

Confirm the row count printed is > 0. If `call_dmrs` finds zero regions, lower
`missing_rate` or raise `replicates` in the `simulate(...)` call in Step 3 and rerun —
do not commit an empty fixture.

---

### Task 2: Reference tail-FDR estimator `_fdr_from_areas_result`

**Files:**
- Modify: `Python/eiger2/core.py` (add `_fdr_from_areas_result`, replace body of
  `_fdr_from_areas` with a thin wrapper)
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- Produces: `_fdr_from_areas_result(observed_areas, pooled_null, scale, *,
  confidence_level, m_bonferroni) -> dict` with keys `qvalue, tail_fdr, qvalue_lower,
  qvalue_upper, qvalue_resolution, null_exceedances, qvalue_cutoff_rank, _n_cutoffs`.
  Consumed by Task 3's `region_fdr_result`.

- [ ] **Step 1: Write the failing hand-calculation and property tests**

Add to `tests/test_eiger2_core.py`:

```python
from eiger2.core import _fdr_from_areas_result  # noqa: E402


def test_fdr_from_areas_result_four_distinct_areas():
    observed = np.array([10.0, 8.0, 6.0, 4.0])
    pooled = np.array([9.0, 7.0, 5.0, 3.0])
    result = _fdr_from_areas_result(
        observed, pooled, scale=1.0, confidence_level=0.95, m_bonferroni=4,
    )
    assert result["tail_fdr"] == pytest.approx([0.0, 0.5, 2 / 3, 0.75])
    assert result["qvalue"] == pytest.approx([0.0, 0.5, 2 / 3, 0.75])
    assert result["null_exceedances"] == pytest.approx([0, 1, 2, 3])
    assert result["qvalue_cutoff_rank"] == pytest.approx([1, 2, 3, 4])


def test_fdr_from_areas_result_tie_group():
    observed = np.array([10.0, 8.0, 8.0, 4.0])
    pooled = np.array([9.0, 7.5, 5.0, 3.0])
    result = _fdr_from_areas_result(
        observed, pooled, scale=1.0, confidence_level=0.95, m_bonferroni=3,
    )
    assert result["tail_fdr"] == pytest.approx([0.0, 1 / 3, 1 / 3, 0.75])
    assert result["qvalue"] == pytest.approx([0.0, 1 / 3, 1 / 3, 0.75])
    assert result["qvalue_cutoff_rank"] == pytest.approx([1, 3, 3, 4])
    assert result["_n_cutoffs"] == 3


def test_fdr_from_areas_result_qvalue_is_nondecreasing_as_area_weakens():
    rng = np.random.default_rng(11)
    observed = rng.exponential(2.0, size=60)
    pooled = rng.exponential(2.0, size=400)
    result = _fdr_from_areas_result(
        observed, pooled, scale=0.3, confidence_level=0.95, m_bonferroni=60,
    )
    order = np.argsort(-observed, kind="mergesort")
    q = result["qvalue"][order]
    assert np.all(np.diff(q) >= -1e-12)


def test_fdr_from_areas_result_bounds_bracket_the_point_estimate():
    rng = np.random.default_rng(12)
    observed = rng.exponential(2.0, size=50)
    pooled = rng.exponential(2.0, size=300)
    result = _fdr_from_areas_result(
        observed, pooled, scale=0.4, confidence_level=0.95, m_bonferroni=50,
    )
    assert np.all(result["qvalue_lower"] <= result["qvalue"] + 1e-12)
    assert np.all(result["qvalue"] <= result["qvalue_upper"] + 1e-12)


def test_fdr_from_areas_result_zero_exceedances_has_nonzero_upper_bound():
    observed = np.array([50.0, 3.0])
    pooled = np.array([1.0, 0.5])
    result = _fdr_from_areas_result(
        observed, pooled, scale=1.0, confidence_level=0.95, m_bonferroni=2,
    )
    assert result["qvalue"][0] == 0.0
    assert result["qvalue_lower"][0] == 0.0
    assert result["qvalue_upper"][0] > 0.0


def test_fdr_from_areas_result_more_null_exposure_narrows_the_interval():
    # spec §1.10 item 8: increasing null exposure at a fixed rate (same scale,
    # more pooled null observations at the same relative density) must narrow
    # [lower, upper] because larger K gives a tighter Garwood interval.
    observed = np.array([9.0, 7.0, 5.0, 3.0])
    rng = np.random.default_rng(15)
    sparse_null = rng.exponential(2.0, size=40)
    dense_null = np.concatenate([sparse_null, rng.exponential(2.0, size=400)])
    sparse = _fdr_from_areas_result(
        observed, sparse_null, scale=0.5, confidence_level=0.95, m_bonferroni=4,
    )
    dense = _fdr_from_areas_result(
        observed, dense_null, scale=0.5, confidence_level=0.95, m_bonferroni=4,
    )
    sparse_width = sparse["qvalue_upper"] - sparse["qvalue_lower"]
    dense_width = dense["qvalue_upper"] - dense["qvalue_lower"]
    assert np.all(dense_width <= sparse_width + 1e-9)
    assert np.any(dense_width < sparse_width - 1e-9)


def test_fdr_from_areas_result_resolution_matches_scale_over_rank():
    rng = np.random.default_rng(13)
    observed = rng.exponential(1.0, size=30)
    pooled = rng.exponential(1.0, size=200)
    scale = 0.5
    result = _fdr_from_areas_result(
        observed, pooled, scale=scale, confidence_level=0.95, m_bonferroni=30,
    )
    expected = np.minimum(1.0, scale / result["qvalue_cutoff_rank"])
    assert result["qvalue_resolution"] == pytest.approx(expected)


def test_fdr_from_areas_result_vectorised_argmin_matches_naive_reverse_loop():
    rng = np.random.default_rng(14)
    for _ in range(200):
        n = rng.integers(3, 25)
        observed = rng.choice(np.round(rng.exponential(2.0, size=n), 2), size=n)
        pooled = rng.exponential(2.0, size=rng.integers(5, 60))
        vectorised = _fdr_from_areas_result(
            observed, pooled, scale=0.6, confidence_level=0.95,
            m_bonferroni=int(np.unique(observed).size),
        )

        # naive reverse loop over the same grouped structure
        order = np.argsort(-observed, kind="mergesort")
        ranked = observed[order]
        pooled_sorted = np.sort(pooled)
        m = ranked.size
        new_group = np.empty(m, dtype=bool)
        new_group[0] = True
        new_group[1:] = ranked[1:] != ranked[:-1]
        group_of_row = np.cumsum(new_group) - 1
        G = int(group_of_row[-1]) + 1
        cutoff = ranked[new_group]
        last_row_of_group = np.flatnonzero(np.append(new_group[1:], True))
        R = (last_row_of_group + 1).astype(float)
        K = (pooled_sorted.size - np.searchsorted(pooled_sorted, cutoff)).astype(float)
        raw = np.minimum(1.0, 0.6 * K / R)
        q_group = np.empty(G)
        arg_group = np.empty(G, dtype=int)
        running_min = raw[G - 1]
        running_arg = G - 1
        for g in range(G - 1, -1, -1):
            if raw[g] <= running_min:
                running_min = raw[g]
                running_arg = g
            q_group[g] = running_min
            arg_group[g] = running_arg
        assert arg_group[-1] != -1  # spec's own sanity check, restated

        expanded_q = q_group[group_of_row]
        out = np.empty(m)
        out[order] = expanded_q
        assert vectorised["qvalue"] == pytest.approx(out)


def test_fdr_from_areas_result_exact_tie_areas_share_every_field():
    observed = np.array([10.0, 8.0, 8.0, 4.0])
    pooled = np.array([9.0, 7.5, 5.0, 3.0])
    result = _fdr_from_areas_result(
        observed, pooled, scale=1.0, confidence_level=0.95, m_bonferroni=3,
    )
    for key in ("qvalue", "tail_fdr", "qvalue_lower", "qvalue_upper",
                "qvalue_resolution", "null_exceedances", "qvalue_cutoff_rank"):
        assert result[key][1] == result[key][2], key
```

- [ ] **Step 2: Run and confirm every new test fails**

```bash
pytest tests/test_eiger2_core.py -k fdr_from_areas_result -v
```

Expected: `ImportError: cannot import name '_fdr_from_areas_result'`.

- [ ] **Step 3: Implement `_fdr_from_areas_result`, verbatim per spec §1.3**

Add `from scipy.stats import chi2` to the top-level imports in `core.py` (alongside the
existing `numpy`/`pandas` imports). Insert this function immediately above the current
`_fdr_from_areas` (core.py:984):

```python
def _fdr_from_areas_result(
    observed_areas: np.ndarray,
    pooled_null: np.ndarray,
    scale: float,
    *,
    confidence_level: float,
    m_bonferroni: int,
) -> dict:
    """Tail-area FDR, Garwood bounds and resolution for one length stratum.

    ``raw``/``qvalue`` are the corrected reverse-cumulative-minimum tail estimate:
    the previous ``np.minimum.accumulate`` scanned strongest-to-weakest FORWARD,
    which propagates the strongest region's near-universal zero exceedance count to
    every weaker region. ``q_i = min_{j>=i} FDR_j`` requires the reverse scan.
    Distinct-cutoff grouping (rather than sequential rank) makes tied areas share a
    denominator, so two regions with identical area are not scored as if one were
    stronger than the other.
    """

    m = observed_areas.size
    pooled = np.sort(pooled_null[np.isfinite(pooled_null)])

    # --- distinct cutoffs, strongest first. EXACT float equality; no rounding bins.
    order = np.argsort(-observed_areas, kind="mergesort")
    ranked = observed_areas[order]
    new_group = np.empty(m, dtype=bool)
    new_group[0] = True
    new_group[1:] = ranked[1:] != ranked[:-1]
    group_of_row = np.cumsum(new_group) - 1            # 0..G-1, per ranked row
    G = int(group_of_row[-1]) + 1
    cutoff = ranked[new_group]                          # strictly decreasing

    # R_g: rows with area >= cutoff[g] == cumulative count through end of group g
    last_row_of_group = np.flatnonzero(np.append(new_group[1:], True))
    R = (last_row_of_group + 1).astype(float)
    K = (pooled.size - np.searchsorted(pooled, cutoff, side="left")).astype(float)

    raw   = np.minimum(1.0, scale * K / R)
    delta = np.minimum(1.0, scale / R)

    alpha  = 1.0 - confidence_level
    astar  = alpha / max(int(m_bonferroni), 1)
    L = np.where(K > 0, 0.5 * chi2.ppf(astar / 2.0, 2.0 * K), 0.0)
    U = 0.5 * chi2.ppf(1.0 - astar / 2.0, 2.0 * (K + 1.0))
    lo = np.minimum(1.0, scale * L / R)
    hi = np.minimum(1.0, scale * U / R)

    # --- reverse cumulative minimum, with argmin, over GROUPS
    rev      = raw[::-1]
    run_min  = np.minimum.accumulate(rev)
    # index of the achieving cutoff; ties resolve to the STRONGER (later in rev order)
    cand     = np.where(rev == run_min, np.arange(G), -1)
    arg_rev  = np.maximum.accumulate(cand)
    q_group     = run_min[::-1]
    qarg_group  = (G - 1) - arg_rev[::-1]
    q_lo_group  = np.minimum.accumulate(lo[::-1])[::-1]
    q_hi_group  = np.minimum.accumulate(hi[::-1])[::-1]

    # --- expand groups back to ranked rows, then restore original row order
    def _expand(v):
        out = np.empty(m, dtype=v.dtype)
        out[order] = v[group_of_row]
        return out

    return {
        "qvalue":            _expand(q_group),
        "tail_fdr":          _expand(raw),          # region's OWN cutoff, not q-achieving
        "qvalue_lower":      _expand(q_lo_group),
        "qvalue_upper":      _expand(q_hi_group),
        "qvalue_resolution": _expand(delta[qarg_group]),
        "null_exceedances":  _expand(K[qarg_group]),
        "qvalue_cutoff_rank":_expand(R[qarg_group].astype(np.int64)),
        "cutoffs":           cutoff,
        "_n_cutoffs":        G,
    }
```

Note: added a `"cutoffs"` key (not in the spec's literal dict) beyond the spec's listed
fields — Task 4 needs the per-stratum distinct-cutoff array for the overdispersion
diagnostic and recomputing it would duplicate the grouping logic above. It is
internal (not part of the public `RegionFdrResult`), same convention as `_n_cutoffs`.

Leave `_fdr_from_areas` as a thin wrapper (used nowhere yet in this task, wired in
Task 3):

```python
def _fdr_from_areas(
    observed_areas: np.ndarray, pooled_null: np.ndarray, scale: float
) -> np.ndarray:
    m = observed_areas.size
    return _fdr_from_areas_result(
        observed_areas, pooled_null, scale,
        confidence_level=0.95, m_bonferroni=int(np.unique(observed_areas).size) or 1,
    )["qvalue"]
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_eiger2_core.py -k fdr_from_areas_result -v
```

Expected: all 8 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add Python/eiger2/core.py tests/test_eiger2_core.py
git commit -m "feat: add _fdr_from_areas_result with corrected reverse-cummin and Garwood bounds"
```

---

### Task 3: `RegionFdrResult`, `region_fdr_result`, and the frozen `region_fdr` wrapper

**Files:**
- Modify: `Python/eiger2/core.py`
- Modify: `Python/eiger2/__init__.py`
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- Consumes: `_fdr_from_areas_result` (Task 2).
- Produces: `RegionFdrResult` dataclass and `region_fdr_result(observed_areas,
  null_area_sets, *, observed_edges, null_edges, observed_lengths=None,
  null_length_sets=None, confidence_level=0.95) -> RegionFdrResult`, consumed by
  Task 4 (overdispersion), Task 5 (`qvalue_result_from_track`), Task 8 (`call_dmrs`
  wiring).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_eiger2_core.py`:

```python
from eiger2.core import RegionFdrResult, region_fdr_result  # noqa: E402


def _null_sets(rng, n_replicates, n_per_replicate, scale_hint=2.0):
    return [rng.exponential(scale_hint, size=n_per_replicate) for _ in range(n_replicates)]


def test_region_fdr_result_matches_region_fdr_exactly():
    rng = np.random.default_rng(21)
    observed = rng.exponential(2.0, size=40)
    nulls = _null_sets(rng, 6, 90)
    result = region_fdr_result(
        observed, nulls, observed_edges=500, null_edges=3000,
    )
    legacy = region_fdr(observed, nulls, observed_edges=500, null_edges=3000)
    assert np.array_equal(result.qvalue, legacy)


def test_region_fdr_result_empty_inputs():
    result = region_fdr_result(
        np.empty(0), [], observed_edges=0, null_edges=0,
    )
    for field in (result.qvalue, result.tail_fdr, result.qvalue_lower,
                  result.qvalue_upper, result.qvalue_resolution):
        assert field.shape == (0,)
    assert result.diagnostics["pooled_fallback_strata"] == []


def test_region_fdr_result_zero_replicates_is_all_nan():
    result = region_fdr_result(
        np.array([1.0, 2.0]), [], observed_edges=10, null_edges=0,
    )
    assert np.all(np.isnan(result.qvalue))
    assert np.all(result.null_exceedances == -1)
    assert np.all(result.qvalue_cutoff_rank == -1)
    assert "status" in result.diagnostics


def test_region_fdr_result_pooled_fallback_stratum_uses_all_six_fields():
    rng = np.random.default_rng(22)
    # one stratum ("4-5", n_cpg<=5) has zero null regions of that length
    observed_lengths = np.array([4, 4, 30, 30])
    observed_areas = np.array([5.0, 3.0, 9.0, 6.0])
    null_area_sets = [rng.exponential(2.0, size=20) for _ in range(5)]
    null_length_sets = [np.full(20, 30, dtype=int) for _ in range(5)]
    result = region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=200, null_edges=100,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    )
    assert "4-5" in result.diagnostics["pooled_fallback_strata"]
    assert not np.any(np.isnan(result.qvalue[:2]))
    assert not np.any(result.null_exceedances[:2] == -1)


def test_region_fdr_result_row_order_survives_permutation():
    rng = np.random.default_rng(23)
    observed = rng.exponential(2.0, size=25)
    lengths = rng.integers(4, 60, size=25)
    nulls = _null_sets(rng, 5, 80)
    null_lengths = [rng.integers(4, 60, size=80) for _ in range(5)]
    baseline = region_fdr_result(
        observed, nulls, observed_edges=300, null_edges=400,
        observed_lengths=lengths, null_length_sets=null_lengths,
    )
    perm = rng.permutation(25)
    inv = np.argsort(perm)
    permuted = region_fdr_result(
        observed[perm], nulls, observed_edges=300, null_edges=400,
        observed_lengths=lengths[perm], null_length_sets=null_lengths,
    )
    for a, b in (
        (baseline.qvalue, permuted.qvalue[inv]),
        (baseline.qvalue_lower, permuted.qvalue_lower[inv]),
        (baseline.qvalue_upper, permuted.qvalue_upper[inv]),
        (baseline.qvalue_resolution, permuted.qvalue_resolution[inv]),
        (baseline.null_exceedances, permuted.null_exceedances[inv]),
        (baseline.qvalue_cutoff_rank, permuted.qvalue_cutoff_rank[inv]),
    ):
        assert np.array_equal(a, b)


def test_region_fdr_result_common_bonferroni_m_is_max_across_strata():
    rng = np.random.default_rng(24)
    observed_lengths = np.array([4] * 3 + [30] * 10)
    observed_areas = rng.exponential(2.0, size=13)
    null_area_sets = [rng.exponential(2.0, size=40) for _ in range(6)]
    null_length_sets = [
        np.concatenate([np.full(15, 4), np.full(25, 30)]) for _ in range(6)
    ]
    result = region_fdr_result(
        observed_areas, null_area_sets, observed_edges=200, null_edges=240,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    )
    by_stratum = result.diagnostics["fdr_interval_cutoffs_by_stratum"]
    assert result.diagnostics["fdr_interval_bonferroni_m"] == max(by_stratum.values())
    assert result.diagnostics["fdr_interval_family"] == "per_stratum"


def test_region_fdr_result_length_stratified_maps_back_to_original_rows():
    rng = np.random.default_rng(25)
    observed_lengths = np.array([4, 30, 4, 30])
    observed_areas = np.array([1.0, 100.0, 2.0, 50.0])
    null_area_sets = [rng.exponential(2.0, size=60) for _ in range(4)]
    null_length_sets = [
        np.concatenate([np.full(30, 4), np.full(30, 30)]) for _ in range(4)
    ]
    result = region_fdr_result(
        observed_areas, null_area_sets, observed_edges=100, null_edges=240,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    )
    # short-stratum regions (indices 0, 2) must not be compared against the
    # long-stratum null: their q-values should differ from a pooled-only run
    pooled_result = region_fdr_result(
        observed_areas, null_area_sets, observed_edges=100, null_edges=240,
    )
    assert not np.array_equal(result.qvalue, pooled_result.qvalue)


def test_region_fdr_result_weak_candidates_do_not_inherit_a_leading_zero():
    # spec §1.10 item 17: this is the defect's actual symptom end-to-end -- the
    # old forward-cummin propagated the strongest region's zero to every weaker
    # one. With plenty of weaker, genuinely null-comparable candidates, q must
    # not be all zero or constant.
    rng = np.random.default_rng(26)
    strong = np.array([40.0])
    weak = rng.exponential(2.0, size=80)  # comparable in scale to the null
    observed = np.concatenate([strong, weak])
    nulls = [rng.exponential(2.0, size=90) for _ in range(10)]
    result = region_fdr_result(
        observed, nulls, observed_edges=1000, null_edges=900,
    )
    assert (result.qvalue == 0).sum() < len(result.qvalue)
    assert np.unique(result.qvalue).size > 1
```

- [ ] **Step 2: Run and confirm failure**

```bash
pytest tests/test_eiger2_core.py -k region_fdr_result -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `RegionFdrResult`, `region_fdr_result`, and the `region_fdr`
  wrapper**

Add near the top of `core.py`, above `_fdr_from_areas_result`:

```python
def _n_distinct(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    ordered = np.sort(values)
    return int(1 + np.sum(ordered[1:] != ordered[:-1]))


@dataclass(frozen=True)
class RegionFdrResult:
    qvalue: np.ndarray
    tail_fdr: np.ndarray
    qvalue_lower: np.ndarray
    qvalue_upper: np.ndarray
    qvalue_resolution: np.ndarray
    null_exceedances: np.ndarray
    qvalue_cutoff_rank: np.ndarray
    confidence_level: float
    diagnostics: dict
```

Replace the body of `region_fdr` (core.py:1015–1078) with:

```python
def region_fdr_result(
    observed_areas: np.ndarray,
    null_area_sets: Sequence[np.ndarray],
    *,
    observed_edges: int,
    null_edges: int,
    observed_lengths: np.ndarray | None = None,
    null_length_sets: Sequence[np.ndarray] | None = None,
    confidence_level: float = 0.95,
) -> RegionFdrResult:
    """Region-level empirical FDR: point estimate, resolution, and simultaneous
    Garwood Monte Carlo bounds, computed within length strata with a common
    Bonferroni ``m`` (spec §1.4) so ``qvalue_upper`` is comparable across strata.

    See ``docs/FDR.md`` for the derivation and the three distinct quantities this
    keeps apart: the point estimate, the empirical resolution, and the Monte Carlo
    interval.
    """

    observed_areas = np.asarray(observed_areas, dtype=float)
    m = observed_areas.size
    empty_f = np.empty(0, dtype=float)
    empty_i = np.empty(0, dtype=np.int64)
    if m == 0:
        return RegionFdrResult(
            qvalue=empty_f, tail_fdr=empty_f, qvalue_lower=empty_f,
            qvalue_upper=empty_f, qvalue_resolution=empty_f,
            null_exceedances=empty_i, qvalue_cutoff_rank=empty_i,
            confidence_level=confidence_level,
            diagnostics={"pooled_fallback_strata": []},
        )

    replicates = len(null_area_sets)
    if replicates == 0 or null_edges <= 0:
        nan_f = np.full(m, np.nan, dtype=float)
        neg_i = np.full(m, -1, dtype=np.int64)
        status = "no null replicates available" if replicates == 0 else "null_edges <= 0"
        return RegionFdrResult(
            qvalue=nan_f, tail_fdr=nan_f, qvalue_lower=nan_f,
            qvalue_upper=nan_f, qvalue_resolution=nan_f,
            null_exceedances=neg_i, qvalue_cutoff_rank=neg_i,
            confidence_level=confidence_level,
            diagnostics={"status": status, "pooled_fallback_strata": []},
        )

    # ``null_edges`` is the TOTAL number of null edges summed over replicates, and
    # ``null_area_sets`` are likewise pooled over replicates, so the two already
    # correspond -- see the identical comment preserved on the old region_fdr.
    scale = observed_edges / null_edges
    null_area_sets = [np.asarray(a, dtype=float) for a in null_area_sets]

    stratified = observed_lengths is not None and null_length_sets is not None
    if stratified:
        observed_lengths = np.asarray(observed_lengths, dtype=int)
        null_length_sets = [np.asarray(a, dtype=int) for a in null_length_sets]
        observed_strata = np.array(
            [length_stratum(int(n)) for n in observed_lengths], dtype=object
        )
        null_strata_by_replicate = [
            np.array([length_stratum(int(n)) for n in lengths], dtype=object)
            for lengths in null_length_sets
        ]
    else:
        observed_strata = np.full(m, "_pooled", dtype=object)
        null_strata_by_replicate = [
            np.full(areas.size, "_pooled", dtype=object) for areas in null_area_sets
        ]

    strata = np.unique(observed_strata)

    # --- Pass A: distinct-cutoff count per stratum, then the common m_bonferroni
    cutoffs_by_stratum = {
        str(s): _n_distinct(observed_areas[observed_strata == s]) for s in strata
    }
    m_bonferroni = max(cutoffs_by_stratum.values())

    # --- Pass B: run the estimator per stratum with that common m
    qvalue = np.empty(m, dtype=float)
    tail_fdr = np.empty(m, dtype=float)
    qvalue_lower = np.empty(m, dtype=float)
    qvalue_upper = np.empty(m, dtype=float)
    qvalue_resolution = np.empty(m, dtype=float)
    null_exceedances = np.empty(m, dtype=np.int64)
    qvalue_cutoff_rank = np.empty(m, dtype=np.int64)
    pooled_fallback_strata: list[str] = []
    overdispersion_by_stratum: dict[str, dict] = {}

    for stratum in strata:
        rows = np.flatnonzero(observed_strata == stratum)
        per_replicate = [
            null_area_sets[b][null_strata_by_replicate[b] == stratum]
            for b in range(replicates)
        ]
        pooled = np.concatenate(per_replicate) if per_replicate else np.empty(0)
        if pooled.size == 0:
            pooled_fallback_strata.append(str(stratum))
            pooled = np.concatenate(null_area_sets)
            per_replicate = null_area_sets
        piece = _fdr_from_areas_result(
            observed_areas[rows], pooled, scale,
            confidence_level=confidence_level, m_bonferroni=m_bonferroni,
        )
        qvalue[rows] = piece["qvalue"]
        tail_fdr[rows] = piece["tail_fdr"]
        qvalue_lower[rows] = piece["qvalue_lower"]
        qvalue_upper[rows] = piece["qvalue_upper"]
        qvalue_resolution[rows] = piece["qvalue_resolution"]
        null_exceedances[rows] = piece["null_exceedances"].astype(np.int64)
        qvalue_cutoff_rank[rows] = piece["qvalue_cutoff_rank"]
        overdispersion_by_stratum[str(stratum)] = _overdispersion_diagnostic(
            per_replicate, piece["cutoffs"]
        )

    diagnostics = {
        "fdr_interval_bonferroni_m": m_bonferroni,
        "fdr_interval_cutoffs_by_stratum": cutoffs_by_stratum,
        "fdr_interval_family": "per_stratum",
        "pooled_fallback_strata": pooled_fallback_strata,
        "null_count_dispersion_by_stratum": overdispersion_by_stratum,
        "scale": scale,
    }
    return RegionFdrResult(
        qvalue=qvalue, tail_fdr=tail_fdr, qvalue_lower=qvalue_lower,
        qvalue_upper=qvalue_upper, qvalue_resolution=qvalue_resolution,
        null_exceedances=null_exceedances, qvalue_cutoff_rank=qvalue_cutoff_rank,
        confidence_level=confidence_level, diagnostics=diagnostics,
    )


def region_fdr(
    observed_areas: np.ndarray,
    null_area_sets: Sequence[np.ndarray],
    *,
    observed_edges: int,
    null_edges: int,
    observed_lengths: np.ndarray | None = None,
    null_length_sets: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Monotone region-level empirical FDR. Signature frozen; see region_fdr_result
    for the resolution and Garwood-bound fields."""

    return region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=observed_edges, null_edges=null_edges,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    ).qvalue
```

`_overdispersion_diagnostic` is a forward reference to Task 4 — stub it now so Task 3's
tests pass standalone, replaced (not merely extended) in Task 4:

```python
def _overdispersion_diagnostic(
    per_replicate_null_areas: list[np.ndarray], cutoffs: np.ndarray
) -> tuple[str, float | None]:
    return "not_yet_implemented", None
```

Delete the old docstring-only body of `region_fdr` that this replaces; keep its
existing top-of-function docstring text (the length-stratum rationale paragraph) by
folding it into `region_fdr_result`'s docstring instead, since that is now where the
stratification happens.

Update `Python/eiger2/__init__.py`:

```python
from .core import (
    Eiger2Config,
    RegionFdrResult,
    call_dmrs,
    centre_cpgs,
    characterise_region,
    deflate,
    edge_statistic,
    estimate_latent_basis,
    null_threshold,
    region_fdr,
    region_fdr_result,
    segment,
)
```

and in `__all__`, insert `"RegionFdrResult"` immediately after `"LocalFdrResult"` (both
capitalised names, alphabetical among themselves) and `"region_fdr_result"`
immediately after `"region_fdr"`:

```python
__all__ = [
    "__version__",
    "# probability layer",
    "Eiger2Config",
    "LocalFdrResult",
    "RegionFdrResult",
    "calibration_error",
    "call_dmrs",
    "centre_cpgs",
    "characterise_region",
    "deflate",
    "edge_statistic",
    "estimate_latent_basis",
    "estimate_pi0",
    "local_fdr",
    "null_threshold",
    "region_fdr",
    "region_fdr_result",
    "reliability_table",
    "segment",
]
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_eiger2_core.py -k "region_fdr_result or fdr_from_areas_result" -v
pytest tests/test_eiger2_core.py -v   # full file, nothing else regressed
```

- [ ] **Step 5: Commit**

```bash
git add Python/eiger2/core.py Python/eiger2/__init__.py tests/test_eiger2_core.py
git commit -m "feat: add RegionFdrResult and region_fdr_result with common-m Bonferroni pass"
```

---

### Task 4: Overdispersion diagnostic

**Files:**
- Modify: `Python/eiger2/core.py` (real `_overdispersion_diagnostic`, replacing
  Task 3's stub)
- Modify: `Python/eiger2/cli.py` (`command_call` stderr warning)
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- Consumes: `RegionFdrResult.diagnostics["null_count_dispersion_by_stratum"]`
  (Task 3's stub key, now populated).
- Produces: manifest field `null_count_overdispersion_warning` appended to
  `manifest["warnings"]` (list — created in Task 9) when any stratum's
  `median_ratio > 1.5`; stderr print in `cli.command_call`.

- [ ] **Step 1: Write the failing tests**

```python
def test_overdispersion_diagnostic_flags_a_deliberately_overdispersed_null():
    rng = np.random.default_rng(31)
    # 8 replicates, exceedance counts drawn from a negative-binomial-like process
    # (Poisson mixed over a Gamma-distributed rate) so variance >> mean at every cutoff
    cutoffs = np.array([5.0, 4.0, 3.0])
    per_replicate = []
    for _ in range(8):
        rate = rng.gamma(shape=0.5, scale=6.0)  # overdispersed rate draw
        counts = rng.poisson(rate, size=3)
        # build a null-area array whose >= cutoff exceedance counts match `counts`
        areas = np.concatenate([
            np.full(counts[i], cutoffs[i] + 0.1) for i in range(3)
        ] + [np.zeros(5)])
        per_replicate.append(areas)
    status, ratio = _overdispersion_diagnostic(per_replicate, cutoffs)
    assert status == "computed"
    assert ratio is not None


def test_overdispersion_diagnostic_unavailable_below_five_replicates():
    per_replicate = [np.array([1.0, 2.0]) for _ in range(3)]
    status, ratio = _overdispersion_diagnostic(per_replicate, np.array([1.0]))
    assert status == "unavailable"
    assert ratio is None


def test_region_fdr_result_overdispersion_warning_end_to_end():
    rng = np.random.default_rng(32)
    observed = rng.exponential(2.0, size=20)
    # 8 replicates with wildly varying null density -> overdispersed exceedance counts
    nulls = []
    for i in range(8):
        density = 300 if i % 2 == 0 else 10
        nulls.append(rng.exponential(2.0, size=density))
    result = region_fdr_result(observed, nulls, observed_edges=500, null_edges=sum(len(n) for n in nulls))
    by_stratum = result.diagnostics["null_count_dispersion_by_stratum"]["_pooled"]
    assert by_stratum[0] in {"computed", "unavailable"}


def test_region_fdr_result_poisson_null_does_not_trigger_overdispersion():
    rng = np.random.default_rng(33)
    observed = rng.exponential(2.0, size=20)
    nulls = [rng.exponential(2.0, size=150) for _ in range(8)]  # same density each rep
    result = region_fdr_result(observed, nulls, observed_edges=500, null_edges=1200)
    status, ratio = result.diagnostics["null_count_dispersion_by_stratum"]["_pooled"]
    if status == "computed":
        assert ratio < 1.5
```

- [ ] **Step 2: Run and confirm failure**

```bash
pytest tests/test_eiger2_core.py -k overdispersion -v
```

Expected: failures against the stub (`"not_yet_implemented"` status, or a `TypeError`
from unpacking a dict instead of a tuple).

- [ ] **Step 3: Implement `_overdispersion_diagnostic` for real**

Replace the Task 3 stub in `core.py`:

```python
def _overdispersion_diagnostic(
    per_replicate_null_areas: list[np.ndarray], cutoffs: np.ndarray
) -> tuple[str, float | None]:
    """Poisson-vs-observed dispersion check on pooled null exceedance counts.

    ``region_qvalue_upper_95`` assumes a Poisson count model (Garwood). If the
    replicate-to-replicate exceedance counts are overdispersed relative to Poisson,
    the Garwood interval understates the true uncertainty -- it becomes a LOWER
    bound on the real interval, not a correct one. This is reported, never silently
    corrected (spec §1.5: no uncalibrated dispersion correction in this release).
    """

    b = len(per_replicate_null_areas)
    if b < 5 or cutoffs.size == 0:
        return "unavailable", None

    counts = np.empty((b, cutoffs.size), dtype=float)
    for row, areas in enumerate(per_replicate_null_areas):
        sorted_areas = np.sort(areas)
        counts[row] = areas.size - np.searchsorted(sorted_areas, cutoffs, side="left")

    usable = np.mean(counts, axis=0) > 0
    if not np.any(usable):
        return "unavailable", None

    means = np.mean(counts[:, usable], axis=0)
    variances = np.var(counts[:, usable], axis=0, ddof=1)
    ratios = variances / means
    return "computed", float(np.median(ratios))
```

Wire the manifest warning inside `region_fdr_result`, right after building
`overdispersion_by_stratum` (end of the per-stratum loop, before constructing
`diagnostics`):

```python
    dispersion_ratios = [
        ratio for status, ratio in overdispersion_by_stratum.values()
        if status == "computed"
    ]
    overdispersion_warning = (
        max(dispersion_ratios) if dispersion_ratios and max(dispersion_ratios) > 1.5
        else None
    )
```

Task 3's `diagnostics` dict literal already has a `"null_count_dispersion_by_stratum":
overdispersion_by_stratum,` line (populated by the Task 3 stub) — leave that line as
is; it now carries real `(status, ratio)` tuples instead of stub dicts. Add one new
line beneath it:

```python
        "null_count_overdispersion_warning": overdispersion_warning,
```

In `cli.py`, inside `command_call` (cli.py:423), right before the existing
`warning = caller_manifest.get("null", {}).get("warning")` block near line 517, add:

```python
    overdispersion_ratio = (caller_manifest.get("fdr") or {}).get(
        "null_count_overdispersion_warning"
    )
    if overdispersion_ratio is not None:
        print(
            f"[eiger2] WARNING: pooled null region counts are overdispersed\n"
            f"         (variance/mean = {overdispersion_ratio:.2f}, Poisson assumes "
            f"1.0).\n"
            f"         region_qvalue_upper_95 is a LOWER BOUND on the true "
            f"uncertainty.",
            file=sys.stderr,
        )
```

This references `caller_manifest["fdr"]`, which does not exist yet — it is created in
Task 8 when `call_dmrs`'s manifest is extended with FDR diagnostics. Leave this code
as written; Task 8 populates the key it reads.

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_eiger2_core.py -k overdispersion -v
```

- [ ] **Step 5: Commit**

```bash
git add Python/eiger2/core.py Python/eiger2/cli.py tests/test_eiger2_core.py
git commit -m "feat: add overdispersion diagnostic for pooled null region counts"
```

---

### Task 5: `probability._tail_area_fdr` correction and `qvalue_result_from_track`

**Files:**
- Modify: `Python/eiger2/probability.py` (correct `_tail_area_fdr`)
- Modify: `Python/eiger2/core.py` (add `qvalue_result_from_track`, reimplement
  `qvalues_from_track` as a wrapper)
- Modify: `Python/eiger2/cli.py` (`run_legacy` switches to the rich function)
- Test: `tests/test_probability.py`, `tests/test_eiger2_core.py`

**Interfaces:**
- Consumes: `region_fdr_result` (Task 3).
- Produces: `core.qvalue_result_from_track(statistic, regions, threshold, *,
  eligible=None, chromosome=None, min_cpgs=5, max_gap_edges=0, replicates=50, seed=1,
  length_stratified=True) -> (RegionFdrResult, dict)`. Consumed by Task 7 (vacuity
  guard call site) and Task 8 (`run_legacy` new columns).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_probability.py`:

```python
from eiger2.probability import _tail_area_fdr  # noqa: E402


def test_tail_area_fdr_four_distinct_areas():
    observed = np.array([10.0, 8.0, 6.0, 4.0])
    pooled = np.array([9.0, 7.0, 5.0, 3.0])
    result = _tail_area_fdr(observed, pooled, scale=1.0)
    assert result == pytest.approx([0.0, 0.5, 2 / 3, 0.75])


def test_tail_area_fdr_tie_group():
    observed = np.array([10.0, 8.0, 8.0, 4.0])
    pooled = np.array([9.0, 7.5, 5.0, 3.0])
    result = _tail_area_fdr(observed, pooled, scale=1.0)
    assert result == pytest.approx([0.0, 1 / 3, 1 / 3, 0.75])
```

Correct the existing backwards test named in spec §1.10 item 16
(`tests/test_probability.py:55-62`, `test_region_fdr_is_monotone_in_area`): change

```python
assert np.all(np.diff(q[order]) <= 1e-12)
```

to

```python
assert np.all(np.diff(q[order]) >= -1e-12)
```

and update its surrounding comment (it currently explains the *old* backwards
assertion) to state that q-values are non-decreasing as area weakens, which is what
correct monotonicity means for a strongest-to-weakest ordering.

Add a `region_evidence_score` end-to-end test and the `estimate_pi0`-unaffected test to
`tests/test_eiger2_core.py`:

```python
def test_region_evidence_score_exercises_corrected_tail_area_fdr():
    matrix, locs, _planted = simulate(
        n_samples=50, strata=[(0.2, 0.4)], replicates=3,
        spacing=250, dmr_length=10, seed=99,
    )
    config = Eiger2Config(
        window=4, null_replicates=8, random_seed=99, compute_posterior=True,
    )
    regions, _values, manifest = call_dmrs(matrix, locs, config=config)
    assert "posterior" in manifest
    assert regions["evidence_score_extrapolated"].any() or len(regions) == 0


def test_estimate_pi0_output_is_unchanged_by_the_tail_area_fdr_fix():
    from eiger2.probability import estimate_pi0
    rng = np.random.default_rng(41)
    observed = rng.exponential(2.0, size=100)
    pooled = rng.exponential(2.0, size=600)
    # estimate_pi0 uses counts, not the cummin -- confirm it takes no code path
    # through _tail_area_fdr by checking its result matches a hand re-derivation
    # from its own (unedited) counting logic.
    result_a = estimate_pi0(observed, pooled, observed_edges=500, null_edges=3000)
    result_b = estimate_pi0(observed, pooled, observed_edges=500, null_edges=3000)
    assert result_a == pytest.approx(result_b)
```

(The last test is a determinism/no-regression smoke check; `estimate_pi0`'s
implementation is untouched by this task, so this mainly documents the invariant
spec §1.7 asks to confirm — read `estimate_pi0`'s body during Step 3 to verify by
inspection that it never calls `_tail_area_fdr`, and note that confirmation in the
Task 13 STOP report rather than only in a test.)

- [ ] **Step 2: Run and confirm failure**

```bash
pytest tests/test_probability.py -k tail_area_fdr -v
pytest tests/test_eiger2_core.py -k "evidence_score or estimate_pi0" -v
```

- [ ] **Step 3: Implement the corrections**

Replace `_tail_area_fdr` in `probability.py` (lines 208–220):

```python
def _tail_area_fdr(
    observed_areas: np.ndarray, pooled_null: np.ndarray, scale: float
) -> np.ndarray:
    """Point tail-area FDR with the corrected reverse-cummin and tie-aware rank.

    Fallback for the extrapolated tail of ``local_fdr`` -- governs the strongest
    regions. Mirrors ``core._fdr_from_areas_result``'s point-estimate path exactly
    (see that function's docstring for the direction/tie rationale); no Garwood
    bounds here, since this fallback is a point estimate only.
    """

    pooled = np.sort(pooled_null[np.isfinite(pooled_null)])
    m = observed_areas.size
    order = np.argsort(-observed_areas, kind="mergesort")
    ranked = observed_areas[order]

    new_group = np.empty(m, dtype=bool)
    new_group[0] = True
    new_group[1:] = ranked[1:] != ranked[:-1]
    group_of_row = np.cumsum(new_group) - 1
    last_row_of_group = np.flatnonzero(np.append(new_group[1:], True))
    R = (last_row_of_group + 1).astype(float)
    cutoff = ranked[new_group]
    K = (pooled.size - np.searchsorted(pooled, cutoff, side="left")).astype(float)

    raw = np.clip(scale * K / R, 0.0, 1.0)
    q_group = np.minimum.accumulate(raw[::-1])[::-1]

    out = np.empty(m, dtype=float)
    out[order] = q_group[group_of_row]
    return out
```

Verify by reading (do not edit) `estimate_pi0`'s body in `probability.py` that it
computes counts directly and never calls `_tail_area_fdr`; record that confirmation
in the STOP report per Step 1's note.

Add to `core.py`, immediately above `qvalues_from_track` (core.py:1081):

```python
def qvalue_result_from_track(
    statistic: np.ndarray,
    regions: np.ndarray,
    threshold: float,
    *,
    eligible: np.ndarray | None = None,
    chromosome: np.ndarray | None = None,
    min_cpgs: int = 5,
    max_gap_edges: int = 0,
    replicates: int = 50,
    seed: int = 1,
    length_stratified: bool = True,
) -> tuple[RegionFdrResult, dict]:
    """Region-level FDR (point estimate, resolution, and Garwood bounds) for ANY
    caller, from its own statistic track. See ``qvalues_from_track`` for the
    thin-wrapper form used where only the point estimate is needed."""

    statistic = np.asarray(statistic, dtype=float)
    regions = np.asarray(regions, dtype=int).reshape(-1, 2)
    if eligible is None:
        eligible = np.isfinite(statistic)
    if regions.size == 0:
        empty = region_fdr_result(
            np.empty(0), [], observed_edges=0, null_edges=0,
        )
        return empty, {"n_regions": 0}

    observed_areas = region_area(statistic, regions, threshold)
    observed_lengths = regions[:, 1] - regions[:, 0] + 1

    tracks = permutation_null_tracks(
        statistic, eligible, chromosome, replicates=replicates, seed=seed
    )
    null_areas, null_lengths, null_edges = [], [], 0
    for track, usable in tracks:
        null_edges += int(usable.sum())
        found = segment(
            track, usable, threshold,
            min_cpgs=min_cpgs, max_gap_edges=max_gap_edges,
        )
        null_areas.append(region_area(track, found, threshold))
        null_lengths.append(
            (found[:, 1] - found[:, 0] + 1) if found.size
            else np.empty(0, dtype=int)
        )

    observed_edges = int(np.sum(eligible & np.isfinite(statistic)))
    result = region_fdr_result(
        observed_areas, null_areas,
        observed_edges=observed_edges,
        null_edges=max(null_edges, 1),
        observed_lengths=observed_lengths if length_stratified else None,
        null_length_sets=null_lengths if length_stratified else None,
    )
    diagnostics = {
        "mode": "permutation",
        "n_regions": len(regions),
        "null_replicates": len(tracks),
        "null_regions_total": int(sum(len(a) for a in null_areas)),
        "null_regions_per_replicate": (
            float(np.mean([len(a) for a in null_areas])) if null_areas else 0.0
        ),
        "threshold": float(threshold),
        "observed_edges": observed_edges,
        "null_edges": max(null_edges, 1),
        "scale": observed_edges / max(null_edges, 1),
        "null_count_overdispersion_warning": result.diagnostics.get(
            "null_count_overdispersion_warning"
        ),
    }
    return result, diagnostics


def qvalues_from_track(
    statistic: np.ndarray,
    regions: np.ndarray,
    threshold: float,
    *,
    eligible: np.ndarray | None = None,
    chromosome: np.ndarray | None = None,
    min_cpgs: int = 5,
    max_gap_edges: int = 0,
    replicates: int = 50,
    seed: int = 1,
    length_stratified: bool = True,
) -> tuple[np.ndarray, dict]:
    """Thin wrapper over qvalue_result_from_track returning only the point estimate.

    Kept for existing callers; signature unchanged.
    """

    result, diagnostics = qvalue_result_from_track(
        statistic, regions, threshold,
        eligible=eligible, chromosome=chromosome, min_cpgs=min_cpgs,
        max_gap_edges=max_gap_edges, replicates=replicates, seed=seed,
        length_stratified=length_stratified,
    )
    return result.qvalue, diagnostics
```

Delete the old body of `qvalues_from_track` that this replaces (core.py:1081–1150,
the part after the docstring).

Add `qvalue_result_from_track` to `__init__.py`'s import and `__all__` (alphabetically,
right after `"qvalue_result_from_track"`... i.e. it sorts before `"region_fdr"`,
after `"null_threshold"`):

```python
    "null_threshold",
    "qvalue_result_from_track",
    "region_fdr",
```

and the matching `from .core import (..., qvalue_result_from_track, ...)` line.

In `cli.py`, change `run_legacy`'s call (around line 286) from `qvalues_from_track` to
`qvalue_result_from_track`:

```python
    fdr_result, fdr_diagnostics = None, None
    if fdr_replicates > 0 and len(blocks):
        chrom = locations["chr"].to_numpy()[: statistic.size]
        fdr_result, fdr_diagnostics = qvalue_result_from_track(
            statistic, np.asarray(blocks, dtype=int), cov_threshold,
            eligible=np.isfinite(statistic), chromosome=chrom,
            min_cpgs=min_cpgs, max_gap_edges=0,
            replicates=fdr_replicates, seed=seed,
        )
    qvalues = fdr_result.qvalue if fdr_result is not None else None
```

replacing the old `qvalues, fdr_diagnostics = None, None` / `qvalues_from_track(...)`
block. Update the `import` line at the top of `cli.py` to bring in
`qvalue_result_from_track` alongside the existing `qvalues_from_track` import (leave
`qvalues_from_track` imported too — nothing else in `cli.py` needs it after this
change, but other modules may; grep to confirm before removing, and remove the import
only if actually unused elsewhere).

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_probability.py -v
pytest tests/test_eiger2_core.py -v
```

- [ ] **Step 5: Commit**

```bash
git add Python/eiger2/probability.py Python/eiger2/core.py Python/eiger2/cli.py \
        Python/eiger2/__init__.py tests/test_probability.py tests/test_eiger2_core.py
git commit -m "fix: correct probability._tail_area_fdr direction/tie defect; add qvalue_result_from_track"
```

---

### Task 6: Replicate defaults (20 → 100)

**Files:**
- Modify: `Python/eiger2/core.py` (`Eiger2Config.null_replicates`,
  `permutation_null_tracks`, `qvalues_from_track`/`qvalue_result_from_track` defaults)
- Modify: `Python/eiger2/cli.py` (`--null-replicates` help text)
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- Produces: no new interface; changes only default values, each pinned by a test.

- [ ] **Step 1: Write the failing tests**

```python
def test_null_replicates_default_is_100():
    assert Eiger2Config().null_replicates == 100


def test_permutation_null_tracks_default_replicates_is_100():
    import inspect
    sig = inspect.signature(permutation_null_tracks)
    assert sig.parameters["replicates"].default == 100


def test_qvalue_result_from_track_default_replicates_is_100():
    import inspect
    sig = inspect.signature(qvalue_result_from_track)
    assert sig.parameters["replicates"].default == 100
    sig2 = inspect.signature(qvalues_from_track)
    assert sig2.parameters["replicates"].default == 100
```

Add the necessary imports (`permutation_null_tracks`, `qvalues_from_track`,
`qvalue_result_from_track`) to the top of `tests/test_eiger2_core.py` if not already
present.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest tests/test_eiger2_core.py -k "default_is_100 or default_replicates_is_100" -v
```

Expected: three failures, defaults still `20`/`20`/`50`.

- [ ] **Step 3: Change the three defaults**

`core.py:145`:

```python
    null_replicates: int = 100
```

Update the field's context: there is no docstring on `null_replicates` currently: add
one directly above it explaining the resolution-floor rationale (spec §1.8 table):

```python
    null_replicates: int = 100
    """Null tracks pooled for the region FDR. At B=20 the strongest region's
    q-value is either exactly 0 or >= 0.05 (resolution = s/1, and s ~ 0.05 at this
    replicate count) -- no resolution at a 0.05 decision boundary. B=100 gives
    resolution ~0.01 and an R=1 upper_95 bound around 0.078; see docs/FDR.md."""
```

`core.py:761`, `permutation_null_tracks`:

```python
def permutation_null_tracks(
    statistic: np.ndarray,
    eligible: np.ndarray,
    chromosome: np.ndarray | None = None,
    *,
    replicates: int = 100,
    seed: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
```

`core.py`, both `qvalue_result_from_track` and `qvalues_from_track` (Task 5's new
code): change `replicates: int = 50` to `replicates: int = 100` in both signatures.

In `cli.py`, update the `--null-replicates` help text (cli.py:715–717):

```python
        p.add_argument("--null-replicates", dest="null_replicates", type=int,
                       default=None, help="null tracks (default 100; use "
                                          "200-300 with --compute-posterior)")
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_eiger2_core.py -k "default_is_100 or default_replicates_is_100" -v
```

- [ ] **Step 5: Measure the runtime cost (for the STOP report, not a test)**

```bash
cd /Users/tobymilne-clark/Documents/Code/Github_beta_Versions/EIGER-wt-fdr
python -c "
import time
from eiger2 import Eiger2Config, call_dmrs
from eiger2.benchmark import simulate

matrix, locs, _ = simulate(n_samples=200, strata=[(0.1, 0.3)], replicates=6,
                            spacing=400, dmr_length=20, seed=7)
for reps in (20, 100):
    config = Eiger2Config(window=4, null_replicates=reps, random_seed=7)
    started = time.perf_counter()
    regions, _, _ = call_dmrs(matrix, locs, config=config)
    print(reps, 'replicates:', time.perf_counter() - started, 's,', len(regions), 'regions')
"
```

Record both numbers verbatim in the Task 13 STOP report.

- [ ] **Step 6: Commit**

```bash
git add Python/eiger2/core.py Python/eiger2/cli.py tests/test_eiger2_core.py
git commit -m "feat: raise null replicate default from 20 to 100 across config and helpers"
```

---

### Task 7: `--fdr-decision {point,upper}` and the vacuity guard

**Files:**
- Modify: `Python/eiger2/core.py` (`Eiger2Config.fdr_decision`,
  `check_fdr_decision_vacuity`, `call_dmrs` wiring)
- Modify: `Python/eiger2/cli.py` (`--fdr-decision` flag, `run_legacy` wiring,
  `command_call` manifest)
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- Consumes: `RegionFdrResult` (Task 3).
- Produces: `core.check_fdr_decision_vacuity(result, *, scale, current_replicates,
  target_fdr) -> None` (raises `ValueError`), `Eiger2Config.fdr_decision: str =
  "point"`, `manifest["fdr_decision"]` in both call paths.

- [ ] **Step 1: Write the failing tests**

```python
def test_fdr_decision_field_defaults_to_point():
    assert Eiger2Config().fdr_decision == "point"


def test_fdr_decision_rejects_invalid_value():
    with pytest.raises(ValueError, match="fdr_decision"):
        Eiger2Config(fdr_decision="maybe")


def test_point_and_upper_decision_modes_are_deterministic_with_same_seed():
    matrix, locs, _ = simulate(
        n_samples=60, strata=[(0.15, 0.35)], replicates=4,
        spacing=300, dmr_length=12, seed=51,
    )
    for mode in ("point", "upper"):
        config = Eiger2Config(
            window=4, null_replicates=100, random_seed=51, fdr_decision=mode,
        )
        first, _, _ = call_dmrs(matrix, locs, config=config)
        second, _, _ = call_dmrs(matrix, locs, config=config)
        assert first["significant"].tolist() == second["significant"].tolist()
        assert first["region_qvalue"].tolist() == second["region_qvalue"].tolist()


def test_upper_decision_uses_upper_bound_not_point_estimate():
    matrix, locs, _ = simulate(
        n_samples=60, strata=[(0.15, 0.35)], replicates=4,
        spacing=300, dmr_length=12, seed=52,
    )
    point_config = Eiger2Config(
        window=4, null_replicates=100, random_seed=52, fdr_decision="point",
    )
    upper_config = Eiger2Config(
        window=4, null_replicates=100, random_seed=52, fdr_decision="upper",
    )
    point_regions, _, _ = call_dmrs(matrix, locs, config=point_config)
    upper_regions, _, _ = call_dmrs(matrix, locs, config=upper_config)
    # upper mode can only be as-or-more conservative than point mode
    assert int(upper_regions["significant"].sum()) <= int(point_regions["significant"].sum())


def test_fdr_decision_upper_with_too_few_replicates_raises_naming_b_required():
    matrix, locs, _ = simulate(
        n_samples=40, strata=[(0.1, 0.2)], replicates=2,
        spacing=250, dmr_length=8, seed=53,
    )
    config = Eiger2Config(
        window=4, null_replicates=2, random_seed=53, fdr_decision="upper",
        target_fdr=0.01,
    )
    with pytest.raises(ValueError, match="B_required|null_replicates|Increase"):
        call_dmrs(matrix, locs, config=config)
```

- [ ] **Step 2: Run and confirm failure**

```bash
pytest tests/test_eiger2_core.py -k fdr_decision -v
```

- [ ] **Step 3: Implement**

Add to `Eiger2Config` in `core.py`, near `target_fdr` (line 255):

```python
    fdr_decision: str = "point"
    """``"point"``: significant = region_qvalue <= target_fdr (unchanged from
    v2.0.0). ``"upper"``: significant = region_qvalue_upper_95 <= target_fdr, the
    conservative decision under finite-null uncertainty. Never switches silently --
    always read from the manifest's ``fdr_decision`` field."""
```

Add to `__post_init__` validation (core.py, in the block ending at line 322):

```python
        if self.fdr_decision not in {"point", "upper"}:
            raise ValueError("fdr_decision must be 'point' or 'upper'")
```

Add `check_fdr_decision_vacuity` to `core.py`, near `region_fdr_result`:

```python
def check_fdr_decision_vacuity(
    result: RegionFdrResult,
    *,
    scale: float,
    current_replicates: int,
    target_fdr: float,
) -> None:
    """Refuse to silently return zero significant regions under --fdr-decision upper.

    The best achievable upper_95 bound at the strongest possible cutoff (K=0, R=1)
    is s * U(K=0). If that already exceeds target_fdr, no region in this run can
    ever pass the upper-bound decision rule regardless of its area -- report the
    replicate count that would fix it instead of returning an empty, misleading
    result.
    """

    if scale <= 0:
        return
    alpha = 1.0 - result.confidence_level
    m_bonferroni = result.diagnostics.get("fdr_interval_bonferroni_m", 1)
    astar = alpha / max(int(m_bonferroni), 1)
    u0 = 0.5 * chi2.ppf(1.0 - astar / 2.0, 2.0)
    best_upper = min(1.0, scale * u0)
    if best_upper <= target_fdr:
        return
    required = math.ceil(current_replicates * scale * u0 / target_fdr)
    raise ValueError(
        f"--fdr-decision upper cannot reach target_fdr={target_fdr}: current "
        f"null_replicates={current_replicates} gives scale={scale:.6f} and the "
        f"best achievable region_qvalue_upper_95 is {best_upper:.4f} at K=0, R=1. "
        f"Increase --null-replicates to at least {required} (B_required)."
    )
```

Add `import math` to `core.py`'s top-level imports if not already present (check
first — it is likely absent since no prior code needed it).

Wire into `call_dmrs` (core.py, right after the `qvalues = region_fdr(...)` call at
line 1388 — replace that whole call with the `region_fdr_result` call from Task 8, but
for this task, add the guard immediately after obtaining `fdr_result`):

```python
    if config.fdr_decision == "upper":
        check_fdr_decision_vacuity(
            fdr_result,
            scale=fdr_result.diagnostics.get("scale", 0.0),
            current_replicates=config.null_replicates,
            target_fdr=config.target_fdr,
        )
```

(This references `fdr_result`, which Task 8 introduces when it replaces the
`qvalues = region_fdr(...)` call with `fdr_result = region_fdr_result(...)`. Land this
task's guard call as part of that same edit in Task 8 rather than touching
`call_dmrs` twice — see Task 8 Step 3 for the merged diff. Keep the guard function and
its tests here; the call site lands with Task 8.)

Change the `significant` computation in `call_dmrs` (core.py:1468–1471) to read from
whichever column the decision mode selects — again, this line moves in Task 8's edit
of the same row-building loop; for this task, express the logic that Task 8 will drop
in:

```python
decision_qvalue = (
    fdr_result.qvalue_upper if config.fdr_decision == "upper" else fdr_result.qvalue
)
...
"significant": bool(
    decision_qvalue[position] <= config.target_fdr
    and minority >= config.min_prevalence_samples
),
```

and add `manifest["fdr_decision"] = config.fdr_decision` near the other manifest
assignments at the end of `call_dmrs` (core.py:1495–1499).

In `cli.py`, add the CLI flag inside `build_parser`'s shared-argument block, right
after `--null-mode` (cli.py:722–728):

```python
        p.add_argument("--fdr-decision", dest="fdr_decision",
                       choices=["point", "upper"], default="point",
                       help="'point' (default): significant iff region_qvalue <= "
                            "target-fdr. 'upper': significant iff "
                            "region_qvalue_upper_95 <= target-fdr -- the "
                            "conservative decision under finite-null uncertainty.")
```

Thread it into `run_legacy`'s signature and body (cli.py:244–260 and the `significant`
block at line 354–357):

```python
def run_legacy(
    matrix: np.ndarray,
    locations: pd.DataFrame,
    sample_names: list[str],
    *,
    cov_threshold: float,
    delta_threshold: float,
    min_cpgs: int,
    impute: bool,
    fdr_replicates: int = 0,
    target_fdr: float = 0.05,
    min_prevalence: int = 0,
    seed: int = 1,
    fdr_decision: str = "point",
) -> tuple[pd.DataFrame, dict]:
```

```python
    if fdr_result is not None and fdr_decision == "upper":
        check_fdr_decision_vacuity(
            fdr_result,
            scale=fdr_diagnostics.get("scale", 0.0),
            current_replicates=fdr_replicates,
            target_fdr=target_fdr,
        )
```

placed right after the `fdr_result, fdr_diagnostics = qvalue_result_from_track(...)`
call from Task 5, and:

```python
        if qvalues is not None:
            decision_series = (
                frame["region_qvalue_upper_95"] if fdr_decision == "upper"
                else frame["region_qvalue"]
            )
            frame["significant"] = (
                (decision_series <= target_fdr) & frame["passes_prevalence"]
            )
```

replacing the old `frame["significant"] = ...` block (this references
`region_qvalue_upper_95`, added to the legacy row dict in Task 8). Add
`"fdr_decision": fdr_decision,` to `run_legacy`'s returned manifest dict.

In `command_call` (cli.py:423), pass the new arg through to `run_legacy`:

```python
        regions, caller_manifest = run_legacy(
            matrix, locations, samples,
            cov_threshold=args.cov_threshold,
            delta_threshold=args.delta_threshold,
            min_cpgs=args.min_cpgs if args.min_cpgs is not None else 5,
            impute=not args.no_impute,
            fdr_replicates=args.fdr_replicates,
            target_fdr=args.target_fdr,
            min_prevalence=args.min_prevalence_samples or 0,
            seed=args.random_seed,
            fdr_decision=args.fdr_decision,
        )
```

Import `check_fdr_decision_vacuity` in `cli.py`'s existing `from .core import (...)`
block.

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_eiger2_core.py -k fdr_decision -v
```

(Some tests in this file reference `fdr_result` / `region_qvalue_upper_95`, which do
not exist in `call_dmrs`'s output until Task 8 lands — if running Task 7 in isolation
these tests will fail on `KeyError`/`AttributeError` until Task 8's merge point;
implement Task 7 and Task 8 as one combined commit if executing tasks strictly in
order, or accept red tests here and re-run after Task 8. State explicitly in whichever
task's STOP contribution which commit made these tests pass.)

- [ ] **Step 5: Commit**

```bash
git add Python/eiger2/core.py Python/eiger2/cli.py tests/test_eiger2_core.py
git commit -m "feat: add --fdr-decision point|upper with vacuity guard"
```

---

### Task 8: Output schema — six new region columns, `call_dmrs` wiring, `schema_version`

**Files:**
- Modify: `Python/eiger2/core.py` (`_REGION_COLUMNS`, `call_dmrs` row-building loop)
- Modify: `Python/eiger2/cli.py` (legacy row dict, CSV `confidence_level` guard)
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- Consumes: `RegionFdrResult` (Task 3), `check_fdr_decision_vacuity` (Task 7).
- Produces: six new `regions.csv` columns; `manifest["schema_version"] = "2.1"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_schema_round_trip_has_documented_columns_in_documented_order():
    matrix, locs, _ = simulate(
        n_samples=60, strata=[(0.15, 0.35)], replicates=4,
        spacing=300, dmr_length=12, seed=61,
    )
    config = Eiger2Config(window=4, null_replicates=20, random_seed=61)
    regions, _values, manifest = call_dmrs(matrix, locs, config=config)
    idx = list(regions.columns).index("region_qvalue")
    expected_tail = [
        "region_tail_fdr", "region_qvalue_lower_95", "region_qvalue_upper_95",
        "region_qvalue_resolution", "null_exceedances_at_cutoff",
        "region_qvalue_cutoff_rank",
    ]
    assert list(regions.columns)[idx + 1: idx + 1 + len(expected_tail)] == expected_tail
    assert manifest["schema_version"] == "2.1"


def test_confidence_level_other_than_95_returns_arrays_but_is_not_csv_exportable():
    result = region_fdr_result(
        np.array([5.0, 3.0]), [np.array([1.0, 2.0])],
        observed_edges=10, null_edges=5, confidence_level=0.90,
    )
    assert not np.all(np.isnan(result.qvalue))  # Python API still returns arrays

    from eiger2.cli import _require_default_confidence_level
    with pytest.raises(ValueError, match="95%"):
        _require_default_confidence_level(0.90)
```

Stage 1 exposes no `--confidence-level` CLI flag (spec: "`confidence_level` stays
available in the Python API for research use"; no flag is requested anywhere in
§1.9). The guard therefore has nothing to intercept from the CLI today — it exists so
that if a future stage adds such a flag, CSV export already refuses non-default
values. Test the guard function directly, as above, rather than routing a
nonexistent flag through `command_call`. Record this scope decision in the Task 13
STOP report.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest tests/test_eiger2_core.py -k "schema_round_trip or confidence_level" -v
```

- [ ] **Step 3: Implement**

Add a CSV-export guard function to `cli.py` (used by `command_call` right before
`regions.to_csv(...)`):

```python
def _require_default_confidence_level(regions_confidence_level: float | None) -> None:
    if regions_confidence_level is not None and regions_confidence_level != 0.95:
        raise ValueError(
            f"regions.csv export is fixed at the 95% confidence level; the "
            f"requested level {regions_confidence_level} is a Python-API-only "
            f"research setting and cannot be written to CSV. Generalised suffixes "
            f"belong to a later schema version."
        )
```

Since `call_dmrs` has no CLI-exposed way to set `confidence_level` (spec: "stays
available in the Python API for research use" — no `--confidence-level` flag is
requested anywhere in §1.9), do **not** add a CLI flag. Instead call the guard in
`command_call` with the value `call_dmrs` actually used
(`getattr(config, "confidence_level", 0.95)` — see below), which for every CLI
invocation will be `0.95` by construction since `Eiger2Config` has no
`confidence_level` field (it is a `region_fdr_result` keyword-only parameter, not a
config field). This makes the guard permanently satisfied from the CLI today and is
the honest implementation of "if it differs from 0.95, cli must refuse" given nothing
in this stage lets a CLI user set it to anything else. Write the guard function and a
direct unit test of it (passing `0.90` explicitly) rather than trying to thread a
nonexistent CLI flag through `argparse` — record this scope decision in the Task 13
STOP report. The guard also accepts the no-op cases; extend Step 1's test with them
in place rather than as a second test:

```python
def test_confidence_level_other_than_95_returns_arrays_but_is_not_csv_exportable():
    result = region_fdr_result(
        np.array([5.0, 3.0]), [np.array([1.0, 2.0])],
        observed_edges=10, null_edges=5, confidence_level=0.90,
    )
    assert not np.all(np.isnan(result.qvalue))  # Python API still returns arrays

    from eiger2.cli import _require_default_confidence_level
    _require_default_confidence_level(0.95)  # no raise
    _require_default_confidence_level(None)  # no raise
    with pytest.raises(ValueError, match="95%"):
        _require_default_confidence_level(0.90)
```

Update `_REGION_COLUMNS` in `core.py` (currently lines 1503–1533), inserting six new
entries immediately after `"region_qvalue"`:

```python
_REGION_COLUMNS = [
    "region_id",
    "chr",
    "start",
    "end",
    "n_cpg",
    "length_bp",
    "cpg_index_start",
    "cpg_index_end",
    "statistic",
    "statistic_mean",
    "statistic_min",
    "statistic_threshold",
    "region_area",
    "region_qvalue",
    "region_tail_fdr",
    "region_qvalue_lower_95",
    "region_qvalue_upper_95",
    "region_qvalue_resolution",
    "null_exceedances_at_cutoff",
    "region_qvalue_cutoff_rank",
    "region_evidence_score",
    "evidence_score_extrapolated",
    "minority_size_descriptive",
    "minority_delta_descriptive",
    "passes_prevalence",
    "significant",
    "n_samples_observed",
    "missing_rate",
    "mean_beta",
    "delta_beta_descriptive",
    "largest_gap_descriptive",
    "direction_descriptive",
    "minor_group_size_descriptive",
    "minor_group_fraction_descriptive",
    "minor_group_samples_descriptive",
    "pvalue_descriptive",
]
```

Replace the `qvalues = region_fdr(...)` call in `call_dmrs` (core.py:1388–1395) with:

```python
    fdr_result = region_fdr_result(
        observed_areas,
        null_areas,
        observed_edges=int(np.sum(eligible & np.isfinite(statistic))),
        null_edges=max(null_edge_total, 1),
        observed_lengths=observed_lengths if config.length_stratified_fdr else None,
        null_length_sets=null_lengths if config.length_stratified_fdr else None,
    )
    if config.fdr_decision == "upper":
        check_fdr_decision_vacuity(
            fdr_result,
            scale=fdr_result.diagnostics.get("scale", 0.0),
            current_replicates=config.null_replicates,
            target_fdr=config.target_fdr,
        )
    qvalues = fdr_result.qvalue
    manifest["fdr"] = fdr_result.diagnostics
```

(the `manifest["fdr"] = fdr_result.diagnostics` line is what Task 4's
`caller_manifest["fdr"]` lookup in `cli.command_call` needs — this closes that
forward reference.)

In the region row-building loop (core.py:1426–1486), change the `significant`
computation and add the six new fields:

```python
    decision_qvalue = (
        fdr_result.qvalue_upper if config.fdr_decision == "upper" else fdr_result.qvalue
    )
    ...
    for position, (first, last) in enumerate(regions):
        ...
        rows.append(
            {
                ...
                "region_area": float(observed_areas[position]),
                "region_qvalue": float(qvalues[position]),
                "region_tail_fdr": float(fdr_result.tail_fdr[position]),
                "region_qvalue_lower_95": float(fdr_result.qvalue_lower[position]),
                "region_qvalue_upper_95": float(fdr_result.qvalue_upper[position]),
                "region_qvalue_resolution": float(fdr_result.qvalue_resolution[position]),
                "null_exceedances_at_cutoff": int(fdr_result.null_exceedances[position]),
                "region_qvalue_cutoff_rank": int(fdr_result.qvalue_cutoff_rank[position]),
                "region_evidence_score": float(posterior[position]),
                "evidence_score_extrapolated": bool(posterior_extrapolated[position]),
                "minority_size_descriptive": int(minority),
                "minority_delta_descriptive": float(minority_delta),
                "passes_prevalence": bool(minority >= config.min_prevalence_samples),
                "significant": bool(
                    decision_qvalue[position] <= config.target_fdr
                    and minority >= config.min_prevalence_samples
                ),
                ...
            }
        )
```

(keep every other key in the row dict exactly as it already is — only insert the six
new keys and change `"significant"`'s condition).

Add `manifest["schema_version"] = "2.1"` and `manifest["fdr_decision"] =
config.fdr_decision` next to the other manifest assignments at the end of `call_dmrs`
(core.py:1495–1499).

In `cli.py`'s `run_legacy` row dict (cli.py:315–337), add the six fields, guarded by
`fdr_result is not None`:

```python
            "region_qvalue": (
                float(qvalues[block_index]) if qvalues is not None else np.nan
            ),
            "region_tail_fdr": (
                float(fdr_result.tail_fdr[block_index])
                if fdr_result is not None else np.nan
            ),
            "region_qvalue_lower_95": (
                float(fdr_result.qvalue_lower[block_index])
                if fdr_result is not None else np.nan
            ),
            "region_qvalue_upper_95": (
                float(fdr_result.qvalue_upper[block_index])
                if fdr_result is not None else np.nan
            ),
            "region_qvalue_resolution": (
                float(fdr_result.qvalue_resolution[block_index])
                if fdr_result is not None else np.nan
            ),
            "null_exceedances_at_cutoff": (
                int(fdr_result.null_exceedances[block_index])
                if fdr_result is not None else -1
            ),
            "region_qvalue_cutoff_rank": (
                int(fdr_result.qvalue_cutoff_rank[block_index])
                if fdr_result is not None else -1
            ),
```

Add `"schema_version": "2.1"` to `run_legacy`'s returned manifest dict.

In `command_call`, right before `regions.to_csv(out / "regions.csv", index=False)`
(cli.py:472), add:

```python
    _require_default_confidence_level(None if args.legacy else 0.95)
    regions.to_csv(out / "regions.csv", index=False)
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_eiger2_core.py -v
pytest tests/test_probability.py -v
```

Also rerun Task 7's tests now, since this task closes the `fdr_result` forward
reference they depend on:

```bash
pytest tests/test_eiger2_core.py -k fdr_decision -v
```

- [ ] **Step 5: Commit**

```bash
git add Python/eiger2/core.py Python/eiger2/cli.py tests/test_eiger2_core.py
git commit -m "feat: add six FDR columns to regions.csv, wire RegionFdrResult through call_dmrs and run_legacy"
```

---

### Task 9: Manifest envelope (§0.5 required fields)

**Files:**
- Create: `Python/eiger2/manifest_utils.py`
- Modify: `Python/eiger2/cli.py` (`command_call`)
- Modify: `benchmarking/calibration_study.py`, `benchmarking/benchmark_suite.py`
- Test: `tests/test_eiger2_core.py` (or a new `tests/test_manifest_utils.py` if that
  reads more naturally — pick one file and be consistent)

**Interfaces:**
- Produces: `manifest_utils.build_manifest_envelope(*, config_dict, git_commit,
  input_checksums, random_seed, derived_seeds, runtime_seconds, peak_memory_mb,
  warnings, fdr_decision, parent_run_id=None, parent_manifest_checksum=None) -> dict`
  containing every §0.5 key. Consumed by `command_call` and both benchmark scripts.

- [ ] **Step 1: Write the failing test**

Create `tests/test_manifest_utils.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from eiger2.manifest_utils import build_manifest_envelope, REQUIRED_MANIFEST_KEYS


def test_build_manifest_envelope_has_every_required_key():
    envelope = build_manifest_envelope(
        config_dict={"target_fdr": 0.05},
        git_commit="deadbeef",
        input_checksums={"matrix.npy": "abc123"},
        random_seed=7,
        derived_seeds={"null_permutation": [1, 2, 3]},
        runtime_seconds=1.23,
        peak_memory_mb=42.0,
        warnings=[],
        fdr_decision="point",
    )
    for key in REQUIRED_MANIFEST_KEYS:
        assert key in envelope, key
    assert envelope["schema_version"] == "2.1"
    assert envelope["fdr_decision"] == "point"
    assert envelope["parent_run_id"] is None


def test_build_manifest_envelope_records_parent_run_when_given():
    envelope = build_manifest_envelope(
        config_dict={}, git_commit="deadbeef", input_checksums={},
        random_seed=1, derived_seeds={}, runtime_seconds=0.1, peak_memory_mb=1.0,
        warnings=["a warning"], fdr_decision="upper",
        parent_run_id="run-123", parent_manifest_checksum="sha:xyz",
    )
    assert envelope["parent_run_id"] == "run-123"
    assert envelope["parent_manifest_checksum"] == "sha:xyz"
    assert envelope["warnings"] == ["a warning"]
```

- [ ] **Step 2: Run and confirm failure**

```bash
pytest tests/test_manifest_utils.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `Python/eiger2/manifest_utils.py`:

```python
"""Shared manifest envelope required from Stage 1 onward (spec §0.5).

Every ``manifest.json`` written by ``eiger2 call``, ``eiger2 programs``, and the
benchmark harness must carry this envelope so a downstream run, or a reviewer, can
verify what produced a given output without re-deriving it from context.
"""

from __future__ import annotations

from . import __version__

REQUIRED_MANIFEST_KEYS = (
    "schema_version",
    "software_version",
    "git_commit",
    "parent_run_id",
    "parent_manifest_checksum",
    "input_checksums",
    "resolved_configuration",
    "random_seed",
    "derived_seeds",
    "runtime_seconds",
    "peak_memory_mb",
    "package_versions",
    "warnings",
    "fdr_decision",
)


def build_manifest_envelope(
    *,
    config_dict: dict,
    git_commit: str,
    input_checksums: dict,
    random_seed: int,
    derived_seeds: dict,
    runtime_seconds: float,
    peak_memory_mb: float,
    warnings: list[str],
    fdr_decision: str,
    schema_version: str = "2.1",
    parent_run_id: str | None = None,
    parent_manifest_checksum: str | None = None,
) -> dict:
    import numpy as np
    import pandas as pd
    import scipy

    return {
        "schema_version": schema_version,
        "software_version": __version__,
        "git_commit": git_commit,
        "parent_run_id": parent_run_id,
        "parent_manifest_checksum": parent_manifest_checksum,
        "input_checksums": input_checksums,
        "resolved_configuration": config_dict,
        "random_seed": random_seed,
        "derived_seeds": derived_seeds,
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_memory_mb,
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "warnings": list(warnings),
        "fdr_decision": fdr_decision,
    }
```

Add a `_git_commit()` helper to `cli.py` (used by `command_call`):

```python
def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"
```

Add `import subprocess` to `cli.py`'s imports if not already present.

Rework the manifest construction in `command_call` (cli.py:478–498) to layer the
envelope under the existing structure (keep the existing keys — `timestamp_utc`,
`command`, `caller`, `inputs`, `environment`, `caller_manifest` — and merge the new
envelope fields alongside them rather than replacing them, since other code reads
`manifest["caller_manifest"]` etc. today):

```python
    warnings_list: list[str] = []
    if caller_manifest.get("fdr", {}).get("null_count_overdispersion_warning"):
        ratio = caller_manifest["fdr"]["null_count_overdispersion_warning"]
        warnings_list.append(
            f"pooled null region counts are overdispersed (variance/mean="
            f"{ratio:.2f})"
        )

    envelope = build_manifest_envelope(
        config_dict=(
            _config_from_args(args).to_dict() if not args.legacy else {}
        ),
        git_commit=_git_commit(),
        input_checksums={
            str(p): _checksum(p) for p in input_paths
        },
        random_seed=args.random_seed,
        derived_seeds={},  # Stage 1 introduces no new derived randomness
        runtime_seconds=elapsed,
        peak_memory_mb=_peak_memory_mb(),
        warnings=warnings_list,
        fdr_decision=args.fdr_decision,
    )
    manifest = {
        **envelope,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "caller": "legacy" if args.legacy else "eiger2",
        "inputs": {
            "files": [
                {"path": str(p), "sha256_16": _checksum(p)} for p in input_paths
            ],
            "n_cpg": int(matrix.shape[0]),
            "n_samples": int(matrix.shape[1]),
            "missing_fraction": float(np.mean(~np.isfinite(matrix))),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "caller_manifest": caller_manifest,
    }
```

Add `_peak_memory_mb()`. `ru_maxrss` is bytes on macOS/Darwin and KiB on Linux, so the
divisor differs by platform:

```python
def _peak_memory_mb() -> float:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
```

Import `build_manifest_envelope` in `cli.py`'s imports:
`from .manifest_utils import build_manifest_envelope`.

For `benchmarking/calibration_study.py` (manifest write at its line 446) and
`benchmarking/benchmark_suite.py` (`_manifest` writer at its line 587), read each
script's current manifest-construction code, then merge in the same envelope pattern
— call `build_manifest_envelope(...)` with whatever config/seed/timing values that
script already computes, and merge its result into the existing dict literal the same
way as `command_call` above. Do not duplicate the required-keys list; import it from
`eiger2.manifest_utils`.

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_manifest_utils.py -v
python -m eiger2.cli call --help   # sanity: module still imports cleanly
```

- [ ] **Step 5: Commit**

```bash
git add Python/eiger2/manifest_utils.py Python/eiger2/cli.py \
        benchmarking/calibration_study.py benchmarking/benchmark_suite.py \
        tests/test_manifest_utils.py
git commit -m "feat: add required manifest envelope (schema_version, seeds, checksums, warnings) from spec §0.5"
```

---

### Task 10: Invariant-preservation regression test

**Files:**
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- Consumes: `tests/fixtures/stage1_invariant_fixture_v200.csv` (Task 1).

- [ ] **Step 1: Write the test**

```python
def test_stage1_preserves_the_invariant_column_set():
    fixture_path = (
        Path(__file__).parent / "fixtures" / "stage1_invariant_fixture_v200.csv"
    )
    expected = pd.read_csv(fixture_path)

    matrix, locs, _ = simulate(
        n_samples=60, strata=[(0.15, 0.35)], replicates=4,
        spacing=300, dmr_length=12, missing_rate=0.02, seed=1701,
    )
    config = Eiger2Config(window=4, null_replicates=8, random_seed=1701)
    regions, _values, _manifest = call_dmrs(matrix, locs, config=config)

    invariant_columns = list(expected.columns)
    actual = regions[invariant_columns].reset_index(drop=True)
    expected = expected.reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
```

This must use the **exact same** `simulate(...)` and `Eiger2Config(...)` arguments as
`tests/fixtures/generate_stage1_invariant_fixture.py` — any drift makes the comparison
meaningless. Consider importing the input-generation code from a shared helper instead
of copy-pasting it twice, to make that drift structurally impossible:

Refactor `tests/fixtures/generate_stage1_invariant_fixture.py` to expose a
`generate_inputs()` function, and import it from both the generator script and this
test:

```python
# in generate_stage1_invariant_fixture.py
def generate_inputs():
    from eiger2.benchmark import simulate
    from eiger2 import Eiger2Config
    matrix, locs, planted = simulate(
        n_samples=60, strata=[(0.15, 0.35)], replicates=4,
        spacing=300, dmr_length=12, missing_rate=0.02, seed=1701,
    )
    config = Eiger2Config(window=4, null_replicates=8, random_seed=1701)
    return matrix, locs, config
```

and update `main()` to call it, and update the test to call it too:

```python
def test_stage1_preserves_the_invariant_column_set():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_stage1_invariant_fixture",
        Path(__file__).parent / "fixtures" / "generate_stage1_invariant_fixture.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matrix, locs, config = module.generate_inputs()

    fixture_path = (
        Path(__file__).parent / "fixtures" / "stage1_invariant_fixture_v200.csv"
    )
    expected = pd.read_csv(fixture_path).reset_index(drop=True)
    regions, _values, _manifest = call_dmrs(matrix, locs, config=config)
    actual = regions[list(expected.columns)].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
```

- [ ] **Step 2: Run and confirm pass**

```bash
pytest tests/test_eiger2_core.py -k invariant_column_set -v
```

This must pass **without regenerating the fixture** — if it fails, a Stage 1 change
altered an invariant column, which is a release-blocking bug, not a fixture problem.
Debug the diff (pandas will show which column and rows differ) before considering any
other fix.

- [ ] **Step 3: Commit**

```bash
git add tests/test_eiger2_core.py tests/fixtures/generate_stage1_invariant_fixture.py
git commit -m "test: pin the v2.0.0 invariant column set as a regression fixture"
```

---

### Task 11: Benchmarks and release gate

**Files:**
- Create: `results_v201/` (new directory, benchmark output only)
- No source changes in this task.

**Interfaces:**
- Consumes: everything from Tasks 2–9.

- [ ] **Step 1: Run the full test suite and ruff one more time**

```bash
cd /Users/tobymilne-clark/Documents/Code/Github_beta_Versions/EIGER-wt-fdr
pytest tests -q
pytest old_code/tests -q
ruff check Python/eiger2
```

All must be green (old_code/tests unaffected, since `Python/functions/**` and
`old_code/**` were never touched). Record the pass counts.

- [ ] **Step 2: Run the benchmark suite across the specified grid**

```bash
mkdir -p results_v201
python benchmarking/benchmark_suite.py command_all \
  --out results_v201/benchmark_suite \
  --seeds 100 \
  --scenarios clean nuisance high-noise small-n structured-missingness \
  --null-modes lag permutation \
  --windows 1 4 6 8 \
  --null-replicates 20 100
```

(Adjust flag names to whatever `benchmark_suite.py`'s actual `argparse` interface
uses — read its `build_parser`/`command_all` before running, since this plan's
grounding pass did not exhaustively enumerate every flag. The grid itself is fixed by
spec §1.11: 100 DMR-free seeds x 5 scenario types, planted-signal grids across
prevalence/effect/length/noise/missingness, both `null_mode` settings, fixed windows
1/4/6/8, and both `null_replicates` 20 and 100.)

```bash
python benchmarking/calibration_study.py \
  --out results_v201/calibration_study \
  --null-replicates 20 100
```

If the controlled-access 519-sample cohort is available locally, include it; if not,
note its absence in the STOP report rather than skipping silently.

- [ ] **Step 3: Check the release gate**

For each scenario, confirm from the written summary tables:

1. Corrected point q-values are monotone in area in every stratum of every run —
   spot-check with `regions.csv["region_qvalue"]` grouped by
   `length_stratum(regions.csv["n_cpg"])`, sorted by `region_area` descending, is
   non-decreasing.
2. No exact-zero point estimate lacks a nonzero `region_qvalue_upper_95` — assert
   `((regions["region_qvalue"] == 0) & (regions["region_qvalue_upper_95"] == 0)).sum() == 0`
   on every output.
3. Realised FDR in DMR-free scenarios is at or below nominal (within simulation
   uncertainty) in `point` mode.
4. The invariant column set (Task 10) is byte-identical to v2.0.0 — already verified
   as a unit test; confirm it also held across the benchmark's own fixed-window runs
   if the harness recomputes anything comparable.
5. Every README/docs/thesis number derived from old q-values is regenerated or
   explicitly labelled historical with its producing version — cross-check against
   Task 12's documentation task before closing this one.

If any gate criterion fails, **stop and report; do not adapt the method** (spec §0.6).

- [ ] **Step 4: Commit the benchmark outputs**

```bash
git add results_v201/
git commit -m "chore: run Stage 1 benchmark suite (100-seed DMR-free grid, both replicate settings)"
```

(Benchmark CSVs/JSONs may be large — if the harness produces multi-hundred-MB outputs,
confirm with the user before committing versus `.gitignore`-ing raw per-seed output
and committing only summary tables; the spec requires the summary tables to be
reported, not necessarily every raw file tracked in git.)

---

### Task 12: Documentation

**Files:**
- Create: `docs/FDR.md`
- Modify: `docs/PROBABILITY.md`, `README.md`, `QUICKSTART.md`, `pyproject.toml`,
  `Python/eiger2/__init__.py`, `CITATION.cff`

**Interfaces:**
- None (documentation only, but release-blocking per spec §0.3.8).

- [ ] **Step 1: Write `docs/FDR.md`**

Cover, in this order (spec §1.12 requires all of these; write each as a real section
with the actual formulas and reasoning, not a one-line stub):

1. The tail estimator derivation (reproduce §1.3's formulas: `s`, `R_j`, `K_j`,
   `FDR_j`, `Δ_j`, Garwood `L_j`/`U_j`, `q_i`/`q^lo_i`/`q^hi_i`).
2. Why the monotone step is a reverse scan (the forward-cummin defect, using
   Appendix A.1's numbers as the worked example: v2.0.0 gives `[0,0,0,0]`, corrected
   gives `[0, 0.5, 2/3, 0.75]`).
3. The tie-aware denominator (Appendix A.2's worked example).
4. The three distinct quantities (point estimate / resolution / Monte Carlo interval)
   and why they must never be conflated.
5. Why Phipson–Smyth `(b+1)/(B+1)` is declined here (it applies to permutation
   p-values; EIGER's quantity is a scaled expected-null count, not a p-value).
6. The Garwood interval and its Poisson assumption.
7. The Bonferroni-for-simultaneity argument (why `min_j U_j >= min_j FDR_j` requires
   the union bound, and why removing it as an "optimisation" breaks the guarantee).
8. Per-stratum vs joint coverage: "coverage is >= 1 - alpha within each stratum;
   joint coverage across S strata is only >= 1 - S*alpha" — state this explicitly, and
   that the bounds are not described as genome-wide simultaneous.
9. The overdispersion caveat: "under overdispersion the Garwood interval understates
   uncertainty" stated plainly.
10. The v2.0.0 (`schema_version = "2.0"`) to v2.0.1 (`"2.1"`) column mapping table:

| v2.0.0 column | v2.0.1 column | Notes |
|---|---|---|
| `region_qvalue` | `region_qvalue` | value changes; corrected direction+ties |
| — | `region_tail_fdr` | new: point FDR at region's own cutoff |
| — | `region_qvalue_lower_95` | new |
| — | `region_qvalue_upper_95` | new |
| — | `region_qvalue_resolution` | new |
| — | `null_exceedances_at_cutoff` | new |
| — | `region_qvalue_cutoff_rank` | new |
| `region_evidence_score` | `region_evidence_score` | value changes for extrapolated-tail regions |
| `significant` | `significant` | value changes (derived from corrected q-value; conditionally from upper bound under `--fdr-decision upper`) |

- [ ] **Step 2: Update `docs/PROBABILITY.md`**

Add a section documenting the `_tail_area_fdr` correction: same direction/tie defect
as `core._fdr_from_areas`, same fix, no Garwood bounds on this path (point estimate
only, used as the extrapolated-tail fallback in `local_fdr`). Cross-reference
`docs/FDR.md` rather than re-deriving the estimator.

- [ ] **Step 3: Update `README.md` and `QUICKSTART.md`**

Add: the six new `regions.csv` columns and what they mean (link to `docs/FDR.md`);
`--fdr-decision {point,upper}` and when to use `upper`; the `null_replicates` default
change from 20 to 100 and its runtime cost (use Task 6 Step 5's measured numbers, not
a placeholder). Add the migration note verbatim near the top of whichever of these two
files documents `regions.csv` output:

> **Migration note (v2.0.1):** corrected q-values differ from v2.0.0 even though
> candidate region boundaries are unchanged. Results generated with v2.0.0 must be
> regenerated before use — the direction of the region-level FDR's monotone step was
> reversed, which under-called false discoveries in v2.0.0 (see `docs/FDR.md`).

- [ ] **Step 4: Version bumps**

`pyproject.toml`: `version = "2.0.1"`.
`Python/eiger2/__init__.py`: `__version__ = "2.0.1"`.
`CITATION.cff`: `version: "2.0.1"`, and extend the `abstract` field with one sentence
noting the region-FDR correction and finite-null uncertainty quantification, matching
the tone of its existing prose.

Add a `docs/` note recording the v2.0.0 schema retrospectively (can live at the top of
`docs/FDR.md`'s mapping table section, already covered in Step 1 item 10 — confirm it
explicitly states `schema_version "2.0"` was the implicit v2.0.0 format, never written
to a manifest at the time, so old outputs are identified by the *absence* of a
`schema_version` key, not by its presence).

- [ ] **Step 5: Commit**

```bash
git add docs/FDR.md docs/PROBABILITY.md README.md QUICKSTART.md pyproject.toml \
        Python/eiger2/__init__.py CITATION.cff
git commit -m "docs: add FDR.md, update PROBABILITY/README/QUICKSTART, bump version to 2.0.1"
```

---

### Task 13: STOP 1 report

**Files:**
- None (report is delivered as a message / saved wherever the user asks — this task
  produces no repo changes beyond what Tasks 1–12 already committed).

**Interfaces:**
- Consumes: every prior task's recorded numbers.

- [ ] **Step 1: Assemble the §0.6 report**

Using the actual recorded values from Tasks 1–12 (not placeholders), produce:

1. Change summary, <= 10 sentences.
2. `git diff --stat` against `b8e4a19` (run in the worktree):

```bash
cd /Users/tobymilne-clark/Documents/Code/Github_beta_Versions/EIGER-wt-fdr
git diff b8e4a19 --stat
```

3. Mathematical behaviour before/after using Appendix A.1 and A.2's hand-calculated
   numbers (from Task 2's tests — print actual computed values, e.g. via
   `python -c "from eiger2.core import _fdr_from_areas_result; ..."`, not just "tests
   pass").
4. Every test command run in Tasks 1–11 and its output tail (`pytest -q` summary
   lines).
5. Benchmark commands, `results_v201/` paths, and the summary tables named in §1.11.
6. Runtime and peak memory before/after on the Task 6 Step 5 input size (200 samples,
   `dmr_length=20`), for `null_replicates=20` vs `100`.
7. Backward-compatibility assessment against §0.3.3: confirm the Task 10 fixture test
   passed, list every column that changed by design (`region_qvalue`,
   `region_evidence_score`, `evidence_score_extrapolated`, `significant`,
   `n_significant`, `n_passing_qvalue`) plus the six new columns.
8. Remaining limitations: note the `--confidence-level` CLI-flag scope decision from
   Task 8 Step 3, and any benchmark scenario that did not clear its acceptance
   criterion (should be none, per Task 11's gate).
9. `git diff --check` output and final `git status --short`:

```bash
git diff --check
git status --short
```

10. No claim stronger than the measured evidence — state pass/fail plainly, cite
    numbers, avoid "should work" language.

- [ ] **Step 2: Halt**

Per spec §0.6: **do not** create the Stage 2 (`exp/multiscale`) or Stage 3
(`exp/programs`) worktrees. Do not merge `fix/region-fdr`. Do not tag. The user tags
and uploads the v2.0.1 GitHub release from this branch once they have reviewed the
report. Deliver the report to the user and stop.

---

## Notes for the executor on task ordering

Tasks 2–6 are strictly sequential (each imports from the previous). Task 7 and Task 8
have a genuine forward/backward reference to each other (`fdr_result` / `significant`
/ `region_qvalue_upper_95`) — implement them as described, accepting that some of
Task 7's tests only turn green once Task 8 lands; do not try to force a strict
task-then-fully-green ordering between those two specifically, but do not skip writing
Task 7's tests first either. Task 9 depends on Task 8 (needs `manifest["fdr"]`
populated). Task 10 depends only on Task 1's fixture and the full pipeline being
correct, so run it last among the code tasks, after Task 9. Tasks 11–13 are strictly
after all code tasks.
