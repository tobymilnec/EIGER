# Stage 1c — support-conditioned threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After Stage 1b's null-window fix, structured missingness still yields ~0.25
realised FDR against a 0.05 target because observed exceedances cluster more tightly
than null ones under uneven local measurement support. Add a threshold conditioned on
local support (samples observed at every CpG in the window), computed identically for
observed and null so the two stay exchangeable, auto-enabled when support varies
enough to matter, with a mandatory region-level exposure diagnostic and a documented
escalation path if per-edge conditioning alone doesn't fix region-level clustering.

**Architecture:** One new shared function computes "support" for both observed
windows and null member-sets from the same underlying finite-mask array (so the two
constructions cannot drift). Binning derives quantile edges from the observed track
and applies the identical numeric edges to null positions. `segment()`/`region_area()`
are generalized to accept either a scalar or a per-edge array threshold, with the
scalar path staying bit-identical. Everything downstream (row-building, manifest,
CLI) is wired through `call_dmrs`.

**Tech Stack:** Python 3.10+, NumPy, pandas, pytest, ruff — no new dependencies.

**Spec:** `EIGER_STAGE1BCD_IMPLEMENTATION_SPEC.md` (repo root, one level above this
worktree), STAGE 1c (§1c.0–§1c.8), plus the still-binding §0 engineering contract
and §1 revised invariant contract from the same document.

## Global Constraints

- **Continue on `fix/region-fdr`, same worktree `EIGER-wt-fdr`.** No new worktree.
- **User ruling on exchangeability-gate methodology (applies from this stage
  forward):** wherever an exchangeability-ratio gate is evaluated across multiple
  seeds (benchmark commands, any Stage 1c test that checks the ratio across a seed
  loop), gate on the **per-scenario median across seeds ≤ 1.25**, and **report
  per-instance outliers without failing the run because of them alone**. This does
  NOT change the per-run CLI warning (Stage 1b, `cli.py`), which has only one
  instance per real call and stays as-is. Apply this consistently to
  `benchmarking/benchmark_suite.py::exchangeability`'s summary (Task 1, a small
  prerequisite update to already-shipped Stage 1b code) and to this stage's own
  multi-seed exchangeability checks (§1c.7 items 6 and 8).
- Per-edge conditioning changes `statistic_threshold`'s meaning (mean per-edge
  threshold over a region's edges when binning is active) — this is already
  within the narrowed §1 contract Stage 1b established (`statistic_threshold`
  changes by design); no further contract narrowing needed, but document the
  meaning change per §1c.5.
- `schema_version` bumps to `"2.2"` in **this** stage (Stage 1b correctly held it at
  `"2.1"`) — this stage adds columns and changes `statistic_threshold`'s meaning.
- The scalar-threshold path through `segment()`/`region_area()` must stay
  bit-identical when `support_binned_threshold="off"` — pin this with a test.
- Support must be computed by **one shared function** for both observed windows and
  null member-sets, taking explicit row-offset lists — never two independently
  written implementations that could drift.
- Bin edges are derived from the **observed** track only, then the identical numeric
  edges are applied to null positions — never re-derived from null data.
- The auto-enable cut is **calibrated, not guessed** — run the detector across a
  real (if necessarily bounded-scale, pending Stage 1e's full grid) benchmark sweep
  and choose the smallest cut that leaves `clean`/`nuisance`/`small_n`/`high_noise`
  unbinned while enabling `missingness`; record the calibration grid, the observed
  dispersion per scenario, and the chosen cut in this stage's STOP report (full
  `docs/FDR.md` write-up is Stage 1e's job, but the calibration itself must happen
  now, not be deferred).
- Do not implement the §1c.6 region-FDR-by-support-bin escalation speculatively —
  implement the diagnostic and its gate check first, run it for real, and only build
  the escalation path if the gate actually fires in this stage's own validation.
  Report clearly either way. **Update (post-Task 9): the gate fired** (283x–811x
  observed/null region-rate ratio across every evaluable support bin on DMR-free
  `missingness`/`small_n` data, one bin infinite, confirmed not a small-count
  artifact). Task 10 was inserted, following spec §1c.6's exact escalation
  instructions, per the user's explicit ruling to proceed with the spec-prescribed
  escalation rather than any alternative.
- Docs/tests ship in the same commit as the code they cover, except the full
  `docs/FDR.md`/`docs/PROBABILITY.md`/`docs/BENCHMARK.md` rewrite, which is
  explicitly Stage 1e's job (mirror Stage 1b's precedent here).

---

## Verified baseline (read 2026-08-19, do not re-derive)

- Worktree `EIGER-wt-fdr` clean, HEAD `15b3bd7` (Stage 1b's STOP report). Stage 1b's
  fix (`_lag_null_track`, `null_exchangeability`) is merged and independently
  verified multiple times.
- `segment(statistic, eligible, threshold, *, min_cpgs, max_gap_edges=0)` (core.py:1037)
  and `region_area(statistic, regions, threshold)` (core.py:1085) both currently take
  `threshold: float` as a bare scalar, compared via `statistic >= threshold` /
  `block - threshold`. Both need generalizing to accept `float | np.ndarray` of
  `statistic`'s shape, with the scalar path unchanged (numpy broadcasting already
  makes `statistic >= threshold` and `block - threshold` work identically whether
  `threshold` is scalar or a same-shape array that's been sliced the same way as
  `block` — the only real code change is validating shape and slicing the array
  threshold alongside `block` in `region_area`).
- `_window_from_lags`'s documented window convention (core.py:614): window centred on
  edge `e` covers CpGs `[e - window//2 + 1, e + window//2]`; `w=1` returns the raw
  lag-1 statistic directly (2-CpG adjacent pair).
- `_lag_null_track` (core.py, post-Stage-1b-fix): null window at position `i`, lag
  `L`, comprises CpGs `{i, i+L, ..., i+(w-1)L}`; `w=1` returns the raw single-lag
  statistic (2-CpG pair `(i, i+L)`).
- `_pair_statistic` (core.py:547) already has `min_paired_samples`/`min_paired_fraction`
  floors — untouched by this stage.
- `Eiger2Config.window: int = 4` (core.py:80). `__post_init__` validation ends around
  core.py:310–337; new fields' validation goes there. `to_dict()` at core.py:338.
- `_REGION_COLUMNS` (core.py:1945) is the hard-coded ordered list — new columns not
  added there are silently dropped. Current tail: `...pvalue_descriptive`.
- `call_dmrs`'s relevant section (core.py:1734–1800): `statistic = edge_statistic(...)`,
  `eligible = eligible_edges(...)`, `threshold, null_diagnostics = null_threshold(...)`,
  branch on `null_mode` to get `tracks`, `regions = segment(statistic, eligible,
  threshold, ...)`, `observed_areas = region_area(statistic, regions, threshold)`,
  then a loop over `tracks` calling `segment`/`region_area` again per null replicate
  with the same scalar `threshold`. The row-building loop (core.py:~1870+) currently
  writes `"statistic_threshold": threshold` (the bare scalar) per row.
- `benchmarking/benchmark_suite.py::command_exchangeability` (Stage 1b) currently
  prints `worst = frame["null_exchangeability_ratio"].max(); ... PASS/FAIL at gate
  1.25` — a worst-case-across-everything gate. This needs updating per this stage's
  Global Constraints (per-scenario median, outliers reported not gated).
- `cli.py`'s `build_parser` shared-argument block has an established pattern:
  `p.add_argument("--window", type=int, default=None)`,
  `p.add_argument("--null-mode", dest="null_mode", choices=[...], default=None)` —
  new `--support-binned-threshold`/`--support-bins` flags follow this exact pattern
  (`default=None` so `_config_from_args`'s `if v is not None` filter lets the
  `Eiger2Config` default take over when omitted).

---

### Task 1: Apply the exchangeability-gate median ruling to the benchmark command

**Files:**
- Modify: `benchmarking/benchmark_suite.py`

**Interfaces:** none new; changes `command_exchangeability`'s summary/gate logic only.

- [ ] **Step 1: Update the gate computation**

Replace `command_exchangeability`'s tail (the `worst = ...` block) with:

```python
    per_scenario = frame.groupby("scenario")["null_exchangeability_ratio"].median()
    print("\nper-scenario median (the gate):")
    print(per_scenario.round(4).to_string())
    outliers = frame[frame["null_exchangeability_ratio"] > 1.25]
    if len(outliers):
        print(f"\n{len(outliers)} per-instance outlier(s) above 1.25 (reported, "
              f"not gated):")
        print(outliers[["scenario", "seed", "null_exchangeability_ratio"]]
              .to_string(index=False))
    worst_median = float(per_scenario.max())
    print(f"\nworst per-scenario median: {worst_median:.3f} "
          f"({'PASS' if worst_median <= 1.25 else 'FAIL'} at gate 1.25)")
```

- [ ] **Step 2: Smoke-test**

```bash
conda activate eiger && python -m benchmarking.benchmark_suite exchangeability \
  --out-dir /tmp/exch_median_smoke --seeds 5 --scenarios clean missingness
```

Confirm the new output shows per-scenario medians, lists any per-instance outliers
without failing on them, and gates only on the worst median.

- [ ] **Step 3: Commit**

```bash
git add benchmarking/benchmark_suite.py
git commit -m "feat: gate exchangeability on per-scenario median across seeds, report per-instance outliers (Stage 1c policy)"
```

---

### Task 2: Baseline confirmation

**Files:** none.

- [ ] **Step 1**

```bash
cd /Users/tobymilne-clark/Documents/Code/Github_beta_Versions/EIGER-wt-fdr
conda activate eiger && which python && which ruff
pytest tests -q
pytest old_code/tests -q
ruff check Python/eiger2 && ruff check tests && ruff check benchmarking
```

Expect: 99 passed / 27 passed, all three ruff scopes clean (matching
`STAGE1B_STOP_REPORT.md`), plus Task 1's benchmark_suite.py change not affecting test
counts (it's not covered by a unit test — benchmark commands aren't). If not green,
stop and report before proceeding.

---

### Task 3: Shared window-support function

**Files:**
- Modify: `Python/eiger2/core.py`
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- Produces: `_window_support(finite, base_positions, offsets, *, chunk=200_000) ->
  np.ndarray` — the one function both observed and null support computation call.
  Consumed by Task 5.

- [ ] **Step 1: Write the failing tests**

```python
def test_window_support_counts_samples_finite_at_every_member():
    finite = np.array([
        [True,  True,  False],   # row 0
        [True,  False, True],    # row 1
        [True,  True,  True],    # row 2
        [False, True,  True],    # row 3
    ])
    # window covering rows {0,1}: sample 0 finite at both (T,T), sample 1 not
    # (T,F), sample 2 not (F,T) -> support = 1
    support = _window_support(finite, base_positions=np.array([0]), offsets=np.array([0, 1]))
    assert support.tolist() == [1]
    # window covering rows {1,2}: sample0 (T,T)=finite, sample1 (F,T)=not,
    # sample2 (T,T)=finite -> support = 2
    support2 = _window_support(finite, base_positions=np.array([1]), offsets=np.array([0, 1]))
    assert support2.tolist() == [2]


def test_window_support_matches_naive_loop_on_random_data():
    rng = np.random.default_rng(11)
    n_cpg, n_samples = 200, 30
    finite = rng.random((n_cpg, n_samples)) > 0.3
    offsets = np.array([0, 3, 7])
    base_positions = np.arange(n_cpg - offsets.max())
    fast = _window_support(finite, base_positions, offsets)
    naive = np.array([
        finite[base + offsets].all(axis=0).sum() for base in base_positions
    ])
    np.testing.assert_array_equal(fast, naive)


def test_window_support_chunking_does_not_change_result():
    rng = np.random.default_rng(12)
    finite = rng.random((500, 20)) > 0.4
    offsets = np.array([0, 1, 2, 3])
    base_positions = np.arange(496)
    whole = _window_support(finite, base_positions, offsets, chunk=200_000)
    chunked = _window_support(finite, base_positions, offsets, chunk=50)
    np.testing.assert_array_equal(whole, chunked)
```

- [ ] **Step 2: Run and confirm failure**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py -k window_support -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

Add to `core.py`, near `_pair_statistic`/`_lag_statistic` (support is a sibling
concept to the statistic computation):

```python
def _window_support(
    finite: np.ndarray,
    base_positions: np.ndarray,
    offsets: np.ndarray,
    *,
    chunk: int = 200_000,
) -> np.ndarray:
    """Per-position count of samples observed at every window member.

    ``finite``: ``(n_cpg, n_samples)`` boolean, finiteness of the underlying
    matrix. ``base_positions``: the reference CpG row for each position.
    ``offsets``: row offsets from each base position making up that position's
    window members -- ``[0, 1, ..., w-1]`` for the observed convention (window
    starts at the base position), ``[0, lag, 2*lag, ..., (w-1)*lag]`` for a null
    position at the given lag. The SAME function computes both, so the two
    conventions cannot silently drift apart -- callers only choose ``offsets``.

    Never materialises an ``(n_positions, n_samples, w)`` array: accumulates a
    boolean AND over ``w`` two-dimensional ``(chunk, n_samples)`` slices instead.
    """

    n_samples = finite.shape[1]
    n_positions = base_positions.size
    out = np.empty(n_positions, dtype=np.int64)
    for start in range(0, n_positions, chunk):
        stop = min(start + chunk, n_positions)
        block_base = base_positions[start:stop]
        acc = np.ones((stop - start, n_samples), dtype=bool)
        for offset in offsets:
            acc &= finite[block_base + offset]
        out[start:stop] = acc.sum(axis=1)
    return out
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py -k window_support -v
```

- [ ] **Step 5: Commit**

```bash
git add Python/eiger2/core.py tests/test_eiger2_core.py
git commit -m "feat: add _window_support, the shared observed/null support computation (Stage 1c)"
```

---

### Task 4: Generalize `segment()`/`region_area()` to accept a per-edge threshold array

**Files:**
- Modify: `Python/eiger2/core.py`
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- `segment`/`region_area`'s `threshold` parameter accepts `float | np.ndarray` of
  `statistic`'s shape. Scalar path unchanged. Consumed by Task 6.

- [ ] **Step 1: Write the failing tests**

```python
def test_segment_constant_array_threshold_matches_scalar():
    rng = np.random.default_rng(21)
    statistic = rng.normal(size=300)
    eligible = np.ones(300, dtype=bool)
    scalar_regions = segment(statistic, eligible, 0.5, min_cpgs=4, max_gap_edges=1)
    array_regions = segment(
        statistic, eligible, np.full(300, 0.5), min_cpgs=4, max_gap_edges=1,
    )
    np.testing.assert_array_equal(scalar_regions, array_regions)


def test_region_area_constant_array_threshold_matches_scalar():
    rng = np.random.default_rng(22)
    statistic = rng.normal(size=300)
    regions = np.array([[10, 25], [100, 130]])
    scalar_areas = region_area(statistic, regions, 0.3)
    array_areas = region_area(statistic, regions, np.full(300, 0.3))
    np.testing.assert_array_equal(scalar_areas, array_areas)


def test_segment_and_region_area_reject_mismatched_array_threshold_shape():
    statistic = np.zeros(10)
    with pytest.raises(ValueError):
        segment(statistic, np.ones(10, dtype=bool), np.zeros(5), min_cpgs=2)
    with pytest.raises(ValueError):
        region_area(statistic, np.array([[0, 2]]), np.zeros(5))


def test_region_area_array_threshold_uses_per_edge_values():
    statistic = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    threshold = np.array([0.5, 1.5, 1.0, 3.5, 0.0])
    regions = np.array([[0, 4]])
    areas = region_area(statistic, regions, threshold)
    expected = (1.0 - 0.5) + (2.0 - 1.5) + (3.0 - 1.0) + (4.0 - 3.5)
    assert areas[0] == pytest.approx(expected)
```

- [ ] **Step 2: Run and confirm failure**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py -k "segment_constant_array or region_area_constant_array or mismatched_array_threshold or region_area_array_threshold" -v
```

- [ ] **Step 3: Implement**

In `segment()` (core.py:1037), replace the top validation block:

```python
def segment(
    statistic: np.ndarray,
    eligible: np.ndarray,
    threshold: float | np.ndarray,
    *,
    min_cpgs: int,
    max_gap_edges: int = 0,
) -> np.ndarray:
    """Contiguous runs of passing edges, tolerating short internal gaps.

    ``threshold`` may be a scalar or a per-edge array matching ``statistic``'s
    shape (support-conditioned mode, Stage 1c). The scalar path is unchanged and
    stays bit-identical.

    Returns an ``(n_regions, 2)`` array of inclusive CpG index bounds. A run of
    ``E`` passing edges spans ``E + 1`` CpGs; ``min_cpgs`` counts CpGs.
    """

    statistic = np.asarray(statistic, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if statistic.shape != eligible.shape:
        raise ValueError("statistic and eligible must have the same shape")
    threshold_arr = np.asarray(threshold, dtype=float)
    if threshold_arr.shape not in ((), statistic.shape):
        raise ValueError("threshold must be a scalar or match statistic's shape")
    passing = np.isfinite(statistic) & (statistic >= threshold_arr) & eligible

    # ... rest of the function body is UNCHANGED from here on (the while-loop
    # segmentation logic doesn't reference `threshold` again).
```

In `region_area()` (core.py:1085):

```python
def region_area(
    statistic: np.ndarray, regions: np.ndarray, threshold: float | np.ndarray
) -> np.ndarray:
    """Summed excess of the statistic over the threshold, per region.

    ``threshold`` may be a scalar or a per-edge array matching ``statistic``'s
    shape (Stage 1c support-conditioned mode); the scalar path is unchanged.

    Area rewards both strength and length, which is the behaviour wanted for a
    region-level statistic and matches the ``bumphunter`` convention.
    """

    if regions.size == 0:
        return np.empty(0, dtype=float)
    values = np.asarray(statistic, dtype=float)
    threshold_arr = np.asarray(threshold, dtype=float)
    if threshold_arr.shape not in ((), values.shape):
        raise ValueError("threshold must be a scalar or match statistic's shape")
    is_array_threshold = threshold_arr.shape == values.shape
    areas = np.empty(len(regions), dtype=float)
    for position, (first, last) in enumerate(regions):
        first, last = int(first), int(last)
        block = values[first:last]
        block_threshold = threshold_arr[first:last] if is_array_threshold else threshold_arr
        finite_mask = np.isfinite(block)
        block = block[finite_mask]
        block_threshold = (
            block_threshold[finite_mask] if is_array_threshold else block_threshold
        )
        areas[position] = float(np.sum(block - block_threshold)) if block.size else 0.0
    return areas
```

- [ ] **Step 4: Run tests, confirm pass, then full regression**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py -v
conda activate eiger && pytest tests -q
```

Confirm the full suite is still 99 passed (this task is purely additive/generalizing;
every existing scalar-threshold call site and test must be unaffected).

- [ ] **Step 5: Commit**

```bash
git add Python/eiger2/core.py tests/test_eiger2_core.py
git commit -m "feat: generalize segment()/region_area() to accept a per-edge threshold array (Stage 1c)"
```

---

### Task 5: Support binning

**Files:**
- Modify: `Python/eiger2/core.py`
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- Produces: `_support_bin_edges(observed_support, n_bins, min_null_values,
  pooled_null_support) -> dict` with keys `bin_edges` (np.ndarray), `merges` (list of
  merge records), `collapsed` (bool). Consumed by Task 6.
- `_assign_support_bin(support, bin_edges) -> np.ndarray[int]` — vectorized bin
  index lookup, shared by observed and null callers.

- [ ] **Step 1: Write the failing tests**

```python
def test_support_bin_edges_are_observed_quantiles():
    rng = np.random.default_rng(31)
    observed_support = rng.integers(10, 100, size=2000)
    pooled_null_support = rng.integers(10, 100, size=20000)
    result = _support_bin_edges(observed_support, n_bins=4,
                                 min_null_values=1000,
                                 pooled_null_support=pooled_null_support)
    expected_edges = np.quantile(observed_support, [0.25, 0.5, 0.75])
    np.testing.assert_allclose(result["bin_edges"], expected_edges)
    assert not result["collapsed"]


def test_support_bins_merge_upward_when_null_underpopulated():
    # construct null support so that the top bin has very few values
    rng = np.random.default_rng(32)
    observed_support = rng.integers(0, 100, size=2000)
    pooled_null_support = np.concatenate([
        rng.integers(0, 75, size=5000),   # plenty in the lower 3 quartiles
        rng.integers(75, 100, size=10),   # almost nothing in the top quartile
    ])
    result = _support_bin_edges(observed_support, n_bins=4,
                                 min_null_values=1000,
                                 pooled_null_support=pooled_null_support)
    assert len(result["merges"]) >= 1
    # fewer final bins than requested, since the top one merged into its neighbour
    assert result["bin_edges"].size < 3


def test_support_bins_collapse_to_single_when_null_too_thin_everywhere():
    observed_support = np.arange(2000)
    pooled_null_support = np.arange(50)  # far too few null values overall
    result = _support_bin_edges(observed_support, n_bins=4,
                                 min_null_values=1000,
                                 pooled_null_support=pooled_null_support)
    assert result["collapsed"]
    assert result["bin_edges"].size == 0


def test_assign_support_bin_matches_searchsorted_semantics():
    bin_edges = np.array([10.0, 20.0, 30.0])
    support = np.array([5, 10, 15, 20, 25, 30, 35])
    bins = _assign_support_bin(support, bin_edges)
    # 4 bins: [-inf,10), [10,20), [20,30), [30,inf)
    assert bins.tolist() == [0, 1, 1, 2, 2, 3, 3]
```

- [ ] **Step 2: Run and confirm failure**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py -k "support_bin" -v
```

- [ ] **Step 3: Implement**

```python
def _assign_support_bin(support: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Bin index for each support value, given interior bin edges.

    ``bin_edges`` are the interior boundaries (``n_bins - 1`` of them); values
    below the first edge are bin 0, at/above the last edge are bin ``n_bins-1``.
    Shared by observed and null callers so the two are binned identically.
    """

    return np.searchsorted(bin_edges, support, side="right")


def _support_bin_edges(
    observed_support: np.ndarray,
    n_bins: int,
    min_null_values: int,
    pooled_null_support: np.ndarray,
) -> dict:
    """Quantile bin edges from observed support, merged upward until every bin
    has at least ``min_null_values`` pooled null values, per spec §1c.2.

    Returns ``{"bin_edges": np.ndarray, "merges": list[dict], "collapsed": bool}``.
    ``bin_edges`` has one fewer entry than the final bin count (interior edges
    only, consumed by ``_assign_support_bin``). ``collapsed=True`` means every
    candidate merge still left an underpopulated bin, so the caller must fall
    back to the unconditioned single threshold.
    """

    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    candidate_edges = np.quantile(observed_support, quantiles)
    merges: list[dict] = []

    edges = list(candidate_edges)
    while True:
        bins = _assign_support_bin(pooled_null_support, np.array(edges) if edges else np.array([]))
        counts = np.bincount(bins, minlength=len(edges) + 1)
        underpopulated = np.flatnonzero(counts < min_null_values)
        if underpopulated.size == 0:
            break
        if len(edges) == 0:
            return {"bin_edges": np.empty(0), "merges": merges, "collapsed": True}
        # merge the first underpopulated bin into its upward neighbour by
        # dropping the edge above it (merges bin b into bin b+1)
        merge_at = int(underpopulated[0])
        drop_edge_index = min(merge_at, len(edges) - 1)
        merges.append({
            "merged_bin": merge_at,
            "into_bin": merge_at + 1 if merge_at < len(edges) else merge_at - 1,
            "null_count_before_merge": int(counts[merge_at]),
        })
        del edges[drop_edge_index]

    return {"bin_edges": np.array(edges), "merges": merges, "collapsed": False}
```

Note: `min_null_values` here reuses `Eiger2Config.min_null_values` (already exists,
core.py:270, default 1,000) — Task 6 passes `config.min_null_values`, not a new field.

- [ ] **Step 4: Run tests, confirm pass**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py -k support_bin -v
```

- [ ] **Step 5: Commit**

```bash
git add Python/eiger2/core.py tests/test_eiger2_core.py
git commit -m "feat: add support-bin edge derivation with upward merging and collapse fallback (Stage 1c)"
```

---

### Task 6: Auto-enable config, per-bin thresholds, and `call_dmrs` wiring

**Files:**
- Modify: `Python/eiger2/core.py`
- Modify: `Python/eiger2/cli.py`
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- Consumes: `_window_support`, `_support_bin_edges`, `_assign_support_bin`,
  generalized `segment`/`region_area` (Tasks 3–5).
- Adds `Eiger2Config` fields: `support_binned_threshold: str = "auto"`,
  `support_bins: int = 4`, `support_dispersion_cut: float | None = None`.
- Produces per-run manifest fields: `support_dispersion`, `support_dispersion_cut`,
  `support_binning`, `support_bin_edges`, `support_bin_merges`,
  `support_thresholds`.

- [ ] **Step 1: Write the failing tests**

```python
def test_support_binned_threshold_off_reproduces_scalar_path_bit_for_bit():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=4,
        spacing=400, dmr_length=15, missing_rate=0.25, seed=41,
    )
    base_config = Eiger2Config(window=4, null_replicates=50, random_seed=41,
                                support_binned_threshold="off")
    regions_off, values_off, manifest_off = call_dmrs(matrix, locs, config=base_config)
    # compare against a config from BEFORE this task's changes semantically --
    # "off" must produce the same region set/areas/threshold as Stage 1b's output
    # for identical inputs. Since Stage 1b's code path IS the "off" path structurally,
    # this is really a self-consistency check: run "off" twice, confirm determinism,
    # and confirm statistic_threshold is a bare scalar (float), not array-derived.
    regions_off2, _, _ = call_dmrs(matrix, locs, config=base_config)
    pd.testing.assert_frame_equal(regions_off, regions_off2, check_exact=True)
    assert manifest_off["support_binning"] == "forced_off"


def test_support_dispersion_detector_computed_and_recorded():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=4,
        spacing=400, dmr_length=15, missing_rate=0.25, seed=42,
    )
    config = Eiger2Config(window=4, null_replicates=50, random_seed=42)
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    assert "support_dispersion" in manifest
    assert manifest["support_dispersion"] >= 0
    assert "support_dispersion_cut" in manifest
    assert manifest["support_binning"] in {
        "auto_on", "auto_off", "forced_on", "forced_off", "collapsed_to_single",
    }


def test_support_binned_threshold_on_produces_per_bin_thresholds():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=15, missing_rate=0.4, seed=43,
    )
    config = Eiger2Config(window=4, null_replicates=100, random_seed=43,
                           support_binned_threshold="on")
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    assert manifest["support_binning"] in {"forced_on", "collapsed_to_single"}
    if manifest["support_binning"] == "forced_on":
        assert len(manifest["support_thresholds"]) >= 2
        assert len(regions) >= 0  # ran without error
        if len(regions):
            assert (regions["region_threshold_mode"] == "support_binned").all()


def test_determinism_same_seed_same_bins_same_thresholds_same_regions():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=15, missing_rate=0.4, seed=44,
    )
    config = Eiger2Config(window=4, null_replicates=100, random_seed=44,
                           support_binned_threshold="on")
    regions1, _, manifest1 = call_dmrs(matrix, locs, config=config)
    regions2, _, manifest2 = call_dmrs(matrix, locs, config=config)
    pd.testing.assert_frame_equal(regions1, regions2, check_exact=True)
    assert manifest1["support_bin_edges"] == manifest2["support_bin_edges"]
```

- [ ] **Step 2: Run and confirm failure**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py -k "support_binned_threshold or support_dispersion or determinism_same_seed_same_bins" -v
```

- [ ] **Step 3: Add config fields**

In `Eiger2Config` (near `window`, core.py:80):

```python
    support_binned_threshold: str = "auto"
    """``"auto"`` | ``"on"`` | ``"off"``. When enabled, the null-quantile
    threshold is computed per support bin (samples observed at every CpG in the
    window) instead of once genome-wide, correcting the region-level FDR
    inflation that survives Stage 1b's null-window fix under structured
    missingness. ``"auto"`` enables it only when observed support varies enough
    to matter (``support_dispersion > support_dispersion_cut``). See
    docs/FDR.md, "Support conditioning" (added Stage 1e)."""

    support_bins: int = 4
    """Quantile bins for support-conditioned thresholding. Adjacent bins merge
    upward if the pooled null has fewer than ``min_null_values`` values in a
    bin; collapses to the unconditioned single threshold if merging cannot
    satisfy every bin."""

    support_dispersion_cut: float | None = None
    """Auto-enable threshold for ``support_binned_threshold="auto"``. ``None``
    uses the calibrated default (see docs/FDR.md, Stage 1e) -- calibrated in
    this stage's STOP report pending Stage 1e's full-scale confirmation."""
```

Add to `__post_init__`:

```python
        if self.support_binned_threshold not in {"auto", "on", "off"}:
            raise ValueError("support_binned_threshold must be 'auto', 'on', or 'off'")
        if self.support_bins < 2:
            raise ValueError("support_bins must be at least 2")
```

For the calibrated default: Task 8 determines the actual numeric value. This task
ships a provisional value (`0.5`) as the module-level constant
`_DEFAULT_SUPPORT_DISPERSION_CUT` (see Step 4b) — Task 8 changes only that
constant's value and its comment, nothing else.

- [ ] **Step 4a: Thread the per-replicate lag through `null_tracks`**

Verified: `null_tracks` (core.py:764–795) currently draws `lag = int(rng.integers(low,
high))` as a loop-local variable and discards it — the caller never sees it. Extend
its return type from `list[tuple[np.ndarray, np.ndarray]]` to
`list[tuple[np.ndarray, np.ndarray, int]]` (track, eligible, lag). This is an
internal function (not part of any stage's frozen public API), so extending its
tuple shape is acceptable, but **every call site must be updated in this same
commit**. Grep confirms exactly these:

- `null_threshold` (core.py:888— the line `pooled = np.concatenate([t[e] for t, e in
  tracks])`): change to `for t, e, _ in tracks`.
- `call_dmrs`'s non-permutation branch (`tracks = null_tracks(deflated, config,
  frame)`): the consuming loop changes per Step 4b below.
- Any direct test-file usage: `grep -rn "null_tracks(" tests/` before editing, and
  update every 2-tuple unpack found (`for track, eligible in ...`) to 3-tuple
  (`for track, eligible, lag in ...`, using `_` for `lag` where a test doesn't need
  it).

```python
def null_tracks(
    residuals: np.ndarray,
    config: Eiger2Config,
    locations: pd.DataFrame | None = None,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    # ... unchanged docstring and body up to the loop ...
    tracks: list[tuple[np.ndarray, np.ndarray, int]] = []
    for _ in range(config.null_replicates):
        lag = int(rng.integers(low, high))
        track = _lag_null_track(residuals, lag, config)
        if not track.size:
            continue
        eligible = _null_eligibility(chromosome, lag, config, track.size)
        if eligible.any():
            tracks.append((track, eligible, lag))
    return tracks
```

- [ ] **Step 4b: Implement the support/binning/threshold pipeline in `call_dmrs`**

Insert between the existing `threshold, null_diagnostics = null_threshold(...)` call
and `regions = segment(...)` in `call_dmrs` (core.py:1739–1761), and update the
existing null-track consuming loop (`for track, null_eligible in tracks:`) to unpack
three elements per Step 4a:

```python
    # --- Stage 1c: support-conditioned threshold ---
    n_cpg_total = deflated.shape[0]
    w = max(config.window, 1)
    finite_mask = np.isfinite(deflated)

    def _member_offsets(window: int, lag: int = 1) -> np.ndarray:
        """Window-member offsets from a base position, per spec §1c.1: the
        adjacent pair at w=1 (offsets [0, lag]), else [0, lag, 2*lag, ..., (w-1)*lag].
        `lag=1` gives the observed-track convention; any other `lag` gives a null
        position's member set, matching `_lag_null_track`'s own construction."""
        if window == 1:
            return np.array([0, lag])
        return np.arange(window) * lag

    obs_offsets = _member_offsets(w)
    obs_offset0 = 0 if w == 1 else w // 2 - 1
    obs_base = np.arange(statistic.size) - obs_offset0
    valid_obs = (obs_base >= 0) & (obs_base + obs_offsets.max() < n_cpg_total)
    observed_support = np.zeros(statistic.size, dtype=np.int64)
    observed_support[valid_obs] = _window_support(
        finite_mask, obs_base[valid_obs], obs_offsets,
    )

    support_dispersion = float("nan")
    usable_mask = eligible & np.isfinite(statistic)
    if np.any(usable_mask):
        usable_support = observed_support[usable_mask]
        median = np.median(usable_support)
        if median > 0:
            support_dispersion = float(
                (np.quantile(usable_support, 0.9) - np.quantile(usable_support, 0.1))
                / median
            )

    dispersion_cut = (
        config.support_dispersion_cut if config.support_dispersion_cut is not None
        else _DEFAULT_SUPPORT_DISPERSION_CUT  # calibrated in Task 8, module-level constant
    )

    if config.support_binned_threshold == "off":
        support_binning = "forced_off"
    elif config.support_binned_threshold == "on":
        support_binning = "forced_on"
    elif np.isfinite(support_dispersion) and support_dispersion > dispersion_cut:
        support_binning = "auto_on"
    else:
        support_binning = "auto_off"

    support_bin_edges: np.ndarray = np.empty(0)
    support_bin_merges: list = []
    support_thresholds: list = []
    per_edge_threshold: float | np.ndarray = threshold

    if support_binning in ("forced_on", "auto_on") and config.null_mode != "permutation":
        # Per null replicate, compute that replicate's OWN support (using its own
        # lag, mirroring _lag_null_track's member set exactly) and pool this
        # replicate's STATISTIC VALUES -- not its support values -- into the bin
        # each position's support falls into. threshold[b] is then the
        # null_quantile of all pooled statistic values across every replicate
        # whose position landed in bin b (spec §1c.3: "threshold[b] = null_quantile
        # of the pooled null values whose null-position support falls in bin b").
        null_support_parts = []
        null_value_by_replicate = []  # (support_array, value_array) per replicate
        for track, track_eligible, lag in tracks:
            null_offsets = _member_offsets(w, lag)
            null_base = np.arange(track.size)
            valid_null = null_base + null_offsets.max() < n_cpg_total
            support = np.zeros(track.size, dtype=np.int64)
            support[valid_null] = _window_support(
                finite_mask, null_base[valid_null], null_offsets,
            )
            usable = track_eligible & np.isfinite(track) & valid_null
            null_support_parts.append(support[usable])
            null_value_by_replicate.append((support[usable], track[usable]))

        pooled_null_support = (
            np.concatenate(null_support_parts) if null_support_parts else np.empty(0)
        )
        bin_result = _support_bin_edges(
            observed_support[usable_mask], config.support_bins,
            config.min_null_values, pooled_null_support,
        )
        if bin_result["collapsed"]:
            support_binning = "collapsed_to_single"
        else:
            support_bin_edges = bin_result["bin_edges"]
            support_bin_merges = bin_result["merges"]
            n_final_bins = support_bin_edges.size + 1
            observed_bins = _assign_support_bin(observed_support, support_bin_edges)

            per_bin_threshold = np.full(n_final_bins, np.nan)
            for b in range(n_final_bins):
                bin_values = np.concatenate([
                    values[_assign_support_bin(support, support_bin_edges) == b]
                    for support, values in null_value_by_replicate
                ]) if null_value_by_replicate else np.empty(0)
                if bin_values.size:
                    per_bin_threshold[b] = float(
                        np.quantile(bin_values, config.null_quantile)
                    )
            # a bin with zero pooled null values (shouldn't happen after merging
            # satisfied min_null_values, but guard defensively) falls back to the
            # unconditioned scalar threshold for that bin only
            per_bin_threshold = np.where(
                np.isfinite(per_bin_threshold), per_bin_threshold, threshold,
            )
            per_edge_threshold = per_bin_threshold[observed_bins]
            support_thresholds = per_bin_threshold.tolist()
```

Note the `config.null_mode != "permutation"` guard: support-conditioned thresholding
requires per-replicate lag-null tracks (each with their own member-set support);
`null_mode="permutation"` has no such per-replicate geometry (it permutes the
observed track's own values). Leave permutation mode on the unconditioned scalar
threshold in this stage — Stage 1d addresses permutation mode's own calibration
separately, and combining the two is out of this stage's scope. Record this as a
`support_binning` value of `"forced_off"` when `null_mode == "permutation"` even if
`support_binned_threshold != "off"`, with a note in `manifest["support_dispersion"]`'s
sibling diagnostics that binning was skipped for this reason (add
`manifest["support_binning_skipped_reason"] = "permutation_mode_not_supported"` in
that branch).

Add the module-level constant near `Eiger2Config` (Task 8 replaces its value, not
its existence):

```python
_DEFAULT_SUPPORT_DISPERSION_CUT = 0.5  # calibrated in Task 8 -- see that task's commit
```

After computing `per_edge_threshold`, use it (not the bare scalar `threshold`) for
`segment`/`region_area` on both the observed track and every null track's
segmentation, exactly mirroring the existing scalar flow. Store `manifest["support_dispersion"]
= support_dispersion`, `manifest["support_dispersion_cut"] = dispersion_cut`,
`manifest["support_binning"] = support_binning`,
`manifest["support_bin_edges"] = support_bin_edges.tolist()`,
`manifest["support_bin_merges"] = support_bin_merges`,
`manifest["support_thresholds"] = support_thresholds`.

Compute `region_support_median` per region (median of `observed_support` over the
region's edges) and `region_threshold_mode` (`"support_binned"` if
`support_binning` ended up `"auto_on"`/`"forced_on"`, else `"single"`) for the
row-building loop (Task 7 wires these into `_REGION_COLUMNS`).

- [ ] **Step 5: CLI flags**

In `cli.py`'s `build_parser`, near `--null-mode`:

```python
        p.add_argument("--support-binned-threshold", dest="support_binned_threshold",
                       choices=["auto", "on", "off"], default=None,
                       help="condition the null-quantile threshold on local "
                            "measurement support (samples observed at every "
                            "CpG in the window). 'auto' (default) enables it "
                            "only when support varies enough to matter.")
        p.add_argument("--support-bins", dest="support_bins", type=int, default=None)
```

- [ ] **Step 6: Run tests, confirm pass, then full regression**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py -v
conda activate eiger && pytest tests -q
```

`support_binned_threshold="off"` (or the `"forced_off"` path) must reproduce every
existing test's expected values exactly — if anything regresses here, the "off" path
has a bug, since it must be behaviourally identical to Stage 1b's code.

- [ ] **Step 7: Commit**

```bash
git add Python/eiger2/core.py Python/eiger2/cli.py tests/test_eiger2_core.py
git commit -m "feat: support-conditioned per-bin threshold, auto-enable detector, call_dmrs wiring (Stage 1c)"
```

---

### Task 7: Output schema — three new columns, `schema_version` bump

**Files:**
- Modify: `Python/eiger2/core.py`

**Interfaces:**
- Adds `region_support_median`, `region_threshold_mode` to `_REGION_COLUMNS`
  (`statistic_threshold` already exists, just changes meaning per §1c.5, no new
  column needed for it).

- [ ] **Step 1: Write the failing test**

```python
def test_schema_v22_has_support_columns_and_version():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=4,
        spacing=400, dmr_length=15, missing_rate=0.25, seed=51,
    )
    config = Eiger2Config(window=4, null_replicates=50, random_seed=51)
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    assert "region_support_median" in regions.columns
    assert "region_threshold_mode" in regions.columns
    assert manifest["schema_version"] == "2.2"
    if len(regions):
        assert regions["region_threshold_mode"].isin(["single", "support_binned"]).all()
```

- [ ] **Step 2: Run, confirm failure, then implement**

Add `"region_support_median"` and `"region_threshold_mode"` to `_REGION_COLUMNS`
immediately after `"statistic_threshold"` (core.py:1957). Wire both into the
row-building loop (from Task 6's per-region `region_support_median`/
`region_threshold_mode` computation) and into `"statistic_threshold": ` — this
becomes the region's mean per-edge threshold when `region_threshold_mode ==
"support_binned"`, else the existing scalar (unchanged). Change
`manifest["schema_version"] = "2.1"` to `"2.2"` (search both `call_dmrs` and any
other place Stage 1/1b set it — should be exactly one place in `core.py`).

- [ ] **Step 3: Run tests, confirm pass, full regression**

```bash
conda activate eiger && pytest tests -q
```

- [ ] **Step 4: Commit**

```bash
git add Python/eiger2/core.py tests/test_eiger2_core.py
git commit -m "feat: add region_support_median/region_threshold_mode columns, bump schema_version to 2.2 (Stage 1c)"
```

---

### Task 8: Calibrate `support_dispersion_cut`

**Files:**
- Modify: `Python/eiger2/core.py` (replace the Task 6 placeholder constant)
- No new source logic — this task runs real data and picks a number.

**Interfaces:**
- Consumes: Task 6's `support_dispersion` computation.

- [ ] **Step 1: Measure `support_dispersion` per scenario**

Using the 5 standard scenarios (`clean`, `nuisance`, `missingness`, `small_n`,
`high_noise`, matching `benchmarking/benchmark_suite.py::SCENARIOS`) at, e.g., 10
seeds each (bounded scale — Stage 1e's full grid will confirm this at 100 seeds),
run `call_dmrs` with `support_binned_threshold="off"` (so `support_dispersion` is
still computed and recorded, but binning doesn't yet activate) and record
`manifest["support_dispersion"]` per run.

- [ ] **Step 2: Choose the cut**

Per spec §1c.4: "choose the smallest cut that leaves clean, nuisance, small_n and
high_noise unbinned while enabling missingness." From the measured per-scenario
dispersion distributions, pick the smallest value strictly greater than the highest
dispersion seen among the four non-missingness scenarios and strictly less than the
lowest dispersion seen for missingness (if such a value exists — if the
distributions overlap, pick the value that best separates the two groups by
whatever margin exists, and report the overlap honestly rather than picking a cut
that doesn't actually work).

- [ ] **Step 3: Update the code**

Update the module-level `_DEFAULT_SUPPORT_DISPERSION_CUT = 0.5` constant (Task 6,
core.py) to the chosen literal, updating its trailing comment to state the
calibration basis (e.g. `# calibrated: separates missingness (median X) from
clean/nuisance/small_n/high_noise (max Y) at N seeds/scenario — see STOP 1c report`).
Also update `Eiger2Config.support_dispersion_cut`'s docstring if it references a
specific value.

- [ ] **Step 4: Verify auto-enable fires correctly**

```python
def test_auto_enable_fires_only_on_missingness_at_calibrated_cut():
    # run all 5 scenarios at a few seeds with support_binned_threshold="auto"
    # (the calibrated default), assert manifest["support_binning"] == "auto_on"
    # for missingness and "auto_off" for the other four, at every tested seed.
```

Write this test into `tests/test_eiger2_core.py` for real, using the calibrated cut
now in place — this is spec §1c.7 test item 6.

- [ ] **Step 5: Commit**

```bash
git add Python/eiger2/core.py tests/test_eiger2_core.py
git commit -m "feat: calibrate support_dispersion_cut default from measured per-scenario dispersion (Stage 1c)"
```

Record the full calibration table (scenario, seeds, measured dispersion range,
chosen cut, margin) for the STOP 1c report — do not just state the final number.

---

### Task 9: Region-level exposure diagnostic and its gate

**Files:**
- Modify: `Python/eiger2/core.py`
- Modify: `Python/eiger2/cli.py` (write `region_rate_by_bin.csv` alongside the
  regular outputs, if the benchmark harness is the natural place — check whether
  this belongs in `cli.py`'s `command_call` or is benchmark-only; spec §1c.6 says
  "Write to the manifest as `region_rate_by_support_bin` and to
  `benchmarking/.../region_rate_by_bin.csv`" — the manifest write belongs in
  `core.py`/`call_dmrs`; the CSV write is benchmark-specific, so it likely belongs
  in a new or existing benchmark experiment, not every `eiger2 call` — read
  spec §1c.6 again and decide; if ambiguous, put the CSV write in
  `benchmarking/benchmark_suite.py` as part of a benchmark run rather than every
  production call, and note this decision in your report).
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- Produces `manifest["region_rate_by_support_bin"]`: per-bin `{observed_rate,
  null_rate, ratio}`, keyed by bin index, computed whenever `support_binning` is
  `"auto_on"` or `"forced_on"`.

- [ ] **Step 1: Write the failing test**

```python
def test_region_rate_by_support_bin_populated_when_binning_active():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=15, missing_rate=0.4, seed=61,
    )
    config = Eiger2Config(window=4, null_replicates=100, random_seed=61,
                           support_binned_threshold="on")
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    if manifest["support_binning"] == "forced_on":
        assert "region_rate_by_support_bin" in manifest
        for bin_stats in manifest["region_rate_by_support_bin"].values():
            assert "observed_rate" in bin_stats
            assert "null_rate" in bin_stats
            assert "ratio" in bin_stats
```

- [ ] **Step 2: Implement**

Per bin `b`: `observed_regions_in_bin` = count of called regions whose
`region_threshold_mode == "support_binned"` and `_assign_support_bin(
region_support_median, bin_edges) == b`; `observed_eligible_edges_in_bin` = count
of eligible finite observed edges whose `observed_support` falls in bin `b`;
`observed_rate = 10_000 * observed_regions_in_bin / observed_eligible_edges_in_bin`.
Symmetrically for null: pool null regions across all replicates, bin each by its own
support-median, divide by pooled null eligible-edge count in that bin, ×10,000.
`ratio = observed_rate / null_rate` (guard divide-by-zero: report `null` explicitly
in the dict rather than raising, per the "no exact zero presented as certainty"
spirit — if `null_rate == 0` and `observed_rate > 0`, record `ratio: None` and a
`"note"` explaining the null had zero regions in that bin).

Add this computation to `call_dmrs`, gated on `support_binning in ("auto_on",
"forced_on")`, storing the result in `manifest["region_rate_by_support_bin"]`.

- [ ] **Step 3: Run tests, confirm pass, full regression**

```bash
conda activate eiger && pytest tests -q
```

- [ ] **Step 4: Check the gate on real data**

Using a missingness-scenario run at reasonable scale (several seeds,
`support_binned_threshold="on"`), check whether any bin's `ratio` exceeds 2.0.
**If no bin exceeds 2.0 in this stage's own testing:** do not implement the §1c.6
escalation (stratifying region-FDR exposure counts by support bin) — record clearly
in the STOP 1c report that the gate was checked and did not fire, per this plan's
Global Constraints. **If a bin does exceed 2.0:** stop this task, do not proceed to
Task 10, and report the finding — the escalation implementation becomes a new task,
inserted here, following spec §1c.6's exact instructions (stratify `region_fdr_result`
by support bin in addition to length, with the ≥50-null-regions-per-joint-stratum
fallback to length-only stratification, recorded when triggered).

- [ ] **Step 5: Commit**

```bash
git add Python/eiger2/core.py Python/eiger2/cli.py tests/test_eiger2_core.py
git commit -m "feat: add region_rate_by_support_bin diagnostic and its 2.0 gate check (Stage 1c)"
```

---

### Task 10: Escalation — stratify region-FDR exposure counts by support bin (spec §1c.6, triggered)

**Context — why this task exists:** Task 9's real-data gate check fired. On DMR-free
`missingness`/`small_n` data (20 seeds, `null_replicates=500`, `support_binned_threshold
="on"`), every evaluable support bin's observed/null region-formation ratio was
283x–811x, one bin infinite — confirmed not a small-count artifact (pooled null
exposure was 11–51 million eligible edges per bin). Per this plan's own Global
Constraints ("only build the escalation path if the gate actually fires in this stage's
own validation... report clearly either way") and spec §1c.6's exact instructions, the
escalation is now required. This is not a redesign of the per-edge threshold mechanism
(Tasks 3–8 stand as-is) — it is a second, independent correction at the FDR-estimation
layer: `region_fdr_result` currently pools null regions within a stratum keyed only by
length (`length_stratum()`); it must additionally key by support bin, with spec's exact
fallback when a joint stratum is too thin.

**Files:**
- Modify: `Python/eiger2/core.py` — `region_fdr_result` (core.py:1414) and its call
  site inside `call_dmrs` (core.py:2232, immediately after Task 9's `region_rate_by_
  support_bin` block)
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- `region_fdr_result` gains two new optional parameters, parallel in shape/ordering to
  the existing `observed_lengths`/`null_length_sets`: `observed_support_bins: np.ndarray
  | None = None`, `null_support_bin_sets: Sequence[np.ndarray] | None = None`. Leaving
  both at `None` (the default) must reproduce the exact pre-existing behavior bit for
  bit — this is what preserves bit-identity when support binning is off, and what every
  EXISTING caller of `region_fdr_result` (there are several — `qvalue_result_from_track`
  at core.py:1619, `region_fdr` at core.py:1600, and any test not touched by this task)
  continues to get automatically, unmodified.
- New module constant, placed near `length_stratum()` (core.py:1369):
  `_MIN_NULL_REGIONS_PER_JOINT_STRATUM = 50` (spec's own number, not user-configurable —
  spec does not ask for this to be a config field).
- `RegionFdrResult.diagnostics` (the dict already returned) gains three new keys:
  `"support_stratified"` (bool — whether support-bin stratification was attempted at
  all, i.e. whether `observed_support_bins` was not `None`), `"support_stratification_
  fallback_strata"` (`list[str]`, the joint `"<length_stratum>|sbin<b>"` labels that
  fell back to length-only pooling because their pooled null-region count was below
  `_MIN_NULL_REGIONS_PER_JOINT_STRATUM`), `"min_null_regions_per_joint_stratum"` (int,
  echoes the constant, for traceability in the STOP report).
- **Explicitly out of scope:** `local_fdr` (core.py, called separately inside `call_dmrs`
  a few lines after `region_fdr_result`, for `config.compute_posterior`) is NOT touched.
  Spec §1c.6 names `region_fdr_result` specifically ("stratify the region-FDR exposure
  counts by bin... run `region_fdr_result` per support bin"), not the posterior/local-FDR
  path. Do not extend `local_fdr` under this task even though its signature looks similar
  — if you think it should also be extended, note that as an observation in your report,
  do not do it speculatively.
- **Consumes from Task 9** (already committed, core.py inside `call_dmrs`, ~2085–2231):
  `null_regions_by_replicate: list[np.ndarray]` (one `segment()` output array per null
  replicate, each row `[first, last]` edge indices), `null_full_support: list[np.ndarray]`
  (Task 6 — one full-length support array per replicate), `region_bins` (Task 9's local
  variable inside its `if support_binning in ("auto_on", "forced_on"):` block — each
  OBSERVED region's support-bin index, computed via `_assign_support_bin(manifest[
  "region_support_median_descriptive"], support_bin_edges)`), `support_bin_edges`
  (Task 5/6), `support_binning` (Task 6, one of `"auto_on"|"auto_off"|"forced_on"|
  "forced_off"|"collapsed_to_single"`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eiger2_core.py`, near the other `region_fdr_result` tests (~line
900–1090):

```python
def test_region_fdr_result_support_stratification_diagnostics():
    rng = np.random.default_rng(77)
    # Two observed regions, same length stratum ("6-9"), different support bins.
    observed_areas = np.array([12.0, 9.0])
    observed_lengths = np.array([6, 6])
    observed_support_bins = np.array([0, 1])

    # 3 null replicates. Support bin 0 gets 20 null regions per replicate (60
    # pooled, >= _MIN_NULL_REGIONS_PER_JOINT_STRATUM -> joint stratum "6-9|sbin0"
    # stays joint). Support bin 1 gets 1 null region per replicate (3 pooled,
    # < 50 -> falls back to length-only pooling for that region).
    null_area_sets = []
    null_length_sets = []
    null_support_bin_sets = []
    for _ in range(3):
        bin0_areas = rng.uniform(1.0, 8.0, size=20)
        bin1_areas = rng.uniform(1.0, 8.0, size=1)
        null_area_sets.append(np.concatenate([bin0_areas, bin1_areas]))
        null_length_sets.append(np.full(21, 6))
        null_support_bin_sets.append(
            np.concatenate([np.zeros(20, dtype=int), np.ones(1, dtype=int)])
        )

    result = region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=1000, null_edges=3000,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
        observed_support_bins=observed_support_bins,
        null_support_bin_sets=null_support_bin_sets,
    )

    assert result.diagnostics["support_stratified"] is True
    assert result.diagnostics["min_null_regions_per_joint_stratum"] == 50
    assert result.diagnostics["support_stratification_fallback_strata"] == ["6-9|sbin1"]
    cutoffs = result.diagnostics["fdr_interval_cutoffs_by_stratum"]
    assert "6-9|sbin0" in cutoffs  # well-populated joint stratum stayed joint
    assert "6-9" in cutoffs        # region 1 fell back to the length-only label
    assert "6-9|sbin1" not in cutoffs  # never appears as a final stratum itself
    assert np.all(np.isfinite(result.qvalue))
    assert result.qvalue.shape == (2,)

    # None (the default) must reproduce pre-existing behavior exactly: no new
    # diagnostics keys, same shape.
    unstratified = region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=1000, null_edges=3000,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    )
    assert unstratified.diagnostics["support_stratified"] is False
    assert unstratified.diagnostics["support_stratification_fallback_strata"] == []
```

Note: this test deliberately does NOT assert exact numeric equality of `qvalue`/
`tail_fdr` between the support-stratified and length-only-only calls — the q-value
construction is a reverse-cumulative-minimum monotonized within each stratum's full
observed set (see `region_fdr_result`'s docstring and `docs/FDR.md`), so removing
region 0 from region 1's grouping (which is exactly what putting region 0 into its own
`"6-9|sbin0"` stratum does) can legitimately change region 1's monotonized qvalue even
though its underlying pooled null set is unchanged. Do not add such an equality
assertion without actually running the code and confirming it holds — the diagnostics/
structural assertions above are what this step commits to.

- [ ] **Step 2: Run test to verify it fails**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py::test_region_fdr_result_support_stratification_diagnostics -v
```

Expected: FAIL — `region_fdr_result() got an unexpected keyword argument 'observed_support_bins'`.

- [ ] **Step 3: Implement the stratification change in `region_fdr_result`**

Replace the existing stratification block (core.py, from `stratified = observed_lengths
is not None and null_length_sets is not None` through `strata = np.unique
(observed_strata)`, i.e. lines ~1478–1495) with:

```python
    stratified = observed_lengths is not None and null_length_sets is not None
    if stratified:
        observed_lengths = np.asarray(observed_lengths, dtype=int)
        null_length_sets = [np.asarray(a, dtype=int) for a in null_length_sets]
        observed_length_labels = np.array(
            [length_stratum(int(n)) for n in observed_lengths], dtype=object
        )
        null_length_labels_by_replicate = [
            np.array([length_stratum(int(n)) for n in lengths], dtype=object)
            for lengths in null_length_sets
        ]
    else:
        observed_length_labels = np.full(m, "_pooled", dtype=object)
        null_length_labels_by_replicate = [
            np.full(areas.size, "_pooled", dtype=object) for areas in null_area_sets
        ]

    support_stratified = (
        stratified
        and observed_support_bins is not None
        and null_support_bin_sets is not None
    )
    support_fallback_joint_strata: list[str] = []
    null_joint_labels_by_replicate = null_length_labels_by_replicate
    if support_stratified:
        observed_support_bins = np.asarray(observed_support_bins, dtype=int)
        null_support_bin_sets = [
            np.asarray(a, dtype=int) for a in null_support_bin_sets
        ]
        observed_joint_labels = np.array(
            [
                f"{length_stratum(int(n))}|sbin{int(b)}"
                for n, b in zip(observed_lengths, observed_support_bins)
            ],
            dtype=object,
        )
        null_joint_labels_by_replicate = [
            np.array(
                [
                    f"{length_stratum(int(n))}|sbin{int(b)}"
                    for n, b in zip(lengths, bins)
                ],
                dtype=object,
            )
            for lengths, bins in zip(
                null_length_sets, null_support_bin_sets, strict=True
            )
        ]
        joint_null_counts: dict[str, int] = {}
        for labels in null_joint_labels_by_replicate:
            if labels.size == 0:
                continue
            values, counts = np.unique(labels, return_counts=True)
            for value, count in zip(values, counts):
                joint_null_counts[str(value)] = (
                    joint_null_counts.get(str(value), 0) + int(count)
                )

        observed_strata = np.empty(m, dtype=object)
        for i in range(m):
            joint_label = str(observed_joint_labels[i])
            if joint_null_counts.get(joint_label, 0) >= _MIN_NULL_REGIONS_PER_JOINT_STRATUM:
                observed_strata[i] = joint_label
            else:
                observed_strata[i] = observed_length_labels[i]
                support_fallback_joint_strata.append(joint_label)
    else:
        observed_strata = observed_length_labels

    strata = np.unique(observed_strata)
```

Then, in Pass B's loop (the `for stratum in strata:` block, ~line 1514), replace the
line `per_replicate = [null_area_sets[b][null_strata_by_replicate[b] == stratum] for b
in range(replicates)]` with:

```python
    for stratum in strata:
        rows = np.flatnonzero(observed_strata == stratum)
        is_joint_stratum = support_stratified and "|sbin" in str(stratum)
        labels_by_replicate = (
            null_joint_labels_by_replicate
            if is_joint_stratum
            else null_length_labels_by_replicate
        )
        per_replicate = [
            null_area_sets[b][labels_by_replicate[b] == stratum]
            for b in range(replicates)
        ]
```

(the rest of the loop body — `pooled = np.concatenate(...)`, the `pooled_fallback_
strata` empty-pool safety net, `_fdr_from_areas_result(...)`, the five `[rows] = ...`
assignments — is unchanged; only the `per_replicate` line and its immediate
`labels_by_replicate` selection change).

Update the function signature to add the two new parameters (after `null_length_sets`,
before `confidence_level`) and update the `diagnostics` dict literal near the end of
the function (~line 1549) to add the three new keys:

```python
    diagnostics = {
        "fdr_interval_bonferroni_m": m_bonferroni,
        "fdr_interval_cutoffs_by_stratum": cutoffs_by_stratum,
        "fdr_interval_family": "per_stratum",
        "pooled_fallback_strata": pooled_fallback_strata,
        "null_count_dispersion_by_stratum": overdispersion_by_stratum,
        "null_count_overdispersion_warning": overdispersion_warning,
        "scale": scale,
        "support_stratified": support_stratified,
        "support_stratification_fallback_strata": sorted(set(support_fallback_joint_strata)),
        "min_null_regions_per_joint_stratum": _MIN_NULL_REGIONS_PER_JOINT_STRATUM,
    }
```

Also update the function's docstring to describe the new optional joint stratification
(one or two sentences — state the fallback rule in words, matching spec §1c.6).

- [ ] **Step 4: Run the new test, confirm pass**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py::test_region_fdr_result_support_stratification_diagnostics -v
```

- [ ] **Step 5: Wire the new parameters into `call_dmrs`**

Immediately before the existing `fdr_result = region_fdr_result(...)` call (core.py:2232,
right after Task 9's `if support_binning in ("auto_on", "forced_on"):` block closes),
add:

```python
    observed_support_bins_for_fdr = None
    null_support_bin_sets_for_fdr = None
```

Then, inside Task 9's existing `if support_binning in ("auto_on", "forced_on"):` block
(the one that already computes `region_bins`), add — after `region_bins` is computed,
before that block ends — and gated additionally on `config.length_stratified_fdr` (joint
stratification without length stratification is not spec'd and should not activate):

```python
        if config.length_stratified_fdr:
            observed_support_bins_for_fdr = region_bins
            null_region_support_medians_by_replicate = [
                (
                    np.array(
                        [
                            float(np.median(support[first:last]))
                            for first, last in null_regions
                        ],
                        dtype=float,
                    )
                    if null_regions.size
                    else np.empty(0, dtype=float)
                )
                for null_regions, support in zip(
                    null_regions_by_replicate, null_full_support, strict=True
                )
            ]
            null_support_bin_sets_for_fdr = [
                (
                    _assign_support_bin(medians, support_bin_edges)
                    if medians.size
                    else np.empty(0, dtype=int)
                )
                for medians in null_region_support_medians_by_replicate
            ]
```

Then change the `region_fdr_result(...)` call to add the two new arguments:

```python
    fdr_result = region_fdr_result(
        observed_areas,
        null_areas,
        observed_edges=int(np.sum(eligible & np.isfinite(statistic))),
        null_edges=max(null_edge_total, 1),
        observed_lengths=observed_lengths if config.length_stratified_fdr else None,
        null_length_sets=null_lengths if config.length_stratified_fdr else None,
        observed_support_bins=observed_support_bins_for_fdr,
        null_support_bin_sets=null_support_bin_sets_for_fdr,
    )
```

- [ ] **Step 6: Integration test — support stratification activates through `call_dmrs`, and `off` stays bit-identical**

```python
def test_call_dmrs_support_stratified_fdr_activates_when_binning_on():
    matrix, locs, _ = simulate(
        n_samples=80, strata=[], replicates=0,
        spacing=19600, missing_rate=0.25, seed=71,
    )
    config = Eiger2Config(window=4, null_replicates=200, random_seed=71,
                           support_binned_threshold="on")
    regions, _, manifest = call_dmrs(matrix, locs, config=config)
    if manifest["support_binning"] == "forced_on" and len(regions):
        assert manifest["fdr"]["support_stratified"] is True
    # "off" must be unaffected: no support stratification, same diagnostics shape
    # as before this task.
    config_off = Eiger2Config(window=4, null_replicates=200, random_seed=71,
                               support_binned_threshold="off")
    regions_off, _, manifest_off = call_dmrs(matrix, locs, config=config_off)
    if len(regions_off):
        assert manifest_off["fdr"]["support_stratified"] is False
        assert manifest_off["fdr"]["support_stratification_fallback_strata"] == []
```

- [ ] **Step 7: Run full regression, lint, commit**

```bash
conda activate eiger && pytest tests -q
conda activate eiger && ruff check Python/eiger2 && ruff check tests
git add Python/eiger2/core.py tests/test_eiger2_core.py
git commit -m "feat: stratify region-FDR exposure counts by support bin (Stage 1c escalation, spec §1c.6)"
```

Report, for the STOP 1c report: confirm the gate that triggered this task (Task 9's
numbers), confirm this escalation is now live, and confirm — this is the one thing
that must not regress — that `support_binned_threshold="off"` still reproduces Stage
1b output bit-for-bit end to end (re-run Task 6/7's existing bit-identity test as part
of your full-suite run and confirm it's still passing, don't just assume it).

---

### Task 11: Remaining §1c.7 tests, and the §1c.8 target validation

**Files:**
- Modify: `tests/test_eiger2_core.py`

**Interfaces:** none new — validation and remaining test coverage.

- [ ] **Step 1: Fill any remaining §1c.7 gaps**

Cross-check spec §1c.7's 10 items against what Tasks 3–10 already cover:
1. constant-array-threshold-matches-scalar (Task 4) ✓
2. support computed identically for observed/null (Task 3, via the shared function
   — add an explicit end-to-end test if Task 3's unit tests don't already prove this
   at the `call_dmrs` integration level: hand-build a small matrix, confirm
   `observed_support` and a null replicate's support at the SAME window-member
   pattern produce the same value on identical underlying finiteness) ✓/verify
3. bin edges from observed applied unchanged to null (Task 5's tests + Task 6's
   wiring — add an explicit assertion that the null threshold computation uses
   EXACTLY `support_bin_edges` from the observed derivation, not a re-quantiled
   value) ✓/verify
4. bin merging fires and is recorded (Task 5) ✓
5. single-bin collapse reproduces unconditioned path exactly (Task 5's collapse test
   + an integration test: force collapse via extreme parameters, confirm the
   resulting regions match a `support_binned_threshold="off"` run on the same
   input) — write this integration test now if missing
6. auto fires correctly (Task 8) ✓
7. `off` reproduces Stage 1b output bit-for-bit (Task 6's `test_support_binned_
   threshold_off_...` test — strengthen if it doesn't yet directly compare against
   a Stage-1b-era fixture or an equivalent hand-computed expectation, not just
   self-consistency)
8. exchangeability ratio ≤1.25 under missingness with binning on (write this
   explicitly: run `null_exchangeability` — Stage 1b's function — against a
   binning-on missingness run, confirm the ratio; apply the Global Constraints'
   median-across-seeds policy if run across multiple seeds)
9. `region_rate_by_support_bin` populated and gate raises/escalates as specified
   (Task 9 populated it, Task 10 implemented the escalation the gate triggered) ✓ —
   also confirm `manifest["fdr"]["support_stratified"]` is `True` and `"support_
   stratification_fallback_strata"` is populated on a real missingness run with
   binning on, since Task 10's own tests only exercise `region_fdr_result` directly
   and one small `call_dmrs` integration case, not a full realistic-scale run
10. determinism (Task 6) ✓

Write any test identified as missing/weak above, for real, with real assertions.

- [ ] **Step 2: §1c.8 target validation**

Run the planted-signal missingness benchmark (reuse `heuristic_sweep`'s or a
dedicated script's signal-planting convention, matching Stage 1b's Task 5
methodology) at `missing_rate=0.25` (the value Stage 1b's Table 3 used for its
interpretable before/after comparison — 0.25→0.05 is exactly what §1c.8 asks to
confirm), `support_binned_threshold="auto"`, several seeds. Measure realised FDR and
recall. **Target: FDR falls to at or below 0.05 within simulation uncertainty,
without recall dropping more than 5 points below Stage 1b's 0.667 figure.** If the
target is met only via a recall collapse, this is spec's own named failure mode —
stop and report, do not treat it as a pass.

- [ ] **Step 3: Run full regression, commit**

```bash
conda activate eiger && pytest tests -q
conda activate eiger && ruff check Python/eiger2 && ruff check tests && ruff check benchmarking
git add tests/test_eiger2_core.py
git commit -m "test: complete spec §1c.7 test coverage and validate the §1c.8 FDR/recall target (Stage 1c)"
```

---

### Task 12: STOP 1c report

**Files:** none (report only).

- [ ] **Step 1: Assemble the §0.6 report**

Per spec: "Report per §0.6, including the calibration table for
`support_dispersion_cut`, the per-bin region-rate table, and the FDR/recall
comparison against Stage 1b." Include:

1. Change summary.
2. `git diff --stat` for this stage's commits.
3. The calibration table (Task 8): scenarios, seeds, measured dispersion ranges,
   chosen cut, margin.
4. Every test command and output tail.
5. The per-bin region-rate table (Task 9) and whether the 2.0 gate fired — it did
   (283x–811x, one bin infinite) — and the escalation this triggered (Task 10):
   support-bin stratification of `region_fdr_result`'s exposure counts, with the
   spec's <50-pooled-null-regions fallback to length-only stratification.
6. Runtime/memory if materially different from Stage 1b.
7. Backward-compatibility: `schema_version` now `"2.2"`, `statistic_threshold`'s
   meaning change documented, three new columns named, `region_fdr_result`'s two new
   optional parameters (default `None`, fully backward compatible with every other
   caller).
8. Remaining limitations: whether §1c.8's target was met; explicitly flag that this
   stage's calibration and benchmark validation ran at bounded scale (not the full
   spec-mandated 100-seed grid, which is Stage 1e's job), mirroring Stage 1b's
   precedent; note the open question Task 9's implementer flagged and Task 10
   resolved empirically — whether length-stratification alone was already absorbing
   some of the support-driven region-formation imbalance — state what Task 10's own
   validation run(s) actually showed about this, not just what was hypothesized.
9. `git diff --check` and final `git status --short`.
10. No claim stronger than the measured evidence.

- [ ] **Step 2: Halt**

Per spec: do not begin Stage 1d. Produce the report and stop for review.
