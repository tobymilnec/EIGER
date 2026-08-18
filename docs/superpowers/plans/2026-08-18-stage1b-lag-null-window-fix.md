# Stage 1b — correct the lag-null window assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the single indexing defect in `core._lag_null_track` that assembles the
null window from the wrong CpGs (9 distinct CpGs at 3 base positions instead of the 6
pairwise statistics among one set of `w` CpGs), add a permanent null-exchangeability
gate as both a benchmark experiment and a per-run manifest diagnostic, and reproduce
the diagnosis's measured before/after tables to confirm the fix in this environment.

**Architecture:** One function replacement (`_lag_null_track`), reusing the existing
`_lag_statistic`/`_pair_statistic` machinery unchanged. A new shared function,
`null_exchangeability`, computes the split-half calibration/held-out ratio once per
`call_dmrs` run (reusing the null tracks `null_threshold` already generates — no
second null-track generation) and is reused by a new `benchmarking/benchmark_suite.py`
experiment for the standalone gate report.

**Tech Stack:** Python 3.10+, NumPy, pandas, pytest, ruff — no new dependencies.

**Spec:** `EIGER_STAGE1BCD_IMPLEMENTATION_SPEC.md` (repo root of `Github_beta_Versions`),
STAGE 1b (§1b.0–§1b.5), plus §1 ("Revised backward-compatibility contract") and the
shared engineering contract in `EIGER_WP1-3_IMPLEMENTATION_SPEC.md` §0 (unchanged
except where §1 of the new spec revises it). Root-cause derivation:
`EIGER_NULL_CALIBRATION_DIAGNOSIS.md`.

## Global Constraints

- **Continue on `fix/region-fdr`, same worktree `EIGER-wt-fdr`.** Do not open a new
  worktree. Do not create `exp/multiscale` or `exp/programs`.
- `_window_from_lags` is correct for the **observed** track and must not be modified.
  Every change is confined to `_lag_null_track`.
- The 50% finite-pair rule and the per-pair `min_paired_samples` floor inside
  `_pair_statistic` are preserved verbatim — do not touch `_pair_statistic`.
- `_null_eligibility` already checks the correct member set (chromosome membership at
  offsets `step * lag` for `step = 1 … span−1`) — verify this by test, do not change
  the function.
- The fix must be a **bit-identical no-op at `window ∈ {1, 2}`** — `w=1` was already
  exactly calibrated before the fix and this is the single strongest regression check
  available. Any deviation at `w ≤ 2` is a bug in the replacement, not an acceptable
  side effect.
- **Revised invariant contract (spec §1, supersedes WP1–3 spec §0.3.3 for this and
  later stages):** `statistic_threshold`, `region_area`, the region set and its
  boundaries, every FDR column, `region_evidence_score`, `significant`, and all
  manifest counts **change by design** in this stage. What stays byte-identical to
  v2.0.0: the observed edge statistic track (`edge_statistic()`), the eligibility mask
  (`eligible_edges()`), the latent basis and deflated residuals
  (`estimate_latent_basis`, `deflate`), and `characterise_region()` output for a given
  region span. Do not write a test asserting `region_area`/`statistic_threshold`
  unchanged — that assertion is now wrong by design.
- Bump `schema_version` to `"2.2"` only in Stage 1c (which adds columns and changes
  `statistic_threshold`'s meaning) — **do not bump it in Stage 1b.** Stage 1b changes
  values, not the schema shape.
- Determinism: any new seeded split (calibration/held-out halves) must use a named,
  documented, deterministic derivation from `config.random_seed` — never call order.
- Docs/tests ship in the same commit as the code they cover.
- Reproduce spec §1b.4's tables to within Monte Carlo noise before proceeding past
  Task 4. If a reproduction differs qualitatively (wrong sign, order-of-magnitude
  mismatch), **stop and report** — do not proceed on an environment that doesn't match
  the diagnosis.

---

## Verified baseline (read 2026-08-18, do not re-derive)

- Worktree `EIGER-wt-fdr` is clean, HEAD `6965fd7` (Stage 1's STOP report, on top of
  `ba046c9` which is what this spec calls "Stage 1 complete" — no code differs between
  the two, `6965fd7` is docs-only).
- `_lag_null_track` (core.py:686–712) confirmed to call
  `_window_from_lags(lag_statistics, config.window)` at line 712 — exactly the defect
  described: reuses the observed-track assembly function on statistics gathered at
  separations `1·lag … (w−1)·lag`, which `_window_from_lags` misinterprets as
  consecutive-index adjacency.
- `_window_from_lags` (core.py:614–648): unchanged reference, do not touch.
- `_lag_statistic` (core.py:594–611): `stat(CpG i, CpG i+lag)` for every valid `i`,
  chunked. Reused unchanged by the replacement.
- `_null_eligibility` (core.py:715–738): already loops `for step in range(1, span):
  offset = step * lag` checking chromosome membership at exactly the member-set
  offsets `{0, lag, 2·lag, ..., (w-1)·lag}` — confirmed correct as-is, per spec.
- `null_tracks` (core.py:741– ): seeds via `np.random.default_rng(config.random_seed +
  977)`, draws one lag per replicate, calls `_lag_null_track` then `_null_eligibility`.
  Untouched by this stage except for whatever `null_exchangeability` needs from it.
- `null_threshold` (core.py:880– ): generates tracks (or accepts them via optional
  `tracks=` param), pools finite values, `threshold = quantile(pooled,
  config.null_quantile)`, returns `(threshold, diagnostics)`. **This is the natural
  integration point for `null_exchangeability_ratio`**, since it already generates/
  receives the pooled null and is called from `call_dmrs` after `statistic`/`eligible`
  are both already computed.
- `call_dmrs` (core.py, `null_diagnostics` section): calls `statistic =
  edge_statistic(deflated, config)` and `eligible = eligible_edges(frame, config)`
  **before** `threshold, null_diagnostics = null_threshold(deflated, config, frame)` —
  so `statistic`/`eligible` are available to pass into `null_threshold` for the
  exchangeability computation without restructuring control flow. Note: in the
  `null_mode != "permutation"` branch, `call_dmrs` currently calls `null_tracks(...)`
  a **second** time after `null_threshold` (pre-existing, wasteful-but-deterministic
  double generation, out of scope to fix here) — extending `null_threshold` to do the
  exchangeability split avoids adding a *third* generation.
- `Eiger2Config` fields relevant here: `window: int = 80` (core.py:80 — actually
  `window: int = 4`, confirm by read before editing anything nearby), `null_quantile:
  float = 0.995` (core.py:152), `null_min_lag: int = 1000` (core.py:153).
- `cli.py`'s existing stderr-warning pattern (overdispersion, Stage 1) is at
  `caller_manifest.get("fdr", {}).get("null_count_overdispersion_warning")` →
  conditional print — mirror this exact pattern for
  `null_exchangeability_ratio` above 1.25, reading it from
  `caller_manifest.get("null", {}).get("null_exchangeability_ratio")` (the
  `null_diagnostics` dict is stored at `manifest["null"]` in `call_dmrs`).
- `benchmarking/benchmark_suite.py`: four existing commands
  (`negative`/`sweep`/`scaling`/`calibration`), each a `command_X(args)` function
  calling a pure `X(...)` data-producing function, writing CSV(s) via `_write`, printed
  via a compact summary. `command_all` runs all four in sequence. `build_parser`'s
  `command` positional has `choices=["negative", "sweep", "scaling", "calibration",
  "all"]`. `null_tracks`/`null_threshold` are not currently imported into this file.
- `tests/test_eiger2_core.py` imports `permutation_null_tracks` (not `null_tracks`,
  `_lag_null_track`, or `null_threshold` — add these to the import block as needed) and
  has a `locations(n_cpg, chrom="chr1")` helper and `FAST = {"null_replicates": 6}`
  config-override dict, both reusable here.

---

### Task 1: Baseline confirmation on the continuing worktree

**Files:** none (verification only).

**Interfaces:** none.

- [ ] **Step 1: Confirm worktree state and re-run the full baseline**

```bash
cd /Users/tobymilne-clark/Documents/Code/Github_beta_Versions/EIGER-wt-fdr
git status --short
git rev-parse HEAD
conda activate eiger && pytest tests -q
conda activate eiger && pytest old_code/tests -q
conda activate eiger && ruff check Python/eiger2
conda activate eiger && ruff check tests
```

Expect: clean status, HEAD `6965fd7` (or a descendant if more work landed since), 87
passed / 27 passed / both ruff scopes clean — matching the Stage 1 STOP report
(`STAGE1_STOP_REPORT.md`) exactly. If this baseline is not genuinely green, stop and
report before touching anything — do not proceed on a red baseline.

**Environment note, carried forward from Stage 1's own lessons:** every command above
needs the `eiger` conda environment explicitly active. Several points during Stage 1
hit spurious `ModuleNotFoundError`/`command not found` failures from shells that
looked active but weren't — always verify with `which python` / `which ruff` /
`python -c "import numba"` if a failure looks environment-shaped before concluding
it's a real regression.

- [ ] **Step 2: Reproduce §1b.4's baseline (pre-fix) numbers**

Before writing any fix, reproduce the diagnosis's "before" column so this environment
is confirmed to match. Write a throwaway script (not committed) reproducing the
per-edge exceedance ratio table:

```python
import sys; sys.path.insert(0, "Python")
import numpy as np
from eiger2.core import Eiger2Config, centre_cpgs, estimate_latent_basis, deflate, \
    edge_statistic, eligible_edges, null_threshold
from eiger2.benchmark import simulate

SCENARIOS = {
    "clean": dict(n_samples=80, n_latent=0, noise_sd=0.04, missing_rate=0.0),
    "nuisance": dict(n_samples=80, n_latent=2, latent_loading_sd=0.07,
                     noise_sd=0.04, missing_rate=0.0),
    "missingness": dict(n_samples=80, n_latent=2, latent_loading_sd=0.07,
                        noise_sd=0.04, missing_rate=0.25),
    "small_n": dict(n_samples=12, n_latent=1, latent_loading_sd=0.07,
                    noise_sd=0.04, missing_rate=0.0),
    "high_noise": dict(n_samples=40, n_latent=2, latent_loading_sd=0.10,
                       noise_sd=0.08, missing_rate=0.0),
}

for name, params in SCENARIOS.items():
    matrix, locs, _ = simulate(seed=0, strata=[], replicates=0,
                                spacing=20000, **params)
    config = Eiger2Config(window=4, statistic="correlation", null_replicates=100,
                           random_seed=0)
    residuals = centre_cpgs(matrix)
    basis, _ = estimate_latent_basis(residuals, config)
    deflated = deflate(residuals, basis, missing=config.deflation_missing)
    statistic = edge_statistic(deflated, config)
    eligible = eligible_edges(locs, config)
    threshold, diag = null_threshold(deflated, config, locs)
    obs_exceed = float(np.mean(statistic[eligible & np.isfinite(statistic)] >= threshold))
    print(f"{name:12s} threshold={threshold:.4f}  obs_exceed={obs_exceed:.5f}  "
          f"ratio_vs_nominal={obs_exceed / (1 - config.null_quantile):.2f}")
```

Run it, compare against spec §1b.4's "before" column (clean 1.70×, nuisance 1.69×,
missingness 6.19×, small_n 2.53×, high_noise 1.78× — noting the spec table's numbers
come from a held-out-half comparison, not this simpler whole-pool comparison, so exact
match is not expected; the requirement is qualitative agreement — missingness clearly
the worst by several-fold, clean/nuisance mildly elevated, not the reverse). If the
qualitative pattern doesn't hold, stop and report before proceeding.

- [ ] **Step 3: Record and commit nothing yet**

This task produces no commit — it's a go/no-go gate. Proceed to Task 2 only if Steps 1
and 2 both pass.

---

### Task 2: Replace `_lag_null_track` with the correct window assembly

**Files:**
- Modify: `Python/eiger2/core.py`
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- Consumes: `_lag_statistic` (unchanged).
- Produces: `_lag_null_track(residuals, lag, config) -> np.ndarray` — same signature,
  corrected body. Consumed unchanged by `null_tracks` (Task 3 doesn't touch the
  call site).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_eiger2_core.py`:

```python
from eiger2.core import _lag_null_track, _lag_statistic, _null_eligibility  # noqa: E402


def _reference_lag_null_track(residuals, lag, config):
    """Old (buggy) implementation, kept ONLY as a fixed reference point for the
    no-op-at-w<=2 test below -- never reintroduced into the package."""
    from eiger2.core import _window_from_lags
    span = max(config.window, 2)
    usable = residuals.shape[0] - lag * (span - 1)
    if usable < 2:
        return np.empty(0, dtype=float)
    lag_statistics = []
    for separation in range(1, span):
        values = _lag_statistic(residuals, separation * lag, config)
        lag_statistics.append(values[:usable])
    if config.window == 1:
        return lag_statistics[0]
    return _window_from_lags(lag_statistics, config.window)


def test_lag_null_track_is_a_noop_at_window_1_and_2():
    rng = np.random.default_rng(101)
    residuals = rng.normal(size=(200, 30))
    for w in (1, 2):
        config = Eiger2Config(window=w, random_seed=1)
        old = _reference_lag_null_track(residuals, lag=17, config=config)
        new = _lag_null_track(residuals, lag=17, config=config)
        assert new.shape == old.shape
        np.testing.assert_array_equal(new, old)


def test_lag_null_track_member_set_is_exactly_the_lag_multiples():
    # Build a residual matrix where stat(row j, row k) is recoverable from the
    # pair itself: use two samples, values j and k respectively, so
    # correlation/covariance of (j, j2, ..) vs (k, k2, ..) across a handful of
    # synthetic "samples" encodes (j, k) in a way we can invert. Simpler and
    # exact: monkeypatch _lag_statistic to return a matrix whose value at
    # position i is the pair (i, i+d) encoded as i * 10_000 + d, then confirm
    # _lag_null_track's output at window position i decodes to exactly the
    # pairs {(i + a*lag, i + b*lag) : 0 <= a < b < w} and no others.
    import eiger2.core as core_mod

    lag = 5
    w = 4
    n_positions = 30

    def fake_lag_statistic(residuals, d, config, chunk=200_000):
        n_pairs = residuals.shape[0] - d
        # value at row i encodes the pair (i, i + d) uniquely
        return np.array([i * 100_000 + d for i in range(n_pairs)], dtype=float)

    original = core_mod._lag_statistic
    core_mod._lag_statistic = fake_lag_statistic
    try:
        config = Eiger2Config(window=w, random_seed=1)
        dummy_residuals = np.zeros((n_positions + lag * (w - 1) + 5, 3))
        out = core_mod._lag_null_track(dummy_residuals, lag, config)
    finally:
        core_mod._lag_statistic = original

    # decode window position i=0: expect the average of pair-encodings for
    # {(0, lag), (0, 2*lag), (0, 3*lag), (lag, 2*lag), (lag, 3*lag), (2*lag, 3*lag)}
    expected_pairs = [(a * lag, b * lag) for a in range(w) for b in range(a + 1, w)]
    expected_values = [j * 100_000 + (k - j) for (j, k) in expected_pairs]
    assert out[0] == pytest.approx(np.mean(expected_values))

    # and NOT the old (buggy) construction's pairs, which would include e.g.
    # (0+lag, 0+2) -- a base offset that isn't a multiple of lag
    wrong_pairs_value = 0 * 100_000 + 2  # what the old code would have pulled in
    assert wrong_pairs_value not in expected_values


def test_null_eligibility_offsets_match_lag_null_track_member_set():
    lag = 7
    for w in (1, 2, 3, 5):
        config = Eiger2Config(window=w, random_seed=1)
        span = max(w, 2)
        size = 50
        chromosome = np.array(["chr1"] * (size + lag * (span - 1) + 5))
        eligible = _null_eligibility(chromosome, lag, config, size)
        # hand-derive the same offsets _lag_null_track's member set implies
        expected = np.ones(size, dtype=bool)
        first = chromosome[:size]
        for step in range(1, span):
            offset = step * lag
            expected &= first == chromosome[offset : offset + size]
        np.testing.assert_array_equal(eligible, expected)


def test_lag_null_track_slice_bounds_hold_for_large_window_and_lag():
    rng = np.random.default_rng(202)
    for w in (1, 2, 4, 8, 12):
        for lag in (2, 50, 1999):
            n = 4000
            residuals = rng.normal(size=(n, 10))
            config = Eiger2Config(window=w, random_seed=1)
            out = _lag_null_track(residuals, lag, config)
            span = max(w, 2)
            expected_usable = max(n - lag * (span - 1), 0)
            if expected_usable < 2:
                assert out.size == 0
            else:
                assert out.size == expected_usable
```

- [ ] **Step 2: Run and confirm failure**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py -k "lag_null_track or null_eligibility_offsets" -v
```

Expected: `test_lag_null_track_is_a_noop_at_window_1_and_2` and
`test_null_eligibility_offsets_match_lag_null_track_member_set` likely already PASS
(the defect is a no-op at `w<=2`, and eligibility is already correct) — that's fine,
they're regression guards, not red/green markers for this change. The member-set test
and slice-bounds test should FAIL or, for member-set, assert against the *wrong*
current output. Run each individually and confirm `test_lag_null_track_member_set_is_
exactly_the_lag_multiples` fails against the current buggy implementation (it should,
since the current code pulls in `_lag_statistic`'s consecutive-index values, not the
lag-multiple ones the test expects).

- [ ] **Step 3: Replace `_lag_null_track`**

Replace the function body at `core.py:686–712` with the spec's exact replacement
(spec §1b.2, reproduced verbatim — this is release math, transcribe exactly):

```python
def _lag_null_track(
    residuals: np.ndarray, lag: int, config: Eiger2Config
) -> np.ndarray:
    """One null track: window members are ``lag`` CpGs apart, not adjacent.

    The window at null position ``i`` comprises CpGs ``i, i+lag, ..., i+(w-1)*lag``
    and the statistic is the mean over the complete pairwise set among *those*
    members: pair ``(a, b)`` is ``S_{b-a}[i + a*lag]`` where
    ``S_d[j] = stat(CpG j, CpG j + d*lag)``.

    Reusing ``_window_from_lags`` here would be wrong: it assumes members sit at
    consecutive indices, which is true of the observed track and false of this
    one. See docs/FDR.md, "Null window assembly".
    """

    w = config.window
    span = max(w, 2)
    usable = residuals.shape[0] - lag * (span - 1)
    if usable < 2:
        return np.empty(0, dtype=float)

    S = {d: _lag_statistic(residuals, d * lag, config) for d in range(1, span)}
    if w == 1:
        return S[1][:usable]

    total = np.zeros(usable, dtype=float)
    count = np.zeros(usable, dtype=float)
    for a in range(w):
        for b in range(a + 1, w):
            values = S[b - a][a * lag : a * lag + usable]
            total += np.nan_to_num(values, nan=0.0)
            count += np.isfinite(values)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, total / np.maximum(count, 1.0), np.nan)
    expected = w * (w - 1) / 2.0
    return np.where(count >= 0.5 * expected, out, np.nan)
```

Preserve the original docstring's opening paragraph (the "every sample moves
together... only adjacency is broken" rationale) by folding it into this
docstring too, since it explains *why* this construction (not just *what* it
computes) — append it after the "See docs/FDR.md" line.

Per spec §1b.2's note: assert the slice-bounds algebra rather than only relying on
it — add, right before the `for a in range(w):` loop:

```python
    assert (w - 2) * lag + usable <= S[1].size if w >= 3 else True, (
        "slice bounds violated -- see docs/FDR.md, 'Null window assembly'"
    )
```

(This assertion should be provably always true per the spec's derivation, but per
§1b.2's explicit instruction ("assert it rather than relying on the algebra"), keep it
in the shipped function, not just the test suite — it is cheap and catches any future
change to the lag/usable computation that would silently violate it.)

- [ ] **Step 4: Run tests, confirm pass**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py -k "lag_null_track or null_eligibility_offsets" -v
```

All 5 new tests pass, including the no-op-at-w<=2 guard.

- [ ] **Step 5: Run the full regression suite**

```bash
conda activate eiger && pytest tests -q
```

Expect count to change (existing tests that assert specific `region_area`/
`statistic_threshold`/region-set values from before this fix will now fail — this is
**expected** per the revised invariant contract in this plan's Global Constraints, not
a regression). Catalogue every failing test by name in this task's report; do not
"fix" any of them yet — Task 6 handles retargeting the invariant test, and any other
now-stale assertions get evaluated case by case (most existing FDR/region-count tests
use `pytest.approx`/structural assertions or fresh synthetic data per test, not pinned
literal region-set values, so the expected blast radius is small — likely just the
Stage 1 invariant fixture test).

- [ ] **Step 6: Commit**

```bash
git add Python/eiger2/core.py tests/test_eiger2_core.py
git commit -m "fix: correct _lag_null_track window assembly (Stage 1b)"
```

---

### Task 3: Null-exchangeability diagnostic, wired into every `eiger2 call`

**Files:**
- Modify: `Python/eiger2/core.py` (`null_exchangeability`, `null_threshold` extension)
- Modify: `Python/eiger2/cli.py` (stderr warning)
- Test: `tests/test_eiger2_core.py`

**Interfaces:**
- Produces: `core.null_exchangeability(tracks, statistic, eligible, config) -> dict`
  with keys `p_null, p_obs, null_exchangeability_ratio, n_calibration_replicates,
  n_holdout_replicates, calibration_threshold`. Consumed by `null_threshold` (adds
  these keys to its returned diagnostics dict) and by Task 4's benchmark command.
- `null_threshold`'s signature gains keyword-only `statistic=None, eligible=None`,
  defaulting to the old behaviour (no exchangeability fields) when omitted — no
  existing caller breaks.

- [ ] **Step 1: Write the failing tests**

```python
from eiger2.core import null_exchangeability, null_threshold, null_tracks  # noqa: E402


def test_null_exchangeability_self_consistency_control():
    # held-out null exceedance must equal 1 - null_quantile to within Monte Carlo
    # error -- this is the control that proves the gate measures the right thing
    rng = np.random.default_rng(303)
    residuals = rng.normal(size=(3000, 40))
    config = Eiger2Config(window=1, random_seed=7, null_replicates=200)
    tracks = null_tracks(residuals, config)
    statistic = rng.normal(size=2999)  # arbitrary observed track, same length class
    eligible = np.ones(2999, dtype=bool)
    result = null_exchangeability(tracks, statistic, eligible, config)
    assert result["p_null"] == pytest.approx(1 - config.null_quantile, abs=0.003)


def test_null_exchangeability_ratio_is_one_when_observed_equals_null():
    rng = np.random.default_rng(404)
    residuals = rng.normal(size=(3000, 40))
    config = Eiger2Config(window=1, random_seed=7, null_replicates=200)
    tracks = null_tracks(residuals, config)
    # use one of the null's own held-out-eligible values as a stand-in "observed"
    # track drawn from the identical distribution -- ratio should be close to 1
    pooled_all = np.concatenate([t[np.isfinite(t) & e] for t, e in tracks])
    observed_like_null = rng.choice(pooled_all, size=2000, replace=True)
    eligible = np.ones(2000, dtype=bool)
    result = null_exchangeability(tracks, observed_like_null, eligible, config)
    assert result["null_exchangeability_ratio"] == pytest.approx(1.0, abs=0.3)


def test_null_exchangeability_deterministic_split():
    rng = np.random.default_rng(505)
    residuals = rng.normal(size=(2000, 30))
    config = Eiger2Config(window=1, random_seed=9, null_replicates=100)
    tracks = null_tracks(residuals, config)
    statistic = rng.normal(size=1999)
    eligible = np.ones(1999, dtype=bool)
    first = null_exchangeability(tracks, statistic, eligible, config)
    second = null_exchangeability(tracks, statistic, eligible, config)
    assert first == second


def test_null_threshold_reports_exchangeability_when_statistic_given():
    rng = np.random.default_rng(606)
    residuals = rng.normal(size=(2000, 30))
    config = Eiger2Config(window=1, random_seed=3, null_replicates=100)
    statistic = rng.normal(size=1999)
    eligible = np.ones(1999, dtype=bool)
    threshold, diagnostics = null_threshold(
        residuals, config, statistic=statistic, eligible=eligible,
    )
    assert "null_exchangeability_ratio" in diagnostics
    assert diagnostics["null_exchangeability_ratio"] > 0


def test_null_threshold_omits_exchangeability_when_statistic_not_given():
    rng = np.random.default_rng(707)
    residuals = rng.normal(size=(2000, 30))
    config = Eiger2Config(window=1, random_seed=3, null_replicates=100)
    threshold, diagnostics = null_threshold(residuals, config)
    assert "null_exchangeability_ratio" not in diagnostics
```

- [ ] **Step 2: Run and confirm failure**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py -k null_exchangeability -v
```

Expected: `ImportError` (function doesn't exist).

- [ ] **Step 3: Implement `null_exchangeability`**

Add to `core.py`, above `null_threshold`:

```python
def null_exchangeability(
    tracks: list[tuple[np.ndarray, np.ndarray]],
    statistic: np.ndarray,
    eligible: np.ndarray,
    config: Eiger2Config,
    *,
    calibration_fraction: float = 0.5,
) -> dict:
    """Split-half check that the null distribution matches the observed one.

    Splits the null replicates into a calibration half (sets the threshold) and a
    held-out half (measures exceedance against it). ``p_null`` is the held-out
    exceedance and is a control: by construction it must equal
    ``1 - config.null_quantile``, confirming the null is internally self-consistent.
    ``p_obs`` is the observed track's exceedance against the SAME threshold. The
    ratio ``p_obs / p_null`` is the actual gate: it is 1.0 when the null and the
    data share the same upper-tail shape, and rises when the observed track is
    more heavy-tailed than the null represents -- exactly the failure mode
    described in docs/FDR.md, "Null exchangeability".

    The split is deterministic given ``config.random_seed`` -- a fixed permutation
    of replicate indices, not re-derived from tracks' own order, so results are
    reproducible independent of how many replicates happen to survive filtering
    upstream.
    """

    n = len(tracks)
    rng = np.random.default_rng(config.random_seed + 30011)
    order = rng.permutation(n)
    n_calibration = max(1, int(round(n * calibration_fraction)))
    calibration_idx = order[:n_calibration]
    holdout_idx = order[n_calibration:]

    def _pool(indices):
        if indices.size == 0:
            return np.empty(0, dtype=float)
        values = np.concatenate([
            tracks[i][0][np.isfinite(tracks[i][0]) & tracks[i][1]]
            for i in indices
        ])
        return values

    calibration_pool = _pool(calibration_idx)
    holdout_pool = _pool(holdout_idx)

    calibration_threshold = (
        float(np.quantile(calibration_pool, config.null_quantile))
        if calibration_pool.size else float("nan")
    )

    p_null = (
        float(np.mean(holdout_pool >= calibration_threshold))
        if holdout_pool.size and np.isfinite(calibration_threshold) else float("nan")
    )

    observed_eligible = statistic[eligible & np.isfinite(statistic)]
    p_obs = (
        float(np.mean(observed_eligible >= calibration_threshold))
        if observed_eligible.size and np.isfinite(calibration_threshold)
        else float("nan")
    )

    ratio = (
        p_obs / p_null if p_null and np.isfinite(p_null) and p_null > 0
        else float("nan")
    )

    return {
        "p_null": p_null,
        "p_obs": p_obs,
        "null_exchangeability_ratio": ratio,
        "n_calibration_replicates": int(calibration_idx.size),
        "n_holdout_replicates": int(holdout_idx.size),
        "calibration_threshold": calibration_threshold,
    }
```

Extend `null_threshold`'s signature (core.py:880) to accept the observed track and
compute exchangeability when given:

```python
def null_threshold(
    residuals: np.ndarray,
    config: Eiger2Config,
    locations: pd.DataFrame | None = None,
    tracks: list | None = None,
    *,
    statistic: np.ndarray | None = None,
    eligible: np.ndarray | None = None,
) -> tuple[float, dict]:
    """Upper-quantile threshold of the null statistic distribution."""

    if tracks is None:
        tracks = null_tracks(residuals, config, locations)
    pooled = (
        np.concatenate([t[e] for t, e in tracks]) if tracks else np.empty(0)
    )
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        raise ValueError(
            "the locality-breaking null produced no usable pairs; the matrix is "
            "probably too small or too sparse"
        )
    threshold = float(np.quantile(pooled, config.null_quantile))
    offered = sum(int(t.size) for t, _ in tracks) if tracks else 0
    coverage = float(pooled.size / offered) if offered else 0.0
    diagnostics = {
        "null_replicates": len(tracks),
        "null_values": int(pooled.size),
        "null_positions_offered": offered,
        "null_coverage": coverage,
        "null_quantile": config.null_quantile,
        "null_median": float(np.median(pooled)),
        "threshold": threshold,
    }
    if statistic is not None and eligible is not None:
        diagnostics.update(
            null_exchangeability(tracks, statistic, eligible, config)
        )
    if pooled.size < config.min_null_values or coverage < config.min_null_coverage:
        diagnostics["warning"] = (
            f"null is thin: {pooled.size} usable positions "
            f"({coverage:.1%} of those offered). The threshold and every "
            f"q-value derived from it are unreliable. Reduce null_min_lag, "
            f"raise null_replicates, or call chromosomes separately."
        )
    return threshold, diagnostics
```

(Keep everything after the shown block — e.g. any trailing lines this task's Read
didn't need to reproduce — exactly as they are; this replaces only the signature line
and inserts the one new `if statistic is not None...` block before the existing
`min_null_values`/`min_null_coverage` warning check, which stays where it is.)

In `call_dmrs`, change the call site (search for `threshold, null_diagnostics =
null_threshold(deflated, config, frame)`) to:

```python
    threshold, null_diagnostics = null_threshold(
        deflated, config, frame, statistic=statistic, eligible=eligible,
    )
```

(`statistic` and `eligible` are already computed above this line in `call_dmrs` — see
Verified Baseline above.)

- [ ] **Step 4: CLI stderr warning**

In `cli.py`'s `command_call`, add — mirroring the existing overdispersion-warning
block exactly in style, placed adjacently to it:

```python
    exchangeability_ratio = (caller_manifest.get("null") or {}).get(
        "null_exchangeability_ratio"
    )
    if exchangeability_ratio is not None and exchangeability_ratio > 1.25:
        print(
            f"[eiger2] WARNING: observed edges exceed the null threshold "
            f"{exchangeability_ratio:.2f}x more often than the null itself "
            f"does (nominal 1.0x). The null is not exchangeable with the data; "
            f"region_qvalue and significant are unreliable. See docs/FDR.md, "
            f"'Null exchangeability'.",
            file=sys.stderr,
        )
```

Place this near the existing overdispersion print in `command_call` (search for
`overdispersion_ratio = (caller_manifest.get("fdr")...`) — same function, same general
area, since both are pre-flight-quality-of-results warnings.

- [ ] **Step 5: Run tests, confirm pass**

```bash
conda activate eiger && pytest tests/test_eiger2_core.py -k null_exchangeability -v
conda activate eiger && pytest tests -q
```

- [ ] **Step 6: Commit**

```bash
git add Python/eiger2/core.py Python/eiger2/cli.py tests/test_eiger2_core.py
git commit -m "feat: add null_exchangeability diagnostic, wired into null_threshold and eiger2 call"
```

---

### Task 4: `benchmarking/benchmark_suite.py::exchangeability` command

**Files:**
- Modify: `benchmarking/benchmark_suite.py`

**Interfaces:**
- Consumes: `core.null_exchangeability`, `core.null_tracks`, `core.null_threshold`
  (Task 3).
- Produces: `exchangeability(seeds, scenarios) -> pd.DataFrame`, `command_exchangeability`,
  registered in `build_parser`'s choices and `main`'s dispatch dict, included in
  `command_all`.

- [ ] **Step 1: Implement the experiment function**

Add to `benchmarking/benchmark_suite.py`, near the other experiment functions (after
`negative_control`, matching its style):

```python
def exchangeability(
    seeds: Sequence[int],
    scenarios: Sequence[str],
    *,
    statistic: str = "correlation",
) -> pd.DataFrame:
    """Split-half null self-consistency check, per spec §1b.3.

    For each DMR-free scenario/seed: generate null replicates, split into
    calibration/held-out halves, threshold on the calibration half, measure
    exceedance on the held-out half (the control -- must equal
    ``1 - null_quantile``) and on the observed track (the actual gate).
    """

    from eiger2.core import (
        Eiger2Config, centre_cpgs, deflate, edge_statistic, eligible_edges,
        estimate_latent_basis, null_exchangeability, null_tracks,
    )

    rows: list[dict] = []
    for scenario_name in scenarios:
        for seed in seeds:
            matrix, locations, _ = simulate(
                seed=seed, strata=[], replicates=0,
                spacing=20000, **SCENARIOS[scenario_name],
            )
            config = Eiger2Config(statistic=statistic, random_seed=seed)
            residuals = centre_cpgs(matrix)
            basis, _ = estimate_latent_basis(residuals, config)
            deflated = deflate(residuals, basis, missing=config.deflation_missing)
            statistic_track = edge_statistic(deflated, config)
            eligible = eligible_edges(locations, config)
            tracks = null_tracks(deflated, config, locations)
            result = null_exchangeability(tracks, statistic_track, eligible, config)
            rows.append({
                "scenario": scenario_name, "seed": seed,
                "n_cpg": int(matrix.shape[0]), "n_samples": int(matrix.shape[1]),
                **result,
            })
    return pd.DataFrame(rows)


def command_exchangeability(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    seeds = list(range(args.seeds))
    frame = exchangeability(seeds, args.scenarios, statistic=args.statistic)
    _write(frame, out / "exchangeability.csv")
    summary = frame.groupby("scenario")[
        ["p_null", "p_obs", "null_exchangeability_ratio"]
    ].mean()
    print(summary.round(4).to_string())
    worst = frame["null_exchangeability_ratio"].max()
    print(f"  worst null_exchangeability_ratio: {worst:.3f} "
          f"({'PASS' if worst <= 1.25 else 'FAIL'} at gate 1.25)")
```

- [ ] **Step 2: Register the command**

In `build_parser`, change:
```python
    parser.add_argument("command", choices=[
        "negative", "sweep", "scaling", "calibration", "all"
    ])
```
to:
```python
    parser.add_argument("command", choices=[
        "negative", "sweep", "scaling", "calibration", "exchangeability", "all"
    ])
```

In `main`'s `dispatch` dict, add `"exchangeability": command_exchangeability,`.

In `command_all`, add `("exchangeability", command_exchangeability),` to the tuple of
`(name, function)` pairs.

Update the module's top docstring to document the fifth experiment, in the same style
as the existing four (a short paragraph under a new `` ``exchangeability`` `` heading).

- [ ] **Step 3: Smoke-test**

```bash
conda activate eiger && python -m benchmarking.benchmark_suite exchangeability \
  --out-dir /tmp/exch_smoke --seeds 3 --scenarios clean missingness
```

Confirm it runs to completion and prints a plausible table (missingness's ratio should
be visibly closer to 1.0 than the pre-fix diagnosis's 6.19, though at 3 seeds exact
values will differ from the 100-seed §1e figures).

- [ ] **Step 4: Commit**

```bash
git add benchmarking/benchmark_suite.py
git commit -m "feat: add exchangeability benchmark command (spec §1b.3)"
```

---

### Task 5: Reproduce spec §1b.4's expected-results tables

**Files:** none shipped — this task produces the STOP 1b report's evidence, not code.

**Interfaces:**
- Consumes: everything from Tasks 2–4.

- [ ] **Step 1: Reproduce the per-edge exceedance and tail-shape tables**

Using the same script structure as Task 1 Step 2, now against the FIXED
`_lag_null_track`, at `window=4, correlation, null_replicates=100, seed 0, 20000 CpG,
DMR-free`, reproduce:
- per-edge exceedance ratio (`p_obs/p_null` via `null_exchangeability`, not the
  simpler Task-1 approximation) for all 5 scenarios,
- null/observed q99.5 and excess kurtosis (via `scipy.stats.kurtosis` on the pooled
  null and the eligible observed track) for all 5 scenarios.

Compare against spec §1b.4's "after" columns. Record actual printed numbers in the
STOP 1b report — not "matches expectations," the real numbers.

- [ ] **Step 2: Reproduce the DMR-free significant-region counts**

Run `call_dmrs` end-to-end (not just the threshold) on the same 5 scenarios × 3 seeds
(0, 1, 2), DMR-free, and record `significant.sum()` per run. Compare against spec
§1b.4's "after" table (clean 1/0/0, nuisance 0/1/0, missingness 5/0/0, small_n 0/2/1,
high_noise 3/1/2). Exact match is not required (seeds/simulator details may differ
slightly from the diagnosis's exact harness) but the qualitative pattern — missingness
sharply reduced from the pre-fix 51/56/0, others small counts — must hold.

- [ ] **Step 3: Reproduce the planted-signal FDR/recall table**

Using `benchmarking/benchmark_suite.py`'s `heuristic_sweep`-style planted-signal
generation (24 DMRs, 3 seeds pooled — check `heuristic_sweep`'s existing signal-
planting parameters and reuse them, don't invent new ones), compute realised FDR and
recall per scenario, before (git-stash Task 2's commit temporarily, or use the
`_reference_lag_null_track` kept in the test file) vs. after. Compare against spec
§1b.4's table. Recall must not regress by more than 5 percentage points in any
scenario (this is also pinned by Task 2's/this task's test coverage — see Task 6).

- [ ] **Step 4: Write findings to the report, no commit**

Compile all three tables with real numbers into a scratch note
(`.superpowers/sdd/<workspace>/stage1b-reproduction-tables.md` or similar) for the
STOP 1b report to draw from directly. If any table's qualitative pattern doesn't
match the spec's, stop and report before proceeding to Task 6.

---

### Task 6: Recall-regression test, exchangeability-gate tests, invariant test retarget

**Files:**
- Modify: `tests/test_eiger2_core.py`
- Modify: `benchmarking/benchmark_suite.py` tests if any exist, else skip
- Modify: `tests/test_eiger2_core.py::test_stage1_preserves_the_invariant_column_set`
  (retarget to the narrowed contract from this plan's Global Constraints)

**Interfaces:**
- Consumes: everything from Tasks 2–5.

- [ ] **Step 1: Write the failing/target tests**

```python
def test_exchangeability_ratio_at_most_1_25_on_dmr_free_scenarios():
    # spec §1b.5 item 4: all five scenarios, 10 seeds. This is a correctness gate,
    # not an exploratory check -- do not reduce scenario/seed coverage for speed.
    from eiger2.benchmark import simulate
    from eiger2.core import (
        centre_cpgs, deflate, edge_statistic, eligible_edges,
        estimate_latent_basis, null_exchangeability, null_tracks,
    )
    scenarios = {
        "clean": dict(n_samples=80, n_latent=0, noise_sd=0.04, missing_rate=0.0),
        "nuisance": dict(n_samples=80, n_latent=2, latent_loading_sd=0.07,
                         noise_sd=0.04, missing_rate=0.0),
        "missingness": dict(n_samples=80, n_latent=2, latent_loading_sd=0.07,
                            noise_sd=0.04, missing_rate=0.25),
        "small_n": dict(n_samples=12, n_latent=1, latent_loading_sd=0.07,
                        noise_sd=0.04, missing_rate=0.0),
        "high_noise": dict(n_samples=40, n_latent=2, latent_loading_sd=0.10,
                           noise_sd=0.08, missing_rate=0.0),
    }
    for name, params in scenarios.items():
        for seed in range(10):
            matrix, locs, _ = simulate(seed=seed, strata=[], replicates=0,
                                        spacing=5000, **params)
            config = Eiger2Config(window=4, random_seed=seed, null_replicates=100)
            residuals = centre_cpgs(matrix)
            basis, _ = estimate_latent_basis(residuals, config)
            deflated = deflate(residuals, basis, missing=config.deflation_missing)
            statistic = edge_statistic(deflated, config)
            eligible = eligible_edges(locs, config)
            tracks = null_tracks(deflated, config, locs)
            result = null_exchangeability(tracks, statistic, eligible, config)
            assert result["null_exchangeability_ratio"] <= 1.25, (name, seed, result)


def test_recall_does_not_regress_more_than_5_points_after_lag_null_fix():
    # Uses the kept _reference_lag_null_track (old, buggy construction) as the
    # "before" comparison point, monkeypatched in, against the shipped "after".
    import eiger2.core as core_mod
    from eiger2.benchmark import simulate, evaluate

    matrix, locs, planted = simulate(
        n_samples=80, strata=[(0.2, 0.35)], replicates=6,
        spacing=400, dmr_length=15, seed=11,
    )
    config = Eiger2Config(window=4, null_replicates=50, random_seed=11)

    original = core_mod._lag_null_track
    core_mod._lag_null_track = lambda residuals, lag, config: _reference_lag_null_track(
        residuals, lag, config
    )
    try:
        before_regions, _, _ = call_dmrs(matrix, locs, config=config)
    finally:
        core_mod._lag_null_track = original
    after_regions, _, _ = call_dmrs(matrix, locs, config=config)

    before_eval = evaluate(before_regions, planted)
    after_eval = evaluate(after_regions, planted)
    assert after_eval["recall"] >= before_eval["recall"] - 0.05
```

(Check `eiger2.benchmark.evaluate`'s actual return keys before writing this test —
adjust `"recall"` to whatever key it actually uses; read the function first.)

Retarget the Stage 1 invariant test. Read the current
`test_stage1_preserves_the_invariant_column_set` and its committed fixture
(`tests/fixtures/stage1_invariant_fixture_v200.csv`, 24 columns). Per this plan's
Global Constraints, the columns that changed by design now include `region_area` and
`statistic_threshold`, which that fixture currently pins. Do **not** delete the old
fixture (spec §1: "retained for historical reference"). Instead:

1. Rename the test to `test_stage1_preserves_the_narrowed_invariant_set` (or similar),
   asserting only the columns spec §1 lists as still byte-identical:
   `edge_statistic()`'s raw output and `eligible_edges()`'s raw output — **not** a
   `regions.csv` column comparison at all, since region boundaries themselves now
   change. Generate a new fixture,
   `tests/fixtures/stage1b_invariant_fixture_v201.csv` (or a `.npz`/`.npy` pair, since
   this is raw arrays, not a region table — pick whichever format is cleaner given
   what `edge_statistic`/`eligible_edges` actually return; check their return types
   before deciding), from the SAME `generate_inputs()` used by the old fixture (reuse
   `tests/fixtures/generate_stage1_invariant_fixture.py`'s `generate_inputs()` so the
   two fixtures are directly comparable / share the same simulated input), run through
   `edge_statistic`/`eligible_edges` on the **v2.0.1** code (i.e. after Stage 1 but
   before Stage 1b — check out `ba046c9` in a scratch location, or simpler: since
   Task 2/3 of this stage do not touch `edge_statistic`/`eligible_edges` at all, the
   CURRENT worktree's output for these two functions IS the v2.0.1 baseline; generate
   the fixture from the current worktree state, confirm by reading `edge_statistic`'s
   diff across this whole plan is empty).
2. Keep the OLD test's fixture file and a note in its docstring/module comment stating
   it documents v2.0.0 output and is retained for historical reference only, per spec
   §1 — do not run it against post-Stage-1b code (or if you do keep it running, assert
   only on the subset of its 24 columns that are still invariant per spec §1's list:
   `region_id`? No — region_id depends on region boundaries which change. Re-derive
   which of the OLD fixture's 24 columns, if any, remain meaningful given the narrowed
   contract; if none do (likely, since almost everything in that fixture is
   region-boundary-derived), retire the old test to a skip/xfail with a clear comment
   rather than deleting the fixture file itself, per spec §1's explicit "do not
   delete" instruction).

- [ ] **Step 2: Run and confirm pass**

```bash
conda activate eiger && pytest tests -q
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_eiger2_core.py tests/fixtures/
git commit -m "test: add exchangeability/recall regression tests, retarget invariant test to the narrowed Stage 1b contract"
```

---

### Task 7: STOP 1b report

**Files:** none (report only).

- [ ] **Step 1: Assemble the §0.6 report** (per WP1–3 spec §0.6, referenced by this
  spec's STOP 1b instruction)

Using real data from Tasks 1–6:

1. Change summary, ≤10 sentences.
2. `git diff --stat` for this stage's commits.
3. The four §1b.4 tables (per-edge exceedance, tail shape, DMR-free region counts,
   planted-signal FDR/recall), actual printed numbers from Task 5, both before and
   after.
4. Every test command run and its output tail.
5. The `exchangeability` benchmark command's output (Task 4 Step 3, plus a fuller run
   if time allows — note if only smoke-tested, be honest about scale exactly as Stage
   1's STOP report was).
6. Runtime/memory, if materially different from Stage 1's measurements (likely not,
   since this stage doesn't change the replicate count or algorithm complexity class).
7. Backward-compatibility assessment against the REVISED (narrowed) contract from this
   plan's Global Constraints — name every column/behavior that changed, matching spec
   §1 exactly.
8. Remaining limitations: the missingness residual (§5 of the diagnosis — fix removes
   most inflation but 0.25 realised FDR with planted signal remains, per diagnosis
   §5 — this is exactly what Stage 1c exists to address next).
9. `git diff --check` and final `git status --short`.
10. No claim stronger than the measured evidence.

- [ ] **Step 2: Halt**

Per spec: do not begin Stage 1c. Produce the report and stop for review.
