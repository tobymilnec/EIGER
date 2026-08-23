"""Stage 1f Task 3 / Stage 1g Task B.4: the full benchmark grid the paper needs.

NOT RUN as part of Stage 1f/1g -- this is genuinely hours of compute (see the
runtime estimate in STAGE1F_STOP_REPORT.md §3), explicitly out-of-band per the
spec ("Run it out-of-band; it is hours of compute, not a session"). This script
is the concrete design: every parameter combination is real and runnable, not
sketched.

What this fixes relative to Stage 1e's grid: DEFAULT_STRATA
((0.05,0.35),(0.10,0.30),(0.25,0.25),(0.50,0.20)) is comfortably detectable -- recall
1.000 and realised FDR 0.000 everywhere, on every cell measured so far. A grid on
which the method never makes a mistake cannot demonstrate its error rate is
controlled; it only demonstrates the signal was easy. This script extends the signal
grid DOWN to the detection boundary (small deltas, low prevalence at the two-carrier
floor, small sample sizes in combination with both, dmr_length=min_cpgs) so some
cells produce recall well below 1 and realised FDR measurably near nominal, and adds
a dedicated power-curve sweep (recall vs. effect size at fixed prevalence).

Usage (out-of-band, per stage):
    python paper_grid.py calibration   --seeds 100 --jobs 16 --resume
    python paper_grid.py signal_grid   --seeds 20  --jobs 16 --resume
    python paper_grid.py power_curve   --seeds 10  --jobs 16 --resume
    python paper_grid.py gap_sweep     --seeds 20  --jobs 16 --resume
    python paper_grid.py multichrom    --seeds 20  --jobs 16 --resume

At the measured 225s/cell median (STAGE1F_STOP_REPORT.md §3), serial totals are
~62.5h/17.5h/11.3h/6.7h; `--jobs 16` on a 16-core machine brings each to roughly
1/16th of that, since every cell is independent.

``multichrom`` (spec §2b.5) is DEFINED, not yet run or timed even provisionally --
added alongside Stage 2b's chromosome-aware null construction fix, per-cell cost is
unmeasured (likely higher than the single-chromosome cells above: `layout_chrom=None`
spans all three fixture chromosomes at once). Execution belongs here, with the rest
of this grid, at §2f.

``--jobs N`` (default 1, serial, no new dependency) parallelises across cells with
``concurrent.futures.ProcessPoolExecutor``. One cell per task; each cell's result is
written to its own file under ``results_v210/cells/<subcommand>/`` as soon as it
completes, so a crash loses at most one cell, not the whole run. ``--resume`` skips
any cell whose output file already exists. Workers get ``OMP_NUM_THREADS=1`` and
``NUMBA_NUM_THREADS=1`` so BLAS/numba do not each try to thread within a worker and
oversubscribe against the process pool -- set in the parent's environment before the
pool is created (each worker is a fresh interpreter that imports numpy/numba only
after inheriting that environment) and reasserted in the pool's ``initializer`` for
robustness.

Each subcommand writes its own combined CSV to results_v210/ and never overwrites
another subcommand's file or an earlier results_* directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "layouts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from benchmark_suite import (  # noqa: E402
    DEFAULT_STRATA,
    GAP_SWEEP_VALUES,
    SCENARIOS,
    _classify,
    _v2_called,
    gap_sweep,
)

from eiger2.benchmark import evaluate, simulate  # noqa: E402
from eiger2.core import (  # noqa: E402
    Eiger2Config,
    call_dmrs,
    centre_cpgs,
    chromosome_lag_feasibility,
    null_tracks,
)

OUT = Path(__file__).resolve().parent / "results_v210"
OUT.mkdir(exist_ok=True)
CELLS_DIR = OUT / "cells"

LAYOUTS = ("wgbs_real", "epic_fitted")
SCENARIO_NAMES = ("clean", "nuisance", "missingness", "small_n", "high_noise")
GENOME_SPACING = 50_000          # planted-signal calls: n_cpg ~= spacing*12 + 400
CALIB_SPACING = 600_000          # DMR-free calls (strata=[]): n_cpg = spacing*1 + 400
DENSE_EXCLUDED_NOTE = (
    "dense is excluded from every FDR-facing criterion here (Stage 1d-D SS5) -- "
    "if it is ever added to this script's LAYOUTS for statistic-level continuity "
    "checks, it must not be pooled into the FDR/recall numbers below."
)

# --------------------------------------------------------------------------
# the extended signal grid (Task 3's "what to add")
# --------------------------------------------------------------------------

#: same prevalences as DEFAULT_STRATA, effect sizes pushed down to where recall
#: is not 1.0 -- 0.05 is EIGER's smallest previously-tested delta (already the
#: bottom rung of DEFAULT_STRATA at fraction=0.50); the additional 0.10 rung and
#: reusing 0.05 at the OTHER three prevalences is what is new.
SMALL_DELTA_STRATA: list[tuple[float, float]] = [
    (0.05, 0.10), (0.05, 0.05),
    (0.10, 0.10), (0.10, 0.05),
    (0.25, 0.10), (0.25, 0.05),
    (0.50, 0.10), (0.50, 0.05),
]

#: effect sizes already used in DEFAULT_STRATA, at the two-carrier floor
#: prevalence -- the floor depends on n_samples (2/n_samples), so it is
#: resolved per call, not hard-coded.
LOW_PREVALENCE_EFFECT_SIZES: tuple[float, ...] = (0.35, 0.30, 0.25, 0.20)


def two_carrier_floor_strata(
    n_samples: int, effect_sizes=LOW_PREVALENCE_EFFECT_SIZES,
) -> list[tuple[float, float]]:
    floor = 2.0 / n_samples
    return [(floor, delta) for delta in effect_sizes]


#: (n_samples, dmr_length) combinations Task 3 asks for "in combination with the
#: above, not only in isolation" -- the small-sample-size axis crossed with the
#: extended-strata axis, not tested alone the way SCENARIOS["small_n"] already does.
SMALL_N_VALUES: tuple[int, ...] = (12, 20)
MIN_CPGS_DMR_LENGTH = 4  # Task 3: "DMR length at min_cpgs exactly"


def _layout_kwargs(layout: str, seed: int, **extra) -> dict:
    kwargs = dict(layout=layout, seed=seed, **extra)
    if layout == "wgbs_real":
        kwargs["layout_offset"] = int(np.random.default_rng(seed).integers(0, 1_000_000))
    else:
        kwargs["layout_profile"] = "chr1"
    return kwargs


# --------------------------------------------------------------------------
# generic parallel cell-grid driver (Stage 1g Task B.4)
# --------------------------------------------------------------------------


def _cell_id(subcommand: str, key: tuple) -> str:
    digest = hashlib.md5(repr(key).encode()).hexdigest()[:10]
    readable = "_".join(str(k) for k in key)[:60].replace("/", "-").replace(" ", "")
    return f"{readable}_{digest}"


def _worker_init() -> None:
    # Reasserted here (in addition to being set in the parent before the pool
    # is created) so this is correct regardless of the platform's default
    # multiprocessing start method.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    # Opt-in per _pair_statistic's design (core.py): the fused kernel is not
    # automatic even when numba is installed, since it differs from the
    # numpy path at the ~1e-15 floating-point noise floor, enough to break
    # exact-equality invariant tests elsewhere. Safe and worthwhile here
    # specifically: this sweep is not compared against a frozen fixture, and
    # single-threaded numba (NUMBA_NUM_THREADS=1, set above, so this does
    # not oversubscribe against the process pool) still measured 1.4x over
    # numpy by avoiding numpy's intermediate array allocations
    # (STAGE1G_STOP_REPORT.md SS B.1). A no-op if numba is not installed --
    # core.py's own _HAS_NUMBA check falls back to numpy regardless of this
    # variable.
    os.environ["EIGER2_USE_NUMBA"] = "1"


def _run_grid_parallel(
    subcommand: str,
    cells: list[tuple],
    cell_fn,
    *,
    jobs: int = 1,
    resume: bool = False,
) -> pd.DataFrame:
    """Run ``cell_fn(*args)`` for every ``args`` in ``cells``, one row per
    result. Each cell's result is written to its own JSON file under
    ``results_v210/cells/<subcommand>/`` as soon as it completes -- a crash
    loses at most that one cell. ``resume=True`` skips cells whose file
    already exists, loading the prior result instead of recomputing.
    """

    cell_dir = CELLS_DIR / subcommand
    cell_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[str, tuple]] = []
    rows: list[dict] = []
    for args in cells:
        cid = _cell_id(subcommand, args)
        path = cell_dir / f"{cid}.json"
        if resume and path.exists():
            rows.append(json.loads(path.read_text()))
            continue
        pending.append((cid, args))

    print(
        f"[{subcommand}] {len(rows)} cells resumed, {len(pending)} to run "
        f"(jobs={jobs})",
        flush=True,
    )

    def _write_and_collect(cid: str, result: dict) -> None:
        (cell_dir / f"{cid}.json").write_text(json.dumps(result, default=str))
        rows.append(result)
        print(subcommand, result, flush=True)

    if jobs <= 1:
        for cid, args in pending:
            _write_and_collect(cid, cell_fn(*args))
    else:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["NUMBA_NUM_THREADS"] = "1"
        with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init) as pool:
            futures = {
                pool.submit(cell_fn, *args): cid for cid, args in pending
            }
            for future in as_completed(futures):
                _write_and_collect(futures[future], future.result())

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. DMR-free calibration: 100 seeds x 5 scenarios x 2 layouts
# --------------------------------------------------------------------------


def _calibration_cell(layout: str, scenario: str, seed: int) -> dict:
    kwargs = _layout_kwargs(
        layout, seed, strata=[], replicates=0, spacing=CALIB_SPACING,
        **SCENARIOS[scenario],
    )
    matrix, locations, _ = simulate(**kwargs)
    config = Eiger2Config(
        window=4, null_min_lag=1000, random_seed=seed, max_cpg_gap_bp=300,
    )
    t0 = time.time()
    _, _, manifest = call_dmrs(matrix, locations, config=config)
    elapsed = time.time() - t0
    gates = manifest["gates"]
    # Gate 2 (region_exchangeability) needs at least one candidate region to
    # compare against the null; the "no_regions" early return carries Gate 1
    # only -- .get() rather than direct indexing so a DMR-free cell that
    # happens to call nothing doesn't crash this grid.
    region_exch = gates.get("region_exchangeability")
    return dict(
        layout=layout, scenario=scenario, seed=seed,
        n_cpg=len(locations), runtime_seconds=elapsed,
        n_candidates=int(manifest.get("n_regions", 0)),
        acf_passed=gates["acf"]["passed"],
        region_exchangeability_ratio=(
            region_exch["region_exchangeability_ratio"] if region_exch else float("nan")
        ),
        null_count_adequate=(
            region_exch["null_count_adequate"] if region_exch else None
        ),
        s=gates.get("s"),
        overdispersion_ratio=manifest.get("fdr", {}).get(
            "null_count_overdispersion_warning"
        ),
    )


def dmr_free_calibration(
    seeds: range, *, jobs: int = 1, resume: bool = False,
) -> pd.DataFrame:
    """Extends Stage 1e's 2-seed calibration_member_step.csv to the full
    100-seed design. Identical per-cell logic to that run (same manifest
    fields), just more seeds.
    """
    cells = [
        (layout, scenario, seed)
        for layout in LAYOUTS for scenario in SCENARIO_NAMES for seed in seeds
    ]
    return _run_grid_parallel(
        "calibration", cells, _calibration_cell, jobs=jobs, resume=resume,
    )


# --------------------------------------------------------------------------
# 1b. Multi-chromosome cell (spec §2b.5) -- DEFINED, NOT RUN as part of
# Stage 2b. Execution belongs to §2f, alongside the rest of this grid.
# --------------------------------------------------------------------------

#: Real hg38, EVERY chromosome present at once (layout_chrom=None -- spec
#: §2b.5), not sliced to one as every cell above and every benchmark ever
#: run before Stage 2b always was (that is precisely why the genome-wide
#: lag defect survived seven review stages). Spacing chosen so `n_cpg`
#: comfortably clears member_step's ~103,000-CpG floor at defaults, so this
#: cell exercises the actual production null_replicates=100.
MULTICHROM_SPACING = 60_000  # n_cpg = spacing*1 + 400, strata=[] (DMR-free)


def _multichrom_cell(scenario: str, seed: int) -> dict:
    """The pytest-scale checks ``tests/test_chromosome_aware_null.py``
    (spec §2b.5) pins, at full production scale and crossed with every
    nuisance scenario, not just one seed at one scale: surviving null
    tracks (>= 90% of null_replicates), null coverage, chromosome-exposure
    (spec §2b.3), the composition check (each chromosome's own share of
    pooled null positions vs. its share of eligible observed edges, within
    a factor of 1.2 -- the check a cap-only fix would fail), and whether
    every structural gate (spec §2a.2) passes.
    """
    kwargs = dict(
        seed=seed, strata=[], replicates=0, spacing=MULTICHROM_SPACING,
        layout="wgbs_real", layout_chrom=None, **SCENARIOS[scenario],
    )
    matrix, locations, _ = simulate(**kwargs)
    config = Eiger2Config(
        window=4, null_min_lag=1000, random_seed=seed, max_cpg_gap_bp=300,
    )
    t0 = time.time()
    _, _, manifest = call_dmrs(matrix, locations, config=config)
    elapsed = time.time() - t0
    structural = manifest["gates"]["structural"]

    # Composition check: not exposed by call_dmrs's own manifest (which
    # reports the pooled null, not a per-chromosome breakdown of it), so
    # recomputed directly here from null_tracks() -- the same construction
    # call_dmrs itself makes internally, not a second null.
    residuals = centre_cpgs(matrix)
    tracks = null_tracks(residuals, config, locations)
    null_positions_by_chromosome: dict[str, int] = {}
    for _track, eligible, segments in tracks:
        position = 0
        for name, _lag, _start, length in segments:
            segment_eligible = eligible[position : position + length]
            null_positions_by_chromosome[name] = (
                null_positions_by_chromosome.get(name, 0)
                + int(segment_eligible.sum())
            )
            position += length
    feasibility = chromosome_lag_feasibility(locations, config)
    total_edges = sum(v["eligible_observed_edges"] for v in feasibility.values())
    total_null = sum(null_positions_by_chromosome.values())
    composition_ratio_by_chromosome = {
        name: (
            (null_positions_by_chromosome.get(name, 0) / total_null)
            / (v["eligible_observed_edges"] / total_edges)
        ) if total_null and total_edges and v["eligible_observed_edges"] else None
        for name, v in feasibility.items()
    }
    composition_within_factor_1_2 = all(
        ratio is not None and 1 / 1.2 <= ratio <= 1.2
        for ratio in composition_ratio_by_chromosome.values()
    )

    return dict(
        scenario=scenario, seed=seed, n_cpg=len(locations),
        chromosome_order=manifest.get("chromosome_order"),
        runtime_seconds=elapsed,
        surviving_null_tracks_fraction=structural["surviving_null_tracks"]["fraction"],
        null_coverage=structural["null_coverage"]["value"],
        chromosome_exposure_missing=structural["chromosome_exposure"]["missing_exposure"],
        composition_ratio_by_chromosome=composition_ratio_by_chromosome,
        composition_within_factor_1_2=composition_within_factor_1_2,
        structural_gates_all_passed=all(g["passed"] for g in structural.values()),
    )


def multichrom(seeds: range, *, jobs: int = 1, resume: bool = False) -> pd.DataFrame:
    cells = [(scenario, seed) for scenario in SCENARIO_NAMES for seed in seeds]
    return _run_grid_parallel(
        "multichrom", cells, _multichrom_cell, jobs=jobs, resume=resume,
    )


# --------------------------------------------------------------------------
# 2. Extended planted-signal grid
# --------------------------------------------------------------------------


def _signal_grid_cells_for(layout: str, seed: int) -> list[tuple]:
    grid_cells: list[tuple[str, list[tuple[float, float]], int, int]] = [
        ("default_n80_len10", DEFAULT_STRATA, 80, 10),
        ("small_delta_n80_len10", SMALL_DELTA_STRATA, 80, 10),
        ("small_delta_n80_len4", SMALL_DELTA_STRATA, 80, MIN_CPGS_DMR_LENGTH),
    ]
    for n_samples in SMALL_N_VALUES:
        floor_strata = two_carrier_floor_strata(n_samples)
        grid_cells.append(
            (f"low_prevalence_n{n_samples}_len10", floor_strata, n_samples, 10)
        )
        grid_cells.append((
            f"small_delta_n{n_samples}_len{MIN_CPGS_DMR_LENGTH}",
            SMALL_DELTA_STRATA, n_samples, MIN_CPGS_DMR_LENGTH,
        ))
    return [
        (layout, cell_name, strata, n_samples, dmr_length, seed)
        for cell_name, strata, n_samples, dmr_length in grid_cells
    ]


def _signal_grid_cell(
    layout: str, cell_name: str, strata: list[tuple[float, float]],
    n_samples: int, dmr_length: int, seed: int,
) -> dict:
    kwargs = _layout_kwargs(
        layout, seed, strata=strata, replicates=3, spacing=GENOME_SPACING,
        dmr_length=dmr_length, n_samples=n_samples, n_latent=2,
        latent_loading_sd=0.07, noise_sd=0.04, missing_rate=0.0,
    )
    matrix, locations, truth = simulate(**kwargs)
    config = Eiger2Config(
        window=4, null_min_lag=1000, random_seed=seed,
        max_cpg_gap_bp=300, min_cpgs=4,
    )
    t0 = time.time()
    regions, _, manifest = call_dmrs(matrix, locations, config=config)
    elapsed = time.time() - t0
    called = _v2_called(regions)
    tp, fp, recovered = _classify(called, truth, min_overlap=4)
    n_called = tp + fp
    return dict(
        layout=layout, cell=cell_name, seed=seed, n_samples=n_samples,
        dmr_length=dmr_length, n_cpg=len(locations), runtime_seconds=elapsed,
        n_truth=len(truth), n_candidates=int(manifest.get("n_regions", 0)),
        n_significant=n_called, true_positive=tp, false_positive=fp,
        realised_fdr=(fp / n_called if n_called else float("nan")),
        recall=(len(recovered) / len(truth) if truth else float("nan")),
    )


def signal_grid(seeds: range, *, jobs: int = 1, resume: bool = False) -> pd.DataFrame:
    """DEFAULT_STRATA (the easy grid, kept as a continuity baseline) plus
    SMALL_DELTA_STRATA and the two-carrier-floor strata, each also at
    n_samples in {12, 20} (Task 3: "not only in isolation") and at
    dmr_length=min_cpgs=4 as well as the standard length -- so the boundary
    cells combine small sample size, small effect, low prevalence, and
    minimum region length rather than varying one axis at a time.
    """
    cells = [
        cell
        for layout in LAYOUTS for seed in seeds
        for cell in _signal_grid_cells_for(layout, seed)
    ]
    return _run_grid_parallel(
        "signal_grid", cells, _signal_grid_cell, jobs=jobs, resume=resume,
    )


# --------------------------------------------------------------------------
# 3. Power curve: recall vs. effect size at fixed prevalence
# --------------------------------------------------------------------------

#: fixed prevalence anchor (matches DEFAULT_STRATA's (0.25, 0.25) point, the
#: middle of the existing grid, so the curve is comparable to it); one stratum
#: at a time (not mixed) so recall-per-delta is cleanly attributable.
POWER_CURVE_PREVALENCE = 0.25
POWER_CURVE_DELTAS: tuple[float, ...] = (
    0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35,
)


def _power_curve_cell(layout: str, delta: float, seed: int) -> dict:
    kwargs = _layout_kwargs(
        layout, seed,
        strata=[(POWER_CURVE_PREVALENCE, delta)], replicates=10,
        spacing=GENOME_SPACING, dmr_length=10, n_samples=80,
        n_latent=2, latent_loading_sd=0.07, noise_sd=0.04,
        missing_rate=0.0,
    )
    matrix, locations, truth = simulate(**kwargs)
    config = Eiger2Config(
        window=4, null_min_lag=1000, random_seed=seed,
        max_cpg_gap_bp=300, min_cpgs=4,
    )
    regions, _, manifest = call_dmrs(matrix, locations, config=config)
    called = _v2_called(regions)
    tp, fp, recovered = _classify(called, truth, min_overlap=4)
    n_called = tp + fp
    return dict(
        layout=layout, prevalence=POWER_CURVE_PREVALENCE, delta=delta,
        seed=seed, n_truth=len(truth),
        recall=(len(recovered) / len(truth) if truth else float("nan")),
        realised_fdr=(fp / n_called if n_called else float("nan")),
    )


def power_curve(seeds: range, *, jobs: int = 1, resume: bool = False) -> pd.DataFrame:
    cells = [
        (layout, delta, seed)
        for layout in LAYOUTS for delta in POWER_CURVE_DELTAS for seed in seeds
    ]
    return _run_grid_parallel(
        "power_curve", cells, _power_curve_cell, jobs=jobs, resume=resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subcommand",
        choices=[
            "calibration", "signal_grid", "power_curve", "gap_sweep", "multichrom",
        ],
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="parallel worker processes; 1 (default) runs serially with no "
             "new dependency. Every cell is independent.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="skip cells whose per-cell result file already exists under "
             "results_v210/cells/<subcommand>/",
    )
    args = parser.parse_args()

    # Set before any ProcessPoolExecutor worker is spawned: each worker is a
    # fresh interpreter that imports numpy/numba only after inheriting this
    # environment, so BLAS/numba pick up the single-thread setting at their
    # own initialisation rather than after the fact.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    # Opt-in per _pair_statistic's design (core.py): the fused kernel is not
    # automatic even when numba is installed, since it differs from the
    # numpy path at the ~1e-15 floating-point noise floor, enough to break
    # exact-equality invariant tests elsewhere. Safe and worthwhile here
    # specifically: this sweep is not compared against a frozen fixture, and
    # single-threaded numba (NUMBA_NUM_THREADS=1, set above, so this does
    # not oversubscribe against the process pool) still measured 1.4x over
    # numpy by avoiding numpy's intermediate array allocations
    # (STAGE1G_STOP_REPORT.md SS B.1). A no-op if numba is not installed --
    # core.py's own _HAS_NUMBA check falls back to numpy regardless of this
    # variable.
    os.environ["EIGER2_USE_NUMBA"] = "1"

    t0 = time.time()
    if args.subcommand == "calibration":
        frame = dmr_free_calibration(range(args.seeds), jobs=args.jobs, resume=args.resume)
        frame.to_csv(OUT / "calibration_member_step_full.csv", index=False)
    elif args.subcommand == "signal_grid":
        frame = signal_grid(range(args.seeds), jobs=args.jobs, resume=args.resume)
        frame.to_csv(OUT / "signal_grid_extended.csv", index=False)
    elif args.subcommand == "power_curve":
        frame = power_curve(range(args.seeds), jobs=args.jobs, resume=args.resume)
        frame.to_csv(OUT / "power_curve.csv", index=False)
    elif args.subcommand == "multichrom":
        frame = multichrom(range(args.seeds), jobs=args.jobs, resume=args.resume)
        frame.to_csv(OUT / "multichrom.csv", index=False)
    else:
        frame = gap_sweep(list(range(args.seeds)))
        frame.to_csv(OUT / "gap_sweep_full.csv", index=False)
    print(f"wrote {len(frame)} rows in {time.time() - t0:.0f}s", flush=True)
