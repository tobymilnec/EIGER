"""Layout diagnostics shipped with the installed ``eiger2`` package.

``describe_layout`` and ``eligible_run_lengths`` are required in every ``eiger2 call``
manifest (master spec §6.1) for whatever coordinates a caller supplies -- real data, not
just the benchmark fixtures -- so they live here rather than in ``benchmarking/``, which
is not installed (see ``pyproject.toml``: only ``eiger2*``/``functions*`` ship).

The benchmark-only loaders that need the large committed fixture files
(``load_wgbs_layout``, ``generate_epic_layout``) stay in
``benchmarking/layouts/layouts.py``, which re-exports the two functions below so there is
exactly one implementation -- its own module docstring calls ``describe_layout`` "the
same function the benchmark and the caller should both use". Eligibility here is computed
via :func:`eiger2.core.eligible_edges` (not re-derived) so a layout diagnostic can never
disagree with what the caller itself will use to build regions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import Eiger2Config, eligible_edges

__all__ = [
    "eligible_run_lengths", "eligible_run_spans", "describe_layout",
    "realised_length_strata", "LENGTH_STRATA", "plantability_fraction",
    "window_eligibility_fraction",
]

# Mirrors core.length_stratum()'s bin boundaries (4-5 / 6-9 / 10-19 / 20-49 / 50+).
# Not imported from there to avoid coupling this module's stratum ORDER (needed to
# find "the largest populated stratum") to that function's implementation; kept in
# sync by test_layouts.py cross-checking every boundary against length_stratum()
# directly.
LENGTH_STRATA = ("4-5", "6-9", "10-19", "20-49", "50+")
_LENGTH_STRATUM_LOWER_BOUND = {"4-5": 4, "6-9": 6, "10-19": 10, "20-49": 20, "50+": 50}


def realised_length_strata(layout_max_detectable_region_cpgs: int) -> tuple[str, ...]:
    """Which of ``length_stratum()``'s 5 canonical bins this layout can ever populate.

    Master spec §6.1: a layout's ``layout_max_detectable_region_cpgs`` ceiling
    (:func:`describe_layout`) may make some of the higher bins geometrically
    impossible -- no region can ever be that long, however many CpGs are observed,
    because no eligible run is. Those bins collapse upward into the largest bin the
    layout CAN reach, which becomes this layout's effective top stratum. Returns the
    reachable prefix of ``_LENGTH_STRATA``; empty if the layout cannot even reach the
    smallest bin (``layout_max_detectable_region_cpgs < 4``).
    """

    return tuple(
        s for s in LENGTH_STRATA
        if _LENGTH_STRATUM_LOWER_BOUND[s] <= layout_max_detectable_region_cpgs
    )


def _run_boundaries(locations: pd.DataFrame, max_cpg_gap_bp: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Shared scaffolding for both run-length and run-span views (edge-index space)."""

    eligible = eligible_edges(locations, Eiger2Config(max_cpg_gap_bp=max_cpg_gap_bp))
    breaks = np.flatnonzero(~eligible)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [eligible.size]))
    return starts, ends


def eligible_run_lengths(locations: pd.DataFrame, max_cpg_gap_bp: int | None) -> np.ndarray:
    """Length in EDGES of every maximal eligible run. Zero means an isolated CpG."""

    starts, ends = _run_boundaries(locations, max_cpg_gap_bp)
    return (ends - starts).astype(int)


def eligible_run_spans(
    locations: pd.DataFrame, max_cpg_gap_bp: int | None
) -> list[tuple[int, int]]:
    """Every maximal eligible run as an inclusive ``(start_cpg_index, end_cpg_index)``
    span, in genome order. These spans partition every CpG index in ``locations``
    exactly once, including isolated CpGs as a one-CpG run -- there is no CpG index
    that belongs to no run. A run's CpG count is ``end - start + 1``; its length in
    edges (:func:`eligible_run_lengths`'s unit) is ``end - start``.

    Used to plant DMRs that respect the layout (master spec §6.1): a DMR cannot span
    an ineligible edge, so it must fit entirely inside one span.
    """

    starts, ends = _run_boundaries(locations, max_cpg_gap_bp)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def plantability_fraction(
    locations: pd.DataFrame, max_cpg_gap_bp: int | None, dmr_length: int
) -> float:
    """Fraction of CpG-index start positions where a ``dmr_length``-CpG DMR is NOT
    plantable, i.e. does not fit entirely inside one eligible run (master spec §6.1).

    Complete and position-agnostic: every CpG index is treated as a candidate start,
    not a thinned sample along some spacing-based grid (that is what
    ``eiger2.benchmark._plant_dmrs`` measures for one specific run instead, via
    ``locations.attrs["n_dmrs_not_plantable"]`` -- a report of what a specific
    simulated grid actually hit, not this function's exhaustive "how much of the grid
    even COULD be hit" answer). A run of CpG count ``C`` admits ``max(0, C -
    dmr_length + 1)`` valid start positions; this is one minus their sum over
    ``len(locations)``.
    """

    if len(locations) == 0:
        return float("nan")
    spans = eligible_run_spans(locations, max_cpg_gap_bp)
    plantable = sum(max(0, (end - start + 1) - dmr_length + 1) for start, end in spans)
    return 1.0 - plantable / len(locations)


def window_eligibility_fraction(
    locations: pd.DataFrame, max_cpg_gap_bp: int | None, window: int
) -> float:
    """Fraction of ELIGIBLE edges at which a ``window``-CpG window is geometrically
    eligible (master spec §7, the WP2 MultiScale measurement; corrected 20 Aug 2026 --
    see ``docs/MULTISCALE_BLOCKED.md``).

    Mirrors ``core.edge_statistic``'s own window construction exactly, not a looser
    "does *some* window fit somewhere in this edge's run" existence check: for edge
    ``i``, the window's base CpG index is ``i - (0 if window == 1 else window // 2 -
    1)`` (``core.call_dmrs``'s ``obs_offset0``), and the window is geometrically
    eligible there iff every one of the ``window - 1`` sub-edges it spans, starting at
    that base, is itself gap-eligible -- i.e. the whole window lies inside ONE eligible
    run, at the SPECIFIC alignment EIGER actually uses, not merely inside a run that
    happens to be long enough somewhere. An earlier version used the looser criterion
    and overstated eligibility (e.g. chr1 w=4: 97.2% loose vs 87.5% alignment-aware);
    reasoning from "the run has enough CpGs" without checking the specific window
    alignment EIGER computes is the same class of error as reasoning from a per-run
    summary when the per-edge quantity is the operative one.

    The denominator is eligible edges specifically (an ineligible edge is never a
    candidate for any window regardless of ``window``); the numerator additionally
    requires the edge itself be eligible, which it is by construction of the
    denominator.
    """

    eligible = eligible_edges(locations, Eiger2Config(max_cpg_gap_bp=max_cpg_gap_bp))
    n_edges = eligible.size
    denom = int(eligible.sum())
    if denom == 0:
        return float("nan")

    offset0 = 0 if window == 1 else window // 2 - 1
    base = np.arange(n_edges) - offset0
    span = window - 1  # sub-edges spanned by a `window`-CpG window
    in_bounds = (base >= 0) & (base + span <= n_edges)

    prefix = np.concatenate(([0], np.cumsum(eligible.astype(np.int64))))
    ok = np.zeros(n_edges, dtype=bool)
    idx = np.flatnonzero(in_bounds)
    b = base[idx]
    ok[idx] = (prefix[b + span] - prefix[b]) == span

    return int((ok & eligible).sum()) / denom


def describe_layout(
    locations: pd.DataFrame,
    max_cpg_gap_bp: int | None,
    depth: np.ndarray | None = None,
) -> dict:
    """Layout diagnostics for the run manifest.

    ``layout_max_detectable_region_cpgs`` is the hard ceiling on region length imposed
    by the layout: a region cannot span an ineligible edge, so the longest possible
    region is the longest eligible run plus one CpG. Compare it against ``min_cpgs`` and
    against the length strata before interpreting a length-stratified FDR.
    """

    chrom = locations["chr"].to_numpy()
    same = chrom[:-1] == chrom[1:]
    starts = locations["start"].to_numpy(dtype=np.int64)
    ends = locations["end"].to_numpy(dtype=np.int64)
    gap = starts[1:] - ends[:-1]
    eligible = eligible_edges(locations, Eiger2Config(max_cpg_gap_bp=max_cpg_gap_bp))
    runs = eligible_run_lengths(locations, max_cpg_gap_bp)
    within = gap[same]
    q = lambda a, p: float(np.quantile(a, p)) if len(a) else float("nan")  # noqa: E731

    out = {
        "layout_n_cpg": int(len(locations)),
        "layout_n_chromosomes": int(pd.unique(chrom).size),
        "layout_eligible_fraction": float(eligible.mean()) if eligible.size else float("nan"),
        "layout_gap_quantiles_bp": {
            "q10": q(within, 0.10), "q50": q(within, 0.50),
            "q90": q(within, 0.90), "q99": q(within, 0.99),
        },
        "layout_eligible_run_lengths_edges": {
            "n_runs": int(runs.size),
            "q10": q(runs, 0.10), "q50": q(runs, 0.50), "q90": q(runs, 0.90),
            "max": int(runs.max()) if runs.size else 0,
            "fraction_under_3_edges": float(np.mean(runs < 3)) if runs.size else float("nan"),
        },
        "layout_max_detectable_region_cpgs": int(runs.max()) + 1 if runs.size else 0,
    }
    if depth is not None:
        depth = np.asarray(depth)
        out["layout_depth"] = {
            "median": float(np.median(depth)),
            "fraction_ge5": float(np.mean(depth >= 5)),
            "fraction_ge10": float(np.mean(depth >= 10)),
        }
    # classification from the run-length distribution, never from a filename
    median_run = out["layout_eligible_run_lengths_edges"]["q50"]
    if median_run >= 1000:
        out["layout_class"] = "contiguous"
    elif median_run >= 20:
        out["layout_class"] = "fragmented"
    else:
        out["layout_class"] = "highly_fragmented"
    return out
