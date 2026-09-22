"""Coordinated-region discovery with a per-CpG permutation region FDR.

This is EIGER's supported discovery method. One run does three things:

1. **Observed scan.** A caller profile turns the CpG-by-sample matrix into an
   adjacent-edge statistic, sets a scan threshold, and segments runs of
   passing, eligible edges into candidate regions. Each candidate's score is
   ``S = sum over its edges of (statistic - scan threshold)``.
2. **Complete permutation scans.** For each of ``B`` outer replicates the
   sample values are permuted independently *within every CpG row*, and the
   permuted matrix is passed through exactly the same scan, including the
   recomputation of the scan threshold. Permuted rows are shared by every
   window that reads them; finished statistics are never shuffled.
3. **Region FDR.** At every distinct observed score ``t``::

       R(t)        = observed regions with score >= t
       N_b(t)      = regions with score >= t in complete permutation scan b
       FDR_plus(t) = min(1, (1 + sum_b N_b(t)) / (B * max(R(t), 1)))

   and a region scoring ``S`` receives ``q(S) = min FDR_plus(t)`` over the
   distinct observed scores ``t <= S`` (an ascending cumulative minimum). Ties
   share counts and q-values. The added count is a safeguard against a zero
   estimate, not the exact permutation p-value theorem, and the q-values are
   never passed through Benjamini-Hochberg.

A region is selected when ``coordination_qvalue <= target_fdr``. All candidate
regions of one run -- the whole analysed genomic scope, for one caller profile
-- form one decision family. There is no stratification by chromosome, length
or exposure.

Two caller profiles are supported:

``newer`` (default)
    Per-CpG centring, no deflation, window-four pairwise-complete Pearson
    correlation (at least five paired samples), scan threshold = the 0.995
    quantile of a distant-lag null (eight internal lag draws, minimum lag 1,000
    retained CpG rows), regions of at least four CpGs, no failing-edge bridges,
    edges eligible only on one chromosome with ``0 <= next.start - prev.end <=
    300`` bp.
``legacy``
    The original covariance statistic (``functions.processing.
    covariance_blocks(pairwise=False)``) on row-mean imputed values (an
    all-missing row is filled with 0.5), a fixed threshold of 0.01, and the same
    eligibility, segmentation and score as ``newer``. This is the legacy
    statistic inside the tested wrapper, not every historical legacy CLI
    behaviour: the old delta/prevalence criteria are not computed.

The scan, permutation and estimator are ports of the frozen validation
implementation (``vfdr/callers.py``, ``permute.py``, ``fdr.py``,
``streamed.py``, ``pipeline2.py``); the internal seed schedule is the
reference's: observed scan seed ``random_seed`` (default 1) and outer replicate
``b`` (numbered from 1) seed ``random_seed + b``. A separate
``permutation_seed`` drives the one generator that permutes rows.
"""

from __future__ import annotations

import math
import resource
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .core import (
    Eiger2Config,
    _chromosome_blocks,
    _lag_null_track,
    _null_eligibility,
    _standardise_locations,
    centre_cpgs,
    chromosome_lag_feasibility,
    chromosome_sort_permutation,
    edge_statistic,
    eligible_edges,
    null_threshold,
    region_area,
    segment,
)

__all__ = [
    "DEFAULT_FDR_REPLICATES",
    "FDR_FORMULA",
    "FDR_FORMULA_VERSION",
    "LEGACY",
    "MASK_MODES",
    "NEWER",
    "PROFILES",
    "RUN_SCHEMA",
    "SCAN_MODES",
    "CallerProfile",
    "CandidateThresholdError",
    "CoordinationFdr",
    "PermutationScanError",
    "RegionFdrConfig",
    "RegionFdrRun",
    "ScanResult",
    "call_coordinated_regions",
    "coordination_fdr",
    "permute_rows",
    "permute_rows_into",
    "scan",
    "scan_by_chromosome",
]

RUN_SCHEMA = "eiger.region_fdr_run/1"
FDR_FORMULA_VERSION = "fdr_plus_cummin/1"
FDR_FORMULA = (
    "FDR_plus(t) = min(1, (1 + sum_b N_b(t)) / (B * max(R(t), 1))); "
    "q(S) = min FDR_plus(t) over distinct observed scores t <= S"
)
DEFAULT_FDR_REPLICATES = 99
MASK_MODES = ("values_with_mask", "mask_fixed")
SCAN_MODES = ("chromosome", "whole")
STORAGE_DTYPES = ("float64", "float32")
AXIS_MODES = ("descriptive", "off", "experimental_inference")


class CandidateThresholdError(ValueError):
    """The scan threshold cannot be computed for this input and profile."""


class PermutationScanError(RuntimeError):
    """A permutation scan failed; the run has no valid FDR and is aborted."""


# --------------------------------------------------------------------------
# caller profiles
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CallerProfile:
    """Frozen caller settings. Everything data-dependent is recomputed per scan.

    Field for field the reference ``vfdr.callers.Profile``; the defaults of
    :data:`NEWER` and :data:`LEGACY` are the reference profiles
    ``newer_nodeflate`` and ``legacy_cov010``.
    """

    name: str
    kind: str                      # "legacy" | "newer"
    min_cpgs: int = 4
    max_gap_edges: int = 0
    max_cpg_gap_bp: int | None = 300
    # --- legacy only
    cov_threshold: float = 0.01
    impute_row_means: bool = True
    # --- newer only
    statistic: str = "correlation"
    window: int = 4
    null_quantile: float = 0.995
    null_replicates: int = 8
    null_min_lag: int = 1000
    min_paired_samples: int = 5
    min_paired_fraction: float = 0.0
    deflate_components: int | None = 0   # 0 == deflation disabled

    def validate(self) -> None:
        if self.kind not in {"newer", "legacy"}:
            raise ValueError("profile kind must be 'newer' or 'legacy'")
        if self.min_cpgs < 2:
            raise ValueError("min_cpgs must be at least 2")
        if self.max_gap_edges != 0:
            raise ValueError(
                "max_gap_edges must be 0: failing-edge bridges are not part of "
                "the supported method"
            )
        if self.max_cpg_gap_bp is not None and self.max_cpg_gap_bp < 0:
            raise ValueError("max_cpg_gap_bp must be non-negative")
        if self.deflate_components != 0:
            raise ValueError(
                "deflation is not part of the supported method; "
                "deflate_components must be 0"
            )
        if self.kind == "newer":
            if self.statistic not in {"correlation", "covariance"}:
                raise ValueError("statistic must be 'correlation' or 'covariance'")
            if self.window < 1:
                raise ValueError("window must be at least 1")
            if not 0.0 < self.null_quantile < 1.0:
                raise ValueError("null_quantile must lie in (0, 1)")
            if self.null_replicates < 1:
                raise ValueError("null_replicates must be at least 1")
            if self.null_min_lag < 1:
                raise ValueError("null_min_lag must be at least 1")
            if self.min_paired_samples < 2:
                raise ValueError("min_paired_samples must be at least 2")
        elif not math.isfinite(self.cov_threshold):
            raise ValueError("cov_threshold must be finite")

    def reference_default(self) -> CallerProfile:
        return NEWER if self.kind == "newer" else LEGACY

    def is_reference_default(self) -> bool:
        return replace(self, name=self.reference_default().name) == (
            self.reference_default()
        )


NEWER = CallerProfile(name="newer", kind="newer")
LEGACY = CallerProfile(name="legacy", kind="legacy", cov_threshold=0.01)
PROFILES = {NEWER.name: NEWER, LEGACY.name: LEGACY}
REFERENCE_PROFILE_NAMES = {"newer": "newer_nodeflate", "legacy": "legacy_cov010"}


def eiger2_config(profile: CallerProfile, seed: int = 1) -> Eiger2Config:
    """The :class:`Eiger2Config` the newer profile's library calls read."""

    return Eiger2Config(
        statistic=profile.statistic,
        window=profile.window,
        deflate_components=profile.deflate_components,
        max_cpg_gap_bp=profile.max_cpg_gap_bp,
        min_paired_samples=profile.min_paired_samples,
        min_paired_fraction=profile.min_paired_fraction,
        null_quantile=profile.null_quantile,
        null_replicates=profile.null_replicates,
        null_min_lag=profile.null_min_lag,
        random_seed=seed,
    )


def _legacy_config(profile: CallerProfile) -> Eiger2Config:
    """Only used for ``eligible_edges`` -- the gap/chromosome geometry."""

    return Eiger2Config(
        statistic="covariance", window=1, deflate_components=0,
        max_cpg_gap_bp=profile.max_cpg_gap_bp,
    )


def _legacy_processing():
    try:
        from functions import processing
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the legacy profile runs the original implementation in "
            "functions/processing.py, which needs numba and scikit-learn; "
            "install them with: pip install numba scikit-learn (or install "
            "the EIGER wheel with its [legacy] extra)"
        ) from exc
    return processing


# --------------------------------------------------------------------------
# one complete scan
# --------------------------------------------------------------------------


@dataclass
class ScanResult:
    regions: np.ndarray        # (n, 2) inclusive CpG index bounds
    areas: np.ndarray          # region score: summed excess over the threshold
    threshold: float
    n_candidate_edges: int
    n_eligible_edges: int
    statistic: np.ndarray | None = None
    diagnostics: dict[str, Any] | None = None


def _nanmean_quiet(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _mean_impute(matrix: np.ndarray) -> np.ndarray:
    """Row-mean imputation; an all-missing row is filled with 0.5."""

    work = np.array(matrix, dtype=float, copy=True)
    # numpy warns "mean of empty slice" for an all-missing row; its NaN mean is
    # replaced by 0.5 on the next line, so the warning carries no information.
    row_means = _nanmean_quiet(np.where(np.isfinite(work), work, np.nan), axis=1)
    row_means = np.where(np.isfinite(row_means), row_means, 0.5)
    gaps = ~np.isfinite(work)
    work[gaps] = np.take(row_means, np.nonzero(gaps)[0])
    return work


def _finish(
    stat: np.ndarray,
    eligible: np.ndarray,
    threshold: float,
    profile: CallerProfile,
    keep_statistic: bool,
    diagnostics: dict,
) -> ScanResult:
    regions = segment(
        stat, eligible, threshold,
        min_cpgs=profile.min_cpgs, max_gap_edges=profile.max_gap_edges,
    )
    areas = region_area(stat, regions, threshold)
    passing = int(np.sum(np.isfinite(stat) & (stat >= threshold) & eligible))
    return ScanResult(
        regions=regions, areas=np.asarray(areas, float), threshold=float(threshold),
        n_candidate_edges=passing, n_eligible_edges=int(eligible.sum()),
        statistic=stat if keep_statistic else None,
        diagnostics=diagnostics,
    )


def scan(
    betas: np.ndarray,
    locations: pd.DataFrame,
    profile: CallerProfile,
    *,
    seed: int = 1,
    keep_statistic: bool = False,
) -> ScanResult:
    """One complete scan of a whole matrix (reference ``vfdr.callers.scan``).

    Identical code path for observed and permuted matrices; the only thing a
    caller may vary between them is ``betas`` (and the internal ``seed``).
    """

    if profile.kind == "legacy":
        cfg = _legacy_config(profile)
        work = (
            _mean_impute(betas) if profile.impute_row_means
            else np.asarray(betas, float)
        )
        stat = np.asarray(
            _legacy_processing().covariance_blocks(work, pairwise=False), float
        )
        threshold: float = float(profile.cov_threshold)
        diagnostics: dict[str, Any] = {"threshold_source": "fixed_constant"}
    elif profile.kind == "newer":
        cfg = eiger2_config(profile, seed=seed)
        residuals = centre_cpgs(np.asarray(betas, float))
        stat = edge_statistic(residuals, cfg)
        try:
            threshold, diag = null_threshold(residuals, cfg, locations)
        except ValueError as exc:
            raise CandidateThresholdError(_infeasible_message(locations, cfg)) from exc
        diagnostics = {
            "threshold_source": "lag_null_quantile",
            "null_values": diag["null_values"],
            "null_coverage": diag["null_coverage"],
            "null_replicates": diag["null_replicates"],
        }
    else:  # pragma: no cover - validated upstream
        raise ValueError(profile.kind)

    eligible = eligible_edges(locations, cfg)
    return _finish(stat, eligible, threshold, profile, keep_statistic, diagnostics)


def _draw_lags(blocks, feasibility, cfg) -> list[dict[str, int]]:
    """``null_tracks``' lag draws, in its order, before any chromosome is read."""

    rng = np.random.default_rng(cfg.random_seed + 977)
    return [
        {name: int(rng.integers(cfg.null_min_lag, feasibility[name]["l_max"] + 1))
         for name, _s, _e in blocks}
        for _ in range(cfg.null_replicates)
    ]


def _infeasible_message(locations: pd.DataFrame, cfg: Eiger2Config) -> str:
    feasibility = chromosome_lag_feasibility(locations, cfg)
    longest = max((v["n_cpg"] for v in feasibility.values()), default=0)
    span = max(cfg.window, 2)
    needed = cfg.null_min_lag * (span - 1) + 2
    return (
        "the distant-lag candidate threshold cannot be computed: no chromosome "
        f"offers a usable lag of at least {cfg.null_min_lag:,} CpG rows "
        f"(longest chromosome has {longest:,} retained CpGs; the newer profile "
        f"needs at least {needed:,} on some chromosome, with enough observed "
        "pairs). Supply more CpGs per chromosome, or use --legacy, whose fixed "
        "threshold needs no lag null."
    )


def scan_by_chromosome(
    provider: Callable[[int, int], np.ndarray],
    locations: pd.DataFrame,
    profile: CallerProfile,
    *,
    seed: int = 1,
    keep_statistic: bool = False,
) -> ScanResult:
    """The same scan, one chromosome at a time (reference ``scan_streamed``).

    ``provider(start, stop)`` returns rows ``start:stop`` of the matrix as
    float64. One genome-wide candidate threshold and one genome-wide decision
    family are kept; only the order of work changes, so the whole float64
    residual matrix never exists at once.

    Reproduced from the whole-matrix scan: the lag draws (``null_tracks``'
    ``default_rng(seed + 977)`` sequence, drawn before any chromosome is read),
    the window support (each chromosome's statistic is computed on its rows
    padded by ``max(window, 2)`` rows on each side, then trimmed -- the padding
    crosses chromosome boundaries exactly as the monolithic window does), and
    the null (the library's own lag-track and eligibility functions on the
    unpadded chromosome slice). The inter-chromosome edge itself is set to NaN;
    it is ineligible in both paths, so no region or score depends on it.
    Agreement with :func:`scan` is to floating-point round-off in the
    statistic; thresholds and regions agree exactly on the reference checks.
    """

    n_cpg = len(locations)
    chrom = locations["chr"].to_numpy()
    all_blocks = _chromosome_blocks(chrom)

    if profile.kind == "legacy":
        cfg = _legacy_config(profile)
        eligible = eligible_edges(locations, cfg)
        processing = _legacy_processing()
        stat = np.full(n_cpg - 1, np.nan)
        for _name, start, end in all_blocks:
            work = provider(start, end)
            if profile.impute_row_means:
                work = _mean_impute(work)
            piece = np.asarray(
                processing.covariance_blocks(work, pairwise=False), float
            )
            stat[start:end - 1] = piece[:end - 1 - start]
            del work, piece
        threshold = float(profile.cov_threshold)
        diagnostics: dict[str, Any] = {
            "threshold_source": "fixed_constant", "streamed": True,
        }
    elif profile.kind == "newer":
        cfg = eiger2_config(profile, seed=seed)
        eligible = eligible_edges(locations, cfg)
        feasibility = chromosome_lag_feasibility(
            locations, cfg, observed_eligible=eligible
        )
        blocks = [(n, s, e) for n, s, e in all_blocks if feasibility[n]["feasible"]]
        lags = _draw_lags(blocks, feasibility, cfg)
        pad = max(cfg.window, 2)

        stat = np.full(n_cpg - 1, np.nan)
        pooled: list[list[np.ndarray]] = [[] for _ in lags]
        offered = 0
        for name, start, end in all_blocks:
            lo, hi = max(start - pad, 0), min(end + pad, n_cpg)
            residuals = centre_cpgs(provider(lo, hi))
            padded_stat = edge_statistic(residuals, cfg)
            stat[start:end - 1] = padded_stat[start - lo:end - 1 - lo]
            del padded_stat
            if feasibility[name]["feasible"]:
                core = residuals[start - lo:end - lo]
                obs_elig_c = eligible[start:max(end - 1, start)]
                for rep, lag_map in enumerate(lags):
                    piece = _lag_null_track(core, lag_map[name], cfg)
                    if not piece.size:
                        continue
                    piece_elig = _null_eligibility(
                        None, lag_map[name], cfg, piece.size,
                        observed_eligible=obs_elig_c,
                    )
                    if not piece_elig.any():
                        continue
                    piece_elig = piece_elig.copy()
                    piece_elig[-1] = False
                    offered += int(piece.size)
                    kept = piece[piece_elig]
                    pooled[rep].append(kept[np.isfinite(kept)])
                    del piece, piece_elig, kept
                del core
            del residuals
        values = [v for rep in pooled for v in rep if v.size]
        if not values:
            raise CandidateThresholdError(_infeasible_message(locations, cfg))
        flat = np.concatenate(values)
        threshold = float(np.quantile(flat, cfg.null_quantile))
        diagnostics = {
            "threshold_source": "lag_null_quantile", "streamed": True,
            "null_values": int(flat.size),
            "null_positions_offered": int(offered),
            "null_coverage": float(flat.size / offered) if offered else 0.0,
            "null_replicates": int(sum(1 for rep in pooled if rep)),
            "n_feasible_chromosomes": len(blocks),
            "n_chromosomes": len(all_blocks),
        }
        del flat, values, pooled
    else:  # pragma: no cover - validated upstream
        raise ValueError(profile.kind)

    return _finish(stat, eligible, threshold, profile, keep_statistic, diagnostics)


# --------------------------------------------------------------------------
# the permutation
# --------------------------------------------------------------------------


def permute_rows(
    betas: np.ndarray, rng: np.random.Generator, mask_mode: str = "values_with_mask"
) -> np.ndarray:
    """Independent within-row permutation of the sample axis (reference).

    ``values_with_mask`` (default): the whole row -- values and their
        missingness -- is permuted together. Assumes samples are exchangeable
        within a CpG including their observation pattern.
    ``mask_fixed``: missing positions stay put and only the observed values
        are permuted among the observed positions. Assumes exchangeability
        conditional on the observation mask.
    """

    x = np.asarray(betas, dtype=float)
    if mask_mode == "values_with_mask":
        return rng.permuted(x, axis=1)
    if mask_mode != "mask_fixed":
        raise ValueError(f"mask_mode must be one of {MASK_MODES}")

    observed = np.isfinite(x)
    if observed.all():
        return rng.permuted(x, axis=1)
    return _mask_fixed_rows(x, observed, rng)


def _mask_fixed_rows(
    x: np.ndarray, observed: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    n_rows, n_cols = x.shape
    keys = np.where(observed, rng.random(x.shape), np.inf)
    take = np.argsort(keys, axis=1, kind="stable")          # observed cols, shuffled
    put = np.argsort(~observed, axis=1, kind="stable")      # observed cols, in order
    counts = observed.sum(axis=1)
    slots = np.arange(n_cols)[None, :] < counts[:, None]
    rows = np.repeat(np.arange(n_rows)[:, None], n_cols, axis=1)
    out = np.full_like(x, np.nan)
    out[rows[slots], put[slots]] = x[rows[slots], take[slots]]
    return out


def permute_rows_into(
    base: np.ndarray,
    out: np.ndarray,
    rng: np.random.Generator,
    mask_mode: str = "values_with_mask",
    *,
    chunk_rows: int = 200_000,
    all_observed: bool | None = None,
) -> None:
    """:func:`permute_rows` written into a preallocated buffer, in row chunks.

    Draw-for-draw identical to ``permute_rows(base, rng, mask_mode)`` on the
    whole matrix: the generator is consumed row by row in the same order
    either way. For ``mask_fixed`` the fully-observed shortcut is decided once
    for the WHOLE matrix (``all_observed``), as the reference does, so a
    fully-observed chunk inside a matrix with gaps still draws sort keys.
    """

    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}")
    if out.shape != base.shape:
        raise ValueError("out must have the same shape as base")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be at least 1")
    n_rows = base.shape[0]
    shortcut = mask_mode == "values_with_mask"
    if not shortcut:
        if all_observed is None:
            all_observed = bool(np.isfinite(base).all())
        shortcut = all_observed
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        if shortcut:
            rng.permuted(base[start:stop], axis=1, out=out[start:stop])
        else:
            block = np.asarray(base[start:stop], dtype=float)
            out[start:stop] = _mask_fixed_rows(block, np.isfinite(block), rng)


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------


@dataclass
class CoordinationFdr:
    thresholds: np.ndarray        # unique observed scores, ascending
    r_at: np.ndarray              # R(t)
    null_sum_at: np.ndarray       # sum_b N_b(t)
    fdr_plus_at: np.ndarray
    qvalue: np.ndarray            # per observed region, FDR_plus cumulative minimum
    n_permutations: int = 0
    null_counts_per_perm: np.ndarray = field(
        default_factory=lambda: np.empty(0, int)
    )


def coordination_fdr(
    observed_scores: np.ndarray,
    null_scores_per_perm: list[np.ndarray],
    *,
    n_permutations: int | None = None,
) -> CoordinationFdr:
    """``FDR_plus`` and its q-values (reference ``vfdr.fdr.region_fdr``).

    Only observed scores need be examined: between two consecutive observed
    scores ``R(t)`` is constant while ``sum_b N_b(t)`` is non-increasing, so
    the minimum over that gap is attained at its upper end, itself an observed
    score. Equal scores share both the discovery count and the q-value.
    """

    observed = np.asarray(observed_scores, dtype=float)
    b = int(n_permutations if n_permutations is not None else len(null_scores_per_perm))
    if b <= 0:
        raise ValueError("at least one complete permutation scan is required")
    if len(null_scores_per_perm) != b:
        raise ValueError(
            f"{len(null_scores_per_perm)} permutation score sets were supplied for "
            f"B = {b}; every permutation scan must be complete"
        )
    if observed.size and not np.all(np.isfinite(observed)):
        raise ValueError("observed region scores must be finite")

    pooled = (
        np.concatenate([np.asarray(a, float) for a in null_scores_per_perm])
        if null_scores_per_perm else np.empty(0, float)
    )
    counts_per_perm = np.array([len(a) for a in null_scores_per_perm], dtype=int)

    if observed.size == 0:
        empty = np.empty(0, float)
        return CoordinationFdr(empty, empty.astype(int), empty, empty, empty,
                               b, counts_per_perm)

    thresholds = np.unique(observed)                       # ascending
    sorted_obs = np.sort(observed)
    sorted_null = np.sort(pooled)
    r_at = observed.size - np.searchsorted(sorted_obs, thresholds, side="left")
    null_sum_at = sorted_null.size - np.searchsorted(
        sorted_null, thresholds, side="left"
    )
    denom = b * np.maximum(r_at, 1)
    fdr_plus_at = np.minimum(1.0, (1.0 + null_sum_at) / denom)
    # q at threshold u_k = min over u_1..u_k: every threshold that includes a
    # region scoring u_k. Higher score, lower or equal q.
    q_plus_at = np.minimum.accumulate(fdr_plus_at)
    index = np.searchsorted(thresholds, observed, side="left")
    return CoordinationFdr(
        thresholds=thresholds, r_at=r_at, null_sum_at=null_sum_at,
        fdr_plus_at=fdr_plus_at, qvalue=q_plus_at[index],
        n_permutations=b, null_counts_per_perm=counts_per_perm,
    )


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionFdrConfig:
    """Resolved settings of one coordination-FDR run.

    ``fdr_replicates`` is the number ``B`` of complete outer permutation scans
    (default 99). ``profile.null_replicates`` is a different thing: the eight
    internal distant-lag draws that set each scan's candidate threshold.
    """

    profile: CallerProfile = NEWER
    target_fdr: float = 0.05
    fdr_replicates: int = DEFAULT_FDR_REPLICATES
    random_seed: int = 1
    permutation_seed: int = 1
    mask_mode: str = "values_with_mask"
    scan_mode: str = "chromosome"
    storage_dtype: str = "float64"
    permutation_chunk_rows: int = 200_000
    min_sample_fraction: float = 0.0
    discovery_only: bool = False
    axis_mode: str = "descriptive"
    axis_target_fdr: float = 0.05

    def validate(self) -> None:
        self.profile.validate()
        if not 0.0 < self.target_fdr < 1.0:
            raise ValueError("target_fdr must lie in (0, 1)")
        if isinstance(self.fdr_replicates, bool) or not isinstance(
            self.fdr_replicates, (int, np.integer)
        ):
            raise ValueError("fdr_replicates must be an integer")
        if not self.discovery_only and self.fdr_replicates < 1:
            raise ValueError(
                "fdr_replicates must be at least 1 (default 99); use discovery "
                "only mode for candidates without q-values"
            )
        if self.mask_mode not in MASK_MODES:
            raise ValueError(f"mask_mode must be one of {MASK_MODES}")
        if self.scan_mode not in SCAN_MODES:
            raise ValueError(f"scan_mode must be one of {SCAN_MODES}")
        if self.storage_dtype not in STORAGE_DTYPES:
            raise ValueError(f"storage_dtype must be one of {STORAGE_DTYPES}")
        if self.permutation_chunk_rows < 1:
            raise ValueError("permutation_chunk_rows must be at least 1")
        if not 0.0 <= self.min_sample_fraction <= 1.0:
            raise ValueError("min_sample_fraction must lie in [0, 1]")
        if self.axis_mode not in AXIS_MODES:
            raise ValueError(f"axis_mode must be one of {AXIS_MODES}")
        if not 0.0 < self.axis_target_fdr < 1.0:
            raise ValueError("axis_target_fdr must lie in (0, 1)")

    def to_dict(self) -> dict:
        out = asdict(self)
        out["profile"] = asdict(self.profile)
        return out


@dataclass
class RegionFdrRun:
    regions: pd.DataFrame          # every candidate, primary + axis columns
    manifest: dict
    permutations: pd.DataFrame     # one row per completed outer replicate
    null_scores: list[np.ndarray]  # region scores of every permutation scan

    @property
    def discoveries(self) -> pd.DataFrame:
        selected = self.regions["coordination_selected"].astype("boolean")
        return self.regions[selected.fillna(False).to_numpy(dtype=bool)].reset_index(
            drop=True
        )


PRIMARY_COLUMNS = [
    "region_id", "chr", "start", "end", "n_cpg",
    "cpg_index_start", "cpg_index_end", "caller",
    "region_score", "statistic_mean", "scan_threshold",
    "coordination_qvalue", "coordination_selected",
]


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / 2**20 if sys.platform == "darwin" else value / 1024)


def _plain(value):
    """Convert numpy scalars/arrays inside diagnostics to JSON-ready values."""

    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _prepare(betas, locations, config: RegionFdrConfig):
    matrix = np.asarray(betas)
    if matrix.ndim != 2:
        raise ValueError("betas must be a two-dimensional CpG-by-sample matrix")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("betas must be numeric")
    n_cpg, n_samples = matrix.shape
    frame = _standardise_locations(locations, n_cpg)
    permutation, chromosome_order = chromosome_sort_permutation(frame)
    frame = frame.iloc[permutation].reset_index(drop=True)
    # Row counts and the range check without a full-size copy of the values.
    observed_input = np.isfinite(matrix).sum(axis=1)
    if observed_input.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low, high = float(np.nanmin(matrix)), float(np.nanmax(matrix))
        if low < -1e-9 or high > 1 + 1e-9:
            raise ValueError(
                "betas must be methylation proportions in [0, 1]; values range "
                f"from {low:.3f} to {high:.3f} (divide percentages by 100)"
            )
    observed_per_row = observed_input[permutation]
    keep = np.ones(n_cpg, dtype=bool)
    if config.min_sample_fraction > 0:
        keep = observed_per_row >= math.ceil(config.min_sample_fraction * n_samples)
    dtype = np.dtype(config.storage_dtype)
    if keep.all() and np.array_equal(permutation, np.arange(n_cpg)):
        # Already in canonical order with nothing filtered: use the caller's
        # matrix as is (no copy when it is already C-contiguous in `dtype`).
        # Nothing in the pipeline writes to it.
        base = np.ascontiguousarray(matrix, dtype=dtype)
    else:
        base = np.ascontiguousarray(matrix[permutation[keep]], dtype=dtype)
    frame = frame.loc[keep].reset_index(drop=True)
    source_index = np.flatnonzero(keep)
    input_summary = {
        "n_cpg_input": int(n_cpg),
        "n_cpg_analysed": int(base.shape[0]),
        "n_samples": int(n_samples),
        "n_cpg_dropped_min_sample_fraction": int((~keep).sum()),
        "min_sample_fraction": float(config.min_sample_fraction),
        "n_cpg_all_missing_analysed": int(np.sum(observed_per_row[keep] == 0)),
        "missing_fraction_analysed": (
            float(1.0 - observed_per_row[keep].sum() / base.size) if base.size else 0.0
        ),
        "chromosome_order": [str(c) for c in chromosome_order],
        "n_chromosomes_analysed": int(frame["chr"].nunique()),
        "input_order_changed_by_sort": bool(
            not np.array_equal(permutation, np.arange(n_cpg))
        ),
        "storage_dtype": config.storage_dtype,
    }
    return base, frame, source_index, input_summary


def _missingness_contract(profile: CallerProfile, mask_mode: str) -> dict:
    if profile.kind == "newer":
        statistic = (
            "pairwise-complete per edge; an edge needs at least "
            f"{profile.min_paired_samples} co-observed samples, else it is NaN "
            "and breaks a region"
        )
    elif profile.impute_row_means:
        statistic = (
            "row-mean imputation before the complete-case covariance; an "
            "all-missing row is filled with 0.5"
        )
    else:
        statistic = "complete-case covariance without imputation (non-default)"
    permutation = {
        "values_with_mask": (
            "values and their missingness are permuted together within each "
            "CpG row; assumes samples are exchangeable including observation "
            "pattern"
        ),
        "mask_fixed": (
            "missing positions stay fixed and only observed values are "
            "permuted among observed positions; assumes exchangeability "
            "conditional on the observation mask (alternative, not the default)"
        ),
    }[mask_mode]
    return {"statistic": statistic, "permutation": permutation,
            "mask_mode": mask_mode}


def _region_table(
    obs: ScanResult,
    frame: pd.DataFrame,
    source_index: np.ndarray,
    profile: CallerProfile,
    qvalue: np.ndarray | None,
    target_fdr: float,
    statistic: np.ndarray | None,
) -> pd.DataFrame:
    regions = np.asarray(obs.regions, dtype=int).reshape(-1, 2)
    n = len(regions)
    first, last = regions[:, 0], regions[:, 1]
    chrom = frame["chr"].to_numpy()
    starts = frame["start"].to_numpy(dtype=np.int64)
    ends = frame["end"].to_numpy(dtype=np.int64)
    prefix = "EIGER1" if profile.kind == "legacy" else "EIGER2"
    stat_mean = np.full(n, np.nan)
    if statistic is not None:
        for i, (a, b) in enumerate(regions):
            block = statistic[a:b]
            block = block[np.isfinite(block)]
            stat_mean[i] = float(block.mean()) if block.size else np.nan
    table = pd.DataFrame({
        "region_id": [
            f"{prefix}_{chrom[a]}_{int(starts[a])}_{int(ends[b])}"
            for a, b in regions
        ],
        "chr": chrom[first].astype(str) if n else np.empty(0, dtype=object),
        "start": starts[first] if n else np.empty(0, dtype=np.int64),
        "end": ends[last] if n else np.empty(0, dtype=np.int64),
        "n_cpg": (last - first + 1).astype(np.int64),
        "cpg_index_start": source_index[first] if n else np.empty(0, dtype=np.int64),
        "cpg_index_end": source_index[last] if n else np.empty(0, dtype=np.int64),
        "caller": profile.name,
        "region_score": obs.areas,
        "statistic_mean": stat_mean,
        "scan_threshold": float(obs.threshold),
    })
    if qvalue is None:
        table["coordination_qvalue"] = np.full(n, np.nan)
        table["coordination_selected"] = pd.array([pd.NA] * n, dtype="boolean")
    else:
        table["coordination_qvalue"] = qvalue
        table["coordination_selected"] = pd.array(qvalue <= target_fdr, dtype="boolean")
    return table


def call_coordinated_regions(
    betas,
    locations,
    *,
    sample_names: list[str] | None = None,
    config: RegionFdrConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> RegionFdrRun:
    """Run the supported method end to end on one matrix.

    ``betas`` is CpG-by-sample, proportions in ``[0, 1]`` with NaN for missing;
    ``locations`` has ``chr``, ``start`` and optionally ``end`` per row. Rows
    are grouped by chromosome in canonical order and sorted by position before
    anything else; ``cpg_index_start``/``cpg_index_end`` index that sorted
    input (before any ``min_sample_fraction`` filtering).

    Returns every candidate region (``regions``), the run manifest, the
    per-permutation summary and the permutation region scores. Raises
    :class:`CandidateThresholdError` when the newer profile's lag threshold
    cannot be computed, and :class:`PermutationScanError` when any permutation
    scan fails -- a failed scan is never counted as a scan with no regions.
    """

    config = config or RegionFdrConfig()
    config.validate()
    profile = config.profile
    say = progress or (lambda _message: None)
    started = time.perf_counter()

    base, frame, source_index, input_summary = _prepare(betas, locations, config)
    n_cpg, n_samples = base.shape
    if sample_names is not None and len(sample_names) != n_samples:
        raise ValueError("sample_names must match the matrix columns")
    minimum_samples = profile.min_paired_samples if profile.kind == "newer" else 2
    if n_samples < minimum_samples:
        raise ValueError(
            f"at least {minimum_samples} samples are required; got {n_samples}"
        )
    if n_cpg < 2:
        raise ValueError("at least two CpGs are required after filtering")

    def provider(source: np.ndarray) -> Callable[[int, int], np.ndarray]:
        return lambda a, b: np.asarray(source[a:b], dtype=float)

    def run_scan(source: np.ndarray, seed: int, keep_statistic: bool = False):
        if config.scan_mode == "chromosome":
            return scan_by_chromosome(
                provider(source), frame, profile, seed=seed,
                keep_statistic=keep_statistic,
            )
        return scan(np.asarray(source, dtype=float), frame, profile, seed=seed,
                    keep_statistic=keep_statistic)

    say(f"observed scan: {n_cpg:,} CpGs x {n_samples} samples, profile "
        f"{profile.name}")
    t0 = time.perf_counter()
    observed = run_scan(base, config.random_seed, keep_statistic=True)
    t_observed = time.perf_counter() - t0
    n_candidates = int(len(observed.regions))
    say(f"observed scan: {n_candidates:,} candidate regions, scan threshold "
        f"{observed.threshold:.6g} ({t_observed:.1f}s)")

    # Secondary axis annotations: estimated on the pre-FDR candidate family of
    # this observed scan, before and independently of any permutation. They
    # read the analysed matrix and nothing downstream reads them back.
    from .axis import annotate_axes, empty_axis_columns

    t_axis = 0.0
    if config.axis_mode != "off" and n_candidates:
        say(f"axis annotations ({config.axis_mode}) on {n_candidates:,} "
            "candidate regions")
        t1 = time.perf_counter()
        axis_columns, axis_manifest = annotate_axes(
            base, observed.regions, mode=config.axis_mode,
            random_seed=config.random_seed, target_fdr=config.axis_target_fdr,
        )
        t_axis = time.perf_counter() - t1
    else:
        axis_columns, axis_manifest = empty_axis_columns(
            n_candidates, mode=config.axis_mode
        )

    null_scores: list[np.ndarray] = []
    per_perm: list[dict] = []
    status = "ok"
    t_null = 0.0
    run_permutations = not config.discovery_only and n_candidates > 0
    if config.discovery_only:
        status = "discovery_only"
    elif n_candidates == 0:
        status = "no_candidates"
    if run_permutations:
        rng = np.random.default_rng(config.permutation_seed)
        buffer = np.empty_like(base)
        all_observed = (
            bool(np.isfinite(base).all()) if config.mask_mode == "mask_fixed" else None
        )
        b_total = int(config.fdr_replicates)
        t1 = time.perf_counter()
        last_report = 0.0
        for b in range(1, b_total + 1):
            internal_seed = config.random_seed + b
            try:
                permute_rows_into(
                    base, buffer, rng, config.mask_mode,
                    chunk_rows=config.permutation_chunk_rows,
                    all_observed=all_observed,
                )
                result = run_scan(buffer, internal_seed)
            except Exception as exc:
                raise PermutationScanError(
                    f"permutation scan {b} of {b_total} failed "
                    f"({type(exc).__name__}: {exc}); no FDR is reported for an "
                    "incomplete set of permutation scans"
                ) from exc
            null_scores.append(np.asarray(result.areas, float))
            per_perm.append({
                "replicate": b,
                "internal_seed": internal_seed,
                "scan_threshold": float(result.threshold),
                "n_candidate_edges": int(result.n_candidate_edges),
                "n_regions": int(len(result.regions)),
                "null_values": (result.diagnostics or {}).get("null_values"),
            })
            elapsed = time.perf_counter() - t1
            if b == 1 or b == b_total or elapsed - last_report >= 30.0:
                last_report = elapsed
                eta = elapsed / b * (b_total - b)
                say(f"permutation {b}/{b_total}: {len(result.regions):,} regions, "
                    f"elapsed {elapsed:.0f}s, remaining ~{eta:.0f}s, peak RSS "
                    f"{_peak_rss_mb():,.0f} MB")
        del buffer
        t_null = time.perf_counter() - t1
        if len(null_scores) != b_total:  # pragma: no cover - loop raises first
            raise PermutationScanError("permutation scans are incomplete")

    fdr = None
    qvalue = None
    if run_permutations:
        fdr = coordination_fdr(observed.areas, null_scores,
                               n_permutations=int(config.fdr_replicates))
        qvalue = fdr.qvalue
    elif status == "no_candidates" and not config.discovery_only:
        qvalue = np.empty(0, dtype=float)

    regions = _region_table(
        observed, frame, source_index, profile, qvalue, config.target_fdr,
        observed.statistic,
    )
    for column, values in axis_columns.items():
        regions[column] = values

    n_selected = (
        int(regions["coordination_selected"].fillna(False).sum())
        if qvalue is not None else None
    )
    thresholds = np.array([p["scan_threshold"] for p in per_perm], dtype=float)
    ratio: dict[str, Any] = {"observed_threshold": float(observed.threshold)}
    if thresholds.size:
        mean_perm = float(thresholds.mean())
        ratio["mean_permuted_threshold"] = mean_perm
        ratio["sd_permuted_threshold"] = (
            float(thresholds.std(ddof=1)) if thresholds.size > 1 else 0.0
        )
        if math.isfinite(mean_perm) and mean_perm > 0:
            ratio["observed_over_mean_permuted"] = float(observed.threshold / mean_perm)
        else:
            ratio["observed_over_mean_permuted"] = None
            ratio["undefined_reason"] = (
                "mean permuted threshold is not a positive finite number"
            )
    else:
        ratio["observed_over_mean_permuted"] = None
        ratio["undefined_reason"] = "no permutation scans were run"
    ratio["note"] = (
        "descriptive only; no warning or decision depends on this ratio"
    )

    counts = np.array([p["n_regions"] for p in per_perm], dtype=int)
    manifest: dict[str, Any] = {
        "run_schema": RUN_SCHEMA,
        "software_version": __version__,
        "status": status,
        "method": {
            "name": "per-CpG permutation region FDR",
            "fdr_formula": FDR_FORMULA,
            "fdr_formula_version": FDR_FORMULA_VERSION,
            "pseudocount": 1,
            "q_rule": "ascending cumulative minimum of FDR_plus over distinct "
                      "observed scores; ties share counts and q-values; no BH",
            "region_score": "sum over region edges of (statistic - scan threshold)",
            "decision": "coordination_qvalue <= target_fdr",
            "decision_family": (
                "every candidate region of this run's observed scan across the "
                "whole analysed genomic scope, for this caller profile; no "
                "chromosome, length or exposure strata"
            ),
            "permutation": "sample values permuted independently within each "
                           "CpG row; each permuted matrix rescanned completely, "
                           "including the scan threshold",
            "automatic_deflation": False,
        },
        "profile": asdict(profile),
        "profile_is_reference_default": profile.is_reference_default(),
        "reference_profile": REFERENCE_PROFILE_NAMES[profile.kind],
        "settings": {
            "target_fdr": float(config.target_fdr),
            "fdr_replicates_requested": int(config.fdr_replicates),
            "random_seed": int(config.random_seed),
            "permutation_seed": int(config.permutation_seed),
            "internal_seed_schedule": (
                "observed scan: random_seed; outer replicate b (1-based): "
                "random_seed + b; lag draws inside a scan use seed + 977"
            ),
            "mask_mode": config.mask_mode,
            "scan_mode": config.scan_mode,
            "storage_dtype": config.storage_dtype,
            "permutation_chunk_rows": int(config.permutation_chunk_rows),
            "discovery_only": bool(config.discovery_only),
            "axis_mode": config.axis_mode,
        },
        "input": input_summary,
        "missingness": _missingness_contract(profile, config.mask_mode),
        "observed_scan": {
            "scan_threshold": float(observed.threshold),
            "n_candidate_edges": int(observed.n_candidate_edges),
            "n_eligible_edges": int(observed.n_eligible_edges),
            "n_candidate_regions": n_candidates,
            "diagnostics": _plain(observed.diagnostics or {}),
        },
        "permutations": {
            "requested": 0 if not run_permutations else int(config.fdr_replicates),
            "completed": len(per_perm),
            "not_run_reason": (
                None if run_permutations
                else ("discovery_only requested" if config.discovery_only
                      else "no observed candidate regions")
            ),
            "null_regions_total": int(counts.sum()) if counts.size else 0,
            "null_regions_per_permutation": counts.tolist(),
            "scan_thresholds": thresholds.tolist(),
            "per_permutation_table": "permutations.csv",
        },
        "threshold_ratio": ratio,
        "results": {
            "n_candidates": n_candidates,
            "n_selected": n_selected,
            "candidate_retention_fraction": (
                float(n_selected / n_candidates)
                if n_selected is not None and n_candidates else None
            ),
            "qvalue_min": (
                float(np.min(qvalue)) if qvalue is not None and qvalue.size else None
            ),
            "qvalues_available": qvalue is not None,
        },
        "axis": axis_manifest,
        "timing": {
            "observed_scan_s": t_observed,
            "axis_s": t_axis,
            "permutations_s": t_null,
            "per_permutation_s": t_null / len(per_perm) if per_perm else None,
            "total_s": time.perf_counter() - started,
        },
        "peak_rss_mb": _peak_rss_mb(),
    }
    if fdr is not None:
        manifest["fdr_curve"] = {
            "n_distinct_scores": int(fdr.thresholds.size),
            "stored": "not stored; recompute from region_score and the "
                      "permutation scores with coordination_fdr()",
        }
    permutations = pd.DataFrame(
        per_perm,
        columns=["replicate", "internal_seed", "scan_threshold",
                 "n_candidate_edges", "n_regions", "null_values"],
    )
    return RegionFdrRun(regions=regions, manifest=manifest,
                        permutations=permutations, null_scores=null_scores)
