"""Stage 1d-C evaluation script: C.3 (member-step lag null) against the two gates,
the edge-level gate, null adequacy, and recall.

Not wired into ``benchmark_suite.py``'s CLI and not a production
``region_null_mode`` -- this stage's own instruction is "a recommendation with a
table, not an integration" (STAGE1D_C_COMPRESSED_SPEC.md sec6). Kept here, runnable
directly (``python stage1d_c_c3_evaluation.py``), so the numbers cited in
STAGE1D_C_STOP_REPORT.md can be reproduced rather than taken on faith.

Requires a full repo checkout (the vendored fixture); see
``eiger2.benchmark._fixture_dir``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "layouts"))

import numpy as np
import pandas as pd
from layouts import generate_epic_layout, load_wgbs_layout

from eiger2.benchmark import evaluate, simulate
from eiger2.core import (
    Eiger2Config,
    acf_gate,
    centre_cpgs,
    deflate,
    edge_statistic,
    eligible_edges,
    estimate_latent_basis,
    length_stratum,
    member_step_lag_null_ensemble,
    null_exchangeability,
    null_threshold,
    null_tracks,
    region_area,
    region_exchangeability_gate,
    region_fdr_result,
    segment,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "layouts"

SCENARIOS: dict[str, dict] = {
    "clean": dict(n_samples=80, n_latent=0, noise_sd=0.04, missing_rate=0.0),
    "nuisance": dict(n_samples=80, n_latent=2, latent_loading_sd=0.07, noise_sd=0.04, missing_rate=0.0),
    "missingness": dict(n_samples=80, n_latent=2, latent_loading_sd=0.07, noise_sd=0.04, missing_rate=0.25),
    "small_n": dict(n_samples=12, n_latent=1, latent_loading_sd=0.07, noise_sd=0.04, missing_rate=0.0),
    "high_noise": dict(n_samples=40, n_latent=2, latent_loading_sd=0.10, noise_sd=0.08, missing_rate=0.0),
}
DEFAULT_STRATA = [(0.05, 0.35), (0.10, 0.30), (0.25, 0.25), (0.50, 0.20)]

#: dense stays at the pre-existing continuity-only scale (matches STOP 1d-B);
#: the two real layouts are sized so member-step sub-tracks are genome-scale, per
#: spec sec3's explicit warning ("evaluate C.3 on the real fixture at full
#: chromosome scale, not on a 20k slice").
LAYOUT_N_CPG = {"dense": 20_400, "wgbs_real": 600_000, "epic_fitted": 600_000}


def _dmr_free_data(layout: str, scenario_name: str, seed: int):
    kwargs = dict(SCENARIOS[scenario_name])
    n_samples = kwargs.pop("n_samples")
    n_latent = kwargs.pop("n_latent")
    latent_loading_sd = kwargs.pop("latent_loading_sd", 0.07)
    noise_sd = kwargs.pop("noise_sd")
    missing_rate = kwargs.pop("missing_rate")
    n_cpg = LAYOUT_N_CPG[layout]
    rng = np.random.default_rng(seed)

    if layout == "dense":
        locations = pd.DataFrame(
            {"chr": "chr1", "start": np.arange(n_cpg) * 2, "end": np.arange(n_cpg) * 2 + 1}
        )
    elif layout == "wgbs_real":
        offset = int(np.random.default_rng(seed).integers(0, 1_500_000))
        locations, _ = load_wgbs_layout(
            FIXTURE_DIR / "wgbs_layout_hg38.npz", chrom="chr1", n_cpg=n_cpg, offset=offset,
        )
    else:
        locations, _ = generate_epic_layout(
            FIXTURE_DIR / "epic_layout_params.json", n_cpg, seed=seed, profile="chr1",
        )
    n_cpg = len(locations)
    matrix = 0.5 + rng.normal(0, noise_sd, (n_cpg, n_samples))
    if n_latent:
        latent = rng.normal(0, 1, (n_samples, n_latent))
        loading = rng.normal(0, latent_loading_sd, (n_cpg, n_latent))
        matrix += loading @ latent.T
    matrix = np.clip(matrix, 0.001, 0.999)
    if missing_rate > 0:
        propensity = np.clip(rng.beta(1.2, 1.2, (n_cpg, 1)) * 2 * missing_rate, 0, 0.95)
        matrix[rng.random(matrix.shape) < propensity] = np.nan
    return matrix, locations


def measure_gates(layout: str, scenario_name: str, seed: int, *, n_lag_draws: int = 8) -> dict:
    """Gate 1, Gate 2, the edge-level gate, and null adequacy for C.3."""

    matrix, locations = _dmr_free_data(layout, scenario_name, seed)
    config = Eiger2Config(window=4, null_min_lag=1000, random_seed=seed, max_cpg_gap_bp=300)
    residuals = centre_cpgs(matrix)
    basis, _ = estimate_latent_basis(residuals, config)
    deflated = deflate(residuals, basis, missing=config.deflation_missing)
    statistic = edge_statistic(deflated, config)
    eligible = eligible_edges(locations, config)

    tracks, lags = member_step_lag_null_ensemble(
        deflated, config, locations, n_lag_draws=n_lag_draws
    )
    if not tracks:
        return {"layout": layout, "scenario": scenario_name, "seed": seed, "status": "no_null_tracks"}

    edge_gate = null_exchangeability(tracks, statistic, eligible, config)
    acf_result = acf_gate(statistic, eligible, tracks, config.window)

    threshold, _ = null_threshold(
        deflated, config, locations, tracks=tracks, statistic=statistic, eligible=eligible,
    )
    observed_regions = segment(
        statistic, eligible, threshold, min_cpgs=config.min_cpgs, max_gap_edges=config.max_gap_edges,
    )
    null_regions_total = 0
    null_edge_total = 0
    null_by_stratum: dict[str, int] = {}
    for track, sub_eligible, _off in tracks:
        usable = sub_eligible & np.isfinite(track)
        null_edge_total += int(usable.sum())
        found = segment(
            track, usable, threshold, min_cpgs=config.min_cpgs, max_gap_edges=config.max_gap_edges,
        )
        null_regions_total += len(found)
        for first, last in found:
            key = length_stratum(last - first + 1)
            null_by_stratum[key] = null_by_stratum.get(key, 0) + 1

    observed_by_stratum: dict[str, int] = {}
    for first, last in observed_regions:
        key = length_stratum(last - first + 1)
        observed_by_stratum[key] = observed_by_stratum.get(key, 0) + 1

    region_gate = region_exchangeability_gate(
        observed_regions=len(observed_regions),
        observed_eligible_edges=int(np.sum(eligible & np.isfinite(statistic))),
        null_regions_total=null_regions_total,
        null_eligible_edges=null_edge_total,
        min_null_regions_for_fdr=config.min_null_regions_for_fdr,
    )
    lens = [t.size for t, _, _ in tracks]
    return {
        "layout": layout, "scenario": scenario_name, "seed": seed, "status": "ok",
        "n_cpg": len(locations), "n_subtracks": len(tracks), "lags_used": lags,
        "subtrack_len_min": int(min(lens)),
        "subtrack_len_median": int(sorted(lens)[len(lens) // 2]),
        "subtrack_len_max": int(max(lens)),
        **{f"edge_{k}": v for k, v in edge_gate.items()},
        "acf_passed": acf_result["passed"], "acf_diff": acf_result["diff"],
        "region_exchangeability_ratio": region_gate["region_exchangeability_ratio"],
        "region_null_count_adequate": region_gate["null_count_adequate"],
        "n_pooled_null_regions": null_regions_total,
        "n_candidate_observed_regions": len(observed_regions),
        "null_regions_by_stratum": null_by_stratum,
        "observed_regions_by_stratum": observed_by_stratum,
        "s_scale": region_gate["observed_eligible_edges"] / max(null_edge_total, 1),
    }


def _call_with_null(matrix, locations, tracks, config, target_fdr=0.05):
    """A deliberately simplified (non-support-conditioned) FDR path, applied
    IDENTICALLY to both nulls being compared, so the comparison isolates the null
    construction's effect on recall rather than machinery differences from
    call_dmrs's full Stage 1c pipeline."""

    residuals = centre_cpgs(matrix)
    basis, _ = estimate_latent_basis(residuals, config)
    deflated = deflate(residuals, basis, missing=config.deflation_missing)
    statistic = edge_statistic(deflated, config)
    eligible = eligible_edges(locations, config)
    threshold, _ = null_threshold(
        deflated, config, locations, tracks=tracks, statistic=statistic, eligible=eligible,
    )
    observed_regions = segment(
        statistic, eligible, threshold, min_cpgs=config.min_cpgs, max_gap_edges=config.max_gap_edges,
    )
    if observed_regions.size == 0:
        return observed_regions, np.array([])
    observed_areas = region_area(statistic, observed_regions, threshold)
    null_areas, null_lengths, null_edge_total = [], [], 0
    for track, sub_eligible, _ in tracks:
        usable = sub_eligible & np.isfinite(track)
        null_edge_total += int(usable.sum())
        found = segment(
            track, usable, threshold, min_cpgs=config.min_cpgs, max_gap_edges=config.max_gap_edges,
        )
        null_areas.append(region_area(track, found, threshold))
        null_lengths.append(
            (found[:, 1] - found[:, 0] + 1) if found.size else np.empty(0, dtype=int)
        )
    observed_lengths = observed_regions[:, 1] - observed_regions[:, 0] + 1
    observed_edges = int(np.sum(eligible & np.isfinite(statistic)))
    fdr_result = region_fdr_result(
        observed_areas, null_areas, observed_edges=observed_edges,
        null_edges=max(null_edge_total, 1),
        observed_lengths=observed_lengths, null_length_sets=null_lengths,
        min_null_regions_for_fdr=config.min_null_regions_for_fdr,
    )
    return observed_regions, fdr_result.qvalue <= target_fdr


def _to_frame(observed_regions, significant):
    return pd.DataFrame({
        "cpg_index_start": observed_regions[:, 0] if observed_regions.size else [],
        "cpg_index_end": observed_regions[:, 1] if observed_regions.size else [],
        "significant": significant if len(significant) else [],
    })


def measure_recall(layout: str, seed: int, *, n_lag_draws: int = 8) -> dict:
    """Criterion 4: recall on planted signal, C.3 vs the shipped null.

    ``min_null_regions_for_fdr=0``: this isolates the null CONSTRUCTION's effect
    on recall from the separate, already-measured Stage 1c-bis null-adequacy
    guard (which both nulls trip at these replicate counts on DMR-free data --
    the guard is not what this criterion is about).
    """

    if layout == "dense":
        matrix, locations, truth = simulate(
            strata=DEFAULT_STRATA, replicates=3, spacing=400, dmr_length=10,
            seed=seed, layout="dense",
        )
    elif layout == "wgbs_real":
        matrix, locations, truth = simulate(
            strata=DEFAULT_STRATA, replicates=3, spacing=1000, dmr_length=10, seed=seed,
            layout="wgbs_real",
            layout_offset=int(np.random.default_rng(seed).integers(0, 1_000_000)),
        )
    else:
        matrix, locations, truth = simulate(
            strata=DEFAULT_STRATA, replicates=3, spacing=1000, dmr_length=10, seed=seed,
            layout="epic_fitted", layout_profile="chr1",
        )
    config = Eiger2Config(
        window=4, null_min_lag=1000, random_seed=seed, max_cpg_gap_bp=300,
        min_cpgs=4, min_null_regions_for_fdr=0,
    )

    shipped_tracks = null_tracks(centre_cpgs(matrix), config, locations)
    shipped_regions, shipped_sig = _call_with_null(matrix, locations, shipped_tracks, config)
    shipped_recovered, shipped_false = evaluate(
        _to_frame(shipped_regions, shipped_sig), truth, min_overlap=4,
    )

    residuals = centre_cpgs(matrix)
    basis, _ = estimate_latent_basis(residuals, config)
    deflated = deflate(residuals, basis, missing=config.deflation_missing)
    c3_tracks, _lags = member_step_lag_null_ensemble(
        deflated, config, locations, n_lag_draws=n_lag_draws,
    )
    c3_regions, c3_sig = _call_with_null(matrix, locations, c3_tracks, config)
    c3_recovered, c3_false = evaluate(_to_frame(c3_regions, c3_sig), truth, min_overlap=4)

    n_truth = len(truth)
    return {
        "layout": layout, "seed": seed, "n_truth": n_truth,
        "n_shipped_candidates": len(shipped_regions),
        "shipped_recall": len(shipped_recovered) / n_truth if n_truth else float("nan"),
        "shipped_false": shipped_false,
        "n_c3_candidates": len(c3_regions),
        "c3_recall": len(c3_recovered) / n_truth if n_truth else float("nan"),
        "c3_false": c3_false,
        "n_dmrs_not_plantable": locations.attrs.get("n_dmrs_not_plantable"),
    }


if __name__ == "__main__":
    t0 = time.time()
    gate_rows = []
    for layout in ("dense", "wgbs_real", "epic_fitted"):
        for scenario in SCENARIOS:
            for seed in range(3):
                r = measure_gates(layout, scenario, seed)
                gate_rows.append(r)
                print(layout, scenario, seed, "->", r.get("status"), r.get("acf_passed"),
                      r.get("region_exchangeability_ratio"), r.get("n_pooled_null_regions"),
                      "subtrack_med=", r.get("subtrack_len_median"), flush=True)
    recall_rows = []
    for layout in ("dense", "wgbs_real", "epic_fitted"):
        for seed in range(3):
            r = measure_recall(layout, seed)
            recall_rows.append(r)
            print(r, flush=True)
    print("TOTAL TIME", time.time() - t0)
    out = Path(__file__).resolve().parent / "results_stage1d_c"
    out.mkdir(exist_ok=True)
    (out / "c3_gates.json").write_text(json.dumps(gate_rows, indent=2, default=str))
    (out / "c3_recall.json").write_text(json.dumps(recall_rows, indent=2, default=str))
    print(f"wrote {out}/c3_gates.json and {out}/c3_recall.json")
