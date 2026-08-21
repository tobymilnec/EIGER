"""Operating-characteristic benchmark for EIGER 2.

Run with ``python -m eiger2.benchmark`` from the ``Python`` directory. Every
number quoted in EIGER2_DESIGN.md is produced here, so the claims can be
re-checked whenever the core changes.

Scenarios
    null            DMR-free matrix with latent structure -> false-positive rate
    prevalence      planted DMRs across a grid of subgroup fraction and effect
    missingness     the same grid with structured dropout
    statistic       correlation vs covariance under a hard, low-SNR setting
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .core import Eiger2Config, call_dmrs
from .layouts import eligible_run_spans

LAYOUTS = ("dense", "wgbs_real", "epic_fitted")


def _fixture_dir() -> Path:
    """``benchmarking/layouts/`` relative to a repo checkout.

    Only ``eiger2*``/``functions*`` ship (``pyproject.toml``); the vendored fixture
    data does not. ``layout="dense"`` (the default) never calls this, so an installed
    wheel is unaffected -- ``wgbs_real``/``epic_fitted`` are repo-checkout-only, like
    the rest of this benchmark module already is (see its module docstring).
    """

    return Path(__file__).resolve().parents[2] / "benchmarking" / "layouts"


def _layout_locations(
    layout: str,
    n_cpg: int,
    *,
    seed: int,
    layout_chrom: str,
    layout_profile: str,
    layout_offset: int | None,
) -> pd.DataFrame:
    """Build the ``chr``/``start``/``end`` frame for one of ``LAYOUTS``.

    ``dense`` (default): uniformly spaced CpGs on one synthetic chromosome -- the
    layout every prior EIGER benchmark number used, unchanged. ``wgbs_real``: a
    contiguous slice of real hg38 coordinates from the vendored fixture, at a varied
    offset per seed unless ``layout_offset`` is given explicitly -- a fixed offset
    would always sample the same (non-representative) stretch of the chromosome, see
    ``benchmarking/layouts/README.md``. ``epic_fitted``: a generated targeted-panel
    layout fitted to ``layout_profile``'s empirical gap/run-length/depth structure.
    """

    if layout == "dense":
        return pd.DataFrame(
            {"chr": "chr1", "start": np.arange(n_cpg) * 2, "end": np.arange(n_cpg) * 2 + 1}
        )
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")

    fixtures = _fixture_dir()
    root = str(fixtures.parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from benchmarking.layouts.layouts import generate_epic_layout, load_wgbs_layout

    if layout == "wgbs_real":
        npz = fixtures / "wgbs_layout_hg38.npz"
        if not npz.exists():
            raise FileNotFoundError(
                f"{npz} not found -- layout='wgbs_real' needs a full repo checkout "
                f"with the vendored fixture, not an installed wheel"
            )
        if layout_offset is None:
            full, _ = load_wgbs_layout(npz, chrom=layout_chrom)
            max_offset = max(0, len(full) - n_cpg)
            layout_offset = (
                int(np.random.default_rng(seed).integers(0, max_offset + 1))
                if max_offset > 0 else 0
            )
        locations, _depth = load_wgbs_layout(
            npz, chrom=layout_chrom, n_cpg=n_cpg, offset=layout_offset
        )
        return locations

    params = fixtures / "epic_layout_params.json"
    if not params.exists():
        raise FileNotFoundError(
            f"{params} not found -- layout='epic_fitted' needs a full repo checkout "
            f"with the vendored fixture, not an installed wheel"
        )
    locations, _depth = generate_epic_layout(
        params, n_cpg, seed=seed, profile=layout_profile, chrom=layout_chrom
    )
    return locations


def _plant_dmrs(
    matrix: np.ndarray,
    locations: pd.DataFrame,
    *,
    strata: list[tuple[float, float]],
    replicates: int,
    spacing: int,
    dmr_length: int,
    max_cpg_gap_bp: int | None,
    rng: np.random.Generator,
    n_samples: int,
    start_position: int = 200,
) -> tuple[list[tuple[int, int, float, float, int, int]], int]:
    """Plant DMRs on the sequential (position += spacing) grid used since EIGER 2's
    first benchmark, but respecting the layout (master spec §6.1): a DMR cannot span
    an ineligible edge, so each grid position is planted inside whichever eligible run
    contains it, clamped to fit, or skipped entirely if that run is shorter than
    ``dmr_length`` CpGs -- **a DMR longer than its run is not plantable**. Every
    planted truth entry records the eligible run it sits in as its trailing
    ``(run_start, run_end)`` (inclusive CpG-index span), so a consumer can tell which
    calls were even geometrically possible.

    Under ``layout="dense"`` the whole track is one eligible run at least as long as
    the grid needs, so every candidate is always plantable at its nominal
    position -- this reduces exactly to the pre-§6.1 behaviour.
    """

    spans = eligible_run_spans(locations, max_cpg_gap_bp)
    run_starts = np.array([s for s, _ in spans])
    run_ends = np.array([e for _, e in spans])
    n_cpg = len(locations)

    truth: list[tuple[int, int, float, float, int, int]] = []
    n_not_plantable = 0
    position = start_position
    for fraction, delta in strata:
        for _ in range(replicates):
            if position >= n_cpg:
                # off the end of the track entirely -- nothing there to plant in
                n_not_plantable += 1
                position += spacing
                continue
            k = int(np.searchsorted(run_ends, position))
            run_start, run_end = int(run_starts[k]), int(run_ends[k])
            if run_end - run_start + 1 < dmr_length:
                n_not_plantable += 1
                position += spacing
                continue
            actual_start = min(position, run_end - dmr_length + 1)
            actual_start = max(actual_start, run_start)
            size = max(1, int(round(fraction * n_samples)))
            group = np.zeros(n_samples)
            group[rng.choice(n_samples, size, replace=False)] = 1
            matrix[actual_start : actual_start + dmr_length] += delta * group
            truth.append((
                actual_start, actual_start + dmr_length - 1, fraction, delta,
                run_start, run_end,
            ))
            position += spacing
    return truth, n_not_plantable


def simulate(
    *,
    n_samples: int = 80,
    n_latent: int = 2,
    latent_loading_sd: float = 0.07,
    noise_sd: float = 0.04,
    strata: list[tuple[float, float]] | None = None,
    replicates: int = 6,
    spacing: int = 400,
    dmr_length: int = 40,
    missing_rate: float = 0.0,
    seed: int = 4,
    layout: str = "dense",
    max_cpg_gap_bp: int | None = 300,
    layout_chrom: str = "chr1",
    layout_profile: str = "chr1",
    layout_offset: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[tuple[int, int, float, float, int, int]]]:
    """A CpG-by-sample matrix with latent structure and planted DMRs.

    ``layout`` (master spec §6.1) selects the CpG coordinate structure: ``"dense"``
    (default, unchanged from every prior EIGER benchmark) is uniformly spaced CpGs on
    one synthetic chromosome, so every existing call is unaffected. ``"wgbs_real"`` and
    ``"epic_fitted"`` draw from the vendored ``benchmarking/layouts/`` fixture -- real
    hg38 coordinates and a fitted targeted-panel generator, respectively -- and require
    a full repo checkout (see ``_fixture_dir``). Planted DMRs always respect the
    layout's eligibility structure; see ``_plant_dmrs``.
    """

    rng = np.random.default_rng(seed)
    strata = strata if strata is not None else []
    n_dmr = len(strata) * replicates
    n_cpg = spacing * max(n_dmr, 1) + 400

    locations = _layout_locations(
        layout, n_cpg, seed=seed, layout_chrom=layout_chrom,
        layout_profile=layout_profile, layout_offset=layout_offset,
    )
    n_cpg = len(locations)  # real/fitted layouts return exactly n_cpg rows, but
                             # this keeps the matrix and locations sizes provably tied

    matrix = 0.5 + rng.normal(0, noise_sd, (n_cpg, n_samples))
    if n_latent:
        latent = rng.normal(0, 1, (n_samples, n_latent))
        loading = rng.normal(0, latent_loading_sd, (n_cpg, n_latent))
        matrix += loading @ latent.T

    truth, n_not_plantable = _plant_dmrs(
        matrix, locations, strata=strata, replicates=replicates, spacing=spacing,
        dmr_length=dmr_length, max_cpg_gap_bp=max_cpg_gap_bp, rng=rng,
        n_samples=n_samples,
    )

    matrix = np.clip(matrix, 0.001, 0.999)
    if missing_rate > 0:
        # structured dropout: a per-CpG propensity, so missingness clusters
        propensity = np.clip(
            rng.beta(1.2, 1.2, (n_cpg, 1)) * 2 * missing_rate, 0, 0.95
        )
        matrix[rng.random(matrix.shape) < propensity] = np.nan

    # DataFrame.attrs, not a 4th return value: every existing 3-tuple unpack
    # (`matrix, locations, truth = simulate(...)`) stays valid. Records how often
    # the layout constrained the planting grid (master spec §6.1).
    locations.attrs["layout"] = layout
    locations.attrs["n_dmrs_requested"] = n_dmr
    locations.attrs["n_dmrs_not_plantable"] = n_not_plantable
    return matrix, locations, truth


def evaluate(
    regions: pd.DataFrame,
    truth: list[tuple[int, int, float, float]],
    *,
    min_overlap: int = 20,
) -> tuple[set[int], int]:
    """Return the set of recovered truth indices and the false-region count."""

    called = regions[regions["significant"]] if len(regions) else regions
    recovered: set[int] = set()
    false_regions = 0
    for _, row in called.iterrows():
        matched = False
        for index, (start, end, _, _, *_run) in enumerate(truth):
            overlap = (
                min(int(row["cpg_index_end"]), end)
                - max(int(row["cpg_index_start"]), start)
                + 1
            )
            if overlap >= min_overlap:
                recovered.add(index)
                matched = True
        if not matched:
            false_regions += 1
    return recovered, false_regions


def scenario_null(config_kwargs: dict) -> None:
    print("=" * 78)
    print("NULL: no DMRs, two latent factors. Nothing should be called.")
    print("=" * 78)
    for seed in range(4):
        matrix, locations, _ = simulate(strata=[], seed=100 + seed)
        for statistic in ("correlation", "covariance"):
            config = Eiger2Config(statistic=statistic, **config_kwargs)
            regions, _, manifest = call_dmrs(matrix, locations, config=config)
            significant = int(regions["significant"].sum()) if len(regions) else 0
            print(
                f"  seed={100 + seed} {statistic:>11}: "
                f"K={manifest['latent_structure']['n_components']} "
                f"threshold={manifest['null']['threshold']:.4f} "
                f"regions={manifest.get('n_regions', 0):>3} "
                f"significant at q<=0.05 = {significant}"
            )


def scenario_prevalence(config_kwargs: dict, missing_rate: float = 0.0) -> None:
    label = "PREVALENCE" + (
        f" WITH {missing_rate:.0%} STRUCTURED MISSINGNESS" if missing_rate else ""
    )
    print("=" * 78)
    print(f"{label}: 48 planted DMRs across subgroup fraction x effect size")
    print("=" * 78)
    strata = [
        (0.05, 0.35),
        (0.05, 0.55),
        (0.10, 0.30),
        (0.10, 0.50),
        (0.25, 0.25),
        (0.25, 0.40),
        (0.50, 0.20),
        (0.50, 0.35),
    ]
    matrix, locations, truth = simulate(strata=strata, missing_rate=missing_rate)
    results = {}
    for statistic in ("correlation", "covariance"):
        config = Eiger2Config(statistic=statistic, **config_kwargs)
        regions, _, manifest = call_dmrs(matrix, locations, config=config)
        recovered, false_regions = evaluate(regions, truth)
        results[statistic] = recovered
        print(
            f"  {statistic:>11}: K={manifest['latent_structure']['n_components']} "
            f"threshold={manifest['null']['threshold']:.4f} -> "
            f"{len(recovered):>2}/{len(truth)} recovered, {false_regions} false regions"
        )
    print(f"    {'f':>6} {'delta':>6} | {'correlation':>12} {'covariance':>11}")
    for fraction, delta in strata:
        index = [
            i
            for i, (_, _, f, d, *_run) in enumerate(truth)
            if f == fraction and d == delta
        ]
        counts = [
            len([i for i in index if i in results[s]])
            for s in ("correlation", "covariance")
        ]
        print(
            f"    {fraction:6.2f} {delta:6.2f} | {counts[0]:>9}/{len(index)} "
            f"{counts[1]:>8}/{len(index)}"
        )


def scenario_statistic(config_kwargs: dict) -> None:
    print("=" * 78)
    print("HARD SETTING: small cohort, rare subgroups, low SNR")
    print("=" * 78)
    strata = [(0.05, 0.25), (0.08, 0.25), (0.10, 0.20), (0.15, 0.20)]
    for n_samples in (20, 40):
        matrix, locations, truth = simulate(
            n_samples=n_samples,
            strata=strata,
            replicates=6,
            noise_sd=0.06,
            seed=17,
        )
        for statistic in ("correlation", "covariance"):
            for window in (1, 6):
                config = Eiger2Config(
                    statistic=statistic, window=window, **config_kwargs
                )
                regions, _, _ = call_dmrs(matrix, locations, config=config)
                recovered, false_regions = evaluate(regions, truth)
                print(
                    f"  n={n_samples:>3} {statistic:>11} window={window}: "
                    f"{len(recovered):>2}/{len(truth)} recovered, "
                    f"{false_regions} false regions"
                )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", "null", "prevalence", "missingness", "statistic"],
    )
    parser.add_argument("--null-replicates", type=int, default=8)
    args = parser.parse_args(argv)
    config_kwargs = {"null_replicates": args.null_replicates}

    if args.scenario in ("all", "null"):
        scenario_null(config_kwargs)
        print()
    if args.scenario in ("all", "prevalence"):
        scenario_prevalence(config_kwargs)
        print()
    if args.scenario in ("all", "missingness"):
        scenario_prevalence(config_kwargs, missing_rate=0.25)
        print()
    if args.scenario in ("all", "statistic"):
        scenario_statistic(config_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
