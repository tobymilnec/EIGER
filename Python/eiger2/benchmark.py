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

#: Per-CpG value distributions ``simulate`` can draw from (master spec §6.2 /
#: FIX_SPEC WP6, WP6b). ``"unimodal"`` is the marginal every prior EIGER number
#: was produced on -- every CpG centred at 0.5 with a common ``noise_sd`` -- and
#: stays the default, so no committed number moves. The two fitted marginals
#: replace it with a real one measured by
#: ``benchmarking/layouts/fit_beta_marginal.py``: per-CpG baselines drawn from a
#: Beta mixture (bimodal, mass near 0 and 1) and per-CpG total sd targeted to the
#: empirical sd(mu) curve (heteroscedastic, compressed at the boundaries). They
#: differ only in whose sd(mu) they target, and together they bracket the truth:
#:
#: ``"bimodal_fitted"``
#:     the ICGC cohort's *total* per-CpG sd. That total includes the real
#:     between-sample biology a DMR caller exists to find, so it is an UPPER
#:     bound on the nuisance level -- recall under it is a lower bound.
#: ``"bimodal_atlas"``
#:     within-cell-type between-donor sd across purified normal leukocytes, with
#:     the binomial read-sampling component subtracted (FIX_SPEC WP6b). Purified
#:     normal cell types carry no cohort substructure, so this is close to a pure
#:     nuisance FLOOR -- recall under it is the other end of the bracket.
MARGINALS = ("unimodal", "bimodal_fitted", "bimodal_atlas")

#: The vendored fit each fitted marginal reads. Same schema either way, so
#: ``_marginal_params`` generalises by filename alone and no other code path
#: distinguishes them (FIX_SPEC WP6b.6).
MARGINAL_PARAM_FILES: dict[str, str] = {
    "bimodal_fitted": "beta_marginal_params.json",
    "bimodal_atlas": "beta_marginal_atlas_params.json",
}

#: Clip bounds applied to every simulated matrix, whatever the marginal. Also the
#: interval ``fit_beta_marginal.py`` clamps its fitted mean support to, so the
#: fitted and realisable supports are the same set.
BETA_CLIP: tuple[float, float] = (0.001, 0.999)


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
    layout_chrom: str | None,
    layout_profile: str,
    layout_offset: int | None,
) -> pd.DataFrame:
    """Build the ``chr``/``start``/``end`` frame for one of ``LAYOUTS``.

    ``dense`` (default): uniformly spaced CpGs on one synthetic chromosome -- the
    layout every prior EIGER benchmark number used, unchanged. ``wgbs_real``: a
    contiguous slice of real hg38 coordinates from the vendored fixture, at a varied
    offset per seed unless ``layout_offset`` is given explicitly -- a fixed offset
    would always sample the same (non-representative) stretch of the chromosome, see
    ``benchmarking/layouts/README.md``. ``layout_chrom=None`` (``wgbs_real`` only,
    spec §2b.5): every chromosome in the fixture at once, each proportioned to its
    own real share of ``n_cpg`` (not a single random offset slice of the
    concatenated array, which would not reliably span more than one chromosome).
    ``epic_fitted``: a generated targeted-panel layout fitted to ``layout_profile``'s
    empirical gap/run-length/depth structure.
    """

    if layout == "dense":
        return pd.DataFrame({
            "chr": "chr1",
            "start": np.arange(n_cpg) * 2,
            "end": np.arange(n_cpg) * 2 + 1,
        })
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
        if layout_chrom is None:
            from benchmarking.layouts.layouts import wgbs_chromosome_sizes

            sizes = wgbs_chromosome_sizes(npz)
            total = sum(sizes.values())
            rng = np.random.default_rng(seed)
            names = list(sizes)
            frames = []
            allocated = 0
            for i, name in enumerate(names):
                if i == len(names) - 1:
                    share = n_cpg - allocated  # remainder: exact total, no rounding gap
                else:
                    share = round(n_cpg * sizes[name] / total)
                    allocated += share
                share = max(1, min(share, sizes[name]))
                max_offset = max(0, sizes[name] - share)
                offset = int(rng.integers(0, max_offset + 1)) if max_offset else 0
                chrom_locations, _depth = load_wgbs_layout(
                    npz, chrom=name, n_cpg=share, offset=offset,
                )
                frames.append(chrom_locations)
            return pd.concat(frames, ignore_index=True)
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


def _marginal_params(marginal: str) -> dict:
    """The vendored beta-marginal fit behind one of the fitted marginals.

    Same repo-checkout-only contract as the layout fixtures (see
    ``_fixture_dir``): the JSONs are not shipped in the wheel and the default
    ``marginal="unimodal"`` never reads one. The two fits share a schema, so the
    only thing that varies here is the filename (FIX_SPEC WP6b.6).
    """

    import json

    path = _fixture_dir() / MARGINAL_PARAM_FILES[marginal]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- marginal={marginal!r} needs a full repo "
            f"checkout with the vendored fit, not an installed wheel; produce it "
            f"with benchmarking/layouts/fit_beta_marginal.py"
        )
    return json.loads(path.read_text())


def _bimodal_baseline_mu(
    rng: np.random.Generator, params: dict, n_cpg: int
) -> np.ndarray:
    """Per-CpG baseline means drawn from the fitted Beta mixture.

    One component draw then one Beta draw per CpG, clamped to the fit's own
    support (``mean_clamp``, which is ``BETA_CLIP``) so a baseline can never sit
    outside the interval the matrix is clipped to.
    """

    mixture = params["mixture"]
    weights = np.asarray(mixture["weights"], dtype=float)
    a = np.asarray(mixture["a"], dtype=float)
    b = np.asarray(mixture["b"], dtype=float)
    component = rng.choice(len(weights), size=n_cpg, p=weights / weights.sum())
    low, high = params.get("mean_clamp", BETA_CLIP)
    return np.clip(rng.beta(a[component], b[component]), low, high)


def _target_total_sd(mu: np.ndarray, params: dict) -> np.ndarray:
    """The fitted empirical sd(mu) curve, evaluated at each CpG's baseline.

    Piecewise-linear over the binned curve (``np.interp``), which is the
    shape-preserving read of an arch-shaped, non-monotone curve -- it cannot
    overshoot between bin centres the way an unconstrained cubic can, and needs
    nothing beyond numpy at simulate time. Outside the fitted bin range
    ``np.interp`` holds the end values flat, which is the intended behaviour:
    the extreme-mu bins are exactly where the real curve is flattest.

    ``sd_median`` is the binned median the fit reports, and is what the ICGC arm
    targets. A fit MAY also publish ``sd_simulator``, and this prefers it when
    present: the atlas arm's per-pair sd is a 3-donor variance with the binomial
    term subtracted, so more than half its pairs clip to zero in the rail bins
    and the median there is exactly zero -- a true statement about what 3 donors
    resolve, and an impossible target for a simulator, which would draw those
    CpGs as constant rows. See ``fit_beta_marginal.sd_curve`` for the estimator
    and why it is the conservative one to have chosen. Selection is by key, not
    by marginal name: ``beta_marginal_params.json`` carries no ``sd_simulator``
    and is byte-identical under this branch.
    """

    curve = params["sd_curve"]
    return np.interp(
        mu,
        np.asarray(curve["mu"], dtype=float),
        np.asarray(curve.get("sd_simulator", curve["sd_median"]), dtype=float),
    )


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
    baseline_mu: np.ndarray,
    start_position: int = 200,
) -> tuple[list[tuple[int, int, float, float, int, int]], int, list[tuple]]:
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

    **Direction (FIX_SPEC WP6.3, option (a), architect-confirmed).** The shift is
    applied toward whichever side of the CpG's baseline has more headroom
    (``sign = -1 if baseline_mu > 0.5 else +1``), per CpG. Under a realistic
    bimodal marginal a ``delta = 0.35`` planted upward at a baseline of 0.9 is
    not plantable -- the clip would silently eat most of it and recall-vs-delta
    would stop comparing like with like across marginals. Under
    ``marginal="unimodal"`` every baseline is exactly 0.5, so the sign is ``+1``
    everywhere and this reduces to the previous unsigned ``+= delta * group``.

    The third return value is one ``(start, end, carriers, pre_block)`` entry per
    planted DMR, in ``truth`` order, carrying the pre-plant values of the rows it
    touched. ``simulate`` uses it after clipping to measure what was *actually*
    planted (``effective_delta``), which is the only honest x-axis for a
    recall-vs-effect curve once the clip is in play.
    """

    spans = eligible_run_spans(locations, max_cpg_gap_bp)
    run_starts = np.array([s for s, _ in spans])
    run_ends = np.array([e for _, e in spans])
    n_cpg = len(locations)

    truth: list[tuple[int, int, float, float, int, int]] = []
    plants: list[tuple] = []
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
            block = slice(actual_start, actual_start + dmr_length)
            # -1 where the baseline sits above 0.5, +1 otherwise: the shift goes
            # toward the side with headroom. Column vector so it broadcasts per
            # CpG against the per-sample carrier row.
            sign = np.where(baseline_mu[block] > 0.5, -1.0, 1.0)[:, None]
            plants.append((
                actual_start, actual_start + dmr_length - 1,
                group > 0, matrix[block].copy(),
            ))
            matrix[block] += sign * delta * group
            truth.append((
                actual_start, actual_start + dmr_length - 1, fraction, delta,
                run_start, run_end,
            ))
            position += spacing
    return truth, n_not_plantable, plants


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
    layout_chrom: str | None = "chr1",
    layout_profile: str = "chr1",
    layout_offset: int | None = None,
    marginal: str = "unimodal",
) -> tuple[
    np.ndarray, pd.DataFrame, list[tuple[int, int, float, float, int, int, float]]
]:
    """A CpG-by-sample matrix with latent structure and planted DMRs.

    ``layout`` (master spec §6.1) selects the CpG coordinate structure: ``"dense"``
    (default, unchanged from every prior EIGER benchmark) is uniformly spaced CpGs on
    one synthetic chromosome, so every existing call is unaffected. ``"wgbs_real"`` and
    ``"epic_fitted"`` draw from the vendored ``benchmarking/layouts/`` fixture -- real
    hg38 coordinates and a fitted targeted-panel generator, respectively -- and require
    a full repo checkout (see ``_fixture_dir``). Planted DMRs always respect the
    layout's eligibility structure; see ``_plant_dmrs``.

    ``marginal`` (FIX_SPEC WP6, WP6b) selects the per-CpG VALUE distribution, the
    other half of what the layouts replicate. ``"unimodal"`` (default) is
    byte-identical to every prior EIGER benchmark at every seed: every CpG centred
    at 0.5 with a common ``noise_sd``. ``"bimodal_fitted"`` and ``"bimodal_atlas"``
    draw each CpG's baseline from a Beta mixture fitted to real data and scale
    that CpG's noise draw and latent loading row together so its total sd matches
    the fitted sd(mu) curve -- together, per-CpG scale factors, so the
    noise:latent energy ratio and hence ``n_latent``'s meaning are preserved
    exactly. The two differ only in which fit they read (see ``MARGINALS``): an
    upper bound on the nuisance level from a cancer cohort's total sd, and a
    nuisance floor from between-donor sd in purified normal cell types. Both need
    the same full repo checkout the real layouts do, and both draw the same number
    of values in the same order, so a seed's matrix depends on which fit is read
    but never on this branch's shape.

    Each truth entry is ``(start, end, fraction, delta, run_start, run_end,
    effective_delta)``. ``effective_delta`` is the mean absolute post-clip shift
    actually realised over the planted block's carrier entries: equal to ``delta``
    whenever the clip did not bite (always so under ``"unimodal"`` at the deltas
    the benchmarks use), and below it where a bimodal baseline sat close enough to
    a boundary that part of the shift was clipped away.
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

    if marginal not in MARGINALS:
        raise ValueError(f"marginal must be one of {MARGINALS}, got {marginal!r}")

    if marginal == "unimodal":
        # Unchanged, draw for draw, from every prior EIGER benchmark -- the
        # bimodal branch is separate rather than a generalisation of this one
        # precisely so no existing seed's matrix can move (see MARGINALS).
        baseline_mu = np.full(n_cpg, 0.5)
        matrix = 0.5 + rng.normal(0, noise_sd, (n_cpg, n_samples))
        if n_latent:
            latent = rng.normal(0, 1, (n_samples, n_latent))
            loading = rng.normal(0, latent_loading_sd, (n_cpg, n_latent))
            matrix += loading @ latent.T
    else:
        params = _marginal_params(marginal)
        baseline_mu = _bimodal_baseline_mu(rng, params, n_cpg)
        noise = rng.normal(0, noise_sd, (n_cpg, n_samples))
        latent = rng.normal(0, 1, (n_samples, n_latent)) if n_latent else None
        loading = (
            rng.normal(0, latent_loading_sd, (n_cpg, n_latent)) if n_latent else None
        )
        # The generative total sd of CpG i is sqrt(noise_sd^2 + ||loading_i||^2):
        # the latent factors are unit-variance and independent. Scaling the noise
        # draw and the loading ROW by one common per-CpG factor therefore hits the
        # fitted sd(mu_i) exactly while leaving the noise:latent energy ratio --
        # and so the number and meaning of the latent factors -- untouched.
        nominal_sd = np.full(n_cpg, float(noise_sd))
        if loading is not None:
            nominal_sd = np.sqrt(noise_sd**2 + (loading**2).sum(axis=1))
        scale = np.where(
            nominal_sd > 0, _target_total_sd(baseline_mu, params) / nominal_sd, 1.0
        )
        matrix = baseline_mu[:, None] + scale[:, None] * noise
        if n_latent:
            matrix += (scale[:, None] * loading) @ latent.T

    truth, n_not_plantable, plants = _plant_dmrs(
        matrix, locations, strata=strata, replicates=replicates, spacing=spacing,
        dmr_length=dmr_length, max_cpg_gap_bp=max_cpg_gap_bp, rng=rng,
        n_samples=n_samples, baseline_mu=baseline_mu,
    )

    matrix = np.clip(matrix, *BETA_CLIP)

    # What the plant actually achieved, measured on the clipped matrix against the
    # clipped counterfactual: |clip(pre + shift) - clip(pre)|, averaged over the
    # block's carrier entries. Recorded per truth entry (FIX_SPEC WP6.3) so a
    # recall-vs-effect curve can be plotted against the realised effect rather
    # than a nominal delta the boundary may have eaten.
    truth = [
        entry + (
            float(np.mean(np.abs(
                matrix[start : end + 1][:, carriers]
                - np.clip(pre_block[:, carriers], *BETA_CLIP)
            ))),
        )
        for entry, (start, end, carriers, pre_block) in zip(
            truth, plants, strict=True
        )
    ]
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
    locations.attrs["marginal"] = marginal
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
