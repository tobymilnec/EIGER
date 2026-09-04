"""Operating-characteristic benchmark for the EIGER 2 simulator.

Run with ``python -m eiger2.benchmark`` from anywhere once ``eiger`` is
installed: the scenarios below are self-contained and read no files.

Scenarios
    null            DMR-free matrix with latent structure -> false-positive rate
    prevalence      planted DMRs across a grid of subgroup fraction and effect
    missingness     the same grid with structured dropout
    statistic       correlation vs covariance under a hard, low-SNR setting

Scope of the installed package
    ``simulate`` ships one layout, ``"dense"`` (uniformly spaced CpGs on one
    synthetic chromosome), and one per-CpG value distribution,
    ``"unimodal"`` (every CpG centred at 0.5 with a common ``noise_sd``).
    Together they preserve the historical self-contained package scenarios and
    their regression hashes, and they depend on nothing outside this package.

    Real WGBS/EPIC-seq coordinate layouts and marginals fitted to real cohorts
    are *paper* material: their fixtures, fits and provenance are maintained in
    the separate ``eiger_paper`` repository and are deliberately not shipped in
    the wheel. ``simulate`` does not load them, and naming one
    (``layout="wgbs_real"``, ``marginal="bimodal_fitted"``, ...) is an error.
    Paper code injects what it loaded through the ``locations`` and
    ``marginal_sampler`` parameters instead; see ``simulate``.

Planted-DMR direction
    A planted DMR is a *coordinated* region: one direction, shared by every CpG
    and every carrier in it. ``simulate(planting_direction=...)`` says how that
    direction is chosen. ``"region_headroom"``, the default, chooses one scalar
    direction per planted region from the region's aggregate beta headroom.
    ``"per_cpg_headroom"`` chooses per CpG from that CpG's own baseline, which
    under a non-unimodal supplied marginal can give adjacent CpGs opposite-signed
    shifts; it exists for byte-exact reproduction of a run made before
    ``planting_direction`` existed, and for an explicit heterogeneous-direction
    stress test. A supplied-marginal signal simulation that does not record
    ``"region_headroom"`` is therefore not coherent-DMR power evidence; DMR-free
    and ``marginal="unimodal"`` results are unaffected. See ``simulate`` and
    ``PLANTING_DIRECTIONS``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import pandas as pd

from .core import Eiger2Config, call_dmrs
from .layouts import eligible_run_spans

#: CpG coordinate layouts this package generates on its own. Real coordinate
#: layouts are loaded by ``eiger_paper`` and passed in as ``locations``.
LAYOUTS = ("dense",)

#: Per-CpG value distributions this package draws on its own. Marginals fitted
#: to real data are loaded by ``eiger_paper`` and passed in as a
#: ``marginal_sampler``.
MARGINALS = ("unimodal",)

#: Mode names that named a fixture maintained in ``eiger_paper`` when the
#: benchmark harness and the software lived in one repository. They are rejected
#: by name -- rather than falling through to a generic "unknown mode" message or
#: to a search for a paper fixture directory that an installed package does not
#: have -- so the failure states where the fixture went and how to supply it.
PAPER_ONLY_LAYOUTS = ("wgbs_real", "epic_fitted")
PAPER_ONLY_MARGINALS = ("bimodal_fitted", "bimodal_atlas")

#: Clip bounds applied to every simulated matrix, whatever the marginal.
BETA_CLIP: tuple[float, float] = (0.001, 0.999)

#: How ``_plant_dmrs`` chooses the sign of the carrier shift it applies.
#:
#: ``"region_headroom"`` -- the default, and the coherent-DMR mode -- picks ONE
#: scalar direction for the whole planted block from the block's aggregate beta
#: headroom, so every CpG and every carrier in one planted region moves the same
#: way, which is what "a differentially methylated region" means.
#:
#: ``"per_cpg_headroom"`` picks the direction independently for each CpG from
#: that CpG's own baseline. Under a bimodal or otherwise non-unimodal supplied
#: marginal adjacent CpGs can then receive opposite-signed shifts, so the block is
#: locally anti-correlated rather than coordinated. It exists for two purposes
#: only: byte-exact reproduction of a run made before ``planting_direction``
#: existed, and an explicit heterogeneous-direction stress test.
#:
#: Consequence for stored results: a supplied-marginal *signal* simulation that
#: does not record ``"region_headroom"`` is not coherent-DMR recall-versus-effect
#: evidence, whatever it was labelled at the time, because the mode it ran under
#: was the per-CpG rule. Results with no planted signal (DMR-free null and
#: any-call calibration) and results under ``marginal="unimodal"`` are unaffected:
#: the first never reaches this code, and in the second every baseline is exactly
#: 0.5, so the two modes are byte-identical.
PLANTING_DIRECTIONS = ("region_headroom", "per_cpg_headroom")

#: Absolute tolerance on the ``"region_headroom"`` comparison. Both sides of it
#: are means of beta-scale headrooms, so they live in ``[0, 1]`` and an absolute
#: tolerance is well scaled. It exists so that a case which ties in exact
#: arithmetic -- a block whose baselines are symmetric about 0.5, the commonest
#: mixed-baseline case there is -- takes the documented ``+delta`` tie-break
#: rather than whichever side happened to win by a last-bit rounding difference.
HEADROOM_TIE_ATOL: float = 1e-12

#: What ``simulate`` expects of a caller-supplied marginal: given the run's
#: generator and the CpG count, return ``(baseline_mu, total_sd)`` -- the
#: per-CpG baseline mean and the per-CpG total standard deviation to hit. Both
#: are length-``n_cpg`` float arrays. ``simulate`` scales each CpG's noise draw
#: and latent-loading row by one common factor to realise ``total_sd``, which
#: leaves the noise:latent energy ratio -- and so the meaning of ``n_latent`` --
#: untouched. Draw from ``rng`` and nothing else, so a seed still determines the
#: run: ``simulate`` draws the noise, latent and loading arrays after this
#: returns, in that order.
MarginalSampler = Callable[
    [np.random.Generator, int], tuple[np.ndarray, np.ndarray]
]


def _reject_paper_only(kind: str, name: str, paper_only: tuple[str, ...],
                       supported: tuple[str, ...], injection: str) -> None:
    """Raise the boundary message for a mode that ``eiger_paper`` owns."""

    if name in paper_only:
        raise ValueError(
            f"{kind}={name!r} is maintained in the eiger_paper repository, not "
            f"in the installed eiger2 package; the shipped simulator supports "
            f"{kind} in {supported}. Load it with the eiger_paper helper and "
            f"pass the result to simulate({injection}=...)."
        )


def _check_planting_direction(planting_direction: str) -> None:
    """Reject an unknown planting direction with one actionable line."""

    if planting_direction not in PLANTING_DIRECTIONS:
        raise ValueError(
            f"planting_direction must be one of {PLANTING_DIRECTIONS}, got "
            f"{planting_direction!r}; 'region_headroom' gives one planted region "
            f"one direction (the default), 'per_cpg_headroom' reproduces the "
            f"historical per-CpG direction for old runs and heterogeneous-"
            f"direction stress tests"
        )


def _region_direction(block_mu: np.ndarray, delta: float) -> float:
    """The one scalar sign a whole planted block is shifted by.

    Aggregate beta headroom: for the block's baseline means, and the same
    ``BETA_CLIP`` bounds the finished matrix is clipped to, compare the mean
    attainable shift of ``+delta`` against that of ``-delta`` and take the larger.

        up_i   = clip(mu_i + delta) - clip(mu_i)
        down_i = clip(mu_i) - clip(mu_i - delta)

    Both are non-negative. The direction is ``+1`` when ``mean(up)`` is greater
    than ``mean(down)`` or the two agree to within ``HEADROOM_TIE_ATOL``, and
    ``-1`` otherwise: the documented tie-break is ``+delta``. The tolerance is
    what makes that tie-break reachable -- a block whose baselines are symmetric
    about 0.5 ties exactly in real arithmetic but not always in floating point,
    and it would otherwise be decided by a last-bit rounding difference rather
    than by the rule.

    The rule reads the block's baselines, ``delta`` and ``BETA_CLIP`` and nothing
    else: no biological label, no called region, no q-value, no benchmark outcome,
    and no random draw. It is therefore deterministic, and choosing a mode costs
    no random number.
    """

    low, high = BETA_CLIP
    at = np.clip(block_mu, low, high)
    up = float((np.clip(block_mu + delta, low, high) - at).mean())
    down = float((at - np.clip(block_mu - delta, low, high)).mean())
    return 1.0 if up > down or abs(up - down) <= HEADROOM_TIE_ATOL else -1.0


def _layout_locations(layout: str, n_cpg: int) -> pd.DataFrame:
    """Build the ``chr``/``start``/``end`` frame for one of ``LAYOUTS``.

    ``dense`` (the only shipped layout, and the default): uniformly spaced CpGs
    on one synthetic chromosome. Its retained regression outputs are unchanged.
    A real coordinate layout is loaded by ``eiger_paper`` and passed to
    ``simulate`` as ``locations``; it never arrives here.
    """

    if layout == "dense":
        return pd.DataFrame({
            "chr": "chr1",
            "start": np.arange(n_cpg) * 2,
            "end": np.arange(n_cpg) * 2 + 1,
        })
    _reject_paper_only("layout", layout, PAPER_ONLY_LAYOUTS, LAYOUTS, "locations")
    raise ValueError(
        f"layout must be one of {LAYOUTS}, got {layout!r}; pass a coordinate "
        f"frame as simulate(locations=...) to simulate any other layout"
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
    planting_direction: str = "region_headroom",
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

    **Direction.** The shift is applied toward the side with beta headroom, so a
    ``delta = 0.35`` planted at a baseline of 0.9 is not silently eaten by the clip
    and recall-vs-delta keeps comparing like with like across marginals.
    ``planting_direction`` says at what granularity that choice is made:

    ``"region_headroom"`` (the default)
        One scalar sign for the whole planted block, from the block's aggregate
        headroom; see ``_region_direction``. Every CpG and every carrier in one
        planted region moves the same way, which is what a planted DMR is meant to
        be. Adjacent CpGs can never receive opposite-signed effects.

    ``"per_cpg_headroom"``
        The historical rule, ``sign = -1 if baseline_mu > 0.5 else +1``, applied
        per CpG. Under a bimodal or otherwise non-unimodal supplied marginal
        adjacent CpGs can then be shifted in opposite directions, making the block
        locally anti-correlated rather than a coordinated region. It exists for
        byte-exact reproduction of a run made before ``planting_direction``
        existed, and for an explicit heterogeneous-direction stress test; a
        supplied-marginal result produced under it is not coherent-DMR power
        evidence. See ``PLANTING_DIRECTIONS``.

    Under ``marginal="unimodal"`` every baseline is exactly 0.5, so both modes give
    ``+1`` everywhere and both reduce to the original unsigned ``+= delta * group``.
    Neither mode consumes a random number, so the carrier draws -- and every
    matrix, truth coordinate and layout placement -- are identical between them at
    the same seed.

    The third return value is one ``(start, end, carriers, pre_block)`` entry per
    planted DMR, in ``truth`` order, carrying the pre-plant values of the rows it
    touched. ``simulate`` uses it after clipping to measure what was *actually*
    planted (``effective_delta``), which is the only honest x-axis for a
    recall-vs-effect curve once the clip is in play.
    """

    _check_planting_direction(planting_direction)
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
            # The shift goes toward the side with headroom: one scalar for the
            # whole block, or -- historically -- one per CpG as a column vector
            # that broadcasts against the per-sample carrier row.
            if planting_direction == "region_headroom":
                sign = _region_direction(baseline_mu[block], delta)
            else:
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
    marginal: str = "unimodal",
    planting_direction: str = "region_headroom",
    locations: pd.DataFrame | None = None,
    marginal_sampler: MarginalSampler | None = None,
) -> tuple[
    np.ndarray, pd.DataFrame, list[tuple[int, int, float, float, int, int, float]]
]:
    """A CpG-by-sample matrix with latent structure and planted DMRs.

    ``layout`` (master spec §6.1) selects the CpG coordinate structure. The
    installed package generates one, ``"dense"`` (the default): uniformly spaced
    CpGs on one synthetic chromosome. To simulate on any other layout, load its
    coordinates yourself and pass the ``chr``/``start``/``end`` frame as
    ``locations``; ``layout`` then only labels what you passed, and ``n_cpg`` is
    ``len(locations)`` rather than derived from ``spacing`` and ``strata``. Real
    WGBS/EPIC-seq layouts and their fixtures live in ``eiger_paper``, which owns
    the loading; naming one here without supplying ``locations`` is an error.
    Planted DMRs always respect the layout's eligibility structure; see
    ``_plant_dmrs``.

    ``marginal`` selects the per-CpG VALUE distribution, the other half of what a
    layout replicates. The installed package draws one, ``"unimodal"`` (the
    default): every CpG centred at 0.5 with a common ``noise_sd``. This keeps the
    previous dense/unimodal draw path unchanged. To simulate under a fitted
    marginal, pass a ``marginal_sampler`` returning per-CpG
    ``(baseline_mu, total_sd)``; each CpG's noise draw and latent loading row are
    then scaled together by one common factor so its total sd matches
    ``total_sd``, preserving the noise:latent energy ratio and hence
    ``n_latent``'s meaning exactly. See ``MarginalSampler`` for the draw order a
    sampler must respect. Marginals fitted to real cohorts live in
    ``eiger_paper``, which owns the fits; naming one here without supplying a
    ``marginal_sampler`` is an error.

    ``planting_direction`` selects how the sign of a planted shift is chosen. It
    is validated before anything is drawn and must be one of
    ``PLANTING_DIRECTIONS``:

    ``"region_headroom"`` (the default)
        One scalar direction for a whole planted region, chosen from the region's
        aggregate beta headroom (see ``_region_direction``): every CpG and every
        carrier in that region is shifted the same way, so the plant is a
        coordinated, region-consistent DMR. This is the correct default for a
        power simulation, and the only mode whose blocks match the documented
        meaning of a planted DMR under a non-unimodal marginal.

    ``"per_cpg_headroom"``
        The direction is chosen per CpG from that CpG's own baseline, exactly as
        it was before this argument existed. Under a bimodal or otherwise
        non-unimodal supplied marginal, adjacent CpGs of one block can then move
        in opposite directions, so the block is locally anti-correlated rather
        than coordinated. It exists for byte-exact reproduction of a run made
        before ``planting_direction`` existed, and for an explicit
        heterogeneous-direction stress test. It is not a coherent-DMR power
        simulation.

    The two modes consume the same random numbers in the same order, so carrier
    identities, truth coordinates and layout placement are identical between them
    at one seed; only the sign pattern inside a planted block differs. Under
    ``marginal="unimodal"`` they are byte-identical, every baseline being 0.5.

    **Which stored results this bears on.** A supplied-marginal simulation with
    planted signal that does not record ``"region_headroom"`` ran under the
    per-CpG rule and is not coherent-DMR recall-versus-effect evidence; it may be
    retained as heterogeneous-direction stress evidence if labelled that way.
    Results with no planted signal -- DMR-free null and any-call calibration,
    including under a supplied marginal -- and results under
    ``marginal="unimodal"`` are unaffected, the modes being unreachable in the
    first case and byte-identical in the second. Record the mode with the run:
    it is reported at ``locations.attrs["planting_direction"]``.

    Each truth entry is ``(start, end, fraction, delta, run_start, run_end,
    effective_delta)``. ``effective_delta`` is the mean absolute post-clip shift
    actually realised over the planted block's carrier entries: equal to ``delta``
    whenever the clip did not bite (always so under ``"unimodal"`` at the deltas
    the benchmarks use), and below it where a supplied baseline sat close enough
    to a boundary that part of the shift was clipped away. Under
    ``"region_headroom"`` a block whose baselines straddle 0.5 pays for coherence
    here rather than silently: the one chosen direction has headroom at some of
    the block's CpGs and less at others, so ``effective_delta`` falls below
    ``delta``. Plot against ``effective_delta``, not ``delta``, whenever a
    supplied marginal is in play.
    """

    _check_planting_direction(planting_direction)
    rng = np.random.default_rng(seed)
    strata = strata if strata is not None else []
    n_dmr = len(strata) * replicates
    n_cpg = spacing * max(n_dmr, 1) + 400

    if locations is None:
        locations = _layout_locations(layout, n_cpg)
    else:
        locations = locations.reset_index(drop=True)
    n_cpg = len(locations)  # supplied layouts set the size; for ``dense`` this
                             # keeps the matrix and locations sizes provably tied

    if marginal_sampler is None:
        _reject_paper_only(
            "marginal", marginal, PAPER_ONLY_MARGINALS, MARGINALS, "marginal_sampler"
        )
        if marginal not in MARGINALS:
            raise ValueError(
                f"marginal must be one of {MARGINALS}, got {marginal!r}; pass a "
                f"sampler as simulate(marginal_sampler=...) to simulate any other "
                f"marginal"
            )
        # Unchanged, draw for draw, from the previous dense/unimodal path -- the
        # sampler branch is separate rather than a generalisation of this one so
        # retained regression seeds do not move (see MARGINALS).
        baseline_mu = np.full(n_cpg, 0.5)
        matrix = 0.5 + rng.normal(0, noise_sd, (n_cpg, n_samples))
        if n_latent:
            latent = rng.normal(0, 1, (n_samples, n_latent))
            loading = rng.normal(0, latent_loading_sd, (n_cpg, n_latent))
            matrix += loading @ latent.T
    else:
        baseline_mu, target_total_sd = marginal_sampler(rng, n_cpg)
        baseline_mu = np.asarray(baseline_mu, dtype=float)
        target_total_sd = np.asarray(target_total_sd, dtype=float)
        noise = rng.normal(0, noise_sd, (n_cpg, n_samples))
        latent = rng.normal(0, 1, (n_samples, n_latent)) if n_latent else None
        loading = (
            rng.normal(0, latent_loading_sd, (n_cpg, n_latent)) if n_latent else None
        )
        # The generative total sd of CpG i is sqrt(noise_sd^2 + ||loading_i||^2):
        # the latent factors are unit-variance and independent. Scaling the noise
        # draw and the loading ROW by one common per-CpG factor therefore hits the
        # requested total sd exactly while leaving the noise:latent energy ratio --
        # and so the number and meaning of the latent factors -- untouched.
        nominal_sd = np.full(n_cpg, float(noise_sd))
        if loading is not None:
            nominal_sd = np.sqrt(noise_sd**2 + (loading**2).sum(axis=1))
        scale = np.where(nominal_sd > 0, target_total_sd / nominal_sd, 1.0)
        matrix = baseline_mu[:, None] + scale[:, None] * noise
        if n_latent:
            matrix += (scale[:, None] * loading) @ latent.T

    truth, n_not_plantable, plants = _plant_dmrs(
        matrix, locations, strata=strata, replicates=replicates, spacing=spacing,
        dmr_length=dmr_length, max_cpg_gap_bp=max_cpg_gap_bp, rng=rng,
        n_samples=n_samples, baseline_mu=baseline_mu,
        planting_direction=planting_direction,
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
    locations.attrs["planting_direction"] = planting_direction
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


def _run(matrix, locations, config) -> tuple:
    """One ``call_dmrs`` run, plus what a scenario is allowed to assume about it.

    ``call_dmrs`` returns rather than raises when a structural gate fails, and an
    aborted run carries no ``null`` block and no decision column -- so exit code
    zero is not complete inference (plan section 3.1 rule 8). Every scenario
    reads the run through here so a refusal is reported as a refusal instead of
    surfacing as a ``KeyError`` on a manifest key the run never earned.
    """

    regions, _, manifest = call_dmrs(matrix, locations, config=config)
    decided = bool(len(regions)) and "significant" in regions.columns
    return regions, manifest, manifest.get("status", "unknown"), decided


def _summary(manifest: dict) -> str:
    """``K``, the edge threshold and the run status, refusal-tolerant."""

    n_components = manifest.get("latent_structure", {}).get("n_components", "?")
    null = manifest.get("null")
    threshold = f"{null['threshold']:.4f}" if null else "refused"
    return (
        f"K={n_components} threshold={threshold} "
        f"status={manifest.get('status', 'unknown')}"
    )


def _refused_strata(manifest: dict) -> list[str]:
    """Length strata whose q-values were refused for want of null opportunity.

    ``evaluate`` filters on a boolean column in which a refused region is ``pd.NA``,
    so a refused truth region is silently counted as not recovered. Reporting the
    refused strata beside the recall line keeps a refusal from reading as a miss
    (plan section 3.1 rule 8).
    """

    strata = manifest.get("fdr", {}).get("insufficient_null_opportunity_strata", [])
    return [entry["stratum"] for entry in strata]


def _refusal_reason(manifest: dict) -> str:
    """The first failed structural gate, for a one-line refusal report."""

    gates = manifest.get("gates", {}).get("structural", {})
    for name, gate in gates.items():
        if not gate.get("passed", True):
            return f"{name}: {gate.get('error', 'gate failed')}"
    return "no decisions reported"


def scenario_null(config_kwargs: dict) -> None:
    print("=" * 78)
    print("NULL: no DMRs, two latent factors. Nothing should be called.")
    print("=" * 78)
    for seed in range(4):
        matrix, locations, _ = simulate(strata=[], seed=100 + seed)
        for statistic in ("correlation", "covariance"):
            config = Eiger2Config(statistic=statistic, **config_kwargs)
            regions, manifest, _status, decided = _run(matrix, locations, config)
            significant = (
                str(int(regions["significant"].sum())) if decided else "refused"
            )
            print(
                f"  seed={100 + seed} {statistic:>11}: {_summary(manifest)} "
                f"regions={manifest.get('n_regions', 0):>3} "
                f"significant at q<=0.05 = {significant}"
            )
            if not decided:
                print(f"      {_refusal_reason(manifest)}")


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
        regions, manifest, _status, decided = _run(matrix, locations, config)
        recovered, false_regions = evaluate(regions, truth) if decided else (set(), 0)
        results[statistic] = recovered
        outcome = (
            f"{len(recovered):>2}/{len(truth)} recovered, {false_regions} false regions"
            if decided else _refusal_reason(manifest)
        )
        print(f"  {statistic:>11}: {_summary(manifest)} -> {outcome}")
        refused = _refused_strata(manifest)
        if refused:
            print(
                f"      refused length strata {refused}: their regions carry no "
                f"decision, so they count as not recovered above -- a refusal, "
                f"not a miss"
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
                regions, manifest, _status, decided = _run(
                    matrix, locations, config
                )
                if decided:
                    recovered, false_regions = evaluate(regions, truth)
                    outcome = (
                        f"{len(recovered):>2}/{len(truth)} recovered, "
                        f"{false_regions} false regions"
                    )
                else:
                    outcome = _refusal_reason(manifest)
                refused = _refused_strata(manifest)
                suffix = f"; refused strata {refused}" if refused else ""
                print(
                    f"  n={n_samples:>3} {statistic:>11} window={window}: "
                    f"{outcome}{suffix}"
                )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
