"""EIGER 2 reference core.

The whole method is five steps. Each is a separate, testable function.

    1. centre_cpgs        remove the per-CpG mean; keep NaN as NaN
    2. estimate_latent_basis / deflate
                          remove K genome-wide sample-space directions,
                          estimated on a THINNED CpG set so that focal
                          regional signal cannot enter the basis
    3. edge_statistic     local co-variation per adjacent CpG pair (or per
                          w-CpG window), pairwise-complete
    4. null_threshold     locality-breaking null built by pairing CpGs that are
                          far apart on the same chromosome; preserves every
                          cross-sample dependency, destroys only adjacency
    5. segment + region_fdr
                          contiguous runs above threshold, then a region-level
                          empirical FDR obtained by pushing null statistic
                          tracks through the *identical* segmentation

Because the FDR is calibrated by running the same segmenter on null tracks, the
segmentation heuristics (window width, gap tolerance, minimum length) do not
need to be individually justified: whatever they do to the data they also do to
the null, and the reported error rate absorbs it.

Characterisation (which samples split, delta beta, direction) happens after a
region has been called and is labelled descriptive throughout. It is
post-selection and no q-value is attached to it.
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

__all__ = [
    "CANONICAL_CHROMOSOMES",
    "Eiger2Config",
    "call_dmrs",
    "canonical_chromosome_order",
    "centre_cpgs",
    "characterise_region",
    "chromosome_lag_feasibility",
    "chromosome_sort_permutation",
    "deflate",
    "edge_statistic",
    "estimate_latent_basis",
    "member_step_chromosome_feasibility",
    "null_threshold",
    "observed_edge_threshold",
    "permutation_null_tracks",
    "region_fdr",
    "region_minority_size",
    "biological_support_diagnostics",
    "classify_biological_support",
    "main_region_table",
    "segment",
]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


# Spec §2a.3: the label attached to every calibration diagnostic (ACF match,
# region/edge exchangeability, overdispersion) -- confounded with genuine
# signal on real data, so none of them suppress inference; see docs/FDR.md
# §14.1. Shared with cli.py so the wording has one source.
CALIBRATION_DIAGNOSTIC_NOTE = (
    "Calibration diagnostic. Valid as a hard criterion only on data known to "
    "be null. On a cohort containing genuine coordinated regions these "
    "ratios are expected to exceed 1, and a high value is not by itself "
    "evidence of a broken null."
)

# FIX_SPEC §WP2: the label attached to a run that calls a significant region
# in the bottom length stratum ("4-5" CpGs). Measured (docs/FDR.md, Stage 2e
# `signal_grid`): weak-effect 4-CpG cells realise 6.9-15.4% per-run FDP
# against a 5% nominal target -- 6.9% at n=80, 15.4% at n=20, so the gap
# widens as the cohort shrinks. This is a NOTE, not a gate: it changes no
# q-value, no `significant` call and no run status, it only says out loud at
# point of use what the docs already say. Shared with cli.py so the wording
# has one source.
BOTTOM_STRATUM_FDR_NOTE = (
    "Short-region calibration limitation. Regions of 4-5 CpGs carrying weak "
    "effects realised 6.9-15.4% FDR against the 5% nominal target in the "
    "Stage 2e signal_grid measurement (worst in small cohorts). Regions at "
    "or near min_cpgs are not demonstrated to be FDR-controlled -- treat "
    "them as ranked candidates. See docs/FDR.md."
)

# FIX_SPEC §WP4: the label attached to a run whose significant list contains
# regions carried by only one or two samples. A single aberrant sample
# deviating across a run of CpGs produces genuine adjacent correlation, so
# such a region is not a false positive under any null -- it is simply not a
# differentially methylated *region* in the sense usually intended. The
# shipped `min_prevalence_samples` stays 0 (see that field's docstring for
# the measurement that decided it), so this count is REPORTED rather than
# filtered: a NOTE, not a gate. It changes no q-value, no `significant` call
# and no run status. Shared with cli.py so the wording has one source.
# The default path builds no region null, so there is no region FDR against
# the genome background to report. Shared with cli.py so the wording has one
# source, and written into the manifest so a reader of the output alone is told
# what the run does not claim.
CONFIRMATION_NOTE = (
    "Confirmed regions reject unimodality of their own per-sample mean "
    "methylation (Hartigan's dip test, BH across this run's candidates at "
    "target_fdr). That is an error rate against the hypothesis 'these samples "
    "are one population', not against the genome background. A DMR present in "
    "every sample is invisible to it, a continuously graded effect stays a "
    "candidate, and a rare subset has little power. See docs/CONFIRM.md."
)

NO_CALIBRATED_NULL_NOTE = (
    "No region FDR against the genome background. region_null_mode='off' is "
    "the default: every self-calibrated background null measured on "
    "WGBS-realistic data failed (docs/FDR.md section 18), so region_qvalue, "
    "region_qvalue_upper_95 and significant are refused and the regions are "
    "reported as candidates ranked by region_area. Region FDR against the "
    "genome background requires matched control matrices; the calibrated "
    "opt-in path that consumes them (--null-reference) is specified in "
    "docs/FDR.md section 18 and is not implemented in this release."
)

LOW_PREVALENCE_SAMPLE_CEILING = 2
LOW_PREVALENCE_NOTE = (
    "Low-prevalence regions. Some significant regions are carried by one or "
    "two samples. Their coordination is real, but a one- or two-sample "
    "deviation is usually an outlier or an epivariant rather than a "
    "subgroup-defining DMR. min_prevalence_samples defaults to 0 so nothing "
    "is hidden; pass --min-prevalence-samples 3 to drop them from "
    "`significant` (region_qvalue is calibrated on the unfiltered set either "
    "way, so the filter changes no q-value)."
)

# FIX_SPEC §WP6b item 2 (WP6 disposition option E): how close a candidate
# component's loading concentration has to sit to `deflation_max_concentration`
# before the run says so out loud. 0.02 is not a round number chosen for
# tidiness -- it is set from the measurement that motivated the note. WP6's
# realistic-marginal sweep found the concentration band flat in cohort size
# under the synthetic marginal (0.084 at n = 12, 20 and 80) but cohort-size
# dependent under a fitted one, where the n = 12 band runs 0.0880-0.1095 against
# the shipped 0.10 cutoff and 20 of 80 runs cross it (the bimodal-marginal
# sweep, generated by ``bimodal_marginal.py`` in the eiger_paper repository and
# not shipped with this package). A margin of 0.02 spans that band, so the one
# configuration where the filter was measured changing its
# mind is caught at point of use. Like WP2's and WP4's notes this is a NOTE, not
# a gate: it changes no verdict, no component is deflated or retained
# differently because of it, and no run status depends on it. Shared with cli.py
# so the wording has one source.
#
# WP7b moved the cutoff to 0.12 and the margin stays 0.02, so the watch band is
# now (0.10, 0.14). That is the right window for a note whose job is to say the
# decision was close, and it was: (0.10, 0.14) brackets every number the flip
# was argued over -- Table S2's M arm at 0.1144-0.1199, knowingly crossed and
# below the cutoff now; the immune atlas's genome-wide leading component at
# 0.1258, kept with 0.0058 of headroom; ICGC's smallest at 0.1335, kept with
# 0.0135. Both real cohorts therefore now emit this note, which is the intended
# reading: they are kept, and they are kept by a margin a different cohort could
# close. A run inside the band is one whose deflation verdict WP7b's evidence
# pack could not settle from the outside.
DEFLATION_NEAR_CUTOFF_MARGIN = 0.02
NEAR_CUTOFF_CONCENTRATION_NOTE = (
    "Deflation decision near its cutoff. One or more candidate components have "
    "a loading concentration within "
    f"{DEFLATION_NEAR_CUTOFF_MARGIN} of deflation_max_concentration, so a small "
    "change in the data would flip whether they are treated as nuisance and "
    "projected out. The concentration is an absolute cutoff on a quantity whose "
    "scale depends on the data's marginal, its units and the cohort size, and "
    "the measured near-cutoff band is a small-cohort effect (n = 12). Nothing "
    "is wrong with this run; check the reported concentrations against "
    "deflation_max_concentration before treating the deflation verdict as "
    "settled."
)

# FIX_SPEC §WP7a item 2: the same note under `deflation_rule="relative"`, where
# the boundary is the run's own permutation-null quantile rather than the
# shipped constant, so the absolute wording ("an absolute cutoff", the constant
# named) would be wrong. Same margin, same NOTE-not-gate contract, separate
# wording so the absolute path's stderr text is unchanged to the byte.
NEAR_BOUNDARY_CONCENTRATION_NOTE = (
    "Deflation decision near its boundary. One or more candidate components "
    "have a loading concentration within "
    f"{DEFLATION_NEAR_CUTOFF_MARGIN} of the permutation-null boundary this run "
    "derived (deflation_rule='relative'), so a small change in the data would "
    "flip whether they are treated as nuisance and projected out. The boundary "
    "moves with the data's marginal, its units and the cohort size by "
    "construction, so a near-boundary component is a genuinely marginal "
    "component rather than a scale artefact. Nothing is wrong with this run; "
    "check the reported concentrations against the recorded boundary and its "
    "null band before treating the deflation verdict as settled."
)

_DEFAULT_SUPPORT_DISPERSION_CUT = 0.5
# Measured across standard benchmark scenarios: with no missingness, support
# dispersion is structurally 0 (every eligible window has full support, so the
# 90th/10th percentile spread is zero); under 25% structured missingness it
# measured 1.04-1.13. 0.5 sits with comfortable margin on both sides of that
# gap and reliably separates the two regimes.


@dataclass(frozen=True)
class Eiger2Config:
    """Everything the caller can tune. Defaults are the validated settings."""

    # --- discovery statistic
    statistic: str = "correlation"
    """``"correlation"`` (scale-free, recommended) or ``"covariance"`` (carries
    effect-size information but is intrinsically prevalence-limited: for a
    two-group structure with minor fraction ``f`` and effect ``delta``, the
    adjacent covariance is ``f(1-f)delta**2`` and cannot see rare subgroups,
    whereas correlation, ``f(1-f)delta**2 / (f(1-f)delta**2 + sigma**2)``,
    saturates towards 1 for any ``f`` once ``delta`` exceeds the noise).
    """

    window: int = 4
    """Number of CpGs in the co-variation window. 1 means adjacent pairs.

    ``w > 1`` uses the mean off-diagonal correlation of the ``w x w`` matrix,
    which has roughly ``sqrt(w(w-1)/2)`` times less null variance at no cost in
    the signal case. Worth using below ~30 samples; unnecessary above.
    """

    # --- latent structure removal
    deflate_components: int | None = None
    """Number of sample-space directions to project out. ``None`` = estimate by
    parallel analysis on the thinned set."""

    deflation_thin: int = 50
    """Use every Nth CpG to estimate the latent basis.

    This is the load-bearing detail. A focal DMR of a few dozen CpGs contributes
    at most one row to a 1-in-50 thin and therefore cannot form a component,
    while a genuine genome-wide factor (purity, composition, batch) is fully
    represented. Estimating the basis on *all* CpGs deletes focal signal.
    """

    deflation_max_concentration: float = 0.12
    """Only project out components whose genomic loading is DIFFUSE.

    A nuisance factor (purity, composition, batch) loads on the whole genome;
    a biological subgroup (cell type, tumour status, a rare clone) loads on
    marker regions. The fraction of a component's loading energy in the top
    1% of CpGs separates them cleanly: on the synthetic marginal nuisance
    components score 0.08-0.09 and biological components 0.15-0.40. Without
    this filter, blind deflation deletes exactly the signal EIGER exists to
    find -- on a cohort of 4 cell types plus tumour status plus a rare
    subgroup, parallel analysis selects K=5 and recovery collapses from 70/70
    regions to 0/70. Set to 1.0 to disable the filter and deflate every
    component.

    **0.12, and it is a measured sandwich rather than a round number**
    (FIX_SPEC WP7b, adjudicated 2026-08-28). The separation above is a
    synthetic-marginal statement; on a realistic bimodal marginal the nuisance
    band rises into the old 0.10 cutoff and the filter half-fires -- median
    0.0996 on the atlas marginal, 158 of 280 runs with a nuisance component
    the cutoff kept -- which is a deflation stage switched off on exactly the
    data the tool is for. The constant is set from both sides:

    - *From below, the simulated nuisance bands it has to cover:* 0.12 covers
      98.6% of the atlas-marginal candidates (max 0.1450, q95 0.1146), 100% of
      ``bimodal_fitted`` (max 0.1095) and 100% of ``unimodal`` (max 0.0850).
      At 0.10 the atlas figure is 51.6%, with 117 of 280 runs deflating
      nothing at all; at 0.12 no run deflates nothing, and small-cohort recall
      recovers from 0.46-0.56 to 0.72-0.83 (the bimodal-marginal sweep,
      generated by ``bimodal_marginal.py`` in the eiger_paper repository).
    - *From above, the real cohorts it must not touch:* measured at deployment
      scope -- one whole-genome basis -- the immune atlas scores its leading
      component at 0.1258 (26.3 M CpGs x 36 samples, measured by
      ``atlas_genome_deflation.py`` in eiger_paper) and ICGC at 0.1335 (WP3).
      Neither deflates at 0.12. A cutoff that reaches them (0.15) churns the
      ICGC significant set (Jaccard 0.328), which brackets the range from
      above. Two cohorts is the evidence there is; it is a measurement, not a
      guarantee about WGBS in general.
    - *Knowingly crossed:* one committed synthetic artefact sits inside the
      move -- Table S2's M-value arm, candidates 0.1144-0.1199, whose
      recall collapse *is* its deflation column. The beta-vs-M comparison is
      therefore cutoff-conditional and was re-measured at 0.12, with the 0.10
      generation kept as the record of why the constant moved (WP7b item 3).
      An audit of all committed artefacts recording a concentration found no
      other crossing (``cutoff_audit.py`` and ``tests/test_wp7b_cutoff.py``,
      both in eiger_paper).

    **Deflation verdicts have the scope of the basis estimation.** The
    concentration is the top-1% energy share of a component fitted to the
    matrix in front of it, so the same nuisance factor scored per chromosome
    scatters about +/-0.05 around its genome-wide value (the atlas cohort:
    0.1074-0.1802 per chromosome, 0.1258 pooled). Genome-scope basis
    estimation is the recommended deployment; chromosome-scope runs are a
    memory workaround whose verdicts carry that scatter, and
    ``DEFLATION_NEAR_CUTOFF_MARGIN`` is the point-of-use guard either way.
    """

    deflation_rule: str = "absolute"
    """Where the nuisance/biology boundary comes from (FIX_SPEC WP7a).

    ``"absolute"`` (default, and the only rule any committed number was
    produced under): the boundary is the constant ``deflation_max_concentration``
    above. ``"relative"``: the boundary is the
    ``deflation_null_quantile``-th quantile of the loading concentration of
    components fitted to the SAME permutation ensemble parallel analysis
    already draws for its rank test -- so it is referenced to the run's own
    data rather than to a constant.

    The option exists because the constant was measured non-invariant from four
    independent directions. On the ICGC cohort every candidate component scores
    0.1335-0.2707, so the shipped cutoff deflates nothing on the paper's own
    real-data application (WP3); the concentration is not invariant to a
    monotone rescaling of the input (WP5, the logit); not invariant to the
    data's marginal, and under a realistic marginal not invariant to cohort
    size (WP6); and on the atlas marginal the nuisance band straddles the
    cutoff -- median 0.0996, 158 of 280 runs crossing -- so the filter
    half-fires, and holding it on recovers small-cohort recall from 0.46-0.56
    to 0.72-0.83 (WP6b; the atlas arm of the bimodal-marginal sweep, generated
    by ``bimodal_marginal.py`` in eiger_paper). A permutation-null boundary
    moves with the marginal, the units and the cohort size because the
    candidates and the null are fitted on the same matrix.

    ``"relative"`` is measure-only in v2.2.0: the default is unchanged, every
    committed number stands, and whether the default should flip is a separate
    decision taken on WP7a's evidence pack (FIX_SPEC WP7b). The rule is inert
    when ``deflate_components`` is set explicitly, which overrides the filter
    entirely under either rule.

    **Measured, and not currently usable as a default.** On the signal grid the
    rule deflates almost nothing and recall collapses (the deflation-rule arm
    of the sensitivity sweep, generated by ``deflation_sensitivity.py`` in
    eiger_paper). The reason is structural rather than a matter of tuning
    ``deflation_null_quantile``: a
    candidate is the maximiser of loading concentration on the matrix it is
    scored on, while a null component is fitted where locality is broken and is,
    on that matrix, an arbitrary direction -- which cannot reach the maximiser's
    concentration. The candidate therefore sits above its own null at every
    rank, and a quantile of the null cannot rise above the candidates it is
    meant to classify. Measured on a grid-shaped matrix: two genuine
    genome-wide nuisance factors at 0.0807 and 0.0833 against a null q0.95 of
    0.0782, both deflated by the shipped 0.10 cutoff and neither by this rule.
    Kept as a measured negative result and as the scaffolding for whatever
    boundary WP7b settles on, not as a recommendation.
    """

    deflation_null_quantile: float = 0.95
    """Quantile of the permutation-null concentration distribution that becomes
    the boundary under ``deflation_rule="relative"``. Inert under
    ``"absolute"``.

    A candidate is classified nuisance when it is no more concentrated than
    this quantile of components fitted where locality is broken -- i.e. when its
    genomic loading is not distinguishable from a direction with no structure to
    load on. 0.95 is the starting value FIX_SPEC WP7a specifies; the ensemble it
    is read on is ``parallel_analysis_replicates`` permutations x the candidate
    rank, so at the shipped 5 replicates it sits near the top of a small band,
    and the whole band is recorded in the manifest so the choice can be re-read
    without re-running."""

    deflation_concentration_quantile: float = 0.01
    deflation_max_components: int = 20

    deflation_min_rows: int = 20
    """Minimum thinned CpGs required to estimate the latent basis.

    The basis is estimated only from CpGs observed in *every* sample, because a
    partially observed row contributes mean-filled zeros to the SVD. Below this
    many complete rows the estimate falls back to partially observed rows and
    records the fallback in the manifest."""

    deflation_min_observed_fraction: float = 0.80
    """Preferred completeness for a thinned CpG entering the basis under the
    fallback described in ``deflation_min_rows``. Relaxed automatically, but
    never below ``deflation_floor_observed_fraction``."""

    deflation_floor_observed_fraction: float = 0.50
    """Absolute floor on completeness for a thinned CpG entering the basis.

    A row observed in fewer than half the samples contributes more mean-fill
    than data to the SVD. If fewer than three rows clear this floor the basis is
    not estimated at all and no deflation is applied, which is recorded in the
    manifest as ``basis_source = insufficient_rows``."""
    parallel_analysis_replicates: int = 5
    parallel_analysis_quantile: float = 0.99
    deflation_missing: str = "exact"
    """``"impute"`` (fast, CpG-mean fill before projection) or ``"exact"``
    (per-CpG least squares on observed samples only)."""

    # --- null calibration
    null_replicates: int = 100
    """Independent genome-wide null realisations pooled for the region FDR.

    They are also the cluster units for the quasi-Poisson uncertainty band, so
    increasing this value improves both empirical resolution and precision.
    Whether an upper-bound decision can reach a target depends on the complete
    observed cutoff curve (including its number of discoveries), not on an
    artificial one-region denominator; see docs/FDR.md.
    """
    null_quantile: float = 0.995
    """Upper quantile that sets the edge threshold, under whichever
    ``edge_threshold`` source is in use. 0.995 is the v2.2 default for the
    ``"lag_null"`` source; the ``"observed_quantile"`` source was validated on
    the benchmark at 0.99 (docs/FDR.md section 18)."""

    edge_threshold: str = "lag_null"
    """``"lag_null"`` | ``"observed_quantile"``: where the edge threshold comes
    from. It decides which individual edges count as elevated, and therefore
    which runs become candidates. It is NOT the region-level FDR, which
    ``region_null_mode`` governs.

    ``"lag_null"`` (default): the ``null_quantile`` of the locality-breaking
    edge null -- pairs of CpGs at least ``null_min_lag`` rows apart, so the
    threshold is anchored to this cohort's cross-sample structure without
    locality. This is the v2.2 rule. On the 519-sample ICGC cohort it gives
    0.267 and 81,979 candidates. Its region-level FDR is misspecified on WGBS
    backgrounds (docs/FDR.md section 18); as a CANDIDATE threshold it is not
    a calibration claim, only an anchor, and it does not assume signal is
    sparse.

    ``"observed_quantile"``: the observed track's own ``null_quantile`` within
    joint bp-gap and sample-support bins (``observed_edge_threshold``). No
    null ensemble is built, so it is cheap, and on sparse-signal inputs it
    gave 90.9% candidate recall on the benchmark. It ASSUMES signal is sparse:
    it keeps a fixed fraction of edges whatever the biology. On the ICGC
    cohort, where a tumour axis coordinates a large share of the genome, the
    99th percentile of observed correlation was 0.932 and only 1,915
    candidates survived, 2.3% of the lag-null set. Use it only when most of
    the genome is expected to carry no coordinated signal.
    """
    null_min_lag: int = 1000
    """Minimum genomic separation, in retained CpG rows, between the CpGs paired
    to build the null. Must comfortably exceed the methylation correlation
    length."""

    null_stride: int = 1

    null_mode: str = "lag"
    """``"lag"`` is the only supported value for the edge-level null: it builds
    the null by pairing CpGs far apart on the same chromosome, testing *is
    there any regional coordination here?*

    ``"permutation"`` (edge-wise permutation of the observed statistic within
    each chromosome) is no longer available: it destroys the same run-length
    autocorrelation the region-level null depends on, measured at zero usable
    null regions. Requesting it raises immediately, naming the replacement.
    The large-cohort saturation problem it was written to fix (a null that
    becomes trivially uninformative once genuine coordination is estimated
    precisely across most of the genome) is addressed by ``region_null_mode``
    below instead.

    Sign-agnostic: a hypomethylated region produces the same positive adjacent
    correlation as a hypermethylated one. Direction is recovered afterwards
    from delta beta.
    """

    confirm_regions: bool = True
    """Run tier two: confirm each candidate from its own per-sample means.

    Hartigan's dip test on the region's per-sample logit mean betas, BH across
    the run's candidates, and a two-component fit for the effect size -- see
    ``eiger2.confirm`` and ``docs/CONFIRM.md``. It asks whether the region's
    own samples come from one population, which is answerable without a
    reference panel, unlike region FDR against the genome background.

    On by default because it is the only error rate this tool can honestly
    report unsupervised, and cheap (an interpolated-table p-value per region).
    Set ``False`` (``--no-confirm``) to report ranked candidates only; the
    columns are then absent rather than empty.
    """

    region_null_mode: str = "off"
    """``"off"`` | ``"member_step"`` | ``"lag"``. The REGION-level null
    construction, distinct from ``null_mode`` above (which is edge-level:
    which edges count as elevated). This axis decides which *runs* of elevated
    edges are significant.

    ``"off"`` (default): no region null is built at all, and no region FDR
    against the genome background is claimed. ``region_qvalue``,
    ``region_qvalue_upper_95`` and ``significant`` are refused (``NaN`` /
    ``-1`` / ``pd.NA``) with ``fdr_status = "no_calibrated_null"``, and the
    output is the ranked candidate list. This is the default because every
    self-calibrated background null measured on WGBS-realistic data failed:
    the locality-breaking null tests a hypothesis that is false everywhere on
    such a background (``docs/FDR.md`` section 18). Region FDR against the
    genome background requires matched control matrices.

    ``"auto"`` was REMOVED: it resolved to ``"member_step"`` or ``"lag"``,
    both of which are now off the default path, so a silent resolution to a
    misspecified null is exactly what must not happen. Requesting it raises,
    naming this default.

    ``"member_step"``: consecutive null "positions" step by one LAG (not one
    CpG), so they share ``window - 1`` of ``window`` members exactly as
    consecutive OBSERVED windows do -- reproducing the observed track's
    autocorrelation by construction, which ``"lag"`` (below) does not (see
    ``member_step_lag_null_ensemble``). Passes both region-level gates on real
    layouts at genome scale with recall identical to ``"lag"``; see
    ``docs/FDR.md``. Raises below its minimum-size floor rather than
    degrading silently. Available only by explicit request, and warns: its
    q-values are calibrated against the locality-breaking null whose
    misspecification is documented in ``docs/FDR.md`` section 18.

    ``"lag"``: the region-level null built the same way as the edge-level one
    -- consecutive positions slide by one CpG rather than sharing members the
    way the observed track does. **Fails region-level exchangeability on real
    layouts** -- kept for small inputs and reproducibility, available only by
    explicit request and with the same warning. Requesting either mode also
    restores the locality-breaking edge null as the source of the edge
    threshold, since the null tracks must be segmented against a threshold of
    their own.
    """

    n_lag_draws: int = 75
    """``region_null_mode="member_step"`` only: number of independent lag draws
    pooled into the null, each treated as one "replicate" -- directly
    comparable to ``null_replicates`` for the ``"lag"`` construction.

    Derived from a resolution target, not convenience: finite-null resolution
    is ``Delta = s / R`` where ``s = observed_eligible_edges /
    null_eligible_edges`` -- SMALLER ``s`` is better, since it means MORE null
    volume. Each lag draw contributes about one observed track's worth of null
    positions, so ``s ~ 1.3-1.45 / n_lag_draws``.

    A sensitivity sweep (``n_lag_draws`` in {40, 60, 75, 100, 150}, both real
    layouts, 3 seeds each, genome scale) measured `s = 0.0187-0.0195` at
    `n_lag_draws=75` -- comfortably resolving decisions at a 0.05 target -- with
    both region-level gates passing, an adequate pooled null, and recall/realised
    FDR unchanged from `150` (1.0 / 0.0 throughout) on every cell. `150` was
    chosen to match `s <= 0.01`, the resolution `null_replicates=100` was
    originally picked for; that target predates the corrected realised-FDR
    measurement (a 95% upper bound of 1.6% against the 5% nominal target, i.e.
    substantially conservative), which the extra resolution beyond `75` was not
    found to be buying. `75` cuts measured wall-clock at genome scale from
    ~219s to ~160s per cell (~27%, and closer to a 2x cut in the null
    construction's own share of that time) with no measured cost to any gate.
    Set `n_lag_draws=150` explicitly to restore the previous default if
    resolution finer than 0.02 is specifically wanted.
    """

    min_subtrack_length: int = 100
    """``region_null_mode="member_step"`` only: the PREFERRED shortest
    sub-track length. Chromosomes able to meet it draw lags from
    ``[null_min_lag, n_cpg // (min_subtrack_length + window - 1)]``.

    A chromosome that cannot meet this preferred floor is no longer discarded
    wholesale. If it can still form the configured minimum detectable region,
    it uses a measured short-subtrack fallback whose lower length is derived
    from ``min_cpgs``. The existing per-length opportunity guard then decides
    which strata received enough null exposure. Chromosomes unable to support
    even ``min_cpgs`` positions at ``null_min_lag`` remain structurally
    infeasible and are reported in the region-null exposure gate.
    """

    # --- segmentation
    min_cpgs: int = 4
    max_gap_edges: int = 0
    """Number of consecutive sub-threshold edges tolerated inside a region.

    ``2`` bridges a single dropped or filtered CpG, which breaks two edges and is
    the commonest cause of spurious fragmentation. Safe to raise because the
    null tracks are segmented with the same tolerance.
    """

    # --- data handling
    min_paired_samples: int = 5
    min_paired_fraction: float = 0.0
    """Minimum fraction of the cohort observed at *both* CpGs of an edge.

    Off by default, and measured to be counter-productive on the benchmark:
    at 25% structured missingness, raising it from 0.0 to 0.5 dropped
    recovery from 45/48 to 0/48 while *increasing* false regions from 17 to
    136 -- it removes DMR edges faster than artefact edges and fragments true
    regions below ``min_cpgs``. Exposed as a diagnostic lever for cohorts
    with a genuinely bimodal coverage profile; do not raise it without
    re-running the benchmark on your own missingness pattern.
    """
    max_cpg_gap_bp: int | None = 300
    min_observed_samples: int = 2

    min_prevalence_samples: int = 0
    """Minimum number of samples that must carry a region's deviation.

    A single aberrant sample deviating across a run of CpGs produces genuine
    adjacent correlation, so it forms a region and is not a false positive under
    any null: the coordination is real. It is simply not a differentially
    methylated *region* in the sense usually intended. On the 519-sample ICGC
    cohort 95% of otherwise unclassifiable regions were driven by one or two
    samples.

    This is a biological criterion rather than a statistical one, so it is
    applied after calibration and reported separately: ``region_qvalue`` is
    calibrated on the unfiltered set, ``minority_size_descriptive`` records the
    size of the deviating group, and ``significant`` requires both the q-value
    and this floor. Setting it to 0 disables the filter. It is exactly analogous
    to requiring a fold-change floor alongside a p-value in differential
    expression.

    **0, not 3, and measured** (FIX_SPEC WP4; the prevalence-default sweep,
    generated by ``prevalence_default.py`` in eiger_paper). Both settings were
    run over the full 280-cell ``signal_grid`` and over the ICGC cohort. In 10 of
    the 14 cells a floor of 3 costs exactly nothing -- mean recall is identical
    to the last decimal, because their planted subgroups are 4 samples or
    larger. In the four two-carrier-floor cells (prevalence 2/n, the smallest
    prevalence the grid plants) it costs 91.4-94.3 recall POINTS: mean recall
    0.99-1.00 falls to 0.05-0.09. Pooled over all 280 runs the floor removes
    641 of 2,051 significant calls but only 3 of the 90 false ones -- 638 true
    calls per 3 false -- so the mean per-run false discovery proportion RISES
    from 6.2% to 11.7%.

    On ICGC the same floor drops 1,980 of 81,979 significant regions (2.4%),
    and 1,973 of those 1,980 already carry ``biological_class ==
    "rare_epivariant"`` under ``biological_support_policy="five_class_v1"`` --
    the descriptive policy already labels them, at higher resolution than a hard
    floor and without removing anything.

    So the default stays 0 and ``eiger2 call`` instead reports how many
    significant regions are carried by one or two samples (see
    ``LOW_PREVALENCE_NOTE``). Set this to 3 when the analysis genuinely
    concerns subgroup-defining regions and rare epivariants are noise; do not
    set it on a cohort whose minority groups may be at the two-carrier floor.
    """

    min_sample_fraction: float = 0.0
    """Drop a CpG observed in fewer than this fraction of samples.

    The primary control for targeted assays: on an EPIC-style panel most rows
    are absent or nearly so, and cost time at every stage while contributing
    nothing (a sparsely covered CpG cannot form a usable edge, needing
    ``min_paired_samples`` co-observed samples, but is still centred,
    deflated and scanned). Setting this to, say, 0.5 typically removes most
    rows on a panel and is close to free statistically, since those rows
    could not have contributed edges. Dropped CpGs never appear in the
    output; region coordinates are reported in the original input indexing,
    so filtering does not shift them. 0.0 (default) disables the filter.
    """

    length_stratified_fdr: bool = True
    """Compute the region FDR within CpG-count strata (4-5, 6-9, 10-19, 20-49, 50+).

    Region area grows with length, so a pooled FDR makes short regions look worse
    than long ones for a reason that has nothing to do with evidence. Stratifying
    lets a 4-CpG DMR be judged against 4-CpG null regions. Set to ``False`` for a
    single pooled null.
    """

    # --- reporting
    target_fdr: float = 0.05
    random_seed: int = 1

    fdr_decision: str = "point"
    """``"point"``: significant = region_qvalue <= target_fdr (unchanged from
    v2.0.0). ``"upper"``: significant = region_qvalue_upper_95 <= target_fdr, the
    conservative decision under the documented replicate-cluster quasi-Poisson
    model for finite-null uncertainty. Never switches silently -- always read
    from the manifest's ``fdr_decision`` field.

    Depends on ``min_null_regions_for_fdr`` being an aggregate (not
    per-stratum) check: a stratum whose own null pool is empty still gets a
    finite positive ``K=0`` cluster upper bound rather than being refused, which is
    what makes this mode usable rather than rejecting almost everything -- see
    ``docs/FDR.md``.
    """

    biological_support_policy: str = "off"
    """``off`` or the post-selection descriptive ``five_class_v1`` policy."""

    min_null_values: int = 1_000
    """Warn below this many usable null positions, POOLED across the whole
    run -- also the ``null_construction`` structural gate's hard floor on
    that same pooled count (spec §2a.2: "too few pooled null values to
    constitute a valid construction").

    The threshold is an upper quantile and needs a populated tail. At the
    default ``null_quantile`` of 0.99 this floor leaves at least ten null
    values above the threshold, which is the minimum for it to mean anything.
    Genome-scale runs will exceed it by orders of magnitude; the floor exists to
    catch small or heavily fragmented inputs, not to set a target.

    **Not** the support-bin collapse trigger -- see
    ``min_null_values_per_support_bin``, a different (per-bin, not pooled)
    quantity."""

    min_null_values_per_support_bin: int = 1_000
    """Merge support bins upward until every bin has at least this many
    pooled null values (``_support_bin_edges``). Deliberately separate from
    ``min_null_values`` above: that is a POOLED, run-level floor now also
    backing a hard structural gate; this is a PER-BIN threshold purely about
    when support stratification is too fragmented to estimate per bin.
    Conflating the two meant a caller couldn't force bin collapse (by
    raising the per-bin floor above what any bin could reach) without the
    exact same value then failing the run outright as an inadequate pooled
    null -- two different questions with two different honest floors."""

    min_null_coverage: float = 0.25
    """Warn when the chromosome-aware mask discards more than this fraction of
    offered null positions, which happens when ``null_min_lag`` approaches the
    length of a chromosome."""

    max_missing_null_exposure: float = 0.05
    """Structural gate floor (spec §2b.3) on ``missing_exposure`` -- the share
    of this run's own eligible observed edges that sit on a chromosome with no
    feasible locality-breaking lag (``chromosome_lag_feasibility``), and so
    contribute no null volume in any replicate. Explicit and independently
    tunable (not derived from ``min_null_coverage``, a different quantity: one
    is about how much of the OFFERED null was usable, the other about how much
    of the OBSERVED genome has no null representation at all) -- sweep this
    value in the §2f grid to validate it, the same as any other floor here."""

    compute_posterior: bool = False
    """Attach the per-region two-groups evidence score, ``1 - fdr(area)``.
    ``region_qvalue`` is a tail-area rate; ``region_evidence_score`` is the
    per-region quantity from :mod:`eiger2.probability`.

    **Named a score, not a probability, because measurement says it is not
    calibrated.** On planted-truth simulations the expected calibration error
    is 0.35-0.50: regions scoring 0.86 were real 0% of the time. Cause
    (``PROBABILITY.md``): null tracks come from signal-free data, but false
    positives in real data arise beside real signal and inherit its area
    distribution, so ``f0`` is the wrong null density for them. Still a
    useful monotone ranking; not a probability, must not be thresholded as
    one.

    Off by default. A density also needs far more null regions than a tail
    count does, so enabling this without raising ``null_replicates`` (to
    100-300) will usually return ``status = insufficient_null_regions`` and
    leave the columns NaN -- intended, not a failure.
    """

    min_null_regions_for_posterior: int = 50
    """Refuse the posterior below this many pooled null regions."""

    min_null_regions_for_fdr: int = 50
    """Refuse ``region_qvalue`` in :func:`region_exchangeability_gate` (a
    structural gate) when the pooled null-region count falls below this
    floor. **No longer used by** :func:`region_fdr_result` -- spec §2c
    replaced that function's own AGGREGATE pooled-count floor with a
    per-length-stratum null-OPPORTUNITY floor; see ``m_opportunity`` below.
    The aggregate check this field used to drive there checked the wrong
    quantity: a run could hold thousands of null regions in the short strata
    while a long stratum's null never had the CHANCE to grow a region that
    long, and every long observed region then received ``q=0`` from a null
    that could never have contradicted it. That is what the audit found.
    """

    m_opportunity: float = 10.0
    """Per-length-stratum null-opportunity floor for :func:`region_fdr_result`
    (spec §2c). For length stratum ``s`` with observed opportunity ``O_s``,
    null opportunity ``N_s`` (both: sum over maximal eligible runs of
    ``max(0, run_len - min_edges(s) + 1)``, observed and pooled-null
    respectively) and ``R_s`` observed regions in that stratum, the stratum is
    adequate iff ``N_s >= m_opportunity * O_s / R_s`` -- equivalently, the
    null would have been expected to produce at least ``m_opportunity``
    regions had the observed rate applied to it. Below the floor, that
    stratum's rows (only that stratum's -- see
    ``manifest["fdr"]["insufficient_null_opportunity_strata"]``) are refused
    (``NaN`` / ``-1`` / ``significant = pd.NA``) and the run status becomes
    ``partial_inference``, not ``inference_unreliable``: the null is sound,
    it simply never had the chance to speak to regions this long.

    ``0`` disables this adequacy multiplier, but does not invent an FDR scale:
    ordinary length strata still require positive ``O_s`` and ``N_s`` for
    ``scale_s=O_s/N_s``. Adequate ``K=0`` strata are reported via
    ``zero_exceedance_strata``. ``10`` is a starting value -- sweep it in
    the §2f grid and record the value that attains nominal coverage.
    """

    support_binned_threshold: str = "auto"
    """``"auto"`` | ``"on"`` | ``"off"``. When enabled, the null-quantile
    threshold is computed per support bin (samples observed at every CpG in the
    window) instead of once genome-wide, correcting region-level FDR inflation
    under structured missingness. ``"auto"`` enables it only when observed
    support varies enough to matter (``support_dispersion >
    support_dispersion_cut``). See ``docs/FDR.md``, "Support conditioning"."""

    support_bins: int = 4
    """Quantile bins for support-conditioned thresholding. Adjacent bins merge
    upward if the pooled null has fewer than ``min_null_values_per_support_bin``
    values in a bin; collapses to the unconditioned single threshold if
    merging cannot satisfy every bin."""

    support_dispersion_cut: float | None = None
    """Auto-enable threshold for ``support_binned_threshold="auto"``. ``None``
    uses the calibrated default, ``_DEFAULT_SUPPORT_DISPERSION_CUT = 0.5``
    (see that constant's own comment for the measurement behind it)."""

    def __post_init__(self) -> None:
        if self.statistic not in {"correlation", "covariance"}:
            raise ValueError("statistic must be 'correlation' or 'covariance'")
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.deflate_components is not None and self.deflate_components < 0:
            raise ValueError("deflate_components cannot be negative")
        if self.deflation_thin < 1:
            raise ValueError("deflation_thin must be at least 1")
        if self.deflation_missing not in {"impute", "exact"}:
            raise ValueError("deflation_missing must be 'impute' or 'exact'")
        if self.deflation_rule not in {"absolute", "relative"}:
            raise ValueError("deflation_rule must be 'absolute' or 'relative'")
        if not 0.0 < self.deflation_null_quantile <= 1.0:
            raise ValueError("deflation_null_quantile must lie in (0, 1]")
        if not 0.5 < self.null_quantile < 1.0:
            raise ValueError("null_quantile must lie in (0.5, 1)")
        if self.null_replicates < 1:
            raise ValueError("null_replicates must be positive")
        if self.min_cpgs < 2:
            raise ValueError("min_cpgs must be at least 2")
        if self.max_gap_edges < 0:
            raise ValueError("max_gap_edges cannot be negative")
        if self.min_paired_samples < 3:
            raise ValueError("min_paired_samples must be at least 3")
        if self.min_null_regions_for_fdr < 0:
            raise ValueError("min_null_regions_for_fdr cannot be negative")
        if self.m_opportunity < 0:
            raise ValueError("m_opportunity cannot be negative")
        if self.null_mode not in {"lag", "permutation"}:
            raise ValueError("null_mode must be 'lag' or 'permutation'")
        if self.null_mode == "permutation":
            raise ValueError(
                "null_mode='permutation' is broken and no longer supported: "
                "measured at zero null regions across 400 replicates "
                "(edge-wise permutation destroys the same autocorrelation "
                "the corrected 'lag' null exists to preserve). Use the "
                "default null_mode='lag' for the edge-level null; for the "
                "region-level null, see region_null_mode='member_step', "
                "which must now be requested by name."
            )
        if self.region_null_mode == "auto":
            raise ValueError(
                "region_null_mode='auto' was removed: it resolved silently to "
                "'member_step' or 'lag', both of which are calibrated against "
                "the locality-breaking null that is misspecified on WGBS "
                "backgrounds (docs/FDR.md section 18). Use the default "
                "region_null_mode='off' for ranked candidates, or request "
                "'member_step'/'lag' by name and accept the warning."
            )
        if self.edge_threshold not in {"lag_null", "observed_quantile"}:
            raise ValueError(
                "edge_threshold must be 'lag_null' or 'observed_quantile'"
            )
        if self.region_null_mode not in {"member_step", "lag", "off"}:
            raise ValueError(
                "region_null_mode must be 'off', 'member_step', or 'lag'"
            )
        if self.n_lag_draws < 1:
            raise ValueError("n_lag_draws must be positive")
        if self.min_subtrack_length < 1:
            raise ValueError("min_subtrack_length must be positive")
        if not 0 <= self.min_paired_fraction <= 1:
            raise ValueError("min_paired_fraction must lie in [0, 1]")
        if not 0 < self.target_fdr < 1:
            raise ValueError("target_fdr must lie in (0, 1)")
        if self.fdr_decision not in {"point", "upper"}:
            raise ValueError("fdr_decision must be 'point' or 'upper'")
        if self.biological_support_policy not in {"off", "five_class_v1"}:
            raise ValueError(
                "biological_support_policy must be 'off' or 'five_class_v1'"
            )
        if self.support_binned_threshold not in {"auto", "on", "off"}:
            raise ValueError("support_binned_threshold must be 'auto', 'on', or 'off'")
        if self.support_bins < 2:
            raise ValueError("support_bins must be at least 2")

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# step 1 — centring
# --------------------------------------------------------------------------


def centre_cpgs(betas: np.ndarray) -> np.ndarray:
    """Subtract the per-CpG mean over observed samples. NaN stays NaN."""

    values = np.asarray(betas, dtype=float)
    if values.ndim != 2:
        raise ValueError("betas must be a two-dimensional CpG-by-sample matrix")
    # A CpG with no observation in any sample -- an off-panel or failed probe --
    # yields an empty slice here. That is expected on targeted assays, so the
    # warning is suppressed and the mean is set to zero; the row is dropped by
    # ``min_observed_samples`` in ``call_dmrs`` before it can reach the
    # statistic.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(np.where(np.isfinite(values), values, np.nan), axis=1)
    means = np.where(np.isfinite(means), means, 0.0)
    return values - means[:, None]


# --------------------------------------------------------------------------
# step 2 — latent structure
# --------------------------------------------------------------------------


def _fill_rows(residuals: np.ndarray) -> np.ndarray:
    """Fill missing entries with 0 (the CpG mean, post-centring)."""

    filled = np.array(residuals, dtype=float, copy=True)
    filled[~np.isfinite(filled)] = 0.0
    return filled


def _loading_concentration(
    filled: np.ndarray, directions: np.ndarray, quantile: float
) -> np.ndarray:
    """Share of each direction's genomic loading energy in its top CpGs.

    ``filled`` is the mean-filled residual matrix, ``directions`` a ``(K,
    n_samples)`` block of orthonormal sample-space rows. A nuisance factor
    (purity, composition, batch) spreads its loading over the whole genome and
    scores low; a biological subgroup concentrates it in marker regions and
    scores high. See ``Eiger2Config.deflation_max_concentration`` for the
    measured separation.

    Factored out of ``estimate_latent_basis`` for FIX_SPEC WP7a: the relative
    rule's boundary is only meaningful if the permutation-null directions are
    scored by exactly the routine that scores the candidates, on exactly the
    same matrix. Sharing the code is the guarantee; the arithmetic is unchanged
    from the pre-WP7a inline loop, so the absolute path's concentrations are
    identical to the bit.
    """

    loadings = filled @ directions.T
    top_n = max(1, int(round(quantile * loadings.shape[0])))
    concentration = np.empty(directions.shape[0], dtype=float)
    for component in range(directions.shape[0]):
        energy = loadings[:, component] ** 2
        total = energy.sum()
        concentration[component] = (
            float(np.sort(energy)[-top_n:].sum() / total) if total > 0 else 1.0
        )
    return concentration


def estimate_latent_basis(
    residuals: np.ndarray, config: Eiger2Config
) -> tuple[np.ndarray, dict]:
    """Estimate genome-wide sample-space directions from a thinned CpG set.

    Returns ``(V, diagnostics)`` where ``V`` has shape ``(K, n_samples)`` with
    orthonormal rows. ``K == 0`` yields an empty basis.
    """

    residuals = np.asarray(residuals, dtype=float)
    n_samples = residuals.shape[1]
    raw = residuals[:: config.deflation_thin]
    observed = np.isfinite(raw)

    # The basis must be estimated on rows that are actually observed. Filling
    # first and filtering afterwards makes the filter inert, because nothing is
    # non-finite once the gaps are zeroed, and the zeros then enter the SVD as
    # if they were data. On targeted or panel assays, where per-CpG coverage
    # varies widely, that pulls the estimated basis towards zero exactly in the
    # rows with least information.
    complete = observed.all(axis=1) & (np.nansum(np.abs(raw), axis=1) > 0)
    basis_source = "complete_rows"
    thinned = raw[complete]

    relaxed_to = 1.0
    if thinned.shape[0] < config.deflation_min_rows:
        # Too few complete rows, which is the normal situation on targeted
        # panels and on any cohort with appreciable dropout. Relax the
        # completeness requirement only as far as necessary to reach
        # ``deflation_min_rows``, never below ``deflation_floor_observed_fraction``,
        # and mean-fill the remainder. The threshold actually reached is recorded
        # because it determines how much fill enters the basis.
        fraction = observed.mean(axis=1)
        nonzero = np.nansum(np.abs(raw), axis=1) > 0
        eligible = nonzero & (fraction >= config.deflation_floor_observed_fraction)
        ranked = np.sort(fraction[eligible])[::-1]
        if ranked.size >= 3:
            index = min(config.deflation_min_rows, ranked.size) - 1
            relaxed_to = float(
                min(config.deflation_min_observed_fraction, ranked[index])
            )
            usable = nonzero & (fraction >= relaxed_to)
            thinned = _fill_rows(raw[usable])
            basis_source = "partially_observed_rows_mean_filled"
        else:
            thinned = raw[complete]
            basis_source = "insufficient_rows"

    diagnostics: dict = {
        "thin_stride": int(config.deflation_thin),
        "thin_rows_offered": int(raw.shape[0]),
        "thin_rows_complete": int(complete.sum()),
        "thin_rows_used": int(thinned.shape[0]),
        "basis_source": basis_source,
        "min_observed_fraction_used": relaxed_to,
    }
    if basis_source != "complete_rows":
        diagnostics["warning"] = (
            f"only {int(complete.sum())} of {int(raw.shape[0])} thinned CpGs are "
            f"observed in every sample, so the latent basis was estimated from "
            f"{basis_source}. Reduce deflation_thin to offer more rows, or set "
            f"deflate_components=0 to disable deflation."
        )
    max_components = min(
        config.deflation_max_components, n_samples - 1, max(thinned.shape[0] - 1, 0)
    )
    if thinned.shape[0] < 3 or max_components < 1:
        diagnostics.update(n_components=0, method="insufficient_rows")
        return np.empty((0, n_samples), dtype=float), diagnostics

    _, singular, right = np.linalg.svd(thinned, full_matrices=False)

    # FIX_SPEC §WP7a item 2: under deflation_rule="relative" the boundary is read
    # off the SAME permutation ensemble the rank test already draws, so the null
    # components' directions have to be kept as well as their singular values.
    # Kept only when that rule is in force: under "absolute" the SVD call, its
    # LAPACK driver and therefore every committed number are untouched, and the
    # rng is consumed identically either way (the permutations are drawn before
    # the branch, in the same order, whatever is done with them afterwards).
    relative = config.deflation_rule == "relative"
    null_directions: list[np.ndarray] = []
    if config.deflate_components is not None:
        n_components = int(min(config.deflate_components, max_components))
        diagnostics["method"] = "fixed"
    else:
        rng = np.random.default_rng(config.random_seed)
        null_singular = np.empty(
            (config.parallel_analysis_replicates, singular.size), dtype=float
        )
        for replicate in range(config.parallel_analysis_replicates):
            permuted = np.empty_like(thinned)
            for column in range(n_samples):
                permuted[:, column] = rng.permutation(thinned[:, column])
            if relative:
                _, null_values, null_right = np.linalg.svd(
                    permuted, full_matrices=False
                )
                null_singular[replicate] = null_values
                null_directions.append(null_right)
            else:
                null_singular[replicate] = np.linalg.svd(permuted, compute_uv=False)
        cutoff = np.quantile(
            null_singular, config.parallel_analysis_quantile, axis=0
        )
        exceeds = singular > cutoff
        n_components = int(np.argmin(exceeds)) if not exceeds.all() else exceeds.size
        n_components = int(min(n_components, max_components))
        diagnostics["method"] = "parallel_analysis"
        diagnostics["singular_values"] = singular[:10].tolist()
        diagnostics["null_cutoff"] = cutoff[:10].tolist()

    diagnostics["n_candidates"] = n_components
    if n_components == 0:
        diagnostics["n_components"] = 0
        return np.empty((0, n_samples), dtype=float), diagnostics

    # Keep only the candidates whose genomic loading is diffuse. A nuisance
    # factor spreads its loading over the whole genome; a biological subgroup
    # concentrates it in marker regions and must not be deflated away.
    candidates = np.ascontiguousarray(right[:n_components])
    filled_full = _fill_rows(residuals)
    quantile = config.deflation_concentration_quantile
    concentration = _loading_concentration(filled_full, candidates, quantile)

    # The boundary the concentrations are read against. Under "absolute" it is
    # the shipped constant, so this whole block reduces to the pre-WP7a
    # comparison; under "relative" it is a quantile of the concentration of
    # components fitted where locality is broken -- the same directions the rank
    # test permuted, scored through the same routine on the same residual
    # matrix, so candidate and null differ in exactly one thing: whether the
    # sample-space direction was fitted to real structure or to permuted noise.
    boundary = float(config.deflation_max_concentration)
    boundary_source = "deflation_max_concentration"
    null_concentration = np.array([], dtype=float)
    if relative and null_directions:
        null_concentration = np.concatenate([
            _loading_concentration(
                filled_full,
                np.ascontiguousarray(null_right[:n_components]),
                quantile,
            )
            for null_right in null_directions
        ])
        boundary = float(
            np.quantile(null_concentration, config.deflation_null_quantile)
        )
        boundary_source = "permutation_null_quantile"

    if config.deflate_components is not None:
        # an explicit K is a user override; report the verdicts but honour it
        diffuse = np.ones(n_components, dtype=bool)
    else:
        diffuse = concentration <= boundary
    diagnostics["loading_concentration"] = concentration.round(4).tolist()
    diagnostics["component_verdict"] = [
        "nuisance_deflated" if flag else "biological_kept" for flag in diffuse
    ]
    diagnostics["n_components"] = int(diffuse.sum())
    # FIX_SPEC §WP6b item 2: every candidate sitting within
    # DEFLATION_NEAR_CUTOFF_MARGIN of the cutoff, on either side, named with its
    # concentration and the cutoff it is near. Recorded on every run that gets as
    # far as scoring candidates, empty list included, so a consumer can tell
    # "none near the cutoff" from "not computed". The verdict travels with each
    # entry because the note is about the decision being marginal, not about
    # which way it went.
    diagnostics["deflation_max_concentration"] = float(
        config.deflation_max_concentration
    )
    diagnostics["deflation_near_cutoff_margin"] = DEFLATION_NEAR_CUTOFF_MARGIN
    # FIX_SPEC §WP7a item 2: the realised boundary is recorded exactly as the
    # absolute path records its cutoff, plus -- under the relative rule -- the
    # null band it was read off, so a consumer can re-read the decision at
    # another quantile without re-running. Under "absolute" `deflation_boundary`
    # equals `deflation_max_concentration` by construction and the null band is
    # absent, so the two paths' manifests differ only where they genuinely do.
    diagnostics["deflation_rule"] = config.deflation_rule
    diagnostics["deflation_boundary"] = float(boundary)
    diagnostics["deflation_boundary_source"] = boundary_source
    if null_concentration.size:
        diagnostics["deflation_null_quantile"] = float(
            config.deflation_null_quantile
        )
        diagnostics["deflation_null_concentration"] = {
            "n": int(null_concentration.size),
            "n_replicates": int(len(null_directions)),
            "min": float(null_concentration.min()),
            "median": float(np.median(null_concentration)),
            "max": float(null_concentration.max()),
        }
    diagnostics["near_cutoff_components"] = [
        {
            "component": int(index),
            "loading_concentration": float(round(value, 4)),
            "verdict": (
                "nuisance_deflated" if diffuse[index] else "biological_kept"
            ),
        }
        for index, value in enumerate(concentration)
        if abs(float(value) - boundary) <= DEFLATION_NEAR_CUTOFF_MARGIN
    ]
    if not diffuse.any():
        return np.empty((0, n_samples), dtype=float), diagnostics
    return np.ascontiguousarray(candidates[diffuse]), diagnostics


def deflate(
    residuals: np.ndarray, basis: np.ndarray, *, missing: str = "impute"
) -> np.ndarray:
    """Project ``basis`` out of every CpG row, preserving the missing mask."""

    residuals = np.asarray(residuals, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if basis.size == 0:
        return residuals.copy()
    if basis.shape[1] != residuals.shape[1]:
        raise ValueError("basis and residuals must share the sample axis")

    observed = np.isfinite(residuals)
    if missing == "impute" or observed.all():
        filled = _fill_rows(residuals)
        deflated = filled - (filled @ basis.T) @ basis
    elif missing == "exact":
        deflated = np.array(residuals, dtype=float, copy=True)
        for row in range(residuals.shape[0]):
            mask = observed[row]
            if mask.sum() <= basis.shape[0]:
                deflated[row, mask] = 0.0
                continue
            design = basis[:, mask].T
            target = residuals[row, mask]
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
            deflated[row, mask] = target - design @ coefficients
    else:
        raise ValueError("missing must be 'impute' or 'exact'")

    deflated[~observed] = np.nan
    return deflated


# --------------------------------------------------------------------------
# step 3 — the local co-variation statistic
# --------------------------------------------------------------------------

# Optional accelerated kernel (the `fast` extra, `pip install eiger[fast]`).
# numba is NOT a hard dependency: every path below works identically, if more
# slowly, without it. Import failure of any kind (not installed, incompatible
# version, no working C compiler for the JIT) falls back to the pure-numpy
# path silently -- this must never be the reason a user's analysis fails.
#
# Even when numba IS installed, this kernel is opt-in, not automatic: it
# computes the identical quantity via a different summation order (a scalar
# accumulation loop, not numpy's vectorised reduction), which differs from
# the numpy path at the ~1e-15 floating-point noise floor (see
# test_pair_statistic_numba_matches_numpy_nan_pattern_exactly for the
# measured bound). That is invisible at any statistical scale this method
# operates at, but it is enough to break exact-equality invariant fixtures
# elsewhere in the test suite, so a caller must ask for it explicitly --
# `EIGER2_USE_NUMBA=1` in the environment (the parallel paper-grid sweep this
# kernel exists for sets it in every worker) -- rather than every user with
# numba installed for an unrelated reason (e.g. the `legacy` extra) silently
# getting numerically different results from the same input.
try:
    from numba import njit, prange

    @njit(cache=True, parallel=True)
    def _pair_statistic_numba(
        left: np.ndarray,
        right: np.ndarray,
        want_correlation: bool,
        required: int,
    ) -> np.ndarray:
        # fastmath=True MUST NOT be added to this decorator. It was measured
        # to produce an apparent 60x speedup that was entirely fake:
        # fastmath permits the compiler to assume no NaNs and elides the
        # `l == l` finiteness check below, so the kernel silently returned
        # ALL-NaN. Every downstream gate happened to report "no data" rather
        # than a wrong number that time, but a different code path could as
        # easily return a plausible wrong answer instead. See
        # test_pair_statistic_numba_matches_numpy_nan_pattern_exactly, which
        # would catch a regression here.
        n_pairs, n_samples = left.shape
        out = np.empty(n_pairs, dtype=np.float64)
        for i in prange(n_pairs):
            count = 0
            sum_l = 0.0
            sum_r = 0.0
            sum_lr = 0.0
            sum_ll = 0.0
            sum_rr = 0.0
            for j in range(n_samples):
                lv = left[i, j]
                rv = right[i, j]
                if lv == lv and rv == rv:  # finite check; NaN != NaN
                    count += 1
                    sum_l += lv
                    sum_r += rv
                    sum_lr += lv * rv
                    if want_correlation:
                        sum_ll += lv * lv
                        sum_rr += rv * rv
            if count < 2 or count < required:
                out[i] = np.nan
                continue
            denom = count - 1
            cross = sum_lr - sum_l * sum_r / count
            covariance = cross / denom
            if want_correlation:
                var_l = (sum_ll - sum_l * sum_l / count) / denom
                var_r = (sum_rr - sum_r * sum_r / count) / denom
                scale = np.sqrt(var_l * var_r)
                out[i] = covariance / scale if scale > 0 else np.nan
            else:
                out[i] = covariance
        return out

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _pair_statistic_numpy(
    left: np.ndarray,
    right: np.ndarray,
    statistic: str,
    min_paired_samples: int,
    min_paired_fraction: float = 0.0,
) -> np.ndarray:
    """Pairwise-complete covariance or correlation, row by row -- the
    reference numpy implementation, always available and always correct.

    Pairwise-complete matters: filling missing entries with the CpG mean
    contributes a zero residual to the cross-product while still counting in the
    denominator, which deflates the statistic by exactly ``(1 - m)**2`` at
    missing rate ``m``. Missingness in WGBS tracks GC content, mappability and
    coverage, so that is a structured bias, not noise.
    """

    mask = np.isfinite(left) & np.isfinite(right)
    counts = mask.sum(axis=1)
    left_z = np.where(mask, left, 0.0)
    right_z = np.where(mask, right, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        safe = np.where(counts > 0, counts, 1).astype(float)
        sum_left = left_z.sum(axis=1)
        sum_right = right_z.sum(axis=1)
        cross = (left_z * right_z).sum(axis=1) - sum_left * sum_right / safe
        denominator = np.where(counts > 1, counts - 1, np.nan).astype(float)
        covariance = cross / denominator
        if statistic == "covariance":
            result = covariance
        else:
            var_left = (
                (left_z * left_z).sum(axis=1) - sum_left * sum_left / safe
            ) / denominator
            var_right = (
                (right_z * right_z).sum(axis=1) - sum_right * sum_right / safe
            ) / denominator
            scale = np.sqrt(var_left * var_right)
            result = np.where(scale > 0, covariance / scale, np.nan)

    required = max(
        int(min_paired_samples),
        int(np.ceil(min_paired_fraction * left.shape[1])),
    )
    return np.where(counts >= required, result, np.nan)


def _pair_statistic(
    left: np.ndarray,
    right: np.ndarray,
    statistic: str,
    min_paired_samples: int,
    min_paired_fraction: float = 0.0,
) -> np.ndarray:
    """Pairwise-complete covariance or correlation, row by row.

    Uses the fused numba kernel above when the optional ``fast`` extra is
    installed AND ``EIGER2_USE_NUMBA=1`` is set in the environment
    (bit-for-bit NaN pattern, max abs difference <= 1e-12 against
    ``_pair_statistic_numpy`` -- see the acceptance test); otherwise defers
    to ``_pair_statistic_numpy`` directly. See the module-level comment above
    the kernel for why this is opt-in rather than automatic.
    """

    if not (_HAS_NUMBA and os.environ.get("EIGER2_USE_NUMBA") == "1"):
        return _pair_statistic_numpy(
            left, right, statistic, min_paired_samples, min_paired_fraction,
        )
    required = max(
        int(min_paired_samples),
        int(np.ceil(min_paired_fraction * left.shape[1])),
    )
    return _pair_statistic_numba(
        np.ascontiguousarray(left, dtype=np.float64),
        np.ascontiguousarray(right, dtype=np.float64),
        statistic != "covariance",
        required,
    )


def _lag_statistic(
    residuals: np.ndarray, lag: int, config: Eiger2Config, chunk: int = 200_000
) -> np.ndarray:
    """Statistic between CpG ``i`` and CpG ``i + lag`` for every valid ``i``."""

    n_rows = residuals.shape[0]
    n_pairs = max(n_rows - lag, 0)
    out = np.full(n_pairs, np.nan, dtype=float)
    for start in range(0, n_pairs, chunk):
        stop = min(start + chunk, n_pairs)
        out[start:stop] = _pair_statistic(
            residuals[start:stop],
            residuals[start + lag : stop + lag],
            config.statistic,
            config.min_paired_samples,
            config.min_paired_fraction,
        )
    return out


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
    has at least ``min_null_values`` pooled null values.

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
        edges_array = np.array(edges) if edges else np.array([])
        bins = _assign_support_bin(pooled_null_support, edges_array)
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


def _window_from_lags(lag_statistics: list[np.ndarray], window: int) -> np.ndarray:
    """Mean off-diagonal statistic of every ``window``-CpG block.

    ``lag_statistics[d - 1][i]`` is the statistic between CpG ``i`` and
    ``i + d``. The window centred on edge ``e`` covers CpGs
    ``[e - window//2 + 1, e + window//2]``; the returned array is indexed by
    edge, so it aligns with the adjacent-pair track.
    """

    if window == 1:
        return lag_statistics[0].copy()
    n_edges = lag_statistics[0].size
    n_rows = n_edges + 1
    offset = window // 2 - 1
    totals = np.zeros(n_edges, dtype=float)
    counts = np.zeros(n_edges, dtype=float)
    edges = np.arange(n_edges)
    starts = edges - offset
    for lag in range(1, window):
        values = lag_statistics[lag - 1]
        padded = np.concatenate(([0.0], np.nan_to_num(values, nan=0.0)))
        valid = np.concatenate(([0.0], np.isfinite(values).astype(float)))
        cumulative = np.cumsum(padded)
        cumulative_valid = np.cumsum(valid)
        # pairs (i, i+lag) with start <= i and i + lag <= start + window - 1
        lower = np.clip(starts, 0, values.size)
        upper = np.clip(starts + window - lag, 0, values.size)
        upper = np.maximum(upper, lower)
        totals += cumulative[upper] - cumulative[lower]
        counts += cumulative_valid[upper] - cumulative_valid[lower]
    del n_rows
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(counts > 0, totals / np.maximum(counts, 1.0), np.nan)
    expected = window * (window - 1) / 2.0
    return np.where(counts >= 0.5 * expected, result, np.nan)


def edge_statistic(
    residuals: np.ndarray, config: Eiger2Config, chunk: int = 200_000
) -> np.ndarray:
    """Co-variation statistic for every adjacent CpG edge."""

    residuals = np.asarray(residuals, dtype=float)
    if residuals.shape[0] < 2:
        return np.empty(0, dtype=float)
    lags = [
        _lag_statistic(residuals, lag, config, chunk=chunk)
        for lag in range(1, max(config.window, 2))
    ]
    return _window_from_lags(lags, config.window)


def eligible_edges(
    locations: pd.DataFrame, config: Eiger2Config
) -> np.ndarray:
    """Which adjacent edges may sit inside the same region."""

    chrom = locations["chr"].to_numpy()
    same = chrom[:-1] == chrom[1:]
    if config.max_cpg_gap_bp is None:
        return same
    starts = locations["start"].to_numpy(dtype=np.int64)
    ends = locations["end"].to_numpy(dtype=np.int64)
    gaps = starts[1:] - ends[:-1]
    return same & (gaps >= 0) & (gaps <= config.max_cpg_gap_bp)


# --------------------------------------------------------------------------
# step 4 — the locality-breaking null
# --------------------------------------------------------------------------


def _lag_null_track(
    residuals: np.ndarray, lag: int, config: Eiger2Config
) -> np.ndarray:
    """One null track: window members are ``lag`` CpGs apart, not adjacent.

    The window at null position ``i`` comprises CpGs ``i, i+lag, ..., i+(w-1)*lag``
    (rows ``i, i+lag, i+2*lag, ...`` instead of ``i, i+1, i+2, ...``) and the
    statistic is the mean over the complete pairwise set among *those*
    members: pair ``(a, b)`` is ``S_{b-a}[i + a*lag]`` where
    ``S_d[j] = stat(CpG j, CpG j + d*lag)``. Reusing ``_window_from_lags``
    here would be wrong: it assumes members sit at consecutive indices, which
    is true of the observed track and false of this one. See docs/FDR.md,
    "Null window assembly".

    Every sample moves together, so batch, purity, cell composition and every
    other cross-sample dependency survives intact -- that is what EIGER 1's
    per-sample resampling null destroyed, which is why its threshold came out
    several-fold too low and DMR-free matrices lit up. Consecutive null
    positions still slide by one CpG, so locally autocorrelated nuisance
    (runs of low coverage or high missingness) is preserved as well. Only
    adjacency, and therefore only regional coordination, is broken.
    """

    w = config.window
    span = max(w, 2)
    usable = residuals.shape[0] - lag * (span - 1)
    if usable < 2:
        return np.empty(0, dtype=float)

    S = {d: _lag_statistic(residuals, d * lag, config) for d in range(1, span)}
    if w == 1:
        return S[1][:usable]

    assert (w - 2) * lag + usable <= S[1].size if w >= 3 else True, (
        "slice bounds violated -- see docs/FDR.md, 'Null window assembly'"
    )

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


def _null_eligibility(
    chromosome: np.ndarray | None,
    lag: int,
    config: Eiger2Config,
    size: int,
    observed_eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Which null positions are admissible, by the same rule as the data.

    The observed track is masked by :func:`eligible_edges`, which forbids an
    edge spanning two chromosomes OR whose bp gap exceeds ``max_cpg_gap_bp``.
    The null must be masked the same way, or the two are not passed through
    the identical segmenter and the calibration argument does not hold. A
    null position ``i`` at lag ``L`` draws on rows ``i, i+L, ..., i+(w-1)L``;
    the chromosome check below requires all of them lie on one chromosome.

    ``observed_eligible`` is the SAME ``eligible_edges()`` array the observed
    track is held to, additionally requiring the LOCAL adjacent edge at EVERY
    member offset (not just the chromosome loop's ``w-1`` pairs) to itself be
    gap-eligible -- each null member must sit in as ordinary a local
    neighbourhood as any observed CpG does (16-50% of null positions violate
    this on realistic layouts if left unchecked). Deliberately NOT applied to
    the long lag ``L`` *between* members -- that gap is the intentional
    locality-breaking part of the construction. ``observed_eligible=None``
    (the default) skips the check and is a no-op on ``dense`` regardless.
    """

    if chromosome is None:
        eligible = np.ones(size, dtype=bool)
    else:
        span = max(config.window, 2)
        eligible = np.ones(size, dtype=bool)
        first = chromosome[:size]
        for step in range(1, span):
            offset = step * lag
            if offset + size > chromosome.size:
                return np.zeros(size, dtype=bool)
            eligible &= first == chromosome[offset : offset + size]

    if observed_eligible is not None:
        # `observed_eligible` is edge-indexed (n_cpg - 1 entries: entry k is the
        # edge AFTER CpG k), one shorter than the CpG-indexed `chromosome` array
        # the loop above bounds against. A member sitting at the very last CpG
        # has no following edge to check -- not a violation, just nothing to
        # check -- so pad one trailing `True` (vacuously eligible) rather than
        # bounds-failing the whole null position over a member that has no edge
        # to have failed.
        padded = np.concatenate([observed_eligible, [True]])
        w = max(config.window, 1)
        for a in range(w):
            offset = a * lag
            if offset + size > padded.size:
                return np.zeros(size, dtype=bool)
            eligible &= padded[offset : offset + size]
    return eligible


def _chromosome_blocks(chromosome: np.ndarray) -> list[tuple[str, int, int]]:
    """``(name, start, end)`` for each contiguous chromosome block in
    ``chromosome`` (``end`` exclusive), in the order the blocks appear.

    Assumes rows are already grouped by chromosome -- true after
    :func:`chromosome_sort_permutation` (spec §2b prerequisite), which every
    caller of this function runs first. A chromosome appearing twice,
    non-contiguously, would silently split into two blocks here; nothing
    re-validates that upstream guarantee at this point.
    """

    if chromosome.size == 0:
        return []
    boundaries = np.flatnonzero(chromosome[1:] != chromosome[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [chromosome.size]))
    return [
        (str(chromosome[s]), int(s), int(e))
        for s, e in zip(starts, ends, strict=True)
    ]


def chromosome_lag_feasibility(
    locations: pd.DataFrame,
    config: Eiger2Config,
    *,
    observed_eligible: np.ndarray | None = None,
) -> dict[str, dict]:
    """Per chromosome: whether a feasible locality-breaking lag exists, and
    that chromosome's own share of eligible observed edges (spec §2b.1/§2b.3).

    A chromosome admits a feasible lag only if ``l_max >= config.null_min_lag``,
    where ``l_max = (n_c - 2) // (span - 1)`` -- the largest lag at which
    :func:`_lag_null_track` still has at least 2 usable positions
    (``usable = n_c - lag * (span - 1) >= 2``; matching the pre-§2b
    genome-wide code's own ``largest = (n_rows - 2) // (span - 1)``
    convention, not just ``n_c // (span - 1)``, which overestimates ``l_max``
    by enough that a lag drawn at the reported boundary can still leave
    ``usable < 2`` and silently produce an empty piece). The largest
    chromosome(s) dominate a single genome-wide lag draw (spec §2b.1); this
    measures feasibility PER chromosome instead, the basis both for
    :func:`null_tracks`'s per-chromosome construction (§2b.2) and for the
    missing-exposure diagnostic an infeasible chromosome is reported under
    (§2b.3), rather than excluding it behind a fixed, arbitrary threshold.

    ``observed_eligible`` lets a caller that already computed
    ``eligible_edges(locations, config)`` pass it straight through instead
    of this function recomputing it.
    """

    chromosome = locations["chr"].to_numpy()
    if observed_eligible is None:
        observed_eligible = eligible_edges(locations, config)
    span = max(config.window, 2)
    result: dict[str, dict] = {}
    for name, start, end in _chromosome_blocks(chromosome):
        n_c = end - start
        l_max = max(n_c - 2, 0) // max(span - 1, 1)
        # observed_eligible is edge-indexed (one shorter than CpG-indexed):
        # this chromosome's OWN internal edges are entries start..end-2; the
        # boundary edge at end-1 (into the NEXT chromosome, if any) belongs
        # to neither block and is correctly excluded by the end-1 stop.
        edges = int(observed_eligible[start : max(end - 1, start)].sum())
        result[name] = {
            "n_cpg": int(n_c),
            "l_max": int(l_max),
            "feasible": bool(l_max >= config.null_min_lag),
            "eligible_observed_edges": edges,
        }
    return result


def member_step_chromosome_feasibility(
    locations: pd.DataFrame,
    config: Eiger2Config,
    *,
    observed_eligible: np.ndarray | None = None,
) -> dict[str, dict]:
    """Per-chromosome feasibility and observed-edge exposure for the
    member-step region null.

    ``min_subtrack_length`` is preferred, not an all-or-nothing chromosome
    floor.  When it cannot be met, a chromosome may still contribute through
    a short-subtrack fallback down to ``min_cpgs`` positions -- exactly the
    shortest track capable of carrying ``min_cpgs - 1`` passing edges after
    the forced boundary break.  The per-length opportunity guard, rather than
    this constructor, decides whether those shorter tracks provide enough
    information for each reported length stratum.
    """

    chromosome = locations["chr"].to_numpy()
    if observed_eligible is None:
        observed_eligible = eligible_edges(locations, config)
    result: dict[str, dict] = {}
    for name, start, end in _chromosome_blocks(chromosome):
        n_c = end - start
        limits = _member_step_lag_limits(n_c, config)
        edges = int(observed_eligible[start : max(end - 1, start)].sum())
        result[name] = {
            "n_cpg": int(n_c),
            **limits,
            "eligible_observed_edges": edges,
        }
    return result


def _member_step_lag_limits(n_cpg: int, config: Eiger2Config) -> dict:
    """Return the preferred/fallback member-step lag limits for one block.

    A sub-track of ``t`` statistic positions needs ``lag * (t + window - 1)``
    CpG rows under the member-step slicing convention.  ``t=min_cpgs`` is the
    absolute construction floor: the last statistic position is made
    ineligible to prevent cross-offset merging, leaving ``min_cpgs - 1``
    consecutive passing edges, exactly enough for a ``min_cpgs`` region.
    """

    w = max(config.window, 1)
    preferred_span = config.min_subtrack_length + w - 1
    minimum_span = config.min_cpgs + w - 1
    preferred_l_max = int(n_cpg // preferred_span)
    fallback_l_max = int(n_cpg // minimum_span)
    if preferred_l_max >= config.null_min_lag:
        construction = "preferred"
        l_max = preferred_l_max
    elif fallback_l_max >= config.null_min_lag:
        construction = "fallback"
        l_max = fallback_l_max
    else:
        construction = "infeasible"
        l_max = fallback_l_max
    return {
        "l_max": int(l_max),
        "preferred_l_max": preferred_l_max,
        "fallback_l_max": fallback_l_max,
        "preferred_subtrack_length": int(config.min_subtrack_length),
        "minimum_subtrack_length": int(config.min_cpgs),
        "construction": construction,
        "feasible": construction != "infeasible",
    }


def null_tracks(
    residuals: np.ndarray,
    config: Eiger2Config,
    locations: pd.DataFrame | None = None,
) -> list[tuple[np.ndarray, np.ndarray, list[tuple[str, int, int, int]]]]:
    """Statistic tracks under the locality-breaking null, one per replicate.

    Chromosome-aware (spec §2b.2): each replicate draws an INDEPENDENT
    feasible lag per chromosome -- not one genome-wide lag applied
    everywhere -- builds that chromosome's null segment from its own rows,
    and concatenates every feasible chromosome's segment into one
    genome-wide realisation, with the eligibility mask broken at every
    concatenation seam so a window can never straddle two chromosomes'
    segments. This preserves chromosome composition in every replicate:
    each chromosome contributes null volume roughly in proportion to its own
    share of eligible observed edges. A single genome-wide lag draw does not
    -- it survives almost entirely on the largest chromosome(s), since a
    lag large enough to be locality-breaking on the whole genome is rarely
    small enough to fit inside a smaller one (spec §2b.1; the audited ICGC
    run: 7 of 100 tracks survived, drawn almost entirely from the largest
    chromosomes, though the observed track spans all of them).

    Chromosomes with no feasible lag (``n_c // (span - 1) < null_min_lag``)
    contribute nothing to the null in any replicate; their exposure is
    measured separately by :func:`chromosome_lag_feasibility` (spec §2b.3),
    not this function's concern.

    Returns ``(track, eligible, segments)`` triples. ``segments`` is
    ``[(chromosome, lag, start, length), ...]`` in concatenation order,
    describing which slice of ``track``/``eligible`` came from which
    chromosome, at what lag, and at what genome-wide row offset (``start``)
    -- needed by the support-conditioned threshold to reconstruct each
    segment's own window-member set (mirroring ``_lag_null_track``'s
    construction) against ``deflated``'s genome-wide row indexing when
    computing that segment's own support; a single scalar lag can no longer
    describe a whole replicate now that one replicate spans multiple
    chromosomes' own lags.

    Passing ``locations=None`` treats every row as one chromosome (the
    single-chromosome / genome-wide construction, unchanged) and is retained
    for single-chromosome use and for tests that do not exercise chromosome
    awareness at all.
    """

    span = max(config.window, 2)
    if locations is None:
        chromosome = np.full(residuals.shape[0], "", dtype=object)
        l_max = max(residuals.shape[0] - 2, 0) // max(span - 1, 1)
        feasibility = {
            "": {
                "n_cpg": residuals.shape[0],
                "l_max": l_max,
                "feasible": l_max >= config.null_min_lag,
            }
        }
        observed_eligible = None
    else:
        chromosome = locations["chr"].to_numpy()
        observed_eligible = eligible_edges(locations, config)
        feasibility = chromosome_lag_feasibility(
            locations, config, observed_eligible=observed_eligible,
        )

    blocks = [
        (name, start, end) for name, start, end in _chromosome_blocks(chromosome)
        if feasibility[name]["feasible"]
    ]
    rng = np.random.default_rng(config.random_seed + 977)
    tracks: list[tuple[np.ndarray, np.ndarray, list[tuple[str, int, int, int]]]] = []
    for _ in range(config.null_replicates):
        track_pieces: list[np.ndarray] = []
        eligible_pieces: list[np.ndarray] = []
        segments: list[tuple[str, int, int, int]] = []
        for name, start, end in blocks:
            l_max = feasibility[name]["l_max"]
            lag = int(rng.integers(config.null_min_lag, l_max + 1))
            piece = _lag_null_track(residuals[start:end], lag, config)
            if not piece.size:
                continue
            observed_eligible_c = (
                observed_eligible[start : max(end - 1, start)]
                if observed_eligible is not None else None
            )
            piece_eligible = _null_eligibility(
                None, lag, config, piece.size,
                observed_eligible=observed_eligible_c,
            )
            if not piece_eligible.any():
                continue
            # Broken at the boundary, not left as `eligible_edges` would
            # naturally leave it: a run must never cross from one
            # chromosome's segment into the next's, even though both are
            # individually eligible right up to their own last position --
            # exactly `member_step_lag_null_ensemble`'s existing device for
            # the same reason (concatenated offset sub-tracks).
            piece_eligible = piece_eligible.copy()
            piece_eligible[-1] = False
            track_pieces.append(piece)
            eligible_pieces.append(piece_eligible)
            segments.append((name, lag, int(start), int(piece.size)))
        if not track_pieces:
            continue
        tracks.append((
            np.concatenate(track_pieces), np.concatenate(eligible_pieces), segments,
        ))
    return tracks


def member_step_lag_null_tracks(
    residuals: np.ndarray,
    lag: int,
    config: Eiger2Config,
    locations: pd.DataFrame | None = None,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """The member-step lag null: one sub-track per starting offset at a fixed lag.

    ``null_tracks()``'s construction steps its position index ``p`` by one
    CpG: position ``p``'s members are ``{p, p+lag, ..., p+(w-1)*lag}`` and
    position ``p+1``'s are ``{p+1, p+1+lag, ...}`` -- DISJOINT sets, unlike
    consecutive OBSERVED windows, which share ``w-1`` of ``w`` members. That
    mismatch is why that null's autocorrelation does not match the observed
    track's.

    This construction re-parameterises: fix a starting offset ``i`` and step a
    SECOND index ``j``, so the null "position" is ``p = i + j*lag`` and
    consecutive positions share ``w-1`` of ``w`` members by construction --
    reproducing the autocorrelation directly, not tuning toward it.

    Implementation: ``_lag_null_track(residuals, lag, config)`` already
    computes the per-``p`` value for every ``p``; re-slicing its output at
    stride ``lag`` from each offset ``i`` in ``[0, lag)`` recovers exactly the
    ``j``-indexed sub-track for that offset, with no new pair-assembly code.
    Eligibility likewise re-slices ``_null_eligibility``'s existing output.

    Returns one ``(sub_track, sub_eligible, offset)`` triple per offset ``i``
    with at least one eligible position. ``offset`` takes the third-element
    slot ``null_tracks()`` uses for ``lag``, so both are drop-in compatible
    with ``acf_gate``'s ``(track, eligible, *_rest)`` unpacking.
    """

    full = _lag_null_track(residuals, lag, config)
    if not full.size:
        return []
    chromosome = locations["chr"].to_numpy() if locations is not None else None
    observed_eligible = (
        eligible_edges(locations, config) if locations is not None else None
    )
    full_eligible = _null_eligibility(
        chromosome, lag, config, full.size, observed_eligible=observed_eligible,
    )
    sub_tracks: list[tuple[np.ndarray, np.ndarray, int]] = []
    for i in range(min(lag, full.size)):
        sub_track = full[i::lag]
        sub_eligible = full_eligible[i::lag]
        if sub_track.size and sub_eligible.any():
            sub_tracks.append((sub_track, sub_eligible, i))
    return sub_tracks


def member_step_min_n_cpg(config: Eiger2Config) -> int:
    """Minimum TOTAL ``n_cpg`` for ``region_null_mode="member_step"`` to be
    worth attempting at all: the smallest input able to carry the configured
    minimum detectable region at ``null_min_lag``. Reported by ``cli.py``
    before a run, so a caller who is about to request ``"member_step"`` sees
    the constraint before the pipeline runs rather than after it raises.

    Deliberately genome-wide, not per-chromosome (spec §2b.4's own
    construction IS per-chromosome): this is a cheap, approximate,
    conservative pre-check on whether member_step is worth attempting at
    all, run before any per-chromosome breakdown is available. The
    authoritative, per-chromosome check is
    ``member_step_lag_null_ensemble``'s own -- it raises when NO chromosome
    individually clears the floor, which this function cannot detect (a
    genome comfortably above this total could still have every individual
    chromosome too small, if fragmented across many of them).
    """

    span = config.min_cpgs + max(config.window, 1) - 1
    return config.null_min_lag * span


def member_step_lag_null_ensemble(
    residuals: np.ndarray,
    config: Eiger2Config,
    locations: pd.DataFrame | None = None,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray, list[tuple[str, int, int]]]],
    list[list[tuple[str, int, int]]],
]:
    """The region-level null pool: ``config.n_lag_draws`` independent
    replicates, directly interchangeable with ``null_tracks()``'s output
    everywhere it is consumed (``acf_gate``, ``null_exchangeability``,
    ``_overdispersion_diagnostic``), each replicate standing in for one
    "replicate" exactly as one ``"lag"``-construction replicate does.

    Chromosome-aware (spec §2b.4): the same audit as :func:`null_tracks`
    (§2b.2) applies here -- ``L_max`` was derived from the genome-wide
    ``n_cpg``, the same class of error (on hg38, ``L_max`` from the whole
    genome survives on chr1 so this is not catastrophic there, but on a
    300,000-row chromosome only lags below 100,000 -- about 35% of the
    genome-wide draw range -- actually fit inside it). Each replicate now
    draws an INDEPENDENT lag per feasible chromosome. The preferred ceiling is
    ``n_c // (min_subtrack_length + window - 1)``. Chromosomes below that
    preferred floor use a measured fallback ceiling of
    ``n_c // (min_cpgs + window - 1)`` rather than disappearing entirely;
    builds that
    chromosome's own member-step sub-tracks (:func:`member_step_lag_null_tracks`,
    one sub-track per starting offset ``i in [0, lag_c)``) from its own rows,
    and concatenates every feasible chromosome's sub-tracks into one
    genome-wide realisation, with the eligibility mask broken at every
    sub-track boundary AND every chromosome boundary -- ``segment()`` can
    then never merge two unrelated offsets, or two different chromosomes,
    into one spurious region.

    Only a chromosome unable to support even ``min_cpgs`` positions at
    ``null_min_lag`` contributes nothing. Its eligible observed-edge exposure
    is measured separately by :func:`member_step_chromosome_feasibility` and
    gated by :func:`call_dmrs`.

    Raises when NO chromosome clears the floor, rather than silently
    degrading to short sub-tracks.

    Returns ``(replicates, lags)``. ``replicates`` is ``(track, eligible,
    segments)`` triples, ``segments`` = ``[(chromosome, lag, n_subtracks),
    ...]`` in concatenation order. ``lags`` has one entry per replicate,
    each itself that replicate's own ``segments`` list (mirroring
    ``null_tracks``'s ``region_lags`` convention for the ``"lag"`` case) --
    ``len(lags) == config.n_lag_draws`` always, recorded as
    ``manifest["gates"]["n_lag_draws"]``.
    """

    if locations is None:
        blocks = [("", 0, residuals.shape[0])]
    else:
        blocks = _chromosome_blocks(locations["chr"].to_numpy())

    if locations is None:
        feasibility = {
            "": {
                **_member_step_lag_limits(residuals.shape[0], config),
                "n_cpg": int(residuals.shape[0]),
            }
        }
        feasible_blocks = [
            (name, start, end, feasibility[name]["l_max"])
            for name, start, end in blocks
            if feasibility[name]["feasible"]
        ]
    else:
        feasibility = member_step_chromosome_feasibility(locations, config)
        feasible_blocks = [
            (name, start, end, feasibility[name]["l_max"])
            for name, start, end in blocks
            if feasibility[name]["feasible"]
        ]
    if not feasible_blocks:
        minimum_n_cpg = member_step_min_n_cpg(config)
        largest_l_max = max(
            (_member_step_lag_limits(e - s, config)["l_max"] for _, s, e in blocks),
            default=0,
        )
        largest_n_cpg = max((e - s for _, s, e in blocks), default=0)
        raise ValueError(
            f"region_null_mode='member_step' needs at least one CHROMOSOME "
            f"able to produce a minimum detectable sub-track of "
            f"min_cpgs={config.min_cpgs} at "
            f"null_min_lag={config.null_min_lag}, i.e. a chromosome with "
            f"n_cpg >= null_min_lag * (min_cpgs + window - 1) = "
            f"{minimum_n_cpg}; the largest chromosome here has "
            f"n_cpg={largest_n_cpg} (fallback L_max={largest_l_max}). Use "
            f"region_null_mode='lag' on input this small; do not lower "
            f"null_min_lag merely to force feasibility because it is the "
            f"locality-breaking floor."
        )

    rng = np.random.default_rng(config.random_seed + 61_093)
    replicates: list[tuple[np.ndarray, np.ndarray, list[tuple[str, int, int]]]] = []
    lags: list[list[tuple[str, int, int]]] = []
    for _ in range(config.n_lag_draws):
        track_pieces: list[np.ndarray] = []
        eligible_pieces: list[np.ndarray] = []
        segments: list[tuple[str, int, int]] = []
        for name, start, end, l_max_c in feasible_blocks:
            lag_c = int(rng.integers(config.null_min_lag, l_max_c + 1))
            sub_tracks = member_step_lag_null_tracks(
                residuals[start:end], lag_c, config,
                (
                    locations.iloc[start:end].reset_index(drop=True)
                    if locations is not None else None
                ),
            )
            if not sub_tracks:
                continue
            chrom_pieces, chrom_eligible = [], []
            for sub_track, sub_eligible, _offset in sub_tracks:
                chrom_pieces.append(sub_track)
                # Never merge into the next offset's piece.
                boundary = sub_eligible.copy()
                if boundary.size:
                    boundary[-1] = False
                chrom_eligible.append(boundary)
            if not chrom_pieces:
                continue
            # The last sub-track's own boundary break (above) already makes
            # this chromosome's concatenated mask end ineligible, which is
            # exactly the break needed before the NEXT chromosome's segment
            # -- one device, two purposes, not reapplied redundantly.
            track_pieces.append(np.concatenate(chrom_pieces))
            eligible_pieces.append(np.concatenate(chrom_eligible))
            segments.append((name, lag_c, len(sub_tracks)))
        if not track_pieces:
            continue
        replicates.append((
            np.concatenate(track_pieces), np.concatenate(eligible_pieces), segments,
        ))
        lags.append(segments)
    return replicates, lags


def permutation_null_tracks(
    statistic: np.ndarray,
    eligible: np.ndarray,
    chromosome: np.ndarray | None = None,
    *,
    replicates: int = 100,
    seed: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Retired: edge-wise permutation (block length 1) of the observed statistic
    within each chromosome, measured at zero usable null regions across 400
    replicates. Always raises; kept only so old call sites get a clear error
    naming the replacement rather than an import failure."""

    raise ValueError(
        "permutation_null_tracks is edge-wise permutation (block length 1), "
        "measured at zero null regions across 400 replicates, and is no "
        "longer usable. For the region-level null, use "
        "region_null_mode='member_step' (or leave it at its default 'auto', "
        "which resolves to 'member_step' automatically on input large enough "
        "to support it); for the edge-level null, null_mode='lag' is the "
        "only supported value."
    )


def _two_means_partition(region_means: np.ndarray) -> dict:
    """Return a deterministic one-dimensional two-means split."""

    means = np.asarray(region_means, dtype=float)
    finite = np.isfinite(means)
    index = np.flatnonzero(finite)
    if index.size < 4:
        return {
            "minority_size": 0,
            "minority_fraction": float("nan"),
            "minority_delta": 0.0,
            "minority_samples": np.empty(0, dtype=int),
            "boundary_gap": float("nan"),
            "boundary_gap_fraction_delta": float("nan"),
        }
    order = index[np.argsort(means[index], kind="mergesort")]
    ordered = means[order]
    n = ordered.size
    cumulative = np.cumsum(ordered)
    cumulative_square = np.cumsum(ordered * ordered)
    left = np.arange(1, n)
    mean_left = cumulative[:-1] / left
    mean_right = (cumulative[-1] - cumulative[:-1]) / (n - left)
    within = (
        (cumulative_square[:-1] - left * mean_left ** 2)
        + ((cumulative_square[-1] - cumulative_square[:-1])
           - (n - left) * mean_right ** 2)
    )
    cut = int(np.argmin(within))
    lower = order[: cut + 1]
    upper = order[cut + 1 :]
    minority = lower if lower.size <= upper.size else upper
    delta = float(abs(mean_left[cut] - mean_right[cut]))
    boundary_gap = float(ordered[cut + 1] - ordered[cut])
    return {
        "minority_size": int(minority.size),
        "minority_fraction": float(minority.size / n),
        "minority_delta": delta,
        "minority_samples": minority,
        "boundary_gap": boundary_gap,
        "boundary_gap_fraction_delta": (
            float(boundary_gap / delta) if delta > 0 else float("nan")
        ),
    }


def region_minority_size(
    values: np.ndarray, region: Sequence[int], matrix: np.ndarray
) -> tuple[int, float]:
    """Size of the deviating sample group for a region, and its effect.

    Uses a one-dimensional two-means partition of per-sample region means and
    returns the smaller group and the difference between cluster means.
    """

    del values  # historical parameter retained for API compatibility
    first, last = int(region[0]), int(region[1])
    block = matrix[first : last + 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(block, axis=0)
    # Unlike the largest-gap diagnostic, two-means estimates a group partition.
    split = _two_means_partition(means)
    return int(split["minority_size"]), float(split["minority_delta"])


def null_exchangeability(
    tracks: list[tuple[np.ndarray, np.ndarray]],
    statistic: np.ndarray,
    eligible: np.ndarray,
    config: Eiger2Config,
    *,
    calibration_fraction: float = 0.5,
) -> dict:
    """Split-half check that the null distribution matches the observed one.

    Splits the null replicates into a calibration half (sets the threshold)
    and a held-out half (measures exceedance against it). ``p_null`` is the
    held-out exceedance and is a control: by construction it must equal
    ``1 - config.null_quantile``, confirming the null is internally
    self-consistent. ``p_obs`` is the observed track's exceedance against the
    SAME threshold. The ratio ``p_obs / p_null`` is the actual gate: 1.0 when
    the null and data share the same upper-tail shape, rising when the
    observed track is more heavy-tailed than the null represents -- see
    docs/FDR.md, "Null exchangeability".

    The split is deterministic given ``config.random_seed`` (a fixed
    permutation of replicate indices, not re-derived from tracks' own order),
    so results are reproducible independent of how many replicates happen to
    survive filtering upstream.
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


def track_autocorrelation(
    values: np.ndarray, eligible: np.ndarray, lags: Sequence[int]
) -> dict[int, float]:
    """Lag-``k`` autocorrelation of ``values`` at every ``k`` in ``lags``.

    Pairs ``(values[i], values[i+k])`` are used where both positions are
    eligible and finite; the correlation is the Pearson coefficient over that
    pooled pair set. ``NaN`` for a lag with fewer than 2 usable pairs or a
    constant series (undefined correlation), not raised.
    """

    values = np.asarray(values, dtype=float)
    usable = np.asarray(eligible, dtype=bool) & np.isfinite(values)
    out: dict[int, float] = {}
    for k in lags:
        if k <= 0 or k >= values.size:
            out[k] = float("nan")
            continue
        paired = usable[:-k] & usable[k:]
        x = values[:-k][paired]
        y = values[k:][paired]
        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            out[k] = float("nan")
        else:
            out[k] = float(np.corrcoef(x, y)[0, 1])
    return out


def acf_gate(
    observed_statistic: np.ndarray,
    eligible: np.ndarray,
    null_tracks_: Sequence[tuple],
    window: int,
    *,
    tolerance: float = 0.05,
) -> dict:
    """Gate 1: null autocorrelation must match the observed track's within
    ``tolerance`` at lags 1..``window``, on DMR-free data. A sliding-window
    statistic manufactures autocorrelation by construction (windows at edges
    ``e`` and ``e+k`` share CpG pairs whenever ``|k| < w``), so validating a
    null's marginal and variance is not enough -- its dependence structure has
    to match too, or a run of elevated edges that is pure geometry looks like
    signal.

    ``null_tracks_`` pools every replicate's usable pairs at each lag BEFORE computing
    the correlation (not an average of per-replicate correlations), matching how the
    null is pooled everywhere else in this codebase. Accepts either
    ``null_tracks()``'s ``(track, eligible, segments)`` triples or
    ``permutation_null_tracks()``'s ``(track, eligible)`` pairs -- only the first two
    elements are used, mirroring the ``*_rest`` unpacking ``call_dmrs`` already uses
    for the same reason. Passes only when every lag has a finite, in-tolerance
    difference; a lag that could not be measured (too few usable pairs) fails the gate
    rather than being silently skipped, since an unmeasurable lag is not evidence of a
    match.
    """

    lags = list(range(1, max(int(window), 1) + 1))
    observed_acf = track_autocorrelation(observed_statistic, eligible, lags)

    null_acf: dict[int, float] = {}
    for k in lags:
        xs, ys = [], []
        for track, track_eligible, *_rest in null_tracks_:
            track = np.asarray(track, dtype=float)
            usable = np.asarray(track_eligible, dtype=bool) & np.isfinite(track)
            if k <= 0 or k >= track.size:
                continue
            paired = usable[:-k] & usable[k:]
            xs.append(track[:-k][paired])
            ys.append(track[k:][paired])
        x = np.concatenate(xs) if xs else np.empty(0)
        y = np.concatenate(ys) if ys else np.empty(0)
        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            null_acf[k] = float("nan")
        else:
            null_acf[k] = float(np.corrcoef(x, y)[0, 1])

    diff: dict[int, float] = {}
    for k in lags:
        o, n = observed_acf[k], null_acf[k]
        diff[k] = (n - o) if (np.isfinite(o) and np.isfinite(n)) else float("nan")

    passed = bool(lags) and all(
        np.isfinite(diff[k]) and abs(diff[k]) <= tolerance for k in lags
    )
    return {
        "lags": lags,
        "observed_acf": observed_acf,
        "null_acf": null_acf,
        "diff": diff,
        "tolerance": tolerance,
        "passed": passed,
    }


def region_exchangeability_gate(
    *,
    observed_regions: int,
    observed_eligible_edges: int,
    null_regions_total: int,
    null_eligible_edges: int,
    min_null_regions_for_fdr: int,
) -> dict:
    """Gate 2: region-level exchangeability, on DMR-free data.

    ``observed_rate``/``null_rate`` are candidate REGIONS (pre-FDR, i.e. segmenter
    output) per 10,000 eligible edges. ``region_exchangeability_ratio`` is their
    ratio; ``inf`` is reported EXPLICITLY (not coerced to ``NaN`` or dropped) when
    ``null_rate`` is zero and ``observed_rate`` is positive -- that must be visible,
    not hidden. ``NaN`` only when BOTH rates are zero (nothing happened on either
    side; the ratio is genuinely undefined, not infinite). ``null_count_adequate``
    records whether the pooled null-region count clears
    ``min_null_regions_for_fdr``, i.e. whether the ratio is "estimated from
    something" -- reported alongside the ratio, not a refusal gate like the FDR
    estimator's own guard; the ratio is still returned either way, for the same
    reason ``inf`` is reported rather than hidden.
    """

    observed_rate = (
        10_000 * observed_regions / observed_eligible_edges
        if observed_eligible_edges > 0 else float("nan")
    )
    null_rate = (
        10_000 * null_regions_total / null_eligible_edges
        if null_eligible_edges > 0 else float("nan")
    )
    if not (np.isfinite(observed_rate) and np.isfinite(null_rate)):
        ratio = float("nan")
    elif null_rate > 0:
        ratio = observed_rate / null_rate
    elif observed_rate > 0:
        ratio = float("inf")
    else:
        ratio = float("nan")

    return {
        "observed_regions": int(observed_regions),
        "observed_eligible_edges": int(observed_eligible_edges),
        "observed_rate": observed_rate,
        "null_regions_total": int(null_regions_total),
        "null_eligible_edges": int(null_eligible_edges),
        "null_rate": null_rate,
        "region_exchangeability_ratio": ratio,
        "null_count_adequate": null_regions_total >= min_null_regions_for_fdr,
        "min_null_regions_for_fdr": int(min_null_regions_for_fdr),
    }


def _null_quantile_by_chromosome(
    tracks: list[tuple[np.ndarray, np.ndarray, list[tuple[str, int, int, int]]]],
    quantile: float,
) -> dict[str, float]:
    """The SAME ``null_quantile`` computed separately per chromosome, from
    the pooled null values :func:`null_tracks` already produced for it
    (``segments`` metadata) -- no recomputation of anything upstream, just
    grouping and re-quantiling data already in hand (spec §2b.3's
    sensitivity report). Shows whether pooling every feasible chromosome's
    null together masks one whose own null behaves differently, at the cost
    of one extra quantile per chromosome, not a second run.
    """

    pools: dict[str, list[np.ndarray]] = {}
    for track, eligible, segments in tracks:
        position = 0
        for name, _lag, _start, length in segments:
            piece = track[position : position + length]
            usable = eligible[position : position + length] & np.isfinite(piece)
            if usable.any():
                pools.setdefault(name, []).append(piece[usable])
            position += length
    return {
        name: float(np.quantile(np.concatenate(values), quantile))
        for name, values in pools.items()
    }


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
        np.concatenate([t[e] for t, e, _ in tracks]) if tracks else np.empty(0)
    )
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        raise ValueError(
            "the locality-breaking null produced no usable pairs; the matrix is "
            "probably too small or too sparse"
        )
    threshold = float(np.quantile(pooled, config.null_quantile))
    threshold_by_chromosome = _null_quantile_by_chromosome(
        tracks, config.null_quantile,
    )
    # A chromosome-aware null discards positions whose window members straddle a
    # chromosome boundary. With many short chromosomes and a large lag this can
    # remove most of the null, leaving the threshold poorly determined, so the
    # retained fraction is reported and flagged rather than left implicit.
    offered = sum(int(t.size) for t, _, _lag in tracks) if tracks else 0
    coverage = float(pooled.size / offered) if offered else 0.0
    diagnostics = {
        "null_replicates": len(tracks),
        "null_values": int(pooled.size),
        "null_positions_offered": offered,
        "null_coverage": coverage,
        "null_quantile": config.null_quantile,
        "null_median": float(np.median(pooled)),
        "threshold": threshold,
        # spec §2b.3's sensitivity report: the same quantile recomputed per
        # feasible chromosome (never a second run -- see
        # _null_quantile_by_chromosome), alongside the pooled "threshold"
        # above it is a comparison for.
        "threshold_by_chromosome": threshold_by_chromosome,
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


EDGE_GAP_BIN_UPPER_BP = np.array([4, 19, 49, 99, 300], dtype=np.int64)
"""Upper bp gap of each edge-threshold gap bin; the last bin is open-ended.

A CpG pair 3 bp apart and one 4 kb apart do not have the same null
coordination, so one genome-wide quantile is not one threshold: it is a
mixture whose composition varies along the genome. Conditioning on the gap
(and on how many samples actually observe the whole window) makes the quantile
mean the same thing everywhere.
"""

MIN_EDGES_PER_THRESHOLD_BIN = 1_000
"""Below this many usable edges a threshold bin borrows: joint -> gap -> global.

A quantile of a few hundred values is mostly noise; 1,000 edges put at least
ten above a 0.99 quantile, the same resolution floor ``min_null_values`` uses.
"""


def observed_edge_threshold(
    statistic: np.ndarray,
    eligible: np.ndarray,
    locations: pd.DataFrame,
    support: np.ndarray,
    config: Eiger2Config,
) -> tuple[np.ndarray, dict]:
    """Per-edge threshold: the OBSERVED track's own ``null_quantile``, within
    joint bp-gap x complete-window-support bins.

    This is the default discovery threshold. It asks "which edges are extreme
    for this genome, given this gap and this support?" and answers it from the
    genome in front of it, rather than from the locality-breaking null, whose
    hypothesis is false everywhere on a WGBS background (docs/FDR.md §18).

    A joint bin with fewer than ``MIN_EDGES_PER_THRESHOLD_BIN`` usable edges
    takes its quantile from its gap bin, then from every usable edge. The
    returned array always has ``statistic``'s shape; ineligible edges carry the
    global quantile and are never segmented anyway.
    """

    statistic = np.asarray(statistic, dtype=float)
    usable = np.asarray(eligible, dtype=bool) & np.isfinite(statistic)
    support = np.asarray(support, dtype=np.int64)
    usable &= support >= 0
    if not usable.any():
        raise ValueError(
            "no usable observed edges: every edge is ineligible, non-finite or "
            "observed in no complete window"
        )

    starts = locations["start"].to_numpy(dtype=np.int64)
    ends = locations["end"].to_numpy(dtype=np.int64)
    gaps = starts[1:] - ends[:-1]
    gap_bin = np.searchsorted(EDGE_GAP_BIN_UPPER_BP, gaps, side="left")

    support_edges = np.unique(np.quantile(support[usable], (0.25, 0.5, 0.75)))
    support_bin = _assign_support_bin(support, support_edges)
    joint = gap_bin * (support_edges.size + 1) + support_bin

    global_threshold = float(np.quantile(statistic[usable], config.null_quantile))
    thresholds = np.full(statistic.size, global_threshold, dtype=float)
    sources = {"joint": 0, "gap": 0, "global": 0}
    per_bin: list[dict] = []
    for label in np.unique(joint[usable]):
        rows = usable & (joint == label)
        source, source_name = rows, "joint"
        if int(rows.sum()) < MIN_EDGES_PER_THRESHOLD_BIN:
            source = usable & (gap_bin == gap_bin[np.flatnonzero(rows)[0]])
            source_name = "gap"
        if int(source.sum()) < MIN_EDGES_PER_THRESHOLD_BIN:
            source, source_name = usable, "global"
        value = float(np.quantile(statistic[source], config.null_quantile))
        thresholds[rows] = value
        sources[source_name] += int(rows.sum())
        per_bin.append({
            "gap_bin": int(label // (support_edges.size + 1)),
            "support_bin": int(label % (support_edges.size + 1)),
            "edges": int(rows.sum()),
            "source": source_name,
            "threshold": value,
        })

    diagnostics = {
        "mode": "observed_gap_support_quantile",
        "null_quantile": config.null_quantile,
        "usable_edges": int(usable.sum()),
        "gap_bins_upper_bp": EDGE_GAP_BIN_UPPER_BP.tolist(),
        "support_bin_edges": support_edges.tolist(),
        "min_edges_per_bin": MIN_EDGES_PER_THRESHOLD_BIN,
        "threshold": global_threshold,
        "threshold_edges_by_source": sources,
        "threshold_by_bin": per_bin,
        "note": (
            "the edge threshold is the observed track's own quantile within "
            "joint gap and support bins; the locality-breaking null is not "
            "used on this path (docs/FDR.md section 18)"
        ),
    }
    return thresholds, diagnostics


# --------------------------------------------------------------------------
# step 5 — segmentation and region-level FDR
# --------------------------------------------------------------------------


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
    shape (support-conditioned mode). The scalar path is unchanged and stays
    bit-identical.

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

    regions: list[tuple[int, int]] = []
    index = 0
    n_edges = passing.size
    while index < n_edges:
        if not passing[index]:
            index += 1
            continue
        start = index
        end = index
        cursor = index + 1
        gap = 0
        while cursor < n_edges:
            if passing[cursor]:
                end = cursor
                gap = 0
            elif eligible[cursor] and gap < max_gap_edges:
                gap += 1
            else:
                break
            cursor += 1
        if (end - start + 2) >= min_cpgs:
            regions.append((start, end + 1))
        index = max(end + 1, start + 1)
    if not regions:
        return np.empty((0, 2), dtype=int)
    return np.asarray(regions, dtype=int)


def region_area(
    statistic: np.ndarray, regions: np.ndarray, threshold: float | np.ndarray
) -> np.ndarray:
    """Summed excess of the statistic over the threshold, per region.

    ``threshold`` may be a scalar or a per-edge array matching ``statistic``'s
    shape (support-conditioned mode); the scalar path is unchanged.

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
        block_threshold = (
            threshold_arr[first:last] if is_array_threshold else threshold_arr
        )
        finite_mask = np.isfinite(block)
        block = block[finite_mask]
        block_threshold = (
            block_threshold[finite_mask] if is_array_threshold else block_threshold
        )
        areas[position] = float(np.sum(block - block_threshold)) if block.size else 0.0
    return areas


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


def _fdr_from_areas_result(
    observed_areas: np.ndarray,
    pooled_null: np.ndarray,
    scale: float,
    *,
    confidence_level: float,
    m_bonferroni: int,
    dispersion: float = 1.0,
) -> dict:
    """Tail-area FDR, scaled-count bounds and resolution for one stratum.

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

    # Replicate-cluster quasi-Poisson interval. `dispersion` is estimated from
    # one count per independently drawn genome-wide null realisation by
    # `region_fdr_result`; it is never estimated by pretending the many
    # regions within a replicate are independent.  Scaling the effective
    # count K/phi gives the usual exact-Poisson/Garwood interval at phi=1 and
    # widens it when replicate counts are overdispersed.  The common
    # Bonferroni alpha makes the complete reported cutoff curve simultaneous
    # within a stratum.  At K=0 the upper endpoint remains finite and positive.
    phi = max(float(dispersion), 1.0)
    alpha  = 1.0 - confidence_level
    astar  = alpha / max(int(m_bonferroni), 1)
    effective_k = K / phi
    L = np.where(
        K > 0,
        0.5 * phi * chi2.ppf(astar / 2.0, 2.0 * effective_k),
        0.0,
    )
    U = 0.5 * phi * chi2.ppf(
        1.0 - astar / 2.0, 2.0 * (effective_k + 1.0)
    )
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
        "tail_fdr":          _expand(raw),  # region's OWN cutoff, not q-achieving
        "qvalue_lower":      _expand(q_lo_group),
        "qvalue_upper":      _expand(q_hi_group),
        "qvalue_resolution": _expand(delta[qarg_group]),
        "null_exceedances":  _expand(K[qarg_group]),
        "qvalue_cutoff_rank":_expand(R[qarg_group].astype(np.int64)),
        "cutoffs":           cutoff,
        "_n_cutoffs":        G,
    }


def _fdr_from_areas(
    observed_areas: np.ndarray, pooled_null: np.ndarray, scale: float
) -> np.ndarray:
    return _fdr_from_areas_result(
        observed_areas, pooled_null, scale,
        confidence_level=0.95, m_bonferroni=int(np.unique(observed_areas).size) or 1,
    )["qvalue"]


def length_stratum(n_cpg: int) -> str:
    """Bin a region by CpG count for length-stratified FDR."""

    if n_cpg <= 5:
        return "4-5"
    if n_cpg <= 9:
        return "6-9"
    if n_cpg <= 19:
        return "10-19"
    if n_cpg <= 49:
        return "20-49"
    return "50+"


# Minimum pooled null-region count for a joint (length x support-bin) stratum
# to be estimated on its own. Below this the stratum falls back to
# length-only pooling. Not user-configurable.
_MIN_NULL_REGIONS_PER_JOINT_STRATUM = 50

# Lower bound in EDGE-run-length units for each CpG-count stratum's shortest
# member -- `min_edges(s)` in spec §2c.1. A region spanning E passing edges
# contains E + 1 CpGs, hence the one-less conversion below. Kept next to
# `length_stratum` so the two cannot silently drift apart.
_LENGTH_STRATUM_MIN_EDGES = {
    "4-5": 3, "6-9": 5, "10-19": 9, "20-49": 19, "50+": 49,
}


def _run_lengths_from_mask(usable: np.ndarray) -> np.ndarray:
    """Maximal contiguous run lengths in a boolean eligibility mask.

    Mask-native counterpart of ``layouts.py``'s ``_run_boundaries``, which
    takes a locations frame and recomputes eligibility internally; this
    operates directly on an already-materialised mask (e.g. ``eligible &
    np.isfinite(statistic)``), needed for spec §2c's per-stratum null
    opportunity (see ``_stratum_opportunity`` below).
    """

    usable = np.asarray(usable, dtype=bool)
    if usable.size == 0:
        return np.empty(0, dtype=np.int64)
    breaks = np.flatnonzero(~usable)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [usable.size]))
    lengths = ends - starts
    return lengths[lengths > 0]


def _stratum_opportunity(
    run_lengths: np.ndarray, min_cpgs: int = 4
) -> dict[str, float]:
    """Count possible region starts per length stratum.

    The bottom stratum follows ``min_cpgs`` when it is below the default of 4;
    all higher stratum boundaries are fixed.
    """

    run_lengths = np.asarray(run_lengths, dtype=np.int64)
    min_edges = dict(_LENGTH_STRATUM_MIN_EDGES)
    min_edges["4-5"] = min(min_edges["4-5"], max(int(min_cpgs) - 1, 1))
    return {
        stratum: float(np.sum(np.maximum(0, run_lengths - edges + 1)))
        for stratum, edges in min_edges.items()
    }


def _overdispersion_diagnostic(
    per_replicate_null_areas: list[np.ndarray], cutoffs: np.ndarray
) -> tuple[str, float | None]:
    """Legacy Poisson-vs-observed dispersion summary.

    Retained for diagnostic/API compatibility. The current interval uses
    :func:`_replicate_cluster_dispersion` and incorporates its exposure-adjusted
    dispersion; this helper is no longer the basis of the reported bound.
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


# Five independent genome-wide realisations is the fixed minimum for reporting
# replicate-dispersion uncertainty. Two clusters are algebraically estimable,
# but one residual degree of freedom is not defensible for an upper inference
# claim. This matches the established null-dispersion diagnostic convention and
# is a method requirement, not a user-adjustable tuning knob.
_MIN_REGION_NULL_CLUSTERS = 5


def _replicate_cluster_dispersion(
    per_replicate_null_areas: Sequence[np.ndarray],
    cutoff: float,
    null_edge_counts: np.ndarray,
) -> tuple[str, float | None, dict]:
    """Estimate quasi-Poisson dispersion from independent null replicates.

    Each genome-wide null realisation contributes exactly one exceedance
    count and one exposure. The Pearson estimate therefore measures
    replicate-to-replicate variation without treating regions inside a
    replicate as independent. ``cutoff`` is the weakest reported cutoff in
    the stratum, where the count is most stable; a single dispersion is then
    used for the whole nested cutoff curve.
    """

    b = len(per_replicate_null_areas)
    exposures = np.asarray(null_edge_counts, dtype=float)
    if exposures.shape != (b,) or np.any(~np.isfinite(exposures)):
        raise ValueError("null_edge_counts must contain one finite value per replicate")
    if np.any(exposures < 0):
        raise ValueError("null_edge_counts must be non-negative")
    counts = np.array([
        np.sum(np.asarray(areas, dtype=float) >= cutoff)
        for areas in per_replicate_null_areas
    ], dtype=float)
    positive = exposures > 0
    if np.any((~positive) & (counts > 0)):
        raise ValueError("a zero-exposure null replicate cannot contain regions")
    exposures = exposures[positive]
    counts = counts[positive]
    b = int(positive.sum())
    if b < _MIN_REGION_NULL_CLUSTERS:
        return "insufficient_clusters", None, {
            "clusters": b,
            "minimum_clusters": _MIN_REGION_NULL_CLUSTERS,
        }

    total_count = float(counts.sum())
    if total_count <= 0:
        return "unavailable_zero_count", None, {
            "clusters": b,
            "cutoff": float(cutoff),
            "total_count": 0,
        }

    rate = total_count / float(exposures.sum())
    expected = exposures * rate
    pearson = float(np.sum((counts - expected) ** 2 / expected) / (b - 1))
    return "computed", max(1.0, pearson), {
        "clusters": b,
        "cutoff": float(cutoff),
        "total_count": int(total_count),
        "pearson_dispersion_raw": pearson,
    }


def region_fdr_result(
    observed_areas: np.ndarray,
    null_area_sets: Sequence[np.ndarray],
    *,
    observed_edges: int,
    null_edges: int,
    observed_lengths: np.ndarray | None = None,
    null_length_sets: Sequence[np.ndarray] | None = None,
    observed_support_bins: np.ndarray | None = None,
    null_support_bin_sets: Sequence[np.ndarray] | None = None,
    confidence_level: float = 0.95,
    stratum_opportunity: Mapping[str, tuple[float, float]] | None = None,
    m_opportunity: float = 10.0,
    null_edge_counts: Sequence[int] | None = None,
    null_stratum_exposures: Mapping[str, Sequence[float]] | None = None,
) -> RegionFdrResult:
    """Region-level empirical FDR: pooled point estimate, resolution, and
    simultaneous replicate-cluster quasi-Poisson bounds, computed within
    length strata with a common Bonferroni ``m``. See
    ``docs/FDR.md`` for the full derivation and the three distinct quantities
    this keeps apart (point estimate, empirical resolution, Monte Carlo
    interval).

    For a cut-off ``a``, the expected number of null regions at least as
    extreme is the mean count across replicates, rescaled to the size of the
    real genome. The FDR is that expectation divided by the observed count.

    When ``observed_lengths``/``null_length_sets`` are supplied, the
    calculation is done **within length strata**: region area grows with
    length, so a pooled FDR silently makes short regions look worse than long
    ones. A 4-CpG DMR should compete against 4-CpG null regions, not 50-CpG
    ones. When ``stratum_opportunity`` is available, the point estimate and
    both bounds use ``scale_s = observed_opportunity_s / null_opportunity_s``
    for that length. This is required when short-subtrack fallback gives
    different length support. Without measured length opportunity, direct
    callers retain an explicitly diagnosed global-edge fallback.

    When ``observed_support_bins``/``null_support_bin_sets`` are ALSO
    supplied, the null pool is stratified jointly, by
    ``"<length_stratum>|sbin<b>"`` -- equalising the per-edge threshold across
    support bins does not equalise the rate regions FORM per bin. A joint
    stratum whose pooled null count is below
    ``_MIN_NULL_REGIONS_PER_JOINT_STRATUM`` falls back to the length-only
    pool, listed in ``diagnostics["support_stratification_fallback_strata"]``.

    ``stratum_opportunity``/``m_opportunity`` guard null adequacy PER LENGTH
    STRATUM (spec §2c), replacing an earlier AGGREGATE pooled-count floor that
    checked the wrong quantity: a run could hold thousands of null regions in
    short strata while a long stratum's null never had the CHANCE to grow a
    region that long, and every long observed region then received ``q=0``
    from a null that could never have contradicted it.

    ``stratum_opportunity`` maps each length-stratum name (``length_stratum``'s
    return values) to that stratum's ``(O_s, N_s)`` -- observed and pooled-null
    opportunity, both the sum over maximal eligible runs of
    ``max(0, run_len - min_edges(s) + 1)``. A stratum with ``R_s`` observed
    regions is adequate iff ``N_s >= m_opportunity * O_s / R_s``: the null
    would have been expected to produce at least ``m_opportunity`` regions had
    the observed rate applied to it. Inadequate strata are refused -- every
    inferential field ``NaN``, every count field ``-1``, for that stratum's
    rows only -- and named, with ``O_s``/``N_s``/``R_s`` and the opportunity
    required, in ``diagnostics["insufficient_null_opportunity_strata"]``.
    Adequate strata are let through, including ones whose own pool is
    genuinely empty: ``K=0`` at a stratum's own cutoff, with real opportunity
    behind it, gives point estimate ``q=0`` but a finite positive
    replicate-cluster ``qvalue_upper_95``, named in
    ``diagnostics["zero_exceedance_strata"]`` instead -- the two lists are
    mutually exclusive by construction, never conflated.

    ``stratum_opportunity=None`` (the default) skips the adequacy check
    entirely, unstratified callers (e.g. the frozen :func:`region_fdr`
    wrapper) having no length data to key it against.
    ``null_edge_counts`` supplies the usable exposure in each independent
    null realisation. If omitted, equal exposure summing to ``null_edges`` is
    used for compatibility with direct callers. At least five independent
    realisations are required to estimate replicate-level dispersion; below
    that, affected inference is explicitly refused.
    ``null_stratum_exposures``, when length stratification is active, supplies
    each replicate's actual opportunity for each length stratum. Its pooled
    sum supplies ``N_s`` for the length-specific point/bound scale, while the
    individual values are the cluster-dispersion offsets. Joint support strata
    without joint opportunity retain an explicitly diagnosed global fallback.

    ``diagnostics["insufficient_null_strata"]`` is kept for schema
    compatibility and is now always ``[]``.
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
            diagnostics={
                "pooled_fallback_strata": [], "insufficient_null_strata": [],
                "zero_exceedance_strata": [],
                "insufficient_null_opportunity_strata": [],
                "insufficient_cluster_replicates_strata": [],
                "insufficient_scale_opportunity_strata": [],
            },
        )

    replicates = len(null_area_sets)
    if replicates == 0 or null_edges <= 0:
        nan_f = np.full(m, np.nan, dtype=float)
        neg_i = np.full(m, -1, dtype=np.int64)
        status = (
            "no null replicates available" if replicates == 0 else "null_edges <= 0"
        )
        return RegionFdrResult(
            qvalue=nan_f, tail_fdr=nan_f, qvalue_lower=nan_f,
            qvalue_upper=nan_f, qvalue_resolution=nan_f,
            null_exceedances=neg_i, qvalue_cutoff_rank=neg_i,
            confidence_level=confidence_level,
            diagnostics={
                "status": status, "pooled_fallback_strata": [],
                "zero_exceedance_strata": [],
                "insufficient_null_strata": [],
                "insufficient_null_opportunity_strata": [],
                "insufficient_cluster_replicates_strata": [],
                "insufficient_scale_opportunity_strata": [],
            },
        )

    # ``null_edges`` is the TOTAL number of null edges summed over replicates, and
    # ``null_area_sets`` are likewise pooled over replicates, so the two already
    # correspond -- see the identical comment preserved on the old region_fdr.
    scale = observed_edges / null_edges
    null_area_sets = [np.asarray(a, dtype=float) for a in null_area_sets]
    if null_edge_counts is None:
        # Direct callers historically supplied only the pooled exposure. Equal
        # allocation preserves that exact pooled point estimator while still
        # making the independent-replicate count explicit for uncertainty.
        per_replicate_edge_counts = np.full(
            replicates, float(null_edges) / replicates, dtype=float
        )
        null_edge_exposure_source = "equal_from_pooled_total"
    else:
        per_replicate_edge_counts = np.asarray(null_edge_counts, dtype=float)
        if per_replicate_edge_counts.shape != (replicates,):
            raise ValueError(
                "null_edge_counts must have one entry per null_area_sets replicate"
            )
        if np.any(~np.isfinite(per_replicate_edge_counts)) or np.any(
            per_replicate_edge_counts < 0
        ):
            raise ValueError("null_edge_counts must be finite and non-negative")
        if not np.isclose(per_replicate_edge_counts.sum(), null_edges):
            raise ValueError("null_edge_counts must sum to null_edges")
        null_edge_exposure_source = "measured_per_replicate"

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

    per_length_cluster_exposures: dict[str, np.ndarray] = {}
    if null_stratum_exposures is not None:
        if not stratified:
            raise ValueError(
                "null_stratum_exposures requires observed_lengths and "
                "null_length_sets"
            )
        for label, values in null_stratum_exposures.items():
            exposure = np.asarray(values, dtype=float)
            if exposure.shape != (replicates,):
                raise ValueError(
                    "null_stratum_exposures must contain one value per replicate"
                )
            if np.any(~np.isfinite(exposure)) or np.any(exposure < 0):
                raise ValueError(
                    "null_stratum_exposures must be finite and non-negative"
                )
            if stratum_opportunity is not None and str(label) in stratum_opportunity:
                pooled_opportunity = float(stratum_opportunity[str(label)][1])
                if not np.isclose(exposure.sum(), pooled_opportunity):
                    raise ValueError(
                        "null_stratum_exposures must sum to the pooled null "
                        "stratum opportunity"
                    )
            per_length_cluster_exposures[str(label)] = exposure
        missing_labels = set(np.unique(observed_length_labels)) - set(
            per_length_cluster_exposures
        )
        if missing_labels:
            raise ValueError(
                "null_stratum_exposures missing observed length strata: "
                + ", ".join(sorted(missing_labels))
            )

    # Spec §1c.6 escalation: joint (length x support-bin) null pooling, layered on
    # top of the length strata above. Never activates unless BOTH support arrays
    # are supplied AND length stratification is already on, so every pre-existing
    # caller (which passes neither) keeps bit-identical behaviour.
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
                for n, b in zip(observed_lengths, observed_support_bins, strict=True)
            ],
            dtype=object,
        )
        null_joint_labels_by_replicate = [
            np.array(
                [
                    f"{length_stratum(int(n))}|sbin{int(b)}"
                    for n, b in zip(lengths, bins, strict=True)
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
            for value, count in zip(values, counts, strict=True):
                joint_null_counts[str(value)] = (
                    joint_null_counts.get(str(value), 0) + int(count)
                )

        observed_strata = np.empty(m, dtype=object)
        for i in range(m):
            joint_label = str(observed_joint_labels[i])
            if (
                joint_null_counts.get(joint_label, 0)
                >= _MIN_NULL_REGIONS_PER_JOINT_STRATUM
            ):
                observed_strata[i] = joint_label
            else:
                observed_strata[i] = observed_length_labels[i]
                support_fallback_joint_strata.append(joint_label)
    else:
        observed_strata = observed_length_labels

    strata = np.unique(observed_strata)

    # --- Pass A: distinct-cutoff count per stratum, then the common m_bonferroni
    cutoffs_by_stratum = {
        str(s): _n_distinct(observed_areas[observed_strata == s]) for s in strata
    }
    m_bonferroni = max(cutoffs_by_stratum.values())
    pooled_null_regions = int(sum(a.size for a in null_area_sets))

    # Spec §2c: null-opportunity adequacy is a per-LENGTH-stratum decision,
    # not a per-joint-stratum one -- opportunity is a property of run-length
    # structure, independent of support-bin membership. A length stratum that
    # fails here is refused in full, even if support stratification split it
    # into several joint sub-strata (Pass B below refuses every row whose
    # LENGTH label matches, regardless of which joint stratum it fell into).
    # Run AFTER stratification's own validation (the strict support-bin zips
    # above) and Pass A, so a caller bug still raises regardless of null
    # adequacy and every diagnostics key stays present either way.
    insufficient_null_opportunity_strata: list[dict] = []
    refused_length_strata: set[str] = set()
    if stratum_opportunity is not None and m_opportunity > 0:
        # Unstratified callers report one "_pooled" stratum spanning every
        # length, so their opportunity check must be pooled too -- summing
        # every length stratum's (O_s, N_s) into one bucket, not looking up
        # a "_pooled" key that `stratum_opportunity` (always keyed by real
        # length-stratum names) was never given.
        opportunity_lookup = (
            stratum_opportunity if stratified
            else {
                "_pooled": (
                    sum(o for o, _ in stratum_opportunity.values()),
                    sum(n for _, n in stratum_opportunity.values()),
                )
            }
        )
        for length_label in np.unique(observed_length_labels):
            length_label = str(length_label)
            r_s = int(np.sum(observed_length_labels == length_label))
            if r_s == 0:
                continue
            o_s, n_s = opportunity_lookup.get(length_label, (0.0, 0.0))
            if o_s <= 0:
                required_n = None
                inadequate = True
            else:
                required_n = m_opportunity * o_s / r_s
                inadequate = n_s < required_n
            if inadequate:
                refused_length_strata.add(length_label)
                insufficient_null_opportunity_strata.append({
                    "stratum": length_label,
                    "observed_opportunity": o_s,
                    "null_opportunity": n_s,
                    "observed_regions": r_s,
                    "required_null_opportunity": required_n,
                    "m_opportunity": m_opportunity,
                })

    # --- Pass B: prepare each stratum and estimate replicate-cluster dispersion.
    # The many regions inside one genome-wide member-step draw are correlated;
    # one draw is therefore one statistical unit. Dispersion is estimated at
    # the weakest reported cutoff (the best-populated count on the curve) and
    # used across the nested curve. Empty strata borrow the largest estimable
    # dispersion from this run, rather than claiming Poisson certainty; if no
    # stratum can estimate it, the explicitly recorded quasi-Poisson floor 1 is
    # used.
    qvalue = np.empty(m, dtype=float)
    tail_fdr = np.empty(m, dtype=float)
    qvalue_lower = np.empty(m, dtype=float)
    qvalue_upper = np.empty(m, dtype=float)
    qvalue_resolution = np.empty(m, dtype=float)
    null_exceedances = np.empty(m, dtype=np.int64)
    qvalue_cutoff_rank = np.empty(m, dtype=np.int64)
    # DEPRECATED: always [] now. Kept for schema compatibility with older
    # manifest parsers; scheduled for removal at the next schema bump.
    pooled_fallback_strata: list[str] = []
    # DEPRECATED: the per-stratum refusal that used to populate this list was
    # retracted -- see this function's docstring. Always [] now, kept for
    # schema compatibility alongside `pooled_fallback_strata` above.
    insufficient_null_strata: list[dict] = []
    zero_exceedance_strata: list[dict] = []
    insufficient_cluster_replicates_strata: list[dict] = []
    insufficient_scale_opportunity_strata: list[dict] = []
    cluster_inputs: dict[str, dict] = {}
    cluster_dispersion_by_stratum: dict[str, dict] = {}
    fdr_scale_by_stratum: dict[str, dict] = {}

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
        pooled = np.concatenate(per_replicate) if per_replicate else np.empty(0)
        weakest_cutoff = float(np.min(observed_areas[rows]))
        length_label = str(observed_length_labels[rows[0]])
        if stratified and not is_joint_stratum:
            if stratum_opportunity is None:
                piece_scale = float(scale)
                scale_source = "global_edge_fallback_missing_length_opportunity"
                scale_o = None
                scale_n = None
            else:
                opportunity_pair = stratum_opportunity.get(length_label)
                if opportunity_pair is None:
                    piece_scale = None
                    scale_source = "unavailable_missing_length_opportunity"
                    scale_o = None
                    scale_n = None
                else:
                    scale_o, scale_n = map(float, opportunity_pair)
                    piece_scale = (
                        scale_o / scale_n
                        if scale_o > 0 and scale_n > 0
                        else None
                    )
                    scale_source = (
                        "per_length_opportunity"
                        if piece_scale is not None
                        else "unavailable_nonpositive_length_opportunity"
                    )
        elif is_joint_stratum:
            # No joint length x support-bin opportunity is currently measured.
            # Retain the historical edge scale, but label it explicitly rather
            # than presenting it as opportunity-calibrated.
            piece_scale = float(scale)
            scale_source = "global_edge_fallback_joint_support_no_joint_opportunity"
            scale_o = None
            scale_n = None
        else:
            piece_scale = float(scale)
            scale_source = "global_edge_fallback_unstratified"
            scale_o = None
            scale_n = None
        fdr_scale_by_stratum[str(stratum)] = {
            "scale": piece_scale,
            "source": scale_source,
            "observed_opportunity": scale_o,
            "pooled_null_opportunity": scale_n,
        }
        cluster_exposures = per_length_cluster_exposures.get(
            length_label, per_replicate_edge_counts
        )
        status, dispersion, detail = _replicate_cluster_dispersion(
            per_replicate, weakest_cutoff, cluster_exposures
        )
        cluster_inputs[str(stratum)] = {
            "rows": rows, "per_replicate": per_replicate, "pooled": pooled,
            "scale": piece_scale,
        }
        cluster_dispersion_by_stratum[str(stratum)] = {
            "status": status,
            "estimated_dispersion": dispersion,
            "exposure_type": (
                "per_length_opportunity"
                if length_label in per_length_cluster_exposures
                else "usable_edges"
            ),
            **detail,
        }

    estimable_dispersions = [
        entry["estimated_dispersion"]
        for entry in cluster_dispersion_by_stratum.values()
        if entry["status"] == "computed"
        and entry["estimated_dispersion"] is not None
    ]
    borrowed_dispersion = max(estimable_dispersions, default=1.0)

    for stratum in strata:
        stratum_key = str(stratum)
        rows = cluster_inputs[stratum_key]["rows"]
        # Every row sharing one `observed_strata` value also shares one
        # LENGTH label (a joint label embeds its length stratum verbatim; a
        # length-only label IS its length stratum), so any row is
        # representative for the refusal check below.
        length_label = str(observed_length_labels[rows[0]])
        cluster_entry = cluster_dispersion_by_stratum[stratum_key]
        cluster_inadequate = cluster_entry["status"] == "insufficient_clusters"
        scale_inadequate = cluster_inputs[stratum_key]["scale"] is None
        if cluster_inadequate:
            insufficient_cluster_replicates_strata.append({
                "stratum": stratum_key,
                "clusters": cluster_entry["clusters"],
                "minimum_clusters": _MIN_REGION_NULL_CLUSTERS,
                "reason": (
                    "too few independent genome-wide null realisations to "
                    "estimate replicate-cluster dispersion"
                ),
            })
        if scale_inadequate:
            scale_entry = fdr_scale_by_stratum[stratum_key]
            insufficient_scale_opportunity_strata.append({
                "stratum": stratum_key,
                "observed_opportunity": scale_entry["observed_opportunity"],
                "null_opportunity": scale_entry["pooled_null_opportunity"],
                "reason": scale_entry["source"],
            })
        if (
            length_label in refused_length_strata
            or cluster_inadequate
            or scale_inadequate
        ):
            qvalue[rows] = np.nan
            tail_fdr[rows] = np.nan
            qvalue_lower[rows] = np.nan
            qvalue_upper[rows] = np.nan
            qvalue_resolution[rows] = np.nan
            null_exceedances[rows] = -1
            qvalue_cutoff_rank[rows] = -1
            continue
        per_replicate = cluster_inputs[stratum_key]["per_replicate"]
        pooled = cluster_inputs[stratum_key]["pooled"]
        # A stratum whose own pool is completely empty is not refused -- its
        # LENGTH stratum already cleared the opportunity floor above (or the
        # check is off), so an empty own pool means the null, given real
        # opportunity, essentially never produces a region this long: `K=0`
        # at every cutoff, arithmetically well-defined below (point estimate
        # `q=0`, a finite positive cluster-robust upper bound), not an absence of
        # information. Recorded, not refused, so a reader can see the
        # evidence was real.
        if pooled.size == 0:
            zero_exceedance_strata.append({
                "stratum": stratum_key,
                "null_regions": 0,
                "replicates": replicates,
                "null_regions_per_replicate": 0.0,
            })

        if cluster_entry["status"] == "computed":
            dispersion_used = float(cluster_entry["estimated_dispersion"])
            dispersion_source = "own_stratum"
        elif estimable_dispersions:
            dispersion_used = float(borrowed_dispersion)
            dispersion_source = "maximum_estimable_stratum"
        else:
            dispersion_used = 1.0
            dispersion_source = "quasi_poisson_floor"
        cluster_entry["dispersion_used"] = dispersion_used
        cluster_entry["dispersion_source"] = dispersion_source

        piece = _fdr_from_areas_result(
            observed_areas[rows], pooled, cluster_inputs[stratum_key]["scale"],
            confidence_level=confidence_level, m_bonferroni=m_bonferroni,
            dispersion=dispersion_used,
        )
        qvalue[rows] = piece["qvalue"]
        tail_fdr[rows] = piece["tail_fdr"]
        qvalue_lower[rows] = piece["qvalue_lower"]
        qvalue_upper[rows] = piece["qvalue_upper"]
        qvalue_resolution[rows] = piece["qvalue_resolution"]
        null_exceedances[rows] = piece["null_exceedances"].astype(np.int64)
        qvalue_cutoff_rank[rows] = piece["qvalue_cutoff_rank"]

    diagnostics = {
        "fdr_interval_bonferroni_m": m_bonferroni,
        "fdr_interval_cutoffs_by_stratum": cutoffs_by_stratum,
        "fdr_interval_family": "per_stratum",
        "pooled_fallback_strata": pooled_fallback_strata,
        "insufficient_null_strata": insufficient_null_strata,
        "zero_exceedance_strata": zero_exceedance_strata,
        "insufficient_null_opportunity_strata": insufficient_null_opportunity_strata,
        "insufficient_cluster_replicates_strata": (
            insufficient_cluster_replicates_strata
        ),
        "insufficient_scale_opportunity_strata": (
            insufficient_scale_opportunity_strata
        ),
        "m_opportunity": m_opportunity,
        "pooled_null_regions": pooled_null_regions,
        "fdr_uncertainty_method": "replicate_cluster_quasi_poisson",
        "fdr_uncertainty_assumptions": (
            "independent genome-wide null realisations; Var(K_b)=phi*E(K_b); "
            "one common phi across each nested cutoff curve; empty curves "
            "borrow the maximum estimable phi across reported strata"
        ),
        "fdr_uncertainty_cluster_count": replicates,
        "fdr_uncertainty_minimum_clusters": _MIN_REGION_NULL_CLUSTERS,
        "fdr_uncertainty_simultaneous": True,
        "fdr_uncertainty_multiplicity_correction": "bonferroni_per_stratum_curve",
        "fdr_uncertainty_zero_count_safeguard": (
            "scaled-count upper bound with borrowed dispersion or phi=1 floor"
        ),
        "null_edge_exposure_source": null_edge_exposure_source,
        "fdr_cluster_exposure_source": (
            "measured_per_length_opportunity"
            if per_length_cluster_exposures
            else null_edge_exposure_source
        ),
        "null_count_dispersion_by_stratum": cluster_dispersion_by_stratum,
        # Compatibility key: the old warning is superseded because the measured
        # dispersion now widens the reported interval instead of merely warning
        # that a Poisson-only interval is too narrow.
        "null_count_overdispersion_warning": None,
        "null_count_overdispersion_handling": "incorporated_in_cluster_interval",
        "fdr_cluster_dispersion_max": float(borrowed_dispersion),
        # Compatibility key: the global edge scale remains available for old
        # consumers and explicit fallbacks. Ordinary length strata use the
        # opportunity-specific values recorded below.
        "scale": scale,
        "global_edge_scale": scale,
        "fdr_scale_method": "per_length_opportunity_with_explicit_fallback",
        "fdr_scale_by_stratum": fdr_scale_by_stratum,
        "fdr_scale_fallback_strata": sorted(
            key for key, value in fdr_scale_by_stratum.items()
            if value["source"].startswith("global_edge_fallback")
        ),
        "support_stratified": support_stratified,
        "support_stratification_fallback_strata": sorted(
            set(support_fallback_joint_strata)
        ),
        "min_null_regions_per_joint_stratum": _MIN_NULL_REGIONS_PER_JOINT_STRATUM,
    }
    return RegionFdrResult(
        qvalue=qvalue, tail_fdr=tail_fdr, qvalue_lower=qvalue_lower,
        qvalue_upper=qvalue_upper, qvalue_resolution=qvalue_resolution,
        null_exceedances=null_exceedances, qvalue_cutoff_rank=qvalue_cutoff_rank,
        confidence_level=confidence_level, diagnostics=diagnostics,
    )


def check_fdr_decision_vacuity(
    result: RegionFdrResult,
    *,
    scale: float,
    current_replicates: int,
    target_fdr: float,
    replicate_knob: str = "null_replicates",
) -> dict:
    """Describe upper-decision usability without treating no rejection as failure.

    A conservative FDR rule must be allowed to reject nothing, particularly on
    DMR-free controls. The former guard raised whenever no upper q-value cleared
    the target and therefore conflated a scientifically valid no-call result
    with inadequate null replication. Replicate inadequacy is now handled
    separately by ``insufficient_cluster_replicates_strata``; this helper only
    records the complete already-computed curve's outcome for transparency.

    ``replicate_knob`` names the setting that actually controls
    ``current_replicates`` for the caller's advice message: under
    ``region_null_mode="member_step"`` that is ``n_lag_draws``, not
    ``null_replicates`` -- see spec §2a.6/§2e.7. Passing the wrong name told the
    user to change a setting that has no effect on this count.
    """

    diagnostic = {
        "replicate_setting": replicate_knob,
        "independent_clusters": int(current_replicates),
        "target_fdr": float(target_fdr),
        "scale": float(scale),
        "has_finite_upper_curve": False,
        "any_upper_rejection": False,
        "best_region_qvalue_upper_95": None,
    }
    if scale <= 0 or result.qvalue_upper.size == 0:
        return diagnostic
    finite = np.flatnonzero(np.isfinite(result.qvalue_upper))
    if finite.size == 0:
        return diagnostic
    best_position = int(finite[np.argmin(result.qvalue_upper[finite])])
    best_upper = float(result.qvalue_upper[best_position])
    diagnostic.update({
        "has_finite_upper_curve": True,
        "any_upper_rejection": bool(best_upper <= target_fdr),
        "best_region_qvalue_upper_95": best_upper,
        "null_exceedances_at_best_upper": int(
            result.null_exceedances[best_position]
        ),
    })
    return diagnostic


def region_fdr(
    observed_areas: np.ndarray,
    null_area_sets: Sequence[np.ndarray],
    *,
    observed_edges: int,
    null_edges: int,
    observed_lengths: np.ndarray | None = None,
    null_length_sets: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Monotone region-level empirical FDR. Signature frozen; see
    :func:`region_fdr_result` for resolution and replicate-cluster bounds."""

    return region_fdr_result(
        observed_areas, null_area_sets,
        observed_edges=observed_edges, null_edges=null_edges,
        observed_lengths=observed_lengths, null_length_sets=null_length_sets,
    ).qvalue


def qvalue_result_from_track(
    statistic: np.ndarray,
    regions: np.ndarray,
    threshold: float,
    *,
    eligible: np.ndarray | None = None,
    chromosome: np.ndarray | None = None,
    min_cpgs: int = 5,
    max_gap_edges: int = 0,
    replicates: int = 100,
    seed: int = 1,
    length_stratified: bool = True,
) -> tuple[RegionFdrResult, dict]:
    """DEPRECATED: raises for any non-empty ``regions``, because the
    permutation null it depends on (``permutation_null_tracks``) was retired
    (measured at zero usable null regions across 400 replicates). Use
    ``region_fdr_result`` directly with a null built from ``null_tracks()``
    (``null_mode='lag'``) or ``member_step_lag_null_ensemble()``
    (``region_null_mode='member_step'``) instead. See ``qvalues_from_track``
    for the thin-wrapper form used where only the point estimate is needed."""

    warnings.warn(
        "qvalue_result_from_track is deprecated and raises for any non-empty "
        "regions: it depends on permutation_null_tracks, which was retired. "
        "Use region_fdr_result with a null from null_tracks() or "
        "member_step_lag_null_ensemble() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    replicates: int = 100,
    seed: int = 1,
    length_stratified: bool = True,
) -> tuple[np.ndarray, dict]:
    """DEPRECATED: thin wrapper over ``qvalue_result_from_track``, which raises
    for any non-empty ``regions`` (see its docstring for the replacement).
    Kept for existing callers; signature unchanged.
    """

    result, diagnostics = qvalue_result_from_track(
        statistic, regions, threshold,
        eligible=eligible, chromosome=chromosome, min_cpgs=min_cpgs,
        max_gap_edges=max_gap_edges, replicates=replicates, seed=seed,
        length_stratified=length_stratified,
    )
    return result.qvalue, diagnostics


# --------------------------------------------------------------------------
# characterisation — strictly post-selection, strictly descriptive
# --------------------------------------------------------------------------


def characterise_region(values: np.ndarray) -> dict:
    """Describe the sample structure of an already-called region.

    The split is chosen to maximise separation and then measured on the same
    data, so ``delta_beta`` is upward-biased under the null and the returned
    ``pvalue_descriptive`` is not a valid p-value. Both are reported because they
    are useful for reading a region; neither may be used to decide whether the
    region is real. That decision belongs to ``region_qvalue``.
    """

    values = np.asarray(values, dtype=float)
    observed_index = np.flatnonzero(np.isfinite(values))
    observed = values[observed_index]
    empty = {
        "n_observed": int(observed.size),
        "mean_beta": float(np.nan),
        "delta_beta": float(np.nan),
        "largest_gap": float(np.nan),
        "direction": "Uncertain",
        "minor_group_size": 0,
        "minor_group_fraction": float(np.nan),
        "minor_group_samples": np.array([], dtype=int),
        "pvalue_descriptive": float(np.nan),
    }
    if observed.size < 3:
        return empty

    order = np.argsort(observed, kind="mergesort")
    ordered = observed[order]
    gaps = np.diff(ordered)
    split = int(np.argmax(gaps))
    lower = observed_index[order[: split + 1]]
    upper = observed_index[order[split + 1 :]]
    if len(lower) <= len(upper):
        minor, major = lower, upper
    else:
        minor, major = upper, lower

    minor_values = values[minor]
    major_values = values[major]
    delta = float(np.mean(minor_values) - np.mean(major_values))

    pvalue = np.nan
    if minor.size >= 2 and major.size >= 2:
        from scipy import stats

        spread = np.concatenate([minor_values, major_values])
        if np.ptp(spread) > 0:
            pvalue = float(
                stats.ttest_ind(
                    minor_values, major_values, equal_var=False
                ).pvalue
            )

    return {
        "n_observed": int(observed.size),
        "mean_beta": float(np.mean(observed)),
        "delta_beta": delta,
        "largest_gap": float(gaps[split]),
        "direction": "Hypomethylation" if delta < 0 else "Hypermethylation",
        "minor_group_size": int(minor.size),
        "minor_group_fraction": float(minor.size / observed.size),
        "minor_group_samples": minor,
        "pvalue_descriptive": pvalue,
    }


_FIVE_CLASS_POLICY = {
    "name": "five_class_v1",
    "status": "provisional_descriptive_pending_validation",
    "rare_max_fraction": 0.01,
    "rare_minimum_max_samples": 2,
    "cohort_min_fraction": 0.10,
    "stable_group_min_delta_beta": 0.10,
    "stable_group_min_boundary_gap_fraction_delta": 0.10,
    "direction_coherence_min": 0.80,
    "outlier_influence_min": 0.50,
    "bounded_score_retention_min": 0.50,
    "winsor_tail_fraction": 0.05,
    "labels": [
        "cohort_supported",
        "recurrent_subgroup",
        "rare_epivariant",
        "outlier_sensitive",
        "diffuse_coordination",
    ],
    "note": (
        "Post-selection description only. It does not alter region_qvalue, "
        "coordination_significant, or the empirical coordination FDR. "
        "rare_epivariant means candidate rare epivariant; coherent technical "
        "blocks are not identifiable from the selected region alone. Influence "
        "screens the furthest-mean one/two samples, not every possible removal."
    ),
}


# Defaults keep computed and skipped diagnostic rows on the same schema.
_SUPPORT_DIAGNOSTIC_DEFAULTS: dict[str, float | int] = {
    "furthest_mean_sample_index": -1,
    "adjacent_support_score": float("nan"),
    "adjacent_support_score_without_furthest1": float("nan"),
    "adjacent_support_score_without_furthest2": float("nan"),
    "furthest1_adjacent_score_influence_fraction": float("nan"),
    "furthest2_adjacent_score_influence_fraction": float("nan"),
    "winsorised_adjacent_support_score": float("nan"),
    "winsorised_score_retention": float("nan"),
    "minority_direction_coherence": float("nan"),
}


def _adjacent_support_score(block: np.ndarray, config: Eiger2Config) -> float:
    """Mean pairwise-complete adjacent statistic within a selected region."""

    values = np.asarray(block, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        return float("nan")
    per_edge = _pair_statistic(
        values[:-1], values[1:], config.statistic,
        config.min_paired_samples, config.min_paired_fraction,
    )
    finite = np.isfinite(per_edge)
    return float(np.mean(per_edge[finite])) if finite.any() else float("nan")


def _winsorise_rows(values: np.ndarray, tail_fraction: float) -> np.ndarray:
    """Row-wise deterministic winsorisation preserving missing values."""

    result = np.array(values, dtype=float, copy=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lower = np.nanquantile(result, tail_fraction, axis=1)
        upper = np.nanquantile(result, 1.0 - tail_fraction, axis=1)
    result = np.maximum(result, lower[:, None], where=np.isfinite(result), out=result)
    result = np.minimum(result, upper[:, None], where=np.isfinite(result), out=result)
    return result


def biological_support_diagnostics(
    raw_block: np.ndarray,
    statistic_block: np.ndarray,
    region_means: np.ndarray,
    config: Eiger2Config,
) -> dict:
    """Compute post-selection prevalence and influence diagnostics.

    Influence is measured on the adjacent support score after removing the
    one or two furthest-mean samples. Winsorisation is descriptive because the
    discovery and null tracks are not rebuilt with the same transform.
    """

    raw = np.asarray(raw_block, dtype=float)
    score_values = np.asarray(statistic_block, dtype=float)
    means = np.asarray(region_means, dtype=float)
    split = _two_means_partition(means)
    finite_means = np.flatnonzero(np.isfinite(means))
    top_sample = -1
    top_samples = np.empty(0, dtype=int)
    if finite_means.size:
        centre = float(np.nanmedian(means))
        ranked = finite_means[
            np.argsort(-np.abs(means[finite_means] - centre), kind="mergesort")
        ]
        top_samples = ranked[:2]
        top_sample = int(top_samples[0])

    score = _adjacent_support_score(score_values, config)
    score_without_top1 = float("nan")
    if top_sample >= 0:
        without = np.array(score_values, copy=True)
        without[:, top_sample] = np.nan
        score_without_top1 = _adjacent_support_score(without, config)
    score_without_top2 = float("nan")
    if top_samples.size >= 2:
        without_two = np.array(score_values, copy=True)
        without_two[:, top_samples] = np.nan
        score_without_top2 = _adjacent_support_score(without_two, config)
    bounded = _winsorise_rows(
        score_values, _FIVE_CLASS_POLICY["winsor_tail_fraction"]
    )
    bounded_score = _adjacent_support_score(bounded, config)

    def retention(candidate: float) -> float:
        if not (np.isfinite(score) and score > 0 and np.isfinite(candidate)):
            return float("nan")
        return float(candidate / score)

    loo_retention = retention(score_without_top1)
    influence = (
        float(np.clip(1.0 - loo_retention, 0.0, 1.0))
        if np.isfinite(loo_retention) else float("nan")
    )
    loo2_retention = retention(score_without_top2)
    top2_influence = (
        float(np.clip(1.0 - loo2_retention, 0.0, 1.0))
        if np.isfinite(loo2_retention) else float("nan")
    )
    bounded_retention = retention(bounded_score)

    # Measure whether the two-means minority shifts consistently across CpGs.
    minority_samples = np.asarray(split["minority_samples"], dtype=int)
    coherence = float("nan")
    if minority_samples.size and raw.shape[0]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row_medians = np.nanmedian(raw, axis=1)
            minority_shift = np.nanmean(
                raw[:, minority_samples] - row_medians[:, None], axis=1
            )
        finite_shift = np.isfinite(minority_shift) & (minority_shift != 0)
        if finite_shift.any():
            direction = np.sign(np.nanmedian(minority_shift[finite_shift]))
            coherence = float(
                np.mean(np.sign(minority_shift[finite_shift]) == direction)
            )

    computed = {
        "furthest_mean_sample_index": top_sample,
        "adjacent_support_score": score,
        "adjacent_support_score_without_furthest1": score_without_top1,
        "adjacent_support_score_without_furthest2": score_without_top2,
        "furthest1_adjacent_score_influence_fraction": influence,
        "furthest2_adjacent_score_influence_fraction": top2_influence,
        "winsorised_adjacent_support_score": bounded_score,
        "winsorised_score_retention": bounded_retention,
        "minority_direction_coherence": coherence,
    }
    assert computed.keys() == _SUPPORT_DIAGNOSTIC_DEFAULTS.keys(), (
        "biological_support_diagnostics' return keys drifted from "
        "_SUPPORT_DIAGNOSTIC_DEFAULTS -- update both together"
    )
    return {**split, **computed}


def classify_biological_support(
    diagnostics: Mapping[str, object],
    *,
    coordination_significant: bool | None,
    n_observed: int,
) -> str | None:
    """Apply the mutually exclusive provisional five-class hierarchy."""

    if coordination_significant is not True:
        return None
    minority = int(diagnostics.get("minority_size", 0))
    fraction = float(diagnostics.get("minority_fraction", np.nan))
    delta = float(diagnostics.get("minority_delta", np.nan))
    gap_fraction = float(
        diagnostics.get("boundary_gap_fraction_delta", np.nan)
    )
    coherence = float(diagnostics.get("minority_direction_coherence", np.nan))
    influence = float(diagnostics.get(
        "furthest1_adjacent_score_influence_fraction", np.nan
    ))
    influence2 = float(diagnostics.get(
        "furthest2_adjacent_score_influence_fraction", np.nan
    ))
    bounded_retention = float(
        diagnostics.get("winsorised_score_retention", np.nan)
    )
    rare_max = max(
        int(_FIVE_CLASS_POLICY["rare_minimum_max_samples"]),
        int(_FIVE_CLASS_POLICY["rare_max_fraction"] * n_observed),
    )
    coherent = np.isfinite(coherence) and (
        coherence >= _FIVE_CLASS_POLICY["direction_coherence_min"]
    )
    bounded = np.isfinite(bounded_retention) and (
        bounded_retention >= _FIVE_CLASS_POLICY["bounded_score_retention_min"]
    )
    influential = (
        (np.isfinite(influence) and
         influence >= _FIVE_CLASS_POLICY["outlier_influence_min"])
        or (np.isfinite(influence2) and
            influence2 >= _FIVE_CLASS_POLICY["outlier_influence_min"])
    )

    # Coherent private events remain rare candidates if winsorisation attenuates
    # them.
    if 0 < minority <= rare_max and coherent:
        return "rare_epivariant"
    if influential and not bounded:
        return "outlier_sensitive"
    stable_discrete = (
        minority > 0
        and np.isfinite(delta)
        and delta >= _FIVE_CLASS_POLICY["stable_group_min_delta_beta"]
        and np.isfinite(gap_fraction)
        and gap_fraction >= _FIVE_CLASS_POLICY[
            "stable_group_min_boundary_gap_fraction_delta"
        ]
        # Coherence without single-sample influence is an alternative when
        # winsorisation attenuates a small subgroup by construction.
        and (bounded or (coherent and not influential))
    )
    if stable_discrete and np.isfinite(fraction):
        if fraction >= _FIVE_CLASS_POLICY["cohort_min_fraction"]:
            return "cohort_supported"
        if minority > rare_max:
            return "recurrent_subgroup"
    return "diffuse_coordination"


def _diagnostic_status(coordination_call: bool | None) -> str:
    if coordination_call is None:
        return "unavailable"
    return "computed" if coordination_call else "not_selected"


_SUPPORT_OUTPUT_COLUMNS = [
    "region_id",
    "biological_support_policy",
    "biological_class",
    "coordination_significant",
    "two_means_minority_size",
    "two_means_minority_fraction",
    "two_means_delta_beta",
    "two_means_boundary_gap",
    "two_means_boundary_gap_fraction_delta",
    "largest_gap_tail_size_legacy",
    "largest_gap_tail_fraction_legacy",
    "largest_gap_delta_beta_legacy",
    "largest_adjacent_gap_legacy",
    "direction_descriptive",
    "largest_gap_pvalue_descriptive_invalid_for_inference",
    "n_samples_observed",
    "missing_rate",
    "mean_beta",
    "furthest_mean_sample",
    "adjacent_support_score",
    "adjacent_support_score_without_furthest1",
    "furthest1_adjacent_score_influence_fraction",
    "adjacent_support_score_without_furthest2",
    "furthest2_adjacent_score_influence_fraction",
    "winsorised_adjacent_support_score",
    "winsorised_score_retention",
    "minority_direction_coherence",
    "diagnostic_status",
]


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def _standardise_locations(locations, n_rows: int) -> pd.DataFrame:
    frame = pd.DataFrame(locations).copy()
    if len(frame) != n_rows:
        raise ValueError("locations and betas must have the same number of rows")
    lower = {str(c).lower(): c for c in frame.columns}
    chrom_column = next(
        (lower[k] for k in ("chr", "chrom", "chromosome") if k in lower),
        frame.columns[0],
    )
    start_column = lower.get("start", frame.columns[1])
    starts = pd.to_numeric(frame[start_column], errors="raise").astype("int64")
    end_column = lower.get("end")
    ends = (
        pd.to_numeric(frame[end_column], errors="raise").astype("int64")
        if end_column is not None
        else starts + 1
    )
    return pd.DataFrame(
        {
            "chr": frame[chrom_column].astype(str).to_numpy(),
            "start": starts.to_numpy(),
            "end": ends.to_numpy(),
        }
    )


# Human reference chromosomes in their natural biological order -- not
# lexicographic ("chr10" < "chr2" as strings, which a prior version's
# reject-on-mismatch input check enforced, rejecting valid natural-order
# input over it -- spec §2b prerequisite). Any other chromosome name
# (unplaced/alt/random contigs) sorts after this set, lexicographically
# among themselves: deterministic and independent of the order they
# happened to be presented in, which is what makes the canonical order
# reproducible regardless of input order.
CANONICAL_CHROMOSOMES: tuple[str, ...] = tuple(
    f"chr{i}" for i in range(1, 23)
) + ("chrX", "chrY", "chrM")
_CANONICAL_CHROMOSOME_RANK = {
    name: rank for rank, name in enumerate(CANONICAL_CHROMOSOMES)
}


def canonical_chromosome_order(names: Sequence[str]) -> list[str]:
    """The canonical presentation order for a SET of chromosome names.

    A pure function of the SET of distinct names present -- not of the order
    they were given in -- so the same chromosomes always canonicalise to the
    same order regardless of whether the input happened to be presented in
    natural order, lexicographic order, or shuffled (spec §2b prerequisite).
    ``chr1``..``chr22``, ``chrX``, ``chrY``, ``chrM`` sort in that biological
    order; anything else (unplaced, alt, or random contigs) sorts after them,
    lexicographically among themselves.
    """

    unique = set(names)
    known = sorted(
        (n for n in unique if n in _CANONICAL_CHROMOSOME_RANK),
        key=lambda n: _CANONICAL_CHROMOSOME_RANK[n],
    )
    unplaced = sorted(n for n in unique if n not in _CANONICAL_CHROMOSOME_RANK)
    return known + unplaced


def chromosome_sort_permutation(
    locations: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Permutation that groups ``locations`` by chromosome in canonical
    order and sorts positions ascending within each chromosome, plus the
    canonical order of the chromosomes actually present -- recorded in the
    manifest (``chromosome_order``) so a run is reproducible regardless of
    input order (spec §2b prerequisite). Apply the SAME permutation to the
    corresponding matrix rows to keep them aligned with ``locations``.

    Every downstream step that infers genomic proximity from row-index
    proximity (:func:`eligible_edges`, :func:`_null_eligibility`, and
    therefore the null construction §2b builds on) depends on this -- a
    chromosome split across two non-adjacent row blocks, or positions not
    ascending within a chromosome, corrupts the null it builds without
    raising, since nothing here currently checks for the CONDITION this
    permutation guarantees, only relies on it.
    """

    chrom = locations["chr"].to_numpy()
    starts = locations["start"].to_numpy()
    order = canonical_chromosome_order(chrom)
    rank = {name: index for index, name in enumerate(order)}
    chrom_rank = np.array([rank[c] for c in chrom], dtype=np.int64)
    # lexsort's LAST key is primary: chromosome rank first, start ascending
    # within each chromosome second. Stable, so rows already in canonical
    # order (including duplicate positions) keep their relative order.
    permutation = np.lexsort((starts, chrom_rank))
    return permutation, order


def call_dmrs(
    betas,
    locations,
    *,
    sample_names: Sequence[str] | None = None,
    config: Eiger2Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Discover regions of coordinated methylation and describe them.

    Returns ``(regions, region_values, manifest)``.

    ``regions`` carries one row per region. ``region_qvalue`` is the only field
    that expresses confidence that the region is real; every ``*_descriptive``
    field is post-selection and is for reading, not for filtering.
    """

    config = config or Eiger2Config()
    matrix = np.asarray(betas, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("betas must be a two-dimensional CpG-by-sample matrix")
    n_cpg, n_samples = matrix.shape
    if n_samples < config.min_paired_samples:
        raise ValueError(
            f"at least {config.min_paired_samples} samples are required "
            f"(min_paired_samples); got {n_samples}"
        )
    if sample_names is None:
        sample_names = [f"sample_{i + 1}" for i in range(n_samples)]
    sample_names = [str(s) for s in sample_names]
    if len(sample_names) != n_samples or len(set(sample_names)) != n_samples:
        raise ValueError("sample_names must be unique and match the matrix columns")

    frame = _standardise_locations(locations, n_cpg)
    # Spec §2b prerequisite: group by chromosome in canonical order, sort
    # ascending within each chromosome, regardless of the order the caller
    # presented rows in. Applied here, before any other indexing, so every
    # downstream index (`source_index`, `cpg_index_start`/`_end`) is already
    # relative to this canonical order -- no separate remapping needed.
    permutation, chromosome_order = chromosome_sort_permutation(frame)
    frame = frame.iloc[permutation].reset_index(drop=True)
    matrix = matrix[permutation]
    observed_per_cpg = np.sum(np.isfinite(matrix), axis=1)
    keep = observed_per_cpg >= config.min_observed_samples
    if config.min_sample_fraction > 0:
        keep &= observed_per_cpg >= np.ceil(
            config.min_sample_fraction * n_samples
        )
    matrix = matrix[keep]
    frame = frame.loc[keep].reset_index(drop=True)
    source_index = np.flatnonzero(keep)

    support_policy_manifest = (
        dict(_FIVE_CLASS_POLICY)
        if config.biological_support_policy == "five_class_v1"
        else {
            "name": "off",
            "status": "disabled",
            "labels": list(_FIVE_CLASS_POLICY["labels"]),
            "note": (
                "Enable biological_support_policy='five_class_v1' for the "
                "provisional post-selection classification."
            ),
        }
    )
    manifest: dict = {
        "config": config.to_dict(),
        "n_cpg_input": int(n_cpg),
        "n_cpg_retained": int(matrix.shape[0]),
        "n_cpg_dropped_unobserved": int(np.sum(observed_per_cpg == 0)),
        "n_cpg_dropped_low_coverage": int(
            np.sum(~keep) - np.sum(observed_per_cpg == 0)
        ),
        "min_samples_required": int(
            max(config.min_observed_samples,
                np.ceil(config.min_sample_fraction * n_samples))
        ),
        "n_samples": int(n_samples),
        "chromosome_order": chromosome_order,
        "schema_version": "2.3",
        "fdr_decision": config.fdr_decision,
        "biological_support": {
            "policy": support_policy_manifest,
            "diagnostics_file": (
                "region_support_diagnostics.csv"
                if config.biological_support_policy == "five_class_v1"
                else None
            ),
            "classification_counts": {
                str(label): 0 for label in _FIVE_CLASS_POLICY["labels"]
            },
            "not_classified": 0,
        },
    }
    empty_regions = pd.DataFrame(columns=_REGION_COLUMNS)
    empty_values = pd.DataFrame(columns=["region_id", *sample_names])
    if config.biological_support_policy == "five_class_v1":
        empty_regions.attrs["region_support_diagnostics"] = pd.DataFrame(
            columns=_SUPPORT_OUTPUT_COLUMNS
        )
    if matrix.shape[0] < config.min_cpgs:
        manifest["status"] = "insufficient_cpgs"
        return empty_regions, empty_values, manifest

    residuals = centre_cpgs(matrix)
    basis, basis_diagnostics = estimate_latent_basis(residuals, config)
    manifest["latent_structure"] = basis_diagnostics
    deflated = deflate(residuals, basis, missing=config.deflation_missing)

    statistic = edge_statistic(deflated, config)
    eligible = eligible_edges(frame, config)

    # Deferred import: eiger2.layouts imports FROM this module, so importing it
    # back here (module already fully initialised) avoids a circular import at
    # module load time.
    from .layouts import LENGTH_STRATA, describe_layout, realised_length_strata

    layout_diagnostics = describe_layout(frame, config.max_cpg_gap_bp)
    realised = realised_length_strata(
        layout_diagnostics["layout_max_detectable_region_cpgs"]
    )
    layout_diagnostics["realised_length_strata"] = list(realised)
    layout_diagnostics["collapsed_length_strata"] = [
        s for s in LENGTH_STRATA if s not in realised
    ]
    manifest["layout"] = layout_diagnostics

    # --- support-conditioned threshold ---
    n_cpg_total = deflated.shape[0]
    w = max(config.window, 1)
    finite_mask = np.isfinite(deflated)

    def _member_offsets(window: int, lag: int = 1) -> np.ndarray:
        """Window-member offsets from a base position: the adjacent pair at
        w=1 (offsets [0, lag]), else [0, lag, 2*lag, ..., (w-1)*lag].
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

    # --- the edge threshold ---
    #
    # Default path (`region_null_mode="off"`): the threshold is the OBSERVED
    # track's own `null_quantile` within joint bp-gap and support bins. The
    # locality-breaking null is not built at all -- it tests "is there any
    # regional coordination here?", a hypothesis that is false everywhere on a
    # WGBS background, so its quantile is not the quantile of anything this
    # genome is (docs/FDR.md section 18). Skipping it also removes the whole
    # null ensemble from the default run.
    #
    # `"lag"`/`"member_step"` (explicit request only): unchanged. Those modes
    # segment null TRACKS, whose region areas can only be compared with the
    # observed ones when both are measured against a threshold built the same
    # way, so the legacy null-derived threshold stays with them.
    resolved_region_null_mode = config.region_null_mode
    if resolved_region_null_mode != "off":
        # A library caller never sees cli.py's stderr, and this is not a
        # detail to find out about from the manifest afterwards.
        warnings.warn(
            f"region_null_mode={resolved_region_null_mode!r} calibrates "
            "region_qvalue against the locality-breaking null, which is "
            "misspecified on WGBS backgrounds -- see docs/FDR.md section 18. "
            "The default is 'off': ranked candidates, no genome-background "
            "FDR.",
            UserWarning,
            stacklevel=2,
        )
    structural_gates: dict = {}
    tracks: list = []
    observed_per_edge_threshold: np.ndarray | None = None
    if config.edge_threshold == "observed_quantile":
        if resolved_region_null_mode != "off":
            raise ValueError(
                "edge_threshold='observed_quantile' builds no null ensemble, "
                "so region_null_mode must be 'off' with it; 'member_step' and "
                "'lag' segment null tracks against the lag-null threshold."
            )
        try:
            observed_per_edge_threshold, null_diagnostics = observed_edge_threshold(
                statistic, eligible, frame, observed_support, config,
            )
        except ValueError as exc:
            # Same fail-closed contract as the null path below: no threshold
            # means no candidates are possible, so say why in the manifest
            # rather than raise.
            manifest["status"] = "inference_unreliable"
            manifest["gates"] = {
                "structural": {
                    "numerical_failure": {"passed": False, "error": str(exc)},
                },
            }
            return empty_regions, empty_values, manifest
        threshold = float(null_diagnostics["threshold"])
        manifest["null"] = null_diagnostics
    else:
        # The threshold always comes from the locality-breaking null: it decides
        # which individual edges count as elevated, and that question is unaffected
        # by cohort size in the way the region-level question is.
        try:
            # Construct the expensive edge-null ensemble once. ``null_threshold``
            # accepts precomputed tracks specifically so thresholding, diagnostics,
            # structural gates, and the lag-mode region null all share the exact
            # same deterministic realisations.
            tracks = null_tracks(deflated, config, frame)
            threshold, null_diagnostics = null_threshold(
                deflated, config, frame, tracks=tracks,
                statistic=statistic, eligible=eligible,
            )
        except ValueError as exc:
            # Structural gate (spec §2a.2): numerical failure in the null
            # pipeline -- e.g. the locality-breaking null produced no usable
            # pairs at all. Fail closed with a manifest naming the failure
            # rather than an unhandled exception with no output at all; there is
            # no threshold to segment against, so no candidates are possible.
            manifest["status"] = "inference_unreliable"
            manifest["gates"] = {
                "structural": {
                    "numerical_failure": {"passed": False, "error": str(exc)},
                },
            }
            return empty_regions, empty_values, manifest
        null_diagnostics["mode"] = config.null_mode
        # null_mode="permutation" is rejected at Eiger2Config construction, so
        # this is the only edge-level null reachable here.
        manifest["null"] = null_diagnostics

        # Spec §2b.3: a chromosome too small to admit a feasible lag
        # (chromosome_lag_feasibility, computed once here and reused by
        # null_tracks() above via its own internal call) contributes nothing to
        # the null in any replicate. It is not silently dropped behind a fixed
        # warning threshold -- its share of the genome's own eligible observed
        # edges is measured (`missing_exposure`) and escalated to a structural
        # gate when material.
        chromosome_feasibility = chromosome_lag_feasibility(
            frame, config, observed_eligible=eligible,
        )
        infeasible_chromosome_edges = sum(
            v["eligible_observed_edges"] for v in chromosome_feasibility.values()
            if not v["feasible"]
        )
        total_chromosome_edges = sum(
            v["eligible_observed_edges"] for v in chromosome_feasibility.values()
        )
        missing_exposure = (
            infeasible_chromosome_edges / total_chromosome_edges
            if total_chromosome_edges else 0.0
        )

        # Structural gates (spec §2a.2): signal-independent conditions that are
        # broken regardless of whether real signal is present, so they can hard-
        # fail without the false-rejection risk revision 1's calibration-ratio
        # gates carried (§0.1) -- these do not depend on the observed track at
        # all, only on how much of the null actually survived construction.
        surviving_fraction = (
            len(tracks) / config.null_replicates if config.null_replicates else 0.0
        )
        structural_gates = {
            "surviving_null_tracks": {
                "surviving": len(tracks),
                "null_replicates": config.null_replicates,
                "fraction": surviving_fraction,
                "floor_fraction": 0.5,
                "passed": surviving_fraction >= 0.5,
            },
            "null_coverage": {
                "value": null_diagnostics["null_coverage"],
                "floor": config.min_null_coverage,
                "passed": null_diagnostics["null_coverage"] >= config.min_null_coverage,
            },
            "null_construction": {
                "value": null_diagnostics["null_values"],
                "floor": config.min_null_values,
                "passed": null_diagnostics["null_values"] >= config.min_null_values,
                "note": (
                    "too few pooled null values to constitute a valid construction "
                    "-- promoted from null_threshold's pre-existing warning "
                    "(config.min_null_values) to a hard gate."
                ),
            },
            "chromosome_exposure": {
                "missing_exposure": missing_exposure,
                # Explicit and independently tunable (spec §2b.3's own wording:
                # "rather than an arbitrary percentage" -- an EXPLICIT, DOCUMENTED,
                # SWEEPABLE percentage is the honest form of that, not a floor
                # borrowed from min_null_coverage, a different quantity: that one
                # is about how much of the OFFERED null was usable, this one
                # about how much of the OBSERVED genome has no null
                # representation at all).
                "floor": config.max_missing_null_exposure,
                "passed": missing_exposure <= config.max_missing_null_exposure,
                "infeasible_chromosomes": sorted(
                    name for name, v in chromosome_feasibility.items()
                    if not v["feasible"]
                ),
                "note": (
                    "no feasible lag on these chromosomes (spec §2a.2/§2b.3) -- "
                    "excluded from the null in every replicate, not silently, "
                    "since their share of this run's own eligible observed "
                    "edges is measured and gated here explicitly. See "
                    "manifest['null']['threshold_by_chromosome'] for the cheap "
                    "half of the sensitivity report: the same null_quantile "
                    "recomputed per feasible chromosome, alongside the pooled "
                    "threshold it is a comparison for."
                ),
            },
        }

        # The member-step region null has its own per-chromosome construction
        # requirement. A preferred 100-position track is not a structural floor:
        # shorter chromosomes may contribute down to the configured minimum
        # detectable region, with per-length opportunity checked later. Only
        # chromosomes unable to support even that absolute minimum count as
        # missing exposure. The edge-null exposure diagnostic remains separate.
        if resolved_region_null_mode == "member_step":
            member_step_feasibility = member_step_chromosome_feasibility(
                frame, config, observed_eligible=eligible,
            )
            member_step_missing_edges = sum(
                v["eligible_observed_edges"]
                for v in member_step_feasibility.values()
                if not v["feasible"]
            )
            member_step_total_edges = sum(
                v["eligible_observed_edges"] for v in member_step_feasibility.values()
            )
            member_step_missing_exposure = (
                member_step_missing_edges / member_step_total_edges
                if member_step_total_edges else 0.0
            )
            structural_gates["region_null_chromosome_exposure"] = {
                "mode": "member_step",
                "missing_exposure": member_step_missing_exposure,
                "floor": config.max_missing_null_exposure,
                "passed": (
                    member_step_missing_exposure <= config.max_missing_null_exposure
                ),
                "infeasible_chromosomes": sorted(
                    name for name, value in member_step_feasibility.items()
                    if not value["feasible"]
                ),
                "preferred_chromosomes": sorted(
                    name for name, value in member_step_feasibility.items()
                    if value["construction"] == "preferred"
                ),
                "fallback_chromosomes": sorted(
                    name for name, value in member_step_feasibility.items()
                    if value["construction"] == "fallback"
                ),
                "chromosomes": member_step_feasibility,
                "note": (
                    "member-step-specific exposure gate. min_subtrack_length is "
                    "preferred; fallback chromosomes use the largest feasible "
                    "lag that still supports min_cpgs positions. Only chromosomes "
                    "unable to support min_cpgs at null_min_lag are absent."
                ),
            }

    structural_gate_failure = not all(
        gate["passed"] for gate in structural_gates.values()
    )

    if resolved_region_null_mode == "lag":
        region_tracks = tracks
        # One entry per replicate, each itself a list of that replicate's
        # per-chromosome (chromosome, lag, start, length) segments (spec
        # §2b.2) -- a single scalar lag no longer describes a whole
        # replicate now that one replicate spans multiple chromosomes.
        region_lags = [segments for _, _, segments in tracks]
    elif resolved_region_null_mode == "off":
        region_tracks = []
        region_lags = []
    else:
        region_tracks, region_lags = member_step_lag_null_ensemble(
            deflated, config, frame,
        )
    manifest["gates"] = {
        "region_null_mode": resolved_region_null_mode,
        "n_lag_draws": len(region_lags),
        "lags_used": region_lags,
        "structural": structural_gates,
        "calibration_diagnostic_note": CALIBRATION_DIAGNOSTIC_NOTE,
    }
    if resolved_region_null_mode == "off":
        manifest["region_fdr_note"] = NO_CALIBRATED_NULL_NOTE
    if resolved_region_null_mode == "member_step" and region_lags:
        # Per chromosome, not genome-wide (spec §2b.4): each segment's own
        # implied sub-track length depends on ITS OWN chromosome's row
        # count, not the genome-wide total -- flattened across every
        # replicate and chromosome, min/max the range actually realised.
        w_for_subtrack = max(config.window, 1)
        chromosome_sizes = {
            name: end - start
            for name, start, end in _chromosome_blocks(frame["chr"].to_numpy())
        }
        implied_lengths: list[int] = []
        implied_by_chromosome: dict[str, list[int]] = {}
        forced_boundaries = 0
        offered_positions = 0
        for replicate_segments in region_lags:
            for name, lag, n_subtracks in replicate_segments:
                full_size = chromosome_sizes[name] - lag * (w_for_subtrack - 1)
                if full_size <= 0:
                    continue
                shortest, remainder = divmod(full_size, lag)
                longest = shortest + int(remainder > 0)
                implied_lengths.extend((shortest, longest))
                implied_by_chromosome.setdefault(name, []).extend(
                    (shortest, longest)
                )
                forced_boundaries += int(n_subtracks)
                offered_positions += int(full_size)
        if implied_lengths:
            manifest["gates"]["implied_subtrack_length_range"] = [
                max(implied_lengths), min(implied_lengths),
            ]
            gate = structural_gates["region_null_chromosome_exposure"]
            gate["realised_subtrack_length_range"] = {
                name: [min(values), max(values)]
                for name, values in implied_by_chromosome.items()
            }
            gate["forced_boundary_fraction"] = (
                forced_boundaries / offered_positions if offered_positions else 0.0
            )
            gate["forced_boundaries"] = forced_boundaries
            gate["offered_null_positions"] = offered_positions

    # Gate 1 (ACF match): computed unconditionally, here, because it needs
    # only the observed track / eligibility / region null already built above
    # -- unlike Gate 2 (region exchangeability, below), it does not depend on
    # any region ever being called, so it is available even on the
    # "no_regions" early-return path. Informative regardless of whether real
    # signal is present; only a red flag on genuinely DMR-free data. Uses
    # `region_tracks`, not the edge-level `tracks` -- Gate 1 validates
    # whichever null actually forms regions.
    # Gate 1 needs a region null to compare the observed track against. With
    # `region_null_mode="off"` there is none, so it is recorded as skipped
    # rather than computed against an empty ensemble and failed.
    manifest["gates"]["acf"] = (
        acf_gate(statistic, eligible, region_tracks, config.window)
        if region_tracks
        else {"skipped": True, "reason": "region_null_mode='off'"}
    )

    dispersion_cut = (
        config.support_dispersion_cut if config.support_dispersion_cut is not None
        else _DEFAULT_SUPPORT_DISPERSION_CUT
    )

    if config.support_binned_threshold == "off":
        support_binning = "forced_off"
    elif config.support_binned_threshold == "on":
        support_binning = "forced_on"
    elif np.isfinite(support_dispersion) and support_dispersion > dispersion_cut:
        support_binning = "auto_on"
    else:
        support_binning = "auto_off"

    support_binning_skipped_reason: str | None = None
    if resolved_region_null_mode != "lag" and support_binning != "forced_off":
        # Support-conditioned thresholding's per-replicate reconstruction
        # assumes each null track's per-position member set follows
        # _lag_null_track's single-lag formula on a CONTIGUOUS track -- true
        # of "lag"'s tracks, false of "member_step"'s (several offset
        # sub-tracks concatenated, so the formula misdescribes positions near
        # a boundary) and vacuous for "off". Not extended to either -- a
        # known, documented limitation, not a silent miscalibration -- so
        # both fall back to the plain scalar threshold (see
        # `region_track_thresholds` below).
        support_binning_skipped_reason = "region_null_mode_not_lag"
        support_binning = "forced_off"

    support_bin_edges: np.ndarray = np.empty(0)
    support_bin_merges: list = []
    support_thresholds: list = []
    # `null_full_support`/`pooled_null_support` are only ever assigned inside the
    # `if binning_requested and ...:` block below, but diagnostic code later
    # references `null_full_support` unconditionally under a
    # `support_binning in ("auto_on", "forced_on")` gate that is today always
    # equivalent to "that block ran". There's no structural guarantee of that
    # equivalence holding forever, so initialize defensively here to turn a
    # future NameError into an empty (inert) list instead.
    null_full_support: list[np.ndarray] = []
    # The default path already has its per-edge array from
    # `observed_edge_threshold`; the null-derived paths start scalar and are
    # replaced below if support binning succeeds.
    per_edge_threshold: float | np.ndarray = (
        threshold if observed_per_edge_threshold is None
        else observed_per_edge_threshold
    )
    # Per-replicate threshold arrays fed to segment()/region_area() for each
    # null track, aligned 1:1 with `tracks`. Defaults to the scalar threshold;
    # replaced below when binning succeeds. Each null track's threshold must
    # come from THAT replicate's own support
    # geometry (own lag), never from slicing the observed per_edge_threshold
    # array, since null tracks are not the same length as the observed track
    # and do not share its position-to-bin mapping.
    null_edge_thresholds: list[float | np.ndarray] = [threshold] * len(tracks)

    binning_requested = support_binning in ("forced_on", "auto_on")
    if binning_requested:
        # Per null replicate, compute that replicate's OWN support (using its own
        # lag, mirroring _lag_null_track's member set exactly) and pool this
        # replicate's STATISTIC VALUES -- not its support values -- into the bin
        # each position's support falls into. threshold[b] is then the
        # null_quantile of all pooled statistic values across every replicate
        # whose position landed in bin b.
        null_support_parts = []
        # (support_array, value_array) per replicate, usable-filtered
        null_value_by_replicate = []
        null_full_support = []  # full-length (track.size) support array per replicate
        for track, track_eligible, segments in tracks:
            # Each segment reconstructs its OWN support against `deflated`'s
            # genome-wide row indexing, using that segment's own lag and
            # genome-wide start offset (spec §2b.2: one replicate now spans
            # several chromosomes, each with its own lag, so a single scalar
            # lag can no longer describe the whole track). `valid_null` is
            # bounded to the segment's own extent, not the genome-wide
            # total, so a window can never be read as reaching into the
            # NEXT chromosome's rows.
            support = np.zeros(track.size, dtype=np.int64)
            valid_null = np.zeros(track.size, dtype=bool)
            position = 0
            for _name, seg_lag, seg_start, seg_length in segments:
                null_offsets = _member_offsets(w, seg_lag)
                null_base = seg_start + np.arange(seg_length)
                seg_valid = null_base + null_offsets.max() < seg_start + seg_length
                seg_support = np.zeros(seg_length, dtype=np.int64)
                seg_support[seg_valid] = _window_support(
                    finite_mask, null_base[seg_valid], null_offsets,
                )
                support[position : position + seg_length] = seg_support
                valid_null[position : position + seg_length] = seg_valid
                position += seg_length
            null_full_support.append(support)
            usable = track_eligible & np.isfinite(track) & valid_null
            null_support_parts.append(support[usable])
            null_value_by_replicate.append((support[usable], track[usable]))

        pooled_null_support = (
            np.concatenate(null_support_parts) if null_support_parts else np.empty(0)
        )
        bin_result = _support_bin_edges(
            observed_support[usable_mask], config.support_bins,
            config.min_null_values_per_support_bin, pooled_null_support,
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
            # satisfied min_null_values_per_support_bin, but guard defensively)
            # falls back to the unconditioned scalar threshold for that bin only
            per_bin_threshold = np.where(
                np.isfinite(per_bin_threshold), per_bin_threshold, threshold,
            )
            per_edge_threshold = per_bin_threshold[observed_bins]
            support_thresholds = per_bin_threshold.tolist()
            # each null track's own threshold array, from its OWN support's bin
            # assignment -- never the observed track's per_edge_threshold
            null_edge_thresholds = [
                per_bin_threshold[_assign_support_bin(support, support_bin_edges)]
                for support in null_full_support
            ]

    manifest["support_dispersion"] = support_dispersion
    manifest["support_dispersion_cut"] = dispersion_cut
    manifest["support_binning"] = support_binning
    manifest["support_bin_edges"] = support_bin_edges.tolist()
    manifest["support_bin_merges"] = support_bin_merges
    manifest["support_thresholds"] = support_thresholds
    if support_binning_skipped_reason is not None:
        manifest["support_binning_skipped_reason"] = support_binning_skipped_reason

    region_threshold_mode_value = (
        "observed_gap_support_binned" if observed_per_edge_threshold is not None
        else "support_binned" if support_binning in ("auto_on", "forced_on")
        else "single"
    )

    # `null_edge_thresholds` is 1:1 with `tracks` (the edge null); when
    # `region_null_mode="lag"`, `region_tracks IS tracks`, so it applies
    # directly. Otherwise support-conditioning is forced off above (this
    # `region_threshold_mode_value` will read "single"), so every region-null
    # entry gets the same plain scalar `threshold` the observed track itself
    # gets in that case -- observed and null stay on the identical segmenter.
    region_track_thresholds: list[float | np.ndarray] = (
        null_edge_thresholds if resolved_region_null_mode == "lag"
        else [threshold] * len(region_tracks)
    )

    regions = segment(
        statistic,
        eligible,
        per_edge_threshold,
        min_cpgs=config.min_cpgs,
        max_gap_edges=config.max_gap_edges,
    )
    manifest["n_regions"] = len(regions)
    if regions.size == 0:
        # A structural gate failure is a property of the null's construction,
        # independent of whether any candidate happened to be called -- so it
        # still headlines "status" here, even though there are no rows to
        # annotate either way.
        manifest["status"] = (
            "inference_unreliable" if structural_gate_failure else "no_regions"
        )
        return empty_regions, empty_values, manifest

    # Computed once, in one place, so both the row-building loop below and the
    # region-level exposure diagnostic can read it by position rather than
    # recomputing it.
    manifest["region_support_median_descriptive"] = [
        float(np.median(observed_support[first:last])) for first, last in regions
    ]

    observed_areas = region_area(statistic, regions, per_edge_threshold)
    null_areas = []
    null_lengths = []
    null_edge_total = 0
    null_edge_counts: list[int] = []
    # Reused (rather than re-segmenting) for `region_rate_by_support_bin`:
    # each entry is the `segment()` output for that replicate, already
    # computed against that replicate's OWN per-bin threshold array
    # (`null_track_threshold` below), exactly mirroring the observed-region
    # convention. Populated for every replicate regardless of whether support
    # binning is active, since it's cheap to keep and the gating on
    # `support_binning in ("auto_on", "forced_on")` happens once, below,
    # before it's used.
    null_regions_by_replicate: list[np.ndarray] = []
    # Spec §2c: retain per-replicate opportunity as the cluster dispersion
    # offset, as well as the pooled N_s used by the adequacy guard. This is
    # essential for fallback chromosomes: short- and long-region opportunity
    # can differ materially even when total usable-edge counts are similar.
    null_stratum_opportunity = dict.fromkeys(_LENGTH_STRATUM_MIN_EDGES, 0.0)
    null_stratum_opportunity_by_replicate = {
        s: [] for s in _LENGTH_STRATUM_MIN_EDGES
    }
    for (track, null_eligible, *_rest), null_track_threshold in zip(
        region_tracks, region_track_thresholds, strict=True
    ):
        # the null is masked by the same kind of rule as the data, so that the
        # two are genuinely passed through the identical segmenter. Each track's
        # threshold comes from ITS OWN support-conditioned bin assignment (see
        # `null_edge_thresholds` above), exactly mirroring the existing scalar
        # flow -- previously every null shared the single scalar `threshold`;
        # now every null shares the same per-bin threshold VALUES but each
        # applies them through its own support geometry.
        usable = null_eligible & np.isfinite(track)
        replicate_edge_count = int(usable.sum())
        null_edge_counts.append(replicate_edge_count)
        null_edge_total += replicate_edge_count
        replicate_opportunity = _stratum_opportunity(
            _run_lengths_from_mask(usable), min_cpgs=config.min_cpgs
        )
        for s, opportunity in replicate_opportunity.items():
            null_stratum_opportunity[s] += opportunity
            null_stratum_opportunity_by_replicate[s].append(opportunity)
        null_regions = segment(
            track,
            usable,
            null_track_threshold,
            min_cpgs=config.min_cpgs,
            max_gap_edges=config.max_gap_edges,
        )
        null_areas.append(region_area(track, null_regions, null_track_threshold))
        null_lengths.append(
            (null_regions[:, 1] - null_regions[:, 0] + 1)
            if null_regions.size
            else np.empty(0, dtype=int)
        )
        null_regions_by_replicate.append(null_regions)
    observed_lengths = regions[:, 1] - regions[:, 0] + 1

    # Gate 2 (region-level exchangeability). Unlike Gate 1 above, this needs
    # `null_regions_by_replicate` -- the null tracks actually segmented into
    # candidate regions -- which is only built on this (regions.size > 0)
    # path, matching the region_rate_by_support_bin diagnostic's own scope
    # below: both are silent on the "no_regions" early return, same as
    # `manifest["fdr"]`.
    # Like Gate 1 above: with no region null there is nothing to compare the
    # observed region rate against, so the gate is skipped rather than failed
    # against an empty pool.
    manifest["gates"]["region_exchangeability"] = (
        {"skipped": True, "reason": "region_null_mode='off'"}
        if not region_tracks
        else region_exchangeability_gate(
        observed_regions=len(regions),
        observed_eligible_edges=int(np.sum(eligible & np.isfinite(statistic))),
        null_regions_total=int(sum(len(r) for r in null_regions_by_replicate)),
        null_eligible_edges=null_edge_total,
        min_null_regions_for_fdr=config.min_null_regions_for_fdr,
        )
    )
    # `s = observed_eligible_edges / null_eligible_edges` is
    # `region_fdr_result`'s finite-null resolution scale
    # (`region_qvalue_resolution = s / R`) -- SMALLER is better (more null
    # volume). Recorded explicitly, the right way round, so a user can see
    # what their q-values are quantised at without re-deriving it from
    # `region_exchangeability`'s edge counts.
    manifest["gates"]["s"] = (
        int(np.sum(eligible & np.isfinite(statistic))) / max(null_edge_total, 1)
    )
    manifest["gates"]["implied_region_qvalue_resolution_at_R1"] = manifest["gates"]["s"]

    # --- region-level exposure diagnostic ---
    # Compares the observed rate of CALLED REGIONS per support bin against the
    # null's rate (both per 10,000 eligible edges): equalising the per-edge
    # threshold does not guarantee regions FORM at the same rate per bin --
    # clustering could still be support-driven even with fair edges. Null
    # regions reuse `null_regions_by_replicate` (already segmented against
    # each replicate's own per-bin threshold) rather than re-segmenting, so
    # this cannot drift from how the null was actually segmented; each null
    # region's support-median comes from that replicate's own full-length
    # support array. Eligible-edge counts per bin reuse
    # `observed_support[usable_mask]`/`pooled_null_support` from the binning
    # setup above. These same per-region bin assignments also stratify the
    # region-FDR null pool below; default None is length-only pooling, which
    # is what every run with support binning off gets.
    observed_support_bins_for_fdr = None
    null_support_bin_sets_for_fdr = None
    if support_binning in ("auto_on", "forced_on"):
        n_final_bins = support_bin_edges.size + 1
        region_bins = _assign_support_bin(
            np.asarray(manifest["region_support_median_descriptive"], dtype=float),
            support_bin_edges,
        )
        observed_regions_by_bin = np.bincount(region_bins, minlength=n_final_bins)

        null_bin_hits = []
        for null_regions, support in zip(
            null_regions_by_replicate, null_full_support, strict=True
        ):
            for first, last in null_regions:
                support_median = float(np.median(support[first:last]))
                null_bin_hits.append(support_median)
        if null_bin_hits:
            null_region_bins = _assign_support_bin(
                np.asarray(null_bin_hits, dtype=float), support_bin_edges
            )
            null_regions_by_bin = np.bincount(
                null_region_bins, minlength=n_final_bins
            )
        else:
            null_regions_by_bin = np.zeros(n_final_bins, dtype=int)

        observed_edges_by_bin = np.bincount(
            _assign_support_bin(observed_support[usable_mask], support_bin_edges),
            minlength=n_final_bins,
        )
        null_edges_by_bin = np.bincount(
            _assign_support_bin(pooled_null_support, support_bin_edges),
            minlength=n_final_bins,
        )

        region_rate_by_support_bin: dict = {}
        for b in range(n_final_bins):
            obs_edges = int(observed_edges_by_bin[b])
            n_null_edges = int(null_edges_by_bin[b])
            obs_regions = int(observed_regions_by_bin[b])
            n_null_regions = int(null_regions_by_bin[b])
            observed_rate = (
                10_000 * obs_regions / obs_edges if obs_edges > 0 else float("nan")
            )
            null_rate = (
                10_000 * n_null_regions / n_null_edges
                if n_null_edges > 0
                else float("nan")
            )
            entry = {
                "observed_regions": obs_regions,
                "observed_eligible_edges": obs_edges,
                "observed_rate": observed_rate,
                "null_regions": n_null_regions,
                "null_eligible_edges": n_null_edges,
                "null_rate": null_rate,
            }
            if n_null_edges == 0:
                entry["ratio"] = None
                entry["note"] = (
                    "null had zero eligible edges in this bin; ratio undefined"
                )
            elif null_rate == 0.0:
                entry["ratio"] = None
                entry["note"] = (
                    "null had zero regions in this bin; ratio undefined"
                    if observed_rate > 0
                    else "both observed and null had zero regions in this bin"
                )
            else:
                entry["ratio"] = observed_rate / null_rate
            region_rate_by_support_bin[int(b)] = entry
        manifest["region_rate_by_support_bin"] = region_rate_by_support_bin

        # Joint (length x support-bin) FDR stratification is defined ON TOP of
        # the length strata, so it only activates when those are on too. Each
        # null region's bin comes from its OWN replicate's full-length support
        # array via the same median-then-_assign_support_bin recipe used for the
        # observed regions above, so observed and null are binned identically.
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
    # Spec §2c: O_s from the OBSERVED track's own usable mask -- independent
    # of `config.length_stratified_fdr`, since opportunity is a property of
    # where a region of stratum s's length COULD have formed at all, not of
    # whether the estimator is currently reporting per length stratum.
    observed_stratum_opportunity = _stratum_opportunity(
        _run_lengths_from_mask(eligible & np.isfinite(statistic)),
        min_cpgs=config.min_cpgs,
    )
    stratum_opportunity = {
        s: (observed_stratum_opportunity[s], null_stratum_opportunity[s])
        for s in _LENGTH_STRATUM_MIN_EDGES
    }
    fdr_result = region_fdr_result(
        observed_areas,
        null_areas,
        observed_edges=int(np.sum(eligible & np.isfinite(statistic))),
        null_edges=max(null_edge_total, 1),
        observed_lengths=observed_lengths if config.length_stratified_fdr else None,
        null_length_sets=null_lengths if config.length_stratified_fdr else None,
        observed_support_bins=observed_support_bins_for_fdr,
        null_support_bin_sets=null_support_bin_sets_for_fdr,
        stratum_opportunity=stratum_opportunity,
        m_opportunity=config.m_opportunity,
        null_edge_counts=null_edge_counts,
        null_stratum_exposures=(
            null_stratum_opportunity_by_replicate
            if config.length_stratified_fdr else None
        ),
    )
    if structural_gate_failure:
        # A structural gate (§2a.2) failed even though region_fdr_result's
        # own aggregate-adequacy check happened to clear -- e.g. coverage
        # below floor with enough pooled null REGIONS by count alone. That
        # internal check is orthogonal (and, per §2c, on its way to being
        # replaced by a per-stratum one), so it is left untouched; every
        # inferential field is overridden here instead, uniformly, whichever
        # gate tripped.
        m = len(fdr_result.qvalue)
        nan_f = np.full(m, np.nan, dtype=float)
        neg_i = np.full(m, -1, dtype=np.int64)
        fdr_result = RegionFdrResult(
            qvalue=nan_f, tail_fdr=nan_f, qvalue_lower=nan_f,
            qvalue_upper=nan_f, qvalue_resolution=nan_f,
            null_exceedances=neg_i, qvalue_cutoff_rank=neg_i,
            confidence_level=fdr_result.confidence_level,
            diagnostics={
                **fdr_result.diagnostics,
                "status": "structural_gate_failure",
                "structural_gate_failure_reasons": [
                    name for name, g in structural_gates.items() if not g["passed"]
                ],
            },
        )
    elif config.fdr_decision == "upper":
        # Record, but never reject, a valid no-call result. Replicate
        # inadequacy is the separate nullable-inference path above. Name the
        # setting that actually controls independent region-null clusters:
        # member_step uses n_lag_draws, not null_replicates.
        if resolved_region_null_mode == "member_step":
            vacuity_replicate_knob = "n_lag_draws"
            vacuity_current_replicates = len(region_lags)
        else:
            vacuity_replicate_knob = "null_replicates"
            vacuity_current_replicates = config.null_replicates
        fdr_result.diagnostics["upper_decision_diagnostic"] = (
            check_fdr_decision_vacuity(
                fdr_result,
                scale=fdr_result.diagnostics.get("scale", 0.0),
                current_replicates=vacuity_current_replicates,
                target_fdr=config.target_fdr,
                replicate_knob=vacuity_replicate_knob,
            )
        )
    qvalues = fdr_result.qvalue
    manifest["fdr"] = fdr_result.diagnostics
    manifest["null_regions_per_replicate"] = (
        float(np.mean([len(a) for a in null_areas])) if null_areas else float("nan")
    )

    posterior = np.full(len(regions), np.nan, dtype=float)
    posterior_extrapolated = np.zeros(len(regions), dtype=bool)
    if config.compute_posterior:
        from .probability import local_fdr

        estimate = local_fdr(
            observed_areas,
            null_areas,
            observed_edges=int(np.sum(eligible & np.isfinite(statistic))),
            null_edges=max(null_edge_total, 1),
            observed_lengths=(
                observed_lengths if config.length_stratified_fdr else None
            ),
            null_length_sets=(
                null_lengths if config.length_stratified_fdr else None
            ),
            min_null_regions=config.min_null_regions_for_posterior,
        )
        posterior = estimate.posterior
        posterior_extrapolated = estimate.extrapolated
        manifest["posterior"] = {
            "status": estimate.status,
            "pi0": estimate.pi0,
            **estimate.diagnostics,
        }
        # `local_fdr` above is only ever given `observed_lengths`/`null_length_sets`
        # (never support-bin arguments), so this posterior path is length-stratified
        # at most, never support-stratified -- unlike `region_fdr_result` above.
        # Recorded here so a consumer of `manifest["posterior"]` doesn't assume
        # parity with the region-level q-values.
        manifest["posterior"]["support_stratified"] = False

    decision_qvalue = (
        fdr_result.qvalue_upper if config.fdr_decision == "upper" else fdr_result.qvalue
    )
    globally_unavailable = fdr_result.diagnostics.get("status") is not None
    unavailable_strata = {
        entry["stratum"]
        for entry in (
            fdr_result.diagnostics.get("insufficient_null_opportunity_strata") or []
        )
    }
    unavailable_strata.update(
        entry["stratum"].split("|sbin", 1)[0]
        for entry in (
            fdr_result.diagnostics.get("insufficient_cluster_replicates_strata") or []
        )
    )
    unavailable_strata.update(
        entry["stratum"].split("|sbin", 1)[0]
        for entry in (
            fdr_result.diagnostics.get("insufficient_scale_opportunity_strata") or []
        )
    )

    # --- tier two: confirmation from the region's own per-sample means ---
    # A different question from region FDR, with a null that is a property of
    # the region rather than of the genome background, so it needs no
    # reference panel (docs/CONFIRM.md). BH runs across THIS run's candidates,
    # which is why it is computed once here rather than per row.
    if config.confirm_regions:
        from .confirm import MIN_CONFIRM_SAMPLES
        from .confirm import confirm_regions as _confirm

        confirmation = _confirm(matrix, regions, target_fdr=config.target_fdr)
        manifest["confirmation"] = {
            "test": "hartigan_dip_interpolated_table",
            "family": "candidates of this run",
            "target_fdr": config.target_fdr,
            "min_samples_tested": MIN_CONFIRM_SAMPLES,
            "regions_tested": confirmation["n_tested_regions"],
            "regions_not_tested": int(len(regions) - confirmation["n_tested_regions"]),
            "n_confirmed": int(
                sum(value is True for value in confirmation["confirmed"])
            ),
            "note": CONFIRMATION_NOTE,
        }
    else:
        confirmation = None
        manifest["confirmation"] = {"test": None, "note": "confirmation disabled"}

    rows = []
    support_rows: list[dict] | None = (
        [] if config.biological_support_policy == "five_class_v1" else None
    )
    value_rows = []
    for position, (first, last) in enumerate(regions):
        first, last = int(first), int(last)
        block = matrix[first : last + 1]
        counts = np.sum(np.isfinite(block), axis=0)
        means = np.full(n_samples, np.nan, dtype=float)
        np.divide(
            np.nansum(block, axis=0), counts, out=means, where=counts > 0
        )
        description = characterise_region(means)
        coordination_call = (
            None if not np.isfinite(decision_qvalue[position])
            else bool(decision_qvalue[position] <= config.target_fdr)
        )
        if (
            coordination_call
            and config.biological_support_policy == "five_class_v1"
        ):
            support = biological_support_diagnostics(
                block, deflated[first : last + 1], means, config
            )
        else:
            split = _two_means_partition(means)
            support = {**split, **_SUPPORT_DIAGNOSTIC_DEFAULTS}
        minority = int(support["minority_size"])
        minority_delta = float(support["minority_delta"])
        biological_class = (
            classify_biological_support(
                support,
                coordination_significant=coordination_call,
                n_observed=int(description["n_observed"]),
            )
            if config.biological_support_policy == "five_class_v1"
            else None
        )
        chrom = frame["chr"].iloc[first]
        start = int(frame["start"].iloc[first])
        end = int(frame["end"].iloc[last])
        region_id = f"EIGER2_{chrom}_{start}_{end}"
        rows.append(
            {
                "region_id": region_id,
                "chr": str(chrom),
                "start": start,
                "end": end,
                "n_cpg": last - first + 1,
                "length_bp": max(0, end - start),
                "cpg_index_start": int(source_index[first]),
                "cpg_index_end": int(source_index[last]),
                "statistic": config.statistic,
                "statistic_mean": float(
                    np.nanmean(statistic[first:last]) if last > first else np.nan
                ),
                "statistic_min": float(
                    np.nanmin(statistic[first:last]) if last > first else np.nan
                ),
                "statistic_threshold": (
                    float(np.mean(per_edge_threshold[first:last]))
                    if np.ndim(per_edge_threshold)
                    else threshold
                ),
                "region_support_median": manifest[
                    "region_support_median_descriptive"
                ][position],
                "region_threshold_mode": region_threshold_mode_value,
                "region_area": float(observed_areas[position]),
                "region_qvalue": float(qvalues[position]),
                "region_tail_fdr": float(fdr_result.tail_fdr[position]),
                "region_qvalue_lower_95": float(fdr_result.qvalue_lower[position]),
                "region_qvalue_upper_95": float(fdr_result.qvalue_upper[position]),
                "region_qvalue_resolution": float(
                    fdr_result.qvalue_resolution[position]
                ),
                "null_exceedances_at_cutoff": int(
                    fdr_result.null_exceedances[position]
                ),
                "region_qvalue_cutoff_rank": int(
                    fdr_result.qvalue_cutoff_rank[position]
                ),
                "region_evidence_score": float(posterior[position]),
                "evidence_score_extrapolated": bool(posterior_extrapolated[position]),
                "minority_size_descriptive": int(minority),
                "minority_delta_descriptive": float(minority_delta),
                "passes_prevalence": bool(minority >= config.min_prevalence_samples),
                "coordination_significant": (
                    pd.NA if coordination_call is None else coordination_call
                ),
                "biological_class": biological_class,
                # Spec §2a.4: `False` means "tested, not significant" -- a
                # different statement from "not tested because inference was
                # unavailable". A refused run/stratum used to collapse both
                # into `False` (`NaN <= target_fdr` is `False`), the same
                # category error as reporting `q = 0` from an empty null.
                # `pd.NA` (cast to BooleanDtype below) keeps them apart.
                "significant": (
                    pd.NA if not np.isfinite(decision_qvalue[position])
                    else bool(
                        decision_qvalue[position] <= config.target_fdr
                        and minority >= config.min_prevalence_samples
                    )
                ),
                "fdr_status": (
                    "available"
                    if np.isfinite(decision_qvalue[position])
                    # Not "unavailable": nothing failed. No region null was
                    # built, so there is no genome-background FDR to report
                    # (NO_CALIBRATED_NULL_NOTE).
                    else "no_calibrated_null"
                    if resolved_region_null_mode == "off"
                    else (
                        "unavailable_run"
                        if globally_unavailable
                        else (
                            "unavailable_stratum"
                            if length_stratum(last - first + 1) in unavailable_strata
                            # Fail closed if a future refusal path produces a
                            # non-finite decision without naming its stratum.
                            else "unavailable_run"
                        )
                    )
                ),
                **(
                    {
                        "confirm_pvalue": float(
                            confirmation["confirm_pvalue"][position]
                        ),
                        "confirm_qvalue": float(
                            confirmation["confirm_qvalue"][position]
                        ),
                        "confirmed": confirmation["confirmed"][position],
                        "delta_hat": float(confirmation["delta_hat"][position]),
                        "f_hat": float(confirmation["f_hat"][position]),
                        "direction": float(confirmation["direction"][position]),
                        "n_samples_tested": int(
                            confirmation["n_samples_tested"][position]
                        ),
                    }
                    if confirmation is not None
                    else {}
                ),
                "n_samples_observed": description["n_observed"],
                "missing_rate": float(np.mean(~np.isfinite(block))),
                "mean_beta": description["mean_beta"],
                "delta_beta_descriptive": description["delta_beta"],
                "largest_gap_descriptive": description["largest_gap"],
                "direction_descriptive": description["direction"],
                "minor_group_size_descriptive": description["minor_group_size"],
                "minor_group_fraction_descriptive": description[
                    "minor_group_fraction"
                ],
                "minor_group_samples_descriptive": ";".join(
                    sample_names[i] for i in description["minor_group_samples"]
                ),
                "pvalue_descriptive": description["pvalue_descriptive"],
            }
        )
        if support_rows is not None:
            top_sample = int(support["furthest_mean_sample_index"])
            support_rows.append(
                {
                    "region_id": region_id,
                    "biological_support_policy": config.biological_support_policy,
                    "biological_class": biological_class,
                    "coordination_significant": (
                        pd.NA if coordination_call is None else coordination_call
                    ),
                    "two_means_minority_size": minority,
                    "two_means_minority_fraction": float(
                        support["minority_fraction"]
                    ),
                    "two_means_delta_beta": minority_delta,
                    "two_means_boundary_gap": support["boundary_gap"],
                    "two_means_boundary_gap_fraction_delta": support[
                        "boundary_gap_fraction_delta"
                    ],
                    "largest_gap_tail_size_legacy": description[
                        "minor_group_size"
                    ],
                    "largest_gap_tail_fraction_legacy": description[
                        "minor_group_fraction"
                    ],
                    "largest_gap_delta_beta_legacy": description["delta_beta"],
                    "largest_adjacent_gap_legacy": description["largest_gap"],
                    "direction_descriptive": description["direction"],
                    "largest_gap_pvalue_descriptive_invalid_for_inference": (
                        description["pvalue_descriptive"]
                    ),
                    "n_samples_observed": description["n_observed"],
                    "missing_rate": float(np.mean(~np.isfinite(block))),
                    "mean_beta": description["mean_beta"],
                    "furthest_mean_sample": (
                        sample_names[top_sample] if top_sample >= 0 else None
                    ),
                    "adjacent_support_score": support["adjacent_support_score"],
                    "adjacent_support_score_without_furthest1": support[
                        "adjacent_support_score_without_furthest1"
                    ],
                    "furthest1_adjacent_score_influence_fraction": support[
                        "furthest1_adjacent_score_influence_fraction"
                    ],
                    "adjacent_support_score_without_furthest2": support[
                        "adjacent_support_score_without_furthest2"
                    ],
                    "furthest2_adjacent_score_influence_fraction": support[
                        "furthest2_adjacent_score_influence_fraction"
                    ],
                    "winsorised_adjacent_support_score": support[
                        "winsorised_adjacent_support_score"
                    ],
                    "winsorised_score_retention": support[
                        "winsorised_score_retention"
                    ],
                    "minority_direction_coherence": support[
                        "minority_direction_coherence"
                    ],
                    "diagnostic_status": _diagnostic_status(coordination_call),
                }
            )
        value_rows.append(means)

    region_frame = pd.DataFrame(rows)
    # Ranking (spec: the discovery output is a ranked list). Chromosomes keep
    # canonical order; within a chromosome the strongest region comes first,
    # by `region_area` descending. `rank` is dense, 1..n, in exactly the order
    # of the rows it labels, so a reader can take "the top k" without sorting
    # anything themselves and without a q-value.
    chromosome_position = {
        name: index
        for index, name in enumerate(
            canonical_chromosome_order(region_frame["chr"].unique().tolist())
        )
    }
    order = np.lexsort(
        (
            -region_frame["region_area"].to_numpy(dtype=float),
            region_frame["chr"].map(chromosome_position).to_numpy(dtype=np.int64),
        )
    )
    region_frame = region_frame.iloc[order].reset_index(drop=True)
    value_rows = [value_rows[position] for position in order]
    if support_rows is not None:
        support_rows = [support_rows[position] for position in order]
    region_frame.insert(0, "rank", np.arange(1, len(region_frame) + 1, dtype=np.int64))
    region_frame = region_frame[
        [column for column in _REGION_COLUMNS if column in region_frame]
    ]
    # pandas would otherwise infer `object` dtype from a column mixing `bool`
    # and `pd.NA`; the nullable BooleanDtype is what makes `pd.NA` survive a
    # CSV round trip as a genuinely missing value rather than a stringified
    # "<NA>" object (spec §2a.4).
    region_frame["significant"] = region_frame["significant"].astype("boolean")
    region_frame["coordination_significant"] = region_frame[
        "coordination_significant"
    ].astype("boolean")
    if "confirmed" in region_frame:
        # Same contract as `significant`: pd.NA is "not tested" (fewer than
        # MIN_CONFIRM_SAMPLES samples observed enough of the region), which is
        # a different statement from False, "tested, not separated".
        region_frame["confirmed"] = region_frame["confirmed"].astype("boolean")
    if support_rows is not None:
        support_frame = pd.DataFrame(
            support_rows, columns=_SUPPORT_OUTPUT_COLUMNS
        )
        if len(support_frame):
            support_frame["coordination_significant"] = support_frame[
                "coordination_significant"
            ].astype("boolean")
        # Preserve the three-return-value API while exposing opt-in diagnostics.
        region_frame.attrs["region_support_diagnostics"] = support_frame
    values_frame = pd.DataFrame(value_rows, columns=sample_names)
    values_frame.insert(0, "region_id", region_frame["region_id"].to_numpy())

    # Run status (spec §2a.1): `inference_unreliable` when any inferential
    # column was suppressed for the whole run -- region_fdr_result sets a
    # "status" diagnostics key on every one of its own refusal paths, and the
    # structural-gate override above does the same, so its mere presence
    # (never set on the success path) is sufficient. `ok_with_warnings` is a
    # REPORTING label, not a gate (§0.1/§2a.3): calibration diagnostics are
    # confounded with genuine signal on real data, so none of these suppress
    # any column here, they only change which status string a reader sees.
    inference_suppressed = fdr_result.diagnostics.get("status") is not None
    acf_gate_result = manifest["gates"].get("acf")
    region_exch = manifest["gates"].get("region_exchangeability")
    null_exchangeability_ratio = null_diagnostics.get("null_exchangeability_ratio")
    calibration_concern = (
        (acf_gate_result is not None and not acf_gate_result.get("passed", True))
        or (
            region_exch is not None
            and not region_exch.get("skipped")
            and region_exch["observed_regions"] > 0
            and not math.isfinite(region_exch["region_exchangeability_ratio"])
        )
        or bool(fdr_result.diagnostics.get("null_count_overdispersion_warning"))
        or (
            null_exchangeability_ratio is not None
            and math.isfinite(null_exchangeability_ratio)
            and null_exchangeability_ratio > 1.25
        )
    )
    # Spec §2c/§2a.1: a length stratum refused for insufficient null
    # opportunity is a STRATUM-LOCAL failure, not a global one -- the null
    # construction itself is sound, it just never had the chance to speak to
    # regions this long. Checked after the global gate (which still wins:
    # nothing is "partially" fine when the null is invalid as a whole) and
    # before calibration concerns (an actual suppression of rows outranks a
    # diagnostic that suppresses nothing).
    stratum_refused = bool(
        fdr_result.diagnostics.get("insufficient_null_opportunity_strata")
        or fdr_result.diagnostics.get("insufficient_cluster_replicates_strata")
        or fdr_result.diagnostics.get("insufficient_scale_opportunity_strata")
    )
    if resolved_region_null_mode == "off":
        # The default path. Inference was not attempted, so it was not
        # suppressed: "inference_unreliable" would report a failure that did
        # not happen, and "ok" would imply a q-value that does not exist.
        manifest["status"] = "discovery_only"
    elif inference_suppressed:
        manifest["status"] = "inference_unreliable"
    elif stratum_refused:
        manifest["status"] = "partial_inference"
    elif calibration_concern:
        manifest["status"] = "ok_with_warnings"
    else:
        manifest["status"] = "ok"
    manifest["n_significant"] = int(region_frame["significant"].sum())
    if "confirmed" in region_frame:
        manifest["n_confirmed"] = int(region_frame["confirmed"].fillna(False).sum())
    # FIX_SPEC §WP2: how many of those significant calls sit in the bottom
    # length stratum, the one length x effect regime measured off target (see
    # BOTTOM_STRATUM_FDR_NOTE). Recorded on every run that reaches the FDR
    # path, zero included, so a consumer can tell "none" from "not computed";
    # `significant` is a nullable boolean, so a refused stratum's pd.NA is
    # filled to False rather than truth-tested.
    manifest["fdr"]["n_significant_bottom_stratum"] = int(
        (
            region_frame["significant"].fillna(False)
            & region_frame["n_cpg"].map(
                lambda n: length_stratum(int(n)) == "4-5"
            )
        ).sum()
    )
    manifest["n_passing_qvalue"] = int(
        (region_frame["region_qvalue"] <= config.target_fdr).sum()
    )
    manifest["n_passing_prevalence"] = int(region_frame["passes_prevalence"].sum())
    # FIX_SPEC §WP4: how many significant calls are carried by only one or two
    # samples -- what the CLI's one-per-run note reads (LOW_PREVALENCE_NOTE).
    # Recorded on every run that reaches the FDR path, zero included, so a
    # consumer can tell "none" from "not computed". Counted against the
    # *significant* set, so raising `min_prevalence_samples` above the ceiling
    # empties it by construction and the note falls silent, which is correct:
    # the user has already acted on it.
    manifest["n_significant_low_prevalence"] = int(
        (
            region_frame["significant"].fillna(False)
            & region_frame["minority_size_descriptive"].between(
                1, LOW_PREVALENCE_SAMPLE_CEILING
            )
        ).sum()
    )
    manifest["biological_support"] = {
        "policy": support_policy_manifest,
        "diagnostics_file": (
            "region_support_diagnostics.csv"
            if config.biological_support_policy == "five_class_v1"
            else None
        ),
        "classification_counts": {
            str(label): int((region_frame["biological_class"] == label).sum())
            for label in _FIVE_CLASS_POLICY["labels"]
        },
        "not_classified": int(region_frame["biological_class"].isna().sum()),
    }
    manifest["schema_version"] = "2.3"
    manifest["fdr_decision"] = config.fdr_decision
    return region_frame, values_frame, manifest


_REGION_COLUMNS = [
    "rank",
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
    "region_support_median",
    "region_threshold_mode",
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
    "coordination_significant",
    "biological_class",
    "significant",
    "fdr_status",
    "confirm_pvalue",
    "confirm_qvalue",
    "confirmed",
    "delta_hat",
    "f_hat",
    "direction",
    "n_samples_tested",
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


_REGION_MAIN_COLUMNS = [
    "rank", "region_id", "chr", "start", "end", "n_cpg", "length_bp",
    "cpg_index_start", "cpg_index_end", "statistic", "statistic_mean",
    "statistic_min", "statistic_threshold", "region_support_median",
    "region_threshold_mode", "region_area", "region_qvalue", "region_tail_fdr",
    "region_qvalue_lower_95", "region_qvalue_upper_95",
    "region_qvalue_resolution", "null_exceedances_at_cutoff",
    "region_qvalue_cutoff_rank", "region_evidence_score",
    "evidence_score_extrapolated", "coordination_significant",
    "biological_class", "passes_prevalence", "significant", "fdr_status",
    "confirm_pvalue", "confirm_qvalue", "confirmed", "delta_hat", "f_hat",
    "direction", "n_samples_tested",
]


def main_region_table(regions: pd.DataFrame) -> pd.DataFrame:
    """Return the concise CLI-facing subset of the region table."""

    columns = [column for column in _REGION_MAIN_COLUMNS if column in regions]
    return regions.loc[:, columns].copy()
