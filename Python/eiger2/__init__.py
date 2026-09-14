"""EIGER 2 — stripped-back co-variation DMR discovery.

Discovery is: centre CpGs, remove genome-wide latent structure estimated off a
thinned CpG set, compute a local co-variation statistic, segment it against a
locality-breaking null, and assign a region-level FDR by pushing null tracks
through the identical segmentation.

Everything about *which* samples differ and *by how much* is post-selection
characterisation and is kept strictly downstream of discovery.
"""

__version__ = "2.3.0"

from .confirm import confirm_regions, region_sample_means, two_component_fit
from .core import (
    Eiger2Config,
    RegionFdrResult,
    biological_support_diagnostics,
    call_dmrs,
    centre_cpgs,
    characterise_region,
    classify_biological_support,
    deflate,
    edge_statistic,
    estimate_latent_basis,
    main_region_table,
    null_threshold,
    region_fdr,
    region_fdr_result,
    segment,
)
from .probability import (
    LocalFdrResult,
    calibration_error,
    estimate_pi0,
    local_fdr,
    reliability_table,
)

__all__ = [
    "__version__",
    "Eiger2Config",
    "LocalFdrResult",
    "RegionFdrResult",
    "calibration_error",
    "biological_support_diagnostics",
    "call_dmrs",
    "centre_cpgs",
    "confirm_regions",
    "characterise_region",
    "deflate",
    "edge_statistic",
    "estimate_latent_basis",
    "classify_biological_support",
    "main_region_table",
    "estimate_pi0",
    "local_fdr",
    "null_threshold",
    "region_fdr",
    "region_fdr_result",
    "reliability_table",
    "region_sample_means",
    "segment",
    "two_component_fit",
]
