"""EIGER -- label-free discovery of regionally coordinated DNA methylation.

The supported workflow is :func:`call_coordinated_regions` (``eiger2 call`` on
the command line): scan the CpG-by-sample matrix for runs of coordinated
adjacent CpGs, rescan ``B`` complete within-CpG permutations of the matrix, and
report each candidate region's coordination q-value. Axis effects are
secondary, descriptive annotations of the same candidates and never change a
primary call. See :mod:`eiger2.coordination` and :mod:`eiger2.axis`.
"""

__version__ = "2.2.0"

from .axis import annotate_axes
from .coordination import (
    LEGACY,
    NEWER,
    CallerProfile,
    CandidateThresholdError,
    PermutationScanError,
    RegionFdrConfig,
    RegionFdrRun,
    call_coordinated_regions,
    coordination_fdr,
)

__all__ = [
    "__version__",
    "CallerProfile",
    "CandidateThresholdError",
    "LEGACY",
    "NEWER",
    "PermutationScanError",
    "RegionFdrConfig",
    "RegionFdrRun",
    "annotate_axes",
    "call_coordinated_regions",
    "coordination_fdr",
]
