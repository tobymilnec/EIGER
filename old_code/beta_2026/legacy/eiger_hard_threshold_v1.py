"""Runnable snapshot profile for EIGER's pre-probability calling behaviour.

This profile preserves the original adjacent-covariance hard threshold,
largest-gap sample partition, delta-only call status, and historical Laplacian
percentile exposed as ``stability_score``. Regression tests lock the behaviour.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

try:
    from ..eiger_core import EigerConfig, call_dmrs_from_matrix as _call
except ImportError:  # direct import from the Python directory
    from eiger_core import EigerConfig, call_dmrs_from_matrix as _call


LEGACY_ALGORITHM_VERSION = "eiger-hard-threshold-v1"


def legacy_config(config: Optional[EigerConfig] = None) -> EigerConfig:
    """Return a configuration pinned to the preserved legacy caller."""

    return replace(
        config or EigerConfig(),
        calling_method="legacy",
        stability_bootstraps=0,
    )


def call_dmrs_from_matrix(*args, config: Optional[EigerConfig] = None, **kwargs):
    """Call DMRs with the preserved hard-threshold algorithm."""

    return _call(*args, config=legacy_config(config), **kwargs)
