"""Shared manifest envelope required from Stage 1 onward (spec §0.5).

Every ``manifest.json`` written by ``eiger2 call``, ``eiger2 programs``, and the
benchmark harness must carry this envelope so a downstream run, or a reviewer, can
verify what produced a given output without re-deriving it from context.
"""

from __future__ import annotations

from . import __version__

REQUIRED_MANIFEST_KEYS = (
    "schema_version",
    "software_version",
    "git_commit",
    "git_worktree_dirty",
    "git_worktree_fingerprint_sha256",
    "parent_run_id",
    "parent_manifest_checksum",
    "input_checksums",
    "resolved_configuration",
    "random_seed",
    "derived_seeds",
    "runtime_seconds",
    "peak_memory_mb",
    "package_versions",
    "warnings",
    "fdr_decision",
)


def build_manifest_envelope(
    *,
    config_dict: dict,
    git_commit: str,
    input_checksums: dict,
    random_seed: int,
    derived_seeds: dict,
    runtime_seconds: float,
    peak_memory_mb: float,
    warnings: list[str],
    fdr_decision: str,
    git_worktree_dirty: bool | None = None,
    git_worktree_fingerprint_sha256: str | None = None,
    schema_version: str = "2.3",
    parent_run_id: str | None = None,
    parent_manifest_checksum: str | None = None,
) -> dict:
    import numpy as np
    import pandas as pd
    import scipy

    return {
        "schema_version": schema_version,
        "software_version": __version__,
        "git_commit": git_commit,
        "git_worktree_dirty": git_worktree_dirty,
        "git_worktree_fingerprint_sha256": git_worktree_fingerprint_sha256,
        "parent_run_id": parent_run_id,
        "parent_manifest_checksum": parent_manifest_checksum,
        "input_checksums": input_checksums,
        "resolved_configuration": config_dict,
        "random_seed": random_seed,
        "derived_seeds": derived_seeds,
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_memory_mb,
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "warnings": list(warnings),
        "fdr_decision": fdr_decision,
    }
