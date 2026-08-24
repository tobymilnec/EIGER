from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python"))

from eiger2.manifest_utils import REQUIRED_MANIFEST_KEYS, build_manifest_envelope


def test_build_manifest_envelope_has_every_required_key():
    envelope = build_manifest_envelope(
        config_dict={"target_fdr": 0.05},
        git_commit="deadbeef",
        input_checksums={"matrix.npy": "abc123"},
        random_seed=7,
        derived_seeds={"null_permutation": [1, 2, 3]},
        runtime_seconds=1.23,
        peak_memory_mb=42.0,
        warnings=[],
        fdr_decision="point",
        git_worktree_dirty=True,
        git_worktree_fingerprint_sha256="f" * 64,
    )
    for key in REQUIRED_MANIFEST_KEYS:
        assert key in envelope, key
    assert envelope["schema_version"] == "2.3"
    assert envelope["fdr_decision"] == "point"
    assert envelope["parent_run_id"] is None
    assert envelope["git_worktree_dirty"] is True
    assert envelope["git_worktree_fingerprint_sha256"] == "f" * 64


def test_build_manifest_envelope_records_parent_run_when_given():
    envelope = build_manifest_envelope(
        config_dict={}, git_commit="deadbeef", input_checksums={},
        random_seed=1, derived_seeds={}, runtime_seconds=0.1, peak_memory_mb=1.0,
        warnings=["a warning"], fdr_decision="upper",
        parent_run_id="run-123", parent_manifest_checksum="sha:xyz",
    )
    assert envelope["parent_run_id"] == "run-123"
    assert envelope["parent_manifest_checksum"] == "sha:xyz"
    assert envelope["warnings"] == ["a warning"]
