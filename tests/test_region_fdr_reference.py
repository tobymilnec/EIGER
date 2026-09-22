"""The package's coordination FDR against the frozen validation implementation.

``fixtures/region_fdr_reference_v1.npz`` holds outputs of the frozen reference
(``eiger_fdr_validation_v2/src/vfdr`` importing EIGER-clean ``b11af46``) on
synthetic inputs regenerated here by ``region_fdr_data.make_dataset``. Provenance
and source hashes are in the companion JSON. The whole-matrix path must match
the reference's monolithic scan bit for bit, and the default chromosome path its
streamed scan bit for bit; both must make the same selections.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Python"))
sys.path.insert(0, str(HERE))

from region_fdr_data import make_dataset  # noqa: E402

from eiger2.coordination import (  # noqa: E402
    LEGACY,
    NEWER,
    RegionFdrConfig,
    call_coordinated_regions,
)

FIXTURE = np.load(HERE / "fixtures" / "region_fdr_reference_v1.npz")
INFO = json.loads((HERE / "fixtures" / "region_fdr_reference_v1.json").read_text())
CASES = sorted(INFO["cases"])


def _dataset(name):
    spec = INFO["cases"][name]["spec"]
    data = dict(spec["data"])
    if "chromosomes" in data:
        data["chromosomes"] = tuple((c, int(n)) for c, n in data["chromosomes"])
    betas, locations = make_dataset(**data)
    digest = hashlib.sha256(np.ascontiguousarray(betas).tobytes()).hexdigest()
    if digest != INFO["cases"][name]["input_sha256"]:
        pytest.skip(
            "the synthetic generator no longer reproduces the fixture input on "
            "this NumPy; regenerate the reference fixture"
        )
    return spec, betas, locations


def _run(spec, betas, locations, scan_mode):
    profile = NEWER if spec["profile"] == "newer" else LEGACY
    config = RegionFdrConfig(
        profile=profile,
        fdr_replicates=spec["B"],
        permutation_seed=spec["permutation_seed"],
        mask_mode=spec["mask_mode"],
        scan_mode=scan_mode,
        axis_mode="off",
    )
    return call_coordinated_regions(betas, locations, config=config)


def _ref(name, key):
    return FIXTURE[f"{name}__{key}"]


@pytest.mark.parametrize("name", CASES)
def test_whole_matrix_path_matches_frozen_monolithic_reference(name):
    spec, betas, locations = _dataset(name)
    run = _run(spec, betas, locations, "whole")
    regions = run.regions[["cpg_index_start", "cpg_index_end"]].to_numpy()
    np.testing.assert_array_equal(
        regions.reshape(-1, 2), _ref(name, "obs_regions").reshape(-1, 2)
    )
    np.testing.assert_array_equal(
        run.regions["region_score"].to_numpy(), _ref(name, "obs_areas")
    )
    assert run.manifest["observed_scan"]["scan_threshold"] == float(
        _ref(name, "obs_threshold")
    )
    if not len(regions):
        assert run.manifest["status"] == "no_candidates"
        return
    np.testing.assert_array_equal(
        run.permutations["scan_threshold"].to_numpy(), _ref(name, "null_thresholds")
    )
    np.testing.assert_array_equal(
        run.permutations["n_regions"].to_numpy(), _ref(name, "null_counts")
    )
    np.testing.assert_array_equal(
        np.concatenate(run.null_scores), _ref(name, "null_pooled")
    )
    q = run.regions["coordination_qvalue"].to_numpy()
    np.testing.assert_array_equal(q, _ref(name, "q_plus"))
    assert (
        int(run.regions["coordination_selected"].sum())
        == (INFO["cases"][name]["n_selected_at_0.05"])
    )


@pytest.mark.parametrize("name", CASES)
def test_default_chromosome_path_matches_streamed_reference_and_its_decisions(name):
    spec, betas, locations = _dataset(name)
    run = _run(spec, betas, locations, "chromosome")
    regions = (
        run.regions[["cpg_index_start", "cpg_index_end"]].to_numpy().reshape(-1, 2)
    )
    np.testing.assert_array_equal(regions, _ref(name, "obs_s_regions").reshape(-1, 2))
    np.testing.assert_array_equal(
        run.regions["region_score"].to_numpy(), _ref(name, "obs_s_areas")
    )
    assert run.manifest["observed_scan"]["scan_threshold"] == float(
        _ref(name, "obs_s_threshold")
    )
    # The streamed statistic differs from the monolithic one only by round-off.
    np.testing.assert_array_equal(regions, _ref(name, "obs_regions").reshape(-1, 2))
    np.testing.assert_allclose(
        run.regions["region_score"].to_numpy(),
        _ref(name, "obs_areas"),
        rtol=0,
        atol=1e-10,
    )
    if not len(regions):
        return
    q = run.regions["coordination_qvalue"].to_numpy()
    if f"{name}__q_s_plus" in FIXTURE.files:
        np.testing.assert_array_equal(q, _ref(name, "q_s_plus"))
        np.testing.assert_array_equal(
            np.concatenate(run.null_scores), _ref(name, "null_s_pooled")
        )
    np.testing.assert_array_equal(q <= 0.05, _ref(name, "q_plus") <= 0.05)
