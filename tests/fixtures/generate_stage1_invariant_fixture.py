"""Regenerate tests/fixtures/stage1_invariant_fixture_v200.csv.

Run ONLY against unmodified v2.0.0 code (git tag v2.0.0 / commit b8e4a19). Do not
rerun this after Stage 1 changes land -- the fixture is the pre-correction baseline
that tests/test_eiger2_core.py::test_stage1_preserves_the_invariant_column_set diffs
against.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Python"))

from eiger2 import Eiger2Config, call_dmrs
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_eiger2_core import locations  # reuse the shared synthetic-locations helper
from eiger2.benchmark import simulate

INVARIANT_COLUMNS = [
    "region_id", "chr", "start", "end", "n_cpg", "cpg_index_start", "cpg_index_end",
    "statistic", "statistic_mean", "statistic_min", "statistic_threshold",
    "region_area", "minority_size_descriptive", "minority_delta_descriptive",
    "n_samples_observed", "missing_rate", "mean_beta", "delta_beta_descriptive",
    "largest_gap_descriptive", "direction_descriptive",
    "minor_group_size_descriptive", "minor_group_fraction_descriptive",
    "minor_group_samples_descriptive", "pvalue_descriptive",
]


def generate_inputs():
    matrix, locs, _planted = simulate(
        n_samples=60, strata=[(0.15, 0.35)], replicates=4,
        spacing=300, dmr_length=12, missing_rate=0.02, seed=1701,
    )
    config = Eiger2Config(window=4, null_replicates=8, random_seed=1701)
    return matrix, locs, config


def main() -> None:
    matrix, locs, config = generate_inputs()
    regions, _values, _manifest = call_dmrs(matrix, locs, config=config)
    out = Path(__file__).parent / "stage1_invariant_fixture_v200.csv"
    regions[INVARIANT_COLUMNS].to_csv(out, index=False)
    print(f"wrote {len(regions)} rows to {out}")


if __name__ == "__main__":
    main()
