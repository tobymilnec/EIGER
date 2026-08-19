"""Regenerate tests/fixtures/stage1b_invariant_fixture_v201.npz.

Per Stage 1b's Global Constraints, the invariant-preservation contract this
plan leaves Stage 1 with is NARROWER than the one
``stage1_invariant_fixture_v200.csv`` pinned: only ``edge_statistic()``'s raw
output and ``eligible_edges()``'s raw output stay byte-identical from
v2.0.0/v2.0.1 through Stage 1b. Region boundaries, ``region_area``, and
``statistic_threshold`` all change by design -- the lag-null fix (Task 2 of
this stage, commit 3ca07ef) corrects the null construction, which changes the
threshold, which changes segmentation, which cascades into every
region-table column. This fixture pins the part of the pipeline that is
still meant to be invariant.

Reuses ``generate_stage1_invariant_fixture.py``'s ``generate_inputs()`` --
the SAME simulated matrix/locations/config the old v2.0.0 fixture was built
from -- so the two fixtures are directly comparable (same input, different
slice of the pipeline pinned).

Captures the raw arrays via a temporary monkeypatch of
``eiger2.core.edge_statistic``/``eiger2.core.eligible_edges`` around a real
``call_dmrs`` run, rather than hand-reproducing call_dmrs's internal
pipeline (centre_cpgs -> estimate_latent_basis -> deflate -> edge_statistic /
eligible_edges) here. This guarantees the captured arrays are exactly what
the end-to-end pipeline computes -- including call_dmrs's low-coverage
keep-filter on the input matrix/locations -- not a parallel reproduction
that could silently diverge from it.

Confirmed before this fixture was generated: neither ``edge_statistic`` nor
``eligible_edges`` appears in Task 2's (3ca07ef) or Task 3's (898968d) diff
of Python/eiger2/core.py, and a direct text diff of both functions' bodies
between commit 306fa8f (the pre-Stage-1 commit the old v2.0.0 fixture was
generated from) and the current worktree is empty. The current worktree
state is therefore the correct v2.0.1-equivalent baseline to pin -- see
task-6-report.md for the full investigation.

Run ONLY when ``edge_statistic``/``eligible_edges`` are believed to still be
unchanged since this fixture was generated (i.e. essentially never again
without a deliberate decision to re-baseline -- if either function's
behaviour is ever intentionally changed, this fixture must be regenerated
and the change called out explicitly, not silently absorbed).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Python"))

import numpy as np  # noqa: E402

import eiger2.core as core_mod  # noqa: E402
from eiger2 import Eiger2Config, call_dmrs  # noqa: E402, F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_stage1_invariant_fixture import generate_inputs  # noqa: E402


def main() -> None:
    matrix, locs, config = generate_inputs()

    captured: dict[str, np.ndarray] = {}
    original_edge_statistic = core_mod.edge_statistic
    original_eligible_edges = core_mod.eligible_edges

    def capturing_edge_statistic(residuals, cfg, chunk=200_000):
        result = original_edge_statistic(residuals, cfg, chunk=chunk)
        captured["edge_statistic"] = result
        return result

    def capturing_eligible_edges(locations, cfg):
        result = original_eligible_edges(locations, cfg)
        captured["eligible_edges"] = result
        return result

    core_mod.edge_statistic = capturing_edge_statistic
    core_mod.eligible_edges = capturing_eligible_edges
    try:
        core_mod.call_dmrs(matrix, locs, config=config)
    finally:
        core_mod.edge_statistic = original_edge_statistic
        core_mod.eligible_edges = original_eligible_edges

    assert "edge_statistic" in captured and "eligible_edges" in captured, (
        "call_dmrs did not reach edge_statistic/eligible_edges -- "
        "generate_inputs()'s config/matrix must have changed upstream"
    )

    out = Path(__file__).parent / "stage1b_invariant_fixture_v201.npz"
    np.savez(
        out,
        edge_statistic=captured["edge_statistic"],
        eligible_edges=captured["eligible_edges"],
    )
    print(
        f"wrote edge_statistic shape={captured['edge_statistic'].shape} "
        f"eligible_edges shape={captured['eligible_edges'].shape} to {out}"
    )


if __name__ == "__main__":
    main()
