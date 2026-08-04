#!/usr/bin/env python3
"""Repair EIGER DMR coordinates produced from wgbstools CpG references."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover genomic DMR endpoints from zero-based cpg_start/cpg_end "
            "row indices while preserving stable dmr_id values."
        )
    )
    parser.add_argument("--cpg-reference", required=True)
    parser.add_argument("--dmrs", nargs="+", required=True)
    parser.add_argument(
        "--backup-suffix",
        default=".pre_coordinate_fix",
        help="Suffix inserted before the CSV extension for original backups",
    )
    return parser


def _read_inputs(paths: list[Path]) -> tuple[list[pd.DataFrame], set[int]]:
    frames = []
    indices: set[int] = set()
    required = {"dmr_id", "chr", "start", "end", "cpg_start", "cpg_end"}
    for path in paths:
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{path} lacks required columns: {sorted(missing)}")
        for column in ("cpg_start", "cpg_end"):
            values = pd.to_numeric(frame[column], errors="raise").astype("int64")
            if (values < 0).any():
                raise ValueError(f"{path} contains negative {column} values")
            frame[column] = values
            indices.update(values.tolist())
        frames.append(frame)
    return frames, indices


def _reference_positions(
    path: Path,
    requested_indices: set[int],
) -> dict[int, tuple[str, int]]:
    if not requested_indices:
        return {}
    maximum = max(requested_indices)
    positions: dict[int, tuple[str, int]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for row_index, line in enumerate(handle):
            if row_index > maximum:
                break
            if row_index not in requested_indices:
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(
                    f"Malformed CpG reference row {row_index + 1}: {line!r}"
                )
            positions[row_index] = (str(fields[0]), int(fields[1]))
    missing = requested_indices.difference(positions)
    if missing:
        examples = sorted(missing)[:5]
        raise IndexError(
            "CpG indices exceed the supplied reference; examples: "
            f"{examples}"
        )
    return positions


def _backup_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _repair_frame(
    frame: pd.DataFrame,
    positions: dict[int, tuple[str, int]],
    source: Path,
) -> pd.DataFrame:
    repaired = frame.copy()
    starts = repaired["cpg_start"].map(lambda value: positions[int(value)])
    ends = repaired["cpg_end"].map(lambda value: positions[int(value)])
    start_chrom = starts.map(lambda value: value[0])
    end_chrom = ends.map(lambda value: value[0])
    output_chrom = repaired["chr"].astype(str)
    cross_chromosome = (start_chrom != end_chrom) | (start_chrom != output_chrom)
    if cross_chromosome.any():
        examples = repaired.loc[cross_chromosome, "dmr_id"].head(5).tolist()
        raise ValueError(
            f"{source} contains DMRs inconsistent with the CpG reference: "
            f"{examples}"
        )

    genomic_starts = starts.map(lambda value: value[1]).astype("int64")
    genomic_ends = (ends.map(lambda value: value[1]) + 1).astype("int64")
    reported_starts = pd.to_numeric(
        repaired["start"], errors="raise"
    ).astype("int64")
    if not reported_starts.equals(genomic_starts):
        mismatch = repaired.loc[
            reported_starts.ne(genomic_starts), "dmr_id"
        ].head(5).tolist()
        raise ValueError(
            f"{source} does not match the supplied CpG reference; "
            f"start-coordinate examples: {mismatch}"
        )
    if (genomic_ends <= genomic_starts).any():
        raise ValueError(f"{source} produced non-positive repaired intervals")

    repaired["start"] = genomic_starts
    repaired["end"] = genomic_ends
    repaired["length_bp"] = genomic_ends - genomic_starts
    return repaired


def _update_manifest(path: Path, reference: Path, backup: Path) -> None:
    manifest_path = path.with_name(f"{path.stem}_run.json")
    if not manifest_path.exists():
        return
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    manifest["coordinate_repair"] = {
        "applied": True,
        "method": "cpg_row_index_lookup",
        "reference": str(reference.resolve()),
        "interval_convention": "0-based_half-open",
        "dmr_ids_preserved": True,
        "original_csv": str(backup.resolve()),
    }
    temporary = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    with open(temporary, "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    temporary.replace(manifest_path)


def main() -> None:
    args = build_parser().parse_args()
    paths = [Path(path).resolve() for path in args.dmrs]
    reference = Path(args.cpg_reference).resolve()
    frames, indices = _read_inputs(paths)
    print(
        f"Reading {len(indices):,} required CpG positions from {reference}...",
        flush=True,
    )
    positions = _reference_positions(reference, indices)

    for path, frame in zip(paths, frames):
        repaired = _repair_frame(frame, positions, path)
        backup = _backup_path(path, args.backup_suffix)
        if not backup.exists():
            shutil.copy2(path, backup)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        repaired.to_csv(temporary, index=False)
        temporary.replace(path)
        _update_manifest(path, reference, backup)
        print(
            f"Repaired {len(repaired):,} DMRs in {path}; "
            f"original retained as {backup.name}"
        )


if __name__ == "__main__":
    main()
