#!/usr/bin/env python3
"""Build the real-hg38 WGBS layout fixture for the EIGER benchmark.

Layout and coverage only. The methylated channel of the ``.beta`` is discarded and
never written, so the fixture contains no methylome -- only where CpGs are and how
deeply they were sequenced.

Both inputs are public:

  CpG.bed.gz   wgbstools hg38 CpG index, 29,401,795 rows, 1-based.
               Row i of this file is row i of any hg38 .beta file.
  *.beta       uint8 pairs (methylated, total) per CpG index row.
               GSM5652215 (Loyfer et al. cell-type atlas) was used for the
               committed fixture.

Usage:
    python extract_wgbs_layout.py --cpg-index CpG.bed.gz --beta SAMPLE.hg38.beta \
        --chroms chr1 chr7 chr19 --out wgbs_layout_hg38.npz

Output npz arrays:
    chrom_names   (n_chrom,)  str
    chrom_start   (n_chrom,)  int64  index into pos_delta/depth where each chrom begins
    chrom_first   (n_chrom,)  int64  first CpG position on that chromosome
    pos_delta     (n_cpg,)    int32  gap to previous CpG; first of each chrom is 0
    depth         (n_cpg,)    uint8  total reads, uint8-saturated at 255 by wgbstools

Positions are delta-encoded because consecutive gaps are small, which compresses to a
fraction of the raw int32 coordinates.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

BETA_ROW_BYTES = 2


def _checksum(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()[:16]


def read_index(path: Path, wanted: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {chrom: (positions, zero_based_row_ids)} for the requested chromosomes."""

    pos: dict[str, list[int]] = {c: [] for c in wanted}
    rid: dict[str, list[int]] = {c: [] for c in wanted}
    seen: set[str] = set()
    with gzip.open(path, "rt") as fh:
        for line in fh:
            chrom, position, index = line.split("\t")
            if chrom in wanted:
                pos[chrom].append(int(position))
                rid[chrom].append(int(index) - 1)  # index is 1-based
                seen.add(chrom)
            elif seen == wanted and chrom not in wanted:
                # the index is in genomic order; once every wanted chromosome has been
                # seen and we are past them all, there is nothing left to find
                if len(seen) == len(wanted):
                    break
    missing = wanted - seen
    if missing:
        raise SystemExit(f"chromosomes not present in the index: {sorted(missing)}")
    return {c: (np.asarray(pos[c], dtype=np.int64), np.asarray(rid[c], dtype=np.int64))
            for c in wanted}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpg-index", required=True, type=Path)
    ap.add_argument("--beta", required=True, type=Path)
    ap.add_argument("--chroms", nargs="+", default=["chr1", "chr7", "chr19"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--accession", default="GSM5652215",
                    help="recorded in the provenance sidecar")
    args = ap.parse_args()

    wanted = set(args.chroms)
    index = read_index(args.cpg_index, wanted)

    beta = np.fromfile(args.beta, dtype=np.uint8)
    if beta.size % BETA_ROW_BYTES:
        raise SystemExit("beta file length is not a multiple of 2 bytes")
    beta = beta.reshape(-1, BETA_ROW_BYTES)
    n_index = sum(len(index[c][0]) for c in wanted)
    if beta.shape[0] < max(index[c][1].max() for c in wanted) + 1:
        raise SystemExit(
            f"beta has {beta.shape[0]:,} rows but the index references higher rows; "
            "the two are not the same genome build"
        )
    # column 0 is the methylated count. It is never read.
    total = beta[:, 1]
    del beta

    names, starts, firsts, deltas, depths = [], [], [], [], []
    cursor = 0
    for chrom in args.chroms:
        positions, rows = index[chrom]
        order = np.argsort(positions, kind="mergesort")
        positions, rows = positions[order], rows[order]
        delta = np.diff(positions, prepend=positions[0]).astype(np.int32)
        names.append(chrom)
        starts.append(cursor)
        firsts.append(int(positions[0]))
        deltas.append(delta)
        depths.append(total[rows])
        cursor += positions.size

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        chrom_names=np.asarray(names),
        chrom_start=np.asarray(starts, dtype=np.int64),
        chrom_first=np.asarray(firsts, dtype=np.int64),
        pos_delta=np.concatenate(deltas),
        depth=np.concatenate(depths),
    )

    provenance = {
        "source_accession": args.accession,
        "beta_file": args.beta.name,
        "beta_sha256_head16": _checksum(args.beta),
        "cpg_index_file": args.cpg_index.name,
        "cpg_index_sha256_head16": _checksum(args.cpg_index),
        "cpg_index_rows": 29_401_795,
        "chromosomes": args.chroms,
        "n_cpg": int(cursor),
        "contains": "CpG position and total read depth only; "
                    "the methylated channel is discarded and never written",
        "fixture_bytes": out.stat().st_size,
    }
    out.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB) for {cursor:,} CpGs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
