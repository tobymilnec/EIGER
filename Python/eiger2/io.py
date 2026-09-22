"""Efficient Bismark `.cov.gz` loading for EIGER 2.

Design follows wgbstools: pay the text-parsing cost **once**, then work from a
fixed CpG grid with memory-mapped binary files. After conversion, pulling a
chromosome slice for 500 samples is a memmap read, not a parse.

    step 1   build_cpg_reference(...)   union of CpG positions -> reference grid
    step 2   convert_cov_files(...)     one .npy per sample, uint16 [meth, total]
    step 3   load_slice(...)            memmap read of a CpG range, all samples

On disk each sample costs 4 bytes per reference CpG (~112 MB for 28M CpGs), so
500 samples is ~56 GB of cache — but you never hold more than one slice in RAM.

Bismark `.cov` columns are `chrom, start, end, percent_methylated, n_meth,
n_unmeth`, one row per **covered** CpG, and the files are not necessarily
chromosome-ordered. Both are handled.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "CpGReference",
    "build_cpg_reference",
    "convert_cov_files",
    "depth_covariance_diagnostic",
    "load_matrix",
    "load_slice",
]

MAIN_CHROMOSOMES = tuple(str(i) for i in range(1, 23)) + ("X",)
_COUNT_MAX = np.iinfo(np.uint16).max


def _normalise_chromosome(value: str) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    return "MT" if text.upper() in {"M", "MT"} else text


def _read_cov(path: Path, usecols=(0, 1, 4, 5)) -> pd.DataFrame:
    """Parse one Bismark cov/cov.gz with the C engine and fixed dtypes."""

    all_names = ["chrom", "pos", "end", "percent", "meth", "unmeth"]
    wanted = [all_names[i] for i in usecols]
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=all_names,
        usecols=wanted,
        dtype={
            "chrom": str,
            "pos": np.int64,
            "meth": np.int32,
            "unmeth": np.int32,
        },
        engine="c",
        compression="gzip" if str(path).endswith(".gz") else None,
        quoting=3,
    )
    frame["chrom"] = frame["chrom"].map(_normalise_chromosome)
    return frame


class CpGReference:
    """A sorted CpG grid shared by every sample in a cohort."""

    def __init__(self, chrom_codes, chrom_names, positions):
        self.chrom_names = list(chrom_names)
        self.chrom_codes = np.asarray(chrom_codes, dtype=np.int32)
        self.positions = np.asarray(positions, dtype=np.int64)
        # int64 sort key: chromosome in the high bits, position in the low bits
        self._keys = (self.chrom_codes.astype(np.int64) << 32) | self.positions

    def __len__(self) -> int:
        return int(self.positions.size)

    def keys_for(self, chrom: Sequence[str], pos: np.ndarray) -> np.ndarray:
        lookup = {name: index for index, name in enumerate(self.chrom_names)}
        codes = np.array([lookup.get(c, -1) for c in chrom], dtype=np.int64)
        return np.where(codes < 0, -1, (codes << 32) | np.asarray(pos, dtype=np.int64))

    def index_of(self, keys: np.ndarray) -> np.ndarray:
        """Row index of each key, or -1 when the key is not on the grid."""

        position = np.searchsorted(self._keys, keys)
        position = np.clip(position, 0, self._keys.size - 1)
        matched = self._keys[position] == keys
        return np.where(matched, position, -1)

    def chromosome_slices(self) -> list[tuple[str, int, int]]:
        boundaries = np.flatnonzero(np.diff(self.chrom_codes)) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(self)]
        return [
            (self.chrom_names[int(self.chrom_codes[s])], int(s), int(e))
            for s, e in zip(starts, ends, strict=True)
        ]

    def locations(self, start: int = 0, end: int | None = None) -> pd.DataFrame:
        end = len(self) if end is None else end
        codes = self.chrom_codes[start:end]
        names = np.array(self.chrom_names, dtype=object)[codes]
        pos = self.positions[start:end]
        return pd.DataFrame({"chr": names, "start": pos - 1, "end": pos})

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            chrom_codes=self.chrom_codes,
            positions=self.positions,
            chrom_names=np.array(self.chrom_names, dtype=object),
        )

    @classmethod
    def load(cls, path) -> CpGReference:
        data = np.load(path, allow_pickle=True)
        return cls(
            data["chrom_codes"], list(data["chrom_names"]), data["positions"]
        )


def build_cpg_reference(
    cov_files: Iterable[Path],
    *,
    chromosomes: Sequence[str] | None = MAIN_CHROMOSOMES,
    batch_size: int = 25,
    progress: bool = True,
) -> CpGReference:
    """Union of CpG positions across the cohort, sorted by chromosome then position.

    Bismark files only list *covered* CpGs, and coverage differs between samples,
    so the grid has to be the union. Positions are merged in batches so peak
    memory stays near one batch rather than the whole cohort.

    Pass a genome-wide CpG BED instead if you have one — it removes this pass and
    makes the grid identical across projects.
    """

    files = [Path(f) for f in cov_files]
    allowed = set(chromosomes) if chromosomes else None
    names: list[str] = sorted(allowed) if allowed else []
    lookup = {name: index for index, name in enumerate(names)}
    merged = np.empty(0, dtype=np.int64)

    for batch_start in range(0, len(files), batch_size):
        batch = files[batch_start : batch_start + batch_size]
        chunk_keys = []
        for path in batch:
            frame = _read_cov(path, usecols=(0, 1))
            if allowed is not None:
                frame = frame[frame["chrom"].isin(allowed)]
            elif not names:
                for chrom in frame["chrom"].unique():
                    if chrom not in lookup:
                        lookup[chrom] = len(names)
                        names.append(chrom)
            codes = frame["chrom"].map(lookup).to_numpy(dtype=np.int64)
            chunk_keys.append((codes << 32) | frame["pos"].to_numpy(np.int64))
            del frame
        merged = np.union1d(merged, np.unique(np.concatenate(chunk_keys)))
        del chunk_keys
        if progress:
            print(
                f"  reference: {min(batch_start + batch_size, len(files))}"
                f"/{len(files)} files, {merged.size:,} CpGs",
                flush=True,
            )
    return CpGReference(merged >> 32, names, merged & 0xFFFFFFFF)


def convert_cov_files(
    cov_files: Sequence[Path],
    reference: CpGReference,
    cache_dir: Path,
    *,
    sample_names: Sequence[str] | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> list[Path]:
    """Write one uint16 ``(n_cpg, 2)`` array per sample: [methylated, total].

    Counts are clipped at 65535, which is far above any real WGBS depth. Missing
    CpGs get total = 0, which is how downstream code recognises them.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for f in cov_files]
    if sample_names is None:
        sample_names = [f.name.split(".")[0] for f in files]
    outputs = []
    for index, (path, name) in enumerate(
        zip(files, sample_names, strict=True)
    ):
        target = cache_dir / f"{name}.counts.npy"
        outputs.append(target)
        if target.exists() and not overwrite:
            if progress:
                print(f"  [{index+1}/{len(files)}] {name}: cached", flush=True)
            continue
        frame = _read_cov(path)
        keys = reference.keys_for(frame["chrom"].to_numpy(), frame["pos"].to_numpy())
        rows = reference.index_of(keys)
        keep = rows >= 0
        counts = np.zeros((len(reference), 2), dtype=np.uint16)
        meth = np.clip(frame["meth"].to_numpy()[keep], 0, _COUNT_MAX)
        total = np.clip(
            (frame["meth"].to_numpy() + frame["unmeth"].to_numpy())[keep],
            0,
            _COUNT_MAX,
        )
        target_rows = rows[keep]
        # Bismark writes one row per CpG, so direct assignment is correct and
        # roughly 100x faster than np.add.at. Only fall back to the buffered
        # accumulate if the file really does contain duplicate positions (e.g.
        # both strands reported separately).
        if np.unique(target_rows).size == target_rows.size:
            counts[target_rows, 0] = meth.astype(np.uint16)
            counts[target_rows, 1] = total.astype(np.uint16)
        else:
            summed_meth = np.bincount(
                target_rows, weights=meth, minlength=len(reference)
            )
            summed_total = np.bincount(
                target_rows, weights=total, minlength=len(reference)
            )
            counts[:, 0] = np.clip(summed_meth, 0, _COUNT_MAX).astype(np.uint16)
            counts[:, 1] = np.clip(summed_total, 0, _COUNT_MAX).astype(np.uint16)
        np.save(target, counts)
        if progress:
            print(
                f"  [{index+1}/{len(files)}] {name}: {keep.sum():,} CpGs matched "
                f"({keep.mean():.1%} of rows)",
                flush=True,
            )
        del frame, counts
    manifest = {
        "n_cpg": len(reference),
        "samples": list(sample_names),
        "sources": [str(f) for f in files],
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return outputs


def load_slice(
    cache_files: Sequence[Path], start: int, end: int
) -> tuple[np.ndarray, np.ndarray]:
    """Memory-mapped read of CpGs ``[start, end)`` for every sample.

    Returns ``(beta, total)``. ``beta`` is float32 with NaN where total == 0.
    Nothing is parsed and only the requested slice is touched.
    """

    n_rows = end - start
    n_samples = len(cache_files)
    beta = np.full((n_rows, n_samples), np.nan, dtype=np.float32)
    total = np.zeros((n_rows, n_samples), dtype=np.uint16)
    for index, path in enumerate(cache_files):
        block = np.load(path, mmap_mode="r")[start:end]
        meth = block[:, 0].astype(np.float32)
        depth = block[:, 1]
        observed = depth > 0
        np.divide(meth, depth, out=beta[:, index], where=observed)
        beta[~observed, index] = np.nan
        total[:, index] = depth
    return beta, total


def load_matrix(
    cache_files: Sequence[Path],
    reference: CpGReference,
    *,
    chromosome: str | None = None,
    min_observed_samples: int = 2,
    min_depth: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load one chromosome (or everything) ready for ``call_coordinated_regions``.

    ``min_depth`` masks individual CpG/sample observations below a read-depth
    floor. Set it to ~10 if the shared-read diagnostic shows a depth trend.
    """

    if chromosome is None:
        start, end = 0, len(reference)
    else:
        matches = [s for s in reference.chromosome_slices() if s[0] == chromosome]
        if not matches:
            raise ValueError(f"chromosome {chromosome!r} is not in the reference")
        _, start, end = matches[0]

    beta, total = load_slice(cache_files, start, end)
    if min_depth > 0:
        beta[total < min_depth] = np.nan
    keep = np.sum(np.isfinite(beta), axis=1) >= min_observed_samples
    locations = reference.locations(start, end).loc[keep].reset_index(drop=True)
    return beta[keep], total[keep], locations


# --------------------------------------------------------------------------
# shared-read diagnostic
# --------------------------------------------------------------------------


def depth_covariance_diagnostic(
    beta: np.ndarray,
    total: np.ndarray,
    locations: pd.DataFrame,
    *,
    n_bins: int = 20,
    max_gap_bp: int = 300,
    min_paired_samples: int = 5,
) -> pd.DataFrame:
    """Median adjacent covariance and correlation as a function of read depth.

    A flat curve means shared reads are not inflating the statistic. A curve
    rising steeply towards low depth means they are: the excess scales as
    ``phi * r * p(1-p) / n``, which is a property of the sequencing, not the
    biology, and it is absent from EIGER's locality-breaking null because distant
    CpG pairs share no reads.
    """

    beta = np.asarray(beta, dtype=np.float64)
    left, right = beta[:-1], beta[1:]
    mask = np.isfinite(left) & np.isfinite(right)
    counts = mask.sum(axis=1)

    starts = locations["start"].to_numpy(np.int64)
    same_chrom = locations["chr"].to_numpy()[:-1] == locations["chr"].to_numpy()[1:]
    gaps = starts[1:] - starts[:-1]
    eligible = same_chrom & (gaps > 0) & (gaps <= max_gap_bp)
    eligible &= counts >= min_paired_samples

    left_z = np.where(mask, left, 0.0)
    right_z = np.where(mask, right, 0.0)
    safe = np.where(counts > 0, counts, 1).astype(float)
    sum_l, sum_r = left_z.sum(1), right_z.sum(1)
    denom = np.where(counts > 1, counts - 1, np.nan)
    cov = ((left_z * right_z).sum(1) - sum_l * sum_r / safe) / denom
    var_l = ((left_z**2).sum(1) - sum_l**2 / safe) / denom
    var_r = ((right_z**2).sum(1) - sum_r**2 / safe) / denom
    scale = np.sqrt(var_l * var_r)
    corr = np.where(scale > 0, cov / scale, np.nan)

    depth_left = np.where(total[:-1] > 0, total[:-1], np.nan).astype(float)
    depth_right = np.where(total[1:] > 0, total[1:], np.nan).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        harmonic = 2.0 / (
            1.0 / np.nanmean(depth_left, axis=1) + 1.0 / np.nanmean(depth_right, axis=1)
        )

    use = eligible & np.isfinite(cov) & np.isfinite(harmonic)
    if use.sum() < n_bins * 10:
        raise ValueError("too few eligible adjacent pairs for the diagnostic")

    quantiles = np.quantile(harmonic[use], np.linspace(0, 1, n_bins + 1))
    index = np.clip(np.digitize(harmonic, quantiles[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = use & (index == b)
        if sel.sum() < 10:
            continue
        rows.append(
            {
                "depth_bin": b,
                "mean_depth": float(np.mean(harmonic[sel])),
                "n_edges": int(sel.sum()),
                "median_cov": float(np.median(cov[sel])),
                "q90_cov": float(np.quantile(cov[sel], 0.90)),
                "median_corr": float(np.nanmedian(corr[sel])),
                "median_paired_samples": float(np.median(counts[sel])),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) > 2:
        slope, intercept = np.polyfit(
            1.0 / frame["mean_depth"], frame["median_cov"], 1
        )
        frame.attrs["shared_read_slope"] = float(slope)
        frame.attrs["depth_free_intercept"] = float(intercept)
    return frame
