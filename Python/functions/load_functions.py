#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26 May 2023
@author: Toby Milne-Clark
"""

import os.path as op
import os
import gzip
import pathlib
import warnings
from pathlib import Path
from dataclasses import dataclass
from subprocess import call
from functools import partial

import numpy as np
import numpy.typing as npt
import pandas as pd
import typing
from typing import (
    Literal
)

Read = npt.NDArray
MetFreqArr = npt.NDArray
ListFiles = list[str|Path]


def infer_methylation_file_type(files: ListFiles) -> Literal['beta', 'cov']:
    """Infer one consistent input type from beta/lbeta or Bismark cov files."""
    if not files:
        raise ValueError("At least one methylation file is required")

    detected = set()
    for file_path in files:
        name = str(file_path).lower()
        if name.endswith(('.beta', '.lbeta')):
            detected.add('beta')
        elif name.endswith(('.cov', '.cov.gz')):
            detected.add('cov')
        else:
            raise ValueError(
                f"Cannot infer methylation input type from: {file_path}. "
                "Expected .beta, .lbeta, .cov, or .cov.gz."
            )

    if len(detected) != 1:
        raise ValueError(
            "All methylation files in one EIGER run must have the same input type"
        )
    return detected.pop()


def _normalise_chromosome(value) -> str:
    chromosome = str(value).strip()
    if chromosome.lower().startswith('chr'):
        chromosome = chromosome[3:]
    if chromosome.upper() in {'M', 'MT'}:
        return 'MT'
    return chromosome


def _reference_chromosome_index(locations):
    if locations.shape[1] < 2:
        raise ValueError("The CpG location file must contain chromosome and position")

    chromosomes = locations.iloc[:, 0].astype(str).to_numpy()
    positions = locations.iloc[:, 1].to_numpy(dtype=np.int64, copy=False)
    if len(positions) == 0:
        raise ValueError("The CpG location file is empty")

    boundaries = np.empty(len(chromosomes), dtype=bool)
    boundaries[0] = True
    boundaries[1:] = chromosomes[1:] != chromosomes[:-1]
    starts = np.flatnonzero(boundaries)
    ends = np.r_[starts[1:], len(chromosomes)]

    reference = {}
    for start, end in zip(starts, ends):
        chromosome = _normalise_chromosome(chromosomes[start])
        if chromosome in reference:
            raise ValueError(
                f"CpG locations for chromosome {chromosome!r} are not contiguous"
            )
        chromosome_positions = positions[start:end]
        if np.any(chromosome_positions[1:] < chromosome_positions[:-1]):
            raise ValueError(
                f"CpG locations are not sorted within chromosome {chromosome!r}"
            )
        reference[chromosome] = (int(start), chromosome_positions)
    return reference


def _matching_reference_rows(chunk, reference, coordinate_offsets):
    """Yield reference rows, trying strand-aware coordinate offsets in order."""
    if isinstance(coordinate_offsets, int):
        coordinate_offsets = (coordinate_offsets,)

    for raw_chromosome, rows in chunk.groupby('chromosome', sort=False):
        chromosome = _normalise_chromosome(raw_chromosome)
        chromosome_reference = reference.get(chromosome)
        if chromosome_reference is None:
            continue

        reference_start, reference_positions = chromosome_reference
        source_indices = rows.index.to_numpy()
        positions = rows['position'].to_numpy(dtype=np.int64, copy=False)
        unmatched = np.ones(len(rows), dtype=bool)
        for coordinate_offset in coordinate_offsets:
            candidates = np.flatnonzero(unmatched)
            if len(candidates) == 0:
                break
            query_positions = positions[candidates] + coordinate_offset
            local_indices = np.searchsorted(reference_positions, query_positions)
            in_bounds = local_indices < len(reference_positions)
            matched = np.zeros(len(local_indices), dtype=bool)
            matched[in_bounds] = (
                reference_positions[local_indices[in_bounds]]
                == query_positions[in_bounds]
            )
            if not np.any(matched):
                continue
            matched_candidates = candidates[matched]
            yield (
                source_indices[matched_candidates],
                reference_start + local_indices[matched],
            )
            unmatched[matched_candidates] = False


def _detect_cov_coordinate_offsets(chunk, reference) -> tuple[tuple[int, ...], str]:
    scores = {}
    coordinate_options = (
        ((0, -1), 'one-based'),
        ((1, 0), 'zero-based'),
    )
    for offsets, label in coordinate_options:
        scores[(offsets, label)] = sum(
            len(reference_rows)
            for _, reference_rows in _matching_reference_rows(
                chunk, reference, offsets
            )
        )
    # Standard Bismark coverage is one-based, so it wins an exact tie.
    return max(scores, key=lambda item: (scores[item], item[1] == 'one-based'))


def load_bismark_coverage_betas(
        files: ListFiles,
        locations,
        coordinate_system: Literal['auto', 'one-based', 'zero-based'] = 'auto',
        chunk_size: int = 1_000_000,
        minimum_match_rate: float = 0.5,
        progress: bool = False,
        return_total_counts: bool = False,
):
    """Load Bismark .cov/.cov.gz files against an EIGER CpG reference.

    Coverage rows are streamed in chunks and aligned by chromosome and position.
    The returned matrix has shape (CpG sites, samples), with unobserved sites as
    NaN. Bismark methylated and unmethylated read counts are combined if more
    than one row maps to the same CpG.
    """
    if not files:
        raise ValueError("At least one Bismark coverage file is required")
    if coordinate_system not in {'auto', 'one-based', 'zero-based'}:
        raise ValueError(
            "coordinate_system must be 'auto', 'one-based', or 'zero-based'"
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not 0 <= minimum_match_rate <= 1:
        raise ValueError("minimum_match_rate must be between 0 and 1")

    reference = _reference_chromosome_index(locations)
    betas = np.full(
        (len(locations), len(files)),
        np.nan,
        dtype=float,
    )
    coverage = (
        np.zeros((len(locations), len(files)), dtype=np.uint32)
        if return_total_counts
        else None
    )
    summaries = []
    column_names = [
        'chromosome',
        'position',
        'end',
        'percent_methylated',
        'methylated',
        'unmethylated',
    ]

    for sample_index, file_path in enumerate(files):
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Coverage file not found: {file_path}")
        if progress:
            print(
                f"  [{sample_index + 1}/{len(files)}] {file_path.name}",
                flush=True,
            )

        methylated_counts = np.zeros(len(locations), dtype=np.uint32)
        total_counts = np.zeros(len(locations), dtype=np.uint32)
        total_rows = 0
        matched_rows = 0
        chosen_offsets = None
        chosen_system = coordinate_system

        try:
            chunks = pd.read_csv(
                file_path,
                sep='\t',
                header=None,
                names=column_names,
                usecols=[0, 1, 4, 5],
                dtype={
                    'chromosome': 'string',
                    'position': np.int64,
                    'methylated': np.float64,
                    'unmethylated': np.float64,
                },
                comment='#',
                chunksize=chunk_size,
            )
            for chunk in chunks:
                if chunk.empty:
                    continue
                chunk.reset_index(drop=True, inplace=True)
                if chosen_offsets is None:
                    if coordinate_system == 'auto':
                        chosen_offsets, chosen_system = (
                            _detect_cov_coordinate_offsets(chunk, reference)
                        )
                    else:
                        chosen_offsets = (
                            (0, -1)
                            if coordinate_system == 'one-based'
                            else (1, 0)
                        )

                methylated = chunk['methylated'].to_numpy(dtype=float, copy=False)
                unmethylated = chunk['unmethylated'].to_numpy(
                    dtype=float, copy=False
                )
                if (
                    np.any(~np.isfinite(methylated))
                    or np.any(~np.isfinite(unmethylated))
                    or np.any(methylated < 0)
                    or np.any(unmethylated < 0)
                    or np.any(methylated != np.floor(methylated))
                    or np.any(unmethylated != np.floor(unmethylated))
                ):
                    raise ValueError(
                        f"Coverage counts must be finite non-negative integers: "
                        f"{file_path}"
                    )
                total = methylated + unmethylated
                if np.any(total > np.iinfo(np.uint32).max):
                    raise ValueError(f"Coverage count exceeds uint32: {file_path}")

                total_rows += len(chunk)
                for source_rows, reference_rows in _matching_reference_rows(
                    chunk, reference, chosen_offsets
                ):
                    np.add.at(
                        methylated_counts,
                        reference_rows,
                        methylated[source_rows].astype(np.uint32),
                    )
                    np.add.at(
                        total_counts,
                        reference_rows,
                        total[source_rows].astype(np.uint32),
                    )
                    matched_rows += len(reference_rows)
        except (OSError, pd.errors.ParserError, ValueError) as error:
            raise ValueError(
                f"Could not load Bismark coverage file {file_path}: {error}"
            ) from error

        if total_rows == 0:
            raise ValueError(f"Coverage file is empty: {file_path}")
        match_rate = matched_rows / total_rows
        if match_rate < minimum_match_rate:
            raise ValueError(
                f"Only {match_rate:.1%} of rows in {file_path.name} matched the "
                f"CpG reference (minimum {minimum_match_rate:.1%}). Check the "
                "genome build and coordinate system."
            )
        if match_rate < 0.95:
            warnings.warn(
                f"{file_path.name}: {match_rate:.1%} of coverage rows matched "
                "the CpG reference",
                RuntimeWarning,
            )

        observed = total_counts > 0
        betas[observed, sample_index] = (
            methylated_counts[observed] / total_counts[observed]
        )
        if coverage is not None:
            coverage[:, sample_index] = total_counts
        summaries.append(
            {
                'file': str(file_path.resolve()),
                'coordinate_system': chosen_system,
                'rows': int(total_rows),
                'matched_rows': int(matched_rows),
                'match_rate': float(match_rate),
                'observed_cpgs': int(np.count_nonzero(observed)),
            }
        )
        if progress:
            print(
                f"      {match_rate:.1%} rows matched; "
                f"{np.count_nonzero(observed):,} CpGs observed "
                f"({chosen_system})",
                flush=True,
            )

    if coverage is not None:
        return betas, summaries, coverage
    return betas, summaries

def load_beta_frequencies(
        fn, slice_start:int=None, slice_end:int=None
) -> MetFreqArr:
    """Load wgbstools .beta file.
    Slices specify CpG sites to load. Standard python indexing.
    None means start/end of file.
    Returns a np.ndarray[uint8] with shape (NSITES, 2).
    Columns give met freq, and freq site observed."""
    suff = op.splitext(fn)
    if suff[1] == '.lbeta':
        sizet = 2
        dtype = np.uint16
    else:
        sizet = 1
        dtype = np.uint8
    i, j = slice_start, slice_end
    if (i is not None) or (j is not None):
        with open(fn, 'rb') as f:
            if i is None:
                i = 0
            f.seek(i * 2 * sizet)
            if j is None:
                arr = np.fromfile(f, dtype=dtype)
            else:
                arr = np.fromfile(
                    f, dtype=dtype,
                    count=((j - i) * 2)
                )
    else:
        arr = np.fromfile(fn, dtype=dtype)
    return arr.reshape((-1, 2))


def beta_site_count(fn: str | Path) -> int:
    """Return the number of CpG rows encoded in a beta or lbeta file."""
    path = Path(fn)
    if not path.is_file():
        raise FileNotFoundError(f"Beta file not found: {path}")
    bytes_per_site = 4 if path.name.lower().endswith('.lbeta') else 2
    file_size = path.stat().st_size
    if file_size % bytes_per_site:
        raise ValueError(f"Invalid beta file size: {path}")
    return file_size // bytes_per_site


def load_beta_values(
        files: ListFiles,
        slice_start: int = 0,
        slice_end: int | None = None,
        dtype=np.float32,
        return_total_counts: bool = False,
):
    """Load a CpG slice directly as a beta-value matrix.

    This avoids constructing the much larger site-by-sample-by-count array used
    by the legacy whole-genome loader.
    """
    if not files:
        raise ValueError("At least one beta file is required")
    total_sites = beta_site_count(files[0])
    slice_end = total_sites if slice_end is None else int(slice_end)
    slice_start = int(slice_start)
    if not 0 <= slice_start <= slice_end <= total_sites:
        raise ValueError("Invalid beta slice boundaries")

    values = np.full(
        (slice_end - slice_start, len(files)),
        np.nan,
        dtype=dtype,
    )
    coverage = (
        np.zeros((slice_end - slice_start, len(files)), dtype=np.uint32)
        if return_total_counts
        else None
    )
    for sample_index, file_path in enumerate(files):
        sample_sites = beta_site_count(file_path)
        if sample_sites != total_sites:
            raise ValueError(
                f"Beta files have inconsistent CpG counts: {files[0]} has "
                f"{total_sites:,}, but {file_path} has {sample_sites:,}"
            )
        counts = load_beta_frequencies(
            file_path,
            slice_start=slice_start,
            slice_end=slice_end,
        )
        observed = counts[:, 1] > 0
        if coverage is not None:
            coverage[:, sample_index] = counts[:, 1]
        np.divide(
            counts[:, 0],
            counts[:, 1],
            out=values[:, sample_index],
            where=observed,
        )
    if coverage is not None:
        return values, coverage
    return values


def load_beta_frequencies_from_bed(
        fn, dotbeta_dir='.', bedgraph:bool= 'infer',
        rm_dotbeta=False, force_overwrite=False, *args, **kwargs
) -> MetFreqArr:
    """Load a bed file and convert to .beta file.
    Supports Bismark style bedgraph ".cov". Is inferred from filename
    or set coveragefile=False for .bed, True for .bedgraph|.cov
    """
    fn = str(fn)
    if bedgraph == 'infer':
        if fn.endswith('.bedgraph') or fn.endswith('.cov'):
            bedgraph = True
        elif fn.endswith('.bed'):
            bedgraph = False
        else:
            raise RuntimeError("Can't infer file type, specify `coveragefile=False|True`")
    opts = []
    if bedgraph:
        opts.append('--bedgraph')
    if force_overwrite:
        opts.append('-f')
    opts = ' '.join(opts)
    call(f'wgbstools bed2beta  -o {dotbeta_dir} {opts} {fn}')
    # Load the converted .beta file, not the original bed file
    tmpfn = f"{dotbeta_dir}/{op.splitext(op.basename(fn))[0]}.beta"
    betafreq = load_beta_frequencies(tmpfn, *args, **kwargs)
    if rm_dotbeta:
        os.remove(tmpfn)
    return betafreq


def load_methylation_frequencies(
        files:ListFiles,
        file_type:Literal['infer', 'beta', 'bed', 'cov', 'bedgraph']='infer',
        *loadBedArgs, **loadBedKW
):
    """Load array of beta frequences, indexed by (site, sample, freq).
    With bf[:, :, n]; n=0 gives the met count, and n=1 gives
    total observed.
    If a bed|bedgraph|cov is being loaded, *args, **kwargs will be passed to
    `load_betafreq`, `load_dotbeta_from_bed`. Notable kwargs:
        dotbeta_dir: Output dir for created .beta files.
        force_overwrite: If True, overwrite .beta files that exist in dotbeta_dir
        """
    if file_type == 'infer':
        ext = op.splitext(str(files[-1]))[1].lstrip('.')
        # .beta and .lbeta share the same loader
        file_type = 'beta' if ext in ('beta', 'lbeta') else ext
    loaders = {
        'beta': load_beta_frequencies,
        'bed':partial(load_beta_frequencies_from_bed, bedgraph=False, *loadBedArgs, **loadBedKW, ),
        'bedgraph':partial(load_beta_frequencies_from_bed, bedgraph=True, *loadBedArgs, **loadBedKW, ),
        'cov':partial(load_beta_frequencies_from_bed, bedgraph=True, *loadBedArgs, **loadBedKW, ),
    }
    try:
        load_file = loaders[file_type]
    except KeyError:
        raise RuntimeError(f"Unrecognised file type: {file_type}")
    first = load_file(files[0])
    # create empty array
    arr = np.zeros(
        shape=(first.shape[0], len(files), 2),
        dtype=first.dtype
    )
    # The first sample is already loaded so lets just add it
    arr[:, 0, :] = first
    # populate the rest of the array
    for i, fn in enumerate(files[1:]):
        arr[:, i+1, :] = load_file(fn)
    return arr


def _test_load_met_freq(fn='path/to/file.beta'):
    files = [fn] * 40

    bf = load_methylation_frequencies(files)
    # calculate beta values
    betas = bf[:, :, 0] / bf[:, :, 1]
    # stdev of cpgs
    betas.std(axis=0, where=~np.isnan(betas))


@dataclass(slots=True)
class Pattern:
    """Represents a read in the wgbs .pat format.

    Site is 1 indexed."""
    chrm:str
    site:int
    pattern:str
    freq:int

def iterate_pat(fn) -> Pattern:
    with gzip.open(fn, 'rt') as f:
        for line in f:
            c, s, p, f = line.strip().split('\t')
            yield Pattern(c, s, p, f)

@dataclass(slots=True)
class Methylation:
    met:   MetFreqArr
    obs:   MetFreqArr
    beta:  npt.NDArray
    files: ListFiles|None
    @classmethod
    def from_freq_array(cls, freq:MetFreqArr, files=None):
        beta = freq[:, :, 0] / freq[:, :, 1]
        return cls(freq[:, :, 0], freq[:, :, 1], beta, files)
    @classmethod
    def from_dotbeta(cls, files:ListFiles):
        freq = load_methylation_frequencies(files)
        return cls.from_freq_array(freq, files)
    @classmethod
    def from_bedfiles(cls, files, dotbeta_outdir):
        freq = load_methylation_frequencies(files, bed=True, dotbeta_outdir=dotbeta_outdir)
        return cls.from_freq_array(freq, files)
