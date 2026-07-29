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
from pathlib import Path
from dataclasses import dataclass
from subprocess import call
from functools import partial

import numpy as np
import numpy.typing as npt
import typing
from typing import (
    Literal
)

Read = npt.NDArray
MetFreqArr = npt.NDArray
ListFiles = list[str|Path]

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

