#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26 May 2023
@author: Toby Milne-Clark
"""

##### DMR calling functions
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from numba import jit
import seaborn as sns

### for calling DMRs from methylation blocks (chr or targeted regions)
# Elapsed (no jit) = 14.89403999998467s
# Elapsed (after jit compilation) = 0.482649749959819s
# Speed increase is noticible 

@jit(nopython=True)
def covariance_blocks(chr_arr):
    """
    Calculate the covariance between consecutive rows of a 2D array.

    Parameters:
    chr_arr (numpy.ndarray): The input 2D array.

    Returns:
    list: A list of covariances between consecutive rows of the input array.
    """
    M = len(chr_arr)-1
    covariance = [np.cov(chr_arr[j, :], chr_arr[j+1, :], ddof=1)[1][0] for j in range(M)]
    return covariance
def covariance_blocks(chr_arr):
    M = len(chr_arr)-1
    covariance = [np.cov(chr_arr[j, :], chr_arr[j+1, :],ddof=1)[1][0] for j in range(M)]
    return covariance

@jit(nopython=True)
def correlation_blocks(chr_arr):
    """
    Calculate the correlation between consecutive rows of a 2D array.

    Parameters:
    chr_arr (ndarray): The input 2D array.

    Returns:
    correlation (list): A list of correlation coefficients between consecutive rows.
    """
    M = len(chr_arr)-1
    correlation = [np.corrcoef(chr_arr[j, :], chr_arr[j+1, :])[1][0] for j in range(M)]
    return correlation


### define regions of continous covariance/correlation
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""
    # Find the indicies of changes in "condition"
    d = np.diff(condition,n=1, axis=0)
    idx, _ = d.nonzero() 
    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right. -JK
        # LB this copy to increment is horrible but I get 
    # ValueError: output array is read-only without it 
    mutable_idx = np.array(idx)
    mutable_idx +=  1
    idx = mutable_idx
    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]
    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit
    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

### Extracting the index of the coords
def extract_blocks(arr, DMR_length_threshold=5, cov_threshold=0.1):
    """
    Extracts contiguous blocks from an array based on given thresholds.

    Parameters:
    - arr: numpy array
        The input array from which blocks will be extracted.
    - DMR_length_threshold: int, optional (default=5)
        The minimum length of a block to be considered as a DMR (Differentially Methylated Region).
    - cov_threshold: float, optional (default=0.1)
        The coverage threshold for a value in the array to be considered as a match.

    Returns:
    - out: numpy array
        An array containing the start and stop indices of the extracted blocks.

    """
    match_bools = np.array(arr).reshape((-1, 1)) >= cov_threshold
    out = np.empty((0), int)
    for start, stop in contiguous_regions(match_bools):
        if (stop - start >= DMR_length_threshold):
            out = np.append(out, [start, stop])
    out.shape = (-1, 2)
    return out


### Plot DMR region
def plot_region(df,labels):
    region = df[labels,]
    S = np.cov(region)
    labels = list(labels)
    n = S.shape[0]
    mask = np.flip(np.tri(n, k=1))
    S_masked = np.ma.array(S, mask=mask)
    # create rotation/scaling matrix
    t = np.array([[1, 0.5], [-1, 0.5]])
    # create coordinate matrix and transform it
    coordinate_matrix = np.dot(np.array([(i[1], i[0])for i in itertools.product(range(n, -1, -1), range(0, n + 1, 1))]), t)
    # plot
    fig, (ax0, ax1) = plt.subplots(nrows=2,sharex=True)
    ax0.plot(region)
    ax0.set_xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation='vertical', fontsize=6)
    ax1.set_ylim(-(n-1),0)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax1.pcolormesh(coordinate_matrix[:, 1].reshape(n + 1, n + 1), coordinate_matrix[:, 0].reshape(n + 1, n + 1), np.flipud(S_masked))
    fig.subplots_adjust(hspace=0.1)
    fig.show()


