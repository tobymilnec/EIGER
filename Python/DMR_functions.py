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
import math

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


# Correction functions (unchanged)
def f_rare_gaussian(p_obs, p_target, sigma):
    return np.exp(-((p_obs - p_target) ** 2) / (2 * sigma ** 2))

def f_rare_logistic(p_obs, p_target, tau):
    return 1 / (1 + np.exp((p_obs - p_target) / tau))

def find_separators(df, region, meta, 
                    sample_avg_func=np.mean,   # or your custom within_mean function
                    thres=0.5, 
                    N_target=3, 
                    sigma=0.05, 
                    tau=0.02,
                    tissue_col=4,    # column index for tissue grouping
                    celltype_col=3,  # column index for cell type grouping
                    subtype_col=2):  # extra column (not used in this example)
    """
    For a given region (row indices in 'region') of the data matrix df and metadata table meta:
    
      (A) Sample-level separation:
          - Computes per-sample averages.
          - Sorts the averages and finds the first gap over a given threshold (thres).
          - Returns that gap as delta_minmax and the contiguous block ("sample_index").
    
      (B) Group-level separation:
          - For each grouping variable (cell type and tissue), computes the group-average (across all samples).
          - Sorts these group averages and finds the first gap over the threshold.
          - If a gap is found, the separated group is assumed to be the one with the lowest average.
          - Also computes a p-value (and corresponding scores) for the difference between the separated group and the rest.
    
      (C) Classification:
          - Based on flags from sample-level separation, cell-type separation, and tissue separation,
            a classification label is assigned (e.g. "sample separation", "grouped celltype separation", 
            "sample + celltype separation", etc.).
          - The output fields 'celltypes_block' and 'tissues_block' will reflect the separated group if one is found,
            or the sample-level information otherwise.
          - The overall direction is computed from the separated groups (if available) or from the sample-level separation.
    
    The function returns a dictionary containing:
      - Sample-level separation indices and metrics.
      - Group-level separation indices (if found) and corresponding scores.
      - A classification label and direction.
      - Other diagnostic values.
    """
    # ---------------------------
    # (A) SAMPLE-LEVEL SEPARATION
    # ---------------------------
    avg = sample_avg_func(df[region, :], axis=0)
    sd = np.std(avg)
    N_total = avg.shape[0]
    
    # Sort sample indices by average methylation
    sorted_idx = np.argsort(avg)
    sorted_avg = avg[sorted_idx]
    sorted_diffs = np.abs(np.diff(sorted_avg))
    
    if sorted_diffs.size > 0 and np.any(sorted_diffs > thres):
        break_idx = np.argmax(sorted_diffs > thres)
        delta_minmax = sorted_diffs[break_idx]
    else:
        break_idx = len(sorted_idx) - 1
        delta_minmax = 0
    
    sample_values = sorted_idx[:break_idx+1]
    sample_remaining = sorted_idx[break_idx+1:] if (break_idx+1) < len(sorted_idx) else np.array([], dtype=int)
    sample_sep_flag = (delta_minmax > thres)
    
    # Prevalence corrections for sample-level separation:
    p_obs_sample = len(sample_values) / N_total
    p_target_sample = N_target / N_total
    correction_gaussian_sample = f_rare_gaussian(p_obs_sample, p_target_sample, sigma)
    correction_logistic_sample = f_rare_logistic(p_obs_sample, p_target_sample, tau)
    if sample_remaining.size > 0:
        pval_sample = stats.ttest_ind(avg[sample_values], avg[sample_remaining], nan_policy='omit')[1]
    else:
        pval_sample = 1.0
    if pval_sample == 0:
        pval_sample = 1e-4
    base_score_sample = (np.abs(delta_minmax) / sd) * (-math.log10(pval_sample))
    score_sample_gaussian = base_score_sample * correction_gaussian_sample
    score_sample_logistic = base_score_sample * correction_logistic_sample
    
    # Determine sample-level direction (based on the means of the two groups)
    if sample_remaining.size > 0:
        direction_sample = "Hypomethylation" if np.mean(avg[sample_values]) < np.mean(avg[sample_remaining]) else "Hypermethylation"
    else:
        direction_sample = "Uncertain"
    
    # ---------------------------
    # (B) GROUP-LEVEL SEPARATION: CELL TYPE
    # ---------------------------
    unique_celltypes = np.unique(meta[:, celltype_col])
    celltype_avgs = {}
    for ct in unique_celltypes:
        indices = np.where(meta[:, celltype_col] == ct)[0]
        celltype_avgs[ct] = np.mean(avg[indices])
    sorted_ct = sorted(celltype_avgs.items(), key=lambda x: x[1])
    if len(sorted_ct) > 1:
        gap_ct = sorted_ct[1][1] - sorted_ct[0][1]
        if gap_ct > thres:
            delta_celltype = gap_ct
            cell_sep_flag = True
            separated_celltype = sorted_ct[0][0]  # the group with the lowest average
        else:
            delta_celltype = 0
            cell_sep_flag = False
    else:
        delta_celltype = 0
        cell_sep_flag = False
    if cell_sep_flag:
        group_cell_indices = np.where(meta[:, celltype_col] == separated_celltype)[0]
    else:
        group_cell_indices = np.array([], dtype=int)
    
    if cell_sep_flag and group_cell_indices.size > 0:
        other_cell_indices = np.setdiff1d(np.arange(N_total), group_cell_indices)
        if other_cell_indices.size > 0:
            pval_cell = stats.ttest_ind(avg[group_cell_indices], avg[other_cell_indices], nan_policy='omit')[1]
        else:
            pval_cell = 1.0
        if pval_cell == 0:
            pval_cell = 1e-4
        base_score_cell = (np.abs(delta_celltype) / sd) * (-math.log10(pval_cell))
        p_obs_cell = len(group_cell_indices) / N_total
        correction_gaussian_cell = f_rare_gaussian(p_obs_cell, p_target_sample, sigma)
        correction_logistic_cell = f_rare_logistic(p_obs_cell, p_target_sample, tau)
        score_cell_gaussian = base_score_cell * correction_gaussian_cell
        score_cell_logistic = base_score_cell * correction_logistic_cell
        # Direction for cell group separation:
        mean_sep_cell = np.mean(avg[group_cell_indices])
        mean_other_cell = np.mean(avg[other_cell_indices]) if other_cell_indices.size > 0 else mean_sep_cell
        direction_cell = "Hypomethylation" if mean_sep_cell < mean_other_cell else "Hypermethylation"
    else:
        delta_celltype = 0
        pval_cell = None
        base_score_cell = None
        score_cell_gaussian = None
        score_cell_logistic = None
        direction_cell = None
    
    # ---------------------------
    # (B) GROUP-LEVEL SEPARATION: TISSUE
    # ---------------------------
    unique_tissues = np.unique(meta[:, tissue_col])
    tissue_avgs = {}
    for t in unique_tissues:
        indices = np.where(meta[:, tissue_col] == t)[0]
        tissue_avgs[t] = np.mean(avg[indices])
    sorted_t = sorted(tissue_avgs.items(), key=lambda x: x[1])
    if len(sorted_t) > 1:
        gap_t = sorted_t[1][1] - sorted_t[0][1]
        if gap_t > thres:
            delta_tissue = gap_t
            tissue_sep_flag = True
            separated_tissue = sorted_t[0][0]  # the tissue group with the lowest average
        else:
            delta_tissue = 0
            tissue_sep_flag = False
    else:
        delta_tissue = 0
        tissue_sep_flag = False
    if tissue_sep_flag:
        group_tissue_indices = np.where(meta[:, tissue_col] == separated_tissue)[0]
    else:
        group_tissue_indices = np.array([], dtype=int)
    
    if tissue_sep_flag and group_tissue_indices.size > 0:
        other_tissue_indices = np.setdiff1d(np.arange(N_total), group_tissue_indices)
        if other_tissue_indices.size > 0:
            pval_tissue = stats.ttest_ind(avg[group_tissue_indices], avg[other_tissue_indices], nan_policy='omit')[1]
        else:
            pval_tissue = 1.0
        if pval_tissue == 0:
            pval_tissue = 1e-4
        base_score_tissue = (np.abs(delta_tissue) / sd) * (-math.log10(pval_tissue))
        p_obs_tissue = len(group_tissue_indices) / N_total
        correction_gaussian_tissue = f_rare_gaussian(p_obs_tissue, p_target_sample, sigma)
        correction_logistic_tissue = f_rare_logistic(p_obs_tissue, p_target_sample, tau)
        score_tissue_gaussian = base_score_tissue * correction_gaussian_tissue
        score_tissue_logistic = base_score_tissue * correction_logistic_tissue
        # Direction for tissue group separation:
        mean_sep_tissue = np.mean(avg[group_tissue_indices])
        mean_other_tissue = np.mean(avg[other_tissue_indices]) if other_tissue_indices.size > 0 else mean_sep_tissue
        direction_tissue = "Hypomethylation" if mean_sep_tissue < mean_other_tissue else "Hypermethylation"
    else:
        delta_tissue = 0
        pval_tissue = None
        base_score_tissue = None
        score_tissue_gaussian = None
        score_tissue_logistic = None
        direction_tissue = None

    # ---------------------------
    # (C) CLASSIFICATION & OUTPUT ASSIGNMENT
    # ---------------------------
    # Flags for the separations:
    # sample_sep_flag is already defined; cell_sep_flag and tissue_sep_flag from above.
    if sample_sep_flag and cell_sep_flag and tissue_sep_flag:
        classification = "sample + celltype + tissue separation"
    elif sample_sep_flag and cell_sep_flag:
        classification = "sample + celltype separation"
    elif sample_sep_flag and tissue_sep_flag:
        classification = "sample + tissue separation"
    elif cell_sep_flag and tissue_sep_flag:
        classification = "grouped celltype + tissue separation"
    elif sample_sep_flag:
        classification = "sample separation"
    elif cell_sep_flag:
        classification = "grouped celltype separation"
    elif tissue_sep_flag:
        classification = "tissue separation"
    else:
        classification = "uncertain"
    
    # For the output, if a grouped separation is found, override the sample-level block info.
    if cell_sep_flag:
        celltypes_block = str(separated_celltype)
    else:
        celltypes_block = ','.join(np.unique(meta[sample_values, celltype_col]).astype(str))
    
    if tissue_sep_flag:
        tissues_block = str(separated_tissue)
    else:
        tissues_block = ','.join(np.unique(meta[sample_values, tissue_col]).astype(str))
    
    # Also, define group indices to return:
    group_cell_index = group_cell_indices if cell_sep_flag else np.array([], dtype=int)
    group_tissue_index = group_tissue_indices if tissue_sep_flag else np.array([], dtype=int)
    
    # Decide overall direction: if sample-level separation exists, use that;
    # otherwise, if group-level exists, use that.
    if sample_sep_flag:
        overall_direction = direction_sample
    elif cell_sep_flag:
        overall_direction = direction_cell
    elif tissue_sep_flag:
        overall_direction = direction_tissue
    else:
        overall_direction = "Uncertain"
    
    # Build the output dictionary.
    result = {
        'CpG_start': region[0],
        'CpG_end': region[-1],
        'sample_index': sample_values,         # sample-level contiguous block
        'group_cell_index': group_cell_index,    # indices for separated cell type, if any
        'group_tissue_index': group_tissue_index,  # indices for separated tissue, if any
        'celltypes_block': celltypes_block,        # separated cell type if found, else from sample block
        'tissues_block': tissues_block,            # separated tissue if found, else from sample block
        'classification': classification,
        'direction': overall_direction,
        'delta_minmax': delta_minmax,              # sample-level gap
        'delta_celltype_next': delta_celltype,     # group-level gap for cell type
        'delta_tissue_next': delta_tissue,         # group-level gap for tissue
        'stdev': sd,
        'pvalue_sample': pval_sample,
        'score_sample_gaussian': score_sample_gaussian,
        'score_sample_logistic': score_sample_logistic,
        'pvalue_cell': pval_cell,
        'score_cell_gaussian': score_cell_gaussian,
        'score_cell_logistic': score_cell_logistic,
        'pvalue_tissue': pval_tissue,
        'score_tissue_gaussian': score_tissue_gaussian,
        'score_tissue_logistic': score_tissue_logistic,
        'averages': avg
    }
    
    return result

def expand_averages_column(df, averages_column='averages', sample_names=None):
    """
    Given a DataFrame (df) that contains a column (averages_column) where each row 
    is an array or list of per-sample averages, this function splits that column into
    separate columns, one for each sample.
    
    Parameters:
      df : pandas.DataFrame
          The DataFrame that contains the averages column.
      averages_column : str, default 'averages'
          The name of the column in df that contains the array of averages.
      sample_names : list of str or None, default None
          Optional list of sample names to use as column headers. If None, defaults to 
          ['sample_0', 'sample_1', ...].
    
    Returns:
      df_out : pandas.DataFrame
          The DataFrame with the averages column replaced by individual sample columns.
    """
    # Convert the column of arrays/lists into a list of lists.
    avg_list = df[averages_column].tolist()
    
    # Determine the number of samples (assumes every row has the same length array)
    n_samples = len(avg_list[0])
    
    # If no sample names are provided, use default names.
    if sample_names is None:
        sample_names = [f'sample_{i}' for i in range(n_samples)]
    
    # Create a new DataFrame from the list of averages.
    averages_df = pd.DataFrame(avg_list, index=df.index, columns=sample_names)
    
    # Optionally drop the original averages column and join the new columns.
    df_out = df.drop(columns=[averages_column]).join(averages_df)
    
    return df_out