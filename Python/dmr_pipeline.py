#!/usr/bin/env python3
"""
dmr_pipeline.py

A command-line (or importable) module for running a DMR-calling pipeline.
The user supplies:
  - A meta CSV file containing (at least) beta file names in the first column.
  - The directory where the beta files are stored.
  - A CpG bed file (e.g. gzipped) for mapping coordinates.
The pipeline loads the beta files, computes DMRs, annotates them using a 
find_seperators function (with parallel processing), computes Laplacian scores,
maps CpG coordinates, and outputs a CSV of results.
"""

import os
import argparse
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from joblib import Parallel, delayed

from functions.load_functions import load_methylation_frequencies
from functions.processing import covariance_blocks, extract_blocks, find_seperators, compute_laplacian_score, expand_averages_column, fdr

# ============================
# MAIN FUNCTION
# ============================

def main():
    parser = argparse.ArgumentParser(description="DMR-calling pipeline")
    parser.add_argument("--meta", required=True, help="Path to meta CSV file with meta table")
    parser.add_argument("--beta-dir", required=True, help="Directory containing beta files")
    parser.add_argument("--location-file", required=True, help="Path to CpG bed file (e.g. gzipped) for coordinates")
    parser.add_argument("--output", default="DMRs_with_coords.csv", help="Output CSV file name")
    args = parser.parse_args()

    # --- Load meta table ---
    print("Loading meta table...")
    meta_df = pd.read_csv(args.meta)
    # Assume the first column contains beta file names.
    beta_files_list = meta_df.iloc[:, 0].tolist()
    # Build full paths (assumes file names in meta table are relative to beta-dir)
    beta_files_list = [join(args.beta_dir, f) for f in beta_files_list]

    # --- Load beta files ---
    print("Loading beta files...")
    beta_files = load_methylation_frequencies(beta_files_list, 'beta')
    # Assume beta_files has shape (n_CpGs, n_files, 2)
    betas = beta_files[:, :, 0] / beta_files[:, :, 1]

    # --- Compute covariance blocks and extract blocks ---
    print("Computing covariance blocks and extracting DMR blocks...")
    beta_cov = covariance_blocks(betas)
    blks = extract_blocks(beta_cov, DMR_length_threshold=4, cov_threshold=0.05)
    print(f"Found {len(blks)} blocks.")

    # --- Compute block means (DMRs) ---
    print("Computing block means (DMRs)...")
    from joblib import Parallel, delayed

    DMRS = Parallel(n_jobs=-1)(
        delayed(np.mean)(betas[range(blks[j, 0], blks[j, 1] + 1), :], axis=0)
        for j in range(len(blks))
    )
    DMRS_np = np.array(DMRS)
    # --- Compute Laplacian scores ---
    print("Computing Laplacian scores...")
    laplacian_score_DMRs = compute_laplacian_score(DMRS_np.T)

    # --- Annotate each block via parallel processing ---
    print("Annotating DMR blocks...")
    # Convert meta DataFrame to numpy array (if required by find_seperators)
    meta_arr = meta_df.to_numpy()
    DMRS_anno = Parallel(n_jobs=-1)(
        delayed(find_seperators)(
            betas, range(blks[j, 0], blks[j, 1] + 1), meta_arr,
            sample_avg_func=np.mean,
            thres=0.4, N_target=3, sigma=0.05, tau=0.02,
            tissue_col=3, celltype_col=2, subtype_col=2
        )
        for j in range(len(blks))
    )
    DMRS_anno_df = pd.DataFrame(DMRS_anno)

    # --- Add Laplacian score column ---
    DMRS_anno_df['laplacian_score'] = laplacian_score_DMRs

    # --- Expand the 'averages' column if present ---
    try:
        # If available, use the first column of meta as sample names.
        sample_names = meta_arr[:, 0]
    except Exception:
        sample_names = None
    if 'averages' in DMRS_anno_df.columns:
        DMRS_anno_df = expand_averages_column(DMRS_anno_df, averages_column='averages', sample_names=sample_names)

    # --- Optionally apply FDR correction on p-values (if column exists) ---
    if 'pvalue' in DMRS_anno_df.columns:
        DMRS_anno_df['fdr'] = fdr(DMRS_anno_df['pvalue'].to_numpy(dtype='float'))

    # --- Read CpG location file ---
    print("Reading CpG location file...")
    location = pd.read_csv(args.location_file, compression='gzip', header=None, sep='\t', quotechar='"')

    # --- Map DMR coordinates using location file ---
    print("Mapping DMR coordinates...")
    DMR_location_df = DMRS_anno_df.apply(
        lambda row: pd.Series({
            'chr': location.iloc[row['CpG_start'], 0],
            'start': location.iloc[row['CpG_start'], 1],
            'end': location.iloc[row['CpG_end'], 1]
        }),
        axis=1
    )

    # --- Concatenate location info with DMR annotation ---
    new_dataframe = pd.concat([DMR_location_df, DMRS_anno_df], axis=1)

    # --- Save final output ---
    print(f"Saving results to {args.output} ...")
    new_dataframe.to_csv(args.output, index=False)
    print("DMR pipeline finished successfully.")

if __name__ == "__main__":
    main()