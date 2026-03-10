#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMR region plotting utilities.

Provides plot_region(), which stacks three types of panel:
  1. Methylation beta-value line plot (one line per sample)
  2. Optional genomic tracks — ATAC-seq or ChIP-seq, as either:
       * peak intervals  (BED-style DataFrame with 'start' / 'end' columns)
       * continuous signal (1-D array aligned to the region CpGs)
  3. Rotated tip-down covariance heatmap (matching the R version)
"""

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def plot_region(
    df,
    labels,
    cpg_coords,
    highlight_start=None,
    highlight_end=None,
    tracks=None,
    sample_colors=None,
    cmap='viridis',
    figsize=None,
):
    """
    Plot a DMR region as a stacked multi-panel figure.

    Parameters
    ----------
    df : array-like, shape (total_cpgs, n_samples)
        Beta-value matrix. Rows are CpG sites, columns are samples.
    labels : array-like of int
        Row indices into `df` defining the region to plot.
    cpg_coords : pd.DataFrame
        Genomic coordinates for the CpGs selected by `labels`. Must contain
        a 'start' column (genomic bp position). Used for x-axis tick labels
        and for mapping BED-format peak coordinates onto the plot x-axis.
    highlight_start : int or None
        1-based CpG index within the region to start the highlight shade.
    highlight_end : int or None
        1-based CpG index within the region to end the highlight shade.
    tracks : list of dict or None
        Each dict describes one genomic track panel inserted between the
        methylation plot and the covariance triangle. Required keys:

          'label' : str
              Panel y-axis label (shown rotated on the left).

        Exactly one of:

          'peaks' : pd.DataFrame
              BED-style table with 'start' and 'end' columns (genomic bp).
              Each row is drawn as a filled rectangle.
          'signal' : array-like, length n
              Continuous per-CpG signal (e.g. normalised read depth).
              Drawn as a filled area plot.

        Optional keys:

          'color' : str   — fill colour (default 'steelblue')

    sample_colors : list of str or None
        Per-sample line colours for the methylation panel. If None, all lines
        are drawn in dark grey.
    cmap : str
        Matplotlib colormap name for the covariance heatmap (default 'viridis').
    figsize : tuple or None
        (width, height) in inches. Inferred automatically if None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    region = np.asarray(df)[list(labels), :]   # (n_cpg, n_samples)
    cpg_pos = np.asarray(cpg_coords['start'])  # genomic bp, one per CpG in region
    n = region.shape[0]
    n_tracks = len(tracks) if tracks else 0

    # x-axis: CpG i occupies the interval [i, i+1]; tick at i+0.5 (0-based)
    tick_x = np.arange(n) + 0.5

    # --- Figure / GridSpec layout ---
    # Rows: methylation | track_0 … track_k | covariance triangle
    height_ratios = [3] + [1] * n_tracks + [max(n // 2, 3)]
    if figsize is None:
        fig_w = max(8, n * 0.35)
        fig_h = 0.6 * sum(height_ratios)
        figsize = (fig_w, fig_h)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2 + n_tracks, 1,
        figure=fig,
        hspace=0.05,
        height_ratios=height_ratios,
    )
    ax_line = fig.add_subplot(gs[0])
    ax_tracks = [fig.add_subplot(gs[1 + i], sharex=ax_line) for i in range(n_tracks)]
    ax_tri = fig.add_subplot(gs[1 + n_tracks], sharex=ax_line)

    xlim = (0, n)

    # -----------------------------------------------------------------------
    # Panel 1: methylation line plot
    # -----------------------------------------------------------------------
    if sample_colors is not None:
        for i, col in enumerate(sample_colors):
            ax_line.plot(tick_x, region[:, i], color=col, alpha=0.7, linewidth=0.8)
    else:
        ax_line.plot(tick_x, region, color='0.35', alpha=0.6, linewidth=0.8)

    if highlight_start is not None and highlight_end is not None:
        # highlight_start/end are 1-based CpG indices within the region
        hs = highlight_start - 1   # 0-based
        he = highlight_end - 1
        ax_line.axvspan(hs, he + 1, color='lightblue', alpha=0.35, linewidth=0)
        ax_line.axvline(hs,      color='red', linestyle='--', linewidth=0.9)
        ax_line.axvline(he + 1,  color='red', linestyle='--', linewidth=0.9)

    ax_line.set_xlim(*xlim)
    ax_line.set_ylim(-0.05, 1.05)
    ax_line.set_ylabel('Beta', fontsize=8)
    ax_line.set_xticks(tick_x)
    ax_line.set_xticklabels([])
    ax_line.tick_params(axis='x', length=0)
    ax_line.spines['bottom'].set_visible(False)

    # -----------------------------------------------------------------------
    # Panels 2 … k: ATAC / ChIP tracks
    # -----------------------------------------------------------------------
    for ax_t, track in zip(ax_tracks, tracks or []):
        color = track.get('color', 'steelblue')
        label = track.get('label', '')

        if 'signal' in track:
            sig = np.asarray(track['signal'], dtype=float)
            ax_t.fill_between(tick_x, 0, sig, color=color, alpha=0.85, linewidth=0)
            ax_t.set_ylim(0, sig.max() * 1.2 if sig.max() > 0 else 1)

        elif 'peaks' in track:
            peaks = track['peaks']
            for _, peak in peaks.iterrows():
                # Map genomic bp coordinates to the plot [0, n] x-axis.
                # CpG i occupies x in [i, i+1]; searchsorted gives the CpG
                # boundary nearest each peak edge.
                x0 = float(np.searchsorted(cpg_pos, peak['start'], side='left'))
                x1 = float(np.searchsorted(cpg_pos, peak['end'],   side='right'))
                x0 = max(x0, 0.0)
                x1 = min(x1, float(n))
                if x1 > x0:
                    ax_t.broken_barh(
                        [(x0, x1 - x0)], (0.1, 0.8),
                        facecolors=color, alpha=0.85,
                    )
            ax_t.set_ylim(0, 1)

        ax_t.set_xlim(*xlim)
        ax_t.set_ylabel(
            label, rotation=0, ha='right', va='center',
            labelpad=6, fontsize=8,
        )
        ax_t.set_xticks(tick_x)
        ax_t.set_xticklabels([])
        ax_t.tick_params(axis='x', length=0)
        ax_t.set_yticks([])
        for spine in ax_t.spines.values():
            spine.set_visible(False)

    # -----------------------------------------------------------------------
    # Last panel: rotated tip-down covariance heatmap
    # -----------------------------------------------------------------------
    S = np.cov(region)                        # (n, n) — CpG × CpG covariance
    mask = np.flip(np.tri(n, k=1))            # mask upper triangle + diagonal
    S_masked = np.ma.array(S, mask=mask.astype(bool))

    # Coordinate transform: rotate the matrix 45° so the diagonal points down.
    # Generates an (n+1) × (n+1) mesh of corner coordinates.
    t = np.array([[1, 0.5], [-1, 0.5]])
    pts = np.array(
        [(col, row)
         for row in range(n, -1, -1)
         for col in range(0, n + 1)]
    )
    coord = pts @ t
    # coord[:, 0] = col − row  (x before rescaling)
    # coord[:, 1] = 0.5*(col+row)  (y)
    X_raw = coord[:, 1].reshape(n + 1, n + 1)
    Y_raw = coord[:, 0].reshape(n + 1, n + 1)

    # Rescale X so CpG i aligns with tick_x[i] = i + 0.5
    # X_raw spans [0, n]; we map it to [0, n] (same domain as xlim).
    X_scaled = X_raw  # already in [0, n] — matches the shared x-axis

    ax_tri.pcolormesh(X_scaled, Y_raw, np.flipud(S_masked), cmap=cmap)
    ax_tri.set_xlim(*xlim)
    ax_tri.set_ylim(Y_raw.min(), Y_raw.max())
    ax_tri.set_xticks(tick_x)
    ax_tri.set_xticklabels(cpg_pos, rotation=90, fontsize=6)
    ax_tri.tick_params(axis='x', length=0)
    for spine in ax_tri.spines.values():
        spine.set_visible(False)
    ax_tri.get_yaxis().set_visible(False)

    fig.align_ylabels([ax_line] + ax_tracks)
    return fig
