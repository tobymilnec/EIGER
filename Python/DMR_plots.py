#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18 September 2024
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
