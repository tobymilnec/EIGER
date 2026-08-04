# Preserved EIGER hard-threshold caller

EIGER 3 retains the method that preceded probabilistic segmentation as the
`eiger-hard-threshold-v1` profile. It preserves:

- adjacent sample covariance as the discovery signal;
- a fixed or label-free resampled covariance threshold;
- contiguous runs in which every edge passes that threshold;
- the largest adjacent sample gap as the inferred partition;
- absolute cluster-mean delta beta as the call-status criterion; and
- the Laplacian rank percentile in the historical `stability_score` field.

Run this method through the normal command-line pipeline with:

```bash
python Python/dmr_pipeline.py \
    --calling-method legacy \
    [the remaining usual EIGER arguments]
```

Matrix API users can import
`Python.legacy.eiger_hard_threshold_v1.call_dmrs_from_matrix`.

The probabilistic caller does **not** replace covariance. It calculates the
same adjacent covariance and threshold, then quantifies uncertainty around
that threshold before spatial segmentation.
