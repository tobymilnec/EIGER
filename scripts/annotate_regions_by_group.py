#!/usr/bin/env python3
"""Classify called regions by the sample groups they separate.

Deliberately OUTSIDE the eiger2 package. Group assignment is study-specific and
post-selection: it describes regions that discovery has already chosen, using
the same data, so nothing here carries an error rate. Keeping it separate stops
that distinction from blurring, and stops the caller accumulating cohort-specific
logic.

Groups are read from a suffix on the sample column name, which for the CRUK-ICGC
matrix is the last underscore-delimited field: ``..._L_TUM`` -> ``TUM``.

Classes assigned, in priority order
-----------------------------------
``tumour_vs_benign``   the tumour and benign group means differ by at least
                       ``--delta``; the contrast the cohort was designed for
``monotonic``          group means increase (or decrease) strictly through the
                       stated progression order, spanning at least ``--delta``
``tumour_subgroup``    a minority of tumours separates from the remaining
                       tumours by at least ``--delta``; small-n tumour biology
                       that a group-mean contrast cannot see
``unlabelled_subgroup`` a minority of samples separates from the rest by at
                       least ``--delta``, and the split is not explained by any
                       group label
``ungrouped``          none of the above

Usage
-----
    python scripts/annotate_regions_by_group.py \\
        --regions results/regions.csv \\
        --sample-means results/region_sample_means.csv \\
        --out results/region_groups.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#: Progression order for the monotonic test. Groups absent from the data are
#: dropped, and the test needs at least three of them.
PROGRESSION = ("BEN", "NRM", "BPH", "PIN", "TUM")

#: Groups excluded from every test: unknown or unassigned.
EXCLUDE = {"XXX", "NA", "UNK", ""}


def parse_groups(columns: list[str], field: int = -1) -> pd.Series:
    """Group label per sample, from an underscore-delimited field of the name."""

    return pd.Series(
        [str(c).split("_")[field].upper() for c in columns], index=columns
    )


def largest_gap_split(values: np.ndarray) -> tuple[float, np.ndarray]:
    """Bipartition the samples by minimising within-cluster sum of squares.

    Returns ``(|mean difference|, boolean mask of the minority side)``.

    The name is historical. The original ``find_separators`` split at the
    largest adjacent gap in the sorted sample means; that criterion collapses
    once the cohort is large, because the sorted means become near-continuous
    and the biggest single gap sits beside an outlier rather than at a group
    boundary. Measured on 3,000 regions whose tumour and benign group means
    differed by at least 0.20, the largest gap returned a minority of one sample
    in the median case, against 221 for the two-means criterion used here --
    the correct order for a 267 versus 227 split.

    Two-means still isolates genuinely rare subgroups: four strongly displaced
    samples minimise the within-cluster criterion at a four-versus-rest cut. The
    minority side is returned rather than the low side, so a hypermethylated
    subgroup is not reported as a hypomethylated majority.
    """

    finite = np.isfinite(values)
    if finite.sum() < 4:
        return 0.0, np.zeros(values.size, dtype=bool)
    index = np.flatnonzero(finite)
    order = index[np.argsort(values[index])]
    sorted_values = values[order]
    n = sorted_values.size

    cumulative = np.cumsum(sorted_values, dtype=np.float64)
    cumulative_square = np.cumsum(sorted_values.astype(np.float64) ** 2)
    left = np.arange(1, n)
    mean_left = cumulative[:-1] / left
    mean_right = (cumulative[-1] - cumulative[:-1]) / (n - left)
    within = (
        (cumulative_square[:-1] - left * mean_left ** 2)
        + ((cumulative_square[-1] - cumulative_square[:-1])
           - (n - left) * mean_right ** 2)
    )
    cut = int(np.argmin(within))
    delta = float(abs(mean_left[cut] - mean_right[cut]))
    low, high = order[: cut + 1], order[cut + 1 :]
    minor = low if low.size <= high.size else high
    mask = np.zeros(values.size, dtype=bool)
    mask[minor] = True
    return delta, mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regions", required=True)
    parser.add_argument("--sample-means", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--delta", type=float, default=0.20,
                        help="minimum difference in methylation proportion "
                             "(0.20 = 20 percentage points)")
    parser.add_argument("--max-subgroup-fraction", type=float, default=0.25,
                        help="a 'subgroup' is a minority no larger than this "
                             "fraction of the relevant samples")
    parser.add_argument("--min-subgroup-size", type=int, default=3)
    parser.add_argument("--group-field", type=int, default=-1,
                        help="underscore-delimited field of the sample name "
                             "holding the group label (default last)")
    parser.add_argument("--chunk", type=int, default=20000)
    args = parser.parse_args()

    regions = pd.read_csv(args.regions)
    header = pd.read_csv(args.sample_means, nrows=0)
    samples = [c for c in header.columns if c != "region_id"]
    groups = parse_groups(samples, args.group_field)

    counts = groups.value_counts()
    print("Sample groups:")
    for name, n in counts.items():
        flag = "  (excluded)" if name in EXCLUDE else ""
        print(f"  {name:5s} {n:4d}{flag}")

    usable = ~groups.isin(EXCLUDE)
    order = [g for g in PROGRESSION if g in set(groups[usable])]
    has_tumour = "TUM" in order
    benign_like = [g for g in ("BEN", "NRM") if g in order]
    print(f"\nProgression order for the monotonic test: {' < '.join(order)}")
    if len(order) < 3:
        print("  (fewer than three ordered groups; monotonic test skipped)")

    rows = []
    reader = pd.read_csv(
        args.sample_means, chunksize=args.chunk,
        dtype=dict.fromkeys(samples, np.float32),
    )
    seen = 0
    for chunk in reader:
        ids = chunk["region_id"].to_numpy()
        values = chunk[samples].to_numpy(dtype=np.float32)
        seen += len(chunk)

        group_mean = {
            g: np.nanmean(values[:, (groups == g).to_numpy()], axis=1)
            for g in order
        }
        tum = group_mean.get("TUM")
        ben = (
            np.nanmean(
                np.column_stack([group_mean[g] for g in benign_like]), axis=1
            ) if benign_like else None
        )

        for i in range(len(chunk)):
            row = values[i]
            record = {"region_id": ids[i]}
            for g in order:
                record[f"mean_{g}"] = float(group_mean[g][i])

            # 1. tumour versus benign
            tvb = (
                float(tum[i] - ben[i])
                if tum is not None and ben is not None else np.nan
            )
            record["delta_tumour_benign"] = tvb

            # 2. monotonic progression
            monotonic = False
            if len(order) >= 3:
                means = np.array([group_mean[g][i] for g in order])
                if np.all(np.isfinite(means)):
                    span = float(means[-1] - means[0])
                    increasing = np.all(np.diff(means) > 0)
                    decreasing = np.all(np.diff(means) < 0)
                    monotonic = (increasing or decreasing) and abs(span) >= args.delta
                    record["monotonic_span"] = span
            record["monotonic"] = bool(monotonic)

            # 3. a minority of tumours separating from the rest
            tum_delta, tum_minor = 0.0, None
            if has_tumour:
                mask = (groups == "TUM").to_numpy()
                tum_delta, minor = largest_gap_split(row[mask])
                size = int(minor.sum())
                if (args.min_subgroup_size <= size
                        <= args.max_subgroup_fraction * mask.sum()
                        and tum_delta >= args.delta):
                    tum_minor = size
            record["tumour_subgroup_delta"] = tum_delta
            record["tumour_subgroup_size"] = tum_minor if tum_minor else 0

            # 4. any minority split, and whether a label explains it
            all_delta, all_minor = largest_gap_split(row[usable.to_numpy()])
            minor_size = int(all_minor.sum())
            label_purity = np.nan
            if minor_size:
                minor_groups = groups[usable].to_numpy()[all_minor]
                label_purity = float(
                    pd.Series(minor_groups).value_counts(normalize=True).max()
                )
            record["split_delta"] = all_delta
            record["split_minority_size"] = minor_size
            # recomputed here so an existing run can be corrected without
            # re-calling; supersedes minority_size_descriptive where they differ
            record["prevalence_recomputed"] = minor_size
            record["split_label_purity"] = label_purity

            # classification, in priority order
            if np.isfinite(tvb) and abs(tvb) >= args.delta:
                label = "tumour_vs_benign"
            elif monotonic:
                label = "monotonic"
            elif tum_minor:
                label = "tumour_subgroup"
            elif (all_delta >= args.delta
                  and args.min_subgroup_size <= minor_size
                  <= args.max_subgroup_fraction * usable.sum()
                  and (not np.isfinite(label_purity) or label_purity < 0.9)):
                label = "unlabelled_subgroup"
            else:
                label = "ungrouped"
            record["class"] = label
            record["direction"] = (
                "hyper" if (np.isfinite(tvb) and tvb > 0) else
                "hypo" if np.isfinite(tvb) else "NA"
            )
            rows.append(record)

        print(f"  ...{seen:,} regions", end="\r")

    frame = pd.DataFrame(rows)
    merged = regions.merge(frame, on="region_id", how="left")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    print(f"\n\nClassified {len(frame):,} regions "
          f"(delta >= {args.delta}, subgroup <= "
          f"{args.max_subgroup_fraction:.0%} of samples)\n")
    summary = frame["class"].value_counts()
    for name, n in summary.items():
        print(f"  {name:22s} {n:7,}  ({100 * n / len(frame):5.1f}%)")

    tvb = frame[frame["class"] == "tumour_vs_benign"]
    if len(tvb):
        print(f"\n  tumour vs benign: {int((tvb.delta_tumour_benign > 0).sum()):,} "
              f"hyper, {int((tvb.delta_tumour_benign < 0).sum()):,} hypo; "
              f"median |delta| {tvb.delta_tumour_benign.abs().median():.3f}")
    mono = frame[frame["class"] == "monotonic"]
    if len(mono) and "monotonic_span" in mono:
        print(f"  monotonic: {int((mono.monotonic_span > 0).sum()):,} increasing, "
              f"{int((mono.monotonic_span < 0).sum()):,} decreasing")
    sub = frame[frame["class"] == "tumour_subgroup"]
    if len(sub):
        print(f"  tumour subgroups: median size "
              f"{sub.tumour_subgroup_size.median():.0f} of "
              f"{int((groups == 'TUM').sum())} tumours")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
