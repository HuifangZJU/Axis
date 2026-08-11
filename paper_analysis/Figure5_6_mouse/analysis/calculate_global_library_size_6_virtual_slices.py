#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate one global library-size normalization target from the six raw/count-like
virtual mouse-brain slices used in the disease analysis.

For each spatial location/cell i:
    library_size_i = sum_g X[i, g]

The final normalization target is the median library size pooled across ALL
valid locations from all six virtual slices.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


# ============================================================
# CONFIG
# ============================================================

VIRTUAL_FILES = [
    (
        "Disease_2.5m",
        "/media/huifang/data/registration/xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_2.500.h5ad",
    ),
    (
        "Normal_2.5m",
        "/media/huifang/data/registration/xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_102.500.h5ad",
    ),
    (
        "Disease_5.7m",
        "/media/huifang/data/registration/xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_5.700.h5ad",
    ),
    (
        "Normal_5.7m",
        "/media/huifang/data/registration/xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_105.700.h5ad",
    ),
    (
        "Disease_13.4m",
        "/media/huifang/data/registration/xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_13.400.h5ad",
    ),
    (
        "Normal_13.4m",
        "/media/huifang/data/registration/xenium/mouse_brain/"
        "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
        "virtual_Disease_time_113.400.h5ad",
    ),
]

VALID_KEY = "valid"
USE_VALID_ONLY = True

OUTDIR = (
    "/media/huifang/data/registration/paper_figure/"
    "Figure6_disease/library_size_diagnosis"
)

SUMMARY_CSV = os.path.join(
    OUTDIR,
    "six_virtual_slices_library_size_summary.csv",
)


# ============================================================
# HELPERS
# ============================================================

def get_valid_mask(adata):
    if VALID_KEY not in adata.obs.columns:
        return np.ones(adata.n_obs, dtype=bool)

    values = adata.obs[VALID_KEY].to_numpy()

    if values.dtype == bool:
        return values.astype(bool, copy=False)

    if np.issubdtype(values.dtype, np.number):
        return np.asarray(values, dtype=float) != 0

    values = np.asarray(values).astype(str)
    values = np.char.lower(np.char.strip(values))
    return np.isin(values, ["true", "1", "yes", "y", "t"])


def calculate_library_sizes(adata):
    """
    Sum raw/count-like expression across ALL genes for every location.
    """
    if sp.issparse(adata.X):
        totals = np.asarray(adata.X.sum(axis=1)).ravel()
    else:
        totals = np.asarray(adata.X, dtype=np.float64).sum(axis=1)

    return np.asarray(totals, dtype=np.float64)


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "n_locations": 0,
            "mean": np.nan,
            "median": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "min": np.nan,
            "max": np.nan,
            "zero_library_fraction": np.nan,
        }

    return {
        "n_locations": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q05": float(np.percentile(values, 5)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "q95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "zero_library_fraction": float(np.mean(values <= 0)),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    pooled_library_sizes = []
    rows = []

    print("=" * 78)
    print("Six-slice virtual-tissue library-size diagnosis")
    print("=" * 78)
    print(f"Use valid locations only: {USE_VALID_ONLY}")
    print()

    for label, path in VIRTUAL_FILES:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing file for {label}:\n{path}"
            )

        adata = sc.read_h5ad(path)

        # IMPORTANT: calculate library size from the full raw/count-like X
        # before any gene subsetting, normalization, or log transform.
        library_sizes_all = calculate_library_sizes(adata)

        if USE_VALID_ONLY:
            valid_mask = get_valid_mask(adata)
            library_sizes = library_sizes_all[valid_mask]
        else:
            valid_mask = np.ones(adata.n_obs, dtype=bool)
            library_sizes = library_sizes_all

        # Zero-library locations cannot be normalized meaningfully and should
        # not determine the normalization target.
        positive_library_sizes = library_sizes[
            np.isfinite(library_sizes) & (library_sizes > 0)
        ]

        if positive_library_sizes.size == 0:
            raise ValueError(
                f"{label} contains no positive library-size locations."
            )

        stats_all = summarize(library_sizes)
        stats_pos = summarize(positive_library_sizes)

        rows.append(
            {
                "slice": label,
                "path": path,
                "n_obs_total": int(adata.n_obs),
                "n_valid": int(valid_mask.sum()),
                "n_genes": int(adata.n_vars),
                **stats_all,
                "positive_n_locations": stats_pos["n_locations"],
                "positive_mean": stats_pos["mean"],
                "positive_median": stats_pos["median"],
                "positive_q05": stats_pos["q05"],
                "positive_q25": stats_pos["q25"],
                "positive_q75": stats_pos["q75"],
                "positive_q95": stats_pos["q95"],
            }
        )

        pooled_library_sizes.append(positive_library_sizes)

        print(
            f"{label:16s} | "
            f"n={positive_library_sizes.size:8d} | "
            f"median={np.median(positive_library_sizes):8.3f} | "
            f"Q25={np.percentile(positive_library_sizes, 25):8.3f} | "
            f"Q75={np.percentile(positive_library_sizes, 75):8.3f} | "
            f"mean={np.mean(positive_library_sizes):8.3f}"
        )

    pooled = np.concatenate(pooled_library_sizes)
    pooled_stats = summarize(pooled)

    global_median_library_size = float(np.median(pooled))
    global_median_library_size_rounded = int(
        np.rint(global_median_library_size)
    )

    summary_df = pd.DataFrame(rows)

    pooled_row = {
        "slice": "ALL_6_POOLED",
        "path": "",
        "n_obs_total": int(summary_df["n_obs_total"].sum()),
        "n_valid": int(summary_df["n_valid"].sum()),
        "n_genes": (
            int(summary_df["n_genes"].iloc[0])
            if summary_df["n_genes"].nunique() == 1
            else np.nan
        ),
        **pooled_stats,
        "positive_n_locations": pooled_stats["n_locations"],
        "positive_mean": pooled_stats["mean"],
        "positive_median": pooled_stats["median"],
        "positive_q05": pooled_stats["q05"],
        "positive_q25": pooled_stats["q25"],
        "positive_q75": pooled_stats["q75"],
        "positive_q95": pooled_stats["q95"],
    }

    summary_df = pd.concat(
        [summary_df, pd.DataFrame([pooled_row])],
        ignore_index=True,
    )

    summary_df.to_csv(SUMMARY_CSV, index=False)

    print()
    print("=" * 78)
    print("GLOBAL NORMALIZATION TARGET")
    print("=" * 78)
    print(f"Pooled positive locations: {pooled.size:,}")
    print(
        f"Global median library size: "
        f"{global_median_library_size:.6f}"
    )
    print(
        f"Rounded config value: "
        f"{global_median_library_size_rounded}"
    )
    print()
    print("Use this in downstream scripts:")
    print(
        f"NORMALIZATION_TARGET_SUM = "
        f"{global_median_library_size_rounded}"
    )
    print()
    print(f"[SAVE] {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
