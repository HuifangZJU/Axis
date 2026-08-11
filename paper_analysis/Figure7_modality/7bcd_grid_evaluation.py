#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grid-based evaluation of a virtual protein slice against a real protein slice.

Why grid-based?
----------------
When the real and virtual protein slices have similar global spatial layout but
individual cells are not guaranteed to match one-to-one, nearest-neighbor
cell matching can be overly strict. This script instead aggregates local cells
into shared spatial grids and compares marker patterns at the grid level.

This is often more biologically appropriate for protein spatial pattern fidelity:
- less sensitive to single-cell position jitter
- emphasizes local neighborhood / regional pattern
- easier to interpret visually and quantitatively

Inputs
------
1. real protein h5ad
2. virtual protein h5ad

Outputs
-------
Tables
- alignment_and_grid_summary.csv
- marker_metrics.csv

Figures
- marker_metric_heatmap.png
- metric_distribution_boxplot.png
- ranked_grid_fidelity_barplot.png
- marker_panels/<marker>_grid_panel.png

Main metrics
------------
At the shared grid level, for each marker:
- Pearson
- Spearman
- R²
- Cosine similarity
- RMSE
- MAE
- Hotspot Jaccard (top-expression bins)
- Spatial Pearson (after 3x3 local smoothing)
- Spatial Spearman (after 3x3 local smoothing)
- Spatial hotspot Jaccard (smoothed)

Edit the CONFIG section and run:
    python evaluate_virtual_protein_spatial_grid.py
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import uniform_filter
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

REAL_PROTEIN_PATH = "/media/huifang/data/registration/phenocycler/huifang/h5ad/LUAD_2_A_protein.h5ad"
VIRTUAL_PROTEIN_PATH = "/media/huifang/data/registration/phenocycler/huifang/virtual_tissue/modality_LUAD_2_A/analysis/virtual_protein_1.000.h5ad"

OUTDIR = "/media/huifang/data/registration/phenocycler/huifang/virtual_tissue/modality_LUAD_2_A/analysis/evaluation/"

# markers to evaluate; if None, use all common var_names
MARKERS: Optional[List[str]] = None

COORD_KEY = "spatial"
FILTER_VALID = True
VALID_KEY = "valid"
USE_LOG1P = True
INVERT_Y = True

# -------- grid configuration --------
# If BIN_SIZE is None, the script uses GRID_N_X bins along x and preserves aspect ratio for y.
BIN_SIZE: Optional[float] = None
GRID_N_X = 200
MIN_CELLS_PER_BIN = 3
HOTSPOT_QUANTILE = 0.90
MIN_VALID_BINS = 12

# mean is recommended for protein intensity.
# choose from: "mean", "sum"
AGG_FUNC = "mean"

# optional smoothing on the binned map for spatial fidelity
SMOOTH_SIZE = 3  # 3x3 local mean filter

# plotting
FIG_DPI = 300
N_MARKERS_TO_PLOT = 8
PLOT_MARKERS: Optional[List[str]] = ['IDO1']
REAL_CMAP = "Blues"
DIFF_CMAP = "vlag"

METRIC_ORDER = [
    "pearson_r",
    "spearman_r",
    # "r2",
    "cosine",
    "spatial_pearson_r",
    "spatial_spearman_r",
    # "hotspot_jaccard",
    # "spatial_hotspot_jaccard",
    # "rmse",
    # "mae",
]

METRIC_DISPLAY = {
    "pearson_r": "Pearson",
    "spearman_r": "Spearman",
    # "r2": "R²",
    "cosine": "Cosine",
    "spatial_pearson_r": "Spatial Pearson",
    "spatial_spearman_r": "Spatial Spearman",
    # "hotspot_jaccard": "Hotspot Jaccard",
    # "spatial_hotspot_jaccard": "Spatial hotspot Jaccard",
    # "rmse": "RMSE",
    # "mae": "MAE",
}


# ============================================================
# STYLE
# ============================================================

def set_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.15,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# HELPERS
# ============================================================

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def save_or_show(fig, save_path=None):
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved: {save_path}")
    plt.close(fig)


def read_h5ad(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ad = sc.read_h5ad(path)
    if FILTER_VALID and VALID_KEY in ad.obs.columns:
        ad = ad[ad.obs[VALID_KEY].astype(bool)].copy()
    return ad


def load_coords(adata, coord_key=COORD_KEY):
    if coord_key not in adata.obsm:
        raise KeyError(f"Coordinate key '{coord_key}' not found in adata.obsm")
    return np.asarray(adata.obsm[coord_key], dtype=np.float32)


def extract_feature_vals(adata, feature: str, use_log1p: bool = USE_LOG1P) -> np.ndarray:
    if feature in adata.var_names:
        j = int(np.where(adata.var_names == feature)[0][0])
        v = adata.X[:, j]
        if sp.issparse(v):
            v = np.asarray(v.toarray()).ravel().astype(np.float32)
        else:
            v = np.asarray(v).ravel().astype(np.float32)
        if use_log1p and np.nanmin(v) >= 0:
            v = np.log1p(v)
    elif feature in adata.obs.columns:
        v = np.asarray(adata.obs[feature]).ravel().astype(np.float32)
    else:
        raise ValueError(f"Feature '{feature}' not found in var_names or obs.")
    return np.asarray(v, dtype=np.float32)


def get_common_markers(ad_real, ad_virtual) -> List[str]:
    return sorted(list(set(ad_real.var_names).intersection(set(ad_virtual.var_names))))


def robust_vrange(vals_list, low_pct=1.0, high_pct=99.0):
    arrs = []
    for v in vals_list:
        if v is None:
            continue
        v = np.asarray(v).ravel()
        v = v[np.isfinite(v)]
        if len(v) > 0:
            arrs.append(v)
    if len(arrs) == 0:
        return 0.0, 1.0
    arr = np.concatenate(arrs)
    vmin = np.percentile(arr, low_pct)
    vmax = np.percentile(arr, high_pct)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def safe_corr(x, y, fn):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < MIN_VALID_BINS:
        return np.nan
    try:
        return float(fn(x[good], y[good])[0])
    except Exception:
        return np.nan


def safe_r2(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < MIN_VALID_BINS:
        return np.nan
    try:
        return float(r2_score(x[good], y[good]))
    except Exception:
        return np.nan


def safe_cosine(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < MIN_VALID_BINS:
        return np.nan
    try:
        return float(cosine_similarity(x[good].reshape(1, -1), y[good].reshape(1, -1))[0, 0])
    except Exception:
        return np.nan


def hotspot_jaccard(a: np.ndarray, b: np.ndarray, quantile: float = HOTSPOT_QUANTILE) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    good = np.isfinite(a) & np.isfinite(b)
    if np.sum(good) < MIN_VALID_BINS:
        return np.nan

    a = a[good]
    b = b[good]
    thr_a = np.quantile(a, quantile)
    thr_b = np.quantile(b, quantile)
    hot_a = a >= thr_a
    hot_b = b >= thr_b
    union = np.sum(hot_a | hot_b)
    if union == 0:
        return np.nan
    inter = np.sum(hot_a & hot_b)
    return float(inter / union)


# ============================================================
# GRID CONSTRUCTION
# ============================================================

def compute_common_grid(coords_real: np.ndarray, coords_virtual: np.ndarray):
    coords_all = np.vstack([coords_real, coords_virtual])
    xmin, ymin = coords_all.min(axis=0)
    xmax, ymax = coords_all.max(axis=0)

    if BIN_SIZE is not None:
        x_edges = np.arange(xmin, xmax + BIN_SIZE, BIN_SIZE)
        y_edges = np.arange(ymin, ymax + BIN_SIZE, BIN_SIZE)
        if len(x_edges) < 2:
            x_edges = np.array([xmin, xmax + 1e-6])
        if len(y_edges) < 2:
            y_edges = np.array([ymin, ymax + 1e-6])
    else:
        nx = max(int(GRID_N_X), 2)
        width = xmax - xmin
        height = ymax - ymin
        if width <= 0:
            width = 1.0
        if height <= 0:
            height = 1.0
        dx = width / nx
        ny = max(int(np.round(height / dx)), 2)
        x_edges = np.linspace(xmin, xmax, nx + 1)
        y_edges = np.linspace(ymin, ymax, ny + 1)

    return x_edges, y_edges


def assign_bins(coords: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray):
    x_idx = np.digitize(coords[:, 0], x_edges) - 1
    y_idx = np.digitize(coords[:, 1], y_edges) - 1
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    valid = (x_idx >= 0) & (x_idx < nx) & (y_idx >= 0) & (y_idx < ny)
    return x_idx[valid], y_idx[valid], valid


def aggregate_to_grid(coords: np.ndarray, vals: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, agg_func: str = AGG_FUNC):
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    sum_grid = np.zeros((ny, nx), dtype=np.float64)
    count_grid = np.zeros((ny, nx), dtype=np.int32)

    x_idx, y_idx, valid = assign_bins(coords, x_edges, y_edges)
    v = vals[valid]

    good = np.isfinite(v)
    x_idx = x_idx[good]
    y_idx = y_idx[good]
    v = v[good]

    np.add.at(sum_grid, (y_idx, x_idx), v)
    np.add.at(count_grid, (y_idx, x_idx), 1)

    if agg_func == "sum":
        grid = sum_grid.copy()
        grid[count_grid == 0] = np.nan
    elif agg_func == "mean":
        grid = np.full((ny, nx), np.nan, dtype=np.float64)
        nz = count_grid > 0
        grid[nz] = sum_grid[nz] / count_grid[nz]
    else:
        raise ValueError("AGG_FUNC must be 'mean' or 'sum'")

    # require a minimum number of cells per bin for stable evaluation
    grid[count_grid < MIN_CELLS_PER_BIN] = np.nan
    return grid, count_grid


def smooth_grid_with_nan(grid: np.ndarray, size: int = SMOOTH_SIZE):
    mask = np.isfinite(grid).astype(np.float32)
    data = np.nan_to_num(grid, nan=0.0).astype(np.float32)
    num = uniform_filter(data, size=size, mode="nearest")
    den = uniform_filter(mask, size=size, mode="nearest")
    out = np.full_like(num, np.nan, dtype=np.float32)
    valid = den > 1e-6
    out[valid] = num[valid] / den[valid]
    return out


def compare_grids(real_grid: np.ndarray, virtual_grid: np.ndarray):
    good = np.isfinite(real_grid) & np.isfinite(virtual_grid)
    if np.sum(good) < MIN_VALID_BINS:
        return {k: np.nan for k in METRIC_ORDER}, good

    x = real_grid[good]
    y = virtual_grid[good]

    real_s = smooth_grid_with_nan(real_grid, size=SMOOTH_SIZE)
    virt_s = smooth_grid_with_nan(virtual_grid, size=SMOOTH_SIZE)
    good_s = np.isfinite(real_s) & np.isfinite(virt_s)
    xs = real_s[good_s]
    ys = virt_s[good_s]

    metrics = {
        "pearson_r": safe_corr(x, y, pearsonr),
        "spearman_r": safe_corr(x, y, spearmanr),
        "r2": safe_r2(x, y),
        "cosine": safe_cosine(x, y),
        "spatial_pearson_r": safe_corr(xs, ys, pearsonr),
        "spatial_spearman_r": safe_corr(xs, ys, spearmanr),
        "hotspot_jaccard": hotspot_jaccard(x, y, quantile=HOTSPOT_QUANTILE),
        "spatial_hotspot_jaccard": hotspot_jaccard(xs, ys, quantile=HOTSPOT_QUANTILE),
        "rmse": float(np.sqrt(mean_squared_error(x, y))),
        "mae": float(mean_absolute_error(x, y)),
    }
    return metrics, good


# ============================================================
# PLOTTING
# ============================================================

def plot_metric_heatmap(metric_df: pd.DataFrame, save_path=None):
    set_style()
    plot_df = metric_df.set_index("marker")[METRIC_ORDER].copy()
    plot_df.columns = [METRIC_DISPLAY[c] for c in plot_df.columns]

    fig_w = max(8.5, 0.9 * plot_df.shape[1] + 2.5)
    fig_h = max(5.0, 0.35 * plot_df.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        plot_df,
        cmap="vlag",
        center=0,
        linewidths=0.15,
        linecolor="white",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Metric value"},
        ax=ax,
    )
    ax.set_title("Real vs virtual protein fidelity by marker (grid-based)")
    ax.set_xlabel("")
    ax.set_ylabel("Marker")
    save_or_show(fig, save_path)


def plot_metric_distribution_boxplot(metric_df: pd.DataFrame, save_path=None):
    set_style()
    long_df = metric_df.melt(
        id_vars=["marker"],
        value_vars=METRIC_ORDER,
        var_name="metric",
        value_name="value",
    )
    long_df["metric_display"] = long_df["metric"].map(METRIC_DISPLAY)
    keep_metrics = [
        "pearson_r", "spearman_r", "cosine",
        "spatial_pearson_r", "spatial_spearman_r"
    ]
    long_df = long_df[long_df["metric"].isin(keep_metrics)].copy()
    order = [METRIC_DISPLAY[m] for m in keep_metrics]

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.boxplot(
        data=long_df,
        x="metric_display",
        y="value",
        order=order,
        color="#edf1f7",
        width=0.62,
        fliersize=0,
        linewidth=1.15,
        ax=ax,
    )
    sns.stripplot(
        data=long_df,
        x="metric_display",
        y="value",
        order=order,
        hue="marker",
        dodge=False,
        jitter=0.22,
        size=5.2,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.25,
        ax=ax,
        zorder=3,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


    ax.set_title("Distribution of marker fidelity metrics (grid-based)")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=26)
    ax.axhline(0, color="black", lw=0.8, alpha=0.55)
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_ranked_grid_fidelity_bar(metric_df: pd.DataFrame, save_path=None):
    set_style()
    df = metric_df.copy()
    df["combined_grid_score"] = (
        0.30 * df["spatial_pearson_r"].fillna(0) +
        0.20 * df["spatial_spearman_r"].fillna(0) +
        0.20 * df["hotspot_jaccard"].fillna(0) +
        0.20 * df["spatial_hotspot_jaccard"].fillna(0) +
        0.10 * df["pearson_r"].fillna(0)
    )
    df = df.sort_values("combined_grid_score", ascending=False)

    fig_h = max(4.8, 0.40 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    ax.barh(df["marker"], df["combined_grid_score"])
    ax.invert_yaxis()
    ax.set_title("Ranked spatial fidelity of virtual protein markers")
    ax.set_xlabel("Combined grid-based fidelity score")
    ax.set_ylabel("Marker")
    save_or_show(fig, save_path)


def plot_marker_panel(marker: str,
                      real_grid: np.ndarray,
                      virtual_grid: np.ndarray,
                      count_real: np.ndarray,
                      count_virtual: np.ndarray,
                      metrics_row: pd.Series,
                      save_path=None):
    set_style()

    diff_grid = virtual_grid - real_grid
    vmin, vmax = robust_vrange([real_grid, virtual_grid], 1, 99)
    finite_diff = diff_grid[np.isfinite(diff_grid)]
    dv = np.nanpercentile(np.abs(finite_diff), 99) if len(finite_diff) else 1.0
    dv = max(float(dv), 1e-6)

    good = np.isfinite(real_grid) & np.isfinite(virtual_grid)
    x = real_grid[good]
    y = virtual_grid[good]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.2))
    axes = axes.ravel()

    im0 = axes[0].imshow(real_grid, cmap=REAL_CMAP, vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title(f"Real protein grid: {marker}")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    if INVERT_Y:
        axes[0].invert_yaxis()
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

    im1 = axes[1].imshow(virtual_grid, cmap=REAL_CMAP, vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title(f"Virtual protein grid: {marker}")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    if INVERT_Y:
        axes[1].invert_yaxis()
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    im2 = axes[2].imshow(diff_grid, cmap=DIFF_CMAP, vmin=-dv, vmax=dv, origin="lower")
    axes[2].set_title(f"Virtual - real grid: {marker}")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    if INVERT_Y:
        axes[2].invert_yaxis()
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)

    axes[3].scatter(x, y, s=16, alpha=0.65, edgecolor="white", linewidth=0.25)
    if len(x):
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
    else:
        lo, hi = 0.0, 1.0
    axes[3].plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="0.4")
    axes[3].set_title("Grid-wise agreement")
    axes[3].set_xlabel("Real")
    axes[3].set_ylabel("Virtual")

    txt = (
        f"Pearson = {metrics_row['pearson_r']:.2f}\n"
        f"Spearman = {metrics_row['spearman_r']:.2f}\n"
        f"Spatial Pearson = {metrics_row['spatial_pearson_r']:.2f}\n"
        f"Spatial hotspot Jaccard = {metrics_row['spatial_hotspot_jaccard']:.2f}\n"
        f"R² = {metrics_row['r2']:.2f}\n"
        f"RMSE = {metrics_row['rmse']:.2f}"
    )
    axes[3].text(
        0.03, 0.97, txt,
        transform=axes[3].transAxes,
        va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.84, edgecolor="0.8"),
    )

    fig.suptitle(f"Grid-based marker fidelity panel: {marker}", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_or_show(fig, save_path)


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_outdir(OUTDIR)
    panel_dir = os.path.join(OUTDIR, "marker_panels")
    ensure_outdir(panel_dir)

    ad_real = read_h5ad(REAL_PROTEIN_PATH)
    ad_virtual = read_h5ad(VIRTUAL_PROTEIN_PATH)

    coords_real = load_coords(ad_real, 'spatial_axis')
    coords_virtual = load_coords(ad_virtual, 'spatial')

    if MARKERS is None:
        markers = get_common_markers(ad_real, ad_virtual)
    else:
        markers = []
        for m in MARKERS:
            ok = ((m in ad_real.var_names and m in ad_virtual.var_names) or
                  (m in ad_real.obs.columns and m in ad_virtual.obs.columns))
            if ok:
                markers.append(m)
    if len(markers) == 0:
        raise ValueError("No valid markers found for evaluation.")

    x_edges, y_edges = compute_common_grid(coords_real, coords_virtual)
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    # occupancy summary using cell counts only
    dummy_real = np.ones(coords_real.shape[0], dtype=np.float32)
    dummy_virtual = np.ones(coords_virtual.shape[0], dtype=np.float32)
    _, real_counts = aggregate_to_grid(coords_real, dummy_real, x_edges, y_edges, agg_func="sum")
    _, virtual_counts = aggregate_to_grid(coords_virtual, dummy_virtual, x_edges, y_edges, agg_func="sum")
    occ_real = real_counts >= MIN_CELLS_PER_BIN
    occ_virtual = virtual_counts >= MIN_CELLS_PER_BIN
    occ_overlap = occ_real & occ_virtual

    summary_df = pd.DataFrame([{
        "n_real_cells": int(coords_real.shape[0]),
        "n_virtual_cells": int(coords_virtual.shape[0]),
        "grid_nx": int(nx),
        "grid_ny": int(ny),
        "n_total_bins": int(nx * ny),
        "n_real_occupied_bins": int(np.sum(occ_real)),
        "n_virtual_occupied_bins": int(np.sum(occ_virtual)),
        "n_shared_occupied_bins": int(np.sum(occ_overlap)),
        "shared_occupancy_fraction_vs_real": float(np.sum(occ_overlap) / max(np.sum(occ_real), 1)),
        "shared_occupancy_fraction_vs_virtual": float(np.sum(occ_overlap) / max(np.sum(occ_virtual), 1)),
        "agg_func": AGG_FUNC,
        "min_cells_per_bin": MIN_CELLS_PER_BIN,
        "bin_size": BIN_SIZE if BIN_SIZE is not None else np.nan,
        "grid_n_x_requested": GRID_N_X if BIN_SIZE is None else np.nan,
    }])
    summary_df.to_csv(os.path.join(OUTDIR, "alignment_and_grid_summary.csv"), index=False)

    rows = []
    marker_cache = {}

    for marker in markers:
        real_vals = extract_feature_vals(ad_real, marker, use_log1p=USE_LOG1P)
        virt_vals = extract_feature_vals(ad_virtual, marker, use_log1p=USE_LOG1P)

        real_grid, count_real = aggregate_to_grid(coords_real, real_vals, x_edges, y_edges, agg_func=AGG_FUNC)
        virt_grid, count_virtual = aggregate_to_grid(coords_virtual, virt_vals, x_edges, y_edges, agg_func=AGG_FUNC)

        metrics, good_mask = compare_grids(real_grid, virt_grid)
        row = {
            "marker": marker,
            "n_valid_bins": int(np.sum(good_mask)),
            **metrics,
        }
        rows.append(row)
        marker_cache[marker] = {
            "real_grid": real_grid,
            "virt_grid": virt_grid,
            "count_real": count_real,
            "count_virtual": count_virtual,
        }

    metric_df = pd.DataFrame(rows).sort_values("spatial_pearson_r", ascending=False).reset_index(drop=True)
    metric_df.to_csv(os.path.join(OUTDIR, "marker_metrics.csv"), index=False)

    plot_metric_heatmap(metric_df, save_path=os.path.join(OUTDIR, "marker_metric_heatmap.png"))
    plot_metric_distribution_boxplot(metric_df, save_path=os.path.join(OUTDIR, "metric_distribution_boxplot.png"))
    plot_ranked_grid_fidelity_bar(metric_df, save_path=os.path.join(OUTDIR, "ranked_grid_fidelity_barplot.png"))

    # if PLOT_MARKERS is not None:
    #     plot_markers = [m for m in PLOT_MARKERS if m in metric_df["marker"].tolist()]
    # else:
    #     plot_markers = metric_df["marker"].head(N_MARKERS_TO_PLOT).tolist()
    #
    # for marker in plot_markers:
    #     cache = marker_cache[marker]
    #     row = metric_df[metric_df["marker"] == marker].iloc[0]
    #     plot_marker_panel(
    #         marker=marker,
    #         real_grid=cache["real_grid"],
    #         virtual_grid=cache["virt_grid"],
    #         count_real=cache["count_real"],
    #         count_virtual=cache["count_virtual"],
    #         metrics_row=row,
    #         save_path=os.path.join(panel_dir, f"{marker}_grid_panel.png"),
    #     )
    #
    # print(f"[DONE] Outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
