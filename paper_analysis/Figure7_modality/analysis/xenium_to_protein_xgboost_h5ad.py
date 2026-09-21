#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-modality prediction from Xenium gene reads to protein reads using XGBoost.

What this script does
---------------------
1. Reads two h5ad files:
   - Xenium gene h5ad
   - protein h5ad
2. Aligns cells between modalities using shared coordinates (or matching obs_names if available).
3. Splits cells into train / validation / test.
4. Selects XGBoost hyperparameters on the validation set and trains one regressor per protein.
5. Saves predictions for all cells and for held-out test cells.
6. Produces:
   - train/val/test spatial layout
   - per-protein evaluation tables
   - global metric plots
   - spatial case-study plots for selected proteins

Widely used evaluation criteria
-------------------------------
For each protein on the held-out test cells:
- Pearson correlation
- Spearman correlation
- R²
- MAE
- RMSE

Spatially aggregated evaluation on held-out test cells:
- grid Pearson
- grid Spearman
- grid R²
- grid MAE
- grid RMSE

Outputs
-------
<outdir>/
  aligned_pred_all.h5ad
  aligned_pred_test_only.h5ad
  per_protein_metrics_test.csv
  run_summary.json
  train_val_test_layout.png
  metric_heatmap.png
  ranked_pearson.png
  ranked_r2.png
  case_study_<protein>_true.png
  case_study_<protein>_pred.png
  case_study_<protein>_diff.png
  case_study_<protein>_scatter.png
  case_study_<protein>_grid_true.png
  case_study_<protein>_grid_pred.png
  case_study_<protein>_grid_diff.png
"""

import os
import re
import json
import time
import math
import argparse
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Timing / logging
# ============================================================

class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.last = self.t0

    def log(self, msg: str) -> None:
        now = time.time()
        print(f"[{now - self.t0:8.1f}s | +{now - self.last:6.1f}s] {msg}", flush=True)
        self.last = now


# ============================================================
# Basic utilities
# ============================================================

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_dense_2d(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def as_float32_matrix(X):
    if sp.issparse(X):
        return X.astype(np.float32).tocsr()
    return np.asarray(X, dtype=np.float32)


def setup_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 220,
        "savefig.dpi": 220,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# Coordinate extraction / alignment
# ============================================================

def get_coords_from_adata(
    adata: ad.AnnData,
    obsm_key: Optional[str] = "spatial",
    obs_x_keys: Sequence[str] = ("x", "X", "center_x", "x_centroid", "row"),
    obs_y_keys: Sequence[str] = ("y", "Y", "center_y", "y_centroid", "col"),
) -> pd.DataFrame:
    """
    Return a DataFrame with columns ['x', 'y'].
    Preference:
      1) adata.obsm[obsm_key]
      2) obs columns
    """
    if obsm_key is not None and obsm_key in adata.obsm:
        arr = np.asarray(adata.obsm[obsm_key])
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"adata.obsm['{obsm_key}'] must be (n_cells, >=2), got {arr.shape}")
        coords = pd.DataFrame(arr[:, :2], columns=["x", "y"], index=adata.obs_names)
        coords[["x", "y"]] = coords[["x", "y"]].astype(float)
        return coords

    x_col = None
    y_col = None
    for c in obs_x_keys:
        if c in adata.obs.columns:
            x_col = c
            break
    for c in obs_y_keys:
        if c in adata.obs.columns:
            y_col = c
            break

    if x_col is None or y_col is None:
        raise ValueError(
            "Could not find coordinates in adata.obsm['spatial'] or obs columns. "
            f"Available obsm keys: {list(adata.obsm.keys())}; obs columns: {list(adata.obs.columns)[:30]}"
        )

    coords = pd.DataFrame({
        "x": pd.to_numeric(adata.obs[x_col], errors="coerce"),
        "y": pd.to_numeric(adata.obs[y_col], errors="coerce"),
    }, index=adata.obs_names)
    if coords.isna().any().any():
        raise ValueError("Coordinate columns contain NaN values after numeric conversion.")
    return coords


def _coord_key(coords: pd.DataFrame, decimals: int = 6) -> pd.Series:
    xr = coords["x"].round(decimals).astype(str)
    yr = coords["y"].round(decimals).astype(str)
    return xr + "__" + yr


def align_gene_and_protein_adatas(
    gene_adata: ad.AnnData,
    protein_adata: ad.AnnData,
    coord_obsm_key_gene: str = "spatial",
    coord_obsm_key_protein: str = "spatial",
    coord_round_decimals: int = 6,
) -> Tuple[ad.AnnData, ad.AnnData, pd.DataFrame]:
    """
    Align two adatas to the same cells.

    Priority:
      1) exact same obs_names
      2) intersection of obs_names
      3) exact coordinate matching after rounding

    Returns:
      gene_adata_aligned, protein_adata_aligned, coords_df
    """
    gene_coords = get_coords_from_adata(gene_adata, obsm_key=coord_obsm_key_gene)
    prot_coords = get_coords_from_adata(protein_adata, obsm_key=coord_obsm_key_protein)

    # Case 1: exact same obs_names in same order
    if gene_adata.n_obs == protein_adata.n_obs and np.array_equal(gene_adata.obs_names, protein_adata.obs_names):
        coords = gene_coords.loc[gene_adata.obs_names].copy()
        return gene_adata.copy(), protein_adata.copy(), coords.reset_index(drop=True)

    # Case 2: overlap by obs_names
    common = gene_adata.obs_names.intersection(protein_adata.obs_names)
    if len(common) > 0:
        common = common.sort_values()
        g = gene_adata[common].copy()
        p = protein_adata[common].copy()
        coords = gene_coords.loc[common].copy()
        return g, p, coords.reset_index(drop=True)

    # Case 3: align by coordinates
    gk = _coord_key(gene_coords, decimals=coord_round_decimals)
    pk = _coord_key(prot_coords, decimals=coord_round_decimals)

    if gk.duplicated().any():
        dup = int(gk.duplicated().sum())
        raise ValueError(f"Gene h5ad has {dup} duplicated coordinate keys; cannot align uniquely by coordinates.")
    if pk.duplicated().any():
        dup = int(pk.duplicated().sum())
        raise ValueError(f"Protein h5ad has {dup} duplicated coordinate keys; cannot align uniquely by coordinates.")

    gene_map = pd.DataFrame({"coord_key": gk.values, "gene_obs": gene_adata.obs_names.values})
    prot_map = pd.DataFrame({"coord_key": pk.values, "prot_obs": protein_adata.obs_names.values})

    merged = gene_map.merge(prot_map, on="coord_key", how="inner")
    if len(merged) == 0:
        raise ValueError("No overlapping cells found by obs_names or coordinates.")

    g = gene_adata[merged["gene_obs"].values].copy()
    p = protein_adata[merged["prot_obs"].values].copy()
    coords = get_coords_from_adata(g, obsm_key=coord_obsm_key_gene).copy()
    coords = coords.reset_index(drop=True)
    return g, p, coords


# ============================================================
# Data extraction
# ============================================================

def get_feature_names(adata: ad.AnnData) -> List[str]:
    if adata.var_names is None:
        return [f"feature_{i}" for i in range(adata.n_vars)]
    return [str(x) for x in adata.var_names]


def extract_gene_X(gene_adata: ad.AnnData, do_log1p: bool) -> np.ndarray:
    X = gene_adata.X
    if sp.issparse(X):
        X = X.tocsr().astype(np.float32)
        if do_log1p:
            X = X.copy()
            X.data = np.log1p(X.data)
        return X
    X = np.asarray(X, dtype=np.float32)
    if do_log1p:
        X = np.log1p(X)
    return X


def extract_protein_Y(protein_adata: ad.AnnData, do_log1p: bool) -> np.ndarray:
    Y = np.asarray(protein_adata.X, dtype=np.float32)
    if do_log1p:
        Y = np.log1p(np.clip(Y, a_min=0, a_max=None))
    return Y


# ============================================================
# Metrics
# ============================================================

def safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    try:
        if method == "pearson":
            return float(pearsonr(x, y)[0])
        if method == "spearman":
            return float(spearmanr(x, y)[0])
    except Exception:
        return np.nan
    raise ValueError("method must be pearson or spearman")


def per_protein_metrics(y_true: np.ndarray, y_pred: np.ndarray, protein_names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(protein_names):
        yt = np.asarray(y_true[:, j], dtype=float)
        yp = np.asarray(y_pred[:, j], dtype=float)

        rows.append({
            "protein": str(name),
            "pearson_r": safe_corr(yt, yp, "pearson"),
            "spearman_r": safe_corr(yt, yp, "spearman"),
            "r2": float(r2_score(yt, yp)) if np.std(yt) > 0 else np.nan,
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        })
    return pd.DataFrame(rows)


# ============================================================
# Grid spatial evaluation
# ============================================================

def grid_aggregate(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    n_bins: int = 60,
    agg: str = "mean",
    xy_range: Optional[Tuple[float, float, float, float]] = None,
    min_count: int = 4,
):
    x = np.asarray(x)
    y = np.asarray(y)
    v = np.asarray(v)

    if xy_range is None:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
    else:
        xmin, xmax, ymin, ymax = xy_range

    x_edges = np.linspace(xmin, xmax, n_bins + 1)
    y_edges = np.linspace(ymin, ymax, n_bins + 1)

    xi = np.searchsorted(x_edges, x, side="right") - 1
    yi = np.searchsorted(y_edges, y, side="right") - 1

    valid = (xi >= 0) & (xi < n_bins) & (yi >= 0) & (yi < n_bins) & np.isfinite(v)
    xi = xi[valid]
    yi = yi[valid]
    v = v[valid]

    counts = np.zeros((n_bins, n_bins), dtype=np.int32)
    sums = np.zeros((n_bins, n_bins), dtype=np.float64)

    np.add.at(counts, (yi, xi), 1)
    np.add.at(sums, (yi, xi), v)

    if agg == "mean":
        grid = sums / np.maximum(counts, 1)
    elif agg == "sum":
        grid = sums
    else:
        raise ValueError("agg must be mean or sum")

    grid = grid.astype(np.float64)
    grid[counts < min_count] = np.nan
    extent = (xmin, xmax, ymin, ymax)
    return grid, counts, extent


def grid_similarity_metrics(truth_grid: np.ndarray, pred_grid: np.ndarray) -> dict:
    t = truth_grid.ravel()
    p = pred_grid.ravel()
    m = np.isfinite(t) & np.isfinite(p)
    t = t[m]
    p = p[m]

    if t.size < 10:
        return {
            "n_grid_cells": int(t.size),
            "grid_pearson_r": np.nan,
            "grid_spearman_r": np.nan,
            "grid_r2": np.nan,
            "grid_mae": np.nan,
            "grid_rmse": np.nan,
        }

    return {
        "n_grid_cells": int(t.size),
        "grid_pearson_r": safe_corr(t, p, "pearson"),
        "grid_spearman_r": safe_corr(t, p, "spearman"),
        "grid_r2": float(r2_score(t, p)) if np.std(t) > 0 else np.nan,
        "grid_mae": float(mean_absolute_error(t, p)),
        "grid_rmse": float(np.sqrt(mean_squared_error(t, p))),
    }


# ============================================================
# Plotting
# ============================================================

def robust_vlim(*arrays, q=(1, 99)):
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(vals, q)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def plot_split_layout(coords: pd.DataFrame, split_labels: np.ndarray, out_png: str, invert_y: bool = True):
    plt.figure(figsize=(7.2, 6.4))
    colors = {
        "train": "#4C78A8",
        "val": "#F58518",
        "test": "#E45756",
    }
    for label in ["train", "val", "test"]:
        m = split_labels == label
        plt.scatter(coords.loc[m, "x"], coords.loc[m, "y"], s=0.1, alpha=0.65, label=label, c=colors[label])
    plt.gca().set_aspect("equal", adjustable="box")
    if invert_y:
        plt.gca().invert_yaxis()
    plt.title("Train / validation / test spatial layout")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(frameon=False, markerscale=2.5)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_metric_heatmap(metrics_df: pd.DataFrame, out_png: str):
    cols = ["pearson_r", "spearman_r", "r2", "mae", "rmse", "grid_pearson_r", "grid_spearman_r", "grid_r2"]
    keep = [c for c in cols if c in metrics_df.columns]
    dat = metrics_df[keep].to_numpy(dtype=float)

    plt.figure(figsize=(max(6, len(keep) * 1.1), max(6, len(metrics_df) * 0.28)))
    im = plt.imshow(dat, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.xticks(range(len(keep)), keep, rotation=45, ha="right")
    plt.yticks(range(len(metrics_df)), metrics_df["protein"].tolist())
    plt.title("Per-protein evaluation metrics")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_ranked_metric(metrics_df: pd.DataFrame, metric: str, out_png: str, higher_is_better: bool = True):
    df = metrics_df.copy()
    df = df.sort_values(metric, ascending=not higher_is_better)

    plt.figure(figsize=(8.8, max(5.0, 0.28 * len(df))))
    y = np.arange(len(df))
    plt.barh(y, df[metric].to_numpy(dtype=float))
    plt.yticks(y, df["protein"].tolist())
    plt.gca().invert_yaxis()
    plt.xlabel(metric)
    plt.title(f"Protein ranking by {metric}")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_case_spatial(
    coords: pd.DataFrame,
    values: np.ndarray,
    title: str,
    out_png: str,
    cmap: str = "viridis",
    invert_y: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    plt.figure(figsize=(6.4, 5.8))
    sc = plt.scatter(coords["x"], coords["y"], c=values, s=0.5, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    plt.gca().set_aspect("equal", adjustable="box")
    if invert_y:
        plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(sc, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_case_scatter(
    truth: np.ndarray,
    pred: np.ndarray,
    title: str,
    out_png: str,
    metric_text: Optional[str] = None,
):
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    vmin, vmax = robust_vlim(truth, pred, q=(0, 100))
    plt.figure(figsize=(5.8, 5.5))
    plt.scatter(truth, pred, s=0.1, alpha=0.35)
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    if metric_text is not None:
        plt.text(0.03, 0.97, metric_text, transform=plt.gca().transAxes,
                 va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.8"))
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_grid_image(
    grid: np.ndarray,
    extent: Tuple[float, float, float, float],
    title: str,
    out_png: str,
    cmap: str = "viridis",
    invert_y: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    plt.figure(figsize=(6.2, 5.7))
    plt.imshow(grid, origin="lower", extent=extent, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    if invert_y:
        plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ============================================================
# Optional paired-cell subsampling
# ============================================================

def subsample_aligned_cells(
    gene_adata: ad.AnnData,
    protein_adata: ad.AnnData,
    coords: pd.DataFrame,
    fraction: float = 1.0,
    seed: int = 0,
) -> Tuple[ad.AnnData, ad.AnnData, pd.DataFrame, np.ndarray]:
    """Randomly subsample aligned gene/protein cells while preserving pairing.

    Subsampling is applied *after* the two modalities have been aligned and
    *before* the train/validation/test split.  The same cell indices are used
    for the gene matrix, protein matrix, and spatial coordinates.  Sampling is
    deterministic for a fixed seed.

    Parameters
    ----------
    fraction
        Fraction of aligned cells to retain. Must satisfy 0 < fraction <= 1.
        ``fraction=1.0`` leaves the dataset unchanged.
    seed
        Random seed used only for selecting the retained paired cells.
    """
    fraction = float(fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            f"cell_subsample_fraction must be in (0, 1], got {fraction}."
        )

    n_total = int(gene_adata.n_obs)
    if int(protein_adata.n_obs) != n_total or len(coords) != n_total:
        raise ValueError(
            "Gene data, protein data, and coordinates must contain the same "
            "number of aligned cells before subsampling."
        )

    if fraction >= 1.0:
        keep_idx = np.arange(n_total, dtype=int)
        return gene_adata, protein_adata, coords.reset_index(drop=True), keep_idx

    n_keep = max(1, int(round(n_total * fraction)))
    rng = np.random.default_rng(int(seed))
    keep_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False)).astype(int)

    gene_sub = gene_adata[keep_idx].copy()
    protein_sub = protein_adata[keep_idx].copy()
    coords_sub = coords.iloc[keep_idx].reset_index(drop=True).copy()
    return gene_sub, protein_sub, coords_sub, keep_idx


# ============================================================
# Refined two-stage validation-set hyperparameter search
# ============================================================

HYPERPARAM_SELECTION_METRIC = "rmse"


def _unique_dicts(dicts: Sequence[dict]) -> List[dict]:
    """De-duplicate dictionaries while preserving their original order."""
    out = []
    seen = set()
    for d in dicts:
        key = tuple(sorted(d.items()))
        if key not in seen:
            seen.add(key)
            out.append(dict(d))
    return out


def build_stage1_structure_candidates(args) -> List[dict]:
    """Build a focused, deterministic search over tree structure/regularization.

    The first validation-search run showed that most proteins preferred a more
    regularized region around depth~6, colsample~0.8, min_child_weight~3 and
    reg_lambda~3.  Stage 1 therefore searches that region systematically while
    retaining both higher-capacity and more strongly regularized alternatives.

    ``n_estimators`` and ``learning_rate`` are intentionally held fixed in this
    stage. ``subsample`` is also held fixed (default 1.0), because the pilot
    search did not provide evidence that row subsampling was beneficial.
    """
    refined = [
        # Empirical anchor from the pilot validation search.
        {"max_depth": 6, "colsample_bytree": 0.8, "min_child_weight": 3.0, "reg_lambda": 3.0, "reg_alpha": 0.0},

        # One-factor depth variations around the anchor.
        {"max_depth": 4, "colsample_bytree": 0.8, "min_child_weight": 3.0, "reg_lambda": 3.0, "reg_alpha": 0.0},
        {"max_depth": 5, "colsample_bytree": 0.8, "min_child_weight": 3.0, "reg_lambda": 3.0, "reg_alpha": 0.0},
        {"max_depth": 7, "colsample_bytree": 0.8, "min_child_weight": 3.0, "reg_lambda": 3.0, "reg_alpha": 0.0},
        {"max_depth": 8, "colsample_bytree": 0.8, "min_child_weight": 3.0, "reg_lambda": 3.0, "reg_alpha": 0.0},

        # Feature-subsampling variations.
        {"max_depth": 6, "colsample_bytree": 0.7, "min_child_weight": 3.0, "reg_lambda": 3.0, "reg_alpha": 0.0},
        {"max_depth": 6, "colsample_bytree": 0.9, "min_child_weight": 3.0, "reg_lambda": 3.0, "reg_alpha": 0.0},
        {"max_depth": 6, "colsample_bytree": 1.0, "min_child_weight": 3.0, "reg_lambda": 3.0, "reg_alpha": 0.0},

        # Leaf-complexity variations.
        {"max_depth": 6, "colsample_bytree": 0.8, "min_child_weight": 1.0, "reg_lambda": 3.0, "reg_alpha": 0.0},
        {"max_depth": 6, "colsample_bytree": 0.8, "min_child_weight": 5.0, "reg_lambda": 3.0, "reg_alpha": 0.0},
        {"max_depth": 6, "colsample_bytree": 0.8, "min_child_weight": 8.0, "reg_lambda": 3.0, "reg_alpha": 0.0},

        # L2 regularization variations.
        {"max_depth": 6, "colsample_bytree": 0.8, "min_child_weight": 3.0, "reg_lambda": 1.0, "reg_alpha": 0.0},
        {"max_depth": 6, "colsample_bytree": 0.8, "min_child_weight": 3.0, "reg_lambda": 5.0, "reg_alpha": 0.0},
        {"max_depth": 6, "colsample_bytree": 0.8, "min_child_weight": 3.0, "reg_lambda": 8.0, "reg_alpha": 0.0},

        # Mild L1 regularization.
        {"max_depth": 6, "colsample_bytree": 0.8, "min_child_weight": 3.0, "reg_lambda": 3.0, "reg_alpha": 0.1},

        # Joint regimes to allow genuinely protein-specific solutions.
        # Higher-capacity model: useful for broad/structured targets.
        {"max_depth": 8, "colsample_bytree": 0.8, "min_child_weight": 1.0, "reg_lambda": 1.0, "reg_alpha": 0.1},
        # Moderately flexible model.
        {"max_depth": 7, "colsample_bytree": 0.9, "min_child_weight": 2.0, "reg_lambda": 2.0, "reg_alpha": 0.05},
        # Conservative model.
        {"max_depth": 5, "colsample_bytree": 0.7, "min_child_weight": 5.0, "reg_lambda": 5.0, "reg_alpha": 0.1},
        # Strongly regularized model: useful for sparse/noisy targets.
        {"max_depth": 4, "colsample_bytree": 0.7, "min_child_weight": 8.0, "reg_lambda": 8.0, "reg_alpha": 0.3},
    ]

    # Preserve the original command-line configuration as an explicit candidate
    # so a user-specified setting can still win if it validates better.
    user_baseline = {
        "max_depth": int(args.max_depth),
        "colsample_bytree": float(args.colsample_bytree),
        "min_child_weight": float(args.min_child_weight),
        "reg_lambda": float(args.reg_lambda),
        "reg_alpha": float(args.reg_alpha),
    }
    return _unique_dicts([user_baseline] + refined)


def build_stage2_boosting_schedules(args) -> List[dict]:
    """Build a small schedule search around the original boosting rate/budget.

    For the default settings this evaluates approximately:
      1200 / 0.07, 1600 / 0.06, 2000 / 0.05,
      2500 / 0.04, and 3000 / 0.03
    for (n_estimators / learning_rate).
    """
    schedules = [
        {
            "n_estimators": max(200, int(round(args.n_estimators * 0.60))),
            "learning_rate": min(0.30, max(0.005, float(args.learning_rate) * 1.40)),
        },
        {
            "n_estimators": max(200, int(round(args.n_estimators * 0.80))),
            "learning_rate": min(0.30, max(0.005, float(args.learning_rate) * 1.20)),
        },
        {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
        },
        {
            "n_estimators": max(200, int(round(args.n_estimators * 1.25))),
            "learning_rate": min(0.30, max(0.005, float(args.learning_rate) * 0.80)),
        },
        {
            "n_estimators": max(200, int(round(args.n_estimators * 1.50))),
            "learning_rate": min(0.30, max(0.005, float(args.learning_rate) * 0.60)),
        },
    ]
    return _unique_dicts(schedules)


def _validation_rmse_for_params(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    params: dict,
) -> float:
    """Fit one candidate on the training set and return validation RMSE."""
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    pred_val = model.predict(X_val)
    score = float(np.sqrt(mean_squared_error(y_val, pred_val)))
    del model
    return score


def select_xgb_params_two_stage(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    base_params: dict,
    args,
    structure_candidates: Sequence[dict],
    boosting_schedules: Sequence[dict],
) -> Tuple[dict, float, float, List[dict], List[dict]]:
    """Select protein-specific XGBoost parameters in two validation stages.

    Stage 1 searches tree structure/regularization while fixing the boosting
    schedule to the user-supplied n_estimators and learning_rate.
    Stage 2 keeps the selected structure fixed and refines the boosting schedule.

    The test set is never used here. No early stopping is used.
    """
    stage1_scores = []
    best_structure = None
    best_stage1_rmse = np.inf

    fixed_stage1 = {
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "subsample": float(args.subsample),
    }

    for candidate_id, structure in enumerate(structure_candidates):
        params = dict(base_params)
        params.update(fixed_stage1)
        params.update(structure)
        score = _validation_rmse_for_params(
            X_train, y_train, X_val, y_val, params
        )
        stage1_scores.append({
            "candidate_id": int(candidate_id),
            **{k: (int(v) if k == "max_depth" else float(v)) for k, v in structure.items()},
            "validation_rmse": float(score),
        })
        if np.isfinite(score) and score < best_stage1_rmse:
            best_stage1_rmse = score
            best_structure = dict(structure)

    if best_structure is None:
        raise RuntimeError("Stage-1 validation search failed to produce a finite RMSE.")

    # Stage 2 starts with the already-evaluated default schedule, avoiding one
    # redundant model fit. Other schedules are evaluated with the selected
    # protein-specific structure fixed.
    default_schedule = {
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
    }
    best_schedule = dict(default_schedule)
    best_final_rmse = float(best_stage1_rmse)
    stage2_scores = []

    for schedule_id, schedule in enumerate(boosting_schedules):
        is_default = (
            int(schedule["n_estimators"]) == int(default_schedule["n_estimators"])
            and np.isclose(float(schedule["learning_rate"]), float(default_schedule["learning_rate"]))
        )
        if is_default:
            score = float(best_stage1_rmse)
        else:
            params = dict(base_params)
            params.update({
                "subsample": float(args.subsample),
                **best_structure,
                **schedule,
            })
            score = _validation_rmse_for_params(
                X_train, y_train, X_val, y_val, params
            )

        stage2_scores.append({
            "schedule_id": int(schedule_id),
            "n_estimators": int(schedule["n_estimators"]),
            "learning_rate": float(schedule["learning_rate"]),
            "validation_rmse": float(score),
        })
        if np.isfinite(score) and score < best_final_rmse:
            best_final_rmse = score
            best_schedule = dict(schedule)

    selected = {
        "n_estimators": int(best_schedule["n_estimators"]),
        "learning_rate": float(best_schedule["learning_rate"]),
        "max_depth": int(best_structure["max_depth"]),
        "subsample": float(args.subsample),
        "colsample_bytree": float(best_structure["colsample_bytree"]),
        "min_child_weight": float(best_structure["min_child_weight"]),
        "reg_lambda": float(best_structure["reg_lambda"]),
        "reg_alpha": float(best_structure["reg_alpha"]),
    }
    return selected, float(best_final_rmse), float(best_stage1_rmse), stage1_scores, stage2_scores


# ============================================================
# Main training / evaluation
# ============================================================

def main():
    setup_matplotlib()

    parser = argparse.ArgumentParser(description="Predict protein reads from Xenium gene reads using XGBoost.")
    parser.add_argument("--gene_h5ad", type=str, default="./presta_pair/LUAD_2_A_xenium_0.000.h5ad", help="Input Xenium h5ad (genes in X).")
    parser.add_argument("--protein_h5ad", type=str, default="./virtual_pair/virtual_protein_1.000.h5ad", help="Input protein h5ad (proteins in X).")
    parser.add_argument("--outdir", type=str, default="./virtual_pair/xgb_prediction_presta_x_virtual_protein_test0.4/")

    parser.add_argument("--coord_obsm_key_gene", type=str, default="spatial")
    parser.add_argument("--coord_obsm_key_protein", type=str, default="spatial")
    parser.add_argument("--coord_round_decimals", type=int, default=6)

    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--val_size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cell_subsample_fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of aligned paired cells retained before the train/val/test "
            "split. Use a value such as 0.10 or 0.20 for fast search-strategy "
            "experiments; 1.0 uses the full dataset."
        ),
    )

    parser.add_argument("--log1p_gene",default=True, help="Apply log1p to gene matrix.")
    parser.add_argument("--log1p_protein", default=True, help="Apply log1p to protein matrix.")

    parser.add_argument("--use_gpu", default=True)
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--min_child_weight", type=float, default=0.3)
    parser.add_argument("--reg_lambda", type=float, default=0.3)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--early_stopping_rounds", type=int, default=150)
    parser.add_argument("--n_jobs", type=int, default=8)

    parser.add_argument("--case_study_proteins", nargs="*", default=None,
                        help="Protein names for detailed plots. If omitted, top proteins by Pearson are used.")
    parser.add_argument("--n_case_study", type=int, default=20)
    parser.add_argument("--grid_bins", type=int, default=200)
    parser.add_argument("--grid_min_count", type=int, default=4)
    parser.add_argument("--invert_y", action="store_true")

    args = parser.parse_args()
    tm = Timer()

    ensure_dir(args.outdir)

    # -------------------------
    # Load and align
    # -------------------------
    tm.log(f"Loading gene h5ad: {args.gene_h5ad}")
    gene_adata = ad.read_h5ad(args.gene_h5ad)
    tm.log(f"Loading protein h5ad: {args.protein_h5ad}")
    protein_adata = ad.read_h5ad(args.protein_h5ad)

    tm.log("Aligning modalities...")
    gene_adata, protein_adata, coords = align_gene_and_protein_adatas(
        gene_adata,
        protein_adata,
        coord_obsm_key_gene=args.coord_obsm_key_gene,
        coord_obsm_key_protein=args.coord_obsm_key_protein,
        coord_round_decimals=args.coord_round_decimals,
    )
    n_cells_aligned_full = int(gene_adata.n_obs)
    tm.log(
        f"Aligned cells: {n_cells_aligned_full}; genes: {gene_adata.n_vars}; "
        f"proteins: {protein_adata.n_vars}"
    )

    # Optional fast-search mode: retain a deterministic random subset of paired
    # cells before constructing the train/validation/test split.  This changes
    # only the number of cells used in the current run; all downstream logic and
    # output formats remain the same.
    gene_adata, protein_adata, coords, subsample_idx = subsample_aligned_cells(
        gene_adata=gene_adata,
        protein_adata=protein_adata,
        coords=coords,
        fraction=args.cell_subsample_fraction,
        seed=args.seed,
    )
    if args.cell_subsample_fraction < 1.0:
        tm.log(
            f"Cell subsampling: retained {gene_adata.n_obs}/{n_cells_aligned_full} "
            f"paired cells ({gene_adata.n_obs / n_cells_aligned_full:.1%}); "
            f"seed={args.seed}"
        )
    else:
        tm.log("Cell subsampling disabled: using all aligned paired cells")

    gene_names = get_feature_names(gene_adata)
    protein_names = get_feature_names(protein_adata)
    #protein_names = protein_names

    X = extract_gene_X(gene_adata, do_log1p=args.log1p_gene)
    Y = extract_protein_Y(protein_adata, do_log1p=args.log1p_protein)

    n_cells = gene_adata.n_obs
    idx_all = np.arange(n_cells)

    # -------------------------
    # Split
    # -------------------------
    tm.log("Creating train/val/test split...")
    train_idx, test_idx = train_test_split(
        idx_all,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
    )

    split_labels = np.array(["train"] * n_cells, dtype=object)
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    X_train = X[train_idx] if sp.issparse(X) else X[train_idx, :]
    X_val = X[val_idx] if sp.issparse(X) else X[val_idx, :]
    X_test = X[test_idx] if sp.issparse(X) else X[test_idx, :]

    Y_train = Y[train_idx]
    Y_val = Y[val_idx]
    Y_test = Y[test_idx]

    plot_split_layout(
        coords=coords,
        split_labels=split_labels,
        out_png=os.path.join(args.outdir, "train_val_test_layout.png"),
        invert_y=args.invert_y,
    )

    # -------------------------
    # XGBoost config + validation-set hyperparameter search
    # -------------------------
    tree_method = "gpu_hist" if args.use_gpu else "hist"
    predictor = "gpu_predictor" if args.use_gpu else "auto"

    # Parameters that are not searched remain fixed throughout the run.
    base_params = dict(
        objective="reg:squarederror",
        tree_method=tree_method,
        predictor=predictor,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbosity=0,
    )
    structure_candidates = build_stage1_structure_candidates(args)
    boosting_schedules = build_stage2_boosting_schedules(args)
    n_additional_stage2_fits = max(0, len(boosting_schedules) - 1)
    tm.log(
        f"XGBoost refined validation search: tree_method={tree_method}; "
        f"stage 1={len(structure_candidates)} structure/regularization candidates; "
        f"stage 2={len(boosting_schedules)} boosting schedules "
        f"({n_additional_stage2_fits} additional fits because the default schedule is reused); "
        f"selection metric={HYPERPARAM_SELECTION_METRIC}; early stopping disabled"
    )

    # After validation-based model selection, refit the selected configuration
    # on train+validation cells before evaluating once on the untouched test set.
    trainval_idx = np.concatenate([train_idx, val_idx])
    X_trainval = X[trainval_idx] if sp.issparse(X) else X[trainval_idx, :]
    Y_trainval = Y[trainval_idx]

    # -------------------------
    # Train one model per protein
    # -------------------------
    n_proteins = Y.shape[1]
    Y_pred_test = np.zeros((len(test_idx), n_proteins), dtype=np.float32)
    Y_pred_all = np.zeros((n_cells, n_proteins), dtype=np.float32)
    best_iters = []  # retained for output compatibility; no early stopping is used
    validation_rmse = []
    stage1_validation_rmse = []
    selected_params_per_protein = []
    validation_search_details = {}

    for j, pname in enumerate(protein_names):
        t0 = time.time()

        best_search_params, best_val_rmse, best_stage1_rmse, stage1_scores, stage2_scores = select_xgb_params_two_stage(
            X_train=X_train,
            y_train=Y_train[:, j],
            X_val=X_val,
            y_val=Y_val[:, j],
            base_params=base_params,
            args=args,
            structure_candidates=structure_candidates,
            boosting_schedules=boosting_schedules,
        )

        final_params = dict(base_params)
        final_params.update(best_search_params)
        model = XGBRegressor(**final_params)
        model.fit(X_trainval, Y_trainval[:, j], verbose=False)

        Y_pred_test[:, j] = model.predict(X_test).astype(np.float32, copy=False)
        Y_pred_all[:, j] = model.predict(X).astype(np.float32, copy=False)

        best_iters.append(-1)
        validation_rmse.append(float(best_val_rmse))
        stage1_validation_rmse.append(float(best_stage1_rmse))
        selected_params_per_protein.append(dict(best_search_params))
        validation_search_details[str(pname)] = {
            "stage1_structure_scores": stage1_scores,
            "stage2_boosting_scores": stage2_scores,
        }
        tm.log(
            f"[{j+1}/{n_proteins}] {pname}: stage1 RMSE={best_stage1_rmse:.5f}; "
            f"final val RMSE={best_val_rmse:.5f}; selected={best_search_params}; "
            f"total={time.time() - t0:.1f}s"
        )

    # -------------------------
    # Metrics: cell-wise
    # -------------------------
    tm.log("Computing held-out cell-wise metrics...")
    metrics_df = per_protein_metrics(Y_test, Y_pred_test, protein_names)
    metrics_df["best_iteration"] = best_iters
    metrics_df["validation_rmse"] = validation_rmse
    metrics_df["stage1_validation_rmse"] = stage1_validation_rmse
    metrics_df["selected_n_estimators"] = [p["n_estimators"] for p in selected_params_per_protein]
    metrics_df["selected_learning_rate"] = [p["learning_rate"] for p in selected_params_per_protein]
    metrics_df["selected_max_depth"] = [p["max_depth"] for p in selected_params_per_protein]
    metrics_df["selected_subsample"] = [p["subsample"] for p in selected_params_per_protein]
    metrics_df["selected_colsample_bytree"] = [p["colsample_bytree"] for p in selected_params_per_protein]
    metrics_df["selected_min_child_weight"] = [p["min_child_weight"] for p in selected_params_per_protein]
    metrics_df["selected_reg_lambda"] = [p["reg_lambda"] for p in selected_params_per_protein]
    metrics_df["selected_reg_alpha"] = [p["reg_alpha"] for p in selected_params_per_protein]

    # -------------------------
    # Metrics: spatial grid on test split
    # -------------------------
    tm.log("Computing grid-based spatial metrics on held-out test cells...")
    x_test = coords.iloc[test_idx]["x"].to_numpy(dtype=np.float32)
    y_test = coords.iloc[test_idx]["y"].to_numpy(dtype=np.float32)
    xy_range_test = (
        float(np.min(x_test)),
        float(np.max(x_test)),
        float(np.min(y_test)),
        float(np.max(y_test)),
    )

    grid_rows = []
    for j, pname in enumerate(protein_names):
        truth = np.asarray(Y_test[:, j], dtype=float)
        pred = np.asarray(Y_pred_test[:, j], dtype=float)

        truth_grid, _, extent = grid_aggregate(
            x_test, y_test, truth,
            n_bins=args.grid_bins,
            agg="mean",
            xy_range=xy_range_test,
            min_count=args.grid_min_count,
        )
        pred_grid, _, _ = grid_aggregate(
            x_test, y_test, pred,
            n_bins=args.grid_bins,
            agg="mean",
            xy_range=xy_range_test,
            min_count=args.grid_min_count,
        )
        grid_rows.append({"protein": pname, **grid_similarity_metrics(truth_grid, pred_grid)})

    grid_df = pd.DataFrame(grid_rows)
    metrics_df = metrics_df.merge(grid_df, on="protein", how="left")
    metrics_df = metrics_df.sort_values(["pearson_r", "r2"], ascending=[False, False]).reset_index(drop=True)

    metrics_csv = os.path.join(args.outdir, "per_protein_metrics_test.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    plot_metric_heatmap(metrics_df, os.path.join(args.outdir, "metric_heatmap.png"))
    plot_ranked_metric(metrics_df, "pearson_r", os.path.join(args.outdir, "ranked_pearson.png"), higher_is_better=True)
    plot_ranked_metric(metrics_df, "r2", os.path.join(args.outdir, "ranked_r2.png"), higher_is_better=True)

    # -------------------------
    # Save predictions
    # -------------------------
    tm.log("Saving prediction h5ad outputs...")
    pred_all = ad.AnnData(
        X=np.asarray(Y_pred_all, dtype=np.float32),
        obs=pd.DataFrame(index=gene_adata.obs_names.copy()),
        var=pd.DataFrame(index=pd.Index(protein_names, name="protein")),
    )
    pred_all.obs["split"] = split_labels
    pred_all.obs["x"] = coords["x"].to_numpy()
    pred_all.obs["y"] = coords["y"].to_numpy()
    pred_all.layers["truth"] = np.asarray(Y, dtype=np.float32)
    pred_all.layers["abs_error"] = np.abs(np.asarray(Y_pred_all, dtype=np.float32) - np.asarray(Y, dtype=np.float32))
    pred_all.obsm["spatial"] = coords[["x", "y"]].to_numpy(dtype=np.float32)
    pred_all.uns["description"] = "Predicted protein reads from Xenium gene reads using XGBoost."
    pred_all.write_h5ad(os.path.join(args.outdir, "aligned_pred_all.h5ad"))

    pred_test = ad.AnnData(
        X=np.asarray(Y_pred_test, dtype=np.float32),
        obs=pd.DataFrame(index=gene_adata.obs_names[test_idx].copy()),
        var=pd.DataFrame(index=pd.Index(protein_names, name="protein")),
    )
    pred_test.obs["split"] = "test"
    pred_test.obs["x"] = coords.iloc[test_idx]["x"].to_numpy()
    pred_test.obs["y"] = coords.iloc[test_idx]["y"].to_numpy()
    pred_test.layers["truth"] = np.asarray(Y_test, dtype=np.float32)
    pred_test.layers["abs_error"] = np.abs(np.asarray(Y_pred_test, dtype=np.float32) - np.asarray(Y_test, dtype=np.float32))
    pred_test.obsm["spatial"] = coords.iloc[test_idx][["x", "y"]].to_numpy(dtype=np.float32)
    pred_test.write_h5ad(os.path.join(args.outdir, "aligned_pred_test_only.h5ad"))

    # -------------------------
    # Case-study proteins
    # -------------------------
    if args.case_study_proteins is not None and len(args.case_study_proteins) > 0:
        case_names = [p for p in args.case_study_proteins if p in protein_names]
    else:
        case_names = metrics_df.head(args.n_case_study)["protein"].tolist()

    tm.log(f"Generating case-study plots for: {case_names}")

    coords_test_df = coords.iloc[test_idx].reset_index(drop=True)
    for pname in case_names:
        j = protein_names.index(pname)
        truth = np.asarray(Y_test[:, j], dtype=float)
        pred = np.asarray(Y_pred_test[:, j], dtype=float)
        diff = pred - truth

        vmin, vmax = robust_vlim(truth, pred, q=(1, 99))
        dv = np.nanpercentile(np.abs(diff), 99)
        if not np.isfinite(dv) or dv == 0:
            dv = max(float(np.nanmax(np.abs(diff))), 1e-6)

        # cell-wise plots on test cells
        plot_case_spatial(
            coords_test_df, truth,
            title=f"{pname}: true protein (test cells)",
            out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_true.png"),
            cmap="viridis",
            invert_y=args.invert_y,
            vmin=vmin, vmax=vmax,
        )
        plot_case_spatial(
            coords_test_df, pred,
            title=f"{pname}: predicted protein (test cells)",
            out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_pred.png"),
            cmap="viridis",
            invert_y=args.invert_y,
            vmin=vmin, vmax=vmax,
        )
        plot_case_spatial(
            coords_test_df, diff,
            title=f"{pname}: predicted - true (test cells)",
            out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_diff.png"),
            cmap="coolwarm",
            invert_y=args.invert_y,
            vmin=-dv, vmax=dv,
        )

        # scatter with annotations
        row = metrics_df.loc[metrics_df["protein"] == pname].iloc[0]
        metric_text = (
            f"Pearson = {row['pearson_r']:.3f}\n"
            f"Spearman = {row['spearman_r']:.3f}\n"
            f"R² = {row['r2']:.3f}\n"
            f"MAE = {row['mae']:.3f}\n"
            f"RMSE = {row['rmse']:.3f}\n"
            f"Grid Pearson = {row['grid_pearson_r']:.3f}"
        )
        plot_case_scatter(
            truth, pred,
            title=f"{pname}: true vs predicted (test cells)",
            out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_scatter.png"),
            metric_text=metric_text,
        )

        # grid plots
        truth_grid, _, extent = grid_aggregate(
            x_test, y_test, truth,
            n_bins=args.grid_bins,
            agg="mean",
            xy_range=xy_range_test,
            min_count=args.grid_min_count,
        )
        pred_grid, _, _ = grid_aggregate(
            x_test, y_test, pred,
            n_bins=args.grid_bins,
            agg="mean",
            xy_range=xy_range_test,
            min_count=args.grid_min_count,
        )
        diff_grid = pred_grid - truth_grid

        gvmin, gvmax = robust_vlim(
            truth_grid[np.isfinite(truth_grid)],
            pred_grid[np.isfinite(pred_grid)],
            q=(1, 99),
        )
        gdv = np.nanpercentile(np.abs(diff_grid[np.isfinite(diff_grid)]), 99) if np.isfinite(diff_grid).any() else 1.0
        if not np.isfinite(gdv) or gdv == 0:
            gdv = 1.0

        plot_grid_image(
            truth_grid, extent,
            title=f"{pname}: grid-aggregated true",
            out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_grid_true.png"),
            cmap="viridis", invert_y=args.invert_y, vmin=gvmin, vmax=gvmax,
        )
        plot_grid_image(
            pred_grid, extent,
            title=f"{pname}: grid-aggregated predicted",
            out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_grid_pred.png"),
            cmap="viridis", invert_y=args.invert_y, vmin=gvmin, vmax=gvmax,
        )
        plot_grid_image(
            diff_grid, extent,
            title=f"{pname}: grid-aggregated predicted - true",
            out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_grid_diff.png"),
            cmap="coolwarm", invert_y=args.invert_y, vmin=-gdv, vmax=gdv,
        )

    # -------------------------
    # Summary
    # -------------------------
    summary = {
        "n_cells_aligned_before_subsampling": int(n_cells_aligned_full),
        "cell_subsample_fraction_requested": float(args.cell_subsample_fraction),
        "n_cells_total": int(n_cells),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_genes": int(X.shape[1]),
        "n_proteins": int(Y.shape[1]),
        "gene_h5ad": args.gene_h5ad,
        "protein_h5ad": args.protein_h5ad,
        "log1p_gene": bool(args.log1p_gene),
        "log1p_protein": bool(args.log1p_protein),
        "case_study_proteins": case_names,
        "xgb_params": {
            **base_params,
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "min_child_weight": float(args.min_child_weight),
            "reg_lambda": float(args.reg_lambda),
            "reg_alpha": float(args.reg_alpha),
        },
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "early_stopping_used": False,
        "hyperparameter_selection": {
            "method": "two_stage_validation_set_search_per_protein",
            "selection_metric": HYPERPARAM_SELECTION_METRIC,
            "stage1": {
                "description": "structure_and_regularization",
                "fixed_n_estimators": int(args.n_estimators),
                "fixed_learning_rate": float(args.learning_rate),
                "fixed_subsample": float(args.subsample),
                "n_candidates": int(len(structure_candidates)),
                "candidate_space": structure_candidates,
            },
            "stage2": {
                "description": "boosting_schedule",
                "n_schedules": int(len(boosting_schedules)),
                "candidate_space": boosting_schedules,
            },
            "refit_after_selection": "train_plus_validation",
            "test_used_for_selection": False,
        },
        "validation_search_details_by_protein": validation_search_details,
        "selected_hyperparameters_by_protein": {
            str(name): {
                **selected_params_per_protein[i],
                "validation_rmse": float(validation_rmse[i]),
            }
            for i, name in enumerate(protein_names)
        },
        "metric_summary": {
            "median_pearson_r": float(np.nanmedian(metrics_df["pearson_r"])),
            "median_spearman_r": float(np.nanmedian(metrics_df["spearman_r"])),
            "median_r2": float(np.nanmedian(metrics_df["r2"])),
            "median_mae": float(np.nanmedian(metrics_df["mae"])),
            "median_rmse": float(np.nanmedian(metrics_df["rmse"])),
            "median_grid_pearson_r": float(np.nanmedian(metrics_df["grid_pearson_r"])),
            "median_grid_r2": float(np.nanmedian(metrics_df["grid_r2"])),
        },
    }

    with open(os.path.join(args.outdir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    tm.log("Done.")


if __name__ == "__main__":
    main()
