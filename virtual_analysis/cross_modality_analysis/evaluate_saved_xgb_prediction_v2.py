#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate and plot saved XGBoost cross-modality protein predictions.

Professional recommendation implemented
---------------------------------------
When prediction and truth have different dynamic ranges, do NOT replace the
raw evaluation with z-scored evaluation only.

Instead, evaluate in three complementary views:

1. Raw-scale fidelity
   - Pearson / Spearman
   - R2 / MAE / RMSE
   - grid-level metrics
   These preserve the original amplitude scale.

2. Pattern-only normalized fidelity
   - z-score truth and prediction separately per protein
   - evaluate zsep_R2 / zsep_MAE / zsep_RMSE
   This asks whether spatial/protein patterns match after removing scale and
   mean differences.

3. Calibration / amplitude diagnostics
   - Lin's concordance correlation coefficient (CCC)
   - prediction-vs-truth slope
   - intercept
   - std_ratio = pred_std / true_std
   These tell you whether predictions are over-smoothed or range-compressed.

This script reads saved prediction files only. It does not retrain any model.
"""

import os
import re
import json
import argparse
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr


# ============================================================
# helpers
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


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


def to_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def get_coords(adata: ad.AnnData) -> pd.DataFrame:

    if "spatial" in adata.obsm:
        arr = np.asarray(adata.obsm["spatial"])
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"obsm['spatial'] has invalid shape: {arr.shape}")
        return pd.DataFrame({"x": arr[:, 1].astype(float), "y": arr[:, 0].astype(float)}, index=adata.obs_names)

    if "x" in adata.obs.columns and "y" in adata.obs.columns:
        return pd.DataFrame({
            "x": pd.to_numeric(adata.obs["x"], errors="coerce"),
            "y": pd.to_numeric(adata.obs["y"], errors="coerce"),
        }, index=adata.obs_names)

    raise ValueError("Could not find coordinates in obsm['spatial'] or obs['x']/obs['y'].")


def robust_vlim(*arrays, q=(1, 99)):
    vals = []
    for a in arrays:
        a = np.asarray(a).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if len(vals) == 0:
        return 0.0, 1.0
    vals = np.concatenate(vals)
    vmin, vmax = np.percentile(vals, q)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    try:
        if method == "pearson":
            return float(pearsonr(x, y)[0])
        elif method == "spearman":
            return float(spearmanr(x, y)[0])
    except Exception:
        return np.nan
    raise ValueError("method must be pearson or spearman")


def zscore_ref(v: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if (not np.isfinite(sigma)) or sigma <= 1e-12:
        sigma = 1.0
    return (np.asarray(v, dtype=float) - float(mu)) / sigma


def zscore_self(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    mu = np.nanmean(v)
    sigma = np.nanstd(v)
    if (not np.isfinite(sigma)) or sigma <= 1e-12:
        sigma = 1.0
    return (v - mu) / sigma


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return np.nan
    mux = np.mean(x)
    muy = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    if vx <= 0 and vy <= 0:
        return np.nan
    cov = np.mean((x - mux) * (y - muy))
    denom = vx + vy + (mux - muy) ** 2
    if denom <= 0:
        return np.nan
    return float(2.0 * cov / denom)


def calibration_line(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    if yt.size < 3 or np.std(yt) <= 1e-12:
        return np.nan, np.nan
    slope = np.cov(yt, yp, ddof=0)[0, 1] / np.var(yt)
    intercept = float(np.mean(yp) - slope * np.mean(yt))
    return float(slope), intercept


# ============================================================
# grid metrics
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


def grid_similarity_metrics(truth_grid: np.ndarray, pred_grid: np.ndarray, prefix: str = "grid") -> dict:
    t = truth_grid.ravel()
    p = pred_grid.ravel()
    m = np.isfinite(t) & np.isfinite(p)
    t = t[m]
    p = p[m]

    if t.size < 10:
        return {
            f"{prefix}_n_cells": int(t.size),
            f"{prefix}_pearson_r": np.nan,
            f"{prefix}_spearman_r": np.nan,
            f"{prefix}_r2": np.nan,
            f"{prefix}_mae": np.nan,
            f"{prefix}_rmse": np.nan,
            f"{prefix}_ccc": np.nan,
        }

    return {
        f"{prefix}_n_cells": int(t.size),
        f"{prefix}_pearson_r": safe_corr(t, p, "pearson"),
        f"{prefix}_spearman_r": safe_corr(t, p, "spearman"),
        f"{prefix}_r2": float(r2_score(t, p)) if np.std(t) > 0 else np.nan,
        f"{prefix}_mae": float(mean_absolute_error(t, p)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(t, p))),
        f"{prefix}_ccc": lin_ccc(t, p),
    }


# ============================================================
# metrics
# ============================================================

def compute_per_protein_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protein_names: Sequence[str],
    coords: pd.DataFrame,
    grid_bins: int,
    grid_min_count: int,
) -> pd.DataFrame:
    x = coords["x"].to_numpy(dtype=float)
    y = coords["y"].to_numpy(dtype=float)
    xy_range = (float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y)))

    rows = []
    for j, pname in enumerate(protein_names):
        yt = np.asarray(y_true[:, j], dtype=float)
        yp = np.asarray(y_pred[:, j], dtype=float)

        mu_t = float(np.nanmean(yt))
        sd_t = float(np.nanstd(yt))
        mu_p = float(np.nanmean(yp))
        sd_p = float(np.nanstd(yp))

        # truth-referenced z-score
        yt_zref = zscore_ref(yt, mu_t, sd_t)
        yp_zref = zscore_ref(yp, mu_t, sd_t)

        # separately normalized z-score
        yt_zsep = zscore_self(yt)
        yp_zsep = zscore_self(yp)

        slope, intercept = calibration_line(yt, yp)

        row = {
            "protein": str(pname),

            # raw-scale
            "pearson_r": safe_corr(yt, yp, "pearson"),
            "spearman_r": safe_corr(yt, yp, "spearman"),
            "ccc": lin_ccc(yt, yp),
            "r2": float(r2_score(yt, yp)) if np.std(yt) > 0 else np.nan,
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),

            # diagnostics
            "true_mean": mu_t,
            "pred_mean": mu_p,
            "true_std": sd_t,
            "pred_std": sd_p,
            "std_ratio": float(sd_p / sd_t) if sd_t > 1e-12 else np.nan,
            "slope": slope,
            "intercept": intercept,

            # truth-referenced z-score metrics
            "zref_r2": float(r2_score(yt_zref, yp_zref)) if np.std(yt_zref) > 0 else np.nan,
            "zref_mae": float(mean_absolute_error(yt_zref, yp_zref)),
            "zref_rmse": float(np.sqrt(mean_squared_error(yt_zref, yp_zref))),

            # separately normalized pattern metrics
            "zsep_r2": float(r2_score(yt_zsep, yp_zsep)) if np.std(yt_zsep) > 0 else np.nan,
            "zsep_mae": float(mean_absolute_error(yt_zsep, yp_zsep)),
            "zsep_rmse": float(np.sqrt(mean_squared_error(yt_zsep, yp_zsep))),
        }

        # raw grid metrics
        truth_grid, _, _ = grid_aggregate(
            x, y, yt, n_bins=grid_bins, agg="mean",
            xy_range=xy_range, min_count=grid_min_count
        )
        pred_grid, _, _ = grid_aggregate(
            x, y, yp, n_bins=grid_bins, agg="mean",
            xy_range=xy_range, min_count=grid_min_count
        )
        row.update(grid_similarity_metrics(truth_grid, pred_grid, prefix="grid"))

        # separately normalized grid metrics: pattern only
        truth_grid_zsep, _, _ = grid_aggregate(
            x, y, yt_zsep, n_bins=grid_bins, agg="mean",
            xy_range=xy_range, min_count=grid_min_count
        )
        pred_grid_zsep, _, _ = grid_aggregate(
            x, y, yp_zsep, n_bins=grid_bins, agg="mean",
            xy_range=xy_range, min_count=grid_min_count
        )
        row.update(grid_similarity_metrics(truth_grid_zsep, pred_grid_zsep, prefix="zsep_grid"))

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["zsep_r2", "pearson_r", "ccc"], ascending=[False, False, False]).reset_index(drop=True)
    return df


# ============================================================
# plots
# ============================================================

def plot_layout(coords: pd.DataFrame, split: Optional[pd.Series], out_png: str, invert_y: bool = True):
    plt.figure(figsize=(7.2, 6.4))
    if split is not None and split.nunique() > 1:
        colors = {"train": "#4C78A8", "val": "#F58518", "test": "#E45756"}
        for label in pd.unique(split):
            m = split == label
            c = colors.get(str(label), None)
            plt.scatter(coords.loc[m, "x"], coords.loc[m, "y"], s=0.1, alpha=0.65, label=str(label), c=c)
        plt.legend(frameon=False, markerscale=2.5)
        title = "Train / validation / test spatial layout"
    else:
        plt.scatter(coords["x"], coords["y"], s=0.1, alpha=0.65)
        title = "Evaluation spatial layout"

    plt.gca().set_aspect("equal", adjustable="box")
    if invert_y:
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_metric_heatmap(metrics_df: pd.DataFrame, out_png: str):
    cols = [
        "pearson_r", "ccc", "r2", "zsep_r2",
        "mae", "rmse", "zsep_rmse",
        "grid_pearson_r", "zsep_grid_pearson_r",
        "std_ratio", "slope",
    ]
    keep = [c for c in cols if c in metrics_df.columns]
    dat = metrics_df[keep].to_numpy(dtype=float)

    plt.figure(figsize=(max(8.0, len(keep) * 1.0), max(6.0, len(metrics_df) * 0.28)))
    im = plt.imshow(dat, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.xticks(range(len(keep)), keep, rotation=45, ha="right")
    plt.yticks(range(len(metrics_df)), metrics_df["protein"].tolist())
    plt.title("Per-protein evaluation metrics: raw, normalized, and calibration")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_ranked_metric(metrics_df: pd.DataFrame, metric: str, out_png: str, higher_is_better: bool = True):
    df = metrics_df.copy().sort_values(metric, ascending=not higher_is_better)

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


def plot_global_scatter_panels(y_true: np.ndarray, y_pred: np.ndarray, metrics_df: pd.DataFrame, out_png: str, normalized: bool = False):
    top = metrics_df.head(min(9, len(metrics_df)))["protein"].tolist()
    protein_order = [str(x) for x in metrics_df["protein"].tolist()]
    name_to_idx = {p: i for i, p in enumerate(protein_order)}

    n = len(top)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.2 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, pname in zip(axes, top):
        j = name_to_idx[pname]
        yt = np.asarray(y_true[:, j], dtype=float)
        yp = np.asarray(y_pred[:, j], dtype=float)

        if normalized:
            yt_plot = zscore_self(yt)
            yp_plot = zscore_self(yp)
            xlabel, ylabel = "True z-score", "Predicted z-score"
            row = metrics_df.loc[metrics_df["protein"] == pname].iloc[0]
            txt = f"Pearson={row['pearson_r']:.2f}\nzsep_R²={row['zsep_r2']:.2f}"
        else:
            yt_plot = yt
            yp_plot = yp
            xlabel, ylabel = "True", "Predicted"
            row = metrics_df.loc[metrics_df["protein"] == pname].iloc[0]
            txt = f"CCC={row['ccc']:.2f}\nR²={row['r2']:.2f}"

        vmin, vmax = robust_vlim(yt_plot, yp_plot, q=(0, 100))
        ax.scatter(yt_plot, yp_plot, s=0.1, alpha=0.30)
        ax.plot([vmin, vmax], [vmin, vmax], linestyle="--")
        ax.set_title(pname)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(
            0.03, 0.97, txt,
            transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="0.8")
        )

    for ax in axes[n:]:
        ax.axis("off")

    title = "Top proteins: normalized true vs predicted" if normalized else "Top proteins: true vs predicted"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


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
    sc = plt.scatter(coords["x"], coords["y"], c=values, s=1, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    plt.gca().set_aspect("equal", adjustable="box")
    if invert_y:
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(sc, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_case_scatter(truth: np.ndarray, pred: np.ndarray, title: str, out_png: str, metric_text: Optional[str] = None):
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    vmin, vmax = robust_vlim(truth, pred, q=(0, 100))
    plt.figure(figsize=(5.8, 5.5))
    plt.scatter(truth, pred, s=0.05, alpha=0.35)
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    if metric_text is not None:
        plt.text(
            0.03, 0.97, metric_text,
            transform=plt.gca().transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.8"),
        )
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
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_rank_curve(truth: np.ndarray, pred: np.ndarray, title: str, out_png: str, normalized: bool = False):
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if normalized:
        truth = zscore_self(truth)
        pred = zscore_self(pred)

    order = np.argsort(-truth)
    truth_sorted = truth[order]
    pred_sorted = pred[order]

    plt.figure(figsize=(6.4, 5.2))
    plt.plot(truth_sorted, label="True", linewidth=2.0)
    plt.plot(pred_sorted, label="Predicted", linewidth=2.0, linestyle="--")
    plt.xlabel("Cells ranked by true expression")
    plt.ylabel("Expression" if not normalized else "Expression z-score")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ============================================================
# main
# ============================================================

def main():
    setup_matplotlib()

    parser = argparse.ArgumentParser(description="Evaluate and plot saved XGBoost cross-modality predictions.")
    parser.add_argument("--pred_h5ad", type=str, default="/media/huifang/data/registration/phenocycler/cross_modality_prediction/xgb_out/xgb_prediction_presta_x_virtual_protein_test0.2/aligned_pred_test_only.h5ad",
                        help="Saved prediction h5ad, e.g. aligned_pred_test_only.h5ad or aligned_pred_all.h5ad")
    parser.add_argument("--metrics_csv", type=str, default=None,
                        help="Optional existing metrics CSV. The script will still recompute metrics.")
    parser.add_argument("--outdir", type=str, default="/media/huifang/data/registration/phenocycler/cross_modality_prediction/xgb_out/xgb_prediction_presta_x_virtual_protein_test0.2/evaluation")
    parser.add_argument("--case_study_proteins", nargs="*", default=None,
                        help="Protein names for detailed plots. If omitted, top proteins by zsep_R2 are used.")
    parser.add_argument("--n_case_study", type=int, default=36)
    parser.add_argument("--grid_bins", type=int, default=200)
    parser.add_argument("--grid_min_count", type=int, default=4)
    parser.add_argument("--invert_y", default=True)
    args = parser.parse_args()

    ensure_dir(args.outdir)

    adata = ad.read_h5ad(args.pred_h5ad)
    if "truth" not in adata.layers:
        raise ValueError("Prediction h5ad must contain adata.layers['truth'].")

    coords = get_coords(adata)
    y_pred = to_dense(adata.X).astype(np.float32)
    y_true = to_dense(adata.layers["truth"]).astype(np.float32)
    protein_names = [str(x) for x in adata.var_names.tolist()]

    if y_pred.shape != y_true.shape:
        raise ValueError(f"Prediction and truth shapes differ: {y_pred.shape} vs {y_true.shape}")

    split = adata.obs["split"] if "split" in adata.obs.columns else None
    if split is not None and pd.Series(split).nunique() > 1 and "test" in set(pd.Series(split).astype(str)):
        m_eval = pd.Series(split).astype(str).to_numpy() == "test"
        eval_coords = coords.loc[m_eval].reset_index(drop=True)
        eval_true = y_true[m_eval]
        eval_pred = y_pred[m_eval]
    else:
        eval_coords = coords.reset_index(drop=True)
        eval_true = y_true
        eval_pred = y_pred

    metrics_df = compute_per_protein_metrics(
        eval_true, eval_pred, protein_names, eval_coords,
        grid_bins=args.grid_bins, grid_min_count=args.grid_min_count
    )
    metrics_df.to_csv(os.path.join(args.outdir, "per_protein_metrics_recomputed.csv"), index=False)

    if args.metrics_csv is not None and os.path.exists(args.metrics_csv):
        old_df = pd.read_csv(args.metrics_csv)
        merge_cols = [c for c in ["protein", "pearson_r", "r2", "grid_pearson_r"] if c in old_df.columns]
        if "protein" in merge_cols:
            merged = metrics_df.merge(
                old_df[merge_cols].rename(columns={c: f"old_{c}" for c in merge_cols if c != "protein"}),
                on="protein",
                how="left",
            )
            merged.to_csv(os.path.join(args.outdir, "metrics_with_previous_reference.csv"), index=False)

    plot_layout(
        coords=coords,
        split=split if split is None else pd.Series(split).astype(str),
        out_png=os.path.join(args.outdir, "split_layout_or_eval_layout.png"),
        invert_y=args.invert_y,
    )

    plot_metric_heatmap(metrics_df, os.path.join(args.outdir, "metric_heatmap_normalized.png"))
    plot_ranked_metric(metrics_df, "pearson_r", os.path.join(args.outdir, "ranked_pearson.png"), higher_is_better=True)
    # plot_ranked_metric(metrics_df, "ccc", os.path.join(args.outdir, "ranked_ccc.png"), higher_is_better=True)
    plot_ranked_metric(metrics_df, "zsep_r2", os.path.join(args.outdir, "ranked_zsep_r2.png"), higher_is_better=True)
    plot_ranked_metric(metrics_df, "std_ratio", os.path.join(args.outdir, "ranked_std_ratio.png"), higher_is_better=True)
    plot_ranked_metric(metrics_df, "grid_pearson_r", os.path.join(args.outdir, "ranked_grid_pearson.png"), higher_is_better=True)

    plot_global_scatter_panels(eval_true, eval_pred, metrics_df, os.path.join(args.outdir, "global_scatter_panels_raw.png"), normalized=False)
    plot_global_scatter_panels(eval_true, eval_pred, metrics_df, os.path.join(args.outdir, "global_scatter_panels_zscore.png"), normalized=True)

    if args.case_study_proteins is not None and len(args.case_study_proteins) > 0:
        case_names = [p for p in args.case_study_proteins if p in protein_names]
    else:
        case_names = metrics_df.head(args.n_case_study)["protein"].tolist()

    x_eval = eval_coords["x"].to_numpy(dtype=float)
    y_eval = eval_coords["y"].to_numpy(dtype=float)
    xy_range = (float(np.min(x_eval)), float(np.max(x_eval)), float(np.min(y_eval)), float(np.max(y_eval)))
    name_to_idx = {p: i for i, p in enumerate(protein_names)}

    for pname in case_names:
        j = name_to_idx[pname]
        truth = np.asarray(eval_true[:, j], dtype=float)
        pred = np.asarray(eval_pred[:, j], dtype=float)
        diff = pred - truth

        truth_z = zscore_self(truth)
        pred_z = zscore_self(pred)
        diff_z = pred_z - truth_z

        vmin, vmax = robust_vlim(truth, pred, q=(1, 99))
        dv = np.nanpercentile(np.abs(diff), 99)
        if not np.isfinite(dv) or dv == 0:
            dv = max(float(np.nanmax(np.abs(diff))), 1e-6)

        zvmin, zvmax = robust_vlim(truth_z, pred_z, q=(1, 99))
        zdv = np.nanpercentile(np.abs(diff_z), 99)
        if not np.isfinite(zdv) or zdv == 0:
            zdv = max(float(np.nanmax(np.abs(diff_z))), 1e-6)

        # raw maps
        # plot_case_spatial(
        #     eval_coords, truth,
        #     title=f"{pname}: true",
        #     out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_true.png"),
        #     cmap="viridis", invert_y=args.invert_y, vmin=vmin, vmax=vmax,
        # )
        # plot_case_spatial(
        #     eval_coords, pred,
        #     title=f"{pname}: predicted",
        #     out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_pred.png"),
        #     cmap="viridis", invert_y=args.invert_y, vmin=vmin, vmax=vmax,
        # )
        # plot_case_spatial(
        #     eval_coords, diff,
        #     title=f"{pname}: predicted - true",
        #     out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_diff.png"),
        #     cmap="coolwarm", invert_y=args.invert_y, vmin=-dv, vmax=dv,
        # )

        # normalized maps
        plot_case_spatial(
            eval_coords, truth_z,
            title=f"{pname}: true (z-score)",
            out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_true_zscore.png"),
            cmap="Blues", invert_y=args.invert_y, vmin=zvmin, vmax=zvmax,
        )
        plot_case_spatial(
            eval_coords, pred_z,
            title=f"{pname}: predicted (z-score)",
            out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_pred_zscore.png"),
            cmap="Blues", invert_y=args.invert_y, vmin=zvmin, vmax=zvmax,
        )
        # plot_case_spatial(
        #     eval_coords, diff_z,
        #     title=f"{pname}: predicted - true (z-score)",
        #     out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_diff_zscore.png"),
        #     cmap="coolwarm", invert_y=args.invert_y, vmin=-zdv, vmax=zdv,
        # )

        row = metrics_df.loc[metrics_df["protein"] == pname].iloc[0]
        # raw_metric_text = (
        #     f"Pearson = {row['pearson_r']:.3f}\n"
        #     f"Spearman = {row['spearman_r']:.3f}\n"
        #     f"CCC = {row['ccc']:.3f}\n"
        #     f"R² = {row['r2']:.3f}\n"
        #     f"MAE = {row['mae']:.3f}\n"
        #     f"RMSE = {row['rmse']:.3f}\n"
        #     f"Slope = {row['slope']:.3f}\n"
        #     f"std ratio = {row['std_ratio']:.3f}"
        # )
        # plot_case_scatter(
        #     truth, pred,
        #     title=f"{pname}: true vs predicted (raw)",
        #     out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_scatter_raw.png"),
        #     metric_text=raw_metric_text,
        # )

        z_metric_text = (
            f"Pearson = {row['pearson_r']:.3f}\n"
            f"zsep_R² = {row['zsep_r2']:.3f}\n"
            f"zsep_MAE = {row['zsep_mae']:.3f}\n"
            f"zsep_RMSE = {row['zsep_rmse']:.3f}\n"
            f"z-grid Pearson = {row['zsep_grid_pearson_r']:.3f}\n"
            f"z-grid R² = {row['zsep_grid_r2']:.3f}"
        )
        plot_case_scatter(
            truth_z, pred_z,
            title=f"{pname}: true vs predicted (z-score)",
            out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_scatter_zscore.png"),
            metric_text=z_metric_text,
        )

        # truth_grid, _, extent = grid_aggregate(
        #     x_eval, y_eval, truth, n_bins=args.grid_bins, agg="mean",
        #     xy_range=xy_range, min_count=args.grid_min_count
        # )
        # pred_grid, _, _ = grid_aggregate(
        #     x_eval, y_eval, pred, n_bins=args.grid_bins, agg="mean",
        #     xy_range=xy_range, min_count=args.grid_min_count
        # )
        # diff_grid = pred_grid - truth_grid
        #
        # gvmin, gvmax = robust_vlim(
        #     truth_grid[np.isfinite(truth_grid)],
        #     pred_grid[np.isfinite(pred_grid)],
        #     q=(1, 99),
        # )
        # gdv = np.nanpercentile(np.abs(diff_grid[np.isfinite(diff_grid)]), 99) if np.isfinite(diff_grid).any() else 1.0
        # if not np.isfinite(gdv) or gdv == 0:
        #     gdv = 1.0

        # plot_grid_image(
        #     truth_grid, extent,
        #     title=f"{pname}: grid true",
        #     out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_grid_true.png"),
        #     cmap="Blues", invert_y=args.invert_y, vmin=gvmin, vmax=gvmax,
        # )
        # plot_grid_image(
        #     pred_grid, extent,
        #     title=f"{pname}: grid predicted",
        #     out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_grid_pred.png"),
        #     cmap="Blues", invert_y=args.invert_y, vmin=gvmin, vmax=gvmax,
        # )
        # plot_grid_image(
        #     diff_grid, extent,
        #     title=f"{pname}: grid predicted - true",
        #     out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_grid_diff.png"),
        #     cmap="coolwarm", invert_y=args.invert_y, vmin=-gdv, vmax=gdv,
        # )
        #
        # plot_rank_curve(
        #     truth, pred,
        #     title=f"{pname}: peak / rank curve (raw)",
        #     out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_rank_curve_raw.png"),
        #     normalized=False,
        # )
        # plot_rank_curve(
        #     truth, pred,
        #     title=f"{pname}: peak / rank curve (z-score)",
        #     out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_rank_curve_zscore.png"),
        #     normalized=True,
        # )

    summary = {
        "n_cells_in_pred_h5ad": int(adata.n_obs),
        "n_eval_cells": int(eval_true.shape[0]),
        "n_proteins": int(eval_true.shape[1]),
        "case_study_proteins": case_names,

        "median_raw_pearson_r": float(np.nanmedian(metrics_df["pearson_r"])),
        "median_raw_spearman_r": float(np.nanmedian(metrics_df["spearman_r"])),
        "median_raw_ccc": float(np.nanmedian(metrics_df["ccc"])),
        "median_raw_r2": float(np.nanmedian(metrics_df["r2"])),
        "median_raw_mae": float(np.nanmedian(metrics_df["mae"])),
        "median_raw_rmse": float(np.nanmedian(metrics_df["rmse"])),

        "median_zsep_r2": float(np.nanmedian(metrics_df["zsep_r2"])),
        "median_zsep_mae": float(np.nanmedian(metrics_df["zsep_mae"])),
        "median_zsep_rmse": float(np.nanmedian(metrics_df["zsep_rmse"])),

        "median_grid_pearson_r": float(np.nanmedian(metrics_df["grid_pearson_r"])),
        "median_zsep_grid_pearson_r": float(np.nanmedian(metrics_df["zsep_grid_pearson_r"])),

        "median_std_ratio": float(np.nanmedian(metrics_df["std_ratio"])),
        "median_slope": float(np.nanmedian(metrics_df["slope"])),
    }
    with open(os.path.join(args.outdir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Saved evaluation and plots to: {args.outdir}")


if __name__ == "__main__":
    main()
