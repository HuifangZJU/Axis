#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate and plot saved XGBoost cross-modality protein predictions.

Revisions in this version
-------------------------
1. Remove all grid-based evaluation, metrics, and plots.
2. Replace the per-protein metric heatmap with a boxplot + per-protein dots.
3. Replace ranked barplots with density plots of metric distributions.
4. Overlay raw and z-scored scatter plots in the same panel with different colors.

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
from scipy.stats import pearsonr, spearmanr, gaussian_kde


# ============================================================
# helpers
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def setup_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 350,
        "savefig.dpi": 350,
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
        return pd.DataFrame({"x": arr[:, 0].astype(float), "y": arr[:, 1].astype(float)}, index=adata.obs_names)

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
# metrics
# ============================================================

def compute_per_protein_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protein_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for j, pname in enumerate(protein_names):
        yt = np.asarray(y_true[:, j], dtype=float)
        yp = np.asarray(y_pred[:, j], dtype=float)

        mu_t = float(np.nanmean(yt))
        sd_t = float(np.nanstd(yt))
        mu_p = float(np.nanmean(yp))
        sd_p = float(np.nanstd(yp))

        yt_zref = zscore_ref(yt, mu_t, sd_t)
        yp_zref = zscore_ref(yp, mu_t, sd_t)

        yt_zsep = zscore_self(yt)
        yp_zsep = zscore_self(yp)

        slope, intercept = calibration_line(yt, yp)
        z_slope, z_intercept = calibration_line(yt_zsep, yp_zsep)

        row = {
            "protein": str(pname),

            # raw-scale metrics
            "pearson_r": safe_corr(yt, yp, "pearson"),
            "spearman_r": safe_corr(yt, yp, "spearman"),
            "ccc": lin_ccc(yt, yp),
            "r2": float(r2_score(yt, yp)) if np.std(yt) > 0 else np.nan,
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),

            # raw diagnostics
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
            "zsep_pearson_r": safe_corr(yt_zsep, yp_zsep, "pearson"),
            "zsep_spearman_r": safe_corr(yt_zsep, yp_zsep, "spearman"),
            "zsep_ccc": lin_ccc(yt_zsep, yp_zsep),
            "zsep_r2": float(r2_score(yt_zsep, yp_zsep)) if np.std(yt_zsep) > 0 else np.nan,
            "zsep_mae": float(mean_absolute_error(yt_zsep, yp_zsep)),
            "zsep_rmse": float(np.sqrt(mean_squared_error(yt_zsep, yp_zsep))),
            "zsep_slope": z_slope,
            "zsep_intercept": z_intercept,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["zsep_r2", "pearson_r", "ccc"], ascending=[False, False, False]).reset_index(drop=True)
    return df


# ============================================================
# plots
# ============================================================


def plot_metric_boxplot(metrics_df: pd.DataFrame, out_png: str):
    """
    Replace heatmap with boxplot + per-protein dots.
    Show the same set of raw and z-scored metrics.
    """
    metric_map = [
        ("pearson_r", "Pearson\n(raw)"),
        ("zsep_pearson_r", "Pearson\n(z)"),
        ("spearman_r", "Spearman\n(raw)"),
        ("zsep_spearman_r", "Spearman\n(z)"),
        ("ccc", "CCC\n(raw)"),
        ("zsep_ccc", "CCC\n(z)"),
        # ("r2", "R²\n(raw)"),
        # ("zsep_r2", "R²\n(z)"),
        ("mae", "MAE\n(raw)"),
        ("zsep_mae", "MAE\n(z)"),
        ("rmse", "RMSE\n(raw)"),
        ("zsep_rmse", "RMSE\n(z)"),
    ]

    present = [(c, lab) for c, lab in metric_map if c in metrics_df.columns]
    if not present:
        return

    values = []
    labels = []
    for c, lab in present:
        arr = metrics_df[c].to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        values.append(arr)
        labels.append(lab)

    fig, ax = plt.subplots(figsize=(1.05 * len(labels) + 3.5, 7.2))

    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        medianprops=dict(linewidth=1.3),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )

    # subtle alternating fill to distinguish raw vs z
    for i, patch in enumerate(bp["boxes"]):
        patch.set_alpha(0.35)
        patch.set_facecolor("#4C78A8" if "(raw)" in labels[i] else "#F58518")

    # overlay per-protein dots
    rng = np.random.default_rng(0)
    for i, (c, _) in enumerate(present, start=1):
        arr = metrics_df[c].to_numpy(dtype=float)
        good = np.isfinite(arr)
        xj = rng.normal(i, 0.06, size=good.sum())
        ax.scatter(xj, arr[good], s=16, alpha=0.70, edgecolors="none")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Per-protein evaluation metrics: boxplot with per-protein dots")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _plot_density(ax, arr, label=None, color=None):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    # histogram density
    ax.hist(arr, bins=10, density=True, alpha=0.22, color=color, edgecolor="none")
    # smooth density when possible
    if arr.size >= 10 and np.std(arr) > 0:
        xs = np.linspace(np.min(arr), np.max(arr), 300)
        kde = gaussian_kde(arr)
        ax.plot(xs, kde(xs), linewidth=2.0, label=label, color=color)
    else:
        if label is not None:
            ax.plot([], [], label=label, color=color)


def plot_metric_density_panels(metrics_df: pd.DataFrame, out_png: str):
    """
    Replace ranked barplots with density plots of protein-wise metric distributions.
    """
    pairs = [
        ("pearson_r", "zsep_pearson_r", "Pearson"),
        ("spearman_r", "zsep_spearman_r", "Spearman"),
        ("ccc", "zsep_ccc", "CCC"),
        ("r2", "zsep_r2", "R²"),
        ("mae", "zsep_mae", "MAE"),
        ("rmse", "zsep_rmse", "RMSE"),
    ]

    ncol = 3
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.8 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, (raw_c, z_c, title) in zip(axes, pairs):
        _plot_density(ax, metrics_df[raw_c].to_numpy(dtype=float), label="raw", color="#4C78A8")
        _plot_density(ax, metrics_df[z_c].to_numpy(dtype=float), label="z-score", color="#F58518")
        ax.set_title(title)
        ax.set_xlabel("Metric value")
        ax.set_ylabel("Density")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.legend(frameon=False)

    for ax in axes[len(pairs):]:
        ax.axis("off")

    fig.suptitle("Distribution of per-protein metrics: raw vs z-scored", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_global_scatter_panels(y_true: np.ndarray, y_pred: np.ndarray, metrics_df: pd.DataFrame, out_png: str):
    """
    Overlay raw scatter and z-scored scatter in the same panel with different colors.
    """
    top = metrics_df.head(min(9, len(metrics_df)))["protein"].tolist()
    protein_order = [str(x) for x in metrics_df["protein"].tolist()]
    name_to_idx = {p: i for i, p in enumerate(protein_order)}

    n = len(top)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 4.4 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, pname in zip(axes, top):
        j = name_to_idx[pname]
        yt = np.asarray(y_true[:, j], dtype=float)
        yp = np.asarray(y_pred[:, j], dtype=float)

        yt_z = zscore_self(yt)
        yp_z = zscore_self(yp)

        lo_raw, hi_raw = robust_vlim(yt, yp, q=(0, 100))
        lo_z, hi_z = robust_vlim(yt_z, yp_z, q=(0, 100))
        lo = min(lo_raw, lo_z)
        hi = max(hi_raw, hi_z)

        ax.scatter(yt, yp, s=0.08, alpha=0.16, label="raw", color="#4C78A8")
        ax.scatter(yt_z, yp_z, s=0.08, alpha=0.16, label="z-score", color="#F58518")
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="0.35")

        row = metrics_df.loc[metrics_df["protein"] == pname].iloc[0]
        txt = (
            f"raw r={row['pearson_r']:.2f}, R²={row['r2']:.2f}\n"
            f"z r={row['zsep_pearson_r']:.2f}, zR²={row['zsep_r2']:.2f}"
        )

        ax.set_title(pname)
        ax.set_xlabel("True / true z-score")
        ax.set_ylabel("Predicted / predicted z-score")
        ax.text(
            0.03, 0.97, txt,
            transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="0.8")
        )
        ax.legend(frameon=False, markerscale=8, loc="lower right")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Top proteins: raw and z-scored scatter overlays", y=0.995)
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
    plt.figure(figsize=(6.4,3.8))
    sc = plt.scatter(coords["x"], coords["y"], c=values, s=1, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    plt.gca().set_aspect("equal", adjustable="box")
    if invert_y:
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(sc, fraction=0.03, pad=0.02,shrink=0.5)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_case_scatter_overlay(truth: np.ndarray, pred: np.ndarray, title: str, out_png: str, metric_text: Optional[str] = None):
    """
    Case-study scatter: overlay raw and z-scored scatters in different colors.
    """
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)


    lo, hi = robust_vlim(truth, pred, q=(0, 100))


    plt.figure(figsize=(6.0, 5.8))
    plt.scatter(truth, pred, s=0.8, alpha=0.18, label="raw", color="#4C78A8")
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="0.35")
    plt.xlabel("True")
    plt.ylabel("Predicted ")
    plt.title(title)
    plt.legend(frameon=False, markerscale=10, loc="lower right")
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


def plot_rank_curve(truth: np.ndarray, pred: np.ndarray, title: str, out_png: str):
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)

    truth_z = zscore_self(truth)
    pred_z = zscore_self(pred)

    order_raw = np.argsort(-truth)
    order_z = np.argsort(-truth_z)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), squeeze=False)
    ax1, ax2 = axes.ravel()

    ax1.plot(truth[order_raw], label="True", linewidth=2.0)
    ax1.plot(pred[order_raw], label="Predicted", linewidth=2.0, linestyle="--")
    ax1.set_title("Raw")
    ax1.set_xlabel("Cells ranked by true expression")
    ax1.set_ylabel("Expression")
    ax1.legend(frameon=False)

    ax2.plot(truth_z[order_z], label="True z-score", linewidth=2.0)
    ax2.plot(pred_z[order_z], label="Predicted z-score", linewidth=2.0, linestyle="--")
    ax2.set_title("Z-score")
    ax2.set_xlabel("Cells ranked by true z-score")
    ax2.set_ylabel("Expression z-score")
    ax2.legend(frameon=False)

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ============================================================
# main
# ============================================================

def main():
    setup_matplotlib()

    parser = argparse.ArgumentParser(description="Evaluate and plot saved XGBoost cross-modality predictions.")
    parser.add_argument("--pred_h5ad", type=str, default="//media/huifang/data/sennet/Axis_generated_proteins/xgb_test_pair_predictions/virtual_sennet_24_protein_predicted.h5ad",
                        help="Saved prediction h5ad, e.g. aligned_pred_test_only.h5ad or aligned_pred_all.h5ad")
    # parser.add_argument("--gt_h5ad", type=str,
    #                     default="//media/huifang/data/sennet/Axis_generated_proteins/virtual_sennet_1_protein.h5ad",
    #                     help="Saved prediction h5ad, e.g. aligned_pred_test_only.h5ad or aligned_pred_all.h5ad")
    parser.add_argument("--gt_h5ad", type=str,
                        default="//media/huifang/data/sennet/Pretsa_pairs/presta_sennet_24_protein_knot30_fitlog1p1.h5ad",
                        help="Saved prediction h5ad, e.g. aligned_pred_test_only.h5ad or aligned_pred_all.h5ad")
    parser.add_argument("--metrics_csv", type=str, default=None,
                        help="Optional existing metrics CSV. The script will still recompute metrics.")
    parser.add_argument("--outdir", type=str, default="/media/huifang/data/sennet/Axis_generated_proteins/xgb_test_pair_predictions/evaluation24/")
    parser.add_argument("--case_study_proteins", nargs="*", default=['p16','HSP47'],
                        help="Protein names for detailed plots. If omitted, top proteins by zsep_R2 are used.")
    parser.add_argument("--n_case_study", type=int, default=9)
    parser.add_argument("--invert_y", type=lambda x: str(x).lower() in {"1","true","yes","y"}, default=False)
    args = parser.parse_args()

    ensure_dir(args.outdir)

    pred_adata = ad.read_h5ad(args.pred_h5ad)
    gt_adata = ad.read_h5ad(args.gt_h5ad)

    coords = get_coords(gt_adata)
    y_pred = to_dense(pred_adata.X).astype(np.float32)
    y_true = to_dense(gt_adata.X).astype(np.float32)
    protein_names = [str(x) for x in gt_adata.var_names.tolist()]


    if y_pred.shape != y_true.shape:
        raise ValueError(f"Prediction and truth shapes differ: {y_pred.shape} vs {y_true.shape}")


    eval_coords = coords.reset_index(drop=True)
    eval_true = np.log1p(y_true)
    eval_pred = np.log1p(y_pred)

    metrics_df = compute_per_protein_metrics(eval_true, eval_pred, protein_names)
    metrics_df.to_csv(os.path.join(args.outdir, "per_protein_metrics_recomputed.csv"), index=False)

    if args.metrics_csv is not None and os.path.exists(args.metrics_csv):
        old_df = pd.read_csv(args.metrics_csv)
        merge_cols = [c for c in ["protein", "pearson_r", "r2"] if c in old_df.columns]
        if "protein" in merge_cols:
            merged = metrics_df.merge(
                old_df[merge_cols].rename(columns={c: f"old_{c}" for c in merge_cols if c != "protein"}),
                on="protein",
                how="left",
            )
            merged.to_csv(os.path.join(args.outdir, "metrics_with_previous_reference.csv"), index=False)

    # new plots
    plot_metric_boxplot(metrics_df, os.path.join(args.outdir, "metric_boxplot_raw_and_zscore.png"))
    plot_metric_density_panels(metrics_df, os.path.join(args.outdir, "metric_density_panels_raw_and_zscore.png"))
    plot_global_scatter_panels(eval_true, eval_pred, metrics_df, os.path.join(args.outdir, "global_scatter_panels_overlay.png"))

    if args.case_study_proteins is not None and len(args.case_study_proteins) > 0:
        case_names = [p for p in args.case_study_proteins if p in protein_names]
    else:
        case_names = metrics_df.head(args.n_case_study)["protein"].tolist()

    name_to_idx = {p: i for i, p in enumerate(protein_names)}

    for pname in case_names:
        j = name_to_idx[pname]
        truth = np.asarray(eval_true[:, j], dtype=float)
        pred = np.asarray(eval_pred[:, j], dtype=float)
        truth_z = zscore_self(truth)
        pred_z = zscore_self(pred)

        vmin, vmax = robust_vlim(truth, pred, q=(1, 99))
        zvmin, zvmax = robust_vlim(truth_z, pred_z, q=(1, 99))

        # spatial maps: keep raw + z-score
        plot_case_spatial(
            eval_coords, truth,
            title=f"{pname}: true",
            out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_true.png"),
            cmap="coolwarm", invert_y=args.invert_y, vmin=vmin, vmax=vmax,
        )
        plot_case_spatial(
            eval_coords, pred,
            title=f"{pname}: predicted",
            out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_pred.png"),
            cmap="coolwarm", invert_y=args.invert_y, vmin=vmin, vmax=vmax,
        )
        # plot_case_spatial(
        #     eval_coords, truth_z,
        #     title=f"{pname}: true (z-score)",
        #     out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_true_zscore.png"),
        #     cmap="coolwarm", invert_y=args.invert_y, vmin=zvmin, vmax=zvmax,
        # )
        # plot_case_spatial(
        #     eval_coords, pred_z,
        #     title=f"{pname}: predicted (z-score)",
        #     out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_pred_zscore.png"),
        #     cmap="coolwarm", invert_y=args.invert_y, vmin=zvmin, vmax=zvmax,
        # )

        row = metrics_df.loc[metrics_df["protein"] == pname].iloc[0]
        metric_text = (
            f"Raw: r={row['pearson_r']:.3f}, ρ={row['spearman_r']:.3f}, CCC={row['ccc']:.3f}, R²={row['r2']:.3f}\n"
            f"Raw: MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}\n"
            # f"Z: r={row['zsep_pearson_r']:.3f}, ρ={row['zsep_spearman_r']:.3f}, CCC={row['zsep_ccc']:.3f}, R²={row['zsep_r2']:.3f}\n"
            # f"Z: MAE={row['zsep_mae']:.3f}, RMSE={row['zsep_rmse']:.3f}"
        )
        plot_case_scatter_overlay(
            truth, pred,
            title=f"{pname}: true vs predicted (raw + z-score)",
            out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_scatter_overlay.png"),
            metric_text=metric_text,
        )

        # plot_rank_curve(
        #     truth, pred,
        #     title=f"{pname}: rank curves (raw and z-score)",
        #     out_png=os.path.join(args.outdir, f"case_study_{safe_name(pname)}_rank_curve.png"),
        # )

    summary = {
        "n_eval_cells": int(eval_true.shape[0]),
        "n_proteins": int(eval_true.shape[1]),
        "case_study_proteins": case_names,

        "median_raw_pearson_r": float(np.nanmedian(metrics_df["pearson_r"])),
        "median_raw_spearman_r": float(np.nanmedian(metrics_df["spearman_r"])),
        "median_raw_ccc": float(np.nanmedian(metrics_df["ccc"])),
        "median_raw_r2": float(np.nanmedian(metrics_df["r2"])),
        "median_raw_mae": float(np.nanmedian(metrics_df["mae"])),
        "median_raw_rmse": float(np.nanmedian(metrics_df["rmse"])),

        "median_zsep_pearson_r": float(np.nanmedian(metrics_df["zsep_pearson_r"])),
        "median_zsep_spearman_r": float(np.nanmedian(metrics_df["zsep_spearman_r"])),
        "median_zsep_ccc": float(np.nanmedian(metrics_df["zsep_ccc"])),
        "median_zsep_r2": float(np.nanmedian(metrics_df["zsep_r2"])),
        "median_zsep_mae": float(np.nanmedian(metrics_df["zsep_mae"])),
        "median_zsep_rmse": float(np.nanmedian(metrics_df["zsep_rmse"])),

        "median_std_ratio": float(np.nanmedian(metrics_df["std_ratio"])),
        "median_slope": float(np.nanmedian(metrics_df["slope"])),
    }
    with open(os.path.join(args.outdir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Saved evaluation and plots to: {args.outdir}")


if __name__ == "__main__":
    main()
