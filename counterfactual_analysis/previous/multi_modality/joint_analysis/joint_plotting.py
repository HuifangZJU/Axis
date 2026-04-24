#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import anndata as ad
import scipy.sparse as sp

import pyreadr


def canonical_protein_name(s: str) -> str:
    """
    Make names comparable between sources:
      - lower
      - replace '.' and '-' with ''
      - remove spaces and underscores
      - keep alphanumerics only
    Example: 'PD-L1' and 'PD.L1' -> 'pdl1'
    """
    s = str(s).strip().lower()
    s = s.replace(".", "").replace("-", "").replace("_", "").replace(" ", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def build_name_map(cols):
    """
    Map canonical_name -> original_name (first occurrence).
    """
    m = {}
    for c in cols:
        k = canonical_protein_name(c)
        if k and k not in m:
            m[k] = c
    return m


def align_protein_channels(protein_df: pd.DataFrame, pred_df: pd.DataFrame, drop_original=("DAPI",)):
    """
    Drop unwanted channels in protein_df, then align both matrices by canonicalized names.
    Returns: protein_aligned, pred_aligned, aligned_names (canonical), original_name_pairs
    """
    # drop DAPI (and other requested)
    drop_set = set(drop_original)
    keep_cols = [c for c in protein_df.columns if c not in drop_set]
    protein_df = protein_df[keep_cols].copy()

    map_prot = build_name_map(protein_df.columns)
    map_pred = build_name_map(pred_df.columns)

    common_keys = sorted(set(map_prot.keys()).intersection(set(map_pred.keys())))
    if len(common_keys) == 0:
        raise ValueError("No common proteins after canonical matching.")

    # use the predicted names as canonical display? Better: use original protein names for display
    prot_cols = [map_prot[k] for k in common_keys]
    pred_cols = [map_pred[k] for k in common_keys]

    protein_aligned = protein_df[prot_cols].copy()
    pred_aligned = pred_df[pred_cols].copy()

    # Rename both to the SAME final names (use original protein names for nicer display)
    final_names = prot_cols
    protein_aligned.columns = final_names
    pred_aligned.columns = final_names

    name_pairs = [(map_prot[k], map_pred[k]) for k in common_keys]

    return protein_aligned, pred_aligned, final_names, name_pairs


def normalize_log1p_matrix(X, target_sum=1e4):
    """
    Normalize each row to target_sum (like scanpy normalize_total), then log1p.
    Works for dense numpy arrays and CSR sparse matrices.
    """
    if sp.issparse(X):
        X = X.tocsr().astype(np.float32, copy=False)
        row_sum = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        scale = np.zeros_like(row_sum, dtype=np.float32)
        nz = row_sum > 0
        scale[nz] = (target_sum / row_sum[nz]).astype(np.float32)

        # row-wise scaling
        X = X.multiply(scale[:, None])
        # log1p on nonzeros
        X.data = np.log1p(X.data).astype(np.float32, copy=False)
        return X

    # dense
    X = np.asarray(X, dtype=np.float32)
    X = np.maximum(X, 0.0)  # guard in case of tiny negatives
    row_sum = X.sum(axis=1, keepdims=True).astype(np.float32)
    scale = np.zeros_like(row_sum, dtype=np.float32)
    nz = row_sum > 0
    scale[nz] = (target_sum / row_sum[nz]).astype(np.float32)
    X = X * scale
    X = np.log1p(X).astype(np.float32, copy=False)
    return X

# -------------------------
# Utilities
# -------------------------
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))


def ensure_numeric_df(df: pd.DataFrame, fillna=0.0) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(fillna)
    return out


def read_xenium_location_weird_csv(path: str) -> pd.DataFrame:
    """
    Xenium location CSV has an invalid first line like ",0,1"
    then rows: idx,x,y
    """
    df = pd.read_csv(path, header=None, skiprows=1)
    if df.shape[1] < 3:
        raise ValueError(f"Xenium location CSV has <3 columns after skiprows=1: {df.shape}")
    df = df.iloc[:, :3].copy()
    df.columns = ["row_id", "x", "y"]
    df[["x", "y"]] = ensure_numeric_df(df[["x", "y"]])
    return df


def read_rds_as_df_pred(path: str) -> pd.DataFrame:
    """Read an RDS that pyreadr can parse into a pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    res = pyreadr.read_r(path)
    if len(res) == 0:
        raise ValueError(f"No object found in RDS: {path}")
    obj = next(iter(res.values()))
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected DataFrame from RDS, got {type(obj)} for {path}")
    return obj


def orient_df_to_match_rows(df: pd.DataFrame, n_rows: int, name: str) -> pd.DataFrame:
    """
    Ensure df has n_rows rows; transpose if df has n_rows columns instead.
    """
    r, c = df.shape
    if r == n_rows:
        return df
    if c == n_rows:
        print(f"[TRANSPOSE] {name} {df.shape} -> {df.T.shape}")
        return df.T
    raise ValueError(f"{name} shape {df.shape} does not match n_rows={n_rows} in rows or cols.")


# -------------------------
# Plot 1: per-channel original vs predicted with same legend
# -------------------------
def plot_two_spatial_panels(
    x1, y1, v1, title1,
    x2, y2, v2, title2,
    out_png,
    invert_y=True,
    point_size=1,
    clip_low=1.0,
    clip_high=99.0,
    max_points=0,
    seed=0,
):
    x1 = np.asarray(x1); y1 = np.asarray(y1); v1 = np.asarray(v1)
    x2 = np.asarray(x2); y2 = np.asarray(y2); v2 = np.asarray(v2)

    if max_points and len(v1) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(v1), size=max_points, replace=False)
        x1, y1, v1 = x1[idx], y1[idx], v1[idx]

    if max_points and len(v2) > max_points:
        rng = np.random.default_rng(seed + 1)
        idx = rng.choice(len(v2), size=max_points, replace=False)
        x2, y2, v2 = x2[idx], y2[idx], v2[idx]

    both = np.concatenate([v1[np.isfinite(v1)], v2[np.isfinite(v2)]])
    if both.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.percentile(both, clip_low)
        vmax = np.percentile(both, clip_high)
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
            vmin = float(np.nanmin(both))
            vmax = float(np.nanmax(both))
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
                vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)

    sc1 = axes[0].scatter(x1, y1, c=v1, s=point_size, vmin=vmin, vmax=vmax)
    axes[0].set_title(title1)
    axes[0].axis("off")
    axes[0].set_aspect("equal", adjustable="box")
    if invert_y:
        axes[0].invert_yaxis()

    sc2 = axes[1].scatter(x2, y2, c=v2, s=point_size, vmin=vmin, vmax=vmax)
    axes[1].set_title(title2)
    axes[1].axis("off")
    axes[1].set_aspect("equal", adjustable="box")
    if invert_y:
        axes[1].invert_yaxis()

    # Give extra right margin for the colorbar so it never overlaps the plots
    fig.subplots_adjust(right=0.88, wspace=0.08)

    # Put colorbar in the reserved right margin
    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)

    plt.tight_layout(rect=[0, 0, 0.88, 1])  # keep space on the right
    plt.savefig(out_png)
    plt.close()


# -------------------------
# Plot 2: grid similarity (mean)
# -------------------------
def grid_aggregate(x, y, v, n_bins=200, agg="mean", xy_range=None, min_count=3):
    x = np.asarray(x); y = np.asarray(y); v = np.asarray(v)

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
    xi = xi[valid]; yi = yi[valid]; v = v[valid]

    counts = np.zeros((n_bins, n_bins), dtype=np.int32)
    sums = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(counts, (yi, xi), 1)
    np.add.at(sums,   (yi, xi), v)

    if agg == "mean":
        grid = sums / np.maximum(counts, 1)
    elif agg == "sum":
        grid = sums
    else:
        raise ValueError("agg must be 'mean' or 'sum'")

    grid = grid.astype(np.float64)
    grid[counts < min_count] = np.nan
    return grid, counts


def grid_metrics(truth_grid, pred_grid):
    t = truth_grid.ravel()
    p = pred_grid.ravel()
    m = np.isfinite(t) & np.isfinite(p)
    t = t[m]; p = p[m]
    if t.size < 10:
        return {"n": int(t.size), "pearson": np.nan, "spearman": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan}

    # pearson
    def corr(a, b):
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom = np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0))
        return np.nan if denom == 0 else float(np.sum(a0 * b0) / denom)

    pear = corr(t, p)

    # spearman (rank-based)
    tr = pd.Series(t).rank(method="average").to_numpy()
    pr = pd.Series(p).rank(method="average").to_numpy()
    spear = corr(tr, pr)

    # r2
    t_mean = t.mean()
    ss_tot = np.sum((t - t_mean) ** 2)
    ss_res = np.sum((t - p) ** 2)
    r2 = np.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    mae = float(np.mean(np.abs(t - p)))

    return {"n": int(t.size), "pearson": pear, "spearman": spear, "r2": r2, "rmse": rmse, "mae": mae}


def plot_grid_scatter(truth_grid, pred_grid, out_png, title="", hexbin=True):
    t = truth_grid.ravel()
    p = pred_grid.ravel()
    m = np.isfinite(t) & np.isfinite(p)
    t = t[m]; p = p[m]
    if t.size == 0:
        print(f"[WARN] No valid grid cells for '{title}'. Consider smaller n_bins or min_count.")
        return

    plt.figure(figsize=(5.5, 5.0), dpi=200)
    plt.scatter(t, p, s=1, alpha=0.4)

    both = np.concatenate([t, p])
    hi = np.nanpercentile(both, 99)
    if not np.isfinite(hi) or hi <= 0:
        hi = float(np.nanmax(both))
        if not np.isfinite(hi) or hi <= 0:
            hi = 1.0

    lo = 0.0
    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Original protein (grid mean)")
    plt.ylabel("Predicted on Xenium (grid mean)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_grid_heatmaps(truth_grid, pred_grid, out_png, title=""):
    plt.figure(figsize=(10, 4), dpi=200)

    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(truth_grid, origin="lower")
    ax1.set_title(f"{title}\nOriginal grid")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(pred_grid, origin="lower")
    ax2.set_title(f"{title}\nPred grid")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# -------------------------
# Plot 3: location mismatch
# -------------------------

def _downsample_xy(xy: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points and xy.shape[0] > max_points:
        idx = rng.choice(xy.shape[0], size=max_points, replace=False)
        return xy[idx]
    return xy

def compute_zoom_box(all_xy: np.ndarray, frac: float = 0.10) -> tuple:
    """
    Compute a zoom box centered at median, with width/height = frac*(p95-p5) span.
    Returns (x0, x1, y0, y1).
    """
    cx = float(np.median(all_xy[:, 0]))
    cy = float(np.median(all_xy[:, 1]))
    xspan = float(np.percentile(all_xy[:, 0], 95) - np.percentile(all_xy[:, 0], 5))
    yspan = float(np.percentile(all_xy[:, 1], 95) - np.percentile(all_xy[:, 1], 5))
    half_w = frac * xspan
    half_h = frac * yspan
    return (cx - half_w, cx + half_w, cy - half_h, cy + half_h)

def plot_overlay(
    protein_xy: np.ndarray,
    xenium_xy: np.ndarray,
    out_png: str,
    title: str,
    invert_y: bool = True,
    max_points: int = 200000,
    seed: int = 0,
    xlim=None,
    ylim=None,
):
    """
    Overlay two point clouds; optional xlim/ylim for zoom.
    """
    rng = np.random.default_rng(seed)

    P = _downsample_xy(protein_xy, max_points, rng)
    X = _downsample_xy(xenium_xy, max_points, rng)

    plt.figure(figsize=(7, 7), dpi=200)
    plt.scatter(P[:, 0], P[:, 1], s=0.1, alpha=0.5, label="Protein")
    plt.scatter(X[:, 0], X[:, 1], s=0.1, alpha=0.5, label="Xenium")

    plt.legend(loc="best", frameon=True)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    # plt.savefig(out_png)
    plt.show()
    plt.close()

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_folder", default="/media/huifang/data/registration/phenocycler/haotian")
    ap.add_argument("--invert_y", default=True)
    ap.add_argument("--scatter_max_points", type=int, default=200000)
    ap.add_argument("--grid_bins", type=int, default=250)
    ap.add_argument("--grid_min_count", type=int, default=5)
    args = ap.parse_args()

    root = args.root_folder

    # Output folders
    out_root = os.path.join(root, "joint_signals")
    out_scatter = os.path.join(out_root, "scatter_channels")
    out_grid = os.path.join(out_root, "grid_similarity")
    out_mismatch = os.path.join(out_root, "location_mismatch")
    os.makedirs(out_scatter, exist_ok=True)
    os.makedirs(out_grid, exist_ok=True)
    os.makedirs(out_mismatch, exist_ok=True)

    # Inputs
    prot_h5ad = os.path.join(root, "LUAD_2_A_protein_axis.h5ad")
    xenium_csv = os.path.join(root, "LUAD_2_A_xenium_axis.csv")
    pred_rds = os.path.join(root, "xenium_protein_predict.rds")

    # ---- Load original protein (h5ad)
    print("[LOAD] original protein h5ad:", prot_h5ad)
    adata_p = ad.read_h5ad(prot_h5ad)

    prot_xy = pd.DataFrame(np.asarray(adata_p.obsm["spatial_axis"]), columns=["x", "y"])
    prot_xy = ensure_numeric_df(prot_xy)

    Xp = adata_p.X  # original protein reads
    Xp = normalize_log1p_matrix(Xp, target_sum=1e4)  # <-- normalize + log1p

    # convert to DataFrame with channel names
    if sp.issparse(Xp):
        protein_df = pd.DataFrame.sparse.from_spmatrix(Xp, columns=adata_p.var_names)
    else:
        protein_df = pd.DataFrame(np.asarray(Xp), columns=adata_p.var_names)

    protein_df = ensure_numeric_df(protein_df)

    if protein_df.shape[0] != prot_xy.shape[0]:
        raise ValueError(f"Protein h5ad rows mismatch: X={protein_df.shape}, coords={prot_xy.shape}")

    print(f"[INFO] original protein (log-normalized): X={protein_df.shape}, coords={prot_xy.shape}")



    print(f"[INFO] original protein: X={protein_df.shape}, coords={prot_xy.shape}")

    # ---- Load xenium location (csv)
    print("[LOAD] xenium location:", xenium_csv)
    xen_loc_df = read_xenium_location_weird_csv(xenium_csv)
    xen_xy = xen_loc_df[["x", "y"]].copy()
    xen_xy = ensure_numeric_df(xen_xy)
    nX = xen_xy.shape[0]
    print(f"[INFO] xenium coords: {xen_xy.shape}")

    # protein original coords (pre-registration)
    if "spatial" not in adata_p.obsm.keys():
        raise KeyError("Protein AnnData missing obsm['spatial'] for original coordinates.")
    prot_spatial = np.asarray(adata_p.obsm["spatial"])
    prot_spatial = prot_spatial[:, :2].astype(np.float32, copy=False)  # keep first 2 cols

    # protein registered coords (after registration)
    if "spatial_axis" not in adata_p.obsm.keys():
        raise KeyError("Protein AnnData missing obsm['spatial_axis'] for registered coordinates.")
    prot_axis = np.asarray(adata_p.obsm["spatial_axis"])
    prot_axis = prot_axis[:, :2].astype(np.float32, copy=False)

    # xenium coords (already a DataFrame with x/y)
    xenium_xy = xen_xy[["x", "y"]].to_numpy(dtype=np.float32, copy=False)

    # ---- Load predicted protein on xenium (rds)
    print("[LOAD] predicted protein (xenium) rds:", pred_rds)
    pred_df = ensure_numeric_df(read_rds_as_df_pred(pred_rds))

    # Orient predicted matrix to match xenium coords rows
    pred_df = orient_df_to_match_rows(pred_df, nX, "predicted_protein")
    pred_df = ensure_numeric_df(pred_df)
    print(f"[INFO] predicted protein: {pred_df.shape} (aligned to xenium coords)")

    # ---- Determine common channels
    protein_df, pred_df, common, name_pairs = align_protein_channels(
        protein_df=protein_df,
        pred_df=pred_df,
        drop_original=("DAPI",)  # omit DAPI
    )

    print(f"[INFO] common protein channels after canonical match: {len(common)}")
    print("[INFO] example matched pairs (orig -> pred):", name_pairs[:10])


    # # 1) Per-channel scatter (original vs predicted)
    # print("[PLOT] per-channel spatial scatter...")
    # for c in common:
    #     fname = safe_name(c) + "_orig_vs_pred.png"
    #     out_png = os.path.join(out_scatter, fname)
    #     plot_two_spatial_panels(
    #         x1=prot_xy["x"].to_numpy(), y1=prot_xy["y"].to_numpy(), v1=protein_df[c].to_numpy(),
    #         title1=f"{c} (Original protein)",
    #         x2=xen_xy["x"].to_numpy(),  y2=xen_xy["y"].to_numpy(),  v2=pred_df[c].to_numpy(),
    #         title2=f"{c} (Predicted on Xenium)",
    #         out_png=out_png,
    #         invert_y=args.invert_y,
    #         point_size=0.1,
    #         clip_low=1.0,
    #         clip_high=99.0,
    #         max_points=args.scatter_max_points,
    #         seed=0,
    #     )
    #
    # # 2) Grid similarity for all channels (mean aggregation)
    # print("[GRID] grid similarity (mean) for all channels...")
    # all_xy = np.vstack([prot_xy[["x", "y"]].to_numpy(), xen_xy[["x", "y"]].to_numpy()])
    # xy_range = (
    #     float(all_xy[:, 0].min()), float(all_xy[:, 0].max()),
    #     float(all_xy[:, 1].min()), float(all_xy[:, 1].max()),
    # )
    #
    # metrics_rows = []
    # for c in common:
    #     truth_grid, _ = grid_aggregate(
    #         prot_xy["x"].to_numpy(), prot_xy["y"].to_numpy(),
    #         protein_df[c].to_numpy(),
    #         n_bins=args.grid_bins,
    #         agg="mean",
    #         xy_range=xy_range,
    #         min_count=args.grid_min_count,
    #     )
    #     pred_grid, _ = grid_aggregate(
    #         xen_xy["x"].to_numpy(), xen_xy["y"].to_numpy(),
    #         pred_df[c].to_numpy(),
    #         n_bins=args.grid_bins,
    #         agg="mean",
    #         xy_range=xy_range,
    #         min_count=args.grid_min_count,
    #     )
    #
    #     met = grid_metrics(truth_grid, pred_grid)
    #     met["protein"] = c
    #     met["grid_bins"] = args.grid_bins
    #     met["min_count"] = args.grid_min_count
    #     metrics_rows.append(met)
    #
    #     scatter_png = os.path.join(out_grid, safe_name(c) + "_grid_scatter.png")
    #     plot_grid_scatter(
    #         truth_grid, pred_grid,
    #         out_png=scatter_png,
    #         title=f"{c} grid-mean | r={met['pearson']:.3f}, ρ={met['spearman']:.3f}, R²={met['r2']:.3f}",
    #         hexbin=True,
    #     )
    #
    #     heat_png = os.path.join(out_grid, safe_name(c) + "_grid_heatmaps.png")
    #     plot_grid_heatmaps(truth_grid, pred_grid, out_png=heat_png, title=c)
    #
    # metrics_df = pd.DataFrame(metrics_rows).sort_values("r2", ascending=False)
    # metrics_csv = os.path.join(out_grid, "grid_similarity_metrics_all_proteins.csv")
    # metrics_df.to_csv(metrics_csv, index=False)
    # print("[SAVED] grid metrics:", metrics_csv)

    # 3) Location mismatch overlay + zoom (4 plots)
    print("[PLOT] location mismatch overlays (spatial vs spatial_axis)...")

    out_mismatch = os.path.join(out_root, "location_mismatch")
    os.makedirs(out_mismatch, exist_ok=True)

    # Use ONE shared zoom box computed from ALL relevant points, so zoom is comparable
    all_xy_for_zoom = np.vstack([prot_spatial, prot_axis, xenium_xy])
    x0, x1, y0, y1 = compute_zoom_box(all_xy_for_zoom, frac=0.10)

    # (1) Whole: spatial_axis vs xenium
    plot_overlay(
        protein_xy=prot_axis,
        xenium_xy=xenium_xy,
        out_png=os.path.join(out_mismatch, "overlay_spatial_axis_vs_xenium_whole.png"),
        title="Overlay: protein obsm['spatial_axis'] vs xenium (whole)",
        invert_y=args.invert_y,
        max_points=200000,
        seed=1,
    )

    # (2) Whole: spatial vs xenium
    plot_overlay(
        protein_xy=prot_spatial,
        xenium_xy=xenium_xy,
        out_png=os.path.join(out_mismatch, "overlay_spatial_vs_xenium_whole.png"),
        title="Overlay: protein obsm['spatial'] vs xenium (whole)",
        invert_y=args.invert_y,
        max_points=200000,
        seed=0,
    )



    # (3) Zoom: spatial vs xenium
    plot_overlay(
        protein_xy=prot_spatial,
        xenium_xy=xenium_xy,
        out_png=os.path.join(out_mismatch, "overlay_spatial_vs_xenium_zoom.png"),
        title=f"Overlay (zoom): obsm['spatial'] vs xenium\nbox: x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]",
        invert_y=args.invert_y,
        max_points=200000,
        seed=2,
        xlim=(x0, x1),
        ylim=(y0, y1),
    )

    # (4) Zoom: spatial_axis vs xenium
    plot_overlay(
        protein_xy=prot_axis,
        xenium_xy=xenium_xy,
        out_png=os.path.join(out_mismatch, "overlay_spatial_axis_vs_xenium_zoom.png"),
        title=f"Overlay (zoom): obsm['spatial_axis'] vs xenium\nbox: x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]",
        invert_y=args.invert_y,
        max_points=200000,
        seed=3,
        xlim=(x0, x1),
        ylim=(y0, y1),
    )

    # # Summary
    # summary = {
    #     "root_folder": root,
    #     "outputs": {
    #         "root": out_root,
    #         "scatter_channels": out_scatter,
    #         "grid_similarity": out_grid,
    #         "location_mismatch": out_mismatch,
    #         "grid_metrics_csv": metrics_csv,
    #     },
    #     "counts": {
    #         "n_protein_points": int(prot_xy.shape[0]),
    #         "n_xenium_points": int(xen_xy.shape[0]),
    #         "n_common_channels": int(len(common)),
    #     },
    #     "params": {
    #         "invert_y": bool(args.invert_y),
    #         "scatter_max_points": int(args.scatter_max_points),
    #         "grid_bins": int(args.grid_bins),
    #         "grid_min_count": int(args.grid_min_count),
    #     },
    # }
    # with open(os.path.join(out_root, "summary.json"), "w") as f:
    #     json.dump(summary, f, indent=2)
    #
    # print("[DONE] Outputs saved to:", out_root)


if __name__ == "__main__":
    main()
