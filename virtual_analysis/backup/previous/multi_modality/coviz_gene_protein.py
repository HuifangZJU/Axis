#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Co-visualize a (gene, protein) pair on Xenium spatial coordinates.

Reads:
  - Gene reads from h5ad (adata.X)  -> normalize_total + log1p (REQUIRED)
  - Protein reads from RDS (predicted protein on Xenium locations; assumed already normalized/logged)

Outputs (default under <root_folder>/pair_viz/):
  1) <protein>__<gene>__spatial.png      (side-by-side spatial maps)
  2) <protein>__<gene>__scatter.png      (cell-level scatter: gene vs protein)
  3) <protein>__<gene>__grid_scatter.png (grid-mean scatter + metrics)
  4) <protein>__<gene>__grid_heatmaps.png(optional QC)
  5) <protein>__<gene>__metrics.json
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import anndata as ad

import pyreadr

try:
    import pyreadstat
except Exception:
    pyreadstat = None


# -------------------------
# Helpers
# -------------------------
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))


def canonical_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(".", "").replace("-", "").replace("_", "").replace(" ", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def normalize_log1p_matrix(X, target_sum=1e4):
    """
    Per-row normalize to target_sum, then log1p.
    Works for dense numpy and CSR sparse matrices.
    """
    if sp.issparse(X):
        X = X.tocsr().astype(np.float32, copy=False)
        rs = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        scale = np.zeros_like(rs, dtype=np.float32)
        nz = rs > 0
        scale[nz] = (target_sum / rs[nz]).astype(np.float32)
        X = X.multiply(scale[:, None])  # may become COO depending on scipy version
        X = X.tocsr()  # <-- force CSR so slicing works
        X.data = np.log1p(X.data).astype(np.float32, copy=False)
        return X

    X = np.asarray(X, dtype=np.float32)
    X = np.maximum(X, 0.0)
    rs = X.sum(axis=1, keepdims=True).astype(np.float32)
    scale = np.zeros_like(rs, dtype=np.float32)
    nz = rs > 0
    scale[nz] = (target_sum / rs[nz]).astype(np.float32)
    X = X * scale
    X = np.log1p(X).astype(np.float32, copy=False)
    return X


def read_xenium_location_weird_csv(path: str) -> pd.DataFrame:
    """
    Xenium location CSV has invalid first line like ",0,1"
    then rows: idx,x,y
    """
    df = pd.read_csv(path, header=None, skiprows=1)
    if df.shape[1] < 3:
        raise ValueError(f"Xenium location CSV has <3 cols after skiprows=1: {df.shape}")
    df = df.iloc[:, :3].copy()
    df.columns = ["row_id", "x", "y"]
    df[["x", "y"]] = df[["x", "y"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def read_rds_as_df(path: str) -> pd.DataFrame:
    """
    Read a matrix/data.frame-like RDS into pandas DataFrame.
    Try pyreadr, then pyreadstat fallback.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        res = pyreadr.read_r(path)
        if len(res) == 0:
            raise ValueError(f"No objects found in RDS: {path}")
        obj = next(iter(res.values()))
        if isinstance(obj, pd.DataFrame):
            return obj
        raise TypeError(f"pyreadr returned {type(obj)} for {path}")
    except Exception as e1:
        if pyreadstat is None:
            raise RuntimeError(
                f"pyreadr failed to read {path}:\n{e1}\n"
                f"Install pyreadstat for fallback: pip install pyreadstat"
            )
        df, meta = pyreadstat.read_rds(path)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"pyreadstat returned {type(df)} for {path}")
        return df


def orient_df_to_match_rows(df: pd.DataFrame, n_rows: int, name: str) -> pd.DataFrame:
    r, c = df.shape
    if r == n_rows:
        return df
    if c == n_rows:
        print(f"[TRANSPOSE] {name} {df.shape} -> {df.T.shape}")
        return df.T
    raise ValueError(f"{name} shape {df.shape} does not match n_rows={n_rows} in rows or cols.")


def find_feature_index(names, query: str) -> int:
    """
    Find feature index in a list-like by exact match first, then canonical match.
    """
    if query in names:
        return list(names).index(query)
    q = canonical_name(query)
    canon = [canonical_name(x) for x in names]
    if q in canon:
        return canon.index(q)
    raise KeyError(f"Cannot find feature '{query}' (canonical='{q}') in provided names.")


def clip_range(v, low=1.0, high=99.0):
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return 0.0, 1.0
    vmin = np.percentile(vv, low)
    vmax = np.percentile(vv, high)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(vv))
        vmax = float(np.nanmax(vv))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return 0.0, 1.0
    return float(vmin), float(vmax)


# -------------------------
# Grid similarity
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

    def corr(a, b):
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom = np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0))
        return np.nan if denom == 0 else float(np.sum(a0 * b0) / denom)

    pear = corr(t, p)

    tr = pd.Series(t).rank(method="average").to_numpy()
    pr = pd.Series(p).rank(method="average").to_numpy()
    spear = corr(tr, pr)

    t_mean = t.mean()
    ss_tot = np.sum((t - t_mean) ** 2)
    ss_res = np.sum((t - p) ** 2)
    r2 = np.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    mae = float(np.mean(np.abs(t - p)))

    return {"n": int(t.size), "pearson": pear, "spearman": spear, "r2": r2, "rmse": rmse, "mae": mae}


# -------------------------
# Plotting
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def _alpha_from_values(v, a_min=0.05, a_max=0.95, p_low=5, p_high=99):
    """
    Map values -> alpha in [a_min, a_max] using percentile clipping.
    Handles negative values gracefully.
    """
    v = np.asarray(v, dtype=float)
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return np.full_like(v, a_min, dtype=float)

    lo = np.percentile(vv, p_low)
    hi = np.percentile(vv, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(vv)
        hi = np.nanmax(vv)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.full_like(v, a_min, dtype=float)

    x = (v - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return a_min + x * (a_max - a_min)

def plot_spatial_overlay_gene_protein(
    x, y,
    gene_vals, prot_vals,
    gene_name, prot_name,
    out_png,
    invert_y=True,
    s=1.0,
    gene_rgb=(0.10, 0.35, 0.95),   # blue
    prot_rgb=(0.95, 0.25, 0.10),   # red/orange
    a_min=0.05,
    a_max=0.95,
    p_low=5,
    p_high=99,
    max_points=200000,
    seed=0,
):
    """
    Single overlay plot:
      - gene points in one color, alpha scaled by gene intensity
      - protein points in another color, alpha scaled by protein intensity
    """
    x = np.asarray(x); y = np.asarray(y)
    g = np.asarray(gene_vals); p = np.asarray(prot_vals)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(g) & np.isfinite(p)
    x, y, g, p = x[m], y[m], g[m], p[m]

    # downsample for plotting only
    if max_points and x.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, size=max_points, replace=False)
        x, y, g, p = x[idx], y[idx], g[idx], p[idx]

    a_g = _alpha_from_values(g, a_min=a_min, a_max=a_max, p_low=p_low, p_high=p_high)
    a_p = _alpha_from_values(p, a_min=a_min, a_max=a_max, p_low=p_low, p_high=p_high)

    # RGBA per point
    gene_rgba = np.zeros((x.size, 4), dtype=float)
    gene_rgba[:, 0:3] = gene_rgb
    gene_rgba[:, 3] = a_g

    prot_rgba = np.zeros((x.size, 4), dtype=float)
    prot_rgba[:, 0:3] = prot_rgb
    prot_rgba[:, 3] = a_p

    plt.figure(figsize=(7.5, 7.0), dpi=220)

    # draw gene first, then protein on top (usually easier to see protein boundaries)
    plt.scatter(x, y, c=gene_rgba, s=s, linewidths=0)
    plt.scatter(x, y, c=prot_rgba, s=s, linewidths=0)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    ax.axis("off")
    plt.title(f"{prot_name} (protein) + {gene_name} (gene) overlay")

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=f"Gene: {gene_name}",
               markerfacecolor=gene_rgb, markersize=8),
        Line2D([0], [0], marker='o', color='w', label=f"Protein: {prot_name}",
               markerfacecolor=prot_rgb, markersize=8),
    ]
    plt.legend(handles=legend_handles, loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()





# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_folder", default="/media/huifang/data/registration/phenocycler/haotian")
    ap.add_argument("--gene_h5ad", default="adata_xenium_cell_level.h5ad")
    ap.add_argument("--xenium_loc_csv", default="LUAD_2_A_xenium_axis.csv")
    ap.add_argument("--protein_rds", default="xenium_protein_predict.rds")

    ap.add_argument("--gene", required=True, help="Gene name (e.g., SFTPB)")
    ap.add_argument("--protein", required=True, help="Protein name (e.g., Collagen-IV or Ki67)")

    ap.add_argument("--invert_y", action="store_true")
    ap.add_argument("--target_sum", type=float, default=1e4, help="normalize_total target sum for genes")
    ap.add_argument("--plot_max_points", type=int, default=200000, help="Downsample points for plotting only")
    ap.add_argument("--grid_bins", type=int, default=200)
    ap.add_argument("--grid_min_count", type=int, default=5)
    args = ap.parse_args()

    root = args.root_folder
    out_dir = os.path.join(root, "pair_viz")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load gene h5ad
    h5ad_path = os.path.join(root, args.gene_h5ad)
    print("[LOAD] gene h5ad:", h5ad_path)
    adata = ad.read_h5ad(h5ad_path)

    # coords: prefer obsm if present; else use CSV
    coords = None
    for key in ["spatial"]:
        if key in adata.obsm.keys():
            arr = np.asarray(adata.obsm[key])
            if arr.shape[1] >= 2:
                coords = pd.DataFrame(arr[:, :2], columns=["y", "x"])
                print(f"[INFO] using coords from adata.obsm['{key}']")
                break

    if coords is None:
        loc_path = os.path.join(root, args.xenium_loc_csv)
        print("[LOAD] xenium loc csv:", loc_path)
        loc_df = read_xenium_location_weird_csv(loc_path)
        coords = loc_df[["x", "y"]].copy()

    coords = coords.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    nC = coords.shape[0]

    # gene matrix
    Xg = adata.X
    if sp.issparse(Xg):
        Xg = Xg.tocsr()
    else:
        Xg = np.asarray(Xg)

    # orient by row count if transposed
    if Xg.shape[0] != nC and Xg.shape[1] == nC:
        print(f"[TRANSPOSE] gene matrix {Xg.shape} -> {(Xg.T).shape}")
        Xg = Xg.T

    if Xg.shape[0] != nC:
        raise ValueError(f"Gene matrix rows do not match coords: X={Xg.shape}, coords={coords.shape}")

    # normalize + log1p (required)
    print("[PROC] normalize_total + log1p for gene matrix...")
    Xg_n = normalize_log1p_matrix(Xg, target_sum=args.target_sum)

    # extract gene vector
    gene_names = list(adata.var_names)
    gi = find_feature_index(gene_names, args.gene)
    gene_name = gene_names[gi]

    if sp.issparse(Xg_n):
        Xg_n = Xg_n.tocsr()  # extra safety
        gene_vals = Xg_n.getcol(gi).toarray().ravel()
    else:
        gene_vals = np.asarray(Xg_n[:, gi]).ravel()

    # ---- Load protein predicted RDS + align to coords by row count
    prot_path = os.path.join(root, args.protein_rds)
    print("[LOAD] protein rds:", prot_path)
    pred_df = read_rds_as_df(prot_path)
    pred_df = orient_df_to_match_rows(pred_df, nC, "predicted_protein")
    pred_df = pred_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # find protein column
    prot_cols = list(pred_df.columns)
    pj = find_feature_index(prot_cols, args.protein)
    prot_name = prot_cols[pj]
    prot_vals = pred_df[prot_name].to_numpy()

    # ---- optional downsample for plotting only
    x = coords["x"].to_numpy()
    y = coords["y"].to_numpy()
    n_all = len(x)

    if args.plot_max_points and n_all > args.plot_max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_all, size=args.plot_max_points, replace=False)
        x_plot, y_plot = x[idx], y[idx]
        gene_plot = gene_vals[idx]
        prot_plot = prot_vals[idx]
    else:
        x_plot, y_plot = x, y
        gene_plot = gene_vals
        prot_plot = prot_vals

    tag = safe_name(prot_name) + "__" + safe_name(gene_name)

    # ---- 1) Spatial side-by-side
    out_overlay = os.path.join(out_dir, tag + "__overlay.png")
    plot_spatial_overlay_gene_protein(
        x=x_plot, y=y_plot,
        gene_vals=gene_plot,
        prot_vals=prot_plot,
        gene_name=gene_name,
        prot_name=prot_name,
        out_png=out_overlay,
        invert_y=args.invert_y,
        s=1.0,
        max_points=args.plot_max_points,
        seed=0
    )
    print("[SAVED]", out_overlay)

if __name__ == "__main__":
    main()
