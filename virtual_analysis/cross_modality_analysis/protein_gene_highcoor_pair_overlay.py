#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch overlay plots for "high-correlation" protein–gene pairs from xenium_DE.rds.

Default input files (under --root_folder):
  - xenium_DE.rds
  - adata_xenium_cell_level.h5ad
  - xenium_protein_predict.rds
  - LUAD_2_A_xenium_axis.csv  (invalid first line like ",0,1")

Definition of "high-correlation pairs":
  For each protein marker, take top-K genes by adj.P.Val (FDR), tie-break by |logFC| desc.

Outputs:
  <root_folder>/pair_overlays_topK/
    - manifest_topK_pairs.csv
    - per_marker/<marker>/<marker>__<gene>__logFC...__FDR....png
"""

import os
import re
import argparse
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import pyreadr

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def _robust_minmax(v, p_low=5, p_high=99):
    v = np.asarray(v, dtype=float)
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return 0.0, 1.0
    lo = np.percentile(vv, p_low)
    hi = np.percentile(vv, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(vv); hi = np.nanmax(vv)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0, 1.0
    return float(lo), float(hi)

def _scale_01(v, p_low=5, p_high=99):
    lo, hi = _robust_minmax(v, p_low, p_high)
    v = np.asarray(v, dtype=float)
    x = (v - lo) / (hi - lo + 1e-12)
    return np.clip(x, 0.0, 1.0)

def plot_ratio_alpha_3color(
    x, y,
    gene_vals, prot_vals,
    gene_name, prot_name,
    out_png,
    invert_y=True,
    s=1.0,
    p_low=5,
    p_high=99,
    a_min=0.05,
    a_max=0.95,
    # endpoint colors
    prot_rgb=(0.10, 0.35, 0.95),   # protein-dominant (blue)
    gene_rgb=(0.95, 0.25, 0.10),   # gene-dominant (red/orange)
    # third color for "both-strong"
    both_rgb=(0.55, 0.05, 0.65),   # purple (change if you prefer)
    # how strongly to push toward both_rgb when ratio~0.5 & total high
    both_gamma=1.6,                # larger -> more localized around 0.5
    max_points=200000,
    seed=0,
):
    x = np.asarray(x); y = np.asarray(y)
    g = np.asarray(gene_vals); p = np.asarray(prot_vals)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(g) & np.isfinite(p)
    x, y, g, p = x[m], y[m], g[m], p[m]

    # downsample for plotting only
    if max_points and x.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, size=max_points, replace=False)
        x, y, g, p = x[idx], y[idx], g[idx], p[idx]

    # robust scale each modality to 0..1
    g01 = _scale_01(g, p_low=p_low, p_high=p_high)
    p01 = _scale_01(p, p_low=p_low, p_high=p_high)

    # ratio and total intensity
    frac_gene = g01 / (g01 + p01 + 1e-8)        # 0..1
    total01 = np.clip((g01 + p01) / 2.0, 0, 1)  # 0..1

    # alpha from total intensity
    alpha = a_min + total01 * (a_max - a_min)

    # --- color design ---
    # base mix (ratio only): protein_rgb -> gene_rgb
    prot_rgb = np.array(prot_rgb, dtype=float)
    gene_rgb = np.array(gene_rgb, dtype=float)
    both_rgb = np.array(both_rgb, dtype=float)

    base = (1.0 - frac_gene)[:, None] * prot_rgb[None, :] + frac_gene[:, None] * gene_rgb[None, :]

    # "both strong" weight:
    #   high when ratio near 0.5 AND total is high
    center = 1.0 - np.clip(np.abs(frac_gene - 0.5) / 0.5, 0.0, 1.0)  # 0..1, peaks at 0.5
    w_both = (center ** both_gamma) * total01                         # 0..1

    # final RGB: push base toward both_rgb only where both-strong weight is high
    rgb = (1.0 - w_both)[:, None] * base + w_both[:, None] * both_rgb[None, :]

    rgba = np.concatenate([rgb, alpha[:, None]], axis=1)

    # plot
    plt.figure(figsize=(7.6, 7.0), dpi=220)
    plt.scatter(x, y, c=rgba, s=s, linewidths=0)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
        ax.invert_xaxis()
    ax.axis("off")
    plt.title(f"Gene={gene_name}, Protein={prot_name}")

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', label=f"Protein-dominant (p≈0): {prot_name}",
               markerfacecolor=prot_rgb, markersize=8),
        Line2D([0], [0], marker='o', color='w', label="Both-strong (p≈0.5, high total)",
               markerfacecolor=both_rgb, markersize=8),
        Line2D([0], [0], marker='o', color='w', label=f"Gene-dominant (p≈1): {gene_name}",
               markerfacecolor=gene_rgb, markersize=8),
    ]
    plt.legend(handles=handles, loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _robust_minmax(v, p_low=5, p_high=99):
    v = np.asarray(v, dtype=float)
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return 0.0, 1.0
    lo = np.percentile(vv, p_low)
    hi = np.percentile(vv, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(vv); hi = np.nanmax(vv)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0, 1.0
    return float(lo), float(hi)

def _scale_01(v, p_low=5, p_high=99):
    lo, hi = _robust_minmax(v, p_low, p_high)
    v = np.asarray(v, dtype=float)
    x = (v - lo) / (hi - lo + 1e-12)
    return np.clip(x, 0.0, 1.0)

def plot_ratio_alpha(
    x, y,
    gene_vals, prot_vals,
    gene_name, prot_name,
    out_png,
    invert_y=True,
    s=1.0,
    # scaling (important because gene/protein are on different scales)
    p_low=5,
    p_high=99,
    # alpha mapping
    a_min=0.05,
    a_max=0.95,
    # colormap for ratio: 0=protein, 1=gene
    cmap="coolwarm",
    max_points=200000,
    seed=0,
):
    x = np.asarray(x); y = np.asarray(y)
    g = np.asarray(gene_vals); p = np.asarray(prot_vals)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(g) & np.isfinite(p)
    x, y, g, p = x[m], y[m], g[m], p[m]

    # downsample for plotting only
    if max_points and x.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, size=max_points, replace=False)
        x, y, g, p = x[idx], y[idx], g[idx], p[idx]

    # scale each to 0..1 with robust clipping
    g01 = _scale_01(g, p_low=p_low, p_high=p_high)
    p01 = _scale_01(p, p_low=p_low, p_high=p_high)

    # ratio (gene fraction)
    frac_gene = g01 / (g01 + p01 + 1e-8)     # in [0,1]

    # intensity for alpha (total strength)
    total = g01 + p01                        # in [0,2]
    total01 = np.clip(total / 2.0, 0.0, 1.0) # back to [0,1]
    alpha = a_min + total01 * (a_max - a_min)

    # map ratio -> RGB, then attach alpha
    cm = mpl.cm.get_cmap(cmap)
    rgb = cm(frac_gene)[:, :3]
    rgba = np.concatenate([rgb, alpha[:, None]], axis=1)

    plt.figure(figsize=(7.6, 7.0), dpi=220)
    plt.scatter(x, y, c=rgba, s=s, linewidths=0)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    ax.axis("off")

    plt.title(f"Ratio map (color) + intensity (alpha)\nGene={gene_name}, Protein={prot_name}")

    # colorbar for ratio
    sm = mpl.cm.ScalarMappable(cmap=cm, norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.03, pad=0.02)
    cbar.set_label(f"Gene fraction  p = gene / (gene + protein)\n0={prot_name}, 1={gene_name}")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# -------------------------
# General helpers
# -------------------------
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))


def canonical_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(".", "").replace("-", "").replace("_", "").replace(" ", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def read_rds_df(path: str) -> pd.DataFrame:
    """Read an RDS data.frame into pandas. Try pyreadr then pyreadstat."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    res = pyreadr.read_r(path)
    if len(res) == 0:
        raise ValueError(f"No objects in RDS: {path}")
    obj = next(iter(res.values()))
    if isinstance(obj, pd.DataFrame):
        return obj


def ensure_numeric_df(df: pd.DataFrame, fillna=0.0) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(fillna)
    return out


def read_xenium_location_weird_csv(path: str) -> pd.DataFrame:
    """CSV with invalid first line like ',0,1', then rows: idx,x,y"""
    df = pd.read_csv(path, header=None, skiprows=1)
    if df.shape[1] < 3:
        raise ValueError(f"Xenium location CSV has <3 cols: {df.shape}")
    df = df.iloc[:, :3].copy()
    df.columns = ["row_id", "x", "y"]
    df[["x", "y"]] = ensure_numeric_df(df[["x", "y"]])
    return df


def orient_df_to_match_rows(df: pd.DataFrame, n_rows: int, name: str) -> pd.DataFrame:
    r, c = df.shape
    if r == n_rows:
        return df
    if c == n_rows:
        print(f"[TRANSPOSE] {name} {df.shape} -> {df.T.shape}")
        return df.T
    raise ValueError(f"{name} shape {df.shape} does not match n_rows={n_rows} in rows or cols.")


def build_canon_map(names):
    """canonical -> actual (first occurrence)"""
    m = {}
    for n in names:
        k = canonical_name(n)
        if k and k not in m:
            m[k] = n
    return m


# -------------------------
# Efficient gene normalization/log extraction on-the-fly
# -------------------------
def compute_row_scale(X_csr, target_sum: float):
    """scale[i] = target_sum / row_sum[i] (0 if row_sum==0)."""
    row_sum = np.asarray(X_csr.sum(axis=1)).ravel().astype(np.float32)
    scale = np.zeros_like(row_sum, dtype=np.float32)
    nz = row_sum > 0
    scale[nz] = (target_sum / row_sum[nz]).astype(np.float32)
    return scale


def extract_gene_log1p_on_sample(X, X_is_sparse, X_csc, scale, gene_idx, sample_idx, pos_map):
    """
    Return gene values on sample_idx after normalize_total + log1p.
    Does NOT materialize full normalized matrix.
    """
    n_s = sample_idx.size
    out = np.zeros(n_s, dtype=np.float32)

    if not X_is_sparse:
        raw = X[sample_idx, gene_idx].astype(np.float32, copy=False)
        out = np.log1p(raw * scale[sample_idx]).astype(np.float32, copy=False)
        return out

    # sparse: use CSC column, fill only sample positions with nonzeros
    col = X_csc.getcol(gene_idx).tocoo()
    # positions in sample for each nonzero row
    p = pos_map[col.row]
    m = p >= 0
    if np.any(m):
        out[p[m]] = (col.data[m].astype(np.float32, copy=False) * scale[col.row[m]]).astype(np.float32, copy=False)
    out = np.log1p(out).astype(np.float32, copy=False)
    return out


# -------------------------
# Overlay plotting (gene + protein in one plot with alpha from intensity)
# -------------------------
def _alpha_from_values(v, a_min=0.05, a_max=0.95, p_low=5, p_high=99):
    v = np.asarray(v, dtype=float)
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return np.full_like(v, a_min, dtype=float)
    lo = np.percentile(vv, p_low)
    hi = np.percentile(vv, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(vv); hi = np.nanmax(vv)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.full_like(v, a_min, dtype=float)
    x = (v - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return a_min + x * (a_max - a_min)


def plot_overlay_gene_protein(
    x, y, gene_vals, prot_vals,
    gene_name, prot_name,
    out_png,
    title_extra="",
    invert_y=True,
    s=1.0,
    gene_rgb=(0.10, 0.35, 0.95),   # blue
    prot_rgb=(0.95, 0.25, 0.10),   # red/orange
    a_min=0.05,
    a_max=0.95,
    p_low=5,
    p_high=99,
):
    x = np.asarray(x); y = np.asarray(y)
    g = np.asarray(gene_vals); p = np.asarray(prot_vals)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(g) & np.isfinite(p)
    x, y, g, p = x[m], y[m], g[m], p[m]

    a_g = _alpha_from_values(g, a_min=a_min, a_max=a_max, p_low=p_low, p_high=p_high)
    a_p = _alpha_from_values(p, a_min=a_min, a_max=a_max, p_low=p_low, p_high=p_high)

    gene_rgba = np.zeros((x.size, 4), dtype=float)
    gene_rgba[:, :3] = gene_rgb
    gene_rgba[:, 3] = a_g

    prot_rgba = np.zeros((x.size, 4), dtype=float)
    prot_rgba[:, :3] = prot_rgb
    prot_rgba[:, 3] = a_p

    plt.figure(figsize=(7.5, 7.0), dpi=220)
    # gene first, protein on top
    plt.scatter(x, y, c=gene_rgba, s=s, linewidths=0)
    plt.scatter(x, y, c=prot_rgba, s=s, linewidths=0)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    ax.axis("off")

    title = f"{prot_name} (protein) + {gene_name} (gene)"
    if title_extra:
        title += f"\n{title_extra}"
    plt.title(title)

    # simple legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', label=f"Gene: {gene_name}",
               markerfacecolor=gene_rgb, markersize=8),
        Line2D([0], [0], marker='o', color='w', label=f"Protein: {prot_name}",
               markerfacecolor=prot_rgb, markersize=8),
    ]
    plt.legend(handles=handles, loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# -------------------------
# Pair selection from DE
# -------------------------
def topk_pairs_per_marker(df_de: pd.DataFrame, k: int):
    d = df_de.copy()
    d["logFC"] = pd.to_numeric(d["logFC"], errors="coerce")
    d["AveExpr"] = pd.to_numeric(d["AveExpr"], errors="coerce")
    d["adj.P.Val"] = pd.to_numeric(d["adj.P.Val"], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["marker", "gene", "logFC", "adj.P.Val"])
    d["abs_logFC"] = d["logFC"].abs()

    # guard: remove duplicated columns if any upstream bug reintroduces them
    if d.columns.duplicated().any():
        d = d.loc[:, ~d.columns.duplicated()].copy()

    d = d.sort_values(["marker", "adj.P.Val", "abs_logFC"], ascending=[True, True, False])
    return d.groupby("marker", group_keys=False).head(k).copy()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_folder", default="/media/huifang/data/registration/phenocycler/haotian")

    ap.add_argument("--de_rds", default="xenium_DE.rds")
    ap.add_argument("--gene_h5ad", default="adata_xenium_cell_level.h5ad")
    ap.add_argument("--protein_rds", default="xenium_protein_predict.rds")
    ap.add_argument("--xenium_loc_csv", default="LUAD_2_A_xenium_axis.csv")

    ap.add_argument("--topk", type=int, default=10, help="Top-K genes per protein marker (by FDR)")
    ap.add_argument("--target_sum", type=float, default=1e4, help="normalize_total target sum for genes")
    ap.add_argument("--max_pairs", type=int, default=0, help="Optional cap on total pairs (0 = no cap)")
    ap.add_argument("--plot_points", type=int, default=200000, help="Downsample cells for plotting only")
    ap.add_argument("--point_size", type=float, default=1.0)
    ap.add_argument("--invert_y", default=True)
    args = ap.parse_args()

    root = args.root_folder
    out_root = os.path.join(root, f"pair_overlays_top{args.topk}")
    os.makedirs(out_root, exist_ok=True)
    out_per_marker = os.path.join(out_root, "per_marker_ratio")
    os.makedirs(out_per_marker, exist_ok=True)

    # ---- Load DE
    de_path = os.path.join(root, args.de_rds)
    print("[LOAD] DE:", de_path)
    df_de = read_rds_df(de_path)


    required = {"logFC", "AveExpr", "adj.P.Val", "gene", "marker"}
    if not required.issubset(df_de.columns):
        raise ValueError(f"DE missing columns {required - set(df_de.columns)}. Columns: {list(df_de.columns)}")

    # ---- Load xenium coords
    loc_path = os.path.join(root, args.xenium_loc_csv)
    print("[LOAD] Xenium coords:", loc_path)
    loc_df = read_xenium_location_weird_csv(loc_path)
    coords = loc_df[["x", "y"]].copy()
    nC = coords.shape[0]
    x_all = coords["y"].to_numpy(dtype=np.float32)
    y_all = coords["x"].to_numpy(dtype=np.float32)

    # ---- Choose plotting sample indices ONCE
    rng = np.random.default_rng(0)
    if args.plot_points and nC > args.plot_points:
        sample_idx = rng.choice(nC, size=args.plot_points, replace=False).astype(np.int32)
    else:
        sample_idx = np.arange(nC, dtype=np.int32)

    # mapping row -> position in sample, for sparse extraction
    pos_map = np.full(nC, -1, dtype=np.int32)
    pos_map[sample_idx] = np.arange(sample_idx.size, dtype=np.int32)

    x_plot = x_all[sample_idx]
    y_plot = y_all[sample_idx]

    # ---- Load predicted protein matrix and align to Xenium rows
    prot_path = os.path.join(root, args.protein_rds)
    print("[LOAD] Predicted protein:", prot_path)
    pred_df = ensure_numeric_df(read_rds_df(prot_path))
    pred_df = orient_df_to_match_rows(pred_df, nC, "predicted_protein")
    pred_df = ensure_numeric_df(pred_df)

    prot_map = build_canon_map(pred_df.columns)

    # ---- Load gene h5ad
    h5ad_path = os.path.join(root, args.gene_h5ad)
    print("[LOAD] Gene h5ad:", h5ad_path)
    adata = ad.read_h5ad(h5ad_path)

    gene_names = list(adata.var_names)
    gene_map = build_canon_map(gene_names)

    X = adata.X
    X_is_sparse = sp.issparse(X)
    if X_is_sparse:
        X_csr = X.tocsr()
        X_csc = X.tocsc()
    else:
        X_csr = None
        X_csc = None
        X = np.asarray(X)

    # orient genes if transposed
    if (X_csr is not None and X_csr.shape[0] != nC and X_csr.shape[1] == nC) or (X_csr is None and X.shape[0] != nC and X.shape[1] == nC):
        print("[TRANSPOSE] Gene matrix appears transposed; transposing.")
        if X_is_sparse:
            X_csr = X_csr.T.tocsr()
            X_csc = X_csc.T.tocsc()
        else:
            X = X.T

    # final row check
    n_rows = X_csr.shape[0] if X_is_sparse else X.shape[0]
    if n_rows != nC:
        raise ValueError(f"Gene matrix rows do not match xenium coords: gene_rows={n_rows}, coords={nC}")

    # precompute row scaling once
    if X_is_sparse:
        scale = compute_row_scale(X_csr, args.target_sum)
    else:
        row_sum = X.sum(axis=1).astype(np.float32)
        scale = np.zeros_like(row_sum, dtype=np.float32)
        nz = row_sum > 0
        scale[nz] = (args.target_sum / row_sum[nz]).astype(np.float32)

    # ---- Filter DE to only available genes/proteins (canonical match)
    df_de = df_de.copy()
    df_de["marker_c"] = df_de["marker"].map(canonical_name)
    df_de["gene_c"] = df_de["gene"].map(canonical_name)

    df_de = df_de[df_de["marker_c"].isin(prot_map.keys()) & df_de["gene_c"].isin(gene_map.keys())].copy()
    if df_de.empty:
        raise ValueError("After canonical filtering, DE table has no rows matching available genes/proteins.")

    # Map to actual names in your data objects (no duplicate columns)
    df_sel = df_de.copy()

    # write actual names into the existing columns (no duplicates)
    df_sel["marker"] = df_sel["marker_c"].map(prot_map)
    df_sel["gene"] = df_sel["gene_c"].map(gene_map)

    # drop helpers if you want
    df_sel = df_sel.drop(columns=["marker_c", "gene_c"], errors="ignore")

    # now safe: only one 'marker' and one 'gene'
    df_sel = topk_pairs_per_marker(df_sel, k=args.topk)
    # Optional cap
    if args.max_pairs and df_sel.shape[0] > args.max_pairs:
        df_sel = df_sel.sort_values(["adj.P.Val", "abs_logFC"], ascending=[True, False]).head(args.max_pairs).copy()

    # Save manifest
    manifest_path = os.path.join(out_root, "manifest_topK_pairs.csv")
    df_sel.sort_values(["marker", "adj.P.Val"]).to_csv(manifest_path, index=False)
    print("[SAVED] pair manifest:", manifest_path)
    print(f"[INFO] plotting {df_sel.shape[0]} overlays on {sample_idx.size} sampled cells")

    # ---- caches for speed
    gene_cache = {}   # gene -> values on sample_idx (normalized+log1p)
    prot_cache = {}   # protein -> values on sample_idx (already logged/normalized)

    # quick lookup indices
    gene_index = {g: i for i, g in enumerate(gene_names)}
    prot_index = {p: p for p in pred_df.columns}  # name exists in columns

    # ---- iterate pairs
    df_sel = df_sel.copy()
    df_sel["FDR"] = pd.to_numeric(df_sel["adj.P.Val"], errors="coerce")
    df_sel["LOGFC"] = pd.to_numeric(df_sel["logFC"], errors="coerce")
    for i, row in enumerate(df_sel.itertuples(index=False), start=1):
        marker = getattr(row, "marker")
        gene = getattr(row, "gene")
        logfc = float(row.LOGFC)
        fdr = float(row.FDR)

        # cache protein values
        if marker not in prot_cache:
            prot_cache[marker] = pred_df[marker].to_numpy(dtype=np.float32, copy=False)[sample_idx]
        prot_vals = prot_cache[marker]

        # cache gene values (normalize_total + log1p)
        if gene not in gene_cache:
            gi = gene_index[gene]
            if X_is_sparse:
                gvals = extract_gene_log1p_on_sample(
                    X=None, X_is_sparse=True, X_csc=X_csc, scale=scale,
                    gene_idx=gi, sample_idx=sample_idx, pos_map=pos_map
                )
            else:
                gvals = np.log1p(X[sample_idx, gi].astype(np.float32) * scale[sample_idx]).astype(np.float32)
            gene_cache[gene] = gvals
        gene_vals = gene_cache[gene]

        # output path
        mk_dir = os.path.join(out_per_marker, safe_name(marker))
        os.makedirs(mk_dir, exist_ok=True)
        # out_png = os.path.join(
        #     mk_dir,
        #     f"{safe_name(marker)}__{safe_name(gene)}__logFC{logfc:+.3f}__FDR{fdr:.2e}.png"
        # )

        # title_extra = f"logFC={logfc:+.3f}, FDR={fdr:.2e}   ({i}/{df_sel.shape[0]})"
        # plot_overlay_gene_protein(
        #     x_plot, y_plot, gene_vals, prot_vals,
        #     gene_name=gene, prot_name=marker,
        #     out_png=out_png,
        #     title_extra=title_extra,
        #     invert_y=args.invert_y,
        #     s=args.point_size,
        # )

        out_ratio = os.path.join(
            mk_dir,
            f"{safe_name(marker)}__{safe_name(gene)}__logFC{logfc:+.3f}__FDR{fdr:.2e}.png"
        )
        # plot_ratio_alpha(
        #     x=x_plot, y=y_plot,
        #     gene_vals=gene_vals,
        #     prot_vals=prot_vals,
        #     gene_name=gene,
        #     prot_name=marker,
        #     out_png=out_ratio,
        #     invert_y=args.invert_y,
        #     s=1.0,
        #     max_points=args.plot_points,
        #     seed=0
        # )

        plot_ratio_alpha_3color(
            x=x_plot, y=y_plot,
            gene_vals=gene_vals,
            prot_vals=prot_vals,
            gene_name=gene,
            prot_name=marker,
            out_png=out_ratio,
            invert_y=args.invert_y,
            s=1.0,
            max_points=args.plot_points,
        )

        print("[SAVED]", out_ratio)

        if i % 10 == 0 or i == df_sel.shape[0]:
            print(f"[PROGRESS] {i}/{df_sel.shape[0]} saved", flush=True)

    print("[DONE] overlays saved under:", out_root)


if __name__ == "__main__":
    main()
