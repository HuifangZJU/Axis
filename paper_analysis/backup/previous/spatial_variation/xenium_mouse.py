#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TgCRND8 time-course: per-domain, per-gene spatial visualization with two coordinate systems.

Figure layout (2 x 3) for each (domain, gene):
Row 1: gene expression at 2.5m, 5.7m, 17.9m using obsm["spatial_axis"] (irregular, aligned)
Row 2:
  Col 1: domain location on full tissue using obsm["spatial_axis"] (all gray, current domain red) [uses 5.7m]
  Col 2: Δ(5.7 - 2.5) using obsm["spatial_axis_pixel"] (pixel-regularized; averaged per pixel)
  Col 3: Δ(17.9 - 5.7) using obsm["spatial_axis_pixel"] (pixel-regularized; averaged per pixel)

Normalization:
  norm_expr = log1p( (counts / library_size) * target_sum ), with target_sum=1e4
"""

import os
import io
import csv
import re
import warnings
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
# H5AD_ROOT = "/media/huifang/data/registration/xenium/mouse_brain/output/group_0/h5ads/Tg"
# RANK_ROOT = "/media/huifang/data/registration/xenium/mouse_brain/haotian/Tg"
#
# H5AD_25  = os.path.join(H5AD_ROOT, "xenium_Tg_2.5m.h5ad")
# H5AD_57  = os.path.join(H5AD_ROOT, "xenium_Tg_5.7m.h5ad")
# H5AD_179 = os.path.join(H5AD_ROOT, "xenium_Tg_17.9m.h5ad")


# H5AD_25  = os.path.join(H5AD_ROOT, "xenium_Tg_2.5m.h5ad")
# H5AD_57  = os.path.join(H5AD_ROOT, "xenium_Tg_5.7m.h5ad")
# H5AD_179 = os.path.join(H5AD_ROOT, "xenium_Tg_17.9m.h5ad")


# H5AD_ROOT = "/media/huifang/data/registration/xenium/mouse_brain/output/group_0/h5ads/widetype"
# RANK_ROOT = "/media/huifang/data/registration/xenium/mouse_brain/haotian/WT"
# H5AD_25  = os.path.join(H5AD_ROOT, "xenium_WT_2.5m.h5ad")
# H5AD_57  = os.path.join(H5AD_ROOT, "xenium_WT_5.7m.h5ad")
# H5AD_179 = os.path.join(H5AD_ROOT, "xenium_WT_13.4m.h5ad")
#
# RES12 = os.path.join(RANK_ROOT, "res12.csv")  # 5.7 vs 2.5
# RES23 = os.path.join(RANK_ROOT, "res23.csv")  # 17.9 vs 5.7


H5AD_ROOT = "/media/huifang/data/registration/xenium/mouse_brain/output/group_0/h5ads/unified"
RANK_ROOT = "/media/huifang/data/registration/xenium/mouse_brain/haotian/cross_condition"
H5AD_25  = os.path.join(H5AD_ROOT, "xenium_WT_5.7m.h5ad")
H5AD_57  = os.path.join(H5AD_ROOT, "xenium_WT_2.5m.h5ad")
H5AD_179 = os.path.join(H5AD_ROOT, "xenium_WT_13.4m.h5ad")

RES12 = os.path.join(RANK_ROOT, "resD12.csv")  # 5.7 vs 2.5
RES23 = os.path.join(RANK_ROOT, "resD23.csv")  # 17.9 vs 5.7
SPATIAL_AXIS_KEY = "spatial_axis"
SPATIAL_PIXEL_KEY = "spatial_axis_pixel"

# Domain IDs column in adata.obs. If None, will auto-detect.
DOMAIN_KEY = "domain_id_shared"

# If your raw counts are stored in a layer (e.g., "counts"), set it here.
# Otherwise the script uses adata.X.
COUNTS_LAYER = None  # e.g. "counts"

N_DOMAINS = 14
DOMAINS = list(range(N_DOMAINS))
TOPK = 5
TARGET_SUM = 1e4

OUTDIR = os.path.join(H5AD_ROOT, "domain_topgenes_spatial_changes_v2")
os.makedirs(OUTDIR, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\.\-]+", "_", s)
    return s[:200]



def read_rankings_csv(path: str) -> pd.DataFrame:
    """
    Robust CSV parser for lines like:
      domain","gene","fdr","logpval","pval","fstat","knotnum
    and rows:
      "0","Cpne4",...,0
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    lines = [ln.strip() for ln in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [ln for ln in lines if ln != ""]
    if len(lines) < 2:
        raise ValueError(f"{path}: not enough lines to parse.")

    header = lines[0]
    if header.startswith('domain","') or header.startswith('domain"'):
        header = '"' + header  # fix missing leading quote
    lines[0] = header

    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=",", quotechar='"', skipinitialspace=True)
    rows = list(reader)
    cols = [c.strip().strip('"').strip("'") for c in rows[0]]
    df = pd.DataFrame(rows[1:], columns=cols)
    df.columns = [c.replace('"', "").replace("'", "").strip() for c in df.columns]

    if "domain" not in df.columns or "gene" not in df.columns:
        raise ValueError(f"{path}: missing required columns. got={df.columns.tolist()}")

    df["domain"] = pd.to_numeric(df["domain"], errors="coerce")
    df["gene"] = df["gene"].astype(str)

    for c in ["fdr", "logpval", "pval", "fstat", "knotnum"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["domain", "gene"])
    df["domain"] = df["domain"].astype(int)

    return df


def pick_top_genes_per_domain(df: pd.DataFrame, topk: int = 5) -> Dict[int, List[str]]:
    for c in ["fdr", "fstat", "logpval"]:
        if c not in df.columns:
            df[c] = np.nan

    out = {}
    for d in DOMAINS:
        sub = df[df["domain"] == d].copy()
        if sub.empty:
            out[d] = []
            continue

        sub["fdr_fill"] = sub["fdr"].fillna(1.0)
        sub["fstat_fill"] = sub["fstat"].fillna(-np.inf)
        sub["logp_fill"] = sub["logpval"].fillna(-np.inf)

        sub = sub.sort_values(["fdr_fill", "fstat_fill", "logp_fill"],
                              ascending=[True, False, False])

        genes = sub["gene"].astype(str).tolist()
        seen, uniq = set(), []
        for g in genes:
            if g not in seen:
                seen.add(g)
                uniq.append(g)
            if len(uniq) >= topk:
                break
        out[d] = uniq
    return out


def get_counts_matrix(adata: ad.AnnData):
    if COUNTS_LAYER is not None:
        if COUNTS_LAYER not in adata.layers:
            raise KeyError(f"COUNTS_LAYER='{COUNTS_LAYER}' not found in adata.layers")
        return adata.layers[COUNTS_LAYER]
    return adata.X


def compute_size_factors(counts) -> np.ndarray:
    """
    size_factor[i] = TARGET_SUM / total_counts[i]
    """
    if sparse.issparse(counts):
        totals = np.asarray(counts.sum(axis=1)).ravel()
    else:
        totals = np.asarray(counts.sum(axis=1)).ravel()

    totals = totals.astype(np.float64, copy=False)
    totals[~np.isfinite(totals)] = 0.0
    sf = np.zeros_like(totals, dtype=np.float64)
    nz = totals > 0
    sf[nz] = TARGET_SUM / totals[nz]
    return sf


def extract_log_norm_gene(adata: ad.AnnData, gene: str, size_factors: np.ndarray) -> np.ndarray:
    """
    Returns 1D array of log-normalized expression for one gene:
      log1p(counts * size_factor)
    """
    if gene not in adata.var_names:
        raise KeyError(f"Gene '{gene}' not in adata.var_names")

    counts = get_counts_matrix(adata)
    idx = int(np.where(adata.var_names == gene)[0][0])

    col = counts[:, idx]
    if sparse.issparse(col):
        v = col.toarray().ravel().astype(np.float64, copy=False)
    else:
        v = np.asarray(col).ravel().astype(np.float64, copy=False)

    v = v * size_factors
    v = np.log1p(v)
    return v.astype(np.float32, copy=False)


def aggregate_pixels_log_norm(
    adata: ad.AnnData,
    genes: List[str],
    domain_key: str,
    size_factors: np.ndarray,
    pixel_key: str = SPATIAL_PIXEL_KEY,
    chunk_size: int = 64,
) -> pd.DataFrame:
    """
    Compute mean log-normalized expression per (domain, xpix, ypix) for all genes in `genes`.
    Uses obsm[pixel_key], rounds to integer pixel.
    """
    if pixel_key not in adata.obsm_keys():
        raise KeyError(f"Missing obsm['{pixel_key}']")

    coords = np.asarray(adata.obsm[pixel_key])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"obsm['{pixel_key}'] expected Nx2, got {coords.shape}")

    xpix = np.rint(coords[:, 0]).astype(np.int32)
    ypix = np.rint(coords[:, 1]).astype(np.int32)

    dom = pd.to_numeric(adata.obs[domain_key], errors="coerce").fillna(-1).astype(np.int32).values
    valid = (dom >= 0) & (dom < N_DOMAINS)
    if valid.sum() == 0:
        raise ValueError(f"No valid domain IDs 0..{N_DOMAINS-1} found in '{domain_key}'")

    xpix = xpix[valid]
    ypix = ypix[valid]
    dom = dom[valid]
    sf = size_factors[valid]
    ad_sub = adata[valid].copy()

    genes_present = [g for g in genes if g in ad_sub.var_names]
    missing = [g for g in genes if g not in ad_sub.var_names]
    if missing:
        warnings.warn(f"{len(missing)} genes missing in this h5ad (ignored). Example: {missing[:5]}")

    meta = pd.DataFrame({"domain": dom, "x": xpix, "y": ypix})
    out = meta.drop_duplicates().copy()

    if not genes_present:
        return out

    counts = get_counts_matrix(ad_sub)

    for i in range(0, len(genes_present), chunk_size):
        chunk = genes_present[i:i+chunk_size]
        idxs = [int(np.where(ad_sub.var_names == g)[0][0]) for g in chunk]

        X = counts[:, idxs]
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        # normalize + log
        X *= sf[:, None]
        X = np.log1p(X).astype(np.float32, copy=False)

        tmp = pd.DataFrame(X, columns=chunk)
        tmp = pd.concat([meta.reset_index(drop=True), tmp], axis=1)
        tmp = tmp.groupby(["domain", "x", "y"], as_index=False).mean(numeric_only=True)

        out = out.merge(tmp, on=["domain", "x", "y"], how="left")

    return out


def compute_delta_pixel(df_a: pd.DataFrame, df_b: pd.DataFrame, gene: str) -> pd.DataFrame:
    """
    delta = b - a aligned on (x,y); expects columns x,y,gene
    """
    key = ["x", "y"]
    a = df_a[key + [gene]].dropna()
    b = df_b[key + [gene]].dropna()
    m = a.merge(b, on=key, how="inner", suffixes=("_a", "_b"))
    if m.empty:
        return pd.DataFrame(columns=["x", "y", "delta"])
    m["delta"] = m[f"{gene}_b"] - m[f"{gene}_a"]
    return m[["x", "y", "delta"]]


def plot_gene_domain_2x3(
    domain_id: int,
    gene: str,
    tp25: dict,
    tp57: dict,
    tp179: dict,
    df25_pix: pd.DataFrame,
    df57_pix: pd.DataFrame,
    df179_pix: pd.DataFrame,
    out_png: str,
    out_pdf: str,
):
    """
    Updated plotting:
      1) Fixed subplot sizes (colorbars in separate cax axes; do NOT shrink subplots)
      2) Colorbars placed outside to avoid overlap
      3) Shared axis limits for row1 (3 timepoints) + row2 deltas
    """
    # --- masks for domain ---
    dom25 = tp25["dom"]
    dom57 = tp57["dom"]
    dom179 = tp179["dom"]

    m25 = dom25 == domain_id
    m57 = dom57 == domain_id
    m179 = dom179 == domain_id

    if (m25.sum() == 0) and (m57.sum() == 0) and (m179.sum() == 0):
        warnings.warn(f"Domain {domain_id}: no points in any timepoint. Skipping {gene}.")
        return

    # --- gene availability ---
    gene_ok = [
        (gene in tp25["adata"].var_names),
        (gene in tp57["adata"].var_names),
        (gene in tp179["adata"].var_names),
    ]

    expr25 = expr57 = expr179 = None
    if gene_ok[0]:
        expr25 = extract_log_norm_gene(tp25["adata"], gene, tp25["size_factors"])[m25]
    if gene_ok[1]:
        expr57 = extract_log_norm_gene(tp57["adata"], gene, tp57["size_factors"])[m57]
    if gene_ok[2]:
        expr179 = extract_log_norm_gene(tp179["adata"], gene, tp179["size_factors"])[m179]

    # --- shared vmax for expression (within this domain across available tps) ---
    expr_all = []
    for e in (expr25, expr57, expr179):
        if e is not None and e.size > 0:
            expr_all.append(e)
    if len(expr_all) == 0:
        vmax = 1.0
    else:
        expr_all = np.concatenate(expr_all)
        vmax = float(np.nanpercentile(expr_all, 99))
        if (not np.isfinite(vmax)) or vmax <= 0:
            vmax = float(np.nanmax(expr_all)) if expr_all.size else 1.0
        vmax = max(vmax, 1e-6)

    # --- pixel-level deltas (already log-normalized in df*_pix) ---
    d25p = df25_pix[df25_pix["domain"] == domain_id]
    d57p = df57_pix[df57_pix["domain"] == domain_id]
    d179p = df179_pix[df179_pix["domain"] == domain_id]

    delta12 = pd.DataFrame(columns=["x", "y", "delta"])
    delta23 = pd.DataFrame(columns=["x", "y", "delta"])
    if (gene in d25p.columns) and (gene in d57p.columns):
        delta12 = compute_delta_pixel(d25p, d57p, gene)
    if (gene in d57p.columns) and (gene in d179p.columns):
        delta23 = compute_delta_pixel(d57p, d179p, gene)

    deltas_all = []
    if not delta12.empty:
        deltas_all.append(delta12["delta"].values)
    if not delta23.empty:
        deltas_all.append(delta23["delta"].values)
    if len(deltas_all) == 0:
        dmax = 1.0
    else:
        deltas_all = np.concatenate(deltas_all)
        dmax = float(np.nanpercentile(np.abs(deltas_all), 99))
        if (not np.isfinite(dmax)) or dmax <= 0:
            dmax = float(np.nanmax(np.abs(deltas_all))) if deltas_all.size else 1.0
        dmax = max(dmax, 1e-6)

    # --- coords for spatial_axis panels ---
    coords25 = np.asarray(tp25["adata"].obsm[SPATIAL_AXIS_KEY])[m25]
    coords57 = np.asarray(tp57["adata"].obsm[SPATIAL_AXIS_KEY])[m57]
    coords179 = np.asarray(tp179["adata"].obsm[SPATIAL_AXIS_KEY])[m179]

    # shared axis limits for row1 (domain-only across 3 tps)
    coords_domain_all = []
    for c in (coords25, coords57, coords179):
        if c is not None and c.shape[0] > 0:
            coords_domain_all.append(c)
    if len(coords_domain_all) > 0:
        C = np.vstack(coords_domain_all)
        xmin, ymin = C.min(axis=0)
        xmax, ymax = C.max(axis=0)
        pad_x = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
        pad_y = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
        xlim_row1 = (xmin - pad_x, xmax + pad_x)
        ylim_row1 = (ymin - pad_y, ymax + pad_y)
    else:
        xlim_row1 = None
        ylim_row1 = None

    # shared axis limits for delta panels (use union of domain pixels across all tps)
    pix_coords_all = []
    for dfp in (d25p, d57p, d179p):
        if (dfp is not None) and (not dfp.empty):
            pix_coords_all.append(dfp[["x", "y"]].to_numpy())
    if len(pix_coords_all) > 0:
        P = np.vstack(pix_coords_all)
        pxmin, pymin = P.min(axis=0)
        pxmax, pymax = P.max(axis=0)
        ppad_x = (pxmax - pxmin) * 0.05 if pxmax > pxmin else 1.0
        ppad_y = (pymax - pymin) * 0.05 if pymax > pymin else 1.0
        xlim_delta = (pxmin - ppad_x, pxmax + ppad_x)
        ylim_delta = (pymin - ppad_y, pymax + ppad_y)
    else:
        xlim_delta = None
        ylim_delta = None

    # --- figure: reserve space on the right for colorbars so subplots keep same size ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    # Reserve right margin for colorbars (no resizing of subplots by fig.colorbar)
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.06, top=0.92, wspace=0.18, hspace=0.22)

    # Titles
    axes[0, 0].set_title(f"{gene} @ 2.5m (spatial_axis)", fontsize=12)
    axes[0, 1].set_title(f"{gene} @ 5.7m (spatial_axis)", fontsize=12)
    axes[0, 2].set_title(f"{gene} @ 17.9m (spatial_axis)", fontsize=12)
    axes[1, 0].set_title(f"Domain {domain_id} in full tissue (5.7m spatial_axis)", fontsize=12)
    axes[1, 1].set_title("Δ(5.7-2.5) (spatial_axis_pixel)", fontsize=12)
    axes[1, 2].set_title("Δ(17.9-5.7) (spatial_axis_pixel)", fontsize=12)

    # Row 1 plotting helper
    def plot_expr(ax, coords, expr, label):
        if coords is None or coords.shape[0] == 0 or expr is None:
            ax.text(0.5, 0.5, f"{label}\n(missing/empty)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            return None
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=expr, s=2, linewidths=0,
                        cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
        if xlim_row1 is not None:
            ax.set_xlim(*xlim_row1)
        if ylim_row1 is not None:
            ax.set_ylim(*ylim_row1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        return sc

    sc0 = plot_expr(axes[0, 0], coords25, expr25, "2.5m")
    sc1 = plot_expr(axes[0, 1], coords57, expr57, "5.7m")
    sc2 = plot_expr(axes[0, 2], coords179, expr179, "17.9m")

    # Row 2 col 1: domain overlay full tissue (use 5.7m)
    coords_all = np.asarray(tp57["adata"].obsm[SPATIAL_AXIS_KEY])
    dom_all = tp57["dom"]
    mdom = dom_all == domain_id
    axes[1, 0].scatter(coords_all[:, 0], coords_all[:, 1], c="lightgray", s=1, linewidths=0, rasterized=True)
    if mdom.sum() > 0:
        axes[1, 0].scatter(coords_all[mdom, 0], coords_all[mdom, 1], c="red", s=2, linewidths=0, rasterized=True)
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([]); axes[1, 0].set_aspect("equal")

    # Row 2 delta helper
    def plot_delta(ax, ddf, label):
        if ddf is None or ddf.empty:
            ax.text(0.5, 0.5, f"{label}\n(empty)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            return None
        sc = ax.scatter(ddf["x"], ddf["y"], c=ddf["delta"], s=8, linewidths=0,
                        cmap="coolwarm", vmin=-dmax, vmax=dmax, rasterized=True)
        if xlim_delta is not None:
            ax.set_xlim(*xlim_delta)
        if ylim_delta is not None:
            ax.set_ylim(*ylim_delta)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        return sc

    scd1 = plot_delta(axes[1, 1], delta12, "Δ12")
    scd2 = plot_delta(axes[1, 2], delta23, "Δ23")

    # --- Colorbars in dedicated axes (no overlap, no subplot resizing) ---
    # Expression colorbar (top-right)
    ref_sc = sc0 if sc0 is not None else (sc1 if sc1 is not None else sc2)
    if ref_sc is not None:
        cax_expr = fig.add_axes([0.90, 0.55, 0.015, 0.32])  # [left, bottom, width, height]
        cb_expr = fig.colorbar(ref_sc, cax=cax_expr)
        cb_expr.ax.tick_params(labelsize=9)
        cb_expr.set_label("log1p(norm counts)", fontsize=10)

    # Delta colorbar (bottom-right)
    ref_sd = scd2 if scd2 is not None else scd1
    if ref_sd is not None:
        cax_del = fig.add_axes([0.90, 0.12, 0.015, 0.32])
        cb_del = fig.colorbar(ref_sd, cax=cax_del)
        cb_del.ax.tick_params(labelsize=9)
        cb_del.set_label("Δ log1p(norm counts)", fontsize=10)

    fig.suptitle(f"Domain {domain_id} | Gene {gene} | TgCRND8 (2.5m, 5.7m, 17.9m)", fontsize=14, y=0.98)

    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf, dpi=220)
    plt.close(fig)


def load_timepoint(h5ad_path: str, name: str, domain_key: str) -> dict:
    adata = ad.read_h5ad(h5ad_path)

    # sanity
    for k in [SPATIAL_AXIS_KEY, SPATIAL_PIXEL_KEY]:
        if k not in adata.obsm_keys():
            raise KeyError(f"{name}: missing obsm['{k}']")

    dom = pd.to_numeric(adata.obs[domain_key], errors="coerce").fillna(-1).astype(np.int32).values

    # size factors from counts
    counts = get_counts_matrix(adata)
    sf = compute_size_factors(counts)

    return {
        "name": name,
        "adata": adata,
        "dom": dom,
        "size_factors": sf,
    }


def main():
    print("Reading ranking CSVs...")
    df12 = read_rankings_csv(RES12)  # 5.7 vs 2.5
    df23 = read_rankings_csv(RES23)  # 17.9 vs 5.7

    top12 = pick_top_genes_per_domain(df12, topk=TOPK)
    top23 = pick_top_genes_per_domain(df23, topk=TOPK)

    # union genes across all domains
    genes_all = sorted(set(sum([top12[d] for d in DOMAINS], []) + sum([top23[d] for d in DOMAINS], [])))
    print(f"Total unique selected genes across domains: {len(genes_all)}")

    # infer domain key
    domain_key = DOMAIN_KEY
    print(f"Using domain key: adata.obs['{domain_key}']")

    # load 3 timepoints once
    print("Loading timepoints and computing size factors...")
    tp25 = load_timepoint(H5AD_25, "2.5m", domain_key)
    tp57 = load_timepoint(H5AD_57, "5.7m", domain_key)
    tp179 = load_timepoint(H5AD_179, "17.9m", domain_key)

    # aggregate pixel means (log-normalized) for delta plots
    print("Aggregating pixel mean log-normalized expression (2.5m)...")
    df25_pix = aggregate_pixels_log_norm(tp25["adata"], genes_all, domain_key, tp25["size_factors"])
    print("Aggregating pixel mean log-normalized expression (5.7m)...")
    df57_pix = aggregate_pixels_log_norm(tp57["adata"], genes_all, domain_key, tp57["size_factors"])
    print("Aggregating pixel mean log-normalized expression (17.9m)...")
    df179_pix = aggregate_pixels_log_norm(tp179["adata"], genes_all, domain_key, tp179["size_factors"])

    # save selected genes table
    sel_rows = []
    for d in DOMAINS:
        sel_rows.append({
            "domain": d,
            "top5_57vs25": ";".join(top12.get(d, [])),
            "top5_179vs57": ";".join(top23.get(d, [])),
        })
    sel_df = pd.DataFrame(sel_rows)
    sel_path = os.path.join(OUTDIR, "selected_top_genes_per_domain.csv")
    sel_df.to_csv(sel_path, index=False)
    print(f"Wrote: {sel_path}")

    # per-domain, per-gene plots
    print("Plotting per-domain, per-gene 2x3 figures...")
    for d in DOMAINS:
        genes_d = []
        for g in top12.get(d, []):
            if g not in genes_d:
                genes_d.append(g)
        for g in top23.get(d, []):
            if g not in genes_d:
                genes_d.append(g)

        if len(genes_d) == 0:
            continue

        dom_dir = os.path.join(OUTDIR, f"domain_{d:02d}")
        os.makedirs(dom_dir, exist_ok=True)

        for gene in genes_d:
            gene_safe = sanitize_filename(gene)
            out_png = os.path.join(dom_dir, f"domain_{d:02d}__{gene_safe}.png")
            out_pdf = os.path.join(dom_dir, f"domain_{d:02d}__{gene_safe}.pdf")

            try:
                plot_gene_domain_2x3(
                    domain_id=d,
                    gene=gene,
                    tp25=tp25,
                    tp57=tp57,
                    tp179=tp179,
                    df25_pix=df25_pix,
                    df57_pix=df57_pix,
                    df179_pix=df179_pix,
                    out_png=out_png,
                    out_pdf=out_pdf,
                )
            except Exception as e:
                warnings.warn(f"Failed domain={d} gene={gene}: {e}")

        print(f"  domain {d}: saved {len(genes_d)} genes -> {dom_dir}")

    print("Done.")


if __name__ == "__main__":
    main()