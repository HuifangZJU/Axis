#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Protein–gene differential visualization with globally-computed marker-correlation
heatmap + aligned protein–gene dotplot.

Main revision
-------------
The marker-correlation map is now computed from the FULL gene universe for the
displayed protein markers, rather than only the displayed top genes. This makes
the protein-module structure much less selection-biased and more defensible.

Compared with the previous version, this script:
1. Uses all genes for marker–marker correlation and clustering.
2. Still displays only the selected protein markers in the final figure.
3. Keeps the combined heatmap + dotplot strictly column-aligned in a single plot.
4. Restores gene-block separators and improves figure readability.
5. Adds optional protein-module separators based on the clustered correlation map.

Expected input RDS columns
--------------------------
marker, gene, logFC, AveExpr, adj.P.Val
"""

import os
import re
import argparse
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm
import seaborn as sns
import pyreadr

from scipy.cluster.hierarchy import linkage, leaves_list, fcluster, optimal_leaf_ordering
from scipy.spatial.distance import pdist


DEFAULT_PROTEIN_MARKERS = ["CD8", "CD20", "CD68", "a.SMA", "IDO1", "NKX2-1", "CXCL13", "E-cadherin"]


# ============================================================
# Utilities
# ============================================================

def canon(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(".", "").replace("-", "").replace("_", "").replace(" ", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _iqr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def read_rds_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    res = pyreadr.read_r(path)
    if len(res) == 0:
        raise ValueError(f"No objects found in RDS: {path}")
    obj = next(iter(res.values()))
    if not isinstance(obj, pd.DataFrame):
        raise ValueError(f"RDS does not contain a data.frame: {path}")
    return obj


def validate_columns(df: pd.DataFrame):
    required = ["logFC", "AveExpr", "adj.P.Val", "gene", "marker"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")


def to_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ["logFC", "AveExpr", "adj.P.Val"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["gene"] = d["gene"].astype(str)
    d["marker"] = d["marker"].astype(str)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["logFC", "AveExpr", "adj.P.Val", "gene", "marker"])
    d["abs_logFC"] = d["logFC"].abs()
    d["neglog10_fdr"] = -np.log10(d["adj.P.Val"].clip(lower=1e-300))
    return d


def setup_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# Marker ranking / gene selection
# ============================================================

def rank_markers_by_directional_variability(
    df: pd.DataFrame,
    fdr_thresh: float = 0.05,
    min_pairs: int = 20,
) -> pd.DataFrame:
    rows = []
    for marker, g in df.groupby("marker"):
        g = g.copy()
        g_sig = g[g["adj.P.Val"] <= fdr_thresh].copy()
        n_pairs = int(len(g))
        n_sig = int(len(g_sig))

        g_use = g_sig if n_sig >= min_pairs else g
        signed_iqr = _iqr(g_use["logFC"].to_numpy())
        signed_sd = float(np.std(g_use["logFC"].to_numpy()))
        mean_neglog10_fdr = float(np.mean(g_use["neglog10_fdr"]))
        mean_aveexpr = float(np.mean(g_use["AveExpr"]))

        frac_up_sig = float(np.mean(g_sig["logFC"] > 0)) if n_sig > 0 else 0.0
        frac_down_sig = float(np.mean(g_sig["logFC"] < 0)) if n_sig > 0 else 0.0
        sign_mix_sig = float(2 * min(frac_up_sig, frac_down_sig))

        score = (
            2.5 * signed_iqr
            + 1.2 * signed_sd
            + 1.5 * sign_mix_sig
            + 0.25 * np.log1p(n_sig)
            + 0.10 * mean_neglog10_fdr
            + 0.05 * mean_aveexpr
        )

        rows.append({
            "marker": marker,
            "score": score,
            "n_pairs": n_pairs,
            "n_sig": n_sig,
            "iqr_signed_logFC": signed_iqr,
            "sd_signed_logFC": signed_sd,
            "mean_neglog10_fdr": mean_neglog10_fdr,
            "mean_AveExpr": mean_aveexpr,
            "frac_up_sig": frac_up_sig,
            "frac_down_sig": frac_down_sig,
            "sign_mix_sig": sign_mix_sig,
        })

    return pd.DataFrame(rows).sort_values(
        ["iqr_signed_logFC", "sign_mix_sig", "n_sig", "mean_neglog10_fdr", "score"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def pick_topk_genes_per_marker(
    df: pd.DataFrame,
    markers: Sequence[str],
    topk: int = 10,
    fdr_thresh: float = 0.05,
    prefer_significant: bool = True,
) -> pd.DataFrame:
    d = df[df["marker"].isin(markers)].copy()
    out_parts = []

    for marker in markers:
        g = d[d["marker"] == marker].copy()
        if len(g) == 0:
            continue
        g_sig = g[g["adj.P.Val"] <= fdr_thresh].copy()
        g_use = g_sig if (prefer_significant and len(g_sig) > 0) else g
        g_use = g_use.sort_values(
            ["adj.P.Val", "abs_logFC", "AveExpr"],
            ascending=[True, False, False],
        )
        out_parts.append(g_use.head(topk))

    if not out_parts:
        return d.head(0).copy()
    return pd.concat(out_parts, ignore_index=True)


def build_gene_profile_matrix(
    df: pd.DataFrame,
    genes: Sequence[str],
    markers: Sequence[str],
    value_col: str = "logFC",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    mat = (
        df[df["gene"].isin(genes) & df["marker"].isin(markers)]
        .pivot_table(index="gene", columns="marker", values=value_col, aggfunc="first")
        .reindex(index=list(genes), columns=list(markers))
    )
    return mat.fillna(fill_value)


def build_global_marker_program_matrix(
    df: pd.DataFrame,
    markers: Sequence[str],
    value_col: str = "logFC",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Build gene x marker matrix using the FULL gene universe for the selected markers.
    This is the key revision: correlation is computed from the whole available program,
    not from the displayed top genes only.
    """
    markers = list(markers)
    mat = (
        df[df["marker"].isin(markers)]
        .pivot_table(index="gene", columns="marker", values=value_col, aggfunc="first")
        .reindex(columns=markers)
    )
    mat = mat.fillna(fill_value)
    # drop fully-constant genes to stabilize correlation-distance clustering
    keep = np.nanstd(mat.to_numpy(dtype=float), axis=1) > 0
    if np.any(keep):
        mat = mat.loc[keep]
    return mat


def cluster_markers_from_global_program(
    df_for_corr: pd.DataFrame,
    markers: Sequence[str],
    rank_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[str], pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Cluster selected markers from the FULL marker program matrix.
    Returns:
        ordered markers,
        ordered marker-marker correlation matrix,
        linkage matrix Z,
        ordered module labels
    """
    markers = list(markers)
    mat = build_global_marker_program_matrix(df_for_corr, markers, value_col="logFC", fill_value=0.0)

    if len(markers) <= 1:
        corr = pd.DataFrame([[1.0]], index=markers, columns=markers)
        return markers, corr, None, np.array([1], dtype=int)

    X = mat.loc[:, markers].to_numpy(dtype=float).T  # markers x genes

    try:
        dist = pdist(X, metric="correlation")
        if not np.any(np.isfinite(dist)) or np.allclose(dist, 0):
            raise ValueError("degenerate distance")
        Z = linkage(dist, method="average")
        try:
            Z = optimal_leaf_ordering(Z, dist)
        except Exception:
            pass
        leaves = leaves_list(Z)
        ordered = [markers[i] for i in leaves]
    except Exception:
        Z = None
        # fallback: order by ranking table if available
        if rank_df is not None:
            ordered = [m for m in rank_df["marker"].tolist() if m in markers]
        else:
            ordered = markers

    corr = np.corrcoef(X)
    corr = pd.DataFrame(corr, index=markers, columns=markers)
    corr = corr.loc[ordered, ordered]

    if Z is not None and len(markers) >= 4:
        maxclust = min(4, len(markers))
        raw_labels = fcluster(Z, t=maxclust, criterion="maxclust")
        ordered_labels = np.array([raw_labels[i] for i in leaves], dtype=int)
    else:
        ordered_labels = np.ones(len(ordered), dtype=int)

    return ordered, corr, Z, ordered_labels


def best_marker_for_gene(df: pd.DataFrame, genes: Sequence[str], markers: Sequence[str]) -> pd.DataFrame:
    d = df[df["gene"].isin(genes) & df["marker"].isin(markers)].copy()
    d = d.sort_values(["gene", "adj.P.Val", "abs_logFC", "AveExpr"], ascending=[True, True, False, False])
    idx = d.groupby("gene")["adj.P.Val"].idxmin()
    return d.loc[idx, ["gene", "marker", "adj.P.Val", "abs_logFC", "AveExpr"]].copy()


def cluster_gene_order_within_best_marker_blocks(
    df: pd.DataFrame,
    genes: Sequence[str],
    marker_order: Sequence[str],
    value_col: str = "logFC",
    standardize_rows: bool = True,
) -> Tuple[List[str], List[Tuple[str, int, int]]]:
    genes = list(dict.fromkeys(genes))
    marker_order = list(marker_order)

    mat = build_gene_profile_matrix(df, genes, marker_order, value_col=value_col, fill_value=0.0)
    best = best_marker_for_gene(df, genes, marker_order)
    best_map = dict(zip(best["gene"], best["marker"]))

    final_order: List[str] = []
    block_ranges: List[Tuple[str, int, int]] = []
    start = 0

    for marker in marker_order:
        block_genes = [g for g in genes if best_map.get(g, None) == marker]
        if len(block_genes) == 0:
            continue

        sub = mat.loc[block_genes].to_numpy(dtype=float)
        if standardize_rows and sub.shape[1] > 1:
            mu = sub.mean(axis=1, keepdims=True)
            sd = sub.std(axis=1, keepdims=True)
            sd[sd == 0] = 1.0
            sub = (sub - mu) / sd

        if len(block_genes) <= 2:
            ordered_block = block_genes
        else:
            try:
                dist = pdist(sub, metric="correlation")
                if np.any(np.isfinite(dist)) and not np.allclose(dist, 0):
                    Z = linkage(dist, method="average")
                    try:
                        Z = optimal_leaf_ordering(Z, dist)
                    except Exception:
                        pass
                    leaves = leaves_list(Z)
                    ordered_block = [block_genes[i] for i in leaves]
                else:
                    ordered_block = block_genes
            except Exception:
                ordered_block = block_genes

        final_order.extend(ordered_block)
        end = start + len(ordered_block)
        block_ranges.append((marker, start, end))
        start = end

    leftover = [g for g in genes if g not in final_order]
    if leftover:
        final_order.extend(leftover)
        block_ranges.append(("leftover", start, start + len(leftover)))

    return final_order, block_ranges


# ============================================================
# Plot helpers
# ============================================================

def map_sizes(values, smin=24, smax=220):
    v = np.asarray(values, dtype=float)
    lo = np.nanpercentile(v, 5)
    hi = np.nanpercentile(v, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(v)
        hi = np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(v, (smin + smax) / 2.0)
    vv = np.clip(v, lo, hi)
    return smin + (vv - lo) * (smax - smin) / (hi - lo)


def add_gene_block_separators(ax, block_ranges: Sequence[Tuple[str, int, int]], n_markers: int, n_genes: int):
    for marker, start, end in block_ranges:
        if start > 0:
            yline = n_genes - start - 0.5
            ax.axhline(y=yline, color="0.70", lw=0.8, alpha=0.9)
        if marker != "leftover":
            y_center = n_genes - (start + end - 1) / 2.0 - 1.0
            ax.text(
                n_markers + 0.18,
                y_center,
                marker,
                va="center",
                ha="left",
                fontsize=8,
                color="0.35",
            )


def module_boundaries(module_labels: np.ndarray) -> List[int]:
    bounds = []
    if len(module_labels) <= 1:
        return bounds
    for i in range(1, len(module_labels)):
        if module_labels[i] != module_labels[i - 1]:
            bounds.append(i)
    return bounds


def plot_marker_correlation_heatmap(corr: pd.DataFrame, out_png: str, title: str):
    n = len(corr)
    W = max(5.8, 0.42 * n + 2.8)
    H = max(5.4, 0.42 * n + 2.6)

    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.28,
        linecolor="white",
        annot=True if n <= 12 else False,
        fmt=".2f",
        cbar_kws={"label": "Marker–marker correlation", "shrink": 0.90},
        ax=ax,
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def dotplot_on_axis(
    ax,
    df_plot: pd.DataFrame,
    marker_order: Sequence[str],
    gene_order: Sequence[str],
    block_ranges: Sequence[Tuple[str, int, int]],
    title: str,
    cmap: str = "coolwarm",
    vmax_percentile: float = 95,
    sig_outline_fdr: float = 0.05,
):
    marker_order = list(marker_order)
    gene_order = list(gene_order)

    x_map = {m: i for i, m in enumerate(marker_order)}
    y_display_order = gene_order[::-1]
    y_map = {g: i for i, g in enumerate(y_display_order)}

    d = df_plot.copy()
    d = d[d["marker"].isin(marker_order) & d["gene"].isin(gene_order)].copy()
    d["x"] = d["marker"].map(x_map)
    d["y"] = d["gene"].map(y_map)
    d = d.dropna(subset=["x", "y", "logFC", "AveExpr", "adj.P.Val"]).copy()

    x = d["x"].to_numpy(dtype=int)
    y = d["y"].to_numpy(dtype=int)
    logfc = d["logFC"].to_numpy(dtype=float)
    ave = d["AveExpr"].to_numpy(dtype=float)
    sig = d["adj.P.Val"].to_numpy(dtype=float) <= sig_outline_fdr

    vmax = np.nanpercentile(np.abs(logfc), vmax_percentile)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.nanmax(np.abs(logfc))) if np.isfinite(np.nanmax(np.abs(logfc))) else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    sizes = map_sizes(ave, smin=26, smax=250)

    m0 = ~sig
    ax.scatter(
        x[m0], y[m0],
        c=np.clip(logfc[m0], -vmax, vmax),
        s=sizes[m0],
        cmap=cmap,
        norm=norm,
        linewidths=0.0,
        alpha=0.92,
    )
    sc = ax.scatter(
        x[sig], y[sig],
        c=np.clip(logfc[sig], -vmax, vmax),
        s=sizes[sig],
        cmap=cmap,
        norm=norm,
        linewidths=0.55,
        edgecolors="black",
        alpha=0.98,
    )

    ax.set_xticks(np.arange(len(marker_order)))
    ax.set_xticklabels(marker_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(gene_order)))
    ax.set_yticklabels(y_display_order, fontsize=8)
    ax.set_xlim(-0.5, len(marker_order) - 0.5)
    ax.set_ylim(-0.5, len(gene_order) - 0.5)
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="both", linestyle=":", alpha=0.16)
    ax.set_xlabel("Protein markers")
    ax.set_ylabel("Genes")
    ax.set_title(title, pad=8)
    add_gene_block_separators(ax, block_ranges, len(marker_order), len(gene_order))
    return sc

def plot_dotplot_only(
    df_plot: pd.DataFrame,
    marker_order: Sequence[str],
    gene_order: Sequence[str],
    block_ranges: Sequence[Tuple[str, int, int]],
    out_png: str,
    title: str,
    sig_outline_fdr: float = 0.05,
):
    W = max(8.5, 0.34 * len(marker_order) + 4.0)
    H = max(7.2, 0.16 * len(gene_order) + 3.1)

    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    sc = dotplot_on_axis(
        ax=ax,
        df_plot=df_plot,
        marker_order=marker_order,
        gene_order=gene_order,
        block_ranges=block_ranges,
        title=title,
        sig_outline_fdr=sig_outline_fdr,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.024, pad=0.02)
    cbar.set_label("logFC")

    ref = np.nanpercentile(df_plot["AveExpr"], [20, 50, 80])
    ref_sizes = map_sizes(ref, smin=24, smax=220)
    handles = [
        ax.scatter([], [], s=ref_sizes[i], c="gray", alpha=0.7, label=f"AveExpr≈{ref[i]:.2g}")
        for i in range(len(ref))
    ]
    handles.append(
        ax.scatter([], [], s=78, facecolors="white", edgecolors="black", linewidths=0.7,
                   label=f"adj.P.Val≤{sig_outline_fdr:g}")
    )
    ax.legend(
        handles=handles,
        title="Dot encoding",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_combined_corr_dotplot_single_axis(
    corr: pd.DataFrame,
    df_plot: pd.DataFrame,
    marker_order: Sequence[str],
    gene_order: Sequence[str],
    block_ranges: Sequence[Tuple[str, int, int]],
    module_labels: np.ndarray,
    out_png: str,
    title: str,
    sig_outline_fdr: float = 0.05,
    gap_rows: float = 1.8,
):
    """
    Combined figure in one main axis:
    - top: marker-correlation heatmap
    - bottom: protein-gene dotplot
    Columns are exactly aligned.
    """
    marker_order = list(marker_order)
    gene_order = list(gene_order)
    n_markers = len(marker_order)
    n_genes = len(gene_order)

    x_map = {m: i for i, m in enumerate(marker_order)}
    y_display_order = gene_order[::-1]
    y_map = {g: i for i, g in enumerate(y_display_order)}

    d = df_plot.copy()
    d = d[d["marker"].isin(marker_order) & d["gene"].isin(gene_order)].copy()
    d["x"] = d["marker"].map(x_map)
    d["y"] = d["gene"].map(y_map)
    d = d.dropna(subset=["x", "y", "logFC", "AveExpr", "adj.P.Val"]).copy()

    x = d["x"].to_numpy(dtype=int)
    y = d["y"].to_numpy(dtype=float)
    logfc = d["logFC"].to_numpy(dtype=float)
    ave = d["AveExpr"].to_numpy(dtype=float)
    sig = d["adj.P.Val"].to_numpy(dtype=float) <= sig_outline_fdr

    vmax = np.nanpercentile(np.abs(logfc), 97)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = max(float(np.nanmax(np.abs(logfc))), 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    sizes = map_sizes(ave, smin=24, smax=220)

    corr_use = corr.loc[marker_order, marker_order].copy()
    corr_plot = corr_use.iloc[::-1].to_numpy(dtype=float)

    y_heat_start = n_genes + gap_rows
    y_heat_end = y_heat_start + n_markers - 1

    W = max(3.8, 0.55 * n_markers )
    H = max(8.8, 0.17 * (n_genes + n_markers) + 2.6)

    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0.08, 0.08, 0.66, 0.84])

    # Heatmap
    im_corr = ax.imshow(
        corr_plot,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        origin="lower",
        extent=(-0.5, n_markers - 0.5, y_heat_start - 0.5, y_heat_end + 0.5),
        interpolation="nearest",
        aspect="auto",
        zorder=1,
    )

    # heatmap grid
    for i in range(n_markers + 1):
        ax.plot([i - 0.5, i - 0.5], [y_heat_start - 0.5, y_heat_end + 0.5],
                color="white", lw=0.45, alpha=0.85, zorder=2)
        ax.plot([-0.5, n_markers - 0.5], [y_heat_start - 0.5 + i, y_heat_start - 0.5 + i],
                color="white", lw=0.45, alpha=0.85, zorder=2)

    # module separators across both panels
    bounds = module_boundaries(module_labels)
    for b in bounds:
        xb = b - 0.5
        ax.axvline(xb, color="0.25", lw=1.3, alpha=0.95, zorder=5)

    # faint grid only for dotplot region
    for xi in range(n_markers + 1):
        ax.plot([xi - 0.5, xi - 0.5], [-0.5, n_genes - 0.5], color="0.92", lw=0.55, zorder=0)
    for yi in range(n_genes + 1):
        ax.plot([-0.5, n_markers - 0.5], [yi - 0.5, yi - 0.5], color="0.94", lw=0.40, zorder=0)

    # Dotplot
    m0 = ~sig
    ax.scatter(
        x[m0], y[m0],
        c=np.clip(logfc[m0], -vmax, vmax),
        s=sizes[m0],
        cmap="coolwarm",
        norm=norm,
        linewidths=0.0,
        alpha=0.93,
        zorder=3,
    )
    sc = ax.scatter(
        x[sig], y[sig],
        c=np.clip(logfc[sig], -vmax, vmax),
        s=sizes[sig],
        cmap="coolwarm",
        norm=norm,
        linewidths=0.55,
        edgecolors="black",
        alpha=0.98,
        zorder=4,
    )

    # gene block separators
    add_gene_block_separators(ax, block_ranges, n_markers, n_genes)

    # panel separator line
    sep_y = n_genes + gap_rows / 2.0 - 0.5
    ax.axhline(sep_y, color="0.35", lw=1.0, alpha=0.9)

    # colored module strip just above heatmap
    unique_mods = list(dict.fromkeys(module_labels.tolist()))
    palette = sns.color_palette("Set2", n_colors=max(len(unique_mods), 3))
    mod_to_color = {m: palette[i % len(palette)] for i, m in enumerate(unique_mods)}
    strip_y0 = y_heat_end + 0.18
    strip_y1 = y_heat_end + 0.60
    for i, lab in enumerate(module_labels):
        ax.add_patch(
            plt.Rectangle((i - 0.5, strip_y0), 1.0, strip_y1 - strip_y0,
                          facecolor=mod_to_color[lab], edgecolor="none", alpha=0.95, zorder=6)
        )

    # text panel labels
    ax.text((n_markers - 1) / 2.0, y_heat_end + 1.1,
            "Marker–marker correlation (computed from full gene universe)",
            ha="center", va="bottom", fontsize=12)
    ax.text((n_markers - 1) / 2.0, n_genes + 0.15,
            "Protein–gene differential dotplot",
            ha="center", va="bottom", fontsize=12)

    # axes ticks
    ax.set_xticks(np.arange(n_markers))
    ax.set_xticklabels(marker_order, rotation=45, ha="right", fontsize=10)

    heatmap_row_labels_bottom_to_top = list(corr_use.index[::-1])
    y_ticks_bottom = np.arange(n_genes)
    y_ticks_top = np.arange(n_markers) + y_heat_start
    ax.set_yticks(list(y_ticks_bottom) + list(y_ticks_top))
    ax.set_yticklabels(y_display_order + heatmap_row_labels_bottom_to_top, fontsize=8)

    # color protein labels by module
    for tick, lab in zip(ax.get_xticklabels(), module_labels):
        tick.set_color(mod_to_color[lab])
        tick.set_fontweight("bold")

    ax.set_xlim(-0.5, n_markers - 0.5)
    ax.set_ylim(-0.5, y_heat_end + 1.4)
    ax.set_xlabel("Protein markers")
    ax.set_ylabel("")
    ax.set_title(title, pad=16, fontsize=15)

    # colorbars
    cax1 = fig.add_axes([0.77, 0.56, 0.022, 0.33])
    cb1 = fig.colorbar(im_corr, cax=cax1)
    cb1.set_label("Marker–marker correlation", fontsize=10)

    cax2 = fig.add_axes([0.77, 0.18, 0.022, 0.22])
    cb2 = fig.colorbar(sc, cax=cax2)
    cb2.set_label("logFC", fontsize=10)

    # dot legend
    ref = np.nanpercentile(df_plot["AveExpr"], [20, 50, 80])
    ref_sizes = map_sizes(ref, smin=24, smax=220)
    handles = [
        ax.scatter([], [], s=ref_sizes[i], c="gray", alpha=0.7, label=f"AveExpr≈{ref[i]:.2g}")
        for i in range(len(ref))
    ]
    handles.append(
        ax.scatter([], [], s=78, facecolors="white", edgecolors="black", linewidths=0.7,
                   label=f"adj.P.Val≤{sig_outline_fdr:g}")
    )
    ax.legend(
        handles=handles,
        title="Dot encoding",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.52),
        frameon=False,
    )

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Analysis runner
# ============================================================

def run_panel_analysis(
    df: pd.DataFrame,
    markers_subset: Optional[Sequence[str]],
    preserve_user_order: bool,
    topk: int,
    fdr_thresh: float,
    out_prefix: str,
    out_dir: str,
    title_prefix: str,
):
    rank_df = rank_markers_by_directional_variability(df, fdr_thresh=fdr_thresh)

    if markers_subset is None:
        marker_seed = rank_df["marker"].tolist()
        df_corr = df.copy()
    else:
        subset_canon = {canon(x): x for x in markers_subset}
        rank_df = rank_df.copy()
        rank_df["marker_c"] = rank_df["marker"].map(canon)
        rank_df = rank_df[rank_df["marker_c"].isin(subset_canon.keys())].copy()

        if preserve_user_order:
            order_map = {canon(m): i for i, m in enumerate(markers_subset)}
            rank_df["user_rank"] = rank_df["marker_c"].map(order_map)
            rank_df = rank_df.sort_values(["user_rank", "score"], ascending=[True, False]).reset_index(drop=True)
        else:
            rank_df = rank_df.reset_index(drop=True)

        marker_seed = rank_df["marker"].tolist()
        rank_df = rank_df.drop(columns=[c for c in ["marker_c", "user_rank"] if c in rank_df.columns])
        df_corr = df[df["marker"].map(canon).isin(subset_canon.keys())].copy()

    if len(marker_seed) == 0:
        return None

    # representative genes for display only
    top = pick_topk_genes_per_marker(
        df=df if markers_subset is None else df_corr,
        markers=marker_seed,
        topk=topk,
        fdr_thresh=fdr_thresh,
        prefer_significant=True,
    )
    genes_seed = sorted(pd.unique(top["gene"]))
    df_plot_base = (df if markers_subset is None else df_corr)
    df_seed = df_plot_base[df_plot_base["gene"].isin(genes_seed) & df_plot_base["marker"].isin(marker_seed)].copy()

    # FINAL displayed marker order from FULL gene universe, not top genes only
    marker_order, corr, Z, ordered_module_labels = cluster_markers_from_global_program(
        df_for_corr=df_corr,
        markers=marker_seed,
        rank_df=rank_df,
    )

    gene_order, block_ranges = cluster_gene_order_within_best_marker_blocks(
        df=df_seed,
        genes=genes_seed,
        marker_order=marker_order,
        value_col="logFC",
        standardize_rows=True,
    )

    df_plot = df_seed[df_seed["gene"].isin(gene_order) & df_seed["marker"].isin(marker_order)].copy()

    rank_df.to_csv(os.path.join(out_dir, f"marker_ranking_{out_prefix}.csv"), index=False)
    top.sort_values(["marker", "adj.P.Val", "abs_logFC"], ascending=[True, True, False]).to_csv(
        os.path.join(out_dir, f"selected_pairs_{out_prefix}.csv"), index=False
    )
    pd.DataFrame({"marker": marker_order, "module": ordered_module_labels}).to_csv(
        os.path.join(out_dir, f"marker_order_{out_prefix}.csv"), index=False
    )
    pd.DataFrame({"gene": gene_order}).to_csv(
        os.path.join(out_dir, f"gene_order_{out_prefix}.csv"), index=False
    )

    plot_marker_correlation_heatmap(
        corr=corr,
        out_png=os.path.join(out_dir, f"marker_corr_{out_prefix}.png"),
        title=f"{title_prefix}: marker–marker correlation",
    )

    plot_dotplot_only(
        df_plot=df_plot,
        marker_order=marker_order,
        gene_order=gene_order,
        block_ranges=block_ranges,
        out_png=os.path.join(out_dir, f"dotplot_{out_prefix}.png"),
        title=f"{title_prefix}: protein–gene differential dotplot",
        sig_outline_fdr=fdr_thresh,
    )

    plot_combined_corr_dotplot_single_axis(
        corr=corr,
        df_plot=df_plot,
        marker_order=marker_order,
        gene_order=gene_order,
        block_ranges=block_ranges,
        module_labels=ordered_module_labels,
        out_png=os.path.join(out_dir, f"marker_corr_dotplot_{out_prefix}.png"),
        title=f"{title_prefix}: marker modules and protein–gene differential map",
        sig_outline_fdr=fdr_thresh,
        gap_rows=1.8,
    )

    return {
        "rank_df": rank_df,
        "marker_order": marker_order,
        "gene_order": gene_order,
        "corr": corr,
        "df_plot": df_plot,
        "block_ranges": block_ranges,
    }


# ============================================================
# Main
# ============================================================

def main():
    setup_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--root_folder", default="/media/huifang/data/registration/phenocycler/haotian/")
    ap.add_argument("--in_rds", default="xenium_DE.rds")
    ap.add_argument("--out_dir", default="de_dotplot_matrix_v6")
    ap.add_argument("--topk_all", type=int, default=10)
    ap.add_argument("--topk_selected", type=int, default=10)
    ap.add_argument("--fdr_thresh", type=float, default=0.05)
    ap.add_argument("--selected_markers", nargs="*", default=DEFAULT_PROTEIN_MARKERS)
    ap.add_argument("--preserve_selected_order", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.root_folder, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    in_path = os.path.join(args.root_folder, args.in_rds)
    print("[LOAD]", in_path)

    df = read_rds_df(in_path)
    validate_columns(df)
    df = to_numeric_cols(df)

    run_panel_analysis(
        df=df,
        markers_subset=None,
        preserve_user_order=False,
        topk=args.topk_all,
        fdr_thresh=args.fdr_thresh,
        out_prefix="all_markers",
        out_dir=out_dir,
        title_prefix="All markers",
    )

    sel_markers = args.selected_markers if args.selected_markers is not None else DEFAULT_PROTEIN_MARKERS
    run_panel_analysis(
        df=df,
        markers_subset=sel_markers,
        preserve_user_order=args.preserve_selected_order,
        topk=args.topk_selected,
        fdr_thresh=args.fdr_thresh,
        out_prefix="selected_markers",
        out_dir=out_dir,
        title_prefix="Selected markers",
    )

    print("[DONE] Results saved to:", out_dir)
    print("Marker-correlation is now computed from the full gene universe for the displayed proteins.")
    print("The displayed dotplot still uses the selected top genes only.")
    print("The combined figure remains strictly column-aligned.")


if __name__ == "__main__":
    main()
