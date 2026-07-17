#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Protein–gene differential visualization with ONE-AXIS aligned
marker-correlation heatmap + protein–gene dotplot.

Key revision
------------
The combined figure is now drawn in a single main plotting axis so that:
- the correlation heatmap columns
- the dotplot columns

are strictly one-to-one aligned.

This avoids the visual drift that can happen with stacked subplots, even when
they nominally share the same marker order.

Design logic
------------
1. Representative genes are selected for the displayed marker set.
2. Marker order is computed exactly from the marker-correlation heatmap logic:
   gene x marker logFC matrix -> marker correlation -> hierarchical clustering.
3. The SAME marker order is then reused directly in the dotplot.
4. Genes are grouped by their best-associated marker, then clustered within
   each marker block.

This produces a genomics-style joint figure:
- top: protein modules
- bottom: genes driving those modules
"""

import os
import re
import argparse
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import seaborn as sns
import pyreadr

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist


# DEFAULT_PROTEIN_MARKERS = ["CD4", "CD8", "CD20", "CD68", "Ki67", "a.SMA", "IDO1", "CD44", "PD.1","NKX2-1","CXCL13","HLA-DPB1","E-cadherin"]
DEFAULT_PROTEIN_MARKERS = [ "CD8","CD20",  "CD68",  "a.SMA", "IDO1", "NKX2-1","CXCL13","E-cadherin"]

# ============================================================
# Basic utilities
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
    d["direction"] = np.where(d["logFC"] >= 0, "up", "down")
    return d


# ============================================================
# Marker ranking / selection
# ============================================================

def rank_markers_by_directional_variability(
    df: pd.DataFrame,
    fdr_thresh: float = 0.05,
    min_pairs: int = 20,
) -> pd.DataFrame:
    """
    Exploratory ranking table only.
    Final displayed order comes from marker-correlation clustering.
    """
    rows = []
    for marker, g in df.groupby("marker"):
        g = g.copy()
        g_sig = g[g["adj.P.Val"] <= fdr_thresh].copy()
        n_pairs = int(len(g))
        n_sig = int(len(g_sig))
        frac_sig = float(n_sig / max(n_pairs, 1))

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
            "frac_sig": frac_sig,
            "iqr_signed_logFC": signed_iqr,
            "sd_signed_logFC": signed_sd,
            "median_abs_logFC": float(np.median(g_use["abs_logFC"])),
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
    topk: int = 5,
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


def marker_order_from_v2_heatmap_logic(
    df_plot: pd.DataFrame,
    marker_order_seed: Sequence[str],
    gene_order_seed: Sequence[str],
) -> Tuple[List[str], pd.DataFrame]:
    """
    Exact ordering logic from the earlier marker-correlation heatmap:
    - build gene x marker logFC matrix using the selected plotted genes
    - compute marker–marker correlation
    - cluster the correlation matrix using euclidean distance + average linkage
    """
    marker_order_seed = list(marker_order_seed)
    gene_order_seed = list(gene_order_seed)

    mat = build_gene_profile_matrix(
        df_plot,
        gene_order_seed,
        marker_order_seed,
        value_col="logFC",
        fill_value=0.0,
    ).loc[gene_order_seed, marker_order_seed]

    corr = mat.corr(method="pearson").loc[marker_order_seed, marker_order_seed]

    ordered = marker_order_seed.copy()
    if len(marker_order_seed) > 2:
        try:
            dist = pdist(corr.to_numpy(dtype=float), metric="euclidean")
            if np.any(np.isfinite(dist)) and not np.allclose(dist, 0):
                Z = linkage(dist, method="average")
                leaves = leaves_list(Z)
                ordered = [marker_order_seed[i] for i in leaves]
                corr = corr.loc[ordered, ordered]
        except Exception:
            pass

    return ordered, corr


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

def map_sizes(values, smin=26, smax=240):
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
            ax.axhline(y=yline, color="0.72", lw=0.8, alpha=0.8)
        if marker != "leftover":
            y_center = n_genes - (start + end - 1) / 2.0 - 1.0
            ax.text(
                n_markers + 0.12,
                y_center,
                marker,
                va="center",
                ha="left",
                fontsize=8,
                color="0.35",
            )


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


def plot_marker_correlation_heatmap(
    corr: pd.DataFrame,
    out_png: str,
    out_pdf: str,
    title: str,
):
    n = len(corr)
    W = max(5.5, 0.38 * n + 3)
    H = max(5.5, 0.38 * n + 3)

    fig, ax = plt.subplots(figsize=(W, H), dpi=250)
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.25,
        linecolor="white",
        annot=True if n <= 18 else False,
        fmt=".2f",
        cbar_kws={"label": "Marker–marker correlation"},
        ax=ax,
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_dotplot_only(
    df_plot: pd.DataFrame,
    marker_order: Sequence[str],
    gene_order: Sequence[str],
    block_ranges: Sequence[Tuple[str, int, int]],
    out_png: str,
    out_pdf: str,
    title: str,
    sig_outline_fdr: float = 0.05,
):
    W = max(8, 0.30 * len(marker_order) + 4)
    H = max(7, 0.14 * len(gene_order) + 3)

    fig, ax = plt.subplots(figsize=(W, H), dpi=250)
    sc = dotplot_on_axis(
        ax=ax,
        df_plot=df_plot,
        marker_order=marker_order,
        gene_order=gene_order,
        block_ranges=block_ranges,
        title=title,
        sig_outline_fdr=sig_outline_fdr,
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("logFC")

    ref = np.nanpercentile(df_plot["AveExpr"], [20, 50, 80])
    ref_sizes = map_sizes(ref, smin=26, smax=250)
    handles = [
        ax.scatter([], [], s=ref_sizes[i], c="gray", alpha=0.7, label=f"AveExpr≈{ref[i]:.2g}")
        for i in range(len(ref))
    ]
    handles.append(
        ax.scatter([], [], s=80, facecolors="white", edgecolors="black", linewidths=0.7,
                   label=f"adj.P.Val≤{sig_outline_fdr:g}")
    )
    ax.legend(
        handles=handles,
        title="Dot encoding",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )

    plt.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_combined_corr_dotplot_single_axis(
    corr: pd.DataFrame,
    df_plot: pd.DataFrame,
    marker_order: Sequence[str],
    gene_order: Sequence[str],
    block_ranges: Sequence[Tuple[str, int, int]],
    out_png: str,
    out_pdf: str,
    title: str,
    sig_outline_fdr: float = 0.05,
    gap_rows: float = 2.0,
):
    """
    Draw the correlation heatmap and the dotplot in ONE main axis.

    Layout:
    - bottom block: gene dotplot
    - small gap
    - top block: marker-marker correlation heatmap

    Because both are drawn in the same x-coordinate system, the columns align exactly.
    """
    marker_order = list(marker_order)
    gene_order = list(gene_order)
    n_markers = len(marker_order)
    n_genes = len(gene_order)

    # ---------- prepare dotplot data ----------
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

    vmax = np.nanpercentile(np.abs(logfc), 95)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.nanmax(np.abs(logfc))) if np.isfinite(np.nanmax(np.abs(logfc))) else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    sizes = map_sizes(ave, smin=26, smax=250)

    # ---------- prepare correlation heatmap ----------
    corr_use = corr.loc[marker_order, marker_order].copy()
    # display first row at top while keeping origin='lower'
    corr_plot = corr_use.iloc[::-1].to_numpy(dtype=float)

    y_dot_start = 0.0
    y_dot_end = n_genes - 1

    y_heat_start = n_genes + gap_rows
    y_heat_end = y_heat_start + n_markers - 1

    # ---------- figure geometry ----------
    W = max(4.0, 0.34 * n_markers )
    H = max(9.0, 0.18 * (n_genes + n_markers) + 3.4)

    fig = plt.figure(figsize=(W, H), dpi=250)
    ax = fig.add_axes([0.08, 0.09, 0.72, 0.82])  # main single axis

    # top heatmap in the same axis
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

    # add heatmap cell borders for readability
    for i in range(n_markers + 1):
        ax.plot([i - 0.5, i - 0.5], [y_heat_start - 0.5, y_heat_end + 0.5],
                color="white", lw=0.4, alpha=0.7, zorder=2)
        ax.plot([-0.5, n_markers - 0.5], [y_heat_start - 0.5 + i, y_heat_start - 0.5 + i],
                color="white", lw=0.4, alpha=0.7, zorder=2)

    # dotplot in the same axis
    m0 = ~sig
    ax.scatter(
        x[m0], y[m0] + y_dot_start,
        c=np.clip(logfc[m0], -vmax, vmax),
        s=sizes[m0],
        cmap="coolwarm",
        norm=norm,
        linewidths=0.0,
        alpha=0.92,
        zorder=3,
    )
    sc = ax.scatter(
        x[sig], y[sig] + y_dot_start,
        c=np.clip(logfc[sig], -vmax, vmax),
        s=sizes[sig],
        cmap="coolwarm",
        norm=norm,
        linewidths=0.55,
        edgecolors="black",
        alpha=0.98,
        zorder=4,
    )

    # gap separator
    # sep_y = n_genes + gap_rows / 2.0 - 0.5
    # ax.axhline(sep_y, color="0.55", lw=1.0, alpha=0.9)

    # # gene block separators in bottom section only
    # for marker, start, end in block_ranges:
    #     if start > 0:
    #         yline = n_genes - start - 0.5
    #         ax.axhline(y=yline, color="0.72", lw=0.8, alpha=0.8, zorder=0)
    #     if marker != "leftover":
    #         y_center = n_genes - (start + end - 1) / 2.0 - 1.0
    #         ax.text(
    #             n_markers + 0.15,
    #             y_center,
    #             marker,
    #             va="center",
    #             ha="left",
    #             fontsize=8,
    #             color="0.35",
    #         )

    # x ticks strictly shared
    ax.set_xticks(np.arange(n_markers))
    ax.set_xticklabels(marker_order, rotation=45, ha="right", fontsize=9)

    # y ticks: genes + heatmap rows
    heatmap_row_labels_bottom_to_top = list(corr_use.index[::-1])
    y_ticks_bottom = np.arange(n_genes)
    y_ticks_top = np.arange(n_markers) + y_heat_start
    ax.set_yticks(list(y_ticks_bottom) + list(y_ticks_top))
    ax.set_yticklabels(y_display_order + heatmap_row_labels_bottom_to_top, fontsize=8)

    ax.set_xlim(-0.5, n_markers - 0.5)
    ax.set_ylim(-0.5, y_heat_end + 0.5)
    #
    # # background / grid only for dotplot region
    # for xi in range(n_markers):
    #     ax.axvline(xi - 0.5, ymin=0, ymax=(n_genes) / (y_heat_end + 1), color="0.92", lw=0.6, zorder=0)
    # for yi in range(n_genes):
    #     ax.axhline(yi - 0.5, xmin=0, xmax=1, color="0.94", lw=0.4, zorder=0)

    # panel labels
    ax.text(
        n_markers / 2.0 - 0.5,
        y_heat_end + 1.0,
        "Marker–marker correlation",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    ax.text(
        n_markers / 2.0 - 0.5,
        n_genes + 0.2,
        "Protein–gene differential dotplot",
        ha="center",
        va="bottom",
        fontsize=12,
    )

    ax.set_xlabel("Protein markers")
    ax.set_ylabel("")
    ax.set_title(title, pad=14)

    # colorbars
    cax1 = fig.add_axes([0.82, 0.52, 0.018, 0.39])
    cb1 = fig.colorbar(im_corr, cax=cax1)
    cb1.set_label("Marker–marker correlation", fontsize=10)

    cax2 = fig.add_axes([0.82, 0.20, 0.018, 0.22])
    cb2 = fig.colorbar(sc, cax=cax2)
    cb2.set_label("logFC", fontsize=10)

    # dot size legend
    ref = np.nanpercentile(df_plot["AveExpr"], [20, 50, 80])
    ref_sizes = map_sizes(ref, smin=26, smax=250)
    handles = [
        ax.scatter([], [], s=ref_sizes[i], c="gray", alpha=0.7, label=f"AveExpr≈{ref[i]:.2g}")
        for i in range(len(ref))
    ]
    handles.append(
        ax.scatter([], [], s=80, facecolors="white", edgecolors="black", linewidths=0.7,
                   label=f"adj.P.Val≤{sig_outline_fdr:g}")
    )
    ax.legend(
        handles=handles,
        title="Dot encoding",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.54),
        frameon=False,
    )

    fig.savefig(out_png, bbox_inches="tight")
    # fig.savefig(out_pdf, bbox_inches="tight")
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

    if len(marker_seed) == 0:
        return None

    # representative genes
    top = pick_topk_genes_per_marker(
        df=df,
        markers=marker_seed,
        topk=topk,
        fdr_thresh=fdr_thresh,
        prefer_significant=True,
    )
    genes_seed = sorted(pd.unique(top["gene"]))

    df_seed = df[df["gene"].isin(genes_seed) & df["marker"].isin(marker_seed)].copy()

    # final displayed marker order EXACTLY from heatmap logic
    marker_order, corr = marker_order_from_v2_heatmap_logic(
        df_plot=df_seed,
        marker_order_seed=marker_seed,
        gene_order_seed=genes_seed,
    )

    # gene order based on that final marker order
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
    pd.DataFrame({"marker": marker_order}).to_csv(
        os.path.join(out_dir, f"marker_order_{out_prefix}.csv"), index=False
    )
    pd.DataFrame({"gene": gene_order}).to_csv(
        os.path.join(out_dir, f"gene_order_{out_prefix}.csv"), index=False
    )

    # # optional separate outputs
    # plot_marker_correlation_heatmap(
    #     corr=corr,
    #     out_png=os.path.join(out_dir, f"marker_corr_{out_prefix}.png"),
    #     out_pdf=os.path.join(out_dir, f"marker_corr_{out_prefix}.pdf"),
    #     title=f"{title_prefix}: marker–marker correlation",
    # )
    #
    # plot_dotplot_only(
    #     df_plot=df_plot,
    #     marker_order=marker_order,
    #     gene_order=gene_order,
    #     block_ranges=block_ranges,
    #     out_png=os.path.join(out_dir, f"dotplot_{out_prefix}.png"),
    #     out_pdf=os.path.join(out_dir, f"dotplot_{out_prefix}.pdf"),
    #     title=f"{title_prefix}: protein–gene differential dotplot",
    #     sig_outline_fdr=fdr_thresh,
    # )

    # new strictly aligned single-axis figure
    plot_combined_corr_dotplot_single_axis(
        corr=corr,
        df_plot=df_plot,
        marker_order=marker_order,
        gene_order=gene_order,
        block_ranges=block_ranges,
        out_png=os.path.join(out_dir, f"marker_corr_dotplot_{out_prefix}.png"),
        out_pdf=os.path.join(out_dir, f"marker_corr_dotplot_{out_prefix}.pdf"),
        title=f"{title_prefix}: marker modules and protein–gene differential map",
        sig_outline_fdr=fdr_thresh,
        gap_rows=2.0,
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
    sns.set_style("whitegrid")

    ap = argparse.ArgumentParser()
    ap.add_argument("--root_folder", default="/media/huifang/data/registration/phenocycler/haotian/")
    ap.add_argument("--in_rds", default="xenium_DE.rds")
    ap.add_argument("--out_dir", default="de_dotplot_matrix")
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
    sel_canon = set(canon(x) for x in sel_markers)
    df_sel = df[df["marker"].map(canon).isin(sel_canon)].copy()

    run_panel_analysis(
        df=df_sel,
        markers_subset=sel_markers,
        preserve_user_order=args.preserve_selected_order,
        topk=args.topk_selected,
        fdr_thresh=args.fdr_thresh,
        out_prefix="selected_markers",
        out_dir=out_dir,
        title_prefix="Selected markers",
    )

    print("[DONE] Results saved to:", out_dir)
    print("Combined figure now uses one main plotting axis, so heatmap and dotplot columns are exactly aligned.")


if __name__ == "__main__":
    main()
