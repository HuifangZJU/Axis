#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-series delta analysis from 3 delta h5ad files.

What this script does
---------------------
Given three delta AnnData files (for example, three time intervals or three
different pairwise contrasts), it will:

1. Build domain-wise gene delta summaries for each interval
2. Quantify domain-level delta burden over intervals
3. Summarize directional changes (fraction positive / negative)
4. Classify simple temporal delta patterns for each gene within each domain
5. Plot:
   - domain x interval absolute-delta burden heatmap
   - domain x interval signed-delta burden heatmap
   - domain x interval fraction-up / fraction-down heatmaps
   - selected-gene temporal line panels
   - per-domain top-gene heatmaps across intervals
   - optional spatial delta maps for selected genes, if coordinates exist

Input expectation
-----------------
Each delta file should be an h5ad whose X is the signed delta matrix.
All files should share genes, and ideally domain labels via obs[DOMAIN_KEY].

Typical examples of DELTA_FILES labels:
- ("2.5→5.7", "/path/to/delta_5p7_minus_2p5.h5ad")
- ("5.7→13.4", "/path/to/delta_13p4_minus_5p7.h5ad")
- ("2.5→13.4", "/path/to/delta_13p4_minus_2p5.h5ad")

Run
---
Edit the CONFIG block below, then run:

    python time_series_delta_analysis.py
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIG
# ============================================================

DELTA_FILES: List[Tuple[str, str]] = [
    ("interval_1", "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/differences_pairs/diff_forward_2.500_minus_102.500.h5ad"),
    ("interval_2", "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/differences_pairs/diff_forward_5.700_minus_105.700.h5ad"),
    ("interval_3", "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/differences_pairs/diff_forward_13.400_minus_113.400.h5ad"),
]

OUTDIR = "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/differences_pairs/evaluation"

DOMAIN_KEY = "axis_joint_domain"
FILTER_VALID = True
VALID_KEY = "valid"

# delta files are usually already signed matrices; do not log1p by default
APPLY_LOG1P = False

SELECTED_DOMAINS: Optional[List[str]] = None
MIN_SPOTS_PER_DOMAIN = 20

# gene handling
MANUAL_GENES: Optional[List[str]] = None
MAX_GENES_FOR_METRICS = 2000
MIN_GLOBAL_NONZERO_FRAC = 0.005

MARKER_GENES: Optional[List[str]] = None
N_SELECTED_GENES = 12
TOP_GENES_PER_DOMAIN_HEATMAP = 20
TOP_DOMAINS_FOR_GENE_PANEL = 6
TOP_SPATIAL_GENES = 6

# thresholds / summaries
SMALL_DELTA_EPS = 0.05
TOP_ABS_QUANTILE = 0.90

# plotting
FIG_DPI = 300
SPATIAL_POINT_SIZE = 6


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
        "grid.alpha": 0.16,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# BASIC HELPERS
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


def get_domain_labels(ad, domain_key=DOMAIN_KEY):
    if domain_key not in ad.obs.columns:
        raise KeyError(f"Missing ad.obs['{domain_key}']")
    return ad.obs[domain_key].astype(str).to_numpy()


def get_selected_domains(ad_dict: Dict[str, "AnnData"]) -> List[str]:
    if SELECTED_DOMAINS is not None:
        return [str(x) for x in SELECTED_DOMAINS]

    ds = set()
    for ad in ad_dict.values():
        ds.update(np.unique(get_domain_labels(ad, DOMAIN_KEY)))
    ds = sorted([d for d in ds if d != "nan"])
    return ds


def build_domain_palette(domain_order: Sequence[str]):
    n = max(20, len(domain_order))
    colors = sns.color_palette("husl", n_colors=n)
    return {str(dom): colors[i % len(colors)] for i, dom in enumerate(domain_order)}


def compute_full_ylim(values: List[np.ndarray], pad_frac: float = 0.08):
    arrs = []
    for v in values:
        v = np.asarray(v, dtype=np.float32).ravel()
        v = v[np.isfinite(v)]
        if len(v) > 0:
            arrs.append(v)
    if len(arrs) == 0:
        return (-1, 1)
    x = np.concatenate(arrs)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if lo == hi:
        pad = 0.2 if lo == 0 else 0.08 * abs(lo)
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


# ============================================================
# EXACT GENE EXTRACTION
# ============================================================

def extract_gene_vector_exact(ad, gene: str, apply_log1p: bool = APPLY_LOG1P) -> np.ndarray:
    if gene not in ad.var_names:
        raise KeyError(f"Missing gene: {gene}")
    j = int(np.where(ad.var_names == gene)[0][0])
    v = ad.X[:, j]
    if sp.issparse(v):
        v = np.asarray(v.toarray()).ravel().astype(np.float32)
    else:
        v = np.asarray(v).ravel().astype(np.float32)
    if apply_log1p:
        v = np.log1p(v)
    return v


def extract_gene_matrix_exact(ad, genes: Sequence[str], apply_log1p: bool = APPLY_LOG1P) -> np.ndarray:
    genes = list(genes)
    missing = [g for g in genes if g not in ad.var_names]
    if missing:
        raise KeyError(f"Missing genes: {missing[:5]} ...")
    idx = [int(np.where(ad.var_names == g)[0][0]) for g in genes]
    X = ad.X[:, idx]
    if sp.issparse(X):
        X = np.asarray(X.toarray(), dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)
    if apply_log1p:
        X = np.log1p(X)
    return X


def validate_extraction_consistency(ads: Dict[str, "AnnData"], genes: Sequence[str], n_check: int = 8):
    genes = list(genes[: min(n_check, len(genes))])
    if not genes:
        return

    for name, ad in ads.items():
        X = extract_gene_matrix_exact(ad, genes, apply_log1p=APPLY_LOG1P)
        for gi, gene in enumerate(genes):
            v = extract_gene_vector_exact(ad, gene, apply_log1p=APPLY_LOG1P)
            if not np.allclose(X[:, gi], v, atol=1e-6, equal_nan=True):
                diff = np.nanmax(np.abs(X[:, gi] - v))
                raise RuntimeError(f"Extraction mismatch for {gene} in {name}; max diff = {diff}")
    print("[CHECK] Batched extraction matches exact single-gene extraction.")


# ============================================================
# GENE SELECTION
# ============================================================

def get_common_genes(ad_dict: Dict[str, "AnnData"]) -> List[str]:
    common = None
    for ad in ad_dict.values():
        g = set(ad.var_names.tolist())
        common = g if common is None else common.intersection(g)
    return sorted(common) if common is not None else []


def select_genes_for_analysis(ad_dict: Dict[str, "AnnData"]) -> List[str]:
    if MANUAL_GENES is not None:
        common = get_common_genes(ad_dict)
        return [g for g in MANUAL_GENES if g in common]

    common_genes = get_common_genes(ad_dict)
    if len(common_genes) == 0:
        return []

    mean_list = []
    nonzero_list = []

    for ad in ad_dict.values():
        X = extract_gene_matrix_exact(ad, common_genes, apply_log1p=APPLY_LOG1P)
        mean_list.append(np.mean(np.abs(X), axis=0))
        nonzero_list.append(np.mean(np.abs(X) > 0, axis=0))

    mean_mat = np.row_stack(mean_list)
    nonzero_mat = np.row_stack(nonzero_list)

    mean_abs_score = np.nanmean(mean_mat, axis=0)
    var_abs_score = np.nanvar(mean_mat, axis=0)
    nonzero_score = np.nanmean(nonzero_mat, axis=0)

    keep = nonzero_score >= MIN_GLOBAL_NONZERO_FRAC
    score = var_abs_score * np.log1p(mean_abs_score + 1e-8)

    df = pd.DataFrame({
        "gene": common_genes,
        "mean_abs": mean_abs_score,
        "var_abs": var_abs_score,
        "nonzero": nonzero_score,
        "score": score,
        "keep": keep,
    })
    df = df[df["keep"]].sort_values("score", ascending=False).head(MAX_GENES_FOR_METRICS)
    return df["gene"].tolist()


def select_plot_genes(domain_gene_delta_df: pd.DataFrame, all_genes: Sequence[str]) -> List[str]:
    if MARKER_GENES is not None:
        keep = [g for g in MARKER_GENES if g in domain_gene_delta_df["gene"].unique().tolist()]
        if len(keep) > 0:
            return keep

    tmp = domain_gene_delta_df.copy()
    gene_score = (
        tmp.groupby("gene", as_index=False)
        .agg(
            mean_abs=("delta_mean", lambda x: np.mean(np.abs(x))),
            var_delta=("delta_mean", "var"),
        )
    )
    gene_score["score"] = gene_score["mean_abs"].fillna(0) + 0.5 * gene_score["var_delta"].fillna(0)
    gene_score = gene_score.sort_values("score", ascending=False)

    plot_genes = gene_score["gene"].head(N_SELECTED_GENES).tolist()
    if len(plot_genes) == 0:
        plot_genes = list(all_genes[: min(N_SELECTED_GENES, len(all_genes))])
    return plot_genes


# ============================================================
# SPATIAL HELPERS
# ============================================================

def get_spatial_coords(ad):
    if "spatial" in ad.obsm:
        coords = np.asarray(ad.obsm["spatial"])
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2].astype(np.float32)

    xy_candidates = [
        ("x", "y"),
        ("X", "Y"),
        ("cx", "cy"),
        ("center_x", "center_y"),
        ("pixel_x", "pixel_y"),
    ]
    for xk, yk in xy_candidates:
        if xk in ad.obs.columns and yk in ad.obs.columns:
            return np.column_stack([
                np.asarray(ad.obs[xk]).astype(np.float32),
                np.asarray(ad.obs[yk]).astype(np.float32),
            ])
    return None


# ============================================================
# DOMAIN / GENE TABLES
# ============================================================

def build_domain_gene_delta_table(ad_dict: Dict[str, "AnnData"], interval_order: Sequence[str], domains: Sequence[str], genes: Sequence[str]) -> pd.DataFrame:
    rows = []

    for interval in interval_order:
        ad = ad_dict[interval]
        labels = get_domain_labels(ad, DOMAIN_KEY)
        X = extract_gene_matrix_exact(ad, genes, apply_log1p=APPLY_LOG1P)
        gene_to_idx = {g: i for i, g in enumerate(genes)}

        for dom in domains:
            m = labels == str(dom)
            n = int(np.sum(m))
            if n == 0:
                continue

            Xd = X[m]
            mean_vec = np.mean(Xd, axis=0).astype(np.float32)

            for gene in genes:
                gi = gene_to_idx[gene]
                rows.append({
                    "interval": interval,
                    "domain": str(dom),
                    "gene": gene,
                    "delta_mean": float(mean_vec[gi]),
                    "n_spots": n,
                })

    return pd.DataFrame(rows)


def build_domain_interval_summary(domain_gene_delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dom, interval), dfd in domain_gene_delta_df.groupby(["domain", "interval"]):
        vals = dfd["delta_mean"].to_numpy(dtype=np.float32)
        vals_good = vals[np.isfinite(vals)]
        if len(vals_good) == 0:
            continue

        abs_vals = np.abs(vals_good)
        top_thr = float(np.quantile(abs_vals, TOP_ABS_QUANTILE))
        top_mask = abs_vals >= top_thr

        rows.append({
            "domain": str(dom),
            "interval": interval,
            "n_genes": int(len(vals_good)),
            "mean_abs_delta": float(np.mean(abs_vals)),
            "signed_mean_delta": float(np.mean(vals_good)),
            "median_abs_delta": float(np.median(abs_vals)),
            "frac_positive": float(np.mean(vals_good > SMALL_DELTA_EPS)),
            "frac_negative": float(np.mean(vals_good < -SMALL_DELTA_EPS)),
            "frac_near_zero": float(np.mean(np.abs(vals_good) <= SMALL_DELTA_EPS)),
            "top_abs_delta_mean": float(np.mean(abs_vals[top_mask])) if np.any(top_mask) else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# TEMPORAL PATTERN CLASSIFICATION
# ============================================================

def _sign_symbol(v: float, eps: float = SMALL_DELTA_EPS) -> str:
    if not np.isfinite(v):
        return "0"
    if v > eps:
        return "+"
    if v < -eps:
        return "-"
    return "0"


def classify_sign_pattern(vals: Sequence[float], eps: float = SMALL_DELTA_EPS) -> str:
    signs = [_sign_symbol(v, eps=eps) for v in vals]

    if all(s == "+" for s in signs):
        return "persistent_up"
    if all(s == "-" for s in signs):
        return "persistent_down"
    if all(s == "0" for s in signs):
        return "stable_near_zero"

    nz = [s for s in signs if s != "0"]

    if len(nz) == 1:
        if nz[0] == "+":
            return "focal_up"
        return "focal_down"

    if "+" in nz and "-" not in nz:
        return "mixed_up"
    if "-" in nz and "+" not in nz:
        return "mixed_down"

    if len(nz) >= 2:
        if nz[0] == "+" and nz[-1] == "-":
            return "reversal_up_to_down"
        if nz[0] == "-" and nz[-1] == "+":
            return "reversal_down_to_up"

    return "mixed_complex"


def build_gene_pattern_table(domain_gene_delta_df: pd.DataFrame, interval_order: Sequence[str]) -> pd.DataFrame:
    rows = []

    pivot = (
        domain_gene_delta_df
        .pivot_table(index=["domain", "gene"], columns="interval", values="delta_mean", aggfunc="mean")
        .reindex(columns=interval_order)
    )

    for (dom, gene), row in pivot.iterrows():
        vals = row.to_numpy(dtype=np.float32)
        pattern = classify_sign_pattern(vals, eps=SMALL_DELTA_EPS)
        row_dict = {"domain": str(dom), "gene": gene, "pattern": pattern}
        for i, interval in enumerate(interval_order):
            row_dict[f"delta_{interval}"] = float(vals[i]) if np.isfinite(vals[i]) else np.nan
        rows.append(row_dict)

    return pd.DataFrame(rows)


# ============================================================
# PLOTS: DOMAIN-INTERVAL HEATMAPS
# ============================================================

def plot_domain_interval_heatmap(summary_df: pd.DataFrame, value_col: str, title: str, cbar_label: str, center=None, cmap="vlag", save_path=None):
    set_style()

    mat = summary_df.pivot_table(index="domain", columns="interval", values=value_col, aggfunc="mean")
    fig_w = max(5.6, 1.25 * max(3, mat.shape[1]) + 2.5)
    fig_h = max(5.0, 0.45 * max(6, mat.shape[0]) + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        mat,
        cmap=cmap,
        center=center,
        linewidths=0.2,
        linecolor="white",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Interval")
    ax.set_ylabel("Domain")
    save_or_show(fig, save_path)


def plot_pattern_counts_heatmap(pattern_df: pd.DataFrame, save_path=None):
    set_style()

    count_df = (
        pattern_df.groupby(["domain", "pattern"])
        .size()
        .reset_index(name="count")
        .pivot(index="domain", columns="pattern", values="count")
        .fillna(0)
    )

    ordered_cols = [
        "persistent_up", "persistent_down", "stable_near_zero",
        "focal_up", "focal_down",
        "mixed_up", "mixed_down",
        "reversal_up_to_down", "reversal_down_to_up",
        "mixed_complex",
    ]
    existing = [c for c in ordered_cols if c in count_df.columns]
    count_df = count_df.reindex(columns=existing)

    fig_w = max(7.8, 1.0 * max(6, count_df.shape[1]) + 2.0)
    fig_h = max(5.0, 0.45 * max(6, count_df.shape[0]) + 1.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        count_df,
        cmap="YlGnBu",
        linewidths=0.2,
        linecolor="white",
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Gene count"},
        ax=ax,
    )
    ax.set_title("Temporal delta-pattern counts by domain")
    ax.set_xlabel("Pattern category")
    ax.set_ylabel("Domain")
    ax.tick_params(axis="x", rotation=35)
    save_or_show(fig, save_path)


# ============================================================
# PLOTS: SELECTED GENES
# ============================================================

def plot_selected_gene_temporal_panel(domain_gene_delta_df: pd.DataFrame, plot_genes: Sequence[str], summary_df: pd.DataFrame, interval_order: Sequence[str], save_path=None):
    set_style()

    dom_rank = (
        summary_df.groupby("domain", as_index=False)["mean_abs_delta"]
        .mean()
        .sort_values("mean_abs_delta", ascending=False)
    )
    domain_order = dom_rank["domain"].head(TOP_DOMAINS_FOR_GENE_PANEL).tolist()
    domain_pal = build_domain_palette(domain_order)

    ncol = 4
    nrow = int(np.ceil(len(plot_genes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.2 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, gene in zip(axes, plot_genes):
        dfg = domain_gene_delta_df[
            (domain_gene_delta_df["gene"] == gene) &
            (domain_gene_delta_df["domain"].isin(domain_order))
        ].copy()

        if len(dfg) == 0:
            ax.axis("off")
            continue

        for dom in domain_order:
            dfd = dfg[dfg["domain"] == dom].copy()
            dfd["interval"] = pd.Categorical(dfd["interval"], categories=list(interval_order), ordered=True)
            dfd = dfd.sort_values("interval")
            ax.plot(
                np.arange(len(dfd)),
                dfd["delta_mean"].to_numpy(dtype=float),
                marker="o",
                lw=2.0,
                label=str(dom),
                color=domain_pal[dom],
                alpha=0.95,
            )

        ax.axhline(0, color="black", lw=0.8, alpha=0.65)
        ax.set_title(gene)
        ax.set_xticks(np.arange(len(interval_order)))
        ax.set_xticklabels(interval_order, rotation=30, ha="right")
        ax.set_xlabel("Interval")
        ax.set_ylabel("Domain mean delta")

        vals = dfg["delta_mean"].to_numpy(dtype=float)
        lo, hi = compute_full_ylim([vals], pad_frac=0.12)
        ax.set_ylim(lo, hi)

    for ax in axes[len(plot_genes):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels() if len(plot_genes) > 0 else ([], [])
    if handles:
        fig.legend(handles, labels, title="Domain", frameon=False, loc="upper right")
    fig.suptitle("Selected genes: temporal delta trajectories by domain", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.96, 1.0])
    save_or_show(fig, save_path)


def plot_per_domain_top_gene_heatmaps(domain_gene_delta_df: pd.DataFrame, interval_order: Sequence[str], save_dir: str):
    set_style()
    ensure_outdir(save_dir)

    for dom in sorted(domain_gene_delta_df["domain"].unique().tolist()):
        dfd = domain_gene_delta_df[domain_gene_delta_df["domain"] == dom].copy()

        gene_rank = (
            dfd.groupby("gene", as_index=False)
            .agg(mean_abs=("delta_mean", lambda x: np.mean(np.abs(x))), var_delta=("delta_mean", "var"))
        )
        gene_rank["score"] = gene_rank["mean_abs"].fillna(0) + 0.5 * gene_rank["var_delta"].fillna(0)
        top_genes = gene_rank.sort_values("score", ascending=False)["gene"].head(TOP_GENES_PER_DOMAIN_HEATMAP).tolist()

        mat = (
            dfd[dfd["gene"].isin(top_genes)]
            .pivot_table(index="gene", columns="interval", values="delta_mean", aggfunc="mean")
            .reindex(index=top_genes, columns=interval_order)
        )

        fig_w = max(5.4, 1.15 * max(3, len(interval_order)) + 1.8)
        fig_h = max(5.5, 0.34 * max(10, len(top_genes)) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            mat,
            cmap="vlag",
            center=0,
            linewidths=0.1,
            linecolor="white",
            cbar_kws={"label": "Domain mean delta"},
            ax=ax,
        )
        ax.set_title(f"Domain {dom}: top temporal delta genes")
        ax.set_xlabel("Interval")
        ax.set_ylabel("Gene")
        save_or_show(fig, os.path.join(save_dir, f"domain_{dom}_top_gene_heatmap.png"))


# ============================================================
# PLOTS: OPTIONAL SPATIAL MAPS
# ============================================================

def plot_spatial_gene_panels(ad_dict: Dict[str, "AnnData"], interval_order: Sequence[str], genes: Sequence[str], save_dir: str):
    set_style()
    ensure_outdir(save_dir)

    coords_dict = {interval: get_spatial_coords(ad_dict[interval]) for interval in interval_order}
    if all(coords_dict[k] is None for k in interval_order):
        print("[INFO] No spatial coordinates found; skipping spatial gene panels.")
        return

    for gene in genes:
        vals_list = []
        for interval in interval_order:
            ad = ad_dict[interval]
            if gene not in ad.var_names:
                continue
            v = extract_gene_vector_exact(ad, gene, apply_log1p=APPLY_LOG1P)
            vals_list.append(v)

        if len(vals_list) == 0:
            continue

        finite_chunks = [v[np.isfinite(v)] for v in vals_list if np.any(np.isfinite(v))]
        if len(finite_chunks) == 0:
            continue
        all_vals = np.concatenate(finite_chunks)
        vmax = float(np.max(np.abs(all_vals))) if len(all_vals) else 1.0
        vmax = max(vmax, 1e-6)

        ncol = len(interval_order)
        fig, axes = plt.subplots(1, ncol, figsize=(4.5 * ncol, 4.3), squeeze=False)
        axes = axes.ravel()
        sca = None

        for ax, interval in zip(axes, interval_order):
            ad = ad_dict[interval]
            coords = coords_dict[interval]
            ax.set_title(interval)

            if coords is None or gene not in ad.var_names:
                ax.axis("off")
                continue

            vals = extract_gene_vector_exact(ad, gene, apply_log1p=APPLY_LOG1P)
            sca = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=vals,
                s=SPATIAL_POINT_SIZE,
                cmap="vlag",
                vmin=-vmax,
                vmax=vmax,
                linewidths=0,
                rasterized=True,
            )
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        if sca is not None:
            cbar = fig.colorbar(sca, ax=axes.tolist(), shrink=0.8, pad=0.01)
            cbar.set_label("Delta")
        fig.suptitle(f"Spatial delta maps: {gene}", y=0.995, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_or_show(fig, os.path.join(save_dir, f"gene_{gene}_spatial.png"))


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_outdir(OUTDIR)
    figdir = os.path.join(OUTDIR, "figures")
    ensure_outdir(figdir)

    delta_files = [(label, path) for label, path in DELTA_FILES if str(path).strip()]
    if len(delta_files) < 2:
        raise ValueError("Please provide at least 2 delta files in DELTA_FILES.")

    interval_order = [label for label, _ in delta_files]
    ad_dict = {label: read_h5ad(path) for label, path in delta_files}

    domains = get_selected_domains(ad_dict)
    genes = select_genes_for_analysis(ad_dict)

    print(f"[INFO] intervals: {interval_order}")
    print(f"[INFO] domains: {domains}")
    print(f"[INFO] selected genes for analysis: {len(genes)}")

    validate_extraction_consistency(ad_dict, genes)

    domain_gene_delta_df = build_domain_gene_delta_table(
        ad_dict=ad_dict,
        interval_order=interval_order,
        domains=domains,
        genes=genes,
    )

    keep_domains = []
    for dom, dfd in domain_gene_delta_df.groupby("domain"):
        if int(np.nanmedian(dfd["n_spots"])) >= MIN_SPOTS_PER_DOMAIN:
            keep_domains.append(dom)

    domain_gene_delta_df = domain_gene_delta_df[domain_gene_delta_df["domain"].isin(keep_domains)].copy()

    summary_df = build_domain_interval_summary(domain_gene_delta_df)
    pattern_df = build_gene_pattern_table(domain_gene_delta_df, interval_order=interval_order)
    plot_genes = select_plot_genes(domain_gene_delta_df, genes)

    domain_gene_delta_df.to_csv(os.path.join(OUTDIR, "domain_gene_interval_delta.csv"), index=False)
    summary_df.to_csv(os.path.join(OUTDIR, "domain_interval_summary.csv"), index=False)
    pattern_df.to_csv(os.path.join(OUTDIR, "gene_pattern_table.csv"), index=False)

    plot_domain_interval_heatmap(
        summary_df=summary_df,
        value_col="mean_abs_delta",
        title="Domain-wise absolute delta burden across intervals",
        cbar_label="Mean |delta|",
        center=None,
        cmap="YlOrRd",
        save_path=os.path.join(figdir, "domain_interval_abs_burden_heatmap.png"),
    )

    plot_domain_interval_heatmap(
        summary_df=summary_df,
        value_col="signed_mean_delta",
        title="Domain-wise signed delta burden across intervals",
        cbar_label="Mean delta",
        center=0,
        cmap="vlag",
        save_path=os.path.join(figdir, "domain_interval_signed_burden_heatmap.png"),
    )

    plot_domain_interval_heatmap(
        summary_df=summary_df,
        value_col="frac_positive",
        title="Fraction of positive-shift genes by domain and interval",
        cbar_label="Fraction positive",
        center=None,
        cmap="YlGn",
        save_path=os.path.join(figdir, "domain_interval_fraction_positive_heatmap.png"),
    )

    plot_domain_interval_heatmap(
        summary_df=summary_df,
        value_col="frac_negative",
        title="Fraction of negative-shift genes by domain and interval",
        cbar_label="Fraction negative",
        center=None,
        cmap="YlGnBu",
        save_path=os.path.join(figdir, "domain_interval_fraction_negative_heatmap.png"),
    )

    plot_pattern_counts_heatmap(
        pattern_df=pattern_df,
        save_path=os.path.join(figdir, "domain_pattern_counts_heatmap.png"),
    )

    plot_selected_gene_temporal_panel(
        domain_gene_delta_df=domain_gene_delta_df,
        plot_genes=plot_genes,
        summary_df=summary_df,
        interval_order=interval_order,
        save_path=os.path.join(figdir, "selected_gene_temporal_panel.png"),
    )

    plot_per_domain_top_gene_heatmaps(
        domain_gene_delta_df=domain_gene_delta_df,
        interval_order=interval_order,
        save_dir=os.path.join(figdir, "per_domain_top_gene_heatmaps"),
    )

    spatial_genes = plot_genes[: min(TOP_SPATIAL_GENES, len(plot_genes))]
    plot_spatial_gene_panels(
        ad_dict=ad_dict,
        interval_order=interval_order,
        genes=spatial_genes,
        save_dir=os.path.join(figdir, "spatial_gene_panels"),
    )

    print(f"[DONE] All outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
