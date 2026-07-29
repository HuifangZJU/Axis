#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare domain-wise delta maps using four input slices.

Comparison
----------
real_delta(domain, gene)    = mean_expr(real_disease)    - mean_expr(real_normal)
virtual_delta(domain, gene) = mean_expr(virtual_disease) - mean_expr(virtual_normal)

All four inputs go through the same extraction and aggregation pipeline, so the
comparison is fair and avoids mismatch from separately precomputed delta files.

Inputs
------
1. real normal slice
2. real disease slice
3. virtual normal slice
4. virtual disease slice

Outputs
-------
Tables
- domain_gene_delta_table.csv
- domain_delta_metrics.csv
- domain_metric_long_for_boxplot.csv

Figures
- domain_metric_heatmap.png
- domain_metric_boxplot_all.png
- all_domain_scatter.png
- selected_gene_domain_panel.png
- selected_gene_domain_heatmap.png
- per_domain/domain_<id>_detail.png

Run
---
Edit the CONFIG block below, then run:

    python compare_real_virtual_delta_from_four_slices_boxplot.py
"""

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


# ============================================================
# CONFIG
# ============================================================


REAL_NORMAL_PATH = "/media/huifang/data/registration/xenium/mouse_brain/h5ads/0_4.h5ad"
REAL_DISEASE_PATH = "/media/huifang/data/registration/xenium/mouse_brain/h5ads/0_1.h5ad"
VIRTUAL_NORMAL_PATH = "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/virtual_Disease_time_105.700.h5ad"
VIRTUAL_DISEASE_PATH = "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/virtual_Disease_time_5.700.h5ad"

OUTDIR = "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/evaluation/disease"

DOMAIN_KEY = "axis_joint_domain"
FILTER_VALID = True
VALID_KEY = "valid"

# Apply the SAME transform to all four slices before subtraction.
# Set False if the four slices are already in the same raw-expression space.
# Set True if you want to compare delta in log-expression space.
USE_LOG1P = False

SELECTED_DOMAINS: Optional[List[str]] = None
MIN_SPOTS_PER_DOMAIN = 20

MANUAL_GENES: Optional[List[str]] = None
MAX_GENES_FOR_METRICS = 2000
MIN_GLOBAL_DETECTION_FRAC = 0.01

MARKER_GENES: Optional[List[str]] = ['Gfap','Arc','Mbp','Slc17a7']
AUTO_PLOT_GENE_N = 12
TOP_LABEL_GENES = 10
TOP_HEATMAP_GENES = 20

# delta-specific evaluation helpers
REAL_EFFECT_ABS_MIN = 0.10
TOP_DELTA_QUANTILE = 0.90
TOP_K_OVERLAP = 20

FIG_DPI = 300

PALETTE = {
    "real": "#2f2f2f",
    "virtual": "#cc6b6b",
}

METRIC_ORDER = [
    "pearson_r",
    "spearman_r",
    "cosine",
    "sign_accuracy",
    "auroc_high_abs_delta",
]

METRIC_DISPLAY = {
    "pearson_r": "Pearson",
    "spearman_r": "Spearman",
    "cosine": "Cosine",
    "sign_accuracy": "Sign accuracy",
    "auroc_high_abs_delta": "AUROC high |Δ|",
}


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
# IO / BASIC HELPERS
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


def get_selected_domains(ad_list):
    if SELECTED_DOMAINS is not None:
        return [str(x) for x in SELECTED_DOMAINS]

    ds = set()
    for ad in ad_list:
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

def extract_gene_vector_exact(ad, gene: str, apply_log1p: bool = USE_LOG1P) -> np.ndarray:
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


def extract_gene_matrix_exact(ad, genes: Sequence[str], apply_log1p: bool = USE_LOG1P) -> np.ndarray:
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
        X = extract_gene_matrix_exact(ad, genes, apply_log1p=USE_LOG1P)
        for gi, gene in enumerate(genes):
            v = extract_gene_vector_exact(ad, gene, apply_log1p=USE_LOG1P)
            if not np.allclose(X[:, gi], v, atol=1e-6, equal_nan=True):
                diff = np.nanmax(np.abs(X[:, gi] - v))
                raise RuntimeError(f"Extraction mismatch for {gene} in {name}; max diff = {diff}")
    print("[CHECK] Batched extraction matches exact single-gene extraction.")


# ============================================================
# GENE SELECTION
# ============================================================

def get_common_genes(ad_list) -> List[str]:
    common = None
    for ad in ad_list:
        g = set(ad.var_names.tolist())
        common = g if common is None else common.intersection(g)
    return sorted(common) if common is not None else []


def select_genes_for_metrics(ad_real_n, ad_real_d, ad_virt_n, ad_virt_d) -> List[str]:
    if MANUAL_GENES is not None:
        return [
            g for g in MANUAL_GENES
            if g in ad_real_n.var_names and g in ad_real_d.var_names
            and g in ad_virt_n.var_names and g in ad_virt_d.var_names
        ]

    common_genes = get_common_genes([ad_real_n, ad_real_d, ad_virt_n, ad_virt_d])
    if len(common_genes) == 0:
        return []

    mean_list = []
    detect_list = []
    for ad in [ad_real_n, ad_real_d, ad_virt_n, ad_virt_d]:
        X = extract_gene_matrix_exact(ad, common_genes, apply_log1p=USE_LOG1P)
        mean_list.append(np.mean(X, axis=0))
        detect_list.append(np.mean(X > 0, axis=0))

    mean_mat = np.row_stack(mean_list)
    detect_mat = np.row_stack(detect_list)

    mean_score = np.nanmean(mean_mat, axis=0)
    var_score = np.nanvar(mean_mat, axis=0)
    detect_score = np.nanmean(detect_mat, axis=0)

    keep = detect_score >= MIN_GLOBAL_DETECTION_FRAC
    score = var_score * np.log1p(mean_score + 1e-8)

    df = pd.DataFrame({
        "gene": common_genes,
        "detect": detect_score,
        "mean": mean_score,
        "var": var_score,
        "score": score,
        "keep": keep,
    })
    df = df[df["keep"]].sort_values("score", ascending=False).head(MAX_GENES_FOR_METRICS)
    return df["gene"].tolist()


def select_plot_genes(delta_df: pd.DataFrame, all_metric_genes: Sequence[str]) -> List[str]:
    if MARKER_GENES is not None:
        keep = [g for g in MARKER_GENES if g in delta_df["gene"].unique().tolist()]
        if len(keep) > 0:
            return keep

    tmp = delta_df.copy()
    tmp["abs_diff"] = np.abs(tmp["real_delta"] - tmp["virtual_delta"])
    tmp["abs_real"] = np.abs(tmp["real_delta"])
    gene_score = (
        tmp.groupby("gene", as_index=False)
        .agg(abs_diff_mean=("abs_diff", "mean"), abs_real_mean=("abs_real", "mean"))
    )
    gene_score["score"] = gene_score["abs_diff_mean"].fillna(0) + 0.3 * gene_score["abs_real_mean"].fillna(0)
    gene_score = gene_score.sort_values("score", ascending=False)

    plot_genes = gene_score["gene"].head(AUTO_PLOT_GENE_N).tolist()
    if len(plot_genes) == 0:
        plot_genes = list(all_metric_genes[: min(AUTO_PLOT_GENE_N, len(all_metric_genes))])
    return plot_genes


# ============================================================
# DOMAIN / GENE SUMMARY
# ============================================================

def build_domain_gene_means(ad, domains: Sequence[str], genes: Sequence[str], dataset_label: str) -> pd.DataFrame:
    labels = get_domain_labels(ad, DOMAIN_KEY)
    X = extract_gene_matrix_exact(ad, genes, apply_log1p=USE_LOG1P)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    rows = []
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
                "dataset": dataset_label,
                "domain": str(dom),
                "gene": gene,
                "mean_expr": float(mean_vec[gi]),
                "n_spots": n,
            })

    return pd.DataFrame(rows)


def build_domain_delta_table(ad_real_n, ad_real_d, ad_virt_n, ad_virt_d, domains: Sequence[str], genes: Sequence[str]) -> pd.DataFrame:
    df_rn = build_domain_gene_means(ad_real_n, domains, genes, "real_normal")
    df_rd = build_domain_gene_means(ad_real_d, domains, genes, "real_disease")
    df_vn = build_domain_gene_means(ad_virt_n, domains, genes, "virtual_normal")
    df_vd = build_domain_gene_means(ad_virt_d, domains, genes, "virtual_disease")

    merged = (
        df_rd.rename(columns={"mean_expr": "real_disease_mean", "n_spots": "real_disease_n"})
        .merge(
            df_rn.rename(columns={"mean_expr": "real_normal_mean", "n_spots": "real_normal_n"})[
                ["domain", "gene", "real_normal_mean", "real_normal_n"]
            ],
            on=["domain", "gene"],
            how="inner",
        )
        .merge(
            df_vd.rename(columns={"mean_expr": "virtual_disease_mean", "n_spots": "virtual_disease_n"})[
                ["domain", "gene", "virtual_disease_mean", "virtual_disease_n"]
            ],
            on=["domain", "gene"],
            how="inner",
        )
        .merge(
            df_vn.rename(columns={"mean_expr": "virtual_normal_mean", "n_spots": "virtual_normal_n"})[
                ["domain", "gene", "virtual_normal_mean", "virtual_normal_n"]
            ],
            on=["domain", "gene"],
            how="inner",
        )
    )

    merged["real_delta"] = merged["real_disease_mean"] - merged["real_normal_mean"]
    merged["virtual_delta"] = merged["virtual_disease_mean"] - merged["virtual_normal_mean"]
    merged["abs_error"] = np.abs(merged["real_delta"] - merged["virtual_delta"])
    merged["sq_error"] = (merged["real_delta"] - merged["virtual_delta"]) ** 2
    merged["abs_real_delta"] = np.abs(merged["real_delta"])
    merged["abs_virtual_delta"] = np.abs(merged["virtual_delta"])
    merged["informative_gene"] = merged["abs_real_delta"] >= REAL_EFFECT_ABS_MIN
    return merged


# ============================================================
# METRICS
# ============================================================

def safe_corr(x, y, fn):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 5:
        return np.nan
    try:
        return float(fn(x[good], y[good])[0])
    except Exception:
        return np.nan


def safe_cosine(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 5:
        return np.nan
    try:
        return float(cosine_similarity(x[good].reshape(1, -1), y[good].reshape(1, -1))[0, 0])
    except Exception:
        return np.nan


def safe_auc_high(real_values, pred_values, high_quantile=TOP_DELTA_QUANTILE):
    x = np.asarray(real_values, dtype=np.float32)
    y = np.asarray(pred_values, dtype=np.float32)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 5:
        return np.nan

    x = x[good]
    y = y[good]

    thr = float(np.quantile(np.abs(x), high_quantile))
    lab = (np.abs(x) >= thr).astype(int)
    if len(np.unique(lab)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(lab, np.abs(y)))
    except Exception:
        return np.nan


def safe_topk_overlap(real_values, pred_values, k=TOP_K_OVERLAP):
    x = np.asarray(real_values, dtype=np.float32)
    y = np.asarray(pred_values, dtype=np.float32)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < max(5, k):
        return np.nan

    idx = np.where(good)[0]
    xr = np.abs(x[good])
    yr = np.abs(y[good])

    k_eff = min(k, len(xr))
    top_x = set(idx[np.argsort(-xr)[:k_eff]].tolist())
    top_y = set(idx[np.argsort(-yr)[:k_eff]].tolist())

    inter = len(top_x.intersection(top_y))
    union = len(top_x.union(top_y))
    return float(inter / union) if union > 0 else np.nan


def compute_domain_metrics(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dom, dfd in delta_df.groupby("domain"):
        x = dfd["real_delta"].to_numpy(dtype=np.float32)
        y = dfd["virtual_delta"].to_numpy(dtype=np.float32)

        pearson_r = safe_corr(x, y, pearsonr)
        spearman_r = safe_corr(x, y, spearmanr)
        cosine = safe_cosine(x, y)

        good = np.isfinite(x) & np.isfinite(y)
        rmse = float(np.sqrt(mean_squared_error(x[good], y[good]))) if np.sum(good) >= 5 else np.nan
        mae = float(mean_absolute_error(x[good], y[good])) if np.sum(good) >= 5 else np.nan

        informative = dfd["informative_gene"].to_numpy(dtype=bool) & good
        sign_accuracy = float(np.mean(np.sign(x[informative]) == np.sign(y[informative]))) if np.sum(informative) >= 5 else np.nan

        auroc_high = safe_auc_high(x, y, high_quantile=TOP_DELTA_QUANTILE)
        topk_jaccard = safe_topk_overlap(x, y, k=TOP_K_OVERLAP)

        rows.append({
            "domain": str(dom),
            "n_genes": int(np.sum(good)),
            "n_informative": int(np.sum(informative)),
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "cosine": cosine,
            "sign_accuracy": sign_accuracy,
            "auroc_high_abs_delta": auroc_high,
        })

    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False).reset_index(drop=True)


def build_metric_long_df(metric_df: pd.DataFrame) -> pd.DataFrame:
    long_df = metric_df.melt(
        id_vars=["domain"],
        value_vars=METRIC_ORDER,
        var_name="metric",
        value_name="value",
    ).copy()
    long_df["metric_display"] = long_df["metric"].map(METRIC_DISPLAY)
    long_df["metric_display"] = pd.Categorical(
        long_df["metric_display"],
        categories=[METRIC_DISPLAY[m] for m in METRIC_ORDER],
        ordered=True,
    )
    return long_df


# ============================================================
# PLOTS: SUMMARY
# ============================================================

def plot_metric_heatmap(metric_df: pd.DataFrame, save_path=None):
    set_style()

    cols = METRIC_ORDER
    heat_df = metric_df.set_index("domain")[cols].copy()

    fig, ax = plt.subplots(figsize=(10.2, 0.55 * max(4, heat_df.shape[0]) + 1.8))
    sns.heatmap(
        heat_df,
        cmap="vlag",
        center=0 if np.nanmin(heat_df[["pearson_r", "spearman_r"]].values) < 0 else None,
        linewidths=0.2,
        linecolor="white",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Metric value"},
        ax=ax,
    )
    ax.set_title("Domain-level real-delta vs virtual-delta agreement summary")
    ax.set_xlabel("")
    ax.set_ylabel("Domain")
    save_or_show(fig, save_path)


def plot_metric_boxplot_all(metric_df: pd.DataFrame, save_path=None):
    set_style()

    long_df = build_metric_long_df(metric_df)
    domain_order = metric_df["domain"].tolist()
    domain_pal = build_domain_palette(domain_order)

    fig, ax = plt.subplots(figsize=(9.8, 7.0))

    sns.boxplot(
        data=long_df,
        x="metric_display",
        y="value",
        order=[METRIC_DISPLAY[m] for m in METRIC_ORDER],
        color="#eef2f7",
        width=0.62,
        fliersize=0,
        linewidth=1.15,
        ax=ax,
    )

    sns.stripplot(
        data=long_df,
        x="metric_display",
        y="value",
        order=[METRIC_DISPLAY[m] for m in METRIC_ORDER],
        hue="domain",
        palette=domain_pal,
        dodge=False,
        jitter=0.22,
        size=6.2,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.35,
        ax=ax,
        zorder=3,
    )

    # remove auto legend; build cleaner one
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    ax.axhline(0, color="black", lw=0.8, alpha=0.55)
    ax.set_title("Real vs virtual: domain-wise metric distribution", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.18)


    handles = [Line2D([0], [0], marker="o", linestyle="", color=domain_pal[d], label=d, markersize=6) for d in domain_order]
    handles.append(Line2D([0], [0], color="#6b7280", lw=2.1, marker="D", label="Median across domains"))
    ncol = 1 if len(handles) <= 12 else 2

    ax.legend(
        handles=handles,
        frameon=False,
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        title="Domain",
        ncol=ncol,
    )

    fig.tight_layout(rect=[0, 0, 0.86, 1.0])
    save_or_show(fig, save_path)


def plot_all_domain_scatter(delta_df: pd.DataFrame, save_path=None):
    set_style()
    domain_order = sorted(delta_df["domain"].unique().tolist())
    domain_pal = build_domain_palette(domain_order)

    fig, ax = plt.subplots(figsize=(6.6, 6.3))
    for dom in domain_order:
        dfd = delta_df[delta_df["domain"] == dom]
        ax.scatter(
            dfd["real_delta"], dfd["virtual_delta"],
            s=24, alpha=0.65, color=domain_pal[dom],
            edgecolor="white", linewidth=0.25,
            label=dom,
        )

    x = delta_df["real_delta"].to_numpy(dtype=np.float32)
    y = delta_df["virtual_delta"].to_numpy(dtype=np.float32)
    lo, hi = compute_full_ylim([x, y], pad_frac=0.08)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.1, color="0.4")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    pear = safe_corr(x, y, pearsonr)
    spear = safe_corr(x, y, spearmanr)
    ax.text(
        0.03, 0.97,
        f"Pearson = {pear:.2f}\nSpearman = {spear:.2f}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    ax.set_title("All domains: real delta vs virtual delta")
    ax.set_xlabel("Real delta")
    ax.set_ylabel("Virtual delta")
    save_or_show(fig, save_path)


# ============================================================
# PLOTS: GENE PANELS
# ============================================================

def plot_selected_gene_domain_panel(delta_df: pd.DataFrame, plot_genes: Sequence[str], domain_order: Sequence[str], save_path=None):
    set_style()

    ncol = 4
    nrow = int(np.ceil(len(plot_genes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, gene in zip(axes, plot_genes):
        dfg = delta_df[delta_df["gene"] == gene].copy()
        if len(dfg) == 0:
            ax.axis("off")
            continue

        dfg["domain"] = pd.Categorical(dfg["domain"], categories=list(domain_order), ordered=True)
        dfg = dfg.sort_values("domain")

        x = np.arange(len(dfg))
        ax.plot(x, dfg["real_delta"], color=PALETTE["real"], lw=2.0, marker="o", label="Real delta")
        ax.plot(x, dfg["virtual_delta"], color=PALETTE["virtual"], lw=2.0, marker="o", label="Virtual delta")

        ax.axhline(0, color="black", lw=0.8, alpha=0.7)
        ax.set_title(gene)
        ax.set_xticks(x)
        ax.set_xticklabels(dfg["domain"].tolist(), rotation=35, ha="right")
        ax.set_xlabel("Domain")
        ax.set_ylabel("Delta")

        lo, hi = compute_full_ylim([dfg["real_delta"].values, dfg["virtual_delta"].values], pad_frac=0.08)
        ax.set_ylim(lo, hi)

    for ax in axes[len(plot_genes):]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=PALETTE["real"], lw=2, marker="o", label="Real delta"),
        plt.Line2D([0], [0], color=PALETTE["virtual"], lw=2, marker="o", label="Virtual delta"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper right")
    fig.suptitle("Selected genes across domains: real vs virtual delta", y=0.995, fontsize=16)
    fig.tight_layout()
    save_or_show(fig, save_path)


# def plot_selected_gene_domain_heatmap(delta_df: pd.DataFrame, plot_genes: Sequence[str], domain_order: Sequence[str], save_path=None):
#     set_style()
#
#     dfg = delta_df[delta_df["gene"].isin(plot_genes)].copy()
#
#     real_mat = (
#         dfg.pivot_table(index="gene", columns="domain", values="real_delta", aggfunc="mean")
#         .reindex(index=plot_genes, columns=domain_order)
#     )
#     virt_mat = (
#         dfg.pivot_table(index="gene", columns="domain", values="virtual_delta", aggfunc="mean")
#         .reindex(index=plot_genes, columns=domain_order)
#     )
#
#     concat = pd.concat({"real": real_mat, "virtual": virt_mat}, axis=1)
#
#     fig, ax = plt.subplots(figsize=(0.75 * max(8, concat.shape[1]) + 2.8, 0.42 * max(8, concat.shape[0]) + 1.8))
#     sns.heatmap(
#         concat,
#         cmap="vlag",
#         center=0,
#         linewidths=0.1,
#         linecolor="white",
#         cbar_kws={"label": "Delta"},
#         ax=ax,
#     )
#     ax.set_title("Selected genes across domains: real vs virtual delta")
#     ax.set_xlabel("")
#     ax.set_ylabel("Gene")
#     save_or_show(fig, save_path)

import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr

def plot_selected_gene_domain_heatmap(
    delta_df: pd.DataFrame,
    plot_genes: Sequence[str],
    domain_order: Sequence[str],
    save_path=None,
):
    set_style()

    dfg = delta_df[delta_df["gene"].isin(plot_genes)].copy()
    if dfg.empty:
        raise ValueError("No rows found for the requested plot_genes.")

    # consistent domain colors across all subplots
    domain_colors = build_domain_palette(domain_order)

    n_genes = len(plot_genes)
    ncol = min(4, max(1, n_genes))
    nrow = math.ceil(n_genes / ncol)

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(4.0 * ncol, 4.0 * nrow),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, gene in zip(axes, plot_genes):
        sub = dfg[dfg["gene"] == gene].copy()

        # aggregate in case there are repeated rows
        sub = (
            sub.groupby("domain", as_index=False)
               .agg(real_delta=("real_delta", "mean"),
                    virtual_delta=("virtual_delta", "mean"))
        )

        sub["domain"] = pd.Categorical(sub["domain"], categories=list(domain_order), ordered=True)
        sub = sub.sort_values("domain")

        x = sub["real_delta"].to_numpy(dtype=float)
        y = sub["virtual_delta"].to_numpy(dtype=float)
        doms = sub["domain"].astype(str).tolist()

        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        doms = [d for d, g in zip(doms, good) if g]

        if len(x) == 0:
            ax.axis("off")
            continue

        # limits with padding, same scale on x/y inside each gene panel
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        if lo == hi:
            pad = 0.2 if lo == 0 else 0.08 * abs(lo)
        else:
            pad = 0.08 * (hi - lo)
        lo -= pad
        hi += pad

        # zero reference lines
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)

        # diagonal line y=x
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="0.4", zorder=1)

        # scatter points
        for xd, yd, dom in zip(x, y, doms):
            ax.scatter(
                xd,
                yd,
                s=70,
                color=domain_colors.get(dom, "#4c78a8"),
                edgecolor="white",
                linewidth=0.6,
                alpha=0.95,
                zorder=3,
            )

            # small text label for domain
            dx = 0.012 * (hi - lo)
            dy = 0.012 * (hi - lo)
            ax.text(
                xd + dx,
                yd + dy,
                dom,
                fontsize=8,
                color=domain_colors.get(dom, "#4c78a8"),
                zorder=4,
            )

        # optional correlation across domains for this gene
        if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            r = pearsonr(x, y)[0]
            title = f"{gene}\nr = {r:.2f}"
        else:
            title = gene

        ax.set_title(title)
        ax.set_xlabel("Real delta")
        ax.set_ylabel("Virtual delta")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

    # turn off unused axes
    for ax in axes[len(plot_genes):]:
        ax.axis("off")

    fig.suptitle("Selected genes across domains: real vs virtual delta", y=0.995, fontsize=16)
    fig.tight_layout()
    save_or_show(fig, save_path)

# ============================================================
# PLOTS: PER DOMAIN
# ============================================================

def pick_top_genes_for_domain(dfd: pd.DataFrame, n_label=TOP_LABEL_GENES, n_heat=TOP_HEATMAP_GENES):
    dfd = dfd.copy()
    dfd["rank_score"] = np.maximum(np.abs(dfd["real_delta"]), np.abs(dfd["virtual_delta"]))
    top = dfd.sort_values("rank_score", ascending=False)
    label_genes = top["gene"].head(n_label).tolist()
    heat_genes = top["gene"].head(n_heat).tolist()
    return label_genes, heat_genes


def plot_one_domain_detail(dfd: pd.DataFrame, metric_row: pd.Series, domain: str, save_path=None):
    set_style()

    label_genes, heat_genes = pick_top_genes_for_domain(dfd)
    dfd_sorted = dfd.sort_values("real_delta", ascending=False).reset_index(drop=True)
    domain_pal = build_domain_palette([domain])

    fig = plt.figure(figsize=(11.8, 7.2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.05], hspace=0.35, wspace=0.28)

    # scatter
    ax1 = fig.add_subplot(gs[0, 0])
    x = dfd["real_delta"].to_numpy(dtype=np.float32)
    y = dfd["virtual_delta"].to_numpy(dtype=np.float32)
    ax1.scatter(
        x, y,
        s=22, color=domain_pal[domain], alpha=0.72,
        edgecolor="white", linewidth=0.25
    )
    lo, hi = compute_full_ylim([x, y], pad_frac=0.10)
    ax1.plot([lo, hi], [lo, hi], ls="--", lw=1.1, color="0.4")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_title(f"Domain {domain}: real delta vs virtual delta")
    ax1.set_xlabel("Real delta")
    ax1.set_ylabel("Virtual delta")

    for gene in label_genes:
        row = dfd[dfd["gene"] == gene].iloc[0]
        if np.isfinite(row["real_delta"]) and np.isfinite(row["virtual_delta"]):
            ax1.text(row["real_delta"], row["virtual_delta"], gene, fontsize=8)

    txt = (
        f"Pearson = {metric_row['pearson_r']:.2f}\n"
        f"Spearman = {metric_row['spearman_r']:.2f}\n"
        f"Cosine = {metric_row['cosine']:.2f}\n"
        f"Sign acc = {metric_row['sign_accuracy']:.2f}"
    )
    ax1.text(
        0.04, 0.96, txt,
        transform=ax1.transAxes, va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.82, edgecolor="0.8")
    )

    # ranked curves
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(np.arange(len(dfd_sorted)), dfd_sorted["real_delta"], color=PALETTE["real"], lw=2.0, label="Real delta")
    ax2.plot(np.arange(len(dfd_sorted)), dfd_sorted["virtual_delta"], color=PALETTE["virtual"], lw=2.0, label="Virtual delta")
    ax2.axhline(0, color="black", lw=0.8, alpha=0.7)
    ax2.set_title("Ranked delta curves")
    ax2.set_xlabel("Genes ordered by real delta")
    ax2.set_ylabel("Delta")
    ax2.legend(frameon=False)
    ax2.set_ylim(*compute_full_ylim([dfd_sorted["real_delta"].values, dfd_sorted["virtual_delta"].values], pad_frac=0.08))

    for gene in label_genes[: min(8, len(label_genes))]:
        idx = np.where(dfd_sorted["gene"].values == gene)[0]
        if len(idx):
            ii = int(idx[0])
            yy = float(dfd_sorted.iloc[ii]["real_delta"])
            ax2.text(ii, yy, gene, fontsize=8)

    # heatmap
    ax3 = fig.add_subplot(gs[1, :])
    heat_df = (
        dfd[dfd["gene"].isin(heat_genes)][["gene", "real_delta", "virtual_delta"]]
        .set_index("gene")
        .loc[heat_genes]
    )
    sns.heatmap(
        heat_df,
        cmap="vlag",
        center=0,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Delta"},
        ax=ax3,
    )
    ax3.set_title("Top domain genes: real delta vs virtual delta")
    ax3.set_xlabel("")
    ax3.set_ylabel("Gene")

    fig.suptitle(f"Detailed delta validation for domain {domain}", y=0.995, fontsize=16)
    save_or_show(fig, save_path)


def plot_all_domain_details(delta_df: pd.DataFrame, metric_df: pd.DataFrame, save_dir: str):
    ensure_outdir(save_dir)

    for dom in metric_df["domain"].tolist():
        dfd = delta_df[delta_df["domain"] == dom].copy()
        mrow = metric_df[metric_df["domain"] == dom]
        if len(dfd) == 0 or len(mrow) == 0:
            continue
        plot_one_domain_detail(
            dfd=dfd,
            metric_row=mrow.iloc[0],
            domain=dom,
            save_path=os.path.join(save_dir, f"domain_{dom}_detail.png"),
        )


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_outdir(OUTDIR)
    figdir = os.path.join(OUTDIR, "figures")
    ensure_outdir(figdir)
    per_domain_dir = os.path.join(figdir, "per_domain")
    ensure_outdir(per_domain_dir)

    ad_real_n = read_h5ad(REAL_NORMAL_PATH)
    ad_real_d = read_h5ad(REAL_DISEASE_PATH)
    ad_virt_n = read_h5ad(VIRTUAL_NORMAL_PATH)
    ad_virt_d = read_h5ad(VIRTUAL_DISEASE_PATH)

    ads = {
        "real_normal": ad_real_n,
        "real_disease": ad_real_d,
        "virtual_normal": ad_virt_n,
        "virtual_disease": ad_virt_d,
    }

    domains = get_selected_domains([ad_real_n, ad_real_d, ad_virt_n, ad_virt_d])
    genes = select_genes_for_metrics(ad_real_n, ad_real_d, ad_virt_n, ad_virt_d)

    print(f"[INFO] domains: {domains}")
    print(f"[INFO] selected genes for metrics: {len(genes)}")

    validate_extraction_consistency(ads, genes)

    delta_df = build_domain_delta_table(
        ad_real_n=ad_real_n,
        ad_real_d=ad_real_d,
        ad_virt_n=ad_virt_n,
        ad_virt_d=ad_virt_d,
        domains=domains,
        genes=genes,
    )

    keep_domains = []
    for dom, dfd in delta_df.groupby("domain"):
        n_vals = [
            int(np.nanmedian(dfd["real_disease_n"])),
            int(np.nanmedian(dfd["real_normal_n"])),
            int(np.nanmedian(dfd["virtual_disease_n"])),
            int(np.nanmedian(dfd["virtual_normal_n"])),
        ]
        if min(n_vals) >= MIN_SPOTS_PER_DOMAIN:
            keep_domains.append(dom)

    delta_df = delta_df[delta_df["domain"].isin(keep_domains)].copy()
    metric_df = compute_domain_metrics(delta_df)
    metric_long_df = build_metric_long_df(metric_df)

    plot_genes = select_plot_genes(delta_df, genes)
    domain_order = metric_df["domain"].tolist()

    delta_df.to_csv(os.path.join(OUTDIR, "domain_gene_delta_table.csv"), index=False)
    metric_df.to_csv(os.path.join(OUTDIR, "domain_delta_metrics.csv"), index=False)
    metric_long_df.to_csv(os.path.join(OUTDIR, "domain_metric_long_for_boxplot.csv"), index=False)

    # summary plots
    plot_metric_boxplot_all(metric_df, save_path=os.path.join(figdir, "domain_metric_boxplot_all.png"))
    plot_selected_gene_domain_heatmap(
        delta_df=delta_df,
        plot_genes=plot_genes,
        domain_order=domain_order,
        save_path=os.path.join(figdir, "selected_gene_domain_heatmap.png"),
    )

    # per-domain detail
    plot_all_domain_details(delta_df, metric_df, save_dir=per_domain_dir)

    print(f"[DONE] All outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
