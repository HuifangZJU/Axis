#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare domain-wise delta maps using four input slices.

Comparison
----------
The quantitative evaluation preserves the original V2 domain-level definition:

real_delta(domain, gene)
    = mean(real_disease within domain) - mean(real_normal within domain)

virtual_delta(domain, gene)
    = mean(virtual_disease within domain) - mean(virtual_normal within domain)

All four count-like inputs use the same preprocessing before domain averaging:
per-location library-size normalization to NORMALIZATION_TARGET_SUM, followed by
optional log1p. Real and virtual domain-level metrics otherwise remain exactly
the original bulk/domain evaluation design.

Spatial delta map
-----------------
The lower-left per-domain panel is intentionally different: it uses the unified
location-wise spatial delta from spatial_delta_utils on the aligned virtual
disease/normal pair. The full virtual delta tissue is plotted, as in the disease
pipeline figure, while the requested domain is visually emphasized.

Validity handling
-----------------
- FILTER_VALID_FOR_DELTA controls the independent domain-level means used by the
  quantitative evaluation. It does NOT define the spatial-delta support.
- The spatial delta is always calculated after one-to-one alignment using the
  intersection of virtual disease/normal valid locations.
- FILTER_VALID_FOR_VISUALIZATION controls only display of the spatial-delta map.

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
- domain_gene_ranking_<real|virtual>.csv

Figures
- domain_metric_heatmap.png
- domain_metric_boxplot_all.png
- all_domain_scatter.png
- selected_gene_domain_panel.png
- selected_gene_domain_heatmap.png
- per_domain/domain_<id>_detail.png

Per-domain detail layout
------------------------
- upper left: all-gene real-delta vs virtual-delta scatter
- lower left: full location-level virtual disease-minus-normal delta map for
  the representative marker gene, with the selected domain emphasized
- right: ranked real/virtual domain-level delta heatmap

Run
---
Edit the CONFIG block below, then run:

    python compare_real_virtual_delta_from_four_slices_boxplot.py
"""

import os
import argparse
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from spatial_delta_utils import (
    compute_spatial_delta,
    extract_preprocessed_matrix,
)


# ============================================================
# CONFIG
# ============================================================


REAL_NORMAL_PATH = "/media/huifang/data/registration/xenium/mouse_brain/h5ads/0_4.h5ad"
REAL_DISEASE_PATH = "/media/huifang/data/registration/xenium/mouse_brain/h5ads/0_1.h5ad"
VIRTUAL_NORMAL_PATH = "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/virtual_Disease_time_105.700.h5ad"
VIRTUAL_DISEASE_PATH = "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/virtual_Disease_time_5.700.h5ad"

OUTDIR = "/media/huifang/data/registration/paper_figure/Figure6_disease/delta_fidelity"

# Saved evaluation object required by the per-domain spatial plots.
VIRTUAL_SPATIAL_DELTA_PATH = os.path.join(OUTDIR, "virtual_spatial_delta.h5ad")

DOMAIN_KEY = "axis_joint_domain"

# IMPORTANT:
# Keep all locations in memory so disease-minus-normal subtraction can be
# performed on the original matched virtual grids. Valid filtering is applied
# separately for quantitative aggregation and visualization.
FILTER_VALID_FOR_DELTA = True
FILTER_VALID_FOR_VISUALIZATION = True
VALID_KEY = "valid"

# Unified preprocessing rule used by BOTH domain-level and spatial-delta paths.
# Raw/count-like values are first normalized per location to the SAME target
# library size, then optionally log1p-transformed before any subtraction.
LIBRARY_SIZE_NORMALIZE = True
NORMALIZATION_TARGET_SUM = 253
LOG1P_BEFORE_ANALYSIS = True

# The spatial map always uses paired common-valid virtual support, independently
# of FILTER_VALID_FOR_DELTA, because location-wise subtraction requires matched
# disease/normal positions.
SPATIAL_DELTA_USE_COMMON_VALID = True
SPATIAL_DELTA_VMAX_PERCENTILE = 99.0
SPATIAL_DELTA_POINT_SIZE = 3.0
SPATIAL_DELTA_OTHER_ALPHA_SCALE = 0.38
SPATIAL_DELTA_ALPHA_MIN = 0.08
SPATIAL_DELTA_ALPHA_MAX = 0.95

SELECTED_DOMAINS: Optional[List[str]] = None
MIN_SPOTS_PER_DOMAIN = 20

MANUAL_GENES: Optional[List[str]] = None
MAX_GENES_FOR_METRICS = 2000
MIN_GLOBAL_DETECTION_FRAC = 0.01

MARKER_GENES: Optional[List[str]] = ['Gfap','Arc','Mbp','Slc17a7']
AUTO_PLOT_GENE_N = 12
TOP_LABEL_GENES = 10
TOP_HEATMAP_GENES = 20

# Per-domain top-gene ranking source.
#
# "virtual":
#   The virtual delta nominates the strongest genes, and the corresponding
#   real delta is shown as an external biological consistency check.
#
# "real":
#   The real delta defines the strongest reference genes, and virtual delta
#   is evaluated on that real-defined ordering.
#
# The default uses absolute delta magnitude so both strongly increased and
# strongly decreased genes can be selected. Set False to rank only from the
# largest positive signed delta.
TOP_GENE_RANK_MODE = "virtual"   # choices: "real", "virtual"
TOP_GENE_RANK_USE_ABS = True

# ------------------------------------------------------------
# Spatial-support control for per-domain marker nomination only
# ------------------------------------------------------------
# This does NOT change:
#   - real_delta / virtual_delta calculation,
#   - overall metrics,
#   - the ranking score itself.
#
# It only removes a candidate marker from the per-domain ranked list when the
# virtual spatial delta is supported by too few paired locations in that
# domain.
#
# For a gene with positive virtual bulk delta:
#     require fraction(location_delta > 0) >= threshold
#
# For a gene with negative virtual bulk delta:
#     require fraction(location_delta < 0) >= threshold
#
# The default 0.10 is the same mild 10% spatial-support idea used in the
# trajectory analysis: it guards against marker nomination driven by only a
# very small number of locations without forcing the response to be globally
# coherent across the domain.
USE_MARKER_LOCATION_FRACTION_FILTER = True
MIN_MARKER_DIRECTION_FRACTION = 0.10

# Compute the support fraction on paired virtual locations valid in BOTH
# virtual disease and virtual normal. This mask is used only for the marker
# eligibility test; the original bulk-delta evaluation remains unchanged.
MARKER_FRACTION_USE_COMMON_VALID = True

# Batch genes to keep the location-wise support calculation memory efficient.
MARKER_FRACTION_BATCH_SIZE = 128

# delta-specific evaluation helpers
REAL_EFFECT_ABS_MIN = 0.10
TOP_DELTA_QUANTILE = 0.90
TOP_K_OVERLAP = 20

FIG_DPI = 400

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
    """
    Read the complete AnnData without dropping invalid locations.

    This is essential for the virtual disease-minus-normal subtraction:
    virtual normal and virtual disease must remain on their original full
    spatial supports so their locations can be aligned before subtraction.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return sc.read_h5ad(path)


def get_valid_mask(ad, valid_key: str = VALID_KEY) -> np.ndarray:
    """
    Return a robust boolean validity mask without subsetting the AnnData.

    Missing valid labels are treated as all-valid.
    """
    if valid_key not in ad.obs.columns:
        return np.ones(ad.n_obs, dtype=bool)

    values = ad.obs[valid_key].to_numpy()

    if values.dtype == bool:
        return values.astype(bool, copy=False)

    # Handle common numeric encodings.
    if np.issubdtype(values.dtype, np.number):
        return np.asarray(values, dtype=float) != 0

    # Handle strings/categorical values safely.
    values = np.asarray(values).astype(str)
    values = np.char.lower(np.char.strip(values))
    return np.isin(values, ["true", "1", "yes", "y", "t"])


def get_analysis_mask(ad) -> np.ndarray:
    """
    Mask used for domain-level delta calculations.
    """
    if FILTER_VALID_FOR_DELTA:
        return get_valid_mask(ad)
    return np.ones(ad.n_obs, dtype=bool)


def get_domain_labels(ad, domain_key=DOMAIN_KEY):
    if domain_key not in ad.obs.columns:
        raise KeyError(f"Missing ad.obs['{domain_key}']")
    return ad.obs[domain_key].astype(str).to_numpy()


def get_selected_domains(ad_list):
    if SELECTED_DOMAINS is not None:
        return [str(x) for x in SELECTED_DOMAINS]

    ds = set()
    for ad in ad_list:
        labels = get_domain_labels(ad, DOMAIN_KEY)
        mask = get_analysis_mask(ad)
        ds.update(np.unique(labels[mask]))

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

def extract_gene_vector_exact(
    ad,
    gene: str,
    apply_log1p: bool = LOG1P_BEFORE_ANALYSIS,
) -> np.ndarray:
    """
    Extract one gene after the canonical count preprocessing.

    Library-size factors are calculated from ALL genes in ad.X, not from the
    requested marker alone.
    """
    if gene not in ad.var_names:
        raise KeyError(f"Missing gene: {gene}")

    matrix = extract_preprocessed_matrix(
        ad,
        row_indices=None,
        genes=[gene],
        normalize_total=LIBRARY_SIZE_NORMALIZE,
        target_sum=NORMALIZATION_TARGET_SUM,
        use_log1p=apply_log1p,
    )
    return matrix[:, 0].astype(np.float32, copy=False)


def extract_gene_matrix_exact(
    ad,
    genes: Sequence[str],
    apply_log1p: bool = LOG1P_BEFORE_ANALYSIS,
) -> np.ndarray:
    """
    Extract genes after the canonical count preprocessing.

    Library-size normalization always uses the complete gene library in ad.X.
    """
    genes = list(genes)
    missing = [g for g in genes if g not in ad.var_names]
    if missing:
        raise KeyError(f"Missing genes: {missing[:5]} ...")

    return extract_preprocessed_matrix(
        ad,
        row_indices=None,
        genes=genes,
        normalize_total=LIBRARY_SIZE_NORMALIZE,
        target_sum=NORMALIZATION_TARGET_SUM,
        use_log1p=apply_log1p,
    )


def extract_delta_gene_vector(ad_delta, gene: str) -> np.ndarray:
    """Read an already preprocessed/subtracted delta gene without reprocessing."""
    if gene not in ad_delta.var_names:
        raise KeyError(f"Missing delta gene: {gene}")
    j = int(ad_delta.var_names.get_loc(gene))
    values = ad_delta.X[:, j]
    if sp.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32).reshape(-1)


def extract_delta_gene_matrix(ad_delta, genes: Sequence[str]) -> np.ndarray:
    """Read already preprocessed/subtracted delta values without reprocessing."""
    genes = list(genes)
    idx = ad_delta.var_names.get_indexer(genes)
    if np.any(idx < 0):
        missing = [g for g, i in zip(genes, idx) if i < 0]
        raise KeyError(f"Missing delta genes: {missing[:5]}")
    matrix = ad_delta.X[:, idx]
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def validate_extraction_consistency(ads: Dict[str, "AnnData"], genes: Sequence[str], n_check: int = 8):
    genes = list(genes[: min(n_check, len(genes))])
    if not genes:
        return

    for name, ad in ads.items():
        X = extract_gene_matrix_exact(ad, genes, apply_log1p=LOG1P_BEFORE_ANALYSIS)
        for gi, gene in enumerate(genes):
            v = extract_gene_vector_exact(ad, gene, apply_log1p=LOG1P_BEFORE_ANALYSIS)
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
        X = extract_gene_matrix_exact(ad, common_genes, apply_log1p=LOG1P_BEFORE_ANALYSIS)
        analysis_mask = get_analysis_mask(ad)
        X = X[analysis_mask]

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

def build_domain_gene_means(
    ad,
    domains: Sequence[str],
    genes: Sequence[str],
    dataset_label: str,
) -> pd.DataFrame:
    """
    Build domain-level means using the quantitative-analysis mask.

    With FILTER_VALID_FOR_DELTA=False, all original locations are retained.
    """
    labels = get_domain_labels(ad, DOMAIN_KEY)
    analysis_mask = get_analysis_mask(ad)
    X = extract_gene_matrix_exact(ad, genes, apply_log1p=LOG1P_BEFORE_ANALYSIS)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    rows = []
    for dom in domains:
        m = (labels == str(dom)) & analysis_mask
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


def build_domain_delta_table(
    ad_real_n,
    ad_real_d,
    ad_virt_n,
    ad_virt_d,
    domains: Sequence[str],
    genes: Sequence[str],
) -> pd.DataFrame:
    """
    Build the ORIGINAL V2 domain-level delta table.

    This function deliberately does NOT summarize the location-wise spatial
    delta AnnData. Each of the four tissues is aggregated independently within
    each domain, and disease-minus-normal is then calculated from those domain
    means. This preserves the original evaluation semantics even when virtual
    disease and virtual normal have different valid-location sets.

    With LOG1P_BEFORE_ANALYSIS=True, each expression matrix is transformed by
    log1p before its domain mean is calculated.
    """
    df_rn = build_domain_gene_means(ad_real_n, domains, genes, "real_normal")
    df_rd = build_domain_gene_means(ad_real_d, domains, genes, "real_disease")
    df_vn = build_domain_gene_means(ad_virt_n, domains, genes, "virtual_normal")
    df_vd = build_domain_gene_means(ad_virt_d, domains, genes, "virtual_disease")

    merged = (
        df_rd.rename(
            columns={
                "mean_expr": "real_disease_mean",
                "n_spots": "real_disease_n",
            }
        )
        .merge(
            df_rn.rename(
                columns={
                    "mean_expr": "real_normal_mean",
                    "n_spots": "real_normal_n",
                }
            )[["domain", "gene", "real_normal_mean", "real_normal_n"]],
            on=["domain", "gene"],
            how="inner",
        )
        .merge(
            df_vd.rename(
                columns={
                    "mean_expr": "virtual_disease_mean",
                    "n_spots": "virtual_disease_n",
                }
            )[["domain", "gene", "virtual_disease_mean", "virtual_disease_n"]],
            on=["domain", "gene"],
            how="inner",
        )
        .merge(
            df_vn.rename(
                columns={
                    "mean_expr": "virtual_normal_mean",
                    "n_spots": "virtual_normal_n",
                }
            )[["domain", "gene", "virtual_normal_mean", "virtual_normal_n"]],
            on=["domain", "gene"],
            how="inner",
        )
    )

    merged["real_delta"] = (
        merged["real_disease_mean"] - merged["real_normal_mean"]
    )
    merged["virtual_delta"] = (
        merged["virtual_disease_mean"] - merged["virtual_normal_mean"]
    )
    merged["abs_error"] = np.abs(
        merged["real_delta"] - merged["virtual_delta"]
    )
    merged["sq_error"] = (
        merged["real_delta"] - merged["virtual_delta"]
    ) ** 2
    merged["abs_real_delta"] = np.abs(merged["real_delta"])
    merged["abs_virtual_delta"] = np.abs(merged["virtual_delta"])
    merged["informative_gene"] = (
        merged["abs_real_delta"] >= REAL_EFFECT_ABS_MIN
    )
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

def plot_metric_boxplot_all(
    metric_df: pd.DataFrame,
    save_path=None,
):
    set_style()

    long_df = build_metric_long_df(metric_df)

    # ============================================================
    # Domain palette
    #
    # Match the pipeline plotting exactly:
    #   1. convert domain IDs to integers
    #   2. sort numerically
    #   3. assign colors sequentially from tab20
    #
    # long_df["domain"] is kept as string, so convert the palette
    # keys back to strings for seaborn.
    # ============================================================

    domain_labels = sorted(
        {
            int(domain)
            for domain in metric_df["domain"].astype(str)
            if int(domain) >= 0
        }
    )

    cmap = plt.get_cmap("tab20")

    domain_pal_int = {
        label: cmap(i % 20)
        for i, label in enumerate(domain_labels)
    }

    domain_pal = {
        str(label): domain_pal_int[label]
        for label in domain_labels
    }

    # Use numeric domain order for legend as well.
    domain_order = [
        str(label)
        for label in domain_labels
    ]

    # Make sure hue values have matching string representation.
    long_df = long_df.copy()
    long_df["domain"] = long_df["domain"].astype(str)

    # ============================================================
    # Figure
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(8.5,6)
    )

    # ------------------------------------------------------------
    # Boxplots
    # ------------------------------------------------------------
    sns.boxplot(
        data=long_df,
        x="metric_display",
        y="value",
        order=[
            METRIC_DISPLAY[m]
            for m in METRIC_ORDER
        ],
        color="#eef2f7",
        width=0.62,
        fliersize=0,
        linewidth=1.5,
        ax=ax,
    )

    # ------------------------------------------------------------
    # Domain-colored points
    # ------------------------------------------------------------
    sns.stripplot(
        data=long_df,
        x="metric_display",
        y="value",
        order=[
            METRIC_DISPLAY[m]
            for m in METRIC_ORDER
        ],
        hue="domain",
        hue_order=domain_order,
        palette=domain_pal,
        dodge=False,
        jitter=0.22,
        size=8.0,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.55,
        ax=ax,
        zorder=3,
    )

    # Remove automatic seaborn legend.
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    # ============================================================
    # Axis formatting / enlarged typography
    # ============================================================

    ax.axhline(
        0,
        color="black",
        lw=1.0,
        alpha=0.55,
    )

    ax.set_title(
        "Real vs virtual: domain-wise comparison",
        fontsize=18,
        fontweight="bold",
        pad=16,
    )

    ax.set_xlabel("")

    ax.set_ylabel(
        "Metric value",
        fontsize=15,
        labelpad=10,
    )

    ax.tick_params(
        axis="x",
        labelsize=13,
        rotation=28,
        pad=6,
    )

    ax.tick_params(
        axis="y",
        labelsize=13,
    )

    # Make x labels slightly easier to read.
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.9,
    )

    # Slightly stronger visible spines.
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    # ============================================================
    # Legend
    # ============================================================

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=domain_pal[d],
            markeredgecolor="white",
            markeredgewidth=0.5,
            color=domain_pal[d],
            label=d,
            markersize=8.5,
        )
        for d in domain_order
    ]


    ax.legend(
        handles=handles,
        frameon=False,
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        title="Domain",
        ncol=1,
        fontsize=12.5,
        title_fontsize=14,
        labelspacing=0.70,
        handletextpad=0.65,
        borderaxespad=0.0,
    )

    # Reserve right margin for legend.
    fig.tight_layout(
        rect=[0.0, 0.0, 0.93, 1.0]
    )

    # plt.show()

    save_or_show(
        fig,
        save_path,
    )


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

def plot_selected_gene_domain_dotplot(
    delta_df: pd.DataFrame,
    plot_genes: Sequence[str],
    domain_order: Sequence[str],
    save_path=None,
):
    set_style()

    dfg = delta_df[delta_df["gene"].isin(plot_genes)].copy()
    if dfg.empty:
        raise ValueError("No rows found for the requested plot_genes.")

    # ------------------------------------------------------------
    # Use the same domain palette logic as the pipeline figure
    # ------------------------------------------------------------
    domain_labels = sorted(
        {
            int(d)
            for d in domain_order
            if int(d) >= 0
        }
    )
    cmap = plt.get_cmap("tab20")
    domain_colors = {
        str(label): cmap(i % 20)
        for i, label in enumerate(domain_labels)
    }

    ordered_domains = [str(d) for d in domain_labels]

    n_genes = len(plot_genes)
    ncol = min(4, max(1, n_genes))
    nrow = math.ceil(n_genes / ncol)

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(4.8 * ncol, 4.8 * nrow),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, gene in zip(axes, plot_genes):
        sub = dfg[dfg["gene"] == gene].copy()

        # aggregate in case there are repeated rows
        sub = (
            sub.groupby("domain", as_index=False)
               .agg(
                   real_delta=("real_delta", "mean"),
                   virtual_delta=("virtual_delta", "mean"),
               )
        )

        sub["domain"] = sub["domain"].astype(str)
        sub["domain"] = pd.Categorical(
            sub["domain"],
            categories=ordered_domains,
            ordered=True,
        )
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
        ax.axhline(0, color="0.85", lw=1.0, zorder=0)
        ax.axvline(0, color="0.85", lw=1.0, zorder=0)

        # diagonal line y=x
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="0.4", zorder=1)

        # scatter points
        for xd, yd, dom in zip(x, y, doms):
            color = domain_colors.get(dom, "#4c78a8")
            ax.scatter(
                xd,
                yd,
                s=110,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.95,
                zorder=3,
            )

            dx = 0.014 * (hi - lo)
            dy = 0.014 * (hi - lo)
            ax.text(
                xd + dx,
                yd + dy,
                dom,
                fontsize=10.5,
                color=color,
                fontweight="bold",
                zorder=4,
            )

        # correlation across domains for this gene
        if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            r = pearsonr(x, y)[0]
            title = f"{gene}\nr = {r:.2f}"
        else:
            title = gene

        ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
        ax.set_xlabel("Real delta", fontsize=13)
        ax.set_ylabel("Virtual delta", fontsize=13)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=11)

    # turn off unused axes
    for ax in axes[len(plot_genes):]:
        ax.axis("off")

    # fig.suptitle(
    #     "Selected genes across domains: real vs virtual delta",
    #     y=0.995,
    #     fontsize=18,
    #     fontweight="bold",
    # )
    fig.tight_layout()

    save_or_show(fig, save_path)
# ============================================================
# PLOTS: PER DOMAIN
# ============================================================

def validate_top_gene_rank_mode(rank_mode: str) -> str:
    """Validate and normalize the per-domain top-gene ranking mode."""
    rank_mode = str(rank_mode).strip().lower()
    if rank_mode not in {"real", "virtual"}:
        raise ValueError(
            f"Invalid rank_mode={rank_mode!r}; expected 'real' or 'virtual'."
        )
    return rank_mode


def rank_genes_for_domain(
    dfd: pd.DataFrame,
    rank_mode: str = TOP_GENE_RANK_MODE,
    use_absolute: bool = TOP_GENE_RANK_USE_ABS,
    location_fraction_df: Optional[pd.DataFrame] = None,
    min_direction_fraction: float = MIN_MARKER_DIRECTION_FRACTION,
) -> pd.DataFrame:
    """
    Rank genes by the ORIGINAL V2 domain-level delta rule, with an optional
    location-fraction eligibility filter.

    Ranking score
    -------------
    Unchanged from V2:

        rank_score = abs(rank_delta)

    when ``use_absolute=True``.

    Spatial eligibility
    -------------------
    When USE_MARKER_LOCATION_FRACTION_FILTER=True and a location-fraction
    table is supplied, a candidate must show the same delta direction as its
    virtual bulk delta in at least ``min_direction_fraction`` of paired virtual
    locations in that domain.

    Positive virtual bulk delta:
        marker_direction_fraction = positive_fraction

    Negative virtual bulk delta:
        marker_direction_fraction = negative_fraction

    The fraction changes eligibility only. It is NOT multiplied into the
    ranking score, so the scientific ranking criterion remains exactly the
    original absolute virtual/real bulk delta.
    """
    rank_mode = validate_top_gene_rank_mode(rank_mode)
    source_col = f"{rank_mode}_delta"

    required = {"gene", "real_delta", "virtual_delta", source_col}
    missing = sorted(required.difference(dfd.columns))
    if missing:
        raise KeyError(
            f"Missing columns for domain gene ranking: {missing}"
        )

    ranked = dfd.copy()
    ranked["rank_source"] = rank_mode
    ranked["rank_delta"] = pd.to_numeric(
        ranked[source_col],
        errors="coerce",
    )

    if use_absolute:
        ranked["rank_score"] = np.abs(ranked["rank_delta"])
    else:
        ranked["rank_score"] = ranked["rank_delta"]

    ranked = ranked[
        np.isfinite(ranked["rank_score"])
    ].copy()

    # --------------------------------------------------------
    # Optional 10% spatial-support eligibility control.
    # --------------------------------------------------------
    ranked["positive_fraction"] = np.nan
    ranked["negative_fraction"] = np.nan
    ranked["nonzero_fraction"] = np.nan
    ranked["n_fraction_locations"] = np.nan
    ranked["marker_direction_fraction"] = np.nan
    ranked["passes_marker_fraction"] = True

    if (
        USE_MARKER_LOCATION_FRACTION_FILTER
        and location_fraction_df is not None
        and not location_fraction_df.empty
    ):
        domains_here = ranked["domain"].astype(str).unique().tolist()
        if len(domains_here) != 1:
            raise ValueError(
                "rank_genes_for_domain expects rows from exactly one domain "
                "when location-fraction filtering is enabled."
            )

        domain = domains_here[0]

        frac = location_fraction_df[
            location_fraction_df["domain"].astype(str) == domain
        ][
            [
                "gene",
                "positive_fraction",
                "negative_fraction",
                "nonzero_fraction",
                "n_fraction_locations",
            ]
        ].copy()

        # Remove the placeholder columns before merging real values.
        ranked = ranked.drop(
            columns=[
                "positive_fraction",
                "negative_fraction",
                "nonzero_fraction",
                "n_fraction_locations",
            ]
        ).merge(
            frac,
            on="gene",
            how="left",
        )

        virtual_delta = pd.to_numeric(
            ranked["virtual_delta"],
            errors="coerce",
        ).to_numpy(dtype=float)
        positive_fraction = pd.to_numeric(
            ranked["positive_fraction"],
            errors="coerce",
        ).to_numpy(dtype=float)
        negative_fraction = pd.to_numeric(
            ranked["negative_fraction"],
            errors="coerce",
        ).to_numpy(dtype=float)
        nonzero_fraction = pd.to_numeric(
            ranked["nonzero_fraction"],
            errors="coerce",
        ).to_numpy(dtype=float)

        # For zero bulk delta, use the overall nonzero fraction. Such genes
        # normally rank very low anyway, but this makes the rule complete.
        direction_fraction = np.where(
            virtual_delta > 0,
            positive_fraction,
            np.where(
                virtual_delta < 0,
                negative_fraction,
                nonzero_fraction,
            ),
        )

        ranked["marker_direction_fraction"] = direction_fraction
        ranked["passes_marker_fraction"] = (
            np.isfinite(direction_fraction)
            & (
                direction_fraction
                >= float(min_direction_fraction)
            )
        )

        ranked = ranked[
            ranked["passes_marker_fraction"]
        ].copy()

    # Stable and reproducible ordering. Fraction is an eligibility threshold
    # only; it does NOT enter the score.
    ranked = ranked.sort_values(
        ["rank_score", "gene"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    ranked["rank"] = np.arange(
        1,
        len(ranked) + 1,
        dtype=int,
    )

    return ranked

def select_top_genes_for_domain(
    dfd: pd.DataFrame,
    rank_mode: str = TOP_GENE_RANK_MODE,
    use_absolute: bool = TOP_GENE_RANK_USE_ABS,
    n_label: int = TOP_LABEL_GENES,
    n_heat: int = TOP_HEATMAP_GENES,
    location_fraction_df: Optional[pd.DataFrame] = None,
):
    """
    Select scatter-label and heatmap genes from one common ranked ordering.

    The ranking score remains the original bulk-delta score. When enabled,
    the location-fraction rule is applied first as a candidate eligibility
    threshold.
    """
    ranked_df = rank_genes_for_domain(
        dfd=dfd,
        rank_mode=rank_mode,
        use_absolute=use_absolute,
        location_fraction_df=location_fraction_df,
        min_direction_fraction=MIN_MARKER_DIRECTION_FRACTION,
    )

    label_genes = (
        ranked_df["gene"]
        .head(int(n_label))
        .tolist()
    )
    heat_genes = (
        ranked_df["gene"]
        .head(int(n_heat))
        .tolist()
    )

    return ranked_df, label_genes, heat_genes

def build_domain_gene_ranking_table(
    delta_df: pd.DataFrame,
    rank_mode: str = TOP_GENE_RANK_MODE,
    use_absolute: bool = TOP_GENE_RANK_USE_ABS,
    location_fraction_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the exportable per-domain marker ranking table.

    When the marker-location filter is enabled, the exported rank corresponds
    to the eligible marker candidates after the fraction threshold.
    """
    rows = []

    for domain, dfd in delta_df.groupby(
        "domain",
        sort=False,
    ):
        ranked = rank_genes_for_domain(
            dfd=dfd,
            rank_mode=rank_mode,
            use_absolute=use_absolute,
            location_fraction_df=location_fraction_df,
            min_direction_fraction=MIN_MARKER_DIRECTION_FRACTION,
        ).copy()

        rows.append(ranked)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(
        rows,
        ignore_index=True,
    )

    keep_columns = [
        "domain",
        "rank",
        "rank_source",
        "gene",
        "rank_delta",
        "rank_score",
        "positive_fraction",
        "negative_fraction",
        "nonzero_fraction",
        "marker_direction_fraction",
        "passes_marker_fraction",
        "n_fraction_locations",
        "real_delta",
        "virtual_delta",
        "abs_error",
    ]

    keep_columns = [
        col
        for col in keep_columns
        if col in result.columns
    ]

    return result[keep_columns]



def _extract_single_gene(ad, gene: str, log1p: bool = True) -> np.ndarray:
    if gene not in ad.var_names:
        raise KeyError(f"Gene {gene} not found.")
    x = ad[:, [gene]].X
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if log1p:
        x = np.log1p(np.clip(x, a_min=0, a_max=None))
    return x


def _align_virtual_normal_to_disease(
    ad_virtual_normal,
    ad_virtual_disease,
    spatial_key: str = "spatial",
):
    """
    Align virtual-normal locations to virtual-disease order for valid
    location-level subtraction.

    Priority:
    1. identical obs_names in identical order;
    2. identical obs_name sets, reordered to disease order;
    3. identical spatial coordinates in identical row order.

    Otherwise an error is raised rather than silently subtracting unmatched
    locations.
    """
    if ad_virtual_normal.n_obs != ad_virtual_disease.n_obs:
        raise ValueError(
            "Virtual normal and virtual disease have different numbers of "
            f"locations: {ad_virtual_normal.n_obs} vs {ad_virtual_disease.n_obs}. "
            "Cannot compute a location-level delta map without explicit matching."
        )

    normal_names = np.asarray(ad_virtual_normal.obs_names.astype(str))
    disease_names = np.asarray(ad_virtual_disease.obs_names.astype(str))

    if np.array_equal(normal_names, disease_names):
        return np.arange(ad_virtual_disease.n_obs, dtype=int)

    if (
        len(np.unique(normal_names)) == len(normal_names)
        and len(np.unique(disease_names)) == len(disease_names)
        and set(normal_names.tolist()) == set(disease_names.tolist())
    ):
        normal_lookup = {name: i for i, name in enumerate(normal_names)}
        return np.asarray(
            [normal_lookup[name] for name in disease_names],
            dtype=int,
        )

    coords_n = np.asarray(
        ad_virtual_normal.obsm[spatial_key],
        dtype=np.float32,
    )[:, :2]
    coords_d = np.asarray(
        ad_virtual_disease.obsm[spatial_key],
        dtype=np.float32,
    )[:, :2]

    if coords_n.shape == coords_d.shape and np.allclose(
        coords_n,
        coords_d,
        atol=1e-5,
        rtol=0.0,
    ):
        return np.arange(ad_virtual_disease.n_obs, dtype=int)

    raise ValueError(
        "Virtual normal and virtual disease could not be matched one-to-one "
        "by obs_names or spatial coordinates. Explicit spatial matching is "
        "required before a location-level disease-minus-normal delta can be drawn."
    )


def build_common_virtual_valid_mask(
    ad_virtual_normal,
    ad_virtual_disease,
    normal_to_disease_idx: np.ndarray,
) -> np.ndarray:
    """
    Build the common valid set in virtual-disease row order.

    The subtraction itself uses the complete aligned virtual grids. For
    visualization, only locations valid in BOTH virtual normal and virtual
    disease are displayed when FILTER_VALID_FOR_VISUALIZATION=True.

    This avoids:
    - losing grid alignment by filtering before subtraction, and
    - showing invalid/background grid locations in the spatial map.
    """
    if not FILTER_VALID_FOR_VISUALIZATION:
        return np.ones(ad_virtual_disease.n_obs, dtype=bool)

    valid_d = get_valid_mask(ad_virtual_disease)
    valid_n_raw = get_valid_mask(ad_virtual_normal)
    valid_n_aligned = valid_n_raw[normal_to_disease_idx]

    return valid_d & valid_n_aligned



def build_virtual_marker_location_fraction_table(
    ad_virtual_delta,
    domains: Sequence[str],
    genes: Sequence[str],
    use_common_valid: bool = MARKER_FRACTION_USE_COMMON_VALID,
    batch_size: int = MARKER_FRACTION_BATCH_SIZE,
    domain_key: str = DOMAIN_KEY,
) -> pd.DataFrame:
    """
    Calculate directional location support directly from the canonical spatial
    delta AnnData. No subtraction or log transform is repeated here.
    """
    genes = list(genes)
    domains = [str(x) for x in domains]

    if len(genes) == 0:
        return pd.DataFrame(
            columns=[
                "domain",
                "gene",
                "positive_fraction",
                "negative_fraction",
                "nonzero_fraction",
                "n_fraction_locations",
            ]
        )

    labels = get_domain_labels(ad_virtual_delta, domain_key)

    if use_common_valid and "_delta_common_valid" in ad_virtual_delta.obs.columns:
        support_mask = ad_virtual_delta.obs["_delta_common_valid"].to_numpy(dtype=bool)
    else:
        support_mask = np.ones(ad_virtual_delta.n_obs, dtype=bool)

    rows = []
    batch_size = max(1, int(batch_size))

    for start_index in range(0, len(genes), batch_size):
        batch_genes = genes[start_index:start_index + batch_size]
        location_delta = extract_delta_gene_matrix(
            ad_virtual_delta,
            batch_genes,
        )

        for domain in domains:
            domain_mask = (labels == str(domain)) & support_mask
            n_locations = int(np.sum(domain_mask))
            if n_locations == 0:
                continue

            delta_domain = location_delta[domain_mask, :]
            positive_fraction = np.mean(delta_domain > 0, axis=0)
            negative_fraction = np.mean(delta_domain < 0, axis=0)
            nonzero_fraction = np.mean(delta_domain != 0, axis=0)

            rows.extend(
                {
                    "domain": str(domain),
                    "gene": gene,
                    "positive_fraction": float(positive_fraction[index]),
                    "negative_fraction": float(negative_fraction[index]),
                    "nonzero_fraction": float(nonzero_fraction[index]),
                    "n_fraction_locations": n_locations,
                }
                for index, gene in enumerate(batch_genes)
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(
            "No virtual marker-location fraction rows were generated."
        )
    return result



def plot_virtual_domain_delta_map(
    ax,
    ad_virtual_delta,
    domain: str,
    gene: str,
    norm: Normalize,
    cmap_name: str = "RdBu_r",
    spatial_key: str = "spatial",
    domain_key: str = DOMAIN_KEY,
    point_size: float = SPATIAL_DELTA_POINT_SIZE,
):
    """
    Plot the location-wise virtual disease-minus-normal spatial delta.

    No separate colorbar is created. The spatial map uses the same colormap
    and normalization as the domain-level heatmap, so the heatmap colorbar
    serves as the shared delta legend for both panels.
    """

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    coords = np.asarray(
        ad_virtual_delta.obsm[spatial_key],
        dtype=np.float32,
    )[:, :2]

    labels = get_domain_labels(
        ad_virtual_delta,
        domain_key,
    )

    delta_values = extract_delta_gene_vector(
        ad_virtual_delta,
        gene,
    )

    # ------------------------------------------------------------
    # Visualization support
    # ------------------------------------------------------------
    if (
        FILTER_VALID_FOR_VISUALIZATION
        and "_delta_common_valid" in ad_virtual_delta.obs.columns
    ):
        vis_mask = ad_virtual_delta.obs[
            "_delta_common_valid"
        ].to_numpy(dtype=bool)
    else:
        vis_mask = np.ones(
            ad_virtual_delta.n_obs,
            dtype=bool,
        )

    vis_mask &= np.isfinite(delta_values)
    vis_mask &= np.all(
        np.isfinite(coords),
        axis=1,
    )

    domain_mask = (
        (labels == str(domain))
        & vis_mask
    )

    other_mask = (
        vis_mask
        & ~domain_mask
    )

    if not np.any(domain_mask):
        raise ValueError(
            f"No virtual delta locations found for domain {domain}."
        )

    domain_values = delta_values[
        domain_mask
    ]

    cmap = plt.get_cmap(
        cmap_name
    )

    # ------------------------------------------------------------
    # Whole-brain context
    # ------------------------------------------------------------
    if np.any(other_mask):
        ax.scatter(
            coords[other_mask, 0],
            coords[other_mask, 1],
            s=max(
                0.8,
                point_size * 0.72,
            ),
            color="#D5D5D5",
            alpha=0.28,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    # ------------------------------------------------------------
    # Selected domain
    #
    # COLOR uses the shared heatmap normalization.
    #
    # Alpha is still calculated locally so weak spatial signal does
    # not disappear simply because another panel has a larger Δ range.
    # ------------------------------------------------------------
    colors = cmap(
        norm(domain_values)
    )

    finite_values = domain_values[
        np.isfinite(domain_values)
    ]

    if finite_values.size:
        local_vmax = float(
            np.percentile(
                np.abs(finite_values),
                SPATIAL_DELTA_VMAX_PERCENTILE,
            )
        )
    else:
        local_vmax = 1.0

    if (
        not np.isfinite(local_vmax)
        or local_vmax <= 0
    ):
        local_vmax = 1.0

    strength = np.clip(
        np.abs(domain_values) / local_vmax,
        0.0,
        1.0,
    )

    colors[:, 3] = (
        SPATIAL_DELTA_ALPHA_MIN
        + (
            SPATIAL_DELTA_ALPHA_MAX
            - SPATIAL_DELTA_ALPHA_MIN
        )
        * strength
    )

    ax.scatter(
        coords[domain_mask, 0],
        coords[domain_mask, 1],
        c=colors,
        s=point_size * 1.15,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    # ------------------------------------------------------------
    # Spatial formatting
    # ------------------------------------------------------------
    vis_coords = coords[
        vis_mask
    ]

    ax.set_xlim(
        float(np.min(vis_coords[:, 0])),
        float(np.max(vis_coords[:, 0])),
    )

    ax.set_ylim(
        float(np.max(vis_coords[:, 1])),
        float(np.min(vis_coords[:, 1])),
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_anchor("C")

    ax.set_title(
        f"{display_gene_name(gene)} | Spatial Δ in domain {domain}",
        fontsize=12.5,
        pad=6,
    )

    ax.axis("off")

def display_gene_name(gene: str) -> str:
    """Convert legacy gene symbols to preferred display names."""
    gene_aliases = {
        "2010300C02Rik": "Cracdl",
        "2010300C02RIK": "Cracdl",
    }
    return gene_aliases.get(str(gene), str(gene))


def plot_one_domain_detail(
    dfd: pd.DataFrame,
    metric_row: pd.Series,
    domain: str,
    ad_virt_delta,
    location_fraction_df: Optional[pd.DataFrame] = None,
    save_path=None,
    rank_mode: str = TOP_GENE_RANK_MODE,
    use_absolute: bool = TOP_GENE_RANK_USE_ABS,
):
    """
    Layout
    ------
    gs[0,0] : all-gene agreement scatter
    gs[1,0] : spatial delta map
    gs[:,1] : ranked real/virtual delta heatmap

    The spatial map and heatmap use one shared diverging color scale.
    Therefore only the heatmap colorbar is required.
    """

    set_style()

    # ============================================================
    # Gene ranking
    # ============================================================
    rank_mode = validate_top_gene_rank_mode(
        rank_mode
    )

    comparison_mode = (
        "real"
        if rank_mode == "virtual"
        else "virtual"
    )

    source_col = f"{rank_mode}_delta"
    comparison_col = f"{comparison_mode}_delta"

    source_display = rank_mode.capitalize()
    comparison_display = comparison_mode.capitalize()

    ranked_df, label_genes, heat_genes = (
        select_top_genes_for_domain(
            dfd=dfd,
            rank_mode=rank_mode,
            use_absolute=use_absolute,
            n_label=TOP_LABEL_GENES,
            n_heat=TOP_HEATMAP_GENES,
            location_fraction_df=location_fraction_df,
        )
    )

    if ranked_df.empty:
        return

    rep_gene = str(
        heat_genes[0]
        if heat_genes
        else ranked_df.iloc[0]["gene"]
    )

    rank_notation = (
        f"|{rank_mode} delta|"
        if use_absolute
        else f"signed {rank_mode} delta"
    )

    # ============================================================
    # Heatmap values
    #
    # Build this BEFORE plotting so we can calculate one common
    # color normalization for heatmap + spatial map.
    # ============================================================
    heat_columns = [
        source_col,
        comparison_col,
    ]

    heat_df = (
        ranked_df[
            ranked_df["gene"].isin(heat_genes)
        ]
        .set_index("gene")[heat_columns]
        .loc[heat_genes]
    )

    heat_df.columns = [
        f"{source_display} delta",
        f"{comparison_display} delta",
    ]

    # Display preferred gene symbols without changing the underlying ranking/data.
    heat_df.index = [
        display_gene_name(gene)
        for gene in heat_df.index
    ]

    # ============================================================
    # Selected-domain spatial values
    # ============================================================
    spatial_labels = get_domain_labels(
        ad_virt_delta,
        DOMAIN_KEY,
    )

    spatial_values_all = extract_delta_gene_vector(
        ad_virt_delta,
        rep_gene,
    )

    if (
        FILTER_VALID_FOR_VISUALIZATION
        and "_delta_common_valid" in ad_virt_delta.obs.columns
    ):
        spatial_valid = ad_virt_delta.obs[
            "_delta_common_valid"
        ].to_numpy(dtype=bool)
    else:
        spatial_valid = np.ones(
            ad_virt_delta.n_obs,
            dtype=bool,
        )

    spatial_mask = (
        spatial_valid
        & (spatial_labels == str(domain))
        & np.isfinite(spatial_values_all)
    )

    spatial_values = spatial_values_all[
        spatial_mask
    ]

    # ============================================================
    # ONE shared delta scale
    #
    # Heatmap:
    #     full max |Δ|
    #
    # Spatial:
    #     percentile-based max as in the original spatial map,
    #     preventing isolated spatial outliers from determining scale.
    # ============================================================
    heat_values = heat_df.to_numpy(
        dtype=float
    )

    finite_heat = heat_values[
        np.isfinite(heat_values)
    ]

    if finite_heat.size:
        heat_vmax = float(
            np.max(
                np.abs(finite_heat)
            )
        )
    else:
        heat_vmax = 0.0

    finite_spatial = spatial_values[
        np.isfinite(spatial_values)
    ]

    if finite_spatial.size:
        spatial_vmax = float(
            np.percentile(
                np.abs(finite_spatial),
                SPATIAL_DELTA_VMAX_PERCENTILE,
            )
        )
    else:
        spatial_vmax = 0.0

    vmax = max(
        heat_vmax,
        spatial_vmax,
        1e-6,
    )

    shared_norm = Normalize(
        vmin=-vmax,
        vmax=vmax,
        clip=True,
    )

    shared_cmap = "RdBu_r"

    # ============================================================
    # Main figure
    # ============================================================
    fig = plt.figure(
        figsize=(10.6, 6.4)
    )


    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[
            1.0,
            0.30,
        ],
        height_ratios=[
            1.0,
            1.0,
        ],
        left=0.075,
        right=0.955,
        bottom=0.075,
        top=0.93,
        wspace=-0.35,
        hspace=0.30,
    )

    # ============================================================
    # Upper-left: all-gene agreement
    # ============================================================
    ax1 = fig.add_subplot(
        gs[0, 0]
    )

    x = dfd[
        "real_delta"
    ].to_numpy(
        dtype=np.float32
    )

    y = dfd[
        "virtual_delta"
    ].to_numpy(
        dtype=np.float32
    )

    ax1.scatter(
        x,
        y,
        s=26,
        color="#ee8ca0",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.3,
    )

    lo, hi = compute_full_ylim(
        [x, y],
        pad_frac=0.10,
    )

    ax1.axhline(
        0,
        color="0.86",
        lw=0.8,
        zorder=0,
    )

    ax1.axvline(
        0,
        color="0.86",
        lw=0.8,
        zorder=0,
    )

    ax1.plot(
        [lo, hi],
        [lo, hi],
        ls="--",
        lw=1.1,
        color="0.45",
        zorder=1,
    )

    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)

    ax1.set_aspect(
        "equal",
        adjustable="box",
    )

    ax1.set_title(
        "All-gene delta agreement",
        fontsize=13,
        pad=7,
    )

    ax1.set_xlabel(
        "Real delta",
        fontsize=11.5,
    )

    ax1.set_ylabel(
        "Virtual delta",
        fontsize=11.5,
    )

    # ------------------------------------------------------------
    # Gene labels
    # ------------------------------------------------------------
    gene_lookup = (
        dfd
        .drop_duplicates("gene")
        .set_index("gene")
    )

    offset = (
        0.012
        * max(
            hi - lo,
            1e-8,
        )
    )

    for gene in label_genes:

        if gene not in gene_lookup.index:
            continue

        row = gene_lookup.loc[
            gene
        ]

        rx = float(
            row["real_delta"]
        )

        vy = float(
            row["virtual_delta"]
        )

        if np.isfinite(rx) and np.isfinite(vy):
            ax1.text(
                rx + offset,
                vy + offset,
                display_gene_name(gene),
                fontsize=10,
            )

    # ------------------------------------------------------------
    # Metric annotation
    # ------------------------------------------------------------
    txt = (
        f"Pearson = {metric_row['pearson_r']:.2f}\n"
        f"Spearman = {metric_row['spearman_r']:.2f}\n"
        f"Cosine = {metric_row['cosine']:.2f}\n"
        f"Sign acc = {metric_row['sign_accuracy']:.2f}"
    )

    ax1.text(
        0.04,
        0.96,
        txt,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            alpha=0.88,
            edgecolor="0.8",
        ),
    )

    # ============================================================
    # Lower-left: spatial map
    #
    # No nested GridSpec and no spatial colorbar are needed anymore.
    # ============================================================
    ax2 = fig.add_subplot(
        gs[1, 0]
    )

    plot_virtual_domain_delta_map(
        ax=ax2,
        ad_virtual_delta=ad_virt_delta,
        domain=domain,
        gene=rep_gene,
        norm=shared_norm,
        cmap_name=shared_cmap,
    )

    # ============================================================
    # Right: full-height heatmap
    # ============================================================
    ax3 = fig.add_subplot(
        gs[:, 1]
    )

    sns.heatmap(
        heat_df,
        ax=ax3,
        cmap=shared_cmap,
        norm=shared_norm,
        linewidths=0.4,
        linecolor="white",

        # This is now the SINGLE colorbar shared conceptually by
        # both the heatmap and spatial delta map.
        cbar=True,
        cbar_kws={
            "label": "Delta",
            "fraction": 0.055,
            "pad": 0.04,
        },
    )

    ax3.set_title(
        f"Top genes ranked by {rank_notation}",
        fontsize=11.5,
        pad=7,
    )

    ax3.set_xlabel("")
    ax3.set_ylabel("")

    ax3.tick_params(
        axis="y",
        labelsize=12,
        pad=3,
    )

    ax3.tick_params(
        axis="x",
        labelsize=12,
        rotation=0,
    )

    # ------------------------------------------------------------
    # Highlight gene shown in spatial panel
    # ------------------------------------------------------------
    rep_gene_display = display_gene_name(rep_gene)

    if rep_gene_display in heat_df.index:
        rep_idx = list(heat_df.index).index(rep_gene_display)

        ax3.add_patch(
            plt.Rectangle(
                (0, rep_idx),
                len(heat_df.columns),
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.6,
                clip_on=False,
            )
        )
    # plt.show()


    save_or_show(
        fig,
        save_path,
    )










def plot_all_domain_details(
    delta_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    ad_virt_delta,
    location_fraction_df: Optional[pd.DataFrame],
    save_dir: str,
    rank_mode: str = TOP_GENE_RANK_MODE,
    use_absolute: bool = TOP_GENE_RANK_USE_ABS,
):
    """Generate streamlined per-domain detail plots."""
    ensure_outdir(save_dir)
    rank_mode = validate_top_gene_rank_mode(rank_mode)

    for dom in metric_df["domain"].tolist():
        dfd = delta_df[delta_df["domain"] == dom].copy()
        mrow = metric_df[metric_df["domain"] == dom]
        if len(dfd) == 0 or len(mrow) == 0:
            continue

        plot_one_domain_detail(
            dfd=dfd,
            metric_row=mrow.iloc[0],
            domain=dom,
            ad_virt_delta=ad_virt_delta,
            location_fraction_df=location_fraction_df,
            save_path=os.path.join(
                save_dir,
                f"domain_{dom}_detail_{rank_mode}_ranked.png",
            ),
            rank_mode=rank_mode,
            use_absolute=use_absolute,
        )

def load_coords_npz(net_in: str, group_id: int, sample_id: int) -> np.ndarray:
    f = os.path.join(net_in, f"{group_id}_{sample_id}_validation.npz")
    d = np.load(f)
    return d["coord"].astype(np.float32)

def transform_loaded_coords(coords: np.ndarray) -> np.ndarray:
    """
    Convert raw coordinates into the final plotting orientation once at load.

    Processing order
    ----------------
    1. Keep the first two coordinate columns.
    2. Swap x and y when SWAP_XY is True.
    3. Reflect x within FIXED_BOX when INVERT_X is True.
    4. Reflect y within FIXED_BOX when INVERT_Y is True.

    With the current settings:
        SWAP_XY = True
        INVERT_X = True
        INVERT_Y = False
    """
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"Expected coordinates with shape (n, 2+), got {coords.shape}"
        )

    coords = coords[:, :2].copy()

    coords = coords[:, [1, 0]]

    xmin, xmax, ymin, ymax =(0.0, 1024.0, 0.0, 1024.0)


    coords[:, 0] = xmin + xmax - coords[:, 0]

    coords[:, 1] = ymin + ymax - coords[:, 1]

    return coords

# ============================================================
# EVALUATION / PLOTTING WORKFLOWS
# ============================================================

def normalize_loaded_result_types(
    delta_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    marker_location_fraction_df: pd.DataFrame,
):
    """
    Restore identifier dtypes after CSV loading.

    Pandas may infer numeric-looking domain labels such as "0" or "11" as
    integers. Plotting/ranking code consistently treats domain IDs as strings,
    so normalize them immediately in plot-only mode.
    """
    delta_df = delta_df.copy()
    metric_df = metric_df.copy()
    marker_location_fraction_df = marker_location_fraction_df.copy()

    if "domain" in delta_df.columns:
        delta_df["domain"] = delta_df["domain"].astype(str)
    if "gene" in delta_df.columns:
        delta_df["gene"] = delta_df["gene"].astype(str)

    if "domain" in metric_df.columns:
        metric_df["domain"] = metric_df["domain"].astype(str)

    if "domain" in marker_location_fraction_df.columns:
        marker_location_fraction_df["domain"] = (
            marker_location_fraction_df["domain"].astype(str)
        )
    if "gene" in marker_location_fraction_df.columns:
        marker_location_fraction_df["gene"] = (
            marker_location_fraction_df["gene"].astype(str)
        )

    return delta_df, metric_df, marker_location_fraction_df


def evaluate_and_save():
    """
    Run the expensive evaluation only and save everything needed by plot mode.

    In addition to the existing CSV tables, save the canonical virtual spatial
    delta AnnData because the per-domain detail figures require its location-
    level delta values and coordinates.
    """
    ensure_outdir(OUTDIR)

    ad_real_n = read_h5ad(REAL_NORMAL_PATH)
    ad_real_d = read_h5ad(REAL_DISEASE_PATH)

    real_n_coor = load_coords_npz(
        "/media/huifang/data/registration/xenium/mouse_brain/network_input/",
        0,
        4,
    )
    real_d_coor = load_coords_npz(
        "/media/huifang/data/registration/xenium/mouse_brain/network_input/",
        0,
        1,
    )
    ad_real_n.obsm["spatial"] = transform_loaded_coords(real_n_coor)
    ad_real_d.obsm["spatial"] = transform_loaded_coords(real_d_coor)

    ad_virt_n = read_h5ad(VIRTUAL_NORMAL_PATH)
    ad_virt_d = read_h5ad(VIRTUAL_DISEASE_PATH)
    ad_virt_n.obsm["spatial"] = transform_loaded_coords(
        ad_virt_n.obsm["spatial"]
    )
    ad_virt_d.obsm["spatial"] = transform_loaded_coords(
        ad_virt_d.obsm["spatial"]
    )

    ads = {
        "real_normal": ad_real_n,
        "real_disease": ad_real_d,
        "virtual_normal": ad_virt_n,
        "virtual_disease": ad_virt_d,
    }

    domains = get_selected_domains(
        [ad_real_n, ad_real_d, ad_virt_n, ad_virt_d]
    )
    genes = select_genes_for_metrics(
        ad_real_n,
        ad_real_d,
        ad_virt_n,
        ad_virt_d,
    )

    print(f"[INFO] domains: {domains}")
    print(f"[INFO] selected genes for metrics: {len(genes)}")

    validate_extraction_consistency(
        ads,
        genes,
    )

    # --------------------------------------------------------
    # Canonical location-wise virtual spatial delta.
    # --------------------------------------------------------
    virtual_delta_result = compute_spatial_delta(
        disease=ad_virt_d,
        normal=ad_virt_n,
        normalize_total=LIBRARY_SIZE_NORMALIZE,
        target_sum=NORMALIZATION_TARGET_SUM,
        use_log1p=LOG1P_BEFORE_ANALYSIS,
        valid_key=VALID_KEY,
        spatial_key="spatial",
        domain_key=DOMAIN_KEY,
        genes=genes,
        use_common_valid=SPATIAL_DELTA_USE_COMMON_VALID,
        coordinate_decimals=5,
        return_result=True,
    )
    ad_virt_delta = virtual_delta_result.adata

    if (
        FILTER_VALID_FOR_VISUALIZATION
        and "_delta_common_valid" in ad_virt_delta.obs.columns
    ):
        n_visual = int(
            ad_virt_delta.obs["_delta_common_valid"].sum()
        )
    else:
        n_visual = int(
            ad_virt_delta.n_obs
        )

    print(
        "[INFO] Preprocessing: "
        f"normalize_total={LIBRARY_SIZE_NORMALIZE}, "
        f"target_sum={NORMALIZATION_TARGET_SUM:g}, "
        f"log1p={LOG1P_BEFORE_ANALYSIS}"
    )
    print(
        "[INFO] Spatial delta definition: "
        "preprocess(disease) - preprocess(normal)"
    )
    print(
        "[INFO] Virtual matched locations: "
        f"{virtual_delta_result.matched_count}; "
        f"delta locations retained={virtual_delta_result.common_valid_count}; "
        f"visualized={n_visual}"
    )

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

    delta_df = delta_df[
        delta_df["domain"].isin(keep_domains)
    ].copy()

    marker_location_fraction_df = (
        build_virtual_marker_location_fraction_table(
            ad_virtual_delta=ad_virt_delta,
            domains=keep_domains,
            genes=genes,
            use_common_valid=MARKER_FRACTION_USE_COMMON_VALID,
            batch_size=MARKER_FRACTION_BATCH_SIZE,
            domain_key=DOMAIN_KEY,
        )
    )

    metric_df = compute_domain_metrics(
        delta_df
    )
    metric_long_df = build_metric_long_df(
        metric_df
    )

    ranking_df = build_domain_gene_ranking_table(
        delta_df=delta_df,
        rank_mode=TOP_GENE_RANK_MODE,
        use_absolute=TOP_GENE_RANK_USE_ABS,
        location_fraction_df=marker_location_fraction_df,
    )

    # --------------------------------------------------------
    # Save all pre-calculated results used by plot-only mode.
    # --------------------------------------------------------
    delta_df.to_csv(
        os.path.join(
            OUTDIR,
            "domain_gene_delta_table.csv",
        ),
        index=False,
    )

    metric_df.to_csv(
        os.path.join(
            OUTDIR,
            "domain_delta_metrics.csv",
        ),
        index=False,
    )

    metric_long_df.to_csv(
        os.path.join(
            OUTDIR,
            "domain_metric_long_for_boxplot.csv",
        ),
        index=False,
    )

    marker_location_fraction_df.to_csv(
        os.path.join(
            OUTDIR,
            "virtual_marker_location_fraction.csv",
        ),
        index=False,
    )

    ranking_df.to_csv(
        os.path.join(
            OUTDIR,
            f"domain_gene_ranking_{TOP_GENE_RANK_MODE}.csv",
        ),
        index=False,
    )

    # Needed only for spatial maps in the detailed per-domain figures.
    ad_virt_delta.write_h5ad(
        VIRTUAL_SPATIAL_DELTA_PATH
    )

    print(
        f"[SAVE] {VIRTUAL_SPATIAL_DELTA_PATH}"
    )
    print(
        f"[DONE] Evaluation results saved to: {OUTDIR}"
    )


def load_precalculated_results():
    """
    Load only pre-calculated evaluation results.

    No raw real/virtual tissue is read and no delta/metric calculation is
    repeated in plot mode.
    """
    paths = {
        "delta": os.path.join(
            OUTDIR,
            "domain_gene_delta_table.csv",
        ),
        "metric": os.path.join(
            OUTDIR,
            "domain_delta_metrics.csv",
        ),
        "marker_fraction": os.path.join(
            OUTDIR,
            "virtual_marker_location_fraction.csv",
        ),
        "spatial_delta": VIRTUAL_SPATIAL_DELTA_PATH,
    }

    missing = [
        path
        for path in paths.values()
        if not os.path.isfile(path)
    ]

    if missing:
        raise FileNotFoundError(
            "Missing pre-calculated result(s). "
            "Run with --mode eval first:\n  "
            + "\n  ".join(missing)
        )

    delta_df = pd.read_csv(
        paths["delta"]
    )
    metric_df = pd.read_csv(
        paths["metric"]
    )
    marker_location_fraction_df = pd.read_csv(
        paths["marker_fraction"]
    )

    (
        delta_df,
        metric_df,
        marker_location_fraction_df,
    ) = normalize_loaded_result_types(
        delta_df,
        metric_df,
        marker_location_fraction_df,
    )

    # This is a pre-computed delta object, not a raw input tissue.
    ad_virt_delta = read_h5ad(
        paths["spatial_delta"]
    )

    return (
        delta_df,
        metric_df,
        marker_location_fraction_df,
        ad_virt_delta,
    )


def plot_precalculated_results():
    """
    Plot from saved evaluation results only.
    """
    figdir = os.path.join(
        OUTDIR,
        "figures",
    )
    ensure_outdir(
        figdir
    )

    per_domain_dir = os.path.join(
        figdir,
        "per_domain",
    )
    ensure_outdir(
        per_domain_dir
    )

    (
        delta_df,
        metric_df,
        marker_location_fraction_df,
        ad_virt_delta,
    ) = load_precalculated_results()

    genes = sorted(
        delta_df["gene"].astype(str).unique().tolist()
    )

    plot_genes = select_plot_genes(
        delta_df,
        genes,
    )

    domain_order = (
        metric_df["domain"]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # Summary plots
    # --------------------------------------------------------
    plot_metric_boxplot_all(
        metric_df,
        save_path=os.path.join(
            figdir,
            "domain_metric_boxplot_all.png",
        ),
    )
    #
    # plot_selected_gene_domain_dotplot(
    #     delta_df=delta_df,
    #     plot_genes=plot_genes,
    #     domain_order=domain_order,
    #     save_path=os.path.join(
    #         figdir,
    #         "selected_gene_domain_heatmap.png",
    #     ),
    # )

    # --------------------------------------------------------
    # Per-domain detail plots
    # --------------------------------------------------------
    plot_all_domain_details(
        delta_df,
        metric_df,
        ad_virt_delta=ad_virt_delta,
        location_fraction_df=marker_location_fraction_df,
        save_dir=per_domain_dir,
        rank_mode=TOP_GENE_RANK_MODE,
        use_absolute=TOP_GENE_RANK_USE_ABS,
    )

    print(
        f"[DONE] Figures saved to: {figdir}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("eval", "plot", "all"),
        default="plot",
        help=(
            "eval: calculate and save numerical results only; "
            "plot: load saved results and plot only; "
            "all: run eval then plot"
        ),
    )
    args = parser.parse_args()

    if args.mode in {"eval", "all"}:
        evaluate_and_save()

    if args.mode in {"plot", "all"}:
        plot_precalculated_results()


if __name__ == "__main__":
    main()
