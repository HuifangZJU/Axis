#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLPFC virtual-gene generation evaluation against measured depth anchors.

The analysis evaluates marker genes and selected highly expressed/spatially
variable genes using shift-tolerant transition-map NCC to the measured
10-µm and 310-µm anchor sections. Marker genes are evaluated across the
virtual depth series; top genes are evaluated at the configured case depth.
Plotting includes a publication-style case-depth summary and marker-gene
NCC/shift trajectories. Spatial case-study maps are intentionally omitted.
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from anndata import AnnData
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIG
# ============================================================
ROOT = "/media/huifang/data/registration/DLPFC/huifang"
H5AD_DIR = os.path.join(ROOT, "h5ad")
VIRTUAL_DIR = os.path.join(ROOT, "virtual_tissue")
OUTDIR = "//media/huifang/data/registration/paper_analysis/Figure4_dlfpc/fg_gene_metrics/"
os.makedirs(OUTDIR, exist_ok=True)

COORD_KEY = "spatial"
NORMALIZE_TOTAL = True
TARGET_SUM = 1e4
LOG1P = True

# Load all real slices if present.  Real 10 and 310 are the interpolation anchors.
REAL_SLICE_PATHS = {
    0: os.path.join(H5AD_DIR, "1_0.h5ad"),
    10: os.path.join(H5AD_DIR, "1_1.h5ad"),
    310: os.path.join(H5AD_DIR, "1_2.h5ad"),
    320: os.path.join(H5AD_DIR, "1_3.h5ad"),
}
ANCHOR_Z_LOW = 10
ANCHOR_Z_HIGH = 310

VIRTUAL_DEPTHS = np.arange(20, 310, 10)
CASE_VIRTUAL_Z = 160

SPATIALZ_SLICE_PATHS = {
    depth: os.path.join(
        VIRTUAL_DIR,
        "spatialz",
        "dense_reconstruction",
        "group1",
        f"spatialz_{depth}.000.h5ad",
    )
    for depth in VIRTUAL_DEPTHS
}

AXIS_SLICE_PATHS = {
    depth: os.path.join(
        VIRTUAL_DIR,
        "dlpfc_z_group1_grid80_spatial1",
        "query_full_gene_set",
        f"virtual_dlpfc_z_{depth}.000.h5ad",
    )
    for depth in VIRTUAL_DEPTHS
}

# MARKER_GENES = [
#     "MOBP", "CUX2", "RORB", "ENC1", "PCP4", "KCNIP2",
#     "FOXP2", "MBP", "MFGE8", "SCGB2A2", "GFAP",
# ]
MARKER_GENES = [ "PCP4"]
TOP_N_HVG_GENES = 100
TOP_HVG_PREFILTER_N = 2000   # first-pass highly variable genes before expression/spatial-variance ranking
EXCLUDE_MARKER_FROM_TOP = True

METHOD_ORDER = ["AXIS", "SpatialZ"]
PLOT_METHOD_ORDER = ["Real", "AXIS", "SpatialZ"]
METHOD_PALETTE = {
    "SpatialZ": "#7C8DA6",
    "AXIS": "#9E2F2A",
}
PLOT_PALETTE = {
    "Real": "#111111",
    **METHOD_PALETTE,
}

# Gene-selection QC on Real 10 and 310.
TOP_QC_MIN_DETECTED_SPOTS = 30
TOP_QC_MIN_DETECTED_FRAC = 0.03
TOP_QC_MIN_MEAN_LOG1P = 0.02

# Rasterization / gradient.
FIELD_GRID_BINS = 80
REGION_TREND_SIGMA_GRID = 3.2
VECTOR_FIELD_SMOOTH_SIGMA_GRID = 1.3
VALID_WEIGHT_EPS = 1e-4
EPS = 1e-8

# Shift-tolerant transition-map NCC metric.
TRANSITION_NCC_MAX_SHIFT_Y_BINS = 20
TRANSITION_NCC_MAX_SHIFT_X_BINS = 10
MIN_MAP_OVERLAP_PIXELS = 80

FIG_DPI = 350

RUN_MODE = "plot"  # "eval", "plot", or "both"

# Outputs.
GENE_SELECTION_CSV = os.path.join(OUTDIR, "hvg_top_gene_selection.csv")
METRICS_CSV = os.path.join(OUTDIR, "dlpfc_transition_ncc_metrics_long.csv")
SUMMARY_CSV = os.path.join(OUTDIR, "dlpfc_transition_ncc_summary.csv")
FIELD_CACHE_PATH = os.path.join(OUTDIR, "dlpfc_transition_ncc_field_cache.pkl")
SUMMARY_FIG = os.path.join(OUTDIR, "dlpfc_transition_ncc_marker_topgene_summary.png")
MARKER_TRAJECTORY_DIR = os.path.join(OUTDIR, "marker_transition_trajectories")



# ============================================================
# PLOTTING STYLE
# ============================================================
@dataclass(frozen=True)
class LegendStyle:
    """Reusable legend positioning and typography controls."""

    loc: str = "right"
    bbox_to_anchor: Optional[Tuple[float, float]] = (0.55, 0.925)
    ncol: int = 1
    fontsize: float = 11.0
    frameon: bool = False
    handletextpad: float = 0.55
    columnspacing: float = 1.25
    labelspacing: float = 0.35
    borderaxespad: float = 0.0


@dataclass(frozen=True)
class SummaryStyle:
    """SCC-matched layout controls for the DLPFC NCC summary."""

    figsize: Tuple[float, float] = (3.8, 7)
    dpi: int = 350

    panel_title_fontsize: float = 13.0
    panel_title_weight: str = "normal"
    panel_title_pad: float =14.0
    axis_label_fontsize: float = 12.0
    tick_fontsize: float = 11.0

    box_width: float = 0.28
    gene_set_offset: float = 0.17
    marker_box_alpha: float = 0.70
    box_edge_width: float = 0.95
    median_width: float = 1.15
    whisker_width: float = 0.85

    point_size: float = 10.5
    marker_point_alpha: float = 0.45
    top_point_alpha: float = 0.58
    point_jitter: float = 0.055

    grid_alpha: float = 0.18
    left: float = 0.18
    right: float = 0.98
    bottom: float = 0.18
    top: float = 0.96
    hspace: float = 0.30

    y_limits: Tuple[float, float] = (0.2, 1.0)
    legend: LegendStyle = field(default_factory=LegendStyle)


SUMMARY_STYLE = SummaryStyle()


def set_publication_style() -> None:
    """Cell/Nature-style plotting defaults shared across summary and trajectories."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.titleweight": "normal",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# DATA HELPERS
# ============================================================
def load_slice_dict(path_dict: Dict[int, str], coord_key: str = COORD_KEY, flip_y: bool = True) -> Dict[int, AnnData]:
    out = {}
    for z, p in path_dict.items():
        if p is None or str(p).strip() == "":
            continue
        if not os.path.exists(p):
            print(f"[WARN] missing file for z={z}: {p}")
            continue
        ad = sc.read_h5ad(p)
        if "valid" in ad.obs.columns:
            ad = ad[ad.obs["valid"]].copy()
        if flip_y:
            if coord_key not in ad.obsm:
                raise KeyError(f"{coord_key!r} not found in ad.obsm for z={z}")
            coords = np.asarray(ad.obsm[coord_key], dtype=np.float32).copy()
            coords[:, 1] = -coords[:, 1]
            ad.obsm[coord_key] = coords
        out[int(z)] = ad
    return out


def preprocess_adata(adata: AnnData) -> AnnData:
    ad = adata.copy()
    if NORMALIZE_TOTAL:
        sc.pp.normalize_total(ad, target_sum=TARGET_SUM)
    if LOG1P:
        sc.pp.log1p(ad)
    return ad


def get_coords(adata: AnnData, coord_key: str = COORD_KEY) -> np.ndarray:
    if coord_key not in adata.obsm:
        raise KeyError(f"{coord_key!r} not found in adata.obsm")
    return np.asarray(adata.obsm[coord_key], dtype=np.float64)


def extract_gene_vector(adata: AnnData, gene: str) -> np.ndarray:
    if gene not in adata.var_names:
        raise KeyError(f"Gene {gene!r} not found in var_names")
    x = adata[:, [gene]].X
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x).reshape(-1).astype(np.float64)


def clean_coords_values(coords: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).ravel()
    keep = np.all(np.isfinite(coords), axis=1) & np.isfinite(x)
    return coords[keep], x[keep]


def normalized_xy(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    mn = np.nanmin(coords, axis=0)
    mx = np.nanmax(coords, axis=0)
    rng = np.maximum(mx - mn, EPS)
    return (coords - mn[None, :]) / rng[None, :]


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.full_like(x, np.nan, dtype=np.float64)
    keep = np.isfinite(x)
    if np.sum(keep) == 0:
        return out
    mu = float(np.mean(x[keep]))
    sd = float(np.std(x[keep]))
    if sd <= EPS:
        out[keep] = 0.0
    else:
        out[keep] = (x[keep] - mu) / sd
    return out


def safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    if np.sum(keep) < 3:
        return np.nan
    if np.nanstd(x[keep]) <= EPS or np.nanstd(y[keep]) <= EPS:
        return np.nan
    return float(pearsonr(x[keep], y[keep])[0])


def get_nearest_available_z(ad_dict: Dict[int, object], target_z: int) -> Optional[int]:
    if not ad_dict:
        return None
    zs = np.array(sorted(ad_dict.keys()), dtype=int)
    return int(zs[np.argmin(np.abs(zs - target_z))])


def common_genes_across(adatas: Sequence[AnnData]) -> List[str]:
    if len(adatas) == 0:
        return []
    common = set(adatas[0].var_names.astype(str))
    for ad in adatas[1:]:
        common &= set(ad.var_names.astype(str))
    return sorted(common)


# ============================================================
# TRANSITION-FIELD COMPUTATION
# ============================================================
def rasterize_points_to_grid(
    coords: np.ndarray,
    values: np.ndarray,
    bins: int = FIELD_GRID_BINS,
) -> Dict[str, np.ndarray]:
    """Rasterize spot values to a normalized 2D grid."""
    coords, values = clean_coords_values(coords, values)
    coords_n = normalized_xy(coords)
    x = np.clip((coords_n[:, 0] * bins).astype(int), 0, bins - 1)
    y = np.clip((coords_n[:, 1] * bins).astype(int), 0, bins - 1)

    num = np.zeros((bins, bins), dtype=np.float64)
    den = np.zeros((bins, bins), dtype=np.float64)
    for xi, yi, value in zip(x, y, values):
        num[yi, xi] += float(value)
        den[yi, xi] += 1.0

    image = np.divide(
        num,
        np.maximum(den, 1.0),
        out=np.full_like(num, np.nan),
        where=den > 0,
    )
    return {"image": image, "valid": den > 0}


def masked_gaussian_smooth(
    image: np.ndarray,
    valid: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian smooth while respecting missing grid cells."""
    image = np.asarray(image, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(image)
    numerator = gaussian_filter(
        np.where(valid, image, 0.0),
        sigma=float(sigma),
        mode="nearest",
    )
    denominator = gaussian_filter(
        valid.astype(np.float64),
        sigma=float(sigma),
        mode="nearest",
    )
    smoothed = np.divide(
        numerator,
        np.maximum(denominator, EPS),
        out=np.full_like(numerator, np.nan),
        where=denominator > VALID_WEIGHT_EPS,
    )
    return smoothed, denominator > VALID_WEIGHT_EPS


def compute_gene_field(adata: AnnData, gene: str) -> Dict[str, object]:
    """Compute only the transition-field quantities required by NCC evaluation."""
    coords, expression = clean_coords_values(
        get_coords(adata),
        extract_gene_vector(adata, gene),
    )
    z_expression = zscore(expression)
    z_grid = rasterize_points_to_grid(
        coords,
        z_expression,
        bins=FIELD_GRID_BINS,
    )

    z_trend, valid_trend = masked_gaussian_smooth(
        z_grid["image"],
        z_grid["valid"],
        sigma=REGION_TREND_SIGMA_GRID,
    )
    z_fill = np.where(np.isfinite(z_trend), z_trend, 0.0)
    gy_raw, gx_raw = np.gradient(z_fill)

    gx_smooth, valid_gx = masked_gaussian_smooth(
        gx_raw,
        valid_trend,
        sigma=VECTOR_FIELD_SMOOTH_SIGMA_GRID,
    )
    gy_smooth, valid_gy = masked_gaussian_smooth(
        gy_raw,
        valid_trend,
        sigma=VECTOR_FIELD_SMOOTH_SIGMA_GRID,
    )
    valid_eval = valid_trend & valid_gx & valid_gy

    gx_smooth = np.where(valid_eval, gx_smooth, np.nan)
    gy_smooth = np.where(valid_eval, gy_smooth, np.nan)
    grad_mag = np.sqrt(gx_smooth**2 + gy_smooth**2)

    return {
        "gene": gene,
        "grad_mag": grad_mag,
        "valid_eval": valid_eval,
    }


# ============================================================
# GENE SELECTION
# ============================================================
def matrix_gene_stats(adata: AnnData, genes: Sequence[str]) -> pd.DataFrame:
    """Fast per-gene expression statistics for a fixed gene order.

    The expensive laminar R2 calculation is done only after this vectorized
    high-variance prefilter. Works with sparse or dense AnnData.X.
    """
    genes = [str(g) for g in genes if str(g) in adata.var_names]
    if len(genes) == 0:
        return pd.DataFrame(columns=["gene", "mean_expr", "var_expr", "detected_spots", "detected_frac"])
    X = adata[:, genes].X
    n = max(int(adata.n_obs), 1)
    if sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel().astype(np.float64)
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel().astype(np.float64)
        detected = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.int64)
    else:
        X = np.asarray(X, dtype=np.float64)
        mean = np.nanmean(X, axis=0)
        mean_sq = np.nanmean(X * X, axis=0)
        detected = np.sum(np.isfinite(X) & (X > 0), axis=0).astype(np.int64)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return pd.DataFrame({
        "gene": genes,
        "mean_expr": mean,
        "var_expr": var,
        "detected_spots": detected,
        "detected_frac": detected.astype(np.float64) / float(n),
    })


def hvg_prefilter_table(real10: AnnData, real310: AnnData, common_genes: Sequence[str]) -> pd.DataFrame:
    """Rank genes by a cheap HVG-like score before laminar R2 calculation."""
    genes = [str(g) for g in common_genes if str(g) in real10.var_names and str(g) in real310.var_names]
    marker_set = set(MARKER_GENES) if EXCLUDE_MARKER_FROM_TOP else set()
    genes = [g for g in genes if g not in marker_set]
    if len(genes) == 0:
        return pd.DataFrame()

    s10 = matrix_gene_stats(real10, genes).rename(columns={
        "mean_expr": "mean_expr_10",
        "var_expr": "var_expr_10",
        "detected_spots": "detected_spots_10",
        "detected_frac": "detected_frac_10",
    })
    s310 = matrix_gene_stats(real310, genes).rename(columns={
        "mean_expr": "mean_expr_310",
        "var_expr": "var_expr_310",
        "detected_spots": "detected_spots_310",
        "detected_frac": "detected_frac_310",
    })
    df = s10.merge(s310, on="gene", how="inner")
    if df.empty:
        return df

    # Variance values are already on the same log-normalized scale. Use a rank
    # average to make the score robust to slice-specific variance ranges.
    df["hvg_rank_score"] = 0.5 * (
        df["var_expr_10"].rank(pct=True, method="average") +
        df["var_expr_310"].rank(pct=True, method="average")
    )
    df["hvg_mean_var"] = 0.5 * (df["var_expr_10"] + df["var_expr_310"])
    df = df.sort_values(["hvg_rank_score", "hvg_mean_var"], ascending=[False, False]).reset_index(drop=True)
    df["hvg_prefilter_rank"] = np.arange(1, len(df) + 1, dtype=int)
    df["in_hvg_prefilter"] = df["hvg_prefilter_rank"] <= int(TOP_HVG_PREFILTER_N)
    return df

def select_top_hvg_genes(real_ads: Dict[int, AnnData], common_genes: Sequence[str]) -> Tuple[List[str], pd.DataFrame]:
    """Select simple top genes by expression support + spatial variability.

    Selection is intentionally simple and fast:
      1) compute vectorized expression / variance statistics in Real 10 and Real 310
      2) keep genes passing expression/support QC
      3) rank by a combined score of log1p expression abundance and spatial variability

    No laminar R2 or gradient field is computed during gene selection.
    """
    if ANCHOR_Z_LOW not in real_ads or ANCHOR_Z_HIGH not in real_ads:
        raise RuntimeError(f"Real anchor slices {ANCHOR_Z_LOW} and {ANCHOR_Z_HIGH} are required for top-gene selection.")
    real10 = real_ads[ANCHOR_Z_LOW]
    real310 = real_ads[ANCHOR_Z_HIGH]

    df = hvg_prefilter_table(real10, real310, common_genes)
    if df.empty:
        return [], df

    df["pass_expression_qc"] = (
        (df["detected_spots_10"] >= TOP_QC_MIN_DETECTED_SPOTS) &
        (df["detected_spots_310"] >= TOP_QC_MIN_DETECTED_SPOTS) &
        (df["detected_frac_10"] >= TOP_QC_MIN_DETECTED_FRAC) &
        (df["detected_frac_310"] >= TOP_QC_MIN_DETECTED_FRAC) &
        (np.minimum(df["mean_expr_10"], df["mean_expr_310"]) >= TOP_QC_MIN_MEAN_LOG1P)
    )
    df["mean_expr_anchor"] = 0.5 * (df["mean_expr_10"] + df["mean_expr_310"])
    df["var_expr_anchor"] = 0.5 * (df["var_expr_10"] + df["var_expr_310"])
    df["expr_rank_score"] = df["mean_expr_anchor"].rank(pct=True, method="average")
    df["spatial_var_rank_score"] = df["hvg_rank_score"]
    # simple, intuitive combined score: high expression + spatial variability
    df["top_gene_score"] = 0.40 * df["expr_rank_score"] + 0.60 * df["spatial_var_rank_score"]

    sub = df[df["in_hvg_prefilter"] & df["pass_expression_qc"]].copy()
    sub = sub.sort_values(["top_gene_score", "spatial_var_rank_score", "mean_expr_anchor"],
                          ascending=[False, False, False])
    top = sub["gene"].astype(str).head(TOP_N_HVG_GENES).tolist()

    df["gene_set"] = np.where(df["gene"].isin(top), "top100_hvg", "other")
    print(f"[GENE SELECT] common non-marker genes: {len(df)}")
    print(f"[GENE SELECT] HVG prefilter: {int(df['in_hvg_prefilter'].sum())}")
    print(f"[GENE SELECT] HVG + expression QC: {len(sub)}")
    print(f"[GENE SELECT] selected top HVG/spatial genes: {len(top)}")
    return top, df


# ============================================================
# METRICS
# ============================================================
def shifted_map_ncc_vertical(a: np.ndarray, b: np.ndarray,
                             max_shift_y: int = TRANSITION_NCC_MAX_SHIFT_Y_BINS,
                             max_shift_x: int = TRANSITION_NCC_MAX_SHIFT_X_BINS,
                             min_overlap: int = MIN_MAP_OVERLAP_PIXELS) -> Tuple[float, int, int]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    h, w = a.shape
    best = -np.inf
    best_dy, best_dx = 0, 0
    for dy in range(-int(max_shift_y), int(max_shift_y) + 1):
        for dx in range(-int(max_shift_x), int(max_shift_x) + 1):
            y0a, y1a = max(0, -dy), min(h, h - dy)
            y0b, y1b = max(0, dy), min(h, h + dy)
            x0a, x1a = max(0, -dx), min(w, w - dx)
            x0b, x1b = max(0, dx), min(w, w + dx)
            if y1a <= y0a or y1b <= y0b or x1a <= x0a or x1b <= x0b:
                continue
            aa = a[y0a:y1a, x0a:x1a].ravel()
            bb = b[y0b:y1b, x0b:x1b].ravel()
            keep = np.isfinite(aa) & np.isfinite(bb)
            if np.sum(keep) < min_overlap:
                continue
            r = safe_pearson(aa[keep], bb[keep])
            if np.isfinite(r) and r > best:
                best = r
                best_dy, best_dx = dy, dx
    return (float(best) if np.isfinite(best) else np.nan), int(best_dy), int(best_dx)


def evaluate_candidate_against_real_refs(gene: str, method: str, z: int,
                                     cand: Dict[str, object],
                                     real10: Dict[str, object],
                                     real310: Dict[str, object]) -> Dict[str, object]:
    """Compute shift-tolerant transition-map NCC to the two measured anchors."""
    ncc10, dy10, dx10 = shifted_map_ncc_vertical(
        real10["grad_mag"], cand["grad_mag"],
        max_shift_y=TRANSITION_NCC_MAX_SHIFT_Y_BINS,
        max_shift_x=TRANSITION_NCC_MAX_SHIFT_X_BINS,
        min_overlap=MIN_MAP_OVERLAP_PIXELS,
    )
    ncc310, dy310, dx310 = shifted_map_ncc_vertical(
        real310["grad_mag"], cand["grad_mag"],
        max_shift_y=TRANSITION_NCC_MAX_SHIFT_Y_BINS,
        max_shift_x=TRANSITION_NCC_MAX_SHIFT_X_BINS,
        min_overlap=MIN_MAP_OVERLAP_PIXELS,
    )
    return {
        "gene": gene,
        "method": method,
        "z": int(z),
        "transition_ncc_to_real10": float(ncc10) if np.isfinite(ncc10) else np.nan,
        "transition_best_shift_y_to_real10": int(dy10),
        "transition_best_shift_x_to_real10": int(dx10),
        "transition_ncc_to_real310": float(ncc310) if np.isfinite(ncc310) else np.nan,
        "transition_best_shift_y_to_real310": int(dy310),
        "transition_best_shift_x_to_real310": int(dx310),
        "n_valid_gradient_pixels": int(np.sum(cand["valid_eval"])),
    }


# ============================================================
# EVALUATION / CACHE
# ============================================================
def load_and_preprocess_all():
    real_ads = {z: preprocess_adata(ad) for z, ad in load_slice_dict(REAL_SLICE_PATHS).items()}
    spatialz_ads = {z: preprocess_adata(ad) for z, ad in load_slice_dict(SPATIALZ_SLICE_PATHS).items()}
    axis_ads = {z: preprocess_adata(ad) for z, ad in load_slice_dict(AXIS_SLICE_PATHS).items()}
    if ANCHOR_Z_LOW not in real_ads or ANCHOR_Z_HIGH not in real_ads:
        raise RuntimeError(f"Real anchor slices {ANCHOR_Z_LOW} and {ANCHOR_Z_HIGH} are required.")
    return real_ads, spatialz_ads, axis_ads


def genes_to_evaluate(real_ads: Dict[int, AnnData], spatialz_ads: Dict[int, AnnData], axis_ads: Dict[int, AnnData]) -> Tuple[List[str], List[str], pd.DataFrame]:
    target_sp = get_nearest_available_z(spatialz_ads, CASE_VIRTUAL_Z)
    target_ax = get_nearest_available_z(axis_ads, CASE_VIRTUAL_Z)
    ads_for_common = [real_ads[ANCHOR_Z_LOW], real_ads[ANCHOR_Z_HIGH]]
    if target_sp is not None:
        ads_for_common.append(spatialz_ads[target_sp])
    if target_ax is not None:
        ads_for_common.append(axis_ads[target_ax])
    common = common_genes_across(ads_for_common)

    marker_genes = [g for g in MARKER_GENES if g in common]
    top_genes, selection_df = select_top_hvg_genes(real_ads, common)
    selection_df.to_csv(GENE_SELECTION_CSV, index=False)
    print(f"[SAVE] {GENE_SELECTION_CSV}")
    print(f"[INFO] marker genes: {len(marker_genes)} | top highly variable genes: {len(top_genes)}")
    return marker_genes, top_genes, selection_df


def make_field_cache(real_ads: Dict[int, AnnData], spatialz_ads: Dict[int, AnnData],
                     axis_ads: Dict[int, AnnData], genes: Sequence[str],
                     marker_genes: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Cache fields.

    For marker genes, compute all available virtual depths so trajectories can be
    plotted.  For top HVG/spatial genes, compute only the case depth to keep the
    light evaluation fast.
    """
    target_sp = get_nearest_available_z(spatialz_ads, CASE_VIRTUAL_Z)
    target_ax = get_nearest_available_z(axis_ads, CASE_VIRTUAL_Z)
    marker_set = set(marker_genes or [])

    cache = {
        "genes": {},
        "case_virtual_z": int(CASE_VIRTUAL_Z),
        "target_spatialz_z": int(target_sp) if target_sp is not None else None,
        "target_axis_z": int(target_ax) if target_ax is not None else None,
    }
    for gene in genes:
        fields_by_method: Dict[str, Dict[int, Dict[str, object]]] = {}

        # Always cache both real reference slices.
        collections = [("Real", {ANCHOR_Z_LOW: real_ads[ANCHOR_Z_LOW], ANCHOR_Z_HIGH: real_ads[ANCHOR_Z_HIGH]})]

        # Marker genes get a full virtual z trajectory. Top genes use only the
        # case depth.
        if gene in marker_set:
            if len(spatialz_ads) > 0:
                collections.append(("SpatialZ", spatialz_ads))
            if len(axis_ads) > 0:
                collections.append(("AXIS", axis_ads))
        else:
            if target_sp is not None:
                collections.append(("SpatialZ", {int(target_sp): spatialz_ads[target_sp]}))
            if target_ax is not None:
                collections.append(("AXIS", {int(target_ax): axis_ads[target_ax]}))

        for method, ads in collections:
            fields_by_method[method] = {}
            for z, ad in sorted(ads.items()):
                print(f"[FIELD] {gene} | {method} | z={z}")
                t0 = perf_counter()
                fields_by_method[method][int(z)] = compute_gene_field(ad, gene)
                print(f"[TIME] field {gene} {method} z={z}: {perf_counter() - t0:.2f}s")
        cache["genes"][gene] = {"fields": fields_by_method}
    return cache

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
def evaluate_all() -> Tuple[pd.DataFrame, Dict[str, object], List[str], List[str]]:
    ensure_dir(OUTDIR)
    real_ads, spatialz_ads, axis_ads = load_and_preprocess_all()
    marker_genes, top_genes, _ = genes_to_evaluate(real_ads, spatialz_ads, axis_ads)
    all_genes = list(dict.fromkeys(marker_genes + top_genes))
    field_cache = make_field_cache(real_ads, spatialz_ads, axis_ads, all_genes, marker_genes=marker_genes)

    rows = []
    for gene in all_genes:
        fields_by_method = field_cache["genes"][gene]["fields"]
        real10 = fields_by_method["Real"][ANCHOR_Z_LOW]
        real310 = fields_by_method["Real"][ANCHOR_Z_HIGH]
        for method in METHOD_ORDER:
            if method not in fields_by_method:
                continue
            for z, cand in sorted(fields_by_method[method].items()):
                row = evaluate_candidate_against_real_refs(gene, method, int(z), cand, real10, real310)
                row["gene_set"] = "marker" if gene in marker_genes else ("top100_hvg" if gene in top_genes else "case_only")
                row["is_case_depth"] = bool(int(z) == int(CASE_VIRTUAL_Z))
                rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"[SAVE] {METRICS_CSV}")

    with open(FIELD_CACHE_PATH, "wb") as f:
        pickle.dump({
            "field_cache": field_cache,
            "marker_genes": marker_genes,
            "top_genes": top_genes,
            "metrics_df": metrics_df,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SAVE] {FIELD_CACHE_PATH}")
    return metrics_df, field_cache, marker_genes, top_genes


def load_eval_cache():
    if not os.path.exists(FIELD_CACHE_PATH):
        raise FileNotFoundError(f"Cache not found: {FIELD_CACHE_PATH}. Run RUN_MODE='eval' or 'both'.")
    with open(FIELD_CACHE_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["metrics_df"], obj["field_cache"], obj["marker_genes"], obj["top_genes"]


# ============================================================
# PLOTTING
# ============================================================
SUMMARY_METRICS = (
    ("transition_ncc_to_real10", f"Similarity to measured {ANCHOR_Z_LOW} µm section"),
    ("transition_ncc_to_real310", f"Similarity to measured {ANCHOR_Z_HIGH} µm section"),
)


def _summary_gene_set_legend(
    style: SummaryStyle
):
    """Filled boxes denote marker genes; open boxes denote top genes."""
    return [
        Patch(
            facecolor="0.60",
            edgecolor="black",
            linewidth=style.box_edge_width,
            alpha=style.marker_box_alpha,
            label=f"Marker genes",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            linewidth=style.box_edge_width,
            label=f"Top {TOP_N_HVG_GENES} HVGs",
        ),
    ]


def _plot_summary_box_pair(
    ax,
    data: pd.DataFrame,
    metric: str,
    methods: Sequence[str],
    style: SummaryStyle,
    seed: int,
) -> None:
    """SCC-matched summary: marker=filled; top genes=open; method color encodes method."""
    rng = np.random.default_rng(seed)
    center_positions = np.arange(len(methods), dtype=float)
    offsets = {
        "marker": -style.gene_set_offset,
        "top": style.gene_set_offset,
    }

    for method_idx, method in enumerate(methods):
        method_df = data[data["method"].astype(str) == str(method)]
        method_color = METHOD_PALETTE[method]

        for gene_set in ("marker", "top"):
            values = (
                method_df.loc[
                    method_df["summary_gene_set"].astype(str) == gene_set,
                    metric,
                ]
                .astype(float)
                .to_numpy()
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            position = float(method_idx) + offsets[gene_set]
            is_marker = gene_set == "marker"
            facecolor = method_color if is_marker else "none"

            ax.boxplot(
                [values],
                positions=[position],
                widths=style.box_width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={
                    "facecolor": facecolor,
                    "edgecolor": "black",
                    "linewidth": style.box_edge_width,
                    "alpha": style.marker_box_alpha if is_marker else 1.0,
                },
                medianprops={
                    "color": "black",
                    "linewidth": style.median_width,
                },
                whiskerprops={
                    "color": "black",
                    "linewidth": style.whisker_width,
                },
                capprops={
                    "color": "black",
                    "linewidth": style.whisker_width,
                },
                zorder=2,
            )

            jitter = (
                rng.uniform(
                    -style.point_jitter,
                    style.point_jitter,
                    size=values.size,
                )
                if values.size > 1
                else np.zeros(values.size, dtype=float)
            )
            ax.scatter(
                np.full(values.size, position) + jitter,
                values,
                s=style.point_size,
                facecolor=method_color,
                edgecolor="none",
                alpha=(
                    style.marker_point_alpha
                    if is_marker
                    else style.top_point_alpha
                ),
                rasterized=True,
                zorder=3,
            )

    ax.set_xlim(-0.55, len(methods) - 0.45)
    ax.set_xticks(center_positions)
    ax.set_xticklabels(methods, fontsize=style.tick_fontsize)
    ax.tick_params(axis="y", labelsize=style.tick_fontsize)
    ax.grid(
        axis="y",
        linewidth=0.6,
        color="0.87",
        alpha=style.grid_alpha,
        zorder=0,
    )
    ax.grid(axis="x", visible=False)

    legend_kwargs = dict(
        handles=_summary_gene_set_legend(style),
        loc=style.legend.loc,
        ncol=style.legend.ncol,
        frameon=style.legend.frameon,
        fontsize=style.legend.fontsize,
        handletextpad=style.legend.handletextpad,
        columnspacing=style.legend.columnspacing,
        labelspacing=style.legend.labelspacing,
        borderaxespad=style.legend.borderaxespad,
    )
    if style.legend.bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = style.legend.bbox_to_anchor
    ax.legend(**legend_kwargs)


def plot_metric_summary(
    metrics_df: pd.DataFrame,
    save_path: str = SUMMARY_FIG,
    style: SummaryStyle = SUMMARY_STYLE,
) -> Optional[pd.DataFrame]:
    """Case-depth anchor-referenced NCC summary in the SCC publication style."""
    df = metrics_df[
        metrics_df["gene_set"].isin(["marker", "top100_hvg"])
    ].copy()
    df = df[
        df["method"].isin(METHOD_ORDER)
        & df["is_case_depth"].astype(bool)
    ].copy()

    if df.empty:
        print("[WARN] no case-depth metrics for summary")
        return None

    df["summary_gene_set"] = np.where(
        df["gene_set"].astype(str) == "marker",
        "marker",
        "top",
    )
    methods = [
        method
        for method in METHOD_ORDER
        if method in set(df["method"].astype(str))
    ]
    if not methods:
        print("[WARN] no candidate methods available for summary")
        return None

    set_publication_style()
    fig, axes = plt.subplots(
        nrows=len(SUMMARY_METRICS),
        ncols=1,
        figsize=style.figsize,
        squeeze=False,
        sharex=True,
    )
    axes = axes[:, 0]

    for metric_idx, (metric, title) in enumerate(SUMMARY_METRICS):
        ax = axes[metric_idx]
        if metric not in df.columns:
            ax.text(
                0.5,
                0.5,
                "Metric unavailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=style.axis_label_fontsize,
                color="0.45",
            )
            ax.set_axis_off()
            continue

        _plot_summary_box_pair(
            ax=ax,
            data=df,
            metric=metric,
            methods=methods,
            style=style,
            seed=CASE_VIRTUAL_Z * 100 + metric_idx,
        )
        ax.set_title(
            title,
            fontsize=style.panel_title_fontsize,
            fontweight=style.panel_title_weight,
            pad=style.panel_title_pad,
            x=0.4,
        )
        ax.set_ylabel("Shift-tolerant NCC", fontsize=style.axis_label_fontsize)
        ax.set_ylim(*style.y_limits)

        # if metric_idx < len(SUMMARY_METRICS) - 1:
        #     ax.tick_params(axis="x", labelbottom=False)
        # else:
        #     ax.set_xlabel("Method", fontsize=style.axis_label_fontsize)



    fig.subplots_adjust(
        left=style.left,
        right=style.right,
        bottom=style.bottom,
        top=style.top,
        hspace=style.hspace,
    )

    fig.savefig(
        save_path,
        dpi=style.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"[SAVE] {save_path}")
    plt.close(fig)

    summary_rows = []
    for gene_set in ("marker", "top"):
        set_df = df[df["summary_gene_set"].astype(str) == gene_set]
        for method in methods:
            sub = set_df[set_df["method"].astype(str) == str(method)]
            row = {
                "gene_set": gene_set,
                "method": method,
                "n_genes": int(sub["gene"].nunique()),
            }
            for metric, _ in SUMMARY_METRICS:
                values = (
                    pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
                    if metric in sub.columns
                    else np.asarray([], dtype=float)
                )
                values = values[np.isfinite(values)]
                row[f"{metric}_median"] = (
                    float(np.nanmedian(values)) if values.size else np.nan
                )
                row[f"{metric}_mean"] = (
                    float(np.nanmean(values)) if values.size else np.nan
                )
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"[SAVE] {SUMMARY_CSV}")
    return summary_df


def get_real_reference_rows_for_transition_ncc(
    gene: str,
    field_cache: Dict[str, object],
) -> pd.DataFrame:
    """Build pseudo-Real rows so marker stability plots can show the real anchor range.

    For transition NCC:
      - to Real 10:  Real10->Real10 = 1.0, Real310->Real10 = NCC(real310, real10)
      - to Real 310: Real10->Real310 = NCC(real10, real310), Real310->Real310 = 1.0
    """
    genes_block = field_cache.get("genes", {}) if isinstance(field_cache, dict) else {}
    if gene not in genes_block:
        return pd.DataFrame()
    fields_by_method = genes_block[gene].get("fields", {})
    if "Real" not in fields_by_method:
        return pd.DataFrame()
    real_fields = fields_by_method["Real"]
    real10 = real_fields.get(ANCHOR_Z_LOW, None)
    real310 = real_fields.get(ANCHOR_Z_HIGH, None)
    if real10 is None or real310 is None:
        return pd.DataFrame()

    ncc_310_to_10, dy_310_to_10, dx_310_to_10 = shifted_map_ncc_vertical(
        real10["grad_mag"], real310["grad_mag"],
        max_shift_y=TRANSITION_NCC_MAX_SHIFT_Y_BINS,
        max_shift_x=TRANSITION_NCC_MAX_SHIFT_X_BINS,
        min_overlap=MIN_MAP_OVERLAP_PIXELS,
    )
    ncc_10_to_310, dy_10_to_310, dx_10_to_310 = shifted_map_ncc_vertical(
        real310["grad_mag"], real10["grad_mag"],
        max_shift_y=TRANSITION_NCC_MAX_SHIFT_Y_BINS,
        max_shift_x=TRANSITION_NCC_MAX_SHIFT_X_BINS,
        min_overlap=MIN_MAP_OVERLAP_PIXELS,
    )

    rows = [
        {
            "gene": gene,
            "method": "Real",
            "z": int(ANCHOR_Z_LOW),
            "gene_set": "marker",
            "transition_ncc_to_real10": 1.0,
            "transition_ncc_to_real310": float(ncc_10_to_310) if np.isfinite(ncc_10_to_310) else np.nan,
            "transition_best_shift_y_to_real10": 0,
            "transition_best_shift_x_to_real10": 0,
            "transition_best_shift_y_to_real310": int(dy_10_to_310),
            "transition_best_shift_x_to_real310": int(dx_10_to_310),
        },
        {
            "gene": gene,
            "method": "Real",
            "z": int(ANCHOR_Z_HIGH),
            "gene_set": "marker",
            "transition_ncc_to_real10": float(ncc_310_to_10) if np.isfinite(ncc_310_to_10) else np.nan,
            "transition_ncc_to_real310": 1.0,
            "transition_best_shift_y_to_real10": int(dy_310_to_10),
            "transition_best_shift_x_to_real10": int(dx_310_to_10),
            "transition_best_shift_y_to_real310": 0,
            "transition_best_shift_x_to_real310": 0,
        },
    ]
    return pd.DataFrame(rows)



def _plot_real_range_stability_panel(
    ax,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    y_limits: Optional[Tuple[float, float]] = None,
    show_legend: bool = False,
):
    """Shared Cell-style stability panel: Real anchor range + scattered virtual slices."""
    method_order = list(PLOT_METHOD_ORDER)
    method_offsets = {"Real": -2.2, "SpatialZ": 0.0, "AXIS": 2.2}
    method_markers = {"Real": "o", "SpatialZ": "s", "AXIS": "D"}

    real_vals = df[(df["method"] == "Real") & np.isfinite(df[metric])][metric].to_numpy(dtype=np.float64)
    if real_vals.size > 0:
        ref_min = float(np.nanmin(real_vals))
        ref_max = float(np.nanmax(real_vals))
        ref_mean = float(np.nanmean(real_vals))
        if np.isclose(ref_min, ref_max):
            pad = max(abs(ref_mean) * 0.02, 0.01)
            ref_min -= pad
            ref_max += pad
        ax.axhspan(ref_min, ref_max, color="0.2", alpha=0.10, zorder=0,
                   label="Measured-reference range" if show_legend else None)
        ax.axhline(ref_mean, color="0.25", linewidth=1.4, linestyle="--", alpha=0.80, zorder=1,
                   label="Measured-reference mean" if show_legend else None)

    for method in method_order:
        ss = df[(df["method"] == method) & np.isfinite(df[metric])].sort_values("z")
        if ss.empty:
            continue
        x = ss["z"].to_numpy(dtype=float) + method_offsets.get(method, 0.0)
        y = ss[metric].to_numpy(dtype=float)
        ax.scatter(
            x, y,
            s=42 if method == "Real" else 36,
            marker=method_markers.get(method, "o"),
            facecolor=PLOT_PALETTE[method],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.96,
            zorder=3,
            label = "Measured sections" if method == "Real" else method,
            # label=method if show_legend else None,
        )
        if method in ["SpatialZ", "AXIS"] and len(ss) >= 2:
            med = float(np.nanmedian(y))
            xmin = float(np.nanmin(ss["z"]))
            xmax = float(np.nanmax(ss["z"]))
            ax.hlines(med, xmin, xmax, colors=METHOD_PALETTE[method],
                      linestyles="-", linewidth=2.2, alpha=0.38, zorder=2)
        txt_lines = []
        for method in ["AXIS","SpatialZ"]:
            vals = df[(df["method"] == method) & np.isfinite(df[metric])][metric].to_numpy(dtype=np.float64)
            if vals.size == 0:
                continue
            med = float(np.nanmedian(vals))
            txt_lines.append(f"{method}: median={med:.2f}")
        if txt_lines:
            ax.text(
                0.985, 0.04,
                "\n".join(txt_lines),
                transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.85", alpha=0.94),
            )
    ax.set_title(ylabel,fontsize=14)
    ax.set_ylabel("Shift-tolerant NCC", fontsize=10.5)
    ax.grid(alpha=0.16)
    ax.axhline(0.0, color="0.60", linewidth=0.8, linestyle=":", alpha=0.6)
    if y_limits is not None:
        ax.set_ylim(*y_limits)


def plot_marker_metric_trajectory(
    metrics_df: pd.DataFrame,
    field_cache: Dict[str, object],
    gene: str,
    save_path: Optional[str] = None,
):
    """Plot transition-map NCC across anatomical depth."""

    set_publication_style()
    ensure_dir(MARKER_TRAJECTORY_DIR)

    if save_path is None:
        save_path = os.path.join(
            MARKER_TRAJECTORY_DIR,
            f"{gene}_transition_ncc_stability.png",
        )

    df = metrics_df[
        (metrics_df["gene"] == gene)
        & (metrics_df["gene_set"] == "marker")
    ].copy()

    if df.empty:
        print(f"[WARN] no metrics for gene={gene}; skip NCC trajectory plot")
        return

    # Add measured reference sections.
    real_df = get_real_reference_rows_for_transition_ncc(
        gene,
        field_cache,
    )

    df = pd.concat(
        [real_df, df],
        axis=0,
        ignore_index=True,
    )

    df["method"] = pd.Categorical(
        df["method"],
        categories=PLOT_METHOD_ORDER,
        ordered=True,
    )
    df = df.sort_values(["method", "z"])

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    metrics = [
        (
            "transition_ncc_to_real10",
            f"{gene} similarity to measured {ANCHOR_Z_LOW} µm section",
        ),
        (
            "transition_ncc_to_real310",
            f"{gene} similarity to measured {ANCHOR_Z_HIGH} µm section",
        ),
    ]

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(7, 3.0 * len(metrics)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    # ---------------------------------------------------------
    # Explicit anatomical-depth ticks
    # ---------------------------------------------------------
    depth_ticks = [
        ANCHOR_Z_LOW,
        60,
        110,
        CASE_VIRTUAL_Z,
        210,
        260,
        ANCHOR_Z_HIGH,
    ]

    for row_id, (ax, (metric, ylabel)) in enumerate(
        zip(axes, metrics)
    ):
        _plot_real_range_stability_panel(
            ax=ax,
            df=df,
            metric=metric,
            ylabel=ylabel,
            y_limits=(0.0, 1.02),

            # Only the first axis needs all legend handles.
            show_legend=(row_id == 0),
        )

        # Show the same anatomical-depth ticks on BOTH panels.
        ax.set_xticks(depth_ticks)
        ax.set_xticklabels(
            [str(z) for z in depth_ticks],
            fontsize=10,
        )

        # sharex=True normally hides upper-panel labels.
        ax.tick_params(
            axis="x",
            labelbottom=True,
        )

        ax.set_xlabel(
            "Anatomical depth (µm)",
            fontsize=10,
        )

    # ---------------------------------------------------------
    # Bottom legend with controlled ordering
    # ---------------------------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()

    handle_lookup = {
        label: handle
        for handle, label in zip(handles, labels)
    }

    legend_order = [
        "AXIS",
        "SpatialZ",
        "Measured sections",
        "Measured-reference mean",
        "Measured-reference range",
    ]

    ordered_handles = [
        handle_lookup[label]
        for label in legend_order
        if label in handle_lookup
    ]

    ordered_labels = [
        label
        for label in legend_order
        if label in handle_lookup
    ]

    axes[-1].legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.62),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handletextpad=0.55,
        columnspacing=1.1,
    )

    plt.tight_layout()

    fig.savefig(
        save_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"[SAVE] {save_path}")
    plt.close(fig)


def plot_marker_shift_trajectory(
    metrics_df: pd.DataFrame,
    field_cache: Dict[str, object],
    gene: str,
    save_path: Optional[str] = None,
):
    """Plot best vertical y-shift stability against Real 10 and Real 310 anchors."""
    set_publication_style()
    ensure_dir(MARKER_TRAJECTORY_DIR)
    if save_path is None:
        save_path = os.path.join(MARKER_TRAJECTORY_DIR, f"{gene}_transition_yshift_stability.png")

    df = metrics_df[(metrics_df["gene"] == gene) & (metrics_df["gene_set"] == "marker")].copy()
    if df.empty:
        print(f"[WARN] no metrics for gene={gene}; skip y-shift trajectory plot")
        return

    real_df = get_real_reference_rows_for_transition_ncc(gene, field_cache)
    df = pd.concat([real_df, df], axis=0, ignore_index=True)
    df["method"] = pd.Categorical(df["method"], categories=PLOT_METHOD_ORDER, ordered=True)
    df = df.sort_values(["method", "z"])

    max_shift = float(max(abs(int(TRANSITION_NCC_MAX_SHIFT_Y_BINS)), 1))
    metrics = [
        ("transition_best_shift_y_to_real10", f"Best y-shift to Real {ANCHOR_Z_LOW} µm (px)"),
        ("transition_best_shift_y_to_real310", f"Best y-shift to Real {ANCHOR_Z_HIGH} µm (px)"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(6.2, 3.0 * len(metrics)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for row_id, (ax, (metric, ylabel)) in enumerate(zip(axes, metrics)):
        _plot_real_range_stability_panel(
            ax=ax,
            df=df,
            metric=metric,
            ylabel=ylabel,
            y_limits=(-max_shift - 0.5, max_shift + 0.5),
            show_legend=(row_id == 0),
        )
    axes[-1].set_xlabel("z position (µm)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.34),
                       ncol=5, frameon=False, fontsize=9.4)
    fig.suptitle(f"{gene} marker best vertical-shift stability across z", y=1.01,
                 fontsize=15.0, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"[SAVE] {save_path}")
    plt.close(fig)


def plot_all(
    metrics_df: Optional[pd.DataFrame] = None,
    field_cache: Optional[Dict[str, object]] = None,
) -> None:
    if metrics_df is None or field_cache is None:
        metrics_df, field_cache, marker_genes, _ = load_eval_cache()
    else:
        marker_genes = [g for g in MARKER_GENES if g in field_cache.get("genes", {})]
    plot_metric_summary(metrics_df, SUMMARY_FIG)
    for gene in marker_genes:
        plot_marker_metric_trajectory(
            metrics_df=metrics_df,
            field_cache=field_cache,
            gene=gene,
            save_path=os.path.join(MARKER_TRAJECTORY_DIR, f"{gene}_transition_ncc_stability.png"),
        )
        # plot_marker_shift_trajectory(
        #     metrics_df=metrics_df,
        #     field_cache=field_cache,
        #     gene=gene,
        #     save_path=os.path.join(MARKER_TRAJECTORY_DIR, f"{gene}_transition_yshift_stability.png"),
        # )


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_dir(OUTDIR)
    if RUN_MODE == "eval":
        evaluate_all()
    elif RUN_MODE == "plot":
        metrics_df, field_cache, _, _ = load_eval_cache()
        plot_all(metrics_df=metrics_df, field_cache=field_cache)
    elif RUN_MODE == "both":
        metrics_df, field_cache, _, _ = evaluate_all()
        plot_all(metrics_df=metrics_df, field_cache=field_cache)
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()

