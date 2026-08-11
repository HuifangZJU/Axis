#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light DLPFC transition-map NCC evaluation for virtual slices.

Purpose
-------
This version focuses on DLPFC cortical-layer biology rather than generic
trajectory plots.  It evaluates marker genes and computationally selected
highly expressed / spatially variable genes at the case depth, usually 160 µm.

Main changes from the previous marker-gradient script
-----------------------------------------------------
1. Top genes are selected by:
   expression/support QC in Real 10 µm and Real 310 µm, then
   ranking by mean laminar R² across Real 10 µm and Real 310 µm.

2. Main reported metrics:
   - shift-tolerant laminar profile NCC
   - shift-tolerant transition-map NCC
   - largest component fraction

   Additional diagnostic metrics are still written to CSV, but the final figure
   focuses on the three most interpretable laminar-layer metrics.

3. Case study:
   only the selected case depth is shown.  No trajectory figures are generated.
   The first row is raw spot-level log1p expression, not a pixelized gene map.

Assumption
----------
DLPFC cortical layers are approximately horizontal after the same y-flip used in
the original script.  Therefore normalized y is used as the laminar-depth axis.
"""

from __future__ import annotations

import os
import pickle
import warnings
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

from anndata import AnnData
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, label as ndi_label
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIG
# ============================================================
ROOT = "/media/huifang/data/registration/DLPFC/huifang"
H5AD_DIR = os.path.join(ROOT, "h5ad")
VIRTUAL_DIR = os.path.join(ROOT, "virtual_tissue")
OUTDIR = "/media/huifang/data/registration/evaluation/virtual_generation/dlpfc/"
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

MARKER_GENES = ["MOBP", "CUX2", "RORB", "ENC1", "PCP4", "KCNIP2", "FOXP2", "MBP", "MFGE8","SCGB2A2","GFAP"]
CASE_STUDY_GENE = "PCP4"
TOP_N_HVG_GENES = 100
TOP_HVG_PREFILTER_N = 2000   # first-pass highly variable genes before expression/spatial-variance ranking
EXCLUDE_MARKER_FROM_TOP = True

METHOD_ORDER = ["SpatialZ", "AXIS"]
PLOT_METHOD_ORDER = ["Real", "SpatialZ", "AXIS"]
METHOD_PALETTE = {
    "Real": "#111111",
    "SpatialZ": "#7C8DA6",
    "AXIS": "#C4473A",
}

# Gene-selection QC on Real 10 and 310.
TOP_QC_MIN_DETECTED_SPOTS = 30
TOP_QC_MIN_DETECTED_FRAC = 0.03
TOP_QC_MIN_MEAN_LOG1P = 0.02

# Rasterization / gradient.
FIELD_GRID_BINS = 80
DEPTH_PROFILE_BINS = 48
PIXELS_PER_SPOT_FOR_DISPLAY = 1.0
REGION_TREND_SIGMA_GRID = 3.2
VECTOR_FIELD_SMOOTH_SIGMA_GRID = 1.3
VALID_WEIGHT_EPS = 1e-4
EPS = 1e-8

# Shift-tolerant transition-map NCC metric.
LAMINAR_PROFILE_MAX_SHIFT_BINS = 5
TRANSITION_NCC_MAX_SHIFT_Y_BINS = 20
TRANSITION_NCC_MAX_SHIFT_X_BINS = 10
MIN_PROFILE_OVERLAP_BINS = 16
MIN_MAP_OVERLAP_PIXELS = 80

# Hotspot / laminar-band metrics.
HOTSPOT_TOP_FRAC = 0.20
HOTSPOT_MIN_PIXELS = 10

# Case-study plotting.
STREAM_DOWNSAMPLE_FACTOR = 1
STREAMLINE_DENSITY = 0.82
STREAMLINE_LINEWIDTH = 0.72
STREAMLINE_ARROW_SIZE = 0.82
STREAMLINE_MIN_MAG_PERCENTILE_FOR_DISPLAY = 30.0
GRADIENT_MAX_MAG_PERCENTILE_FOR_SUMMARY = 98.5
FIG_DPI = 280

RUN_MODE = "plot"  # "eval", "plot", or "both"

# Outputs.
GENE_SELECTION_CSV = os.path.join(OUTDIR, "hvg_top_gene_selection.csv")
METRICS_CSV = os.path.join(OUTDIR, "dlpfc_transition_ncc_metrics_long.csv")
SUMMARY_CSV = os.path.join(OUTDIR, "dlpfc_transition_ncc_summary.csv")
FIELD_CACHE_PATH = os.path.join(OUTDIR, "dlpfc_transition_ncc_field_cache.pkl")
SUMMARY_FIG = os.path.join(OUTDIR, "dlpfc_transition_ncc_marker_topgene_summary.png")
MARKER_TRAJECTORY_DIR = os.path.join(OUTDIR, "marker_transition_trajectories")


def case_fig_path(gene: str) -> str:
    return os.path.join(OUTDIR, f"{gene}_light_case_study_{CASE_VIRTUAL_Z}um.png")


# ============================================================
# STYLE
# ============================================================
def set_cell_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.8,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "axes.labelsize": 10.5,
        "axes.titlesize": 12,
    })


def make_expression_cmap():
    colors = ["#fff7f3", "#fde0dd", "#fcc5c0", "#fa9fb5", "#f768a1", "#c51b8a", "#7a0177"]
    return LinearSegmentedColormap.from_list("cell_expression_rose", colors, N=256)


def make_gradient_cmap():
    colors = ["#fbfcfd", "#e6f2ef", "#c8e6dc", "#8fcdbf", "#4da6a0", "#2e7ca8", "#173f7a"]
    return LinearSegmentedColormap.from_list("cell_gradient_teal_blue", colors, N=256)


def make_hotspot_cmap():
    colors = ["#ffffff", "#f2f0f7", "#cbc9e2", "#9e9ac8", "#756bb1", "#54278f"]
    return LinearSegmentedColormap.from_list("cell_hotspot_purple", colors, N=256)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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
# LAMINAR DEPTH / FIELD COMPUTATION
# ============================================================
def depth_values_from_coords(coords: np.ndarray) -> np.ndarray:
    """Normalized y coordinate is used as cortical-depth axis."""
    coords_n = normalized_xy(coords)
    return coords_n[:, 1]


def laminar_depth_profile_from_points(coords: np.ndarray, values: np.ndarray, bins: int = DEPTH_PROFILE_BINS) -> np.ndarray:
    coords, values = clean_coords_values(coords, values)
    depth = depth_values_from_coords(coords)
    prof = np.full(bins, np.nan, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(depth, edges) - 1, 0, bins - 1)
    for b in range(bins):
        vals = values[idx == b]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            prof[b] = float(np.mean(vals))
    return prof


def depth_explained_variance_from_points(coords: np.ndarray, values: np.ndarray, bins: int = DEPTH_PROFILE_BINS) -> float:
    coords, values = clean_coords_values(coords, values)
    if values.size < 20:
        return np.nan
    depth = depth_values_from_coords(coords)
    total_var = float(np.nanvar(values))
    if not np.isfinite(total_var) or total_var <= EPS:
        return np.nan
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(depth, edges) - 1, 0, bins - 1)
    pred = np.full_like(values, np.nan, dtype=np.float64)
    for b in range(bins):
        m = idx == b
        if np.any(m):
            pred[m] = np.nanmean(values[m])
    keep = np.isfinite(pred) & np.isfinite(values)
    if np.sum(keep) < 20:
        return np.nan
    explained = float(np.nanvar(pred[keep]))
    return float(np.clip(explained / total_var, 0.0, 1.0))


def rasterize_points_to_grid(coords: np.ndarray, values: np.ndarray, bins: int = FIELD_GRID_BINS) -> Dict[str, np.ndarray]:
    coords, values = clean_coords_values(coords, values)
    coords_n = normalized_xy(coords)
    x = np.clip((coords_n[:, 0] * bins).astype(int), 0, bins - 1)
    y = np.clip((coords_n[:, 1] * bins).astype(int), 0, bins - 1)

    num = np.zeros((bins, bins), dtype=np.float64)
    den = np.zeros((bins, bins), dtype=np.float64)
    for xi, yi, val in zip(x, y, values):
        num[yi, xi] += float(val)
        den[yi, xi] += 1.0

    img = np.divide(num, np.maximum(den, 1.0), out=np.full_like(num, np.nan), where=den > 0)
    valid = den > 0
    return {"image": img, "valid": valid, "count": den}


def masked_gaussian_smooth(image: np.ndarray, valid: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(image)
    num = gaussian_filter(np.where(valid, image, 0.0), sigma=float(sigma), mode="nearest")
    den = gaussian_filter(valid.astype(np.float64), sigma=float(sigma), mode="nearest")
    out = np.divide(num, np.maximum(den, EPS), out=np.full_like(num, np.nan), where=den > VALID_WEIGHT_EPS)
    return out, den > VALID_WEIGHT_EPS


def compute_gene_field(adata: AnnData, gene: str) -> Dict[str, object]:
    coords, raw = clean_coords_values(get_coords(adata), extract_gene_vector(adata, gene))
    z = zscore(raw)

    raw_grid = rasterize_points_to_grid(coords, raw, bins=FIELD_GRID_BINS)
    z_grid = rasterize_points_to_grid(coords, z, bins=FIELD_GRID_BINS)

    z_trend, valid_trend = masked_gaussian_smooth(z_grid["image"], z_grid["valid"], sigma=REGION_TREND_SIGMA_GRID)
    z_fill = np.where(np.isfinite(z_trend), z_trend, 0.0)
    gy_raw, gx_raw = np.gradient(z_fill)

    gx_s, valid_gx = masked_gaussian_smooth(gx_raw, valid_trend, sigma=VECTOR_FIELD_SMOOTH_SIGMA_GRID)
    gy_s, valid_gy = masked_gaussian_smooth(gy_raw, valid_trend, sigma=VECTOR_FIELD_SMOOTH_SIGMA_GRID)
    valid_eval = valid_trend & valid_gx & valid_gy

    gx_s = np.where(valid_eval, gx_s, np.nan)
    gy_s = np.where(valid_eval, gy_s, np.nan)
    grad_mag = np.sqrt(gx_s ** 2 + gy_s ** 2)
    theta = np.mod(np.arctan2(gy_s, gx_s), np.pi)

    depth_profile = laminar_depth_profile_from_points(coords, z, bins=DEPTH_PROFILE_BINS)
    laminar_r2 = depth_explained_variance_from_points(coords, z, bins=DEPTH_PROFILE_BINS)

    return {
        "gene": gene,
        "coords": coords,
        "raw_values": raw,
        "z_values": z,
        "raw_grid": raw_grid["image"],
        "z_grid": z_grid["image"],
        "z_trend": z_trend,
        "valid_eval": valid_eval,
        "gx": gx_s,
        "gy": gy_s,
        "grad_mag": grad_mag,
        "theta": theta,
        "depth_profile": depth_profile,
        "depth_explained_variance": laminar_r2,
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

def expression_qc_for_gene(real10: AnnData, real310: AnnData, gene: str) -> Dict[str, float]:
    x10 = extract_gene_vector(real10, gene)
    x310 = extract_gene_vector(real310, gene)
    detected10 = int(np.sum(np.isfinite(x10) & (x10 > 0)))
    detected310 = int(np.sum(np.isfinite(x310) & (x310 > 0)))
    out = {
        "gene": gene,
        "mean_expr_10": float(np.nanmean(x10)),
        "mean_expr_310": float(np.nanmean(x310)),
        "detected_spots_10": detected10,
        "detected_spots_310": detected310,
        "detected_frac_10": float(detected10 / max(real10.n_obs, 1)),
        "detected_frac_310": float(detected310 / max(real310.n_obs, 1)),
    }
    out["pass_expression_qc"] = bool(
        out["detected_spots_10"] >= TOP_QC_MIN_DETECTED_SPOTS
        and out["detected_spots_310"] >= TOP_QC_MIN_DETECTED_SPOTS
        and out["detected_frac_10"] >= TOP_QC_MIN_DETECTED_FRAC
        and out["detected_frac_310"] >= TOP_QC_MIN_DETECTED_FRAC
        and min(out["mean_expr_10"], out["mean_expr_310"]) >= TOP_QC_MIN_MEAN_LOG1P
    )
    return out


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
def interp_weight(z: int, lo: int = ANCHOR_Z_LOW, hi: int = ANCHOR_Z_HIGH) -> float:
    if hi == lo:
        return 0.5
    return float(np.clip((float(z) - lo) / (hi - lo), 0.0, 1.0))


def nan_interp(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = (1.0 - w) * a + w * b
    out = np.where(np.isfinite(a) & np.isfinite(b), out, np.nan)
    out = np.where(np.isfinite(a) & ~np.isfinite(b), a, out)
    out = np.where(~np.isfinite(a) & np.isfinite(b), b, out)
    return out


def shifted_vector_ncc(a: np.ndarray, b: np.ndarray, max_shift: int, min_overlap: int) -> Tuple[float, int]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    best = -np.inf
    best_shift = 0
    for s in range(-int(max_shift), int(max_shift) + 1):
        if s < 0:
            aa, bb = a[-s:], b[:len(b)+s]
        elif s > 0:
            aa, bb = a[:-s], b[s:]
        else:
            aa, bb = a, b
        keep = np.isfinite(aa) & np.isfinite(bb)
        if np.sum(keep) < min_overlap:
            continue
        r = safe_pearson(aa[keep], bb[keep])
        if np.isfinite(r) and r > best:
            best = r
            best_shift = s
    return (float(best) if np.isfinite(best) else np.nan), int(best_shift)


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


def weighted_orientation_fidelity(ref_gx: np.ndarray, ref_gy: np.ndarray,
                                  cand_gx: np.ndarray, cand_gy: np.ndarray,
                                  ref_mag: np.ndarray, cand_mag: np.ndarray) -> float:
    ref_theta = np.mod(np.arctan2(ref_gy, ref_gx), np.pi)
    cand_theta = np.mod(np.arctan2(cand_gy, cand_gx), np.pi)
    valid = (
        np.isfinite(ref_theta) & np.isfinite(cand_theta)
        & np.isfinite(ref_mag) & np.isfinite(cand_mag)
        & (ref_mag > 0) & (cand_mag > 0)
    )
    if np.sum(valid) < 20:
        return np.nan
    weights = np.sqrt(np.maximum(ref_mag[valid], 0.0) * np.maximum(cand_mag[valid], 0.0))
    if np.sum(weights) <= EPS:
        return np.nan
    diff = ref_theta[valid] - cand_theta[valid]
    return float(np.sum(weights * np.cos(2.0 * diff)) / np.sum(weights))


def hotspot_mask_from_map(z_map: np.ndarray, valid: np.ndarray, top_frac: float = HOTSPOT_TOP_FRAC) -> np.ndarray:
    z_map = np.asarray(z_map, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(z_map)
    if np.sum(valid) < HOTSPOT_MIN_PIXELS:
        return np.zeros_like(valid, dtype=bool)
    q = 100.0 * (1.0 - float(top_frac))
    thr = float(np.nanpercentile(z_map[valid], q))
    mask = valid & (z_map >= thr)
    if np.sum(mask) < HOTSPOT_MIN_PIXELS:
        return np.zeros_like(valid, dtype=bool)
    return mask


def laminar_compactness_from_map(z_map: np.ndarray, valid: np.ndarray) -> float:
    mask = hotspot_mask_from_map(z_map, valid, top_frac=HOTSPOT_TOP_FRAC)
    if np.sum(mask) < HOTSPOT_MIN_PIXELS:
        return np.nan
    yy = np.arange(z_map.shape[0], dtype=np.float64) / max(z_map.shape[0] - 1, 1)
    depth = yy[:, None] * np.ones_like(z_map, dtype=np.float64)
    vals = depth[mask]
    return float(np.nanstd(vals))


def largest_component_fraction_from_map(z_map: np.ndarray, valid: np.ndarray) -> float:
    mask = hotspot_mask_from_map(z_map, valid, top_frac=HOTSPOT_TOP_FRAC)
    total = int(np.sum(mask))
    if total < HOTSPOT_MIN_PIXELS:
        return np.nan
    lab, nlab = ndi_label(mask)
    if nlab < 1:
        return np.nan
    counts = np.bincount(lab.ravel())
    if counts.size <= 1:
        return np.nan
    largest = int(np.max(counts[1:]))
    return float(largest / max(total, 1))


def build_interpolated_anchor(real_lo: Dict[str, object], real_hi: Dict[str, object], z: int) -> Dict[str, object]:
    w = interp_weight(z)
    ref = {
        "z": z,
        "depth_profile": nan_interp(real_lo["depth_profile"], real_hi["depth_profile"], w),
        "z_trend": nan_interp(real_lo["z_trend"], real_hi["z_trend"], w),
        "grad_mag": nan_interp(real_lo["grad_mag"], real_hi["grad_mag"], w),
        "gx": nan_interp(real_lo["gx"], real_hi["gx"], w),
        "gy": nan_interp(real_lo["gy"], real_hi["gy"], w),
        "valid_eval": np.asarray(real_lo["valid_eval"], bool) | np.asarray(real_hi["valid_eval"], bool),
        "depth_explained_variance": float((1.0 - w) * real_lo["depth_explained_variance"] + w * real_hi["depth_explained_variance"]),
    }
    return ref


def evaluate_candidate_against_real_refs(gene: str, method: str, z: int,
                                     cand: Dict[str, object],
                                     real10: Dict[str, object],
                                     real310: Dict[str, object]) -> Dict[str, object]:
    """Compute only vertical-shift-tolerant transition-map NCC to Real10 and Real310."""
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


def evaluate_all() -> Tuple[pd.DataFrame, Dict[str, object], List[str], List[str]]:
    ensure_dir(OUTDIR)
    real_ads, spatialz_ads, axis_ads = load_and_preprocess_all()
    marker_genes, top_genes, _ = genes_to_evaluate(real_ads, spatialz_ads, axis_ads)
    all_genes = list(dict.fromkeys(marker_genes + top_genes + [CASE_STUDY_GENE]))
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
def imshow_clean(ax, img, title, cmap, vmin=None, vmax=None):
    cm = cmap.copy() if hasattr(cmap, "copy") else cmap
    try:
        cm.set_bad(color="#ffffff", alpha=0.0)
    except Exception:
        pass
    im = ax.imshow(np.ma.masked_invalid(img), origin="lower", cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10.6, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    return im


def plot_spot_expression(ax, fields: Dict[str, object], title: str, cmap, vmin: float, vmax: float):
    coords = np.asarray(fields["coords"], dtype=np.float64)
    raw = np.asarray(fields["raw_values"], dtype=np.float64)
    ax.scatter(coords[:, 0], coords[:, 1], c=raw, cmap=cmap, vmin=vmin, vmax=vmax,
               s=5.2, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(title, fontsize=10.6, fontweight="bold")


def plot_depth_profile(ax, fields: Dict[str, object], title: str, color: str, ref_profile: Optional[np.ndarray] = None):
    prof = np.asarray(fields["depth_profile"], dtype=np.float64)
    x = np.linspace(0, 1, len(prof))
    if ref_profile is not None:
        ax.plot(x, ref_profile, color="0.20", lw=1.7, linestyle="--", label="interp. Real10/310")
    ax.plot(x, prof, color=color, lw=2.3, label="slice")
    ax.set_title(title, fontsize=10.6, fontweight="bold")
    ax.set_xlabel("Normalized cortical depth")
    ax.set_ylabel("Mean z(expr)")
    ax.grid(alpha=0.16)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)


def plot_transition_stream_field(ax, fields: Dict[str, object], title: str, mag_vmax: float):
    grad_cmap = make_gradient_cmap()
    mag = np.asarray(fields["grad_mag"], dtype=np.float64)
    valid = np.asarray(fields["valid_eval"], dtype=bool) & np.isfinite(mag)
    gx = np.asarray(fields["gx"], dtype=np.float64)
    gy = np.asarray(fields["gy"], dtype=np.float64)
    imshow_clean(ax, mag, title, grad_cmap, vmin=0, vmax=mag_vmax)

    valid = valid & np.isfinite(gx) & np.isfinite(gy)
    if np.sum(valid) < 10:
        return
    tiny = np.nanpercentile(mag[valid], STREAMLINE_MIN_MAG_PERCENTILE_FOR_DISPLAY)
    valid = valid & (mag >= tiny)

    u = np.nan_to_num(gx, nan=0.0).copy()
    v = np.nan_to_num(gy, nan=0.0).copy()
    u[~valid] = 0.0
    v[~valid] = 0.0
    norm = np.sqrt(u ** 2 + v ** 2)
    u = np.where(norm > EPS, u / np.maximum(norm, EPS), 0.0)
    v = np.where(norm > EPS, v / np.maximum(norm, EPS), 0.0)

    ds = int(max(1, STREAM_DOWNSAMPLE_FACTOR))
    u_ds = u[::ds, ::ds]
    v_ds = v[::ds, ::ds]
    valid_ds = valid[::ds, ::ds]
    if u_ds.shape[0] < 3 or u_ds.shape[1] < 3 or np.sum(valid_ds) < 10:
        return
    x = np.arange(u_ds.shape[1], dtype=np.float64) * ds
    y = np.arange(u_ds.shape[0], dtype=np.float64) * ds
    try:
        ax.streamplot(
            x, y,
            np.ma.array(u_ds, mask=~valid_ds),
            np.ma.array(v_ds, mask=~valid_ds),
            color="black",
            density=STREAMLINE_DENSITY,
            linewidth=STREAMLINE_LINEWIDTH,
            arrowsize=STREAMLINE_ARROW_SIZE,
            minlength=0.08,
            maxlength=8.0,
            integration_direction="both",
            zorder=3,
        )
    except Exception:
        pass


def plot_hotspot_map(ax, fields: Dict[str, object], title: str, cmap, vmin: float, vmax: float):
    zmap = np.asarray(fields["z_trend"], dtype=np.float64)
    valid = np.asarray(fields["valid_eval"], dtype=bool)
    imshow_clean(ax, zmap, title, cmap, vmin=vmin, vmax=vmax)
    mask = hotspot_mask_from_map(zmap, valid, top_frac=HOTSPOT_TOP_FRAC)
    if np.sum(mask) >= HOTSPOT_MIN_PIXELS:
        try:
            ax.contour(mask.astype(float), levels=[0.5], colors="#111111", linewidths=0.7, origin="lower")
        except Exception:
            pass


def _box_strip_axis(ax, sub: pd.DataFrame, metric: str, title: str):
    positions = []
    for i, method in enumerate(METHOD_ORDER):
        vals = sub[sub["method"] == method][metric].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        positions.append(i)
        color = METHOD_PALETTE[method]
        if vals.size:
            ax.boxplot([vals], positions=[i], widths=0.45, patch_artist=True,
                       showfliers=False, manage_ticks=False,
                       medianprops=dict(color="black", linewidth=1.1),
                       boxprops=dict(facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.42),
                       whiskerprops=dict(color="black", linewidth=0.75),
                       capprops=dict(color="black", linewidth=0.75))
            rng = np.random.default_rng(23 + vals.size + i)
            jitter = rng.uniform(-0.08, 0.08, size=vals.size) if vals.size > 1 else np.zeros_like(vals)
            ax.scatter(np.full(vals.size, i) + jitter, vals, s=17,
                       facecolor="white", edgecolor="black", linewidth=0.45, alpha=0.90, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(METHOD_ORDER, fontsize=8.8)
    ax.set_title(title, fontsize=10.2, fontweight="bold")
    if metric.startswith("transition_ncc"):
        ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.16)


# def plot_metric_summary(metrics_df: pd.DataFrame, save_path: str = SUMMARY_FIG):
#     """Case-depth summary of transition-map NCC to Real10 and Real310."""
#     set_cell_style()
#     df = metrics_df[metrics_df["gene_set"].isin(["marker", "top100_hvg"])].copy()
#     df = df[df["method"].isin(METHOD_ORDER) & df["is_case_depth"]]
#     if df.empty:
#         print("[WARN] no case-depth metrics for summary")
#         return
#
#     specs = [
#         ("transition_ncc_to_real10", f"Transition NCC\nto Real {ANCHOR_Z_LOW} µm ↑"),
#         ("transition_ncc_to_real310", f"Transition NCC\nto Real {ANCHOR_Z_HIGH} µm ↑"),
#     ]
#
#     fig, axes = plt.subplots(2, len(specs), figsize=(4.4 * len(specs), 6.5), squeeze=False)
#     for ridx, gene_set in enumerate(["marker", "top100_hvg"]):
#         sub = df[df["gene_set"] == gene_set]
#         n_genes = int(sub["gene"].nunique())
#         for cidx, (metric, title) in enumerate(specs):
#             ax = axes[ridx, cidx]
#             _box_strip_axis(ax, sub, metric, title if ridx == 0 else "")
#             ax.set_ylabel("NCC" if cidx == 0 else "")
#             if cidx == 0:
#                 label = "Marker genes" if gene_set == "marker" else f"Top {TOP_N_HVG_GENES} expressed / spatial genes"
#                 ax.text(0.03, 0.95, f"{label}\nn={n_genes}", transform=ax.transAxes,
#                         ha="left", va="top", fontsize=8.0,
#                         bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="0.82", alpha=0.94))
#     fig.suptitle(f"DLPFC final transition-map NCC at {CASE_VIRTUAL_Z} µm", y=1.02, fontsize=15.5, fontweight="bold")
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
#     print(f"[SAVE] {save_path}")
#     plt.close(fig)
#
#     summary_rows = []
#     for gene_set in ["marker", "top100_hvg"]:
#         for method in METHOD_ORDER:
#             ss = df[(df["gene_set"] == gene_set) & (df["method"] == method)]
#             row = {"gene_set": gene_set, "method": method, "n_genes": int(ss["gene"].nunique())}
#             for metric, _ in specs:
#                 vals = ss[metric].astype(float).to_numpy()
#                 vals = vals[np.isfinite(vals)]
#                 row[f"{metric}_median"] = float(np.nanmedian(vals)) if vals.size else np.nan
#                 row[f"{metric}_mean"] = float(np.nanmean(vals)) if vals.size else np.nan
#             summary_rows.append(row)
#     pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
#     print(f"[SAVE] {SUMMARY_CSV}")

def plot_metric_summary(
    metrics_df: pd.DataFrame,
    save_path: str = SUMMARY_FIG,
):
    """Plot case-depth transition NCC in a 2 x 1 layout with both gene sets."""
    set_cell_style()

    gene_set_order = ["marker", "top100_hvg"]
    gene_set_labels = {
        "marker": "Marker genes",
        "top100_hvg": f"Top {TOP_N_HVG_GENES} hvg genes",
    }
    gene_set_palette = {
        "marker": "#D55E00",
        "top100_hvg": "#0072B2",
    }

    df = metrics_df[
        metrics_df["gene_set"].isin(gene_set_order)
    ].copy()
    df = df[
        df["method"].isin(METHOD_ORDER)
        & df["is_case_depth"]
    ].copy()

    if df.empty:
        print("[WARN] no case-depth metrics for summary")
        return

    specs = [
        (
            "transition_ncc_to_real10",
            f"Transition NCC to Real {ANCHOR_Z_LOW} µm ↑",
        ),
        (
            "transition_ncc_to_real310",
            f"Transition NCC to Real {ANCHOR_Z_HIGH} µm ↑",
        ),
    ]

    n_genes_by_set = {
        gene_set: int(
            df.loc[df["gene_set"] == gene_set, "gene"].nunique()
        )
        for gene_set in gene_set_order
    }
    legend_labels = {
        gene_set: (
            f"{gene_set_labels[gene_set]} "
            f"(n={n_genes_by_set[gene_set]})"
        )
        for gene_set in gene_set_order
    }

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(4.5, 7),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for row_index, (metric, title) in enumerate(specs):
        ax = axes[row_index]

        plot_df = df[
            ["method", "gene_set", "gene", metric]
        ].copy()
        plot_df[metric] = pd.to_numeric(
            plot_df[metric],
            errors="coerce",
        )
        plot_df = plot_df[np.isfinite(plot_df[metric])]

        sns.boxplot(
            data=plot_df,
            x="method",
            y=metric,
            hue="gene_set",
            order=METHOD_ORDER,
            hue_order=gene_set_order,
            palette=gene_set_palette,
            width=0.38,
            dodge=True,
            showfliers=False,
            linewidth=1.0,
            ax=ax,
        )

        sns.stripplot(
            data=plot_df,
            x="method",
            y=metric,
            hue="gene_set",
            order=METHOD_ORDER,
            hue_order=gene_set_order,
            palette=gene_set_palette,
            dodge=True,
            jitter=0.18,
            size=3.0,
            alpha=0.58,
            linewidth=0,
            ax=ax,
        )

        ax.set_title(
            title,
            loc="left",
            fontsize=11.0,
            fontweight="bold",
            pad=5,
        )
        ax.set_ylabel("NCC")
        ax.set_xlabel("")
        ax.set_ylim(0.2, 0.9)
        ax.grid(
            axis="y",
            color="0.90",
            linewidth=0.7,
            zorder=0,
        )
        ax.set_axisbelow(True)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        unique_handles = []
        unique_labels = []
        seen = set()

        for handle, label in zip(handles, labels):
            if label in gene_set_order and label not in seen:
                seen.add(label)
                unique_handles.append(handle)
                unique_labels.append(legend_labels[label])

        if row_index == 0:
            ax.legend(
                unique_handles,
                unique_labels,
                frameon=False,
                loc="upper right",
                fontsize=8.0,
                title_fontsize=8.0,
            )
        elif ax.legend_ is not None:
            ax.legend_.remove()

    axes[-1].tick_params(axis="x", rotation=0)

    # fig.suptitle(
    #     f"Transition-map NCC at {CASE_VIRTUAL_Z} µm",
    #     y=0.995,
    #     fontsize=13.0,
    #     fontweight="bold",
    # )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        save_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    print(f"[SAVE] {save_path}")
    plt.close(fig)

    summary_rows = []

    for gene_set in gene_set_order:
        for method in METHOD_ORDER:
            subset = df[
                (df["gene_set"] == gene_set)
                & (df["method"] == method)
            ]

            row = {
                "gene_set": gene_set,
                "method": method,
                "n_genes": int(subset["gene"].nunique()),
            }

            for metric, _ in specs:
                values = pd.to_numeric(
                    subset[metric],
                    errors="coerce",
                ).to_numpy(dtype=float)
                values = values[np.isfinite(values)]

                row[f"{metric}_median"] = (
                    float(np.nanmedian(values))
                    if values.size
                    else np.nan
                )
                row[f"{metric}_mean"] = (
                    float(np.nanmean(values))
                    if values.size
                    else np.nan
                )

            summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        SUMMARY_CSV,
        index=False,
    )
    print(f"[SAVE] {SUMMARY_CSV}")


def plot_marker_case_study(metrics_df: pd.DataFrame, field_cache: Dict[str, object],
                           gene: str = CASE_STUDY_GENE, save_path: Optional[str] = None):
    """Compact case study: raw expression and transition map only.

    Candidate transition maps are annotated with NCC to Real10 and Real310 and
    the best vertical/horizontal pixel shifts used by the NCC search.
    """
    set_cell_style()
    if save_path is None:
        save_path = case_fig_path(gene)
    if gene not in field_cache["genes"]:
        print(f"[WARN] gene {gene} not in cache")
        return

    fields_by_method = field_cache["genes"][gene]["fields"]
    real10 = fields_by_method["Real"][ANCHOR_Z_LOW]
    real310 = fields_by_method["Real"][ANCHOR_Z_HIGH]
    target_sp = get_nearest_available_z(fields_by_method.get("SpatialZ", {}), CASE_VIRTUAL_Z)
    target_ax = get_nearest_available_z(fields_by_method.get("AXIS", {}), CASE_VIRTUAL_Z)
    panels = [
        ("Real", ANCHOR_Z_LOW, f"Real {ANCHOR_Z_LOW} µm", real10),
        ("Real", ANCHOR_Z_HIGH, f"Real {ANCHOR_Z_HIGH} µm", real310),
    ]
    if target_sp is not None:
        panels.append(("SpatialZ", target_sp, f"SpatialZ {target_sp} µm", fields_by_method["SpatialZ"][target_sp]))
    if target_ax is not None:
        panels.append(("AXIS", target_ax, f"AXIS {target_ax} µm", fields_by_method["AXIS"][target_ax]))

    raw_vals = [p[3]["raw_values"][np.isfinite(p[3]["raw_values"])] for p in panels]
    mag_vals = [p[3]["grad_mag"][np.isfinite(p[3]["grad_mag"])] for p in panels]
    raw_pool = np.concatenate([v for v in raw_vals if v.size]) if any(v.size for v in raw_vals) else np.array([1.0])
    mag_pool = np.concatenate([v for v in mag_vals if v.size]) if any(v.size for v in mag_vals) else np.array([1.0])
    raw_vmax = max(float(np.nanpercentile(raw_pool, 99.0)), 1e-6)
    mag_vmax = max(float(np.nanpercentile(mag_pool, GRADIENT_MAX_MAG_PERCENTILE_FOR_SUMMARY)), 1e-6)

    expr_cmap = make_expression_cmap()
    fig, axes = plt.subplots(2, len(panels), figsize=(4.35 * len(panels), 7.0), squeeze=False)

    for col, (method, z, title, fields) in enumerate(panels):
        plot_spot_expression(axes[0, col], fields, f"{title}\nspot-level log1p expression", expr_cmap, vmin=0, vmax=raw_vmax)
        plot_transition_stream_field(axes[1, col], fields, f"{title}\ntransition map", mag_vmax=mag_vmax)

        if method in METHOD_ORDER:
            rr = metrics_df[(metrics_df["gene"] == gene) & (metrics_df["method"] == method) & (metrics_df["z"] == int(z))]
            if not rr.empty:
                r = rr.iloc[0]
                txt = (
                    f"R10 NCC={r.get('transition_ncc_to_real10', np.nan):.2f}; "
                    f"dy={int(r.get('transition_best_shift_y_to_real10', 0))}, dx={int(r.get('transition_best_shift_x_to_real10', 0))}\n"
                    f"R310 NCC={r.get('transition_ncc_to_real310', np.nan):.2f}; "
                    f"dy={int(r.get('transition_best_shift_y_to_real310', 0))}, dx={int(r.get('transition_best_shift_x_to_real310', 0))}"
                )
                axes[1, col].text(0.98, 0.96, txt, transform=axes[1, col].transAxes,
                                  ha="right", va="top", fontsize=7.8,
                                  bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.80", alpha=0.93))

    cax1 = fig.add_axes([0.925, 0.60, 0.010, 0.25])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(cmap=expr_cmap, norm=plt.Normalize(vmin=0, vmax=raw_vmax)), cax=cax1)
    cb1.set_label(f"{gene} log1p", fontsize=8.3)
    cax2 = fig.add_axes([0.925, 0.18, 0.010, 0.25])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(cmap=make_gradient_cmap(), norm=plt.Normalize(vmin=0, vmax=mag_vmax)), cax=cax2)
    cb2.set_label("gradient magnitude", fontsize=8.3)

    fig.suptitle(f"{gene} DLPFC transition-map NCC case study at {CASE_VIRTUAL_Z} µm",
                 y=0.995, fontsize=16.0, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 0.915, 0.97])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"[SAVE] {save_path}")
    plt.close(fig)


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
                   label="Real range" if show_legend else None)
        ax.axhline(ref_mean, color="0.25", linewidth=1.4, linestyle="--", alpha=0.80, zorder=1,
                   label="Real mean" if show_legend else None)

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
            facecolor=METHOD_PALETTE[method],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.96,
            zorder=3,
            label=method if show_legend else None,
        )
        if method in ["SpatialZ", "AXIS"] and len(ss) >= 2:
            med = float(np.nanmedian(y))
            xmin = float(np.nanmin(ss["z"]))
            xmax = float(np.nanmax(ss["z"]))
            ax.hlines(med, xmin, xmax, colors=METHOD_PALETTE[method],
                      linestyles="-", linewidth=2.2, alpha=0.38, zorder=2)
        txt_lines = []
        for method in ["SpatialZ", "AXIS"]:
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
                ha="right", va="bottom", fontsize=8.4,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.85", alpha=0.94),
            )

    ax.set_ylabel(ylabel, fontsize=10.5)
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
    """Plot marker transition-map NCC stability against Real 10 and Real 310 anchors."""
    set_cell_style()
    ensure_dir(MARKER_TRAJECTORY_DIR)
    if save_path is None:
        save_path = os.path.join(MARKER_TRAJECTORY_DIR, f"{gene}_transition_ncc_stability.png")

    df = metrics_df[(metrics_df["gene"] == gene) & (metrics_df["gene_set"] == "marker")].copy()
    if df.empty:
        print(f"[WARN] no metrics for gene={gene}; skip NCC trajectory plot")
        return

    real_df = get_real_reference_rows_for_transition_ncc(gene, field_cache)
    df = pd.concat([real_df, df], axis=0, ignore_index=True)
    df["method"] = pd.Categorical(df["method"], categories=PLOT_METHOD_ORDER, ordered=True)
    df = df.sort_values(["method", "z"])

    metrics = [
        ("transition_ncc_to_real10", f"Transition NCC to Real {ANCHOR_Z_LOW} µm ↑"),
        ("transition_ncc_to_real310", f"Transition NCC to Real {ANCHOR_Z_HIGH} µm ↑"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(6.2, 3.0 * len(metrics)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for row_id, (ax, (metric, ylabel)) in enumerate(zip(axes, metrics)):
        _plot_real_range_stability_panel(
            ax=ax,
            df=df,
            metric=metric,
            ylabel=ylabel,
            y_limits=(0.0, 1.02),
            show_legend=(row_id == 0),
        )
    axes[-1].set_xlabel("z position (µm)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.34),
                       ncol=5, frameon=False, fontsize=9.4)
    fig.suptitle(f"{gene} marker transition-map NCC stability across z", y=1.01,
                 fontsize=15.0, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"[SAVE] {save_path}")
    plt.close(fig)


def plot_marker_shift_trajectory(
    metrics_df: pd.DataFrame,
    field_cache: Dict[str, object],
    gene: str,
    save_path: Optional[str] = None,
):
    """Plot best vertical y-shift stability against Real 10 and Real 310 anchors."""
    set_cell_style()
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


def plot_all(metrics_df: Optional[pd.DataFrame] = None, field_cache: Optional[Dict[str, object]] = None):
    if metrics_df is None or field_cache is None:
        metrics_df, field_cache, marker_genes, top_genes = load_eval_cache()
    else:
        marker_genes = [g for g in MARKER_GENES if g in field_cache.get("genes", {})]
    # plot_marker_case_study(metrics_df=metrics_df, field_cache=field_cache, gene=CASE_STUDY_GENE, save_path=case_fig_path(CASE_STUDY_GENE))
    plot_metric_summary(metrics_df, SUMMARY_FIG)
    # for gene in marker_genes:
    #     plot_marker_metric_trajectory(
    #         metrics_df=metrics_df,
    #         field_cache=field_cache,
    #         gene=gene,
    #         save_path=os.path.join(MARKER_TRAJECTORY_DIR, f"{gene}_transition_ncc_stability.png"),
    #     )
    #     plot_marker_shift_trajectory(
    #         metrics_df=metrics_df,
    #         field_cache=field_cache,
    #         gene=gene,
    #         save_path=os.path.join(MARKER_TRAJECTORY_DIR, f"{gene}_transition_yshift_stability.png"),
    #     )


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
