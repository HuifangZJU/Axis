#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SCC spatial gene-pattern evaluation and publication plotting.

The workflow supports three modes:

    --mode eval
        Compute/cache numerical results when needed. Existing cache files are reused
        unless --force-recompute is supplied.

    --mode plot
        Plot directly from the existing numerical CSV and cached spatial fields.

    --mode both
        Ensure evaluation results exist, then generate all figures.

Publication outputs:
1. Per-patient metric summary comparing SpatialZ and AXIS for marker genes and
   the top 100 non-marker genes.
2. Marker-gene 3 x 3 case figures arranged by representation:
       row 1: gene expression [Real, SpatialZ, AXIS]
       row 2: spot gradient   [Real, SpatialZ, AXIS]
       row 3: orientation     [Real, SpatialZ, AXIS]

Numerical evaluation definitions are preserved from the supplied script.
"""

from __future__ import annotations
from axis.settings import data_path
from axis.config import load_config
import argparse
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from anndata import AnnData
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, rotate as ndi_rotate, shift as ndi_shift, binary_dilation, binary_closing, binary_fill_holes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

from matplotlib.ticker import FormatStrFormatter
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import pickle
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
ROOT = load_config('scc').root
H5AD_DIR = os.path.join(ROOT, "h5ad")
VIRTUAL_DIR = os.path.join(ROOT, "virtual_tissue")
OUTDIR = data_path("paper_analysis/Figure3_scc/defg/")

GROUP_IDS = [2, 10]
CASE_GROUPS = [2, 10]

COORD_KEY_REAL = "spatial_image_coor"
COORD_KEY_VIRTUAL = "spatial"

NORMALIZE_TOTAL = True
TARGET_SUM = 1e4
LOG1P = True

METHOD_ORDER = [ "AXIS", "SpatialZ"]
PLOT_METHOD_ORDER = ["Real",  "AXIS", "SpatialZ"]
METHOD_PALETTE = {
    "Real": "#111111",
    "SpatialZ": "#7C8DA6",
    "AXIS": "#9E2F2A",
}

MARKER_GENES = [
    "KRT1", "KRT5", "KRT10", "KRT14", "KRT15",
    "COL1A1", "COL1A2", "DCN",
    "PECAM1", "VWF",
    "PTPRC", "CD3D"
]

# Gene QC: skip very sparse / weakly expressed genes because gradient maps become artifact-prone.
MIN_REAL_TOTAL_COUNT = 10.0
MIN_REAL_DETECTED_SPOTS = 3
MIN_REAL_DETECTED_FRAC = 0.02

# Gradient calculation.
K_GRADIENT = 8
K_TREND = 28
K_SPATIAL_SMOOTH = 12
USE_SPATIAL_SMOOTH_FOR_GRADIENT = True
GRADIENT_ZSCORE = True
SPATIAL_SMOOTH_SELF_WEIGHT = 0.50
RIDGE_LAMBDA = 1e-3

# Grids / images.
TRANSITION_GRID_BINS = 64       # main transition-map grid for image metrics
GRID_MASK_DILATE_ITER = 1
GRID_MIN_SPOTS_PER_CELL = 1
GRID_SMOOTH_SIGMA = 1.0

# Image-metric search.
SEARCH_ANGLE_FINE = np.arange(-12, 13, 3, dtype=float)  # residual around PCA estimate
SEARCH_SHIFT_PIXELS = np.arange(-6, 7, 2, dtype=int)
IMAGE_SMOOTH_SIGMA = 1.3
SSIM_WIN = 15
SSIM_GAUSS_SIGMA = 2.0
MIN_OVERLAP_PIXELS = 60

# Histograms.
HIST_BINS_STRENGTH = 18
HIST_BINS_ORIENTATION = 18
GRID_ORIENT_MIN_PERCENTILE = 60.0   # use stronger grid-gradient cells for orientation histogram
HOG_BINS = 18

# Plotting.
FIG_DPI = 320
MAX_QUIVER_ARROWS = 350
STREAM_GRID_BINS = 22
STREAM_MAG_PERCENTILE = 28.0
RANDOM_SEED = 0

# Outputs.
METRICS_CSV = os.path.join(OUTDIR, "marker_transition_alignment_metrics.csv")
SUMMARY_CSV = os.path.join(OUTDIR, "marker_transition_alignment_patient_summary.csv")
TOP_GENE_QC_CSV = os.path.join(OUTDIR, "top_gene_qc_table_all_patients.csv")
TOP_N_SELECTED_GENES = 100


# ============================================================
# GENERIC HELPERS
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# BASIC UTILITIES
# ============================================================
def to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return np.nan
    x = x[keep]
    y = y[keep]
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])




def normalize_total_then_log1p(adata: AnnData) -> AnnData:
    ad = adata.copy()
    if NORMALIZE_TOTAL:
        sc.pp.normalize_total(ad, target_sum=TARGET_SUM)
    if LOG1P:
        sc.pp.log1p(ad)
    return ad


def load_ad(path: str) -> AnnData:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ad = sc.read_h5ad(path)
    if "valid" in ad.obs.columns:
        ad = ad[ad.obs["valid"]].copy()
    return ad


def load_group_data(group_id: int) -> Dict[str, AnnData]:
    real_path = os.path.join(H5AD_DIR, f"{group_id}_1.h5ad")
    axis_path = os.path.join(
        VIRTUAL_DIR,
        f"scc_z_group{group_id}_grid40_spatial3_gene_weight",
        "query",
        "virtual_scc_z_10.000.h5ad",
    )
    spatialz_path = os.path.join(
        VIRTUAL_DIR,
        "spatialz",
        f"group{group_id}",
        f"patient{group_id}_generation_10.h5ad",
    )
    return {
        "Real": load_ad(real_path),
        "SpatialZ": load_ad(spatialz_path),
        "AXIS": load_ad(axis_path),
    }


def common_gene_intersection(adatas: Sequence[AnnData]) -> List[str]:
    common = set(adatas[0].var_names.astype(str))
    for ad in adatas[1:]:
        common &= set(ad.var_names.astype(str))
    return sorted(common)


def subset_to_genes(adata: AnnData, genes: Sequence[str]) -> AnnData:
    genes = [str(g) for g in genes if str(g) in adata.var_names]
    return adata[:, genes].copy()


def coord_key_for_method(method: str) -> str:
    return COORD_KEY_REAL if method == "Real" else COORD_KEY_VIRTUAL


def get_coords(adata: AnnData, method: str) -> np.ndarray:
    key = coord_key_for_method(method)
    if key not in adata.obsm:
        raise KeyError(f"{key!r} not found in adata.obsm for method={method}")
    return np.asarray(adata.obsm[key], dtype=np.float32)


def extract_gene_vector(adata: AnnData, gene: str) -> np.ndarray:
    if gene not in adata.var_names:
        return np.full(adata.n_obs, np.nan, dtype=np.float32)
    X = adata[:, [gene]].X
    return to_dense(X).reshape(-1).astype(np.float32, copy=False)


def finite_coords_mask(coords: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(coords), axis=1)


def median_nn_distance(coords: np.ndarray) -> float:
    coords = np.asarray(coords, dtype=np.float32)
    coords = coords[np.all(np.isfinite(coords), axis=1)]
    if coords.shape[0] < 3:
        return 1.0
    tree = cKDTree(coords)
    try:
        dist, _ = tree.query(coords, k=2, workers=-1)
    except TypeError:
        dist, _ = tree.query(coords, k=2)
    nn = dist[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(np.median(nn)) if nn.size else 1.0


def knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[0] < 2:
        return np.zeros((coords.shape[0], 0), dtype=np.int64)
    kk = min(k + 1, coords.shape[0])
    tree = cKDTree(coords)
    try:
        _, idx = tree.query(coords, k=kk, workers=-1)
    except TypeError:
        _, idx = tree.query(coords, k=kk)
    if idx.ndim == 1:
        idx = idx[:, None]
    return idx[:, 1:].astype(np.int64, copy=False)


def prepare_local_geometry(coords_raw: np.ndarray, k: int) -> Dict[str, np.ndarray]:
    coords_raw = np.asarray(coords_raw, dtype=np.float32)
    valid = finite_coords_mask(coords_raw)
    coords_raw_v = coords_raw[valid]
    spot_unit = median_nn_distance(coords_raw_v)
    coords = coords_raw_v / max(spot_unit, 1e-8)
    nn = knn_indices(coords, k=k)
    n, kk = nn.shape

    if kk == 0:
        A = np.zeros((n, 2, 0), dtype=np.float32)
    else:
        D = coords[nn] - coords[:, None, :]
        W = 1.0 / np.maximum(np.sqrt(np.sum(D ** 2, axis=2)), 1e-3)
        A = np.zeros((n, 2, kk), dtype=np.float32)
        eye = np.eye(2, dtype=np.float32) * float(RIDGE_LAMBDA)
        for i in range(n):
            Di = D[i].astype(np.float32)
            wi = W[i].astype(np.float32)
            G = (Di.T * wi) @ Di + eye
            B = (Di.T * wi)
            try:
                A[i] = np.linalg.solve(G, B).astype(np.float32)
            except np.linalg.LinAlgError:
                A[i] = (np.linalg.pinv(G) @ B).astype(np.float32)
    return {
        "valid_mask": valid,
        "coords_raw": coords_raw_v,
        "coords_spot_units": coords.astype(np.float32),
        "spot_unit": np.asarray(spot_unit, dtype=np.float32),
        "nn_idx": nn,
        "A": A,
    }


def spatial_smooth_vector(x: np.ndarray, nn_idx: np.ndarray, self_weight: float = SPATIAL_SMOOTH_SELF_WEIGHT) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if nn_idx.shape[1] == 0 or x.size == 0:
        return x.copy()
    neigh = np.nanmean(x[nn_idx], axis=1).astype(np.float32)
    w = float(np.clip(self_weight, 0.0, 1.0))
    return (w * x + (1.0 - w) * neigh).astype(np.float32)


def zscore_vector(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / sd).astype(np.float32)


def local_gradient_from_vector(x: np.ndarray, geom: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nn = geom["nn_idx"]
    A = geom["A"]
    if nn.shape[1] == 0:
        z = np.full_like(x, np.nan, dtype=np.float32)
        return z, z, z
    dY = x[nn] - x[:, None]
    beta = np.einsum("ndk,nk->nd", A, dY, optimize=True).astype(np.float32)
    gx = beta[:, 0]
    gy = beta[:, 1]
    strength = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
    return gx, gy, strength


def aggregate_gradient_trend(gx: np.ndarray, gy: np.ndarray, geom_trend: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nn = geom_trend["nn_idx"]
    if nn.shape[1] == 0:
        return gx.copy(), gy.copy(), np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
    gx_t = np.nanmean(gx[nn], axis=1)
    gy_t = np.nanmean(gy[nn], axis=1)
    s_t = np.sqrt(gx_t ** 2 + gy_t ** 2)
    return gx_t.astype(np.float32), gy_t.astype(np.float32), s_t.astype(np.float32)


def normalized_xy(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    mn = np.nanmin(coords, axis=0)
    mx = np.nanmax(coords, axis=0)
    rng = np.maximum(mx - mn, 1e-8)
    return (coords - mn[None, :]) / rng[None, :]


def smooth_binary_mask(mask: np.ndarray, dilate_iter: int = GRID_MASK_DILATE_ITER,
                       close_iter: int = 2, fill_holes_flag: bool = True) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if dilate_iter > 0:
        m = binary_dilation(m, iterations=dilate_iter)
    if close_iter > 0:
        m = binary_closing(m, iterations=close_iter)
    if fill_holes_flag:
        m = binary_fill_holes(m)
    return m.astype(np.uint8)








def build_grid_maps(coords: np.ndarray, strength: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                    bins: int = TRANSITION_GRID_BINS, sigma: float = GRID_SMOOTH_SIGMA) -> Dict[str, np.ndarray]:
    coords_n = normalized_xy(coords)
    x = np.clip((coords_n[:, 0] * bins).astype(int), 0, bins - 1)
    y = np.clip((coords_n[:, 1] * bins).astype(int), 0, bins - 1)

    num_s = np.zeros((bins, bins), dtype=np.float64)
    num_gx = np.zeros((bins, bins), dtype=np.float64)
    num_gy = np.zeros((bins, bins), dtype=np.float64)
    den = np.zeros((bins, bins), dtype=np.float64)

    keep = np.isfinite(strength) & np.isfinite(gx) & np.isfinite(gy)
    for xi, yi, sv, g1, g2 in zip(x[keep], y[keep], strength[keep], gx[keep], gy[keep]):
        num_s[yi, xi] += float(sv)
        num_gx[yi, xi] += float(g1)
        num_gy[yi, xi] += float(g2)
        den[yi, xi] += 1.0

    mask = den >= float(GRID_MIN_SPOTS_PER_CELL)
    mask = smooth_binary_mask(mask) > 0

    mean_s = np.divide(num_s, np.maximum(den, 1.0), out=np.full_like(num_s, np.nan), where=den > 0)
    mean_gx = np.divide(num_gx, np.maximum(den, 1.0), out=np.full_like(num_gx, np.nan), where=den > 0)
    mean_gy = np.divide(num_gy, np.maximum(den, 1.0), out=np.full_like(num_gy, np.nan), where=den > 0)

    if sigma and sigma > 0:
        valid = np.isfinite(mean_s) & (den > 0)
        w = valid.astype(np.float64)
        smooth_s = gaussian_filter(np.where(valid, mean_s, 0.0), sigma=sigma, mode="nearest")
        smooth_gx = gaussian_filter(np.where(valid, mean_gx, 0.0), sigma=sigma, mode="nearest")
        smooth_gy = gaussian_filter(np.where(valid, mean_gy, 0.0), sigma=sigma, mode="nearest")
        smooth_w = gaussian_filter(w, sigma=sigma, mode="nearest")
        mean_s = np.divide(smooth_s, np.maximum(smooth_w, 1e-8), out=np.full_like(smooth_s, np.nan), where=smooth_w > 1e-6)
        mean_gx = np.divide(smooth_gx, np.maximum(smooth_w, 1e-8), out=np.full_like(smooth_gx, np.nan), where=smooth_w > 1e-6)
        mean_gy = np.divide(smooth_gy, np.maximum(smooth_w, 1e-8), out=np.full_like(smooth_gy, np.nan), where=smooth_w > 1e-6)

    mean_s = np.where(mask, mean_s, np.nan)
    mean_gx = np.where(mask, mean_gx, np.nan)
    mean_gy = np.where(mask, mean_gy, np.nan)
    return {
        "grid_strength": mean_s.astype(np.float32),
        "grid_gx": mean_gx.astype(np.float32),
        "grid_gy": mean_gy.astype(np.float32),
        "grid_mask": mask.astype(np.uint8),
    }


def orientation_histogram_from_vectors(gx: np.ndarray, gy: np.ndarray, strength: np.ndarray,
                                       n_bins: int = HOG_BINS) -> np.ndarray:
    gx = np.asarray(gx, dtype=np.float64).ravel()
    gy = np.asarray(gy, dtype=np.float64).ravel()
    strength = np.asarray(strength, dtype=np.float64).ravel()
    keep = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(strength) & (strength > 0)
    if keep.sum() < 5:
        return np.full(n_bins, np.nan, dtype=np.float32)
    theta = np.mod(np.arctan2(gy[keep], gx[keep]), np.pi)
    hist, _ = np.histogram(theta, bins=n_bins, range=(0.0, np.pi), weights=strength[keep])
    hist = hist.astype(np.float64)
    if hist.sum() <= 0:
        return np.full(n_bins, np.nan, dtype=np.float32)
    return (hist / hist.sum()).astype(np.float32)










def build_gene_qc_table(adata: AnnData, genes: Sequence[str]) -> pd.DataFrame:
    """Vectorized real-slice gene QC and selection table."""
    genes = [str(g) for g in genes if str(g) in adata.var_names]
    if len(genes) == 0:
        return pd.DataFrame(columns=[
            "gene", "real_total_count", "real_detected_spots", "real_detected_fraction",
            "pass_qc", "selection_score"
        ])
    ad = adata[:, genes]
    X = ad.X
    if sparse.issparse(X):
        total = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
        detected = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.int64)
    else:
        XX = np.asarray(X, dtype=np.float64)
        total = np.nansum(XX, axis=0)
        detected = np.sum(np.isfinite(XX) & (XX > 0), axis=0).astype(np.int64)
    frac = detected.astype(np.float64) / max(ad.n_obs, 1)
    df = pd.DataFrame({
        "gene": genes,
        "real_total_count": total,
        "real_detected_spots": detected,
        "real_detected_fraction": frac,
    })
    df["pass_qc"] = (
        (df["real_total_count"] >= MIN_REAL_TOTAL_COUNT) &
        (df["real_detected_spots"] >= MIN_REAL_DETECTED_SPOTS) &
        (df["real_detected_fraction"] >= MIN_REAL_DETECTED_FRAC)
    )
    # Composite score for abundant / spatially supported genes.
    pct_total = df["real_total_count"].rank(method="average", pct=True)
    pct_detect = df["real_detected_spots"].rank(method="average", pct=True)
    pct_frac = df["real_detected_fraction"].rank(method="average", pct=True)
    df["selection_score"] = 0.50 * pct_total + 0.30 * pct_detect + 0.20 * pct_frac
    df = df.sort_values(["selection_score", "real_total_count", "real_detected_spots"], ascending=[False, False, False]).reset_index(drop=True)
    return df


def select_top_genes(qc_df: pd.DataFrame, marker_genes: Sequence[str], top_n: int = TOP_N_SELECTED_GENES) -> List[str]:
    marker_set = set(str(g) for g in marker_genes)
    sub = qc_df[qc_df["pass_qc"]].copy()
    sub = sub[~sub["gene"].astype(str).isin(marker_set)]
    if sub.empty:
        return []
    sub = sub.sort_values(["selection_score", "real_total_count", "real_detected_spots"], ascending=[False, False, False])
    return sub["gene"].astype(str).head(int(top_n)).tolist()


def compute_marker_field(adata: AnnData, method: str, gene: str) -> Dict[str, np.ndarray]:
    coords = get_coords(adata, method)
    geom_grad = prepare_local_geometry(coords, k=K_GRADIENT)
    geom_trend = prepare_local_geometry(coords, k=K_TREND)
    geom_smooth = prepare_local_geometry(coords, k=K_SPATIAL_SMOOTH)
    valid = geom_grad["valid_mask"]

    raw_all = extract_gene_vector(adata, gene)
    x_raw = raw_all[valid].astype(np.float32)
    x_for_grad = x_raw.copy()
    if USE_SPATIAL_SMOOTH_FOR_GRADIENT:
        x_for_grad = spatial_smooth_vector(x_for_grad, geom_smooth["nn_idx"], SPATIAL_SMOOTH_SELF_WEIGHT)
    if GRADIENT_ZSCORE:
        x_for_grad = zscore_vector(x_for_grad)

    gx, gy, strength = local_gradient_from_vector(x_for_grad, geom_grad)
    gx_t, gy_t, strength_t = aggregate_gradient_trend(gx, gy, geom_trend)
    grid_maps = build_grid_maps(geom_grad["coords_raw"], strength_t, gx_t, gy_t, bins=TRANSITION_GRID_BINS)
    hog_hist = orientation_histogram_from_vectors(gx_t, gy_t, strength_t, n_bins=HOG_BINS)

    return {
        "coords_raw": geom_grad["coords_raw"],
        "spot_unit": float(geom_grad["spot_unit"]),
        "expr_raw": x_raw,
        "expr_grad_input": x_for_grad,
        "gx": gx,
        "gy": gy,
        "strength": strength,
        "gx_trend": gx_t,
        "gy_trend": gy_t,
        "strength_trend": strength_t,
        "grid_strength": grid_maps["grid_strength"],
        "grid_gx": grid_maps["grid_gx"],
        "grid_gy": grid_maps["grid_gy"],
        "grid_mask": grid_maps["grid_mask"],
        "eval_mask": grid_maps["grid_mask"],
        "hog_hist": hog_hist,
    }


# ============================================================
# RIGID NORMALIZATION / IMAGE METRICS
# ============================================================
def normalize_map_for_similarity(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).copy()
    m = np.asarray(mask, dtype=bool)
    arr[~m] = np.nan
    vals = arr[m & np.isfinite(arr)]
    if vals.size == 0:
        return np.full_like(arr, np.nan, dtype=np.float64)
    mu = float(np.nanmean(vals))
    sd = float(np.nanstd(vals))
    if not np.isfinite(sd) or sd < 1e-8:
        arr[m] = 0.0
    else:
        arr[m] = (arr[m] - mu) / sd
    arr[~m] = np.nan
    if IMAGE_SMOOTH_SIGMA > 0:
        valid = np.isfinite(arr)
        num = gaussian_filter(np.where(valid, arr, 0.0), sigma=IMAGE_SMOOTH_SIGMA, mode="nearest")
        den = gaussian_filter(valid.astype(np.float64), sigma=IMAGE_SMOOTH_SIGMA, mode="nearest")
        arr = np.divide(num, np.maximum(den, 1e-8), out=np.full_like(num, np.nan), where=den > 1e-6)
        arr[~m] = np.nan
    return arr


def masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mm = np.asarray(mask, dtype=bool)
    keep = mm & np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(keep)) < MIN_OVERLAP_PIXELS:
        return np.nan
    return safe_pearson(aa[keep], bb[keep])




def transform_image(img: np.ndarray, angle_deg: float = 0.0,
                    shift_xy: Tuple[float, float] = (0.0, 0.0),
                    order: int = 1, cval: float = np.nan) -> np.ndarray:
    """Rotate and translate a 2D image without changing output shape.

    shift_xy is in image pixels as (x_shift, y_shift). This helper was
    accidentally removed in v5 when the mask-alignment step was removed, but
    the NCC search still needs it for small rotation/translation tolerance.
    """
    out = np.asarray(img, dtype=np.float64)
    if abs(float(angle_deg)) > 1e-8:
        out = ndi_rotate(
            out,
            angle=float(angle_deg),
            reshape=False,
            order=int(order),
            mode="constant",
            cval=float(cval) if np.isfinite(cval) else np.nan,
            prefilter=(int(order) > 1),
        )
    sx, sy = shift_xy
    if abs(float(sx)) > 1e-8 or abs(float(sy)) > 1e-8:
        out = ndi_shift(
            out,
            shift=(float(sy), float(sx)),
            order=int(order),
            mode="constant",
            cval=float(cval) if np.isfinite(cval) else np.nan,
            prefilter=(int(order) > 1),
        )
    return out


def search_transition_similarity(real_map: np.ndarray, real_mask: np.ndarray,
                                 cand_map: np.ndarray, cand_mask: np.ndarray,
                                 rigid_hint: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    real_mask = np.asarray(real_mask, dtype=bool)
    cand_mask = np.asarray(cand_mask, dtype=bool)
    real_norm = normalize_map_for_similarity(real_map, real_mask)
    best = {
        "best_ncc": -np.inf,
        "best_angle": 0.0,
        "best_shift_x": 0.0,
        "best_shift_y": 0.0,
        "best_map": None,
        "best_mask": None,
        "overlap_n": 0,
        "response_heatmap": None,
    }

    base_angle = float(rigid_hint["angle_deg"]) if rigid_hint is not None else 0.0
    angle_list = [base_angle + da for da in SEARCH_ANGLE_FINE]
    best_angle_for_heat = None
    best_rot_map = None
    best_rot_mask = None

    for ang in angle_list:
        rot_map = transform_image(cand_map, angle_deg=ang, shift_xy=(0.0, 0.0), order=1, cval=np.nan)
        rot_mask_f = transform_image(cand_mask.astype(np.float64), angle_deg=ang, shift_xy=(0.0, 0.0), order=0, cval=0.0)
        rot_mask = rot_mask_f > 0.5
        response = np.full((len(SEARCH_SHIFT_PIXELS), len(SEARCH_SHIFT_PIXELS)), np.nan, dtype=np.float64)
        for ix, dx in enumerate(SEARCH_SHIFT_PIXELS):
            for iy, dy in enumerate(SEARCH_SHIFT_PIXELS):
                moved = transform_image(rot_map, angle_deg=0.0, shift_xy=(dx, dy), order=1, cval=np.nan)
                moved_mask_f = transform_image(rot_mask.astype(np.float64), angle_deg=0.0, shift_xy=(dx, dy), order=0, cval=0.0)
                moved_mask = moved_mask_f > 0.5
                overlap = real_mask & moved_mask
                if np.sum(overlap) < 9:
                    continue
                moved_norm = normalize_map_for_similarity(moved, overlap)
                ncc = masked_ncc(real_norm, moved_norm, overlap)
                response[iy, ix] = ncc
                if np.isfinite(ncc) and ncc > best["best_ncc"]:
                    best.update({
                        "best_ncc": float(ncc),
                        "best_angle": float(ang),
                        "best_shift_x": float(dx),
                        "best_shift_y": float(dy),
                        "best_map": moved,
                        "best_mask": overlap.astype(np.uint8),
                        "overlap_n": int(np.sum(overlap)),
                    })
                    best_angle_for_heat = float(ang)
                    best_rot_map = rot_map
                    best_rot_mask = rot_mask
        if best_angle_for_heat is not None and abs(ang - best_angle_for_heat) < 1e-8:
            best["response_heatmap"] = response.copy()

    if best_angle_for_heat is not None and best["response_heatmap"] is None and best_rot_map is not None:
        response = np.full((len(SEARCH_SHIFT_PIXELS), len(SEARCH_SHIFT_PIXELS)), np.nan, dtype=np.float64)
        for ix, dx in enumerate(SEARCH_SHIFT_PIXELS):
            for iy, dy in enumerate(SEARCH_SHIFT_PIXELS):
                moved = transform_image(best_rot_map, angle_deg=0.0, shift_xy=(dx, dy), order=1, cval=np.nan)
                moved_mask_f = transform_image(best_rot_mask.astype(np.float64), angle_deg=0.0, shift_xy=(dx, dy), order=0, cval=0.0)
                moved_mask = moved_mask_f > 0.5
                overlap = real_mask & moved_mask
                if np.sum(overlap) >= 9:
                    response[iy, ix] = masked_ncc(real_norm, normalize_map_for_similarity(moved, overlap), overlap)
        best["response_heatmap"] = response

    if not np.isfinite(best["best_ncc"]):
        best["best_ncc"] = np.nan
    return best

def axial_orientation_components(gx: np.ndarray, gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return axial orientation components cos(2θ), sin(2θ), and magnitude."""
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.arctan2(gy, gx)
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    c2[~np.isfinite(mag)] = np.nan
    s2[~np.isfinite(mag)] = np.nan
    return c2, s2, mag


def spatially_ordered_grid_pearsons(real_field: Dict[str, np.ndarray],
                                    cand_field: Dict[str, np.ndarray],
                                    orient_percentile: float = GRID_ORIENT_MIN_PERCENTILE) -> Dict[str, float]:
    """Spatially ordered grid Pearson metrics.

    Metrics preserve grid-cell order instead of collapsing values into a global
    histogram before comparison. Orientation uses axial encoding, so 0° and
    180° are equivalent.
    """
    rm = np.asarray(real_field.get("eval_mask", real_field["grid_mask"]), dtype=bool)
    cm = np.asarray(cand_field.get("eval_mask", cand_field["grid_mask"]), dtype=bool)
    base = rm & cm

    r_strength = np.asarray(real_field["grid_strength"], dtype=np.float64)
    c_strength = np.asarray(cand_field["grid_strength"], dtype=np.float64)
    strength_mask = base & np.isfinite(r_strength) & np.isfinite(c_strength)
    strength_r = safe_pearson(r_strength[strength_mask], c_strength[strength_mask]) if np.sum(strength_mask) >= 3 else np.nan

    r_c2, r_s2, r_mag = axial_orientation_components(real_field["grid_gx"], real_field["grid_gy"])
    c_c2, c_s2, c_mag = axial_orientation_components(cand_field["grid_gx"], cand_field["grid_gy"])

    orient_mask = (
        base & np.isfinite(r_c2) & np.isfinite(r_s2) & np.isfinite(c_c2) & np.isfinite(c_s2) &
        np.isfinite(r_mag) & np.isfinite(c_mag) & (r_mag > 0) & (c_mag > 0)
    )
    if np.sum(orient_mask) >= 5 and orient_percentile is not None:
        thr = np.nanpercentile(r_mag[orient_mask], float(orient_percentile))
        orient_mask = orient_mask & (r_mag >= thr)
    if np.sum(orient_mask) >= 5:
        r_ori = np.r_[r_c2[orient_mask], r_s2[orient_mask]]
        c_ori = np.r_[c_c2[orient_mask], c_s2[orient_mask]]
        orient_r = safe_pearson(r_ori, c_ori)
    else:
        orient_r = np.nan

    weighted_mask = (
        base & np.isfinite(r_c2) & np.isfinite(r_s2) & np.isfinite(c_c2) & np.isfinite(c_s2) &
        np.isfinite(r_mag) & np.isfinite(c_mag) & (r_mag > 0) & (c_mag > 0)
    )
    if np.sum(weighted_mask) >= 5:
        r_scale = np.nanpercentile(r_mag[weighted_mask], 99.0)
        c_scale = np.nanpercentile(c_mag[weighted_mask], 99.0)
        r_scale = max(float(r_scale), 1e-8) if np.isfinite(r_scale) else 1.0
        c_scale = max(float(c_scale), 1e-8) if np.isfinite(c_scale) else 1.0
        r_w = np.clip(r_mag / r_scale, 0.0, 1.0)
        c_w = np.clip(c_mag / c_scale, 0.0, 1.0)
        r_hog_like = np.r_[(r_w * r_c2)[weighted_mask], (r_w * r_s2)[weighted_mask]]
        c_hog_like = np.r_[(c_w * c_c2)[weighted_mask], (c_w * c_s2)[weighted_mask]]
        weighted_r = safe_pearson(r_hog_like, c_hog_like)
    else:
        weighted_r = np.nan

    return {
        "spatial_grid_strength_pearson": strength_r,
        "spatial_grid_orientation_pearson": orient_r,
        "spatial_grid_strength_weighted_orientation_pearson": weighted_r,
        "spatial_grid_strength_n_cells": int(np.sum(strength_mask)),
        "spatial_grid_orientation_n_cells": int(np.sum(orient_mask)),
        "spatial_grid_strength_weighted_orientation_n_cells": int(np.sum(weighted_mask)),
    }

# ============================================================
# EVALUATION
# ============================================================
def evaluate_group(group_id: int) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, object]]:
    raw = load_group_data(group_id)
    common_genes = common_gene_intersection([raw[m] for m in PLOT_METHOD_ORDER])
    qc_df = build_gene_qc_table(raw["Real"], common_genes)
    qc_df["group_id"] = int(group_id)

    marker_genes = [g for g in MARKER_GENES if g in set(common_genes)]
    marker_genes = [g for g in marker_genes if bool(qc_df.loc[qc_df["gene"] == g, "pass_qc"].any())]
    top_genes = select_top_genes(qc_df, marker_genes, top_n=TOP_N_SELECTED_GENES)
    least_gene = top_genes[-1] if len(top_genes) > 0 else None
    eval_genes = list(dict.fromkeys(marker_genes + top_genes))

    qc_df["gene_set"] = np.where(qc_df["gene"].isin(marker_genes), "marker",
                           np.where(qc_df["gene"].isin(top_genes), "top", "other"))
    qc_df.to_csv(os.path.join(OUTDIR, f"patient_{group_id}_gene_qc.csv"), index=False)
    print(f"[INFO] patient {group_id}: {len(marker_genes)} marker genes, {len(top_genes)} top genes selected")

    data = {k: normalize_total_then_log1p(v) for k, v in raw.items()}
    data = {k: subset_to_genes(v, eval_genes) for k, v in data.items()}

    store_genes = set(marker_genes)
    if least_gene is not None:
        store_genes.add(least_gene)
    field_bundle: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    rows: List[Dict[str, object]] = []

    for gene in eval_genes:
        qc_row = qc_df.loc[qc_df["gene"] == gene].iloc[0].to_dict()
        gene_set = "marker" if gene in marker_genes else "top"
        temp_fields: Dict[str, Dict[str, np.ndarray]] = {}
        for method in PLOT_METHOD_ORDER:
            temp_fields[method] = compute_marker_field(data[method], method, gene)
        real_field = temp_fields["Real"]

        gene_store = {"gene_qc": qc_row} if gene in store_genes else None
        if gene_store is not None:
            gene_store.update({m: temp_fields[m] for m in PLOT_METHOD_ORDER})

        for cand in METHOD_ORDER:
            cand_field = temp_fields[cand]
            transition = search_transition_similarity(
                real_field["grid_strength"], np.asarray(real_field.get("eval_mask", real_field["grid_mask"]), dtype=bool),
                cand_field["grid_strength"], np.asarray(cand_field.get("eval_mask", cand_field["grid_mask"]), dtype=bool),
                rigid_hint=None,
            )
            spatial_metrics = spatially_ordered_grid_pearsons(real_field, cand_field)
            row = {
                "group_id": int(group_id),
                "gene": gene,
                "gene_set": gene_set,
                "candidate": cand,
                "selection_score": float(qc_row.get("selection_score", np.nan)),
                "real_total_count": float(qc_row.get("real_total_count", np.nan)),
                "real_detected_spots": int(qc_row.get("real_detected_spots", 0)),
                "real_detected_fraction": float(qc_row.get("real_detected_fraction", np.nan)),
                "transition_ncc": float(transition.get("best_ncc", np.nan)),
                "transition_best_angle": float(transition.get("best_angle", np.nan)),
                "transition_best_shift_x": float(transition.get("best_shift_x", np.nan)),
                "transition_best_shift_y": float(transition.get("best_shift_y", np.nan)),
                "transition_overlap_n": int(transition.get("overlap_n", 0)),
                **spatial_metrics,
            }
            rows.append(row)
            if gene_store is not None:
                gene_store[f"{cand}_spatial_metrics"] = row.copy()
        if gene_store is not None:
            field_bundle[gene] = gene_store

    info = {
        "marker_genes": marker_genes,
        "top_genes": top_genes,
        "least_gene": least_gene,
        "qc_table": qc_df,
    }
    return pd.DataFrame(rows), field_bundle, info


CASE_BUNDLE_PKL = Path(OUTDIR) / "case_bundle.pkl"
INFO_BUNDLE_PKL = Path(OUTDIR) / "info_bundle.pkl"


def run_all(
    force_recompute: bool = False,
) -> Tuple[
    pd.DataFrame,
    Dict[int, Dict[str, Dict[str, Dict[str, np.ndarray]]]],
    Dict[int, Dict[str, object]],
]:
    """
    Evaluate all SCC groups or load previously cached outputs.

    Parameters
    ----------
    force_recompute
        If True, ignore existing output files and rerun the full evaluation.

    Returns
    -------
    metrics_df
        Combined evaluation metrics.

    case_bundle
        Per-group spatial fields used for case-study visualization.

    info_bundle
        Per-group auxiliary information and QC results.
    """
    ensure_dir(OUTDIR)

    metrics_path = Path(METRICS_CSV)
    qc_path = Path(TOP_GENE_QC_CSV)

    required_cache_files = [
        metrics_path,
        qc_path,
        CASE_BUNDLE_PKL,
        INFO_BUNDLE_PKL,
    ]

    # ------------------------------------------------------------------
    # Load existing results
    # ------------------------------------------------------------------
    cache_is_complete = all(path.exists() for path in required_cache_files)

    if cache_is_complete and not force_recompute:
        print("[INFO] Existing evaluation outputs found. Loading cached results.")

        metrics_df = pd.read_csv(metrics_path)

        with CASE_BUNDLE_PKL.open("rb") as file:
            case_bundle = pickle.load(file)

        with INFO_BUNDLE_PKL.open("rb") as file:
            info_bundle = pickle.load(file)

        print(f"[OK] loaded metrics: {metrics_path}")
        print(f"[OK] loaded QC table: {qc_path}")
        print(f"[OK] loaded case bundle: {CASE_BUNDLE_PKL}")
        print(f"[OK] loaded info bundle: {INFO_BUNDLE_PKL}")

        return metrics_df, case_bundle, info_bundle

    # ------------------------------------------------------------------
    # Run full evaluation
    # ------------------------------------------------------------------
    missing_files = [
        str(path)
        for path in required_cache_files
        if not path.exists()
    ]

    if force_recompute:
        print("[INFO] force_recompute=True; rerunning full evaluation.")
    else:
        print("[INFO] Cached outputs are incomplete; running full evaluation.")
        if missing_files:
            print("[INFO] Missing cache files:")
            for path in missing_files:
                print(f"       {path}")

    all_rows = []
    qc_tables = []

    case_bundle: Dict[
        int,
        Dict[str, Dict[str, Dict[str, np.ndarray]]],
    ] = {}

    info_bundle: Dict[int, Dict[str, object]] = {}

    for gid in GROUP_IDS:
        gid = int(gid)

        print(
            f"[INFO] evaluating SCC gene-transition fidelity "
            f"for patient {gid}"
        )

        df, fields, info = evaluate_group(gid)

        if df is not None and not df.empty:
            all_rows.append(df)

        case_bundle[gid] = fields
        info_bundle[gid] = info

        if (
            isinstance(info, dict)
            and isinstance(info.get("qc_table"), pd.DataFrame)
        ):
            qc_table = info["qc_table"]

            if not qc_table.empty:
                qc_tables.append(qc_table)

    # ------------------------------------------------------------------
    # Combine outputs
    # ------------------------------------------------------------------
    metrics_df = (
        pd.concat(
            all_rows,
            axis=0,
            ignore_index=True,
        )
        if all_rows
        else pd.DataFrame()
    )

    qc_df = (
        pd.concat(
            qc_tables,
            axis=0,
            ignore_index=True,
        )
        if qc_tables
        else pd.DataFrame()
    )

    # ------------------------------------------------------------------
    # Save all cache files
    # ------------------------------------------------------------------
    metrics_df.to_csv(
        metrics_path,
        index=False,
    )
    print(f"[OK] saved: {metrics_path}")

    # Always save the QC CSV, including when it is empty. This ensures that
    # the cache-completeness check behaves consistently on subsequent runs.
    qc_df.to_csv(
        qc_path,
        index=False,
    )
    print(f"[OK] saved: {qc_path}")

    with CASE_BUNDLE_PKL.open("wb") as file:
        pickle.dump(
            case_bundle,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[OK] saved: {CASE_BUNDLE_PKL}")

    with INFO_BUNDLE_PKL.open("wb") as file:
        pickle.dump(
            info_bundle,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[OK] saved: {INFO_BUNDLE_PKL}")

    return metrics_df, case_bundle, info_bundle

# ============================================================
# PLOTTING CONFIGURATION
# ============================================================
# None -> plot all cached marker genes. Tuple -> plot only selected cases.
CASE_GENES: Optional[Tuple[str, ...]] = ("KRT1", "COL1A1")


@dataclass(frozen=True)
class LegendStyle:
    """Reusable legend positioning and typography controls."""

    loc: str = "upper left"
    bbox_to_anchor: Optional[Tuple[float, float]] = None
    ncol: int = 2
    fontsize: float = 12
    frameon: bool = False
    handletextpad: float = 0.55
    columnspacing: float = 1.25
    labelspacing: float = 0.35
    borderaxespad: float = 0.0


@dataclass(frozen=True)
class SummaryStyle:
    """All layout controls for the per-patient metric-summary figure."""

    figsize: Tuple[float, float] = (3.8, 12)
    dpi: int = 350

    figure_title_fontsize: float = 18.0
    figure_title_weight: str = "normal"
    figure_title_y: float = 0.99
    panel_title_fontsize: float = 16.0
    panel_title_weight: str = "normal"
    panel_title_pad: float = 8.0
    axis_label_fontsize: float = 16
    tick_fontsize: float = 16

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
    bottom: float = 0.15
    top: float = 0.95
    hspace: float = 0.15

    legend: LegendStyle = field(
        default_factory=lambda: LegendStyle(
            loc="upper left",
            bbox_to_anchor=(0., 1.06),
            ncol=1,
            fontsize=16,
            columnspacing=1.25,
        )
    )


@dataclass(frozen=True)
class CaseStyle:
    """All layout controls for the 3 x 3 marker-gene case figure."""

    figsize: Tuple[float, float] = (11.2, 10.5)
    dpi: int = 350

    # Spatial maps.
    expression_point_size: float = 25.0
    gradient_point_size: float = 25.0
    invert_y: bool = True

    # Preserve the original spot-gradient streamline visualization.
    gradient_overlay_stream: bool = True
    stream_grid_bins: int = STREAM_GRID_BINS
    stream_mag_percentile: float = STREAM_MAG_PERCENTILE
    stream_density: float = 1.0
    stream_linewidth: float = 0.85
    stream_arrowsize: float = 0.8

    # Orientation field; defaults preserve the original plot.
    orientation_step: int = 3
    orientation_strength_percentile: float = 60.0
    orientation_linewidth: float = 0.60
    orientation_line_alpha: float = 0.62
    orientation_segment_scale: float = 0.78

    # Typography. All titles intentionally use regular weight.
    figure_title_fontsize: float = 18.0
    figure_title_weight: str = "normal"
    figure_title_y: float = 0.985
    column_title_fontsize: float = 18
    column_title_weight: str = "normal"
    column_title_pad: float = 5.0
    row_label_fontsize: float = 18
    row_label_weight: str = "normal"
    row_label_pad: float = 12.0

    # Metric annotation on orientation maps.
    annotation_fontsize: float =13
    # annotation_xy: Tuple[float, float] = (0.98, 0.97)
    annotation_xy: Tuple[float, float] = (0.05, 0.97)
    annotation_box_alpha: float = 0.94

    # Colorbars.
    colorbar_width_ratio: float = 0.06
    colorbar_length_ratio: float = 0.72  # 1.0 = full row height; centered vertically
    colorbar_label_fontsize: float = 15
    colorbar_tick_fontsize: float = 14

    # Figure geometry.
    left: float = 0.06
    right: float = 0.94
    bottom: float = 0.055
    top: float = 0.93
    hspace: float = 0.03
    wspace: float = -0.02


SUMMARY_STYLE = SummaryStyle()
CASE_STYLE = CaseStyle()


# ============================================================
# PUBLICATION STYLE
# ============================================================
def set_publication_style() -> None:
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


def make_vlag_red_cmap() -> LinearSegmentedColormap:
    """Return only the white-to-red half of seaborn's vlag colormap."""
    vlag = sns.color_palette("vlag", as_cmap=True)
    colors = vlag(np.linspace(0.50, 1.00, 256))
    return LinearSegmentedColormap.from_list("vlag_red", colors, N=256)


def make_gradient_cmap() -> LinearSegmentedColormap:
    colors = [
        "#fbfcfd", "#e6f2ef", "#c8e6dc", "#8fcdbf",
        "#4da6a0", "#2e7ca8", "#173f7a",
    ]
    return LinearSegmentedColormap.from_list(
        "cell_gradient_teal_blue", colors, N=256
    )


def make_orientation_cmap():
    return plt.get_cmap("hsv")


def _style_spatial_axis(ax, invert_y: bool = True) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if invert_y:
        ax.invert_yaxis()


# ============================================================
# CACHE LOADING FOR PLOT MODE
# ============================================================
def require_cached_results() -> None:
    required = [Path(METRICS_CSV), CASE_BUNDLE_PKL, INFO_BUNDLE_PKL]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing cached evaluation output(s):\n  " + "\n  ".join(missing)
            + "\nRun with --mode eval or --mode both first."
        )


def load_cached_results():
    require_cached_results()
    metrics_df = pd.read_csv(METRICS_CSV)
    with Path(CASE_BUNDLE_PKL).open("rb") as handle:
        case_bundle = pickle.load(handle)
    with Path(INFO_BUNDLE_PKL).open("rb") as handle:
        info_bundle = pickle.load(handle)
    return metrics_df, case_bundle, info_bundle


# ============================================================
# METRIC SUMMARY
# ============================================================
SUMMARY_METRICS = (
    ("transition_ncc", "Gradient-map similarity"),
    ("spatial_grid_orientation_pearson", "Orientation similarity"),
    (
        "spatial_grid_strength_weighted_orientation_pearson",
        "Weighted orientation similarity",
    ),

)


def _summary_gene_set_legend(
    style: SummaryStyle,
    gene_counts: Mapping[str, int],
):
    """Neutral legend: fill encodes gene set; x position/hue encodes method."""
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
            label=f"Top {gene_counts.get('top', 0)} HEGs",
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
    """Marker=filled method-colored box; top100=unfilled box; all edges black."""
    rng = np.random.default_rng(seed)
    center_positions = np.arange(len(methods), dtype=float)
    offsets = {
        "marker": -style.gene_set_offset,
        "top": style.gene_set_offset,
    }

    for method_idx, method in enumerate(methods):
        method_df = data[data["candidate"].astype(str) == str(method)]
        method_color = METHOD_PALETTE[method]

        for gene_set in ("marker", "top"):
            values = (
                method_df.loc[
                    method_df["gene_set"].astype(str) == gene_set,
                    metric,
                ]
                .astype(float)
                .to_numpy()
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            position = float(method_idx) + offsets[gene_set]
            marker_set = gene_set == "marker"
            facecolor = method_color if marker_set else "none"

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
                    "alpha": style.marker_box_alpha if marker_set else 1.0,
                },
                medianprops={"color": "black", "linewidth": style.median_width},
                whiskerprops={"color": "black", "linewidth": style.whisker_width},
                capprops={"color": "black", "linewidth": style.whisker_width},
                zorder=2,
            )

            jitter = (
                rng.uniform(-style.point_jitter, style.point_jitter, size=values.size)
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
                    if marker_set
                    else style.top_point_alpha
                ),
                rasterized=True,
                zorder=3,
            )

    ax.set_xlim(-0.55, len(methods) - 0.45)
    ax.set_ylim(-0.3, 1.1)
    ax.set_xticks(center_positions)
    ax.set_xticklabels(methods, fontsize=style.tick_fontsize)
    ax.tick_params(axis="y", labelsize=style.tick_fontsize)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(axis="y", linewidth=0.6, color="0.87", alpha=style.grid_alpha, zorder=0)
    ax.grid(axis="x", visible=False)


def plot_patient_summary(
    metrics_df: pd.DataFrame,
    group_id: int,
    outdir: str = OUTDIR,
    style: SummaryStyle = SUMMARY_STYLE,
) -> Optional[pd.DataFrame]:
    patient_df = metrics_df[metrics_df["group_id"] == int(group_id)].copy()
    if patient_df.empty:
        print(f"[WARN] no metrics for P{group_id}")
        return None

    methods = [
        method for method in METHOD_ORDER
        if method in set(patient_df["candidate"].astype(str))
    ]
    if not methods:
        print(f"[WARN] no candidate metrics for P{group_id}")
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
        if metric not in patient_df.columns:
            ax.text(
                0.5, 0.5, "Metric unavailable",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=style.axis_label_fontsize,
                color="0.45",
            )
            ax.set_axis_off()
            continue

        _plot_summary_box_pair(
            ax=ax,
            data=patient_df,
            metric=metric,
            methods=methods,
            style=style,
            seed=int(group_id) * 100 + metric_idx,
        )
        if metric_idx==0:
            ax.set_ylim(0., 1.1)
        # ax.set_title(
        #     f"P{group_id} | " + title,
        #     fontsize=style.panel_title_fontsize,
        #     fontweight=style.panel_title_weight,
        #     pad=style.panel_title_pad,
        # )
        ax.set_ylabel(title, fontsize=style.axis_label_fontsize)
        if metric_idx < len(SUMMARY_METRICS) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Method", fontsize=style.axis_label_fontsize)

        gene_counts = {
            gene_set: int(
                patient_df.loc[
                    patient_df["gene_set"].astype(str) == gene_set,
                    "gene",
                ].nunique()
            )
            for gene_set in ("marker", "top")
        }
        legend_kwargs = dict(
            handles=_summary_gene_set_legend(style, gene_counts),
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



    fig.suptitle(
        f"P{group_id}",
        fontsize=style.figure_title_fontsize,
        fontweight=style.figure_title_weight,
        y=style.figure_title_y,
    )
    fig.subplots_adjust(
        left=style.left,
        right=style.right,
        bottom=style.bottom,
        top=style.top,
        hspace=style.hspace,
    )

    ensure_dir(outdir)
    save_path = os.path.join(
        outdir,
        f"patient_{group_id}_transition_metric_summary.png",
    )
    fig.savefig(
        save_path,
        dpi=style.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"[OK] saved: {save_path}")
    plt.close(fig)

    summary_rows = []
    for gene_set in ("marker", "top"):
        set_df = patient_df[patient_df["gene_set"].astype(str) == gene_set]
        for method in methods:
            sub = set_df[set_df["candidate"].astype(str) == str(method)]
            row = {
                "group_id": int(group_id),
                "gene_set": gene_set,
                "candidate": method,
                "n_genes": int(sub["gene"].nunique()),
            }
            for metric, _ in SUMMARY_METRICS:
                values = (
                    sub[metric].astype(float).to_numpy()
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
    return pd.DataFrame(summary_rows)


def plot_patient_summaries(
    metrics_df: pd.DataFrame,
    outdir: str = OUTDIR,
    style: SummaryStyle = SUMMARY_STYLE,
) -> None:
    summary_frames = []
    for group_id in GROUP_IDS:
        summary = plot_patient_summary(
            metrics_df,
            group_id=int(group_id),
            outdir=outdir,
            style=style,
        )
        if summary is not None and not summary.empty:
            summary_frames.append(summary)

    if summary_frames:
        pd.concat(summary_frames, axis=0, ignore_index=True).to_csv(
            SUMMARY_CSV,
            index=False,
        )
        print(f"[OK] saved: {SUMMARY_CSV}")


# ============================================================
# CASE-STUDY PLOTTING
# ============================================================
def grid_vector_field(
    coords: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    bins: int = STREAM_GRID_BINS,
):
    """Same streamline field construction as the supplied plotting code."""
    coords_n = normalized_xy(coords)
    xbin = np.clip((coords_n[:, 0] * bins).astype(int), 0, bins - 1)
    ybin = np.clip((coords_n[:, 1] * bins).astype(int), 0, bins - 1)
    gx_num = np.zeros((bins, bins), dtype=np.float64)
    gy_num = np.zeros((bins, bins), dtype=np.float64)
    den = np.zeros((bins, bins), dtype=np.float64)
    finite = np.isfinite(gx) & np.isfinite(gy)
    for xb, yb, g1, g2 in zip(
        xbin[finite], ybin[finite], gx[finite], gy[finite]
    ):
        gx_num[yb, xb] += float(g1)
        gy_num[yb, xb] += float(g2)
        den[yb, xb] += 1.0
    gxg = np.divide(
        gx_num,
        np.maximum(den, 1.0),
        out=np.zeros_like(gx_num),
        where=den > 0,
    )
    gyg = np.divide(
        gy_num,
        np.maximum(den, 1.0),
        out=np.zeros_like(gy_num),
        where=den > 0,
    )
    gxg = gaussian_filter(gxg, sigma=GRID_SMOOTH_SIGMA, mode="nearest")
    gyg = gaussian_filter(gyg, sigma=GRID_SMOOTH_SIGMA, mode="nearest")
    mag = np.sqrt(gxg ** 2 + gyg ** 2)
    xv = (np.arange(bins) + 0.5) / bins
    yv = (np.arange(bins) + 0.5) / bins
    return xv, yv, gxg, gyg, mag


def plot_expression_axis(
    ax,
    field: Mapping[str, np.ndarray],
    cmap,
    vmin: float,
    vmax: float,
    style: CaseStyle,
):
    coords = np.asarray(field["coords_raw"], dtype=np.float64)
    vals = np.asarray(field["expr_raw"], dtype=np.float64)
    artist = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=vals,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=style.expression_point_size,
        edgecolors="none",
        rasterized=True,
    )
    _style_spatial_axis(ax, style.invert_y)
    return artist


def plot_spot_gradient_axis(
    ax,
    field: Mapping[str, np.ndarray],
    cmap,
    vmin: float,
    vmax: float,
    style: CaseStyle,
):
    """Original spot-gradient rendering, including black streamlines."""
    coords = np.asarray(field["coords_raw"], dtype=np.float64)
    vals = np.asarray(field["strength_trend"], dtype=np.float64)
    artist = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=vals,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=style.gradient_point_size,
        edgecolors="none",
        rasterized=True,
    )

    if style.gradient_overlay_stream:
        xv, yv, gxg, gyg, mag = grid_vector_field(
            coords,
            np.asarray(field["gx_trend"], dtype=np.float64),
            np.asarray(field["gy_trend"], dtype=np.float64),
            bins=style.stream_grid_bins,
        )
        if np.any(np.isfinite(mag)):
            threshold = np.nanpercentile(
                mag[np.isfinite(mag)],
                style.stream_mag_percentile,
            )
            gxg = np.where(mag >= threshold, gxg, 0.0)
            gyg = np.where(mag >= threshold, gyg, 0.0)

        xmn, xmx = float(np.nanmin(coords[:, 0])), float(np.nanmax(coords[:, 0]))
        ymn, ymx = float(np.nanmin(coords[:, 1])), float(np.nanmax(coords[:, 1]))
        xplot = xmn + xv * (xmx - xmn)
        yplot = ymn + yv * (ymx - ymn)
        try:
            ax.streamplot(
                xplot,
                yplot,
                gxg,
                gyg,
                density=style.stream_density,
                color="black",
                linewidth=style.stream_linewidth,
                arrowsize=style.stream_arrowsize,
            )
        except Exception:
            pass

    _style_spatial_axis(ax, style.invert_y)
    return artist


def plot_orientation_field_axis(
    ax,
    field: Mapping[str, np.ndarray],
    style: CaseStyle,
    annotation_lines: Optional[Sequence[str]] = None,
) -> None:
    """Original orientation-field visual encoding with configurable layout."""
    orient_cmap = make_orientation_cmap()
    gx = np.asarray(field["grid_gx"], dtype=np.float64)
    gy = np.asarray(field["grid_gy"], dtype=np.float64)
    grid_mask = np.asarray(
        field.get("eval_mask", field["grid_mask"]),
        dtype=bool,
    )
    coords = np.asarray(field["coords_raw"], dtype=np.float64)
    xmn, xmx = float(np.nanmin(coords[:, 0])), float(np.nanmax(coords[:, 0]))
    ymn, ymx = float(np.nanmin(coords[:, 1])), float(np.nanmax(coords[:, 1]))

    theta = np.mod(np.arctan2(gy, gx), np.pi)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    valid = grid_mask & np.isfinite(theta) & np.isfinite(mag) & (mag > 0)

    if np.any(valid):
        mag_thr = float(
            np.nanpercentile(
                mag[valid],
                np.clip(style.orientation_strength_percentile, 0.0, 100.0),
            )
        )
        mag_hi = float(np.nanpercentile(mag[valid], 98.5))
        if not np.isfinite(mag_hi) or mag_hi <= 0:
            mag_hi = float(np.nanmax(mag[valid]))
        alpha = np.clip(
            (mag - mag_thr) / max(mag_hi - mag_thr, 1e-8),
            0.0,
            1.0,
        )
        alpha = 0.10 + 0.90 * alpha
        alpha[~valid] = 0.0
    else:
        alpha = np.zeros_like(mag, dtype=np.float64)

    rgba = orient_cmap(np.clip(theta / np.pi, 0.0, 1.0))
    rgba[..., 3] = alpha
    ax.imshow(
        rgba,
        origin="lower",
        extent=[xmn, xmx, ymn, ymx],
        aspect="equal",
        interpolation="nearest",
    )

    h, w = theta.shape
    dx = (xmx - xmn) / max(w, 1)
    dy = (ymx - ymn) / max(h, 1)
    segment_length = style.orientation_segment_scale * min(dx, dy)
    segments = []
    glyph_mask = valid & (alpha > 0.18)
    step = max(1, int(style.orientation_step))
    for iy in range(0, h, step):
        for ix in range(0, w, step):
            if not glyph_mask[iy, ix]:
                continue
            angle = float(theta[iy, ix])
            cx = xmn + (ix + 0.5) * dx
            cy = ymn + (iy + 0.5) * dy
            ux = np.cos(angle)
            uy = np.sin(angle)
            segments.append([
                (
                    cx - 0.5 * segment_length * ux,
                    cy - 0.5 * segment_length * uy,
                ),
                (
                    cx + 0.5 * segment_length * ux,
                    cy + 0.5 * segment_length * uy,
                ),
            ])

    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors="black",
                linewidths=style.orientation_linewidth,
                alpha=style.orientation_line_alpha,
            )
        )

    _style_spatial_axis(ax, style.invert_y)
    if annotation_lines:
        ax.text(
            style.annotation_xy[0],
            style.annotation_xy[1],
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            # ha="right",
            ha="left",
            va="top",
            fontsize=style.annotation_fontsize,
            bbox=dict(
                boxstyle="round,pad=0.24",
                facecolor="white",
                edgecolor="0.82",
                alpha=style.annotation_box_alpha,
            ),
        )


def _pool_finite(
    fields: Sequence[Mapping[str, np.ndarray]],
    key: str,
) -> np.ndarray:
    values = []
    for field in fields:
        arr = np.asarray(field.get(key, np.array([])), dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    return np.concatenate(values) if values else np.asarray([], dtype=np.float64)


def _candidate_annotation(metric_row: Mapping[str, object]) -> Sequence[str]:
    ncc = float(metric_row.get("transition_ncc", np.nan))
    weighted = float(
        metric_row.get(
            "spatial_grid_strength_weighted_orientation_pearson",
            np.nan,
        )
    )
    orient = float(metric_row.get("spatial_grid_orientation_pearson", np.nan))
    return (
        f"NCC = {ncc:.3f}" if np.isfinite(ncc) else "NCC = NA",
        f"r(w) = {weighted:.3f}" if np.isfinite(weighted) else "wOri r = NA",
        f"r = {orient:.3f}" if np.isfinite(orient) else "Ori r = NA",
    )


def _set_vertical_colorbar_length(cax, length_ratio: float) -> None:
    """Center-shrink a vertical colorbar axis while preserving its width."""
    ratio = float(np.clip(length_ratio, 0.05, 1.0))
    pos = cax.get_position()
    new_height = pos.height * ratio
    new_y0 = pos.y0 + 0.5 * (pos.height - new_height)
    cax.set_position([pos.x0, new_y0, pos.width, new_height])


def plot_combined_case(
    group_id: int,
    gene: str,
    gene_bundle: Mapping[str, object],
    outdir: str = OUTDIR,
    style: CaseStyle = CASE_STYLE,
) -> None:
    """3 x 3 case with rows=representation and columns=method."""
    set_publication_style()

    fields = {method: gene_bundle[method] for method in PLOT_METHOD_ORDER}
    field_list = [fields[method] for method in PLOT_METHOD_ORDER]
    expr_cmap = "Reds"
    grad_cmap = make_gradient_cmap()

    expr_pool = _pool_finite(field_list, "expr_raw")
    expr_vmax = (
        max(float(np.nanpercentile(expr_pool, 99.0)), 1e-6)
        if expr_pool.size
        else 1.0
    )

    # Preserve the original shared gradient scale across both spot and grid strength.
    spot_pool = _pool_finite(field_list, "strength_trend")
    map_pool = _pool_finite(field_list, "grid_strength")
    grad_vmax = max(
        float(np.nanpercentile(spot_pool, 98.5)) if spot_pool.size else 0.0,
        float(np.nanpercentile(map_pool, 98.5)) if map_pool.size else 0.0,
        1e-6,
    )

    fig = plt.figure(figsize=style.figsize)
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.0, 1.0, 1.0, style.colorbar_width_ratio],
        left=style.left,
        right=style.right,
        bottom=style.bottom,
        top=style.top,
        hspace=style.hspace,
        wspace=style.wspace,
    )

    axes = np.empty((3, 3), dtype=object)
    for row in range(3):
        for col in range(3):
            axes[row, col] = fig.add_subplot(gs[row, col])

    # Row 1: all gene-expression maps.
    for col, method in enumerate(PLOT_METHOD_ORDER):
        plot_expression_axis(
            axes[0, col],
            fields[method],
            cmap=expr_cmap,
            vmin=0.0,
            vmax=expr_vmax,
            style=style,
        )
        if method=="Real":
            axes[0, col].set_title(
                "Measured (held-out)",
                fontsize=style.column_title_fontsize,
                fontweight=style.column_title_weight,
                pad=style.column_title_pad,
            )
        else:
            axes[0, col].set_title(
                method,
                fontsize=style.column_title_fontsize,
                fontweight=style.column_title_weight,
                pad=style.column_title_pad,
            )

    # Row 2: all spot-gradient maps, retaining the original streamlines.
    for col, method in enumerate(PLOT_METHOD_ORDER):
        plot_spot_gradient_axis(
            axes[1, col],
            fields[method],
            cmap=grad_cmap,
            vmin=0.0,
            vmax=grad_vmax,
            style=style,
        )

    # Row 3: all spatial orientation fields, retaining the original annotations.
    for col, method in enumerate(PLOT_METHOD_ORDER):
        if method == "Real":
            annotation = ["Measured reference"]
            pass
        else:
            annotation = _candidate_annotation(
                gene_bundle.get(f"{method}_spatial_metrics", {})
            )
        plot_orientation_field_axis(
            axes[2, col],
            fields[method],
            style=style,
            annotation_lines=annotation,
        )

    row_labels = (
        "Gene expression",
        "Expression gradient",
        "Gradient orientation",
    )
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(
            label,
            fontsize=style.row_label_fontsize,
            fontweight=style.row_label_weight,
            labelpad=style.row_label_pad,
        )

    # Vertical colorbars match the three representation rows.
    cax_expr = fig.add_subplot(gs[0, 3])
    _set_vertical_colorbar_length(cax_expr, style.colorbar_length_ratio)
    cb_expr = fig.colorbar(
        plt.cm.ScalarMappable(
            cmap=expr_cmap,
            norm=Normalize(vmin=0.0, vmax=expr_vmax),
        ),
        cax=cax_expr,
    )
    cb_expr.set_label(f"log1p expression", fontsize=style.colorbar_label_fontsize)
    cb_expr.ax.tick_params(labelsize=style.colorbar_tick_fontsize)

    cax_grad = fig.add_subplot(gs[1, 3])
    _set_vertical_colorbar_length(cax_grad, style.colorbar_length_ratio)
    cb_grad = fig.colorbar(
        plt.cm.ScalarMappable(
            cmap=grad_cmap,
            norm=Normalize(vmin=0.0, vmax=grad_vmax),
        ),
        cax=cax_grad,
    )
    cb_grad.set_label("Gradient strength", fontsize=style.colorbar_label_fontsize)
    cb_grad.ax.tick_params(labelsize=style.colorbar_tick_fontsize)

    cax_orientation = fig.add_subplot(gs[2, 3])
    _set_vertical_colorbar_length(cax_orientation, style.colorbar_length_ratio)
    cb_orientation = fig.colorbar(
        plt.cm.ScalarMappable(
            cmap=make_orientation_cmap(),
            norm=Normalize(vmin=0.0, vmax=180.0),
        ),
        cax=cax_orientation,
        ticks=[0, 45, 90, 135, 180],
    )
    cb_orientation.set_label(
        "Gradient orientation (°)",
        fontsize=style.colorbar_label_fontsize,
    )
    cb_orientation.ax.tick_params(labelsize=style.colorbar_tick_fontsize)
    if gene =="KRT1":
        fig.suptitle(
            f"{gene} | epithelial marker",
            fontsize=style.figure_title_fontsize,
            fontweight=style.figure_title_weight,
            y=style.figure_title_y,
        )
    else:
        fig.suptitle(
            f"{gene} | stromal marker",
            fontsize=style.figure_title_fontsize,
            fontweight=style.figure_title_weight,
            y=style.figure_title_y,
        )

    # Save directly in OUTDIR; no patient-specific case folder is created.
    ensure_dir(outdir)
    save_path = os.path.join(
        outdir,
        f"P{group_id}_{gene}_marker_transition_case.png",
    )
    fig.savefig(
        save_path,
        dpi=style.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"[OK] saved: {save_path}")
    plt.close(fig)


def make_case_figures(
    case_bundle: Mapping[int, Mapping[str, object]],
    info_bundle: Mapping[int, Mapping[str, object]],
    outdir: str = OUTDIR,
    style: CaseStyle = CASE_STYLE,
) -> None:
    ensure_dir(outdir)
    for group_id in CASE_GROUPS:
        if group_id not in case_bundle:
            print(f"[WARN] P{group_id} missing from case bundle")
            continue

        cached_marker_genes = list(
            info_bundle.get(group_id, {}).get("marker_genes", [])
        )
        selected_genes = (
            cached_marker_genes
            if CASE_GENES is None
            else list(CASE_GENES)
        )

        for gene in selected_genes:
            if gene not in case_bundle[group_id]:
                print(f"[WARN] P{group_id} | {gene}: cached field unavailable")
                continue
            plot_combined_case(
                group_id=int(group_id),
                gene=str(gene),
                gene_bundle=case_bundle[group_id][gene],
                outdir=outdir,
                style=style,
            )


# ============================================================
# EVAL / PLOT / BOTH CONTROL
# ============================================================
def plot_saved_results(
    metrics_df: Optional[pd.DataFrame] = None,
    case_bundle: Optional[Mapping[int, Mapping[str, object]]] = None,
    info_bundle: Optional[Mapping[int, Mapping[str, object]]] = None,
) -> None:
    if metrics_df is None or case_bundle is None or info_bundle is None:
        metrics_df, case_bundle, info_bundle = load_cached_results()

    plot_patient_summaries(
        metrics_df=metrics_df,
        outdir=OUTDIR,
        style=SUMMARY_STYLE,
    )
    make_case_figures(
        case_bundle=case_bundle,
        info_bundle=info_bundle,
        outdir=OUTDIR,
        style=CASE_STYLE,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="SCC spatial gene-pattern evaluation and plotting."
    )
    parser.add_argument(
        "--mode",
        choices=("eval", "plot", "both"),
        default="both",
        help=(
            "eval: ensure/cache numerical results; "
            "plot: plot existing cached results; "
            "both: evaluate/load cache and then plot"
        ),
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute evaluation even when all current cache files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUTDIR)

    metrics_df = case_bundle = info_bundle = None

    if args.mode in {"eval", "both"}:
        metrics_df, case_bundle, info_bundle = run_all(
            force_recompute=args.force_recompute,
        )

    if args.mode == "plot":
        metrics_df, case_bundle, info_bundle = load_cached_results()

    print("Patient 2 markers:", info_bundle[2]["marker_genes"])
    print("Patient 10 markers:", info_bundle[10]["marker_genes"])

    print("Patient 2 n =", len(info_bundle[2]["marker_genes"]))
    print("Patient 10 n =", len(info_bundle[10]["marker_genes"]))


    if args.mode in {"plot", "both"}:
        plot_saved_results(
            metrics_df=metrics_df,
            case_bundle=case_bundle,
            info_bundle=info_bundle,
        )

    print(f"[DONE] SCC workflow finished in mode={args.mode}.")


if __name__ == "__main__":
    main()
