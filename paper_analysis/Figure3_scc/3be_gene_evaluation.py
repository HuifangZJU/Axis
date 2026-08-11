#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light SCC marker-gene spatial-gradient evaluation.

Goal
----
Test marker-gene spatial-gradient fidelity before scaling to whole-gene metrics.
This script evaluates SCC marker genes in two complementary ways:

1) Tissue-masked gradient transition-map similarity
   - Build a tissue-masked image/grid of gradient strength.
   - Compare candidate (SpatialZ / AXIS) vs Real with image metrics.
   - Allow small rigid tolerance (rotation + translation) during the image-metric
     search so trivial global orientation mismatch does not dominate.
   - Save case-study figures with:
       * raw expression map
       * tissue-masked gradient transition map
       * NCC annotation on the gradient transition map

2) Histogram diagnostics
   - Histogram comparisons are reported with shifted Pearson.
   - Compare:
       * grid-strength histogram, reported by shifted Pearson
       * grid-orientation histogram, reported by shifted Pearson
       * HOG orientation histogram, reported by shifted Pearson

This is a light version for marker genes only. If the marker-gene behavior is
useful, the same logic can be extended to whole-gene evaluation later.
"""

from __future__ import annotations
from matplotlib.lines import Line2D
import os
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

from anndata import AnnData
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, rotate as ndi_rotate, shift as ndi_shift, binary_dilation, binary_closing, binary_fill_holes, binary_erosion
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover
    skimage_ssim = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import pickle
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
ROOT = "/media/huifang/data/registration/SCC/huifang"
H5AD_DIR = os.path.join(ROOT, "h5ad")
VIRTUAL_DIR = os.path.join(ROOT, "virtual_tissue")
OUTDIR = "/media/huifang/data/registration/evaluation/virtual_generation/scc/"

GROUP_IDS = [2, 10]
CASE_GROUPS = [2, 10]

COORD_KEY_REAL = "spatial_image_coor"
COORD_KEY_VIRTUAL = "spatial"

NORMALIZE_TOTAL = True
TARGET_SUM = 1e4
LOG1P = True

METHOD_ORDER = ["SpatialZ", "AXIS"]
PLOT_METHOD_ORDER = ["Real", "SpatialZ", "AXIS"]
METHOD_PALETTE = {
    "Real": "#111111",
    "SpatialZ": "#7C8DA6",
    "AXIS": "#C4473A",
}

MARKER_GENES = [
    "KRT1", "KRT5", "KRT10", "KRT14", "KRT15",
    "COL1A1", "COL1A2", "DCN",
    "PECAM1", "VWF",
    "PTPRC", "CD3D", "MS4A1", "LYZ",
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
# STYLE
# ============================================================
def set_cell_style() -> None:
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
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })


def make_expression_cmap():
    colors = ["#fbfbfd", "#f4e6ec", "#e7c2d2", "#d589ad", "#b84c7a", "#772458"]
    return LinearSegmentedColormap.from_list("cell_marker_rose", colors, N=256)


def make_gradient_cmap():
    colors = ["#fbfcfd", "#e6f2ef", "#c8e6dc", "#8fcdbf", "#4da6a0", "#2e7ca8", "#173f7a"]
    return LinearSegmentedColormap.from_list("cell_gradient_teal_blue", colors, N=256)


def make_orientation_cmap():
    """High-contrast rainbow orientation palette for 0–180° axial angles."""
    return plt.cm.hsv


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


def safe_cosine(x: Sequence[float], y: Sequence[float], eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2:
        return np.nan
    x = x[keep]
    y = y[keep]
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den <= eps:
        return np.nan
    return float(np.dot(x, y) / den)


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




def apply_eval_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    out[~np.asarray(mask, dtype=bool)] = np.nan
    return out




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


def rotate_axial_histogram_by_angle(hist: np.ndarray, angle_deg: float) -> np.ndarray:
    hist = np.asarray(hist, dtype=np.float64)
    if hist.size == 0 or not np.all(np.isfinite(hist)):
        return np.full_like(hist, np.nan, dtype=np.float64)
    bin_width = 180.0 / hist.size
    shift_bins = int(np.round(angle_deg / bin_width))
    return np.roll(hist, shift_bins)


def circular_shift_similarity(real_hist: np.ndarray, cand_hist: np.ndarray) -> Tuple[float, float, int]:
    real_hist = np.asarray(real_hist, dtype=np.float64)
    cand_hist = np.asarray(cand_hist, dtype=np.float64)
    if real_hist.size != cand_hist.size or real_hist.size == 0:
        return np.nan, np.nan, 0
    if not (np.all(np.isfinite(real_hist)) and np.all(np.isfinite(cand_hist))):
        return np.nan, np.nan, 0
    best_r = -np.inf
    best_cos = -np.inf
    best_shift = 0
    for s in range(real_hist.size):
        shifted = np.roll(cand_hist, s)
        r = safe_pearson(real_hist, shifted)
        c = safe_cosine(real_hist, shifted)
        score = r if np.isfinite(r) else -np.inf
        if score > best_r:
            best_r = score
            best_cos = c
            best_shift = s
    return float(best_r) if np.isfinite(best_r) else np.nan, float(best_cos) if np.isfinite(best_cos) else np.nan, int(best_shift)


def gene_qc_stats(adata: AnnData, gene: str) -> Dict[str, float]:
    x = extract_gene_vector(adata, gene).astype(np.float64)
    total = float(np.nansum(x))
    detected = int(np.sum(np.isfinite(x) & (x > 0)))
    frac = float(detected / max(adata.n_obs, 1))
    return {
        "real_total_count": total,
        "real_detected_spots": detected,
        "real_detected_fraction": frac,
    }


def keep_gene_by_qc(stats: Dict[str, float]) -> bool:
    return (
        stats["real_total_count"] >= MIN_REAL_TOTAL_COUNT and
        stats["real_detected_spots"] >= MIN_REAL_DETECTED_SPOTS and
        stats["real_detected_fraction"] >= MIN_REAL_DETECTED_FRAC
    )


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


def masked_ssim_map(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray]:
    if skimage_ssim is None:
        score = masked_ncc(a, b, mask)
        return score, np.where(mask, score, np.nan)

    mm = np.asarray(mask, dtype=bool)
    keep = mm & np.isfinite(a) & np.isfinite(b)
    if int(np.sum(keep)) < MIN_OVERLAP_PIXELS:
        return np.nan, np.full_like(a, np.nan, dtype=np.float64)

    yy, xx = np.where(keep)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    a_crop = np.asarray(a[y0:y1, x0:x1], dtype=np.float64)
    b_crop = np.asarray(b[y0:y1, x0:x1], dtype=np.float64)
    m_crop = np.asarray(keep[y0:y1, x0:x1], dtype=bool)

    fill_a = np.where(m_crop, a_crop, 0.0)
    fill_b = np.where(m_crop, b_crop, 0.0)
    vals = np.r_[fill_a[m_crop], fill_b[m_crop]]
    dr = float(np.nanmax(vals) - np.nanmin(vals)) if vals.size else 1.0
    dr = max(dr, 1e-5)
    win = min(SSIM_WIN, fill_a.shape[0] if fill_a.shape[0] % 2 == 1 else fill_a.shape[0] - 1,
              fill_a.shape[1] if fill_a.shape[1] % 2 == 1 else fill_a.shape[1] - 1)
    win = max(win, 5)
    if win % 2 == 0:
        win -= 1
    try:
        score, s_map = skimage_ssim(fill_a, fill_b, data_range=dr, win_size=win, gaussian_weights=True,
                                    sigma=SSIM_GAUSS_SIGMA, full=True)
    except Exception:
        score = masked_ncc(a, b, keep)
        return score, np.where(keep, score, np.nan)
    out = np.full_like(a, np.nan, dtype=np.float64)
    temp = np.where(m_crop, s_map, np.nan)
    out[y0:y1, x0:x1] = temp
    score_masked = float(np.nanmean(temp[m_crop])) if np.any(m_crop) else np.nan
    return score_masked, out


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


# ============================================================
# HISTOGRAM VISUALIZATION AND GRID-ORDERED METRICS
# ============================================================
def histogram_from_values(vals: np.ndarray, bins: int, vmin: float, vmax: float) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    edges = np.linspace(vmin, vmax, bins + 1)
    if vals.size == 0:
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers.astype(np.float32), np.full(bins, np.nan, dtype=np.float32)
    hist, _ = np.histogram(np.clip(vals, vmin, vmax), bins=edges)
    hist = hist.astype(np.float64)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.astype(np.float32), hist.astype(np.float32)


def strength_histogram(field: Dict[str, np.ndarray], bins: int = HIST_BINS_STRENGTH,
                       vmax: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    vals = apply_eval_mask(field["grid_strength"], field.get("eval_mask", field["grid_mask"]))
    vals = vals[np.isfinite(vals)]
    if vmax is None:
        vmax = max(float(np.nanpercentile(vals, 99.0)), 1e-6) if vals.size else 1.0
    return histogram_from_values(vals, bins=bins, vmin=0.0, vmax=float(vmax))


def grid_orientation_histogram(field: Dict[str, np.ndarray], angle_offset_deg: float = 0.0,
                               bins: int = HIST_BINS_ORIENTATION) -> np.ndarray:
    eval_mask = np.asarray(field.get("eval_mask", field["grid_mask"]), dtype=bool)
    gx = apply_eval_mask(field["grid_gx"], eval_mask)
    gy = apply_eval_mask(field["grid_gy"], eval_mask)
    s = np.sqrt(gx ** 2 + gy ** 2)
    keep = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(s) & (s > 0)
    if np.sum(keep) < 5:
        return np.full(bins, np.nan, dtype=np.float32)
    thresh = np.nanpercentile(s[keep], GRID_ORIENT_MIN_PERCENTILE)
    keep = keep & (s >= thresh)
    if np.sum(keep) < 5:
        return np.full(bins, np.nan, dtype=np.float32)
    theta = np.mod(np.arctan2(gy[keep], gx[keep]) + np.deg2rad(angle_offset_deg), np.pi)
    hist, _ = np.histogram(theta, bins=bins, range=(0.0, np.pi), weights=s[keep])
    hist = hist.astype(np.float64)
    if hist.sum() <= 0:
        return np.full(bins, np.nan, dtype=np.float32)
    return (hist / hist.sum()).astype(np.float32)


def hog_hist_after_angle(field: Dict[str, np.ndarray], angle_offset_deg: float = 0.0) -> np.ndarray:
    hist = np.asarray(field["hog_hist"], dtype=np.float64)
    if not np.all(np.isfinite(hist)):
        return np.full_like(hist, np.nan, dtype=np.float64)
    return rotate_axial_histogram_by_angle(hist, angle_offset_deg).astype(np.float32)


def histogram_similarity(real_hist: np.ndarray, cand_hist: np.ndarray) -> Dict[str, float]:
    """Histogram similarity with shifted Pearson as the primary reported score.

    For orientation/HOG histograms the shift is circular because the orientation
    axis is axial over [0, 180). For strength histograms, we use the same shifted
    comparison as a tolerance to small bin-wise scale displacement, as requested
    for the light diagnostic version.
    """
    r_shift, c_shift, best_bins = circular_shift_similarity(real_hist, cand_hist)
    return {
        "pearson_unshifted": safe_pearson(real_hist, cand_hist),
        "cosine_unshifted": safe_cosine(real_hist, cand_hist),
        "shifted_pearson": r_shift,
        "shifted_cosine": c_shift,
        "shifted_bins": best_bins,
        # Backward-compatible aliases are shifted by design.
        "pearson": r_shift,
        "cosine": c_shift,
        "best_shift_pearson": r_shift,
        "best_shift_cosine": c_shift,
        "best_shift_bins": best_bins,
    }


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
# PLOTTING HELPERS
# ============================================================
def grid_vector_field(coords: np.ndarray, gx: np.ndarray, gy: np.ndarray, bins: int = STREAM_GRID_BINS):
    coords_n = normalized_xy(coords)
    xbin = np.clip((coords_n[:, 0] * bins).astype(int), 0, bins - 1)
    ybin = np.clip((coords_n[:, 1] * bins).astype(int), 0, bins - 1)
    gx_num = np.zeros((bins, bins), dtype=np.float64)
    gy_num = np.zeros((bins, bins), dtype=np.float64)
    den = np.zeros((bins, bins), dtype=np.float64)
    finite = np.isfinite(gx) & np.isfinite(gy)
    for xb, yb, g1, g2 in zip(xbin[finite], ybin[finite], gx[finite], gy[finite]):
        gx_num[yb, xb] += float(g1)
        gy_num[yb, xb] += float(g2)
        den[yb, xb] += 1.0
    gxg = np.divide(gx_num, np.maximum(den, 1.0), out=np.zeros_like(gx_num), where=den > 0)
    gyg = np.divide(gy_num, np.maximum(den, 1.0), out=np.zeros_like(gy_num), where=den > 0)
    gxg = gaussian_filter(gxg, sigma=GRID_SMOOTH_SIGMA, mode='nearest')
    gyg = gaussian_filter(gyg, sigma=GRID_SMOOTH_SIGMA, mode='nearest')
    mag = np.sqrt(gxg**2 + gyg**2)
    xv = (np.arange(bins) + 0.5) / bins
    yv = (np.arange(bins) + 0.5) / bins
    return xv, yv, gxg, gyg, mag


def plot_expression_axis(ax, field: Dict[str, np.ndarray], title: str, cmap, vmin: float, vmax: float):
    coords = field["coords_raw"]
    vals = field["expr_raw"]
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                     s=11.0, edgecolors="none", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(title, fontsize=11.2, fontweight="bold")
    return sca


def plot_spot_gradient_axis(ax, field: Dict[str, np.ndarray], title: str, cmap, vmin: float, vmax: float,
                            overlay_stream: bool = True):
    coords = np.asarray(field["coords_raw"], dtype=np.float64)
    vals = np.asarray(field["strength_trend"], dtype=np.float64)
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                     s=11.0, edgecolors="none", rasterized=True)
    if overlay_stream:
        xv, yv, gxg, gyg, mag = grid_vector_field(coords, field["gx_trend"], field["gy_trend"], bins=STREAM_GRID_BINS)
        if np.any(np.isfinite(mag)):
            thr = np.nanpercentile(mag[np.isfinite(mag)], STREAM_MAG_PERCENTILE)
            gxg = np.where(mag >= thr, gxg, 0.0)
            gyg = np.where(mag >= thr, gyg, 0.0)
        xmn, xmx = float(np.nanmin(coords[:, 0])), float(np.nanmax(coords[:, 0]))
        ymn, ymx = float(np.nanmin(coords[:, 1])), float(np.nanmax(coords[:, 1]))
        xplot = xmn + xv * (xmx - xmn)
        yplot = ymn + yv * (ymx - ymn)
        try:
            ax.streamplot(xplot, yplot, gxg, gyg, density=1.0, color="black", linewidth=0.55, arrowsize=0.8)
        except Exception:
            pass
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(title, fontsize=11.2, fontweight="bold")
    return sca


def plot_transition_map_axis(ax, field: Dict[str, np.ndarray], title: str, cmap, vmin: float, vmax: float,
                             overlay_stream: bool = False):
    grid = np.asarray(field["grid_strength"], dtype=np.float64)
    coords = np.asarray(field["coords_raw"], dtype=np.float64)
    xmn, xmx = float(np.nanmin(coords[:, 0])), float(np.nanmax(coords[:, 0]))
    ymn, ymx = float(np.nanmin(coords[:, 1])), float(np.nanmax(coords[:, 1]))
    cmap_local = cmap.copy() if hasattr(cmap, "copy") else cmap
    try:
        cmap_local.set_bad(color="#ffffff", alpha=0.0)
    except Exception:
        pass
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap_local, vmin=vmin, vmax=vmax,
                   origin="lower", extent=[xmn, xmx, ymn, ymx], aspect="equal", interpolation="nearest")
    if overlay_stream:
        xv, yv, gxg, gyg, mag = grid_vector_field(coords, field["gx_trend"], field["gy_trend"], bins=STREAM_GRID_BINS)
        if np.any(np.isfinite(mag)):
            thr = np.nanpercentile(mag[np.isfinite(mag)], STREAM_MAG_PERCENTILE)
            gxg = np.where(mag >= thr, gxg, 0.0)
            gyg = np.where(mag >= thr, gyg, 0.0)
        xplot = xmn + xv * (xmx - xmn)
        yplot = ymn + yv * (ymx - ymn)
        try:
            ax.streamplot(xplot, yplot, gxg, gyg, density=1.0, color="black", linewidth=0.55, arrowsize=0.8)
        except Exception:
            pass
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(title, fontsize=11.2, fontweight="bold")
    return im


def plot_orientation_field_axis(ax, field: Dict[str, np.ndarray], title: str,
                                annotation_lines: Optional[Sequence[str]] = None,
                                step: int = 3, strength_percentile: float = 60.0):
    """2D spatial orientation field map.

    Color encodes orientation (0–180°) using a smooth blue→purple→pink scale,
    while transparency encodes gradient strength. Sparse black line glyphs
    provide explicit local orientation cues.
    """
    orient_cmap = make_orientation_cmap()
    gx = np.asarray(field['grid_gx'], dtype=np.float64)
    gy = np.asarray(field['grid_gy'], dtype=np.float64)
    grid_mask = np.asarray(field.get('eval_mask', field['grid_mask']), dtype=bool)
    coords = np.asarray(field['coords_raw'], dtype=np.float64)
    xmn, xmx = float(np.nanmin(coords[:, 0])), float(np.nanmax(coords[:, 0]))
    ymn, ymx = float(np.nanmin(coords[:, 1])), float(np.nanmax(coords[:, 1]))

    theta = np.mod(np.arctan2(gy, gx), np.pi)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    valid = grid_mask & np.isfinite(theta) & np.isfinite(mag) & (mag > 0)

    if np.any(valid):
        mag_thr = float(np.nanpercentile(mag[valid], max(0.0, min(100.0, strength_percentile))))
        mag_hi = float(np.nanpercentile(mag[valid], 98.5))
        if not np.isfinite(mag_hi) or mag_hi <= 0:
            mag_hi = float(np.nanmax(mag[valid])) if np.any(valid) else 1.0
        mag_den = max(mag_hi - mag_thr, 1e-8)
        alpha = np.clip((mag - mag_thr) / mag_den, 0.0, 1.0)
        alpha = 0.10 + 0.90 * alpha
        alpha[~valid] = 0.0
    else:
        alpha = np.zeros_like(mag, dtype=np.float64)

    norm = np.clip(theta / np.pi, 0.0, 1.0)
    rgba = orient_cmap(norm)
    rgba[..., 3] = alpha
    ax.imshow(rgba, origin='lower', extent=[xmn, xmx, ymn, ymx], aspect='equal', interpolation='nearest')

    h, w = theta.shape
    dx = (xmx - xmn) / max(w, 1)
    dy = (ymx - ymn) / max(h, 1)
    seg_len = 0.78 * min(dx, dy)
    segs = []
    glyph_mask = valid & (alpha > 0.18)
    for iy in range(0, h, max(1, int(step))):
        for ix in range(0, w, max(1, int(step))):
            if not glyph_mask[iy, ix]:
                continue
            ang = float(theta[iy, ix])
            cx = xmn + (ix + 0.5) * dx
            cy = ymn + (iy + 0.5) * dy
            ux = np.cos(ang)
            uy = np.sin(ang)
            x0, y0 = cx - 0.5 * seg_len * ux, cy - 0.5 * seg_len * uy
            x1, y1 = cx + 0.5 * seg_len * ux, cy + 0.5 * seg_len * uy
            segs.append([(x0, y0), (x1, y1)])
    if segs:
        lc = LineCollection(segs, colors='black', linewidths=0.60, alpha=0.62)
        ax.add_collection(lc)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal'); ax.invert_yaxis()
    ax.set_title(title, fontsize=11.2, fontweight='bold')
    if annotation_lines:
        ax.text(0.98, 0.97, "\n".join(annotation_lines), transform=ax.transAxes,
                ha='right', va='top', fontsize=8.1,
                bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor='0.82', alpha=0.94))


# def make_orientation_colorbar(fig, cax):
#     """Create a compact 0–180° orientation colorbar."""
#     xs = np.linspace(0, 1, 256)
#     hsv = np.zeros((1, 256, 3), dtype=np.float64)
#     hsv[..., 0] = xs[None, :]
#     hsv[..., 1] = 0.82
#     hsv[..., 2] = 0.96
#     rgb = hsv_to_rgb(hsv)
#     cax.imshow(rgb, aspect='auto', origin='lower', extent=[0, 180, 0, 1])
#     cax.set_yticks([])
#     cax.set_xlim(0, 180)
#     cax.set_xticks([0, 45, 90, 135, 180])
#     cax.set_xlabel('Orientation (deg)', fontsize=8.7)
def make_orientation_colorbar(fig, cax):
    """Create a compact vertical 0–180° orientation colorbar."""
    ys = np.linspace(0, 1, 256)

    hsv = np.zeros((256, 1, 3), dtype=np.float64)
    hsv[..., 0] = ys[:, None]
    hsv[..., 1] = 0.82
    hsv[..., 2] = 0.96

    rgb = hsv_to_rgb(hsv)

    cax.imshow(
        rgb,
        aspect="auto",
        origin="lower",
        extent=[0, 1, 0, 180],
    )

    cax.set_xticks([])
    cax.set_ylim(0, 180)
    cax.set_yticks([0, 45, 90, 135, 180])
    cax.set_ylabel(
        "Orientation (deg)",
        fontsize=8.7,
    )
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")


def plot_similarity_map_axis(ax, ssim_map: np.ndarray, title: str):
    vals = np.asarray(ssim_map, dtype=np.float64)
    if np.all(~np.isfinite(vals)):
        ax.text(0.5, 0.5, "No valid overlap", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11.2, fontweight="bold")
        return None
    im = ax.imshow(np.ma.masked_invalid(vals), cmap="viridis", vmin=-0.2, vmax=1.0, origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11.2, fontweight="bold")
    return im


def plot_response_heatmap_axis(ax, response: np.ndarray, title: str):
    if response is None or np.all(~np.isfinite(response)):
        ax.text(0.5, 0.5, "No response", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11.2, fontweight="bold")
        return None
    im = ax.imshow(response, cmap="mako", vmin=np.nanpercentile(response, 2), vmax=np.nanpercentile(response, 98), origin="lower")
    ax.set_xticks(np.arange(len(SEARCH_SHIFT_PIXELS)))
    ax.set_yticks(np.arange(len(SEARCH_SHIFT_PIXELS)))
    ax.set_xticklabels(SEARCH_SHIFT_PIXELS)
    ax.set_yticklabels(SEARCH_SHIFT_PIXELS)
    ax.set_xlabel("shift x (px)")
    ax.set_ylabel("shift y (px)")
    ax.set_title(title, fontsize=11.2, fontweight="bold")
    return im


def overlay_mask_rgb(real_mask: np.ndarray, cand_mask: np.ndarray, cand_color: str) -> np.ndarray:
    from matplotlib.colors import to_rgb
    rgb = np.ones((real_mask.shape[0], real_mask.shape[1], 3), dtype=np.float32)
    rc = np.array(to_rgb("#111111"), dtype=np.float32)
    cc = np.array(to_rgb(cand_color), dtype=np.float32)
    r = real_mask > 0
    c = cand_mask > 0
    rgb[r] = 0.85 * rgb[r] + 0.15 * rc
    rgb[c] = 0.80 * rgb[c] + 0.20 * cc
    rgb[r & c] = 0.5 * rc + 0.5 * cc
    return np.clip(rgb, 0, 1)


def plot_hist_bar_axis(ax, centers: np.ndarray, hist: np.ndarray,
                       title: str, color: str, xlab: str,
                       ann: Optional[str] = None, orientation_deg: bool = False,
                       ann_xy: Tuple[float, float] = (0.98, 0.94), ann_ha: str = "right"):
    centers = np.asarray(centers, dtype=np.float64)
    hist = np.asarray(hist, dtype=np.float64)
    if np.all(~np.isfinite(hist)) or centers.size == 0:
        ax.text(0.5, 0.5, "No valid histogram", ha="center", va="center", transform=ax.transAxes, fontsize=9)
    else:
        if centers.size > 1:
            width = float(np.nanmedian(np.diff(centers))) * 0.86
        else:
            width = 8.0 if orientation_deg else 0.8
        ax.bar(centers, hist, width=width, color=color, alpha=0.88,
               edgecolor="white", linewidth=0.55)
    if orientation_deg:
        ax.set_xlim(0, 180)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Fraction")
    ax.set_title(title, fontsize=11.0, fontweight="bold")
    ax.grid(axis="y", alpha=0.18)
    if ann is not None:
        ax.text(ann_xy[0], ann_xy[1], ann, transform=ax.transAxes, ha=ann_ha, va="top", fontsize=8.15,
                linespacing=1.12,
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="0.82", alpha=0.95))



def build_spatially_ordered_profiles(real_field: Dict[str, np.ndarray],
                                     sp_field: Dict[str, np.ndarray],
                                     axis_field: Dict[str, np.ndarray],
                                     orient_percentile: float = GRID_ORIENT_MIN_PERCENTILE) -> Dict[str, object]:
    """Build case-study profiles that preserve grid-cell spatial order.

    The profiles use the same underlying grid fields as the quantitative metrics,
    but instead of collapsing them into global histograms, they keep the grid-cell
    order (row-major over the common valid mask) so spatial location is encoded
    along the x-axis.
    """
    rm = np.asarray(real_field.get("eval_mask", real_field["grid_mask"]), dtype=bool)
    sm = np.asarray(sp_field.get("eval_mask", sp_field["grid_mask"]), dtype=bool)
    am = np.asarray(axis_field.get("eval_mask", axis_field["grid_mask"]), dtype=bool)
    common = rm & sm & am

    def ordered_flat(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        yy, xx = np.where(mask)
        if yy.size == 0:
            return np.array([], dtype=np.float64)
        order = np.lexsort((xx, yy))
        return arr[yy[order], xx[order]]

    # Strength profile.
    r_strength = np.asarray(real_field["grid_strength"], dtype=np.float64)
    s_strength = np.asarray(sp_field["grid_strength"], dtype=np.float64)
    a_strength = np.asarray(axis_field["grid_strength"], dtype=np.float64)
    strength_mask = common & np.isfinite(r_strength) & np.isfinite(s_strength) & np.isfinite(a_strength)

    # Orientation profile: degree representation, but only for sufficiently strong real gradients.
    r_c2, r_s2, r_mag = axial_orientation_components(real_field["grid_gx"], real_field["grid_gy"])
    s_c2, s_s2, s_mag = axial_orientation_components(sp_field["grid_gx"], sp_field["grid_gy"])
    a_c2, a_s2, a_mag = axial_orientation_components(axis_field["grid_gx"], axis_field["grid_gy"])

    orient_mask = (
        common & np.isfinite(r_mag) & np.isfinite(s_mag) & np.isfinite(a_mag) &
        np.isfinite(r_c2) & np.isfinite(s_c2) & np.isfinite(a_c2) &
        np.isfinite(r_s2) & np.isfinite(s_s2) & np.isfinite(a_s2) &
        (r_mag > 0) & (s_mag > 0) & (a_mag > 0)
    )
    if np.sum(orient_mask) >= 5 and orient_percentile is not None:
        thr = np.nanpercentile(r_mag[orient_mask], float(orient_percentile))
        orient_mask = orient_mask & (r_mag >= thr)

    def theta_deg(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return np.mod(np.arctan2(gy, gx), np.pi) * 180.0 / np.pi

    # Strength-weighted orientation (HOG-like) scalar profile.
    weighted_mask = orient_mask.copy()
    if np.sum(weighted_mask) >= 1:
        pooled_mag = np.r_[r_mag[weighted_mask], s_mag[weighted_mask], a_mag[weighted_mask]]
        w_scale = max(float(np.nanpercentile(pooled_mag, 99.0)), 1e-8) if pooled_mag.size else 1.0
    else:
        w_scale = 1.0

    def hog_scalar(gx: np.ndarray, gy: np.ndarray, mag: np.ndarray) -> np.ndarray:
        theta = np.mod(np.arctan2(gy, gx), np.pi)
        w = np.clip(mag / w_scale, 0.0, 1.0)
        # scalar HOG-like index in [0, 1], preserving both orientation and strength
        return w * (0.5 * (np.cos(2.0 * theta) + 1.0))

    profiles = {
        "strength_x": np.arange(int(np.sum(strength_mask))),
        "strength": {
            "Real": ordered_flat(r_strength, strength_mask),
            "SpatialZ": ordered_flat(s_strength, strength_mask),
            "AXIS": ordered_flat(a_strength, strength_mask),
        },
        "orientation_x": np.arange(int(np.sum(orient_mask))),
        "orientation": {
            "Real": ordered_flat(theta_deg(real_field["grid_gx"], real_field["grid_gy"]), orient_mask),
            "SpatialZ": ordered_flat(theta_deg(sp_field["grid_gx"], sp_field["grid_gy"]), orient_mask),
            "AXIS": ordered_flat(theta_deg(axis_field["grid_gx"], axis_field["grid_gy"]), orient_mask),
        },
        "weighted_x": np.arange(int(np.sum(weighted_mask))),
        "weighted": {
            "Real": ordered_flat(hog_scalar(real_field["grid_gx"], real_field["grid_gy"], r_mag), weighted_mask),
            "SpatialZ": ordered_flat(hog_scalar(sp_field["grid_gx"], sp_field["grid_gy"], s_mag), weighted_mask),
            "AXIS": ordered_flat(hog_scalar(axis_field["grid_gx"], axis_field["grid_gy"], a_mag), weighted_mask),
        },
    }
    return profiles



def smooth_spatial_profile(y: np.ndarray, sigma_frac: float = 0.018) -> np.ndarray:
    """Smooth a 1D spatially ordered profile to reveal large-scale trends.

    The ordered-grid profiles can be visually jagged because adjacent grid cells
    may still differ locally. For figure presentation, we use a gentle Gaussian
    smoothing along the 1D ordered-grid axis so that the large spatial pattern is
    emphasized without changing the underlying evaluation metric.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size <= 3:
        return y.copy()
    sigma = max(1.0, float(y.size) * float(sigma_frac))
    return gaussian_filter(y, sigma=sigma, mode="nearest")


def plot_ordered_hog_profile_axis(ax,
                                  real_y: np.ndarray,
                                  candidate_y: Optional[np.ndarray],
                                  candidate_label: str,
                                  candidate_color: str,
                                  title: str,
                                  annotation_lines: Optional[Sequence[str]] = None,
                                  ylim: Optional[Tuple[float, float]] = None) -> None:
    """Presentation plot for the spatially ordered HOG-like profile.

    Real is shown as a thin black reference trend, while the candidate is shown
    as a colored filled curve. This is easier to read than overlaying all three
    methods in one panel.
    """
    real_y = np.asarray(real_y, dtype=np.float64)
    x = np.arange(real_y.size, dtype=np.float64)
    if real_y.size == 0:
        ax.text(0.5, 0.5, 'No valid ordered grid cells', transform=ax.transAxes,
                ha='center', va='center', fontsize=9.5)
        ax.set_title(title, fontsize=11.2, fontweight='bold')
        return

    real_sm = smooth_spatial_profile(real_y)
    base = 0.0 if ylim is None else float(ylim[0])

    # Real reference
    ax.plot(x, real_sm, color=METHOD_PALETTE['Real'], lw=1.9, alpha=0.95, label='Real reference', zorder=3)
    ax.fill_between(x, base, real_sm, color='0.55', alpha=0.12, linewidth=0, zorder=1)

    # Candidate, if any
    if candidate_y is None or candidate_label == 'Real':
        ax.plot(x, real_sm, color='0.20', lw=2.4, alpha=0.98, label='Real', zorder=4)
        ax.fill_between(x, base, real_sm, color='0.35', alpha=0.18, linewidth=0, zorder=2)
    else:
        cand = np.asarray(candidate_y, dtype=np.float64)
        n = min(real_sm.size, cand.size)
        x = x[:n]
        real_sm = real_sm[:n]
        cand_sm = smooth_spatial_profile(cand[:n])
        ax.plot(x, cand_sm, color=candidate_color, lw=2.4, alpha=0.98, label=candidate_label, zorder=5)
        ax.fill_between(x, base, cand_sm, color=candidate_color, alpha=0.18, linewidth=0, zorder=2)

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11.2, fontweight='bold')
    ax.set_xlabel('Ordered grid position')
    ax.set_ylabel('Weighted orientation index')
    ax.grid(axis='y', alpha=0.18)
    ax.legend(loc='upper left', frameon=False, fontsize=8.7, ncol=2)
    if annotation_lines:
        ax.text(0.985, 0.96, "\n".join(annotation_lines), transform=ax.transAxes,
                ha='right', va='top', fontsize=8.15,
                bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor='0.82', alpha=0.94))


def plot_spatial_profile_axis(ax, x: np.ndarray, series_dict: Dict[str, np.ndarray],
                              title: str, ylabel: str,
                              annotation_lines: Optional[Sequence[str]] = None,
                              ylim: Optional[Tuple[float, float]] = None) -> None:
    """Plot Real / SpatialZ / AXIS together as continuous spatially ordered curves."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        ax.text(0.5, 0.5, 'No valid ordered grid cells', transform=ax.transAxes,
                ha='center', va='center', fontsize=9.5)
        ax.set_title(title, fontsize=11.2, fontweight='bold')
        ax.set_xlabel('Ordered grid position')
        ax.set_ylabel(ylabel)
        return

    for method in ["Real", "SpatialZ", "AXIS"]:
        y = np.asarray(series_dict.get(method, np.array([])), dtype=np.float64)
        if y.size != x.size or y.size == 0:
            continue
        color = METHOD_PALETTE[method]
        lw = 2.4 if method == 'Real' else 1.8
        alpha_fill = 0.14 if method == 'Real' else 0.16
        ax.plot(x, y, color=color, lw=lw, alpha=0.98, label=method)
        baseline = 0.0 if ylim is None else float(ylim[0])
        ax.fill_between(x, baseline, y, color=color, alpha=alpha_fill, linewidth=0)

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11.2, fontweight='bold')
    ax.set_xlabel('Ordered grid position')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.18)
    ax.legend(loc='upper left', frameon=False, fontsize=8.8, ncol=3)
    if annotation_lines:
        ax.text(0.985, 0.96, "\n".join(annotation_lines), transform=ax.transAxes,
                ha='right', va='top', fontsize=8.15,
                bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor='0.82', alpha=0.94))


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
# SUMMARY PLOTS
# ============================================================
def _manual_box_strip(ax, data: pd.DataFrame, metric: str, order: Sequence[str]) -> None:
    positions = np.arange(len(order), dtype=float)
    for pos, method in zip(positions, order):
        vals = data.loc[data["candidate"] == method, metric].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        color = METHOD_PALETTE.get(method, "#777777")
        if vals.size >= 1:
            ax.boxplot(
                [vals], positions=[pos], widths=0.48, patch_artist=True,
                showfliers=False, manage_ticks=False,
                medianprops=dict(color="black", linewidth=1.25),
                boxprops=dict(facecolor=color, edgecolor="black", linewidth=0.9, alpha=0.48),
                whiskerprops=dict(color="black", linewidth=0.85),
                capprops=dict(color="black", linewidth=0.85),
            )
            rng = np.random.default_rng(RANDOM_SEED + int(pos) + vals.size)
            jitter = rng.uniform(-0.09, 0.09, size=vals.size) if vals.size > 1 else np.zeros_like(vals)
            ax.scatter(np.full(vals.size, pos) + jitter, vals, s=20,
                       facecolor="white", edgecolor="black", linewidth=0.50, alpha=0.92, zorder=3)
        else:
            ax.text(pos, 0.5, "NA", ha="center", va="center", transform=ax.get_xaxis_transform(), fontsize=9)
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=0)
    ax.grid(axis="y", alpha=0.18)
    sns.despine(ax=ax)


def plot_patient_summaries(metrics_df: pd.DataFrame) -> None:
    """
    Plot marker genes and top genes jointly in a 3 x 1 metric layout.

    Colors are taken from the color cycle configured by set_cell_style(),
    rather than introducing a new palette.
    """
    if metrics_df.empty:
        print("[WARN] no metrics available for patient summaries")
        return

    set_cell_style()

    from matplotlib.patches import Patch

    specs = [
        ("transition_ncc", "Transition-map NCC"),
        (
            "spatial_grid_strength_weighted_orientation_pearson",
            "Weighted orientation fidelity",
        ),
        (
            "spatial_grid_orientation_pearson",
            "Orientation fidelity",
        ),
    ]

    order = [
        method
        for method in METHOD_ORDER
        if method in set(metrics_df["candidate"].astype(str))
    ]

    gene_sets = [
        ("marker", "Marker genes"),
        (
            "top",
            f"Top {TOP_N_SELECTED_GENES}genes",
        ),
    ]

    # Reuse the first two colors established by set_cell_style().
    style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    gene_set_colors = {
        "marker": style_colors[0],
        "top": style_colors[1],
    }

    offsets = {
        "marker": -0.18,
        "top": 0.18,
    }

    box_width = 0.30
    summary_rows = []

    def _plot_joint_box_strip(
        ax: plt.Axes,
        data: pd.DataFrame,
        metric: str,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)

        for method_idx, method in enumerate(order):
            method_df = data[
                data["candidate"].astype(str) == str(method)
            ]

            for gene_set, _ in gene_sets:
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

                position = method_idx + offsets[gene_set]
                color = gene_set_colors[gene_set]

                ax.boxplot(
                    [values],
                    positions=[position],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    boxprops={
                        "facecolor": color,
                        "edgecolor": "0.25",
                        "linewidth": 0.8,
                        "alpha": 0.78,
                    },
                    medianprops={
                        "color": "0.10",
                        "linewidth": 1.25,
                    },
                    whiskerprops={
                        "color": "0.35",
                        "linewidth": 0.8,
                    },
                    capprops={
                        "color": "0.35",
                        "linewidth": 0.8,
                    },
                    zorder=2,
                )

                jitter = rng.uniform(
                    -0.07,
                    0.07,
                    size=values.size,
                )

                ax.scatter(
                    np.full(values.size, position) + jitter,
                    values,
                    s=9,
                    color=color,
                    alpha=0.42,
                    edgecolors="none",
                    rasterized=True,
                    zorder=3,
                )

        ax.set_xlim(
            -0.55,
            len(order) - 0.45,
        )
        ax.set_xticks(
            np.arange(len(order))
        )
        ax.set_xticklabels(
            order,
            fontsize=9,
        )

        ax.grid(
            axis="y",
            linewidth=0.55,
            color="0.88",
            alpha=0.7,
            zorder=0,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for gid in GROUP_IDS:
        patient_df = metrics_df[
            metrics_df["group_id"] == gid
        ].copy()

        if patient_df.empty:
            continue

        gene_counts = {
            gene_set: int(
                patient_df.loc[
                    patient_df["gene_set"].astype(str) == gene_set,
                    "gene",
                ].nunique()
            )
            for gene_set, _ in gene_sets
        }

        # Three rows and one column.
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(3.4, 12.2),
            squeeze=False,
            sharex=True,
        )
        axes = axes[:, 0]

        for metric_idx, (metric, title) in enumerate(specs):
            ax = axes[metric_idx]

            if metric not in patient_df.columns:
                ax.text(
                    0.5,
                    0.5,
                    "Metric not available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="0.45",
                )
                ax.set_axis_off()
                continue

            _plot_joint_box_strip(
                ax=ax,
                data=patient_df,
                metric=metric,
                seed=int(gid) * 100 + metric_idx,
            )

            ax.set_title(
                title,
                fontsize=11.6,
                fontweight="semibold",
                pad=7,
            )
            ax.set_ylabel(
                "Score",
                fontsize=9.5,
            )

            if metric_idx < len(specs) - 1:
                ax.tick_params(
                    axis="x",
                    labelbottom=False,
                )
            else:
                ax.set_xlabel(
                    "Method",
                    fontsize=9.5,
                )

        legend_handles = [
            Patch(
                facecolor=gene_set_colors[gene_set],
                edgecolor="0.25",
                linewidth=0.8,
                alpha=0.78,
                label=f"{label} (n={gene_counts[gene_set]})",
            )
            for gene_set, label in gene_sets
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=2,
            frameon=False,
            fontsize=8.6,
            columnspacing=1.5,
        )

        fig.suptitle(
            f"Patient {gid}: gene-transition fidelity",
            y=0.995,
            fontsize=14.2,
            fontweight="bold",
        )

        fig.subplots_adjust(
            left=0.15,
            right=0.97,
            bottom=0.08,
            top=0.89,
            hspace=0.30,
        )

        ppath = os.path.join(
            OUTDIR,
            f"patient_{gid}_transition_metric_summary.png",
        )

        fig.savefig(
            ppath,
            dpi=FIG_DPI,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"[OK] saved: {ppath}")
        plt.close(fig)

        # Keep the original per-gene-set summary-table structure.
        for gene_set, _ in gene_sets:
            gene_set_df = patient_df[
                patient_df["gene_set"].astype(str) == gene_set
            ]

            for method in order:
                sub = gene_set_df[
                    gene_set_df["candidate"].astype(str)
                    == str(method)
                ]

                row = {
                    "group_id": gid,
                    "gene_set": gene_set,
                    "candidate": method,
                    "n_genes": int(sub["gene"].nunique()),
                }

                for metric, _ in specs:
                    if metric not in sub.columns:
                        row[f"{metric}_median"] = np.nan
                        row[f"{metric}_mean"] = np.nan
                        continue

                    values = (
                        sub[metric]
                        .astype(float)
                        .to_numpy()
                    )
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

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            SUMMARY_CSV,
            index=False,
        )
        print(f"[OK] saved: {SUMMARY_CSV}")
# ============================================================
# CASE STUDIES
# ============================================================
def plot_combined_case(group_id: int, gene: str, gene_bundle: Dict[str, Dict[str, np.ndarray]], outdir: str) -> None:
    set_cell_style()
    # expr_cmap = make_expression_cmap()
    expr_cmap = "Reds"
    grad_cmap = make_gradient_cmap()

    real_field = gene_bundle["Real"]
    sp_field = gene_bundle["SpatialZ"]
    axis_field = gene_bundle["AXIS"]
    qc = gene_bundle.get("gene_qc", {})
    fields = [real_field, sp_field, axis_field]

    def _pool_finite(field_list: Sequence[Dict[str, np.ndarray]], key: str) -> np.ndarray:
        vals = []
        for ff in field_list:
            arr = np.asarray(ff.get(key, np.array([])), dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                vals.append(arr)
        return np.concatenate(vals) if len(vals) > 0 else np.array([], dtype=np.float64)

    expr_pool = _pool_finite(fields, "expr_raw")
    expr_vmax = max(float(np.nanpercentile(expr_pool, 99.0)), 1e-6) if expr_pool.size else 1.0

    spot_pool = _pool_finite(fields, "strength_trend")
    map_pool = _pool_finite(fields, "grid_strength")
    grad_vmax = max(
        float(np.nanpercentile(spot_pool, 98.5)) if spot_pool.size else 0.0,
        float(np.nanpercentile(map_pool, 98.5)) if map_pool.size else 0.0,
        1e-6,
    )

    spm = gene_bundle.get("SpatialZ_spatial_metrics", {})
    axm = gene_bundle.get("AXIS_spatial_metrics", {})


    fig = plt.figure(figsize=(12,14))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], hspace=0.05, wspace=0.08)
    rows = [("Real", real_field), ("SpatialZ", sp_field), ("AXIS", axis_field)]

    for r, (method, field) in enumerate(rows):
        ax = fig.add_subplot(gs[r, 0])
        plot_expression_axis(ax, field, f"{method}: {gene}", expr_cmap, 0, expr_vmax)
        # if method == "Real":
        #     txt = (
        #         f"QC passed\n"
        #         f"Total count = {qc.get('real_total_count', np.nan):.0f}\n"
        #         f"Detected spots = {qc.get('real_detected_spots', np.nan):.0f}"
        #     )
        #     ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=8.35,
        #             bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.82", alpha=0.94))

        # Column 2: spot-level gradient map with streamlines.
        ax = fig.add_subplot(gs[r, 1])
        plot_spot_gradient_axis(ax, field, f"{method}: spot gradient map", grad_cmap, 0, grad_vmax, overlay_stream=True)

        # # Column 3: rasterized/grid transition map without streamlines.
        # ax = fig.add_subplot(gs[r, 2])
        # plot_transition_map_axis(ax, field, f"{method}: rasterized transition map", grad_cmap, 0, grad_vmax, overlay_stream=False)
        # if method == "SpatialZ":
        #     ax.text(0.03, 0.97, f"NCC = {spm.get('transition_ncc', np.nan):.3f}",
        #             transform=ax.transAxes, ha="left", va="top", fontsize=8.35,
        #             bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.82", alpha=0.94))
        # elif method == "AXIS":
        #     ax.text(0.03, 0.97, f"NCC = {axm.get('transition_ncc', np.nan):.3f}",
        #             transform=ax.transAxes, ha="left", va="top", fontsize=8.35,
        #             bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.82", alpha=0.94))

        # Column 4: high-contrast orientation field.
        ax = fig.add_subplot(gs[r, 2])
        if method == "Real":
            plot_orientation_field_axis(
                ax, field,
                title="Real: spatial orientation field",
                annotation_lines=["Reference field"],
            )
        elif method == "SpatialZ":
            plot_orientation_field_axis(
                ax, field,
                title="SpatialZ: spatial orientation field",
                annotation_lines=[
                    f"Ncc  = {spm.get('transition_ncc', np.nan):.3f}",
                    f"wOri r = {spm.get('spatial_grid_strength_weighted_orientation_pearson', np.nan):.3f}",
                    f"Ori r = {spm.get('spatial_grid_orientation_pearson', np.nan):.3f}",
                ],
            )
        else:
            plot_orientation_field_axis(
                ax, field,
                title="AXIS: spatial orientation field",
                annotation_lines=[
                    f"Ncc  = {spm.get('transition_ncc', np.nan):.3f}",
                    f"wOri r = {axm.get('spatial_grid_strength_weighted_orientation_pearson', np.nan):.3f}",
                    f"Ori r = {axm.get('spatial_grid_orientation_pearson', np.nan):.3f}",
                ],
            )

    cax1 = fig.add_axes([0.932, 0.67, 0.010, 0.22])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(cmap=expr_cmap, norm=plt.Normalize(vmin=0, vmax=expr_vmax)), cax=cax1)
    cb1.set_label(f"{gene} log1p", fontsize=8.7)

    cax2 = fig.add_axes([0.932, 0.37, 0.010, 0.22])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(cmap=grad_cmap, norm=plt.Normalize(vmin=0, vmax=grad_vmax)), cax=cax2)
    cb2.set_label("gradient strength", fontsize=8.7)

    cax3 = fig.add_axes([0.932, 0.1, 0.010, 0.22])
    make_orientation_colorbar(fig, cax3)

    # fig.suptitle(
    #     f"Patient {group_id} | {gene}: raw expression, spot gradient, rasterized transition map, and spatial orientation field",
    #     y=0.995, fontsize=16.0, fontweight="bold",
    # )
    fig.tight_layout(rect=[0, 0.045, 0.922, 0.98])
    save_path = os.path.join(outdir, f"P{group_id}_{gene}_marker_transition_case.png")
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"[OK] saved: {save_path}")
    plt.close(fig)

def plot_least_gene_diagnosis(group_id: int, gene: str, gene_bundle: Dict[str, Dict[str, np.ndarray]], outdir: str) -> None:
    set_cell_style()
    expr_cmap = make_expression_cmap()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), squeeze=False)
    axes = axes.ravel()
    fields = [("Real", gene_bundle["Real"]), ("SpatialZ", gene_bundle["SpatialZ"]), ("AXIS", gene_bundle["AXIS"])]
    expr_pool = np.concatenate([
        f["expr_raw"][np.isfinite(f["expr_raw"])] for _, f in fields if np.any(np.isfinite(f["expr_raw"]))
    ])
    expr_vmax = max(float(np.nanpercentile(expr_pool, 99.0)), 1e-6) if expr_pool.size else 1.0
    for ax, (method, field) in zip(axes, fields):
        plot_expression_axis(ax, field, method, expr_cmap, 0, expr_vmax)
        total = float(np.nansum(field["expr_raw"]))
        detect = int(np.sum(np.isfinite(field["expr_raw"]) & (field["expr_raw"] > 0)))
        ax.text(0.03, 0.97, f"count={total:.0f}\nspots={detect}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8.0,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="0.84", alpha=0.94))
    fig.suptitle(f"Patient {group_id} | low-support evaluated gene: {gene}", y=1.03, fontsize=14.4, fontweight="bold")
    fig.tight_layout()
    cax = fig.add_axes([0.92, 0.18, 0.012, 0.58])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=expr_cmap, norm=plt.Normalize(vmin=0, vmax=expr_vmax)), cax=cax)
    cb.set_label(f"{gene} log1p", fontsize=8.6)
    pth = os.path.join(outdir, f"P{group_id}_{gene}_low_support_diagnosis.png")
    fig.savefig(pth, dpi=FIG_DPI, bbox_inches="tight")
    print(f"[OK] saved: {pth}")
    plt.close(fig)


def make_case_figures(case_bundle: Dict[int, Dict[str, Dict[str, Dict[str, np.ndarray]]]], info_bundle: Dict[int, Dict[str, object]]) -> None:
    # for gid in GROUP_IDS:
    #     if gid not in case_bundle:
    #         continue
    #     outdir = os.path.join(OUTDIR, f"patient_{gid}_cases")
    #     ensure_dir(outdir)
    #     least_gene = info_bundle.get(gid, {}).get("least_gene", None)
    #     if least_gene is not None and least_gene in case_bundle[gid]:
    #         plot_least_gene_diagnosis(gid, least_gene, case_bundle[gid][least_gene], outdir)

    for gid in CASE_GROUPS:
        if gid not in case_bundle:
            continue
        outdir = os.path.join(OUTDIR, f"patient_{gid}_marker_cases")
        ensure_dir(outdir)
        marker_genes = list(info_bundle.get(gid, {}).get("marker_genes", []))
        if len(marker_genes) == 0:
            continue
        # for gene in marker_genes:
        for gene in ["KRT1","COL1A1"]:
            if gene not in case_bundle[gid]:
                continue
            plot_combined_case(gid, gene, case_bundle[gid][gene], outdir)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    ensure_dir(OUTDIR)
    set_cell_style()
    metrics_df, case_bundle, info_bundle = run_all()
    # plot_patient_summaries(metrics_df)
    make_case_figures(case_bundle, info_bundle)
    print("[DONE] SCC marker / top-gene transition evaluation finished.")


if __name__ == "__main__":
    main()
