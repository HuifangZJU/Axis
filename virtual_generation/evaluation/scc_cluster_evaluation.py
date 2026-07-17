#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCC virtual-slice cluster-neighborhood fidelity evaluation.

Streamlined revision. It keeps only the analyses requested in the current workflow:
cluster composition, KNN cluster-connectivity, bulk gene total-count support,
within-cluster NN distribution distance to Real, and marker-gene cluster case studies.

Main additions
--------------
1. Per-patient cluster connectivity heatmaps
   For each spot, find its nearest KNN_CLUSTER_NEIGHBOR neighbors in the same
   slice. For every source cluster i, report the fraction of its neighbor spots
   assigned to each target cluster j. This yields a row-normalized
   source-cluster -> neighbor-cluster connectivity matrix for Real, SpatialZ,
   and AXIS.

2. Per-cluster case study figures
   For each case patient and each real cluster:
     - Real / SpatialZ / AXIS spatial maps
       * all tissue spots: light gray
       * current source cluster: black
       * kNN neighbor spots of the source cluster: fixed global cluster colors
     - cluster composition comparison
     - source-cluster connectivity profile comparison
     - same-cluster nearest-neighbor ECDF

3. Removed all-gene program agreement
   No pseudobulk, all-gene Pearson/MAE, or gene-rank curve is computed or
   plotted in the case-study figures.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

from anndata import AnnData
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance, ks_2samp
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CONFIG
# ============================================================
ROOT = "/media/huifang/data/registration/SCC/huifang"
H5AD_DIR = os.path.join(ROOT, "h5ad")
VIRTUAL_DIR = os.path.join(ROOT, "virtual_tissue")

OUTDIR = os.path.join(VIRTUAL_DIR, "evaluation_cluster_v7")

GROUP_IDS = [2,10]
CASE_GROUPS = [2, 10]

CLUSTER_KEY = "original_clusters"
COORD_KEY = "spatial"

EPS = 1e-8
KNN_CLUSTER_NEIGHBOR = 6
SAVE_INDIVIDUAL_CLUSTER_PNG = False  # PDF is much faster; set True only when separate PNGs are needed

# Bulk-gene total-count diagnostics.
# These are computed on the loaded expression matrix before normalization/log1p.
GENE_DETECTION_THRESHOLD = 0.0
GENE_COUNT_EPS = 1e-8

# Marker-gene spatial case studies.
CASE_MARKER_GENES = ["KRT1", "COL1A1"]
MARKER_PLOT_LOG1P = True
MARKER_CMAP = "magma"
CASE_SPOT_SIZE_MARKER = 14.0

METHOD_ORDER = ["SpatialZ", "AXIS"]
PLOT_METHOD_ORDER = ["Real", "SpatialZ", "AXIS"]
METHOD_PALETTE = {
    "Real": "#111111",
    "SpatialZ": "#7C8DA6",
    "AXIS": "#C4473A",
}

# Case-study plotting controls.
CASE_SPOT_SIZE_BACKGROUND = 7.0
CASE_SPOT_SIZE_NEIGHBOR = 12.0
CASE_SPOT_SIZE_SOURCE = 14.5
CASE_BACKGROUND_COLOR = "#d9dde3"
CASE_BACKGROUND_ALPHA = 0.55
CASE_NEIGHBOR_ALPHA = 0.92
CASE_SOURCE_ALPHA = 0.98

FIG_DPI = 350


# ============================================================
# STYLE HELPERS
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
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename_token(x: str) -> str:
    return str(x).replace("/", "_").replace("\\", "_").replace(" ", "_")


def method_color(method: str) -> str:
    return METHOD_PALETTE.get(method, "0.25")


def canonical_cluster_order(labels: Sequence) -> List[str]:
    vals = pd.Series(labels).astype(str)
    uniq = list(pd.unique(vals))

    def key_fn(x):
        try:
            return (0, int(x))
        except Exception:
            return (1, str(x))

    return sorted([str(x) for x in uniq], key=key_fn)


def stable_cluster_color(cluster: str):
    """Stable global color for a cluster label, independent of patient/method."""
    label = str(cluster)
    tab20 = plt.get_cmap("tab20")
    try:
        idx = int(label) % tab20.N
    except Exception:
        idx = abs(hash(label)) % tab20.N
    return tab20(idx)


def cluster_color_dict(order: Sequence[str]) -> Dict[str, object]:
    return {str(c): stable_cluster_color(str(c)) for c in order}


def connectivity_cmap():
    # Light-to-heavy Cell/Nature-style teal-blue palette.
    from matplotlib.colors import LinearSegmentedColormap

    colors = [
        "#fbfbfd",
        "#e9f2f0",
        "#cfe8de",
        "#92ccb9",
        "#49a6a1",
        "#2f719f",
        "#153f7a",
    ]
    return LinearSegmentedColormap.from_list("cluster_connectivity_teal_blue", colors, N=256)


# ============================================================
# LOADING AND BASIC DATA ACCESS
# ============================================================
def load_ad(path: str) -> AnnData:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ad = sc.read_h5ad(path)
    if "valid" in ad.obs.columns:
        ad = ad[ad.obs["valid"]].copy()
    return ad


def load_group_data(group_id: int) -> Dict[str, AnnData]:
    real_path = os.path.join(H5AD_DIR, f"{group_id}_{1}.h5ad")
    axis_path = (
        f"/media/huifang/data/registration/SCC/huifang/virtual_tissue/"
        f"scc_z_group{group_id}_grid40_spatial3_gene_weight/query/"
        f"virtual_scc_z_10.000.h5ad"
    )
    spatialz_path = (
        f"/media/huifang/data/registration/SCC/huifang/virtual_tissue/"
        f"spatialz/group{group_id}/patient{group_id}_generation_10.h5ad"
    )

    return {
        "Real": load_ad(real_path),
        "SpatialZ": load_ad(spatialz_path),
        "AXIS": load_ad(axis_path),
    }


def get_cluster_labels(adata: AnnData, cluster_key: str = CLUSTER_KEY) -> np.ndarray:
    if cluster_key not in adata.obs:
        raise KeyError(f"{cluster_key!r} not found in adata.obs")
    return adata.obs[cluster_key].astype(str).values


def get_valid_coords_labels(
    adata: AnnData,
    cluster_key: str = CLUSTER_KEY,
    coord_key: str = COORD_KEY,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coord_key not in adata.obsm:
        raise KeyError(f"{coord_key!r} not found in adata.obsm")
    coords_all = np.asarray(adata.obsm[coord_key], dtype=np.float64)
    labels_all = get_cluster_labels(adata, cluster_key)
    finite = np.all(np.isfinite(coords_all), axis=1)
    valid_idx = np.where(finite)[0]
    return coords_all[finite], labels_all[finite].astype(str), valid_idx


def union_cluster_order(method_adatas: Dict[str, AnnData], cluster_key: str = CLUSTER_KEY) -> List[str]:
    labels = []
    for ad in method_adatas.values():
        labels.extend(get_cluster_labels(ad, cluster_key).astype(str).tolist())
    return canonical_cluster_order(labels)


def safe_mae(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(x[keep] - y[keep])))


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return np.nan
    x = x[keep]
    y = y[keep]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_cosine(x: np.ndarray, y: np.ndarray, eps: float = EPS) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() == 0:
        return np.nan
    x = x[keep]
    y = y[keep]
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= eps:
        return np.nan
    return float(np.dot(x, y) / denom)


def to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def extract_matrix(adata: AnnData, genes: Sequence[str]) -> np.ndarray:
    genes = [str(g) for g in genes if str(g) in set(adata.var_names.astype(str))]
    if len(genes) == 0:
        return np.zeros((adata.n_obs, 0), dtype=np.float32)
    X = adata[:, genes].X
    return to_dense(X).astype(np.float32, copy=False)


def extract_gene_vector(adata: AnnData, gene: str) -> np.ndarray:
    if str(gene) not in set(adata.var_names.astype(str)):
        return np.full(adata.n_obs, np.nan, dtype=np.float32)
    X = adata[:, [str(gene)]].X
    return to_dense(X).reshape(-1).astype(np.float32, copy=False)


def common_gene_intersection(adatas: Sequence[AnnData]) -> List[str]:
    if len(adatas) == 0:
        return []
    common = set(adatas[0].var_names.astype(str))
    for ad in adatas[1:]:
        common &= set(ad.var_names.astype(str))
    return sorted(common)


# ============================================================
# COMPOSITION AND SAME-CLUSTER DISTANCE
# ============================================================
def cluster_composition(adata: AnnData, cluster_key: str, order: Sequence[str]) -> np.ndarray:
    labels = get_cluster_labels(adata, cluster_key)
    counts = pd.Series(labels).astype(str).value_counts()
    vec = np.array([counts.get(str(c), 0) for c in order], dtype=np.float64)
    if vec.sum() > 0:
        vec = vec / vec.sum()
    return vec.astype(np.float32)


def symmetric_relative_abs_error(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return 2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b) + eps)


def signed_relative_error(ref: np.ndarray, pred: np.ndarray, eps: float = EPS) -> np.ndarray:
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return (pred - ref) / (np.abs(ref) + eps)


def slice_min_positive_spot_distance(adata: AnnData, coord_key: str = COORD_KEY, eps: float = EPS) -> float:
    coords, _, _ = get_valid_coords_labels(adata, coord_key=coord_key)
    if coords.shape[0] < 2:
        return np.nan
    tree = cKDTree(coords)
    try:
        dist, _ = tree.query(coords, k=2, workers=-1)
    except TypeError:
        dist, _ = tree.query(coords, k=2)
    nn = np.asarray(dist[:, 1], dtype=np.float64)
    nn = nn[np.isfinite(nn) & (nn > eps)]
    if nn.size == 0:
        return np.nan
    return float(np.min(nn))


def within_slice_same_cluster_nn_distances(
    adata: AnnData,
    cluster: str,
    cluster_key: str = CLUSTER_KEY,
    coord_key: str = COORD_KEY,
    normalization_spot_unit: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    coords, labels, _ = get_valid_coords_labels(adata, cluster_key=cluster_key, coord_key=coord_key)
    mask = labels == str(cluster)
    coords_c = coords[mask]

    if normalization_spot_unit is not None and np.isfinite(normalization_spot_unit) and normalization_spot_unit > 0:
        spot_unit = float(normalization_spot_unit)
    else:
        spot_unit = slice_min_positive_spot_distance(adata, coord_key=coord_key)

    if coords_c.shape[0] < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), spot_unit

    tree = cKDTree(coords_c)
    try:
        dist, _ = tree.query(coords_c, k=2, workers=-1)
    except TypeError:
        dist, _ = tree.query(coords_c, k=2)
    raw = np.asarray(dist[:, 1], dtype=np.float64)
    raw = raw[np.isfinite(raw)]

    if np.isfinite(spot_unit) and spot_unit > 0:
        norm = raw / spot_unit
    else:
        norm = np.full(raw.shape, np.nan, dtype=np.float64)
    return raw.astype(np.float32), norm.astype(np.float32), float(spot_unit) if np.isfinite(spot_unit) else np.nan


def within_cluster_nn_summary(distances: np.ndarray, prefix: str) -> Dict[str, float]:
    d = np.asarray(distances, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_p95": np.nan,
        }
    return {
        f"{prefix}_n": int(d.size),
        f"{prefix}_mean": float(np.mean(d)),
        f"{prefix}_median": float(np.median(d)),
        f"{prefix}_p90": float(np.percentile(d, 90)),
        f"{prefix}_p95": float(np.percentile(d, 95)),
    }


def nn_distribution_similarity_to_real(real_distances: np.ndarray, cand_distances: np.ndarray) -> Dict[str, float]:
    xr = np.asarray(real_distances, dtype=np.float64)
    xc = np.asarray(cand_distances, dtype=np.float64)
    xr = xr[np.isfinite(xr)]
    xc = xc[np.isfinite(xc)]
    if xr.size == 0 or xc.size == 0:
        return {
            "within_cluster_nn_wasserstein": np.nan,
            "within_cluster_nn_ks": np.nan,
            "within_cluster_nn_real_n": int(xr.size),
            "within_cluster_nn_candidate_n": int(xc.size),
        }
    try:
        ks_stat = float(ks_2samp(xr, xc, alternative="two-sided", method="auto").statistic)
    except TypeError:
        ks_stat = float(ks_2samp(xr, xc).statistic)
    return {
        "within_cluster_nn_wasserstein": float(wasserstein_distance(xr, xc)),
        "within_cluster_nn_ks": ks_stat,
        "within_cluster_nn_real_n": int(xr.size),
        "within_cluster_nn_candidate_n": int(xc.size),
    }


# ============================================================
# KNN CLUSTER CONNECTIVITY
# ============================================================
def knn_indices(coords: np.ndarray, k: int = KNN_CLUSTER_NEIGHBOR) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n < 2:
        return np.zeros((n, 0), dtype=np.int64)
    kk = min(k + 1, n)
    tree = cKDTree(coords)
    try:
        _, idx = tree.query(coords, k=kk, workers=-1)
    except TypeError:
        _, idx = tree.query(coords, k=kk)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim == 1:
        idx = idx[:, None]
    return idx[:, 1:].astype(np.int64, copy=False)


def cluster_neighbor_connectivity_long(
    adata: AnnData,
    group_id: int,
    method: str,
    cluster_order: Sequence[str],
    cluster_key: str = CLUSTER_KEY,
    coord_key: str = COORD_KEY,
    knn: int = KNN_CLUSTER_NEIGHBOR,
) -> pd.DataFrame:
    """Row-normalized source-cluster -> neighbor-cluster connectivity table."""
    coords, labels, _ = get_valid_coords_labels(adata, cluster_key=cluster_key, coord_key=coord_key)
    nn_idx = knn_indices(coords, k=knn)
    n_neighbors_actual = int(nn_idx.shape[1])
    rows = []
    for src in cluster_order:
        src = str(src)
        src_mask = labels == src
        n_src = int(np.sum(src_mask))
        if n_src > 0 and n_neighbors_actual > 0:
            nbr_labels = labels[nn_idx[src_mask].ravel()]
            counts = pd.Series(nbr_labels).astype(str).value_counts()
            denom = float(nbr_labels.size)
        else:
            counts = pd.Series(dtype=float)
            denom = 0.0
        for tgt in cluster_order:
            tgt = str(tgt)
            cnt = int(counts.get(tgt, 0))
            frac = float(cnt / denom) if denom > 0 else np.nan
            rows.append({
                "group_id": int(group_id),
                "method": method,
                "source_cluster": src,
                "target_cluster": tgt,
                "edge_count": cnt,
                "fraction": frac,
                "n_source_spots": n_src,
                "n_neighbor_edges": int(denom),
                "knn": int(knn),
                "actual_k": n_neighbors_actual,
            })
    return pd.DataFrame(rows)


def connectivity_matrix(
    connectivity_df: pd.DataFrame,
    group_id: int,
    method: str,
    cluster_order: Sequence[str],
    value: str = "fraction",
) -> pd.DataFrame:
    sub = connectivity_df[
        (connectivity_df["group_id"] == int(group_id))
        & (connectivity_df["method"] == method)
    ].copy()
    pivot = sub.pivot(index="source_cluster", columns="target_cluster", values=value)
    return pivot.reindex(index=[str(c) for c in cluster_order], columns=[str(c) for c in cluster_order])


def connectivity_row_vector(
    connectivity_df: pd.DataFrame,
    group_id: int,
    method: str,
    source_cluster: str,
    cluster_order: Sequence[str],
) -> np.ndarray:
    mat = connectivity_matrix(connectivity_df, group_id, method, cluster_order, value="fraction")
    if str(source_cluster) not in mat.index:
        return np.full(len(cluster_order), np.nan, dtype=np.float64)
    return mat.loc[str(source_cluster)].reindex([str(c) for c in cluster_order]).values.astype(np.float64)


def connectivity_profile_similarity(real_vec: np.ndarray, cand_vec: np.ndarray) -> Dict[str, float]:
    r = np.asarray(real_vec, dtype=np.float64)
    c = np.asarray(cand_vec, dtype=np.float64)
    keep = np.isfinite(r) & np.isfinite(c)
    if keep.sum() == 0:
        return {
            "connectivity_profile_l1": np.nan,
            "connectivity_profile_similarity": np.nan,
            "connectivity_profile_overlap": np.nan,
        }
    l1 = float(np.sum(np.abs(r[keep] - c[keep])))
    # For probability vectors, 1 - 0.5*L1 is total-variation similarity.
    sim = float(1.0 - 0.5 * l1)
    sim = float(np.clip(sim, 0.0, 1.0))
    overlap = float(np.sum(np.minimum(r[keep], c[keep])))
    return {
        "connectivity_profile_l1": l1,
        "connectivity_profile_similarity": sim,
        "connectivity_profile_overlap": overlap,
    }


def connectivity_profile_distances(real_vec: np.ndarray, cand_vec: np.ndarray) -> Dict[str, float]:
    """Distance-based comparison of row-normalized cluster-neighbor profiles."""
    p = np.asarray(real_vec, dtype=np.float64).ravel()
    q = np.asarray(cand_vec, dtype=np.float64).ravel()
    keep = np.isfinite(p) & np.isfinite(q)
    if keep.sum() == 0:
        return {
            "connectivity_profile_l1_distance": np.nan,
            "connectivity_profile_tv_distance": np.nan,
            "connectivity_profile_js_distance": np.nan,
        }

    p = np.clip(p[keep], 0.0, np.inf)
    q = np.clip(q[keep], 0.0, np.inf)
    sp = float(np.sum(p))
    sq = float(np.sum(q))
    if sp <= 0 or sq <= 0:
        return {
            "connectivity_profile_l1_distance": np.nan,
            "connectivity_profile_tv_distance": np.nan,
            "connectivity_profile_js_distance": np.nan,
        }

    p = p / sp
    q = q / sq
    l1 = float(np.sum(np.abs(p - q)))
    tv = float(0.5 * l1)

    mm = 0.5 * (p + q)
    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        keep2 = (a > 0) & (b > 0)
        if not np.any(keep2):
            return 0.0
        return float(np.sum(a[keep2] * np.log(a[keep2] / b[keep2])))

    js_div = 0.5 * _kl(p, mm) + 0.5 * _kl(q, mm)
    js_dist = float(np.sqrt(max(js_div, 0.0) / np.log(2.0)))
    js_dist = float(np.clip(js_dist, 0.0, 1.0))
    return {
        "connectivity_profile_l1_distance": l1,
        "connectivity_profile_tv_distance": tv,
        "connectivity_profile_js_distance": js_dist,
    }


def get_neighbor_indices_for_source_cluster(
    adata: AnnData,
    cluster: str,
    cluster_key: str = CLUSTER_KEY,
    coord_key: str = COORD_KEY,
    knn: int = KNN_CLUSTER_NEIGHBOR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return valid coords/labels plus source and union-neighbor indices."""
    coords, labels, valid_idx = get_valid_coords_labels(adata, cluster_key=cluster_key, coord_key=coord_key)
    nn_idx = knn_indices(coords, k=knn)
    source_idx = np.where(labels == str(cluster))[0]
    if source_idx.size == 0 or nn_idx.shape[1] == 0:
        neighbor_idx = np.array([], dtype=np.int64)
    else:
        neighbor_idx = np.unique(nn_idx[source_idx].ravel()).astype(np.int64)
    return coords, labels, valid_idx, source_idx, neighbor_idx


# ============================================================
# BULK GENE TOTAL-COUNT DIAGNOSTICS
# ============================================================
def cluster_gene_total_table(
    adata: AnnData,
    genes: Sequence[str],
    cluster_order: Sequence[str],
    cluster_key: str = CLUSTER_KEY,
) -> Dict[str, np.ndarray]:
    """Compute per-gene total counts for every cluster with one matrix read.

    This replaces the slow previous implementation that re-extracted the full
    gene matrix once per candidate/cluster.
    """
    genes = [g for g in genes if g in adata.var_names]
    labels = get_cluster_labels(adata, cluster_key)
    if len(genes) == 0:
        return {str(cl): np.zeros(0, dtype=np.float64) for cl in cluster_order}

    X = adata[:, genes].X
    out: Dict[str, np.ndarray] = {}
    for cl in cluster_order:
        mask = labels == str(cl)
        if not np.any(mask):
            out[str(cl)] = np.zeros(len(genes), dtype=np.float64)
            continue
        if sparse.issparse(X):
            vals = np.asarray(X[mask].sum(axis=0)).ravel().astype(np.float64)
        else:
            vals = np.asarray(X[mask].sum(axis=0)).ravel().astype(np.float64)
        out[str(cl)] = vals
    return out


def cluster_gene_mean_table(
    adata: AnnData,
    genes: Sequence[str],
    cluster_order: Sequence[str],
    cluster_key: str = CLUSTER_KEY,
) -> Dict[str, np.ndarray]:
    """Compute per-gene mean expression for every cluster with one matrix read."""
    genes = [g for g in genes if g in adata.var_names]
    labels = get_cluster_labels(adata, cluster_key)
    if len(genes) == 0:
        return {str(cl): np.zeros(0, dtype=np.float64) for cl in cluster_order}

    X = adata[:, genes].X
    out: Dict[str, np.ndarray] = {}
    for cl in cluster_order:
        mask = labels == str(cl)
        if not np.any(mask):
            out[str(cl)] = np.zeros(len(genes), dtype=np.float64)
            continue
        if sparse.issparse(X):
            vals = np.asarray(X[mask].mean(axis=0)).ravel().astype(np.float64)
        else:
            vals = np.asarray(X[mask].mean(axis=0)).ravel().astype(np.float64)
        out[str(cl)] = vals
    return out


def compare_gene_mean_vectors(real_mean: np.ndarray, cand_mean: np.ndarray) -> Dict[str, float]:
    real_mean = np.asarray(real_mean, dtype=np.float64).ravel()
    cand_mean = np.asarray(cand_mean, dtype=np.float64).ravel()
    return {"all_gene_mean_mae": safe_mae(real_mean, cand_mean)}


def compare_gene_total_vectors(
    real_total: np.ndarray,
    cand_total: np.ndarray,
    detect_threshold: float = GENE_DETECTION_THRESHOLD,
    eps: float = GENE_COUNT_EPS,
) -> Dict[str, float]:
    """Minimal gene total-count support metrics discussed for this workflow."""
    real_total = np.asarray(real_total, dtype=np.float64).ravel()
    cand_total = np.asarray(cand_total, dtype=np.float64).ravel()
    if real_total.size == 0 or cand_total.size == 0:
        return {
            "gene_total_count_total_sare": np.nan,
            "gene_total_count_l1_fraction": np.nan,
            "real_detected_gene_n": 0,
            "candidate_detected_gene_n": 0,
            "missing_detected_gene_n": 0,
            "extra_detected_gene_n": 0,
            "detected_gene_recall": np.nan,
        }
    real_detected = real_total > detect_threshold
    cand_detected = cand_total > detect_threshold
    missing = real_detected & (~cand_detected)
    extra = (~real_detected) & cand_detected
    shared = real_detected & cand_detected

    real_sum = float(np.sum(real_total))
    cand_sum = float(np.sum(cand_total))
    l1 = float(np.sum(np.abs(cand_total - real_total)))
    return {
        "gene_total_count_total_sare": float(2.0 * l1 / (real_sum + cand_sum + eps)),
        "gene_total_count_l1_fraction": float(l1 / (real_sum + eps)),
        "real_detected_gene_n": int(np.sum(real_detected)),
        "candidate_detected_gene_n": int(np.sum(cand_detected)),
        "missing_detected_gene_n": int(np.sum(missing)),
        "extra_detected_gene_n": int(np.sum(extra)),
        "detected_gene_recall": float(np.sum(shared) / max(int(np.sum(real_detected)), 1)),
    }



# ============================================================
# EVALUATION
# ============================================================
def evaluate_group(group_id: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    data = load_group_data(group_id)
    method_adatas = {m: data[m] for m in PLOT_METHOD_ORDER}

    cluster_order = union_cluster_order(method_adatas, cluster_key=CLUSTER_KEY)
    real_cluster_order = canonical_cluster_order(get_cluster_labels(method_adatas["Real"], CLUSTER_KEY))
    colors = cluster_color_dict(cluster_order)
    common_genes = common_gene_intersection([method_adatas[m] for m in PLOT_METHOD_ORDER])

    real_unit = slice_min_positive_spot_distance(method_adatas["Real"], coord_key=COORD_KEY)
    axis_unit = slice_min_positive_spot_distance(method_adatas["AXIS"], coord_key=COORD_KEY)
    norm_unit_by_method = {"Real": real_unit, "SpatialZ": axis_unit, "AXIS": axis_unit}

    comp_by_method = {method: cluster_composition(ad, CLUSTER_KEY, cluster_order) for method, ad in method_adatas.items()}
    comp_real = comp_by_method["Real"]

    connectivity_df = pd.concat([
        cluster_neighbor_connectivity_long(
            adata=ad,
            group_id=group_id,
            method=method,
            cluster_order=cluster_order,
            cluster_key=CLUSTER_KEY,
            coord_key=COORD_KEY,
            knn=KNN_CLUSTER_NEIGHBOR,
        )
        for method, ad in method_adatas.items()
    ], axis=0, ignore_index=True)
    real_conn_mat = connectivity_matrix(connectivity_df, group_id, "Real", cluster_order, value="fraction")

    gene_totals_by_method = {
        method: cluster_gene_total_table(ad, common_genes, cluster_order, cluster_key=CLUSTER_KEY)
        for method, ad in method_adatas.items()
    }
    gene_means_by_method = {
        method: cluster_gene_mean_table(ad, common_genes, cluster_order, cluster_key=CLUSTER_KEY)
        for method, ad in method_adatas.items()
    }

    neighbor_rows = []
    neighbor_arrays: Dict[Tuple[str, str], np.ndarray] = {}
    neighbor_summary: Dict[Tuple[str, str], Dict[str, float]] = {}
    for method, ad in method_adatas.items():
        for cl in cluster_order:
            cl = str(cl)
            _, norm_d, spot_unit = within_slice_same_cluster_nn_distances(
                adata=ad,
                cluster=cl,
                cluster_key=CLUSTER_KEY,
                coord_key=COORD_KEY,
                normalization_spot_unit=norm_unit_by_method.get(method, None),
            )
            neighbor_arrays[(method, cl)] = norm_d
            neighbor_summary[(method, cl)] = {
                "structure_spot_unit": float(spot_unit) if np.isfinite(spot_unit) else np.nan,
                **within_cluster_nn_summary(norm_d, "norm_within_cluster_nn"),
            }
            for val in norm_d:
                neighbor_rows.append({
                    "group_id": int(group_id),
                    "method": method,
                    "cluster": cl,
                    "structure_spot_unit": float(spot_unit) if np.isfinite(spot_unit) else np.nan,
                    "within_cluster_nn_distance_spot_units": float(val),
                })
    neighbor_df = pd.DataFrame(neighbor_rows)

    per_cluster_rows = []
    real_labels = get_cluster_labels(method_adatas["Real"], CLUSTER_KEY)
    for cand in METHOD_ORDER:
        cand_labels = get_cluster_labels(method_adatas[cand], CLUSTER_KEY)
        comp_c = comp_by_method[cand]
        comp_sare = symmetric_relative_abs_error(comp_real, comp_c)
        comp_signed = signed_relative_error(comp_real, comp_c)
        cand_conn_mat = connectivity_matrix(connectivity_df, group_id, cand, cluster_order, value="fraction")

        for i, cl in enumerate(cluster_order):
            cl = str(cl)
            real_vec = real_conn_mat.loc[cl].values.astype(np.float64) if cl in real_conn_mat.index else np.full(len(cluster_order), np.nan)
            cand_vec = cand_conn_mat.loc[cl].values.astype(np.float64) if cl in cand_conn_mat.index else np.full(len(cluster_order), np.nan)
            conn_cmp = connectivity_profile_distances(real_vec, cand_vec)
            real_diag = float(real_conn_mat.loc[cl, cl]) if cl in real_conn_mat.index and cl in real_conn_mat.columns else np.nan
            cand_diag = float(cand_conn_mat.loc[cl, cl]) if cl in cand_conn_mat.index and cl in cand_conn_mat.columns else np.nan
            gene_cmp = compare_gene_total_vectors(
                gene_totals_by_method["Real"].get(cl, np.zeros(0)),
                gene_totals_by_method[cand].get(cl, np.zeros(0)),
                detect_threshold=GENE_DETECTION_THRESHOLD,
            )
            mean_cmp = compare_gene_mean_vectors(
                gene_means_by_method["Real"].get(cl, np.zeros(0)),
                gene_means_by_method[cand].get(cl, np.zeros(0)),
            )
            real_nn_dist = neighbor_arrays.get(("Real", cl), np.array([], dtype=np.float32))
            cand_nn_dist = neighbor_arrays.get((cand, cl), np.array([], dtype=np.float32))
            structure_cmp = nn_distribution_similarity_to_real(real_nn_dist, cand_nn_dist)
            real_nn_sum = neighbor_summary.get(("Real", cl), within_cluster_nn_summary(np.array([]), "norm_within_cluster_nn"))
            cand_nn_sum = neighbor_summary.get((cand, cl), within_cluster_nn_summary(np.array([]), "norm_within_cluster_nn"))

            per_cluster_rows.append({
                "group_id": int(group_id),
                "candidate": cand,
                "cluster": cl,
                "real_prop": float(comp_real[i]),
                "candidate_prop": float(comp_c[i]),
                "cluster_comp_abs_error": float(abs(comp_c[i] - comp_real[i])),
                "cluster_comp_sare": float(comp_sare[i]),
                "cluster_comp_signed_rel": float(comp_signed[i]),
                "real_n_spots": int(np.sum(real_labels == cl)),
                "candidate_n_spots": int(np.sum(cand_labels == cl)),
                "real_same_cluster_neighbor_fraction": real_diag,
                "candidate_same_cluster_neighbor_fraction": cand_diag,
                **conn_cmp,
                **gene_cmp,
                **mean_cmp,
                "real_norm_within_cluster_nn_median": float(real_nn_sum.get("norm_within_cluster_nn_median", np.nan)),
                "candidate_norm_within_cluster_nn_median": float(cand_nn_sum.get("norm_within_cluster_nn_median", np.nan)),
                "real_norm_within_cluster_nn_mean": float(real_nn_sum.get("norm_within_cluster_nn_mean", np.nan)),
                "candidate_norm_within_cluster_nn_mean": float(cand_nn_sum.get("norm_within_cluster_nn_mean", np.nan)),
                **structure_cmp,
            })

    per_cluster_df = pd.DataFrame(per_cluster_rows)

    summary_rows = []
    for cand in METHOD_ORDER:
        sub = per_cluster_df[per_cluster_df["candidate"] == cand].copy()
        summary_rows.append({
            "group_id": int(group_id),
            "candidate": cand,
            "mean_cluster_comp_sare": float(np.nanmean(sub["cluster_comp_sare"])),
            "mean_gene_total_count_total_sare": float(np.nanmean(sub["gene_total_count_total_sare"])),
            "mean_missing_detected_gene_n": float(np.nanmean(sub["missing_detected_gene_n"])),
            "mean_within_cluster_nn_wasserstein": float(np.nanmean(sub["within_cluster_nn_wasserstein"])),
            "mean_connectivity_profile_js_distance": float(np.nanmean(sub["connectivity_profile_js_distance"])),
            "mean_all_gene_mean_mae": float(np.nanmean(sub["all_gene_mean_mae"])),
            "n_clusters": int(len(cluster_order)),
            "n_common_genes": int(len(common_genes)),
        })
    summary_df = pd.DataFrame(summary_rows)

    bundle = {
        "group_id": int(group_id),
        "real_ad": method_adatas["Real"],
        "spatialz_ad": method_adatas["SpatialZ"],
        "axis_ad": method_adatas["AXIS"],
        "method_adatas": method_adatas,
        "cluster_order": cluster_order,
        "real_cluster_order": real_cluster_order,
        "cluster_colors": colors,
        "slice_spot_units": norm_unit_by_method,
        "common_genes": common_genes,
    }
    return summary_df, per_cluster_df, neighbor_df, connectivity_df, bundle


def run_evaluation(group_ids: Sequence[int] = GROUP_IDS, outdir: str = OUTDIR):
    ensure_dir(outdir)
    summary_all = []
    per_cluster_all = []
    neighbor_all = []
    connectivity_all = []
    bundle_dict = {}

    for gid in group_ids:
        print(f"[INFO] evaluating group {gid}")
        s, pc, nd, conn, bundle = evaluate_group(int(gid))
        summary_all.append(s)
        per_cluster_all.append(pc)
        neighbor_all.append(nd)
        connectivity_all.append(conn)
        bundle_dict[int(gid)] = bundle

    summary_df = pd.concat(summary_all, axis=0, ignore_index=True)
    per_cluster_df = pd.concat(per_cluster_all, axis=0, ignore_index=True)
    neighbor_df = pd.concat(neighbor_all, axis=0, ignore_index=True) if neighbor_all else pd.DataFrame()
    connectivity_df = pd.concat(connectivity_all, axis=0, ignore_index=True) if connectivity_all else pd.DataFrame()

    summary_path = os.path.join(outdir, "summary_cluster_neighborhood_connectivity_metrics.csv")
    per_cluster_path = os.path.join(outdir, "per_cluster_neighborhood_connectivity_metrics.csv")
    neighbor_path = os.path.join(outdir, "within_cluster_nn_distribution_long_table.csv")
    connectivity_long_path = os.path.join(outdir, "cluster_neighbor_connectivity_long.csv")
    connectivity_matrix_path = os.path.join(outdir, "cluster_neighbor_connectivity_matrix_fraction.csv")

    summary_df.to_csv(summary_path, index=False)
    per_cluster_df.to_csv(per_cluster_path, index=False)
    neighbor_df.to_csv(neighbor_path, index=False)
    connectivity_df.to_csv(connectivity_long_path, index=False)

    connectivity_matrix_df = connectivity_df[[
        "group_id", "method", "source_cluster", "target_cluster", "fraction", "edge_count",
        "n_source_spots", "n_neighbor_edges", "knn", "actual_k"
    ]].copy()
    connectivity_matrix_df.to_csv(connectivity_matrix_path, index=False)

    print(f"[OK] saved: {summary_path}")
    print(f"[OK] saved: {per_cluster_path}")
    print(f"[OK] saved: {neighbor_path}")
    print(f"[OK] saved: {connectivity_long_path}")
    print(f"[OK] saved: {connectivity_matrix_path}")
    return summary_df, per_cluster_df, neighbor_df, connectivity_df, bundle_dict


# ============================================================
# SUMMARY PLOTS
# ============================================================
def add_group_pair_lines(ax, df, x_col, y_col, group_col, order, color="0.80", lw=1.0, alpha=0.85):
    x_map = {c: i for i, c in enumerate(order)}
    for _, sub in df.groupby(group_col):
        sub = sub.dropna(subset=[y_col]).copy()
        if sub.shape[0] < 2:
            continue
        sub = sub.set_index(x_col)
        xs, ys = [], []
        for c in order:
            if c in sub.index:
                val = sub.loc[c, y_col]
                if np.ndim(val) > 0:
                    val = np.asarray(val).ravel()[0]
                if pd.notnull(val):
                    xs.append(x_map[c])
                    ys.append(float(val))
        if len(xs) >= 2:
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=1)


def plot_overall_summary_panels(per_cluster_df: pd.DataFrame, save_path: Optional[str] = None):
    set_cell_style()
    order = [m for m in METHOD_ORDER if m in per_cluster_df["candidate"].unique()]
    patient_order = [g for g in GROUP_IDS if g in per_cluster_df["group_id"].unique()]
    patient_palette = dict(zip(patient_order, sns.color_palette("Set2", n_colors=len(patient_order))))
    specs = [
        ("cluster_comp_sare", "Composition sARE ↓"),
        ("within_cluster_nn_wasserstein", "Cluster local-structure distance ↓"),
        ("connectivity_profile_js_distance", "Neighborhood connectivity JS distance ↓"),
        ("gene_total_count_total_sare", "All-gene total-count error ↓"),
        ("missing_detected_gene_n", "Missing real-supported genes ↓"),
        ("all_gene_mean_mae", "All-gene mean-expression MAE ↓"),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=(26.5, 4.0), squeeze=False)
    axes = axes.ravel()
    box_palette = {m: "#f1f1f1" for m in order}
    for i, (ax, (metric, title)) in enumerate(zip(axes, specs)):
        sns.boxplot(data=per_cluster_df, x="candidate", y=metric, order=order, palette=box_palette,
                    showfliers=False, width=0.52, linewidth=1.05, ax=ax, zorder=1)
        sns.stripplot(data=per_cluster_df, x="candidate", y=metric, order=order, hue="group_id",
                      palette=patient_palette, dodge=False, jitter=0.18, size=5.2, alpha=0.95,
                      linewidth=0.35, edgecolor="black", ax=ax, zorder=2)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title, fontsize=12.2)
        ax.grid(axis="y", alpha=0.18)
        sns.despine(ax=ax)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:len(patient_order)], [f"P{g}" for g in patient_order], title="Patient",
                      loc="best", frameon=False, fontsize=8.5, title_fontsize=9)
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    fig.suptitle("Overall SCC cluster-fidelity summary", y=1.04, fontsize=16, fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_per_patient_cluster_composition_connectivity(
    per_cluster_df: pd.DataFrame,
    group_ids: Sequence[int] = GROUP_IDS,
    save_path: Optional[str] = None,
):
    """Per-patient/per-cluster report combining composition, structure, connectivity, and gene support."""
    set_cell_style()
    group_ids = list(group_ids)
    fig, axes = plt.subplots(len(group_ids), 7, figsize=(36.0, 3.45 * len(group_ids)), squeeze=False)

    def _grouped_bar(ax, sub: pd.DataFrame, cluster_order: Sequence[str], metric: str, title: str,
                     ylabel: str, row: int, ylim: Optional[Tuple[float, float]] = None):
        x = np.arange(len(cluster_order))
        width = 0.36
        for j, meth in enumerate(METHOD_ORDER):
            ss = sub[sub["candidate"] == meth][["cluster", metric]].drop_duplicates("cluster").set_index("cluster").reindex(cluster_order)
            xx = x + (j - 0.5) * width
            vals = ss[metric].values.astype(float)
            ax.bar(xx, vals, width=width, color=METHOD_PALETTE[meth], alpha=0.94, label=meth if row == 0 else None)
        ax.set_title(title, fontsize=12.0)
        ax.set_xticks(x); ax.set_xticklabels(cluster_order, rotation=0)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.18)
        if row == 0:
            ax.legend(loc="best", fontsize=8.3, frameon=False)

    for row, gid in enumerate(group_ids):
        sub = per_cluster_df[per_cluster_df["group_id"] == int(gid)].copy()
        cluster_order = canonical_cluster_order(sub["cluster"].astype(str).values)
        x = np.arange(len(cluster_order))

        ax = axes[row, 0]
        real_sub = sub[["cluster", "real_prop"]].drop_duplicates("cluster").set_index("cluster").reindex(cluster_order)
        ax.plot(x, real_sub["real_prop"].values, color=METHOD_PALETTE["Real"], marker="o", lw=2.3, label="Real")
        for meth in METHOD_ORDER:
            ss = sub[sub["candidate"] == meth][["cluster", "candidate_prop"]].drop_duplicates("cluster").set_index("cluster").reindex(cluster_order)
            ax.plot(x, ss["candidate_prop"].values, color=METHOD_PALETTE[meth], marker="o", lw=2.0, label=meth)
        ax.set_title(f"P{gid}: cluster composition", fontsize=12.0)
        ax.set_xticks(x); ax.set_xticklabels(cluster_order, rotation=0)
        ax.set_ylabel("Proportion")
        ax.legend(loc="best", ncol=3, fontsize=8.2, frameon=False)
        ax.grid(axis="y", alpha=0.18)

        specs = [
            ("cluster_comp_sare", f"P{gid}: composition error", "sARE", None),
            ("within_cluster_nn_wasserstein", f"P{gid}: cluster local structure", "Wasserstein", None),
            ("connectivity_profile_js_distance", f"P{gid}: neighborhood connectivity", "JS distance", (0, 1.05)),
            ("gene_total_count_total_sare", f"P{gid}: all-gene total counts", "sARE", (0, 2.05)),
            ("missing_detected_gene_n", f"P{gid}: missing real-supported genes", "# genes", None),
            ("all_gene_mean_mae", f"P{gid}: all-gene mean expression", "MAE", None),
        ]
        for col, (metric, title, ylabel, ylim) in enumerate(specs, start=1):
            _grouped_bar(axes[row, col], sub, cluster_order, metric, title, ylabel, row, ylim)

    fig.suptitle("Per-patient / per-cluster fidelity report", y=1.01, fontsize=16, fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_patient_connectivity_heatmaps(
    connectivity_df: pd.DataFrame,
    group_id: int,
    cluster_order: Sequence[str],
    save_path: Optional[str] = None,
):
    set_cell_style()
    methods = [m for m in PLOT_METHOD_ORDER if m in set(connectivity_df["method"])]
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(4.4 * n_methods, 4.2), squeeze=False)
    axes = axes.ravel()
    cmap = connectivity_cmap()
    annot = len(cluster_order) <= 10
    for i, (ax, method) in enumerate(zip(axes, methods)):
        mat = connectivity_matrix(connectivity_df, group_id, method, cluster_order, value="fraction")
        sns.heatmap(
            mat,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.6,
            linecolor="white",
            annot=annot,
            fmt=".2f",
            annot_kws={"fontsize": 7.5},
            cbar=(i == n_methods - 1),
            cbar_kws={"label": "Fraction of 6 nearest neighbors"},
        )
        ax.set_title(method, fontsize=14, fontweight="bold")
        ax.set_xlabel("Neighbor cluster")
        ax.set_ylabel("Source cluster" if i == 0 else "")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.suptitle(f"Patient {group_id}: cluster-neighborhood connectivity", y=1.03, fontsize=16, fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_all_patient_connectivity_heatmaps(connectivity_df: pd.DataFrame, bundle_dict: Dict[int, Dict], outdir: str):
    ensure_dir(outdir)
    pdf_path = os.path.join(outdir, "all_patients_cluster_neighbor_connectivity_heatmaps.pdf")
    with PdfPages(pdf_path) as pdf:
        for gid, bundle in bundle_dict.items():
            path = os.path.join(outdir, f"patient_{gid}_cluster_neighbor_connectivity_heatmap.png")
            plot_patient_connectivity_heatmaps(connectivity_df, gid, bundle["cluster_order"], save_path=path)
            # Re-create a figure for PDF to avoid reading PNG back.
            set_cell_style()
            methods = PLOT_METHOD_ORDER
            fig, axes = plt.subplots(1, len(methods), figsize=(4.4 * len(methods), 4.2), squeeze=False)
            axes = axes.ravel()
            for i, (ax, method) in enumerate(zip(axes, methods)):
                mat = connectivity_matrix(connectivity_df, gid, method, bundle["cluster_order"], value="fraction")
                sns.heatmap(
                    mat, ax=ax, cmap=connectivity_cmap(), vmin=0, vmax=1, square=True,
                    linewidths=0.6, linecolor="white", annot=len(bundle["cluster_order"]) <= 10,
                    fmt=".2f", annot_kws={"fontsize": 7.5}, cbar=(i == len(methods) - 1),
                    cbar_kws={"label": "Fraction of 6 nearest neighbors"},
                )
                ax.set_title(method, fontsize=14, fontweight="bold")
                ax.set_xlabel("Neighbor cluster")
                ax.set_ylabel("Source cluster" if i == 0 else "")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            fig.suptitle(f"Patient {gid}: cluster-neighborhood connectivity", y=1.03, fontsize=16, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"[OK] saved: {pdf_path}")


# ============================================================
# CASE-STUDY PLOTS
# ============================================================
def plot_cluster_neighborhood_panel(
    ax,
    adata: AnnData,
    cluster: str,
    cluster_colors: Dict[str, object],
    title: str,
    cluster_key: str = CLUSTER_KEY,
    coord_key: str = COORD_KEY,
    knn: int = KNN_CLUSTER_NEIGHBOR,
):
    coords, labels, _, source_idx, neighbor_idx = get_neighbor_indices_for_source_cluster(
        adata=adata, cluster=cluster, cluster_key=cluster_key, coord_key=coord_key, knn=knn,
    )
    source_mask = np.zeros(labels.shape[0], dtype=bool)
    source_mask[source_idx] = True
    neighbor_mask = np.zeros(labels.shape[0], dtype=bool)
    neighbor_mask[neighbor_idx] = True
    other_neighbor_mask = neighbor_mask & (~source_mask)

    ax.scatter(
        coords[:, 0], coords[:, 1], s=CASE_SPOT_SIZE_BACKGROUND,
        color=CASE_BACKGROUND_COLOR, alpha=CASE_BACKGROUND_ALPHA,
        edgecolors="none", rasterized=True,
    )

    # Plot neighbor spots by their cluster color. Current cluster remains black.
    neighbor_clusters = canonical_cluster_order(labels[other_neighbor_mask]) if np.any(other_neighbor_mask) else []
    for cl in neighbor_clusters:
        m = other_neighbor_mask & (labels == str(cl))
        if not np.any(m):
            continue
        ax.scatter(
            coords[m, 0], coords[m, 1], s=CASE_SPOT_SIZE_NEIGHBOR,
            color=cluster_colors.get(str(cl), stable_cluster_color(str(cl))),
            alpha=CASE_NEIGHBOR_ALPHA, edgecolors="none", rasterized=True,
        )

    if np.any(source_mask):
        ax.scatter(
            coords[source_mask, 0], coords[source_mask, 1], s=CASE_SPOT_SIZE_SOURCE,
            color="black", alpha=CASE_SOURCE_ALPHA, edgecolors="none", rasterized=True,
        )

    n_src = int(source_mask.sum())
    n_neighbor = int(other_neighbor_mask.sum())
    ax.text(
        0.03, 0.97,
        f"source n={n_src}\nneighbor n={n_neighbor}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.82", alpha=0.92),
    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12, fontweight="bold")


def get_case_metric_rows(per_cluster_df: pd.DataFrame, group_id: int, cluster: str) -> pd.DataFrame:
    return per_cluster_df[
        (per_cluster_df["group_id"] == int(group_id))
        & (per_cluster_df["cluster"].astype(str) == str(cluster))
    ].copy()


def get_case_neighbor_distances(bundle: Dict, cluster: str) -> Dict[str, np.ndarray]:
    out = {}
    for meth, key in [("Real", "real_ad"), ("SpatialZ", "spatialz_ad"), ("AXIS", "axis_ad")]:
        _, norm_d, _ = within_slice_same_cluster_nn_distances(
            adata=bundle[key],
            cluster=str(cluster),
            cluster_key=CLUSTER_KEY,
            coord_key=COORD_KEY,
            normalization_spot_unit=bundle["slice_spot_units"].get(meth, None),
        )
        out[meth] = norm_d
    return out


def plot_cluster_composition_axis(ax, metric_rows: pd.DataFrame, cluster: str):
    real_prop = float(metric_rows["real_prop"].iloc[0]) if len(metric_rows) else np.nan
    labels = ["Real"] + METHOD_ORDER
    props = [real_prop]
    sare_values = [0.0]
    for meth in METHOD_ORDER:
        rr = metric_rows[metric_rows["candidate"] == meth]
        if len(rr):
            props.append(float(rr["candidate_prop"].iloc[0]))
            sare_values.append(float(rr["cluster_comp_sare"].iloc[0]))
        else:
            props.append(np.nan)
            sare_values.append(np.nan)
    x = np.arange(len(labels))
    ax.bar(x, props, color=[METHOD_PALETTE[m] for m in labels], width=0.58, alpha=0.94)
    if np.isfinite(real_prop):
        ax.axhline(real_prop, color=METHOD_PALETTE["Real"], linestyle="--", linewidth=1.6, alpha=0.9)
    finite = np.asarray(props, dtype=float)
    finite = finite[np.isfinite(finite)]
    ymax = max(float(np.max(finite)) if finite.size else 1.0, 0.02)
    for j, meth in enumerate(labels):
        if meth == "Real" or not np.isfinite(props[j]) or not np.isfinite(real_prop):
            continue
        ax.text(j, props[j] + 0.04 * ymax, f"sARE={sare_values[j]:.2f}", ha="center", va="bottom", fontsize=8.5, rotation=90, color=METHOD_PALETTE[meth])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, ymax * 1.55)
    ax.set_title(f"Cluster {cluster} composition", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.18)


def plot_cluster_connectivity_profile_axis(
    ax,
    connectivity_df: pd.DataFrame,
    group_id: int,
    cluster: str,
    cluster_order: Sequence[str],
):
    x = np.arange(len(cluster_order))
    real_vec = connectivity_row_vector(connectivity_df, group_id, "Real", cluster, cluster_order)
    for meth in PLOT_METHOD_ORDER:
        vec = connectivity_row_vector(connectivity_df, group_id, meth, cluster, cluster_order)
        lw = 2.7 if meth == "Real" else 2.1
        marker = "o" if meth == "Real" else "s"
        ax.plot(x, vec, color=METHOD_PALETTE[meth], marker=marker, lw=lw, ms=4.5, label=meth, alpha=0.95)
    ann_lines = []
    for meth in METHOD_ORDER:
        vec = connectivity_row_vector(connectivity_df, group_id, meth, cluster, cluster_order)
        cmp = connectivity_profile_distances(real_vec, vec)
        ann_lines.append(f"{meth}: JS={cmp['connectivity_profile_js_distance']:.3f}" if np.isfinite(cmp['connectivity_profile_js_distance']) else f"{meth}: JS=NA")
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_order, rotation=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Neighbor cluster")
    ax.set_ylabel("Fraction among 6-NN")
    ax.set_title("Neighborhood connectivity profile", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper right", ncol=3, fontsize=8.3, frameon=False)
    ax.text(0.97, 0.96, "\n".join(ann_lines), transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.85", alpha=0.95))


def plot_neighbor_distance_ecdf_axis(ax, neighbor_dict: Dict[str, np.ndarray]):
    def _ecdf_xy(arr: np.ndarray):
        x = np.asarray(arr, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.array([]), np.array([])
        x = np.sort(x)
        y = np.arange(1, x.size + 1, dtype=np.float64) / x.size
        return x, y

    for meth in PLOT_METHOD_ORDER:
        d = np.asarray(neighbor_dict.get(meth, []), dtype=np.float64)
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        x, y = _ecdf_xy(d)
        ax.step(x, y, where="post", color=METHOD_PALETTE[meth], lw=2.8 if meth == "Real" else 2.1, alpha=0.98,
                label=f"{meth} (n={len(d)})")

    real = np.asarray(neighbor_dict.get("Real", []), dtype=np.float64)
    real = real[np.isfinite(real)]
    ann_lines = []
    for meth in METHOD_ORDER:
        cand = np.asarray(neighbor_dict.get(meth, []), dtype=np.float64)
        cand = cand[np.isfinite(cand)]
        if real.size == 0 or cand.size == 0:
            ann_lines.append(f"{meth}: W1=NA")
            continue
        cmp = nn_distribution_similarity_to_real(real, cand)
        ann_lines.append(f"{meth}: W1={cmp['within_cluster_nn_wasserstein']:.3f}")

    pooled = []
    for meth in PLOT_METHOD_ORDER:
        arr = np.asarray(neighbor_dict.get(meth, []), dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            pooled.append(arr)
    if pooled:
        pooled = np.concatenate(pooled)
        x_upper = max(np.percentile(pooled, 99.2), 1.05)
        ax.set_xlim(left=0, right=x_upper * 1.03)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Nearest same-cluster distance\n(spot-spacing units)")
    ax.set_ylabel("ECDF")
    ax.set_title("Cluster compactness / local structure", fontsize=12, fontweight="bold")
    ax.grid(axis="both", alpha=0.18)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.text(0.97, 0.24, "\n".join(ann_lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.85", alpha=0.95))


def add_cluster_color_legend(fig, cluster_order: Sequence[str], cluster_colors: Dict[str, object], ncol: int = 10):
    handles = [Line2D([0], [0], marker="o", linestyle="None", color="black", markersize=6.5, label="current cluster")]
    handles.extend([
        Line2D([0], [0], marker="o", linestyle="None", color=cluster_colors[str(c)], markersize=6.5, label=f"neighbor {c}")
        for c in cluster_order
    ])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=min(ncol, len(handles)), frameon=False, fontsize=8.5)


def marker_gene_values_for_valid_coords(adata: AnnData, gene: str, coord_key: str = COORD_KEY) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords_all = np.asarray(adata.obsm[coord_key], dtype=np.float64)
    labels_all = get_cluster_labels(adata, CLUSTER_KEY)
    finite = np.all(np.isfinite(coords_all), axis=1)
    coords = coords_all[finite]
    labels = labels_all[finite].astype(str)
    vals = extract_gene_vector(adata, gene)[finite]
    if MARKER_PLOT_LOG1P:
        vals = np.log1p(np.maximum(vals, 0.0))
    return coords, labels, vals.astype(np.float32)


def plot_cluster_marker_gene_axis(
    ax,
    adata: AnnData,
    cluster: str,
    gene: str,
    title: str,
    vmin: float,
    vmax: float,
    real_cluster_mean_raw: Optional[float] = None,
    cmap: str = MARKER_CMAP,
):
    coords, labels, vals = marker_gene_values_for_valid_coords(adata, gene)
    coords_all = np.asarray(adata.obsm[COORD_KEY], dtype=np.float64)
    finite = np.all(np.isfinite(coords_all), axis=1)
    raw_vals = extract_gene_vector(adata, gene)[finite]
    mask = labels == str(cluster)
    cluster_raw = np.maximum(raw_vals[mask], 0.0) if raw_vals.shape[0] == mask.shape[0] else np.array([], dtype=np.float64)
    total_gene_reads = float(np.nansum(cluster_raw)) if cluster_raw.size else np.nan
    cluster_mean_raw = float(np.nanmean(cluster_raw)) if cluster_raw.size else np.nan
    mean_abs_error = abs(cluster_mean_raw - real_cluster_mean_raw) if (real_cluster_mean_raw is not None and np.isfinite(real_cluster_mean_raw) and np.isfinite(cluster_mean_raw)) else np.nan
    if real_cluster_mean_raw is None and np.isfinite(cluster_mean_raw):
        mean_abs_error = 0.0

    ax.scatter(coords[:, 0], coords[:, 1], s=CASE_SPOT_SIZE_BACKGROUND, color=CASE_BACKGROUND_COLOR,
               alpha=0.38, edgecolors="none", rasterized=True)
    if np.any(mask):
        sca = ax.scatter(coords[mask, 0], coords[mask, 1], c=vals[mask], cmap=cmap, vmin=vmin, vmax=vmax,
                         s=CASE_SPOT_SIZE_MARKER, edgecolors="none", rasterized=True)
        ax.text(0.03, 0.97, f"total = {total_gene_reads:.1f}\nmean abs. error = {mean_abs_error:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="0.82", alpha=0.92))
    else:
        sca = None
        ax.text(0.5, 0.5, "cluster absent", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11.3, fontweight="bold")
    return sca


def marker_gene_vmax_for_case(method_to_ad: Dict[str, AnnData], cluster: str, gene: str) -> float:
    pooled = []
    for ad in method_to_ad.values():
        coords, labels, vals = marker_gene_values_for_valid_coords(ad, gene)
        m = labels == str(cluster)
        if np.any(m):
            pooled.append(vals[m][np.isfinite(vals[m])])
    if not pooled:
        return 1.0
    vals = np.concatenate([x for x in pooled if x.size > 0])
    if vals.size == 0:
        return 1.0
    return max(float(np.nanpercentile(vals, 99.0)), 1e-6)



def plot_cluster_case_study(
    group_id: int,
    cluster: str,
    bundle: Dict,
    per_cluster_df: pd.DataFrame,
    connectivity_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    set_cell_style()
    metric_rows = get_case_metric_rows(per_cluster_df, group_id, cluster)
    cluster_order = bundle["cluster_order"]
    neighbor_dict = get_case_neighbor_distances(bundle, cluster)

    method_to_ad = {
        "Real": bundle["real_ad"],
        "SpatialZ": bundle["spatialz_ad"],
        "AXIS": bundle["axis_ad"],
    }

    fig = plt.figure(figsize=(16.9, 14.4))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1.0, 1.0, 1.0, 0.95], hspace=0.30, wspace=0.28)

    for row, gene in enumerate(CASE_MARKER_GENES):
        vmax = marker_gene_vmax_for_case(method_to_ad, cluster, gene)
        real_vals = extract_gene_vector(bundle["real_ad"], gene)
        real_labels = get_cluster_labels(bundle["real_ad"], CLUSTER_KEY)
        real_mean = float(np.nanmean(real_vals[real_labels == str(cluster)])) if np.any(real_labels == str(cluster)) else np.nan
        last_sca = None
        for col, meth in enumerate(PLOT_METHOD_ORDER):
            ax = fig.add_subplot(gs[row, col])
            sca = plot_cluster_marker_gene_axis(
                ax=ax,
                adata=method_to_ad[meth],
                cluster=cluster,
                gene=gene,
                title=f"{meth}: {gene} in cluster {cluster}",
                vmin=0.0,
                vmax=vmax,
                real_cluster_mean_raw=real_mean,
            )
            if sca is not None:
                last_sca = sca
        if last_sca is not None:
            cax = fig.add_axes([0.925, 0.74 - row * 0.25, 0.010, 0.16])
            cb = fig.colorbar(last_sca, cax=cax)
            cb.set_label(f"{gene} {'log1p' if MARKER_PLOT_LOG1P else 'count'}", fontsize=8.5)
            cb.ax.tick_params(labelsize=8)

    for col, meth in enumerate(PLOT_METHOD_ORDER):
        ax = fig.add_subplot(gs[2, col])
        plot_cluster_neighborhood_panel(
            ax=ax,
            adata=method_to_ad[meth],
            cluster=cluster,
            cluster_colors=bundle["cluster_colors"],
            title=f"{meth}: cluster {cluster} and 6-NN context",
        )

    ax_comp = fig.add_subplot(gs[3, 0])
    plot_cluster_composition_axis(ax_comp, metric_rows, cluster)
    ax_nn = fig.add_subplot(gs[3, 1])
    plot_neighbor_distance_ecdf_axis(ax_nn, neighbor_dict)
    ax_conn = fig.add_subplot(gs[3, 2])
    plot_cluster_connectivity_profile_axis(ax_conn, connectivity_df, group_id, cluster, cluster_order)

    add_cluster_color_legend(fig, cluster_order, bundle["cluster_colors"], ncol=8)
    fig.suptitle(
        f"Patient {group_id} | cluster {cluster}: marker expression, spatial context, and neighborhood fidelity",
        y=0.995, fontsize=16, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.965])
    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    return fig


def plot_all_clusters_for_case_patient(
    group_id: int,
    bundle: Dict,
    per_cluster_df: pd.DataFrame,
    connectivity_df: pd.DataFrame,
    outdir: str,
):
    ensure_dir(outdir)
    cluster_order = [str(x) for x in bundle["real_cluster_order"]]
    pdf_path = os.path.join(outdir, f"patient_{group_id}_all_clusters_case_study.pdf")
    with PdfPages(pdf_path) as pdf:
        for cl in cluster_order:
            png_path = None
            if SAVE_INDIVIDUAL_CLUSTER_PNG:
                png_path = os.path.join(outdir, f"P{group_id}_cluster_{sanitize_filename_token(cl)}_case_study.png")
            fig = plot_cluster_case_study(
                group_id=group_id,
                cluster=cl,
                bundle=bundle,
                per_cluster_df=per_cluster_df,
                connectivity_df=connectivity_df,
                save_path=png_path,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"[OK] saved: {pdf_path}")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    ensure_dir(OUTDIR)
    set_cell_style()

    summary_df, per_cluster_df, neighbor_df, connectivity_df, bundle_dict = run_evaluation(
        group_ids=GROUP_IDS,
        outdir=OUTDIR,
    )

    plot_overall_summary_panels(
        per_cluster_df=per_cluster_df,
        save_path=os.path.join(OUTDIR, "overall_summary_cluster_neighborhood_connectivity.png"),
    )

    plot_per_patient_cluster_composition_connectivity(
        per_cluster_df=per_cluster_df,
        group_ids=GROUP_IDS,
        save_path=os.path.join(OUTDIR, "per_patient_cluster_composition_connectivity.png"),
    )

    for gid in CASE_GROUPS:
        if gid not in bundle_dict:
            continue
        case_dir = os.path.join(OUTDIR, f"patient_{gid}_cluster_neighborhood_cases")
        plot_all_clusters_for_case_patient(
            group_id=int(gid),
            bundle=bundle_dict[int(gid)],
            per_cluster_df=per_cluster_df,
            connectivity_df=connectivity_df,
            outdir=case_dir,
        )

    print("[DONE] SCC cluster fidelity evaluation finished.")


if __name__ == "__main__":
    main()
