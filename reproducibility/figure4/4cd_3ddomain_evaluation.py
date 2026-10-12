#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLPFC 3D domain-continuity and neighborhood-connectivity evaluation.

This script evaluates SpatialZ and AXIS virtual 3D DLPFC reconstructions using
three complementary outputs:

1. 3D domain continuity
   For each virtual spot, measure the fraction of selected cross-slice 3D
   neighbors carrying the same layer/domain label. Scores are summarized by
   domain for SpatialZ and AXIS.

2. 2D-reference neighborhood agreement
   Build a real 2D neighborhood-composition reference by computing a row-
   normalized KNN domain-connectivity matrix independently in each observed
   real slice and averaging the matrices with equal slice weight. For each
   reconstructed method and source domain, compare its 3D cross-slice
   neighborhood-composition row with the corresponding real 2D reference row:

       agreement = 1 - 0.5 * sum_j |p_3D(j) - p_2D_ref(j)|

   This mirrors the neighborhood-composition / total-variation design used in
   the mouse-brain analysis, while using the available 2D observations as the
   reference because no experimental 3D reference volume exists for DLPFC.

3. 3D neighborhood-connectivity heatmaps
   Plot the row-normalized source-domain -> neighbor-domain matrices for
   SpatialZ and AXIS. Both use the white-to-red half of seaborn's ``vlag``
   colormap. Each method has its own colorbar axis; by default both panels use
   the same normalization for direct comparison.

Evaluation and plotting are separated. Use:

    python dlpfc_3d_domain_evaluation_with_2d_reference_agreement.py --mode eval
    python dlpfc_3d_domain_evaluation_with_2d_reference_agreement.py --mode plot
    python dlpfc_3d_domain_evaluation_with_2d_reference_agreement.py --mode both

The neighborhood-diagnosis case-study plotting code from the previous version
has been removed.
"""

from __future__ import annotations
from axis.settings import data_path
from axis.config import load_config

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


# ============================================================
# DATA / EVALUATION CONFIG
# ============================================================
ROOT = load_config('dlpfc').root
OUTDIR = data_path("paper_analysis/Figure4_dlfpc/cd_domain_metrics/")
os.makedirs(OUTDIR, exist_ok=True)

COORD_KEY = "spatial"
DOMAIN_KEY = "layer_guess_reordered"
# DOMAIN_KEY = "axis_joint_domain"

REFERENCE_REAL_Z = 10
VISIUM_NEAREST_SPOT_DISTANCE_UM = 100.0

# Fair comparison: use the common subset available for both methods.
VIRTUAL_DEPTHS = np.arange(40, 310, 30)
Z_ORDER = [10] + VIRTUAL_DEPTHS.tolist() + [310]

REAL_SLICE_PATHS: Dict[int, str] = {
    10: os.path.join(ROOT, "h5ad", "1_1.h5ad"),
    310: os.path.join(ROOT, "h5ad", "1_2.h5ad"),
}

SPATIALZ_SLICE_PATHS: Dict[int, str] = {
    int(depth): os.path.join(
        ROOT,
        "virtual_tissue",
        "spatialz",
        "dense_reconstruction",
        "group1",
        f"spatialz_{depth}.000.h5ad",
    )
    for depth in VIRTUAL_DEPTHS
}

AXIS_SLICE_PATHS: Dict[int, str] = {
    int(depth): os.path.join(
        ROOT,
        "virtual_tissue",
        "dlpfc_z_group1_grid80_spatial1",
        "query_full_gene_set",
        f"virtual_dlpfc_z_{depth}.000.h5ad",
    )
    for depth in VIRTUAL_DEPTHS
}

METHOD_ORDER = [ "AXIS","SpatialZ"]
METHOD_PALETTE = {
    "SpatialZ": "#7C8DA6",
    "AXIS": "#C4473A",
}

# Score only virtual slices by default. The real slices are used separately to
# build the observed 2D neighborhood reference.
USE_REAL_ANCHORS_IN_SCORE = False
Z_ORDER_SCORE = Z_ORDER if USE_REAL_ANCHORS_IN_SCORE else VIRTUAL_DEPTHS.tolist()

# Cross-slice-only scoring is usually most meaningful for 3D continuity.
INCLUDE_SAME_SLICE_NEIGHBORS = False

# Number of neighbors considered in each eligible target slice.
# None = all radius-limited neighbors.
N_NEIGHBORS_PER_TARGET_SLICE: Optional[int] = None

# Nearby-slice rule. max_z_diff_um = MAX_Z_DIFF_FACTOR * median z step.
MAX_Z_DIFF_FACTOR = 2.0

# Physical 3D radius.
# radius_um = RADIUS_FACTOR * sqrt((100 µm)^2 + (median dz)^2)
RADIUS_FACTOR = 2.0

MIN_POINTS_PER_DOMAIN = 80
INVALID_DOMAIN_LABELS = {"nan", "NaN", "None", "", "unknown"}

# Real 2D reference neighborhood definition. Keeping k=50 matches the mouse-
# brain neighborhood-component analysis design.
REFERENCE_2D_KNN_K = 50

SAVE_PER_SPOT = True
FIG_DPI = 300


# ============================================================
# FIGURE STYLE CONFIG
# Centralized so geometry / typography can be adjusted in one place.
# ============================================================
@dataclass(frozen=True)
class MetricFigureStyle:
    figsize: Tuple[float, float] = (5, 6.0)
    box_width: float = 0.52
    box_linewidth: float = 1.1
    point_size: float = 95.0
    point_jitter: float = 0.08
    point_edge_width: float = 0.7
    title_fontsize: float = 16.0
    label_fontsize: float = 16.0
    tick_fontsize: float = 15.0
    legend_fontsize: float = 10.0
    legend_title_fontsize: float = 13.0
    legend_bbox: Tuple[float, float] = (1.1, 0.76)
    grid_alpha: float = 0.18
    # Set to e.g. (0.5, 1.0) if a fixed range is desired for both metrics.
    ylim: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class HeatmapFigureStyle:
    figure_width: float = 8.4
    row_height: float = 4.65
    heatmap_to_cbar_width_ratio: Tuple[float, float] = (1., 0.025)
    hspace: float = 0.34
    wspace: float = -0.5
    title_fontsize: float = 22.0
    axis_label_fontsize: float = 18.0
    tick_fontsize: float = 15.0
    annotation_fontsize: float = 18.0
    cbar_tick_fontsize: float = 16.0
    cbar_label_fontsize: float = 16.0
    linewidths: float = 1.05
    linecolor: str = "#f3f3f3"
    annotation_white_threshold_fraction: float = 0.45
    # True = method panels are directly comparable; each still has its own cbar.
    share_color_scale: bool = True
    power_gamma: float = 0.7


METRIC_STYLE = MetricFigureStyle()
HEATMAP_STYLE = HeatmapFigureStyle()


# ============================================================
# OUTPUTS
# ============================================================
OVERALL_CSV = os.path.join(OUTDIR, "overall_cross_slice_domain_continuity.csv")
PER_DOMAIN_CSV = os.path.join(OUTDIR, "per_domain_cross_slice_domain_continuity.csv")
PER_SPOT_CSV = os.path.join(OUTDIR, "per_spot_cross_slice_domain_continuity.csv")
CONNECTIVITY_LONG_CSV = os.path.join(OUTDIR, "neighborhood_connectivity_long.csv")
CONNECTIVITY_MATRIX_CSV = os.path.join(OUTDIR, "neighborhood_connectivity_matrix_fraction.csv")

REFERENCE_2D_MATRIX_CSV = os.path.join(
    OUTDIR, "real_2d_neighborhood_reference_matrix_fraction.csv"
)
REFERENCE_2D_AGREEMENT_CSV = os.path.join(
    OUTDIR, "per_domain_2d_reference_neighborhood_agreement.csv"
)
ANALYSIS_CONFIG_JSON = os.path.join(OUTDIR, "analysis_config.json")

COMBINED_CONTINUITY_FIG = os.path.join(OUTDIR, "combined_3d_domain_continuity.png")
REFERENCE_2D_AGREEMENT_FIG = os.path.join(
    OUTDIR, "2d_reference_neighborhood_agreement.png"
)
CONNECTIVITY_HEATMAP_SPATIALZ_FIG = os.path.join(
    OUTDIR, "neighborhood_connectivity_heatmap_spatialz.png"
)
CONNECTIVITY_HEATMAP_AXIS_FIG = os.path.join(
    OUTDIR, "neighborhood_connectivity_heatmap_axis.png"
)


# ============================================================
# BASIC HELPERS
# ============================================================
def domain_sort_key(x: object) -> Tuple[int, object]:
    s = str(x)
    try:
        return (0, int(float(s)))
    except Exception:
        return (1, s)


def sort_domains(domains: Iterable[object]) -> List[str]:
    values = [str(x) for x in domains if str(x) not in INVALID_DOMAIN_LABELS]
    return sorted(pd.unique(pd.Series(values)).tolist(), key=domain_sort_key)


def load_slice_dict(path_dict: Mapping[int, str]) -> Dict[int, sc.AnnData]:
    out: Dict[int, sc.AnnData] = {}
    for z, path in path_dict.items():
        if path is None or str(path).strip() == "":
            continue
        if not os.path.exists(path):
            print(f"[WARN] missing file for z={z}: {path}")
            continue
        adata = sc.read_h5ad(path)
        if "valid" in adata.obs.columns:
            adata = adata[adata.obs["valid"].astype(bool)].copy()
        out[int(z)] = adata
    return out


def get_coords(adata: sc.AnnData, coord_key: str = COORD_KEY) -> np.ndarray:
    if coord_key not in adata.obsm:
        raise KeyError(f"{coord_key!r} not found in adata.obsm")
    coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coordinate shape: {coords.shape}")
    return coords[:, :2]


def get_labels(adata: sc.AnnData, domain_key: str = DOMAIN_KEY) -> np.ndarray:
    if domain_key not in adata.obs:
        raise KeyError(f"{domain_key!r} not found in adata.obs")
    return adata.obs[domain_key].astype(str).to_numpy()


def get_slice_for_z(
    real_ads: Mapping[int, sc.AnnData],
    method_ads: Mapping[int, sc.AnnData],
    z: int,
) -> Tuple[Optional[sc.AnnData], bool]:
    if z in real_ads:
        return real_ads[z], True
    if z in method_ads:
        return method_ads[z], False
    return None, False


def infer_network_nn_distance(
    adata_ref: sc.AnnData,
    coord_key: str = COORD_KEY,
) -> float:
    coords = get_coords(adata_ref, coord_key=coord_key)
    coords = coords[np.all(np.isfinite(coords), axis=1)]
    if coords.shape[0] < 3:
        raise ValueError("Too few spots to infer nearest-neighbor spacing.")
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nn = distances[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        raise ValueError("Failed to infer nearest-neighbor distance.")
    return float(np.median(nn))


def collect_domain_order(
    *ads_groups: Mapping[int, sc.AnnData],
    domain_key: str = DOMAIN_KEY,
) -> List[str]:
    labels: List[str] = []
    for ads in ads_groups:
        for adata in ads.values():
            labels.extend(get_labels(adata, domain_key=domain_key).tolist())
    return sort_domains(labels)


def compute_radius_and_z_threshold(
    z_order_score: Sequence[int],
    visium_nn_um: float = VISIUM_NEAREST_SPOT_DISTANCE_UM,
    radius_factor: float = RADIUS_FACTOR,
    max_z_diff_factor: float = MAX_Z_DIFF_FACTOR,
) -> Tuple[float, float, float]:
    z_sorted = np.sort(np.asarray(z_order_score, dtype=np.float32))
    dz_med = float(np.median(np.diff(z_sorted))) if z_sorted.size >= 2 else 0.0
    radius_um = float(radius_factor * np.sqrt(visium_nn_um**2 + dz_med**2))
    max_z_diff_um = (
        float(max_z_diff_factor * dz_med) if dz_med > 0 else np.inf
    )
    return radius_um, max_z_diff_um, dz_med


# ============================================================
# PHYSICAL SLICE DATA
# ============================================================
def build_physical_slice_data(
    real_ads: Mapping[int, sc.AnnData],
    method_ads: Mapping[int, sc.AnnData],
    z_order_score: Sequence[int],
    xy_scale_to_um: float,
    domain_key: str = DOMAIN_KEY,
    coord_key: str = COORD_KEY,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Build per-slice XY coordinates in physical µm and valid domain labels."""
    slice_data: Dict[int, Dict[str, np.ndarray]] = {}
    for z in z_order_score:
        adata, is_real = get_slice_for_z(real_ads, method_ads, int(z))
        if adata is None:
            print(f"[WARN] missing scored slice at z={z}")
            continue

        xy = get_coords(adata, coord_key=coord_key)
        labels = get_labels(adata, domain_key=domain_key)
        valid = np.all(np.isfinite(xy), axis=1) & np.asarray(
            [str(label) not in INVALID_DOMAIN_LABELS for label in labels],
            dtype=bool,
        )

        slice_data[int(z)] = {
            "xy_um": xy[valid].astype(np.float32) * float(xy_scale_to_um),
            "labels": labels[valid].astype(str),
            "z_um": np.float32(z),
            "is_real": np.array(bool(is_real)),
        }
    return slice_data


# ============================================================
# 3D NEIGHBORHOOD CONTINUITY + CONNECTIVITY
# ============================================================
def eligible_target_slices(
    z_src: int,
    z_available: Sequence[int],
    radius_um: float,
    max_z_diff_um: float,
    include_same_slice_neighbors: bool,
) -> List[int]:
    out: List[int] = []
    for z_tgt in z_available:
        dz = abs(float(z_tgt) - float(z_src))
        if (not include_same_slice_neighbors) and (z_tgt == z_src):
            continue
        if dz > max_z_diff_um + 1e-8:
            continue
        if dz >= radius_um:
            continue
        out.append(int(z_tgt))
    return out


def query_neighbors_with_optional_topk(
    tree: cKDTree,
    xy_src: np.ndarray,
    radius_xy: float,
    n_neighbors: Optional[int] = N_NEIGHBORS_PER_TARGET_SLICE,
) -> List[np.ndarray]:
    """Return radius-limited target-slice neighbors for each source spot."""
    if n_neighbors is None:
        raw = tree.query_ball_point(xy_src, r=float(radius_xy))
        return [np.asarray(indices, dtype=np.int64) for indices in raw]

    k = int(n_neighbors)
    if k <= 0:
        raise ValueError("N_NEIGHBORS_PER_TARGET_SLICE must be positive or None.")

    distances, indices = tree.query(
        xy_src,
        k=k,
        distance_upper_bound=float(radius_xy),
    )
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    n_target = tree.n
    out: List[np.ndarray] = []
    for d_row, i_row in zip(distances, indices):
        keep = np.isfinite(d_row) & (i_row >= 0) & (i_row < n_target)
        out.append(i_row[keep].astype(np.int64, copy=False))
    return out


def connectivity_counts_to_long(
    counts: np.ndarray,
    domain_order: Sequence[str],
) -> pd.DataFrame:
    domain_order = list(domain_order)
    row_total = counts.sum(axis=1)
    all_total = int(counts.sum())
    rows: List[Dict[str, object]] = []

    for i, src in enumerate(domain_order):
        for j, tgt in enumerate(domain_order):
            n_pairs = int(counts[i, j])
            rows.append(
                {
                    "source_domain": str(src),
                    "target_domain": str(tgt),
                    "n_neighbor_pairs": n_pairs,
                    "source_domain_neighbor_pairs": int(row_total[i]),
                    "fraction_of_source_domain_neighbors": (
                        float(n_pairs / row_total[i]) if row_total[i] > 0 else np.nan
                    ),
                    "fraction_of_all_neighbor_pairs": (
                        float(n_pairs / all_total) if all_total > 0 else np.nan
                    ),
                    "is_same_domain": bool(str(src) == str(tgt)),
                }
            )
    return pd.DataFrame(rows)


def compute_neighbor_purity_and_connectivity(
    slice_data: Dict[int, Dict[str, np.ndarray]],
    z_order_score: Sequence[int],
    domain_order: Sequence[str],
    radius_um: float,
    max_z_diff_um: float,
    include_same_slice_neighbors: bool = INCLUDE_SAME_SLICE_NEIGHBORS,
    n_neighbors_per_target_slice: Optional[int] = N_NEIGHBORS_PER_TARGET_SLICE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-spot 3D purity and source-domain -> neighbor-domain connectivity."""
    z_available = [int(z) for z in z_order_score if int(z) in slice_data]
    if not z_available:
        raise ValueError("No scored slices available.")

    domain_order = list(domain_order)
    domain_to_idx = {domain: i for i, domain in enumerate(domain_order)}
    n_domains = len(domain_order)
    connectivity_counts = np.zeros((n_domains, n_domains), dtype=np.int64)

    trees = {z: cKDTree(slice_data[z]["xy_um"]) for z in z_available}
    label_codes = {
        z: np.asarray(
            [domain_to_idx[str(label)] for label in slice_data[z]["labels"]],
            dtype=np.int32,
        )
        for z in z_available
    }

    per_spot_parts: List[pd.DataFrame] = []

    for z_src in z_available:
        xy_src = slice_data[z_src]["xy_um"]
        labels_src = slice_data[z_src]["labels"].astype(str)
        src_codes = label_codes[z_src]
        n_src = xy_src.shape[0]

        total_neighbors = np.zeros(n_src, dtype=np.int32)
        same_neighbors = np.zeros(n_src, dtype=np.int32)

        for z_tgt in eligible_target_slices(
            z_src=z_src,
            z_available=z_available,
            radius_um=radius_um,
            max_z_diff_um=max_z_diff_um,
            include_same_slice_neighbors=include_same_slice_neighbors,
        ):
            dz = abs(float(z_tgt) - float(z_src))
            radius_xy = np.sqrt(max(radius_um**2 - dz**2, 0.0))
            if radius_xy <= 0:
                continue

            target_codes = label_codes[z_tgt]
            neighbor_lists = query_neighbors_with_optional_topk(
                tree=trees[z_tgt],
                xy_src=xy_src,
                radius_xy=float(radius_xy),
                n_neighbors=n_neighbors_per_target_slice,
            )

            for i, indices in enumerate(neighbor_lists):
                if indices.size == 0:
                    continue
                if include_same_slice_neighbors and z_tgt == z_src:
                    indices = indices[indices != i]
                    if indices.size == 0:
                        continue

                target_codes_i = target_codes[indices]
                src_code = int(src_codes[i])
                counts = np.bincount(target_codes_i, minlength=n_domains)

                connectivity_counts[src_code, :] += counts
                total_neighbors[i] += int(target_codes_i.size)
                same_neighbors[i] += int(counts[src_code])

        has_neighbors = total_neighbors > 0
        purity = np.full(n_src, np.nan, dtype=np.float32)
        purity[has_neighbors] = (
            same_neighbors[has_neighbors] / total_neighbors[has_neighbors]
        )

        per_spot_parts.append(
            pd.DataFrame(
                {
                    "slice_z": float(z_src),
                    "domain": labels_src,
                    "n_neighbors": total_neighbors.astype(int),
                    "n_same_domain_neighbors": same_neighbors.astype(int),
                    "neighbor_purity": purity.astype(float),
                    "has_neighbors": has_neighbors.astype(bool),
                }
            )
        )

    per_spot_df = pd.concat(per_spot_parts, axis=0, ignore_index=True)
    connectivity_df = connectivity_counts_to_long(connectivity_counts, domain_order)
    return per_spot_df, connectivity_df


def summarize_neighbor_continuity(
    per_spot_df: pd.DataFrame,
    connectivity_df: pd.DataFrame,
    min_points_per_domain: int = MIN_POINTS_PER_DOMAIN,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Summarize per-domain and overall 3D domain continuity."""
    valid_df = per_spot_df[per_spot_df["has_neighbors"]].copy()
    overall_score = (
        float(valid_df["neighbor_purity"].mean()) if len(valid_df) else np.nan
    )
    overall_coverage = (
        float(per_spot_df["has_neighbors"].mean()) if len(per_spot_df) else np.nan
    )

    diagonal = connectivity_df[connectivity_df["is_same_domain"]].copy()
    diagonal = diagonal.set_index("source_domain")

    rows: List[Dict[str, object]] = []
    for domain, subset in per_spot_df.groupby("domain", sort=False):
        n_total = int(len(subset))
        if n_total < min_points_per_domain:
            continue

        valid = subset[subset["has_neighbors"]]
        n_valid = int(len(valid))
        domain_score = float(valid["neighbor_purity"].mean()) if n_valid else np.nan
        coverage = n_valid / max(n_total, 1)

        pair_purity = np.nan
        pair_count = 0
        if str(domain) in diagonal.index:
            row = diagonal.loc[str(domain)]
            pair_purity = float(row["fraction_of_source_domain_neighbors"])
            pair_count = int(row["source_domain_neighbor_pairs"])

        rows.append(
            {
                "domain": str(domain),
                "n_points": n_total,
                "n_points_with_neighbors": n_valid,
                "neighbor_coverage": float(coverage),
                "continuity_score": domain_score,
                "neighbor_pair_purity": pair_purity,
                "source_domain_neighbor_pairs": pair_count,
            }
        )

    same_pairs = connectivity_df.loc[
        connectivity_df["is_same_domain"], "n_neighbor_pairs"
    ].sum()
    all_pairs = connectivity_df["n_neighbor_pairs"].sum()

    overall = {
        "overall_continuity_score": overall_score,
        "overall_neighbor_coverage": overall_coverage,
        "overall_neighbor_pair_purity": (
            float(same_pairs / all_pairs) if all_pairs > 0 else np.nan
        ),
        "n_total_points": int(len(per_spot_df)),
        "n_points_with_neighbors": int(per_spot_df["has_neighbors"].sum()),
        "n_neighbor_pairs": int(all_pairs),
    }
    return pd.DataFrame(rows), overall


# ============================================================
# REAL 2D REFERENCE + 3D-to-2D NEIGHBORHOOD AGREEMENT
# ============================================================
def compute_2d_knn_neighborhood_matrix(
    adata: sc.AnnData,
    domain_order: Sequence[str],
    k: int = REFERENCE_2D_KNN_K,
) -> pd.DataFrame:
    """Compute a row-normalized 2D neighborhood-composition matrix.

    This matches the mouse-brain component-matrix design: each spot contributes
    a composition vector over its k nearest within-slice neighbors, and these
    vectors are averaged over spots belonging to the same focal domain.
    """
    if k <= 0:
        raise ValueError("REFERENCE_2D_KNN_K must be positive.")

    coords = get_coords(adata)
    labels = get_labels(adata).astype(str)
    valid = np.all(np.isfinite(coords), axis=1) & np.asarray(
        [label not in INVALID_DOMAIN_LABELS for label in labels],
        dtype=bool,
    )
    coords = coords[valid]
    labels = labels[valid]

    domains = [str(x) for x in domain_order]
    domain_to_index = {domain: i for i, domain in enumerate(domains)}
    label_index = np.asarray(
        [domain_to_index.get(str(label), -1) for label in labels],
        dtype=np.int64,
    )

    n_obs = coords.shape[0]
    n_domains = len(domains)
    matrix = np.full((n_domains, n_domains), np.nan, dtype=np.float32)
    if n_obs < 2:
        return pd.DataFrame(matrix, index=domains, columns=domains)

    n_neighbors = min(int(k) + 1, n_obs)
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(coords)
    neighbor_indices = model.kneighbors(coords, return_distance=False)[:, 1:]

    per_spot = np.zeros((n_obs, n_domains), dtype=np.float32)
    for i in range(n_obs):
        neighbor_codes = label_index[neighbor_indices[i]]
        neighbor_codes = neighbor_codes[neighbor_codes >= 0]
        if neighbor_codes.size == 0:
            per_spot[i, :] = np.nan
            continue
        counts = np.bincount(neighbor_codes, minlength=n_domains).astype(np.float32)
        per_spot[i, :] = counts / counts.sum()

    for domain_index in range(n_domains):
        focal_mask = label_index == domain_index
        if not np.any(focal_mask):
            continue
        rows = per_spot[focal_mask]
        rows = rows[np.all(np.isfinite(rows), axis=1)]
        if rows.size:
            matrix[domain_index, :] = np.mean(rows, axis=0)

    return pd.DataFrame(matrix, index=domains, columns=domains)


def mean_reference_matrices(
    matrices: Sequence[pd.DataFrame],
    domain_order: Sequence[str],
) -> pd.DataFrame:
    """Equal-slice-weight mean of row-normalized real 2D matrices."""
    if not matrices:
        raise ValueError("At least one real 2D reference matrix is required.")

    labels = [str(x) for x in domain_order]
    arrays = [
        matrix.reindex(index=labels, columns=labels).to_numpy(dtype=float)
        for matrix in matrices
    ]
    with np.errstate(invalid="ignore"):
        mean_values = np.nanmean(np.stack(arrays, axis=0), axis=0)
    return pd.DataFrame(mean_values, index=labels, columns=labels)


def build_real_2d_reference_matrix(
    real_ads: Mapping[int, sc.AnnData],
    domain_order: Sequence[str],
    k: int = REFERENCE_2D_KNN_K,
) -> pd.DataFrame:
    """Build the observed 2D reference by averaging per-slice matrices."""
    matrices: List[pd.DataFrame] = []
    for z in sorted(real_ads):
        matrix = compute_2d_knn_neighborhood_matrix(
            real_ads[z],
            domain_order=domain_order,
            k=k,
        )
        matrices.append(matrix)
    return mean_reference_matrices(matrices, domain_order)


def connectivity_matrix(
    connectivity_df: pd.DataFrame,
    method: str,
    domain_order: Sequence[str],
    value_col: str = "fraction_of_source_domain_neighbors",
) -> pd.DataFrame:
    subset = connectivity_df[connectivity_df["method"] == method]
    if subset.empty:
        raise KeyError(f"No connectivity rows found for method {method!r}.")
    matrix = subset.pivot(
        index="source_domain",
        columns="target_domain",
        values=value_col,
    )
    labels = [str(x) for x in domain_order]
    return matrix.reindex(index=labels, columns=labels).astype(float)


def neighborhood_composition_agreement(
    query_row: np.ndarray,
    reference_row: np.ndarray,
) -> float:
    """Return 1 - total-variation distance between two composition vectors."""
    query = np.asarray(query_row, dtype=float)
    reference = np.asarray(reference_row, dtype=float)
    valid = np.isfinite(query) & np.isfinite(reference)
    if not np.any(valid):
        return np.nan

    query = query[valid]
    reference = reference[valid]

    # For supported source domains both rows should already sum to one. Renormalize
    # over the common finite support only to make the comparison robust to missing
    # domains without changing the intended metric.
    query_sum = float(np.sum(query))
    reference_sum = float(np.sum(reference))
    if query_sum <= 0 or reference_sum <= 0:
        return np.nan
    query = query / query_sum
    reference = reference / reference_sum

    tv_distance = 0.5 * float(np.sum(np.abs(query - reference)))
    return float(np.clip(1.0 - tv_distance, 0.0, 1.0))


def compute_2d_reference_agreement_table(
    connectivity_df: pd.DataFrame,
    real_reference_2d: pd.DataFrame,
    domain_order: Sequence[str],
) -> pd.DataFrame:
    """Compare each method's 3D connectivity profile with the real 2D reference."""
    rows: List[Dict[str, object]] = []
    labels = [str(x) for x in domain_order]
    reference = real_reference_2d.reindex(index=labels, columns=labels)

    for method in METHOD_ORDER:
        if method not in set(connectivity_df["method"].astype(str)):
            continue
        method_matrix = connectivity_matrix(connectivity_df, method, labels)
        for domain in labels:
            agreement = neighborhood_composition_agreement(
                method_matrix.loc[domain].to_numpy(dtype=float),
                reference.loc[domain].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "method": method,
                    "domain": domain,
                    "neighborhood_agreement": agreement,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# METHOD EVALUATION
# ============================================================
def evaluate_method(
    method: str,
    method_ads: Mapping[int, sc.AnnData],
    real_ads: Mapping[int, sc.AnnData],
    domain_order: Sequence[str],
    xy_scale_to_um: float,
    radius_um: float,
    max_z_diff_um: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], pd.DataFrame]:
    print(f"[RUN] {method}")
    slice_data = build_physical_slice_data(
        real_ads=real_ads,
        method_ads=method_ads,
        z_order_score=Z_ORDER_SCORE,
        xy_scale_to_um=xy_scale_to_um,
        domain_key=DOMAIN_KEY,
        coord_key=COORD_KEY,
    )

    per_spot_df, connectivity_df = compute_neighbor_purity_and_connectivity(
        slice_data=slice_data,
        z_order_score=Z_ORDER_SCORE,
        domain_order=domain_order,
        radius_um=radius_um,
        max_z_diff_um=max_z_diff_um,
        include_same_slice_neighbors=INCLUDE_SAME_SLICE_NEIGHBORS,
        n_neighbors_per_target_slice=N_NEIGHBORS_PER_TARGET_SLICE,
    )

    per_domain_df, overall = summarize_neighbor_continuity(
        per_spot_df=per_spot_df,
        connectivity_df=connectivity_df,
        min_points_per_domain=MIN_POINTS_PER_DOMAIN,
    )

    per_spot_df.insert(0, "method", method)
    per_domain_df.insert(0, "method", method)
    connectivity_df.insert(0, "method", method)
    overall = {"method": method, **overall}
    return per_spot_df, per_domain_df, overall, connectivity_df


def load_inputs_and_metadata():
    real_ads = load_slice_dict(REAL_SLICE_PATHS)
    spatialz_ads = load_slice_dict(SPATIALZ_SLICE_PATHS)
    axis_ads = load_slice_dict(AXIS_SLICE_PATHS)

    if REFERENCE_REAL_Z not in real_ads:
        raise ValueError(
            f"REFERENCE_REAL_Z={REFERENCE_REAL_Z} not found in REAL_SLICE_PATHS"
        )

    nn_current = infer_network_nn_distance(real_ads[REFERENCE_REAL_Z])
    xy_scale_to_um = float(VISIUM_NEAREST_SPOT_DISTANCE_UM / nn_current)
    radius_um, max_z_diff_um, dz_med = compute_radius_and_z_threshold(Z_ORDER_SCORE)
    domain_order = collect_domain_order(
        real_ads,
        spatialz_ads,
        axis_ads,
        domain_key=DOMAIN_KEY,
    )

    metadata = {
        "nn_current": nn_current,
        "xy_scale_to_um": xy_scale_to_um,
        "radius_um": radius_um,
        "max_z_diff_um": max_z_diff_um,
        "dz_med": dz_med,
        "domain_order": domain_order,
    }
    return real_ads, spatialz_ads, axis_ads, metadata


# ============================================================
# EVAL / SAVE
# ============================================================
def save_matrix_as_csv(
    matrix: pd.DataFrame,
    path: str,
    index_name: str = "source_domain",
) -> None:
    output = matrix.copy()
    output.index = output.index.astype(str)
    output.index.name = index_name
    output.reset_index().to_csv(path, index=False)
    print(f"[OK] saved: {path}")


def compute_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real_ads, spatialz_ads, axis_ads, metadata = load_inputs_and_metadata()

    nn_current = float(metadata["nn_current"])
    xy_scale_to_um = float(metadata["xy_scale_to_um"])
    radius_um = float(metadata["radius_um"])
    max_z_diff_um = float(metadata["max_z_diff_um"])
    dz_med = float(metadata["dz_med"])
    domain_order = [str(x) for x in metadata["domain_order"]]

    print(f"[INFO] current network-space NN distance: {nn_current:.4f}")
    print(f"[INFO] XY scale factor to physical µm: {xy_scale_to_um:.6f}")
    print(f"[INFO] scored slices: {Z_ORDER_SCORE}")
    print(f"[INFO] median z step: {dz_med:.2f} µm")
    print(f"[INFO] 3D neighborhood radius: {radius_um:.2f} µm")
    print(f"[INFO] max z difference: {max_z_diff_um:.2f} µm")
    print(f"[INFO] include same-slice neighbors: {INCLUDE_SAME_SLICE_NEIGHBORS}")
    print(f"[INFO] neighbors per eligible target slice: {N_NEIGHBORS_PER_TARGET_SLICE}")
    print(f"[INFO] real 2D reference KNN k: {REFERENCE_2D_KNN_K}")
    print(f"[INFO] domain order: {domain_order}")

    outputs = []
    for method, ads in (("SpatialZ", spatialz_ads), ("AXIS", axis_ads)):
        outputs.append(
            evaluate_method(
                method=method,
                method_ads=ads,
                real_ads=real_ads,
                domain_order=domain_order,
                xy_scale_to_um=xy_scale_to_um,
                radius_um=radius_um,
                max_z_diff_um=max_z_diff_um,
            )
        )

    per_spot_df = pd.concat([item[0] for item in outputs], ignore_index=True)
    per_domain_df = pd.concat([item[1] for item in outputs], ignore_index=True)
    overall_df = pd.DataFrame([item[2] for item in outputs])
    connectivity_df = pd.concat([item[3] for item in outputs], ignore_index=True)

    for dataframe in (overall_df, per_domain_df, connectivity_df):
        dataframe["xy_scale_to_um"] = xy_scale_to_um
        dataframe["radius_um"] = radius_um
        dataframe["max_z_diff_um"] = max_z_diff_um
        dataframe["z_order_score"] = ",".join(map(str, Z_ORDER_SCORE))
        dataframe["include_same_slice_neighbors"] = INCLUDE_SAME_SLICE_NEIGHBORS
        dataframe["n_neighbors_per_target_slice"] = (
            N_NEIGHBORS_PER_TARGET_SLICE
            if N_NEIGHBORS_PER_TARGET_SLICE is not None
            else "all_radius_neighbors"
        )

    # Real 2D neighborhood reference and method-vs-reference agreement.
    real_reference_2d = build_real_2d_reference_matrix(
        real_ads=real_ads,
        domain_order=domain_order,
        k=REFERENCE_2D_KNN_K,
    )
    agreement_df = compute_2d_reference_agreement_table(
        connectivity_df=connectivity_df,
        real_reference_2d=real_reference_2d,
        domain_order=domain_order,
    )
    agreement_df["reference_2d_knn_k"] = int(REFERENCE_2D_KNN_K)

    # Save numerical results.
    overall_df.to_csv(OVERALL_CSV, index=False)
    per_domain_df.to_csv(PER_DOMAIN_CSV, index=False)
    connectivity_df.to_csv(CONNECTIVITY_LONG_CSV, index=False)
    agreement_df.to_csv(REFERENCE_2D_AGREEMENT_CSV, index=False)

    print(f"[OK] saved: {OVERALL_CSV}")
    print(f"[OK] saved: {PER_DOMAIN_CSV}")
    print(f"[OK] saved: {CONNECTIVITY_LONG_CSV}")
    print(f"[OK] saved: {REFERENCE_2D_AGREEMENT_CSV}")

    if SAVE_PER_SPOT:
        per_spot_df.to_csv(PER_SPOT_CSV, index=False)
        print(f"[OK] saved: {PER_SPOT_CSV}")

    # Matrix-form 3D connectivity output.
    matrix_rows = []
    for method in METHOD_ORDER:
        if method not in set(connectivity_df["method"]):
            continue
        matrix = connectivity_matrix(connectivity_df, method, domain_order)
        matrix.insert(0, "source_domain", matrix.index.astype(str))
        matrix.insert(0, "method", method)
        matrix_rows.append(matrix.reset_index(drop=True))
    matrix_df = (
        pd.concat(matrix_rows, ignore_index=True) if matrix_rows else pd.DataFrame()
    )
    matrix_df.to_csv(CONNECTIVITY_MATRIX_CSV, index=False)
    print(f"[OK] saved: {CONNECTIVITY_MATRIX_CSV}")

    save_matrix_as_csv(real_reference_2d, REFERENCE_2D_MATRIX_CSV)

    config = {
        "domain_order": domain_order,
        "coord_key": COORD_KEY,
        "domain_key": DOMAIN_KEY,
        "reference_real_z": REFERENCE_REAL_Z,
        "visium_nearest_spot_distance_um": VISIUM_NEAREST_SPOT_DISTANCE_UM,
        "virtual_depths": [int(x) for x in VIRTUAL_DEPTHS.tolist()],
        "z_order_score": [int(x) for x in Z_ORDER_SCORE],
        "use_real_anchors_in_score": USE_REAL_ANCHORS_IN_SCORE,
        "include_same_slice_neighbors": INCLUDE_SAME_SLICE_NEIGHBORS,
        "n_neighbors_per_target_slice": N_NEIGHBORS_PER_TARGET_SLICE,
        "radius_factor": RADIUS_FACTOR,
        "max_z_diff_factor": MAX_Z_DIFF_FACTOR,
        "radius_um": radius_um,
        "max_z_diff_um": max_z_diff_um,
        "median_z_step_um": dz_med,
        "xy_scale_to_um": xy_scale_to_um,
        "reference_2d_knn_k": REFERENCE_2D_KNN_K,
        "method_palette": METHOD_PALETTE,
        "metric_figure_style": asdict(METRIC_STYLE),
        "heatmap_figure_style": asdict(HEATMAP_STYLE),
    }
    with open(ANALYSIS_CONFIG_JSON, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"[OK] saved: {ANALYSIS_CONFIG_JSON}")

    print("[RESULT] overall 3D continuity:")
    print(
        overall_df[
            [
                "method",
                "overall_continuity_score",
                "overall_neighbor_pair_purity",
                "overall_neighbor_coverage",
            ]
        ]
    )
    print("[RESULT] 2D-reference neighborhood agreement:")
    print(
        agreement_df.groupby("method", as_index=False)["neighborhood_agreement"]
        .mean()
        .sort_values("neighborhood_agreement", ascending=False)
    )

    return overall_df, per_domain_df, agreement_df, connectivity_df


# ============================================================
# LOAD SAVED RESULTS FOR PLOT MODE
# ============================================================
def _normalize_identifier_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for column in (
        "method",
        "domain",
        "source_domain",
        "target_domain",
    ):
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype(str)
    return dataframe


def load_existing_outputs() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, object],
]:
    required = [
        OVERALL_CSV,
        PER_DOMAIN_CSV,
        REFERENCE_2D_AGREEMENT_CSV,
        CONNECTIVITY_LONG_CSV,
        ANALYSIS_CONFIG_JSON,
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Missing saved evaluation outputs. Run --mode eval first:\n  "
            + "\n  ".join(missing)
        )

    overall_df = _normalize_identifier_columns(pd.read_csv(OVERALL_CSV))
    per_domain_df = _normalize_identifier_columns(pd.read_csv(PER_DOMAIN_CSV))
    agreement_df = _normalize_identifier_columns(
        pd.read_csv(REFERENCE_2D_AGREEMENT_CSV)
    )
    connectivity_df = _normalize_identifier_columns(
        pd.read_csv(CONNECTIVITY_LONG_CSV)
    )
    with open(ANALYSIS_CONFIG_JSON, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["domain_order"] = [str(x) for x in config["domain_order"]]

    return overall_df, per_domain_df, agreement_df, connectivity_df, config


# ============================================================
# PLOTTING HELPERS
# ============================================================
def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.8,
            "axes.titleweight": "normal",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def domain_palette(domain_order: Sequence[str]) -> Dict[str, object]:
    """Assign domains sequentially using the first five discrete Tab10 colors."""
    cmap = plt.get_cmap("tab10")

    return {
        str(domain): cmap(i)
        for i, domain in enumerate(domain_order)
    }

def plot_domain_metric_combined(
    metric_df: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    save_path: Optional[str],
    style: MetricFigureStyle = METRIC_STYLE,
) -> None:
    """Shared template for 3D continuity and 2D-reference agreement."""
    set_plot_style()
    if metric_df.empty:
        print(f"[WARN] empty metric table for {title!r}; skip plot")
        return

    method_order = [
        method
        for method in METHOD_ORDER
        if method in set(metric_df["method"].astype(str))
    ]
    if not method_order:
        print(f"[WARN] no configured methods available for {title!r}; skip plot")
        return

    x_map = {method: i for i, method in enumerate(method_order)}
    domain_order = sort_domains(metric_df["domain"].unique())
    palette = domain_palette(domain_order)

    fig, ax = plt.subplots(figsize=style.figsize)
    sns.boxplot(
        data=metric_df,
        x="method",
        y=value_column,
        order=method_order,
        hue="method",
        hue_order=method_order,
        palette=METHOD_PALETTE,
        legend=False,
        showfliers=False,
        width=style.box_width,
        linewidth=style.box_linewidth,
        ax=ax,
        zorder=1,
    )

    rng = np.random.default_rng(0)
    for _, row in metric_df.iterrows():
        method = str(row["method"])
        domain = str(row["domain"])
        value = row[value_column]
        if method not in x_map or pd.isnull(value):
            continue
        ax.scatter(
            x_map[method] + rng.uniform(-style.point_jitter, style.point_jitter),
            value,
            s=style.point_size,
            color=palette[domain],
            edgecolor="black",
            linewidth=style.point_edge_width,
            zorder=3,
        )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=style.label_fontsize)
    ax.set_title(title, fontsize=style.title_fontsize,pad=25)
    from matplotlib.ticker import FormatStrFormatter
    ax.tick_params(axis="both", labelsize=style.tick_fontsize)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.grid(axis="y", alpha=style.grid_alpha)
    if style.ylim is not None:
        ax.set_ylim(*style.ylim)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=palette[domain],
            markeredgecolor="black",
            label=str(domain),
        )
        for domain in domain_order
    ]
    ax.legend(
        handles=handles,
        title="Cortical\n layer",
        bbox_to_anchor=style.legend_bbox,
        loc="upper left",
        frameon=False,
        fontsize=style.legend_fontsize,
        title_fontsize=style.legend_title_fontsize,
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================
# CONNECTIVITY HEATMAPS
# ============================================================
def make_vlag_white_to_red_cmap():
    base = sns.color_palette("vlag", as_cmap=True)
    return mpl.colors.LinearSegmentedColormap.from_list(
        "vlag_white_to_red",
        base(np.linspace(0.5, 1.0, 256)),
    )


def _heatmap_norm(matrix: pd.DataFrame, gamma: float) -> mpl.colors.Normalize:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(finite)) if finite.size else 1.0
    vmax = max(vmax, 1e-8)
    return mpl.colors.PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)


def plot_neighborhood_connectivity_heatmaps(
    connectivity_df: pd.DataFrame,
    domain_order: Sequence[str],
    save_paths: Mapping[str, str],
    style: HeatmapFigureStyle = HEATMAP_STYLE,
) -> None:
    """Plot SpatialZ and AXIS connectivity heatmaps as separate figures.

    Both figures use the white-to-red half of ``vlag`` and each has its own
    colorbar. When ``share_color_scale`` is True, the two independently saved
    figures use the same numerical normalization, preserving direct visual
    comparability while keeping separate colorbar legends.
    """
    sns.set_theme(style="white", context="paper")
    if connectivity_df.empty:
        print("[WARN] empty connectivity_df; skip heatmaps")
        return

    methods = [
        method
        for method in METHOD_ORDER
        if method in set(connectivity_df["method"].astype(str))
    ]
    if not methods:
        print("[WARN] no valid methods in connectivity_df; skip heatmaps")
        return

    matrices = {
        method: connectivity_matrix(connectivity_df, method, domain_order)
        for method in methods
    }

    # Keep method figures quantitatively comparable by default. Each figure
    # still receives its own colorbar axis/legend.
    if style.share_color_scale:
        finite_blocks = []
        for matrix in matrices.values():
            values = matrix.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_blocks.append(finite)
        vmax = (
            float(np.max(np.concatenate(finite_blocks)))
            if finite_blocks
            else 1.0
        )
        vmax = max(vmax, 1e-8)
        shared_norm = mpl.colors.PowerNorm(
            gamma=style.power_gamma,
            vmin=0.0,
            vmax=vmax,
        )
        norms = {method: shared_norm for method in methods}
    else:
        norms = {
            method: _heatmap_norm(matrix, style.power_gamma)
            for method, matrix in matrices.items()
        }

    cmap = make_vlag_white_to_red_cmap()

    for method in methods:
        matrix = matrices[method]
        norm = norms[method]
        matrix_values = matrix.to_numpy(dtype=float)

        figure = plt.figure(
            figsize=(style.figure_width, style.row_height)
        )
        grid = figure.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=list(style.heatmap_to_cbar_width_ratio),
            wspace=style.wspace,
        )
        axis = figure.add_subplot(grid[0, 0])
        cbar_axis = figure.add_subplot(grid[0, 1])

        heatmap = sns.heatmap(
            matrix,
            ax=axis,
            cmap=cmap,
            norm=norm,
            annot=True,
            fmt=".2f",
            annot_kws={
                "fontsize": style.annotation_fontsize,
                "fontweight": "normal",
            },
            square=True,
            linewidths=style.linewidths,
            linecolor=style.linecolor,
            cbar=True,
            cbar_ax=cbar_axis,
            cbar_kws={"label": "Fraction of neighbor layers"},
        )

        nrow, ncol = matrix_values.shape
        threshold = style.annotation_white_threshold_fraction * float(norm.vmax)
        for text_index, text in enumerate(heatmap.texts):
            r = text_index // ncol
            c = text_index % ncol
            value = (
                float(matrix_values[r, c])
                if np.isfinite(matrix_values[r, c])
                else 0.0
            )
            text.set_color("white" if value >= threshold else "#22313f")

        # No figure-level suptitle: each standalone heatmap is identified only
        # by its method title, matching the compact mouse-brain heatmap style.
        axis.set_title(
            f"{method} | Layer connectivity",
            fontsize=style.title_fontsize,
            pad=12,
        )
        axis.set_xlabel(
            "Neighbor layer",
            fontsize=style.axis_label_fontsize,
        )
        axis.set_ylabel(
            "Source layer",
            fontsize=style.axis_label_fontsize,
        )
        axis.set_xticklabels(
            axis.get_xticklabels(),
            rotation=35,
            ha="right",
            rotation_mode="anchor",
            fontsize=style.tick_fontsize,
        )
        axis.set_yticklabels(
            axis.get_yticklabels(),
            rotation=0,
            fontsize=style.tick_fontsize,
        )
        axis.tick_params(axis="both", length=0)
        axis.set_facecolor("#fcfcfc")
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("0.50")

        cbar_axis.tick_params(labelsize=style.cbar_tick_fontsize)
        cbar_axis.set_ylabel(
            "Fraction of neighbor layers",
            fontsize=style.cbar_label_fontsize,
            labelpad=10,
        )

        save_path = save_paths.get(method)
        if save_path:
            figure.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
            print(f"[OK] saved: {save_path}")
        else:
            plt.show()
        plt.close(figure)


# ============================================================
# PLOT WORKFLOW
# ============================================================
def plot_all(
    per_domain_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    connectivity_df: pd.DataFrame,
    config: Mapping[str, object],
) -> None:
    domain_order = [str(x) for x in config["domain_order"]]

    plot_domain_metric_combined(
        metric_df=per_domain_df,
        value_column="continuity_score",
        title="3D laminar continuity",
        ylabel="Laminar continuity score",
        save_path=COMBINED_CONTINUITY_FIG,
    )

    plot_domain_metric_combined(
        metric_df=agreement_df,
        value_column="neighborhood_agreement",
        title="Layer-neighborhood agreement",
        ylabel="Agreement with measured sections",
        save_path=REFERENCE_2D_AGREEMENT_FIG,
    )

    plot_neighborhood_connectivity_heatmaps(
        connectivity_df=connectivity_df,
        domain_order=domain_order,
        save_paths={
            "SpatialZ": CONNECTIVITY_HEATMAP_SPATIALZ_FIG,
            "AXIS": CONNECTIVITY_HEATMAP_AXIS_FIG,
        },
    )


# ============================================================
# ENTRY POINT
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DLPFC 3D domain and neighborhood evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=("eval", "plot", "both"),
        default="plot",
        help=(
            "eval: compute and save numerical results only; "
            "plot: load saved results and generate figures only; "
            "both: evaluate then plot"
        ),
    )
    args = parser.parse_args()

    if args.mode in {"eval", "both"}:
        overall_df, per_domain_df, agreement_df, connectivity_df = compute_all()
        with open(ANALYSIS_CONFIG_JSON, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        (
            overall_df,
            per_domain_df,
            agreement_df,
            connectivity_df,
            config,
        ) = load_existing_outputs()

    if args.mode in {"plot", "both"}:
        plot_all(
            per_domain_df=per_domain_df,
            agreement_df=agreement_df,
            connectivity_df=connectivity_df,
            config=config,
        )


if __name__ == "__main__":
    main()
