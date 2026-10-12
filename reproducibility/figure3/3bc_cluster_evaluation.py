#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SCC cluster-fidelity evaluation and publication plotting.

This streamlined version keeps only the analyses requested for the SCC result:

1. Per-cluster composition error.
2. Per-cluster local-structure distance.
3. Per-cluster neighborhood-connectivity distance.
4. Standalone spatial cluster maps for Real, SpatialZ, and AXIS.

Modes
-----
eval
    Compute the three per-cluster metrics for patients P2 and P10 and save them
    to CSV. No figures are generated.

plot
    Read the saved CSV and generate six standalone metric figures (three per
    patient), plus one standalone spatial map per patient-method case.
    Evaluation metrics are not recomputed.

both
    Run evaluation first, then plotting.
"""

from __future__ import annotations
from axis.settings import data_path
from axis.config import load_config
from matplotlib.ticker import FormatStrFormatter
import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Dataset / output configuration
# =============================================================================
ROOT = load_config('scc').root
H5AD_DIR = os.path.join(ROOT, "h5ad")
OUTDIR = data_path("paper_analysis/Figure3_scc/bc/")

GROUP_IDS = (2, 10)
METHOD_ORDER = ("AXIS","SpatialZ")
PLOT_METHOD_ORDER = ("Real", "AXIS","SpatialZ")

CLUSTER_KEY = "original_clusters"
COORD_KEY = "spatial"
KNN_CLUSTER_NEIGHBOR = 6
EPS = 1e-8

PER_CLUSTER_CSV = os.path.join(OUTDIR, "per_cluster_fidelity_metrics.csv")

METHOD_PALETTE = {
    "SpatialZ": "#7C8DA6",
    "AXIS": "#C4473A",
}


# =============================================================================
# Plot-layout configuration
# =============================================================================
@dataclass(frozen=True)
class MetricPlotStyle:
    """Controls all standalone per-patient metric figures."""

    figsize: Tuple[float, float] = (4.8, 2.7)
    dpi: int = 350

    bar_width: float = 0.36
    bar_alpha: float = 0.94

    title_fontsize: float = 14.0
    title_fontweight: str = None
    title_pad: float = 8.0

    ylabel_fontsize: float = 12
    x_tick_fontsize: float = 12.0
    y_tick_fontsize: float = 12.0
    x_tick_rotation: float = 0.0

    legend_loc: str = "upper left"
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None
    legend_ncol: int = 1
    legend_fontsize: float = 12.0
    legend_markerscale: float = 1.0
    legend_handletextpad: float = 0.45
    legend_labelspacing: float = 0.35
    legend_columnspacing: float = 0.8

    grid_alpha: float = 0.18
    left: float = 0.14
    right: float = 0.98
    bottom: float = 0.16
    top: float = 0.90


@dataclass(frozen=True)
class SpatialMapStyle:
    """Controls the Real/SpatialZ/AXIS categorical spatial-map comparison."""

    figsize: Tuple[float, float] = (11.2, 3.8)
    dpi: int = 400

    point_size: float = 40.0
    point_alpha: float = 0.90
    invert_y: bool = True
    share_limits: bool = True
    pad_ratio: float = 0.03

    method_title_fontsize: float = 14.0
    method_title_fontweight: str = "bold"
    method_title_pad: float = 6.0

    suptitle_fontsize: float = 20.0
    suptitle_fontweight: str = None
    suptitle_y: float = 0.985
    suptitle_x: float = 0.45

    # Legend control.
    show_legend: bool = True
    legend_title: str = "    Cell\npopulation"
    legend_loc: str = "center left"
    legend_bbox_to_anchor: Tuple[float, float] = (1.01, 0.50)
    legend_ncol: int = 1
    legend_fontsize: float = 14.0
    legend_title_fontsize: float = 14
    legend_marker_size: float = 6.0
    legend_markerscale: float = 1.0
    legend_handletextpad: float = 0.35
    legend_labelspacing: float = 0.10
    legend_columnspacing: float = 0.5

    # Figure margins. ``right`` reserves space for the legend.
    left: float = 0.02
    right: float = 0.84
    bottom: float = 0.04
    top: float = 0.88
    wspace: float = 0.04


METRIC_STYLE = MetricPlotStyle()
SPATIAL_STYLE = SpatialMapStyle()


# =============================================================================
# Generic helpers
# =============================================================================
def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def canonical_cluster_order(labels: Sequence) -> List[str]:
    values = pd.Series(labels).astype(str)
    unique = list(pd.unique(values))

    def key_fn(value: str):
        try:
            return (0, float(value))
        except Exception:
            return (1, str(value))

    return sorted([str(x) for x in unique], key=key_fn)


def load_ad(path: str) -> AnnData:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    adata = sc.read_h5ad(path)
    if "valid" in adata.obs.columns:
        adata = adata[adata.obs["valid"]].copy()
    return adata


def load_group_data(group_id: int) -> Dict[str, AnnData]:
    """Load Real, SpatialZ, and AXIS for one SCC patient."""
    real_path = os.path.join(H5AD_DIR, f"{group_id}_1.h5ad")
    axis_path = (
        data_path(f"SCC/huifang/virtual_tissue/"
        f"scc_z_group{group_id}_grid40_spatial3_gene_weight/query/"
        f"virtual_scc_z_10.000.h5ad")
    )
    spatialz_path = (
        data_path(f"SCC/huifang/virtual_tissue/"
        f"spatialz/group{group_id}/patient{group_id}_generation_10.h5ad")
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
) -> Tuple[np.ndarray, np.ndarray]:
    if coord_key not in adata.obsm:
        raise KeyError(f"{coord_key!r} not found in adata.obsm")

    coords = np.asarray(adata.obsm[coord_key], dtype=np.float64)
    labels = get_cluster_labels(adata, cluster_key)
    finite = np.all(np.isfinite(coords[:, :2]), axis=1)
    return coords[finite, :2], labels[finite].astype(str)


def union_cluster_order(method_adatas: Mapping[str, AnnData]) -> List[str]:
    labels: List[str] = []
    for adata in method_adatas.values():
        labels.extend(get_cluster_labels(adata).astype(str).tolist())
    return canonical_cluster_order(labels)


# =============================================================================
# Metric 1: cluster composition error
# =============================================================================
def cluster_composition(adata: AnnData, cluster_order: Sequence[str]) -> np.ndarray:
    labels = get_cluster_labels(adata)
    counts = pd.Series(labels).astype(str).value_counts()
    values = np.asarray([counts.get(str(cluster), 0) for cluster in cluster_order], dtype=float)
    if values.sum() > 0:
        values /= values.sum()
    return values


def symmetric_relative_abs_error(
    reference: np.ndarray,
    candidate: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    return 2.0 * np.abs(candidate - reference) / (
        np.abs(reference) + np.abs(candidate) + eps
    )


# =============================================================================
# Metric 2: within-cluster local structure
# =============================================================================
def slice_min_positive_spot_distance(
    adata: AnnData,
    coord_key: str = COORD_KEY,
    eps: float = EPS,
) -> float:
    coords, _ = get_valid_coords_labels(adata, coord_key=coord_key)
    if len(coords) < 2:
        return np.nan

    tree = cKDTree(coords)
    try:
        dist, _ = tree.query(coords, k=2, workers=-1)
    except TypeError:
        dist, _ = tree.query(coords, k=2)

    nearest = np.asarray(dist[:, 1], dtype=float)
    nearest = nearest[np.isfinite(nearest) & (nearest > eps)]
    return float(np.min(nearest)) if nearest.size else np.nan


def within_cluster_nn_distances_spot_units(
    adata: AnnData,
    cluster: str,
    normalization_spot_unit: Optional[float],
) -> np.ndarray:
    coords, labels = get_valid_coords_labels(adata)
    coords_cluster = coords[labels == str(cluster)]
    if len(coords_cluster) < 2:
        return np.asarray([], dtype=float)

    tree = cKDTree(coords_cluster)
    try:
        dist, _ = tree.query(coords_cluster, k=2, workers=-1)
    except TypeError:
        dist, _ = tree.query(coords_cluster, k=2)

    nearest = np.asarray(dist[:, 1], dtype=float)
    nearest = nearest[np.isfinite(nearest)]
    if (
        normalization_spot_unit is None
        or not np.isfinite(normalization_spot_unit)
        or normalization_spot_unit <= 0
    ):
        return np.full_like(nearest, np.nan, dtype=float)
    return nearest / float(normalization_spot_unit)


def local_structure_distance_to_real(
    real_distances: np.ndarray,
    candidate_distances: np.ndarray,
) -> float:
    real = np.asarray(real_distances, dtype=float)
    candidate = np.asarray(candidate_distances, dtype=float)
    real = real[np.isfinite(real)]
    candidate = candidate[np.isfinite(candidate)]
    if real.size == 0 or candidate.size == 0:
        return np.nan
    return float(wasserstein_distance(real, candidate))


# =============================================================================
# Metric 3: cluster-neighborhood connectivity
# =============================================================================
def knn_indices(coords: np.ndarray, k: int = KNN_CLUSTER_NEIGHBOR) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2:
        return np.zeros((n, 0), dtype=np.int64)

    kk = min(k + 1, n)
    tree = cKDTree(coords)
    try:
        _, indices = tree.query(coords, k=kk, workers=-1)
    except TypeError:
        _, indices = tree.query(coords, k=kk)

    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim == 1:
        indices = indices[:, None]
    return indices[:, 1:]


def cluster_connectivity_matrix(
    adata: AnnData,
    cluster_order: Sequence[str],
    knn: int = KNN_CLUSTER_NEIGHBOR,
) -> np.ndarray:
    """Row-normalized source-cluster -> neighboring-cluster fractions."""
    coords, labels = get_valid_coords_labels(adata)
    neighbor_indices = knn_indices(coords, k=knn)
    matrix = np.full((len(cluster_order), len(cluster_order)), np.nan, dtype=float)

    for row, source_cluster in enumerate(cluster_order):
        source_mask = labels == str(source_cluster)
        if not np.any(source_mask) or neighbor_indices.shape[1] == 0:
            continue

        neighbor_labels = labels[neighbor_indices[source_mask].ravel()]
        denominator = float(len(neighbor_labels))
        if denominator <= 0:
            continue

        counts = pd.Series(neighbor_labels).astype(str).value_counts()
        for col, target_cluster in enumerate(cluster_order):
            matrix[row, col] = float(counts.get(str(target_cluster), 0)) / denominator

    return matrix


def connectivity_js_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Jensen-Shannon distance normalized to [0, 1]."""
    p = np.asarray(reference, dtype=float).ravel()
    q = np.asarray(candidate, dtype=float).ravel()
    keep = np.isfinite(p) & np.isfinite(q)
    if not np.any(keep):
        return np.nan

    p = np.clip(p[keep], 0.0, np.inf)
    q = np.clip(q[keep], 0.0, np.inf)
    if p.sum() <= 0 or q.sum() <= 0:
        return np.nan

    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        valid = (a > 0) & (b > 0)
        if not np.any(valid):
            return 0.0
        return float(np.sum(a[valid] * np.log(a[valid] / b[valid])))

    js_divergence = 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)
    distance = np.sqrt(max(js_divergence, 0.0) / np.log(2.0))
    return float(np.clip(distance, 0.0, 1.0))


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_group(group_id: int) -> pd.DataFrame:
    data = load_group_data(group_id)
    cluster_order = union_cluster_order(data)

    composition = {
        method: cluster_composition(adata, cluster_order)
        for method, adata in data.items()
    }
    real_composition = composition["Real"]

    # Preserve the original normalization convention: Real uses the Real spot
    # unit; SpatialZ and AXIS both use the AXIS layout's spot unit.
    real_unit = slice_min_positive_spot_distance(data["Real"])
    axis_unit = slice_min_positive_spot_distance(data["AXIS"])
    normalization_unit = {
        "Real": real_unit,
        "SpatialZ": axis_unit,
        "AXIS": axis_unit,
    }

    local_distances: Dict[Tuple[str, str], np.ndarray] = {}
    for method, adata in data.items():
        for cluster in cluster_order:
            local_distances[(method, str(cluster))] = within_cluster_nn_distances_spot_units(
                adata,
                cluster=str(cluster),
                normalization_spot_unit=normalization_unit[method],
            )

    connectivity = {
        method: cluster_connectivity_matrix(adata, cluster_order)
        for method, adata in data.items()
    }

    rows = []
    for candidate in METHOD_ORDER:
        composition_error = symmetric_relative_abs_error(
            real_composition,
            composition[candidate],
        )

        for index, cluster in enumerate(cluster_order):
            cluster = str(cluster)
            rows.append(
                {
                    "group_id": int(group_id),
                    "candidate": candidate,
                    "cluster": cluster,
                    "cluster_comp_sare": float(composition_error[index]),
                    "within_cluster_nn_wasserstein": local_structure_distance_to_real(
                        local_distances[("Real", cluster)],
                        local_distances[(candidate, cluster)],
                    ),
                    "connectivity_profile_js_distance": connectivity_js_distance(
                        connectivity["Real"][index],
                        connectivity[candidate][index],
                    ),
                }
            )

    return pd.DataFrame(rows)


def evaluate_and_save(
    group_ids: Sequence[int] = GROUP_IDS,
    outdir: str = OUTDIR,
) -> pd.DataFrame:
    ensure_dir(outdir)
    frames = []
    for group_id in group_ids:
        print(f"[INFO] evaluating P{group_id}")
        frames.append(evaluate_group(int(group_id)))

    result = pd.concat(frames, axis=0, ignore_index=True)
    result.to_csv(PER_CLUSTER_CSV, index=False)
    print(f"[OK] saved: {PER_CLUSTER_CSV}")
    return result


# =============================================================================
# Standalone metric figures
# =============================================================================
METRIC_SPECS = {
    "composition_error": {
        "column": "cluster_comp_sare",
        "title": "Cell-population composition error \N{DOWNWARDS ARROW}",
        "ylabel": "Composition error (sARE)",
        "ylim": None,
    },
    "cluster_local_structure": {
        "column": "within_cluster_nn_wasserstein",
        "title": "Within-population spatial organization error \N{DOWNWARDS ARROW}",
        "ylabel": "Wasserstein distance",
        "ylim": None,
    },
    "neighborhood_connectivity": {
        "column": "connectivity_profile_js_distance",
        "title": "Neighborhood connectivity error\N{DOWNWARDS ARROW}",
        "ylabel": "Jensen-Shannon distance",
        "ylim": (0.0, 1.02),
    },
}


def _add_metric_legend(ax, style: MetricPlotStyle) -> None:
    kwargs = dict(
        loc=style.legend_loc,
        ncol=style.legend_ncol,
        frameon=False,
        fontsize=style.legend_fontsize,
        markerscale=style.legend_markerscale,
        handletextpad=style.legend_handletextpad,
        labelspacing=style.legend_labelspacing,
        columnspacing=style.legend_columnspacing,
    )
    if style.legend_bbox_to_anchor is not None:
        kwargs["bbox_to_anchor"] = style.legend_bbox_to_anchor
    ax.legend(**kwargs)


def plot_patient_metric(
    per_cluster_df: pd.DataFrame,
    group_id: int,
    metric_key: str,
    save_path: Optional[str] = None,
    style: MetricPlotStyle = METRIC_STYLE,
    show: bool = False,
):
    if metric_key not in METRIC_SPECS:
        raise KeyError(f"Unknown metric_key={metric_key!r}")

    set_publication_style()
    spec = METRIC_SPECS[metric_key]
    metric = spec["column"]

    sub = per_cluster_df[per_cluster_df["group_id"] == int(group_id)].copy()
    if sub.empty:
        raise ValueError(f"No saved per-cluster metrics for P{group_id}")

    cluster_order = canonical_cluster_order(sub["cluster"].astype(str).values)
    methods = [m for m in METHOD_ORDER if m in set(sub["candidate"])]
    x = np.arange(len(cluster_order), dtype=float)

    fig, ax = plt.subplots(figsize=style.figsize)
    n_methods = max(len(methods), 1)
    width = style.bar_width
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width

    for method_index, method in enumerate(methods):
        values = (
            sub[sub["candidate"] == method][["cluster", metric]]
            .drop_duplicates("cluster")
            .set_index("cluster")
            .reindex(cluster_order)[metric]
            .astype(float)
            .values
        )
        ax.bar(
            x + offsets[method_index],
            values,
            width=width,
            color=METHOD_PALETTE[method],
            alpha=style.bar_alpha,
            label=method,
            linewidth=0,
        )

    ax.set_title(
        f"{spec['title']}",
        fontsize=style.title_fontsize,
        fontweight="normal",
        pad=style.title_pad,
    )
    ax.set_ylabel(spec["ylabel"], fontsize=style.ylabel_fontsize)
    ax.set_xlabel("Cell population", fontsize=style.ylabel_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(
        cluster_order,
        fontsize=style.x_tick_fontsize,
        rotation=style.x_tick_rotation,
    )
    ax.tick_params(axis="y", labelsize=style.y_tick_fontsize)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(axis="y", alpha=style.grid_alpha, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    if spec["ylim"] is not None:
        ax.set_ylim(*spec["ylim"])

    _add_metric_legend(ax, style)
    fig.subplots_adjust(
        left=style.left,
        right=style.right,
        bottom=style.bottom,
        top=style.top,
    )

    if save_path is not None:
        ensure_dir(os.path.dirname(os.path.abspath(save_path)))
        fig.savefig(save_path, dpi=style.dpi, bbox_inches="tight", facecolor="white")
        print(f"[OK] saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_all_patient_metrics(
    per_cluster_df: pd.DataFrame,
    group_ids: Sequence[int] = GROUP_IDS,
    outdir: str = OUTDIR,
    style: MetricPlotStyle = METRIC_STYLE,
    show: bool = False,
) -> None:
    ensure_dir(outdir)
    for group_id in group_ids:
        for metric_key in METRIC_SPECS:
            path = os.path.join(outdir, f"P{group_id}_{metric_key}.png")
            plot_patient_metric(
                per_cluster_df=per_cluster_df,
                group_id=int(group_id),
                metric_key=metric_key,
                save_path=path,
                style=style,
                show=show,
            )


# =============================================================================
# Spatial cluster-map comparison
# =============================================================================
def tab10_cluster_color_dict(cluster_order: Sequence[str]) -> Dict[str, object]:
    """Discrete tab10 mapping; numeric cluster IDs keep stable tab10 indices."""
    cmap = plt.get_cmap("tab20")
    color_map: Dict[str, object] = {}
    for fallback_index, cluster in enumerate(cluster_order):
        cluster = str(cluster)
        try:
            color_index = int(float(cluster)) % cmap.N
        except Exception:
            color_index = fallback_index % cmap.N
        color_map[cluster] = cmap(color_index)
    return color_map


def spatial_limits(
    coordinate_arrays: Sequence[np.ndarray],
    pad_ratio: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    finite_arrays = []
    for coords in coordinate_arrays:
        coords = np.asarray(coords, dtype=float)
        keep = np.all(np.isfinite(coords[:, :2]), axis=1)
        if np.any(keep):
            finite_arrays.append(coords[keep, :2])

    if not finite_arrays:
        raise ValueError("No finite coordinates available for spatial plotting")

    merged = np.vstack(finite_arrays)
    low = merged.min(axis=0)
    high = merged.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    pad = span * pad_ratio
    return (
        (float(low[0] - pad[0]), float(high[0] + pad[0])),
        (float(low[1] - pad[1]), float(high[1] + pad[1])),
    )


def _style_spatial_axis(
    ax,
    method: str,
    style: SpatialMapStyle,
) -> None:
    # ax.set_title(
    #     method,
    #     fontsize=style.method_title_fontsize,
    #     # fontweight=style.method_title_fontweight,
    #     pad=style.method_title_pad,
    # )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_spatial_legend(
    ax,
    cluster_order: Sequence[str],
    color_map: Mapping[str, object],
    style: SpatialMapStyle,
) -> None:
    if not style.show_legend:
        return

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color_map[str(cluster)],
            markeredgecolor="none",
            markersize=style.legend_marker_size,
            label=str(cluster),
        )
        for cluster in cluster_order
    ]

    legend = ax.legend(
        handles=handles,
        loc=style.legend_loc,
        bbox_to_anchor=style.legend_bbox_to_anchor,
        ncol=style.legend_ncol,
        frameon=False,
        title=style.legend_title,
        fontsize=style.legend_fontsize,
        markerscale=style.legend_markerscale,
        handletextpad=style.legend_handletextpad,
        labelspacing=style.legend_labelspacing,
        columnspacing=style.legend_columnspacing,
        borderaxespad=0.0,
    )
    legend.get_title().set_fontsize(style.legend_title_fontsize)


def _save_or_show(fig, save_path: Optional[str], show: bool) -> None:
    if save_path is not None:
        ensure_dir(os.path.dirname(os.path.abspath(save_path)))
        fig.savefig(save_path, dpi=SPATIAL_STYLE.dpi, bbox_inches="tight", facecolor="white")
        print(f"[OK] saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_single_patient_spatial_map(
    group_id: int,
    method: str,
    save_path: Optional[str] = None,
    style: SpatialMapStyle = SPATIAL_STYLE,
    show: bool = False,
):
    """Save one standalone spatial map per patient-method case."""
    set_publication_style()
    data = load_group_data(group_id)
    if method not in data:
        raise KeyError(f"Unknown method {method!r}. Available: {list(data)}")

    cluster_order = union_cluster_order(data)
    color_map = tab10_cluster_color_dict(cluster_order)
    coords, labels = get_valid_coords_labels(data[method])

    fig, ax = plt.subplots(1, 1, figsize=style.figsize)

    for cluster in cluster_order:
        keep = labels == str(cluster)
        if not np.any(keep):
            continue
        ax.scatter(
            coords[keep, 0],
            coords[keep, 1],
            s=style.point_size,
            color=[color_map[str(cluster)]],
            alpha=style.point_alpha,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )

    _style_spatial_axis(ax, method, style)

    xlim, ylim = spatial_limits([coords], pad_ratio=style.pad_ratio)
    ax.set_xlim(*xlim)
    if style.invert_y:
        ax.set_ylim(ylim[1], ylim[0])
    else:
        ax.set_ylim(*ylim)


    if method =="Real":
        fig.suptitle(
            "Measured (held-out)",
            fontsize=style.suptitle_fontsize,
            fontweight=style.suptitle_fontweight,
            y=style.suptitle_y,
            x=style.suptitle_x,
        )
    else:
        fig.suptitle(
            f"{method}",
            fontsize=style.suptitle_fontsize,
            fontweight=style.suptitle_fontweight,
            y=style.suptitle_y,
            x=style.suptitle_x,
        )
        _add_spatial_legend(ax, cluster_order, color_map, style)
    fig.subplots_adjust(
        left=style.left,
        right=style.right,
        bottom=style.bottom,
        top=style.top,
    )

    if save_path is not None:
        ensure_dir(os.path.dirname(os.path.abspath(save_path)))
        fig.savefig(save_path, dpi=style.dpi, bbox_inches="tight", facecolor="white")
        print(f"[OK] saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_all_spatial_maps(
    group_ids: Sequence[int] = GROUP_IDS,
    outdir: str = OUTDIR,
    style: SpatialMapStyle = SPATIAL_STYLE,
    show: bool = False,
) -> None:
    ensure_dir(outdir)
    for group_id in group_ids:
        for method in PLOT_METHOD_ORDER:
            filename = f"P{group_id}_{method.lower()}_spatial_map.png"
            plot_single_patient_spatial_map(
                group_id=int(group_id),
                method=method,
                save_path=os.path.join(outdir, filename),
                style=style,
                show=show,
            )


# =============================================================================
# Plot mode
# =============================================================================
def require_saved_evaluation() -> None:
    if not os.path.isfile(PER_CLUSTER_CSV):
        raise FileNotFoundError(
            f"Missing saved evaluation: {PER_CLUSTER_CSV}\n"
            "Run with --mode eval or --mode both first."
        )


def plot_saved_results(show: bool = False) -> None:
    require_saved_evaluation()
    ensure_dir(OUTDIR)
    per_cluster_df = pd.read_csv(PER_CLUSTER_CSV, dtype={"cluster": str})

    required = {
        "group_id",
        "candidate",
        "cluster",
        "cluster_comp_sare",
        "within_cluster_nn_wasserstein",
        "connectivity_profile_js_distance",
    }
    missing = sorted(required - set(per_cluster_df.columns))
    if missing:
        raise ValueError(f"Saved metrics CSV is missing columns: {missing}")

    plot_all_patient_metrics(
        per_cluster_df=per_cluster_df,
        group_ids=GROUP_IDS,
        outdir=OUTDIR,
        style=METRIC_STYLE,
        show=show,
    )
    plot_all_spatial_maps(
        group_ids=GROUP_IDS,
        outdir=OUTDIR,
        style=SPATIAL_STYLE,
        show=show,
    )


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="SCC cluster-fidelity evaluation with separated eval/plot workflows."
    )
    parser.add_argument(
        "--mode",
        choices=("eval", "plot", "both"),
        default="plot",
        help=(
            "eval: compute/save metrics only; "
            "plot: generate figures from saved metrics; "
            "both: evaluate then plot"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUTDIR)

    if args.mode in {"eval", "both"}:
        evaluate_and_save(group_ids=GROUP_IDS, outdir=OUTDIR)

    if args.mode in {"plot", "both"}:
        plot_saved_results(show=args.show)

    print("[DONE] SCC cluster-fidelity workflow finished.")


if __name__ == "__main__":
    main()
