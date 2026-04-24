#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overlay AXIS joint domains across virtual time for analysis/query slices.

Folder structure
----------------
root/
    analysis/
        *_<time>.h5ad
    query/
        *_<time>.h5ad

Requirements
------------
pip install anndata scipy numpy pandas matplotlib

What it does
------------
1. Read paired .h5ad files from analysis/ and query/
2. Match pairs by time parsed from filename suffix *_<time>.h5ad
3. Read obs['axis_joint_domain'] and coordinates from obsm[coord_key]
4. Rasterize categorical labels onto a visualization grid using nearest-neighbor
   assignment with a distance support mask to avoid filling far-away background
5. For each domain ID, generate a sequential overlay figure across time:
      - analysis domain: blue fill + dashed boundary
      - query domain: orange fill + solid boundary
6. Optionally save a CSV with rasterized area fractions per domain and time
"""

from __future__ import annotations

import re

from pathlib import Path
from typing import Dict, List, Tuple, Any

import scipy.sparse as sp
from scipy.spatial import cKDTree
import anndata as ad

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _sort_domain_labels(labels):
    labels = [str(x) for x in pd.unique(pd.Series(labels).astype(str))]

    def _key(x):
        try:
            return (0, int(x))
        except Exception:
            try:
                return (1, float(x))
            except Exception:
                return (2, x)

    return sorted(labels, key=_key)


def _domain_mask(label_grid: np.ndarray, valid_mask: np.ndarray, domain_id: str) -> np.ndarray:
    return (label_grid == str(domain_id)) & valid_mask


def _area_fraction(mask: np.ndarray, valid_mask: np.ndarray) -> float:
    denom = int(np.sum(valid_mask))
    if denom == 0:
        return np.nan
    return float(np.sum(mask) / denom)


def build_domain_fraction_summary_from_pairs_rasterized(
    pairs_rasterized: List[Dict[str, Any]],
    domain_ids: List[str] | None = None,
) -> pd.DataFrame:
    """
    Build a long-format summary DataFrame from precomputed rasterized pairs.

    Expected structure of each item in pairs_rasterized:
    {
        "time": float,
        "analysis": {
            "label_grid": np.ndarray of str labels,
            "valid_mask": np.ndarray of bool
        },
        "query": {
            "label_grid": np.ndarray of str labels,
            "valid_mask": np.ndarray of bool
        }
    }
    """
    if len(pairs_rasterized) == 0:
        raise ValueError("pairs_rasterized is empty.")

    if domain_ids is None:
        collected = []
        for item in pairs_rasterized:
            a = item["analysis"]["label_grid"][item["analysis"]["valid_mask"]]
            q = item["query"]["label_grid"][item["query"]["valid_mask"]]
            collected.extend(a.astype(str).tolist())
            collected.extend(q.astype(str).tolist())
        domain_ids = _sort_domain_labels(collected)
    else:
        domain_ids = [str(x) for x in domain_ids]

    rows = []
    for item in pairs_rasterized:
        t = float(item["time"])

        a_grid = item["analysis"]["label_grid"]
        a_valid = item["analysis"]["valid_mask"]

        q_grid = item["query"]["label_grid"]
        q_valid = item["query"]["valid_mask"]

        for d in domain_ids:
            a_mask = _domain_mask(a_grid, a_valid, d)
            q_mask = _domain_mask(q_grid, q_valid, d)

            rows.append(
                {
                    "time": t,
                    "domain_id": str(d),
                    "analysis_area_fraction": _area_fraction(a_mask, a_valid),
                    "query_area_fraction": _area_fraction(q_mask, q_valid),
                }
            )

    df = pd.DataFrame(rows).sort_values(["time", "domain_id"]).reset_index(drop=True)
    return df


def _build_domain_colors(domain_ids: List[str]) -> Dict[str, tuple]:
    """
    Consistent categorical color assignment.
    """
    n = len(domain_ids)
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(n)]
    else:
        cmap = plt.get_cmap("gist_ncar")
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    return {d: c for d, c in zip(domain_ids, colors)}


def plot_domain_variation_fancy(
    summary_df: pd.DataFrame,
    domain_ids,
    title: str = "Domain variation across virtual time",
    figsize: tuple = (15, 12),
    top_n_line_domains: int = 8,
    domain_order: str = "variation",
    save_path: str | None = None,
    dpi: int = 220,
):
    """
    Generate a 3-panel summary figure:

    Panel A: stacked area of query-space domain fractions over time
    Panel B: per-domain trajectories (solid=query, dashed=analysis)
    Panel C: heatmap of query-analysis area-fraction difference

    Parameters
    ----------
    summary_df
        Long-format dataframe with columns:
            - time
            - domain_id
            - analysis_area_fraction
            - query_area_fraction
    title
        Figure title
    figsize
        Figure size
    top_n_line_domains
        Number of most dynamic domains to show in Panel B
    domain_order
        'variation', 'mean_query', or 'label'
    save_path
        Optional path to save the figure
    dpi
        Figure dpi
    """
    required = {
        "time",
        "domain_id",
        "analysis_area_fraction",
        "query_area_fraction",
    }
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary_df is missing required columns: {missing}")

    df = summary_df.copy()
    df["domain_id"] = df["domain_id"].astype(str)

    times = np.array(sorted(df["time"].unique()), dtype=float)
    # Pivot to matrices: rows=domain, cols=time
    query_pivot = (
        df.pivot(index="domain_id", columns="time", values="query_area_fraction")
        .reindex(index=domain_ids, columns=times)
    )
    analysis_pivot = (
        df.pivot(index="domain_id", columns="time", values="analysis_area_fraction")
        .reindex(index=domain_ids, columns=times)
    )

    query_mat = query_pivot.to_numpy(dtype=float)
    analysis_mat = analysis_pivot.to_numpy(dtype=float)
    delta_mat = query_mat - analysis_mat

    # Order domains for nicer visualization
    if domain_order == "variation":
        score = np.nanmax(query_mat, axis=1) - np.nanmin(query_mat, axis=1)
        order = np.argsort(score)
    elif domain_order == "mean_query":
        score = np.nanmean(query_mat, axis=1)
        order = np.argsort(score)
    elif domain_order == "label":
        order = np.arange(len(domain_ids))
    else:
        raise ValueError("domain_order must be one of: 'variation', 'mean_query', 'label'")

    domain_ids = [domain_ids[i] for i in order]
    query_mat = query_mat[order]
    analysis_mat = analysis_mat[order]
    delta_mat = delta_mat[order]

    color_map = _build_domain_colors(domain_ids)
    colors = [color_map[d] for d in domain_ids]

    # Domains to highlight in Panel B
    variability = np.nanmax(query_mat, axis=1) - np.nanmin(query_mat, axis=1)
    n_show = min(top_n_line_domains, len(domain_ids))
    line_idx = np.argsort(-variability)[:n_show]
    line_domains = [domain_ids[i] for i in line_idx]

    # ----- Figure layout -----
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        3, 1,
        height_ratios=[1.25, 1.15, 1.0],
        hspace=0.28,
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # =========================================================
    # Panel A: Stacked area plot of query-space fractions
    # =========================================================
    ax1.stackplot(
        times,
        query_mat,
        labels=domain_ids,
        colors=colors,
        alpha=0.95,
        linewidth=0.4,
    )
    ax1.set_ylabel("Query area fraction")
    ax1.set_title("A. Domain composition in query space", loc="left", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(1.0, np.nanmax(np.nansum(query_mat, axis=0)) * 1.02))
    ax1.grid(alpha=0.18, linewidth=0.6)

    # =========================================================
    # Panel B: Query vs analysis trajectories
    # =========================================================
    for d in line_domains:
        i = domain_ids.index(d)
        q = query_mat[i]
        a = analysis_mat[i]
        c = color_map[d]

        ax2.plot(
            times, q,
            marker="o",
            markersize=3.5,
            linewidth=2.2,
            color=c,
            label=f"Domain {d}",
        )
        ax2.plot(
            times, a,
            linestyle="--",
            linewidth=1.6,
            color=c,
            alpha=0.95,
        )
        ax2.fill_between(
            times, q, a,
            color=c,
            alpha=0.12,
            linewidth=0,
        )

    ax2.set_ylabel("Area fraction")
    ax2.set_title(
        f"B. Dominant changing domains (solid=query, dashed=analysis)",
        loc="left", fontsize=12, fontweight="bold"
    )
    ax2.grid(alpha=0.18, linewidth=0.6)

    # Custom legend for panel B
    domain_handles = [
        Line2D([0], [0], color=color_map[d], lw=2.2, marker="o", markersize=4, label=f"Domain {d}")
        for d in line_domains
    ]
    style_handles = [
        Line2D([0], [0], color="black", lw=2.2, ls="-", label="query"),
        Line2D([0], [0], color="black", lw=1.6, ls="--", label="analysis"),
    ]
    ax2.legend(
        handles=domain_handles + style_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9,
    )

    # =========================================================
    # Panel C: Heatmap of query-analysis difference
    # =========================================================
    vmax = np.nanmax(np.abs(delta_mat))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-6

    im = ax3.imshow(
        delta_mat,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[times.min(), times.max(), len(domain_ids) - 0.5, -0.5],
    )

    ax3.set_yticks(np.arange(len(domain_ids)))
    ax3.set_yticklabels([f"D{d}" for d in domain_ids], fontsize=8)
    ax3.set_xlabel("Virtual time")
    ax3.set_ylabel("Domain")
    ax3.set_title(
        "C. Query − analysis area fraction",
        loc="left", fontsize=12, fontweight="bold"
    )

    cbar = fig.colorbar(im, ax=ax3, fraction=0.02, pad=0.015)
    cbar.set_label("Δ area fraction", rotation=90)

    # Shared x limits
    ax1.set_xlim(times.min(), times.max())

    # Overall title
    fig.suptitle(title, fontsize=15, y=0.995)

    # Domain palette legend for stacked area can get huge; keep it compact
    if len(domain_ids) <= 16:
        palette_handles = [
            Patch(facecolor=color_map[d], edgecolor="none", label=f"Domain {d}")
            for d in domain_ids
        ]
        fig.legend(
            handles=palette_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=min(8, len(domain_ids)),
            frameon=False,
            fontsize=9,
        )
        fig.subplots_adjust(bottom=0.10)
    else:
        fig.subplots_adjust(bottom=0.06)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, (ax1, ax2, ax3)

# ============================================================
# Utilities
# ============================================================

def parse_time_from_filename(path: Path) -> float:
    """
    Parse time from filename of the form *_<time>.h5ad.

    Examples
    --------
    sample_0.h5ad      -> 0.0
    virtual_1.25.h5ad  -> 1.25
    state_-0.5.h5ad    -> -0.5
    """
    name = path.name
    m = re.search(r'_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.h5ad$', name)
    if m is None:
        raise ValueError(
            f"Could not parse time from filename: {path.name}. "
            f"Expected pattern '*_[time].h5ad'."
        )
    return float(m.group(1))


def sort_labels_nicely(labels: List[Any]) -> List[str]:
    """
    Sort labels numerically if possible, otherwise lexicographically.
    """
    labels = [str(x) for x in pd.unique(pd.Series(labels).astype(str))]

    def key_fn(x: str):
        try:
            return (0, int(x))
        except Exception:
            try:
                return (1, float(x))
            except Exception:
                return (2, x)

    return sorted(labels, key=key_fn)


def collect_time_matched_pairs(root: Path) -> List[Dict[str, Any]]:
    """
    Collect and match analysis/query files by parsed time.
    """
    analysis_dir = root / "analysis"
    query_dir = root / "query"

    if not analysis_dir.exists():
        raise FileNotFoundError(f"Missing folder: {analysis_dir}")
    if not query_dir.exists():
        raise FileNotFoundError(f"Missing folder: {query_dir}")

    analysis_files = sorted(analysis_dir.glob("*.h5ad"))
    query_files = sorted(query_dir.glob("*.h5ad"))

    if len(analysis_files) == 0:
        raise FileNotFoundError(f"No .h5ad files found in: {analysis_dir}")
    if len(query_files) == 0:
        raise FileNotFoundError(f"No .h5ad files found in: {query_dir}")

    analysis_map = {}
    for p in analysis_files:
        t = parse_time_from_filename(p)
        if t in analysis_map:
            raise ValueError(f"Duplicate analysis time {t} in: {p}")
        analysis_map[t] = p

    query_map = {}
    for p in query_files:
        t = parse_time_from_filename(p)
        if t in query_map:
            raise ValueError(f"Duplicate query time {t} in: {p}")
        query_map[t] = p

    common_times = sorted(set(analysis_map.keys()) & set(query_map.keys()))
    if len(common_times) == 0:
        raise ValueError("No matched times found between analysis/ and query/ folders.")

    missing_analysis = sorted(set(query_map.keys()) - set(analysis_map.keys()))
    missing_query = sorted(set(analysis_map.keys()) - set(query_map.keys()))

    if len(missing_analysis) > 0:
        print(f"[WARN] Times present in query but missing in analysis: {missing_analysis}")
    if len(missing_query) > 0:
        print(f"[WARN] Times present in analysis but missing in query: {missing_query}")

    pairs = []
    for t in common_times:
        pairs.append(
            {
                "time": t,
                "analysis_path": analysis_map[t],
                "query_path": query_map[t],
            }
        )
    return pairs


def ensure_xy(coords, path: Path, key: str) -> np.ndarray:
    """
    Validate coordinates as (n_cells, >=2) dense array and return first 2 dims.
    """
    if sp.issparse(coords):
        coords = coords.toarray()
    coords = np.asarray(coords)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"{path}: obsm['{key}'] must have shape (n_cells, >=2), got {coords.shape}"
        )
    if coords.shape[0] == 0:
        raise ValueError(f"{path}: obsm['{key}'] has zero rows.")
    if not np.all(np.isfinite(coords[:, :2])):
        raise ValueError(f"{path}: obsm['{key}'] contains non-finite coordinates.")

    return coords[:, :2].astype(np.float32)


def load_slice_minimal(
    path: Path,
    coord_key: str = "spatial",
    domain_key: str = "axis_joint_domain",
) -> Dict[str, Any]:
    """
    Load minimal data needed for overlay plotting.
    """
    adata = ad.read_h5ad(path)

    if coord_key not in adata.obsm:
        raise KeyError(f"{path}: missing obsm['{coord_key}']")
    if domain_key not in adata.obs:
        raise KeyError(f"{path}: missing obs['{domain_key}']")

    coords = ensure_xy(adata.obsm[coord_key], path=path, key=coord_key)
    labels = adata.obs[domain_key].astype(str).to_numpy()

    if len(labels) != coords.shape[0]:
        raise ValueError(
            f"{path}: obs['{domain_key}'] length {len(labels)} != n_coords {coords.shape[0]}"
        )

    return {
        "path": path,
        "coords": coords,
        "labels": labels,
        "n_obs": coords.shape[0],
    }


def estimate_support_radius(coords: np.ndarray) -> float:
    """
    Estimate a support radius for nearest-neighbor rasterization.

    The radius is based on local point spacing so the background is not filled
    too aggressively far from sampled tissue points.
    """
    n = coords.shape[0]
    if n <= 1:
        return 0.0

    tree = cKDTree(coords)
    k = min(3, n)
    dists, _ = tree.query(coords, k=k)

    # dists[:, 0] is self-distance = 0
    # use nearest non-self distance
    if k == 1:
        nn = np.zeros(n, dtype=np.float32)
    else:
        nn = dists[:, 1]

    nn = nn[np.isfinite(nn)]
    nn = nn[nn > 0]
    if len(nn) == 0:
        return 0.0

    # Slightly generous support to make contours continuous
    return float(np.median(nn))



def compute_global_extent(
    pairs_data: List[Dict[str, Any]],
    pad_ratio: float = 0.03,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute one global extent across all analysis/query slices.
    """
    all_x = []
    all_y = []

    for item in pairs_data:
        for space in ("analysis", "query"):
            xy = item[space]["coords"]
            all_x.append(xy[:, 0])
            all_y.append(xy[:, 1])

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    xspan = max(xmax - xmin, 1e-6)
    yspan = max(ymax - ymin, 1e-6)

    xpad = xspan * pad_ratio
    ypad = yspan * pad_ratio

    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def make_visual_grid_from_extent(
    xlim,
    ylim,
    nx: int = 250,
):
    xmin, xmax = xlim
    ymin, ymax = ylim

    xspan = max(xmax - xmin, 1e-6)
    yspan = max(ymax - ymin, 1e-6)

    ny = max(50, int(round(nx * (yspan / xspan))))

    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(gx, gy)
    return X, Y


def compute_domain_local_extent(
    analysis_coords: np.ndarray,
    analysis_labels: np.ndarray,
    query_coords: np.ndarray,
    query_labels: np.ndarray,
    domain_id: str,
    pad_ratio: float = 0.08,
    min_pad: float = 5.0,
):
    """
    Compute a tight plotting extent around the current domain,
    using the union of analysis/query points belonging to that domain.
    """
    domain_id = str(domain_id)

    a_mask = (analysis_labels.astype(str) == domain_id)
    q_mask = (query_labels.astype(str) == domain_id)

    pts = []
    if np.any(a_mask):
        pts.append(analysis_coords[a_mask, :2])
    if np.any(q_mask):
        pts.append(query_coords[q_mask, :2])

    if len(pts) == 0:
        return None, None

    pts = np.vstack(pts)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    xspan = max(float(xmax - xmin), 1e-6)
    yspan = max(float(ymax - ymin), 1e-6)

    xpad = max(xspan * pad_ratio, min_pad)
    ypad = max(yspan * pad_ratio, min_pad)

    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)

def make_visual_grid(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    nx: int = 350,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a visualization grid with aspect-ratio-aware ny.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim
    xspan = max(xmax - xmin, 1e-6)
    yspan = max(ymax - ymin, 1e-6)

    ny = max(50, int(round(nx * (yspan / xspan))))

    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(gx, gy)
    return X, Y


# def rasterize_labels_nearest(
#     coords: np.ndarray,
#     labels: np.ndarray,
#     X: np.ndarray,
#     Y: np.ndarray,
#     support_radius: float | None = None,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Rasterize categorical labels by nearest-neighbor assignment on grid.
#
#     Returns
#     -------
#     label_grid : ndarray of shape X.shape, dtype object/str
#     valid_mask : ndarray of shape X.shape, bool
#         Points farther than support_radius from any observation are invalid.
#     """
#     if coords.shape[0] == 0:
#         raise ValueError("Cannot rasterize empty coordinate set.")
#
#     tree = cKDTree(coords)
#     query_points = np.column_stack([X.ravel(), Y.ravel()])
#     dists, idx = tree.query(query_points, k=1)
#
#
#     label_grid = labels[idx].reshape(X.shape)
#
#
#     if support_radius is None or support_radius <= 0:
#         valid_mask = np.ones(X.shape, dtype=bool)
#     else:
#         valid_mask = (dists.reshape(X.shape) <= support_radius)
#
#     return label_grid, valid_mask

def rasterize_labels_nearest(
    coords: np.ndarray,
    labels: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    support_radius: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterize categorical labels by nearest-neighbor assignment on grid.

    Returns
    -------
    label_grid : ndarray of shape X.shape, dtype object/str
    valid_mask : ndarray of shape X.shape, bool
        Points farther than support_radius from any observation are invalid.
        Also marks NaN-labeled locations as invalid.
    """
    if coords.shape[0] == 0:
        raise ValueError("Cannot rasterize empty coordinate set.")

    tree = cKDTree(coords)
    query_points = np.column_stack([X.ravel(), Y.ravel()])
    dists, idx = tree.query(query_points, k=1)

    label_grid = labels[idx].reshape(X.shape)

    if support_radius is None or support_radius <= 0:
        valid_mask = np.ones(X.shape, dtype=bool)
    else:
        valid_mask = (dists.reshape(X.shape) <= support_radius)

    nan_locations = (label_grid == "nan")
    valid_mask[nan_locations] = False
    return label_grid, valid_mask


def prepare_pair_rasters(
    pair_item: Dict[str, Any],
    X: np.ndarray,
    Y: np.ndarray,
) -> Dict[str, Any]:
    """
    Precompute rasterized label grids for one time pair.
    """
    out = dict(pair_item)

    for space in ("analysis", "query"):
        coords = pair_item[space]["coords"]
        labels = pair_item[space]["labels"]
        radius = estimate_support_radius(coords)


        label_grid, valid_mask = rasterize_labels_nearest(
            coords=coords,
            labels=labels,
            X=X,
            Y=Y,
            support_radius=radius,
        )



        out[space] = {
            **pair_item[space],
            "support_radius": radius,
            "label_grid": label_grid,
            "valid_mask": valid_mask,
        }

    return out


def domain_mask(
    label_grid: np.ndarray,
    valid_mask: np.ndarray,
    domain_id: str,
) -> np.ndarray:
    """
    Build binary mask for one domain.
    """
    return (label_grid == str(domain_id)) & valid_mask


def area_fraction(mask: np.ndarray, valid_mask: np.ndarray) -> float:
    """
    Fraction of valid pixels occupied by the domain.
    """
    denom = int(np.sum(valid_mask))
    if denom == 0:
        return np.nan
    return float(np.sum(mask) / denom)


# ============================================================
# Plotting
# ============================================================

def plot_one_overlay_panel(
    ax,
    X: np.ndarray,
    Y: np.ndarray,
    analysis_mask: np.ndarray,
    query_mask: np.ndarray,
    time_value: float,
    analysis_color: str = "#4C78A8",
    query_color: str = "#F58518",
    analysis_alpha: float = 0.28,
    query_alpha: float = 0.28,
    annotate_fraction: bool = True,
    analysis_fraction: float | None = None,
    query_fraction: float | None = None,
    invert_y: bool = True,
):
    """
    Plot one time-point overlay panel for one domain.
    """
    # Filled regions
    if np.any(analysis_mask):
        ax.contourf(
            X, Y, analysis_mask.astype(float),
            levels=[0.5, 1.5],
            colors=[analysis_color],
            alpha=analysis_alpha,
            antialiased=True,
        )
        ax.contour(
            X, Y, analysis_mask.astype(float),
            levels=[0.5],
            colors=[analysis_color],
            linewidths=1.4,
            linestyles="--",
        )

    if np.any(query_mask):
        ax.contourf(
            X, Y, query_mask.astype(float),
            levels=[0.5, 1.5],
            colors=[query_color],
            alpha=query_alpha,
            antialiased=True,
        )
        ax.contour(
            X, Y, query_mask.astype(float),
            levels=[0.5],
            colors=[query_color],
            linewidths=1.6,
            linestyles="-",
        )

    ax.set_title(f"time = {time_value:g}", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    if invert_y:
        ax.invert_yaxis()

    if annotate_fraction:
        a_txt = "nan" if analysis_fraction is None or np.isnan(analysis_fraction) else f"{analysis_fraction*100:.1f}%"
        q_txt = "nan" if query_fraction is None or np.isnan(query_fraction) else f"{query_fraction*100:.1f}%"
        ax.text(
            0.02, 0.82,
            f"A: {a_txt}\nQ: {q_txt}",
            transform=ax.transAxes,
            fontsize=16,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )


def plot_domain_sequence(
    pairs_rasterized: List[Dict[str, Any]],
    domain_id: str,
    X: np.ndarray,
    Y: np.ndarray,
    out_path: Path,
    ncols: int = 4,
    analysis_color: str = "#4C78A8",
    query_color: str = "#F58518",
    analysis_alpha: float = 0.28,
    query_alpha: float = 0.28,
    annotate_fraction: bool = True,
    invert_y: bool = True,
    dpi: int = 220,
):
    """
    Make one sequential figure for a single domain across all times.
    """
    n_time = len(pairs_rasterized)
    ncols = min(max(1, ncols), n_time)
    nrows = int(math.ceil(n_time / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 4.2 * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    legend_handles = [
        Patch(facecolor=analysis_color, edgecolor="none", alpha=analysis_alpha, label="analysis fill"),
        Line2D([0], [0], color=analysis_color, lw=1.6, ls="--", label="analysis boundary"),
        Patch(facecolor=query_color, edgecolor="none", alpha=query_alpha, label="query fill"),
        Line2D([0], [0], color=query_color, lw=1.8, ls="-", label="query boundary"),
    ]

    for ax, item in zip(axes_flat, pairs_rasterized):
        a_mask = domain_mask(
            label_grid=item["analysis"]["label_grid"],
            valid_mask=item["analysis"]["valid_mask"],
            domain_id=domain_id,
        )
        q_mask = domain_mask(
            label_grid=item["query"]["label_grid"],
            valid_mask=item["query"]["valid_mask"],
            domain_id=domain_id,
        )


        a_frac = area_fraction(a_mask, item["analysis"]["valid_mask"])
        q_frac = area_fraction(q_mask, item["query"]["valid_mask"])
        #
        # plot_one_overlay_panel(
        #     ax=ax,
        #     X=X,
        #     Y=Y,
        #     analysis_mask=a_mask,
        #     query_mask=q_mask,
        #     time_value=item["time"],
        #     analysis_color=analysis_color,
        #     query_color=query_color,
        #     analysis_alpha=analysis_alpha,
        #     query_alpha=query_alpha,
        #     annotate_fraction=annotate_fraction,
        #     analysis_fraction=a_frac,
        #     query_fraction=q_frac,
        #     invert_y=invert_y,
        # )
        xlim, ylim = compute_domain_local_extent(
            analysis_coords=item["analysis"]["coords"],
            analysis_labels=item["analysis"]["labels"],
            query_coords=item["query"]["coords"],
            query_labels=item["query"]["labels"],
            domain_id=domain_id,
            pad_ratio=0.08,
            min_pad=5.0,
        )

        if xlim is None:
            ax.axis("off")
            continue

        X_local, Y_local = make_visual_grid_from_extent(
            xlim=xlim,
            ylim=ylim,
            nx=250,
        )

        # re-rasterize only inside this local box
        a_label_grid, a_valid_mask = rasterize_labels_nearest(
            coords=item["analysis"]["coords"],
            labels=item["analysis"]["labels"],
            X=X_local,
            Y=Y_local,
            support_radius=item["analysis"]["support_radius"],
        )
        q_label_grid, q_valid_mask = rasterize_labels_nearest(
            coords=item["query"]["coords"],
            labels=item["query"]["labels"],
            X=X_local,
            Y=Y_local,
            support_radius=item["query"]["support_radius"],
        )

        a_mask = domain_mask(a_label_grid, a_valid_mask, domain_id)
        q_mask = domain_mask(q_label_grid, q_valid_mask, domain_id)



        plot_one_overlay_panel(
            ax=ax,
            X=X_local,
            Y=Y_local,
            analysis_mask=a_mask,
            query_mask=q_mask,
            time_value=item["time"],
            analysis_color=analysis_color,
            query_color=query_color,
            analysis_alpha=analysis_alpha,
            query_alpha=query_alpha,
            annotate_fraction=annotate_fraction,
            analysis_fraction=a_frac,
            query_fraction=q_frac,
            invert_y=invert_y,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Turn off unused axes
    for j in range(len(pairs_rasterized), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(f"Domain {domain_id}: analysis/query overlay across virtual time", y=0.995, fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main pipeline
# ============================================================

def generate_domain_overlay_sequences(
    root: str | Path,
    analysis_coord_key: str = "spatial",
    query_coord_key: str = "spatial",
    domain_key: str = "axis_joint_domain",
    nx: int = 350,
    ncols: int = 4,
    global_extent: bool = False,
    analysis_color: str = "#4C78A8",
    query_color: str = "#F58518",
    analysis_alpha: float = 0.28,
    query_alpha: float = 0.28,
    annotate_fraction: bool = True,
    invert_y: bool = False,
    save_area_csv: bool = True,
    out_dir_name: str = "domain_overlay_sequence",
    dpi: int = 220,
):
    """
    Main entry point.
    """
    root = Path(root)
    pairs = collect_time_matched_pairs(root)

    print(f"[INFO] Found {len(pairs)} matched time pairs")

    # Load minimal data once
    pairs_data = []
    all_domains = []

    for p in pairs:
        analysis_item = load_slice_minimal(
            p["analysis_path"],
            coord_key=analysis_coord_key,
            domain_key=domain_key,
        )
        query_item = load_slice_minimal(
            p["query_path"],
            coord_key=query_coord_key,
            domain_key=domain_key,
        )

        pair_item = {
            "time": p["time"],
            "analysis": analysis_item,
            "query": query_item,
        }
        pairs_data.append(pair_item)

        all_domains.extend(analysis_item["labels"])
        all_domains.extend(query_item["labels"])

    domain_ids = sort_labels_nicely([d for d in all_domains if pd.notna(d) and str(d).lower() != "nan"])
    print(f"[INFO] Found {len(domain_ids)} unique domain IDs: {domain_ids}")

    # Global extent or per-time extent
    if global_extent:
        xlim, ylim = compute_global_extent(pairs_data)
    else:
        # fallback: compute from all anyway for simpler plotting consistency
        xlim, ylim = compute_global_extent(pairs_data)

    X, Y = make_visual_grid(xlim=xlim, ylim=ylim, nx=nx)

    # Precompute raster grids
    pairs_rasterized = []
    for item in pairs_data:
        pairs_rasterized.append(prepare_pair_rasters(item, X=X, Y=Y))

    # Save sequential plots
    out_dir = root / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for domain_id in domain_ids:
        print(f"[INFO] Plotting domain {domain_id}")

        # collect summary while looping
        for item in pairs_rasterized:
            a_mask = domain_mask(item["analysis"]["label_grid"], item["analysis"]["valid_mask"], domain_id)
            q_mask = domain_mask(item["query"]["label_grid"], item["query"]["valid_mask"], domain_id)


            summary_rows.append(
                {
                    "time": item["time"],
                    "domain_id": domain_id,
                    "analysis_area_fraction": area_fraction(a_mask, item["analysis"]["valid_mask"]),
                    "query_area_fraction": area_fraction(q_mask, item["query"]["valid_mask"]),
                    "analysis_n_valid_pixels": int(np.sum(item["analysis"]["valid_mask"])),
                    "query_n_valid_pixels": int(np.sum(item["query"]["valid_mask"])),
                    "analysis_n_domain_pixels": int(np.sum(a_mask)),
                    "query_n_domain_pixels": int(np.sum(q_mask)),
                    "analysis_file": str(item["analysis"]["path"]),
                    "query_file": str(item["query"]["path"]),
                }
            )
        #
        summary_df = build_domain_fraction_summary_from_pairs_rasterized(pairs_rasterized)

        # fig, axes = plot_domain_variation_fancy(
        #     summary_df,
        #     domain_ids,
        #     title="AXIS virtual domain variation across developmental time",
        #     top_n_line_domains=8,
        #     domain_order="variation",
        #     save_path=None,
        # )
        # plt.show()

        safe_domain_name = str(domain_id).replace("/", "_")
        out_path = out_dir / f"domain_{safe_domain_name}.png"

        plot_domain_sequence(
            pairs_rasterized=pairs_rasterized,
            domain_id=domain_id,
            X=X,
            Y=Y,
            out_path=out_path,
            ncols=ncols,
            analysis_color=analysis_color,
            query_color=query_color,
            analysis_alpha=analysis_alpha,
            query_alpha=query_alpha,
            annotate_fraction=annotate_fraction,
            invert_y=invert_y,
            dpi=dpi,
        )

    if save_area_csv:
        df = pd.DataFrame(summary_rows).sort_values(["domain_id", "time"]).reset_index(drop=True)
        csv_path = out_dir / "domain_area_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved area summary to: {csv_path}")

    print(f"[INFO] Saved overlay figures to: {out_dir}")


# ============================================================
# CLI
# ============================================================


root= "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/Tg_time_grid256"

generate_domain_overlay_sequences(
    root=root,
    analysis_coord_key="spatial",
    query_coord_key="spatial",
    domain_key="axis_joint_domain",
    nx=250,
    ncols=6,
    save_area_csv=False,
    out_dir_name="domain_overlay",
    dpi=220,
)
