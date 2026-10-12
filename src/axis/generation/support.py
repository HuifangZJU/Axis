"""User-defined spatial supports for generated tissues.

A virtual tissue is generated on an output support ``U = {u_m}`` that can follow
a Visium-like hexagonal spot layout, a regular pixel grid, observed single-cell
coordinates or any other sampling scheme (Methods, "Evidence-weighted spatial
generation"). Locations farther than ``max_dist`` from the nearest projected
evidence point are excluded by a tissue-support filter.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

__all__ = [
    "make_common_grid_from_coords",
    "infer_grid_spacing",
    "estimate_grid_spacing",
    "keep_largest_connected_component_on_grid",
    "smooth_mask_on_grid",
    "compute_dense_tissue_mask_on_any_grid",
]


def infer_grid_spacing(grid_pts: np.ndarray) -> float:
    """Half of the median axis-aligned spacing of a grid (default spatial bandwidth)."""
    grid_pts = np.asarray(grid_pts, dtype=np.float32)
    if grid_pts.ndim != 2 or grid_pts.shape[1] != 2:
        raise ValueError("grid_pts must have shape (M, 2).")
    ux = np.unique(np.round(grid_pts[:, 0], 6))
    uy = np.unique(np.round(grid_pts[:, 1], 6))
    vals = []
    if ux.size > 1:
        vals.append(np.median(np.diff(np.sort(ux))))
    if uy.size > 1:
        vals.append(np.median(np.diff(np.sort(uy))))
    if not vals:
        return 1.0
    return float(np.median(vals)) / 2


def estimate_grid_spacing(grid_pts: np.ndarray) -> float:
    """Median nearest-neighbour distance of a (square or hexagonal) grid."""
    grid_pts = np.asarray(grid_pts, dtype=np.float32)
    if grid_pts.ndim != 2 or grid_pts.shape[1] != 2:
        raise ValueError(f"grid_pts must have shape (N,2), got {grid_pts.shape}")
    if grid_pts.shape[0] < 2:
        raise ValueError("Need at least 2 grid points to estimate spacing")
    dists, _ = cKDTree(grid_pts).query(grid_pts, k=2)
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        raise ValueError("Could not estimate grid spacing")
    return float(np.median(nn))


def keep_largest_connected_component_on_grid(grid_pts: np.ndarray, mask: np.ndarray, neighbor_radius: float) -> np.ndarray:
    """Keep only the largest radius-connected component of the masked grid points."""
    grid_pts = np.asarray(grid_pts, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask)[0]
    if idx.size <= 1:
        return mask
    tree = cKDTree(grid_pts[idx])
    pairs = tree.query_pairs(r=float(neighbor_radius), output_type="ndarray")
    if pairs.size == 0:
        return mask
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    graph = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(idx), len(idx)))
    n_comp, labels = connected_components(graph, directed=False)
    if n_comp <= 1:
        return mask
    keep_label = np.argmax(np.bincount(labels))
    new_mask = np.zeros_like(mask, dtype=bool)
    new_mask[idx[labels == keep_label]] = True
    return new_mask


def smooth_mask_on_grid(
    grid_pts: np.ndarray, mask: np.ndarray, neighbor_radius: float, min_keep_neighbors: int = 2
) -> np.ndarray:
    """Keep a grid point if at least ``min_keep_neighbors`` points within the radius are active."""
    grid_pts = np.asarray(grid_pts, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    neighbors = cKDTree(grid_pts).query_ball_point(grid_pts, r=float(neighbor_radius))
    counts = np.fromiter((np.sum(mask[n]) for n in neighbors), dtype=np.int64, count=len(neighbors))
    return counts >= int(min_keep_neighbors)


def compute_dense_tissue_mask_on_any_grid(
    grid_pts,
    union_spots,
    max_dist_px: float = 9.0,
    smooth: bool = True,
    smooth_radius_factor: float = 1.25,
    min_keep_neighbors: int = 2,
    keep_largest: bool = True,
) -> np.ndarray:
    """Tissue-support filter for an output grid.

    1. keep grid points within ``max_dist_px`` of an evidence point;
    2. optional graph-based smoothing;
    3. optionally keep the largest connected supported region.
    """
    grid_pts = np.asarray(grid_pts, dtype=np.float32)
    union_spots = np.asarray(union_spots, dtype=np.float32)
    if grid_pts.ndim != 2 or grid_pts.shape[1] != 2:
        raise ValueError(f"grid_pts must have shape (M,2), got {grid_pts.shape}")
    if union_spots.ndim != 2 or union_spots.shape[1] != 2:
        raise ValueError(f"union_spots must have shape (N,2), got {union_spots.shape}")

    dist, _ = cKDTree(union_spots).query(grid_pts, k=1)
    mask = dist <= float(max_dist_px)
    neighbor_radius = float(smooth_radius_factor) * estimate_grid_spacing(grid_pts)
    if smooth:
        mask = smooth_mask_on_grid(grid_pts, mask, neighbor_radius, min_keep_neighbors=min_keep_neighbors)
    if keep_largest:
        mask = keep_largest_connected_component_on_grid(grid_pts, mask, neighbor_radius)
    return mask


def make_common_grid_from_coords(
    grid_size: int,
    pad_ratio: float = 0.00,
    type: str = "pixel",
    max_x: float = 512,
    max_y: float = 512,
    min_x: float = 0,
    min_y: float = 0,
) -> Tuple[np.ndarray, Union[Tuple[np.ndarray, np.ndarray], Dict[str, object]]]:
    """Regular output supports over a bounding box.

    ``type="pixel"`` returns ``grid_size x grid_size`` cell centres and the cell
    edges; ``type="visium"`` returns a staggered hexagonal lattice with about
    ``grid_size**2`` spots and a metadata dict.
    """
    dx = (max_x - min_x) * pad_ratio
    dy = (max_y - min_y) * pad_ratio
    min_x, max_x, min_y, max_y = min_x - dx, max_x + dx, min_y - dy, max_y + dy

    if type == "pixel":
        x_edges = np.linspace(min_x, max_x, grid_size + 1, dtype=np.float32)
        y_edges = np.linspace(min_y, max_y, grid_size + 1, dtype=np.float32)
        gx, gy = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2.0, (y_edges[:-1] + y_edges[1:]) / 2.0)
        return np.vstack([gx.ravel(), gy.ravel()]).T.astype(np.float32), (x_edges, y_edges)

    if type == "visium":
        width, height = max_x - min_x, max_y - min_y
        target_n = max(int(grid_size) ** 2, 1)
        sx = np.sqrt((2.0 * width * height) / (np.sqrt(3.0) * target_n))
        sy = (np.sqrt(3.0) / 2.0) * sx
        pts = []
        for row_idx, y in enumerate(np.arange(min_y, max_y + sy, sy, dtype=np.float32)):
            x_vals = np.arange(min_x + (0.5 * sx if row_idx % 2 == 1 else 0.0), max_x + sx, sx, dtype=np.float32)
            x_vals = x_vals[(x_vals >= min_x) & (x_vals <= max_x)]
            if x_vals.size > 0:
                pts.append(np.column_stack([x_vals, np.full_like(x_vals, y)]))
        grid_pts = np.vstack(pts).astype(np.float32) if pts else np.empty((0, 2), dtype=np.float32)
        meta = {
            "bbox": (float(min_x), float(min_y), float(max_x), float(max_y)),
            "width": float(width),
            "height": float(height),
            "target_n": int(target_n),
            "n_generated": int(grid_pts.shape[0]),
            "x_spacing": float(sx),
            "y_spacing": float(sy),
            "layout": "staggered_hex",
        }
        return grid_pts, meta
    raise ValueError(f"Unknown grid type {type!r}; use 'pixel' or 'visium'.")
