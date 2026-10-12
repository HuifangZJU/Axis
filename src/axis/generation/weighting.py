"""Evidence weights (Methods, Eqs. (28)-(33)).

For an output location ``u`` with neighbourhood ``N(u)`` of transported
evidence points, the contribution of point ``(r, i)`` is

``w_ri(u) = beta_r * K_sigma_x(|u - x~_ri|) * q_ri(u)``

* ``beta_r`` – biological-axis weight of the source anchor
  ([`compute_axis_kernel_weights`][axis.generation.evidence.compute_axis_kernel_weights]);
* ``K_sigma_x`` – Gaussian kernel of the local spatial distance;
* ``q_ri(u)`` – molecular-quality weight from total counts and detected
  features relative to the local medians, clipped to ``[0.5, 2]`` and raised to
  ``gamma_C = 1.0`` and ``gamma_G = 0.7``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from ..settings import log
from .support import infer_grid_spacing

__all__ = [
    "compute_qc_covariates",
    "compute_local_relative_quality_weights",
    "prepare_local_query_support",
]


def compute_qc_covariates(X) -> Tuple[np.ndarray, np.ndarray]:
    """Total counts ``C_ri`` and number of detected features ``G_ri`` per evidence point."""
    if sp.issparse(X):
        return (
            np.asarray(X.sum(axis=1)).ravel().astype(np.float32),
            np.asarray((X > 0).sum(axis=1)).ravel().astype(np.float32),
        )
    Xd = np.asarray(X, dtype=np.float32)
    return Xd.sum(axis=1).astype(np.float32), (Xd > 0).sum(axis=1).astype(np.float32)


def compute_local_relative_quality_weights(
    qc_total_local,
    qc_ngenes_local=None,
    gamma_counts: float = 0.5,
    gamma_genes: float = 0.5,
    rel_clip: Tuple[float, float] = (0.5, 2.0),
    eps: float = 1e-8,
) -> np.ndarray:
    """``q_ri(u) = clip(c_ri / median_N(u) c)^gamma_C * clip(g_ri / median_N(u) g)^gamma_G``.

    Inputs are ``(Q, k)`` arrays of raw counts / detected features of the ``k``
    neighbours of ``Q`` output locations; both are ``log1p``-transformed first.
    """
    q_total = np.log1p(np.asarray(qc_total_local, dtype=np.float32))
    rel_total = q_total / np.maximum(np.median(q_total, axis=1, keepdims=True), eps)
    rel_total = np.clip(rel_total, rel_clip[0], rel_clip[1]) ** gamma_counts
    if qc_ngenes_local is None:
        return rel_total.astype(np.float32, copy=False)
    q_ngenes = np.log1p(np.asarray(qc_ngenes_local, dtype=np.float32))
    rel_ngenes = q_ngenes / np.maximum(np.median(q_ngenes, axis=1, keepdims=True), eps)
    rel_ngenes = np.clip(rel_ngenes, rel_clip[0], rel_clip[1]) ** gamma_genes
    return (rel_total * rel_ngenes).astype(np.float32, copy=False)


def prepare_local_query_support(
    coords,
    grid_pts,
    grid_support_mask=None,
    point_weights=None,
    qc_total_counts=None,
    qc_n_genes=None,
    spatial_bandwidth: Optional[float] = None,
    knn: int = 48,
    min_neighbors: int = 6,
    radius: Optional[float] = None,
    gamma_counts: float = 1.0,
    gamma_genes: float = 0.7,
    rel_clip: Tuple[float, float] = (0.5, 2.0),
    dtype=np.float32,
) -> Dict[str, np.ndarray]:
    """Neighbourhoods and normalised weights ``w_ri(u)`` for every supported output location.

    Parameters
    ----------
    coords
        ``(N, 2)`` evidence coordinates in the query frame.
    grid_pts
        ``(M, 2)`` output locations.
    grid_support_mask
        Boolean mask of output locations eligible for generation.
    point_weights
        Per-evidence biological-axis weights ``beta_r`` (default 1).
    qc_total_counts, qc_n_genes
        Per-evidence quality covariates; when both are given the
        molecular-quality weight is applied.
    spatial_bandwidth
        ``sigma_x`` in canvas pixels. Defaults to ``infer_grid_spacing(grid_pts)``,
        half the median axis-aligned coordinate spacing of the output locations:
        half the pitch on a ``"pixel"`` grid, about one third of the spot pitch
        on the hexagonal ``"visium"`` grid.
    knn
        Number of nearest evidence points per output location; the
        ``max(knn, min_neighbors)`` nearest are used.
    min_neighbors
        Lower bound on the number of nearest evidence points; with ``radius``,
        the ``min_neighbors`` nearest are always kept.
    radius
        Optional radius: farther evidence points are dropped, except the
        ``min_neighbors`` nearest.
    """
    coords = np.asarray(coords, dtype=dtype)
    grid_pts = np.asarray(grid_pts, dtype=dtype)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2).")
    if grid_pts.ndim != 2 or grid_pts.shape[1] != 2:
        raise ValueError("grid_pts must have shape (M, 2).")
    n_points, n_grid = coords.shape[0], grid_pts.shape[0]

    point_weights = (
        np.ones(n_points, dtype=dtype) if point_weights is None else np.asarray(point_weights, dtype=dtype).reshape(-1)
    )
    if point_weights.shape[0] != n_points:
        raise ValueError("point_weights length must match coords rows.")
    valid_mask = (
        np.ones(n_grid, dtype=bool) if grid_support_mask is None else np.asarray(grid_support_mask, dtype=bool).reshape(-1)
    )
    if valid_mask.shape[0] != n_grid:
        raise ValueError("grid_support_mask length must match grid_pts rows.")

    if spatial_bandwidth is None:
        spatial_bandwidth = infer_grid_spacing(grid_pts)
    spatial_bandwidth = max(float(spatial_bandwidth), 1e-6)
    log(f"[axis] spatial bandwidth = {spatial_bandwidth}", level=2)

    confidence = np.zeros(n_grid, dtype=dtype)
    nearest_dist = np.full(n_grid, np.inf, dtype=dtype)
    active_query_support = np.zeros(n_grid, dtype=bool)

    k = max(1, min(int(max(knn, min_neighbors)), n_points))
    query_idx = np.where(valid_mask)[0]
    tree = cKDTree(coords)
    dists, nn_idx = tree.query(grid_pts[query_idx], k=k, workers=-1)
    if k == 1:
        dists, nn_idx = dists[:, None], nn_idx[:, None]
    dists = np.where(np.isfinite(dists), dists, 1e9)
    nearest_dist[query_idx] = dists[:, 0].astype(dtype, copy=False)

    neighbor_mask = np.ones_like(dists, dtype=bool)
    if radius is not None:
        neighbor_mask = dists <= float(radius)
        neighbor_mask[:, : min(min_neighbors, k)] = True

    kernel_w = np.exp(-0.5 * (dists / spatial_bandwidth) ** 2).astype(dtype, copy=False)
    kernel_w *= neighbor_mask.astype(dtype, copy=False)
    local_w_raw = kernel_w * point_weights[nn_idx]

    if qc_total_counts is not None and qc_n_genes is not None:
        local_w_raw *= compute_local_relative_quality_weights(
            np.asarray(qc_total_counts, dtype=dtype)[nn_idx],
            np.asarray(qc_n_genes, dtype=dtype)[nn_idx],
            gamma_counts=gamma_counts,
            gamma_genes=gamma_genes,
            rel_clip=rel_clip,
        )

    confidence_valid = np.sum(local_w_raw, axis=1).astype(dtype, copy=False)
    confidence[query_idx] = confidence_valid
    active_rows = np.where(confidence_valid > 0)[0]
    active_query_support[query_idx[active_rows]] = True
    query_idx_active = query_idx[active_rows]
    nn_idx_active = nn_idx[active_rows]
    local_w = local_w_raw[active_rows].copy()

    denom = np.sum(local_w, axis=1).astype(dtype, copy=False)
    zero_rows = denom <= 0
    if np.any(zero_rows):
        local_w[zero_rows, 0] = np.maximum(point_weights[nn_idx_active][zero_rows, 0], 1e-8)
    local_w /= np.maximum(np.sum(local_w, axis=1, keepdims=True), 1e-8)

    return {
        "valid_mask": valid_mask,
        "active_query_support": active_query_support,
        "confidence": confidence,
        "nearest_dist": nearest_dist,
        "spatial_bandwidth": spatial_bandwidth,
        "query_idx_active": query_idx_active,
        "nn_idx_active": nn_idx_active,
        "local_w": local_w,
    }
