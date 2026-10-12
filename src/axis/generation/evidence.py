"""Evidence-plane construction at a query state (Methods, Eqs. (24)-(29)).

For a query ``tau_q`` between anchors ``tau_a < tau_q < tau_b`` the learned
bidirectional fields are scaled by the relative position
``alpha_q = (tau_q - tau_a) / (tau_b - tau_a)``,

``D_{a->q} = alpha_q D_{a->b}``,  ``D_{b->q} = (1 - alpha_q) D_{b->a}``,

anchor entities are transported into the query anatomy, and the transported
points – which keep their molecular values, annotations and source-anchor
identity – are pooled into the evidence plane ``E_q``.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..settings import log

__all__ = [
    "relative_position_on_axis",
    "infer_axis_bandwidth",
    "axis_kernel",
    "compute_axis_kernel_weights",
    "get_anchor_pair_for_query",
    "matrix_log1p",
    "ensure_same_genes_keep_order",
    "subset_genes",
    "stack_weighted_evidence",
    "stack_attribute_evidence",
    "build_query_evidence",
]


def relative_position_on_axis(query_axis_value, axis_start_value, axis_end_value, clip=None) -> float:
    """``alpha = (q - start) / (end - start)``; ``<0`` or ``>1`` means extrapolation."""
    start, end, q = float(axis_start_value), float(axis_end_value), float(query_axis_value)
    denom = end - start
    if abs(denom) < 1e-12:
        raise ValueError("axis_start_value and axis_end_value must be different.")
    alpha = (q - start) / denom
    if clip is not None:
        alpha = float(np.clip(alpha, clip[0], clip[1]))
    return float(alpha)


def infer_axis_bandwidth(observed_axis_positions, min_bandwidth: float = 1e-6) -> float:
    """Default ``sigma_tau``: one third of the median spacing between anchors.

    With two anchors the weight of each anchor then decays to ~0.01 at the
    position of the other one.
    """
    pos = np.unique(np.sort(np.asarray(observed_axis_positions, dtype=np.float32).reshape(-1)))
    if pos.size <= 1:
        return 1.0
    diffs = np.diff(pos)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    return max(float(np.median(diffs)) / 3.0, float(min_bandwidth))


def axis_kernel(d, kernel: str = "gaussian", bandwidth: float = 1.0, power: float = 2.0, eps: float = 1e-8) -> np.ndarray:
    """Unnormalised kernel of the biological-axis distance ``d >= 0``."""
    d = np.asarray(d, dtype=np.float32)
    bandwidth = max(float(bandwidth), eps)
    if kernel == "gaussian":
        raw = np.exp(-0.5 * (d / bandwidth) ** 2)
    elif kernel == "laplacian":
        raw = np.exp(-(d / bandwidth))
    elif kernel == "inverse_distance":
        raw = 1.0 / np.maximum(d, eps) ** float(power)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")
    return raw.astype(np.float32, copy=False)


def compute_axis_kernel_weights(
    observed_axis_positions, query_axis_position, bandwidth: Optional[float] = None, eps: float = 1e-8
):
    """Normalised anchor weights ``beta_r``, Eqs. (28)-(29).

    Returns ``(weights, bandwidth_used)``.
    """
    obs = np.asarray(observed_axis_positions, dtype=np.float32).reshape(-1)
    if obs.size == 0:
        raise ValueError("observed_axis_positions must be non-empty.")
    if bandwidth is None:
        bandwidth = infer_axis_bandwidth(obs)
    bandwidth = max(float(bandwidth), eps)
    log(f"[axis] biological-axis bandwidth = {bandwidth}", level=2)
    raw = np.exp(-0.5 * (np.abs(obs - float(query_axis_position)) / bandwidth) ** 2)
    s = float(np.sum(raw))
    if s <= 0:
        raw = np.ones_like(raw, dtype=np.float32)
        s = float(np.sum(raw))
    return (raw / s).astype(np.float32, copy=False), float(bandwidth)


def get_anchor_pair_for_query(query_axis_value, axis_positions: Mapping, anchor_ids: Sequence) -> Tuple:
    """The two anchors bracketing the query (or the two nearest at an edge)."""
    ordered = sorted(anchor_ids, key=lambda k: float(axis_positions[k]))
    z = [float(axis_positions[k]) for k in ordered]
    q = float(query_axis_value)
    if len(ordered) < 2:
        raise ValueError("At least two anchors are required.")
    if q <= z[0]:
        return ordered[0], ordered[1]
    if q >= z[-1]:
        return ordered[-2], ordered[-1]
    for i in range(len(ordered) - 1):
        if z[i] <= q <= z[i + 1]:
            return ordered[i], ordered[i + 1]
    raise RuntimeError("Failed to find anchor pair for query.")


def matrix_log1p(X):
    """``log1p`` for dense or sparse matrices."""
    if sp.issparse(X):
        X = X.copy().tocsr()
        X.data = np.log1p(X.data).astype(np.float32, copy=False)
        return X
    return np.log1p(np.asarray(X, dtype=np.float32)).astype(np.float32, copy=False)


def ensure_same_genes_keep_order(Xa, ga, Xb, gb):
    """Intersect features keeping the order of ``ga``; returns ``(Xa', Xb', genes)``."""
    ga, gb = np.asarray(ga), np.asarray(gb)
    index_a, index_b = {}, {}
    for i, g in enumerate(ga.tolist()):
        index_a.setdefault(g, i)
    for i, g in enumerate(gb.tolist()):
        index_b.setdefault(g, i)
    common = [g for g in index_a if g in index_b]
    if not common:
        raise ValueError("No common genes between anchors.")
    idx_a = np.array([index_a[g] for g in common], dtype=np.int64)
    idx_b = np.array([index_b[g] for g in common], dtype=np.int64)
    return Xa[:, idx_a], Xb[:, idx_b], np.asarray(common)


def subset_genes(genes: np.ndarray, gene_subset: Optional[Sequence[str]], *matrices):
    """Restrict ``matrices`` (columns) and ``genes`` to ``gene_subset`` (in its order)."""
    if gene_subset is None:
        return (genes, *matrices)
    lookup = {g: i for i, g in enumerate(np.asarray(genes).tolist())}
    keep = [lookup[g] for g in gene_subset if g in lookup]
    if not keep:
        raise ValueError("None of the requested genes found in common gene list.")
    return (np.asarray(genes)[keep], *[m[:, keep] for m in matrices])


def stack_weighted_evidence(coords_blocks, X_blocks, source_weights, dtype=np.float32):
    """Concatenate transported evidence blocks and expand per-source weights per point.

    Returns ``(coords (N, 2), X (N, G) dense or CSR, point_weights (N,))``.
    """
    if len(coords_blocks) != len(X_blocks):
        raise ValueError("coords_blocks and X_blocks must have the same length.")
    source_weights = np.asarray(source_weights, dtype=np.float32).reshape(-1)
    if source_weights.shape[0] != len(coords_blocks):
        raise ValueError("source_weights length must match number of blocks.")

    coords_out, weights_out = [], []
    for i, (coords_i, X_i) in enumerate(zip(coords_blocks, X_blocks)):
        coords_i = np.asarray(coords_i, dtype=dtype)
        if coords_i.ndim != 2 or coords_i.shape[1] != 2:
            raise ValueError(f"coords_blocks[{i}] must have shape (Ni, 2).")
        if X_i.shape[0] != coords_i.shape[0]:
            raise ValueError(f"Row mismatch between coords_blocks[{i}] and X_blocks[{i}].")
        coords_out.append(coords_i)
        weights_out.append(np.full((coords_i.shape[0],), source_weights[i], dtype=dtype))

    coords_cat = np.vstack(coords_out).astype(dtype, copy=False)
    point_weights = np.concatenate(weights_out).astype(dtype, copy=False)
    if any(sp.issparse(x) for x in X_blocks):
        X_cat = sp.vstack([x if sp.issparse(x) else sp.csr_matrix(np.asarray(x, dtype=dtype)) for x in X_blocks]).tocsr()
    else:
        X_cat = np.vstack([np.asarray(x, dtype=dtype) for x in X_blocks]).astype(dtype, copy=False)
    return coords_cat, X_cat, point_weights


def stack_attribute_evidence(*attrs) -> Optional[pd.DataFrame]:
    """Concatenate per-anchor annotation tables (``None`` if any is missing)."""
    if any(a is None for a in attrs):
        return None
    return pd.concat(list(attrs), axis=0, ignore_index=True)


def build_query_evidence(
    alpha: float,
    grid_pts: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    expr_a,
    expr_b,
    flow_a_to_b: np.ndarray,
    flow_b_to_a: np.ndarray,
    source_weights,
    max_dist_px: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, object, np.ndarray]:
    """Evidence plane for an intermediate state, Eqs. (25)-(27).

    ``flow_a_to_b`` registers anchor ``a`` (moving) onto anchor ``b`` (fixed),
    i.e. the file ``{b}_{a}_flow.npy``.

    Returns ``(grid_support, coords_cat, X_cat, point_weights)``.
    """
    from ..registration.warp import find_inverse_warp_coords
    from .support import compute_dense_tissue_mask_on_any_grid

    coords_a_q = np.asarray(
        find_inverse_warp_coords(np.asarray(coords_a, np.float32), (alpha * flow_a_to_b).astype(np.float32)), np.float32
    )
    coords_b_q = np.asarray(
        find_inverse_warp_coords(np.asarray(coords_b, np.float32), ((1.0 - alpha) * flow_b_to_a).astype(np.float32)), np.float32
    )
    grid_support = compute_dense_tissue_mask_on_any_grid(
        grid_pts=grid_pts,
        union_spots=np.vstack([coords_a_q, coords_b_q]),
        max_dist_px=max_dist_px,
        keep_largest=True,
    )
    coords_cat, X_cat, point_weights = stack_weighted_evidence([coords_a_q, coords_b_q], [expr_a, expr_b], source_weights)
    return grid_support, coords_cat, X_cat, point_weights
