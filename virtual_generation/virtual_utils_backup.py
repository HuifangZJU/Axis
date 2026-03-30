import os
import json
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from sklearn.decomposition import NMF
from scipy.spatial import cKDTree
from axis.coordinate_utils import find_inverse_warp_coords
from scipy.ndimage import binary_closing, binary_fill_holes, label

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from anndata import AnnData
from matplotlib import pyplot as plt

def relative_position_on_axis(query_axis_value, axis_start_value, axis_end_value, clip=None):
    """
    Map a query position on a biological axis to a relative scalar alpha.

    alpha = 0   -> axis_start_value
    alpha = 1   -> axis_end_value
    alpha < 0   -> backward extrapolation
    alpha > 1   -> forward extrapolation
    """
    axis_start_value = float(axis_start_value)
    axis_end_value = float(axis_end_value)
    query_axis_value = float(query_axis_value)

    denom = axis_end_value - axis_start_value
    if abs(denom) < 1e-12:
        raise ValueError("axis_start_value and axis_end_value must be different.")

    alpha = (query_axis_value - axis_start_value) / denom
    if clip is not None:
        alpha = float(np.clip(alpha, clip[0], clip[1]))
    return float(alpha)


def infer_axis_bandwidth(observed_axis_positions, min_bandwidth=1e-6):
    """
    Infer a reasonable bandwidth from observed positions on the biological axis.
    Uses the median spacing of unique sorted positions.
    """
    pos = np.asarray(observed_axis_positions, dtype=np.float32).reshape(-1)
    pos = np.unique(np.sort(pos))
    if pos.size <= 1:
        return 1.0
    diffs = np.diff(pos)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    bw = float(np.median(diffs))/3.0

    return max(bw, float(min_bandwidth))


def _axis_kernel_raw_from_distance(d, kernel="gaussian", bandwidth=1.0, power=2.0, eps=1e-8):
    """
    Raw unnormalized kernel value as a function of absolute distance d.
    d can be scalar or ndarray, must be >= 0.
    """
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

def plot_axis_kernel_weight_debug(
    observed_axis_positions,
    query_axis_position,
    kernel="gaussian",
    bandwidth=None,
    power=2.0,
    eps=1e-8,
    axis_min=None,
    axis_max=None,
    n_axis=400,
    max_decay_dist=None,
    n_decay=400,
    figsize=(12, 4),
    save_path=None,
    show=True,
):
    """
    Visualize:
      1) raw kernel decay K(d)
      2) normalized weights across the biological axis for two anchor points
      3) query point and its weight values

    Best for two observed points, but works with more.
    """
    obs = np.asarray(observed_axis_positions, dtype=np.float32).reshape(-1)
    if obs.size < 2:
        raise ValueError("Please provide at least two observed_axis_positions.")
    if obs.size != 2:
        print(f"[WARN] plot_axis_kernel_weight_debug is clearest with exactly 2 points, got {obs.size}.")

    q = float(query_axis_position)

    # infer bandwidth the same way as compute_axis_kernel_weights
    if bandwidth is None:
        diffs = np.diff(np.unique(np.sort(obs)))
        diffs = diffs[diffs > 0]
        bandwidth = float(np.median(diffs)) if diffs.size > 0 else 1.0
        bandwidth = bandwidth/3
    bandwidth = max(float(bandwidth), eps)

    # -------- left panel: raw kernel decay K(d) --------
    if max_decay_dist is None:
        max_decay_dist = max(2.5 * bandwidth, 1.2 * np.max(np.abs(obs - q)), 1e-3)
    dist_grid = np.linspace(0.0, float(max_decay_dist), int(n_decay), dtype=np.float32)
    decay_vals = _axis_kernel_raw_from_distance(
        d=dist_grid,
        kernel=kernel,
        bandwidth=bandwidth,
        power=power,
        eps=eps,
    )

    # -------- right panel: normalized weights across query axis --------
    if axis_min is None:
        axis_min = float(min(np.min(obs), q) - 0.8 * (np.max(obs) - np.min(obs) + 1e-6))
    if axis_max is None:
        axis_max = float(max(np.max(obs), q) + 0.8 * (np.max(obs) - np.min(obs) + 1e-6))
    axis_grid = np.linspace(axis_min, axis_max, int(n_axis), dtype=np.float32)

    weight_curves = []
    for tau in axis_grid:
        w, _ = compute_axis_kernel_weights(
            observed_axis_positions=obs,
            query_axis_position=float(tau),
            kernel=kernel,
            bandwidth=bandwidth,
            power=power,
            eps=eps,
        )
        weight_curves.append(w)
    weight_curves = np.asarray(weight_curves, dtype=np.float32)  # (n_axis, n_obs)

    wq, _ = compute_axis_kernel_weights(
        observed_axis_positions=obs,
        query_axis_position=q,
        kernel=kernel,
        bandwidth=bandwidth,
        power=power,
        eps=eps,
    )

    # -------- plotting --------
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: raw decay
    ax = axes[0]
    ax.plot(dist_grid, decay_vals, linewidth=2)
    ax.set_title(f"Raw {kernel} kernel decay")
    ax.set_xlabel("Absolute biological distance")
    ax.set_ylabel("Raw kernel value")
    ax.axvline(bandwidth, linestyle="--", linewidth=1)
    ax.text(
        bandwidth,
        float(np.max(decay_vals)) * 0.9,
        f" bandwidth={bandwidth:.3f}",
        rotation=90,
        va="top",
        ha="left",
        fontsize=9,
    )

    # Panel 2: normalized weights across axis
    ax = axes[1]
    for i in range(obs.size):
        ax.plot(axis_grid, weight_curves[:, i], linewidth=2, label=f"anchor {i} @ {obs[i]:.3f}")

    for i, pos in enumerate(obs):
        ax.axvline(float(pos), linestyle="--", linewidth=1)
        ax.scatter([pos], [0.0], s=60, zorder=5)
        ax.text(float(pos), -0.06, f"{pos:.3f}", ha="center", va="top", fontsize=9)

    ax.axvline(q, linestyle=":", linewidth=2)
    ax.scatter([q] * obs.size, wq, s=70, zorder=6)

    # annotate query weights
    txt_lines = [f"query={q:.3f}"]
    for i, wi in enumerate(wq):
        txt_lines.append(f"w{i}={wi:.3f}")
        ax.text(q, float(wi), f"  {wi:.3f}", va="center", ha="left", fontsize=9)

    ax.text(
        0.02,
        0.98,
        "\n".join(txt_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
    )

    ax.set_title("Normalized weights along biological axis")
    ax.set_xlabel("Biological axis position")
    ax.set_ylabel("Normalized weight")
    ax.set_ylim(-0.08, 1.05)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def compute_axis_kernel_weights(
    observed_axis_positions,
    query_axis_position,
    bandwidth=None,
    power=2.0,
    eps=1e-8,
):
    """
    Compute normalized source weights for any number of evidence slices.

    Parameters
    ----------
    observed_axis_positions : array-like, shape (S,)
        Biological-axis positions of S observed slices/evidence sources.
    query_axis_position : float
        Queried biological-axis position.
    bandwidth : float or None
        Kernel bandwidth. If None, inferred from observed positions.
    power : float
        Used only for inverse-distance weighting.
    eps : float
        Numerical stability for inverse-distance weighting.

    Returns
    -------
    weights : np.ndarray, shape (S,)
        Normalized weights summing to 1.
    bandwidth : float
        The bandwidth actually used.
    """
    obs = np.asarray(observed_axis_positions, dtype=np.float32).reshape(-1)
    query_axis_position = float(query_axis_position)

    if obs.size == 0:
        raise ValueError("observed_axis_positions must be non-empty.")

    if bandwidth is None:
        bandwidth = infer_axis_bandwidth(obs)
    bandwidth = max(float(bandwidth), eps)

    d = np.abs(obs - query_axis_position)

    #kernel == "gaussian":
    raw = np.exp(-0.5 * (d / bandwidth) ** 2)

    s = float(np.sum(raw))
    if s <= 0:
        raw = np.ones_like(raw, dtype=np.float32)
        s = float(np.sum(raw))

    weights = (raw / s).astype(np.float32, copy=False)
    return weights, float(bandwidth)


def matrix_log1p(X):
    """
    Safe log1p for dense or sparse matrices.
    """
    if sp.issparse(X):
        X = X.copy().tocsr()
        X.data = np.log1p(X.data).astype(np.float32, copy=False)
        return X
    return np.log1p(np.asarray(X, dtype=np.float32)).astype(np.float32, copy=False)


def stack_weighted_evidence(coords_blocks, X_blocks, source_weights, dtype=np.float32):
    """
    Concatenate multiple transported evidence blocks and expand source weights
    into per-point weights.

    Parameters
    ----------
    coords_blocks : list of (Ni, 2) arrays
    X_blocks : list of (Ni, G) matrices (dense or sparse)
    source_weights : array-like, shape (S,)
        One scalar weight per block/source.

    Returns
    -------
    coords_cat : (N, 2) float32
    X_cat : (N, G) dense or CSR matrix
    point_weights : (N,) float32
    """
    if len(coords_blocks) != len(X_blocks):
        raise ValueError("coords_blocks and X_blocks must have the same length.")

    source_weights = np.asarray(source_weights, dtype=np.float32).reshape(-1)
    if source_weights.shape[0] != len(coords_blocks):
        raise ValueError("source_weights length must match number of blocks.")

    coords_out = []
    point_weights_out = []
    x_is_sparse = any(sp.issparse(x) for x in X_blocks)

    for i, (coords_i, X_i) in enumerate(zip(coords_blocks, X_blocks)):
        coords_i = np.asarray(coords_i, dtype=dtype)
        if coords_i.ndim != 2 or coords_i.shape[1] != 2:
            raise ValueError(f"coords_blocks[{i}] must have shape (Ni, 2).")

        ni = coords_i.shape[0]
        if X_i.shape[0] != ni:
            raise ValueError(f"Row mismatch between coords_blocks[{i}] and X_blocks[{i}].")

        coords_out.append(coords_i)
        point_weights_out.append(np.full((ni,), source_weights[i], dtype=dtype))

    coords_cat = np.vstack(coords_out).astype(dtype, copy=False)
    point_weights = np.concatenate(point_weights_out, axis=0).astype(dtype, copy=False)

    if x_is_sparse:
        X_cat = sp.vstack([
            Xi if sp.issparse(Xi) else sp.csr_matrix(np.asarray(Xi, dtype=dtype))
            for Xi in X_blocks
        ]).tocsr()
    else:
        X_cat = np.vstack([np.asarray(Xi, dtype=dtype) for Xi in X_blocks]).astype(dtype, copy=False)

    return coords_cat, X_cat, point_weights


def infer_grid_spacing(grid_pts):
    """
    Infer approximate grid spacing from a fixed grid of centers.
    """
    grid_pts = np.asarray(grid_pts, dtype=np.float32)
    if grid_pts.ndim != 2 or grid_pts.shape[1] != 2:
        raise ValueError("grid_pts must have shape (M, 2).")

    ux = np.unique(np.round(grid_pts[:, 0], 6))
    uy = np.unique(np.round(grid_pts[:, 1], 6))

    dx = np.diff(np.sort(ux)) if ux.size > 1 else np.array([], dtype=np.float32)
    dy = np.diff(np.sort(uy)) if uy.size > 1 else np.array([], dtype=np.float32)

    vals = []
    if dx.size > 0:
        vals.append(np.median(dx))
    if dy.size > 0:
        vals.append(np.median(dy))

    if len(vals) == 0:
        return 1.0
    return float(np.median(vals))/2.0


import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from anndata import AnnData


def spatial_resample_evidence_regression_fixed_grid(
    coords,
    X,
    grid_pts,
    grid_support_mask=None,
    point_weights=None,
    spatial_bandwidth=None,
    knn=48,
    min_neighbors=6,
    radius=None,
    chunk_size=256,
    fill_value=None,
    dtype=np.float32,
    # --- new query-specific support controls ---
    enforce_query_support=True,
    max_support_dist=None,
    min_confidence_abs=None,
    min_confidence_ratio=0.05,
    confidence_ref="median",
):

    coords = np.asarray(coords, dtype=dtype)
    grid_pts = np.asarray(grid_pts, dtype=dtype)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2).")
    if grid_pts.ndim != 2 or grid_pts.shape[1] != 2:
        raise ValueError("grid_pts must have shape (M, 2).")
    if X.shape[0] != coords.shape[0]:
        raise ValueError("X rows must match coords rows.")

    n_points = coords.shape[0]
    n_grid = grid_pts.shape[0]
    n_genes = X.shape[1]

    if point_weights is None:
        point_weights = np.ones((n_points,), dtype=dtype)
    else:
        point_weights = np.asarray(point_weights, dtype=dtype).reshape(-1)
        if point_weights.shape[0] != n_points:
            raise ValueError("point_weights length must match coords rows.")

    if grid_support_mask is None:
        valid_mask = np.ones((n_grid,), dtype=bool)
    else:
        valid_mask = np.asarray(grid_support_mask, dtype=bool).reshape(-1)
        if valid_mask.shape[0] != n_grid:
            raise ValueError("grid_support_mask length must match grid_pts rows.")

    if spatial_bandwidth is None:
        spatial_bandwidth = infer_grid_spacing(grid_pts)
    spatial_bandwidth = max(float(spatial_bandwidth), 1e-6)
    print(f"spatial_bandwidth is {spatial_bandwidth}")



    if max_support_dist is None:
        max_support_dist = 1.5 * spatial_bandwidth
    max_support_dist = float(max_support_dist)

    Y = np.full((n_grid, n_genes), fill_value, dtype=dtype)
    confidence = np.zeros((n_grid,), dtype=dtype)
    nearest_dist = np.full((n_grid,), np.inf, dtype=dtype)
    active_query_support = np.zeros((n_grid,), dtype=bool)

    if n_points == 0 or not np.any(valid_mask):
        ad = AnnData(X=Y, dtype=Y.dtype)
        ad.obsm["spatial"] = grid_pts.astype(dtype, copy=False)
        ad.obs["valid"] = valid_mask
        ad.obs["active_query_support"] = active_query_support
        ad.obs["confidence"] = confidence
        ad.obs["nearest_dist"] = nearest_dist
        ad.uns["spatial_bandwidth"] = float(spatial_bandwidth)
        ad.uns["knn"] = int(knn)
        ad.uns["min_neighbors"] = int(min_neighbors)
        ad.uns["radius"] = None if radius is None else float(radius)
        ad.uns["max_support_dist"] = float(max_support_dist)
        ad.uns["min_confidence_abs"] = None if min_confidence_abs is None else float(min_confidence_abs)
        ad.uns["min_confidence_ratio"] = float(min_confidence_ratio)
        ad.uns["confidence_ref"] = str(confidence_ref)
        ad.uns["generator"] = "transition_guided_spatial_kernel_regression"
        return ad

    k = int(max(knn, min_neighbors))
    k = max(1, min(k, n_points))

    tree = cKDTree(coords)
    query_idx = np.where(valid_mask)[0]
    query_pts = grid_pts[query_idx]

    try:
        dists, nn_idx = tree.query(query_pts, k=k, workers=-1)
    except TypeError:
        dists, nn_idx = tree.query(query_pts, k=k)

    if k == 1:
        dists = dists[:, None]
        nn_idx = nn_idx[:, None]

    # nearest transported-evidence distance (for active support mask)
    nearest_dist_valid = dists[:, 0].astype(dtype, copy=False)
    nearest_dist[query_idx] = nearest_dist_valid

    # neighbor selection for regression
    neighbor_mask = np.ones_like(dists, dtype=bool)
    if radius is not None:
        neighbor_mask = dists <= float(radius)
        if min_neighbors > 0:
            neighbor_mask[:, :min(min_neighbors, k)] = True

    if np.any(~np.isfinite(dists)):
        neighbor_mask &= np.isfinite(dists)
        dists = np.where(np.isfinite(dists), dists, 1e9)

    # spatial kernel
    kernel_w = np.exp(-0.5 * (dists / spatial_bandwidth) ** 2).astype(dtype, copy=False)
    kernel_w *= neighbor_mask.astype(dtype, copy=False)

    # combine with per-point/source evidence weights
    local_source_w = point_weights[nn_idx]  # (Q, k)
    local_w_raw = kernel_w * local_source_w

    # local evidence mass BEFORE normalization
    confidence_valid = np.sum(local_w_raw, axis=1).astype(dtype, copy=False)
    confidence[query_idx] = confidence_valid

    # -------- query-specific active tissue mask --------
    if enforce_query_support:
        positive_conf = confidence_valid[confidence_valid > 0]

        if min_confidence_abs is not None:
            confidence_thresh = float(min_confidence_abs)
        else:
            if positive_conf.size == 0:
                conf_ref_value = 0.0
            else:
                if confidence_ref == "median":
                    conf_ref_value = float(np.median(positive_conf))
                elif confidence_ref == "mean":
                    conf_ref_value = float(np.mean(positive_conf))
                elif confidence_ref == "max":
                    conf_ref_value = float(np.max(positive_conf))
                elif confidence_ref == "p75":
                    conf_ref_value = float(np.percentile(positive_conf, 75))
                else:
                    raise ValueError("confidence_ref must be one of {'median','mean','max','p75'}")

            confidence_thresh = float(min_confidence_ratio) * conf_ref_value

        active_valid = (
            (nearest_dist_valid <= max_support_dist) &
            (confidence_valid >= confidence_thresh)
        )
    else:
        confidence_thresh = 0.0
        active_valid = np.ones_like(confidence_valid, dtype=bool)

    active_query_support[query_idx] = active_valid

    # only regress on active query support points
    active_rows_local = np.where(active_valid)[0]
    # normalize local weights ONLY for active rows
    local_w = local_w_raw[active_rows_local].copy()
    nn_idx_active = nn_idx[active_rows_local]
    denom = np.sum(local_w, axis=1).astype(dtype, copy=False)

    zero_rows = denom <= 0
    if np.any(zero_rows):
        fallback_sw = local_source_w[active_rows_local][zero_rows, 0]
        local_w[zero_rows, 0] = np.maximum(fallback_sw, 1e-8)
        denom = np.sum(local_w, axis=1).astype(dtype, copy=False)

    local_w = local_w / np.maximum(denom[:, None], 1e-8)

    is_sparse = sp.issparse(X)
    if is_sparse:
        X = X.tocsr()

    query_idx_active = query_idx[active_rows_local]

    for start in range(0, query_idx_active.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), query_idx_active.shape[0])

        q_rows = query_idx_active[start:end]
        idx_block = nn_idx_active[start:end]   # (B, k)
        w_block = local_w[start:end]           # (B, k)
        bsz = idx_block.shape[0]

        flat_idx = idx_block.reshape(-1)
        flat_w = w_block.reshape(-1)

        if is_sparse:
            X_sel = X[flat_idx]  # (B*k, G)
            row_ids = np.repeat(np.arange(bsz), k)
            col_ids = np.arange(bsz * k)
            W_sparse = sp.csr_matrix(
                (flat_w, (row_ids, col_ids)),
                shape=(bsz, bsz * k),
                dtype=dtype,
            )
            Y_block = (W_sparse @ X_sel).astype(dtype)
            if sp.issparse(Y_block):
                Y_block = Y_block.toarray()
        else:
            X_sel = np.asarray(X[flat_idx], dtype=dtype).reshape(bsz, k, n_genes)
            Y_block = np.einsum("bk,bkg->bg", w_block, X_sel, optimize=True).astype(dtype, copy=False)

        Y[q_rows] = Y_block

    ad = AnnData(X=Y, dtype=Y.dtype)
    ad.obsm["spatial"] = grid_pts.astype(dtype, copy=False)
    ad.obs["valid"] = valid_mask
    ad.obs["active_query_support"] = active_query_support
    ad.obs["confidence"] = confidence
    ad.obs["nearest_dist"] = nearest_dist
    ad.uns["spatial_bandwidth"] = float(spatial_bandwidth)
    ad.uns["knn"] = int(knn)
    ad.uns["min_neighbors"] = int(min_neighbors)
    ad.uns["radius"] = None if radius is None else float(radius)
    ad.uns["max_support_dist"] = float(max_support_dist)
    ad.uns["min_confidence_abs"] = None if min_confidence_abs is None else float(min_confidence_abs)
    ad.uns["min_confidence_ratio"] = float(min_confidence_ratio)
    ad.uns["confidence_ref"] = str(confidence_ref)
    ad.uns["confidence_threshold_used"] = float(confidence_thresh)
    ad.uns["generator"] = "transition_guided_spatial_kernel_regression"
    return ad



# =========================
# load data
# =========================

def load_coords_npz(NET_IN, group_id, sample_id):
    path = os.path.join(NET_IN, "{}_{}_validation.npz".format(group_id, sample_id))
    d = np.load(path)
    return d["coord"].astype(np.float32)

def load_axis_flow(AXIS_OUT, fixed_id, moving_id):
    path = os.path.join(AXIS_OUT, "{}_{}_flow.npy".format(fixed_id, moving_id))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)

def load_h5ad_X_and_genes(H5AD_DIR, group_id, sample_id):
    path = os.path.join(H5AD_DIR, "{}_{}.h5ad".format(group_id, sample_id))
    ad = sc.read_h5ad(path)
    return ad.X, np.asarray(ad.var_names)

def load_h5ad_X_and_genes_and_coords(H5AD_DIR, group_id, sample_id):
    path = os.path.join(H5AD_DIR, "{}_{}.h5ad".format(group_id, sample_id))
    ad = sc.read_h5ad(path)

    return ad.X, np.asarray(ad.var_names),np.asarray(ad.obsm['spatial_axis'])

def ensure_same_genes_keep_order(Xa, ga, Xb, gb):
    """
    Intersect genes, KEEP slice0 order, subset both matrices.
    """
    ga = np.asarray(ga)
    gb = np.asarray(gb)
    gb_set = set(gb.tolist())
    common = [g for g in ga.tolist() if g in gb_set]
    if len(common) == 0:
        raise ValueError("No common genes between anchors.")
    idx_a = np.array([int(np.where(ga == g)[0][0]) for g in common], dtype=np.int64)
    idx_b = np.array([int(np.where(gb == g)[0][0]) for g in common], dtype=np.int64)
    return Xa[:, idx_a], Xb[:, idx_b], np.asarray(common)



# ==================================================
# make consistent grid for virtual tissue generation
# ==================================================
def compute_dense_tissue_mask_on_grid(
    grid_pts, union_spots,
    grid_size,
    max_dist_px=9.0,
    closing_iters=2,
    keep_largest=True,
):
    """
    Returns mask (M,) bool on grid cells: a solid tissue region with holes filled.
    """
    # 1) initial distance-threshold mask
    tree = cKDTree(np.asarray(union_spots, dtype=np.float32))
    dist, _ = tree.query(np.asarray(grid_pts, dtype=np.float32), k=1)
    mask = (dist <= float(max_dist_px)).reshape(grid_size, grid_size)

    # 2) connect gaps (closing) + 3) fill holes
    if closing_iters > 0:
        mask = binary_closing(mask, iterations=int(closing_iters))
    mask = binary_fill_holes(mask)

    # 4) optionally keep only the largest connected component
    if keep_largest:
        lab, n = label(mask)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0  # background
            keep = sizes.argmax()
            mask = (lab == keep)

    return mask.reshape(-1)

def make_common_grid_from_coords(coords, grid_size, pad_ratio=0.02):
    """
    coords: (N,2)
    pad_ratio: small padding to avoid clipping boundary spots
    Returns: grid_pts (M,2), (x_edges, y_edges)
    """
    coords = np.asarray(coords, dtype=np.float32)
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)

    dx = (max_x - min_x) * pad_ratio
    dy = (max_y - min_y) * pad_ratio
    min_x -= dx; max_x += dx
    min_y -= dy; max_y += dy

    x_edges = np.linspace(min_x, max_x, grid_size + 1, dtype=np.float32)
    y_edges = np.linspace(min_y, max_y, grid_size + 1, dtype=np.float32)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    gx, gy = np.meshgrid(x_centers, y_centers)
    grid_pts = np.vstack([gx.ravel(), gy.ravel()]).T.astype(np.float32)
    return grid_pts, (x_edges, y_edges)




# ==================================================
# spatial resampling of gene data
# ==================================================
def spatial_resample_nmf_fixed_grid(
    coords, X, grid_pts, x_edges, y_edges,
    n_components=20,
    fill_empty=True,
    grid_support_mask=None,   # (M,) bool, optional
    nmf_init="nndsvda",
    nmf_max_iter=800,
    nmf_random_state=0,
):
    """
    coords: (N,2) projected points for this t
    X: (N,G) expression for those points (sparse or dense)
    grid_pts: (M,2) fixed grid centers
    x_edges/y_edges: fixed bin edges
    """
    coords = np.asarray(coords, dtype=np.float32)

    # infer grid_size from edges; M must match
    grid_size = int(len(x_edges) - 1)
    M = int(grid_size * grid_size)
    if grid_pts.shape[0] != M:
        raise ValueError("grid_pts length mismatch: {} vs {}".format(grid_pts.shape[0], M))

    # densify (same as your original function behavior)
    if sp.issparse(X):
        Xd = X.toarray().astype(np.float32)
    else:
        Xd = np.asarray(X, dtype=np.float32)

    # bin points into fixed grid
    x_bin = np.digitize(coords[:, 0], x_edges) - 1
    y_bin = np.digitize(coords[:, 1], y_edges) - 1
    x_bin = np.clip(x_bin, 0, grid_size - 1)
    y_bin = np.clip(y_bin, 0, grid_size - 1)
    cell_id = (y_bin.astype(np.int64) * grid_size + x_bin.astype(np.int64))

    flat_grid = np.zeros((M, Xd.shape[1]), dtype=np.float32)
    counts = np.zeros((M,), dtype=np.float32)
    for i in range(coords.shape[0]):
        cid = int(cell_id[i])
        flat_grid[cid] += Xd[i]
        counts[cid] += 1.0

    # cells hit by at least one point
    valid_by_points = counts > 0

    # optional: only consider a support region (recommended)
    if grid_support_mask is not None:
        grid_support_mask = np.asarray(grid_support_mask, dtype=bool)
        if grid_support_mask.shape[0] != M:
            raise ValueError("grid_support_mask length mismatch")
        fit_mask = valid_by_points & grid_support_mask
    else:
        fit_mask = valid_by_points

    # optional fill (you asked to disable this for interpolation)
    if fill_empty:
        # fill ONLY inside support (if provided); never fill background
        if grid_support_mask is None:
            fill_region = np.ones((M,), dtype=bool)
        else:
            fill_region = grid_support_mask.copy()

        counts2 = counts.reshape(grid_size, grid_size)
        fill_region2 = fill_region.reshape(grid_size, grid_size)

        non_empty_idx = np.argwhere((counts2 > 0) & fill_region2)
        empty_idx = np.argwhere((counts2 == 0) & fill_region2)

        if empty_idx.shape[0] > 0 and non_empty_idx.shape[0] > 0:
            tree = cKDTree(non_empty_idx)
            _, nn = tree.query(empty_idx, k=1)
            base = flat_grid.reshape(grid_size, grid_size, -1)
            out = base.copy()
            for k, (r, c) in enumerate(empty_idx):
                rr, cc = non_empty_idx[nn[k]]
                out[r, c, :] = base[rr, cc, :]
            flat_grid = out.reshape(M, -1).astype(np.float32, copy=False)

    # ---- NMF: fit on supported rows only, then write back (much better than fitting on all M) ----
    # If nothing to fit (rare), return zeros
    smoothed = np.zeros_like(flat_grid, dtype=np.float32)
    if np.any(fit_mask):
        X_fit = flat_grid[fit_mask]
        model = NMF(
            n_components=n_components,
            solver="mu",
            beta_loss="kullback-leibler",
            init=nmf_init,
            random_state=nmf_random_state,
            max_iter=nmf_max_iter,
            tol=1e-4,
            verbose=False,
        )
        W = model.fit_transform(X_fit)
        H = model.components_.astype(np.float32, copy=False)
        smoothed_fit = (W @ H).astype(np.float32, copy=False)
        smoothed[fit_mask] = smoothed_fit

    ad = AnnData(X=smoothed, dtype=smoothed.dtype)
    ad.obsm["spatial"] = np.asarray(grid_pts, dtype=np.float32)
    ad.obs["valid"] = fit_mask.astype(bool)
    return ad


def spatial_resample_wasserstein_barycenter_fixed_grid(
    coords0, X0, coords3, X3,
    grid_pts, x_edges, y_edges,
    t,
    grid_support_mask=None,          # (M,) bool or None
    reg=0.01,                        # entropic regularization (bigger = smoother/faster)
    n_iter=200,                      # Sinkhorn barycenter iterations inside POT
    tol=1e-7,
    chunk_genes=256,                 # process genes in chunks to control RAM
    eps=1e-12,
):
    """
    Wasserstein barycenter generation on a fixed 2D grid.

    - Bins slice0-at-t and slice3-at-t to 2D grids per gene.
    - For each gene: computes 2D entropic Wasserstein barycenter between the two histograms.
    - Returns AnnData with X=(M,G) on grid_pts.

    REQUIREMENT: POT installed -> pip install pot
    """
    try:
        import ot
        from ot.bregman import convolutional_barycenter2d
    except Exception as e:
        raise ImportError(
            "POT is required for Wasserstein barycenter. Install with: pip install pot"
        ) from e

    coords0 = np.asarray(coords0, dtype=np.float32)
    coords3 = np.asarray(coords3, dtype=np.float32)

    # infer grid
    grid_size = int(len(x_edges) - 1)
    M = grid_size * grid_size
    if grid_pts.shape[0] != M:
        raise ValueError("grid_pts length mismatch")

    # densify X for arithmetic (you can optimize later if needed)
    if sp.issparse(X0):
        X0d = X0.toarray().astype(np.float32)
    else:
        X0d = np.asarray(X0, dtype=np.float32)
    if sp.issparse(X3):
        X3d = X3.toarray().astype(np.float32)
    else:
        X3d = np.asarray(X3, dtype=np.float32)

    G = X0d.shape[1]
    if X3d.shape[1] != G:
        raise ValueError("X0 and X3 gene dimension mismatch")

    # --- helper: bin points to grid (weighted mean per cell; here uniform weights) ---
    def _bin_to_grid(coords, Xd):
        x_bin = np.digitize(coords[:, 0], x_edges) - 1
        y_bin = np.digitize(coords[:, 1], y_edges) - 1
        x_bin = np.clip(x_bin, 0, grid_size - 1)
        y_bin = np.clip(y_bin, 0, grid_size - 1)
        cell_id = (y_bin.astype(np.int64) * grid_size + x_bin.astype(np.int64))

        grid_sum = np.zeros((M, G), dtype=np.float32)
        w_sum = np.zeros((M,), dtype=np.float32)

        for i in range(coords.shape[0]):
            cid = int(cell_id[i])
            grid_sum[cid] += Xd[i]
            w_sum[cid] += 1.0

        valid = w_sum > 0
        grid = np.zeros_like(grid_sum, dtype=np.float32)
        grid[valid] = grid_sum[valid] / (w_sum[valid, None] + eps)
        return grid

    grid0 = _bin_to_grid(coords0, X0d)  # (M,G)
    grid3 = _bin_to_grid(coords3, X3d)  # (M,G)

    # optional: restrict to tissue support
    if grid_support_mask is not None:
        mask = np.asarray(grid_support_mask, dtype=bool)
        if mask.shape[0] != M:
            raise ValueError("grid_support_mask length mismatch")
        grid0 = grid0 * mask[:, None]
        grid3 = grid3 * mask[:, None]

    # reshape to 2D images for convolutional barycenter
    grid0_2d = grid0.reshape(grid_size, grid_size, G)
    grid3_2d = grid3.reshape(grid_size, grid_size, G)

    w = np.array([1.0 - float(t), float(t)], dtype=np.float64)

    out = np.zeros((grid_size, grid_size, G), dtype=np.float32)

    # process genes in chunks (still O(G) barycenters; expensive but correct)
    for s in range(0, G, int(chunk_genes)):
        e = min(G, s + int(chunk_genes))

        for j in range(s, e):
            A0 = grid0_2d[:, :, j].astype(np.float64, copy=False)
            A3 = grid3_2d[:, :, j].astype(np.float64, copy=False)

            m0 = float(A0.sum())
            m3 = float(A3.sum())
            mt = (1.0 - float(t)) * m0 + float(t) * m3

            if mt <= 0.0:
                continue  # stays zero

            # balanced OT barycenter expects probability measures
            P0 = A0 / (m0 + eps) if m0 > 0 else np.full_like(A0, 1.0 / (grid_size * grid_size))
            P3 = A3 / (m3 + eps) if m3 > 0 else np.full_like(A3, 1.0 / (grid_size * grid_size))

            A = np.stack([P0, P3], axis=0)  # (2, H, W)

            # 2D entropic Wasserstein barycenter (convolutional Sinkhorn)
            B = convolutional_barycenter2d(
                A, reg=reg, weights=w, numItermax=n_iter, stopThr=tol
            )

            # rescale back to interpolated total mass
            out[:, :, j] = (B * mt).astype(np.float32, copy=False)

    X_out = out.reshape(M, G).astype(np.float32, copy=False)
    ad = AnnData(X=X_out, dtype=X_out.dtype)
    ad.obsm["spatial"] = np.asarray(grid_pts, dtype=np.float32)
    if grid_support_mask is not None:
        ad.obs["valid"] = np.asarray(grid_support_mask, dtype=bool)
    else:
        ad.obs["valid"] = np.ones((M,), dtype=bool)
    ad.uns["t"] = float(t)
    ad.uns["ot"] = {"reg": float(reg), "n_iter": int(n_iter), "tol": float(tol)}
    return ad
