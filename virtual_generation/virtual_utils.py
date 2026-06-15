import os
import scanpy as sc
from scipy.ndimage import binary_closing, binary_fill_holes, label
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from anndata import AnnData
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF
from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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
        bandwidth = infer_axis_bandwidth(obs)

    # -------- left panel: raw kernel decay K(d) --------
    if max_decay_dist is None:
        max_decay_dist = max(2.5 * bandwidth, 1.2 * np.max(np.abs(obs - q)), 1e-3)
    dist_grid = np.linspace(0.0, float(max_decay_dist), int(n_decay), dtype=np.float32)
    decay_vals = _axis_kernel_raw_from_distance(
        d=dist_grid,
        kernel=kernel,
        bandwidth=bandwidth,
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
            # kernel=kernel,
            bandwidth=bandwidth,

            eps=eps,
        )
        weight_curves.append(w)
    weight_curves = np.asarray(weight_curves, dtype=np.float32)  # (n_axis, n_obs)

    wq, _ = compute_axis_kernel_weights(
        observed_axis_positions=obs,
        query_axis_position=q,
        # kernel=kernel,
        bandwidth=bandwidth,

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
    print(f"biological bandwidth is {bandwidth}")

    d = np.abs(obs - query_axis_position)


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
    return float(np.median(vals))/2






# =========================
# load data
# =========================

def load_coords_npz(NET_IN, group_id, sample_id):
    path = os.path.join(NET_IN, "{}_{}_validation.npz".format(group_id, sample_id))
    d = np.load(path)
    return d["coord"].astype(np.float32)

def load_axis_flow(AXIS_OUT, fixed_id, moving_id,basesize=512):
    if fixed_id == moving_id:
        return np.zeros((2,basesize,basesize))
    else:
        path = os.path.join(AXIS_OUT, "{}_{}_flow.npy".format(fixed_id, moving_id))
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return np.load(path).astype(np.float32)

def load_h5ad_X_and_genes(H5AD_DIR, group_id, sample_id, attr_columns=None):
    path = os.path.join(H5AD_DIR, "{}_{}.h5ad".format(group_id, sample_id))
    ad = sc.read_h5ad(path)

    attrs = None
    if attr_columns is not None and len(attr_columns) > 0:
        missing = [c for c in attr_columns if c not in ad.obs.columns]
        if len(missing) > 0:
            raise KeyError(f"Missing obs columns in {path}: {missing}")
        attrs = ad.obs.loc[:, attr_columns].copy().reset_index(drop=True)

    return ad.X, np.asarray(ad.var_names), attrs

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

def stack_attribute_evidence(attrs_a, attrs_b):
    if attrs_a is None or attrs_b is None:
        return None
    return pd.concat([attrs_a, attrs_b], axis=0, ignore_index=True)

# ==================================================
# make consistent grid for virtual tissue generation
# ==================================================

def estimate_grid_spacing(grid_pts: np.ndarray) -> float:
    """
    Estimate the local spacing of the generated grid from nearest-neighbor distances.
    Works for both square and hex-like grids.
    """
    grid_pts = np.asarray(grid_pts, dtype=np.float32)
    if grid_pts.ndim != 2 or grid_pts.shape[1] != 2:
        raise ValueError(f"grid_pts must have shape (N,2), got {grid_pts.shape}")
    if grid_pts.shape[0] < 2:
        raise ValueError("Need at least 2 grid points to estimate spacing")

    tree = cKDTree(grid_pts)
    dists, _ = tree.query(grid_pts, k=2)   # self + nearest neighbor
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        raise ValueError("Could not estimate grid spacing")
    return float(np.median(nn))


def keep_largest_connected_component_on_grid(
    grid_pts: np.ndarray,
    mask: np.ndarray,
    neighbor_radius: float,
) -> np.ndarray:
    """
    Keep only the largest connected component among masked grid points,
    using a radius-based graph on grid_pts.
    """
    grid_pts = np.asarray(grid_pts, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    idx = np.where(mask)[0]
    if idx.size == 0:
        return mask
    if idx.size == 1:
        return mask

    pts = grid_pts[idx]
    tree = cKDTree(pts)
    neighbors = tree.query_ball_tree(tree, r=float(neighbor_radius))

    rows = []
    cols = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if i != j:
                rows.append(i)
                cols.append(j)

    if len(rows) == 0:
        # nothing connected, keep all masked points as-is
        return mask

    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(idx), len(idx)),
    )

    n_comp, labels = connected_components(graph, directed=False)
    if n_comp <= 1:
        return mask

    sizes = np.bincount(labels)
    keep_label = np.argmax(sizes)

    new_mask = np.zeros_like(mask, dtype=bool)
    new_mask[idx[labels == keep_label]] = True
    return new_mask


def smooth_mask_on_grid(
    grid_pts: np.ndarray,
    mask: np.ndarray,
    neighbor_radius: float,
    min_keep_neighbors: int = 2,
) -> np.ndarray:
    """
    Simple graph-based smoothing:
    keep a point if enough of its nearby grid neighbors are also active.

    This is a lightweight replacement for binary closing/opening that works
    on both square and hex grids.
    """
    grid_pts = np.asarray(grid_pts, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    tree = cKDTree(grid_pts)
    neighbors = tree.query_ball_point(grid_pts, r=float(neighbor_radius))

    new_mask = np.zeros_like(mask, dtype=bool)
    for i, nbrs in enumerate(neighbors):
        active = np.sum(mask[nbrs])
        new_mask[i] = active >= int(min_keep_neighbors)

    return new_mask


def compute_dense_tissue_mask_on_any_grid(
    grid_pts,
    union_spots,
    max_dist_px=9.0,
    smooth=True,
    smooth_radius_factor=1.25,
    min_keep_neighbors=2,
    keep_largest=True,
):
    """
    Support estimation that works for both square pixel grids and Visium-like hex grids.

    Parameters
    ----------
    grid_pts : (M,2)
        Generated common grid points.
    union_spots : (N,2)
        Union of spots from the slices.
    max_dist_px : float
        Initial nearest-spot threshold.
    smooth : bool
        Whether to apply simple graph-based smoothing.
    smooth_radius_factor : float
        Neighbor radius relative to grid spacing for smoothing and connectivity.
    min_keep_neighbors : int
        Minimum active neighbors for smoothing.
    keep_largest : bool
        Whether to keep only the largest connected supported region.

    Returns
    -------
    mask : (M,) bool
        Support mask directly aligned with grid_pts.
    """
    grid_pts = np.asarray(grid_pts, dtype=np.float32)
    union_spots = np.asarray(union_spots, dtype=np.float32)

    if grid_pts.ndim != 2 or grid_pts.shape[1] != 2:
        raise ValueError(f"grid_pts must have shape (M,2), got {grid_pts.shape}")
    if union_spots.ndim != 2 or union_spots.shape[1] != 2:
        raise ValueError(f"union_spots must have shape (N,2), got {union_spots.shape}")

    # 1) initial distance-threshold mask
    tree = cKDTree(union_spots)
    dist, _ = tree.query(grid_pts, k=1)
    mask = dist <= float(max_dist_px)

    # 2) graph-based smoothing (works on any grid layout)
    spacing = estimate_grid_spacing(grid_pts)
    neighbor_radius = float(smooth_radius_factor) * spacing

    if smooth:
        mask = smooth_mask_on_grid(
            grid_pts=grid_pts,
            mask=mask,
            neighbor_radius=neighbor_radius,
            min_keep_neighbors=min_keep_neighbors,
        )

    # 3) keep largest connected component on the grid graph
    if keep_largest:
        mask = keep_largest_connected_component_on_grid(
            grid_pts=grid_pts,
            mask=mask,
            neighbor_radius=neighbor_radius,
        )

    return mask







def make_common_grid_from_coords(grid_size, pad_ratio=0.00,type='pixel',max_x=512,max_y=512,min_x=0,min_y=0):
    """
    coords: (N,2)
    pad_ratio: small padding to avoid clipping boundary spots
    Returns: grid_pts (M,2), (x_edges, y_edges)
    """
    if type == 'pixel':
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
    elif type == 'visium':
        # small padding
        dx = (max_x - min_x) * pad_ratio
        dy = (max_y - min_y) * pad_ratio
        min_x -= dx
        max_x += dx
        min_y -= dy
        max_y += dy


        width = max_x - min_x
        height = max_y - min_y

        # target_x - min_x
        height = max_y - min_y
        area = width * height

        # target about grid_size^2 hex centers over the area
        target_n = max(int(grid_size) ** 2, 1)

        # For a staggered hex lattice:
        # horizontal spacing = sx
        # vertical spacing   = sy = sqrt(3)/2 * sx
        # density ~ 1 / (sx * sy) = 2 / (sqrt(3) * sx^2)
        sx = np.sqrt((2.0 * area) / (np.sqrt(3.0) * target_n))
        sy = (np.sqrt(3.0) / 2.0) * sx

        # generate staggered rows
        y_vals = np.arange(min_y, max_y + sy, sy, dtype=np.float32)

        pts = []
        for row_idx, y in enumerate(y_vals):
            # alternate row offset by half-step to form hex arrangement
            x_offset = 0.5 * sx if (row_idx % 2 == 1) else 0.0
            x_vals = np.arange(min_x + x_offset, max_x + sx, sx, dtype=np.float32)

            # keep only points inside padded bbox
            keep = (x_vals >= min_x) & (x_vals <= max_x)
            x_vals = x_vals[keep]

            if x_vals.size > 0:
                row_pts = np.column_stack([x_vals, np.full_like(x_vals, y)])
                pts.append(row_pts)

        if len(pts) == 0:
            grid_pts = np.empty((0, 2), dtype=np.float32)
        else:
            grid_pts = np.vstack(pts).astype(np.float32)
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



# ============================================================
# Gene-set helpers
# ============================================================
def _row_sums_dense_or_sparse(X):
    if sp.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    return np.asarray(X.sum(axis=1), dtype=np.float32).ravel()


def _row_normalize_dense(A, eps=1e-8):
    rs = np.sum(A, axis=1, keepdims=True)
    return A / np.maximum(rs, eps)


def _extract_dense_rows(X, row_idx, dtype=np.float32):
    if sp.issparse(X):
        return np.asarray(X[row_idx].toarray(), dtype=dtype)
    return np.asarray(X[row_idx], dtype=dtype)


def fit_latent_gene_program_model(
    X,
    n_programs=32,
    max_fit_cells=50000,
    random_state=0,
    max_iter=400,
    eps=1e-8,
    dtype=np.float32,
):
    """
    Fit once, reuse for all virtual query slices.

    The model is fit on row-normalized compositions.
    """
    n = X.shape[0]
    if n > max_fit_cells:
        rng = np.random.default_rng(random_state)
        keep = np.sort(rng.choice(n, size=max_fit_cells, replace=False))
        X_fit = X[keep]
    else:
        X_fit = X

    X_fit = _extract_dense_rows(X_fit, np.arange(X_fit.shape[0]), dtype=dtype)
    X_fit = np.maximum(X_fit, 0.0)
    X_fit = _row_normalize_dense(X_fit, eps=eps)

    nmf = NMF(
        n_components=n_programs,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        random_state=random_state,
        max_iter=max_iter,
        tol=1e-4,
    )
    W = nmf.fit_transform(X_fit).astype(dtype, copy=False)
    H = nmf.components_.astype(dtype, copy=False)

    # convert gene programs to simplex form
    H_topic = H / np.maximum(H.sum(axis=1, keepdims=True), eps)

    return {
        "nmf_model": nmf,
        "H_topic": H_topic.astype(dtype, copy=False),
        "n_programs": int(n_programs),
        "eps": float(eps),
    }


def _weighted_geometric_barycenter(P, W, eps=1e-8):
    """
    P: (B, k, D) simplex-like vectors
    W: (B, k) normalized local weights
    Returns: (B, D), normalized weighted geometric mean
    """
    logP = np.log(np.maximum(P, eps))
    Z = np.exp(np.einsum("bk,bkd->bd", W, logP, optimize=True)).astype(np.float32, copy=False)
    Z = Z / np.maximum(Z.sum(axis=1, keepdims=True), eps)
    return Z


def _prepare_local_query_support(
    coords,
    grid_pts,
    grid_support_mask=None,
    point_weights=None,
    qc_total_counts=None,
    qc_n_genes=None,
    spatial_bandwidth=None,
    knn=48,
    min_neighbors=6,
    radius=None,
    dtype=np.float32,
):
    coords = np.asarray(coords, dtype=dtype)
    grid_pts = np.asarray(grid_pts, dtype=dtype)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2).")
    if grid_pts.ndim != 2 or grid_pts.shape[1] != 2:
        raise ValueError("grid_pts must have shape (M, 2).")

    n_points, n_grid = coords.shape[0], grid_pts.shape[0]

    point_weights = np.ones(n_points, dtype=dtype) if point_weights is None else np.asarray(point_weights, dtype=dtype).reshape(-1)
    if point_weights.shape[0] != n_points:
        raise ValueError("point_weights length must match coords rows.")

    valid_mask = np.ones(n_grid, dtype=bool) if grid_support_mask is None else np.asarray(grid_support_mask, dtype=bool).reshape(-1)
    if valid_mask.shape[0] != n_grid:
        raise ValueError("grid_support_mask length must match grid_pts rows.")

    if spatial_bandwidth is None:
        spatial_bandwidth = infer_grid_spacing(grid_pts)
    spatial_bandwidth = max(float(spatial_bandwidth), 1e-6)
    print(f'spatial bandwidth is {spatial_bandwidth}')


    confidence = np.zeros(n_grid, dtype=dtype)
    nearest_dist = np.full(n_grid, np.inf, dtype=dtype)
    active_query_support = np.zeros(n_grid, dtype=bool)

    k = max(1, min(int(max(knn, min_neighbors)), n_points))
    query_idx = np.where(valid_mask)[0]
    query_pts = grid_pts[query_idx]

    tree = cKDTree(coords)
    try:
        dists, nn_idx = tree.query(query_pts, k=k, workers=-1)
    except TypeError:
        dists, nn_idx = tree.query(query_pts, k=k)

    if k == 1:
        dists = dists[:, None]
        nn_idx = nn_idx[:, None]

    dists = np.where(np.isfinite(dists), dists, 1e9)
    nearest_dist[query_idx] = dists[:, 0].astype(dtype, copy=False)

    neighbor_mask = np.ones_like(dists, dtype=bool)
    if radius is not None:
        neighbor_mask = dists <= float(radius)
        neighbor_mask[:, :min(min_neighbors, k)] = True

    kernel_w = np.exp(-0.5 * (dists / spatial_bandwidth) ** 2).astype(dtype, copy=False)
    kernel_w *= neighbor_mask.astype(dtype, copy=False)

    local_w_raw = kernel_w * point_weights[nn_idx]

    if qc_total_counts is not None and qc_n_genes is not None:
        q_rel = compute_local_relative_quality_weights(
            qc_total_local=np.asarray(qc_total_counts, dtype=dtype)[nn_idx],
            qc_ngenes_local=np.asarray(qc_n_genes, dtype=dtype)[nn_idx],
            gamma_counts=1.0,
            gamma_genes=0.7,
            rel_clip=(0.5, 2.0),
        )
        local_w_raw *= q_rel

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
        denom = np.sum(local_w, axis=1).astype(dtype, copy=False)

    local_w /= np.maximum(denom[:, None], 1e-8)

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


def _prepare_local_query_support_attr(
    coords,
    grid_pts,
    grid_support_mask=None,
    point_weights=None,
    spatial_bandwidth=None,
    knn=48,
    min_neighbors=6,
    radius=None,
    dtype=np.float32,
):
    coords = np.asarray(coords, dtype=dtype)
    grid_pts = np.asarray(grid_pts, dtype=dtype)
    n_points, n_grid = coords.shape[0], grid_pts.shape[0]

    point_weights = np.ones(n_points, dtype=dtype) if point_weights is None else np.asarray(point_weights, dtype=dtype).reshape(-1)
    valid_mask = np.ones(n_grid, dtype=bool) if grid_support_mask is None else np.asarray(grid_support_mask, dtype=bool).reshape(-1)
    spatial_bandwidth = max(float(spatial_bandwidth), 1e-6)

    confidence = np.zeros(n_grid, dtype=dtype)
    nearest_dist = np.full(n_grid, np.inf, dtype=dtype)
    active_query_support = np.zeros(n_grid, dtype=bool)

    k = max(1, min(int(max(knn, min_neighbors)), n_points))
    query_idx = np.where(valid_mask)[0]
    query_pts = grid_pts[query_idx]

    tree = cKDTree(coords)
    try:
        dists, nn_idx = tree.query(query_pts, k=k, workers=-1)
    except TypeError:
        dists, nn_idx = tree.query(query_pts, k=k)

    if k == 1:
        dists = dists[:, None]
        nn_idx = nn_idx[:, None]

    dists = np.where(np.isfinite(dists), dists, 1e9)
    nearest_dist[query_idx] = dists[:, 0].astype(dtype, copy=False)

    neighbor_mask = np.ones_like(dists, dtype=bool)
    if radius is not None:
        neighbor_mask = dists <= float(radius)
        neighbor_mask[:, :min(min_neighbors, k)] = True

    kernel_w = np.exp(-0.5 * (dists / spatial_bandwidth) ** 2).astype(dtype, copy=False)
    kernel_w *= neighbor_mask.astype(dtype, copy=False)

    local_w_raw = kernel_w * point_weights[nn_idx]
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
        denom = np.sum(local_w, axis=1).astype(dtype, copy=False)

    local_w /= np.maximum(denom[:, None], 1e-8)

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



def infer_attr_schema_from_df(df):
    schema = {}
    for col in df.columns:
        s = df[col]
        name = str(col).lower()

        if is_bool_dtype(s):
            schema[col] = "boolean"
        elif is_categorical_dtype(s) or is_object_dtype(s) or is_string_dtype(s):
            schema[col] = "categorical"
        elif is_numeric_dtype(s):
            # common label-like names should still be categorical
            if any(k in name for k in ["label", "layer", "cluster", "domain", "celltype", "type", "guess", "region"]):
                schema[col] = "categorical"
            else:
                schema[col] = "continuous"
        else:
            schema[col] = "categorical"
    return schema





def compute_qc_covariates(X):
    if sp.issparse(X):
        total_counts = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.float32)
    else:
        Xd = np.asarray(X, dtype=np.float32)
        total_counts = Xd.sum(axis=1).astype(np.float32)
        n_genes = (Xd > 0).sum(axis=1).astype(np.float32)
    return total_counts, n_genes

def compute_local_relative_quality_weights(
    qc_total_local,
    qc_ngenes_local=None,
    gamma_counts=0.5,
    gamma_genes=0.5,
    rel_clip=(0.5, 2.0),
    eps=1e-8,
):
    """
    qc_total_local: (Q, k)
    qc_ngenes_local: (Q, k) or None

    Returns
    -------
    q_rel : (Q, k)
        Local relative quality factors for each neighborhood.
    """
    qc_total_local = np.asarray(qc_total_local, dtype=np.float32)
    q_total = np.log1p(qc_total_local)

    med_total = np.median(q_total, axis=1, keepdims=True)
    rel_total = q_total / np.maximum(med_total, eps)
    rel_total = np.clip(rel_total, rel_clip[0], rel_clip[1]) ** gamma_counts

    if qc_ngenes_local is None:
        return rel_total.astype(np.float32, copy=False)

    qc_ngenes_local = np.asarray(qc_ngenes_local, dtype=np.float32)
    q_ngenes = np.log1p(qc_ngenes_local)

    med_ngenes = np.median(q_ngenes, axis=1, keepdims=True)
    rel_ngenes = q_ngenes / np.maximum(med_ngenes, eps)
    rel_ngenes = np.clip(rel_ngenes, rel_clip[0], rel_clip[1]) ** gamma_genes

    q_rel = rel_total * rel_ngenes
    return q_rel.astype(np.float32, copy=False)


def generate_gene_sets(grid_pts,X,support,fill_value,chunk_size,dtype):
    n_grid = grid_pts.shape[0]
    n_genes = X.shape[1]
    query_idx_active = support["query_idx_active"]
    nn_idx_active = support["nn_idx_active"]
    local_w = support["local_w"]
    Y = np.full((n_grid, n_genes), fill_value, dtype=dtype)
    if query_idx_active.size > 0:
        for start in range(0, query_idx_active.shape[0], int(chunk_size)):
            end = min(start + int(chunk_size), query_idx_active.shape[0])

            q_rows = query_idx_active[start:end]
            idx_block = nn_idx_active[start:end]  # (B, k)
            w_block = local_w[start:end]  # (B, k)
            B, k = idx_block.shape

            flat_idx = idx_block.reshape(-1)
            X_block = _extract_dense_rows(X, flat_idx, dtype=dtype).reshape(B, k, n_genes)
            X_block = np.maximum(X_block, 0.0)
            Y_block = np.einsum("bk,bkg->bg", w_block, X_block, optimize=True).astype(dtype, copy=False)
            Y[q_rows] = Y_block

    return Y

def generate_attr_sets(ad,grid_pts,support,attrs,attr_schema,dtype):
    n_grid = grid_pts.shape[0]
    query_idx_active = support["query_idx_active"]
    nn_idx_active = support["nn_idx_active"]
    local_w = support["local_w"]

    if attrs is not None:
        if not isinstance(attrs, pd.DataFrame):
            attrs = pd.DataFrame(attrs)

        if attr_schema is None:
            attr_schema = infer_attr_schema_from_df(attrs)

        for col, kind in attr_schema.items():
            values = attrs[col].values

            if kind == "continuous":
                out = np.full((n_grid,), np.nan, dtype=dtype)
                if query_idx_active.size > 0:
                    v = np.asarray(values, dtype=dtype)
                    pred = np.sum(local_w * v[nn_idx_active], axis=1).astype(dtype, copy=False)
                    out[query_idx_active] = pred
                ad.obs[col] = out

            elif kind == "categorical":
                codes, categories = pd.factorize(values, sort=True)
                n_cat = len(categories)

                probs = np.zeros((n_grid, n_cat), dtype=dtype)
                labels = np.full((n_grid,), None, dtype=object)

                if query_idx_active.size > 0:
                    code_block = codes[nn_idx_active]  # (Q, k)
                    rows = np.repeat(np.arange(query_idx_active.shape[0]), nn_idx_active.shape[1])
                    cols = code_block.reshape(-1)
                    vals_w = local_w.reshape(-1)

                    probs_active = np.zeros((query_idx_active.shape[0], n_cat), dtype=dtype)
                    np.add.at(probs_active, (rows, cols), vals_w)

                    probs[query_idx_active] = probs_active
                    pred_idx = np.argmax(probs_active, axis=1)
                    labels_active = categories[pred_idx]
                    labels[query_idx_active] = labels_active

                ad.obs[col] = pd.Categorical(labels, categories=list(categories))
                ad.obsm[f"{col}_probs"] = probs
                # ad.uns[f"{col}_categories"] = list(categories)

            elif kind == "boolean":
                v = np.asarray(values).astype(np.float32)
                p_true = np.full((n_grid,), np.nan, dtype=dtype)
                label = np.full((n_grid,), False, dtype=bool)

                if query_idx_active.size > 0:
                    pred = np.sum(local_w * v[nn_idx_active], axis=1).astype(dtype, copy=False)
                    p_true[query_idx_active] = pred
                    label[query_idx_active] = pred >= 0.5

                ad.obs[f"{col}_prob_true"] = p_true
                ad.obs[col] = label
                ad.obsm[f"{col}_probs"] = np.stack(
                    [1.0 - np.nan_to_num(p_true, nan=0.0), np.nan_to_num(p_true, nan=0.0)],
                    axis=1
                )
                ad.uns[f"{col}_categories"] = [False, True]

            else:
                raise ValueError(f"Unsupported attr kind for '{col}': {kind}")
    return ad

def generate_virtual_tissue_fixed_grid(
    coords,
    X,
    grid_pts,
    var_names=None,
    attrs=None,
    attr_schema=None,
    grid_support_mask=None,
    point_weights=None,
    spatial_bandwidth=None,
    attr_spatial_bandwidth=None,
    knn=48,
    min_neighbors=6,
    radius=None,
    chunk_size=64,
    fill_value=0.0,
    dtype=np.float32,
):

    if X.shape[0] != coords.shape[0]:
        raise ValueError("X rows must match coords rows.")


    qc_total_counts,qc_n_genes = compute_qc_covariates(X)
    # -------------------------
    # 1) shared local query support
    # -------------------------
    support = _prepare_local_query_support(
        coords=coords,
        grid_pts=grid_pts,
        grid_support_mask=grid_support_mask,
        point_weights=point_weights,
        qc_total_counts=qc_total_counts,
        qc_n_genes=qc_n_genes,
        spatial_bandwidth=spatial_bandwidth,
        knn=knn,
        min_neighbors=min_neighbors,
        radius=radius,
        dtype=dtype,

    )
    # -------------------------
    # 2) gene generation
    # -------------------------
    Y = generate_gene_sets(grid_pts,X,support,fill_value,chunk_size,dtype)

    # -------------------------
    # 3) create final AnnData
    # -------------------------

    ad = AnnData(X=Y, dtype=Y.dtype)
    if var_names is not None:
        ad.var_names = np.asarray(var_names).astype(str)

    ad.obsm["spatial"] = np.asarray(grid_pts, dtype=dtype)
    ad.obs["valid"] = support["valid_mask"]
    ad.obs["active_query_support"] = support["active_query_support"]
    ad.obs["confidence"] = support["confidence"]
    ad.obs["nearest_dist"] = support["nearest_dist"]

    # -------------------------
    # 4) generate attributes into the same AnnData
    # -------------------------
    if attr_spatial_bandwidth:
        attr_support = _prepare_local_query_support(
            coords=coords,
            grid_pts=grid_pts,
            grid_support_mask=grid_support_mask,
            point_weights=point_weights,
            qc_total_counts=qc_total_counts,
            qc_n_genes=qc_n_genes,
            spatial_bandwidth=attr_spatial_bandwidth,
            knn=knn,
            min_neighbors=min_neighbors,
            radius=radius,
            dtype=dtype,
        )
        ad = generate_attr_sets(ad, grid_pts, attr_support, attrs, attr_schema, dtype)
    else:
        ad = generate_attr_sets(ad,grid_pts,support,attrs,attr_schema,dtype)

    # -------------------------
    # 5) metadata
    # -------------------------
    ad.uns["spatial_bandwidth"] = float(support["spatial_bandwidth"])
    if attr_spatial_bandwidth is not None:
        ad.uns["attr_spatial_bandwidth"] = attr_spatial_bandwidth
    return ad

# ===========================================================================
# GENERATE delta based virtual tissues
# ===========================================================================
def generate_anchor_state_on_grid(
    coords,
    X,
    grid_pts,
    var_names=None,
    attrs=None,
    attr_schema=None,
    grid_support_mask=None,
    spatial_bandwidth=None,
    knn=48,
    min_neighbors=6,
    radius=None,
    chunk_size=64,
    fill_value=0.0,
    dtype=np.float32,
):
    """
    Generate one anchor-specific state on the grid.
    No biological-axis fusion here. Only local spatial regression from one anchor.
    """

    n_grid = grid_pts.shape[0]
    n_genes = X.shape[1]

    # use existing support preparation, but with all point weights = 1
    support = _prepare_local_query_support(
        coords=coords,
        grid_pts=grid_pts,
        grid_support_mask=grid_support_mask,
        point_weights=np.ones((coords.shape[0],), dtype=dtype),
        spatial_bandwidth=spatial_bandwidth,
        knn=knn,
        min_neighbors=min_neighbors,
        radius=radius,
        dtype=dtype
    )

    query_idx_active = support["query_idx_active"]
    nn_idx_active = support["nn_idx_active"]
    local_w = support["local_w"]

    Y = np.full((n_grid, n_genes), fill_value, dtype=dtype)

    if query_idx_active.size > 0:
        for start in range(0, query_idx_active.shape[0], int(chunk_size)):
            end = min(start + int(chunk_size), query_idx_active.shape[0])

            q_rows = query_idx_active[start:end]
            idx_block = nn_idx_active[start:end]
            w_block = local_w[start:end]
            B, k = idx_block.shape

            flat_idx = idx_block.reshape(-1)

            if sp.issparse(X):
                X_block = np.asarray(X[flat_idx].toarray(), dtype=dtype).reshape(B, k, n_genes)
            else:
                X_block = np.asarray(X[flat_idx], dtype=dtype).reshape(B, k, n_genes)

            Y_block = np.einsum("bk,bkg->bg", w_block, X_block, optimize=True).astype(dtype, copy=False)
            Y[q_rows] = Y_block

    ad = AnnData(X=Y, dtype=Y.dtype)
    if var_names is not None:
        ad.var_names = np.asarray(var_names).astype(str)

    ad.obsm["spatial"] = np.asarray(grid_pts, dtype=dtype)
    ad.obs["valid"] = support["valid_mask"]
    ad.obs["confidence"] = support["confidence"]
    ad.obs["nearest_dist"] = support["nearest_dist"]

    # attributes
    if attrs is not None:
        if not isinstance(attrs, pd.DataFrame):
            attrs = pd.DataFrame(attrs)

        if attr_schema is None:
            attr_schema = infer_attr_schema_from_df(attrs)

        for col, kind in attr_schema.items():
            values = attrs[col].values

            if kind == "continuous":
                out = np.full((n_grid,), np.nan, dtype=dtype)
                if query_idx_active.size > 0:
                    v = np.asarray(values, dtype=dtype)
                    pred = np.sum(local_w * v[nn_idx_active], axis=1).astype(dtype, copy=False)
                    out[query_idx_active] = pred
                ad.obs[col] = out

            elif kind == "categorical":
                codes, categories = pd.factorize(values, sort=True)
                n_cat = len(categories)
                probs = np.zeros((n_grid, n_cat), dtype=dtype)

                if query_idx_active.size > 0:
                    code_block = codes[nn_idx_active]
                    rows = np.repeat(np.arange(query_idx_active.shape[0]), nn_idx_active.shape[1])
                    cols = code_block.reshape(-1)
                    vals_w = local_w.reshape(-1)

                    probs_active = np.zeros((query_idx_active.shape[0], n_cat), dtype=dtype)
                    np.add.at(probs_active, (rows, cols), vals_w)
                    probs[query_idx_active] = probs_active

                labels = np.full((n_grid,), "unknown", dtype=object)
                if query_idx_active.size > 0:
                    pred_idx = np.argmax(probs[query_idx_active], axis=1)
                    labels_active = categories[pred_idx].astype(object)
                    labels[query_idx_active] = labels_active

                ad.obs[col] = pd.Series(labels, index=ad.obs_names, dtype="object")
                ad.obsm[f"{col}_probs"] = probs
                ad.uns[f"{col}_categories"] = [str(x) for x in list(categories)]

            elif kind == "boolean":
                v = np.asarray(values).astype(np.float32)
                p_true = np.full((n_grid,), np.nan, dtype=dtype)

                if query_idx_active.size > 0:
                    pred = np.sum(local_w * v[nn_idx_active], axis=1).astype(dtype, copy=False)
                    p_true[query_idx_active] = pred

                ad.obs[f"{col}_prob_true"] = p_true
                ad.obs[col] = np.nan_to_num(p_true, nan=0.0) >= 0.5
                ad.obsm[f"{col}_probs"] = np.stack(
                    [1.0 - np.nan_to_num(p_true, nan=0.0), np.nan_to_num(p_true, nan=0.0)],
                    axis=1
                )
                ad.uns[f"{col}_categories"] = [False, True]

    ad.uns["generator"] = "anchor_state_on_grid"
    return ad


def compose_virtual_state_from_pair(
    ad_a,
    ad_b,
    alpha,
    gene_space="log1p",   # "raw" or "log1p"
    clip_min=0.0,
):
    """
    Compose a query state from two anchor states using delta-based generation:
        S_q = S_a + alpha * (S_b - S_a)
    """

    alpha = float(alpha)

    # geometry and var_names must match
    if ad_a.n_obs != ad_b.n_obs:
        raise ValueError("Anchor states have different n_obs.")
    if ad_a.n_vars != ad_b.n_vars:
        raise ValueError("Anchor states have different n_vars.")
    if not np.array_equal(np.asarray(ad_a.var_names), np.asarray(ad_b.var_names)):
        raise ValueError("Anchor states have different var_names.")
    if not np.allclose(ad_a.obsm["spatial"], ad_b.obsm["spatial"]):
        raise ValueError("Anchor states have different spatial coordinates.")

    Xa = ad_a.X.toarray() if sp.issparse(ad_a.X) else np.asarray(ad_a.X, dtype=np.float32)
    Xb = ad_b.X.toarray() if sp.issparse(ad_b.X) else np.asarray(ad_b.X, dtype=np.float32)

    Xa = Xa.astype(np.float32, copy=False)
    Xb = Xb.astype(np.float32, copy=False)

    if gene_space == "log1p":
        Za = np.log1p(np.maximum(Xa, 0.0))
        Zb = np.log1p(np.maximum(Xb, 0.0))
        Zq = Za + alpha * (Zb - Za)
        Xq = np.expm1(Zq).astype(np.float32, copy=False)
        Xq = np.maximum(Xq, clip_min)
    elif gene_space == "raw":
        Xq = Xa + alpha * (Xb - Xa)
        Xq = np.maximum(Xq.astype(np.float32, copy=False), clip_min)
    else:
        raise ValueError("gene_space must be one of {'raw','log1p'}")

    ad_q = AnnData(X=Xq, dtype=np.float32)
    ad_q.var_names = ad_a.var_names.copy()
    ad_q.obsm["spatial"] = np.asarray(ad_a.obsm["spatial"], dtype=np.float32)

    # keep support / diagnostics simple
    if "valid" in ad_a.obs.columns and "valid" in ad_b.obs.columns:
        ad_q.obs["valid"] = np.asarray(ad_a.obs["valid"], dtype=bool) & np.asarray(ad_b.obs["valid"], dtype=bool)

    for key in ["confidence", "nearest_dist"]:
        if key in ad_a.obs.columns and key in ad_b.obs.columns:
            va = np.asarray(ad_a.obs[key], dtype=np.float32)
            vb = np.asarray(ad_b.obs[key], dtype=np.float32)
            ad_q.obs[key] = va + alpha * (vb - va)

    # compose attributes
    shared_obs_cols = sorted(set(ad_a.obs.columns).intersection(set(ad_b.obs.columns)))
    skip_cols = {"valid", "confidence", "nearest_dist"}
    for col in shared_obs_cols:
        if col in skip_cols:
            continue

        sa = ad_a.obs[col]
        sb = ad_b.obs[col]

        # continuous
        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
            va = pd.to_numeric(sa, errors="coerce").to_numpy(dtype=np.float32)
            vb = pd.to_numeric(sb, errors="coerce").to_numpy(dtype=np.float32)
            vq = va + alpha * (vb - va)
            ad_q.obs[col] = vq.astype(np.float32)
            continue

        # boolean via stored probability if available
        prob_key = f"{col}_probs"
        if prob_key in ad_a.obsm and prob_key in ad_b.obsm:
            pa = np.asarray(ad_a.obsm[prob_key], dtype=np.float32)
            pb = np.asarray(ad_b.obsm[prob_key], dtype=np.float32)
            pq = pa + alpha * (pb - pa)

            # renormalize if multi-class
            if pq.ndim == 2 and pq.shape[1] > 1:
                pq = np.maximum(pq, 0.0)
                row_sum = np.maximum(pq.sum(axis=1, keepdims=True), 1e-8)
                pq = pq / row_sum

            ad_q.obsm[prob_key] = pq

            # categorical assignment
            if f"{col}_categories" in ad_a.uns:
                cats = [str(x) for x in ad_a.uns[f"{col}_categories"]]
                idx = np.argmax(pq, axis=1)
                labels = np.asarray([cats[i] for i in idx], dtype=object)
                ad_q.obs[col] = pd.Series(labels, index=ad_q.obs_names, dtype="object")
            elif col.endswith("_prob_true") or col.startswith("is_"):
                if pq.ndim == 2 and pq.shape[1] == 2:
                    ad_q.obs[col] = pq[:, 1] >= 0.5
            continue

        # fallback: keep anchor a label if alpha < 0.5 else anchor b label
        labels_a = sa.astype(object).to_numpy()
        labels_b = sb.astype(object).to_numpy()
        labels_q = np.where(alpha < 0.5, labels_a, labels_b)
        ad_q.obs[col] = pd.Series(labels_q, index=ad_q.obs_names, dtype="object")

    ad_q.uns["generator"] = "delta_composed_virtual_tissue"
    ad_q.uns["relative_position_alpha"] = float(alpha)
    ad_q.uns["gene_space"] = gene_space
    return ad_q

def generate_virtual_tissue_delta_fixed_grid(
    coords_a,
    X_a,
    coords_b,
    X_b,
    grid_pts,
    var_names=None,
    attrs_a=None,
    attrs_b=None,
    attr_schema=None,
    grid_support_mask=None,
    spatial_bandwidth=None,
    knn=48,
    min_neighbors=6,
    radius=None,
    chunk_size=64,
    fill_value=0.0,
    dtype=np.float32,
    alpha=0.5,
    gene_space="log1p",
):
    """
    Delta-based generation:
      1) estimate anchor A state on grid
      2) estimate anchor B state on grid
      3) compose query by delta
    """
    ad_a = generate_anchor_state_on_grid(
        coords=coords_a,
        X=X_a,
        grid_pts=grid_pts,
        var_names=var_names,
        attrs=attrs_a,
        attr_schema=attr_schema,
        grid_support_mask=grid_support_mask,
        spatial_bandwidth=spatial_bandwidth,
        knn=knn,
        min_neighbors=min_neighbors,
        radius=radius,
        chunk_size=chunk_size,
        fill_value=fill_value,
        dtype=dtype,
    )

    ad_b = generate_anchor_state_on_grid(
        coords=coords_b,
        X=X_b,
        grid_pts=grid_pts,
        var_names=var_names,
        attrs=attrs_b,
        attr_schema=attr_schema,
        grid_support_mask=grid_support_mask,
        spatial_bandwidth=spatial_bandwidth,
        knn=knn,
        min_neighbors=min_neighbors,
        radius=radius,
        chunk_size=chunk_size,
        fill_value=fill_value,
        dtype=dtype,
    )

    ad_q = compose_virtual_state_from_pair(
        ad_a=ad_a,
        ad_b=ad_b,
        alpha=alpha,
        gene_space=gene_space,
    )

    ad_q.uns["source_anchor_ids"] = None
    return ad_q