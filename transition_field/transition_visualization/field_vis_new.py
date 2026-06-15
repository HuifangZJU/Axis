import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from visualization_utils import *
from axis.coordinate_utils import find_inverse_warp_coords
# ============================================================
# Helper 1: build a coarse support grid on the reference slice
# ============================================================
def build_reference_support_grid(
    coords,
    spacing=28.0,
    support_radius=None,
    pad_ratio=0.03,
):
    """
    Build a coarse regular mesh on the reference slice, then keep only
    grid nodes that stay near the tissue support.

    Parameters
    ----------
    coords : (N,2) array
        Reference slice coordinates.
    spacing : float
        Mesh spacing in the same coordinate unit as coords.
    support_radius : float or None
        Keep a grid node only if its nearest observed spot is within this radius.
        If None, defaults to 0.8 * spacing.
    pad_ratio : float
        Small boundary padding.

    Returns
    -------
    grid_xy : (M,2) float32
        Valid mesh nodes.
    idx_map : (ny,nx) int
        Mapping from regular lattice location to valid-node index, -1 if invalid.
    """
    coords = np.asarray(coords, dtype=np.float32)
    if support_radius is None:
        support_radius = 0.8 * spacing

    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    pad = (max_xy - min_xy) * pad_ratio

    x_min, y_min = min_xy - pad
    x_max, y_max = max_xy + pad

    xs = np.arange(x_min, x_max + spacing, spacing, dtype=np.float32)
    ys = np.arange(y_min, y_max + spacing, spacing, dtype=np.float32)

    gx, gy = np.meshgrid(xs, ys)
    grid_all = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)

    tree = cKDTree(coords)
    dist, _ = tree.query(grid_all, k=1)

    keep = dist <= float(support_radius)

    idx_map = -np.ones(gx.shape, dtype=int)
    keep_idx = np.where(keep)[0]
    idx_map.ravel()[keep_idx] = np.arange(len(keep_idx), dtype=int)

    grid_xy = grid_all[keep]

    return grid_xy, idx_map


# ============================================================
# Helper 2: within-plane mesh edges
# ============================================================
def build_mesh_edges(idx_map):
    """
    Build horizontal and vertical mesh edges from idx_map.
    """
    edges = []
    ny, nx = idx_map.shape

    # horizontal neighbors
    for r in range(ny):
        for c in range(nx - 1):
            a = idx_map[r, c]
            b = idx_map[r, c + 1]
            if a >= 0 and b >= 0:
                edges.append((a, b))

    # vertical neighbors
    for r in range(ny - 1):
        for c in range(nx):
            a = idx_map[r, c]
            b = idx_map[r + 1, c]
            if a >= 0 and b >= 0:
                edges.append((a, b))

    return edges

def find_group_dir(root, group_id, pairwise_subdir="pairwise_output"):
    p1 = os.path.join(root, pairwise_subdir, f"group_{group_id}")
    p2 = os.path.join(root, f"group_{group_id}")

    if os.path.isdir(p1):
        return p1
    if os.path.isdir(p2):
        return p2

    raise FileNotFoundError(
        f"Cannot find pairwise folder for group {group_id}.\n"
        f"Tried:\n  {p1}\n  {p2}"
    )
def parse_flow_path(root, group_id, fixed_id, moving_id, pairwise_subdir="pairwise_output"):
    group_dir = find_group_dir(root, group_id, pairwise_subdir=pairwise_subdir)
    return os.path.join(group_dir, f"{fixed_id}_{moving_id}_flow.npy")
def inverse_warp_points(points_xy, flow_hw2):
    """
    Wrapper around axis.coordinate_utils.find_inverse_warp_coords.

    This helper tries a few common calling conventions because the exact
    signature may differ across local versions of the utility.
    """
    trials = [
        lambda: find_inverse_warp_coords(points_xy, flow_hw2),
    ]

    last_err = None
    for fn in trials:
        try:
            out = fn()

            # handle tuple/list returns like (coords, valid_mask)
            if isinstance(out, (tuple, list)):
                out = out[0]

            out = np.asarray(out, dtype=np.float32)
            if out.ndim != 2 or out.shape[1] < 2:
                raise ValueError(f"Unexpected inverse-warp output shape: {out.shape}")

            return out[:, :2]

        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Failed to call find_inverse_warp_coords with the tried signatures. "
        "Please adapt inverse_warp_points() to your local function signature.\n"
        f"Last error: {last_err}"
    )
# ============================================================
# Helper 3: propagate mesh nodes through learned pairwise fields
# ============================================================
def propagate_reference_mesh_through_slices(
    root,
    group_id,
    sample_ids,
    grid_xy,
    pairwise_subdir="pairwise_output",
):
    """
    Propagate a reference mesh node set sequentially across slices using the
    learned transition fields, following the same logic as your earlier code.

    Returns
    -------
    traj : (S, M, 2) float32
        Mesh node coordinates on each slice plane; NaN where invalid.
    """
    sample_ids = list(sample_ids)
    n_slices = len(sample_ids)
    n_nodes = grid_xy.shape[0]

    traj = np.full((n_slices, n_nodes, 2), np.nan, dtype=np.float32)
    traj[0] = grid_xy.copy()

    active = np.ones(n_nodes, dtype=bool)
    current = grid_xy.copy()

    for i in range(n_slices - 1):
        fixed_id = sample_ids[i]
        moving_id = sample_ids[i + 1]

        # same calling pattern as in your current code
        flow_path = parse_flow_path(
            root=root,
            group_id=group_id,
            fixed_id=moving_id,
            moving_id=fixed_id,
            pairwise_subdir=pairwise_subdir,
        )
        if not os.path.exists(flow_path):
            raise FileNotFoundError(f"Missing flow file: {flow_path}")

        flow = ensure_flow_hw2(np.load(flow_path))

        next_all = np.full((n_nodes, 2), np.nan, dtype=np.float32)

        if np.any(active):
            next_pts = inverse_warp_points(current[active], flow)

            H, W, _ = flow.shape
            valid = np.all(np.isfinite(next_pts), axis=1)
            valid &= (
                (next_pts[:, 0] >= 0) & (next_pts[:, 0] < W) &
                (next_pts[:, 1] >= 0) & (next_pts[:, 1] < H)
            )

            old_active_idx = np.where(active)[0]
            keep_idx = old_active_idx[valid]
            next_all[keep_idx] = next_pts[valid]

            new_active = np.zeros_like(active)
            new_active[keep_idx] = True
            active = new_active
            current = next_all.copy()

        traj[i + 1] = next_all

    return traj


# ============================================================
# Helper 4: densify display planes between measured slices
# ============================================================
def densify_mesh_display(
    traj,
    plane_pos,
    n_between=3,
):
    """
    Insert display-only intermediate mesh states between measured slices.

    This is purely for visualization so the field reads as continuous.
    Intermediate node positions are linearly interpolated between adjacent
    measured mesh states.

    Returns
    -------
    dense_states : list[(M,2)]
    dense_x      : list[float]
    dense_is_anchor : list[bool]
    """
    n_slices = traj.shape[0]

    dense_states = []
    dense_x = []
    dense_is_anchor = []

    for i in range(n_slices - 1):
        p0 = traj[i]
        p1 = traj[i + 1]
        x0 = plane_pos[i]
        x1 = plane_pos[i + 1]

        # anchor
        if i == 0:
            dense_states.append(p0.copy())
            dense_x.append(float(x0))
            dense_is_anchor.append(True)

        # display-only intermediate planes
        for k in range(1, n_between + 1):
            t = k / (n_between + 1.0)
            interp = np.full_like(p0, np.nan, dtype=np.float32)

            valid = np.all(np.isfinite(p0), axis=1) & np.all(np.isfinite(p1), axis=1)
            interp[valid] = (1 - t) * p0[valid] + t * p1[valid]

            dense_states.append(interp)
            dense_x.append((1 - t) * x0 + t * x1)
            dense_is_anchor.append(False)

        # next anchor
        dense_states.append(p1.copy())
        dense_x.append(float(x1))
        dense_is_anchor.append(True)

    return dense_states, dense_x, dense_is_anchor


# ============================================================
# Helper 5: draw a slice frame
# ============================================================
def draw_plane_frame(
    ax,
    x_plane,
    y_min,
    y_max,
    z_min,
    z_max,
    color="0.55",
    alpha=0.35,
    linestyle="--",
    linewidth=0.8,
):
    lines = [
        [(x_plane, y_min, z_min), (x_plane, y_max, z_min)],
        [(x_plane, y_max, z_min), (x_plane, y_max, z_max)],
        [(x_plane, y_max, z_max), (x_plane, y_min, z_max)],
        [(x_plane, y_min, z_max), (x_plane, y_min, z_min)],
    ]
    lc = Line3DCollection(lines, colors=color, linewidths=linewidth,
                          linestyles=linestyle, alpha=alpha)
    ax.add_collection3d(lc)


# ============================================================
# Helper 6: draw one wireframe mesh on one plane
# ============================================================
def draw_mesh_on_plane(
    ax,
    xy,
    x_plane,
    edges,
    color="#E46D5E",
    alpha=0.95,
    linewidth=1.0,
):
    segments = []
    for a, b in edges:
        pa = xy[a]
        pb = xy[b]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            segments.append([
                (x_plane, pa[0], pa[1]),
                (x_plane, pb[0], pb[1]),
            ])

    if len(segments) == 0:
        return

    lc = Line3DCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(lc)


# ============================================================
# Helper 7: draw sparse longitudinal streamlines
# ============================================================
def draw_longitudinal_field_streamlines(
    ax,
    dense_states,
    dense_x,
    idx_map,
    color="#B63C2F",
    alpha=0.26,
    linewidth=1.1,
    row_stride=2,
    col_stride=2,
):
    """
    Draw sparse field streamlines by following selected mesh nodes across
    the densified display planes.
    """
    ny, nx = idx_map.shape
    segments = []

    for r in range(0, ny, row_stride):
        for c in range(0, nx, col_stride):
            node_id = idx_map[r, c]
            if node_id < 0:
                continue

            pts = []
            for x_plane, state in zip(dense_x, dense_states):
                p = state[node_id]
                if np.all(np.isfinite(p)):
                    pts.append((x_plane, p[0], p[1]))

            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    segments.append([pts[i], pts[i + 1]])

    if len(segments) == 0:
        return

    lc = Line3DCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(lc)
def parse_h5ad_path(root, group_id, sample_id):
    return os.path.join(root, "h5ad", f"{group_id}_{sample_id}.h5ad")


def parse_network_npz_path(root, group_id, sample_id):
    return os.path.join(root, "network_input", f"{group_id}_{sample_id}_validation.npz")
def load_npz_coord(npz_path):
    data = np.load(npz_path)
    if "coord" not in data:
        raise KeyError(f"'coord' not found in {npz_path}")
    coord = np.asarray(data["coord"], dtype=np.float32)
    if coord.ndim != 2 or coord.shape[1] < 2:
        raise ValueError(f"coord in {npz_path} must have shape [N,2+], got {coord.shape}")
    return coord[:, :2]

def load_plot_slice_coords_aligned(
    root,
    group_id,
    sample_id,
    reference_id=0,
    coord_key="spatial",
    pairwise_subdir="pairwise_output",
):
    """
    Coordinates used for 3D plotting should be in the SAME coordinate system
    as the pairwise flow / inverse-warp propagation.

    Priority:
      1) adata.obsm[coord_key]
      2) reference slice -> network_input coord
      3) moving slice -> warped coord from reference
    """
    h5ad_path = parse_h5ad_path(root, group_id, sample_id)
    adata = sc.read_h5ad(h5ad_path)

    if coord_key in adata.obsm:
        coord = np.asarray(adata.obsm[coord_key], dtype=np.float32)
        return adata, coord[:, :2]

    if sample_id == reference_id:
        npz_path = parse_network_npz_path(root, group_id, sample_id)
        coord = load_npz_coord(npz_path)
        return adata, coord[:, :2]

    group_dir = find_group_dir(root, group_id, pairwise_subdir=pairwise_subdir)
    warped_path = os.path.join(group_dir, f"{reference_id}_{sample_id}_warped_coor.npy")
    if os.path.exists(warped_path):
        coord = np.asarray(np.load(warped_path), dtype=np.float32)
        return adata, coord[:, :2]

    raise ValueError(
        f"Could not find aligned plotting coordinates for group={group_id}, sample={sample_id}. "
        f"Please assign obsm['{coord_key}'] first, or provide warped coords."
    )
import os
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm, TwoSlopeNorm
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.lines import Line2D


# ============================================================
# 1. Generic helpers
# ============================================================
def build_regular_grid_with_support(
    coords_xy,
    step=24.0,
    support_radius=None,
    pad_ratio=0.03,
):
    """
    Build a regular grid and keep only nodes close to tissue support.
    Returns:
        grid_xy   : (M,2)
        idx_map   : (ny,nx), valid node id or -1
        xs, ys    : 1D lattice coordinates
        keep_mask : (ny,nx) bool
    """
    coords_xy = np.asarray(coords_xy, dtype=np.float32)
    if support_radius is None:
        support_radius = 0.8 * step

    mn = coords_xy.min(axis=0)
    mx = coords_xy.max(axis=0)
    pad = (mx - mn) * pad_ratio

    xs = np.arange(mn[0] - pad[0], mx[0] + pad[0] + step, step, dtype=np.float32)
    ys = np.arange(mn[1] - pad[1], mx[1] + pad[1] + step, step, dtype=np.float32)

    gx, gy = np.meshgrid(xs, ys)
    grid_all = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)

    tree = cKDTree(coords_xy)
    dist, _ = tree.query(grid_all, k=1)
    keep = dist <= float(support_radius)

    idx_map = -np.ones(gx.shape, dtype=int)
    keep_idx = np.where(keep)[0]
    idx_map.ravel()[keep_idx] = np.arange(len(keep_idx), dtype=int)

    return grid_all[keep], idx_map, xs, ys, keep.reshape(gx.shape)


def build_mesh_edges(idx_map):
    edges = []
    ny, nx = idx_map.shape

    # horizontal
    for r in range(ny):
        for c in range(nx - 1):
            a = idx_map[r, c]
            b = idx_map[r, c + 1]
            if a >= 0 and b >= 0:
                edges.append((a, b))

    # vertical
    for r in range(ny - 1):
        for c in range(nx):
            a = idx_map[r, c]
            b = idx_map[r + 1, c]
            if a >= 0 and b >= 0:
                edges.append((a, b))

    return edges


def sample_pair_transition_field(
    root,
    group_id,
    fixed_id,
    moving_id,
    grid_xy,
    pairwise_subdir="pairwise_output",
):
    """
    Sample the learned transition field on arbitrary points from fixed_id plane
    to moving_id plane, using the same forward-propagation logic as your
    current trajectory code:
      load flow with parse_flow_path(... fixed_id=moving_id, moving_id=fixed_id)
      then inverse_warp_points(...)
    """
    flow_path = parse_flow_path(
        root=root,
        group_id=group_id,
        fixed_id=moving_id,
        moving_id=fixed_id,
        pairwise_subdir=pairwise_subdir,
    )
    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Missing flow file: {flow_path}")

    flow = ensure_flow_hw2(np.load(flow_path))
    mapped = inverse_warp_points(grid_xy, flow)

    H, W, _ = flow.shape
    valid = np.all(np.isfinite(mapped), axis=1)
    valid &= (
        (mapped[:, 0] >= 0) & (mapped[:, 0] < W) &
        (mapped[:, 1] >= 0) & (mapped[:, 1] < H)
    )

    out = np.full_like(grid_xy, np.nan, dtype=np.float32)
    out[valid] = mapped[valid]
    delta = out - grid_xy
    return out, delta, valid


def grid_values_from_sparse_nodes(idx_map, vals_sparse):
    """
    Put sparse node values back into a dense regular lattice.
    """
    grid = np.full(idx_map.shape, np.nan, dtype=np.float32)
    keep = idx_map >= 0
    grid[keep] = vals_sparse[idx_map[keep]]
    return grid


def upsample_nan_grid(grid, factor=4, order=3):
    """
    Smooth upsampling for display.
    NaNs are handled by upsampling both values and validity mask.
    """
    if factor is None or factor <= 1:
        return grid

    valid = np.isfinite(grid).astype(np.float32)
    fill = np.nan_to_num(grid, nan=0.0)

    up_fill = zoom(fill, zoom=factor, order=order)
    up_valid = zoom(valid, zoom=factor, order=1)

    out = np.full_like(up_fill, np.nan, dtype=np.float32)
    keep = up_valid > 0.15
    out[keep] = up_fill[keep] / np.maximum(up_valid[keep], 1e-8)
    return out


def draw_plane_frame(
    ax,
    x_plane,
    y_min,
    y_max,
    z_min,
    z_max,
    color="0.55",
    alpha=0.28,
    linestyle="--",
    linewidth=0.8,
):
    segs = [
        [(x_plane, y_min, z_min), (x_plane, y_max, z_min)],
        [(x_plane, y_max, z_min), (x_plane, y_max, z_max)],
        [(x_plane, y_max, z_max), (x_plane, y_min, z_max)],
        [(x_plane, y_min, z_max), (x_plane, y_min, z_min)],
    ]
    lc = Line3DCollection(
        segs,
        colors=color,
        linewidths=linewidth,
        linestyles=linestyle,
        alpha=alpha,
    )
    ax.add_collection3d(lc)


def draw_mesh_on_plane(
    ax,
    x_plane,
    xy,
    edges,
    color="#E76F51",
    alpha=0.95,
    linewidth=1.0,
):
    segs = []
    for a, b in edges:
        pa = xy[a]
        pb = xy[b]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            segs.append([
                (x_plane, pa[0], pa[1]),
                (x_plane, pb[0], pb[1]),
            ])
    if len(segs) == 0:
        return

    lc = Line3DCollection(segs, colors=color, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(lc)


def draw_mesh_bridges(
    ax,
    x0,
    x1,
    p0,
    p1,
    idx_map,
    color="#B63C2F",
    alpha=0.18,
    linewidth=0.7,
    row_stride=2,
    col_stride=2,
):
    """
    Sparse inter-plane connection lines for representative field continuity.
    """
    ny, nx = idx_map.shape
    segs = []

    for r in range(0, ny, row_stride):
        for c in range(0, nx, col_stride):
            node_id = idx_map[r, c]
            if node_id < 0:
                continue
            a = p0[node_id]
            b = p1[node_id]
            if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                segs.append([
                    (x0, a[0], a[1]),
                    (x1, b[0], b[1]),
                ])

    if len(segs) == 0:
        return

    lc = Line3DCollection(segs, colors=color, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(lc)

def extract_gene_vals(adata, gene, use_log1p=True):
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names")

    j = int(np.where(adata.var_names == gene)[0][0])
    v = adata.X[:, j]

    if sp.issparse(v):
        v = np.asarray(v.toarray()).ravel().astype(np.float32)
    else:
        v = np.asarray(v).ravel().astype(np.float32)

    if use_log1p:
        v = np.log1p(v)

    return v

def prepare_slice_values(
    adata,
    gene_or_obs,
    use_log1p=True,
    categorical_cmap="tab10",
):
    """
    Supports either gene expression or obs attribute.
    Returns:
        vals_plot, meta
    """
    try:
        vals = extract_gene_vals(adata, gene_or_obs, use_log1p=use_log1p)
        meta = {
            "kind": "continuous",
            "categories": None,
            "cmap": "viridis",
        }
        return vals.astype(np.float32), meta
    except Exception:
        vals, meta0 = extract_obs_vals(adata, gene_or_obs)
        meta = {
            "kind": meta0["kind"],
            "categories": meta0["categories"],
            "cmap": categorical_cmap if meta0["is_discrete"] else "viridis",
        }
        return vals.astype(np.float32), meta

def extract_obs_vals(
    adata,
    obs_key,
    use_log1p_continuous=False,
    categorical_as_codes=True,
):
    """
    Extract obs values in a visualization-friendly numeric format.

    Returns
    -------
    vals : np.ndarray, shape (n_obs,), float32
        Numeric values ready for scatter(c=vals, ...).
        For categorical attributes, this is category code as float32.
        Missing values are returned as np.nan.
    meta : dict
        {
            "kind": "continuous" | "categorical" | "boolean",
            "categories": list[str] or None,
            "is_discrete": bool,
        }
    """

    if obs_key not in adata.obs.columns:
        raise ValueError(f"obs key '{obs_key}' not found in adata.obs.")

    s = adata.obs[obs_key]

    # ---- boolean ----
    if pd.api.types.is_bool_dtype(s):
        vals = s.astype(np.float32).to_numpy()
        meta = {
            "kind": "boolean",
            "categories": ["False", "True"],
            "is_discrete": True,
        }
        return vals, meta

    # ---- numeric continuous ----
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)
        if use_log1p_continuous:
            valid = np.isfinite(vals)
            if np.any(vals[valid] < 0):
                raise ValueError(f"obs['{obs_key}'] contains negative values; cannot apply log1p safely.")
            vals[valid] = np.log1p(vals[valid]).astype(np.float32)
        meta = {
            "kind": "continuous",
            "categories": None,
            "is_discrete": False,
        }
        return vals, meta

    # ---- categorical / object / string ----
    cat = pd.Categorical(s)
    codes = cat.codes.astype(np.float32)
    codes[codes < 0] = np.nan   # missing values become NaN

    meta = {
        "kind": "categorical",
        "categories": [str(x) for x in cat.categories.tolist()],
        "is_discrete": True,
    }
    return codes, meta
def compute_global_style(slice_cache, gene_or_obs, use_log1p=True, categorical_cmap="tab10"):
    """
    Build a shared color style across slices.
    """
    vals_list = []
    metas = []
    for sid in slice_cache:
        vals, meta = prepare_slice_values(
            slice_cache[sid]["adata"],
            gene_or_obs=gene_or_obs,
            use_log1p=use_log1p,
            categorical_cmap=categorical_cmap,
        )
        slice_cache[sid]["vals_plot"] = vals
        metas.append(meta)
        vals_list.append(vals)

    kind = metas[0]["kind"]
    if kind == "continuous":
        x = np.concatenate(vals_list)
        vmin = float(np.nanpercentile(x, 1))
        vmax = float(np.nanpercentile(x, 99))
        norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))
        cmap = get_cmap("viridis")
        legend_info = {"kind": "continuous", "label": gene_or_obs}
    else:
        cats = metas[0]["categories"]
        n_cat = len(cats)
        cmap = get_cmap(categorical_cmap, n_cat)
        norm = BoundaryNorm(np.arange(-0.5, n_cat + 0.5, 1), n_cat)
        legend_info = {"kind": "categorical", "categories": cats, "cmap": cmap}

    return cmap, norm, legend_info
import os
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.ndimage import zoom
from matplotlib.colors import Normalize, BoundaryNorm, TwoSlopeNorm
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ============================================================
# Helpers
# ============================================================
def build_regular_grid_with_support(
    coords_xy,
    step=24.0,
    support_radius=None,
    pad_ratio=0.03,
):
    """
    Build a regular grid and keep only nodes close to tissue support.

    Returns
    -------
    grid_xy   : (M,2)
    idx_map   : (ny,nx), valid node id or -1
    xs, ys    : 1D lattice coordinates
    keep_mask : (ny,nx) bool
    """
    coords_xy = np.asarray(coords_xy, dtype=np.float32)
    if support_radius is None:
        support_radius = 0.8 * step

    mn = coords_xy.min(axis=0)
    mx = coords_xy.max(axis=0)
    pad = (mx - mn) * pad_ratio

    xs = np.arange(mn[0] - pad[0], mx[0] + pad[0] + step, step, dtype=np.float32)
    ys = np.arange(mn[1] - pad[1], mx[1] + pad[1] + step, step, dtype=np.float32)

    gx, gy = np.meshgrid(xs, ys)
    grid_all = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)

    tree = cKDTree(coords_xy)
    dist, _ = tree.query(grid_all, k=1)
    keep = dist <= float(support_radius)

    idx_map = -np.ones(gx.shape, dtype=int)
    keep_idx = np.where(keep)[0]
    idx_map.ravel()[keep_idx] = np.arange(len(keep_idx), dtype=int)

    return grid_all[keep], idx_map, xs, ys, keep.reshape(gx.shape)


def build_mesh_edges(idx_map):
    edges = []
    ny, nx = idx_map.shape

    # horizontal
    for r in range(ny):
        for c in range(nx - 1):
            a = idx_map[r, c]
            b = idx_map[r, c + 1]
            if a >= 0 and b >= 0:
                edges.append((a, b))

    # vertical
    for r in range(ny - 1):
        for c in range(nx):
            a = idx_map[r, c]
            b = idx_map[r + 1, c]
            if a >= 0 and b >= 0:
                edges.append((a, b))

    return edges


def sample_pair_transition_field(
    root,
    group_id,
    fixed_id,
    moving_id,
    grid_xy,
    pairwise_subdir="pairwise_output",
):
    """
    Sample the learned transition field on arbitrary points from fixed_id plane
    to moving_id plane, following the same calling pattern as your previous code.
    """
    flow_path = parse_flow_path(
        root=root,
        group_id=group_id,
        fixed_id=moving_id,
        moving_id=fixed_id,
        pairwise_subdir=pairwise_subdir,
    )
    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Missing flow file: {flow_path}")

    flow = ensure_flow_hw2(np.load(flow_path))
    mapped = inverse_warp_points(grid_xy, flow)

    H, W, _ = flow.shape
    valid = np.all(np.isfinite(mapped), axis=1)
    valid &= (
        (mapped[:, 0] >= 0) & (mapped[:, 0] < W) &
        (mapped[:, 1] >= 0) & (mapped[:, 1] < H)
    )

    out = np.full_like(grid_xy, np.nan, dtype=np.float32)
    out[valid] = mapped[valid]
    delta = out - grid_xy
    return out, delta, valid


def grid_values_from_sparse_nodes(idx_map, vals_sparse):
    grid = np.full(idx_map.shape, np.nan, dtype=np.float32)
    keep = idx_map >= 0
    grid[keep] = vals_sparse[idx_map[keep]]
    return grid


def upsample_nan_grid(grid, factor=4, order=3):
    """
    Smooth upsampling for display.
    """
    if factor is None or factor <= 1:
        return grid

    valid = np.isfinite(grid).astype(np.float32)
    fill = np.nan_to_num(grid, nan=0.0)

    up_fill = zoom(fill, zoom=factor, order=order)
    up_valid = zoom(valid, zoom=factor, order=1)

    out = np.full_like(up_fill, np.nan, dtype=np.float32)
    keep = up_valid > 0.15
    out[keep] = up_fill[keep] / np.maximum(up_valid[keep], 1e-8)
    return out


def draw_plane_frame(
    ax,
    x_plane,
    y_min,
    y_max,
    z_min,
    z_max,
    color="0.55",
    alpha=0.28,
    linestyle="--",
    linewidth=0.8,
):
    segs = [
        [(x_plane, y_min, z_min), (x_plane, y_max, z_min)],
        [(x_plane, y_max, z_min), (x_plane, y_max, z_max)],
        [(x_plane, y_max, z_max), (x_plane, y_min, z_max)],
        [(x_plane, y_min, z_max), (x_plane, y_min, z_min)],
    ]
    lc = Line3DCollection(
        segs,
        colors=color,
        linewidths=linewidth,
        linestyles=linestyle,
        alpha=alpha,
    )
    ax.add_collection3d(lc)


def draw_mesh_on_plane(
    ax,
    x_plane,
    xy,
    edges,
    color="#E76F51",
    alpha=0.95,
    linewidth=1.0,
):
    segs = []
    for a, b in edges:
        pa = xy[a]
        pb = xy[b]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            segs.append([
                (x_plane, pa[0], pa[1]),
                (x_plane, pb[0], pb[1]),
            ])
    if len(segs) == 0:
        return

    lc = Line3DCollection(segs, colors=color, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(lc)


def draw_mesh_bridges(
    ax,
    x0,
    x1,
    p0,
    p1,
    idx_map,
    color="#B63C2F",
    alpha=0.16,
    linewidth=0.7,
    row_stride=2,
    col_stride=2,
):
    segs = []
    ny, nx = idx_map.shape

    for r in range(0, ny, row_stride):
        for c in range(0, nx, col_stride):
            node_id = idx_map[r, c]
            if node_id < 0:
                continue
            a = p0[node_id]
            b = p1[node_id]
            if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                segs.append([
                    (x0, a[0], a[1]),
                    (x1, b[0], b[1]),
                ])

    if len(segs) == 0:
        return

    lc = Line3DCollection(segs, colors=color, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(lc)


def prepare_slice_values(
    adata,
    gene_or_obs,
    use_log1p=True,
    categorical_cmap="tab10",
):
    """
    Supports either gene expression or obs attribute.
    """
    try:
        vals = extract_gene_vals(adata, gene_or_obs, use_log1p=use_log1p)
        meta = {
            "kind": "continuous",
            "categories": None,
            "cmap": "viridis",
        }
        return vals.astype(np.float32), meta
    except Exception:
        vals, meta0 = extract_obs_vals(adata, gene_or_obs)
        meta = {
            "kind": meta0["kind"],
            "categories": meta0["categories"],
            "cmap": categorical_cmap if meta0["is_discrete"] else "viridis",
        }
        return vals.astype(np.float32), meta


def compute_global_style(slice_cache, gene_or_obs, use_log1p=True, categorical_cmap="tab10"):
    vals_list = []
    metas = []
    for sid in slice_cache:
        vals, meta = prepare_slice_values(
            slice_cache[sid]["adata"],
            gene_or_obs=gene_or_obs,
            use_log1p=use_log1p,
            categorical_cmap=categorical_cmap,
        )
        slice_cache[sid]["vals_plot"] = vals
        metas.append(meta)
        vals_list.append(vals)

    kind = metas[0]["kind"]
    if kind == "continuous":
        x = np.concatenate(vals_list)
        vmin = float(np.nanpercentile(x, 1))
        vmax = float(np.nanpercentile(x, 99))
        norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))
        cmap = get_cmap("viridis")
        legend_info = {"kind": "continuous", "label": gene_or_obs}
    else:
        cats = metas[0]["categories"]
        n_cat = len(cats)
        cmap = get_cmap(categorical_cmap, n_cat)
        norm = BoundaryNorm(np.arange(-0.5, n_cat + 0.5, 1), n_cat)
        legend_info = {"kind": "categorical", "categories": cats, "cmap": cmap}

    return cmap, norm, legend_info


def _overlay_source_spots(ax, coord, color="black", alpha=0.22, s=4):
    ax.scatter(
        coord[:, 0],
        coord[:, 1],
        s=s,
        c=color,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )


def _style_2d_axis(ax, title=None, invert_x=False, invert_y=False):
    if title is not None:
        ax.set_title(title, fontsize=16, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()


# ============================================================
# Figure 1: 3D transition-field view on measured slices only
# ============================================================
def plot_transition_field_3d(
    root,
    group_id,
    sample_ids,
    gene="layer_guess_reordered",
    reference_id=0,
    coord_key="spatial",
    pairwise_subdir="pairwise_output",
    axis_positions=None,
    depth_gap=180.0,
    physical_spacing=False,
    use_log1p=True,

    # slice data
    slice_point_size=1.2,
    slice_alpha=0.55,
    categorical_cmap="tab10",

    # field mesh
    mesh_step=28.0,
    mesh_support_radius=None,
    mesh_color="0.45",
    mesh_color_target="0.45",
    mesh_linewidth=1,
    bridge_color="#F4A261",
    bridge_alpha=0.3,
    bridge_row_stride=2,
    bridge_col_stride=2,

    # view
    elev=12,
    azim=-145,
    fig_width=12,
    fig_height=8.5,
    xlim=None,
    ylim=None,
    zlim=None,
    invert_x=False,
    invert_y=False,
    show_slice_labels=True,
    save_path=None,
    show=True,
):
    sample_ids = list(sample_ids)
    if len(sample_ids) < 2:
        raise ValueError("Need at least two slices.")

    # plane positions
    if axis_positions is not None:
        if len(axis_positions) != len(sample_ids):
            raise ValueError("axis_positions must match sample_ids length.")
        plane_pos = {sid: float(pos) for sid, pos in zip(sample_ids, axis_positions)}
    else:
        if physical_spacing:
            s0 = float(sample_ids[0])
            plane_pos = {sid: (float(sid) - s0) * float(depth_gap) for sid in sample_ids}
        else:
            plane_pos = {sid: i * float(depth_gap) for i, sid in enumerate(sample_ids)}

    # load slices
    slice_cache = {}
    all_y = []
    all_z = []

    for sid in sample_ids:
        adata, coord = load_plot_slice_coords_aligned(
            root=root,
            group_id=group_id,
            sample_id=sid,
            reference_id=reference_id,
            coord_key=coord_key,
            pairwise_subdir=pairwise_subdir,
        )

        n = min(adata.n_obs, coord.shape[0])
        adata = adata[:n].copy()
        coord = coord[:n, :2]

        slice_cache[sid] = {
            "adata": adata,
            "coord": coord,
            "x_plane": plane_pos[sid],
        }
        all_y.append(coord[:, 0])
        all_z.append(coord[:, 1])

    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    cmap_slice, norm_slice, legend_info = compute_global_style(
        slice_cache=slice_cache,
        gene_or_obs=gene,
        use_log1p=use_log1p,
        categorical_cmap=categorical_cmap,
    )

    y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    z_min_, z_max_ = float(np.nanmin(all_z)), float(np.nanmax(all_z))
    y_pad = max(10.0, 0.03 * (y_max - y_min + 1e-6))
    z_pad = max(10.0, 0.03 * (z_max_ - z_min_ + 1e-6))

    # precompute field mesh for each adjacent pair
    pair_mesh = []
    for i in range(len(sample_ids) - 1):
        sid0 = sample_ids[i]
        sid1 = sample_ids[i + 1]

        coord0 = slice_cache[sid0]["coord"]
        mesh_xy, mesh_idx_map, _, _, _ = build_regular_grid_with_support(
            coord0,
            step=mesh_step,
            support_radius=mesh_support_radius,
        )
        mesh_edges = build_mesh_edges(mesh_idx_map)

        mesh_xy_next, _, _ = sample_pair_transition_field(
            root=root,
            group_id=group_id,
            fixed_id=sid0,
            moving_id=sid1,
            grid_xy=mesh_xy,
            pairwise_subdir=pairwise_subdir,
        )

        pair_mesh.append({
            "sid0": sid0,
            "sid1": sid1,
            "x0": plane_pos[sid0],
            "x1": plane_pos[sid1],
            "mesh_xy": mesh_xy,
            "mesh_xy_next": mesh_xy_next,
            "mesh_idx_map": mesh_idx_map,
            "mesh_edges": mesh_edges,
        })

    # plot
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.set_proj_type("ortho")
    ax3d.set_position([0.03, 0.08, 0.78, 0.84])

    x_extent = max(plane_pos.values()) - min(plane_pos.values()) + 1e-6
    y_extent = y_max - y_min + 1e-6
    z_extent = z_max_ - z_min_ + 1e-6
    ax3d.set_box_aspect((3 * x_extent, y_extent, z_extent))

    # frames
    for sid in sample_ids:
        draw_plane_frame(
            ax=ax3d,
            x_plane=plane_pos[sid],
            y_min=y_min - y_pad,
            y_max=y_max + y_pad,
            z_min=z_min_ - z_pad,
            z_max=z_max_ + z_pad,
            color="0.55",
            alpha=0.25,
            linestyle="--",
            linewidth=0.8,
        )

    # slice data
    for sid in sample_ids:
        coord = slice_cache[sid]["coord"]
        vals = slice_cache[sid]["vals_plot"]
        x_plane = slice_cache[sid]["x_plane"]

        # rgba = cmap_slice(norm_slice(vals))
        # rgba[:, 3] = slice_alpha

        vmax = robust_vmax([vals], q=0.995)
        norm = Normalize(vmin=0.0, vmax=vmax)
        rgba = make_rgba_with_value_alpha_abs(
            vals=vals,
            norm=norm,
            cmap_name=categorical_cmap,
            alpha_min=1.0,
            alpha_max=1.0
        )

        # ax3d.scatter(
        #     np.full(coord.shape[0], x_plane, dtype=np.float32),
        #     coord[:, 0],
        #     coord[:, 1],
        #     c=rgba,
        #     s=slice_point_size,
        #     linewidths=0,
        #     depthshade=False,
        #     rasterized=True,
        # )

    # field mesh
    for i, pdict in enumerate(pair_mesh):
        if i==0:
            draw_mesh_on_plane(
                ax=ax3d,
                x_plane=pdict["x0"],
                xy=pdict["mesh_xy"],
                edges=pdict["mesh_edges"],
                color=mesh_color,
                alpha=0.95,
                linewidth=mesh_linewidth,
            )
        draw_mesh_on_plane(
            ax=ax3d,
            x_plane=pdict["x1"],
            xy=pdict["mesh_xy_next"],
            edges=pdict["mesh_edges"],
            color=mesh_color_target,
            alpha=0.85,
            linewidth=mesh_linewidth,
        )
        # draw_mesh_bridges(
        #     ax=ax3d,
        #     x0=pdict["x0"],
        #     x1=pdict["x1"],
        #     p0=pdict["mesh_xy"],
        #     p1=pdict["mesh_xy_next"],
        #     idx_map=pdict["mesh_idx_map"],
        #     color=bridge_color,
        #     alpha=bridge_alpha,
        #     linewidth=0.95,
        #     row_stride=bridge_row_stride,
        #     col_stride=bridge_col_stride,
        # )

    # limits
    if xlim is not None:
        ax3d.set_xlim(xlim)
    else:
        dmin = min(plane_pos.values())
        dmax = max(plane_pos.values())
        dpad = max(20.0, 0.05 * (dmax - dmin + 1e-6))
        ax3d.set_xlim(dmin - dpad, dmax + dpad)

    if ylim is not None:
        ax3d.set_ylim(ylim[::-1] if invert_x else ylim)
    else:
        ax3d.set_ylim([y_max + y_pad, y_min - y_pad] if invert_x else [y_min - y_pad, y_max + y_pad])

    if zlim is not None:
        ax3d.set_zlim(zlim[::-1] if invert_y else zlim)
    else:
        ax3d.set_zlim([z_max_ + z_pad, z_min_ - z_pad] if invert_y else [z_min_ - z_pad, z_max_ + z_pad])

    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    ax3d.grid(False)
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_title("Transition field on measured slices", fontsize=18, pad=14)

    if show_slice_labels:
        for sid in sample_ids:
            ax3d.text(
                plane_pos[sid],
                y_min + 0.05 * (y_max - y_min),
                z_min_ + 0.07 * (z_max_ - z_min_),
                f"S{sid}",
                fontsize=20,
                color="black"
            )

    # legends
    if legend_info["kind"] == "continuous":
        sm = ScalarMappable(norm=norm_slice, cmap=cmap_slice)
        sm.set_array([])
        cax = fig.add_axes([0.82, 0.18, 0.015, 0.18])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(gene, fontsize=11)
        cb.ax.tick_params(labelsize=9)
    else:
        handles = []
        for i, cat in enumerate(legend_info["categories"]):
            handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=legend_info["cmap"](i),
                    markeredgecolor="none",
                    markersize=7,
                    label=str(cat),
                )
            )
        fig.legend(
            handles=handles,
            title="Slice data",
            loc="center left",
            bbox_to_anchor=(0.80, 0.68),
            frameon=False,
            fontsize=10,
            title_fontsize=11,
            labelspacing=0.25,
        )

    field_handles = [
        Line2D([0, 1], [0, 0], color=mesh_color, lw=1.2, label="Field mesh on source slice"),
        Line2D([0, 1], [0, 0], color=mesh_color_target, lw=1.2, label="Transported mesh on target slice"),
        Line2D([0, 1], [0, 0], color=bridge_color, lw=1.0, alpha=0.45, label="Field continuity"),
    ]
    fig.legend(
        handles=field_handles,
        title="Transition field",
        loc="center left",
        bbox_to_anchor=(0.80, 0.48),
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        labelspacing=0.45,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=350, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Figure 2: one adjacent pair, three 2D subplots
# ============================================================
def plot_transition_field_pair_views(
    root,
    group_id,
    fixed_id,
    moving_id,
    reference_id=0,
    coord_key="spatial",
    pairwise_subdir="pairwise_output",

    # heatmap grid / display
    heat_step=10.0,
    heat_support_radius=None,
    heat_upsample=4,
    heat_cmap="coolwarm",

    # arrows
    arrow_step=28.0,
    arrow_support_radius=None,
    arrow_color="black",
    arrow_alpha=0.70,
    arrow_width=0.0045,
    arrow_scale=1.0,

    # point overlays
    source_point_color="black",
    source_point_alpha=0.22,
    source_point_size=5,
    target_point_color="#E76F51",
    target_point_alpha=0.55,
    target_point_size=5,

    # axes
    invert_x=False,
    invert_y=False,
    fig_width=15,
    fig_height=4.8,
    save_path=None,
    show=True,
):
    """
    Three subplots:
      1. Δx heatmap + source slice spots + x-direction arrows
      2. Δy heatmap + source slice spots + y-direction arrows
      3. full 2D movement (source spots + arrows + destination positions)
    """
    # load source / target slices
    ad0, coord0 = load_plot_slice_coords_aligned(
        root=root,
        group_id=group_id,
        sample_id=fixed_id,
        reference_id=reference_id,
        coord_key=coord_key,
        pairwise_subdir=pairwise_subdir,
    )
    ad1, coord1 = load_plot_slice_coords_aligned(
        root=root,
        group_id=group_id,
        sample_id=moving_id,
        reference_id=reference_id,
        coord_key=coord_key,
        pairwise_subdir=pairwise_subdir,
    )

    n0 = min(ad0.n_obs, coord0.shape[0])
    n1 = min(ad1.n_obs, coord1.shape[0])
    coord0 = coord0[:n0, :2]
    coord1 = coord1[:n1, :2]

    # dense grid for heatmaps
    heat_xy, heat_idx_map, heat_xs, heat_ys, _ = build_regular_grid_with_support(
        coord0,
        step=heat_step,
        support_radius=heat_support_radius,
    )
    heat_xy_next, heat_delta, heat_valid = sample_pair_transition_field(
        root=root,
        group_id=group_id,
        fixed_id=fixed_id,
        moving_id=moving_id,
        grid_xy=heat_xy,
        pairwise_subdir=pairwise_subdir,
    )

    dx_grid = grid_values_from_sparse_nodes(heat_idx_map, heat_delta[:, 0])
    dy_grid = grid_values_from_sparse_nodes(heat_idx_map, heat_delta[:, 1])

    dx_grid_up = upsample_nan_grid(dx_grid, factor=heat_upsample, order=3)
    dy_grid_up = upsample_nan_grid(dy_grid, factor=heat_upsample, order=3)

    max_abs_dx = max(float(np.nanmax(np.abs(dx_grid_up))), 1e-6) if np.any(np.isfinite(dx_grid_up)) else 1.0
    max_abs_dy = max(float(np.nanmax(np.abs(dy_grid_up))), 1e-6) if np.any(np.isfinite(dy_grid_up)) else 1.0

    norm_dx = TwoSlopeNorm(vmin=-max_abs_dx, vcenter=0.0, vmax=max_abs_dx)
    norm_dy = TwoSlopeNorm(vmin=-max_abs_dy, vcenter=0.0, vmax=max_abs_dy)

    # coarser grid for arrows
    arrow_xy, _, _, _, _ = build_regular_grid_with_support(
        coord0,
        step=arrow_step,
        support_radius=arrow_support_radius,
    )
    arrow_xy_next, arrow_delta, arrow_valid = sample_pair_transition_field(
        root=root,
        group_id=group_id,
        fixed_id=fixed_id,
        moving_id=moving_id,
        grid_xy=arrow_xy,
        pairwise_subdir=pairwise_subdir,
    )

    arrow_src = arrow_xy[arrow_valid]
    arrow_dst = arrow_xy_next[arrow_valid]
    arrow_dx = arrow_delta[arrow_valid, 0]
    arrow_dy = arrow_delta[arrow_valid, 1]

    # extent for heatmaps
    extent = [heat_xs[0], heat_xs[-1], heat_ys[0], heat_ys[-1]]

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    ax1, ax2, ax3 = axes

    cmap_field = get_cmap(heat_cmap)

    # --------------------------------------------------------
    # 1. Δx heatmap
    # --------------------------------------------------------
    im1 = ax1.imshow(
        dx_grid_up,
        cmap=cmap_field,
        norm=norm_dx,
        origin="lower",
        interpolation="bilinear",
        extent=extent,
        aspect="equal",
    )
    _overlay_source_spots(ax1, coord0, color=source_point_color, alpha=source_point_alpha, s=source_point_size)
    # ax1.quiver(
    #     arrow_src[:, 0], arrow_src[:, 1],
    #     arrow_dx, np.zeros_like(arrow_dx),
    #     angles="xy",
    #     scale_units="xy",
    #     scale=arrow_scale,
    #     color=arrow_color,
    #     alpha=arrow_alpha,
    #     width=arrow_width,
    # )
    _style_2d_axis(ax1, title=rf"$Continous \Delta x$: s{fixed_id}$\rightarrow$s{moving_id}", invert_x=invert_x, invert_y=invert_y)

    # --------------------------------------------------------
    # 2. Δy heatmap
    # --------------------------------------------------------
    im2 = ax2.imshow(
        dy_grid_up,
        cmap=cmap_field,
        norm=norm_dy,
        origin="lower",
        interpolation="bilinear",
        extent=extent,
        aspect="equal",
    )
    _overlay_source_spots(ax2, coord0, color=source_point_color, alpha=source_point_alpha, s=source_point_size)
    # ax2.quiver(
    #     arrow_src[:, 0], arrow_src[:, 1],
    #     np.zeros_like(arrow_dy), arrow_dy,
    #     angles="xy",
    #     scale_units="xy",
    #     scale=arrow_scale,
    #     color=arrow_color,
    #     alpha=arrow_alpha,
    #     width=arrow_width,
    # )
    _style_2d_axis(ax2, title=rf"Continous $\Delta y$: s{fixed_id}$\rightarrow$s{moving_id}", invert_x=invert_x, invert_y=invert_y)

    # --------------------------------------------------------
    # 3. Full 2D movement
    # --------------------------------------------------------
    _overlay_source_spots(ax3, coord0, color=source_point_color, alpha=source_point_alpha, s=source_point_size)
    ax3.quiver(
        arrow_src[:, 0], arrow_src[:, 1],
        arrow_dx, arrow_dy,
        angles="xy",
        scale_units="xy",
        scale=arrow_scale,
        color=arrow_color,
        alpha=0.75,
        width=arrow_width,
    )
    ax3.scatter(
        arrow_dst[:, 0],
        arrow_dst[:, 1],
        s=target_point_size,
        c=target_point_color,
        alpha=target_point_alpha,
        linewidths=0,
        rasterized=True,
    )
    _style_2d_axis(ax3, title=f"Estimated spot-wise displacement", invert_x=invert_x, invert_y=invert_y)

    # colorbars for first two panels
    # cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    # cbar1.set_label(r"$\Delta x$", fontsize=10)
    # cbar1.ax.tick_params(labelsize=9)

    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046,shrink=0.7, pad=0.03)
    cbar2.set_label("Delta in pixels", fontsize=14)
    cbar2.ax.tick_params(labelsize=9)

    # small legend for final panel
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=source_point_color,
               markeredgecolor="none", markersize=6, alpha=0.6, label=f"Slice {fixed_id} spots"),
        Line2D([0, 1], [0, 0], color=arrow_color, lw=1.2, alpha=0.75, label="Field displacement"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=target_point_color,
               markeredgecolor="none", markersize=6, alpha=0.8, label=f"Mapped positions on {moving_id}"),
    ]
    ax3.legend(handles=handles, frameon=False, fontsize=9, loc="upper right",bbox_to_anchor=(1.45, 0.9))

    fig.suptitle("Pairwise transition field visualization", fontsize=17, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        fig.savefig(save_path, dpi=350, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


plot_transition_field_3d(
    root="/media/huifang/data/registration/DLPFC/huifang",
    group_id=1,
    sample_ids=[1, 2],
    gene="layer_guess_reordered",
    reference_id=0,
    coord_key="spatial",
    pairwise_subdir="pairwise_output",
    depth_gap=270.0,
    slice_point_size=5.0,
    slice_alpha=0.55,
    mesh_step=20.0,
    mesh_support_radius=20.0,
    invert_x=True,
    invert_y=True,
    save_path="/media/huifang/data/registration/DLPFC/evaluation/mesh_transition.png",
    show=True,
)
#
# plot_transition_field_pair_views(
#     root="/media/huifang/data/registration/DLPFC/huifang",
#     group_id=1,
#     fixed_id=1,
#     moving_id=2,
#     reference_id=0,
#     coord_key="spatial",
#     pairwise_subdir="pairwise_output",
#     heat_step=10.0,
#     heat_support_radius=10.0,
#     heat_upsample=10,
#     arrow_step=28.0,
#     arrow_support_radius=22.0,
#     invert_x=False,
#     invert_y=True,
#     save_path="/media/huifang/data/registration/DLPFC/evaluation/point_flow.png",
#     show=True,
# )