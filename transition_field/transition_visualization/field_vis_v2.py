import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from visualization_utils import *
from axis.coordinate_utils import find_inverse_warp_coords

from scipy.ndimage import zoom
# ---------------------------------------------------------------------
# This code assumes the following helpers already exist in your script:
#   - find_group_dir
#   - parse_flow_path
#   - inverse_warp_points
#   - build_regular_grid_with_support
#   - grid_values_from_sparse_nodes
#   - sample_pair_transition_field
#   - parse_h5ad_path
#   - parse_network_npz_path
#   - load_npz_coord
# If any of them are not yet defined, keep your original implementations.
# ---------------------------------------------------------------------
from copy import copy

def make_transparent_cmap(cmap_name, bad_rgba=(1, 1, 1, 0)):
    cmap = copy(get_cmap(cmap_name))
    cmap.set_bad(bad_rgba)
    return cmap

def parse_h5ad_path(root, group_id, sample_id):
    return os.path.join(root, "h5ad", f"{group_id}_{sample_id}.h5ad")
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
# ============================================================
# 1) Coordinate loading
# ============================================================
def load_slice_observed_coords(
    root,
    group_id,
    sample_id,
    coord_key="spatial",
):
    """
    Observed (unwarped) coordinates for one slice.
    Priority:
      1) adata.obsm[coord_key]
      2) network_input coord
    """
    h5ad_path = parse_h5ad_path(root, group_id, sample_id)
    adata = sc.read_h5ad(h5ad_path)

    if coord_key in adata.obsm:
        coord = np.asarray(adata.obsm[coord_key], dtype=np.float32)
        return adata, coord[:, :2]

    npz_path = parse_network_npz_path(root, group_id, sample_id)
    coord = load_npz_coord(npz_path)
    return adata, coord[:, :2]


def load_slice_common_frame_coords(
    root,
    group_id,
    sample_id,
    reference_id,
    coord_key="spatial",
    pairwise_subdir="pairwise_output",
):
    """
    Coordinates in the reference frame.

    For reference slice:
      observed / native coordinates

    For non-reference slice:
      1) use precomputed warped_coor if present
      2) otherwise directly map observed coords into reference frame
    """
    h5ad_path = parse_h5ad_path(root, group_id, sample_id)
    adata = sc.read_h5ad(h5ad_path)

    if sample_id == reference_id:
        if coord_key in adata.obsm:
            coord = np.asarray(adata.obsm[coord_key], dtype=np.float32)
            return adata, coord[:, :2]

        npz_path = parse_network_npz_path(root, group_id, sample_id)
        coord = load_npz_coord(npz_path)
        return adata, coord[:, :2]

    group_dir = find_group_dir(root, group_id, pairwise_subdir=pairwise_subdir)
    warped_path = os.path.join(group_dir, f"{reference_id}_{sample_id}_warped_coor.npy")
    if os.path.exists(warped_path):
        coord = np.asarray(np.load(warped_path), dtype=np.float32)
        return adata, coord[:, :2]

    # fallback: map observed coords into reference frame on the fly
    _, coord_obs = load_slice_observed_coords(
        root=root,
        group_id=group_id,
        sample_id=sample_id,
        coord_key=coord_key,
    )
    mapped, _, _ = sample_pair_transition_field(
        root=root,
        group_id=group_id,
        fixed_id=sample_id,
        moving_id=reference_id,
        grid_xy=coord_obs,
        pairwise_subdir=pairwise_subdir,
    )
    return adata, mapped[:, :2]


# ============================================================
# 2) Shared categorical style for spatial domains
# ============================================================
def make_shared_categorical_codes(adatas, obs_key):
    """
    Build one shared category ordering across all provided AnnData objects.
    Returns:
        codes_list : list[np.ndarray]
        cats       : list[str]
    """
    cats = []
    seen = set()

    for ad in adatas:
        if obs_key not in ad.obs.columns:
            raise ValueError(f"obs key '{obs_key}' not found in one of the AnnData objects.")
        s = ad.obs[obs_key].astype(str).fillna("NA")
        for v in s.tolist():
            if v not in seen:
                seen.add(v)
                cats.append(v)

    cat_to_code = {c: i for i, c in enumerate(cats)}
    codes_list = []
    for ad in adatas:
        s = ad.obs[obs_key].astype(str).fillna("NA")
        codes = np.asarray([cat_to_code[v] for v in s.tolist()], dtype=np.float32)
        codes_list.append(codes)

    return codes_list, cats


def make_categorical_style(n_cat, cmap_name="tab20"):
    base = get_cmap(cmap_name, max(n_cat, 3))
    colors = [base(i) for i in range(n_cat)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n_cat + 0.5, 1), n_cat)
    return cmap, norm


def scatter_domains(ax, coords, codes, cmap, norm, s=8, alpha=1.0, zorder=3):
    valid = np.isfinite(codes) & np.all(np.isfinite(coords), axis=1)
    if not np.any(valid):
        return
    ax.scatter(
        coords[valid, 0],
        coords[valid, 1],
        c=codes[valid],
        cmap=cmap,
        norm=norm,
        s=s,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
        zorder=zorder,
    )

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
# ============================================================
# 3) Reference-frame domain contours
# ============================================================
def rasterize_reference_domains(
    coords_ref,
    codes_ref,
    step=6.0,
    support_radius=None,
    pad_ratio=0.03,
):
    """
    Rasterize categorical domains onto a fine regular grid by nearest-neighbor
    assignment inside tissue support.
    """
    grid_xy, idx_map, xs, ys, keep_mask = build_regular_grid_with_support(
        coords_ref,
        step=step,
        support_radius=support_radius,
        pad_ratio=pad_ratio,
    )

    tree = cKDTree(coords_ref)
    dist, nn = tree.query(grid_xy, k=1)

    vals = codes_ref[nn].astype(np.float32)
    domain_grid = grid_values_from_sparse_nodes(idx_map, vals)

    return domain_grid, xs, ys, keep_mask
def grid_values_from_sparse_nodes(idx_map, vals_sparse):
    """
    Put sparse node values back into a dense regular lattice.
    """
    grid = np.full(idx_map.shape, np.nan, dtype=np.float32)
    keep = idx_map >= 0
    grid[keep] = vals_sparse[idx_map[keep]]
    return grid

def draw_domain_contours(
    ax,
    domain_grid,
    xs,
    ys,
    n_cat,
    color="0.35",
    lw=1.0,
    alpha=0.9,
    zorder=1,
):
    levels = np.arange(n_cat - 1, dtype=np.float32) + 0.5
    if len(levels) == 0:
        return

    X, Y = np.meshgrid(xs, ys)
    ax.contour(
        X,
        Y,
        domain_grid,
        levels=levels,
        colors=color,
        linewidths=lw,
        alpha=alpha,
        zorder=zorder,
    )
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

# ============================================================
# 4) Transition-field centerpiece
# ============================================================
def build_transition_field_display(
    root,
    group_id,
    reference_id,
    moving_id,
    coords_ref,
    pairwise_subdir="pairwise_output",
    field_step=8.0,
    field_support_radius=None,
    field_upsample=4,
    smooth_sigma=1.0,
):
    """
    Build a fancy 2D centerpiece for the learned transition field:
    - displacement magnitude as background
    - streamlines / vector flow
    - dx, dy inset maps
    """
    grid_xy, idx_map, xs, ys, keep_mask = build_regular_grid_with_support(
        coords_ref,
        step=field_step,
        support_radius=field_support_radius,
        pad_ratio=0.03,
    )

    mapped_xy, delta_xy, valid = sample_pair_transition_field(
        root=root,
        group_id=group_id,
        fixed_id=reference_id,
        moving_id=moving_id,
        grid_xy=grid_xy,
        pairwise_subdir=pairwise_subdir,
    )

    dx_grid = grid_values_from_sparse_nodes(idx_map, delta_xy[:, 0])
    dy_grid = grid_values_from_sparse_nodes(idx_map, delta_xy[:, 1])
    mag_grid = np.sqrt(dx_grid ** 2 + dy_grid ** 2)

    # display-friendly smooth maps
    dx_disp = upsample_nan_grid(dx_grid, factor=field_upsample, order=3)
    dy_disp = upsample_nan_grid(dy_grid, factor=field_upsample, order=3)
    mag_disp = upsample_nan_grid(mag_grid, factor=field_upsample, order=3)

    # dense arrays for streamplot
    X, Y = np.meshgrid(xs, ys)
    valid_mask = np.isfinite(dx_grid) & np.isfinite(dy_grid)

    pts_valid = np.column_stack([X[valid_mask], Y[valid_mask]])
    dx_valid = dx_grid[valid_mask]
    dy_valid = dy_grid[valid_mask]

    # nearest fill is robust for irregular support
    U = griddata(pts_valid, dx_valid, (X, Y), method="nearest")
    V = griddata(pts_valid, dy_valid, (X, Y), method="nearest")

    if smooth_sigma is not None and smooth_sigma > 0:
        U = gaussian_filter(U, sigma=smooth_sigma)
        V = gaussian_filter(V, sigma=smooth_sigma)

    support = gaussian_filter(valid_mask.astype(np.float32), sigma=1.2) > 0.08
    U[~support] = np.nan
    V[~support] = np.nan

    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    return {
        "grid_xy": grid_xy,
        "mapped_xy": mapped_xy,
        "delta_xy": delta_xy,
        "valid": valid,
        "dx_grid": dx_grid,
        "dy_grid": dy_grid,
        "mag_grid": mag_grid,
        "dx_disp": dx_disp,
        "dy_disp": dy_disp,
        "mag_disp": mag_disp,
        "U": U,
        "V": V,
        "xs": xs,
        "ys": ys,
        "extent": extent,
    }


# ============================================================
# 5) Zoom helpers
# ============================================================
def choose_zoom_center(
    coords_ref,
    field_dict,
    zoom_center_ref=None,
):
    """
    If user does not specify a zoom center in the reference frame, automatically
    choose a point with relatively large displacement near the tissue interior.
    """
    if zoom_center_ref is not None:
        return np.asarray(zoom_center_ref, dtype=np.float32)

    gxy = field_dict["grid_xy"]
    dxy = field_dict["delta_xy"]
    valid = np.all(np.isfinite(gxy), axis=1) & np.all(np.isfinite(dxy), axis=1)
    if not np.any(valid):
        return np.asarray(np.median(coords_ref, axis=0), dtype=np.float32)

    mag = np.linalg.norm(dxy, axis=1)
    x_lo, x_hi = np.quantile(coords_ref[:, 0], [0.20, 0.80])
    y_lo, y_hi = np.quantile(coords_ref[:, 1], [0.20, 0.80])

    cand = valid.copy()
    cand &= (gxy[:, 0] >= x_lo) & (gxy[:, 0] <= x_hi)
    cand &= (gxy[:, 1] >= y_lo) & (gxy[:, 1] <= y_hi)

    if np.any(cand):
        idx = np.argmax(mag * cand.astype(np.float32))
    else:
        idx = np.argmax(mag * valid.astype(np.float32))

    return gxy[idx].astype(np.float32)


def map_one_point_between_slices(
    root,
    group_id,
    fixed_id,
    moving_id,
    xy_ref,
    pairwise_subdir="pairwise_output",
):
    mapped, _, _ = sample_pair_transition_field(
        root=root,
        group_id=group_id,
        fixed_id=fixed_id,
        moving_id=moving_id,
        grid_xy=np.asarray(xy_ref, dtype=np.float32).reshape(1, 2),
        pairwise_subdir=pairwise_subdir,
    )
    return mapped[0]


def add_zoom_rectangle(ax, center_xy, half_size, color="black", lw=1.4):
    rect = Rectangle(
        (center_xy[0] - half_size, center_xy[1] - half_size),
        2 * half_size,
        2 * half_size,
        fill=False,
        edgecolor=color,
        linewidth=lw,
        zorder=6,
    )
    ax.add_patch(rect)


def set_window(ax, center_xy, half_size, invert_x=False, invert_y=False):
    x0, x1 = center_xy[0] - half_size, center_xy[0] + half_size
    y0, y1 = center_xy[1] - half_size, center_xy[1] + half_size
    ax.set_xlim([x1, x0] if invert_x else [x0, x1])
    ax.set_ylim([y1, y0] if invert_y else [y0, y1])


def style_axis(ax, title=None):
    if title is not None:
        ax.set_title(title, fontsize=13, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)


# ============================================================
# 6) Main figure
# ============================================================
def plot_transition_field_layout(
    root="/media/huifang/data/registration/DLPFC/huifang",
    group_id=1,
    reference_id=1,
    moving_id=2,
    obs_key="layer_guess_reordered",
    coord_key="spatial",
    pairwise_subdir="pairwise_output",

    # observed/aligned state appearance
    state_point_size=7,
    state_alpha=1.0,
    cmap_name="tab20",

    # contour scaffold
    contour_step=5.0,
    contour_support_radius=8.0,
    contour_color="0.25",
    contour_lw=1.0,

    # field centerpiece
    field_step=8.0,
    field_support_radius=10.0,
    field_upsample=5,
    field_smooth_sigma=1.0,
    field_cmap="magma",
    stream_color="white",
    stream_density=1.0,
    stream_lw=1.0,

    # dx/dy inset maps
    inset_cmap="coolwarm",

    # zoom
    zoom_center_ref=None,
    zoom_half_size=70.0,

    # axes direction
    invert_x=True,
    invert_y=True,

    # figure
    fig_width=18.5,
    fig_height=5.2,
    save_path=None,
    show=True,
):
    # --------------------------------------------------------
    # Load observed coordinates
    # --------------------------------------------------------
    ad_ref_obs, coord_ref_obs = load_slice_observed_coords(
        root=root, group_id=group_id, sample_id=reference_id, coord_key=coord_key
    )
    ad_mov_obs, coord_mov_obs = load_slice_observed_coords(
        root=root, group_id=group_id, sample_id=moving_id, coord_key=coord_key
    )

    # row-alignment safety
    n_ref_obs = min(ad_ref_obs.n_obs, coord_ref_obs.shape[0])
    n_mov_obs = min(ad_mov_obs.n_obs, coord_mov_obs.shape[0])
    ad_ref_obs = ad_ref_obs[:n_ref_obs].copy()
    ad_mov_obs = ad_mov_obs[:n_mov_obs].copy()
    coord_ref_obs = coord_ref_obs[:n_ref_obs, :2]
    coord_mov_obs = coord_mov_obs[:n_mov_obs, :2]

    # --------------------------------------------------------
    # Load common-frame coordinates
    # --------------------------------------------------------
    ad_ref_aln, coord_ref_aln = load_slice_common_frame_coords(
        root=root,
        group_id=group_id,
        sample_id=reference_id,
        reference_id=reference_id,
        coord_key=coord_key,
        pairwise_subdir=pairwise_subdir,
    )
    ad_mov_aln, coord_mov_aln = load_slice_common_frame_coords(
        root=root,
        group_id=group_id,
        sample_id=moving_id,
        reference_id=reference_id,
        coord_key=coord_key,
        pairwise_subdir=pairwise_subdir,
    )

    n_ref_aln = min(ad_ref_aln.n_obs, coord_ref_aln.shape[0])
    n_mov_aln = min(ad_mov_aln.n_obs, coord_mov_aln.shape[0])
    ad_ref_aln = ad_ref_aln[:n_ref_aln].copy()
    ad_mov_aln = ad_mov_aln[:n_mov_aln].copy()
    coord_ref_aln = coord_ref_aln[:n_ref_aln, :2]
    coord_mov_aln = coord_mov_aln[:n_mov_aln, :2]

    # --------------------------------------------------------
    # Shared categorical coding
    # --------------------------------------------------------
    (codes_ref_obs, codes_mov_obs, codes_ref_aln, codes_mov_aln), cats = make_shared_categorical_codes(
        [ad_ref_obs, ad_mov_obs, ad_ref_aln, ad_mov_aln],
        obs_key=obs_key,
    )
    cmap_dom, norm_dom = make_categorical_style(len(cats), cmap_name=cmap_name)

    # --------------------------------------------------------
    # Reference contours + transition field
    # --------------------------------------------------------
    domain_grid, xs_contour, ys_contour, _ = rasterize_reference_domains(
        coord_ref_aln,
        codes_ref_aln,
        step=contour_step,
        support_radius=contour_support_radius,
        pad_ratio=0.03,
    )

    field_dict = build_transition_field_display(
        root=root,
        group_id=group_id,
        reference_id=reference_id,
        moving_id=moving_id,
        coords_ref=coord_ref_aln,
        pairwise_subdir=pairwise_subdir,
        field_step=field_step,
        field_support_radius=field_support_radius,
        field_upsample=field_upsample,
        smooth_sigma=field_smooth_sigma,
    )

    # --------------------------------------------------------
    # Zoom center
    # --------------------------------------------------------
    zoom_center_ref = choose_zoom_center(
        coords_ref=coord_ref_aln,
        field_dict=field_dict,
        zoom_center_ref=zoom_center_ref,
    )
    zoom_center_mov_obs = map_one_point_between_slices(
        root=root,
        group_id=group_id,
        fixed_id=reference_id,
        moving_id=moving_id,
        xy_ref=zoom_center_ref,
        pairwise_subdir=pairwise_subdir,
    )

    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        1, 3,
        width_ratios=[1.40, 1.55, 1.40],
        wspace=0.10,
        left=0.03, right=0.99, top=0.88, bottom=0.08
    )

    # observed block
    gs_left = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[0],
        height_ratios=[1.0, 0.42],
        hspace=0.10, wspace=0.06
    )
    ax_obs_ref = fig.add_subplot(gs_left[0, 0])
    ax_obs_mov = fig.add_subplot(gs_left[0, 1])
    # ax_obs_ref_zoom = fig.add_subplot(gs_left[1, 0])
    # ax_obs_mov_zoom = fig.add_subplot(gs_left[1, 1])

    # centerpiece
    ax_field = fig.add_subplot(gs[1])

    # aligned block
    gs_right = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[2],
        height_ratios=[1.0, 0.42],
        hspace=0.10, wspace=0.06
    )
    ax_aln_ref = fig.add_subplot(gs_right[0, 0])
    ax_aln_mov = fig.add_subplot(gs_right[0, 1])
    # ax_aln_ref_zoom = fig.add_subplot(gs_right[1, 0])
    # ax_aln_mov_zoom = fig.add_subplot(gs_right[1, 1])

    # --------------------------------------------------------
    # Left: observed states
    # --------------------------------------------------------
    scatter_domains(ax_obs_ref, coord_ref_obs, codes_ref_obs, cmap_dom, norm_dom,
                    s=state_point_size, alpha=state_alpha)
    scatter_domains(ax_obs_mov, coord_mov_obs, codes_mov_obs, cmap_dom, norm_dom,
                    s=state_point_size, alpha=state_alpha)
    #
    # add_zoom_rectangle(ax_obs_ref, zoom_center_ref, zoom_half_size, color="black", lw=1.5)
    # add_zoom_rectangle(ax_obs_mov, zoom_center_mov_obs, zoom_half_size, color="black", lw=1.5)

    style_axis(ax_obs_ref, title="Reference anchor")
    style_axis(ax_obs_mov, title="Moving anchor")
    if invert_x:
        ax_obs_ref.invert_xaxis()
        ax_obs_mov.invert_xaxis()
    if invert_y:
        ax_obs_ref.invert_yaxis()
        ax_obs_mov.invert_yaxis()

    # observed zooms
    # scatter_domains(ax_obs_ref_zoom, coord_ref_obs, codes_ref_obs, cmap_dom, norm_dom,
    #                 s=state_point_size + 2, alpha=state_alpha)
    # scatter_domains(ax_obs_mov_zoom, coord_mov_obs, codes_mov_obs, cmap_dom, norm_dom,
    #                 s=state_point_size + 2, alpha=state_alpha)
    # set_window(ax_obs_ref_zoom, zoom_center_ref, zoom_half_size, invert_x=invert_x, invert_y=invert_y)
    # set_window(ax_obs_mov_zoom, zoom_center_mov_obs, zoom_half_size, invert_x=invert_x, invert_y=invert_y)
    # style_axis(ax_obs_ref_zoom, title="Local patch")
    # style_axis(ax_obs_mov_zoom, title="Corresponding patch")

    # --------------------------------------------------------
    # Middle: transition field centerpiece
    # --------------------------------------------------------
    draw_domain_contours(
        ax_field,
        domain_grid,
        xs_contour,
        ys_contour,
        n_cat=len(cats),
        color=contour_color,
        lw=contour_lw,
        alpha=0.70,
        zorder=1,
    )

    mag_disp = field_dict["mag_disp"]
    extent = field_dict["extent"]
    vmax_mag = np.nanpercentile(mag_disp[np.isfinite(mag_disp)], 99) if np.any(np.isfinite(mag_disp)) else 1.0

    mag_masked = np.ma.masked_invalid(mag_disp)
    field_cmap_obj = make_transparent_cmap(field_cmap)

    im = ax_field.imshow(
        mag_masked,
        cmap=field_cmap_obj,
        norm=Normalize(vmin=0.0, vmax=max(vmax_mag, 1e-6)),
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        alpha=0.92,
        zorder=2,
    )
    U = np.ma.masked_invalid(field_dict["U"])
    V = np.ma.masked_invalid(field_dict["V"])
    ax_field.streamplot(
        field_dict["xs"],
        field_dict["ys"],
        U,
        V,
        color=stream_color,
        density=stream_density,
        linewidth=stream_lw,
        arrowsize=0.95,
        zorder=3,
    )

    # main zoom rectangle on field / common frame
    # add_zoom_rectangle(ax_field, zoom_center_ref, zoom_half_size, color="white", lw=1.8)

    style_axis(ax_field, title="Continuous tissue-state transition field")
    if invert_x:
        ax_field.invert_xaxis()
    if invert_y:
        ax_field.invert_yaxis()

    # dx / dy insets
    dx_disp = field_dict["dx_disp"]
    dy_disp = field_dict["dy_disp"]

    max_abs_dx = np.nanpercentile(np.abs(dx_disp[np.isfinite(dx_disp)]), 99) if np.any(np.isfinite(dx_disp)) else 1.0
    max_abs_dy = np.nanpercentile(np.abs(dy_disp[np.isfinite(dy_disp)]), 99) if np.any(np.isfinite(dy_disp)) else 1.0
    norm_dx = TwoSlopeNorm(vmin=-max_abs_dx, vcenter=0.0, vmax=max_abs_dx)
    norm_dy = TwoSlopeNorm(vmin=-max_abs_dy, vcenter=0.0, vmax=max_abs_dy)

    ax_dx = inset_axes(ax_field, width="26%", height="26%", loc="upper right", borderpad=1.0)
    ax_dy = inset_axes(ax_field, width="26%", height="26%", loc="lower right", borderpad=1.0)

    ax_dx.imshow(
        dx_disp,
        cmap=inset_cmap,
        norm=norm_dx,
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        alpha=np.where(np.isfinite(dx_disp), 1.0, 0.0),
    )
    style_axis(ax_dx, title=r"$\Delta x$")
    if invert_x:
        ax_dx.invert_xaxis()
    if invert_y:
        ax_dx.invert_yaxis()

    dy_masked = np.ma.masked_invalid(dy_disp)
    inset_cmap_dy = make_transparent_cmap(inset_cmap)

    ax_dy.imshow(
        dy_masked,
        cmap=inset_cmap_dy,
        norm=norm_dy,
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        alpha=1.0,
    )
    style_axis(ax_dy, title=r"$\Delta y$")
    if invert_x:
        ax_dy.invert_xaxis()
    if invert_y:
        ax_dy.invert_yaxis()

    # --------------------------------------------------------
    # Right: common-frame aligned states
    # --------------------------------------------------------
    draw_domain_contours(ax_aln_ref, domain_grid, xs_contour, ys_contour, len(cats),
                         color=contour_color, lw=contour_lw, alpha=0.60, zorder=1)
    draw_domain_contours(ax_aln_mov, domain_grid, xs_contour, ys_contour, len(cats),
                         color=contour_color, lw=contour_lw, alpha=0.60, zorder=1)

    scatter_domains(ax_aln_ref, coord_ref_aln, codes_ref_aln, cmap_dom, norm_dom,
                    s=state_point_size, alpha=state_alpha, zorder=3)
    scatter_domains(ax_aln_mov, coord_mov_aln, codes_mov_aln, cmap_dom, norm_dom,
                    s=state_point_size, alpha=state_alpha, zorder=3)

    # add_zoom_rectangle(ax_aln_ref, zoom_center_ref, zoom_half_size, color="black", lw=1.5)
    # add_zoom_rectangle(ax_aln_mov, zoom_center_ref, zoom_half_size, color="black", lw=1.5)

    style_axis(ax_aln_ref, title="Reference state")
    style_axis(ax_aln_mov, title="Mapped state")
    if invert_x:
        ax_aln_ref.invert_xaxis()
        ax_aln_mov.invert_xaxis()
    if invert_y:
        ax_aln_ref.invert_yaxis()
        ax_aln_mov.invert_yaxis()

    # # aligned zooms
    # draw_domain_contours(ax_aln_ref_zoom, domain_grid, xs_contour, ys_contour, len(cats),
    #                      color=contour_color, lw=contour_lw, alpha=0.80, zorder=1)
    # draw_domain_contours(ax_aln_mov_zoom, domain_grid, xs_contour, ys_contour, len(cats),
    #                      color=contour_color, lw=contour_lw, alpha=0.80, zorder=1)
    # scatter_domains(ax_aln_ref_zoom, coord_ref_aln, codes_ref_aln, cmap_dom, norm_dom,
    #                 s=state_point_size + 2, alpha=state_alpha, zorder=3)
    # scatter_domains(ax_aln_mov_zoom, coord_mov_aln, codes_mov_aln, cmap_dom, norm_dom,
    #                 s=state_point_size + 2, alpha=state_alpha, zorder=3)
    # set_window(ax_aln_ref_zoom, zoom_center_ref, zoom_half_size, invert_x=invert_x, invert_y=invert_y)
    # set_window(ax_aln_mov_zoom, zoom_center_ref, zoom_half_size, invert_x=invert_x, invert_y=invert_y)
    # style_axis(ax_aln_ref_zoom, title="Preserved patch")
    # style_axis(ax_aln_mov_zoom, title="Mapped patch")

    # --------------------------------------------------------
    # Group headers
    # --------------------------------------------------------
    fig.text(0.14, 0.95, "Observed tissue states", ha="center", va="center",
             fontsize=20, fontweight="bold")
    fig.text(0.50, 0.95, "Continuous tissue-state transition field", ha="center", va="center",
             fontsize=20, fontweight="bold")
    fig.text(0.86, 0.95, "Common-frame aligned states", ha="center", va="center",
             fontsize=20, fontweight="bold")
    fig.text(0.01, 0.95, "a.", ha="left", va="center", fontsize=26, fontweight="bold")


    # center colorbar
    cax = fig.add_axes([0.61, 0.13, 0.010, 0.22])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Field magnitude", fontsize=11)
    cb.ax.tick_params(labelsize=9)

    if save_path is not None:
        fig.savefig(save_path, dpi=350, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# 7) Example call
# ============================================================
plot_transition_field_layout(
    root="/media/huifang/data/registration/DLPFC/huifang",
    group_id=1,
    reference_id=1,
    moving_id=2,
    obs_key="layer_guess_reordered",
    coord_key="spatial",
    pairwise_subdir="pairwise_output",

    state_point_size=8,
    state_alpha=1.0,
    cmap_name="tab10",

    contour_step=5.0,
    contour_support_radius=10.0,
    contour_color="0.20",
    contour_lw=2.0,

    field_step=8.0,
    field_support_radius=10.0,
    field_upsample=5,
    field_smooth_sigma=1.0,
    field_cmap="magma",
    stream_color="white",
    stream_density=1.0,
    stream_lw=0.95,

    zoom_center_ref=None,   # auto-pick; or set manually as (x, y)
    zoom_half_size=40.0,

    invert_x=False,
    invert_y=True,

    fig_width=18.5,
    fig_height=5.2,
    save_path="/media/huifang/data/registration/DLPFC/evaluation/transition_field_layout_group1_s1_s2.png",
    show=True,
)