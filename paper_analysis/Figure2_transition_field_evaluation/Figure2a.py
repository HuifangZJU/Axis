"""Plot AXIS registration results as three independent figures.

Figure 1: observed PCP4 signals over H&E images with value-scaled RGBA opacity.
Figure 2: reverse-direction continuous transition field with separate dx/dy maps.
Figure 3: common-frame PCP4 alignment with metric_utils alpha-shape boundaries
          and a zoomed overlap panel.
"""

import os
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.sparse import issparse
from scipy.spatial import cKDTree
from shapely.affinity import translate as translate_geometry
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

from axis.coordinate_utils import find_inverse_warp_coords
from utils_metrics import create_boundary_mask


# -----------------------------------------------------------------------------
# I/O and coordinate mapping
# -----------------------------------------------------------------------------
def find_group_dir(root, group_id, pairwise_subdir="pairwise_output"):
    candidates = [
        os.path.join(root, pairwise_subdir, f"group_{group_id}"),
        os.path.join(root, f"group_{group_id}"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Cannot find group {group_id}. Tried: {candidates}")


def load_slice(root, group_id, sample_id, coord_key="spatial"):
    h5ad_path = os.path.join(root, "h5ad", f"{group_id}_{sample_id}.h5ad")
    adata = sc.read_h5ad(h5ad_path)

    if coord_key in adata.obsm:
        coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    else:
        npz_path = os.path.join(
            root, "network_input", f"{group_id}_{sample_id}_validation.npz"
        )
        with np.load(npz_path) as data:
            if "coord" not in data:
                raise KeyError(f"'coord' not found in {npz_path}")
            coords = np.asarray(data["coord"], dtype=np.float32)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected coordinates with shape (N, 2+), got {coords.shape}")

    n = min(adata.n_obs, coords.shape[0])
    return adata[:n].copy(), coords[:n, :2]


def load_he_image(root, group_id, sample_id):
    image_path = os.path.join(
        root, "network_input", f"{group_id}_{sample_id}_image.png"
    )
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"H&E image not found: {image_path}")
    return plt.imread(image_path)


def get_gene_signal(adata, gene, layer=None, use_raw=False, log1p=True):
    if use_raw and layer is not None:
        raise ValueError("gene_layer and gene_use_raw cannot be used together")
    source = adata.raw if use_raw and adata.raw is not None else adata
    var_names = source.var_names
    if gene not in var_names:
        raise KeyError(
            f"Gene '{gene}' was not found in "
            f"{'adata.raw.var_names' if source is adata.raw else 'adata.var_names'}"
        )

    gene_idx = var_names.get_loc(gene)
    matrix = source.X if layer is None else adata.layers[layer]
    values = matrix[:, gene_idx]
    if issparse(values):
        values = values.toarray()
    values = np.asarray(values).reshape(-1).astype(np.float32)
    return np.log1p(np.maximum(values, 0)) if log1p else values


def inverse_warp_points(points_xy, flow_hw2):
    mapped = find_inverse_warp_coords(points_xy, flow_hw2)
    if isinstance(mapped, (tuple, list)):
        mapped = mapped[0]
    mapped = np.asarray(mapped, dtype=np.float32)
    if mapped.ndim != 2 or mapped.shape[1] < 2:
        raise ValueError(f"Unexpected inverse-warp output shape: {mapped.shape}")
    return mapped[:, :2]


def sample_transition(
    root,
    group_id,
    fixed_id,
    moving_id,
    query_xy,
    pairwise_subdir="pairwise_output",
):
    group_dir = find_group_dir(root, group_id, pairwise_subdir)
    flow_path = os.path.join(group_dir, f"{moving_id}_{fixed_id}_flow.npy")
    if not os.path.exists(flow_path):
        raise FileNotFoundError(flow_path)

    flow = np.load(flow_path)
    query_xy = np.asarray(query_xy, dtype=np.float32)
    mapped = inverse_warp_points(query_xy, flow)

    height, width, _ = flow.shape
    valid = np.all(np.isfinite(mapped), axis=1)
    valid &= (
        (mapped[:, 0] >= 0)
        & (mapped[:, 0] < width)
        & (mapped[:, 1] >= 0)
        & (mapped[:, 1] < height)
    )

    mapped_valid = np.full_like(query_xy, np.nan)
    mapped_valid[valid] = mapped[valid]
    return mapped_valid, mapped_valid - query_xy, valid


def load_moving_aligned_coords(
    root,
    group_id,
    reference_id,
    moving_id,
    moving_observed,
    pairwise_subdir="pairwise_output",
):
    group_dir = find_group_dir(root, group_id, pairwise_subdir)
    warped_path = os.path.join(
        group_dir, f"{reference_id}_{moving_id}_warped_coor.npy"
    )
    if os.path.exists(warped_path):
        return np.asarray(np.load(warped_path), dtype=np.float32)[:, :2]

    mapped, _, _ = sample_transition(
        root,
        group_id,
        fixed_id=moving_id,
        moving_id=reference_id,
        query_xy=moving_observed,
        pairwise_subdir=pairwise_subdir,
    )
    return mapped


# -----------------------------------------------------------------------------
# Transition field
# -----------------------------------------------------------------------------
def regular_tissue_grid(coords, step, support_radius=None, pad_ratio=0.03):
    coords = np.asarray(coords, dtype=np.float32)
    coords = coords[np.all(np.isfinite(coords), axis=1)]
    if coords.size == 0:
        raise ValueError("No finite coordinates available")

    support_radius = 0.8 * step if support_radius is None else support_radius
    coord_min, coord_max = coords.min(0), coords.max(0)
    pad = (coord_max - coord_min) * pad_ratio

    xs = np.arange(coord_min[0] - pad[0], coord_max[0] + pad[0] + step, step)
    ys = np.arange(coord_min[1] - pad[1], coord_max[1] + pad[1] + step, step)
    xx, yy = np.meshgrid(xs, ys)
    all_nodes = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    keep = cKDTree(coords).query(all_nodes, k=1)[0] <= support_radius
    index_grid = np.full(xx.shape, -1, dtype=np.int32)
    index_grid.ravel()[np.flatnonzero(keep)] = np.arange(np.count_nonzero(keep))
    return all_nodes[keep], index_grid, xs, ys


def values_to_grid(index_grid, values):
    grid = np.full(index_grid.shape, np.nan, dtype=np.float32)
    keep = index_grid >= 0
    grid[keep] = values[index_grid[keep]]
    return grid


def build_transition_field(
    root,
    group_id,
    source_id,
    target_id,
    source_xy,
    pairwise_subdir,
    step,
    support_radius,
    smooth_sigma,
):
    """Build a transition field sampled on the source tissue support.

    This function is used for Figure 2, where we visualize the transition
    from the moving anchor to the reference anchor. Therefore, the regular
    grid and the background support must be derived from the source slice
    (the moving anchor), not the target slice.
    """
    grid_xy, index_grid, xs, ys = regular_tissue_grid(
        source_xy, step, support_radius
    )
    mapped, delta, valid = sample_transition(
        root,
        group_id,
        fixed_id=source_id,
        moving_id=target_id,
        query_xy=grid_xy,
        pairwise_subdir=pairwise_subdir,
    )

    dx = values_to_grid(index_grid, delta[:, 0])
    dy = values_to_grid(index_grid, delta[:, 1])
    magnitude = np.hypot(dx, dy)

    xx, yy = np.meshgrid(xs, ys)
    support = np.isfinite(dx) & np.isfinite(dy)
    if not np.any(support):
        raise ValueError("No valid transition-field samples on tissue support")

    points = np.column_stack([xx[support], yy[support]])
    u = griddata(points, dx[support], (xx, yy), method="nearest")
    v = griddata(points, dy[support], (xx, yy), method="nearest")

    if smooth_sigma > 0:
        u = gaussian_filter(u, smooth_sigma)
        v = gaussian_filter(v, smooth_sigma)

    smooth_support = gaussian_filter(support.astype(float), 1.2) > 0.08
    u[~smooth_support] = np.nan
    v[~smooth_support] = np.nan

    return {
        "dx": dx,
        "dy": dy,
        "magnitude": magnitude,
        "u": u,
        "v": v,
        "xs": xs,
        "ys": ys,
        "extent": [xs[0], xs[-1], ys[0], ys[-1]],
    }


# -----------------------------------------------------------------------------
# Alpha-shape domain boundaries
# -----------------------------------------------------------------------------
def iter_polygon_exteriors(boundary):
    """Yield exterior coordinate arrays from supported Shapely geometries."""
    if isinstance(boundary, Polygon):
        yield np.asarray(boundary.exterior.coords)
    elif isinstance(boundary, MultiPolygon):
        for polygon in boundary.geoms:
            yield np.asarray(polygon.exterior.coords)
    elif isinstance(boundary, GeometryCollection):
        for geometry in boundary.geoms:
            if isinstance(geometry, Polygon):
                yield np.asarray(geometry.exterior.coords)
            elif isinstance(geometry, MultiPolygon):
                for polygon in geometry.geoms:
                    yield np.asarray(polygon.exterior.coords)


def _domain_local_frame(coords, padding=2.0):
    """Translate one domain to a compact positive frame for mask rasterization."""
    coords = np.asarray(coords, dtype=float)
    offset = np.floor(coords.min(axis=0) - padding)
    local_coords = coords - offset
    max_xy = np.ceil(local_coords.max(axis=0) + padding).astype(int)
    mask_shape = (max(max_xy[1] + 1, 2), max(max_xy[0] + 1, 2))
    return local_coords, mask_shape, offset


def build_domain_boundaries(
    coords,
    labels,
    alpha=0.01,
    remove_isolated=True,
    neighbor_radius=None,
    min_neighbors=3,
    radius_scale=1.8,
):
    """Build closed domain geometries using metric_utils.create_boundary_mask.

    Each domain is temporarily translated into a compact local mask frame. This
    avoids allocating a full-slide mask for every domain and prevents clipping
    when common-frame coordinates do not begin at zero. The returned geometry is
    translated back to the original common-frame coordinates for plotting.
    """
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels)
    regions = {}

    for label in np.unique(labels):
        region_coords = coords[labels == label]
        region_coords = region_coords[np.all(np.isfinite(region_coords), axis=1)]
        if region_coords.shape[0] < 3:
            continue

        local_coords, mask_shape, offset = _domain_local_frame(region_coords)
        boundary, mask = create_boundary_mask(
            local_coords,
            shape=mask_shape,
            alpha=alpha,
            remove_isolated=remove_isolated,
            neighbor_radius=neighbor_radius,
            min_neighbors=min_neighbors,
            radius_scale=radius_scale,
        )
        if boundary is None or getattr(boundary, "is_empty", False):
            continue

        regions[label] = {
            "boundary": translate_geometry(
                boundary,
                xoff=float(offset[0]),
                yoff=float(offset[1]),
            )

        }

    return regions


def draw_boundaries(ax, regions, color_map, linewidth=1.5, alpha=0.9):
    """Draw closed alpha-shape exteriors returned by build_domain_boundaries."""
    for label, region in regions.items():
        boundary = region["boundary"]
        color = color_map[label]
        for exterior in iter_polygon_exteriors(boundary):
            ax.plot(
                exterior[:, 0],
                exterior[:, 1],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=2,
            )


# -----------------------------------------------------------------------------
# Shared preparation
# -----------------------------------------------------------------------------
def prepare_plot_data(
    root,
    group_id,
    reference_id,
    moving_id,
    obs_key,
    gene="PCP4",
    gene_layer=None,
    gene_use_raw=False,
    gene_log1p=True,
    coord_key="spatial",
    pairwise_subdir="pairwise_output",
    field_step=3.0,
    field_support_radius=10.0,
    field_smooth_sigma=1.0,
    boundary_cmap="tab10",
):
    ad_ref, ref_observed = load_slice(root, group_id, reference_id, coord_key)
    ad_mov, mov_observed = load_slice(root, group_id, moving_id, coord_key)
    mov_aligned = load_moving_aligned_coords(
        root,
        group_id,
        reference_id,
        moving_id,
        mov_observed,
        pairwise_subdir,
    )

    n_mov = min(ad_mov.n_obs, len(mov_observed), len(mov_aligned))
    ad_mov = ad_mov[:n_mov].copy()
    mov_observed = mov_observed[:n_mov]
    mov_aligned = mov_aligned[:n_mov]

    if obs_key not in ad_ref.obs or obs_key not in ad_mov.obs:
        raise KeyError(f"obs['{obs_key}'] must exist in both AnnData objects")

    ref_labels = ad_ref.obs[obs_key].astype("string").fillna("NA").astype(str).to_numpy()
    mov_labels = ad_mov.obs[obs_key].astype("string").fillna("NA").astype(str).to_numpy()
    categories = list(dict.fromkeys(np.concatenate([ref_labels, mov_labels])))
    palette = get_cmap(boundary_cmap, max(len(categories), 3))
    boundary_colors = {label: palette(i) for i, label in enumerate(categories)}

    ref_signal = get_gene_signal(ad_ref, gene, gene_layer, gene_use_raw, gene_log1p)
    mov_signal = get_gene_signal(ad_mov, gene, gene_layer, gene_use_raw, gene_log1p)

    # Figure 2 visualizes the transition from the moving anchor to the
    # reference anchor, so the field is sampled on the moving tissue support.
    field = build_transition_field(
        root,
        group_id,
        moving_id,
        reference_id,
        mov_observed,
        pairwise_subdir,
        field_step,
        field_support_radius,
        field_smooth_sigma,
    )

    return {
        "gene": gene,
        "gene_scale_label": f"log1p {gene} expression" if gene_log1p else f"{gene} expression",
        "ref_observed": ref_observed,
        "mov_observed": mov_observed,
        "ref_aligned": ref_observed.copy(),
        "mov_aligned": mov_aligned,
        "ref_signal": ref_signal,
        "mov_signal": mov_signal,
        "ref_labels": ref_labels,
        "mov_labels": mov_labels,
        "ref_image": load_he_image(root, group_id, reference_id),
        "mov_image": load_he_image(root, group_id, moving_id),
        "boundary_colors": boundary_colors,
        "field": field,
    }


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def style_axis(ax, title, invert_x, invert_y):
    ax.set_title(title, fontsize=13, pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()


def transparent_cmap(name):
    cmap = copy(get_cmap(name))
    cmap.set_bad((1, 1, 1, 0))
    return cmap


def finite_percentile(values, percentile, default=1.0):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return max(float(np.percentile(values, percentile)), 1e-8) if values.size else default


def gene_norm(*signals, lower_percentile=0, upper_percentile=99):
    values = np.concatenate([np.asarray(x)[np.isfinite(x)] for x in signals])
    if values.size == 0:
        return Normalize(0, 1)
    vmin = float(np.percentile(values, lower_percentile))
    vmax = float(np.percentile(values, upper_percentile))
    if vmax <= vmin:
        vmax = vmin + 1e-8
    return Normalize(vmin, vmax)


def scatter_gene(
    ax,
    coords,
    signal,
    norm,
    marker="o",
    size=12,
    alpha=0.9,
    alpha_by_value=False,
    edgecolors="none",
    linewidths=0,
    label=None,
    zorder=3,
):
    """Scatter a gene signal using explicit per-point RGBA colors.

    Explicit RGBA colors avoid a Matplotlib/TkAgg incompatibility in older
    Matplotlib releases: a colorbar cannot safely inherit a per-point alpha
    array from a scatter collection. When ``alpha_by_value=True``, zero-read
    spots are fully transparent and opacity increases with normalized signal.
    """
    coords = np.asarray(coords)
    signal = np.asarray(signal).reshape(-1)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must have shape (N, 2+), got {coords.shape}")
    if coords.shape[0] != signal.shape[0]:
        raise ValueError(
            "Coordinate/gene length mismatch: "
            f"{coords.shape[0]} coordinates versus {signal.shape[0]} gene values"
        )

    valid = np.all(np.isfinite(coords[:, :2]), axis=1) & np.isfinite(signal)
    coords_valid = coords[valid, :2]
    signal_valid = signal[valid]

    scaled = np.clip(np.asarray(norm(signal_valid), dtype=float), 0.0, 1.0)
    facecolors = np.asarray(get_cmap("Reds")(scaled), dtype=float)

    if alpha_by_value:
        point_alpha = float(alpha) * scaled
        point_alpha[signal_valid <= 0] = 0.0
    else:
        point_alpha = np.full(signal_valid.shape, float(alpha), dtype=float)

    facecolors[:, 3] = point_alpha

    return ax.scatter(
        coords_valid[:, 0],
        coords_valid[:, 1],
        c=facecolors,
        marker=marker,
        s=size,
        edgecolors=edgecolors,
        linewidths=linewidths,
        rasterized=True,
        label=label,
        zorder=zorder,
    )


def make_gene_mappable(norm):
    """Create a colorbar mappable that carries no per-point alpha array."""
    mappable = ScalarMappable(norm=norm, cmap=get_cmap("Reds"))
    mappable.set_array(np.array([norm.vmin, norm.vmax], dtype=float))
    return mappable


def set_common_limits(axes, *coords_arrays, pad_ratio=0.03):
    finite = [x[np.all(np.isfinite(x), axis=1)] for x in coords_arrays]
    coords = np.vstack([x for x in finite if len(x)])
    low, high = coords.min(0), coords.max(0)
    pad = np.maximum((high - low) * pad_ratio, 1e-6)
    for ax in axes:
        ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
        ax.set_ylim(low[1] - pad[1], high[1] + pad[1])


def finish_figure(fig, save_path, show, dpi):
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 1: observed PCP4 over H&E
# -----------------------------------------------------------------------------
def plot_observed_gene_on_he(
    data,
    point_size=12,
    point_alpha=0.9,
    image_alpha=1.0,
    invert_x=False,
    invert_y=True,
    figsize=(11, 5),
    save_path=None,
    show=True,
    dpi=350,
):
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    norm = gene_norm(data["ref_signal"], data["mov_signal"])

    for ax, image, coords, signal, title in [
        (axes[0], data["ref_image"], data["ref_observed"], data["ref_signal"], "Reference anchor"),
        (axes[1], data["mov_image"], data["mov_observed"], data["mov_signal"], "Moving anchor"),
    ]:
        height, width = image.shape[:2]
        ax.imshow(
            image,
            origin="lower",
            extent=[0, width, 0, height],
            alpha=image_alpha,
            zorder=0,
        )
        scatter_gene(
            ax,
            coords,
            signal,
            norm,
            size=point_size,
            alpha=point_alpha,
            alpha_by_value=True,
        )
        style_axis(ax, title, invert_x, invert_y)

    fig.suptitle(f"Observed spatial {data['gene']} signal", fontsize=18, fontweight="bold")
    colorbar = fig.colorbar(
        make_gene_mappable(norm), ax=axes, shrink=0.72, pad=0.02
    )
    colorbar.set_label(data["gene_scale_label"], fontsize=10)
    finish_figure(fig, save_path, show, dpi)
    return fig


# -----------------------------------------------------------------------------
# Figure 2: reverse transition field, no contours
# -----------------------------------------------------------------------------
def plot_continuous_transition_field(
    data,
    field_cmap="magma",
    delta_cmap="coolwarm",
    stream_color="white",
    stream_density=0.5,
    stream_linewidth=0.95,
    invert_x=False,
    invert_y=True,
    figsize=(13, 5.8),
    save_path=None,
    show=True,
    dpi=350,
):
    field = data["field"]
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[2.25, 0.06, 1.0, 0.06])

    ax_field = fig.add_subplot(gs[:, 0])
    cax_mag = fig.add_subplot(gs[:, 1])
    ax_dx = fig.add_subplot(gs[0, 2])
    ax_dy = fig.add_subplot(gs[1, 2])
    cax_delta = fig.add_subplot(gs[:, 3])

    magnitude_limit = finite_percentile(field["magnitude"], 99)
    im_mag = ax_field.imshow(
        np.ma.masked_invalid(field["magnitude"]),
        origin="lower",
        extent=field["extent"],
        cmap=transparent_cmap(field_cmap),
        norm=Normalize(0, magnitude_limit),
        interpolation="bilinear",
    )

    # The field is built directly from moving anchor -> reference anchor,
    # so the displayed streamline direction should follow (u, v) as-is.
    ax_field.streamplot(
        field["xs"],
        field["ys"],
        np.ma.masked_invalid(field["u"]),
        np.ma.masked_invalid(field["v"]),
        color=stream_color,
        density=stream_density,
        linewidth=stream_linewidth,
        arrowsize=0.95,
    )
    style_axis(ax_field, "Continuous tissue-state transition field", invert_x, invert_y)
    cb_mag = fig.colorbar(im_mag, cax=cax_mag,shrink=0.6)
    cb_mag.set_label("Displacement magnitude", fontsize=10)

    signed_values = np.concatenate(
        [
            np.abs(field["dx"][np.isfinite(field["dx"])]),
            np.abs(field["dy"][np.isfinite(field["dy"])]),
        ]
    )
    delta_limit = finite_percentile(signed_values, 99)
    delta_norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0, vmax=delta_limit)

    im_dx = ax_dx.imshow(
        np.ma.masked_invalid(field["dx"]),
        origin="lower",
        extent=field["extent"],
        cmap=transparent_cmap(delta_cmap),
        norm=delta_norm,
        interpolation="bilinear",
    )
    ax_dy.imshow(
        np.ma.masked_invalid(field["dy"]),
        origin="lower",
        extent=field["extent"],
        cmap=transparent_cmap(delta_cmap),
        norm=delta_norm,
        interpolation="bilinear",
    )
    style_axis(ax_dx, r"Horizontal displacement, $\Delta x$", invert_x, invert_y)
    style_axis(ax_dy, r"Vertical displacement, $\Delta y$", invert_x, invert_y)

    cb_delta = fig.colorbar(im_dx, cax=cax_delta,shrink=0.6)
    cb_delta.set_label("Signed displacement", fontsize=10)
    finish_figure(fig, save_path, show, dpi)
    return fig


# -----------------------------------------------------------------------------
# Figure 3: common-frame PCP4 alignment and zoomed overlap
# -----------------------------------------------------------------------------
def choose_zoom_window(ref_xy, mov_xy, zoom_center=None, zoom_half_width=75.0, zoom_half_height=None):
    zoom_half_height = zoom_half_width if zoom_half_height is None else zoom_half_height
    if zoom_center is None:
        common = np.vstack([
            ref_xy[np.all(np.isfinite(ref_xy), axis=1)],
            mov_xy[np.all(np.isfinite(mov_xy), axis=1)],
        ])
        zoom_center = np.nanmedian(common, axis=0)
    return np.asarray(zoom_center, dtype=float), float(zoom_half_width), float(zoom_half_height)


def apply_zoom(ax, center, half_width, half_height, invert_x, invert_y):
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_height, center[1] + half_height)
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()


def plot_common_frame_aligned_states(
    data,
    ref_marker="o",
    moving_marker="^",
    ref_point_size=14,
    moving_point_size=18,
    ref_alpha=0.75,
    moving_alpha=0.75,
    boundary_alpha_shape=0.01,
    boundary_remove_isolated=True,
    boundary_neighbor_radius=None,
    boundary_min_neighbors=3,
    boundary_radius_scale=1.8,
    boundary_linewidth=3,
    boundary_alpha=0.9,
    zoom_center=None,
    zoom_half_width=75.0,
    zoom_half_height=None,
    invert_x=False,
    invert_y=True,
    figsize=(16, 5.2),
    save_path=None,
    show=True,
    dpi=350,
):
    """Plot reference, mapped, and zoomed common-frame tissue states.

    A single flat GridSpec is used intentionally. Older Matplotlib/TkAgg
    versions can fail when constrained_layout handles a colorbar spanning
    axes from different nested GridSpec levels.
    """
    fig = plt.figure(figsize=figsize, constrained_layout=False)

    # One flat layout: reference | mapped | legend/zoom | colorbar.
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        width_ratios=[1.0, 1.0, 0.80, 0.035],
        height_ratios=[0.13, 0.87],
        left=0.035,
        right=0.965,
        bottom=0.09,
        top=0.86,
        wspace=0.10,
        hspace=0.03,
    )

    ax_ref = fig.add_subplot(gs[:, 0])
    ax_mapped = fig.add_subplot(gs[:, 1])
    ax_zoom_legend = fig.add_subplot(gs[0, 2])
    ax_zoom = fig.add_subplot(gs[1, 2])
    cax = fig.add_subplot(gs[:, 3])
    ax_zoom_legend.axis("off")

    norm = gene_norm(data["ref_signal"], data["mov_signal"])

    ref_regions = build_domain_boundaries(
        data["ref_aligned"],
        data["ref_labels"],
        alpha=boundary_alpha_shape,
        remove_isolated=boundary_remove_isolated,
        neighbor_radius=boundary_neighbor_radius,
        min_neighbors=boundary_min_neighbors,
        radius_scale=boundary_radius_scale,
    )
    mov_regions = build_domain_boundaries(
        data["mov_aligned"],
        data["mov_labels"],
        alpha=boundary_alpha_shape,
        remove_isolated=boundary_remove_isolated,
        neighbor_radius=boundary_neighbor_radius,
        min_neighbors=boundary_min_neighbors,
        radius_scale=boundary_radius_scale,
    )

    # ------------------------------------------------------------------
    # Reference state
    # ------------------------------------------------------------------
    draw_boundaries(
        ax_ref,
        ref_regions,
        data["boundary_colors"],
        linewidth=boundary_linewidth,
        alpha=boundary_alpha,
    )
    scatter_gene(
        ax_ref,
        data["ref_aligned"],
        data["ref_signal"],
        norm,
        marker=ref_marker,
        size=ref_point_size,
        alpha=ref_alpha,
    )
    style_axis(ax_ref, f"Reference {data['gene']} state", invert_x, invert_y)

    # ------------------------------------------------------------------
    # Full mapped-state overlap
    # ------------------------------------------------------------------
    draw_boundaries(
        ax_mapped,
        mov_regions,
        data["boundary_colors"],
        linewidth=boundary_linewidth,
        alpha=boundary_alpha,
    )
    scatter_gene(
        ax_mapped,
        data["ref_aligned"],
        data["ref_signal"],
        norm,
        marker=ref_marker,
        size=ref_point_size,
        alpha=ref_alpha,
        label="Reference",
    )
    scatter_gene(
        ax_mapped,
        data["mov_aligned"],
        data["mov_signal"],
        norm,
        marker=moving_marker,
        size=moving_point_size,
        alpha=moving_alpha,
        edgecolors="black",
        linewidths=0.25,
        label="Mapped moving",
        zorder=4,
    )
    style_axis(ax_mapped, f"Mapped {data['gene']} states", invert_x, invert_y)

    center, half_width, half_height = choose_zoom_window(
        data["ref_aligned"],
        data["mov_aligned"],
        zoom_center,
        zoom_half_width,
        zoom_half_height,
    )
    rectangle = Rectangle(
        (center[0] - half_width, center[1] - half_height),
        2 * half_width,
        2 * half_height,
        fill=False,
        edgecolor="black",
        linewidth=1.6,
        zorder=6,
    )
    ax_mapped.add_patch(rectangle)

    # ------------------------------------------------------------------
    # Zoomed overlap
    # ------------------------------------------------------------------
    draw_boundaries(
        ax_zoom,
        mov_regions,
        data["boundary_colors"],
        linewidth=boundary_linewidth + 0.3,
        alpha=boundary_alpha,
    )
    scatter_gene(
        ax_zoom,
        data["ref_aligned"],
        data["ref_signal"],
        norm,
        marker=ref_marker,
        size=ref_point_size + 50,
        alpha=min(ref_alpha + 0.1, 1.0),
        label="Reference",
    )
    scatter_gene(
        ax_zoom,
        data["mov_aligned"],
        data["mov_signal"],
        norm,
        marker=moving_marker,
        size=moving_point_size + 12,
        alpha=min(moving_alpha + 0.1, 1.0),
        edgecolors="black",
        linewidths=0.35,
        label="Mapped moving",
        zorder=4,
    )
    style_axis(ax_zoom, "Zoomed mapped-state overlap", False, False)
    apply_zoom(ax_zoom, center, half_width, half_height, invert_x, invert_y)

    # style_axis hides spines; restore them as a black frame.
    for spine in ax_zoom.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    # Place the zoom legend in its own reserved axis.
    handles, labels = ax_zoom.get_legend_handles_labels()
    ax_zoom_legend.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=1.1,
        borderaxespad=0.0,
    )

    # Keep the two full-state panels on identical limits.
    set_common_limits(
        [ax_ref, ax_mapped],
        data["ref_aligned"],
        data["mov_aligned"],
    )
    for ax in (ax_ref, ax_mapped):
        if invert_x:
            ax.invert_xaxis()
        if invert_y:
            ax.invert_yaxis()

    fig.suptitle(
        "Common-frame aligned states",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )

    # Dedicated colorbar axis avoids cross-GridSpec layout calculations.
    colorbar = fig.colorbar(make_gene_mappable(norm), cax=cax)
    colorbar.set_label(data["gene_scale_label"], fontsize=10)
    colorbar.ax.tick_params(labelsize=9)

    finish_figure(fig, save_path, show, dpi)
    return fig


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def plot_transition_figures(
    root,
    group_id,
    reference_id,
    moving_id,
    obs_key,
    gene="PCP4",
    gene_layer=None,
    gene_use_raw=False,
    gene_log1p=True,
    coord_key="spatial",
    pairwise_subdir="pairwise_output",
    point_size=12,
    point_alpha=0.9,
    image_alpha=1.0,
    field_step=1.0,
    field_support_radius=10.0,
    field_smooth_sigma=1.0,
    field_cmap="magma",
    delta_cmap="coolwarm",
    stream_color="white",
    stream_density=1.0,
    stream_linewidth=0.95,
    boundary_alpha_shape=0.01,
    boundary_remove_isolated=True,
    boundary_neighbor_radius=None,
    boundary_min_neighbors=3,
    boundary_radius_scale=1.8,
    boundary_cmap="tab10",
    boundary_linewidth=1.5,
    zoom_center=None,
    zoom_half_width=75.0,
    zoom_half_height=None,
    invert_x=False,
    invert_y=True,
    save_dir=None,
    filename_prefix=None,
    show=True,
    dpi=350,
):
    data = prepare_plot_data(
        root=root,
        group_id=group_id,
        reference_id=reference_id,
        moving_id=moving_id,
        obs_key=obs_key,
        gene=gene,
        gene_layer=gene_layer,
        gene_use_raw=gene_use_raw,
        gene_log1p=gene_log1p,
        coord_key=coord_key,
        pairwise_subdir=pairwise_subdir,
        field_step=field_step,
        field_support_radius=field_support_radius,
        field_smooth_sigma=field_smooth_sigma,
        boundary_cmap=boundary_cmap,
    )

    prefix = filename_prefix or f"group{group_id}_{reference_id}_to_{moving_id}"
    paths = {"observed": None, "transition": None, "aligned": None}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        paths = {
            "observed": os.path.join(save_dir, f"{prefix}_01_observed_{gene}_HE.png"),
            "transition": os.path.join(save_dir, f"{prefix}_02_reverse_transition_field.png"),
            "aligned": os.path.join(save_dir, f"{prefix}_03_aligned_{gene}_states.png"),
        }

    plot_observed_gene_on_he(
        data,
        point_size=point_size,
        point_alpha=point_alpha,
        image_alpha=image_alpha,
        invert_x=invert_x,
        invert_y=invert_y,
        save_path=paths["observed"],
        show=show,
        dpi=dpi,
    )
    plot_continuous_transition_field(
        data,
        field_cmap=field_cmap,
        delta_cmap=delta_cmap,
        stream_color=stream_color,
        stream_density=stream_density,
        stream_linewidth=stream_linewidth,
        invert_x=invert_x,
        invert_y=invert_y,
        save_path=paths["transition"],
        show=show,
        dpi=dpi,
    )
    plot_common_frame_aligned_states(
        data,
        ref_point_size=point_size,
        moving_point_size=point_size + 4,
        ref_alpha=point_alpha,
        moving_alpha=point_alpha,
        boundary_alpha_shape=boundary_alpha_shape,
        boundary_remove_isolated=boundary_remove_isolated,
        boundary_neighbor_radius=boundary_neighbor_radius,
        boundary_min_neighbors=boundary_min_neighbors,
        boundary_radius_scale=boundary_radius_scale,
        boundary_linewidth=boundary_linewidth,
        zoom_center=zoom_center,
        zoom_half_width=zoom_half_width,
        zoom_half_height=zoom_half_height,
        invert_x=invert_x,
        invert_y=invert_y,
        save_path=paths["aligned"],
        show=show,
        dpi=dpi,
    )
    return paths


if __name__ == "__main__":
    plot_transition_figures(
        root="/media/huifang/data/registration/DLPFC/huifang",
        group_id=1,
        reference_id=1,
        moving_id=2,
        obs_key="layer_guess_reordered",
        gene="PCP4",
        gene_layer=None,
        gene_use_raw=False,
        gene_log1p=True,
        coord_key="spatial",
        pairwise_subdir="pairwise_output",
        point_size=14,
        point_alpha=0.85,
        image_alpha=1.0,
        field_step=2.0,
        field_support_radius=10.0,
        field_smooth_sigma=1.0,
        boundary_alpha_shape=0.01,
        boundary_remove_isolated=True,
        boundary_min_neighbors=3,
        boundary_radius_scale=1.8,
        boundary_linewidth=1.6,
        zoom_center=None,       # Set (x, y) manually when needed.
        zoom_half_width=75.0,
        zoom_half_height=75.0,
        invert_x=False,
        invert_y=True,
        save_dir=None,
        show=True,
    )
