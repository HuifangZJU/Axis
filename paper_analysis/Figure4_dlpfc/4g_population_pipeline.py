#!/usr/bin/env python3
"""
Simplified lifted-3D transition-field visualization for DLPFC.

Core idea
---------
For each adjacent slice pair:
1. find corresponding points using the saved AXIS pairwise warp;
2. build a smooth 2D curve in the tissue plane for each correspondence;
3. lift that 2D curve into 3D by assigning depth values between the two slices.

This works for:
- two slices: one lifted 2D transition segment;
- multiple slices: concatenate lifted pairwise segments across all neighboring pairs.

This script intentionally focuses on plotting style first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from axis.coordinate_utils import find_inverse_warp_coords
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.interpolate import RegularGridInterpolator


# =============================================================================
# Default configuration
# =============================================================================

ROOT = "/media/huifang/data/registration/DLPFC/huifang"
GROUP_ID = 1
SAMPLE_IDS = [0,1, 2,3]
REFERENCE_ID = 0
PAIRWISE_SUBDIR = "pairwise_output"
COORD_KEY = "spatial"

# None -> equally spaced slice planes using DEPTH_GAP.
AXIS_POSITIONS: Optional[Sequence[float]] = None
DEPTH_GAP = 160.0

DOMAIN_KEY = "layer_guess_reordered"
TRAJECTORY_DOMAINS = ["Layer4"]
# TRAJECTORY_DOMAINS = ["Layer4"]
MAX_SEED_POINTS = 220
RANDOM_SEED = 0

# Context points on slice planes
# SLICE_COLOR_KEY: Optional[str] = "layer_guess_reordered"   # obs column, gene name, or None
SLICE_COLOR_KEY: Optional[str] = "PCP4"   # obs column, gene name, or None
USE_LOG1P_FOR_GENE = True
SLICE_POINT_SIZE = 5
SLICE_POINT_ALPHA = 0.56

# Pairwise 2D curve shape
CURVE_CONTROL_SCALE = 0.32
CURVE_STEPS_PER_PAIR = 32

# 3D rendering
CURVE_CMAP = "viridis"
CURVE_LINEWIDTH = 1
CURVE_ALPHA = 0.7
ELEV = 13
AZIM = -145
FIGSIZE = (9, 7.5)

OUT_DIR: Optional[Path] = None


# =============================================================================
# File and data loading (recycled from the original pipeline)
# =============================================================================

def parse_h5ad_path(root: str, group_id: int, sample_id: int) -> str:
    return os.path.join(root, "h5ad", f"{group_id}_{sample_id}.h5ad")


def parse_network_npz_path(root: str, group_id: int, sample_id: int) -> str:
    return os.path.join(
        root,
        "network_input",
        f"{group_id}_{sample_id}_validation.npz",
    )


def find_group_dir(
    root: str,
    group_id: int,
    pairwise_subdir: str = "pairwise_output",
) -> str:
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


def parse_flow_path(
    root: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
    pairwise_subdir: str = "pairwise_output",
) -> str:
    group_dir = find_group_dir(root, group_id, pairwise_subdir=pairwise_subdir)
    return os.path.join(group_dir, f"{fixed_id}_{moving_id}_flow.npy")


def load_npz_coord(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)

    if "coord" not in data:
        raise KeyError(f"'coord' not found in {npz_path}")

    coord = np.asarray(data["coord"], dtype=np.float32)
    if coord.ndim != 2 or coord.shape[1] < 2:
        raise ValueError(
            f"coord in {npz_path} must have shape [N,2+], got {coord.shape}"
        )

    return coord[:, :2]


def load_plot_slice_coords_aligned(
    root: str,
    group_id: int,
    sample_id: int,
    reference_id: int = 0,
    coord_key: str = "spatial",
    pairwise_subdir: str = "pairwise_output",
):
    """
    Load aligned coordinates in the same style as the original script.
    Priority:
      1. adata.obsm[coord_key]
      2. reference slice -> network_input coord
      3. pre-saved warped coordinates to the reference
    """
    adata = sc.read_h5ad(parse_h5ad_path(root, group_id, sample_id))

    if coord_key in adata.obsm:
        coord = np.asarray(adata.obsm[coord_key], dtype=np.float32)
        return adata, coord[:, :2]

    if sample_id == reference_id:
        coord = load_npz_coord(parse_network_npz_path(root, group_id, sample_id))
        return adata, coord

    group_dir = find_group_dir(root, group_id, pairwise_subdir=pairwise_subdir)
    warped_path = os.path.join(group_dir, f"{reference_id}_{sample_id}_warped_coor.npy")

    if os.path.exists(warped_path):
        coord = np.asarray(np.load(warped_path), dtype=np.float32)
        return adata, coord[:, :2]

    raise FileNotFoundError(
        f"Could not find aligned coordinates for group={group_id}, sample={sample_id}."
    )


def ensure_flow_hw2(flow: np.ndarray) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)

    if flow.ndim != 3:
        raise ValueError(f"Flow must be 3D, got {flow.shape}")

    if flow.shape[-1] == 2:
        return flow

    if flow.shape[0] == 2:
        return np.moveaxis(flow, 0, -1)

    raise ValueError(f"Cannot interpret flow shape {flow.shape} as [H, W, 2]")


def inverse_warp_points(points_xy: np.ndarray, flow_hw2: np.ndarray) -> np.ndarray:
    warped = find_inverse_warp_coords(points_xy, flow_hw2)

    if isinstance(warped, (tuple, list)):
        warped = warped[0]

    warped = np.asarray(warped, dtype=np.float32)

    if warped.ndim != 2 or warped.shape[1] < 2:
        raise ValueError(f"Unexpected inverse-warp output shape: {warped.shape}")

    return warped[:, :2]


# =============================================================================
# Display-value helpers
# =============================================================================

def extract_gene_values(
    adata,
    gene: str,
    use_log1p: bool = True,
) -> np.ndarray:
    if gene not in adata.var_names:
        raise KeyError(f"Gene {gene!r} not found.")

    values = adata[:, gene].X
    if sp.issparse(values):
        values = values.toarray().reshape(-1)
    else:
        values = np.asarray(values).reshape(-1)

    values = values.astype(np.float32)

    if use_log1p:
        values = np.log1p(np.clip(values, 0, None))

    return values


def load_slice_display_values(
    adata,
    color_key: Optional[str],
    use_log1p_gene: bool,
):
    """
    Return values and value kind:
        kind = "none" | "categorical" | "continuous"
    """
    if color_key is None:
        return None, "none"

    if color_key in adata.obs.columns:
        series = adata.obs[color_key]

        if (
            pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            return series.astype(str).to_numpy(), "categorical"

        return (
            pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32),
            "continuous",
        )

    if color_key in adata.var_names:
        return extract_gene_values(adata, color_key, use_log1p=use_log1p_gene), "continuous"

    raise KeyError(f"{color_key!r} is neither an obs column nor a gene.")


def build_stable_category_palette(
    values_by_slice: Dict[int, np.ndarray],
    cmap_name: str = "tab10",
) -> Dict[str, Tuple[float, float, float, float]]:
    all_labels = sorted(
        {str(v) for values in values_by_slice.values() for v in values}
    )
    cmap = plt.get_cmap(cmap_name, max(len(all_labels), 1))
    return {label: cmap(i) for i, label in enumerate(all_labels)}


# =============================================================================
# Seed and slice organization
# =============================================================================

def build_plane_positions(
    sample_ids: Sequence[int],
    axis_positions: Optional[Sequence[float]],
    depth_gap: float,
) -> Dict[int, float]:
    if axis_positions is not None:
        if len(axis_positions) != len(sample_ids):
            raise ValueError("axis_positions must have the same length as sample_ids.")
        return {sid: float(z) for sid, z in zip(sample_ids, axis_positions)}

    return {sid: i * float(depth_gap) for i, sid in enumerate(sample_ids)}


def make_domain_mask(
    adata,
    include_domains: Optional[Sequence[str]],
    domain_key: str,
    n_keep: int,
) -> np.ndarray:
    labels = np.asarray(adata.obs[domain_key].iloc[:n_keep].astype(str))

    if include_domains is None:
        return np.ones(len(labels), dtype=bool)

    keep = set(map(str, include_domains))
    return np.array([x in keep for x in labels], dtype=bool)


def choose_seed_points(
    coords: np.ndarray,
    max_keep: int,
    seed: int,
) -> np.ndarray:
    if len(coords) <= max_keep:
        return coords.copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(coords), size=max_keep, replace=False)
    return coords[np.sort(idx)]


# =============================================================================
# Pairwise 2D streamline construction
# =============================================================================

def make_flow_interpolator(flow_hw2: np.ndarray):
    """
    Bilinear interpolator for a 2D flow field.

    Input point convention for query:
        [y, x]
    Output:
        [dx, dy]
    """
    h, w = flow_hw2.shape[:2]
    y = np.arange(h, dtype=float)
    x = np.arange(w, dtype=float)

    interp_dx = RegularGridInterpolator(
        (y, x),
        flow_hw2[..., 0],
        bounds_error=False,
        fill_value=np.nan,
    )
    interp_dy = RegularGridInterpolator(
        (y, x),
        flow_hw2[..., 1],
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp_dx, interp_dy


def sample_flow_vector(
    point_xy: np.ndarray,
    flow_interp,
) -> np.ndarray:
    interp_dx, interp_dy = flow_interp
    query = np.array([[point_xy[1], point_xy[0]]], dtype=float)

    dx = float(interp_dx(query)[0])
    dy = float(interp_dy(query)[0])

    if not np.isfinite(dx) or not np.isfinite(dy):
        return np.array([np.nan, np.nan], dtype=float)

    return np.array([dx, dy], dtype=float)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))

    if not np.isfinite(norm) or norm <= 1e-8:
        return np.array([np.nan, np.nan], dtype=float)

    return vec / norm


def cubic_bezier_2d(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(n_steps), dtype=float)[:, None]
    omt = 1.0 - t

    curve = (
        omt**3 * p0[None, :]
        + 3.0 * omt**2 * t * p1[None, :]
        + 3.0 * omt * t**2 * p2[None, :]
        + t**3 * p3[None, :]
    )
    return curve.astype(np.float32)


def build_pairwise_2d_curve(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    forward_flow_interp,
    backward_flow_interp=None,
    control_scale: float = 0.32,
    n_steps: int = 32,
) -> np.ndarray:
    """
    Build one smooth 2D transition curve for one point pair.

    The start and end tangents are guided by local flow directions.
    If local flow is invalid, the direct displacement direction is used.
    """
    start_xy = np.asarray(start_xy, dtype=float)
    end_xy = np.asarray(end_xy, dtype=float)

    disp = end_xy - start_xy
    dist = float(np.linalg.norm(disp))

    if not np.isfinite(dist) or dist <= 1e-8:
        return np.repeat(start_xy[None, :], n_steps, axis=0).astype(np.float32)

    unit_disp = disp / dist

    v0 = sample_flow_vector(start_xy, forward_flow_interp)
    v0 = normalize_vector(v0)
    if not np.isfinite(v0).all():
        v0 = unit_disp

    if backward_flow_interp is not None:
        vb = sample_flow_vector(end_xy, backward_flow_interp)
        vb = normalize_vector(vb)
        if np.isfinite(vb).all():
            v1 = -vb
        else:
            v1 = unit_disp
    else:
        v1 = unit_disp

    handle_len = control_scale * dist
    c1 = start_xy + handle_len * v0
    c2 = end_xy - handle_len * v1

    return cubic_bezier_2d(
        p0=start_xy,
        p1=c1,
        p2=c2,
        p3=end_xy,
        n_steps=n_steps,
    )


def lift_2d_curve_to_3d(
    curve_xy: np.ndarray,
    depth_start: float,
    depth_end: float,
) -> np.ndarray:
    """
    Convert [N,2] xy curve to [N,3] with columns:
        [depth, tissue_x, tissue_y]
    """
    n = len(curve_xy)
    depth = np.linspace(depth_start, depth_end, n, dtype=float)
    return np.column_stack([depth, curve_xy[:, 0], curve_xy[:, 1]]).astype(np.float32)


# =============================================================================
# Trajectory construction across multiple slices
# =============================================================================

def build_lifted_pairwise_trajectories(
    root: str,
    group_id: int,
    sample_ids: Sequence[int],
    seed_points_first_slice: np.ndarray,
    plane_positions: Dict[int, float],
    pairwise_subdir: str,
    curve_control_scale: float,
    curve_steps_per_pair: int,
) -> List[np.ndarray]:
    """
    For every seed point, build a piecewise-lifted 3D curve.

    Each neighboring slice pair contributes one lifted 2D segment.
    """
    n_seeds = len(seed_points_first_slice)
    curves_3d = [[] for _ in range(n_seeds)]

    current_points = seed_points_first_slice.copy()
    active = np.ones(n_seeds, dtype=bool)

    for pair_index in range(len(sample_ids) - 1):
        sid_a = sample_ids[pair_index]
        sid_b = sample_ids[pair_index + 1]

        flow_forward = ensure_flow_hw2(
            np.load(
                parse_flow_path(
                    root=root,
                    group_id=group_id,
                    fixed_id=sid_b,
                    moving_id=sid_a,
                    pairwise_subdir=pairwise_subdir,
                )
            )
        )
        forward_interp = make_flow_interpolator(flow_forward)

        backward_interp = None
        reverse_path = parse_flow_path(
            root=root,
            group_id=group_id,
            fixed_id=sid_a,
            moving_id=sid_b,
            pairwise_subdir=pairwise_subdir,
        )
        if os.path.exists(reverse_path):
            flow_backward = ensure_flow_hw2(np.load(reverse_path))
            backward_interp = make_flow_interpolator(flow_backward)

        next_points = np.full_like(current_points, np.nan, dtype=np.float32)

        if np.any(active):
            idx = np.flatnonzero(active)
            warped = inverse_warp_points(current_points[idx], flow_forward)

            h, w = flow_forward.shape[:2]
            valid = np.all(np.isfinite(warped), axis=1)
            valid &= (warped[:, 0] >= 0) & (warped[:, 0] < w)
            valid &= (warped[:, 1] >= 0) & (warped[:, 1] < h)

            idx_valid = idx[valid]
            next_points[idx_valid] = warped[valid]

            new_active = np.zeros_like(active)
            new_active[idx_valid] = True
            active = new_active

        z0 = plane_positions[sid_a]
        z1 = plane_positions[sid_b]

        for i in np.flatnonzero(active):
            start_xy = current_points[i]
            end_xy = next_points[i]

            curve_xy = build_pairwise_2d_curve(
                start_xy=start_xy,
                end_xy=end_xy,
                forward_flow_interp=forward_interp,
                backward_flow_interp=backward_interp,
                control_scale=curve_control_scale,
                n_steps=curve_steps_per_pair,
            )
            curve_3d = lift_2d_curve_to_3d(curve_xy, z0, z1)

            if len(curves_3d[i]) == 0:
                curves_3d[i].append(curve_3d)
            else:
                curves_3d[i].append(curve_3d[1:])  # avoid duplicate join point

        current_points = next_points

    out = []
    for pieces in curves_3d:
        if len(pieces) > 0:
            out.append(np.concatenate(pieces, axis=0))

    return out


# =============================================================================
# Plotting
# =============================================================================

def draw_slice_plane(
    ax,
    depth: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
) -> None:
    x0, x1 = x_bounds
    y0, y1 = y_bounds

    verts = [[
        (depth, x0, y0),
        (depth, x1, y0),
        (depth, x1, y1),
        (depth, x0, y1),
    ]]

    poly = Poly3DCollection(
        verts,
        facecolors=(0.92, 0.92, 0.92, 0.025),
        edgecolors=(0.45, 0.45, 0.45, 0.25),
        linewidths=0.6,
    )
    ax.add_collection3d(poly)


def add_curve_collection(
    ax,
    curves_3d: Sequence[np.ndarray],
    cmap_name: str,
    linewidth: float,
    alpha: float,
):
    segments_all = []
    values_all = []

    for curve in curves_3d:
        if len(curve) < 2:
            continue

        segments = np.stack([curve[:-1], curve[1:]], axis=1)
        values = 0.5 * (curve[:-1, 0] + curve[1:, 0])   # average depth

        segments_all.append(segments)
        values_all.append(values)

    segments_all = np.concatenate(segments_all, axis=0)
    values_all = np.concatenate(values_all, axis=0)

    norm = Normalize(
        vmin=float(np.nanmin(values_all)),
        vmax=float(np.nanmax(values_all)),
    )

    coll = Line3DCollection(
        segments_all,
        cmap=cmap_name,
        norm=norm,
        linewidths=linewidth,
        alpha=alpha,
    )
    coll.set_array(values_all)
    ax.add_collection3d(coll)
    return coll


def save_or_show(
    fig: plt.Figure,
    outdir: Optional[Path],
    stem: str,
    dpi: int = 400,
) -> None:
    if outdir is None:
        plt.show()
        return

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig.savefig(outdir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_lifted_transition_field_3d(
    root: str,
    group_id: int,
    sample_ids: Sequence[int],
    *,
    reference_id: int = 0,
    coord_key: str = "spatial",
    pairwise_subdir: str = "pairwise_output",
    axis_positions: Optional[Sequence[float]] = None,
    depth_gap: float = 160.0,
    domain_key: str = "layer_guess_reordered",
    trajectory_domains: Optional[Sequence[str]] = None,
    max_seed_points: int = 220,
    random_seed: int = 0,
    slice_color_key: Optional[str] = "layer_guess_reordered",
    use_log1p_for_gene: bool = True,
    slice_point_size: float = 1.2,
    slice_point_alpha: float = 0.16,
    curve_control_scale: float = 0.32,
    curve_steps_per_pair: int = 32,
    curve_cmap: str = "viridis",
    curve_linewidth: float = 1.2,
    curve_alpha: float = 0.86,
    elev: float = 18,
    azim: float = -128,
    figsize: Tuple[float, float] = (11, 7.5),
    outdir: Optional[Path] = None,
    stem: str = "dlpfc_lifted_transition_field_3d",
):
    """
    Main plotting function.
    """
    if len(sample_ids) < 2:
        raise ValueError("At least two slices are required.")

    plane_positions = build_plane_positions(
        sample_ids=sample_ids,
        axis_positions=axis_positions,
        depth_gap=depth_gap,
    )

    slice_cache = {}
    values_by_slice = {}
    value_kinds = []

    all_x = []
    all_y = []

    for sid in sample_ids:
        adata, coords = load_plot_slice_coords_aligned(
            root=root,
            group_id=group_id,
            sample_id=sid,
            reference_id=reference_id,
            coord_key=coord_key,
            pairwise_subdir=pairwise_subdir,
        )

        n_keep = min(adata.n_obs, len(coords))
        adata = adata[:n_keep].copy()
        coords = coords[:n_keep]

        values, kind = load_slice_display_values(
            adata,
            color_key=slice_color_key,
            use_log1p_gene=use_log1p_for_gene,
        )

        slice_cache[sid] = {
            "adata": adata,
            "coords": coords,
            "values": values,
            "kind": kind,
        }
        if values is not None:
            values_by_slice[sid] = values
        value_kinds.append(kind)

        all_x.append(coords[:, 0])
        all_y.append(coords[:, 1])

    # Seed points come from the first slice
    first_sid = sample_ids[0]
    first_record = slice_cache[first_sid]

    if domain_key not in first_record["adata"].obs.columns:
        raise KeyError(f"{domain_key!r} not found in the first slice obs.")

    seed_mask = make_domain_mask(
        first_record["adata"],
        include_domains=trajectory_domains,
        domain_key=domain_key,
        n_keep=len(first_record["coords"]),
    )
    seed_coords = first_record["coords"][seed_mask]

    if len(seed_coords) == 0:
        raise ValueError(f"No seed points found for domains={trajectory_domains}.")

    seed_coords = choose_seed_points(
        seed_coords,
        max_keep=max_seed_points,
        seed=random_seed,
    )

    curves_3d = build_lifted_pairwise_trajectories(
        root=root,
        group_id=group_id,
        sample_ids=sample_ids,
        seed_points_first_slice=seed_coords,
        plane_positions=plane_positions,
        pairwise_subdir=pairwise_subdir,
        curve_control_scale=curve_control_scale,
        curve_steps_per_pair=curve_steps_per_pair,
    )

    if len(curves_3d) == 0:
        raise RuntimeError("No lifted trajectories were generated.")

    print(
        f"[INFO] seeds selected: {len(seed_coords)} | "
        f"lifted curves generated: {len(curves_3d)}"
    )

    # Stable context coloring
    category_palette = None
    continuous_norm = None

    unique_kinds = set(value_kinds)
    if slice_color_key is not None:
        if unique_kinds == {"categorical"}:
            category_palette = build_stable_category_palette(
                values_by_slice,
                cmap_name="tab10",
            )
        elif unique_kinds == {"continuous"}:
            combined = np.concatenate(
                [v[np.isfinite(v)] for v in values_by_slice.values()]
            )
            vmax = float(np.nanpercentile(combined, 99.5))
            continuous_norm = Normalize(
                vmin=float(np.nanmin(combined)),
                vmax=max(vmax, 1e-6),
            )

    tissue_x = np.concatenate(all_x)
    tissue_y = np.concatenate(all_y)

    x_bounds = (float(np.nanmin(tissue_x)), float(np.nanmax(tissue_x)))
    y_bounds = (float(np.nanmin(tissue_y)), float(np.nanmax(tissue_y)))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")

    # Slice planes
    for sid in sample_ids:
        draw_slice_plane(
            ax,
            depth=plane_positions[sid],
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )

    # Context points
    for sid in sample_ids:
        record = slice_cache[sid]
        coords = record["coords"]
        values = record["values"]
        kind = record["kind"]
        depth = plane_positions[sid]

        if kind == "categorical":
            colors = [category_palette[str(v)] for v in values]
            alpha = slice_point_alpha
        elif kind == "continuous":
            colors = plt.get_cmap("Reds")(continuous_norm(values))
            colors[:, 3] = slice_point_alpha
            alpha = None
        else:
            colors = (0.45, 0.45, 0.45, slice_point_alpha)
            alpha = None

        ax.scatter(
            np.full(len(coords), depth, dtype=float),
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=slice_point_size,
            alpha=alpha,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )

    curve_collection = add_curve_collection(
        ax,
        curves_3d=curves_3d,
        cmap_name=curve_cmap,
        linewidth=curve_linewidth,
        alpha=curve_alpha,
    )

    # Axis limits
    z_depths = np.asarray([plane_positions[sid] for sid in sample_ids], dtype=float)
    depth_min = float(np.min(z_depths))
    depth_max = float(np.max(z_depths))

    dx = 0.04 * max(x_bounds[1] - x_bounds[0], 1.0)
    dy = 0.04 * max(y_bounds[1] - y_bounds[0], 1.0)
    dz = 0.04 * max(depth_max - depth_min, 1.0)

    ax.set_xlim(depth_min - dz, depth_max + dz)
    ax.set_ylim(x_bounds[1] + dx, x_bounds[0] - dx)
    ax.set_zlim(y_bounds[1] + dy, y_bounds[0] - dy)

    ax.set_xticks(z_depths)
    ax.set_xticklabels([f"Slice {sid}" for sid in sample_ids], fontsize=9)
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("Depth", labelpad=10)
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)

    ax.set_box_aspect(
        (
            max(depth_max - depth_min, 1.0) * 1.35,
            max(x_bounds[1] - x_bounds[0], 1.0),
            max(y_bounds[1] - y_bounds[0], 1.0),
        )
    )

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))

    domain_text = (
        "all domains"
        if trajectory_domains is None
        else ", ".join(map(str, trajectory_domains))
    )
    ax.set_title(
        f"Lifted continuous transition field | {domain_text}",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    cbar = fig.colorbar(
        curve_collection,
        ax=ax,
        fraction=0.028,
        pad=0.06,
        shrink=0.68,
    )
    cbar.set_label("Depth progression", fontsize=9)

    if category_palette is not None:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=5,
                label=label,
            )
            for label, color in category_palette.items()
        ]
        ax.legend(
            handles=handles,
            title=slice_color_key,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=False,
            fontsize=8,
            title_fontsize=8.5,
        )

    fig.tight_layout()
    save_or_show(fig, outdir=outdir, stem=stem)
    return fig, ax, curves_3d


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    plot_lifted_transition_field_3d(
        root=ROOT,
        group_id=GROUP_ID,
        sample_ids=SAMPLE_IDS,
        reference_id=REFERENCE_ID,
        coord_key=COORD_KEY,
        pairwise_subdir=PAIRWISE_SUBDIR,
        axis_positions=AXIS_POSITIONS,
        depth_gap=DEPTH_GAP,
        domain_key=DOMAIN_KEY,
        trajectory_domains=TRAJECTORY_DOMAINS,
        max_seed_points=MAX_SEED_POINTS,
        random_seed=RANDOM_SEED,
        slice_color_key=SLICE_COLOR_KEY,
        use_log1p_for_gene=USE_LOG1P_FOR_GENE,
        slice_point_size=SLICE_POINT_SIZE,
        slice_point_alpha=SLICE_POINT_ALPHA,
        curve_control_scale=CURVE_CONTROL_SCALE,
        curve_steps_per_pair=CURVE_STEPS_PER_PAIR,
        curve_cmap=CURVE_CMAP,
        curve_linewidth=CURVE_LINEWIDTH,
        curve_alpha=CURVE_ALPHA,
        elev=ELEV,
        azim=AZIM,
        figsize=FIGSIZE,
        outdir=OUT_DIR,
    )
