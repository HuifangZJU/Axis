#!/usr/bin/env python3
"""
Streamlined DLPFC dense-reconstruction visualization.

This script generates two figures with exactly the same 3D camera, limits,
aspect ratio, real-slice rendering, and virtual-slice frame geometry:

Figure 1
--------
Two measured endpoint slices are shown. Intermediate virtual locations are
represented only by dashed rectangular frames; no virtual points are scattered.

Figure 2
--------
The same two measured endpoint slices and the same dashed virtual frames are
shown, with projected virtual states filled in at every intermediate plane.

Transition-field lines are intentionally not plotted. Pairwise flows are used
only to determine the intermediate virtual-state locations.

Every real and virtual plane is annotated directly in 3D using the same
plane-label style as the SCC visualization. Display depth and biological depth
remain separate: DEPTH_TOTAL controls only 3D spacing, whereas plane-label text
uses the biological z positions configured in SLICE_Z_UM.
"""

from __future__ import annotations
from axis.settings import data_path
from axis.config import load_config

import math
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns

from axis.registration import find_inverse_warp_coords
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = load_config('dlpfc').root
GROUP_ID = 1
SLICE_A = 1
SLICE_B = 2

PAIRWISE_SUBDIR = "pairwise_output"
COORD_KEY = "spatial"

# Real-slice display:
#   "gene"   -> PCP4 expression
#   "domain" -> spatial-domain labels
REAL_SLICE_MODE = "gene"
# REAL_SLICE_MODE = "domain"
GENE = "PCP4"
USE_LOG1P = True

REAL_DOMAIN_KEY = "layer_guess_reordered"
REAL_DOMAIN_CMAP = "tab10"
PREFERRED_DOMAIN_ORDER = [
    # "Layer1",
    # "Layer2",
    "Layer3",
    "Layer4",
    "Layer5",
    "Layer6",
    "WM",
]

_dlpfc_tab10 = sns.color_palette("tab10", n_colors=10)
DLPFC_DOMAIN_COLORS = {
    # "Layer1": _dlpfc_tab10[0],
    # "Layer2": _dlpfc_tab10[1],
    "Layer3": _dlpfc_tab10[0],
    "Layer4": _dlpfc_tab10[1],
    "Layer5": _dlpfc_tab10[2],
    "Layer6": _dlpfc_tab10[3],
    "WM": _dlpfc_tab10[4],
}

# Dense reconstruction geometry.
N_VIRTUAL = 3
DEPTH_TOTAL = 650.0

# Biological z positions used only for plane-label text. These are independent
# of DEPTH_TOTAL, which controls the visual separation of planes in the 3D plot.
# For the DLPFC group used here, slice 1 and slice 2 are at 10 and 310 um.
SLICE_Z_UM: Mapping[int, float] = {
    0: 0.0,
    1: 10.0,
    2: 310.0,
    3: 320.0,
}

# Direct labels drawn next to each 3D plane.
PLANE_LABEL_FONTSIZE = 20.0
PLANE_LABEL_REAL_COLOR = "black"
SHOW_VIRTUAL_PLANE_LABELS = True

# Source points used to construct intermediate virtual states.
# 10,000 exceeds the usual DLPFC spot count and therefore normally retains
# essentially the full source set while preserving the original grid selector.
MAX_SEED_POINTS = 10000
SEED_GRID_NX: Optional[int] = None
SEED_GRID_NY: Optional[int] = None
RANDOM_SEED = 0

# Optional source-domain restriction. None -> all source points.
TRAJECTORY_DOMAIN_KEY = "layer_guess_reordered"
TRAJECTORY_DOMAINS = None

# Smooth interpolation used only to obtain intermediate projected locations.
CONTROL_SCALE = 0.34
STEPS_PER_SECTION = 20

# Real-slice point rendering.
REAL_POINT_SIZE = 8
REAL_POINT_ALPHA = 0.72

# Virtual-state points in Figure 2.
PROJECTION_POINT_SIZE = 8.0
PROJECTION_POINT_ALPHA = 0.45
PROJECTION_POINT_EDGECOLOR = "0.55"
PROJECTION_POINT_LINEWIDTH = 0.35
PROJECTION_MARKER_A_TO_B = "o"
PROJECTION_MARKER_B_TO_A = "^"

# Virtual-slice dashed frames; used identically in Figures 1 and 2.
VIRTUAL_FRAME_COLOR = "#666666"
VIRTUAL_FRAME_LINEWIDTH = 1.15
VIRTUAL_FRAME_ALPHA = 0.82
VIRTUAL_FRAME_DASH_PATTERN = (0, (4.0, 3.0))

# Shared 3D view settings. These are applied identically to both figures.
ELEV = 10
AZIM = -130
FIGSIZE = (13.5, 7.8)
DEPTH_ASPECT_SCALE = 1.35
XY_PAD_RATIO = 0.05
DEPTH_PAD_RATIO = 0.04
PROJECTION_TYPE = "ortho"

OUT_DIR: Optional[Path] = Path(data_path("paper_analysis/Figure4_dlfpc/a_pipeline/"))
STEM_PREFIX = "dlpfc_dense_reconstruction"


# =============================================================================
# STYLE
# =============================================================================

def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 12,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 350,
            "figure.dpi": 120,
        }
    )


# =============================================================================
# PATHS AND DATA LOADING
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
    candidates = [
        os.path.join(root, pairwise_subdir, f"group_{group_id}"),
        os.path.join(root, f"group_{group_id}"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Cannot find pairwise folder for group {group_id}.\n"
        + "\n".join(f"  {path}" for path in candidates)
    )


def parse_flow_path(
    root: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
    pairwise_subdir: str = "pairwise_output",
) -> str:
    group_dir = find_group_dir(
        root,
        group_id,
        pairwise_subdir=pairwise_subdir,
    )
    return os.path.join(
        group_dir,
        f"{fixed_id}_{moving_id}_flow.npy",
    )


def load_npz_coord(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)
    if "coord" not in data:
        raise KeyError(f"'coord' not found in {npz_path}")
    coords = np.asarray(data["coord"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"coord in {npz_path} must have shape [N,2+], got {coords.shape}"
        )
    return coords[:, :2]


def load_slice_native(
    root: str,
    group_id: int,
    sample_id: int,
    coord_key: str = "spatial",
):
    adata = sc.read_h5ad(
        parse_h5ad_path(root, group_id, sample_id)
    )

    if coord_key in adata.obsm:
        coords = np.asarray(
            adata.obsm[coord_key],
            dtype=np.float32,
        )[:, :2]
    else:
        coords = load_npz_coord(
            parse_network_npz_path(
                root,
                group_id,
                sample_id,
            )
        )

    n_keep = min(adata.n_obs, len(coords))
    if adata.n_obs != len(coords):
        print(
            f"[WARN] sample={sample_id}: "
            f"adata.n_obs={adata.n_obs}, coords={len(coords)}; "
            f"using first {n_keep}."
        )

    return adata[:n_keep].copy(), coords[:n_keep]


# =============================================================================
# REAL-SLICE VALUES / COLORS
# =============================================================================

def extract_gene_values(
    adata,
    gene: str,
    use_log1p: bool = True,
) -> np.ndarray:
    if gene not in adata.var_names:
        raise KeyError(
            f"Gene {gene!r} not found in adata.var_names."
        )

    values = adata[:, gene].X
    if sp.issparse(values):
        values = values.toarray().reshape(-1)
    else:
        values = np.asarray(values).reshape(-1)

    values = values.astype(np.float32)
    values[~np.isfinite(values)] = 0.0

    if use_log1p:
        values = np.log1p(
            np.clip(values, 0.0, None)
        )

    return values


def canonical_domain_label(value) -> str:
    if pd.isna(value):
        return "NA"

    text = str(value).strip()
    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
    }:
        return "NA"

    return text


def ordered_domain_labels(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    preferred_order: Sequence[str],
) -> List[str]:
    present = {
        canonical_domain_label(value)
        for value in list(labels_a) + list(labels_b)
    }
    ordered = [
        label
        for label in preferred_order
        if label in present
    ]
    extras = sorted(
        present - set(ordered)
    )
    return ordered + extras


def build_domain_color_map(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    preferred_order: Sequence[str],
    explicit_colors: Mapping[str, str],
    fallback_cmap: str = "tab10",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    labels_a = np.asarray(
        [
            canonical_domain_label(x)
            for x in labels_a
        ],
        dtype=str,
    )
    labels_b = np.asarray(
        [
            canonical_domain_label(x)
            for x in labels_b
        ],
        dtype=str,
    )

    categories = ordered_domain_labels(
        labels_a,
        labels_b,
        preferred_order=preferred_order,
    )

    fallback_labels = [
        label
        for label in categories
        if label not in explicit_colors
    ]
    fallback = plt.get_cmap(
        fallback_cmap,
        max(len(fallback_labels), 1),
    )

    color_map: Dict[str, object] = {}
    fallback_index = 0

    for label in categories:
        if label in explicit_colors:
            color_map[label] = explicit_colors[label]
        else:
            color_map[label] = fallback(fallback_index)
            fallback_index += 1

    return labels_a, labels_b, color_map


# =============================================================================
# PAIRWISE FLOW HELPERS
# =============================================================================

def ensure_flow_hw2(flow: np.ndarray) -> np.ndarray:
    flow = np.asarray(
        flow,
        dtype=np.float32,
    )

    if flow.ndim != 3:
        raise ValueError(
            f"Flow must be three-dimensional, got {flow.shape}."
        )

    if flow.shape[-1] == 2:
        return flow

    if flow.shape[0] == 2:
        return np.moveaxis(
            flow,
            0,
            -1,
        )

    raise ValueError(
        f"Cannot interpret flow shape {flow.shape} as [H,W,2]."
    )


def inverse_warp_points(
    points_xy: np.ndarray,
    flow_hw2: np.ndarray,
) -> np.ndarray:
    warped = find_inverse_warp_coords(
        points_xy,
        flow_hw2,
    )

    if isinstance(
        warped,
        (tuple, list),
    ):
        warped = warped[0]

    warped = np.asarray(
        warped,
        dtype=np.float32,
    )

    if warped.ndim != 2 or warped.shape[1] < 2:
        raise ValueError(
            f"Unexpected inverse-warp output shape: {warped.shape}"
        )

    return warped[:, :2]


def make_flow_interpolator(
    flow_hw2: np.ndarray,
):
    height, width = flow_hw2.shape[:2]
    y_axis = np.arange(
        height,
        dtype=float,
    )
    x_axis = np.arange(
        width,
        dtype=float,
    )

    interp_dx = RegularGridInterpolator(
        (y_axis, x_axis),
        flow_hw2[..., 0],
        bounds_error=False,
        fill_value=np.nan,
    )
    interp_dy = RegularGridInterpolator(
        (y_axis, x_axis),
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

    query = np.array(
        [[point_xy[1], point_xy[0]]],
        dtype=float,
    )
    dx = float(
        interp_dx(query)[0]
    )
    dy = float(
        interp_dy(query)[0]
    )

    if not np.isfinite(dx) or not np.isfinite(dy):
        return np.array(
            [np.nan, np.nan],
            dtype=float,
        )

    return np.array(
        [dx, dy],
        dtype=float,
    )


def normalize_vector(
    vec: np.ndarray,
) -> np.ndarray:
    vec = np.asarray(
        vec,
        dtype=float,
    )
    magnitude = float(
        np.linalg.norm(vec)
    )

    if (
        not np.isfinite(magnitude)
        or magnitude <= 1e-8
    ):
        return np.array(
            [np.nan, np.nan],
            dtype=float,
        )

    return vec / magnitude


# =============================================================================
# SOURCE-POINT SELECTION
# =============================================================================

def make_domain_mask(
    adata,
    domain_key: str,
    include_domains: Optional[Sequence[str]],
    n_keep: int,
) -> np.ndarray:
    if include_domains is None:
        return np.ones(
            n_keep,
            dtype=bool,
        )

    if domain_key not in adata.obs.columns:
        raise KeyError(
            f"{domain_key!r} not found in adata.obs."
        )

    labels = np.asarray(
        adata.obs[domain_key]
        .iloc[:n_keep]
        .astype(str)
    )
    keep = set(
        map(
            str,
            include_domains,
        )
    )

    return np.array(
        [
            label in keep
            for label in labels
        ],
        dtype=bool,
    )


def choose_grid_shape(
    coords: np.ndarray,
    target_points: int,
    grid_nx: Optional[int] = None,
    grid_ny: Optional[int] = None,
) -> Tuple[int, int]:
    if (
        grid_nx is not None
        and grid_ny is not None
    ):
        return (
            max(int(grid_nx), 1),
            max(int(grid_ny), 1),
        )

    x_span = float(
        np.nanmax(coords[:, 0])
        - np.nanmin(coords[:, 0])
    )
    y_span = float(
        np.nanmax(coords[:, 1])
        - np.nanmin(coords[:, 1])
    )

    x_span = (
        x_span
        if np.isfinite(x_span) and x_span > 0
        else 1.0
    )
    y_span = (
        y_span
        if np.isfinite(y_span) and y_span > 0
        else 1.0
    )

    aspect = x_span / y_span
    target_points = max(
        int(target_points),
        1,
    )

    if (
        grid_nx is None
        and grid_ny is None
    ):
        nx = max(
            int(
                round(
                    math.sqrt(
                        target_points * aspect
                    )
                )
            ),
            1,
        )
        ny = max(
            int(
                math.ceil(
                    target_points / nx
                )
            ),
            1,
        )
    elif grid_nx is None:
        ny = max(
            int(grid_ny),
            1,
        )
        nx = max(
            int(
                math.ceil(
                    target_points / ny
                )
            ),
            1,
        )
    else:
        nx = max(
            int(grid_nx),
            1,
        )
        ny = max(
            int(
                math.ceil(
                    target_points / nx
                )
            ),
            1,
        )

    return nx, ny


def choose_seed_points_grid(
    coords: np.ndarray,
    max_keep: int,
    seed: int,
    mask: Optional[np.ndarray] = None,
    grid_nx: Optional[int] = None,
    grid_ny: Optional[int] = None,
) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
    """Return selected source points, grid shape, and original indices."""
    coords = np.asarray(
        coords,
        dtype=np.float32,
    )
    original_indices = np.arange(
        len(coords),
        dtype=int,
    )

    if mask is not None:
        candidate_coords = coords[mask]
        candidate_indices = original_indices[mask]
    else:
        candidate_coords = coords
        candidate_indices = original_indices

    finite = np.all(
        np.isfinite(candidate_coords),
        axis=1,
    )
    candidate_coords = candidate_coords[finite]
    candidate_indices = candidate_indices[finite]

    if len(candidate_coords) == 0:
        raise ValueError(
            "No finite source points remain after masking."
        )

    nx, ny = choose_grid_shape(
        candidate_coords,
        target_points=max_keep,
        grid_nx=grid_nx,
        grid_ny=grid_ny,
    )

    x_min = float(
        candidate_coords[:, 0].min()
    )
    x_max = float(
        candidate_coords[:, 0].max()
    )
    y_min = float(
        candidate_coords[:, 1].min()
    )
    y_max = float(
        candidate_coords[:, 1].max()
    )

    x_edges = np.linspace(
        x_min,
        x_max + 1e-8,
        nx + 1,
    )
    y_edges = np.linspace(
        y_min,
        y_max + 1e-8,
        ny + 1,
    )

    best_by_cell: Dict[
        Tuple[int, int],
        Tuple[float, np.ndarray, int],
    ] = {}

    for point, point_idx in zip(
        candidate_coords,
        candidate_indices,
    ):
        ix = int(
            np.clip(
                np.searchsorted(
                    x_edges,
                    point[0],
                    side="right",
                )
                - 1,
                0,
                nx - 1,
            )
        )
        iy = int(
            np.clip(
                np.searchsorted(
                    y_edges,
                    point[1],
                    side="right",
                )
                - 1,
                0,
                ny - 1,
            )
        )

        center = np.array(
            [
                0.5
                * (
                    x_edges[ix]
                    + x_edges[ix + 1]
                ),
                0.5
                * (
                    y_edges[iy]
                    + y_edges[iy + 1]
                ),
            ],
            dtype=float,
        )

        distance2 = float(
            np.sum(
                (point - center) ** 2
            )
        )
        key = (ix, iy)

        if (
            key not in best_by_cell
            or distance2
            < best_by_cell[key][0]
        ):
            best_by_cell[key] = (
                distance2,
                point.copy(),
                int(point_idx),
            )

    selected = np.asarray(
        [
            entry[1]
            for entry
            in best_by_cell.values()
        ],
        dtype=np.float32,
    )
    selected_idx = np.asarray(
        [
            entry[2]
            for entry
            in best_by_cell.values()
        ],
        dtype=int,
    )

    if len(selected) > max_keep:
        rng = np.random.default_rng(seed)
        keep = np.sort(
            rng.choice(
                len(selected),
                size=max_keep,
                replace=False,
            )
        )
        selected = selected[keep]
        selected_idx = selected_idx[keep]

    return (
        selected,
        (nx, ny),
        selected_idx,
    )


# =============================================================================
# INTERMEDIATE VIRTUAL-STATE GEOMETRY
# =============================================================================

def cubic_bezier_2d(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    t = np.linspace(
        0.0,
        1.0,
        int(n_steps),
        dtype=float,
    )[:, None]

    one_minus_t = 1.0 - t

    curve = (
        one_minus_t**3 * p0[None, :]
        + 3.0
        * one_minus_t**2
        * t
        * p1[None, :]
        + 3.0
        * one_minus_t
        * t**2
        * p2[None, :]
        + t**3 * p3[None, :]
    )

    return curve.astype(
        np.float32
    )


def build_pairwise_curve_2d(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    forward_interp,
    backward_interp=None,
    control_scale: float = 0.34,
    n_sections: int = 5,
    steps_per_section: int = 20,
) -> np.ndarray:
    start_xy = np.asarray(
        start_xy,
        dtype=float,
    )
    end_xy = np.asarray(
        end_xy,
        dtype=float,
    )

    displacement = (
        end_xy - start_xy
    )
    distance = float(
        np.linalg.norm(displacement)
    )

    total_steps = (
        n_sections
        * steps_per_section
        + 1
    )

    if (
        not np.isfinite(distance)
        or distance <= 1e-8
    ):
        return np.repeat(
            start_xy[None, :],
            total_steps,
            axis=0,
        ).astype(np.float32)

    displacement_direction = (
        displacement / distance
    )

    start_direction = normalize_vector(
        sample_flow_vector(
            start_xy,
            forward_interp,
        )
    )
    if not np.isfinite(
        start_direction
    ).all():
        start_direction = (
            displacement_direction
        )

    if backward_interp is not None:
        backward_direction = normalize_vector(
            sample_flow_vector(
                end_xy,
                backward_interp,
            )
        )
        end_direction = (
            -backward_direction
            if np.isfinite(
                backward_direction
            ).all()
            else displacement_direction
        )
    else:
        end_direction = (
            displacement_direction
        )

    control_length = (
        control_scale * distance
    )
    control_1 = (
        start_xy
        + control_length
        * start_direction
    )
    control_2 = (
        end_xy
        - control_length
        * end_direction
    )

    return cubic_bezier_2d(
        p0=start_xy,
        p1=control_1,
        p2=control_2,
        p3=end_xy,
        n_steps=total_steps,
    )


def lift_curve_to_3d(
    curve_xy: np.ndarray,
    depth_start: float,
    depth_end: float,
) -> np.ndarray:
    depth = np.linspace(
        depth_start,
        depth_end,
        len(curve_xy),
        dtype=float,
    )

    return np.column_stack(
        [
            depth,
            curve_xy[:, 0],
            curve_xy[:, 1],
        ]
    ).astype(np.float32)


def build_bidirectional_curves(
    seeds_src: np.ndarray,
    ends_tgt: np.ndarray,
    forward_interp,
    backward_interp,
    depth_start: float,
    depth_end: float,
    n_sections: int,
    steps_per_section: int,
    control_scale: float,
) -> List[np.ndarray]:
    """
    Build interpolation trajectories only to sample the intermediate virtual
    locations. These trajectories are never rendered as field lines.
    """
    curves: List[np.ndarray] = []

    for start_xy, end_xy in zip(
        seeds_src,
        ends_tgt,
    ):
        curve_xy = build_pairwise_curve_2d(
            start_xy=start_xy,
            end_xy=end_xy,
            forward_interp=forward_interp,
            backward_interp=backward_interp,
            control_scale=control_scale,
            n_sections=n_sections,
            steps_per_section=steps_per_section,
        )
        curves.append(
            lift_curve_to_3d(
                curve_xy,
                depth_start=depth_start,
                depth_end=depth_end,
            )
        )

    return curves


def collect_projection_points(
    curves_3d: Sequence[np.ndarray],
    boundary_indices: np.ndarray,
    n_virtual: int,
) -> np.ndarray:
    """Collect all curve/virtual-plane intersections."""
    points = []

    for curve in curves_3d:
        for virtual_id in range(
            1,
            n_virtual + 1,
        ):
            point_index = int(
                boundary_indices[virtual_id]
            )
            points.append(
                curve[point_index]
            )

    if not points:
        return np.empty(
            (0, 3),
            dtype=float,
        )

    return np.asarray(
        points,
        dtype=float,
    )


def repeat_payload_for_virtual_slices(
    payload: np.ndarray,
    n_virtual: int,
):
    """Repeat one source payload value/label across all virtual planes."""
    if payload is None or len(payload) == 0:
        return payload

    return np.repeat(
        payload,
        n_virtual,
        axis=0,
    )


# =============================================================================
# 3D PLOTTING HELPERS
# =============================================================================

def draw_real_slice_plane(
    ax,
    depth: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
) -> None:
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    vertices = [
        [
            (depth, x_min, y_min),
            (depth, x_max, y_min),
            (depth, x_max, y_max),
            (depth, x_min, y_max),
        ]
    ]

    plane = Poly3DCollection(
        vertices,
        facecolors=(
            0.92,
            0.92,
            0.92,
            0.055,
        ),
        edgecolors=(
            0.55,
            0.55,
            0.55,
            0.34,
        ),
        linewidths=0.75,
    )
    ax.add_collection3d(
        plane
    )


def draw_virtual_slice_frame(
    ax,
    depth: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    color: str,
    linewidth: float,
    alpha: float,
    linestyle,
) -> None:
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    frame = np.array(
        [
            [depth, x_min, y_min],
            [depth, x_max, y_min],
            [depth, x_max, y_max],
            [depth, x_min, y_max],
            [depth, x_min, y_min],
        ],
        dtype=float,
    )

    ax.plot(
        frame[:, 0],
        frame[:, 1],
        frame[:, 2],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
    )


def add_plane_label(
    ax,
    depth: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    text: str,
    *,
    color: str = "black",
    fontsize: float = 10.0,
) -> None:
    """Place a label beside one 3D tissue plane.

    This reproduces the plane-label geometry used in the SCC 3D visualization:
    the label is anchored near the lower-left corner of the tissue plane, with a
    small offset outside the plane so it does not overlap the point cloud.
    """
    x0, x1 = x_bounds
    y0, y1 = y_bounds

    x_text = x0-200 + 0.5 * (x1 - x0)
    y_text = y0 - 0.2 * (y1 - y0)

    ax.text(
        depth,
        x_text,
        y_text,
        text,
        color=color,
        fontsize=fontsize,
        ha="left",
        va="top",
        zorder=10,
    )


def resolve_slice_z_um(
    sample_id: int,
    slice_z_um: Mapping[int, float] = SLICE_Z_UM,
) -> float:
    """Return biological z for a measured DLPFC slice.

    The explicit mapping is preferred because DLPFC sample indices are not
    uniformly spaced in biological z. A sid*10 fallback is retained only for
    an unlisted integer sample id.
    """
    if int(sample_id) in slice_z_um:
        return float(slice_z_um[int(sample_id)])
    return float(sample_id) * 10.0


def make_plane_labels(
    slice_a: int,
    slice_b: int,
    n_virtual: int,
    slice_z_um: Mapping[int, float] = SLICE_Z_UM,
) -> List[str]:
    """Create labels for the two measured planes and intermediate states.

    Biological z values of virtual planes are linearly interpolated between the
    two measured endpoint z positions. This affects annotation text only and
    does not change the 3D display geometry.
    """
    z_a = resolve_slice_z_um(slice_a, slice_z_um)
    z_b = resolve_slice_z_um(slice_b, slice_z_um)
    z_all = np.linspace(z_a, z_b, int(n_virtual) + 2, dtype=float)

    labels = [f"Slice {slice_a}\n({z_a:g} um)"]
    labels.extend(
        f"Virtual\n({z_value:g} um)"
        for index, z_value in enumerate(z_all[1:-1], start=1)
    )
    labels.append(f"Slice {slice_b}\n({z_b:g} um)")
    return labels


def plot_real_slices_as_gene(
    ax,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    values_a: np.ndarray,
    values_b: np.ndarray,
    depth_a: float,
    depth_b: float,
    point_size: float,
    point_alpha: float,
    cmap_name: str = "Reds",
):
    combined = np.concatenate(
        [
            values_a,
            values_b,
        ]
    )
    finite = combined[
        np.isfinite(combined)
    ]

    vmax = float(
        np.nanpercentile(
            finite,
            99.5,
        )
    )
    norm = Normalize(
        vmin=0.0,
        vmax=max(vmax, 1e-6),
    )
    cmap = plt.get_cmap(
        cmap_name
    )

    for coords, values, depth in (
        (
            coords_a,
            values_a,
            depth_a,
        ),
        (
            coords_b,
            values_b,
            depth_b,
        ),
    ):
        colors = cmap(
            norm(values)
        )
        colors[:, 3] = (
            point_alpha
        )

        ax.scatter(
            np.full(
                len(coords),
                depth,
                dtype=float,
            ),
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=point_size,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )

    return norm, cmap


def plot_real_slices_as_domains(
    ax,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    depth_a: float,
    depth_b: float,
    point_size: float,
    point_alpha: float,
    color_map: Mapping[str, object],
) -> None:
    for coords, labels, depth in (
        (
            coords_a,
            labels_a,
            depth_a,
        ),
        (
            coords_b,
            labels_b,
            depth_b,
        ),
    ):
        colors = [
            color_map[str(label)]
            for label in labels
        ]

        ax.scatter(
            np.full(
                len(coords),
                depth,
                dtype=float,
            ),
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=point_size,
            alpha=point_alpha,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )


def configure_shared_3d_view(
    ax,
    depth_positions: np.ndarray,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    depth_total: float,
    slice_a: int,
    slice_b: int,
    elev: float,
    azim: float,
) -> None:
    """
    Apply the same camera, limits, labels, projection, and box aspect to both
    Figure 1 and Figure 2.
    """
    x_pad = (
        XY_PAD_RATIO
        * max(
            x_bounds[1]
            - x_bounds[0],
            1.0,
        )
    )
    y_pad = (
        XY_PAD_RATIO
        * max(
            y_bounds[1]
            - y_bounds[0],
            1.0,
        )
    )
    depth_pad = (
        DEPTH_PAD_RATIO
        * max(
            depth_total,
            1.0,
        )
    )

    ax.set_proj_type(
        PROJECTION_TYPE
    )

    ax.set_xlim(
        depth_positions[0] - depth_pad,
        depth_positions[-1] + depth_pad,
    )
    ax.set_ylim(
        x_bounds[1] + x_pad,
        x_bounds[0] - x_pad,
    )
    ax.set_zlim(
        y_bounds[1] + y_pad,
        y_bounds[0] - y_pad,
    )

    # slice_labels = (
    #     [f"Real {slice_a}"]
    #     + [
    #         f"V{index}"
    #         for index in range(
    #             1,
    #             len(depth_positions) - 1,
    #         )
    #     ]
    #     + [f"Real {slice_b}"]
    # )
    #
    # ax.set_xticks(
    #     depth_positions
    # )
    # ax.set_xticklabels(
    #     slice_labels,
    #     fontsize=8.5,
    # )
    ax.set_xticks([]
    )
    ax.set_yticks([])
    ax.set_zticks([])

    # ax.set_xlabel(
    #     "Dense reconstruction along depth",
    #     labelpad=10,
    # )
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.view_init(
        elev=elev,
        azim=azim,
    )
    ax.grid(False)

    ax.set_box_aspect(
        (
            max(depth_total, 1.0)
            * DEPTH_ASPECT_SCALE,
            max(
                x_bounds[1]
                - x_bounds[0],
                1.0,
            ),
            max(
                y_bounds[1]
                - y_bounds[0],
                1.0,
            ),
        )
    )

    for axis in (
        ax.xaxis,
        ax.yaxis,
        ax.zaxis,
    ):
        axis.pane.fill = False
        axis.pane.set_edgecolor(
            (
                0.82,
                0.82,
                0.82,
                0.25,
            )
        )


def draw_common_planes_and_real_slices(
    ax,
    depth_positions: np.ndarray,
    plane_labels: Sequence[str],
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    real_slice_mode: str,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    gene_a: Optional[np.ndarray],
    gene_b: Optional[np.ndarray],
    domain_labels_a: Optional[np.ndarray],
    domain_labels_b: Optional[np.ndarray],
    domain_color_map: Optional[Mapping[str, object]],
    real_point_size: float,
    real_point_alpha: float,
    virtual_frame_color: str,
    virtual_frame_linewidth: float,
    virtual_frame_alpha: float,
    virtual_frame_dash_pattern,
    plane_label_fontsize: float = PLANE_LABEL_FONTSIZE,
    plane_label_real_color: str = PLANE_LABEL_REAL_COLOR,
    show_virtual_plane_labels: bool = SHOW_VIRTUAL_PLANE_LABELS,
):
    """
    Draw content shared by both figures:
      - two endpoint real-slice planes;
      - all dashed intermediate virtual frames;
      - direct plane labels for measured and virtual states;
      - the two real-slice spatial maps.

    Returns the gene normalization/cmap in gene mode.
    """
    if len(plane_labels) != len(depth_positions):
        raise ValueError(
            "plane_labels and depth_positions must have the same length."
        )

    # Measured endpoint A.
    draw_real_slice_plane(
        ax,
        depth=depth_positions[0],
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    )
    add_plane_label(
        ax,
        depth_positions[0],
        x_bounds,
        y_bounds,
        plane_labels[0],
        color=plane_label_real_color,
        fontsize=plane_label_fontsize,
    )

    # Intermediate virtual locations.
    for virtual_index, virtual_depth in enumerate(
        depth_positions[1:-1],
        start=1,
    ):
        draw_virtual_slice_frame(
            ax,
            depth=virtual_depth,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            color=virtual_frame_color,
            linewidth=virtual_frame_linewidth,
            alpha=virtual_frame_alpha,
            linestyle=virtual_frame_dash_pattern,
        )
        if show_virtual_plane_labels:
            add_plane_label(
                ax,
                virtual_depth,
                x_bounds,
                y_bounds,
                plane_labels[virtual_index],
                color=virtual_frame_color,
                fontsize=plane_label_fontsize,
            )

    # Measured endpoint B.
    draw_real_slice_plane(
        ax,
        depth=depth_positions[-1],
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    )
    add_plane_label(
        ax,
        depth_positions[-1],
        x_bounds,
        y_bounds,
        plane_labels[-1],
        color=plane_label_real_color,
        fontsize=plane_label_fontsize,
    )

    gene_norm = None
    gene_cmap = None

    if real_slice_mode == "gene":
        gene_norm, gene_cmap = plot_real_slices_as_gene(
            ax,
            coords_a=coords_a,
            coords_b=coords_b,
            values_a=gene_a,
            values_b=gene_b,
            depth_a=depth_positions[0],
            depth_b=depth_positions[-1],
            point_size=real_point_size,
            point_alpha=real_point_alpha,
            cmap_name="Reds",
        )
    else:
        plot_real_slices_as_domains(
            ax,
            coords_a=coords_a,
            coords_b=coords_b,
            labels_a=domain_labels_a,
            labels_b=domain_labels_b,
            depth_a=depth_positions[0],
            depth_b=depth_positions[-1],
            point_size=real_point_size,
            point_alpha=real_point_alpha,
            color_map=domain_color_map,
        )

    return gene_norm, gene_cmap


def add_legends(
    fig,
    ax,
    real_slice_mode: str,
    gene: str,
    use_log1p: bool,
    gene_norm,
    gene_cmap,
    domain_color_map: Optional[Mapping[str, object]],
    virtual_frame_color: str,
    virtual_frame_linewidth: float,
    virtual_frame_dash_pattern,
    include_virtual_points: bool,
    projection_marker_a_to_b: str,
    projection_marker_b_to_a: str,
    projection_point_size: float,
    projection_point_edgecolor: str,
) -> None:
    """Add only legends relevant to what is actually visible."""
    legend_handles = []

    if real_slice_mode == "gene":
        scalar_mappable = mpl.cm.ScalarMappable(
            norm=gene_norm,
            cmap=gene_cmap,
        )
        scalar_mappable.set_array([])

        colorbar = fig.colorbar(
            scalar_mappable,
            ax=ax,
            fraction=0.030,
            pad=0.01,
            shrink=0.22,
        )
        colorbar.set_label(
            f"log1p({gene})"
            if use_log1p
            else gene,
            fontsize=9,
        )
        colorbar.ax.tick_params(
            labelsize=8
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=5.2,
                markerfacecolor=gene_cmap(0.75),
                markeredgecolor="none",
                label="Real slices: gene map",
            )
        )

    else:
        domain_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=5.0,
                label=label,
            )
            for label, color
            in domain_color_map.items()
        ]

        domain_legend = ax.legend(
            handles=domain_handles,
            title="Layers",
            loc="upper right",
            bbox_to_anchor=(1.10, 0.60),
            frameon=False,
            fontsize=7.5,
            title_fontsize=8.0,
        )
        ax.add_artist(
            domain_legend
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=virtual_frame_color,
            linewidth=virtual_frame_linewidth,
            linestyle=virtual_frame_dash_pattern,
            label="Virtual slice",
        )
    )

    if include_virtual_points:
        marker_size = max(
            np.sqrt(
                projection_point_size
            ),
            4.0,
        )

        legend_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker=projection_marker_a_to_b,
                    linestyle="",
                    markersize=marker_size,
                    markerfacecolor=(
                        gene_cmap(0.72)
                        if real_slice_mode == "gene"
                        else "0.45"
                    ),
                    markeredgecolor=projection_point_edgecolor,
                    label="Projected A→Q",
                ),
                Line2D(
                    [0],
                    [0],
                    marker=projection_marker_b_to_a,
                    linestyle="",
                    markersize=marker_size,
                    markerfacecolor=(
                        gene_cmap(0.92)
                        if real_slice_mode == "gene"
                        else "0.70"
                    ),
                    markeredgecolor=projection_point_edgecolor,
                    label="Projected B→Q",
                ),
            ]
        )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.18, 0.80),
        frameon=False,
        fontsize=8,
    )


def scatter_virtual_states(
    ax,
    real_slice_mode: str,
    proj_ab: np.ndarray,
    proj_ba: np.ndarray,
    proj_payload_ab,
    proj_payload_ba,
    gene_norm,
    gene_cmap,
    domain_color_map,
    projection_point_size: float,
    projection_point_alpha: float,
    projection_point_edgecolor: str,
    projection_point_linewidth: float,
    projection_marker_a_to_b: str,
    projection_marker_b_to_a: str,
) -> None:
    """Scatter the filled intermediate virtual states used only in Figure 2."""
    if real_slice_mode == "gene":
        if len(proj_ab) > 0:
            colors = gene_cmap(
                gene_norm(
                    proj_payload_ab
                )
            )
            colors[:, 3] = (
                projection_point_alpha
            )
            ax.scatter(
                proj_ab[:, 0],
                proj_ab[:, 1],
                proj_ab[:, 2],
                c=colors,
                s=projection_point_size,
                edgecolors=projection_point_edgecolor,
                linewidths=projection_point_linewidth,
                depthshade=False,
                rasterized=True,
                marker=projection_marker_a_to_b,
                zorder=7,
            )

        if len(proj_ba) > 0:
            colors = gene_cmap(
                gene_norm(
                    proj_payload_ba
                )
            )
            colors[:, 3] = (
                projection_point_alpha
            )
            ax.scatter(
                proj_ba[:, 0],
                proj_ba[:, 1],
                proj_ba[:, 2],
                c=colors,
                s=projection_point_size,
                edgecolors=projection_point_edgecolor,
                linewidths=projection_point_linewidth,
                depthshade=False,
                rasterized=True,
                marker=projection_marker_b_to_a,
                zorder=7,
            )

    else:
        if len(proj_ab) > 0:
            colors = [
                domain_color_map[str(label)]
                for label in proj_payload_ab
            ]
            ax.scatter(
                proj_ab[:, 0],
                proj_ab[:, 1],
                proj_ab[:, 2],
                c=colors,
                s=projection_point_size,
                alpha=projection_point_alpha,
                edgecolors=projection_point_edgecolor,
                linewidths=projection_point_linewidth,
                depthshade=False,
                rasterized=True,
                marker=projection_marker_a_to_b,
                zorder=7,
            )

        if len(proj_ba) > 0:
            colors = [
                domain_color_map[str(label)]
                for label in proj_payload_ba
            ]
            ax.scatter(
                proj_ba[:, 0],
                proj_ba[:, 1],
                proj_ba[:, 2],
                c=colors,
                s=projection_point_size,
                alpha=projection_point_alpha,
                edgecolors=projection_point_edgecolor,
                linewidths=projection_point_linewidth,
                depthshade=False,
                rasterized=True,
                marker=projection_marker_b_to_a,
                zorder=7,
            )


# =============================================================================
# SAVE / SHOW
# =============================================================================

def save_or_show(
    fig: plt.Figure,
    outdir: Optional[Path],
    stem: str,
    dpi: int = 350,
) -> None:
    if outdir is None:
        plt.show()
        return

    outdir = Path(outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        outdir
        / f"{stem}_{REAL_SLICE_MODE}.png"
    )
    # pdf_path = (
    #     outdir
    #     / f"{stem}.pdf"
    # )

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    # fig.savefig(
    #     pdf_path,
    #     bbox_inches="tight",
    #     pad_inches=0.04,
    # )
    plt.close(fig)

    print(
        f"[OK] saved: {png_path}"
    )
    # print(
    #     f"[OK] saved: {pdf_path}"
    # )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def plot_dense_reconstruction_two_stage(
    root: str = ROOT,
    group_id: int = GROUP_ID,
    slice_a: int = SLICE_A,
    slice_b: int = SLICE_B,
    pairwise_subdir: str = PAIRWISE_SUBDIR,
    coord_key: str = COORD_KEY,
    real_slice_mode: str = REAL_SLICE_MODE,
    gene: str = GENE,
    use_log1p: bool = USE_LOG1P,
    real_domain_key: str = REAL_DOMAIN_KEY,
    real_domain_cmap: str = REAL_DOMAIN_CMAP,
    preferred_domain_order: Sequence[str] = PREFERRED_DOMAIN_ORDER,
    explicit_domain_colors: Mapping[str, str] = DLPFC_DOMAIN_COLORS,
    n_virtual: int = N_VIRTUAL,
    depth_total: float = DEPTH_TOTAL,
    max_seed_points: int = MAX_SEED_POINTS,
    seed_grid_nx: Optional[int] = SEED_GRID_NX,
    seed_grid_ny: Optional[int] = SEED_GRID_NY,
    random_seed: int = RANDOM_SEED,
    trajectory_domain_key: str = TRAJECTORY_DOMAIN_KEY,
    trajectory_domains: Optional[Sequence[str]] = TRAJECTORY_DOMAINS,
    control_scale: float = CONTROL_SCALE,
    steps_per_section: int = STEPS_PER_SECTION,
    real_point_size: float = REAL_POINT_SIZE,
    real_point_alpha: float = REAL_POINT_ALPHA,
    projection_point_size: float = PROJECTION_POINT_SIZE,
    projection_point_alpha: float = PROJECTION_POINT_ALPHA,
    projection_point_edgecolor: str = PROJECTION_POINT_EDGECOLOR,
    projection_point_linewidth: float = PROJECTION_POINT_LINEWIDTH,
    projection_marker_a_to_b: str = PROJECTION_MARKER_A_TO_B,
    projection_marker_b_to_a: str = PROJECTION_MARKER_B_TO_A,
    virtual_frame_color: str = VIRTUAL_FRAME_COLOR,
    virtual_frame_linewidth: float = VIRTUAL_FRAME_LINEWIDTH,
    virtual_frame_alpha: float = VIRTUAL_FRAME_ALPHA,
    virtual_frame_dash_pattern=VIRTUAL_FRAME_DASH_PATTERN,
    elev: float = ELEV,
    azim: float = AZIM,
    figsize: Tuple[float, float] = FIGSIZE,
    outdir: Optional[Path] = OUT_DIR,
    stem_prefix: str = STEM_PREFIX,
):
    """
    Generate:
      Figure 1: measured endpoints + empty dashed virtual-plane frames.
      Figure 2: same view + filled projected virtual states.

    No transition-field lines are rendered.
    """
    set_publication_style()

    real_slice_mode = str(
        real_slice_mode
    ).lower()

    if real_slice_mode not in {
        "gene",
        "domain",
    }:
        raise ValueError(
            "real_slice_mode must be 'gene' or 'domain'."
        )

    if n_virtual < 1:
        raise ValueError(
            "n_virtual must be at least 1."
        )

    # ------------------------------------------------------------------
    # Load measured endpoint slices.
    # ------------------------------------------------------------------
    adata_a, coords_a = load_slice_native(
        root,
        group_id,
        slice_a,
        coord_key=coord_key,
    )
    adata_b, coords_b = load_slice_native(
        root,
        group_id,
        slice_b,
        coord_key=coord_key,
    )

    # ------------------------------------------------------------------
    # Prepare endpoint display and virtual-state payload.
    # ------------------------------------------------------------------
    gene_a = None
    gene_b = None
    domain_labels_a = None
    domain_labels_b = None
    domain_color_map = None

    if real_slice_mode == "gene":
        gene_a = extract_gene_values(
            adata_a,
            gene,
            use_log1p=use_log1p,
        )
        gene_b = extract_gene_values(
            adata_b,
            gene,
            use_log1p=use_log1p,
        )
    else:
        if real_domain_key not in adata_a.obs.columns:
            raise KeyError(
                f"{real_domain_key!r} not found in slice {slice_a} obs."
            )
        if real_domain_key not in adata_b.obs.columns:
            raise KeyError(
                f"{real_domain_key!r} not found in slice {slice_b} obs."
            )

        labels_a_raw = (
            adata_a.obs[real_domain_key]
            .astype(str)
            .to_numpy()
        )
        labels_b_raw = (
            adata_b.obs[real_domain_key]
            .astype(str)
            .to_numpy()
        )

        (
            domain_labels_a,
            domain_labels_b,
            domain_color_map,
        ) = build_domain_color_map(
            labels_a_raw,
            labels_b_raw,
            preferred_order=preferred_domain_order,
            explicit_colors=explicit_domain_colors,
            fallback_cmap=real_domain_cmap,
        )

    # ------------------------------------------------------------------
    # Select source points used for virtual-state interpolation.
    # ------------------------------------------------------------------
    source_mask_a = make_domain_mask(
        adata_a,
        trajectory_domain_key,
        trajectory_domains,
        len(coords_a),
    )
    source_mask_b = make_domain_mask(
        adata_b,
        trajectory_domain_key,
        trajectory_domains,
        len(coords_b),
    )

    (
        seeds_a,
        seed_grid_shape_a,
        seed_idx_a,
    ) = choose_seed_points_grid(
        coords_a,
        max_keep=max_seed_points,
        seed=random_seed,
        mask=source_mask_a,
        grid_nx=seed_grid_nx,
        grid_ny=seed_grid_ny,
    )

    (
        seeds_b,
        seed_grid_shape_b,
        seed_idx_b,
    ) = choose_seed_points_grid(
        coords_b,
        max_keep=max_seed_points,
        seed=random_seed,
        mask=source_mask_b,
        grid_nx=seed_grid_nx,
        grid_ny=seed_grid_ny,
    )

    # ------------------------------------------------------------------
    # Load pairwise flows and obtain endpoint correspondences.
    # ------------------------------------------------------------------
    flow_a_to_b = ensure_flow_hw2(
        np.load(
            parse_flow_path(
                root,
                group_id,
                fixed_id=slice_b,
                moving_id=slice_a,
                pairwise_subdir=pairwise_subdir,
            )
        )
    )
    flow_b_to_a = ensure_flow_hw2(
        np.load(
            parse_flow_path(
                root,
                group_id,
                fixed_id=slice_a,
                moving_id=slice_b,
                pairwise_subdir=pairwise_subdir,
            )
        )
    )

    interp_a_to_b = make_flow_interpolator(
        flow_a_to_b
    )
    interp_b_to_a = make_flow_interpolator(
        flow_b_to_a
    )

    ends_b_from_a = inverse_warp_points(
        seeds_a,
        flow_a_to_b,
    )
    ends_a_from_b = inverse_warp_points(
        seeds_b,
        flow_b_to_a,
    )

    h_ab, w_ab = (
        flow_a_to_b.shape[:2]
    )
    h_ba, w_ba = (
        flow_b_to_a.shape[:2]
    )

    valid_ab = np.all(
        np.isfinite(
            ends_b_from_a
        ),
        axis=1,
    )
    valid_ab &= (
        (
            ends_b_from_a[:, 0]
            >= 0
        )
        & (
            ends_b_from_a[:, 0]
            < w_ab
        )
        & (
            ends_b_from_a[:, 1]
            >= 0
        )
        & (
            ends_b_from_a[:, 1]
            < h_ab
        )
    )

    seeds_a = seeds_a[valid_ab]
    ends_b_from_a = ends_b_from_a[valid_ab]
    seed_idx_a = seed_idx_a[valid_ab]

    valid_ba = np.all(
        np.isfinite(
            ends_a_from_b
        ),
        axis=1,
    )
    valid_ba &= (
        (
            ends_a_from_b[:, 0]
            >= 0
        )
        & (
            ends_a_from_b[:, 0]
            < w_ba
        )
        & (
            ends_a_from_b[:, 1]
            >= 0
        )
        & (
            ends_a_from_b[:, 1]
            < h_ba
        )
    )

    seeds_b = seeds_b[valid_ba]
    ends_a_from_b = ends_a_from_b[valid_ba]
    seed_idx_b = seed_idx_b[valid_ba]

    if (
        len(seeds_a) == 0
        and len(seeds_b) == 0
    ):
        raise RuntimeError(
            "No valid bidirectional correspondences remained after warping."
        )

    # Source evidence carried to the virtual planes.
    if real_slice_mode == "gene":
        proj_values_ab = (
            gene_a[seed_idx_a]
            if len(seed_idx_a)
            else np.empty(
                0,
                dtype=float,
            )
        )
        proj_values_ba = (
            gene_b[seed_idx_b]
            if len(seed_idx_b)
            else np.empty(
                0,
                dtype=float,
            )
        )
    else:
        proj_labels_ab = (
            domain_labels_a[seed_idx_a]
            if len(seed_idx_a)
            else np.empty(
                0,
                dtype=str,
            )
        )
        proj_labels_ba = (
            domain_labels_b[seed_idx_b]
            if len(seed_idx_b)
            else np.empty(
                0,
                dtype=str,
            )
        )

    # ------------------------------------------------------------------
    # Construct trajectories only to sample virtual-plane intersections.
    # ------------------------------------------------------------------
    n_sections = (
        n_virtual + 1
    )
    depth_positions = np.linspace(
        0.0,
        float(depth_total),
        n_virtual + 2,
        dtype=float,
    )
    plane_labels = make_plane_labels(
        slice_a=slice_a,
        slice_b=slice_b,
        n_virtual=n_virtual,
    )
    boundary_indices = (
        np.arange(
            n_sections + 1
        )
        * steps_per_section
    )

    curves_a_to_b = build_bidirectional_curves(
        seeds_src=seeds_a,
        ends_tgt=ends_b_from_a,
        forward_interp=interp_a_to_b,
        backward_interp=interp_b_to_a,
        depth_start=depth_positions[0],
        depth_end=depth_positions[-1],
        n_sections=n_sections,
        steps_per_section=steps_per_section,
        control_scale=control_scale,
    )

    curves_b_to_a = build_bidirectional_curves(
        seeds_src=seeds_b,
        ends_tgt=ends_a_from_b,
        forward_interp=interp_b_to_a,
        backward_interp=interp_a_to_b,
        depth_start=depth_positions[-1],
        depth_end=depth_positions[0],
        n_sections=n_sections,
        steps_per_section=steps_per_section,
        control_scale=control_scale,
    )

    proj_ab = collect_projection_points(
        curves_a_to_b,
        boundary_indices,
        n_virtual=n_virtual,
    )
    proj_ba = collect_projection_points(
        curves_b_to_a,
        boundary_indices,
        n_virtual=n_virtual,
    )

    if real_slice_mode == "gene":
        proj_payload_ab = repeat_payload_for_virtual_slices(
            np.asarray(
                proj_values_ab
            ),
            n_virtual,
        )
        proj_payload_ba = repeat_payload_for_virtual_slices(
            np.asarray(
                proj_values_ba
            ),
            n_virtual,
        )
    else:
        proj_payload_ab = repeat_payload_for_virtual_slices(
            np.asarray(
                proj_labels_ab,
                dtype=object,
            ),
            n_virtual,
        )
        proj_payload_ba = repeat_payload_for_virtual_slices(
            np.asarray(
                proj_labels_ba,
                dtype=object,
            ),
            n_virtual,
        )

    print(
        f"[INFO] real-slice mode: {real_slice_mode} | "
        f"A->B sources: {len(seeds_a)} "
        f"(grid {seed_grid_shape_a[0]}x{seed_grid_shape_a[1]}) | "
        f"B->A sources: {len(seeds_b)} "
        f"(grid {seed_grid_shape_b[0]}x{seed_grid_shape_b[1]})"
    )

    # ------------------------------------------------------------------
    # Shared 3D extents used identically by Figure 1 and Figure 2.
    # ------------------------------------------------------------------
    all_x = np.concatenate(
        [
            coords_a[:, 0],
            coords_b[:, 0],
        ]
    )
    all_y = np.concatenate(
        [
            coords_a[:, 1],
            coords_b[:, 1],
        ]
    )

    x_bounds = (
        float(
            np.nanmin(all_x)
        ),
        float(
            np.nanmax(all_x)
        ),
    )
    y_bounds = (
        float(
            np.nanmin(all_y)
        ),
        float(
            np.nanmax(all_y)
        ),
    )

    # ==================================================================
    # FIGURE 1
    # Endpoint real slices + empty dashed virtual frames.
    # ==================================================================
    fig1 = plt.figure(
        figsize=figsize
    )
    ax1 = fig1.add_subplot(
        111,
        projection="3d",
    )

    (
        gene_norm_1,
        gene_cmap_1,
    ) = draw_common_planes_and_real_slices(
        ax=ax1,
        depth_positions=depth_positions,
        plane_labels=plane_labels,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        real_slice_mode=real_slice_mode,
        coords_a=coords_a,
        coords_b=coords_b,
        gene_a=gene_a,
        gene_b=gene_b,
        domain_labels_a=domain_labels_a,
        domain_labels_b=domain_labels_b,
        domain_color_map=domain_color_map,
        real_point_size=real_point_size,
        real_point_alpha=real_point_alpha,
        virtual_frame_color=virtual_frame_color,
        virtual_frame_linewidth=virtual_frame_linewidth,
        virtual_frame_alpha=virtual_frame_alpha,
        virtual_frame_dash_pattern=virtual_frame_dash_pattern,
        show_virtual_plane_labels = False,
    )

    configure_shared_3d_view(
        ax=ax1,
        depth_positions=depth_positions,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        depth_total=depth_total,
        slice_a=slice_a,
        slice_b=slice_b,
        elev=elev,
        azim=azim,
    )
    #
    # ax1.set_title(
    #     "Dense reconstruction setup\n"
    #     "Measured endpoint slices and target virtual locations",
    #     fontsize=12.5,
    #     fontweight="bold",
    #     pad=16,
    # )
    #
    # add_legends(
    #     fig=fig1,
    #     ax=ax1,
    #     real_slice_mode=real_slice_mode,
    #     gene=gene,
    #     use_log1p=use_log1p,
    #     gene_norm=gene_norm_1,
    #     gene_cmap=gene_cmap_1,
    #     domain_color_map=domain_color_map,
    #     virtual_frame_color=virtual_frame_color,
    #     virtual_frame_linewidth=virtual_frame_linewidth,
    #     virtual_frame_dash_pattern=virtual_frame_dash_pattern,
    #     include_virtual_points=False,
    #     projection_marker_a_to_b=projection_marker_a_to_b,
    #     projection_marker_b_to_a=projection_marker_b_to_a,
    #     projection_point_size=projection_point_size,
    #     projection_point_edgecolor=projection_point_edgecolor,
    # )

    fig1.tight_layout()

    save_or_show(
        fig1,
        outdir=outdir,
        stem=f"{stem_prefix}_figure1_empty_virtual_frames",
    )

    # ==================================================================
    # FIGURE 2
    # Same view + filled intermediate virtual states.
    # ==================================================================
    fig2 = plt.figure(
        figsize=figsize
    )
    ax2 = fig2.add_subplot(
        111,
        projection="3d",
    )

    (
        gene_norm_2,
        gene_cmap_2,
    ) = draw_common_planes_and_real_slices(
        ax=ax2,
        depth_positions=depth_positions,
        plane_labels=plane_labels,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        real_slice_mode=real_slice_mode,
        coords_a=coords_a,
        coords_b=coords_b,
        gene_a=gene_a,
        gene_b=gene_b,
        domain_labels_a=domain_labels_a,
        domain_labels_b=domain_labels_b,
        domain_color_map=domain_color_map,
        real_point_size=real_point_size,
        real_point_alpha=real_point_alpha,
        virtual_frame_color=virtual_frame_color,
        virtual_frame_linewidth=virtual_frame_linewidth,
        virtual_frame_alpha=virtual_frame_alpha,
        virtual_frame_dash_pattern=virtual_frame_dash_pattern,
    )

    scatter_virtual_states(
        ax=ax2,
        real_slice_mode=real_slice_mode,
        proj_ab=proj_ab,
        proj_ba=proj_ba,
        proj_payload_ab=proj_payload_ab,
        proj_payload_ba=proj_payload_ba,
        gene_norm=gene_norm_2,
        gene_cmap=gene_cmap_2,
        domain_color_map=domain_color_map,
        projection_point_size=projection_point_size,
        projection_point_alpha=projection_point_alpha,
        projection_point_edgecolor=projection_point_edgecolor,
        projection_point_linewidth=projection_point_linewidth,
        projection_marker_a_to_b=projection_marker_a_to_b,
        projection_marker_b_to_a=projection_marker_b_to_a,
    )

    configure_shared_3d_view(
        ax=ax2,
        depth_positions=depth_positions,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        depth_total=depth_total,
        slice_a=slice_a,
        slice_b=slice_b,
        elev=elev,
        azim=azim,
    )

    if real_slice_mode == "gene":
        title_detail = (
            f"{gene} on real slices and projected virtual states"
        )
    else:
        title_detail = (
            f"{real_domain_key} on real slices and projected virtual states"
        )

    # ax2.set_title(
    #     "Dense reconstruction with intermediate virtual states\n"
    #     f"{title_detail}",
    #     fontsize=12.5,
    #     fontweight="bold",
    #     pad=16,
    # )
    #
    # add_legends(
    #     fig=fig2,
    #     ax=ax2,
    #     real_slice_mode=real_slice_mode,
    #     gene=gene,
    #     use_log1p=use_log1p,
    #     gene_norm=gene_norm_2,
    #     gene_cmap=gene_cmap_2,
    #     domain_color_map=domain_color_map,
    #     virtual_frame_color=virtual_frame_color,
    #     virtual_frame_linewidth=virtual_frame_linewidth,
    #     virtual_frame_dash_pattern=virtual_frame_dash_pattern,
    #     include_virtual_points=True,
    #     projection_marker_a_to_b=projection_marker_a_to_b,
    #     projection_marker_b_to_a=projection_marker_b_to_a,
    #     projection_point_size=projection_point_size,
    #     projection_point_edgecolor=projection_point_edgecolor,
    # )

    fig2.tight_layout()

    save_or_show(
        fig2,
        outdir=outdir,
        stem=f"{stem_prefix}_figure2_filled_virtual_states",
    )

    return {
        "figure1": {
            "fig": fig1,
            "ax": ax1,
        },
        "figure2": {
            "fig": fig2,
            "ax": ax2,
        },
        "depth_positions": depth_positions,
        "plane_labels": plane_labels,
        "projection_a_to_b": proj_ab,
        "projection_b_to_a": proj_ba,
        "seeds_a": seeds_a,
        "seeds_b": seeds_b,
        "ends_b_from_a": ends_b_from_a,
        "ends_a_from_b": ends_a_from_b,
        "seed_indices_a": seed_idx_a,
        "seed_indices_b": seed_idx_b,
    }


if __name__ == "__main__":
    plot_dense_reconstruction_two_stage()
