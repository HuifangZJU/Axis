#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Left-to-right 3D time-axis pipeline figures for the Tg mouse-brain analysis.

Core visual design
------------------
- Biological time runs LEFT -> RIGHT.
- The tissue maps are the only objects rendered in 3D.
- The timeline is a 2D screen-space scaffold, so it is always exactly
  horizontal relative to the figure bounding box, independent of the 3D camera.
- Each tissue map is visually "hung" from its corresponding age marker by a
  vertical screen-space suspension line connected to the projected top-center
  of the tissue frame.
- The 3D spatial maps can be rotated with --view-elev / --view-azim without
  tilting the timeline.
- Real slices use solid frames; virtual slices use dashed frames.
- In Figure 3, virtual gene/domain points use subtle gray marker edges.

Figures
-------
1. Real slices: spatial-omics gene maps along a horizontal age scaffold.
2. Real slices plus virtual-query spatial support: mirrored real gene/domain
   maps together with 256 x 256 support pixels at 4.1, 8.25, 11.8, and 14.85m.
   Query-time support planes use dashed frames.
3. Real + virtual slices: mirrored gene/domain maps along the same scaffold.
"""

from axis.settings import data_path
import os
import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# ============================================================
# DATASET CONFIG
# ============================================================
ROOT = data_path("mouse_brain")
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ads", "Tg")
VIRTUAL_DIR = os.path.join(
    ROOT,
    "virtual_tissue",
    "AD_time_grid256_spatial2.0",
    "query",
)

OUTDIR = data_path("paper_analysis/Figure56_mouse/5a/")

GROUP_ID = 0

REAL_SLICE_SPECS = [
    {"age": 2.5, "label": "2.5", "sample_id": 0},
    {"age": 5.7, "label": "5.7", "sample_id": 1},
    {"age": 17.9, "label": "17.9", "sample_id": 2},
]

VIRTUAL_SLICE_SPECS = [
    {"age": 4.1, "label": "4.1", "query_value": 4.100},
    {"age": 8.25, "label": "8.25", "query_value": 8.250},
    {"age": 11.8, "label": "11.8", "query_value": 11.800},
    {"age": 14.85, "label": "14.85", "query_value": 14.850},
]

GENE_NAME = "Gfap"
REAL_DOMAIN_KEY = "axis_joint_domain"
VIRTUAL_DOMAIN_KEY = "axis_joint_domain"
VIRTUAL_COORD_KEY = "spatial"


# ============================================================
# PLOTTING CONFIG
# ============================================================
FIG_DPI = 400
FIGURE_MODE = "both"  # "save", "show", "both"
SHOW_BLOCK = True

# Final display orientation of loaded spatial coordinates.
SWAP_XY = True
INVERT_X = True
INVERT_Y = False

# Shared spatial frame for all slices.
FIXED_BOX = (0.0, 1024.0, 0.0, 1024.0)

# Mirrored geometry: gene on left, horizontally mirrored domains on right.
MIRROR_GAP = -20.0
MIRROR_CENTER_FOLD = 0.0

# Time is now the WORLD-X coordinate.
# Older ages move farther to the right.
TIME_SCALE_PER_MONTH = 95.0

# Uniform visual scaling of the spatial Y/Z planes relative to temporal X.
SPATIAL_DISPLAY_SCALE = 0.3

# 3D point appearance.
POINT_SIZE_GENE = 0.5
POINT_SIZE_DOMAIN = 0.5
MAX_POINTS_PER_SLICE = 25000

# Figure 2 virtual-query support. The support is shown as a full 256 x 256
# pixel raster over the whole spatial plane, rendered with grid lines instead
# of occupied-pixel markers.
VIRTUAL_SUPPORT_GRID_SIZE = 64
VIRTUAL_SUPPORT_GRID_COLOR = "#808080"
VIRTUAL_SUPPORT_GRID_ALPHA = 0.82
VIRTUAL_SUPPORT_GRID_LW = 0.28

REAL_ALPHA = 0.72
VIRTUAL_ALPHA = 0.72
DOMAIN_ALPHA = 0.62

# Figure 3 only: virtual gene/domain points receive a subtle marker boundary,
# matching the population-pipeline convention.
FIG3_VIRTUAL_MARKER_EDGE_COLOR = "0.55"
FIG3_VIRTUAL_MARKER_EDGE_WIDTH = 0.02

REAL_FRAME_COLOR = "#9A9A9A"
VIRTUAL_FRAME_COLOR = "#9A9A9A"
REAL_LABEL_COLOR = "#1F1F1F"
VIRTUAL_LABEL_COLOR = "#8D8D8D"
BACKGROUND_GRAY = "#D0D0D0"

SOLID_LW = 0.7
DASHED_LW = 0.7
DASHED_STYLE = (0, (4.0, 2.4))

# Screen-space horizontal timeline styling.
TIMELINE_COLOR = "#2F2F2F"
TIMELINE_LW = 1.9
TIMELINE_ARROW_SCALE = 13.0
TIMELINE_MARKER_SIZE = 7.8
TIMELINE_MARKER_EDGE_COLOR = "white"
TIMELINE_MARKER_EDGE_WIDTH = 0.9
TIMELINE_CLEARANCE_FIG = 0.010
TIMELINE_MIN_Y_FIG = 0.535
TIMELINE_MAX_Y_FIG = 0.905
TIMELINE_X_PAD_FIG = 0.020
TIMELINE_LABEL_OFFSET_FIG = 0.01
TIMELINE_LABEL_SECOND_TIER_FIG = 0.036

# Suspension / hanger styling.
HANGER_COLOR = "#7A7A7A"
HANGER_LW = 0.85
HANGER_ALPHA = 0.55
HANGER_TOP_GAP_FIG = 0.006
HANGER_BOTTOM_GAP_FIG = 0.004
HANGER_ANCHOR_SIZE = 3.8

# Camera applies ONLY to the spatial maps.
VIEW_ELEV = 5.0
VIEW_AZIM = -50.0
VIEW_ROLL = 0.0

# Domain colors.
ALL_LABELS = list(range(13))
BASE_CMAP = plt.get_cmap("tab20")
LABEL_TO_COLOR = {lab: BASE_CMAP(i % 20) for i, lab in enumerate(ALL_LABELS)}


# ============================================================
# DATA STRUCTURE
# ============================================================
@dataclass
class SliceRecord:
    age: float
    label: str
    path: str
    is_real: bool
    coords: np.ndarray
    gene_values: np.ndarray
    domains: np.ndarray


# ============================================================
# BASIC HELPERS
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finalize_figure(fig, save_path: str, figure_mode: Optional[str] = None) -> None:
    mode = FIGURE_MODE if figure_mode is None else str(figure_mode).lower()
    if mode not in {"save", "show", "both"}:
        raise ValueError(f"Invalid figure mode {mode!r}.")

    if mode in {"save", "both"}:
        ensure_dir(os.path.dirname(save_path) or ".")
        fig.savefig(
            save_path,
            dpi=FIG_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"[SAVE] {save_path}")

    if mode in {"show", "both"}:
        plt.show(block=SHOW_BLOCK)

    plt.close(fig)


def _to_dense_1d(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=float).ravel()


def _normalize_domain_array(values: Sequence) -> np.ndarray:
    out = []
    for value in np.asarray(values):
        try:
            out.append(int(float(value)))
        except Exception:
            out.append(-1)
    return np.asarray(out, dtype=int)


def load_coords_npz(net_in: str, group_id: int, sample_id: int) -> np.ndarray:
    path = os.path.join(net_in, f"{group_id}_{sample_id}_validation.npz")
    data = np.load(path)
    return data["coord"].astype(np.float32)


def transform_loaded_coords(coords: np.ndarray) -> np.ndarray:
    """Transform coordinates once into their final spatial display orientation."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected coordinate array of shape (n, 2+), got {coords.shape}")

    coords = coords[:, :2].copy()

    if SWAP_XY:
        coords = coords[:, [1, 0]]

    xmin, xmax, ymin, ymax = FIXED_BOX
    if INVERT_X:
        coords[:, 0] = xmin + xmax - coords[:, 0]
    if INVERT_Y:
        coords[:, 1] = ymin + ymax - coords[:, 1]

    return coords


def get_real_h5ad_path(group_id: int, sample_id: int) -> str:
    return os.path.join(H5AD_DIR, f"{group_id}_{sample_id}.h5ad")


def get_virtual_h5ad_path(query_value: float) -> str:
    return os.path.join(VIRTUAL_DIR, f"virtual_AD_time_{query_value:.3f}.h5ad")


def _validate_record_alignment(
    coords: np.ndarray,
    gene_values: np.ndarray,
    domains: np.ndarray,
    label: str,
) -> None:
    n = coords.shape[0]
    if len(gene_values) != n or len(domains) != n:
        raise ValueError(
            f"Row mismatch for {label}: coords={n}, gene={len(gene_values)}, "
            f"domains={len(domains)}"
        )


def load_real_slice_record(
    group_id: int,
    sample_id: int,
    age: float,
    label: str,
    gene_name: str = GENE_NAME,
    domain_key: str = REAL_DOMAIN_KEY,
) -> SliceRecord:
    path = get_real_h5ad_path(group_id, sample_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    adata = sc.read_h5ad(path)
    coords = transform_loaded_coords(load_coords_npz(NET_IN, group_id, sample_id))

    if gene_name not in adata.var_names:
        raise KeyError(f"Gene {gene_name!r} not found in {path}")
    gene_values = _to_dense_1d(adata[:, [gene_name]].X)
    gene_values = np.log1p(np.clip(gene_values, a_min=0.0, a_max=None))

    if domain_key not in adata.obs:
        raise KeyError(f"{domain_key!r} not found in adata.obs for {path}")
    domains = _normalize_domain_array(adata.obs[domain_key].to_numpy())

    _validate_record_alignment(coords, gene_values, domains, label)

    return SliceRecord(
        age=float(age),
        label=str(label),
        path=path,
        is_real=True,
        coords=coords,
        gene_values=gene_values,
        domains=domains,
    )


def load_virtual_slice_record(
    query_value: float,
    age: float,
    label: str,
    gene_name: str = GENE_NAME,
    coord_key: str = VIRTUAL_COORD_KEY,
    domain_key: str = VIRTUAL_DOMAIN_KEY,
) -> SliceRecord:
    path = get_virtual_h5ad_path(query_value)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    adata = sc.read_h5ad(path)
    if "valid" in adata.obs.columns:
        adata = adata[adata.obs["valid"]].copy()

    if coord_key not in adata.obsm:
        raise KeyError(f"{coord_key!r} not found in adata.obsm for {path}")
    coords = transform_loaded_coords(np.asarray(adata.obsm[coord_key], dtype=float))

    if gene_name not in adata.var_names:
        raise KeyError(f"Gene {gene_name!r} not found in {path}")
    gene_values = _to_dense_1d(adata[:, [gene_name]].X)
    gene_values = np.log1p(np.clip(gene_values, a_min=0.0, a_max=None))

    if domain_key not in adata.obs:
        raise KeyError(f"{domain_key!r} not found in adata.obs for {path}")
    domains = _normalize_domain_array(adata.obs[domain_key].to_numpy())

    _validate_record_alignment(coords, gene_values, domains, label)

    return SliceRecord(
        age=float(age),
        label=str(label),
        path=path,
        is_real=False,
        coords=coords,
        gene_values=gene_values,
        domains=domains,
    )


def load_all_slices() -> Tuple[List[SliceRecord], List[SliceRecord]]:
    real_records = [
        load_real_slice_record(
            GROUP_ID,
            item["sample_id"],
            item["age"],
            item["label"],
        )
        for item in REAL_SLICE_SPECS
    ]

    virtual_records = [
        load_virtual_slice_record(
            item["query_value"],
            item["age"],
            item["label"],
        )
        for item in VIRTUAL_SLICE_SPECS
    ]

    return (
        sorted(real_records, key=lambda r: r.age),
        sorted(virtual_records, key=lambda r: r.age),
    )


# ============================================================
# GEOMETRY / COLORS
# ============================================================
def get_record_extent() -> Tuple[float, float, float, float]:
    return FIXED_BOX


def get_mirrored_half_extents() -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    xmin, xmax, ymin, ymax = FIXED_BOX
    width = xmax - xmin

    left_x0 = xmin + MIRROR_CENTER_FOLD
    left_x1 = xmax + MIRROR_CENTER_FOLD

    right_x0 = xmax + MIRROR_GAP - MIRROR_CENTER_FOLD
    right_x1 = right_x0 + width

    left_extent = (left_x0, left_x1, ymin, ymax)
    right_extent = (right_x0, right_x1, ymin, ymax)
    return left_extent, right_extent


def get_mirrored_extent() -> Tuple[float, float, float, float]:
    left_extent, right_extent = get_mirrored_half_extents()
    xmin = min(left_extent[0], right_extent[0])
    xmax = max(left_extent[1], right_extent[1])
    ymin = min(left_extent[2], right_extent[2])
    ymax = max(left_extent[3], right_extent[3])
    return xmin, xmax, ymin, ymax


def mirrored_half_coords(record: SliceRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct matched gene/domain coordinates for a mirrored slice."""
    xmin, xmax, ymin, ymax = FIXED_BOX
    left_extent, right_extent = get_mirrored_half_extents()
    left_x0, _, _, _ = left_extent
    right_x0, _, _, _ = right_extent

    x_all = np.asarray(record.coords[:, 0], dtype=float)
    y_all = np.asarray(record.coords[:, 1], dtype=float)

    keep_mask = (
        np.isfinite(x_all)
        & np.isfinite(y_all)
        & (x_all >= xmin)
        & (x_all <= xmax)
        & (y_all >= ymin)
        & (y_all <= ymax)
    )

    x = x_all[keep_mask]
    y = y_all[keep_mask]

    gene_xy = np.column_stack([x + MIRROR_CENTER_FOLD, y])
    domain_xy = np.column_stack([right_x0 + (xmax - x), y])

    return gene_xy, domain_xy, keep_mask


def gene_facecolors(values: np.ndarray, vmax: float, alpha_factor: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    normalized = np.clip(values / max(vmax, 1e-8), 0.0, 1.0)
    colors = plt.cm.Reds(normalized)
    colors[:, 3] = np.clip((0.08 + 0.92 * normalized) * alpha_factor, 0.03, 1.0)
    return colors


def domain_facecolors(domains: np.ndarray) -> np.ndarray:
    return np.asarray(
        [LABEL_TO_COLOR.get(int(d), BACKGROUND_GRAY) for d in domains],
        dtype=object,
    )


def shared_gene_vmax(records: Sequence[SliceRecord], percentile: float = 99.0) -> float:
    values = np.concatenate([np.asarray(r.gene_values, dtype=float) for r in records])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    return max(float(np.percentile(values, percentile)), 1e-8)


def sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    indices = np.arange(n, dtype=int)
    if n <= max_points:
        return indices

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_points, replace=False))


def age_to_time_x(age: float, youngest_age: float) -> float:
    """Map age to WORLD-X; older ages move to the right."""
    return (float(age) - float(youngest_age)) * TIME_SCALE_PER_MONTH


# ============================================================
# 3D SPATIAL DRAWING HELPERS
# ============================================================
def draw_plane_frame_3d(
    ax,
    time_x: float,
    extent: Tuple[float, float, float, float],
    is_real: bool,
) -> None:
    """
    Draw a spatial frame in a Y-Z plane at fixed temporal X.

    World coordinates:
        X = time
        Y = displayed spatial x
        Z = displayed spatial y
    """
    umin, umax, vmin, vmax = extent

    x = np.full(5, float(time_x), dtype=float)
    y = np.asarray([umin, umax, umax, umin, umin], dtype=float)
    z = np.asarray([vmin, vmin, vmax, vmax, vmin], dtype=float)

    ax.plot(
        x,
        y,
        z,
        color=REAL_FRAME_COLOR if is_real else VIRTUAL_FRAME_COLOR,
        linewidth=SOLID_LW if is_real else DASHED_LW,
        linestyle="-" if is_real else DASHED_STYLE,
        alpha=0.96,
        zorder=1,
    )


def draw_mirror_seam_3d(ax, time_x: float, is_real: bool) -> None:
    left_extent, right_extent = get_mirrored_half_extents()
    seam_u = 0.5 * (left_extent[1] + right_extent[0])
    vmin = left_extent[2]
    vmax = left_extent[3]

    ax.plot(
        [time_x, time_x],
        [seam_u, seam_u],
        [vmin, vmax],
        color=REAL_FRAME_COLOR if is_real else VIRTUAL_FRAME_COLOR,
        linewidth=0.8,
        linestyle="-" if is_real else DASHED_STYLE,
        alpha=0.90,
        zorder=1,
    )


def style_3d_axis(
    ax,
    extent: Tuple[float, float, float, float],
    time_positions: Sequence[float],
) -> None:
    """Style only the 3D spatial scene. The timeline is drawn separately in 2D."""
    umin, umax, vmin, vmax = extent
    time_positions = np.asarray(time_positions, dtype=float)

    t_span = max(float(np.ptp(time_positions)), 1.0)
    u_span = max(float(umax - umin), 1.0)
    v_span = max(float(vmax - vmin), 1.0)

    t_pad = max(0.06 * t_span, 35.0)
    u_pad = 60.0
    v_pad = 70.0

    t0 = float(np.min(time_positions) - t_pad)
    t1 = float(np.max(time_positions) + t_pad)
    u0, u1 = umin - u_pad, umax + u_pad
    v0, v1 = vmin - v_pad, vmax + v_pad

    ax.set_xlim(t0, t1)
    ax.set_ylim(u0, u1)
    ax.set_zlim(v0, v1)

    # Temporal X stays proportional to age spacing; spatial Y/Z are scaled
    # uniformly to control how large the hanging maps appear.
    display_span = np.asarray(
        [
            (t1 - t0),
            (u1 - u0) * SPATIAL_DISPLAY_SCALE,
            (v1 - v0) * SPATIAL_DISPLAY_SCALE,
        ],
        dtype=float,
    )
    display_span /= max(float(np.max(display_span)), 1e-8)

    try:
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM, roll=VIEW_ROLL)
    except TypeError:
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    ax.set_proj_type("ortho")
    ax.set_box_aspect(tuple(display_span))
    ax.set_axis_off()


# ============================================================
# SCREEN-SPACE HORIZONTAL TIMELINE
# ============================================================
def _project_world_to_figure(fig, ax, xyz: Sequence[float]) -> Tuple[float, float]:
    """Project one 3D world coordinate to figure-fraction coordinates."""
    x3, y3, z3 = [float(v) for v in xyz]
    x2, y2, _ = proj3d.proj_transform(x3, y3, z3, ax.get_proj())
    x_disp, y_disp = ax.transData.transform((x2, y2))
    x_fig, y_fig = fig.transFigure.inverted().transform((x_disp, y_disp))
    return float(x_fig), float(y_fig)


def _top_center_anchor(
    time_x: float,
    extent: Tuple[float, float, float, float],
) -> Tuple[float, float, float]:
    """Top-center attachment location on a spatial Y-Z frame."""
    umin, umax, _, vmax = extent
    return float(time_x), 0.5 * (umin + umax), float(vmax)


def draw_horizontal_time_scaffold(
    fig,
    ax,
    records: Sequence[SliceRecord],
    time_positions: Sequence[float],
    extent: Tuple[float, float, float, float],
    title_label: str = "Age",
) -> None:
    """
    Draw the timeline entirely in 2D figure coordinates.

    Consequences
    ------------
    1. The timeline is exactly horizontal on screen under every 3D camera angle.
    2. Each timeline marker uses the screen-x coordinate of that slice's
       projected top-center frame anchor.
    3. A vertical screen-space hanger connects the marker to the tissue plane.
       Hence the spatial maps visually hang from the horizontal timeline.
    """
    if len(records) != len(time_positions):
        raise ValueError("records and time_positions must have identical length")

    # Projection depends on the final axes geometry; force a draw first.
    fig.canvas.draw()

    anchor_fig = np.asarray(
        [
            _project_world_to_figure(
                fig,
                ax,
                _top_center_anchor(t, extent),
            )
            for t in time_positions
        ],
        dtype=float,
    )

    marker_x = anchor_fig[:, 0]
    anchor_y = anchor_fig[:, 1]

    # The marker x positions are kept exactly at the projected tissue-anchor
    # positions so every suspension line remains truly vertical in screen space.
    # For left-to-right chronological ordering, use the default negative azimuth
    # (or another negative azimuth that keeps +world-X projecting to the right).
    if len(marker_x) >= 2 and marker_x[-1] < marker_x[0]:
        print(
            "[WARN] Current camera projects older ages to the left. "
            "Use a negative --view-azim (for example -60) for the intended "
            "left-to-right chronological layout."
        )

    timeline_y = max(
        TIMELINE_MIN_Y_FIG,
        float(np.max(anchor_y) + TIMELINE_CLEARANCE_FIG),
    )
    timeline_y = min(timeline_y, TIMELINE_MAX_Y_FIG)

    line_start = max(0.02, float(np.min(marker_x) - TIMELINE_X_PAD_FIG))
    line_end = min(0.94, float(np.max(marker_x) + TIMELINE_X_PAD_FIG + 0.010))

    # Perfectly horizontal arrow in figure coordinates.
    arrow = FancyArrowPatch(
        (line_start, timeline_y),
        (line_end, timeline_y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=TIMELINE_ARROW_SCALE,
        linewidth=TIMELINE_LW,
        color=TIMELINE_COLOR,
        shrinkA=0,
        shrinkB=0,
        clip_on=False,
        zorder=200,
    )
    fig.add_artist(arrow)

    # Age label at the arrow end.
    fig.text(
        min(line_end + 0.008, 0.965),
        timeline_y + 0.001,
        title_label,
        fontsize=10,
        # fontweight="bold",
        color=TIMELINE_COLOR,
        ha="left",
        va="center",
        zorder=205,
    )

    # use_two_tiers = len(records) >= 5
    use_two_tiers = False

    for i, (record, xmark, y_anchor) in enumerate(zip(records, marker_x, anchor_y)):
        color = REAL_LABEL_COLOR if record.is_real else VIRTUAL_LABEL_COLOR

        # Vertical screen-space suspension line; leaves a tiny visual gap at
        # both the timeline and tissue attachment point.
        y_top = timeline_y - HANGER_TOP_GAP_FIG
        y_bottom = y_anchor + HANGER_BOTTOM_GAP_FIG
        if y_bottom < y_top:
            hanger = Line2D(
                [xmark, xmark],
                [y_top, y_bottom],
                transform=fig.transFigure,
                color=HANGER_COLOR,
                linewidth=HANGER_LW,
                alpha=HANGER_ALPHA,
                solid_capstyle="round",
                clip_on=False,
                zorder=180,
            )
            fig.add_artist(hanger)

        # Timeline intersection marker.
        marker = Line2D(
            [xmark],
            [timeline_y],
            transform=fig.transFigure,
            marker="o",
            linestyle="",
            markersize=TIMELINE_MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=TIMELINE_MARKER_EDGE_COLOR,
            markeredgewidth=TIMELINE_MARKER_EDGE_WIDTH,
            clip_on=False,
            zorder=210,
        )
        fig.add_artist(marker)

        # Small attachment dot on the spatial frame.
        anchor_dot = Line2D(
            [xmark],
            [y_anchor],
            transform=fig.transFigure,
            marker="o",
            linestyle="",
            markersize=HANGER_ANCHOR_SIZE,
            markerfacecolor=HANGER_COLOR,
            markeredgecolor="none",
            alpha=0.65,
            clip_on=False,
            zorder=185,
        )
        fig.add_artist(anchor_dot)

        # Age labels remain horizontal. Dense seven-slice figures alternate
        # between two shallow tiers to reduce collisions.
        if use_two_tiers:
            label_dy = (
                TIMELINE_LABEL_OFFSET_FIG
                if i % 2 == 0
                else TIMELINE_LABEL_SECOND_TIER_FIG
            )
        else:
            label_dy = TIMELINE_LABEL_OFFSET_FIG

        fig.text(
            xmark,
            timeline_y + label_dy,
            record.label,
            fontsize=10,
            # fontweight="bold",
            color=color,
            ha="center",
            va="bottom",
            zorder=215,
        )


# ============================================================
# LAYER PLOTTING
# ============================================================
def plot_gene_layer_3d(
    ax,
    record: SliceRecord,
    time_x: float,
    vmax: float,
    seed: int,
) -> None:
    indices = sample_indices(len(record.coords), MAX_POINTS_PER_SLICE, seed)

    coords = np.asarray(record.coords, dtype=float)[indices]
    values = np.asarray(record.gene_values, dtype=float)[indices]

    alpha_factor = REAL_ALPHA if record.is_real else VIRTUAL_ALPHA
    colors = gene_facecolors(values, vmax=vmax, alpha_factor=alpha_factor)

    ax.scatter(
        np.full(len(coords), float(time_x)),
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=POINT_SIZE_GENE,
        linewidths=0,
        depthshade=False,
        rasterized=True,
        zorder=2,
    )

    draw_plane_frame_3d(
        ax,
        time_x,
        get_record_extent(),
        is_real=record.is_real,
    )


def plot_mirrored_layer_3d(
    ax,
    record: SliceRecord,
    time_x: float,
    vmax: float,
    seed: int,
    *,
    virtual_marker_edges: bool = False,
) -> None:
    gene_xy, domain_xy, keep_mask = mirrored_half_coords(record)

    retained_values = np.asarray(record.gene_values, dtype=float)[keep_mask]
    retained_domains = np.asarray(record.domains)[keep_mask]

    if not (
        len(gene_xy)
        == len(domain_xy)
        == len(retained_values)
        == len(retained_domains)
    ):
        raise RuntimeError(
            f"Mirrored row alignment failed for {record.label}: "
            f"gene_xy={len(gene_xy)}, domain_xy={len(domain_xy)}, "
            f"gene_values={len(retained_values)}, domains={len(retained_domains)}"
        )

    indices = sample_indices(len(retained_values), MAX_POINTS_PER_SLICE, seed)
    gene_xy = gene_xy[indices]
    domain_xy = domain_xy[indices]
    values = retained_values[indices]
    domains = retained_domains[indices]

    alpha_factor = REAL_ALPHA if record.is_real else VIRTUAL_ALPHA

    use_virtual_edges = bool(virtual_marker_edges and not record.is_real)
    marker_edgecolor = (
        FIG3_VIRTUAL_MARKER_EDGE_COLOR
        if use_virtual_edges
        else "none"
    )
    marker_edgewidth = (
        FIG3_VIRTUAL_MARKER_EDGE_WIDTH
        if use_virtual_edges
        else 0.0
    )

    gene_colors = gene_facecolors(values, vmax=vmax, alpha_factor=alpha_factor)
    ax.scatter(
        np.full(len(gene_xy), float(time_x)),
        gene_xy[:, 0],
        gene_xy[:, 1],
        c=gene_colors,
        s=POINT_SIZE_GENE,
        edgecolors=marker_edgecolor,
        linewidths=marker_edgewidth,
        depthshade=False,
        rasterized=True,
        zorder=2,
    )

    domain_colors = domain_facecolors(domains)
    ax.scatter(
        np.full(len(domain_xy), float(time_x)),
        domain_xy[:, 0],
        domain_xy[:, 1],
        c=domain_colors,
        s=POINT_SIZE_DOMAIN,
        alpha=DOMAIN_ALPHA if record.is_real else DOMAIN_ALPHA * 0.95,
        edgecolors=marker_edgecolor,
        linewidths=marker_edgewidth,
        depthshade=False,
        rasterized=True,
        zorder=2,
    )

    draw_plane_frame_3d(
        ax,
        time_x,
        get_mirrored_extent(),
        is_real=record.is_real,
    )
    draw_mirror_seam_3d(ax, time_x, is_real=record.is_real)




def draw_grid_raster_lines_3d(
    ax,
    time_x: float,
    extent: Tuple[float, float, float, float],
    grid_size: int = VIRTUAL_SUPPORT_GRID_SIZE,
    color: str = VIRTUAL_SUPPORT_GRID_COLOR,
    alpha: float = VIRTUAL_SUPPORT_GRID_ALPHA,
    linewidth: float = VIRTUAL_SUPPORT_GRID_LW,
) -> None:
    """
    Draw a full-space pixel raster using grid lines.

    This is a visualization-only representation of the 256 x 256 spatial
    support layout used for virtual generation. The whole plane is shown as a
    regular raster, without differentiating tissue-valid versus background
    pixels.
    """
    umin, umax, vmin, vmax = extent
    du = (umax - umin) / float(grid_size)
    dv = (vmax - vmin) / float(grid_size)

    # Vertical grid lines in the Y-Z plane.
    for i in range(grid_size + 1):
        u = umin + i * du
        ax.plot(
            [time_x, time_x],
            [u, u],
            [vmin, vmax],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )

    # Horizontal grid lines in the Y-Z plane.
    for j in range(grid_size + 1):
        v = vmin + j * dv
        ax.plot(
            [time_x, time_x],
            [umin, umax],
            [v, v],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )



def plot_virtual_support_layer_3d(
    ax,
    record: SliceRecord,
    time_x: float,
) -> None:
    """
    Plot only the virtual spatial support as full-space 256 x 256 raster grids.

    Important display rule:
    each mirrored half corresponds to one original 1024 x 1024 spatial map, so
    the raster should be shown as TWO square 256 x 256 grids rather than one
    wide rectangular frame spanning both halves.
    """
    left_extent, right_extent = get_mirrored_half_extents()

    # Draw a complete square raster for each original 1024x1024 map.
    # draw_grid_raster_lines_3d(ax, time_x, left_extent)
    # draw_grid_raster_lines_3d(ax, time_x, right_extent)

    # Draw two dashed square bounding boxes, one for each mirrored half, so the
    # viewer reads them as two square support lattices rather than a single
    # rectangular mirrored canvas.
    draw_plane_frame_3d(
        ax,
        time_x,
        left_extent,
        is_real=False,
    )
    draw_plane_frame_3d(
        ax,
        time_x,
        right_extent,
        is_real=False,
    )





# ============================================================
# FIGURE 1
# ============================================================
def plot_figure1_real_gene_stack(
    real_records: Sequence[SliceRecord],
    outdir: str,
) -> None:
    set_style()
    fig = plt.figure(figsize=(11.6, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    ordered = sorted(real_records, key=lambda r: r.age)
    vmax = shared_gene_vmax(ordered)
    youngest_age = min(r.age for r in ordered)
    time_positions = [age_to_time_x(r.age, youngest_age) for r in ordered]

    for i, (record, time_x) in enumerate(zip(ordered, time_positions)):
        plot_gene_layer_3d(
            ax,
            record,
            time_x,
            vmax=vmax,
            seed=100 + i,
        )

    style_3d_axis(
        ax,
        get_record_extent(),
        time_positions,
    )

    scalar_mappable = ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=vmax),
        cmap="Reds",
    )
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        fraction=0.03,
        pad=0.02,
        shrink=0.70,
    )
    colorbar.set_label(f"log1p {GENE_NAME} expression", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)

    ax.set_title(
        "Observed spatial omics through time",
        pad=18,
        fontsize=15,
        # fontweight="bold",
    )

    # Must be last: projection uses final camera and axes geometry.
    draw_horizontal_time_scaffold(
        fig,
        ax,
        ordered,
        time_positions,
        get_record_extent(),
        title_label="Age",
    )

    finalize_figure(
        fig,
        os.path.join(outdir, "figure1_real_gene_3d_left_to_right_hanging.png"),
    )


# ============================================================
# FIGURE 2
# ============================================================
def plot_figure2_real_mirrored_stack(
    real_records: Sequence[SliceRecord],
    virtual_records: Sequence[SliceRecord],
    outdir: str,
) -> None:
    """
    Plot measured mirrored states together with virtual-query spatial support.

    Real ages (2.5, 5.7, 17.9m)
        Full gene-expression + spatial-domain mirrored maps with solid frames.

    Query ages (4.1, 8.25, 11.8, 14.85m)
        Binary 256 x 256 spatial-support raster only, shown as gray square
        pixels in the same mirrored layout with dashed bounding boxes.
    """
    set_style()
    fig = plt.figure(figsize=(14.2, 7.8))
    ax = fig.add_subplot(111, projection="3d")

    real_ordered = sorted(real_records, key=lambda r: r.age)
    virtual_ordered = sorted(virtual_records, key=lambda r: r.age)
    ordered = sorted(real_ordered + virtual_ordered, key=lambda r: r.age)

    # Figure 2's expression scale is defined only from measured tissue states.
    vmax = shared_gene_vmax(real_ordered)
    youngest_age = min(r.age for r in ordered)
    time_positions = [age_to_time_x(r.age, youngest_age) for r in ordered]
    real_positions = [age_to_time_x(r.age, youngest_age) for r in real_ordered]

    for i, (record, time_x) in enumerate(zip(ordered, time_positions)):
        if record.is_real:
            plot_mirrored_layer_3d(
                ax,
                record,
                time_x,
                vmax=vmax,
                seed=200 + i,
            )
        else:
            plot_virtual_support_layer_3d(
                ax,
                record,
                time_x,
            )

    style_3d_axis(
        ax,
        get_mirrored_extent(),
        time_positions,
    )

    draw_horizontal_time_scaffold(
        fig,
        ax,
        real_ordered,
        real_positions,
        get_mirrored_extent(),
        title_label=" Age\n(months)",
    )

    finalize_figure(
        fig,
        os.path.join(
            outdir,
            "figure2_real_plus_virtual_support_3d_left_to_right_hanging.png",
        ),
    )


# ============================================================
# FIGURE 3
# ============================================================
def plot_figure3_all_mirrored_stack(
    records: Sequence[SliceRecord],
    outdir: str,
) -> None:
    set_style()
    fig = plt.figure(figsize=(14.2, 7.8))
    ax = fig.add_subplot(111, projection="3d")

    ordered = sorted(records, key=lambda r: r.age)
    vmax = shared_gene_vmax(ordered)
    youngest_age = min(r.age for r in ordered)
    time_positions = [age_to_time_x(r.age, youngest_age) for r in ordered]

    for i, (record, time_x) in enumerate(zip(ordered, time_positions)):
        plot_mirrored_layer_3d(
            ax,
            record,
            time_x,
            vmax=vmax,
            seed=300 + i,
            virtual_marker_edges=True,
        )

    style_3d_axis(
        ax,
        get_mirrored_extent(),
        time_positions,
    )

    draw_horizontal_time_scaffold(
        fig,
        ax,
        ordered,
        time_positions,
        get_mirrored_extent(),
        title_label=" Age\n(months)",
    )

    finalize_figure(
        fig,
        os.path.join(outdir, "figure3_all_mirrored_3d_left_to_right_hanging.png"),
    )


# ============================================================
# RUN
# ============================================================
def run_all() -> None:
    ensure_dir(OUTDIR)

    real_records, virtual_records = load_all_slices()
    all_records = sorted(real_records + virtual_records, key=lambda r: r.age)

    # plot_figure1_real_gene_stack(real_records, OUTDIR)
    plot_figure2_real_mirrored_stack(real_records, virtual_records, OUTDIR)
    plot_figure3_all_mirrored_stack(all_records, OUTDIR)

    print(f"[DONE] Figures written/displayed from: {OUTDIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate left-to-right 3D mouse time-pipeline figures with a "
            "screen-horizontal timeline and hanging spatial maps."
        )
    )
    parser.add_argument(
        "--figure-mode",
        choices=["save", "show", "both"],
        default=FIGURE_MODE,
        help="Figure handling mode.",
    )
    parser.add_argument(
        "--gene",
        default=GENE_NAME,
        help="Marker gene to visualize.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=TIME_SCALE_PER_MONTH,
        help=(
            "World-X distance per month. Larger values increase left-to-right "
            "age separation between tissue planes."
        ),
    )
    # Backward-compatible alias for scripts/commands using the old option.
    parser.add_argument(
        "--z-scale",
        type=float,
        dest="time_scale_alias",
        default=None,
        help="Backward-compatible alias for --time-scale.",
    )
    parser.add_argument(
        "--spatial-display-scale",
        type=float,
        default=SPATIAL_DISPLAY_SCALE,
        help=(
            "Uniform display scaling of the spatial Y/Z planes relative to "
            "temporal X. Smaller values make the hanging maps more compact."
        ),
    )
    parser.add_argument(
        "--virtual-support-grid-lw",
        type=float,
        default=VIRTUAL_SUPPORT_GRID_LW,
        help=(
            "Line width used for the 256x256 virtual spatial-support grid shown in Figure 2."
        ),
    )
    parser.add_argument(
        "--mirror-center-fold",
        type=float,
        default=MIRROR_CENTER_FOLD,
        help="Shift mirrored gene/domain halves inward to reduce center margin.",
    )
    parser.add_argument(
        "--view-elev",
        type=float,
        default=VIEW_ELEV,
        help="3D elevation for spatial-map viewing. Timeline remains horizontal.",
    )
    parser.add_argument(
        "--view-azim",
        type=float,
        default=VIEW_AZIM,
        help="3D azimuth for spatial-map viewing. Timeline remains horizontal.",
    )
    parser.add_argument(
        "--view-roll",
        type=float,
        default=VIEW_ROLL,
        help="Optional 3D spatial-map roll. Timeline remains horizontal.",
    )
    parser.add_argument(
        "--timeline-clearance",
        type=float,
        default=TIMELINE_CLEARANCE_FIG,
        help="Figure-fraction vertical clearance between map anchors and timeline.",
    )
    args = parser.parse_args()

    globals()["FIGURE_MODE"] = args.figure_mode
    globals()["GENE_NAME"] = args.gene
    globals()["TIME_SCALE_PER_MONTH"] = float(
        args.time_scale_alias if args.time_scale_alias is not None else args.time_scale
    )
    globals()["SPATIAL_DISPLAY_SCALE"] = float(args.spatial_display_scale)
    globals()["VIRTUAL_SUPPORT_GRID_LW"] = float(args.virtual_support_grid_lw)
    globals()["MIRROR_CENTER_FOLD"] = float(args.mirror_center_fold)
    globals()["VIEW_ELEV"] = float(args.view_elev)
    globals()["VIEW_AZIM"] = float(args.view_azim)
    globals()["VIEW_ROLL"] = float(args.view_roll)
    globals()["TIMELINE_CLEARANCE_FIG"] = float(args.timeline_clearance)

    print(f"[INFO] FIGURE_MODE = {FIGURE_MODE}")
    print(f"[INFO] GENE_NAME = {GENE_NAME}")
    print(f"[INFO] TIME_SCALE_PER_MONTH = {TIME_SCALE_PER_MONTH}")
    print(f"[INFO] SPATIAL_DISPLAY_SCALE = {SPATIAL_DISPLAY_SCALE}")
    print(f"[INFO] VIRTUAL_SUPPORT_GRID_LW = {VIRTUAL_SUPPORT_GRID_LW}")
    print(f"[INFO] MIRROR_CENTER_FOLD = {MIRROR_CENTER_FOLD}")
    print(f"[INFO] VIEW_ELEV = {VIEW_ELEV}")
    print(f"[INFO] VIEW_AZIM = {VIEW_AZIM}")
    print(f"[INFO] VIEW_ROLL = {VIEW_ROLL}")
    print(f"[INFO] TIMELINE_CLEARANCE_FIG = {TIMELINE_CLEARANCE_FIG}")

    run_all()


if __name__ == "__main__":
    main()

