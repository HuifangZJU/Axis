#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mouse-brain virtual comparison: AXIS versus SpatialZ.

This version produces ONE combined 2x2 comparison figure:

    row 1: AXIS      | 4.1 months | 11.8 months
    row 2: SpatialZ  | 4.1 months | 11.8 months

Each subplot retains the mirrored layout:
    left  = Gfap gene-expression map
    right = horizontally mirrored spatial-domain map

Key visualization changes
-------------------------
1. AXIS is rendered as a pixel/raster image because its generated virtual data
   lie on a regular spatial grid.
2. SpatialZ retains the point/scatter rendering used previously.
3. All four panels share one Gfap expression scale.
4. Per-panel domain legends are removed.
5. A single anatomical domain legend is placed below the 2x2 figure and includes
   Cerebral cortex, Hippocampus, Thalamus, Hypothalamus, and Other.
6. Missing/non-rendered raster locations remain white.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# ============================================================
# CONFIG
# ============================================================

ROOT = "/media/huifang/data/registration/mouse_brain"

# -----------------------------------------------------------------------------
# AXIS virtual data
# -----------------------------------------------------------------------------
AXIS_VIRTUAL_DIR = os.path.join(
    ROOT,
    "virtual_tissue",
    "AD_time_grid256_spatial2.0",
    "query",
)
AXIS_VIRTUAL_PATH_FORMAT = "virtual_AD_time_{query_value:.3f}.h5ad"

# -----------------------------------------------------------------------------
# SpatialZ virtual data
# -----------------------------------------------------------------------------
SPATIALZ_VIRTUAL_DIR = os.path.join(
    ROOT,
    "virtual_tissue",
    "spatialz",
)
SPATIALZ_VIRTUAL_PATH_FORMAT = "{query_value:.1f}.h5ad"

METHOD_CONFIGS = {
    "Axis": {
        "virtual_dir": AXIS_VIRTUAL_DIR,
        "path_format": AXIS_VIRTUAL_PATH_FORMAT,
    },
    "SpatialZ": {
        "virtual_dir": SPATIALZ_VIRTUAL_DIR,
        "path_format": SPATIALZ_VIRTUAL_PATH_FORMAT,
    },
}

VIRTUAL_SLICES = {
    4.1: {"query_value": 4.100, "label": "4.1m"},
    11.8: {"query_value": 11.800, "label": "11.8m"},
}

GENE_NAME = "Gfap"
VIRTUAL_DOMAIN_KEY = "axis_joint_domain"
VIRTUAL_COORD_KEY = "spatial"

OUTDIR = "/media/huifang/data/registration/paper_analysis/Figure56_mouse/5c"

# Domain palette retained from the original Figure-5 workflow.
ALL_LABELS = list(range(13))
BASE_CMAP = plt.get_cmap("tab20")
LABEL_TO_COLOR = {lab: BASE_CMAP(i) for i, lab in enumerate(ALL_LABELS)}

# 11.8-month domains retained from the original code.
HIGHLIGHT_118 = [0, 1, 3, 4, 6, 8, 9, 10, 11]

# Anatomical grouping used for the 11.8-month legend.
ANATOMICAL_GROUPS_118 = [
    {"name": "Cerebral cortex", "labels": [1, 4, 6, 9, 10]},
    {"name": "Hippocampus", "labels": [8, 11]},
    {"name": "Thalamus", "labels": [0]},
    {"name": "Hypothalamus", "labels": [3]},
]

# Figure handling.
FIG_DPI = 400
FIGURE_MODE = "save"  # "save", "show", or "both"
SHOW_BLOCK = True

# Spatial rendering.
GENE_POINT_SIZE = 2.0
DOMAIN_POINT_SIZE = 2.0
VIRTUAL_ALPHA = 0.95
GENE_BG_ALPHA = 0.12

# Gene-expression colorbar.
GENE_COLOR_PERCENTILE = 99.0

# [left, bottom, width, height] in figure coordinates
GENE_COLORBAR_RECT = [
    0.7,   # horizontal location
    0.36,    # vertical location
    0.018,   # width
    0.48,    # length
]

GENE_COLORBAR_LABEL_FONTSIZE = 14
GENE_COLORBAR_TICK_FONTSIZE = 12


BACKGROUND_GRAY = "#CFCFCF"
FRAME_COLOR = "#A9A9A9"
TEXT_COLOR = "#222222"
DASHED_STYLE = (0, (4.0, 2.4))
MIRROR_FRAME_WIDTH = 1.6

# Fixed plotting frame and coordinate orientation retained from the original.
FIXED_BOX = (0.0, 1024.0, 0.0, 1024.0)
SWAP_XY = True
INVERT_X = True
INVERT_Y = False
MIRROR_RIGHT_TRIM = 70.0
DRAW_MIRROR_DIVIDER = True

# Standalone figure geometry. Extra bottom margin is reserved for legends.
# Combined 2x2 figure geometry.
COMBINED_FIGSIZE = (13.6, 10.2)
TITLE_FONTSIZE = 18
ROW_LABEL_FONTSIZE = 18
COLUMN_LABEL_FONTSIZE = 17
LEGEND_FONTSIZE = 12
LEGEND_MARKER_SIZE = 8

# 2x2 panel spacing and outer margins.
SUBPLOT_WSPACE = 0.01
SUBPLOT_HSPACE = -0.25
FIG_LEFT = 0.075
FIG_RIGHT = 0.925
FIG_TOP = 0.91
FIG_BOTTOM = 0.20

# Shared gene-expression colorbar for all four panels.
SHARED_GENE_COLORBAR_RECT = [0.94, 0.38, 0.014, 0.30]

# Shared bottom annotations for the left/right mirrored halves.
# Recommendation:
#   Keep these labels at the bottom, because the column titles already occupy the
#   top margin and the row labels sit on the left. Bottom annotations make the
#   left-right mapping easier to scan without crowding the top.
BOTTOM_HALF_LABEL_Y = 0.24
BOTTOM_GENE_LABEL_X = 0.5
BOTTOM_DOMAIN_LABEL_X = 0.68
BOTTOM_HALF_LABEL_FONTSIZE = 13

# Shared anatomical legend placement and styling.
LEGEND_RECT = [0.09, 0.12, 0.88, 0.09]
LEGEND_TITLE_Y = 0.90
LEGEND_ROW_Y = [0.7, 0.40, 0.12]
LEGEND_DOT_TEXT_DX = 0.012
LEGEND_X_POSITIONS = {
    "Cerebral cortex": 0.01,
    "Hippocampus": 0.34,
    "Thalamus": 0.5,
    "Hypothalamus": 0.62,
    "Other": 0.76,
}
LEGEND_COLUMN_DX = {
    "Cerebral cortex": 0.1,
    "Hippocampus": 0.1,
    "Thalamus": 0.1,
    "Hypothalamus": 0.1,
    "Other": 0.10,
}

# AXIS raster controls.
AXIS_PIXEL_RENDERING = True
AXIS_PIXEL_COORD_DECIMALS = 6

# Panel-specific rendering controls.
# 4.1-month case: show all spatial domains individually.
SHOW_FULL_DOMAINS_4P1 = True
# 11.8-month case: show gene expression only on known anatomical regions.
SHOW_GENE_ONLY_ON_KNOWN_11P8 = True
# Use full domain coloring (rather than grouped-gray "Other") in the domain panel.
SHOW_FULL_DOMAIN_COLORS = True

# Color for the grouped "Other" category in the legend, if ever needed.
OTHER_DOMAIN_COLOR = "#D0D0D0"


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class SliceRecord:
    method: str
    age: float
    label: str
    path: str
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
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finalize_figure(fig, save_path: str) -> None:
    mode = str(FIGURE_MODE).lower()
    if mode not in {"save", "show", "both"}:
        raise ValueError("FIGURE_MODE must be 'save', 'show', or 'both'.")

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


# ============================================================
# METHOD-SPECIFIC VIRTUAL PATHS
# ============================================================

def get_virtual_h5ad_path(method: str, query_value: float) -> str:
    if method not in METHOD_CONFIGS:
        raise KeyError(
            f"Unknown method {method!r}. Available methods: {list(METHOD_CONFIGS)}"
        )

    cfg = METHOD_CONFIGS[method]
    filename = str(cfg["path_format"]).format(query_value=float(query_value))
    return os.path.join(str(cfg["virtual_dir"]), filename)


# ============================================================
# DATA LOADING
# ============================================================

def transform_loaded_coords(coords: np.ndarray) -> np.ndarray:
    """Apply the same display transformation as the original Figure-5 code."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected coordinates with shape (n, 2+), got {coords.shape}")

    coords = coords[:, :2].copy()

    if SWAP_XY:
        coords = coords[:, [1, 0]]

    xmin, xmax, ymin, ymax = FIXED_BOX

    if INVERT_X:
        coords[:, 0] = xmin + xmax - coords[:, 0]
    if INVERT_Y:
        coords[:, 1] = ymin + ymax - coords[:, 1]

    return coords


def load_virtual_slice_record(
    method: str,
    age: float,
    query_value: float,
    gene_name: str = GENE_NAME,
    coord_key: str = VIRTUAL_COORD_KEY,
    domain_key: str = VIRTUAL_DOMAIN_KEY,
) -> SliceRecord:
    path = get_virtual_h5ad_path(method, query_value)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{method} virtual slice not found for {age:g} month: {path}"
        )

    adata = sc.read_h5ad(path)

    if "valid" in adata.obs.columns:
        valid = adata.obs["valid"]
        if valid.dtype == bool:
            mask = valid.to_numpy(dtype=bool)
        elif np.issubdtype(valid.dtype, np.number):
            mask = valid.to_numpy(dtype=float) != 0
        else:
            mask = (
                valid.astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes", "y", "t"])
                .to_numpy()
            )
        adata = adata[mask].copy()

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

    if not (len(coords) == len(gene_values) == len(domains)):
        raise ValueError(
            f"Length mismatch in {path}: coords={len(coords)}, "
            f"gene={len(gene_values)}, domains={len(domains)}"
        )

    return SliceRecord(
        method=str(method),
        age=float(age),
        label=f"{age:g}m",
        path=path,
        coords=coords,
        gene_values=gene_values,
        domains=domains,
    )
def add_panel_annotations(ax, gene_name: str) -> None:
    """Gene name at upper-left of the gene map; 'Spatial domain' at upper-right of the domain map."""
    ax.text(
        0.03, 0.97,
        gene_name +" | Spatial domain",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=14,
        fontstyle="italic",
        color=TEXT_COLOR,
        zorder=30,
    )
    # ax.text(
    #     0.97, 0.97,
    #     "Spatial domain",
    #     transform=ax.transAxes,
    #     ha="right", va="top",
    #     fontsize=14,
    #     color=TEXT_COLOR,
    #     zorder=30,
    # )

# ============================================================
# COLOR / GEOMETRY HELPERS
# ============================================================

def get_gene_color_vmax(values: np.ndarray) -> float:
    """
    Upper color limit for gene expression.

    The gene values have already been log1p-transformed.
    The 99th percentile is retained from the original plotting code.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return 1.0

    vmax = np.percentile(
        finite,
        GENE_COLOR_PERCENTILE,
    )

    return max(
        float(vmax),
        1e-8,
    )


def gene_facecolors(
    values: np.ndarray,
    vmax: Optional[float] = None,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.size == 0:
        return np.zeros(
            (0, 4),
            dtype=float,
        )

    if vmax is None:
        vmax = get_gene_color_vmax(
            values
        )

    vnorm = np.clip(
        values / vmax,
        0.0,
        1.0,
    )

    colors = plt.cm.Reds(
        vnorm
    )

    colors[:, 3] = np.clip(
        0.08 + 0.92 * vnorm,
        0.08,
        1.0,
    )

    return colors

def add_gene_expression_colorbar(
    fig,
    vmax: float,
    gene_name: str,
    rect: Sequence[float] = SHARED_GENE_COLORBAR_RECT,
) -> None:
    """Add one shared quantitative gene-expression colorbar."""
    cax = fig.add_axes(rect)

    mappable = ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=vmax, clip=True),
        cmap=plt.cm.Reds,
    )
    mappable.set_array([])

    cbar = fig.colorbar(
        mappable,
        cax=cax,
        orientation="vertical",
    )
    cbar.set_label(
        "log1p expression",
        rotation=90,
        labelpad=8,
        fontsize=GENE_COLORBAR_LABEL_FONTSIZE,
    )
    cbar.ax.tick_params(
        labelsize=GENE_COLORBAR_TICK_FONTSIZE,
        length=2.2,
    )
    cbar.outline.set_visible(False)



def anatomical_domain_labels() -> set:
    """Return the union of domain IDs represented by named anatomical groups."""
    labels = set()
    for group in ANATOMICAL_GROUPS_118:
        labels.update(int(x) for x in group["labels"])
    return labels


NAMED_ANATOMICAL_LABELS = anatomical_domain_labels()
OTHER_DOMAIN_LABELS = sorted(set(ALL_LABELS) - NAMED_ANATOMICAL_LABELS)


def domain_facecolors(
    domains: np.ndarray,
    selected_domains: Optional[Sequence[int]] = None,
    group_other: bool = False,
) -> np.ndarray:
    """
    Convert domain IDs to colors.

    Parameters
    ----------
    domains
        Domain array.
    selected_domains
        Optional subset of domains to retain. If provided, any domain not in the
        subset is rendered as transparent white.
    group_other
        If True, domains outside the named anatomical groups are collapsed to a
        shared gray color. If False, every domain keeps its individual color.
    """
    domains = _normalize_domain_array(domains)

    if selected_domains is None:
        selected = set(int(x) for x in ALL_LABELS)
    else:
        selected = set(int(x) for x in selected_domains)

    colors = []
    for domain in domains:
        domain = int(domain)
        if domain not in selected:
            colors.append((1, 1, 1, 0))
        elif group_other and (domain not in NAMED_ANATOMICAL_LABELS):
            colors.append(OTHER_DOMAIN_COLOR)
        else:
            colors.append(LABEL_TO_COLOR.get(domain, OTHER_DOMAIN_COLOR))
    return np.asarray(colors, dtype=object)


def get_mirrored_parameters() -> Tuple[float, float, float, float, float]:
    xmin, xmax, ymin, ymax = FIXED_BOX
    usable_xmax = xmax - float(MIRROR_RIGHT_TRIM)
    if usable_xmax <= xmin:
        raise ValueError("MIRROR_RIGHT_TRIM is too large for FIXED_BOX.")
    seam_x = usable_xmax
    return xmin, usable_xmax, ymin, ymax, seam_x


def get_mirrored_extent() -> Tuple[float, float, float, float]:
    xmin, usable_xmax, ymin, ymax, seam_x = get_mirrored_parameters()
    width = usable_xmax - xmin
    return xmin, seam_x + width, ymin, ymax


def mirrored_half_coords(record: SliceRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, usable_xmax, ymin, ymax, seam_x = get_mirrored_parameters()

    x_all = np.asarray(record.coords[:, 0], dtype=float)
    y_all = np.asarray(record.coords[:, 1], dtype=float)

    keep = (
        np.isfinite(x_all)
        & np.isfinite(y_all)
        & (x_all >= xmin)
        & (x_all <= usable_xmax)
        & (y_all >= ymin)
        & (y_all <= ymax)
    )

    x = x_all[keep]
    y = y_all[keep]

    gene_xy = np.column_stack([x, y])
    domain_xy = np.column_stack([seam_x + (usable_xmax - x), y])
    return gene_xy, domain_xy, keep


def draw_mirrored_frame(ax) -> None:
    xmin, xmax, ymin, ymax = get_mirrored_extent()
    rect = Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        edgecolor=FRAME_COLOR,
        linewidth=MIRROR_FRAME_WIDTH,
        linestyle=DASHED_STYLE,
        zorder=20,
    )
    ax.add_patch(rect)

    if DRAW_MIRROR_DIVIDER:
        seam_x = get_mirrored_parameters()[4]
        ax.plot(
            [seam_x, seam_x],
            [ymin, ymax],
            color=FRAME_COLOR,
            linewidth=0.9,
            linestyle=DASHED_STYLE,
            zorder=19,
        )


def style_mirrored_axis(ax) -> None:
    xmin, xmax, ymin, ymax = get_mirrored_extent()
    xpad = 0.02 * (xmax - xmin)
    ypad = 0.02 * (ymax - ymin)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ============================================================
# AXIS PIXEL/RASTER HELPERS
# ============================================================

def _axis_regular_grid(
    xy: np.ndarray,
    values: np.ndarray,
    decimals: int = AXIS_PIXEL_COORD_DECIMALS,
):
    """
    Convert regularly arranged AXIS coordinates to a dense x/y raster grid.

    Only supplied coordinates are filled. Missing locations remain NaN, which
    is rendered as white. The function intentionally does not interpolate.
    """
    xy = np.asarray(xy, dtype=float)
    values = np.asarray(values)

    if len(xy) != len(values):
        raise ValueError("Coordinate/value length mismatch in AXIS rasterization.")
    if len(xy) == 0:
        return np.array([]), np.array([]), np.empty((0, 0))

    xr = np.round(xy[:, 0], decimals)
    yr = np.round(xy[:, 1], decimals)

    x_unique = np.sort(np.unique(xr))
    y_unique = np.sort(np.unique(yr))

    x_to_i = {float(v): i for i, v in enumerate(x_unique)}
    y_to_i = {float(v): i for i, v in enumerate(y_unique)}

    # Numeric maps use float/NaN. Categorical domain maps also store domain IDs
    # numerically and are converted to RGBA separately.
    grid = np.full((len(y_unique), len(x_unique)), np.nan, dtype=float)

    for (xv, yv), value in zip(zip(xr, yr), values):
        xi = x_to_i[float(xv)]
        yi = y_to_i[float(yv)]
        try:
            grid[yi, xi] = float(value)
        except Exception:
            continue

    return x_unique, y_unique, grid


def _grid_extent(x_unique: np.ndarray, y_unique: np.ndarray) -> Tuple[float, float, float, float]:
    """Build an imshow extent centered on the regular AXIS grid coordinates."""
    if len(x_unique) == 0 or len(y_unique) == 0:
        return 0.0, 1.0, 0.0, 1.0

    if len(x_unique) > 1:
        dx = float(np.median(np.diff(x_unique)))
    else:
        dx = 1.0

    if len(y_unique) > 1:
        dy = float(np.median(np.diff(y_unique)))
    else:
        dy = 1.0

    return (
        float(x_unique[0] - dx / 2.0),
        float(x_unique[-1] + dx / 2.0),
        float(y_unique[0] - dy / 2.0),
        float(y_unique[-1] + dy / 2.0),
    )


def _render_axis_gene_pixels(
    ax,
    xy: np.ndarray,
    values: np.ndarray,
    vmax: float,
) -> None:
    """Render AXIS gene values as true pixel-like raster cells."""
    x_unique, y_unique, grid = _axis_regular_grid(xy, values)
    extent = _grid_extent(x_unique, y_unique)

    cmap = plt.cm.Reds.copy()
    cmap.set_bad("white")

    masked = np.ma.masked_invalid(grid)
    ax.imshow(
        masked,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
        zorder=2,
    )


def _render_axis_domain_pixels(
    ax,
    xy: np.ndarray,
    domains: np.ndarray,
    selected_domains: Optional[Sequence[int]] = None,
    group_other: bool = False,
) -> None:
    """Render AXIS spatial domains as pixel-like raster cells."""
    x_unique, y_unique, grid = _axis_regular_grid(xy, domains)
    extent = _grid_extent(x_unique, y_unique)

    rgba = np.ones((grid.shape[0], grid.shape[1], 4), dtype=float)
    rgba[..., :3] = 1.0
    rgba[..., 3] = 1.0

    valid = np.isfinite(grid)
    if np.any(valid):
        flat_domains = grid[valid].astype(int)
        domain_colors = domain_facecolors(
            flat_domains,
            selected_domains=selected_domains,
            group_other=group_other,
        )

        from matplotlib.colors import to_rgba
        rgba_values = np.asarray([to_rgba(c) for c in domain_colors], dtype=float)
        rgba[valid] = rgba_values

    ax.imshow(
        rgba,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
        zorder=2,
    )


# ============================================================
# MIRRORED VIRTUAL SLICE
# ============================================================


def plot_mirrored_virtual_slice(
    ax,
    record: SliceRecord,
    gene_vmax: float,
) -> None:
    """
    Render one mirrored AXIS/SpatialZ comparison panel.

    Left half:
        Gfap expression
    Right half:
        mirrored spatial domains

    Rendering rules requested by the current design:
    1. 4.1-month case shows the full spatial-domain map.
    2. 11.8-month case shows gene expression only on the known anatomical
       regions listed in HIGHLIGHT_118.
    3. Domain panels keep full per-domain colors, which makes the shared legend
       easier to interpret.
    """
    gene_xy, domain_xy, retained_mask = mirrored_half_coords(record)

    gene_values = np.asarray(record.gene_values)[retained_mask]
    domains = np.asarray(record.domains)[retained_mask]

    is_axis = record.method.lower() == "axis"
    is_11p8 = np.isclose(float(record.age), 11.8)
    is_4p1 = np.isclose(float(record.age), 4.1)

    # Gene-map masking for the 11.8-month case.
    gene_keep_mask = np.ones(len(gene_values), dtype=bool)
    if is_11p8 and SHOW_GENE_ONLY_ON_KNOWN_11P8:
        gene_keep_mask = np.isin(domains.astype(int), HIGHLIGHT_118)

    gene_xy_plot = gene_xy[gene_keep_mask]
    gene_values_plot = gene_values[gene_keep_mask]

    # Domain rendering setup.
    if is_4p1 and SHOW_FULL_DOMAINS_4P1:
        domain_selected = None
    else:
        domain_selected = None  # keep all domains unless changed later

    group_other = False if SHOW_FULL_DOMAIN_COLORS else True

    # ------------------------------------------------------------------
    # Left half: gene expression
    # ------------------------------------------------------------------
    if is_axis and AXIS_PIXEL_RENDERING:
        _render_axis_gene_pixels(
            ax=ax,
            xy=gene_xy_plot,
            values=gene_values_plot,
            vmax=gene_vmax,
        )
    else:
        gene_colors = gene_facecolors(
            gene_values_plot,
            vmax=gene_vmax,
        ).copy()
        gene_colors[:, 3] = np.clip(
            gene_colors[:, 3] * VIRTUAL_ALPHA,
            0.03,
            1.0,
        )
        ax.scatter(
            gene_xy_plot[:, 0],
            gene_xy_plot[:, 1],
            c=gene_colors,
            s=GENE_POINT_SIZE,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    # ------------------------------------------------------------------
    # Right half: mirrored spatial domains
    # ------------------------------------------------------------------
    if is_axis and AXIS_PIXEL_RENDERING:
        _render_axis_domain_pixels(
            ax=ax,
            xy=domain_xy,
            domains=domains,
            selected_domains=domain_selected,
            group_other=group_other,
        )
    else:
        domain_colors = domain_facecolors(
            domains,
            selected_domains=domain_selected,
            group_other=group_other,
        )
        ax.scatter(
            domain_xy[:, 0],
            domain_xy[:, 1],
            c=domain_colors,
            s=DOMAIN_POINT_SIZE,
            alpha=VIRTUAL_ALPHA,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    draw_mirrored_frame(ax)
    style_mirrored_axis(ax)



# ============================================================
# SHARED BOTTOM ANATOMICAL LEGEND
# ============================================================


def add_shared_bottom_anatomical_legend(fig) -> None:
    """
    One shared anatomical legend for the full 2x2 figure.

    The named groups are shown first. The "Other" section additionally lists the
    individual domain IDs and colors of the remaining domains, so the viewer can
    still identify the specific domain colors on the domain maps.
    """
    import math

    legend_rect = LEGEND_RECT
    title_fontsize = LEGEND_FONTSIZE
    item_fontsize = LEGEND_FONTSIZE
    marker_size = LEGEND_MARKER_SIZE

    title_y = LEGEND_TITLE_Y
    row_y = LEGEND_ROW_Y
    dot_text_dx = LEGEND_DOT_TEXT_DX

    groups = [
        ("Cerebral cortex", [1, 4, 6, 9, 10], 3),
        ("Hippocampus", [8, 11], 1),
        ("Thalamus", [0], 1),
        ("Hypothalamus", [3], 1),
        ("Other", OTHER_DOMAIN_LABELS, 2),
    ]

    ax_leg = fig.add_axes(legend_rect)
    ax_leg.set_axis_off()
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)

    def draw_group(title, labels, ncols, x0):
        ax_leg.text(
            x0,
            title_y,
            title,
            ha="left",
            va="bottom",
            fontsize=title_fontsize,
            fontweight="bold",
            color=TEXT_COLOR,
        )

        if ncols > 1:
            nrows = math.ceil(len(labels) / ncols)
        else:
            nrows = len(labels)

        col_dx = LEGEND_COLUMN_DX.get(title, 0.12)

        for idx, lab in enumerate(labels):
            if ncols > 1:
                r = idx % nrows
                c = idx // nrows
            else:
                r = idx
                c = 0

            x = x0 + c * col_dx
            y = row_y[min(r, len(row_y) - 1)]

            lab = int(lab)
            color = LABEL_TO_COLOR.get(lab, OTHER_DOMAIN_COLOR)
            label_txt = f"Domain {lab}"

            ax_leg.scatter(
                [x],
                [y],
                s=marker_size**2,
                color=color,
                edgecolors="none",
                clip_on=False,
            )
            ax_leg.text(
                x + dot_text_dx,
                y,
                label_txt,
                ha="left",
                va="center",
                fontsize=item_fontsize,
                color=TEXT_COLOR,
            )

    for title, labels, ncols in groups:
        draw_group(
            title=title,
            labels=labels,
            ncols=ncols,
            x0=LEGEND_X_POSITIONS[title],
        )



# ============================================================
# SHARED BOTTOM LABELS FOR MIRRORED HALVES
# ============================================================

def add_bottom_half_annotations(fig) -> None:
    """
    Add one shared pair of labels for the mirrored halves of all four panels.

    The positions are controlled entirely by:
        BOTTOM_GENE_LABEL_X
        BOTTOM_DOMAIN_LABEL_X
        BOTTOM_HALF_LABEL_Y
        BOTTOM_HALF_LABEL_FONTSIZE
    """
    fig.text(
        BOTTOM_GENE_LABEL_X,
        BOTTOM_HALF_LABEL_Y,
        rf"$\it{{{GENE_NAME}}}$ expression" + " | Spatial domains",
        ha="center",
        va="center",
        fontsize=BOTTOM_HALF_LABEL_FONTSIZE,
        color=TEXT_COLOR,
    )

    # fig.text(
    #     BOTTOM_DOMAIN_LABEL_X,
    #     BOTTOM_HALF_LABEL_Y,
    #     "Spatial domains",
    #     ha="center",
    #     va="center",
    #     fontsize=BOTTOM_HALF_LABEL_FONTSIZE,
    #     color=TEXT_COLOR,
    # )
    #

# ============================================================
# COMBINED 2x2 OUTPUT FIGURE

# ============================================================

def load_all_comparison_records() -> Dict[Tuple[str, float], SliceRecord]:
    """Load the four AXIS/SpatialZ virtual tissues used in the comparison."""
    records: Dict[Tuple[str, float], SliceRecord] = {}

    for method in ("Axis", "SpatialZ"):
        for age in (4.1, 11.8):
            spec = VIRTUAL_SLICES[float(age)]
            records[(method, age)] = load_virtual_slice_record(
                method=method,
                age=float(age),
                query_value=float(spec["query_value"]),
            )

    return records


def shared_gene_vmax(records: Sequence[SliceRecord]) -> float:
    """Use one expression scale across every AXIS/SpatialZ comparison panel."""
    vals = []
    for record in records:
        x = np.asarray(record.gene_values, dtype=float)
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)

    if not vals:
        return 1.0

    all_values = np.concatenate(vals)
    vmax = np.percentile(all_values, GENE_COLOR_PERCENTILE)
    return max(float(vmax), 1e-8)



def plot_combined_virtual_comparison() -> None:
    """
    Generate one 2x2 AXIS-versus-SpatialZ comparison.

    Layout
    ------
            4.1 months       11.8 months
    AXIS       [0,0]             [0,1]
    SpatialZ   [1,0]             [1,1]
    """
    ensure_dir(OUTDIR)
    set_style()

    records = load_all_comparison_records()
    vmax = shared_gene_vmax(list(records.values()))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=COMBINED_FIGSIZE,
        squeeze=False,
    )

    panel_order = [
        ("Axis", 4.1),
        ("Axis", 11.8),
        ("SpatialZ", 4.1),
        ("SpatialZ", 11.8),
    ]

    for ax, (method, age) in zip(axes.ravel(), panel_order):
        record = records[(method, age)]
        plot_mirrored_virtual_slice(
            ax=ax,
            record=record,
            gene_vmax=vmax,
        )

    # Column labels: ages.
    axes[0, 0].set_title(
        "Virtual tissue at 4.1 months",
        fontsize=COLUMN_LABEL_FONTSIZE,
        fontweight="normal",
        pad=6,
    )
    axes[0, 1].set_title(
        "Virtual tissue at 11.8 months",
        fontsize=COLUMN_LABEL_FONTSIZE,
        fontweight="normal",
        pad=6,
    )

    # Row labels: methods.
    fig.text(
        0.04,
        0.68,
        "AXIS",
        ha="center",
        va="center",
        # rotation=90,
        fontsize=ROW_LABEL_FONTSIZE,
        # fontweight="bold",
        color=TEXT_COLOR,
    )
    fig.text(
        0.04,
        0.38,
        "SpatialZ",
        ha="center",
        va="center",
        # rotation=90,
        fontsize=ROW_LABEL_FONTSIZE,
        # fontweight="bold",
        color=TEXT_COLOR,
    )

    fig.subplots_adjust(
        left=FIG_LEFT,
        right=FIG_RIGHT,
        top=FIG_TOP,
        bottom=FIG_BOTTOM,
        wspace=SUBPLOT_WSPACE,
        hspace=SUBPLOT_HSPACE,
    )

    add_gene_expression_colorbar(
        fig=fig,
        vmax=vmax,
        gene_name=GENE_NAME,
        rect=SHARED_GENE_COLORBAR_RECT,
    )

    add_bottom_half_annotations(fig)
    add_shared_bottom_anatomical_legend(fig)

    save_path = os.path.join(

        OUTDIR,
        "axis_vs_spatialz_virtual_4p1m_11p8m_2x2.png",
    )
    finalize_figure(fig, save_path)


# ============================================================
# MAIN
# ============================================================

def run_all() -> None:
    plot_combined_virtual_comparison()
    print(f"[DONE] Combined comparison written/displayed from: {OUTDIR}")


if __name__ == "__main__":
    run_all()
