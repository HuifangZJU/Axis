#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mouse-time pipeline figures (updated for the Tg mouse-brain dataset).

This version uses:
- real slices from:
    ROOT/h5ads/Tg/{GROUP_ID}_{sample_id}.h5ad
- real-slice coordinates from:
    ROOT/network_input/{GROUP_ID}_{sample_id}_validation.npz
- virtual slices from:
    ROOT/virtual_tissue/AD_time_grid256_spatial2.0/query/virtual_AD_time_{query_value:.3f}.h5ad

Figures
-------
1. Real-slice stacks (gene stack and domain stack).
2. Real + virtual slice stacks in time order (gene stack and domain stack).
3. Mirrored stack of all slices (Allen-atlas style).
4. 2D mirrored grid of real slices.
5. 2D mirrored grid of virtual slices.
6. 3D temporal stack with separate gene and domain views.
7. 3D mirrored temporal stack.

Notes
-----
- Gene maps use log1p(gene) with Reds colormap and alpha proportional to value.
- Real slices read domains from h5ad, but read coordinates from the NPZ validation files.
- Virtual slices read both domains and coordinates from the h5ad file.
- Coordinate swapping and x/y reflection are applied once during data loading.
- Mirrored panels then place the gene map on the left and a horizontal mirror
  of the domain map on the right without any further coordinate swapping.
- For the 11.8m virtual slice, only domains [1,4,6,8,10,11] are highlighted;
  all other domains are shown in gray for both the gene and domain panels.
"""

import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# ============================================================
# DATASET-SPECIFIC CONFIG
# ============================================================
ROOT = "/media/huifang/data/registration/xenium/mouse_brain"
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ads", "Tg")
VIRTUAL_DIR = os.path.join(
    ROOT,
    "virtual_tissue",
    "AD_time_grid256_spatial2.0",
    "query",
)

# VIRTUAL_DIR = "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz"



GROUP_ID = 0

# Real slices: sample_id 0,1,2 correspond to 2.5m, 5.7m, 17.9m.
REAL_SLICE_SPECS = [
    {"age": 2.5, "label": "2.5m", "sample_id": 0},
    {"age": 5.7, "label": "5.7m", "sample_id": 1},
    {"age": 17.9, "label": "17.9m", "sample_id": 2},
]

# Virtual slices: update/add query values here when needed.
# The filename pattern is virtual_AD_time_{query_value:.3f}.h5ad
VIRTUAL_SLICE_SPECS = [
    {"age": 4.1, "label": "4.1m", "query_value": 4.100},
    {"age": 11.8, "label": "11.8m", "query_value": 11.800},
]

GENE_NAME = "Gfap"
REAL_DOMAIN_KEY = "axis_joint_domain"
VIRTUAL_DOMAIN_KEY = "axis_joint_domain"
VIRTUAL_COORD_KEY = "spatial"

OUTDIR = "/media/huifang/data/registration/paper_figure/Figure5_mouse_time/pipeline"

# Domain colors specified by the user.
ALL_LABELS = list(range(13))
BASE_CMAP = plt.get_cmap("tab20")
LABEL_TO_COLOR = {lab: BASE_CMAP(i) for i, lab in enumerate(ALL_LABELS)}

# Highlighted domains for the 11.8m virtual panel.
HIGHLIGHT_118 = [0,1,3,4, 6, 8,9, 10, 11]

# Style
FIG_DPI = 400

# Figure output control:
#   "save" -> save files only
#   "show" -> display interactively only
#   "both" -> save and display every figure
FIGURE_MODE = "show"
SHOW_BLOCK = True

BACKGROUND_GRAY = "#CFCFCF"
FRAME_COLOR = "#A9A9A9"
TEXT_COLOR = "#222222"

STACK_DX = -55.0
STACK_DY = 38.0

GENE_POINT_SIZE = 5.0
DOMAIN_POINT_SIZE = 5.0

REAL_ALPHA = 0.85
VIRTUAL_ALPHA = 0.72
GENE_BG_ALPHA = 0.12

MIRROR_FRAME_WIDTH = 1.6
STACK_FRAME_WIDTH = 1.5
DASHED_STYLE = (0, (4.0, 2.4))

# Fixed square bounding box used for every slice in plotting coordinates.
FIXED_BOX = (0.0, 1024.0, 0.0, 1024.0)

# Global coordinate orientation.
# These transformations are applied once, immediately when coordinates load.
# The plotting functions therefore always receive coordinates in their final
# display orientation.
SWAP_XY = True
INVERT_X = True
INVERT_Y = False

# Draw the division between the left gene half and right mirrored-domain half.
DRAW_MIRROR_DIVIDER = True

# Mirrored-view geometry: keep the original x-scale, but trim a small right
# margin before mirroring so the center fold becomes a straight line.
MIRROR_RIGHT_TRIM = 70.0

# 3D temporal-stack controls. These figures are added in addition to the
# existing 2D stack and grid figures.
GENERATE_3D_SEPARATE_STACK = True
GENERATE_3D_MIRRORED_STACK = True

THREE_D_POINT_SIZE = 2.0
THREE_D_MAX_POINTS_PER_SLICE = 30000
THREE_D_GENE_ALPHA_MIN = 0.05
THREE_D_GENE_ALPHA_MAX = 0.82
THREE_D_DOMAIN_ALPHA = 0.72
THREE_D_BACKGROUND_ALPHA = 0.12

# Camera angle. These can be adjusted interactively when --figure-mode show
# or --figure-mode both is used.
THREE_D_ELEV = 18
THREE_D_AZIM = -68

# Physical aspect of the rendered 3D boxes.
THREE_D_BOX_ASPECT = (1.0, 1.0, 0.72)
THREE_D_MIRROR_BOX_ASPECT = (1.75, 1.0, 0.72)

# Optional vertical guides connecting the slice-frame corners.
THREE_D_CONNECT_CORNERS = False


# ============================================================
# DATA STRUCTURES
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
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finalize_figure(
    fig,
    save_path: str,
    figure_mode: Optional[str] = None,
) -> None:
    """
    Save, show, or both, then close the figure.

    Parameters
    ----------
    fig
        Matplotlib figure.
    save_path
        Output path used when mode is "save" or "both".
    figure_mode
        "save", "show", or "both". When None, uses FIGURE_MODE.
    """
    mode = FIGURE_MODE if figure_mode is None else str(figure_mode).lower()
    if mode not in {"save", "show", "both"}:
        raise ValueError(
            f"Invalid figure mode {mode!r}; choose 'save', 'show', or 'both'."
        )

    if mode in {"save", "both"}:
        ensure_dir(os.path.dirname(save_path) or ".")
        fig.savefig(
            save_path,
            dpi=FIG_DPI,
            bbox_inches="tight",
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
    f = os.path.join(net_in, f"{group_id}_{sample_id}_validation.npz")
    d = np.load(f)
    return d["coord"].astype(np.float32)


def get_real_h5ad_path(group_id: int, sample_id: int) -> str:
    return os.path.join(H5AD_DIR, f"{group_id}_{sample_id}.h5ad")


def get_virtual_h5ad_path(query_value: float) -> str:
    return os.path.join(VIRTUAL_DIR, f"virtual_AD_time_{query_value:.3f}.h5ad")
    # return os.path.join(VIRTUAL_DIR, f"{query_value:.1f}.h5ad")


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
    coords = transform_loaded_coords(
        load_coords_npz(NET_IN, group_id, sample_id)
    )

    if gene_name not in adata.var_names:
        raise KeyError(f"Gene {gene_name!r} not found in {path}")
    gene_values = _to_dense_1d(adata[:, [gene_name]].X)
    gene_values = np.log1p(np.clip(gene_values, a_min=0.0, a_max=None))

    if domain_key not in adata.obs:
        raise KeyError(f"{domain_key!r} not found in adata.obs for {path}")
    domains = _normalize_domain_array(adata.obs[domain_key].to_numpy())

    if coords.shape[0] != adata.n_obs:
        raise ValueError(
            f"Coordinate size mismatch for real slice {label}: "
            f"{coords.shape[0]} coords vs {adata.n_obs} observations"
        )

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
    coords = transform_loaded_coords(
        np.asarray(adata.obsm[coord_key], dtype=float)
    )

    if gene_name not in adata.var_names:
        raise KeyError(f"Gene {gene_name!r} not found in {path}")
    gene_values = _to_dense_1d(adata[:, [gene_name]].X)
    gene_values = np.log1p(np.clip(gene_values, a_min=0.0, a_max=None))

    if domain_key not in adata.obs:
        raise KeyError(f"{domain_key!r} not found in adata.obs for {path}")
    domains = _normalize_domain_array(adata.obs[domain_key].to_numpy())

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
        load_real_slice_record(GROUP_ID, item["sample_id"], item["age"], item["label"])
        for item in REAL_SLICE_SPECS
    ]
    virtual_records = [
        load_virtual_slice_record(item["query_value"], item["age"], item["label"])
        for item in VIRTUAL_SLICE_SPECS
    ]
    real_records = sorted(real_records, key=lambda r: r.age)
    virtual_records = sorted(virtual_records, key=lambda r: r.age)
    return real_records, virtual_records


def extent_corners(
    extent: Tuple[float, float, float, float],
    offset: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    xmin, xmax, ymin, ymax = extent
    dx, dy = offset
    return np.asarray(
        [
            [xmin + dx, ymin + dy],
            [xmin + dx, ymax + dy],
            [xmax + dx, ymin + dy],
            [xmax + dx, ymax + dy],
        ],
        dtype=float,
    )


def transform_loaded_coords(coords: np.ndarray) -> np.ndarray:
    """
    Convert raw coordinates into the final plotting orientation once at load.

    Processing order
    ----------------
    1. Keep the first two coordinate columns.
    2. Swap x and y when SWAP_XY is True.
    3. Reflect x within FIXED_BOX when INVERT_X is True.
    4. Reflect y within FIXED_BOX when INVERT_Y is True.

    With the current settings:
        SWAP_XY = True
        INVERT_X = True
        INVERT_Y = False
    """
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"Expected coordinates with shape (n, 2+), got {coords.shape}"
        )

    coords = coords[:, :2].copy()

    if SWAP_XY:
        coords = coords[:, [1, 0]]

    xmin, xmax, ymin, ymax = FIXED_BOX

    if INVERT_X:
        coords[:, 0] = xmin + xmax - coords[:, 0]

    if INVERT_Y:
        coords[:, 1] = ymin + ymax - coords[:, 1]

    return coords


def get_record_extent(record: SliceRecord) -> Tuple[float, float, float, float]:
    # Use one fixed square bounding box for all slices in plotting coordinates
    # so the frames are directly comparable.
    return FIXED_BOX


def get_mirrored_parameters() -> Tuple[float, float, float, float, float]:
    """
    Geometry for the mirrored view after trimming a small right margin.

    Returns
    -------
    xmin, usable_xmax, ymin, ymax, seam_x
        Left gene view uses [xmin, usable_xmax].
        Right mirrored domain view uses [seam_x, seam_x + (usable_xmax - xmin)].
    """
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


def set_data_limits(
    ax,
    arrays: Sequence[np.ndarray],
    pad_ratio: float = 0.06,
) -> None:
    """
    Set limits from coordinates already transformed into display orientation.

    No additional x/y reflection is performed here.
    """
    all_xy = np.vstack(arrays)
    xmin, ymin = np.min(all_xy, axis=0)
    xmax, ymax = np.max(all_xy, axis=0)

    padx = max((xmax - xmin) * pad_ratio, 1e-8)
    pady = max((ymax - ymin) * pad_ratio, 1e-8)

    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def gene_facecolors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros((0, 4), dtype=float)

    finite = values[np.isfinite(values)]
    vmax = np.percentile(finite, 99.0) if finite.size else 1.0
    vmax = max(float(vmax), 1e-8)
    vnorm = np.clip(values / vmax, 0.0, 1.0)

    colors = plt.cm.Reds(vnorm)
    colors[:, 3] = np.clip(0.08 + 0.92 * vnorm, 0.08, 1.0)
    return colors


def domain_facecolors(
    domains: np.ndarray,
    highlight_domains: Optional[Sequence[int]] = None,
    other_color=BACKGROUND_GRAY,
) -> np.ndarray:
    domains = _normalize_domain_array(domains)
    use_subset = highlight_domains is not None
    keep = set(int(x) for x in highlight_domains) if use_subset else set()

    colors = []
    for d in domains:
        if use_subset and d not in keep:
            colors.append(other_color)
        else:
            colors.append(LABEL_TO_COLOR.get(int(d), other_color))
    return np.asarray(colors, dtype=object)


def draw_frame(
    ax,
    extent: Tuple[float, float, float, float],
    offset: Tuple[float, float] = (0.0, 0.0),
    linestyle="-",
    linewidth: float = STACK_FRAME_WIDTH,
    edgecolor=FRAME_COLOR,
    zorder: int = 20,
) -> None:
    xmin, xmax, ymin, ymax = extent
    dx, dy = offset
    rect = Rectangle(
        (xmin + dx, ymin + dy),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(rect)


def add_slice_label(
    ax,
    record: SliceRecord,
    extent: Tuple[float, float, float, float],
    offset: Tuple[float, float] = (0.0, 0.0),
    fontsize: float = 11.0,
) -> None:
    xmin, xmax, ymin, ymax = extent
    dx, dy = offset
    ax.text(
        xmin + dx,
        ymin + dy - 0.04 * (ymax - ymin),
        record.label,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=TEXT_COLOR,
    )


def plot_single_gene_map(
    ax,
    record: SliceRecord,
    offset: Tuple[float, float] = (0.0, 0.0),
    selected_domains: Optional[Sequence[int]] = None,
    point_size: float = GENE_POINT_SIZE,
    draw_bbox: bool = True,
    dashed: bool = False,
    label_slice: bool = False,
) -> np.ndarray:
    coords = record.coords.copy()
    coords[:, 0] += offset[0]
    coords[:, 1] += offset[1]

    colors = gene_facecolors(record.gene_values)
    if selected_domains is not None:
        keep = np.isin(record.domains, np.asarray(selected_domains, dtype=int))
        bg = ~keep
        if np.any(bg):
            ax.scatter(
                coords[bg, 0], coords[bg, 1],
                c=BACKGROUND_GRAY,
                s=point_size,
                alpha=GENE_BG_ALPHA,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )
        coords = coords[keep]
        colors = colors[keep]

    alpha_factor = REAL_ALPHA if record.is_real else VIRTUAL_ALPHA
    colors = colors.copy()
    colors[:, 3] = np.clip(colors[:, 3] * alpha_factor, 0.03, 1.0)

    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors,
        s=point_size,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    extent = get_record_extent(record)
    if draw_bbox:
        draw_frame(
            ax, extent, offset=offset,
            linestyle=DASHED_STYLE if dashed else "-",
            linewidth=STACK_FRAME_WIDTH,
        )
    if label_slice:
        add_slice_label(ax, record, extent, offset=offset)

    extent = get_record_extent(record)
    return np.vstack([coords, extent_corners(extent, offset=offset)])


def plot_single_domain_map(
    ax,
    record: SliceRecord,
    offset: Tuple[float, float] = (0.0, 0.0),
    selected_domains: Optional[Sequence[int]] = None,
    point_size: float = DOMAIN_POINT_SIZE,
    draw_bbox: bool = True,
    dashed: bool = False,
    label_slice: bool = False,
) -> np.ndarray:
    coords = record.coords.copy()
    coords[:, 0] += offset[0]
    coords[:, 1] += offset[1]

    colors = domain_facecolors(record.domains, highlight_domains=selected_domains)
    alpha_value = REAL_ALPHA if record.is_real else VIRTUAL_ALPHA

    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha_value,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    extent = get_record_extent(record)
    if draw_bbox:
        draw_frame(
            ax, extent, offset=offset,
            linestyle=DASHED_STYLE if dashed else "-",
            linewidth=STACK_FRAME_WIDTH,
        )
    if label_slice:
        add_slice_label(ax, record, extent, offset=offset)

    extent = get_record_extent(record)
    return np.vstack([coords, extent_corners(extent, offset=offset)])


def mirrored_half_coords(
    record: SliceRecord,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create the Allen-atlas-style mirrored view from display-ready coordinates.

    Design
    ------
    1. Coordinates are already transformed into the final display orientation.
    2. The left half (gene map) keeps the original x-scale.
    3. The right half (domain map) is mirrored at the same x-scale.
    4. A small right margin is removed before mirroring so the two halves meet
       along one perfectly straight center seam.
    """
    xmin, usable_xmax, ymin, ymax, seam_x = get_mirrored_parameters()

    x_all = np.asarray(record.coords[:, 0], dtype=float)
    y_all = np.asarray(record.coords[:, 1], dtype=float)

    keep_mask = (
        np.isfinite(x_all)
        & np.isfinite(y_all)
        & (x_all >= xmin)
        & (x_all <= usable_xmax)
        & (y_all >= ymin)
        & (y_all <= ymax)
    )

    x = x_all[keep_mask]
    y = y_all[keep_mask]

    # Left: original x-scale.
    gene_xy = np.column_stack([x + offset[0], y + offset[1]])

    # Right: mirror at the same x-scale.
    domain_xy = np.column_stack([
        seam_x + (usable_xmax - x) + offset[0],
        y + offset[1],
    ])

    return gene_xy, domain_xy, keep_mask


def plot_mirrored_slice(
    ax,
    record: SliceRecord,
    offset: Tuple[float, float] = (0.0, 0.0),
    selected_domains: Optional[Sequence[int]] = None,
    gene_point_size: float = GENE_POINT_SIZE,
    domain_point_size: float = DOMAIN_POINT_SIZE,
    draw_bbox: bool = True,
    dashed: bool = False,
    label_slice: bool = False,
) -> np.ndarray:
    gene_xy, dom_xy, keep_mask = mirrored_half_coords(record, offset=offset)

    gene_values = np.asarray(record.gene_values)[keep_mask]
    kept_domains = np.asarray(record.domains)[keep_mask]

    gene_colors = gene_facecolors(gene_values).copy()
    alpha_factor = REAL_ALPHA if record.is_real else VIRTUAL_ALPHA

    if selected_domains is not None:
        keep_gene = np.isin(kept_domains, np.asarray(selected_domains, dtype=int))
        bg_gene = ~keep_gene

        if np.any(bg_gene):
            ax.scatter(
                gene_xy[bg_gene, 0], gene_xy[bg_gene, 1],
                c=BACKGROUND_GRAY,
                s=gene_point_size,
                alpha=GENE_BG_ALPHA,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )

        gene_xy_plot = gene_xy[keep_gene]
        gene_colors_plot = gene_colors[keep_gene]
    else:
        gene_xy_plot = gene_xy
        gene_colors_plot = gene_colors

    gene_colors_plot[:, 3] = np.clip(
        gene_colors_plot[:, 3] * alpha_factor, 0.03, 1.0
    )
    ax.scatter(
        gene_xy_plot[:, 0], gene_xy_plot[:, 1],
        c=gene_colors_plot,
        s=gene_point_size,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    dom_colors = domain_facecolors(kept_domains, highlight_domains=selected_domains)
    ax.scatter(
        dom_xy[:, 0], dom_xy[:, 1],
        c=dom_colors,
        s=domain_point_size,
        alpha=alpha_factor,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    extent = get_mirrored_extent()
    if draw_bbox:
        draw_frame(
            ax, extent, offset=offset,
            linestyle=DASHED_STYLE if dashed else "-",
            linewidth=MIRROR_FRAME_WIDTH,
        )

        if DRAW_MIRROR_DIVIDER:
            _, _, ymin, ymax = extent
            seam_x = get_mirrored_parameters()[4] + offset[0]
            ax.plot(
                [seam_x, seam_x],
                [ymin + offset[1], ymax + offset[1]],
                color=FRAME_COLOR,
                linewidth=0.9,
                linestyle=DASHED_STYLE if dashed else "-",
                zorder=19,
            )

    if label_slice:
        add_slice_label(ax, record, extent, offset=offset)

    return np.vstack([gene_xy, dom_xy, extent_corners(extent, offset=offset)])


def make_domain_legend_handles(
    labels: Sequence[int],
    include_other: bool = False,
    other_label: str = "Other domains",
) -> List[Line2D]:
    """Create a flat domain legend."""
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=LABEL_TO_COLOR.get(int(label), BACKGROUND_GRAY),
            markeredgecolor="none",
            markersize=7,
            label=f"Domain {int(label)}",
        )
        for label in labels
    ]

    if include_other:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=BACKGROUND_GRAY,
                markeredgecolor="none",
                markersize=7,
                label=other_label,
            )
        )

    return handles


def make_grouped_domain_legend_handles(
    groups: Sequence[Dict],
    include_other: bool = False,
    other_label: str = "Other domains",
) -> Tuple[List[Line2D], set]:
    """
    Create a grouped domain legend with text-only section headers.

    Example
    -------
    Cortex
      Domain 1
      Domain 4
      Domain 6
      Domain 10

    Hippocampus
      Domain 8
      Domain 11
    """
    handles: List[Line2D] = []
    section_headers = set()

    for group_index, group in enumerate(groups):
        group_name = str(group["name"])
        group_labels = [int(label) for label in group["labels"]]

        # Text-only section header.
        handles.append(
            Line2D(
                [],
                [],
                linestyle="",
                marker="",
                color="none",
                label=group_name,
            )
        )
        section_headers.add(group_name)

        for label in group_labels:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=LABEL_TO_COLOR.get(
                        label,
                        BACKGROUND_GRAY,
                    ),
                    markeredgecolor="none",
                    markersize=7,
                    label=f"Domain {label}",
                )
            )

        # Add visual separation between anatomical groups.
        if group_index < len(groups) - 1:
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="",
                    marker="",
                    color="none",
                    label=" ",
                )
            )

    if include_other:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=BACKGROUND_GRAY,
                markeredgecolor="none",
                markersize=7,
                label=other_label,
            )
        )

    return handles, section_headers


# ============================================================
# FIGURES
# ============================================================
def plot_figure1_real_stacks(real_records: Sequence[SliceRecord], outdir: str) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    ax_gene, ax_domain = axes

    all_coords_gene = []
    all_coords_domain = []

    for idx, record in enumerate(reversed(real_records)):
        ridx = len(real_records) - 1 - idx
        offset = (ridx * STACK_DX, ridx * STACK_DY)
        all_coords_gene.append(
            plot_single_gene_map(ax_gene, record, offset=offset, dashed=False, label_slice=True)
        )
        all_coords_domain.append(
            plot_single_domain_map(ax_domain, record, offset=offset, dashed=False, label_slice=True)
        )

    ax_gene.set_title(f"Observed data | {GENE_NAME} gene map")
    ax_domain.set_title("Observed data | Spatial domains")
    set_data_limits(ax_gene, all_coords_gene)
    set_data_limits(ax_domain, all_coords_domain)

    fig.tight_layout()
    save_path = os.path.join(outdir, "figure1_observed_real_stacks.png")
    finalize_figure(fig, save_path)


def plot_figure2_all_stacks(records: Sequence[SliceRecord], outdir: str) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6))
    ax_gene, ax_domain = axes

    all_gene_xy = []
    all_domain_xy = []

    ordered = sorted(records, key=lambda r: r.age)
    for idx, record in enumerate(reversed(ordered)):
        ridx = len(ordered) - 1 - idx
        offset = (ridx * STACK_DX, ridx * STACK_DY)
        selected = None
        all_gene_xy.append(
            plot_single_gene_map(
                ax_gene, record, offset=offset,
                selected_domains=selected,
                dashed=(not record.is_real), label_slice=True,
            )
        )
        all_domain_xy.append(
            plot_single_domain_map(
                ax_domain, record, offset=offset,
                selected_domains=selected,
                dashed=(not record.is_real), label_slice=True,
            )
        )

    ax_gene.set_title(f"Time-ordered slices | {GENE_NAME} gene map")
    ax_domain.set_title("Time-ordered slices | Spatial domains")
    set_data_limits(ax_gene, all_gene_xy)
    set_data_limits(ax_domain, all_domain_xy)

    fig.tight_layout()
    save_path = os.path.join(outdir, "figure2_all_slices_stacks.png")
    finalize_figure(fig, save_path)


def plot_figure3_mirrored_stack(records: Sequence[SliceRecord], outdir: str) -> None:
    set_style()
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 6.4))

    all_xy = []
    ordered = sorted(records, key=lambda r: r.age)
    for idx, record in enumerate(reversed(ordered)):
        ridx = len(ordered) - 1 - idx
        offset = (ridx * STACK_DX, ridx * STACK_DY)
        selected = HIGHLIGHT_118 if (not record.is_real and abs(record.age - 11.8) < 1e-8) else None
        all_xy.append(
            plot_mirrored_slice(
                ax, record, offset=offset,
                selected_domains=selected,
                dashed=(not record.is_real), label_slice=True,
            )
        )

    ax.set_title("Mirrored temporal stack | Gene (left) and domain (right)")
    set_data_limits(ax, all_xy)

    fig.tight_layout()
    save_path = os.path.join(outdir, "figure3_mirrored_all_slices_stack.png")
    finalize_figure(fig, save_path)


def plot_mirrored_grid(
    records: Sequence[SliceRecord],
    save_path: str,
    title: str,
    per_panel_selected_domains: Optional[Dict[str, Sequence[int]]] = None,
    per_panel_legend: Optional[Dict[str, Dict]] = None,
) -> None:
    set_style()

    n_panels = len(records)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(5.2 * n_panels, 5.4),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, record in zip(axes, records):
        selected_domains = (
            None
            if per_panel_selected_domains is None
            else per_panel_selected_domains.get(record.label)
        )

        all_xy = plot_mirrored_slice(
            ax,
            record,
            offset=(0.0, 0.0),
            selected_domains=selected_domains,
            dashed=(not record.is_real),
            label_slice=False,
        )

        set_data_limits(ax, [all_xy])
        ax.set_title(record.label)

        if not per_panel_legend or record.label not in per_panel_legend:
            continue

        spec = per_panel_legend[record.label]
        section_headers = set()

        if "groups" in spec:
            handles, section_headers = make_grouped_domain_legend_handles(
                groups=spec["groups"],
                include_other=spec.get("include_other", False),
                other_label=spec.get(
                    "other_label",
                    "Other domains",
                ),
            )
        else:
            handles = make_domain_legend_handles(
                labels=spec.get("labels", []),
                include_other=spec.get("include_other", False),
                other_label=spec.get(
                    "other_label",
                    "Other domains",
                ),
            )

        legend = ax.legend(
            handles=handles,
            loc=spec.get("loc", "upper left"),
            bbox_to_anchor=spec.get(
                "bbox_to_anchor",
                (1.01, 1.0),
            ),
            frameon=False,
            title=spec.get("title"),
            fontsize=spec.get("fontsize", 8.5),
            title_fontsize=spec.get("title_fontsize", 9.5),
            handlelength=1.0,
            handletextpad=0.55,
            labelspacing=0.25,
            borderaxespad=0.0,
        )

        # Format the grouped anatomical section headers.
        for text in legend.get_texts():
            label = text.get_text()

            if label in section_headers:
                text.set_fontweight("bold")
                text.set_fontsize(
                    spec.get("section_fontsize", 9.5)
                )
            elif label == " ":
                text.set_text("")

    fig.suptitle(
        title,
        y=1.02,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    finalize_figure(fig, save_path)



# ============================================================
# 3D TEMPORAL STACKS
# ============================================================
def _sample_indices(
    indices: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    """Deterministically subsample a one-dimensional integer index array."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= max_points:
        return indices

    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(
            indices,
            size=max_points,
            replace=False,
        )
    )


def _select_3d_indices(
    record: SliceRecord,
    selected_domains: Optional[Sequence[int]],
    seed: int,
) -> np.ndarray:
    """
    Select points for a 3D layer.

    For the restricted 11.8m slice, highlighted cortical/hippocampal points are
    preferentially retained, while a smaller gray background sample preserves
    the full anatomical silhouette.
    """
    n_points = len(record.coords)
    all_indices = np.arange(n_points, dtype=int)

    if n_points <= THREE_D_MAX_POINTS_PER_SLICE:
        return all_indices

    if selected_domains is None:
        return _sample_indices(
            all_indices,
            THREE_D_MAX_POINTS_PER_SLICE,
            seed,
        )

    selected_mask = np.isin(
        record.domains,
        np.asarray(selected_domains, dtype=int),
    )
    foreground = np.flatnonzero(selected_mask)
    background = np.flatnonzero(~selected_mask)

    foreground_target = int(
        round(THREE_D_MAX_POINTS_PER_SLICE * 0.78)
    )
    foreground_indices = _sample_indices(
        foreground,
        min(foreground_target, len(foreground)),
        seed,
    )

    remaining = THREE_D_MAX_POINTS_PER_SLICE - len(
        foreground_indices
    )
    background_indices = _sample_indices(
        background,
        min(remaining, len(background)),
        seed + 1,
    )

    selected = np.concatenate(
        [foreground_indices, background_indices]
    )

    # Fill unused capacity when one group is small.
    if len(selected) < THREE_D_MAX_POINTS_PER_SLICE:
        unused = np.setdiff1d(
            all_indices,
            selected,
            assume_unique=False,
        )
        fill = _sample_indices(
            unused,
            min(
                THREE_D_MAX_POINTS_PER_SLICE - len(selected),
                len(unused),
            ),
            seed + 2,
        )
        selected = np.concatenate([selected, fill])

    return np.sort(selected)


def _shared_gene_vmax(
    records: Sequence[SliceRecord],
    percentile: float = 99.0,
) -> float:
    """Compute one gene-expression scale shared by every 3D time layer."""
    values = np.concatenate(
        [
            np.asarray(record.gene_values, dtype=float)
            for record in records
        ]
    )
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 1.0

    return max(
        float(np.percentile(values, percentile)),
        1e-8,
    )


def _gene_colors_3d(
    values: np.ndarray,
    vmax: float,
    alpha_factor: float,
) -> np.ndarray:
    """Create Reds colors with expression-dependent opacity."""
    values = np.asarray(values, dtype=float)
    normalized = np.clip(values / vmax, 0.0, 1.0)

    colors = plt.cm.Reds(normalized)
    expression_alpha = (
        THREE_D_GENE_ALPHA_MIN
        + (
            THREE_D_GENE_ALPHA_MAX
            - THREE_D_GENE_ALPHA_MIN
        )
        * normalized
    )
    colors[:, 3] = np.clip(
        expression_alpha * alpha_factor,
        0.02,
        1.0,
    )
    return colors


def _draw_3d_frame(
    ax,
    age_value: float,
    extent: Tuple[float, float, float, float],
    is_real: bool,
    offset: Tuple[float, float] = (0.0, 0.0),
    linewidth: float = 1.3,
) -> None:
    """
    Draw a solid real-slice frame or dashed virtual-slice frame.

    3D orientation
    --------------
    x-axis : spatial x
    y-axis : age / temporal depth
    z-axis : spatial y
    """
    xmin, xmax, ymin, ymax = extent
    dx, dy = offset

    x = np.asarray(
        [xmin, xmax, xmax, xmin, xmin],
        dtype=float,
    ) + dx
    y_depth = np.full(5, float(age_value))
    z_spatial = np.asarray(
        [ymin, ymin, ymax, ymax, ymin],
        dtype=float,
    ) + dy

    ax.plot(
        x,
        y_depth,
        z_spatial,
        color=FRAME_COLOR,
        linewidth=linewidth,
        linestyle="-" if is_real else DASHED_STYLE,
        alpha=0.95,
        zorder=20,
    )


def _draw_3d_mirror_seam(
    ax,
    age_value: float,
    offset: Tuple[float, float] = (0.0, 0.0),
    is_real: bool = True,
) -> None:
    """Draw the middle fold line in a mirrored 3D layer."""
    _, _, ymin, ymax = get_mirrored_extent()
    seam_x = get_mirrored_parameters()[4] + offset[0]

    ax.plot(
        [seam_x, seam_x],
        [float(age_value), float(age_value)],
        [ymin + offset[1], ymax + offset[1]],
        color=FRAME_COLOR,
        linewidth=0.8,
        linestyle="-" if is_real else DASHED_STYLE,
        alpha=0.9,
        zorder=19,
    )


def _style_3d_axis(
    ax,
    ages: Sequence[float],
    title: str,
    mirrored: bool = False,
) -> None:
    """
    Apply a consistent camera, aspect, ticks, and transparent panes.

    3D orientation
    --------------
    x-axis : spatial x
    y-axis : age / temporal depth (front to back)
    z-axis : spatial y
    """
    if mirrored:
        xmin, xmax, ymin, ymax = get_mirrored_extent()
        ax.set_box_aspect(THREE_D_MIRROR_BOX_ASPECT)
    else:
        xmin, xmax, ymin, ymax = FIXED_BOX
        ax.set_box_aspect(THREE_D_BOX_ASPECT)

    ages = np.asarray(sorted(ages), dtype=float)
    age_pad = max(
        0.8,
        0.06 * (ages.max() - ages.min()),
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(
        float(ages.min() - age_pad),
        float(ages.max() + age_pad),
    )
    ax.set_zlim(ymin, ymax)

    ax.set_xticks([xmin, 0.5 * (xmin + xmax), xmax])
    ax.set_yticks(ages)
    ax.set_yticklabels([f"{age:g}m" for age in ages])
    ax.set_zticks([ymin, 0.5 * (ymin + ymax), ymax])

    ax.set_xlabel("Spatial x", labelpad=7)
    ax.set_ylabel("Age", labelpad=8)
    ax.set_zlabel("Spatial y", labelpad=7)
    ax.set_title(title, pad=14)

    ax.view_init(
        elev=THREE_D_ELEV,
        azim=THREE_D_AZIM,
    )
    ax.set_proj_type("ortho")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((0.75, 0.75, 0.75, 0.55))

    ax.grid(True, alpha=0.13)


def _draw_optional_3d_corner_guides(
    ax,
    ages: Sequence[float],
    extent: Tuple[float, float, float, float],
) -> None:
    """Optionally connect corresponding frame corners through time."""
    if not THREE_D_CONNECT_CORNERS:
        return

    xmin, xmax, ymin, ymax = extent
    age_min = min(ages)
    age_max = max(ages)

    for x, z in [
        (xmin, ymin),
        (xmin, ymax),
        (xmax, ymin),
        (xmax, ymax),
    ]:
        ax.plot(
            [x, x],
            [age_min, age_max],
            [z, z],
            color=FRAME_COLOR,
            linewidth=0.55,
            alpha=0.35,
            zorder=0,
        )


def plot_figure6_3d_time_stack(
    ax,
    ages: Sequence[float],
    extent: Tuple[float, float, float, float],
) -> None:
    """Optionally connect corresponding frame corners through time."""
    if not THREE_D_CONNECT_CORNERS:
        return

    xmin, xmax, ymin, ymax = extent
    zmin = min(ages)
    zmax = max(ages)

    for x, y in [
        (xmin, ymin),
        (xmin, ymax),
        (xmax, ymin),
        (xmax, ymax),
    ]:
        ax.plot(
            [x, x],
            [y, y],
            [zmin, zmax],
            color=FRAME_COLOR,
            linewidth=0.55,
            alpha=0.35,
            zorder=0,
        )


def plot_figure6_3d_time_stack(
    records: Sequence[SliceRecord],
    outdir: str,
) -> None:
    """
    Plot separate 3D temporal stacks for gene expression and spatial domains.

    Each slice lies in its original spatial plane, with z equal to age in months.
    Real slices use solid frames; virtual slices use dashed frames.
    """
    set_style()
    ordered = sorted(records, key=lambda record: record.age)
    shared_vmax = _shared_gene_vmax(ordered)

    fig = plt.figure(figsize=(17.2, 8.2))
    ax_gene = fig.add_subplot(1, 2, 1, projection="3d")
    ax_domain = fig.add_subplot(1, 2, 2, projection="3d")

    for layer_index, record in enumerate(ordered):
        selected_domains = (
            HIGHLIGHT_118
            if (
                not record.is_real
                and abs(record.age - 11.8) < 1e-8
            )
            else None
        )

        indices = _select_3d_indices(
            record,
            selected_domains=selected_domains,
            seed=1000 + layer_index,
        )

        coords = np.asarray(record.coords, dtype=float)[indices]
        values = np.asarray(record.gene_values, dtype=float)[indices]
        domains = np.asarray(record.domains)[indices]
        age_depth = np.full(len(indices), record.age, dtype=float)

        alpha_factor = (
            REAL_ALPHA
            if record.is_real
            else VIRTUAL_ALPHA
        )

        # ---------------------------
        # Gene-expression stack
        # ---------------------------
        if selected_domains is not None:
            highlighted = np.isin(
                domains,
                np.asarray(selected_domains, dtype=int),
            )
            background = ~highlighted

            if np.any(background):
                ax_gene.scatter(
                    coords[background, 0],
                    age_depth[background],
                    coords[background, 1],
                    c=BACKGROUND_GRAY,
                    s=THREE_D_POINT_SIZE,
                    alpha=THREE_D_BACKGROUND_ALPHA,
                    linewidths=0,
                    depthshade=False,
                    rasterized=True,
                )

            gene_coords = coords[highlighted]
            gene_values = values[highlighted]
            gene_depth = age_depth[highlighted]
        else:
            gene_coords = coords
            gene_values = values
            gene_depth = age_depth

        gene_colors = _gene_colors_3d(
            gene_values,
            vmax=shared_vmax,
            alpha_factor=alpha_factor,
        )
        ax_gene.scatter(
            gene_coords[:, 0],
            gene_depth,
            gene_coords[:, 1],
            c=gene_colors,
            s=THREE_D_POINT_SIZE,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )

        # ---------------------------
        # Domain stack
        # ---------------------------
        domain_colors = domain_facecolors(
            domains,
            highlight_domains=selected_domains,
        )
        ax_domain.scatter(
            coords[:, 0],
            age_depth,
            coords[:, 1],
            c=domain_colors,
            s=THREE_D_POINT_SIZE,
            alpha=THREE_D_DOMAIN_ALPHA,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )

        for ax in (ax_gene, ax_domain):
            _draw_3d_frame(
                ax,
                age_value=record.age,
                extent=FIXED_BOX,
                is_real=record.is_real,
            )

            # Add the age label near the front-left corner.
            ax.text(
                FIXED_BOX[0],
                record.age,
                FIXED_BOX[2] - 35.0,
                record.label,
                fontsize=8.5,
                fontweight="bold",
                color=TEXT_COLOR,
            )

    ages = [record.age for record in ordered]
    _style_3d_axis(
        ax_gene,
        ages,
        title=f"{GENE_NAME} expression through time (front-to-back age depth)",
    )
    _style_3d_axis(
        ax_domain,
        ages,
        title="Spatial domains through time (front-to-back age depth)",
    )

    for ax in (ax_gene, ax_domain):
        _draw_optional_3d_corner_guides(
            ax,
            ages,
            FIXED_BOX,
        )

    # Shared gene-expression colorbar.
    scalar_mappable = ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=shared_vmax),
        cmap="Reds",
    )
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax_gene,
        fraction=0.025,
        pad=0.06,
        shrink=0.68,
    )
    colorbar.set_label(
        f"log1p {GENE_NAME} expression",
        fontsize=9.5,
    )
    colorbar.ax.tick_params(labelsize=8.5)

    # Real/virtual line-style legend.
    frame_handles = [
        Line2D(
            [0],
            [0],
            color=FRAME_COLOR,
            linewidth=1.8,
            linestyle="-",
            label="Observed",
        ),
        Line2D(
            [0],
            [0],
            color=FRAME_COLOR,
            linewidth=1.8,
            linestyle=DASHED_STYLE,
            label="Virtual",
        ),
    ]
    fig.legend(
        handles=frame_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
    )

    # Compact all-domain legend outside the right subplot.
    domain_handles = make_domain_legend_handles(
        labels=ALL_LABELS,
        include_other=True,
        other_label="Other at 11.8m",
    )
    ax_domain.legend(
        handles=domain_handles,
        loc="center left",
        bbox_to_anchor=(1.03, 0.50),
        frameon=False,
        fontsize=7.8,
        ncol=2,
        title="Domains",
        title_fontsize=9.0,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    fig.suptitle(
        "3D temporal stack of observed and virtual mouse-brain slices",
        y=0.995,
        fontsize=16,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.84,
        bottom=0.03,
        top=0.91,
        wspace=0.02,
    )

    save_path = os.path.join(
        outdir,
        "figure6_3d_gene_domain_time_stack.png",
    )
    finalize_figure(fig, save_path)


def plot_figure7_3d_mirrored_time_stack(
    records: Sequence[SliceRecord],
    outdir: str,
) -> None:
    """
    Plot the mirrored gene/domain composition as separated 3D temporal planes.

    This is the direct 3D alternative to the crowded 2D mirrored stack.
    """
    set_style()
    ordered = sorted(records, key=lambda record: record.age)
    shared_vmax = _shared_gene_vmax(ordered)

    fig = plt.figure(figsize=(12.8, 8.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for layer_index, record in enumerate(ordered):
        selected_domains = (
            HIGHLIGHT_118
            if (
                not record.is_real
                and abs(record.age - 11.8) < 1e-8
            )
            else None
        )

        gene_xy, domain_xy, retained_mask = mirrored_half_coords(
            record,
            offset=(0.0, 0.0),
        )

        retained_indices = np.flatnonzero(retained_mask)
        retained_record = SliceRecord(
            age=record.age,
            label=record.label,
            path=record.path,
            is_real=record.is_real,
            coords=record.coords[retained_mask],
            gene_values=record.gene_values[retained_mask],
            domains=record.domains[retained_mask],
        )

        local_indices = _select_3d_indices(
            retained_record,
            selected_domains=selected_domains,
            seed=2000 + layer_index,
        )

        gene_xy = gene_xy[local_indices]
        domain_xy = domain_xy[local_indices]
        values = retained_record.gene_values[local_indices]
        domains = retained_record.domains[local_indices]
        age_depth = np.full(len(local_indices), record.age, dtype=float)

        alpha_factor = (
            REAL_ALPHA
            if record.is_real
            else VIRTUAL_ALPHA
        )

        # Gene half.
        if selected_domains is not None:
            highlighted = np.isin(
                domains,
                np.asarray(selected_domains, dtype=int),
            )
            background = ~highlighted

            if np.any(background):
                ax.scatter(
                    gene_xy[background, 0],
                    age_depth[background],
                    gene_xy[background, 1],
                    c=BACKGROUND_GRAY,
                    s=THREE_D_POINT_SIZE,
                    alpha=THREE_D_BACKGROUND_ALPHA,
                    linewidths=0,
                    depthshade=False,
                    rasterized=True,
                )

            gene_points = gene_xy[highlighted]
            gene_values = values[highlighted]
            gene_depth = age_depth[highlighted]
        else:
            gene_points = gene_xy
            gene_values = values
            gene_depth = age_depth

        gene_colors = _gene_colors_3d(
            gene_values,
            vmax=shared_vmax,
            alpha_factor=alpha_factor,
        )
        ax.scatter(
            gene_points[:, 0],
            gene_depth,
            gene_points[:, 1],
            c=gene_colors,
            s=THREE_D_POINT_SIZE,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )

        # Mirrored domain half.
        domain_colors = domain_facecolors(
            domains,
            highlight_domains=selected_domains,
        )
        ax.scatter(
            domain_xy[:, 0],
            age_depth,
            domain_xy[:, 1],
            c=domain_colors,
            s=THREE_D_POINT_SIZE,
            alpha=THREE_D_DOMAIN_ALPHA,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )

        _draw_3d_frame(
            ax,
            age_value=record.age,
            extent=get_mirrored_extent(),
            is_real=record.is_real,
        )
        if DRAW_MIRROR_DIVIDER:
            _draw_3d_mirror_seam(
                ax,
                age_value=record.age,
                is_real=record.is_real,
            )

        ax.text(
            get_mirrored_extent()[0],
            record.age,
            get_mirrored_extent()[2] - 35.0,
            record.label,
            fontsize=8.5,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    ages = [record.age for record in ordered]
    _style_3d_axis(
        ax,
        ages,
        title=(
            f"Mirrored 3D temporal stack: "
            f"{GENE_NAME} gene map (left) and domains (right)"
        ),
        mirrored=True,
    )

    _draw_optional_3d_corner_guides(
        ax,
        ages,
        get_mirrored_extent(),
    )

    scalar_mappable = ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=shared_vmax),
        cmap="Reds",
    )
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        fraction=0.025,
        pad=0.045,
        shrink=0.62,
    )
    colorbar.set_label(
        f"log1p {GENE_NAME} expression",
        fontsize=9.5,
    )
    colorbar.ax.tick_params(labelsize=8.5)

    frame_handles = [
        Line2D(
            [0],
            [0],
            color=FRAME_COLOR,
            linewidth=1.8,
            linestyle="-",
            label="Observed",
        ),
        Line2D(
            [0],
            [0],
            color=FRAME_COLOR,
            linewidth=1.8,
            linestyle=DASHED_STYLE,
            label="Virtual",
        ),
    ]
    ax.legend(
        handles=frame_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=False,
    )

    fig.tight_layout()
    save_path = os.path.join(
        outdir,
        "figure7_3d_mirrored_time_stack.png",
    )
    finalize_figure(fig, save_path)


# ============================================================
# RUN ALL
# ============================================================
def run_all() -> None:
    ensure_dir(OUTDIR)
    real_records, virtual_records = load_all_slices()
    all_records = sorted(real_records + virtual_records, key=lambda r: r.age)

    # plot_figure1_real_stacks(real_records, OUTDIR)
    # plot_figure2_all_stacks(all_records, OUTDIR)
    # plot_figure3_mirrored_stack(all_records, OUTDIR)

    # plot_mirrored_grid(
    #     records=real_records,
    #     save_path=os.path.join(OUTDIR, "figure4_real_slices_mirrored_grid.png"),
    #     title="Observed real slices | Mirrored gene and domain views",
    # )

    per_panel_selected_domains = {
        "4.1m": None,
        "11.8m": HIGHLIGHT_118,
    }
    per_panel_legend = {
        "4.1m": {
            "labels": ALL_LABELS,
            "include_other": False,
            "title": "All domains",
        },
        "11.8m": {
            "groups": [
                {
                    "name": "Cortex",
                    "labels": [1, 4, 6, 9,10],
                },
                {
                    "name": "Hippocampus",
                    "labels": [8, 11],
                },
                {
                    "name": "Thalamus",
                    "labels": [0],
                },
                {
                    "name": "Hypothalamus",
                    "labels": [3],
                },

            ],
            "include_other": True,
            "other_label": "Other domains",
            "title": None,
        },
    }
    plot_mirrored_grid(
        records=virtual_records,
        save_path=os.path.join(OUTDIR, "figure5_virtual_slices_mirrored_grid.png"),
        title="Virtual slices | Mirrored gene and domain views",
        per_panel_selected_domains=per_panel_selected_domains,
        per_panel_legend=per_panel_legend,
    )

    if GENERATE_3D_SEPARATE_STACK:
        plot_figure6_3d_time_stack(
            records=all_records,
            outdir=OUTDIR,
        )

    if GENERATE_3D_MIRRORED_STACK:
        plot_figure7_3d_mirrored_time_stack(
            records=all_records,
            outdir=OUTDIR,
        )

    print(f"[DONE] Figures saved to: {OUTDIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mouse-time pipeline figures."
    )
    parser.add_argument(
        "--figure-mode",
        choices=["save", "show", "both"],
        default=FIGURE_MODE,
        help=(
            "Figure handling: 'save' writes files only; "
            "'show' displays figures only; "
            "'both' writes and displays every figure."
        ),
    )
    args = parser.parse_args()

    globals()["FIGURE_MODE"] = args.figure_mode
    print(f"[INFO] Figure mode: {FIGURE_MODE}")

    run_all()


if __name__ == "__main__":
    main()
