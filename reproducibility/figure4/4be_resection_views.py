#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone DLPFC 3D virtual-tissue / in-silico resectioning visualization.

The script is self-contained: it recycles the relevant loading, color scaling,
3D scatter, frame, and resection logic from the original wrapper and
visualization_utils.py, but does not import visualization_utils.py.

Design
------
For each reconstruction method (AXIS and SpatialZ), the script can generate:
    1) an un-sectioned 3D spatial-domain volume;
    2) an un-sectioned 3D marker-gene volume;
    3) spatial-domain resection figures using the domain-specific cut geometry;
    4) marker-gene resection figures using the gene-specific cut geometry.

The resection plane is intentionally restricted to one of the two vertical
standing planes of the plotted 3D coordinate system:

    X = biological depth (z_um * DEPTH_SCALE)
    Y = tissue coordinate 0 (adata.obsm['spatial'][:, 0])
    Z = tissue coordinate 1 (adata.obsm['spatial'][:, 1])

    base plane 'XZ' -> constant Y = cut location
                         (plane spans depth and tissue-coordinate 1)
    base plane 'YZ' -> constant X = cut location * DEPTH_SCALE
                         (plane is parallel to an original tissue slice)

Domain and gene resectioning have independent base-plane/location controls.
No arbitrary 3D rotation is used.

Output mode
-----------
OUTPUT_MODE = 'show'  : show each selected figure interactively, one by one.
OUTPUT_MODE = 'save'  : save and close each selected figure.
OUTPUT_MODE = 'both'  : save, then show.

Use FIGURES_TO_RUN to tune one figure at a time, for example:
    FIGURES_TO_RUN = ('axis_domain_cut_02',)
    FIGURES_TO_RUN = ('axis_gene_cut_02',)
Then adjust METHOD_CONFIGS['AXIS'].elev / azim or the shared style parameters.
Set FIGURES_TO_RUN = ('all',) to render all configured figures.

With the default settings, each method produces 8 figures:
2 un-sectioned views + 3 domain cuts + 3 gene cuts.
"""

from __future__ import annotations
from axis.settings import data_path
from axis.config import load_config

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import get_cmap  # matplotlib.cm.get_cmap was removed in matplotlib 3.9
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =============================================================================
# USER CONFIGURATION
# =============================================================================

SAVE_OR_SHOW = "save"  # "show", "save", or "both"
OUTPUT_MODE = SAVE_OR_SHOW
FIGURES_TO_RUN: Tuple[str, ...] = ("all",)
# Examples:
# FIGURES_TO_RUN = ("axis_full_domain",)
# FIGURES_TO_RUN = ("axis_full_gene",)
# FIGURES_TO_RUN = ("axis_cut_01", "spatialz_cut_01")

ROOT = load_config('dlpfc').root
GROUP_ID = 1
H5AD_DIR = os.path.join(ROOT, "h5ad")
OUTDIR = data_path("paper_analysis/Figure4_dlfpc/b_3d_views/")

# Measured anchor sections used in the original virtual-volume wrapper.
ORIG_Z_UM: Mapping[int, float] = {0: 0.0, 1: 10.0, 2: 310.0, 3: 320.0}
REAL_SAMPLE_IDS: Tuple[int, ...] = (1, 2)

# Features.
DOMAIN_KEY = "layer_guess_reordered"
MARKER_GENE = "PCP4"
USE_LOG1P = True
GEN_USE_VALID_MASK = True

# Preferred DLPFC domain order. Unexpected labels are appended naturally.
PREFERRED_DOMAIN_ORDER: Tuple[str, ...] = (
    "Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"
)
DOMAIN_CMAP_NAME = "tab10"
GENE_CMAP_NAME = "Reds"

# Robust continuous color scaling, kept from the original scripts.
VMAX_PCT_REAL = 0.995
VMAX_PCT_VIRTUAL = 0.995

# Tissue-coordinate display range and 3D convention.
X_LIM: Tuple[float, float] = (0.0, 512.0)  # tissue coord 0 -> plotted Y
Y_LIM: Tuple[float, float] = (0.0, 512.0)  # tissue coord 1 -> plotted Z
DEPTH_SCALE = 0.6
INVERT_X = True
INVERT_Y = True

# -----------------------------------------------------------------------------
# Feature-specific resection geometry.
#
# XZ: cut location is tissue coord-0, usually in [0, 512].
# YZ: cut location is biological depth in um, usually in [10, 310].
# -----------------------------------------------------------------------------
DOMAIN_CUT_BASE_PLANE = "XZ"
DOMAIN_CUT_LOCATIONS: Tuple[float, ...] = (120.0, 256.0, 390.0)

GENE_CUT_BASE_PLANE = "YZ"
GENE_CUT_LOCATIONS: Tuple[float, ...] = (50.0, 140.0, 210.0)

CUT_SLAB_HALF_WIDTH = 10.0
CUT_KEEP_POSITIVE_SIDE = True
SHOW_REMOVED_SIDE = False

# Which slice frames to retain in the figures.
SHOW_REAL_SLICE_FRAMES = False
SHOW_VIRTUAL_SLICE_FRAMES = False
SHOW_REAL_SLICE_LABELS = False
SHOW_VIRTUAL_SLICE_LABELS = False

SHOW_TITLE = False
SHOW_FRAME_LEGEND = False

# Save controls.
SAVE_FORMATS: Tuple[str, ...] = ("png",)
PNG_DPI = 350


@dataclass(frozen=True)
class MethodConfig:
    name: str
    virtual_dir: str
    elev: float
    azim: float
    s_real: float
    s_virtual: float
    alpha_min_real: float
    alpha_min_virtual: float
    alpha_max_real: float
    alpha_max_virtual: float
    alpha_gamma: float = 10.6


METHOD_CONFIGS: Mapping[str, MethodConfig] = {
    # AXIS values follow the original get_3b_paras() visualization settings.
    "AXIS": MethodConfig(
        name="AXIS",
        virtual_dir=os.path.join(
            ROOT,
            "virtual_tissue",
            f"dlpfc_z_group{GROUP_ID}_grid80_spatial1",
            "query/full_depth",
        ),
        elev=15.0,
        azim=145.0,
        s_real=5.0,
        s_virtual=5.0,
        alpha_min_real=0.0,
        alpha_min_virtual=0.3,
        alpha_max_real=1.0,
        alpha_max_virtual=1.0,
    ),
    # SpatialZ values follow the original get_3a2_paras() settings.
    "SpatialZ": MethodConfig(
        name="SpatialZ",
        virtual_dir=os.path.join(
            ROOT,
            "virtual_tissue",
            "spatialz",
            "dense_reconstruction",
            f"group{GROUP_ID}",
        ),
        elev=15.0,
        azim=145.0,
        s_real=5.0,
        s_virtual=5.0,
        alpha_min_real=0.0,
        alpha_min_virtual=0.3,
        alpha_max_real=1.0,
        alpha_max_virtual=1.0,
    ),
}


@dataclass(frozen=True)
class VolumeStyle:
    figsize: Tuple[float, float] = (7.0, 7.0)
    axes_position: Tuple[float, float, float, float] = (0.03, 0.08, 0.80, 0.84)

    # One geometry parameter is shared by full-volume and resection views so
    # the apparent 3D volume thickness stays identical across all figures.
    depth_exaggeration: float = 2.5
    depth_pad_ratio: float = 0.02
    projection: str = "ortho"

    real_frame_color: str = "black"
    virtual_frame_color: str = "0.35"
    real_frame_style: str = "-"
    virtual_frame_style: str = "--"
    real_frame_width: float = 1.4
    virtual_frame_width: float = 1.4
    virtual_frame_alpha: float = 0.70

    # Cutaway appearance follows the original fancy resectioning function.
    cut_frame_color: str = "crimson"
    cut_frame_width: float = 2.2
    cut_frame_alpha: float = 0.95
    cut_face_alpha: float = 0.0
    alpha_volume_real: float = 0.10
    alpha_plane_real: float = 0.95
    alpha_removed_real: float = 0.03
    alpha_volume_virtual: float = 0.10
    alpha_plane_virtual: float = 0.70
    alpha_removed_virtual: float = 0.02
    plane_point_scale: float = 1.35
    volume_point_scale: float = 0.75

    title_fontsize: float = 14.0
    title_y: float = 0.97

    # Domain legend: dot marker followed by domain name.
    # bbox_to_anchor = (x, y) in figure coordinates.
    domain_legend_bbox: Tuple[float, float] = (0.8, 0.50)
    domain_legend_marker_size: float = 7.5
    domain_legend_fontsize: float = 16.0
    domain_legend_title_fontsize: float = 18.0
    domain_legend_labelspacing: float = 0.55

    # One shared gene-expression colorbar for measured + virtual tissue.
    # Rectangle = (left, bottom, width, height) in figure coordinates.
    gene_cbar_rect: Tuple[float, float, float, float] = (0.8, 0.3, 0.03, 0.38)
    gene_cbar_label_fontsize: float = 18
    gene_cbar_tick_fontsize: float = 16


STYLE = VolumeStyle()


# =============================================================================
# DATA STRUCTURES AND HELPERS
# =============================================================================


@dataclass
class SlicePanel:
    title: str
    coords: np.ndarray
    z: float
    gene_vals: np.ndarray
    domain_labels: np.ndarray
    domain_vals: Optional[np.ndarray] = None


def natural_sort_key(value: object) -> List[object]:
    text = str(value)
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def parse_z_from_filename(path: str) -> float:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    token = stem.split("_")[-1]
    return float(token)


def get_valid_mask(adata) -> np.ndarray:
    if (not GEN_USE_VALID_MASK) or ("valid" not in adata.obs.columns):
        return np.ones(adata.n_obs, dtype=bool)
    return np.asarray(adata.obs["valid"].values, dtype=bool)


def extract_gene_vals(adata, gene: str) -> np.ndarray:
    if gene not in adata.var_names:
        raise ValueError(f"Gene {gene!r} not found in var_names.")
    j = int(np.where(np.asarray(adata.var_names) == gene)[0][0])
    values = adata.X[:, j]
    if sp.issparse(values):
        values = np.asarray(values.toarray()).ravel().astype(np.float32)
    else:
        values = np.asarray(values).ravel().astype(np.float32)
    if USE_LOG1P:
        values = np.log1p(np.clip(values, 0.0, None)).astype(np.float32)
    return values


def extract_domain_labels(adata, obs_key: str) -> np.ndarray:
    if obs_key not in adata.obs.columns:
        raise ValueError(f"obs key {obs_key!r} not found in adata.obs.")
    series = pd.Series(adata.obs[obs_key])
    out = np.empty(len(series), dtype=object)
    for i, value in enumerate(series):
        out[i] = None if pd.isna(value) else str(value)
    return out


def robust_vrange(
    values_list: Sequence[np.ndarray],
    low_pct: float = 0.005,
    high_pct: float = 0.995,
) -> Tuple[float, float]:
    finite_parts: List[np.ndarray] = []
    for values in values_list:
        arr = np.asarray(values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_parts.append(arr)
    if not finite_parts:
        return 0.0, 1.0
    all_values = np.concatenate(finite_parts)
    vmin = float(np.quantile(all_values, low_pct))
    vmax = float(np.quantile(all_values, high_pct))
    if vmin >= vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def load_one_panel(path: str, z: float, title: str, use_valid_mask: bool) -> SlicePanel:
    adata = sc.read_h5ad(path)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    gene_vals = extract_gene_vals(adata, MARKER_GENE)
    domain_labels = extract_domain_labels(adata, DOMAIN_KEY)

    if use_valid_mask:
        mask = get_valid_mask(adata)
        coords = coords[mask]
        gene_vals = gene_vals[mask]
        domain_labels = domain_labels[mask]

    panel = SlicePanel(
        title=title,
        coords=coords,
        z=float(z),
        gene_vals=np.asarray(gene_vals, dtype=np.float32),
        domain_labels=np.asarray(domain_labels, dtype=object),
    )
    del adata
    return panel


def load_real_panels() -> List[SlicePanel]:
    panels: List[SlicePanel] = []
    for sample_id in REAL_SAMPLE_IDS:
        z = float(ORIG_Z_UM[sample_id])
        path = os.path.join(H5AD_DIR, f"{GROUP_ID}_{sample_id}.h5ad")
        print(f"[LOAD] Real z={z:g}: {path}")
        panels.append(load_one_panel(path, z, f"Real ({z:g}um)", use_valid_mask=False))
    return panels


def load_virtual_panels(method: MethodConfig) -> List[SlicePanel]:
    pattern = os.path.join(method.virtual_dir, "*.h5ad")
    files = sorted(glob.glob(pattern), key=parse_z_from_filename)
    if not files:
        raise FileNotFoundError(f"No H5AD files found for {method.name}: {pattern}")

    panels: List[SlicePanel] = []
    for path in files:
        z = float(parse_z_from_filename(path))
        print(f"[LOAD] {method.name} z={z:g}: {path}")
        panels.append(
            load_one_panel(
                path,
                z,
                f"{method.name} ({z:g}um)",
                use_valid_mask=True,
            )
        )
    return panels


def build_domain_mapping(panel_groups: Iterable[Sequence[SlicePanel]]) -> Tuple[Dict[str, int], List[str]]:
    labels: List[str] = []
    for panels in panel_groups:
        for panel in panels:
            for value in panel.domain_labels:
                if value is not None and str(value).lower() not in {"nan", "none", "<na>"}:
                    labels.append(str(value))

    unique = set(labels)
    preferred = [label for label in PREFERRED_DOMAIN_ORDER if label in unique]
    extras = sorted([label for label in unique if label not in preferred], key=natural_sort_key)
    ordered = preferred + extras
    if not ordered:
        raise ValueError(f"No valid labels found for DOMAIN_KEY={DOMAIN_KEY!r}.")
    if len(ordered) > 10:
        raise ValueError(
            f"DOMAIN_CMAP_NAME='tab10' supports at most 10 stable discrete colors; found {len(ordered)} labels."
        )
    return {label: i for i, label in enumerate(ordered)}, ordered


def apply_domain_mapping(panels: Sequence[SlicePanel], mapping: Mapping[str, int]) -> None:
    for panel in panels:
        values = np.full(len(panel.domain_labels), np.nan, dtype=np.float32)
        for i, label in enumerate(panel.domain_labels):
            if label is None:
                continue
            key = str(label)
            if key in mapping:
                values[i] = float(mapping[key])
        panel.domain_vals = values


def make_domain_cmap(n_domains: int) -> ListedColormap:
    base = get_cmap(DOMAIN_CMAP_NAME)
    colors = [base(i) for i in range(n_domains)]
    return ListedColormap(colors, name=f"{DOMAIN_CMAP_NAME}_{n_domains}")


def feature_values(panel: SlicePanel, feature: str) -> np.ndarray:
    feature = str(feature).lower()
    if feature == "gene":
        return panel.gene_vals
    if feature == "domain":
        if panel.domain_vals is None:
            raise RuntimeError("Domain values have not been encoded.")
        return panel.domain_vals
    raise ValueError("feature must be 'gene' or 'domain'.")


# =============================================================================
# COLOR / SCATTER HELPERS
# =============================================================================


def make_gene_rgba(
    values: np.ndarray,
    norm: Normalize,
    cmap_name: str,
    alpha_min: float,
    alpha_max: float,
    gamma: float,
) -> np.ndarray:
    """Reproduce the value-dependent transparency used in visualization_utils.py."""
    values = np.asarray(values, dtype=np.float32)
    frac = np.clip(norm(values), 0.0, 1.0)
    rgba = np.asarray(get_cmap(cmap_name)(frac), dtype=np.float32)

    abs_values = np.abs(values)
    abs_max = float(np.max(abs_values)) if abs_values.size else 1.0
    if abs_max < 1e-12:
        alpha_frac = np.zeros_like(abs_values, dtype=np.float32)
    else:
        alpha_frac = abs_values / abs_max
    alpha = alpha_min + (alpha_frac ** gamma) * (alpha_max - alpha_min)
    rgba[:, 3] = np.clip(alpha, 0.0, 1.0)
    return rgba


def make_fixed_alpha_rgba(
    values: np.ndarray,
    feature: str,
    cmap,
    norm: Optional[Normalize],
    alpha: float,
) -> np.ndarray:
    values = np.asarray(values)
    if feature == "domain":
        rgba = np.asarray(cmap(values.astype(int)), dtype=np.float32)
    else:
        rgba = np.asarray(cmap(norm(values)), dtype=np.float32)
    if rgba.ndim == 1:
        rgba = rgba[None, :]
    rgba[:, 3] = float(alpha)
    return rgba


def scatter_full_panel(
    ax,
    panel: SlicePanel,
    feature: str,
    norm: Optional[Normalize],
    cmap,
    point_size: float,
    alpha_min: float,
    alpha_max: float,
    alpha_gamma: float,
) -> None:
    values = feature_values(panel, feature)
    valid = np.isfinite(values) & np.all(np.isfinite(panel.coords), axis=1)
    if not np.any(valid):
        return
    coords = panel.coords[valid]
    values = values[valid]

    if feature == "gene":
        rgba = make_gene_rgba(
            values,
            norm=norm,
            cmap_name=GENE_CMAP_NAME,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            gamma=alpha_gamma,
        )
    else:
        rgba = make_fixed_alpha_rgba(values, feature, cmap, norm, alpha=alpha_max)

    depth_x = np.full(coords.shape[0], panel.z * DEPTH_SCALE, dtype=np.float32)
    ax.scatter(
        depth_x,
        coords[:, 0],
        coords[:, 1],
        c=rgba,
        s=point_size,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )


def split_by_axis_aligned_plane(
    panel: SlicePanel,
    base_plane: str,
    location: float,
    slab_half_width: float,
    keep_positive_side: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return kept-volume, near-plane, and removed-side masks.

    XZ plane: Y is constant -> signed distance is coords[:,0] - location.
    YZ plane: X is constant -> signed distance is panel.z - location in um.
    """
    base_plane = str(base_plane).upper()
    if base_plane == "XZ":
        signed = panel.coords[:, 0].astype(np.float32) - float(location)
    elif base_plane == "YZ":
        signed = np.full(panel.coords.shape[0], panel.z - float(location), dtype=np.float32)
    else:
        raise ValueError("CUT_BASE_PLANE must be 'XZ' or 'YZ'.")

    plane_mask = np.abs(signed) <= float(slab_half_width)
    if keep_positive_side:
        keep_mask = signed > float(slab_half_width)
        remove_mask = signed < -float(slab_half_width)
    else:
        keep_mask = signed < -float(slab_half_width)
        remove_mask = signed > float(slab_half_width)
    return keep_mask, plane_mask, remove_mask


def scatter_resection_panel(
    ax,
    panel: SlicePanel,
    feature: str,
    norm: Optional[Normalize],
    cmap,
    base_plane: str,
    location: float,
    slab_half_width: float,
    keep_positive_side: bool,
    point_size_volume: float,
    point_size_plane: float,
    alpha_volume: float,
    alpha_plane: float,
    show_removed_side: bool,
    alpha_removed: float,
) -> None:
    values = feature_values(panel, feature)
    valid = np.isfinite(values) & np.all(np.isfinite(panel.coords), axis=1)
    if not np.any(valid):
        return

    keep, plane, removed = split_by_axis_aligned_plane(
        panel,
        base_plane=base_plane,
        location=location,
        slab_half_width=slab_half_width,
        keep_positive_side=keep_positive_side,
    )
    keep &= valid
    plane &= valid
    removed &= valid

    x = np.full(panel.coords.shape[0], panel.z * DEPTH_SCALE, dtype=np.float32)
    y = panel.coords[:, 0]
    z = panel.coords[:, 1]

    def draw(mask: np.ndarray, size: float, alpha: float) -> None:
        if not np.any(mask):
            return
        rgba = make_fixed_alpha_rgba(values[mask], feature, cmap, norm, alpha=alpha)
        ax.scatter(
            x[mask], y[mask], z[mask],
            s=size,
            c=rgba,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )

    if show_removed_side:
        draw(removed, point_size_volume, alpha_removed)
    draw(keep, point_size_volume, alpha_volume)
    draw(plane, point_size_plane, alpha_plane)


# =============================================================================
# 3D GEOMETRY / LAYOUT
# =============================================================================


def volume_depth_limits(real_panels: Sequence[SlicePanel], virtual_panels: Sequence[SlicePanel]) -> Tuple[float, float]:
    depths = [panel.z * DEPTH_SCALE for panel in list(real_panels) + list(virtual_panels)]
    if not depths:
        raise ValueError("No panels available for depth limits.")
    lo, hi = float(min(depths)), float(max(depths))
    pad = STYLE.depth_pad_ratio * max(hi - lo, 1.0)
    return lo - pad, hi + pad


def setup_3d_axis(
    ax,
    real_panels: Sequence[SlicePanel],
    virtual_panels: Sequence[SlicePanel],
    method: MethodConfig,
) -> Tuple[float, float]:
    depth_lim = volume_depth_limits(real_panels, virtual_panels)
    x_range = depth_lim[1] - depth_lim[0]
    y_range = abs(X_LIM[1] - X_LIM[0])
    z_range = abs(Y_LIM[1] - Y_LIM[0])

    ax.set_box_aspect((x_range * STYLE.depth_exaggeration, y_range, z_range))
    ax.set_position(STYLE.axes_position)
    ax.set_proj_type(STYLE.projection)
    ax.set_xlim(*depth_lim)
    ax.set_ylim(X_LIM[::-1] if INVERT_X else X_LIM)
    ax.set_zlim(Y_LIM[::-1] if INVERT_Y else Y_LIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=method.elev, azim=method.azim)
    return depth_lim


def draw_slice_plane_frame(
    ax,
    panel: SlicePanel,
    linestyle: str,
    color: str,
    linewidth: float,
    alpha: float,
    label: Optional[str] = None,
    label_fontsize: float = 11.0,
) -> None:
    x_depth = float(panel.z) * DEPTH_SCALE
    y0, y1 = float(min(X_LIM)), float(max(X_LIM))
    z0, z1 = float(min(Y_LIM)), float(max(Y_LIM))

    xs = np.full(5, x_depth, dtype=np.float32)
    ys = np.array([y0, y1, y1, y0, y0], dtype=np.float32)
    zs = np.array([z0, z0, z1, z1, z0], dtype=np.float32)
    ax.plot(
        xs, ys, zs,
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=0,
    )
    if label:
        ax.text(
            x_depth,
            y0,
            z0 - 0.06 * (z1 - z0),
            label,
            color=color,
            fontsize=label_fontsize,
            ha="left",
            va="top",
            zorder=10,
        )


def draw_configured_slice_frames(ax, real_panels: Sequence[SlicePanel], virtual_panels: Sequence[SlicePanel]) -> None:
    if SHOW_REAL_SLICE_FRAMES:
        for panel in real_panels:
            draw_slice_plane_frame(
                ax,
                panel,
                linestyle=STYLE.real_frame_style,
                color=STYLE.real_frame_color,
                linewidth=STYLE.real_frame_width,
                alpha=1.0,
                label=panel.title if SHOW_REAL_SLICE_LABELS else None,
            )
    if SHOW_VIRTUAL_SLICE_FRAMES:
        for panel in virtual_panels:
            draw_slice_plane_frame(
                ax,
                panel,
                linestyle=STYLE.virtual_frame_style,
                color=STYLE.virtual_frame_color,
                linewidth=STYLE.virtual_frame_width,
                alpha=STYLE.virtual_frame_alpha,
                label=panel.title if SHOW_VIRTUAL_SLICE_LABELS else None,
            )


def draw_axis_aligned_cut_plane(
    ax,
    base_plane: str,
    location: float,
    depth_lim: Tuple[float, float],
) -> None:
    base_plane = str(base_plane).upper()
    x0, x1 = depth_lim
    y0, y1 = float(min(X_LIM)), float(max(X_LIM))
    z0, z1 = float(min(Y_LIM)), float(max(Y_LIM))

    if base_plane == "XZ":
        y = float(location)
        corners = np.asarray(
            [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]],
            dtype=np.float32,
        )
    elif base_plane == "YZ":
        x = float(location) * DEPTH_SCALE
        corners = np.asarray(
            [[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]],
            dtype=np.float32,
        )
    else:
        raise ValueError("CUT_BASE_PLANE must be 'XZ' or 'YZ'.")

    closed = np.vstack([corners, corners[0]])
    ax.plot(
        closed[:, 0], closed[:, 1], closed[:, 2],
        color=STYLE.cut_frame_color,
        linewidth=STYLE.cut_frame_width,
        alpha=STYLE.cut_frame_alpha,
        zorder=10,
    )
    face = Poly3DCollection([corners])
    face.set_facecolor(STYLE.cut_frame_color)
    face.set_alpha(STYLE.cut_face_alpha)
    face.set_edgecolor("none")
    ax.add_collection3d(face)


def maybe_add_frame_legend(ax) -> None:
    if not SHOW_FRAME_LEGEND:
        return
    handles = []
    if SHOW_REAL_SLICE_FRAMES:
        handles.append(
            Line2D(
                [0], [0],
                color=STYLE.real_frame_color,
                linestyle=STYLE.real_frame_style,
                linewidth=STYLE.real_frame_width,
                label="Measured anchor slices",
            )
        )
    if SHOW_VIRTUAL_SLICE_FRAMES:
        handles.append(
            Line2D(
                [0], [0],
                color=STYLE.virtual_frame_color,
                linestyle=STYLE.virtual_frame_style,
                linewidth=STYLE.virtual_frame_width,
                label="Virtual slices",
            )
        )
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False)


# =============================================================================
# LEGENDS / COLORBAR
# =============================================================================


def add_gene_colorbar(
    fig,
    norm: Normalize,
) -> None:
    """Add one shared gene-expression colorbar for real and virtual tissue."""
    sm = ScalarMappable(norm=norm, cmap=GENE_CMAP_NAME)
    sm.set_array([])

    cax = fig.add_axes(STYLE.gene_cbar_rect)
    cbar = fig.colorbar(sm, cax=cax)

    label = (
        f"{MARKER_GENE} expression (log1p)"
        if USE_LOG1P
        else f"{MARKER_GENE} expression"
    )
    cbar.set_label(
        label,
        rotation=90,
        fontsize=STYLE.gene_cbar_label_fontsize,
    )
    cbar.ax.tick_params(labelsize=STYLE.gene_cbar_tick_fontsize)


def add_domain_legend(
    fig,
    cmap: ListedColormap,
    domain_labels: Sequence[str],
) -> None:
    """Add a categorical point legend: colored dot followed by domain name."""
    handles: List[Line2D] = []
    for index, label in enumerate(domain_labels):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=cmap(index),
                markeredgecolor="none",
                markersize=STYLE.domain_legend_marker_size,
                label=str(label),
            )
        )

    legend = fig.legend(
        handles=handles,
        title="Cortical\n layer",
        loc="center left",
        bbox_to_anchor=STYLE.domain_legend_bbox,
        frameon=False,
        fontsize=STYLE.domain_legend_fontsize,
        labelspacing=STYLE.domain_legend_labelspacing,
        handlelength=1.0,
        handletextpad=0.55,
        borderaxespad=0.0,
    )
    legend.get_title().set_fontsize(STYLE.domain_legend_title_fontsize)


# =============================================================================
# FIGURE CREATION
# =============================================================================


def build_feature_norms(
    feature: str,
    real_panels: Sequence[SlicePanel],
    virtual_panels: Sequence[SlicePanel],
) -> Tuple[Optional[Normalize], Optional[Normalize]]:
    """Return feature normalization.

    Domain values are categorical and require no Normalize object.

    Gene values use ONE shared robust range across measured and virtual tissue.
    Returning the same Normalize instance twice keeps all scatter rendering
    compatible with a single fused gene-expression colorbar.
    """
    if feature == "domain":
        return None, None

    # Use a common percentile range over all measured + virtual expression.
    # The two percentile settings are retained for compatibility; with the
    # current configuration they are identical (0.995).
    high_q = min(float(VMAX_PCT_REAL), float(VMAX_PCT_VIRTUAL))
    low_q = 1.0 - high_q
    shared_vmin, shared_vmax = robust_vrange(
        [panel.gene_vals for panel in list(real_panels) + list(virtual_panels)],
        low_pct=low_q,
        high_pct=high_q,
    )
    shared_norm = Normalize(shared_vmin, shared_vmax)
    print(
        f"[SCALE] Shared 3D {MARKER_GENE}: "
        f"{shared_vmin:.4f} .. {shared_vmax:.4f}"
    )
    return shared_norm, shared_norm


def plot_full_volume(
    method: MethodConfig,
    real_panels: Sequence[SlicePanel],
    virtual_panels: Sequence[SlicePanel],
    feature: str,
    domain_cmap: ListedColormap,
    domain_labels: Sequence[str],
):
    feature = str(feature).lower()
    real_norm, virtual_norm = build_feature_norms(feature, real_panels, virtual_panels)
    cmap = domain_cmap if feature == "domain" else get_cmap(GENE_CMAP_NAME)

    fig = plt.figure(figsize=STYLE.figsize)
    ax = fig.add_subplot(111, projection="3d")
    setup_3d_axis(ax, real_panels, virtual_panels, method)

    for panel in real_panels:
        scatter_full_panel(
            ax,
            panel,
            feature=feature,
            norm=real_norm,
            cmap=cmap,
            point_size=method.s_real,
            alpha_min=method.alpha_min_real,
            alpha_max=method.alpha_max_real,
            alpha_gamma=method.alpha_gamma,
        )
    for panel in virtual_panels:
        scatter_full_panel(
            ax,
            panel,
            feature=feature,
            norm=virtual_norm,
            cmap=cmap,
            point_size=method.s_virtual,
            alpha_min=method.alpha_min_virtual,
            alpha_max=method.alpha_max_virtual,
            alpha_gamma=method.alpha_gamma,
        )

    # draw_configured_slice_frames(ax, real_panels, virtual_panels)
    maybe_add_frame_legend(ax)

    if SHOW_TITLE:
        feature_name = "spatial domains" if feature == "domain" else MARKER_GENE
        fig.suptitle(
            f"{method.name}: un-sectioned 3D {feature_name}",
            y=STYLE.title_y,
            fontsize=STYLE.title_fontsize,
        )

    if feature == "domain":
        add_domain_legend(fig, domain_cmap, domain_labels)
    else:
        add_gene_colorbar(fig, real_norm)
    return fig


def plot_resection_volume(
    method: MethodConfig,
    real_panels: Sequence[SlicePanel],
    virtual_panels: Sequence[SlicePanel],
    feature: str,
    base_plane: str,
    location: float,
    domain_cmap: ListedColormap,
    domain_labels: Sequence[str],
):
    feature = str(feature).lower()
    real_norm, virtual_norm = build_feature_norms(feature, real_panels, virtual_panels)
    cmap = domain_cmap if feature == "domain" else get_cmap(GENE_CMAP_NAME)

    fig = plt.figure(figsize=STYLE.figsize)
    ax = fig.add_subplot(111, projection="3d")
    depth_lim = setup_3d_axis(ax, real_panels, virtual_panels, method)

    for panel in real_panels:
        scatter_resection_panel(
            ax=ax,
            panel=panel,
            feature=feature,
            norm=real_norm,
            cmap=cmap,
            base_plane=base_plane,
            location=location,
            slab_half_width=CUT_SLAB_HALF_WIDTH,
            keep_positive_side=CUT_KEEP_POSITIVE_SIDE,
            point_size_volume=max(1.0, method.s_real * STYLE.volume_point_scale),
            point_size_plane=max(2.0, method.s_real * STYLE.plane_point_scale),
            alpha_volume=STYLE.alpha_volume_real,
            alpha_plane=STYLE.alpha_plane_real,
            show_removed_side=SHOW_REMOVED_SIDE,
            alpha_removed=STYLE.alpha_removed_real,
        )

    for panel in virtual_panels:
        scatter_resection_panel(
            ax=ax,
            panel=panel,
            feature=feature,
            norm=virtual_norm,
            cmap=cmap,
            base_plane=base_plane,
            location=location,
            slab_half_width=CUT_SLAB_HALF_WIDTH,
            keep_positive_side=CUT_KEEP_POSITIVE_SIDE,
            point_size_volume=max(1.0, method.s_virtual * STYLE.volume_point_scale),
            point_size_plane=max(2.0, method.s_virtual * STYLE.plane_point_scale),
            alpha_volume=STYLE.alpha_volume_virtual,
            alpha_plane=STYLE.alpha_plane_virtual,
            show_removed_side=SHOW_REMOVED_SIDE,
            alpha_removed=STYLE.alpha_removed_virtual,
        )

    draw_configured_slice_frames(ax, real_panels, virtual_panels)
    draw_axis_aligned_cut_plane(ax, base_plane, location, depth_lim)
    maybe_add_frame_legend(ax)

    if SHOW_TITLE:
        feature_name = "spatial domains" if feature == "domain" else MARKER_GENE
        loc_unit = "tissue-coordinate units" if base_plane.upper() == "XZ" else "um depth"
        fig.suptitle(
            f"{method.name}: {feature_name} resection | {base_plane.upper()} @ {location:g} {loc_unit}",
            y=STYLE.title_y,
            fontsize=STYLE.title_fontsize,
        )

    if feature == "domain":
        add_domain_legend(fig, domain_cmap, domain_labels)
    else:
        add_gene_colorbar(fig, real_norm)
    return fig


# =============================================================================
# SAVE / SHOW / JOB CONTROL
# =============================================================================


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_").lower()


def finalize_figure(fig, stem: Path) -> None:
    mode = OUTPUT_MODE.strip().lower()
    if mode not in {"show", "save", "both"}:
        raise ValueError("OUTPUT_MODE must be 'show', 'save', or 'both'.")

    if mode in {"save", "both"}:
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in SAVE_FORMATS:
            ext = str(ext).lower().lstrip(".")
            path = stem.with_suffix(f".{ext}")
            kwargs = dict(bbox_inches="tight", facecolor="white", pad_inches=0.04)
            if ext == "png":
                kwargs["dpi"] = PNG_DPI
            fig.savefig(path, **kwargs)
            print(f"[SAVE] {path}")

    if mode in {"show", "both"}:
        plt.show()
    plt.close(fig)


def requested(job_id: str) -> bool:
    selection = {str(x).strip().lower() for x in FIGURES_TO_RUN}
    return "all" in selection or job_id.lower() in selection


def method_prefix(method_name: str) -> str:
    return safe_name(method_name)


def render_method_group(
    method: MethodConfig,
    real_panels: Sequence[SlicePanel],
    virtual_panels: Sequence[SlicePanel],
    domain_cmap: ListedColormap,
    domain_labels: Sequence[str],
) -> None:
    prefix = method_prefix(method.name)

    job_id = f"{prefix}_full_domain"
    if requested(job_id):
        print(f"[FIG] {job_id}")
        fig = plot_full_volume(method, real_panels, virtual_panels, "domain", domain_cmap, domain_labels)
        finalize_figure(fig, Path(OUTDIR) / job_id)

    job_id = f"{prefix}_full_gene"
    if requested(job_id):
        print(f"[FIG] {job_id}")
        fig = plot_full_volume(method, real_panels, virtual_panels, "gene", domain_cmap, domain_labels)
        finalize_figure(fig, Path(OUTDIR) / job_id)

    # -------------------------------------------------------------
    # Domain resectioning: XZ plane at 120, 256, and 390.
    # -------------------------------------------------------------
    for index, location in enumerate(DOMAIN_CUT_LOCATIONS, start=1):
        job_id = f"{prefix}_domain_cut_{index:02d}"
        if not requested(job_id):
            continue
        print(
            f"[FIG] {job_id} | "
            f"{DOMAIN_CUT_BASE_PLANE} @ {location:g}"
        )
        fig = plot_resection_volume(
            method=method,
            real_panels=real_panels,
            virtual_panels=virtual_panels,
            feature="domain",
            base_plane=DOMAIN_CUT_BASE_PLANE,
            location=float(location),
            domain_cmap=domain_cmap,
            domain_labels=domain_labels,
        )
        finalize_figure(fig, Path(OUTDIR) / job_id)

    # -------------------------------------------------------------
    # Gene resectioning: YZ plane at 50, 120, and 210 um depth.
    # -------------------------------------------------------------
    for index, location in enumerate(GENE_CUT_LOCATIONS, start=1):
        job_id = f"{prefix}_gene_cut_{index:02d}"
        if not requested(job_id):
            continue
        print(
            f"[FIG] {job_id} | "
            f"{GENE_CUT_BASE_PLANE} @ {location:g}"
        )
        fig = plot_resection_volume(
            method=method,
            real_panels=real_panels,
            virtual_panels=virtual_panels,
            feature="gene",
            base_plane=GENE_CUT_BASE_PLANE,
            location=float(location),
            domain_cmap=domain_cmap,
            domain_labels=domain_labels,
        )
        finalize_figure(fig, Path(OUTDIR) / job_id)


def main() -> None:
    print("[INFO] Loading measured anchor slices once.")
    real_panels = load_real_panels()

    virtual_by_method: Dict[str, List[SlicePanel]] = {}
    for method_name, method in METHOD_CONFIGS.items():
        print(f"[INFO] Loading {method_name} virtual volume.")
        virtual_by_method[method_name] = load_virtual_panels(method)

    # One shared mapping guarantees that a cortical layer has the same color in
    # Real, AXIS, and SpatialZ figures.
    mapping, domain_labels = build_domain_mapping(
        [real_panels] + [virtual_by_method[name] for name in METHOD_CONFIGS]
    )
    apply_domain_mapping(real_panels, mapping)
    for panels in virtual_by_method.values():
        apply_domain_mapping(panels, mapping)
    domain_cmap = make_domain_cmap(len(domain_labels))

    print(f"[DOMAIN] {mapping}")
    per_method_count = (
        2
        + len(DOMAIN_CUT_LOCATIONS)
        + len(GENE_CUT_LOCATIONS)
    )
    literal_count = len(METHOD_CONFIGS) * per_method_count
    print(
        f"[INFO] Configured figure count = {literal_count} "
        f"({len(METHOD_CONFIGS)} methods x "
        f"[2 full + {len(DOMAIN_CUT_LOCATIONS)} domain cuts + "
        f"{len(GENE_CUT_LOCATIONS)} gene cuts])."
    )

    for method_name, method in METHOD_CONFIGS.items():
        render_method_group(
            method=method,
            real_panels=real_panels,
            virtual_panels=virtual_by_method[method_name],
            domain_cmap=domain_cmap,
            domain_labels=domain_labels,
        )


if __name__ == "__main__":
    main()
