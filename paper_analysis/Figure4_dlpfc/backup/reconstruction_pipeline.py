#!/usr/bin/env python3
"""
3D DLPFC dense-reconstruction illustration pipeline.

This script generates five figures in a consistent Cell/Nature-style 3D design.
It supports two visualization modes:
    - gene   : PCP4 expression in Reds
    - domain : layer/domain labels in a stable categorical palette

Figures
-------
Figure 1
    Full 3D stack of the four observed slices at the anatomical depths
    0, 10, 310, and 320 um.

Figure 2
    Two reserved anchors only. The anchor pair is configurable:
        - inner anchors: 10 and 310 um
        - outer anchors: 0 and 320 um

Figure 3
    Transition-field truncation figure.
    Two real anchors + 3–5 virtual slices + piecewise lifted transition fields.
    Virtual slices use dashed bounding boxes. Sequential field segments use
    a consistent hue with progressively lighter/darker shades.

Figure 4
    Core reconstruction/evidence-collection figure.
    Similar to the SCC Figure 3 style:
        - transparent real anchors
        - Visium template on each virtual plane
        - collected evidence on each virtual plane
        - lifted evidence-transport curves from both anchors

Figure 5
    Final reconstructed slices at their anatomical depths.
    Virtual slices use dashed bounding boxes.

Notes
-----
1. Data-loading for observed DLPFC slices and pairwise flows is implemented.
2. The exact paths for generated virtual slices and generation metadata can vary
   across experiments. If you are unsure, fill the placeholders:
       - GENERATED_VIRTUAL_PATHS
       - GENERATION_META_PATHS
   or set GENERATED_DIR / META_DIR with the templates below.
3. Figures 4 and 5 require generated-query metadata and/or generated virtual
   slices. If those files are not configured, the script will skip them with a
   clear message.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy.sparse as sp
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# External helper used in the previous DLPFC scripts.
# The import is only required at runtime when inverse warping is actually used.
from axis.coordinate_utils import find_inverse_warp_coords


# =============================================================================
# Defaults
# =============================================================================

ROOT = Path("/media/huifang/data/registration/DLPFC/huifang")
GROUP_ID = 1

OBSERVED_SAMPLE_IDS = [0, 1, 2, 3]
OBSERVED_DEPTHS_UM: Dict[int, float] = {
    0: 0.0,
    1: 10.0,
    2: 310.0,
    3: 320.0,
}

# Reserve either the inner or the outer anchor pair.
ANCHOR_PAIR_NAME = "inner"  # "inner" -> 10/310, "outer" -> 0/320
ANCHOR_PAIR_OPTIONS = {
    "inner": (1, 2),
    "outer": (0, 3),
}

DISPLAY_MODE = "domain"  # "gene" or "domain"
MARKER_GENE = "PCP4"
USE_LOG1P = True

DOMAIN_KEY = "layer_guess_reordered"
DOMAIN_PALETTE = "tab10"
PREFERRED_DOMAIN_ORDER = [
    "Layer1",
    "Layer2",
    "Layer3",
    "Layer4",
    "Layer5",
    "Layer6",
    "WM",
]
DLPFC_DOMAIN_COLORS = {
    "Layer1": "#7CC8E8",
    "Layer2": "#3F8FC5",
    "Layer3": "#4E79A7",
    "Layer4": "#F28E2B",
    "Layer5": "#3E9651",
    "Layer6": "#D95852",
    "WM": "#8F6FA6",
}

# Observed / flow locations
H5AD_SUBDIR = "h5ad"
H5AD_TEMPLATE = "{group_id}_{sample_id}.h5ad"
NETWORK_SUBDIR = "network_input"
NETWORK_TEMPLATE = "{group_id}_{sample_id}_validation.npz"
PAIRWISE_SUBDIR = "pairwise_output"
FLOW_TEMPLATE = "{fixed_id}_{moving_id}_flow.npy"
REAL_COORD_KEY = "spatial"

# Generated virtual slices and generation metadata.
# Fill these if you know the exact files. Keys are the virtual depths (um).
GENERATED_VIRTUAL_PATHS: Dict[float, str] = {
    80.0:  "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/dlpfc_z_group1_grid80_spatial1/query/virtual_dlpfc_z_80.000.h5ad",
    160.0: "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/dlpfc_z_group1_grid80_spatial1/query/virtual_dlpfc_z_160.000.h5ad",
    240.0: "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/dlpfc_z_group1_grid80_spatial1/query/virtual_dlpfc_z_240.000.h5ad",
}
GENERATION_META_PATHS: Dict[float, str] = {
    80.0: "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/dlpfc_z_group1_grid80_spatial1/query/virtual_dlpfc_z_80.000_generation_meta.h5ad",
    160.0: "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/dlpfc_z_group1_grid80_spatial1/query/virtual_dlpfc_z_160.000_generation_meta.h5ad",
    240.0: "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/dlpfc_z_group1_grid80_spatial1/query/virtual_dlpfc_z_240.000_generation_meta.h5ad",
}

# Optional folder+template fallback if you do not want to enumerate the paths.
GENERATED_DIR: Optional[Path] = None
META_DIR: Optional[Path] = None
GENERATED_TEMPLATE = "virtual_dlpfc_z_{depth:.3f}.h5ad"
META_TEMPLATE = "virtual_dlpfc_z_{depth:.3f}_generation_meta.h5ad"

# Reconstruction density
N_VIRTUAL = 3
VIRTUAL_QUERY_DEPTHS: Optional[Sequence[float]] = None  # if None -> linearly spaced

# Figure 5 optionally includes the two non-anchor real slices as context.
INCLUDE_ALL_OBSERVED_IN_FIG5 = True

# Trajectories and reconstruction evidence
MAX_CURVES_PER_ANCHOR = 550
RANDOM_SEED = 0
CONTROL_SCALE = 0.34
CURVE_STEPS = 44
TRAJECTORY_DOMAIN_FILTER: Optional[Sequence[str]] = ["Layer4"]
# TRAJECTORY_DOMAIN_FILTER: Optional[Sequence[str]] = None
# Appearance
DEPTH_SCALE = 0.5   # anatomical depth scale retained by default
FIGSIZE = (11.6, 7.9)
ELEV = 15
AZIM = -145

REAL_POINT_SIZE = 5.0
REAL_POINT_ALPHA = 0.92
REAL_POINT_ALPHA_FADED = 0.34
VIRTUAL_POINT_SIZE = 5
VIRTUAL_POINT_ALPHA = 0.92
GRID_POINT_SIZE = 14.0
EVIDENCE_POINT_SIZE = 11.0
EVIDENCE_POINT_ALPHA = 0.88

GENE_CMAP = "Reds"
UPPER_PERCENTILE = 99.5

REAL_FRAME_COLOR = "#3D3D3D"
REAL_FRAME_LINEWIDTH = 1.05
REAL_FRAME_ALPHA = 0.74
VIRTUAL_FRAME_COLOR = "#666666"
VIRTUAL_FRAME_LINEWIDTH = 1.15
VIRTUAL_FRAME_ALPHA = 0.86
VIRTUAL_FRAME_DASH_PATTERN = (0, (4.0, 3.0))

ARROW_HEAD_LENGTH_SCALE = 0.48
ARROW_HEAD_WIDTH_SCALE = 0.24
CURVE_LINEWIDTH = 2.10
CURVE_ALPHA = 0.28

OUT_DIR: Optional[Path] = None
INVERT_X = True
INVERT_Y = True


# =============================================================================
# Style and saving
# =============================================================================

def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.8,
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
            "legend.title_fontsize": 7.8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 350,
            "figure.dpi": 120,
        }
    )


def save_or_show(
    fig: plt.Figure,
    outdir: Optional[Path] = None,
    stem: Optional[str] = None,
    *,
    dpi: int = 400,
    close: bool = True,
) -> Optional[Dict[str, Path]]:
    if outdir is None:
        plt.show()
        if close:
            plt.close(fig)
        return None

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if stem is None:
        raise ValueError("`stem` must be provided when `outdir` is not None.")

    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)

    if close:
        plt.close(fig)

    print(f"[OK] saved PNG: {png_path}")
    print(f"[OK] saved PDF: {pdf_path}")
    return {"png": png_path, "pdf": pdf_path}


# =============================================================================
# Path helpers
# =============================================================================

def parse_h5ad_path(root: Path, group_id: Any, sample_id: Any) -> Path:
    return root / H5AD_SUBDIR / H5AD_TEMPLATE.format(group_id=group_id, sample_id=sample_id)


def parse_network_npz_path(root: Path, group_id: Any, sample_id: Any) -> Path:
    return root / NETWORK_SUBDIR / NETWORK_TEMPLATE.format(group_id=group_id, sample_id=sample_id)


def find_group_dir(root: Path, group_id: Any, pairwise_subdir: str = PAIRWISE_SUBDIR) -> Path:
    candidates = [
        root / pairwise_subdir / f"group_{group_id}",
        root / f"group_{group_id}",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Cannot find pairwise folder for group {group_id}. Tried:\n" + "\n".join(str(x) for x in candidates)
    )


def parse_flow_path(root: Path, group_id: Any, fixed_id: Any, moving_id: Any) -> Path:
    group_dir = find_group_dir(root, group_id, pairwise_subdir=PAIRWISE_SUBDIR)
    return group_dir / FLOW_TEMPLATE.format(fixed_id=fixed_id, moving_id=moving_id)


def get_anchor_ids(pair_name: str = ANCHOR_PAIR_NAME) -> Tuple[int, int]:
    if pair_name not in ANCHOR_PAIR_OPTIONS:
        raise KeyError(
            f"Unknown anchor pair name {pair_name!r}. "
            f"Available: {list(ANCHOR_PAIR_OPTIONS.keys())}"
        )
    return ANCHOR_PAIR_OPTIONS[pair_name]


def get_virtual_depths(
    pair_name: str = ANCHOR_PAIR_NAME,
    n_virtual: int = N_VIRTUAL,
    explicit_depths: Optional[Sequence[float]] = VIRTUAL_QUERY_DEPTHS,
) -> List[float]:
    if explicit_depths is not None:
        return [float(x) for x in explicit_depths]

    sid_a, sid_b = get_anchor_ids(pair_name)
    za = float(OBSERVED_DEPTHS_UM[sid_a])
    zb = float(OBSERVED_DEPTHS_UM[sid_b])
    depths = np.linspace(za, zb, n_virtual + 2, dtype=float)[1:-1]
    return [float(x) for x in depths]


def resolve_generated_virtual_path(depth_um: float) -> Path:
    if depth_um in GENERATED_VIRTUAL_PATHS and GENERATED_VIRTUAL_PATHS[depth_um]:
        return Path(GENERATED_VIRTUAL_PATHS[depth_um])

    if GENERATED_DIR is not None:
        return Path(GENERATED_DIR) / GENERATED_TEMPLATE.format(depth=depth_um)

    raise FileNotFoundError(
        f"Generated virtual path for depth {depth_um} um is not configured.\n"
        "Fill GENERATED_VIRTUAL_PATHS or set GENERATED_DIR."
    )


def resolve_generation_meta_path(depth_um: float) -> Path:
    if depth_um in GENERATION_META_PATHS and GENERATION_META_PATHS[depth_um]:
        return Path(GENERATION_META_PATHS[depth_um])

    if META_DIR is not None:
        return Path(META_DIR) / META_TEMPLATE.format(depth=depth_um)

    raise FileNotFoundError(
        f"Generation-meta path for depth {depth_um} um is not configured.\n"
        "Fill GENERATION_META_PATHS or set META_DIR."
    )


# =============================================================================
# Data loading
# =============================================================================

def load_npz_coord(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "coord" not in data:
        raise KeyError(f"'coord' not found in {npz_path}")
    coords = np.asarray(data["coord"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coord in {npz_path} must have shape [N,2+], got {coords.shape}")
    return coords[:, :2]


def load_observed_slice(
    root: Path,
    group_id: Any,
    sample_id: Any,
    coord_key: str = REAL_COORD_KEY,
) -> Tuple[ad.AnnData, np.ndarray]:
    adata = sc.read_h5ad(parse_h5ad_path(root, group_id, sample_id))

    if coord_key in adata.obsm:
        coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)[:, :2]
    else:
        coords = load_npz_coord(parse_network_npz_path(root, group_id, sample_id))

    n_keep = min(adata.n_obs, len(coords))
    if adata.n_obs != len(coords):
        print(
            f"[WARN] sample={sample_id}: adata.n_obs={adata.n_obs}, coords={len(coords)}; using first {n_keep}."
        )
    return adata[:n_keep].copy(), coords[:n_keep]


def read_generated_virtual(depth_um: float) -> Tuple[ad.AnnData, np.ndarray, Path]:
    path = resolve_generated_virtual_path(depth_um)
    if not path.exists():
        raise FileNotFoundError(path)
    adata = ad.read_h5ad(path)
    if "valid" in adata.obs.columns:
        adata = adata[adata.obs["valid"].to_numpy().astype(bool)].copy()
    coords = get_spatial_coords(adata, source=f"generated virtual {path.name}")
    return adata, coords, path


def read_generation_meta(depth_um: float) -> Tuple[ad.AnnData, np.ndarray, np.ndarray, Path]:
    path = resolve_generation_meta_path(depth_um)
    if not path.exists():
        raise FileNotFoundError(path)
    meta = ad.read_h5ad(path)
    evidence_coords = get_spatial_coords(meta, source=f"generation meta {path.name}")
    grid_pts = get_grid_points(meta)
    return meta, evidence_coords, grid_pts, path


def get_spatial_coords(adata: ad.AnnData, *, source: str, spatial_key: str = "spatial") -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"{source}: obsm[{spatial_key!r}] not found. Available obsm keys: {list(adata.obsm.keys())}"
        )
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"{source}: expected spatial coordinates with shape [N,2+], got {coords.shape}")
    coords = coords[:, :2]
    if coords.shape[0] != adata.n_obs:
        raise ValueError(f"{source}: coordinate rows ({coords.shape[0]}) do not match n_obs ({adata.n_obs})")
    return coords


def get_grid_points(meta: ad.AnnData) -> np.ndarray:
    if "grid_pts" not in meta.uns:
        raise KeyError(
            "Generation metadata does not contain uns['grid_pts']. "
            f"Available uns keys: {list(meta.uns.keys())}"
        )
    grid_pts = np.asarray(meta.uns["grid_pts"], dtype=float)
    if grid_pts.ndim != 2 or grid_pts.shape[1] < 2:
        raise ValueError(f"uns['grid_pts'] must have shape [M,2+], received {grid_pts.shape}")
    return grid_pts[:, :2]


# =============================================================================
# Value extraction
# =============================================================================

def extract_gene_values(adata: ad.AnnData, gene: str, use_log1p: bool = USE_LOG1P) -> np.ndarray:
    if gene not in adata.var_names:
        matches = [g for g in adata.var_names if str(g).upper() == str(gene).upper()]
        if len(matches) == 1:
            gene = matches[0]
        else:
            raise KeyError(f"Gene {gene!r} not found in adata.var_names.")

    values = adata[:, gene].X
    if sp.issparse(values):
        values = values.toarray().reshape(-1)
    else:
        values = np.asarray(values).reshape(-1)
    values = values.astype(np.float32)
    values[~np.isfinite(values)] = 0.0
    if use_log1p:
        values = np.log1p(np.clip(values, 0.0, None))
    return values


def canonical_label(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return "NA"
    return text


def sort_key(value: Any):
    text = canonical_label(value)
    try:
        return (0, float(text), text)
    except Exception:
        return (1, text)


def extract_attribute_values(adata: ad.AnnData, attribute: str) -> np.ndarray:
    if attribute in adata.obs.columns:
        return np.asarray([canonical_label(x) for x in adata.obs[attribute].to_numpy()], dtype=str)
    raise KeyError(
        f"Attribute {attribute!r} not found in adata.obs. Available obs columns: {list(adata.obs.columns)}"
    )


def ordered_domain_labels(*arrays: Sequence[str], preferred_order: Sequence[str]) -> List[str]:
    present = set()
    for arr in arrays:
        present.update(canonical_label(x) for x in arr)
    ordered = [label for label in preferred_order if label in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def build_domain_color_map(*arrays: Sequence[str]) -> Tuple[List[np.ndarray], Dict[str, object]]:
    canonical_arrays = [np.asarray([canonical_label(x) for x in arr], dtype=str) for arr in arrays]
    categories = ordered_domain_labels(*canonical_arrays, preferred_order=PREFERRED_DOMAIN_ORDER)

    fallback_labels = [label for label in categories if label not in DLPFC_DOMAIN_COLORS]
    fallback = sns.color_palette(DOMAIN_PALETTE, n_colors=max(len(fallback_labels), 1))

    color_map: Dict[str, object] = {}
    fallback_idx = 0
    for label in categories:
        if label in DLPFC_DOMAIN_COLORS:
            color_map[label] = DLPFC_DOMAIN_COLORS[label]
        else:
            color_map[label] = fallback[fallback_idx]
            fallback_idx += 1

    return canonical_arrays, color_map


def make_shared_gene_norm(*arrays: np.ndarray, upper_percentile: float = UPPER_PERCENTILE) -> Normalize:
    combined = np.concatenate([np.asarray(x, dtype=float).reshape(-1) for x in arrays])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = 0.0 if np.nanmin(finite) >= 0 else float(np.nanmin(finite))
    vmax = float(np.nanpercentile(finite, upper_percentile))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax, clip=True)


def expression_rgba(
    values: np.ndarray,
    *,
    norm: Normalize,
    cmap_name: str = GENE_CMAP,
    alpha_min: float = 0.04,
    alpha_max: float = 0.96,
    alpha_gamma: float = 0.75,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scaled = np.clip(norm(values), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(scaled)
    alpha = alpha_min + (alpha_max - alpha_min) * np.power(scaled, alpha_gamma)
    rgba[:, 3] = alpha
    return rgba


# =============================================================================
# Geometry and 3D helpers
# =============================================================================

def finite_rows(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    return np.all(np.isfinite(coords[:, :2]), axis=1)


def combined_xy_limits(arrays: Iterable[np.ndarray], pad_fraction: float = 0.05) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    coords = np.vstack([np.asarray(x, dtype=float)[:, :2] for x in arrays if x is not None and len(x) > 0])
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0)
    pad = spans * pad_fraction
    return (
        (float(mins[0] - pad[0]), float(maxs[0] + pad[0])),
        (float(mins[1] - pad[1]), float(maxs[1] + pad[1])),
    )


def draw_slice_plane(
    ax,
    depth: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    *,
    edgecolor: str,
    linewidth: float,
    alpha: float,
    face_alpha: float = 0.035,
    dashed: bool = False,
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
        facecolors=(0.92, 0.92, 0.92, face_alpha),
        edgecolors=(0, 0, 0, 0),
        linewidths=0,
    )
    ax.add_collection3d(poly)

    frame = np.array(
        [
            [depth, x0, y0],
            [depth, x1, y0],
            [depth, x1, y1],
            [depth, x0, y1],
            [depth, x0, y0],
        ],
        dtype=float,
    )
    line, = ax.plot(frame[:, 0], frame[:, 1], frame[:, 2], color=edgecolor, linewidth=linewidth, alpha=alpha)
    if dashed:
        line.set_dashes(VIRTUAL_FRAME_DASH_PATTERN[1])


def add_plane_label(
    ax,
    depth: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    text: str,
    *,
    color: str = "black",
    fontsize: float = 9.6,
) -> None:
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    x_text = x0 + 0.04 * (x1 - x0)
    y_text = y0 - 0.085 * (y1 - y0)
    ax.text(depth, x_text, y_text, text, color=color, fontsize=fontsize, ha="left", va="top", zorder=10)


def style_3d_axis(
    ax,
    *,
    depth_positions: Sequence[float],
    depth_labels: Sequence[str],
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    invert_x: bool = INVERT_X,
    invert_y: bool = INVERT_Y,
) -> None:
    x_pad = 0.05 * max(x_bounds[1] - x_bounds[0], 1.0)
    y_pad = 0.05 * max(y_bounds[1] - y_bounds[0], 1.0)
    d_pad = 0.08 * max(depth_positions[-1] - depth_positions[0], 1.0)

    ax.set_xlim(depth_positions[0] - d_pad, depth_positions[-1] + d_pad)
    if invert_x:
        ax.set_ylim(x_bounds[1] + x_pad, x_bounds[0] - x_pad)
    else:
        ax.set_ylim(x_bounds[0] - x_pad, x_bounds[1] + x_pad)
    if invert_y:
        ax.set_zlim(y_bounds[1] + y_pad, y_bounds[0] - y_pad)
    else:
        ax.set_zlim(y_bounds[0] - y_pad, y_bounds[1] + y_pad)

    ax.set_xticks(depth_positions)
    ax.set_xticklabels(depth_labels, fontsize=8.1)
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("Anatomical depth (um)", labelpad=10)
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    ax.set_proj_type("ortho")
    ax.view_init(elev=ELEV, azim=AZIM)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0.82, 0.82, 0.82, 0.25))

    depth_span = max(depth_positions[-1] - depth_positions[0], 1.0)
    x_span = max(x_bounds[1] - x_bounds[0], 1.0)
    y_span = max(y_bounds[1] - y_bounds[0], 1.0)
    ax.set_box_aspect((depth_span * 1.15, x_span, y_span))


def plot_template_grid_3d(ax, grid_pts: np.ndarray, *, depth: float, point_size: float) -> None:
    ax.scatter(
        np.full(len(grid_pts), depth),
        grid_pts[:, 0],
        grid_pts[:, 1],
        s=point_size,
        facecolors="none",
        edgecolors="#A8A8A8",
        linewidths=0.55,
        depthshade=False,
        rasterized=True,
    )


def plot_expression_points_3d(
    ax,
    coords: np.ndarray,
    values: np.ndarray,
    *,
    depth: float,
    norm: Normalize,
    point_size: float,
    alpha_max: float = 0.92,
) -> None:
    order = np.argsort(np.asarray(values, dtype=float))
    rgba = expression_rgba(values, norm=norm, cmap_name=GENE_CMAP, alpha_max=alpha_max)
    ax.scatter(
        np.full(len(coords), depth)[order],
        coords[order, 0],
        coords[order, 1],
        c=rgba[order],
        s=point_size,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )


def plot_category_points_3d(
    ax,
    coords: np.ndarray,
    categories: np.ndarray,
    *,
    depth: float,
    color_map: Mapping[str, object],
    point_size: float,
    alpha: float,
) -> None:
    colors = [color_map[str(cat)] for cat in categories]
    ax.scatter(
        np.full(len(coords), depth),
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )


def add_gene_colorbar(fig: plt.Figure, ax, norm: Normalize, *, label: str) -> None:
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=GENE_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.05, shrink=0.72)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def add_domain_legend(
    fig: plt.Figure,
    color_map: Mapping[str, object],
    *,
    title: str,
    anchor: Tuple[float, float] = (0.87, 0.50),
) -> None:
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color, markeredgecolor="none", markersize=5.2, label=str(cat))
        for cat, color in color_map.items()
    ]
    fig.legend(
        handles=handles,
        title=title,
        loc="center left",
        bbox_to_anchor=anchor,
        frameon=False,
        ncol=1 if len(handles) <= 10 else 2,
        handletextpad=0.35,
        columnspacing=0.8,
        labelspacing=0.28,
    )


# =============================================================================
# Flow and correspondence helpers
# =============================================================================

def ensure_flow_hw2(flow: np.ndarray) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3:
        raise ValueError(f"Flow must be three-dimensional, got {flow.shape}.")
    if flow.shape[-1] == 2:
        return flow
    if flow.shape[0] == 2:
        return np.moveaxis(flow, 0, -1)
    raise ValueError(f"Cannot interpret flow shape {flow.shape} as [H,W,2].")


def inverse_warp_points(points_xy: np.ndarray, flow_hw2: np.ndarray) -> np.ndarray:
    warped = find_inverse_warp_coords(points_xy, flow_hw2)
    if isinstance(warped, (tuple, list)):
        warped = warped[0]
    warped = np.asarray(warped, dtype=np.float32)
    if warped.ndim != 2 or warped.shape[1] < 2:
        raise ValueError(f"Unexpected inverse-warp output shape: {warped.shape}")
    return warped[:, :2]


def choose_seed_indices(
    coords: np.ndarray,
    adata: ad.AnnData,
    *,
    max_points: int,
    random_seed: int,
    domain_filter: Optional[Sequence[str]] = TRAJECTORY_DOMAIN_FILTER,
    domain_key: str = DOMAIN_KEY,
) -> np.ndarray:
    idx = np.arange(min(len(coords), adata.n_obs))
    if domain_filter is not None:
        labels = extract_attribute_values(adata[: len(idx)].copy(), domain_key)
        keep = np.isin(labels, np.asarray([canonical_label(x) for x in domain_filter], dtype=str))
        idx = idx[keep]
    if len(idx) <= max_points:
        return idx
    rng = np.random.default_rng(random_seed)
    idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    return idx


# =============================================================================
# Smooth curves and lifted piecewise fields
# =============================================================================

def cubic_bezier_2d(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n_steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(n_steps), dtype=float)[:, None]
    omt = 1.0 - t
    return (
        omt**3 * p0[None, :]
        + 3.0 * omt**2 * t * p1[None, :]
        + 3.0 * omt * t**2 * p2[None, :]
        + t**3 * p3[None, :]
    ).astype(np.float32)


def build_smooth_curve_2d(start_xy: np.ndarray, end_xy: np.ndarray, *, control_scale: float = CONTROL_SCALE, n_steps: int = CURVE_STEPS) -> np.ndarray:
    start_xy = np.asarray(start_xy, dtype=float)
    end_xy = np.asarray(end_xy, dtype=float)
    disp = end_xy - start_xy
    dist = float(np.linalg.norm(disp))
    if not np.isfinite(dist) or dist <= 1e-8:
        return np.repeat(start_xy[None, :], n_steps, axis=0).astype(np.float32)
    unit = disp / dist
    perp = np.array([-unit[1], unit[0]], dtype=float)
    sign = -1.0 if start_xy[0] < end_xy[0] else 1.0
    bend = 0.18 * dist * sign * perp
    handle = control_scale * dist
    c1 = start_xy + handle * unit + 0.55 * bend
    c2 = end_xy - handle * unit + 0.25 * bend
    return cubic_bezier_2d(start_xy, c1, c2, end_xy, n_steps)


def split_curve_xy(curve_xy: np.ndarray, n_sections: int) -> List[np.ndarray]:
    n = len(curve_xy)
    idx_edges = np.linspace(0, n - 1, n_sections + 1, dtype=int)
    pieces = []
    for i in range(n_sections):
        lo, hi = idx_edges[i], idx_edges[i + 1]
        if hi <= lo:
            hi = min(lo + 1, n - 1)
        seg = curve_xy[lo : hi + 1]
        if len(seg) < 2:
            seg = np.vstack([curve_xy[lo], curve_xy[min(lo + 1, n - 1)]])
        pieces.append(seg)
    return pieces


def lift_curve_segment(seg_xy: np.ndarray, depth_start: float, depth_end: float) -> np.ndarray:
    depths = np.linspace(depth_start, depth_end, len(seg_xy), dtype=float)
    return np.column_stack([depths, seg_xy[:, 0], seg_xy[:, 1]]).astype(np.float32)


def make_arrowhead_segments_3d(tip: np.ndarray, previous: np.ndarray, *, head_length_scale: float = ARROW_HEAD_LENGTH_SCALE, head_width_scale: float = ARROW_HEAD_WIDTH_SCALE) -> Optional[np.ndarray]:
    tip = np.asarray(tip, dtype=float)
    previous = np.asarray(previous, dtype=float)
    tangent = tip - previous
    step_len = float(np.linalg.norm(tangent))
    if not np.isfinite(step_len) or step_len <= 1e-8:
        return None
    tangent = tangent / step_len
    head_len = head_length_scale * step_len
    head_width = head_width_scale * head_len
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(tangent, ref)) > 0.90:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    side = np.cross(tangent, ref)
    side_norm = float(np.linalg.norm(side))
    if not np.isfinite(side_norm) or side_norm <= 1e-8:
        return None
    side = side / side_norm
    base = tip - head_len * tangent
    wing1 = base + head_width * side
    wing2 = base - head_width * side
    return np.array([[tip, wing1], [tip, wing2]], dtype=float)


def add_curve_collection(ax, curves: Sequence[np.ndarray], *, color: str, linewidth: float, alpha: float, add_arrowheads: bool = True) -> None:
    for curve in curves:
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, linewidth=linewidth, alpha=alpha, zorder=3)
        if add_arrowheads and len(curve) >= 3:
            segs = make_arrowhead_segments_3d(tip=curve[-1], previous=curve[-3])
            if segs is not None:
                for seg in segs:
                    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=linewidth, alpha=alpha, zorder=4)


def make_progressive_palette(base_color: str, n_sections: int) -> List[Tuple[float, float, float]]:
    # Deeper/lighter progression within one consistent hue.
    return sns.light_palette(base_color, n_colors=n_sections, reverse=True)


# =============================================================================
# Context assembly
# =============================================================================

class Context:
    pass


def build_context(root: Path = ROOT, group_id: Any = GROUP_ID, pair_name: str = ANCHOR_PAIR_NAME) -> Context:
    ctx = Context()
    ctx.root = Path(root)
    ctx.group_id = group_id
    ctx.pair_name = pair_name
    ctx.anchor_a, ctx.anchor_b = get_anchor_ids(pair_name)
    ctx.anchor_depth_a = float(OBSERVED_DEPTHS_UM[ctx.anchor_a]) * DEPTH_SCALE
    ctx.anchor_depth_b = float(OBSERVED_DEPTHS_UM[ctx.anchor_b]) * DEPTH_SCALE
    ctx.virtual_depths = [float(x) for x in get_virtual_depths(pair_name)]
    ctx.virtual_depths_scaled = [float(x) * DEPTH_SCALE for x in ctx.virtual_depths]

    ctx.observed = {}
    obs_coords_all = []
    for sid in OBSERVED_SAMPLE_IDS:
        adata_i, coords_i = load_observed_slice(root, group_id, sid)
        ctx.observed[sid] = {
            "adata": adata_i,
            "coords": coords_i,
            "depth_um": float(OBSERVED_DEPTHS_UM[sid]),
            "depth_scaled": float(OBSERVED_DEPTHS_UM[sid]) * DEPTH_SCALE,
            "label": f"Real {int(OBSERVED_DEPTHS_UM[sid])} um",
        }
        obs_coords_all.append(coords_i)

    # Generated/meta are optional.
    ctx.generated = {}
    for depth in ctx.virtual_depths:
        try:
            adata_i, coords_i, path_i = read_generated_virtual(depth)
            ctx.generated[depth] = {"adata": adata_i, "coords": coords_i, "path": path_i}
            obs_coords_all.append(coords_i)
        except Exception as exc:
            ctx.generated[depth] = {"error": exc}

    ctx.meta = {}
    for depth in ctx.virtual_depths:
        try:
            meta_i, evidence_coords_i, grid_pts_i, path_i = read_generation_meta(depth)
            ctx.meta[depth] = {
                "adata": meta_i,
                "coords": evidence_coords_i,
                "grid_pts": grid_pts_i,
                "path": path_i,
            }
            obs_coords_all.append(evidence_coords_i)
            obs_coords_all.append(grid_pts_i)
        except Exception as exc:
            ctx.meta[depth] = {"error": exc}

    # Common plotting bounds.
    ctx.x_bounds, ctx.y_bounds = combined_xy_limits(obs_coords_all)
    return ctx


# =============================================================================
# Shared renderers for gene/domain mode
# =============================================================================

def render_slices_in_mode(
    ax,
    items: Sequence[Tuple[np.ndarray, ad.AnnData, float]],
    *,
    display_mode: str,
    point_size: float,
    point_alpha: float,
) -> Dict[str, Any]:
    display_mode = str(display_mode).lower()

    if display_mode == "gene":
        values_per_item = [extract_gene_values(adata_i, MARKER_GENE, use_log1p=USE_LOG1P) for _, adata_i, _ in items]
        norm = make_shared_gene_norm(*values_per_item)
        for (coords_i, _, depth_i), vals_i in zip(items, values_per_item):
            plot_expression_points_3d(ax, coords_i, vals_i, depth=depth_i, norm=norm, point_size=point_size, alpha_max=point_alpha)
        return {"norm": norm}

    attr_values = [extract_attribute_values(adata_i, DOMAIN_KEY) for _, adata_i, _ in items]
    arrays, color_map = build_domain_color_map(*attr_values)
    for (coords_i, _, depth_i), arr_i in zip(items, arrays):
        plot_category_points_3d(ax, coords_i, arr_i, depth=depth_i, color_map=color_map, point_size=point_size, alpha=point_alpha)
    return {"color_map": color_map}


# =============================================================================
# Figure 1: all observed slices
# =============================================================================

def plot_figure1_observed_stack_3d(
    *,
    display_mode: str = DISPLAY_MODE,
    outdir: Optional[Path] = OUT_DIR,
    stem: str = "figure1_dlpfc_observed_stack_3d",
):
    set_publication_style()
    ctx = build_context(pair_name=ANCHOR_PAIR_NAME)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    items = []
    for sid in OBSERVED_SAMPLE_IDS:
        info = ctx.observed[sid]
        draw_slice_plane(
            ax,
            info["depth_scaled"],
            ctx.x_bounds,
            ctx.y_bounds,
            edgecolor=REAL_FRAME_COLOR,
            linewidth=REAL_FRAME_LINEWIDTH,
            alpha=REAL_FRAME_ALPHA,
            face_alpha=0.035,
            dashed=False,
        )
        add_plane_label(ax, info["depth_scaled"], ctx.x_bounds, ctx.y_bounds, info["label"])
        items.append((info["coords"], info["adata"], info["depth_scaled"]))

    result = render_slices_in_mode(
        ax,
        items,
        display_mode=display_mode,
        point_size=REAL_POINT_SIZE,
        point_alpha=REAL_POINT_ALPHA,
    )

    style_3d_axis(
        ax,
        depth_positions=[ctx.observed[sid]["depth_scaled"] for sid in OBSERVED_SAMPLE_IDS],
        depth_labels=[f"{int(ctx.observed[sid]['depth_um'])}" for sid in OBSERVED_SAMPLE_IDS],
        x_bounds=ctx.x_bounds,
        y_bounds=ctx.y_bounds,
    )

    if str(display_mode).lower() == "gene":
        add_gene_colorbar(fig, ax, result["norm"], label=f"log1p({MARKER_GENE})")
        subtitle = f"{MARKER_GENE} across all observed slices"
    else:
        add_domain_legend(fig, result["color_map"], title=DOMAIN_KEY)
        subtitle = f"{DOMAIN_KEY} across all observed slices"

    ax.set_title(
        "Observed DLPFC stack\n" + subtitle,
        pad=15,
        fontsize=12.2,
        fontweight="bold",
    )
    fig.tight_layout()
    return save_or_show(fig, outdir=outdir, stem=f"{stem}_{display_mode}")


# =============================================================================
# Figure 2: reserved anchors
# =============================================================================

def plot_figure2_reserved_anchors_3d(
    *,
    display_mode: str = DISPLAY_MODE,
    pair_name: str = ANCHOR_PAIR_NAME,
    outdir: Optional[Path] = OUT_DIR,
    stem: str = "figure2_dlpfc_reserved_anchors_3d",
):
    set_publication_style()
    ctx = build_context(pair_name=pair_name)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    items = []
    for sid in [ctx.anchor_a, ctx.anchor_b]:
        info = ctx.observed[sid]
        draw_slice_plane(
            ax,
            info["depth_scaled"],
            ctx.x_bounds,
            ctx.y_bounds,
            edgecolor=REAL_FRAME_COLOR,
            linewidth=REAL_FRAME_LINEWIDTH,
            alpha=REAL_FRAME_ALPHA,
            face_alpha=0.04,
            dashed=False,
        )
        add_plane_label(ax, info["depth_scaled"], ctx.x_bounds, ctx.y_bounds, f"Anchor {int(info['depth_um'])} um")
        items.append((info["coords"], info["adata"], info["depth_scaled"]))

    result = render_slices_in_mode(ax, items, display_mode=display_mode, point_size=REAL_POINT_SIZE, point_alpha=REAL_POINT_ALPHA)

    style_3d_axis(
        ax,
        depth_positions=[ctx.anchor_depth_a, ctx.anchor_depth_b],
        depth_labels=[f"{int(OBSERVED_DEPTHS_UM[ctx.anchor_a])}", f"{int(OBSERVED_DEPTHS_UM[ctx.anchor_b])}"],
        x_bounds=ctx.x_bounds,
        y_bounds=ctx.y_bounds,
    )

    if str(display_mode).lower() == "gene":
        add_gene_colorbar(fig, ax, result["norm"], label=f"log1p({MARKER_GENE})")
        subtitle = f"Reserved anchors: {int(OBSERVED_DEPTHS_UM[ctx.anchor_a])} and {int(OBSERVED_DEPTHS_UM[ctx.anchor_b])} um"
    else:
        add_domain_legend(fig, result["color_map"], title=DOMAIN_KEY)
        subtitle = f"Reserved anchors: {int(OBSERVED_DEPTHS_UM[ctx.anchor_a])} and {int(OBSERVED_DEPTHS_UM[ctx.anchor_b])} um"

    ax.set_title("Reserved anchor slices\n" + subtitle, pad=15, fontsize=12.2, fontweight="bold")
    fig.tight_layout()
    return save_or_show(fig, outdir=outdir, stem=f"{stem}_{pair_name}_{display_mode}")


# =============================================================================
# Figure 3: transition-field truncation
# =============================================================================

def plot_figure3_transition_truncation_3d(
    *,
    display_mode: str = DISPLAY_MODE,
    pair_name: str = ANCHOR_PAIR_NAME,
    outdir: Optional[Path] = OUT_DIR,
    stem: str = "figure3_dlpfc_transition_truncation_3d",
):
    set_publication_style()
    ctx = build_context(pair_name=pair_name)

    # Load the pairwise flow from anchor A -> anchor B reference frame.
    flow_path = parse_flow_path(ROOT, GROUP_ID, fixed_id=ctx.anchor_b, moving_id=ctx.anchor_a)
    if not flow_path.exists():
        raise FileNotFoundError(f"Missing pairwise flow file: {flow_path}")

    flow_ab = ensure_flow_hw2(np.load(flow_path))
    coords_a = ctx.observed[ctx.anchor_a]["coords"]
    adata_a = ctx.observed[ctx.anchor_a]["adata"]
    coords_b = ctx.observed[ctx.anchor_b]["coords"]
    adata_b = ctx.observed[ctx.anchor_b]["adata"]

    # Choose seed points on anchor A and inverse-warp them to anchor B.
    seed_idx = choose_seed_indices(coords_a, adata_a, max_points=MAX_CURVES_PER_ANCHOR, random_seed=RANDOM_SEED)
    start_xy = coords_a[seed_idx]
    end_xy = inverse_warp_points(start_xy, flow_ab)

    valid = finite_rows(start_xy) & finite_rows(end_xy)
    start_xy = start_xy[valid]
    end_xy = end_xy[valid]

    plane_depths = [ctx.anchor_depth_a] + ctx.virtual_depths_scaled + [ctx.anchor_depth_b]
    plane_labels = [f"{int(OBSERVED_DEPTHS_UM[ctx.anchor_a])}"] + [f"{int(x)}" for x in ctx.virtual_depths] + [f"{int(OBSERVED_DEPTHS_UM[ctx.anchor_b])}"]

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    # Draw anchor and virtual planes.
    for depth, label, is_virtual in zip(plane_depths, plane_labels, [False] + [True] * len(ctx.virtual_depths) + [False]):
        draw_slice_plane(
            ax,
            depth,
            ctx.x_bounds,
            ctx.y_bounds,
            edgecolor=VIRTUAL_FRAME_COLOR if is_virtual else REAL_FRAME_COLOR,
            linewidth=VIRTUAL_FRAME_LINEWIDTH if is_virtual else REAL_FRAME_LINEWIDTH,
            alpha=VIRTUAL_FRAME_ALPHA if is_virtual else REAL_FRAME_ALPHA,
            face_alpha=0.015 if is_virtual else 0.03,
            dashed=is_virtual,
        )
        tag = f"Virtual {label} um" if is_virtual else f"Real {label} um"
        add_plane_label(ax, depth, ctx.x_bounds, ctx.y_bounds, tag, color=VIRTUAL_FRAME_COLOR if is_virtual else "black")

    # Plot anchors only.
    anchor_items = [
        (coords_a, adata_a, ctx.anchor_depth_a),
        (coords_b, adata_b, ctx.anchor_depth_b),
    ]
    result = render_slices_in_mode(ax, anchor_items, display_mode=display_mode, point_size=REAL_POINT_SIZE, point_alpha=REAL_POINT_ALPHA)

    # Piecewise lifted fields.
    n_sections = len(plane_depths) - 1
    section_colors = make_progressive_palette("#4E79A7", n_sections)

    all_segment_curves: List[List[np.ndarray]] = [[] for _ in range(n_sections)]
    for p0, p1 in zip(start_xy, end_xy):
        curve_xy = build_smooth_curve_2d(p0, p1, n_steps=CURVE_STEPS)
        pieces = split_curve_xy(curve_xy, n_sections=n_sections)
        for i, seg_xy in enumerate(pieces):
            seg_3d = lift_curve_segment(seg_xy, plane_depths[i], plane_depths[i + 1])
            all_segment_curves[i].append(seg_3d)

    for i, curves_i in enumerate(all_segment_curves):
        add_curve_collection(
            ax,
            curves_i,
            color=section_colors[i],
            linewidth=CURVE_LINEWIDTH,
            alpha=CURVE_ALPHA,
            add_arrowheads=True,
        )

    style_3d_axis(ax, depth_positions=plane_depths, depth_labels=plane_labels, x_bounds=ctx.x_bounds, y_bounds=ctx.y_bounds)

    aux_handles = [
        Line2D([0], [0], color=section_colors[i], linewidth=CURVE_LINEWIDTH, label=f"Section {i + 1}")
        for i in range(n_sections)
    ]
    aux_handles.append(Line2D([0], [0], color=VIRTUAL_FRAME_COLOR, linestyle="--", linewidth=VIRTUAL_FRAME_LINEWIDTH, label="Virtual slice"))
    leg = ax.legend(handles=aux_handles, loc="upper left", bbox_to_anchor=(0.01, 0.96), frameon=False, ncol=1)
    ax.add_artist(leg)

    if str(display_mode).lower() == "gene":
        add_gene_colorbar(fig, ax, result["norm"], label=f"log1p({MARKER_GENE})")
        subtitle = f"Piecewise truncation between {int(OBSERVED_DEPTHS_UM[ctx.anchor_a])} and {int(OBSERVED_DEPTHS_UM[ctx.anchor_b])} um"
    else:
        add_domain_legend(fig, result["color_map"], title=DOMAIN_KEY)
        subtitle = f"Piecewise truncation between {int(OBSERVED_DEPTHS_UM[ctx.anchor_a])} and {int(OBSERVED_DEPTHS_UM[ctx.anchor_b])} um"

    ax.set_title("Transition-field truncation for dense reconstruction\n" + subtitle, pad=15, fontsize=12.2, fontweight="bold")
    fig.tight_layout()
    return save_or_show(fig, outdir=outdir, stem=f"{stem}_{pair_name}_{display_mode}")


# =============================================================================
# Figure 4: core reconstruction / evidence collection
# =============================================================================

def infer_evidence_group_indices(meta: ad.AnnData, n_a: int, n_b: int, anchor_a_id: Any, anchor_b_id: Any) -> Tuple[np.ndarray, np.ndarray]:
    source_columns = ["source_anchor_id", "source_id", "anchor_id", "anchor_sample_id", "source_sample_id"]
    for column in source_columns:
        if column in meta.obs.columns:
            raw = np.asarray([canonical_label(x) for x in meta.obs[column].to_numpy()], dtype=str)
            idx_a = np.where(raw == canonical_label(anchor_a_id))[0]
            idx_b = np.where(raw == canonical_label(anchor_b_id))[0]
            if len(idx_a) > 0 and len(idx_b) > 0:
                return idx_a, idx_b
    total_expected = int(n_a + n_b)
    if meta.n_obs < total_expected:
        raise ValueError(
            f"Generation metadata has {meta.n_obs} evidence points, but {total_expected} are expected from the two anchors."
        )
    return np.arange(n_a), np.arange(n_a, n_a + n_b)


def plot_figure4_core_reconstruction_3d(
    *,
    display_mode: str = DISPLAY_MODE,
    pair_name: str = ANCHOR_PAIR_NAME,
    outdir: Optional[Path] = OUT_DIR,
    stem: str = "figure4_dlpfc_core_reconstruction_3d",
):
    set_publication_style()
    ctx = build_context(pair_name=pair_name)

    missing = [depth for depth in ctx.virtual_depths if "error" in ctx.meta[depth]]
    if missing:
        raise RuntimeError(
            "Figure 4 requires generation metadata for each virtual slice. Missing or unreadable depths: "
            + ", ".join(str(x) for x in missing)
        )

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    # Draw real anchors.
    for sid in [ctx.anchor_a, ctx.anchor_b]:
        info = ctx.observed[sid]
        draw_slice_plane(ax, info["depth_scaled"], ctx.x_bounds, ctx.y_bounds, edgecolor=REAL_FRAME_COLOR, linewidth=REAL_FRAME_LINEWIDTH, alpha=REAL_FRAME_ALPHA, face_alpha=0.028, dashed=False)
        add_plane_label(ax, info["depth_scaled"], ctx.x_bounds, ctx.y_bounds, f"Anchor {int(info['depth_um'])} um")

    # Draw virtual support planes and grid templates.
    for depth, depth_scaled in zip(ctx.virtual_depths, ctx.virtual_depths_scaled):
        draw_slice_plane(ax, depth_scaled, ctx.x_bounds, ctx.y_bounds, edgecolor=VIRTUAL_FRAME_COLOR, linewidth=VIRTUAL_FRAME_LINEWIDTH, alpha=VIRTUAL_FRAME_ALPHA, face_alpha=0.015, dashed=True)
        add_plane_label(ax, depth_scaled, ctx.x_bounds, ctx.y_bounds, f"Support {int(depth)} um", color=VIRTUAL_FRAME_COLOR)
        plot_template_grid_3d(ax, ctx.meta[depth]["grid_pts"], depth=depth_scaled, point_size=GRID_POINT_SIZE)

    # Plot anchors with transparency.
    anchor_items = [
        (ctx.observed[ctx.anchor_a]["coords"], ctx.observed[ctx.anchor_a]["adata"], ctx.anchor_depth_a),
        (ctx.observed[ctx.anchor_b]["coords"], ctx.observed[ctx.anchor_b]["adata"], ctx.anchor_depth_b),
    ]
    result = render_slices_in_mode(ax, anchor_items, display_mode=display_mode, point_size=REAL_POINT_SIZE, point_alpha=REAL_POINT_ALPHA_FADED)

    # Plot evidence on each virtual support and transport curves from both anchors.
    if str(display_mode).lower() == "gene":
        ev_vals_all = []
        for depth in ctx.virtual_depths:
            ev_vals_all.append(extract_gene_values(ctx.meta[depth]["adata"], MARKER_GENE, use_log1p=USE_LOG1P))
        anchor_vals = [extract_gene_values(ctx.observed[ctx.anchor_a]["adata"], MARKER_GENE, use_log1p=USE_LOG1P), extract_gene_values(ctx.observed[ctx.anchor_b]["adata"], MARKER_GENE, use_log1p=USE_LOG1P)]
        norm = make_shared_gene_norm(*(anchor_vals + ev_vals_all))
    else:
        attr_arrays = [extract_attribute_values(ctx.observed[ctx.anchor_a]["adata"], DOMAIN_KEY), extract_attribute_values(ctx.observed[ctx.anchor_b]["adata"], DOMAIN_KEY)]
        for depth in ctx.virtual_depths:
            attr_arrays.append(extract_attribute_values(ctx.meta[depth]["adata"], DOMAIN_KEY))
        _, color_map = build_domain_color_map(*attr_arrays)

    colors_forward = make_progressive_palette("#4E79A7", len(ctx.virtual_depths))
    colors_backward = make_progressive_palette("#D3725B", len(ctx.virtual_depths))

    for i, depth in enumerate(ctx.virtual_depths):
        meta_i = ctx.meta[depth]["adata"]
        coords_i = ctx.meta[depth]["coords"]
        depth_scaled = depth * DEPTH_SCALE

        idx_a, idx_b = infer_evidence_group_indices(meta_i, len(ctx.observed[ctx.anchor_a]["coords"]), len(ctx.observed[ctx.anchor_b]["coords"]), ctx.anchor_a, ctx.anchor_b)

        if str(display_mode).lower() == "gene":
            vals_i = extract_gene_values(meta_i, MARKER_GENE, use_log1p=USE_LOG1P)
            plot_expression_points_3d(ax, coords_i, vals_i, depth=depth_scaled, norm=norm, point_size=EVIDENCE_POINT_SIZE, alpha_max=EVIDENCE_POINT_ALPHA)
        else:
            attrs_i = extract_attribute_values(meta_i, DOMAIN_KEY)
            plot_category_points_3d(ax, coords_i, attrs_i, depth=depth_scaled, color_map=color_map, point_size=EVIDENCE_POINT_SIZE, alpha=EVIDENCE_POINT_ALPHA)

        curves_a = []
        curves_b = []
        start_a = ctx.observed[ctx.anchor_a]["coords"][: len(idx_a)]
        end_a = coords_i[idx_a]
        start_b = ctx.observed[ctx.anchor_b]["coords"][: len(idx_b)]
        end_b = coords_i[idx_b]

        keep_a = np.sort(np.random.default_rng(RANDOM_SEED + i).choice(len(end_a), size=min(MAX_CURVES_PER_ANCHOR, len(end_a)), replace=False)) if len(end_a) > 0 else np.array([], dtype=int)
        keep_b = np.sort(np.random.default_rng(RANDOM_SEED + 37 + i).choice(len(end_b), size=min(MAX_CURVES_PER_ANCHOR, len(end_b)), replace=False)) if len(end_b) > 0 else np.array([], dtype=int)

        for p0, p1 in zip(start_a[keep_a], end_a[keep_a]):
            seg = build_smooth_curve_2d(p0, p1, n_steps=CURVE_STEPS)
            curves_a.append(lift_curve_segment(seg, ctx.anchor_depth_a, depth_scaled))
        for p0, p1 in zip(start_b[keep_b], end_b[keep_b]):
            seg = build_smooth_curve_2d(p0, p1, n_steps=CURVE_STEPS)
            curves_b.append(lift_curve_segment(seg, ctx.anchor_depth_b, depth_scaled))

        add_curve_collection(ax, curves_a, color=colors_forward[i], linewidth=CURVE_LINEWIDTH, alpha=0.44, add_arrowheads=True)
        add_curve_collection(ax, curves_b, color=colors_backward[i], linewidth=CURVE_LINEWIDTH, alpha=0.44, add_arrowheads=True)

    plane_depths = [ctx.anchor_depth_a] + ctx.virtual_depths_scaled + [ctx.anchor_depth_b]
    plane_labels = [f"{int(OBSERVED_DEPTHS_UM[ctx.anchor_a])}"] + [f"{int(x)}" for x in ctx.virtual_depths] + [f"{int(OBSERVED_DEPTHS_UM[ctx.anchor_b])}"]
    style_3d_axis(ax, depth_positions=plane_depths, depth_labels=plane_labels, x_bounds=ctx.x_bounds, y_bounds=ctx.y_bounds)

    aux_handles = [
        Line2D([0], [0], color=VIRTUAL_FRAME_COLOR, linestyle="--", linewidth=VIRTUAL_FRAME_LINEWIDTH, label="Visium support"),
        Line2D([0], [0], color=colors_forward[-1] if colors_forward else "#4E79A7", linewidth=CURVE_LINEWIDTH, label=f"Anchor {int(OBSERVED_DEPTHS_UM[ctx.anchor_a])} → support"),
        Line2D([0], [0], color=colors_backward[-1] if colors_backward else "#D3725B", linewidth=CURVE_LINEWIDTH, label=f"Anchor {int(OBSERVED_DEPTHS_UM[ctx.anchor_b])} → support"),
    ]
    leg = ax.legend(handles=aux_handles, loc="upper left", bbox_to_anchor=(0.01, 0.96), frameon=False)
    ax.add_artist(leg)

    if str(display_mode).lower() == "gene":
        add_gene_colorbar(fig, ax, norm, label=f"log1p({MARKER_GENE})")
        subtitle = "Visium template + collected gene evidence"
    else:
        add_domain_legend(fig, color_map, title=DOMAIN_KEY)
        subtitle = "Visium template + collected domain evidence"

    ax.set_title("Core dense-reconstruction process\n" + subtitle, pad=15, fontsize=12.2, fontweight="bold")
    fig.tight_layout()
    return save_or_show(fig, outdir=outdir, stem=f"{stem}_{pair_name}_{display_mode}")


# =============================================================================
# Figure 5: final reconstructed stack
# =============================================================================

def plot_figure5_final_reconstructed_stack_3d(
    *,
    display_mode: str = DISPLAY_MODE,
    pair_name: str = ANCHOR_PAIR_NAME,
    include_all_observed: bool = INCLUDE_ALL_OBSERVED_IN_FIG5,
    outdir: Optional[Path] = OUT_DIR,
    stem: str = "figure5_dlpfc_final_reconstructed_stack_3d",
):
    set_publication_style()
    ctx = build_context(pair_name=pair_name)

    missing = [depth for depth in ctx.virtual_depths if "error" in ctx.generated[depth]]
    if missing:
        raise RuntimeError(
            "Figure 5 requires generated virtual slices for each virtual depth. Missing or unreadable depths: "
            + ", ".join(str(x) for x in missing)
        )

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    items: List[Tuple[np.ndarray, ad.AnnData, float]] = []
    depths_for_axis: List[float] = []
    labels_for_axis: List[str] = []

    real_ids_to_plot = OBSERVED_SAMPLE_IDS if include_all_observed else [ctx.anchor_a, ctx.anchor_b]
    for sid in real_ids_to_plot:
        info = ctx.observed[sid]
        draw_slice_plane(ax, info["depth_scaled"], ctx.x_bounds, ctx.y_bounds, edgecolor=REAL_FRAME_COLOR, linewidth=REAL_FRAME_LINEWIDTH, alpha=REAL_FRAME_ALPHA, face_alpha=0.034, dashed=False)
        add_plane_label(ax, info["depth_scaled"], ctx.x_bounds, ctx.y_bounds, f"Real {int(info['depth_um'])} um")
        items.append((info["coords"], info["adata"], info["depth_scaled"]))
        depths_for_axis.append(info["depth_scaled"])
        labels_for_axis.append(f"{int(info['depth_um'])}")

    for depth in ctx.virtual_depths:
        depth_scaled = depth * DEPTH_SCALE
        draw_slice_plane(ax, depth_scaled, ctx.x_bounds, ctx.y_bounds, edgecolor=VIRTUAL_FRAME_COLOR, linewidth=VIRTUAL_FRAME_LINEWIDTH, alpha=VIRTUAL_FRAME_ALPHA, face_alpha=0.018, dashed=True)
        add_plane_label(ax, depth_scaled, ctx.x_bounds, ctx.y_bounds, f"Virtual {int(depth)} um", color=VIRTUAL_FRAME_COLOR)
        items.append((ctx.generated[depth]["coords"], ctx.generated[depth]["adata"], depth_scaled))
        depths_for_axis.append(depth_scaled)
        labels_for_axis.append(f"{int(depth)}")

    # sort by depth for better axis/stack order
    sorter = np.argsort(depths_for_axis)
    items = [items[i] for i in sorter]
    depths_for_axis = [depths_for_axis[i] for i in sorter]
    labels_for_axis = [labels_for_axis[i] for i in sorter]

    result = render_slices_in_mode(ax, items, display_mode=display_mode, point_size=VIRTUAL_POINT_SIZE, point_alpha=VIRTUAL_POINT_ALPHA)

    style_3d_axis(ax, depth_positions=depths_for_axis, depth_labels=labels_for_axis, x_bounds=ctx.x_bounds, y_bounds=ctx.y_bounds)

    aux_handles = [Line2D([0], [0], color=VIRTUAL_FRAME_COLOR, linestyle="--", linewidth=VIRTUAL_FRAME_LINEWIDTH, label="Virtual slice")]
    leg = ax.legend(handles=aux_handles, loc="upper left", bbox_to_anchor=(0.01, 0.96), frameon=False)
    ax.add_artist(leg)

    if str(display_mode).lower() == "gene":
        add_gene_colorbar(fig, ax, result["norm"], label=f"log1p({MARKER_GENE})")
        subtitle = f"Final reconstruction of {MARKER_GENE}"
    else:
        add_domain_legend(fig, result["color_map"], title=DOMAIN_KEY)
        subtitle = f"Final reconstruction of {DOMAIN_KEY}"

    ax.set_title("Final reconstructed stack\n" + subtitle, pad=15, fontsize=12.2, fontweight="bold")
    fig.tight_layout()
    return save_or_show(fig, outdir=outdir, stem=f"{stem}_{pair_name}_{display_mode}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    plot_figure1_observed_stack_3d(display_mode=DISPLAY_MODE)
    plot_figure2_reserved_anchors_3d(display_mode=DISPLAY_MODE, pair_name=ANCHOR_PAIR_NAME)
    plot_figure3_transition_truncation_3d(display_mode=DISPLAY_MODE, pair_name=ANCHOR_PAIR_NAME)

    try:
        plot_figure4_core_reconstruction_3d(display_mode=DISPLAY_MODE, pair_name=ANCHOR_PAIR_NAME)
    except Exception as exc:
        print(f"[SKIP] Figure 4 was not generated: {exc}")

    try:
        plot_figure5_final_reconstructed_stack_3d(display_mode=DISPLAY_MODE, pair_name=ANCHOR_PAIR_NAME)
    except Exception as exc:
        print(f"[SKIP] Figure 5 was not generated: {exc}")


if __name__ == "__main__":
    main()
