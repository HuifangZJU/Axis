#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLPFC spatially matched analysis pipeline visualization.

Generates three figures with a shared 3D visual language:

Figure 1
    Four original DLPFC slices in 3D.

Figure 2
    Slice 0 as a standalone reference plane, followed by the measured and
    spatially matched states for slices 1-3, with pairwise transition fields.

Figure 3
    Reference and spatially matched slices followed by the integrated tissue.

Important implementation choices
--------------------------------
1. Point rendering and transition-field rendering have independent controls.
2. The display positions in 3D (AXIS_POSITIONS) are independent from true
   biological depths (BIOLOGICAL_DEPTHS), which are used only for labels.
3. Figure 2 exposes separate maximum opacity controls for measured/matched states
   and applies linear gene-value-dependent alpha for transition-field visibility.
4. Figure 3 has independent camera controls and displays reference/matched slices
   before the integrated population-level tissue.
5. The old anndata/pandas ``_clean_uns`` incompatibility is handled by a
   targeted retry that temporarily patches the exact global used by
   ``anndata.read_h5ad``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from axis.coordinate_utils import find_inverse_warp_coords
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


# =============================================================================
# USER CONFIGURATION
# =============================================================================

ROOT = "/media/huifang/data/registration/DLPFC/huifang"
GROUP_ID = 1
SAMPLE_IDS: Tuple[int, ...] = (0, 1, 2, 3)
REFERENCE_ID = 0
PAIRWISE_SUBDIR = "pairwise_output"
COORD_KEY = "spatial"

# Keep the current visual spacing used in the original script.
AXIS_POSITIONS: Tuple[float, ...] = (0.0, 250.0, 500.0, 750.0)

# True anatomical depths used for the plane labels only.
BIOLOGICAL_DEPTHS: Mapping[int, float] = {
    0: 0.0,
    1: 10.0,
    2: 310.0,
    3: 320.0,
}

# Slice coloring: gene name or obs-column name.
SLICE_COLOR_KEY: Optional[str] = "PCP4"
USE_LOG1P_FOR_GENE = True

# -----------------------------------------------------------------------------
# Point controls -- shared visual scale, but independently adjustable by figure.
# -----------------------------------------------------------------------------
ORIGINAL_POINT_SIZE = 7.0
ORIGINAL_POINT_ALPHA = 0.8

MATCHED_POINT_SIZE = 4.0

# Figure 2: measured slices are lighter; reference and matched slices share alpha.
FIG2_MEASURED_POINT_ALPHA = 0.4
FIG2_MATCHED_POINT_ALPHA = 0.95

# Figure 2 matched-state spot boundary.
FIG2_MATCHED_POINT_EDGE_COLOR = "0.55"
FIG2_MATCHED_POINT_EDGE_WIDTH = 0.15

# Gene-map color normalization follows the dense-reconstruction pipeline:
#   - log1p values (controlled by USE_LOG1P_FOR_GENE)
#   - fixed Reds colormap
#   - linear Normalize(vmin=0, vmax=99.5th percentile)
# Figure 2 additionally uses a LINEAR expression-dependent alpha so that
# low-expression spots remain translucent and the transition field stays visible.
GENE_NORM_PERCENTILE = 99.9
FIG2_GENE_ALPHA_FLOOR = 0.3

FINAL_POINT_SIZE = 30.0
FINAL_POINT_ALPHA = 0.55

# -----------------------------------------------------------------------------
# Transition-field controls -- independent from point rendering.
# -----------------------------------------------------------------------------
DOMAIN_KEY = "layer_guess_reordered"
# TRAJECTORY_DOMAINS: Optional[Sequence[str]] = None
TRAJECTORY_DOMAINS: Optional[Sequence[str]] = ("Layer5",)
RANDOM_SEED = 0

# -----------------------------------------------------------------------------
# Plane / annotation controls.
# -----------------------------------------------------------------------------
# All plane frames use one visual identity. Geometry distinguishes state type:
#   real/reference/integrated -> solid frame
#   matched/generated         -> dashed frame
#   Figure-2 projection-only original slices -> no frame
PLANE_FACE_ALPHA = 0.02
PLANE_FRAME_COLOR = "0.20"
PLANE_FRAME_ALPHA = 0.72
PLANE_FRAME_LINEWIDTH = 0.95
REAL_PLANE_LINESTYLE = "-"
MATCHED_PLANE_LINESTYLE = (0, (4.0, 3.0))

# Same label geometry/style used in the earlier dense-reconstruction workflow.
PLANE_LABEL_FONTSIZE = 26.0
PLANE_LABEL_COLOR = "black"

# Initial display depth assigned when the final record is loaded.
# Figure 3 later places this single integrated plane at display depth 0.
FINAL_DISPLAY_DEPTH = 0.5 * (min(AXIS_POSITIONS) + max(AXIS_POSITIONS))
# FINAL_PLANE_LABEL = "Integrated virtual tissue"
FINAL_PLANE_LABEL = None
# -----------------------------------------------------------------------------
# 3D view
# -----------------------------------------------------------------------------
ELEV = 10
AZIM = -120
PROJECTION_TYPE = "ortho"

# Figure 3 can be viewed independently from Figures 1-2.
FIG3_ELEV = 10
FIG3_AZIM = -120
FIG3_PROJECTION_TYPE = "ortho"

# Keep height fixed; width grows with the number of displayed planes.
FIGURE_HEIGHT = 7.8
FIGURE_MIN_WIDTH = 7.2
FIGURE_WIDTH_PER_EXTRA_PLANE = 1.55

# -----------------------------------------------------------------------------
# Figure 1: four measured sections
# -----------------------------------------------------------------------------
FIG1_SLICE_GAP = 150.0
FIG1_XY_PAD_RATIO = 0.035
FIG1_DEPTH_PAD_RATIO = 0.035
FIG1_DEPTH_ASPECT_SCALE = 1.35

FIG1_DISPLAY_POSITIONS = tuple(
    i * FIG1_SLICE_GAP
    for i in range(len(SAMPLE_IDS))
)

# -----------------------------------------------------------------------------
# Figure 2: three non-reference measured/matched pairs
# -----------------------------------------------------------------------------
FIG2_WITHIN_PAIR_GAP = 150.0
FIG2_BETWEEN_PAIR_GAP = 180.0
FIG2_REFERENCE_TO_PAIR_GAP = 220.0

FIG2_XY_PAD_RATIO = 0.035
FIG2_DEPTH_PAD_RATIO = 0.025
FIG2_DEPTH_ASPECT_SCALE = 1.35

# -----------------------------------------------------------------------------
# Figure 3: reference + matched slices, followed by the integrated tissue
# -----------------------------------------------------------------------------
FIG3_SINGLE_SLICE_GAP = 50.0
FIG3_INTEGRATED_GAP = 300.0
FIG3_SINGLE_POINT_ALPHA = 0.58
FIG3_SINGLE_POINT_SIZE = 8

# Retained for compatibility with style_3d_axis; ignored for multi-plane Figure 3.
FIG3_SINGLE_PLANE_DEPTH_SPAN = 1.0

FIG3_XY_PAD_RATIO = 0.02
FIG3_DEPTH_PAD_RATIO = 0.025
FIG3_DEPTH_ASPECT_SCALE = 1.15








# -----------------------------------------------------------------------------
# Color style
# -----------------------------------------------------------------------------

# Reference slice and all spatially matched states use Reds.
REFERENCE_CMAP_NAME = "Reds"

# Original/measured slice colors.
# These can be changed independently here.
SLICE_CMAP_NAMES = {
    0: "Reds",      # reference
    1: "Reds",
    2: "Reds",
    3: "Reds",
}

# Fallback; Figure 4 continues to use Reds.
GENE_CMAP_NAME = "Reds"

# -----------------------------------------------------------------------------
# Fused original -> matched transition visualization
# -----------------------------------------------------------------------------

PAIRWISE_TRANSITION_COLOR_LEVEL = 0.72
PAIRWISE_TRANSITION_LINEWIDTH = 2.6
PAIRWISE_TRANSITION_ALPHA = 0.05
PAIRWISE_TRANSITION_STEPS = 32

# Number of transition trajectories per original/matched pair.
PAIRWISE_MAX_FIELD_SEEDS = 500

PAIRWISE_PLANE_LABEL_FONTSIZE = 15.0
CATEGORICAL_CMAP_NAME = "tab10"
# -----------------------------------------------------------------------------
# Files.
# -----------------------------------------------------------------------------
VIRTUAL_PATH = Path(
    "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/"
    "dlpfc_z_group1_grid80_spatial1/integration/"
    "integrated_dlpfc_z_ref_0_visium_ot.h5ad"
)

MATCHED_STATE_TEMPLATE = (
    "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/"
    "dlpfc_z_group1_grid80_spatial1/analysis/virtual_dlpfc_z_{depth}.h5ad"
)
MATCHED_STATE_DEPTHS: Tuple[str, ...] = (
    "0.000",
    "10.000",
    "310.000",
    "320.000",
)

# -----------------------------------------------------------------------------
# Output.
# -----------------------------------------------------------------------------
OUT_DIR = Path(
    "/media/huifang/data/registration/paper_analysis/Figure4_dlfpc/g_pipeline"
)
FIGURE_MODE = "both"  # "save", "show", or "both"
DPI = 400

# Choose which figures to render. Examples:
# FIGURES_TO_RUN = ("figure2",)
# FIGURES_TO_RUN = ("figure1", "figure3")
FIGURES_TO_RUN: Tuple[str, ...] = ("figure2",)


def get_slice_expression_cmap(sample_id: int):
    """Colormap assigned to one original/measured section."""
    cmap_name = SLICE_CMAP_NAMES.get(
        int(sample_id),
        GENE_CMAP_NAME,
    )
    return plt.get_cmap(cmap_name)


# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class PlaneRecord:
    name: str
    display_depth: float
    biological_depth: Optional[float]
    coords: np.ndarray
    values: Optional[np.ndarray]
    value_kind: str
    label: Optional[str]


# =============================================================================
# STYLE / IO
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
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
            "figure.dpi": 120,
        }
    )


def requested(figure_id: str) -> bool:
    chosen = {str(x).strip().lower() for x in FIGURES_TO_RUN}
    return "all" in chosen or figure_id.lower() in chosen


def save_or_show(fig: plt.Figure, stem: str) -> None:
    mode = str(FIGURE_MODE).lower()
    if mode not in {"save", "show", "both"}:
        raise ValueError("FIGURE_MODE must be 'save', 'show', or 'both'.")

    if mode in {"save", "both"}:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        png_path = OUT_DIR / f"{stem}.png"
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
        print(f"[SAVE] {png_path}")

    if mode in {"show", "both"}:
        plt.show()

    plt.close(fig)


def safe_read_h5ad(path: Union[str, Path]):
    """Read H5AD robustly and consistently apply the optional ``valid`` mask."""
    path = Path(path)

    try:
        data = ad.read_h5ad(str(path))
    except TypeError as exc:
        if "Unordered Categoricals can only compare equality or not" not in str(exc):
            raise

        read_globals = getattr(ad.read_h5ad, "__globals__", None)
        if not isinstance(read_globals, dict) or "_clean_uns" not in read_globals:
            raise

        old_clean_uns = read_globals["_clean_uns"]
        print(
            f"[WARN] {path.name}: legacy anndata _clean_uns failed; "
            "retrying with that cleanup step disabled."
        )
        try:
            read_globals["_clean_uns"] = lambda d: None
            data = ad.read_h5ad(str(path))
        finally:
            read_globals["_clean_uns"] = old_clean_uns

    if "valid" in data.obs.columns:
        data = data[data.obs["valid"].to_numpy().astype(bool)].copy()

    return data


# =============================================================================
# PATH / DATA HELPERS
# =============================================================================

def parse_h5ad_path(root: str, group_id: int, sample_id: int) -> str:
    return os.path.join(root, "h5ad", f"{group_id}_{sample_id}.h5ad")


def parse_network_npz_path(root: str, group_id: int, sample_id: int) -> str:
    return os.path.join(root, "network_input", f"{group_id}_{sample_id}_validation.npz")


def find_group_dir(
    root: str,
    group_id: int,
    pairwise_subdir: str = PAIRWISE_SUBDIR,
) -> str:
    p1 = os.path.join(root, pairwise_subdir, f"group_{group_id}")
    p2 = os.path.join(root, f"group_{group_id}")
    if os.path.isdir(p1):
        return p1
    if os.path.isdir(p2):
        return p2
    raise FileNotFoundError(f"Cannot find pairwise folder for group {group_id}.")


def parse_flow_path(
    root: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
    pairwise_subdir: str = PAIRWISE_SUBDIR,
) -> str:
    group_dir = find_group_dir(root, group_id, pairwise_subdir=pairwise_subdir)
    return os.path.join(group_dir, f"{fixed_id}_{moving_id}_flow.npy")


def load_npz_coord(npz_path: str) -> np.ndarray:
    with np.load(npz_path) as data:
        if "coord" not in data:
            raise KeyError(f"'coord' not found in {npz_path}")
        coords = np.asarray(data["coord"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected [N,2+] coordinates, got {coords.shape}: {npz_path}")
    return coords[:, :2]


def load_original_slice(
    sample_id: int,
    reference_id: int = REFERENCE_ID,
):
    """Load the same per-slice coordinates used by the original pipeline."""
    adata = safe_read_h5ad(parse_h5ad_path(ROOT, GROUP_ID, sample_id))

    if COORD_KEY in adata.obsm:
        coords = np.asarray(adata.obsm[COORD_KEY], dtype=np.float32)[:, :2]
        return adata, coords

    if sample_id == reference_id:
        coords = load_npz_coord(parse_network_npz_path(ROOT, GROUP_ID, sample_id))
        return adata, coords

    group_dir = find_group_dir(ROOT, GROUP_ID, pairwise_subdir=PAIRWISE_SUBDIR)
    warped_path = os.path.join(group_dir, f"{reference_id}_{sample_id}_warped_coor.npy")
    if os.path.exists(warped_path):
        coords = np.asarray(np.load(warped_path), dtype=np.float32)[:, :2]
        return adata, coords

    raise FileNotFoundError(
        f"Could not find coordinates for group={GROUP_ID}, sample={sample_id}."
    )


def get_spatial_coords(adata: ad.AnnData, spatial_key: str = COORD_KEY) -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm[{spatial_key!r}] not found; available keys: {list(adata.obsm.keys())}"
        )
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected obsm[{spatial_key!r}] with shape [N,2+], got {coords.shape}")
    return coords[:, :2]


# =============================================================================
# VALUE EXTRACTION / COLORS
# =============================================================================

def extract_gene_values(adata, gene: str, use_log1p: bool = True) -> np.ndarray:
    if gene not in adata.var_names:
        raise KeyError(f"Gene {gene!r} not found in var_names.")

    values = adata[:, gene].X
    if hasattr(values, "toarray"):
        values = values.toarray().ravel()
    else:
        values = np.asarray(values).ravel()

    values = np.asarray(values, dtype=float)
    values[~np.isfinite(values)] = 0.0

    if use_log1p:
        values = np.log1p(np.clip(values, 0.0, None))
    return values


def canonical_category(value) -> str:
    if value is None:
        return "NA"
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return "NA"
    return text


def load_display_values(
    adata,
    color_key: Optional[str],
    use_log1p_gene: bool = USE_LOG1P_FOR_GENE,
):
    if color_key is None:
        return None, "none"

    if color_key in adata.obs.columns:
        values = np.asarray(
            [canonical_category(x) for x in adata.obs[color_key].to_numpy()],
            dtype=str,
        )
        return values, "categorical"

    if color_key in adata.var_names:
        return extract_gene_values(adata, color_key, use_log1p=use_log1p_gene), "continuous"

    raise KeyError(f"{color_key!r} is neither an obs column nor a gene name.")


def make_expression_cmap():
    """Gene spatial maps use the fixed Reds colormap."""
    return plt.get_cmap(GENE_CMAP_NAME)


def build_category_palette(values_by_source: Mapping[str, np.ndarray]):
    categories = sorted(
        {str(v) for values in values_by_source.values() for v in values}
    )
    cmap = plt.get_cmap(CATEGORICAL_CMAP_NAME, max(len(categories), 1))
    return {category: cmap(i) for i, category in enumerate(categories)}


def build_category_style(record_groups: Sequence[Sequence[PlaneRecord]]):
    """Build one stable categorical palette across all plots.

    Category colors remain shared so the same domain/category keeps the same
    color across figures. Gene-expression normalization is intentionally NOT
    built here; it is computed independently for each figure.
    """
    values_by_source: Dict[str, np.ndarray] = {}
    kinds = set()

    for records in record_groups:
        for record in records:
            kinds.add(record.value_kind)
            if record.values is not None and record.value_kind == "categorical":
                values_by_source[record.name] = record.values

    non_none_kinds = {k for k in kinds if k != "none"}
    if len(non_none_kinds) > 1:
        raise ValueError(
            "SLICE_COLOR_KEY resolves to different value types across files: "
            f"{sorted(non_none_kinds)}"
        )

    if non_none_kinds == {"categorical"}:
        return build_category_palette(values_by_source)
    return None


def build_figure_value_style(records: Sequence[PlaneRecord]):
    """Build the gene-expression display style used by the dense-reconstruction plot.

    Visualization rule:
      1. expression values are already log1p-transformed at loading time when
         USE_LOG1P_FOR_GENE=True;
      2. all supplied measured records share one linear normalization;
      3. vmin is fixed at 0;
      4. vmax is the 99.5th percentile of finite measured values;
      5. the colormap is the standard Matplotlib Reds map.

    Generated/matched states should reuse this normalization rather than define
    a new color scale.
    """
    expression_cmap = make_expression_cmap()
    kinds = {record.value_kind for record in records if record.value_kind != "none"}

    if not kinds or kinds == {"categorical"}:
        return None, expression_cmap

    if kinds != {"continuous"}:
        raise ValueError(
            "Figure records contain mixed value types: "
            f"{sorted(kinds)}"
        )

    finite_parts = []
    for record in records:
        if record.value_kind != "continuous" or record.values is None:
            continue
        arr = np.asarray(record.values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_parts.append(arr)

    if not finite_parts:
        return Normalize(vmin=0.0, vmax=1.0), expression_cmap

    combined = np.concatenate(finite_parts)
    vmax = float(
        np.nanpercentile(
            combined,
            GENE_NORM_PERCENTILE,
        )
    )

    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(combined))

    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    return (
        Normalize(
            vmin=0.0,
            vmax=max(vmax, 1e-6),
        ),
        expression_cmap,
    )


# =============================================================================
# FLOW / TRAJECTORY HELPERS
# =============================================================================

def ensure_flow_hw2(flow: np.ndarray) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3:
        raise ValueError(f"Flow must be 3D, got {flow.shape}")
    if flow.shape[-1] == 2:
        return flow
    if flow.shape[0] == 2:
        return np.moveaxis(flow, 0, -1)
    raise ValueError(f"Cannot interpret flow shape {flow.shape} as [H,W,2]")


def inverse_warp_points(points_xy: np.ndarray, flow_hw2: np.ndarray) -> np.ndarray:
    warped = find_inverse_warp_coords(points_xy, flow_hw2)
    if isinstance(warped, (tuple, list)):
        warped = warped[0]
    warped = np.asarray(warped, dtype=np.float32)
    if warped.ndim != 2 or warped.shape[1] < 2:
        raise ValueError(f"Unexpected inverse-warp result shape: {warped.shape}")
    return warped[:, :2]




# =============================================================================
# RECORD LOADING
# =============================================================================

def plane_position_map() -> Dict[int, float]:
    if len(SAMPLE_IDS) != len(FIG1_DISPLAY_POSITIONS):
        raise ValueError(
            "SAMPLE_IDS and FIG1_DISPLAY_POSITIONS must have equal length."
        )

    return {
        sid: float(depth)
        for sid, depth in zip(
            SAMPLE_IDS,
            FIG1_DISPLAY_POSITIONS,
        )
    }

def load_original_records() -> List[PlaneRecord]:
    positions = plane_position_map()
    records: List[PlaneRecord] = []

    for sid in SAMPLE_IDS:
        adata, coords = load_original_slice(sid)
        n_keep = min(adata.n_obs, len(coords))
        adata = adata[:n_keep].copy()
        coords = coords[:n_keep]
        values, kind = load_display_values(adata, SLICE_COLOR_KEY)
        biological_depth = float(BIOLOGICAL_DEPTHS[sid])

        records.append(
            PlaneRecord(
                name=f"original_{sid}",
                display_depth=positions[sid],
                biological_depth=biological_depth,
                coords=coords,
                values=values,
                value_kind=kind,
                label=f"Slice {sid+1}",
            )
        )

    return records


def load_matched_records() -> List[PlaneRecord]:
    if len(MATCHED_STATE_DEPTHS) != len(SAMPLE_IDS):
        raise ValueError("MATCHED_STATE_DEPTHS must match SAMPLE_IDS length.")

    positions = plane_position_map()
    records: List[PlaneRecord] = []

    for sid, depth_text in zip(SAMPLE_IDS, MATCHED_STATE_DEPTHS):
        path = Path(MATCHED_STATE_TEMPLATE.format(depth=depth_text))
        if not path.exists():
            raise FileNotFoundError(f"Matched state not found: {path}")

        adata = safe_read_h5ad(path)
        coords = get_spatial_coords(adata, spatial_key=COORD_KEY)
        values, kind = load_display_values(adata, SLICE_COLOR_KEY)
        biological_depth = float(BIOLOGICAL_DEPTHS[sid])

        records.append(
            PlaneRecord(
                name=f"matched_{sid}",
                display_depth=positions[sid],
                biological_depth=biological_depth,
                coords=coords,
                values=values,
                value_kind=kind,
                label=f"Slice {sid+1}",
            )
        )

    return records


def load_final_record() -> PlaneRecord:
    if not VIRTUAL_PATH.exists():
        raise FileNotFoundError(f"VIRTUAL_PATH does not exist: {VIRTUAL_PATH}")

    adata = safe_read_h5ad(VIRTUAL_PATH)
    coords = get_spatial_coords(adata, spatial_key=COORD_KEY)
    values, kind = load_display_values(adata, SLICE_COLOR_KEY)

    return PlaneRecord(
        name="final_integrated",
        display_depth=float(FINAL_DISPLAY_DEPTH),
        biological_depth=None,
        coords=coords,
        values=values,
        value_kind=kind,
        label=FINAL_PLANE_LABEL,
    )


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def combined_xy_bounds(record_groups: Sequence[Sequence[PlaneRecord]]):
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for records in record_groups:
        for record in records:
            coords = np.asarray(record.coords, dtype=float)
            finite = np.all(np.isfinite(coords[:, :2]), axis=1)
            if np.any(finite):
                xs.append(coords[finite, 0])
                ys.append(coords[finite, 1])

    if not xs:
        raise ValueError("No finite coordinates are available for plotting.")

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return (
        (float(np.nanmin(x)), float(np.nanmax(x))),
        (float(np.nanmin(y)), float(np.nanmax(y))),
    )


def draw_plane_frame(
    ax,
    depth: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    *,
    linestyle,
    fill: bool,
) -> None:
    """Draw a consistently colored tissue-plane frame."""
    x0, x1 = x_bounds
    y0, y1 = y_bounds

    if fill:
        vertices = [[
            (depth, x0, y0),
            (depth, x1, y0),
            (depth, x1, y1),
            (depth, x0, y1),
        ]]
        plane = Poly3DCollection(
            vertices,
            facecolors=(1.0, 1.0, 1.0, PLANE_FACE_ALPHA),
            edgecolors="none",
        )
        ax.add_collection3d(plane)

    frame = np.asarray(
        [
            [depth, x0, y0],
            [depth, x1, y0],
            [depth, x1, y1],
            [depth, x0, y1],
            [depth, x0, y0],
        ],
        dtype=float,
    )
    ax.plot(
        frame[:, 0],
        frame[:, 1],
        frame[:, 2],
        color=PLANE_FRAME_COLOR,
        alpha=PLANE_FRAME_ALPHA,
        linewidth=PLANE_FRAME_LINEWIDTH,
        linestyle=linestyle,
    )


def draw_real_slice_frame(
    ax,
    depth: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
) -> None:
    draw_plane_frame(
        ax,
        depth,
        x_bounds,
        y_bounds,
        linestyle=REAL_PLANE_LINESTYLE,
        fill=True,
    )


def draw_matched_slice_frame(
    ax,
    depth: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
) -> None:
    draw_plane_frame(
        ax,
        depth,
        x_bounds,
        y_bounds,
        linestyle=MATCHED_PLANE_LINESTYLE,
        fill=False,
    )


def record_frame_mode(
    record: PlaneRecord,
    *,
    hide_projection_source_frames: bool = False,
) -> str:
    """Return ``real``, ``matched``, or ``none`` for one displayed record."""
    if record.name.startswith("matched_"):
        return "matched"

    if (
        hide_projection_source_frames
        and record.name.startswith("original_")
        and record.name != f"original_{REFERENCE_ID}"
    ):
        return "none"

    return "real"


def add_plane_label(
    ax,
    depth: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    text: str,
    color: str = PLANE_LABEL_COLOR,
    fontsize: float = PLANE_LABEL_FONTSIZE,
) -> None:
    """Same direct plane-label geometry used in the earlier workflow."""
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    x_text = x0 + 0.01 * (x1 - x0)
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
        zorder=500,
    )


def plot_expression_points(
    ax,
    record: PlaneRecord,
    norm: Normalize,
    cmap,
    point_size: float,
    point_alpha: float,
    linear_alpha_by_expression: bool = False,
    alpha_floor: float = 0.08,
    edge_color="none",
    edge_width: float = 0.0,
) -> None:
    values = np.asarray(record.values, dtype=float)
    coords = np.asarray(record.coords, dtype=float)
    valid = np.isfinite(values) & np.all(np.isfinite(coords), axis=1)
    if not np.any(valid):
        return

    values = values[valid]
    coords = coords[valid]
    order = np.argsort(values)

    # IMPORTANT: keep the gene color mapping strictly linear.
    # No PowerNorm/gamma transform is applied; the displayed red intensity
    # therefore follows the normalized log1p gene reads directly.
    scaled = np.clip(np.asarray(norm(values), dtype=float), 0.0, 1.0)
    rgba = np.asarray(cmap(scaled), dtype=float)

    if linear_alpha_by_expression:
        # Figure 2 only: make opacity increase linearly with the same normalized
        # gene value used for color.  point_alpha is the maximum opacity and
        # alpha_floor keeps zero/low-expression spots faintly visible.
        alpha_floor = float(np.clip(alpha_floor, 0.0, 1.0))
        alpha_ceiling = float(np.clip(point_alpha, alpha_floor, 1.0))
        rgba[:, 3] = (
            alpha_floor
            + (alpha_ceiling - alpha_floor) * scaled
        )
    else:
        rgba[:, 3] = float(point_alpha)

    ax.scatter(
        np.full(len(coords), record.display_depth)[order],
        coords[order, 0],
        coords[order, 1],
        c=rgba[order],
        s=point_size,
        edgecolors=edge_color,
        linewidths=edge_width,
        depthshade=False,
        rasterized=True,
    )


def plot_categorical_points(
    ax,
    record: PlaneRecord,
    color_map: Mapping[str, object],
    point_size: float,
    point_alpha: float,
) -> None:
    coords = np.asarray(record.coords, dtype=float)
    values = np.asarray(record.values, dtype=str)
    valid = np.all(np.isfinite(coords), axis=1)
    if not np.any(valid):
        return

    coords = coords[valid]
    values = values[valid]
    colors = [color_map[str(value)] for value in values]

    ax.scatter(
        np.full(len(coords), record.display_depth),
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=point_size,
        alpha=point_alpha,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )


def plot_records(
    ax,
    records: Sequence[PlaneRecord],
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    category_palette,
    continuous_norm,
    expression_cmap,
    point_size: float,
    point_alpha: float,
    record_point_alphas: Optional[Mapping[str, float]] = None,
    plane_label_fontsize: float = PLANE_LABEL_FONTSIZE,
    linear_gene_alpha: bool = False,
    gene_alpha_floor: float = 0.08,
    hide_projection_source_frames: bool = False,
    matched_point_edge_color=None,
    matched_point_edge_width: float = 0.0,
) -> None:
    """Render records with state-aware plane frames and point styling."""
    for record in records:
        this_point_alpha = (
            float(record_point_alphas.get(record.name, point_alpha))
            if record_point_alphas is not None
            else float(point_alpha)
        )

        frame_mode = record_frame_mode(
            record,
            hide_projection_source_frames=hide_projection_source_frames,
        )
        if frame_mode == "real":
            draw_real_slice_frame(
                ax,
                record.display_depth,
                x_bounds,
                y_bounds,
            )
        elif frame_mode == "matched":
            draw_matched_slice_frame(
                ax,
                record.display_depth,
                x_bounds,
                y_bounds,
            )

        if record.label:
            add_plane_label(
                ax,
                record.display_depth,
                x_bounds,
                y_bounds,
                record.label,
                fontsize=plane_label_fontsize,
            )

        is_matched = record.name.startswith("matched_")
        edge_color = (
            matched_point_edge_color
            if is_matched and matched_point_edge_color is not None
            else "none"
        )
        edge_width = (
            float(matched_point_edge_width)
            if is_matched and matched_point_edge_color is not None
            else 0.0
        )

        if record.value_kind == "continuous":
            plot_expression_points(
                ax,
                record,
                norm=continuous_norm,
                cmap=expression_cmap,
                point_size=point_size,
                point_alpha=this_point_alpha,
                linear_alpha_by_expression=linear_gene_alpha,
                alpha_floor=gene_alpha_floor,
                edge_color=edge_color,
                edge_width=edge_width,
            )

        elif record.value_kind == "categorical":
            plot_categorical_points(
                ax,
                record,
                color_map=category_palette,
                point_size=point_size,
                point_alpha=this_point_alpha,
            )

        else:
            coords = np.asarray(record.coords, dtype=float)
            valid = np.all(np.isfinite(coords), axis=1)
            coords = coords[valid]
            ax.scatter(
                np.full(len(coords), record.display_depth),
                coords[:, 0],
                coords[:, 1],
                c="0.5",
                s=point_size,
                alpha=this_point_alpha,
                linewidths=0,
                depthshade=False,
                rasterized=True,
            )


def make_pairwise_display_records(
    original_records: Sequence[PlaneRecord],
    matched_records: Sequence[PlaneRecord],
):
    """
    Figure 2 layout:

        Slice 0 (reference),
        matched slice 1, measured slice 1,
        matched slice 2, measured slice 2,
        matched slice 3, measured slice 3

    The reference slice is shown once as a standalone plane because it already
    defines the common anatomical frame. No redundant matched copy is drawn.
    """

    if len(original_records) != len(SAMPLE_IDS):
        raise ValueError(
            "Unexpected number of original records."
        )

    if len(matched_records) != len(SAMPLE_IDS):
        raise ValueError(
            "Unexpected number of matched records."
        )

    records = []
    matched_positions = {}
    original_positions = {}

    reference_depth = 0.0
    visible_pair_index = 0

    for sid, matched, original in zip(
        SAMPLE_IDS,
        matched_records,
        original_records,
    ):
        sid = int(sid)

        # ------------------------------------------------------
        # Standalone reference plane
        # ------------------------------------------------------
        if sid == int(REFERENCE_ID):
            records.append(
                replace(
                    original,
                    display_depth=reference_depth,
                    label=(
                        f"Slice {sid+1}\n"
                        "(reference)"
                    ),
                )
            )
            continue

        # ------------------------------------------------------
        # Measured -> matched pair for each non-reference slice
        # ------------------------------------------------------
        pair_start = (
            reference_depth
            + FIG2_REFERENCE_TO_PAIR_GAP
            + visible_pair_index
            * (
                FIG2_WITHIN_PAIR_GAP
                + FIG2_BETWEEN_PAIR_GAP
            )
        )

        matched_depth = float(pair_start)

        original_depth = float(
            pair_start
            + FIG2_WITHIN_PAIR_GAP
        )

        matched_positions[sid] = matched_depth
        original_positions[sid] = original_depth

        records.append(
            replace(
                matched,
                display_depth=matched_depth,
                label=(
                    f"Matched\n"
                    f"slice {sid+1}"
                ),
            )
        )

        records.append(
            replace(
                original,
                display_depth=original_depth,
                # label=(
                #     f"Measured\n"
                #     f"slice {sid+1}"
                # ),
                label=None,
            )
        )

        visible_pair_index += 1

    return (
        records,
        original_positions,
        matched_positions,
    )


def build_pairwise_record_alphas(
    records: Sequence[PlaneRecord],
) -> Dict[str, float]:
    """
    Figure 2 opacity map.

    The standalone reference and all matched slices share the higher alpha.
    The remaining original/measured slices use the lighter measured alpha.
    """
    alphas: Dict[str, float] = {}

    reference_name = f"original_{REFERENCE_ID}"

    for record in records:
        if record.name.startswith("matched_") or record.name == reference_name:
            alphas[record.name] = float(FIG2_MATCHED_POINT_ALPHA)
        else:
            alphas[record.name] = float(FIG2_MEASURED_POINT_ALPHA)

    return alphas


def smooth_pairwise_transition_curve(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    start_depth: float,
    end_depth: float,
    n_steps: int = PAIRWISE_TRANSITION_STEPS,
) -> np.ndarray:
    """
    Smooth trajectory from an original spot to its spatially matched position.

    Endpoints are exact. Smoothstep only controls how the trajectory bends
    through the display-depth dimension.
    """

    t = np.linspace(
        0.0,
        1.0,
        int(n_steps),
        dtype=float,
    )

    # Cubic smoothstep.
    s = t * t * (3.0 - 2.0 * t)

    start_xy = np.asarray(start_xy, dtype=float)
    end_xy = np.asarray(end_xy, dtype=float)

    xy = (
        (1.0 - s[:, None]) * start_xy[None, :]
        + s[:, None] * end_xy[None, :]
    )

    depth = (
        (1.0 - t) * float(start_depth)
        + t * float(end_depth)
    )

    return np.column_stack(
        [
            depth,
            xy[:, 0],
            xy[:, 1],
        ]
    ).astype(np.float32)



def build_pairwise_state_transition_curves(
    original_records: Sequence[PlaneRecord],
    matched_records: Sequence[PlaneRecord],
    original_positions: Mapping[int, float],
    matched_positions: Mapping[int, float],
):
    """
    Build trajectories from each original measured state into the
    common anatomical frame.

    Important:
    matched states may be represented on a different/common spatial
    support and therefore do not need to contain the same number of
    points as the original measured section.

    For non-reference slices, endpoints are obtained directly from the
    AXIS pairwise deformation field mapping the original slice into the
    reference frame.
    """

    rng = np.random.default_rng(RANDOM_SEED)
    curves_by_sid = {}

    for sid, original in zip(
        SAMPLE_IDS,
        original_records,
    ):
        sid = int(sid)
        if sid==0:
            continue

        original_xy = np.asarray(
            original.coords,
            dtype=np.float32,
        )

        valid = np.all(
            np.isfinite(original_xy),
            axis=1,
        )

        # ------------------------------------------------------
        # Optional anatomical-domain filtering
        # ------------------------------------------------------
        if TRAJECTORY_DOMAINS is not None:

            adata, _ = load_original_slice(sid)

            if DOMAIN_KEY not in adata.obs.columns:
                raise KeyError(
                    f"{DOMAIN_KEY!r} not found for sample {sid}"
                )

            labels = (
                adata.obs[DOMAIN_KEY]
                .astype(str)
                .to_numpy()
            )

            n = min(
                len(labels),
                len(original_xy),
            )

            domain_mask = np.zeros(
                len(original_xy),
                dtype=bool,
            )

            domain_mask[:n] = np.isin(
                labels[:n],
                np.asarray(
                    TRAJECTORY_DOMAINS,
                    dtype=str,
                ),
            )

            valid &= domain_mask

        source_indices = np.flatnonzero(valid)

        # ------------------------------------------------------
        # Map original coordinates into the reference frame
        # ------------------------------------------------------
        if sid == int(REFERENCE_ID):

            # The reference is already in the common frame.
            warped_xy = original_xy.copy()
            warped_valid = np.all(
                np.isfinite(warped_xy),
                axis=1,
            )

        else:

            # fixed = reference; moving = current slice
            flow_path = parse_flow_path(
                ROOT,
                GROUP_ID,
                int(REFERENCE_ID),
                sid,
                PAIRWISE_SUBDIR,
            )

            if not os.path.exists(flow_path):
                raise FileNotFoundError(
                    f"Missing flow for sample {sid} -> "
                    f"reference {REFERENCE_ID}: {flow_path}"
                )

            flow = ensure_flow_hw2(
                np.load(flow_path)
            )

            warped_xy = np.full_like(
                original_xy,
                np.nan,
                dtype=np.float32,
            )

            if len(source_indices) > 0:

                transformed = inverse_warp_points(
                    original_xy[source_indices],
                    flow,
                )

                warped_xy[source_indices] = transformed

            warped_valid = np.all(
                np.isfinite(warped_xy),
                axis=1,
            )

            # Keep only points whose transformed coordinates lie
            # within the flow/reference frame.
            height, width = flow.shape[:2]

            warped_valid &= (
                (warped_xy[:, 0] >= 0)
                & (warped_xy[:, 0] < width)
                & (warped_xy[:, 1] >= 0)
                & (warped_xy[:, 1] < height)
            )

        valid_indices = np.flatnonzero(
            valid & warped_valid
        )

        # ------------------------------------------------------
        # Subsample only for visualization
        # ------------------------------------------------------
        if (
            PAIRWISE_MAX_FIELD_SEEDS is not None
            and len(valid_indices) > PAIRWISE_MAX_FIELD_SEEDS
        ):
            valid_indices = np.sort(
                rng.choice(
                    valid_indices,
                    size=PAIRWISE_MAX_FIELD_SEEDS,
                    replace=False,
                )
            )

        # ------------------------------------------------------
        # Construct curved original -> matched trajectories
        # ------------------------------------------------------
        curves = []

        for idx in valid_indices:

            curve = smooth_pairwise_transition_curve(
                start_xy=original_xy[idx],
                end_xy=warped_xy[idx],
                start_depth=original_positions[sid],
                end_depth=matched_positions[sid],
                n_steps=PAIRWISE_TRANSITION_STEPS,
            )

            curves.append(curve)

        curves_by_sid[sid] = curves

        print(
            f"[PAIRWISE FIELD] sample={sid} | "
            f"original spots={len(original_xy)} | "
            f"valid transformed={len(valid_indices)} | "
            f"matched support={len(matched_records[SAMPLE_IDS.index(sid)].coords)}"
        )

    return curves_by_sid










def interpolate_rgba(
    start_rgba,
    end_rgba,
    t,
):
    start_rgba = np.asarray(
        start_rgba,
        dtype=float,
    )

    end_rgba = np.asarray(
        end_rgba,
        dtype=float,
    )

    t = np.asarray(
        t,
        dtype=float,
    )[:, None]

    return (
        (1.0 - t) * start_rgba[None, :]
        + t * end_rgba[None, :]
    )

def add_pairwise_state_transition_field(
    ax,
    curves_by_sid,
):
    """
    Transition hue starts from the original section's hue and converges
    to the reference Reds hue.
    """

    reference_cmap = plt.get_cmap(
        REFERENCE_CMAP_NAME
    )

    reference_color = reference_cmap(
        PAIRWISE_TRANSITION_COLOR_LEVEL
    )

    for sid in SAMPLE_IDS:

        source_cmap = get_slice_expression_cmap(
            int(sid)
        )

        source_color = source_cmap(
            PAIRWISE_TRANSITION_COLOR_LEVEL
        )

        for curve in curves_by_sid.get(
            int(sid),
            [],
        ):
            if len(curve) < 2:
                continue

            segments = np.stack(
                [
                    curve[:-1],
                    curve[1:],
                ],
                axis=1,
            )

            t_mid = (
                np.arange(
                    len(segments),
                    dtype=float,
                )
                + 0.5
            ) / len(segments)

            colors = interpolate_rgba(
                source_color,
                reference_color,
                t_mid,
            )

            colors[:, 3] = (
                PAIRWISE_TRANSITION_ALPHA
            )

            collection = Line3DCollection(
                segments,
                colors=colors,
                linewidths=(
                    PAIRWISE_TRANSITION_LINEWIDTH
                ),
            )

            ax.add_collection3d(collection)



def style_3d_axis(
    ax,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    depth_positions: Sequence[float],
    xy_pad_ratio: float,
    depth_pad_ratio: float,
    depth_aspect_scale: float,
    single_plane_depth_span: float = 1.0,
    elev: float = ELEV,
    azim: float = AZIM,
    projection_type: str = PROJECTION_TYPE,
) -> None:
    """
    Apply figure-specific 3D geometry.

    depth_positions defines only the planes actually shown in this figure.
    """

    depth_positions = np.asarray(
        depth_positions,
        dtype=float,
    )

    if depth_positions.size == 0:
        raise ValueError(
            "depth_positions cannot be empty."
        )

    # ---------------------------------------------------------
    # Depth range
    # ---------------------------------------------------------
    if (
        depth_positions.size == 1
        or np.isclose(
            np.nanmin(depth_positions),
            np.nanmax(depth_positions),
        )
    ):
        center = float(
            np.nanmean(depth_positions)
        )

        half_span = (
            0.5
            * float(single_plane_depth_span)
        )

        depth_min = center - half_span
        depth_max = center + half_span

    else:
        depth_min = float(
            np.nanmin(depth_positions)
        )

        depth_max = float(
            np.nanmax(depth_positions)
        )

    # ---------------------------------------------------------
    # Spatial padding
    # ---------------------------------------------------------
    x_extent = max(
        x_bounds[1] - x_bounds[0],
        1.0,
    )

    y_extent = max(
        y_bounds[1] - y_bounds[0],
        1.0,
    )

    depth_extent = max(
        depth_max - depth_min,
        1e-6,
    )

    x_pad = (
        float(xy_pad_ratio)
        * x_extent
    )

    y_pad = (
        float(xy_pad_ratio)
        * y_extent
    )

    depth_pad = (
        float(depth_pad_ratio)
        * depth_extent
    )

    # ---------------------------------------------------------
    # Limits
    # ---------------------------------------------------------
    ax.set_xlim(
        depth_min - depth_pad,
        depth_max + depth_pad,
    )

    ax.set_ylim(
        x_bounds[1] + x_pad,
        x_bounds[0] - x_pad,
    )

    ax.set_zlim(
        y_bounds[1] + y_pad,
        y_bounds[0] - y_pad,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.grid(False)

    ax.set_proj_type(
        projection_type
    )

    ax.view_init(
        elev=elev,
        azim=azim,
    )

    # ---------------------------------------------------------
    # Aspect ratio
    # ---------------------------------------------------------
    ax.set_box_aspect(
        (
            depth_extent
            * float(depth_aspect_scale),
            x_extent,
            y_extent,
        )
    )

    for axis in (
        ax.xaxis,
        ax.yaxis,
        ax.zaxis,
    ):
        axis.pane.fill = False
        axis.pane.set_edgecolor(
            (0.82, 0.82, 0.82, 0.25)
        )


def add_value_legend(
    fig,
    ax,
    category_palette,
    continuous_norm,
    expression_cmap,
) -> None:
    if category_palette is not None:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=5.5,
                label=str(category),
            )
            for category, color in category_palette.items()
        ]
        ax.legend(
            handles=handles,
            title=SLICE_COLOR_KEY,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.98),
            frameon=False,
            fontsize=8.5,
            title_fontsize=9.0,
            borderaxespad=0.0,
        )
        return

    if continuous_norm is not None:
        sm = mpl.cm.ScalarMappable(norm=continuous_norm, cmap=expression_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.015, shrink=0.70)
        label = (
            f"log1p({SLICE_COLOR_KEY})"
            if USE_LOG1P_FOR_GENE
            else str(SLICE_COLOR_KEY)
        )
        cbar.set_label(label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)


# =============================================================================
# TRANSITION FIELD PREPARATION
# =============================================================================



# =============================================================================
# FIGURES
# =============================================================================

def figure_width_for_planes(
    n_planes: int,
) -> float:
    """
    Fixed figure height; increase width as more 3D planes are shown.
    """
    n_planes = max(int(n_planes), 1)

    return (
        FIGURE_MIN_WIDTH
        + FIGURE_WIDTH_PER_EXTRA_PLANE
        * (n_planes - 1)
    )


def make_base_figure(
    n_planes: int,
):
    width = figure_width_for_planes(
        n_planes
    )

    fig = plt.figure(
        figsize=(
            width,
            FIGURE_HEIGHT,
        )
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    return fig, ax

def plot_figure1(
    original_records,
    x_bounds,
    y_bounds,
    category_palette,
    continuous_norm,
    expression_cmap,
) -> None:

    fig, ax = make_base_figure(
        len(original_records)
    )

    plot_records(
        ax,
        original_records,
        x_bounds,
        y_bounds,
        category_palette,
        continuous_norm,
        expression_cmap,
        point_size=ORIGINAL_POINT_SIZE*1.5,
        point_alpha=ORIGINAL_POINT_ALPHA,
    )

    style_3d_axis(
        ax,
        x_bounds,
        y_bounds,
        depth_positions=[
            record.display_depth
            for record in original_records
        ],
        xy_pad_ratio=FIG1_XY_PAD_RATIO,
        depth_pad_ratio=FIG1_DEPTH_PAD_RATIO,
        depth_aspect_scale=FIG1_DEPTH_ASPECT_SCALE,
    )

    fig.tight_layout()

    save_or_show(
        fig,
        "figure1_original_four_slices",
    )


def plot_figure2(
    pairwise_records,
    curves_by_sid,
    x_bounds,
    y_bounds,
    category_palette,
    continuous_norm,
    expression_cmap,
) -> None:
    """
    Pairwise transformation from original states into
    a shared reference frame.
    """

    fig, ax = make_base_figure(
        len(pairwise_records)
    )

    record_point_alphas = build_pairwise_record_alphas(
        pairwise_records
    )

    plot_records(
        ax,
        pairwise_records,
        x_bounds,
        y_bounds,
        category_palette,
        continuous_norm,
        expression_cmap,
        point_size=MATCHED_POINT_SIZE,
        point_alpha=FIG2_MATCHED_POINT_ALPHA,
        record_point_alphas=record_point_alphas,
        plane_label_fontsize=(
            PAIRWISE_PLANE_LABEL_FONTSIZE
        ),
        # Figure 2 only: use the original linear gene scale for BOTH color
        # and transparency. This keeps the transition field visible without
        # introducing a nonlinear gamma/PowerNorm transformation.
        linear_gene_alpha=True,
        gene_alpha_floor=FIG2_GENE_ALPHA_FLOOR,
        hide_projection_source_frames=True,
        matched_point_edge_color=FIG2_MATCHED_POINT_EDGE_COLOR,
        matched_point_edge_width=FIG2_MATCHED_POINT_EDGE_WIDTH,
    )

    add_pairwise_state_transition_field(
        ax,
        curves_by_sid,
    )

    style_3d_axis(
        ax,
        x_bounds,
        y_bounds,
        depth_positions=[
            record.display_depth
            for record in pairwise_records
        ],
        xy_pad_ratio=FIG2_XY_PAD_RATIO,
        depth_pad_ratio=FIG2_DEPTH_PAD_RATIO,
        depth_aspect_scale=FIG2_DEPTH_ASPECT_SCALE,
    )

    fig.tight_layout()

    save_or_show(
        fig,
        "figure2_original_to_spatially_matched_states",
    )


def make_figure3_display_records(
    original_records: Sequence[PlaneRecord],
    matched_records: Sequence[PlaneRecord],
    final_record: PlaneRecord,
):
    """
    Figure 3 order:
        reference -> matched slice 2 -> matched slice 3 -> matched slice 4
        -> larger gap -> integrated population-level tissue.
    """
    ref_idx = SAMPLE_IDS.index(REFERENCE_ID)

    single_records: List[PlaneRecord] = [
        replace(
            original_records[ref_idx],
            display_depth=0.0,
            label=None,
        )
    ]

    next_depth = 0.0
    for sid, matched in zip(SAMPLE_IDS, matched_records):
        if int(sid) == int(REFERENCE_ID):
            continue
        next_depth += FIG3_SINGLE_SLICE_GAP
        if sid==1:
            single_records.append(
                replace(
                    matched,
                    display_depth=float(next_depth),
                    label=f"Matched slices",
                )
            )
        else:
            single_records.append(
                replace(
                    matched,
                    display_depth=float(next_depth),
                    label=None,
                )
            )

    integrated_depth = float(next_depth + FIG3_INTEGRATED_GAP)
    integrated_record = replace(
        final_record,
        display_depth=integrated_depth,
        label=FINAL_PLANE_LABEL,
    )

    return single_records, integrated_record


def plot_figure3(
    single_records,
    integrated_record,
    x_bounds,
    y_bounds,
    category_palette,
    single_continuous_norm,
    integrated_continuous_norm,
    expression_cmap,
) -> None:

    all_records = list(single_records) + [integrated_record]

    fig, ax = make_base_figure(
        len(all_records)
    )

    # Reference + matched single slices: higher opacity.
    plot_records(
        ax,
        single_records,
        x_bounds,
        y_bounds,
        category_palette,
        single_continuous_norm,
        expression_cmap,
        point_size=FIG3_SINGLE_POINT_SIZE,
        point_alpha=FIG3_SINGLE_POINT_ALPHA,
        plane_label_fontsize=28,
    )

    # Integrated population-level representation: lower opacity and larger gap.
    plot_records(
        ax,
        [integrated_record],
        x_bounds,
        y_bounds,
        category_palette,
        integrated_continuous_norm,
        expression_cmap,
        point_size=FINAL_POINT_SIZE,
        point_alpha=FINAL_POINT_ALPHA,
        plane_label_fontsize=28,
    )

    style_3d_axis(
        ax,
        x_bounds,
        y_bounds,
        depth_positions=[
            record.display_depth
            for record in all_records
        ],
        xy_pad_ratio=FIG3_XY_PAD_RATIO,
        depth_pad_ratio=FIG3_DEPTH_PAD_RATIO,
        depth_aspect_scale=FIG3_DEPTH_ASPECT_SCALE,
        single_plane_depth_span=FIG3_SINGLE_PLANE_DEPTH_SPAN,
        elev=FIG3_ELEV,
        azim=FIG3_AZIM,
        projection_type=FIG3_PROJECTION_TYPE,
    )

    fig.tight_layout(
        pad=0.05
    )

    save_or_show(
        fig,
        "figure3_reference_matched_and_population_level_tissue",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    set_publication_style()

    original_records = load_original_records()
    matched_records = load_matched_records()
    final_record = load_final_record()

    # Keep categorical colors stable across figures, but DO NOT share the gene
    # normalization. Each figure computes its own continuous gene range.
    category_palette = build_category_style(
        [original_records, matched_records, [final_record]]
    )
    pairwise_records, original_pair_positions, matched_pair_positions = (
        make_pairwise_display_records(
            original_records,
            matched_records,
        )
    )

    figure3_single_records, figure3_integrated_record = (
        make_figure3_display_records(
            original_records,
            matched_records,
            final_record,
        )
    )

    figure1_norm, figure1_cmap = (
        build_figure_value_style(
            original_records
        )
    )

    # Figure 2 follows the dense-reconstruction rule exactly:
    # derive the gene scale from measured slices only, then reuse it for matched states.
    figure2_norm, figure2_cmap = (
        build_figure_value_style(
            original_records
        )
    )

    # Figure 3 uses two independent LINEAR gene-expression ranges while keeping
    # the same Reds colormap for both groups:
    #   1) reference + matched single slices share one range;
    #   2) the integrated population-level slice uses its own range.
    # This preserves relative expression within each representation without
    # forcing the integrated slice onto the single-slice dynamic range.
    figure3_single_norm, figure3_cmap = (
        build_figure_value_style(
            figure3_single_records
        )
    )

    figure3_integrated_norm, _ = (
        build_figure_value_style(
            [figure3_integrated_record]
        )
    )


    if figure1_norm is not None:
        print(f"[GENE SCALE] Figure 1: {figure1_norm.vmin:.4f} .. {figure1_norm.vmax:.4f}")
    if figure2_norm is not None:
        print(f"[GENE SCALE] Figure 2 (measured-derived): {figure2_norm.vmin:.4f} .. {figure2_norm.vmax:.4f}")
    if figure3_single_norm is not None:
        print(
            f"[GENE SCALE] Figure 3 single slices: "
            f"{figure3_single_norm.vmin:.4f} .. {figure3_single_norm.vmax:.4f}"
        )
    if figure3_integrated_norm is not None:
        print(
            f"[GENE SCALE] Figure 3 integrated slice: "
            f"{figure3_integrated_norm.vmin:.4f} .. {figure3_integrated_norm.vmax:.4f}"
        )


    # Figure-specific x/y extents keep each panel tightly framed.
    # Figure 1 bounds: measured sections only.
    fig1_x_bounds, fig1_y_bounds = combined_xy_bounds(
        [original_records]
    )

    # Figure 2 bounds: standalone reference + six measured/matched states.
    fig2_x_bounds, fig2_y_bounds = combined_xy_bounds(
        [pairwise_records]
    )

    # Figure 3 bounds: reference + matched slices + integrated tissue.
    fig3_x_bounds, fig3_y_bounds = combined_xy_bounds(
        [list(figure3_single_records) + [figure3_integrated_record]]
    )


    if requested("figure1"):
        plot_figure1(
            original_records,
            fig1_x_bounds,
            fig1_y_bounds,
            category_palette,
            figure1_norm,
            figure1_cmap,
        )

    curves_by_sid = {}

    if requested("figure2"):
        curves_by_sid = (
            build_pairwise_state_transition_curves(
                original_records=original_records,
                matched_records=matched_records,
                original_positions=original_pair_positions,
                matched_positions=matched_pair_positions,
            )
        )

        plot_figure2(
            pairwise_records,
            curves_by_sid,
            fig2_x_bounds,
            fig2_y_bounds,
            category_palette,
            figure2_norm,
            figure2_cmap,
        )

    if requested("figure3"):
        plot_figure3(
            figure3_single_records,
            figure3_integrated_record,
            fig3_x_bounds,
            fig3_y_bounds,
            category_palette,
            figure3_single_norm,
            figure3_integrated_norm,
            figure3_cmap,
        )
    print(f"[DONE] output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()

