#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Left-to-right 3D pipeline for spatially matched AD progression analysis.

Visual design
-------------
- WORLD X = progression stage / analytical flow (left -> right).
- WORLD Y/Z = spatial coordinates.
- The horizontal progression scaffold is rendered in 2D figure space so it
  remains perfectly horizontal under any 3D map rotation.
- Every spatial map hangs from the scaffold using the retained hanger design.
- The three observed maps are annotated only as Stage 1, Stage 2, and Stage 3
  and are equally spaced. Literal month differences are intentionally weakened.
- Figure 4 places each delta map midway between its bracketing stages. Instead
  of an arrow matrix, two screen-space curved arrows point from the upper-frame
  midpoint of the two bracketing stage maps to the upper-frame midpoint of the
  corresponding delta map.
- Figure 5 is an idea-oriented trend cartoon with smooth curves that share one
  starting point and diverge over progression stages.

Figures
-------
1. Raw observed gene maps: Stage 1 -> Stage 2 -> Stage 3.
2. Projection-to-common-frame transition fields: for each observed stage,
   a repeated Stage-1 shared frame is placed immediately to its left and smooth
   3D curves project observed coordinates onto the nearest pixel locations of
   the full Stage-1 common-frame support.
3. Common-frame observed gene maps: Stage 1 -> Stage 2 -> Stage 3.
4. Stage maps + delta maps with paired curved arrows into each delta.
5. Conceptual gene-variation trends across progression stages.
"""

import argparse
import hashlib
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# ============================================================
# DATA CONFIG
# ============================================================
RAW_ROOT = "/media/huifang/data/registration/mouse_brain"
RAW_NET_IN = os.path.join(RAW_ROOT, "network_input")
RAW_H5AD_DIR = os.path.join(RAW_ROOT, "h5ads", "Tg")
GROUP_ID = 0

RAW_SLICE_SPECS = [
    {"age": 2.5, "label": "2.5m", "sample_id": 0},
    {"age": 5.7, "label": "5.7m", "sample_id": 1},
    {"age": 17.9, "label": "17.9m", "sample_id": 2},
]

COMMON_FRAME_SPECS = [
    {
        "label": "2.5m",
        "time": 2.5,
        "path": (
            "/media/huifang/data/registration/mouse_brain/"
            "virtual_tissue/AD_time_grid256_spatial2.0/analysis/"
            "virtual_AD_time_2.500.h5ad"
        ),
    },
    {
        "label": "5.7m",
        "time": 5.7,
        "path": (
            "/media/huifang/data/registration/mouse_brain/"
            "virtual_tissue/AD_time_grid256_spatial2.0/analysis/"
            "virtual_AD_time_5.700.h5ad"
        ),
    },
    {
        "label": "17.9m",
        "time": 17.9,
        "path": (
            "/media/huifang/data/registration/mouse_brain/"
            "virtual_tissue/AD_time_grid256_spatial2.0/analysis/"
            "virtual_AD_time_17.900.h5ad"
        ),
    },
]

SOURCE_OUTDIR = "/media/huifang/data/registration/paper_analysis/Figure56_mouse/5ghi/"
SPATIAL_MAP_DIR = os.path.join(SOURCE_OUTDIR, "spatial_maps")
SPATIAL_MAP_TEMPLATE = "{gene}_spatial_temporal_map.npz"
OUTDIR = "/media/huifang/data/registration/paper_analysis/Figure56_mouse/5f/"

COORD_KEY = "spatial"
GENE_NAME = "Gfap"

# Common-frame spatial-support plane used in Figure 2.
COMMON_FRAME_DOMAIN_KEY = "axis_joint_domain"
COMMON_FRAME_VALID_KEY = "valid"
COMMON_FRAME_GRID_SIZE = 256
COMMON_FRAME_GRID_COLOR = "#A8A8A8"
COMMON_FRAME_GRID_LW = 0.18
COMMON_FRAME_GRID_ALPHA = 0.55
# For visual clarity in the paper figure, show the pixelized support as
# an overlaid raster grid on top of the domain-colored plane. Drawing all 256
# cell boundaries is usually too dense, so we render readable overlay strides.
COMMON_FRAME_GRID_DISPLAY_STEP = 6
COMMON_FRAME_GRID_MAJOR_STEP = 24
COMMON_FRAME_GRID_OVERLAY_COLOR = "white"
COMMON_FRAME_GRID_OVERLAY_ALPHA = 0.55
COMMON_FRAME_GRID_OVERLAY_LW = 0.35
COMMON_FRAME_GRID_MAJOR_COLOR = "#6E6E6E"
COMMON_FRAME_GRID_MAJOR_ALPHA = 0.75
COMMON_FRAME_GRID_MAJOR_LW = 0.70
COMMON_FRAME_BACKGROUND = "#D0D0D0"

# Figure-2 projection-target layering.
# Figure 2 uses explicit artist z-orders on newer Matplotlib versions. Older
# Matplotlib releases do not support ``computed_zorder``; for those releases a
# small physical WORLD-X separation enforces the same local visual order:
#   common-frame domain raster -> grid/frame -> transition field -> observed state.
COMMON_FRAME_SURFACE_ALPHA = 0.90
COMMON_FRAME_SURFACE_X_OFFSET = 0.0
COMMON_FRAME_GRID_X_OFFSET = 0.0
COMMON_FRAME_FRAME_X_OFFSET = 0.0
TRANSITION_TARGET_X_OFFSET = 0.0

# Compatibility fallback for older Matplotlib releases that do not expose
# Axes3D(computed_zorder=...).  In that case, use a very small physical WORLD-X
# separation inside each [shared frame -> transition -> stage] pair.  The
# nominal scaffold positions are unchanged, but the rendering layers become:
#   domain surface < grid/frame < transition endpoint < observed stage.
FIG2_FALLBACK_SURFACE_BACKSTEP = 7.2
FIG2_FALLBACK_GRID_BACKSTEP = 4.4
FIG2_FALLBACK_FRAME_BACKSTEP = 4.0
FIG2_FALLBACK_TARGET_FORWARDSTEP = -1.8

FIG2_ZORDER_COMMON_SURFACE = 2
FIG2_ZORDER_COMMON_GRID = 5
FIG2_ZORDER_COMMON_FRAME = 6
FIG2_ZORDER_TRANSITION_CURVE = 10
FIG2_ZORDER_TRANSITION_ARROW = 11
FIG2_ZORDER_TRANSITION_ENDPOINT = 12
FIG2_ZORDER_STAGE_POINTS = 20
FIG2_ZORDER_STAGE_FRAME = 21

# Projection/transition caches are stored beside the running script so repeated
# paper-figure rendering can reuse BOTH the coordinate projection and the fully
# built 3D transition curves.
TEMP_RESULT_DIR = Path(__file__).resolve().parent / "temp_result"
PROJECTION_CACHE_VERSION = 2
TRANSITION_FIELD_CACHE_VERSION = 2
PROJECTION_CACHE_COORD_DECIMALS = 4
TRANSITION_CURVE_N_STEPS = 44

# Registration-transition visualization used only in Figure 2.
# Each stage is shown as a pair: [shared Stage-1 frame] -> [observed state].
# The shared-frame copy is placed immediately to the LEFT of its source state.
REGISTRATION_PAIR_GAP = 135.0
REGISTRATION_GROUP_GAP = 360.0
MAX_TRANSITION_CURVES_PER_STAGE = 420
TRANSITION_RANDOM_SEED = 37
TRANSITION_CONTROL_SCALE = 0.34
TRANSITION_CURVE_LW = 1
TRANSITION_CURVE_ALPHA = 0.12
TRANSITION_TARGET_TAIL_FRACTION = 0.38
TRANSITION_TARGET_TAIL_ZORDER = 1000
TRANSITION_TARGET_ARROW_ZORDER = 1001
TRANSITION_TARGET_ENDPOINT_ZORDER = 1002
TRANSITION_TARGET_TAIL_ALPHA = 0.22
TRANSITION_ARROW_HEAD_LENGTH_SCALE = 0.55
TRANSITION_ARROW_HEAD_WIDTH_SCALE = 0.50
TRANSITION_ENDPOINT_SIZE = 2.0
TRANSITION_ENDPOINT_ALPHA = 0.40
TRANSITION_CURVE_COLORS = ["#4E79A7", "#59A14F", "#D3725B"]

# Stable IDs are retained in the data containers for provenance, but Figure 2
# no longer requires one-to-one correspondence. Observations are projected
# directly onto the regular common-frame pixel lattice.
TRANSITION_MATCH_ID_KEYS = (
    "source_obs_name",
    "source_obs_id",
    "source_spot_id",
    "source_barcode",
    "original_obs_name",
    "original_barcode",
    "barcode",
    "spot_id",
)


# ============================================================
# PLOT CONFIG
# ============================================================
FIG_DPI = 400
FIGURE_MODE = "save"  # save/show/both
SHOW_BLOCK = True

FIXED_BOX = (0.0, 1024.0, 0.0, 1024.0)
RAW_SWAP_XY = True
COMMON_FRAME_SWAP_XY = True
SAVED_SPATIAL_SWAP_XY = False
INVERT_DISPLAY_X = True
INVERT_DISPLAY_Y = False

# WORLD-X progression.
# Main 3D panels use equal stage spacing rather than month-proportional spacing.
# This intentionally weakens the literal time-distance encoding and highlights
# stage-wise temporal progression. 2.5m, 5.7m, 17.9m are placed at 0, 1, 2 gaps.
STAGE_GAP = 360.0

STAGE_LABELS = ["Stage 1", "Stage 2", "Stage 3"]
DELTA_LABELS = ["Δ Stage 1→2", "Δ Stage 2→3"]

# Screen-space curved comparison arrows used in Figure 3.
CURVE_ARROW_COLOR = "#707070"
CURVE_ARROW_ALPHA = 0.72
CURVE_ARROW_LW = 1.7
CURVE_ARROW_SCALE = 12.0
CURVE_ARROW_RAD = 1.
CURVE_ENDPOINT_SHRINK = 4.0

# Retained only for backward CLI compatibility; no longer used to position the
# three observed stages in Figures 1-3.
TIME_SCALE_PER_MONTH = 60.0


# Uniform visual scale of spatial Y/Z planes relative to WORLD-X.
SPATIAL_DISPLAY_SCALE = 0.3

POINT_SIZE_GENE = 1.2
POINT_SIZE_DELTA = 1.7
POINT_SIZE_TREND = 1.8
MAX_POINTS_PER_PLANE = 30000

GENE_ALPHA_FIG1 = 0.74
GENE_ALPHA_FIG2 = 0.74
GENE_ALPHA_WITH_DELTA = 0.40
DELTA_ALPHA = 0.56
TREND_ALPHA = 0.95

OBSERVED_FRAME_COLOR = "#303030"
DERIVED_FRAME_COLOR = "#8A8A8A"
TREND_FRAME_COLOR = "#555555"
OBSERVED_LABEL_COLOR = "#1F1F1F"
DERIVED_LABEL_COLOR = "#707070"
TREND_LABEL_COLOR = "#555555"
BACKGROUND_GRAY = "#D0D0D0"
FLOW_TEXT_COLOR = "#6B6B6B"

SOLID_LW = 1.4
DERIVED_LW = 1.3
DERIVED_STYLE = (0, (4.0, 2.4))

# Screen-space horizontal scaffold.
TIMELINE_COLOR = "#2F2F2F"
TIMELINE_LW = 1.9
TIMELINE_ARROW_SCALE = 13.0
TIMELINE_MARKER_SIZE = 7.8
TIMELINE_MARKER_EDGE_COLOR = "white"
TIMELINE_MARKER_EDGE_WIDTH = 0.9
TIMELINE_CLEARANCE_FIG = 0.035
TIMELINE_MIN_Y_FIG = 0.72
TIMELINE_MAX_Y_FIG = 0.68
TIMELINE_X_PAD_FIG = 0.025
TIMELINE_LABEL_OFFSET_FIG = 0.005
TIMELINE_LABEL_SECOND_TIER_FIG = 0.034

# Hangers: intentionally retained from the previous design.
HANGER_COLOR = "#7A7A7A"
HANGER_LW = 0.85
HANGER_ALPHA = 0.55
HANGER_TOP_GAP_FIG = 0.003
HANGER_BOTTOM_GAP_FIG = 0.003
HANGER_ANCHOR_SIZE = 3.8


# Camera affects ONLY 3D objects.
VIEW_ELEV = 10.0
VIEW_AZIM = -50.0
VIEW_ROLL = 0.0

PATTERN_COLORS = {
    "Increasing": "#D55E00",
    "Decreasing": "#0072B2",
    "Peak": "#009E73",
    "Trough": "#CC79A7",
    "Stable": "#B9B9B9",
}
PATTERN_ORDER = ["Increasing", "Decreasing", "Peak", "Trough", "Stable"]



# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class GeneMap:
    label: str
    time: float
    path: str
    coords: np.ndarray
    values: np.ndarray
    obs_names: np.ndarray
    match_ids: np.ndarray
    valid_mask: np.ndarray


@dataclass
class CommonFrameSupport:
    path: str
    coords: np.ndarray
    domains: np.ndarray
    valid_mask: np.ndarray


@dataclass
class SpatialTemporalMap:
    gene: str
    path: str
    coords: np.ndarray
    domains: np.ndarray
    obs_names: np.ndarray
    valid_domain_mask: np.ndarray
    times: np.ndarray
    time_labels: np.ndarray
    value_2p5: np.ndarray
    value_5p7: np.ndarray
    value_17p9: np.ndarray
    delta_12: np.ndarray
    delta_23: np.ndarray
    delta_13: np.ndarray
    pattern: np.ndarray
    amplitude: np.ndarray
    pattern_threshold: float


@dataclass
class LayerMeta:
    label: str
    kind: str  # observed / delta / trend


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
        raise ValueError(f"Invalid figure mode: {mode!r}")
    if mode in {"save", "both"}:
        ensure_dir(os.path.dirname(save_path) or ".")
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        print(f"[SAVE] {save_path}")
    if mode in {"show", "both"}:
        plt.show(block=SHOW_BLOCK)
    plt.close(fig)


def to_dense_1d(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=float).ravel()


def validate_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected coordinates with shape (N, 2+), got {coords.shape}")
    return coords[:, :2].copy()


def sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    idx = np.arange(n, dtype=int)
    if n <= max_points:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_points, replace=False))


def robust_positive_vmax(arrays: Sequence[np.ndarray], q: float = 99.0) -> float:
    x = np.concatenate([np.asarray(v, dtype=float).ravel() for v in arrays])
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    return max(float(np.percentile(x, q)), 1e-8)


def robust_symmetric_vmax(arrays: Sequence[np.ndarray], q: float = 99.0) -> float:
    x = np.concatenate([np.asarray(v, dtype=float).ravel() for v in arrays])
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    return max(float(np.percentile(np.abs(x), q)), 1e-8)


def age_to_world_x(age: float, youngest_age: float) -> float:
    """Legacy month-proportional mapping; retained for compatibility only."""
    return (float(age) - float(youngest_age)) * TIME_SCALE_PER_MONTH


def equal_stage_positions(n: int, gap: float = STAGE_GAP) -> np.ndarray:
    """Return equally spaced WORLD-X positions for ordered developmental stages."""
    if n < 1:
        return np.asarray([], dtype=float)
    return np.arange(n, dtype=float) * float(gap)


# ============================================================
# DISPLAY TRANSFORMS / LOAD DATA
# ============================================================
def transform_coords(coords: np.ndarray, *, swap_xy: bool) -> np.ndarray:
    coords = validate_coords(coords)
    if swap_xy:
        coords = coords[:, [1, 0]]
    xmin, xmax, ymin, ymax = FIXED_BOX
    if INVERT_DISPLAY_X:
        coords[:, 0] = xmin + xmax - coords[:, 0]
    if INVERT_DISPLAY_Y:
        coords[:, 1] = ymin + ymax - coords[:, 1]
    return coords


def read_gene_for_display(ad: sc.AnnData, gene: str) -> np.ndarray:
    if gene not in ad.var_names:
        raise KeyError(f"Gene {gene!r} not found.")
    values = to_dense_1d(ad[:, [gene]].X)
    return np.log1p(np.clip(values, a_min=0.0, a_max=None))


def _extract_transition_match_ids(ad: sc.AnnData) -> np.ndarray:
    """Return the best available stable source/spot identifiers for matching."""
    for key in TRANSITION_MATCH_ID_KEYS:
        if key in ad.obs.columns:
            values = ad.obs[key].astype(str).to_numpy()
            if len(values) == ad.n_obs:
                return values
    return ad.obs_names.astype(str).to_numpy()


def load_original_observed_maps(gene: str) -> List[GeneMap]:
    records = []
    for spec in RAW_SLICE_SPECS:
        sample_id = int(spec["sample_id"])
        h5ad_path = os.path.join(RAW_H5AD_DIR, f"{GROUP_ID}_{sample_id}.h5ad")
        coord_path = os.path.join(RAW_NET_IN, f"{GROUP_ID}_{sample_id}_validation.npz")
        if not os.path.isfile(h5ad_path):
            raise FileNotFoundError(h5ad_path)
        if not os.path.isfile(coord_path):
            raise FileNotFoundError(coord_path)
        ad = sc.read_h5ad(h5ad_path)
        values = read_gene_for_display(ad, gene)
        coords = transform_coords(np.load(coord_path)["coord"], swap_xy=RAW_SWAP_XY)
        if len(coords) != len(values):
            raise ValueError(f"Row mismatch at {spec['label']}: coords={len(coords)}, values={len(values)}")
        obs_names = ad.obs_names.astype(str).to_numpy()
        match_ids = _extract_transition_match_ids(ad)
        valid_mask = np.ones(ad.n_obs, dtype=bool)
        records.append(
            GeneMap(
                str(spec["label"]), float(spec["age"]), h5ad_path, coords, values,
                obs_names, match_ids, valid_mask,
            )
        )
    return sorted(records, key=lambda r: r.time)


def load_common_frame_gene_maps(gene: str) -> List[GeneMap]:
    records = []
    for spec in COMMON_FRAME_SPECS:
        path = spec["path"]
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        ad = sc.read_h5ad(path)
        if COORD_KEY not in ad.obsm:
            raise KeyError(f"{COORD_KEY!r} not found in ad.obsm for {path}")
        coords = transform_coords(np.asarray(ad.obsm[COORD_KEY], dtype=float), swap_xy=COMMON_FRAME_SWAP_XY)
        values = read_gene_for_display(ad, gene)
        if len(coords) != len(values):
            raise ValueError(f"Row mismatch at {spec['label']}: coords={len(coords)}, values={len(values)}")
        obs_names = ad.obs_names.astype(str).to_numpy()
        match_ids = _extract_transition_match_ids(ad)
        valid_mask = (
            ad.obs[COMMON_FRAME_VALID_KEY].to_numpy().astype(bool)
            if COMMON_FRAME_VALID_KEY in ad.obs.columns
            else np.ones(ad.n_obs, dtype=bool)
        )
        records.append(
            GeneMap(
                str(spec["label"]), float(spec["time"]), path, coords, values,
                obs_names, match_ids, valid_mask,
            )
        )
    return sorted(records, key=lambda r: r.time)


def load_stage1_common_frame_support() -> CommonFrameSupport:
    """
    Load the Stage-1 common-frame domain layout used to visualize the shared
    spatial support plane in Figure 2.

    The full 256 x 256 raster is drawn later. Coordinates carrying a valid
    Stage-1 spatial-domain label are colored by domain; all other pixels remain
    gray background.
    """
    path = COMMON_FRAME_SPECS[0]["path"]
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ad = sc.read_h5ad(path)
    if COORD_KEY not in ad.obsm:
        raise KeyError(f"{COORD_KEY!r} not found in ad.obsm for {path}")
    if COMMON_FRAME_DOMAIN_KEY not in ad.obs.columns:
        raise KeyError(
            f"{COMMON_FRAME_DOMAIN_KEY!r} not found in ad.obs for {path}"
        )

    coords = transform_coords(
        np.asarray(ad.obsm[COORD_KEY], dtype=float),
        swap_xy=COMMON_FRAME_SWAP_XY,
    )
    domains = ad.obs[COMMON_FRAME_DOMAIN_KEY].astype(str).to_numpy()

    domain_series = pd.Series(domains, dtype="object")
    domain_valid = ~domain_series.str.lower().isin({"nan", "none", "na", ""})
    valid_mask = domain_valid.to_numpy(dtype=bool)

    if COMMON_FRAME_VALID_KEY in ad.obs.columns:
        valid_mask &= ad.obs[COMMON_FRAME_VALID_KEY].to_numpy().astype(bool)

    if len(coords) != len(domains):
        raise ValueError(
            f"Shared-frame row mismatch: coords={len(coords)}, domains={len(domains)}"
        )

    return CommonFrameSupport(
        path=path,
        coords=coords,
        domains=domains,
        valid_mask=valid_mask,
    )


def normalize_pattern_strings(values: Sequence) -> np.ndarray:
    lookup = {
        "increasing": "Increasing",
        "decreasing": "Decreasing",
        "peak": "Peak",
        "trough": "Trough",
        "stable": "Stable",
    }
    return np.asarray(
        [lookup.get(str(v).strip().lower(), str(v).strip()) for v in np.asarray(values, dtype=object).ravel()],
        dtype=object,
    )


def saved_spatial_map_path(gene: str, explicit_path: Optional[str] = None) -> str:
    if explicit_path is not None:
        return explicit_path
    return os.path.join(SPATIAL_MAP_DIR, SPATIAL_MAP_TEMPLATE.format(gene=gene))


def load_saved_spatial_temporal_map(gene: str, explicit_path: Optional[str] = None) -> SpatialTemporalMap:
    path = saved_spatial_map_path(gene, explicit_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing saved spatial map: {path}. Run the updated analysis script first.")
    payload = np.load(path, allow_pickle=True)
    stm = SpatialTemporalMap(
        gene=gene,
        path=path,
        coords=transform_coords(payload["coords"], swap_xy=SAVED_SPATIAL_SWAP_XY),
        domains=np.asarray(payload["domains"], dtype=object),
        obs_names=np.asarray(payload["obs_names"], dtype=object),
        valid_domain_mask=np.asarray(payload["valid_domain_mask"], dtype=bool),
        times=np.asarray(payload["times"], dtype=float),
        time_labels=np.asarray(payload["time_labels"], dtype=object),
        value_2p5=np.asarray(payload["value_2p5"], dtype=float).ravel(),
        value_5p7=np.asarray(payload["value_5p7"], dtype=float).ravel(),
        value_17p9=np.asarray(payload["value_17p9"], dtype=float).ravel(),
        delta_12=np.asarray(payload["delta_5p7_minus_2p5"], dtype=float).ravel(),
        delta_23=np.asarray(payload["delta_17p9_minus_5p7"], dtype=float).ravel(),
        delta_13=np.asarray(payload["delta_17p9_minus_2p5"], dtype=float).ravel(),
        pattern=normalize_pattern_strings(payload["pattern"]),
        amplitude=np.asarray(payload["amplitude"], dtype=float).ravel(),
        pattern_threshold=float(np.asarray(payload["pattern_threshold"]).ravel()[0]),
    )
    print(f"[LOAD] Saved spatial temporal map: {path}")
    return stm


# ============================================================
# COLORS
# ============================================================
def gene_facecolors(values: np.ndarray, vmax: float, alpha_factor: float) -> np.ndarray:
    normalized = np.clip(np.asarray(values, dtype=float) / max(vmax, 1e-8), 0.0, 1.0)
    colors = plt.cm.Reds(normalized)
    colors[:, 3] = np.clip((0.08 + 0.92 * normalized) * alpha_factor, 0.02, 1.0)
    return colors


def delta_facecolors(values: np.ndarray, vmax: float, alpha: float = DELTA_ALPHA) -> np.ndarray:
    norm = TwoSlopeNorm(vmin=-max(vmax, 1e-8), vcenter=0.0, vmax=max(vmax, 1e-8))
    colors = plt.cm.coolwarm(norm(np.asarray(values, dtype=float)))
    colors[:, 3] = alpha
    return colors


# ============================================================
# 3D WORLD GEOMETRY: X=FLOW, Y/Z=SPATIAL
# ============================================================
def draw_plane_frame_3d(
    ax,
    flow_x: float,
    extent: Tuple[float, float, float, float],
    kind: str,
    *,
    zorder: float = 20,
) -> None:
    ymin, ymax, zmin, zmax = extent
    x = np.full(5, float(flow_x), dtype=float)
    y = np.asarray([ymin, ymax, ymax, ymin, ymin], dtype=float)
    z = np.asarray([zmin, zmin, zmax, zmax, zmin], dtype=float)

    if kind == "observed":
        color, linestyle, linewidth = OBSERVED_FRAME_COLOR, "-", SOLID_LW
    elif kind == "trend":
        color, linestyle, linewidth = TREND_FRAME_COLOR, DERIVED_STYLE, DERIVED_LW
    else:
        color, linestyle, linewidth = DERIVED_FRAME_COLOR, DERIVED_STYLE, DERIVED_LW

    ax.plot(
        x, y, z,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=0.96,
        zorder=zorder,
    )


def style_3d_axis(
    ax,
    extent: Tuple[float, float, float, float],
    flow_positions: Sequence[float],
    spatial_scale=SPATIAL_DISPLAY_SCALE,
) -> None:
    ymin, ymax, zmin, zmax = extent
    flow_positions = np.asarray(flow_positions, dtype=float)

    x_span = max(float(np.ptp(flow_positions)), 1.0)
    x_pad = max(0.06 * x_span, 45.0)
    y_pad = 70.0
    z_pad = 70.0

    x0 = float(np.min(flow_positions) - x_pad)
    x1 = float(np.max(flow_positions) + x_pad)
    y0, y1 = ymin - y_pad, ymax + y_pad
    z0, z1 = zmin - z_pad, zmax + z_pad

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)

    display_span = np.asarray(
        [
            (x1 - x0),
            (y1 - y0) * spatial_scale,
            (z1 - z0) * spatial_scale,
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
# SCREEN-SPACE HORIZONTAL SCAFFOLD + HANGERS
# ============================================================
def _project_world_to_figure(fig, ax, xyz: Sequence[float]) -> Tuple[float, float]:
    x3, y3, z3 = [float(v) for v in xyz]
    x2, y2, _ = proj3d.proj_transform(x3, y3, z3, ax.get_proj())
    x_disp, y_disp = ax.transData.transform((x2, y2))
    x_fig, y_fig = fig.transFigure.inverted().transform((x_disp, y_disp))
    return float(x_fig), float(y_fig)


def _top_center_anchor(
    flow_x: float,
    extent: Tuple[float, float, float, float],
) -> Tuple[float, float, float]:
    ymin, ymax, _, zmax = extent
    return float(flow_x), 0.5 * (ymin + ymax), float(zmax)


def _kind_color(kind: str) -> str:
    if kind == "observed":
        return OBSERVED_LABEL_COLOR
    if kind == "trend":
        return TREND_LABEL_COLOR
    return DERIVED_LABEL_COLOR


def _kind_marker(kind: str) -> str:
    if kind == "observed":
        return "o"
    if kind == "trend":
        return "s"
    return "D"


def draw_horizontal_hanging_scaffold(
    fig,
    ax,
    layers: Sequence[LayerMeta],
    flow_positions: Sequence[float],
    extent: Tuple[float, float, float, float],
    title_label: str,
    line_mask: Optional[Sequence[bool]] = None,
) -> Tuple[float, np.ndarray]:
    """
    Draw a perfectly horizontal screen-space arrow plus vertical hangers.

    `line_mask` controls which layer markers belong to the main scaffold line.
    This is useful for Figure 4 where the trend map must lie beyond the time
    axis and should not be interpreted as a timepoint.
    """
    if len(layers) != len(flow_positions):
        raise ValueError("layers and flow_positions must have the same length")

    fig.canvas.draw()
    anchors = np.asarray(
        [
            _project_world_to_figure(fig, ax, _top_center_anchor(x, extent))
            for x in flow_positions
        ],
        dtype=float,
    )
    marker_x = anchors[:, 0]
    anchor_y = anchors[:, 1]

    if line_mask is None:
        line_mask = np.ones(len(layers), dtype=bool)
    line_mask = np.asarray(line_mask, dtype=bool)
    if line_mask.shape[0] != len(layers):
        raise ValueError("line_mask must match layers length")

    scaffold_y = max(TIMELINE_MIN_Y_FIG, float(np.max(anchor_y[line_mask]) + TIMELINE_CLEARANCE_FIG))
    scaffold_y = min(scaffold_y, TIMELINE_MAX_Y_FIG)

    main_x = marker_x[line_mask]
    line_start = max(0.02, float(np.min(main_x) - TIMELINE_X_PAD_FIG))
    line_end = min(0.94, float(np.max(main_x) + TIMELINE_X_PAD_FIG + 0.010))

    arrow = FancyArrowPatch(
        (line_start, scaffold_y),
        (line_end, scaffold_y),
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

    # fig.text(
    #     min(line_end + 0.008, 0.965),
    #     scaffold_y + 0.001,
    #     title_label,
    #     fontsize=14.5,
    #     fontweight="bold",
    #     color=TIMELINE_COLOR,
    #     ha="left",
    #     va="center",
    #     zorder=205,
    # )

    # Main scaffold markers + hangers.
    main_indices = np.where(line_mask)[0]
    use_two_tiers = len(main_indices) >= 6
    main_counter = 0

    for i, (layer, xmark, y_anchor) in enumerate(zip(layers, marker_x, anchor_y)):
        if not line_mask[i]:
            continue

        color = _kind_color(layer.kind)
        marker_shape = _kind_marker(layer.kind)

        y_top = scaffold_y - HANGER_TOP_GAP_FIG
        y_bottom = y_anchor + HANGER_BOTTOM_GAP_FIG
        if y_bottom < y_top:
            fig.add_artist(
                Line2D(
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
            )

        fig.add_artist(
            Line2D(
                [xmark],
                [scaffold_y],
                transform=fig.transFigure,
                marker=marker_shape,
                linestyle="",
                markersize=TIMELINE_MARKER_SIZE,
                markerfacecolor=color,
                markeredgecolor=TIMELINE_MARKER_EDGE_COLOR,
                markeredgewidth=TIMELINE_MARKER_EDGE_WIDTH,
                clip_on=False,
                zorder=210,
            )
        )

        fig.add_artist(
            Line2D(
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
        )

        if use_two_tiers:
            label_dy = TIMELINE_LABEL_OFFSET_FIG if main_counter % 2 == 0 else TIMELINE_LABEL_SECOND_TIER_FIG
        else:
            label_dy = TIMELINE_LABEL_OFFSET_FIG
        main_counter += 1

        fig.text(
            xmark,
            scaffold_y + label_dy,
            layer.label,
            fontsize=13.0 if use_two_tiers else 12.2,
            fontweight="bold",
            color=color,
            ha="center",
            va="bottom",
            zorder=215,
        )

    return scaffold_y, anchors


# ============================================================
# CURVED SCREEN-SPACE COMPARISON ARROWS
# ============================================================

def draw_curved_map_arrow(fig, ax, source_x, target_x, extent, rad):
    """Draw a curved arrow between projected upper-frame edge midpoints."""
    fig.canvas.draw()
    src = _project_world_to_figure(fig, ax, _top_center_anchor(source_x, extent))
    dst = _project_world_to_figure(fig, ax, _top_center_anchor(target_x, extent))
    patch = FancyArrowPatch(
        src, dst,
        transform=fig.transFigure,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=CURVE_ARROW_SCALE,
        linewidth=CURVE_ARROW_LW,
        color=CURVE_ARROW_COLOR,
        alpha=CURVE_ARROW_ALPHA,
        shrinkA=CURVE_ENDPOINT_SHRINK,
        shrinkB=CURVE_ENDPOINT_SHRINK,
        clip_on=False,
        zorder=175,
    )
    fig.add_artist(patch)


def draw_bracket_to_delta_curves(fig, ax, left_stage_x, delta_x, right_stage_x, extent):
    """Two curved arrows from bracketing stage maps into one delta map."""
    draw_curved_map_arrow(fig, ax, left_stage_x, delta_x, extent, rad=-abs(CURVE_ARROW_RAD))
    draw_curved_map_arrow(fig, ax, right_stage_x, delta_x, extent, rad=abs(CURVE_ARROW_RAD))


# ============================================================
# COMMON-FRAME SUPPORT PLANE
# ============================================================
def _domain_sort_key(value: str):
    try:
        return (0, float(value))
    except Exception:
        return (1, str(value))


def _domain_color(value: str):
    """Stable tab20 color assignment for numeric/string domain labels."""
    try:
        idx = int(float(value))
    except Exception:
        idx = abs(hash(str(value))) % 20
    return plt.get_cmap("tab20")(idx % 20)


def rasterize_common_frame_domains(
    support: CommonFrameSupport,
    grid_size: int = COMMON_FRAME_GRID_SIZE,
) -> np.ndarray:
    """
    Rasterize Stage-1 domain labels onto the complete common-frame lattice.

    Returns an RGBA array with shape (grid_size, grid_size, 4). Every pixel is
    gray by default. Pixels represented by valid Stage-1 domain locations are
    replaced by that domain's categorical color.
    """
    xmin, xmax, ymin, ymax = FIXED_BOX
    background = np.asarray(plt.matplotlib.colors.to_rgba(COMMON_FRAME_BACKGROUND))
    colors = np.empty((grid_size, grid_size, 4), dtype=float)
    colors[:] = background

    coords = np.asarray(support.coords, dtype=float)
    domains = np.asarray(support.domains, dtype=object)
    valid = np.asarray(support.valid_mask, dtype=bool)

    finite = (
        np.isfinite(coords[:, 0])
        & np.isfinite(coords[:, 1])
        & (coords[:, 0] >= xmin)
        & (coords[:, 0] <= xmax)
        & (coords[:, 1] >= ymin)
        & (coords[:, 1] <= ymax)
    )
    use = valid & finite

    if not np.any(use):
        return colors

    xy = coords[use]
    dom = domains[use]

    gx = np.floor(
        (xy[:, 0] - xmin) / max(xmax - xmin, 1e-8) * grid_size
    ).astype(int)
    gy = np.floor(
        (xy[:, 1] - ymin) / max(ymax - ymin, 1e-8) * grid_size
    ).astype(int)
    gx = np.clip(gx, 0, grid_size - 1)
    gy = np.clip(gy, 0, grid_size - 1)

    # One common-frame location is normally associated with one raster pixel.
    # If multiple rows map to one pixel, the final valid domain assignment is
    # used; this is deterministic for the already aligned common frame.
    for ix, iy, domain in zip(gx, gy, dom):
        colors[ix, iy] = _domain_color(str(domain))

    return colors


def draw_common_frame_grid_overlay_3d(
    ax,
    flow_x: float,
    frame_extent: Tuple[float, float, float, float] = FIXED_BOX,
    grid_size: int = COMMON_FRAME_GRID_SIZE,
    display_step: int = COMMON_FRAME_GRID_DISPLAY_STEP,
    major_step: int = COMMON_FRAME_GRID_MAJOR_STEP,
    *,
    zorder_fine: float = 12,
    zorder_major: float = 13,
) -> None:
    """Overlay a readable raster grid on top of the domain-colored plane.

    A wireframe-based overlay is more visually stable across repeated common
    frames than individually plotting hundreds of line segments.
    """
    ymin, ymax, zmin, zmax = frame_extent

    def _wire(step, color, lw, alpha, zorder):
        idx = np.arange(0, grid_size + 1, max(1, int(step)), dtype=int)
        if idx[-1] != grid_size:
            idx = np.r_[idx, grid_size]
        y_edges = np.linspace(ymin, ymax, grid_size + 1)[idx]
        z_edges = np.linspace(zmin, zmax, grid_size + 1)[idx]
        Y, Z = np.meshgrid(y_edges, z_edges, indexing="ij")
        X = np.full_like(Y, float(flow_x), dtype=float)
        ax.plot_wireframe(
            X, Y, Z,
            rstride=1,
            cstride=1,
            color=color,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
        )

    _wire(
        display_step,
        COMMON_FRAME_GRID_OVERLAY_COLOR,
        COMMON_FRAME_GRID_OVERLAY_LW,
        COMMON_FRAME_GRID_OVERLAY_ALPHA,
        zorder_fine,
    )
    _wire(
        major_step,
        COMMON_FRAME_GRID_MAJOR_COLOR,
        COMMON_FRAME_GRID_MAJOR_LW,
        COMMON_FRAME_GRID_MAJOR_ALPHA,
        zorder_major,
    )


def plot_common_frame_support_plane_3d(

    ax,
    support: CommonFrameSupport,
    flow_x: float,
    frame_extent: Tuple[float, float, float, float] = FIXED_BOX,
    *,
    toward_stage_sign: float = 1.0,
    use_physical_depth_fallback: bool = False,
    surface_zorder: float = FIG2_ZORDER_COMMON_SURFACE,
    grid_zorder: float = FIG2_ZORDER_COMMON_GRID,
    frame_zorder: float = FIG2_ZORDER_COMMON_FRAME,
) -> None:
    """
    Draw the repeated Stage-1 shared frame as an explicit projection target.

    Figure 2 uses manual z-ordering when supported. On older Matplotlib, a
    small WORLD-X separation is used so the common-frame domain map remains
    physically behind the transition endpoint and observed state.
    """
    ymin, ymax, zmin, zmax = frame_extent
    grid_size = int(COMMON_FRAME_GRID_SIZE)
    facecolors = rasterize_common_frame_domains(support, grid_size=grid_size)

    sign = 1.0 if float(toward_stage_sign) >= 0 else -1.0
    if use_physical_depth_fallback:
        # Stable physical layering used in the project environment:
        # common-frame domain surface -> common-frame grid/frame -> transition
        # field -> observed stage.
        surface_x = float(flow_x) - sign * float(FIG2_FALLBACK_SURFACE_BACKSTEP)
        grid_x = float(flow_x) - sign * float(FIG2_FALLBACK_GRID_BACKSTEP)
        frame_x = float(flow_x) - sign * float(FIG2_FALLBACK_FRAME_BACKSTEP)
    else:
        surface_x = float(flow_x) + float(COMMON_FRAME_SURFACE_X_OFFSET)
        grid_x = float(flow_x) + float(COMMON_FRAME_GRID_X_OFFSET)
        frame_x = float(flow_x) + float(COMMON_FRAME_FRAME_X_OFFSET)

    y_edges = np.linspace(ymin, ymax, grid_size + 1)
    z_edges = np.linspace(zmin, zmax, grid_size + 1)
    Y, Z = np.meshgrid(y_edges, z_edges, indexing="ij")
    X = np.full_like(Y, surface_x, dtype=float)

    surface = ax.plot_surface(
        X, Y, Z,
        rstride=1,
        cstride=1,
        facecolors=facecolors,
        edgecolor="none",
        linewidth=0.0,
        alpha=COMMON_FRAME_SURFACE_ALPHA,
        shade=False,
        antialiased=False,
        zorder=surface_zorder,
    )
    surface.set_rasterized(True)

    draw_common_frame_grid_overlay_3d(
        ax,
        grid_x,
        frame_extent=frame_extent,
        grid_size=grid_size,
        zorder_fine=grid_zorder,
        zorder_major=grid_zorder + 0.2,
    )

    draw_plane_frame_3d(
        ax,
        frame_x,
        frame_extent,
        kind="delta",
        zorder=frame_zorder,
    )


def _figure2_create_3d_axis(fig):
    """Create a Figure-2 3D axis using the backward-compatible fallback path.

    The project environment uses an older Matplotlib, so Figure 2 always uses
    explicit physical depth separation rather than version-dependent
    ``computed_zorder`` behavior.
    """
    ax = fig.add_subplot(111, projection="3d")
    return ax, False


# ============================================================
# 3D REGISTRATION-TRANSITION FIELD HELPERS (FIGURE 2)
# ============================================================
def _projection_coord_fingerprint(coords: np.ndarray) -> str:
    """Compact fingerprint used to invalidate stale cached projections."""
    arr = np.asarray(coords, dtype=float)[:, :2]
    arr = np.round(arr, decimals=PROJECTION_CACHE_COORD_DECIMALS)
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _projection_cache_path(source: GeneMap) -> Path:
    TEMP_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    label = "".join(ch if ch.isalnum() else "_" for ch in str(source.label)).strip("_")
    return TEMP_RESULT_DIR / f"figure2_projection_{label}.npz"


def _project_to_regular_common_frame_grid(source_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project finite observations analytically onto the nearest 256x256 pixel
    centers of the full 1024x1024 common frame.

    Because the target support is a regular lattice, this is faster and more
    faithful to the visual common-frame definition than building a KD-tree.
    """
    src_all = np.asarray(source_coords, dtype=float)[:, :2]
    finite = np.all(np.isfinite(src_all), axis=1)
    src = src_all[finite]
    if len(src) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    xmin, xmax, ymin, ymax = FIXED_BOX
    n = float(COMMON_FRAME_GRID_SIZE)
    pixel_w = (xmax - xmin) / n
    pixel_h = (ymax - ymin) / n

    # Nearest pixel-center index on the complete support. Values outside the
    # nominal frame are clipped to the closest boundary pixel.
    gx = np.rint((src[:, 0] - xmin) / pixel_w - 0.5).astype(int)
    gy = np.rint((src[:, 1] - ymin) / pixel_h - 0.5).astype(int)
    gx = np.clip(gx, 0, COMMON_FRAME_GRID_SIZE - 1)
    gy = np.clip(gy, 0, COMMON_FRAME_GRID_SIZE - 1)

    tgt = np.column_stack(
        [
            xmin + (gx + 0.5) * pixel_w,
            ymin + (gy + 0.5) * pixel_h,
        ]
    )
    return src.astype(np.float32), tgt.astype(np.float32)


def project_observations_to_common_frame(
    source: GeneMap,
    support: CommonFrameSupport,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project observations onto the full Stage-1 common-frame pixel lattice.

    Results are cached under ``<script_dir>/temp_result`` and automatically
    reused when the source coordinates and common-frame definition have not
    changed. No IDs or one-to-one correspondences are required.
    """
    cache_path = _projection_cache_path(source)
    source_fp = _projection_coord_fingerprint(source.coords)
    support_fp = _projection_coord_fingerprint(support.coords)

    if cache_path.is_file():
        try:
            payload = np.load(cache_path, allow_pickle=False)
            cache_version = int(np.asarray(payload["cache_version"]).reshape(-1)[0])
            cached_source_fp = str(np.asarray(payload["source_fingerprint"]).reshape(-1)[0])
            cached_support_fp = str(np.asarray(payload["support_fingerprint"]).reshape(-1)[0])
            cached_grid_size = int(np.asarray(payload["grid_size"]).reshape(-1)[0])

            if (
                cache_version == PROJECTION_CACHE_VERSION
                and cached_source_fp == source_fp
                and cached_support_fp == support_fp
                and cached_grid_size == COMMON_FRAME_GRID_SIZE
            ):
                source_xy = np.asarray(payload["source_xy"], dtype=np.float32)
                target_xy = np.asarray(payload["target_xy"], dtype=np.float32)
                print(f"[CACHE] Figure 2 projection {source.label}: {cache_path}")
                return source_xy, target_xy
            print(f"[CACHE] Stale Figure 2 projection cache; recomputing: {cache_path}")
        except Exception as exc:
            print(f"[WARN] Could not read projection cache {cache_path}: {exc}. Recomputing.")

    source_xy, target_xy = _project_to_regular_common_frame_grid(source.coords)

    np.savez_compressed(
        cache_path,
        cache_version=np.asarray(PROJECTION_CACHE_VERSION, dtype=np.int32),
        source_label=np.asarray(str(source.label)),
        source_path=np.asarray(str(source.path)),
        support_path=np.asarray(str(support.path)),
        source_fingerprint=np.asarray(source_fp),
        support_fingerprint=np.asarray(support_fp),
        grid_size=np.asarray(COMMON_FRAME_GRID_SIZE, dtype=np.int32),
        source_xy=source_xy,
        target_xy=target_xy,
    )
    unique_targets = len(np.unique(target_xy, axis=0)) if len(target_xy) else 0
    print(
        f"[SAVE] Figure 2 projection {source.label}: {len(source_xy)} observations -> "
        f"{unique_targets} common-frame pixels | {cache_path}"
    )
    return source_xy, target_xy


def _transition_field_cache_path(source: GeneMap) -> Path:
    TEMP_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    label = "".join(ch if ch.isalnum() else "_" for ch in str(source.label)).strip("_")
    return TEMP_RESULT_DIR / f"figure2_transition_field_{label}.npz"


def _load_or_build_transition_field(
    source: GeneMap,
    support: CommonFrameSupport,
    *,
    flow_source: float,
    flow_target: float,
    max_curves: int,
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load or build the displayed transition field for one stage.

    Cache design
    ------------
    The expensive spatial Bezier curves are cached in Y/Z only. WORLD-X is
    added at plotting time. Therefore changing Figure-2 spacing or physical
    depth offsets does NOT invalidate the cache.
    """
    source_fp = _projection_coord_fingerprint(source.coords)
    support_fp = _projection_coord_fingerprint(support.coords)
    cache_path = _transition_field_cache_path(source)

    expected = {
        "cache_version": int(TRANSITION_FIELD_CACHE_VERSION),
        "source_fingerprint": source_fp,
        "support_fingerprint": support_fp,
        "grid_size": int(COMMON_FRAME_GRID_SIZE),
        "max_curves": int(max_curves),
        "seed": int(seed),
        "n_steps": int(TRANSITION_CURVE_N_STEPS),
        "control_scale": float(TRANSITION_CONTROL_SCALE),
    }

    spatial_curves = None
    target_shown = None

    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                checks = [
                    int(np.asarray(payload["cache_version"]).reshape(-1)[0]) == expected["cache_version"],
                    str(np.asarray(payload["source_fingerprint"]).reshape(-1)[0]) == expected["source_fingerprint"],
                    str(np.asarray(payload["support_fingerprint"]).reshape(-1)[0]) == expected["support_fingerprint"],
                    int(np.asarray(payload["grid_size"]).reshape(-1)[0]) == expected["grid_size"],
                    int(np.asarray(payload["max_curves"]).reshape(-1)[0]) == expected["max_curves"],
                    int(np.asarray(payload["seed"]).reshape(-1)[0]) == expected["seed"],
                    int(np.asarray(payload["n_steps"]).reshape(-1)[0]) == expected["n_steps"],
                    np.isclose(float(np.asarray(payload["control_scale"]).reshape(-1)[0]), expected["control_scale"]),
                ]
                if all(checks) and "spatial_curves" in payload.files:
                    spatial_curves = np.asarray(payload["spatial_curves"], dtype=np.float32)
                    target_shown = np.asarray(payload["target_shown"], dtype=np.float32)
                    print(
                        f"[CACHE HIT] Figure 2 transition geometry {source.label}: "
                        f"{len(spatial_curves)} curves | {cache_path}"
                    )
                else:
                    print(f"[CACHE MISS] Figure 2 transition geometry changed: {cache_path}")
        except Exception as exc:
            print(f"[WARN] Could not read transition cache {cache_path}: {exc}. Recomputing.")

    if spatial_curves is None:
        source_xy, target_xy = project_observations_to_common_frame(source, support)

        n = min(len(source_xy), len(target_xy))
        source_xy = np.asarray(source_xy[:n], dtype=float)
        target_xy = np.asarray(target_xy[:n], dtype=float)
        valid = (
            np.all(np.isfinite(source_xy[:, :2]), axis=1)
            & np.all(np.isfinite(target_xy[:, :2]), axis=1)
        )
        source_xy = source_xy[valid]
        target_xy = target_xy[valid]

        if len(source_xy) == 0:
            spatial_curves = np.empty((0, TRANSITION_CURVE_N_STEPS, 2), dtype=np.float32)
            target_shown = np.empty((0, 2), dtype=np.float32)
        else:
            keep = choose_transition_subset_indices(len(source_xy), max_curves, seed)
            source_keep = source_xy[keep]
            target_shown = target_xy[keep].astype(np.float32)
            spatial_curves = np.stack(
                [build_smooth_spatial_curve(p0, p1) for p0, p1 in zip(source_keep, target_shown)],
                axis=0,
            ).astype(np.float32)

        np.savez_compressed(
            cache_path,
            cache_version=np.asarray(expected["cache_version"], dtype=np.int32),
            source_fingerprint=np.asarray(expected["source_fingerprint"]),
            support_fingerprint=np.asarray(expected["support_fingerprint"]),
            grid_size=np.asarray(expected["grid_size"], dtype=np.int32),
            max_curves=np.asarray(expected["max_curves"], dtype=np.int32),
            seed=np.asarray(expected["seed"], dtype=np.int32),
            n_steps=np.asarray(expected["n_steps"], dtype=np.int32),
            control_scale=np.asarray(expected["control_scale"], dtype=np.float32),
            spatial_curves=spatial_curves,
            target_shown=np.asarray(target_shown, dtype=np.float32),
        )
        print(
            f"[CACHE SAVE] Figure 2 transition geometry {source.label}: "
            f"{len(spatial_curves)} curves | {cache_path}"
        )

    # WORLD-X is cheap and is intentionally reconstructed every plotting run.
    if len(spatial_curves) == 0:
        return [], np.asarray(target_shown, dtype=np.float32)

    flow_axis = np.linspace(
        float(flow_source),
        float(flow_target),
        spatial_curves.shape[1],
        dtype=np.float32,
    )
    curves = [
        np.column_stack([flow_axis, curve[:, 0], curve[:, 1]]).astype(np.float32)
        for curve in spatial_curves
    ]
    return curves, np.asarray(target_shown, dtype=np.float32)


def choose_transition_subset_indices(n_total: int, n_keep: int, seed: int) -> np.ndarray:
    if n_total <= n_keep:
        return np.arange(n_total, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_keep, replace=False))


def cubic_bezier_2d(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    n_steps: int = TRANSITION_CURVE_N_STEPS,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(n_steps), dtype=float)[:, None]
    omt = 1.0 - t
    return (
        omt**3 * p0[None, :]
        + 3.0 * omt**2 * t * p1[None, :]
        + 3.0 * omt * t**2 * p2[None, :]
        + t**3 * p3[None, :]
    ).astype(np.float32)


def build_smooth_spatial_curve(
    start_yz: np.ndarray,
    end_yz: np.ndarray,
    control_scale: float = TRANSITION_CONTROL_SCALE,
    n_steps: int = TRANSITION_CURVE_N_STEPS,
) -> np.ndarray:
    start_yz = np.asarray(start_yz, dtype=float)
    end_yz = np.asarray(end_yz, dtype=float)
    disp = end_yz - start_yz
    dist = float(np.linalg.norm(disp))
    if not np.isfinite(dist) or dist <= 1e-8:
        return np.repeat(start_yz[None, :], n_steps, axis=0).astype(np.float32)

    unit = disp / dist
    perp = np.array([-unit[1], unit[0]], dtype=float)
    sign = -1.0 if start_yz[0] < end_yz[0] else 1.0
    bend = 0.18 * dist * sign * perp
    handle = control_scale * dist
    c1 = start_yz + handle * unit + 0.55 * bend
    c2 = end_yz - handle * unit + 0.25 * bend
    return cubic_bezier_2d(start_yz, c1, c2, end_yz, n_steps=n_steps)


def lift_transition_curve_to_3d(
    curve_yz: np.ndarray,
    flow_start: float,
    flow_end: float,
) -> np.ndarray:
    flow = np.linspace(float(flow_start), float(flow_end), len(curve_yz), dtype=float)
    return np.column_stack([flow, curve_yz[:, 0], curve_yz[:, 1]]).astype(np.float32)


def make_transition_arrowhead_segments_3d(
    tip: np.ndarray,
    previous: np.ndarray,
) -> Optional[np.ndarray]:
    tip = np.asarray(tip, dtype=float)
    previous = np.asarray(previous, dtype=float)
    tangent = tip - previous
    step_len = float(np.linalg.norm(tangent))
    if not np.isfinite(step_len) or step_len <= 1e-8:
        return None
    tangent = tangent / step_len

    head_len = TRANSITION_ARROW_HEAD_LENGTH_SCALE * step_len
    head_width = TRANSITION_ARROW_HEAD_WIDTH_SCALE * head_len

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


def build_transition_curves(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    flow_source: float,
    flow_target: float,
    max_curves: int,
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    n = min(len(source_coords), len(target_coords))
    if n == 0:
        return [], np.empty((0, 2), dtype=float)

    source_coords = np.asarray(source_coords[:n], dtype=float)
    target_coords = np.asarray(target_coords[:n], dtype=float)
    valid = (
        np.all(np.isfinite(source_coords[:, :2]), axis=1)
        & np.all(np.isfinite(target_coords[:, :2]), axis=1)
    )
    source_coords = source_coords[valid]
    target_coords = target_coords[valid]
    if len(source_coords) == 0:
        return [], np.empty((0, 2), dtype=float)

    keep = choose_transition_subset_indices(len(source_coords), max_curves, seed)
    source_coords = source_coords[keep]
    target_coords = target_coords[keep]

    curves = []
    for p0, p1 in zip(source_coords, target_coords):
        curve_yz = build_smooth_spatial_curve(p0, p1)
        curves.append(
            lift_transition_curve_to_3d(
                curve_yz,
                flow_start=flow_source,
                flow_end=flow_target,
            )
        )
    return curves, target_coords


def add_transition_curve_collection(
    ax,
    curves: Sequence[np.ndarray],
    color: str,
    *,
    curve_zorder: float = FIG2_ZORDER_TRANSITION_CURVE,
    arrow_zorder: float = FIG2_ZORDER_TRANSITION_ARROW,
) -> None:
    """Draw transition curves with a guaranteed visible landing on the target.

    Older mplot3d automatically reorders surface/grid collections. To keep the
    common-frame side of every field visible, only the final fraction of each
    3D curve is redrawn as a high-zorder Line3D. This avoids raising the whole
    field above the observed stage while guaranteeing:

        common frame -> grid -> transition landing -> observed stage.
    """
    for curve in curves:
        ax.plot(
            curve[:, 0], curve[:, 1], curve[:, 2],
            color=color,
            linewidth=TRANSITION_CURVE_LW,
            alpha=TRANSITION_CURVE_ALPHA,
            zorder=curve_zorder,
        )

        # Repaint only the common-frame-side tail above all 3D collections.
        n = len(curve)
        tail_n = max(4, int(np.ceil(n * float(TRANSITION_TARGET_TAIL_FRACTION))))
        tail = curve[-tail_n:]
        ax.plot(
            tail[:, 0], tail[:, 1], tail[:, 2],
            color=color,
            linewidth=TRANSITION_CURVE_LW,
            alpha=max(TRANSITION_CURVE_ALPHA, TRANSITION_TARGET_TAIL_ALPHA),
            zorder=TRANSITION_TARGET_TAIL_ZORDER,
        )

        if len(curve) >= 3:
            arrow_segments = make_transition_arrowhead_segments_3d(
                tip=curve[-1],
                previous=curve[-3],
            )
            if arrow_segments is not None:
                for seg in arrow_segments:
                    ax.plot(
                        seg[:, 0], seg[:, 1], seg[:, 2],
                        color=color,
                        linewidth=TRANSITION_CURVE_LW,
                        alpha=min(max(TRANSITION_TARGET_TAIL_ALPHA * 1.5, 0.30), 0.55),
                        zorder=TRANSITION_TARGET_ARROW_ZORDER,
                    )



# ============================================================
# MAP PLOTTING
# ============================================================
def _subset_by_mask(
    coords: np.ndarray,
    values: np.ndarray,
    mask: Optional[np.ndarray],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if mask is None:
        mask = np.ones(len(coords), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    idx_all = np.where(mask)[0]
    idx_keep = sample_indices(len(idx_all), MAX_POINTS_PER_PLANE, seed)
    idx = idx_all[idx_keep]
    return np.asarray(coords, dtype=float)[idx], np.asarray(values)[idx]


def plot_gene_plane_3d(
    ax,
    coords: np.ndarray,
    values: np.ndarray,
    flow_x: float,
    vmax: float,
    alpha: float,
    seed: int,
    frame_extent: Tuple[float, float, float, float] = FIXED_BOX,
    *,
    points_zorder: float = 2,
    frame_zorder: float = 20,
) -> None:
    c, v = _subset_by_mask(coords, values, None, seed)
    colors = gene_facecolors(v, vmax=vmax, alpha_factor=alpha)
    ax.scatter(
        np.full(len(c), float(flow_x)),
        c[:, 0],
        c[:, 1],
        c=colors,
        s=POINT_SIZE_GENE,
        linewidths=0,
        depthshade=False,
        rasterized=True,
        zorder=points_zorder,
    )
    draw_plane_frame_3d(ax, flow_x, frame_extent, kind="observed", zorder=frame_zorder)


def plot_delta_plane_3d(
    ax,
    coords: np.ndarray,
    values: np.ndarray,
    flow_x: float,
    vmax: float,
    seed: int,
    frame_extent: Tuple[float, float, float, float] = FIXED_BOX,
    mask: Optional[np.ndarray] = None,
) -> None:
    c, v = _subset_by_mask(coords, values, mask, seed)
    colors = delta_facecolors(v, vmax=vmax)
    ax.scatter(
        np.full(len(c), float(flow_x)),
        c[:, 0],
        c[:, 1],
        c=colors,
        s=POINT_SIZE_DELTA,
        linewidths=0,
        depthshade=False,
        rasterized=True,
        zorder=5,
    )
    draw_plane_frame_3d(ax, flow_x, frame_extent, kind="delta")


def plot_trend_plane_3d(
    ax,
    coords: np.ndarray,
    patterns: np.ndarray,
    flow_x: float,
    seed: int,
    frame_extent: Tuple[float, float, float, float] = FIXED_BOX,
    mask: Optional[np.ndarray] = None,
) -> None:
    if mask is None:
        mask = np.ones(len(coords), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    idx_all = np.where(mask)[0]
    idx_keep = sample_indices(len(idx_all), MAX_POINTS_PER_PLANE, seed)
    idx = idx_all[idx_keep]

    c = np.asarray(coords, dtype=float)[idx]
    p = np.asarray(patterns, dtype=object)[idx]

    stable = p == "Stable"
    if np.any(stable):
        ax.scatter(
            np.full(np.sum(stable), float(flow_x)),
            c[stable, 0],
            c[stable, 1],
            color=PATTERN_COLORS["Stable"],
            s=POINT_SIZE_TREND,
            alpha=0.58,
            linewidths=0,
            depthshade=False,
            rasterized=True,
            zorder=4,
        )

    for patt in ["Increasing", "Decreasing", "Peak", "Trough"]:
        m = p == patt
        if np.any(m):
            ax.scatter(
                np.full(np.sum(m), float(flow_x)),
                c[m, 0],
                c[m, 1],
                color=PATTERN_COLORS[patt],
                s=POINT_SIZE_TREND,
                alpha=TREND_ALPHA,
                linewidths=0,
                depthshade=False,
                rasterized=True,
                zorder=6,
            )

    draw_plane_frame_3d(ax, flow_x, frame_extent, kind="trend")


# ============================================================
# LEGENDS / COLORBARS
# ============================================================
def add_gene_colorbar(fig, ax, vmax: float, shrink: float = 0.68, pad: float = 0.02) -> None:
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap="Reds")
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=pad, shrink=shrink)
    cb.set_label(f"log1p {GENE_NAME} expression", fontsize=10)
    cb.ax.tick_params(labelsize=9)


def add_delta_colorbar(fig, ax, vmax: float, shrink: float = 0.68, pad: float = 0.08) -> None:
    sm = ScalarMappable(
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        cmap="coolwarm",
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=pad, shrink=shrink)
    cb.set_label("Spatial expression change", fontsize=10)
    cb.ax.tick_params(labelsize=9)


def trend_legend_handles() -> List[Line2D]:
    return [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=PATTERN_COLORS[p],
            markeredgecolor="none",
            markersize=7.2,
            label=p,
        )
        for p in PATTERN_ORDER
    ]



# ============================================================
# FIGURE 1
# ============================================================
def plot_figure1_observed_raw(
    raw_records: Sequence[GeneMap],
    outdir: str,
) -> None:
    set_style()
    fig = plt.figure(figsize=(11.6, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    ordered = sorted(raw_records, key=lambda r: r.time)
    # Stage-oriented display: 5.7m is exactly centered between 2.5m and 17.9m.
    flow_positions = equal_stage_positions(len(ordered), gap=STAGE_GAP).tolist()
    vmax = robust_positive_vmax([r.values for r in ordered], q=99.0)

    for i, (record, x) in enumerate(zip(ordered, flow_positions)):
        plot_gene_plane_3d(
            ax, record.coords, record.values, x,
            vmax=vmax,
            alpha=GENE_ALPHA_FIG1,
            seed=100 + i,
        )

    style_3d_axis(ax, FIXED_BOX, flow_positions)
    # add_gene_colorbar(fig, ax, vmax=vmax)

    layers = [LayerMeta(STAGE_LABELS[i], "observed") for i, _ in enumerate(ordered)]
    draw_horizontal_hanging_scaffold(
        fig, ax, layers, flow_positions, FIXED_BOX, title_label="  AD\n stage"
    )

    # ax.set_title(
    #     "Observed spatial omics across AD progression stages",
    #     pad=18,
    #     fontsize=15,
    #     fontweight="bold",
    # )
    finalize_figure(
        fig,
        os.path.join(outdir, f"figure1_observed_raw_left_to_right_hanging_{GENE_NAME}.png"),
    )


# ============================================================
# FIGURE 2 — OBSERVED STATES -> REPEATED STAGE-1 COMMON FRAME
# ============================================================
def plot_figure2_registration_transition_fields(
    raw_records: Sequence[GeneMap],
    common_maps: Sequence[GeneMap],
    outdir: str,
) -> None:
    """
    For each observed stage, place a copy of the Stage-1 shared spatial frame
    immediately to its left and visualize a 3D projection field from the
    observed state onto that common frame.

    Layout:
        Shared frame | Stage 1    Shared frame | Stage 2    Shared frame | Stage 3

    All three shared-frame copies represent the SAME Stage-1 reference support.
    Curve endpoints are quantized directly onto the full 256x256 Stage-1
    common-frame lattice; no one-to-one spot correspondence is assumed.
    """
    set_style()
    fig = plt.figure(figsize=(15.6, 7.6))
    # Newer Matplotlib versions can disable automatic collection sorting.
    # Older versions (including the Python-3.8 environment used for this
    # project) use a small physical depth separation instead.
    ax, fig2_manual_zorder = _figure2_create_3d_axis(fig)

    raw_ordered = sorted(raw_records, key=lambda r: r.time)
    common_ordered = sorted(common_maps, key=lambda r: r.time)
    if len(raw_ordered) != 3 or len(common_ordered) != 3:
        raise ValueError("Figure 2 expects exactly three observed and three common-frame maps.")

    support = load_stage1_common_frame_support()
    vmax = robust_positive_vmax([r.values for r in raw_ordered], q=99.0)

    frame_positions = []
    stage_positions = []
    flow_positions = []
    layers: List[LayerMeta] = []

    for i in range(3):
        x_frame = i * float(REGISTRATION_GROUP_GAP)
        x_stage = x_frame + float(REGISTRATION_PAIR_GAP)
        frame_positions.append(x_frame)
        stage_positions.append(x_stage)
        flow_positions.extend([x_frame, x_stage])
        layers.extend([
            LayerMeta("Shared frame", "delta"),
            LayerMeta(STAGE_LABELS[i], "observed"),
        ])

    # ------------------------------------------------------------
    # Draw each pair locally in the exact intended visual order:
    # common frame -> common-frame grid/frame -> transition field -> stage.
    # ------------------------------------------------------------
    for i, (raw_map, x_frame, x_stage) in enumerate(
        zip(raw_ordered, frame_positions, stage_positions)
    ):
        toward_sign = np.sign(float(x_stage) - float(x_frame)) or 1.0

        # 1) Domain-colored common-frame surface.
        # 2) Grid/frame overlay is drawn inside this helper, physically a little
        #    closer to the stage than the domain surface.
        plot_common_frame_support_plane_3d(
            ax,
            support=support,
            flow_x=x_frame,
            frame_extent=FIXED_BOX,
            toward_stage_sign=toward_sign,
            use_physical_depth_fallback=True,
        )

        # 3) Transition field. Endpoints sit on the stage-facing side of the
        #    grid/frame, and the target-side curve tails are explicitly repainted
        #    above all surface/grid collections for old Matplotlib.
        flow_target = (
            float(x_frame)
            + float(toward_sign) * float(FIG2_FALLBACK_TARGET_FORWARDSTEP)
        )
        curves, target_shown = _load_or_build_transition_field(
            raw_map,
            support,
            flow_source=x_stage,
            flow_target=flow_target,
            max_curves=MAX_TRANSITION_CURVES_PER_STAGE,
            seed=TRANSITION_RANDOM_SEED + i,
        )
        curve_color = TRANSITION_CURVE_COLORS[i % len(TRANSITION_CURVE_COLORS)]
        add_transition_curve_collection(ax, curves, color=curve_color)

        if len(target_shown) > 0:
            # Use Line3D markers rather than scatter so old mplot3d cannot
            # re-sort the endpoint markers behind the common-frame surface.
            ax.plot(
                np.full(len(target_shown), flow_target),
                target_shown[:, 0],
                target_shown[:, 1],
                linestyle="",
                marker="o",
                markersize=max(np.sqrt(float(TRANSITION_ENDPOINT_SIZE)), 1.8),
                markerfacecolor=curve_color,
                markeredgecolor="none",
                alpha=max(TRANSITION_ENDPOINT_ALPHA, 0.52),
                zorder=TRANSITION_TARGET_ENDPOINT_ZORDER,
            )

        # 4) Observed stage, unchanged from Figure 1.
        plot_gene_plane_3d(
            ax,
            raw_map.coords,
            raw_map.values,
            x_stage,
            vmax=vmax,
            alpha=GENE_ALPHA_FIG1,
            seed=240 + i,
            points_zorder=FIG2_ZORDER_STAGE_POINTS,
            frame_zorder=FIG2_ZORDER_STAGE_FRAME,
        )

    style_3d_axis(ax, FIXED_BOX, flow_positions)

    draw_horizontal_hanging_scaffold(
        fig,
        ax,
        layers,
        flow_positions,
        FIXED_BOX,
        title_label="Projection",
    )

    handles = [
        Line2D(
            [0], [0],
            color=DERIVED_FRAME_COLOR,
            linewidth=1.5,
            linestyle=DERIVED_STYLE,
            label="Shared Stage-1 frame",
        ),
        Line2D(
            [0], [0],
            color=OBSERVED_FRAME_COLOR,
            linewidth=1.5,
            linestyle="-",
            label="Observed state",
        ),
    ]
    handles.extend(
        Line2D(
            [0], [0],
            color=TRANSITION_CURVE_COLORS[i],
            linewidth=2.0,
            label=f"{STAGE_LABELS[i]} → shared frame",
        )
        for i in range(3)
    )
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
        fontsize=8.5,
    )

    ax.text2D(
        0.035,
        0.035,
        "Each dashed pixel/domain plane is the same Stage-1 reference frame; "
        "curves project observed locations onto its 256×256 pixel lattice.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="0.36",
        ha="left",
        va="bottom",
    )
    # ax.set_title(
    #     "Spatial registration into the shared Stage-1 frame",
    #     pad=18,
    #     fontsize=15,
    #     fontweight="bold",
    # )
    finalize_figure(
        fig,
        os.path.join(
            outdir,
            f"figure2_observed_to_common_frame_transition_field_{GENE_NAME}.png",
        ),
    )


# ============================================================
# FIGURE 3 — ORIGINAL COMMON-FRAME STAGE STACK
# ============================================================
def plot_figure3_common_frame(
    common_maps: Sequence[GeneMap],
    outdir: str,
) -> None:
    """Original three-plane common-frame visualization, without support plane."""
    set_style()
    fig = plt.figure(figsize=(11.6, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    ordered = sorted(common_maps, key=lambda r: r.time)
    flow_positions = equal_stage_positions(len(ordered), gap=STAGE_GAP).tolist()
    vmax = robust_positive_vmax([r.values for r in ordered], q=99.0)

    for i, (record, x) in enumerate(zip(ordered, flow_positions)):
        plot_gene_plane_3d(
            ax,
            record.coords,
            record.values,
            x,
            vmax=vmax,
            alpha=GENE_ALPHA_FIG2,
            seed=200 + i,
        )

    style_3d_axis(ax, FIXED_BOX, flow_positions)

    layers = [LayerMeta(STAGE_LABELS[i], "observed") for i, _ in enumerate(ordered)]
    draw_horizontal_hanging_scaffold(
        fig,
        ax,
        layers,
        flow_positions,
        FIXED_BOX,
        title_label="  AD \nstage",
    )

    ax.text2D(
        0.04,
        0.04,
        "Common-frame tissue maps loaded directly",
        transform=ax.transAxes,
        fontsize=10.5,
        color="0.35",
        ha="left",
        va="bottom",
    )
    # ax.set_title(
    #     "Aligned tissues across AD progression stages",
    #     pad=18,
    #     fontsize=15,
    #     fontweight="bold",
    # )
    finalize_figure(
        fig,
        os.path.join(
            outdir,
            f"figure3_common_frame_left_to_right_hanging_{GENE_NAME}.png",
        ),
    )


# ============================================================
# FIGURE 4
# ============================================================
def plot_figure4_stage_delta_curves(
    common_maps: Sequence[GeneMap],
    spatial_map: SpatialTemporalMap,
    outdir: str,
) -> None:
    set_style()
    fig = plt.figure(figsize=(14.0, 7.8))
    ax = fig.add_subplot(111, projection="3d")

    # Equal stage spacing intentionally emphasizes progression stage rather than
    # literal month differences.
    x_t1, x_t2, x_t3 = equal_stage_positions(3, gap=STAGE_GAP)
    x_d12 = 0.5 * (x_t1 + x_t2)
    x_d23 = 0.5 * (x_t2 + x_t3)
    flow_positions = [x_t1, x_d12, x_t2, x_d23, x_t3]

    layers = [
        LayerMeta("Stage 1", "observed"),
        LayerMeta("Δ 1→2", "delta"),
        LayerMeta("Stage 2", "observed"),
        LayerMeta("Δ 2→3", "delta"),
        LayerMeta("Stage 3", "observed"),
    ]

    gene_vmax = robust_positive_vmax([r.values for r in common_maps], q=99.0)
    delta_vmax = robust_symmetric_vmax([spatial_map.delta_12, spatial_map.delta_23], q=99.0)
    valid_mask = spatial_map.valid_domain_mask

    plot_gene_plane_3d(ax, common_maps[0].coords, common_maps[0].values, x_t1, gene_vmax, GENE_ALPHA_WITH_DELTA, seed=300)
    plot_delta_plane_3d(ax, spatial_map.coords, spatial_map.delta_12, x_d12, delta_vmax, seed=301, mask=valid_mask)
    plot_gene_plane_3d(ax, common_maps[1].coords, common_maps[1].values, x_t2, gene_vmax, GENE_ALPHA_WITH_DELTA, seed=302)
    plot_delta_plane_3d(ax, spatial_map.coords, spatial_map.delta_23, x_d23, delta_vmax, seed=303, mask=valid_mask)
    plot_gene_plane_3d(ax, common_maps[2].coords, common_maps[2].values, x_t3, gene_vmax, GENE_ALPHA_WITH_DELTA, seed=304)

    style_3d_axis(ax, FIXED_BOX, flow_positions,spatial_scale=0.2)
    # add_gene_colorbar(fig, ax, vmax=gene_vmax, shrink=0.62, pad=0.02)
    # add_delta_colorbar(fig, ax, vmax=delta_vmax, shrink=0.62, pad=0.09)

    draw_horizontal_hanging_scaffold(
        fig, ax, layers, flow_positions, FIXED_BOX,
        title_label="  AD\nstage",
    )

    # Each delta is explicitly derived from its two bracketing stage maps.
    # Arrows connect the projected upper-frame edge midpoints, so the 3D maps
    # remain rotatable while the arrows retain a clean paper-schematic form.
    draw_bracket_to_delta_curves(fig, ax, x_t1, x_d12, x_t2, FIXED_BOX)
    draw_bracket_to_delta_curves(fig, ax, x_t2, x_d23, x_t3, FIXED_BOX)

    handles = [
        Line2D([0], [0], color=OBSERVED_FRAME_COLOR, linewidth=1.5, linestyle="-", label="Stage map"),
        Line2D([0], [0], color=DERIVED_FRAME_COLOR, linewidth=1.5, linestyle=DERIVED_STYLE, label="Spatial Δ map"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.98, 0.97), frameon=False, fontsize=8.8)

    # ax.text2D(
    #     0.04, 0.04,
    #     "Spatially matched comparison: each Δ map is derived from its two bracketing progression stages",
    #     transform=ax.transAxes, fontsize=9.8, color="0.38", ha="left", va="bottom",
    # )
    # ax.set_title(
    #     f"Spatially matched change across AD progression stages ({spatial_map.gene})",
    #     pad=18, fontsize=15, fontweight="bold",
    # )
    finalize_figure(
        fig,
        os.path.join(outdir, f"figure4_stage_delta_curved_arrows_{GENE_NAME}.png"),
    )


# ============================================================
# FIGURE 5 — CONCEPTUAL TREND ILLUSTRATION
# ============================================================
def plot_figure5_trend_cartoon(outdir: str) -> None:
    """
    Conceptual illustration of spatially resolved temporal trend patterns.

    All curves start from the same Stage-1 baseline.

    The Mixed case is illustrated by several weak / irregular trajectories
    confined within positive and negative thresholds. Changes inside this band
    are treated as having no clear temporal trend.
    """
    set_style()

    x_stage = np.array([1.0, 2.0, 3.0])
    x_smooth = np.linspace(1.0, 3.0, 240)

    # --------------------------------------------------------
    # Threshold defining the no-clear-trend / Mixed region.
    # --------------------------------------------------------
    stable_threshold = 0.18

    # Clear temporal-pattern examples.
    trend_points = {
        "Increasing": np.array([0.0, 0.42, 0.92]),
        "Decreasing": np.array([0.0, -0.40, -0.88]),
        "Peak":       np.array([0.0, 0.86, 0.26]),
        "Trough":     np.array([0.0, -0.84, -0.24]),
    }

    # Multiple conceptual Mixed trajectories.
    # They all remain within +/- mixed_threshold.
    stable_curves = [
        np.array([0.0,  0.10, -0.05]),
        np.array([0.0, -0.08,  0.12]),
        np.array([0.0,  0.04,  0.02]),
        np.array([0.0, -0.12, -0.03]),
    ]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # ========================================================
    # NO-TREND THRESHOLD BAND
    # ========================================================
    ax.axhline(
        +stable_threshold,
        color="0.55",
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.axhline(
        -stable_threshold,
        color="0.55",
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        zorder=0,
    )

    # Light visual band between thresholds.
    ax.axhspan(
        -stable_threshold,
        +stable_threshold,
        color="0.94",
        alpha=0.55,
        zorder=-1,
    )

    # ax.text(
    #     3.08,
    #     +stable_threshold,
    #     "+ threshold",
    #     fontsize=9.0,
    #     color="0.45",
    #     ha="left",
    #     va="center",
    # )
    #
    # ax.text(
    #     3.08,
    #     -stable_threshold,
    #     "− threshold",
    #     fontsize=9.0,
    #     color="0.45",
    #     ha="left",
    #     va="center",
    # )

    # ========================================================
    # CLEAR TREND CURVES
    # ========================================================
    for patt in ["Increasing", "Decreasing", "Peak", "Trough"]:

        y_pts = trend_points[patt]

        # Smooth quadratic curve through the three stages.
        coeff = np.polyfit(
            x_stage,
            y_pts,
            deg=2,
        )
        y_smooth = np.polyval(
            coeff,
            x_smooth,
        )

        ax.plot(
            x_smooth,
            y_smooth,
            color=PATTERN_COLORS[patt],
            linewidth=2.45,
            alpha=0.96,
            solid_capstyle="round",
            zorder=3,
        )

        ax.scatter(
            x_stage,
            y_pts,
            s=42,
            color=PATTERN_COLORS[patt],
            edgecolors="white",
            linewidths=0.75,
            zorder=4,
        )

        ax.text(
            3.07,
            y_pts[-1],
            patt,
            color=PATTERN_COLORS[patt],
            fontsize=20,
            ha="left",
            va="center",
        )

    # ========================================================
    # MIXED / NO-CLEAR-TREND CASE
    # ========================================================
    for i, y_pts in enumerate(stable_curves):

        coeff = np.polyfit(
            x_stage,
            y_pts,
            deg=2,
        )
        y_smooth = np.polyval(
            coeff,
            x_smooth,
        )

        ax.plot(
            x_smooth,
            y_smooth,
            color=PATTERN_COLORS["Stable"],
            linewidth=1.65,
            alpha=0.72,
            solid_capstyle="round",
            zorder=2,
        )

        ax.scatter(
            x_stage,
            y_pts,
            s=23,
            color=PATTERN_COLORS["Stable"],
            edgecolors="white",
            linewidths=0.45,
            alpha=0.80,
            zorder=3,
        )

    ax.text(
        3.07,
        0.0,
        "Stable",
        color=PATTERN_COLORS["Stable"],
        fontsize=20,
        ha="left",
        va="center",
        zorder=5,
    )

    # ax.text(
    #     2.15,
    #     -0.075,
    #     "no clear trend",
    #     color="0.45",
    #     fontsize=8.5,
    #     ha="center",
    #     va="center",
    #     zorder=5,
    # )

    # ========================================================
    # SHARED STAGE-1 BASELINE
    # ========================================================
    ax.scatter(
        [1.0],
        [0.0],
        s=84,
        facecolor="white",
        edgecolor="#333333",
        linewidth=1.4,
        zorder=7,
    )

    # ax.text(
    #     1.06,
    #     0.055,
    #     "shared baseline",
    #     fontsize=9.2,
    #     color="0.35",
    #     ha="left",
    #     va="bottom",
    # )

    # ========================================================
    # AXIS DESIGN
    # ========================================================
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Zero-reference line.
    ax.axhline(
        0.0,
        color="0.76",
        linewidth=0.8,
        linestyle=(0, (2, 2)),
        zorder=0,
    )

    # Vertical stage guides.
    ax.grid(
        axis="x",
        color="0.90",
        linewidth=0.8,
        linestyle=(0, (2, 2)),
    )

    # Compatible with older Matplotlib versions.
    ax.set_xticks(x_stage)
    ax.set_xticklabels(
        ["Stage 1", "Stage 2", "Stage 3"]
    )
    ax.set_xticklabels([]
    )
    ax.set_yticklabels([]
                       )

    ax.set_xlabel(
        "Progression stage",
        fontsize=16,
        labelpad=9,
    )

    ax.set_ylabel(
        "Gene variation",
        fontsize=16,
        labelpad=9,
    )

    ax.set_xlim(0.90, 3.58)
    ax.set_ylim(-1.05, 1.08)

    ax.tick_params(
        axis="both",
        labelsize=9.5,
    )

    # ========================================================
    # TITLE / ANNOTATION
    # ========================================================

    fig.tight_layout()

    finalize_figure(
        fig,
        os.path.join(
            outdir,
            "figure5_progression_stage_trend_cartoon.png",
        ),
    )

# ============================================================
# RUN
# ============================================================
def run_all(
    gene: str = GENE_NAME,
    spatial_map_path: Optional[str] = None,
    make_cartoon: bool = True,
) -> None:
    ensure_dir(OUTDIR)
    TEMP_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Figure 2 cache directory: {TEMP_RESULT_DIR}")

    raw_maps = load_original_observed_maps(gene=gene)
    common_maps = load_common_frame_gene_maps(gene=gene)
    spatial_map = load_saved_spatial_temporal_map(
        gene=gene,
        explicit_path=spatial_map_path,
    )

    globals()["GENE_NAME"] = gene

    plot_figure1_observed_raw(raw_maps, OUTDIR)
    # plot_figure2_registration_transition_fields(raw_maps, common_maps, OUTDIR)
    plot_figure3_common_frame(common_maps, OUTDIR)
    plot_figure4_stage_delta_curves(common_maps, spatial_map, OUTDIR)
    if make_cartoon:
        plot_figure5_trend_cartoon(OUTDIR)

    print(f"[DONE] Plot-only left-to-right pipeline figures written/displayed from: {OUTDIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate left-to-right 3D spatial-matched temporal pipeline figures "
            "with a screen-horizontal scaffold and hanging maps."
        )
    )
    parser.add_argument("--figure-mode", choices=["save", "show", "both"], default=FIGURE_MODE)
    parser.add_argument("--gene", default=GENE_NAME)
    parser.add_argument("--spatial-map", default=None)

    parser.add_argument(
        "--stage-gap",
        type=float,
        default=STAGE_GAP,
        help="Equal WORLD-X spacing between Stage 1, Stage 2, and Stage 3.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=TIME_SCALE_PER_MONTH,
        help="Legacy month-proportional scale; retained for compatibility.",
    )
    parser.add_argument("--spatial-display-scale", type=float, default=SPATIAL_DISPLAY_SCALE)
    parser.add_argument(
        "--registration-pair-gap",
        type=float,
        default=REGISTRATION_PAIR_GAP,
        help="WORLD-X spacing from each shared-frame copy to its observed stage in Figure 2.",
    )
    parser.add_argument(
        "--max-transition-curves",
        type=int,
        default=MAX_TRANSITION_CURVES_PER_STAGE,
        help="Maximum number of 3D transition curves drawn per stage in Figure 2.",
    )

    parser.add_argument("--view-elev", type=float, default=VIEW_ELEV)
    parser.add_argument("--view-azim", type=float, default=VIEW_AZIM)
    parser.add_argument("--view-roll", type=float, default=VIEW_ROLL)

    parser.add_argument("--timeline-clearance", type=float, default=TIMELINE_CLEARANCE_FIG)


    parser.add_argument("--no-cartoon", action="store_true")
    args = parser.parse_args()

    globals()["FIGURE_MODE"] = str(args.figure_mode)
    globals()["GENE_NAME"] = str(args.gene)
    globals()["STAGE_GAP"] = float(args.stage_gap)
    globals()["REGISTRATION_GROUP_GAP"] = float(args.stage_gap)
    globals()["TIME_SCALE_PER_MONTH"] = float(args.time_scale)
    globals()["SPATIAL_DISPLAY_SCALE"] = float(args.spatial_display_scale)
    globals()["REGISTRATION_PAIR_GAP"] = float(args.registration_pair_gap)
    globals()["MAX_TRANSITION_CURVES_PER_STAGE"] = int(args.max_transition_curves)

    globals()["VIEW_ELEV"] = float(args.view_elev)
    globals()["VIEW_AZIM"] = float(args.view_azim)
    globals()["VIEW_ROLL"] = float(args.view_roll)
    globals()["TIMELINE_CLEARANCE_FIG"] = float(args.timeline_clearance)


    print(f"[INFO] FIGURE_MODE = {FIGURE_MODE}")
    print(f"[INFO] GENE_NAME = {GENE_NAME}")
    print(f"[INFO] STAGE_GAP = {STAGE_GAP}")
    print(f"[INFO] TIME_SCALE_PER_MONTH = {TIME_SCALE_PER_MONTH} (legacy)")
    print(f"[INFO] SPATIAL_DISPLAY_SCALE = {SPATIAL_DISPLAY_SCALE}")
    print(f"[INFO] REGISTRATION_PAIR_GAP = {REGISTRATION_PAIR_GAP}")
    print(f"[INFO] MAX_TRANSITION_CURVES_PER_STAGE = {MAX_TRANSITION_CURVES_PER_STAGE}")
    print(f"[INFO] VIEW_ELEV = {VIEW_ELEV}")
    print(f"[INFO] VIEW_AZIM = {VIEW_AZIM}")
    print(f"[INFO] VIEW_ROLL = {VIEW_ROLL}")
    print(f"[INFO] TIMELINE_CLEARANCE_FIG = {TIMELINE_CLEARANCE_FIG}")

    run_all(
        gene=GENE_NAME,
        spatial_map_path=args.spatial_map,
        make_cartoon=not args.no_cartoon,
    )


if __name__ == "__main__":
    main()
