#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Left-to-right 3D pipeline for spatially matched AD progression analysis.

Visual design
-------------
- WORLD X = progression stage / analytical flow (left -> right).
- WORLD Y/Z = spatial coordinates.
- The horizontal progression scaffold is rendered in 2D figure space so it
  remains horizontal under 3D map rotation.
- Observed slices use solid frames; spatially matched slices use dashed frames.
- In Figure 2, observed Stage 2/3 maps are projection-only sources: they have
  no frame, timeline marker, hanger, or scaffold label.
- Figure 2 uses independent spacing controls for reference->matched Stage 2,
- matched Stage 2->matched Stage 3, and matched->projection slices.
- Figure 3 places each delta map midway between its bracketing matched stages.
- Figure 4 is an idea-oriented trend cartoon.

Figures
-------
1. Raw observed gene maps: Stage 1 -> Stage 2 -> Stage 3.
2. Spatial-matching transition fields: Stage 1 reference plus matched Stage 2/3,
   with projection-only observed Stage 2/3 source maps.
3. Spatially matched stage maps + delta maps with paired curved arrows.
4. Conceptual gene-variation trends across progression stages.
"""

from axis.settings import data_path
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from axis.registration import find_inverse_warp_coords
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree


# ============================================================
# DATA CONFIG
# ============================================================
RAW_ROOT = data_path("mouse_brain")
RAW_NET_IN = os.path.join(RAW_ROOT, "network_input")
RAW_H5AD_DIR = os.path.join(RAW_ROOT, "h5ads", "Tg")
GROUP_ID = 0
PAIRWISE_SUBDIR = "pairwise_output"

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
            data_path("mouse_brain/"
            "virtual_tissue/AD_time_grid256_spatial2.0/analysis/"
            "virtual_AD_time_2.500.h5ad")
        ),
    },
    {
        "label": "5.7m",
        "time": 5.7,
        "path": (
            data_path("mouse_brain/"
            "virtual_tissue/AD_time_grid256_spatial2.0/analysis/"
            "virtual_AD_time_5.700.h5ad")
        ),
    },
    {
        "label": "17.9m",
        "time": 17.9,
        "path": (
            data_path("mouse_brain/"
            "virtual_tissue/AD_time_grid256_spatial2.0/analysis/"
            "virtual_AD_time_17.900.h5ad")
        ),
    },
]

SOURCE_OUTDIR = data_path("paper_analysis/Figure56_mouse/5ghi/")
SPATIAL_MAP_DIR = os.path.join(SOURCE_OUTDIR, "spatial_maps")
SPATIAL_MAP_TEMPLATE = "{gene}_spatial_temporal_map.npz"
OUTDIR = data_path("paper_analysis/Figure56_mouse/5f/")

COORD_KEY = "spatial"
GENE_NAME = "Arc"

# Spatially matched H5AD validity flag.
COMMON_FRAME_VALID_KEY = "valid"

# Figure 2 transition trajectories.
TRAJECTORY_DOMAIN_KEY = "axis_joint_domain"
TRAJECTORY_DOMAINS: Optional[Sequence[str]] = ("8", "11")

# Figure 2 layout:
# - reference -> matched Stage 2
# - matched Stage 2 -> matched Stage 3
# - each matched state -> its projection-only observed source
FIG2_REFERENCE_TO_STAGE2_GAP = 300.0
FIG2_STAGE2_TO_STAGE3_GAP = 500.0
REGISTRATION_PAIR_GAP = 220.0

MAX_TRANSITION_CURVES_PER_STAGE = 420
TRANSITION_RANDOM_SEED = 37
TRANSITION_CURVE_N_STEPS = 44
TRANSITION_CURVE_LW = 1.35
TRANSITION_CURVE_ALPHA = 0.10
TRANSITION_CURVE_COLORS = ["#4E79A7", "#D3725B", "#D3725B"]

# Figure 2 opacity:
# - reference + matched states share one control;
# - observed Stage 2/3 projection sources use a separate control.
FIG2_MAIN_POINT_ALPHA = 0.40
FIG2_PROJECTION_POINT_ALPHA = 0.2

# Matched-state point boundaries in Figure 2.
FIG2_MATCHED_MARKER_EDGE_COLOR = "0.5"
FIG2_MATCHED_MARKER_EDGE_WIDTH = 0.02

# Unified slice-frame style.
SLICE_FRAME_COLOR = "#8A8A8A"
SLICE_FRAME_LINEWIDTH = 1.0
MATCHED_FRAME_DASH_STYLE = (0, (4.0, 2.4))
FIG2_REFERENCE_PLANE_FACE_ALPHA = 0.02

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
FIGURE_MODE = "both"  # save/show/both
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
# Screen-space curved comparison arrows used in Figure 3.
CURVE_ARROW_COLOR = "#707070"
CURVE_ARROW_ALPHA = 0.72
CURVE_ARROW_LW = 1.7
CURVE_ARROW_SCALE = 12.0
CURVE_ARROW_RAD = 1.
CURVE_ENDPOINT_SHRINK = 4.0

# Retained only for backward CLI compatibility; no longer used to position the
# three observed stages in the stage-based panels.
TIME_SCALE_PER_MONTH = 60.0


# Uniform visual scale of spatial Y/Z planes relative to WORLD-X.
SPATIAL_DISPLAY_SCALE = 0.35

POINT_SIZE_GENE = 1.2
POINT_SIZE_DELTA = 1.7
MAX_POINTS_PER_PLANE = 30000

GENE_ALPHA_FIG1 = 0.74
GENE_ALPHA_WITH_DELTA = 0.40
DELTA_ALPHA = 0.56

OBSERVED_LABEL_COLOR = "#1F1F1F"
DERIVED_LABEL_COLOR = "#707070"
TREND_LABEL_COLOR = "#555555"

DERIVED_STYLE = MATCHED_FRAME_DASH_STYLE

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
HANGER_TOP_GAP_FIG = -0.003
HANGER_BOTTOM_GAP_FIG = -0.003
HANGER_ANCHOR_SIZE = 3.8


# Camera affects ONLY 3D objects.
VIEW_ELEV = 5.0
VIEW_AZIM = -50.0
VIEW_ROLL = 0.0

PATTERN_COLORS = {
    "Increase": "#D55E00",
    "Decrease": "#0072B2",
    "Mid-stage peak": "#009E73",
    "Mid-stage trough": "#CC79A7",
    "Stable": "#797979",
}
PATTERN_ORDER = [
    "Increase",
    "Decrease",
    "Mid-stage peak",
    "Mid-stage trough",
    "Stable",
]



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


def find_pairwise_group_dir(
    root: str,
    group_id: int,
    pairwise_subdir: str = PAIRWISE_SUBDIR,
) -> str:
    """Return the AXIS pairwise-registration directory for one group."""
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


def parse_pairwise_flow_path(
    root: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
    pairwise_subdir: str = PAIRWISE_SUBDIR,
) -> str:
    """AXIS convention: <fixed>_<moving>_flow.npy."""
    group_dir = find_pairwise_group_dir(
        root,
        group_id,
        pairwise_subdir=pairwise_subdir,
    )
    return os.path.join(
        group_dir,
        f"{fixed_id}_{moving_id}_flow.npy",
    )


def ensure_flow_hw2(flow: np.ndarray) -> np.ndarray:
    """Normalize a deformation field to [H, W, 2]."""
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3:
        raise ValueError(f"Flow must be 3D, got {flow.shape}.")
    if flow.shape[-1] == 2:
        return flow
    if flow.shape[0] == 2:
        return np.moveaxis(flow, 0, -1)
    raise ValueError(f"Cannot interpret flow shape {flow.shape} as [H,W,2].")


def inverse_warp_points(
    points_xy: np.ndarray,
    flow_hw2: np.ndarray,
) -> np.ndarray:
    """Map moving/source coordinates into the fixed/reference frame."""
    warped = find_inverse_warp_coords(points_xy, flow_hw2)
    if isinstance(warped, (tuple, list)):
        warped = warped[0]
    warped = np.asarray(warped, dtype=np.float32)
    if warped.ndim != 2 or warped.shape[1] < 2:
        raise ValueError(
            f"Unexpected inverse-warp result shape: {warped.shape}"
        )
    return warped[:, :2]


def raw_sample_id_for_record(record: GeneMap) -> int:
    """Recover the source sample ID for one observed GeneMap."""
    for spec in RAW_SLICE_SPECS:
        if (
            str(spec["label"]) == str(record.label)
            and np.isclose(float(spec["age"]), float(record.time))
        ):
            return int(spec["sample_id"])
    raise KeyError(
        f"Could not map observed record {record.label!r}, time={record.time} "
        "to RAW_SLICE_SPECS."
    )


def raw_native_coords_for_sample(sample_id: int) -> np.ndarray:
    """Load untransformed registration coordinates used by the pairwise flow."""
    coord_path = os.path.join(
        RAW_NET_IN,
        f"{GROUP_ID}_{int(sample_id)}_validation.npz",
    )
    if not os.path.isfile(coord_path):
        raise FileNotFoundError(coord_path)
    coords = np.asarray(np.load(coord_path)["coord"], dtype=np.float32)
    return validate_coords(coords)


def _canonical_domain_token(value) -> str:
    """Normalize numeric-like domain labels, e.g. 2 and 2.0 -> "2"."""
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and float(number).is_integer():
            return str(int(number))
    except Exception:
        pass
    return text



def trajectory_domain_mask_for_record(
    source: GeneMap,
    n_points: int,
) -> np.ndarray:
    """
    Select source locations used to draw Figure-2 transition trajectories.

    TRAJECTORY_DOMAINS=None keeps every finite source location. Otherwise,
    only source spots whose TRAJECTORY_DOMAIN_KEY belongs to the configured
    domain list are eligible.
    """
    if TRAJECTORY_DOMAINS is None:
        return np.ones(int(n_points), dtype=bool)

    ad = sc.read_h5ad(source.path)
    if TRAJECTORY_DOMAIN_KEY not in ad.obs.columns:
        raise KeyError(
            f"{TRAJECTORY_DOMAIN_KEY!r} not found in {source.path}. "
            f"Available obs columns: {list(ad.obs.columns)}"
        )

    labels_raw = ad.obs[TRAJECTORY_DOMAIN_KEY].to_numpy()
    labels = np.asarray(
        [_canonical_domain_token(x) for x in labels_raw],
        dtype=str,
    )
    selected_domains = np.asarray(
        [_canonical_domain_token(x) for x in TRAJECTORY_DOMAINS],
        dtype=str,
    )

    n = min(int(n_points), len(labels))

    mask = np.zeros(int(n_points), dtype=bool)
    mask[:n] = np.isin(
        labels[:n],
        selected_domains,
    )

    if not np.any(mask):
        available = sorted(set(labels[:n].astype(str)))
        raise ValueError(
            "No Figure-2 trajectory seeds remain for "
            f"TRAJECTORY_DOMAINS={TRAJECTORY_DOMAINS}. "
            f"Available {TRAJECTORY_DOMAIN_KEY} values include: {available[:30]}"
        )

    return mask


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
        if "valid" in ad.obs.columns:
            ad = ad[ad.obs["valid"]].copy()
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


def normalize_pattern_strings(values: Sequence) -> np.ndarray:
    """Normalize saved trend labels to the publication-facing terminology."""
    lookup = {
        "increasing": "Increase",
        "increase": "Increase",
        "decreasing": "Decrease",
        "decrease": "Decrease",
        "peak": "Mid-stage peak",
        "mid-stage peak": "Mid-stage peak",
        "trough": "Mid-stage trough",
        "mid-stage trough": "Mid-stage trough",
        "stable": "Stable",
    }
    return np.asarray(
        [
            lookup.get(str(v).strip().lower(), str(v).strip())
            for v in np.asarray(values, dtype=object).ravel()
        ],
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
    """Draw one consistently colored slice/derived-map frame."""
    ymin, ymax, zmin, zmax = extent
    x = np.full(5, float(flow_x), dtype=float)
    y = np.asarray([ymin, ymax, ymax, ymin, ymin], dtype=float)
    z = np.asarray([zmin, zmin, zmax, zmax, zmin], dtype=float)

    linestyle = "-" if kind == "observed" else MATCHED_FRAME_DASH_STYLE

    ax.plot(
        x,
        y,
        z,
        color=SLICE_FRAME_COLOR,
        linewidth=SLICE_FRAME_LINEWIDTH,
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
    False entries suppress scaffold markers, hangers, and labels while leaving
    the corresponding 3D map available for projection illustration. All active
    scaffold markers use the same circular dot, independent of layer type.
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
                marker="o",
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
            # fontweight="bold",
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
# FIGURE-2 TRANSITION FIELD
# ============================================================
def choose_transition_subset_indices(
    n_total: int,
    n_keep: int,
    seed: int,
) -> np.ndarray:
    if n_total <= n_keep:
        return np.arange(n_total, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_keep, replace=False))


def smooth_pairwise_matched_curve(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    start_flow: float,
    end_flow: float,
    n_steps: int = TRANSITION_CURVE_N_STEPS,
) -> np.ndarray:
    """
    Population-pipeline-v7 style trajectory.

    Spatial coordinates use cubic smoothstep interpolation, while WORLD-X
    progresses linearly from the observed state to its matched state.
    """
    t = np.linspace(0.0, 1.0, int(n_steps), dtype=float)
    s = t * t * (3.0 - 2.0 * t)

    start_xy = np.asarray(start_xy, dtype=float)
    end_xy = np.asarray(end_xy, dtype=float)

    xy = (
        (1.0 - s[:, None]) * start_xy[None, :]
        + s[:, None] * end_xy[None, :]
    )
    flow = (
        (1.0 - t) * float(start_flow)
        + t * float(end_flow)
    )

    return np.column_stack(
        [flow, xy[:, 0], xy[:, 1]]
    ).astype(np.float32)


def build_observed_to_matched_transition_curves(
    source: GeneMap,
    matched: GeneMap,
    flow_source: float,
    flow_target: float,
    max_curves: int,
    seed: int,
) -> List[np.ndarray]:
    """
    Build Figure-2 trajectories from one observed stage to its AXIS-matched state.

    The source locations are first filtered by TRAJECTORY_DOMAINS. Their
    pairwise AXIS deformation maps them into the Stage-1 reference coordinate
    frame. Each transformed endpoint is then snapped to the nearest valid point
    actually displayed in the corresponding matched state, so every trajectory
    lands on a plotted matched location.
    """
    sample_id = raw_sample_id_for_record(source)
    reference_id = int(RAW_SLICE_SPECS[0]["sample_id"])

    if sample_id == reference_id:
        return []

    native_xy = raw_native_coords_for_sample(sample_id)
    display_xy = np.asarray(source.coords, dtype=float)[:, :2]

    n = min(
        len(native_xy),
        len(display_xy),
        len(source.values),
    )
    native_xy = native_xy[:n]
    display_xy = display_xy[:n]

    valid = (
        np.all(np.isfinite(native_xy), axis=1)
        & np.all(np.isfinite(display_xy), axis=1)
    )
    valid &= trajectory_domain_mask_for_record(source, n)

    source_indices = np.flatnonzero(valid)
    if len(source_indices) == 0:
        return []

    flow_path = parse_pairwise_flow_path(
        RAW_ROOT,
        GROUP_ID,
        fixed_id=reference_id,
        moving_id=sample_id,
        pairwise_subdir=PAIRWISE_SUBDIR,
    )
    if not os.path.isfile(flow_path):
        raise FileNotFoundError(
            f"Missing pairwise flow for Stage {sample_id + 1} -> "
            f"Stage 1 reference: {flow_path}"
        )

    flow = ensure_flow_hw2(np.load(flow_path))
    warped_native = inverse_warp_points(
        native_xy[source_indices],
        flow,
    )

    height, width = flow.shape[:2]
    warped_valid = (
        np.all(np.isfinite(warped_native), axis=1)
        & (warped_native[:, 0] >= 0)
        & (warped_native[:, 0] < width)
        & (warped_native[:, 1] >= 0)
        & (warped_native[:, 1] < height)
    )

    source_indices = source_indices[warped_valid]
    warped_native = warped_native[warped_valid]

    if len(source_indices) == 0:
        return []

    # Transform the pairwise-flow endpoint into the same display convention
    # used by the spatially matched H5AD files.
    warped_display = transform_coords(
        warped_native,
        swap_xy=COMMON_FRAME_SWAP_XY,
    )

    # Snap to locations that are actually present in the matched state.
    matched_coords = np.asarray(matched.coords, dtype=float)[:, :2]
    matched_valid = np.all(np.isfinite(matched_coords), axis=1)
    if matched.valid_mask is not None:
        matched_valid &= np.asarray(matched.valid_mask, dtype=bool)

    matched_support = matched_coords[matched_valid]
    if len(matched_support) == 0:
        raise ValueError(
            f"No valid matched-state coordinates are available for {matched.label}."
        )

    tree = cKDTree(matched_support)
    _, nearest = tree.query(warped_display, k=1)
    target_xy = matched_support[np.asarray(nearest, dtype=int)]

    # Visualization-only subsampling, after domain filtering and matching.
    keep = choose_transition_subset_indices(
        len(source_indices),
        int(max_curves),
        int(seed),
    )
    source_indices = source_indices[keep]
    target_xy = target_xy[keep]

    curves = [
        smooth_pairwise_matched_curve(
            start_xy=display_xy[idx],
            end_xy=tgt,
            start_flow=flow_source,
            end_flow=flow_target,
            n_steps=TRANSITION_CURVE_N_STEPS,
        )
        for idx, tgt in zip(source_indices, target_xy)
    ]

    print(
        f"[PAIRWISE FIELD] {source.label} -> matched {matched.label} | "
        f"domains={TRAJECTORY_DOMAINS} | curves={len(curves)}"
    )
    return curves


def add_v7_transition_field(
    ax,
    curves: Sequence[np.ndarray],
    color: str,
) -> None:
    """
    Draw Figure-2 transition trajectories AFTER all tissue maps.

    Important
    ---------
    The previous implementation used Line3DCollection. In older mplot3d,
    Path3DCollection (scatter) and Line3DCollection are automatically depth
    sorted as 3D collections, so the matched-state scatter can still be drawn
    over a transition field even when the field is added later.

    Here each trajectory is rendered as a Line3D instead. This preserves the
    same geometry and plotting order as the population pipeline, while making
    the intended local overlay deterministic in the current environment:

        matched-state scatter/frame  ->  transition trajectory

    No ``computed_zorder`` option is required.
    """
    for curve in curves:
        if len(curve) < 2:
            continue

        ax.plot(
            curve[:, 0],
            curve[:, 1],
            curve[:, 2],
            color=color,
            linewidth=TRANSITION_CURVE_LW,
            alpha=TRANSITION_CURVE_ALPHA,
            solid_capstyle="round",
            rasterized=True,
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



def draw_figure2_reference_plane(
    ax,
    flow_x: float,
    frame_extent: Tuple[float, float, float, float] = FIXED_BOX,
) -> None:
    """Solid reference frame with a nearly transparent white plane."""
    ymin, ymax, zmin, zmax = frame_extent
    vertices = [[
        (float(flow_x), ymin, zmin),
        (float(flow_x), ymax, zmin),
        (float(flow_x), ymax, zmax),
        (float(flow_x), ymin, zmax),
    ]]
    plane = Poly3DCollection(
        vertices,
        facecolors=(1.0, 1.0, 1.0, FIG2_REFERENCE_PLANE_FACE_ALPHA),
        edgecolors=SLICE_FRAME_COLOR,
        linewidths=SLICE_FRAME_LINEWIDTH,
    )
    ax.add_collection3d(plane)


def draw_figure2_matched_frame(
    ax,
    flow_x: float,
    frame_extent: Tuple[float, float, float, float] = FIXED_BOX,
) -> None:
    """Dashed frame for a spatially matched state."""
    ymin, ymax, zmin, zmax = frame_extent
    frame = np.asarray(
        [
            [float(flow_x), ymin, zmin],
            [float(flow_x), ymax, zmin],
            [float(flow_x), ymax, zmax],
            [float(flow_x), ymin, zmax],
            [float(flow_x), ymin, zmin],
        ],
        dtype=float,
    )
    ax.plot(
        frame[:, 0],
        frame[:, 1],
        frame[:, 2],
        color=SLICE_FRAME_COLOR,
        linewidth=SLICE_FRAME_LINEWIDTH,
        linestyle=MATCHED_FRAME_DASH_STYLE,
    )


def plot_figure2_gene_plane(
    ax,
    coords: np.ndarray,
    values: np.ndarray,
    flow_x: float,
    vmax: float,
    alpha: float,
    seed: int,
    *,
    frame_mode: str,
    matched_marker_edges: bool = False,
    frame_extent: Tuple[float, float, float, float] = FIXED_BOX,
) -> None:
    """
    Figure-2 gene map.

    frame_mode:
      - "observed": solid reference frame;
      - "matched": dashed matched-state frame;
      - "none": projection-only source map with no frame.
    """
    if frame_mode == "observed":
        draw_figure2_reference_plane(ax, flow_x, frame_extent)
    elif frame_mode == "matched":
        draw_figure2_matched_frame(ax, flow_x, frame_extent)
    elif frame_mode != "none":
        raise ValueError(f"Unknown Figure-2 frame_mode: {frame_mode!r}")

    c, v = _subset_by_mask(coords, values, None, seed)
    colors = gene_facecolors(v, vmax=vmax, alpha_factor=alpha)
    order = np.argsort(v)

    edgecolor = (
        FIG2_MATCHED_MARKER_EDGE_COLOR
        if matched_marker_edges
        else "none"
    )
    edgewidth = (
        FIG2_MATCHED_MARKER_EDGE_WIDTH
        if matched_marker_edges
        else 0.0
    )

    ax.scatter(
        np.full(len(c), float(flow_x))[order],
        c[order, 0],
        c[order, 1],
        c=colors[order],
        s=POINT_SIZE_GENE,
        edgecolors=edgecolor,
        linewidths=edgewidth,
        depthshade=False,
        rasterized=True,
    )


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
    frame_kind: str = "observed",
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
    draw_plane_frame_3d(
        ax,
        flow_x,
        frame_extent,
        kind=frame_kind,
        zorder=frame_zorder,
    )


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


# ============================================================
# FIGURE 1
# ============================================================
def plot_figure1_observed_raw(
    raw_records: Sequence[GeneMap],
    outdir: str,
) -> None:
    set_style()
    fig = plt.figure(figsize=(7.6, 4.8))
    ax = fig.add_subplot(111, projection="3d")

    ordered = sorted(raw_records, key=lambda r: r.time)
    # Stage-oriented display: 5.7m is exactly centered between 2.5m and 17.9m.
    flow_positions = equal_stage_positions(len(ordered), gap=STAGE_GAP/1.2).tolist()
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
# FIGURE 2 — OBSERVED STATES -> SPATIALLY MATCHED STATES
# ============================================================
def plot_figure2_registration_transition_fields(
    raw_records: Sequence[GeneMap],
    common_maps: Sequence[GeneMap],
    outdir: str,
) -> None:
    """
    Spatial matching into the shared anatomical frame.

    Only three states are attached to the progression scaffold:
      Stage 1 reference, matched Stage 2, matched Stage 3.

    Their horizontal layout is controlled by three Figure-2 gap parameters:
      - FIG2_REFERENCE_TO_STAGE2_GAP
      - FIG2_STAGE2_TO_STAGE3_GAP
      - REGISTRATION_PAIR_GAP

    Observed Stage 2/3 remain visible only as projection-source point clouds.
    They have no frame, scaffold marker, hanger, or scaffold label.
    """
    set_style()
    fig = plt.figure(figsize=(14.6, 7.4))
    ax = fig.add_subplot(111, projection="3d")

    raw_ordered = sorted(raw_records, key=lambda r: r.time)
    matched_ordered = sorted(common_maps, key=lambda r: r.time)

    if len(raw_ordered) != 3 or len(matched_ordered) != 3:
        raise ValueError(
            "Figure 2 expects exactly three observed and three spatially matched maps."
        )

    # One measured-derived expression range for reference, matched, and source maps.
    vmax = robust_positive_vmax([r.values for r in raw_ordered], q=99.0)

    # Three independent Figure-2 gap controls:
    #   reference -> matched Stage 2
    #   matched Stage 2 -> matched Stage 3
    #   matched state -> projection-only observed source
    x_reference = 0.0
    x_matched_2 = float(FIG2_REFERENCE_TO_STAGE2_GAP)
    x_matched_3 = x_matched_2 + float(FIG2_STAGE2_TO_STAGE3_GAP)

    x_observed_2 = x_matched_2 + float(REGISTRATION_PAIR_GAP)
    x_observed_3 = x_matched_3 + float(REGISTRATION_PAIR_GAP)

    flow_positions = [
        x_reference,
        x_matched_2,
        x_observed_2,
        x_matched_3,
        x_observed_3,
    ]

    layers = [
        LayerMeta("Stage 1\n(reference)", "observed"),
        LayerMeta("Stage 2\n(matched)", "delta"),
        LayerMeta("Stage 2\n(observed)", "observed"),
        LayerMeta("Stage 3\n(matched)", "delta"),
        LayerMeta("Stage 3\n(observed)", "observed"),
    ]

    # Only these three maps are attached to the scaffold.
    hanging_mask = [True, True, False, True, False]

    # Reference: main alpha, solid frame, no point edge.
    plot_figure2_gene_plane(
        ax,
        raw_ordered[0].coords,
        raw_ordered[0].values,
        x_reference,
        vmax=vmax,
        alpha=FIG2_MAIN_POINT_ALPHA,
        seed=240,
        frame_mode="observed",
    )

    # Matched Stage 2: main alpha, dashed frame, subtle point edge.
    plot_figure2_gene_plane(
        ax,
        matched_ordered[1].coords,
        matched_ordered[1].values,
        x_matched_2,
        vmax=vmax,
        alpha=FIG2_MAIN_POINT_ALPHA,
        seed=241,
        frame_mode="matched",
        matched_marker_edges=True,
    )

    # Projection-only observed Stage 2: separate alpha, no frame.
    plot_figure2_gene_plane(
        ax,
        raw_ordered[1].coords,
        raw_ordered[1].values,
        x_observed_2,
        vmax=vmax,
        alpha=FIG2_PROJECTION_POINT_ALPHA,
        seed=242,
        frame_mode="none",
    )

    # Matched Stage 3: main alpha, dashed frame, subtle point edge.
    plot_figure2_gene_plane(
        ax,
        matched_ordered[2].coords,
        matched_ordered[2].values,
        x_matched_3,
        vmax=vmax,
        alpha=FIG2_MAIN_POINT_ALPHA,
        seed=243,
        frame_mode="matched",
        matched_marker_edges=True,
    )

    # Projection-only observed Stage 3: separate alpha, no frame.
    plot_figure2_gene_plane(
        ax,
        raw_ordered[2].coords,
        raw_ordered[2].values,
        x_observed_3,
        vmax=vmax,
        alpha=FIG2_PROJECTION_POINT_ALPHA,
        seed=244,
        frame_mode="none",
    )

    curves_stage2 = build_observed_to_matched_transition_curves(
        source=raw_ordered[1],
        matched=matched_ordered[1],
        flow_source=x_observed_2,
        flow_target=x_matched_2,
        max_curves=MAX_TRANSITION_CURVES_PER_STAGE,
        seed=TRANSITION_RANDOM_SEED + 1,
    )
    curves_stage3 = build_observed_to_matched_transition_curves(
        source=raw_ordered[2],
        matched=matched_ordered[2],
        flow_source=x_observed_3,
        flow_target=x_matched_3,
        max_curves=MAX_TRANSITION_CURVES_PER_STAGE,
        seed=TRANSITION_RANDOM_SEED + 2,
    )

    # Draw after point clouds so the transition field remains visually legible.
    add_v7_transition_field(
        ax,
        curves_stage2,
        color=TRANSITION_CURVE_COLORS[1],
    )
    add_v7_transition_field(
        ax,
        curves_stage3,
        color=TRANSITION_CURVE_COLORS[2],
    )

    style_3d_axis(ax, FIXED_BOX, flow_positions)

    draw_horizontal_hanging_scaffold(
        fig,
        ax,
        layers,
        flow_positions,
        FIXED_BOX,
        title_label="Spatial matching",
        line_mask=hanging_mask,
    )

    finalize_figure(
        fig,
        os.path.join(
            outdir,
            f"figure2_observed_to_spatially_matched_transition_field_{GENE_NAME}.png",
        ),
    )


# ============================================================
# FIGURE 3 — SPATIALLY MATCHED STAGES + SPATIAL CHANGES
# ============================================================
def plot_figure3_stage_delta_curves(
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
        LayerMeta("Stage 1\n(reference)", "observed"),
        LayerMeta("Δ 1→2", "delta"),
        LayerMeta("Stage 2\n(matched)", "observed"),
        LayerMeta("Δ 2→3", "delta"),
        LayerMeta("Stage 3\n(matched)", "observed"),
    ]

    gene_vmax = robust_positive_vmax([r.values for r in common_maps], q=99.0)
    delta_vmax = robust_symmetric_vmax([spatial_map.delta_12, spatial_map.delta_23], q=99.0)
    valid_mask = spatial_map.valid_domain_mask

    plot_gene_plane_3d(
        ax, common_maps[0].coords, common_maps[0].values, x_t1,
        gene_vmax, GENE_ALPHA_WITH_DELTA, seed=300, frame_kind="matched",
    )
    plot_delta_plane_3d(
        ax, spatial_map.coords, spatial_map.delta_12, x_d12,
        delta_vmax, seed=301, mask=valid_mask,
    )
    plot_gene_plane_3d(
        ax, common_maps[1].coords, common_maps[1].values, x_t2,
        gene_vmax, GENE_ALPHA_WITH_DELTA, seed=302, frame_kind="matched",
    )
    plot_delta_plane_3d(
        ax, spatial_map.coords, spatial_map.delta_23, x_d23,
        delta_vmax, seed=303, mask=valid_mask,
    )
    plot_gene_plane_3d(
        ax, common_maps[2].coords, common_maps[2].values, x_t3,
        gene_vmax, GENE_ALPHA_WITH_DELTA, seed=304, frame_kind="matched",
    )

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
        Line2D(
            [0], [0],
            color=SLICE_FRAME_COLOR,
            linewidth=1.5,
            linestyle=MATCHED_FRAME_DASH_STYLE,
            label="Spatially matched stage",
        ),
        Line2D(
            [0], [0],
            color=SLICE_FRAME_COLOR,
            linewidth=1.5,
            linestyle=MATCHED_FRAME_DASH_STYLE,
            label="Spatial Δ map",
        ),
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
        os.path.join(outdir, f"figure3_stage_delta_curved_arrows_{GENE_NAME}.png"),
    )


# ============================================================
# FIGURE 4 — CONCEPTUAL TREND ILLUSTRATION
# ============================================================
def plot_figure4_trend_cartoon(outdir: str) -> None:
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
        "Increase": np.array([0.0, 0.42, 0.92]),
        "Decrease": np.array([0.0, -0.40, -0.88]),
        "Mid-stage peak": np.array([0.0, 0.86, 0.26]),
        "Mid-stage trough": np.array([0.0, -0.84, -0.24]),
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
        color="0.35",
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.axhline(
        -stable_threshold,
        color="0.35",
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
    for patt in ["Increase", "Decrease", "Mid-stage peak", "Mid-stage trough"]:

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
        "Age stage",
        fontsize=20,
        labelpad=9,
    )

    ax.set_ylabel(
        "Relative expression change",
        fontsize=20,
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
            "figure4_progression_stage_trend_cartoon.png",
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

    raw_maps = load_original_observed_maps(gene=gene)
    common_maps = load_common_frame_gene_maps(gene=gene)
    spatial_map = load_saved_spatial_temporal_map(
        gene=gene,
        explicit_path=spatial_map_path,
    )

    globals()["GENE_NAME"] = gene

    # plot_figure1_observed_raw(raw_maps, OUTDIR)
    plot_figure2_registration_transition_fields(raw_maps, common_maps, OUTDIR)
    # plot_figure3_stage_delta_curves(common_maps, spatial_map, OUTDIR)
    if make_cartoon:
        plot_figure4_trend_cartoon(OUTDIR)

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
        "--figure2-reference-to-stage2-gap",
        type=float,
        default=FIG2_REFERENCE_TO_STAGE2_GAP,
        help="WORLD-X spacing from the Figure-2 Stage 1 reference to matched Stage 2.",
    )
    parser.add_argument(
        "--figure2-stage2-to-stage3-gap",
        type=float,
        default=FIG2_STAGE2_TO_STAGE3_GAP,
        help="WORLD-X spacing from the Figure-2 matched Stage 2 to matched Stage 3.",
    )
    parser.add_argument(
        "--registration-pair-gap",
        type=float,
        default=REGISTRATION_PAIR_GAP,
        help="WORLD-X spacing between each matched state and its observed source state in Figure 2.",
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
    globals()["TIME_SCALE_PER_MONTH"] = float(args.time_scale)
    globals()["SPATIAL_DISPLAY_SCALE"] = float(args.spatial_display_scale)
    globals()["FIG2_REFERENCE_TO_STAGE2_GAP"] = float(args.figure2_reference_to_stage2_gap)
    globals()["FIG2_STAGE2_TO_STAGE3_GAP"] = float(args.figure2_stage2_to_stage3_gap)
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
    print(f"[INFO] FIG2_REFERENCE_TO_STAGE2_GAP = {FIG2_REFERENCE_TO_STAGE2_GAP}")
    print(f"[INFO] FIG2_STAGE2_TO_STAGE3_GAP = {FIG2_STAGE2_TO_STAGE3_GAP}")
    print(f"[INFO] REGISTRATION_PAIR_GAP = {REGISTRATION_PAIR_GAP}")
    print(f"[INFO] MAX_TRANSITION_CURVES_PER_STAGE = {MAX_TRANSITION_CURVES_PER_STAGE}")
    print(f"[INFO] TRAJECTORY_DOMAIN_KEY = {TRAJECTORY_DOMAIN_KEY}")
    print(f"[INFO] TRAJECTORY_DOMAINS = {TRAJECTORY_DOMAINS}")
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



