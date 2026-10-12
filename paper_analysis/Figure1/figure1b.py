#!/usr/bin/env python3
"""
Simulate Visium-like coarse spot measurements from three mouse Xenium slices,
save the simulated data, and generate mirrored pipeline-style figures.

Main outputs
------------
1. Simulated Visium-style h5ad files for slices 0_0, 0_1, 0_2
   with channels:
   - cell_density
   - Arc
   - Gfap
   - axis_joint_domain (majority vote)

2. Figures 1-3:
   For each slice, TWO separate transparent-background figures:
     (a) mirrored whole-brain channel rendering
     (b) mirrored gray-location layout

3. Figure 4:
   Real Arc disease-vs-normal delta at 5.7m, computed directly from the same
   disease/normal H5AD pair used by the Arc mirrored case study, re-aggregated
   onto the same simulated Visium lattice, and rendered as
   (a) mirrored delta values and (b) mirrored gray locations.

The simulation is intended for schematic/pipeline visualization rather than
quantitative analysis, so the aggregation is deliberately simple and robust.
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from anndata import AnnData
import scanpy as sc

from spatial_delta_utils import compute_spatial_delta_from_files

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Default inputs / outputs
# ============================================================
INPUT_H5ADS = [
    "/media/huifang/data/registration/mouse_brain/h5ads/0_0.h5ad",
    "/media/huifang/data/registration/mouse_brain/virtual_tissue/AD_time_grid256_spatial1.0/query/virtual_AD_time_11.800.h5ad",
    "/media/huifang/data/registration/mouse_brain/h5ads/0_2.h5ad",
]
OUTDIR = "/media/huifang/data/registration/paper_analysis/Figure1/1b"

# Figure-4 real Arc delta source.
#
# This recycles the exact disease-vs-normal 5.7m input pair and preprocessing
# used by the Arc mirrored case-study script. The delta is computed directly
# from the two H5AD files with compute_spatial_delta_from_files(), then passed
# through the same Visium-like simulation used for the other panels.
DELTA_GENE = "Arc"
DELTA_AGE = 5.7
DELTA_AGE_LABEL = "5.7m"

DELTA_DISEASE_PATH = (
    "/media/huifang/data/registration/mouse_brain/"
    "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
    "virtual_Disease_time_5.700.h5ad"
)
DELTA_NORMAL_PATH = (
    "/media/huifang/data/registration/mouse_brain/"
    "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
    "virtual_Disease_time_105.700.h5ad"
)

# Match the disease-delta case-study preprocessing exactly.
DELTA_LIBRARY_SIZE_NORMALIZE = True
DELTA_NORMALIZATION_TARGET_SUM = 253
DELTA_LOG1P_BEFORE_ANALYSIS = True
DELTA_USE_COMMON_VALID = True
DELTA_COORDINATE_DECIMALS = 5
DELTA_SPATIAL_KEY = "spatial"
DELTA_DOMAIN_KEY = "axis_joint_domain"
DELTA_VALID_KEY = "valid"

# After the real delta is computed, apply the same orientation/deformation
# pipeline used for the simulated Visium panels.
DELTA_APPLY_DEFORMATION = True
DELTA_DEFORMATION_INDEX = 1

# ============================================================
# Default simulation and figure controls
# ============================================================
VISIUM_SCALE = 50.0             # center-to-center spacing of simulated Visium spots
SPOT_RADIUS_FACTOR = 0.58       # simulated spot radius = factor * VISIUM_SCALE
MIN_CELLS_PER_SPOT = 3

MIRROR_GAP = 50.0              # gap between mirrored halves
FIGURE1_CHANNEL = "Gfap"  # cell_density / Arc / Gfap / axis_joint_domain

# Stylized Figure-1 rendering controls.
# These are display-only enhancements for the schematic pipeline figure.
from matplotlib.colors import LinearSegmentedColormap

MANIFOLD_GENE_CMAP_STRONG = LinearSegmentedColormap.from_list(
    "manifold_gene_strong",
    [
        (0.00, "#b79bd4"),  # soft purple
        (0.20, "#c7d6f3"),  # pale blue
        (0.45, "#86bce3"),  # blue
        (0.72, "#8fd39f"),  # green
        (1.00, "#1f6b4f"),  # deep green
    ],
    N=256,
)

FIGURE1_CMAP = MANIFOLD_GENE_CMAP_STRONG
ENHANCE_STRUCTURE = True
CELL_DENSITY_CORE_BLEND = 0.88
GENE_CORE_BLEND = 0.42

# Grouped-anatomy center-to-boundary controls. These are computed after domain
# merging, so domains 1/4/6/9/10 form one continuous cortex.
GROUP_CORE_BOUNDARY_K = 1
GROUP_CORE_GAMMA = 0.72

# Location filtering controls for cleaner simulated Visium layouts.
REMOVE_HIDDEN_DOMAIN_SPOTS = True
VISIUM_NEIGHBOR_RADIUS_FACTOR = 1.18
MIN_VISIUM_NEIGHBORS = 2
ITERATIVE_NEIGHBOR_PRUNE = True

# Display/readout noise controls for the schematic gene/cell-density maps only.
# The real delta map is never perturbed by these settings.
DISPLAY_NOISE_SD = 0.035
DISPLAY_NOISE_SMOOTH_RADIUS_FACTOR = 1.9

# Global spatial-orientation controls.
# Operation order: swap x/y -> invert x -> invert y.
GLOBAL_SWAPXY = True
GLOBAL_INVERTX = False
GLOBAL_INVERTY = False

# Each paired view is now saved as two separate single-panel figures.
PANEL_FIG_W = 4.3
PANEL_FIG_H = 4.2
DELTA_FIG_W = 4.8
DELTA_FIG_H = 4.6
DPI = 300
GRAY_DOT_COLOR = "0.7"
GRAY_DOT_ALPHA = 0.9

# Slice-specific spot-size defaults for the panels (Figures 1-3)
CHANNEL_SPOT_SIZES = [30.0, 25.0, 30.0]
GRAY_SPOT_SIZES = [30.0, 25.0, 30.0]

# Default delta-map spot size (Figure 4)
DELTA_SPOT_SIZE = 50.0

DOMAIN_KEY = "axis_joint_domain"

# Palette for domain plotting (kept soft and publication-friendly)
DOMAIN_PALETTE = {
    0: "#d77d73",
    1: "#8bb7d9",
    2: "#a6c79e",
    3: "#e3a374",
    4: "#b0cde7",
    5: "#b9afd9",
    6: "#7ea3d6",
    7: "#c1c1c1",
    8: "#f0a39a",
    9: "#5f8fc3",
    10: "#9fc4b8",
    11: "#bf7a68",
}

# Grouped domain rendering for clearer Figure-1 pipeline-style visuals.
# 2 and 12 are intentionally suppressed in the grouped rendering.
GROUPED_DOMAIN_LABELS = {
    "d0": "domain 0",
    "cortex": "cortical layers",
    "d3": "domain 3",
    "group57": "domains 5/7",
    "hippocampus": "hippocampus",
    "hidden": "hidden",
}
GROUPED_DOMAIN_PALETTE = {
    "d0": "#d77d73",
    "cortex": "#89b3d8",
    "d3": "#e3a374",
    "group57": "#a8c99a",
    "hippocampus": "#c58a7d",
    "hidden": (1.0, 1.0, 1.0, 0.0),
}
GROUPED_DOMAIN_MAP = {
    0: "d0",
    1: "cortex",
    3: "d3",
    4: "cortex",
    5: "group57",
    6: "cortex",
    7: "group57",
    8: "hippocampus",
    9: "cortex",
    10: "cortex",
    11: "hippocampus",
    2: "hidden",
    12: "hidden",
}

# Real-delta plotting controls.
DELTA_CMAP = "coolwarm"
DELTA_VMAX_PERCENTILE = 99.0
DELTA_EDGE_COLOR = "0.55"
DELTA_EDGE_WIDTH = 0.30
DELTA_GRAY_SPOT_SIZE = 30.0

# Smooth slice-specific spatial deformation (applied before Visium aggregation).
SPATIAL_DEFORMATION_ENABLED = True
DEFORMATION_STRENGTH_X = [0.045, 0.065, 0.055]
DEFORMATION_STRENGTH_Y = [0.035, 0.050, 0.040]
DEFORMATION_PHASES = [0.0, 0.85, 1.55]

# Middle slice is a virtual tissue state. We down-scale its gene strength for plotting.
VIRTUAL_SLICE_INDEX = 1
VIRTUAL_GENE_SCALE = 0.82
VIRTUAL_STATE_EDGECOLOR = "0.55"
VIRTUAL_STATE_EDGEWIDTH = 0.35

@dataclass
class SimulatedSlice:
    name: str
    adata: AnnData
    spatial: np.ndarray
    cell_density: np.ndarray
    arc: np.ndarray
    gfap: np.ndarray
    domain: np.ndarray
    grouped_domain: np.ndarray
    structural_group: np.ndarray
    n_cells_per_spot: np.ndarray
    domain_core: np.ndarray
    is_virtual: bool = False
    slice_index: int = -1


@dataclass
class SimulatedDeltaMap:
    name: str
    gene_name: str
    age: float
    age_label: str
    disease_path: str
    normal_path: str
    adata: AnnData
    spatial: np.ndarray
    delta: np.ndarray
    domain: np.ndarray
    grouped_domain: np.ndarray
    n_values_per_spot: np.ndarray


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def to_dense_vector(x):
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    return np.ravel(x)


def find_gene_name(var_names, target: str) -> str:
    if target in var_names:
        return target
    lower_map = {str(v).lower(): str(v) for v in var_names}
    if target.lower() in lower_map:
        return lower_map[target.lower()]
    stripped = {str(v).strip().lower(): str(v) for v in var_names}
    key = target.strip().lower()
    if key in stripped:
        return stripped[key]
    raise KeyError(f"Gene '{target}' not found in var_names.")


def get_gene_vector(adata: AnnData, gene: str) -> np.ndarray:
    gene_name = find_gene_name(adata.var_names, gene)
    idx = int(np.where(adata.var_names == gene_name)[0][0])
    vals = adata.X[:, idx]
    vals = to_dense_vector(vals).astype(float)
    return vals


def get_obs_array(adata: AnnData, key: str) -> np.ndarray:
    if key not in adata.obs.columns:
        raise KeyError(f"obs key '{key}' not found.")
    return np.asarray(adata.obs[key])


def transform_spatial_coords(
    coords: np.ndarray,
    swapxy: bool = GLOBAL_SWAPXY,
    invertx: bool = GLOBAL_INVERTX,
    inverty: bool = GLOBAL_INVERTY,
) -> np.ndarray:
    """
    Apply a global orientation transform to spatial coordinates.
    The operation order is: swap x/y first, then invert x, then invert y.
    """
    out = np.asarray(coords, dtype=float).copy()
    if out.shape[1] < 2:
        raise ValueError("spatial coordinates must be at least 2D.")
    out = out[:, :2]
    out[:, 0] = 1.3 * out[:, 0]
    if swapxy:
        out = out[:, [1, 0]]
    if invertx:
        out[:, 0] = -out[:, 0]
    if inverty:
        out[:, 1] = -out[:, 1]
    return out


def get_spatial_coords(
    adata: AnnData,
    swapxy: bool = GLOBAL_SWAPXY,
    invertx: bool = GLOBAL_INVERTX,
    inverty: bool = GLOBAL_INVERTY,
) -> np.ndarray:
    if "spatial" not in adata.obsm:
        raise KeyError("obsm['spatial'] not found.")
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    return transform_spatial_coords(xy, swapxy=swapxy, invertx=invertx, inverty=inverty)


def get_slice_strength(values, idx, default):
    if isinstance(values, (list, tuple, np.ndarray)):
        if len(values) == 0:
            return default
        return float(values[min(max(int(idx), 0), len(values) - 1)])
    return float(values)


def apply_spatial_deformation(
    coords: np.ndarray,
    slice_index: int,
    enabled: bool = SPATIAL_DEFORMATION_ENABLED,
    strengths_x = DEFORMATION_STRENGTH_X,
    strengths_y = DEFORMATION_STRENGTH_Y,
    phases = DEFORMATION_PHASES,
) -> np.ndarray:
    """Apply a smooth slice-specific spatial warp for schematic visualization."""
    coords = np.asarray(coords, dtype=float)
    if (not enabled) or len(coords) == 0:
        return coords.copy()

    out = coords.copy()
    xmin, ymin = out.min(axis=0)
    xmax, ymax = out.max(axis=0)
    width = max(xmax - xmin, 1e-8)
    height = max(ymax - ymin, 1e-8)

    x = (out[:, 0] - xmin) / width
    y = (out[:, 1] - ymin) / height
    x_c = 2.0 * (x - 0.5)
    y_c = 2.0 * (y - 0.5)

    sx = get_slice_strength(strengths_x, slice_index, 0.05)
    sy = get_slice_strength(strengths_y, slice_index, 0.04)
    phase = get_slice_strength(phases, slice_index, 0.0)

    dx = width * sx * (
        0.65 * np.sin(np.pi * (y + 0.18 * x) + phase)
        + 0.20 * x_c * y_c
        - 0.12 * np.cos(2.0 * np.pi * x + 0.5 * phase)
    )
    dy = height * sy * (
        0.55 * np.sin(np.pi * (x - 0.20 * y) + 0.6 * phase)
        + 0.18 * (1.0 - x_c**2) * y_c
        + 0.10 * np.cos(2.0 * np.pi * y + phase)
    )

    out[:, 0] += dx
    out[:, 1] += dy
    return out


def robust_unit_interval(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.nanpercentile(values, low))
    hi = float(np.nanpercentile(values, high))
    if not np.isfinite(lo):
        lo = float(np.nanmin(values))
    if not np.isfinite(hi):
        hi = float(np.nanmax(values))
    if np.isclose(lo, hi):
        return np.zeros_like(values, dtype=float)
    out = (values - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def group_domain_labels(domain_vals: np.ndarray) -> np.ndarray:
    out = []
    for d in np.asarray(domain_vals):
        try:
            key = int(d)
        except Exception:
            key = None
        out.append(GROUPED_DOMAIN_MAP.get(key, "hidden"))
    return np.asarray(out, dtype=object)


def fill_hidden_groups_for_structure(
    coords: np.ndarray,
    grouped_domain: np.ndarray,
) -> np.ndarray:
    """
    Fill hidden domains 2/12 only for the continuous structural field.

    Their categorical labels remain hidden. For the continuous cell-density/gene
    display, assigning them to the nearest visible grouped anatomy prevents holes
    from artificially fragmenting the cortex or hippocampus.
    """
    coords = np.asarray(coords, dtype=float)
    groups = np.asarray(grouped_domain, dtype=object).copy()
    hidden = groups == "hidden"
    visible = ~hidden

    if not np.any(hidden) or not np.any(visible):
        return groups

    tree = cKDTree(coords[visible])
    _, nn = tree.query(coords[hidden], k=1)
    groups[hidden] = groups[visible][np.asarray(nn, dtype=int)]
    return groups


def compute_group_core_score(
    coords: np.ndarray,
    structural_group: np.ndarray,
    gamma: float = GROUP_CORE_GAMMA,
) -> np.ndarray:
    """
    Smooth center-to-boundary score for the COMBINED anatomical groups.

    All cortical domains are first merged, and each spot is scored by distance
    to the nearest DIFFERENT grouped anatomy. The cortex therefore behaves as a
    single continuous shell.
    """
    coords = np.asarray(coords, dtype=float)
    groups = np.asarray(structural_group, dtype=object)
    score = np.zeros(len(coords), dtype=float)

    for g in pd.unique(groups):
        mask = groups == g
        if not np.any(mask):
            continue

        outside = ~mask
        if not np.any(outside):
            score[mask] = 1.0
            continue

        outside_tree = cKDTree(coords[outside])
        dist, _ = outside_tree.query(coords[mask], k=GROUP_CORE_BOUNDARY_K)
        dist = np.asarray(dist, dtype=float)
        if dist.ndim > 1:
            dist = np.mean(dist, axis=1)

        lo = float(np.nanpercentile(dist, 3.0))
        hi = float(np.nanpercentile(dist, 97.0))
        if not np.isfinite(lo):
            lo = float(np.nanmin(dist))
        if not np.isfinite(hi):
            hi = float(np.nanmax(dist))

        if np.isclose(lo, hi):
            local = np.ones_like(dist)
        else:
            local = np.clip((dist - lo) / (hi - lo), 0.0, 1.0)

        score[mask] = np.power(local, gamma)

    return np.clip(score, 0.0, 1.0)


def prune_visium_spots(
    coords: np.ndarray,
    grouped_domains: np.ndarray,
    visium_scale: float,
    min_neighbors: int = MIN_VISIUM_NEIGHBORS,
    radius_factor: float = VISIUM_NEIGHBOR_RADIUS_FACTOR,
    iterative: bool = ITERATIVE_NEIGHBOR_PRUNE,
) -> np.ndarray:
    """Remove suppressed regions and isolated spots from simulated Visium layout."""
    coords = np.asarray(coords, dtype=float)
    groups = np.asarray(grouped_domains, dtype=object)
    keep = np.ones(len(coords), dtype=bool)
    if REMOVE_HIDDEN_DOMAIN_SPOTS:
        keep &= (groups != "hidden")
    if len(coords) == 0:
        return keep

    radius = max(float(visium_scale) * float(radius_factor), 1e-8)
    changed = True
    while changed:
        changed = False
        idx = np.where(keep)[0]
        if idx.size == 0:
            break
        tree = cKDTree(coords[idx])
        nbrs = tree.query_ball_point(coords[idx], r=radius)
        counts = np.array([max(0, len(n) - 1) for n in nbrs], dtype=int)
        drop = counts <= int(min_neighbors)
        if np.any(drop):
            keep[idx[drop]] = False
            changed = True
            if not iterative:
                break
    return keep


def make_spatially_smoothed_noise(
    coords: np.ndarray,
    scale: float,
    sd: float = DISPLAY_NOISE_SD,
    radius_factor: float = DISPLAY_NOISE_SMOOTH_RADIUS_FACTOR,
    seed: int = 0,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0 or sd <= 0:
        return np.zeros(len(coords), dtype=float)
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, sd, len(coords))
    tree = cKDTree(coords)
    radius = max(float(scale) * float(radius_factor), 1e-8)
    nbrs = tree.query_ball_point(coords, r=radius)
    smooth = raw.copy()
    for i, ids in enumerate(nbrs):
        if len(ids) > 1:
            smooth[i] = 0.35 * raw[i] + 0.65 * np.mean(raw[np.asarray(ids, dtype=int)])
    return smooth


def make_stylized_display_values(sim: "SimulatedSlice", channel: str) -> np.ndarray:
    """Display-only values for clearer pipeline-style Visium renderings."""
    if channel == "cell_density":
        base = robust_unit_interval(np.log1p(sim.cell_density))
        val = np.clip((1.0 - CELL_DENSITY_CORE_BLEND) * base + CELL_DENSITY_CORE_BLEND * sim.domain_core, 0.0, 1.0) if ENHANCE_STRUCTURE else base
    elif channel.lower() == "arc":
        base = robust_unit_interval(np.log1p(np.maximum(sim.arc, 0.0)))
        val = np.clip((1.0 - GENE_CORE_BLEND) * base + GENE_CORE_BLEND * sim.domain_core, 0.0, 1.0) if ENHANCE_STRUCTURE else base
    elif channel.lower() == "gfap":
        base = robust_unit_interval(np.log1p(np.maximum(sim.gfap, 0.0)))
        val = np.clip((1.0 - GENE_CORE_BLEND) * base + GENE_CORE_BLEND * sim.domain_core, 0.0, 1.0) if ENHANCE_STRUCTURE else base
    else:
        raise ValueError(f"Stylized display values are not defined for channel: {channel}")

    noise = make_spatially_smoothed_noise(
        sim.spatial,
        VISIUM_SCALE,
        sd=DISPLAY_NOISE_SD,
        seed=abs(hash((sim.name, channel))) % (2**32 - 1),
    )
    return np.clip(val + noise, 0.0, 1.0)


def normalize_to_panel(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    width = max(xmax - xmin, 1e-6)
    height = max(ymax - ymin, 1e-6)
    scale = 1.0 / max(width, height)

    out = coords.copy()
    out[:, 0] = (out[:, 0] - xmin) * scale
    out[:, 1] = (out[:, 1] - ymin) * scale

    xr = out[:, 0].max() - out[:, 0].min()
    yr = out[:, 1].max() - out[:, 1].min()
    out[:, 0] += (1.0 - xr) / 2.0 - out[:, 0].min()
    out[:, 1] += (1.0 - yr) / 2.0 - out[:, 1].min()
    return out


def normalize_domains(values) -> np.ndarray:
    """Normalize heterogeneous domain labels to integer IDs."""
    out = []
    for value in np.asarray(values):
        try:
            out.append(int(float(value)))
        except Exception:
            out.append(-1)
    return np.asarray(out, dtype=int)


def load_case_study_delta(
    gene: str = DELTA_GENE,
    disease_path: str = DELTA_DISEASE_PATH,
    normal_path: str = DELTA_NORMAL_PATH,
    swapxy: bool = GLOBAL_SWAPXY,
    invertx: bool = GLOBAL_INVERTX,
    inverty: bool = GLOBAL_INVERTY,
    apply_deformation: bool = DELTA_APPLY_DEFORMATION,
    deformation_index: int = DELTA_DEFORMATION_INDEX,
):
    """Compute the real disease-vs-normal delta using the Arc 5.7m case-study source."""
    if not os.path.isfile(disease_path):
        raise FileNotFoundError(f"Missing disease H5AD: {disease_path}")
    if not os.path.isfile(normal_path):
        raise FileNotFoundError(f"Missing normal H5AD: {normal_path}")

    delta_adata = compute_spatial_delta_from_files(
        disease_path=disease_path,
        normal_path=normal_path,
        normalize_total=DELTA_LIBRARY_SIZE_NORMALIZE,
        target_sum=DELTA_NORMALIZATION_TARGET_SUM,
        use_log1p=DELTA_LOG1P_BEFORE_ANALYSIS,
        valid_key=DELTA_VALID_KEY,
        spatial_key=DELTA_SPATIAL_KEY,
        domain_key=DELTA_DOMAIN_KEY,
        genes=[gene],
        use_common_valid=DELTA_USE_COMMON_VALID,
        coordinate_decimals=DELTA_COORDINATE_DECIMALS,
        return_result=False,
    )

    coords = np.asarray(delta_adata.obsm[DELTA_SPATIAL_KEY], dtype=float)[:, :2]
    values = to_dense_vector(delta_adata[:, [gene]].X).astype(float)
    domains = normalize_domains(delta_adata.obs[DELTA_DOMAIN_KEY].to_numpy())

    n = min(len(coords), len(values), len(domains))
    coords = coords[:n]
    values = values[:n]
    domains = domains[:n]

    keep = np.all(np.isfinite(coords), axis=1) & np.isfinite(values)
    coords = coords[keep]
    values = values[keep]
    domains = domains[keep]

    coords = transform_spatial_coords(
        coords,
        swapxy=swapxy,
        invertx=invertx,
        inverty=inverty,
    )
    if apply_deformation:
        coords = apply_spatial_deformation(coords, slice_index=deformation_index)

    print(f"[LOAD] Arc disease-vs-normal delta | age={DELTA_AGE_LABEL} | n={len(values)}")
    print(f"[LOAD] disease: {disease_path}")
    print(f"[LOAD] normal:  {normal_path}")
    return coords, values, domains


def simulate_visium_from_case_study_delta(
    gene: str = DELTA_GENE,
    age: float = DELTA_AGE,
    age_label: str = DELTA_AGE_LABEL,
    disease_path: str = DELTA_DISEASE_PATH,
    normal_path: str = DELTA_NORMAL_PATH,
    visium_scale: float = VISIUM_SCALE,
    spot_radius_factor: float = SPOT_RADIUS_FACTOR,
    min_values_per_spot: int = MIN_CELLS_PER_SPOT,
    swapxy: bool = GLOBAL_SWAPXY,
    invertx: bool = GLOBAL_INVERTX,
    inverty: bool = GLOBAL_INVERTY,
) -> SimulatedDeltaMap:
    """Aggregate the real 5.7m disease-vs-normal delta onto the Visium-like lattice."""
    coords, values, domains = load_case_study_delta(
        gene=gene,
        disease_path=disease_path,
        normal_path=normal_path,
        swapxy=swapxy,
        invertx=invertx,
        inverty=inverty,
    )

    centers = generate_hex_centers(coords, spacing=visium_scale)
    radius = float(visium_scale) * float(spot_radius_factor)
    tree = cKDTree(coords)

    kept_centers = []
    mean_delta = []
    maj_domains = []
    n_values = []

    for center in centers:
        idx = tree.query_ball_point(center, r=radius)
        if len(idx) < int(min_values_per_spot):
            continue
        idx = np.asarray(idx, dtype=int)
        vals = values[idx]
        doms = domains[idx]
        finite = np.isfinite(vals)
        if np.sum(finite) < int(min_values_per_spot):
            continue
        vals = vals[finite]
        doms = doms[finite]
        mode = pd.Series(doms).mode(dropna=True)
        maj_dom = mode.iloc[0] if len(mode) > 0 else doms[0]
        kept_centers.append(center)
        mean_delta.append(float(np.mean(vals)))
        maj_domains.append(maj_dom)
        n_values.append(int(len(vals)))

    if not kept_centers:
        raise RuntimeError("No simulated Visium delta spots were generated from the Arc 5.7m delta.")

    kept_centers = np.asarray(kept_centers, dtype=float)
    mean_delta = np.asarray(mean_delta, dtype=float)
    maj_domains = np.asarray(maj_domains)
    grouped_domains = group_domain_labels(maj_domains)
    n_values = np.asarray(n_values, dtype=int)

    keep = prune_visium_spots(
        kept_centers,
        grouped_domains,
        visium_scale=visium_scale,
        min_neighbors=MIN_VISIUM_NEIGHBORS,
        radius_factor=VISIUM_NEIGHBOR_RADIUS_FACTOR,
        iterative=ITERATIVE_NEIGHBOR_PRUNE,
    )
    kept_centers = kept_centers[keep]
    mean_delta = mean_delta[keep]
    maj_domains = maj_domains[keep]
    grouped_domains = grouped_domains[keep]
    n_values = n_values[keep]

    obs = pd.DataFrame(index=[f"{gene}_{age_label}_delta_spot_{i:05d}" for i in range(len(kept_centers))])
    obs["axis_joint_domain"] = maj_domains
    obs["grouped_domain"] = grouped_domains
    obs["n_values_aggregated"] = n_values
    obs["source_gene"] = gene
    obs["age"] = float(age)
    obs["age_label"] = str(age_label)

    ad = AnnData(
        X=mean_delta[:, None].astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"{gene}_disease_minus_normal"]),
    )
    ad.obsm["spatial"] = kept_centers.astype(np.float32)
    ad.uns["simulation_info"] = {
        "source": "compute_spatial_delta_from_files",
        "disease_path": str(disease_path),
        "normal_path": str(normal_path),
        "source_gene": str(gene),
        "age": float(age),
        "age_label": str(age_label),
        "normalize_total": bool(DELTA_LIBRARY_SIZE_NORMALIZE),
        "target_sum": float(DELTA_NORMALIZATION_TARGET_SUM),
        "use_log1p": bool(DELTA_LOG1P_BEFORE_ANALYSIS),
        "use_common_valid": bool(DELTA_USE_COMMON_VALID),
        "coordinate_decimals": int(DELTA_COORDINATE_DECIMALS),
        "visium_scale": float(visium_scale),
        "spot_radius_factor": float(spot_radius_factor),
        "spot_radius": float(radius),
        "min_values_per_spot": int(min_values_per_spot),
        "swapxy": bool(swapxy),
        "invertx": bool(invertx),
        "inverty": bool(inverty),
        "apply_deformation": bool(DELTA_APPLY_DEFORMATION),
        "deformation_index": int(DELTA_DEFORMATION_INDEX),
    }

    return SimulatedDeltaMap(
        name=f"{gene}_{age_label}_disease_minus_normal",
        gene_name=str(gene),
        age=float(age),
        age_label=str(age_label),
        disease_path=str(disease_path),
        normal_path=str(normal_path),
        adata=ad,
        spatial=kept_centers,
        delta=mean_delta,
        domain=maj_domains,
        grouped_domain=grouped_domains,
        n_values_per_spot=n_values,
    )


def save_simulated_delta(sim: SimulatedDeltaMap, outdir: str) -> None:
    delta_dir = os.path.join(outdir, "delta_arc_5p7m")
    ensure_dir(delta_dir)

    h5ad_path = os.path.join(delta_dir, f"{sim.name}_simulated_visium.h5ad")
    csv_path = os.path.join(delta_dir, f"{sim.name}_simulated_visium.csv")
    info_path = os.path.join(delta_dir, f"{sim.name}_simulation_info.json")

    sim.adata.write_h5ad(h5ad_path)
    pd.DataFrame({
        "x": sim.spatial[:, 0],
        "y": sim.spatial[:, 1],
        "delta": sim.delta,
        "axis_joint_domain": sim.domain,
        "grouped_domain": sim.grouped_domain,
        "n_values_aggregated": sim.n_values_per_spot,
    }).to_csv(csv_path, index=False)
    save_json(sim.adata.uns.get("simulation_info", {}), info_path)

    print(f"[SAVE] {h5ad_path}")
    print(f"[SAVE] {csv_path}")


def generate_hex_centers(coords: np.ndarray, spacing: float) -> np.ndarray:
    x = coords[:, 0]
    y = coords[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    dx = spacing
    dy = spacing * np.sqrt(3.0) / 2.0

    xs_all = []
    ys_all = []

    n_rows = int(np.ceil((ymax - ymin) / dy)) + 3
    for r in range(n_rows):
        y0 = ymin + r * dy
        x_offset = 0.5 * dx if (r % 2 == 1) else 0.0
        n_cols = int(np.ceil((xmax - xmin) / dx)) + 3
        for c in range(n_cols):
            x0 = xmin + c * dx + x_offset
            xs_all.append(x0)
            ys_all.append(y0)

    return np.column_stack([xs_all, ys_all])


def simulate_visium_from_xenium(
    adata: AnnData,
    slice_name: str,
    visium_scale: float = VISIUM_SCALE,
    spot_radius_factor: float = SPOT_RADIUS_FACTOR,
    min_cells_per_spot: int = MIN_CELLS_PER_SPOT,
    domain_key: str = DOMAIN_KEY,
    swapxy: bool = GLOBAL_SWAPXY,
    invertx: bool = GLOBAL_INVERTX,
    inverty: bool = GLOBAL_INVERTY,
    slice_index: int = 0,
    is_virtual: bool = False,
    virtual_gene_scale: float = VIRTUAL_GENE_SCALE,
) -> SimulatedSlice:
    coords = get_spatial_coords(adata, swapxy=swapxy, invertx=invertx, inverty=inverty)
    coords = apply_spatial_deformation(coords, slice_index=slice_index)
    domain = get_obs_array(adata, domain_key)
    arc = get_gene_vector(adata, "Arc")
    gfap = get_gene_vector(adata, "Gfap")

    centers = generate_hex_centers(coords, spacing=visium_scale)
    radius = visium_scale * spot_radius_factor
    tree = cKDTree(coords)

    kept_centers = []
    cell_densities = []
    mean_arc = []
    mean_gfap = []
    maj_domains = []
    n_cells = []

    for center in centers:
        idx = tree.query_ball_point(center, r=radius)
        if len(idx) < min_cells_per_spot:
            continue

        idx = np.asarray(idx, dtype=int)
        vals_arc = arc[idx]
        vals_gfap = gfap[idx]
        doms = domain[idx]

        dom_series = pd.Series(doms)
        dom_mode = dom_series.mode(dropna=True)
        maj_dom = dom_mode.iloc[0] if len(dom_mode) > 0 else doms[0]

        kept_centers.append(center)
        n_cells.append(len(idx))
        cell_densities.append(float(len(idx)))
        mean_arc.append(float(np.mean(vals_arc)))
        mean_gfap.append(float(np.mean(vals_gfap)))
        maj_domains.append(maj_dom)

    if len(kept_centers) == 0:
        raise RuntimeError(
            f"No simulated spots were generated for slice '{slice_name}'. "
            f"Try increasing spot radius or lowering min_cells_per_spot."
        )

    kept_centers = np.asarray(kept_centers, dtype=float)
    cell_densities = np.asarray(cell_densities, dtype=float)
    mean_arc = np.asarray(mean_arc, dtype=float)
    mean_gfap = np.asarray(mean_gfap, dtype=float)
    maj_domains = np.asarray(maj_domains)
    grouped_domains = group_domain_labels(maj_domains)
    n_cells = np.asarray(n_cells, dtype=int)

    if is_virtual:
        mean_arc = mean_arc * float(virtual_gene_scale)
        mean_gfap = mean_gfap * float(virtual_gene_scale)

    keep = prune_visium_spots(
        kept_centers,
        grouped_domains,
        visium_scale=visium_scale,
        min_neighbors=MIN_VISIUM_NEIGHBORS,
        radius_factor=VISIUM_NEIGHBOR_RADIUS_FACTOR,
        iterative=ITERATIVE_NEIGHBOR_PRUNE,
    )
    kept_centers = kept_centers[keep]
    cell_densities = cell_densities[keep]
    mean_arc = mean_arc[keep]
    mean_gfap = mean_gfap[keep]
    maj_domains = maj_domains[keep]
    grouped_domains = grouped_domains[keep]
    n_cells = n_cells[keep]

    structural_groups = fill_hidden_groups_for_structure(
        kept_centers,
        grouped_domains,
    )
    domain_core = compute_group_core_score(
        kept_centers,
        structural_groups,
    )

    sim_X = np.column_stack([mean_arc, mean_gfap]).astype(np.float32)
    sim_var = pd.DataFrame(index=["Arc", "Gfap"])
    sim_obs = pd.DataFrame(index=[f"{slice_name}_spot_{i:05d}" for i in range(len(kept_centers))])
    sim_obs["cell_density"] = cell_densities
    sim_obs["axis_joint_domain"] = maj_domains
    sim_obs["grouped_domain"] = grouped_domains
    sim_obs["structural_group"] = structural_groups
    sim_obs["n_cells_aggregated"] = n_cells
    sim_obs["slice_name"] = slice_name
    sim_obs["domain_core"] = domain_core

    sim_adata = AnnData(X=sim_X, obs=sim_obs, var=sim_var)
    sim_adata.obsm["spatial"] = kept_centers.astype(np.float32)
    sim_adata.uns["simulation_info"] = {
        "visium_scale": float(visium_scale),
        "spot_radius_factor": float(spot_radius_factor),
        "spot_radius": float(radius),
        "min_cells_per_spot": int(min_cells_per_spot),
        "domain_key": str(domain_key),
        "swapxy": bool(swapxy),
        "invertx": bool(invertx),
        "inverty": bool(inverty),
        "slice_index": int(slice_index),
        "is_virtual": bool(is_virtual),
        "virtual_gene_scale": float(virtual_gene_scale),
        "spatial_deformation_enabled": bool(SPATIAL_DEFORMATION_ENABLED),
        "group_core_gamma": float(GROUP_CORE_GAMMA),
        "remove_hidden_domain_spots": bool(REMOVE_HIDDEN_DOMAIN_SPOTS),
        "visium_neighbor_radius_factor": float(VISIUM_NEIGHBOR_RADIUS_FACTOR),
        "min_visium_neighbors": int(MIN_VISIUM_NEIGHBORS),
        "iterative_neighbor_prune": bool(ITERATIVE_NEIGHBOR_PRUNE),
        "display_noise_sd": float(DISPLAY_NOISE_SD),
        "display_noise_smooth_radius_factor": float(DISPLAY_NOISE_SMOOTH_RADIUS_FACTOR),
    }

    return SimulatedSlice(
        name=slice_name,
        adata=sim_adata,
        spatial=kept_centers,
        cell_density=cell_densities,
        arc=mean_arc,
        gfap=mean_gfap,
        domain=maj_domains,
        grouped_domain=grouped_domains,
        structural_group=structural_groups,
        n_cells_per_spot=n_cells,
        domain_core=domain_core,
        is_virtual=bool(is_virtual),
        slice_index=int(slice_index),
    )


def save_simulated_slice(sim: SimulatedSlice, outdir: str) -> None:
    slice_dir = os.path.join(outdir, sim.name)
    ensure_dir(slice_dir)

    h5ad_path = os.path.join(slice_dir, f"{sim.name}_simulated_visium.h5ad")
    sim.adata.write_h5ad(h5ad_path)

    coords = np.asarray(sim.spatial)
    csv = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cell_density": sim.cell_density,
        "Arc": sim.arc,
        "Gfap": sim.gfap,
        "axis_joint_domain": sim.domain,
        "grouped_domain": sim.grouped_domain,
        "structural_group": sim.structural_group,
        "n_cells_aggregated": sim.n_cells_per_spot,
        "domain_core": sim.domain_core,
    })
    csv.to_csv(os.path.join(slice_dir, f"{sim.name}_simulated_visium.csv"), index=False)

    save_json(sim.adata.uns.get("simulation_info", {}), os.path.join(slice_dir, "simulation_info.json"))


def make_mirrored_coords(coords: np.ndarray, gap: float = MIRROR_GAP) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    x = coords[:, 0]
    y = coords[:, 1]
    x_medial = float(x.min()) - gap / 2.0
    x_mirror = 2.0 * x_medial - x
    mirrored = np.column_stack([x_mirror, y])
    whole = np.vstack([mirrored, coords])
    return normalize_to_panel(whole)


def duplicate_values_for_mirror(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.concatenate([values, values], axis=0)


def get_domain_colors(domain_vals: np.ndarray, grouped: bool = False):
    out = []
    if grouped:
        labels = group_domain_labels(domain_vals)
        for lab in labels:
            out.append(GROUPED_DOMAIN_PALETTE.get(str(lab), "#bbbbbb"))
        return out

    for d in domain_vals:
        try:
            key = int(d)
        except Exception:
            key = None
        out.append(DOMAIN_PALETTE.get(key, "#bbbbbb"))
    return out


def scatter_continuous(ax, coords, values, cmap="Reds", size=DELTA_SPOT_SIZE, title=None,
                       vmin=None, vmax=None, edgecolor="none", linewidth=0.0, alpha=1.0):
    coords = np.asarray(coords)
    values = np.asarray(values, dtype=float)
    if vmin is None:
        vmin = np.nanpercentile(values, 1.0)
    if vmax is None:
        vmax = np.nanpercentile(values, 99.0)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-8
    order = np.argsort(values)
    ax.scatter(
        coords[order, 0], coords[order, 1],
        c=values[order],
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        s=size,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
        rasterized=True,
    )
    if title is not None:
        ax.set_title(title, fontsize=12, pad=8)
    ax.set_aspect("equal")
    ax.axis("off")


def scatter_categorical(ax, coords, domain_vals, size=DELTA_SPOT_SIZE, title=None,
                        edgecolor="none", linewidth=0.0, grouped: bool = True):
    coords = np.asarray(coords)
    if grouped:
        labels = group_domain_labels(domain_vals)
        keep = labels != "hidden"
        coords_plot = coords[keep]
        labels_plot = labels[keep]
        colors = [GROUPED_DOMAIN_PALETTE.get(str(l), "#bbbbbb") for l in labels_plot]
    else:
        keep = np.ones(len(coords), dtype=bool)
        coords_plot = coords
        labels_plot = np.asarray(domain_vals)
        colors = get_domain_colors(labels_plot, grouped=False)

    ax.scatter(
        coords_plot[:, 0], coords_plot[:, 1],
        c=colors,
        s=size,
        edgecolors=edgecolor,
        linewidths=linewidth,
        rasterized=True,
    )
    if title is not None:
        ax.set_title(title, fontsize=12, pad=8)
    ax.set_aspect("equal")
    ax.axis("off")


def scatter_gray(ax, coords, size=14.0, title=None):
    coords = np.asarray(coords)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=GRAY_DOT_COLOR,
        s=size,
        alpha=0.95,
        edgecolors="none",
        rasterized=True,
    )
    if title is not None:
        ax.set_title(title, fontsize=12, pad=8)
    ax.set_aspect("equal")
    ax.axis("off")


def get_channel_values(sim: SimulatedSlice, channel: str):
    if channel == "cell_density":
        return sim.cell_density
    if channel.lower() == "arc":
        return sim.arc
    if channel.lower() == "gfap":
        return sim.gfap
    if channel == "axis_joint_domain":
        return sim.domain
    raise ValueError(f"Unsupported figure1 channel: {channel}")


def _make_transparent_fig(width: float, height: float):
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig, ax


def plot_slice_pair_figure(
    sim: SimulatedSlice,
    outdir: str,
    channel: str = FIGURE1_CHANNEL,
    mirror_gap: float = MIRROR_GAP,
    channel_spot_size: float = CHANNEL_SPOT_SIZES[0],
    gray_spot_size: float = GRAY_SPOT_SIZES[0],
    figure_number: int = 1,
):
    """
    Save the channel view and the gray-location view as TWO separate
    transparent-background PNGs.
    """
    coords = make_mirrored_coords(sim.spatial, gap=mirror_gap)

    # ---------------- Figure A: channel view ----------------
    fig, ax = _make_transparent_fig(PANEL_FIG_W, PANEL_FIG_H)
    edgecolor = VIRTUAL_STATE_EDGECOLOR if sim.is_virtual else "none"
    linewidth = VIRTUAL_STATE_EDGEWIDTH if sim.is_virtual else 0.0

    if channel == "axis_joint_domain":
        vals = duplicate_values_for_mirror(sim.domain)
        scatter_categorical(
            ax, coords, vals, size=channel_spot_size,
            title=f"{sim.name} mirrored domain view", grouped=True,
            edgecolor=edgecolor, linewidth=linewidth,
        )
    else:
        vals = duplicate_values_for_mirror(make_stylized_display_values(sim, channel))
        scatter_continuous(
            ax, coords, vals, cmap=FIGURE1_CMAP, size=channel_spot_size,
            title=f"{sim.name} mirrored {channel} view", vmin=0.0, vmax=1.0,
            edgecolor=edgecolor, linewidth=linewidth,
        )

    fig.tight_layout()
    path_a = os.path.join(outdir, f"figure{figure_number}a_{sim.name}_{channel}.png")
    fig.savefig(path_a, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.close(fig)

    # ---------------- Figure B: gray locations ----------------
    fig, ax = _make_transparent_fig(PANEL_FIG_W, PANEL_FIG_H)

    scatter_gray(ax, coords, size=gray_spot_size, title=f"{sim.name} mirrored locations")

    fig.tight_layout()
    path_b = os.path.join(outdir, f"figure{figure_number}b_{sim.name}_locations.png")
    fig.savefig(path_b, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"[SAVE] {path_a}")
    print(f"[SAVE] {path_b}")


def plot_figure4_real_delta(
    sim: SimulatedDeltaMap,
    outdir: str,
    mirror_gap: float = MIRROR_GAP,
    delta_spot_size: float = DELTA_SPOT_SIZE,
    gray_spot_size: float = DELTA_GRAY_SPOT_SIZE,
    edgecolor: str = DELTA_EDGE_COLOR,
    linewidth: float = DELTA_EDGE_WIDTH,
):
    """Render real delta values and their Visium-like locations as two panels/files."""
    coords = make_mirrored_coords(sim.spatial, gap=mirror_gap)
    vals = duplicate_values_for_mirror(sim.delta)

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.percentile(np.abs(finite), DELTA_VMAX_PERCENTILE))
        vmax = max(vmax, 1e-8)

    # Figure 4a: real delta values.
    fig, ax = _make_transparent_fig(DELTA_FIG_W, DELTA_FIG_H)
    scatter_continuous(
        ax,
        coords,
        vals,
        cmap=DELTA_CMAP,
        size=delta_spot_size,
        title=f"{sim.gene_name} Δ | {sim.age_label}",
        vmin=-vmax,
        vmax=vmax,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    fig.tight_layout()
    path_a = os.path.join(outdir, f"figure4a_{sim.name}_delta.png")
    fig.savefig(path_a, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.close(fig)

    # Figure 4b: exactly the same simulated Visium locations, in gray.
    fig, ax = _make_transparent_fig(DELTA_FIG_W, DELTA_FIG_H)
    scatter_gray(ax, coords, size=gray_spot_size, title=f"{sim.name} locations")
    fig.tight_layout()
    path_b = os.path.join(outdir, f"figure4b_{sim.name}_locations.png")
    fig.savefig(path_b, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"[SAVE] {path_a}")
    print(f"[SAVE] {path_b}")



def parse_args():
    p = argparse.ArgumentParser(
        description="Simulate Visium-style pipeline views from AD mouse Xenium data."
    )
    p.add_argument("--input-h5ads", nargs="+", default=INPUT_H5ADS, help="Input Xenium h5ad files.")
    p.add_argument("--outdir", default=OUTDIR, help="Output directory.")
    p.add_argument("--visium-scale", type=float, default=VISIUM_SCALE, help="Center-to-center spacing of simulated Visium spots.")
    p.add_argument("--spot-radius-factor", type=float, default=SPOT_RADIUS_FACTOR, help="Simulated spot radius = factor * visium-scale.")
    p.add_argument("--min-cells-per-spot", type=int, default=MIN_CELLS_PER_SPOT, help="Minimum number of cells aggregated into a simulated spot.")
    p.add_argument("--mirror-gap", type=float, default=MIRROR_GAP, help="Gap at the mirror fold boundary.")
    p.add_argument("--figure1-channel", type=str, default=FIGURE1_CHANNEL, choices=["cell_density", "Arc", "Gfap", "axis_joint_domain"], help="Channel used for the slice-wise figures.")
    p.add_argument("--channel-spot-sizes", nargs=3, type=float, default=CHANNEL_SPOT_SIZES, metavar=("S1", "S2", "S3"), help="Slice-specific spot sizes for the channel panels of Figures 1-3.")
    p.add_argument("--gray-spot-sizes", nargs=3, type=float, default=GRAY_SPOT_SIZES, metavar=("S1", "S2", "S3"), help="Slice-specific spot sizes for the gray-location panels of Figures 1-3.")
    p.add_argument("--delta-spot-size", type=float, default=DELTA_SPOT_SIZE, help="Spot size for Figure 4 delta map.")
    p.add_argument("--delta-gray-spot-size", type=float, default=DELTA_GRAY_SPOT_SIZE, help="Spot size for Figure 4 gray-location map.")
    p.add_argument("--delta-gene", type=str, default=DELTA_GENE, help="Gene used for the disease-vs-normal delta; default is Arc.")
    p.add_argument("--delta-disease-path", type=str, default=DELTA_DISEASE_PATH, help="Disease H5AD used to compute the real delta.")
    p.add_argument("--delta-normal-path", type=str, default=DELTA_NORMAL_PATH, help="Normal/WT H5AD used to compute the real delta.")
    p.add_argument("--figure1-cmap", type=str, default=FIGURE1_CMAP, help="Matplotlib colormap used for the stylized channel rendering.")
    p.add_argument("--virtual-gene-scale", type=float, default=VIRTUAL_GENE_SCALE, help="Scaling factor applied to Arc/Gfap of the middle virtual tissue state.")
    p.add_argument("--swapxy", action="store_true", default=GLOBAL_SWAPXY, help="Swap x/y of all input spatial coordinates before all downstream processing.")
    p.add_argument("--invertx", action="store_true", default=GLOBAL_INVERTX, help="Invert the global x-axis after optional x/y swapping.")
    p.add_argument("--inverty", action="store_true", default=GLOBAL_INVERTY, help="Invert the global y-axis after optional x/y swapping.")
    return p.parse_args()


def main():
    args = parse_args()

    ensure_dir(args.outdir)
    sim_data_dir = os.path.join(args.outdir, "simu_data")
    fig_dir = os.path.join(args.outdir, "figures")
    ensure_dir(sim_data_dir)
    ensure_dir(fig_dir)

    simulated_slices: List[SimulatedSlice] = []

    print("[INFO] Simulating Visium-style spot data...")
    for slice_index, h5ad_path in enumerate(args.input_h5ads):
        if not os.path.exists(h5ad_path):
            raise FileNotFoundError(h5ad_path)

        slice_name = os.path.splitext(os.path.basename(h5ad_path))[0]
        adata = sc.read_h5ad(h5ad_path)
        is_virtual = (slice_index == VIRTUAL_SLICE_INDEX) or ("virtual" in slice_name.lower())

        sim = simulate_visium_from_xenium(
            adata=adata,
            slice_name=slice_name,
            visium_scale=args.visium_scale,
            spot_radius_factor=args.spot_radius_factor,
            min_cells_per_spot=args.min_cells_per_spot,
            domain_key=DOMAIN_KEY,
            swapxy=args.swapxy,
            invertx=args.invertx,
            inverty=args.inverty,
            slice_index=slice_index,
            is_virtual=is_virtual,
            virtual_gene_scale=args.virtual_gene_scale,
        )
        save_simulated_slice(sim, sim_data_dir)
        simulated_slices.append(sim)
        print(f"[OK] simulated {slice_name}: {sim.adata.n_obs} spots")

    simulated_slices = sorted(simulated_slices, key=lambda x: x.slice_index)

    global FIGURE1_CMAP
    FIGURE1_CMAP = args.figure1_cmap

    print(f"[INFO] Global spatial transform: swapxy={args.swapxy}, invertx={args.invertx}, inverty={args.inverty}")
    print(f"[INFO] Figure 1 stylized colormap: {FIGURE1_CMAP}")
    print("[INFO] Plotting Figures 1-4...")

    for i, sim in enumerate(simulated_slices[:3]):
        plot_slice_pair_figure(
            sim,
            outdir=fig_dir,
            channel=args.figure1_channel,
            mirror_gap=args.mirror_gap,
            channel_spot_size=float(args.channel_spot_sizes[i]),
            gray_spot_size=float(args.gray_spot_sizes[i]),
            figure_number=i + 1,
        )

    print("[INFO] Simulating Visium-style Arc disease-vs-normal delta...")
    print(f"[INFO] delta gene = {args.delta_gene}")
    print(f"[INFO] delta age = {DELTA_AGE_LABEL}")
    print(f"[INFO] disease H5AD = {args.delta_disease_path}")
    print(f"[INFO] normal H5AD = {args.delta_normal_path}")

    delta_sim = simulate_visium_from_case_study_delta(
        gene=args.delta_gene,
        age=DELTA_AGE,
        age_label=DELTA_AGE_LABEL,
        disease_path=args.delta_disease_path,
        normal_path=args.delta_normal_path,
        visium_scale=args.visium_scale,
        spot_radius_factor=args.spot_radius_factor,
        min_values_per_spot=args.min_cells_per_spot,
        swapxy=args.swapxy,
        invertx=args.invertx,
        inverty=args.inverty,
    )
    save_simulated_delta(delta_sim, sim_data_dir)
    print(f"[OK] simulated real delta: {delta_sim.name}: {delta_sim.adata.n_obs} spots")

    plot_figure4_real_delta(
        delta_sim,
        outdir=fig_dir,
        mirror_gap=args.mirror_gap,
        delta_spot_size=args.delta_spot_size,
        gray_spot_size=args.delta_gray_spot_size,
    )

    print("[DONE]")
    print(f"Simulated data saved to: {sim_data_dir}")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()