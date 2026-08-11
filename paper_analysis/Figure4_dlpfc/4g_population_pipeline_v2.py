#!/usr/bin/env python3
"""
DLPFC lifted 3D transition-field visualization with a front evidence plane.

Updates over the earlier version
--------------------------------
1. Keep the original 3D slice stack and continuous lifted flow.
2. Assume the biological flow direction is from the back slice (320 um)
   toward the front slice (0 um), and add arrowheads indicating this
   back-to-front direction along the continuous field.
3. Insert one evidence plane in front of the 0 um slice. This plane shows:
      - the user-defined Visium support (vacant circles), and
      - all collected evidence overlaid on that support.

The plotting mode is controlled by `SLICE_COLOR_KEY`:
- if it matches a gene in var_names, gene evidence is shown in Reds;
- if it matches an obs column, categorical evidence is shown with tab20.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from axis.coordinate_utils import find_inverse_warp_coords
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.interpolate import RegularGridInterpolator


# =============================================================================
# Configuration
# =============================================================================

ROOT = "/media/huifang/data/registration/DLPFC/huifang"
GROUP_ID = 1
SAMPLE_IDS = [0, 1, 2, 3]
REFERENCE_ID = 0
PAIRWISE_SUBDIR = "pairwise_output"
COORD_KEY = "spatial"

# Use actual anatomical depths.
AXIS_POSITIONS = [0.0, 150.0, 300, 450.0]

DOMAIN_KEY = "layer_guess_reordered"
TRAJECTORY_DOMAINS = ["Layer4"]
MAX_SEED_POINTS = 220
RANDOM_SEED = 0

# Display on slices and evidence plane.
# SLICE_COLOR_KEY: Optional[str] = "PCP4"   # gene name or obs column
SLICE_COLOR_KEY: Optional[str] = "PCP4"   # gene name or obs column
USE_LOG1P_FOR_GENE = True
SLICE_POINT_SIZE = 11
SLICE_POINT_ALPHA = 0.5
EVIDENCE_POINT_SIZE = 11
GRID_POINT_SIZE = 11

# Pairwise 2D curve shape.
CURVE_CONTROL_SCALE = 0.32
CURVE_STEPS_PER_PAIR = 32

# Continuous field rendering.
CURVE_CMAP = "viridis"
CURVE_LINEWIDTH = 1.0
CURVE_ALPHA = 0.72

# Direction arrows (from 320um -> 0um).
ARROW_AT_TARGET_DEPTHS = [310.0, 10.0, 0.0]
ARROW_LINEWIDTH = 0.9
ARROW_ALPHA = 0.95
ARROW_LENGTH_RATIO = 0.45
ARROW_SCALE = 1.0
MAX_ARROWS = 120

# Evidence plane in front of 0um slice.
EVIDENCE_PLANE_LABEL = "Collected evidence\nVisium support"
EVIDENCE_PLANE_OFFSET = 55.0   # plane depth = min_depth - offset

# 3D rendering.
ELEV = 15
AZIM = -145
FIGSIZE = (10.8, 7.8)

OUT_DIR: Optional[Path] = None
STEM = "dlpfc_lifted_transition_field_with_evidence_plane"

# DLPFC integration-generation files.
VIRTUAL_PATH = Path(
    "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/"
    "dlpfc_z_group1_grid80_spatial1/integration/"
    "integrated_dlpfc_z_ref_0_visium_ot.h5ad"
)
META_PATH = Path(
    "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/"
    "dlpfc_z_group1_grid80_spatial1/integration/"
    "integrated_dlpfc_z_ref_0_visium_ot_generation_meta.h5ad"
)


# =============================================================================
# Emergency AnnData read helper
# =============================================================================


def safe_read_h5ad(path: str | Path):
    """Read h5ad while bypassing problematic uns categorical cleanup if needed."""
    try:
        return ad.read_h5ad(path)
    except TypeError as exc:
        if "Unordered Categoricals can only compare equality or not" not in str(exc):
            raise
        import anndata.compat as compat
        old_clean_uns = getattr(compat, "_clean_uns", None)
        try:
            compat._clean_uns = lambda d: None
            return ad.read_h5ad(path)
        finally:
            if old_clean_uns is not None:
                compat._clean_uns = old_clean_uns


# =============================================================================
# IO
# =============================================================================


def parse_h5ad_path(root: str, group_id: int, sample_id: int) -> str:
    return os.path.join(root, "h5ad", f"{group_id}_{sample_id}.h5ad")


def parse_network_npz_path(root: str, group_id: int, sample_id: int) -> str:
    return os.path.join(root, "network_input", f"{group_id}_{sample_id}_validation.npz")


def find_group_dir(root: str, group_id: int, pairwise_subdir: str = "pairwise_output") -> str:
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
    pairwise_subdir: str = "pairwise_output",
) -> str:
    group_dir = find_group_dir(root, group_id, pairwise_subdir=pairwise_subdir)
    return os.path.join(group_dir, f"{fixed_id}_{moving_id}_flow.npy")


def load_npz_coord(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)
    coord = np.asarray(data["coord"], dtype=np.float32)
    return coord[:, :2]


def load_plot_slice_coords_aligned(
    root: str,
    group_id: int,
    sample_id: int,
    reference_id: int = 0,
    coord_key: str = "spatial",
    pairwise_subdir: str = "pairwise_output",
):
    adata = safe_read_h5ad(parse_h5ad_path(root, group_id, sample_id))

    if coord_key in adata.obsm:
        coord = np.asarray(adata.obsm[coord_key], dtype=np.float32)
        return adata, coord[:, :2]

    if sample_id == reference_id:
        coord = load_npz_coord(parse_network_npz_path(root, group_id, sample_id))
        return adata, coord

    group_dir = find_group_dir(root, group_id, pairwise_subdir=pairwise_subdir)
    warped_path = os.path.join(group_dir, f"{reference_id}_{sample_id}_warped_coor.npy")
    if os.path.exists(warped_path):
        coord = np.asarray(np.load(warped_path), dtype=np.float32)
        return adata, coord[:, :2]

    raise FileNotFoundError(
        f"Could not find aligned coordinates for group={group_id}, sample={sample_id}."
    )


def get_grid_points(meta: ad.AnnData) -> np.ndarray:
    if "grid_pts" not in meta.uns:
        raise KeyError(f"uns['grid_pts'] not found. Available keys: {list(meta.uns.keys())}")
    grid_pts = np.asarray(meta.uns["grid_pts"], dtype=float)
    return grid_pts[:, :2]


def get_spatial_coords(adata: ad.AnnData, spatial_key: str = "spatial") -> np.ndarray:
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    return coords[:, :2]


# =============================================================================
# Display helpers
# =============================================================================


def extract_gene_values(adata, gene: str, use_log1p: bool = True) -> np.ndarray:
    if gene not in adata.var_names:
        raise KeyError(f"Gene {gene!r} not found.")
    values = adata[:, gene].X
    if hasattr(values, "toarray"):
        values = values.toarray().ravel()
    else:
        values = np.asarray(values).ravel()
    values = values.astype(float)
    if use_log1p:
        values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
    return values


def canonical_category(x) -> str:
    if x is None:
        return "NA"
    text = str(x).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return "NA"
    return text


def load_slice_display_values(adata, color_key: Optional[str], use_log1p_gene: bool = True):
    if color_key is None:
        return None, "none"
    if color_key in adata.obs.columns:
        vals = np.asarray([canonical_category(v) for v in adata.obs[color_key].to_numpy()], dtype=str)
        return vals, "categorical"
    if color_key in adata.var_names:
        vals = extract_gene_values(adata, color_key, use_log1p=use_log1p_gene)
        return vals, "continuous"
    raise KeyError(f"{color_key!r} is neither an obs column nor a gene name.")


def build_stable_category_palette(values_by_source: Dict[str, np.ndarray], cmap_name: str = "tab20"):
    cats = sorted({str(v) for vals in values_by_source.values() for v in vals})
    cmap = plt.get_cmap(cmap_name, max(len(cats), 1))
    return {cat: cmap(i) for i, cat in enumerate(cats)}


# =============================================================================
# Flow and curve helpers
# =============================================================================


def ensure_flow_hw2(flow: np.ndarray) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    if flow.shape[-1] == 2:
        return flow
    if flow.shape[0] == 2:
        return np.moveaxis(flow, 0, -1)
    raise ValueError(f"Cannot interpret flow shape {flow.shape} as [H, W, 2]")


def inverse_warp_points(points_xy: np.ndarray, flow_hw2: np.ndarray) -> np.ndarray:
    warped = find_inverse_warp_coords(points_xy, flow_hw2)
    if isinstance(warped, (tuple, list)):
        warped = warped[0]
    warped = np.asarray(warped, dtype=np.float32)
    return warped[:, :2]


def make_flow_interpolator(flow_hw2: np.ndarray):
    h, w = flow_hw2.shape[:2]
    yy = np.arange(h, dtype=float)
    xx = np.arange(w, dtype=float)
    return RegularGridInterpolator((yy, xx), flow_hw2, bounds_error=False, fill_value=np.nan)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 1e-8:
        return np.array([np.nan, np.nan], dtype=float)
    return v / n


def sample_flow_vector(point_xy: np.ndarray, flow_interp) -> np.ndarray:
    q = np.asarray([[point_xy[1], point_xy[0]]], dtype=float)
    v = np.asarray(flow_interp(q)[0], dtype=float)
    return v


def cubic_bezier_2d(p0, p1, p2, p3, n_steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(n_steps), dtype=float)[:, None]
    omt = 1.0 - t
    curve = omt**3 * p0[None, :] + 3.0 * omt**2 * t * p1[None, :] + 3.0 * omt * t**2 * p2[None, :] + t**3 * p3[None, :]
    return curve.astype(np.float32)


def build_pairwise_2d_curve(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    forward_flow_interp,
    backward_flow_interp,
    control_scale: float,
    n_steps: int,
) -> np.ndarray:
    disp = end_xy - start_xy
    dist = float(np.linalg.norm(disp))
    if not np.isfinite(dist) or dist <= 1e-6:
        return np.repeat(start_xy[None, :], int(n_steps), axis=0).astype(np.float32)

    unit_disp = disp / dist
    vf = sample_flow_vector(start_xy, forward_flow_interp)
    vf = normalize_vector(vf)
    v0 = vf if np.isfinite(vf).all() else unit_disp

    if backward_flow_interp is not None:
        vb = sample_flow_vector(end_xy, backward_flow_interp)
        vb = normalize_vector(vb)
        v1 = -vb if np.isfinite(vb).all() else unit_disp
    else:
        v1 = unit_disp

    handle_len = control_scale * dist
    c1 = start_xy + handle_len * v0
    c2 = end_xy - handle_len * v1
    return cubic_bezier_2d(start_xy, c1, c2, end_xy, n_steps)


def lift_2d_curve_to_3d(curve_xy: np.ndarray, depth_start: float, depth_end: float) -> np.ndarray:
    depth = np.linspace(depth_start, depth_end, len(curve_xy), dtype=float)
    return np.column_stack([depth, curve_xy[:, 0], curve_xy[:, 1]]).astype(np.float32)


def build_lifted_pairwise_trajectories(
    root: str,
    group_id: int,
    sample_ids: Sequence[int],
    seed_points_first_slice: np.ndarray,
    plane_positions: Dict[int, float],
    pairwise_subdir: str,
    curve_control_scale: float,
    curve_steps_per_pair: int,
) -> List[np.ndarray]:
    n_seeds = len(seed_points_first_slice)
    curves_3d = [[] for _ in range(n_seeds)]
    current_points = seed_points_first_slice.copy()
    active = np.ones(n_seeds, dtype=bool)

    for pair_index in range(len(sample_ids) - 1):
        sid_a = sample_ids[pair_index]
        sid_b = sample_ids[pair_index + 1]

        flow_forward = ensure_flow_hw2(np.load(parse_flow_path(root, group_id, sid_b, sid_a, pairwise_subdir)))
        forward_interp = make_flow_interpolator(flow_forward)

        backward_interp = None
        reverse_path = parse_flow_path(root, group_id, sid_a, sid_b, pairwise_subdir)
        if os.path.exists(reverse_path):
            backward_interp = make_flow_interpolator(ensure_flow_hw2(np.load(reverse_path)))

        next_points = np.full_like(current_points, np.nan, dtype=np.float32)

        if np.any(active):
            idx = np.flatnonzero(active)
            warped = inverse_warp_points(current_points[idx], flow_forward)
            h, w = flow_forward.shape[:2]
            valid = np.all(np.isfinite(warped), axis=1)
            valid &= (warped[:, 0] >= 0) & (warped[:, 0] < w)
            valid &= (warped[:, 1] >= 0) & (warped[:, 1] < h)
            idx_valid = idx[valid]
            next_points[idx_valid] = warped[valid]
            new_active = np.zeros_like(active)
            new_active[idx_valid] = True
            active = new_active

        z0 = plane_positions[sid_a]
        z1 = plane_positions[sid_b]
        for i in np.flatnonzero(active):
            curve_xy = build_pairwise_2d_curve(
                start_xy=current_points[i],
                end_xy=next_points[i],
                forward_flow_interp=forward_interp,
                backward_flow_interp=backward_interp,
                control_scale=curve_control_scale,
                n_steps=curve_steps_per_pair,
            )
            curve_3d = lift_2d_curve_to_3d(curve_xy, z0, z1)
            curves_3d[i].append(curve_3d if len(curves_3d[i]) == 0 else curve_3d[1:])
        current_points = next_points

    out = []
    for pieces in curves_3d:
        if len(pieces) > 0:
            out.append(np.concatenate(pieces, axis=0))
    return out


# =============================================================================
# Plot helpers
# =============================================================================


def build_plane_positions(sample_ids: Sequence[int], axis_positions: Optional[Sequence[float]], depth_gap: float = 160.0):
    if axis_positions is None:
        return {sid: i * depth_gap for i, sid in enumerate(sample_ids)}
    if len(axis_positions) != len(sample_ids):
        raise ValueError("axis_positions length must match sample_ids length.")
    return {sid: float(z) for sid, z in zip(sample_ids, axis_positions)}


def draw_slice_plane(ax, depth: float, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float],
                     face_alpha: float = 0.02, edge_alpha: float = 0.25,
                     edgecolor=(0.45, 0.45, 0.45, 0.25), linewidth=0.65, linestyle="-"):
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    verts = [[(depth, x0, y0), (depth, x1, y0), (depth, x1, y1), (depth, x0, y1)]]
    poly = Poly3DCollection(verts, facecolors=(1, 1, 1, face_alpha), edgecolors=edgecolor, linewidths=linewidth, linestyles=linestyle)
    ax.add_collection3d(poly)


def add_plane_label(ax, depth: float, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float], text: str, color="black"):
    x = x_bounds[0]
    y = y_bounds[1] + 0.03 * max(y_bounds[1] - y_bounds[0], 1.0)
    ax.text(depth, x, y, text, color=color, fontsize=9.5, ha="left", va="bottom")


def add_curve_collection(ax, curves_3d: Sequence[np.ndarray], cmap_name: str, linewidth: float, alpha: float):
    segments_all, values_all = [], []
    for curve in curves_3d:
        if len(curve) < 2:
            continue
        segments = np.stack([curve[:-1], curve[1:]], axis=1)
        values = 0.5 * (curve[:-1, 0] + curve[1:, 0])
        segments_all.append(segments)
        values_all.append(values)
    segments_all = np.concatenate(segments_all, axis=0)
    values_all = np.concatenate(values_all, axis=0)
    norm = Normalize(vmin=float(np.nanmin(values_all)), vmax=float(np.nanmax(values_all)))
    coll = Line3DCollection(segments_all, cmap=cmap_name, norm=norm, linewidths=linewidth, alpha=alpha)
    coll.set_array(values_all)
    ax.add_collection3d(coll)
    return coll


def add_direction_arrows(ax, curves_3d: Sequence[np.ndarray], target_depths: Sequence[float], max_arrows: int = 120):
    if len(curves_3d) == 0:
        return
    if len(curves_3d) > max_arrows:
        idx = np.linspace(0, len(curves_3d) - 1, max_arrows).astype(int)
        curves_use = [curves_3d[i] for i in idx]
    else:
        curves_use = list(curves_3d)

    cmap = plt.get_cmap(CURVE_CMAP)
    depth_min = min(np.nanmin(c[:, 0]) for c in curves_use)
    depth_max = max(np.nanmax(c[:, 0]) for c in curves_use)
    norm = Normalize(vmin=depth_min, vmax=depth_max)

    for curve in curves_use:
        rev = curve[::-1]  # 320um -> 0um
        d = rev[:, 0]
        for tgt in target_depths:
            idx = np.argmin(np.abs(d - tgt))
            if idx < 1 or idx >= len(rev):
                continue
            p0 = rev[idx - 1]
            p1 = rev[idx]
            vec = p1 - p0
            if np.linalg.norm(vec) <= 1e-8:
                continue
            color = cmap(norm(0.5 * (p0[0] + p1[0])))
            ax.quiver(
                p0[0], p0[1], p0[2],
                vec[0], vec[1], vec[2],
                arrow_length_ratio=ARROW_LENGTH_RATIO,
                color=color,
                linewidth=ARROW_LINEWIDTH,
                alpha=ARROW_ALPHA,
                normalize=False,
                length=ARROW_SCALE,
            )


def plot_template_grid_3d(ax, grid_pts: np.ndarray, *, depth: float, point_size: float):
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


def plot_expression_points_3d(ax, coords: np.ndarray, values: np.ndarray, *, depth: float, norm: Normalize,
                              point_size: float, cmap_name: str = "Reds", alpha_max: float = 0.92):
    order = np.argsort(values)
    cmap = plt.get_cmap(cmap_name)
    vals = np.asarray(values, dtype=float)
    rgba = cmap(norm(vals))
    finite = np.isfinite(vals)
    scaled = np.zeros_like(vals, dtype=float)
    if np.any(finite):
        vmin, vmax = norm.vmin, norm.vmax
        scaled[finite] = (np.clip(vals[finite], vmin, vmax) - vmin) / max(vmax - vmin, 1e-8)
    rgba[:, 3] = 0.05 + alpha_max * scaled
    ax.scatter(np.full(len(coords), depth)[order], coords[order, 0], coords[order, 1], c=rgba[order], s=point_size,
               linewidths=0, depthshade=False, rasterized=True)


def plot_category_points_3d(ax, coords: np.ndarray, categories: np.ndarray, *, depth: float,
                            color_map: Dict[str, Tuple[float, float, float, float]], point_size: float,
                            alpha: float = 0.90):
    colors = [color_map[str(v)] for v in categories]
    ax.scatter(np.full(len(coords), depth), coords[:, 0], coords[:, 1], c=colors, s=point_size,
               alpha=alpha, linewidths=0, depthshade=False, rasterized=True)


def save_or_show(fig: plt.Figure, outdir: Optional[Path], stem: str, dpi: int = 400):
    if outdir is None:
        plt.show()
        return
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def plot_lifted_transition_field_3d_with_evidence_plane(
    root: str = ROOT,
    group_id: int = GROUP_ID,
    sample_ids: Sequence[int] = SAMPLE_IDS,
    *,
    reference_id: int = REFERENCE_ID,
    coord_key: str = COORD_KEY,
    pairwise_subdir: str = PAIRWISE_SUBDIR,
    axis_positions: Optional[Sequence[float]] = AXIS_POSITIONS,
    domain_key: str = DOMAIN_KEY,
    trajectory_domains: Optional[Sequence[str]] = TRAJECTORY_DOMAINS,
    max_seed_points: int = MAX_SEED_POINTS,
    random_seed: int = RANDOM_SEED,
    slice_color_key: Optional[str] = SLICE_COLOR_KEY,
    use_log1p_for_gene: bool = USE_LOG1P_FOR_GENE,
    outdir: Optional[Path] = OUT_DIR,
    stem: str = STEM,
):
    if len(sample_ids) < 2:
        raise ValueError("At least two slices are required.")

    plane_positions = build_plane_positions(sample_ids, axis_positions)
    evidence_depth = min(plane_positions.values()) - EVIDENCE_PLANE_OFFSET

    # Load aligned real slices.
    slice_cache: Dict[int, Dict[str, object]] = {}
    values_by_source: Dict[str, np.ndarray] = {}
    value_kinds = []
    all_x, all_y = [], []

    for sid in sample_ids:
        adata, coords = load_plot_slice_coords_aligned(
            root=root, group_id=group_id, sample_id=sid, reference_id=reference_id,
            coord_key=coord_key, pairwise_subdir=pairwise_subdir,
        )
        n_keep = min(adata.n_obs, len(coords))
        adata = adata[:n_keep].copy()
        coords = coords[:n_keep]
        values, kind = load_slice_display_values(adata, slice_color_key, use_log1p_for_gene)
        slice_cache[sid] = {"adata": adata, "coords": coords, "values": values, "kind": kind}
        if values is not None:
            values_by_source[f"slice_{sid}"] = values
        value_kinds.append(kind)
        all_x.append(coords[:, 0])
        all_y.append(coords[:, 1])

    # Load integration-generation metadata for the front evidence plane.
    meta = safe_read_h5ad(META_PATH)
    grid_pts = get_grid_points(meta)
    evidence_coords = get_spatial_coords(meta, spatial_key="spatial")
    evidence_values, evidence_kind = load_slice_display_values(meta, slice_color_key, use_log1p_for_gene)
    values_by_source["evidence"] = evidence_values
    value_kinds.append(evidence_kind)

    all_x.extend([grid_pts[:, 0], evidence_coords[:, 0]])
    all_y.extend([grid_pts[:, 1], evidence_coords[:, 1]])

    # Seed points from the first slice and the chosen domain(s).
    first_sid = sample_ids[0]
    first_record = slice_cache[first_sid]
    adata0 = first_record["adata"]
    coords0 = first_record["coords"]
    if domain_key not in adata0.obs.columns:
        raise KeyError(f"{domain_key!r} not found in the first slice obs.")
    if trajectory_domains is None:
        seed_mask = np.ones(len(coords0), dtype=bool)
    else:
        seed_mask = np.isin(adata0.obs[domain_key].astype(str).to_numpy(), np.asarray(trajectory_domains, dtype=str))
    seed_coords = coords0[seed_mask]
    if len(seed_coords) == 0:
        raise ValueError(f"No seed points found for domains={trajectory_domains}.")
    if len(seed_coords) > max_seed_points:
        rng = np.random.default_rng(random_seed)
        idx = np.sort(rng.choice(len(seed_coords), size=max_seed_points, replace=False))
        seed_coords = seed_coords[idx]

    curves_3d = build_lifted_pairwise_trajectories(
        root=root, group_id=group_id, sample_ids=sample_ids,
        seed_points_first_slice=seed_coords, plane_positions=plane_positions,
        pairwise_subdir=pairwise_subdir, curve_control_scale=CURVE_CONTROL_SCALE,
        curve_steps_per_pair=CURVE_STEPS_PER_PAIR,
    )
    print(f"[INFO] seeds selected: {len(seed_coords)} | lifted curves generated: {len(curves_3d)}")

    # Shared color control.
    unique_kinds = set(value_kinds)
    category_palette = None
    continuous_norm = None
    if slice_color_key is not None:
        if unique_kinds == {"categorical"}:
            category_palette = build_stable_category_palette(values_by_source, cmap_name="tab20")
        elif unique_kinds == {"continuous"}:
            combined = np.concatenate([np.asarray(v, dtype=float)[np.isfinite(v)] for v in values_by_source.values()])
            vmax = float(np.nanpercentile(combined, 99.5))
            continuous_norm = Normalize(vmin=float(np.nanmin(combined)), vmax=max(vmax, 1e-6))

    tissue_x = np.concatenate(all_x)
    tissue_y = np.concatenate(all_y)
    x_bounds = (float(np.nanmin(tissue_x)), float(np.nanmax(tissue_x)))
    y_bounds = (float(np.nanmin(tissue_y)), float(np.nanmax(tissue_y)))

    # Plot.
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")

    # Evidence plane first.
    draw_slice_plane(ax, evidence_depth, x_bounds, y_bounds, face_alpha=0.035, edge_alpha=0.18,
                     edgecolor=(0.55, 0.55, 0.55, 0.55), linewidth=0.8, linestyle="--")
    add_plane_label(ax, evidence_depth, x_bounds, y_bounds, EVIDENCE_PLANE_LABEL, color="#555555")
    plot_template_grid_3d(ax, grid_pts, depth=evidence_depth, point_size=GRID_POINT_SIZE)

    if evidence_kind == "categorical":
        plot_category_points_3d(ax, evidence_coords, evidence_values, depth=evidence_depth,
                                color_map=category_palette, point_size=EVIDENCE_POINT_SIZE, alpha=0.88)
    elif evidence_kind == "continuous":
        plot_expression_points_3d(ax, evidence_coords, evidence_values, depth=evidence_depth,
                                  norm=continuous_norm, point_size=EVIDENCE_POINT_SIZE, alpha_max=0.92)

    # Slice planes and context points.
    for sid in sample_ids:
        depth = plane_positions[sid]
        draw_slice_plane(ax, depth, x_bounds, y_bounds)
        label = f"{int(depth) if float(depth).is_integer() else depth:g} um"
        add_plane_label(ax, depth, x_bounds, y_bounds, label)

        record = slice_cache[sid]
        coords = record["coords"]
        values = record["values"]
        kind = record["kind"]
        if kind == "categorical":
            colors = [category_palette[str(v)] for v in values]
            ax.scatter(np.full(len(coords), depth), coords[:, 0], coords[:, 1], c=colors, s=SLICE_POINT_SIZE,
                       alpha=SLICE_POINT_ALPHA, linewidths=0, depthshade=False, rasterized=True)
        elif kind == "continuous":
            plot_expression_points_3d(ax, coords, values, depth=depth, norm=continuous_norm,
                                      point_size=SLICE_POINT_SIZE, alpha_max=SLICE_POINT_ALPHA)
        else:
            ax.scatter(np.full(len(coords), depth), coords[:, 0], coords[:, 1], c="0.5", s=SLICE_POINT_SIZE,
                       alpha=SLICE_POINT_ALPHA, linewidths=0, depthshade=False, rasterized=True)

    # Continuous lifted field and directional arrows.
    curve_collection = add_curve_collection(ax, curves_3d, CURVE_CMAP, CURVE_LINEWIDTH, CURVE_ALPHA)
    add_direction_arrows(ax, curves_3d, target_depths=ARROW_AT_TARGET_DEPTHS, max_arrows=MAX_ARROWS)

    # Axis styling.
    z_depths = np.asarray([evidence_depth] + [plane_positions[sid] for sid in sample_ids], dtype=float)
    depth_min, depth_max = float(np.min(z_depths)), float(np.max(z_depths))
    dx = 0.04 * max(x_bounds[1] - x_bounds[0], 1.0)
    dy = 0.04 * max(y_bounds[1] - y_bounds[0], 1.0)
    dz = 0.06 * max(depth_max - depth_min, 1.0)

    ax.set_xlim(depth_min - dz, depth_max + dz)
    ax.set_ylim(x_bounds[1] + dx, x_bounds[0] - dx)
    ax.set_zlim(y_bounds[1] + dy, y_bounds[0] - dy)
    ax.set_xticks(z_depths)
    ax.set_xticklabels(["Evidence", "0", "10", "310", "320"], fontsize=9)
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("Depth (um)", labelpad=10)
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.grid(False)
    ax.set_box_aspect((max(depth_max - depth_min, 1.0) * 1.35,
                       max(x_bounds[1] - x_bounds[0], 1.0),
                       max(y_bounds[1] - y_bounds[0], 1.0)))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))

    domain_text = "all domains" if trajectory_domains is None else ", ".join(map(str, trajectory_domains))
    ax.set_title(
        f"Lifted continuous transition field with front evidence plane | {domain_text}",
        fontsize=13, fontweight="bold", pad=14,
    )

    cbar = fig.colorbar(curve_collection, ax=ax, fraction=0.028, pad=0.06, shrink=0.68)
    cbar.set_label("Depth progression of continuous field", fontsize=9)

    if category_palette is not None:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color,
                   markeredgecolor="none", markersize=5, label=label)
            for label, color in category_palette.items()
        ]
        ax.legend(handles=handles, title=slice_color_key, loc="upper left",
                  bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=8, title_fontsize=8.5)
    elif continuous_norm is not None:
        sm = plt.cm.ScalarMappable(norm=continuous_norm, cmap="Reds")
        cbar2 = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.015, shrink=0.68)
        label = f"Evidence / slice value: log1p({slice_color_key})" if use_log1p_for_gene else f"Evidence / slice value: {slice_color_key}"
        cbar2.set_label(label, fontsize=9)

    fig.tight_layout()
    save_or_show(fig, outdir=outdir, stem=stem)
    return fig, ax, curves_3d


if __name__ == "__main__":
    plot_lifted_transition_field_3d_with_evidence_plane()
