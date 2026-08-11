"""
SCC iSTAR gene-map visualization and AXIS alignment.

Produces two figures:

Figure 1
--------
A 1 x 2 comparison for one sample (default: sample 0):
    1. original spot-level marker-gene map
    2. iSTAR-imputed marker-gene map

The two modalities use independent robust expression normalization by default.
This is intentional because one original Visium spot aggregates counts over a
large capture area, while one iSTAR location represents a much finer imputed
support. Their numeric ranges are therefore not directly interchangeable.

Figure 2
--------
A 1 x 2 overlay of all iSTAR marker-gene maps:
    1. before AXIS alignment
    2. after AXIS alignment into reference sample 0

The AXIS transformation follows the convention used in the supplied code:
for source -> target mapping, load the pairwise flow with
    parse_flow_path(fixed_id=target, moving_id=source)
and evaluate
    find_inverse_warp_coords(source_points, flow).

Direct source -> reference mapping is used when available. Otherwise, adjacent
pairwise fields are chained according to sample_ids, e.g. 2 -> 1 -> 0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

# These utilities are used in the original SCC/iSTAR code.
from utils_istar import (
    load_istar_gene_slice,
    load_istar_mask,
    upsample_flow_to_target_shape,
)
from utils_visualization import parse_flow_path, ensure_flow_hw2
from axis.coordinate_utils import find_inverse_warp_coords


# =============================================================================
# Data containers
# =============================================================================
@dataclass
class GeneSlice:
    sample_id: Any
    coords: np.ndarray
    values: np.ndarray


@dataclass
class WarpResult:
    coords: np.ndarray
    valid_mask: np.ndarray
    path: List[Tuple[Any, Any]]
    valid_fraction: float
    median_target_nn_distance: float


# =============================================================================
# Temporary cache for warped iSTAR coordinates
# =============================================================================
def _temp_result_dir(base_dir: Optional[str] = None) -> str:
    if base_dir is None:
        base_dir = os.getcwd()
    out = os.path.join(base_dir, "temp_result")
    os.makedirs(out, exist_ok=True)
    return out


def _warp_cache_path(
    cache_dir: str,
    patient_id: Any,
    gene: str,
    sample_ids: Sequence[Any],
    reference_id: Any,
    source_id: Any,
    use_log1p: bool,
    prefer_direct_flow: bool,
) -> str:
    sample_tag = "-".join(str(x) for x in sample_ids)
    gene_tag = str(gene).replace(os.sep, "_")
    fname = (
        f"patient{patient_id}_gene_{gene_tag}_samples_{sample_tag}"
        f"_ref_{reference_id}_src_{source_id}"
        f"_log1p_{int(bool(use_log1p))}_direct_{int(bool(prefer_direct_flow))}.npz"
    )
    return os.path.join(cache_dir, fname)


def save_warp_result_cache(cache_path: str, result: WarpResult) -> None:
    path_arr = np.asarray(result.path, dtype=object)
    np.savez_compressed(
        cache_path,
        coords=np.asarray(result.coords, dtype=np.float32),
        valid_mask=np.asarray(result.valid_mask, dtype=bool),
        path_edges=path_arr,
        valid_fraction=np.asarray([result.valid_fraction], dtype=np.float32),
        median_target_nn_distance=np.asarray([result.median_target_nn_distance], dtype=np.float32),
    )


def load_warp_result_cache(cache_path: str) -> WarpResult:
    with np.load(cache_path, allow_pickle=True) as data:
        coords = np.asarray(data["coords"], dtype=np.float32)
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)
        path_edges_raw = data.get("path_edges", np.empty((0,), dtype=object))
        path_edges_raw = np.asarray(path_edges_raw, dtype=object)
        path = []
        if path_edges_raw.size > 0:
            flat = path_edges_raw.reshape(-1, 2) if path_edges_raw.ndim >= 2 else [tuple(x) for x in path_edges_raw]
            if isinstance(flat, list):
                path = [(edge[0], edge[1]) for edge in flat]
            else:
                path = [(edge[0], edge[1]) for edge in flat]
        valid_fraction = float(np.asarray(data["valid_fraction"]).reshape(-1)[0])
        median_nn = float(np.asarray(data["median_target_nn_distance"]).reshape(-1)[0])
    return WarpResult(
        coords=coords,
        valid_mask=valid_mask,
        path=path,
        valid_fraction=valid_fraction,
        median_target_nn_distance=median_nn,
    )


def get_or_compute_warp_result(
    axis_root: str,
    istar_root: str,
    patient_id: Any,
    sample_ids: Sequence[Any],
    source_id: Any,
    reference_id: Any,
    source_coords: np.ndarray,
    reference_coords: np.ndarray,
    gene: str,
    pairwise_subdir: str = "pairwise_output",
    prefer_direct: bool = True,
    report: bool = True,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    use_log1p: bool = True,
) -> WarpResult:
    cache_dir = _temp_result_dir(cache_dir)
    cache_path = _warp_cache_path(
        cache_dir=cache_dir,
        patient_id=patient_id,
        gene=gene,
        sample_ids=sample_ids,
        reference_id=reference_id,
        source_id=source_id,
        use_log1p=use_log1p,
        prefer_direct_flow=prefer_direct,
    )

    if use_cache and os.path.exists(cache_path):
        result = load_warp_result_cache(cache_path)
        if report:
            readable_path = " -> ".join(str(x) for x in [source_id] + [b for _, b in result.path])
            if not result.path:
                readable_path = str(reference_id)
            print(
                f"[CACHE hit] sample {source_id} to reference {reference_id} | "
                f"path: {readable_path} | valid: {result.valid_fraction:.2%} | "
                f"median NN to reference support: {result.median_target_nn_distance:.3f} pixels | "
                f"cache: {cache_path}"
            )
        return result

    result = warp_istar_to_reference(
        axis_root=axis_root,
        istar_root=istar_root,
        patient_id=patient_id,
        sample_ids=sample_ids,
        source_id=source_id,
        reference_id=reference_id,
        source_coords=source_coords,
        reference_coords=reference_coords,
        pairwise_subdir=pairwise_subdir,
        prefer_direct=prefer_direct,
        report=report,
    )

    if use_cache:
        save_warp_result_cache(cache_path, result)
        if report:
            print(f"[CACHE save] {cache_path}")
    return result


# =============================================================================
# Generic array helpers
# =============================================================================
def _as_xy(coords: np.ndarray, name: str) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"{name} must have shape [N, 2+], got {coords.shape}")
    return coords[:, :2]


def _flatten_values(values: np.ndarray, name: str) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    values = np.asarray(values).reshape(-1).astype(np.float32)
    if values.ndim != 1:
        raise ValueError(f"{name} could not be converted to a 1D vector")
    return values


def _validate_pair(coords: np.ndarray, values: np.ndarray, context: str) -> None:
    if coords.shape[0] != values.shape[0]:
        raise ValueError(
            f"Coordinate/value mismatch for {context}: "
            f"{coords.shape[0]} coordinates versus {values.shape[0]} values"
        )


def _finite_xy_mask(coords: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(coords), axis=1)


# =============================================================================
# Original spot-level data
# =============================================================================
def _extract_h5ad_gene(
    adata,
    gene: str,
    layer: Optional[str] = None,
    use_raw: bool = False,
    log1p: bool = True,
) -> np.ndarray:
    """Extract one gene from X, a layer, or adata.raw."""
    if use_raw:
        if adata.raw is None:
            raise ValueError("original_use_raw=True, but adata.raw is None")
        if gene not in adata.raw.var_names:
            raise KeyError(f"Gene '{gene}' not found in adata.raw.var_names")
        values = adata.raw[:, gene].X
    elif layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        if gene not in adata.var_names:
            raise KeyError(f"Gene '{gene}' not found in adata.var_names")
        gene_idx = int(adata.var_names.get_loc(gene))
        values = adata.layers[layer][:, gene_idx]
    else:
        if gene not in adata.var_names:
            raise KeyError(f"Gene '{gene}' not found in adata.var_names")
        values = adata[:, gene].X

    values = _flatten_values(values, f"original {gene}")
    values = np.maximum(values, 0.0)
    return np.log1p(values) if log1p else values


def load_original_spot_gene_slice(
    axis_root: str,
    patient_id: Any,
    sample_id: Any,
    gene: str,
    h5ad_subdir: str = "h5ad",
    coord_key: str = "spatial_image_coor",
    original_layer: Optional[str] = None,
    original_use_raw: bool = False,
    use_log1p: bool = True,
) -> GeneSlice:
    path = os.path.join(axis_root, h5ad_subdir, f"{patient_id}_{sample_id}.h5ad")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    adata = sc.read_h5ad(path)
    if coord_key not in adata.obsm:
        raise KeyError(f"obsm['{coord_key}'] not found in {path}")

    coords = _as_xy(adata.obsm[coord_key], f"obsm['{coord_key}']")
    values = _extract_h5ad_gene(
        adata,
        gene=gene,
        layer=original_layer,
        use_raw=original_use_raw,
        log1p=use_log1p,
    )
    _validate_pair(coords, values, f"patient {patient_id}, sample {sample_id}")
    return GeneSlice(sample_id=sample_id, coords=coords, values=values)


# =============================================================================
# iSTAR data
# =============================================================================
def load_istar_gene_map(
    istar_root: str,
    patient_id: Any,
    sample_id: Any,
    gene: str,
    use_log1p: bool = True,
) -> GeneSlice:
    payload = load_istar_gene_slice(
        patient_id=patient_id,
        sample_id=sample_id,
        gene=gene,
        istar_root=istar_root,
        axis_id=sample_id,
        use_log1p=use_log1p,
    )

    coords = _as_xy(payload["coords"], f"iSTAR coordinates, sample {sample_id}")
    values = _flatten_values(payload["values"], f"iSTAR {gene}, sample {sample_id}")
    _validate_pair(coords, values, f"iSTAR sample {sample_id}")
    return GeneSlice(sample_id=sample_id, coords=coords, values=values)


# =============================================================================
# Robust expression normalization
# =============================================================================
def robust_expression_vmax(
    values: np.ndarray,
    percentile: float = 99.5,
    nonzero_only: bool = True,
) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if nonzero_only:
        values = values[values > 0]
    if values.size == 0:
        return 1.0
    return max(float(np.percentile(values, percentile)), 1e-8)


def pooled_expression_vmax(
    arrays: Iterable[np.ndarray],
    percentile: float = 99.5,
    nonzero_only: bool = True,
) -> float:
    kept = []
    for values in arrays:
        values = np.asarray(values, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if nonzero_only:
            values = values[values > 0]
        if values.size:
            kept.append(values)
    if not kept:
        return 1.0
    return max(float(np.percentile(np.concatenate(kept), percentile)), 1e-8)


# =============================================================================
# AXIS flow loading and coordinate transformation
# =============================================================================
def inverse_warp_points(points_xy: np.ndarray, flow_hw2: np.ndarray) -> np.ndarray:
    """Apply the local AXIS inverse-coordinate utility with shape validation."""
    result = find_inverse_warp_coords(
        np.asarray(points_xy, dtype=np.float32),
        ensure_flow_hw2(flow_hw2),
    )
    if isinstance(result, (tuple, list)):
        result = result[0]
    result = np.asarray(result, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] < 2:
        raise ValueError(f"Unexpected inverse-warp result shape: {result.shape}")
    return result[:, :2]


def flow_path_for_source_to_target(
    axis_root: str,
    patient_id: Any,
    source_id: Any,
    target_id: Any,
    pairwise_subdir: str = "pairwise_output",
) -> str:
    """
    Resolve the flow used for source_id -> target_id mapping.

    This is the same orientation as the supplied transition code. In that code,
    a fixed -> moving trajectory loads parse_flow_path with the IDs reversed.
    Therefore, source -> target uses:
        fixed_id=target_id, moving_id=source_id
    followed by find_inverse_warp_coords(source_points, flow).
    """
    path = parse_flow_path(
        root=axis_root,
        group_id=patient_id,
        fixed_id=target_id,
        moving_id=source_id,
        pairwise_subdir=pairwise_subdir,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def load_high_resolution_flow(
    axis_root: str,
    istar_root: str,
    patient_id: Any,
    source_id: Any,
    target_id: Any,
    pairwise_subdir: str = "pairwise_output",
) -> np.ndarray:
    """
    Load the low-resolution AXIS flow and resize it to the source iSTAR support.

    The source-mask convention matches precompute_all_hr_flows in the supplied
    code, where the target shape passed to upsample_flow_to_target_shape is the
    high-resolution mask of the points being transformed.
    """
    flow_path = flow_path_for_source_to_target(
        axis_root=axis_root,
        patient_id=patient_id,
        source_id=source_id,
        target_id=target_id,
        pairwise_subdir=pairwise_subdir,
    )
    low_flow = ensure_flow_hw2(np.load(flow_path))

    source_mask = load_istar_mask(
        patient_id=patient_id,
        axis_id=source_id,
        sample_id=source_id,
        istar_root=istar_root,
    )
    source_shape = tuple(np.asarray(source_mask).shape[:2])
    high_flow = upsample_flow_to_target_shape(low_flow, target_shape=source_shape)
    return ensure_flow_hw2(high_flow)


def _candidate_direct_path_exists(
    axis_root: str,
    patient_id: Any,
    source_id: Any,
    target_id: Any,
    pairwise_subdir: str,
) -> bool:
    try:
        path = parse_flow_path(
            root=axis_root,
            group_id=patient_id,
            fixed_id=target_id,
            moving_id=source_id,
            pairwise_subdir=pairwise_subdir,
        )
    except Exception:
        return False
    return os.path.exists(path)


def choose_mapping_path(
    axis_root: str,
    patient_id: Any,
    sample_ids: Sequence[Any],
    source_id: Any,
    reference_id: Any,
    pairwise_subdir: str,
    prefer_direct: bool = True,
) -> List[Tuple[Any, Any]]:
    """Choose direct source -> reference mapping, otherwise chain adjacent pairs."""
    sample_ids = list(sample_ids)
    if source_id == reference_id:
        return []
    if source_id not in sample_ids or reference_id not in sample_ids:
        raise ValueError("source_id and reference_id must be present in sample_ids")

    if prefer_direct and _candidate_direct_path_exists(
        axis_root,
        patient_id,
        source_id,
        reference_id,
        pairwise_subdir,
    ):
        return [(source_id, reference_id)]

    src_idx = sample_ids.index(source_id)
    ref_idx = sample_ids.index(reference_id)
    path: List[Tuple[Any, Any]] = []

    if src_idx > ref_idx:
        for idx in range(src_idx, ref_idx, -1):
            path.append((sample_ids[idx], sample_ids[idx - 1]))
    else:
        for idx in range(src_idx, ref_idx):
            path.append((sample_ids[idx], sample_ids[idx + 1]))

    for src, tgt in path:
        if not _candidate_direct_path_exists(
            axis_root,
            patient_id,
            src,
            tgt,
            pairwise_subdir,
        ):
            raise FileNotFoundError(
                f"No direct {source_id}->{reference_id} flow and missing chained "
                f"flow for {src}->{tgt}."
            )
    return path


def warp_istar_to_reference(
    axis_root: str,
    istar_root: str,
    patient_id: Any,
    sample_ids: Sequence[Any],
    source_id: Any,
    reference_id: Any,
    source_coords: np.ndarray,
    reference_coords: np.ndarray,
    pairwise_subdir: str = "pairwise_output",
    prefer_direct: bool = True,
    report: bool = True,
) -> WarpResult:
    """
    Map one iSTAR sample into the reference frame and report diagnostics.

    Diagnostics include:
      - fraction of finite coordinates inside the reference mask dimensions
      - median nearest-neighbor distance to the reference iSTAR tissue support
    """
    source_coords = _as_xy(source_coords, "source_coords")
    reference_coords = _as_xy(reference_coords, "reference_coords")

    path = choose_mapping_path(
        axis_root=axis_root,
        patient_id=patient_id,
        sample_ids=sample_ids,
        source_id=source_id,
        reference_id=reference_id,
        pairwise_subdir=pairwise_subdir,
        prefer_direct=prefer_direct,
    )

    current = source_coords.copy()
    for src, tgt in path:
        flow_hr = load_high_resolution_flow(
            axis_root=axis_root,
            istar_root=istar_root,
            patient_id=patient_id,
            source_id=src,
            target_id=tgt,
            pairwise_subdir=pairwise_subdir,
        )
        current = inverse_warp_points(current, flow_hr)

    reference_mask = load_istar_mask(
        patient_id=patient_id,
        axis_id=reference_id,
        sample_id=reference_id,
        istar_root=istar_root,
    )
    ref_h, ref_w = np.asarray(reference_mask).shape[:2]

    valid = _finite_xy_mask(current)
    valid &= (
        (current[:, 0] >= 0)
        & (current[:, 0] < ref_w)
        & (current[:, 1] >= 0)
        & (current[:, 1] < ref_h)
    )

    mapped = np.full_like(current, np.nan, dtype=np.float32)
    mapped[valid] = current[valid]
    valid_fraction = float(valid.mean()) if valid.size else 0.0

    ref_valid = reference_coords[_finite_xy_mask(reference_coords)]
    if np.any(valid) and ref_valid.shape[0] > 0:
        tree = cKDTree(ref_valid)
        nn_dist, _ = tree.query(mapped[valid], k=1)
        median_nn = float(np.median(nn_dist))
    else:
        median_nn = np.nan

    if report:
        readable_path = " -> ".join(str(x) for x in [source_id] + [b for _, b in path])
        if not path:
            readable_path = str(reference_id)
        print(
            f"[AXIS warp] sample {source_id} to reference {reference_id} | "
            f"path: {readable_path} | valid: {valid_fraction:.2%} | "
            f"median NN to reference support: {median_nn:.3f} pixels"
        )

    return WarpResult(
        coords=mapped,
        valid_mask=valid,
        path=path,
        valid_fraction=valid_fraction,
        median_target_nn_distance=median_nn,
    )


# =============================================================================
# Plotting helpers
# =============================================================================
def style_spatial_axis(ax, title: str, invert_y: bool = True) -> None:
    ax.set_title(title, fontsize=14, pad=7,weight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if invert_y:
        ax.invert_yaxis()


def spatial_limits(
    coords_list: Sequence[np.ndarray],
    pad_ratio: float = 0.03,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    finite = []
    for coords in coords_list:
        coords = _as_xy(coords, "coordinates")
        keep = _finite_xy_mask(coords)
        if np.any(keep):
            finite.append(coords[keep])
    if not finite:
        raise ValueError("No finite coordinates available for spatial limits")
    merged = np.vstack(finite)
    lo = merged.min(axis=0)
    hi = merged.max(axis=0)
    pad = np.maximum((hi - lo) * pad_ratio, 1e-6)
    return (
        (float(lo[0] - pad[0]), float(hi[0] + pad[0])),
        (float(lo[1] - pad[1]), float(hi[1] + pad[1])),
    )


def apply_limits(ax, xlim, ylim, invert_y: bool) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if invert_y:
        ax.invert_yaxis()


def gene_rgba(
    values: np.ndarray,
    norm: Normalize,
    cmap_name: str,
    alpha_max: float = 0.75,
    alpha_gamma: float = 1.3,
    alpha_floor: float = 0.0,
) -> np.ndarray:
    """Encode gene value in both color and opacity; zero values are transparent."""
    values = np.asarray(values, dtype=float).reshape(-1)
    scaled = np.asarray(norm(values), dtype=float)
    scaled = np.clip(scaled, 0.0, 1.0)
    rgba = np.asarray(get_cmap(cmap_name)(scaled), dtype=float)

    alpha = alpha_floor + (alpha_max - alpha_floor) * np.power(scaled, alpha_gamma)
    alpha[~np.isfinite(values)] = 0.0
    alpha[values <= 0] = 0.0
    rgba[:, 3] = np.clip(alpha, 0.0, 1.0)
    return rgba


# =============================================================================
# Figure 1: sample 0, original vs iSTAR
# =============================================================================
def plot_original_vs_istar_sample0(
    axis_root: str,
    istar_root: str,
    patient_id: Any,
    gene: str,
    sample_id: Any = 0,
    h5ad_subdir: str = "h5ad",
    coord_key: str = "spatial_image_coor",
    original_layer: Optional[str] = None,
    original_use_raw: bool = False,
    use_log1p: bool = True,
    gene_cmap: str = "Reds",
    original_percentile: float = 99.0,
    istar_percentile: float = 99.5,
    independent_color_scales: bool = True,
    original_point_size: float = 24.0,
    istar_point_size: float = 2.0,
    original_alpha: float = 0.95,
    istar_alpha: float = 0.95,
    share_spatial_limits: bool = True,
    invert_y: bool = True,
    figsize: Tuple[float, float] = (9.2, 4.6),
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
):
    """
    Plot sample 0 as a 1 x 2 comparison.

    By default, the original and iSTAR maps use independent robust color scales.
    This prevents spot-level counts from saturating against a pixel-level iSTAR
    scale while preserving spatial-pattern contrast in each modality.
    """
    original = load_original_spot_gene_slice(
        axis_root=axis_root,
        patient_id=patient_id,
        sample_id=sample_id,
        gene=gene,
        h5ad_subdir=h5ad_subdir,
        coord_key=coord_key,
        original_layer=original_layer,
        original_use_raw=original_use_raw,
        use_log1p=use_log1p,
    )
    istar = load_istar_gene_map(
        istar_root=istar_root,
        patient_id=patient_id,
        sample_id=sample_id,
        gene=gene,
        use_log1p=use_log1p,
    )

    original_vmax = robust_expression_vmax(
        original.values,
        percentile=original_percentile,
        nonzero_only=True,
    )
    istar_vmax = robust_expression_vmax(
        istar.values,
        percentile=istar_percentile,
        nonzero_only=True,
    )

    if independent_color_scales:
        original_norm = Normalize(vmin=0.0, vmax=original_vmax)
        istar_norm = Normalize(vmin=0.0, vmax=istar_vmax)
    else:
        shared_vmax = max(original_vmax, istar_vmax)
        original_norm = Normalize(vmin=0.0, vmax=shared_vmax)
        istar_norm = original_norm

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 0.035, 1.0, 0.035],
        left=0.035,
        right=0.97,
        bottom=0.08,
        top=0.84,
        wspace=0.12,
    )
    ax_original = fig.add_subplot(gs[0, 0])
    cax_original = fig.add_subplot(gs[0, 1])
    ax_istar = fig.add_subplot(gs[0, 2])
    cax_istar = fig.add_subplot(gs[0, 3])

    ax_original.scatter(
        original.coords[:, 0],
        original.coords[:, 1],
        c=original.values,
        cmap=gene_cmap,
        norm=original_norm,
        s=original_point_size,
        alpha=original_alpha,
        linewidths=0,
        rasterized=True,
    )
    style_spatial_axis(
        ax_original,
        f"Original spot-level map\nSample {sample_id}",
        invert_y=invert_y,
    )

    ax_istar.scatter(
        istar.coords[:, 0],
        istar.coords[:, 1],
        c=istar.values,
        cmap=gene_cmap,
        norm=istar_norm,
        s=istar_point_size,
        alpha=istar_alpha,
        linewidths=0,
        rasterized=True,
    )
    style_spatial_axis(
        ax_istar,
        f"iSTAR imputed map\nSample {sample_id}",
        invert_y=invert_y,
    )

    if share_spatial_limits:
        xlim, ylim = spatial_limits([original.coords, istar.coords])
        apply_limits(ax_original, xlim, ylim, invert_y=invert_y)
        apply_limits(ax_istar, xlim, ylim, invert_y=invert_y)

    sm_original = ScalarMappable(norm=original_norm, cmap=gene_cmap)
    sm_original.set_array([])
    cb_original = fig.colorbar(sm_original, cax=cax_original)
    cb_original.set_label(
        f"Original {gene}\nlog1p(counts)" if use_log1p else f"Original {gene}",
        fontsize=9,
    )
    cb_original.ax.tick_params(labelsize=8)

    sm_istar = ScalarMappable(norm=istar_norm, cmap=gene_cmap)
    sm_istar.set_array([])
    cb_istar = fig.colorbar(sm_istar, cax=cax_istar)
    cb_istar.set_label(
        f"iSTAR {gene}\nlog1p(signal)" if use_log1p else f"iSTAR {gene}",
        fontsize=9,
    )
    cb_istar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Patient {patient_id} | {gene} | original spot-level vs iSTAR imputed map",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =============================================================================
# Figure 2: iSTAR gene-map overlay before/after AXIS
# =============================================================================
def resolve_sample_cmaps(
    sample_ids: Sequence[Any],
    sample_cmaps: Optional[Sequence[str]] = None,
    fallback_cmap: str = "Reds",
) -> Dict[Any, str]:
    sample_ids = list(sample_ids)

    if sample_cmaps is None:
        default_pool = [
            "Reds",
            "Blues",
            "Greens",
            "Purples",
            "Oranges",
            "YlOrBr",
            "PuRd",
            "GnBu",
        ]
        cmap_names = default_pool[: len(sample_ids)]
    else:
        if len(sample_cmaps) < len(sample_ids):
            raise ValueError(
                "sample_cmaps must have at least len(sample_ids) elements"
            )
        cmap_names = list(sample_cmaps[: len(sample_ids)])

    if len(cmap_names) == 0:
        cmap_names = [fallback_cmap]

    return {sid: cmap_names[i] for i, sid in enumerate(sample_ids)}

def plot_istar_gene_overlay_before_after_axis(
    axis_root: str,
    istar_root: str,
    patient_id: Any,
    sample_ids: Sequence[Any],
    gene: str,
    reference_id: Any = 0,
    pairwise_subdir: str = "pairwise_output",
    use_log1p: bool = True,
    gene_cmap: str = "Reds",
    sample_cmaps: Optional[Sequence[str]] = None,
    expression_percentile: float = 99.5,
    point_size: float = 1.2,
    alpha_max: float = 0.9,
    alpha_gamma: float = 1.,
    prefer_direct_flow: bool = True,
    report_warp_diagnostics: bool = True,
    use_cached_warp: bool = True,
    cache_dir: Optional[str] = None,
    share_before_after_limits: bool = True,
    invert_y: bool = True,
    figsize: Tuple[float, float] = (11.2, 5.0),
    show_colorbar=False,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
):
    """
    Overlay the actual iSTAR spatial gene maps before and after AXIS alignment.

    Gene values are unchanged by alignment. The same robust normalization and
    value-dependent alpha are used in both panels. All samples are concatenated
    into one scatter collection per panel, preventing slice draw-order bias.
    """
    sample_ids = list(sample_ids)
    if len(sample_ids) < 2:
        raise ValueError("sample_ids must contain at least two samples")
    if reference_id not in sample_ids:
        raise ValueError("reference_id must be included in sample_ids")

    slices: Dict[Any, GeneSlice] = {
        sid: load_istar_gene_map(
            istar_root=istar_root,
            patient_id=patient_id,
            sample_id=sid,
            gene=gene,
            use_log1p=use_log1p,
        )
        for sid in sample_ids
    }

    reference_coords = slices[reference_id].coords
    aligned_coords: Dict[Any, np.ndarray] = {}
    warp_results: Dict[Any, WarpResult] = {}

    for sid in sample_ids:
        if sid == reference_id:
            valid = _finite_xy_mask(slices[sid].coords)
            result = WarpResult(
                coords=slices[sid].coords.copy(),
                valid_mask=valid,
                path=[],
                valid_fraction=float(valid.mean()),
                median_target_nn_distance=0.0,
            )
            if report_warp_diagnostics:
                print(
                    f"[AXIS warp] sample {sid} is the reference; "
                    f"valid: {result.valid_fraction:.2%}"
                )
        else:
            result = get_or_compute_warp_result(
                axis_root=axis_root,
                istar_root=istar_root,
                patient_id=patient_id,
                sample_ids=sample_ids,
                source_id=sid,
                reference_id=reference_id,
                source_coords=slices[sid].coords,
                reference_coords=reference_coords,
                gene=gene,
                pairwise_subdir=pairwise_subdir,
                prefer_direct=prefer_direct_flow,
                report=report_warp_diagnostics,
                use_cache=use_cached_warp,
                cache_dir=cache_dir,
                use_log1p=use_log1p,
            )
        warp_results[sid] = result
        aligned_coords[sid] = result.coords

    # A single expression scale is still used for all samples.
    vmax = pooled_expression_vmax(
        [slices[sid].values for sid in sample_ids],
        percentile=expression_percentile,
        nonzero_only=True,
    )
    norm = Normalize(vmin=0.0, vmax=vmax)

    sample_cmap_map = resolve_sample_cmaps(
        sample_ids=sample_ids,
        sample_cmaps=sample_cmaps,
        fallback_cmap=gene_cmap,
    )
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    if show_colorbar:
        gs = fig.add_gridspec(
            1,
            4,
            width_ratios=[1.0, 1.0, 0.05, 0.28],
            left=0.04,
            right=0.98,
            bottom=0.08,
            top=0.84,
            wspace=0.08,
        )

        ax_before = fig.add_subplot(gs[0, 0])
        ax_after = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])
        ax_legend = fig.add_subplot(gs[0, 3])

    else:
        gs = fig.add_gridspec(
            1,
            3,
            width_ratios=[1.0, 1.0, 0.28],
            left=0.04,
            right=0.98,
            bottom=0.08,
            top=0.84,
            wspace=0.08,
        )

        ax_before = fig.add_subplot(gs[0, 0])
        ax_after = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[0, 2])
        cax = None

    ax_legend.axis("off")

    #
    # gs = fig.add_gridspec(
    #     1,
    #     4,
    #     width_ratios=[1.0, 1.0, 0.05, 0.28],
    #     left=0.04,
    #     right=0.98,
    #     bottom=0.08,
    #     top=0.84,
    #     wspace=0.08,
    # )
    # ax_before = fig.add_subplot(gs[0, 0])
    # ax_after = fig.add_subplot(gs[0, 1])
    # cax = fig.add_subplot(gs[0, 2])
    # ax_legend = fig.add_subplot(gs[0, 3])
    # ax_legend.axis("off")

    legend_handles = []

    # Plot each sample separately, using a distinct colormap.
    for sid in sample_ids:
        cmap_name = sample_cmap_map[sid]
        values = slices[sid].values

        native_xy = slices[sid].coords
        native_valid = _finite_xy_mask(native_xy) & np.isfinite(values)
        native_xy_valid = native_xy[native_valid]
        native_values_valid = values[native_valid]

        mapped_xy = aligned_coords[sid]
        mapped_valid = _finite_xy_mask(mapped_xy) & np.isfinite(values)
        mapped_xy_valid = mapped_xy[mapped_valid]
        mapped_values_valid = values[mapped_valid]

        before_rgba = gene_rgba(
            native_values_valid,
            norm=norm,
            cmap_name=cmap_name,
            alpha_max=alpha_max,
            alpha_gamma=alpha_gamma,
        )
        after_rgba = gene_rgba(
            mapped_values_valid,
            norm=norm,
            cmap_name=cmap_name,
            alpha_max=alpha_max,
            alpha_gamma=alpha_gamma,
        )

        ax_before.scatter(
            native_xy_valid[:, 0],
            native_xy_valid[:, 1],
            c=before_rgba,
            s=point_size,
            linewidths=0,
            rasterized=True,
        )

        ax_after.scatter(
            mapped_xy_valid[:, 0],
            mapped_xy_valid[:, 1],
            c=after_rgba,
            s=point_size,
            linewidths=0,
            rasterized=True,
        )

        # Legend handle: use the high-intensity color of the sample cmap.
        cmap = plt.get_cmap(cmap_name)
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=cmap(0.85),
                markeredgecolor="none",
                markersize=7,
                label=f"Sample {sid}",
            )
        )

    style_spatial_axis(
        ax_before,
        f"Before AXIS",
        invert_y=invert_y,
    )
    style_spatial_axis(
        ax_after,
        f"After AXIS",
        invert_y=invert_y,
    )

    # Shared limits
    all_before = [
        slices[sid].coords[
            _finite_xy_mask(slices[sid].coords) & np.isfinite(slices[sid].values)
            ]
        for sid in sample_ids
    ]
    all_after = [
        aligned_coords[sid][
            _finite_xy_mask(aligned_coords[sid]) & np.isfinite(slices[sid].values)
            ]
        for sid in sample_ids
    ]

    if share_before_after_limits:
        xlim, ylim = spatial_limits(all_before + all_after)
        apply_limits(ax_before, xlim, ylim, invert_y=invert_y)
        apply_limits(ax_after, xlim, ylim, invert_y=invert_y)
    else:
        bx, by = spatial_limits(all_before)
        axx, ayy = spatial_limits(all_after)
        apply_limits(ax_before, bx, by, invert_y=invert_y)
        apply_limits(ax_after, axx, ayy, invert_y=invert_y)

    # Colorbar: use one neutral colorbar to represent expression magnitude only.
    # sm = ScalarMappable(norm=norm, cmap=gene_cmap)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, cax=cax)
    # cbar.set_label(
    #     f"{gene} log1p(iSTAR signal)" if use_log1p else f"{gene} iSTAR signal",
    #     fontsize=10,
    # )
    # cbar.ax.tick_params(labelsize=8)

    ax_legend.legend(
        handles=legend_handles,
        loc="center left",
        frameon=False,
        title="Samples",
        fontsize=9,
        title_fontsize=10,
        borderaxespad=0.0,
        handletextpad=0.4,
        labelspacing=0.6,
    )
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, warp_results


# =============================================================================
# Convenience wrapper
# =============================================================================
def make_scc_istar_gene_figures(
    axis_root: str = "/media/huifang/data/registration/SCC/huifang",
    istar_root: str = "/media/huifang/data/registration/SCC/istar/single_input",
    patient_id: Any = 2,
    sample_ids: Sequence[Any] = (0, 1, 2),
    gene: str = "COL17A1",
    reference_id: Any = 0,
    figure1_sample_id: Any = 0,
    out_dir: Optional[str] = None,
    show: bool = True,
    use_cached_warp: bool = True,
    cache_dir: Optional[str] = None,
):
    fig1_path = None
    fig2_path = None
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig1_path = os.path.join(
            out_dir,
            f"patient{patient_id}_{gene}_sample{figure1_sample_id}_original_vs_istar.png",
        )
        fig2_path = os.path.join(
            out_dir,
            f"patient{patient_id}_{gene}_istar_before_after_axis.png",
        )
    #
    # fig1 = plot_original_vs_istar_sample0(
    #     axis_root=axis_root,
    #     istar_root=istar_root,
    #     patient_id=patient_id,
    #     gene=gene,
    #     sample_id=figure1_sample_id,
    #     save_path=fig1_path,
    #     show=show,
    # )

    plot_istar_gene_overlay_before_after_axis(
        axis_root="/media/huifang/data/registration/SCC/huifang",
        istar_root="/media/huifang/data/registration/SCC/istar/single_input",
        patient_id=2,
        sample_ids=(0, 1, 2),
        gene="COL17A1",
        reference_id=0,
        sample_cmaps=["Greens", "Blues", "Reds"],
        use_cached_warp=True,
        cache_dir=None,
        show=True,
    )


if __name__ == "__main__":
    make_scc_istar_gene_figures(
        axis_root="/media/huifang/data/registration/SCC/huifang",
        istar_root="/media/huifang/data/registration/SCC/istar/single_input",
        patient_id=2,
        sample_ids=(0, 1, 2),
        gene="COL17A1",
        reference_id=0,
        figure1_sample_id=0,
        out_dir=None,
        show=True,
        use_cached_warp=True,
        cache_dir=None,
    )
