from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import alphashape
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from skimage.draw import polygon as rasterize_polygon
from skimage.measure import find_contours

from utils_io import (
    NormalizationMeta,
    case_metadata_path,
    load_case_metadata_npz,
    save_case_metadata_npz,
)


# =============================================================================
# Data containers
# =============================================================================
@dataclass
class RegionMeta:
    domain: str
    coords: np.ndarray
    contours: list[np.ndarray]
    centroid_x: float
    centroid_y: float
    area: float
    mask: Optional[np.ndarray]


@dataclass
class PairEvaluationResult:
    metrics: pd.DataFrame
    per_spot: pd.DataFrame
    coords_fixed: np.ndarray
    coords_moving: np.ndarray
    labels_fixed: np.ndarray
    labels_moving: np.ndarray
    regions_fixed: Dict[str, RegionMeta]
    regions_moving: Dict[str, RegionMeta]
    normalization: NormalizationMeta


# =============================================================================
# Region construction
# =============================================================================
def _string_labels(labels: np.ndarray) -> np.ndarray:
    return np.asarray(labels).reshape(-1).astype(str)


def filter_isolated_points(
    coords: np.ndarray,
    radius: Optional[float],
    min_neighbors: int = 3,
    radius_scale: float = 1.8,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) <= min_neighbors:
        return coords
    tree = cKDTree(coords)
    if radius is None:
        k = min(5, len(coords))
        dists, _ = tree.query(coords, k=k)
        radius = float(np.median(dists[:, -1]) * radius_scale)
    counts = np.asarray([len(x) - 1 for x in tree.query_ball_point(coords, radius)])
    return coords[counts >= min_neighbors]


def _geometry_polygons(geometry: Any) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [g for g in geometry.geoms if isinstance(g, Polygon)]
    return []


def _geometry_to_mask(
    geometry: Any,
    shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    polygons = _geometry_polygons(geometry)
    if not polygons:
        return None
    mask = np.zeros(shape, dtype=bool)
    for poly in polygons:
        xy = np.asarray(poly.exterior.coords, dtype=float)
        rr, cc = rasterize_polygon(xy[:, 1], xy[:, 0], shape=shape)
        mask[rr, cc] = True
        for interior in poly.interiors:
            hole = np.asarray(interior.coords, dtype=float)
            rr, cc = rasterize_polygon(hole[:, 1], hole[:, 0], shape=shape)
            mask[rr, cc] = False
    return mask


def _mask_contours(mask: Optional[np.ndarray]) -> list[np.ndarray]:
    if mask is None or not np.any(mask):
        return []
    output = []
    for contour_rc in find_contours(mask.astype(float), level=0.5):
        output.append(contour_rc[:, [1, 0]].astype(np.float32))
    return output




def build_region_metadata(
    coords: np.ndarray,
    labels: np.ndarray,
    mask_shape: Tuple[int, int],
    alpha: float,
    common_domains: Sequence[str],
    remove_outliers: bool,
    neighbor_radius: Optional[float],
    min_neighbors: int = 3,
) -> Dict[str, RegionMeta]:
    coords = np.asarray(coords, dtype=np.float64)
    labels = _string_labels(labels)
    regions: Dict[str, RegionMeta] = {}

    for domain in common_domains:
        points = coords[labels == domain]
        if remove_outliers:
            points_for_boundary = filter_isolated_points(
                points, radius=neighbor_radius, min_neighbors=min_neighbors
            )
        else:
            points_for_boundary = points

        geometry = None
        if len(points_for_boundary) >= 3:
            geometry = alphashape.alphashape(points_for_boundary, alpha)
        mask = _geometry_to_mask(geometry, mask_shape)
        contours = _mask_contours(mask)

        polygons = _geometry_polygons(geometry)
        if polygons:
            union = geometry
            centroid_x = float(union.centroid.x)
            centroid_y = float(union.centroid.y)
            area = float(union.area)
        else:
            centroid_x = np.nan
            centroid_y = np.nan
            area = 0.0

        regions[domain] = RegionMeta(
            domain=domain,
            coords=points.astype(np.float32),
            contours=contours,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            area=area,
            mask=mask,
        )
    return regions


# =============================================================================
# Domain metrics
# =============================================================================
def _dice(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]) -> float:
    if mask_a is None or mask_b is None:
        return np.nan
    denominator = int(mask_a.sum() + mask_b.sum())
    if denominator == 0:
        return np.nan
    return float(2 * np.logical_and(mask_a, mask_b).sum() / denominator)


def _center_shift(a: RegionMeta, b: RegionMeta) -> float:
    values = [a.centroid_x, a.centroid_y, b.centroid_x, b.centroid_y]
    if not np.all(np.isfinite(values)):
        return np.nan
    return float(math.hypot(a.centroid_x - b.centroid_x, a.centroid_y - b.centroid_y))


def _density_grid_shape(mask_shape: Tuple[int, int], max_dim: int) -> Tuple[int, int]:
    height, width = mask_shape
    if height >= width:
        return max_dim, max(8, int(round(max_dim * width / height)))
    return max(8, int(round(max_dim * height / width))), max_dim


def _density_map(
    coords: np.ndarray,
    mask_shape: Tuple[int, int],
    grid_shape: Tuple[int, int],
    sigma: float,
) -> np.ndarray:
    height, width = mask_shape
    hist, _, _ = np.histogram2d(
        coords[:, 1],
        coords[:, 0],
        bins=grid_shape,
        range=((0, height), (0, width)),
    )
    return ndimage.gaussian_filter(hist, sigma=sigma) if sigma > 0 else hist


def _density_similarity(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    support = (a > 1e-12) | (b > 1e-12)
    if support.sum() < 3:
        return np.nan, np.nan
    x = a[support].astype(float)
    y = b[support].astype(float)

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    den_pearson = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    pearson = float(np.sum(x_centered * y_centered) / den_pearson) if den_pearson > 0 else np.nan

    den_cosine = np.sqrt(np.sum(x**2) * np.sum(y**2))
    cosine = float(np.sum(x * y) / den_cosine) if den_cosine > 0 else np.nan
    return pearson, cosine


def evaluate_domain_metrics(
    coords_fixed: np.ndarray,
    labels_fixed: np.ndarray,
    coords_moving: np.ndarray,
    labels_moving: np.ndarray,
    *,
    mask_shape: Tuple[int, int],
    alpha: float,
    corr_grid_size: int = 100,
    corr_sigma: float = 2.0,
    common_labels: Optional[Sequence[Any]] = None,
    remove_outliers: bool = True,
    neighbor_radius: Optional[float] = None,
    min_neighbors: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, RegionMeta], Dict[str, RegionMeta]]:
    """Calculate per-domain geometry and density metrics in one common frame."""

    labels_fixed = _string_labels(labels_fixed)
    labels_moving = _string_labels(labels_moving)
    shared = set(labels_fixed).intersection(labels_moving)
    if common_labels is None:
        domains = sorted(shared, key=_natural_sort_key)
    else:
        requested = [str(x) for x in common_labels]
        domains = [x for x in requested if x in shared]
    if not domains:
        raise ValueError("No common domains are available for evaluation")

    regions_fixed = build_region_metadata(
        coords_fixed,
        labels_fixed,
        mask_shape,
        alpha,
        domains,
        remove_outliers,
        neighbor_radius,
        min_neighbors,
    )
    regions_moving = build_region_metadata(
        coords_moving,
        labels_moving,
        mask_shape,
        alpha,
        domains,
        remove_outliers,
        neighbor_radius,
        min_neighbors,
    )

    grid_shape = _density_grid_shape(mask_shape, corr_grid_size)
    rows = []
    for domain in domains:
        fixed = regions_fixed[domain]
        moving = regions_moving[domain]
        density_fixed = _density_map(fixed.coords, mask_shape, grid_shape, corr_sigma)
        density_moving = _density_map(moving.coords, mask_shape, grid_shape, corr_sigma)
        spatial_corr, density_cosine = _density_similarity(density_fixed, density_moving)
        rows.append(
            {
                "domain": domain,
                "dice": _dice(fixed.mask, moving.mask),
                "spatial_corr": spatial_corr,
                "density_cosine": density_cosine,
                "center_shift": _center_shift(fixed, moving),
                "area_fixed": fixed.area,
                "area_moving": moving.area,
                "n_points_fixed": len(fixed.coords),
                "n_points_moving": len(moving.coords),
            }
        )
    return pd.DataFrame(rows), regions_fixed, regions_moving


# =============================================================================
# Local label agreement
# =============================================================================
def evaluate_neighborhood_metrics(
    coords_fixed: np.ndarray,
    labels_fixed: np.ndarray,
    coords_moving: np.ndarray,
    labels_moving: np.ndarray,
    *,
    distance_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate moving spots against fixed-slice labels within a radius.

    Unsupported spots receive NaN rather than being automatically scored as an
    error. ``supported_fraction`` reports their prevalence explicitly.
    """

    labels_fixed = _string_labels(labels_fixed)
    labels_moving = _string_labels(labels_moving)
    tree = cKDTree(np.asarray(coords_fixed, dtype=float))
    neighborhoods = tree.query_ball_point(np.asarray(coords_moving, dtype=float), distance_threshold)


    rows = []
    for index, neighbors in enumerate(neighborhoods):
        true_domain = labels_moving[index]

        if not neighbors:
            rows.append(
                {
                    "spot_index_moving": index,
                    "domain": true_domain,
                    "neighbor_count": 0,
                    "neighbor_accuracy": 0,
                    "match_fraction": 0,
                }
            )
            continue

        neighbor_labels = labels_fixed[np.asarray(neighbors, dtype=int)]
        values, counts = np.unique(neighbor_labels, return_counts=True)
        top_labels = set(values[counts == counts.max()])
        rows.append(
            {
                "spot_index_moving": index,
                "domain": true_domain,
                "neighbor_count": len(neighbors),
                "neighbor_accuracy": float(true_domain in top_labels),
                "match_fraction": float(np.mean(neighbor_labels == true_domain)),
            }
        )

    per_spot = pd.DataFrame(rows)
    grouped_rows = []
    for domain, part in per_spot.groupby("domain", sort=False):
        valid = part["neighbor_count"] > 0
        grouped_rows.append(
            {
                "domain": str(domain),
                "neighbor_accuracy": float(part.loc[valid, "neighbor_accuracy"].mean()) if valid.any() else np.nan,
                "match_fraction": float(part.loc[valid, "match_fraction"].mean()) if valid.any() else np.nan,
                "supported_fraction": float(valid.mean()),
                "n_valid_spots": int(valid.sum()),
                "n_total_spots": int(len(part)),
                "mean_neighbor_count": float(part.loc[valid, "neighbor_count"].mean()) if valid.any() else 0.0,
            }
        )
    # for row in grouped_rows:
    #     print(row)
    # test = input()
    return pd.DataFrame(grouped_rows), per_spot


def merge_metric_tables(domain_df: pd.DataFrame, neighborhood_df: pd.DataFrame) -> pd.DataFrame:
    left = domain_df.copy()
    right = neighborhood_df.copy()
    left["domain"] = left["domain"].astype(str)
    right["domain"] = right["domain"].astype(str)
    return left.merge(right, on="domain", how="left", validate="one_to_one")


# =============================================================================
# NumPy-only metadata transfer
# =============================================================================
def _pack_contours(
    regions: Mapping[str, RegionMeta],
    domains: Sequence[str],
) -> dict[str, np.ndarray]:
    points = []
    offsets = [0]
    domain_indices = []
    for domain_index, domain in enumerate(domains):
        for contour in regions[domain].contours:
            contour = np.asarray(contour, dtype=np.float32)
            if len(contour) < 2:
                continue
            points.append(contour)
            offsets.append(offsets[-1] + len(contour))
            domain_indices.append(domain_index)
    return {
        "contour_points": np.vstack(points).astype(np.float32) if points else np.empty((0, 2), dtype=np.float32),
        "contour_offsets": np.asarray(offsets, dtype=np.int64),
        "contour_domain_indices": np.asarray(domain_indices, dtype=np.int32),
    }


def build_case_metadata_payload(
    result: PairEvaluationResult,
    *,
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
) -> dict[str, np.ndarray]:
    domains = sorted(
        set(result.regions_fixed).intersection(result.regions_moving),
        key=_natural_sort_key,
    )
    fixed_pack = _pack_contours(result.regions_fixed, domains)
    moving_pack = _pack_contours(result.regions_moving, domains)

    payload = {
        "method": np.asarray([method], dtype="U64"),
        "group_id": np.asarray([group_id], dtype=np.int64),
        "fixed_id": np.asarray([fixed_id], dtype=np.int64),
        "moving_id": np.asarray([moving_id], dtype=np.int64),
        "coords_fixed": np.asarray(result.coords_fixed, dtype=np.float32),
        "coords_moving": np.asarray(result.coords_moving, dtype=np.float32),
        "labels_fixed": _string_labels(result.labels_fixed).astype("U64"),
        "labels_moving": _string_labels(result.labels_moving).astype("U64"),
        "domains": np.asarray(domains, dtype="U64"),
        "centroids_fixed": np.asarray(
            [[result.regions_fixed[d].centroid_x, result.regions_fixed[d].centroid_y] for d in domains],
            dtype=np.float32,
        ),
        "centroids_moving": np.asarray(
            [[result.regions_moving[d].centroid_x, result.regions_moving[d].centroid_y] for d in domains],
            dtype=np.float32,
        ),
        "areas_fixed": np.asarray([result.regions_fixed[d].area for d in domains], dtype=np.float32),
        "areas_moving": np.asarray([result.regions_moving[d].area for d in domains], dtype=np.float32),
        "normalization": result.normalization.as_array(),
        "fixed_contour_points": fixed_pack["contour_points"],
        "fixed_contour_offsets": fixed_pack["contour_offsets"],
        "fixed_contour_domain_indices": fixed_pack["contour_domain_indices"],
        "moving_contour_points": moving_pack["contour_points"],
        "moving_contour_offsets": moving_pack["contour_offsets"],
        "moving_contour_domain_indices": moving_pack["contour_domain_indices"],
    }
    return payload


# =============================================================================
# Collection
# =============================================================================
def collect_evaluation_results(
    *,
    methods: Sequence[str],
    group_ids: Sequence[int],
    pair_list: Sequence[Tuple[int, int]],
    evaluate_pair: Callable[[str, int, int, int], PairEvaluationResult],
    out_csv: Path,
    metadata_dir: Path,
    distance_name: Optional[str] = None,
    distance_fn: Optional[Callable[[int, int], float]] = None,
    strict: bool = True,
) -> pd.DataFrame:
    all_rows = []
    for method in methods:
        for group_id in group_ids:
            for fixed_id, moving_id in pair_list:
                print(f"[INFO] {method} | group={group_id} | pair={fixed_id}->{moving_id}")
                try:
                    result = evaluate_pair(method, group_id, fixed_id, moving_id)
                except Exception as exc:
                    message = f"Failed: {method}, group={group_id}, pair={fixed_id}->{moving_id}"
                    # message = f"Failed: {method}, group={group_id}, pair={fixed_id}->{moving_id}: {exc}"
                    if strict:
                        raise RuntimeError(message) from exc
                    print(f"[WARN] {message}")
                    continue

                metrics = result.metrics.copy()
                metrics.insert(0, "method", method)
                metrics.insert(0, "moving_id", int(moving_id))
                metrics.insert(0, "fixed_id", int(fixed_id))
                metrics.insert(0, "group_id", int(group_id))
                if distance_name is not None and distance_fn is not None:
                    metrics.insert(4, distance_name, float(distance_fn(fixed_id, moving_id)))
                all_rows.append(metrics)

                payload = build_case_metadata_payload(
                    result,
                    method=method,
                    group_id=group_id,
                    fixed_id=fixed_id,
                    moving_id=moving_id,
                )
                save_case_metadata_npz(
                    case_metadata_path(metadata_dir, method, group_id, fixed_id, moving_id),
                    payload,
                )


    if not all_rows:
        raise RuntimeError("No evaluation results were collected")
    output = pd.concat(all_rows, ignore_index=True)

    output["domain"] = output["domain"].astype(str)
    output = output.sort_values(
        ["method", "group_id", "fixed_id", "moving_id", "domain"]
    ).reset_index(drop=True)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_csv, index=False)
    print(f"[OK] metrics: {out_csv}")
    print(f"[OK] metadata: {metadata_dir}")
    return output


# =============================================================================
# Publication-style 2 x 4 case-study figure
# =============================================================================
def _natural_sort_key(value: Any) -> tuple:
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def _decode_contours(metadata: Mapping[str, np.ndarray], prefix: str):
    points = metadata[f"{prefix}_contour_points"]
    offsets = metadata[f"{prefix}_contour_offsets"]
    domain_indices = metadata[f"{prefix}_contour_domain_indices"]
    domains = metadata["domains"].astype(str)
    for index, domain_index in enumerate(domain_indices):
        start, end = int(offsets[index]), int(offsets[index + 1])
        yield domains[int(domain_index)], points[start:end]


def _aggregate_for_panel(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "dice": float(frame["dice"].mean()),
        "spatial_corr": float(frame["spatial_corr"].mean()),
        "center_shift": float(frame["center_shift"].mean()),
        "neighbor_accuracy": float(frame["neighbor_accuracy"].mean()),
        "n_domains": int(frame["domain"].nunique()),
    }

DOMAIN_COLORS = {
    "0": "#79706E",
    "1":  "#4E79A7",  # blue
    "2":  "#F28E2B",  # orange
    "3":  "#59A14F",  # green
    "4":  "#E15759",  # red
    "5":  "#B07AA1",  # purple
    "6":  "#76B7B2",  # teal
    "7":  "#EDC948",  # yellow
    "8":  "#FF9DA7",  # pink
    "9":  "#9C755F",  # brown
    "10": "#79706E",  # dark gray

    "11": "#A0CBE8",  # light blue
    "12": "#FFBE7D",  # light orange
    "13": "#8CD17D",  # light green
    "14": "#FF9D9A",  # light red
    "15": "#D4A6C8",  # light purple
    "16": "#86BCB6",  # light teal
    "17": "#F1CE63",  # light yellow
    "18": "#FABFD2",  # light pink
    "19": "#D7B5A6",  # light brown
    "20": "#BAB0AC",  # light gray
}

BACKGROUND_COLOR = "#D9D9D9"
def plot_case_study_grid(
    *,
    metrics_csv: Path,
    metadata_dir: Path,
    methods: Sequence[str],
    group_id: int,
    fixed_id: int,
    moving_id: int,
    save_path: Path,
    dataset_title: str,
    method_display_names: Optional[Mapping[str, str]] = None,
    domain_order: Optional[Sequence[Any]] = None,
    domain_cmap: str = "tab10",
    point_size: float = 5.0,
    point_alpha: float = 0.68,
    contour_width: float = 1.0,
    invert_y: bool = True,
    show_domain_legend: bool = True,
    dpi: int = 350,
) -> plt.Figure:
    """Render eight methods together from saved CSV and NPZ metadata only."""

    methods = list(methods)
    if len(methods) != 8:
        raise ValueError(f"The case-study layout requires exactly 8 methods, received {len(methods)}")

    metrics = pd.read_csv(metrics_csv, dtype={"domain": str})
    metrics = metrics[
        (metrics["group_id"] == group_id)
        & (metrics["fixed_id"] == fixed_id)
        & (metrics["moving_id"] == moving_id)
    ].copy()
    if metrics.empty:
        raise ValueError("No matching rows were found in the metrics CSV")

    metadata_by_method = {}
    available_domains = set()
    all_coords = []
    for method in methods:
        path = case_metadata_path(metadata_dir, method, group_id, fixed_id, moving_id)
        if path.exists():
            meta = load_case_metadata_npz(path)
            metadata_by_method[method] = meta
            available_domains.update(meta["domains"].astype(str).tolist())
            all_coords.extend([meta["coords_fixed"], meta["coords_moving"]])

    if domain_order is None:
        domains = sorted(available_domains, key=_natural_sort_key)
    else:
        domains = [str(x) for x in domain_order if str(x) in available_domains]
    if not domains:
        raise ValueError("No domains are available for plotting")

    color_map = {
        domain: DOMAIN_COLORS[str(domain)]
        for domain in domains
    }


    finite = np.vstack([x[np.all(np.isfinite(x), axis=1)] for x in all_coords])
    minimum = finite.min(axis=0)
    maximum = finite.max(axis=0)
    padding = np.maximum((maximum - minimum) * 0.03, 1.0)
    xlim = (minimum[0] - padding[0], maximum[0] + padding[0])
    ylim = (minimum[1] - padding[1], maximum[1] + padding[1])

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(2, 4, figsize=(11.6, 8.5), constrained_layout=True)
        axes = axes.ravel()

        for panel_index, (ax, method) in enumerate(zip(axes, methods)):
            ax.set_aspect("equal")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if invert_y:
                ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            display = method_display_names.get(method, method) if method_display_names else method
            ax.set_title(display, pad=4)
            meta = metadata_by_method.get(method)
            method_metrics = metrics[metrics["method"] == method]
            if meta is None or method_metrics.empty:
                ax.text(0.5, 0.5, "Not available", transform=ax.transAxes, ha="center", va="center")
                continue

            fixed_labels = meta["labels_fixed"].astype(str)
            moving_labels = meta["labels_moving"].astype(str)
            fixed_coords = meta["coords_fixed"]
            moving_coords = meta["coords_moving"]

            for domain in domains:
                color = color_map[domain]
                fixed_mask = fixed_labels == domain
                moving_mask = moving_labels == domain
                if np.any(fixed_mask):
                    ax.scatter(
                        fixed_coords[fixed_mask, 0],
                        fixed_coords[fixed_mask, 1],
                        s=point_size,
                        c=[color],
                        marker="o",
                        alpha=point_alpha,
                        linewidths=0,
                        rasterized=True,
                    )
                if np.any(moving_mask):
                    ax.scatter(
                        moving_coords[moving_mask, 0],
                        moving_coords[moving_mask, 1],
                        s=point_size,
                        c=[color],
                        marker="x",
                        alpha=min(1.0, point_alpha + 0.12),
                        linewidths=0.35,
                        rasterized=True,
                    )

            for domain, contour in _decode_contours(meta, "fixed"):
                if domain in color_map:
                    ax.plot(contour[:, 0], contour[:, 1], color=color_map[domain], lw=contour_width)
            for domain, contour in _decode_contours(meta, "moving"):
                if domain in color_map:
                    ax.plot(
                        contour[:, 0],
                        contour[:, 1],
                        color=color_map[domain],
                        lw=contour_width,
                        linestyle=(0, (3, 2)),
                    )

            summary = _aggregate_for_panel(method_metrics)
            metric_text = (
                f"Dice ↑  {summary['dice']:.3f}\n"
                f"Spatial r ↑  {summary['spatial_corr']:.3f}\n"
                f"Shift ↓  {summary['center_shift']:.1f}\n"
                f"Local ↑  {summary['neighbor_accuracy']:.3f}"
            )
            ax.text(
                0.02,
                0.02,
                metric_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.5,
                linespacing=1.25,
                bbox={"facecolor": "white", "edgecolor": "0.75", "linewidth": 0.5, "alpha": 0.9, "pad": 2.5},
            )

        slice_handles = [
            Line2D([0], [0], marker="o", linestyle="none", color="0.25", markersize=5, label="Fixed slice"),
            Line2D([0], [0], marker="x", linestyle="none", color="0.25", markersize=5, label="Moving slice"),
            Line2D([0], [0], color="0.25", lw=1.1, label="Fixed contour"),
            Line2D([0], [0], color="0.25", lw=1.1, linestyle=(0, (3, 2)), label="Moving contour"),
        ]
        fig.legend(
            handles=slice_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=4,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.4,
        )

        bottom = 0.08
        if show_domain_legend:
            domain_handles = [
                Line2D([0], [0], marker="o", linestyle="none", color=color_map[d], markersize=4, label=str(d))
                for d in domains
            ]
            ncol = min(10, len(domain_handles))
            fig.legend(
                handles=domain_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=ncol,
                frameon=False,
                title="Domain",
                fontsize=7,
                title_fontsize=8,
                handletextpad=0.25,
                columnspacing=0.8,
            )
            bottom = 0.12 if len(domains) <= 10 else 0.17

        fig.suptitle(
            f"{dataset_title} | group {group_id} | pair {fixed_id}→{moving_id}",
            y=0.99,
            fontsize=13,
            fontweight="bold",
        )
        fig.subplots_adjust(left=0.025, right=0.99, top=0.89, bottom=bottom, wspace=0.05, hspace=0.13)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] case study: {save_path}")
        return fig
