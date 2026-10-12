"""Transition-field accuracy metrics (Methods, "Evaluation of tissue alignment and registration").

Anatomical correspondence is evaluated per matched domain in a common frame:

* **Dice** of alpha-shape domain masks;
* **spatial correlation** of Gaussian-smoothed domain density maps,
  ``S_c = <D_c^A, D_c^B> / (|D_c^A| |D_c^B| + eps)`` (plus Pearson);
* **domain-centre shift** between domain centroids.

Molecular-neighbourhood correspondence scores each transformed observation
by whether its label is the majority label of the reference observations
within a fixed radius.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

__all__ = [
    "RegionMeta",
    "filter_isolated_points",
    "build_region_metadata",
    "evaluate_domain_metrics",
    "evaluate_neighborhood_metrics",
    "merge_metric_tables",
]


@dataclass
class RegionMeta:
    domain: str
    coords: np.ndarray
    contours: list[np.ndarray]
    centroid_x: float
    centroid_y: float
    area: float
    mask: Optional[np.ndarray]


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


def _geometry_polygons(geometry: Any) -> list:
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

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
    from skimage.draw import polygon as rasterize_polygon

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
    from skimage.measure import find_contours

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
            points_for_boundary = filter_isolated_points(points, radius=neighbor_radius, min_neighbors=min_neighbors)
        else:
            points_for_boundary = points

        geometry = None
        if len(points_for_boundary) >= 3:
            import alphashape

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


def _natural_sort_key(value: Any) -> tuple:
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)
