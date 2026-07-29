import math
from dataclasses import dataclass
import alphashape
import scipy
from skimage.draw import polygon as rasterize_polygon
from typing import Any, Dict, List, Optional, Tuple
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Any, Optional, Tuple
from scipy.spatial import cKDTree
import inspect
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

INVERT_Y=True
CORR_GRID_SIZE = 100
CORR_SIGMA = 2.0

def filter_isolated_points(
    coords: np.ndarray,
    radius: Optional[float] = None,
    min_neighbors: int = 3,
    radius_scale: float = 1.8,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Remove isolated points based on local neighbor count.

    Parameters
    ----------
    coords : ndarray of shape [N, 2]
    radius : float or None
        Neighborhood radius. If None, estimate from local spacing.
    min_neighbors : int
        Minimum number of neighbors (excluding self) required to keep a point.
    radius_scale : float
        Multiplier for auto-estimated radius.

    Returns
    -------
    coords_kept : ndarray
    keep_mask : ndarray of bool
    radius_used : float
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]

    if n == 0:
        return coords, np.zeros(0, dtype=bool), np.nan
    if n <= min_neighbors + 1:
        return coords, np.ones(n, dtype=bool), np.nan

    tree = cKDTree(coords)

    if radius is None:
        # estimate local spacing from k-th nearest neighbor
        k = min(5, n)
        dists, _ = tree.query(coords, k=k)
        kth = dists[:, -1]
        kth = kth[np.isfinite(kth)]
        if len(kth) == 0:
            radius = 1.0
        else:
            radius = float(np.median(kth) * radius_scale)

    neighborhoods = tree.query_ball_point(coords, r=radius)
    counts = np.array([len(nb) - 1 for nb in neighborhoods], dtype=int)  # exclude self
    keep_mask = counts >= min_neighbors

    return coords[keep_mask], keep_mask, radius

# ============================================================
# Data containers
# ============================================================

@dataclass
class RegionMeta:
    label: Any
    coords: np.ndarray
    boundary: Optional[Any]
    mask: Optional[np.ndarray]
    centroid_x: float
    centroid_y: float
    area: int


# ============================================================
# Low-level helpers
# ============================================================

def _is_missing_label(x: Any) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return False


def _safe_centroid_from_mask(mask: Optional[np.ndarray]) -> Tuple[float, float]:
    if mask is None:
        return np.nan, np.nan
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.nan, np.nan
    return float(xs.mean()), float(ys.mean())


def _geometry_to_polygons(boundary: Any) -> List[np.ndarray]:
    if boundary is None:
        return []

    if isinstance(boundary, Polygon):
        return [np.asarray(boundary.exterior.coords)]

    if isinstance(boundary, MultiPolygon):
        return [np.asarray(poly.exterior.coords) for poly in boundary.geoms]

    if isinstance(boundary, GeometryCollection):
        polys = []
        for geom in boundary.geoms:
            if isinstance(geom, Polygon):
                polys.append(np.asarray(geom.exterior.coords))
        return polys

    return []


def boundary_to_mask(boundary: Any, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    polys = _geometry_to_polygons(boundary)
    if len(polys) == 0:
        return None

    mask = np.zeros(shape, dtype=bool)
    for poly in polys:
        rr, cc = rasterize_polygon(poly[:, 1], poly[:, 0], shape=shape)
        mask[rr, cc] = True
    return mask


# def create_boundary_mask_simple(
#     coords: np.ndarray,
#     shape: Tuple[int, int],
#     alpha: float = 0.5,
# ) -> Tuple[Optional[Any], Optional[np.ndarray]]:
#     coords = np.asarray(coords, dtype=float)
#     if coords.shape[0] < 3:
#         return None, None
#
#     try:
#         boundary = alphashape.alphashape(coords, alpha)
#     except Exception:
#         return None, None
#
#     mask = boundary_to_mask(boundary, shape)
#     return boundary, mask
from sklearn.neighbors import NearestNeighbors
from skimage.draw import polygon
def get_mask(boundary,shape):
    if isinstance(boundary, Polygon):
        boundary = [np.array(boundary.exterior.coords)]
    elif isinstance(boundary, MultiPolygon):
        boundary = [np.array(poly.exterior.coords) for poly in boundary.geoms]
    elif isinstance(boundary, GeometryCollection):
        boundary = [np.array(geom.exterior.coords) for geom in boundary.geoms if isinstance(geom, Polygon)]

    mask = np.zeros(shape, dtype=np.uint8)
    for poly in boundary:
        rr, cc = polygon(poly[:, 1], poly[:, 0], shape)
        mask[rr, cc] = 1  # Fill the region
    return mask
def create_boundary_mask_simple(coords, shape, alpha=0.5, neighbor_k=5, neighbor_thresh=0.05):
    if len(coords) < 3:
        return None, None

    coords = np.asarray(coords)

    # ---- Step 1: filter isolated points ----
    nbrs = NearestNeighbors(n_neighbors=min(neighbor_k, len(coords))).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    boundary = alphashape.alphashape(coords, alpha)
    mask = get_mask(boundary, shape)
    return boundary, mask

def create_boundary_mask(
    coords: np.ndarray,
    shape: Tuple[int, int],
    alpha: float = 0.5,
    remove_isolated: bool = True,
    neighbor_radius: Optional[float] = None,
    min_neighbors: int = 3,
    radius_scale: float = 1.8,
) -> Tuple[Optional[Any], Optional[np.ndarray]]:
    coords = np.asarray(coords, dtype=float)

    if coords.shape[0] < 3:
        return None, None

    if remove_isolated:
        coords, keep_mask, radius_used = filter_isolated_points(
            coords,
            radius=neighbor_radius,
            min_neighbors=min_neighbors,
            radius_scale=radius_scale,
        )

    if coords.shape[0] < 3:
        return None, None

    try:
        boundary = alphashape.alphashape(coords, alpha)
    except Exception:
        return None, None

    mask = boundary_to_mask(boundary, shape)
    return boundary, mask

def dice_coefficient(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]) -> float:
    if mask_a is None or mask_b is None:
        return np.nan

    inter = np.logical_and(mask_a, mask_b).sum()
    denom = mask_a.sum() + mask_b.sum()
    if denom == 0:
        return np.nan
    return float(2.0 * inter / denom)


def center_shift(
    cx_a: float,
    cy_a: float,
    cx_b: float,
    cy_b: float,
) -> float:
    if np.isnan(cx_a) or np.isnan(cy_a) or np.isnan(cx_b) or np.isnan(cy_b):
        return np.nan
    return float(math.hypot(cx_a - cx_b, cy_a - cy_b))


def normalize_to_grid(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    all_coords = np.vstack([coords_a, coords_b]).astype(float)
    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-8)

    def _norm(x):
        z = (x - mins) / span * (grid_size - 1)
        z = np.clip(np.round(z), 0, grid_size - 1)
        return z.astype(int)

    return _norm(coords_a), _norm(coords_b)


def density_map_for_label(
    coords: np.ndarray,
    labels: np.ndarray,
    label: Any,
    grid_size: int = 100,
    sigma: float = 2.0,
) -> np.ndarray:
    density = np.zeros((grid_size, grid_size), dtype=float)
    pts = coords[labels == label]
    for x, y in pts:
        density[y, x] += 1.0
    if sigma > 0:
        density = scipy.ndimage.gaussian_filter(density, sigma=sigma)
    return density


def normalized_spatial_correlation(
    density_a: np.ndarray,
    density_b: np.ndarray,
) -> float:
    num = float(np.sum(density_a * density_b))
    den = float(np.sqrt(np.sum(density_a ** 2) * np.sum(density_b ** 2)))
    if den == 0:
        return np.nan
    return num / (den + 1e-9)


# ============================================================
# Region metadata
# ============================================================

def build_region_metadata(
    coords: np.ndarray,
    labels: np.ndarray,
    mask_shape: Tuple[int, int],
    alpha: float = 0.01,
    remove_outlier=True,
    neighbor_radius: float = None
) -> Dict[Any, RegionMeta]:
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels)

    regions: Dict[Any, RegionMeta] = {}

    for label in np.unique(labels):
        if _is_missing_label(label):
            continue

        keep = labels == label
        region_coords = coords[keep]
        if region_coords.shape[0] == 0:
            continue
        if remove_outlier:
            boundary, mask = create_boundary_mask(region_coords, mask_shape, alpha=alpha,neighbor_radius=neighbor_radius)
        else:
            boundary, mask = create_boundary_mask_simple(region_coords, mask_shape, alpha=alpha)
        cx, cy = _safe_centroid_from_mask(mask)
        area = int(mask.sum()) if mask is not None else 0

        regions[label] = RegionMeta(
            label=label,
            coords=region_coords,
            boundary=boundary,
            mask=mask,
            centroid_x=cx,
            centroid_y=cy,
            area=area,
        )

    return regions


# ============================================================
# Unified domain evaluation
# ============================================================

def evaluate_domain_metrics(
    coords_a: np.ndarray,
    labels_a: np.ndarray,
    coords_b: np.ndarray,
    labels_b: np.ndarray,
    mask_shape: Tuple[int, int] = (100, 100),
    alpha: float = 0.01,
    corr_grid_size: int = 100,
    corr_sigma: float = 2.0,
    common_labels=None,
    remove_outlier=True,
    neighbor_radius=None
) -> Tuple[pd.DataFrame, Dict[Any, RegionMeta], Dict[Any, RegionMeta], Dict[Any, np.ndarray], Dict[Any, np.ndarray]]:
    """
    Returns
    -------
    metrics_df:
        one row per common domain with columns:
        label, dice, center_shift, spatial_corr, area_a, area_b
    region_a, region_b:
        metadata dictionaries
    density_a, density_b:
        per-label density maps for optional visualization
    """
    region_a = build_region_metadata(coords_a, labels_a, mask_shape=mask_shape, alpha=alpha, remove_outlier=remove_outlier, neighbor_radius=neighbor_radius)
    region_b = build_region_metadata(coords_b, labels_b, mask_shape=mask_shape, alpha=alpha, remove_outlier=remove_outlier,neighbor_radius=neighbor_radius)
    if not common_labels:
        common_labels = sorted(set(region_a.keys()) & set(region_b.keys()))
    if len(common_labels) == 0:
        empty = pd.DataFrame(columns=[
            "label", "dice", "center_shift", "spatial_corr", "area_a", "area_b"
        ])
        return empty, region_a, region_b, {}, {}


    norm_a, norm_b = normalize_to_grid(coords_a, coords_b, grid_size=corr_grid_size)

    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    density_maps_a: Dict[Any, np.ndarray] = {}
    density_maps_b: Dict[Any, np.ndarray] = {}
    rows = []

    for label in common_labels:
        ra = region_a[label]
        rb = region_b[label]

        dice = dice_coefficient(ra.mask, rb.mask)
        shift = center_shift(ra.centroid_x, ra.centroid_y, rb.centroid_x, rb.centroid_y)

        da = density_map_for_label(norm_a, labels_a, label, grid_size=corr_grid_size, sigma=corr_sigma)
        db = density_map_for_label(norm_b, labels_b, label, grid_size=corr_grid_size, sigma=corr_sigma)
        corr = normalized_spatial_correlation(da, db)

        density_maps_a[label] = da
        density_maps_b[label] = db

        rows.append({
            "label": label,
            "dice": dice,
            "center_shift": shift,
            "spatial_corr": corr,
            "area_a": ra.area,
            "area_b": rb.area,
            "n_points_a": ra.coords.shape[0],
            "n_points_b": rb.coords.shape[0],
        })

    metrics_df = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    return metrics_df, region_a, region_b, density_maps_a, density_maps_b


def aggregate_domain_metrics(
    metrics_df: pd.DataFrame,
    weighted: bool = False,
    weight_col: str = "area_a",
) -> Dict[str, float]:
    if metrics_df.empty:
        return {
            "dice": np.nan,
            "center_shift": np.nan,
            "spatial_corr": np.nan,
            "n_domains": 0,
        }

    out = {"n_domains": int(len(metrics_df))}

    metric_cols = ["dice", "center_shift", "spatial_corr"]

    if weighted:
        w = metrics_df[weight_col].to_numpy(dtype=float)
        w = np.where(np.isfinite(w), w, 0.0)
        w_sum = w.sum()
        if w_sum <= 0:
            weighted = False

    for col in metric_cols:
        x = metrics_df[col].to_numpy(dtype=float)
        valid = np.isfinite(x)

        if not valid.any():
            out[col] = np.nan
            continue

        if weighted:
            ww = w[valid]
            xx = x[valid]
            out[col] = float(np.average(xx, weights=ww))
        else:
            out[col] = float(np.nanmean(x))

    return out


def call_domain_evaluator(
    coords_fixed: np.ndarray,
    labels_fixed: np.ndarray,
    coords_moving: np.ndarray,
    labels_moving: np.ndarray,
    alpha: float,
    mask_shape: Tuple[int, int],
    common_labels=None,
    neighbor_radius=None
) -> pd.DataFrame:
    """
    Call the user's unified evaluator in a robust way.

    Expected evaluator behavior:
      - returns a DataFrame, or
      - returns a tuple/list whose first element is that DataFrame
    """
    sig = inspect.signature(evaluate_domain_metrics)

    kwargs = {}
    for k, v in {
        "mask_shape": mask_shape,
        "alpha": alpha,
        "corr_grid_size": CORR_GRID_SIZE,
        "corr_sigma": CORR_SIGMA,
        "common_labels": common_labels,
        "neighbor_radius":neighbor_radius
    }.items():
        if k in sig.parameters:
            kwargs[k] = v

    result = evaluate_domain_metrics(
        coords_fixed,
        labels_fixed,
        coords_moving,
        labels_moving,
        **kwargs,
    )

    if isinstance(result, pd.DataFrame):
        metrics_df = result
    elif isinstance(result, (tuple, list)) and len(result) > 0 and isinstance(result[0], pd.DataFrame):
        metrics_df = result[0]
    else:
        raise TypeError(
            "evaluate_domain_metrics(...) returned an unsupported object. "
            "Expected a DataFrame or a tuple/list whose first element is a DataFrame."
        )

    return standardize_metrics_df(metrics_df)



def standardize_metrics_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept evaluator outputs that use either 'label' or 'domain'.
    Return only the required columns for the CSV.
    """
    if not isinstance(metrics_df, pd.DataFrame):
        raise TypeError(f"Expected evaluator to return a DataFrame, got {type(metrics_df)}")

    df = metrics_df.copy()

    if "domain" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "domain"})

    required = ["domain", "dice", "spatial_corr", "center_shift"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Evaluator output missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[required].copy()
    df["domain"] = df["domain"].astype(str)
    return df

def standardize_metrics_df_molecular(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize molecular metric output to a stable plotting/export schema.

    Output keeps:
        domain
        spot_topk

    Extra informative columns are preserved if present.
    """
    df = metrics_df.copy()

    if "domain" not in df.columns and "cluster" in df.columns:
        df = df.rename(columns={"cluster": "domain"})

    required = ["domain", "spot_topk"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"metrics_df is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    keep_cols = ["domain", "spot_topk"]
    optional_cols = [
        "match_fraction",
        "n_valid_spots",
        "n_total_spots",
        "mean_neighbor_count",
        "cluster",
    ]
    keep_cols.extend([c for c in optional_cols if c in df.columns])

    out = df[keep_cols].copy()
    out["domain"] = out["domain"].astype(str)

    if "cluster" not in out.columns:
        out["cluster"] = out["domain"]

    return out

def _metric_row_lookup(metrics_df, label_col="label"):
    """
    Build a dict: label -> row dict
    Supports either 'label' or 'domain' column.
    """
    df = metrics_df.copy()

    if label_col not in df.columns:
        if "domain" in df.columns:
            label_col = "domain"
        elif "label" in df.columns:
            label_col = "label"
        else:
            raise KeyError(
                f"metrics_df must contain either 'label' or 'domain'. "
                f"Available columns: {list(df.columns)}"
            )

    lookup = {}
    for _, row in df.iterrows():
        lookup[row[label_col]] = row.to_dict()
    return lookup


def plot_boundary(boundary, color="red"):
    if boundary is None:
        return

    if isinstance(boundary, Polygon):
        x, y = boundary.exterior.xy
        plt.plot(x, y, color=color, linestyle="--", linewidth=2, alpha=1)

    elif isinstance(boundary, MultiPolygon):
        for poly in boundary.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, color=color, linestyle="--", linewidth=2, alpha=1)

    elif isinstance(boundary, GeometryCollection):
        for geom in boundary.geoms:
            if isinstance(geom, Polygon):
                x, y = geom.exterior.xy
                plt.plot(x, y, color=color, linestyle="--", linewidth=2, alpha=1)

    else:
        raise TypeError(f"Unsupported geometry type: {type(boundary)}")


def visualize_dice_regions(region_A, region_B, metrics_df, average_dice_score, save_suffix, outdir, fontsize=20):
    """
    Overlay regions from slice A and B and annotate per-domain Dice scores.

    region_A, region_B:
        dict[label] -> RegionMeta
    metrics_df:
        DataFrame with columns including label/domain and dice
    """
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({"font.size": fontsize})

    common_labels = sorted(set(region_A.keys()) & set(region_B.keys()))
    colors = sns.color_palette("tab10", n_colors=len(common_labels))
    metric_lookup = _metric_row_lookup(metrics_df)

    for idx, label in enumerate(common_labels):
        if label not in metric_lookup:
            continue

        ra = region_A[label]
        rb = region_B[label]
        row = metric_lookup[label]

        color = colors[idx]

        # Scatter plot of original points
        plt.scatter(
            ra.coords[:, 0], ra.coords[:, 1],
            color=color, alpha=0.5, s=10,
            label=f"Region {label} (A)"
        )
        plt.scatter(
            rb.coords[:, 0], rb.coords[:, 1],
            color=color, alpha=0.5, s=10, marker="x",
            label=f"Region {label} (B)"
        )

        # Overlay boundaries
        plot_boundary(ra.boundary, color)
        plot_boundary(rb.boundary, color)

        midx = (ra.centroid_x + rb.centroid_x) / 2
        midy = (ra.centroid_y + rb.centroid_y) / 2

        dice_value = row["dice"]
        if pd.notna(dice_value):
            plt.text(
                midx,
                midy,
                f"{float(dice_value):.2f}",
                ha="center",
                color="black",
                bbox=dict(facecolor="white", alpha=0.5),
            )

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    if INVERT_Y:
        plt.gca().invert_yaxis()
    plt.title(f"{average_dice_score:.3f}" if pd.notna(average_dice_score) else "nan")
    plt.axis("equal")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.grid(False)
    plt.tight_layout()

    if outdir:
        plt.savefig(f"{outdir}dice_{save_suffix}.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_centroid_shift(region_A, region_B, metrics_df, region_mean_shift, save_suffix, outdir):
    """
    Overlay regions from slice A and B and annotate per-domain centroid shift.
    """
    plt.figure(figsize=(12, 9))
    common_labels = sorted(set(region_A.keys()) & set(region_B.keys()))
    colors = sns.color_palette("tab10", n_colors=len(common_labels))
    metric_lookup = _metric_row_lookup(metrics_df)

    for idx, label in enumerate(common_labels):
        if label not in metric_lookup:
            continue

        ra = region_A[label]
        rb = region_B[label]
        row = metric_lookup[label]

        color = colors[idx]

        # Plot points
        plt.scatter(ra.coords[:, 0], ra.coords[:, 1], color=color, marker="o", s=0.1, alpha=0.6)
        plt.scatter(rb.coords[:, 0], rb.coords[:, 1], color=color, marker="s", s=0.1, alpha=0.6)

        # Plot centroids
        plt.scatter(
            ra.centroid_x, ra.centroid_y,
            edgecolor="black", color=color, marker="o", s=20,
            label=f"Region {label} - Centroid A"
        )
        plt.scatter(
            rb.centroid_x, rb.centroid_y,
            edgecolor="black", color=color, marker="s", s=20,
            label=f"Region {label} - Centroid B"
        )

        if np.isfinite(ra.centroid_x) and np.isfinite(ra.centroid_y) and np.isfinite(rb.centroid_x) and np.isfinite(rb.centroid_y):
            plt.arrow(
                ra.centroid_x,
                ra.centroid_y,
                rb.centroid_x - ra.centroid_x,
                rb.centroid_y - ra.centroid_y,
                color=color,
                width=0.1,
                head_width=0.1,
                alpha=0.8,
            )

        shift_value = row["center_shift"]
        if pd.notna(shift_value):
            mid_x = (ra.centroid_x + rb.centroid_x) / 2
            mid_y = (ra.centroid_y + rb.centroid_y) / 2
            plt.text(
                mid_x,
                mid_y,
                f"{float(shift_value):.2f}",
                ha="center",
                color="black",
            )

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis("equal")
    if INVERT_Y:
        plt.gca().invert_yaxis()
    plt.title(f"{region_mean_shift:.3f}" if pd.notna(region_mean_shift) else "nan")
    # plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=16)
    plt.grid(False)
    plt.tight_layout()

    if outdir:
        plt.savefig(f"{outdir}centroid_shift_{save_suffix}.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_metric_scores(region_a, region_b, metrics_df, save_suffix, outdir=None):
    """
    Generate both Dice and centroid-shift overlays from one evaluator output.
    """
    summary = aggregate_domain_metrics(metrics_df, weighted=False)

    visualize_dice_regions(
        region_A=region_a,
        region_B=region_b,
        metrics_df=metrics_df,
        average_dice_score=summary["dice"],
        save_suffix=save_suffix,
        outdir=outdir,
    )

    visualize_centroid_shift(
        region_A=region_a,
        region_B=region_b,
        metrics_df=metrics_df,
        region_mean_shift=summary["center_shift"],
        save_suffix=save_suffix,
        outdir=outdir,
    )






from pathlib import Path

def spotwise_topk_details(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    labels_A: np.ndarray,
    labels_B: np.ndarray,
    dist_thresh: float = 5.0,
) -> pd.DataFrame:
    """
    Compute per-spot top-k/majority consistency for each spot in B.

    For each spot in B:
      - collect all neighbors in A within radius = dist_thresh
      - majority vote their labels
      - spot_topk = 1 if majority label matches the B label, else 0
      - match_fraction = fraction of neighbors in A with the same label as the B spot

    Returns a per-spot DataFrame.
    """
    nA = min(coords_A.shape[0], len(labels_A))
    nB = min(coords_B.shape[0], len(labels_B))
    coords_A, labels_A = coords_A[:nA], labels_A[:nA]
    coords_B, labels_B = coords_B[:nB], labels_B[:nB]

    nn = NearestNeighbors(radius=dist_thresh, algorithm="auto").fit(coords_A)
    neighbors = nn.radius_neighbors(coords_B, return_distance=False)

    rows = []
    for i in range(nB):
        idxs = neighbors[i]
        true_label = labels_B[i]

        if len(idxs) == 0:
            rows.append(
                {
                    "spot_index_B": i,
                    "domain": true_label,
                    "neighbor_count": 0,
                    "majority_label": None,
                    "spot_topk": np.nan,
                    "match_fraction": np.nan,
                }
            )
            continue

        neigh_labels = labels_A[idxs]
        vals, counts = np.unique(neigh_labels, return_counts=True)
        maj_label = vals[np.argmax(counts)]

        rows.append(
            {
                "spot_index_B": i,
                "domain": true_label,
                "neighbor_count": int(len(idxs)),
                "majority_label": maj_label,
                "spot_topk": float(maj_label == true_label),
                "match_fraction": float(np.mean(neigh_labels == true_label)),
            }
        )

    return pd.DataFrame(rows)


def spotwise_topk_accuracy(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    labels_A: np.ndarray,
    labels_B: np.ndarray,
    dist_thresh: float = 5.0,
) -> float:
    """
    Global average spot_topk across valid spots.
    """
    per_spot_df = spotwise_topk_details(
        coords_A=coords_A,
        coords_B=coords_B,
        labels_A=labels_A,
        labels_B=labels_B,
        dist_thresh=dist_thresh,
    )
    return float(per_spot_df["spot_topk"].mean())


def evaluate_molecular_metrics(
    coords_a: np.ndarray,
    labels_a: np.ndarray,
    coords_b: np.ndarray,
    labels_b: np.ndarray,
    dist_thresh: float = 5.0,
    min_valid_spots: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster-based molecular evaluation.

    Returns
    -------
    metrics_df:
        one row per cluster/domain in slice B
        columns:
            domain
            cluster
            spot_topk
            match_fraction
            n_valid_spots
            n_total_spots
            mean_neighbor_count

    per_spot_df:
        one row per spot in slice B
    """
    per_spot_df = spotwise_topk_details(
        coords_A=coords_a,
        coords_B=coords_b,
        labels_A=labels_a,
        labels_B=labels_b,
        dist_thresh=dist_thresh,
    )

    grouped = (
        per_spot_df.groupby("domain", as_index=False)
        .agg(
            spot_topk=("spot_topk", "mean"),
            match_fraction=("match_fraction", "mean"),
            n_valid_spots=("spot_topk", lambda s: int(s.notna().sum())),
            n_total_spots=("spot_topk", "size"),
            mean_neighbor_count=("neighbor_count", "mean"),
        )
        .sort_values("domain")
        .reset_index(drop=True)
    )

    if min_valid_spots > 1:
        grouped = grouped[grouped["n_valid_spots"] >= min_valid_spots].copy()

    grouped["cluster"] = grouped["domain"].astype(str)
    grouped["domain"] = grouped["domain"].astype(str)

    return grouped, per_spot_df


def _build_label_color_map(labels_A: np.ndarray, labels_B: np.ndarray, label_cmap="tab10"):
    all_labels = pd.Index(np.concatenate([labels_A, labels_B])).unique().tolist()
    label_to_code = {lab: i for i, lab in enumerate(all_labels)}


    cmap = plt.get_cmap(label_cmap, max(len(all_labels), 1))

    codes_A = np.array([label_to_code[x] for x in labels_A], dtype=int)
    codes_B = np.array([label_to_code[x] for x in labels_B], dtype=int)

    return cmap, label_to_code, codes_A, codes_B


def plot_registration_topk_visual(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    labels_A: np.ndarray,
    labels_B: np.ndarray,
    spot_index_B: int = 200,
    dist_thresh: float = 5.0,
    label_cmap="tab10",
    method_name: Optional[str] = None,
    save_suffix: Optional[str] = None,
    outdir: Optional[Path] = None,
):
    """
    Visualize one selected spot in B and its neighborhood in A.
    Assumes coords are already normalized to the same frame used by evaluation.
    """
    nA = min(coords_A.shape[0], len(labels_A))
    nB = min(coords_B.shape[0], len(labels_B))
    coords_A, labels_A = coords_A[:nA], labels_A[:nA]
    coords_B, labels_B = coords_B[:nB], labels_B[:nB]

    if not (0 <= spot_index_B < nB):
        raise IndexError(f"spot_index_B={spot_index_B} out of range for {nB} spots.")

    nn = NearestNeighbors(radius=dist_thresh).fit(coords_A)
    neighbors = nn.radius_neighbors(coords_B, return_distance=False)
    idxs = neighbors[spot_index_B]

    spot_B = coords_B[spot_index_B]
    true_label = labels_B[spot_index_B]

    cmap, label_to_code, codes_A, codes_B = _build_label_color_map(labels_A, labels_B, label_cmap=label_cmap)

    fig, (ax_zoom, ax_main) = plt.subplots(
        1, 2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [1, 2]},
    )

    # full slices
    scA = ax_main.scatter(
        coords_A[:, 0],
        coords_A[:, 1],
        s=30,
        c=codes_A,
        marker="X",
        cmap=cmap,
        alpha=0.3,
        label="Slice A (reference)",
    )
    ax_main.scatter(
        coords_B[:, 0],
        coords_B[:, 1],
        s=30,
        c=codes_B,
        marker="o",
        cmap=cmap,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.6,
        label="Slice B (moving)",
    )

    ax_main.set_title("Cross-slice registration: all spots + labels")
    ax_main.set_aspect("equal")
    ax_main.legend(loc="lower left", frameon=True, fontsize=12)

    margin = dist_thresh * 1.5
    zoom_rect = Rectangle(
        (spot_B[0] - margin, spot_B[1] - margin),
        2 * margin,
        2 * margin,
        linewidth=1.2,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
        alpha=0.8,
    )
    ax_main.add_patch(zoom_rect)

    # zoom panel background
    ax_zoom.scatter(
        coords_A[:, 0],
        coords_A[:, 1],
        s=180,
        c=codes_A,
        marker="X",
        cmap=cmap,
        alpha=0.3,
        label="Points in Slice A",
    )
    ax_zoom.scatter(
        coords_B[:, 0],
        coords_B[:, 1],
        s=180,
        c=codes_B,
        marker="o",
        cmap=cmap,
        alpha=0.6,
        edgecolor="none",
        label="Points in Slice B",
    )

    true_count = 0
    for ni in idxs:
        coord_A = coords_A[ni]
        label_A = labels_A[ni]
        is_match = label_A == true_label
        line_color = "green" if is_match else "red"
        if is_match:
            true_count += 1

        ax_zoom.plot(
            [spot_B[0], coord_A[0]],
            [spot_B[1], coord_A[1]],
            c=line_color,
            linewidth=3.0,
            alpha=0.9,
            zorder=2,
        )

        ax_zoom.scatter(
            coord_A[0],
            coord_A[1],
            s=180,
            c=[cmap(label_to_code[label_A])],
            marker="X",
            edgecolor="black",
            linewidth=0.8,
            alpha=1.0,
            zorder=3,
        )

    acc = np.nan if len(idxs) == 0 else true_count / len(idxs)
    title = f"Acc={acc:.3f}" if np.isfinite(acc) else "Acc=nan"
    if method_name is not None:
        title = f"{method_name} | {title}"
    ax_zoom.set_title(title, fontsize=20)

    ax_zoom.scatter(
        spot_B[0],
        spot_B[1],
        s=180,
        c=[cmap(label_to_code[true_label])],
        marker="o",
        edgecolor="black",
        linewidth=1.0,
        alpha=1.0,
        label="Selected spot (B)",
        zorder=4,
    )

    circ = Circle(
        (spot_B[0], spot_B[1]),
        dist_thresh,
        color="blue",
        fill=False,
        linestyle="--",
        alpha=0.6,
    )
    ax_zoom.add_patch(circ)

    ax_zoom.set_xlim(spot_B[0] - margin, spot_B[0] + margin)
    ax_zoom.set_ylim(spot_B[1] - margin, spot_B[1] + margin)
    ax_zoom.set_aspect("equal")

    match_line = Line2D([0], [0], color="green", lw=1.5, label="Label match")
    mismatch_line = Line2D([0], [0], color="red", lw=1.5, label="Label mismatch")

    ax_zoom.legend(
        handles=[match_line, mismatch_line],
        loc="upper left",
        frameon=True,
        fontsize=12,
    )

    for ax in [ax_zoom, ax_main]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.invert_yaxis()

    plt.tight_layout()

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fname = f"spot_neighbor_{save_suffix}.png" if save_suffix else "spot_neighbor.png"
        plt.savefig(outdir / fname, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
