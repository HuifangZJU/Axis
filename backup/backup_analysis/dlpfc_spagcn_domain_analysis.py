#!/usr/bin/env python3
"""DLPFC spatial-domain analysis using SpaGCN.

Conditions
----------
* Single 0, Single 1, Single 2, Single 3
* AXIS-mean

Design
------
* Every condition is evaluated against its own ``obs[layer_key]`` labels.
* AXIS uses only the mean aggregate.
* One fixed HVG panel is selected from the native slices and reused for all
  conditions.
* SpaGCN coordinates are isotropically normalized by the median positive
  nearest-neighbor spacing, making the graph robust to coordinate ranges such
  as 500 x 500 versus 5000 x 5000.
* Original coordinates are retained for plotting and boundary evaluation.
* Each condition is evaluated once with a fixed SpaGCN seed.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from anndata import AnnData
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

try:
    import SpaGCN as spg
except ImportError as exc:
    raise ImportError(
        "SpaGCN is required. Install it in the active environment before "
        "running this script, for example with `pip install SpaGCN`."
    ) from exc

from spatial_arithmetic_utils import aggregate_virtual_slices


# =============================================================================
# USER CONFIGURATION
# =============================================================================

RUN_MODE = "both"  # "eval", "plot", or "both"


@dataclass(frozen=True)
class Config:
    run_mode: str = RUN_MODE

    biological_axis_name: str = "dlpfc_z"
    group_id: int = 1
    grid_size: int = 80
    spatial_bandwidth: int = 1
    reference_axis_value: float = 0.0

    root: str = "/media/huifang/data/registration/DLPFC/huifang"
    h5ad_dir: str = "/media/huifang/data/registration/DLPFC/huifang/h5ad"
    sample_ids: Tuple[int, ...] = (0, 1, 2, 3)

    layer_key: str = "layer_guess_reordered"
    native_spatial_key: str = "spatial"
    axis_spatial_key: str = "spatial"

    # AXIS-mean aggregate only.
    rebuild_axis_aggregate: bool = False
    aggregation_output_subdir: str = "aggregation_domain"
    aggregation_gene_chunk_size: int = 256

    # Fixed feature panel.
    n_top_genes: int = 2000
    min_cells_per_gene: int = 10

    # Domain count. None -> infer from the native reference slice.
    n_domains: Optional[int] = None
    reference_slice_id: int = 0

    # SpaGCN.
    spagcn_p: float = 0.9
    spagcn_l_tolerance: float = 0.01
    spagcn_l_max_runs: int = 100

    spagcn_resolution_start: float = 0.7
    spagcn_resolution_step: float = 0.1
    spagcn_search_epochs: int = 20

    spagcn_train_epochs: int = 200
    spagcn_tolerance: float = 5e-3
    spagcn_learning_rate: float = 0.05
    spagcn_seed: int = 100

    refine_domains: bool = True
    refine_shape: str = "hexagon"

    # Evaluation.
    boundary_neighbors: int = 8
    boundary_tolerance_in_spot_spacing: float = 1.5

    # Plotting.
    max_points_per_map: int = 50000
    png_dpi: int = 400
    save_formats: Tuple[str, ...] = ("png",)
    preferred_layer_order: Tuple[str, ...] = (
        "Layer1",
        "Layer2",
        "Layer3",
        "Layer4",
        "Layer5",
        "Layer6",
        "WM",
    )

    @property
    def analysis_dir(self) -> Path:
        return (
            Path(self.root)
            / "virtual_tissue"
            / (
                f"{self.biological_axis_name}_group{self.group_id}"
                f"_grid{self.grid_size}_spatial{self.spatial_bandwidth}"
            )
            / "analysis"
        )

    @property
    def outdir(self) -> Path:
        return self.analysis_dir / "spagcn_domain_evaluation"

    @property
    def axis_mean_path(self) -> Path:
        return (
            self.analysis_dir
            / self.aggregation_output_subdir
            / f"aggregate_mean_ref_{self.reference_axis_value:.3f}.h5ad"
        )

    @property
    def metrics_csv(self) -> Path:
        return self.outdir / "spagcn_domain_metrics.csv"

    @property
    def predictions_csv(self) -> Path:
        return self.outdir / "spagcn_domain_predictions.csv.gz"

    @property
    def feature_genes_txt(self) -> Path:
        return self.outdir / "spagcn_feature_genes.txt"

    @property
    def config_json(self) -> Path:
        return self.outdir / "spagcn_domain_config.json"


CFG = Config()


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class Condition:
    name: str
    category: str
    adata: AnnData
    spatial_key: str
    ground_truth: np.ndarray
    source_slice: np.ndarray


@dataclass
class ConditionResult:
    condition: Condition
    coordinates: np.ndarray
    normalized_coordinates: np.ndarray
    coordinate_scale: float
    raw_prediction: np.ndarray
    final_prediction: np.ndarray
    probability: Optional[np.ndarray]
    mapped_prediction: np.ndarray
    metrics: Dict[str, float]
    spagcn_l: float
    spagcn_resolution: float


# =============================================================================
# VALIDATION AND LABEL HELPERS
# =============================================================================


def validate_config(cfg: Config) -> str:
    mode = cfg.run_mode.lower().strip()
    if mode not in {"eval", "plot", "both"}:
        raise ValueError("RUN_MODE must be 'eval', 'plot', or 'both'.")
    if cfg.reference_slice_id not in cfg.sample_ids:
        raise ValueError("reference_slice_id must be present in sample_ids.")
    if not 0.0 < cfg.spagcn_p < 1.0:
        raise ValueError("spagcn_p must lie strictly between 0 and 1.")
    if cfg.refine_shape not in {"hexagon", "square"}:
        raise ValueError("refine_shape must be 'hexagon' or 'square'.")
    return mode


def clean_labels(values: Iterable) -> np.ndarray:
    arr = pd.Series(values, dtype="object").astype(str).to_numpy(dtype=object)
    bad = np.isin(np.char.lower(arr.astype(str)), ["nan", "none", "<na>", ""])
    arr[bad] = None
    return arr


def valid_label_mask(*label_arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(label_arrays[0]), dtype=bool)
    for arr in label_arrays:
        series = pd.Series(arr, dtype="object")
        mask &= series.notna().to_numpy()
        mask &= ~series.astype(str).str.lower().isin(["nan", "none", "<na>", ""]).to_numpy()
    return mask


def canonical_layer_order(values: Iterable, preferred: Sequence[str]) -> List[str]:
    observed = {
        str(value)
        for value in pd.Series(values, dtype="object").dropna().astype(str)
        if str(value).lower() not in {"nan", "none", "<na>", ""}
    }
    ordered = [label for label in preferred if label in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


# =============================================================================
# INPUT AND AXIS-MEAN AGGREGATION
# =============================================================================


def validate_adata_for_condition(
    adata: AnnData,
    path: Path,
    layer_key: str,
    spatial_key: str,
) -> None:
    if layer_key not in adata.obs.columns:
        raise KeyError(
            f"{path} lacks obs[{layer_key!r}]. Every condition must carry its "
            "own ground-truth domain labels."
        )
    if spatial_key not in adata.obsm:
        raise KeyError(f"{path} lacks obsm[{spatial_key!r}].")


def load_single_slices(cfg: Config) -> Dict[int, AnnData]:
    singles: Dict[int, AnnData] = {}
    for sid in cfg.sample_ids:
        path = Path(cfg.h5ad_dir) / f"{cfg.group_id}_{sid}.h5ad"
        if not path.is_file():
            raise FileNotFoundError(path)

        adata = sc.read_h5ad(path)
        validate_adata_for_condition(
            adata,
            path=path,
            layer_key=cfg.layer_key,
            spatial_key=cfg.native_spatial_key,
        )
        singles[int(sid)] = adata
        print(f"[LOAD] Single {sid}: {adata.n_obs:,} spots x {adata.n_vars:,} genes")
    return singles


def build_axis_mean_aggregate(cfg: Config) -> Path:
    expected = cfg.axis_mean_path
    if cfg.rebuild_axis_aggregate or not expected.is_file():
        output = aggregate_virtual_slices(
            input_dir=str(cfg.analysis_dir),
            biological_axis_name=cfg.biological_axis_name,
            reference_axis_value=cfg.reference_axis_value,
            aggregation="mean",
            output_subdir=cfg.aggregation_output_subdir,
            output_name=expected.name,
            gene_chunk_size=cfg.aggregation_gene_chunk_size,
            coordinate_keys=(cfg.axis_spatial_key,),
        )
        expected = Path(output)

    if not expected.is_file():
        raise FileNotFoundError(expected)
    return expected


def load_axis_mean(path: Path, cfg: Config) -> AnnData:
    adata = sc.read_h5ad(path)
    if "valid" in adata.obs.columns:
        adata = adata[adata.obs["valid"].astype(bool)].copy()

    validate_adata_for_condition(
        adata,
        path=path,
        layer_key=cfg.layer_key,
        spatial_key=cfg.axis_spatial_key,
    )
    print(f"[LOAD] AXIS-mean: {adata.n_obs:,} spots x {adata.n_vars:,} genes")
    return adata


def common_gene_order(
    singles: Dict[int, AnnData],
    axis_mean: AnnData,
    reference_slice_id: int,
) -> pd.Index:
    reference_genes = singles[reference_slice_id].var_names
    common = set(reference_genes)
    for adata in list(singles.values()) + [axis_mean]:
        common.intersection_update(adata.var_names)

    if not common:
        raise ValueError("No genes are shared across native slices and AXIS-mean.")

    ordered = reference_genes[reference_genes.isin(common)]
    print(f"[GENES] shared across all conditions: {len(ordered):,}")
    return ordered


def subset_all_to_common(
    singles: Dict[int, AnnData],
    axis_mean: AnnData,
    common_genes: Sequence[str],
) -> Tuple[Dict[int, AnnData], AnnData]:
    single_out = {sid: adata[:, common_genes].copy() for sid, adata in singles.items()}
    return single_out, axis_mean[:, common_genes].copy()


# =============================================================================
# FIXED FEATURE SELECTION
# =============================================================================


def select_fixed_feature_genes(
    singles: Dict[int, AnnData],
    common_genes: Sequence[str],
    cfg: Config,
) -> List[str]:
    """Select one HVG panel from pooled native slices and reuse it everywhere."""
    pooled = sc.concat(
        [singles[sid][:, common_genes].copy() for sid in sorted(singles)],
        axis=0,
        join="inner",
        label="source_slice",
        keys=[str(sid) for sid in sorted(singles)],
        index_unique="-",
        merge="same",
    )

    sc.pp.filter_genes(pooled, min_cells=cfg.min_cells_per_gene)
    sc.pp.normalize_total(pooled, target_sum=1e4)
    sc.pp.log1p(pooled)

    n_top = min(int(cfg.n_top_genes), pooled.n_vars)
    sc.pp.highly_variable_genes(
        pooled,
        n_top_genes=n_top,
        flavor="seurat",
        batch_key="source_slice",
        subset=False,
    )

    genes = pooled.var_names[pooled.var["highly_variable"].to_numpy()].tolist()
    if len(genes) < 50:
        raise ValueError(
            f"Only {len(genes)} fixed feature genes remain. "
            "Reduce min_cells_per_gene or increase n_top_genes."
        )

    print(f"[GENES] fixed SpaGCN feature panel: {len(genes):,}")
    return genes


# =============================================================================
# CONDITIONS
# =============================================================================


def build_conditions(
    singles: Dict[int, AnnData],
    axis_mean: AnnData,
    cfg: Config,
) -> List[Condition]:
    """Use each condition's own obs[layer_key] labels as ground truth."""
    conditions: List[Condition] = []

    for sid in sorted(singles):
        adata = singles[sid]
        conditions.append(
            Condition(
                name=f"Single {sid}",
                category="single",
                adata=adata,
                spatial_key=cfg.native_spatial_key,
                ground_truth=clean_labels(adata.obs[cfg.layer_key].to_numpy()),
                source_slice=np.repeat(str(sid), adata.n_obs).astype(object),
            )
        )

    conditions.append(
        Condition(
            name="AXIS-mean",
            category="axis",
            adata=axis_mean,
            spatial_key=cfg.axis_spatial_key,
            ground_truth=clean_labels(axis_mean.obs[cfg.layer_key].to_numpy()),
            source_slice=np.repeat("atlas", axis_mean.n_obs).astype(object),
        )
    )
    return conditions


# =============================================================================
# SPATIAL-SCALE NORMALIZATION AND SPAGCN
# =============================================================================


def median_nearest_spacing(coords: np.ndarray) -> float:
    """Median nearest strictly positive distance, robust to duplicate points."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coordinates must have shape [n, 2+], got {coords.shape}")
    coords = coords[:, :2]
    if len(coords) < 2:
        return np.nan

    n_neighbors = min(16, len(coords))
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(coords).kneighbors(coords)
    positive_min = []
    for row in distances[:, 1:]:
        positive = row[np.isfinite(row) & (row > 1e-12)]
        if positive.size:
            positive_min.append(float(positive[0]))
    return float(np.median(positive_min)) if positive_min else np.nan


def normalize_spatial_coordinates(coords: np.ndarray) -> Tuple[np.ndarray, float]:
    """Center coordinates and scale them by one isotropic local-spacing unit.

    The transformation preserves neighborhood geometry while removing arbitrary
    coordinate units. A native 5000 x 5000 slice and a 500 x 500 AXIS atlas will
    therefore have comparable SpaGCN distance scales when their local layouts
    are geometrically equivalent.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coordinates must have shape [n, 2+], got {coords.shape}")
    coords = coords[:, :2]

    finite = np.all(np.isfinite(coords), axis=1)
    if not np.all(finite):
        raise ValueError("SpaGCN coordinates contain NaN or infinite values.")

    spacing = median_nearest_spacing(coords)
    if not np.isfinite(spacing) or spacing <= 0:
        # Fallback for highly duplicated or degenerate supports.
        spans = np.ptp(coords, axis=0)
        positive_spans = spans[np.isfinite(spans) & (spans > 0)]
        if positive_spans.size == 0:
            raise ValueError("Cannot determine a positive spatial scale.")
        spacing = float(np.median(positive_spans) / max(np.sqrt(len(coords)), 1.0))

    center = np.median(coords, axis=0, keepdims=True)
    normalized = (coords - center) / spacing
    return normalized.astype(np.float64), float(spacing)


def adaptive_l_search_bounds(adjacency: np.ndarray) -> Tuple[float, float]:
    """Choose SpaGCN l-search bounds from the normalized distance matrix."""
    distance = np.asarray(adjacency, dtype=np.float64)
    positive = distance[np.isfinite(distance) & (distance > 0)]
    if positive.size == 0:
        raise ValueError("SpaGCN adjacency contains no positive distances.")

    low = max(float(np.percentile(positive, 1)) * 1e-3, 1e-4)
    high = max(float(np.percentile(positive, 95)) * 10.0, 10.0)
    high = min(high, 1e5)
    if high <= low:
        high = low * 1e4
    return low, high


def set_fixed_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def prepare_spagcn_adata(adata: AnnData, feature_genes: Sequence[str]) -> AnnData:
    work = adata[:, feature_genes].copy()
    work.var_names_make_unique()

    if sp.issparse(work.X):
        nonzero = np.asarray((work.X > 0).sum(axis=0)).ravel() > 0
    else:
        nonzero = np.any(np.asarray(work.X) > 0, axis=0)

    if nonzero.sum() < 2:
        raise ValueError("SpaGCN input contains fewer than two expressed feature genes.")
    if nonzero.sum() != len(nonzero):
        print(f"[GENES] dropping {int((~nonzero).sum())} all-zero genes in this condition")
        work = work[:, nonzero].copy()

    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    return work


def run_spagcn(
    condition: Condition,
    feature_genes: Sequence[str],
    n_domains: int,
    cfg: Config,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    float,
    float,
    np.ndarray,
    float,
]:
    original_coords = np.asarray(
        condition.adata.obsm[condition.spatial_key],
        dtype=np.float64,
    )[:, :2]
    normalized_coords, coordinate_scale = normalize_spatial_coordinates(original_coords)

    print(
        f"[SPATIAL] {condition.name}: raw range "
        f"x={np.ptp(original_coords[:, 0]):.3f}, "
        f"y={np.ptp(original_coords[:, 1]):.3f}; "
        f"median spacing={coordinate_scale:.6g}; normalized spacing≈1"
    )

    work = prepare_spagcn_adata(condition.adata, feature_genes)
    adjacency = spg.calculate_adj_matrix(
        x=normalized_coords[:, 0].tolist(),
        y=normalized_coords[:, 1].tolist(),
        histology=False,
    )

    l_start, l_end = adaptive_l_search_bounds(adjacency)
    l_value = spg.search_l(
        cfg.spagcn_p,
        adjacency,
        start=l_start,
        end=l_end,
        tol=cfg.spagcn_l_tolerance,
        max_run=cfg.spagcn_l_max_runs,
    )
    if l_value is None or not np.isfinite(l_value):
        raise RuntimeError(
            f"{condition.name}: SpaGCN failed to determine l within "
            f"[{l_start:.4g}, {l_end:.4g}]."
        )

    set_fixed_seed(cfg.spagcn_seed)
    resolution = spg.search_res(
        work,
        adjacency,
        l_value,
        int(n_domains),
        start=cfg.spagcn_resolution_start,
        step=cfg.spagcn_resolution_step,
        tol=cfg.spagcn_tolerance,
        lr=cfg.spagcn_learning_rate,
        max_epochs=cfg.spagcn_search_epochs,
        r_seed=cfg.spagcn_seed,
        t_seed=cfg.spagcn_seed,
        n_seed=cfg.spagcn_seed,
    )
    if resolution is None or not np.isfinite(resolution):
        raise RuntimeError(f"{condition.name}: SpaGCN failed to determine resolution.")

    set_fixed_seed(cfg.spagcn_seed)
    model = spg.SpaGCN()
    model.set_l(float(l_value))
    model.train(
        work,
        adjacency,
        init_spa=True,
        init="louvain",
        res=float(resolution),
        tol=cfg.spagcn_tolerance,
        lr=cfg.spagcn_learning_rate,
        max_epochs=cfg.spagcn_train_epochs,
    )

    raw_prediction, probability = model.predict()
    raw_prediction = np.asarray(raw_prediction).astype(str)
    probability = None if probability is None else np.asarray(probability)

    if cfg.refine_domains:
        final_prediction = np.asarray(
            spg.refine(
                sample_id=work.obs_names.astype(str).tolist(),
                pred=raw_prediction.tolist(),
                dis=adjacency,
                shape=cfg.refine_shape,
            )
        ).astype(str)
    else:
        final_prediction = raw_prediction.copy()

    return (
        raw_prediction,
        final_prediction,
        probability,
        float(l_value),
        float(resolution),
        normalized_coords,
        coordinate_scale,
    )


# =============================================================================
# METRICS
# =============================================================================


def hungarian_mapping(gt: np.ndarray, pred: np.ndarray) -> Dict[str, str]:
    mask = valid_label_mask(gt, pred)
    gt_use = np.asarray(gt, dtype=object)[mask].astype(str)
    pred_use = np.asarray(pred, dtype=object)[mask].astype(str)

    gt_labels = np.unique(gt_use)
    pred_labels = np.unique(pred_use)
    contingency = np.zeros((len(gt_labels), len(pred_labels)), dtype=np.int64)
    gt_index = {label: idx for idx, label in enumerate(gt_labels)}
    pred_index = {label: idx for idx, label in enumerate(pred_labels)}

    for gt_label, pred_label in zip(gt_use, pred_use):
        contingency[gt_index[gt_label], pred_index[pred_label]] += 1

    rows, cols = linear_sum_assignment(-contingency)
    return {pred_labels[col]: gt_labels[row] for row, col in zip(rows, cols)}


def map_prediction_to_ground_truth(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    mapping = hungarian_mapping(gt, pred)
    return np.asarray(
        [mapping.get(str(label), f"Cluster {label}") for label in pred],
        dtype=object,
    )


def macro_dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    mask = valid_label_mask(gt, pred)
    if mask.sum() == 0:
        return np.nan

    gt_use = np.asarray(gt, dtype=object)[mask].astype(str)
    pred_use = np.asarray(pred, dtype=object)[mask].astype(str)
    mapped = map_prediction_to_ground_truth(gt_use, pred_use)

    scores = []
    for label in np.unique(gt_use):
        gt_mask = gt_use == label
        pred_mask = mapped == label
        denominator = int(gt_mask.sum() + pred_mask.sum())
        scores.append(
            2.0 * np.logical_and(gt_mask, pred_mask).sum() / denominator
            if denominator
            else np.nan
        )
    return float(np.nanmean(scores))


def boundary_nodes(labels: np.ndarray, coordinates: np.ndarray, k: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    valid = valid_label_mask(labels)
    output = np.zeros(len(labels), dtype=bool)
    if valid.sum() < 3:
        return output

    coords_valid = np.asarray(coordinates)[valid]
    labels_valid = labels[valid].astype(str)
    n_neighbors = min(int(k) + 1, len(coords_valid))
    indices = (
        NearestNeighbors(n_neighbors=n_neighbors)
        .fit(coords_valid)
        .kneighbors(coords_valid, return_distance=False)
    )
    is_boundary = np.any(labels_valid[indices[:, 1:]] != labels_valid[:, None], axis=1)
    output[np.where(valid)[0]] = is_boundary
    return output


def boundary_f1_score(
    gt: np.ndarray,
    pred: np.ndarray,
    coordinates: np.ndarray,
    k: int,
    tolerance_factor: float,
) -> float:
    gt_boundary = boundary_nodes(gt, coordinates, k)
    pred_boundary = boundary_nodes(pred, coordinates, k)
    gt_coords = np.asarray(coordinates)[gt_boundary]
    pred_coords = np.asarray(coordinates)[pred_boundary]

    if len(gt_coords) == 0 or len(pred_coords) == 0:
        return np.nan

    spacing = median_nearest_spacing(np.asarray(coordinates))
    tolerance = tolerance_factor * spacing
    if not np.isfinite(tolerance) or tolerance <= 0:
        return np.nan

    pred_to_gt = (
        NearestNeighbors(n_neighbors=1)
        .fit(gt_coords)
        .kneighbors(pred_coords, return_distance=True)[0][:, 0]
    )
    gt_to_pred = (
        NearestNeighbors(n_neighbors=1)
        .fit(pred_coords)
        .kneighbors(gt_coords, return_distance=True)[0][:, 0]
    )

    precision = float(np.mean(pred_to_gt <= tolerance))
    recall = float(np.mean(gt_to_pred <= tolerance))
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def clustering_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    coordinates: np.ndarray,
    cfg: Config,
) -> Dict[str, float]:
    mask = valid_label_mask(gt, pred)
    if mask.sum() < 2:
        return {
            "ari": np.nan,
            "nmi": np.nan,
            "macro_dice": np.nan,
            "boundary_f1": np.nan,
            "n_eval": int(mask.sum()),
        }

    gt_use = np.asarray(gt, dtype=object)[mask].astype(str)
    pred_use = np.asarray(pred, dtype=object)[mask].astype(str)
    coords_use = np.asarray(coordinates)[mask]

    return {
        "ari": float(adjusted_rand_score(gt_use, pred_use)),
        "nmi": float(normalized_mutual_info_score(gt_use, pred_use)),
        "macro_dice": macro_dice_score(gt_use, pred_use),
        "boundary_f1": boundary_f1_score(
            gt_use,
            pred_use,
            coords_use,
            k=cfg.boundary_neighbors,
            tolerance_factor=cfg.boundary_tolerance_in_spot_spacing,
        ),
        "n_eval": int(mask.sum()),
    }


# =============================================================================
# EVALUATION TABLES
# =============================================================================


def evaluate_condition(
    condition: Condition,
    feature_genes: Sequence[str],
    n_domains: int,
    cfg: Config,
) -> ConditionResult:
    print(f"\n[SPAGCN] {condition.name}")

    (
        raw_prediction,
        final_prediction,
        probability,
        l_value,
        resolution,
        normalized_coordinates,
        coordinate_scale,
    ) = run_spagcn(
        condition,
        feature_genes=feature_genes,
        n_domains=n_domains,
        cfg=cfg,
    )

    coordinates = np.asarray(
        condition.adata.obsm[condition.spatial_key],
        dtype=np.float32,
    )[:, :2]
    metrics = clustering_metrics(condition.ground_truth, final_prediction, coordinates, cfg)
    mapped = map_prediction_to_ground_truth(condition.ground_truth, final_prediction)

    print(
        f"[EVAL] {condition.name:<12s} "
        f"ARI={metrics['ari']:.3f} | NMI={metrics['nmi']:.3f} | "
        f"Dice={metrics['macro_dice']:.3f} | Boundary F1={metrics['boundary_f1']:.3f}"
    )

    return ConditionResult(
        condition=condition,
        coordinates=coordinates,
        normalized_coordinates=normalized_coordinates,
        coordinate_scale=coordinate_scale,
        raw_prediction=raw_prediction,
        final_prediction=final_prediction,
        probability=probability,
        mapped_prediction=mapped,
        metrics=metrics,
        spagcn_l=l_value,
        spagcn_resolution=resolution,
    )


def metrics_dataframe(results: Sequence[ConditionResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "condition": result.condition.name,
                "category": result.condition.category,
                "n_spots": result.condition.adata.n_obs,
                "n_ground_truth_domains": int(
                    len(
                        np.unique(
                            np.asarray(result.condition.ground_truth, dtype=object)[
                                valid_label_mask(result.condition.ground_truth)
                            ].astype(str)
                        )
                    )
                ),
                "n_domains_detected_raw": int(len(np.unique(result.raw_prediction))),
                "n_domains_detected_final": int(len(np.unique(result.final_prediction))),
                "coordinate_scale_median_spacing": result.coordinate_scale,
                "spagcn_l_normalized_coordinates": result.spagcn_l,
                "spagcn_resolution": result.spagcn_resolution,
                **result.metrics,
            }
        )
    return pd.DataFrame(rows)


def predictions_dataframe(results: Sequence[ConditionResult]) -> pd.DataFrame:
    frames = []
    for result in results:
        condition = result.condition
        frames.append(
            pd.DataFrame(
                {
                    "condition": condition.name,
                    "category": condition.category,
                    "observation_id": condition.adata.obs_names.astype(str),
                    "source_slice": condition.source_slice.astype(str),
                    "x": result.coordinates[:, 0],
                    "y": result.coordinates[:, 1],
                    "x_spagcn_normalized": result.normalized_coordinates[:, 0],
                    "y_spagcn_normalized": result.normalized_coordinates[:, 1],
                    "ground_truth": condition.ground_truth,
                    "spagcn_raw": result.raw_prediction,
                    "spagcn_final": result.final_prediction,
                    "spagcn_mapped": result.mapped_prediction,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# PLOTS
# =============================================================================


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def condition_order(metrics: pd.DataFrame) -> List[str]:
    preferred = ["Single 0", "Single 1", "Single 2", "Single 3", "AXIS-mean"]
    observed = set(metrics["condition"].astype(str))
    return [name for name in preferred if name in observed] + sorted(observed.difference(preferred))


def condition_colors(order: Sequence[str]) -> Dict[str, str]:
    return {
        name: ("#E6550D" if name == "AXIS-mean" else "#BDBDBD")
        for name in order
    }


def layer_palette(layers: Sequence[str]) -> Dict[str, str]:
    base = [
        "#4E79A7",
        "#59A14F",
        "#F28E2B",
        "#E15759",
        "#B07AA1",
        "#76B7B2",
        "#9C755F",
        "#BAB0AC",
    ]
    return {layer: base[idx % len(base)] for idx, layer in enumerate(layers)}


def save_figure(fig: plt.Figure, stem: Path, cfg: Config) -> None:
    for extension in cfg.save_formats:
        path = stem.with_suffix(f".{extension}")
        kwargs = {"bbox_inches": "tight"}
        if extension.lower() == "png":
            kwargs["dpi"] = cfg.png_dpi
        fig.savefig(path, **kwargs)
        print(f"[PLOT] saved: {path}")
    plt.close(fig)


def plot_metric_summary(metrics: pd.DataFrame, cfg: Config) -> None:
    set_publication_style()
    order = condition_order(metrics)
    colors = condition_colors(order)
    data = metrics.set_index("condition").loc[order].reset_index()

    specs = [
        ("ari", "ARI"),
        ("macro_dice", "Macro Dice"),
        ("boundary_f1", "Boundary F1"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), squeeze=False)
    x = np.arange(len(data))

    for ax, (metric, label) in zip(axes[0], specs):
        ax.bar(
            x,
            data[metric].astype(float),
            color=[colors[name] for name in data["condition"]],
            edgecolor="black",
            linewidth=0.6,
            zorder=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(data["condition"], rotation=40, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)

    fig.suptitle("SpaGCN spatial-domain recovery", y=1.02, fontsize=10)
    fig.tight_layout()
    save_figure(fig, cfg.outdir / "spagcn_domain_metric_summary", cfg)


def plot_domain_maps(predictions: pd.DataFrame, metrics: pd.DataFrame, cfg: Config) -> None:
    set_publication_style()
    order = condition_order(metrics)
    layers = canonical_layer_order(predictions["ground_truth"], cfg.preferred_layer_order)
    palette = layer_palette(layers)

    n_columns = 3
    n_rows = int(math.ceil(len(order) / n_columns))

    for value_column, suffix, title in [
        ("ground_truth", "ground_truth", "Ground-truth domains"),
        ("spagcn_mapped", "spagcn", "SpaGCN detected domains"),
    ]:
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(3.0 * n_columns, 2.7 * n_rows),
            squeeze=False,
        )
        axes_flat = axes.ravel()
        rng = np.random.default_rng(0)

        for ax, name in zip(axes_flat, order):
            subset = predictions[predictions["condition"] == name].copy()
            if len(subset) > cfg.max_points_per_map:
                chosen = rng.choice(len(subset), cfg.max_points_per_map, replace=False)
                subset = subset.iloc[chosen]

            labels = subset[value_column].astype(str)
            point_colors = [palette.get(label, "#D0D0D0") for label in labels]
            point_size = max(0.5, min(7.0, 18000.0 / max(len(subset), 1)))
            ax.scatter(
                subset["x"],
                subset["y"],
                c=point_colors,
                s=point_size,
                linewidths=0,
                rasterized=True,
            )

            ari = float(metrics.loc[metrics["condition"] == name, "ari"].iloc[0])
            ax.set_title(f"{name}\nARI = {ari:.3f}")
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.axis("off")

        for ax in axes_flat[len(order):]:
            ax.axis("off")

        handles = [
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=5,
                markerfacecolor=palette[layer],
                markeredgewidth=0,
                label=layer,
            )
            for layer in layers
        ]
        fig.legend(
            handles=handles,
            labels=layers,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=min(len(layers), 7),
            frameon=False,
        )
        fig.suptitle(title, y=0.995, fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        save_figure(fig, cfg.outdir / f"spagcn_domain_maps_{suffix}", cfg)


# =============================================================================
# WORKFLOWS
# =============================================================================


def infer_n_domains(singles: Dict[int, AnnData], cfg: Config) -> int:
    if cfg.n_domains is not None:
        return int(cfg.n_domains)

    labels = clean_labels(
        singles[cfg.reference_slice_id].obs[cfg.layer_key].to_numpy()
    )
    valid = labels[valid_label_mask(labels)]
    n_domains = int(len(np.unique(valid.astype(str))))
    if n_domains < 2:
        raise ValueError("Could not infer at least two domains from the reference slice.")
    return n_domains


def run_evaluation(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    singles = load_single_slices(cfg)
    axis_mean_path = build_axis_mean_aggregate(cfg)
    axis_mean = load_axis_mean(axis_mean_path, cfg)

    common_genes = common_gene_order(singles, axis_mean, cfg.reference_slice_id)
    singles, axis_mean = subset_all_to_common(singles, axis_mean, common_genes)
    feature_genes = select_fixed_feature_genes(singles, common_genes, cfg)
    cfg.feature_genes_txt.write_text("\n".join(feature_genes) + "\n")

    n_domains = infer_n_domains(singles, cfg)
    print(f"[DOMAIN] requested number of SpaGCN domains: {n_domains}")

    conditions = build_conditions(singles, axis_mean, cfg)
    results = [
        evaluate_condition(
            condition,
            feature_genes=feature_genes,
            n_domains=n_domains,
            cfg=cfg,
        )
        for condition in conditions
    ]

    metrics = metrics_dataframe(results)
    predictions = predictions_dataframe(results)

    metrics.to_csv(cfg.metrics_csv, index=False)
    predictions.to_csv(cfg.predictions_csv, index=False, compression="gzip")

    config_payload = asdict(cfg)
    config_payload.update(
        {
            "analysis_dir": str(cfg.analysis_dir),
            "outdir": str(cfg.outdir),
            "axis_mean_path": str(axis_mean_path),
            "n_feature_genes": len(feature_genes),
            "n_domains": int(n_domains),
            "spatial_normalization": (
                "center each condition and divide both axes by its median "
                "positive nearest-neighbor spacing"
            ),
        }
    )
    cfg.config_json.write_text(json.dumps(config_payload, indent=2, default=str))

    print(f"[SAVE] {cfg.metrics_csv}")
    print(f"[SAVE] {cfg.predictions_csv}")
    print(f"[SAVE] {cfg.feature_genes_txt}")
    print("\n[RESULTS]")
    print(
        metrics[
            [
                "condition",
                "ari",
                "nmi",
                "macro_dice",
                "boundary_f1",
                "n_domains_detected_final",
                "coordinate_scale_median_spacing",
                "spagcn_l_normalized_coordinates",
                "spagcn_resolution",
            ]
        ].to_string(index=False)
    )
    return metrics, predictions


def load_saved_results(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for path in [cfg.metrics_csv, cfg.predictions_csv]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing saved result {path}. Run with RUN_MODE='eval' or 'both' first."
            )
    return pd.read_csv(cfg.metrics_csv), pd.read_csv(cfg.predictions_csv)


def run_plotting(metrics: pd.DataFrame, predictions: pd.DataFrame, cfg: Config) -> None:
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    plot_metric_summary(metrics, cfg)
    plot_domain_maps(predictions, metrics, cfg)


def main() -> None:
    mode = validate_config(CFG)

    if mode in {"eval", "both"}:
        metrics, predictions = run_evaluation(CFG)
    else:
        metrics, predictions = load_saved_results(CFG)

    if mode in {"plot", "both"}:
        run_plotting(metrics, predictions, CFG)


if __name__ == "__main__":
    main()
