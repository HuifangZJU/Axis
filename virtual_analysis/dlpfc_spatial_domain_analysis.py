#!/usr/bin/env python3
"""DLPFC spatial-domain benchmark for native, pooled, warped, and AXIS atlases.

The script evaluates the same unsupervised spatial-domain detector on:

    Single 0, Single 1, Single 2, Single 3,
    Pool (raw), Pool (warped),
    AXIS-mean, AXIS-median, AXIS-max.

Primary design
--------------
* A fixed shared gene set and fixed HVG panel are used for every condition.
* The number of domains is inferred from the known DLPFC labels or set manually.
* Clustering never uses ARI or domain identities to tune model parameters.
* Single slices are evaluated against their own manual annotations.
* Integrated conditions are primarily evaluated against slice-0 labels transferred
  into the corresponding coordinate system by nearest-neighbor assignment.
* Pooled conditions additionally report ARI against their source-slice labels.
* Multiple random seeds quantify computational stability, not biological replication.
* The script reports whether an AXIS aggregation is objectively ranked first; it
  never modifies the evaluation to force an AXIS advantage.

Projection-back is intentionally not included. Superior atlas-space ARI does not
mathematically guarantee superior projected labels because projection introduces
additional registration and interpolation error; that claim should be tested later.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


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
    reference_slice_id: int = 0

    root: str = "/media/huifang/data/registration/DLPFC/huifang"
    h5ad_dir: str = "/media/huifang/data/registration/DLPFC/huifang/h5ad"
    sample_ids: Tuple[int, ...] = (0, 1, 2, 3)

    layer_key: str = "layer_guess_reordered"
    native_spatial_key: str = "spatial"
    warped_spatial_key: str = "spatial_axis"
    axis_spatial_key: str = "spatial"

    # Generate all three AXIS atlas variants.
    axis_aggregation_methods: Tuple[str, ...] = ("mean", "median", "max")
    rebuild_axis_aggregates: bool = False
    aggregation_output_subdir: str = "aggregation_domain"
    aggregation_gene_chunk_size: int = 256

    # Fixed feature panel used by every condition.
    n_top_genes: int = 2000
    min_cells_per_gene: int = 10
    n_expression_pcs: int = 40
    exclude_known_markers_from_clustering: bool = False
    heldout_marker_genes: Tuple[str, ...] = (
        "CUX2", "RORB", "ENC1", "PCP4", "KCNIP2",
        "SYT6", "FOXP2", "MOBP", "MBP", "MFGE8",
    )

    # Spatial-expression graph.
    expression_neighbors: int = 20
    spatial_neighbors: int = 8
    expression_graph_weight: float = 0.70
    center_pooled_pcs_by_source_slice: bool = True

    # Domain clustering. Spectral clustering gives exactly K domains and avoids
    # condition-specific Leiden resolution tuning. "leiden" is also supported.
    clustering_engine: str = "spectral"  # "spectral" or "leiden"
    n_domains: Optional[int] = None  # None -> infer from reference-slice labels
    random_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    kmeans_n_init: int = 50
    spectral_extra_components: int = 3
    leiden_resolution_grid: Tuple[float, ...] = tuple(
        np.round(np.linspace(0.10, 3.00, 30), 3)
    )

    # Reference label transfer. All spots are assigned a nearest slice-0 label;
    # transfer distance is reported so poor coordinate correspondence is visible.
    boundary_neighbors: int = 8
    boundary_tolerance_in_spot_spacing: float = 1.5

    # Plotting.
    max_points_per_map: int = 50000
    png_dpi: int = 400
    save_formats: Tuple[str, ...] = ("png",)

    preferred_layer_order: Tuple[str, ...] = (
        "Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"
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
        return self.analysis_dir / "spatial_domain_evaluation"

    @property
    def seed_metrics_csv(self) -> Path:
        return self.outdir / "domain_seed_metrics.csv"

    @property
    def summary_csv(self) -> Path:
        return self.outdir / "domain_method_summary.csv"

    @property
    def representative_predictions_csv(self) -> Path:
        return self.outdir / "domain_representative_predictions.csv.gz"

    @property
    def feature_genes_txt(self) -> Path:
        return self.outdir / "domain_feature_genes.txt"

    @property
    def report_md(self) -> Path:
        return self.outdir / "domain_evaluation_report.md"

    @property
    def config_json(self) -> Path:
        return self.outdir / "domain_evaluation_config.json"


CFG = Config()


# =============================================================================
# DATA STRUCTURES AND VALIDATION
# =============================================================================


@dataclass
class Condition:
    name: str
    adata: AnnData
    spatial_key: str
    primary_gt: np.ndarray
    reference_gt: Optional[np.ndarray]
    source_gt: Optional[np.ndarray]
    category: str
    expression_group: str
    source_slice: np.ndarray
    transfer_distance: Optional[np.ndarray] = None
    transfer_spacing: float = np.nan


@dataclass
class ConditionResult:
    condition: Condition
    coordinates: np.ndarray
    predictions_by_seed: Dict[int, np.ndarray]
    metrics: pd.DataFrame
    representative_seed: int
    representative_prediction: np.ndarray


def validate_config(cfg: Config) -> str:
    mode = cfg.run_mode.lower().strip()
    if mode not in {"eval", "plot", "both"}:
        raise ValueError("RUN_MODE must be 'eval', 'plot', or 'both'.")
    if cfg.reference_slice_id not in cfg.sample_ids:
        raise ValueError("reference_slice_id must be present in sample_ids.")
    if not 0.0 <= cfg.expression_graph_weight <= 1.0:
        raise ValueError("expression_graph_weight must lie in [0, 1].")
    if cfg.clustering_engine not in {"spectral", "leiden"}:
        raise ValueError("clustering_engine must be 'spectral' or 'leiden'.")
    if not cfg.random_seeds:
        raise ValueError("At least one random seed is required.")
    return mode


def clean_labels(values: Iterable) -> np.ndarray:
    arr = pd.Series(values, dtype="object").astype(str).to_numpy()
    bad = np.isin(np.char.lower(arr.astype(str)), ["nan", "none", "<na>", ""])
    arr = arr.astype(object)
    arr[bad] = None
    return arr


def valid_label_mask(*label_arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(label_arrays[0]), dtype=bool)
    for arr in label_arrays:
        s = pd.Series(arr, dtype="object")
        mask &= s.notna().to_numpy()
        mask &= ~s.astype(str).str.lower().isin(["nan", "none", "<na>", ""]).to_numpy()
    return mask


def canonical_layer_order(values: Iterable, preferred: Sequence[str]) -> List[str]:
    observed = {
        str(x)
        for x in pd.Series(values, dtype="object").dropna().astype(str)
        if str(x).lower() not in {"nan", "none", "<na>", ""}
    }
    ordered = [x for x in preferred if x in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


# =============================================================================
# INPUT AND AGGREGATION
# =============================================================================


def load_single_slices(cfg: Config) -> Dict[int, AnnData]:
    out: Dict[int, AnnData] = {}
    for sid in cfg.sample_ids:
        path = Path(cfg.h5ad_dir) / f"{cfg.group_id}_{sid}.h5ad"
        if not path.is_file():
            raise FileNotFoundError(path)
        ad = sc.read_h5ad(path)
        if cfg.layer_key not in ad.obs:
            raise KeyError(f"{path} lacks obs[{cfg.layer_key!r}].")
        if cfg.native_spatial_key not in ad.obsm:
            raise KeyError(f"{path} lacks obsm[{cfg.native_spatial_key!r}].")
        out[sid] = ad
        print(f"[LOAD] Single {sid}: {ad.n_obs:,} x {ad.n_vars:,}")
    return out


def build_axis_aggregates(cfg: Config) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for method in cfg.axis_aggregation_methods:
        expected = (
            cfg.analysis_dir
            / cfg.aggregation_output_subdir
            / f"aggregate_{method}_ref_{cfg.reference_axis_value:.3f}.h5ad"
        )
        if cfg.rebuild_axis_aggregates or not expected.is_file():
            path = aggregate_virtual_slices(
                input_dir=str(cfg.analysis_dir),
                biological_axis_name=cfg.biological_axis_name,
                reference_axis_value=cfg.reference_axis_value,
                aggregation=method,
                output_subdir=cfg.aggregation_output_subdir,
                output_name=expected.name,
                gene_chunk_size=cfg.aggregation_gene_chunk_size,
                coordinate_keys=(cfg.axis_spatial_key,),
            )
            expected = Path(path)
        if not expected.is_file():
            raise FileNotFoundError(expected)
        paths[method] = expected
    return paths


def load_axis_aggregates(paths: Mapping[str, Path], cfg: Config) -> Dict[str, AnnData]:
    out: Dict[str, AnnData] = {}
    for method, path in paths.items():
        ad = sc.read_h5ad(path)
        if "valid" in ad.obs.columns:
            ad = ad[ad.obs["valid"]].copy()
        if cfg.axis_spatial_key not in ad.obsm:
            raise KeyError(f"{path} lacks obsm[{cfg.axis_spatial_key!r}].")
        out[method] = ad
        print(f"[LOAD] AXIS-{method}: {ad.n_obs:,} x {ad.n_vars:,}")
    return out


def common_gene_order(
    singles: Mapping[int, AnnData],
    axis_ads: Mapping[str, AnnData],
    reference_slice_id: int,
) -> pd.Index:
    ref_genes = singles[reference_slice_id].var_names
    common = set(ref_genes)
    for ad in list(singles.values()) + list(axis_ads.values()):
        common.intersection_update(ad.var_names)
    if not common:
        raise ValueError("No common genes across native and AXIS conditions.")
    ordered = ref_genes[ref_genes.isin(common)]
    print(f"[GENES] shared genes across all conditions: {len(ordered):,}")
    return ordered


def subset_all_to_common(
    singles: Mapping[int, AnnData],
    axis_ads: Mapping[str, AnnData],
    common_genes: Sequence[str],
) -> Tuple[Dict[int, AnnData], Dict[str, AnnData]]:
    s = {sid: ad[:, common_genes].copy() for sid, ad in singles.items()}
    a = {method: ad[:, common_genes].copy() for method, ad in axis_ads.items()}
    return s, a


# =============================================================================
# FIXED FEATURE SELECTION
# =============================================================================


def select_fixed_feature_genes(
    singles: Mapping[int, AnnData],
    common_genes: Sequence[str],
    cfg: Config,
) -> List[str]:
    """Select HVGs once from pooled real slices, then reuse them everywhere."""
    pieces = []
    for sid in sorted(singles):
        ad = singles[sid][:, common_genes].copy()
        ad.obs["source_slice"] = str(sid)
        pieces.append(ad)

    ref = sc.concat(
        pieces,
        axis=0,
        join="inner",
        label="source_slice_concat",
        keys=[str(sid) for sid in sorted(singles)],
        index_unique="-",
        merge="same",
    )

    sc.pp.filter_genes(ref, min_cells=cfg.min_cells_per_gene)
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)

    n_top = min(cfg.n_top_genes, ref.n_vars)
    sc.pp.highly_variable_genes(
        ref,
        n_top_genes=n_top,
        flavor="seurat",
        batch_key="source_slice_concat",
        subset=False,
    )
    genes = ref.var_names[ref.var["highly_variable"].to_numpy()].tolist()

    if cfg.exclude_known_markers_from_clustering:
        heldout = set(cfg.heldout_marker_genes)
        genes = [g for g in genes if g not in heldout]

    if len(genes) < 50:
        raise ValueError(
            f"Only {len(genes)} feature genes remain. Reduce filtering or marker exclusion."
        )

    print(
        f"[GENES] fixed clustering panel: {len(genes):,} genes "
        f"(held-out markers excluded={cfg.exclude_known_markers_from_clustering})"
    )
    return genes


# =============================================================================
# REFERENCE LABEL TRANSFER AND CONDITION CONSTRUCTION
# =============================================================================


def median_nearest_spacing(coords: np.ndarray) -> float:
    """Estimate spot spacing robustly even when pooled slices duplicate coordinates."""
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) < 2:
        return np.nan

    # Query several neighbors because aligned pooled slices may contain exact
    # coordinate duplicates. For each observation, use its nearest strictly
    # positive distance rather than assuming neighbor 1 is nonzero.
    n_neighbors = min(16, len(coords))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    dist, _ = nbrs.kneighbors(coords)
    dist = dist[:, 1:]
    positive_min = []
    for row in dist:
        positive = row[row > 1e-12]
        if len(positive):
            positive_min.append(float(positive[0]))
    return float(np.median(positive_min)) if positive_min else np.nan


def transfer_labels_nearest(
    reference_coords: np.ndarray,
    reference_labels: np.ndarray,
    target_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    ref_labels = clean_labels(reference_labels)
    valid = valid_label_mask(ref_labels)
    if valid.sum() == 0:
        raise ValueError("Reference slice contains no valid domain labels.")

    ref_coords = np.asarray(reference_coords, dtype=np.float64)[valid]
    ref_labels = ref_labels[valid]
    target_coords = np.asarray(target_coords, dtype=np.float64)

    nn = NearestNeighbors(n_neighbors=1).fit(ref_coords)
    distances, indices = nn.kneighbors(target_coords)
    labels = ref_labels[indices[:, 0]].astype(object)
    spacing = median_nearest_spacing(ref_coords)
    return labels, distances[:, 0].astype(float), spacing


def make_pool_adata(singles: Mapping[int, AnnData], cfg: Config) -> AnnData:
    pieces: List[AnnData] = []
    for sid in sorted(singles):
        ad = singles[sid].copy()
        if cfg.warped_spatial_key not in ad.obsm:
            raise KeyError(
                f"Single {sid} lacks obsm[{cfg.warped_spatial_key!r}], "
                "which is required for Pool (warped)."
            )
        ad.obs["source_slice"] = str(sid)
        ad.obsm["spatial_native_eval"] = np.asarray(
            ad.obsm[cfg.native_spatial_key], dtype=np.float32
        )
        ad.obsm["spatial_warped_eval"] = np.asarray(
            ad.obsm[cfg.warped_spatial_key], dtype=np.float32
        )
        pieces.append(ad)

    pool = sc.concat(
        pieces,
        axis=0,
        join="inner",
        label="source_slice_concat",
        keys=[str(sid) for sid in sorted(singles)],
        index_unique="-",
        merge="same",
    )
    return pool


def build_conditions(
    singles: Mapping[int, AnnData],
    axis_ads: Mapping[str, AnnData],
    cfg: Config,
) -> List[Condition]:
    ref_ad = singles[cfg.reference_slice_id]
    ref_labels = clean_labels(ref_ad.obs[cfg.layer_key].to_numpy())
    ref_native_coords = np.asarray(ref_ad.obsm[cfg.native_spatial_key])
    ref_atlas_key = (
        cfg.warped_spatial_key
        if cfg.warped_spatial_key in ref_ad.obsm
        else cfg.native_spatial_key
    )
    ref_atlas_coords = np.asarray(ref_ad.obsm[ref_atlas_key])

    conditions: List[Condition] = []

    for sid in sorted(singles):
        ad = singles[sid]
        labels = clean_labels(ad.obs[cfg.layer_key].to_numpy())
        conditions.append(
            Condition(
                name=f"Single {sid}",
                adata=ad,
                spatial_key=cfg.native_spatial_key,
                primary_gt=labels,
                reference_gt=None,
                source_gt=labels,
                category="single",
                expression_group=f"single_{sid}",
                source_slice=np.repeat(str(sid), ad.n_obs).astype(object),
            )
        )

    pool = make_pool_adata(singles, cfg)
    pool_source_gt = clean_labels(pool.obs[cfg.layer_key].to_numpy())
    pool_source_slice = pool.obs["source_slice"].astype(str).to_numpy()

    raw_ref_gt, raw_dist, raw_spacing = transfer_labels_nearest(
        ref_native_coords,
        ref_labels,
        np.asarray(pool.obsm["spatial_native_eval"]),
    )
    conditions.append(
        Condition(
            name="Pool (raw)",
            adata=pool,
            spatial_key="spatial_native_eval",
            primary_gt=raw_ref_gt,
            reference_gt=raw_ref_gt,
            source_gt=pool_source_gt,
            category="pool",
            expression_group="pool",
            source_slice=pool_source_slice,
            transfer_distance=raw_dist,
            transfer_spacing=raw_spacing,
        )
    )

    warped_ref_gt, warped_dist, warped_spacing = transfer_labels_nearest(
        ref_atlas_coords,
        ref_labels,
        np.asarray(pool.obsm["spatial_warped_eval"]),
    )
    conditions.append(
        Condition(
            name="Pool (warped)",
            adata=pool,
            spatial_key="spatial_warped_eval",
            primary_gt=warped_ref_gt,
            reference_gt=warped_ref_gt,
            source_gt=pool_source_gt,
            category="pool",
            expression_group="pool",
            source_slice=pool_source_slice,
            transfer_distance=warped_dist,
            transfer_spacing=warped_spacing,
        )
    )

    for method in cfg.axis_aggregation_methods:
        ad = axis_ads[method]
        axis_gt, axis_dist, axis_spacing = transfer_labels_nearest(
            ref_atlas_coords,
            ref_labels,
            np.asarray(ad.obsm[cfg.axis_spatial_key]),
        )
        conditions.append(
            Condition(
                name=f"AXIS-{method}",
                adata=ad,
                spatial_key=cfg.axis_spatial_key,
                primary_gt=axis_gt,
                reference_gt=axis_gt,
                source_gt=None,
                category="axis",
                expression_group=f"axis_{method}",
                source_slice=np.repeat("atlas", ad.n_obs).astype(object),
                transfer_distance=axis_dist,
                transfer_spacing=axis_spacing,
            )
        )

    return conditions


# =============================================================================
# EXPRESSION PROCESSING AND GRAPH CONSTRUCTION
# =============================================================================


def normalized_log_matrix(ad: AnnData, genes: Sequence[str]) -> AnnData:
    out = ad[:, genes].copy()
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    return out


def compute_expression_pcs(
    ad: AnnData,
    genes: Sequence[str],
    n_pcs: int,
    random_state: int = 0,
) -> np.ndarray:
    work = normalized_log_matrix(ad, genes)
    max_components = min(work.n_obs - 1, work.n_vars - 1)
    n_components = min(n_pcs, max_components)
    if n_components < 2:
        raise ValueError("Insufficient matrix rank for expression embedding.")

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    pcs = svd.fit_transform(work.X)
    pcs = StandardScaler().fit_transform(pcs)
    return np.asarray(pcs, dtype=np.float32)


def center_pcs_within_batch(pcs: np.ndarray, batches: np.ndarray) -> np.ndarray:
    centered = np.asarray(pcs, dtype=np.float64).copy()
    global_mean = centered.mean(axis=0, keepdims=True)
    for batch in pd.unique(pd.Series(batches).astype(str)):
        mask = pd.Series(batches).astype(str).to_numpy() == str(batch)
        centered[mask] = centered[mask] - centered[mask].mean(axis=0, keepdims=True)
    centered += global_mean
    return StandardScaler().fit_transform(centered).astype(np.float32)


def knn_affinity(features: np.ndarray, n_neighbors: int) -> sp.csr_matrix:
    features = np.asarray(features, dtype=np.float64)
    n_obs = len(features)
    if n_obs < 3:
        raise ValueError("At least three observations are required for kNN graph construction.")
    k = min(max(1, int(n_neighbors)), n_obs - 1)

    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    model.fit(features)
    distances, indices = model.kneighbors(features)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    positive = distances[distances > 0]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    sigma = max(sigma, 1e-8)
    weights = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))

    rows = np.repeat(np.arange(n_obs), k)
    cols = indices.ravel()
    data = weights.ravel()
    graph = sp.csr_matrix((data, (rows, cols)), shape=(n_obs, n_obs))
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def row_normalized_graph(graph: sp.csr_matrix) -> sp.csr_matrix:
    row_sum = np.asarray(graph.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sum, dtype=np.float64)
    np.divide(1.0, row_sum, out=inv, where=row_sum > 0)
    return sp.diags(inv) @ graph


def combined_expression_spatial_graph(
    pcs: np.ndarray,
    coordinates: np.ndarray,
    expression_neighbors: int,
    spatial_neighbors: int,
    expression_weight: float,
) -> sp.csr_matrix:
    expr = knn_affinity(pcs, expression_neighbors)
    spatial = knn_affinity(coordinates, spatial_neighbors)

    expr = row_normalized_graph(expr)
    spatial = row_normalized_graph(spatial)

    combined = expression_weight * expr + (1.0 - expression_weight) * spatial
    combined = 0.5 * (combined + combined.T)
    combined = combined.tocsr()
    combined.setdiag(0)
    combined.eliminate_zeros()
    return combined


# =============================================================================
# CLUSTERING
# =============================================================================


def normalized_spectral_embedding(
    graph: sp.csr_matrix,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    n = graph.shape[0]
    k = min(max(2, n_components), n - 1)
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degrees, dtype=np.float64)
    np.divide(1.0, np.sqrt(degrees), out=inv_sqrt, where=degrees > 0)
    norm_graph = sp.diags(inv_sqrt) @ graph @ sp.diags(inv_sqrt)
    norm_graph = 0.5 * (norm_graph + norm_graph.T)
    norm_graph = norm_graph + sp.eye(n, format="csr") * 1e-8

    try:
        _, vecs = eigsh(norm_graph, k=k, which="LA")
        embedding = vecs
    except Exception as exc:
        warnings.warn(
            f"eigsh failed ({exc}); using TruncatedSVD fallback.",
            RuntimeWarning,
        )
        embedding = TruncatedSVD(
            n_components=k,
            random_state=random_state,
        ).fit_transform(norm_graph)

    embedding = normalize(embedding, norm="l2", axis=1)
    return np.asarray(embedding, dtype=np.float32)


def spectral_cluster_many_seeds(
    graph: sp.csr_matrix,
    n_domains: int,
    seeds: Sequence[int],
    n_init: int,
    extra_components: int,
) -> Dict[int, np.ndarray]:
    embedding = normalized_spectral_embedding(
        graph,
        n_components=n_domains + extra_components,
        random_state=int(seeds[0]),
    )
    out: Dict[int, np.ndarray] = {}
    for seed in seeds:
        model = KMeans(
            n_clusters=n_domains,
            n_init=n_init,
            random_state=int(seed),
        )
        out[int(seed)] = model.fit_predict(embedding).astype(str)
    return out


def leiden_cluster_many_seeds(
    graph: sp.csr_matrix,
    n_domains: int,
    seeds: Sequence[int],
    resolution_grid: Sequence[float],
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    template = AnnData(X=sp.csr_matrix((graph.shape[0], 1), dtype=np.float32))

    for seed in seeds:
        best_labels = None
        best_delta = math.inf
        best_resolution = None

        for resolution in resolution_grid:
            ad = template.copy()
            sc.tl.leiden(
                ad,
                adjacency=graph,
                resolution=float(resolution),
                random_state=int(seed),
                key_added="domain",
                directed=False,
            )
            labels = ad.obs["domain"].astype(str).to_numpy()
            delta = abs(len(np.unique(labels)) - n_domains)
            if delta < best_delta:
                best_delta = delta
                best_labels = labels
                best_resolution = resolution
            if delta == 0:
                break

        if best_labels is None:
            raise RuntimeError("Leiden clustering failed to produce labels.")
        if best_delta != 0:
            warnings.warn(
                f"Seed {seed}: closest Leiden solution has "
                f"{len(np.unique(best_labels))} clusters at resolution "
                f"{best_resolution}; requested {n_domains}.",
                RuntimeWarning,
            )
        out[int(seed)] = best_labels

    return out


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
    gt_index = {lab: i for i, lab in enumerate(gt_labels)}
    pred_index = {lab: i for i, lab in enumerate(pred_labels)}
    for g, p in zip(gt_use, pred_use):
        contingency[gt_index[g], pred_index[p]] += 1

    row, col = linear_sum_assignment(-contingency)
    return {pred_labels[c]: gt_labels[r] for r, c in zip(row, col)}


def macro_dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    mask = valid_label_mask(gt, pred)
    if mask.sum() == 0:
        return np.nan
    gt_use = np.asarray(gt, dtype=object)[mask].astype(str)
    pred_use = np.asarray(pred, dtype=object)[mask].astype(str)
    mapping = hungarian_mapping(gt_use, pred_use)
    mapped = np.asarray([mapping.get(p, f"unmatched_{p}") for p in pred_use])

    dice = []
    for label in np.unique(gt_use):
        g = gt_use == label
        p = mapped == label
        denom = int(g.sum() + p.sum())
        dice.append(2.0 * np.logical_and(g, p).sum() / denom if denom else np.nan)
    return float(np.nanmean(dice))


def boundary_nodes(labels: np.ndarray, coordinates: np.ndarray, k: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    valid = valid_label_mask(labels)
    n = len(labels)
    out = np.zeros(n, dtype=bool)
    if valid.sum() < 3:
        return out

    coords_valid = np.asarray(coordinates)[valid]
    labels_valid = labels[valid].astype(str)
    n_neighbors = min(k + 1, len(coords_valid))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords_valid)
    _, idx = nn.kneighbors(coords_valid)
    neighbors = idx[:, 1:]
    is_boundary = np.any(labels_valid[neighbors] != labels_valid[:, None], axis=1)
    out[np.where(valid)[0]] = is_boundary
    return out


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

    dist_pred, _ = NearestNeighbors(n_neighbors=1).fit(gt_coords).kneighbors(pred_coords)
    dist_gt, _ = NearestNeighbors(n_neighbors=1).fit(pred_coords).kneighbors(gt_coords)
    precision = float(np.mean(dist_pred[:, 0] <= tolerance))
    recall = float(np.mean(dist_gt[:, 0] <= tolerance))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


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
# PER-CONDITION EVALUATION
# =============================================================================


def choose_representative_seed(metrics: pd.DataFrame) -> int:
    """Choose the seed whose primary ARI is closest to the seed-average ARI."""
    valid = metrics[np.isfinite(metrics["ari_primary"])].copy()
    if valid.empty:
        return int(metrics.iloc[0]["seed"])
    mean_ari = valid["ari_primary"].mean()
    idx = (valid["ari_primary"] - mean_ari).abs().idxmin()
    return int(valid.loc[idx, "seed"])


def evaluate_condition(
    condition: Condition,
    pcs: np.ndarray,
    n_domains: int,
    cfg: Config,
) -> ConditionResult:
    coordinates = np.asarray(condition.adata.obsm[condition.spatial_key], dtype=np.float32)
    graph = combined_expression_spatial_graph(
        pcs=pcs,
        coordinates=coordinates,
        expression_neighbors=cfg.expression_neighbors,
        spatial_neighbors=cfg.spatial_neighbors,
        expression_weight=cfg.expression_graph_weight,
    )

    if cfg.clustering_engine == "spectral":
        predictions = spectral_cluster_many_seeds(
            graph,
            n_domains=n_domains,
            seeds=cfg.random_seeds,
            n_init=cfg.kmeans_n_init,
            extra_components=cfg.spectral_extra_components,
        )
    else:
        predictions = leiden_cluster_many_seeds(
            graph,
            n_domains=n_domains,
            seeds=cfg.random_seeds,
            resolution_grid=cfg.leiden_resolution_grid,
        )

    rows = []
    for seed, pred in predictions.items():
        primary = clustering_metrics(condition.primary_gt, pred, coordinates, cfg)
        row = {
            "condition": condition.name,
            "category": condition.category,
            "seed": int(seed),
            "n_domains_requested": int(n_domains),
            "n_domains_detected": int(len(np.unique(pred))),
            "ari_primary": primary["ari"],
            "nmi_primary": primary["nmi"],
            "macro_dice_primary": primary["macro_dice"],
            "boundary_f1_primary": primary["boundary_f1"],
            "n_eval_primary": primary["n_eval"],
        }

        if condition.reference_gt is not None:
            ref = clustering_metrics(condition.reference_gt, pred, coordinates, cfg)
            row.update(
                {
                    "ari_reference": ref["ari"],
                    "nmi_reference": ref["nmi"],
                    "macro_dice_reference": ref["macro_dice"],
                    "boundary_f1_reference": ref["boundary_f1"],
                }
            )
        else:
            row.update(
                {
                    "ari_reference": np.nan,
                    "nmi_reference": np.nan,
                    "macro_dice_reference": np.nan,
                    "boundary_f1_reference": np.nan,
                }
            )

        if condition.source_gt is not None:
            source = clustering_metrics(condition.source_gt, pred, coordinates, cfg)
            row.update(
                {
                    "ari_source": source["ari"],
                    "nmi_source": source["nmi"],
                    "macro_dice_source": source["macro_dice"],
                    "boundary_f1_source": source["boundary_f1"],
                }
            )
        else:
            row.update(
                {
                    "ari_source": np.nan,
                    "nmi_source": np.nan,
                    "macro_dice_source": np.nan,
                    "boundary_f1_source": np.nan,
                }
            )

        if condition.transfer_distance is not None:
            spacing = condition.transfer_spacing
            normalized_dist = (
                condition.transfer_distance / spacing
                if np.isfinite(spacing) and spacing > 0
                else np.full_like(condition.transfer_distance, np.nan)
            )
            row["reference_transfer_distance_median"] = float(
                np.nanmedian(condition.transfer_distance)
            )
            row["reference_transfer_distance_p95"] = float(
                np.nanpercentile(condition.transfer_distance, 95)
            )
            row["reference_transfer_distance_spot_units_median"] = float(
                np.nanmedian(normalized_dist)
            )
            row["reference_transfer_distance_spot_units_p95"] = float(
                np.nanpercentile(normalized_dist, 95)
            )
        else:
            row["reference_transfer_distance_median"] = np.nan
            row["reference_transfer_distance_p95"] = np.nan
            row["reference_transfer_distance_spot_units_median"] = np.nan
            row["reference_transfer_distance_spot_units_p95"] = np.nan

        rows.append(row)

    metrics = pd.DataFrame(rows)
    representative_seed = choose_representative_seed(metrics)
    representative_prediction = predictions[representative_seed]

    print(
        f"[EVAL] {condition.name:<16s} "
        f"ARI={metrics['ari_primary'].mean():.3f}±{metrics['ari_primary'].std(ddof=1):.3f}"
    )
    return ConditionResult(
        condition=condition,
        coordinates=coordinates,
        predictions_by_seed=predictions,
        metrics=metrics,
        representative_seed=representative_seed,
        representative_prediction=representative_prediction,
    )


def summarize_seed_metrics(seed_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "ari_primary", "nmi_primary", "macro_dice_primary", "boundary_f1_primary",
        "ari_reference", "nmi_reference", "macro_dice_reference", "boundary_f1_reference",
        "ari_source", "nmi_source", "macro_dice_source", "boundary_f1_source",
        "reference_transfer_distance_spot_units_median",
        "reference_transfer_distance_spot_units_p95",
    ]
    rows = []
    for (condition, category), df in seed_df.groupby(["condition", "category"], sort=False):
        row = {"condition": condition, "category": category, "n_seeds": len(df)}
        for metric in metric_cols:
            values = pd.to_numeric(df[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def representative_predictions_dataframe(results: Sequence[ConditionResult]) -> pd.DataFrame:
    frames = []
    for result in results:
        c = result.condition
        pred = result.representative_prediction
        mapping = hungarian_mapping(c.primary_gt, pred)
        pred_mapped = np.asarray([mapping.get(str(x), f"Cluster {x}") for x in pred], dtype=object)
        coords = result.coordinates
        frame = pd.DataFrame(
            {
                "condition": c.name,
                "category": c.category,
                "representative_seed": result.representative_seed,
                "observation_id": c.adata.obs_names.astype(str),
                "source_slice": c.source_slice.astype(str),
                "x": coords[:, 0],
                "y": coords[:, 1],
                "ground_truth": c.primary_gt,
                "predicted_cluster": pred.astype(str),
                "predicted_mapped": pred_mapped,
            }
        )
        if c.reference_gt is not None:
            frame["reference_ground_truth"] = c.reference_gt
        else:
            frame["reference_ground_truth"] = None
        if c.source_gt is not None:
            frame["source_ground_truth"] = c.source_gt
        else:
            frame["source_ground_truth"] = None
        if c.transfer_distance is not None:
            frame["reference_transfer_distance"] = c.transfer_distance
            frame["reference_transfer_distance_spot_units"] = (
                c.transfer_distance / c.transfer_spacing
                if np.isfinite(c.transfer_spacing) and c.transfer_spacing > 0
                else np.nan
            )
        else:
            frame["reference_transfer_distance"] = np.nan
            frame["reference_transfer_distance_spot_units"] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# REPORTING
# =============================================================================


def integrated_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    integrated = summary[summary["category"].isin(["pool", "axis"])].copy()
    integrated["ranking_ari"] = integrated["ari_reference_mean"]
    integrated = integrated.sort_values(
        ["ranking_ari", "macro_dice_reference_mean", "boundary_f1_reference_mean"],
        ascending=False,
    ).reset_index(drop=True)
    integrated["rank"] = np.arange(1, len(integrated) + 1)
    return integrated


def write_report(summary: pd.DataFrame, cfg: Config) -> None:
    ranking = integrated_ranking(summary)
    top = ranking.iloc[0]
    axis_best = str(top["condition"]).startswith("AXIS-")

    lines = [
        "# DLPFC spatial-domain evaluation",
        "",
        f"- Clustering engine: `{cfg.clustering_engine}`",
        f"- Expression/spatial graph weight: {cfg.expression_graph_weight:.2f} / "
        f"{1-cfg.expression_graph_weight:.2f}",
        f"- Random seeds: {len(cfg.random_seeds)}",
        f"- Feature genes: {cfg.n_top_genes} requested; fixed across conditions",
        "- Single slices: evaluated against their own manual domains",
        "- Integrated conditions: evaluated against slice-0 domains transferred "
        "to the corresponding coordinate system",
        "",
        "## Integrated-condition ranking",
        "",
        "| Rank | Condition | Reference ARI | Macro Dice | Boundary F1 |",
        "|---:|---|---:|---:|---:|",
    ]
    for _, row in ranking.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['condition']} | "
            f"{row['ari_reference_mean']:.3f} ± {row['ari_reference_sd']:.3f} | "
            f"{row['macro_dice_reference_mean']:.3f} | "
            f"{row['boundary_f1_reference_mean']:.3f} |"
        )

    lines.extend(["", "## Objective conclusion", ""])
    if axis_best:
        lines.append(
            f"The top integrated condition is **{top['condition']}** with mean "
            f"reference ARI {top['ari_reference_mean']:.3f}. Under this fixed "
            "benchmark, an AXIS atlas is ranked first."
        )
    else:
        best_axis = ranking[ranking["condition"].str.startswith("AXIS-")].iloc[0]
        lines.append(
            f"The top integrated condition is **{top['condition']}**, not AXIS. "
            f"The best AXIS condition is **{best_axis['condition']}** with mean "
            f"reference ARI {best_axis['ari_reference_mean']:.3f}. The current "
            "results do not support claiming that AXIS is best; inspect feature "
            "scaling, registration, graph construction, and aggregation behavior."
        )

    lines.extend(
        [
            "",
            "## Interpretation constraint",
            "",
            "Atlas-space superiority does not by itself prove that atlas labels "
            "will remain superior after projection to native slices. Projection "
            "adds deformation, interpolation, and boundary-transfer errors and "
            "must be evaluated separately.",
        ]
    )
    cfg.report_md.write_text("\n".join(lines))


# =============================================================================
# PUBLICATION-STYLE PLOTS
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


def condition_order(summary: pd.DataFrame) -> List[str]:
    preferred = [
        "Single 0", "Single 1", "Single 2", "Single 3",
        "Pool (raw)", "Pool (warped)",
        "AXIS-mean", "AXIS-median", "AXIS-max",
    ]
    observed = set(summary["condition"].astype(str))
    return [x for x in preferred if x in observed] + sorted(observed.difference(preferred))


def condition_colors(order: Sequence[str]) -> Dict[str, str]:
    out = {}
    for name in order:
        if name.startswith("Single"):
            out[name] = "#BDBDBD"
        elif name == "Pool (raw)":
            out[name] = "#9ECAE1"
        elif name == "Pool (warped)":
            out[name] = "#3182BD"
        elif name == "AXIS-mean":
            out[name] = "#E6550D"
        elif name == "AXIS-median":
            out[name] = "#FD8D3C"
        elif name == "AXIS-max":
            out[name] = "#A63603"
        else:
            out[name] = "#777777"
    return out


def save_figure(fig: plt.Figure, stem: Path, cfg: Config) -> None:
    for ext in cfg.save_formats:
        path = stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight"}
        if ext.lower() == "png":
            kwargs["dpi"] = cfg.png_dpi
        fig.savefig(path, **kwargs)
        print(f"[PLOT] saved: {path}")
    plt.close(fig)


def plot_ari_summary(seed_df: pd.DataFrame, summary: pd.DataFrame, cfg: Config) -> None:
    set_publication_style()
    order = condition_order(summary)
    colors = condition_colors(order)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    rng = np.random.default_rng(0)
    for i, name in enumerate(order):
        vals = seed_df.loc[seed_df["condition"] == name, "ari_primary"].dropna().to_numpy()
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=13,
            color=colors[name],
            alpha=0.55,
            linewidths=0,
            zorder=2,
        )
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ax.errorbar(
            i,
            mean,
            yerr=sd,
            fmt="o",
            markersize=5,
            color="black",
            markerfacecolor=colors[name],
            capsize=2.5,
            linewidth=1.0,
            zorder=3,
        )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Adjusted Rand index")
    ax.set_ylim(-0.05, 1.02)
    ax.set_title("Spatial-domain recovery")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    fig.tight_layout()
    save_figure(fig, cfg.outdir / "domain_ari_summary", cfg)


def layer_palette(layers: Sequence[str]) -> Dict[str, str]:
    base = [
        "#4E79A7", "#59A14F", "#F28E2B", "#E15759",
        "#B07AA1", "#76B7B2", "#9C755F", "#BAB0AC",
    ]
    return {layer: base[i % len(base)] for i, layer in enumerate(layers)}


def plot_domain_maps(pred_df: pd.DataFrame, summary: pd.DataFrame, cfg: Config) -> None:
    set_publication_style()
    order = condition_order(summary)
    layers = canonical_layer_order(pred_df["ground_truth"], cfg.preferred_layer_order)
    palette = layer_palette(layers)

    ncol = 3
    nrow = int(math.ceil(len(order) / ncol))
    for value_col, suffix, title_prefix in [
        ("ground_truth", "ground_truth", "Ground truth"),
        ("predicted_mapped", "predicted", "Detected domains"),
    ]:
        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(3.0 * ncol, 2.7 * nrow),
            squeeze=False,
        )
        axes_flat = axes.ravel()
        rng = np.random.default_rng(0)

        for ax, name in zip(axes_flat, order):
            sub = pred_df[pred_df["condition"] == name].copy()
            if len(sub) > cfg.max_points_per_map:
                take = rng.choice(len(sub), cfg.max_points_per_map, replace=False)
                sub = sub.iloc[take]

            labels = sub[value_col].astype(str)
            point_colors = [palette.get(x, "#D0D0D0") for x in labels]
            size = max(0.5, min(7.0, 18000.0 / max(len(sub), 1)))
            ax.scatter(
                sub["x"],
                sub["y"],
                c=point_colors,
                s=size,
                linewidths=0,
                rasterized=True,
            )
            ari_row = summary.loc[summary["condition"] == name]
            ari = float(ari_row["ari_primary_mean"].iloc[0]) if len(ari_row) else np.nan
            ax.set_title(f"{name}\nARI = {ari:.3f}")
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.axis("off")

        for ax in axes_flat[len(order):]:
            ax.axis("off")

        handles = [
            mpl.lines.Line2D(
                [0], [0], marker="o", linestyle="none", markersize=5,
                markerfacecolor=palette[layer], markeredgewidth=0, label=layer,
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
        fig.suptitle(title_prefix, y=0.995, fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        save_figure(fig, cfg.outdir / f"domain_maps_{suffix}", cfg)


def plot_integrated_metric_comparison(summary: pd.DataFrame, cfg: Config) -> None:
    set_publication_style()
    data = summary[summary["category"].isin(["pool", "axis"])].copy()
    order = condition_order(data)
    data["condition"] = pd.Categorical(data["condition"], categories=order, ordered=True)
    data = data.sort_values("condition")
    colors = condition_colors(order)

    metrics = [
        ("ari_reference_mean", "ari_reference_sd", "ARI"),
        ("macro_dice_reference_mean", "macro_dice_reference_sd", "Macro Dice"),
        ("boundary_f1_reference_mean", "boundary_f1_reference_sd", "Boundary F1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    x = np.arange(len(data))

    for ax, (mean_col, sd_col, label) in zip(axes, metrics):
        ax.bar(
            x,
            data[mean_col],
            yerr=data[sd_col],
            color=[colors[str(c)] for c in data["condition"]],
            edgecolor="black",
            linewidth=0.6,
            capsize=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(data["condition"].astype(str), rotation=40, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)

    fig.suptitle("Reference-anchored integrated-domain performance", y=1.02, fontsize=10)
    fig.tight_layout()
    save_figure(fig, cfg.outdir / "integrated_domain_metric_comparison", cfg)


# =============================================================================
# EVALUATION AND PLOTTING WORKFLOWS
# =============================================================================


def run_evaluation(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    singles = load_single_slices(cfg)
    aggregate_paths = build_axis_aggregates(cfg)
    axis_ads = load_axis_aggregates(aggregate_paths, cfg)

    common_genes = common_gene_order(singles, axis_ads, cfg.reference_slice_id)
    singles, axis_ads = subset_all_to_common(singles, axis_ads, common_genes)
    feature_genes = select_fixed_feature_genes(singles, common_genes, cfg)
    cfg.feature_genes_txt.write_text("\n".join(feature_genes) + "\n")

    ref_labels = clean_labels(
        singles[cfg.reference_slice_id].obs[cfg.layer_key].to_numpy()
    )
    valid_ref = ref_labels[valid_label_mask(ref_labels)]
    n_domains = cfg.n_domains or len(np.unique(valid_ref.astype(str)))
    print(f"[DOMAIN] requested number of domains: {n_domains}")

    conditions = build_conditions(singles, axis_ads, cfg)

    # Cache expression PCs. Pool raw and Pool warped intentionally share the
    # same expression embedding; only their spatial graphs differ.
    pcs_cache: Dict[str, np.ndarray] = {}
    results: List[ConditionResult] = []

    for condition in conditions:
        if condition.expression_group not in pcs_cache:
            pcs = compute_expression_pcs(
                condition.adata,
                genes=feature_genes,
                n_pcs=cfg.n_expression_pcs,
                random_state=0,
            )
            if (
                condition.category == "pool"
                and cfg.center_pooled_pcs_by_source_slice
            ):
                pcs = center_pcs_within_batch(pcs, condition.source_slice)
            pcs_cache[condition.expression_group] = pcs
        results.append(
            evaluate_condition(
                condition,
                pcs=pcs_cache[condition.expression_group],
                n_domains=n_domains,
                cfg=cfg,
            )
        )

    seed_df = pd.concat([r.metrics for r in results], ignore_index=True)
    summary = summarize_seed_metrics(seed_df)
    pred_df = representative_predictions_dataframe(results)

    seed_df.to_csv(cfg.seed_metrics_csv, index=False)
    summary.to_csv(cfg.summary_csv, index=False)
    pred_df.to_csv(cfg.representative_predictions_csv, index=False, compression="gzip")

    config_payload = {
        "run_mode": cfg.run_mode,
        "group_id": cfg.group_id,
        "sample_ids": list(cfg.sample_ids),
        "reference_slice_id": cfg.reference_slice_id,
        "reference_axis_value": cfg.reference_axis_value,
        "axis_aggregation_methods": list(cfg.axis_aggregation_methods),
        "n_feature_genes": len(feature_genes),
        "n_domains": n_domains,
        "clustering_engine": cfg.clustering_engine,
        "random_seeds": list(cfg.random_seeds),
        "expression_graph_weight": cfg.expression_graph_weight,
        "expression_neighbors": cfg.expression_neighbors,
        "spatial_neighbors": cfg.spatial_neighbors,
        "exclude_known_markers_from_clustering": cfg.exclude_known_markers_from_clustering,
        "heldout_marker_genes": list(cfg.heldout_marker_genes),
    }
    cfg.config_json.write_text(json.dumps(config_payload, indent=2))
    write_report(summary, cfg)

    print(f"[SAVE] {cfg.seed_metrics_csv}")
    print(f"[SAVE] {cfg.summary_csv}")
    print(f"[SAVE] {cfg.representative_predictions_csv}")
    print(f"[SAVE] {cfg.report_md}")

    ranking = integrated_ranking(summary)
    print("\n[INTEGRATED RANKING]")
    print(
        ranking[
            [
                "rank", "condition", "ari_reference_mean",
                "macro_dice_reference_mean", "boundary_f1_reference_mean",
            ]
        ].to_string(index=False)
    )
    return seed_df, summary, pred_df


def load_saved_results(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [
        cfg.seed_metrics_csv,
        cfg.summary_csv,
        cfg.representative_predictions_csv,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing saved result {path}. Run with RUN_MODE='eval' or 'both' first."
            )
    seed_df = pd.read_csv(cfg.seed_metrics_csv)
    summary = pd.read_csv(cfg.summary_csv)
    pred_df = pd.read_csv(cfg.representative_predictions_csv)
    return seed_df, summary, pred_df


def run_plotting(
    seed_df: pd.DataFrame,
    summary: pd.DataFrame,
    pred_df: pd.DataFrame,
    cfg: Config,
) -> None:
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    plot_ari_summary(seed_df, summary, cfg)
    plot_integrated_metric_comparison(summary, cfg)
    plot_domain_maps(pred_df, summary, cfg)


def main() -> None:
    mode = validate_config(CFG)
    if mode in {"eval", "both"}:
        seed_df, summary, pred_df = run_evaluation(CFG)
    else:
        seed_df, summary, pred_df = load_saved_results(CFG)

    if mode in {"plot", "both"}:
        run_plotting(seed_df, summary, pred_df, CFG)


if __name__ == "__main__":
    main()
