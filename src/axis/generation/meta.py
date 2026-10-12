"""Saving the evidence plane behind a generated tissue for inspection and plotting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

__all__ = ["save_generation_meta"]


def _get_attribute_names(
    attr_schema: Optional[Any],
    n_attributes: int,
) -> Sequence[str]:
    """
    Derive stable attribute names from ATTRIBUTE_SCHEMA when possible.
    """
    if isinstance(attr_schema, dict) and len(attr_schema) == n_attributes:
        return [str(name) for name in attr_schema.keys()]

    if isinstance(attr_schema, (list, tuple)) and len(attr_schema) == n_attributes:
        names = []

        for i, item in enumerate(attr_schema):
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict) and "name" in item:
                names.append(str(item["name"]))
            else:
                names.append(f"attribute_{i}")

        return names

    return [f"attribute_{i}" for i in range(n_attributes)]


def _store_evidence_attributes(
    adata: ad.AnnData,
    attrs_cat: Optional[Union[np.ndarray, pd.DataFrame]],
    attr_schema: Optional[Any],
) -> None:
    """
    Store evidence attributes without using object arrays.

    Numeric matrix:
        adata.obsm["attributes"]

    String/categorical matrix:
        one column per attribute in adata.obs
    """
    if attrs_cat is None:
        return

    if isinstance(attrs_cat, pd.DataFrame):
        attrs_df = attrs_cat.reset_index(drop=True).copy()

        if len(attrs_df) != adata.n_obs:
            raise ValueError(f"attrs_cat row count does not match the evidence count: {len(attrs_df)} versus {adata.n_obs}.")

        for column in attrs_df.columns:
            values = attrs_df[column]

            if pd.api.types.is_numeric_dtype(values):
                adata.obs[f"attr__{column}"] = values.to_numpy()
            else:
                adata.obs[f"attr__{column}"] = values.astype(str).to_numpy()

        return

    attrs = np.asarray(attrs_cat)

    if attrs.ndim == 1:
        attrs = attrs[:, None]

    if attrs.ndim != 2:
        raise ValueError(f"attrs_cat must be a one- or two-dimensional array, but received shape {attrs.shape}.")

    if attrs.shape[0] != adata.n_obs:
        raise ValueError(f"attrs_cat row count does not match the evidence count: {attrs.shape[0]} versus {adata.n_obs}.")

    attribute_names = _get_attribute_names(
        attr_schema=attr_schema,
        n_attributes=attrs.shape[1],
    )

    if np.issubdtype(attrs.dtype, np.number) or attrs.dtype == bool:
        adata.obsm["attributes"] = attrs.astype(np.float32, copy=False)
        adata.uns["attribute_names"] = list(attribute_names)
    else:
        for j, attribute_name in enumerate(attribute_names):
            adata.obs[f"attr__{attribute_name}"] = attrs[:, j].astype(str)


def save_generation_meta(
    virtual_tissue_path: Union[str, Path],
    *,
    grid_pts: np.ndarray,
    grid_support_mask: np.ndarray,
    coords_cat: np.ndarray,
    X_cat,
    genes: Sequence[str],
    attrs_cat: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    attr_schema: Optional[Any] = None,
    point_weights: Optional[np.ndarray] = None,
    source_anchor_ids: Optional[Sequence[int]] = None,
    source_point_counts: Optional[Sequence[int]] = None,
    source_axis_positions: Optional[Sequence[float]] = None,
    source_weights: Optional[Sequence[float]] = None,
    biological_axis_name: Optional[str] = None,
    query_axis_value: Optional[float] = None,
    relative_position_alpha: Optional[float] = None,
    mode: Optional[str] = None,
    apply_log1p_before_generation: Optional[bool] = None,
    axis_weight_bandwidth: Optional[float] = None,
    spatial_bandwidth: Optional[float] = None,
    knn: Optional[int] = None,
    min_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
) -> Path:
    """
    Save intermediate virtual-tissue generation evidence for visualization.

    The metadata H5AD contains
    --------------------------
    X
        Concatenated evidence expression matrix.

    obsm["spatial"]
        Concatenated evidence coordinates in the query spatial plane.

    obs["point_weight"]
        Final evidence weight used during spatial interpolation.

    obs["source_anchor_id"]
        Anchor from which each evidence point originated, when source
        counts are provided.

    obsm["attributes"] or obs["attr__*"]
        Concatenated attribute evidence.

    uns["grid_pts"]
        Complete user-defined query grid.

    uns["grid_support_mask"]
        Boolean mask indicating supported query-grid locations.

    uns["virtual_tissue_file"]
        Path to the final generated virtual-tissue H5AD.
    """
    virtual_tissue_path = Path(virtual_tissue_path)

    if virtual_tissue_path.suffix.lower() != ".h5ad":
        raise ValueError(f"virtual_tissue_path must end with '.h5ad', received: {virtual_tissue_path}")

    coords_cat = np.asarray(coords_cat, dtype=np.float32)
    grid_pts = np.asarray(grid_pts, dtype=np.float32)
    grid_support_mask = np.asarray(
        grid_support_mask,
        dtype=bool,
    ).reshape(-1)

    if coords_cat.ndim != 2 or coords_cat.shape[1] != 2:
        raise ValueError(f"coords_cat must have shape [N, 2], got {coords_cat.shape}.")

    if grid_pts.ndim != 2 or grid_pts.shape[1] != 2:
        raise ValueError(f"grid_pts must have shape [M, 2], got {grid_pts.shape}.")

    if len(grid_support_mask) != len(grid_pts):
        raise ValueError(f"grid_support_mask length does not match grid_pts: {len(grid_support_mask)} versus {len(grid_pts)}.")

    if not hasattr(X_cat, "shape") or len(X_cat.shape) != 2:
        raise ValueError("X_cat must be a two-dimensional matrix.")

    n_evidence, n_genes = X_cat.shape

    if n_evidence != len(coords_cat):
        raise ValueError(
            f"X_cat and coords_cat contain different numbers of evidence points: {n_evidence} versus {len(coords_cat)}."
        )

    genes = np.asarray(genes, dtype=str)

    if len(genes) != n_genes:
        raise ValueError(f"The number of genes does not match X_cat columns: {len(genes)} versus {n_genes}.")

    if point_weights is None:
        point_weights = np.ones(n_evidence, dtype=np.float32)
    else:
        point_weights = np.asarray(
            point_weights,
            dtype=np.float32,
        ).reshape(-1)

        if len(point_weights) != n_evidence:
            raise ValueError(f"point_weights length does not match the evidence count: {len(point_weights)} versus {n_evidence}.")

    obs = pd.DataFrame(
        index=pd.Index(
            [f"evidence_{i:07d}" for i in range(n_evidence)],
            name="evidence_id",
        )
    )

    obs["point_weight"] = point_weights

    if source_anchor_ids is not None or source_point_counts is not None:
        if source_anchor_ids is None or source_point_counts is None:
            raise ValueError("source_anchor_ids and source_point_counts must be supplied together.")

        source_anchor_ids = list(source_anchor_ids)
        source_point_counts = np.asarray(
            source_point_counts,
            dtype=int,
        ).reshape(-1)

        if len(source_anchor_ids) != len(source_point_counts):
            raise ValueError("source_anchor_ids and source_point_counts must have the same length.")

        if source_point_counts.sum() != n_evidence:
            raise ValueError(
                "source_point_counts do not sum to the number of evidence "
                f"points: {source_point_counts.sum()} versus {n_evidence}. "
                "This indicates that build_query_evidence changed the number "
                "or ordering of source points."
            )

        obs["source_anchor_id"] = np.concatenate(
            [
                np.repeat(str(anchor_id), count)
                for anchor_id, count in zip(
                    source_anchor_ids,
                    source_point_counts,
                )
            ]
        )

        obs["source_point_index"] = np.concatenate([np.arange(count, dtype=int) for count in source_point_counts])

    if sparse.issparse(X_cat):
        X_save = X_cat.copy()
    else:
        X_save = np.asarray(X_cat, dtype=np.float32)

    evidence_adata = ad.AnnData(
        X=X_save,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )

    evidence_adata.obsm["spatial"] = coords_cat

    _store_evidence_attributes(
        adata=evidence_adata,
        attrs_cat=attrs_cat,
        attr_schema=attr_schema,
    )

    # User-defined spatial support
    evidence_adata.uns["grid_pts"] = grid_pts
    evidence_adata.uns["grid_support_mask"] = grid_support_mask
    evidence_adata.uns["supported_grid_pts"] = grid_pts[grid_support_mask]

    # Link to the final generated tissue instead of duplicating its matrix.
    evidence_adata.uns["virtual_tissue_file"] = str(virtual_tissue_path.resolve())
    evidence_adata.uns["generation_meta_version"] = "1.0"

    if attr_schema is not None:
        evidence_adata.uns["attribute_schema_json"] = json.dumps(
            attr_schema,
            default=str,
        )

    optional_uns: Dict[str, Any] = {
        "biological_axis_name": biological_axis_name,
        "query_axis_value": query_axis_value,
        "relative_position_alpha": relative_position_alpha,
        "mode": mode,
        "apply_log1p_before_generation": apply_log1p_before_generation,
        "axis_weight_bandwidth": axis_weight_bandwidth,
        "spatial_bandwidth": spatial_bandwidth,
        "knn": knn,
        "min_neighbors": min_neighbors,
        "radius": radius,
    }

    for key, value in optional_uns.items():
        if value is not None:
            evidence_adata.uns[key] = value

    if source_anchor_ids is not None:
        evidence_adata.uns["source_anchor_ids"] = [int(x) for x in source_anchor_ids]

    if source_axis_positions is not None:
        evidence_adata.uns["source_axis_positions"] = np.asarray(
            source_axis_positions,
            dtype=np.float32,
        )

    if source_weights is not None:
        evidence_adata.uns["source_weights"] = np.asarray(
            source_weights,
            dtype=np.float32,
        )

    meta_path = virtual_tissue_path.with_name(f"{virtual_tissue_path.stem}_generation_meta.h5ad")

    meta_path.parent.mkdir(parents=True, exist_ok=True)

    evidence_adata.write_h5ad(
        meta_path,
        compression="gzip",
    )

    return meta_path
