"""Evidence-weighted generation on a user-defined support (Methods, Eqs. (33)-(35)).

Molecular measurements and continuous annotations are generated as weighted
averages of the neighbouring evidence,

``y^(u) = sum_N(u) w_ri(u) y_ri / (sum_N(u) w_ri(u) + eps)``,

and categorical annotations by their weighted support,

``p(c | u) = sum_N(u) w_ri(u) 1(y_ri = c) / (sum_N(u) w_ri(u) + eps)``,

assigning the category with the largest support. Molecular values and
annotations are generated independently on the same support.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from .weighting import compute_qc_covariates, prepare_local_query_support

__all__ = [
    "infer_attr_schema_from_df",
    "generate_gene_sets",
    "generate_attr_sets",
    "generate_virtual_tissue_fixed_grid",
]

_LABEL_HINTS = ("label", "layer", "cluster", "domain", "celltype", "type", "guess", "region")


def infer_attr_schema_from_df(df: pd.DataFrame) -> Dict[str, str]:
    """Guess ``"categorical"``, ``"continuous"`` or ``"boolean"`` for every column."""
    schema = {}
    for col in df.columns:
        s = df[col]
        name = str(col).lower()
        if pd.api.types.is_bool_dtype(s):
            schema[col] = "boolean"
        elif isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            schema[col] = "categorical"
        elif pd.api.types.is_numeric_dtype(s):
            schema[col] = "categorical" if any(k in name for k in _LABEL_HINTS) else "continuous"
        else:
            schema[col] = "categorical"
    return schema


def _extract_dense_rows(X, row_idx, dtype=np.float32) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X[row_idx].toarray(), dtype=dtype)
    return np.asarray(X[row_idx], dtype=dtype)


def generate_gene_sets(grid_pts, X, support, fill_value=0.0, chunk_size=64, dtype=np.float32) -> np.ndarray:
    """Weighted average of the evidence molecular profiles at every active location."""
    n_grid, n_genes = grid_pts.shape[0], X.shape[1]
    q_active, nn_active, local_w = support["query_idx_active"], support["nn_idx_active"], support["local_w"]
    Y = np.full((n_grid, n_genes), fill_value, dtype=dtype)
    for start in range(0, q_active.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), q_active.shape[0])
        idx_block = nn_active[start:end]
        b, k = idx_block.shape
        X_block = np.maximum(_extract_dense_rows(X, idx_block.reshape(-1), dtype=dtype).reshape(b, k, n_genes), 0.0)
        Y[q_active[start:end]] = np.einsum("bk,bkg->bg", local_w[start:end], X_block, optimize=True).astype(dtype, copy=False)
    return Y


def generate_attr_sets(ad: AnnData, grid_pts, support, attrs, attr_schema=None, dtype=np.float32) -> AnnData:
    """Transfer annotations (categorical / continuous / boolean) onto ``ad`` in place."""
    if attrs is None:
        return ad
    if not isinstance(attrs, pd.DataFrame):
        attrs = pd.DataFrame(attrs)
    if attr_schema is None:
        attr_schema = infer_attr_schema_from_df(attrs)

    n_grid = grid_pts.shape[0]
    q_active, nn_active, local_w = support["query_idx_active"], support["nn_idx_active"], support["local_w"]
    for col, kind in attr_schema.items():
        values = attrs[col].values
        if kind == "continuous":
            out = np.full((n_grid,), np.nan, dtype=dtype)
            if q_active.size > 0:
                out[q_active] = np.sum(local_w * np.asarray(values, dtype=dtype)[nn_active], axis=1).astype(dtype, copy=False)
            ad.obs[col] = out
        elif kind == "categorical":
            codes, categories = pd.factorize(values, sort=True)
            probs = np.zeros((n_grid, len(categories)), dtype=dtype)
            labels = np.full((n_grid,), None, dtype=object)
            if q_active.size > 0:
                rows = np.repeat(np.arange(q_active.shape[0]), nn_active.shape[1])
                probs_active = np.zeros((q_active.shape[0], len(categories)), dtype=dtype)
                np.add.at(probs_active, (rows, codes[nn_active].reshape(-1)), local_w.reshape(-1))
                probs[q_active] = probs_active
                labels[q_active] = categories[np.argmax(probs_active, axis=1)]
            ad.obs[col] = pd.Categorical(labels, categories=list(categories))
            ad.obsm[f"{col}_probs"] = probs
        elif kind == "boolean":
            v = np.asarray(values).astype(np.float32)
            p_true = np.full((n_grid,), np.nan, dtype=dtype)
            label = np.zeros((n_grid,), dtype=bool)
            if q_active.size > 0:
                pred = np.sum(local_w * v[nn_active], axis=1).astype(dtype, copy=False)
                p_true[q_active] = pred
                label[q_active] = pred >= 0.5
            ad.obs[f"{col}_prob_true"] = p_true
            ad.obs[col] = label
            p = np.nan_to_num(p_true, nan=0.0)
            ad.obsm[f"{col}_probs"] = np.stack([1.0 - p, p], axis=1)
            ad.uns[f"{col}_categories"] = [False, True]
        else:
            raise ValueError(f"Unsupported attr kind for '{col}': {kind}")
    return ad


def generate_virtual_tissue_fixed_grid(
    coords,
    X,
    grid_pts,
    var_names=None,
    attrs=None,
    attr_schema: Optional[Mapping[str, str]] = None,
    grid_support_mask=None,
    point_weights=None,
    spatial_bandwidth: Optional[float] = None,
    attr_spatial_bandwidth: Optional[float] = None,
    use_gene_weight: bool = True,
    knn: int = 48,
    min_neighbors: int = 6,
    radius: Optional[float] = None,
    chunk_size: int = 64,
    fill_value: float = 0.0,
    dtype=np.float32,
) -> AnnData:
    """Generate molecular profiles and annotations on ``grid_pts`` from an evidence plane.

    Parameters
    ----------
    coords, X
        Evidence coordinates ``(N, 2)`` in the query frame and molecular values ``(N, G)``.
    grid_pts
        Output locations ``(M, 2)``.
    attrs
        Optional evidence annotations (DataFrame with ``N`` rows).
    attr_schema
        Kind of each annotation (``"categorical"``, ``"continuous"`` or
        ``"boolean"``). Inferred from the column dtypes when omitted; numeric
        columns whose names contain e.g. ``cluster`` or ``domain`` count as
        categorical.
    grid_support_mask
        Tissue-support filter of the output locations.
    point_weights
        Per-evidence biological-axis weights.
    spatial_bandwidth, attr_spatial_bandwidth
        ``sigma_x`` for molecular values and (optionally separately) annotations.
    use_gene_weight
        Apply the molecular-quality weight (requires count-like ``X``).

    Returns
    -------
    anndata.AnnData
        ``X`` ``(M, G)``, ``obsm["spatial"] = grid_pts`` and the support
        diagnostics ``obs["valid"]``, ``obs["active_query_support"]``,
        ``obs["confidence"]``, ``obs["nearest_dist"]``.
    """
    if X.shape[0] != coords.shape[0]:
        raise ValueError("X rows must match coords rows.")
    grid_pts = np.asarray(grid_pts, dtype=dtype)
    qc_total, qc_genes = compute_qc_covariates(X) if use_gene_weight else (None, None)

    common = dict(
        coords=coords,
        grid_pts=grid_pts,
        grid_support_mask=grid_support_mask,
        point_weights=point_weights,
        qc_total_counts=qc_total,
        qc_n_genes=qc_genes,
        knn=knn,
        min_neighbors=min_neighbors,
        radius=radius,
        dtype=dtype,
    )
    support = prepare_local_query_support(spatial_bandwidth=spatial_bandwidth, **common)
    Y = generate_gene_sets(grid_pts, X, support, fill_value, chunk_size, dtype)

    ad = AnnData(X=Y)
    if var_names is not None:
        ad.var_names = np.asarray(var_names).astype(str)
    ad.obsm["spatial"] = grid_pts
    ad.obs["valid"] = support["valid_mask"]
    ad.obs["active_query_support"] = support["active_query_support"]
    ad.obs["confidence"] = support["confidence"]
    ad.obs["nearest_dist"] = support["nearest_dist"]

    attr_support = (
        prepare_local_query_support(spatial_bandwidth=attr_spatial_bandwidth, **common) if attr_spatial_bandwidth else support
    )
    ad = generate_attr_sets(ad, grid_pts, attr_support, attrs, attr_schema, dtype)

    ad.uns["spatial_bandwidth"] = float(support["spatial_bandwidth"])
    if attr_spatial_bandwidth is not None:
        ad.uns["attr_spatial_bandwidth"] = attr_spatial_bandwidth
    return ad
