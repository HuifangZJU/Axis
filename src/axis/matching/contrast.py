"""Location-wise molecular contrasts (Methods, "Location-wise molecular contrasts").

Shared preprocessing and spatial-delta utilities, used for the AD-WT disease
contrasts and the temporal remodeling analysis of the mouse-brain data.

Canonical preprocessing
-----------------------
For raw/count-like h5ad matrices, the recommended analysis space is:

    raw counts
        -> per-location library-size normalization
        -> optional log1p
        -> subtraction / downstream analysis

For a matched disease/normal virtual-tissue pair and gene g at location i:

    normalize_total=True, use_log1p=True:
        D'_ig = log1p(D_ig / sum_g(D_ig) * target_sum)
        N'_ig = log1p(N_ig / sum_g(N_ig) * target_sum)
        delta_ig = D'_ig - N'_ig

    normalize_total=True, use_log1p=False:
        delta_ig = normalized(D_ig) - normalized(N_ig)

    normalize_total=False, use_log1p=True:
        delta_ig = log1p(D_ig) - log1p(N_ig)

    normalize_total=False, use_log1p=False:
        delta_ig = D_ig - N_ig

Important implementation detail
-------------------------------
Library-size factors are ALWAYS calculated from the complete gene library in
adata.X for each location, even when the caller requests only a subset of genes.
This avoids the common mistake of normalizing using only selected marker genes.

The returned spatial-delta AnnData contains matched locations (and, by default,
only locations valid in BOTH disease and normal), allowing direct spatial maps
and location-level analyses in one consistent value space.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


@dataclass
class SpatialDeltaResult:
    """Container returned when alignment details are requested."""

    adata: ad.AnnData
    disease_indices: np.ndarray
    normal_indices: np.ndarray
    genes: List[str]
    normalize_total: bool
    target_sum: float
    use_log1p: bool
    matched_count: int
    common_valid_count: int


def get_valid_mask(
    adata: ad.AnnData,
    valid_key: str = "valid",
) -> np.ndarray:
    """Return a robust boolean validity mask; a missing key means all-valid."""
    if valid_key not in adata.obs.columns:
        return np.ones(adata.n_obs, dtype=bool)

    values = adata.obs[valid_key].to_numpy()

    if pd.api.types.is_bool_dtype(values):
        return np.asarray(values, dtype=bool)

    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        numeric = np.asarray(arr, dtype=float)
        return np.isfinite(numeric) & (numeric != 0)

    string_values = arr.astype(str)
    string_values = np.char.lower(np.char.strip(string_values))
    return np.isin(string_values, ["true", "1", "yes", "y", "t"])


def _ordered_common_genes(
    disease: ad.AnnData,
    normal: ad.AnnData,
    genes: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return common genes while preserving requested or disease order."""
    normal_set = set(normal.var_names.astype(str))

    if genes is None:
        ordered = [str(gene) for gene in disease.var_names.astype(str) if str(gene) in normal_set]
    else:
        disease_set = set(disease.var_names.astype(str))
        ordered = [str(gene) for gene in genes if str(gene) in disease_set and str(gene) in normal_set]

    if not ordered:
        raise ValueError("Disease and normal AnnData share no requested genes.")

    return ordered


def _validate_nonnegative(matrix, label: str) -> None:
    """Raise clearly when count preprocessing is requested on negative values."""
    if sp.issparse(matrix):
        values = matrix.data
    else:
        values = np.asarray(matrix)

    finite = np.isfinite(values)
    if np.any(values[finite] < 0):
        minimum = float(np.min(values[finite]))
        raise ValueError(
            f"{label} contains negative values (min={minimum:.6g}). "
            "Library-size normalization/log1p expects nonnegative count-like "
            "expression before subtraction."
        )


def _row_library_sizes(
    adata: ad.AnnData,
    row_indices: np.ndarray,
) -> np.ndarray:
    """Compute per-location library sizes from ALL genes in adata.X."""
    full_rows = adata.X[row_indices]
    _validate_nonnegative(full_rows, "adata.X")

    if sp.issparse(full_rows):
        totals = np.asarray(full_rows.sum(axis=1)).ravel()
    else:
        totals = np.asarray(full_rows, dtype=np.float64).sum(axis=1)

    totals = np.asarray(totals, dtype=np.float64)
    if np.any(~np.isfinite(totals)):
        raise ValueError("Non-finite library sizes found in adata.X.")

    return totals


def _extract_raw_matrix(
    adata: ad.AnnData,
    row_indices: np.ndarray,
    genes: Sequence[str],
) -> np.ndarray:
    """Extract selected raw genes as dense float32 values."""
    gene_indices = adata.var_names.get_indexer(list(genes))
    if np.any(gene_indices < 0):
        missing = [gene for gene, index in zip(genes, gene_indices) if index < 0]
        raise KeyError(f"Missing genes: {missing[:5]}")

    matrix = adata.X[row_indices][:, gene_indices]
    _validate_nonnegative(matrix, "Selected expression matrix")

    if sp.issparse(matrix):
        matrix = matrix.toarray()

    return np.asarray(matrix, dtype=np.float32)


def transform_expression(
    matrix: np.ndarray,
    *,
    normalize_total: bool = False,
    target_sum: float = 1e4,
    use_log1p: bool = False,
    label: str = "Expression matrix",
) -> np.ndarray:
    """
    Preprocess an in-memory COMPLETE expression matrix.

    This helper is appropriate only when ``matrix`` contains the complete gene
    library used to define each row's library size. For selected-gene extraction
    from AnnData, use :func:`extract_preprocessed_matrix` instead.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    _validate_nonnegative(matrix, label)

    out = matrix.astype(np.float32, copy=True)

    if normalize_total:
        if not np.isfinite(target_sum) or target_sum <= 0:
            raise ValueError("target_sum must be a positive finite number.")

        totals = np.sum(out, axis=1, dtype=np.float64)
        factors = np.zeros_like(totals, dtype=np.float64)
        positive = totals > 0
        factors[positive] = float(target_sum) / totals[positive]
        out *= factors[:, None].astype(np.float32)

    if use_log1p:
        out = np.log1p(out).astype(np.float32)

    return out


def extract_preprocessed_matrix(
    adata: ad.AnnData,
    *,
    row_indices: Optional[Sequence[int]] = None,
    genes: Optional[Sequence[str]] = None,
    normalize_total: bool = True,
    target_sum: float = 1e4,
    use_log1p: bool = True,
) -> np.ndarray:
    """
    Extract selected genes after consistent count preprocessing.

    Crucially, library sizes are calculated from ALL genes in ``adata.X`` for
    the requested rows, not only from ``genes``.
    """
    if row_indices is None:
        row_indices = np.arange(adata.n_obs, dtype=int)
    else:
        row_indices = np.asarray(row_indices, dtype=int)

    if genes is None:
        genes = adata.var_names.astype(str).tolist()
    else:
        genes = list(genes)

    matrix = _extract_raw_matrix(
        adata=adata,
        row_indices=row_indices,
        genes=genes,
    )

    if normalize_total:
        if not np.isfinite(target_sum) or target_sum <= 0:
            raise ValueError("target_sum must be a positive finite number.")

        totals = _row_library_sizes(
            adata=adata,
            row_indices=row_indices,
        )
        factors = np.zeros_like(totals, dtype=np.float64)
        positive = totals > 0
        factors[positive] = float(target_sum) / totals[positive]
        matrix = matrix * factors[:, None].astype(np.float32)

    if use_log1p:
        matrix = np.log1p(matrix).astype(np.float32)

    return np.asarray(matrix, dtype=np.float32)


def align_spatial_pair(
    disease: ad.AnnData,
    normal: ad.AnnData,
    spatial_key: str = "spatial",
    coordinate_decimals: int = 5,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Align disease and normal virtual-tissue locations without pre-filtering.

    Priority
    --------
    1. identical observation names in identical order;
    2. shared unique observation names;
    3. identical coordinates in identical order;
    4. coordinate intersection after rounding.
    """
    disease_names = np.asarray(disease.obs_names.astype(str))
    normal_names = np.asarray(normal.obs_names.astype(str))

    if disease.n_obs == normal.n_obs and np.array_equal(disease_names, normal_names):
        index = np.arange(disease.n_obs, dtype=int)
        return index, index.copy(), "obs_names_same_order"

    if (
        len(np.unique(disease_names)) == len(disease_names)
        and len(np.unique(normal_names)) == len(normal_names)
        and set(disease_names.tolist()) == set(normal_names.tolist())
    ):
        normal_lookup = {name: index for index, name in enumerate(normal_names)}
        disease_indices = np.arange(disease.n_obs, dtype=int)
        normal_indices = np.asarray(
            [normal_lookup[name] for name in disease_names],
            dtype=int,
        )
        return disease_indices, normal_indices, "same_obs_name_set"

    if spatial_key not in disease.obsm or spatial_key not in normal.obsm:
        raise ValueError("Could not align by obs_names and spatial coordinates are unavailable.")

    disease_coords = np.asarray(disease.obsm[spatial_key], dtype=np.float64)[:, :2]
    normal_coords = np.asarray(normal.obsm[spatial_key], dtype=np.float64)[:, :2]

    if disease_coords.shape == normal_coords.shape and np.allclose(
        disease_coords,
        normal_coords,
        atol=10 ** (-coordinate_decimals),
        rtol=0.0,
    ):
        index = np.arange(disease.n_obs, dtype=int)
        return index, index.copy(), "spatial_same_order"

    rounded_normal = np.round(normal_coords, decimals=coordinate_decimals)
    normal_lookup: Dict[Tuple[float, float], int] = {}
    duplicate_keys = set()

    for index, (x, y) in enumerate(rounded_normal):
        key = (float(x), float(y))
        if key in normal_lookup:
            duplicate_keys.add(key)
        else:
            normal_lookup[key] = index

    for key in duplicate_keys:
        normal_lookup.pop(key, None)

    disease_indices = []
    normal_indices = []
    for disease_index, (x, y) in enumerate(np.round(disease_coords, decimals=coordinate_decimals)):
        normal_index = normal_lookup.get((float(x), float(y)))
        if normal_index is not None:
            disease_indices.append(disease_index)
            normal_indices.append(normal_index)

    if not disease_indices:
        raise ValueError("Disease and normal virtual tissues could not be aligned by obs_names or spatial coordinates.")

    return (
        np.asarray(disease_indices, dtype=int),
        np.asarray(normal_indices, dtype=int),
        "spatial_intersection",
    )


def compute_spatial_delta(
    disease: ad.AnnData,
    normal: ad.AnnData,
    *,
    normalize_total: bool = True,
    target_sum: float = 1e4,
    use_log1p: bool = True,
    valid_key: str = "valid",
    spatial_key: str = "spatial",
    domain_key: Optional[str] = "axis_joint_domain",
    genes: Optional[Sequence[str]] = None,
    use_common_valid: bool = True,
    coordinate_decimals: int = 5,
    return_result: bool = False,
) -> Union[ad.AnnData, SpatialDeltaResult]:
    """
    Compute location-wise disease-minus-normal delta in one unified value space.

    Alignment happens on the complete virtual grids. When ``use_common_valid``
    is True, only locations valid in BOTH conditions are retained after
    alignment. Count normalization and log1p are applied independently to the
    disease and normal source values BEFORE subtraction.
    """
    disease_indices, normal_indices, alignment_method = align_spatial_pair(
        disease=disease,
        normal=normal,
        spatial_key=spatial_key,
        coordinate_decimals=coordinate_decimals,
    )

    matched_count = int(len(disease_indices))
    if matched_count == 0:
        raise ValueError("No matched disease/normal locations were found.")

    disease_valid = get_valid_mask(disease, valid_key=valid_key)[disease_indices]
    normal_valid = get_valid_mask(normal, valid_key=valid_key)[normal_indices]

    if use_common_valid:
        keep = disease_valid & normal_valid
    else:
        keep = np.ones(matched_count, dtype=bool)

    disease_indices = disease_indices[keep]
    normal_indices = normal_indices[keep]
    disease_valid = disease_valid[keep]
    normal_valid = normal_valid[keep]

    retained_count = int(len(disease_indices))
    if retained_count == 0:
        raise ValueError("No matched locations remain after validity filtering.")

    common_genes = _ordered_common_genes(
        disease=disease,
        normal=normal,
        genes=genes,
    )

    disease_matrix = extract_preprocessed_matrix(
        disease,
        row_indices=disease_indices,
        genes=common_genes,
        normalize_total=normalize_total,
        target_sum=target_sum,
        use_log1p=use_log1p,
    )
    normal_matrix = extract_preprocessed_matrix(
        normal,
        row_indices=normal_indices,
        genes=common_genes,
        normalize_total=normalize_total,
        target_sum=target_sum,
        use_log1p=use_log1p,
    )

    delta_matrix = (disease_matrix - normal_matrix).astype(np.float32)

    obs = disease.obs.iloc[disease_indices].copy()
    obs["_delta_disease_index"] = disease_indices
    obs["_delta_normal_index"] = normal_indices
    obs["_delta_disease_valid"] = disease_valid
    obs["_delta_normal_valid"] = normal_valid
    obs["_delta_common_valid"] = disease_valid & normal_valid

    disease_gene_indices = disease.var_names.get_indexer(common_genes)
    var = disease.var.iloc[disease_gene_indices].copy()
    var.index = pd.Index(common_genes)

    delta_adata = ad.AnnData(
        X=delta_matrix,
        obs=obs,
        var=var,
    )

    for key, values in disease.obsm.items():
        values = np.asarray(values)
        if values.shape[0] == disease.n_obs:
            delta_adata.obsm[key] = values[disease_indices].copy()

    if normalize_total and use_log1p:
        definition = "log1p(normalize_total(disease))-log1p(normalize_total(normal))"
    elif normalize_total:
        definition = "normalize_total(disease)-normalize_total(normal)"
    elif use_log1p:
        definition = "log1p(disease)-log1p(normal)"
    else:
        definition = "disease-normal"

    delta_adata.uns["spatial_delta"] = {
        "definition": definition,
        "normalize_total": bool(normalize_total),
        "target_sum": float(target_sum),
        "use_log1p": bool(use_log1p),
        "use_common_valid": bool(use_common_valid),
        "valid_key": str(valid_key),
        "spatial_key": str(spatial_key),
        "domain_key": None if domain_key is None else str(domain_key),
        "alignment_method": alignment_method,
        "matched_locations_before_valid_intersection": matched_count,
        "retained_locations": retained_count,
        "n_genes": len(common_genes),
    }

    if domain_key is not None and domain_key not in delta_adata.obs.columns:
        raise KeyError(
            f"Disease AnnData does not contain obs[{domain_key!r}], so the delta AnnData cannot be summarized by that domain key."
        )

    result = SpatialDeltaResult(
        adata=delta_adata,
        disease_indices=disease_indices,
        normal_indices=normal_indices,
        genes=common_genes,
        normalize_total=bool(normalize_total),
        target_sum=float(target_sum),
        use_log1p=bool(use_log1p),
        matched_count=matched_count,
        common_valid_count=retained_count,
    )

    return result if return_result else delta_adata


def compute_spatial_delta_from_files(
    disease_path: str,
    normal_path: str,
    *,
    normalize_total: bool = True,
    target_sum: float = 1e4,
    use_log1p: bool = True,
    valid_key: str = "valid",
    spatial_key: str = "spatial",
    domain_key: Optional[str] = "axis_joint_domain",
    genes: Optional[Sequence[str]] = None,
    use_common_valid: bool = True,
    coordinate_decimals: int = 5,
    return_result: bool = False,
) -> Union[ad.AnnData, SpatialDeltaResult]:
    """Read two complete h5ads and compute the canonical spatial delta."""
    if not os.path.isfile(disease_path):
        raise FileNotFoundError(disease_path)
    if not os.path.isfile(normal_path):
        raise FileNotFoundError(normal_path)

    disease = sc.read_h5ad(disease_path)
    normal = sc.read_h5ad(normal_path)

    return compute_spatial_delta(
        disease=disease,
        normal=normal,
        normalize_total=normalize_total,
        target_sum=target_sum,
        use_log1p=use_log1p,
        valid_key=valid_key,
        spatial_key=spatial_key,
        domain_key=domain_key,
        genes=genes,
        use_common_valid=use_common_valid,
        coordinate_decimals=coordinate_decimals,
        return_result=return_result,
    )


def domain_mean_delta_table(
    delta_adata: ad.AnnData,
    *,
    domain_key: str = "axis_joint_domain",
    genes: Optional[Sequence[str]] = None,
    min_spots: int = 1,
) -> pd.DataFrame:
    """Summarize an already preprocessed/subtracted delta AnnData by domain."""
    if domain_key not in delta_adata.obs.columns:
        raise KeyError(f"Missing delta_adata.obs[{domain_key!r}]")

    if genes is None:
        genes = delta_adata.var_names.astype(str).tolist()
    else:
        genes = [gene for gene in genes if gene in delta_adata.var_names]

    gene_indices = delta_adata.var_names.get_indexer(list(genes))
    matrix = delta_adata.X[:, gene_indices]
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)

    labels = delta_adata.obs[domain_key].astype(str).to_numpy()

    rows = []
    for domain in sorted(set(labels)):
        if domain == "nan":
            continue
        mask = labels == domain
        n_spots = int(np.sum(mask))
        if n_spots < min_spots:
            continue
        mean_delta = np.mean(matrix[mask], axis=0)
        rows.extend(
            {
                "domain": str(domain),
                "gene": gene,
                "delta": float(mean_delta[index]),
                "n_spots": n_spots,
            }
            for index, gene in enumerate(genes)
        )

    return pd.DataFrame(rows)
