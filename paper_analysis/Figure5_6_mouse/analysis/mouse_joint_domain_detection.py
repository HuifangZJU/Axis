#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Joint spatial domain detection across multiple AXIS-aligned .h5ad slices.

Workflow
--------
1. Recursively collect all .h5ad files under a given root folder
2. Read obsm['spatial_axis'] and obsm['X_scvi'] from each file
3. Concatenate all cells into one joint graph space while tracking source slice
4. Build:
      - a spatial kNN graph from obsm['spatial_axis']
      - a gene-latent kNN graph from obsm['X_scvi']
   and combine them into one fused adjacency graph
5. Run Leiden clustering on the fused graph
6. Write the joint domain labels back to each original file as:
      obs['axis_joint_domain']
7. Overwrite the original .h5ad files

Requirements
------------
pip install scanpy anndata scipy scikit-learn pandas numpy

Notes
-----
- This script assumes every target .h5ad has:
      obsm['spatial_axis']
      obsm['X_scvi']
- The clustering is joint across all slices in the shared aligned coordinate frame.
- Domain labels are written back per original cell, preserving original slice membership.
"""

from __future__ import annotations

import anndata as ad
from pathlib import Path
from math import ceil
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler



import numpy as np
import scipy.sparse as sp


def finite_row_mask(*arrays) -> np.ndarray:
    """
    Return a boolean mask of rows that are finite across all given arrays.
    Each array must have shape (n_samples, n_features).
    """
    mask = None
    for arr in arrays:
        arr = np.asarray(arr)
        cur = np.all(np.isfinite(arr), axis=1)
        mask = cur if mask is None else (mask & cur)
    return mask


def safe_standardize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Standardize features safely, avoiding nan/inf from zero variance columns.
    """
    X = np.asarray(X, dtype=np.float64)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std[~np.isfinite(std)] = 1.0
    std[std < eps] = 1.0

    X = (X - mean) / std
    X[~np.isfinite(X)] = 0.0
    return X.astype(np.float32)


def sanitize_sparse_graph(W: sp.csr_matrix, name: str = "graph") -> sp.csr_matrix:
    """
    Remove non-finite values from a sparse graph and clip negatives.
    """
    W = W.tocsr(copy=True)

    if W.nnz == 0:
        return W

    data = W.data
    bad = ~np.isfinite(data)
    if np.any(bad):
        print(f"[WARN] {name}: found {bad.sum()} non-finite edge weights; replacing with 0.")
        data[bad] = 0.0

    neg = data < 0
    if np.any(neg):
        print(f"[WARN] {name}: found {neg.sum()} negative edge weights; clipping to 0.")
        data[neg] = 0.0

    W.data = data
    W.eliminate_zeros()
    return W


def check_sparse_graph_finite(W: sp.csr_matrix, name: str = "graph"):
    """
    Raise an informative error if the sparse graph still contains nan/inf.
    """
    if W.nnz == 0:
        raise ValueError(f"{name} has no edges after sanitization.")

    bad = ~np.isfinite(W.data)
    if np.any(bad):
        idx = np.where(bad)[0][:10]
        vals = W.data[idx]
        raise ValueError(
            f"{name} still contains non-finite weights after sanitization. "
            f"Examples: {vals}"
        )




# -----------------------------
# Utility functions
# -----------------------------

def find_h5ad_files(root: Path) -> List[Path]:
    """Recursively collect all .h5ad files under root."""
    paths = sorted([p for p in root.rglob("*.h5ad") if p.is_file()])
    if len(paths) == 0:
        raise FileNotFoundError(f"No .h5ad files found under: {root}")
    return paths


def ensure_2d_array(x, name: str, path: Path) -> np.ndarray:
    """Convert input to a dense 2D numpy array and validate shape."""
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x)

    if x.ndim != 2:
        raise ValueError(
            f"{name} in file '{path}' must be a 2D array, got shape {x.shape}"
        )
    if x.shape[0] == 0:
        raise ValueError(
            f"{name} in file '{path}' has zero rows."
        )
    return x


def gaussian_knn_affinity(
    X: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    sigma: float | None = None,
    include_self: bool = False,
) -> sp.csr_matrix:
    """
    Build a Gaussian-weighted kNN affinity matrix.

    Parameters
    ----------
    X
        Shape (n_cells, n_features)
    n_neighbors
        Number of neighbors excluding self
    metric
        Distance metric for kNN
    sigma
        Gaussian bandwidth. If None, estimated from nonzero neighbor distances.
    include_self
        Whether to keep self-loops

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric weighted adjacency matrix
    """
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 cells to build a graph.")

    k = min(n_neighbors + 1, n)
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)
    dists, inds = nn.kneighbors(X, return_distance=True)

    # Remove self-neighbor in the first column when present
    if k > 1:
        dists = dists[:, 1:]
        inds = inds[:, 1:]
    else:
        # Degenerate case: only one sample
        dists = np.empty((n, 0), dtype=np.float32)
        inds = np.empty((n, 0), dtype=np.int64)

    if dists.size == 0:
        # Fallback to empty graph
        return sp.csr_matrix((n, n), dtype=np.float32)

    if sigma is None:
        positive = dists[dists > 0]
        if positive.size == 0:
            sigma = 1.0
        else:
            sigma = float(np.median(positive))
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 1.0

    weights = np.exp(-(dists ** 2) / (2.0 * sigma ** 2)).astype(np.float32)

    rows = np.repeat(np.arange(n), dists.shape[1])
    cols = inds.reshape(-1)
    vals = weights.reshape(-1)

    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)

    if include_self:
        W = W + sp.eye(n, dtype=np.float32, format="csr")

    # Symmetrize with max to preserve strong mutual edges
    W = W.maximum(W.T)
    W.eliminate_zeros()
    return W


def row_normalize_sparse(W: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalize a sparse matrix."""
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    inv = sp.diags(1.0 / row_sum.astype(np.float32))
    return inv @ W


def build_joint_graph(
    spatial_axis: np.ndarray,
    x_scvi: np.ndarray,
    k_spatial: int = 15,
    k_gene: int = 15,
    spatial_weight: float = 1.0,
    gene_weight: float = 1.0,
    scale_spatial: bool = True,
    scale_gene: bool = True,
) -> sp.csr_matrix:
    """
    Build a fused graph from spatial proximity and latent gene similarity.
    """
    spatial_axis = np.asarray(spatial_axis, dtype=np.float32)
    x_scvi = np.asarray(x_scvi, dtype=np.float32)

    if not np.all(np.isfinite(spatial_axis)):
        raise ValueError("spatial_axis still contains non-finite values before graph building.")
    if not np.all(np.isfinite(x_scvi)):
        raise ValueError("X_scvi still contains non-finite values before graph building.")

    if scale_spatial:
        spatial_axis = safe_standardize(spatial_axis)
    if scale_gene:
        x_scvi = safe_standardize(x_scvi)

    W_spatial = gaussian_knn_affinity(
        spatial_axis,
        n_neighbors=k_spatial,
        metric="euclidean",
    )
    W_gene = gaussian_knn_affinity(
        x_scvi,
        n_neighbors=k_gene,
        metric="euclidean",
    )

    W_spatial = row_normalize_sparse(W_spatial)
    W_gene = row_normalize_sparse(W_gene)

    W_spatial = sanitize_sparse_graph(W_spatial, name="W_spatial")
    W_gene = sanitize_sparse_graph(W_gene, name="W_gene")

    W = (spatial_weight * W_spatial) + (gene_weight * W_gene)
    W = 0.5 * (W + W.T)
    W = W.tocsr()
    W = sanitize_sparse_graph(W, name="W_fused")
    W.eliminate_zeros()

    check_sparse_graph_finite(W, name="W_fused")

    if W.nnz == 0:
        raise ValueError("Final fused graph has no edges.")

    return W


def run_leiden_on_graph(
    adjacency: sp.csr_matrix,
    resolution: float = 1.0,
    key_added: str = "axis_joint_domain",
    random_state: int = 0,
) -> np.ndarray:
    """
    Run Leiden clustering from a precomputed adjacency matrix.
    """
    adjacency = adjacency.tocsr(copy=True)
    adjacency = sanitize_sparse_graph(adjacency, name="adjacency")
    check_sparse_graph_finite(adjacency, name="adjacency")

    n = adjacency.shape[0]
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Adjacency must be square, got {adjacency.shape}")

    print(
        f"[INFO] Leiden input graph: n={n}, nnz={adjacency.nnz}, "
        f"min={adjacency.data.min():.6g}, max={adjacency.data.max():.6g}"
    )

    adata_joint = ad.AnnData(X=sp.csr_matrix((n, 1), dtype=np.float32))
    adata_joint.obsp["axis_joint_connectivities"] = adjacency

    sc.tl.leiden(
        adata_joint,
        adjacency=adata_joint.obsp["axis_joint_connectivities"],
        resolution=resolution,
        key_added=key_added,
        directed=False,
        use_weights=True,
        random_state=random_state,
    )

    labels = adata_joint.obs[key_added].astype(str).to_numpy()
    return labels


# -----------------------------
# Main workflow
# -----------------------------

def downsample_adata(adata, dsrate):
    n_sub = max(1, int(adata.n_obs / dsrate))
    print(f"Downsampling from {adata.n_obs} → {n_sub} cells")

    # deterministic: evenly spaced indices
    step = max(1, int(np.floor(adata.n_obs / n_sub)))
    subset_idx = np.arange(0, adata.n_obs, step)[:n_sub]

    adata_sub = adata[subset_idx].copy()
    return adata_sub

def load_all_slices(
    paths,
    spatial_key: str = "spatial_axis",
    latent_key: str = "X_scvi",
):
    """
    Load all slices and collect joint arrays, using only cells with finite
    spatial_axis and X_scvi rows for joint clustering.

    Returns
    -------
    adatas
        Original AnnData objects, unchanged
    spatial_all
        Concatenated valid spatial_axis arrays
    latent_all
        Concatenated valid X_scvi arrays
    meta
        Per-slice metadata with valid row mapping
    """
    adatas = []
    spatial_list = []
    latent_list = []
    meta = []

    cursor = 0

    for i, path in enumerate(paths):
        print(f"[{i+1}/{len(paths)}] Reading: {path}")
        adata = sc.read_h5ad(path)
        # adata = downsample_adata(adata,100)
        if spatial_key not in adata.obsm:
            raise KeyError(f"Missing obsm['{spatial_key}'] in file: {path}")
        if latent_key not in adata.obsm:
            raise KeyError(f"Missing obsm['{latent_key}'] in file: {path}")
        if "spatial" not in adata.obsm:
            raise KeyError(f"Missing obsm['spatial'] in file: {path}")

        spatial_axis = ensure_2d_array(adata.obsm[spatial_key], spatial_key, path)
        x_scvi = ensure_2d_array(adata.obsm[latent_key], latent_key, path)

        n_total = adata.n_obs
        if spatial_axis.shape[0] != n_total:
            raise ValueError(
                f"Row mismatch in {path}: n_obs={n_total}, "
                f"{spatial_key}.shape[0]={spatial_axis.shape[0]}"
            )
        if x_scvi.shape[0] != n_total:
            raise ValueError(
                f"Row mismatch in {path}: n_obs={n_total}, "
                f"{latent_key}.shape[0]={x_scvi.shape[0]}"
            )

        valid_mask = finite_row_mask(spatial_axis, x_scvi)
        kept_idx = np.where(valid_mask)[0]
        bad_idx = np.where(~valid_mask)[0]

        if len(bad_idx) > 0:
            print(
                f"[WARN] {path}: excluding {len(bad_idx)} / {n_total} cells from joint clustering "
                f"due to non-finite values in {spatial_key} or {latent_key}"
            )

        spatial_valid = spatial_axis[kept_idx].astype(np.float32)
        latent_valid = x_scvi[kept_idx].astype(np.float32)

        adatas.append(adata)
        spatial_list.append(spatial_valid)
        latent_list.append(latent_valid)

        meta.append(
            {
                "path": path,
                "slice_id": str(path),
                "start": cursor,
                "end": cursor + len(kept_idx),
                "n_total": n_total,
                "n_valid": len(kept_idx),
                "kept_idx": kept_idx,
                "bad_idx": bad_idx,
            }
        )
        cursor += len(kept_idx)

    if len(spatial_list) == 0 or sum(x.shape[0] for x in spatial_list) == 0:
        raise ValueError("No valid cells remain after finite-value filtering.")

    spatial_all = np.vstack(spatial_list).astype(np.float32)
    latent_all = np.vstack(latent_list).astype(np.float32)

    return adatas, spatial_all, latent_all, meta

def assign_labels_back_and_overwrite(
    adatas,
    meta,
    labels,
    output_obs_key: str = "axis_joint_domain",
    invalid_label: str = "invalid",
) -> None:
    """
    Write joint labels back to each original AnnData and overwrite the file.

    Cells excluded from joint clustering due to non-finite inputs receive
    the label `invalid_label`.
    """
    labels = np.asarray(labels).astype(str)

    total_expected = sum(m["n_valid"] for m in meta)
    if len(labels) != total_expected:
        raise ValueError(
            f"Label length mismatch: got {len(labels)}, expected {total_expected}"
        )

    for adata, m in zip(adatas, meta):
        start, end = m["start"], m["end"]
        cur_labels = labels[start:end]

        if len(cur_labels) != m["n_valid"]:
            raise ValueError(
                f"Slice label length mismatch for {m['path']}: "
                f"got {len(cur_labels)}, expected {m['n_valid']}"
            )

        full_labels = np.full(adata.n_obs, invalid_label, dtype=object)
        full_labels[m["kept_idx"]] = cur_labels

        adata.obs[output_obs_key] = pd.Categorical(full_labels)

        adata.uns["axis_joint_domain_info"] = {
            "method": "joint_leiden_on_fused_spatial_and_scvi_graph",
            "spatial_key": "spatial_axis",
            "latent_key": "X_scvi",
            "label_key": output_obs_key,
            "invalid_label": invalid_label,
            "n_valid_for_joint_clustering": int(m["n_valid"]),
            "n_invalid_excluded": int(len(m["bad_idx"])),
        }

        print(f"Overwriting: {m['path']}")
        adata.write_h5ad(m["path"])


def _sorted_domain_labels(labels) -> List[str]:
    """
    Sort labels numerically when possible, otherwise lexicographically.
    """
    labels = list(pd.unique(pd.Series(labels).astype(str)))

    def _key(x):
        try:
            return (0, int(x))
        except Exception:
            return (1, str(x))

    return sorted(labels, key=_key)


def visualize_joint_domains_from_memory(
    adatas,
    meta,
    labels,
    spatial_key: str = "spatial",
    invalid_label: str = "invalid",
    ncols: int = 4,
    point_size: float = 8.0,
    alpha: float = 0.9,
    figsize_per_col: float = 4.5,
    figsize_per_row: float = 4.5,
    show_legend: bool = True,
    legend_max_cols: int = 6,
    invert_y: bool = True,
    title: str | None = None,
):
    from math import ceil
    from pathlib import Path
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    labels = np.asarray(labels).astype(str)

    total_expected = sum(m["n_valid"] for m in meta)
    if len(labels) != total_expected:
        raise ValueError(
            f"Label length mismatch: got {len(labels)}, expected {total_expected}"
        )

    # Build full labels for all slices first
    full_slice_labels = []
    all_domains = []

    for adata, m in zip(adatas, meta):
        start, end = m["start"], m["end"]
        cur_labels = labels[start:end]

        full_labels = np.full(adata.n_obs, invalid_label, dtype=object)
        full_labels[m["kept_idx"]] = cur_labels

        full_slice_labels.append(full_labels)
        all_domains.extend(full_labels.astype(str))

    unique_domains = _sorted_domain_labels(all_domains)

    if len(unique_domains) <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(len(unique_domains))]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / max(len(unique_domains), 1)) for i in range(len(unique_domains))]

    domain_to_color = {d: c for d, c in zip(unique_domains, colors)}

    n_slices = len(adatas)
    ncols = min(max(1, ncols), n_slices)
    nrows = ceil(n_slices / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, adata, m, slice_labels in zip(axes_flat, adatas, meta, full_slice_labels):
        coords = adata.obsm[spatial_key]
        if sp.issparse(coords):
            coords = coords.toarray()
        coords = np.asarray(coords)

        point_colors = [domain_to_color[d] for d in slice_labels.astype(str)]

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=point_colors,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

        slice_name = Path(str(m["path"])).stem
        ax.set_title(
            f"{slice_name}\nN={adata.n_obs}, valid={m['n_valid']}",
            fontsize=10,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if invert_y:
            ax.invert_yaxis()

    for j in range(len(adatas), len(axes_flat)):
        axes_flat[j].axis("off")

    if show_legend:
        handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=domain_to_color[d],
                markeredgecolor="none",
                label=str(d),
            )
            for d in unique_domains
        ]
        ncol_legend = min(legend_max_cols, max(1, len(handles)))
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol_legend,
            frameon=False,
            title="axis_joint_domain",
        )
        fig.subplots_adjust(bottom=0.10)

    if title is None:
        title = f"Computed joint spatial domains ({spatial_key})"
    fig.suptitle(title, y=0.995, fontsize=14)

    plt.tight_layout()
    return fig, axes, domain_to_color


def visualize_joint_domains_each_slice(
    root: str | Path,
    obs_key: str = "axis_joint_domain",
    spatial_key: str = "spatial",
    pattern: str = "*.h5ad",
    recursive: bool = True,
    ncols: int = 4,
    point_size: float = 8.0,
    alpha: float = 0.9,
    figsize_per_col: float = 4.5,
    figsize_per_row: float = 4.5,
    show_legend: bool = True,
    legend_max_cols: int = 6,
    invert_y: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 200,
):
    root = Path(root)
    paths = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No files matching '{pattern}' found under: {root}")

    # First pass: collect all domain labels across all slices
    all_domains = []
    slice_data = []

    for path in paths:
        adata = sc.read_h5ad(path)

        if obs_key not in adata.obs:
            raise KeyError(f"Missing obs['{obs_key}'] in file: {path}")
        if spatial_key not in adata.obsm:
            raise KeyError(f"Missing obsm['{spatial_key}'] in file: {path}")

        coords = adata.obsm[spatial_key]
        if sp.issparse(coords):
            coords = coords.toarray()
        coords = np.asarray(coords)

        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                f"obsm['{spatial_key}'] in file {path} must have shape (n_cells, >=2), got {coords.shape}"
            )

        domains = adata.obs[obs_key].astype(str).to_numpy()
        all_domains.extend(domains)

        slice_data.append(
            {
                "path": path,
                "name": path.stem,
                "coords": coords[:, :2],
                "domains": domains,
                "n_obs": adata.n_obs,
            }
        )

    unique_domains = _sorted_domain_labels(all_domains)

    # Build a consistent palette across all slices
    # tab20 works well up to ~20 categories; for more, fall back to hsv
    if len(unique_domains) <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(len(unique_domains))]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / max(len(unique_domains), 1)) for i in range(len(unique_domains))]

    domain_to_color: Dict[str, Tuple[float, float, float, float]] = {
        d: c for d, c in zip(unique_domains, colors)
    }

    # Figure layout
    n_slices = len(slice_data)
    ncols = min(max(1, ncols), n_slices)
    nrows = ceil(n_slices / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    # Plot each slice
    for ax, item in zip(axes_flat, slice_data):
        xy = item["coords"]
        domains = item["domains"]

        point_colors = [domain_to_color[d] for d in domains]

        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=point_colors,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

        ax.set_title(f"{item['name']}\nN={item['n_obs']}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        if invert_y:
            ax.invert_yaxis()

    # Turn off any extra axes
    for j in range(len(slice_data), len(axes_flat)):
        axes_flat[j].axis("off")

    # Global legend
    if show_legend:
        handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=domain_to_color[d],
                markeredgecolor="none",
                label=str(d),
            )
            for d in unique_domains
        ]

        ncol_legend = min(legend_max_cols, max(1, len(handles)))
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol_legend,
            frameon=False,
            title=obs_key,
        )
        fig.subplots_adjust(bottom=0.10)
    else:
        fig.subplots_adjust(bottom=0.04)

    fig.suptitle(
        f"Joint spatial domains per slice ({spatial_key})",
        y=0.995,
        fontsize=14,
    )
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes, domain_to_color


for ROOT in ["/media/huifang/data/registration/xenium/mouse_brain/h5ads/WT","/media/huifang/data/registration/xenium/mouse_brain/h5ads/Tg","/media/huifang/data/registration/xenium/mouse_brain/h5ads/unified"]:

    SPATIAL_KEY = 'spatial_axis'
    LATENT_KEY = 'X_scvi'
    OUTPUT_OBS_KEY = 'axis_joint_domain'
    K_SPATIAL = 15
    K_GENE = 15
    SPATIAL_WEIGHT=1.0
    GENE_WEIGHT = 1.0
    RESOLUTION = 1.0
    RANDOM_STATE=0

    root = Path(ROOT)

    visualize_joint_domains_each_slice(root,invert_y=False)
    plt.show()

    # if not root.exists():
    #     raise FileNotFoundError(f"Root folder does not exist: {root}")
    #
    # print("Collecting .h5ad files ...")
    # paths = find_h5ad_files(root)
    # print(f"Found {len(paths)} .h5ad files")
    #
    # print("Loading slices ...")
    # adatas, spatial_all, latent_all, meta = load_all_slices(
    #     paths=paths,
    #     spatial_key=SPATIAL_KEY,
    #     latent_key=LATENT_KEY,
    # )
    #
    # print(f"Total cells: {spatial_all.shape[0]}")
    # print(f"Spatial dims: {spatial_all.shape[1]}")
    # print(f"Latent dims: {latent_all.shape[1]}")
    #
    # print("Building fused joint graph ...")
    # W = build_joint_graph(
    #     spatial_axis=spatial_all,
    #     x_scvi=latent_all,
    #     k_spatial=K_SPATIAL,
    #     k_gene=K_GENE,
    #     spatial_weight=SPATIAL_WEIGHT,
    #     gene_weight=GENE_WEIGHT,
    #     scale_spatial=True,
    #     scale_gene=True,
    # )
    # print(f"Graph shape: {W.shape}, nnz={W.nnz}")
    #
    # print("Running Leiden clustering ...")
    # labels = run_leiden_on_graph(
    #     adjacency=W,
    #     resolution=RESOLUTION,
    #     key_added=OUTPUT_OBS_KEY,
    #     random_state=RANDOM_STATE,
    # )
    #
    # n_domains = len(np.unique(labels))
    # print(f"Detected {n_domains} joint domains")
    #
    # # fig, axes, domain_to_color = visualize_joint_domains_from_memory(
    # #     adatas=adatas,
    # #     meta=meta,
    # #     labels=labels,
    # #     spatial_key="spatial",   # original slice layout
    # #     ncols=4,
    # #     point_size=8,
    # #     invert_y=False,
    # #     alpha=0.9,
    # # )
    # # plt.show()
    #
    # print("Writing labels back to original slices ...")
    # assign_labels_back_and_overwrite(
    #     adatas=adatas,
    #     meta=meta,
    #     labels=labels,
    #     output_obs_key=OUTPUT_OBS_KEY,
    # )
    #
    # print("Done.")
