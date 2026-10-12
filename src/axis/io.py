"""Reading and writing the on-disk layout shared by the AXIS pipelines.

Each dataset root (``AXIS_DATA_DIR/<dataset>``) follows::

    h5ad/{group}_{sample}.h5ad                      measured sections (counts in X)
    network_input/{group}_{sample}_validation.npz   canvas coordinates ("coord") and labels
    pairwise_output/group_{group}/{F}_{M}_flow.npy  field registering M (moving) onto F (fixed)
    pairwise_output/group_{group}/{F}_{M}_warped_coor.npy
    virtual_tissue/...                              generated / matched tissues
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "load_coords_npz",
    "load_axis_flow",
    "load_h5ad_X_and_genes",
    "load_h5ad_X_and_genes_and_coords",
    "save_network_input",
]


def load_coords_npz(net_in: str, group_id, sample_id) -> np.ndarray:
    """Canvas coordinates of one sample from ``network_input``."""
    return np.load(os.path.join(net_in, f"{group_id}_{sample_id}_validation.npz"))["coord"].astype(np.float32)


def load_axis_flow(axis_out: str, fixed_id, moving_id, basesize: int = 512) -> np.ndarray:
    """Field registering ``moving_id`` onto ``fixed_id`` (zeros when they are equal)."""
    if fixed_id == moving_id:
        return np.zeros((2, basesize, basesize))
    path = os.path.join(axis_out, f"{fixed_id}_{moving_id}_flow.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


def _read_attrs(adata, path: str, attr_columns: Optional[Sequence[str]]):
    if not attr_columns:
        return None
    missing = [c for c in attr_columns if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing obs columns in {path}: {missing}")
    return adata.obs.loc[:, list(attr_columns)].copy().reset_index(drop=True)


def load_h5ad_X_and_genes(h5ad_dir: str, group_id, sample_id, attr_columns: Optional[Sequence[str]] = None):
    """``(X, var_names, attrs)`` of ``h5ad/{group}_{sample}.h5ad``."""
    import anndata as ad

    path = os.path.join(h5ad_dir, f"{group_id}_{sample_id}.h5ad")
    adata = ad.read_h5ad(path)
    return adata.X, np.asarray(adata.var_names), _read_attrs(adata, path, attr_columns)


def load_h5ad_X_and_genes_and_coords(
    h5ad_dir: str, group_id, sample_id, attr_columns: Optional[Sequence[str]] = None, coord_key: str = "spatial_axis"
):
    """``(X, var_names, attrs, coords)``, with coordinates from ``obsm[coord_key]``."""
    import anndata as ad

    path = os.path.join(h5ad_dir, f"{group_id}_{sample_id}.h5ad")
    adata = ad.read_h5ad(path)
    return adata.X, np.asarray(adata.var_names), _read_attrs(adata, path, attr_columns), np.asarray(adata.obsm[coord_key])


def save_network_input(out_dir, prefix, image, feature_matrix, gene_mask, coords, labels, suffix: str = "network_input") -> str:
    """Write the network inputs of one sample (image, pseudo-image, mask, validation)."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    folder = Path(out_dir) / suffix
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / prefix
    plt.imsave(f"{base}_image.png", image)
    np.save(f"{base}_pca_out.npy", feature_matrix)
    np.save(f"{base}_pca_mask.npy", gene_mask)
    np.savez(f"{base}_validation.npz", coord=coords, label=labels)
    return str(base)
