#!/usr/bin/env python
"""
Visualize paired gene–protein signal (a-SMA / ACTA2) before and after AXIS registration,
using only the H5ADs_axis files.

Assumptions in H5ADs_axis:
  - obsm["spatial"]       : original per-modality coordinates
  - obsm["spatial_axis"]  : AXIS-aligned common coordinates

File layout:
  /media/huifang/data/registration/phenocycler/huifang/H5ADs_axis/
      [data_name]_protein_axis.h5ad
      [data_name]_xenium_axis.h5ad
"""

from pathlib import Path
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt


# ---------------- Configuration ---------------- #

ROOT = Path("/media/huifang/data/registration/phenocycler/huifang")
H5AD_DIR_AXIS = ROOT / "H5ADs_axis"

DATA_NAMES = ["LUAD_2_A", "TSU_35"]

PROTEIN_MODALITY = "protein"
GENE_MODALITY = "xenium"   # change to "visium" if needed

PROTEIN_MARKER = "a-SMA"
GENE_NAME = "ACTA2"

COORD_KEY_ORIG = "spatial"        # original frame in *_axis.h5ad
COORD_KEY_AXIS = "spatial_axis"   # AXIS-aligned frame


# ---------------- Helper functions ---------------- #

def load_axis_adata(data_name: str, modality: str) -> sc.AnnData:
    """
    Load axis-augmented AnnData:
        [data_name]_[modality]_axis.h5ad
    """
    path = H5AD_DIR_AXIS / f"{data_name}_{modality}_axis.h5ad"
    if not path.is_file():
        raise FileNotFoundError(f"Missing axis h5ad file: {path}")
    return sc.read_h5ad(path)


def get_coord(adata: sc.AnnData, key: str) -> np.ndarray:
    """Extract coordinates from adata.obsm[key] as a (N,2) float array."""
    if key not in adata.obsm_keys():
        raise KeyError(f"obsm['{key}'] not found in AnnData")
    coords = np.asarray(adata.obsm[key], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Coordinates under obsm['{key}'] must be (N,2), got {coords.shape}")
    return coords


def get_feature_vector(adata: sc.AnnData, feature_name: str) -> np.ndarray:
    """
    Extract a 1D expression vector for a given feature from .X or .raw.

    Works for both protein markers (in var_names) and genes.
    """
    # Primary: adata.var_names
    if feature_name in adata.var_names:
        idx = np.where(adata.var_names == feature_name)[0][0]
        X = adata.X
        if hasattr(X, "toarray"):
            vec = X[:, idx].toarray().ravel().astype(np.float32)
        else:
            vec = np.asarray(X[:, idx]).ravel().astype(np.float32)
        return vec

    # Fallback: adata.raw
    if adata.raw is not None and feature_name in adata.raw.var_names:
        idx = np.where(adata.raw.var_names == feature_name)[0][0]
        X = adata.raw.X
        if hasattr(X, "toarray"):
            vec = X[:, idx].toarray().ravel().astype(np.float32)
        else:
            vec = np.asarray(X[:, idx]).ravel().astype(np.float32)
        return vec

    raise KeyError(f"Feature '{feature_name}' not found in var_names or raw.var_names")


def robust_color_limits(arr: np.ndarray, low=2, high=98) -> tuple:
    """Compute robust vmin/vmax ignoring NaNs."""
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = np.percentile(vals, low)
    vmax = np.percentile(vals, high)
    if vmin == vmax:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


# ---------------- Main visualization per dataset ---------------- #

def visualize_pair_for_dataset(data_name: str):
    print(f"\n[INFO] Visualizing a-SMA / ACTA2 for {data_name}")

    # Load protein & gene AnnData from H5ADs_axis
    adata_prot = load_axis_adata(data_name, PROTEIN_MODALITY)
    adata_gene = load_axis_adata(data_name, GENE_MODALITY)

    # Coordinates: original and AXIS-aligned
    coords_prot_orig = get_coord(adata_prot, COORD_KEY_ORIG)
    coords_gene_orig = get_coord(adata_gene, COORD_KEY_ORIG)

    coords_prot_axis = get_coord(adata_prot, COORD_KEY_AXIS)
    coords_gene_axis = get_coord(adata_gene, COORD_KEY_AXIS)

    # Expression vectors
    prot_vec = get_feature_vector(adata_prot, PROTEIN_MARKER)
    gene_vec = get_feature_vector(adata_gene, GENE_NAME)
    # prot_vec = prot_vec-prot_vec.min()
    prot_vec = np.log1p(prot_vec)
    gene_vec = np.log1p(gene_vec)

    # Color scales
    vmin_prot, vmax_prot = robust_color_limits(prot_vec)
    vmin_gene, vmax_gene = robust_color_limits(gene_vec)

    # 2x2 panel: (before/after) × (protein/gene)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 1) Before – protein
    ax = axes[0, 0]
    sc1 = ax.scatter(
        coords_prot_orig[:, 0],
        coords_prot_orig[:, 1],
        c=prot_vec,
        s=1,
        cmap="magma",
        vmin=vmin_prot,
        vmax=vmax_prot,
    )
    ax.set_title(f"{data_name} – Protein {PROTEIN_MARKER}\nBefore registration ({PROTEIN_MODALITY} frame)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    # 2) Before – gene
    ax = axes[0, 1]
    sc2 = ax.scatter(
        coords_gene_orig[:, 0],
        coords_gene_orig[:, 1],
        c=gene_vec,
        s=1,
        cmap="viridis",
        vmin=vmin_gene,
        vmax=vmax_gene,
    )
    ax.set_title(f"{data_name} – Gene {GENE_NAME}\nBefore registration ({GENE_MODALITY} frame)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    # 3) After – protein in AXIS frame
    ax = axes[1, 0]
    sc3 = ax.scatter(
        coords_prot_axis[:, 0],
        coords_prot_axis[:, 1],
        c=prot_vec,
        s=1,
        cmap="magma",
        vmin=vmin_prot,
        vmax=vmax_prot,
    )
    ax.set_title(f"{data_name} – Protein {PROTEIN_MARKER}\nAfter registration (AXIS frame)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    # 4) After – gene in AXIS frame
    ax = axes[1, 1]
    sc4 = ax.scatter(
        coords_gene_axis[:, 0],
        coords_gene_axis[:, 1],
        c=gene_vec,
        s=1,
        cmap="viridis",
        vmin=vmin_gene,
        vmax=vmax_gene,
    )
    ax.set_title(f"{data_name} – Gene {GENE_NAME}\nAfter registration (AXIS frame)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    # Colorbars in side axes to avoid overlap
    cax1 = fig.add_axes([0.08, 0.52, 0.015, 0.3])
    cbar1 = fig.colorbar(sc1, cax=cax1)
    cbar1.set_label(f"{PROTEIN_MARKER} intensity")

    cax2 = fig.add_axes([0.93, 0.52, 0.015, 0.3])
    cbar2 = fig.colorbar(sc2, cax=cax2)
    cbar2.set_label(f"{GENE_NAME} expression")

    fig.suptitle(
        f"{data_name} – a-SMA (protein) vs ACTA2 (gene)\n"
        "Before vs After AXIS registration",
        fontsize=14,
    )
    plt.tight_layout(rect=[0.06, 0.03, 0.94, 0.95])
    plt.show()


def main():
    for data_name in DATA_NAMES:
        try:
            visualize_pair_for_dataset(data_name)
        except Exception as e:
            print(f"[WARN] Skipping {data_name} due to error: {e}")


if __name__ == "__main__":
    main()
