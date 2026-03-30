#!/usr/bin/env python
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import entropy


# =========================
# CONFIGURATION
# =========================

ROOT_DIR = "/media/huifang/data/registration/xenium/mouse_brain"
GROUP_ID = 0

H5AD_DIR = (
    Path(ROOT_DIR)
    / "output"
    / f"group_{GROUP_ID}"
    / "h5ads"
    / "unified"
)

# Adjust filenames if needed
FILE_MAP = {
    # wildtype
    "WT_2.5": "xenium_WT_2.5m.h5ad",
    "WT_5.7": "xenium_WT_5.7m.h5ad",
    "WT_13.4": "xenium_WT_13.4m.h5ad",
    # TgCRND8 (AD model)
    "Tg_2.5": "xenium_Tg_2.5m.h5ad",
    "Tg_5.7": "xenium_Tg_5.7m.h5ad",
    "Tg_17.9": "xenium_Tg_17.9m.h5ad",
}


CELL_TYPE_COLUMN = "cell_type"
SPATIAL_KEY_AXIS = "spatial_axis"
SPATIAL_KEY_PIXEL = "spatial_axis_pixel"
GRID_SIZE = 50.0  # AXIS units per pixel

# AD-relevant modules (you should intersect with var_names later)
MODULES = {
    "Microglial_activation": [
        "C1qa", "C1qb", "C1qc", "Trem2", "Tyrobp", "Apoe", "Itgax",
        "Csf1r", "Lpl", "Ctsd", "Lgals3", "Spp1"
    ],
    "Astrocyte_reactivity": [
        "Gfap", "Serpina3n", "Vim", "Lgals3", "C3", "C4b"
    ],
    "Neuronal_synaptic": [
        "Gria1", "Gria2", "Grin1", "Syp", "Snap25", "Sv2a"
    ],
    "Oligo_myelin": [
        "Mbp", "Mog", "Plp1", "Cnp", "Mag"
    ],
}


# =========================
# IO & GRID HELPERS
# =========================

def load_samples(h5ad_dir: Path, file_map: Dict[str, str]) -> Dict[str, sc.AnnData]:
    """Load h5ad files according to FILE_MAP."""
    adata_dict = {}
    for key, fname in file_map.items():
        path = h5ad_dir / fname
        if not path.is_file():
            raise FileNotFoundError(f"Expected h5ad file not found: {path}")
        print(f"[INFO] Loading {key} from {path}")
        ad = sc.read_h5ad(path)
        sc.pp.filter_cells(ad, min_counts=100)
        sc.pp.filter_genes(ad, min_cells=10)
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
        adata_dict[key] = ad
    return adata_dict


def assign_pixel_coords_dict(
    adata_dict: Dict[str, sc.AnnData],
    grid_size: float = GRID_SIZE,
    spatial_key: str = SPATIAL_KEY_AXIS,
    out_key: str = SPATIAL_KEY_PIXEL,
):
    """
    Ensure each AnnData has .obsm[out_key] with integer pixel/grid
    coordinates on a common grid in AXIS space.
    """
    if not adata_dict:
        raise ValueError("adata_dict is empty.")

    xs = np.concatenate([ad.obsm[spatial_key][:, 0] for ad in adata_dict.values()])
    ys = np.concatenate([ad.obsm[spatial_key][:, 1] for ad in adata_dict.values()])

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    nx = int(np.ceil((max_x - min_x) / grid_size))
    ny = int(np.ceil((max_y - min_y) / grid_size))

    if nx <= 0 or ny <= 0:
        raise ValueError(
            f"Invalid grid dimensions: nx={nx}, ny={ny}. "
            "Check grid_size and coordinate ranges."
        )

    print(f"[INFO] Global AXIS bounds: x=[{min_x:.1f}, {max_x:.1f}], "
          f"y=[{min_y:.1f}, {max_y:.1f}], grid_size={grid_size}")
    print(f"[INFO] Grid shape: nx={nx}, ny={ny}")

    for name, adata in adata_dict.items():
        coords = adata.obsm[spatial_key]  # (n_cells, 2)
        ix = np.floor((coords[:, 0] - min_x) / grid_size).astype(int)
        iy = np.floor((coords[:, 1] - min_y) / grid_size).astype(int)

        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)

        axis_pixel_spatial = np.column_stack([ix, iy])
        adata.obsm[out_key] = axis_pixel_spatial
        print(f"[INFO] {name}: assigned {out_key} with shape {axis_pixel_spatial.shape}")

    return adata_dict


def pad_to(arr: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Pad (H, W) to (ny, nx) with zeros."""
    hh, ww = arr.shape
    out = np.zeros((ny, nx), dtype=arr.dtype)
    out[:hh, :ww] = arr
    return out


# =========================
# MODULE SCORES
# =========================

def score_modules_per_cell(
    adata_dict: Dict[str, sc.AnnData],
    modules: Dict[str, List[str]],
) -> Dict[str, sc.AnnData]:
    """
    For each AnnData, compute gene-module scores using scanpy.tl.score_genes.
    Adds adata.obs[module_name] columns.
    """
    for sname, ad in adata_dict.items():
        print(f"[INFO] Scoring modules in {sname}")
        for mname, genes in modules.items():
            # Restrict to genes present in this dataset
            g_present = [g for g in genes if g in ad.var_names]
            if len(g_present) < 2:
                print(f"  [WARN] {mname}: <2 genes present in {sname}, skipping.")
                continue
            sc.tl.score_genes(ad, gene_list=g_present, score_name=mname, use_raw=False)
            print(f"  [OK] {mname}: scored with {len(g_present)} genes.")
    return adata_dict


def aggregate_module_to_grid(
    adata: sc.AnnData,
    module_name: str,
    spatial_pixel_key: str = SPATIAL_KEY_PIXEL,
) -> np.ndarray:
    """
    Aggregate per-cell module score to per-pixel average on AXIS grid.

    Returns:
        grid: (ny, nx) array of mean module score per pixel.
    """
    if module_name not in adata.obs:
        raise KeyError(f"{module_name} not found in adata.obs")

    coords = adata.obsm[spatial_pixel_key]  # (n_cells, 2)
    ix = coords[:, 0].astype(int)
    iy = coords[:, 1].astype(int)

    scores = adata.obs[module_name].to_numpy()

    nx = ix.max() + 1
    ny = iy.max() + 1

    sum_grid = np.zeros((ny, nx), dtype=float)
    cnt_grid = np.zeros((ny, nx), dtype=float)

    for i in range(len(ix)):
        gx, gy = ix[i], iy[i]
        val = scores[i]
        if np.isnan(val):
            continue
        sum_grid[gy, gx] += val
        cnt_grid[gy, gx] += 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        grid = np.where(cnt_grid > 0, sum_grid / cnt_grid, np.nan)

    return grid


def plot_module_maps_across_samples(
    adata_dict: Dict[str, sc.AnnData],
    module_name: str,
    vmax_abs: float = None,
):
    """
    Plot spatial module score maps for all samples (WT and Tg) in AXIS pixel space.
    """
    sample_order = ["WT_2.5", "WT_5.7", "WT_13.4", "Tg_2.5", "Tg_5.7", "Tg_17.9"]
    existing = [s for s in sample_order if s in adata_dict]

    grids = {}
    ny_max, nx_max = 0, 0
    max_abs = 0.0

    for sname in existing:
        ad = adata_dict[sname]
        if module_name not in ad.obs:
            print(f"[WARN] {module_name} not found in {sname}, skipping.")
            continue
        grid = aggregate_module_to_grid(ad, module_name)
        grids[sname] = grid
        ny_max = max(ny_max, grid.shape[0])
        nx_max = max(nx_max, grid.shape[1])
        # ignore NaN when computing max_abs
        finite_vals = grid[np.isfinite(grid)]
        if finite_vals.size > 0:
            max_abs = max(max_abs, np.max(np.abs(finite_vals)))

    if not grids:
        print(f"[WARN] No grids computed for module {module_name}.")
        return

    if vmax_abs is None:
        vmax_abs = max_abs if max_abs > 0 else 1.0

    # Plot
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for idx, sname in enumerate(sample_order):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        if sname not in grids:
            ax.axis("off")
            continue

        grid = grids[sname]
        if grid.shape != (ny_max, nx_max):
            grid = pad_to(grid, ny_max, nx_max)

        im = ax.imshow(
            grid,
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            cmap="coolwarm",
            vmin=-vmax_abs,
            vmax=vmax_abs,
        )
        ax.set_title(sname, fontsize=9)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{module_name} score", fontsize=8)

    fig.suptitle(
        f"Spatial module score maps – {module_name} (WT vs TgCRND8, AXIS space)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# =========================
# ENTROPY / DIVERSITY MAPS
# =========================

def compute_entropy_grid(
    adata: sc.AnnData,
    cell_type_col: str,
    spatial_pixel_key: str = SPATIAL_KEY_PIXEL,
) -> np.ndarray:
    """
    For each pixel, compute Shannon entropy of cell-type composition.
    Higher entropy = more diverse / mixed local neighborhood.
    """
    coords = adata.obsm[spatial_pixel_key]
    ix = coords[:, 0].astype(int)
    iy = coords[:, 1].astype(int)

    ct = adata.obs[cell_type_col].astype(str).values
    ct_categories = sorted(list(set(ct)))
    ct_to_idx = {c: i for i, c in enumerate(ct_categories)}

    nx = ix.max() + 1
    ny = iy.max() + 1

    # count per pixel per cell type
    counts = np.zeros((ny, nx, len(ct_categories)), dtype=float)
    for i in range(len(ix)):
        gx, gy = ix[i], iy[i]
        ci = ct_to_idx[ct[i]]
        counts[gy, gx, ci] += 1.0

    # entropy along last axis
    ent_grid = np.zeros((ny, nx), dtype=float)
    for y in range(ny):
        for x in range(nx):
            vec = counts[y, x, :]
            if vec.sum() == 0:
                ent_grid[y, x] = np.nan
            else:
                p = vec / vec.sum()
                ent_grid[y, x] = entropy(p, base=2)  # bits

    return ent_grid




# =========================
# MAIN
# =========================

def main():
    # 1) Load samples
    adata_dict = load_samples(H5AD_DIR, FILE_MAP)

    # 2) Ensure pixel grid
    need_pixel = any(SPATIAL_KEY_PIXEL not in ad.obsm for ad in adata_dict.values())
    if need_pixel:
        print("[INFO] spatial_axis_pixel missing in some samples – computing from spatial_axis.")
        assign_pixel_coords_dict(
            adata_dict,
            grid_size=GRID_SIZE,
            spatial_key=SPATIAL_KEY_AXIS,
            out_key=SPATIAL_KEY_PIXEL,
        )
    else:
        print("[INFO] spatial_axis_pixel already present in all samples.")

    # 3) Score modules per cell
    score_modules_per_cell(adata_dict, MODULES)

    # 4) Plot module maps
    for module_name in MODULES.keys():
        print(f"\n[INFO] Plotting module maps for {module_name}")
        plot_module_maps_across_samples(adata_dict, module_name)


if __name__ == "__main__":
    main()
