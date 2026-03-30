#!/usr/bin/env python
"""
Low-resolution AXIS-only regression using original Visium spot counts.

For each patient:
  - Choose a fixed slice (FIXED_ID in {0,1,2}).
  - For each sample sid in {0,1,2}:
      * Load H5AD with spot-level counts.
      * Extract given gene expression vector.
      * Get coordinates in the fixed AXIS frame:
          - sid == FIXED_ID: use adata.obsm['spatial_image_coor'] directly (512x512 space)
          - sid != FIXED_ID: load precomputed warped coords:
                group_{patient}/{FIXED_ID}_{sid}_warped_coor.npy
      * Bin spots to a coarse grid (GRID_H x GRID_W) using mean per cell.
  - Run pixel-wise regression slope across the three low-res maps.
  - Visualization:
      * 2x3 figure:
          row 0: original spot scatter for sample0/1/2 in fixed frame
          row 1: low-res gridded maps for sample0/1/2
      * separate figure: slope map
"""

import glob
from pathlib import Path

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt


# --------------------------- Config ---------------------------

ROOT = Path("/media/huifang/data/registration/SCC/huifang")
H5AD_ROOT = ROOT / "H5ADs"
AXIS_OUT_ROOT = ROOT / "output"

PATIENT_IDS = [2, 5, 9, 10]   # SCC patients
SAMPLE_IDS = [0, 1, 2]        # slices 0,1,2
FIXED_ID = 0                  # which slice is fixed in AXIS frame
GRID_H = 64                  # low-res grid height
GRID_W = 64                  # low-res grid width
CANVAS_SIZE = 512.0           # coordinate extent; coords are in ~[0,512)

GENE_NAME = "A2ML1"            # gene of interest for regression


# --------------------------- Helpers ---------------------------

def load_h5ad_gene_and_coords(patient_id: int, sample_id: int, gene_name: str):
    """
    Load H5AD and return:
      - gene_values: (N_spots,) float32
      - coords:      (N_spots,2) float32 from obsm['spatial_image_coor']

    patient_id: 2,5,9,10
    sample_id:  0,1,2  (maps to layer1,2,3)
    """
    layer_idx = sample_id + 1
    h5_path = H5AD_ROOT / f"scc_p{patient_id}_layer{layer_idx}.h5ad"
    if not h5_path.is_file():
        raise FileNotFoundError(f"H5AD not found: {h5_path}")

    adata = sc.read_h5ad(h5_path)

    if "spatial_image_coor" not in adata.obsm:
        raise KeyError(f"obsm['spatial_image_coor'] missing in {h5_path}")

    coords = np.asarray(adata.obsm["spatial_image_coor"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"spatial_image_coor has shape {coords.shape} in {h5_path}, expected (N,2)"
        )

    if gene_name not in adata.var_names:
        raise KeyError(f"Gene {gene_name} not found in {h5_path}")

    gene_idx = np.where(adata.var_names == gene_name)[0][0]
    X = adata.X
    if hasattr(X, "toarray"):
        gene_values = X[:, gene_idx].toarray().ravel().astype(np.float32)
    else:
        gene_values = np.asarray(X[:, gene_idx]).ravel().astype(np.float32)

    return gene_values, coords


def load_warped_coords(patient_id: int, fixed_id: int, moving_id: int, n_spots: int):
    """
    Load warped coordinates for (fixed_id, moving_id) pair from:

        {AXIS_OUT_ROOT}/group_{patient}/{fixed_id}_{moving_id}_warped_coor.npy

    Returns (N_spots, 2) float32.
    """
    group_dir = AXIS_OUT_ROOT / f"group_{patient_id}"
    if not group_dir.is_dir():
        raise FileNotFoundError(f"AXIS group dir not found: {group_dir}")

    pattern = str(group_dir / f"{fixed_id}_{moving_id}_*warped_coor.npy")
    matches = glob.glob(pattern)
    if not matches:
        # fallback: exact name without wildcard
        fallback = group_dir / f"{fixed_id}_{moving_id}_warped_coor.npy"
        if not fallback.is_file():
            raise FileNotFoundError(
                f"No warped_coor file found for {fixed_id}_{moving_id} in {group_dir}"
            )
        warped_path = fallback
    else:
        warped_path = matches[0]

    warped = np.load(warped_path).astype(np.float32)
    if warped.ndim != 2 or warped.shape[1] != 2:
        raise ValueError(f"Unexpected warped_coor shape {warped.shape} in {warped_path}")

    if warped.shape[0] != n_spots:
        print(
            f"[WARN] warped_coor row count {warped.shape[0]} != n_spots {n_spots} "
            f"for patient {patient_id}, {fixed_id}_{moving_id}"
        )

    return warped


def bin_spots_to_grid(coords: np.ndarray,
                      values: np.ndarray,
                      grid_h: int,
                      grid_w: int,
                      canvas_size: float) -> np.ndarray:
    """
    Bin spot-level values into a (grid_h x grid_w) image via mean per cell.

    coords:  (N,2) in [0, canvas_size) for both x,y in the *fixed* frame.
    values:  (N,) float array of gene expression (can be any real).
    Returns:
      grid:   (grid_h, grid_w) float32, with NaN where no spots.
    """
    if coords.shape[0] != values.shape[0]:
        raise ValueError("coords and values must have same length")

    grid = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    count = np.zeros((grid_h, grid_w), dtype=np.int32)
    acc = np.zeros((grid_h, grid_w), dtype=np.float64)

    # map coords from [0, canvas_size) to [0, grid_w/h)
    x = coords[:, 0]
    y = coords[:, 1]

    sx = grid_w / canvas_size
    sy = grid_h / canvas_size

    ix = np.clip((x * sx).astype(int), 0, grid_w - 1)
    iy = np.clip((y * sy).astype(int), 0, grid_h - 1)

    for v, gx, gy in zip(values, ix, iy):
        if np.isnan(v):
            continue
        acc[gy, gx] += float(v)
        count[gy, gx] += 1

    mask = count > 0
    grid[mask] = (acc[mask] / count[mask]).astype(np.float32)

    return grid


def compute_slope_map_nan(g0: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """
    NaN-aware regression slope across slices 0,1,2:

        y ~ a * x + b,  x ∈ {0,1,2}

    Requires at least 2 valid (non-NaN) values per pixel. Otherwise, slope=NaN.
    """
    if not (g0.shape == g1.shape == g2.shape):
        raise ValueError("Input maps must have the same shape")

    H, W = g0.shape
    slope = np.full((H, W), np.nan, dtype=np.float32)

    x_all = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    y_stack = np.stack([g0, g1, g2], axis=0)   # (3,H,W)
    y_flat = y_stack.reshape(3, -1)            # (3,N)
    slope_flat = slope.reshape(-1)

    for idx in range(y_flat.shape[1]):
        y_pix = y_flat[:, idx]
        valid = ~np.isnan(y_pix)
        if valid.sum() < 2:
            continue
        xv = x_all[valid]
        yv = y_pix[valid]
        a, b = np.polyfit(xv, yv, 1)
        slope_flat[idx] = a

    return slope


# --------------------------- Per-patient analysis ---------------------------

def analyze_patient_lowres(patient_id: int,
                           fixed_id: int,
                           gene_name: str,
                           grid_h: int,
                           grid_w: int):
    """
    For one patient:
      - For each sid in {0,1,2}:
          * load gene values + original coords
          * if sid==fixed_id: coords_fixed = original coords
            else: coords_fixed = warped coords (fixed_id_sid_warped_coor.npy)
          * bin into a (grid_h x grid_w) grid via mean
      - compute NaN-aware slope map
      - visualize:
          * 2x3 figure: row0 = spot scatters; row1 = low-res grids
          * separate slope map figure
    """
    print(f"\n[INFO] Patient {patient_id}, FIXED_ID={fixed_id}, gene={gene_name}")

    lowres_maps = {}
    spot_coords_fixed = {}
    spot_vals = {}

    # --- build per-sample maps + keep raw spots ---
    for sid in SAMPLE_IDS:
        print(f"[INFO]  - Processing sample{sid} ...")
        gene_vals, coords_orig = load_h5ad_gene_and_coords(
            patient_id=patient_id,
            sample_id=sid,
            gene_name=gene_name,
        )

        # coords in fixed axis frame
        if sid == fixed_id:
            coords_fixed = coords_orig.astype(np.float32)
        else:
            coords_fixed = load_warped_coords(
                patient_id=patient_id,
                fixed_id=fixed_id,
                moving_id=sid,
                n_spots=coords_orig.shape[0],
            )

        # store for "raw spot" visualization
        spot_coords_fixed[sid] = coords_fixed
        spot_vals[sid] = gene_vals

        # bin to grid
        lowres = bin_spots_to_grid(
            coords=coords_fixed,
            values=gene_vals,
            grid_h=grid_h,
            grid_w=grid_w,
            canvas_size=CANVAS_SIZE,
        )
        lowres_maps[sid] = lowres

    g0 = lowres_maps[0]
    g1 = lowres_maps[1]
    g2 = lowres_maps[2]

    slope_map = compute_slope_map_nan(g0, g1, g2)

    # ---------------- Visualization: 2x3 (raw spots + grids) ----------------

    # shared color scale for gene expression (raw & grid)
    all_vals = np.concatenate([v[~np.isnan(v)] for v in spot_vals.values() if v.size > 0])
    if all_vals.size > 0:
        vmin_gene = np.percentile(all_vals, 2)
        vmax_gene = np.percentile(all_vals, 98)
    else:
        vmin_gene, vmax_gene = 0.0, 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for j, sid in enumerate(SAMPLE_IDS):
        coords = spot_coords_fixed[sid]
        vals = spot_vals[sid]

        # --- row 0: raw spots ---
        ax_scatter = axes[0, j]
        sc_plot = ax_scatter.scatter(
            coords[:, 0],
            coords[:, 1],
            c=vals,
            s=15,
            cmap="viridis",
            vmin=vmin_gene,
            vmax=vmax_gene,
        )
        ax_scatter.set_aspect("equal")
        ax_scatter.set_xlim(0, CANVAS_SIZE)
        ax_scatter.set_ylim(CANVAS_SIZE, 0)  # invert y to match image coordinates
        ax_scatter.set_title(f"Sample{sid} spots → axis_{fixed_id}")
        ax_scatter.axis("off")   # turn off axis completely

        # --- row 1: low-res grids ---
        ax_grid = axes[1, j]
        im_grid = ax_grid.imshow(
            lowres_maps[sid],
            cmap="viridis",
            vmin=vmin_gene,
            vmax=vmax_gene,
            origin="upper",
        )
        ax_grid.set_title(f"Sample{sid} grid {grid_h}×{grid_w}")
        ax_grid.axis("off")      # turn off axis completely

    fig.suptitle(
        f"Patient {patient_id} – AXIS spot + low-res grid for {gene_name}\n"
        f"(fixed sample {fixed_id}, grid {grid_h}×{grid_w})"
    )
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im_grid, cax=cbar_ax, label="Expression")
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

    # ---------------- Visualization: slope map (separate figure) ----------------

    valid_slope = slope_map[~np.isnan(slope_map)]
    if valid_slope.size > 0:
        vmin = np.percentile(valid_slope, 5)
        vmax = np.percentile(valid_slope, 95)
    else:
        vmin, vmax = -1.0, 1.0

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    im_slope = ax2.imshow(slope_map, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax2.set_title(
        f"Patient {patient_id} – slope across slices (0→2)\n{gene_name}, fixed sample {fixed_id}"
    )
    ax2.axis("off")  # turn off axis for slope map

    cax2 = fig2.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar2 = fig2.colorbar(im_slope, cax=cax2)
    cbar2.set_label("Regression slope (Δ expression per slice)")
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

    return lowres_maps, slope_map, spot_coords_fixed, spot_vals


def main():
    for pid in PATIENT_IDS:
        try:
            lowres_maps, slope_map, coords_dict, vals_dict = analyze_patient_lowres(
                patient_id=pid,
                fixed_id=FIXED_ID,
                gene_name=GENE_NAME,
                grid_h=GRID_H,
                grid_w=GRID_W,
            )
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[WARN] Skipping patient {pid}: {e}")


if __name__ == "__main__":
    main()
