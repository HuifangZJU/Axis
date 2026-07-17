from pathlib import Path
from typing import Dict, Union
from matplotlib import pyplot as plt
import json
import numpy as np
import scanpy as sc
from typing import List
from typing import Dict, Union, Optional, Sequence
def load_group_with_axis_coords(
    root_dir: str,
    group_id: Union[int, str],
    unified_subdir: str,
    sample_name_to_id: Dict[str, int],
    obsm_key: str = "spatial_axis",
    axis_roi: Optional[Sequence[float]] = None,
):
    """
    Load original h5ad files for a group and attach AXIS-unified coordinates
    as adata.obsm[obsm_key], using coordinates stored in a given unified
    result folder (e.g. 'unified_refer4').

    Optionally restrict to a rectangular ROI in AXIS space:
        axis_roi = [xmin, ymin, xmax, ymax]
    Any cell whose unified coordinate is outside this box is removed.

    Layout
    ------
    root_dir/
      original_data/
        group_{group_id}/
          {sample_name}.h5ad
      output/
        group_{group_id}/
          {unified_subdir}/
            meta.json
            coords_slice{sid}_to_{ref_id}.npy

    Parameters
    ----------
    root_dir : str
        Base directory that contains `original_data/` and `output/`.
    group_id : int or str
        Group index, used as "group_{group_id}" folder name.
    unified_subdir : str
        Name of the subfolder under output/group_{group_id} where
        unify_group_to_reference wrote the coordinate files.
    sample_name_to_id : dict[str, int]
        Mapping from sample name (h5ad prefix) to AXIS sample id.
    obsm_key : str
        Name of the obsm key to store the unified coordinates, default "spatial_axis".
    axis_roi : [xmin, ymin, xmax, ymax] or None
        Rectangular ROI in AXIS coordinate space. If provided, cells whose
        unified coordinates (x, y) lie outside this box are removed.

    Returns
    -------
    adata_dict : dict[str, AnnData]
        Dictionary keyed by sample_name, with AnnData objects that
        have adata.obsm[obsm_key] set to unified coordinates (possibly ROI-filtered).
    """
    root_dir = Path(root_dir)
    group_folder = f"group_{group_id}"

    original_dir = root_dir / "original_data" / group_folder
    group_dir = root_dir / "output" / group_folder
    unified_dir = group_dir / unified_subdir
    meta_path = unified_dir / "meta.json"

    if not original_dir.is_dir():
        raise NotADirectoryError(f"Original data dir not found: {original_dir}")
    if not unified_dir.is_dir():
        raise NotADirectoryError(f"Unified dir not found: {unified_dir}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found in {unified_dir}")

    # Load meta.json (for logging / sanity)
    with open(meta_path, "r") as f:
        meta = json.load(f)


    # Parse ROI if provided
    if axis_roi is not None:
        if len(axis_roi) != 4:
            raise ValueError("axis_roi must be [xmin, ymin, xmax, ymax].")
        xmin, ymin, xmax, ymax = map(float, axis_roi)
        print(f"[INFO] Applying AXIS ROI: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
    else:
        xmin = ymin = xmax = ymax = None

    adata_dict = {}

    for sample_name, sid in sample_name_to_id.items():
        sid = int(sid)

        h5ad_path = original_dir / f"{sample_name}.h5ad"
        if not h5ad_path.is_file():
            print(f"[WARN] h5ad for sample '{sample_name}' not found: {h5ad_path}, skipping.")
            continue

        coord_path = unified_dir / f"coords_slice{sid}_to_refer.npy"
        if not coord_path.is_file():
            print(f"[WARN] Unified coords not found for sample '{sample_name}' (sid={sid}): "
                  f"{coord_path}, skipping.")
            continue

        # Load unified coords and original AnnData
        coords = np.load(coord_path).astype(np.float32)
        adata = sc.read_h5ad(h5ad_path)

        sc.pp.filter_cells(adata, min_counts=100)
        sc.pp.filter_genes(adata, min_cells=10)

        # Sanity check *before* ROI filtering
        if coords.shape[0] != adata.n_obs:
            raise ValueError(
                f"Row mismatch for '{sample_name}': "
                f"coords={coords.shape[0]}, n_obs={adata.n_obs} (file: {h5ad_path})"
            )

        # Apply ROI filter if requested (in AXIS space)
        if axis_roi is not None:
            x = coords[:, 0]
            y = coords[:, 1]
            mask = (
                (x >= xmin) & (x <= xmax) &
                (y >= ymin) & (y <= ymax)
            )

            n_keep = int(mask.sum())
            if n_keep == 0:
                print(f"[WARN] All cells of '{sample_name}' lie outside the ROI; skipping this sample.")
                continue

            coords = coords[mask]
            adata = adata[mask].copy()
            print(f"[INFO] '{sample_name}': kept {n_keep} cells inside ROI.")

        # Attach AXIS coords
        adata.obsm[obsm_key] = coords
        adata_dict[sample_name] = adata
        print(f"[INFO] {sample_name}: attached obsm['{obsm_key}'] with shape {coords.shape}")

    return adata_dict


def assign_pixel_coords_dict(
    adata_dict: Dict[str, "anndata.AnnData"],
    grid_size: float = 50.0,
    spatial_key: str = "spatial_axis",
    out_key: str = "spatial_axis_pixel",
    visualize: bool = False,
):
    """
    Assign integer pixel/grid coordinates to each cell in a dict of AnnData objects,
    based on a shared AXIS coordinate system, and optionally visualize the number
    of cells/spots per grid cell for each slice separately (with slice-wise colorbars).

    Parameters
    ----------
    adata_dict : dict[str, AnnData]
        Mapping from sample_name -> AnnData, each with
        adata.obsm[spatial_key] of shape (n_cells, 2).
    grid_size : float
        Size of each pixel/grid cell in AXIS coordinate units.
    spatial_key : str
        Key in .obsm containing unified spatial coordinates (e.g. "spatial_axis").
    out_key : str
        Key to store the pixel/grid coordinates in .obsm
        (default "spatial_axis_pixel").
    visualize : bool
        If True, plot per-slice heatmaps of cell counts per grid cell
        in a 3-column subplot layout, each with its own colorbar.

    Returns
    -------
    adata_dict : dict[str, AnnData]
        Same dict, with adata.obsm[out_key] added for each AnnData.
    """
    if not adata_dict:
        raise ValueError("adata_dict is empty.")

    # 1. Global coordinate bounds across all samples
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

    sample_names = list(adata_dict.keys())
    slice_grids = []  # list of (ny, nx) arrays in same order as sample_names

    # 2. Assign pixel indices and build per-slice grids
    for sample_name in sample_names:
        adata = adata_dict[sample_name]
        coords = adata.obsm[spatial_key]  # (n_cells, 2)

        # Compute grid indices
        ix = np.floor((coords[:, 0] - min_x) / grid_size).astype(int)
        iy = np.floor((coords[:, 1] - min_y) / grid_size).astype(int)

        # Clip to valid range
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)

        # Store as (ix, iy) in obsm
        axis_pixel_spatial = np.column_stack([ix, iy])
        adata.obsm[out_key] = axis_pixel_spatial

        # Per-slice grid count
        grid_counts = np.zeros((ny, nx), dtype=int)
        for xg, yg in zip(ix, iy):
            grid_counts[yg, xg] += 1
        slice_grids.append(grid_counts)

    # 3. Optional visualization: per-slice grids with slice-wise colorbars
    if visualize:
        n_slices = len(sample_names)
        n_cols = 3
        n_rows = int(np.ceil(n_slices / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 4 * n_rows),
            squeeze=False,
        )

        for idx, (sample_name, grid_counts) in enumerate(zip(sample_names, slice_grids)):
            r = idx // n_cols
            c = idx % n_cols
            ax = axes[r, c]

            # Per-slice color scaling
            vmin = 0
            vmax = grid_counts.max() if grid_counts.max() > 0 else 1

            im = ax.imshow(
                grid_counts,
                origin="lower",
                interpolation="nearest",
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(sample_name, fontsize=9)
            ax.set_xlabel("Grid index (x)")
            ax.set_ylabel("Grid index (y)")

            # Slice-wise colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Cells per grid cell", fontsize=8)

        # Hide unused subplots
        for idx in range(n_slices, n_rows * n_cols):
            r = idx // n_cols
            c = idx % n_cols
            axes[r, c].axis("off")

        fig.suptitle(
            f"Slice-wise grid occupancy (grid_size={grid_size})",
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    return adata_dict






def save_group_with_axis_coords(
    root_dir: str,
    group_id: int,
    adata_dict: dict,
    out_subdir: str = "h5ads",
):
    """
    Save AnnData objects (with spatial_axis already attached) to:
        {root_dir}/output/group_{group_id}/{out_subdir}/{sample_name}.h5ad
    """
    root_dir = Path(root_dir)
    group_folder = f"group_{group_id}"

    out_dir = root_dir / "output" / group_folder / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for sample_name, adata in adata_dict.items():
        out_path = out_dir / f"{sample_name}.h5ad"
        print(f"[INFO] Writing {sample_name} -> {out_path}")
        adata.write(out_path)
    print(f"[INFO] Done. Saved {len(adata_dict)} files to {out_dir}")


root_dir = "/media/huifang/data/registration/xenium/mouse_brain"
group_id = 0
# unified_subdir = "TgCRND8_refer0"  # the folder produced by unify_group_to_reference
unified_subdir = "widetype_refer4"  # the folder produced by unify_group_to_reference
# unified_subdir = "unified_refer4"  # the folder produced by unify_group_to_reference

sample_name_to_id = {
    # "xenium_Tg_2.5m_domains_shared": 0,
    # "xenium_Tg_5.7m_domains_shared": 1,
    # "xenium_Tg_17.9m_domains_shared": 2,
    "xenium_WT_2.5m_domains_shared": 3,
    "xenium_WT_5.7m_domains_shared": 4,
    "xenium_WT_13.4m_domains_shared":5,
}
# axis_roi = [0.0, 0.0, 1000.0, 900.0]
adata_dict = load_group_with_axis_coords(
    root_dir=root_dir,
    group_id=group_id,
    unified_subdir=unified_subdir,
    sample_name_to_id=sample_name_to_id,
    obsm_key="spatial_axis",  # or any name you like
    # axis_roi=axis_roi
)

adata_dict = assign_pixel_coords_dict(
    adata_dict,
    grid_size=10,
    spatial_key="spatial_axis",
    out_key="spatial_axis_pixel",
    visualize=True,  # now shows slice-wise 3-column plots
)
# for sample_name, ad in adata_dict.items():
#     print(sample_name, ad.obsm["spatial_axis_pixel"].shape)

# save_group_with_axis_coords(
#     root_dir=root_dir,
#     group_id=group_id,
#     adata_dict=adata_dict,
#     out_subdir="h5ads/widetype/",  # as you requested
# )
