#!/usr/bin/env python


import glob
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt


# ---------------- Configuration ---------------- #

ROOT = Path("/media/huifang/data/registration/phenocycler/huifang")

H5AD_DIR = ROOT / "network_input"
AXIS_OUT_DIR = ROOT / "pairwise_output"

# Where to save updated h5ad with spatial_axis
OUTPUT_H5AD_DIR = ROOT / "H5ADs_axis"

DATA_NAMES = ["LUAD_2_A", "TSU_35"]
MODALITIES = ["protein", "xenium"]

# Coordinate key in original h5ad used for registration
COORD_KEY = "spatial"  # change to "spatial" if that's what you used

# Choose fixed reference for spatial_axis assignment
REF_MODALITY = "xenium"          # can switch to "xenium" or "visium"

# Colors for modalities in plots
MODALITY_COLORS = {
    "protein": "tab:blue",
    "xenium": "tab:orange",
    "visium": "tab:green",
}


# ---------------- I/O Helpers ---------------- #

def load_all_h5ads_for_dataset(data_name: str) -> Dict[str, sc.AnnData]:
    """
    Load all modalities for a given dataset.

    Returns:
        dict: modality -> AnnData
    """
    adatas = {}
    for mod in MODALITIES:
        h5_path = H5AD_DIR / f"{data_name}_{mod}_trans.h5ad"
        if not h5_path.is_file():
            raise FileNotFoundError(f"[ERROR] Missing h5ad: {h5_path}")
        adata = sc.read_h5ad(h5_path)

        if COORD_KEY not in adata.obsm_keys():
            raise KeyError(f"[ERROR] {COORD_KEY} missing in {h5_path}")
        adatas[mod] = adata
    return adatas


def load_warped_coords(
    data_name: str,
    fixed_modality: str,
    moving_modality: str,
    n_spots: int
) -> np.ndarray:
    """
    Load warped coordinates for (fixed_modality, moving_modality) pair.

    Path pattern:
        AXIS_OUT_DIR / data_name / f"{fixed_modality}_{moving_modality}_warped_coor.npy"
        or with a wildcard if there's extra naming, e.g. "{...}_something_warped_coor.npy"

    Returns:
        warped_coords: np.ndarray of shape (N_spots, 2)
    """
    group_dir = AXIS_OUT_DIR / data_name
    if not group_dir.is_dir():
        raise FileNotFoundError(f"[ERROR] AXIS group dir not found: {group_dir}")

    # Robust: allow extra tokens between modality names and suffix
    pattern = str(group_dir / f"{fixed_modality}_{moving_modality}_*warped_coor.npy")
    matches = glob.glob(pattern)
    if not matches:
        # Fallback: exact name without wildcard
        fallback = group_dir / f"{fixed_modality}_{moving_modality}_warped_coor.npy"
        if not fallback.is_file():
            raise FileNotFoundError(
                f"[ERROR] No warped_coor file for {fixed_modality}_{moving_modality} in {group_dir}"
            )
        warped_path = fallback
    else:
        warped_path = matches[0]

    warped = np.load(warped_path).astype(np.float32)

    if warped.ndim != 2 or warped.shape[1] != 2:
        raise ValueError(f"[ERROR] Unexpected warped_coor shape {warped.shape} in {warped_path}")

    if warped.shape[0] != n_spots:
        print(
            f"[WARN] warped_coor rows ({warped.shape[0]}) != n_spots ({n_spots}) "
            f"for {data_name} {fixed_modality}_{moving_modality}"
        )

    return warped


# ---------------- 1. Visual Check ---------------- #

def visualize_overlays_for_dataset(data_name: str, adatas: Dict[str, sc.AnnData]) -> None:
    """
    For a given dataset and all modalities, generate figures where each modality is used as
    the reference frame. In each reference frame, overlay coordinates from all modalities.

    Uses:
       - Fixed modality coord:   adata[fixed].obsm[COORD_KEY]
       - Moving modality coords: warped_coor.npy (moving -> fixed)
    """
    print(f"\n[INFO] Visualizing overlays for dataset: {data_name}")

    for fixed_mod in MODALITIES:
        print(f"[INFO]  Fixed modality: {fixed_mod}")

        fig, ax = plt.subplots(figsize=(6, 6))

        for m in MODALITIES:
            adata_m = adatas[m]
            n_spots = adata_m.n_obs

            # if m == fixed_mod:
            #     coords = np.asarray(adata_m.obsm[COORD_KEY], dtype=int)
            # else:
            #     coords = load_warped_coords(
            #         data_name=data_name,
            #         fixed_modality=fixed_mod,
            #         moving_modality=m,
            #         n_spots=n_spots,
            #     )
            coords = np.asarray(adata_m.obsm[COORD_KEY], dtype=int)

            color = MODALITY_COLORS.get(m, None)
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=0.5,
                alpha=0.6,
                label=m,
                c=color,
            )

        ax.set_title(f"{data_name}: overlay in {fixed_mod} frame")
        ax.set_aspect("equal")
        ax.invert_yaxis()  # match typical image coordinates
        ax.axis("off")
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        plt.show()


# ---------------- 2. Save spatial_axis to new h5ad ---------------- #

def compute_axis_coords_for_dataset(
    data_name: str,
    adatas: Dict[str, sc.AnnData],
    ref_modality: str,
) -> Dict[str, np.ndarray]:
    """
    For each modality in a dataset, compute coordinates in the reference (ref_modality) frame.

    Returns:
        coord_dict: modality -> np.ndarray of shape (N_spots, 2)
    """
    if ref_modality not in adatas:
        raise ValueError(f"[ERROR] Reference modality {ref_modality} not found in {data_name}")

    coord_dict: Dict[str, np.ndarray] = {}

    # Reference modality: use its own original coordinates
    ref_adata = adatas[ref_modality]
    coord_dict[ref_modality] = np.asarray(ref_adata.obsm[COORD_KEY], dtype=np.float32)

    # Other modalities: use warped coordinates into ref_modality frame
    for m in MODALITIES:
        if m == ref_modality:
            continue
        adata_m = adatas[m]
        n_spots = adata_m.n_obs

        warped = load_warped_coords(
            data_name=data_name,
            fixed_modality=ref_modality,
            moving_modality=m,
            n_spots=n_spots,
        )
        coord_dict[m] = warped

    return coord_dict


def save_h5ad_with_spatial_axis(
    data_name: str,
    adatas: Dict[str, sc.AnnData],
    coord_dict: Dict[str, np.ndarray],
    out_root: Path,
    ref_modality: str,
) -> None:
    """
    Save updated h5ad files with a new obsm["spatial_axis"] containing AXIS coordinates
    in the common reference frame.

    Files will be written to:
        out_root / f"{data_name}_{modality}_axis.h5ad"
    """
    out_root.mkdir(parents=True, exist_ok=True)

    for m in MODALITIES:
        adata = adatas[m]
        axis_coords = coord_dict[m]

        if axis_coords.shape[0] != adata.n_obs or axis_coords.shape[1] != 2:
            raise ValueError(
                f"[ERROR] spatial_axis shape {axis_coords.shape} does not match obs ({adata.n_obs},2) "
                f"for dataset {data_name}, modality {m}"
            )

        # Assign the new common-frame coordinates
        adata.obsm["spatial_axis"] = axis_coords

        out_path = out_root / f"{data_name}_{m}_axis.h5ad"
        print(
            f"[INFO] Saving updated h5ad with spatial_axis "
            f"(ref={ref_modality}) to: {out_path}"
        )
        adata.write(out_path)


# ---------------- Main driver ---------------- #

def main():
    OUTPUT_H5AD_DIR.mkdir(parents=True, exist_ok=True)

    for data_name in DATA_NAMES:
        print(f"\n========== Dataset: {data_name} ==========")

        # 1) Load all modalities' h5ad
        adatas = load_all_h5ads_for_dataset(data_name)


        # 2) Visual check: overlay for each modality as reference
        visualize_overlays_for_dataset(data_name, adatas)

        # 3) Compute axis coords in chosen reference modality
        coord_dict = compute_axis_coords_for_dataset(
            data_name=data_name,
            adatas=adatas,
            ref_modality=REF_MODALITY,
        )

        # 4) Save new h5ad with obsm["spatial_axis"]
        save_h5ad_with_spatial_axis(
            data_name=data_name,
            adatas=adatas,
            coord_dict=coord_dict,
            out_root=OUTPUT_H5AD_DIR,
            ref_modality=REF_MODALITY,
        )


if __name__ == "__main__":
    main()
