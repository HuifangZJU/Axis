#!/usr/bin/env python
"""
Project iSTAR-imputed spatial gene data to AXIS common coordinates by warping
coordinates (not the image), using find_inverse_warp_coords.

Assumptions:
- AXIS outputs per patient:
    /media/huifang/data/registration/SCC/huifang/output/group_{pid}/
        {fixed_id}_{moving_id}_*flow.npy      # 512×512 flow

- iSTAR single-slice outputs per patient/sample:
    /media/huifang/data/registration/SCC/istar/single_input/group_{pid}/axis_{sid}/sample{sid}/
        gene-names.txt
        cnts-super/{gene_name}.pickle         # each ~H×W 2D array



  where uv_coords are MOVING-frame coordinates and warped_coords are FIXED-frame
  coordinates.

We visualize a chosen gene before/after warping for all patients, with sample0
as the fixed frame.
"""

from pathlib import Path
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from axis.coordinate_utils import find_inverse_warp_coords


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

AXIS_ROOT = Path("/media/huifang/data/registration/SCC/huifang/output")
ISTAR_ROOT = Path("/media/huifang/data/registration/SCC/istar/single_input")

PATIENT_IDS = [2, 5, 9, 10]
FIXED_SAMPLE_ID = 0           # sample0 is fixed for each patient
MOVING_SAMPLE_IDS = [1, 2]    # warp sample1 and sample2 into sample0 frame

GENE_TO_VISUALIZE = "A2ML1"    # change as needed



# ---------------------------------------------------------------------
# I/O Helpers
# ---------------------------------------------------------------------

def load_axis_flow(patient_id: int, fixed_id: int, moving_id: int):
    """
    Load AXIS flow for (fixed, moving) sample pair.

    Expected canonical shape: (2, H, W).

    If the file is accidentally saved as (H, W, 2), we convert it
    to (2, H, W) so that downstream code and find_inverse_warp_coords
    always see [2, H, W].
    """
    group_dir = AXIS_ROOT / f"group_{patient_id}"
    if not group_dir.is_dir():
        raise FileNotFoundError(f"AXIS group dir not found: {group_dir}")

    pattern = str(group_dir / f"{fixed_id}_{moving_id}_*flow.npy")
    matches = glob.glob(pattern)
    if not matches:
        fallback = group_dir / f"{fixed_id}_{moving_id}_flow.npy"
        if not fallback.is_file():
            raise FileNotFoundError(
                f"No flow file found for {fixed_id}_{moving_id} in {group_dir}"
            )
        flow_path = fallback
    else:
        flow_path = matches[0]

    flow = np.load(flow_path)
    flow = np.asarray(flow, dtype=np.float32)

    # Ensure canonical shape (2, H, W)
    if flow.ndim != 3:
        raise ValueError(f"Unexpected flow dims {flow.shape} in {flow_path}")

    if flow.shape[0] == 2:
        # already (2, H, W)
        return flow
    elif flow.shape[-1] == 2:
        # (H, W, 2) -> (2, H, W)
        flow = np.moveaxis(flow, -1, 0)
        return flow
    else:
        raise ValueError(f"Flow must have a channel dimension of size 2, got {flow.shape}")


def upsample_flow(flow, target_shape):
    """
    Upsample flow from (2, Hf, Wf) to (2, Hg, Wg) using interpolation.

    flow:         np.ndarray, shape (2, Hf, Wf)
    target_shape: (Hg, Wg) of the gene map

    NOTE:
    - We assume both flow and gene grids are derived from the same
      original image, so we only interpolate spatially and keep the
      displacement units as is (no extra scaling here).
    """
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"upsample_flow expects (2, H, W), got {flow.shape}")

    _, Hf, Wf = flow.shape
    Hg, Wg = target_shape

    sy = Hg / Hf
    sx = Wg / Wf

    # zoom factors: (channels, H, W) -> (1, sy, sx)
    flow_up = zoom(flow, (1.0, sy, sx), order=1)

    return flow_up  # shape (2, Hg, Wg)


def load_istar_gene_map(patient_id: int, sample_id: int, gene_name: str):
    """
    Load a single iSTAR-imputed gene map (2D array) for a patient/sample.
    """
    sample_root = (
        ISTAR_ROOT
        / f"group_{patient_id}"
        / f"axis_{sample_id}"
        / f"sample{sample_id}"
    )
    cnts_super_dir = sample_root / "cnts-super"
    if not cnts_super_dir.is_dir():
        raise FileNotFoundError(f"cnts-super dir not found: {cnts_super_dir}")

    pkl_path = cnts_super_dir / f"{gene_name}.pickle"
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Gene map not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        arr = pickle.load(f)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Gene map {pkl_path} must be 2D, got shape {arr.shape}")
    return arr  # (Hg, Wg)





def warp_coords_moving_to_fixed(gene_map, flow):
    """
    Given:
      gene_map: (H, W) in moving frame
      flow:     (H, W, 2) flow in same resolution

    Build moving-frame pixel coordinates and warp them to fixed frame
    using find_inverse_warp_coords (which you specified as moving->fixed).
    """
    H, W = gene_map.shape

    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    uv_moving = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)  # (N, 2)
    uv_fixed = find_inverse_warp_coords(uv_moving, flow)
    uv_fixed = np.asarray(uv_fixed, dtype=np.float32)  # (N, 2)
    return uv_moving, uv_fixed


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def scatter_gene_map(coords, gene_map, ax, title="", vmin=None, vmax=None, s=1.0):
    """
    coords: (N,2) [x,y] for each pixel, flattened
    gene_map: (H,W) 2D array
    """
    H, W = gene_map.shape
    vals = np.log1p(gene_map).ravel()  # log1p for dynamic range

    if vmin is None or vmax is None:
        vmin = 0.0
        vmax = np.percentile(vals, 99)

    x = coords[:, 0]
    y = coords[:, 1]

    sc = ax.scatter(
        x,
        y,
        c=vals,
        s=s,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()  # match image coordinates
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return sc


def visualize_gene_for_patient(patient_id: int, gene_name: str):
    """
    For one patient, plot a 2x3 grid:

        Cols:   Sample0         Sample1         Sample2
        Row 0:  original        original        original
        Row 1:  warped(=orig)   warped->0       warped->0

    - Sample0 is the fixed AXIS frame.
    - Sample1/2 are projected into sample0 coordinates using AXIS flow.
    """
    print(f"\n[INFO] Visualizing patient {patient_id}, gene {gene_name}")

    # ---------- Load fixed sample0 ----------
    try:
        gene_fixed = load_istar_gene_map(patient_id, FIXED_SAMPLE_ID, gene_name)
    except FileNotFoundError:
        print(f"[WARN] Patient {patient_id}: sample0 missing {gene_name}, skipping.")
        return

    Hf, Wf = gene_fixed.shape
    yy_f, xx_f = np.meshgrid(
        np.arange(Hf, dtype=np.float32),
        np.arange(Wf, dtype=np.float32),
        indexing="ij",
    )
    coords_fixed = np.stack([xx_f.reshape(-1), yy_f.reshape(-1)], axis=1)

    # ---------- Load moving samples ----------
    gene_moving_maps = {}
    coords_moving_maps = {}
    coords_warped_maps = {}

    all_vals = [np.log1p(gene_fixed).ravel()]

    for mid in MOVING_SAMPLE_IDS:
        try:
            gm = load_istar_gene_map(patient_id, mid, gene_name)
        except FileNotFoundError:
            print(f"[WARN] patient {patient_id}, sample{mid} missing {gene_name}, skip that column.")
            continue

        gene_moving_maps[mid] = gm
        all_vals.append(np.log1p(gm).ravel())

        Hm, Wm = gm.shape
        yy_m, xx_m = np.meshgrid(
            np.arange(Hm, dtype=np.float32),
            np.arange(Wm, dtype=np.float32),
            indexing="ij",
        )
        coords_m = np.stack([xx_m.reshape(-1), yy_m.reshape(-1)], axis=1)
        coords_moving_maps[mid] = coords_m

        # load & upsample flow, then warp coords to fixed frame
        flow = load_axis_flow(patient_id, FIXED_SAMPLE_ID, mid)
        flow_up = upsample_flow(flow, gm.shape)  # (Hm, Wm, 2)

        _, coords_warped = warp_coords_moving_to_fixed(
            gm, flow_up
        )
        coords_warped_maps[mid] = coords_warped

    if len(gene_moving_maps) == 0:
        print(f"[WARN] No moving samples with {gene_name} for patient {patient_id}, skipping.")
        return

    all_vals = np.concatenate(all_vals)
    vmin = 0.0
    vmax = np.percentile(all_vals, 99)

    # ---------- Build 2x3 grid ----------
    # Columns: 0 = sample0, 1 = sample1, 2 = sample2
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)

    # --- Row 0: original gene maps ---
    # Col 0: sample0 original (fixed)
    sc0 = scatter_gene_map(
        coords_fixed,
        gene_fixed,
        ax=axes[0, 0],
        title=f"Sample0 (fixed) – original",
        vmin=vmin,
        vmax=vmax,
        s=1.0,
    )

    # Cols 1 & 2: sample1, sample2 original
    for col, mid in enumerate([1, 2], start=1):
        ax = axes[0, col]
        if mid in gene_moving_maps:
            scatter_gene_map(
                coords_moving_maps[mid],
                gene_moving_maps[mid],
                ax=ax,
                title=f"Sample{mid} – original",
                vmin=vmin,
                vmax=vmax,
                s=1.0,
            )
        else:
            ax.axis("off")
            ax.set_title(f"Sample{mid} – missing", fontsize=9)

    # --- Row 1: warped gene maps ---
    # Col 0: sample0 warped (identical to original, same coords)
    scatter_gene_map(
        coords_fixed,
        gene_fixed,
        ax=axes[1, 0],
        title=f"Sample0 (fixed) – reference frame",
        vmin=vmin,
        vmax=vmax,
        s=1.0,
    )

    # Cols 1 & 2: sample1→0, sample2→0 in fixed coordinates
    for col, mid in enumerate([1, 2], start=1):
        ax = axes[1, col]
        if mid in gene_moving_maps:
            scatter_gene_map(
                coords_warped_maps[mid],
                gene_moving_maps[mid],
                ax=ax,
                title=f"Sample{mid} → Sample0 (warped coords)",
                vmin=vmin,
                vmax=vmax,
                s=1.0,
            )
        else:
            ax.axis("off")
            ax.set_title(f"Sample{mid} – no warp", fontsize=9)

    fig.suptitle(
        f"Patient {patient_id} – {gene_name} before/after AXIS coordinate warping\n"
        f"(Sample0 as fixed frame)",
        y=0.99,
    )
    cbar = fig.colorbar(sc0, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("log1p(expression)", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def visualize_gene_all_patients(gene_name: str):
    for pid in PATIENT_IDS:
        try:
            visualize_gene_for_patient(pid, gene_name)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping patient {pid}: {e}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    visualize_gene_all_patients(GENE_TO_VISUALIZE)
