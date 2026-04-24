#!/usr/bin/env python
import glob
import pickle
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
# ------------------- Config -------------------

AXIS_ROOT = Path("/media/huifang/data/registration/SCC/huifang/pairwise_output")
ISTAR_ROOT = Path("/media/huifang/data/registration/SCC/istar/single_input")

PATIENT_IDS = [2, 5, 9, 10]   # SCC patients
SAMPLE_IDS = [0, 1, 2]        # slices 0,1,2
FIXED_ID = 0                  # treat sample 0 as fixed frame
GENE_NAME = "A2ML1"            # change to your gene of interest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------- Flow & gene loaders -------------------

def load_flow_np(patient_id: int, fixed_id: int, moving_id: int) -> np.ndarray:
    """
    Load AXIS flow for (fixed, moving) pair from:
      AXIS_ROOT/group_{patient}/{fixed_id}_{moving_id}_*flow.npy

    Normalize to shape (2, H, W), same ordering as in the network.
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

    flow = np.load(flow_path).astype(np.float32)
    if flow.ndim != 3:
        raise ValueError(f"Unexpected flow dims {flow.shape} in {flow_path}")

    # Normalize to (2, H, W)
    if flow.shape[0] == 2:
        pass
    elif flow.shape[-1] == 2:
        flow = np.moveaxis(flow, -1, 0)  # (H,W,2) -> (2,H,W)
    else:
        raise ValueError(f"Flow must have channel dimension 2, got {flow.shape}")

    return flow


def get_flow_resolution(patient_id: int, fixed_id: int):
    """
    Infer (H_f, W_f) from any flow file with this fixed_id.
    """
    group_dir = AXIS_ROOT / f"group_{patient_id}"
    pattern = str(group_dir / f"{fixed_id}_*flow.npy")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No flow files for patient {patient_id} with fixed_id={fixed_id} in {group_dir}"
        )
    f = np.load(matches[0]).astype(np.float32)
    if f.ndim != 3:
        raise ValueError(f"Unexpected flow dims {f.shape} in {matches[0]}")

    if f.shape[0] == 2:
        _, Hf, Wf = f.shape
    elif f.shape[-1] == 2:
        Hf, Wf, _ = f.shape
    else:
        raise ValueError(f"Flow must have channel dimension 2, got {f.shape}")
    return Hf, Wf


def load_gene_map_np(patient_id: int, axis_id: int, sample_id: int, gene_name: str) -> np.ndarray:
    """
    Load iSTAR-imputed gene map from:
      ISTAR_ROOT/group_{patient}/axis_{axis_id}/sample{sample_id}/cnts-super/{gene_name}.pickle

    Returns 2D numpy array (H_g, W_g).
    """
    sample_root = (
        ISTAR_ROOT
        / f"group_{patient_id}"
        / f"axis_{axis_id}"
        / f"sample{sample_id}"
    )
    cnts_dir = sample_root / "cnts-super"
    pkl_path = cnts_dir / f"{gene_name}.pickle"

    if not pkl_path.is_file():
        raise FileNotFoundError(f"Gene map missing: {pkl_path}")

    with open(pkl_path, "rb") as f:
        arr = pickle.load(f)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Gene map {pkl_path} must be 2D, got {arr.shape}")

    return arr


def load_mask_np(patient_id: int, axis_id: int, sample_id: int) -> np.ndarray:
    """
    Load mask-small.png from the corresponding iSTAR result folder:

      ISTAR_ROOT/group_{patient}/axis_{axis_id}/sample{sample_id}/mask-small.png

    Returns 2D numpy array (H_g, W_g) with values 0/1 (or 0/255).
    """
    sample_root = (
        ISTAR_ROOT
        / f"group_{patient_id}"
        / f"axis_{axis_id}"
        / f"sample{sample_id}"
    )
    mask_path = sample_root / "mask-small.png"

    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file missing: {mask_path}")

    mask_img = Image.open(mask_path)
    mask_arr = np.array(mask_img)

    # If mask is RGB, take one channel; if it's 0/255, binarize
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    mask_bin = (mask_arr > 0).astype(np.uint8)  # 1 for tissue, 0 for background

    return mask_bin


# ------------------- Exact SpatialTransformer-style warp -------------------

def warp_gene_with_axis_flow(gene_map_np: np.ndarray, flow_np: np.ndarray) -> np.ndarray:
    """
    Warp gene map using the same logic as your SpatialTransformer_block.

    - Resize gene map to flow resolution (H_f, W_f).
    - Apply pull-based warp (grid_sample) using flow as displacement field.
    - Return warped gene map in fixed frame, shape (H_f, W_f).

    gene_map_np: 2D numpy array (H_g, W_g) (may contain NaNs on background).
    flow_np:     2×H_f×W_f or H_f×W_f×2 displacement field (moving -> fixed).
    """
    gene_np = np.asarray(gene_map_np, dtype=np.float32)
    src = torch.from_numpy(gene_np).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,Hg,Wg]

    f = np.asarray(flow_np, dtype=np.float32)
    if f.ndim != 3:
        raise ValueError(f"Flow must be 3D, got {f.shape}")

    # (2,Hf,Wf)
    if f.shape[0] == 2:
        pass
    elif f.shape[-1] == 2:
        f = np.moveaxis(f, -1, 0)
    else:
        raise ValueError(f"Flow must have channel dim 2, got {f.shape}")

    _, Hf, Wf = f.shape
    flow = torch.from_numpy(f).float().unsqueeze(0).to(device)  # [1,2,Hf,Wf]

    # Resize src to flow resolution so src & flow share spatial shape
    src = F.interpolate(src, size=(Hf, Wf), mode="bilinear", align_corners=False)  # [1,1,Hf,Wf]

    # ---- EXACTLY mimic SpatialTransformer_block ----

    shape = flow.shape[2:]              # (Hf, Wf)
    H, W = shape

    vectors = [torch.arange(0, s, device=device, dtype=torch.float32) for s in shape]
    grids = torch.meshgrid(vectors, indexing="ij")  # grids[0]=y, grids[1]=x

    grid = torch.stack(grids, dim=0).unsqueeze(0)   # [1,2,H,W]

    # new_locs = grid + flow (pixel coordinates)
    new_locs = grid + flow                          # [1,2,H,W]

    # normalize each channel to [-1,1]
    for i, s in enumerate(shape):
        new_locs[:, i, ...] = 2.0 * (new_locs[:, i, ...] / (s - 1.0) - 0.5)

    # permute to [N,H,W,2] and swap to (x,y)
    new_locs = new_locs.permute(0, 2, 3, 1)   # [1,H,W,2] with (y_norm,x_norm)
    new_locs = new_locs[..., [1, 0]]          # -> (x_norm,y_norm)

    warped = F.grid_sample(
        src,
        new_locs,
        align_corners=False,
        mode="bilinear",
        padding_mode="border",
    )  # [1,1,Hf,Wf]

    warped_np = warped.squeeze().detach().cpu().numpy().astype(np.float32)  # (Hf,Wf)
    return warped_np


# ------------------- NaN-aware regression -------------------

def compute_slope_map_nan(g0: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """
    Pixel-wise regression slope across slices 0,1,2:

        y ~ a * x + b,  x ∈ {0,1,2}.

    NaN-aware: for each pixel, use only non-NaN entries across slices.
    Require at least 2 valid points; otherwise slope is NaN.
    """
    if not (g0.shape == g1.shape == g2.shape):
        raise ValueError("All warped gene maps must have the same shape")

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
            continue  # leave as NaN
        xv = x_all[valid]
        yv = y_pix[valid]
        a, b = np.polyfit(xv, yv, 1)
        slope_flat[idx] = a

    return slope


# ------------------- Per-patient analysis -------------------

def analyze_patient_pull(patient_id: int, fixed_id: int, gene_name: str):
    """
    For one patient:
      - Determine flow resolution (H_f, W_f) from any flow with fixed_id.
      - For each sid in {0,1,2}:
          * Load gene map in its own axis_{sid}/sample{sid} frame.
          * Load mask-small.png and set background to NaN.
          * If sid == fixed_id: warp with zero flow (identity).
            Else: load flow_{fixed_id,sid}, warp gene with warp_gene_with_axis_flow.
      - Compute NaN-aware pixel-wise slope map and visualize.
    """
    print(f"\n[INFO] Patient {patient_id}, FIXED_ID={fixed_id}, gene={gene_name}")

    # Flow / fixed resolution
    Hf, Wf = get_flow_resolution(patient_id, fixed_id)
    print(f"[INFO] Flow resolution: H_f={Hf}, W_f={Wf}")

    warped_maps = {}

    for sid in SAMPLE_IDS:
        print(f"[INFO]  - Warping sample{sid} -> axis_{fixed_id} ...")
        # Load gene map & mask from its own axis_{sid} result
        gene_map = load_gene_map_np(
            patient_id=patient_id, axis_id=sid, sample_id=sid, gene_name=gene_name
        )
        mask = load_mask_np(
            patient_id=patient_id, axis_id=sid, sample_id=sid
        )

        if mask.shape != gene_map.shape:
            raise ValueError(
                f"Mask shape {mask.shape} != gene_map shape {gene_map.shape} "
                f"for patient {patient_id}, axis_{sid}, sample{sid}"
            )

        # Set background to NaN so it does not contribute to regression
        gene_map = gene_map.copy()
        gene_map[mask == 0] = np.nan

        if sid == fixed_id:
            # identity warp: zero flow
            flow_np = np.zeros((2, Hf, Wf), dtype=np.float32)
        else:
            try:
                flow_np = load_flow_np(patient_id, fixed_id, sid)
            except FileNotFoundError as e:
                print(f"[WARN]    Flow missing for {fixed_id}_{sid}: {e}")
                print(f"[WARN]    Using identity warp for sample{sid}.")
                flow_np = np.zeros((2, Hf, Wf), dtype=np.float32)

        warped_np = warp_gene_with_axis_flow(gene_map, flow_np)
        warped_maps[sid] = warped_np

    g0 = warped_maps[0]
    g1 = warped_maps[1]
    g2 = warped_maps[2]

    slope_map = compute_slope_map_nan(g0, g1, g2)

    # ------------------- Visualization -------------------

    # robust slope color limits ignoring NaNs
    valid_slope = slope_map[~np.isnan(slope_map)]
    if valid_slope.size > 0:
        vmin = np.percentile(valid_slope, 5)
        vmax = np.percentile(valid_slope, 95)
    else:
        vmin, vmax = -1.0, 1.0

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    im0 = axes[0].imshow(g0, cmap="viridis")
    axes[0].set_title(f"Patient {patient_id} – {gene_name}\nSample0 → axis_{fixed_id}")
    axes[0].axis("off")

    im1 = axes[1].imshow(g1, cmap="viridis")
    axes[1].set_title(f"Sample1 → axis_{fixed_id}")
    axes[1].axis("off")

    im2 = axes[2].imshow(g2, cmap="viridis")
    axes[2].set_title(f"Sample2 → axis_{fixed_id}")
    axes[2].axis("off")

    im3 = axes[3].imshow(slope_map, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[3].set_title("Slope across slices (0→2)")
    axes[3].axis("off")
    im3 = axes[3].imshow(slope_map, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[3].set_title("Slope across slices (0→2)")
    axes[3].axis("off")

    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im3, cax=cax)
    cbar.set_label("Regression slope (Δ expression per slice)")

    # fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    # im3 = axes.imshow(slope_map, cmap="coolwarm", vmin=vmin, vmax=vmax)
    # axes.set_title("Slope across slices (0→2)")
    # axes.axis("off")
    #
    # divider = make_axes_locatable(axes)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # cbar = fig.colorbar(im3, cax=cax)
    # cbar.set_label("Regression slope (Δ expression per slice)")

    plt.tight_layout()
    plt.show()

    return warped_maps, slope_map


def main():
    for pid in PATIENT_IDS:
        try:
            warped_maps, slope_map = analyze_patient_pull(pid, FIXED_ID, GENE_NAME)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping patient {pid}: {e}")


if __name__ == "__main__":
    main()
