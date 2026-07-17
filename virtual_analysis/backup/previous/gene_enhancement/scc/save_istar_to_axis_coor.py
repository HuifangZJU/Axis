import time
from pathlib import Path
import glob
from axis.coordinate_utils import find_inverse_warp_coords
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

# ---------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------

AXIS_ROOT = Path("/media/huifang/data/registration/SCC/huifang/output")
ISTAR_ROOT = Path("/media/huifang/data/registration/SCC/istar/single_input")

PATIENT_IDS = [2, 5, 9, 10]
SAMPLE_IDS = [0, 1, 2]  # {0,1,2} for layer indices



# ---------------------------------------------------------------------
# Flow helpers
# ---------------------------------------------------------------------

def load_axis_flow(patient_id: int, fixed_id: int, moving_id: int):
    """
    Load AXIS flow for (fixed, moving) sample pair.

    Canonical shape: (2, H, W).

    If the file is (H, W, 2), convert to (2, H, W).
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

    # Normalize to (2, H, W)
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
    Upsample flow from (2, Hf, Wf) to (2, Hg, Wg) by interpolation only.

    flow:         np.ndarray, shape (2, Hf, Wf)
    target_shape: (Hg, Wg)
    """
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"upsample_flow expects (2, H, W), got {flow.shape}")

    _, Hf, Wf = flow.shape
    Hg, Wg = target_shape

    sy = Hg / Hf
    sx = Wg / Wf

    # zoom factors: (channels, H, W) -> (1, sy, sx)
    flow_up = zoom(flow, (1.0, sy, sx), order=1)
    return flow_up.astype(np.float32)  # (2, Hg, Wg)


# ---------------------------------------------------------------------
# Main coordinate warping
# ---------------------------------------------------------------------

def compute_and_save_axis_coords_for_sample(patient_id: int, sid1: int, sid2: int):
    sample_root = (
        ISTAR_ROOT
        / f"group_{patient_id}"
        / f"axis_{sid1}"
        / f"sample{sid2}"
    )
    if not sample_root.is_dir():
        return  # silently skip; not all combinations exist

    out_path = sample_root / "axis_coordinates.npy"
    if out_path.is_file():
        print(f"[SKIP] {out_path} already exists.")
        return

    mask_path = sample_root / "mask-small.png"
    if not mask_path.is_file():
        print(f"[WARN] mask-small.png missing: {mask_path}, skip.")
        return

    # Get grid shape from mask
    mask_img = Image.open(mask_path)
    mask_arr = np.array(mask_img)
    H, W = mask_arr.shape[:2]


    # Build uv_coords in moving frame: [x,y]
    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    uv_coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)  # (N,2)

    if sid1 == sid2:
        # Identity mapping: axis coords = grid coords
        axis_coords = uv_coords
        print(
            f"[INFO] Patient {patient_id}, axis_{sid1}/sample{sid2}: "
            f"identity coords (H={H}, W={W})"
        )

    else:

        # Load flow (fixed=sid1, moving=sid2)
        try:
            flow = load_axis_flow(patient_id, sid1, sid2)
        except FileNotFoundError as e:
            print(f"[WARN] {e}, skip this pair.")
            return

        # Upsample flow to mask resolution
        flow_up = upsample_flow(flow, (H, W))  # (2, H, W)
        t0 = time.time()
        # Warp coordinates into fixed (axis sid1) frame
        axis_coords = find_inverse_warp_coords(
            uv_coords, flow_up
        )
        t1 = time.time()
        axis_coords = np.asarray(axis_coords, dtype=np.float32)
        print(
            f"[INFO] Patient {patient_id}, axis_{sid1}/sample{sid2}: "
            f"warped coords computed (H={H}, W={W}) in {t1 - t0:.3f} s"
        )

    # Save axis coordinates
    np.save(out_path, axis_coords)
    print(f"[SAVE] {out_path}")


def main():
    for pid in PATIENT_IDS:
        for sid1 in SAMPLE_IDS:
            for sid2 in SAMPLE_IDS:
                compute_and_save_axis_coords_for_sample(pid, sid1, sid2)


if __name__ == "__main__":
    main()
