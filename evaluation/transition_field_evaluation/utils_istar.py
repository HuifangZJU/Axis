import torch
import torch.nn.functional as F

import pickle
import numpy as np
from pathlib import Path
from PIL import Image


def load_istar_gene_map(patient_id, axis_id, sample_id, gene_name, istar_root):
    pkl_path = (
        Path(istar_root)
        / f"group_{patient_id}"
        / f"axis_{axis_id}"
        / f"sample{sample_id}"
        / "cnts-super"
        / f"{gene_name}.pickle"
    )

    if not pkl_path.is_file():
        raise FileNotFoundError(f"Gene map missing: {pkl_path}")

    with open(pkl_path, "rb") as f:
        arr = pickle.load(f)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Gene map must be 2D, got {arr.shape}")

    return arr


def load_istar_mask(patient_id, axis_id, sample_id, istar_root):
    mask_path = (
        Path(istar_root)
        / f"group_{patient_id}"
        / f"axis_{axis_id}"
        / f"sample{sample_id}"
        / "mask-small.png"
    )

    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file missing: {mask_path}")

    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]

    return (mask > 0).astype(np.uint8)


def load_istar_gene_slice(
    patient_id,
    sample_id,
    gene,
    istar_root,
    axis_id=None,
    use_log1p=True,
):
    """
    Load one iSTAR slice and convert valid pixels to coordinate/value form.

    Returns
    -------
    payload : dict
        {
            "coords": (N,2) float32, pixel-space xy coordinates,
            "values": (N,) float32,
            "gene_img": (H,W) float32,
            "mask": (H,W) uint8,
            "shape": (H,W),
            "sample_id": int,
        }
    """
    if axis_id is None:
        axis_id = sample_id

    gene_img = load_istar_gene_map(
        patient_id=patient_id,
        axis_id=axis_id,
        sample_id=sample_id,
        gene_name=gene,
        istar_root=istar_root,
    )
    mask = load_istar_mask(
        patient_id=patient_id,
        axis_id=axis_id,
        sample_id=sample_id,
        istar_root=istar_root,
    )

    if gene_img.shape != mask.shape:
        raise ValueError(f"Gene map and mask shape mismatch: {gene_img.shape} vs {mask.shape}")

    gene_img = gene_img.astype(np.float32)
    gene_img[mask == 0] = np.nan

    if use_log1p:
        gene_img = np.log1p(np.clip(gene_img, a_min=0, a_max=None))

    yy, xx = np.where(mask > 0)
    coords = np.column_stack([xx, yy]).astype(np.float32)
    values = gene_img[yy, xx].astype(np.float32)

    return {
        "coords": coords,
        "values": values,
        "gene_img": gene_img,
        "mask": mask,
        "shape": gene_img.shape,
        "sample_id": int(sample_id),
    }

def upsample_flow_to_target_shape(flow_np: np.ndarray, target_shape):
    """
    Upsample a low-resolution flow to target_shape.

    Parameters
    ----------
    flow_np : np.ndarray
        Shape (2, H, W) or (H, W, 2).
        Assumed channel order is (y, x) if shape is (2, H, W).
    target_shape : tuple
        (H_target, W_target)

    Returns
    -------
    flow_hr : np.ndarray
        Shape (2, H_target, W_target), float32
    """
    flow = np.asarray(flow_np, dtype=np.float32)

    if flow.ndim != 3:
        raise ValueError(f"flow must be 3D, got {flow.shape}")

    if flow.shape[0] == 2:
        pass
    elif flow.shape[-1] == 2:
        flow = np.moveaxis(flow, -1, 0)
    else:
        raise ValueError(f"flow must have a channel dimension of size 2, got {flow.shape}")

    _, H0, W0 = flow.shape
    H1, W1 = target_shape

    flow_t = torch.from_numpy(flow).unsqueeze(0)  # [1,2,H,W]
    flow_up = F.interpolate(
        flow_t,
        size=(H1, W1),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).cpu().numpy()

    # scale displacement magnitude
    scale_y = H1 / float(H0)
    scale_x = W1 / float(W0)

    flow_up[0] *= scale_y   # y displacement
    flow_up[1] *= scale_x   # x displacement

    return flow_up.astype(np.float32)


def save_istar_hr_flow(
    flow_np: np.ndarray,
    save_path,
    fixed_id: int,
    moving_id: int,
    fixed_shape,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        flow=flow_np.astype(np.float32),
        fixed_shape=np.asarray(fixed_shape, dtype=np.int32),
        fixed_id=np.int32(fixed_id),
        moving_id=np.int32(moving_id),
        channel_order="yx",
        coord_system="pixel_xy",
    )


def load_istar_hr_flow(path):
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    flow = data["flow"].astype(np.float32)
    meta = {
        "fixed_shape": tuple(data["fixed_shape"].tolist()),
        "fixed_id": int(data["fixed_id"]),
        "moving_id": int(data["moving_id"]),
        "channel_order": str(data["channel_order"]),
        "coord_system": str(data["coord_system"]),
    }
    return flow, meta