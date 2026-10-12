"""Applying transition fields to images, pseudo-images and point sets.

Conventions
-----------
* A transition field ``flow`` for the pair ``(fixed, moving)`` is an array of
  shape ``(2, H, W)`` holding ``(dy, dx)`` displacements in pixels on the fixed
  image grid. It warps the moving image by pull-back sampling,
  ``warped(p) = moving(p + flow(p))``.
* Point coordinates are ``(x, y)`` pixel positions, shape ``(N, 2)``.
* Transporting moving-sample points into the fixed frame therefore requires the
  inverse map, which [`find_inverse_warp_coords`][axis.registration.warp.find_inverse_warp_coords] approximates by a
  nearest-neighbour search over the forward-mapped pixel grid.
* Scaling a field by ``alpha`` (Methods, Eq. (25)) gives the partial transport
  used for intermediate-state generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "find_inverse_warp_coords",
    "find_inverse_warp_coords_local",
    "transport_points",
    "ensure_flow_2hw",
    "upsample_flow",
    "warp_image",
    "warp_highres_image_with_lowres_flow_cpu",
    "unify_group_to_reference",
    "CoordinateWarper",
    "CoordinateAffineWarper",
]

ArrayLike = Union[np.ndarray, torch.Tensor]


def ensure_flow_2hw(flow: ArrayLike) -> ArrayLike:
    """Return ``flow`` as ``(2, H, W)`` = ``(dy, dx)``.

    Accepts ``(2, H, W)``, ``(H, W, 2)``, ``(1, 2, H, W)``, ``(3, H, W)``,
    ``(1, 3, 1, H, W)`` (raw network output) and similar layouts.
    """
    is_np = isinstance(flow, np.ndarray)
    t = torch.from_numpy(np.asarray(flow)) if is_np else flow
    while t.ndim > 3 and t.shape[0] == 1:
        t = t[0]
    if t.ndim == 4 and t.shape[1] == 1:  # (C, 1, H, W)
        t = t[:, 0]
    if t.ndim != 3:
        raise ValueError(f"Cannot interpret flow of shape {tuple(flow.shape)}")
    if t.shape[0] == 3:
        t = t[1:3]
    elif t.shape[0] != 2 and t.shape[-1] == 2:
        t = t.permute(2, 0, 1)
    if t.shape[0] != 2:
        raise ValueError(f"Flow must have a channel dimension of size 2, got {tuple(flow.shape)}")
    t = t.contiguous()
    return t.numpy() if is_np else t


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_inverse_warp_coords(
    uv_coords: ArrayLike,
    flow: ArrayLike,
    batch_size: int = 1024,
    device: Optional[Union[str, torch.device]] = None,
) -> ArrayLike:
    """Map moving-frame points into the fixed frame of ``flow``.

    For each query point ``u`` the fixed-grid pixel ``p`` whose forward-mapped
    position ``p + flow(p)`` is closest to ``u`` is returned.

    Parameters
    ----------
    uv_coords
        ``(N, 2)`` ``(x, y)`` coordinates in the moving image.
    flow
        Transition field (see [`ensure_flow_2hw`][axis.registration.warp.ensure_flow_2hw] for accepted layouts).
    batch_size
        Number of query points per distance computation.
    device
        Torch device. Defaults to CUDA when available, otherwise CPU.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Coordinates in the fixed frame, with the same container type as
        ``uv_coords`` (NumPy array, or tensor on the input device).
    """
    uv_was_numpy = isinstance(uv_coords, np.ndarray)
    uv = torch.from_numpy(np.ascontiguousarray(uv_coords)) if uv_was_numpy else uv_coords
    orig_device = uv.device
    flow_t = torch.from_numpy(np.ascontiguousarray(flow)) if isinstance(flow, np.ndarray) else flow
    flow_t = ensure_flow_2hw(flow_t)

    device = torch.device(device) if device is not None else _default_device()
    flow_t = flow_t.to(device=device, dtype=torch.float32)
    uv = uv.to(device=device, dtype=torch.float32)

    dy, dx = flow_t[0], flow_t[1]
    H, W = dx.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    base_grid = torch.stack([x, y], dim=-1)
    warped_flat = (base_grid + torch.stack([dx, dy], dim=-1)).reshape(-1, 2)
    base_flat = base_grid.reshape(-1, 2)

    if device.type == "cpu":
        # A KD-tree gives the same nearest neighbour as the brute-force GPU
        # search at a fraction of the memory cost on CPU.
        from scipy.spatial import cKDTree

        tree = cKDTree(warped_flat.numpy())
        _, idx = tree.query(uv.numpy(), k=1)
        result = base_flat[torch.from_numpy(np.asarray(idx, dtype=np.int64)).reshape(-1)]
    else:
        nn_indices = []
        for start in range(0, uv.shape[0], batch_size):
            dists = torch.cdist(uv[start : start + batch_size].unsqueeze(0), warped_flat.unsqueeze(0))
            nn_indices.append(dists.argmin(dim=-1).squeeze(0))
            del dists
        result = base_flat[torch.cat(nn_indices, dim=0)] if nn_indices else torch.empty((0, 2), device=device)

    if uv_was_numpy:
        return result.cpu().numpy()
    return result.to(orig_device)


def find_inverse_warp_coords_local(uv_coords, flow, window: int = 15, tau: float = 0.1, eps: float = 1e-8):
    """Soft (sub-pixel) inverse warp restricted to a local window.

    A differentiable alternative to [`find_inverse_warp_coords`][axis.registration.warp.find_inverse_warp_coords]: the
    fixed-grid positions within ``window`` pixels of each query are averaged
    with soft-min weights ``softmax(-d / tau)``.
    """
    flow = ensure_flow_2hw(flow)
    device = flow.device
    H, W = flow.shape[1:]
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    base_grid = torch.stack([x, y], dim=-1).float()
    warped_grid = base_grid + flow.permute(1, 2, 0)[..., [1, 0]]

    warped_grid_t = warped_grid.permute(2, 0, 1).unsqueeze(0)
    k = 2 * window + 1
    unfolded_warped = F.unfold(warped_grid_t, kernel_size=k, padding=window)
    unfolded_base = F.unfold(base_grid.permute(2, 0, 1).unsqueeze(0), kernel_size=k, padding=window)

    uv_int = uv_coords.round().long()
    uv_int[:, 0] = uv_int[:, 0].clamp(0, W - 1)
    uv_int[:, 1] = uv_int[:, 1].clamp(0, H - 1)
    idxs = uv_int[:, 1] * W + uv_int[:, 0]
    warped_patches = unfolded_warped[0, :, idxs].reshape(2, k * k, -1).permute(2, 1, 0)
    base_patches = unfolded_base[0, :, idxs].reshape(2, k * k, -1).permute(2, 1, 0)

    dists = torch.norm(warped_patches - uv_coords[:, None, :].float(), dim=-1)
    weights = torch.softmax(-dists / (tau + eps), dim=-1)
    return torch.sum(weights[:, :, None] * base_patches, dim=1)


def transport_points(coords: np.ndarray, flow: ArrayLike, alpha: float = 1.0, **kwargs) -> np.ndarray:
    """Transport points with a (possibly scaled) transition field.

    ``alpha = 1`` maps moving-sample points into the fixed frame;
    ``0 < alpha < 1`` gives the partial transport ``D_{a->q} = alpha D_{a->b}``
    of Methods Eqs. (25)-(26).
    """
    flow_np = np.asarray(ensure_flow_2hw(flow if isinstance(flow, np.ndarray) else flow.detach().cpu().numpy()))
    scaled = (float(alpha) * flow_np).astype(np.float32)
    return np.asarray(find_inverse_warp_coords(np.asarray(coords, dtype=np.float32), scaled, **kwargs), dtype=np.float32)


def upsample_flow(flow: ArrayLike, size: Tuple[int, int]) -> ArrayLike:
    """Resize a ``(2, H, W)`` field to ``size`` and rescale its displacements.

    Used to transfer a field learned at the native resolution to a
    higher-resolution representation (Methods, "Resolution transferability").
    """
    is_np = isinstance(flow, np.ndarray)
    f = torch.from_numpy(np.asarray(ensure_flow_2hw(flow), dtype=np.float32)) if is_np else ensure_flow_2hw(flow).float()
    h, w = f.shape[-2:]
    H, W = size
    up = F.interpolate(f.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
    up[0] *= (H - 1) / max(h - 1, 1)
    up[1] *= (W - 1) / max(w - 1, 1)
    return up.numpy() if is_np else up


def warp_image(image: np.ndarray, flow: ArrayLike, mode: str = "bilinear") -> np.ndarray:
    """Warp an ``(H, W)`` or ``(H, W, C)`` image with a ``(2, H', W')`` field.

    The field is resized to the image size first, so a field learned on a
    512-pixel canvas can warp the corresponding high-resolution image.
    """
    img = np.asarray(image)
    squeeze = img.ndim == 2
    if squeeze:
        img = img[..., None]
    H, W = img.shape[:2]
    f = ensure_flow_2hw(flow)
    f = f if isinstance(f, np.ndarray) else f.detach().cpu().numpy()
    if f.shape[1:] != (H, W):
        f = upsample_flow(f, (H, W))
    src = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    ft = torch.from_numpy(np.asarray(f, dtype=np.float32))
    new_y = 2 * ((yy + ft[0]) / (H - 1) - 0.5)
    new_x = 2 * ((xx + ft[1]) / (W - 1) - 0.5)
    grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
    out = F.grid_sample(src, grid, align_corners=False, mode=mode)[0].permute(1, 2, 0).numpy()
    if np.issubdtype(img.dtype, np.integer):
        out = np.clip(np.round(out), np.iinfo(img.dtype).min, np.iinfo(img.dtype).max).astype(img.dtype)
    return out[..., 0] if squeeze else out


def warp_highres_image_with_lowres_flow_cpu(moving_hr, st: nn.Module, flow_lr: torch.Tensor):
    """Warp a high-resolution image with a low-resolution network field (CPU).

    Kept for the reproducibility scripts; [`warp_image`][axis.registration.warp.warp_image] is the general API.
    Returns the warped ``uint8`` RGB image and the upsampled ``(2, H, W)`` field.
    """
    import cv2

    if isinstance(moving_hr, np.ndarray):
        moving_hr = torch.from_numpy(moving_hr)
    if moving_hr.dim() == 3 and moving_hr.shape[-1] in (1, 3):
        moving_hr = moving_hr.permute(2, 0, 1)
    moving_hr = moving_hr.float().cpu()
    _, H_hr, W_hr = moving_hr.shape
    moving_hr = moving_hr.unsqueeze(0).unsqueeze(2)

    flow_lr = flow_lr.float()
    if flow_lr.dim() == 5:
        flow_lr = flow_lr.squeeze(2)
    _, _, H_lr, W_lr = flow_lr.shape
    flow_hr = F.interpolate(flow_lr, size=(H_hr, W_hr), mode="bilinear", align_corners=False)
    flow_hr[:, 1] *= (H_hr - 1) / max(H_lr - 1, 1)
    flow_hr[:, 2] *= (W_hr - 1) / max(W_lr - 1, 1)
    flow_hr = flow_hr.cpu().unsqueeze(2)

    with torch.no_grad():
        warped_hr = st.cpu()(moving_hr, flow_hr)
    warped_hr = warped_hr.squeeze().to(torch.uint8).permute(1, 2, 0).numpy()
    warped_hr = cv2.cvtColor(warped_hr, cv2.COLOR_BGR2RGB)
    return warped_hr, flow_hr[:, 1:3].squeeze()


# =============================================================================
# Group-level unification (reference frame)
# =============================================================================
def unify_group_to_reference(
    group_dir: Union[str, Path],
    sample_ids: Sequence[int],
    paths: Optional[Dict[int, List[int]]] = None,
    ref_coords: Optional[np.ndarray] = None,
    save_meta: bool = True,
    unified_subdir: str = "unified",
) -> Dict[int, np.ndarray]:
    """Express all samples of a group in the frame of one reference sample.

    Pairwise outputs ``{F}_{M}_flow.npy`` and ``{F}_{M}_warped_coor.npy``
    (moving ``M`` registered to fixed ``F``) are read from ``group_dir``. By
    default every sample is mapped directly to the smallest sample id; custom
    cascades such as ``{4: [4, 5, 0]}`` chain several fields.

    Results are written to ``group_dir/unified_subdir/coords_slice{sid}_to_ref.npy``.
    """
    group_dir = Path(group_dir)
    if not group_dir.is_dir():
        raise NotADirectoryError(f"Group dir not found: {group_dir}")

    flows: Dict[Tuple[int, int], np.ndarray] = {}
    warped: Dict[Tuple[int, int], np.ndarray] = {}
    for f_id in sample_ids:
        for m_id in sample_ids:
            if f_id == m_id:
                continue
            fp = group_dir / f"{f_id}_{m_id}_flow.npy"
            wp = group_dir / f"{f_id}_{m_id}_warped_coor.npy"
            if fp.is_file():
                flows[(f_id, m_id)] = np.load(fp)
            if wp.is_file():
                warped[(f_id, m_id)] = np.load(wp)
    slice_ids = sorted(sample_ids)
    if not flows and not warped:
        raise RuntimeError(f"No pairwise files found for sample_ids={list(sample_ids)} in {group_dir}")

    if paths is None:
        ref = slice_ids[0]
        paths = {sid: [sid] if sid == ref else [sid, ref] for sid in slice_ids}
    paths = {int(k): [int(x) for x in v] for k, v in paths.items()}
    refs = {p[-1] for p in paths.values()}
    if len(refs) != 1:
        raise ValueError(f"Paths must all end in the same reference slice; got {refs}")
    ref_id = refs.pop()

    out_dir = group_dir / unified_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    coords: Dict[int, np.ndarray] = {}
    if ref_coords is not None:
        coords[ref_id] = np.asarray(ref_coords, dtype=np.float32)
        np.save(out_dir / f"coords_slice{ref_id}_to_ref.npy", coords[ref_id])

    for sid, path in paths.items():
        if sid == ref_id or sid not in slice_ids:
            continue
        if path[0] != sid or path[-1] != ref_id or len(path) < 2:
            raise ValueError(f"Invalid path for slice {sid}: {path}")
        key = (path[1], path[0])
        if key not in warped:
            raise RuntimeError(f"Missing warped coordinates for fixed={path[1]}, moving={path[0]}")
        current = warped[key].astype(np.float32)
        for j in range(1, len(path) - 1):
            src, dst = path[j], path[j + 1]
            if (dst, src) not in flows:
                raise RuntimeError(f"Missing flow {dst}_{src}_flow.npy")
            current = np.asarray(find_inverse_warp_coords(current, flows[(dst, src)]), dtype=np.float32)
        coords[sid] = current
        np.save(out_dir / f"coords_slice{sid}_to_ref.npy", current)

    if save_meta:
        meta = {
            "reference_slice": ref_id,
            "paths": {int(k): v for k, v in paths.items()},
            "sample_ids": slice_ids,
            "has_ref_coords": ref_coords is not None,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return coords


# =============================================================================
# Differentiable point warpers (used by evaluation scripts)
# =============================================================================
class CoordinateWarper(nn.Module):
    """Sample a ``(B, 2, H, W)`` ``(dy, dx)`` field at points and subtract it."""

    def __init__(self, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, coords: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        H, W = flow.shape[2], flow.shape[3]
        x, y = coords[..., 0], coords[..., 1]
        grid = torch.stack([(x / (W - 1)) * 2 - 1, (y / (H - 1)) * 2 - 1], dim=-1)
        sampled = F.grid_sample(flow, grid.unsqueeze(2), mode=self.mode, align_corners=False)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        return torch.stack([x - sampled[..., 1], y - sampled[..., 0]], dim=-1)


class CoordinateAffineWarper(nn.Module):
    """Apply the inverse of the predicted affine transform to pixel coordinates."""

    def __init__(self, image_size: Tuple[int, int] = (1024, 1024)):
        super().__init__()
        self.image_size = image_size

    def forward(self, points_in: torch.Tensor, affine_para: torch.Tensor) -> torch.Tensor:
        device = points_in.device
        B, N, _ = points_in.shape
        H_in, W_in = self.image_size
        id_mat = torch.zeros((1, 3, 4), device=device)
        id_mat[0, 0, 0] = id_mat[0, 1, 1] = id_mat[0, 2, 2] = 1
        out2in = affine_para.reshape(1, 3, 4) + id_mat
        R, t = out2in[:, :, :3], out2in[:, :, 3]
        R_inv = torch.inverse(R)
        in2out = torch.cat([R_inv, (-torch.bmm(R_inv, t.unsqueeze(2)))], dim=2)

        x_n = 2.0 * points_in[..., 0] / (W_in - 1) - 1.0
        y_n = 2.0 * points_in[..., 1] / (H_in - 1) - 1.0
        hom = torch.stack([x_n, y_n, torch.zeros_like(x_n), torch.ones_like(x_n)], dim=2)
        out = torch.bmm(hom, in2out.transpose(1, 2))
        return torch.stack([(out[..., 0] + 1.0) * (W_in - 1) / 2.0, (out[..., 1] + 1.0) * (H_in - 1) / 2.0], dim=-1)
