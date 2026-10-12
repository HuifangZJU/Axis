"""Predicting and storing pairwise transition fields."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..settings import settings
from .model import load_state_dict_strict
from .warp import find_inverse_warp_coords, warp_highres_image_with_lowres_flow_cpu

__all__ = [
    "predict_field",
    "tensor2img",
    "generate_pairwise_outputs",
    "save_results_per_pair",
]


def tensor2img(tensor) -> np.ndarray:
    """``[1, C, 1, H, W]`` tensor in [-1, 1] -> ``(H, W[, C])`` array clipped to [0, 1]."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    tensor = np.squeeze(tensor)
    if tensor.ndim == 3:
        img = tensor[0] if tensor.shape[0] == 1 else np.transpose(tensor, (1, 2, 0))
    elif tensor.ndim == 2:
        img = tensor
    else:
        raise ValueError(f"Unexpected tensor shape after squeeze: {tensor.shape}")
    return np.clip(img, 0, 1)


@torch.no_grad()
def predict_field(
    model: torch.nn.Module,
    fixed_image: torch.Tensor,
    moving_image: torch.Tensor,
    fixed_gene: Optional[torch.Tensor] = None,
    moving_gene: Optional[torch.Tensor] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Predict the ``(2, H, W)`` ``(dy, dx)`` field registering ``moving`` onto ``fixed``.

    Inputs are ``[C, H, W]`` (or batched ``[1, C, H, W]``) tensors as produced
    by :class:`axis.registration.PairDataset`.
    """
    device = settings.resolve_device(device)
    model.to(device).eval()

    def prep(t):
        t = t if t.ndim == 4 else t.unsqueeze(0)
        return t.to(device).float().unsqueeze(2)

    if fixed_gene is None:
        flow = model(prep(fixed_image), prep(moving_image))[1]
    else:
        flow = model(prep(fixed_image), prep(fixed_gene), prep(moving_image), prep(moving_gene))[2]
    return flow[0, 1:3, 0].detach().cpu().numpy().astype(np.float32)


def _pair_key(pair_obj) -> Tuple:
    if isinstance(pair_obj, torch.Tensor):
        return tuple(int(x) for x in pair_obj.view(-1).tolist())
    if isinstance(pair_obj, (list, tuple)):
        elems = []
        for x in pair_obj:
            if isinstance(x, (list, tuple)) and len(x) == 1:
                x = x[0]
            if isinstance(x, torch.Tensor):
                elems.extend(int(v) for v in x.view(-1).tolist())
            else:
                elems.append(x)
        return tuple(elems)
    return (pair_obj,)


@torch.no_grad()
def generate_pairwise_outputs(
    ckpt_path: Optional[Union[str, os.PathLike]],
    test_dataset: Dataset,
    model: torch.nn.Module,
    run_hires: bool = False,
    device: Optional[str] = None,
    num_workers: int = 0,
) -> Dict[Tuple, Dict[str, np.ndarray]]:
    """Run a trained model over every pair of ``test_dataset`` (``mode="val"``).

    Returns ``{(fixed, moving): {...}}`` with the warped image, the field, and the
    moving spots transported into the fixed frame (``warped_coor``).
    """
    device = settings.resolve_device(device)
    if ckpt_path is not None:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        load_state_dict_strict(model, state_dict)
    model.to(device).eval()
    spatial_transformer = model.SpatialTransformer

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    results: Dict[Tuple, Dict[str, np.ndarray]] = {}
    for inputs, val_set in loader:
        pair_key = _pair_key(inputs["data_pair"])
        fixed_image = inputs["fixed_image"].to(device).float().unsqueeze(2)
        fixed_gene = inputs["fixed_gene"].to(device).float().unsqueeze(2)
        moving_image = inputs["moving_image"].to(device).float().unsqueeze(2)
        moving_gene = inputs["moving_gene"].to(device).float().unsqueeze(2)

        warped_image, _, flow_3d, _, _, _, _ = model(fixed_image, fixed_gene, moving_image, moving_gene)
        flow = flow_3d[:, 1:3].squeeze()
        moving_coor = val_set["moving_coor"].squeeze(0)
        warped_coor = find_inverse_warp_coords(moving_coor, flow)

        results[pair_key] = {
            "fixed_image": tensor2img(fixed_image),
            "moving_image": tensor2img(moving_image),
            "warped_image": tensor2img(warped_image),
            "fixed_coor": val_set["fixed_coor"].squeeze(0).cpu().numpy(),
            "moving_coor": moving_coor.cpu().numpy(),
            "warped_coor": warped_coor.detach().cpu().numpy(),
            "fixed_label": val_set["fixed_label"].squeeze().cpu().numpy(),
            "moving_label": val_set["moving_label"].squeeze().cpu().numpy(),
            "flow": flow.detach().cpu().numpy(),
        }
        if run_hires:
            import cv2

            hires = cv2.imread(inputs["moving_hires_image"][0], cv2.IMREAD_COLOR)
            warped_hr, flow_hr = warp_highres_image_with_lowres_flow_cpu(hires, spatial_transformer, flow_3d)
            results[pair_key].update({"warped_hires": warped_hr, "flow_hires": flow_hr})
    return results


def save_results_per_pair(
    root_dir: Union[str, os.PathLike],
    group_id,
    pair_key: Tuple,
    result_dict: Dict[str, np.ndarray],
    subdir: str = "pairwise_output",
) -> Path:
    """Write the outputs of one pair to ``{root}/{subdir}/group_{id}/{F}_{M}_*``.

    Files: ``_warped_img.png``, ``_warped_coor.npy``, ``_flow.npy`` and, when
    available, ``_hires_warped_img.png`` / ``_hires_flow.npy``. The default
    ``subdir`` matches the location read by the generation and matching steps.
    """
    import cv2

    folder = f"group_{group_id}" if isinstance(group_id, int) else str(group_id)
    out_dir = Path(root_dir) / subdir / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{pair_key[0]}_{pair_key[1]}"

    warped = result_dict.get("warped_image")
    if warped is not None:
        img = (np.asarray(warped) * 255).astype(np.uint8)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{prefix}_warped_img.png"), img)
    if result_dict.get("warped_coor") is not None:
        np.save(out_dir / f"{prefix}_warped_coor.npy", result_dict["warped_coor"])
    if result_dict.get("flow") is not None:
        np.save(out_dir / f"{prefix}_flow.npy", result_dict["flow"])

    hires = result_dict.get("warped_hires")
    if hires is not None:
        hires = hires.detach().cpu().numpy() if isinstance(hires, torch.Tensor) else np.asarray(hires)
        if hires.shape[0] in (1, 3):
            hires = np.transpose(hires, (1, 2, 0))
        cv2.imwrite(
            str(out_dir / f"{prefix}_hires_warped_img.png"),
            cv2.cvtColor(np.clip(hires, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
    hires_flow = result_dict.get("flow_hires")
    if hires_flow is not None:
        hires_flow = hires_flow.detach().cpu().numpy() if isinstance(hires_flow, torch.Tensor) else hires_flow
        np.save(out_dir / f"{prefix}_hires_flow.npy", hires_flow)
    return out_dir
