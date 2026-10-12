"""AXIS transition-field networks.

[`TransitionFieldNet`][axis.registration.model.TransitionFieldNet] is the multimodal coarse-to-fine registration
network described in Methods ("AXIS: Tissue transition field learning"). It
takes the morphology images and molecular pseudo-images of a *fixed* (reference)
and a *moving* (source) sample and predicts a dense, non-rigid displacement
field ``D_{moving -> fixed}`` together with a global affine transform.

[`ImageTransitionNet`][axis.registration.model.ImageTransitionNet] is the morphology-only variant used for
pre-training and for the image-only ablations.

Both classes keep the module names of the original implementation
(``Encoder``, ``Decoder``, ``SpatialTransformer``, ``AffineTransformer``), so
checkpoints trained with the original code load with ``strict=True``. The
legacy class names ``NICE_Trans`` and ``NICE_Trans_img`` are kept as aliases.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from .blocks import AffineTransformer_block, Conv_encoder, SpatialTransformer_block, Trans_decoder

__all__ = [
    "TransitionFieldNet",
    "ImageTransitionNet",
    "NICE_Trans",
    "NICE_Trans_img",
    "build_model",
    "load_model",
    "load_state_dict_flexible",
    "load_state_dict_strict",
    "strip_derived_buffers",
]


def _check_spatial_size(image: torch.Tensor) -> None:
    h, w = image.shape[-2:]
    if h % 16 or w % 16:
        raise ValueError(
            f"Image size {h}x{w} is not divisible by 16; the encoder downsamples four times. "
            "Resize the canvas (e.g. 256, 512 or 1024 pixels)."
        )


class TransitionFieldNet(nn.Module):
    """Multimodal transition-field network.

    Parameters
    ----------
    in_channels
        Number of morphology-image channels (3 for H&E, 1 for DAPI/grayscale).
    in_gene_channels
        Number of pseudo-image channels used by the network (PCA components).
    enc_channels
        Base encoder width ``c_e``.
    dec_channels
        Base decoder width.
    fusion_module
        ``"transformer"`` (default for Visium) or ``"linear"`` (convolution-based
        fusion, default for Xenium / PhenoCycler).
    use_checkpoint
        Use gradient checkpointing to reduce memory during training.

    Notes
    -----
    Inputs are ``[B, C, 1, H, W]`` tensors (see ``axis.registration.blocks``).
    The pseudo-images must be at 1/8 of the image resolution and may carry an
    extra trailing mask channel, which is ignored by the encoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        in_gene_channels: int = 10,
        enc_channels: int = 8,
        dec_channels: int = 16,
        fusion_module: str = "transformer",
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.fusion_ = fusion_module
        self.gene_channels = in_gene_channels
        self.Encoder = Conv_encoder(
            in_channels=in_channels,
            channel_num=enc_channels,
            in_gene_channels=in_gene_channels,
            fusion=fusion_module,
            use_checkpoint=use_checkpoint,
        )
        self.Decoder = Trans_decoder(in_channels=enc_channels, channel_num=dec_channels, use_checkpoint=use_checkpoint)
        self.SpatialTransformer = SpatialTransformer_block(mode="bilinear")
        self.AffineTransformer = AffineTransformer_block(mode="bilinear")

    @property
    def config(self) -> Dict[str, object]:
        """Constructor arguments (stored alongside checkpoints)."""
        enc = self.Encoder
        return {
            "kind": "multimodal",
            "in_channels": enc.conv_1.Conv_1.in_channels,
            "in_gene_channels": self.gene_channels,
            "enc_channels": enc.conv_1.Conv_1.out_channels,
            "dec_channels": self.Decoder.conv_1.Conv_1.out_channels,
            "fusion_module": self.fusion_,
        }

    def forward(self, fixed_image, fixed_gene, moving_image, moving_gene):
        """Register ``moving`` onto ``fixed``.

        Returns
        -------
        tuple of torch.Tensor
            ``(warped_image, warped_gene, flow, flow_gene, affined_image,
            affined_gene, affine_para)``. ``flow`` is the full-resolution field
            ``[B, 3, 1, H, W]`` (channels ``d, y, x``); ``flow_gene`` is the same
            field resized to the pseudo-image grid, with displacements rescaled
            to its pixels (the resizing Γ in Methods, "Image and molecular
            warping").
        """
        _check_spatial_size(fixed_image)
        x_fix = self.Encoder(fixed_image, fixed_gene[:, : self.gene_channels])
        x_mov = self.Encoder(moving_image, moving_gene[:, : self.gene_channels])
        flow, affine_para = self.Decoder(x_fix, x_mov)

        warped_image = self.SpatialTransformer(moving_image, flow)
        affined_image = self.AffineTransformer(moving_image, affine_para)

        # Resize the field to the pseudo-image resolution and rescale its
        # magnitude (pixels at 1/8 resolution).
        scale = moving_gene.shape[-1] / moving_image.shape[-1]
        flow_gene = nnf.interpolate(flow, size=moving_gene.shape[-3:], mode="trilinear", align_corners=False) * scale
        affined_gene = self.AffineTransformer(moving_gene, affine_para)
        warped_gene = self.SpatialTransformer(moving_gene, flow_gene)
        return warped_image, warped_gene, flow, flow_gene, affined_image, affined_gene, affine_para

    @torch.no_grad()
    def predict_field(self, fixed_image, fixed_gene, moving_image, moving_gene) -> torch.Tensor:
        """Return the 2-channel ``(dy, dx)`` field ``[H, W]`` for a single pair."""
        flow = self.forward(fixed_image, fixed_gene, moving_image, moving_gene)[2]
        return flow[0, 1:3, 0]


class ImageTransitionNet(nn.Module):
    """Morphology-only transition-field network (pre-training / ablation)."""

    def __init__(self, in_channels: int = 3, enc_channels: int = 8, dec_channels: int = 16, use_checkpoint: bool = True):
        super().__init__()
        self.Encoder = Conv_encoder(in_channels=in_channels, channel_num=enc_channels, fusion=None, use_checkpoint=use_checkpoint)
        self.Decoder = Trans_decoder(in_channels=enc_channels, channel_num=dec_channels, use_checkpoint=use_checkpoint)
        self.SpatialTransformer = SpatialTransformer_block(mode="bilinear")
        self.AffineTransformer = AffineTransformer_block(mode="bilinear")

    @property
    def config(self) -> Dict[str, object]:
        enc = self.Encoder
        return {
            "kind": "image",
            "in_channels": enc.conv_1.Conv_1.in_channels,
            "enc_channels": enc.conv_1.Conv_1.out_channels,
            "dec_channels": self.Decoder.conv_1.Conv_1.out_channels,
        }

    def forward(self, fixed_image, moving_image):
        """Returns ``warped_image, flow, affined_image, affine_para``."""
        _check_spatial_size(fixed_image)
        x_fix = self.Encoder(fixed_image)
        x_mov = self.Encoder(moving_image)
        flow, affine_para = self.Decoder(x_fix, x_mov)
        warped_image = self.SpatialTransformer(moving_image, flow)
        affined_image = self.AffineTransformer(moving_image, affine_para)
        return warped_image, flow, affined_image, affine_para

    @torch.no_grad()
    def predict_field(self, fixed_image, moving_image) -> torch.Tensor:
        flow = self.forward(fixed_image, moving_image)[1]
        return flow[0, 1:3, 0]


# Backwards-compatible names used throughout the original code base.
NICE_Trans = TransitionFieldNet
"""Original name of [`TransitionFieldNet`][axis.registration.model.TransitionFieldNet] (same class)."""
NICE_Trans_img = ImageTransitionNet
"""Original name of [`ImageTransitionNet`][axis.registration.model.ImageTransitionNet] (same class)."""


def build_model(kind: str = "multimodal", **kwargs) -> nn.Module:
    """Construct a network by kind (``"multimodal"`` or ``"image"``)."""
    if kind == "multimodal":
        return TransitionFieldNet(**kwargs)
    if kind == "image":
        return ImageTransitionNet(**kwargs)
    raise ValueError(f"Unknown model kind {kind!r}")


#: Buffers that are recomputed when a network is constructed. The Swin
#: ``relative_position_index`` tables (int64, 4096 x 4096 per block) make up
#: ~95% of a full ``state_dict``; ``axis.registration.save_checkpoint``
#: leaves them out.
DERIVED_BUFFERS = ("relative_position_index",)


def _is_derived(key: str) -> bool:
    return key.rsplit(".", 1)[-1] in DERIVED_BUFFERS


def strip_derived_buffers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop the deterministic buffers listed in ``DERIVED_BUFFERS`` (the Swin ``relative_position_index`` tables)."""
    return {k: v for k, v in state_dict.items() if not _is_derived(k)}


def load_state_dict_strict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """``load_state_dict(strict=True)`` that accepts checkpoints without derived buffers.

    Checkpoints written by the original code (which include the buffers) and by
    [`axis.registration.save_checkpoint`][axis.registration.train.save_checkpoint]
    (which do not) both load.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not _is_derived(k)]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the network: {len(missing)} missing keys {missing[:3]}, "
            f"{len(unexpected)} unexpected keys {unexpected[:3]}."
        )


def load_state_dict_flexible(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> Tuple[list, list]:
    """Load the parameters whose names and shapes match.

    This reproduces the warm-start used in the original training scripts, where
    a multimodal network was initialised from an image-only checkpoint.

    Returns the lists of loaded and skipped keys.
    """
    own = model.state_dict()
    loaded, skipped = [], []
    for key, value in state_dict.items():
        if key in own and own[key].shape == value.shape:
            own[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    if strict and (skipped or len(loaded) != len(own)):
        raise RuntimeError(f"Checkpoint mismatch: {len(skipped)} unexpected keys, {len(own) - len(loaded)} missing.")
    model.load_state_dict(own)
    return loaded, skipped


def load_model(
    path: Union[str, os.PathLike],
    device: Optional[str] = None,
    **model_kwargs,
) -> nn.Module:
    """Load a checkpoint written by [`axis.registration.save_checkpoint`][axis.registration.train.save_checkpoint] or by the original code.

    [`axis.tl.fit_transition`][axis.tl.fit_transition] writes them to ``out_dir``
    during training.

    Checkpoints saved by AXIS ≥ 1.0 store ``{"config": ..., "state_dict": ...}``;
    plain ``state_dict`` files from the original scripts are also accepted, in
    which case ``model_kwargs`` (e.g. ``in_channels=3, fusion_module="transformer"``)
    must describe the architecture.
    """
    from ..settings import settings

    device = settings.resolve_device(device)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        config = dict(payload.get("config", {}))
        config.update(model_kwargs)
        state_dict = payload["state_dict"]
    else:
        config = {"kind": "multimodal", **model_kwargs}
        state_dict = payload
    kind = config.pop("kind", "multimodal")
    model = build_model(kind, **config)
    load_state_dict_strict(model, state_dict)
    return model.to(device).eval()
