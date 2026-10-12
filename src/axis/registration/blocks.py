"""Building blocks of the AXIS transition-field network.

The network operates on 2-D images that are carried through the layers as
5-D tensors ``[B, C, 1, H, W]`` (a singleton depth axis). This layout is
inherited from the 3-D NICE-Trans architecture the network is derived from and
is kept so that published checkpoints load without modification: **the module
and parameter names in this file define the checkpoint format and must not be
renamed.**

Flow convention
---------------
Displacement fields are stored with three channels ``(d, y, x)``; the depth
channel is ignored by the 2-D warping layers. A field ``D`` maps a *moving*
image onto the *fixed* frame by pull-back sampling,
``warped(p) = moving(p + D(p))``, with displacements in pixels.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal

__all__ = [
    "FusionBlock",
    "TransformerFusion",
    "Conv_encoder",
    "Trans_decoder",
    "AffineTransformer_block",
    "SpatialTransformer_block",
    "ResizeTransformer_block",
    "BoundaryExtractor",
    "Conv_block",
    "DeformHead_block",
    "AffineHead_block",
    "PatchExpanding_block",
    "SwinTrans_stage_block",
    "SwinTrans_Block",
    "MSA_block",
    "MLP_block",
]


def _checkpoint(function, *args):
    # ``use_reentrant=False`` is the recommended mode and silences the warning
    # emitted by recent PyTorch versions; results are identical.
    return checkpoint.checkpoint(function, *args, use_reentrant=False)


# =============================================================================
# Multimodal fusion (Methods, "Intermediate multimodal fusion")
# =============================================================================
class FusionBlock(nn.Module):
    """Convolution-based fusion, Eq. (6): ``W_f * [F^(4); P]``.

    Default for large inputs (Xenium, PhenoCycler).
    """

    def __init__(self, img_channels: int, gene_channels: int, fused_channels: int):
        super().__init__()
        self.fusion_conv = nn.Conv3d(
            in_channels=img_channels + gene_channels,
            out_channels=fused_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, img_feat: torch.Tensor, gene_feat: torch.Tensor) -> torch.Tensor:
        return self.fusion_conv(torch.cat([img_feat, gene_feat], dim=1))


class TransformerFusion(nn.Module):
    """Transformer-based fusion, Eqs. (3)-(5).

    Both modalities are projected to a shared embedding, flattened to token
    sequences, concatenated and passed through a transformer encoder layer; the
    two halves are averaged token-wise, reshaped and refined by a 1x1 conv.
    Default for smaller inputs (Visium).
    """

    def __init__(self, img_channels: int = 64, gene_channels: int = 10, fused_channels: int = 128, num_heads: int = 4):
        super().__init__()
        self.img_proj = nn.Conv2d(img_channels, fused_channels, kernel_size=1)
        self.gene_proj = nn.Conv2d(gene_channels, fused_channels, kernel_size=1)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=fused_channels,
            nhead=num_heads,
            dim_feedforward=fused_channels * 2,
            batch_first=True,
        )
        self.fuse_conv = nn.Conv2d(fused_channels, fused_channels, kernel_size=1)

    def forward(self, img_feat: torch.Tensor, gene_feat: torch.Tensor) -> torch.Tensor:
        b, _, _, h, w = img_feat.shape
        img_embed = self.img_proj(img_feat.squeeze(2))
        gene_embed = self.gene_proj(gene_feat.squeeze(2))

        img_seq = img_embed.flatten(2).permute(0, 2, 1)
        gene_seq = gene_embed.flatten(2).permute(0, 2, 1)
        fused_seq = self.transformer(torch.cat([img_seq, gene_seq], dim=1))

        fused_img_seq, fused_gene_seq = torch.chunk(fused_seq, 2, dim=1)
        fused_seq_avg = (fused_img_seq + fused_gene_seq) / 2
        fused_map = fused_seq_avg.permute(0, 2, 1).reshape(b, -1, h, w)
        return self.fuse_conv(fused_map).unsqueeze(2)


# =============================================================================
# Encoder / decoder (Methods, "Multiscale encoder" and "Coarse-to-fine decoder")
# =============================================================================
class Conv_encoder(nn.Module):
    """Five-level convolutional feature pyramid with optional molecular fusion.

    Channel widths are ``c_e, 2c_e, 4c_e, 8c_e, 16c_e``. When ``fusion`` is not
    ``None`` the molecular pseudo-image is fused into the level-4 features, whose
    spatial resolution (1/8 of the input) matches the pseudo-image.
    """

    def __init__(
        self,
        in_channels: int,
        channel_num: int,
        in_gene_channels: int = 0,
        fusion: Optional[str] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.conv_1 = Conv_block(in_channels, channel_num, use_checkpoint)
        self.conv_2 = Conv_block(channel_num, channel_num * 2, use_checkpoint)
        self.conv_3 = Conv_block(channel_num * 2, channel_num * 4, use_checkpoint)
        self.conv_4 = Conv_block(channel_num * 4, channel_num * 8, use_checkpoint)
        self.conv_5 = Conv_block(channel_num * 8, channel_num * 16, use_checkpoint)

        level4 = channel_num * 8
        if fusion is None:
            self.fusion = None
        elif fusion == "transformer":
            self.fusion = TransformerFusion(
                img_channels=level4, gene_channels=in_gene_channels, fused_channels=level4, num_heads=8
            )
        elif fusion in ("linear", "conv", "convolution"):
            self.fusion = FusionBlock(img_channels=level4, gene_channels=in_gene_channels, fused_channels=level4)
        else:
            raise ValueError(f"Unknown fusion module {fusion!r}; use 'transformer', 'linear' or None.")

        self.downsample = nn.AvgPool2d(2, stride=2)

    def _down(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x.squeeze(dim=2)).unsqueeze(dim=2)

    def forward(self, x_in: torch.Tensor, g_in: Optional[torch.Tensor] = None):
        x_1 = self.conv_1(x_in)
        x_2 = self.conv_2(self._down(x_1))
        x_3 = self.conv_3(self._down(x_2))
        x_4 = self.conv_4(self._down(x_3))
        if self.fusion is not None:
            if g_in is None:
                raise ValueError("This encoder fuses molecular features; a pseudo-image must be supplied.")
            x_4 = self.fusion(x_4, g_in)
        x_5 = self.conv_5(self._down(x_4))
        return [x_1, x_2, x_3, x_4, x_5]


class Trans_decoder(nn.Module):
    """Coarse-to-fine Swin-transformer decoder, Eqs. (8)-(17).

    A global affine head initialises the field at the coarsest scale; residual
    deformation heads refine it at levels 4..1.
    """

    def __init__(
        self, in_channels: int, channel_num: int, use_checkpoint: bool = False, window_size: Sequence[int] = (16, 16, 16)
    ):
        super().__init__()
        winsize = list(window_size)
        self.conv_1 = Conv_block(in_channels * 2 + channel_num, channel_num, use_checkpoint)
        self.trans_2 = SwinTrans_stage_block(channel_num * 2, 4, channel_num // 8, winsize, use_checkpoint=use_checkpoint)
        self.trans_3 = SwinTrans_stage_block(channel_num * 4, 4, channel_num // 4, winsize, use_checkpoint=use_checkpoint)
        self.trans_4 = SwinTrans_stage_block(channel_num * 8, 4, channel_num // 2, winsize, use_checkpoint=use_checkpoint)
        self.trans_5 = SwinTrans_stage_block(channel_num * 16, 4, channel_num, winsize, use_checkpoint=use_checkpoint)

        self.backdim_2 = nn.Conv3d(in_channels * 4 + channel_num * 2, channel_num * 2, kernel_size=1, stride=1, padding="same")
        self.backdim_3 = nn.Conv3d(in_channels * 8 + channel_num * 4, channel_num * 4, kernel_size=1, stride=1, padding="same")
        self.backdim_4 = nn.Conv3d(in_channels * 16 + channel_num * 8, channel_num * 8, kernel_size=1, stride=1, padding="same")
        self.backdim_5 = nn.Conv3d(in_channels * 32, channel_num * 16, kernel_size=1, stride=1, padding="same")

        self.upsample_1 = PatchExpanding_block(embed_dim=channel_num * 2)
        self.upsample_2 = PatchExpanding_block(embed_dim=channel_num * 4)
        self.upsample_3 = PatchExpanding_block(embed_dim=channel_num * 8)
        self.upsample_4 = PatchExpanding_block(embed_dim=channel_num * 16)

        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num * 2, use_checkpoint)
        self.reghead_3 = DeformHead_block(channel_num * 4, use_checkpoint)
        self.reghead_4 = DeformHead_block(channel_num * 8, use_checkpoint)
        self.reghead_5 = AffineHead_block(channel_num * 16)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode="trilinear")
        self.SpatialTransformer = SpatialTransformer_block(mode="bilinear")

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5 = x_fix
        x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5 = x_mov

        # Coarsest scale: global affine initialisation, Eqs. (8)-(11).
        x = self.backdim_5(torch.cat([x_fix_5, x_mov_5], dim=1))
        x_5 = self.trans_5(x)
        flow_5, affine_para = self.reghead_5(x_5)

        # Levels 4..2: upsample, warp moving features, refine, Eqs. (12)-(15).
        flow_5_up = self.ResizeTransformer(flow_5)
        x_mov_4 = self.SpatialTransformer(x_mov_4, flow_5_up)
        x = self.backdim_4(torch.cat([x_fix_4, self.upsample_4(x_5), x_mov_4], dim=1))
        x_4 = self.trans_4(x)
        flow_4 = self.reghead_4(x_4) + flow_5_up

        flow_4_up = self.ResizeTransformer(flow_4)
        x_mov_3 = self.SpatialTransformer(x_mov_3, flow_4_up)
        x = self.backdim_3(torch.cat([x_fix_3, self.upsample_3(x_4), x_mov_3], dim=1))
        x_3 = self.trans_3(x)
        flow_3 = self.reghead_3(x_3) + flow_4_up

        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2 = self.SpatialTransformer(x_mov_2, flow_3_up)
        x = self.backdim_2(torch.cat([x_fix_2, self.upsample_2(x_3), x_mov_2], dim=1))
        x_2 = self.trans_2(x)
        flow_2 = self.reghead_2(x_2) + flow_3_up

        # Finest scale: convolutional refinement, Eqs. (16)-(17).
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1 = self.SpatialTransformer(x_mov_1, flow_2_up)
        x_1 = self.conv_1(torch.cat([x_fix_1, self.upsample_1(x_2), x_mov_1], dim=1))
        flow_1 = self.reghead_1(x_1) + flow_2_up

        return flow_1, affine_para


# =============================================================================
# Warping layers
# =============================================================================
def _identity_affine(device) -> torch.Tensor:
    identity = torch.zeros((1, 3, 4), device=device)
    identity[0, 0, 0] = 1
    identity[0, 1, 1] = 1
    identity[0, 2, 2] = 1
    return identity


class AffineTransformer_block(nn.Module):
    """Warp ``src`` with the affine parameters predicted by the affine head."""

    def __init__(self, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, src: torch.Tensor, affine_para: torch.Tensor) -> torch.Tensor:
        affine_matrix = affine_para.reshape(1, 3, 4) + _identity_affine(src.device)
        affine_grid = nnf.affine_grid(affine_matrix, src.shape, align_corners=False)
        return nnf.grid_sample(src, affine_grid, align_corners=False, mode=self.mode)


class SpatialTransformer_block(nn.Module):
    """Pull-back warping ``W(src, D)`` of a ``[B, C, 1, H, W]`` tensor."""

    def __init__(self, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        src = src.squeeze(dim=2)
        flow = flow.squeeze(dim=2)[:, 1:3, :, :]
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grid = torch.stack(torch.meshgrid(vectors, indexing="ij")).unsqueeze(0)
        grid = grid.type(torch.FloatTensor).to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]

        out_2d = nnf.grid_sample(src, new_locs, align_corners=False, mode=self.mode)
        return out_2d.unsqueeze(dim=2)


class ResizeTransformer_block(nn.Module):
    """Resize a displacement field and rescale its magnitudes accordingly."""

    def __init__(self, resize_factor: float, mode: str = "trilinear"):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor < 1:
            x = nnf.interpolate(x, align_corners=False, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            x = self.factor * x
            x = x.squeeze(dim=2)
            x = nnf.interpolate(x, align_corners=False, scale_factor=self.factor, mode="bilinear")
            x = x.unsqueeze(dim=2)
        return x


# =============================================================================
# Convolutional and head blocks
# =============================================================================
class BoundaryExtractor(nn.Module):
    """Auxiliary boundary head.

    Not used in the forward pass of the published models; it is retained only
    because its parameters are part of the released checkpoints.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.reduce = nn.Conv3d(channels, channels // 2, kernel_size=1, padding=0)
        self.sobel_like = nn.Conv3d(channels // 2, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sobel_like(self.reduce(x))


class Conv_block(nn.Module):
    """Two 3x3 convolutions, each followed by LeakyReLU and instance norm."""

    def __init__(self, in_channels: int, out_channels: int, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.Conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.norm_1 = nn.InstanceNorm3d(out_channels)
        self.Conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.norm_2 = nn.InstanceNorm3d(out_channels)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.boundary_extractor = BoundaryExtractor(out_channels)

    def Conv_forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.norm_1(self.LeakyReLU(self.Conv_1(x_in)))
        return self.norm_2(self.LeakyReLU(self.Conv_2(x)))

    def forward(self, x_in: torch.Tensor, return_boundaries: bool = False):
        if self.use_checkpoint and x_in.requires_grad:
            x_out = _checkpoint(self.Conv_forward, x_in)
        else:
            x_out = self.Conv_forward(x_in)
        if return_boundaries:
            return x_out, self.boundary_extractor(x_out)
        return x_out


class DeformHead_block(nn.Module):
    """Residual flow head ``h_l`` (initialised close to zero)."""

    def __init__(self, in_channels: int, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.reg_head = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1, padding="same")
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and x_in.requires_grad:
            return _checkpoint(self.reg_head, x_in)
        return self.reg_head(x_in)


class AffineHead_block(nn.Module):
    """Global affine head ``h_aff`` and its dense-field representation ``Phi``, Eqs. (9)-(10)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.affine_head_1 = nn.Linear(in_channels, in_channels // 2, bias=False)
        self.affine_head_2 = nn.Linear(in_channels // 2, 12, bias=False)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self, x_in: torch.Tensor):
        x = torch.mean(x_in, dim=(2, 3, 4))
        affine_para = self.Tanh(self.affine_head_2(self.ReLU(self.affine_head_1(x))))
        affine_matrix = affine_para.reshape(1, 3, 4) + _identity_affine(x_in.device)
        affine_grid = nnf.affine_grid(affine_matrix, x_in.shape, align_corners=False)

        shape = x_in.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        grid = torch.stack(torch.meshgrid(vectors, indexing="ij")).unsqueeze(0)
        grid = grid.type(torch.FloatTensor).to(x_in.device)

        affine_grid = affine_grid[..., [2, 1, 0]].permute(0, 4, 1, 2, 3)
        for i in range(len(shape)):
            affine_grid[:, i, ...] = (affine_grid[:, i, ...] / 2.0 + 0.5) * (shape[i] - 1)
        flow = affine_grid - grid
        return flow, affine_para


class PatchExpanding_block(nn.Module):
    """2x transposed-convolution upsampling followed by layer norm."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(embed_dim // 2)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x_in.squeeze(dim=2)).unsqueeze(dim=2)
        x = einops.rearrange(x, "b c d h w -> b d h w c")
        x = self.norm(x)
        return einops.rearrange(x, "b d h w c -> b c d h w")


# =============================================================================
# Swin-transformer stages (Methods, "Swin-transformer stages", Eqs. (18)-(19))
# =============================================================================
class SwinTrans_stage_block(nn.Module):
    """A stage of alternating non-shifted / shifted window attention blocks."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        window_size: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for _ in window_size)
        self.blocks = nn.ModuleList(
            [
                SwinTrans_Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x_in.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x_in.device)

        x = einops.rearrange(x_in, "b c d h w -> b d h w c")
        for block in self.blocks:
            x = block(x, mask_matrix=attn_mask)
        return einops.rearrange(x, "b d h w c -> b c d h w")


class SwinTrans_Block(nn.Module):
    """``X' = X + MSA(LN(X)); X'' = X' + MLP(LN(X'))``."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: list,
        shift_size: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MSA_block(
            embed_dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP_block(hidden_size=embed_dim, mlp_dim=int(embed_dim * mlp_ratio), dropout_rate=drop)

    def forward_part1(self, x_in: torch.Tensor, mask_matrix: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.norm1(x_in)
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = nnf.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)

        if any(i > 0 for i in shift_size):
            x_out = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x_out = shifted_x
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x_out = x_out[:, :d, :h, :w, :].contiguous()
        return x_out

    def forward_part2(self, x_in: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm2(x_in))

    def forward(self, x_in: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_checkpoint and x_in.requires_grad:
            x = x_in + _checkpoint(self.forward_part1, x_in, mask_matrix)
        else:
            x = x_in + self.forward_part1(x_in, mask_matrix)
        if self.use_checkpoint and x.requires_grad:
            return x + _checkpoint(self.forward_part2, x)
        return x + self.forward_part2(x)


class MSA_block(nn.Module):
    """Window multi-head self-attention with relative positional bias."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: list,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads,
            )
        )
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size[0]),
                torch.arange(window_size[1]),
                torch.arange(window_size[2]),
                indexing="ij",
            )
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x_in: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, c = x_in.shape
        qkv = self.qkv(x_in).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.attn_drop(self.Softmax(attn)).to(v.dtype)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class MLP_block(nn.Module):
    """Two-layer GELU MLP."""

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
        self.GELU = nn.GELU()

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.GELU(self.linear1(x_in)))
        return self.drop2(self.linear2(x))


# =============================================================================
# Window helpers
# =============================================================================
def compute_mask(dims, window_size, shift_size, device):
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(attn_mask == 0, 0.0)


def window_partition(x_in, window_size):
    b, d, h, w, c = x_in.shape
    x = x_in.view(
        b,
        d // window_size[0],
        window_size[0],
        h // window_size[1],
        window_size[1],
        w // window_size[2],
        window_size[2],
        c,
    )
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)


def window_reverse(windows, window_size, dims):
    b, d, h, w = dims
    x = windows.view(
        b,
        d // window_size[0],
        h // window_size[1],
        w // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    return tuple(use_window_size), tuple(use_shift_size)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor
