"""Training objective of the transition-field network (Methods, Eqs. (20)-(23)).

``L_reg = λ_m [L_morph(W_aff(M_a), M_b) + L_morph(W(M_a, D), M_b)]
        + λ_p [L_mol(W_aff(P_a), P_b)  + L_mol(W(P_a, D^P), P_b)]
        + λ_D L_smooth(D)``
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["NCC", "MaskedNCC", "Grad", "Regu_loss", "smoothness_loss", "RegistrationLoss"]


class NCC:
    """Squared local normalised cross-correlation, Eq. (21).

    ``loss(I, J) = -mean_u Cov_N(u)(I, J)^2 / (Var_N(u)(I) Var_N(u)(J) + eps)``
    with a ``win x win`` window (default 9; ``RegistrationLoss`` and
    ``axis.tl.fit_transition`` use 21 for morphology).
    """

    def __init__(self, win: int = 9):
        self.win = win

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        Ii, Ji = y_true, y_pred
        c = y_pred.shape[1]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], f"volumes should be 1 to 3 dimensions. found: {ndims}"

        win = [self.win] * ndims
        sum_filt = torch.ones([1, c, *win]).to(y_pred.device)
        pad_no = math.floor(win[0] / 2)
        stride = (1,) * ndims
        padding = (pad_no,) * ndims
        conv_fn = getattr(F, f"conv{ndims}d")

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(Ii * Ii, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(Ji * Ji, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(Ii * Ji, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        return -torch.mean(cc)

    __call__ = loss


def _box_kernel_3d(win_dhw, device, channels):
    kD, kH, kW = win_dhw
    weight = torch.ones(channels, 1, kD, kH, kW, device=device)
    return weight, (kD // 2, kH // 2, kW // 2)


class MaskedNCC:
    """Signed, support-masked local NCC for molecular pseudo-images, Eq. (22).

    The last channel of each input is the binary molecular-support mask; the
    first ``n_channels`` channels are the pseudo-image. Only windows with at
    least ``min_overlap`` (95 %) valid overlap contribute.
    """

    def __init__(self, win: int = 11, eps: float = 1e-5, var_eps: float = 1e-6, n_channels: int = 10, min_overlap: float = 0.95):
        self.win = (1, win, win)
        self.eps = eps
        self.var_eps = var_eps
        self.n_channels = n_channels
        self.min_overlap = min_overlap

    def loss(self, fixed_combo: torch.Tensor, moving_combo: torch.Tensor) -> torch.Tensor:
        fixed = fixed_combo[:, : self.n_channels].float()
        moving = moving_combo[:, : self.n_channels].float()
        fixed_mask = fixed_combo[:, -1]
        moving_mask = moving_combo[:, -1]

        fixed = torch.nan_to_num(fixed, nan=0.0, posinf=0.0, neginf=0.0)
        moving = torch.nan_to_num(moving, nan=0.0, posinf=0.0, neginf=0.0)
        _, C, D, H, W = fixed.shape
        device = fixed.device

        overlap = (fixed_mask * moving_mask > 0.5).float().unsqueeze(1)
        overlap_c = overlap.repeat(1, C, 1, 1, 1)

        kD, kH, kW = self.win
        if D == 1:
            kD = 1
        kH = min(kH | 1, H)
        kW = min(kW | 1, W)
        weight, pad = _box_kernel_3d((kD, kH, kW), device, C)
        win_size = float(kD * kH * kW)

        def gconv(x):
            return F.conv3d(x, weight, bias=None, stride=1, padding=pad, groups=C)

        fixed_m = torch.where(overlap_c > 0, fixed, torch.zeros_like(fixed))
        moving_m = torch.where(overlap_c > 0, moving, torch.zeros_like(moving))

        M = gconv(overlap_c) + self.eps
        I_sum = gconv(fixed_m)
        J_sum = gconv(moving_m)
        I2_sum = gconv(fixed_m * fixed_m)
        J2_sum = gconv(moving_m * moving_m)
        IJ_sum = gconv(fixed_m * moving_m)

        u_I = I_sum / M
        u_J = J_sum / M
        I_var = (I2_sum - 2 * u_I * I_sum + u_I * u_I * M).clamp_min(self.var_eps)
        J_var = (J2_sum - 2 * u_J * J_sum + u_J * u_J * M).clamp_min(self.var_eps)
        cross = IJ_sum - u_I * J_sum - u_J * I_sum + u_I * u_J * M
        ncc = cross / (torch.sqrt(I_var) * torch.sqrt(J_var) + self.eps)

        min_count = max(1, self.min_overlap * int(win_size))
        support = (M >= min_count).float().detach()
        valid = support.sum().clamp_min(1.0)
        return -(ncc * support).sum() / valid

    __call__ = loss


class Grad:
    """Mean squared (``l2``) or absolute (``l1``) finite-difference gradient."""

    def __init__(self, penalty: str = "l1", loss_mult: Optional[float] = None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dy = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
        if self.penalty == "l2":
            dx = dx * dx
            dy = dy * dy
        grad = (torch.mean(dx) + torch.mean(dy)) / 2.0
        if self.loss_mult is not None:
            grad = grad * self.loss_mult
        return grad


def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    """``L_smooth(D) = ½ (E‖∂x D‖² + E‖∂y D‖²)``, Eq. (23)."""
    return Grad("l2").loss(flow)


Regu_loss = smoothness_loss
"""Original name of [`smoothness_loss`][axis.registration.losses.smoothness_loss], used by the original training scripts."""


class RegistrationLoss:
    """The full objective of Eq. (20).

    Parameters
    ----------
    lambda_morph, lambda_mol, lambda_smooth
        Weights ``λ_m``, ``λ_p`` and ``λ_D``.
    morph_win
        Local NCC window size of the morphology term.
    mol_win
        Local NCC window size of the molecular term.
    n_mol_channels
        Number of pseudo-image channels (the mask is the last channel).
    """

    def __init__(
        self,
        lambda_morph: float = 1.0,
        lambda_mol: float = 1.0,
        lambda_smooth: float = 1.0,
        morph_win: int = 21,
        mol_win: int = 15,
        n_mol_channels: int = 10,
        min_overlap: float = 0.95,
    ):
        self.lambda_morph = lambda_morph
        self.lambda_mol = lambda_mol
        self.lambda_smooth = lambda_smooth
        self.morph = NCC(win=morph_win)
        self.mol = MaskedNCC(win=mol_win, n_channels=n_mol_channels, min_overlap=min_overlap)

    def __call__(self, fixed_image, fixed_gene, outputs) -> Dict[str, torch.Tensor]:
        warped_image, warped_gene, flow, _flow_gene, affined_image, affined_gene, _ = outputs
        morph = self.morph(fixed_image, warped_image) + self.morph(fixed_image, affined_image)
        mol = self.mol(fixed_gene, warped_gene) + self.mol(fixed_gene, affined_gene)
        smooth = smoothness_loss(flow[:, 1:3])
        total = self.lambda_morph * morph + self.lambda_mol * mol + self.lambda_smooth * smooth
        return {"loss": total, "morph": morph, "mol": mol, "smooth": smooth}
