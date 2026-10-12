import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
from matplotlib import pyplot as plt

def visualize_channels_in_batches(array_4d, batch_size=32):
    """
    array_4d:  shape (1, 128, 64, 64)
               i.e., (batch, channels, height, width)
    batch_size: number of feature channels to display per figure
    """

    # 1) Remove the first dimension (batch=1) so we have shape (128, 64, 64)
    #    We'll call it (channels, height, width)
    array_4d = array_4d.detach().cpu().numpy()
    array_3d = array_4d[0]  # shape (128, 64, 64)

    num_channels = array_3d.shape[0]  # 128
    height, width = array_3d.shape[1], array_3d.shape[2]

    # 2) Loop over channels in steps of 'batch_size'
    start_idx = 0
    while start_idx < num_channels:
        end_idx = min(start_idx + batch_size, num_channels)
        channels_subset = array_3d[start_idx:end_idx]  # shape (subset_size, 64, 64)

        # 3) Setup subplots
        subset_size = channels_subset.shape[0]
        # e.g., a 4x8 grid if batch_size=32 => we can adapt rows/cols as needed
        rows, cols = 4, 8
        fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
        axes = axes.flatten()

        # 4) Plot each channel in a subplot
        for i in range(subset_size):
            ax = axes[i]
            # channels_subset[i] => shape (64, 64), one feature map
            ax.imshow(channels_subset[i], cmap='gray', origin='upper')
            ax.set_title(f"Channel {start_idx + i}")
            ax.axis('off')

        # For any unused subplot axes (e.g., if subset_size < batch_size), turn them off
        for i in range(subset_size, rows * cols):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

        start_idx += batch_size
# ---------- helpers ----------
@torch.no_grad()
def show_channelwise_loss_maps(ncc_maps, valid_support=None, title_prefix="Masked NCC"):
    """
    ncc_maps: [B,C,D,H,W], here D=1
    valid_support: [B,1,D,H,W] optional, shows where windows had enough overlap
    """
    ncc0 = ncc_maps[0]          # [C,1,H,W]
    C = ncc0.shape[0]
    loss_maps = 1.0 - ncc0[:,0] # [C,H,W]

    ncols = 5
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.ravel()
    for c in range(C):
        ax = axes[c]
        im = ax.imshow(loss_maps[c].cpu().numpy(), origin='upper')
        ax.set_title(f"{title_prefix} ch{c} (mean={loss_maps[c].mean():.3f})")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[C:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    if valid_support is not None:
        plt.figure(figsize=(4,4))
        plt.imshow(valid_support[0,0,0].cpu().numpy(), origin='upper')
        plt.title("Valid overlap support (>= min_count)")
        plt.axis('off')
        plt.colorbar()
        plt.show()




# ---------- masked local NCC ----------

import torch
import torch.nn.functional as F

def _kernel_3d(win_dhw, device, channels):
    kD, kH, kW = win_dhw
    weight = torch.ones(channels, 1, kD, kH, kW, device=device)
    pad = (kD // 2, kH // 2, kW // 2)
    return weight, pad





class MaskedNCC:
    def __init__(self, win=11, eps=1e-5, var_eps=1e-6):
        self.win = (1, win, win)
        self.eps = eps
        self.var_eps = var_eps

    def loss(self, fixed_combo, moving_combo):

        # Optional fp32 block (helps with AMP underflow)

        fixed  = fixed_combo[:,:10,:,:,:].float()
        moving = moving_combo[:,:10,:,:,:].float()
        fixed_mask = fixed_combo[:,-1,:,:,:]
        moving_mask = moving_combo[:,-1,:,:,:]


        # Sanitize inputs: kill NaN/Inf early
        fixed  = torch.nan_to_num(fixed,  nan=0.0, posinf=0.0, neginf=0.0)
        moving = torch.nan_to_num(moving, nan=0.0, posinf=0.0, neginf=0.0)
        B, C, D, H, W = fixed.shape
        device = fixed.device

        overlap = (fixed_mask * moving_mask > 0.5).float()
        overlap_c = overlap.repeat(1, C, 1, 1, 1)
        # f,a = plt.subplots(1,3)
        # a[0].imshow(fixed_mask.detach().cpu().numpy().squeeze())
        # a[1].imshow(moving_mask.detach().cpu().numpy().squeeze())
        # a[2].imshow(overlap.detach().cpu().numpy().squeeze())
        # plt.show()
        # test = input()

        # Ensure kernel fits
        kD, kH, kW = self.win
        if D == 1: kD = 1
        kH = min(kH | 1, H)  # keep odd, <= H
        kW = min(kW | 1, W)
        weight, pad = _kernel_3d((kD, kH, kW), device, C)
        win_size = float(kD * kH * kW)

        def gconv(x):
            return F.conv3d(x, weight, bias=None, stride=1, padding=pad, groups=C)

        # ---- Explicitly zero out outside-mask values (prevents NaN*0) ----
        fixed_m  = torch.where(overlap_c > 0, fixed,  torch.zeros_like(fixed))
        moving_m = torch.where(overlap_c > 0, moving, torch.zeros_like(moving))

        # print(fixed_m.shape)
        # f,a = plt.subplots(1,2)
        # a[0].imshow(fixed_m.detach().cpu().numpy().squeeze()[0,:,:])
        # a[1].imshow(moving_m.detach().cpu().numpy().squeeze()[0, :, :])
        # plt.show()
        # # test = input()


        # ---- Local counts and sums ----
        M      = gconv (overlap_c) + self.eps
        I_sum  = gconv(fixed_m)
        J_sum  = gconv(moving_m)
        I2_sum = gconv(fixed_m * fixed_m)
        J2_sum = gconv(moving_m * moving_m)
        IJ_sum = gconv(fixed_m * moving_m)

        u_I = I_sum / M
        u_J = J_sum / M

        # Variances/cov with a positive floor BEFORE sqrt
        I_var = (I2_sum - 2*u_I*I_sum + u_I*u_I*M).clamp_min(self.var_eps)
        J_var = (J2_sum - 2*u_J*J_sum + u_J*u_J*M).clamp_min(self.var_eps)
        cross =  IJ_sum - u_I*J_sum - u_J*I_sum + u_I*u_J*M

        denom = torch.sqrt(I_var) * torch.sqrt(J_var) + self.eps
        ncc = cross / denom

        # ---- Support (valid windows) ----
        min_count = max(1, 0.95*int(win_size))
        support = (M >= min_count).float().detach()  # detach: no gradients into mask
        #show_channelwise_loss_maps(ncc*support)

        valid = support.sum().clamp_min(1.0)
        ncc_mean = (ncc * support).sum() / valid
        loss = -ncc_mean

        return loss

class NCC:
    def __init__(self, win=9):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred
        c = y_pred.shape[1]

        # get dimension of volume
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, c, *win]).to(y_pred.device)


        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        # print(-torch.mean(cc))
        # visualize_channels_in_batches(cc[:,:,0,:,:])


        return -torch.mean(cc)

class Grad:
    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):

        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dy = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad

def Regu_loss( y_pred):
    return Grad('l2').loss( y_pred)

