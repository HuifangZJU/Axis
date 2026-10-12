import sys
import math
import numpy as np
import einops
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal
from matplotlib import pyplot as plt


########################################################
# Networks
########################################################
    
class NICE_Trans(nn.Module):
    
    def __init__(self, 
                 in_channels: int = 1,
                 enc_channels: int = 4,
                 dec_channels: int = 8,
                 use_checkpoint: bool = True):
        super().__init__()
        
        self.Encoder = Conv_encoder(in_channels=in_channels,
                                    channel_num=enc_channels,
                                    use_checkpoint=use_checkpoint)
        self.Decoder = Trans_decoder(in_channels=enc_channels,
                                     channel_num=dec_channels, 
                                     use_checkpoint=use_checkpoint)
        
        self.SpatialTransformer = SpatialTransformer_block_2D(mode='bilinear')
        self.AffineTransformer = AffineTransformer_block_2D(mode='bilinear')

    def forward(self, fixed_image,fixed_gene, moving_image,moving_gene):
        # print(fixed_image.shape)  #[1, 3, 1024, 1024]
        # print(fixed_gene.shape)   #[1, 10, 64, 64]
        # print(moving_image.shape) #[1, 3, 1024, 1024]
        # print(moving_gene.shape)  #[1, 10, 64, 64]

        x_fix = self.Encoder(fixed_image,fixed_gene)
        x_mov = self.Encoder(moving_image,moving_gene)


        flow, affine_para = self.Decoder(x_fix, x_mov)
        warped_image = self.SpatialTransformer(moving_image, flow)
        affined_image = self.AffineTransformer(moving_image, affine_para)

        flow_gene = nnf.interpolate(
            flow,
            size=moving_gene.shape[-2:],  # (64, 64)
            mode='bilinear',
            align_corners=True
        )  # Output shape: [1, 2, 64, 64]


        warped_gene = self.SpatialTransformer(moving_gene, flow_gene)
        affined_gene = self.AffineTransformer(moving_gene, affine_para)


        return warped_image,warped_gene, flow, affined_image,affined_gene, affine_para


########################################################
# Encoder/Decoder
########################################################
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

class Conv_encoder(nn.Module):
    
    def __init__(self, 
                 in_channels: int,
                 channel_num: int, 
                 use_checkpoint: bool = False):
        super().__init__()
        
        self.conv_1 = Conv_block(in_channels, channel_num, use_checkpoint)
        self.conv_2 = Conv_block(channel_num, channel_num*2, use_checkpoint)
        self.conv_3 = Conv_block(channel_num*2, channel_num*4, use_checkpoint)
        self.conv_4 = Conv_block(channel_num*4, channel_num*8, use_checkpoint)
        self.conv_5 = Conv_block(channel_num*8, channel_num*8, use_checkpoint)

        self.gene_conv_1 = Conv_block(10, channel_num * 8, use_checkpoint)
        self.downsample = nn.AvgPool2d(2, stride=2)

    def forward(self, x_in,g_in):

        x_1 = self.conv_1(x_in)
        x = self.downsample(x_1)

        x_2 = self.conv_2(x)
        x = self.downsample(x_2)

        x_3 = self.conv_3(x)
        x = self.downsample(x_3)

        x_4 = self.conv_4(x)
        x = self.downsample(x_4)

        x_5 = self.conv_5(x)
        g_fe = self.gene_conv_1(g_in)
        x_combined = torch.cat((x_5, g_fe), dim=1)

        visualize_channels_in_batches(x_5, batch_size=32)
        
        return [x_1, x_2, x_3, x_4, x_combined]
        # return [x_1, x_2, x_3, x_4, x_5]
    

class Trans_decoder(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 channel_num: int, 
                 use_checkpoint: bool = False):
        super().__init__()
        
        self.conv_1 = Conv_block(in_channels*2+channel_num, channel_num, use_checkpoint)
        self.trans_2 = SwinTrans_stage_block_2D(embed_dim=channel_num*2,
                                             num_layers=4,
                                             num_heads=channel_num//8,
                                             window_size=[5,5],
                                             use_checkpoint=use_checkpoint)
        self.trans_3 = SwinTrans_stage_block_2D(embed_dim=channel_num*4,
                                             num_layers=4,
                                             num_heads=channel_num//4,
                                             window_size=[5,5],
                                             use_checkpoint=use_checkpoint)
        self.trans_4 = SwinTrans_stage_block_2D(embed_dim=channel_num*8,
                                             num_layers=4,
                                             num_heads=channel_num//2,
                                             window_size=[5,5],
                                             use_checkpoint=use_checkpoint)
        self.trans_5 = SwinTrans_stage_block_2D(embed_dim=channel_num*16,
                                             num_layers=4,
                                             num_heads=channel_num,
                                             window_size=[5,5],
                                             use_checkpoint=use_checkpoint)
        
        self.backdim_2 = nn.Conv2d(in_channels*4+channel_num*2, channel_num*2, kernel_size=1, stride=1, padding='same')
        self.backdim_3 = nn.Conv2d(in_channels*8+channel_num*4, channel_num*4, kernel_size=1, stride=1, padding='same')
        self.backdim_4 = nn.Conv2d(in_channels*16+channel_num*8, channel_num*8, kernel_size=1, stride=1, padding='same')
        self.backdim_5 = nn.Conv2d(in_channels*32, channel_num*16, kernel_size=1, stride=1, padding='same')
        
        self.upsample_1 = PatchExpanding_block_2D(embed_dim=channel_num*2)
        self.upsample_2 = PatchExpanding_block_2D(embed_dim=channel_num*4)
        self.upsample_3 = PatchExpanding_block_2D(embed_dim=channel_num*8)
        self.upsample_4 = PatchExpanding_block_2D(embed_dim=channel_num*16)
        
        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num*2, use_checkpoint)
        self.reghead_3 = DeformHead_block(channel_num*4, use_checkpoint)
        self.reghead_4 = DeformHead_block(channel_num*8, use_checkpoint)
        self.reghead_5 = AffineHead_block_2D(channel_num*16)
        
        self.ResizeTransformer = ResizeTransformer_block_2D(resize_factor=2, mode='bilinear')
        self.SpatialTransformer = SpatialTransformer_block_2D(mode='bilinear')

    def forward(self, x_fix, x_mov):
        
        x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5 = x_fix
        x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5 = x_mov



        # Step 1
        x = torch.cat([x_fix_5, x_mov_5], dim=1)
        x = self.backdim_5(x)
        x_5 = self.trans_5(x)
        # visualize_channels_in_batches(x_5, 32)

        flow_5, affine_para = self.reghead_5(x_5)


        
        # Step 2
        flow_5_up = self.ResizeTransformer(flow_5)
        x_mov_4 = self.SpatialTransformer(x_mov_4, flow_5_up)
        
        x = self.upsample_4(x_5)
        x = torch.cat([x_fix_4, x, x_mov_4], dim=1)
        x = self.backdim_4(x)
        x_4 = self.trans_4(x)
        
        x = self.reghead_4(x_4)
        flow_4 = x + flow_5_up
        
        # Step 3
        flow_4_up = self.ResizeTransformer(flow_4)
        x_mov_3 = self.SpatialTransformer(x_mov_3, flow_4_up)
        
        x = self.upsample_3(x_4)
        x = torch.cat([x_fix_3, x, x_mov_3], dim=1)
        x = self.backdim_3(x)
        x_3 = self.trans_3(x)
        
        x = self.reghead_3(x_3)
        flow_3 = x + flow_4_up
        
        # Step 4
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2 = self.SpatialTransformer(x_mov_2, flow_3_up)
        
        x = self.upsample_2(x_3)
        x = torch.cat([x_fix_2, x, x_mov_2], dim=1)
        x = self.backdim_2(x)
        x_2 = self.trans_2(x)
        
        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up
        
        # Step 5
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1 = self.SpatialTransformer(x_mov_1, flow_2_up)
        
        x = self.upsample_1(x_2)
        x = torch.cat([x_fix_1, x, x_mov_1], dim=1)
        x_1 = self.conv_1(x)
        
        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up
        
        return flow_1, affine_para
    

########################################################
# Blocks
########################################################

class AffineTransformer_block_2D(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src: torch.Tensor, affine_para: torch.Tensor):
        device = src.device
        id_2d = torch.zeros((1, 2, 3), device=device)
        id_2d[0, 0, 0] = 1.0  # x-scale
        id_2d[0, 1, 1] = 1.0  # y-scale

        batch_size = src.shape[0]
        if affine_para.shape[0] == 1 and batch_size > 1:
            affine_para = affine_para.expand(batch_size, -1)

        # Reshape and NEGATE TRANSLATION PARAMETERS (tx, ty)
        affine_matrix_part = affine_para.view(-1, 2, 3).clone()
        affine_matrix_part[:, :, 2] *= -1  # Fix direction of tx, ty
        affine_matrix = affine_matrix_part + id_2d  # (N, 2, 3)

        affine_grid = nnf.affine_grid(
            theta=affine_matrix,
            size=src.shape,
            align_corners=True
        )

        warped = nnf.grid_sample(
            src,
            affine_grid,
            mode=self.mode,
            align_corners=True
        )
        return warped


class SpatialTransformer_block_2D(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        """
        Args:
            src:  (N, C, H, W) source image/tensor
            flow: (N, 2, H, W) displacement field in [dy, dx] order

        Returns:
            warped: (N, C, H, W) the warped image via the given flow
        """
        shape = flow.shape[2:]  # (H, W)
        # 2) Create base grid of coordinates [0..H-1, 0..W-1]
        vectors = [torch.arange(0, s, device=flow.device) for s in shape]
        grids = torch.meshgrid(*vectors, indexing='ij')  # => (H, W)
        grid = torch.stack(grids, dim=0).unsqueeze(0)  # => (1, 2, H, W)
        grid = grid.float()
        new_locs = grid + flow

        for i in range(len(shape)):
            size_i = shape[i]
            new_locs[:, i, ...] = 2.0 * (new_locs[:, i, ...] / (size_i - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)  # => (N,H,W,2)
        new_locs = new_locs[..., [1, 0]]  # => (N,H,W,2)
        warped = nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        return warped

class CoordinateWarper(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, coords, flow):
        """
        Args:
            coords: Tensor of shape [batch, n, 2] (x, y coordinates in pixel space)
            flow:   Tensor of shape [batch, 2, H, W] (displacement field in [dy, dx] order)
        Returns:
            warped_coords: Tensor of shape [batch, n, 2] (warped coordinates)
        """
        batch_size, n, _ = coords.shape
        H, W = flow.shape[2], flow.shape[3]

        # Normalize coordinates to [-1, 1] (required by grid_sample)
        x = coords[..., 0]  # Shape: [batch, n]
        y = coords[..., 1]
        x_norm = (x / (W - 1)) * 2 - 1  # Normalize x-coordinates
        y_norm = (y / (H - 1)) * 2 - 1  # Normalize y-coordinates
        grid = torch.stack([x_norm, y_norm], dim=-1)  # Shape: [batch, n, 2]

        # Sample displacement values at coordinates
        sampled_flow = nnf.grid_sample(
            flow,
            grid.unsqueeze(2),  # Add dummy dimension -> [batch, n, 1, 2]
            mode=self.mode,
            align_corners=False
        )  # Output shape: [batch, 2, n, 1]

        # Reshape and extract dx/dy displacements
        sampled_flow = sampled_flow.squeeze(-1).permute(0, 2, 1)  # [batch, n, 2]
        dx = sampled_flow[..., 1]  # x-displacement (from flow's 2nd channel)
        dy = sampled_flow[..., 0]  # y-displacement (from flow's 1st channel)

        # Apply displacements to original coordinates
        warped_x = x + dx
        warped_y = y + dy
        warped_coords = torch.stack([warped_x, warped_y], dim=-1)

        return warped_coords

class ResizeTransformer_block_2D(nn.Module):
    """
    Resizes a 2D feature map (B, C, H, W) by a specified factor < 1 (down) or > 1 (up).
    Optionally multiplies the feature map by this factor before or after resizing,
    to save memory.
    """

    def __init__(self, resize_factor: float, mode='bilinear'):
        """
        Args:
            resize_factor: scale factor for height and width.
                           if < 1.0 => downsample
                           if > 1.0 => upsample
            mode: interpolation mode ('bilinear', 'bicubic', etc.)
        """
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x: torch.Tensor):
        """
        x shape: (B, C, H, W)

        Returns: resized tensor of shape
                 (B, C, round(H*factor), round(W*factor))
        """
        # Downsampling case
        if self.factor < 1.0:
            # 1) interpolate => smaller shape
            x = nnf.interpolate(
                x,
                scale_factor=self.factor,
                mode=self.mode,
                align_corners=True
            )
            # 2) multiply => scale intensities
            x = self.factor * x

        # Upsampling case
        elif self.factor > 1.0:
            # 1) multiply => scale intensities
            x = self.factor * x
            # 2) interpolate => bigger shape
            x = nnf.interpolate(
                x,
                scale_factor=self.factor,
                mode=self.mode,
                align_corners=True
            )

        return x


class Conv_block(nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 out_channels:int , 
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.Conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_1 = nn.InstanceNorm2d(out_channels)
        
        self.Conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_2 = nn.InstanceNorm2d(out_channels)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def Conv_forward(self, x_in):

        x = self.Conv_1(x_in)
        x = self.LeakyReLU(x)
        x = self.norm_1(x)
        
        x = self.Conv_2(x)
        x = self.LeakyReLU(x)
        x_out = self.norm_2(x)
        
        return x_out
    
    def forward(self, x_in):
        
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Conv_forward, x_in)
        else:
            x_out = self.Conv_forward(x_in)
        
        return x_out
    
    
class DeformHead_block(nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.reg_head = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding='same')
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))
    
    def forward(self, x_in):
        
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.reg_head, x_in)
        else:
            x_out = self.reg_head(x_in)
        
        return x_out


class CoordinateAffineWarper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coords: torch.Tensor, affine_para: torch.Tensor, H: int, W: int):
        """
        Args:
            coords:      [batch, n, 2] (x/y coordinates in pixel space)
            affine_para: [batch, 6] (affine parameters [a11, a12, tx, a21, a22, ty])
            H, W:        Height/width of the image the affine is applied to
        Returns:
            warped_coords: [batch, n, 2] (transformed coordinates)
        """
        batch_size, n, _ = coords.shape

        # 1. Convert pixel coordinates to normalized space (align_corners=True)
        x_pixel, y_pixel = coords[..., 0], coords[..., 1]
        x_norm = (2 * x_pixel / (W - 1)) - 1  # Range: [-1, 1]
        y_norm = (2 * y_pixel / (H - 1)) - 1

        # 2. Create homogeneous coordinates [x_norm, y_norm, 1]
        ones = torch.ones_like(x_norm)
        homogeneous = torch.stack([x_norm, y_norm, ones], dim=-1)  # [batch, n, 3]

        # 3. Construct affine matrix (same as in AffineTransformer_block_2D)
        id_2d = torch.zeros((batch_size, 2, 3), device=coords.device)
        id_2d[:, 0, 0] = 1.0  # Identity x-scale
        id_2d[:, 1, 1] = 1.0  # Identity y-scale
        affine_matrix = affine_para.view(batch_size, 2, 3) + id_2d  # [batch, 2, 3]

        # 4. Apply affine matrix to coordinates
        transformed = torch.bmm(
            affine_matrix,
            homogeneous.permute(0, 2, 1)  # [batch, 3, n]
        ).permute(0, 2, 1)  # [batch, n, 2]

        # 5. Convert normalized coordinates back to pixel space
        x_norm_new, y_norm_new = transformed[..., 0], transformed[..., 1]
        x_pixel_new = (x_norm_new + 1) * (W - 1) / 2
        y_pixel_new = (y_norm_new + 1) * (H - 1) / 2

        return torch.stack([x_pixel_new, y_pixel_new], dim=-1)

class AffineHead_block_2D(nn.Module):
    """
    A 2D affine head that learns 6 parameters:
      [ a11, a12, t1, a21, a22, t2 ]
    forming a 2x3 transform matrix:
      [[1 + a11,      a12,     t1],
       [   a21, 1 + a22,     t2 ]]

    The transform is applied to a (B, C, H, W) feature map.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        # Instead of 12 params (3D), we produce 6 params (2D).
        self.affine_head_1 = nn.Linear(in_channels, in_channels // 2, bias=False)
        self.affine_head_2 = nn.Linear(in_channels // 2, 6, bias=False)

        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

        # Learnable scale factor


    def forward(self, x_in):
        """
        x_in: (B, C, H, W)

        Returns:
          flow: (B, 2, H, W) the displacement field derived from the affine transform
          affine_para: shape (B, 6), the learned parameters in [-1,1]
        """

        # 1) Global average pooling over (H,W), shape => (B, C)
        x = torch.mean(x_in, dim=(2, 3))

        # 2) Two-layer MLP => (B, 6)
        x = self.affine_head_1(x)  # => (B, in_channels//2)
        x = self.ReLU(x)
        x = self.affine_head_2(x)  # => (B, 6)
        affine_para = 0.1*self.Tanh(x)   # each param in [-1,1], shape (B, 6)

        # 3) Construct a 2D identity + offset matrix => shape (B, 2, 3)
        #    Minimal 2D identity: [[1, 0, 0],[0, 1, 0]]
        device = x_in.device
        identity_2d = torch.zeros((1, 2, 3), device=device)
        identity_2d[0, 0, 0] = 1.0  # x scale
        identity_2d[0, 1, 1] = 1.0  # y scale

        affine_matrix = affine_para.view(-1, 2, 3) + identity_2d  # => (B, 2, 3)

        # 4) Build affine grid for shape (B,C,H,W)
        b, c, h, w = x_in.shape
        affine_grid = nnf.affine_grid(affine_matrix, (b, c, h, w), align_corners=True)
        # => shape (B, H, W, 2), coordinate order is (y, x)

        # 5) Create a base coordinate grid => shape (1,2,H,W)
        vectors = [torch.arange(0, s, device=device) for s in (h, w)]
        grids = torch.meshgrid(*vectors, indexing='ij')  # => 2D => (H, W)
        base_grid = torch.stack(grids, dim=0).unsqueeze(0)  # => (1,2,H,W)
        base_grid = base_grid.float()

        # 6) Reorder from (y, x) => (x, y)
        #    So final dimension is [1,0].
        affine_grid = affine_grid[..., [1, 0]]  # => (B,H,W,2) => (B,H,W, (x,y))

        # 7) Permute => (B,2,H,W)
        affine_grid = affine_grid.permute(0, 3, 1, 2).contiguous()  # => (B,2,H,W)

        # 8) Convert normalized coords in [-1,1] => absolute pixel coords [0..(H-1)/(W-1)]
        #    For dimension i, "affine_grid[:, i, ...] = (affine_grid[:, i, ...]/2 + 0.5)*(size[i]-1)"
        for i, size_i in enumerate((h, w)):
            affine_grid[:, i, ...] = (affine_grid[:, i, ...] / 2.0 + 0.5) * float(size_i - 1)

        # 9) Displacement field = (affine_grid - base_grid)
        #    broadcast base_grid from shape (1,2,H,W) => (B,2,H,W)
        flow = affine_grid - base_grid

        return flow, affine_para
    
    
class PatchExpanding_block_2D(nn.Module):
    """
    Expands a feature map (B, embed_dim, H, W) to (B, embed_dim/2, 2H, 2W)
    and applies a LayerNorm on the channel dimension.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        # Transpose conv upsamples spatially by factor=2
        self.up_conv = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=embed_dim // 2,
            kernel_size=2,
            stride=2
        )
        # LayerNorm will be applied over the channel dimension (last dim after rearrange)
        self.norm = nn.LayerNorm(embed_dim // 2)

    def forward(self, x_in: torch.Tensor):
        """
        x_in shape: (B, embed_dim, H, W)

        Returns:
          x_out of shape: (B, embed_dim//2, H*2, W*2)
        """
        # 1) Upsample via transposed conv => (B, embed_dim//2, 2H, 2W)
        x = self.up_conv(x_in)

        # 2) Rearrange to (B, 2H, 2W, channels) so we can apply LayerNorm on the last dim
        x = einops.rearrange(x, 'b c h w -> b h w c')

        # 3) LayerNorm over the channel dimension
        x = self.norm(x)

        # 4) Rearrange back to (B, channels, 2H, 2W)
        x_out = einops.rearrange(x, 'b h w c -> b c h w')

        return x_out

class SwinTrans_stage_block_2D(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_layers: int,
                 num_heads: int,
                 window_size: list,   # e.g. [window_h, window_w]
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 use_checkpoint: bool = False):
        """
        A 2D version of the SwinTransformer stage block:
         - Inputs:  (B, C, H, W)
         - Output:  (B, C, H, W)
        """
        super().__init__()
        self.window_size = window_size
        # shift_size is half of the window size for the "Shifted Window" approach
        self.shift_size = tuple(ws // 2 for ws in window_size)
        self.no_shift = (0, 0)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = SwinTrans_Block_2D(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=self.window_size,
                shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                use_checkpoint=use_checkpoint
            )
            self.blocks.append(block)

    def forward(self, x_in):
        """
        x_in shape: (B, C, H, W)
        Returns:    (B, C, H, W)
        """
        b, c, h, w = x_in.shape

        # 1) Possibly adapt your 2D window/shift computation as needed
        window_size, shift_size = get_window_size_2d(
            (h, w),
            self.window_size,
            self.shift_size
        )

        # 2) Pad to multiples of window_size if needed
        hp = int(np.ceil(h / window_size[0])) * window_size[0]
        wp = int(np.ceil(w / window_size[1])) * window_size[1]

        # 3) Compute a 2D attention mask (for shifted windows)
        #    You'd adapt your "compute_mask" for 2D:
        attn_mask = compute_mask_2d(
            (hp, wp),
            window_size,
            shift_size,
            device=x_in.device
        )

        # 4) Rearrange from (B, C, H, W) -> (B, H, W, C)
        x = einops.rearrange(x_in, 'b c h w -> b h w c')

        # 5) Pass through each Swin Transformer block
        for block in self.blocks:
            x = block(x, mask_matrix=attn_mask)

        # 6) Rearrange back to (B, C, H, W)
        x_out = einops.rearrange(x, 'b h w c -> b c h w')

        return x_out


class SwinTrans_Block_2D(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,    # e.g. [win_h, win_w]
                 shift_size: list,    # e.g. [shift_h, shift_w]
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 use_checkpoint: bool = False):
        """
        Single 2D Swin Transformer block (window-based attention + MLP).
        Expects shape (B, H, W, C) inside forward_part1.
        """
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MSA_block_2D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP_block(
            hidden_size=embed_dim,
            mlp_dim=int(embed_dim * mlp_ratio),
            dropout_rate=drop
        )

    def forward_part1(self, x_in, mask_matrix):
        """
        x_in: (B, H, W, C)
        Returns: (B, H, W, C)
        """
        x = self.norm1(x_in)
        b, h, w, c = x.shape

        # 1) get possibly clamped window/shift
        window_size, shift_size = get_window_size_2d((h, w), self.window_size, self.shift_size)

        # 2) pad if needed to multiples of window_size
        pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
        x = nnf.pad(x, (0,0, 0,pad_r, 0,pad_b))  # => (B, h+pad_b, w+pad_r, C)
        _, hp, wp, _ = x.shape
        dims = (b, hp, wp)

        # 3) Shift if needed
        if (shift_size[0] > 0) or (shift_size[1] > 0):
            # roll along (1,2) => (H, W)
            shifted_x = torch.roll(
                x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # 4) Partition windows => (num_windows*b, win_h*win_w, C)
        x_windows = window_partition_2d(shifted_x, window_size)

        # 5) Attention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # 6) Reverse windows => (B, hp, wp, C)
        shifted_x = window_reverse_2d(attn_windows, window_size, dims)

        # 7) Reverse shift
        if (shift_size[0] > 0) or (shift_size[1] > 0):
            x_out = torch.roll(
                shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2)
            )
        else:
            x_out = shifted_x

        # 8) Remove extra padding
        if pad_b > 0 or pad_r > 0:
            x_out = x_out[:, :h, :w, :].contiguous()

        return x_out

    def forward_part2(self, x_in):
        """
        x_in: (B, H, W, C)
        Returns: same shape
        """
        x = self.norm2(x_in)
        x_out = self.mlp(x)
        return x_out

    def forward(self, x_in, mask_matrix=None):
        """
        x_in: (B, H, W, C)
        """
        if self.use_checkpoint and x_in.requires_grad:
            x = x_in + checkpoint.checkpoint(self.forward_part1, x_in, mask_matrix)
        else:
            x = x_in + self.forward_part1(x_in, mask_matrix)

        if self.use_checkpoint and x.requires_grad:
            x_out = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x_out = x + self.forward_part2(x)

        return x_out

class SwinTrans_stage_block_2D(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_layers: int,
                 num_heads: int,
                 window_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()

        self.window_size = window_size
        self.shift_size = [ws // 2 for ws in window_size]
        self.no_shift = [0, 0]

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = SwinTrans_Block_2D(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=self.window_size,
                shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                use_checkpoint=use_checkpoint
            )
            self.blocks.append(block)

    def forward(self, x_in):
        """
        x_in shape: (B, C, H, W) typically, so first rearrange to (B, H, W, C)
        then apply blocks, and revert.
        """
        b, c, h, w = x_in.shape

        # Possibly compute 2D mask if using shifted windows
        # e.g., we pad to multiples of window_size and call compute_mask_2d
        # for demonstration:
        # (there are many ways to handle partial coverage & dynamic shapes)
        device = x_in.device
        window_size, shift_size = get_window_size_2d((h, w), self.window_size, self.shift_size)
        hp = int(np.ceil(h / window_size[0])) * window_size[0]
        wp = int(np.ceil(w / window_size[1])) * window_size[1]
        attn_mask = compute_mask_2d((hp, wp), window_size, shift_size, device)

        # Rearrange => (B, H, W, C)
        x = x_in.permute(0, 2, 3, 1).contiguous()

        # Pass through each Swin block
        for block in self.blocks:
            x = block(x, mask_matrix=attn_mask)

        # Rearrange back => (B, C, H, W)
        x_out = x.permute(0, 3, 1, 2).contiguous()
        return x_out



class MSA_block_2D(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,      # e.g. [window_h, window_w]
                 qkv_bias: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        """
        2D Multi-Head Self-Attention block using relative position bias,
        adapted from the 3D version.
         - window_size = (window_h, window_w)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.window_size = window_size  # (wh, ww)
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        # 1) Define the size of the relative position bias table
        #    For a 2D window, size = ( (2*wh - 1) * (2*ww - 1), num_heads )
        wh, ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wh - 1) * (2 * ww - 1), num_heads)
        )

        # 2) Create coords for 2D (wh, ww)
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)

        # If you need to check meshgrid kwargs:
        mesh_args = torch.meshgrid.__kwdefaults__
        if mesh_args is not None:
            coords = torch.stack(
                torch.meshgrid(coords_h, coords_w, indexing="ij")
            )  # shape (2, wh, ww)
        else:
            # older PyTorch fallback
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))

        # coords shape: (2, wh, ww)
        coords_flatten = coords.flatten(1)     # shape (2, wh*ww)
        # 3) Compute relative coords
        #    relative_coords: shape (wh*ww, wh*ww, 2)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # => shape (2, (wh*ww), (wh*ww))

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # => shape ((wh*ww), (wh*ww), 2)

        # shift to ensure no negative indices
        relative_coords[:, :, 0] += (wh - 1)
        relative_coords[:, :, 1] += (ww - 1)

        # For indexing, multiply the first coordinate by (2*ww - 1), then add the second
        # => Flatten into a single index
        relative_coords[:, :, 0] *= (2 * ww - 1)

        relative_position_index = relative_coords.sum(dim=-1)
        # => shape (wh*ww, wh*ww)

        self.register_buffer("relative_position_index", relative_position_index)

        # 4) QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 5) Initialize
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x_in, mask=None):
        """
        x_in shape: (B, N, C)
         - B = batch
         - N = number of patches (wh * ww) * num_windows, etc.
         - C = embed_dim
        mask: optional attention mask if using shifted windows. shape depends on #windows, etc.

        Returns shape: (B, N, C)
        """
        b, n, c = x_in.shape  # e.g. b, wh*ww, embed_dim

        # 1) QKV
        # => shape (b, n, 3, num_heads, c//num_heads)
        # Then rearr => (3, b, num_heads, n, c//num_heads)
        qkv = self.qkv(x_in).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2) Scaled attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # shape (b, num_heads, n, n)

        # 3) Add relative position bias
        #    self.relative_position_index shape: (n, n)
        #    self.relative_position_bias_table shape: ( (2*wh-1)*(2*ww-1), num_heads )
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:n, :n].reshape(-1)
        ].reshape(n, n, -1)  # => (n, n, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # => shape (num_heads, n, n)
        attn = attn + relative_position_bias.unsqueeze(0)  # shape (b, num_heads, n, n)

        # 4) If shifted windows => apply mask
        if mask is not None:
            nw = mask.shape[0]
            # shape attn => (b, num_heads, n, n)
            attn = attn.view(b // nw, nw, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        # 5) Softmax + dropout
        attn = self.Softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)

        # 6) Weighted sum of values
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x_out = self.proj_drop(x)

        return x_out


    
class MLP_block(nn.Module):

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.GELU = nn.GELU()

    def forward(self, x_in):
        
        x = self.linear1(x_in)
        x = self.GELU(x)
        x = self.drop1(x)
        
        x = self.linear2(x)
        x_out = self.drop2(x)
        
        return x_out

    
########################################################
# Functions
########################################################

def compute_mask_2d(dims, window_size, shift_size, device):
    """
    Compute the attention mask for 2D shifted windows.

    Args:
        dims:        (H, W) - the padded height, width
        window_size: (win_h, win_w)
        shift_size:  (shift_h, shift_w)
        device:      torch device
    Returns:
        attn_mask:   Tensor of shape (num_windows * B, 1, win_h*win_w, win_h*win_w),
                     to be used in attention to block out cross-window interactions
                     when windows are shifted.
    """
    H, W = dims
    win_h, win_w = window_size
    shift_h, shift_w = shift_size

    # Create a mask for each of the 9 partitions in 2D
    # (Similar to 3D logic, but we only have (H, W))
    img_mask = torch.zeros((1, H, W, 1), device=device)  # shape (B=1, H, W, 1)
    cnt = 0

    # Define 3 slices along H, 3 slices along W => total 3*3 = 9 partitions
    # - slice(-win_h, None) covers the last win_h region
    # - slice(-win_h, -shift_h) covers the area from -win_h to -shift_h
    # - slice(-shift_h, None)   covers the area from -shift_h to end
    for h_slice in [
        slice(-win_h, None),
        slice(-win_h, -shift_h),
        slice(-shift_h, None)
    ]:
        for w_slice in [
            slice(-win_w, None),
            slice(-win_w, -shift_w),
            slice(-shift_w, None)
        ]:
            # Mark this partition with 'cnt'
            img_mask[:, h_slice, w_slice, :] = cnt
            cnt += 1

    # Partition the mask into windows of size (win_h, win_w)
    mask_windows = window_partition_2d(img_mask, (win_h, win_w))
    # shape => (num_windows, win_h*win_w, 1)
    mask_windows = mask_windows.squeeze(-1)
    # shape => (num_windows, win_h*win_w)

    # Construct the attention mask
    # We do pairwise differences:
    #   If they come from different partitions, difference != 0 => -100.0
    #   If from the same partition, difference = 0 => 0.0
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    # shape => (num_windows, win_h*win_w, win_h*win_w)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

def window_partition_2d(x_in, window_size):
    """
    x_in shape: (B, H, W, C)
    window_size: (win_h, win_w)

    Returns:
      windows of shape (num_windows * B, win_h * win_w, C).
    """
    b, h, w, c = x_in.shape
    win_h, win_w = window_size

    # 1) Reshape so we create blocks of size (win_h, win_w)
    #    shape => (B, H//win_h, win_h, W//win_w, win_w, C)
    x = x_in.view(
        b,
        h // win_h, win_h,
        w // win_w, win_w,
        c
    )

    # 2) Permute to group the windows together
    #    shape => (B, H//win_h, W//win_w, win_h, win_w, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # 3) Finally, flatten so each window is a row in the output
    #    shape => ((B * H//win_h * W//win_w), (win_h * win_w), C)
    windows = x.view(-1, win_h * win_w, c)

    return windows


def window_reverse_2d(windows, window_size, dims):
    """
    Inverse of window_partition_2d.

    Args:
        windows:    shape (num_windows * B, win_h * win_w, C)
        window_size: (win_h, win_w)
        dims:       (B, H, W) after any padding

    Returns:
        x_out:      shape (B, H, W, C)
    """
    b, H, W = dims
    win_h, win_w = window_size

    # windows shape: (b * num_windows, win_h*win_w, C)
    # Rebuild to shape (b, H//win_h, W//win_w, win_h, win_w, C)
    c = windows.shape[-1]
    x = windows.view(
        b,
        H // win_h,
        W // win_w,
        win_h,
        win_w,
        c
    )

    # Permute to gather patches into contiguous spatial layout
    # shape => (b, H, W, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x_out = x.view(b, H, W, c)

    return x_out


def get_window_size_2d(x_size, window_size, shift_size=None):
    """
    x_size:       (H, W) of the current feature map
    window_size:  (win_h, win_w)
    shift_size:   optional (shift_h, shift_w)

    If H < win_h, we set use_window_size[0] = H and shift_size[0] = 0.
    Similarly for width dimension.

    Returns:
      - if shift_size is None: (window_h, window_w) (possibly adjusted)
      - else:                  (window_h, window_w), (shift_h, shift_w)
    """
    use_window_size = list(window_size)

    if shift_size is not None:
        use_shift_size = list(shift_size)
    else:
        use_shift_size = None

    for i in range(len(x_size)):  # i in {0,1}
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if use_shift_size is not None:
                use_shift_size[i] = 0

    if use_shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)
    
    
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b) 
        return tensor
    
