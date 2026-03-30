import sys
import math
import numpy as np
import einops
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal

########################################################
# Networks
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


def find_inverse_warp_coords(uv_coords, flow):
    """
    Args:
        uv_coords: [N, 2] (x, y) in the warped image
        flow: [2, H, W] forward flow (dy, dx)

    Returns:
        approx_orig_coords: [N, 2] — approximate original locations
    """
    device = flow.device
    dy, dx = flow[0], flow[1]  # shape: [H, W]
    H, W = dx.shape

    # 1. Build dense grid
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    base_grid = torch.stack([x, y], dim=-1).float()  # [H, W, 2]

    # 2. Apply flow
    warped_grid = base_grid + torch.stack([dx, dy], dim=-1)  # [H, W, 2]

    # 3. Flatten
    warped_flat = warped_grid.reshape(-1, 2)  # [H*W, 2]
    base_flat = base_grid.reshape(-1, 2)  # [H*W, 2]

    # 4. Nearest neighbor search
    dists = torch.cdist(uv_coords.unsqueeze(0), warped_flat.unsqueeze(0))  # [1, N, H*W]
    nn_indices = dists.argmin(dim=-1).squeeze(0)  # [N]

    # 5. Use the corresponding original positions
    approx_orig_coords = base_flat[nn_indices]  # [N, 2]

    return approx_orig_coords



class NICE_Trans(nn.Module):
    
    def __init__(self, 
                 in_channels: int = 3,
                 in_gene_channels: int=10,
                 enc_channels: int = 8,
                 dec_channels: int = 16,
                 fusion_module: str = 'transformer',
                 use_checkpoint: bool = True):
        super().__init__()
        self.fusion_ = fusion_module
        
        self.Encoder = Conv_encoder(in_channels=in_channels,
                                    in_gene_channels=in_gene_channels,
                                    channel_num=enc_channels,
                                    fusion=self.fusion_,
                                    use_checkpoint=use_checkpoint)
        self.Decoder = Trans_decoder(in_channels=enc_channels,
                                     channel_num=dec_channels, 
                                     use_checkpoint=use_checkpoint)
        
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')
        self.AffineTransformer = AffineTransformer_block(mode='bilinear')
        self.gene_channels = in_gene_channels


    def forward(self, fixed_image,fixed_gene, moving_image,moving_gene):



        x_fix = self.Encoder(fixed_image,fixed_gene[:,:self.gene_channels,:,:,:])
        x_mov = self.Encoder(moving_image,moving_gene[:,:self.gene_channels,:,:,:])

        flow, affine_para = self.Decoder(x_fix, x_mov)

        warped_image = self.SpatialTransformer(moving_image, flow)

        affined_image = self.AffineTransformer(moving_image, affine_para)

        # flow = flow[:, 1:3,: :, :]
        # visualize_channels_in_batches(flow[:,:,0,:,:])
        flow_gene = nnf.interpolate(
            flow,
            size=moving_gene.shape[-3:],  # (64, 64)
            mode='trilinear',
            align_corners=False
        )  # Output shape: [1, 2, 64, 64]
        flow_gene = flow_gene/8

        affined_gene = self.AffineTransformer(moving_gene, affine_para)
        warped_gene = self.SpatialTransformer(moving_gene, flow_gene)

        return warped_image,warped_gene, flow, flow_gene,affined_image,affined_gene, affine_para


########################################################
# Encoder/Decoder
########################################################
class FusionBlock(nn.Module):
    def __init__(self, img_channels: int, gene_channels: int, fused_channels: int):
        super(FusionBlock, self).__init__()
        self.fusion_conv = nn.Conv3d(in_channels=img_channels + gene_channels,
                                     out_channels=fused_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)

    def forward(self, img_feat: torch.Tensor, gene_feat: torch.Tensor) -> torch.Tensor:
        # img_feat and gene_feat should be [B, C, D, H, W]
        x = torch.cat([img_feat, gene_feat], dim=1)
        fused = self.fusion_conv(x)
        return fused


class TransformerFusion(nn.Module):
    def __init__(self, img_channels=64, gene_channels=10, fused_channels=128, num_heads=4):
        super().__init__()

        # Linear projection layers to unified embedding space
        self.img_proj = nn.Conv2d(img_channels, fused_channels, kernel_size=1)
        self.gene_proj = nn.Conv2d(gene_channels, fused_channels, kernel_size=1)

        # Transformer encoder layer for fusion
        self.transformer = nn.TransformerEncoderLayer(
            d_model=fused_channels,
            nhead=num_heads,
            dim_feedforward=fused_channels * 2,
            batch_first=True
        )

        # Output convolution to combine and refine features
        self.fuse_conv = nn.Conv2d(fused_channels, fused_channels, kernel_size=1)

    def forward(self, img_feat, gene_feat):
        """
        img_feat: [B, 64, 1, H, W]
        gene_feat: [B, 10, 1, H, W]
        """

        B, _, _, H, W = img_feat.shape

        # Squeeze the depth dimension
        img_feat = img_feat.squeeze(2)  # [B, 64, H, W]
        gene_feat = gene_feat.squeeze(2)  # [B, 10, H, W]

        # Project features to common embedding space
        img_embed = self.img_proj(img_feat)  # [B, fused_channels, H, W]
        gene_embed = self.gene_proj(gene_feat)  # [B, fused_channels, H, W]

        # Flatten spatial dimensions into sequence (H*W)
        img_seq = img_embed.flatten(2).permute(0, 2, 1)  # [B, HW, fused_channels]
        gene_seq = gene_embed.flatten(2).permute(0, 2, 1)  # [B, HW, fused_channels]

        # Concatenate sequences along spatial dimension
        fused_seq = torch.cat([img_seq, gene_seq], dim=1)  # [B, 2*HW, fused_channels]

        # Transformer-based fusion
        fused_seq = self.transformer(fused_seq)  # [B, 2*HW, fused_channels]

        # Split and average back to original spatial size (H*W)
        fused_img_seq, fused_gene_seq = torch.chunk(fused_seq, 2, dim=1)
        fused_seq_avg = (fused_img_seq + fused_gene_seq) / 2  # [B, HW, fused_channels]

        # Reshape back to spatial map
        fused_map = fused_seq_avg.permute(0, 2, 1).reshape(B, -1, H, W)  # [B, fused_channels, H, W]

        # Final convolution refinement
        fused_out = self.fuse_conv(fused_map)  # [B, fused_channels, H, W]

        # Expand depth dimension back to match original
        fused_out = fused_out.unsqueeze(2)  # [B, fused_channels, 1, H, W]

        return fused_out


class Conv_encoder(nn.Module):
    
    def __init__(self, 
                 in_channels: int,
                 in_gene_channels: int,
                 channel_num: int,
                 fusion:str,
                 use_checkpoint: bool = False):
        super().__init__()
        
        self.conv_1 = Conv_block(in_channels, channel_num, use_checkpoint)
        self.conv_2 = Conv_block(channel_num, channel_num*2, use_checkpoint)
        self.conv_3 = Conv_block(channel_num*2, channel_num*4, use_checkpoint)
        self.conv_4 = Conv_block(channel_num*4, channel_num*8, use_checkpoint)
        self.conv_5 = Conv_block(channel_num * 8, channel_num * 16, use_checkpoint)

        if fusion =='transformer':
            self.fusion = TransformerFusion(img_channels=64,gene_channels=in_gene_channels,fused_channels=64, num_heads=8)
        else:
            self.fusion = FusionBlock(img_channels=64,gene_channels=in_gene_channels,fused_channels=64)

        # self.downsample = nn.AvgPool3d(2, stride=2)
        self.downsample = nn.AvgPool2d(2,stride=2)

    def forward(self, x_in,g_in):

        x_1 = self.conv_1(x_in)

        x_1 = x_1.squeeze(dim=2)
        x = self.downsample(x_1)
        x = x.unsqueeze(dim=2)
        x_1 = x_1.unsqueeze(dim=2)


        x_2 = self.conv_2(x)

        x_2 = x_2.squeeze(dim=2)
        x = self.downsample(x_2)
        x = x.unsqueeze(dim=2)
        x_2 = x_2.unsqueeze(dim=2)


        x_3 = self.conv_3(x)

        x_3 = x_3.squeeze(dim=2)
        x = self.downsample(x_3)
        x = x.unsqueeze(dim=2)
        x_3 = x_3.unsqueeze(dim=2)

        x_4 = self.conv_4(x)

        # visualize_channels_in_batches(x_4.squeeze(dim=2))
        x_4 = self.fusion(x_4, g_in)

        x_4 = x_4.squeeze(dim=2)
        x = self.downsample(x_4)
        x = x.unsqueeze(dim=2)
        x_4 = x_4.unsqueeze(dim=2)
        x_5 = self.conv_5(x)



        return [x_1, x_2, x_3, x_4, x_5]


        # return [x_1, x_2, x_3, x_4, x_5]
    

class Trans_decoder(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 channel_num: int, 
                 use_checkpoint: bool = False):
        super().__init__()

        winsize = [16,16,16]
        self.conv_1 = Conv_block(in_channels*2+channel_num, channel_num, use_checkpoint)
        self.trans_2 = SwinTrans_stage_block(embed_dim=channel_num*2,
                                             num_layers=4,
                                             num_heads=channel_num//8,
                                             window_size=winsize,
                                             use_checkpoint=use_checkpoint)
        self.trans_3 = SwinTrans_stage_block(embed_dim=channel_num*4,
                                             num_layers=4,
                                             num_heads=channel_num//4,
                                             window_size=winsize,
                                             use_checkpoint=use_checkpoint)
        self.trans_4 = SwinTrans_stage_block(embed_dim=channel_num*8,
                                             num_layers=4,
                                             num_heads=channel_num//2,
                                             window_size=winsize,
                                             use_checkpoint=use_checkpoint)
        self.trans_5 = SwinTrans_stage_block(embed_dim=channel_num*16,
                                             num_layers=4,
                                             num_heads=channel_num,
                                             window_size=winsize,
                                             use_checkpoint=use_checkpoint)
        
        self.backdim_2 = nn.Conv3d(in_channels*4+channel_num*2, channel_num*2, kernel_size=1, stride=1, padding='same')
        self.backdim_3 = nn.Conv3d(in_channels*8+channel_num*4, channel_num*4, kernel_size=1, stride=1, padding='same')
        self.backdim_4 = nn.Conv3d(in_channels*16+channel_num*8, channel_num*8, kernel_size=1, stride=1, padding='same')
        self.backdim_5 = nn.Conv3d(in_channels*32, channel_num*16, kernel_size=1, stride=1, padding='same')
        
        self.upsample_1 = PatchExpanding_block(embed_dim=channel_num*2)
        self.upsample_2 = PatchExpanding_block(embed_dim=channel_num*4)
        self.upsample_3 = PatchExpanding_block(embed_dim=channel_num*8)
        self.upsample_4 = PatchExpanding_block(embed_dim=channel_num*16)
        
        self.reghead_1 = DeformHead_block(channel_num, use_checkpoint)
        self.reghead_2 = DeformHead_block(channel_num*2, use_checkpoint)
        self.reghead_3 = DeformHead_block(channel_num*4, use_checkpoint)
        self.reghead_4 = DeformHead_block(channel_num*8, use_checkpoint)
        self.reghead_5 = AffineHead_block(channel_num*16)
        
        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, x_fix, x_mov):
        
        x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5 = x_fix
        x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5 = x_mov
        
        # Step 1
        x = torch.cat([x_fix_5, x_mov_5], dim=1)
        x = self.backdim_5(x)
        x_5 = self.trans_5(x)
        flow_5, affine_para = self.reghead_5(x_5)
        
        # Step 2
        flow_5_up = self.ResizeTransformer(flow_5)

        # print('flow5up')
        # print(flow_5_up.shape)
        # visualize_channels_in_batches(flow_5_up[:,:,0,:,:])
        x_mov_4 = self.SpatialTransformer(x_mov_4, flow_5_up)

        # visualize_channels_in_batches(x_mov_4[:,:,0,:,:])


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

class AffineTransformer_block(nn.Module):
    
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, affine_para):
        
        self.id = torch.zeros((1, 3, 4)).to(src.device)
        self.id[0, 0, 0] = 1
        self.id[0, 1, 1] = 1
        self.id[0, 2, 2] = 1

        affine_matrix = affine_para.reshape(1, 3, 4) + self.id

        affine_grid = nnf.affine_grid(affine_matrix, src.shape, align_corners=False)

        return nnf.grid_sample(src, affine_grid, align_corners=False, mode=self.mode)


class SpatialTransformer_block(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):

        src = src.squeeze(dim=2)
        flow = flow.squeeze(dim=2)
        flow = flow[:,1:3,:,:]

        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]

        grids = torch.meshgrid(vectors, indexing='ij')

        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1,0]]

        out_2d = nnf.grid_sample(src, new_locs, align_corners=False, mode=self.mode)
        out_3d = out_2d.unsqueeze(dim=2)

        return out_3d
    
    
class ResizeTransformer_block(nn.Module):

    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=False, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = x.squeeze(dim=2)
            # x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = nnf.interpolate(x, align_corners=False, scale_factor=self.factor, mode='bilinear')
            x = x.unsqueeze(dim=2)

        return x


class BoundaryExtractor(nn.Module):
    """
    A small module that tries to highlight boundary-like features
    from a given feature map. Could be trainable or fixed (Sobel-like).
    """

    def __init__(self, channels):
        super().__init__()
        # Example: 1x1 conv to reduce dimension + conv that simulates derivative
        self.reduce = nn.Conv3d(channels, channels // 2, kernel_size=1, padding=0)
        self.sobel_like = nn.Conv3d(channels // 2, 1, kernel_size=3, padding=1, bias=False)

        # Optionally initialize sobel_like weights as actual 3D derivatives or keep them learned
        # with torch.no_grad():
        #     self.sobel_like.weight = <some derivative kernel initialization>

    def forward(self, x):
        x = self.reduce(x)
        edges = self.sobel_like(x)
        return edges

class Conv_block(nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 out_channels:int , 
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.Conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_1 = nn.InstanceNorm3d(out_channels)
        
        self.Conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_2 = nn.InstanceNorm3d(out_channels)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.boundary_extractor = BoundaryExtractor(out_channels)
        
    def Conv_forward(self, x_in):

        x = self.Conv_1(x_in)
        x = self.LeakyReLU(x)
        x = self.norm_1(x)
        
        x = self.Conv_2(x)
        x = self.LeakyReLU(x)
        x_out = self.norm_2(x)

        return x_out
    
    def forward(self, x_in,  return_boundaries=False):
        
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Conv_forward, x_in)
        else:
            x_out = self.Conv_forward(x_in)

        if return_boundaries:
            boundary_map = self.boundary_extractor(x_out)
            return x_out, boundary_map
        else:
            return x_out
    
class DeformHead_block(nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.reg_head = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1, padding='same')
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))
    
    def forward(self, x_in):
        
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.reg_head, x_in)
        else:
            x_out = self.reg_head(x_in)
        
        return x_out
    
    
class AffineHead_block(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        
        self.affine_head_1 = nn.Linear(in_channels, in_channels//2, bias=False)
        self.affine_head_2 = nn.Linear(in_channels//2, 12, bias=False)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        
    def forward(self, x_in):
        
        x = torch.mean(x_in, dim=(2,3,4))
        x = self.affine_head_1(x)
        x = self.ReLU(x)
        x = self.affine_head_2(x)
        affine_para = self.Tanh(x)
        self.id = torch.zeros((1, 3, 4)).to(x_in.device)
        self.id[0, 0, 0] = 1
        self.id[0, 1, 1] = 1
        self.id[0, 2, 2] = 1
        affine_matrix = affine_para.reshape(1, 3, 4) + self.id
        affine_grid = nnf.affine_grid(affine_matrix, x_in.shape, align_corners=False)
        
        shape = x_in.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(x_in.device)

        affine_grid = affine_grid[..., [2,1,0]]
        affine_grid = affine_grid.permute(0, 4, 1, 2, 3)
        for i in range(len(shape)):
            affine_grid[:, i, ...] = (affine_grid[:, i, ...]/2.0 + 0.5)*(shape[i]-1)
        flow = affine_grid - grid
        
        return flow, affine_para
    
    
class PatchExpanding_block(nn.Module):
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        self.up_conv = nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(embed_dim//2)

    def forward(self, x_in):

        x_in = x_in.squeeze(dim=2)

        x = self.up_conv(x_in)
        x = x.unsqueeze(dim=2)

        x = einops.rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')
        
        return x_out
    

class SwinTrans_stage_block(nn.Module):

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
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = SwinTrans_Block(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    window_size=self.window_size,
                                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop,
                                    attn_drop=attn_drop,
                                    use_checkpoint=use_checkpoint)
            self.blocks.append(block)
        
    def forward(self, x_in):
        
        b, c, d, h, w = x_in.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x_in.device)
        
        x = einops.rearrange(x_in, 'b c d h w -> b d h w c')
        for block in self.blocks:
            x = block(x, mask_matrix=attn_mask)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')

        return x_out
    

class SwinTrans_Block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 shift_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()
                         
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint
                         
        self.norm1 = nn.LayerNorm(embed_dim)  
        self.attn = MSA_block(embed_dim,
                              window_size=window_size,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP_block(hidden_size=embed_dim, 
                             mlp_dim=int(embed_dim * mlp_ratio), 
                             dropout_rate=drop)

    def forward_part1(self, x_in, mask_matrix):
        
        x = self.norm1(x_in)
        
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
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

    def forward_part2(self, x_in):
        
        x = self.norm2(x_in)
        x_out = self.mlp(x)
        return x_out

    def forward(self, x_in, mask_matrix=None):
        
        if self.use_checkpoint and x_in.requires_grad:
            x = x_in + checkpoint.checkpoint(self.forward_part1, x_in, mask_matrix)
        else:
            x = x_in + self.forward_part1(x_in, mask_matrix)
                         
        if self.use_checkpoint and x.requires_grad:
            x_out = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x_out = x + self.forward_part2(x)
        
        return x_out


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
        warped_x = x - dx
        warped_y = y - dy
        warped_coords = torch.stack([warped_x, warped_y], dim=-1)

        return warped_coords

class CoordinateAffineWarper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points_in, affine_para):
        """
        points_in:   (B, N, 2) 2D points in the *input* image's pixel space
                     (x_in in [0..W_in-1], y_in in [0..H_in-1]).
        affine_para: (B, 3, 4) or (1, 3, 4) offsets from identity,
                     forming (out->in) matrix, same as used in grid_sample.
        src_shape:   (B, C, H_in, W_in) so we know the input dims.

        Returns:
            points_out: (B, N, 2) the points in output pixel space
                        (x_out in [0..W_out-1], y_out in [0..H_out-1]).
                        Usually W_out= W_in, H_out= H_in if the warp
                        doesn't change resolution. But the user can vary that.
        """
        device = points_in.device
        B, N, _ = points_in.shape
        H_in = 1024
        W_in = 1024  # input image shape

        # Build 3x4 identity for offset
        id_mat = torch.zeros((1,3,4), device=device)
        id_mat[0,0,0] = 1
        id_mat[0,1,1] = 1
        id_mat[0,2,2] = 1

        affine_matrix = affine_para.reshape(1, 3, 4)
        affine_matrix_out2in = affine_matrix + id_mat

        # Invert it => (in->out)
        affine_matrix_in2out = self.invert_3x4(affine_matrix_out2in)

        # We assume the output has the same (H_out, W_out) as input,
        # but if it differs, put that here
        H_out, W_out = H_in, W_in

        # Now for each point_in (in pixel coords),
        # 1) Convert to normalized coords in [-1,+1], using align_corners=True
        x_in = points_in[..., 0]  # (B,N)
        y_in = points_in[..., 1]  # (B,N)

        # align_corners=True => pixel 0 -> -1, pixel (W_in-1) -> +1
        x_in_norm = 2.0 * x_in / (W_in - 1) - 1.0
        y_in_norm = 2.0 * y_in / (H_in - 1) - 1.0

        # Hom coords: (x_in_norm, y_in_norm, 0, 1)
        ones  = torch.ones((B, N, 1), device=device)
        zeros = torch.zeros((B, N, 1), device=device)
        hom_in_norm = torch.cat([x_in_norm.unsqueeze(-1),
                                 y_in_norm.unsqueeze(-1),
                                 zeros,
                                 ones], dim=2)  # (B,N,4)

        # Multiply by (in->out) in normalized coords:
        # (B,N,4) x (B,4,3) => (B,N,3)
        hom_out_norm_3 = torch.bmm(hom_in_norm, affine_matrix_in2out.transpose(1,2))
        x_out_norm = hom_out_norm_3[...,0]
        y_out_norm = hom_out_norm_3[...,1]
        # z_out_norm = hom_out_norm_3[...,2], usually near 0

        # Convert from normalized coords back to pixel coords in output
        # using align_corners=True
        x_out = (x_out_norm + 1.0) * (W_out - 1) / 2.0
        y_out = (y_out_norm + 1.0) * (H_out - 1) / 2.0

        points_out = torch.stack([x_out, y_out], dim=-1)  # (B,N,2)
        return points_out

    def invert_3x4(self, mat_3x4):
        """
        Invert a batch of 3×4 matrices for 2D transforms.
        mat_3x4 is out->in. We want in->out.
        top-left 3×3 is rotation/scale/shear, last col is translation.
        """
        # shape: (B,3,4)
        R = mat_3x4[:,:, :3]  # (B,3,3)
        t = mat_3x4[:,:, 3]   # (B,3)
        R_inv = torch.inverse(R)
        # new translation = -R_inv * t
        t_inv = -torch.bmm(R_inv, t.unsqueeze(2)).squeeze(2)  # (B,3)
        mat_inv = torch.cat([R_inv, t_inv.unsqueeze(2)], dim=2)  # (B,3,4)
        return mat_inv

class MSA_block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) * 
                                                                     (2 * self.window_size[1] - 1) * 
                                                                     (2 * self.window_size[2] - 1), num_heads))
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x_in, mask=None):
        
        b, n, c = x_in.shape
        qkv = self.qkv(x_in).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.Softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)
        
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

def compute_mask(dims, window_size, shift_size, device):
    
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def window_partition(x_in, window_size):

    b, d, h, w, c = x_in.shape
    x = x_in.view(b,
                  d // window_size[0],
                  window_size[0],
                  h // window_size[1],
                  window_size[1],
                  w // window_size[2],
                  window_size[2],
                  c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        
    return windows


def window_reverse(windows, window_size, dims):

    b, d, h, w = dims
    x = windows.view(b,
                     d // window_size[0],
                     h // window_size[1],
                     w // window_size[2],
                     window_size[0],
                     window_size[1],
                     window_size[2],
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    return x


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
    
