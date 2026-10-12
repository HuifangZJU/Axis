import torch
import torch.nn.functional as F

def find_inverse_warp_coords_local(uv_coords, flow, window=15, tau=0.1, eps=1e-8):
    device = flow.device
    H, W = flow.shape[1:]
    
    # Build base grid (x,y)
    y, x = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device),
                          indexing='ij')
    base_grid = torch.stack([x, y], dim=-1).float()  # [H, W, 2]


    warped_grid = base_grid + flow.permute(1, 2, 0)  # [H, W, 2]
    
    # Normalize uv_coords to [-1,1] for grid_sample
    uv_norm = uv_coords.clone()
    uv_norm[:, 0] = (uv_norm[:, 0] / (W-1)) * 2 - 1
    uv_norm[:, 1] = (uv_norm[:, 1] / (H-1)) * 2 - 1
    uv_norm = uv_norm.unsqueeze(0).unsqueeze(0)  # [1,1,N,2]
    
    # Extract warped patches [N, 2, (2w+1), (2w+1)]
    warped_grid_t = warped_grid.permute(2, 0, 1).unsqueeze(0)  # [1,2,H,W]
    patches = F.grid_sample(
        warped_grid_t, uv_norm, mode='bilinear',
        align_corners=True, padding_mode='border'
    )  # [1,2,1,N] (not yet neighborhood)
    
    # Now expand to local window (approximate)
    # Simpler/faster: directly compute dist to center uv only (not full patch)
    # If you want true window, need unfold() trick (see below)
    
    # Compute distances [N, H*W] is too big, so restrict with unfold
    unfolded_warped = F.unfold(warped_grid_t, kernel_size=2*window+1, padding=window)  # [1,2*(2w+1)^2, H*W]
    unfolded_base = F.unfold(base_grid.permute(2,0,1).unsqueeze(0), kernel_size=2*window+1, padding=window)
    
    # Sample local patches for uv indices
    uv_int = uv_coords.round().long()
    idxs = uv_int[:,1] * W + uv_int[:,0]  # linear index [N]
    
    warped_patches = unfolded_warped[0,:,idxs].reshape(2, -1, (2*window+1)**2).permute(1,2,0)  # [N,P,2]
    base_patches  = unfolded_base[0,:,idxs].reshape(2, -1, (2*window+1)**2).permute(1,2,0)   # [N,P,2]
    
    # Compute distances [N,P]
    dists = torch.norm(warped_patches - uv_coords[:,None,:], dim=-1)
    
    # Softmin
    weights = torch.softmax(-dists / (tau+eps), dim=-1)  # [N,P]
    approx_orig_coords = torch.sum(weights[:,:,None] * base_patches, dim=1)  # [N,2]
    
    return approx_orig_coords









def find_inverse_warp_coords(uv_coords, flow, batch_size=1024):
    device = flow.device
    dy, dx = flow[0], flow[1]
    H, W = dx.shape

    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    base_grid = torch.stack([x, y], dim=-1).float()
    warped_grid = base_grid + torch.stack([dx, dy], dim=-1)

    warped_flat = warped_grid.reshape(-1, 2)  # [H*W, 2]
    base_flat = base_grid.reshape(-1, 2)

    N = uv_coords.shape[0]
    nn_indices = []
    for start in range(0, N, batch_size):
        uv_batch = uv_coords[start:start+batch_size]
        dists = torch.cdist(uv_batch.unsqueeze(0), warped_flat.unsqueeze(0))  # [1, B, H*W]
        nn_batch = dists.argmin(dim=-1).squeeze(0)
        nn_indices.append(nn_batch)
        #del dists
        #torch.cuda.empty_cache()

    nn_indices = torch.cat(nn_indices, dim=0)
    approx_orig_coords = base_flat[nn_indices]
    return approx_orig_coords


def find_inverse_warp_coords_backup(uv_coords, flow):
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
