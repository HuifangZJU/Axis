import numpy as np
import torch


def find_inverse_warp_coords(uv_coords, flow, batch_size=1024):
    """
    Approximate inverse warp: given moving-frame coordinates `uv_coords`
    and a forward flow `flow`, find their corresponding coordinates in
    the base frame.

    Behavior:
      - Always run computation on GPU (device='cuda').
      - Accept `uv_coords` and `flow` as numpy arrays or torch tensors.
      - If `uv_coords` was numpy, return numpy on CPU.
      - If `uv_coords` was a CPU tensor, return a CPU tensor.
      - If `uv_coords` was a CUDA tensor, return a CUDA tensor.

    Expected flow shape: (2, H, W) or (H, W, 2) (will be converted).
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but this function is set to always run on GPU.")

    # -----------------------------
    # Track original type/device
    # -----------------------------
    uv_was_numpy = isinstance(uv_coords, np.ndarray)
    flow_was_numpy = isinstance(flow, np.ndarray)

    if flow_was_numpy:
        flow = torch.from_numpy(flow)
    if uv_was_numpy:
        uv_coords = torch.from_numpy(uv_coords)

    if not torch.is_tensor(uv_coords) or not torch.is_tensor(flow):
        raise TypeError("uv_coords and flow must be numpy arrays or torch tensors.")

    orig_device = uv_coords.device  # cpu or cuda

    # -----------------------------
    # Move everything to GPU
    # -----------------------------
    device = torch.device("cuda")
    flow = flow.to(device=device, dtype=torch.float32)
    uv_coords = uv_coords.to(device=device, dtype=torch.float32)


    # -----------------------------
    # Normalize flow shape to (2, H, W)
    # -----------------------------
    if flow.ndim == 4 and flow.shape[0] == 1:
        # e.g. (1,2,H,W)
        flow = flow.squeeze(0)

    if flow.ndim != 3:
        raise ValueError(f"Expected flow with 3 dims, got {flow.shape}")

    if flow.shape[0] == 2:
        # (2,H,W) OK
        pass
    elif flow.shape[-1] == 2:
        # (H,W,2) -> (2,H,W)
        flow = flow.permute(2, 0, 1).contiguous()
    else:
        raise ValueError(f"Flow must have a channel dim of size 2, got {flow.shape}")

    dy, dx = flow[0], flow[1]
    H, W = dx.shape

    # -----------------------------
    # Build base & warped grids
    # -----------------------------
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    base_grid = torch.stack([x, y], dim=-1)             # [H, W, 2]
    warped_grid = base_grid + torch.stack([dx, dy], dim=-1)

    warped_flat = warped_grid.reshape(-1, 2)            # [H*W, 2]
    base_flat = base_grid.reshape(-1, 2)                # [H*W, 2]

    # -----------------------------
    # Nearest neighbor search
    # -----------------------------
    N = uv_coords.shape[0]
    nn_indices = []

    for start in range(0, N, batch_size):
        uv_batch = uv_coords[start:start + batch_size]  # [B, 2]

        # torch.cdist: (1, B, 2) vs (1, H*W, 2) -> (1, B, H*W)
        dists = torch.cdist(
            uv_batch.unsqueeze(0),
            warped_flat.unsqueeze(0),
        )  # [1, B, H*W]

        nn_batch = dists.argmin(dim=-1).squeeze(0)      # [B]
        nn_indices.append(nn_batch)

        del dists
        torch.cuda.empty_cache()

    nn_indices = torch.cat(nn_indices, dim=0)           # [N]
    approx_orig_coords = base_flat[nn_indices]          # [N, 2], float32 on cuda

    # -----------------------------
    # Convert back to original type/device
    # -----------------------------
    if uv_was_numpy:
        # caller gave numpy -> return numpy on CPU
        return approx_orig_coords.cpu().numpy()
    else:
        # caller gave torch tensor
        if orig_device.type == "cpu":
            return approx_orig_coords.cpu()
        else:
            # original was already on cuda, keep it on cuda
            return approx_orig_coords

