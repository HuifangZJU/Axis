import os
from axis.datasets import Image_pca_pair
from axis.networks import NICE_Trans
from axis.coordinate_utils import find_inverse_warp_coords
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F
import cv2


def min_max_normalize(array):
    min = np.min(array)   # shape (C,)
    max = np.max(array)   # shape (C,)
    range = max - min

    normalized_data = (array - min) / range
    return normalized_data
def tensor2img(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    tensor = np.squeeze(tensor)
    # --- Handle channel order ---
    if tensor.ndim == 3:
        # Likely (C, H, W)
        if tensor.shape[0] == 1:  # grayscale
            img = tensor[0]
        else:  # RGB or multi-channel
            img = np.transpose(tensor, (1, 2, 0))
    elif tensor.ndim == 2:
        # Already (H, W)
        img = tensor
    else:
        raise ValueError(f"Unexpected tensor shape after squeeze: {tensor.shape}")

    # --- Normalize/clip for safe visualization ---
    img = np.clip(img, 0, 1)
    return img



def get_image_transform(mode='RGB'):
    if mode=='RGB':

        img_transforms = transforms.Compose([
            transforms.ToTensor(),                               # Convert to tensor (C, H, W)
            transforms.Normalize(mean=(0.5, 0.5, 0.5),            # Normalize 3 channels
                                 std=(0.5, 0.5, 0.5))
        ])
    else:
        img_transforms =  transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,))
            ])
    return img_transforms





gene_transforms = transforms.Compose([transforms.ToTensor()])
cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(0)
if cuda:
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
    torch.backends.cudnn.deterministic = True
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = 'cpu'


def upsample_flow_to_highres(flow_lr, highres_size):
    """
    Upsample low-res flow to high resolution, scaling displacements.

    Args:
        flow_lr: torch.Tensor
            Low-res flow field from the network.
            Shape: [B, 3, 1, H_lr, W_lr] or [B, 3, H_lr, W_lr].
            Channel 1 = dy, channel 2 = dx.
        highres_size: tuple (H_hr, W_hr)

    Returns:
        flow_hr: torch.Tensor
            High-res flow with same layout as input (4D or 5D),
            but spatial size (H_hr, W_hr), dy/dx scaled to HR pixels.
    """
    if isinstance(flow_lr, np.ndarray):
        flow_lr = torch.from_numpy(flow_lr)

    if flow_lr.dim() == 5:
        B, C, _, H_lr, W_lr = flow_lr.shape
        flow_4d = flow_lr.squeeze(2)  # [B, C, H_lr, W_lr]
    elif flow_lr.dim() == 4:
        B, C, H_lr, W_lr = flow_lr.shape
        flow_4d = flow_lr
    else:
        raise ValueError("flow_lr must be 4D or 5D")

    device = flow_4d.device
    H_hr, W_hr = highres_size

    # Bilinear upsampling of flow
    flow_4d_up = F.interpolate(
        flow_4d,
        size=(H_hr, W_hr),
        mode="bilinear",
        align_corners=False,
    )

    # Scale displacements from low-res pixels to high-res pixels
    sy = (H_hr - 1) / max(H_lr - 1, 1)
    sx = (W_hr - 1) / max(W_lr - 1, 1)

    # Channel 1: dy, channel 2: dx
    flow_4d_up[:, 1, :, :] *= sy
    flow_4d_up[:, 2, :, :] *= sx

    if flow_lr.dim() == 5:
        flow_hr = flow_4d_up.unsqueeze(2)  # [B, C, 1, H_hr, W_hr]
    else:
        flow_hr = flow_4d_up               # [B, C, H_hr, W_hr]

    return flow_hr.to(device)

def warp_highres_image_with_lowres_flow_cpu(moving_hr,st, flow_lr):
    """
    Warp a high-resolution moving image using the low-resolution flow.
    Runs entirely on CPU to avoid GPU memory overload.

    Args:
        moving_hr: numpy or torch array (H, W, C) or (C, H, W)
        flow_lr:   torch tensor of low-res flow (from network, on GPU)
    """
    import torch.nn.functional as F

    # ---------------------------
    # 1) Prepare moving image on CPU
    # ---------------------------
    if isinstance(moving_hr, np.ndarray):
        moving_hr = torch.from_numpy(moving_hr)

    if moving_hr.dim() == 3 and moving_hr.shape[-1] in (1, 3):  # (H,W,C)
        moving_hr = moving_hr.permute(2, 0, 1)  # (C,H,W)

    moving_hr = moving_hr.float().cpu()  # force CPU float32

    # normalize if uint8-like
    # if moving_hr.max() > 1.5:
    #     moving_hr = moving_hr / 255.0

    C, H_hr, W_hr = moving_hr.shape
    moving_hr = moving_hr.unsqueeze(0).unsqueeze(2)  # [1,C,1,H,W]

    # ---------------------------
    # 2) Upsample flow on GPU first
    # ---------------------------
    flow_lr = flow_lr.float()
    if flow_lr.dim() == 5:
        flow_lr = flow_lr.squeeze(2)

    B, C_flow, H_lr, W_lr = flow_lr.shape


    # upsample flow to HR resolution on GPU (MUCH smaller than HR image)
    flow_hr_gpu = F.interpolate(
        flow_lr,
        size=(H_hr, W_hr),
        mode='bilinear',
        align_corners=False
    )

    # scale flow to HR pixel units
    sy = (H_hr - 1) / max(H_lr - 1, 1)
    sx = (W_hr - 1) / max(W_lr - 1, 1)

    flow_hr_gpu[:, 1, :, :] *= sy
    flow_hr_gpu[:, 2, :, :] *= sx


    # Move the HR flow back to CPU
    flow_hr = flow_hr_gpu.cpu().unsqueeze(2)  # [B,3,1,H,W]

    # ---------------------------
    # 3) Warp on CPU using your SpatialTransformer_block
    # ---------------------------
    st_cpu = st.cpu()

    with torch.no_grad():
        warped_hr = st_cpu(moving_hr, flow_hr)  # [1,C,1,H,W]

    warped_hr = warped_hr.squeeze().to(torch.uint8) # [C,H,W]
    warped_hr = warped_hr.detach().cpu()
    warped_hr = warped_hr.permute(1, 2, 0).numpy()
    warped_hr = cv2.cvtColor(warped_hr, cv2.COLOR_BGR2RGB)
    flow_hr = flow_hr[:, 1:3, ::, :]
    flow_hr = flow_hr.squeeze()
    return warped_hr, flow_hr


def generate_pairwise_outputs(ckpt_path,test_dataset,model,run_hires=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Initialize model

    model.load_state_dict(checkpoint)
    model.eval()
    SpatialTransformer = model.SpatialTransformer


    # DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,
    )

    results = {}
    with torch.no_grad():
        for inputs, val_set in test_dataloader:
            pair_obj = inputs["data_pair"]

            if isinstance(pair_obj, torch.Tensor):
                # numeric pair, e.g. tensor([0, 1]) or tensor([[0, 1]])
                pair_key = tuple(int(x) for x in pair_obj.view(-1).tolist())

            elif isinstance(pair_obj, (list, tuple)):
                # could be:
                #   [tensor(0), tensor(1)]
                #   ("protein", "visium")
                #   (("protein",), ("visium",))
                elems = []
                for x in pair_obj:
                    # Unwrap singleton tuples/lists like ('protein',) -> 'protein'
                    if isinstance(x, (list, tuple)) and len(x) == 1:
                        x = x[0]

                    if isinstance(x, torch.Tensor):
                        # scalar tensor
                        if x.numel() == 1:
                            elems.append(int(x.item()))
                        else:
                            # if you ever have multi-element tensors:
                            elems.extend(int(v) for v in x.view(-1).tolist())
                    else:
                        # assume this is already a usable identifier (e.g. "protein")
                        elems.append(x)

                pair_key = tuple(elems)

            else:
                # Fallback: single value (string or int)
                pair_key = (pair_obj,)


            fixed_image = inputs["fixed_image"].to(device).float().unsqueeze(2)
            fixed_gene = inputs["fixed_gene"].to(device).float().unsqueeze(2)
            moving_image = inputs["moving_image"].to(device).float().unsqueeze(2)
            moving_gene = inputs["moving_gene"].to(device).float().unsqueeze(2)

            (
                warped_image,
                warped_gene,
                flow_3d,
                flow_gene,
                affined_image,
                affined_gene,
                affine_para,
            ) = model(fixed_image, fixed_gene, moving_image, moving_gene)
            flow = flow_3d[:, 1:3, ::, :]

            # Keep coords as tensors (likely double) and let the helper cast them
            fixed_coor = val_set["fixed_coor"].squeeze(0)
            moving_coor = val_set["moving_coor"].squeeze(0)

            # If flow has a batch dim, remove it before passing
            flow_for_coords = flow.squeeze()

            warped_coor = find_inverse_warp_coords(moving_coor, flow_for_coords)


            # Convert everything to numpy for storage
            fixed_image_np = tensor2img(fixed_image.detach().cpu())
            moving_image_np = tensor2img(moving_image.detach().cpu())
            warped_image_np = tensor2img(warped_image.detach().cpu())

            fixed_coor_np = fixed_coor.detach().cpu().numpy()
            moving_coor_np = moving_coor.detach().cpu().numpy()
            warped_coor_np = warped_coor.detach().cpu().numpy()

            fixed_label = val_set["fixed_label"].squeeze().cpu().numpy()
            moving_label = val_set["moving_label"].squeeze().cpu().numpy()

            flow_np = flow_for_coords.detach().cpu().numpy()

            results[pair_key] = {
                "fixed_image": fixed_image_np,
                "moving_image": moving_image_np,
                "warped_image": warped_image_np,
                "fixed_coor": fixed_coor_np,
                "moving_coor": moving_coor_np,
                "warped_coor": warped_coor_np,
                "fixed_label": fixed_label,
                "moving_label": moving_label,
                "flow": flow_np,
            }
            if run_hires:
                highres_moving_image_np = cv2.imread(inputs['moving_hires_image'][0],cv2.IMREAD_COLOR)
                # print('warpping high resolution images')
                warped_hr, dense_flow_hr = warp_highres_image_with_lowres_flow_cpu(
                    moving_hr=highres_moving_image_np,
                    st=SpatialTransformer,
                    flow_lr=flow_3d
                )

                results[pair_key].update({
                    "warped_hires": warped_hr,
                    "flow_hires": dense_flow_hr
                })

    return results