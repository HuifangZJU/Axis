from axis.settings import data_path, experiment_path
from PIL import Image
import os
import torch
from argparse import ArgumentParser
from axis.registration.data import Sennet_image_pair
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from axis.registration import NICE_Trans_img
from axis.registration import CoordinateWarper,CoordinateAffineWarper
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch
import scanpy as sc

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


# def find_inverse_warp_coords(uv_coords, flow, upsample_factor=2):
#     """
#     Args:
#         uv_coords: [N, 2] coordinates (x, y) in the warped image
#         flow: [2, H, W] forward flow (dy, dx)
#         upsample_factor: int, how much to upsample flow and grid
#
#     Returns:
#         approx_orig_coords: [N, 2] — approximate original positions
#     """
#     device = flow.device
#     C, H, W = flow.shape
#     assert C == 2, "Flow must have 2 channels [dy, dx]"
#
#     # 1. Upsample flow
#     flow_upsampled = nnf.interpolate(flow.unsqueeze(0), scale_factor=upsample_factor, mode='bilinear', align_corners=True)[0]  # [2, H_up, W_up]
#     H_up, W_up = flow_upsampled.shape[1], flow_upsampled.shape[2]
#
#     # 2. Build denser meshgrid
#     y, x = torch.meshgrid(torch.arange(H_up, device=device), torch.arange(W_up, device=device), indexing='ij')
#     base_grid = torch.stack([x, y], dim=-1).float()  # [H_up, W_up, 2]
#
#     # 3. Apply flow
#     dy_up, dx_up = flow_upsampled[0], flow_upsampled[1]
#     warped_grid = base_grid + torch.stack([dx_up, dy_up], dim=-1)  # [H_up, W_up, 2]
#
#     # 4. Flatten
#     warped_flat = warped_grid.reshape(-1, 2)  # [H_up * W_up, 2]
#     base_flat = base_grid.reshape(-1, 2)      # [H_up * W_up, 2]
#
#     # 5. Nearest neighbor search
#     uv_coords = uv_coords*upsample_factor
#     dists = torch.cdist(uv_coords.unsqueeze(0), warped_flat.unsqueeze(0))  # [1, N, M]
#     nn_indices = dists.argmin(dim=-1).squeeze(0)  # [N]
#
#     # 6. Get original (pre-warp) coords from base grid
#     approx_orig_coords = base_flat[nn_indices]/upsample_factor
#
#     return approx_orig_coords


def create_overlay(img1,img2):
    img1_rgb = np.stack([img1, np.zeros_like(img1), img1], axis=-1)
    # Image 2 → Cyan (G + B)
    img2_rgb = np.stack([np.zeros_like(img2), img2, img2], axis=-1)

    # Blend with maximum pixel value
    overlay = np.maximum(img1_rgb, img2_rgb)
    return overlay
def min_max_normalize(array):
    min = np.min(array)   # shape (C,)
    max = np.max(array)   # shape (C,)
    range = max - min

    normalized_data = (array - min) / range
    return normalized_data
def tensor2img(tensor):
    tensor = tensor.detach().cpu().numpy().squeeze()
    tensor = min_max_normalize(tensor)
    return tensor
def visualize_flow(flow, image=None,step=30, title="Flow field",out_dir=None):
    """
    Visualizes a 2D flow field using quiver plot.
    flow: [2, H, W] or [1, 2, H, W]
    """
    if flow.ndim == 4:
        flow = flow[0]  # remove batch

    dy, dx = flow[0], flow[1]
    H, W = dy.shape

    y, x = torch.meshgrid(torch.arange(0, H, step), torch.arange(0, W, step), indexing='ij')
    y = y.cpu()
    x = x.cpu()

    u = dx[::step, ::step].cpu()
    v = dy[::step, ::step].cpu()

    plt.figure(figsize=(16, 10))
    if image:
        plt.imshow(image)
    plt.quiver(x, y, -u, -v, angles='xy', scale_units='xy', scale=1, color='red', width=0.002)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.axis('equal')
    if out_dir:
        plt.savefig(out_dir, dpi=300)
    else:
        plt.show()



def resize_flow(flow, target_h, target_w):
    """
    Resize flow from [2, H, W] to [2, target_h, target_w] with proper value scaling.
    """
    H, W = flow.shape[1:]
    scale_h = target_h / H
    scale_w = target_w / W

    flow = flow.unsqueeze(0)  # [1, 2, H, W]
    flow_resized = F.interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=True)
    flow_resized = flow_resized.squeeze(0)

    # Scale flow values accordingly
    flow_resized[0] *= scale_w  # x-direction
    flow_resized[1] *= scale_h  # y-direction

    return flow_resized

def rescale_coordinates(coords, original_size, new_size):
    """
    Rescale 2D coordinates from original image size to new image size.

    coords: numpy array of shape [N, 2], where coords[:, 0] is x and coords[:, 1] is y
    original_size: (H, W)
    new_size: (H', W')
    """
    orig_h, orig_w = original_size
    new_h, new_w = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    coords_rescaled = coords.copy()
    coords_rescaled[:, 0] *= scale_x  # x-coordinate (width direction)
    coords_rescaled[:, 1] *= scale_y  # y-coordinate (height direction)

    return coords_rescaled

def save_single_dataset(image, gene_data, datatype, sampleid, regionid, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    # Create base name
    base_name = f"{datatype}_{sampleid}_{regionid}"
    # Save AnnData object
    gene_data.write(os.path.join(output_dir, f"{base_name}_registered.h5ad"))
    # Save aligned image
    img_uint8 = (image * 255).astype('uint8') if image.max() <= 1.0 else image
    img_pil = Image.fromarray(img_uint8)

    # Save using PIL
    img_pil.save(os.path.join(output_dir, f"{base_name}_registered.png"))
    #cv2.imwrite(os.path.join(output_dir, f"{base_name}_registered.png"), img_uint8)
    print(f"Saved: {base_name} to {output_dir}")

parser = ArgumentParser()


parser.add_argument("--load_model", type=str,
                    dest="load_model", default=experiment_path('sennet/sennet_enhanced_image_1st_regu_loss_large_image/saved_models/model_950.pth'),
                    help="load model file to initialize with")
args = parser.parse_args()


img_transforms = transforms.Compose([transforms.Resize((1024,2048)),
                 transforms.ToTensor(),
                  transforms.Normalize(mean=(0.5,), std=(0.5,))])


test_dataloader = DataLoader(Sennet_image_pair(datalist=data_path("sennet/xenium_codex_pairs.txt"),image_transformer=img_transforms),batch_size=1, shuffle=False, num_workers=16)
# device handling
cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(0)
if cuda:
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
    torch.backends.cudnn.deterministic = True
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = 'cpu'

# prepare the model
model = NICE_Trans_img(in_channels=1)
checkpoint = torch.load(args.load_model, map_location="cpu")
model.load_state_dict(checkpoint)
model.to("cuda")
model.eval()


# transfer model
SpatialTransformer = model.SpatialTransformer
SpatialTransformer.to(device)
SpatialTransformer.eval()

SpatialTransformer_Coor = CoordinateWarper()
SpatialTransformer_Coor.to(device)
SpatialTransformer_Coor.eval()

AffineTransformer = model.AffineTransformer
AffineTransformer.to(device)
AffineTransformer.eval()

AffineTransformer_Coor = CoordinateAffineWarper()
AffineTransformer_Coor.to(device)
AffineTransformer_Coor.eval()


file_path = data_path('sennet/xenium_codex_pairs.txt')
file = open(file_path)
sennet_pairs =file.readlines()
for i, batch in enumerate(test_dataloader):
    print(i)
    line = sennet_pairs[i]
    xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = line.rstrip().split(' ')
    with torch.no_grad():
        inputs = batch
        fixed_image = inputs['fixed_image'].to(device).float()
        moving_image = inputs['moving_image'].to(device).float()

        fixed_image = fixed_image.unsqueeze(dim=2)
        moving_image = moving_image.unsqueeze(dim=2)

        warped_image, flow, affined_image, affine_para = \
            model(fixed_image, moving_image)


        fixed_image = tensor2img(fixed_image)
        moving_image = tensor2img(moving_image)
        warped_image = tensor2img(warped_image)
        flow = flow.squeeze().detach().cpu()
        flow = flow[1:,:,:]
        # visualize_flow(flow)


        xenium_image = plt.imread(data_path('sennet/hf_aligned_data/xenium')+f"_{xenium_sampleid}_{xenium_regionid}.png")
        xenium_gene_data = sc.read_h5ad(data_path("sennet/hf_aligned_data/xenium")+ f"_{xenium_sampleid}_{xenium_regionid}.h5ad")
        codex_gene_data = sc.read_h5ad(
            data_path("sennet/hf_aligned_data/codex") + f"_{codex_sampleid}_{codex_regionid}.h5ad")

        xenium_coordinate = np.stack([
            xenium_gene_data.obs['x_trans'].values,
            xenium_gene_data.obs['y_trans'].values
        ], axis=1)
        codex_coordinate = np.stack([
            codex_gene_data.obs['x_trans'].values,
            codex_gene_data.obs['y_trans'].values
        ], axis=1)


        # flow = resize_flow(flow, xenium_image.shape[0], xenium_image.shape[1])
        xenium_coordinate = rescale_coordinates(xenium_coordinate,xenium_image.shape,(1024,2048))
        codex_coordinate = rescale_coordinates(codex_coordinate,xenium_image.shape,(1024,2048))
        warped_codex_coor = find_inverse_warp_coords(torch.from_numpy(codex_coordinate).float(), flow)

        xenium_gene_data.obs['x_aligned'] = xenium_coordinate[:, 0]
        xenium_gene_data.obs['y_aligned'] = xenium_coordinate[:, 1]
        codex_gene_data.obs['x_aligned'] = warped_codex_coor[:, 0]
        codex_gene_data.obs['y_aligned'] = warped_codex_coor[:, 1]

        save_single_dataset(fixed_image, xenium_gene_data, 'xenium', xenium_sampleid, xenium_regionid, data_path('sennet/registered_data/'))
        save_single_dataset(warped_image, codex_gene_data, 'codex', codex_sampleid, codex_regionid,
                            data_path('sennet/registered_data/'))
        visualize_flow(flow,out_dir=data_path('sennet/registered_data/pair')+str(i)+'_flow.png')

        



        # f,a = plt.subplots(2,2)
        # a[0,0].scatter(xenium_coordinate[:,0],xenium_coordinate[:,1])
        # a[0,0].scatter(codex_coordinate[:, 0], codex_coordinate[:, 1])
        # a[0,1].scatter(xenium_coordinate[:, 0], xenium_coordinate[:, 1])
        # a[0,1].scatter(warped_codex_coor[:, 0], warped_codex_coor[:, 1])
        # a[1,0].imshow(create_overlay(fixed_image,moving_image))
        # a[1,1].imshow(create_overlay(fixed_image,warped_image))
        # plt.show()


        # visualize_flow(flow)

        # f,a = plt.subplots(2,3)
        # a[0,0].imshow(fixed_image)
        # a[0,1].imshow(moving_image)
        # a[0,2].imshow(warped_image)
        # a[1,0].imshow(create_overlay(fixed_image,moving_image))
        # a[1,1].imshow(create_overlay(fixed_image,warped_image))
        #
        # # plt.imshow(warped_image)
        # plt.show()


        # fixed_coor = val_set['fixed_coor'].float()/2
        # moving_coor = val_set['moving_coor'].float()/2
        #
        #
        #
        #
        # flow = flow.squeeze(dim=2)
        # # flow_np = tensor2img(flow)
        # # f,a = plt.subplots(1,3)
        # # a[0].imshow(flow_np[:, :, 0])
        # # a[1].imshow(flow_np[:, :, 1])
        # # a[2].imshow(flow_np[:, :, 2])
        # # plt.show()
        #
        #
        # flow = flow.squeeze().detach().cpu()
        #
        #
        # warped_coor = find_inverse_warp_coords(moving_coor.squeeze(0), flow[1:, :, :])
        #
        # fixed_coor = fixed_coor.squeeze().numpy()
        # warped_coor= warped_coor.squeeze().numpy()
        # moving_coor = moving_coor.squeeze().numpy()
        # visualize_flow(flow[1:, :, :], moving_image)
        #
        # fixed_label = val_set['fixed_label'].numpy()
        # moving_label = val_set['moving_label'].numpy()
        #
        # #
        # # np.savez("$AXIS_DATA_DIR/result/nicetrans/" + prefix[i] + "_result", pts1=fixed_coor,
        # #          pts2=warped_coor, img1=fixed_image, img2=warped_image, label1=fixed_label, label2=moving_label)
        #
        # f,a = plt.subplots(2,3,figsize=(12,8))
        # # plt.imshow(image_ovelayed)
        # a[0,0].imshow(0.5*fixed_image + 0.5*moving_image)
        # a[0,1].imshow(moving_image)
        # a[0, 1].scatter(moving_coor[:, 0], moving_coor[:, 1],c=moving_label, cmap='tab10', s=20)
        # a[0,2].imshow(warped_image)
        # a[0, 2].scatter(warped_coor[:, 0], warped_coor[:, 1],c=moving_label, cmap='tab10', s=20)
        #
        #
        # a[1,0].imshow(0.5*fixed_image + 0.5* warped_image)
        # a[1,1].scatter(fixed_coor[:, 0], fixed_coor[:, 1],c=fixed_label, cmap='tab10', s=20)
        # a[1,1].scatter(moving_coor[:, 0], moving_coor[:, 1],c=moving_label, cmap='tab10', s=20)
        # a[1,1].invert_yaxis()
        # a[1,1].axis('equal')
        #
        # a[1,2].scatter(fixed_coor[:,0],fixed_coor[:,1],c=fixed_label, cmap='tab10', s=20)
        # a[1,2].scatter(warped_coor[:,0],warped_coor[:,1],c=moving_label, cmap='tab10', s=20)
        # a[1,2].invert_yaxis()
        # a[1,2].axis('equal')
        # plt.show()
        #





