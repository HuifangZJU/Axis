from axis.settings import data_path, experiment_path
import os
import torch
from argparse import ArgumentParser
from axis.registration.data import Image_svg_pair
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from axis.registration import NICE_Trans_img
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as nnf
from multiprocessing import Pool, cpu_count
from axis.registration import find_inverse_warp_coords




from sklearn.neighbors import NearestNeighbors
def spotwise_topk_accuracy(coords_A, coords_B, labels_A, labels_B, dist_thresh=5.0):
    offset = 5


    # normalize pts1 to [0, 10]
    min_vals = coords_A.min(axis=0)
    max_vals = coords_B.max(axis=0)
    scale = max_vals - min_vals

    coords_A = (coords_A - min_vals) / scale * 100 + offset
    coords_B = (coords_B - min_vals) / scale * 100 + offset

    nn = NearestNeighbors(radius=dist_thresh, algorithm="auto").fit(coords_A)
    neighbors = nn.radius_neighbors(coords_B, return_distance=False)

    nB = coords_B.shape[0]
    correct_majority, correct_any, count_valid = 0, 0, 0

    for i in range(nB):
        idxs = neighbors[i]
        if len(idxs) == 0:
            # no neighbors in threshold → skip this spot
            continue
        neigh_labels = labels_A[idxs]
        true_label = labels_B[i]

        # majority vote
        vals, counts = np.unique(neigh_labels, return_counts=True)
        maj_label = vals[np.argmax(counts)]
        if maj_label == true_label:
            correct_majority += 1

        # relaxed: if true label appears among neighbors
        if true_label in neigh_labels:
            correct_any += 1

        count_valid += 1

    acc_majority = correct_majority / count_valid if count_valid > 0 else np.nan
    return acc_majority


def min_max_normalize(array):
    min = np.min(array)   # shape (C,)
    max = np.max(array)   # shape (C,)
    range = max - min

    normalized_data = (array - min) / range
    return normalized_data


def tensor2img(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # --- Squeeze first (B) and third (D) dimensions ---
    tensor = np.squeeze(tensor)  # remove dims with size 1
    # If you only want to squeeze B and D explicitly:
    # tensor = tensor.squeeze(axis=(0, 2)) if tensor.ndim == 5 else tensor.squeeze()

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

def visualize_flow(flow, image,step=10, title="Flow field"):
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

    plt.figure(figsize=(8, 8))
    # plt.imshow(image)
    plt.quiver(x, y, -u, -v, angles='xy', scale_units='xy', scale=1, color='red')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.axis('equal')
    plt.show()


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


img_mode='L'

if img_mode == 'RGB':
    img_transforms = transforms.Compose([
        transforms.ToTensor(),                               # Convert to tensor (C, H, W)
        transforms.Normalize(mean=(0.5, 0.5, 0.5),            # Normalize 3 channels
                             std=(0.5, 0.5, 0.5))
    ])
    model = NICE_Trans_img(in_channels=3)

else:
    img_transforms =  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    # prepare the model
    model = NICE_Trans_img(in_channels=1)




#for base_size in [1024,2048]:
for base_size in [2048]:
    for group in [1,2]:

        #test_dataloader = DataLoader(Image_svg_pair(datalist=f"breast_imglist/test.txt",base_size=base_size,image_transformer=img_transforms,
        #                                    gene_transformer=None,mode='val'),batch_size=1, shuffle=False, num_workers=8)
        test_dataloader = DataLoader(Image_svg_pair(datalist=f"xenium_imglist/group{group}_test.txt",base_size=base_size,image_transformer=img_transforms,
                                            gene_transformer=None,mode='val'),batch_size=1, shuffle=False, num_workers=8)
        checkpoint = torch.load(experiment_path(f"image/xenium_group{group}_{base_size}/saved_models/model_1000.pth"), map_location="cpu")
        model.load_state_dict(checkpoint)
        model.to("cuda")
        model.eval()
        results=[]
        #pairs=[[0,1],[1,0]]
        pairs=[[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]]
        for i, batch in enumerate(test_dataloader):
            pair = pairs[i]
            with torch.no_grad():
                inputs,val_set = batch
                fixed_image = inputs['fixed_image'].to(device).float()
                moving_image = inputs['moving_image'].to(device).float()

                fixed_image = fixed_image.unsqueeze(dim=2)
                moving_image = moving_image.unsqueeze(dim=2)

                warped_image, flow, affined_image, affine_para = \
                    model(fixed_image, moving_image)
                fixed_image = tensor2img(fixed_image)
                moving_image = tensor2img(moving_image)
                warped_image = tensor2img(warped_image)

                fixed_coor = val_set['fixed_coor'].float()
                moving_coor = val_set['moving_coor'].float().squeeze(0)


                flow = flow.squeeze().detach().cpu()
                flow = flow[1:, :, :]

                warped_coor = find_inverse_warp_coords(moving_coor, flow)

                fixed_coor = fixed_coor.squeeze().numpy()
                warped_coor= warped_coor.squeeze().numpy()
                moving_coor = moving_coor.squeeze().numpy()
                # visualize_flow(flow[1:, :, :], moving_image)

                fixed_label = val_set['fixed_label'].squeeze().numpy()
                moving_label = val_set['moving_label'].squeeze().numpy()

                out_dir = data_path("xenium/result/mouse_brain/nicetrans/") + str(base_size) + "/"
                os.makedirs(out_dir, exist_ok=True)
                prefix = str(group) + "_" + str(pair[0]) + "_" + str(pair[1]) + "_result"
                np.savez(out_dir + prefix,
                          pts1=fixed_coor,
                          pts2=warped_coor, img1=fixed_image, img2=warped_image, label1=fixed_label, label2=moving_label)

                #
                f, a = plt.subplots(2, 3, figsize=(24, 16))
                # plt.imshow(image_ovelayed)
                a[0, 0].imshow(0.5 * fixed_image + 0.5 * moving_image)
                a[0, 1].imshow(moving_image)
                a[0, 1].scatter(moving_coor[:, 0], moving_coor[:, 1], c=moving_label, cmap='tab10', s=1)
                a[0, 2].imshow(warped_image)
                a[0, 2].scatter(warped_coor[:, 0], warped_coor[:, 1], c=moving_label, cmap='tab10', s=1)

                a[1, 0].imshow(0.5 * fixed_image + 0.5 * warped_image)
                a[1, 1].scatter(fixed_coor[:, 0], fixed_coor[:, 1], c=fixed_label, cmap='tab10', s=1)
                a[1, 1].scatter(moving_coor[:, 0], moving_coor[:, 1], c=moving_label, cmap='tab10', s=1)
                # a[1, 1].scatter(fixed_coor[:, 0], fixed_coor[:, 1],  s=10)
                # a[1, 1].scatter(moving_coor[:, 0], moving_coor[:, 1],  s=10)
                a[1, 1].invert_yaxis()
                a[1, 1].axis('equal')

                a[1, 2].scatter(fixed_coor[:, 0], fixed_coor[:, 1], c=fixed_label, cmap='tab10', s=1)
                a[1, 2].scatter(warped_coor[:, 0], warped_coor[:, 1], c=moving_label, cmap='tab10', s=1)
                # a[1, 2].scatter(fixed_coor[:, 0], fixed_coor[:, 1], s=10)
                # a[1, 2].scatter(warped_coor[:, 0], warped_coor[:, 1], s=10)
                a[1, 2].invert_yaxis()
                a[1, 2].axis('equal')
                #plt.show()
                plt.savefig(out_dir + prefix +".png", dpi=300, bbox_inches='tight')










