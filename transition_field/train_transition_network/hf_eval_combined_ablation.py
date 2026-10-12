import os
import torch
from argparse import ArgumentParser
from datasets import Image_svg_pair,scc_ablation_pair
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from hf_networks import NICE_Trans
from matplotlib import pyplot as plt
from metrics import evaluate_single_pair
import torch
import torch.nn.functional as nnf


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

def visualize_flow(flow, image=None,step=10, title="Flow field"):
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
    plt.quiver(x, y, -u, -v, angles='xy', scale_units='xy', scale=1, color='purple')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

def min_max_normalize(array):
    min = np.min(array)   # shape (C,)
    max = np.max(array)   # shape (C,)
    range = max - min

    normalized_data = (array - min) / range
    return normalized_data
def tensor2img(tensor):
    tensor = tensor.detach().cpu().numpy().squeeze()
    tensor = tensor-tensor.min()
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = min_max_normalize(tensor)
    return tensor

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




img_transforms = transforms.Compose([
    transforms.Resize((512,512)),                      # Resize image
    transforms.ToTensor(),                               # Convert to tensor (C, H, W)
    transforms.Normalize(mean=(0.5, 0.5, 0.5),            # Normalize 3 channels
                         std=(0.5, 0.5, 0.5))
])
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
 # prepare the model



model = NICE_Trans()

for patient in [10]:
    print(patient)
    for ablation_level in [1]:
        print(ablation_level)
        model_path = f"/work/hm218/experiment/registration/combined/scc_patient{patient}_ablation{ablation_level}_from_scratch_win_9_5/saved_models/model_1000.pth"
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        model.to("cuda")
        model.eval()

        test_dataloader = DataLoader(scc_ablation_pair(ablation_type='image', patient=patient, ablation_level=ablation_level,
                                                   image_transformer=img_transforms,
                                                   gene_transformer=gene_transforms, mode='val'), batch_size=1, shuffle=False,
                                 num_workers=8)

        results = []
        for i, batch in enumerate(test_dataloader):
            with torch.no_grad():
                inputs,val_set = batch
                fixed_image = inputs['fixed_image'].to(device).float()
                fixed_gene = inputs['fixed_gene'].to(device).float()

                moving_image = inputs['moving_image'].to(device).float()
                moving_gene =inputs['moving_gene'].to(device).float()
    
                fixed_image = fixed_image.unsqueeze(dim=2)
                fixed_gene = fixed_gene.unsqueeze(dim=2)
                moving_image = moving_image.unsqueeze(dim=2)
                moving_gene = moving_gene.unsqueeze(dim=2)

                warped_image, warped_gene, flow,flow_gene, affined_image, affined_gene, affine_para = \
                    model(fixed_image, fixed_gene, moving_image, moving_gene)
                fixed_image = tensor2img(fixed_image)
                moving_image = tensor2img(moving_image)
                warped_image = tensor2img(warped_image)
    
                # visualize_flow(flow)
    
                fixed_coor = val_set['fixed_coor'].float()
                moving_coor = val_set['moving_coor'].float()
    



                flow = flow.squeeze(dim=2)



                flow = flow.squeeze().detach().cpu()


                warped_coor = find_inverse_warp_coords(moving_coor.squeeze(0), flow)

                fixed_coor = fixed_coor.squeeze().numpy()
                warped_coor= warped_coor.squeeze().numpy()
                moving_coor = moving_coor.squeeze().numpy()

                fixed_label = val_set['fixed_label'].squeeze().numpy()
                moving_label = val_set['moving_label'].squeeze().numpy()

                topk_acc = spotwise_topk_accuracy(fixed_coor, warped_coor, fixed_label, moving_label)
                results.append(topk_acc)
        results.append(np.mean(np.asarray(results)))
        print(results)






