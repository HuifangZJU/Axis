import os
import torch
from argparse import ArgumentParser
from datasets import Image_svg_pair
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from hf_networks import NICE_Trans
from matplotlib import pyplot as plt
from metrics import evaluate_single_pair
import torch
import torch.nn.functional as nnf
from coordinate_utils import find_inverse_warp_coords



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



img_mode='L'

if img_mode == 'RGB':
    img_transforms = transforms.Compose([
        transforms.ToTensor(),                               # Convert to tensor (C, H, W)
        transforms.Normalize(mean=(0.5, 0.5, 0.5),            # Normalize 3 channels
                             std=(0.5, 0.5, 0.5))
    ])
    # prepare the model
    model = NICE_Trans(in_channels= 3,
                 in_gene_channels=10)

else:
    img_transforms =  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    # prepare the model
    model = NICE_Trans(in_channels= 1,
                 in_gene_channels=10)

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

pairs=[[0,1],[1,0]]
for base_size in [1024,2048]:
    for group in [1]:
        print(group)
        checkpoint = torch.load(f"/work/hm218/experiment/registration/combined/xenium_breast_group{group}_{base_size}_win31_21/saved_models/model_1000.pth", map_location="cpu")
        # prepare the model
        model.load_state_dict(checkpoint)
        model.to("cuda")
        model.eval()

        test_dataloader = DataLoader(Image_svg_pair(datalist=f"breast_imglist/test.txt",base_size=base_size,image_transformer=img_transforms,
                                            gene_transformer=gene_transforms,mode='val'),batch_size=1, shuffle=False, num_workers=8)



        for i, batch in enumerate(test_dataloader):
            print(i)
            pair = pairs[i]
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

                warped_image, warped_gene, flow, flow_gene, affined_image, affined_gene, affine_para = \
                    model(fixed_image, fixed_gene, moving_image, moving_gene)
                fixed_image = tensor2img(fixed_image)
                moving_image = tensor2img(moving_image)
                warped_image = tensor2img(warped_image)

                # visualize_flow(flow)

                fixed_coor = val_set['fixed_coor'].float()
                moving_coor = val_set['moving_coor'].float()




                flow = flow.squeeze(dim=2)
                # flow_np = tensor2img(flow)
                # f,a = plt.subplots(1,3)
                # a[0].imshow(flow_np[:, :, 0])
                # a[1].imshow(flow_np[:, :, 1])
                # a[2].imshow(flow_np[:, :, 2])
                # plt.show()


                flow = flow.squeeze().detach().cpu()


                warped_coor = find_inverse_warp_coords(moving_coor.squeeze(0), flow)

                fixed_coor = fixed_coor.squeeze().numpy()
                warped_coor= warped_coor.squeeze().numpy()
                moving_coor = moving_coor.squeeze().numpy()

                fixed_label = val_set['fixed_label'].squeeze().numpy()
                moving_label = val_set['moving_label'].squeeze().numpy()


                # # metric = evaluate_single_pair(fixed_image, warped_image, fixed_coor, warped_coor, fixed_label, moving_label)
                out_dir = "/hpc/group/jilab/huifang/code/registration/data/xenium/result/breast/ours/"+str(base_size)+"/"
                os.makedirs(out_dir,exist_ok=True)
                prefix = str(group) +"_"+ str(pair[0])+"_"+str(pair[1]) + "_result"

                np.savez(out_dir + prefix,
                         pts1=fixed_coor,
                         pts2=warped_coor,  img1=fixed_image,img2=warped_image, label1=fixed_label, label2=moving_label)

                #
                f,a = plt.subplots(2,3,figsize=(24,16))
                # plt.imshow(image_ovelayed)
                a[0,0].imshow(0.5*fixed_image + 0.5*moving_image)
                a[0,1].imshow(moving_image)
                a[0, 1].scatter(moving_coor[:, 0], moving_coor[:, 1],c=moving_label, cmap='tab10', s=5)
                a[0,2].imshow(warped_image)
                a[0, 2].scatter(warped_coor[:, 0], warped_coor[:, 1],c=moving_label, cmap='tab10', s=5)


                a[1,0].imshow( 0.5*fixed_image + 0.5*warped_image)
                a[1,1].scatter(fixed_coor[:, 0], fixed_coor[:, 1],c=fixed_label, cmap='tab10', s=5)
                a[1,1].scatter(moving_coor[:, 0], moving_coor[:, 1],c=moving_label, cmap='tab10', s=5)
                # a[1, 1].scatter(fixed_coor[:, 0], fixed_coor[:, 1],  s=10)
                # a[1, 1].scatter(moving_coor[:, 0], moving_coor[:, 1],  s=10)
                a[1,1].invert_yaxis()
                a[1,1].axis('equal')

                a[1,2].scatter(fixed_coor[:,0],fixed_coor[:,1],c=fixed_label, cmap='tab10', s=5)
                a[1,2].scatter(warped_coor[:,0],warped_coor[:,1],c=moving_label, cmap='tab10', s=5)
                # a[1, 2].scatter(fixed_coor[:, 0], fixed_coor[:, 1], s=10)
                # a[1, 2].scatter(warped_coor[:, 0], warped_coor[:, 1], s=10)
                a[1,2].invert_yaxis()
                a[1,2].axis('equal')
                #plt.show()

                plt.savefig(out_dir + prefix +".png", dpi=300, bbox_inches='tight')





