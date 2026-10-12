import os
import torch
from argparse import ArgumentParser
from datasets import Image_gene_pair
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from hf_networks import NICE_Trans,CoordinateWarper,CoordinateAffineWarper

from matplotlib import pyplot as plt

parser = ArgumentParser()

parser.add_argument("--model_dir", type=str,
                    dest="model_dir", default='DLPFC_all_pairwise_align_tissue_mask_finetuned',
                    help="models folder")
parser.add_argument("--load_model", type=str,
                    dest="load_model", default='/media/huifang/data/experiment/registration/DLPFC_all_pairwise_align_tissue_mask_finetuned/saved_models/model_200.pth',
                    help="load model file to initialize with")
args = parser.parse_args()

img_transforms = [transforms.ToTensor(),
                  transforms.Normalize((0.5), (0.5))]
gene_transforms = [transforms.ToTensor()]

test_dataloader = DataLoader(Image_gene_pair(datalist='DLPFC_pairwise_align_validation.txt',image_transformer=img_transforms,
                                            gene_transformer=gene_transforms,mode='val'),batch_size=1, shuffle=False, num_workers=16)

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
model = NICE_Trans()
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

prefix=['0_0','0_1','0_2','1_0','1_1','1_2','2_0','2_1','2_2']

for i, batch in enumerate(test_dataloader):
    with torch.no_grad():
        inputs, val_set = batch
        fixed_mask = inputs['fixed_mask'].to(device).float().unsqueeze(dim=2)
        fixed_gene = inputs['fixed_gene'].to(device).float().unsqueeze(dim=2)
        moving_mask = inputs['moving_mask'].to(device).float().unsqueeze(dim=2)
        moving_gene = inputs['moving_gene'].to(device).float().unsqueeze(dim=2)

        _, _, flow, _, _, _ = \
            model(fixed_mask, fixed_gene, moving_mask, moving_gene)


        moving_image = inputs['moving_image'].to(device).float().unsqueeze(dim=2)
        warped_image = SpatialTransformer(moving_image,flow)

        fixed_image = inputs['fixed_image']

        fixed_coor = val_set['fixed_coor'].squeeze()
        moving_coor = val_set['moving_coor'].to(device).float()
        flow = flow.squeeze(dim=2)
        warped_coor = SpatialTransformer_Coor(moving_coor, flow[:, 1:, :, :])


        fixed_image = fixed_image.squeeze().permute(1, 2, 0).numpy()
        warped_image = warped_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        fixed_coor = fixed_coor.squeeze().numpy()
        warped_coor= warped_coor.detach().cpu().squeeze().numpy()
        moving_coor = moving_coor.detach().cpu().squeeze().numpy()

        fixed_label = val_set['fixed_label'].numpy()
        moving_label = val_set['moving_label'].numpy()


        # np.savez("/home/huifang/workspace/code/registration/result/ours/" + prefix[i] + "_result", pts1=fixed_coor,
        #          pts2=warped_coor, img1=fixed_image, img2=warped_image, label1=fixed_label, label2=moving_label)

        f,a = plt.subplots(1,3)
        a[0].imshow(fixed_image)
        a[0].imshow(warped_image)
        a[1].scatter(fixed_coor[:, 0], fixed_coor[:, 1])
        a[1].scatter(moving_coor[:, 0], moving_coor[:, 1])
        a[1].invert_yaxis()

        a[2].scatter(fixed_coor[:,0],fixed_coor[:,1])
        a[2].scatter(warped_coor[:,0],warped_coor[:,1])
        a[2].invert_yaxis()
        plt.show()






