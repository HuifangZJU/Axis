import os
import sys
import time
import torch
from argparse import ArgumentParser
from datasets import Image_svg_pair,phenocycler_omics_pair
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from hf_networks import NICE_Trans
import losses
import datetime
from tensorboardX import SummaryWriter
import numpy as np
from coordinate_utils import find_inverse_warp_coords,find_inverse_warp_coords_local
from metrics import evaluate_multiple_pairs

def min_max_normalize(array):
    min = np.min(array)   # shape (C,)
    max = np.max(array)   # shape (C,)
    range = max - min
    normalized_data = (array - min) / range
    return normalized_data


def create_overlay(fixed, moving):

    fixed = min_max_normalize(fixed)
    moving = min_max_normalize(moving)
    overlay = 0.5*fixed +0.5* moving
    return overlay



def save_image_registrations(fixed,moving,affined,warpped,epoch,i,label):

    original = create_overlay(fixed,moving)
    after_affine = create_overlay(fixed,affined)
    final_result = create_overlay(fixed,warpped)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    axes[0].imshow(original,cmap='gray')
    axes[0].set_title("Original displacement")

    axes[1].imshow(after_affine,cmap='gray')
    axes[1].set_title("Overlay After Affine transformation (Magenta/Cyan)")

    axes[2].imshow(final_result,cmap='gray')
    axes[2].set_title("Overlay After Deformable Registration (Magenta/Cyan)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(image_save_path + str(epoch) + '_'+str(i)+'_'+label+'.png',dpi=300)
    plt.close()

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
    #tensor = tensor.detach().cpu().numpy().squeeze()
    #tensor = np.transpose(tensor, (1, 2, 0))
    #return tensor

def visual_validation(epoch):
    model.eval()
    for i, test_batch in enumerate(test_dataloader):
        with torch.no_grad():
            inputs,val_set = test_batch
            fixed_image = inputs['fixed_image'].to(device).float()
            fixed_gene = inputs['fixed_gene'].to(device).float()
            moving_image = inputs['moving_image'].to(device).float()
            moving_gene = inputs['moving_gene'].to(device).float()

            fixed_image = fixed_image.unsqueeze(dim=2)
            fixed_gene = fixed_gene.unsqueeze(dim=2)

            moving_image = moving_image.unsqueeze(dim=2)
            moving_gene = moving_gene.unsqueeze(dim=2)

            warped_image, warped_gene, flow,flow_gene, affined_image, affined_gene, affine_para = \
                model(fixed_image, fixed_gene, moving_image, moving_gene)

            fixed_image = tensor2img(fixed_image)
            moving_image = tensor2img(moving_image)
            warped_image = tensor2img(warped_image)
            affined_image = tensor2img(affined_image)

            save_image_registrations(fixed_image,moving_image,affined_image,warped_image,epoch,i,'image')

            fixed_gene = fixed_gene.detach().cpu().numpy().squeeze()
            moving_gene = moving_gene.detach().cpu().numpy().squeeze()
            warped_gene = warped_gene.detach().cpu().numpy().squeeze()
            affined_gene = affined_gene.detach().cpu().numpy().squeeze()

            save_image_registrations(fixed_gene[0, :, :], moving_gene[0, :, :], affined_gene[0, :, :],
                                     warped_gene[0, :, :], epoch, i,'gene_pca')
            #save_image_registrations(fixed_gene[0, :, :], moving_gene[0, :, :], affined_gene[0, :, :],
            #                         warped_gene[0, :, :], epoch, i,'gene_pca')


def metric_evaluation(test_dataloader,epoch):
    model.eval()
    #pairs = []
    for i, batch in enumerate(test_dataloader):
        with torch.no_grad():
            inputs, val_set = batch
            fixed_image = inputs['fixed_image'].to(device).float()
            fixed_gene = inputs['fixed_gene'].to(device).float()

            moving_image = inputs['moving_image'].to(device).float()
            moving_gene = inputs['moving_gene'].to(device).float()

            fixed_image = fixed_image.unsqueeze(dim=2)
            fixed_gene = fixed_gene.unsqueeze(dim=2)
            moving_image = moving_image.unsqueeze(dim=2)
            moving_gene = moving_gene.unsqueeze(dim=2)

            warped_image, warped_gene, flow,flow_gene, affined_image, affined_gene, affine_para = \
                model(fixed_image, fixed_gene, moving_image, moving_gene)
            fixed_image = tensor2img(fixed_image)
            warped_image = tensor2img(warped_image)
            moving_image = tensor2img(moving_image)


            fixed_coor = val_set['fixed_coor'].float()
            moving_coor = val_set['moving_coor'].float()

            flow = flow.squeeze(dim=2)
            flow = flow.squeeze().detach().cpu()
            warped_coor = find_inverse_warp_coords(moving_coor.squeeze(0), flow.squeeze())

            fixed_coor = fixed_coor.squeeze().numpy()
            warped_coor = warped_coor.squeeze().numpy()
            moving_coor = moving_coor.squeeze().numpy()

            fixed_label = val_set['fixed_label'].squeeze().numpy()
            moving_label = val_set['moving_label'].squeeze().numpy()


            f, a = plt.subplots(1, 2, figsize=(24, 12))
            a[0].imshow(warped_image)
            a[0].scatter(warped_coor[:, 0], warped_coor[:, 1], c=moving_label, cmap='tab10', s=0.5)
            a[1].scatter(fixed_coor[:, 0], fixed_coor[:, 1], c=fixed_label, cmap='tab10', s=0.5)
            a[1].scatter(warped_coor[:, 0], warped_coor[:, 1], c=moving_label, cmap='tab10', s=0.5)
            a[1].invert_yaxis()
            a[1].axis('equal')
            for ax in a:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(image_save_path + str(epoch) + '_' + str(i) + '_spatial.png', dpi=300)
            plt.close()


            #pairs.append((fixed_image, warped_image, fixed_coor, warped_coor, fixed_label, moving_label))
    #averaged_metrics = evaluate_multiple_pairs(pairs)
    #print("Averaged Metrics:")
    #print(averaged_metrics)

def train_model():
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5*1e-4)

    img_warpping_loss = losses.NCC(win=41).loss

    #gene_warpping_loss = losses.NCC(win=11).loss
    gene_warpping_loss = losses.MaskedNCC(win=31).loss



    prev_time = time.time()
    logger = SummaryWriter(log_save_path)
    # training/validate loops
    for epoch in range(args.initial_epoch, args.epochs + 1):
        model.train()
        for i, batch in enumerate(train_dataloader):
            fixed_image = batch['fixed_image'].to(device).float()
            fixed_gene = batch['fixed_gene'].to(device).float()

            moving_image = batch['moving_image'].to(device).float()
            moving_gene = batch['moving_gene'].to(device).float()

            fixed_image = fixed_image.unsqueeze(dim=2)
            fixed_gene = fixed_gene.unsqueeze(dim=2)

            moving_image = moving_image.unsqueeze(dim=2)
            moving_gene = moving_gene.unsqueeze(dim=2)

            warped_image, warped_gene, flow,flow_gene, affined_image, affined_gene, affine_para = \
                model(fixed_image, fixed_gene, moving_image, moving_gene)


            # calculate total loss
            deformable_image_loss = img_warpping_loss(fixed_image, warped_image)
            affine_image_loss = img_warpping_loss(fixed_image, affined_image)

            deformable_gene_loss = gene_warpping_loss(fixed_gene, warped_gene)
            affine_gene_loss = gene_warpping_loss(fixed_gene, affined_gene)

            field_loss = 1 * displacement_loss(flow)  
            image_loss = 1 * (deformable_image_loss + affine_image_loss)
            gene_loss =  1  * (deformable_gene_loss + affine_gene_loss)

            loss = image_loss + gene_loss + field_loss
            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + i
            batches_left = args.epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()


            '''
            sys.stdout.write(
                "\r" + args.model_dir + "---[Epoch %d/%d] [Batch %d/%d] [Total loss: %f] [Image loss: %f] [Gene loss: %f] [Regu loss: %f] [Spot loss: %f] ETA: %s" %
                (epoch, args.epochs,
                 i, len(train_dataloader),
                 loss.item(), image_loss.item(), gene_loss.item(), field_loss.item(),spot_loss.item(), time_left))
            '''
            # # --------------tensor board--------------------------------#
            if batches_done % 20 == 0:
                info = {'loss': loss.item(), 'deformable_image_loss': deformable_image_loss.item(),
                         'deformable_gene_loss': deformable_gene_loss.item(), 'regu_loss':field_loss.item()}
                for tag, value in info.items():
                    logger.add_scalar(tag, value, batches_done)

                '''
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.add_histogram(tag, value.data.cpu().numpy(), batches_done)
                    # logger.add_histogram(tag+'grad', value.grad.data.cpu().numpy(),batches_done+1)
                '''
        if epoch % 10 == 0:
            visual_validation(epoch)
        if epoch % 100 == 0:
            metric_evaluation(test_dataloader,epoch)
        if epoch % 100 == 0:
            torch.save(model.state_dict(), model_save_path + '/model_%d.pth' % (epoch))


parser = ArgumentParser()





parser.add_argument("--model_dir", type=str,
                    dest="model_dir", default='LUAD_3_A_protein_xenium',
                    help="models folder")
parser.add_argument("--base_size", type=int,
                    dest="base_size", default=1024,
                    help="base size")
parser.add_argument("--load_model", type=str,
                    dest="load_model", default='./',
                    help="load model file to initialize with")

parser.add_argument("--initial_epoch", type=int,
                    dest="initial_epoch", default=0,
                    help="initial epoch")
parser.add_argument("--epochs", type=int,
                    dest="epochs", default=1001,
                    help="number of epoch")
args = parser.parse_args()



#modalities=('protein','xenium','visium')
modalities=('protein','xenium')
data = 'LUAD_3_A'

experiment_path = '/work/hm218/experiment/registration/combined/'
image_save_path = os.path.join(experiment_path,args.model_dir) + '/images/'
model_save_path = os.path.join(experiment_path,args.model_dir)+ '/saved_models/'
log_save_path = os.path.join(experiment_path,args.model_dir) + '/logs/'
os.makedirs(image_save_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(log_save_path, exist_ok=True)


print(args.model_dir)

img_mode='L'
gene_channel=3
if img_mode == 'RGB':
    img_transforms = transforms.Compose([
        transforms.ToTensor(),                               # Convert to tensor (C, H, W)
        transforms.Normalize(mean=(0.5, 0.5, 0.5),            # Normalize 3 channels
                             std=(0.5, 0.5, 0.5))
    ])
    # prepare the model
    model = NICE_Trans(in_channels= 3,
                 in_gene_channels=gene_channel)

else:
    img_transforms =  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    # prepare the model
    model = NICE_Trans(in_channels= 1,
                 in_gene_channels=gene_channel)


gene_transforms = transforms.Compose([transforms.ToTensor()])


'''
train_dataloader = DataLoader(Image_svg_pair(datalist="xenium_imglist/train_all.txt",base_size=args.base_size,image_transformer=img_transforms,
                                              gene_transformer=gene_transforms),batch_size=1, shuffle=True, num_workers=8)
test_dataloader = DataLoader(Image_svg_pair(datalist="xenium_imglist/test_all.txt",base_size=args.base_size,image_transformer=img_transforms,
                                            gene_transformer=gene_transforms,mode='val'),batch_size=1, shuffle=False, num_workers=8)
'''

'''
train_dataloader = DataLoader(Image_svg_pair(datalist="sennet_imglist/train.txt",base_size=args.base_size,image_transformer=img_transforms,
                                              gene_transformer=gene_transforms),batch_size=1, shuffle=True, num_workers=8)
test_dataloader = DataLoader(Image_svg_pair(datalist="sennet_imglist/test.txt",base_size=args.base_size,image_transformer=img_transforms,
                                            gene_transformer=gene_transforms,mode='val'),batch_size=1, shuffle=False, num_workers=8)
'''

'''
train_dataloader = DataLoader(Image_svg_pair(datalist="breast_imglist/train.txt",base_size=args.base_size,image_transformer=img_transforms,
                                              gene_transformer=gene_transforms),batch_size=1, shuffle=True, num_workers=8)
test_dataloader = DataLoader(Image_svg_pair(datalist="breast_imglist/test.txt",base_size=args.base_size,image_transformer=img_transforms,
                                            gene_transformer=gene_transforms,mode='val'),batch_size=1, shuffle=False, num_workers=8)
'''


train_dataloader = DataLoader(phenocycler_omics_pair(data_name=data,use_modalities=modalities,image_transformer=img_transforms,
                                              gene_transformer=gene_transforms),batch_size=1, shuffle=True, num_workers=8)
test_dataloader = DataLoader(phenocycler_omics_pair(data_name=data,use_modalities=modalities,image_transformer=img_transforms,
                                            gene_transformer=gene_transforms,mode='val'),batch_size=1, shuffle=False, num_workers=8)


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
if args.load_model != './':
    print('loading', args.load_model)

    checkpoint = torch.load(args.load_model, map_location="cpu")


    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to("cuda")
    print('loaded')


else:
    model.to(device)

train_model()










