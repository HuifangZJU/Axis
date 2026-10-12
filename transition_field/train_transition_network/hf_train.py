import os
import glob
import sys
import random
import multiprocessing
import seaborn as sns
import time
import torch
import numpy as np
import scipy.ndimage
from argparse import ArgumentParser
from datasets import Image_gene_pair
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from itertools import cycle
# project imports
import datagenerators
from hf_he_networks import NICE_Trans,CoordinateWarper,CoordinateAffineWarper
# from hf_networks import NICE_Trans,SpatialTransformer_block_2D,AffineTransformer_block_2D,CoordinateWarper,CoordinateAffineWarper
# from networks import NICE_Trans,SpatialTransformer_block,AffineTransformer_block
import losses
import datetime
from tensorboardX import SummaryWriter
from scipy.spatial import cKDTree
import numpy as np
from collections import Counter

def Dice(vol1, vol2, labels=None, nargout=1):
    
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return np.mean(dicem)
    else:
        return (dicem, labels)
    
    
def NJD(displacement):

    D_y = (displacement[1:,:-1,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_x = (displacement[:-1,1:,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_z = (displacement[:-1,:-1,1:,:] - displacement[:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    Ja_value = D1-D2+D3
    
    return np.sum(Ja_value<0)


def knn_label_accuracy(points_x, labels_x, points_y, labels_y, k=1):
    # 1. Build a cKDTree for points_y
    labels_x = np.array(labels_x)
    labels_y = np.array(labels_y)
    labels_x = labels_x.ravel()
    labels_y = labels_y.ravel()
    tree_y = cKDTree(points_y)
    # 2. For each point in X, find the indices of the k nearest neighbors in Y
    distances, neighbor_indices = tree_y.query(points_x, k=k)

    # If k=1, neighbor_indices is shape (n,). Make it (n,k) for consistent indexing
    if k == 1:
        neighbor_indices = neighbor_indices[:, None]
        distances = distances[:, None]

    # 3. Predict label by majority vote among the k neighbors
    predicted_labels = np.empty(len(points_x), dtype=labels_y.dtype)

    for i in range(len(points_x)):
        # labels of the k nearest neighbors
        neighbor_labels = labels_y[neighbor_indices[i]]
        # majority vote
        label_counts = Counter(neighbor_labels)
        # break ties by choosing the smallest label (or define your tie-break rule)
        predicted_labels[i] = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    # 4. Compute accuracy

    correct = (predicted_labels == labels_x)
    accuracy = np.mean(correct)
    return accuracy


def create_gene_overlay(fixed_coors,moving_coors,fixed_labels,moving_labels,ax):
    coors = torch.cat((fixed_coors, moving_coors), dim=0)
    labels = torch.cat((fixed_labels,moving_labels),dim=0)
    unique_labels = np.unique(labels)  # sorted unique label values
    palette = sns.color_palette("deep", len(unique_labels))  # e.g. 'deep', 'tab10', etc.
    color_index = np.searchsorted(unique_labels, labels)
    colors = np.array(palette)[color_index]
    ax.scatter(coors[:, 0], coors[:, 1], s=5, color=colors)
    ax.invert_yaxis()
    return ax


def min_max_normalize(array):
    min = np.min(array)   # shape (C,)
    max = np.max(array)   # shape (C,)
    range = max - min

    normalized_data = (array - min) / range
    return normalized_data


def create_overlay(fixed, moving):
    fixed = min_max_normalize(fixed.detach().cpu().numpy().squeeze())
    moving = min_max_normalize(moving.detach().cpu().numpy().squeeze())

    overlay = np.zeros((fixed.shape[0], fixed.shape[1], 3))
    overlay[..., 0] = fixed  # Magenta - Red Channel
    overlay[..., 1] = moving  # Cyan - Green and Blue Channel
    overlay[..., 2] = moving
    return overlay

def save_image_registrations(fixed,moving,affined,warpped,epoch,i):

    original = create_overlay(fixed,moving)
    after_affine = create_overlay(fixed,affined)
    final_result = create_overlay(fixed,warpped)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original displacement")

    axes[1].imshow(after_affine)
    axes[1].set_title("Overlay After Affine transformation (Magenta/Cyan)")

    axes[2].imshow(final_result)
    axes[2].set_title("Overlay After Deformable Registration (Magenta/Cyan)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(image_save_path + str(epoch) + '_'+str(i)+'_image.png',dpi=300)
    plt.close()
    # plt.show()

def save_gene_registrations(fixed,moving,affined,warpped,fixed_labels,moving_labels,epoch,i):

    # Plot results


    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0]=create_gene_overlay(fixed,moving,fixed_labels,moving_labels,axes[0])
    axes[0].set_title("Original displacement")

    axes[1]=create_gene_overlay(fixed,affined,fixed_labels,moving_labels,axes[1])
    axes[1].set_title("Overlay After Affine transformation (Magenta/Cyan)")

    axes[2]=create_gene_overlay(fixed,warpped,fixed_labels,moving_labels,axes[2])
    axes[2].set_title("Overlay After Deformable Registration (Magenta/Cyan)")


    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(image_save_path + str(epoch) + '_'+str(i)+'_cluster.png',dpi=300)
    # plt.show()
    plt.close()

def validation(epoch):
    model.eval()
    knn_accuracy=[]
    for i, test_batch in enumerate(test_dataloader):
        with torch.no_grad():
            inputs,val_set = test_batch
            fixed_image = inputs['fixed_image'].to(device).float()
            fixed_gene = inputs['fixed_gene'].to(device).float()
            moving_image = inputs['moving_image'].to(device).float()
            moving_gene = inputs['moving_gene'].to(device).float()

            warped_image, warped_gene, flow, affined_image, affined_gene, affine_para = \
                model(fixed_image, fixed_gene, moving_image, moving_gene)

            save_image_registrations(fixed_image,moving_image,affined_image,warped_image,epoch,i)


            fixed_coor = val_set['fixed_coor'].squeeze()
            fixed_label = val_set['fixed_label'].squeeze()
            moving_coor = val_set['moving_coor'].to(device).float()
            moving_label = val_set['moving_label'].squeeze()

            warped_coor = SpatialTransformer_Coor(moving_coor,flow)
            warped_coor = warped_coor.detach().cpu().squeeze()
            affined_coor = AffineTransformer_Coor(moving_coor,affine_para,fixed_image.shape[-1],fixed_image.shape[-1])
            affined_coor = affined_coor.detach().cpu().squeeze()
            moving_coor = moving_coor.detach().cpu().squeeze()


            acc = knn_label_accuracy(fixed_coor.numpy(), fixed_label.numpy(), warped_coor.numpy(), moving_label.numpy(),
                                     5)
            knn_accuracy.append(acc)

            save_gene_registrations(fixed_coor,moving_coor,affined_coor,warped_coor,fixed_label,moving_label,epoch,i)

    average = sum(knn_accuracy) / len(knn_accuracy)
    return average

def validation_he(epoch):
    model.eval()
    for i, test_batch in enumerate(test_dataloader):
        with torch.no_grad():
            inputs,val_set = test_batch
            fixed_image = inputs['fixed_image'].to(device).float()
            moving_image = inputs['moving_image'].to(device).float()
            fixed_image = fixed_image.unsqueeze(dim=1)
            moving_image = moving_image.unsqueeze(dim=1)

            warped_image, flow, affined_image, affine_para = \
                model(fixed_image, moving_image)

            save_image_registrations(fixed_image,moving_image,affined_image,warped_image,epoch,i)



parser = ArgumentParser()

parser.add_argument("--model_dir", type=str,
                    dest="model_dir", default='he_image',
                    help="models folder")
parser.add_argument("--load_model", type=str,
                    dest="load_model", default='./',
                    help="load model file to initialize with")
parser.add_argument("--initial_epoch", type=int,
                    dest="initial_epoch", default=0,
                    help="initial epoch")
parser.add_argument("--epochs", type=int,
                    dest="epochs", default=200,
                    help="number of epoch")
parser.add_argument("--batch_size", type=int,
                    dest="batch_size", default=1,
                    help="batch size")
args = parser.parse_args()



experiment_path = '/home/huifang/Pictures/'
image_save_path = os.path.join(experiment_path,args.model_dir) + '/images/'
model_save_path = os.path.join(experiment_path,args.model_dir)+ '/saved_models/'
log_save_path = os.path.join(experiment_path,args.model_dir) + '/logs/'
os.makedirs(image_save_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(log_save_path, exist_ok=True)




img_transforms = [transforms.ToTensor(),
                  transforms.Normalize((0.5), (0.5))]
gene_transforms = [transforms.ToTensor()]

# debug_dataset = Image_gene_pair(datalist="train_data_list.txt",image_transformer=img_transforms)
# test = debug_dataset.__getitem__(1)
# plt.imshow(test['moving_image'],cmap='gray')
# plt.show()

train_dataloader = DataLoader(Image_gene_pair(datalist="train_data_list.txt",image_transformer=img_transforms,
                                              gene_transformer=gene_transforms),batch_size=args.batch_size, shuffle=False, num_workers=16)
test_dataloader = DataLoader(Image_gene_pair(datalist='validation_data_list.txt',image_transformer=img_transforms,
                                            gene_transformer=gene_transforms,mode='val'),batch_size=1, shuffle=False, num_workers=16)




# prepare model folder
if not os.path.isdir(args.model_dir):
    os.mkdir(args.model_dir)

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
model.to(device)
if args.load_model != './':
    print('loading', args.load_model)
    state_dict = torch.load(args.load_model, map_location=device)
    model.load_state_dict(state_dict)

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

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# prepare losses
warpping_loss = losses.NCC(win=9).loss
displacement_loss = losses.Regu_loss


prev_time = time.time()
logger = SummaryWriter(log_save_path)
# training/validate loops
for epoch in range(args.initial_epoch, args.epochs):
    start_time = time.time()
    # training
    model.train()
    train_losses = []
    train_total_loss = []
    for i, batch in enumerate(train_dataloader):
        fixed_image = batch['fixed_image'].to(device).float()
        fixed_gene = batch['fixed_gene'].to(device).float()

        moving_image = batch['moving_image'].to(device).float()
        moving_gene = batch['moving_gene'].to(device).float()

        # warped_image,warped_gene, flow, affined_image,affined_gene, affine_para = \
        #     model(fixed_image,fixed_gene,moving_image,moving_gene)

        fixed_image = fixed_image.unsqueeze(dim=2)
        moving_image = moving_image.unsqueeze(dim=2)


        warped_image, flow, affined_image, affine_para = \
            model(fixed_image, moving_image)
        # print(fixed_image.shape)
        # print(warped_image.shape)
        # print(affined_image.shape)
        #
        # print(fixed_gene.shape)
        # print(warped_gene.shape)
        # print(affined_gene.shape)


        fixed_image_np = affined_image.detach().cpu().numpy().squeeze()
        # calculate total loss
        deformable_image_loss = warpping_loss(fixed_image,warped_image)
        affine_image_loss = warpping_loss(fixed_image,affined_image)

        # deformable_gene_loss = warpping_loss(fixed_gene,warped_gene)
        # affine_gene_loss = warpping_loss(fixed_gene,affined_gene)

        flow_np = flow.detach().cpu().numpy().squeeze()


        filed_loss = displacement_loss(flow)
        # loss = deformable_image_loss + deformable_gene_loss + affine_image_loss + affine_gene_loss + filed_loss
        loss = deformable_image_loss +  affine_image_loss +  filed_loss
        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Determine approximate time left
        batches_done = epoch * len(train_dataloader) + i
        batches_left = args.epochs * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r" + args.model_dir + "---[Epoch %d/%d] [Batch %d/%d] [Total loss: %f] [Affine loss: %f] [Deformable loss: %f] [Regu loss: %f]  ETA: %s" %
            (epoch, args.epochs,
             i, len(train_dataloader),
             # loss.item(),affine_image_loss.item()+affine_gene_loss.item(),deformable_image_loss.item()+deformable_gene_loss.item(),filed_loss.item(), time_left))
             loss.item(), affine_image_loss.item(), deformable_image_loss.item(), filed_loss.item(), time_left))
        # # --------------tensor board--------------------------------#
        # if batches_done % 20 == 0:
        #     # info = {'loss': loss.item(),'deformable_image_loss':deformable_image_loss,'affine_loass':affine_image_loss,'deformable_gene_loss':deformable_gene_loss,'affine_gene_loss':affine_gene_loss}
        #     info = {'loss': loss.item(), 'deformable_image_loss': deformable_image_loss,'affine_loass': affine_image_loss}
        #     for tag, value in info.items():
        #         logger.add_scalar(tag, value, batches_done)
        #     for tag, value in model.named_parameters():
        #         tag = tag.replace('.', '/')
        #         logger.add_histogram(tag, value.data.cpu().numpy(), batches_done)
        #         # logger.add_histogram(tag+'grad', value.grad.data.cpu().numpy(),batches_done+1)

        # average_acc = validation(epoch)
    validation_he(epoch)
    # logger.add_scalar('val_acc', average_acc, batches_done)
    # save model checkpoint
    if epoch % 20 == 0:
        torch.save(model.state_dict(), model_save_path+'/model_%d.pth' % (epoch))



