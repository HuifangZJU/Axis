from axis.settings import data_path, experiment_path
import os
import sys
import time
import torch
from argparse import ArgumentParser
from axis.registration.data import phenocycler_image_pair
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from axis.registration import losses
import datetime
from tensorboardX import SummaryWriter
import numpy as np
from axis.registration import NICE_Trans_img
from itertools import cycle


def min_max_normalize(array):
    min = np.min(array)   # shape (C,)
    max = np.max(array)   # shape (C,)
    range = max - min

    normalized_data = (array - min) / range
    return normalized_data


def create_overlay(fixed, moving):
    fixed = min_max_normalize(fixed.detach().cpu().squeeze().numpy())
    moving = min_max_normalize(moving.detach().cpu().squeeze().numpy())
    # fixed = min_max_normalize(fixed)
    # moving = min_max_normalize(moving)
    overlay = np.zeros((fixed.shape[0], fixed.shape[1], 3))
    overlay[..., 0] = fixed  # Red
    overlay[..., 1] = moving  # Green
    overlay[..., 2] = moving  # Blue
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
    plt.savefig(image_save_path + str(epoch)+'_'+str(i) +'_image.png')
    plt.close()
    # plt.show()

def validation(epoch,test_dataloader):
    model.eval()
    for i, test_batch in enumerate(test_dataloader):
        with torch.no_grad():


            fixed_image = test_batch['fixed_image'].to(device).float()
            moving_image = test_batch['moving_image'].to(device).float()

            fixed_image = fixed_image.unsqueeze(dim=2)
            moving_image = moving_image.unsqueeze(dim=2)

            warped_image, flow, affined_image, affine_para = \
                model(fixed_image, moving_image)

            warped_image = warped_image.squeeze(dim=2)
            affined_image = affined_image.squeeze(dim=2)

            save_image_registrations(fixed_image, moving_image, affined_image, warped_image, epoch,i)





parser = ArgumentParser()

parser.add_argument("--model_dir", type=str,
                    dest="model_dir", default='phenocycler_TSU_35_small_lr',
                    help="models folder")

parser.add_argument("--data_name", type=str,
                    dest="data_name", default='TSU_35',
                    help="phenocycler data name")
parser.add_argument("--load_model", type=str,
                    dest="load_model", default='./',
                    help="load model file to initialize with")
parser.add_argument("--initial_epoch", type=int,
                    dest="initial_epoch", default=0,
                    help="initial epoch")
parser.add_argument("--epochs", type=int,
                    dest="epochs", default=1000,
                    help="number of epoch")
parser.add_argument("--batch_size", type=int,
                    dest="batch_size", default=1,
                    help="batch size")
args = parser.parse_args()


experiment_root = experiment_path('phenocycler/')
image_save_path = os.path.join(experiment_root,args.model_dir) + '/images/'
model_save_path = os.path.join(experiment_root,args.model_dir)+ '/saved_models/'
log_save_path = os.path.join(experiment_root,args.model_dir) + '/logs/'
os.makedirs(image_save_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(log_save_path, exist_ok=True)


base_size=1024
img_mode='L'
if img_mode == 'RGB':
    img_transforms = transforms.Compose([
        transforms.Resize((base_size,base_size)),# Convert to tensor (C, H, W)
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),            # Normalize 3 channels
                             std=(0.5, 0.5, 0.5))
    ])
else:
    img_transforms =  transforms.Compose([
                transforms.Resize((base_size, base_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])




train_dataloader = DataLoader(phenocycler_image_pair(datalist=data_path(f"phenocycler/{args.data_name}.txt"),image_transformer=img_transforms,mode='train'),batch_size=args.batch_size, shuffle=True, num_workers=8)
test_dataloader = DataLoader(phenocycler_image_pair(datalist=data_path(f"phenocycler/{args.data_name}.txt"),image_transformer=img_transforms,mode='val'),batch_size=args.batch_size, shuffle=False, num_workers=8)
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


if args.load_model != './':
    print('loading', args.load_model)
    # state_dict = torch.load(args.load_model, map_location=device)
    # model.load_state_dict(state_dict)

    checkpoint = torch.load(args.load_model, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to("cuda")
else:
    model.to(device)
# transfer model
SpatialTransformer = model.SpatialTransformer
SpatialTransformer.to(device)
SpatialTransformer.eval()



AffineTransformer = model.AffineTransformer
AffineTransformer.to(device)
AffineTransformer.eval()


# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
img_warpping_loss = losses.NCC(win=31).loss
displacement_loss = losses.Regu_loss


prev_time = time.time()
logger = SummaryWriter(log_save_path)
# training/validate loops
for epoch in range(args.initial_epoch, args.epochs+1):
    start_time = time.time()
    # training
    model.train()
    train_losses = []
    train_total_loss = []
    for i, batch in enumerate(train_dataloader):
        fixed_image = batch['fixed_image'].to(device).float()
        moving_image = batch['moving_image'].to(device).float()



        fixed_image = fixed_image.unsqueeze(dim=2)
        moving_image = moving_image.unsqueeze(dim=2)

        warped_image, flow, affined_image, affine_para = \
            model(fixed_image, moving_image)

        # calculate total loss
        deformable_image_loss = img_warpping_loss(fixed_image,warped_image)
        affine_image_loss = img_warpping_loss(fixed_image,affined_image)

        field_loss = 1*displacement_loss(flow)
        image_loss = 1*deformable_image_loss + 1*affine_image_loss



        loss = image_loss + field_loss

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
            "\r" + args.model_dir + "---[Epoch %d/%d] [Batch %d/%d] [Total loss: %f] [Image loss: %f] [Regu loss: %f]  ETA: %s" %
            (epoch, args.epochs,
             i, len(train_dataloader),
             loss.item(),image_loss.item(),field_loss.item(), time_left))
        # # --------------tensor board--------------------------------#
        if batches_done % 20 == 0:
            info = {'loss': loss.item(),'deformable_image_loss':deformable_image_loss,'affine_image_loss':affine_image_loss}
            for tag, value in info.items():
                logger.add_scalar(tag, value, batches_done)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.add_histogram(tag, value.data.cpu().numpy(), batches_done)
                # logger.add_histogram(tag+'grad', value.grad.data.cpu().numpy(),batches_done+1)
    if epoch % 10 ==0:
        validation(epoch,test_dataloader)
    # save model checkpoint
    if epoch % 100 == 0:
        torch.save(model.state_dict(), model_save_path+'/model_%d.pth' % (epoch))



