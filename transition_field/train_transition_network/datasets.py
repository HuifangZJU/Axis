"""
Datasets file. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
import imgaug.augmenters as iaa

root = "/hpc/group/jilab/huifang/code/registration/data"


def cat_features_to_same_size(A,B):
    # Concatenate along the last axis
    C = np.concatenate([A, B], axis=-1)  # shape: (64, 64, n + m)

    # Pad if needed
    current_channels = C.shape[-1]
    target_channels = 64

    if current_channels < target_channels:
        pad_width = target_channels - current_channels
        pad_shape = ((0, 0), (0, 0), (0, pad_width))  # pad only last axis
        C = np.pad(C, pad_shape, mode='constant', constant_values=0)

    return C

def cat_mask_with_features(feature,mask):
    mask_expanded = mask[..., np.newaxis]  # shape (64, 64, 1)

    # Concatenate along the last axis
    return np.concatenate([feature, mask_expanded], axis=-1)

class Image_svg_pair(Dataset):

    def __init__(self, datalist="train_data_list.txt",base_size=512,image_transformer=None,gene_transformer=None,mode='train'):
        super(Image_svg_pair, self).__init__()
        self.root = root
        self.mode = mode
        self.image_transformer = image_transformer
        self.gene_transformer = gene_transformer
        datapath = os.path.join(root,datalist)
        f = open(datapath, 'r')
        self.data = f.readlines()
        self.base_size = base_size

    def __getitem__(self, index):
        # make consistent with all other datasets
        # return a PIL Image
        
        data_pair = self.data[index % len(self.data)].strip()
        data_pair = data_pair.split(' ')
        fixed_image = Image.open(self.root + data_pair[0] + '_image_'+str(self.base_size)+'.png')
        fixed_pca_feature = np.load(self.root+data_pair[0]+'_pca_out_'+str(self.base_size)+'.npy')
        fixed_pca_mask = np.load(self.root+data_pair[0]+'_pca_mask_'+str(self.base_size)+'.npy')
        fixed_gene_feature = cat_mask_with_features(fixed_pca_feature,fixed_pca_mask)

        moving_image = Image.open(self.root + data_pair[1] + '_image_'+str(self.base_size)+'.png')
        moving_pca_feature = np.load(self.root+data_pair[1]+'_pca_out_'+str(self.base_size)+'.npy')
        moving_pca_mask = np.load(self.root+data_pair[1]+'_pca_mask_'+str(self.base_size)+'.npy')
        moving_gene_feature = cat_mask_with_features(moving_pca_feature,moving_pca_mask)

        if self.image_transformer is not None:
            fixed_image = self.image_transformer(fixed_image)
            moving_image = self.image_transformer(moving_image)
        if self.gene_transformer is not None:
            fixed_gene_feature = self.gene_transformer(fixed_gene_feature)
            moving_gene_feature = self.gene_transformer(moving_gene_feature)


        out = {'fixed_image': fixed_image,'fixed_gene': fixed_gene_feature,
           'moving_image': moving_image, 'moving_gene': moving_gene_feature}


        if self.mode == 'train':
            return out

        if self.mode == 'val':
            fixed_val = np.load(self.root + data_pair[0] + '_validation_'+str(self.base_size)+'.npz')
            fixed_coor = fixed_val["coord"]
            fixed_label = fixed_val["label"]

            moving_val = np.load(self.root + data_pair[1] + '_validation_'+str(self.base_size)+'.npz')
            moving_coor = moving_val["coord"]
            moving_label = moving_val["label"]

            val_set = {'fixed_coor':fixed_coor,'fixed_label':fixed_label,
               'moving_coor':moving_coor,'moving_label':moving_label}
            return out,val_set

    def __len__(self):
        return len(self.data)



class scc_ablation_pair(Dataset):

    def __init__(self, ablation_type='image',ablation_level=1,patient=2,image_transformer=None,gene_transformer=None,mode='train'):
        super(scc_ablation_pair, self).__init__()
        self.root = "/hpc/group/jilab/huifang/code/registration/data/SCC/ablation/"
        self.ablation_type = ablation_type
        self.ablation_level =ablation_level
        self.patient = patient
        self.image_transformer = image_transformer
        self.gene_transformer = gene_transformer
        self.mode = mode
        if self.mode == 'train':
            self.data = [[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]]
        else:
            self.data = [[0, 1], [0, 2]]

    def __getitem__(self, index):
        # make consistent with all other datasets
        # return a PIL Image
        data_pair = self.data[index % len(self.data)]
        fixed_prefix = self.root+self.ablation_type+'/'+str(self.ablation_level)+'/patient_'+str(self.patient)+'_'+str(data_pair[0])
        moving_prefix = self.root+self.ablation_type+'/'+str(self.ablation_level)+'/patient_'+str(self.patient)+'_'+str(data_pair[1])
        fixed_image = Image.open(fixed_prefix + '_image_512.png').convert("RGB")
        fixed_pca_feature = np.load(fixed_prefix+'_pca_out.npy')
        fixed_pca_mask = np.load(fixed_prefix + '_pca_mask.npy')
        fixed_gene_feature = cat_mask_with_features(fixed_pca_feature,fixed_pca_mask)

        moving_image = Image.open(moving_prefix + '_image_512.png').convert("RGB")
        moving_pca_feature = np.load(moving_prefix+'_pca_out.npy')
        moving_pca_mask = np.load(moving_prefix+'_pca_mask.npy')
        moving_gene_feature = cat_mask_with_features(moving_pca_feature,moving_pca_mask)

        if self.image_transformer is not None:
            fixed_image = self.image_transformer(fixed_image)
            moving_image = self.image_transformer(moving_image)
        if self.gene_transformer is not None:
            fixed_gene_feature = self.gene_transformer(fixed_gene_feature)
            moving_gene_feature = self.gene_transformer(moving_gene_feature)
        out = {'fixed_image': fixed_image,'fixed_gene': fixed_gene_feature,
           'moving_image': moving_image, 'moving_gene': moving_gene_feature}
        if self.mode == 'train':
            return out

        if self.mode == 'val':
            fixed_val = np.load(fixed_prefix + '_validation.npz')
            fixed_coor = fixed_val["coord"]
            fixed_label = fixed_val["label"]

            moving_val = np.load(moving_prefix + '_validation.npz')
            moving_coor = moving_val["coord"]
            moving_label = moving_val["label"]

            val_set = {'fixed_coor':fixed_coor,'fixed_label':fixed_label,
               'moving_coor':moving_coor,'moving_label':moving_label}
            return out,val_set

    def __len__(self):
        return len(self.data)




class Image_gene_pair(Dataset):

    def __init__(self, datalist="train_data_list.txt",image_transformer=None,gene_transformer=None,mode='train'):
        super(Image_gene_pair, self).__init__()
        self.root = root
        self.mode = mode
        self.image_transformer = image_transformer
        self.gene_transformer = gene_transformer
        datapath = os.path.join(root,datalist)
        f = open(datapath, 'r')
        self.data = f.readlines()

    def __getitem__(self, index):
        # make consistent with all other datasets
        # return a PIL Image
        data_pair = self.data[index % len(self.data)].strip()
        data_pair = data_pair.split(' ')
        fixed_image = Image.open(self.root + data_pair[0] + '_image.png').convert("RGB")
        fixed_gene_feature = np.load(self.root+data_pair[0]+'_pca_out.npy')


        moving_image = Image.open(self.root + data_pair[1] + '_image.png').convert("RGB")
        moving_gene_feature = np.load(self.root+data_pair[1]+'_pca_out.npy')

        if self.image_transformer is not None:
            fixed_image = self.image_transformer(fixed_image)
            moving_image = self.image_transformer(moving_image)
        if self.gene_transformer is not None:
            fixed_gene_feature = self.gene_transformer(fixed_gene_feature)
            moving_gene_feature = self.gene_transformer(moving_gene_feature)

        out = {'fixed_image': fixed_image,'fixed_gene': fixed_gene_feature,
           'moving_image': moving_image, 'moving_gene': moving_gene_feature}
        if self.mode == 'train':
            return out

        if self.mode == 'val':
            fixed_val = np.load(self.root + data_pair[0] + '_validation.npz')
            fixed_coor = fixed_val["coord"]
            fixed_label = fixed_val["label"]

            moving_val = np.load(self.root + data_pair[1] + '_validation.npz')
            moving_coor = moving_val["coord"]
            moving_label = moving_val["label"]

            val_set = {'fixed_coor':fixed_coor,'fixed_label':fixed_label,
               'moving_coor':moving_coor,'moving_label':moving_label}
            return out,val_set

    def __len__(self):
        return len(self.data)




class Image_pair(Dataset):
    def __init__(self, datalist="train_data_list.txt",image_transformer=None):
        super(Image_pair, self).__init__()
        self.root = root
        # if image_transformer is not None:
        #     self.image_transformer = transforms.Compose(image_transformer)
        # else:
        #     self.image_transformer = None
        self.image_transformer = image_transformer
        datapath = os.path.join(root,datalist)
        f = open(datapath, 'r')
        self.data = f.readlines()

    def __getitem__(self, index):
        # make consistent with all other datasets
        # return a PIL Image
        data_pair = self.data[index % len(self.data)].strip()
        data_pair = data_pair.split(' ')
        fixed_image = Image.open(self.root + data_pair[0] + '_image_512.png').convert("RGB")

        moving_image = Image.open(self.root + data_pair[1] + '_image_512.png').convert("RGB")

        if self.image_transformer is not None:
            # fixed_mask = self.image_transformer(fixed_mask)
            # moving_mask = self.image_transformer(moving_mask)
            fixed_image = self.image_transformer(fixed_image)
            moving_image = self.image_transformer(moving_image)
        # to_tensor = transforms.ToTensor()
        # fixed_image = to_tensor(fixed_image)
        # moving_image = to_tensor(moving_image)

        # out = {'fixed_image': fixed_image,'fixed_mask': fixed_mask,
        #    'moving_image': moving_image,'moving_mask': moving_mask }
        out = {'fixed_image': fixed_image,'moving_image': moving_image}
        return out

    def __len__(self):
        return len(self.data)




class Sennet_image_pair(Dataset):
    def __init__(self, datalist="train_data_list.txt",image_transformer=None):
        super(Sennet_image_pair, self).__init__()
        self.root = "/hpc/group/jilab/huifang/dataset/sennet/hf_aligned_data/"
        self.image_transformer = image_transformer
        f = open(datalist, 'r')
        self.data = f.readlines()

    def __getitem__(self, index):
        # make consistent with all other datasets
        # return a PIL Image
        data_pair = self.data[index % len(self.data)].strip()
        data_pair = data_pair.split(' ')
        xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = data_pair

        xenium_img_name = f"xenium_{xenium_sampleid}_{xenium_regionid}_enhanced.png"
        codex_img_name = f"codex_{codex_sampleid}_{codex_regionid}_enhanced.png"

        fixed_image = Image.open(self.root + xenium_img_name)
        moving_image = Image.open(self.root + codex_img_name)


        if self.image_transformer is not None:
            fixed_image = self.image_transformer(fixed_image)
            moving_image = self.image_transformer(moving_image)
        out = {'fixed_image': fixed_image,'moving_image': moving_image}
        return out
    def __len__(self):
        return len(self.data)

def get_augmentation_parameters_with_size(h,w):
    seq = iaa.Sequential([
        iaa.Resize({"height": h, "width": w}),
        iaa.Affine(
            scale=(0.9, 1.0),  # random scale between 80% and 100%
            rotate=(-10, 10)  # random rotation between -25 to 25 degrees
        ),
        iaa.Multiply((0.9, 1.1)),  # change brightness
        iaa.GaussianBlur(sigma=(0, 1.0))  # apply Gaussian blur
    ], random_order=False)  # apply the augmentations in random order
    return seq

def augment_image(image):
    image = np.array(image)
    image = image.astype(np.uint8)
    h, w = image.shape[:2]
    seq = get_augmentation_parameters_with_size(h, w)
    image_aug = seq(images=[image])
    return Image.fromarray(image_aug[0])



class phenocycler_image_pair(Dataset):
    def __init__(self, datalist="train_data_list.txt",image_transformer=None,mode='train'):
        super(phenocycler_image_pair, self).__init__()
        self.root = "/hpc/group/jilab/huifang/code/registration/data/phenocycler/H5ADs/huifang/"
        self.image_transformer = image_transformer
        f = open(datalist, 'r')
        self.data = f.readlines()
        self.mode=mode

    def __getitem__(self, index):
        # make consistent with all other datasets
        # return a PIL Image
        data = self.data[index % len(self.data)].strip().split(' ')

        data_name = data[0]
        fixed_modality = data[1]
        moving_modality = data[2]


        if fixed_modality == 'visium':
            fixed_image = Image.open(self.root + data_name+"_"+fixed_modality+'_GRAY_trans_enhanced.png')
        else:
            fixed_image = Image.open(self.root + data_name + "_" + fixed_modality + '_DAPI_trans_enhanced.png')

        if moving_modality == 'visium':
            moving_image = Image.open(self.root + data_name + "_" + moving_modality + '_GRAY_trans_enhanced.png')
        else:
            moving_image = Image.open(self.root + data_name + "_" + moving_modality + '_DAPI_trans_enhanced.png')
        '''
        if self.mode == 'train':
            fixed_image= augment_image(fixed_image)
            moving_image = augment_image(moving_image)
        '''
        if self.image_transformer is not None:
            fixed_image = self.image_transformer(fixed_image)
            moving_image = self.image_transformer(moving_image)
        out = {'fixed_image': fixed_image,'moving_image': moving_image}
        return out
    def __len__(self):
        return len(self.data)



class phenocycler_omics_pair(Dataset):

    def __init__(self, data_name='LUAD_2_A', use_modalities=('protein', 'xenium', 'visium'),image_transformer=None,gene_transformer=None,mode='train'):
        super(phenocycler_omics_pair, self).__init__()
        self.root = "/hpc/group/jilab/huifang/code/registration/data/phenocycler/H5ADs/huifang/"
        self.image_transformer = image_transformer
        self.gene_transformer = gene_transformer
        self.mode = mode
        self.data_name = data_name
        all_pairs = [
            ['protein', 'visium'],
            ['protein', 'xenium'],
            ['xenium', 'visium'],
            ['xenium', 'protein'],
            ['visium', 'protein'],
            ['visium', 'xenium'],
        ]
        # keep only those pairs where both modalities are in use_modalities
        self.data = [p for p in all_pairs if all(m in use_modalities for m in p)]

    def __getitem__(self, index):
        data_pair = self.data[index % len(self.data)]
        fixed_modality = data_pair[0]
        moving_modality = data_pair[1]

        fixed_prefix = self.root+self.data_name+'_'+fixed_modality
        moving_prefix = self.root+self.data_name+'_'+moving_modality

        if fixed_modality == 'visium':
            fixed_image = Image.open(fixed_prefix + '_GRAY_trans_enhanced.png')
        else:
            fixed_image = Image.open(fixed_prefix + '_DAPI_trans_enhanced.png')
        fixed_feature = np.load(fixed_prefix+'_feature_out.npy')
        fixed_mask = np.load(fixed_prefix + '_feature_mask.npy')
        fixed_gene_feature = cat_mask_with_features(fixed_feature,fixed_mask)

        if moving_modality == 'visium':
            moving_image = Image.open(moving_prefix + '_GRAY_trans_enhanced.png')
        else:
            moving_image = Image.open(moving_prefix + '_DAPI_trans_enhanced.png')
        moving_feature = np.load(moving_prefix+'_feature_out.npy')
        moving_mask = np.load(moving_prefix+'_feature_mask.npy')
        moving_gene_feature = cat_mask_with_features(moving_feature,moving_mask)

        if self.image_transformer is not None:
            fixed_image = self.image_transformer(fixed_image)
            moving_image = self.image_transformer(moving_image)
        if self.gene_transformer is not None:
            fixed_gene_feature = self.gene_transformer(fixed_gene_feature)
            moving_gene_feature = self.gene_transformer(moving_gene_feature)
        out = {'fixed_image': fixed_image,'fixed_gene': fixed_gene_feature,
           'moving_image': moving_image, 'moving_gene': moving_gene_feature}
        if self.mode == 'train':
            return out
        if self.mode == 'val':
            fixed_val = np.load(fixed_prefix + '_validation.npz')
            fixed_coor = fixed_val["coord"]
            fixed_label = fixed_val["label"]

            moving_val = np.load(moving_prefix + '_validation.npz')
            moving_coor = moving_val["coord"]
            moving_label = moving_val["label"]

            val_set = {'fixed_coor':fixed_coor,'fixed_label':fixed_label,
               'moving_coor':moving_coor,'moving_label':moving_label}
            return out,val_set

    def __len__(self):
        return len(self.data)
