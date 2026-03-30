"""
Datasets file. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import imgaug.augmenters as iaa


def cat_mask_with_features(feature,mask):
    mask_expanded = mask[..., np.newaxis]  # shape (64, 64, 1)

    # Concatenate along the last axis
    return np.concatenate([feature, mask_expanded], axis=-1)

class Image_pca_pair(Dataset):
    def __init__(self, data_dir, group_id, sample_id, base_size=None,
                 image_transformer=None, gene_transformer=None, mode='train'):
        super(Image_pca_pair, self).__init__()
        self.group_id=group_id
        self.mode = mode
        self.data_dir = os.path.join(data_dir,'network_input')
        self.sample_id = sample_id
        self.base_size = base_size
        self.image_transformer = image_transformer
        self.gene_transformer = gene_transformer
        # All ordered pairs (i, j) with i != j
        self.data = [
            [i, j]
            for i in self.sample_id
            for j in self.sample_id
            if i != j
        ]

    def __getitem__(self, index):
        data_pair = self.data[index % len(self.data)]
        fixed_prefix = self.data_dir + "/"+str(self.group_id)+'_'+str(data_pair[0])
        moving_prefix = self.data_dir + "/" + str(self.group_id) + '_' + str(data_pair[1])

        if self.base_size is not None:
            suffix = f"_{self.base_size}"
        else:
            suffix = ""
        # Fixed sample
        fixed_image = Image.open(f"{fixed_prefix}_image{suffix}.png").convert("RGB")
        fixed_pca_feature = np.load(f"{fixed_prefix}_pca_out{suffix}.npy")
        fixed_pca_mask = np.load(f"{fixed_prefix}_pca_mask{suffix}.npy")
        fixed_gene_feature = cat_mask_with_features(fixed_pca_feature, fixed_pca_mask)
        fixed_hires_path = f"{fixed_prefix}_hires_image{suffix}.jpg"

        # Moving sample
        moving_image = Image.open(f"{moving_prefix}_image{suffix}.png").convert("RGB")
        moving_pca_feature = np.load(f"{moving_prefix}_pca_out{suffix}.npy")
        moving_pca_mask = np.load(f"{moving_prefix}_pca_mask{suffix}.npy")
        moving_gene_feature = cat_mask_with_features(moving_pca_feature, moving_pca_mask)
        moving_hires_path = f"{moving_prefix}_hires_image{suffix}.jpg"

        if self.image_transformer is not None:
            fixed_image = self.image_transformer(fixed_image)
            moving_image = self.image_transformer(moving_image)
        if self.gene_transformer is not None:
            fixed_gene_feature = self.gene_transformer(fixed_gene_feature)
            moving_gene_feature = self.gene_transformer(moving_gene_feature)

        out = {'data_pair':data_pair,'fixed_image': fixed_image, 'fixed_gene': fixed_gene_feature,'fixed_hires_image': fixed_hires_path,
               'moving_image': moving_image, 'moving_gene': moving_gene_feature,'moving_hires_image':moving_hires_path}

        if self.mode == 'train':
            return out

        if self.mode == 'val':
            # Fixed sample validation
            fixed_val = np.load(f"{fixed_prefix}_validation{suffix}.npz")
            fixed_coor = fixed_val["coord"]
            fixed_label = fixed_val["label"]

            # Moving sample validation
            moving_val = np.load(f"{moving_prefix}_validation{suffix}.npz")
            moving_coor = moving_val["coord"]
            moving_label = moving_val["label"]

            val_set = {'fixed_coor': fixed_coor, 'fixed_label': fixed_label,
                       'moving_coor': moving_coor, 'moving_label': moving_label}
            return out, val_set


    def __len__(self):
            return len(self.data)



class scc_ablation_pair(Dataset):

    def __init__(self, ablation_type='image',ablation_level=1,patient=2,image_transformer=None,gene_transformer=None,mode='train'):
        super(scc_ablation_pair, self).__init__()
        self.root = "/media/huifang/data/registration/SCC/huifang/ablation/"
        self.ablation_type = ablation_type
        self.ablation_level =ablation_level
        self.patient = patient
        self.image_transformer = image_transformer
        self.gene_transformer = gene_transformer
        self.mode = mode
        if self.mode == 'train':
            self.data = [[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]]
        else:
            self.data = [[0, 1], [1, 2]]

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



def get_augmentation_parameters_with_size(h,w):
    seq = iaa.Sequential([
        iaa.Resize({"height": h, "width": w}),
        iaa.Affine(
            scale=(0.8, 1.0),  # random scale between 80% and 100%
            rotate=(-10, 10)  # random rotation between -25 to 25 degrees
        ),
        iaa.Multiply((0.8, 1.2)),  # change brightness
        iaa.GaussianBlur(sigma=(0, 1.0))  # apply Gaussian blur
    ], random_order=False)  # apply the augmentations in random order
    return seq

def augment_image(image):
    image = np.asarray(image)
    image = image.astype(np.uint8)
    h, w = image.shape[:2]
    seq = get_augmentation_parameters_with_size(h, w)
    image_aug = seq(images=[image])
    return image_aug[0]



class phenocycler_image_pair(Dataset):
    def __init__(self, datalist="train_data_list.txt",image_transformer=None):
        super(phenocycler_image_pair, self).__init__()
        self.root = "/media/huifang/data/registration/phenocycler/H5ADs/"
        self.image_transformer = image_transformer
        f = open(datalist, 'r')
        self.data = f.readlines()

    def __getitem__(self, index):
        # make consistent with all other datasets
        # return a PIL Image
        data = self.data[index % len(self.data)].strip().split(' ')

        data_name = data[0]
        fixed_modality = data[1]
        moving_modality = data[2]

        if fixed_modality == 'visium':
            fixed_image = Image.open(self.root + data_name+"_"+fixed_modality+'_GRAY_strenthened2.png')
        else:
            fixed_image = Image.open(self.root + data_name + "_" + fixed_modality + '_DAPI_strenthened2.png')

        if moving_modality == 'visium':
            moving_image = Image.open(self.root + data_name + "_" + moving_modality + '_GRAY_strenthened2.png')
        else:
            moving_image = Image.open(self.root + data_name + "_" + moving_modality + '_DAPI_strenthened2.png')
        fixed_image= augment_image(fixed_image)
        moving_image = augment_image(moving_image)
        fixed_image = Image.fromarray(fixed_image)
        moving_image = Image.fromarray(moving_image)

        if self.image_transformer is not None:
            fixed_image = self.image_transformer(fixed_image)
            moving_image = self.image_transformer(moving_image)
        out = {'fixed_image': fixed_image,'moving_image': moving_image}
        return out
    def __len__(self):
        return len(self.data)



class phenocycler_omics_pair(Dataset):

    def __init__(self, data_dir, data_name='LUAD_2_A', use_modalities=('protein', 'xenium', 'visium'),image_transformer=None,gene_transformer=None,mode='train'):
        super(phenocycler_omics_pair, self).__init__()
        self.data_dir = os.path.join(data_dir, 'network_input')
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

        fixed_prefix = self.data_dir+"/"+self.data_name+'_'+fixed_modality
        moving_prefix = self.data_dir+"/"+self.data_name+'_'+moving_modality

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
        out = {'data_pair':data_pair,'fixed_image': fixed_image,'fixed_gene': fixed_gene_feature,
           'moving_image': moving_image, 'moving_gene': moving_gene_feature}
        if self.mode == 'train':
            return out
        if self.mode == 'val':
            fixed_val = np.load(fixed_prefix + '_validation_full.npz')
            fixed_coor = fixed_val["coord"]
            fixed_label = fixed_val["label"]

            moving_val = np.load(moving_prefix + '_validation_full.npz')
            moving_coor = moving_val["coord"]
            moving_label = moving_val["label"]

            val_set = {'fixed_coor':fixed_coor,'fixed_label':fixed_label,
               'moving_coor':moving_coor,'moving_label':moving_label}
            return out,val_set

    def __len__(self):
        return len(self.data)