from utils import *
from img_utils import *
Image.MAX_IMAGE_PIXELS = None
import matplotlib as mpl
import scanpy as sc
import cv2
from functools import reduce
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

# ----------------- main degradation -----------------

def apply_deformation(image, coords_xy, dy, dx, order=1):
    H, W = image.shape[:2]
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    map_y = yy + dy
    map_x = xx + dx
    # warp image
    if image.ndim == 2:
        warped_img = map_coordinates(image, [map_y, map_x], order=order, mode="reflect")
    else:
        chans = [map_coordinates(image[...,c], [map_y, map_x], order=order, mode="reflect")
                 for c in range(image.shape[2])]
        warped_img = np.stack(chans, axis=-1)
    # warp coords (x,y)
    x = np.clip(coords_xy[:,0].astype(np.float32), 0, W-1)
    y = np.clip(coords_xy[:,1].astype(np.float32), 0, H-1)
    dxp = map_coordinates(dx, [y, x], order=1, mode="reflect")
    dyp = map_coordinates(dy, [y, x], order=1, mode="reflect")
    warped_coords = np.column_stack([coords_xy[:,0] - dxp, coords_xy[:,1] - dyp])
    return warped_img, warped_coords

def load_scc_data(out_dir):
    path_to_h5ads = out_dir + 'H5ADs/'
    patient_2 = []
    patient_5 = []
    patient_9 = []
    patient_10 = []

    patients = {
        "patient_2": patient_2,
        "patient_5": patient_5,
        "patient_9": patient_9,
        "patient_10": patient_10,
    }
    for k in patients.keys():
        p_id = k.split('_')[-1]
        p_id = int(p_id)
        for i in range(3):
            data = sc.read_h5ad(path_to_h5ads + f"scc_p{p_id}_layer{i + 1}.h5ad")
            patients[k].append(data)
    return patients


out_dir='/media/huifang/data/registration/SCC/huifang/'
patients = load_scc_data(out_dir)


for k, slices in patients.items():
    for sl in slices:
        sc.pp.filter_genes(sl, min_cells=15, inplace=True)
        sc.pp.filter_cells(sl, min_genes=100, inplace=True)
    all_gene_lists = [sl.var.index for sl in slices]
    common_genes = reduce(np.intersect1d, all_gene_lists)
    # 2. Subset each slice to the common genes, gather coordinates & data
    gene_data_list = []
    coords_list = []
    label_list = []
    for sl in slices:
        # Focus on common genes only
        sl_sub = sl[:, common_genes]
        # Convert to a NumPy array
        gene_data = np.array(sl_sub.X.toarray())  # shape: num_spots x num_genes
        gene_data_list.append(gene_data)
        coords=sl.obsm['spatial_image_coor']  # your custom function
        label = sl.obs['original_clusters'].cat.codes.to_numpy()
        coords_list.append(coords)
        label_list.append(label)

    # 3. Concatenate all gene data
    combined_data = np.vstack(gene_data_list)  # shape: (sum_of_all_spots, num_genes)

    # 4. Reduce dimensionality (e.g., PCA)
    reduced_data = reduce_gene_reads(
        combined_data,
        method='pca',
        n_components=10
    )  # shape: (sum_of_all_spots, 15)
    reduced_data = channelwise_min_max_normalize(reduced_data)

    index_start = 0
    for i, data_slice in enumerate(gene_data_list):
        sl = slices[i]
        image = sl.uns["image_array"]

        num_spots = data_slice.shape[0]
        index_end = index_start + num_spots

        # Slice out the portion that belongs to this slice
        reduced_slice_data = reduced_data[index_start:index_end, :]
        index_start = index_end
        coords = coords_list[i]
        labels = label_list[i]

        for level in [1,2,3,4,5]:
            # 1) create a stronger degraded H&E (your ablation)
            warped_img = degrade_he_for_ablation(image, level=0.25+0.05*level, seed=None)

            #visualization
            # plt.figure(figsize=(12,6))
            # plt.subplot(1, 2, 1)
            # plt.imshow(image)
            # # plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', s=20)
            # plt.title('Original')
            # plt.subplot(1, 2, 2)
            # plt.imshow(warped_img)
            # # plt.scatter(warped_coords[:, 0], warped_coords[:, 1], c=labels, cmap='tab10', s=20)
            # plt.title('Degraded + Deformed')
            # plt.show()

            image =warped_img
            feature_matrix,_ = get_gene_feature_matrix(coords, reduced_slice_data, (512, 512), patch_size=16)
            feature_matrix = remove_salt_pepper(feature_matrix)
            feature_matrix = np.stack([
                cv2.resize(feature_matrix[:, :, i], (64, 64), interpolation=cv2.INTER_NEAREST)
                for i in range(feature_matrix.shape[2])
            ], axis=-1)
            valid_mask = (feature_matrix > 0)
            gene_mask = np.any(valid_mask, axis=-1).astype(valid_mask.dtype)
            plot_dimensional_images_side_by_side(feature_matrix)


            # plt.imsave(f"/media/huifang/data/registration/SCC/huifang/ablation/image/{level}/{k}_{i}_image_512.png", image)
            # np.save(f"/media/huifang/data/registration/SCC/huifang/ablation/image/{level}/{k}_{i}_pca_out.npy", feature_matrix)
            # np.save(f"/media/huifang/data/registration/SCC/huifang/ablation/image/{level}/{k}_{i}_pca_mask.npy", gene_mask)
            # np.savez(f"/media/huifang/data/registration/SCC/huifang/ablation/image/{level}/{k}_{i}_validation", coord=coords, label=labels)
            # np.savez(f"/media/huifang/data/registration/SCC/huifang/ablation/image/{level}/{k}_{i}_field.npz", x=dx,y=dy)










