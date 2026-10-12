"""Xenium mouse brain (TgCRND8 / WT): DAPI canvases and pseudo-images (Methods, "Alzheimer's disease mouse brain")."""

import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from axis.pp import (
    channelwise_min_max_normalize,
    get_gene_feature_matrix_soft,
    reduce_gene_reads,
    resize_and_flip_image_coords,
)
from axis.settings import data_path
from preprocessing_plots import plot_dimensional_images_side_by_side


def get_xenium_data():
    data_dir = data_path("xenium/mouse_brain/original_data/group_0")
    sample_ids = [["Tg_2.5m", "Tg_5.7m", "Tg_17.9m"], ["WT_2.5m", "WT_5.7m", "WT_13.4m"]]
    image_paths = [["Xenium_V1_FFPE_TgCRND8_2_5_months", "Xenium_V1_FFPE_TgCRND8_5_7_months", "Xenium_V1_FFPE_TgCRND8_17_9_months"],
                  ["Xenium_V1_FFPE_wildtype_2_5_months", "Xenium_V1_FFPE_wildtype_5_7_months", "Xenium_V1_FFPE_wildtype_13_4_months"]]
    groups = []
    for group_data,image_data in zip(sample_ids,image_paths):
        group = []
        for sid, img in zip(group_data,image_data):
            fn = os.path.join(data_dir, f"xenium_{sid}_domains_shared.h5ad")
            adata = sc.read_h5ad(fn)
            # 1. Filter low-quality cells and genes
            sc.pp.filter_cells(adata, min_counts=100)
            sc.pp.filter_genes(adata, min_cells=10)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata.uns['image_path'] = data_path(f"xenium/raw_data/{img}/morphology_down10x.png")
            group.append(adata)
        groups.append(group)
    return groups



def pca_noarmalize_gene_list():
    # # 3. Concatenate all gene data
    combined_data = np.vstack(gene_data_list)  # shape: (sum_of_all_spots, num_genes)
    # 4. Reduce dimensionality (e.g., PCA)
    reduced_data = reduce_gene_reads(
        combined_data,
        method='pca',
        n_components=10
    )  # shape: (sum_of_all_spots, 15)
    reduced_data = channelwise_min_max_normalize(reduced_data)
    return reduced_data



groups = get_xenium_data()
flip_paras = [
    [ [False, False], [False, False], [True, False] ],
    [[False, False], [True, False], [False, False] ]
]
image_size = 1024
for k,slices in enumerate(groups):


    coords_list = []
    image_list = []
    label_list=[]
    gene_data_list = []

    morpho_list = []
    for i, sl in enumerate(slices):
        morph_data = np.array(sl.obsm["X_morphology"])
        morpho_list.append(morph_data)

        gene_data = np.array(sl.X.toarray())  # shape: num_spots x num_genes
        gene_data_list.append(gene_data)

        # cell_type = sl.obs['cell_type'].tolist()
        # label_list.append(cell_type)
        labels = sl.obs["domain_id_shared"].astype("category").cat.codes.to_numpy()
        labels = np.asarray(labels)
        label_list.append(labels)

        coords = sl.obsm['spatial']/10.
        # imagepath = sl.uns['image_path'][:-7]+"20x.png"
        imagepath = sl.uns['image_path']
        image = plt.imread(imagepath)
        resized_image,resized_coords = resize_and_flip_image_coords(image,coords,image_size,flip_paras, k, i)
        coords_list.append(resized_coords)
        image_list.append(resized_image)


    new_label_list = label_list
    new_gene_data = pca_noarmalize_gene_list()


    index_start = 0
    for i, data_slice in enumerate(gene_data_list):
        coords = coords_list[i]
        image = image_list[i]
        labels = new_label_list[i]

        # fig, axes = plt.subplots(1, 3)
        # axes = np.atleast_1d(axes).ravel()
        # for k in range(3):
        #     ax = axes[k]
        #     v = morphofeature[:, k]
        #     sc = ax.scatter(coords[:, 0], coords[:, 1], c=v, s=7, cmap="gray", edgecolors="none")
        #     ax.invert_yaxis()
        #     ax.set_aspect("equal", "box");
        #     ax.axis("off")
        #     ax.set_title(f"Prototype {k + 1}")
        #     plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        # for j in range(k + 1, len(axes)): axes[j].axis("off")
        # plt.tight_layout();
        # plt.show()
        # continue
        num_spots = data_slice.shape[0]
        index_end = index_start + num_spots
        # Slice out the portion that belongs to this slice
        reduced_slice_data = new_gene_data[index_start:index_end, :]
        index_start = index_end
        # Get the corresponding coordinates
        # plt.imshow(image)
        # plt.scatter(coords[:,0],coords[:,1],c=labels,s=5, cmap="tab20")
        # plt.show()

        feature_matrix, gene_mask= get_gene_feature_matrix_soft(coords, reduced_slice_data, (image_size, image_size),
                                                            patch_size=8)


        # feature_matrix = bilateral_smoothing(feature_matrix)
        # feature_matrix = remove_salt_pepper(feature_matrix)
        feature_matrix[feature_matrix < 0.01] = 0
        valid_mask = (feature_matrix > 0)
        gene_mask = np.any(valid_mask, axis=-1).astype(valid_mask.dtype)
        plot_dimensional_images_side_by_side(feature_matrix)

        # imageio.imwrite(
        #     f"$AXIS_DATA_DIR/xenium/{k}_{i}_image_{image_size}.png",
        #     image.astype(np.uint8)  # scale to 8-bit if image is 0–1 float
        # )
        # plt.imsave("$AXIS_DATA_DIR/xenium/" + str(k) + "_" + str(i) + f"_image_{image_size}.png", image)
        # np.save("$AXIS_DATA_DIR/xenium/" + str(k) + "_" + str(i) + f"_pca_out_{image_size}.npy", feature_matrix)
        # np.save("$AXIS_DATA_DIR/xenium/" + str(k) + "_" + str(i) + f"_pca_mask_{image_size}.npy", gene_mask)
        # np.savez("$AXIS_DATA_DIR/xenium/" + str(k) + "_" + str(i) + f"_validation_{image_size}", coord=coords,label = labels)


