"""DLPFC Visium: canvases, pseudo-images and network inputs (Methods, "Human DLPFC")."""

import json
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from PIL import Image

from axis.config import load_config
from axis.pp import (
    bilateral_smoothing,
    channelwise_min_max_normalize,
    crop_square_then_resize_square,
    get_gene_feature_matrix_soft,
    reduce_gene_reads,
)
from axis.settings import data_path
from preprocessing_plots import plot_dimensional_images_side_by_side

Image.MAX_IMAGE_PIXELS = None

data_dir = data_path('DLPFC/')
out_dir = load_config('dlpfc').root + "/"


# -----------------------------------------------------------------------------
# Helpers used once to attach the full-resolution spot positions of the
# HumanPilot release to the DLPFC h5ad files (kept for provenance).
# -----------------------------------------------------------------------------
def get_10x_spatial_locations(adata,position_prefix):
    try:
        # Try reading as CSV
        positions = pd.read_csv(position_prefix + '.csv', header=None, sep=',')
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        positions = pd.read_csv(position_prefix + '.txt', header=None, sep=',')

    positions.columns = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    ]
    # 1) Get the barcodes from AnnData that are in `positions`
    positions.index = positions["barcode"]
    adata_barcodes = adata.obs['position'].index
    common_barcodes = adata_barcodes[adata_barcodes.isin(positions.index)]
    # 2) Now reindex `positions` in the exact order of `common_barcodes`
    positions_filtered = positions.reindex(common_barcodes)
    spatial_locations = positions_filtered[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()
    return spatial_locations


def updata_dlpfc_spatial(save=False):
    sample_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                   "151675", "151676"]
    adatas = {sample: sc.read_h5ad(data_path('DLPFC/{0}_preprocessed.h5'.format(sample))) for sample in sample_list}
    for id in sample_list:
        adatas[id].obs['position'].index = (
            adatas[id].obs['position'].index
            .str.replace(r"\.\d+$", "", regex=True)
        )
        position_prefix = data_path('humanpilot/{0}/spatial/tissue_positions_list').format(id)
        spatial_locations = get_10x_spatial_locations(adatas[id],position_prefix)
        adatas[id].obsm['spatial'] = spatial_locations
        if save:
            adatas[id].write(data_path("humanpilot/precessed_feature_matrix/")+id+".h5ad")
            print("saved")


def get_DLPFC_data(save=False):
    sample_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                   "151675", "151676"]
    sample_groups = [["151507", "151508", "151509", "151510"], ["151669", "151670", "151671", "151672"],
                     ["151673", "151674", "151675", "151676"]]
    adatas = {sample: sc.read_h5ad(data_dir + 'h5adfile/{0}.h5ad'.format(sample)) for sample in sample_list}
    crop_paras = [(50, 0, 1850), (50, 50, 1900), (50, 120, 1896), (50, 155, 1861),
                  (300, 200, 1650), (350, 220, 1550), (350, 290, 1600), (360, 230, 1600),
                  (160, 10, 1770), (160, 50, 1770), (160, 120, 1750), (180, 20, 1770)]
    image_size = 512
    for index,sampleid in enumerate(sample_list):
        image = Image.open(data_dir + 'original_data/{0}/spatial/tissue_hires_image_image_0.png'.format(sampleid))
        with open(data_dir + 'original_data/{0}/spatial/scalefactors_json.json'.format(sampleid), 'r') as f:
            data = json.load(f)
            low_res_scale = data['tissue_hires_scalef']
        uv_coords = adatas[sampleid].obsm['spatial'] * low_res_scale
        cropped_image, resized_image, _, resized_coor = crop_square_then_resize_square(image, uv_coords, crop_paras[index], image_size)
        adatas[sampleid].obsm['spatial']=resized_coor
        adatas[sampleid].uns['image_array'] = np.asarray(resized_image)

        layer_to_color_map = {'Layer{0}'.format(i + 1): i for i in range(6)}
        layer_to_color_map['WM'] = 6
        labels = list(adatas[sampleid].obs['layer_guess_reordered'].astype(str).map(layer_to_color_map))
        adatas[sampleid].obs['layer_labels'] = labels
        if save:
            h5ad_dir = Path(out_dir) / "h5ad"
            h5ad_dir.mkdir(parents=True, exist_ok=True)
            h5ad_path = h5ad_dir / f"{sampleid}.h5ad"
            adatas[sampleid].write(h5ad_path)
            print(f"Saved: {h5ad_path}")

    layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in
                    range(len(sample_groups))]
    return layer_groups


layer_groups = get_DLPFC_data()

for k,slices in enumerate(layer_groups):
    if k<2:
        continue
    for sl in slices:
        sc.pp.filter_cells(sl, min_genes=200)
        sc.pp.filter_genes(sl, min_cells=3)
        sc.pp.normalize_total(sl, target_sum=1e4)
        sc.pp.log1p(sl)

        # genes=['MFGE8','PCDH8','RORB','COL5A2','NR4A2','MFGE8','PCDH8','RORB','COL5A2','NR4A2']
        # X_loc_gene = sl[:, genes].X
        # if hasattr(X_loc_gene, "toarray"):
        #     X_loc_gene = X_loc_gene.toarray()
        # # plot_marker_genes(sl,genes)
        # visualize_spatial_features(
        #     sl.obsm['spatial'],
        #     X_loc_gene,
        #     normalize=True,
        #     share_color_scale=False,
        #     X_LIM= 512,
        #     Y_LIM = 512
        #
        # )



    all_gene_lists = [sl.var.index for sl in slices]
    common_genes = reduce(np.intersect1d, all_gene_lists)
    gene_data_list = []
    coords_list = []
    image_list = []
    label_list=[]
    for i, sl in enumerate(slices):
        sl_sub = sl[:, common_genes]
        gene_data = np.array(sl_sub.X.toarray())  # shape: num_spots x num_genes
        gene_data_list.append(gene_data)
        labels = np.asarray(sl.obs['layer_labels'])
        label_list.append(labels)
        image_list.append(sl.uns['image_array'])
        coords_list.append(sl.obsm['spatial'])

    # 3. Concatenate all gene data
    combined_data = np.vstack(gene_data_list)  # shape: (sum_of_all_spots, num_genes)
    reduced_data = reduce_gene_reads(
        combined_data,
        method='pca',
        n_components=10
    )  # shape: (sum_of_all_spots, 15)
    reduced_data = channelwise_min_max_normalize(reduced_data)

    index_start = 0
    for i, data_slice in enumerate(gene_data_list):
        num_spots = data_slice.shape[0]
        index_end = index_start + num_spots

        # Slice out the portion that belongs to this slice
        reduced_slice_data = reduced_data[index_start:index_end, :]
        reduced_slice_data = channelwise_min_max_normalize(reduced_slice_data)
        index_start = index_end
        # Get the corresponding coordinates
        coords = coords_list[i]
        image = np.asarray(image_list[i])
        labels = label_list[i]

        #show PCA scatter
        # visualize_spatial_features(
        #     coords,
        #     reduced_slice_data,
        #     normalize=True,
        #     share_color_scale=False,
        #     X_LIM= image.shape[0],
        #     Y_LIM = image.shape[0]
        #
        # )

        feature_matrix, gene_mask = get_gene_feature_matrix_soft(coords, reduced_slice_data, (image.shape[0], image.shape[0]), patch_size=4,sigma=2)
        # print(feature_matrix.shape)
        # test = input()

        plot_dimensional_images_side_by_side(feature_matrix)

        feature_matrix = bilateral_smoothing(feature_matrix,weight=0.1)

        plot_dimensional_images_side_by_side(feature_matrix)

        plt.imshow(gene_mask)
        plt.show()
        # Create the output directory
        # prefix = f"{k}_{i}"
        # save_network_input(out_dir, prefix, image, feature_matrix, gene_mask, coords, labels)