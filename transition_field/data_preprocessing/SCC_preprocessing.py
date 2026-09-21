import matplotlib.pyplot as plt
import scipy.odr
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from functools import reduce
from utils import *
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)


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





gen_scc_h5file()
# print('all done')

out_dir='/media/huifang/data/registration/SCC/huifang/'
patients = load_scc_data(out_dir)

for k, slices in patients.items():
    for sl in slices:
        sc.pp.filter_genes(sl, min_cells=15, inplace=True)
        sc.pp.filter_cells(sl, min_genes=100, inplace=True)
        sc.pp.normalize_total(sl, target_sum=1e4)
        sc.pp.log1p(sl)
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
        # Get the corresponding coordinates
        coords = coords_list[i]
        labels = label_list[i]

        feature_matrix,_ = get_gene_feature_matrix_soft(coords, reduced_slice_data, (512, 512), patch_size=16)
        feature_matrix = remove_salt_pepper(feature_matrix)
        feature_matrix = np.stack([
            cv2.resize(feature_matrix[:, :, i], (64, 64), interpolation=cv2.INTER_NEAREST)
            for i in range(feature_matrix.shape[2])
        ], axis=-1)
        valid_mask = (feature_matrix > 0)
        gene_mask = np.any(valid_mask, axis=-1).astype(valid_mask.dtype)

        plot_dimensional_images_side_by_side(feature_matrix)
        plt.imshow(gene_mask)
        plt.show()


        # pid = k.split("_")[-1]
        # prefix = f"{pid}_{i}"
        # save_network_input(out_dir, prefix, image, feature_matrix, gene_mask, coords, labels)









