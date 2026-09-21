import matplotlib.pyplot as plt
import scipy.odr
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from functools import reduce
from utils import *
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
import anndata as ad
import pickle

def load_scc_istar_data(istar_root):
    """
    Convert iSTAR super-resolution outputs to AnnData.

    Assumes directory structure:
      {istar_root}/single_input/group_{patient_id}/axis_{j}/sample{j}/cnts-super/*.pickle

    For each patient in [2,5,9,10] and sample j in [0,1,2], this function:
      - loads all gene_name.pickle in cnts-super
      - flattens each 2D gene map into a vector (pixels)
      - constructs AnnData with:
          X: (n_pixels, n_genes)
          var_names: gene names
          obs_names: "p{patient}_s{j}_x{col}_y{row}"
          obsm["spatial"]: pixel coordinates [x, y]
      - returns a dict:
          {
            "patient_2": [adata_2_0, adata_2_1, adata_2_2],
            "patient_5": [...],
            ...
          }
    """
    istar_root = Path(istar_root)

    # patient_ids = [2, 5, 9, 10]
    patient_ids = [2]
    patients = {f"patient_{pid}": [] for pid in patient_ids}

    for pid in patient_ids:
        group_root = istar_root / f"single_input" / f"group_{pid}"
        if not group_root.is_dir():
            print(f"[WARN] Missing group folder: {group_root}, skipping patient {pid}")
            continue

        # samples indexed 0,1,2 for axis_0/1/2
        for j in range(3):
            sample_root = group_root / f"axis_{j}" / f"sample{j}"
            cnts_super_dir = sample_root / "cnts-super"
            if not cnts_super_dir.is_dir():
                print(f"[WARN] Missing cnts-super: {cnts_super_dir}, skipping")
                continue

            # use gene-names.txt if available to define ordering
            gene_names_txt = sample_root / "gene-names.txt"
            if gene_names_txt.is_file():
                with open(gene_names_txt, "r") as f:
                    gene_names = [ln.strip() for ln in f if ln.strip()]
            else:
                # fallback: all *.pickle files in cnts-super
                gene_names = sorted(
                    [p.stem for p in cnts_super_dir.glob("*.pickle")]
                )

            gene_maps = []
            used_genes = []
            grid_shape = None

            for g in gene_names:
                pkl_path = cnts_super_dir / f"{g}.pickle"
                if not pkl_path.is_file():
                    # gene was in gene-names.txt but not actually imputed
                    continue
                with open(pkl_path, "rb") as f:
                    arr = pickle.load(f)
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 3:
                    arr = np.squeeze(arr)
                if grid_shape is None:
                    grid_shape = arr.shape
                elif arr.shape != grid_shape:
                    raise ValueError(
                        f"Shape mismatch for gene {g} in {pkl_path}: "
                        f"{arr.shape} vs expected {grid_shape}"
                    )
                gene_maps.append(arr.reshape(-1))  # flatten to 1D
                used_genes.append(g)

            if not gene_maps:
                print(f"[WARN] No gene maps found in {cnts_super_dir}, skipping")
                continue

            # Assemble expression matrix
            X = np.stack(gene_maps, axis=1)  # (n_pixels, n_genes)
            n_pixels, n_genes = X.shape
            H, W = grid_shape

            # Pixel coordinates (x = column, y = row)
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)  # (n_pixels, 2)

            # obs and var names
            obs_names = [
                f"p{pid}_s{j}_x{int(x)}_y{int(y)}" for (x, y) in coords
            ]
            var_names = used_genes

            adata = ad.AnnData(X=X)
            adata.obs_names = obs_names
            adata.var_names = var_names
            adata.obsm["spatial"] = coords  # [:, 0]=x, [:, 1]=y

            # store some metadata for convenience
            adata.uns["patient_id"] = pid
            adata.uns["sample_index"] = j
            adata.uns["grid_shape"] = grid_shape

            patients[f"patient_{pid}"].append(adata)
            print(
                f"[INFO] Built AnnData for patient {pid}, sample {j}: "
                f"{n_pixels} pixels × {n_genes} genes"
            )

    return patients






root_dir = "/media/huifang/data/registration/SCC/istar/"
patients = load_scc_istar_data(root_dir)
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
        sl_sub = sl[:, common_genes]
        gene_data = np.array(sl_sub.X.toarray())  # shape: num_spots x num_genes
        gene_data_list.append(gene_data)
    combined_data = np.vstack(gene_data_list)  # shape: (sum_of_all_spots, num_genes)
    reduced_data = reduce_gene_reads(
        combined_data,
        method='pca',
        n_components=10
    )  # shape: (sum_of_all_spots, 15)
    reduced_data = channelwise_min_max_normalize(reduced_data)
    index_start = 0
    for i, data_slice in enumerate(gene_data_list):
        sl = slices[i]
        # image = sl.uns["image_array"]
        num_spots = data_slice.shape[0]
        index_end = index_start + num_spots

        # Slice out the portion that belongs to this slice
        reduced_slice_data = reduced_data[index_start:index_end, :]
        index_start = index_end
        # Get the corresponding coordinates
        coords = sl.obsm['spatial']  # your custom function
        # labels = sl.obs['original_clusters'].cat.codes.to_numpy()

        feature_matrix,_ = get_gene_feature_matrix(coords, reduced_slice_data, (720, 720), patch_size=8)
        plot_dimensional_images_side_by_side(feature_matrix)
        # feature_matrix = remove_salt_pepper(feature_matrix)
        # feature_matrix = np.stack([
        #     cv2.resize(feature_matrix[:, :, i], (64, 64), interpolation=cv2.INTER_NEAREST)
        #     for i in range(feature_matrix.shape[2])
        # ], axis=-1)
        # valid_mask = (feature_matrix > 0)
        # gene_mask = np.any(valid_mask, axis=-1).astype(valid_mask.dtype)

        plot_dimensional_images_side_by_side(feature_matrix)
        # plt.imshow(gene_mask)
        # plt.show()
        # pid = k.split("_")[-1]
        # prefix = f"{pid}_{i}"
        # save_network_input(out_dir, prefix, image, feature_matrix, gene_mask, coords, labels)









