from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import median_filter
import cv2
from sklearn.decomposition import PCA
import pandas as pd
import scanpy as sc
from pathlib import Path
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
import anndata
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from paste.helper import intersect
from sklearn.decomposition import NMF

def reduce_gene_reads(gene_reads: np.ndarray, method: str = 'pca', n_components: int = 10) -> np.ndarray:

    if not isinstance(gene_reads, np.ndarray):
        raise ValueError("gene_reads must be a NumPy array of shape (n, m).")

    if method.lower() == 'pca':
        # Principal Component Analysis
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(gene_reads)

    elif method.lower() == 'umap':
        # UMAP
        import umap.umap_ as umap
        reducer = umap.UMAP(n_components=n_components)
        reduced_data = reducer.fit_transform(gene_reads)
    else:
        raise ValueError("method must be one of ['pca', 'umap'].")

    return reduced_data



def remove_salt_pepper(feature_matrix, size=2):
    filtered = np.zeros_like(feature_matrix)
    for c in range(feature_matrix.shape[2]):
        filtered[..., c] = median_filter(feature_matrix[..., c], size=size)
    return filtered

def channelwise_min_max_normalize(data):
    """
    data: shape (N, C)
      N = number of samples (e.g., 17952)
      C = number of channels/features (e.g., 10)

    Returns:
      normalized_data of the same shape, where each channel is mapped to [0, 1]
      across the N samples.
    """
    mins = data.min(axis=0)   # shape (C,)
    maxs = data.max(axis=0)   # shape (C,)
    ranges = maxs - mins

    # Avoid division by zero (when all values in a channel are the same)
    ranges[ranges == 0] = 1e-8

    normalized_data = (data - mins) / ranges
    return normalized_data

def get_gene_feature_matrix(coords: np.ndarray,
                            reduced_data: np.ndarray,
                            image_size=(2016, 2016),
                            patch_size=32):            # how many smoothing passes if local
    """
    Build a (H/ps, W/ps, D) patch feature matrix. Empty patches can be filled to soften borders.

    fill_mode:
      - "global_mean": fill empties with per-feature global mean across valid patches.
      - "local_mean" : iteratively fill empties with mean of 8-neighbors (per feature).
    """
    if coords.shape[0] != reduced_data.shape[0]:
        raise ValueError("coords and reduced_data must have the same number of rows.")

    n_spots, n_dims = reduced_data.shape
    height, width = image_size
    out_height = height // patch_size
    out_width  = width  // patch_size

    sum_array = np.zeros((out_height, out_width, n_dims), dtype=float)
    count_array = np.zeros((out_height, out_width), dtype=int)

    # 1) bin spots to patches
    for i in range(n_spots):
        x, y = coords[i]  # expected pixel coords
        px = int(x) // patch_size
        py = int(y) // patch_size
        if 0 <= px < out_width and 0 <= py < out_height:
            sum_array[py, px, :] += reduced_data[i]
            count_array[py, px] += 1

    # 2) per-patch averages where we have data
    patch_matrix = np.zeros_like(sum_array)
    valid_mask = (count_array > 0)
    patch_matrix[valid_mask, :] = sum_array[valid_mask, :] / count_array[valid_mask, None]
    valid_mask = valid_mask.astype(np.uint8)
    return patch_matrix, valid_mask


def get_gene_feature_matrix_soft(
    coords: np.ndarray,
    reduced_data: np.ndarray,
    image_size =(2016, 2016),
    patch_size: int = 32,
    kernel: str = "gaussian",         # "gaussian" or "inverse_distance"
    sigma: float = None,              # if None, defaults to patch_size / 2 for gaussian
    eps: float = 1e-6,                # small constant for inverse-distance
    normalize_per_cell: bool = True   # normalize 9 weights to sum=1 per cell
):
    """
    Soft-assign each cell's vector to its 3x3 neighborhood of grid patches,
    with weights based on distance to patch centers.

    Returns:
        patch_matrix: (H_out, W_out, n_dims) averaged features per patch
        valid_mask:   (H_out, W_out) mask where weight > 0
    """
    if coords.shape[0] != reduced_data.shape[0]:
        raise ValueError("coords and reduced_data must have the same number of rows.")

    n_spots, n_dims = reduced_data.shape
    height, width = image_size
    out_height = height // patch_size
    out_width  = width  // patch_size

    if kernel not in {"gaussian", "inverse_distance"}:
        raise ValueError("kernel must be 'gaussian' or 'inverse_distance'.")

    if sigma is None and kernel == "gaussian":
        sigma = patch_size / 2.0

    # Accumulators
    sum_array   = np.zeros((out_height, out_width, n_dims), dtype=np.float64)
    weight_sum  = np.zeros((out_height, out_width), dtype=np.float64)

    # Precompute neighbor offsets (3x3 neighborhood)
    nbr_offsets = [(-1,-1), (-1,0), (-1,1),
                   ( 0,-1), ( 0,0), ( 0,1),
                   ( 1,-1), ( 1,0), ( 1,1)]

    for i in range(n_spots):
        x, y = coords[i]  # pixel coordinates (float)
        if not (0 <= x < width and 0 <= y < height):
            continue  # skip cells outside the image bounds

        # Patch indices of the cell's containing grid
        px = int(x) // patch_size
        py = int(y) // patch_size

        # Collect neighbor centers and distances
        neigh_idxs = []
        dists = []

        for dy, dx in nbr_offsets:
            ny = py + dy
            nx = px + dx
            if 0 <= nx < out_width and 0 <= ny < out_height:
                # center of the neighbor patch
                cx = (nx + 0.5) * patch_size
                cy = (ny + 0.5) * patch_size
                dist = np.hypot(x - cx, y - cy)
                neigh_idxs.append((ny, nx))
                dists.append(dist)

        if not neigh_idxs:
            continue

        dists = np.asarray(dists, dtype=np.float64)

        # Compute weights
        if kernel == "gaussian":
            # w = exp(-d^2 / (2*sigma^2))
            denom = 2.0 * (sigma ** 2) + 1e-12
            w = np.exp(-(dists ** 2) / denom)
        else:  # inverse_distance
            w = 1.0 / (dists + eps)

        if normalize_per_cell:
            s = w.sum()
            if s > 0:
                w = w / s

        # Accumulate
        v = reduced_data[i].astype(np.float64)
        for (ny, nx), ww in zip(neigh_idxs, w):
            if ww <= 0:
                continue
            sum_array[ny, nx, :] += ww * v
            weight_sum[ny, nx]   += ww

    # Average per patch where weight > 0
    patch_matrix = np.zeros_like(sum_array, dtype=np.float64)
    valid_mask = weight_sum > 0
    patch_matrix[valid_mask, :] = sum_array[valid_mask, :] / weight_sum[valid_mask, None]

    # Cast back to float32 if you prefer
    patch_matrix = patch_matrix.astype(np.float32)
    valid_mask = valid_mask.astype(np.uint8)
    return patch_matrix,valid_mask

def visualize_spatial_features(
    coords,
    features,
    normalize=False,
    cmap="viridis",
    point_size=1,
    figsize=(18, 7),
    share_color_scale=False,
    save_path=None,
    X_LIM=None,
    Y_LIM=None
):
    """
    Visualize 10 spatial feature maps as scatter plots.

    Parameters
    ----------
    coords : np.ndarray
        Shape (n, 2), spatial coordinates.
    features : np.ndarray
        Shape (n, 10), feature matrix.
    normalize : bool
        If True, min-max normalize each feature independently.
    cmap : str
        Matplotlib colormap.
    point_size : int
        Scatter point size.
    figsize : tuple
        Figure size.
    share_color_scale : bool
        If True, use global min/max across all features.
    save_path : str or None
        If provided, save figure to this path.
    """

    assert coords.shape[1] == 2, "coords must be (n, 2)"
    assert features.shape[0] == coords.shape[0], "coords and features must match in n"
    assert features.shape[1] == 10, "features must be (n, 10)"

    coords = np.asarray(coords)
    features = np.asarray(features)

    # Optional normalization
    if normalize:
        f_min = features.min(axis=0, keepdims=True)
        f_max = features.max(axis=0, keepdims=True)
        features = (features - f_min) / (f_max - f_min + 1e-8)

    # Global color scale if needed
    if share_color_scale:
        global_min = features.min()
        global_max = features.max()
    else:
        global_min = None
        global_max = None

    fig, axes = plt.subplots(2, 5, figsize=figsize)
    axes = axes.flatten()

    for i in range(10):
        ax = axes[i]

        vmin = global_min if share_color_scale else features[:, i].min()
        vmax = global_max if share_color_scale else features[:, i].max()

        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=features[:, i],
            cmap=cmap,
            s=point_size,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(f"Feature {i}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(0,X_LIM)
        ax.set_ylim(0,Y_LIM)
        ax.invert_yaxis()
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_dimensional_images_side_by_side(patch_matrix: np.ndarray,ncols=5):
    # 3) Plot each dimension in a subplot
    n_dims = patch_matrix.shape[-1]
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)

    # Flatten axes so we can index easily
    axes_flat = axes.flatten()

    for dim_idx in range(n_dims):
        ax = axes_flat[dim_idx]

        # Extract the 2D patch grid for this dimension
        patch_image = patch_matrix[:, :, dim_idx]

        im = ax.imshow(
            patch_image,           # shape (out_height, out_width)
            origin='upper',        # row=0 at the top
            cmap='viridis',        # or 'gray', etc.
            aspect='auto'
        )
        ax.set_title(f"Dimension {dim_idx}")
        ax.set_xlabel("Patch (x)")
        ax.set_ylabel("Patch (y)")
        ax.axis('equal')
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots if n_dims < nrows*ncols
    for dim_idx in range(n_dims, nrows*ncols):
        axes_flat[dim_idx].axis("off")

    plt.tight_layout()
    plt.show()

def bilateral_smoothing(feature_matrix,weight=0.01):
    if feature_matrix.max()>1:
        feature_matrix = (feature_matrix-feature_matrix.min())/feature_matrix.max()
    preprocessed = np.zeros_like(feature_matrix)
    for i in range(feature_matrix.shape[2]):
        channel = feature_matrix[..., i].astype(np.float32)
        channel_bilat = cv2.bilateralFilter(
            channel,  # source image
            d=5,  # diameter of the pixel neighborhood
            sigmaColor=0.1,  # range sigma for color
            sigmaSpace=25  # range sigma for spatial distance
        )
        channel_bilat = channel_bilat.astype(np.float64)
        channel_denoised = denoise_tv_chambolle(
            channel_bilat,
            weight=weight,  # small weight for gentle smoothing
            eps=1e-4,
        )

        preprocessed[..., i] = channel_denoised
    return preprocessed

def load_gene_expression(adata, genes):
    gene_expr = adata[:, genes].X
    if hasattr(gene_expr, 'toarray'):
        gene_expr = gene_expr.toarray()

    # Double log1p transform
    gene_expr = np.log1p(np.log1p(gene_expr))

    # Min-max normalization per gene
    gene_expr_norm = (gene_expr - gene_expr.min(axis=0)) / (gene_expr.max(axis=0) - gene_expr.min(axis=0) + 1e-6)
    return gene_expr_norm

def plot_marker_genes(adata, genes):
    n = len(genes)
    gene_expr_norm = load_gene_expression(adata, genes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)

    for i, gene in enumerate(genes):
        ax = axes[i] if n > 1 else axes
        expr = gene_expr_norm[:, i]
        cmap = plt.cm.Reds
        colors = cmap(expr)
        colors[:, -1] = expr  # set alpha as expression level

        ax.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1],
                   color=colors, s=10)
        ax.set_title(gene,fontsize=26)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    plt.show()




def crop_square_then_resize_square(
    img: Image.Image,
    original_uv: np.ndarray,
    crop_para,
    final_size
):
    left,top,side_length = crop_para
    right  = left + side_length
    bottom = top  + side_length
    crop_box = (left, top, right, bottom)
    cropped_img = img.crop(crop_box)
    cropped_uv = original_uv - np.array([left, top])  # shape (N, 2)

    standard_img = cropped_img.resize((10000, 10000), Image.LANCZOS)
    standard_scale = 10000/side_length
    standard_uv =cropped_uv * standard_scale

    final_img = cropped_img.resize((final_size, final_size), Image.LANCZOS)
    scale = final_size / side_length
    final_uv = cropped_uv * scale
    # return cropped_img,final_img, cropped_uv, final_uv
    return standard_img,final_img,standard_uv,final_uv


def save_network_input(out_dir,prefix,image,feature_matrix,gene_mask,coords,labels,suffix="network_input"):
    out_dir = Path(out_dir)
    network_io_dir = out_dir / suffix
    network_io_dir.mkdir(parents=True, exist_ok=True)
    base = network_io_dir / prefix  # common base path
    # Save files
    plt.imsave(f"{base}_image.png", image)
    np.save(f"{base}_pca_out.npy", feature_matrix)
    np.save(f"{base}_pca_mask.npy", gene_mask)
    np.savez(f"{base}_validation.npz", coord=coords, label=labels)
    print(f"Saved: {base}")

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
    adatas = {sample: sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
    for id in sample_list:
        adatas[id].obs['position'].index = (
            adatas[id].obs['position'].index
            .str.replace(r"\.\d+$", "", regex=True)
        )
        position_prefix = '/media/huifang/data/registration/humanpilot/{0}/spatial/tissue_positions_list'.format(id)
        spatial_locations = get_10x_spatial_locations(adatas[id],position_prefix)
        adatas[id].obsm['spatial'] = spatial_locations
        if save:
            adatas[id].write("/media/huifang/data/registration/humanpilot/precessed_feature_matrix/"+id+".h5ad")
            print("saved")


def load_scc_layer(patient, sample, metadata,crop_para,save=True):
    """
    Return Layer object of Patient, Sample
    """
    data_dir = "/media/huifang/data/registration/SCC/"
    layer_path = data_dir + f"/paste/scc_p{patient}_layer{sample}"
    layer = layer_path + ".tsv"
    coor_path = layer_path + "_coordinates.tsv"
    adata = anndata.read_csv(layer, delimiter="\t")

    # Data pre-processing
    coor = pd.read_csv(coor_path, sep="\t").iloc[:, :2]
    original_coor = pd.read_csv(coor_path, sep="\t").iloc[:, 4:]
    coor_index = []
    for pair in coor.values:
        coor_index.append('x'.join(str(e) for e in pair))
    coor.index = coor_index
    original_coor.index = coor_index
    # The metadata, coordinates, and gene expression might have missing cells between them
    idx = intersect(coor_index, adata.obs.index)

    df = metadata[metadata['patient'] == patient]
    df = df[df['sample'] == sample]

    meta_idx = []
    for i in df.index:
        meta_idx.append(i.split('_')[1])
    idx = intersect(idx, meta_idx)

    adata = adata[idx, :]

    image = Image.open(data_dir + f"/paste/scc_p{patient}_layer{sample}.jpg")
    original_coor = np.array(original_coor.loc[idx,:])
    image_size=512
    cropped_img,resized_image,cropped_coor,resized_coor = crop_square_then_resize_square(image,original_coor,crop_para,image_size)
    adata.obsm['spatial_image_coor'] = np.array(resized_coor)
    adata.obsm['spatial'] = np.array(cropped_coor)
    adata.uns["image_array"]=np.array(resized_image)

    metadata_idx = ['P' + str(patient) + '_' + i + '_' + str(sample) for i in idx]
    adata.obs['original_clusters'] = [str(x) for x in list(metadata.loc[metadata_idx, 'SCT_snn_res.0.8'])]

    if save:
        h5ad_dir = Path(data_dir+"/huifang/H5ADs/")
        network_dir = Path(data_dir + "/huifang/network_input/")
        h5ad_dir.mkdir(parents=True, exist_ok=True)
        network_dir.mkdir(parents=True, exist_ok=True)
        cropped_img.save(network_dir / f"{patient}_{sample-1}_hires_image.jpg")
        adata.write(h5ad_dir / f"scc_p{patient}_layer{sample}.h5ad")
        print(f"Saved: scc_p{patient}_layer{sample}")
    return adata



def gen_scc_h5file():
    metadata_path = "/media/huifang/data/registration/SCC/paste/ST_all_metadata.txt"
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

    adata_2_1 = load_scc_layer(2, 1, metadata, (2000,2100,10300))
    adata_2_2 = load_scc_layer(2, 2, metadata, (1800,1200,10400))
    adata_2_3 = load_scc_layer(2, 3, metadata, (2000,2800,10300))
    patient_2 = [adata_2_1, adata_2_2, adata_2_3]

    adata_5_1 = load_scc_layer(5, 1, metadata, (1500,1000,13000))
    adata_5_2 = load_scc_layer(5, 2, metadata, (1000,2000,13500))
    adata_5_3 = load_scc_layer(5, 3, metadata, (1100,2000,13000))
    patient_5 = [adata_5_1, adata_5_2, adata_5_3]

    adata_9_1 = load_scc_layer(9, 1, metadata, (1800,500,13000))
    adata_9_2 = load_scc_layer(9, 2, metadata, (1800,1500,13000))
    adata_9_3 = load_scc_layer(9, 3, metadata, (1800,1800,13000))
    patient_9 = [adata_9_1, adata_9_2, adata_9_3]

    adata_10_1 = load_scc_layer(10, 1, metadata, (2700,3000,10000))
    adata_10_2 = load_scc_layer(10, 2, metadata, (2700,3300,10000))
    adata_10_3 = load_scc_layer(10, 3, metadata, (4000,3600,9000))
    patient_10 = [adata_10_1, adata_10_2, adata_10_3]

    patients = {
        "patient_2": patient_2,
        "patient_5": patient_5,
        "patient_9": patient_9,
        "patient_10": patient_10,
    }
    return patients

