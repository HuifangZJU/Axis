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
import numpy as np
import pandas as pd
import anndata
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from pathlib import Path



data_dir = "/media/huifang/data/registration/SCC/"

metadata_path = "/media/huifang/data/registration/SCC/paste/ST_all_metadata.txt"
metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)
pixel_sizes = []
radii = []
for patient in [2,5,9,10]:
    for sample in [1,2,3]:

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

        # 1) nearest-neighbor distance in pixels
        nbrs = NearestNeighbors(n_neighbors=2).fit(original_coor)
        dists, _ = nbrs.kneighbors(original_coor)
        nn_pix = dists[:, 1]  # distance to closest neighbor (exclude self)

        d_pix = float(np.median(nn_pix))  # robust estimate

        # 2) physical center–center distance (from paper) ≈ 150 µm
        center_to_center_um = 150.0

        pixel_size_raw = center_to_center_um / d_pix  # µm per pixel

        # -----------------------------
        # Estimate radius_raw
        # -----------------------------
        # spot diameter ≈ 110 µm → radius = 55 µm
        spot_radius_um = 110.0 / 2.0

        radius_raw = spot_radius_um / pixel_size_raw  # pixels per spot radius

        pixel_sizes.append(pixel_size_raw)
        radii.append(radius_raw)

        print(
            f"Patient {patient}, sample {sample}: "
            f"median NN = {d_pix:.2f} px, "
            f"pixel_size_raw = {pixel_size_raw:.4f} µm/px, "
            f"radius_raw = {radius_raw:.2f} px"
        )

# ------------------------------------
# Aggregate across all SCC slides
# ------------------------------------
pixel_size_raw_global = float(np.median(pixel_sizes))
radius_raw_global = float(np.median(radii))

print("\n=== Aggregated SCC parameters ===")
print(f"pixel_size_raw (median) = {pixel_size_raw_global:.4f} µm/px")
print(f"radius_raw (median)     = {radius_raw_global:.2f} px")

# Save for iStar
out_root = Path(data_dir) / "istar_params"
out_root.mkdir(parents=True, exist_ok=True)

np.savetxt(out_root / "pixel-size-raw.txt", [pixel_size_raw_global])
np.savetxt(out_root / "radius-raw.txt", [radius_raw_global])

print(f"\nSaved pixel-size-raw.txt and radius-raw.txt to {out_root}")