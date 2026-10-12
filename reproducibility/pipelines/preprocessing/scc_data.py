"""Loading the SCC Visium sections (PASTE-format inputs) onto a 512-px canvas.

``gen_scc_h5file`` reads ``$AXIS_DATA_DIR/SCC/paste`` (counts, coordinates,
H&E images and the cluster metadata of Ji et al. 2020), crops every section to
a square around the tissue (crop parameters below), resizes it to 512 px and
writes ``$AXIS_DATA_DIR/SCC/huifang/H5ADs/scc_p{patient}_layer{sample}.h5ad``.
"""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from paste.helper import intersect

from axis.pp import crop_square_then_resize_square
from axis.settings import data_path

Image.MAX_IMAGE_PIXELS = None


def load_scc_layer(patient, sample, metadata,crop_para,save=True):
    """
    Return Layer object of Patient, Sample
    """
    data_dir = data_path("SCC/")
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
    metadata_path = data_path("SCC/paste/ST_all_metadata.txt")
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
