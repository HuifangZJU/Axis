import os
from pathlib import Path

import numpy as np
import torch

from axis.config import load_config
from axis.registration import (
    NICE_Trans,
    default_gene_transform,
    generate_pairwise_outputs,
    save_results_per_pair,
    unify_group_to_reference,
)
from axis.registration import default_image_transform as get_image_transform
from axis.registration.data import Image_pca_pair
from axis.matching import assign_spatial_axis
from axis.settings import settings
from inference_plots import visualize_pair_data, visualize_unified_coords_with_domains

device = settings.resolve_device()
gene_transforms = default_gene_transform()


def run_network():
    ckpt_path = (
            Path(root_dir)
            / "trained_models"
            / "model_1000.pth"
    )
    test_dataset = Image_pca_pair(
        data_dir=root_dir,
        group_id=group_id,
        sample_id=sample_id,
        base_size=1024,
        image_transformer=get_image_transform(mode='L'),
        gene_transformer=gene_transforms,
        mode="val",
    )

    model = NICE_Trans(in_channels=1, in_gene_channels=10, fusion_module='linear').to(device)
    results = generate_pairwise_outputs(ckpt_path,test_dataset,model,run_hires=False)

    for pair_key, pair_dict in results.items():
        save_results_per_pair(
            root_dir=root_dir,
            group_id=group_id,
            pair_key=pair_key,
            result_dict=pair_dict
        )
    if visualization:
        pair01 = results[(0, 2)]
        visualize_pair_data(pair01)

root_dir =load_config('xenium_mouse_brain').root + "/"
visualization = False
for group_id in [0]:
    sample_id=[0,1,2,4,5]
    # run_network()

    ref_id=0
    paths_all = {
        0: [0,ref_id],
        1: [1,ref_id],
        2: [2,ref_id],
        # 3: [3, ref_id],
        4: [4,5, ref_id],
        5: [5, ref_id],
    }

    ref_data = np.load(root_dir +f"network_input/{group_id}_{ref_id}_validation.npz")
    ref_coords = ref_data["coord"]
    coords_dict = unify_group_to_reference(
        group_dir=f"{root_dir}/pairwise_output/group_{group_id}",
        sample_ids=sample_id,
        paths=paths_all,  # direct mode
        ref_coords=ref_coords,  # so we also save coords_slice0_to_0.npy
        unified_subdir="delta_2.5",  # optional; name the subfolder as you like
    )

    visualize_unified_coords_with_domains(
        root_dir=root_dir,
        group_id=group_id,
        sample_ids=sample_id,
        unified_subdir="delta_2.5",  # must match unify_group_to_reference
        label_npz_suffix="validation.npz",  # or "validation_basesize.npz" if that's what you used
        point_size=4.0,
    )









