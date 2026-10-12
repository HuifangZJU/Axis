import os
from pathlib import Path
import glob

import numpy as np
import torch
from matplotlib import pyplot as plt
import scanpy as sc

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

device = settings.resolve_device()
gene_transforms = default_gene_transform()


def generate_pairwise_result():

    for group_id in [2,5,9,10]:
        sample_id=[0,1,2]
        # Load checkpoint
        ckpt_path = (
                Path(root_dir)
                / "trained_models"
                / f"scc_patient{group_id}_{model_name}"
                / "saved_models"
                / "model_1000.pth"
        )

        test_dataset = Image_pca_pair(
            data_dir=root_dir,
            group_id=group_id,
            sample_id=sample_id,
            image_transformer=get_image_transform(),
            gene_transformer=gene_transforms,
            mode="val",
        )

        model = NICE_Trans(in_channels=3, in_gene_channels=10, fusion_module='transformer').to(device)
        results = generate_pairwise_outputs(ckpt_path,test_dataset,model,run_hires=True)
        for pair_key, pair_dict in results.items():
            save_results_per_pair(
                root_dir=root_dir,
                group_id=group_id,
                pair_key=pair_key,
                result_dict=pair_dict
            )

        if visualization:
            pair01 = results[(0, 2)]
            fixed_img = pair01["fixed_image"]
            moving_img = pair01["moving_image"]
            warped_img = pair01["warped_image"]

            fixed_coor = pair01["fixed_coor"]
            moving_coor = pair01["moving_coor"]
            warped_coor = pair01["warped_coor"]

            fixed_label = pair01["fixed_label"]
            moving_label = pair01["moving_label"]

            flow = pair01["flow"]

            hires_warped_img = pair01["warped_hires"]
            hires_flow = pair01["flow_hires"]


            # visualize_flow(flow,step=5)
            # visualize_flow(dense_flow,step=50)

            plt.imshow(hires_warped_img)
            plt.show()

            f,a = plt.subplots(2,3,figsize=(12,8))
            # plt.imshow(image_ovelayed)
            a[0,0].imshow(0.5*fixed_img + 0.5*moving_img)
            a[0,1].imshow(moving_img)
            a[0, 1].scatter(moving_coor[:, 0], moving_coor[:, 1],c=moving_label, cmap='tab10', s=5)
            a[0,2].imshow(warped_img)
            a[0, 2].scatter(warped_coor[:, 0], warped_coor[:, 1],c=moving_label, cmap='tab10', s=5)


            a[1,0].imshow( 0.5*fixed_img + 0.5*warped_img)
            a[1,1].scatter(fixed_coor[:, 0], fixed_coor[:, 1],c=fixed_label, cmap='tab10', s=5)
            a[1,1].scatter(moving_coor[:, 0], moving_coor[:, 1],c=moving_label, cmap='tab10', s=5)
            # a[1, 1].scatter(fixed_coor[:, 0], fixed_coor[:, 1],  s=10)
            # a[1, 1].scatter(moving_coor[:, 0], moving_coor[:, 1],  s=10)
            a[1,1].invert_yaxis()
            a[1,1].axis('equal')

            a[1,2].scatter(fixed_coor[:,0],fixed_coor[:,1],c=fixed_label, cmap='tab10', s=5)
            a[1,2].scatter(warped_coor[:,0],warped_coor[:,1],c=moving_label, cmap='tab10', s=5)
            # a[1, 2].scatter(fixed_coor[:, 0], fixed_coor[:, 1], s=10)
            # a[1, 2].scatter(warped_coor[:, 0], warped_coor[:, 1], s=10)
            a[1,2].invert_yaxis()
            a[1,2].axis('equal')
            plt.show()



model_name = 'img_win9_gene_win5_frac0.95_gene0.5_generegu0'
root_dir =load_config('scc').root + "/"
visualization = False
#generate_pairwise_result()
assign_spatial_axis(root_dir, [2,5,9,10], reference_id=0)

