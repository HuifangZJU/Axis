from matplotlib import pyplot as plt
from network_utils import *
from common_utils import *
import scanpy as sc
import glob

def assign_spatial_axis(root, group_ids, reference_id=0, key="spatial_axis"):
    """
    Assign aligned coordinates to adata.obsm[key] for each provided group.

    Paths:
        warped coords:
            root/group_[group_id]/[fixed_id]_[moving_id]_warped_coor.npy
        h5ad:
            root/h5ad/[group_id]_[sample_id].h5ad
        network coords:
            root/network_input/[group_id]_[sample_id]_validation.npz
            key: ["coord"]
    """
    h5ad_dir = os.path.join(root, "h5ad")
    net_dir = os.path.join(root, "network_input")

    for group_id in group_ids:
        print(f"\n[INFO] Processing group {group_id}")

        # find all h5ad files in this group
        h5ad_files = sorted(glob.glob(os.path.join(h5ad_dir, f"{group_id}_*.h5ad")))
        if len(h5ad_files) == 0:
            print(f"[WARN] No h5ad files found for group {group_id}")
            continue

        # load reference coords once
        ref_npz = os.path.join(net_dir, f"{group_id}_{reference_id}_validation.npz")
        if not os.path.exists(ref_npz):
            raise FileNotFoundError(f"Reference network coord file not found: {ref_npz}")

        ref_coord = np.load(ref_npz)["coord"].astype(np.float32)

        for h5ad_path in h5ad_files:
            fname = os.path.basename(h5ad_path)
            sample_id = int(fname.replace(".h5ad", "").split("_")[1])

            adata = sc.read_h5ad(h5ad_path)

            if sample_id == reference_id:
                coord = ref_coord
                source = ref_npz
            else:
                warped_path = os.path.join(
                    root,"pairwise_output/"
                    f"group_{group_id}",
                    f"{reference_id}_{sample_id}_warped_coor.npy"
                )
                if not os.path.exists(warped_path):
                    raise FileNotFoundError(f"Warped coord file not found: {warped_path}")
                coord = np.load(warped_path).astype(np.float32)
                source = warped_path

            # keep first two columns only
            coord = coord[:, :2]

            if coord.shape[0] != adata.n_obs:
                raise ValueError(
                    f"Length mismatch in group {group_id}, sample {sample_id}: "
                    f"adata.n_obs={adata.n_obs}, coord.shape={coord.shape}, source={source}"
                )

            adata.obsm[key] = coord
            adata.write(h5ad_path)

            print(
                f"[OK] group={group_id}, sample={sample_id}, "
                f"{key}.shape={coord.shape}, source={os.path.basename(source)}"
            )


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
root_dir ="/media/huifang/data/registration/SCC/huifang/"
visualization = False
#generate_pairwise_result()
assign_spatial_axis(root_dir, [2,5,9,10], reference_id=0)

