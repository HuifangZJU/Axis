from matplotlib import pyplot as plt
from network_utils import *
from common_utils import *
from axis.datasets import phenocycler_omics_pair
def visualize_flow(flow, step=50, title="Flow field"):
    """
    Visualize a 2D flow field using a quiver plot.

    Args:
        flow: torch.Tensor or np.ndarray
              Shape [2, H, W] or [1, 2, H, W]
        step: int, subsampling step for arrows
        title: str, plot title
    """
    # Ensure torch tensor on CPU
    if isinstance(flow, np.ndarray):
        flow = torch.from_numpy(flow)
    elif not isinstance(flow, torch.Tensor):
        raise TypeError("flow must be a torch.Tensor or np.ndarray")

    flow = flow.detach().cpu()

    # Remove batch dim if present
    if flow.ndim == 4:
        flow = flow[0]  # [2, H, W]

    assert flow.shape[0] == 2, f"Expected flow shape [2, H, W], got {flow.shape}"

    dy, dx = flow[0], flow[1]  # NOTE: dy = vertical, dx = horizontal
    H, W = dy.shape

    # Subsample grid
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    X, Y = np.meshgrid(xs, ys)  # shapes [H_s, W_s]

    # Subsample flow vectors to match grid
    u = dx[::step, ::step].numpy()  # horizontal
    v = dy[::step, ::step].numpy()  # vertical

    # Optional: auto-scale arrows to avoid looking too tiny/too large
    mag = np.sqrt(u**2 + v**2)
    max_mag = np.max(mag) if mag.size > 0 else 1.0
    if max_mag > 0:
        # normalize to a reasonable range
        u_plot = u / max_mag
        v_plot = v / max_mag
        scale = 0.1  # or tune this
    else:
        u_plot, v_plot, scale = u, v, 1.0

    plt.figure(figsize=(8, 8))
    # If your flow is defined in image coordinates (y downwards),
    # using -v and invert_yaxis keeps directions visually consistent:
    plt.quiver(
        X, Y,
        -u_plot, -v_plot,            # negate to match image coord + invert_yaxis
        angles="xy",
        scale_units="xy",
        scale=scale,
        color="purple",
    )
    plt.gca().invert_yaxis()         # (0,0) at top-left like images
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.show()


model_name = 'all_modalities'
root_dir ="/media/huifang/data/registration/phenocycler/huifang/"
visualization = True
# for group_id in ["LUAD_2_A", "LUAD_3_A", "TSU_24", "TSU_33","TSU_35"]:
for group_id in ["LUAD_2_A","TSU_35"]:
    print(group_id)
    modalities=('protein','xenium','visium')
    # Load checkpoint
    ckpt_path = (
            Path(root_dir)
            / "trained_models"
            / f"phenocycler_{group_id}_{model_name}"
            / "saved_models"
            / "model_1000.pth"
    )

    test_dataset = phenocycler_omics_pair( data_dir= root_dir,data_name=group_id, use_modalities=modalities, image_transformer=get_image_transform('L'),
                           gene_transformer=gene_transforms, mode='val')

    model = NICE_Trans(in_channels=1, in_gene_channels=3, fusion_module='linear').to(device)
    results = generate_pairwise_outputs(ckpt_path,test_dataset,model,run_hires=False)

    # for pair_key, pair_dict in results.items():
    #     save_results_per_pair(
    #         root_dir=root_dir,
    #         group_id=group_id,
    #         pair_key=pair_key,
    #         result_dict=pair_dict
    #     )


    if visualization:
        pair01 = results[("protein", "xenium")]
        fixed_img = pair01["fixed_image"]
        moving_img = pair01["moving_image"]
        warped_img = pair01["warped_image"]

        fixed_coor = pair01["fixed_coor"]
        moving_coor = pair01["moving_coor"]
        warped_coor = pair01["warped_coor"]

        fixed_label = pair01["fixed_label"]
        moving_label = pair01["moving_label"]

        flow = pair01["flow"]

        # hires_warped_img = pair01["warped_hires"]
        # hires_flow = pair01["flow_hires"]
        #
        #
        # # visualize_flow(flow,step=5)
        # # visualize_flow(dense_flow,step=50)
        #
        # plt.imshow(hires_warped_img)
        # plt.show()

        f,a = plt.subplots(2,3,figsize=(12,8))
        # plt.imshow(image_ovelayed)
        a[0,0].imshow(0.5*fixed_img + 0.5*moving_img)
        a[0,1].imshow(moving_img)
        a[0, 1].scatter(moving_coor[:, 0], moving_coor[:, 1],c=moving_label, cmap='tab10', s=5)
        a[0,2].imshow(warped_img)
        a[0, 2].scatter(warped_coor[:, 0], warped_coor[:, 1],c=moving_label, cmap='tab10', s=5)


        a[1,0].imshow( 0.5*fixed_img + 0.5*warped_img)
        a[1,1].scatter(fixed_coor[:, 0], fixed_coor[:, 1], s=5)
        a[1,1].scatter(moving_coor[:, 0], moving_coor[:, 1], s=5)
        # a[1, 1].scatter(fixed_coor[:, 0], fixed_coor[:, 1],  s=10)
        # a[1, 1].scatter(moving_coor[:, 0], moving_coor[:, 1],  s=10)
        a[1,1].invert_yaxis()
        a[1,1].axis('equal')

        a[1,2].scatter(fixed_coor[:,0],fixed_coor[:,1],s=5)
        a[1,2].scatter(warped_coor[:,0],warped_coor[:,1],s=5)
        # a[1, 2].scatter(fixed_coor[:, 0], fixed_coor[:, 1], s=10)
        # a[1, 2].scatter(warped_coor[:, 0], warped_coor[:, 1], s=10)
        a[1,2].invert_yaxis()
        a[1,2].axis('equal')
        plt.show()
    # #





