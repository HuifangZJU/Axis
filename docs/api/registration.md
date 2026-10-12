# `axis.registration` — transition-field learning

::: axis.registration.model
    options:
      members: [TransitionFieldNet, ImageTransitionNet, build_model, load_model, load_state_dict_flexible, load_state_dict_strict, strip_derived_buffers, NICE_Trans, NICE_Trans_img]

::: axis.registration.losses
    options:
      members: [RegistrationLoss, NCC, MaskedNCC, Grad, smoothness_loss, Regu_loss]

::: axis.registration.train
::: axis.registration.inference
::: axis.registration.warp
    options:
      members: [find_inverse_warp_coords, transport_points, ensure_flow_2hw, upsample_flow, warp_image, warp_highres_image_with_lowres_flow_cpu, unify_group_to_reference, find_inverse_warp_coords_local, CoordinateWarper, CoordinateAffineWarper]

::: axis.registration.data
    options:
      members: [PairDataset, cat_mask_with_features, default_image_transform, default_gene_transform, Image_pca_pair, Image_svg_pair, Image_gene_pair, Image_pair, scc_ablation_pair, Sennet_image_pair, phenocycler_image_pair, phenocycler_omics_pair, augment_image]
