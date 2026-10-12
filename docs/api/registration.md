# `axis.registration` — transition-field learning

::: axis.registration.model
    options:
      members: [TransitionFieldNet, ImageTransitionNet, build_model, load_model, load_state_dict_flexible]

::: axis.registration.losses
    options:
      members: [RegistrationLoss, NCC, MaskedNCC, smoothness_loss]

::: axis.registration.train
::: axis.registration.inference
::: axis.registration.warp
    options:
      members: [find_inverse_warp_coords, transport_points, ensure_flow_2hw, upsample_flow, warp_image, unify_group_to_reference, find_inverse_warp_coords_local]

::: axis.registration.data
    options:
      members: [PairDataset, Image_pca_pair, phenocycler_omics_pair]
