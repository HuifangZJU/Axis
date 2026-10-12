"""Transition-field learning (Methods, "AXIS: Tissue transition field learning").

* [`model`][axis.registration.model] – multimodal / image-only networks
* [`losses`][axis.registration.losses] – training objective, Eqs. (20)-(23)
* [`data`][axis.registration.data] – pair datasets
* [`train`][axis.registration.train] – training loop
* [`inference`][axis.registration.inference] – predicting and saving fields
* [`warp`][axis.registration.warp] – applying fields to images and points
"""

from .data import PairDataset, cat_mask_with_features, default_gene_transform, default_image_transform
from .inference import generate_pairwise_outputs, predict_field, save_results_per_pair, tensor2img
from .losses import NCC, Grad, MaskedNCC, RegistrationLoss, smoothness_loss
from .model import (
    ImageTransitionNet,
    NICE_Trans,
    NICE_Trans_img,
    TransitionFieldNet,
    build_model,
    load_model,
    load_state_dict_flexible,
    load_state_dict_strict,
    strip_derived_buffers,
)
from .train import save_checkpoint, seed_everything, train_transition_network
from .warp import (
    CoordinateAffineWarper,
    CoordinateWarper,
    ensure_flow_2hw,
    find_inverse_warp_coords,
    find_inverse_warp_coords_local,
    transport_points,
    unify_group_to_reference,
    upsample_flow,
    warp_image,
)

__all__ = [
    "TransitionFieldNet",
    "ImageTransitionNet",
    "NICE_Trans",
    "NICE_Trans_img",
    "build_model",
    "load_model",
    "load_state_dict_flexible",
    "load_state_dict_strict",
    "strip_derived_buffers",
    "NCC",
    "MaskedNCC",
    "Grad",
    "RegistrationLoss",
    "smoothness_loss",
    "PairDataset",
    "cat_mask_with_features",
    "default_image_transform",
    "default_gene_transform",
    "train_transition_network",
    "save_checkpoint",
    "seed_everything",
    "predict_field",
    "generate_pairwise_outputs",
    "save_results_per_pair",
    "tensor2img",
    "find_inverse_warp_coords",
    "find_inverse_warp_coords_local",
    "transport_points",
    "ensure_flow_2hw",
    "upsample_flow",
    "warp_image",
    "unify_group_to_reference",
    "CoordinateWarper",
    "CoordinateAffineWarper",
]
