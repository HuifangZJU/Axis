"""Preprocessing: canvases and molecular pseudo-images (Methods, "Input preparation").

Typical use:

    for adata in adatas:
        axis.pp.prepare_canvas(adata, spatial_key="spatial_image_coor", canvas_size=512)
    axis.pp.build_pseudoimages(adatas, n_comps=10, preset="visium")
"""

from .image import (
    clahe,
    crop_square_then_resize_square,
    degrade_he_for_ablation,
    prepare_canvas,
    random_deform_field,
    resize_and_flip_image_coords,
    to_grayscale,
)
from .pseudoimage import (
    PSEUDOIMAGE_PRESETS,
    bilateral_smoothing,
    build_pseudoimages,
    channelwise_min_max_normalize,
    get_gene_feature_matrix,
    get_gene_feature_matrix_soft,
    rasterize_pseudoimage,
    reduce_gene_reads,
    remove_salt_pepper,
    resize_channels,
)

__all__ = [
    "prepare_canvas",
    "build_pseudoimages",
    "rasterize_pseudoimage",
    "PSEUDOIMAGE_PRESETS",
    "reduce_gene_reads",
    "channelwise_min_max_normalize",
    "get_gene_feature_matrix",
    "get_gene_feature_matrix_soft",
    "remove_salt_pepper",
    "bilateral_smoothing",
    "resize_channels",
    "crop_square_then_resize_square",
    "to_grayscale",
    "clahe",
    "resize_and_flip_image_coords",
    "degrade_he_for_ablation",
    "random_deform_field",
]
