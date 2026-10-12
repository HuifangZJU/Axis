"""Virtual-tissue generation (Methods, "AXIS: Virtual-tissue generation").

* :mod:`~axis.generation.evidence` – anchor selection, axis weights, evidence planes
* :mod:`~axis.generation.support` – output supports and tissue-support filter
* :mod:`~axis.generation.weighting` – spatial and molecular-quality weights
* :mod:`~axis.generation.generate` – evidence-weighted generation
* :mod:`~axis.generation.meta` – saving the evidence behind a generated tissue
"""

from .evidence import (
    axis_kernel,
    build_query_evidence,
    compute_axis_kernel_weights,
    ensure_same_genes_keep_order,
    get_anchor_pair_for_query,
    infer_axis_bandwidth,
    matrix_log1p,
    relative_position_on_axis,
    stack_attribute_evidence,
    stack_weighted_evidence,
    subset_genes,
)
from .generate import generate_attr_sets, generate_gene_sets, generate_virtual_tissue_fixed_grid, infer_attr_schema_from_df
from .meta import save_generation_meta
from .support import (
    compute_dense_tissue_mask_on_any_grid,
    estimate_grid_spacing,
    infer_grid_spacing,
    keep_largest_connected_component_on_grid,
    make_common_grid_from_coords,
    smooth_mask_on_grid,
)
from .weighting import compute_local_relative_quality_weights, compute_qc_covariates, prepare_local_query_support

__all__ = [
    "relative_position_on_axis",
    "infer_axis_bandwidth",
    "axis_kernel",
    "compute_axis_kernel_weights",
    "get_anchor_pair_for_query",
    "matrix_log1p",
    "ensure_same_genes_keep_order",
    "subset_genes",
    "stack_weighted_evidence",
    "stack_attribute_evidence",
    "build_query_evidence",
    "make_common_grid_from_coords",
    "infer_grid_spacing",
    "estimate_grid_spacing",
    "keep_largest_connected_component_on_grid",
    "smooth_mask_on_grid",
    "compute_dense_tissue_mask_on_any_grid",
    "compute_qc_covariates",
    "compute_local_relative_quality_weights",
    "prepare_local_query_support",
    "infer_attr_schema_from_df",
    "generate_gene_sets",
    "generate_attr_sets",
    "generate_virtual_tissue_fixed_grid",
    "save_generation_meta",
]
