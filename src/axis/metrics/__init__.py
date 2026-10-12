"""Evaluation metrics.

* :mod:`~axis.metrics.alignment` – transition-field accuracy (Dice, spatial
  correlation, centre shift, molecular-neighbourhood correspondence)
* :mod:`~axis.metrics.generation` – generated / matched tissue fidelity
* :mod:`~axis.metrics.monitor` – light-weight summaries used during training
"""

from .alignment import (
    RegionMeta,
    build_region_metadata,
    evaluate_domain_metrics,
    evaluate_neighborhood_metrics,
    filter_isolated_points,
    merge_metric_tables,
)
from .generation import (
    adjacent_layer_aurocs,
    cluster_composition,
    contrast_agreement,
    direction_invariant_auroc,
    js_distance,
    kruskal_eta_h_squared,
    local_structure_distance,
    masked_ncc,
    min_positive_spot_distance,
    neighborhood_connectivity,
    orientation_histogram,
    safe_pearson,
    shifted_map_ncc,
    symmetric_relative_abs_error,
    total_variation_distance,
    within_cluster_nn_distances,
)

__all__ = [
    "RegionMeta",
    "filter_isolated_points",
    "build_region_metadata",
    "evaluate_domain_metrics",
    "evaluate_neighborhood_metrics",
    "merge_metric_tables",
    "safe_pearson",
    "masked_ncc",
    "shifted_map_ncc",
    "orientation_histogram",
    "cluster_composition",
    "symmetric_relative_abs_error",
    "min_positive_spot_distance",
    "within_cluster_nn_distances",
    "local_structure_distance",
    "neighborhood_connectivity",
    "js_distance",
    "total_variation_distance",
    "kruskal_eta_h_squared",
    "direction_invariant_auroc",
    "adjacent_layer_aurocs",
    "contrast_agreement",
]
