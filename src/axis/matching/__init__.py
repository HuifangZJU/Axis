"""Spatially matched analysis (Methods, "AXIS: Spatially matched analysis").

* :mod:`~axis.matching.reference` – reconstruction of samples on a reference support
* :mod:`~axis.matching.contrast` – location-wise molecular contrasts
* :mod:`~axis.matching.population` – population-level integration (Sinkhorn + low rank)
* :mod:`~axis.matching.cross_modality` – cross-modality prediction (XGBoost)
"""

from .contrast import (
    SpatialDeltaResult,
    align_spatial_pair,
    compute_spatial_delta,
    compute_spatial_delta_from_files,
    domain_mean_delta_table,
    extract_preprocessed_matrix,
    get_valid_mask,
    transform_expression,
)
from .cross_modality import (
    CrossModalityResult,
    SearchConfig,
    build_stage1_structure_candidates,
    build_stage2_boosting_schedules,
    fit_cross_modality,
    per_target_metrics,
    safe_corr,
    select_xgb_params_two_stage,
)
from .population import integrate_population, spatial_regrid_fuse_visium_gpu_robust
from .reference import assign_spatial_axis, build_analysis_evidence, reconstruct_on_reference

__all__ = [
    "reconstruct_on_reference",
    "build_analysis_evidence",
    "assign_spatial_axis",
    "SpatialDeltaResult",
    "get_valid_mask",
    "transform_expression",
    "extract_preprocessed_matrix",
    "align_spatial_pair",
    "compute_spatial_delta",
    "compute_spatial_delta_from_files",
    "domain_mean_delta_table",
    "integrate_population",
    "spatial_regrid_fuse_visium_gpu_robust",
    "SearchConfig",
    "build_stage1_structure_candidates",
    "build_stage2_boosting_schedules",
    "select_xgb_params_two_stage",
    "safe_corr",
    "per_target_metrics",
    "CrossModalityResult",
    "fit_cross_modality",
]
