#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np

from utils_io import (
    filter_missing_domain_labels,
    load_pair_for_method,
    normalize_pair_by_fixed_frame,
    save_run_config,
    validate_pair,
)
from utils_metrics import (
    PairEvaluationResult,
    collect_evaluation_results,
    evaluate_domain_metrics,
    evaluate_neighborhood_metrics,
    merge_metric_tables,
    plot_case_study_grid,
)


# =============================================================================
# Dataset configuration
# =============================================================================
ROOT = Path("/media/huifang/data/registration/DLPFC")
AXIS_ROOT = ROOT / "huifang"
H5AD_DIR = AXIS_ROOT / "h5ad"
COMPETING_ROOT = ROOT / "competing_methods"
AXIS_NETWORK_INPUT_DIR = AXIS_ROOT / "network_input"
AXIS_PAIRWISE_DIR = AXIS_ROOT / "pairwise_output"

EVALUATION_DIR = Path("/media/huifang/data/registration/evaluation/transition_field/dlpfc/")
METRICS_CSV = EVALUATION_DIR / "dlpfc_metrics_all_methods.csv"
METADATA_DIR = EVALUATION_DIR / "case_metadata"
CASE_FIGURE = EVALUATION_DIR / "dlpfc_case_study_2x4.png"
CONFIG_JSON = EVALUATION_DIR / "dlpfc_evaluation_config.json"

GROUP_IDS = (0, 1, 2)
PAIR_LIST = ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3))
# GROUP_IDS = (1,)
# PAIR_LIST = ( (1, 2), )

SLICE_POSITION_UM = {0: 0.0, 1: 10.0, 2: 310.0, 3: 320.0}
METHODS = (
    "unaligned",
    "simpleitk",
    "voxelmorph",
    "nicetrans",
    "paste",
    "gpsa",
    "santo",
    "axis",
)
COMPETING_METHODS = tuple(m for m in METHODS if m not in {"unaligned", "axis"})
METHOD_DISPLAY_NAMES = {
    "unaligned": "Unaligned",
    "simpleitk": "SimpleITK",
    "voxelmorph": "VoxelMorph",
    "nicetrans": "NiceTrans",
    "paste": "PASTE",
    "gpsa": "GPSA",
    "santo": "SANTO",
    "axis": "AXIS",
}

H5AD_COORD_KEY = "spatial"
H5AD_LABEL_KEY = "layer_guess_reordered"
AXIS_VALIDATION_COORD_KEY = "coord"
AXIS_VALIDATION_LABEL_KEY = "label"
CORR_GRID_SIZE = 100
CORR_SIGMA = 2.0
DISTANCE_THRESHOLD = 50.0
REMOVE_OUTLIERS = True
NEIGHBOR_RADIUS = None
STRICT = True

CASE_GROUP_ID = 1
CASE_FIXED_ID = 1
CASE_MOVING_ID = 2


def distance_um(fixed_id: int, moving_id: int) -> float:
    return float(abs(SLICE_POSITION_UM[moving_id] - SLICE_POSITION_UM[fixed_id]))


def alpha_for_group(group_id: int) -> float:
    return 0.05 if group_id == 0 else 0.01


def load_pair(
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
):
    source = f"DLPFC/{method}/group_{group_id}/{fixed_id}->{moving_id}"

    coords_fixed, coords_moving, labels_fixed, labels_moving = (
        load_pair_for_method(
            method=method,
            group_id=group_id,
            fixed_id=fixed_id,
            moving_id=moving_id,
            competing_methods=COMPETING_METHODS,
            competing_root=COMPETING_ROOT,
            h5ad_dir=H5AD_DIR,
            h5ad_coord_key=H5AD_COORD_KEY,
            h5ad_label_key=H5AD_LABEL_KEY,
            axis_network_input_dir=AXIS_NETWORK_INPUT_DIR,
            axis_pairwise_dir=AXIS_PAIRWISE_DIR,
            axis_validation_coord_key=AXIS_VALIDATION_COORD_KEY,
            axis_validation_label_key=AXIS_VALIDATION_LABEL_KEY,
        )
    )

    # Most methods share the same region-label arrays.
    # GPSA and SANTO use labels stored with their method-specific subsets.
    if method not in {"gpsa", "santo"}:
        labels_fixed = np.load(
            AXIS_NETWORK_INPUT_DIR
            / f"{group_id}_{fixed_id}_region_label.npy"
        )
        labels_moving = np.load(
            AXIS_NETWORK_INPUT_DIR
            / f"{group_id}_{moving_id}_region_label.npy"
        )

    # Remove missing labels before normalization or metric computation.
    # Numeric canonicalization makes GPSA labels such as "4.0" match "4".
    coords_fixed, labels_fixed = filter_missing_domain_labels(
        coords=coords_fixed,
        labels=labels_fixed,
        source=f"{source}: fixed",
        canonicalize_numeric=True,
    )

    coords_moving, labels_moving = filter_missing_domain_labels(
        coords=coords_moving,
        labels=labels_moving,
        source=f"{source}: moving",
        canonicalize_numeric=True,
    )

    return validate_pair(
        coords_fixed,
        labels_fixed,
        coords_moving,
        labels_moving,
        source=source,
    )


def evaluate_one_pair(
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
) -> PairEvaluationResult:
    coords_fixed, coords_moving, labels_fixed, labels_moving = load_pair(
        method, group_id, fixed_id, moving_id
    )
    coords_fixed, coords_moving, mask_shape, normalization = normalize_pair_by_fixed_frame(
        coords_fixed, coords_moving
    )

    domain_metrics, regions_fixed, regions_moving = evaluate_domain_metrics(
        coords_fixed,
        labels_fixed,
        coords_moving,
        labels_moving,
        mask_shape=mask_shape,
        alpha=alpha_for_group(group_id),
        corr_grid_size=CORR_GRID_SIZE,
        corr_sigma=CORR_SIGMA,
        common_labels=None,
        remove_outliers=REMOVE_OUTLIERS,
        neighbor_radius=NEIGHBOR_RADIUS,
    )
    neighborhood_metrics, per_spot = evaluate_neighborhood_metrics(
        coords_fixed,
        labels_fixed,
        coords_moving,
        labels_moving,
        distance_threshold=DISTANCE_THRESHOLD,
    )
    metrics = merge_metric_tables(domain_metrics, neighborhood_metrics)

    return PairEvaluationResult(
        metrics=metrics,
        per_spot=per_spot,
        coords_fixed=coords_fixed,
        coords_moving=coords_moving,
        labels_fixed=labels_fixed,
        labels_moving=labels_moving,
        regions_fixed=regions_fixed,
        regions_moving=regions_moving,
        normalization=normalization,
    )


def collect_all_metrics():
    save_run_config(
        CONFIG_JSON,
        {
            "dataset": "DLPFC",
            "root": ROOT,
            "methods": METHODS,
            "groups": GROUP_IDS,
            "pairs": PAIR_LIST,
            "slice_position_um": SLICE_POSITION_UM,
            "alpha_by_group": {g: alpha_for_group(g) for g in GROUP_IDS},
            "remove_outliers": REMOVE_OUTLIERS,
            "neighbor_radius": NEIGHBOR_RADIUS,
            "corr_grid_size": CORR_GRID_SIZE,
            "corr_sigma": CORR_SIGMA,
            "distance_threshold": DISTANCE_THRESHOLD,
            "normalization": "isotropic fixed-slice scaling",
        },
    )
    return collect_evaluation_results(
        methods=METHODS,
        group_ids=GROUP_IDS,
        pair_list=PAIR_LIST,
        evaluate_pair=evaluate_one_pair,
        out_csv=METRICS_CSV,
        metadata_dir=METADATA_DIR,
        distance_name="distance_um",
        distance_fn=distance_um,
        strict=STRICT,
    )


def plot_case_study():
    return plot_case_study_grid(
        metrics_csv=METRICS_CSV,
        metadata_dir=METADATA_DIR,
        methods=METHODS,
        group_id=CASE_GROUP_ID,
        fixed_id=CASE_FIXED_ID,
        moving_id=CASE_MOVING_ID,
        save_path=CASE_FIGURE,
        dataset_title="DLPFC",
        method_display_names=METHOD_DISPLAY_NAMES,
        domain_order=None,
        point_size=4.0,
        point_alpha=0.50,
    )


def main():
    collect_all_metrics()
    plot_case_study()


if __name__ == "__main__":
    main()
