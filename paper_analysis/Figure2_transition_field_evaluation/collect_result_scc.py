#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from utils_io import (
    load_axis_pair,
    load_competing_method_pair,
    normalize_pair_by_fixed_frame,
    save_run_config,
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
ROOT = Path("/media/huifang/data/registration/SCC")
COMPETING_ROOT = ROOT / "competing_methods"
AXIS_ROOT = ROOT / "huifang"
AXIS_NETWORK_INPUT_DIR = AXIS_ROOT / "network_input"
AXIS_PAIRWISE_DIR = AXIS_ROOT / "pairwise_output"

EVALUATION_DIR = Path("/media/huifang/data/registration/evaluation/transition_field/scc/")
METRICS_CSV = EVALUATION_DIR / "scc_metrics_all_methods.csv"
METADATA_DIR = EVALUATION_DIR / "case_metadata"
CASE_FIGURE = EVALUATION_DIR / "scc_case_study_2x4.png"
CONFIG_JSON = EVALUATION_DIR / "scc_evaluation_config.json"

GROUP_IDS = (2, 5,9,10)
PAIR_LIST = ((0, 1), (0, 2), (1, 2))
SLICE_POSITION_UM = {0: 0.0, 1: 5.0, 2: 10.0}
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
COMPETING_METHODS = tuple(m for m in METHODS if m != "axis")
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

AXIS_VALIDATION_COORD_KEY = "coord"
AXIS_VALIDATION_LABEL_KEY = "label"
ALPHA = 0.01
REMOVE_OUTLIERS = True
NEIGHBOR_RADIUS = None
CORR_GRID_SIZE = 100
CORR_SIGMA = 2.0
DISTANCE_THRESHOLD = 40.0
STRICT = True

CASE_GROUP_ID = 2
CASE_FIXED_ID = 0
CASE_MOVING_ID = 1


def distance_um(fixed_id: int, moving_id: int) -> float:
    return float(abs(SLICE_POSITION_UM[moving_id] - SLICE_POSITION_UM[fixed_id]))


def load_pair(method: str, group_id: int, fixed_id: int, moving_id: int):
    if method == "axis":
        return load_axis_pair(
            group_id=group_id,
            fixed_id=fixed_id,
            moving_id=moving_id,
            network_input_dir=AXIS_NETWORK_INPUT_DIR,
            pairwise_dir=AXIS_PAIRWISE_DIR,
            validation_coord_key=AXIS_VALIDATION_COORD_KEY,
            validation_label_key=AXIS_VALIDATION_LABEL_KEY,
        )
    return load_competing_method_pair(
        COMPETING_ROOT / method / f"group_{group_id}", fixed_id, moving_id
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
        alpha=ALPHA,
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
            "dataset": "SCC",
            "root": ROOT,
            "methods": METHODS,
            "groups": GROUP_IDS,
            "pairs": PAIR_LIST,
            "slice_position_um": SLICE_POSITION_UM,
            "alpha": ALPHA,
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
        dataset_title="SCC",
        method_display_names=METHOD_DISPLAY_NAMES,
        domain_order=None,
        point_size=2.4,
        point_alpha=0.30,
    )


def main():
    collect_all_metrics()
    plot_case_study()


if __name__ == "__main__":
    main()
