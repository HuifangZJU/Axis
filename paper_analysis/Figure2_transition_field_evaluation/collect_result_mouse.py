#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from utils_io import load_pair_for_method, normalize_pair_by_fixed_frame, save_run_config
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
ROOT = Path("/media/huifang/data/registration/xenium/mouse_brain")
H5AD_DIR = ROOT / "h5ads" / "Tg"
COMPETING_ROOT = ROOT / "competing_methods"
AXIS_NETWORK_INPUT_DIR = ROOT / "network_input"
AXIS_PAIRWISE_DIR = ROOT / "pairwise_output"


EVALUATION_DIR = Path("/media/huifang/data/registration/evaluation/transition_field/mouse/")
METRICS_CSV = EVALUATION_DIR / "mouse_metrics_all_methods.csv"
METADATA_DIR = EVALUATION_DIR / "case_metadata"
CASE_FIGURE = EVALUATION_DIR / "mouse_case_study_2x4.png"
CONFIG_JSON = EVALUATION_DIR / "mouse_evaluation_config.json"

GROUP_IDS = (0,)
PAIR_LIST = ((0, 1), (0, 2), (1, 2),(3, 4), (3, 5), (4, 5))
SAMPLE_AGE = {0: 2.5, 1: 5.7, 2: 17.9, 3: 2.5, 4: 5.7, 5: 13.4}
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

DOMAIN_ORDER = (0, 1, 2, 5, 6, 7, 9, 10, 11)
H5AD_COORD_KEY = "spatial"
H5AD_LABEL_KEY = "domain_id_shared"
AXIS_VALIDATION_COORD_KEY = "coord"
AXIS_VALIDATION_LABEL_KEY = "label"
ALPHA = 0.01
REMOVE_OUTLIERS = True
NEIGHBOR_RADIUS = 5.0
CORR_GRID_SIZE = 100
CORR_SIGMA = 2.0
DISTANCE_THRESHOLD = 10.0
STRICT = False

CASE_GROUP_ID = 0
CASE_FIXED_ID = 1
CASE_MOVING_ID = 2


def distance_age(fixed_id: int, moving_id: int) -> float:
    return float(abs(SAMPLE_AGE[moving_id] - SAMPLE_AGE[fixed_id]))


def load_pair(method: str, group_id: int, fixed_id: int, moving_id: int):
    return load_pair_for_method(
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
        common_labels=DOMAIN_ORDER,
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
            "dataset": "Xenium mouse brain",
            "root": ROOT,
            "methods": METHODS,
            "groups": GROUP_IDS,
            "pairs": PAIR_LIST,
            "sample_age": SAMPLE_AGE,
            "domain_order": DOMAIN_ORDER,
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
        distance_name="distance_age",
        distance_fn=distance_age,
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
        dataset_title="Xenium mouse brain",
        method_display_names=METHOD_DISPLAY_NAMES,
        domain_order=DOMAIN_ORDER,
        point_size=0.55,
        point_alpha=0.20,
        invert_y=False
    )


def main():
    collect_all_metrics()
    # plot_case_study()


if __name__ == "__main__":
    main()
