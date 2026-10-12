#!/usr/bin/env python3
from __future__ import annotations
from axis.settings import data_path
from axis.config import load_config

"""Breast-cancer registration evaluation + reporting pipeline.

Modes
-----
eval
    Collect registration metrics and case-study metadata only.
plot
    Read saved results and generate the performance box/dot report and case-study map.
both
    Run evaluation first, then plotting.

The shared method palette and boxplot layout come from ``utils_statistic_plot.py``
so this dataset uses the same visual language as DLPFC, mouse, and SCC.
"""

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from utils_io import (
    as_labels,
    load_competing_method_pair,
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
from utils_statistic_plot import (
    METHOD_DISPLAY,
    SHARED_BOXPLOT_STYLE,
    load_metrics_csv,
    plot_single_pair_metrics_across_methods_row,
)


# =============================================================================
# Dataset / output configuration
# =============================================================================

ROOT = Path(load_config('xenium_breast').root)
COMPETING_ROOT = ROOT / "competing_methods"
LABEL_DIR = ROOT / "huifang" / "network_input"

OUTPUT_DIR = Path(
    data_path("paper_analysis/Figure2_transition_evaluation/g/")
)
METRICS_CSV = OUTPUT_DIR / "breast_metrics_all_methods.csv"
METADATA_DIR = OUTPUT_DIR / "case_metadata"
CONFIG_JSON = OUTPUT_DIR / "breast_evaluation_config.json"
OVERALL_FIGURE = OUTPUT_DIR / "breast_overall_performance.png"
CASE_FIGURE = OUTPUT_DIR / "breast_case_study_2x4.png"

GROUP_IDS = (0,)
PAIR_LIST = ((0, 1),)
SLICE_POSITION_UM = {0: 0.0, 1: 1.0}

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
METHOD_DISPLAY_NAMES = {
    method: METHOD_DISPLAY.get(method, method)
    for method in METHODS
}

DOMAIN_ORDER = tuple(range(1, 21))
ALPHA = 0.01
REMOVE_OUTLIERS = False
NEIGHBOR_RADIUS = None
CORR_GRID_SIZE = 100
CORR_SIGMA = 2.0
DISTANCE_THRESHOLD = 5
STRICT = True

# One slice pair only: report domain-level distributions so the boxplots remain
# informative, matching the prior breast reporting behavior.
OVERALL_METRICS = (
    "dice",
    "spatial_corr",
    "center_shift",
    "neighbor_accuracy",
)

CASE_GROUP_ID = 0
CASE_FIXED_ID = 0
CASE_MOVING_ID = 1
CASE_POINT_SIZE = 1.2
CASE_POINT_ALPHA = 0.28


# =============================================================================
# Evaluation helpers
# =============================================================================

def distance_um(fixed_id: int, moving_id: int) -> float:
    return float(abs(SLICE_POSITION_UM[moving_id] - SLICE_POSITION_UM[fixed_id]))


def _marker_label_path(method: str, group_id: int, sample_id: int) -> Path:
    suffix_by_method = {
        "gpsa": "marker_label_gpsa.npz",
        "paste": "marker_label_paste.npz",
        "santo": "marker_label_santo.npz",
    }
    suffix = suffix_by_method.get(method, "marker_label_img.npz")
    return LABEL_DIR / f"{group_id}_{sample_id}_{suffix}"


def _load_marker_labels(method: str, group_id: int, sample_id: int) -> np.ndarray:
    path = _marker_label_path(method, group_id, sample_id)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if "label" not in data.files:
            raise KeyError(f"{path}: missing array 'label'")
        return as_labels(data["label"], str(path))


def load_pair(method: str, group_id: int, fixed_id: int, moving_id: int):
    coords_fixed, coords_moving, _, _ = load_competing_method_pair(
        COMPETING_ROOT / method / f"group_{group_id}",
        fixed_id,
        moving_id,
    )
    labels_fixed = _load_marker_labels(method, group_id, fixed_id)
    labels_moving = _load_marker_labels(method, group_id, moving_id)
    return validate_pair(
        coords_fixed,
        labels_fixed,
        coords_moving,
        labels_moving,
        source=f"breast/{method}/group_{group_id}/{fixed_id}->{moving_id}",
    )


def evaluate_one_pair(
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
) -> PairEvaluationResult:
    coords_fixed, coords_moving, labels_fixed, labels_moving = load_pair(
        method,
        group_id,
        fixed_id,
        moving_id,
    )

    # Preserve the original breast evaluation behavior for PASTE.
    if method == "paste":
        coords_fixed = coords_fixed[::2]
        coords_moving = coords_moving[::2]
        labels_fixed = labels_fixed[::2]
        labels_moving = labels_moving[::2]

    (
        coords_fixed,
        coords_moving,
        mask_shape,
        normalization,
    ) = normalize_pair_by_fixed_frame(coords_fixed, coords_moving)

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

    return PairEvaluationResult(
        metrics=merge_metric_tables(domain_metrics, neighborhood_metrics),
        per_spot=per_spot,
        coords_fixed=coords_fixed,
        coords_moving=coords_moving,
        labels_fixed=labels_fixed,
        labels_moving=labels_moving,
        regions_fixed=regions_fixed,
        regions_moving=regions_moving,
        normalization=normalization,
    )


# =============================================================================
# Eval mode
# =============================================================================

def evaluate_and_save():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_run_config(
        CONFIG_JSON,
        {
            "dataset": "Xenium breast cancer",
            "root": ROOT,
            "methods": METHODS,
            "groups": GROUP_IDS,
            "pairs": PAIR_LIST,
            "domain_order": DOMAIN_ORDER,
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

    result = collect_evaluation_results(
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
    print(f"[DONE] evaluation results: {METRICS_CSV}")
    print(f"[DONE] case-study metadata: {METADATA_DIR}")
    return result


# =============================================================================
# Plot mode
# =============================================================================

def _require_saved_evaluation() -> None:
    missing = []
    if not METRICS_CSV.is_file():
        missing.append(str(METRICS_CSV))
    if not METADATA_DIR.is_dir():
        missing.append(str(METADATA_DIR))
    if missing:
        raise FileNotFoundError(
            "Missing saved breast evaluation result(s). Run --mode eval first:\n  "
            + "\n  ".join(missing)
        )


def plot_overall_performance():
    df = load_metrics_csv(METRICS_CSV)
    missing = [metric for metric in OVERALL_METRICS if metric not in df.columns]
    if missing:
        raise ValueError(f"Saved breast metrics are missing columns: {missing}")

    return plot_single_pair_metrics_across_methods_row(
        df=df,
        metrics=OVERALL_METRICS,
        group_id=CASE_GROUP_ID,
        fixed_id=CASE_FIXED_ID,
        moving_id=CASE_MOVING_ID,
        save_path=OVERALL_FIGURE,
        style=SHARED_BOXPLOT_STYLE,
        sort_methods_by_median=True,
        share_y=False,
        show=False,
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
        dataset_title="Xenium breast cancer",
        method_display_names=METHOD_DISPLAY_NAMES,
        domain_order=DOMAIN_ORDER,
        point_size=CASE_POINT_SIZE,
        point_alpha=CASE_POINT_ALPHA,
    )


def plot_saved_results():
    _require_saved_evaluation()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_overall_performance()
    # plot_case_study()
    print(f"[DONE] overall performance figure: {OVERALL_FIGURE}")
    print(f"[DONE] case-study figure: {CASE_FIGURE}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Breast registration analysis with eval/plot/both modes."
    )
    parser.add_argument(
        "--mode",
        choices=("eval", "plot", "both"),
        default="plot",
        help="eval: collect/save only; plot: plot saved results only; both: run both.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode in {"eval", "both"}:
        evaluate_and_save()
    if args.mode in {"plot", "both"}:
        plot_saved_results()


if __name__ == "__main__":
    main()
