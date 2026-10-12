#!/usr/bin/env python3
from __future__ import annotations
from axis.settings import data_path

"""SCC registration evaluation + reporting pipeline.

Modes
-----
eval
    Collect registration metrics and case-study metadata only.
plot
    Read saved results and generate the overall box/dot report and case-study map.
both
    Run evaluation first, then plotting.
"""

import argparse
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
from utils_statistic_plot import (
    METHOD_DISPLAY,
    SHARED_BOXPLOT_STYLE,
    load_metrics_csv,
    plot_overall_metrics_across_methods_row,
)


# =============================================================================
# Dataset / output configuration
# =============================================================================

ROOT = Path(data_path("SCC"))
COMPETING_ROOT = ROOT / "competing_methods"
AXIS_ROOT = ROOT / "huifang"
AXIS_NETWORK_INPUT_DIR = AXIS_ROOT / "network_input"
AXIS_PAIRWISE_DIR = AXIS_ROOT / "pairwise_output"

OUTPUT_DIR = Path(
    data_path("paper_analysis/Figure2_transition_evaluation/i/")
)
METRICS_CSV = OUTPUT_DIR / "scc_metrics_all_methods.csv"
METADATA_DIR = OUTPUT_DIR / "case_metadata"
CONFIG_JSON = OUTPUT_DIR / "scc_evaluation_config.json"
OVERALL_FIGURE = OUTPUT_DIR / "scc_overall_performance.png"
CASE_FIGURE = OUTPUT_DIR / "scc_case_study_2x4.png"

GROUP_IDS = (2, 5, 9, 10)
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
COMPETING_METHODS = tuple(method for method in METHODS if method != "axis")
METHOD_DISPLAY_NAMES = {
    method: METHOD_DISPLAY.get(method, method)
    for method in METHODS
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

# Preserve the established SCC report: overall neighborhood/spot consistency.
OVERALL_METRICS = ("neighbor_accuracy",)

CASE_GROUP_ID = 2
CASE_FIXED_ID = 0
CASE_MOVING_ID = 1
CASE_POINT_SIZE = 2.4
CASE_POINT_ALPHA = 0.30


# =============================================================================
# Evaluation helpers
# =============================================================================

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
        COMPETING_ROOT / method / f"group_{group_id}",
        fixed_id,
        moving_id,
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
            "Missing saved SCC evaluation result(s). Run --mode eval first:\n  "
            + "\n  ".join(missing)
        )


def plot_overall_performance():
    df = load_metrics_csv(METRICS_CSV)
    missing = [metric for metric in OVERALL_METRICS if metric not in df.columns]
    if missing:
        raise ValueError(f"Saved SCC metrics are missing columns: {missing}")

    return plot_overall_metrics_across_methods_row(
        df=df,
        metrics=OVERALL_METRICS,
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
        dataset_title="SCC",
        method_display_names=METHOD_DISPLAY_NAMES,
        domain_order=None,
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
        description="SCC registration analysis with eval/plot/both modes."
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
