#!/usr/bin/env python3
"""
Collect per-domain DLPFC registration metrics for all methods and save to CSV.

Output columns:
    group_id
    fixed_id
    moving_id
    distance_um
    method
    domain
    dice
    spatial_corr
    center_shift
"""

from __future__ import annotations


from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from io_utils import normalize_pair_by_fixed_frame
from io_utils import load_axis_pair,load_competing_method_pair
from metrics_utils import aggregate_domain_metrics,visualize_dice_regions,visualize_centroid_shift,visualize_metric_scores,evaluate_domain_metrics,standardize_metrics_df
import inspect

# =========================
# Configuration
# =========================
ROOT = Path("/media/huifang/data/registration/xenium/mouse_brain")

GROUP_IDS = [0]
PAIR_LIST = [(0, 1), (0, 2),(1, 2)]


# Slice positions for DLPFC
SLICE_POS_UM = {
    0: 2.5,
    1: 5.7,
    2: 17.9,
    # 3: 2.5,
    # 4: 5.7,
    # 5: 13.4,
}

COMPETING_METHODS = [
    "unaligned",
    "simpleitk",
    "voxelmorph",
    # "nicetrans",
    "paste",
    "gpsa",
    "santo",
]

ALL_METHODS = COMPETING_METHODS + ["axis"]
# ALL_METHODS = ["axis"]

# Unaligned baseline
H5AD_DIR = ROOT / "h5ads/Tg"
H5AD_COORD_KEY = "spatial"
H5AD_LABEL_KEY = "domain_id_shared"

# Competing methods
COMPETING_ROOT = ROOT / "competing_methods"

# AXIS
AXIS_ROOT = ROOT
AXIS_PAIRWISE_DIR = AXIS_ROOT / "pairwise_output"
AXIS_NETWORK_INPUT_DIR = AXIS_ROOT / "network_input"

# Output
OUT_CSV = ROOT / "evaluation" / "mouse_time_domain_metrics_all_methods.csv"
OUT_FIG = ROOT / "evaluation" / "overlay_metric_plots"


STRICT = True  # set False to skip missing pairs instead of failing
CORR_GRID_SIZE = 100
CORR_SIGMA = 2.0

def compute_distance_um(fixed_id: int, moving_id: int) -> float:
    return float(abs(SLICE_POS_UM[moving_id] - SLICE_POS_UM[fixed_id]))

def load_pair_for_method(
    H5AD_DIR,
    COMPETING_METHODS,
    COMPETING_ROOT,
    H5AD_COORD_KEY,
    H5AD_LABEL_KEY,
    AXIS_PAIRWISE_DIR,
    AXIS_NETWORK_INPUT_DIR,
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if method in COMPETING_METHODS:
        method_dir = COMPETING_ROOT / method / f"group_{group_id}"
        return load_competing_method_pair(method_dir, fixed_id, moving_id)

    if method == "axis":
        return load_axis_pair(group_id, fixed_id, moving_id,AXIS_NETWORK_INPUT_DIR,AXIS_PAIRWISE_DIR)

    raise ValueError(f"Unknown method: {method}")


def call_domain_evaluator(
    coords_fixed: np.ndarray,
    labels_fixed: np.ndarray,
    coords_moving: np.ndarray,
    labels_moving: np.ndarray,
    alpha: float,
    mask_shape: Tuple[int, int],
    common_labels=None,
    neighbor_radius=None
):
    """
    Call the user's unified evaluator in a robust way.

    Expected evaluator behavior:
      - returns a DataFrame, or
      - returns a tuple/list whose first element is that DataFrame
    """
    sig = inspect.signature(evaluate_domain_metrics)

    kwargs = {}
    for k, v in {
        "mask_shape": mask_shape,
        "alpha": alpha,
        "corr_grid_size": CORR_GRID_SIZE,
        "corr_sigma": CORR_SIGMA,
        "common_labels": common_labels,
        "neighbor_radius":neighbor_radius
    }.items():
        if k in sig.parameters:
            kwargs[k] = v

    result = evaluate_domain_metrics(
        coords_fixed,
        labels_fixed,
        coords_moving,
        labels_moving,
        **kwargs,
    )
    return result

# =========================
# Main wrapper
# =========================
def evaluate_one_pair(
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
) :
    pts1, pts2, label1, label2 = load_pair_for_method(
        H5AD_DIR,
        COMPETING_METHODS,
        COMPETING_ROOT,
        H5AD_COORD_KEY,
        H5AD_LABEL_KEY,
        AXIS_PAIRWISE_DIR,
        AXIS_NETWORK_INPUT_DIR,
        method=method,
        group_id=group_id,
        fixed_id=fixed_id,
        moving_id=moving_id
    )

    alpha_value = 0.1

    pts1_norm, pts2_norm, mask_shape = normalize_pair_by_fixed_frame(pts1, pts2)


    result = call_domain_evaluator(
        coords_fixed=pts1_norm,
        labels_fixed=label1,
        coords_moving=pts2_norm,
        labels_moving=label2,
        mask_shape=mask_shape,
        alpha=alpha_value,
        common_labels=[0, 1, 2, 5, 6, 7, 9, 10, 11],
        neighbor_radius=5
    )

    return result


def collect_all_metrics(
    root: Path = ROOT,
    group_ids: Iterable[int] = GROUP_IDS,
    pair_list: Iterable[Tuple[int, int]] = PAIR_LIST,
    methods: Iterable[str] = ALL_METHODS,
    out_csv: Path = OUT_CSV,
    strict: bool = STRICT,
) -> pd.DataFrame:
    rows = []

    for method in methods:
        for group_id in group_ids:
            for fixed_id, moving_id in pair_list:
                print(f"[INFO] method={method}, group={group_id}, pair={fixed_id}->{moving_id}")

                try:
                    result = evaluate_one_pair(
                        method=method,
                        group_id=group_id,
                        fixed_id=fixed_id,
                        moving_id=moving_id,
                    )
                    metrics_df = result[0]
                    metrics_df = standardize_metrics_df(metrics_df)
                    metrics_df["group_id"] = int(group_id)
                    metrics_df["fixed_id"] = int(fixed_id)
                    metrics_df["moving_id"] = int(moving_id)
                    metrics_df["distance_um"] = compute_distance_um(fixed_id, moving_id)
                    metrics_df["method"] = method

                    metrics_df = metrics_df[
                        [
                            "group_id",
                            "fixed_id",
                            "moving_id",
                            "distance_um",
                            "method",
                            "domain",
                            "dice",
                            "spatial_corr",
                            "center_shift",
                        ]
                    ].copy()

                    if metrics_df.empty:
                        print(
                            f"[WARN] Empty metrics for method={method}, group={group_id}, "
                            f"pair={fixed_id}->{moving_id}"
                        )
                    else:
                        rows.append(metrics_df)

                except Exception as e:
                    msg = (
                        f"[ERROR] Failed for method={method}, group={group_id}, "
                        f"pair={fixed_id}->{moving_id}: {e}"
                    )
                    if strict:
                        raise RuntimeError(msg) from e
                    print(msg)

    if len(rows) == 0:
        raise RuntimeError("No evaluation rows were collected.")

    out_df = pd.concat(rows, axis=0, ignore_index=True)

    out_df = out_df.sort_values(
        by=["method", "group_id", "fixed_id", "moving_id", "domain"]
    ).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"[OK] Saved CSV to: {out_csv}")
    print(f"[OK] Total rows: {len(out_df)}")

    return out_df


# ============================================================
# Visualization wrapper
# ============================================================

def visualize_one_method_overlay(
    method,
    group_id,
    fixed_id,
    moving_id,
    metric="dice",
    outdir=OUT_FIG,
):
    """
    metric:
        "dice"
        "center_shift"
        "both"
    """
    result = evaluate_one_pair(
        method=method,
        group_id=group_id,
        fixed_id=fixed_id,
        moving_id=moving_id,
    )

    metrics_df, region_a, region_b = result[:3]

    summary = aggregate_domain_metrics(metrics_df, weighted=False)

    save_suffix = f"{method}_g{group_id}_{fixed_id}_{moving_id}"

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if metric == "dice":
        visualize_dice_regions(
            region_A=region_a,
            region_B=region_b,
            metrics_df=metrics_df,
            average_dice_score=summary["dice"],
            save_suffix=save_suffix,
            outdir=str(outdir) + "/"
        )

    elif metric == "center_shift":
        visualize_centroid_shift(
            region_A=region_a,
            region_B=region_b,
            metrics_df=metrics_df,
            region_mean_shift=summary["center_shift"],
            save_suffix=save_suffix,
            outdir=str(outdir) + "/",
        )

    elif metric == "both":
        visualize_metric_scores(
            region_a=region_a,
            region_b=region_b,
            metrics_df=metrics_df,
            save_suffix=save_suffix,
            outdir=str(outdir) + "/",
        )

    else:
        raise ValueError("metric must be 'dice', 'center_shift', or 'both'")

    print(
        f"[OK] method={method}, group={group_id}, pair={fixed_id}->{moving_id}, "
        f"metric={metric}, summary={summary}"
    )


def visualize_all_methods_overlay(
    group_id,
    fixed_id,
    moving_id,
    metric="dice",
    methods=ALL_METHODS,
    outdir=OUT_FIG,
    strict=STRICT,
):
    """
    Run region-overlay visualization across all methods for one group/pair.
    """
    for method in methods:
        print(f"[INFO] Visualizing {method} | group={group_id} | pair={fixed_id}->{moving_id}")

        try:
            visualize_one_method_overlay(
                method=method,
                group_id=group_id,
                fixed_id=fixed_id,
                moving_id=moving_id,
                metric=metric,
                outdir=outdir,
            )
        except Exception as e:
            msg = (
                f"[ERROR] method={method}, group={group_id}, pair={fixed_id}->{moving_id}, "
                f"metric={metric}: {e}"
            )
            if strict:
                raise RuntimeError(msg) from e
            print(msg)





def main():
    # collect_all_metrics()

    visualize_all_methods_overlay(
        group_id=0,
        fixed_id=1,
        moving_id=2,
        metric="dice",
    )

    # Example 2: save only center-shift overlays
    # visualize_all_methods_overlay(
    #     group_id=0,
    #     fixed_id=0,
    #     moving_id=2,
    #     metric="center_shift",
    # )

    # Example 3: save both Dice and center-shift overlays
    # visualize_all_methods_overlay(
    #     group_id=0,
    #     fixed_id=0,
    #     moving_id=2,
    #     metric="both",
    # )


if __name__ == "__main__":
    main()