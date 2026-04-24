#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from io_utils import load_axis_pair, load_competing_method_pair, normalize_pair_by_fixed_frame
from metrics_utils import (
    evaluate_molecular_metrics,
    plot_registration_topk_visual,
    standardize_metrics_df_molecular,
)

# =========================
# Configuration
# =========================
ROOT = Path("/media/huifang/data/registration/SCC")

GROUP_IDS = [2, 5, 9, 10]
PAIR_LIST = [(0, 1), (0, 2), (1, 2)]

SLICE_POS_UM = {
    0: 0.0,
    1: 5.0,
    2: 10.0,
}

COMPETING_METHODS = [
    "unaligned",
    "simpleitk",
    "voxelmorph",
    "nicetrans",
    "paste",
    # "gpsa",
    "santo",
]
ALL_METHODS = COMPETING_METHODS + ["axis"]
# Competing methods
COMPETING_ROOT = ROOT / "competing_methods"

# AXIS
AXIS_ROOT = ROOT / "huifang"
AXIS_PAIRWISE_DIR = AXIS_ROOT / "pairwise_output"
AXIS_NETWORK_INPUT_DIR = AXIS_ROOT / "network_input"

OUT_CSV = ROOT / "evaluation" / "scc_molecular_metrics_all_methods.csv"
OUT_FIG = ROOT / "evaluation" / "spotk_metric_plots"

DIST_THRESH = 50.0
TOP_N_SPOTS = 10
STRICT = True


# =========================
# Helpers
# =========================
def compute_distance_um(fixed_id: int, moving_id: int) -> float:
    return float(abs(SLICE_POS_UM[moving_id] - SLICE_POS_UM[fixed_id]))


def safe_score(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True, dtype=float)
    out[np.isnan(out)] = -np.inf
    return out


def load_pair_for_method(
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
):
    """
    Assumptions based on your current SCC layout:
      - competing methods, including 'unaligned', are stored under:
        ROOT/competing_methods/{method}/group_{group_id}/{fixed}_{moving}_result.npz
      - AXIS is loaded from network_input + pairwise_output
    """
    if method in COMPETING_METHODS:
        method_dir = COMPETING_ROOT / method / f"group_{group_id}"
        return load_competing_method_pair(method_dir, fixed_id, moving_id)

    if method == "axis":
        return load_axis_pair(
            group_id=group_id,
            fixed_id=fixed_id,
            moving_id=moving_id,
            AXIS_NETWORK_INPUT_DIR=AXIS_NETWORK_INPUT_DIR,
            AXIS_PAIRWISE_DIR=AXIS_PAIRWISE_DIR,
        )

    raise ValueError(f"Unknown method: {method}")


# =========================
# Evaluation
# =========================
def evaluate_one_pair(
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
    dist_thresh: float = DIST_THRESH,
    use_gpsa_subset=False
):
    pts1, pts2, label1, label2 = load_pair_for_method(
        method=method,
        group_id=group_id,
        fixed_id=fixed_id,
        moving_id=moving_id,
    )
    print(method)
    # if use_gpsa_subset:
    #     if method!='gpsa':
    #         fixed_reserved_id = np.load(f"/media/huifang/data/registration/SCC/competing_methods/gpsa/spot_ids/group_{group_id}/{fixed_id}_reserved_ids.npy")
    #         moving_reserved_id = np.load(
    #             f"/media/huifang/data/registration/SCC/competing_methods/gpsa/spot_ids/group_{group_id}/{moving_id}_reserved_ids.npy")
    #
    #
    #         pts1 = pts1[fixed_reserved_id]
    #         label1 = label1[fixed_reserved_id]
    #
    #         pts2 = pts2[moving_reserved_id]
    #         label2 = label2[moving_reserved_id]


    # Keep molecular metric and visualization in the same normalized frame.
    pts1_norm, pts2_norm, _ = normalize_pair_by_fixed_frame(pts1, pts2)

    metrics_df, per_spot_df = evaluate_molecular_metrics(
        coords_a=pts1_norm,
        labels_a=label1,
        coords_b=pts2_norm,
        labels_b=label2,
        dist_thresh=dist_thresh,
        min_valid_spots=1,
    )

    metrics_df = standardize_metrics_df_molecular(metrics_df)
    return metrics_df, per_spot_df, pts1_norm, pts2_norm, label1, label2


def collect_all_metrics(
    group_ids: Iterable[int] = GROUP_IDS,
    pair_list: Iterable[Tuple[int, int]] = PAIR_LIST,
    methods: Iterable[str] = ALL_METHODS,
    out_csv: Path = OUT_CSV,
) -> pd.DataFrame:
    rows = []

    for method in methods:
        for group_id in group_ids:
            for fixed_id, moving_id in pair_list:
                print(f"[INFO] method={method}, group={group_id}, pair={fixed_id}->{moving_id}")

                try:
                    metrics_df, _, _, _, _, _ = evaluate_one_pair(
                        method=method,
                        group_id=group_id,
                        fixed_id=fixed_id,
                        moving_id=moving_id,
                    )
                except Exception as e:
                    msg = f"[ERROR] method={method}, group={group_id}, pair={fixed_id}->{moving_id}: {e}"
                    if STRICT:
                        raise RuntimeError(msg) from e
                    print(msg)
                    continue

                metrics_df["group_id"] = int(group_id)
                metrics_df["fixed_id"] = int(fixed_id)
                metrics_df["moving_id"] = int(moving_id)
                metrics_df["distance_um"] = compute_distance_um(fixed_id, moving_id)
                metrics_df["method"] = method

                # Keep the same long-format interface as your previous CSVs:
                # metadata + domain + metric
                metrics_df = metrics_df[
                    [
                        "group_id",
                        "fixed_id",
                        "moving_id",
                        "distance_um",
                        "method",
                        "domain",
                        "spot_topk",
                    ]
                ].copy()

                if metrics_df.empty:
                    print(
                        f"[WARN] Empty metrics for method={method}, "
                        f"group={group_id}, pair={fixed_id}->{moving_id}"
                    )
                else:
                    rows.append(metrics_df)

    if not rows:
        raise RuntimeError("No SCC molecular metrics were collected.")

    out_df = pd.concat(rows, axis=0, ignore_index=True)
    out_df = out_df.sort_values(
        by=["method", "group_id", "fixed_id", "moving_id", "domain"]
    ).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"[OK] Saved CSV to: {out_csv}")
    print(f"[OK] Total rows: {len(out_df)}")
    return out_df


# =========================
# Spot-level visualization
# =========================
def find_best_spots(
    group_id: int,
    fixed_id: int,
    moving_id: int,
    methods: Iterable[str] = ALL_METHODS,
    axis_method: str = "axis",
    dist_thresh: float = DIST_THRESH,
    top_n: int = TOP_N_SPOTS,
) -> np.ndarray:
    """
    Find spot indices in slice B where AXIS beats all other methods
    by the largest margin on spot_topk.

    Assumption:
      the moving slice spot order is consistent across methods.
    """
    scores_by_method = {}
    n_spots_ref = None

    for method in methods:
        _, per_spot_df, _, _, _, _ = evaluate_one_pair(
            method=method,
            group_id=group_id,
            fixed_id=fixed_id,
            moving_id=moving_id,
            dist_thresh=dist_thresh,
            use_gpsa_subset=True
        )

        scores = per_spot_df["spot_topk"].to_numpy(dtype=float)
        scores_by_method[method] = scores

        if n_spots_ref is None:
            n_spots_ref = len(scores)
        elif len(scores) != n_spots_ref:
            raise ValueError(
                f"Mismatch in number of moving-slice spots across methods: "
                f"{method} has {len(scores)}, expected {n_spots_ref}."
            )

    axis_scores = safe_score(scores_by_method[axis_method])

    other_methods = [m for m in methods if m != axis_method]
    others = np.stack([safe_score(scores_by_method[m]) for m in other_methods], axis=0)

    best_other = np.max(others, axis=0)
    margins = axis_scores - best_other

    mask = axis_scores > best_other
    candidate_idx = np.where(mask)[0]

    if len(candidate_idx) == 0:
        print("No spot where AXIS strictly beats all other methods.")
        return np.array([], dtype=int)

    order = np.argsort(margins[candidate_idx])[::-1]
    ranked = candidate_idx[order]

    print(f"Number of spots where AXIS beats all others: {len(ranked)}")
    print(f"Top {min(top_n, len(ranked))} spot indices:", ranked[:top_n])
    return ranked[:top_n]


def visualize_all_methods(
    group_id: int,
    fixed_id: int,
    moving_id: int,
    methods: Iterable[str] = ALL_METHODS,
    dist_thresh: float = DIST_THRESH,
    top_n: int = TOP_N_SPOTS,
    outdir: Path = OUT_FIG,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # best_spots = find_best_spots(
    #     group_id=group_id,
    #     fixed_id=fixed_id,
    #     moving_id=moving_id,
    #     methods=methods,
    #     dist_thresh=dist_thresh,
    #     top_n=top_n,
    # )
    best_spots = [6]
    for spot_idx in best_spots:
        print(f"[INFO] Visualizing spot {spot_idx}")
        for method in methods:
            _, _, pts1_norm, pts2_norm, label1, label2 = evaluate_one_pair(
                method=method,
                group_id=group_id,
                fixed_id=fixed_id,
                moving_id=moving_id,
                dist_thresh=dist_thresh,
                use_gpsa_subset=True
            )

            save_suffix = f"{method}_g{group_id}_{fixed_id}_{moving_id}_spot{spot_idx}"

            plot_registration_topk_visual(
                coords_A=pts1_norm,
                coords_B=pts2_norm,
                labels_A=label1,
                labels_B=label2,
                spot_index_B=int(spot_idx),
                dist_thresh=dist_thresh,
                method_name=method,
                save_suffix=save_suffix,
                outdir=outdir,
            )

        print(f"[OK] Saved all methods for spot {spot_idx}")


def main():
    # 1) collect long-format CSV
    # collect_all_metrics()

    # 2) optional spot-level qualitative visualization
    visualize_all_methods(
        group_id=10,
        fixed_id=0,
        moving_id=1,
        top_n=5,
    )


if __name__ == "__main__":
    main()