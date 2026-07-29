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

from io_utils import load_pair_for_method,normalize_pair_by_fixed_frame
from metrics_utils import call_domain_evaluator


# =========================
# Configuration
# =========================
ROOT = Path("/media/huifang/data/registration/DLPFC")

GROUP_IDS = [0, 1, 2]
PAIR_LIST = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
#
# GROUP_IDS = [ 1]
# PAIR_LIST = [(1, 2)]

# Slice positions for DLPFC
SLICE_POS_UM = {
    0: 0,
    1: 10,
    2: 310,
    3: 320,
}

COMPETING_METHODS = [
    "simpleitk",
    "voxelmorph",
    "nicetrans",
    "paste",
    "gpsa",
    "santo",
]

ALL_METHODS = ["unaligned"]+COMPETING_METHODS + ["axis"]
# ALL_METHODS = ["axis"]
# ALL_METHODS=["unaligned"]

# Unaligned baseline

H5AD_COORD_KEY = "spatial"
H5AD_LABEL_KEY = "layer_guess_reordered"

# Competing methods
COMPETING_ROOT = ROOT / "competing_methods"

# AXIS
AXIS_ROOT = ROOT / "huifang"
AXIS_PAIRWISE_DIR = AXIS_ROOT / "pairwise_output"
AXIS_NETWORK_INPUT_DIR = AXIS_ROOT / "network_input"
H5AD_DIR = AXIS_ROOT / "h5ad"

# Output
OUT_CSV = ROOT / "evaluation" / "dlpfc_domain_metrics_all_methods.csv"


STRICT = True  # set False to skip missing pairs instead of failing


def compute_distance_um(fixed_id: int, moving_id: int) -> int:
    return int(abs(SLICE_POS_UM[moving_id] - SLICE_POS_UM[fixed_id]))


# =========================
# Main wrapper
# =========================
def evaluate_one_pair(
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
) -> pd.DataFrame:
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

    if method == 'gpsa' or method =='santo':
        pass
    else:
        label1 = np.load(str(AXIS_NETWORK_INPUT_DIR / f"{group_id}_{fixed_id}_region_label.npy"))
        label2 = np.load(str(AXIS_NETWORK_INPUT_DIR / f"{group_id}_{moving_id}_region_label.npy"))


    if group_id ==0:
        alpha_value = 0.05
    else:
        alpha_value = 0.1

    pts1_norm, pts2_norm, mask_shape = normalize_pair_by_fixed_frame(pts1, pts2)

    metrics_df = call_domain_evaluator(
        coords_fixed=pts1_norm,
        labels_fixed=label1,
        coords_moving=pts2_norm,
        labels_moving=label2,
        mask_shape=mask_shape,
        alpha=alpha_value
    )

    metrics_df["group_id"] = int(group_id)
    metrics_df["fixed_id"] = int(fixed_id)
    metrics_df["moving_id"] = int(moving_id)
    metrics_df["distance_um"] = int(compute_distance_um(fixed_id, moving_id))
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

    return metrics_df


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
                    metrics_df = evaluate_one_pair(
                        method=method,
                        group_id=group_id,
                        fixed_id=fixed_id,
                        moving_id=moving_id,
                    )

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


def main():
    collect_all_metrics()


if __name__ == "__main__":
    main()