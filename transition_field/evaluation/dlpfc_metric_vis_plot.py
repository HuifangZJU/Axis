#!/usr/bin/env python3
"""
Wrapper to visualize two-slice overlay metrics across all methods
for a given DLPFC group and pair.

It uses the existing metric pipeline:
  - evaluate_domain_metrics(...)
  - aggregate_domain_metrics(...)
  - visualize_dice_regions(...)
  - visualize_centroid_shift(...)
  - visualize_metric_scores(...)

Expected output:
  one saved image per method for the selected metric,
  or both images per method if metric="both".
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import scanpy as sc

# ---------------------------------------------------------------------
# Replace these imports with your actual module names if needed
# ---------------------------------------------------------------------
from metrics_utils import (
    evaluate_domain_metrics,
    aggregate_domain_metrics,
    visualize_dice_regions,
    visualize_centroid_shift,
    visualize_metric_scores,
)


# ============================================================
# Config
# ============================================================

ROOT = Path("/media/huifang/data/registration/DLPFC")

METHODS = [
    "unaligned",
    "simpleitk",
    "voxelmorph",
    "nicetrans",
    "paste",
    "gpsa",
    "santo",
    "axis",
]

COMPETING_METHODS = [
    "unaligned",
    "simpleitk",
    "voxelmorph",
    "nicetrans",
    "paste",
    "gpsa",
    "santo",
]

H5AD_DIR = ROOT / "h5adfile"
H5AD_COORD_KEY = "spatial"
H5AD_LABEL_KEY = "layer_guess_reordered"

COMPETING_ROOT = ROOT / "competing_methods"

AXIS_ROOT = ROOT / "huifang"
AXIS_PAIRWISE_DIR = AXIS_ROOT / "pairwise_output"
AXIS_NETWORK_INPUT_DIR = AXIS_ROOT / "network_input"

OUTDIR = ROOT / "evaluation" / "overlay_metric_plots"


CORR_GRID_SIZE = 100
CORR_SIGMA = 2.0


STRICT = True
OFFSET = 5.0
TARGET_RANGE = 100.0


# ============================================================
# Basic helpers
# ============================================================

def clean_labels(labels):
    labels = np.asarray(labels).squeeze().ravel()

    if labels.dtype.kind == "S":
        return labels.astype(str)

    if labels.dtype == object:
        cleaned = []
        for x in labels:
            if isinstance(x, bytes):
                cleaned.append(x.decode("utf-8"))
            else:
                cleaned.append(x)
        return np.asarray(cleaned, dtype=object)

    return labels


def ensure_2d_coords(coords, source="coords"):
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"{source}: expected shape [N,2+], got {coords.shape}")
    return coords[:, :2]


def trim_coords_and_labels(coords, labels, source="data"):
    coords = ensure_2d_coords(coords, source=source)
    labels = clean_labels(labels)

    n = min(coords.shape[0], len(labels))
    if coords.shape[0] != len(labels):
        print(
            f"[WARN] {source}: coords length {coords.shape[0]} != labels length {len(labels)}. "
            f"Using first {n}."
        )

    return coords[:n], labels[:n]


def npz_get_first(data, keys, source):
    for key in keys:
        if key in data.files:
            return data[key]
    raise KeyError(
        f"{source}: none of keys {keys} found. Available keys: {list(data.files)}"
    )


def normalize_pair_by_fixed_frame(
    pts_fixed,
    pts_moving,
    offset=OFFSET,
    target_range=TARGET_RANGE,
):
    """
    Normalize both point sets into one common frame using the fixed slice scale.
    """
    pts_fixed = ensure_2d_coords(pts_fixed, "pts_fixed")
    pts_moving = ensure_2d_coords(pts_moving, "pts_moving")

    fixed_min = pts_fixed.min(axis=0)
    fixed_max = pts_fixed.max(axis=0)
    scale = fixed_max - fixed_min
    scale = np.where(scale <= 1e-8, 1.0, scale)

    pts_combined = np.vstack([pts_fixed, pts_moving])
    combined_min = pts_combined.min(axis=0)

    pts_fixed_norm = (pts_fixed - combined_min) / scale * target_range + offset
    pts_moving_norm = (pts_moving - combined_min) / scale * target_range + offset

    pts_norm_combined = np.vstack([pts_fixed_norm, pts_moving_norm])
    max_x = int(np.ceil(pts_norm_combined[:, 0].max()) + offset)
    max_y = int(np.ceil(pts_norm_combined[:, 1].max()) + offset)

    mask_shape = (max_y, max_x)  # (rows, cols) = (y, x)

    return (
        pts_fixed_norm.astype(np.float32),
        pts_moving_norm.astype(np.float32),
        mask_shape,
    )


# ============================================================
# Loaders
# ============================================================

def load_h5ad_pair(group_id, fixed_id, moving_id):
    fixed_path = H5AD_DIR / f"{group_id}_{fixed_id}.h5ad"
    moving_path = H5AD_DIR / f"{group_id}_{moving_id}.h5ad"

    ad_fixed = sc.read_h5ad(fixed_path)
    ad_moving = sc.read_h5ad(moving_path)

    if H5AD_COORD_KEY not in ad_fixed.obsm:
        raise KeyError(f"{fixed_path}: obsm['{H5AD_COORD_KEY}'] not found")
    if H5AD_COORD_KEY not in ad_moving.obsm:
        raise KeyError(f"{moving_path}: obsm['{H5AD_COORD_KEY}'] not found")
    if H5AD_LABEL_KEY not in ad_fixed.obs:
        raise KeyError(f"{fixed_path}: obs['{H5AD_LABEL_KEY}'] not found")
    if H5AD_LABEL_KEY not in ad_moving.obs:
        raise KeyError(f"{moving_path}: obs['{H5AD_LABEL_KEY}'] not found")

    pts1, label1 = trim_coords_and_labels(
        ad_fixed.obsm[H5AD_COORD_KEY],
        ad_fixed.obs[H5AD_LABEL_KEY].to_numpy(),
        source=str(fixed_path),
    )
    pts2, label2 = trim_coords_and_labels(
        ad_moving.obsm[H5AD_COORD_KEY],
        ad_moving.obs[H5AD_LABEL_KEY].to_numpy(),
        source=str(moving_path),
    )
    return pts1, pts2, label1, label2


def load_competing_method_pair(method, group_id, fixed_id, moving_id):
    path = (
        COMPETING_ROOT
        / method
        / f"group_{group_id}"
        / f"{fixed_id}_{moving_id}_result.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)

    pts1 = npz_get_first(data, ["pts1"], source=str(path))
    pts2 = npz_get_first(data, ["pts2"], source=str(path))
    label1 = npz_get_first(data, ["label1"], source=str(path))
    label2 = npz_get_first(data, ["label2"], source=str(path))

    pts1, label1 = trim_coords_and_labels(pts1, label1, source=f"{path}: fixed")
    pts2, label2 = trim_coords_and_labels(pts2, label2, source=f"{path}: moving")

    return pts1, pts2, label1, label2


def load_axis_validation_npz(path):
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)

    coord = npz_get_first(
        data,
        keys=["coord", "coords", "pts", "points", "spatial"],
        source=str(path),
    )
    label = npz_get_first(
        data,
        keys=[
            "label",
            "labels",
            "layer_guess_reordered",
            "layer",
            "domain",
            "domains",
            "y",
            "target",
        ],
        source=str(path),
    )

    coord, label = trim_coords_and_labels(coord, label, source=str(path))
    return coord, label


def load_axis_pair(group_id, fixed_id, moving_id):
    fixed_npz = AXIS_NETWORK_INPUT_DIR / f"{group_id}_{fixed_id}_validation.npz"
    moving_npz = AXIS_NETWORK_INPUT_DIR / f"{group_id}_{moving_id}_validation.npz"

    fixed_pts, fixed_label = load_axis_validation_npz(fixed_npz)
    _, moving_label = load_axis_validation_npz(moving_npz)

    warped_candidates = [
        AXIS_PAIRWISE_DIR / f"group_{group_id}" / f"{fixed_id}_{moving_id}_waped_coor.npy",
        AXIS_PAIRWISE_DIR / f"group_{group_id}" / f"{fixed_id}_{moving_id}_warped_coor.npy",
    ]

    warped_path = None
    for p in warped_candidates:
        if p.exists():
            warped_path = p
            break

    if warped_path is None:
        raise FileNotFoundError(
            f"No AXIS warped coordinate file found for group={group_id}, pair={fixed_id}->{moving_id}"
        )

    moving_pts_warped = np.load(warped_path, allow_pickle=True)
    moving_pts_warped = ensure_2d_coords(moving_pts_warped, source=str(warped_path))

    n = min(moving_pts_warped.shape[0], len(moving_label))
    if moving_pts_warped.shape[0] != len(moving_label):
        print(
            f"[WARN] {warped_path}: warped coords length {moving_pts_warped.shape[0]} "
            f"!= moving labels length {len(moving_label)}. Using first {n}."
        )

    moving_pts_warped = moving_pts_warped[:n]
    moving_label = moving_label[:n]

    return fixed_pts, moving_pts_warped, fixed_label, moving_label


def load_pair_for_method(method, group_id, fixed_id, moving_id):
    if method == "unaligned":
        return load_h5ad_pair(group_id, fixed_id, moving_id)

    if method in COMPETING_METHODS:
        return load_competing_method_pair(method, group_id, fixed_id, moving_id)

    if method == "axis":
        return load_axis_pair(group_id, fixed_id, moving_id)

    raise ValueError(f"Unknown method: {method}")


# ============================================================
# Evaluation wrapper
# ============================================================

def call_domain_evaluator(pts1, label1, pts2, label2, mask_shape,alpha):
    """
    Assumes your refactored evaluator returns:
      metrics_df, region_a, region_b, density_a, density_b
    or at least:
      metrics_df, region_a, region_b, ...
    """
    out = evaluate_domain_metrics(
        pts1,
        label1,
        pts2,
        label2,
        mask_shape=mask_shape,
        alpha=alpha,
        corr_grid_size=CORR_GRID_SIZE,
        corr_sigma=CORR_SIGMA,
    )

    if isinstance(out, tuple):
        if len(out) < 3:
            raise ValueError(
                "evaluate_domain_metrics(...) returned a tuple with fewer than 3 elements. "
                "Expected at least: metrics_df, region_a, region_b"
            )
        metrics_df, region_a, region_b = out[:3]
    else:
        raise TypeError(
            "evaluate_domain_metrics(...) must return a tuple like "
            "(metrics_df, region_a, region_b, ...)"
        )

    return metrics_df, region_a, region_b


# ============================================================
# Visualization wrapper
# ============================================================

def visualize_one_method_overlay(
    method,
    group_id,
    fixed_id,
    moving_id,
    metric="dice",
    outdir=OUTDIR,
):
    """
    metric:
        "dice"
        "center_shift"
        "both"
    """
    pts1, pts2, label1, label2 = load_pair_for_method(
        method=method,
        group_id=group_id,
        fixed_id=fixed_id,
        moving_id=moving_id,
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

    metrics_df, region_a, region_b = call_domain_evaluator(
        pts1_norm,
        label1,
        pts2_norm,
        label2,
        mask_shape=mask_shape,
        alpha=alpha_value
    )

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
            outdir=str(outdir) + "/",
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
    methods=METHODS,
    outdir=OUTDIR,
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


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Example 1: save only Dice overlays for all methods
    visualize_all_methods_overlay(
        group_id=1,
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