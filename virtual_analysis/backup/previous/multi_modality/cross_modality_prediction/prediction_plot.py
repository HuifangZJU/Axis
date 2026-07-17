#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyreadr


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s))


def load_rds_df(path: str) -> pd.DataFrame:
    res = pyreadr.read_r(path)
    if len(res) == 0:
        raise ValueError(f"No objects found in RDS: {path}")
    obj = next(iter(res.values()))
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected DataFrame from RDS, got {type(obj)}: {path}")
    return obj


def orient_df_to_nrows(df: pd.DataFrame, n_rows: int, name: str) -> pd.DataFrame:
    r, c = df.shape
    if r == n_rows:
        return df
    if c == n_rows:
        print(f"[TRANSPOSE] {name} {df.shape} -> {df.T.shape}")
        return df.T
    raise ValueError(f"{name} shape {df.shape} does not match n_rows={n_rows} in rows or cols.")


def plot_true_pred_panels(
    x, y,
    truth_mat, pred_mat,
    names,
    out_png,
    invert_y=True,
    s=1.0,
    clip_low=1.0,
    clip_high=99.0,
    max_points=0,
    seed=0,
):
    x = np.asarray(x)
    y = np.asarray(y)
    truth_mat = np.asarray(truth_mat)
    pred_mat = np.asarray(pred_mat)

    n = truth_mat.shape[0]
    k = truth_mat.shape[1]

    if max_points and n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        x = x[idx]; y = y[idx]
        truth_mat = truth_mat[idx, :]
        pred_mat = pred_mat[idx, :]
        n = max_points
        print(f"[PLOT] Downsampled points for plotting: {n}")

    fig, axes = plt.subplots(k, 2, figsize=(10, 2.2 * k), dpi=200)
    if k == 1:
        axes = np.array([axes])

    for i in range(k):
        pname = names[i]
        t = truth_mat[:, i]
        p = pred_mat[:, i]

        both = np.concatenate([t[np.isfinite(t)], p[np.isfinite(p)]])
        if both.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = np.percentile(both, clip_low)
            vmax = np.percentile(both, clip_high)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = float(np.nanmin(both)), float(np.nanmax(both))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = 0.0, 1.0

        ax_t = axes[i, 0]
        ax_p = axes[i, 1]

        sc1 = ax_t.scatter(x, y, c=t, s=s, vmin=vmin, vmax=vmax)
        ax_t.set_title(f"{pname} (True)")
        ax_t.axis("off")
        ax_t.set_aspect("equal", adjustable="box")
        if invert_y:
            ax_t.invert_yaxis()

        sc2 = ax_p.scatter(x, y, c=p, s=s, vmin=vmin, vmax=vmax)
        ax_p.set_title(f"{pname} (Pred)")
        ax_p.axis("off")
        ax_p.set_aspect("equal", adjustable="box")
        if invert_y:
            ax_p.invert_yaxis()

        fig.colorbar(sc2, ax=ax_p, fraction=0.046, pad=0.02)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[SAVED] {out_png}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot spatial True vs Pred for top-10 proteins by metric, for ALL cells and TEST cells."
    )
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--invert_y", action="store_true")
    ap.add_argument("--max_points", type=int, default=0, help="Optional downsample for plotting only.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ---- Defaults (edit here if you move files)
    root = "/media/huifang/data/registration/phenocycler/haotian/"
    output_folder = "protein_prediction_outputs_xgb/"
    npz_path = os.path.join(root, output_folder, "protein_predictions_all_labeled.npz")
    metrics_csv = os.path.join(root, output_folder, "protein_metrics_heldout.csv")
    protein_rds = os.path.join(root, output_folder, "xenium_protein_predict.rds")  # for truth if npz lacks 'true'




    # ---- Load metrics and pick top-K
    mdf = pd.read_csv(metrics_csv)
    if "R2" not in mdf.columns or "protein" not in mdf.columns:
        raise ValueError("metrics CSV must contain columns: 'protein' and 'R2'")
    mdf = mdf.sort_values("R2", ascending=False)
    topk_names = mdf["protein"].head(args.topk).tolist()
    print(f"[INFO] Top-{len(topk_names)} proteins by R2:", topk_names)

    # ---- Load NPZ
    d = np.load(npz_path, allow_pickle=True)
    pred = d["pred"]
    proteins = d["proteins"].tolist()
    x_all = d["x"]
    y_all = d["y"]
    split = d["split"]  # 0=train,1=val,2=test

    n_all, n_prot = pred.shape
    print(f"[INFO] NPZ pred shape: {pred.shape}, #proteins={len(proteins)}")

    # Map requested proteins to indices in NPZ
    name_to_j = {p: j for j, p in enumerate(proteins)}
    keep = [p for p in topk_names if p in name_to_j]
    if len(keep) == 0:
        raise ValueError("None of the topK proteins from metrics exist in npz 'proteins' list.")
    missing = [p for p in topk_names if p not in name_to_j]
    if missing:
        print("[WARN] Missing proteins in npz (skipped):", missing)

    cols = [name_to_j[p] for p in keep]
    pred_top = pred[:, cols]

    # ---- Load truth
    if "true" in d.files:
        truth_all = d["true"]
        if truth_all.shape != pred.shape:
            raise ValueError(f"npz['true'] shape {truth_all.shape} != pred shape {pred.shape}")
        truth_top = truth_all[:, cols]
        print("[INFO] Using truth from npz key 'true'")
    else:
        truth_df = load_rds_df(protein_rds)
        truth_df = orient_df_to_nrows(truth_df, n_all, "protein truth")
        miss_truth = [p for p in keep if p not in truth_df.columns]
        if miss_truth:
            raise ValueError(f"Truth RDS missing proteins: {miss_truth[:10]} ...")
        truth_top = truth_df[keep].to_numpy()
        print("[INFO] Loaded truth from RDS:", protein_rds)

    # ---- Plot ALL cells
    all_png = os.path.join(root+output_folder, f"top{len(keep)}_spatial_true_vs_pred_ALL.png")
    plot_true_pred_panels(
        x_all, y_all,
        truth_top, pred_top,
        keep,
        out_png=all_png,
        invert_y=args.invert_y,
        s=1.0,
        clip_low=1.0,
        clip_high=99.0,
        max_points=args.max_points,
        seed=args.seed,
    )

    # ---- Plot TEST only
    test_mask = (split == 2)
    test_png = os.path.join(args.out_root, f"top{len(keep)}_spatial_true_vs_pred_TEST.png")
    plot_true_pred_panels(
        x_all[test_mask], y_all[test_mask],
        truth_top[test_mask, :], pred_top[test_mask, :],
        keep,
        out_png=test_png,
        invert_y=args.invert_y,
        s=1.0,
        clip_low=1.0,
        clip_high=99.0,
        max_points=args.max_points,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
