#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

import anndata as ad
import pyreadr

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from xgboost import XGBRegressor


# -------------------------
# Progress / timing
# -------------------------
class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.last = self.t0

    def log(self, msg: str):
        now = time.time()
        print(f"[{now - self.t0:8.1f}s | +{now - self.last:6.1f}s] {msg}", flush=True)
        self.last = now


# -------------------------
# I/O helpers
# -------------------------
def load_rds_matrix(path: str) -> pd.DataFrame:
    """Read an RDS that pyreadr can parse into a pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    res = pyreadr.read_r(path)
    if len(res) == 0:
        raise ValueError(f"No object found in RDS: {path}")
    obj = next(iter(res.values()))
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected DataFrame from RDS, got {type(obj)} for {path}")
    return obj


def ensure_numeric_df(df: pd.DataFrame, fillna=0.0) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(fillna)
    return out


def read_locations_csv_weird(path: str) -> pd.DataFrame:
    """
    Your CSV has an invalid first line like: ",0,1"
    then rows: idx,x,y
    """
    loc_df = pd.read_csv(path, header=None, skiprows=1)
    if loc_df.shape[1] < 3:
        raise ValueError(f"Locations CSV has <3 columns after skiprows=1. shape={loc_df.shape}")
    loc_df = loc_df.iloc[:, :3].copy()
    loc_df.columns = ["row_id", "x", "y"]
    loc_df[["x", "y"]] = ensure_numeric_df(loc_df[["x", "y"]])
    return loc_df


# -------------------------
# Orientation helpers
# -------------------------
def orient_to_match_locations(M, n_loc: int, name: str):
    """
    Ensure M has n_loc rows. If it has n_loc columns instead, transpose.
    Supports: pandas DataFrame, numpy array, scipy sparse.
    """
    if isinstance(M, pd.DataFrame):
        r, c = M.shape
        if r == n_loc:
            print(f"[OK] {name} shape = {M.shape} (cells x features)", flush=True)
            return M
        if c == n_loc:
            print(f"[TRANSPOSE] {name} {M.shape} -> {M.T.shape}", flush=True)
            return M.T
        raise ValueError(f"{name} shape {M.shape} doesn't match locations n={n_loc} in rows or cols")

    if sp.issparse(M):
        r, c = M.shape
        if r == n_loc:
            print(f"[OK] {name} shape = {M.shape} (cells x features)", flush=True)
            return M
        if c == n_loc:
            print(f"[TRANSPOSE] {name} {M.shape} -> {(M.T).shape}", flush=True)
            return M.T.tocsr()
        raise ValueError(f"{name} shape {M.shape} doesn't match locations n={n_loc} in rows or cols")

    M = np.asarray(M)
    r, c = M.shape
    if r == n_loc:
        print(f"[OK] {name} shape = {M.shape} (cells x features)", flush=True)
        return M
    if c == n_loc:
        print(f"[TRANSPOSE] {name} {M.shape} -> {M.T.shape}", flush=True)
        return M.T
    raise ValueError(f"{name} shape {M.shape} doesn't match locations n={n_loc} in rows or cols")


# -------------------------
# Outlier removal
# -------------------------
def remove_outlier_rows_by_protein(
    X, Y: pd.DataFrame, coords: pd.DataFrame,
    low_q: float = 0.1, high_q: float = 99.9,
    proteins_subset: list = None,
    drop_if_any: bool = True,
    verbose: bool = True,
):
    """
    Remove rows (cells) with extreme protein values, and apply same mask to X and coords.
    Y must be a DataFrame.
    """
    if proteins_subset is None:
        Y_use = Y
        use_names = list(Y.columns)
    else:
        use_names = [p for p in proteins_subset if p in Y.columns]
        if len(use_names) == 0:
            raise ValueError("proteins_subset provided but none found in Y.columns.")
        Y_use = Y[use_names]

    Y_np = Y_use.to_numpy(dtype=np.float32, copy=False)

    lo = np.percentile(Y_np, low_q, axis=0)
    hi = np.percentile(Y_np, high_q, axis=0)

    within = (Y_np >= lo) & (Y_np <= hi)
    if drop_if_any:
        inlier = np.all(within, axis=1)
        mode = "ANY-protein outlier removed"
    else:
        inlier = np.any(within, axis=1)
        mode = "ONLY rows outlier for ALL proteins removed"

    n_before = Y.shape[0]
    n_after = int(inlier.sum())
    if verbose:
        print(
            f"[Outlier filter] {mode}. p{low_q}..p{high_q} using {len(use_names)} proteins. "
            f"Keep {n_after}/{n_before} ({100*n_after/n_before:.2f}%), removed {n_before-n_after}.",
            flush=True
        )

    # Apply to Y and coords
    Y_f = Y.loc[inlier].reset_index(drop=True)
    coords_f = coords.loc[inlier].reset_index(drop=True)

    # Apply to X
    if isinstance(X, pd.DataFrame):
        X_f = X.loc[inlier].reset_index(drop=True)
    elif sp.issparse(X):
        X_f = X[inlier]
    else:
        X_f = np.asarray(X)[inlier]

    return X_f, Y_f, coords_f


# -------------------------
# Metrics
# -------------------------
def per_protein_metrics(y_true: np.ndarray, y_pred: np.ndarray, protein_names):
    rows = []
    for j, name in enumerate(protein_names):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        r2 = np.nan if np.std(yt) == 0 else r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))

        if len(yt) > 2 and np.std(yt) > 0 and np.std(yp) > 0:
            pr = pearsonr(yt, yp)[0]
            sr = spearmanr(yt, yp)[0]
        else:
            pr, sr = np.nan, np.nan

        rows.append({
            "protein": str(name),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "PearsonR": float(pr) if np.isfinite(pr) else np.nan,
            "SpearmanR": float(sr) if np.isfinite(sr) else np.nan,
            "MAE": float(mae),
            "RMSE": rmse,
        })
    return pd.DataFrame(rows)


# -------------------------
# Plot top-3 spatial (2x3): True row, Pred row
# Robust shared vmin/vmax per protein (percentile clipped)
# -------------------------
def plot_top3_spatial(true_vals, pred_vals, x, y, protein_names, out_png, invert_y=True):
    n_show = min(3, true_vals.shape[1], pred_vals.shape[1], len(protein_names))
    fig, axes = plt.subplots(2, n_show, figsize=(4.2 * n_show, 7.0), dpi=200)

    if n_show == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i in range(n_show):
        pname = protein_names[i]
        t = true_vals[:, i]
        p = pred_vals[:, i]

        # Shared robust scale for True/Pred
        vmin = np.percentile(np.concatenate([t, p]), 1)
        vmax = np.percentile(np.concatenate([t, p]), 99)

        ax_true = axes[0, i]
        ax_pred = axes[1, i]

        sc1 = ax_true.scatter(x, y, c=t, s=1, vmin=vmin, vmax=vmax)
        ax_true.set_title(f"{pname} (True)")
        ax_true.axis("off")
        ax_true.set_aspect("equal", adjustable="box")
        if invert_y:
            ax_true.invert_yaxis()
        plt.colorbar(sc1, ax=ax_true, fraction=0.046, pad=0.02)

        sc2 = ax_pred.scatter(x, y, c=p, s=1, vmin=vmin, vmax=vmax)
        ax_pred.set_title(f"{pname} (Pred)")
        ax_pred.axis("off")
        ax_pred.set_aspect("equal", adjustable="box")
        if invert_y:
            ax_pred.invert_yaxis()
        plt.colorbar(sc2, ax=ax_pred, fraction=0.046, pad=0.02)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()



def grid_aggregate(x, y, v, n_bins=200, agg="mean", xy_range=None, min_count=3):
    """
    Bin points (x,y) into a 2D grid and aggregate values v per bin.

    Returns
    -------
    grid : (n_bins, n_bins) float array with aggregated values (NaN where insufficient points)
    counts : (n_bins, n_bins) int array with number of points per bin
    extent : (xmin, xmax, ymin, ymax) for plotting
    """
    x = np.asarray(x); y = np.asarray(y); v = np.asarray(v)

    if xy_range is None:
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
    else:
        xmin, xmax, ymin, ymax = xy_range

    # Bin indices
    x_edges = np.linspace(xmin, xmax, n_bins + 1)
    y_edges = np.linspace(ymin, ymax, n_bins + 1)

    xi = np.searchsorted(x_edges, x, side="right") - 1
    yi = np.searchsorted(y_edges, y, side="right") - 1

    valid = (xi >= 0) & (xi < n_bins) & (yi >= 0) & (yi < n_bins) & np.isfinite(v)
    xi = xi[valid]; yi = yi[valid]; v = v[valid]

    counts = np.zeros((n_bins, n_bins), dtype=np.int32)
    sums = np.zeros((n_bins, n_bins), dtype=np.float64)

    # accumulate
    np.add.at(counts, (yi, xi), 1)   # note (row=y, col=x)
    np.add.at(sums,   (yi, xi), v)

    if agg == "mean":
        grid = sums / np.maximum(counts, 1)
    elif agg == "sum":
        grid = sums
    else:
        raise ValueError("agg must be 'mean' or 'sum'")

    # mask low-count bins
    grid = grid.astype(np.float64)
    grid[counts < min_count] = np.nan

    extent = (xmin, xmax, ymin, ymax)
    return grid, counts, extent


def grid_similarity_metrics(truth_grid, pred_grid):
    """
    Compute similarity metrics on overlapping (non-NaN) grid cells.
    """
    t = truth_grid.ravel()
    p = pred_grid.ravel()
    m = np.isfinite(t) & np.isfinite(p)
    t = t[m]; p = p[m]

    if t.size < 10:
        return {"n": int(t.size), "pearson": np.nan, "spearman": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan}

    pear = pearsonr(t, p)[0] if np.std(t) > 0 and np.std(p) > 0 else np.nan
    spear = spearmanr(t, p)[0] if np.std(t) > 0 and np.std(p) > 0 else np.nan
    r2 = r2_score(t, p) if np.std(t) > 0 else np.nan
    rmse = np.sqrt(mean_squared_error(t, p))
    mae = mean_absolute_error(t, p)

    return {"n": int(t.size), "pearson": float(pear), "spearman": float(spear), "r2": float(r2), "rmse": float(rmse), "mae": float(mae)}
def plot_grid_scatter(truth_grid, pred_grid, title, out_png, hexbin=False):
    """
    Scatter plot of grid truth vs pred; includes y=x line.
    """
    t = truth_grid.ravel()
    p = pred_grid.ravel()
    m = np.isfinite(t) & np.isfinite(p)
    t = t[m]; p = p[m]

    plt.figure(figsize=(5.5, 5.0), dpi=200)
    if hexbin and t.size > 5000:
        hb = plt.hexbin(t, p, gridsize=60, mincnt=1)
        plt.colorbar(hb, label="bin count")
    else:
        plt.scatter(t, p, s=4, alpha=0.4)

    lo = np.nanmin([t.min(), p.min()])
    hi = np.nanmax([t.max(), p.max()])
    plt.plot([lo, hi], [lo, hi], linestyle="--")  # diagonal

    plt.xlabel("Truth (grid aggregated)")
    plt.ylabel("Pred (grid aggregated)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def evaluate_spatial_similarity_by_grid(
    x, y, truth, pred,
    name="marker",
    n_bins=200,
    agg="mean",
    min_count=3,
    out_prefix="grid_eval",
    invert_y_for_images=False
):
    """
    End-to-end:
      - gridize truth/pred using same binning
      - compute metrics
      - save scatter plot truth vs pred
      - (optional) save heatmaps for sanity
    """
    # use shared bounds so grids align
    xy_range = (float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y)))

    truth_grid, counts, extent = grid_aggregate(x, y, truth, n_bins=n_bins, agg=agg, xy_range=xy_range, min_count=min_count)
    pred_grid,  _,      _      = grid_aggregate(x, y, pred,  n_bins=n_bins, agg=agg, xy_range=xy_range, min_count=min_count)

    metrics = grid_similarity_metrics(truth_grid, pred_grid)

    # scatter
    scatter_png = f"{out_prefix}_{name}_truth_vs_pred_scatter.png"
    plot_grid_scatter(
        truth_grid, pred_grid,
        title=f"{name} grid similarity | r={metrics['pearson']:.3f}, ρ={metrics['spearman']:.3f}, R²={metrics['r2']:.3f}",
        out_png=scatter_png,
        hexbin=True
    )

    return metrics, scatter_png

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict all proteins from genes using XGBoost (one model per protein).")
    parser.add_argument("--root", type=str, default="/media/huifang/data/registration/phenocycler/haotian/")
    parser.add_argument("--protein_rds", type=str, default="xenium_protein_predict.rds")
    parser.add_argument("--loc_csv", type=str, default="LUAD_2_A_xenium_axis.csv")
    parser.add_argument("--h5ad", type=str, default="adata_xenium_cell_level.h5ad")

    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--val_size", type=float, default=0.10)  # fraction of TRAIN used for early stopping
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--log1p_X", action="store_true", help="Apply log1p to gene matrix X (common for counts).")

    # XGBoost params
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--n_jobs", type=int, default=8)

    parser.add_argument("--invert_y", action="store_true")

    args = parser.parse_args()

    tm = Timer()


    root = args.root
    protein_path = os.path.join(root, args.protein_rds)
    loc_path = os.path.join(root, args.loc_csv)
    h5ad_path = os.path.join(root, args.h5ad)

    out_dir = os.path.join(root, "protein_prediction_outputs_xgb")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load protein matrix (Y)
    tm.log(f"Loading protein matrix: {protein_path}")
    protein_df = load_rds_matrix(protein_path)
    protein_df = ensure_numeric_df(protein_df)
    protein_df= protein_df.iloc[20:35, :]
    tm.log(f"Protein df shape (raw): {protein_df.shape}")



    # ---- Load locations (coords)
    tm.log(f"Loading locations: {loc_path}")
    loc_df = read_locations_csv_weird(loc_path)
    coords = loc_df[["x", "y"]].copy()
    coords = ensure_numeric_df(coords)
    tm.log(f"Coords shape: {coords.shape}")

    # ---- Load genes from h5ad (X)
    tm.log(f"Loading gene matrix from h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr().astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)
    tm.log(f"adata.X shape (raw): {X.shape}")


    # ---- Orient X and Y to match coords rows
    nC = coords.shape[0]
    X = orient_to_match_locations(X, nC, "Gene matrix X")
    Y = orient_to_match_locations(protein_df, nC, "Protein matrix Y")  # keep as DataFrame for now
    Y = ensure_numeric_df(Y)  # ensure after transpose too

    tm.log(f"After orientation: X={X.shape}, Y={Y.shape}, coords={coords.shape}")

    # ---- OPTIONAL: Downsample AFTER orientation ----
    downsample_n = 5000
    seed = 0

    n_total = X.shape[0]

    if downsample_n is not None and downsample_n < n_total:
        tm.log(f"Downsampling from {n_total} to {downsample_n} cells...")

        rng = np.random.default_rng(seed)
        ds_idx = rng.choice(n_total, size=downsample_n, replace=False)

        # IMPORTANT: operate on oriented matrices
        Y = Y.iloc[ds_idx].reset_index(drop=True)
        coords = coords.iloc[ds_idx].reset_index(drop=True)

        if sp.issparse(X):
            X = X[ds_idx]
        else:
            X = X[ds_idx, :]

        tm.log(f"After downsample: X={X.shape}, Y={Y.shape}, coords={coords.shape}")


    # ---- Preprocess X (optional log1p)
    use_log1p = args.log1p_X
    if use_log1p:
        tm.log("Applying log1p to gene matrix X...")
        if sp.issparse(X):
            X = X.copy()
            X.data = np.log1p(X.data)
        else:
            X = np.log1p(X)

    # ---- Convert Y to numpy
    protein_names = list(Y.columns)
    Y_np = Y.to_numpy(dtype=np.float32, copy=False)

    # ---- Train / test split
    tm.log("Splitting train/test...")
    idx = np.arange(X.shape[0])
    train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed, shuffle=True)

    # For early stopping, split train -> train/val
    train_idx2, val_idx = train_test_split(
        train_idx,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True
    )

    # subset X
    def take_rows(mat, rows):
        if sp.issparse(mat):
            return mat[rows]
        return mat[rows, :]

    X_train = take_rows(X, train_idx2)
    X_val   = take_rows(X, val_idx)
    X_test  = take_rows(X, test_idx)

    Y_train = Y_np[train_idx2]
    Y_val   = Y_np[val_idx]
    Y_test  = Y_np[test_idx]

    # coords for test
    x_test = coords.iloc[test_idx]["x"].to_numpy()
    y_test = coords.iloc[test_idx]["y"].to_numpy()

    tm.log(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}, Y_train={Y_train.shape}")

    # ---- XGBoost config
    tree_method = "gpu_hist" if args.use_gpu else "hist"
    predictor = "gpu_predictor" if args.use_gpu else "auto"

    base_params = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        objective="reg:squarederror",
        tree_method=tree_method,
        predictor=predictor,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbosity=0,
    )

    tm.log(f"XGBoost params: tree_method={tree_method}, n_estimators={args.n_estimators}, lr={args.learning_rate}")

    # ---- Fit one model per protein (progress printed)
    n_prot = Y_train.shape[1]
    Y_pred = np.zeros((Y_test.shape[0], n_prot), dtype=np.float32)
    best_iters = []

    n_all = X.shape[0]
    n_prot = len(protein_names)
    Y_pred_all = np.zeros((n_all, n_prot), dtype=np.float32)

    tm.log("Training XGBoost models (one per protein)...")
    for j, pname in enumerate(protein_names):
        t_start = time.time()

        model = XGBRegressor(**base_params)

        model.fit(
            X_train, Y_train[:, j],
            eval_set=[(X_val, Y_val[:, j])],
            eval_metric="rmse",
            verbose=False,
            early_stopping_rounds=args.early_stopping_rounds
        )

        pred_j = model.predict(X_test)
        Y_pred[:, j] = pred_j.astype(np.float32, copy=False)
        Y_pred_all[:, j] = model.predict(X).astype(np.float32, copy=False)

        best_iter = getattr(model, "best_iteration", None)
        best_iters.append(best_iter if best_iter is not None else np.nan)

        dt = time.time() - t_start
        if (j == 0) and (n_prot > 1):
            # rough ETA after first model
            eta_min = dt * (n_prot - 1) / 60.0
            tm.log(f"[1/{n_prot}] {pname} done in {dt:.1f}s | rough remaining ~{eta_min:.1f} min")
        else:
            tm.log(f"[{j+1}/{n_prot}] {pname} done in {dt:.1f}s")

    tm.log("All protein models trained.")
    # ---- Save ALL predictions with split labels (NPZ only) ----
    x_all = coords["x"].to_numpy(dtype=np.float32, copy=False)
    y_all = coords["y"].to_numpy(dtype=np.float32, copy=False)

    # split codes: 0=train, 1=val, 2=test
    split_code = np.zeros(n_all, dtype=np.int8)
    split_code[val_idx] = 1
    split_code[test_idx] = 2

    npz_path = os.path.join(out_dir, "protein_predictions_all_labeled.npz")
    np.savez_compressed(
        npz_path,
        pred=Y_pred_all,  # (n_cells, n_proteins)
        proteins=np.array(protein_names, dtype=object),
        x=x_all,
        y=y_all,
        split=split_code,  # 0/1/2
        train_idx=train_idx2.astype(np.int32),
        val_idx=val_idx.astype(np.int32),
        test_idx=test_idx.astype(np.int32),
    )
    tm.log(f"Saved NPZ: {npz_path}")

    # ---- Metrics for all proteins
    tm.log("Computing metrics...")
    metrics_df = per_protein_metrics(Y_test, Y_pred, protein_names)
    metrics_df["best_iteration"] = best_iters
    metrics_df_sorted = metrics_df.sort_values("R2", ascending=False)

    metrics_csv = os.path.join(out_dir, "protein_metrics_heldout.csv")
    metrics_df_sorted.to_csv(metrics_csv, index=False)
    tm.log(f"Saved metrics: {metrics_csv}")

    # ---- Plot top-3 spatial (true vs pred)
    top3 = metrics_df_sorted.head(3)["protein"].tolist()
    tm.log(f"Top-3 proteins by held-out R2: {top3}")

    top3_idx = [protein_names.index(p) for p in top3]
    Y_test_top3 = Y_test[:, top3_idx]
    Y_pred_top3 = Y_pred[:, top3_idx]

    x = coords.iloc[test_idx]["x"].to_numpy()
    y = coords.iloc[test_idx]["y"].to_numpy()
    # Example for PD.1 (or any protein/gene)
    for pname in top3:
        if pname not in protein_names:
            print(f"[WARN] Protein not found in protein_names: {pname}")
            continue

        j = protein_names.index(pname)

        truth = np.asarray(Y_test[:, j], dtype=np.float32)
        pred = np.asarray(Y_pred[:, j], dtype=np.float32)

        # Optional: drop non-finite pairs (shouldn't happen, but safe)
        m = np.isfinite(truth) & np.isfinite(pred)
        truth = truth[m]
        pred = pred[m]
        x_use = np.asarray(x)[m]
        y_use = np.asarray(y)[m]

        # Make a filename-safe prefix (e.g., "PD.1" -> "PD_1")
        safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(pname))
        out_prefix = os.path.join(out_dir, f"grid_{safe_name}")


        metrics, scatter_png = evaluate_spatial_similarity_by_grid(
            x=x_use, y=y_use,
            truth=truth, pred=pred,
            name=pname,
            n_bins=35,  # adjust
            agg="mean",
            min_count=5,
            out_prefix=out_prefix
        )

        print(f"[{pname}] Grid metrics: {metrics}")
        print(f"[{pname}] Saved: {scatter_png}")



    out_png = os.path.join(out_dir, "top3_spatial_true_vs_pred.png")
    plot_top3_spatial(
        true_vals=Y_test_top3,
        pred_vals=Y_pred_top3,
        x=x_test, y=y_test,
        protein_names=top3,
        out_png=out_png,
        invert_y=args.invert_y
    )
    tm.log(f"Saved plot: {out_png}")

    # ---- Save summary
    summary = {
        "n_total": int(X.shape[0]),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_genes": int(X.shape[1]),
        "n_proteins": int(n_prot),
        "use_log1p_X": bool(use_log1p),
        "remove_outliers": bool(args.remove_outliers),
        "outlier_low_q": args.outlier_low_q,
        "outlier_high_q": args.outlier_high_q,
        "outlier_any": bool(args.outlier_any),
        "xgb_params": base_params,
        "early_stopping_rounds": args.early_stopping_rounds,
        "top3_by_R2": top3,
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    tm.log("Done.")


if __name__ == "__main__":
    main()
