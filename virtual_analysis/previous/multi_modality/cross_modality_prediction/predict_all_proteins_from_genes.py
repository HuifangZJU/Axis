#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskElasticNetCV, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# rpy2
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

import time

class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.last = self.t0
    def log(self, msg):
        now = time.time()
        print(f"[{now - self.t0:8.1f}s | +{now - self.last:6.1f}s] {msg}", flush=True)
        self.last = now



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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




# --------------------------
# RDS loading via rpy2
# --------------------------
def load_rds_matrix(path: str) -> pd.DataFrame:
    """
    Loads an R object from .rds using rpy2 and returns a pandas DataFrame with shape (n_rows, n_cols).
    Supports:
      - base R matrix
      - data.frame
      - Matrix::dgCMatrix (sparse) and other Matrix classes convertible via as.matrix()
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    readRDS = r["readRDS"]
    obj = readRDS(path)

    # If it's a data.frame, convert directly
    if "data.frame" in list(r["class"](obj)):
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(obj)
        return df

    # Otherwise, try to coerce to matrix (handles dgCMatrix etc.)
    # Use as.matrix() to densify; if huge, you should instead keep sparse and use a sparse model.
    as_matrix = r["as.matrix"]
    mat = as_matrix(obj)

    # Convert to numpy via rpy2 (matrix becomes a vector + dim)
    mat_np = np.array(mat)
    # rpy2 returns a 1D array sometimes; reshape using dim
    dim = tuple(int(x) for x in r["dim"](mat))
    if mat_np.ndim == 1 and len(dim) == 2:
        mat_np = mat_np.reshape(dim, order="F")  # R is column-major
    elif mat_np.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected matrix ndim {mat_np.ndim} from {path} with dim={dim}")

    # Try to get row/col names
    rn = r["rownames"](mat)
    cn = r["colnames"](mat)
    rownames = [str(x) for x in rn] if rn is not ro.NULL else None
    colnames = [str(x) for x in cn] if cn is not ro.NULL else None

    df = pd.DataFrame(mat_np, index=rownames, columns=colnames)
    return df


def orient_to_match_locations(M: pd.DataFrame, n_loc: int, name: str) -> pd.DataFrame:
    """
    Ensure M has n_loc rows (cells). If M has n_loc columns instead, transpose it.
    """
    r, c = M.shape
    if r == n_loc:
        print(f"[OK] {name} already oriented as (cells x features): {M.shape}")
        return M
    if c == n_loc:
        print(f"[TRANSPOSE] {name} seems transposed (features x cells): {M.shape} -> {M.T.shape}")
        return M.T
    raise ValueError(
        f"{name} shape {M.shape} does not match locations n={n_loc} in either rows or columns. "
        f"Expected {n_loc} to be either rows or cols."
    )

# --------------------------
# Utils
# --------------------------
def detect_xy_columns(df: pd.DataFrame):
    patterns = [
        ("x_centroid", "y_centroid"),
        ("x", "y"),
        ("X", "Y"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("global_x", "global_y"),
        ("coord_x", "coord_y"),
    ]
    cols = set(df.columns)
    for xcol, ycol in patterns:
        if xcol in cols and ycol in cols:
            return xcol, ycol

    x_candidates = [c for c in df.columns if re.search(r"(^x$|x_.*|.*_x$|xcentroid|centroid_x)", c, re.I)]
    y_candidates = [c for c in df.columns if re.search(r"(^y$|y_.*|.*_y$|ycentroid|centroid_y)", c, re.I)]
    if x_candidates and y_candidates:
        return x_candidates[0], y_candidates[0]

    raise ValueError(f"Could not detect x/y columns. Available: {list(df.columns)}")


def ensure_numeric_df(df: pd.DataFrame, fillna=0.0) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(fillna)
    return out


def per_protein_metrics(y_true: np.ndarray, y_pred: np.ndarray, protein_names):
    rows = []
    for j, name in enumerate(protein_names):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        # R2 handle constant vectors
        if np.std(yt) == 0:
            r2 = np.nan
        else:
            r2 = r2_score(yt, yp)

        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))

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
            "RMSE": float(rmse),
        })
    return pd.DataFrame(rows)


def plot_top3_spatial(true_vals, pred_vals, x, y, protein_names, out_png, invert_y=True):
    """
    Plot top-3 proteins in a 2x3 grid:
      Row 1: True (3 proteins)
      Row 2: Pred (3 proteins)
    """
    n_show = min(3, true_vals.shape[1], pred_vals.shape[1], len(protein_names))
    fig, axes = plt.subplots(2, n_show, figsize=(4.2 * n_show, 7.0), dpi=200)

    # Ensure axes is 2D: (2, n_show)
    if n_show == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i in range(n_show):
        pname = protein_names[i]

        ax_true = axes[0, i]
        ax_pred = axes[1, i]

        sc1 = ax_true.scatter(x, y, c=true_vals[:, i], s=0.1)
        ax_true.set_title(f"{pname} (True)")
        ax_true.axis("off")
        ax_true.set_aspect("equal", adjustable="box")
        if invert_y:
            ax_true.invert_yaxis()
        plt.colorbar(sc1, ax=ax_true, fraction=0.046, pad=0.02)

        sc2 = ax_pred.scatter(x, y, c=pred_vals[:, i], s=0.1)
        ax_pred.set_title(f"{pname} (Pred)")
        ax_pred.axis("off")
        ax_pred.set_aspect("equal", adjustable="box")
        if invert_y:
            ax_pred.invert_yaxis()
        plt.colorbar(sc2, ax=ax_pred, fraction=0.046, pad=0.02)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def stats(a):
    return np.min(a), np.percentile(a, 1), np.median(a), np.percentile(a, 99), np.max(a)


import scipy.sparse as sp

def remove_outlier_rows_by_protein(
    X,
    Y: pd.DataFrame,
    coords: pd.DataFrame,
    low_q: float = 0.1,
    high_q: float = 99.9,
    proteins_subset: list = None,
    drop_if_any: bool = True,
    verbose: bool = True,
):
    """
    Remove rows (cells) that have extreme protein values, and apply the same mask to X and coords.

    Parameters
    ----------
    X : pd.DataFrame or scipy sparse matrix or np.ndarray
        Gene matrix aligned to locations (rows=cells).
    Y : pd.DataFrame
        Protein matrix aligned to locations (rows=cells, cols=proteins).
    coords : pd.DataFrame
        DataFrame with columns ["x","y"] aligned to locations (rows=cells).
    low_q, high_q : float
        Percentile bounds per protein. Typical: 0.1 / 99.9 or 0.5 / 99.5.
    proteins_subset : list or None
        If provided, compute outlier mask only using these proteins (names must exist in Y.columns).
        If None, use all proteins.
    drop_if_any : bool
        If True, drop a row if it is out-of-bounds for ANY selected protein.
        If False, keep rows unless they are out-of-bounds for ALL selected proteins (less aggressive).
    """
    if not isinstance(Y, pd.DataFrame):
        raise TypeError("Y must be a pandas DataFrame so we can subset columns by protein names.")

    if proteins_subset is None:
        Y_use = Y
        use_names = list(Y.columns)
    else:
        missing = [p for p in proteins_subset if p not in Y.columns]
        if missing and verbose:
            print(f"[Outlier filter] Warning: proteins not found and ignored: {missing}", flush=True)
        use_names = [p for p in proteins_subset if p in Y.columns]
        if len(use_names) == 0:
            raise ValueError("proteins_subset provided but none found in Y.columns.")
        Y_use = Y[use_names]

    Y_np = Y_use.to_numpy(dtype=np.float32, copy=False)

    # per-protein bounds
    lo = np.percentile(Y_np, low_q, axis=0)
    hi = np.percentile(Y_np, high_q, axis=0)

    within = (Y_np >= lo) & (Y_np <= hi)

    if drop_if_any:
        inlier_mask = np.all(within, axis=1)
        mode = "ANY-protein outlier removed"
    else:
        inlier_mask = np.any(within, axis=1)
        mode = "ONLY rows that are outliers for ALL proteins removed"

    n_before = Y.shape[0]
    n_after = int(inlier_mask.sum())

    if verbose:
        print(
            f"[Outlier filter] {mode}. Bounds: p{low_q}..p{high_q} "
            f"using {len(use_names)} proteins. Keep {n_after}/{n_before} "
            f"({100*n_after/n_before:.2f}%), removed {n_before-n_after}.",
            flush=True
        )

    # Apply to Y and coords (both DataFrames)
    Y_f = Y.loc[inlier_mask].reset_index(drop=True)
    coords_f = coords.loc[inlier_mask].reset_index(drop=True)

    # Apply to X depending on type
    if isinstance(X, pd.DataFrame):
        X_f = X.loc[inlier_mask].reset_index(drop=True)
    elif sp.issparse(X):
        X_f = X[inlier_mask]
    else:
        X_f = np.asarray(X)[inlier_mask]

    return X_f, Y_f, coords_f


# --------------------------
# Main
# --------------------------
def main():

    tm = Timer()
    tm.log("Start")


    parser = argparse.ArgumentParser(
        description="Predict all proteins from Xenium gene matrix (RDS) with rpy2; 20% held-out."
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--gene_rds", type=str, required=True)
    parser.add_argument("--protein_rds", type=str, required=True)
    parser.add_argument("--loc_csv", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_genes", type=int, default=0, help="Keep top-N variable genes; 0 keep all.")
    parser.add_argument("--model", type=str, default="multitask_enet",
                        choices=["multitask_enet", "ridge"])
    parser.add_argument("--invert_y", action="store_true")
    args = parser.parse_args()

    root = args.root
    protein_path = os.path.join(root, args.protein_rds)
    loc_path = os.path.join(root, args.loc_csv)

    out_dir = os.path.join(root, "protein_prediction_outputs_all")
    os.makedirs(out_dir, exist_ok=True)
    print("Loading protein matrix:", protein_path)
    protein_df = load_rds_matrix(protein_path)
    Y = ensure_numeric_df(protein_df)


    print("Loading locations:", loc_path)
    loc_df = pd.read_csv(loc_path, header=None, skiprows=1)
    loc_df.columns = ["row_id", "x", "y"]
    # Keep x/y only
    coords = loc_df[["x", "y"]].copy()
    coords = ensure_numeric_df(coords)

    h5ad_path = os.path.join(root, "adata_xenium_cell_level.h5ad")
    print("Loading gene matrix from h5ad:", h5ad_path)

    adata = ad.read_h5ad(h5ad_path)
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()
        X = pd.DataFrame.sparse.from_spmatrix(
            X,
            index=adata.obs_names.astype(str),
            columns=adata.var_names.astype(str)
        )
    else:
        X = pd.DataFrame(
            np.asarray(X),
            index=adata.obs_names.astype(str),
            columns=adata.var_names.astype(str)
        )

    # Ensure numeric (safe)
    X = ensure_numeric_df(X)
    nC = coords.shape[0]
    X = orient_to_match_locations(X, nC, "Gene matrix")
    Y = orient_to_match_locations(Y, nC, "Protein matrix")

    X, Y, coords = remove_outlier_rows_by_protein(
        X, Y, coords,
        low_q=0.1, high_q=99.9,  # start conservative
        proteins_subset=None,  # or ["PD.1","HLA.A","PD.L1","CD20","IDO1"]
        drop_if_any=True,
        verbose=True
    )

    nC = coords.shape[0]

    # Now row counts should match
    print(f"Rows after orientation: genes={X.shape[0]}, proteins={Y.shape[0]}, locations={coords.shape[0]}")


    nX, nY, nC = X.shape[0], Y.shape[0], coords.shape[0]


    if not (nX == nY == nC):
        raise ValueError(
            f"Row counts still do not match after orientation. genes={X.shape[0]}, proteins={Y.shape[0]}, locations={coords.shape[0]}"
        )

    if args.max_genes and args.max_genes > 0 and args.max_genes < X.shape[1]:
        var = X.var(axis=0).sort_values(ascending=False)
        keep = var.index[:args.max_genes]
        X = X[keep]
        print(f"Using top-{args.max_genes} variable genes; X shape = {X.shape}")
    else:
        print(f"Using all genes; X shape = {X.shape}")

    protein_names = list(Y.columns)

    idx = np.arange(nX)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )

    X_train = X.iloc[train_idx].values
    X_test = X.iloc[test_idx].values
    Y_train = Y.iloc[train_idx].values
    Y_test = Y.iloc[test_idx].values

    # X transform + scaling
    X_train_t = np.log1p(X_train)
    X_test_t = np.log1p(X_test)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train_t)
    X_test_s = scaler.transform(X_test_t)


    # if args.model == "multitask_enet":
    #     reg = MultiTaskElasticNetCV(
    #         l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    #         alphas=np.logspace(-4, 1, 30),
    #         cv=5,
    #         max_iter=5000,
    #         n_jobs=-1,
    #         random_state=args.seed
    #     )
    # else:
    #     reg = MultiOutputRegressor(Ridge(alpha=10.0, random_state=args.seed))
    reg = Ridge(alpha=10.0, solver="lsqr", tol=1e-3)


    print("Fitting model...")

    # ---- pilot fit to estimate time
    pilot_frac = 0.02  # 2% of training for estimation
    pilot_n = max(5000, int(pilot_frac * X_train_s.shape[0]))
    pilot_idx = np.random.choice(X_train_s.shape[0], size=pilot_n, replace=False)

    tm.log(f"Pilot fit on {pilot_n} samples to estimate runtime...")
    t_p0 = time.time()
    reg.fit(X_train_s[pilot_idx], Y_train[pilot_idx])
    t_p1 = time.time()

    # crude linear extrapolation in n_samples (often reasonable for these solvers)
    est = (t_p1 - t_p0) * (X_train_s.shape[0] / pilot_n)
    tm.log(f"Pilot fit time: {t_p1 - t_p0:.1f}s; estimated full fit: ~{est / 60:.1f} minutes")

    tm.log("Fitting full model...")
    reg.fit(X_train_s, Y_train)
    tm.log("Fit done")


    Y_pred = reg.predict(X_test_s)

    metrics_df = per_protein_metrics(Y_test, Y_pred, protein_names)
    metrics_df_sorted = metrics_df.sort_values("R2", ascending=False)

    metrics_csv = os.path.join(out_dir, "protein_metrics_heldout.csv")
    metrics_df_sorted.to_csv(metrics_csv, index=False)
    print("Saved metrics:", metrics_csv)

    top3 = metrics_df_sorted.head(3)["protein"].tolist()
    print("Top-5 proteins by held-out R2:", top3)

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
            n_bins=250,  # adjust
            agg="mean",
            min_count=5,
            out_prefix=out_prefix
        )

        print(f"[{pname}] Grid metrics: {metrics}")
        print(f"[{pname}] Saved: {scatter_png}")


    summary = {
        "n_total": int(nX),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "n_genes_used": int(X.shape[1]),
        "n_proteins": int(Y.shape[1]),
        "model": args.model,
        "top5_by_R2": top3,
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Top-5 for plotting
    top5_idx = [protein_names.index(p) for p in top3]
    Y_test_top = Y_test[:, top5_idx]
    Y_pred_top = Y_pred[:, top5_idx]

    x = coords.iloc[test_idx]["x"].values
    y = coords.iloc[test_idx]["y"].values

    out_png = os.path.join(out_dir, "top5_spatial_true_vs_pred.png")
    plot_top3_spatial(
        true_vals=Y_test_top,
        pred_vals=Y_pred_top,
        x=x, y=y,
        protein_names=top3,
        out_png=out_png,
        invert_y=args.invert_y
    )
    print("Saved plot:", out_png)

    print("Done.")


if __name__ == "__main__":
    main()
