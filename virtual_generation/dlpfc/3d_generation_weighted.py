#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from scipy.spatial import cKDTree
from sklearn.decomposition import NMF

from axis.coordinate_utils import find_inverse_warp_coords


# =========================
# CONFIG
# =========================
GROUP_ID = 1
ANCHOR0 = 0
ANCHOR3 = 3

ROOT = "/media/huifang/data/registration/DLPFC/huifang"
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ad")

AXIS_OUT_CANDIDATES = [
    os.path.join(ROOT, "output", "group_{}".format(GROUP_ID)),
    os.path.join(ROOT, "output", "group_[{}]".format(GROUP_ID)),
]

# z positions (um) for your 4 slices (edit if needed)
Z_UM = {0: 0, 1: 10, 2: 310, 3: 320}
STEP_UM = 10
Z_START = int(Z_UM[ANCHOR0])
Z_END = int(Z_UM[ANCHOR3])
Z_GRID = list(range(Z_START, Z_END + 1, STEP_UM))

# global grid resolution
GRID_SIZE = 96
MAX_DIST_PX = 9.0  # used for grid support mask

# ---- Weighting ----
# Anchor z-weights:
#   slice0 candidates weight = (1 - t)
#   slice3 candidates weight = t
# Additional spatial distance weight (spot -> assigned grid cell center):
USE_SPATIAL_DIST_WEIGHT = True
# If None, auto set to 0.5 * cell_size (in coordinate units)
SIGMA_SPATIAL = None

EPS = 1e-8

# NMF per slice
N_COMPONENTS = 20
NMF_MAX_ITER = 200
NMF_INIT = "nndsvda"   # faster/stabler than random
NMF_RANDOM_STATE = 0

OUT_DIR = os.path.join(
    ROOT,
    "generated_3d_fixedgrid_per_slice_nmf",
    f"generated_grid{GRID_SIZE}_step{STEP_UM}um_weighted"
)
os.makedirs(OUT_DIR, exist_ok=True)

# optional preprocessing
APPLY_LOG1P_BEFORE_NMF = False


# =========================
# PATH / IO HELPERS
# =========================
def _find_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

AXIS_OUT = _find_existing_path(AXIS_OUT_CANDIDATES)
if AXIS_OUT is None:
    raise FileNotFoundError("Cannot find AXIS output folder in: {}".format(AXIS_OUT_CANDIDATES))

def load_coords_npz(group_id, sample_id):
    path = os.path.join(NET_IN, "{}_{}_validation.npz".format(group_id, sample_id))
    d = np.load(path)
    return d["coord"].astype(np.float32)

def load_axis_flow(fixed_id, moving_id):
    path = os.path.join(AXIS_OUT, "{}_{}_flow.npy".format(fixed_id, moving_id))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)

def load_h5ad_X_and_genes(group_id, sample_id):
    path = os.path.join(H5AD_DIR, "{}_{}.h5ad".format(group_id, sample_id))
    ad = sc.read_h5ad(path)
    return ad.X, np.asarray(ad.var_names)

def ensure_same_genes_keep_order(Xa, ga, Xb, gb):
    """
    Intersect genes, KEEP slice0 order, subset both matrices.
    """
    ga = np.asarray(ga)
    gb = np.asarray(gb)
    gb_set = set(gb.tolist())
    common = [g for g in ga.tolist() if g in gb_set]
    if len(common) == 0:
        raise ValueError("No common genes between anchors.")
    idx_a = np.array([int(np.where(ga == g)[0][0]) for g in common], dtype=np.int64)
    idx_b = np.array([int(np.where(gb == g)[0][0]) for g in common], dtype=np.int64)
    return Xa[:, idx_a], Xb[:, idx_b], np.asarray(common)

def make_common_grid_from_coords(coords, grid_size, pad_ratio=0.02):
    """
    coords: (N,2)
    Returns: grid_pts (M,2), (x_edges, y_edges)
    """
    coords = np.asarray(coords, dtype=np.float32)
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)

    dx = (max_x - min_x) * pad_ratio
    dy = (max_y - min_y) * pad_ratio
    min_x -= dx; max_x += dx
    min_y -= dy; max_y += dy

    x_edges = np.linspace(min_x, max_x, grid_size + 1, dtype=np.float32)
    y_edges = np.linspace(min_y, max_y, grid_size + 1, dtype=np.float32)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    gx, gy = np.meshgrid(x_centers, y_centers)
    grid_pts = np.vstack([gx.ravel(), gy.ravel()]).T.astype(np.float32)
    return grid_pts, (x_edges, y_edges)

def compute_grid_support_mask(grid_pts, spot_pts_union, max_dist_px):
    """
    grid_pts: (M,2)
    spot_pts_union: (N,2)
    Returns mask_support: (M,) bool where grid cell is within max_dist_px of any spot
    """
    tree = cKDTree(np.asarray(spot_pts_union, dtype=np.float32))
    dist, _ = tree.query(np.asarray(grid_pts, dtype=np.float32), k=1)
    return (dist.astype(np.float32) <= float(max_dist_px))


# =========================
# WEIGHTED fixed-grid resample + KL-NMF
# =========================
def spatial_resample_nmf_fixed_grid(
    coords, X, grid_pts, x_edges, y_edges,
    n_components=20,
    fill_empty=True,
    grid_support_mask=None,   # (M,) bool, optional
    spot_weights=None,        # (N,) float32 optional (candidate-spot weights)
    use_spatial_dist_weight=True,
    sigma_spatial=None,
    nmf_init="nndsvda",
    nmf_max_iter=200,
    nmf_random_state=0,
):
    """
    Weighted bin -> (optional fill) -> KL-NMF on a fixed global grid.

    coords: (N,2) projected points
    X: (N,G) expression aligned to coords (sparse or dense)
    spot_weights: (N,) base weights (e.g., z-weight: w0=1-t, w3=t)
    use_spatial_dist_weight: multiply by exp(-d^2/(2*sigma^2)) where d is distance
                             from point to its assigned grid cell center.
    """
    coords = np.asarray(coords, dtype=np.float32)

    # infer grid_size from edges; M must match
    grid_size = int(len(x_edges) - 1)
    M = int(grid_size * grid_size)
    if grid_pts.shape[0] != M:
        raise ValueError("grid_pts length mismatch: {} vs {}".format(grid_pts.shape[0], M))

    # densify like your original function behavior
    if sp.issparse(X):
        Xd = X.toarray().astype(np.float32)
    else:
        Xd = np.asarray(X, dtype=np.float32)

    N = coords.shape[0]
    if spot_weights is None:
        spot_weights = np.ones((N,), dtype=np.float32)
    else:
        spot_weights = np.asarray(spot_weights, dtype=np.float32)
        if spot_weights.shape[0] != N:
            raise ValueError("spot_weights length mismatch: {} vs {}".format(spot_weights.shape[0], N))

    # bin points into fixed grid
    x_bin = np.digitize(coords[:, 0], x_edges) - 1
    y_bin = np.digitize(coords[:, 1], y_edges) - 1
    x_bin = np.clip(x_bin, 0, grid_size - 1)
    y_bin = np.clip(y_bin, 0, grid_size - 1)
    cell_id = (y_bin.astype(np.int64) * grid_size + x_bin.astype(np.int64))  # (N,)

    # spatial distance weight (point -> cell center)
    weights = spot_weights.copy()
    if use_spatial_dist_weight:
        # auto sigma: half a cell size (in same coordinate units as coords)
        if sigma_spatial is None:
            cell_dx = float(x_edges[1] - x_edges[0])
            cell_dy = float(y_edges[1] - y_edges[0])
            sigma_spatial = 0.5 * float(min(cell_dx, cell_dy))
        sigma2 = float(sigma_spatial) * float(sigma_spatial) + EPS

        centers = grid_pts[cell_id]  # (N,2)
        diff = coords - centers
        d2 = (diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]).astype(np.float32)
        w_sp = np.exp(-d2 / (2.0 * sigma2)).astype(np.float32)
        weights *= w_sp

    # Weighted accumulation: grid_sum and w_sum
    flat_sum = np.zeros((M, Xd.shape[1]), dtype=np.float32)
    w_sum = np.zeros((M,), dtype=np.float32)

    for i in range(N):
        w = float(weights[i])
        if w <= 0.0:
            continue
        cid = int(cell_id[i])
        flat_sum[cid] += w * Xd[i]
        w_sum[cid] += w

    # Weighted mean per cell (prevents magnitude blow-up when both anchors contribute)
    valid_by_weight = w_sum > 0
    flat_grid = np.zeros_like(flat_sum, dtype=np.float32)
    flat_grid[valid_by_weight] = flat_sum[valid_by_weight] / (w_sum[valid_by_weight, None] + EPS)

    # ---- fit mask logic ----
    if grid_support_mask is not None:
        grid_support_mask = np.asarray(grid_support_mask, dtype=bool)
        if grid_support_mask.shape[0] != M:
            raise ValueError("grid_support_mask length mismatch")

        # If fill is ON, NMF should see all support cells (since they'll be filled)
        # If fill is OFF, only cells with actual support should be used
        if fill_empty:
            fit_mask = grid_support_mask.copy()
        else:
            fit_mask = valid_by_weight & grid_support_mask
    else:
        fit_mask = np.ones((M,), dtype=bool) if fill_empty else valid_by_weight

    # ---- optional fill (nearest non-empty) ----
    if fill_empty:
        # fill ONLY inside support if provided; never fill outside tissue
        fill_region = fit_mask.copy()

        w2 = w_sum.reshape(grid_size, grid_size)
        fr2 = fill_region.reshape(grid_size, grid_size)

        non_empty_idx = np.argwhere((w2 > 0) & fr2)
        empty_idx = np.argwhere((w2 == 0) & fr2)

        if empty_idx.shape[0] > 0 and non_empty_idx.shape[0] > 0:
            tree = cKDTree(non_empty_idx)
            _, nn = tree.query(empty_idx, k=1)

            base = flat_grid.reshape(grid_size, grid_size, -1)
            out = base.copy()
            for k, (r, c) in enumerate(empty_idx):
                rr, cc = non_empty_idx[nn[k]]
                out[r, c, :] = base[rr, cc, :]
            flat_grid = out.reshape(M, -1).astype(np.float32, copy=False)

    # ---- NMF: fit on fit_mask rows only; write back ----
    smoothed = np.zeros_like(flat_grid, dtype=np.float32)
    if np.any(fit_mask):
        X_fit = flat_grid[fit_mask]
        model = NMF(
            n_components=n_components,
            solver="mu",
            beta_loss="kullback-leibler",
            init=nmf_init,
            random_state=nmf_random_state,
            max_iter=nmf_max_iter,
            tol=1e-4,
            verbose=False,
        )
        W = model.fit_transform(X_fit)
        H = model.components_.astype(np.float32, copy=False)
        smoothed_fit = (W @ H).astype(np.float32, copy=False)
        smoothed[fit_mask] = smoothed_fit

    ad = AnnData(X=smoothed, dtype=smoothed.dtype)
    ad.obsm["spatial"] = np.asarray(grid_pts, dtype=np.float32)
    ad.obs["valid"] = fit_mask.astype(bool)
    return ad


# =========================
# MAIN
# =========================
def main():
    # Load coords
    X0 = load_coords_npz(GROUP_ID, ANCHOR0)
    X3 = load_coords_npz(GROUP_ID, ANCHOR3)

    # Load flows (convention: fixed_moving means moving->fixed)
    flow_3_0 = load_axis_flow(fixed_id=3, moving_id=0)  # 0 -> 3
    flow_0_3 = load_axis_flow(fixed_id=0, moving_id=3)  # 3 -> 0

    # Load and align genes
    G0_raw, genes0 = load_h5ad_X_and_genes(GROUP_ID, ANCHOR0)
    G3_raw, genes3 = load_h5ad_X_and_genes(GROUP_ID, ANCHOR3)
    G0_raw, G3_raw, genes = ensure_same_genes_keep_order(G0_raw, genes0, G3_raw, genes3)

    # Defensive row alignment
    n0 = min(X0.shape[0], G0_raw.shape[0])
    n3 = min(X3.shape[0], G3_raw.shape[0])
    X0, G0_raw = X0[:n0], G0_raw[:n0]
    X3, G3_raw = X3[:n3], G3_raw[:n3]

    # Build ONE common grid in slice0 frame
    X3_to0 = find_inverse_warp_coords(X3, flow_0_3).astype(np.float32)
    union_coords = np.vstack([X0, X3_to0]).astype(np.float32)
    grid_pts, (x_edges, y_edges) = make_common_grid_from_coords(union_coords, GRID_SIZE)

    # Support mask to avoid full-space background
    grid_support = compute_grid_support_mask(grid_pts, union_coords, MAX_DIST_PX)

    # Manifest
    manifest = {
        "group_id": GROUP_ID,
        "anchors": [ANCHOR0, ANCHOR3],
        "z_um": Z_UM,
        "z_grid": Z_GRID,
        "step_um": STEP_UM,
        "grid_size": GRID_SIZE,
        "weights": {
            "anchor_z": "w0=1-t, w3=t (applied per candidate spot)",
            "use_spatial_dist_weight": USE_SPATIAL_DIST_WEIGHT,
            "sigma_spatial": SIGMA_SPATIAL,
        },
        "nmf": {
            "n_components": N_COMPONENTS,
            "beta_loss": "kullback-leibler",
            "solver": "mu",
            "init": NMF_INIT,
            "max_iter": NMF_MAX_ITER,
            "random_state": NMF_RANDOM_STATE,
        },
        "apply_log1p_before_nmf": APPLY_LOG1P_BEFORE_NMF,
        "fill_empty": True,
        "note": (
            "Global fixed grid built once in slice0 frame from union(X0, X3->0). "
            "Per z: linear warp X0_t and X3_t, concat points+expr, weighted bin (z-weight + optional spatial), "
            "then (optional fill) + KL-NMF on the fixed grid."
        ),
        "axis_out": AXIS_OUT,
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    denom = float(Z_END - Z_START) if (Z_END != Z_START) else 1.0

    for z in Z_GRID:
        t = float((z - Z_START) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        print("[GEN] z={}um  t={:.3f}".format(z, t))

        # Linear partial warp
        X0_t = find_inverse_warp_coords(X0, (t * flow_3_0).astype(np.float32)).astype(np.float32)
        X3_t = find_inverse_warp_coords(X3, ((1.0 - t) * flow_0_3).astype(np.float32)).astype(np.float32)

        # concat coords
        coords_cat = np.vstack([X0_t, X3_t]).astype(np.float32)

        # concat expressions
        if sp.issparse(G0_raw) or sp.issparse(G3_raw):
            X_cat = sp.vstack([G0_raw, G3_raw]).tocsr()
        else:
            X_cat = np.vstack([np.asarray(G0_raw), np.asarray(G3_raw)]).astype(np.float32)

        # optional log1p
        if APPLY_LOG1P_BEFORE_NMF:
            if sp.issparse(X_cat):
                X_cat = X_cat.copy()
                X_cat.data = np.log1p(X_cat.data).astype(np.float32)
            else:
                X_cat = np.log1p(X_cat).astype(np.float32)

        # ---- candidate spot weights: z-distance ----
        w0 = (1.0 - t)
        w3 = t
        w_vec = np.concatenate([
            np.full((X0_t.shape[0],), w0, dtype=np.float32),
            np.full((X3_t.shape[0],), w3, dtype=np.float32),
        ], axis=0)

        ad_grid = spatial_resample_nmf_fixed_grid(
            coords=coords_cat,
            X=X_cat,
            grid_pts=grid_pts,
            x_edges=x_edges,
            y_edges=y_edges,
            n_components=N_COMPONENTS,
            fill_empty=True,
            grid_support_mask=grid_support,
            spot_weights=w_vec,
            use_spatial_dist_weight=USE_SPATIAL_DIST_WEIGHT,
            sigma_spatial=SIGMA_SPATIAL,
            nmf_init=NMF_INIT,
            nmf_max_iter=NMF_MAX_ITER,
            nmf_random_state=NMF_RANDOM_STATE,
        )

        # set genes + metadata
        ad_grid.var_names = genes
        ad_grid.uns["z_um"] = float(z)
        ad_grid.uns["t"] = float(t)
        ad_grid.uns["grid_size"] = int(GRID_SIZE)
        ad_grid.uns["n_components"] = int(N_COMPONENTS)
        ad_grid.uns["apply_log1p_before_nmf"] = bool(APPLY_LOG1P_BEFORE_NMF)
        ad_grid.uns["source_anchors"] = [ANCHOR0, ANCHOR3]
        ad_grid.uns["w0"] = float(w0)
        ad_grid.uns["w3"] = float(w3)
        ad_grid.uns["use_spatial_dist_weight"] = bool(USE_SPATIAL_DIST_WEIGHT)
        ad_grid.uns["sigma_spatial"] = float(SIGMA_SPATIAL) if SIGMA_SPATIAL is not None else None

        out_path = os.path.join(OUT_DIR, "gen_grid_z{}um.h5ad".format(int(z)))
        ad_grid.write_h5ad(out_path)
        print("  saved:", out_path)

    print("DONE. Output folder:", OUT_DIR)


if __name__ == "__main__":
    main()
