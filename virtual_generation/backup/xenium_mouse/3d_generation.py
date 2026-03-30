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
from scipy.ndimage import binary_closing, binary_fill_holes, label

# =========================
# CONFIG
# =========================
GROUP_ID = 0
ANCHOR_start = 1
ANCHOR_end = 2

ROOT = "/media/huifang/data/registration/xenium/mouse_brain"
NET_IN  = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ads/Tg")
# H5AD_DIR = os.path.join(ROOT, "h5ads/widetype")
AXIS_OUT = os.path.join(ROOT, "pairwise_output", f"group_{GROUP_ID}")  # adjust if needed




# z positions (um) for your 4 slices (edit if needed)
Z_UM = {0: 2.5, 1: 5.7, 2: 17.9}
# Z_UM = {3: 2.5, 4: 5.7, 5: 13.4}
STEP_UM = 1.2
Z_START = Z_UM[ANCHOR_start]
Z_END = Z_UM[ANCHOR_end]
# Z_GRID = np.arange(Z_START, Z_END, STEP_UM).tolist()
Z_GRID = [13.4]

# global grid resolution
GRID_SIZE = 128
MAX_DIST_PX=5.0
# NMF per slice (same idea as your function)
N_COMPONENTS = 1
NMF_MAX_ITER = 200
NMF_INIT = "nndsvda"   # faster/stabler than random
NMF_RANDOM_STATE = 0
OUT_DIR = os.path.join(ROOT, "virtual_tissue", f"grid{GRID_SIZE}")
os.makedirs(OUT_DIR, exist_ok=True)
# optional preprocessing (keep False to match your previous “raw-count-like” behavior)
APPLY_LOG1P_BEFORE_NMF = False


# =========================
# PATH / IO HELPERS
# =========================



def compute_dense_tissue_mask_on_grid(
    grid_pts, union_spots,
    grid_size,
    max_dist_px=9.0,
    closing_iters=2,
    keep_largest=True,
):
    """
    Returns mask (M,) bool on grid cells: a solid tissue region with holes filled.
    """
    # 1) initial distance-threshold mask
    tree = cKDTree(np.asarray(union_spots, dtype=np.float32))
    dist, _ = tree.query(np.asarray(grid_pts, dtype=np.float32), k=1)
    mask = (dist <= float(max_dist_px)).reshape(grid_size, grid_size)

    # 2) connect gaps (closing) + 3) fill holes
    if closing_iters > 0:
        mask = binary_closing(mask, iterations=int(closing_iters))
    mask = binary_fill_holes(mask)

    # 4) optionally keep only the largest connected component
    if keep_largest:
        lab, n = label(mask)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0  # background
            keep = sizes.argmax()
            mask = (lab == keep)

    return mask.reshape(-1)

def _find_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None



def load_coords_npz(group_id, sample_id):
    path = os.path.join(NET_IN, "{}_{}_validation.npz".format(group_id, sample_id))
    d = np.load(path)
    return d["coord"].astype(np.float32)

def load_axis_flow(fixed_id, moving_id):
    path = os.path.join(AXIS_OUT, "{}_{}_flow.npy".format(fixed_id, moving_id))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)

def load_h5ad_X_and_genes(t):
    path = os.path.join(H5AD_DIR, f"xenium_Tg_{t}m.h5ad")
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
    pad_ratio: small padding to avoid clipping boundary spots
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

from scipy.spatial import cKDTree

def compute_grid_support_mask(grid_pts, spot_pts_union, max_dist_px):
    """
    grid_pts: (M,2)
    spot_pts_union: (N,2) e.g., vstack([X0, X3_to0])
    Returns mask_support: (M,) bool where grid cell is within max_dist_px of any spot
    """
    tree = cKDTree(np.asarray(spot_pts_union, dtype=np.float32))
    dist, _ = tree.query(np.asarray(grid_pts, dtype=np.float32), k=1)
    return (dist.astype(np.float32) <= float(max_dist_px))



# =========================
# FIXED-GRID VERSION OF YOUR spatial_resample_nmf
# (same algorithm: bin -> fill -> KL-NMF, but using provided edges/grid)
# =========================
def spatial_resample_nmf_fixed_grid(
    coords, X, grid_pts, x_edges, y_edges,
    n_components=20,
    fill_empty=True,
    grid_support_mask=None,   # (M,) bool, optional
    nmf_init="nndsvda",
    nmf_max_iter=800,
    nmf_random_state=0,
):
    """
    coords: (N,2) projected points for this t
    X: (N,G) expression for those points (sparse or dense)
    grid_pts: (M,2) fixed grid centers
    x_edges/y_edges: fixed bin edges
    """
    coords = np.asarray(coords, dtype=np.float32)

    # infer grid_size from edges; M must match
    grid_size = int(len(x_edges) - 1)
    M = int(grid_size * grid_size)
    if grid_pts.shape[0] != M:
        raise ValueError("grid_pts length mismatch: {} vs {}".format(grid_pts.shape[0], M))

    # densify (same as your original function behavior)
    if sp.issparse(X):
        Xd = X.toarray().astype(np.float32)
    else:
        Xd = np.asarray(X, dtype=np.float32)

    # bin points into fixed grid
    x_bin = np.digitize(coords[:, 0], x_edges) - 1
    y_bin = np.digitize(coords[:, 1], y_edges) - 1
    x_bin = np.clip(x_bin, 0, grid_size - 1)
    y_bin = np.clip(y_bin, 0, grid_size - 1)
    cell_id = (y_bin.astype(np.int64) * grid_size + x_bin.astype(np.int64))

    flat_grid = np.zeros((M, Xd.shape[1]), dtype=np.float32)
    counts = np.zeros((M,), dtype=np.float32)
    for i in range(coords.shape[0]):
        cid = int(cell_id[i])
        flat_grid[cid] += Xd[i]
        counts[cid] += 1.0

    # cells hit by at least one point
    valid_by_points = counts > 0

    # optional: only consider a support region (recommended)
    if grid_support_mask is not None:
        grid_support_mask = np.asarray(grid_support_mask, dtype=bool)
        if grid_support_mask.shape[0] != M:
            raise ValueError("grid_support_mask length mismatch")
        fit_mask = valid_by_points & grid_support_mask
    else:
        fit_mask = valid_by_points

    # optional fill (you asked to disable this for interpolation)
    if fill_empty:
        # fill ONLY inside support (if provided); never fill background
        if grid_support_mask is None:
            fill_region = np.ones((M,), dtype=bool)
        else:
            fill_region = grid_support_mask.copy()

        counts2 = counts.reshape(grid_size, grid_size)
        fill_region2 = fill_region.reshape(grid_size, grid_size)

        non_empty_idx = np.argwhere((counts2 > 0) & fill_region2)
        empty_idx = np.argwhere((counts2 == 0) & fill_region2)

        if empty_idx.shape[0] > 0 and non_empty_idx.shape[0] > 0:
            tree = cKDTree(non_empty_idx)
            _, nn = tree.query(empty_idx, k=1)
            base = flat_grid.reshape(grid_size, grid_size, -1)
            out = base.copy()
            for k, (r, c) in enumerate(empty_idx):
                rr, cc = non_empty_idx[nn[k]]
                out[r, c, :] = base[rr, cc, :]
            flat_grid = out.reshape(M, -1).astype(np.float32, copy=False)

    # ---- NMF: fit on supported rows only, then write back (much better than fitting on all M) ----
    # If nothing to fit (rare), return zeros
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


def spatial_resample_wasserstein_barycenter_fixed_grid(
    coords0, X0, coords3, X3,
    grid_pts, x_edges, y_edges,
    t,
    grid_support_mask=None,          # (M,) bool or None
    reg=0.01,                        # entropic regularization (bigger = smoother/faster)
    n_iter=200,                      # Sinkhorn barycenter iterations inside POT
    tol=1e-7,
    chunk_genes=256,                 # process genes in chunks to control RAM
    eps=1e-12,
):
    """
    Wasserstein barycenter generation on a fixed 2D grid.

    - Bins slice0-at-t and slice3-at-t to 2D grids per gene.
    - For each gene: computes 2D entropic Wasserstein barycenter between the two histograms.
    - Returns AnnData with X=(M,G) on grid_pts.

    REQUIREMENT: POT installed -> pip install pot
    """
    try:
        import ot
        from ot.bregman import convolutional_barycenter2d
    except Exception as e:
        raise ImportError(
            "POT is required for Wasserstein barycenter. Install with: pip install pot"
        ) from e

    coords0 = np.asarray(coords0, dtype=np.float32)
    coords3 = np.asarray(coords3, dtype=np.float32)

    # infer grid
    grid_size = int(len(x_edges) - 1)
    M = grid_size * grid_size
    if grid_pts.shape[0] != M:
        raise ValueError("grid_pts length mismatch")

    # densify X for arithmetic (you can optimize later if needed)
    if sp.issparse(X0):
        X0d = X0.toarray().astype(np.float32)
    else:
        X0d = np.asarray(X0, dtype=np.float32)
    if sp.issparse(X3):
        X3d = X3.toarray().astype(np.float32)
    else:
        X3d = np.asarray(X3, dtype=np.float32)

    G = X0d.shape[1]
    if X3d.shape[1] != G:
        raise ValueError("X0 and X3 gene dimension mismatch")

    # --- helper: bin points to grid (weighted mean per cell; here uniform weights) ---
    def _bin_to_grid(coords, Xd):
        x_bin = np.digitize(coords[:, 0], x_edges) - 1
        y_bin = np.digitize(coords[:, 1], y_edges) - 1
        x_bin = np.clip(x_bin, 0, grid_size - 1)
        y_bin = np.clip(y_bin, 0, grid_size - 1)
        cell_id = (y_bin.astype(np.int64) * grid_size + x_bin.astype(np.int64))

        grid_sum = np.zeros((M, G), dtype=np.float32)
        w_sum = np.zeros((M,), dtype=np.float32)

        for i in range(coords.shape[0]):
            cid = int(cell_id[i])
            grid_sum[cid] += Xd[i]
            w_sum[cid] += 1.0

        valid = w_sum > 0
        grid = np.zeros_like(grid_sum, dtype=np.float32)
        grid[valid] = grid_sum[valid] / (w_sum[valid, None] + eps)
        return grid

    grid0 = _bin_to_grid(coords0, X0d)  # (M,G)
    grid3 = _bin_to_grid(coords3, X3d)  # (M,G)

    # optional: restrict to tissue support
    if grid_support_mask is not None:
        mask = np.asarray(grid_support_mask, dtype=bool)
        if mask.shape[0] != M:
            raise ValueError("grid_support_mask length mismatch")
        grid0 = grid0 * mask[:, None]
        grid3 = grid3 * mask[:, None]

    # reshape to 2D images for convolutional barycenter
    grid0_2d = grid0.reshape(grid_size, grid_size, G)
    grid3_2d = grid3.reshape(grid_size, grid_size, G)

    w = np.array([1.0 - float(t), float(t)], dtype=np.float64)

    out = np.zeros((grid_size, grid_size, G), dtype=np.float32)

    # process genes in chunks (still O(G) barycenters; expensive but correct)
    for s in range(0, G, int(chunk_genes)):
        e = min(G, s + int(chunk_genes))

        for j in range(s, e):
            A0 = grid0_2d[:, :, j].astype(np.float64, copy=False)
            A3 = grid3_2d[:, :, j].astype(np.float64, copy=False)

            m0 = float(A0.sum())
            m3 = float(A3.sum())
            mt = (1.0 - float(t)) * m0 + float(t) * m3

            if mt <= 0.0:
                continue  # stays zero

            # balanced OT barycenter expects probability measures
            P0 = A0 / (m0 + eps) if m0 > 0 else np.full_like(A0, 1.0 / (grid_size * grid_size))
            P3 = A3 / (m3 + eps) if m3 > 0 else np.full_like(A3, 1.0 / (grid_size * grid_size))

            A = np.stack([P0, P3], axis=0)  # (2, H, W)

            # 2D entropic Wasserstein barycenter (convolutional Sinkhorn)
            B = convolutional_barycenter2d(
                A, reg=reg, weights=w, numItermax=n_iter, stopThr=tol
            )

            # rescale back to interpolated total mass
            out[:, :, j] = (B * mt).astype(np.float32, copy=False)

    X_out = out.reshape(M, G).astype(np.float32, copy=False)
    ad = AnnData(X=X_out, dtype=X_out.dtype)
    ad.obsm["spatial"] = np.asarray(grid_pts, dtype=np.float32)
    if grid_support_mask is not None:
        ad.obs["valid"] = np.asarray(grid_support_mask, dtype=bool)
    else:
        ad.obs["valid"] = np.ones((M,), dtype=bool)
    ad.uns["t"] = float(t)
    ad.uns["ot"] = {"reg": float(reg), "n_iter": int(n_iter), "tol": float(tol)}
    return ad

# =========================
# MAIN
# =========================
def main():
    # Load coords
    X0 = load_coords_npz(GROUP_ID, ANCHOR_start)
    X3 = load_coords_npz(GROUP_ID, ANCHOR_end)

    # Load flows (convention: fixed_moving means moving->fixed)
    flow_3_0 = load_axis_flow(fixed_id=ANCHOR_end, moving_id=ANCHOR_start)  # 0 -> 3
    flow_0_3 = load_axis_flow(fixed_id=ANCHOR_start, moving_id=ANCHOR_end)  # 3 -> 0

    # Load and align genes
    G0_raw, genes0 = load_h5ad_X_and_genes(Z_START)
    G3_raw, genes3 = load_h5ad_X_and_genes(Z_END)
    G0_raw, G3_raw, genes = ensure_same_genes_keep_order(G0_raw, genes0, G3_raw, genes3)

    GENE_SUBSET = ["Arc"]  # whatever you want

    # after ensure_same_genes_keep_order(...)
    gene_to_idx = {g: i for i, g in enumerate(genes.tolist())}
    keep_idx = [gene_to_idx[g] for g in GENE_SUBSET if g in gene_to_idx]

    if len(keep_idx) == 0:
        raise ValueError("None of the requested genes found in common gene list.")

    genes = genes[keep_idx]
    G0_raw = G0_raw[:, keep_idx]
    G3_raw = G3_raw[:, keep_idx]

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
    # grid_support = compute_grid_support_mask(grid_pts, union_coords, MAX_DIST_PX)
    grid_support = compute_dense_tissue_mask_on_grid(
        grid_pts=grid_pts,
        union_spots=union_coords,
        grid_size=GRID_SIZE,
        max_dist_px=MAX_DIST_PX,
        closing_iters=1,  # try 1~3
        keep_largest=True
    )


    # Manifest
    manifest = {
        "group_id": GROUP_ID,
        "anchors": [ANCHOR_start, ANCHOR_end],
        "z_um": Z_UM,
        "z_grid": Z_GRID,
        "step_um": STEP_UM,
        "grid_size": GRID_SIZE,
        "nmf": {
            "n_components": N_COMPONENTS,
            "beta_loss": "kullback-leibler",
            "solver": "mu",
            "init": NMF_INIT,
            "max_iter": NMF_MAX_ITER,
            "random_state": NMF_RANDOM_STATE,
        },
        "apply_log1p_before_nmf": APPLY_LOG1P_BEFORE_NMF,
        "note": (
            "Global fixed grid built once in slice0 frame from union(X0, X3->0). "
            "Per z: linear warp X0_t and X3_t, concat points+expr, then bin/fill/KL-NMF on the fixed grid."
        ),
        "axis_out": AXIS_OUT,
    }
    with open(os.path.join(OUT_DIR, f"{ANCHOR_start}_{ANCHOR_end}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    denom = float(Z_END - Z_START) if (Z_END != Z_START) else 1.0

    for z in Z_GRID:
        t = float((z - Z_START) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        print("[GEN] z={}um  t={:.3f}".format(z, t))

        # Linear partial warp (same as your projection debug)
        X0_t = find_inverse_warp_coords(X0, (t * flow_3_0).astype(np.float32)).astype(np.float32)
        X3_t = find_inverse_warp_coords(X3, ((1.0 - t) * flow_0_3).astype(np.float32)).astype(np.float32)

        # ad_grid = spatial_resample_wasserstein_barycenter_fixed_grid(
        #     coords0=X0_t, X0=G0_raw,
        #     coords3=X3_t, X3=G3_raw,
        #     grid_pts=grid_pts, x_edges=x_edges, y_edges=y_edges,
        #     t=t,
        #     grid_support_mask=grid_support,
        #     reg=0.001,  # tune: 0.005–0.05
        #     n_iter=500,
        #     chunk_genes=256,  # tune for speed/RAM
        # )
        # ad_grid.var_names = genes

        coords_cat = np.vstack([X0_t, X3_t]).astype(np.float32)




        # concat expressions
        if sp.issparse(G0_raw) or sp.issparse(G3_raw):
            X_cat = sp.vstack([G0_raw, G3_raw]).tocsr()
        else:
            X_cat = np.vstack([np.asarray(G0_raw), np.asarray(G3_raw)]).astype(np.float32)

        w0 = (1.0 - t)
        w3 = t
        w_vec = np.concatenate([
            np.full((X0_t.shape[0],), w0, dtype=np.float32),
            np.full((X3_t.shape[0],), w3, dtype=np.float32),
        ], axis=0)

        X_cat = X_cat.multiply(w_vec[:, None])


        # optional log1p
        if APPLY_LOG1P_BEFORE_NMF:
            if sp.issparse(X_cat):
                X_cat = X_cat.copy()
                X_cat.data = np.log1p(X_cat.data).astype(np.float32)
            else:
                X_cat = np.log1p(X_cat).astype(np.float32)




        # Then per z:
        ad_grid = spatial_resample_nmf_fixed_grid(
            coords=coords_cat,
            X=X_cat,
            grid_pts=grid_pts,
            x_edges=x_edges,
            y_edges=y_edges,
            n_components=N_COMPONENTS,
            fill_empty=True,
            grid_support_mask=grid_support,  # <-- avoids full-space
        )

        # set genes + metadata
        ad_grid.var_names = genes
        ad_grid.uns["z_um"] = float(z)
        ad_grid.uns["t"] = float(t)
        ad_grid.uns["grid_size"] = int(GRID_SIZE)
        ad_grid.uns["n_components"] = int(N_COMPONENTS)
        ad_grid.uns["apply_log1p_before_nmf"] = bool(APPLY_LOG1P_BEFORE_NMF)
        ad_grid.uns["source_anchors"] = [ANCHOR_start, ANCHOR_end]

        out_path = os.path.join(OUT_DIR, f"gen_grid_z{z:.1f}um.h5ad")
        ad_grid.write_h5ad(out_path)
        print("  saved:", out_path)

    print("DONE. Output folder:", OUT_DIR)


if __name__ == "__main__":
    main()
