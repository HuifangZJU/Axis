#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlined standalone DLPFC pixel-grid low-rank reconstruction.

What it does
------------
1. Load the integrated DLPFC h5ad.
2. Reconstruct a low-rank representation on a regular square pixel grid using
   sparse OT + adaptive SVT + pixel-lattice diffusion.
3. Save the regenerated AnnData.
4. Visualize a marker gene (default PCP4) in two low-rank output modes:
   (a) spot-level scatter on reconstructed spatial locations;
   (b) pixel-image rendering on the reconstructed square grid with white background.
5. Also save a compact original-vs-low-rank spot comparison for reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


# =============================================================================
# CONFIG
# =============================================================================
INPUT_H5AD = Path(
    "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/"
    "dlpfc_z_group1_grid200_spatial0.3/integration/"
    "integrated_dlpfc_z_ref_0.h5ad"
)
OUTPUT_DIR = INPUT_H5AD.parent / "low_rank_regeneration_pixel"
OUTPUT_H5AD = OUTPUT_DIR / "integrated_dlpfc_z_ref_0_lowrank_pixel_grid5.h5ad"
MARKER_GENE = "PCP4"

LOW_RANK_KWARGS = dict(
    grid_size=5.0,
    fuse_radius=10.0,
    keep_nuc_frac=0.6,
    knee_mode=False,
    alpha=0.6,
    pca_dim=32,
    outer_loops=2,
    aniso_iters=10,
    aniso_tau=0.15,
    aniso_kappa=1.0,
    gene_chunk=1024,
    epsilon=3.0,
    sinkhorn_iters=300,
    k_mask=64,
    device="cuda",
    use_raw=False,
)

DISPLAY_CMAP = "Reds"
DISPLAY_PERCENTILE = 99.0
DISPLAY_INVERT_Y = True
SPOT_SIZE_ORIGINAL = 6.0
SPOT_SIZE_LOWRANK = 8.0
FIG_DPI = 400


# =============================================================================
# GENERAL HELPERS
# =============================================================================
def _to_dense(X) -> np.ndarray:
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _coords2d(adata: AnnData) -> np.ndarray:
    if "spatial" not in adata.obsm:
        raise KeyError("Missing obsm['spatial'].")
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid spatial coordinates shape: {coords.shape}")
    return coords[:, :2]


def _resolve_gene(adata: AnnData, gene: str) -> str:
    names = adata.var_names.astype(str).to_numpy()
    if gene in names:
        return gene
    idx = np.where(np.char.upper(names) == gene.upper())[0]
    if len(idx) == 1:
        return str(names[idx[0]])
    raise KeyError(f"Gene {gene!r} not uniquely found.")


def _gene_vector(adata: AnnData, gene: str, log1p: bool = True) -> np.ndarray:
    gene = _resolve_gene(adata, gene)
    idx = int(adata.var_names.get_loc(gene))
    vec = _to_dense(adata.X[:, idx]).reshape(-1).astype(np.float32, copy=False)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    vec = np.clip(vec, 0.0, None)
    return np.log1p(vec) if log1p else vec


def _robust_vmax(values: np.ndarray, percentile: float = DISPLAY_PERCENTILE) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    vmax = float(np.percentile(values, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(values)) if values.size else 1.0
    return max(vmax, 1e-8)


def _set_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": FIG_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# PIXEL GRID + LOW-RANK CORE
# =============================================================================
def _make_grid(coords: np.ndarray, grid_size: float) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    h = float(grid_size)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x0 = np.floor((x_min - h) / h) * h
    y0 = np.floor((y_min - h) / h) * h
    x1 = np.ceil((x_max + h) / h) * h
    y1 = np.ceil((y_max + h) / h) * h
    xs = np.arange(x0, x1 + 0.5 * h, h, dtype=np.float64)
    ys = np.arange(y0, y1 + 0.5 * h, h, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)


def _build_lattice_edges(grid: np.ndarray, grid_size: float, rtol: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    if len(grid) < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    h = float(grid_size)
    tol = max(abs(h) * float(rtol), 1e-7)
    tree = cKDTree(np.asarray(grid, dtype=np.float64))
    pairs = tree.query_pairs(r=h + tol, output_type="ndarray")
    if pairs.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    dist = np.linalg.norm(grid[pairs[:, 0], :2] - grid[pairs[:, 1], :2], axis=1)
    keep = np.abs(dist - h) <= tol
    pairs = pairs[keep]
    if pairs.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)


@torch.no_grad()
def _sinkhorn_sparse(a, b, row_idx, col_idx, K_vals, m, eps_iter=300, tol=1e-6):
    dev = K_vals.device
    n = int(a.numel())
    K = torch.sparse_coo_tensor(torch.stack([row_idx, col_idx], dim=0), K_vals, size=(n, int(m)), device=dev).coalesce()
    KT = torch.sparse_coo_tensor(torch.vstack([col_idx, row_idx]), K_vals, size=(int(m), n), device=dev).coalesce()
    u = torch.ones(n, device=dev, dtype=torch.float32)
    v = torch.ones(int(m), device=dev, dtype=torch.float32)
    for _ in range(int(eps_iter)):
        Ku = torch.sparse.mm(K, v.unsqueeze(1)).squeeze(1).clamp_min(1e-30)
        u_new = a / Ku
        Kv = torch.sparse.mm(KT, u_new.unsqueeze(1)).squeeze(1).clamp_min(1e-30)
        v_new = b / Kv
        if torch.max(torch.abs(u_new - u)).item() < float(tol) and torch.max(torch.abs(v_new - v)).item() < float(tol):
            u, v = u_new, v_new
            break
        u, v = u_new, v_new
    return u, v, K, KT


@torch.no_grad()
def _svt_adaptive(X: torch.Tensor, keep_nuc_frac: float = 0.9, knee: bool = False):
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    if knee:
        log_s = torch.log(S + 1e-12)
        d = log_s[:-1] - log_s[1:]
        k = int(torch.argmax(d).item())
        tau = float(S[k + 1].item()) if k + 1 < S.numel() else 0.0
    else:
        keep = float(np.clip(keep_nuc_frac, 0.0, 1.0))
        if keep >= 0.9999:
            tau = 0.0
        elif keep <= 1e-6:
            tau = float(S.max().item())
        else:
            lo, hi = 0.0, float(S.max().item())
            total = S.sum().item()
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                kept = torch.clamp(S - mid, min=0).sum().item()
                if kept / total >= keep:
                    lo = mid
                else:
                    hi = mid
            tau = hi
    Shr = torch.clamp(S - tau, min=0.0)
    q = int((Shr > 0).sum().item())
    if q == 0:
        return torch.zeros_like(X), torch.zeros((X.shape[0], 1), device=X.device), torch.zeros((1, X.shape[1]), device=X.device)
    Uq = U[:, :q]
    Vt = Vh[:q, :]
    A = Uq * Shr[:q]
    return A @ Vt, A, Vt


@torch.no_grad()
def _latent_aniso(A: torch.Tensor, ei, ej, iters: int = 10, tau: float = 0.15, kappa: float = 1.0):
    if int(iters) <= 0 or A.numel() == 0 or len(ei) == 0:
        return A
    dev = A.device
    ei_t = torch.as_tensor(ei, device=dev, dtype=torch.long)
    ej_t = torch.as_tensor(ej, device=dev, dtype=torch.long)
    for _ in range(int(iters)):
        dA = A[ej_t] - A[ei_t]
        w = torch.exp(-(dA.pow(2).sum(dim=1) / (float(kappa) ** 2 + 1e-12))).unsqueeze(1)
        flux = w * dA
        upd = torch.zeros_like(A)
        upd.index_add_(0, ei_t, flux)
        upd.index_add_(0, ej_t, -flux)
        A = A + float(tau) * upd
    return A


def spatial_regrid_fuse_gpu_robust(
    adata: AnnData,
    grid_size: float = 50.0,
    fuse_radius: Optional[float] = None,
    k_mask: int = 64,
    alpha: float = 0.5,
    pca_dim: int = 32,
    outer_loops: int = 2,
    keep_nuc_frac: float = 0.90,
    knee_mode: bool = False,
    aniso_iters: int = 10,
    aniso_tau: float = 0.15,
    aniso_kappa: float = 1.0,
    gene_chunk: int = 1024,
    epsilon: float = 3.0,
    sinkhorn_iters: int = 300,
    device: str = "cuda",
    use_raw: bool = False,
) -> AnnData:
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.")
        device = "cpu"

    coords = _coords2d(adata)
    X_source = adata.raw.X if (use_raw and adata.raw is not None) else adata.X
    X_spot_np = _to_dense(X_source).astype(np.float32)
    var_ref = adata.raw.var.copy() if (use_raw and adata.raw is not None) else adata.var.copy()
    n, G = X_spot_np.shape

    x_spot = torch.tensor(coords, device=device, dtype=torch.float32)
    X_spot = torch.tensor(X_spot_np, device=device, dtype=torch.float32)

    grid_full = _make_grid(coords, grid_size)
    if fuse_radius is not None:
        dmin, _ = cKDTree(coords).query(grid_full, k=1)
        grid = grid_full[dmin <= float(fuse_radius)]
    else:
        grid = grid_full
    if len(grid) == 0:
        raise ValueError("No grid cells retained within fuse_radius.")

    x_grid = torch.tensor(grid, device=device, dtype=torch.float32)
    m = x_grid.shape[0]
    print(f"[INFO] Pixel support: {len(grid):,} cells")

    Xn = X_spot_np / np.clip(X_spot_np.sum(axis=1, keepdims=True), 1.0, None) * 1e4
    Xn = np.log1p(Xn)
    pca = PCA(n_components=min(int(pca_dim), min(n, Xn.shape[1])), random_state=0)
    Xp_spot_np = pca.fit_transform(Xn).astype(np.float32)
    Xp_spot_np /= np.linalg.norm(Xp_spot_np, axis=1, keepdims=True) + 1e-8
    Xp_spot = torch.tensor(Xp_spot_np, device=device, dtype=torch.float32)

    kd = cKDTree(grid)
    nbr_lists = []
    for s in coords:
        idxs = kd.query_ball_point(s, r=float(fuse_radius)) if fuse_radius is not None else list(range(int(m)))
        if len(idxs) == 0:
            _, j = kd.query(s, k=1)
            idxs = [int(j)]
        if k_mask is not None and len(idxs) > int(k_mask):
            idxs = np.asarray(idxs, dtype=np.int64)
            d = np.linalg.norm(grid[idxs] - s[None, :], axis=1)
            idxs = list(idxs[np.argsort(d)[: int(k_mask)]])
        nbr_lists.append([int(v) for v in idxs])

    col_counts = np.zeros(int(m), dtype=np.int64)
    for idxs in nbr_lists:
        col_counts[np.asarray(idxs, dtype=np.int64)] += 1
    keep_cols = col_counts > 0
    if not np.all(keep_cols):
        remap = -np.ones(int(m), dtype=np.int64)
        remap[keep_cols] = np.arange(int(np.sum(keep_cols)), dtype=np.int64)
        grid = grid[keep_cols]
        col_counts = col_counts[keep_cols]
        nbr_lists = [[int(remap[j]) for j in idxs if keep_cols[j]] for idxs in nbr_lists]
        x_grid = torch.tensor(grid, device=device, dtype=torch.float32)
        m = x_grid.shape[0]

    Kmax = max(len(lst) for lst in nbr_lists)
    nbr_idx = -np.ones((n, Kmax), dtype=np.int64)
    for i, lst in enumerate(nbr_lists):
        nbr_idx[i, : len(lst)] = lst
    nbr_idx_t = torch.tensor(nbr_idx, device=device, dtype=torch.long)
    valid_mask = nbr_idx_t >= 0

    gs = x_grid[nbr_idx_t.clamp_min(0)]
    ss = x_spot.unsqueeze(1).expand_as(gs)
    Cs = (ss - gs).pow(2).sum(dim=2)
    s_med = torch.median(Cs[valid_mask]) if valid_mask.any() else torch.tensor(1.0, device=device)
    Cs = Cs / s_med.clamp(min=1e-6)

    with torch.no_grad():
        dgs = torch.linalg.norm(gs - ss, dim=2).clamp(min=1e-12)
        w = (1.0 / dgs) * valid_mask.float()
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        Xp_grid = torch.zeros((int(m), Xp_spot.shape[1]), device=device, dtype=torch.float32)
        for k in range(Kmax):
            mask_k = valid_mask[:, k]
            target_idx = nbr_idx_t[mask_k, k]
            contrib = w[mask_k, k].unsqueeze(1) * Xp_spot[mask_k]
            Xp_grid.index_add_(0, target_idx, contrib)
        Xp_grid = Xp_grid / (Xp_grid.norm(dim=1, keepdim=True) + 1e-8)

    ei_np, ej_np = _build_lattice_edges(grid, grid_size)
    print(f"[INFO] Pixel lattice edges: {len(ei_np):,}")

    a = torch.full((n,), 1.0 / n, device=device, dtype=torch.float32)
    b = torch.tensor(col_counts / col_counts.sum(), device=device, dtype=torch.float32)

    X_grid, col_mass = None, None
    for outer in range(int(outer_loops)):
        Xg_nb = Xp_grid[nbr_idx_t.clamp_min(0)]
        dots = (Xp_spot.unsqueeze(1) * Xg_nb).sum(dim=2)
        Cg = 1.0 - dots
        g_med = torch.median(Cg[valid_mask]) if valid_mask.any() else torch.tensor(1.0, device=device)
        Cg = Cg / g_med.clamp(min=1e-6)
        C = float(alpha) * Cs + (1.0 - float(alpha)) * Cg

        Kvals = torch.zeros_like(C)
        Kvals[valid_mask] = torch.exp(-C[valid_mask] / max(float(epsilon), 1e-12))
        row_ids = torch.arange(n, device=device).unsqueeze(1).expand(n, Kmax)
        row_idx = row_ids[valid_mask]
        col_idx = nbr_idx_t[valid_mask]
        K_vals = Kvals[valid_mask]

        u, v, _, KTsp = _sinkhorn_sparse(a, b, row_idx, col_idx, K_vals, int(m), eps_iter=int(sinkhorn_iters), tol=1e-6)
        col_mass = (torch.sparse.mm(KTsp, u.unsqueeze(1)).squeeze(1) * v).clamp(min=1e-12)

        X_grid = torch.zeros((int(m), G), device=device, dtype=torch.float32)
        for start in range(0, G, int(gene_chunk)):
            end = min(G, start + int(gene_chunk))
            Z = u.unsqueeze(1) * X_spot[:, start:end]
            Y = torch.sparse.mm(KTsp, Z)
            Y = v.unsqueeze(1) * Y
            X_grid[:, start:end] = Y / col_mass.unsqueeze(1)

        X_grid, A_latent, Vt = _svt_adaptive(X_grid, keep_nuc_frac=keep_nuc_frac, knee=knee_mode)
        if int(aniso_iters) > 0 and A_latent.shape[1] > 0 and len(ei_np) > 0:
            A_latent = _latent_aniso(A_latent, ei_np, ej_np, iters=int(aniso_iters), tau=float(aniso_tau), kappa=float(aniso_kappa))
            X_grid = A_latent @ Vt

        Xg_np = X_grid.detach().cpu().numpy()
        Xg_np = np.nan_to_num(Xg_np, nan=0.0, posinf=0.0, neginf=0.0)
        Xg_np[Xg_np < 0.0] = 0.0
        row_sums = Xg_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        Xg_norm = np.log1p(Xg_np / row_sums * 1e4)
        Xg_emb = pca.transform(Xg_norm)
        Xg_emb /= np.linalg.norm(Xg_emb, axis=1, keepdims=True) + 1e-8
        Xp_grid = torch.tensor(Xg_emb, device=device, dtype=torch.float32)
        b = (col_mass / col_mass.sum()).detach()
        print(f"[INFO] Completed outer loop {outer + 1}/{outer_loops}")

    if X_grid is None:
        raise RuntimeError("Low-rank reconstruction failed.")

    out = AnnData(X=X_grid.detach().cpu().numpy(), var=var_ref.copy())
    out.obsm["spatial"] = grid.astype(np.float32)
    out.obs_names = [f"pixel_{i}" for i in range(out.n_obs)]
    out.obs["ot_target_mass"] = col_mass.detach().cpu().numpy().astype(np.float32)
    out.uns.update(
        {
            "grid_type": "regular_square_pixel",
            "grid_size": float(grid_size),
            "fuse_radius": None if fuse_radius is None else float(fuse_radius),
            "alpha": float(alpha),
            "pca_dim": int(pca_dim),
            "outer_loops": int(outer_loops),
            "keep_nuc_frac": float(keep_nuc_frac),
            "knee_mode": bool(knee_mode),
            "aniso_iters": int(aniso_iters),
            "aniso_tau": float(aniso_tau),
            "aniso_kappa": float(aniso_kappa),
            "epsilon": float(epsilon),
            "sinkhorn_iters": int(sinkhorn_iters),
            "gene_chunk": int(gene_chunk),
            "k_mask": None if k_mask is None else int(k_mask),
        }
    )
    return out


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_original_vs_lowrank_spot(
    adata_original: AnnData,
    adata_lowrank: AnnData,
    gene: str,
    outdir: Path,
    stem: str = "PCP4_original_vs_lowrank_spot",
) -> None:
    _set_plot_style()
    outdir.mkdir(parents=True, exist_ok=True)

    expr0 = _gene_vector(adata_original, gene, log1p=True)
    expr1 = _gene_vector(adata_lowrank, gene, log1p=True)
    xy0 = _coords2d(adata_original)
    xy1 = _coords2d(adata_lowrank)
    vmax0 = _robust_vmax(expr0)
    vmax1 = _robust_vmax(expr1)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    sc0 = axes[0].scatter(xy0[:, 0], xy0[:, 1], c=expr0, s=SPOT_SIZE_ORIGINAL, cmap=DISPLAY_CMAP,
                          norm=mpl.colors.Normalize(0.0, vmax0), linewidths=0, rasterized=True)
    sc1 = axes[1].scatter(xy1[:, 0], xy1[:, 1], c=expr1, s=SPOT_SIZE_LOWRANK, cmap=DISPLAY_CMAP,
                          norm=mpl.colors.Normalize(0.0, vmax1), marker="s", linewidths=0, rasterized=True)
    axes[0].set_title(f"Original spot-level\n{adata_original.n_obs:,} spots", fontweight="bold")
    axes[1].set_title(f"Low-rank spot-level\n{adata_lowrank.n_obs:,} reconstructed pixels", fontweight="bold")

    all_xy = np.vstack([xy0, xy1])
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    padx, pady = 0.02 * dx, 0.02 * dy
    for ax in axes:
        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim((ymax + pady, ymin - pady) if DISPLAY_INVERT_Y else (ymin - pady, ymax + pady))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    cb0 = fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.02, shrink=0.84)
    cb1 = fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.02, shrink=0.84)
    cb0.set_label(f"log1p({gene})")
    cb1.set_label(f"log1p({gene})")
    fig.suptitle(f"{gene}: original vs low-rank spot visualization", y=0.98, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.04, wspace=0.18)
    plt.show()

    # png = outdir / f"{stem}.png"
    # pdf = outdir / f"{stem}.pdf"
    # fig.savefig(png, bbox_inches="tight", facecolor="white")
    # fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    # plt.close(fig)
    # print(f"[SAVE] {png}")
    # print(f"[SAVE] {pdf}")


def _adata_gene_to_image(
    adata: AnnData,
    gene: str,
    *,
    grid_size: float,
    background_value: float = np.nan,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    coords = _coords2d(adata).astype(np.float64)
    expr = _gene_vector(adata, gene, log1p=True).astype(np.float64)
    h = float(grid_size)

    x = coords[:, 0]
    y = coords[:, 1]
    x0, y0 = x.min(), y.min()
    col = np.rint((x - x0) / h).astype(int)
    row = np.rint((y - y0) / h).astype(int)
    n_cols = int(col.max()) + 1
    n_rows = int(row.max()) + 1

    img = np.full((n_rows, n_cols), background_value, dtype=np.float64)
    counts = np.zeros((n_rows, n_cols), dtype=np.int32)
    sums = np.zeros((n_rows, n_cols), dtype=np.float64)
    for r, c, val in zip(row, col, expr):
        sums[r, c] += val
        counts[r, c] += 1
    mask = counts > 0
    img[mask] = sums[mask] / counts[mask]

    extent = (x0 - 0.5 * h, x0 + (n_cols - 0.5) * h, y0 - 0.5 * h, y0 + (n_rows - 0.5) * h)
    return img, extent


def plot_lowrank_two_modes(
    adata_lowrank: AnnData,
    gene: str,
    outdir: Path,
    stem: str = "PCP4_lowrank_two_modes",
) -> None:
    _set_plot_style()
    outdir.mkdir(parents=True, exist_ok=True)

    expr = _gene_vector(adata_lowrank, gene, log1p=True)
    xy = _coords2d(adata_lowrank)
    vmax_spot = _robust_vmax(expr)
    norm_spot = mpl.colors.Normalize(vmin=0.0, vmax=vmax_spot, clip=True)

    grid_size = float(adata_lowrank.uns.get("grid_size", LOW_RANK_KWARGS["grid_size"]))
    img, extent = _adata_gene_to_image(adata_lowrank, gene, grid_size=grid_size, background_value=np.nan)
    vmax_img = _robust_vmax(img[np.isfinite(img)])
    norm_img = mpl.colors.Normalize(vmin=0.0, vmax=vmax_img, clip=True)
    cmap_img = mpl.cm.get_cmap(DISPLAY_CMAP).copy()
    cmap_img.set_bad(color="white")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9))

    sc1 = axes[0].scatter(
        xy[:, 0], xy[:, 1], c=expr, s=SPOT_SIZE_LOWRANK, cmap=DISPLAY_CMAP,
        norm=norm_spot, marker="s", linewidths=0, rasterized=True,
    )
    axes[0].set_title("Low-rank spot-level", fontweight="bold")

    im = axes[1].imshow(
        img,
        cmap=cmap_img,
        norm=norm_img,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title("Low-rank pixel-image", fontweight="bold")

    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    padx, pady = 0.02 * dx, 0.02 * dy
    for ax in axes:
        ax.set_xlim(xmin - padx, xmax + padx)
        if DISPLAY_INVERT_Y:
            ax.set_ylim(ymax + pady, ymin - pady)
        else:
            ax.set_ylim(ymin - pady, ymax + pady)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    cb0 = fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.02, shrink=0.84)
    cb1 = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02, shrink=0.84)
    cb0.set_label(f"log1p({gene})")
    cb1.set_label(f"log1p({gene})")

    fig.suptitle(f"{gene}: two visualization modes of low-rank output", y=0.98, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.04, wspace=0.18)

    plt.show()
# =============================================================================
# RUN
# =============================================================================
def run_reconstruction(input_h5ad: Path = INPUT_H5AD, output_h5ad: Path = OUTPUT_H5AD) -> Tuple[AnnData, AnnData]:
    if not Path(input_h5ad).is_file():
        raise FileNotFoundError(f"Input H5AD not found: {input_h5ad}")
    print(f"[LOAD] {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)
    if "valid" in adata.obs.columns:
        adata = adata[adata.obs["valid"].astype(bool)].copy()
    print(f"[INFO] Input AnnData: n_obs={adata.n_obs:,}, n_vars={adata.n_vars:,}")
    _ = _resolve_gene(adata, MARKER_GENE)

    adata_lowrank = spatial_regrid_fuse_gpu_robust(adata, **LOW_RANK_KWARGS)

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    for key in list(adata_lowrank.uns.keys()):
        if adata_lowrank.uns[key] is None:
            del adata_lowrank.uns[key]
    adata_lowrank.write_h5ad(output_h5ad)
    print(f"[SAVE] {output_h5ad}")
    return adata, adata_lowrank


def main() -> None:
    adata_original, adata_lowrank = run_reconstruction()
    plot_original_vs_lowrank_spot(adata_original, adata_lowrank, gene=MARKER_GENE, outdir=OUTPUT_DIR)
    plot_lowrank_two_modes(adata_lowrank, gene=MARKER_GENE, outdir=OUTPUT_DIR)
    print("[DONE] Reconstruction and visualization completed.")


if __name__ == "__main__":
    main()
