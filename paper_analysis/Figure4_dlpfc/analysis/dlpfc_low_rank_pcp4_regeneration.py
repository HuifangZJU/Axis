"""
Drop-in Visium-grid replacement for spatial_regrid_fuse_gpu_robust.

The public function signature and return type are unchanged. Relative to the
original implementation, only two geometric components are changed:

1. A staggered hexagonal/Visium-like grid replaces the rectangular pixel grid.
2. Latent diffusion uses the exact six-neighbor hexagonal lattice.
3. ``obs['layer_guess_reordered']`` is transported through the final OT plan
   when that categorical attribute is available in the input AnnData object.

All OT, PCA, Sinkhorn, expression reconstruction, adaptive SVT, diffusion
parameters, and output expression scale follow the original implementation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from anndata import AnnData
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from dense_generation_pixel import  spatial_regrid_fuse_gpu_robust
# -------------------- small helpers --------------------
def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)
def _make_visium_like_grid(
    coords: np.ndarray,
    grid_size: float,
):
    """
    Construct a staggered hexagonal grid covering the coordinate bounding box.

    Parameters
    ----------
    coords
        Source coordinates with shape (n_obs, 2).
    grid_size
        Nearest-neighbor center-to-center distance on the generated grid.

    Returns
    -------
    grid
        Spatial coordinates with shape (n_grid, 2).
    array_row
        Integer row indices.
    array_col
        Integer staggered column indices following the Visium convention:
        same-row neighbors differ by 2 and adjacent-row neighbors differ by 1.
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"coords must have shape (n_obs, >=2); received {coords.shape}."
        )
    coords = coords[:, :2]

    if coords.shape[0] == 0:
        raise ValueError("Cannot construct a grid from zero coordinates.")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords contains non-finite values.")
    if not np.isfinite(grid_size) or grid_size <= 0:
        raise ValueError("grid_size must be finite and positive.")

    grid_size = float(grid_size)
    row_spacing = np.sqrt(3.0) * grid_size / 2.0

    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)

    # Add one lattice ring around the observed bounding box. Radius pruning in
    # the main function subsequently removes positions away from the tissue.
    x_origin = np.floor((x_min - grid_size) / grid_size) * grid_size
    y_origin = np.floor((y_min - row_spacing) / row_spacing) * row_spacing

    n_rows = int(
        np.ceil((y_max + row_spacing - y_origin) / row_spacing)
    ) + 1
    n_cols = int(
        np.ceil((x_max + grid_size - x_origin) / grid_size)
    ) + 1

    grid_blocks = []
    row_blocks = []
    col_blocks = []

    for row_index in range(n_rows):
        parity = row_index % 2
        x_shift = 0.5 * grid_size * parity
        y_value = y_origin + row_index * row_spacing

        regular_col = np.arange(n_cols, dtype=np.int64)
        x_values = x_origin + regular_col.astype(np.float64) * grid_size + x_shift
        y_values = np.full(n_cols, y_value, dtype=np.float64)

        grid_blocks.append(np.column_stack([x_values, y_values]))
        row_blocks.append(
            np.full(n_cols, row_index, dtype=np.int64)
        )

        # Visium-like integer layout:
        # same-row neighbors: array_col difference = 2
        # adjacent-row neighbors: array_col difference = 1
        col_blocks.append(2 * regular_col + parity)

    grid = np.vstack(grid_blocks).astype(np.float32)
    array_row = np.concatenate(row_blocks)
    array_col = np.concatenate(col_blocks)

    return grid, array_row, array_col


def _build_visium_lattice_edges(
    array_row: np.ndarray,
    array_col: np.ndarray,
):
    """
    Build the exact first-ring six-neighbor graph of a pruned Visium grid.

    Only three forward offsets are queried to avoid duplicate undirected edges:
        (row, col + 2)
        (row + 1, col - 1)
        (row + 1, col + 1)
    """
    array_row = np.asarray(array_row, dtype=np.int64)
    array_col = np.asarray(array_col, dtype=np.int64)

    if array_row.shape != array_col.shape:
        raise ValueError("array_row and array_col must have identical shapes.")

    lookup = {
        (int(row), int(col)): index
        for index, (row, col) in enumerate(
            zip(array_row.tolist(), array_col.tolist())
        )
    }

    edge_i = []
    edge_j = []

    for index, (row, col) in enumerate(
        zip(array_row.tolist(), array_col.tolist())
    ):
        forward_neighbors = (
            (row, col + 2),
            (row + 1, col - 1),
            (row + 1, col + 1),
        )
        for key in forward_neighbors:
            neighbor = lookup.get(key)
            if neighbor is not None:
                edge_i.append(index)
                edge_j.append(neighbor)

    return (
        np.asarray(edge_i, dtype=np.int64),
        np.asarray(edge_j, dtype=np.int64),
    )



def _transport_categorical_obs_by_ot(
    source_values,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    edge_mass: torch.Tensor,
    n_targets: int,
):
    """
    Transport one categorical source annotation through a sparse OT plan.

    For each target grid point and category, this function sums the final OT
    mass arriving from source spots with that category. The hard target label is
    the category with the largest transported probability.

    Returns
    -------
    labels
        Pandas Categorical with one value per target grid point.
    probabilities
        Array of shape (n_targets, n_categories).
    categories
        Category names in probability-column order.
    confidence
        Maximum category probability per target.
    entropy
        Normalized categorical entropy in [0, 1], where 0 is certain.
    """
    source_series = pd.Series(source_values).reset_index(drop=True)

    if len(source_series) == 0:
        raise ValueError("Cannot transport an empty categorical annotation.")

    if pd.api.types.is_categorical_dtype(source_series.dtype):
        categories = [
            str(value)
            for value in source_series.cat.categories
        ]
        source_as_string = source_series.astype("string")
    else:
        source_as_string = source_series.astype("string")
        categories = [
            str(value)
            for value in pd.unique(source_as_string.dropna())
        ]

    if len(categories) == 0:
        labels = pd.Categorical(
            [pd.NA] * int(n_targets),
            categories=[],
        )
        probabilities = np.zeros(
            (int(n_targets), 0),
            dtype=np.float32,
        )
        confidence = np.full(
            int(n_targets),
            np.nan,
            dtype=np.float32,
        )
        entropy = np.full(
            int(n_targets),
            np.nan,
            dtype=np.float32,
        )
        return labels, probabilities, categories, confidence, entropy

    category_to_code = {
        category: index
        for index, category in enumerate(categories)
    }
    source_codes = np.asarray(
        [
            category_to_code.get(str(value), -1)
            if not pd.isna(value)
            else -1
            for value in source_as_string
        ],
        dtype=np.int64,
    )

    row_np = row_idx.detach().cpu().numpy().astype(np.int64)
    col_np = col_idx.detach().cpu().numpy().astype(np.int64)
    mass_np = edge_mass.detach().cpu().numpy().astype(np.float64)

    if row_np.size == 0:
        raise ValueError("The final OT plan contains no edges.")
    if row_np.max() >= len(source_codes):
        raise ValueError(
            "OT source indices exceed the categorical annotation length."
        )

    n_categories = len(categories)
    category_mass = np.zeros(
        (int(n_targets), n_categories),
        dtype=np.float64,
    )

    valid = (
        (source_codes[row_np] >= 0)
        & np.isfinite(mass_np)
        & (mass_np > 0)
    )
    np.add.at(
        category_mass,
        (
            col_np[valid],
            source_codes[row_np[valid]],
        ),
        mass_np[valid],
    )

    labeled_mass = category_mass.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        category_mass,
        labeled_mass,
        out=np.zeros_like(category_mass),
        where=labeled_mass > 0,
    )

    best_code = np.argmax(probabilities, axis=1)
    best_labels = np.asarray(
        categories,
        dtype=object,
    )[best_code]
    best_labels[labeled_mass[:, 0] <= 0] = pd.NA

    labels = pd.Categorical(
        best_labels,
        categories=categories,
        ordered=False,
    )

    confidence = np.max(probabilities, axis=1)
    confidence[labeled_mass[:, 0] <= 0] = np.nan

    if n_categories > 1:
        safe_probability = np.clip(
            probabilities,
            1e-12,
            1.0,
        )
        entropy = -np.sum(
            probabilities * np.log(safe_probability),
            axis=1,
        ) / np.log(float(n_categories))
        entropy[labeled_mass[:, 0] <= 0] = np.nan
    else:
        entropy = np.zeros(int(n_targets), dtype=np.float64)
        entropy[labeled_mass[:, 0] <= 0] = np.nan

    return (
        labels,
        probabilities.astype(np.float32),
        categories,
        confidence.astype(np.float32),
        entropy.astype(np.float32),
    )

# -------------------- sparse Sinkhorn (neighbor-limited) --------------------
@torch.no_grad()
def _sinkhorn_sparse(a, b, row_idx, col_idx, K_vals, m, eps_iter=300, tol=1e-6):
    """
    K is sparse with nnz entries: values K_vals at (row_idx, col_idx).
    Computes u,v s.t. T = diag(u) K diag(v) has marginals a,b.
    """
    dev = K_vals.device
    n = int(a.numel())
    nnz = row_idx.numel()
    # Build sparse COO for K and K^T (coalesced)
    indices = torch.stack([row_idx, col_idx], dim=0)  # (2, nnz)
    K = torch.sparse_coo_tensor(indices, K_vals, size=(n, m), device=dev).coalesce()
    KT = torch.sparse_coo_tensor(torch.vstack([col_idx, row_idx]), K_vals, size=(m, n), device=dev).coalesce()

    u = torch.ones(n, device=dev, dtype=torch.float32)
    v = torch.ones(m, device=dev, dtype=torch.float32)

    one_m = torch.ones(m, device=dev, dtype=torch.float32).unsqueeze(1)  # (m,1)
    one_n = torch.ones(n, device=dev, dtype=torch.float32).unsqueeze(1)  # (n,1)

    for _ in range(eps_iter):
        Ku = torch.sparse.mm(K, v.unsqueeze(1)).squeeze(1).clamp_min(1e-30)   # (n,)
        u_new = a / Ku
        Kv = torch.sparse.mm(KT, u_new.unsqueeze(1)).squeeze(1).clamp_min(1e-30)  # (m,)
        v_new = b / Kv
        if torch.max(torch.abs(u_new - u)).item() < tol and torch.max(torch.abs(v_new - v)).item() < tol:
            u, v = u_new, v_new
            break
        u, v = u_new, v_new
    return u, v, K, KT



@torch.no_grad()
def _svt_adaptive(X: torch.Tensor, keep_nuc_frac: float = 0.9, knee: bool = False):
    U,S,Vh = torch.linalg.svd(X, full_matrices=False)
    if knee:
        log_s = torch.log(S + 1e-12); d = log_s[:-1]-log_s[1:]; k = int(torch.argmax(d).item())
        tau = float(S[k+1].item()) if k+1 < S.numel() else 0.0
    else:
        keep = float(np.clip(keep_nuc_frac,0.0,1.0))
        if keep >= 0.9999: tau = 0.0
        elif keep <= 1e-6: tau = float(S.max().item())
        else:
            lo,hi = 0.0,float(S.max().item()); total=S.sum().item()
            for _ in range(40):
                mid=0.5*(lo+hi); kept=torch.clamp(S-mid,min=0).sum().item()
                if kept/total >= keep: lo=mid
                else: hi=mid
            tau=hi
    Shr = torch.clamp(S - tau, min=0.0)
    q = int((Shr>0).sum().item())
    if q==0:
        return torch.zeros_like(X), torch.zeros((X.shape[0],1),device=X.device), torch.zeros((1,X.shape[1]),device=X.device)
    Uq=U[:,:q]; Vt=Vh[:q,:]; A = Uq*Shr[:q]
    return A@Vt, A, Vt


@torch.no_grad()
def _latent_aniso(A, ei, ej, iters=10, tau=0.15, kappa=1.0):
    if iters<=0 or A.numel()==0: return A
    dev=A.device
    ei_t=torch.tensor(ei,device=dev); ej_t=torch.tensor(ej,device=dev)
    for _ in range(iters):
        dA = A[ej_t]-A[ei_t]
        w = torch.exp(-(dA.pow(2).sum(dim=1)/(kappa**2+1e-12))).unsqueeze(1)
        flux = w*dA
        upd = torch.zeros_like(A)
        upd.index_add_(0, ei_t, flux)
        upd.index_add_(0, ej_t, -flux)
        A = A + tau*upd
    return A


# -------------------- main (batched + sparse) --------------------
def spatial_regrid_fuse_gpu_robust(
    adata: AnnData,
    grid_size: float = 50.0,
    fuse_radius: float = None,
    k_mask: int = 64,
    alpha: float = 0.5,
    pca_dim: int = 32,
    outer_loops: int = 2,
    # adaptive SVT
    keep_nuc_frac: float = 0.90,
    knee_mode: bool = False,
    # latent diffusion
    aniso_iters: int = 10,
    aniso_tau: float = 0.15,
    aniso_kappa: float = 1.0,
    # batched gene matmul
    gene_chunk: int = 1024,
    epsilon: float = 3.0,
    sinkhorn_iters: int = 300,
    device: str = "cuda",
    use_raw: bool = False,
):
    """
    Memory-efficient GPU OT fusion on a generated Visium-like hexagonal grid.

    This is a drop-in replacement for the original rectangular-grid function.
    The function signature, inputs, output AnnData object, and fusion parameters
    are unchanged.

    The meaning of ``grid_size`` is now the nearest-neighbor center-to-center
    distance between generated Visium-like spots.

    When ``adata.obs['layer_guess_reordered']`` exists, the annotation is
    transported through the final sparse OT plan and saved to the same key in
    the output. Per-category probabilities, confidence, and entropy are also
    retained.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if fuse_radius is not None:
        if not np.isfinite(fuse_radius) or fuse_radius <= 0:
            raise ValueError("fuse_radius must be positive or None.")

    if k_mask is not None and int(k_mask) < 1:
        raise ValueError("k_mask must be positive or None.")
    if int(gene_chunk) < 1:
        raise ValueError("gene_chunk must be positive.")
    if int(outer_loops) < 1:
        raise ValueError("outer_loops must be at least 1.")

    # ------------------------------------------------------------------
    # Data: unchanged from the original implementation.
    # ------------------------------------------------------------------
    coords = np.asarray(
        adata.obsm["spatial"],
        dtype=np.float32,
    )
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            "adata.obsm['spatial'] must have shape (n_obs, >=2)."
        )
    coords = coords[:, :2]

    X_source = (
        adata.raw.X
        if use_raw and adata.raw is not None
        else adata.X
    )
    X_spot_np = _to_dense(X_source).astype(np.float32)
    var_ref = (
        adata.raw.var.copy()
        if use_raw and adata.raw is not None
        else adata.var.copy()
    )

    if coords.shape[0] != X_spot_np.shape[0]:
        raise ValueError(
            "Coordinate and expression row counts do not match."
        )
    if not np.all(np.isfinite(coords)):
        raise ValueError("Spatial coordinates contain non-finite values.")
    if not np.all(np.isfinite(X_spot_np)):
        raise ValueError("Expression matrix contains non-finite values.")

    n, G = X_spot_np.shape
    if n == 0 or G == 0:
        raise ValueError("The input AnnData object has no observations or genes.")

    x_spot = torch.tensor(
        coords,
        device=device,
        dtype=torch.float32,
    )
    X_spot = torch.tensor(
        X_spot_np,
        device=device,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Visium-like grid generation.
    # ------------------------------------------------------------------
    grid_full, array_row_full, array_col_full = _make_visium_like_grid(
        coords=coords,
        grid_size=grid_size,
    )

    if fuse_radius is not None:
        dmin, _ = cKDTree(coords).query(grid_full, k=1)
        keep_tissue = dmin <= float(fuse_radius)
        grid = grid_full[keep_tissue]
        array_row = array_row_full[keep_tissue]
        array_col = array_col_full[keep_tissue]
    else:
        grid = grid_full
        array_row = array_row_full
        array_col = array_col_full

    if grid.shape[0] == 0:
        raise ValueError("No Visium-like grid cells within fuse_radius.")

    x_grid = torch.tensor(
        grid,
        device=device,
        dtype=torch.float32,
    )
    m = x_grid.shape[0]

    # ------------------------------------------------------------------
    # Gene embedding for the OT cost: unchanged.
    # ------------------------------------------------------------------
    Xn = (
        X_spot_np
        / np.clip(
            X_spot_np.sum(axis=1, keepdims=True),
            1.0,
            None,
        )
        * 1e4
    )
    Xn = np.log1p(Xn)

    n_components = min(
        int(pca_dim),
        min(n, Xn.shape[1]),
    )
    if n_components < 1:
        raise ValueError("PCA requires at least one component.")

    pca = PCA(
        n_components=n_components,
        random_state=0,
    )
    Xp_spot_np = pca.fit_transform(Xn).astype(np.float32)
    Xp_spot_np /= (
        np.linalg.norm(
            Xp_spot_np,
            axis=1,
            keepdims=True,
        )
        + 1e-8
    )
    Xp_spot = torch.tensor(
        Xp_spot_np,
        device=device,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Source-to-grid sparse neighbor support: unchanged in meaning.
    # ------------------------------------------------------------------
    kd = cKDTree(grid)
    nbr_lists = []

    for source_coord in coords:
        if fuse_radius is not None:
            idxs = kd.query_ball_point(
                source_coord,
                r=float(fuse_radius),
            )
        else:
            idxs = list(range(m))

        if len(idxs) == 0:
            _, nearest = kd.query(source_coord, k=1)
            idxs = [int(nearest)]

        if k_mask is not None and len(idxs) > int(k_mask):
            idxs_array = np.asarray(idxs, dtype=np.int64)
            sub = grid[idxs_array]
            distances = np.linalg.norm(
                sub - source_coord[None, :],
                axis=1,
            )
            order = np.argsort(distances)[: int(k_mask)]
            idxs = list(idxs_array[order])

        nbr_lists.append([int(x) for x in idxs])

    # Drop target grid positions with no incoming source edge, as in the
    # original function.
    col_counts = np.zeros(m, dtype=np.int64)
    for idxs in nbr_lists:
        col_counts[np.asarray(idxs, dtype=np.int64)] += 1

    keep_cols = col_counts > 0
    if not np.all(keep_cols):
        remap = -np.ones(m, dtype=np.int64)
        remap[keep_cols] = np.arange(
            int(np.sum(keep_cols)),
            dtype=np.int64,
        )

        grid = grid[keep_cols]
        array_row = array_row[keep_cols]
        array_col = array_col[keep_cols]
        col_counts = col_counts[keep_cols]

        nbr_lists = [
            [
                int(remap[j])
                for j in idxs
                if keep_cols[j]
            ]
            for idxs in nbr_lists
        ]

        x_grid = torch.tensor(
            grid,
            device=device,
            dtype=torch.float32,
        )
        m = x_grid.shape[0]

    if m == 0:
        raise ValueError(
            "No Visium-like grid cells retained after neighbor filtering."
        )
    if any(len(idxs) == 0 for idxs in nbr_lists):
        raise RuntimeError(
            "At least one source spot lost all grid neighbors after remapping."
        )

    # Pad sparse neighbor lists.
    Kmax = max(len(lst) for lst in nbr_lists)
    nbr_idx = -np.ones(
        (n, Kmax),
        dtype=np.int64,
    )
    for i, lst in enumerate(nbr_lists):
        nbr_idx[i, : len(lst)] = lst

    nbr_idx_t = torch.tensor(
        nbr_idx,
        device=device,
        dtype=torch.long,
    )
    valid_mask = nbr_idx_t >= 0

    # ------------------------------------------------------------------
    # Spatial cost on sparse edges: unchanged.
    # ------------------------------------------------------------------
    gs = x_grid[nbr_idx_t.clamp_min(0)]
    ss = x_spot.unsqueeze(1).expand_as(gs)
    Cs = (ss - gs).pow(2).sum(dim=2)

    s_med = (
        torch.median(Cs[valid_mask])
        if valid_mask.any()
        else torch.tensor(1.0, device=device)
    )
    Cs = Cs / s_med.clamp(min=1e-6)

    # ------------------------------------------------------------------
    # Initial target gene embedding: unchanged.
    # ------------------------------------------------------------------
    with torch.no_grad():
        dgs = torch.linalg.norm(
            gs - ss,
            dim=2,
        ).clamp(min=1e-12)

        w = (1.0 / dgs) * valid_mask.float()
        w = w / (
            w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        )

        Xp_grid = torch.zeros(
            (m, Xp_spot.shape[1]),
            device=device,
            dtype=torch.float32,
        )

        for k in range(Kmax):
            mask_k = valid_mask[:, k]
            target_idx = nbr_idx_t[mask_k, k]
            contribution = (
                w[mask_k, k].unsqueeze(1)
                * Xp_spot[mask_k]
            )
            Xp_grid.index_add_(
                0,
                target_idx,
                contribution,
            )

        Xp_grid = Xp_grid / (
            Xp_grid.norm(dim=1, keepdim=True) + 1e-8
        )

    # Exact six-neighbor Visium lattice for latent diffusion.
    ei_np, ej_np = _build_visium_lattice_edges(
        array_row=array_row,
        array_col=array_col,
    )

    # ------------------------------------------------------------------
    # OT fusion: unchanged from the original implementation.
    # ------------------------------------------------------------------
    a = torch.full(
        (n,),
        1.0 / n,
        device=device,
        dtype=torch.float32,
    )

    b0 = torch.tensor(
        col_counts / col_counts.sum(),
        device=device,
        dtype=torch.float32,
    )
    b = b0.clone()

    X_grid = None
    col_mass = None

    final_row_idx = None
    final_col_idx = None
    final_kernel_values = None
    final_u = None
    final_v = None

    for _ in range(int(outer_loops)):
        Xg_nb = Xp_grid[
            nbr_idx_t.clamp_min(0)
        ]
        dots = (
            Xp_spot.unsqueeze(1) * Xg_nb
        ).sum(dim=2)

        Cg = 1.0 - dots
        g_med = (
            torch.median(Cg[valid_mask])
            if valid_mask.any()
            else torch.tensor(1.0, device=device)
        )
        Cg = Cg / g_med.clamp(min=1e-6)

        C = (
            float(alpha) * Cs
            + (1.0 - float(alpha)) * Cg
        )

        Kvals = torch.zeros_like(C)
        Kvals[valid_mask] = torch.exp(
            -C[valid_mask]
            / max(float(epsilon), 1e-12)
        )

        row_ids = torch.arange(
            n,
            device=device,
        ).unsqueeze(1).expand(n, Kmax)

        row_idx = row_ids[valid_mask]
        col_idx = nbr_idx_t[valid_mask]
        K_vals = Kvals[valid_mask]

        u, v, Ksp, KTsp = _sinkhorn_sparse(
            a,
            b,
            row_idx,
            col_idx,
            K_vals,
            m,
            eps_iter=int(sinkhorn_iters),
            tol=1e-6,
        )

        col_mass = (
            torch.sparse.mm(
                KTsp,
                u.unsqueeze(1),
            ).squeeze(1)
            * v
        )
        col_mass = col_mass.clamp(min=1e-12)

        X_grid = torch.zeros(
            (m, G),
            device=device,
            dtype=torch.float32,
        )

        for start in range(0, G, int(gene_chunk)):
            end = min(G, start + int(gene_chunk))

            Z = (
                u.unsqueeze(1)
                * X_spot[:, start:end]
            )
            Y = torch.sparse.mm(KTsp, Z)
            Y = v.unsqueeze(1) * Y

            X_grid[:, start:end] = (
                Y / col_mass.unsqueeze(1)
            )

        X_grid, A_latent, Vt = _svt_adaptive(
            X_grid,
            keep_nuc_frac=keep_nuc_frac,
            knee=knee_mode,
        )

        if (
            int(aniso_iters) > 0
            and A_latent.shape[1] > 0
            and len(ei_np) > 0
        ):
            A_latent = _latent_aniso(
                A_latent,
                ei_np,
                ej_np,
                iters=int(aniso_iters),
                tau=float(aniso_tau),
                kappa=float(aniso_kappa),
            )
            X_grid = A_latent @ Vt

        Xg_np = X_grid.detach().cpu().numpy()
        Xg_np = np.nan_to_num(
            Xg_np,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        Xg_np[Xg_np < 0.0] = 0.0

        row_sums = Xg_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0

        Xg_norm = np.log1p(
            Xg_np / row_sums * 1e4
        )
        Xg_norm = np.nan_to_num(
            Xg_norm,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        Xg_emb = pca.transform(Xg_norm)
        Xg_emb /= (
            np.linalg.norm(
                Xg_emb,
                axis=1,
                keepdims=True,
            )
            + 1e-8
        )

        Xp_grid = torch.tensor(
            Xg_emb,
            device=device,
            dtype=torch.float32,
        )

        b = (
            col_mass / col_mass.sum()
        ).detach()

        final_row_idx = row_idx
        final_col_idx = col_idx
        final_kernel_values = K_vals
        final_u = u
        final_v = v

    if X_grid is None or col_mass is None:
        raise RuntimeError("OT fusion did not produce an output matrix.")

    # ------------------------------------------------------------------
    # Output: same AnnData return type, now with Visium lattice indices.
    # ------------------------------------------------------------------
    X_output = X_grid.detach().cpu().numpy()

    new_adata = AnnData(
        X=X_output,
        var=var_ref.copy(),
    )
    new_adata.obsm["spatial"] = grid.astype(np.float32)
    new_adata.obs_names = [
        f"grid_{i}"
        for i in range(new_adata.n_obs)
    ]

    new_adata.obs["array_row"] = array_row.astype(np.int64)
    new_adata.obs["array_col"] = array_col.astype(np.int64)
    new_adata.obs["in_tissue"] = 1
    new_adata.obs["ot_target_mass"] = (
        col_mass.detach().cpu().numpy().astype(np.float32)
    )

    # ------------------------------------------------------------------
    # Transport the cortical-layer annotation through the final OT plan.
    # ------------------------------------------------------------------
    layer_key = "layer_guess_reordered"
    layer_transferred = False

    if layer_key in adata.obs:
        if any(
            value is None
            for value in (
                final_row_idx,
                final_col_idx,
                final_kernel_values,
                final_u,
                final_v,
            )
        ):
            raise RuntimeError(
                "The final OT plan is unavailable for attribute transfer."
            )

        # Sparse transport-plan mass:
        # T_ij = u_i * K_ij * v_j
        final_edge_mass = (
            final_u[final_row_idx]
            * final_kernel_values
            * final_v[final_col_idx]
        )

        (
            layer_labels,
            layer_probabilities,
            layer_categories,
            layer_confidence,
            layer_entropy,
        ) = _transport_categorical_obs_by_ot(
            source_values=adata.obs[layer_key],
            row_idx=final_row_idx,
            col_idx=final_col_idx,
            edge_mass=final_edge_mass,
            n_targets=m,
        )

        new_adata.obs[layer_key] = layer_labels
        new_adata.obs[f"{layer_key}_confidence"] = layer_confidence
        new_adata.obs[f"{layer_key}_entropy"] = layer_entropy
        new_adata.obsm[
            f"{layer_key}_probabilities"
        ] = layer_probabilities
        new_adata.uns[
            f"{layer_key}_categories"
        ] = layer_categories
        new_adata.uns[
            f"{layer_key}_transfer_method"
        ] = "maximum categorical probability from final sparse OT mass"
        layer_transferred = True

    # Read-scale diagnostics do not alter the generated values.
    source_totals = np.sum(X_spot_np, axis=1)
    output_totals = np.sum(X_output, axis=1)

    source_positive = source_totals[
        np.isfinite(source_totals)
    ]
    output_positive = output_totals[
        np.isfinite(output_totals)
    ]

    source_median = (
        float(np.median(source_positive))
        if source_positive.size
        else np.nan
    )
    output_median = (
        float(np.median(output_positive))
        if output_positive.size
        else np.nan
    )

    new_adata.uns.update(
        dict(
            grid_type="visium_like_hexagonal",
            grid_size=float(grid_size),
            grid_size_definition=(
                "nearest-neighbor center-to-center distance"
            ),
            visium_row_spacing=float(
                np.sqrt(3.0) * float(grid_size) / 2.0
            ),
            fuse_radius=(
                None
                if fuse_radius is None
                else float(fuse_radius)
            ),
            alpha=float(alpha),
            pca_dim=int(pca_dim),
            outer_loops=int(outer_loops),
            keep_nuc_frac=float(keep_nuc_frac),
            knee_mode=bool(knee_mode),
            aniso_iters=int(aniso_iters),
            aniso_tau=float(aniso_tau),
            aniso_kappa=float(aniso_kappa),
            epsilon=float(epsilon),
            sinkhorn_iters=int(sinkhorn_iters),
            gene_chunk=int(gene_chunk),
            sparse_mode=True,
            k_mask=(
                None
                if k_mask is None
                else int(k_mask)
            ),
            n_grid_full=int(grid_full.shape[0]),
            n_grid_after_radius=int(
                np.sum(keep_tissue)
                if fuse_radius is not None
                else grid_full.shape[0]
            ),
            n_grid_final=int(grid.shape[0]),
            n_visium_lattice_edges=int(len(ei_np)),
            source_median_total_expression=source_median,
            output_median_total_expression=output_median,
            output_to_source_median_total_ratio=(
                float(output_median / source_median)
                if np.isfinite(source_median)
                and source_median > 0
                and np.isfinite(output_median)
                else np.nan
            ),
            use_raw=bool(use_raw),
            layer_guess_reordered_transferred=bool(layer_transferred),
            layer_guess_reordered_source_key=(
                "layer_guess_reordered"
                if layer_transferred
                else None
            ),
        )
    )

    return new_adata

# =============================================================================
# Standalone DLPFC low-rank regeneration + PCP4 visualization
# =============================================================================

INPUT_H5AD = Path(
    "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/"
    "dlpfc_z_group1_grid200_spatial0.3/integration/integrated_dlpfc_z_ref_0.h5ad"
)

OUTPUT_DIR = INPUT_H5AD.parent / "low_rank_regeneration"
OUTPUT_H5AD = OUTPUT_DIR / "integrated_dlpfc_z_ref_0_lowrank_grid1p5.h5ad"

MARKER_GENE = "PCP4"
FIGURE_STEM = "PCP4_before_after_low_rank"

# Exact settings requested for the regenerated low-rank representation.
LOW_RANK_KWARGS = dict(
    grid_size=5,
    fuse_radius=10,
    keep_nuc_frac=0.6,
    knee_mode=False,
    alpha=0.6,
    pca_dim=32,
    device="cuda",
    use_raw=False,
)

# Display settings only; these do not change the analysis.
DISPLAY_CMAP = "Reds"
DISPLAY_PERCENTILE = 99.0
DISPLAY_POINT_SIZE_BEFORE = 5.0
DISPLAY_POINT_SIZE_AFTER = 15.0
DISPLAY_INVERT_Y = True
DISPLAY_SWAP_XY = False
FIGSIZE_COMPARISON = (10.2, 4.8)
FIG_DPI = 400


def _resolve_gene_name(adata: AnnData, requested_gene: str) -> str:
    """Resolve a marker gene case-insensitively without altering var_names."""
    var_names = np.asarray(adata.var_names.astype(str), dtype=str)
    if requested_gene in var_names:
        return requested_gene

    matches = np.where(np.char.upper(var_names) == str(requested_gene).upper())[0]
    if len(matches) == 1:
        return str(var_names[matches[0]])
    if len(matches) == 0:
        raise KeyError(
            f"Gene {requested_gene!r} was not found. "
            f"Input contains {adata.n_vars} genes."
        )
    raise KeyError(
        f"Gene name {requested_gene!r} is ambiguous after case-insensitive matching: "
        f"{var_names[matches].tolist()}"
    )


def _extract_gene_vector(adata: AnnData, gene: str) -> np.ndarray:
    """Extract one gene from adata.X as a finite dense 1D array."""
    resolved = _resolve_gene_name(adata, gene)
    idx = int(adata.var_names.get_loc(resolved))
    col = adata.X[:, idx]
    values = _to_dense(col).reshape(-1).astype(np.float32, copy=False)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values


def _display_coords(adata: AnnData) -> np.ndarray:
    if "spatial" not in adata.obsm:
        raise KeyError(
            "AnnData is missing obsm['spatial']; it is required for PCP4 plotting."
        )
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"obsm['spatial'] must have shape (n_obs, >=2), got {coords.shape}."
        )
    coords = coords[:, :2].copy()
    if DISPLAY_SWAP_XY:
        coords = coords[:, [1, 0]]
    return coords


def _robust_shared_vmax(*arrays: np.ndarray, percentile: float = 99.0) -> float:
    values = np.concatenate([np.asarray(x, dtype=float).reshape(-1) for x in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    vmax = float(np.percentile(values, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(values)) if values.size else 1.0
    return max(vmax, 1e-8)


def set_marker_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": FIG_DPI,
        }
    )


def plot_marker_before_after_low_rank(
    adata_before: AnnData,
    adata_after: AnnData,
    *,
    gene: str = MARKER_GENE,
    outdir: Path = OUTPUT_DIR,
    stem: str = FIGURE_STEM,
) -> None:
    """
    Plot the same marker before and after low-rank spatial reconstruction.

    Both panels use log1p expression and a common robust color scale so the
    visual comparison reflects changes introduced by regridding + OT + SVT +
    latent diffusion rather than independent panel normalization.
    """
    set_marker_figure_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    before_gene = _resolve_gene_name(adata_before, gene)
    after_gene = _resolve_gene_name(adata_after, gene)

    before_raw = _extract_gene_vector(adata_before, before_gene)
    after_raw = _extract_gene_vector(adata_after, after_gene)

    # Visualization only: preserve the analysis matrices and plot on log1p scale.
    before = np.log1p(np.clip(before_raw, 0.0, None))
    after = np.log1p(np.clip(after_raw, 0.0, None))

    before_xy = _display_coords(adata_before)
    after_xy = _display_coords(adata_after)

    vmax = _robust_shared_vmax(before, after, percentile=DISPLAY_PERCENTILE)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax, clip=True)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_COMPARISON)

    scatter_before = axes[0].scatter(
        before_xy[:, 0],
        before_xy[:, 1],
        c=before,
        s=DISPLAY_POINT_SIZE_BEFORE,
        cmap=DISPLAY_CMAP,
        norm=norm,
        linewidths=0,
        rasterized=True,
    )
    axes[0].set_title(
        f"Before low-rank analysis\n{adata_before.n_obs:,} spatial observations",
        fontweight="bold",
    )

    axes[1].scatter(
        after_xy[:, 0],
        after_xy[:, 1],
        c=after,
        s=DISPLAY_POINT_SIZE_AFTER,
        cmap=DISPLAY_CMAP,
        norm=norm,
        linewidths=0,
        rasterized=True,
    )
    axes[1].set_title(
        f"After low-rank analysis\n{adata_after.n_obs:,} Visium-like grid positions",
        fontweight="bold",
    )

    # Keep the same displayed spatial extent whenever possible.
    all_xy = np.vstack([before_xy, after_xy])
    finite = np.all(np.isfinite(all_xy), axis=1)
    if np.any(finite):
        xy = all_xy[finite]
        xmin, ymin = np.min(xy, axis=0)
        xmax, ymax = np.max(xy, axis=0)
        dx = max(xmax - xmin, 1.0)
        dy = max(ymax - ymin, 1.0)
        pad_x = 0.02 * dx
        pad_y = 0.02 * dy
        for ax in axes:
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            if DISPLAY_INVERT_Y:
                ax.set_ylim(ymax + pad_y, ymin - pad_y)
            else:
                ax.set_ylim(ymin - pad_y, ymax + pad_y)

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(
        scatter_before,
        ax=axes,
        fraction=0.030,
        pad=0.025,
        shrink=0.82,
    )
    cbar.set_label(f"log1p({gene}) expression")
    cbar.ax.tick_params(labelsize=8.5)

    fig.suptitle(
        f"{gene} spatial expression before and after low-rank reconstruction",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.03, right=0.91, bottom=0.04, top=0.88, wspace=0.08)
    plt.show()
    # png_path = outdir / f"{stem}.png"
    # # pdf_path = outdir / f"{stem}.pdf"
    # fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    # # fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    # plt.close(fig)

    # print(f"[SAVE] PCP4 comparison PNG: {png_path}")


def regenerate_low_rank_representation(
    input_h5ad: Path = INPUT_H5AD,
    output_h5ad: Path = OUTPUT_H5AD,
) -> tuple[AnnData, AnnData]:
    """Load the integrated DLPFC AnnData and regenerate the low-rank support."""
    input_h5ad = Path(input_h5ad)
    output_h5ad = Path(output_h5ad)

    if not input_h5ad.is_file():
        raise FileNotFoundError(f"Input H5AD not found: {input_h5ad}")

    print(f"[LOAD] {input_h5ad}")
    adata_full = sc.read_h5ad(input_h5ad)
    if "valid" in adata_full.obs.columns:
        adata_full = adata_full[adata_full.obs["valid"].astype(bool)].copy()
    print(
        f"[INFO] Input AnnData: n_obs={adata_full.n_obs:,}, "
        f"n_vars={adata_full.n_vars:,}"
    )

    if "spatial" not in adata_full.obsm:
        raise KeyError(
            f"Input H5AD does not contain obsm['spatial']. "
            f"Available keys: {list(adata_full.obsm.keys())}"
        )

    marker_resolved = _resolve_gene_name(adata_full, MARKER_GENE)
    print(f"[INFO] Marker found: {marker_resolved}")
    print(f"[INFO] Low-rank settings: {LOW_RANK_KWARGS}")

    # adata_lowrank = spatial_regrid_fuse_gpu_robust(
    #     adata_full,
    #     **LOW_RANK_KWARGS,
    # )
    adata_lowrank = spatial_regrid_fuse_gpu_robust(
        adata_full,
        grid_size=5,
        fuse_radius=10,  # ~1.5–2.0 * grid_size works well
        keep_nuc_frac=0.6,
        knee_mode=False,
        alpha=0.6,  # prioritize geometry a bit for deformation
        pca_dim=32,
        device="cuda",
        use_raw=False
    )

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)

    # Older AnnData/HDF5 writers may reject None values in .uns. Remove only
    # such bookkeeping entries before serialization; this does not alter X,
    # spatial coordinates, transported annotations, or analysis parameters.
    for key in list(adata_lowrank.uns.keys()):
        if adata_lowrank.uns[key] is None:
            del adata_lowrank.uns[key]

    adata_lowrank.write_h5ad(output_h5ad)
    print(f"[SAVE] Low-rank AnnData: {output_h5ad}")
    print(
        f"[INFO] Output AnnData: n_obs={adata_lowrank.n_obs:,}, "
        f"n_vars={adata_lowrank.n_vars:,}"
    )

    return adata_full, adata_lowrank


def main() -> None:
    adata_full, adata_lowrank = regenerate_low_rank_representation()

    plot_marker_before_after_low_rank(
        adata_full,
        adata_lowrank,
        gene=MARKER_GENE,
        outdir=OUTPUT_DIR,
    )

    print("[DONE] DLPFC low-rank regeneration and PCP4 visualization completed.")


if __name__ == "__main__":
    main()
