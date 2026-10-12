"""Population-level aggregation and low-rank spatial-pattern analysis.

Implements Methods Eqs. (36)-(39): after spatial matching to a reference
sample, the pooled evidence of all samples is coupled to the reference support
by a sparse Sinkhorn transport whose cost mixes spatial and molecular
dissimilarity,

``C_ri(u) = alpha * C^(x)_ri(u) + (1 - alpha) * C^(z)_ri(u)``  (alpha = 0.6),

the coupling-weighted profiles are denoised by adaptive singular-value
thresholding (retained singular-value mass ``eta``) and edge-preserving
anisotropic diffusion of the latent representation, and the coupling and
low-rank reconstruction are iterated for a fixed number of rounds. Every source
sample receives equal total mass, distributed uniformly over its locations.

The entry point is :func:`integrate_population`
(alias of the original ``spatial_regrid_fuse_visium_gpu_robust``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata import AnnData
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from ..settings import log

__all__ = ["integrate_population", "spatial_regrid_fuse_visium_gpu_robust"]


def _to_dense_local(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _safe_l2_normalize_np(X):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[~np.isfinite(norms) | (norms < 1e-8)] = 1.0
    out = X / norms
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _build_visium_lattice_edges(grid):
    """
    Build first-ring edges directly from the fixed Visium coordinates.

    This replaces the square-pixel `_build_lattice_edges(grid, grid_size)`
    operation. The lattice spacing is inferred from the supported Visium grid,
    preserving the actual coordinate system rather than rebuilding a grid.
    """
    grid = np.asarray(grid, dtype=np.float32)
    m = grid.shape[0]
    if m < 2:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.nan,
        )

    tree = cKDTree(grid)
    k_query = min(8, m)
    dists, inds = tree.query(grid, k=k_query)

    if k_query == 1:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.nan,
        )

    first_positive = dists[:, 1]
    first_positive = first_positive[np.isfinite(first_positive) & (first_positive > 0)]
    if first_positive.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.nan,
        )

    lattice_spacing = float(np.median(first_positive))
    radius = 1.35 * lattice_spacing

    edges = set()
    for i in range(m):
        for distance, j in zip(
            np.atleast_1d(dists[i, 1:]),
            np.atleast_1d(inds[i, 1:]),
        ):
            if not np.isfinite(distance) or distance > radius:
                continue
            j = int(j)
            if i == j:
                continue
            edges.add((min(i, j), max(i, j)))

    if not edges:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            lattice_spacing,
        )

    edges = sorted(edges)
    return (
        np.asarray([x[0] for x in edges], dtype=np.int64),
        np.asarray([x[1] for x in edges], dtype=np.int64),
        lattice_spacing,
    )


def _sinkhorn_sparse_local(
    a,
    b,
    row_idx,
    col_idx,
    kernel_values,
    n_cols,
    eps_iter=300,
    tol=1e-6,
):
    """Sparse Sinkhorn scaling on a neighbor-limited kernel."""
    n_rows = int(a.shape[0])
    indices = torch.stack([row_idx, col_idx], dim=0)

    Ksp = torch.sparse_coo_tensor(
        indices,
        kernel_values,
        size=(n_rows, n_cols),
        device=a.device,
        dtype=a.dtype,
    ).coalesce()

    KTsp = torch.sparse_coo_tensor(
        torch.stack([col_idx, row_idx], dim=0),
        kernel_values,
        size=(n_cols, n_rows),
        device=a.device,
        dtype=a.dtype,
    ).coalesce()

    tiny = torch.tensor(1e-20, device=a.device, dtype=a.dtype)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for iteration in range(int(eps_iter)):
        Kv = torch.sparse.mm(
            Ksp,
            v.unsqueeze(1),
        ).squeeze(1)
        if torch.any(Kv <= 0):
            raise FloatingPointError("Sparse Sinkhorn encountered a row with zero kernel mass.")
        u_new = a / Kv.clamp_min(tiny)

        KTu = torch.sparse.mm(
            KTsp,
            u_new.unsqueeze(1),
        ).squeeze(1)
        if torch.any(KTu <= 0):
            raise FloatingPointError("Sparse Sinkhorn encountered a grid column with zero mass.")
        v_new = b / KTu.clamp_min(tiny)

        if iteration % 10 == 0:
            du = torch.max(torch.abs(u_new - u) / torch.clamp(torch.abs(u), min=1e-8))
            dv = torch.max(torch.abs(v_new - v) / torch.clamp(torch.abs(v), min=1e-8))
            if max(float(du), float(dv)) < tol:
                u, v = u_new, v_new
                break

        u, v = u_new, v_new

    return u, v, Ksp, KTsp


def _select_adaptive_rank(singular_values, keep_nuc_frac, knee):
    singular_values = np.asarray(singular_values, dtype=float)
    singular_values = singular_values[np.isfinite(singular_values) & (singular_values > 0)]
    if singular_values.size == 0:
        return 0

    if knee and singular_values.size >= 3:
        y = np.log1p(singular_values)
        x = np.linspace(0.0, 1.0, len(y))
        y_scaled = (y - y[-1]) / max(y[0] - y[-1], 1e-12)
        line = 1.0 - x
        knee_index = int(np.argmax(y_scaled - line))
        return max(1, knee_index + 1)

    keep_nuc_frac = float(np.clip(keep_nuc_frac, 1e-6, 1.0))
    cumulative = np.cumsum(singular_values)
    threshold = keep_nuc_frac * cumulative[-1]
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def _svt_adaptive_local(
    X,
    keep_nuc_frac=0.90,
    knee=False,
):
    """
    Adaptive singular-value thresholding.

    The retained rank is selected by cumulative nuclear-norm fraction or the
    singular-value knee. Values below the first discarded singular value are
    removed through soft thresholding.
    """
    if X.ndim != 2:
        raise ValueError("SVT input must be a two-dimensional matrix.")
    if min(X.shape) == 0:
        return X, X[:, :0], X[:0, :]

    U, S, Vt = torch.linalg.svd(
        X,
        full_matrices=False,
    )

    rank = _select_adaptive_rank(
        S.detach().cpu().numpy(),
        keep_nuc_frac=keep_nuc_frac,
        knee=knee,
    )
    if rank <= 0:
        return (
            torch.zeros_like(X),
            X[:, :0],
            X[:0, :],
        )

    rank = min(rank, int(S.numel()))
    threshold = S[rank] if rank < int(S.numel()) else torch.zeros((), device=X.device, dtype=X.dtype)
    shrunk = torch.clamp(S[:rank] - threshold, min=0.0)

    # Prevent a degenerate all-zero output when the spectrum is flat.
    if not torch.any(shrunk > 0):
        shrunk = S[:rank]

    A_latent = U[:, :rank] * shrunk.unsqueeze(0)
    Vt_keep = Vt[:rank, :]
    X_reconstructed = A_latent @ Vt_keep

    return X_reconstructed, A_latent, Vt_keep


def _latent_aniso_local(
    latent,
    edge_i,
    edge_j,
    iters=10,
    tau=0.15,
    kappa=1.0,
):
    """Edge-preserving anisotropic diffusion on the fixed Visium lattice."""
    if int(iters) <= 0 or latent.shape[1] == 0 or len(edge_i) == 0:
        return latent

    edge_i_t = torch.as_tensor(
        edge_i,
        device=latent.device,
        dtype=torch.long,
    )
    edge_j_t = torch.as_tensor(
        edge_j,
        device=latent.device,
        dtype=torch.long,
    )

    tau = float(tau)
    kappa_sq = max(float(kappa) ** 2, 1e-12)

    A = latent
    for _ in range(int(iters)):
        delta = A[edge_j_t] - A[edge_i_t]
        squared_distance = torch.mean(delta * delta, dim=1)
        conductance = torch.exp(-squared_distance / kappa_sq).unsqueeze(1)
        flux = conductance * delta

        update = torch.zeros_like(A)
        degree = torch.zeros(
            (A.shape[0], 1),
            device=A.device,
            dtype=A.dtype,
        )

        update.index_add_(0, edge_i_t, flux)
        update.index_add_(0, edge_j_t, -flux)
        degree.index_add_(
            0,
            edge_i_t,
            conductance,
        )
        degree.index_add_(
            0,
            edge_j_t,
            conductance,
        )

        A = A + tau * update / degree.clamp_min(1e-8)

    return A


def _ensure_every_grid_column_has_an_edge(
    coords,
    grid,
    neighbor_lists,
):
    """
    Add a nearest-spot edge for any supported Visium point with no incoming edge.

    This guarantees that the output retains exactly the supplied supported
    Visium grid instead of silently dropping unsupported columns.
    """
    n_grid = grid.shape[0]
    counts = np.zeros(n_grid, dtype=np.int64)
    for neighbors in neighbor_lists:
        counts[np.asarray(neighbors, dtype=np.int64)] += 1

    orphan_grid = np.where(counts == 0)[0]
    if orphan_grid.size == 0:
        return neighbor_lists

    spot_tree = cKDTree(coords)
    _, nearest_spot = spot_tree.query(
        grid[orphan_grid],
        k=1,
    )

    neighbor_sets = [set(items) for items in neighbor_lists]
    for grid_index, spot_index in zip(
        orphan_grid.tolist(),
        np.atleast_1d(nearest_spot).astype(int).tolist(),
    ):
        neighbor_sets[spot_index].add(int(grid_index))

    return [sorted(items) for items in neighbor_sets]


def _transport_attributes_from_plan(
    attrs,
    attr_schema,
    row_idx,
    col_idx,
    edge_mass,
    n_grid,
):
    """
    Transport attributes through the final OT plan.

    Categorical attributes are assigned by maximum transported probability.
    Continuous attributes are assigned by transported weighted mean.
    """
    if attrs is None or attr_schema is None:
        return pd.DataFrame(index=np.arange(n_grid)), {}, {}

    attrs = attrs.reset_index(drop=True)
    if len(attrs) != int(row_idx.max().item()) + 1:
        # row_idx should cover all source rows because each source has a neighbor.
        if len(attrs) < int(row_idx.max().item()) + 1:
            raise ValueError("Attribute rows are not aligned with the OT source rows.")

    row_np = row_idx.detach().cpu().numpy().astype(np.int64)
    col_np = col_idx.detach().cpu().numpy().astype(np.int64)
    mass_np = edge_mass.detach().cpu().numpy().astype(np.float64)

    obs_out = pd.DataFrame(index=np.arange(n_grid))
    probability_matrices = {}
    category_metadata = {}

    for column, kind in attr_schema.items():
        if column not in attrs:
            raise KeyError(f"Attribute column {column!r} is absent from attrs.")

        kind = str(kind).lower()
        values = attrs[column]

        if kind == "categorical":
            categorical = pd.Categorical(values)
            categories = [str(x) for x in categorical.categories]
            codes = categorical.codes.astype(np.int64)
            n_categories = len(categories)

            probabilities = np.zeros(
                (n_grid, n_categories),
                dtype=np.float64,
            )
            valid_source = codes[row_np] >= 0
            np.add.at(
                probabilities,
                (
                    col_np[valid_source],
                    codes[row_np[valid_source]],
                ),
                mass_np[valid_source],
            )

            row_sum = probabilities.sum(axis=1, keepdims=True)
            normalized = np.divide(
                probabilities,
                row_sum,
                out=np.zeros_like(probabilities),
                where=row_sum > 0,
            )

            best = np.argmax(normalized, axis=1)
            labels = np.asarray(categories, dtype=object)[best]
            labels[row_sum[:, 0] <= 0] = pd.NA
            obs_out[column] = pd.Categorical(
                labels,
                categories=categories,
            )

            probability_matrices[column] = normalized.astype(np.float32)
            category_metadata[column] = categories

        elif kind in {"continuous", "numeric", "float"}:
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
            valid_source = np.isfinite(numeric[row_np])

            weighted_sum = np.zeros(n_grid, dtype=np.float64)
            weight_sum = np.zeros(n_grid, dtype=np.float64)
            np.add.at(
                weighted_sum,
                col_np[valid_source],
                mass_np[valid_source] * numeric[row_np[valid_source]],
            )
            np.add.at(
                weight_sum,
                col_np[valid_source],
                mass_np[valid_source],
            )
            obs_out[column] = np.divide(
                weighted_sum,
                weight_sum,
                out=np.full(n_grid, np.nan, dtype=np.float64),
                where=weight_sum > 0,
            )
        else:
            raise ValueError(f"Unsupported attribute type {kind!r} for {column!r}.")

    return obs_out, probability_matrices, category_metadata


def spatial_regrid_fuse_visium_gpu_robust(
    adata: AnnData,
    fixed_grid_pts,
    grid_support_mask,
    point_masses=None,
    fuse_radius: float = None,
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
    attr_schema=None,
):
    """
    Sparse GPU OT fusion on a fixed supported Visium grid.

    This is the fixed-grid adaptation of `spatial_regrid_fuse_gpu_robust`.
    The original `_make_grid(coords, grid_size)` step is deliberately removed.
    Output positions are exactly:

        fixed_grid_pts[grid_support_mask]

    The `grid_size` argument is retained so the exposed call uses the same
    parameter values as the original method. For the fixed-grid adaptation it
    is recorded as provenance; lattice geometry is inferred from the supplied
    Visium coordinates.
    """
    if device == "cuda" and not torch.cuda.is_available():
        log("[axis] CUDA requested but unavailable; using CPU.", level=2)
        device = "cpu"

    coords = np.asarray(
        adata.obsm["spatial"],
        dtype=np.float32,
    )
    X_source = adata.raw.X if use_raw and adata.raw is not None else adata.X
    X_spot_np = _to_dense_local(X_source).astype(np.float32)
    var_ref = adata.raw.var.copy() if use_raw and adata.raw is not None else adata.var.copy()

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must have shape (n, >=2).")
    coords = coords[:, :2]
    if coords.shape[0] != X_spot_np.shape[0]:
        raise ValueError("Coordinate and expression row counts differ.")
    if not np.all(np.isfinite(coords)):
        raise ValueError("Source coordinates contain non-finite values.")
    if not np.all(np.isfinite(X_spot_np)):
        raise ValueError("Source expression contains non-finite values.")
    if np.any(X_spot_np < 0):
        raise ValueError("OT fusion expects non-negative expression values.")

    n, n_genes = X_spot_np.shape
    if n < 2:
        raise ValueError("At least two source observations are required.")
    if n_genes < 1:
        raise ValueError("At least one gene is required.")

    grid_full = np.asarray(
        fixed_grid_pts,
        dtype=np.float32,
    )
    support_mask = np.asarray(
        grid_support_mask,
        dtype=bool,
    )
    if grid_full.ndim != 2 or grid_full.shape[1] < 2 or support_mask.shape != (grid_full.shape[0],):
        raise ValueError("fixed_grid_pts and grid_support_mask are incompatible.")

    grid = grid_full[support_mask, :2].copy()
    if grid.shape[0] == 0:
        raise ValueError("No Visium grid positions are supported.")

    if point_masses is None:
        point_masses = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        point_masses = np.asarray(point_masses, dtype=np.float64)
        if point_masses.shape != (n,):
            raise ValueError(f"point_masses must have shape ({n},).")
        if not np.all(np.isfinite(point_masses)) or np.any(point_masses <= 0):
            raise ValueError("point_masses must be finite and strictly positive.")
        point_masses = point_masses / point_masses.sum()

    x_spot = torch.as_tensor(
        coords,
        device=device,
        dtype=torch.float32,
    )
    X_spot = torch.as_tensor(
        X_spot_np,
        device=device,
        dtype=torch.float32,
    )
    x_grid = torch.as_tensor(
        grid,
        device=device,
        dtype=torch.float32,
    )
    n_grid = grid.shape[0]

    # Gene embedding used only for the OT cost.
    library_size = np.clip(
        X_spot_np.sum(axis=1, keepdims=True),
        1.0,
        None,
    )
    X_norm = np.log1p(X_spot_np / library_size * 1e4)
    X_norm = np.nan_to_num(
        X_norm,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    n_components = max(
        1,
        min(
            int(pca_dim),
            int(n),
            int(X_norm.shape[1]),
        ),
    )
    pca = PCA(
        n_components=n_components,
        random_state=0,
    )
    Xp_spot_np = pca.fit_transform(X_norm).astype(np.float32)
    Xp_spot_np = _safe_l2_normalize_np(Xp_spot_np)
    Xp_spot = torch.as_tensor(
        Xp_spot_np,
        device=device,
        dtype=torch.float32,
    )

    # Neighbor-limited sparse support from source spots to fixed Visium points.
    grid_tree = cKDTree(grid)
    neighbor_lists = []

    for source_coord in coords:
        if fuse_radius is None:
            indices = list(range(n_grid))
        else:
            indices = grid_tree.query_ball_point(
                source_coord,
                r=float(fuse_radius),
            )

        if len(indices) == 0:
            _, nearest = grid_tree.query(
                source_coord,
                k=1,
            )
            indices = [int(nearest)]

        if k_mask is not None and len(indices) > int(k_mask):
            candidate = grid[np.asarray(indices, dtype=np.int64)]
            distances = np.linalg.norm(
                candidate - source_coord[None, :],
                axis=1,
            )
            order = np.argsort(distances)[: int(k_mask)]
            indices = list(np.asarray(indices, dtype=np.int64)[order])

        neighbor_lists.append([int(index) for index in indices])

    neighbor_lists = _ensure_every_grid_column_has_an_edge(
        coords=coords,
        grid=grid,
        neighbor_lists=neighbor_lists,
    )

    column_counts = np.zeros(n_grid, dtype=np.int64)
    for indices in neighbor_lists:
        column_counts[np.asarray(indices, dtype=np.int64)] += 1

    if np.any(column_counts == 0):
        raise RuntimeError("A supported Visium point has no incoming OT edge.")

    max_neighbors = max(len(items) for items in neighbor_lists)
    neighbor_index = -np.ones(
        (n, max_neighbors),
        dtype=np.int64,
    )
    for i, items in enumerate(neighbor_lists):
        neighbor_index[i, : len(items)] = items

    neighbor_index_t = torch.as_tensor(
        neighbor_index,
        device=device,
        dtype=torch.long,
    )
    valid_mask = neighbor_index_t >= 0

    grid_neighbors = x_grid[neighbor_index_t.clamp_min(0)]
    source_expanded = x_spot.unsqueeze(1).expand_as(grid_neighbors)

    spatial_cost = (source_expanded - grid_neighbors).pow(2).sum(dim=2)
    spatial_median = torch.median(spatial_cost[valid_mask]) if valid_mask.any() else torch.tensor(1.0, device=device)
    spatial_cost = spatial_cost / spatial_median.clamp(min=1e-6)

    # Initialize the target gene embedding from geometry, while respecting the
    # supplied equal-source point masses.
    with torch.no_grad():
        distances = torch.linalg.norm(
            grid_neighbors - source_expanded,
            dim=2,
        ).clamp(min=1e-12)

        local_weights = (1.0 / distances) * valid_mask.float()
        local_weights = local_weights / local_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp(min=1e-8)

        mass_t = torch.as_tensor(
            point_masses,
            device=device,
            dtype=torch.float32,
        )
        local_weights = local_weights * mass_t.unsqueeze(1)

        Xp_grid = torch.zeros(
            (n_grid, Xp_spot.shape[1]),
            device=device,
            dtype=torch.float32,
        )
        init_mass = torch.zeros(
            (n_grid, 1),
            device=device,
            dtype=torch.float32,
        )

        for neighbor_slot in range(max_neighbors):
            slot_mask = valid_mask[:, neighbor_slot]
            target_index = neighbor_index_t[
                slot_mask,
                neighbor_slot,
            ]
            slot_weight = local_weights[
                slot_mask,
                neighbor_slot,
            ].unsqueeze(1)

            Xp_grid.index_add_(
                0,
                target_index,
                slot_weight * Xp_spot[slot_mask],
            )
            init_mass.index_add_(
                0,
                target_index,
                slot_weight,
            )

        Xp_grid = Xp_grid / init_mass.clamp_min(1e-12)
        Xp_grid = Xp_grid / (Xp_grid.norm(dim=1, keepdim=True) + 1e-8)

    edge_i, edge_j, inferred_lattice_spacing = _build_visium_lattice_edges(grid)

    a = torch.as_tensor(
        point_masses,
        device=device,
        dtype=torch.float32,
    )
    b = torch.as_tensor(
        column_counts / column_counts.sum(),
        device=device,
        dtype=torch.float32,
    )

    final_row_idx = None
    final_col_idx = None
    final_kernel_values = None
    final_u = None
    final_v = None
    final_column_mass = None
    X_grid = None

    for outer_iteration in range(int(outer_loops)):
        target_neighbor_embedding = Xp_grid[neighbor_index_t.clamp_min(0)]
        cosine_similarity = (Xp_spot.unsqueeze(1) * target_neighbor_embedding).sum(dim=2)
        gene_cost = 1.0 - cosine_similarity

        gene_median = torch.median(gene_cost[valid_mask]) if valid_mask.any() else torch.tensor(1.0, device=device)
        gene_cost = gene_cost / gene_median.clamp(min=1e-6)

        total_cost = float(alpha) * spatial_cost + (1.0 - float(alpha)) * gene_cost

        kernel = torch.zeros_like(total_cost)
        kernel[valid_mask] = torch.exp(-total_cost[valid_mask] / max(float(epsilon), 1e-12)).clamp_min(1e-30)

        row_ids = (
            torch.arange(
                n,
                device=device,
            )
            .unsqueeze(1)
            .expand(
                n,
                max_neighbors,
            )
        )
        row_idx = row_ids[valid_mask]
        col_idx = neighbor_index_t[valid_mask]
        kernel_values = kernel[valid_mask]

        u, v, Ksp, KTsp = _sinkhorn_sparse_local(
            a=a,
            b=b,
            row_idx=row_idx,
            col_idx=col_idx,
            kernel_values=kernel_values,
            n_cols=n_grid,
            eps_iter=sinkhorn_iters,
            tol=1e-6,
        )

        column_mass = (
            torch.sparse.mm(
                KTsp,
                u.unsqueeze(1),
            ).squeeze(1)
            * v
        ).clamp(min=1e-12)

        X_grid = torch.zeros(
            (n_grid, n_genes),
            device=device,
            dtype=torch.float32,
        )

        for start in range(0, n_genes, int(gene_chunk)):
            end = min(n_genes, start + int(gene_chunk))
            weighted_source = u.unsqueeze(1) * X_spot[:, start:end]
            target_chunk = torch.sparse.mm(
                KTsp,
                weighted_source,
            )
            target_chunk = v.unsqueeze(1) * target_chunk
            X_grid[:, start:end] = target_chunk / column_mass.unsqueeze(1)

        X_grid, latent, Vt = _svt_adaptive_local(
            X_grid,
            keep_nuc_frac=keep_nuc_frac,
            knee=knee_mode,
        )

        if int(aniso_iters) > 0 and latent.shape[1] > 0 and len(edge_i) > 0:
            latent = _latent_aniso_local(
                latent,
                edge_i=edge_i,
                edge_j=edge_j,
                iters=aniso_iters,
                tau=aniso_tau,
                kappa=aniso_kappa,
            )
            X_grid = latent @ Vt

        grid_expression = X_grid.detach().cpu().numpy()
        grid_expression = np.nan_to_num(
            grid_expression,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        grid_expression[grid_expression < 0.0] = 0.0

        row_sum = grid_expression.sum(
            axis=1,
            keepdims=True,
        )
        row_sum[row_sum == 0.0] = 1.0
        grid_normalized = np.log1p(grid_expression / row_sum * 1e4)
        grid_embedding = pca.transform(grid_normalized)
        grid_embedding = _safe_l2_normalize_np(grid_embedding)
        Xp_grid = torch.as_tensor(
            grid_embedding,
            device=device,
            dtype=torch.float32,
        )

        b = (column_mass / column_mass.sum()).detach()

        final_row_idx = row_idx
        final_col_idx = col_idx
        final_kernel_values = kernel_values
        final_u = u
        final_v = v
        final_column_mass = column_mass

        log(f"  [OT] outer={outer_iteration + 1}/{outer_loops} edges={len(kernel_values):,} latent_rank={latent.shape[1]}")

    if X_grid is None:
        raise RuntimeError("OT fusion did not execute any outer loop.")

    fused_expression = X_grid.detach().cpu().numpy()
    fused_expression = np.nan_to_num(
        fused_expression,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    fused_expression[fused_expression < 0.0] = 0.0

    final_edge_mass = final_u[final_row_idx] * final_kernel_values * final_v[final_col_idx]

    attrs_source = adata.obs.loc[:, list(attr_schema)].copy() if attr_schema else None
    obs_out, attr_probabilities, attr_categories = _transport_attributes_from_plan(
        attrs=attrs_source,
        attr_schema=attr_schema,
        row_idx=final_row_idx,
        col_idx=final_col_idx,
        edge_mass=final_edge_mass,
        n_grid=n_grid,
    )

    new_adata = AnnData(
        X=fused_expression,
        obs=obs_out,
        var=var_ref.copy(),
    )
    new_adata.obsm["spatial"] = grid.astype(np.float32)
    new_adata.obs_names = [f"visium_grid_{i}" for i in range(new_adata.n_obs)]

    for attribute, probabilities in attr_probabilities.items():
        new_adata.obsm[f"{attribute}_probabilities"] = probabilities
        new_adata.uns[f"{attribute}_categories"] = attr_categories[attribute]

    new_adata.obs["fixed_grid_index"] = np.flatnonzero(support_mask).astype(np.int64)
    new_adata.obs["ot_target_mass"] = final_column_mass.detach().cpu().numpy().astype(np.float32)

    new_adata.uns.update(
        dict(
            integration_generator=("spatial_regrid_fuse_visium_gpu_robust"),
            fixed_visium_grid=True,
            inferred_visium_lattice_spacing=(
                None if not np.isfinite(inferred_lattice_spacing) else float(inferred_lattice_spacing)
            ),
            fuse_radius=(None if fuse_radius is None else float(fuse_radius)),
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
            k_mask=int(k_mask),
            n_ot_edges=int(len(final_kernel_values)),
            n_full_visium_grid_points=int(grid_full.shape[0]),
            n_supported_visium_grid_points=int(grid.shape[0]),
        )
    )

    return new_adata


#: Public name of the population-integration routine.
integrate_population = spatial_regrid_fuse_visium_gpu_robust
