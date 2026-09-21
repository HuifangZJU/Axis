import numpy as np
import torch
from anndata import AnnData
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

# -------------------- small helpers --------------------
def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def _make_grid(coords: np.ndarray, grid: float) -> np.ndarray:
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    n_cols = int(np.ceil((x_max - x_min) / grid)) + 1
    n_rows = int(np.ceil((y_max - y_min) / grid)) + 1
    gx = np.arange(x_min, x_min + n_cols * grid, grid)
    gy = np.arange(y_min, y_min + n_rows * grid, grid)
    return np.array([(x, y) for y in gy for x in gx], dtype=np.float32)

def _build_lattice_edges(grid_xy: np.ndarray, grid_size: float):
    x0, y0 = grid_xy.min(axis=0)
    gx = np.rint((grid_xy[:, 0] - x0) / grid_size).astype(np.int64)
    gy = np.rint((grid_xy[:, 1] - y0) / grid_size).astype(np.int64)
    key = (gx << 32) + gy
    idx_map = {int(k): i for i, k in enumerate(key)}
    offs = [(1,0),(-1,0),(0,1),(0,-1)]
    ei, ej = [], []
    for i,(ix,iy) in enumerate(zip(gx,gy)):
        for dx,dy in offs:
            j = idx_map.get(int(((ix+dx)<<32)+(iy+dy)),-1)
            if j >= 0 and j > i: ei.append(i); ej.append(j)
    if not ei:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2).fit(grid_xy)
        nbrs = nn.kneighbors(return_distance=False)
        for i in range(grid_xy.shape[0]):
            j = int(nbrs[i,1])
            if j>i: ei.append(i); ej.append(j)
    return np.asarray(ei,np.int64), np.asarray(ej,np.int64)

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
    gene_chunk: int = 1024,     # set smaller if G is huge
    epsilon: float = 3.0,       # entropic ε (fixed here for sparse K)
    sinkhorn_iters: int = 300,
    device: str = "cuda",
    use_raw: bool = False,
):
    """
    Memory-efficient GPU OT fusion:
    - neighbor-limited sparse kernel
    - sparse Sinkhorn
    - T^T @ X done in gene-chunks
    """
    if device=="cuda" and not torch.cuda.is_available():
        device="cpu"

    # --- data
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    X_spot_np = _to_dense(adata.raw.X if (use_raw and adata.raw is not None) else adata.X).astype(np.float32)
    var_ref = (adata.raw.var if (use_raw and adata.raw is not None) else adata.var).copy()

    n, G = X_spot_np.shape
    x_spot = torch.tensor(coords, device=device)
    X_spot = torch.tensor(X_spot_np, device=device)

    # --- grid + radius pruning
    grid_full = _make_grid(coords, grid_size)
    if fuse_radius is not None:
        dmin,_ = cKDTree(coords).query(grid_full, k=1)
        keep = dmin <= fuse_radius
        grid = grid_full[keep]
    else:
        grid = grid_full
    if grid.shape[0]==0:
        raise ValueError("No grid cells within fuse_radius.")
    x_grid = torch.tensor(grid, device=device)
    m = x_grid.shape[0]

    # --- gene embedding for cost (spots): lib-norm -> log1p -> PCA -> L2
    Xn = X_spot_np / np.clip(X_spot_np.sum(axis=1, keepdims=True), 1.0, None) * 1e4
    Xn = np.log1p(Xn)
    pca = PCA(n_components=min(pca_dim, min(n, Xn.shape[1])), random_state=0)
    Xp_spot_np = pca.fit_transform(Xn).astype(np.float32)
    Xp_spot_np /= (np.linalg.norm(Xp_spot_np, axis=1, keepdims=True) + 1e-8)
    Xp_spot = torch.tensor(Xp_spot_np, device=device)

    # --- neighbor list (radius + KNN; CPU build)
    kd = cKDTree(grid)
    nbr_lists = []
    for i, s in enumerate(coords):
        idxs = kd.query_ball_point(s, r=fuse_radius) if fuse_radius is not None else list(range(m))
        if len(idxs)==0:
            j = int(np.argmin(np.linalg.norm(grid - s[None,:], axis=1)))
            idxs = [j]
        if k_mask is not None and len(idxs)>k_mask:
            sub = grid[idxs]; d = np.linalg.norm(sub - s[None,:], axis=1)
            order = np.argsort(d)[:k_mask]; idxs = list(np.asarray(idxs)[order])
        nbr_lists.append(idxs)

    # drop grid cols with no incoming edges
    col_counts = np.zeros(m, dtype=np.int64)
    for idxs in nbr_lists:
        for j in idxs: col_counts[j]+=1
    keep_cols = col_counts > 0
    if not keep_cols.all():
        remap = -np.ones(m, dtype=np.int64)
        new_idx = np.cumsum(keep_cols)-1
        remap[keep_cols] = new_idx[keep_cols]
        grid = grid[keep_cols]; x_grid = x_grid[keep_cols]
        m = x_grid.shape[0]
        nbr_lists = [[int(remap[j]) for j in lst if keep_cols[j]] for lst in nbr_lists]

    # pad neighbor lists to K
    Kmax = max(len(lst) for lst in nbr_lists)
    nbr_idx = -np.ones((n, Kmax), dtype=np.int64)
    for i,lst in enumerate(nbr_lists):
        nbr_idx[i,:len(lst)] = lst
    nbr_idx_t = torch.tensor(nbr_idx, device=device)
    valid_mask = (nbr_idx_t >= 0)

    # --- costs ONLY on neighbors
    # spatial
    gs = x_grid[nbr_idx_t.clamp_min(0)]            # (n,K,2)
    ss = x_spot.unsqueeze(1).expand_as(gs)
    Cs = (ss - gs).pow(2).sum(dim=2)
    s_med = torch.median(Cs[valid_mask]) if valid_mask.any() else torch.tensor(1.0, device=device)
    Cs = Cs / s_med.clamp(min=1e-6)

    # init grid embedding from nearby spots (cheap)
    with torch.no_grad():
        # soft average using distance on GPU
        dgs = torch.linalg.norm(gs - ss, dim=2).clamp(min=1e-12)  # (n,K)
        w = (1.0 / dgs) * valid_mask.float()
        w = w / (w.sum(dim=1, keepdim=True).clamp(min=1e-8))
        # build Xp_grid as weighted avg of Xp_spot
        # gather spot embeddings per neighbor structure: sum_i w_ik * Xp_spot[i]
        # We'll do it via scatter-add to grid
        Xp_grid = torch.zeros((m, Xp_spot.shape[1]), device=device)
        for k in range(Kmax):
            maskk = valid_mask[:,k]
            idxj = nbr_idx_t[maskk, k]    # columns
            contrib = (w[maskk, k].unsqueeze(1) * Xp_spot[maskk])  # (Nk, d)
            Xp_grid.index_add_(0, idxj, contrib)
        # L2 normalize
        Xp_grid = Xp_grid / (Xp_grid.norm(dim=1, keepdim=True) + 1e-8)

    # lattice edges (for latent diffusion later)
    ei_np, ej_np = _build_lattice_edges(grid, grid_size)

    # ---------- outer loops ----------
    a = torch.full((n,), 1.0/n, device=device)
    # warm-start b proportional to in-degree
    b0 = torch.tensor(col_counts[keep_cols] / col_counts[keep_cols].sum(), device=device, dtype=torch.float32)
    b = b0.clone()

    for _ in range(outer_loops):
        # gene cost (cosine) on neighbors
        Xg_nb = Xp_grid[nbr_idx_t.clamp_min(0)]                      # (n,K,d)
        dots = (Xp_spot.unsqueeze(1) * Xg_nb).sum(dim=2)             # (n,K)
        Cg = 1.0 - dots
        g_med = torch.median(Cg[valid_mask]) if valid_mask.any() else torch.tensor(1.0, device=device)
        Cg = Cg / g_med.clamp(min=1e-6)

        C = alpha * Cs + (1.0 - alpha) * Cg                          # (n,K)

        # sparse kernel K = exp(-C/eps) only on valid pairs
        Kvals = torch.zeros_like(C)
        Kvals[valid_mask] = torch.exp(-C[valid_mask] / max(epsilon,1e-12))

        # build sparse indices
        row_ids = torch.arange(n, device=device).unsqueeze(1).expand(n, Kmax)
        row_idx = row_ids[valid_mask]                                # (nnz,)
        col_idx = nbr_idx_t[valid_mask]                              # (nnz,)
        K_vals = Kvals[valid_mask]                                   # (nnz,)

        # Sinkhorn with sparse kernel
        u, v, Ksp, KTsp = _sinkhorn_sparse(a, b, row_idx, col_idx, K_vals, m, eps_iter=sinkhorn_iters, tol=1e-6)

        # ---- fused matrix in G-chunks: X_grid = diag(v) @ (K^T @ (diag(u) @ X_spot)) / col_mass
        # col_mass = v * (K^T @ u)
        col_mass = torch.sparse.mm(KTsp, u.unsqueeze(1)).squeeze(1) * v
        col_mass = col_mass.clamp(min=1e-12)

        X_grid = torch.zeros((m, G), device=device, dtype=torch.float32)
        for s in range(0, G, gene_chunk):
            e = min(G, s + gene_chunk)
            Z = (u.unsqueeze(1) * X_spot[:, s:e])                    # (n, gchunk)
            Y = torch.sparse.mm(KTsp, Z)                             # (m, gchunk)
            Y = v.unsqueeze(1) * Y                                   # diag(v)
            X_grid[:, s:e] = Y / col_mass.unsqueeze(1)

        # ---- adaptive SVT (no explicit rank) ----
        X_grid, A_latent, Vt = _svt_adaptive(X_grid, keep_nuc_frac=keep_nuc_frac, knee=knee_mode)

        # ---- latent anisotropic diffusion (edge-preserving; no KNN) ----
        if aniso_iters > 0 and A_latent.shape[1] > 0 and len(ei_np) > 0:
            A_latent = _latent_aniso(A_latent, ei_np, ej_np, iters=aniso_iters, tau=aniso_tau, kappa=aniso_kappa)
            X_grid = A_latent @ Vt

        # update grid embedding for next cost (reuse PCA)
        Xg_np = X_grid.detach().cpu().numpy()
        Xg_np = np.nan_to_num(Xg_np, nan=0.0, posinf=0.0, neginf=0.0)
        Xg_np[Xg_np < 0.0] = 0.0
        rs = Xg_np.sum(axis=1, keepdims=True); rs[rs==0.0] = 1.0
        Xg_norm = np.log1p((Xg_np / rs) * 1e4)
        Xg_norm = np.nan_to_num(Xg_norm, nan=0.0, posinf=0.0, neginf=0.0)
        Xg_emb = pca.transform(Xg_norm)
        Xg_emb /= (np.linalg.norm(Xg_emb, axis=1, keepdims=True) + 1e-8)
        Xp_grid = torch.tensor(Xg_emb, device=device, dtype=torch.float32)

        # refresh b to current mass (helps convergence a bit)
        b = (col_mass / col_mass.sum()).detach()

    # pack AnnData
    new_adata = AnnData(X=X_grid.detach().cpu().numpy(), var=var_ref.copy())
    new_adata.obsm["spatial"] = grid.astype(np.float32)
    new_adata.obs_names = [f"grid_{i}" for i in range(new_adata.n_obs)]
    new_adata.uns.update(dict(
        grid_size=float(grid_size),
        fuse_radius=None if fuse_radius is None else float(fuse_radius),
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
        k_mask=int(k_mask)
    ))
    return new_adata