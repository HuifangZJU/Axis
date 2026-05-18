# ============================================================
# SCC evaluation for AXIS vs SpatialZ
# Structural preservation + cluster-anchored molecular fidelity
# ============================================================

import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

from anndata import AnnData
from scipy import sparse
from scipy.spatial import cKDTree, distance
from scipy.stats import entropy, pearsonr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CONFIG
# ============================================================

ROOT = "/media/huifang/data/registration/SCC/huifang"
NET_IN = os.path.join(ROOT, "network_input")
H5AD_DIR = os.path.join(ROOT, "h5ad")
VIRTUAL_DIR = os.path.join(ROOT, "virtual_tissue")
OUTDIR = os.path.join(VIRTUAL_DIR, "evaluation")

GROUP_IDS = [2, 5, 9, 10]
SAMPLE_IDS = [1]   # target slice id
SID_TO_Z = {
    0: 0,
    1: 10,
    2: 20,
}

cluster_KEY = "original_clusters"
COORD_KEY = "spatial"

# ------------ evaluation parameters ------------
NORMALIZE_TOTAL = True
TARGET_SUM = 1e4
LOG1P = True

TOP_MARKERS_PER_CLUSTER = 30
TOP_CASE_GENES_PER_CLUSTER = 3
KNN_ADJ = 6

# genes for case study; if None, auto-select from real markers
CASE_GENES = ["KRT1"]   # e.g. ["KRT1"]
CASE_GROUP = 10

# plotting
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13


MAIN_METRICS = [
    "cluster_comp_l1",
    "cluster_adj_corr",
    # "mean_cluster_gene_corr",
]

METRIC_LABELS = {
    "cluster_comp_l1": "Cluster composition error (L1)",
    "cluster_adj_corr": "Cluster adjacency similarity",
    # "mean_cluster_gene_corr": "Cluster gene-program fidelity",
}

from matplotlib.patches import Patch

def set_cell_nature_plot_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.8,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })


def get_candidate_palette(order):
    # muted blue-gray for SpatialZ, warm red for AXIS
    base = {
        "SpatialZ": "#7C8DA6",
        "AXIS": "#C4473A",
    }
    return {k: base.get(k, "#888888") for k in order}


def add_group_pair_lines(ax, df, x_col, y_col, group_col, order,
                         color="0.80", lw=1.0, alpha=0.85):
    x_map = {c: i for i, c in enumerate(order)}
    for gid, sub in df.groupby(group_col):
        sub = sub.dropna(subset=[y_col]).copy()
        if sub.shape[0] < 2:
            continue
        sub = sub.set_index(x_col)
        xs, ys = [], []
        for c in order:
            if c in sub.index:
                val = sub.loc[c, y_col]
                if np.ndim(val) > 0:
                    val = np.asarray(val).ravel()[0]
                if pd.notnull(val):
                    xs.append(x_map[c])
                    ys.append(float(val))
        if len(xs) >= 2:
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=1)


# ============================================================
# BASIC UTILITIES
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def extract_matrix(adata: AnnData, genes: Sequence[str]) -> np.ndarray:
    genes = [g for g in genes if g in adata.var_names]
    if len(genes) == 0:
        return np.zeros((adata.n_obs, 0), dtype=np.float32)
    X = adata[:, genes].X
    return to_dense(X).astype(np.float32, copy=False)


def preprocess_adata(
    adata: AnnData,
    normalize_total: bool = True,
    target_sum: float = 1e4,
    log1p: bool = True,
) -> AnnData:
    ad = adata.copy()
    if normalize_total:
        sc.pp.normalize_total(ad, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(ad)
    return ad


def common_gene_intersection(adatas: Sequence[AnnData]) -> List[str]:
    common = set(adatas[0].var_names.astype(str))
    for ad in adatas[1:]:
        common &= set(ad.var_names.astype(str))
    common = sorted(common)
    return common


def subset_to_genes(adata: AnnData, genes: Sequence[str]) -> AnnData:
    genes = [g for g in genes if g in adata.var_names]
    return adata[:, genes].copy()


def canonical_cluster_order(labels: Sequence) -> List[str]:
    vals = pd.Series(labels).astype(str)
    uniq = list(pd.unique(vals))
    # numeric-first sorting if possible
    def key_fn(x):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)
    return sorted(uniq, key=key_fn)


def get_cluster_labels(adata: AnnData, cluster_key: str) -> np.ndarray:
    if cluster_key not in adata.obs:
        raise KeyError(f"'{cluster_key}' not found in adata.obs")
    return adata.obs[cluster_key].astype(str).values


def zscore_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.maximum(sd, eps)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def flatten_upper_triangle(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)
    return 0.5 * entropy(p + eps, m + eps) + 0.5 * entropy(q + eps, m + eps)


def discrete_cmap_from_order(order: Sequence[str], cmap_name: str = "tab10") -> Dict[str, Tuple[float, float, float, float]]:
    cmap = plt.get_cmap(cmap_name)
    color_dict = {}
    for i, k in enumerate(order):
        color_dict[str(k)] = cmap(i % cmap.N)
    return color_dict


# ============================================================
# LOADING
# ============================================================
def load_ad(path):
    ad = sc.read_h5ad(path)
    if "valid" in ad.obs.columns:
        ad = ad[ad.obs["valid"]].copy()
    return ad
def load_group_data(group_id: int) -> Dict[str, AnnData]:

    real_path = os.path.join(H5AD_DIR, f"{group_id}_{1}.h5ad")
    axis_path = f"/media/huifang/data/registration/SCC/huifang/virtual_tissue/scc_z_group{group_id}_grid40_spatial3_gene_weight/query/virtual_scc_z_10.000.h5ad"
    spatialz_path = f"/media/huifang/data/registration/SCC/huifang/virtual_tissue/spatialz/group{group_id}/patient{group_id}_generation_10.h5ad"
    anchor_neg_path = os.path.join(H5AD_DIR, f"{group_id}_{0}.h5ad")
    anchor_pos_path = os.path.join(H5AD_DIR, f"{group_id}_{2}.h5ad")

    data = {
        "Real": load_ad(real_path),
        "AXIS": load_ad(axis_path),
        "SpatialZ": load_ad(spatialz_path),
        "Anchor A (-10 um)": load_ad(anchor_neg_path),
        "Anchor B (+10 um)": load_ad(anchor_pos_path),
    }
    return data


# ============================================================
# STRUCTURAL METRICS
# ============================================================

def cluster_composition(
    adata: AnnData,
    cluster_key: str,
    order: Sequence[str]
) -> np.ndarray:
    labels = get_cluster_labels(adata, cluster_key)
    counts = pd.Series(labels).value_counts()
    vec = np.array([counts.get(c, 0) for c in order], dtype=np.float32)
    if vec.sum() > 0:
        vec = vec / vec.sum()
    return vec


def cluster_adjacency_matrix(
    adata: AnnData,
    cluster_key: str,
    order: Sequence[str],
    coord_key: str = COORD_KEY,
    knn: int = 6,
    symmetrize: bool = True,
) -> np.ndarray:
    coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    labels = get_cluster_labels(adata, cluster_key)

    n = coords.shape[0]
    c2i = {c: i for i, c in enumerate(order)}
    M = np.zeros((len(order), len(order)), dtype=np.float64)

    if n <= 1:
        return M

    k = min(knn + 1, n)
    tree = cKDTree(coords)
    try:
        _, idx = tree.query(coords, k=k, workers=-1)
    except TypeError:
        _, idx = tree.query(coords, k=k)

    if k == 1:
        idx = idx[:, None]

    # remove self
    idx = idx[:, 1:] if idx.shape[1] > 1 else np.zeros((n, 0), dtype=int)

    for i in range(n):
        ci = c2i[str(labels[i])]
        for j in idx[i]:
            cj = c2i[str(labels[j])]
            M[ci, cj] += 1.0

    if symmetrize:
        M = 0.5 * (M + M.T)

    s = M.sum()
    if s > 0:
        M = M / s
    return M.astype(np.float32)


def cluster_centroids(
    adata: AnnData,
    cluster_key: str,
    order: Sequence[str],
    coord_key: str = COORD_KEY
) -> Dict[str, np.ndarray]:
    coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    labels = get_cluster_labels(adata, cluster_key)
    out = {}
    for c in order:
        mask = labels == str(c)
        if np.any(mask):
            out[str(c)] = coords[mask].mean(axis=0)
    return out


def centroid_distance_correlation(
    real_ad: AnnData,
    cand_ad: AnnData,
    cluster_key: str,
    order: Sequence[str],
    coord_key: str = COORD_KEY,
) -> float:
    cent_r = cluster_centroids(real_ad, cluster_key, order, coord_key)
    cent_c = cluster_centroids(cand_ad, cluster_key, order, coord_key)

    shared = [c for c in order if c in cent_r and c in cent_c]
    if len(shared) < 3:
        return np.nan

    Xr = np.stack([cent_r[c] for c in shared], axis=0)
    Xc = np.stack([cent_c[c] for c in shared], axis=0)

    Dr = distance.squareform(distance.pdist(Xr))
    Dc = distance.squareform(distance.pdist(Xc))

    vr = flatten_upper_triangle(Dr)
    vc = flatten_upper_triangle(Dc)
    return safe_pearson(vr, vc)


# ============================================================
# MOLECULAR / CLUSTER-ANCHORED METRICS
# ============================================================

def pseudobulk_by_cluster(
    adata: AnnData,
    cluster_key: str,
    genes: Sequence[str],
    order: Sequence[str]
) -> np.ndarray:
    labels = get_cluster_labels(adata, cluster_key)
    X = extract_matrix(adata, genes)

    out = np.zeros((len(order), len(genes)), dtype=np.float32)
    for i, c in enumerate(order):
        mask = labels == str(c)
        if np.any(mask):
            out[i] = X[mask].mean(axis=0)
    return out


def per_cluster_gene_correlations(
    pb_real: np.ndarray,
    pb_cand: np.ndarray,
    order: Sequence[str]
) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(order):
        r = safe_pearson(pb_real[i], pb_cand[i])
        rows.append({
            "cluster": str(c),
            "cluster_gene_corr": r
        })
    return pd.DataFrame(rows)


def compute_real_marker_sets(
    real_ad: AnnData,
    cluster_key: str,
    genes: Sequence[str],
    order: Sequence[str],
    top_n: int = 30
) -> Dict[str, List[str]]:
    pb = pseudobulk_by_cluster(real_ad, cluster_key, genes, order)
    gene_arr = np.array(genes, dtype=object)

    marker_sets = {}
    for i, c in enumerate(order):
        in_mean = pb[i]
        if len(order) > 1:
            out_mean = pb[np.arange(len(order)) != i].mean(axis=0)
        else:
            out_mean = np.zeros_like(in_mean)

        score = in_mean - out_mean
        idx = np.argsort(score)[::-1]
        keep = idx[score[idx] > 0]
        keep = keep[:top_n]
        marker_sets[str(c)] = gene_arr[keep].tolist()
    return marker_sets


def signature_score_matrix(
    adata: AnnData,
    cluster_key: str,
    marker_sets: Dict[str, List[str]],
    order: Sequence[str]
) -> np.ndarray:
    all_genes = sorted(set(sum(marker_sets.values(), [])))
    X = extract_matrix(adata, all_genes)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    labels = get_cluster_labels(adata, cluster_key)

    S = np.zeros((len(order), len(order)), dtype=np.float32)
    for i, row_cluster in enumerate(order):
        mask = labels == str(row_cluster)
        if not np.any(mask):
            continue
        Xm = X[mask].mean(axis=0)
        for j, sig_cluster in enumerate(order):
            sig_genes = [g for g in marker_sets[str(sig_cluster)] if g in gene_to_idx]
            if len(sig_genes) == 0:
                continue
            idx = [gene_to_idx[g] for g in sig_genes]
            S[i, j] = float(Xm[idx].mean())
    return S


def top_signature_agreement(
    S_real: np.ndarray,
    S_cand: np.ndarray,
    order: Sequence[str]
) -> float:
    # Compare which cluster maximally expresses each real-cluster signature
    if S_real.size == 0 or S_cand.size == 0:
        return np.nan
    top_real = np.argmax(S_real, axis=0)
    top_cand = np.argmax(S_cand, axis=0)
    return float(np.mean(top_real == top_cand))


def marker_signature_corr(
    S_real: np.ndarray,
    S_cand: np.ndarray
) -> float:
    return safe_pearson(S_real.ravel(), S_cand.ravel())


def top_marker_gene_cluster_agreement(
    pb_real: np.ndarray,
    pb_cand: np.ndarray,
    marker_sets: Dict[str, List[str]],
    genes: Sequence[str],
    order: Sequence[str]
) -> Tuple[float, pd.DataFrame]:
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    marker_union = sorted(set(sum(marker_sets.values(), [])))

    rows = []
    agree = []
    for g in marker_union:
        if g not in gene_to_idx:
            continue
        gi = gene_to_idx[g]
        top_real = int(np.argmax(pb_real[:, gi]))
        top_cand = int(np.argmax(pb_cand[:, gi]))
        ok = (top_real == top_cand)
        agree.append(ok)
        rows.append({
            "gene": g,
            "real_top_cluster": order[top_real],
            "cand_top_cluster": order[top_cand],
            "top_cluster_match": float(ok),
        })

    score = float(np.mean(agree)) if len(agree) > 0 else np.nan
    return score, pd.DataFrame(rows)


# ============================================================
# CASE-GENE SUPPORT TABLES
# ============================================================

def gene_cluster_profiles(
    adata: AnnData,
    cluster_key: str,
    genes: Sequence[str],
    order: Sequence[str]
) -> pd.DataFrame:
    pb = pseudobulk_by_cluster(adata, cluster_key, genes, order)
    df = pd.DataFrame(pb.T, index=genes, columns=order)
    return df


def compute_case_gene_table(
    real_ad: AnnData,
    candidate_dict: Dict[str, AnnData],
    cluster_key: str,
    genes: Sequence[str],
    order: Sequence[str]
) -> pd.DataFrame:
    real_prof = gene_cluster_profiles(real_ad, cluster_key, genes, order)

    rows = []
    for cand_name, cand_ad in candidate_dict.items():
        cand_prof = gene_cluster_profiles(cand_ad, cluster_key, genes, order)

        for g in genes:
            xr = real_prof.loc[g].values.astype(np.float32)
            xc = cand_prof.loc[g].values.astype(np.float32)
            corr = safe_pearson(xr, xc)

            top_r = order[int(np.argmax(xr))]
            top_c = order[int(np.argmax(xc))]
            gap_c = float(np.sort(xc)[-1] - np.sort(xc)[-2]) if len(xc) >= 2 else np.nan

            rows.append({
                "candidate": cand_name,
                "gene": g,
                "cluster_profile_corr": corr,
                "real_top_cluster": top_r,
                "candidate_top_cluster": top_c,
                "top_cluster_match": float(top_r == top_c),
                "candidate_top_margin": gap_c,
            })

    return pd.DataFrame(rows)


def auto_choose_case_genes_from_real_markers(
    real_ad: AnnData,
    cluster_key: str,
    genes: Sequence[str],
    order: Sequence[str],
    top_markers_per_cluster: int = 30,
    top_case_genes_per_cluster: int = 2,
) -> List[str]:
    marker_sets = compute_real_marker_sets(
        real_ad=real_ad,
        cluster_key=cluster_key,
        genes=genes,
        order=order,
        top_n=top_markers_per_cluster,
    )
    out = []
    for c in order:
        out.extend(marker_sets[str(c)][:top_case_genes_per_cluster])
    out = list(pd.unique(pd.Series(out).astype(str)))
    return out


def collect_per_cluster_gene_program_similarity(
    group_ids=GROUP_IDS,
    cluster_key=cluster_KEY,
    normalize_total=NORMALIZE_TOTAL,
    target_sum=TARGET_SUM,
    log1p=LOG1P,
    gene_mode="all_common",   # "all_common" or "real_markers"
    top_markers_per_cluster=TOP_MARKERS_PER_CLUSTER,
):
    """
    Collect per-cluster gene-program similarity across all patients.

    Output columns:
      - group_id
      - candidate
      - cluster
      - cluster_gene_corr
      - n_real_spots
      - n_candidate_spots
      - n_genes_used
    """
    rows = []

    for gid in group_ids:
        print(f"[INFO] collecting per-cluster similarity for group {gid}")
        data = load_group_data(gid)

        real_ad = preprocess_adata(data["Real"], normalize_total, target_sum, log1p)
        axis_ad = preprocess_adata(data["AXIS"], normalize_total, target_sum, log1p)
        spatialz_ad = preprocess_adata(data["SpatialZ"], normalize_total, target_sum, log1p)

        ad_list = [real_ad, axis_ad, spatialz_ad]
        common_genes = common_gene_intersection(ad_list)
        if len(common_genes) == 0:
            print(f"[WARN] group {gid}: no common genes, skip")
            continue

        real_ad = subset_to_genes(real_ad, common_genes)
        axis_ad = subset_to_genes(axis_ad, common_genes)
        spatialz_ad = subset_to_genes(spatialz_ad, common_genes)

        cluster_order = canonical_cluster_order(get_cluster_labels(real_ad, cluster_key))

        if gene_mode == "real_markers":
            marker_sets = compute_real_marker_sets(
                real_ad=real_ad,
                cluster_key=cluster_key,
                genes=common_genes,
                order=cluster_order,
                top_n=top_markers_per_cluster,
            )
            genes_used = sorted(set(sum(marker_sets.values(), [])))
            if len(genes_used) == 0:
                genes_used = common_genes
        else:
            genes_used = common_genes

        pb_real = pseudobulk_by_cluster(real_ad, cluster_key, genes_used, cluster_order)
        real_labels = get_cluster_labels(real_ad, cluster_key)

        candidate_dict = {
            "SpatialZ": spatialz_ad,
            "AXIS": axis_ad,
        }

        for cand_name, cand_ad in candidate_dict.items():
            pb_cand = pseudobulk_by_cluster(cand_ad, cluster_key, genes_used, cluster_order)
            cand_labels = get_cluster_labels(cand_ad, cluster_key)

            for i, cl in enumerate(cluster_order):
                corr = safe_pearson(pb_real[i], pb_cand[i])

                rows.append({
                    "group_id": gid,
                    "candidate": cand_name,
                    "cluster": str(cl),
                    "cluster_gene_corr": corr,
                    "n_real_spots": int(np.sum(real_labels == str(cl))),
                    "n_candidate_spots": int(np.sum(cand_labels == str(cl))),
                    "n_genes_used": int(len(genes_used)),
                })

    df = pd.DataFrame(rows)
    return df

# ============================================================
# MAIN METRIC EVALUATION
# ============================================================

def evaluate_group(
    group_id: int,
    cluster_key: str = cluster_KEY,
    coord_key: str = COORD_KEY,
    normalize_total: bool = NORMALIZE_TOTAL,
    target_sum: float = TARGET_SUM,
    log1p: bool = LOG1P,
    top_markers_per_cluster: int = TOP_MARKERS_PER_CLUSTER,
    knn_adj: int = KNN_ADJ,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    data = load_group_data(group_id)

    real_ad = data["Real"]
    axis_ad = data["AXIS"]
    spatialz_ad = data["SpatialZ"]

    # preprocess
    ad_list = [real_ad, axis_ad, spatialz_ad]
    ad_list = [preprocess_adata(ad, normalize_total, target_sum, log1p) for ad in ad_list]
    real_ad, axis_ad, spatialz_ad = ad_list

    common_genes = common_gene_intersection(ad_list)
    if len(common_genes) == 0:
        raise ValueError(f"group {group_id}: no common genes found")

    real_ad = subset_to_genes(real_ad, common_genes)
    axis_ad = subset_to_genes(axis_ad, common_genes)
    spatialz_ad = subset_to_genes(spatialz_ad, common_genes)

    # cluster order from REAL
    cluster_order = canonical_cluster_order(get_cluster_labels(real_ad, cluster_key))

    # published real markers
    marker_sets = compute_real_marker_sets(
        real_ad=real_ad,
        cluster_key=cluster_key,
        genes=common_genes,
        order=cluster_order,
        top_n=top_markers_per_cluster,
    )

    candidate_dict = {
        "SpatialZ": spatialz_ad,
        "AXIS": axis_ad,
    }

    summary_rows = []
    per_cluster_rows = []

    # real references
    comp_real = cluster_composition(real_ad, cluster_key, cluster_order)
    adj_real = cluster_adjacency_matrix(real_ad, cluster_key, cluster_order, coord_key, knn_adj)
    pb_real = pseudobulk_by_cluster(real_ad, cluster_key, common_genes, cluster_order)
    S_real = signature_score_matrix(real_ad, cluster_key, marker_sets, cluster_order)

    for cand_name, cand_ad in candidate_dict.items():
        comp_c = cluster_composition(cand_ad, cluster_key, cluster_order)
        adj_c = cluster_adjacency_matrix(cand_ad, cluster_key, cluster_order, coord_key, knn_adj)
        pb_c = pseudobulk_by_cluster(cand_ad, cluster_key, common_genes, cluster_order)
        S_c = signature_score_matrix(cand_ad, cluster_key, marker_sets, cluster_order)

        comp_jsd = js_divergence(comp_real, comp_c)
        comp_l1 = float(np.abs(comp_real - comp_c).sum())

        adj_corr = safe_pearson(adj_real.ravel(), adj_c.ravel())
        adj_l1 = float(np.abs(adj_real - adj_c).sum())

        cent_corr = centroid_distance_correlation(real_ad, cand_ad, cluster_key, cluster_order, coord_key)

        cluster_gene_df = per_cluster_gene_correlations(pb_real, pb_c, cluster_order)
        cluster_gene_df["group_id"] = group_id
        cluster_gene_df["candidate"] = cand_name
        per_cluster_rows.append(cluster_gene_df)

        mean_cluster_gene_corr = float(np.nanmean(cluster_gene_df["cluster_gene_corr"].values))

        sig_corr = marker_signature_corr(S_real, S_c)
        top_sig_agree = top_signature_agreement(S_real, S_c, cluster_order)

        marker_top_agree, marker_top_df = top_marker_gene_cluster_agreement(
            pb_real=pb_real,
            pb_cand=pb_c,
            marker_sets=marker_sets,
            genes=common_genes,
            order=cluster_order,
        )

        summary_rows.append({
            "group_id": group_id,
            "candidate": cand_name,
            "cluster_comp_jsd": comp_jsd,
            "cluster_comp_l1": comp_l1,
            "cluster_adj_corr": adj_corr,
            "cluster_adj_l1": adj_l1,
            "cluster_centroid_corr": cent_corr,
            "mean_cluster_gene_corr": mean_cluster_gene_corr,
            "marker_signature_corr": sig_corr,
            "top_signature_agreement": top_sig_agree,
            "top_marker_gene_cluster_agreement": marker_top_agree,
            "n_clusters_real": len(cluster_order),
            "n_common_genes": len(common_genes),
        })

    summary_df = pd.DataFrame(summary_rows)
    per_cluster_df = pd.concat(per_cluster_rows, axis=0, ignore_index=True)

    # case-study gene table
    if CASE_GENES is None:
        case_genes = auto_choose_case_genes_from_real_markers(
            real_ad=real_ad,
            cluster_key=cluster_key,
            genes=common_genes,
            order=cluster_order,
            top_markers_per_cluster=top_markers_per_cluster,
            top_case_genes_per_cluster=TOP_CASE_GENES_PER_CLUSTER,
        )
    else:
        case_genes = [g for g in CASE_GENES if g in common_genes]

    case_gene_df = compute_case_gene_table(
        real_ad=real_ad,
        candidate_dict=candidate_dict,
        cluster_key=cluster_key,
        genes=case_genes,
        order=cluster_order,
    )
    case_gene_df["group_id"] = group_id

    out_bundle = {
        "real_ad": real_ad,
        "axis_ad": axis_ad,
        "spatialz_ad": spatialz_ad,
        "anchor_neg_ad": preprocess_adata(data["Anchor A (-10 um)"], normalize_total, target_sum, log1p),
        "anchor_pos_ad": preprocess_adata(data["Anchor B (+10 um)"], normalize_total, target_sum, log1p),
        "cluster_order": cluster_order,
        "common_genes": common_genes,
        "marker_sets": marker_sets,
        "case_genes": case_genes,
    }

    return summary_df, per_cluster_df, case_gene_df, out_bundle


def run_main_evaluation(
    group_ids: Sequence[int] = GROUP_IDS,
    outdir: str = OUTDIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, Dict]]:
    ensure_dir(outdir)

    summary_all = []
    per_cluster_all = []
    case_gene_all = []
    bundle_dict = {}

    for gid in group_ids:
        print(f"[INFO] evaluating group {gid}")
        s_df, pc_df, cg_df, bundle = evaluate_group(gid)
        summary_all.append(s_df)
        per_cluster_all.append(pc_df)
        case_gene_all.append(cg_df)
        bundle_dict[gid] = bundle

    summary_df = pd.concat(summary_all, axis=0, ignore_index=True)
    per_cluster_df = pd.concat(per_cluster_all, axis=0, ignore_index=True)
    case_gene_df = pd.concat(case_gene_all, axis=0, ignore_index=True)

    summary_df.to_csv(os.path.join(outdir, "candidate_summary_metrics.csv"), index=False)
    per_cluster_df.to_csv(os.path.join(outdir, "per_cluster_gene_metrics.csv"), index=False)
    case_gene_df.to_csv(os.path.join(outdir, "case_gene_metrics.csv"), index=False)

    print(f"[OK] saved: {os.path.join(outdir, 'candidate_summary_metrics.csv')}")
    print(f"[OK] saved: {os.path.join(outdir, 'per_cluster_gene_metrics.csv')}")
    print(f"[OK] saved: {os.path.join(outdir, 'case_gene_metrics.csv')}")

    return summary_df, per_cluster_df, case_gene_df, bundle_dict


# ============================================================
# PLOTTING: SUMMARY METRIC FIGURES
# ============================================================

def get_candidate_order(df: pd.DataFrame) -> List[str]:
    pref = ["SpatialZ", "AXIS"]
    order = [x for x in pref if x in df["candidate"].unique()]
    order += [x for x in df["candidate"].unique() if x not in order]
    return order


def get_candidate_palette(order: Sequence[str]) -> Dict[str, str]:
    palette = {
        "SpatialZ": "#8da0cb",
        "AXIS": "#e41a1c",
    }
    out = {}
    fallback = sns.color_palette("Set2", n_colors=len(order))
    for i, k in enumerate(order):
        out[k] = palette.get(k, fallback[i])
    return out


def add_group_pair_lines(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    order: Sequence[str],
    color: str = "0.75",
    lw: float = 1.0,
    alpha: float = 0.8,
):
    x_pos = {k: i for i, k in enumerate(order)}
    for gid, sub in df.groupby(group_col):
        sub = sub.dropna(subset=[y_col])
        if len(sub) < 2:
            continue
        sub = sub.set_index(x_col).reindex(order).dropna(subset=[y_col], how="all")
        xs = []
        ys = []
        for cand in order:
            if cand in sub.index:
                val = sub.loc[cand, y_col]
                if np.ndim(val) > 0:
                    # duplicated rows not expected, take first
                    val = np.asarray(val).ravel()[0]
                if pd.notnull(val):
                    xs.append(x_pos[cand])
                    ys.append(float(val))
        if len(xs) >= 2:
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=1)


def plot_summary_metric_panels(
    summary_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    set_cell_nature_plot_style()

    order = get_candidate_order(summary_df)
    palette = get_candidate_palette(order)

    specs = [
        ("cluster_comp_l1", "Composition error (L1)", False),
        ("cluster_adj_corr", "Adjacency similarity", True),
        # ("mean_cluster_gene_corr", "Cluster gene-program fidelity", True),
    ]

    nrows = len(specs)
    fig, axes = plt.subplots(nrows, 1, figsize=(3.4, 6.2))
    if nrows == 1:
        axes = [axes]

    for ax, (metric, ylabel, higher_better) in zip(axes, specs):
        dfm = summary_df.copy()

        # paired patient lines first
        add_group_pair_lines(
            ax=ax,
            df=dfm,
            x_col="candidate",
            y_col=metric,
            group_col="group_id",
            order=order,
            color="0.82",
            lw=1.0,
            alpha=0.95,
        )

        sns.boxplot(
            data=dfm,
            x="candidate",
            y=metric,
            order=order,
            palette=palette,
            showfliers=False,
            width=0.42,
            linewidth=1.1,
            ax=ax,
            zorder=2,
        )

        sns.stripplot(
            data=dfm,
            x="candidate",
            y=metric,
            order=order,
            palette=palette,
            size=6.0,
            alpha=0.98,
            linewidth=0.45,
            edgecolor="black",
            jitter=0.08,
            ax=ax,
            zorder=3,
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(ylabel, fontsize=14, color="black", pad=4)

        ax.tick_params(axis="x", labelsize=10, pad=1)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.18)

        # tighten horizontal empty space around the two boxes
        ax.set_xlim(-0.32, len(order) - 0.68 + 1e-8)

        sns.despine(ax=ax)

    fig.subplots_adjust(
        left=0.18,
        right=0.96,
        top=0.96,
        bottom=0.08,
        hspace=0.28,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()



def plot_per_cluster_gene_distribution(
    per_cluster_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    order = get_candidate_order(per_cluster_df)
    palette = get_candidate_palette(order)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.boxplot(
        data=per_cluster_df,
        x="candidate",
        y="cluster_gene_corr",
        order=order,
        palette=palette,
        showfliers=False,
        width=0.55,
        linewidth=1.1,
        ax=ax,
    )
    sns.stripplot(
        data=per_cluster_df,
        x="candidate",
        y="cluster_gene_corr",
        order=order,
        hue="cluster",
        dodge=False,
        palette="tab10",
        size=5,
        alpha=0.85,
        ax=ax,
    )

    if ax.get_legend() is not None:
        ax.legend(
            title="Cluster",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
            fontsize=9,
            title_fontsize=12,
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Per-cluster gene-program correlation", fontsize=14)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()


# ============================================================
# PLOTTING: CASE STUDY FIGURE
# ============================================================

def plot_scatter_gene(
    ax,
    adata: AnnData,
    gene: str,
    coord_key: str,
    cmap: str,
    norm,
    s: float = 16,
    title: Optional[str] = None,
    dashed: bool = False,
):
    coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    vals = extract_matrix(adata, [gene]).ravel()

    sca = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=vals, cmap=cmap, norm=norm,
        s=s, edgecolors="none", rasterized=True
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight="bold")

    if dashed:
        for spine in ax.spines.values():
            spine.set_linestyle((0, (5, 3)))
            spine.set_linewidth(1.0)
    return sca


def plot_scatter_clusters(
    ax,
    adata: AnnData,
    cluster_key: str,
    coord_key: str,
    cluster_order: Sequence[str],
    cluster_colors: Dict[str, Tuple[float, float, float, float]],
    s: float = 16,
    title: Optional[str] = None,
    dashed: bool = False,
):
    coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    labels = get_cluster_labels(adata, cluster_key)
    rgba = np.array([cluster_colors[str(x)] for x in labels])

    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=rgba, s=s, edgecolors="none", rasterized=True
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight="bold")

    if dashed:
        for spine in ax.spines.values():
            spine.set_linestyle((0, (5, 3)))
            spine.set_linewidth(1.0)


def plot_case_study_figure(
    group_id: int,
    bundle: Dict,
    case_gene_df: pd.DataFrame,
    save_path: Optional[str] = None,
    gene: str = "KRT1",
    cluster_key: str = cluster_KEY,
    coord_key: str = COORD_KEY,
):
    set_cell_nature_plot_style()

    real_ad = bundle["real_ad"]
    axis_ad = bundle["axis_ad"]
    spatialz_ad = bundle["spatialz_ad"]
    anchor_neg_ad = subset_to_genes(bundle["anchor_neg_ad"], bundle["common_genes"])
    anchor_pos_ad = subset_to_genes(bundle["anchor_pos_ad"], bundle["common_genes"])
    cluster_order = bundle["cluster_order"]

    cluster_colors = discrete_cmap_from_order(cluster_order, "tab10")

    vals_all = []
    for ad in [anchor_neg_ad, anchor_pos_ad, real_ad, spatialz_ad, axis_ad]:
        vals_all.append(extract_matrix(ad, [gene]).ravel())
    vals_all = np.concatenate(vals_all)
    vmin = np.percentile(vals_all, 1)
    vmax = np.percentile(vals_all, 99)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 5, figsize=(15.2, 6.0))

    top_titles = [
        "Anchor A\n(-10 μm)",
        "Anchor B\n(+10 μm)",
        "Real slice",
        "SpatialZ",
        "AXIS",
    ]
    top_ads = [anchor_neg_ad, anchor_pos_ad, real_ad, spatialz_ad, axis_ad]
    dashed_cols = [False, False, False, True, True]

    # gene row
    last_sca = None
    for j, (ad, ttl, dashed) in enumerate(zip(top_ads, top_titles, dashed_cols)):
        last_sca = plot_scatter_gene(
            ax=axes[0, j],
            adata=ad,
            gene=gene,
            coord_key=coord_key,
            cmap="Reds",
            norm=norm,
            s=18,
            title=ttl,
            dashed=dashed,
        )

    # cluster row
    for j, (ad, dashed) in enumerate(zip(top_ads, dashed_cols)):
        plot_scatter_clusters(
            ax=axes[1, j],
            adata=ad,
            cluster_key=cluster_key,
            coord_key=coord_key,
            cluster_order=cluster_order,
            cluster_colors=cluster_colors,
            s=18,
            title=None,
            dashed=dashed,
        )

    axes[0, 0].set_ylabel(
        f"{gene}\nreads",
        fontsize=15,
        fontweight="bold",
        rotation=0,
        labelpad=34,
        va="center",
    )
    axes[1, 0].set_ylabel(
        "Spot\nclusters",
        fontsize=15,
        fontweight="bold",
        rotation=0,
        labelpad=34,
        va="center",
    )

    # direct metric annotation on candidate image titles
    for col, cand_name in [(3, "SpatialZ"), (4, "AXIS")]:
        sub = case_gene_df[
            (case_gene_df["group_id"] == group_id) &
            (case_gene_df["candidate"] == cand_name) &
            (case_gene_df["gene"] == gene)
        ]
        if len(sub) == 1:
            rr = sub.iloc[0]
            axes[0, col].set_title(
                f"{cand_name}\n"
                f"profile r = {rr['cluster_profile_corr']:.2f}\n"
                f"top cluster = {int(rr['top_cluster_match'])}",
                fontsize=11.5,
                fontweight="bold",
            )

    cbar = fig.colorbar(last_sca, ax=axes[0, :], shrink=0.82, pad=0.01)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Group {group_id}: case-study comparison",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()
from matplotlib.patches import Patch

def plot_global_cluster_gene_program_similarity(
    df,
    save_path=None,
    method_order=("SpatialZ", "AXIS"),
    method_palette=None,
    cluster_order=None,
    figsize=(8.5, 4.6),
    title="Gene-program similarity across clusters",
    ylabel="Gene-program similarity",
):
    """
    Global cluster-wise boxplot across patients.

    x-axis = cluster
    hue = method
    y-axis = cluster_gene_corr

    Each point = one patient-cluster-method value.
    Gray lines connect paired SpatialZ and AXIS values for the same patient and cluster.
    """
    df = df.copy()

    if method_palette is None:
        method_palette = {
            "SpatialZ": "#7C8DA6",
            "AXIS": "#C4473A",
        }

    if cluster_order is None:
        cluster_order = canonical_cluster_order(df["cluster"].astype(str).values)

    df["cluster"] = df["cluster"].astype(str)
    df = df[df["candidate"].isin(method_order)].copy()

    fig, ax = plt.subplots(figsize=figsize)

    # paired lines first
    offset = 0.20
    x_map = {cl: i for i, cl in enumerate(cluster_order)}

    for cl in cluster_order:
        sub_cl = df[df["cluster"] == str(cl)]
        for gid, sub_g in sub_cl.groupby("group_id"):
            sub_g = sub_g.set_index("candidate")
            if all(m in sub_g.index for m in method_order):
                y1 = float(sub_g.loc[method_order[0], "cluster_gene_corr"])
                y2 = float(sub_g.loc[method_order[1], "cluster_gene_corr"])
                x0 = x_map[cl]
                ax.plot(
                    [x0 - offset, x0 + offset],
                    [y1, y2],
                    color="0.82",
                    lw=1.1,
                    alpha=0.95,
                    zorder=1,
                )

    sns.boxplot(
        data=df,
        x="cluster",
        y="cluster_gene_corr",
        hue="candidate",
        order=cluster_order,
        hue_order=method_order,
        palette=method_palette,
        showfliers=False,
        width=0.72,
        linewidth=1.1,
        ax=ax,
        zorder=2,
    )

    sns.stripplot(
        data=df,
        x="cluster",
        y="cluster_gene_corr",
        hue="candidate",
        order=cluster_order,
        hue_order=method_order,
        palette=method_palette,
        dodge=True,
        size=6,
        alpha=0.95,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        zorder=3,
    )

    # clean duplicated legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    handles = [
        Patch(facecolor=method_palette[m], edgecolor="black", label=m)
        for m in method_order
    ]
    ax.legend(
        handles=handles,
        title="Method",
        title_fontsize=12,
        fontsize=11,
        frameon=False,
        loc="best",
    )

    ax.set_xlabel("Published cluster", fontsize=14)
    ax.set_ylabel(" ")
    ax.set_title(title, fontsize=15)
    ax.tick_params(axis="x", labelsize=11, rotation=0)
    ax.grid(axis="y", alpha=0.20)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()
# ============================================================
# OPTIONAL: PLOT CASE-GENE TABLE SUMMARY
# ============================================================

def plot_case_gene_summary(
    case_gene_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    sns.boxplot(
        data=case_gene_df,
        x="candidate",
        y="cluster_profile_corr",
        palette=get_candidate_palette(get_candidate_order(case_gene_df)),
        ax=axes[0],
        showfliers=False,
    )
    sns.stripplot(
        data=case_gene_df,
        x="candidate",
        y="cluster_profile_corr",
        color="black",
        alpha=0.45,
        size=4,
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Gene cluster-profile correlation", fontsize=12)
    axes[0].tick_params(axis="x", labelsize=11)
    axes[0].grid(axis="y", alpha=0.2)

    sns.boxplot(
        data=case_gene_df,
        x="candidate",
        y="top_cluster_match",
        palette=get_candidate_palette(get_candidate_order(case_gene_df)),
        ax=axes[1],
        showfliers=False,
    )
    sns.stripplot(
        data=case_gene_df,
        x="candidate",
        y="top_cluster_match",
        color="black",
        alpha=0.45,
        size=4,
        ax=axes[1],
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Top-cluster agreement", fontsize=12)
    axes[1].tick_params(axis="x", labelsize=11)
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Case-study marker-gene summary", fontsize=16, fontweight="bold", y=1.03)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ensure_dir(OUTDIR)

    # summary_df, per_cluster_df, case_gene_df, bundle_dict = run_main_evaluation(
    #     group_ids=GROUP_IDS,
    #     outdir=OUTDIR,
    # )
    # per_cluster_global_df = collect_per_cluster_gene_program_similarity(
    #     group_ids=GROUP_IDS,
    #     cluster_key=cluster_KEY,
    #     gene_mode="all_common",  # or "real_markers"
    # )
    #
    # per_cluster_global_path = os.path.join(OUTDIR, "per_cluster_gene_program_similarity.csv")
    # per_cluster_global_df.to_csv(per_cluster_global_path, index=False)
    # print(f"[OK] saved: {per_cluster_global_path}")


    summary_df = pd.read_csv(os.path.join(OUTDIR, "candidate_summary_metrics.csv"))
    per_cluster_df = pd.read_csv(os.path.join(OUTDIR, "per_cluster_gene_metrics.csv"))
    per_cluster_global_df = pd.read_csv(os.path.join(OUTDIR, "per_cluster_gene_program_similarity.csv"))

    plot_summary_metric_panels(
        summary_df=summary_df,
        save_path=os.path.join(OUTDIR, "summary_metric_panels_main.png"),
    )
    #
    # plot_per_cluster_gene_distribution(
    #     per_cluster_df=per_cluster_df,
    #     # save_path=os.path.join(OUTDIR, "per_cluster_gene_distribution.png"),
    # )

    plot_global_cluster_gene_program_similarity(
        df=per_cluster_global_df,
        save_path=os.path.join(OUTDIR, "global_cluster_gene_program_similarity.png"),
    )

    # if CASE_GROUP in bundle_dict:
    #     bundle = bundle_dict[CASE_GROUP]
    #
    #     case_genes_this = bundle["case_genes"] if CASE_GENES is None else CASE_GENES
    #     case_genes_this = [g for g in case_genes_this if g in bundle["common_genes"]]
    #     if len(case_genes_this) == 0:
    #         case_genes_this = [bundle["case_genes"][0]]
    #
    #     plot_case_study_figure(
    #         group_id=CASE_GROUP,
    #         bundle=bundle,
    #         case_gene_df=case_gene_df,
    #         save_path=os.path.join(OUTDIR, f"group{CASE_GROUP}_case_study_{case_genes_this[0]}.png"),
    #         gene=case_genes_this[0],
    #     )

    print("[DONE] SCC plotting update finished.")