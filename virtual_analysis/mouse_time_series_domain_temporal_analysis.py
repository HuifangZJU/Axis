#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-series domain-restricted temporal gene dynamics from raw h5ad files.

Proofread interpretation of the analysis goal
---------------------------------------------
Your earlier script reads precomputed pairwise differential h5ad files together
with external ranking CSV files, then plots top genes, spatial small multiples,
and summary heatmaps. In contrast, this script starts directly from the original
three time-point h5ad files and recomputes the temporal ranking and pattern
analysis in one place.

Important methodological note
-----------------------------
Because this design appears to have one section per time point, this is best
treated as *domain-restricted temporal differential ranking* rather than
replicate-aware classical DE. The code therefore uses:

1. Per-domain omnibus Kruskal-Wallis statistics across cells from the 3 time points
   (widely recognized nonparametric DE-style screening statistic).
2. Effect-size summaries from domain-level mean trajectories:
   - temporal_range
   - max_abs_interval_delta
   - temporal_sd
   - slope and Spearman trend
3. Pattern clustering of top genes inside each domain using their 3-point mean
   trajectory shape.

This makes the ranking more biologically stable than relying only on p-values
when the design lacks replicate sections.

Main outputs
------------
- per-domain temporal statistics table
- ranked top genes per domain
- cluster assignments and centroids per domain
- top-gene summary panels
- per-domain temporal pattern heatmaps
- per-domain cluster centroid plots
- interval summary heatmap for the two temporal intervals
- spatial module-score maps for representative temporal clusters

Fill the three h5ad paths below and run.
"""

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kruskal, spearmanr
from sklearn.cluster import KMeans
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


# ============================================================
# CONFIG
# ============================================================
ROOT = "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial2.0/analysis/"
OUTDIR = os.path.join(ROOT, "DE_analysis")
os.makedirs(OUTDIR, exist_ok=True)

TIMEPOINTS = [
    {"label": "2p5",  "time": 2.5,  "path": "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial2.0/analysis/virtual_AD_time_2.500.h5ad"},
    {"label": "5p7",  "time": 5.7,  "path": "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial2.0/analysis/virtual_AD_time_5.700.h5ad"},
    {"label": "17p9", "time": 17.9, "path": "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial2.0/analysis/virtual_AD_time_17.900.h5ad"},
]

COORD_KEY = "spatial"
DOMAIN_KEY = "axis_joint_domain"

# domain groups requested by the user
HIPPOCAMPUS_DOMAINS = ["8", "11"]
CORTICAL_LAYER_DOMAINS = ["1", "4", "6", "10"]
SELECTED_DOMAINS = HIPPOCAMPUS_DOMAINS + CORTICAL_LAYER_DOMAINS

# plotting / ranking controls
FIG_DPI = 300
POINT_SIZE_BG = 0.8
POINT_SIZE_FG = 2.0
BG_ALPHA = 0.06
FG_ALPHA = 0.92

TOP_GENES_PER_DOMAIN_BAR = 12
TOP_GENES_PER_DOMAIN_CLUSTER = 60
N_CLUSTERS_PER_DOMAIN = 4
N_SPATIAL_CLUSTERS_PER_DOMAIN = 2
MIN_CELLS_PER_DOMAIN_PER_TIMEPOINT = 30
MODULE_SCORE_MIN_GENES = 3

INVERT_X = False
INVERT_Y = False

# if data are raw counts and you want log1p before temporal analysis, set True
LOG1P_BEFORE_ANALYSIS = False

# optional manual override for spatial cluster figures
MANUAL_SPATIAL_DOMAINS: Optional[List[str]] = None

# optional manual gene case studies (in addition to cluster modules)
MANUAL_CASE_GENES: Optional[List[str]] = ["Gfap", "Cst3", "Arc", "Apoe"]


# ============================================================
# STYLE
# ============================================================
def set_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.16,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# HELPERS
# ============================================================
@dataclass
class TimepointData:
    label: str
    time: float
    adata: sc.AnnData
    X: np.ndarray
    coords: np.ndarray
    domains: np.ndarray
    genes: np.ndarray


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_name(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    good = np.isfinite(pvals)
    if good.sum() == 0:
        return out

    p = pvals[good]
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    tmp = np.empty_like(q)
    tmp[order] = q
    out[good] = tmp
    return out


def apply_axis_orientation(ax):
    if INVERT_X:
        ax.invert_xaxis()
    if INVERT_Y:
        ax.invert_yaxis()


def row_zscore(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    mu = np.nanmean(mat, axis=1, keepdims=True)
    sd = np.nanstd(mat, axis=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    out = (mat - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out


def robust_sym_vmax(vals, q=99.0):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    vmax = np.percentile(np.abs(vals), q)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.max(np.abs(vals))
    return max(float(vmax), 1e-6)


def build_domain_palette(domains: Sequence[str]) -> Dict[str, str]:
    base = plt.get_cmap("tab20")
    uniq = [str(d) for d in domains]
    uniq = list(dict.fromkeys(uniq))
    return {d: base(i % 20) for i, d in enumerate(uniq)}


PATTERN_COLORS = {
    "Increasing": "#D55E00",
    "Decreasing": "#0072B2",
    "Peak@5.7m": "#009E73",
    "Trough@5.7m": "#CC79A7",
    "Mixed": "#7F7F7F",
}


def infer_pattern_name(centroid: np.ndarray) -> str:
    """
    centroid is a 3-point z-scored trajectory: [2.5, 5.7, 17.9]
    """
    c = np.asarray(centroid, dtype=float)
    a, b, c3 = c[0], c[1], c[2]
    amp = float(np.max(c) - np.min(c))
    if amp < 0.35:
        return "Mixed"

    if (c3 > b > a) and (c3 - a > 0.5):
        return "Increasing"
    if (c3 < b < a) and (a - c3 > 0.5):
        return "Decreasing"
    if (b >= a) and (b >= c3) and ((b - max(a, c3)) > 0.35):
        return "Peak@5.7m"
    if (b <= a) and (b <= c3) and ((min(a, c3) - b) > 0.35):
        return "Trough@5.7m"
    return "Mixed"


def read_h5ad(path: str) -> sc.AnnData:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return sc.read_h5ad(path)


def get_coords(ad: sc.AnnData, coord_key: str = COORD_KEY) -> np.ndarray:
    if coord_key not in ad.obsm:
        raise KeyError(f"{coord_key!r} not found in ad.obsm")
    coords = np.asarray(ad.obsm[coord_key], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coordinate array shape: {coords.shape}")
    return coords[:, :2]


def get_domain_labels(ad: sc.AnnData, domain_key: str = DOMAIN_KEY) -> np.ndarray:
    if domain_key not in ad.obs:
        raise KeyError(f"{domain_key!r} not found in ad.obs")
    return ad.obs[domain_key].astype(str).to_numpy()


def make_dense_X(ad: sc.AnnData) -> np.ndarray:
    X = ad.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if LOG1P_BEFORE_ANALYSIS:
        X = np.log1p(X)
    return X


def intersect_genes(adatas: Sequence[sc.AnnData]) -> List[str]:
    common = set(adatas[0].var_names.astype(str))
    for ad in adatas[1:]:
        common &= set(ad.var_names.astype(str))
    genes = sorted(common)
    if len(genes) == 0:
        raise ValueError("No common genes across the 3 h5ad files.")
    return genes


def load_timepoint_data() -> List[TimepointData]:
    ads = [read_h5ad(tp["path"]) for tp in TIMEPOINTS]
    common_genes = intersect_genes(ads)

    data = []
    for meta, ad in zip(TIMEPOINTS, ads):
        ad = ad[:, common_genes].copy()
        coords = get_coords(ad)
        domains = get_domain_labels(ad)
        X = make_dense_X(ad)
        data.append(TimepointData(
            label=str(meta["label"]),
            time=float(meta["time"]),
            adata=ad,
            X=X,
            coords=coords,
            domains=domains,
            genes=np.asarray(common_genes, dtype=object),
        ))
    return data


# ============================================================
# TEMPORAL DOMAIN STATISTICS
# ============================================================
def compute_domain_gene_stats(data: List[TimepointData], selected_domains: Optional[Sequence[str]] = None) -> pd.DataFrame:
    times = [tp.time for tp in data]
    genes = data[0].genes
    all_domains = sorted(set(np.concatenate([tp.domains for tp in data]).astype(str).tolist()))

    if selected_domains is None:
        use_domains = all_domains
    else:
        use_domains = [str(d) for d in selected_domains if str(d) in all_domains]

    rows = []

    for domain in use_domains:
        masks = []
        valid_domain = True
        for tp in data:
            m = tp.domains == str(domain)
            masks.append(m)
            if int(m.sum()) < MIN_CELLS_PER_DOMAIN_PER_TIMEPOINT:
                valid_domain = False

        if not valid_domain:
            print(f"[WARN] Skip domain {domain}: fewer than {MIN_CELLS_PER_DOMAIN_PER_TIMEPOINT} cells in at least one timepoint.")
            continue

        mats = [tp.X[m, :] for tp, m in zip(data, masks)]
        means = [np.asarray(mat.mean(axis=0)).ravel() for mat in mats]
        pct_expr = [np.mean(mat > 0, axis=0) for mat in mats]
        n_cells = [mat.shape[0] for mat in mats]

        kw_stat = np.full(len(genes), np.nan, dtype=float)
        kw_pval = np.full(len(genes), np.nan, dtype=float)
        rho = np.full(len(genes), np.nan, dtype=float)

        for j in range(len(genes)):
            arrs = [mat[:, j] for mat in mats]
            try:
                if any(len(a) == 0 for a in arrs):
                    pass
                elif np.allclose(np.concatenate(arrs), np.concatenate(arrs)[0]):
                    pass
                else:
                    stat, p = kruskal(*arrs)
                    kw_stat[j] = float(stat)
                    kw_pval[j] = float(p)
            except Exception:
                pass

            traj = np.array([means[0][j], means[1][j], means[2][j]], dtype=float)
            if np.std(traj) > 1e-12:
                try:
                    rho[j] = float(spearmanr(times, traj)[0])
                except Exception:
                    pass

        kw_fdr = bh_fdr(kw_pval)

        mean_stack = np.vstack(means).T  # genes x timepoints
        temporal_range = np.max(mean_stack, axis=1) - np.min(mean_stack, axis=1)
        temporal_sd = np.std(mean_stack, axis=1)
        slope = np.polyfit(np.asarray(times, dtype=float), mean_stack.T, deg=1)[0]
        delta_12 = mean_stack[:, 1] - mean_stack[:, 0]
        delta_23 = mean_stack[:, 2] - mean_stack[:, 1]
        delta_13 = mean_stack[:, 2] - mean_stack[:, 0]
        max_abs_interval_delta = np.maximum(np.abs(delta_12), np.abs(delta_23))

        rank_kw = pd.Series(-np.nan_to_num(kw_stat, nan=-np.inf)).rank(method="average", ascending=True).to_numpy()
        rank_range = pd.Series(-temporal_range).rank(method="average", ascending=True).to_numpy()
        rank_interval = pd.Series(-max_abs_interval_delta).rank(method="average", ascending=True).to_numpy()
        rank_sd = pd.Series(-temporal_sd).rank(method="average", ascending=True).to_numpy()
        priority_score = (rank_kw + rank_range + rank_interval + rank_sd) / 4.0

        for j, gene in enumerate(genes):
            rows.append({
                "domain": str(domain),
                "gene": str(gene),
                "n_cells_2p5": int(n_cells[0]),
                "n_cells_5p7": int(n_cells[1]),
                "n_cells_17p9": int(n_cells[2]),
                "mean_2p5": float(means[0][j]),
                "mean_5p7": float(means[1][j]),
                "mean_17p9": float(means[2][j]),
                "pct_expr_2p5": float(pct_expr[0][j]),
                "pct_expr_5p7": float(pct_expr[1][j]),
                "pct_expr_17p9": float(pct_expr[2][j]),
                "delta_5p7_minus_2p5": float(delta_12[j]),
                "delta_17p9_minus_5p7": float(delta_23[j]),
                "delta_17p9_minus_2p5": float(delta_13[j]),
                "temporal_range": float(temporal_range[j]),
                "temporal_sd": float(temporal_sd[j]),
                "max_abs_interval_delta": float(max_abs_interval_delta[j]),
                "slope": float(slope[j]),
                "spearman_rho": float(rho[j]) if np.isfinite(rho[j]) else np.nan,
                "kw_stat": float(kw_stat[j]) if np.isfinite(kw_stat[j]) else np.nan,
                "kw_pval": float(kw_pval[j]) if np.isfinite(kw_pval[j]) else np.nan,
                "kw_fdr": float(kw_fdr[j]) if np.isfinite(kw_fdr[j]) else np.nan,
                "priority_score": float(priority_score[j]),
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("No valid domain-level statistics computed.")
    return df


def sort_domain_genes(stats_df: pd.DataFrame) -> pd.DataFrame:
    return stats_df.sort_values(
        ["domain", "priority_score", "kw_stat", "temporal_range", "max_abs_interval_delta"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)


# ============================================================
# CLUSTER TEMPORAL PATTERNS WITHIN DOMAIN
# ============================================================
def cluster_domain_patterns(stats_df: pd.DataFrame,
                            top_n_genes: int = TOP_GENES_PER_DOMAIN_CLUSTER,
                            n_clusters: int = N_CLUSTERS_PER_DOMAIN) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cluster_rows = []
    centroid_rows = []

    for domain in sorted(stats_df["domain"].astype(str).unique().tolist()):
        dfd = stats_df[stats_df["domain"] == domain].copy()
        dfd = dfd.sort_values(["priority_score", "kw_stat", "temporal_range"], ascending=[True, False, False]).head(top_n_genes)

        if len(dfd) == 0:
            continue

        mat = dfd[["mean_2p5", "mean_5p7", "mean_17p9"]].to_numpy(dtype=float)
        mat_z = row_zscore(mat)

        k = min(n_clusters, len(dfd))
        if k <= 1:
            labels = np.zeros(len(dfd), dtype=int)
            centers = np.mean(mat_z, axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k, random_state=0, n_init=20)
            labels = km.fit_predict(mat_z)
            centers = km.cluster_centers_

        tmp = []
        for cid in range(k):
            cen = centers[cid]
            pname = infer_pattern_name(cen)
            amp = float(np.max(cen) - np.min(cen))
            tmp.append((cid, pname, amp, float(cen[2] - cen[0]), cen))
        tmp = sorted(tmp, key=lambda x: (x[1], -x[2], -abs(x[3])))

        old_to_new = {old: new for new, (old, _, _, _, _) in enumerate(tmp)}
        labels_re = np.array([old_to_new[int(x)] for x in labels], dtype=int)

        pattern_name_map = {}
        for old, pname, amp, long_delta, cen in tmp:
            new_id = old_to_new[old]
            pattern_name_map[new_id] = pname
            centroid_rows.append({
                "domain": str(domain),
                "cluster_id": int(new_id),
                "pattern_name": str(pname),
                "amplitude": float(amp),
                "centroid_2p5": float(cen[0]),
                "centroid_5p7": float(cen[1]),
                "centroid_17p9": float(cen[2]),
                "centroid_delta_5p7_minus_2p5": float(cen[1] - cen[0]),
                "centroid_delta_17p9_minus_5p7": float(cen[2] - cen[1]),
                "centroid_delta_17p9_minus_2p5": float(cen[2] - cen[0]),
            })

        for i, (_, row) in enumerate(dfd.iterrows()):
            cid = int(labels_re[i])
            pname = pattern_name_map.get(cid, "Mixed")
            cluster_rows.append({
                "domain": str(domain),
                "gene": str(row["gene"]),
                "cluster_id": int(cid),
                "pattern_name": str(pname),
                "mean_2p5": float(row["mean_2p5"]),
                "mean_5p7": float(row["mean_5p7"]),
                "mean_17p9": float(row["mean_17p9"]),
                "priority_score": float(row["priority_score"]),
                "kw_stat": float(row["kw_stat"]) if np.isfinite(row["kw_stat"]) else np.nan,
                "temporal_range": float(row["temporal_range"]),
            })

    cluster_df = pd.DataFrame(cluster_rows)
    centroid_df = pd.DataFrame(centroid_rows)
    return cluster_df, centroid_df


# ============================================================
# MODULE SCORE FOR CLUSTER SPATIAL CASE STUDIES
# ============================================================
def compute_module_scores_for_gene_set(data: List[TimepointData], genes: Sequence[str]) -> Dict[str, np.ndarray]:
    genes = [g for g in genes if g in data[0].genes]
    if len(genes) < MODULE_SCORE_MIN_GENES:
        return {}

    gene_to_idx = {g: i for i, g in enumerate(data[0].genes)}
    idx = np.array([gene_to_idx[g] for g in genes], dtype=int)

    pooled = np.concatenate([tp.X[:, idx] for tp in data], axis=0)
    mu = np.mean(pooled, axis=0, keepdims=True)
    sd = np.std(pooled, axis=0, keepdims=True)
    sd[sd <= 1e-12] = 1.0

    scores = {}
    for tp in data:
        z = (tp.X[:, idx] - mu) / sd
        scores[tp.label] = np.mean(z, axis=1)
    return scores


# ============================================================
# PLOTTING
# ============================================================
def plot_domain_reference_map(tp: TimepointData, selected_domains: Sequence[str], save_path: str):
    set_style()
    pal = build_domain_palette(selected_domains)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(tp.coords[:, 0], tp.coords[:, 1], s=POINT_SIZE_BG, c="#D9D9D9", alpha=0.20, linewidths=0, rasterized=True)

    handles = []
    for dom in selected_domains:
        m = tp.domains == str(dom)
        if m.sum() == 0:
            continue
        ax.scatter(tp.coords[m, 0], tp.coords[m, 1], s=POINT_SIZE_FG, c=[pal[str(dom)]], alpha=0.90, linewidths=0, rasterized=True)
        handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor=pal[str(dom)],
                              markeredgecolor="none", markersize=7, label=f"Domain {dom}"))
    ax.set_title(f"Selected domains on reference section ({tp.label})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    apply_axis_orientation(ax)
    ax.legend(handles=handles, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_top_genes_per_domain(stats_df: pd.DataFrame, cluster_df: pd.DataFrame, domains: Sequence[str], save_path: str):
    set_style()
    domains = [str(d) for d in domains if str(d) in stats_df["domain"].astype(str).unique()]
    if len(domains) == 0:
        return

    pattern_map = {}
    if len(cluster_df) > 0:
        pattern_map = {(str(r["domain"]), str(r["gene"])): str(r["pattern_name"]) for _, r in cluster_df.iterrows()}

    n = len(domains)
    ncol = min(3, n)
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.4 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, dom in zip(axes, domains):
        dfd = stats_df[stats_df["domain"] == dom].copy()
        dfd = dfd.sort_values(["priority_score", "kw_stat", "temporal_range"], ascending=[True, False, False]).head(TOP_GENES_PER_DOMAIN_BAR)
        dfd = dfd.iloc[::-1].copy()

        colors = [PATTERN_COLORS.get(pattern_map.get((dom, g), "Mixed"), "#999999") for g in dfd["gene"]]
        ax.barh(np.arange(len(dfd)), dfd["kw_stat"].fillna(0).to_numpy(dtype=float), color=colors, alpha=0.90)
        ax.set_yticks(np.arange(len(dfd)))
        ax.set_yticklabels(dfd["gene"].astype(str).tolist())
        ax.set_title(f"Domain {dom}: top temporal genes")
        ax.set_xlabel("Kruskal-Wallis H")
        ax.set_ylabel("")
        for i, (_, r) in enumerate(dfd.iterrows()):
            txt = f"Δ12={r['delta_5p7_minus_2p5']:.2f}, Δ23={r['delta_17p9_minus_5p7']:.2f}"
            x0 = float(r["kw_stat"]) if np.isfinite(r["kw_stat"]) else 0.0
            ax.text(x0, i, "  " + txt, va="center", ha="left", fontsize=8, color="0.35")

    for ax in axes[len(domains):]:
        ax.axis("off")

    handles = [Line2D([0], [0], color=col, lw=6, label=lab) for lab, col in PATTERN_COLORS.items()]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper right")
    fig.suptitle("Top temporal genes per domain", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.98, 0.96])
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_domain_cluster_heatmap(stats_df: pd.DataFrame, cluster_df: pd.DataFrame, domain: str, save_path: str):
    set_style()
    dfd = cluster_df[cluster_df["domain"] == str(domain)].copy()
    if len(dfd) == 0:
        return

    dfd["pattern_color"] = dfd["pattern_name"].map(lambda x: PATTERN_COLORS.get(x, "#999999"))
    dfd = dfd.sort_values(["cluster_id", "priority_score", "kw_stat"], ascending=[True, True, False]).copy()

    mat = dfd[["mean_2p5", "mean_5p7", "mean_17p9"]].to_numpy(dtype=float)
    mat_z = row_zscore(mat)

    fig_h = max(5.0, 0.18 * len(dfd) + 2.0)
    fig, ax = plt.subplots(figsize=(5.8, fig_h))
    sns.heatmap(
        mat_z,
        cmap="vlag",
        center=0.0,
        yticklabels=dfd["gene"].astype(str).tolist(),
        xticklabels=["2.5m", "5.7m", "17.9m"],
        cbar_kws={"label": "Row-z temporal pattern"},
        ax=ax,
    )

    cluster_breaks = []
    last = None
    for i, cid in enumerate(dfd["cluster_id"].tolist()):
        if last is None:
            last = cid
        elif cid != last:
            cluster_breaks.append(i)
            last = cid
    for y in cluster_breaks:
        ax.axhline(y, color="black", lw=1.0)

    ax.set_title(f"Domain {domain}: clustered temporal gene patterns")
    ax.set_xlabel("Time")
    ax.set_ylabel("Genes")

    for i, color in enumerate(dfd["pattern_color"].tolist()):
        ax.add_patch(plt.Rectangle((-0.38, i), 0.18, 1.0, facecolor=color, edgecolor="none",
                                   transform=ax.transData, clip_on=False))

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_domain_cluster_centroids(centroid_df: pd.DataFrame, domain: str, save_path: str):
    set_style()
    dfd = centroid_df[centroid_df["domain"] == str(domain)].copy()
    if len(dfd) == 0:
        return

    times = np.array([2.5, 5.7, 17.9], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))

    for _, r in dfd.sort_values("cluster_id").iterrows():
        y = np.array([r["centroid_2p5"], r["centroid_5p7"], r["centroid_17p9"]], dtype=float)
        color = PATTERN_COLORS.get(str(r["pattern_name"]), "#999999")
        ax.plot(times, y, marker="o", lw=2.2, ms=6.0,
                color=color, label=f"C{int(r['cluster_id'])}: {r['pattern_name']}")

    ax.axhline(0, color="0.65", lw=0.9)
    ax.set_xticks(times)
    ax.set_xticklabels(["2.5m", "5.7m", "17.9m"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Cluster centroid (row-z)")
    ax.set_title(f"Domain {domain}: temporal cluster centroids")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_interval_cluster_summary(centroid_df: pd.DataFrame, domains: Sequence[str], save_path: str):
    set_style()
    dfd = centroid_df[centroid_df["domain"].isin([str(d) for d in domains])].copy()
    if len(dfd) == 0:
        return

    dfd["row_name"] = dfd.apply(lambda r: f"D{r['domain']} | C{int(r['cluster_id'])} | {r['pattern_name']}", axis=1)
    plot_df = dfd.set_index("row_name")[[
        "centroid_delta_5p7_minus_2p5",
        "centroid_delta_17p9_minus_5p7",
        "centroid_delta_17p9_minus_2p5",
    ]].copy()
    plot_df.columns = ["2.5→5.7", "5.7→17.9", "2.5→17.9"]

    vmax = robust_sym_vmax(plot_df.values, q=98.0)
    fig, ax = plt.subplots(figsize=(6.8, max(4.0, 0.42 * len(plot_df) + 1.6)))
    sns.heatmap(
        plot_df,
        cmap="coolwarm",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.20,
        linecolor="white",
        cbar_kws={"label": "Centroid delta"},
        ax=ax,
    )
    ax.set_title("Domain-wise temporal interval summary from cluster centroids")
    ax.set_xlabel("Interval")
    ax.set_ylabel("Domain / cluster")
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_domain_group_pattern_composition(cluster_df: pd.DataFrame, domain_group: Sequence[str], group_name: str, save_path: str):
    set_style()
    dfd = cluster_df[cluster_df["domain"].isin([str(x) for x in domain_group])].copy()
    if len(dfd) == 0:
        return

    count_df = (
        dfd.groupby(["domain", "pattern_name"])["gene"]
        .count()
        .reset_index(name="n_genes")
        .pivot(index="domain", columns="pattern_name", values="n_genes")
        .fillna(0.0)
    )
    cols = [c for c in ["Increasing", "Decreasing", "Peak@5.7m", "Trough@5.7m", "Mixed"] if c in count_df.columns]
    count_df = count_df[cols]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bottom = np.zeros(len(count_df), dtype=float)
    x = np.arange(len(count_df))

    for col in cols:
        vals = count_df[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=PATTERN_COLORS.get(col, "#999999"), label=col)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(count_df.index.astype(str).tolist(), rotation=20)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Number of clustered genes")
    ax.set_title(f"{group_name}: temporal pattern composition")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_module_panels(data: List[TimepointData],
                               cluster_df: pd.DataFrame,
                               centroid_df: pd.DataFrame,
                               domain: str,
                               save_path: str):
    set_style()

    dfd = cluster_df[cluster_df["domain"] == str(domain)].copy()
    if len(dfd) == 0:
        return

    cend = centroid_df[centroid_df["domain"] == str(domain)].copy()
    if len(cend) == 0:
        return

    cend = cend.sort_values(["amplitude", "cluster_id"], ascending=[False, True]).head(N_SPATIAL_CLUSTERS_PER_DOMAIN)
    chosen_clusters = cend["cluster_id"].astype(int).tolist()

    score_dict = {}
    plot_meta = []
    for cid in chosen_clusters:
        genes = dfd[dfd["cluster_id"] == cid]["gene"].astype(str).tolist()
        genes = genes[:max(MODULE_SCORE_MIN_GENES, min(len(genes), 15))]
        scores = compute_module_scores_for_gene_set(data, genes)
        if len(scores) == 0:
            continue
        pname = cend[cend["cluster_id"] == cid]["pattern_name"].iloc[0]
        score_dict[cid] = scores
        plot_meta.append((cid, pname, genes))

    if len(plot_meta) == 0:
        return

    all_scores = []
    for cid, _, _ in plot_meta:
        for tp in data:
            all_scores.append(score_dict[cid][tp.label])
    vmax = robust_sym_vmax(np.concatenate(all_scores), q=98.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    nrow = len(plot_meta)
    ncol = len(data)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.8 * nrow), squeeze=False)

    for i, (cid, pname, genes) in enumerate(plot_meta):
        for j, tp in enumerate(data):
            ax = axes[i, j]
            scores = score_dict[cid][tp.label]
            m = tp.domains == str(domain)

            ax.scatter(tp.coords[:, 0], tp.coords[:, 1],
                       s=POINT_SIZE_BG, c="#D9D9D9", alpha=BG_ALPHA, linewidths=0, rasterized=True)
            sc = ax.scatter(tp.coords[m, 0], tp.coords[m, 1],
                            s=POINT_SIZE_FG, c=scores[m], cmap="coolwarm", norm=norm,
                            alpha=FG_ALPHA, linewidths=0, rasterized=True)

            ax.set_title(tp.label)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            apply_axis_orientation(ax)

            if j == 0:
                ax.set_ylabel(f"D{domain} | C{cid}\n{pname}\n({len(genes)} genes)", rotation=90)

    fig.suptitle(f"Domain {domain}: spatial module-score variation for representative temporal clusters", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.95, 0.98])

    cax = fig.add_axes([0.96, 0.22, 0.015, 0.56])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Module score (pooled z-average)")

    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_manual_gene_spatial_triptych(data: List[TimepointData], gene: str, domains: Sequence[str], save_path: str):
    set_style()

    genes0 = set(data[0].genes.astype(str).tolist())
    if gene not in genes0:
        return

    gene_to_idx = {g: i for i, g in enumerate(data[0].genes)}
    jg = gene_to_idx[gene]

    selected_domains = [str(d) for d in domains]
    nrow = len(selected_domains)
    ncol = len(data)
    if nrow == 0:
        return

    all_vals = []
    for tp in data:
        all_vals.append(tp.X[:, jg])
    pooled_vals = np.concatenate(all_vals)
    vmin = float(np.percentile(pooled_vals, 1.0))
    vmax_seq = float(np.percentile(pooled_vals, 99.0))
    if not np.isfinite(vmax_seq) or vmax_seq <= vmin:
        vmax_seq = max(vmin + 1e-6, float(np.max(pooled_vals)))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.8 * nrow), squeeze=False)

    for i, dom in enumerate(selected_domains):
        for j, tp in enumerate(data):
            ax = axes[i, j]
            vals = tp.X[:, jg]
            m = tp.domains == dom

            ax.scatter(tp.coords[:, 0], tp.coords[:, 1],
                       s=POINT_SIZE_BG, c="#D9D9D9", alpha=BG_ALPHA, linewidths=0, rasterized=True)
            sc = ax.scatter(tp.coords[m, 0], tp.coords[m, 1],
                            s=POINT_SIZE_FG, c=vals[m], cmap="viridis",
                            vmin=vmin, vmax=vmax_seq,
                            alpha=FG_ALPHA, linewidths=0, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            apply_axis_orientation(ax)
            if i == 0:
                ax.set_title(tp.label)
            if j == 0:
                ax.set_ylabel(f"Domain {dom}")

    fig.suptitle(f"Manual spatial case study: {gene}", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.95, 0.98])

    cax = fig.add_axes([0.96, 0.22, 0.015, 0.56])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Expression")

    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    set_style()
    ensure_dir(OUTDIR)

    data = load_timepoint_data()

    ref_tp = None
    for tp in data:
        if tp.label == "5p7":
            ref_tp = tp
            break
    if ref_tp is None:
        ref_tp = data[0]

    plot_domain_reference_map(
        ref_tp,
        selected_domains=SELECTED_DOMAINS,
        save_path=os.path.join(OUTDIR, "selected_domain_reference_map.png"),
    )

    stats_df = compute_domain_gene_stats(data, selected_domains=SELECTED_DOMAINS)
    stats_df = sort_domain_genes(stats_df)
    stats_df.to_csv(os.path.join(OUTDIR, "domain_gene_temporal_stats.csv"), index=False)

    cluster_df, centroid_df = cluster_domain_patterns(
        stats_df,
        top_n_genes=TOP_GENES_PER_DOMAIN_CLUSTER,
        n_clusters=N_CLUSTERS_PER_DOMAIN,
    )
    cluster_df.to_csv(os.path.join(OUTDIR, "domain_gene_cluster_assignments.csv"), index=False)
    centroid_df.to_csv(os.path.join(OUTDIR, "domain_cluster_centroids.csv"), index=False)

    plot_top_genes_per_domain(
        stats_df=stats_df,
        cluster_df=cluster_df,
        domains=SELECTED_DOMAINS,
        save_path=os.path.join(OUTDIR, "top_temporal_genes_per_domain.png"),
    )

    for dom in SELECTED_DOMAINS:
        plot_domain_cluster_heatmap(
            stats_df=stats_df,
            cluster_df=cluster_df,
            domain=str(dom),
            save_path=os.path.join(OUTDIR, f"domain_{safe_name(dom)}_cluster_heatmap.png"),
        )
        plot_domain_cluster_centroids(
            centroid_df=centroid_df,
            domain=str(dom),
            save_path=os.path.join(OUTDIR, f"domain_{safe_name(dom)}_cluster_centroids.png"),
        )

    plot_interval_cluster_summary(
        centroid_df=centroid_df,
        domains=SELECTED_DOMAINS,
        save_path=os.path.join(OUTDIR, "domain_interval_cluster_summary.png"),
    )

    plot_domain_group_pattern_composition(
        cluster_df=cluster_df,
        domain_group=HIPPOCAMPUS_DOMAINS,
        group_name="Hippocampus domains",
        save_path=os.path.join(OUTDIR, "hippocampus_pattern_composition.png"),
    )
    plot_domain_group_pattern_composition(
        cluster_df=cluster_df,
        domain_group=CORTICAL_LAYER_DOMAINS,
        group_name="Cortical layer domains",
        save_path=os.path.join(OUTDIR, "cortical_pattern_composition.png"),
    )

    spatial_domains = MANUAL_SPATIAL_DOMAINS if MANUAL_SPATIAL_DOMAINS is not None else SELECTED_DOMAINS
    for dom in spatial_domains:
        plot_spatial_module_panels(
            data=data,
            cluster_df=cluster_df,
            centroid_df=centroid_df,
            domain=str(dom),
            save_path=os.path.join(OUTDIR, f"domain_{safe_name(dom)}_cluster_module_spatial.png"),
        )

    if MANUAL_CASE_GENES is not None:
        for gene in MANUAL_CASE_GENES:
            plot_manual_gene_spatial_triptych(
                data=data,
                gene=str(gene),
                domains=SELECTED_DOMAINS,
                save_path=os.path.join(OUTDIR, f"manual_gene_{safe_name(gene)}_spatial_triptych.png"),
            )

    print(f"[DONE] Outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
