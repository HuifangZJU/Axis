#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal gene analysis from 3 aligned h5ad files:
domain-level ranking + pixel-level pattern composition.

Design implemented
------------------
1. Marker case studies (whole landscape, valid domains only):
   For marker genes ['Gfap', 'Arc'], plot
      - 5.7m - 2.5m
      - 17.9m - 5.7m
      - per-location trend map
   with invalid/background domain 'nan' removed.

2. Domain-level ranking:
   Top genes are ranked using domain-level temporal expression summaries, not the
   pixel-level pattern counts. This is intended to be more biologically grounded
   for anatomical regions such as hippocampus and cortical layers.
   Ranking score combines:
      - Kruskal-Wallis H across the 3 time points within the domain
      - temporal range of domain mean expression
      - max absolute domain-mean interval delta

3. Pixel-level percentage bars:
   After domain-level ranking, each top gene is summarized by the fraction of
   matched locations in the domain classified as:
      Increasing / Decreasing / Peak / Trough / Mixed
   using a simple, explicit rule based on the two interval deltas.

4. Domain clustered heatmaps:
   For each domain, cluster top genes by their 3-timepoint trajectory pattern
   using the domain-level mean expression [2.5m, 5.7m, 17.9m], and display
   the 3 time points in the heatmap.

5. Selected merged groups:
      Hippocampus     = domains 8 + 11
      Cortical layers = domains 1 + 4 + 6 + 10
   Representative genes by pattern are selected from the domain-level clustered
   results, and shown as spatial triptychs.
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

from scipy.stats import kruskal
from sklearn.cluster import KMeans


# ============================================================
# CONFIG
# ============================================================
ROOT = "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial2.0/analysis/"
OUTDIR = os.path.join(ROOT, "DE_pixel_analysis_v4")
os.makedirs(OUTDIR, exist_ok=True)

TIMEPOINTS = [
    {"label": "2p5",  "time": 2.5,  "path": "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial2.0/analysis/virtual_AD_time_2.500.h5ad"},
    {"label": "5p7",  "time": 5.7,  "path": "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial2.0/analysis/virtual_AD_time_5.700.h5ad"},
    {"label": "17p9", "time": 17.9, "path": "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial2.0/analysis/virtual_AD_time_17.900.h5ad"},
]

COORD_KEY = "spatial"
DOMAIN_KEY = "axis_joint_domain"

HIPPOCAMPUS_DOMAINS = ["8", "11"]
CORTICAL_LAYER_DOMAINS = ["1", "4", "6", "10"]
DOMAIN_GROUPS = {
    "Hippocampus": HIPPOCAMPUS_DOMAINS,
    "Cortical layers": CORTICAL_LAYER_DOMAINS,

}

MARKER_CASE_GENES = ["Gfap", "Arc"]

COORD_ROUND_DECIMALS = 4
LOG1P_BEFORE_ANALYSIS = True

MIN_LOCATIONS_PER_DOMAIN = 30
TOP_GENES_PER_DOMAIN = 10
TOP_GENES_FOR_HEATMAP = 60
N_GENE_PATTERN_CLUSTERS = 4

# Simple, explicit pixel-level trend rule threshold
PATTERN_MIN_DELTA = 0.05
PATTERN_RELATIVE_FACTOR = 0.1

FIG_DPI = 400
INVERT_X = True
INVERT_Y = False

BG_COLOR = "#D9D9D9"
BG_ALPHA = 0.12
BG_SIZE = 0.5
FG_SIZE = 3.0
FG_ALPHA = 0.95


# ============================================================
# STYLE
# ============================================================
PATTERN_ORDER = ["Increasing", "Decreasing", "Peak", "Trough", "Mixed"]
PATTERN_COLORS = {
    "Increasing": "#D55E00",
    "Decreasing": "#0072B2",
    "Peak": "#009E73",
    "Trough": "#CC79A7",
    "Mixed": "#9E9E9E",
}


def set_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.15,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


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


def sort_domains(domains: Sequence[str]) -> List[str]:
    def _key(x):
        sx = str(x)
        try:
            return (0, float(sx))
        except Exception:
            return (1, sx)
    return sorted([str(d) for d in domains], key=_key)


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


def robust_sym_vmax(vals: np.ndarray, q: float = 99.0) -> float:
    vals = np.asarray(vals, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 1.0
    vmax = np.percentile(np.abs(vals), q)
    return max(float(vmax), 1e-8)


# ============================================================
# DATA CONTAINER
# ============================================================
@dataclass
class AlignedTimepoint:
    label: str
    time: float
    coords: np.ndarray
    domains: np.ndarray
    X: np.ndarray
    genes: np.ndarray
    obs_names: np.ndarray


# ============================================================
# BASIC HELPERS
# ============================================================
def read_h5ad(path: str) -> sc.AnnData:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ad = sc.read_h5ad(path)
    return ad


def get_coords(ad: sc.AnnData, coord_key: str = COORD_KEY) -> np.ndarray:
    if coord_key not in ad.obsm:
        raise KeyError(f"{coord_key!r} not found in ad.obsm")
    arr = np.asarray(ad.obsm[coord_key], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Invalid coordinate shape: {arr.shape}")
    return arr[:, 1::-1].copy()


def get_domains(ad: sc.AnnData, domain_key: str = DOMAIN_KEY) -> np.ndarray:
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
        raise ValueError("No common genes across the input h5ad files.")
    return genes


def coord_keys(coords: np.ndarray, decimals: int = COORD_ROUND_DECIMALS) -> np.ndarray:
    c = np.round(np.asarray(coords, dtype=float), decimals=decimals)
    return np.array([f"{x:.{decimals}f}|{y:.{decimals}f}" for x, y in c], dtype=object)


def valid_domain_mask(domain_array: np.ndarray) -> np.ndarray:
    s = pd.Series(domain_array.astype(str))
    invalid = s.str.lower().isin({"nan", "none", "na", ""})
    return (~invalid).to_numpy()


def valid_domains(domain_array: np.ndarray) -> List[str]:
    mask = valid_domain_mask(domain_array)
    return sort_domains(pd.unique(pd.Series(domain_array[mask].astype(str))))


# ============================================================
# ALIGNMENT
# ============================================================
def align_target_to_reference(
    ref_ad: sc.AnnData,
    tgt_ad: sc.AnnData,
    common_genes: Sequence[str],
    coord_key: str = COORD_KEY,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref_coords = get_coords(ref_ad, coord_key)
    tgt_coords = get_coords(tgt_ad, coord_key)

    ref_keys = coord_keys(ref_coords, decimals=COORD_ROUND_DECIMALS)
    tgt_keys = coord_keys(tgt_coords, decimals=COORD_ROUND_DECIMALS)

    ref_index = pd.Index(ref_keys)
    tgt_index = pd.Index(tgt_keys)
    common = ref_index.intersection(tgt_index)

    if len(common) == 0:
        raise ValueError("No overlapping rounded coordinates found between reference and target.")

    ref_pos = ref_index.get_indexer(common)
    tgt_pos = tgt_index.get_indexer(common)

    order = np.argsort(ref_pos)
    ref_pos = ref_pos[order]
    tgt_pos = tgt_pos[order]

    tgt_sub = tgt_ad[:, list(common_genes)].copy()
    X_tgt = make_dense_X(tgt_sub)[tgt_pos]
    dom_tgt = get_domains(tgt_ad)[tgt_pos]
    obs_tgt = tgt_ad.obs_names.astype(str).to_numpy()[tgt_pos]

    coords_ref_sub = ref_coords[ref_pos]
    return coords_ref_sub, X_tgt, dom_tgt, obs_tgt


def load_aligned_timepoints() -> List[AlignedTimepoint]:
    ads = [read_h5ad(tp["path"]) for tp in TIMEPOINTS]

    common_genes = intersect_genes(ads)

    ref_meta = TIMEPOINTS[0]
    ref_ad = ads[0][:, common_genes].copy()
    ref_coords = get_coords(ref_ad)
    ref_domains = get_domains(ref_ad)
    ref_X = make_dense_X(ref_ad)
    ref_obs = ref_ad.obs_names.astype(str).to_numpy()

    data = [
        AlignedTimepoint(
            label=ref_meta["label"],
            time=float(ref_meta["time"]),
            coords=ref_coords,
            domains=ref_domains,
            X=ref_X,
            genes=np.asarray(common_genes, dtype=object),
            obs_names=ref_obs,
        )
    ]

    for meta, ad in zip(TIMEPOINTS[1:], ads[1:]):
        coords_ref_sub, X_tgt, dom_tgt, obs_tgt = align_target_to_reference(
            ref_ad=ads[0],
            tgt_ad=ad,
            common_genes=common_genes,
            coord_key=COORD_KEY,
        )

        if len(data) == 1 and len(coords_ref_sub) != len(data[0].coords):
            keys_ref_all = coord_keys(data[0].coords)
            keys_keep = coord_keys(coords_ref_sub)
            keep_mask = pd.Index(keys_ref_all).isin(pd.Index(keys_keep))
            data[0] = AlignedTimepoint(
                label=data[0].label,
                time=data[0].time,
                coords=data[0].coords[keep_mask],
                domains=data[0].domains[keep_mask],
                X=data[0].X[keep_mask],
                genes=data[0].genes,
                obs_names=data[0].obs_names[keep_mask],
            )

        ref_keys_keep = coord_keys(data[0].coords)
        cur_keys = coord_keys(coords_ref_sub)
        idx_cur = pd.Index(cur_keys).get_indexer(ref_keys_keep)
        if np.any(idx_cur < 0):
            raise ValueError("Current target failed to align to the retained reference set.")

        data.append(
            AlignedTimepoint(
                label=meta["label"],
                time=float(meta["time"]),
                coords=data[0].coords.copy(),
                domains=dom_tgt[idx_cur],
                X=X_tgt[idx_cur],
                genes=np.asarray(common_genes, dtype=object),
                obs_names=obs_tgt[idx_cur],
            )
        )

    ref_domains_final = data[0].domains.copy()
    for i in range(1, len(data)):
        data[i].domains = ref_domains_final.copy()

    print(f"[INFO] Loaded times: {[tp.time for tp in data]}")
    print(f"[INFO] Matched locations: {data[0].coords.shape[0]}")
    print(f"[INFO] Valid domains: {valid_domains(data[0].domains)}")
    print(f"[INFO] Common genes: {len(data[0].genes)}")
    return data


# ============================================================
# SIMPLE, INTUITIVE PIXEL-LEVEL TREND RULE
# ============================================================
def classify_location_pattern(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Explicit and intuitive rule based on the two interval deltas.

    Let:
        d12 = v2 - v1
        d23 = v3 - v2

    Threshold:
        tau = max(PATTERN_MIN_DELTA, PATTERN_RELATIVE_FACTOR * median amplitude)

    Then:
        Increasing : d12 > +tau and d23 > +tau
        Decreasing : d12 < -tau and d23 < -tau
        Peak       : d12 > +tau and d23 < -tau
        Trough     : d12 < -tau and d23 > +tau
        Mixed      : otherwise
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)

    d12 = v2 - v1
    d23 = v3 - v2
    amp = np.maximum(np.abs(d12), np.abs(d23))

    tau = max(PATTERN_MIN_DELTA, float(np.nanmedian(amp) * PATTERN_RELATIVE_FACTOR))

    pat = np.full(v1.shape[0], "Mixed", dtype=object)
    pat[(d12 > tau) & (d23 > tau)] = "Increasing"
    pat[(d12 < -tau) & (d23 < -tau)] = "Decreasing"
    pat[(d12 > tau) & (d23 < -tau)] = "Peak"
    pat[(d12 < -tau) & (d23 > tau)] = "Trough"

    return pat, amp.astype(np.float32)


def infer_pattern_name_from_trajectory_centroid(centroid: np.ndarray) -> str:
    a, b, c = [float(x) for x in centroid]
    if (b > a) and (c >= b):
        return "Increasing"
    if (b < a) and (c <= b):
        return "Decreasing"
    if (b > a) and (b > c):
        return "Peak"
    if (b < a) and (b < c):
        return "Trough"
    return "Mixed"


# ============================================================
# DOMAIN-LEVEL RANKING + PIXEL-LEVEL PERCENTAGES
# ============================================================
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
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    tmp = np.empty_like(q)
    tmp[order] = q
    out[good] = tmp
    return out


def compute_domain_gene_stats(data: List[AlignedTimepoint]) -> pd.DataFrame:
    X1, X2, X3 = data[0].X, data[1].X, data[2].X
    genes = data[0].genes.astype(str).tolist()
    domains = valid_domains(data[0].domains)

    rows = []
    for dom in domains:
        mask = data[0].domains == str(dom)
        nloc = int(mask.sum())
        if nloc < MIN_LOCATIONS_PER_DOMAIN:
            print(f"[WARN] Skip domain {dom}: only {nloc} matched locations.")
            continue

        A = X1[mask]
        B = X2[mask]
        C = X3[mask]

        mean1 = A.mean(axis=0)
        mean2 = B.mean(axis=0)
        mean3 = C.mean(axis=0)

        delta12 = mean2 - mean1
        delta23 = mean3 - mean2
        delta13 = mean3 - mean1

        temporal_range = np.max(np.vstack([mean1, mean2, mean3]).T, axis=1) - np.min(np.vstack([mean1, mean2, mean3]).T, axis=1)
        max_abs_mean_delta = np.maximum(np.abs(delta12), np.abs(delta23))

        kw_stat = np.full(len(genes), np.nan, dtype=float)
        kw_p = np.full(len(genes), np.nan, dtype=float)

        inc_frac = np.full(len(genes), np.nan, dtype=float)
        dec_frac = np.full(len(genes), np.nan, dtype=float)
        peak_frac = np.full(len(genes), np.nan, dtype=float)
        trough_frac = np.full(len(genes), np.nan, dtype=float)
        mixed_frac = np.full(len(genes), np.nan, dtype=float)

        for j, gene in enumerate(genes):
            v1 = A[:, j]
            v2 = B[:, j]
            v3 = C[:, j]

            try:
                if not np.allclose(np.concatenate([v1, v2, v3]), np.concatenate([v1, v2, v3])[0]):
                    stat, p = kruskal(v1, v2, v3)
                    kw_stat[j] = float(stat)
                    kw_p[j] = float(p)
            except Exception:
                pass

            pat, _ = classify_location_pattern(v1, v2, v3)
            inc_frac[j] = np.mean(pat == "Increasing")
            dec_frac[j] = np.mean(pat == "Decreasing")
            peak_frac[j] = np.mean(pat == "Peak")
            trough_frac[j] = np.mean(pat == "Trough")
            mixed_frac[j] = np.mean(pat == "Mixed")

        kw_fdr = bh_fdr(kw_p)

        rank_kw = pd.Series(-np.nan_to_num(kw_stat, nan=-np.inf)).rank(method="average", ascending=True).to_numpy()
        rank_range = pd.Series(-temporal_range).rank(method="average", ascending=True).to_numpy()
        rank_delta = pd.Series(-max_abs_mean_delta).rank(method="average", ascending=True).to_numpy()
        priority_score = (rank_kw + rank_range + rank_delta) / 3.0

        for j, gene in enumerate(genes):
            rows.append({
                "domain": str(dom),
                "gene": str(gene),
                "n_locations": nloc,

                "mean_2p5": float(mean1[j]),
                "mean_5p7": float(mean2[j]),
                "mean_17p9": float(mean3[j]),

                "mean_delta_5p7_minus_2p5": float(delta12[j]),
                "mean_delta_17p9_minus_5p7": float(delta23[j]),
                "mean_delta_17p9_minus_2p5": float(delta13[j]),

                "temporal_range": float(temporal_range[j]),
                "max_abs_mean_delta": float(max_abs_mean_delta[j]),

                "kw_stat": float(kw_stat[j]) if np.isfinite(kw_stat[j]) else np.nan,
                "kw_pval": float(kw_p[j]) if np.isfinite(kw_p[j]) else np.nan,
                "kw_fdr": float(kw_fdr[j]) if np.isfinite(kw_fdr[j]) else np.nan,

                "inc_frac": float(inc_frac[j]),
                "dec_frac": float(dec_frac[j]),
                "peak_frac": float(peak_frac[j]),
                "trough_frac": float(trough_frac[j]),
                "mixed_frac": float(mixed_frac[j]),

                "priority_score": float(priority_score[j]),
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("No domain-level gene statistics were computed.")

    df = df.sort_values(
        ["domain", "priority_score", "kw_stat", "max_abs_mean_delta"],
        ascending=[True, True, False, False]
    ).reset_index(drop=True)
    return df


def cluster_top_genes_within_domain(
    stats_df: pd.DataFrame,
    top_n: int = TOP_GENES_FOR_HEATMAP,
    n_clusters: int = N_GENE_PATTERN_CLUSTERS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gene_rows = []
    centroid_rows = []

    for dom in sort_domains(pd.unique(stats_df["domain"])):
        dfd = stats_df[stats_df["domain"] == str(dom)].copy()
        dfd = dfd.sort_values(
            ["priority_score", "kw_stat", "max_abs_mean_delta"],
            ascending=[True, False, False]
        ).head(top_n)

        if len(dfd) == 0:
            continue

        mat = dfd[["mean_2p5", "mean_5p7", "mean_17p9"]].to_numpy(dtype=float)
        mat_z = row_zscore(mat)

        k = min(n_clusters, len(dfd))
        if k <= 1:
            labels = np.zeros(len(dfd), dtype=int)
            centers = np.mean(mat_z, axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k, n_init=20, random_state=0)
            labels = km.fit_predict(mat_z)
            centers = km.cluster_centers_

        tmp = []
        for old in range(k):
            cen = centers[old]
            pname = infer_pattern_name_from_trajectory_centroid(cen)
            amp = float(np.max(cen) - np.min(cen))
            tmp.append((old, pname, amp, cen))

        tmp = sorted(
            tmp,
            key=lambda x: (PATTERN_ORDER.index(x[1]) if x[1] in PATTERN_ORDER else 99, -x[2])
        )

        old_to_new = {old: new for new, (old, _, _, _) in enumerate(tmp)}
        labels_new = np.array([old_to_new[int(x)] for x in labels], dtype=int)

        for old, pname, amp, cen in tmp:
            new_id = old_to_new[old]
            centroid_rows.append({
                "domain": str(dom),
                "cluster_id": int(new_id),
                "pattern_name": str(pname),
                "amplitude": float(amp),
                "centroid_2p5": float(cen[0]),
                "centroid_5p7": float(cen[1]),
                "centroid_17p9": float(cen[2]),
            })

        for i, (_, r) in enumerate(dfd.iterrows()):
            cid = int(labels_new[i])
            pname = [x["pattern_name"] for x in centroid_rows if x["domain"] == str(dom) and x["cluster_id"] == cid][-1]
            gene_rows.append({
                "domain": str(dom),
                "gene": str(r["gene"]),
                "cluster_id": cid,
                "pattern_name": pname,
                "mean_2p5": float(r["mean_2p5"]),
                "mean_5p7": float(r["mean_5p7"]),
                "mean_17p9": float(r["mean_17p9"]),
                "mean_delta_5p7_minus_2p5": float(r["mean_delta_5p7_minus_2p5"]),
                "mean_delta_17p9_minus_5p7": float(r["mean_delta_17p9_minus_5p7"]),
                "priority_score": float(r["priority_score"]),
                "kw_stat": float(r["kw_stat"]) if np.isfinite(r["kw_stat"]) else np.nan,
            })

    return pd.DataFrame(gene_rows), pd.DataFrame(centroid_rows)


def choose_representative_genes_by_group_pattern(
    cluster_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    domain_groups: Dict[str, Sequence[str]],
) -> pd.DataFrame:
    rows = []
    merged = cluster_df.merge(
        stats_df[["domain", "gene", "kw_stat"]],
        on=["domain", "gene"],
        how="left",
        suffixes=("", "_stat"),
    )

    for group_name, doms in domain_groups.items():
        subg = merged[merged["domain"].isin([str(x) for x in doms])].copy()
        if len(subg) == 0:
            continue

        for patt in ["Increasing", "Decreasing", "Peak", "Trough"]:
            sub = subg[subg["pattern_name"] == patt].copy()
            if len(sub) == 0:
                continue
            sub = sub.sort_values(["priority_score", "kw_stat"], ascending=[True, False])
            r = sub.iloc[0]
            rows.append({
                "group_name": str(group_name),
                "pattern_name": str(patt),
                "gene": str(r["gene"]),
                "source_domain": str(r["domain"]),
                "priority_score": float(r["priority_score"]),
                "kw_stat": float(r["kw_stat"]) if np.isfinite(r["kw_stat"]) else np.nan,
            })
    return pd.DataFrame(rows)


# ============================================================
# PLOTTING
# ============================================================
def scatter_delta_panel(ax, coords, vals, title, cmap="vlag", s=1.5, alpha=0.95, mask=None):
    vals = np.asarray(vals, dtype=float)
    vmax = robust_sym_vmax(vals, q=99.0)

    if mask is None:
        mask = np.ones(coords.shape[0], dtype=bool)

    mask = np.asarray(mask, dtype=bool)
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=vals[mask], cmap=cmap, vmin=-vmax, vmax=vmax,
               s=s, alpha=alpha, linewidths=0, rasterized=True)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    apply_axis_orientation(ax)


def scatter_trend_panel(ax, coords, patterns, title, mask=None, s=1.5):
    if mask is None:
        mask = np.ones(coords.shape[0], dtype=bool)

    mask = np.asarray(mask, dtype=bool)
    # for patt in ["Increasing", "Decreasing", "Peak", "Trough", "Mixed"]:
    for patt in ["Increasing", "Decreasing", "Peak", "Trough"]:
        m = (np.asarray(patterns) == patt) & mask
        if np.any(m):
            ax.scatter(coords[m, 0], coords[m, 1],
                       color=PATTERN_COLORS[patt], s=s, alpha=0.9, linewidths=0,
                       label=patt, rasterized=True)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    apply_axis_orientation(ax)


def plot_gene_triptych(
    data: List[AlignedTimepoint],
    gene: str,
    save_path: str,
    mask: Optional[np.ndarray] = None,
    title_suffix: Optional[str] = None,
    remove_invalid_background: bool = False,
):
    genes = data[0].genes.astype(str).tolist()
    if gene not in genes:
        print(f"[WARN] Gene {gene} not found. Skip figure.")
        return

    j = genes.index(gene)
    coords = data[0].coords

    valid_mask = valid_domain_mask(data[0].domains)
    if mask is None:
        mask = valid_mask if remove_invalid_background else np.ones(coords.shape[0], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if remove_invalid_background:
            mask = mask & valid_mask

    v1 = data[0].X[:, j]
    v2 = data[1].X[:, j]
    v3 = data[2].X[:, j]

    d12 = v2 - v1
    d23 = v3 - v2
    pats, _ = classify_location_pattern(v1, v2, v3)

    subtitle = title_suffix if title_suffix is not None else "Whole landscape"

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    scatter_delta_panel(axes[0], coords, d12, "5.7m − 2.5m", mask=mask)
    scatter_delta_panel(axes[1], coords, d23, "17.9m − 5.7m", mask=mask)
    scatter_trend_panel(axes[2], coords, pats, "Per-location trend", mask=mask)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PATTERN_COLORS[p],
                   markersize=7, label=p)
        for p in ["Increasing", "Decreasing", "Peak", "Trough", "Mixed"]
    ]
    axes[2].legend(handles=handles, loc="upper right", frameon=False, title="Pattern")

    fig.suptitle(f"{gene} | {subtitle}", y=1.01, fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_all_domains_top_genes_summary(stats_df: pd.DataFrame, save_path: str, top_n_show: int = TOP_GENES_PER_DOMAIN):
    set_style()
    domains = sort_domains(pd.unique(stats_df["domain"]))
    if len(domains) == 0:
        return

    ncol = 3
    nrow = int(math.ceil(len(domains) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.3 * ncol, 4.2 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, dom in zip(axes, domains):
        dfd = stats_df[stats_df["domain"] == str(dom)].copy()
        dfd = dfd.sort_values(
            ["priority_score", "kw_stat", "max_abs_mean_delta"],
            ascending=[True, False, False]
        ).head(top_n_show).iloc[::-1]

        y = np.arange(len(dfd))
        left = np.zeros(len(dfd), dtype=float)

        frac_map = {
            "Increasing": dfd["inc_frac"].to_numpy(dtype=float),
            "Decreasing": dfd["dec_frac"].to_numpy(dtype=float),
            "Peak": dfd["peak_frac"].to_numpy(dtype=float),
            "Trough": dfd["trough_frac"].to_numpy(dtype=float),
            "Mixed": dfd["mixed_frac"].to_numpy(dtype=float),
        }

        # for patt in ["Increasing", "Decreasing", "Peak", "Trough", "Mixed"]:
        for patt in ["Increasing", "Decreasing", "Peak", "Trough"]:
            vals = frac_map[patt]
            ax.barh(y, vals, left=left, color=PATTERN_COLORS[patt], alpha=0.92, label=patt)
            left += vals

        ax.set_yticks(y)
        ax.set_yticklabels(dfd["gene"].astype(str).tolist())
        ax.set_title(f"Domain {dom}")
        ax.set_xlabel("Fraction of matched locations")
        ax.set_xlim(0, 1.0)

        for i, (_, r) in enumerate(dfd.iterrows()):
            # txt = f"H={r['kw_stat']:.0f} | Δ12={r['mean_delta_5p7_minus_2p5']:.2f}, Δ23={r['mean_delta_17p9_minus_5p7']:.2f}"
            txt = f"Δ12={r['mean_delta_5p7_minus_2p5']:.2f}, Δ23={r['mean_delta_17p9_minus_5p7']:.2f}"
            ax.text(1.01, i, txt, va="center", ha="left", fontsize=7.2, color="0.35", transform=ax.get_yaxis_transform())

    for ax in axes[len(domains):]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=PATTERN_COLORS[p], lw=6, label=p)
        for p in ["Increasing", "Decreasing", "Peak", "Trough", "Mixed"]
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Top temporal genes per domain", y=1.05, fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_selected_domains_top_genes_summary(
    stats_df: pd.DataFrame,
    save_path: str,
    top_n_show: int = TOP_GENES_PER_DOMAIN,
    summary_csv_path: str = None,
):
    set_style()

    selected_domains = [str(x) for x in (HIPPOCAMPUS_DOMAINS + CORTICAL_LAYER_DOMAINS)]
    domains = [d for d in sort_domains(pd.unique(stats_df["domain"])) if str(d) in selected_domains]

    if len(domains) == 0:
        return

    # collect the selected-domain summary for saving
    summary_list = []
    for dom in domains:
        dfd = stats_df[stats_df["domain"].astype(str) == str(dom)].copy()
        dfd = dfd.sort_values(
            ["priority_score", "kw_stat", "max_abs_mean_delta"],
            ascending=[True, False, False]
        ).head(top_n_show)

        dfd = dfd.copy()
        dfd["domain"] = str(dom)
        summary_list.append(dfd)

    selected_summary_df = pd.concat(summary_list, axis=0, ignore_index=True)

    # save selected-domain summary
    if summary_csv_path is None:
        summary_csv_path = os.path.splitext(save_path)[0] + ".csv"
    selected_summary_df.to_csv(summary_csv_path, index=False)

    # plotting
    ncol = 3
    nrow = int(math.ceil(len(domains) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.3 * ncol, 4.2 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, dom in zip(axes, domains):
        dfd = selected_summary_df[selected_summary_df["domain"] == str(dom)].copy()
        dfd = dfd.iloc[::-1]  # reverse for top gene at top in barh

        y = np.arange(len(dfd))
        left = np.zeros(len(dfd), dtype=float)

        frac_map = {
            "Increasing": dfd["inc_frac"].to_numpy(dtype=float),
            "Decreasing": dfd["dec_frac"].to_numpy(dtype=float),
            "Peak": dfd["peak_frac"].to_numpy(dtype=float),
            "Trough": dfd["trough_frac"].to_numpy(dtype=float),
            "Mixed": dfd["mixed_frac"].to_numpy(dtype=float),
        }

        # for patt in ["Increasing", "Decreasing", "Peak", "Trough", "Mixed"]:
        for patt in ["Increasing", "Decreasing", "Peak", "Trough"]:
            vals = frac_map[patt]
            ax.barh(
                y,
                vals,
                left=left,
                color=PATTERN_COLORS[patt],
                alpha=0.92,
                label=patt,
            )
            left += vals

        ax.set_yticks(y)
        ax.set_yticklabels(dfd["gene"].astype(str).tolist())
        ax.set_title(f"Domain {dom}")
        ax.set_xlabel("Fraction of matched locations")
        ax.set_xlim(0, 1.0)

        for i, (_, r) in enumerate(dfd.iterrows()):
            txt = (
                f"Δ12={r['mean_delta_5p7_minus_2p5']:.2f}, "
                f"Δ23={r['mean_delta_17p9_minus_5p7']:.2f}"
            )
            ax.text(
                1.01,
                i,
                txt,
                va="center",
                ha="left",
                fontsize=7.2,
                color="0.35",
                transform=ax.get_yaxis_transform(),
            )

    # turn off unused axes
    for ax in axes[len(domains):]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=PATTERN_COLORS[p], lw=6, label=p)
        for p in ["Increasing", "Decreasing", "Peak", "Trough", "Mixed"]
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle("Top temporal genes per selected domains", y=1.05, fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_domain_clustered_heatmap(cluster_df: pd.DataFrame, domain: str, save_path: str):
    set_style()
    dfd = cluster_df[cluster_df["domain"] == str(domain)].copy()
    if len(dfd) == 0:
        return

    dfd = dfd.sort_values(["cluster_id", "priority_score", "kw_stat"], ascending=[True, True, False]).copy()
    mat = dfd[["mean_2p5", "mean_5p7", "mean_17p9"]].to_numpy(dtype=float)
    mat_z = row_zscore(mat)

    fig_h = max(5.0, 0.18 * len(dfd) + 2.0)
    fig, ax = plt.subplots(figsize=(5.9, fig_h))
    sns.heatmap(
        mat_z,
        cmap="vlag",
        center=0.0,
        yticklabels=dfd["gene"].astype(str).tolist(),
        xticklabels=["2.5m", "5.7m", "17.9m"],
        cbar_kws={"label": "Row-z temporal pattern"},
        ax=ax,
    )

    breaks = []
    last = None
    for i, cid in enumerate(dfd["cluster_id"].tolist()):
        if last is None:
            last = cid
        elif cid != last:
            breaks.append(i)
            last = cid
    for y in breaks:
        ax.axhline(y, color="black", lw=1.0)

    ax.set_title(f"Domain {domain}: clustered heatmap of 3 time points")
    ax.set_xlabel("Time")
    ax.set_ylabel("Genes")
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_group_pattern_gene_panel(
    data: List[AlignedTimepoint],
    rep_df: pd.DataFrame,
    group_name: str,
    group_domains: Sequence[str],
    save_path: str,
):
    sub = rep_df[rep_df["group_name"] == str(group_name)].copy()
    if len(sub) == 0:
        return

    pattern_order = ["Increasing", "Decreasing", "Peak", "Trough"]
    sub["pattern_name"] = pd.Categorical(sub["pattern_name"], categories=pattern_order, ordered=True)
    sub = sub.sort_values("pattern_name")

    genes = data[0].genes.astype(str).tolist()
    coords = data[0].coords
    group_mask = np.isin(data[0].domains.astype(str), [str(x) for x in group_domains])

    set_style()
    nrows = len(sub)
    fig, axes = plt.subplots(nrows, 3, figsize=(13.8, 3.8 * nrows), squeeze=False)

    for r, (_, row) in enumerate(sub.iterrows()):
        gene = str(row["gene"])
        if gene not in genes:
            continue
        j = genes.index(gene)

        v1 = data[0].X[:, j]
        v2 = data[1].X[:, j]
        v3 = data[2].X[:, j]

        d12 = v2 - v1
        d23 = v3 - v2
        pats, _ = classify_location_pattern(v1, v2, v3)

        scatter_delta_panel(axes[r, 0], coords, d12, f"{gene} | 5.7m − 2.5m", mask=group_mask)
        scatter_delta_panel(axes[r, 1], coords, d23, f"{gene} | 17.9m − 5.7m", mask=group_mask)
        scatter_trend_panel(axes[r, 2], coords, pats, f"{gene} | {row['pattern_name']}", mask=group_mask)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PATTERN_COLORS[p],
                   markersize=7, label=p)
        for p in ["Increasing", "Decreasing", "Peak", "Trough", "Mixed"]
    ]
    axes[0, 2].legend(handles=handles, loc="upper right", frameon=False, title="Pattern")

    fig.suptitle(f"{group_name}: representative temporal genes by pattern", y=1.01, fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    set_style()

    figdir = os.path.join(OUTDIR, "figures")
    tbldir = os.path.join(OUTDIR, "tables")
    ensure_dir(figdir)
    ensure_dir(tbldir)

    data = load_aligned_timepoints()

    stats_df = compute_domain_gene_stats(data)
    stats_csv = os.path.join(tbldir, "domain_gene_stats.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"[OK] saved: {stats_csv}")

    cluster_df, centroid_df = cluster_top_genes_within_domain(
        stats_df=stats_df,
        top_n=TOP_GENES_FOR_HEATMAP,
        n_clusters=N_GENE_PATTERN_CLUSTERS,
    )
    cluster_csv = os.path.join(tbldir, "domain_top_gene_clusters.csv")
    centroid_csv = os.path.join(tbldir, "domain_cluster_centroids.csv")
    cluster_df.to_csv(cluster_csv, index=False)
    centroid_df.to_csv(centroid_csv, index=False)
    print(f"[OK] saved: {cluster_csv}")
    print(f"[OK] saved: {centroid_csv}")

    rep_df = choose_representative_genes_by_group_pattern(
        cluster_df=cluster_df,
        stats_df=stats_df,
        domain_groups=DOMAIN_GROUPS,
    )
    rep_csv = os.path.join(tbldir, "selected_group_representative_genes.csv")
    rep_df.to_csv(rep_csv, index=False)
    print(f"[OK] saved: {rep_csv}")

    marker_dir = os.path.join(figdir, "marker_case_studies")
    ensure_dir(marker_dir)
    for gene in MARKER_CASE_GENES:
        plot_gene_triptych(
            data=data,
            gene=gene,
            save_path=os.path.join(marker_dir, f"whole_landscape_{safe_name(gene)}.png"),
            mask=None,
            title_suffix="Whole landscape (valid domains only)",
            remove_invalid_background=True,
        )

    plot_all_domains_top_genes_summary(
        stats_df=stats_df,
        save_path=os.path.join(figdir, "all_domains_top_genes_summary.png"),
        top_n_show=TOP_GENES_PER_DOMAIN,
    )
    plot_selected_domains_top_genes_summary(
        stats_df=stats_df,
        save_path=os.path.join(figdir, "selected_domains_top_genes_summary.png"),
        top_n_show=TOP_GENES_PER_DOMAIN,
    )

    heatmap_dir = os.path.join(figdir, "domain_clustered_heatmaps")
    ensure_dir(heatmap_dir)
    for dom in sort_domains(pd.unique(cluster_df["domain"])):
        plot_domain_clustered_heatmap(
            cluster_df=cluster_df,
            domain=str(dom),
            save_path=os.path.join(heatmap_dir, f"domain_{safe_name(dom)}_clustered_heatmap.png"),
        )

    selected_dir = os.path.join(figdir, "selected_domain_spatial_maps")
    ensure_dir(selected_dir)
    for group_name, doms in DOMAIN_GROUPS.items():
        plot_group_pattern_gene_panel(
            data=data,
            rep_df=rep_df,
            group_name=group_name,
            group_domains=doms,
            save_path=os.path.join(selected_dir, f"{safe_name(group_name)}_representative_genes.png"),
        )

    print(f"[DONE] Results saved under: {OUTDIR}")


if __name__ == "__main__":
    main()
