#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare AXIS vs SpatialZ virtual slices for time-series data.

Only compute and plot two metrics:
1. Domain-based neighborhood component agreement to the real-slice group
2. Adjacent domain gene-profile correlation by interval, with real anchor
   time points inserted into each virtual method sequence

Usage
-----
python compare_virtual_timeseries_grouped_minimal.py --mode summarize
python compare_virtual_timeseries_grouped_minimal.py --mode plot
python compare_virtual_timeseries_grouped_minimal.py --mode all
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# CONFIG
# ============================================================

OUTDIR = "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/evaluation"

METHODS = {
    "AXIS": {
        3.3: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial1.0/query/virtual_AD_time_3.300.h5ad",
        4.1: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial1.0/query/virtual_AD_time_4.100.h5ad",
        4.9: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial1.0/query/virtual_AD_time_4.900.h5ad",
        8.75: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial1.0/query/virtual_AD_time_8.750.h5ad",
        11.8: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial1.0/query/virtual_AD_time_11.800.h5ad",
        14.85: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/AD_time_grid256_spatial1.0/query/virtual_AD_time_14.850.h5ad",
    },
    "SpatialZ": {
        3.3: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/3.3.h5ad",
        4.1: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/4.1.h5ad",
        4.9: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/4.9.h5ad",
        8.75: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/8.75.h5ad",
        11.8: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/11.8.h5ad",
        14.85: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/14.85.h5ad",
    },
}

REAL_SLICES = {
    2.5: "/media/huifang/data/registration/xenium/mouse_brain/h5ads/Tg/0_0.h5ad",
    5.7: "/media/huifang/data/registration/xenium/mouse_brain/h5ads/Tg/0_1.h5ad",
    17.9: "/media/huifang/data/registration/xenium/mouse_brain/h5ads/Tg/0_2.h5ad",
}

COORD_KEY = "spatial"
DOMAIN_KEY = "axis_joint_domain"

FILTER_VALID = True
VALID_KEY = "valid"
USE_LOG1P = True

SELECTED_DOMAINS: Optional[List[str]] = None
MANUAL_GENES: Optional[List[str]] = None
MAX_GENES_FOR_METRICS = 300
MIN_GLOBAL_DETECTION_FRAC = 0.01

KNN_K = 50

FIG_DPI = 300
GROUP_PALETTE = {
    "Real": "#7f7f7f",
    "AXIS": "#d45a45",
    "SpatialZ": "#8fa0b8",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_text_lines(lines: Sequence[str], path: str):
    with open(path, "w") as f:
        for x in lines:
            f.write(str(x) + "\n")


def safe_read_csv(path: str, columns: Optional[List[str]] = None):
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        return pd.DataFrame(columns=columns if columns is not None else [])
    return pd.read_csv(path)


def read_h5ad(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ad = sc.read_h5ad(path)
    if FILTER_VALID and (VALID_KEY in ad.obs.columns):
        ad = ad[ad.obs[VALID_KEY].astype(bool)].copy()
    return ad


def get_coords(ad):
    if COORD_KEY not in ad.obsm:
        raise KeyError(f"Missing ad.obsm['{COORD_KEY}']")
    coords = np.asarray(ad.obsm[COORD_KEY], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coordinate shape: {coords.shape}")
    return coords[:, :2]


def get_domains(ad):
    if DOMAIN_KEY not in ad.obs.columns:
        raise KeyError(f"Missing ad.obs['{DOMAIN_KEY}']")
    vals = ad.obs[DOMAIN_KEY].astype(str).to_numpy()
    return np.asarray(vals, dtype=object)


def extract_gene_matrix_original(ad, genes: Sequence[str], use_log1p=True):
    genes = list(genes)
    missing = [g for g in genes if g not in ad.var_names]
    if len(missing) > 0:
        raise KeyError(f"Missing genes: {missing[:5]} ...")

    idx = [int(np.where(ad.var_names == g)[0][0]) for g in genes]
    X = ad.X[:, idx]

    if sp.issparse(X):
        X = np.asarray(X.toarray(), dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    if use_log1p:
        X = np.log1p(X)

    return X


def get_all_common_genes(real_slices: Dict[float, str], methods: Dict[str, Dict[float, str]]):
    common = None

    for _, p in real_slices.items():
        ad = read_h5ad(p)
        genes = set(ad.var_names.tolist())
        common = genes if common is None else common.intersection(genes)

    for _, mapping in methods.items():
        for _, p in mapping.items():
            ad = read_h5ad(p)
            genes = set(ad.var_names.tolist())
            common = genes if common is None else common.intersection(genes)

    if common is None:
        return []
    return sorted(common)


def get_selected_domains(real_slices: Dict[float, str], methods: Dict[str, Dict[float, str]]):
    if SELECTED_DOMAINS is not None:
        return [str(x) for x in SELECTED_DOMAINS]

    ds = set()

    for _, p in real_slices.items():
        ad = read_h5ad(p)
        ds.update(np.unique(get_domains(ad)))

    for _, mapping in methods.items():
        for _, p in mapping.items():
            ad = read_h5ad(p)
            ds.update(np.unique(get_domains(ad)))

    ds = sorted([d for d in ds if str(d).lower() != "nan"])
    return ds


def select_genes_for_metrics(
    real_slices: Dict[float, str],
    methods: Dict[str, Dict[float, str]],
    manual_genes: Optional[List[str]] = None,
    max_genes: int = MAX_GENES_FOR_METRICS,
    min_detect_frac: float = MIN_GLOBAL_DETECTION_FRAC,
):
    if manual_genes is not None:
        return list(manual_genes)

    common_genes = get_all_common_genes(real_slices, methods)
    if len(common_genes) == 0:
        return []

    mean_list = []
    detect_list = []

    for _, p in real_slices.items():
        ad = read_h5ad(p)
        X = extract_gene_matrix_original(ad, common_genes, use_log1p=USE_LOG1P)
        mean_list.append(np.mean(X, axis=0))
        detect_list.append(np.mean(X > 0, axis=0))

    for _, mapping in methods.items():
        for _, p in mapping.items():
            ad = read_h5ad(p)
            X = extract_gene_matrix_original(ad, common_genes, use_log1p=USE_LOG1P)
            mean_list.append(np.mean(X, axis=0))
            detect_list.append(np.mean(X > 0, axis=0))

    mean_mat = np.row_stack(mean_list)
    detect_mat = np.row_stack(detect_list)

    mean_score = np.nanmean(mean_mat, axis=0)
    var_score = np.nanvar(mean_mat, axis=0)
    detect_score = np.nanmean(detect_mat, axis=0)

    keep = detect_score >= float(min_detect_frac)
    score = var_score * np.log1p(mean_score + 1e-8)

    df = pd.DataFrame({
        "gene": common_genes,
        "detect": detect_score,
        "mean": mean_score,
        "var": var_score,
        "score": score,
        "keep": keep,
    })
    df = df[df["keep"]].sort_values("score", ascending=False).head(max_genes)
    return df["gene"].tolist()


def make_domain_palette(domains: Sequence[str]):
    base = sns.color_palette("tab20", n_colors=max(20, len(domains)))
    return {d: base[i % len(base)] for i, d in enumerate(domains)}


# ============================================================
# METRIC 1: DOMAIN-BASED NEIGHBORHOOD COMPONENT AGREEMENT
# ============================================================

def compute_knn_indices(coords: np.ndarray, k: int):
    n = coords.shape[0]
    if n < 2:
        return np.zeros((n, 0), dtype=int)

    nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="auto")
    nn.fit(coords)
    idx = nn.kneighbors(coords, return_distance=False)
    idx = idx[:, 1:]
    return idx


def compute_focal_domain_neighborhood_matrix(ad, domain_order: Sequence[str], knn_k: int = KNN_K):
    coords = get_coords(ad)
    labels = get_domains(ad)
    D = len(domain_order)

    dom_to_idx = {str(d): i for i, d in enumerate(domain_order)}
    label_idx = np.array([dom_to_idx.get(str(x), -1) for x in labels], dtype=int)

    idx = compute_knn_indices(coords, knn_k)
    k_eff = idx.shape[1]

    comp = np.zeros((len(labels), D), dtype=np.float32)

    if k_eff > 0:
        for i in range(len(labels)):
            neigh_lab = label_idx[idx[i]]
            valid = neigh_lab >= 0
            if np.sum(valid) == 0:
                continue
            counts = np.bincount(neigh_lab[valid], minlength=D).astype(np.float32)
            comp[i] = counts / counts.sum()

    neigh_mat = np.full((D, D), np.nan, dtype=np.float32)
    for d_idx, d in enumerate(domain_order):
        m = label_idx == d_idx
        if np.sum(m) > 0:
            neigh_mat[d_idx] = np.mean(comp[m], axis=0)

    return neigh_mat


def build_real_group_component_reference(real_slices: Dict[float, str], selected_domains: Sequence[str]):
    mats = []
    real_cache = {}

    for t, p in real_slices.items():
        ad = read_h5ad(p)
        mat = compute_focal_domain_neighborhood_matrix(ad, selected_domains, knn_k=KNN_K)
        mats.append(mat)
        real_cache[float(t)] = mat

    mat_stack = np.stack(mats, axis=0) if len(mats) > 0 else np.zeros((0, len(selected_domains), len(selected_domains)))
    return {
        "mean_matrix": np.nanmean(mat_stack, axis=0) if len(mats) > 0 else np.full((len(selected_domains), len(selected_domains)), np.nan, dtype=np.float32),
        "stack": mat_stack,
        "real_cache": real_cache,
        "real_times": sorted(real_slices.keys()),
    }


def compute_domain_component_agreement_to_real_group(
    real_slices: Dict[float, str],
    methods: Dict[str, Dict[float, str]],
    selected_domains: Sequence[str],
):
    """
    One score per slice x focal_domain.
    Real uses leave-one-out reference.
    Virtual uses pooled real reference.
    """
    ref = build_real_group_component_reference(real_slices, selected_domains)
    rows = []

    # Real baseline: leave-one-out
    real_times = ref["real_times"]
    stack = ref["stack"]

    for i, t in enumerate(real_times):
        query_mat = ref["real_cache"][float(t)]

        if len(real_times) > 1:
            keep = [j for j in range(len(real_times)) if j != i]
            ref_mat = np.nanmean(stack[keep], axis=0)
        else:
            ref_mat = ref["mean_matrix"]

        for d_idx, dom in enumerate(selected_domains):
            r = ref_mat[d_idx]
            q = query_mat[d_idx]
            good = np.isfinite(r) & np.isfinite(q)
            if np.sum(good) == 0:
                score = np.nan
            else:
                l1 = np.sum(np.abs(r[good] - q[good]))
                score = 1.0 - 0.5 * l1

            rows.append({
                "group": "Real",
                "slice_id": f"real_{t:g}",
                "time": float(t),
                "domain": str(dom),
                "component_agreement": score,
            })

    # Virtual groups
    ref_mat = ref["mean_matrix"]
    for group, mapping in methods.items():
        for t, p in mapping.items():
            ad = read_h5ad(p)
            query_mat = compute_focal_domain_neighborhood_matrix(ad, selected_domains, knn_k=KNN_K)

            for d_idx, dom in enumerate(selected_domains):
                r = ref_mat[d_idx]
                q = query_mat[d_idx]
                good = np.isfinite(r) & np.isfinite(q)
                if np.sum(good) == 0:
                    score = np.nan
                else:
                    l1 = np.sum(np.abs(r[good] - q[good]))
                    score = 1.0 - 0.5 * l1

                rows.append({
                    "group": group,
                    "slice_id": f"{group}_{t:g}",
                    "time": float(t),
                    "domain": str(dom),
                    "component_agreement": score,
                })

    return pd.DataFrame(rows, columns=[
        "group", "slice_id", "time", "domain", "component_agreement"
    ])


# ============================================================
# METRIC 2: ADJACENT DOMAIN GENE-PROFILE CORRELATION
# ============================================================

def compute_domain_gene_summary_grouped(
    real_slices: Dict[float, str],
    methods: Dict[str, Dict[float, str]],
    selected_domains: Sequence[str],
    genes: Sequence[str],
):
    rows = []

    # Real
    for t, p in real_slices.items():
        ad = read_h5ad(p)
        labels = get_domains(ad)
        X = extract_gene_matrix_original(ad, genes, use_log1p=USE_LOG1P)

        for dom in selected_domains:
            m = labels == str(dom)
            mean_vec = np.mean(X[m], axis=0).astype(np.float32) if np.sum(m) > 0 else np.full(len(genes), np.nan, dtype=np.float32)

            for g, mv in zip(genes, mean_vec):
                rows.append({
                    "group": "Real",
                    "time": float(t),
                    "domain": str(dom),
                    "gene": str(g),
                    "mean_logexpr": float(mv) if np.isfinite(mv) else np.nan,
                })

    # Virtual
    for group, mapping in methods.items():
        for t, p in mapping.items():
            ad = read_h5ad(p)
            labels = get_domains(ad)
            X = extract_gene_matrix_original(ad, genes, use_log1p=USE_LOG1P)

            for dom in selected_domains:
                m = labels == str(dom)
                mean_vec = np.mean(X[m], axis=0).astype(np.float32) if np.sum(m) > 0 else np.full(len(genes), np.nan, dtype=np.float32)

                for g, mv in zip(genes, mean_vec):
                    rows.append({
                        "group": group,
                        "time": float(t),
                        "domain": str(dom),
                        "gene": str(g),
                        "mean_logexpr": float(mv) if np.isfinite(mv) else np.nan,
                    })

    return pd.DataFrame(rows)


def build_augmented_time_sequence(method_mapping: Dict[float, str], real_slices: Dict[float, str]):
    """
    Sequence for one virtual method:
    use real slice at real time points, and virtual slice at virtual-only time points.
    If a time exists in both, prefer real.
    """
    all_times = sorted(set(method_mapping.keys()) | set(real_slices.keys()))
    seq = []
    for t in all_times:
        if t in real_slices:
            seq.append({
                "time": float(t),
                "source_group": "Real",
                "is_real": True,
                "label": f"{t:g}(real)",
            })
        else:
            seq.append({
                "time": float(t),
                "source_group": None,   # filled by caller
                "is_real": False,
                "label": f"{t:g}",
            })
    return seq


def compute_adjacent_domain_gene_profile_agreement_with_real_anchors(
    gene_summary_df: pd.DataFrame,
    real_slices: Dict[float, str],
    methods: Dict[str, Dict[float, str]],
    selected_domains: Sequence[str],
):
    """
    For each virtual method:
      combine real anchor times + virtual times
      compute adjacent pseudobulk correlation by domain
    """
    rows = []

    for group, mapping in methods.items():
        seq = build_augmented_time_sequence(mapping, real_slices)

        # fill source group for virtual-only entries
        for item in seq:
            if not item["is_real"]:
                item["source_group"] = group

        for dom in selected_domains:
            mats = {}
            for item in seq:
                t = float(item["time"])
                src = item["source_group"]

                dft = gene_summary_df[
                    (gene_summary_df["group"] == src) &
                    (gene_summary_df["time"] == t) &
                    (gene_summary_df["domain"] == str(dom))
                ].sort_values("gene")

                mats[t] = dft["mean_logexpr"].to_numpy(dtype=np.float32)

            for i in range(len(seq) - 1):
                a = seq[i]
                b = seq[i + 1]
                t0 = float(a["time"])
                t1 = float(b["time"])

                x0 = mats[t0]
                x1 = mats[t1]

                good = np.isfinite(x0) & np.isfinite(x1)
                if np.sum(good) < 5:
                    corr = np.nan
                else:
                    corr = float(np.corrcoef(x0[good], x1[good])[0, 1])

                rows.append({
                    "group": group,
                    "domain": str(dom),
                    "t0": t0,
                    "t1": t1,
                    "pair": f"{a['label']}->{b['label']}",
                    "adjacent_gene_profile_corr": corr,
                })

    return pd.DataFrame(rows, columns=[
        "group", "domain", "t0", "t1", "pair", "adjacent_gene_profile_corr"
    ])


# ============================================================
# STORAGE
# ============================================================

def summarize_and_store():
    ensure_outdir(OUTDIR)

    selected_domains = get_selected_domains(REAL_SLICES, METHODS)
    selected_genes = select_genes_for_metrics(
        REAL_SLICES,
        METHODS,
        manual_genes=MANUAL_GENES,
        max_genes=MAX_GENES_FOR_METRICS,
        min_detect_frac=MIN_GLOBAL_DETECTION_FRAC,
    )

    print(f"[INFO] selected domains: {selected_domains}")
    print(f"[INFO] selected genes: {len(selected_genes)}")

    save_json(
        {
            "selected_domains": selected_domains,
            "selected_genes_count": len(selected_genes),
            "real_slices": {str(t): p for t, p in REAL_SLICES.items()},
            "methods": {m: {str(t): p for t, p in mp.items()} for m, mp in METHODS.items()},
            "use_log1p": USE_LOG1P,
            "knn_k": KNN_K,
        },
        os.path.join(OUTDIR, "config_summary.json"),
    )
    save_text_lines(selected_domains, os.path.join(OUTDIR, "selected_domains.txt"))
    save_text_lines(selected_genes, os.path.join(OUTDIR, "selected_genes.txt"))

    # Metric 1
    comp_df = compute_domain_component_agreement_to_real_group(
        REAL_SLICES,
        METHODS,
        selected_domains,
    )
    comp_df.to_csv(
        os.path.join(OUTDIR, "group_domain_component_agreement.csv"),
        index=False,
    )

    # Metric 2
    gene_summary_df = compute_domain_gene_summary_grouped(
        REAL_SLICES,
        METHODS,
        selected_domains,
        selected_genes,
    )
    gene_summary_df.to_csv(
        os.path.join(OUTDIR, "group_domain_gene_summary.csv.gz"),
        index=False,
        compression="gzip",
    )

    adj_df = compute_adjacent_domain_gene_profile_agreement_with_real_anchors(
        gene_summary_df,
        REAL_SLICES,
        METHODS,
        selected_domains,
    )
    adj_df.to_csv(
        os.path.join(OUTDIR, "group_adjacent_domain_gene_profile_agreement_with_real_anchors.csv"),
        index=False,
    )

    print(f"[DONE] summary results saved to: {OUTDIR}")


# ============================================================
# PLOTTING STYLE
# ============================================================

def set_pub_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.14,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_or_show(fig, save_path=None):
    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[OK] Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================
# PLOTTING: METRIC 1
# ============================================================

def plot_domain_component_agreement(comp_df: pd.DataFrame, save_path=None):
    if len(comp_df) == 0:
        print("[WARN] component agreement table empty; skip plot")
        return

    set_pub_style()

    # sort groups by mean score, descending
    order = (
        comp_df.groupby("group", as_index=False)["component_agreement"]
        .mean()
        .sort_values("component_agreement", ascending=False)["group"]
        .tolist()
    )

    domain_order = sorted(comp_df["domain"].unique().tolist())
    domain_pal = make_domain_palette(domain_order)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    sns.boxplot(
        data=comp_df,
        x="group",
        y="component_agreement",
        order=order,
        palette=GROUP_PALETTE,
        showfliers=False,
        width=0.58,
        linewidth=1.2,
        ax=ax,
    )

    xmap = {g: i for i, g in enumerate(order)}
    rng = np.random.default_rng(0)

    for _, r in comp_df.iterrows():
        if pd.isnull(r["component_agreement"]):
            continue
        x = xmap[r["group"]] + rng.uniform(-0.12, 0.12)
        ax.scatter(
            x,
            r["component_agreement"],
            s=28,
            color=domain_pal[r["domain"]],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )

    ax.set_title("Domain neighborhood component agreement")
    ax.set_xlabel("")
    ax.set_ylabel("Agreement to real-slice group")
    ax.grid(axis="y", alpha=0.16)

    group_handles = [Patch(facecolor=GROUP_PALETTE[g], edgecolor="black", label=g) for g in order]
    domain_handles = [Line2D([0], [0], marker="o", linestyle="", color=domain_pal[d], label=d, markersize=6)
                      for d in domain_order]

    fig.legend(
        handles=group_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.43, 1.02),
        ncol=len(order),
    )
    fig.legend(
        handles=domain_handles,
        frameon=False,
        loc="center right",
        bbox_to_anchor=(1.06, 0.5),
        title="Domain",
    )

    fig.tight_layout(rect=[0, 0, 0.93, 0.96])
    save_or_show(fig, save_path)


# ============================================================
# PLOTTING: METRIC 2
# ============================================================

def plot_adjacent_domain_gene_profile_by_interval(adj_df: pd.DataFrame, save_path=None):
    if len(adj_df) == 0:
        print("[WARN] adjacent correlation table empty; skip plot")
        return

    set_pub_style()

    domain_order = sorted(adj_df["domain"].unique().tolist())
    domain_pal = make_domain_palette(domain_order)

    # keep method comparison only
    plot_df = adj_df[adj_df["group"].isin(["SpatialZ", "AXIS"])].copy()
    pair_order = plot_df["pair"].drop_duplicates().tolist()

    fig, ax = plt.subplots(figsize=(12.8, 5.0))

    sns.boxplot(
        data=plot_df,
        x="pair",
        y="adjacent_gene_profile_corr",
        hue="group",
        hue_order=["SpatialZ", "AXIS"],
        palette={"SpatialZ": GROUP_PALETTE["SpatialZ"], "AXIS": GROUP_PALETTE["AXIS"]},
        showfliers=False,
        linewidth=1.2,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()

    rng = np.random.default_rng(1)
    pair_to_x = {p: i for i, p in enumerate(pair_order)}
    group_offset = {"SpatialZ": -0.18, "AXIS": 0.18}

    for _, r in plot_df.iterrows():
        if pd.isnull(r["adjacent_gene_profile_corr"]):
            continue
        x = pair_to_x[r["pair"]] + group_offset[r["group"]] + rng.uniform(-0.06, 0.06)
        ax.scatter(
            x,
            r["adjacent_gene_profile_corr"],
            s=28,
            color=domain_pal[r["domain"]],
            alpha=0.86,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )

    ax.set_title("Adjacent domain gene-profile correlation by interval")
    ax.set_xlabel("")
    ax.set_ylabel("Correlation")
    ax.set_ylim(0.95, 1.0)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.16)

    method_handles = [
        Patch(facecolor=GROUP_PALETTE["SpatialZ"], edgecolor="black", label="SpatialZ"),
        Patch(facecolor=GROUP_PALETTE["AXIS"], edgecolor="black", label="AXIS"),
    ]
    domain_handles = [Line2D([0], [0], marker="o", linestyle="", color=domain_pal[d], label=d, markersize=6)
                      for d in domain_order]

    fig.legend(
        handles=method_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.34, 1.03),
        ncol=2,
    )
    fig.legend(
        handles=domain_handles,
        frameon=False,
        loc="center right",
        bbox_to_anchor=(1.06, 0.5),
        title="Domain",
    )

    fig.tight_layout(rect=[0, 0, 0.93, 0.96])
    save_or_show(fig, save_path)


# ============================================================
# LOAD + PLOT
# ============================================================

def load_saved_results():
    comp_df = safe_read_csv(
        os.path.join(OUTDIR, "group_domain_component_agreement.csv"),
        columns=["group", "slice_id", "time", "domain", "component_agreement"],
    )
    adj_df = safe_read_csv(
        os.path.join(OUTDIR, "group_adjacent_domain_gene_profile_agreement_with_real_anchors.csv"),
        columns=["group", "domain", "t0", "t1", "pair", "adjacent_gene_profile_corr"],
    )
    cfg = load_json(os.path.join(OUTDIR, "config_summary.json"))
    return comp_df, adj_df, cfg


def plot_all():
    figdir = os.path.join(OUTDIR, "figures")
    ensure_outdir(figdir)

    comp_df, adj_df, cfg = load_saved_results()

    plot_domain_component_agreement(
        comp_df,
        save_path=os.path.join(figdir, "domain_component_agreement.png"),
    )
    #
    # plot_adjacent_domain_gene_profile_by_interval(
    #     adj_df,
    #     save_path=os.path.join(figdir, "adjacent_domain_gene_profile_by_interval.png"),
    # )

    print(f"[DONE] figures saved to: {figdir}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="plot",
        choices=["summarize", "plot", "all"],
    )
    args = parser.parse_args()

    if args.mode == "summarize":
        summarize_and_store()
    elif args.mode == "plot":
        plot_all()
    else:
        summarize_and_store()
        plot_all()


if __name__ == "__main__":
    main()