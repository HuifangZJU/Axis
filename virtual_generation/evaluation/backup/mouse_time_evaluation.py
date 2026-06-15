#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare AXIS vs SpatialZ virtual slices on:
1) domain continuity
2) gene variation continuity

Two-stage workflow:
    python compare_virtual_methods.py --mode summarize
    python compare_virtual_methods.py --mode plot
    python compare_virtual_methods.py --mode all

Edit CONFIG below with your file paths.

Assumptions
-----------
- Each slice is an h5ad
- spatial coordinates in ad.obsm["spatial"]
- spatial domains in ad.obs["axis_joint_domain"]
- genes in ad.var_names
- virtual slices already exist for each method

Author: ChatGPT
"""

import os
import json
import gzip
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.neighbors import NearestNeighbors


# ============================================================
# CONFIG: EDIT THIS PART
# ============================================================

# Output folder for all saved results and figures
OUTDIR = "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/evaluation"

# Methods and time->file mappings
# Replace these with your real paths.
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

# Optional real slices, only used for gene selection support or future extensions
REAL_SLICES = {
    2.5: "/media/huifang/data/registration/xenium/mouse_brain/h5ads/Tg/0_0.h5ad",
    5.7: "/media/huifang/data/registration/xenium/mouse_brain/h5ads/Tg/0_1.h5ad",
    17.9: "/media/huifang/data/registration/xenium/mouse_brain/h5ads/Tg/0_2.h5ad",
}

# AnnData keys
COORD_KEY = "spatial"
DOMAIN_KEY = "axis_joint_domain"

# Preprocessing
USE_LOG1P = True
FILTER_VALID = True
VALID_KEY = "valid"

# Domain analysis
SELECTED_DOMAINS: Optional[List[str]] = ['8','11'] # None -> use all observed domains
KNN_K = 8

# Gene analysis
MANUAL_GENES: Optional[List[str]] = None
MAX_GENES_FOR_METRICS = 300
MIN_GLOBAL_DETECTION_FRAC = 0.01

# Plotting aesthetics
FIG_DPI = 300
METHOD_PALETTE = {
    "AXIS": "#d45a45",
    "SpatialZ": "#8fa0b8",
}
DISPLAY_Y_QUANTILE = 0.98


# ============================================================
# IO / BASIC HELPERS
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
    vals = np.asarray(vals, dtype=object)
    return vals


def extract_gene_matrix_original(ad, genes: Sequence[str], use_log1p=True):
    """
    Original extraction behavior:
    X[:, idx] -> dense -> float32 -> optional log1p
    """
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


def get_all_common_genes(methods: Dict[str, Dict[float, str]], real_slices: Optional[Dict[float, str]] = None):
    common = None

    for mname, mapping in methods.items():
        for _, p in mapping.items():
            ad = read_h5ad(p)
            genes = set(ad.var_names.tolist())
            common = genes if common is None else common.intersection(genes)
    if real_slices is not None:
        for _, p in real_slices.items():
            ad = read_h5ad(p)
            genes = set(ad.var_names.tolist())

            common = genes if common is None else common.intersection(genes)

    if common is None:
        return []
    return sorted(common)


def get_selected_domains(methods: Dict[str, Dict[float, str]]) -> List[str]:
    if SELECTED_DOMAINS is not None:
        return [str(x) for x in SELECTED_DOMAINS]

    ds = set()
    for _, mapping in methods.items():
        for _, p in mapping.items():
            ad = read_h5ad(p)
            ds.update(np.unique(get_domains(ad)))
    ds = sorted([d for d in ds if str(d).lower() != "nan"])
    return ds


def select_genes_for_metrics(
    methods: Dict[str, Dict[float, str]],
    real_slices: Optional[Dict[float, str]] = None,
    manual_genes: Optional[List[str]] = None,
    max_genes: int = MAX_GENES_FOR_METRICS,
    min_detect_frac: float = MIN_GLOBAL_DETECTION_FRAC,
):
    if manual_genes is not None:
        return list(manual_genes)

    common_genes = get_all_common_genes(methods, real_slices)
    if len(common_genes) == 0:
        return []

    mean_list = []
    detect_list = []

    for _, mapping in methods.items():
        for _, p in mapping.items():
            ad = read_h5ad(p)
            X = extract_gene_matrix_original(ad, common_genes, use_log1p=USE_LOG1P)
            mean_list.append(np.mean(X, axis=0))
            detect_list.append(np.mean(X > 0, axis=0))

    if real_slices is not None:
        for _, p in real_slices.items():
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
    df = df[df["keep"]].sort_values("score", ascending=False)
    df = df.head(max_genes)

    return df["gene"].tolist()


# ============================================================
# DOMAIN CONTINUITY
# ============================================================

def compute_domain_proportions(labels: np.ndarray, domain_order: Sequence[str]) -> np.ndarray:
    out = np.zeros(len(domain_order), dtype=np.float32)
    n = len(labels)
    if n == 0:
        return out
    for i, d in enumerate(domain_order):
        out[i] = np.mean(labels == str(d))
    return out


def compute_domain_adjacency_matrix(
    coords: np.ndarray,
    labels: np.ndarray,
    domain_order: Sequence[str],
    knn_k: int = KNN_K,
) -> np.ndarray:
    """
    Build a domain-domain adjacency frequency matrix from spot kNN graph.
    Matrix is normalized to sum to 1.
    """
    n = coords.shape[0]
    D = len(domain_order)
    A = np.zeros((D, D), dtype=np.float32)

    if n < max(3, knn_k + 1):
        return A

    nbrs = NearestNeighbors(n_neighbors=min(knn_k + 1, n), algorithm="auto")
    nbrs.fit(coords)
    _, idx = nbrs.kneighbors(coords)
    idx = idx[:, 1:]  # remove self

    dom_to_idx = {str(d): i for i, d in enumerate(domain_order)}

    for i in range(n):
        di = dom_to_idx.get(str(labels[i]), None)
        if di is None:
            continue
        for j in idx[i]:
            dj = dom_to_idx.get(str(labels[j]), None)
            if dj is None:
                continue
            A[di, dj] += 1.0

    # symmetrize
    A = 0.5 * (A + A.T)
    s = A.sum()
    if s > 0:
        A /= s
    return A


def compute_domain_slice_summaries(
    methods: Dict[str, Dict[float, str]],
    selected_domains: Sequence[str],
):
    """
    Per-slice summaries:
      - domain proportions
      - domain adjacency matrix
    """
    prop_rows = []
    adj_store = {}

    for method, mapping in methods.items():
        times = sorted(mapping.keys())
        for t in times:
            ad = read_h5ad(mapping[t])
            coords = get_coords(ad)
            labels = get_domains(ad)

            prop = compute_domain_proportions(labels, selected_domains)
            A = compute_domain_adjacency_matrix(coords, labels, selected_domains, knn_k=KNN_K)

            for dom, val in zip(selected_domains, prop):
                prop_rows.append({
                    "method": method,
                    "time": float(t),
                    "domain": str(dom),
                    "proportion": float(val),
                })

            adj_store[f"{method}|{t:.3f}"] = A.tolist()

    prop_df = pd.DataFrame(prop_rows)
    return prop_df, adj_store


def compute_domain_step_metrics(
    prop_df: pd.DataFrame,
    adj_store: Dict[str, List[List[float]]],
    selected_domains: Sequence[str],
):
    """
    Adjacent-step metrics + overall roughness.
    """
    step_rows = []
    overall_rows = []

    methods = sorted(prop_df["method"].unique().tolist())

    for method in methods:
        dfm = prop_df[prop_df["method"] == method].copy()
        times = sorted(dfm["time"].unique().tolist())

        prop_mat = []
        adj_mat = []

        for t in times:
            p = (
                dfm[dfm["time"] == t]
                .set_index("domain")
                .reindex(selected_domains)["proportion"]
                .to_numpy(dtype=np.float32)
            )
            prop_mat.append(p)

            A = np.asarray(adj_store[f"{method}|{t:.3f}"], dtype=np.float32)
            adj_mat.append(A)

        prop_mat = np.stack(prop_mat, axis=0)          # (T, D)
        adj_mat = np.stack(adj_mat, axis=0)            # (T, D, D)

        # adjacent shifts
        for i in range(len(times) - 1):
            t0, t1 = times[i], times[i + 1]
            comp_shift = 0.5 * np.sum(np.abs(prop_mat[i + 1] - prop_mat[i]))
            adj_shift = 0.5 * np.sum(np.abs(adj_mat[i + 1] - adj_mat[i]))

            step_rows.append({
                "method": method,
                "t0": t0,
                "t1": t1,
                "pair": f"{t0:g}->{t1:g}",
                "composition_shift": float(comp_shift),
                "adjacency_shift": float(adj_shift),
            })

        # roughness
        if len(times) >= 3:
            comp_rough = []
            adj_rough = []
            for i in range(1, len(times) - 1):
                comp_r = 0.5 * np.sum(np.abs(prop_mat[i + 1] - 2 * prop_mat[i] + prop_mat[i - 1]))
                adj_r = 0.5 * np.sum(np.abs(adj_mat[i + 1] - 2 * adj_mat[i] + adj_mat[i - 1]))
                comp_rough.append(comp_r)
                adj_rough.append(adj_r)

            overall_rows.append({
                "method": method,
                "composition_roughness": float(np.mean(comp_rough)) if len(comp_rough) else np.nan,
                "adjacency_roughness": float(np.mean(adj_rough)) if len(adj_rough) else np.nan,
                "composition_shift_mean": float(np.mean([r["composition_shift"] for r in step_rows if r["method"] == method])),
                "adjacency_shift_mean": float(np.mean([r["adjacency_shift"] for r in step_rows if r["method"] == method])),
            })
        else:
            overall_rows.append({
                "method": method,
                "composition_roughness": np.nan,
                "adjacency_roughness": np.nan,
                "composition_shift_mean": float(np.mean([r["composition_shift"] for r in step_rows if r["method"] == method])),
                "adjacency_shift_mean": float(np.mean([r["adjacency_shift"] for r in step_rows if r["method"] == method])),
            })

    return pd.DataFrame(step_rows), pd.DataFrame(overall_rows)


# ============================================================
# GENE CONTINUITY
# ============================================================

def compute_domain_gene_summary(
    methods: Dict[str, Dict[float, str]],
    selected_domains: Sequence[str],
    genes: Sequence[str],
):
    """
    For each method, time, domain, gene:
      - mean log-expression
      - detection frequency
    """
    rows = []

    for method, mapping in methods.items():
        times = sorted(mapping.keys())
        for t in times:
            ad = read_h5ad(mapping[t])
            labels = get_domains(ad)

            X = extract_gene_matrix_original(ad, genes, use_log1p=USE_LOG1P)

            for dom in selected_domains:
                m = labels == str(dom)
                if np.sum(m) == 0:
                    mean_vec = np.full(len(genes), np.nan, dtype=np.float32)
                    det_vec = np.full(len(genes), np.nan, dtype=np.float32)
                else:
                    Xd = X[m]
                    mean_vec = np.mean(Xd, axis=0).astype(np.float32)
                    det_vec = np.mean(Xd > 0, axis=0).astype(np.float32)

                for g, mv, dv in zip(genes, mean_vec, det_vec):
                    rows.append({
                        "method": method,
                        "time": float(t),
                        "domain": str(dom),
                        "gene": str(g),
                        "mean_logexpr": float(mv) if np.isfinite(mv) else np.nan,
                        "detect_freq": float(dv) if np.isfinite(dv) else np.nan,
                    })

    return pd.DataFrame(rows)


def compute_gene_continuity_metrics(gene_df: pd.DataFrame):
    """
    For each method/domain/gene:
      - mean_adj_shift
      - mean_roughness
      - detect_adj_shift
      - detect_roughness
    """
    rows = []

    methods = sorted(gene_df["method"].unique().tolist())
    domains = sorted(gene_df["domain"].unique().tolist())

    for method in methods:
        for dom in domains:
            dfd = gene_df[(gene_df["method"] == method) & (gene_df["domain"] == dom)].copy()
            genes = sorted(dfd["gene"].unique().tolist())

            for gene in genes:
                dfg = dfd[dfd["gene"] == gene].sort_values("time")
                y_mean = dfg["mean_logexpr"].to_numpy(dtype=np.float32)
                y_det = dfg["detect_freq"].to_numpy(dtype=np.float32)

                good_m = np.isfinite(y_mean)
                good_d = np.isfinite(y_det)

                mean_adj = np.nan
                mean_rough = np.nan
                det_adj = np.nan
                det_rough = np.nan

                if np.sum(good_m) >= 2:
                    ym = y_mean[good_m]
                    mean_adj = float(np.mean(np.abs(np.diff(ym))))
                    if len(ym) >= 3:
                        mean_rough = float(np.mean(np.abs(ym[2:] - 2 * ym[1:-1] + ym[:-2])))

                if np.sum(good_d) >= 2:
                    yd = y_det[good_d]
                    det_adj = float(np.mean(np.abs(np.diff(yd))))
                    if len(yd) >= 3:
                        det_rough = float(np.mean(np.abs(yd[2:] - 2 * yd[1:-1] + yd[:-2])))

                rows.append({
                    "method": method,
                    "domain": dom,
                    "gene": gene,
                    "mean_adj_shift": mean_adj,
                    "mean_roughness": mean_rough,
                    "detect_adj_shift": det_adj,
                    "detect_roughness": det_rough,
                })

    return pd.DataFrame(rows)


def compute_domain_pseudobulk_adjcorr(gene_df: pd.DataFrame):
    """
    For each method/domain and adjacent time pair:
      correlation across gene mean vectors
    """
    rows = []

    methods = sorted(gene_df["method"].unique().tolist())
    domains = sorted(gene_df["domain"].unique().tolist())

    for method in methods:
        for dom in domains:
            dfd = gene_df[(gene_df["method"] == method) & (gene_df["domain"] == dom)].copy()
            times = sorted(dfd["time"].unique().tolist())
            if len(times) < 2:
                continue

            mats = {}
            for t in times:
                dft = dfd[dfd["time"] == t].sort_values("gene")
                mats[t] = dft["mean_logexpr"].to_numpy(dtype=np.float32)

            for i in range(len(times) - 1):
                t0, t1 = times[i], times[i + 1]
                v0, v1 = mats[t0], mats[t1]
                good = np.isfinite(v0) & np.isfinite(v1)

                if np.sum(good) < 5:
                    corr = np.nan
                else:
                    corr = float(np.corrcoef(v0[good], v1[good])[0, 1])

                rows.append({
                    "method": method,
                    "domain": dom,
                    "t0": t0,
                    "t1": t1,
                    "pair": f"{t0:g}->{t1:g}",
                    "pseudobulk_adjcorr": corr,
                })

    return pd.DataFrame(rows)


# ============================================================
# STORAGE
# ============================================================

def summarize_and_store():
    ensure_outdir(OUTDIR)

    selected_domains = get_selected_domains(METHODS)
    genes = select_genes_for_metrics(
        METHODS,
        real_slices=REAL_SLICES if len(REAL_SLICES) > 0 else None,
        manual_genes=MANUAL_GENES,
    )

    print(f"[INFO] selected domains: {selected_domains}")
    print(f"[INFO] selected genes: {len(genes)}")

    save_json(
        {
            "selected_domains": selected_domains,
            "selected_genes_count": len(genes),
            "methods": {k: {str(t): v for t, v in vv.items()} for k, vv in METHODS.items()},
            "real_slices": {str(t): v for t, v in REAL_SLICES.items()},
            "use_log1p": USE_LOG1P,
        },
        os.path.join(OUTDIR, "config_summary.json"),
    )
    save_text_lines(genes, os.path.join(OUTDIR, "selected_genes.txt"))

    # Domain summaries
    prop_df, adj_store = compute_domain_slice_summaries(METHODS, selected_domains)
    prop_df.to_csv(os.path.join(OUTDIR, "slice_domain_proportions.csv"), index=False)
    save_json(adj_store, os.path.join(OUTDIR, "slice_domain_adjacency.json"))

    step_df, overall_df = compute_domain_step_metrics(prop_df, adj_store, selected_domains)
    step_df.to_csv(os.path.join(OUTDIR, "domain_step_metrics.csv"), index=False)
    overall_df.to_csv(os.path.join(OUTDIR, "domain_overall_metrics.csv"), index=False)

    # Gene summaries
    gene_df = compute_domain_gene_summary(METHODS, selected_domains, genes)
    gene_df.to_csv(os.path.join(OUTDIR, "domain_gene_summary.csv.gz"), index=False, compression="gzip")

    gene_metric_df = compute_gene_continuity_metrics(gene_df)
    gene_metric_df.to_csv(os.path.join(OUTDIR, "gene_continuity_metrics.csv"), index=False)

    pb_df = compute_domain_pseudobulk_adjcorr(gene_df)
    pb_df.to_csv(os.path.join(OUTDIR, "domain_pseudobulk_adjcorr.csv"), index=False)

    print(f"[DONE] Saved summaries to: {OUTDIR}")


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
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.16,
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
# PLOTTING: DOMAIN CONTINUITY
# ============================================================

def plot_domain_step_curves(step_df: pd.DataFrame, save_path=None):
    set_pub_style()

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.8), sharex=True)
    metrics = [
        ("composition_shift", "Domain composition shift"),
        ("adjacency_shift", "Domain adjacency shift"),
    ]

    order_pairs = step_df["pair"].drop_duplicates().tolist()

    for ax, (metric, ylabel) in zip(axes, metrics):
        for method in sorted(step_df["method"].unique()):
            dfm = step_df[step_df["method"] == method].copy()
            dfm["pair"] = pd.Categorical(dfm["pair"], categories=order_pairs, ordered=True)
            dfm = dfm.sort_values("pair")

            ax.plot(
                np.arange(len(dfm)),
                dfm[metric].values,
                marker="o",
                markersize=6,
                linewidth=2.2,
                color=METHOD_PALETTE[method],
                label=method,
            )

        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)
        ax.set_title(ylabel)

    axes[-1].set_xticks(np.arange(len(order_pairs)))
    axes[-1].set_xticklabels(order_pairs, rotation=20, ha="right")
    axes[-1].set_xlabel("Adjacent virtual interval")

    handles = [Line2D([0], [0], color=METHOD_PALETTE[m], lw=2.5, marker="o", label=m)
               for m in sorted(step_df["method"].unique())]
    fig.legend(handles=handles, loc="upper right", frameon=False)

    fig.suptitle("Spatial-domain continuity across virtual time", y=0.995, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_or_show(fig, save_path)


def plot_domain_overall_metrics(overall_df: pd.DataFrame, save_path=None):
    set_pub_style()

    long_rows = []
    for _, r in overall_df.iterrows():
        for k in ["composition_shift_mean", "adjacency_shift_mean", "composition_roughness", "adjacency_roughness"]:
            long_rows.append({
                "method": r["method"],
                "metric": k,
                "value": r[k],
            })
    plot_df = pd.DataFrame(long_rows)

    metric_label = {
        "composition_shift_mean": "Mean composition shift",
        "adjacency_shift_mean": "Mean adjacency shift",
        "composition_roughness": "Composition roughness",
        "adjacency_roughness": "Adjacency roughness",
    }
    plot_df["metric_label"] = plot_df["metric"].map(metric_label)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    sns.barplot(
        data=plot_df,
        x="metric_label",
        y="value",
        hue="method",
        palette=METHOD_PALETTE,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Overall domain continuity summary")
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    save_or_show(fig, save_path)


# ============================================================
# PLOTTING: GENE CONTINUITY
# ============================================================

def _plot_method_metric_boxplots_horizontal(
    df: pd.DataFrame,
    metrics: Sequence[str],
    title: str,
    save_path=None,
):
    set_pub_style()

    order = ["AXIS", "SpatialZ"]
    metric_label = {
        "mean_adj_shift": "Mean-expression adjacent shift",
        "mean_roughness": "Mean-expression roughness",
        "detect_adj_shift": "Detection-frequency adjacent shift",
        "detect_roughness": "Detection-frequency roughness",
    }

    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(4.4 * len(metrics), 4.8),
        squeeze=False
    )
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        dfm = df[np.isfinite(df[metric])].copy()

        # sort methods by median for that metric
        med = dfm.groupby("method")[metric].median().sort_values()
        order_local = med.index.tolist()

        sns.boxplot(
            data=dfm,
            y="method",
            x=metric,
            order=order_local,
            palette=METHOD_PALETTE,
            showfliers=False,
            width=0.55,
            linewidth=1.2,
            ax=ax,
            orient="h",
        )

        sns.stripplot(
            data=dfm,
            y="method",
            x=metric,
            order=order_local,
            color="0.35",
            size=3.2,
            alpha=0.28,
            jitter=0.16,
            ax=ax,
            orient="h",
        )

        ax.set_title(metric_label.get(metric, metric))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.18)

    fig.suptitle(title, y=0.995, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_or_show(fig, save_path)


def plot_gene_continuity_boxplots(gene_metric_df: pd.DataFrame, save_path=None):
    _plot_method_metric_boxplots_horizontal(
        gene_metric_df,
        metrics=["mean_adj_shift", "mean_roughness", "detect_adj_shift", "detect_roughness"],
        title="Domain-wise gene variation continuity",
        save_path=save_path,
    )


def plot_domain_pseudobulk_adjcorr(pb_df: pd.DataFrame, save_path=None):
    set_pub_style()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), squeeze=False)
    axes = axes[0]

    # left: by pair
    sns.boxplot(
        data=pb_df,
        x="pair",
        y="pseudobulk_adjcorr",
        hue="method",
        palette=METHOD_PALETTE,
        showfliers=False,
        ax=axes[0],
    )
    sns.stripplot(
        data=pb_df,
        x="pair",
        y="pseudobulk_adjcorr",
        hue="method",
        dodge=True,
        palette=METHOD_PALETTE,
        alpha=0.35,
        size=3,
        linewidth=0,
        ax=axes[0],
    )
    axes[0].set_title("Adjacent pseudobulk correlation by interval")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()

    # right: overall by method
    sns.boxplot(
        data=pb_df,
        x="method",
        y="pseudobulk_adjcorr",
        order=["SpatialZ", "AXIS"],
        palette=METHOD_PALETTE,
        showfliers=False,
        ax=axes[1],
    )
    sns.stripplot(
        data=pb_df,
        x="method",
        y="pseudobulk_adjcorr",
        order=["SpatialZ", "AXIS"],
        color="0.35",
        alpha=0.30,
        size=3.5,
        jitter=0.12,
        linewidth=0,
        ax=axes[1],
    )
    axes[1].set_title("Overall domain pseudobulk continuity")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    handles = [Patch(facecolor=METHOD_PALETTE[m], edgecolor="black", label=m) for m in ["AXIS", "SpatialZ"]]
    fig.legend(handles=handles, frameon=False, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_or_show(fig, save_path)


# ============================================================
# OPTIONAL: HEATMAPS / TRAJECTORIES
# ============================================================

def plot_domain_heatmap(prop_df: pd.DataFrame, save_path=None):
    set_pub_style()

    methods = sorted(prop_df["method"].unique().tolist())
    domains = sorted(prop_df["domain"].unique().tolist())
    times = sorted(prop_df["time"].unique().tolist())

    fig, axes = plt.subplots(1, len(methods), figsize=(5.8 * len(methods), 4.6), squeeze=False)
    axes = axes[0]

    for ax, method in zip(axes, methods):
        dfm = prop_df[prop_df["method"] == method].copy()
        mat = (
            dfm.pivot(index="domain", columns="time", values="proportion")
            .reindex(index=domains, columns=times)
        )
        sns.heatmap(
            mat,
            cmap="mako",
            linewidths=0.15,
            linecolor="white",
            cbar=True,
            ax=ax,
        )
        ax.set_title(method)
        ax.set_xlabel("Virtual time")
        ax.set_ylabel("Domain")

    fig.suptitle("Domain composition over virtual time", y=0.995, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_or_show(fig, save_path)


def plot_selected_gene_trajectories(
    gene_df: pd.DataFrame,
    top_n_genes: int = 12,
    selected_domain: Optional[str] = None,
    save_path=None,
):
    set_pub_style()

    df = gene_df.copy()
    if selected_domain is not None:
        df = df[df["domain"] == str(selected_domain)].copy()

    # pick top variable genes across all methods/times within chosen domain scope
    var_df = (
        df.groupby("gene", as_index=False)["mean_logexpr"]
        .agg(lambda x: np.nanvar(x.to_numpy(dtype=np.float32)))
        .rename(columns={"mean_logexpr": "var"})
        .sort_values("var", ascending=False)
        .head(top_n_genes)
    )
    genes = var_df["gene"].tolist()

    ncol = 4
    nrow = int(np.ceil(len(genes) / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.8 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, gene in zip(axes, genes):
        dfg = df[df["gene"] == gene].copy()
        for method in sorted(dfg["method"].unique()):
            sub = dfg[dfg["method"] == method].groupby("time", as_index=False)["mean_logexpr"].mean()
            ax.plot(
                sub["time"], sub["mean_logexpr"],
                color=METHOD_PALETTE[method],
                linewidth=2.0,
                marker="o",
                markersize=4.5,
                label=method,
            )
        ax.set_title(gene)
        ax.set_xlabel("Time")
        ax.set_ylabel("")
        ax.grid(axis="y", alpha=0.14)

    for ax in axes[len(genes):]:
        ax.axis("off")

    handles = [Line2D([0], [0], color=METHOD_PALETTE[m], lw=2.5, marker="o", label=m)
               for m in sorted(df["method"].unique())]
    fig.legend(handles=handles, frameon=False, loc="upper right")

    main_title = "Selected gene trajectories"
    if selected_domain is not None:
        main_title += f" in domain {selected_domain}"
    fig.suptitle(main_title, y=0.995, fontsize=18)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_or_show(fig, save_path)


# ============================================================
# LOAD SAVED RESULTS
# ============================================================

def load_saved_results():
    prop_df = pd.read_csv(os.path.join(OUTDIR, "slice_domain_proportions.csv"))
    step_df = pd.read_csv(os.path.join(OUTDIR, "domain_step_metrics.csv"))
    overall_df = pd.read_csv(os.path.join(OUTDIR, "domain_overall_metrics.csv"))
    gene_df = pd.read_csv(os.path.join(OUTDIR, "domain_gene_summary.csv.gz"))
    gene_metric_df = pd.read_csv(os.path.join(OUTDIR, "gene_continuity_metrics.csv"))
    pb_df = pd.read_csv(os.path.join(OUTDIR, "domain_pseudobulk_adjcorr.csv"))
    cfg = load_json(os.path.join(OUTDIR, "config_summary.json"))
    return prop_df, step_df, overall_df, gene_df, gene_metric_df, pb_df, cfg


def plot_all():
    ensure_outdir(os.path.join(OUTDIR, "figures"))
    prop_df, step_df, overall_df, gene_df, gene_metric_df, pb_df, cfg = load_saved_results()

    plot_domain_step_curves(
        step_df,
        save_path=os.path.join(OUTDIR, "figures", "domain_step_curves.png"),
    )

    plot_domain_overall_metrics(
        overall_df,
        save_path=os.path.join(OUTDIR, "figures", "domain_overall_metrics.png"),
    )

    plot_gene_continuity_boxplots(
        gene_metric_df,
        save_path=os.path.join(OUTDIR, "figures", "gene_continuity_boxplots.png"),
    )

    plot_domain_pseudobulk_adjcorr(
        pb_df,
        save_path=os.path.join(OUTDIR, "figures", "domain_pseudobulk_adjcorr.png"),
    )

    plot_domain_heatmap(
        prop_df,
        save_path=os.path.join(OUTDIR, "figures", "domain_composition_heatmap.png"),
    )

    # optional trajectory figure: pick first domain if available
    domains = cfg["selected_domains"]
    if len(domains) > 0:
        plot_selected_gene_trajectories(
            gene_df,
            top_n_genes=12,
            selected_domain=domains[0],
            save_path=os.path.join(OUTDIR, "figures", f"selected_gene_trajectories_domain_{domains[0]}.png"),
        )

    print(f"[DONE] Figures saved to: {os.path.join(OUTDIR, 'figures')}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["summarize", "plot", "all"],
        help="summarize: compute/store metrics; plot: load saved results and plot; all: both",
    )
    args = parser.parse_args()

    if args.mode == "summarize":
        summarize_and_store()
    elif args.mode == "plot":
        plot_all()
    elif args.mode == "all":
        summarize_and_store()
        plot_all()


if __name__ == "__main__":
    main()