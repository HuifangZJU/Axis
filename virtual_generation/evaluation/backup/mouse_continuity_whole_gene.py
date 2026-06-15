import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import cKDTree
from matplotlib.lines import Line2D


# ============================================================
# CONFIG
# ============================================================
ROOT = "/media/huifang/data/registration/xenium/mouse_brain"
OUTDIR = os.path.join(ROOT, "virtual_tissue", "evaluation_domain_gene_continuity")
os.makedirs(OUTDIR, exist_ok=True)

COORD_KEY = "spatial"
DOMAIN_KEY = "axis_joint_domain"

REAL_TIME_TO_PATH = {
    2.5: os.path.join(ROOT, "h5ads", "Tg", "0_0.h5ad"),
    5.7: os.path.join(ROOT, "h5ads", "Tg", "0_1.h5ad"),
    17.9: os.path.join(ROOT, "h5ads", "Tg", "0_2.h5ad"),
}

VIRTUAL_TIME_TO_PATH = {
    3.5: os.path.join(ROOT, "virtual_tissue", "Tg_time_grid256", "query", "virtual_Tg_time_3.500.h5ad"),
    7.5: os.path.join(ROOT, "virtual_tissue", "Tg_time_grid256", "query", "virtual_Tg_time_7.500.h5ad"),
    9.5: os.path.join(ROOT, "virtual_tissue", "Tg_time_grid256", "query", "virtual_Tg_time_9.500.h5ad"),
    11.5: os.path.join(ROOT, "virtual_tissue", "Tg_time_grid256", "query", "virtual_Tg_time_11.500.h5ad"),
    13.5: os.path.join(ROOT, "virtual_tissue", "Tg_time_grid256", "query", "virtual_Tg_time_13.500.h5ad"),
    15.5: os.path.join(ROOT, "virtual_tissue", "Tg_time_grid256", "query", "virtual_Tg_time_15.500.h5ad"),
}

# use original extraction logic
USE_LOG1P = True

# domain selection
SELECTED_DOMAINS: Optional[List[str]] = None
# example:
# SELECTED_DOMAINS = ["11", "1", "4", "6", "8", "10"]

# gene selection
MANUAL_GENES: Optional[List[str]] = None
MAX_GENES_FOR_GENEWISE_METRICS = 300
MIN_GLOBAL_DETECTION_FRAC = 0.01
MIN_DOMAIN_SPOTS = 30

# slower optional metrics
COMPUTE_MORAN = False
KNN_K = 8

COMPUTE_GENE_GENE_CORR = True
MAX_GENES_FOR_CORR_METRICS = 80
MIN_STD_FOR_GENE_GENE_CORR = 1e-6

# plotting
FIG_DPI = 300
REAL_COLOR = "#202020"
VIRTUAL_COLOR = "#8A8A8A"


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
    })


# ============================================================
# DATA HELPERS
# ============================================================
def read_h5ad(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return sc.read_h5ad(path)


def load_time_sequence(
    real_map: Dict[float, str],
    virtual_map: Dict[float, str],
):
    seq = []
    for t, p in real_map.items():
        seq.append({"time": float(t), "kind": "real", "path": p, "adata": read_h5ad(p)})
    for t, p in virtual_map.items():
        seq.append({"time": float(t), "kind": "virtual", "path": p, "adata": read_h5ad(p)})
    seq = sorted(seq, key=lambda x: x["time"])
    return seq


def get_coords(ad, coord_key=COORD_KEY):
    if coord_key not in ad.obsm:
        raise KeyError(f"'{coord_key}' not found in ad.obsm")
    coords = np.asarray(ad.obsm[coord_key], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coord shape: {coords.shape}")
    return coords[:, :2]


def get_domain_labels(ad, domain_key=DOMAIN_KEY):
    if domain_key not in ad.obs:
        raise KeyError(f"'{domain_key}' not found in ad.obs")
    return ad.obs[domain_key].astype(str).to_numpy()


def get_common_genes(seq):
    common = None
    for item in seq:
        genes = set(item["adata"].var_names.tolist())
        if common is None:
            common = genes
        else:
            common = common.intersection(genes)
    return sorted(common) if common is not None else []


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


def get_all_domains(seq, domain_key=DOMAIN_KEY):
    ds = set()
    for item in seq:
        ds.update(np.unique(get_domain_labels(item["adata"], domain_key=domain_key)))
    ds = sorted([d for d in ds if d != "nan"])
    return ds


def get_selected_domains(seq):
    if SELECTED_DOMAINS is not None:
        return [str(x) for x in SELECTED_DOMAINS]
    return get_all_domains(seq, domain_key=DOMAIN_KEY)


# ============================================================
# GENE SELECTION
# ============================================================
def select_genes_for_metrics(
    seq,
    max_genes=MAX_GENES_FOR_GENEWISE_METRICS,
    min_global_detect_frac=MIN_GLOBAL_DETECTION_FRAC,
    manual_genes: Optional[Sequence[str]] = MANUAL_GENES,
):
    if manual_genes is not None:
        return list(manual_genes)

    common_genes = get_common_genes(seq)
    if len(common_genes) == 0:
        return []

    mean_list = []
    detect_list = []

    for item in seq:
        ad = item["adata"]
        X = extract_gene_matrix_original(ad, common_genes, use_log1p=USE_LOG1P)
        mean_list.append(np.mean(X, axis=0))
        detect_list.append(np.mean(X > 0, axis=0))

    mean_mat = np.row_stack(mean_list)
    detect_mat = np.row_stack(detect_list)

    mean_score = np.nanmean(mean_mat, axis=0)
    var_score = np.nanvar(mean_mat, axis=0)
    detect_score = np.nanmean(detect_mat, axis=0)

    keep = detect_score >= float(min_global_detect_frac)
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

    if max_genes is not None:
        df = df.head(max_genes)

    return df["gene"].tolist()


# ============================================================
# MORAN'S I
# ============================================================
def moran_i_batch(X: np.ndarray, coords: np.ndarray, k=KNN_K):
    """
    X: (n_spots, n_genes)
    coords: (n_spots, 2)
    """
    n, g = X.shape
    if n < max(4, k + 1):
        return np.full(g, np.nan, dtype=np.float32)

    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=min(k + 1, n))
    if idx.ndim == 1:
        return np.full(g, np.nan, dtype=np.float32)

    idx = idx[:, 1:]  # remove self
    kk = idx.shape[1]
    if kk == 0:
        return np.full(g, np.nan, dtype=np.float32)

    Z = X - np.mean(X, axis=0, keepdims=True)
    denom = np.sum(Z * Z, axis=0)

    neigh_sum = np.zeros_like(Z, dtype=np.float32)
    for j in range(kk):
        neigh_sum += Z[idx[:, j], :]

    num = np.sum(Z * neigh_sum, axis=0)
    W = float(n * kk)

    I = (n / W) * (num / np.maximum(denom, 1e-8))
    I[denom < 1e-8] = np.nan
    return I.astype(np.float32)


# ============================================================
# DOMAIN/TIME SUMMARIES
# ============================================================
def compute_domain_time_summaries(
    seq,
    genes_for_metrics: Sequence[str],
    genes_for_corr: Sequence[str],
    selected_domains: Sequence[str],
):
    """
    Returns
    -------
    summaries:
        dict[(domain, time)] = {
            "kind": kind,
            "mean": (G,),
            "detect": (G,),
            "moran": (G,) or None,
            "pseudobulk": (G,),
            "gene_gene_corr": DataFrame or None,
        }
    """
    summaries = {}

    for item in seq:
        t = item["time"]
        kind = item["kind"]
        ad = item["adata"]
        coords = get_coords(ad, COORD_KEY)
        labels = get_domain_labels(ad, DOMAIN_KEY)

        X_main = extract_gene_matrix_original(ad, genes_for_metrics, use_log1p=USE_LOG1P)

        if len(genes_for_corr) > 0:
            X_corr = extract_gene_matrix_original(ad, genes_for_corr, use_log1p=USE_LOG1P)
        else:
            X_corr = None

        for dom in selected_domains:
            m = labels == str(dom)
            n = int(np.sum(m))
            key = (str(dom), float(t))

            if n < MIN_DOMAIN_SPOTS:
                summaries[key] = {
                    "kind": kind,
                    "n": n,
                    "mean": np.full(len(genes_for_metrics), np.nan, dtype=np.float32),
                    "detect": np.full(len(genes_for_metrics), np.nan, dtype=np.float32),
                    "moran": np.full(len(genes_for_metrics), np.nan, dtype=np.float32) if COMPUTE_MORAN else None,
                    "pseudobulk": np.full(len(genes_for_metrics), np.nan, dtype=np.float32),
                    "gene_gene_corr": None,
                }
                continue

            Xd = X_main[m]
            coords_d = coords[m]

            mean_vec = np.mean(Xd, axis=0).astype(np.float32)
            detect_vec = np.mean(Xd > 0, axis=0).astype(np.float32)
            pseudobulk_vec = mean_vec.copy()

            if COMPUTE_MORAN:
                moran_vec = moran_i_batch(Xd, coords_d, k=KNN_K)
            else:
                moran_vec = None

            # gene-gene correlation on a reduced gene set
            corr_df = None
            if COMPUTE_GENE_GENE_CORR and X_corr is not None:
                Xc = X_corr[m]
                std = np.std(Xc, axis=0)
                keep = np.isfinite(std) & (std > MIN_STD_FOR_GENE_GENE_CORR)
                kept_genes = np.array(genes_for_corr)[keep]

                if np.sum(keep) >= 5:
                    Xc2 = Xc[:, keep]
                    C = np.corrcoef(Xc2, rowvar=False)
                    corr_df = pd.DataFrame(C, index=kept_genes, columns=kept_genes)

            summaries[key] = {
                "kind": kind,
                "n": n,
                "mean": mean_vec,
                "detect": detect_vec,
                "moran": moran_vec,
                "pseudobulk": pseudobulk_vec,
                "gene_gene_corr": corr_df,
            }

    return summaries


# ============================================================
# METRIC COMPUTATION
# ============================================================
def compute_genewise_time_metrics(
    seq,
    selected_domains,
    genes,
    summaries,
    value_key: str,
):
    """
    value_key: one of 'mean', 'detect', 'moran'
    """
    times = [float(item["time"]) for item in seq]
    rows = []

    for dom in selected_domains:
        for gi, gene in enumerate(genes):
            y = []
            kinds = []
            for t in times:
                v = summaries[(str(dom), t)][value_key]
                if v is None:
                    y.append(np.nan)
                else:
                    y.append(v[gi])
                kinds.append(summaries[(str(dom), t)]["kind"])
            y = np.asarray(y, dtype=np.float32)

            good = np.isfinite(y)
            if np.sum(good) < 3:
                continue

            yg = y[good]

            adjacent_shift = np.mean(np.abs(np.diff(yg)))
            roughness = np.mean(np.abs(yg[2:] - 2 * yg[1:-1] + yg[:-2])) if len(yg) >= 3 else np.nan
            span = np.max(yg) - np.min(yg)

            rows.append({
                "domain": str(dom),
                "gene": gene,
                "metric_type": value_key,
                "adjacent_shift": float(adjacent_shift),
                "roughness": float(roughness),
                "span": float(span),
            })

    return pd.DataFrame(rows)


def compute_domain_pseudobulk_adjacent_correlations(
    seq,
    selected_domains,
    summaries,
):
    times = [float(item["time"]) for item in seq]
    rows = []

    for dom in selected_domains:
        for i in range(len(times) - 1):
            t0 = times[i]
            t1 = times[i + 1]

            v0 = summaries[(str(dom), t0)]["pseudobulk"]
            v1 = summaries[(str(dom), t1)]["pseudobulk"]

            good = np.isfinite(v0) & np.isfinite(v1)
            if np.sum(good) < 5:
                corr = np.nan
            else:
                corr = np.corrcoef(v0[good], v1[good])[0, 1]

            rows.append({
                "domain": str(dom),
                "t0": t0,
                "t1": t1,
                "pair": f"{t0:.1f}->{t1:.1f}",
                "corr": float(corr) if np.isfinite(corr) else np.nan,
            })

    return pd.DataFrame(rows)


def compute_domain_gene_gene_corr_continuity(
    seq,
    selected_domains,
    summaries,
):
    times = [float(item["time"]) for item in seq]
    rows = []

    for dom in selected_domains:
        for i in range(len(times) - 1):
            t0 = times[i]
            t1 = times[i + 1]

            C0 = summaries[(str(dom), t0)]["gene_gene_corr"]
            C1 = summaries[(str(dom), t1)]["gene_gene_corr"]

            if C0 is None or C1 is None:
                sim = np.nan
            else:
                common_genes = sorted(set(C0.index).intersection(set(C1.index)))
                if len(common_genes) < 5:
                    sim = np.nan
                else:
                    A = C0.loc[common_genes, common_genes].to_numpy(dtype=np.float32)
                    B = C1.loc[common_genes, common_genes].to_numpy(dtype=np.float32)

                    iu = np.triu_indices_from(A, k=1)
                    a = A[iu]
                    b = B[iu]
                    good = np.isfinite(a) & np.isfinite(b)
                    if np.sum(good) < 10:
                        sim = np.nan
                    else:
                        sim = np.corrcoef(a[good], b[good])[0, 1]

            rows.append({
                "domain": str(dom),
                "t0": t0,
                "t1": t1,
                "pair": f"{t0:.1f}->{t1:.1f}",
                "corr_similarity": float(sim) if np.isfinite(sim) else np.nan,
            })

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING HELPERS
# ============================================================
def domain_palette(domains):
    base = plt.get_cmap("tab20").colors
    return {d: base[i % len(base)] for i, d in enumerate(domains)}


def plot_genewise_violin_panel(
    metric_df_mean: pd.DataFrame,
    metric_df_detect: pd.DataFrame,
    metric_df_moran: Optional[pd.DataFrame],
    save_path=None,
):
    set_style()

    panels = [
        (metric_df_mean, "roughness", "Mean log-expression roughness"),
        (metric_df_mean, "adjacent_shift", "Mean log-expression adjacent shift"),
        (metric_df_detect, "roughness", "Detection-frequency roughness"),
        (metric_df_detect, "adjacent_shift", "Detection-frequency adjacent shift"),
    ]

    if metric_df_moran is not None and len(metric_df_moran) > 0:
        panels.extend([
            (metric_df_moran, "roughness", "Spatial autocorrelation roughness"),
            (metric_df_moran, "adjacent_shift", "Spatial autocorrelation adjacent shift"),
        ])

    n = len(panels)
    ncol = 3 if n > 4 else 2
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow), squeeze=False)
    axes = axes.ravel()

    all_domains = sorted(metric_df_mean["domain"].unique().tolist())
    pal = domain_palette(all_domains)

    for ax, (dfm, metric, title) in zip(axes, panels):
        sns.violinplot(
            data=dfm,
            x="domain",
            y=metric,
            order=all_domains,
            palette=pal,
            inner="quartile",
            linewidth=1.0,
            cut=0,
            ax=ax,
        )
        sns.stripplot(
            data=dfm,
            x="domain",
            y=metric,
            order=all_domains,
            color="0.25",
            alpha=0.18,
            size=1.2,
            jitter=0.18,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Domain")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Domain-wise overall gene continuity", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_domain_adjacent_correlation_boxplots(
    pb_corr_df: pd.DataFrame,
    gg_corr_df: Optional[pd.DataFrame] = None,
    save_path=None,
):
    set_style()

    panels = [(pb_corr_df, "corr", "Domain pseudobulk adjacent correlation")]
    if gg_corr_df is not None and len(gg_corr_df) > 0:
        panels.append((gg_corr_df, "corr_similarity", "Gene–gene correlation continuity"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.4), squeeze=False)
    axes = axes[0]

    all_domains = sorted(pb_corr_df["domain"].unique().tolist())
    pal = domain_palette(all_domains)

    for ax, (dfm, col, title) in zip(axes, panels):
        sns.boxplot(
            data=dfm,
            x="domain",
            y=col,
            order=all_domains,
            palette=pal,
            showfliers=False,
            width=0.55,
            linewidth=1.1,
            ax=ax,
        )
        sns.stripplot(
            data=dfm,
            x="domain",
            y=col,
            order=all_domains,
            color="0.25",
            alpha=0.35,
            size=3,
            jitter=0.15,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Domain")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.16)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_domain_metric_heatmap(
    metric_df_mean: pd.DataFrame,
    metric_df_detect: pd.DataFrame,
    metric_df_moran: Optional[pd.DataFrame],
    pb_corr_df: pd.DataFrame,
    gg_corr_df: Optional[pd.DataFrame],
    save_path=None,
):
    """
    Compact option: aggregate each metric to domain-level median/mean and show heatmap.
    """
    set_style()

    domains = sorted(metric_df_mean["domain"].unique().tolist())
    rows = []

    for d in domains:
        row = {"domain": d}

        tmp = metric_df_mean[metric_df_mean["domain"] == d]
        row["mean_roughness"] = np.nanmedian(tmp["roughness"])
        row["mean_adjshift"] = np.nanmedian(tmp["adjacent_shift"])

        tmp = metric_df_detect[metric_df_detect["domain"] == d]
        row["detect_roughness"] = np.nanmedian(tmp["roughness"])
        row["detect_adjshift"] = np.nanmedian(tmp["adjacent_shift"])

        if metric_df_moran is not None and len(metric_df_moran) > 0:
            tmp = metric_df_moran[metric_df_moran["domain"] == d]
            row["moran_roughness"] = np.nanmedian(tmp["roughness"])
            row["moran_adjshift"] = np.nanmedian(tmp["adjacent_shift"])

        tmp = pb_corr_df[pb_corr_df["domain"] == d]
        row["pseudobulk_corr"] = np.nanmean(tmp["corr"])

        if gg_corr_df is not None and len(gg_corr_df) > 0:
            tmp = gg_corr_df[gg_corr_df["domain"] == d]
            row["gene_gene_corr"] = np.nanmean(tmp["corr_similarity"])

        rows.append(row)

    heat_df = pd.DataFrame(rows).set_index("domain")

    fig, ax = plt.subplots(figsize=(0.95 * heat_df.shape[1] + 2.0, 0.52 * heat_df.shape[0] + 2.2))
    sns.heatmap(
        heat_df,
        cmap="magma",
        linewidths=0.18,
        linecolor="white",
        cbar_kws={"label": "Metric value"},
        ax=ax,
    )
    ax.set_title("Domain-level summary of continuity metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Domain")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_adjacent_correlation_traces(
    pb_corr_df: pd.DataFrame,
    save_path=None,
):
    """
    Option: see if continuity is steady or breaks at specific intervals.
    """
    set_style()

    domains = sorted(pb_corr_df["domain"].unique().tolist())
    pal = domain_palette(domains)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))

    for d in domains:
        dfm = pb_corr_df[pb_corr_df["domain"] == d].copy()
        dfm["mid_t"] = 0.5 * (dfm["t0"] + dfm["t1"])
        ax.plot(dfm["mid_t"], dfm["corr"], color=pal[d], lw=1.6, alpha=0.95)
        ax.scatter(dfm["mid_t"], dfm["corr"], s=34, color=pal[d], edgecolor="black", linewidth=0.3, zorder=3)

    ax.set_xlabel("Midpoint of adjacent interval (months)")
    ax.set_ylabel("Adjacent pseudobulk correlation")
    ax.set_title("Domain-wise adjacent transcriptomic continuity across time")

    handles = [Line2D([0], [0], color=pal[d], lw=2, label=d) for d in domains]
    ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", title="Domain")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    seq = load_time_sequence(REAL_TIME_TO_PATH, VIRTUAL_TIME_TO_PATH)
    print("[INFO] Loaded times:", [x["time"] for x in seq])

    selected_domains = get_selected_domains(seq)
    print("[INFO] Domains:", selected_domains)

    genes_main = select_genes_for_metrics(
        seq,
        max_genes=MAX_GENES_FOR_GENEWISE_METRICS,
        min_global_detect_frac=MIN_GLOBAL_DETECTION_FRAC,
        manual_genes=MANUAL_GENES,
    )
    print(f"[INFO] Genes for gene-wise metrics: {len(genes_main)}")

    if COMPUTE_GENE_GENE_CORR:
        genes_corr = genes_main[: min(MAX_GENES_FOR_CORR_METRICS, len(genes_main))]
    else:
        genes_corr = []
    print(f"[INFO] Genes for gene-gene correlation continuity: {len(genes_corr)}")

    summaries = compute_domain_time_summaries(
        seq=seq,
        genes_for_metrics=genes_main,
        genes_for_corr=genes_corr,
        selected_domains=selected_domains,
    )

    # gene-wise metrics
    metric_df_mean = compute_genewise_time_metrics(
        seq=seq,
        selected_domains=selected_domains,
        genes=genes_main,
        summaries=summaries,
        value_key="mean",
    )
    metric_df_detect = compute_genewise_time_metrics(
        seq=seq,
        selected_domains=selected_domains,
        genes=genes_main,
        summaries=summaries,
        value_key="detect",
    )

    if COMPUTE_MORAN:
        metric_df_moran = compute_genewise_time_metrics(
            seq=seq,
            selected_domains=selected_domains,
            genes=genes_main,
            summaries=summaries,
            value_key="moran",
        )
    else:
        metric_df_moran = pd.DataFrame()

    # domain-level overall metrics
    pb_corr_df = compute_domain_pseudobulk_adjacent_correlations(
        seq=seq,
        selected_domains=selected_domains,
        summaries=summaries,
    )

    if COMPUTE_GENE_GENE_CORR:
        gg_corr_df = compute_domain_gene_gene_corr_continuity(
            seq=seq,
            selected_domains=selected_domains,
            summaries=summaries,
        )
    else:
        gg_corr_df = pd.DataFrame()

    # save tables
    metric_df_mean.to_csv(os.path.join(OUTDIR, "genewise_mean_metrics.csv"), index=False)
    metric_df_detect.to_csv(os.path.join(OUTDIR, "genewise_detect_metrics.csv"), index=False)
    if len(metric_df_moran) > 0:
        metric_df_moran.to_csv(os.path.join(OUTDIR, "genewise_moran_metrics.csv"), index=False)
    pb_corr_df.to_csv(os.path.join(OUTDIR, "domain_pseudobulk_adjacent_corr.csv"), index=False)
    if len(gg_corr_df) > 0:
        gg_corr_df.to_csv(os.path.join(OUTDIR, "domain_gene_gene_corr_continuity.csv"), index=False)

    # figures
    # plot_genewise_violin_panel(
    #     metric_df_mean=metric_df_mean,
    #     metric_df_detect=metric_df_detect,
    #     metric_df_moran=metric_df_moran if len(metric_df_moran) > 0 else None,
    #     save_path=os.path.join(OUTDIR, "genewise_continuity_violin_panel.png"),
    # )

    # plot_domain_adjacent_correlation_boxplots(
    #     pb_corr_df=pb_corr_df,
    #     gg_corr_df=gg_corr_df if len(gg_corr_df) > 0 else None,
    #     save_path=os.path.join(OUTDIR, "domain_adjacent_correlation_boxplots.png"),
    # )

    # plot_domain_metric_heatmap(
    #     metric_df_mean=metric_df_mean,
    #     metric_df_detect=metric_df_detect,
    #     metric_df_moran=metric_df_moran if len(metric_df_moran) > 0 else None,
    #     pb_corr_df=pb_corr_df,
    #     gg_corr_df=gg_corr_df if len(gg_corr_df) > 0 else None,
    #     save_path=os.path.join(OUTDIR, "domain_metric_heatmap.png"),
    # )

    plot_adjacent_correlation_traces(
        pb_corr_df=pb_corr_df,
        save_path=os.path.join(OUTDIR, "adjacent_pseudobulk_correlation_traces.png"),
    )

    print(f"[DONE] Outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()