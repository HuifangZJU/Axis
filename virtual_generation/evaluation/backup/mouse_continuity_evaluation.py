import os
import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


# ============================================================
# CONFIG
# ============================================================
ROOT = "/media/huifang/data/registration/xenium/mouse_brain"
OUTDIR = os.path.join(ROOT, "virtual_tissue", "evaluation_time_continuity")
os.makedirs(OUTDIR, exist_ok=True)

# -------- real / virtual paths --------
REAL_TIME_TO_PATH = {
    2.5: os.path.join(ROOT, "h5ads",  "0_2.5.h5ad"),
    5.7: os.path.join(ROOT, "h5ads",  "0_5.7.h5ad"),
    17.9: os.path.join(ROOT, "h5ads",  "0_17.9.h5ad"),
}

# VIRTUAL_TIME_TO_PATH = {
#     3.3: os.path.join(ROOT, "virtual_tissue", "AD_time_grid256_spatial1.0", "query", "virtual_AD_time_3.300.h5ad"),
#     4.1: os.path.join(ROOT, "virtual_tissue", "AD_time_grid256_spatial1.0", "query", "virtual_AD_time_4.100.h5ad"),
#     4.9: os.path.join(ROOT, "virtual_tissue", "AD_time_grid256_spatial1.0", "query", "virtual_AD_time_4.900.h5ad"),
#     8.75: os.path.join(ROOT, "virtual_tissue", "AD_time_grid256_spatial1.0", "query", "virtual_AD_time_8.750.h5ad"),
#     11.8: os.path.join(ROOT, "virtual_tissue", "AD_time_grid256_spatial1.0", "query", "virtual_AD_time_11.800.h5ad"),
#     14.85: os.path.join(ROOT, "virtual_tissue", "AD_time_grid256_spatial1.0", "query", "virtual_AD_time_14.850.h5ad"),
# }

VIRTUAL_TIME_TO_PATH =  {
    3.3: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/3.3.h5ad",
    4.1: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/4.1.h5ad",
    4.9: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/4.9.h5ad",
    8.75: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/8.75.h5ad",
    11.8: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/11.8.h5ad",
    14.85: "/media/huifang/data/registration/xenium/mouse_brain/virtual_tissue/spatialz/14.85.h5ad",
}


COORD_KEY = "spatial"
DOMAIN_KEY = "axis_joint_domain"

# -------- preprocessing --------
NORMALIZE_TOTAL = False
TARGET_SUM = 1e4
LOG1P = True

# -------- plot preferences --------
REAL_COLOR = "#202020"
VIRTUAL_COLOR = "#8A8A8A"
REAL_MARKER_SIZE = 42
VIRTUAL_MARKER_SIZE = 28

# if None, auto-select dynamic genes for plots
MARKER_GENES: Optional[List[str]] =['Arc','Mbp','Slc17a7','Gfap']

# for ridgeline / violin figures
FOCUS_DOMAIN: Optional[str] = '11'  # e.g. "Hippocampus"
N_AUTO_GENES = 6

# whether to use all domains in domain plots
TOP_N_DOMAINS_FOR_LINE_PLOT = 10

# output dpi
FIG_DPI = 300


# ============================================================
# STYLE
# ============================================================
def set_clean_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.18,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })


# ============================================================
# IO / DATA HELPERS
# ============================================================
def read_one_adata(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ad = sc.read_h5ad(path)
    if "valid" in ad.obs.columns:
        ad = ad[ad.obs["valid"]].copy()
    return ad


def preprocess_adata(ad):
    ad = ad.copy()
    return ad


def load_time_sequence(
    real_map: Dict[float, str],
    virtual_map: Dict[float, str],
):
    seq = []

    for t, p in real_map.items():
        ad = preprocess_adata(read_one_adata(p))
        seq.append({
            "time": float(t),
            "kind": "real",
            "path": p,
            "adata": ad,
        })

    for t, p in virtual_map.items():
        ad = preprocess_adata(read_one_adata(p))
        seq.append({
            "time": float(t),
            "kind": "virtual",
            "path": p,
            "adata": ad,
        })

    seq = sorted(seq, key=lambda x: x["time"])
    return seq


def get_coords(ad, coord_key=COORD_KEY):

    if coord_key not in ad.obsm:
        raise KeyError(f"'{coord_key}' not found in ad.obsm")
    coords = np.asarray(ad.obsm[coord_key], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coordinate shape: {coords.shape}")
    return coords[:, :2]


def get_domain_labels(ad, domain_key=DOMAIN_KEY):
    if domain_key not in ad.obs:
        raise KeyError(f"'{domain_key}' not found in ad.obs")
    labels = ad.obs[domain_key].astype(str).to_numpy()
    return labels


def get_all_domains(seq, domain_key=DOMAIN_KEY):
    ds = set()
    for item in seq:
        ds.update(np.unique(get_domain_labels(item["adata"], domain_key)))
    ds = sorted([d for d in ds if d != "nan"])
    return ds


def get_domain_colors(domains):
    base = list(plt.get_cmap("tab20").colors)
    colors = {dom: base[i % len(base)] for i, dom in enumerate(domains)}
    return colors



def extract_gene_vector(adata, gene):

    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in var_names.")
    j = int(np.where(adata.var_names == gene)[0][0])
    v = adata.X[:, j]
    if sp.issparse(v):
        v = np.asarray(v.toarray()).ravel().astype(np.float32)
    else:
        v = np.asarray(v).ravel().astype(np.float32)

    v = np.log1p(v)

    return v


def available_genes_intersection(seq):
    common = None
    for item in seq:
        genes = set(item["adata"].var_names.tolist())
        if common is None:
            common = genes
        else:
            common = common.intersection(genes)
    return sorted(common) if common is not None else []


# ============================================================
# DOMAIN CONTINUITY TABLES
# ============================================================
def build_domain_composition_table(seq, domain_key=DOMAIN_KEY):
    all_domains = get_all_domains(seq, domain_key=domain_key)
    rows = []

    for item in seq:
        ad = item["adata"]
        labels = get_domain_labels(ad, domain_key)
        n = len(labels)

        vc = pd.Series(labels).value_counts(normalize=True)
        row = {
            "time": item["time"],
            "kind": item["kind"],
        }
        for d in all_domains:
            row[d] = float(vc.get(d, 0.0))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df, all_domains


def build_domain_centroid_table(seq, domain_key=DOMAIN_KEY, coord_key=COORD_KEY):
    rows = []
    all_domains = get_all_domains(seq, domain_key=domain_key)

    for item in seq:
        ad = item["adata"]
        coords = get_coords(ad, coord_key)
        labels = get_domain_labels(ad, domain_key)

        for d in all_domains:
            m = labels == d
            if np.sum(m) == 0:
                rows.append({
                    "time": item["time"],
                    "kind": item["kind"],
                    "domain": d,
                    "cx": np.nan,
                    "cy": np.nan,
                    "n": 0,
                })
            else:
                xy = coords[m]
                rows.append({
                    "time": item["time"],
                    "kind": item["kind"],
                    "domain": d,
                    "cx": float(np.mean(xy[:, 0])),
                    "cy": float(np.mean(xy[:, 1])),
                    "n": int(xy.shape[0]),
                })

    df = pd.DataFrame(rows).sort_values(["domain", "time"]).reset_index(drop=True)
    return df


# ============================================================
# GENE CONTINUITY TABLES
# ============================================================
def build_gene_domain_summary_table(
    seq,
    genes: Sequence[str],
    domains: Optional[Sequence[str]] = None,
    domain_key=DOMAIN_KEY,
):
    if domains is None:
        domains = get_all_domains(seq, domain_key=domain_key)

    rows = []
    for item in seq:
        ad = item["adata"]
        labels = get_domain_labels(ad, domain_key)

        for gene in genes:
            try:
                vals = extract_gene_vector(ad, gene)
            except KeyError:
                continue

            for dom in domains:
                m = labels == dom
                if np.sum(m) == 0:
                    rows.append({
                        "time": item["time"],
                        "kind": item["kind"],
                        "gene": gene,
                        "domain": dom,
                        "mean": np.nan,
                        "median": np.nan,
                        "q25": np.nan,
                        "q75": np.nan,
                        "pct_pos": np.nan,
                        "n": 0,
                    })
                    continue

                v = vals[m]
                rows.append({
                    "time": item["time"],
                    "kind": item["kind"],
                    "gene": gene,
                    "domain": dom,
                    "mean": float(np.mean(v)),
                    "median": float(np.median(v)),
                    "q25": float(np.quantile(v, 0.25)),
                    "q75": float(np.quantile(v, 0.75)),
                    "pct_pos": float(np.mean(v > 0)),
                    "n": int(v.shape[0]),
                })

    df = pd.DataFrame(rows).sort_values(["gene", "domain", "time"]).reset_index(drop=True)
    return df


def build_gene_distribution_long_table(
    seq,
    genes: Sequence[str],
    domains: Optional[Sequence[str]] = None,
    domain_key=DOMAIN_KEY,
    max_cells_per_slice_domain=6000,
    random_seed=0,
):
    """
    Long-form table for violin / ridge plots.
    Randomly subsamples cells per (time, domain) for plotting efficiency.
    """
    rng = np.random.default_rng(random_seed)

    if domains is None:
        domains = get_all_domains(seq, domain_key=domain_key)

    rows = []
    for item in seq:
        ad = item["adata"]
        labels = get_domain_labels(ad, domain_key)

        for dom in domains:
            m_dom = labels == dom
            idx_dom = np.where(m_dom)[0]
            if len(idx_dom) == 0:
                continue

            if len(idx_dom) > max_cells_per_slice_domain:
                idx_dom = rng.choice(idx_dom, size=max_cells_per_slice_domain, replace=False)

            for gene in genes:
                try:
                    vals = extract_gene_vector(ad, gene)[idx_dom]
                except KeyError:
                    continue

                tmp = pd.DataFrame({
                    "time": item["time"],
                    "kind": item["kind"],
                    "domain": dom,
                    "gene": gene,
                    "expr": vals,
                })
                rows.append(tmp)

    if len(rows) == 0:
        return pd.DataFrame(columns=["time", "kind", "domain", "gene", "expr"])
    return pd.concat(rows, axis=0, ignore_index=True)


# ============================================================
# AUTO GENE SELECTION
# ============================================================
def select_time_dynamic_genes(
    seq,
    focus_domain: Optional[str] = None,
    n_top=N_AUTO_GENES,
    domain_key=DOMAIN_KEY,
):
    """
    Automatically choose dynamic genes across time by temporal range of domain-wise mean expression.
    """
    common_genes = available_genes_intersection(seq)
    if len(common_genes) == 0:
        return []

    scores = []
    for gene in common_genes:
        traj = []

        for item in seq:
            ad = item["adata"]
            labels = get_domain_labels(ad, domain_key)
            try:
                vals = extract_gene_vector(ad, gene)
            except KeyError:
                traj = []
                break

            if focus_domain is not None:
                m = labels == focus_domain
                if np.sum(m) == 0:
                    traj.append(np.nan)
                else:
                    traj.append(float(np.mean(vals[m])))
            else:
                traj.append(float(np.mean(vals)))

        traj = np.asarray(traj, dtype=np.float32)
        if np.sum(np.isfinite(traj)) < 3:
            continue

        score = float(np.nanmax(traj) - np.nanmin(traj))
        mean_expr = float(np.nanmean(traj))
        scores.append((gene, score, mean_expr))

    if len(scores) == 0:
        return []

    df = pd.DataFrame(scores, columns=["gene", "range", "mean_expr"])
    df["score"] = df["range"] * np.log1p(df["mean_expr"] + 1e-8)
    df = df.sort_values("score", ascending=False)
    return df["gene"].head(n_top).tolist()


# ============================================================
# TEMPORAL CONTINUITY METRICS
# ============================================================
def compute_gene_temporal_metrics(
    summary_df: pd.DataFrame,
    value_col="median",
):
    """
    Compute continuity metrics from domain-conditioned gene trajectories.
    """
    rows = []
    for (gene, domain), gdf in summary_df.groupby(["gene", "domain"]):
        gdf = gdf.sort_values("time")
        y = gdf[value_col].to_numpy(dtype=np.float32)
        t = gdf["time"].to_numpy(dtype=np.float32)

        good = np.isfinite(y)
        y = y[good]
        t = t[good]
        if len(y) < 3:
            continue

        # adjacent shift
        adj_shift = np.mean(np.abs(np.diff(y)))

        # second-order roughness
        roughness = np.mean(np.abs(y[2:] - 2 * y[1:-1] + y[:-2])) if len(y) >= 3 else np.nan

        # overall range
        span = np.max(y) - np.min(y)

        rows.append({
            "gene": gene,
            "domain": domain,
            "adjacent_shift": float(adj_shift),
            "roughness": float(roughness),
            "span": float(span),
        })

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING: DOMAIN CONTINUITY
# ============================================================
def plot_domain_composition_stacked_area(
    comp_df: pd.DataFrame,
    domains: List[str],
    domain_colors: Dict[str, Tuple[float, float, float]],
    save_path=None,
):
    set_clean_style()

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = comp_df["time"].to_numpy()

    y_stack = np.row_stack([comp_df[d].to_numpy() for d in domains])
    colors = [domain_colors[d] for d in domains]

    ax.stackplot(x, y_stack, labels=domains, colors=colors, alpha=0.88)

    real_df = comp_df[comp_df["kind"] == "real"]
    for _, row in real_df.iterrows():
        ax.axvline(row["time"], color="black", lw=1.1, alpha=0.35, zorder=5)
        ax.text(row["time"], 1.02, f"{row['time']:.1f} m", ha="center", va="bottom", fontsize=10)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Domain proportion")
    ax.set_title("Spatial-domain composition across real and virtual time states")
    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        ncol=1,
        title="Domain"
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_domain_composition_lines(
    comp_df: pd.DataFrame,
    domains: List[str],
    domain_colors: Dict[str, Tuple[float, float, float]],
    top_n=TOP_N_DOMAINS_FOR_LINE_PLOT,
    save_path=None,
):
    set_clean_style()

    mean_prop = {d: comp_df[d].mean() for d in domains}
    show_domains = sorted(domains, key=lambda d: mean_prop[d], reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    for d in show_domains:
        dfm = comp_df[["time", "kind", d]].copy().rename(columns={d: "value"})
        real = dfm[dfm["kind"] == "real"]
        virt = dfm[dfm["kind"] == "virtual"]

        ax.plot(dfm["time"], dfm["value"], color=domain_colors[d], lw=1.6, alpha=0.95)
        ax.scatter(real["time"], real["value"], s=44, color=domain_colors[d], edgecolor="black", linewidth=0.4, zorder=4)
        ax.scatter(virt["time"], virt["value"], s=30, color=domain_colors[d], alpha=0.75, zorder=3)

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Domain proportion")
    ax.set_title("Domain-composition trajectories across time")
    ax.set_xlim(comp_df["time"].min(), comp_df["time"].max())

    handles = [
        Line2D([0], [0], color=domain_colors[d], lw=2, label=d)
        for d in show_domains
    ]
    ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", title="Domain")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_domain_centroid_trajectories(
    centroid_df: pd.DataFrame,
    domains: List[str],
    domain_colors: Dict[str, Tuple[float, float, float]],
    save_path=None,
):
    set_clean_style()

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    axx, axy = axes

    for d in domains:
        gdf = centroid_df[centroid_df["domain"] == d].sort_values("time")
        real = gdf[gdf["kind"] == "real"]
        virt = gdf[gdf["kind"] == "virtual"]

        axx.plot(gdf["time"], gdf["cx"], color=domain_colors[d], lw=1.5, alpha=0.95)
        axx.scatter(real["time"], real["cx"], s=42, color=domain_colors[d], edgecolor="black", linewidth=0.35)
        axx.scatter(virt["time"], virt["cx"], s=26, color=domain_colors[d], alpha=0.75)

        axy.plot(gdf["time"], gdf["cy"], color=domain_colors[d], lw=1.5, alpha=0.95)
        axy.scatter(real["time"], real["cy"], s=42, color=domain_colors[d], edgecolor="black", linewidth=0.35)
        axy.scatter(virt["time"], virt["cy"], s=26, color=domain_colors[d], alpha=0.75)

    axx.set_ylabel("Centroid x")
    axy.set_ylabel("Centroid y")
    axy.set_xlabel("Time (months)")
    axx.set_title("Domain centroid trajectories across time")

    handles = [Line2D([0], [0], color=domain_colors[d], lw=2, label=d) for d in domains]
    axx.legend(handles=handles, frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", title="Domain")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_domain_centroid_paths_2d(
    centroid_df: pd.DataFrame,
    domains: List[str],
    domain_colors: Dict[str, Tuple[float, float, float]],
    save_path=None,
):
    """
    2D spatial centroid movement paths.
    """
    set_clean_style()

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    for d in domains:
        gdf = centroid_df[centroid_df["domain"] == d].sort_values("time")
        ax.plot(gdf["cx"], gdf["cy"], color=domain_colors[d], lw=1.7, alpha=0.95)
        real = gdf[gdf["kind"] == "real"]
        virt = gdf[gdf["kind"] == "virtual"]

        ax.scatter(real["cx"], real["cy"], s=48, color=domain_colors[d], edgecolor="black", linewidth=0.4, zorder=4)
        ax.scatter(virt["cx"], virt["cy"], s=28, color=domain_colors[d], alpha=0.75, zorder=3)

        # annotate only real anchors for readability
        for _, r in real.iterrows():
            if np.isfinite(r["cx"]) and np.isfinite(r["cy"]):
                ax.text(r["cx"], r["cy"], f"{r['time']:.1f}", fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("Centroid x")
    ax.set_ylabel("Centroid y")
    ax.set_title("2D centroid paths of spatial domains across time")
    ax.set_aspect("equal")

    handles = [Line2D([0], [0], color=domain_colors[d], lw=2, label=d) for d in domains]
    ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", title="Domain")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


# ============================================================
# PLOTTING: GENE CONTINUITY
# ============================================================
def plot_gene_domain_ribbons(
    summary_df: pd.DataFrame,
    genes: Sequence[str],
    domains: Sequence[str],
    save_path=None,
):
    """
    Recommended main continuity plot:
    domain-conditioned median + IQR ribbons across time.
    """
    set_clean_style()

    n_rows = len(domains)
    n_cols = len(genes)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.6 * n_cols, 2.8 * n_rows),
        sharex=True,
        squeeze=False,
    )

    for i, dom in enumerate(domains):
        for j, gene in enumerate(genes):
            ax = axes[i, j]
            gdf = summary_df[(summary_df["domain"] == dom) & (summary_df["gene"] == gene)].sort_values("time")
            if gdf.empty:
                ax.axis("off")
                continue

            real = gdf[gdf["kind"] == "real"]
            virt = gdf[gdf["kind"] == "virtual"]

            # whole trajectory
            ax.plot(gdf["time"], gdf["median"], color="#3A6EA5", lw=1.8, alpha=0.95)
            ax.fill_between(
                gdf["time"].to_numpy(),
                gdf["q25"].to_numpy(),
                gdf["q75"].to_numpy(),
                color="#79A9D1",
                alpha=0.28,
                linewidth=0,
            )

            # markers
            ax.scatter(real["time"], real["median"], s=46, color=REAL_COLOR, zorder=4)
            ax.scatter(virt["time"], virt["median"], s=28, color=VIRTUAL_COLOR, zorder=3)

            if i == 0:
                ax.set_title(gene, pad=8)
            if j == 0:
                ax.set_ylabel(dom)
            if i == n_rows - 1:
                ax.set_xlabel("Time (months)")
            ax.set_ylim([0,2])

    fig.suptitle("Domain-conditioned gene-expression continuity", y=0.995, fontsize=16)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_gene_domain_violin_sequences(
    long_df: pd.DataFrame,
    gene: str,
    domains: Sequence[str],
    save_path=None,
):
    """
    Violin option for case-study genes.
    """
    set_clean_style()

    n = len(domains)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.2), squeeze=False)
    axes = axes[0]

    time_order = sorted(long_df["time"].unique().tolist())

    for ax, dom in zip(axes, domains):
        dfm = long_df[(long_df["gene"] == gene) & (long_df["domain"] == dom)].copy()
        if dfm.empty:
            ax.axis("off")
            continue

        sns.violinplot(
            data=dfm,
            x="time",
            y="expr",
            order=time_order,
            inner=None,
            color="#C7D7EA",
            linewidth=1.0,
            cut=0,
            ax=ax,
        )

        real = dfm[dfm["kind"] == "real"]
        virt = dfm[dfm["kind"] == "virtual"]

        sns.stripplot(
            data=real,
            x="time",
            y="expr",
            order=time_order,
            color=REAL_COLOR,
            size=1.4,
            alpha=0.18,
            jitter=0.18,
            ax=ax,
        )
        sns.stripplot(
            data=virt,
            x="time",
            y="expr",
            order=time_order,
            color=VIRTUAL_COLOR,
            size=1.1,
            alpha=0.12,
            jitter=0.18,
            ax=ax,
        )

        ax.set_title(dom)
        ax.set_xlabel("Time (months)")
        if ax is axes[0]:
            ax.set_ylabel(f"{gene} expression")
        else:
            ax.set_ylabel("")

    fig.suptitle(f"Violin continuity view for {gene}", y=1.02, fontsize=16)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_domain_ridgelines(
    long_df: pd.DataFrame,
    domain: str,
    genes: Sequence[str],
    save_path=None,
):
    """
    Ridgeline option for a focus domain (e.g. Hippocampus).
    One subplot per gene, densities offset by time.
    """
    set_clean_style()

    time_order = sorted(long_df["time"].unique().tolist())
    n_genes = len(genes)
    fig, axes = plt.subplots(n_genes, 1, figsize=(8.5, 1.5 * n_genes + 0.6), sharex=False)
    if n_genes == 1:
        axes = [axes]

    for ax, gene in zip(axes, genes):
        dfm = long_df[(long_df["domain"] == domain) & (long_df["gene"] == gene)].copy()
        if dfm.empty:
            ax.axis("off")
            continue

        vmax = np.quantile(dfm["expr"], 0.995)
        xmin = float(np.min(dfm["expr"]))
        xmax = float(max(vmax, xmin + 1e-4))
        xs = np.linspace(xmin, xmax, 240)

        offset_gap = 1.0
        for k, t in enumerate(time_order):
            sub = dfm[dfm["time"] == t]["expr"].to_numpy(dtype=np.float32)
            if len(sub) < 10:
                continue

            try:
                kde = gaussian_kde(sub)
                ys = kde(xs)
            except Exception:
                continue

            ys = ys / (np.max(ys) + 1e-8) * 0.8
            offset = k * offset_gap

            kind = dfm[dfm["time"] == t]["kind"].iloc[0]
            line_color = "#3A6EA5" if kind == "real" else "#7F8EA3"
            fill_alpha = 0.35 if kind == "real" else 0.18
            ls = "-" if kind == "real" else "--"

            ax.fill_between(xs, offset, offset + ys, color=line_color, alpha=fill_alpha)
            ax.plot(xs, offset + ys, color=line_color, lw=1.2, ls=ls)
            ax.text(xmax + 0.02 * (xmax - xmin + 1e-8), offset + 0.10, f"{t:.1f}m",
                    ha="left", va="center", fontsize=9)

        ax.set_yticks([])
        ax.set_ylabel(gene, rotation=0, ha="right", va="center", labelpad=30)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    axes[-1].set_xlabel("Expression")
    fig.suptitle(f"Ridgeline continuity in {domain}", y=0.995, fontsize=16)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_whole_gene_continuity_metrics(
    metrics_df: pd.DataFrame,
    save_path=None,
):
    set_clean_style()

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.3))

    specs = [
        ("roughness", "Temporal roughness"),
        ("adjacent_shift", "Adjacent shift"),
        ("span", "Temporal span"),
    ]

    for ax, (metric, title) in zip(axes, specs):
        sns.boxplot(
            data=metrics_df,
            y=metric,
            color="#D7E3F3",
            width=0.45,
            showfliers=False,
            linewidth=1.1,
            ax=ax,
        )
        sns.stripplot(
            data=metrics_df,
            y=metric,
            color="0.35",
            alpha=0.35,
            size=2.8,
            jitter=0.20,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", alpha=0.18)

    fig.suptitle("Whole-gene temporal continuity metrics", y=1.02, fontsize=16)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    seq = load_time_sequence(REAL_TIME_TO_PATH, VIRTUAL_TIME_TO_PATH)
    print(f"[INFO] Loaded {len(seq)} slices/time states.")
    print("[INFO] Times:", [x["time"] for x in seq])

    all_domains = get_all_domains(seq, domain_key=DOMAIN_KEY)
    domain_colors = get_domain_colors(all_domains)
    print(f"[INFO] Found {len(all_domains)} domains.")

    # --------------------------------------------------------
    # Spatial domain continuity
    # --------------------------------------------------------
    comp_df, domains = build_domain_composition_table(seq, domain_key=DOMAIN_KEY)
    # centroid_df = build_domain_centroid_table(seq, domain_key=DOMAIN_KEY, coord_key=COORD_KEY)
    #
    # plot_domain_composition_stacked_area(
    #     comp_df, domains, domain_colors,
    #     save_path=os.path.join(OUTDIR, "domain_composition_stacked_area.png"),
    # )
    #
    # plot_domain_composition_lines(
    #     comp_df, domains, domain_colors,
    #     top_n=TOP_N_DOMAINS_FOR_LINE_PLOT,
    #     save_path=os.path.join(OUTDIR, "domain_composition_lines.png"),
    # )
    #
    # plot_domain_centroid_trajectories(
    #     centroid_df, domains, domain_colors,
    #     save_path=os.path.join(OUTDIR, "domain_centroid_trajectories.png"),
    # )
    #
    # plot_domain_centroid_paths_2d(
    #     centroid_df, domains, domain_colors,
    #     save_path=os.path.join(OUTDIR, "domain_centroid_paths_2d.png"),
    # )

    # --------------------------------------------------------
    # Gene continuity
    # --------------------------------------------------------
    focus_domain = FOCUS_DOMAIN

    if focus_domain is None and len(all_domains) > 0:
        # use the most frequent domain across sequence as default
        mean_props = {d: comp_df[d].mean() for d in all_domains}
        focus_domain = sorted(all_domains, key=lambda d: mean_props[d], reverse=True)[0]

    genes = MARKER_GENES
    if genes is None:
        genes = select_time_dynamic_genes(
            seq,
            focus_domain=focus_domain,
            n_top=N_AUTO_GENES,
            domain_key=DOMAIN_KEY,
        )
        print(f"[INFO] Auto-selected genes for focus domain '{focus_domain}': {genes}")

    if len(genes) == 0:
        warnings.warn("No genes selected for gene continuity plots. Skipping gene plots.")
        return

    # choose a few domains for ribbon / violin plots
    # ribbon_domains = [focus_domain]
    ribbon_domains=['1','4','6','8','11',]
    # add a couple more common domains if available
    # other_domains = [d for d in all_domains if d != focus_domain]
    # ribbon_domains += other_domains[: min(2, len(other_domains))]

    summary_df = build_gene_domain_summary_table(
        seq,
        genes=genes,
        domains=ribbon_domains,
        domain_key=DOMAIN_KEY,
    )
    summary_df.to_csv(os.path.join(OUTDIR, "gene_domain_summary.csv"), index=False)

    plot_gene_domain_ribbons(
        summary_df=summary_df,
        genes=genes[: min(4, len(genes))],
        domains=ribbon_domains,
        save_path=os.path.join(OUTDIR, "gene_domain_ribbons.png"),
    )

    # violin option using first gene
    long_df = build_gene_distribution_long_table(
        seq,
        genes=genes[: min(4, len(genes))],
        domains=ribbon_domains,
        domain_key=DOMAIN_KEY,
        max_cells_per_slice_domain=5000,
        random_seed=0,
    )
    long_df.to_csv(os.path.join(OUTDIR, "gene_distribution_long.csv"), index=False)

    plot_gene_domain_violin_sequences(
        long_df=long_df,
        gene=genes[0],
        domains=ribbon_domains,
        save_path=os.path.join(OUTDIR, f"gene_violin_{genes[0]}.png"),
    )

    # ridge option for focus domain
    ridge_genes = genes[: min(5, len(genes))]
    plot_domain_ridgelines(
        long_df=long_df,
        domain=focus_domain,
        genes=ridge_genes,
        save_path=os.path.join(OUTDIR, f"ridge_{focus_domain.replace(' ', '_')}.png"),
    )

    # whole-gene continuity metrics
    # use a larger gene set if possible
    metric_genes = select_time_dynamic_genes(
        seq,
        focus_domain=focus_domain,
        n_top=max(30, N_AUTO_GENES),
        domain_key=DOMAIN_KEY,
    )
    metric_summary_df = build_gene_domain_summary_table(
        seq,
        genes=metric_genes,
        domains=[focus_domain],
        domain_key=DOMAIN_KEY,
    )
    metrics_df = compute_gene_temporal_metrics(metric_summary_df, value_col="median")
    metrics_df.to_csv(os.path.join(OUTDIR, "whole_gene_temporal_metrics.csv"), index=False)

    if len(metrics_df) > 0:
        plot_whole_gene_continuity_metrics(
            metrics_df,
            save_path=os.path.join(OUTDIR, "whole_gene_temporal_metrics.png"),
        )

    print(f"[DONE] Outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()