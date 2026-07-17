import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

# ============================================================
# CONFIG
# ============================================================
ROOT = "/media/huifang/data/registration/xenium/mouse_brain"
OUTDIR = os.path.join(ROOT, "virtual_tissue/AD_time_grid256_spatial2.0/analysis/differential/analysis")
os.makedirs(OUTDIR, exist_ok=True)


COORD_KEY = "spatial"
DOMAIN_KEY = "axis_joint_domain"

# ------------------------------------------------------------
# Differential maps
# IMPORTANT:
# The file names suggest "5.700_minus_17.900" and "2.500_minus_5.700".
# If your intended biological interpretation is the opposite direction,
# set sign_flip = -1 below.
# ------------------------------------------------------------
CONTRASTS = {
    "17p9_minus_5p7": {
        "diff_path": os.path.join(
            ROOT, "virtual_tissue", "AD_time_grid256_spatial2.0", "analysis", "differential",
            "diff_forward_17.900_minus_5.700.h5ad"
        ),
        "rank_csv": os.path.join(ROOT, "haotian", "Tg", "res23.csv"),
        "title": "17.9m − 5.7m",
        "sign_flip": 1.0,
    },
    "5p7_minus_2p5": {
        "diff_path": os.path.join(
            ROOT, "virtual_tissue", "AD_time_grid256_spatial2.0", "analysis", "differential",
            "diff_forward_5.700_minus_2.500.h5ad"
        ),
        "rank_csv": os.path.join(ROOT, "haotian", "Tg", "res12.csv"),
        "title": "5.7m − 2.5m",
        "sign_flip": 1.0,
    },
}
# ------------------------------------------------------------
# Domain focus
# ------------------------------------------------------------
FOCUS_DOMAIN = "11"
SECONDARY_DOMAINS = ["1", "4", "6", "8", "10"]
ALL_SELECTED_DOMAINS = [FOCUS_DOMAIN] + SECONDARY_DOMAINS

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
INVERT_X = False
INVERT_Y = False

FIG_DPI = 300
POINT_SIZE_MAP = 1
POINT_SIZE_OVERVIEW = 1
BG_ALPHA = 0.06
MASKED_ALPHA = 0.90

# Diverging cmap for differential values
DIFF_CMAP = "coolwarm"

# domain colors for reference domain map
DOMAIN_COLORS = {
    "1":  "#4E79A7",
    "4":  "#F28E2B",
    "6":  "#59A14F",
    "8":  "#9C755F",
    "10": "#B07AA1",
    "11": "#8CD17D",   # green hippocampus focus
}

# ------------------------------------------------------------
# Optional manual case-study genes
# If None, automatically use top-ranked genes from domain 11
# ------------------------------------------------------------
CASE_STUDY_GENES: Optional[List[str]] = ["Gfap","Arc"]
TOP_N_GENES_PER_PANEL = 6


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
# IO HELPERS
# ============================================================
def read_h5ad(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return sc.read_h5ad(path)


def read_rank_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df


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
    return ad.obs[domain_key].astype(str).to_numpy()


def extract_gene_vector(ad, gene: str):
    if gene not in ad.var_names:
        raise KeyError(f"Gene '{gene}' not found in ad.var_names")
    j = int(np.where(ad.var_names == gene)[0][0])
    v = ad.X[:, j]
    if sp.issparse(v):
        v = np.asarray(v.toarray()).ravel().astype(np.float32)
    else:
        v = np.asarray(v).ravel().astype(np.float32)
    return v


def normalize_domain_column(df: pd.DataFrame):
    out = df.copy()
    out["domain"] = out["domain"].astype(str)
    out["gene"] = out["gene"].astype(str)
    return out


# ============================================================
# DIFFERENTIAL LOADING
# ============================================================
def load_contrast_data(contrast_cfg):
    ad = read_h5ad(contrast_cfg["diff_path"])
    rank_df = normalize_domain_column(read_rank_csv(contrast_cfg["rank_csv"]))
    return ad, rank_df


def load_all_contrasts(contrast_dict):
    data = {}
    for key, cfg in contrast_dict.items():
        ad, rank_df = load_contrast_data(cfg)
        data[key] = {
            "adata": ad,
            "rank_df": rank_df,
            "title": cfg["title"],
            "sign_flip": float(cfg.get("sign_flip", 1.0)),
        }
    return data


# ============================================================
# GENE SELECTION
# ============================================================
def top_ranked_genes_for_domain(rank_df: pd.DataFrame, domain: str, top_n=10):
    dfm = rank_df[rank_df["domain"] == str(domain)].copy()
    if dfm.empty:
        return []
    # already ranked by pval then fstat according to your note
    genes = dfm["gene"].astype(str).tolist()
    # keep unique in ranked order
    seen = set()
    out = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out[:top_n]


def build_union_case_study_genes(all_data, focus_domain=FOCUS_DOMAIN, top_n=TOP_N_GENES_PER_PANEL):
    if CASE_STUDY_GENES is not None:
        return CASE_STUDY_GENES

    genes = []
    for key, item in all_data.items():
        genes += top_ranked_genes_for_domain(item["rank_df"], focus_domain, top_n=top_n)

    seen = set()
    out = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out[:top_n]


# ============================================================
# SPATIAL PLOTTING HELPERS
# ============================================================
def get_diff_gene_vector(ad, gene, sign_flip=1.0):
    vals = extract_gene_vector(ad, gene)
    return sign_flip * vals


def robust_sym_vmax(vals, q=99.0):
    vals = np.asarray(vals, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 1.0
    vmax = np.percentile(np.abs(vals), q)
    return max(float(vmax), 1e-6)


def apply_axis_orientation(ax):
    if INVERT_X:
        ax.invert_xaxis()
    if INVERT_Y:
        ax.invert_yaxis()


def plot_domain_reference_map(
    ad,
    selected_domains=ALL_SELECTED_DOMAINS,
    save_path=None,
):
    """
    Option 0: domain reference figure to show where domain 11 and selected domains lie.
    """
    set_style()

    coords = get_coords(ad)
    labels = get_domain_labels(ad)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    # background
    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=POINT_SIZE_OVERVIEW,
        c="#D9D9D9",
        alpha=0.20,
        linewidths=0,
        rasterized=True,
    )

    handles = []
    for dom in selected_domains:
        m = labels == str(dom)
        if np.sum(m) == 0:
            continue
        color = DOMAIN_COLORS.get(str(dom), plt.get_cmap("tab20")(int(dom) % 20))
        ax.scatter(
            coords[m, 0], coords[m, 1],
            s=POINT_SIZE_OVERVIEW,
            c=[color],
            alpha=0.85,
            linewidths=0,
            rasterized=True,
        )
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color,
                   markeredgecolor="none", markersize=7, label=f"Domain {dom}")
        )

    ax.set_title("Selected anatomical domains for spatial differential analysis")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    apply_axis_orientation(ax)
    ax.legend(handles=handles, frameon=False, loc="upper right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_top_gene_bar(
    rank_df: pd.DataFrame,
    domain: str,
    title: str,
    top_n=12,
    score_col="logpval",
    save_path=None,
):
    """
    Option 1: compact bar plot of top spatially variable genes in one domain.
    """
    set_style()

    dfm = rank_df[rank_df["domain"] == str(domain)].copy()
    if dfm.empty:
        print(f"[WARN] No ranking rows for domain {domain}")
        return

    # already ranked, but keep unique genes in order
    dfm = dfm.drop_duplicates(subset=["gene"], keep="first").head(top_n)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    y = np.arange(len(dfm))

    vals = dfm[score_col].to_numpy(dtype=np.float32)
    genes = dfm["gene"].astype(str).tolist()

    ax.barh(y, vals, color="#6C8EBF", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(genes)
    ax.invert_yaxis()
    ax.set_xlabel(score_col)
    ax.set_title(f"Top spatially variable genes | Domain {domain} | {title}")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_spatial_small_multiples(
    ad,
    rank_df: pd.DataFrame,
    domain: str,
    title: str,
    sign_flip=1.0,
    genes: Optional[Sequence[str]] = None,
    top_n=6,
    q=99.0,
    save_path=None,
):
    """
    Option 2: main reviewer-facing panel.
    Show top genes spatially, restricted to one focus domain, with tissue background.
    """
    set_style()

    coords = get_coords(ad)
    labels = get_domain_labels(ad)

    if genes is None:
        genes = top_ranked_genes_for_domain(rank_df, domain, top_n=top_n)

    if len(genes) == 0:
        print(f"[WARN] No genes to plot for domain {domain}")
        return

    mask_dom = labels == str(domain)
    if np.sum(mask_dom) == 0:
        print(f"[WARN] Domain {domain} not present in differential map")
        return

    # determine shared symmetric color scale across selected genes in this panel
    all_vals = []
    valid_genes = []
    for gene in genes:
        if gene not in ad.var_names:
            continue
        vals = get_diff_gene_vector(ad, gene, sign_flip=sign_flip)
        all_vals.append(vals[mask_dom])
        valid_genes.append(gene)

    if len(valid_genes) == 0:
        print("[WARN] None of the selected genes are present in differential map")
        return

    vmax = robust_sym_vmax(np.concatenate(all_vals), q=q)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    n = len(valid_genes)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 4.2 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, gene in zip(axes, valid_genes):
        vals = get_diff_gene_vector(ad, gene, sign_flip=sign_flip)

        # background tissue
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=POINT_SIZE_MAP,
            c="#D9D9D9",
            alpha=BG_ALPHA,
            linewidths=0,
            rasterized=True,
        )

        # focused domain only
        ax.scatter(
            coords[mask_dom, 0], coords[mask_dom, 1],
            s=POINT_SIZE_MAP,
            c=vals[mask_dom],
            cmap=DIFF_CMAP,
            norm=norm,
            alpha=MASKED_ALPHA,
            linewidths=0,
            rasterized=True,
        )

        ax.set_title(gene, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        apply_axis_orientation(ax)

    for ax in axes[len(valid_genes):]:
        ax.axis("off")

    fig.suptitle(f"Spatial differential maps in domain {domain} | {title}", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.95, 0.98])

    cax = fig.add_axes([0.96, 0.22, 0.015, 0.56])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=DIFF_CMAP), cax=cax)
    cb.set_label("Differential value", fontsize=11)

    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_two_contrast_case_gene(
    all_data: Dict[str, Dict],
    gene: str,
    domain: str = FOCUS_DOMAIN,
    q=99.0,
    save_path=None,
):
    """
    Option 3: same gene across both contrasts, domain-restricted spatial maps.
    """
    set_style()

    keys = list(all_data.keys())
    if len(keys) != 2:
        raise ValueError("This function expects exactly two contrasts.")

    # shared scale across both maps
    all_vals = []
    for key in keys:
        ad = all_data[key]["adata"]
        if gene not in ad.var_names:
            continue
        labels = get_domain_labels(ad)
        mask = labels == str(domain)
        vals = get_diff_gene_vector(ad, gene, sign_flip=all_data[key]["sign_flip"])
        all_vals.append(vals[mask])

    if len(all_vals) == 0:
        print(f"[WARN] Gene {gene} not available.")
        return

    vmax = robust_sym_vmax(np.concatenate(all_vals), q=q)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3), squeeze=False)
    axes = axes[0]

    for ax, key in zip(axes, keys):
        ad = all_data[key]["adata"]
        coords = get_coords(ad)
        labels = get_domain_labels(ad)
        mask = labels == str(domain)

        if gene not in ad.var_names:
            ax.axis("off")
            continue

        vals = get_diff_gene_vector(ad, gene, sign_flip=all_data[key]["sign_flip"])

        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=POINT_SIZE_MAP,
            c="#D9D9D9",
            alpha=BG_ALPHA,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=POINT_SIZE_MAP,
            c=vals[mask],
            cmap=DIFF_CMAP,
            norm=norm,
            alpha=MASKED_ALPHA,
            linewidths=0,
            rasterized=True,
        )

        ax.set_title(all_data[key]["title"], pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        apply_axis_orientation(ax)

    fig.suptitle(f"Case-study spatial differential map: {gene} in domain {domain}", y=0.98, fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

    cax = fig.add_axes([0.93, 0.18, 0.018, 0.62])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=DIFF_CMAP), cax=cax)
    cb.set_label("Differential value", fontsize=11)

    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_domain_gene_score_heatmap(
    all_rankings: Dict[str, pd.DataFrame],
    domains: Sequence[str],
    top_k_per_domain=5,
    score_col="logpval",
    save_path=None,
):
    """
    Option 4: summary heatmap of domain-specific ranked genes across selected domains.
    Uses ranking tables only, so avoids extra spatial comparison artifacts.
    """
    set_style()

    # collect top genes across selected domains and contrasts
    gene_pool = []
    for cname, df in all_rankings.items():
        for dom in domains:
            top_genes = top_ranked_genes_for_domain(df, str(dom), top_n=top_k_per_domain)
            gene_pool.extend(top_genes)

    seen = set()
    genes = []
    for g in gene_pool:
        if g not in seen:
            seen.add(g)
            genes.append(g)

    if len(genes) == 0:
        print("[WARN] No genes found for summary heatmap")
        return

    rows = []
    for cname, df in all_rankings.items():
        for dom in domains:
            dfm = df[df["domain"] == str(dom)].copy()
            score_map = dict(zip(dfm["gene"].astype(str), dfm[score_col].astype(float)))
            row = {"contrast": cname, "domain": str(dom)}
            for g in genes:
                row[g] = score_map.get(g, np.nan)
            rows.append(row)

    mat_df = pd.DataFrame(rows)
    mat_df["row_name"] = mat_df["contrast"] + " | D" + mat_df["domain"]
    mat_df = mat_df.set_index("row_name")

    data = mat_df[genes]

    fig, ax = plt.subplots(figsize=(0.45 * len(genes) + 3.0, 0.55 * data.shape[0] + 1.6))
    sns.heatmap(
        data,
        cmap="magma",
        linewidths=0.15,
        linecolor="white",
        cbar_kws={"label": score_col},
        ax=ax,
    )
    ax.set_title("Ranked spatially variable genes across selected domains and contrasts")
    ax.set_xlabel("Gene")
    ax.set_ylabel("Domain / contrast")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_domain_gene_distributions(
    all_data: Dict[str, Dict],
    genes: Sequence[str],
    domain: str = FOCUS_DOMAIN,
    max_points=12000,
    random_seed=0,
    save_path=None,
):
    """
    Option 5: focused differential-value distributions in one domain across contrasts.
    Useful when you want to show that the selected genes have stronger and more coherent shifts in domain 11.
    """
    set_style()
    rng = np.random.default_rng(random_seed)

    rows = []
    for key, item in all_data.items():
        ad = item["adata"]
        coords = get_coords(ad)
        labels = get_domain_labels(ad)
        mask = labels == str(domain)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        if len(idx) > max_points:
            idx = rng.choice(idx, size=max_points, replace=False)

        for gene in genes:
            if gene not in ad.var_names:
                continue
            vals = get_diff_gene_vector(ad, gene, sign_flip=item["sign_flip"])[idx]
            rows.append(pd.DataFrame({
                "contrast": item["title"],
                "gene": gene,
                "value": vals,
            }))

    if len(rows) == 0:
        print("[WARN] No domain differential distributions to plot.")
        return

    df = pd.concat(rows, axis=0, ignore_index=True)

    fig, axes = plt.subplots(1, len(genes), figsize=(3.8 * len(genes), 4.1), squeeze=False)
    axes = axes[0]

    for ax, gene in zip(axes, genes):
        dfm = df[df["gene"] == gene].copy()
        sns.violinplot(
            data=dfm,
            x="contrast",
            y="value",
            inner="quartile",
            color="#CAD7E8",
            linewidth=1.0,
            cut=0,
            ax=ax,
        )
        sns.stripplot(
            data=dfm,
            x="contrast",
            y="value",
            color="0.30",
            alpha=0.10,
            size=1.2,
            jitter=0.18,
            ax=ax,
        )
        ax.axhline(0, color="0.5", lw=0.8, alpha=0.6)
        ax.set_title(gene, pad=6)
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel(f"Differential value\n(domain {domain})")
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(f"Domain-focused differential distributions in domain {domain}", y=0.99, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    set_style()

    all_data = load_all_contrasts(CONTRASTS)

    # one reference domain map from the first contrast
    first_key = list(all_data.keys())[0]
    ref_ad = all_data[first_key]["adata"]

    plot_domain_reference_map(
        ref_ad,
        selected_domains=ALL_SELECTED_DOMAINS,
        save_path=os.path.join(OUTDIR, "selected_domain_reference_map.png"),
    )

    # top-gene bar plots for focus domain
    for key, item in all_data.items():
        plot_top_gene_bar(
            item["rank_df"],
            domain=FOCUS_DOMAIN,
            title=item["title"],
            top_n=12,
            score_col="logpval",
            save_path=os.path.join(OUTDIR, f"top_genes_domain_{FOCUS_DOMAIN}_{key}.png"),
        )

    # spatial small multiples for focus domain
    case_genes = build_union_case_study_genes(all_data, focus_domain=FOCUS_DOMAIN, top_n=TOP_N_GENES_PER_PANEL)
    print("[INFO] Case-study genes:", case_genes)

    for key, item in all_data.items():
        plot_spatial_small_multiples(
            ad=item["adata"],
            rank_df=item["rank_df"],
            domain=FOCUS_DOMAIN,
            title=item["title"],
            sign_flip=item["sign_flip"],
            genes=case_genes,
            top_n=TOP_N_GENES_PER_PANEL,
            save_path=os.path.join(OUTDIR, f"spatial_multiples_domain_{FOCUS_DOMAIN}_{key}.png"),
        )

    # case-study gene across both contrasts
    if len(case_genes) > 0:
        for gene in case_genes[: min(3, len(case_genes))]:
            plot_two_contrast_case_gene(
                all_data=all_data,
                gene=gene,
                domain=FOCUS_DOMAIN,
                save_path=os.path.join(OUTDIR, f"case_two_contrast_{gene}_domain_{FOCUS_DOMAIN}.png"),
            )

    # summary score heatmap across selected domains
    all_rankings = {k: v["rank_df"] for k, v in all_data.items()}
    plot_domain_gene_score_heatmap(
        all_rankings=all_rankings,
        domains=ALL_SELECTED_DOMAINS,
        top_k_per_domain=5,
        score_col="logpval",
        save_path=os.path.join(OUTDIR, "domain_gene_score_heatmap.png"),
    )

    # domain-focused distributions for selected genes
    if len(case_genes) > 0:
        plot_domain_gene_distributions(
            all_data=all_data,
            genes=case_genes[: min(4, len(case_genes))],
            domain=FOCUS_DOMAIN,
            save_path=os.path.join(OUTDIR, f"domain_{FOCUS_DOMAIN}_gene_distributions.png"),
        )

    print(f"[DONE] Outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()