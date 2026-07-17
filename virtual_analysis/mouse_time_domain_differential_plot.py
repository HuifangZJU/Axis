import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm


# ============================================================
# CONFIG
# ============================================================
ROOT = "/media/huifang/data/registration/xenium/mouse_brain/"
OUTDIR = os.path.join(ROOT, "virtual_tissue/AD_time_grid256_spatial2.0/analysis/differential/evaluation")
os.makedirs(OUTDIR, exist_ok=True)

COORD_KEY = "spatial"
DOMAIN_KEY = "axis_joint_domain"

# ------------------------------------------------------------
# Time-consistent contrasts
# EARLY: 5.7 - 2.5
# LATE:  17.9 - 5.7
#
# If your actual file semantics are reversed, swap these paths.
# ------------------------------------------------------------
CONTRASTS = {
    "early_5p7_minus_2p5": {
        "diff_path": os.path.join(
            ROOT, "virtual_tissue", "AD_time_grid256_spatial2.0", "analysis", "differential",
            "diff_forward_5.700_minus_2.500.h5ad"
        ),
        "rank_csv": os.path.join(ROOT, "haotian", "Tg", "res12.csv"),
        "title": "5.7m − 2.5m",
    },
    "late_17p9_minus_5p7": {
        "diff_path": os.path.join(
            ROOT, "virtual_tissue", "AD_time_grid256_spatial2.0", "analysis", "differential",
            "diff_forward_17.900_minus_5.700.h5ad"
        ),
        "rank_csv": os.path.join(ROOT, "haotian", "Tg", "res23.csv"),
        "title": "17.9m − 5.7m",
    },
}

EARLY_KEY = "early_5p7_minus_2p5"
LATE_KEY = "late_17p9_minus_5p7"

# ------------------------------------------------------------
# Domain focus
# ------------------------------------------------------------
FOCUS_DOMAIN = "11"
SECONDARY_DOMAINS = ["1", "4", "6", "8", "10"]
SELECTED_DOMAINS = [FOCUS_DOMAIN] + SECONDARY_DOMAINS

# ------------------------------------------------------------
# Gene selection / thresholds
# ------------------------------------------------------------
TOP_GENES_PER_DOMAIN_PER_CONTRAST = 50
ANNOTATE_TOP_N = 14
PSEUDOTRAJ_TOP_N = 18
HEATMAP_TOP_N = 40

# stability / trend thresholds
MIN_LOGPVAL = 1.3        # about p < 0.05 in -log10 scale if your logpval is -log10(p)
EFFECT_THRESHOLD_MODE = "quantile"  # "quantile" or "fixed"
EFFECT_THRESHOLD_Q = 0.45
FIXED_EFFECT_THRESHOLD = 0.10

# ------------------------------------------------------------
# Figure settings
# ------------------------------------------------------------
FIG_DPI = 300

CLASS_COLORS = {
    "increase": "#D55E00",   # up then up
    "decrease": "#4E79A7",   # down then down
    "peak":     "#E3A008",   # up then down
    "valley":   "#2CA58D",   # down then up
    "stable":   "#B5B5B5",
}

DOMAIN_COLORS = {
    "11": "#7BC96F",  # green hippocampus focus
    "1":  "#4E79A7",
    "4":  "#F28E2B",
    "6":  "#59A14F",
    "8":  "#9C755F",
    "10": "#B07AA1",
}


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
    df["domain"] = df["domain"].astype(str)
    df["gene"] = df["gene"].astype(str)
    return df


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


# ============================================================
# LOAD CONTRASTS
# ============================================================
def load_all_contrasts(contrast_cfg: Dict):
    out = {}
    for key, cfg in contrast_cfg.items():
        out[key] = {
            "adata": read_h5ad(cfg["diff_path"]),
            "rank_df": read_rank_csv(cfg["rank_csv"]),
            "title": cfg["title"],
        }
    return out


# ============================================================
# GENE / DOMAIN EFFECT TABLE
# ============================================================
def get_ranked_gene_union_for_domain(
    rank_df_early: pd.DataFrame,
    rank_df_late: pd.DataFrame,
    domain: str,
    top_n=TOP_GENES_PER_DOMAIN_PER_CONTRAST,
):
    genes = []
    for df in [rank_df_early, rank_df_late]:
        dfm = df[df["domain"] == str(domain)].copy()
        genes.extend(dfm["gene"].astype(str).drop_duplicates().head(top_n).tolist())

    seen = set()
    out = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def get_rank_score_for_gene(rank_df: pd.DataFrame, domain: str, gene: str):
    dfm = rank_df[(rank_df["domain"] == str(domain)) & (rank_df["gene"] == str(gene))]
    if dfm.empty:
        return {
            "pval": np.nan,
            "logpval": np.nan,
            "fstat": np.nan,
            "fdr": np.nan,
        }
    row = dfm.iloc[0]
    return {
        "pval": float(row["pval"]),
        "logpval": float(row["logpval"]),
        "fstat": float(row["fstat"]),
        "fdr": float(row["fdr"]),
    }


def summarize_domain_effect(
    ad,
    domain: str,
    gene: str,
    domain_key=DOMAIN_KEY,
    summary="median",
):
    labels = get_domain_labels(ad, domain_key=domain_key)
    m = labels == str(domain)
    if np.sum(m) == 0:
        return np.nan

    if gene not in ad.var_names:
        return np.nan

    vals = extract_gene_vector(ad, gene)[m]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan

    if summary == "median":
        return float(np.median(vals))
    elif summary == "mean":
        return float(np.mean(vals))
    elif summary == "trimmed_mean":
        lo = np.quantile(vals, 0.1)
        hi = np.quantile(vals, 0.9)
        vv = vals[(vals >= lo) & (vals <= hi)]
        return float(np.mean(vv)) if len(vv) > 0 else float(np.mean(vals))
    else:
        raise ValueError("summary must be one of: median, mean, trimmed_mean")


def build_domain_gene_trend_table(
    all_data: Dict[str, Dict],
    domains: Sequence[str],
    summary="median",
):
    rows = []
    early_ad = all_data[EARLY_KEY]["adata"]
    late_ad = all_data[LATE_KEY]["adata"]
    early_rank = all_data[EARLY_KEY]["rank_df"]
    late_rank = all_data[LATE_KEY]["rank_df"]

    for dom in domains:
        gene_pool = get_ranked_gene_union_for_domain(
            early_rank, late_rank, domain=dom, top_n=TOP_GENES_PER_DOMAIN_PER_CONTRAST
        )

        for gene in gene_pool:
            early_effect = summarize_domain_effect(
                early_ad, domain=dom, gene=gene, summary=summary
            )
            late_effect = summarize_domain_effect(
                late_ad, domain=dom, gene=gene, summary=summary
            )

            early_score = get_rank_score_for_gene(early_rank, dom, gene)
            late_score = get_rank_score_for_gene(late_rank, dom, gene)

            net_effect = np.nan
            if np.isfinite(early_effect) and np.isfinite(late_effect):
                net_effect = early_effect + late_effect

            rows.append({
                "domain": str(dom),
                "gene": str(gene),

                "early_effect": early_effect,   # 5.7 - 2.5
                "late_effect": late_effect,     # 17.9 - 5.7
                "net_effect": net_effect,       # 17.9 - 2.5, reconstructed

                "early_logpval": early_score["logpval"],
                "late_logpval": late_score["logpval"],
                "early_fstat": early_score["fstat"],
                "late_fstat": late_score["fstat"],
                "early_fdr": early_score["fdr"],
                "late_fdr": late_score["fdr"],

                "max_logpval": np.nanmax([early_score["logpval"], late_score["logpval"]]),
                "max_abs_effect": np.nanmax(np.abs([early_effect, late_effect])),
            })

    df = pd.DataFrame(rows)
    return df


# ============================================================
# TREND CLASSIFICATION
# ============================================================
def get_effect_threshold(trend_df: pd.DataFrame):
    vals = trend_df["max_abs_effect"].to_numpy(dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return FIXED_EFFECT_THRESHOLD

    if EFFECT_THRESHOLD_MODE == "fixed":
        return FIXED_EFFECT_THRESHOLD
    else:
        return float(np.quantile(vals, EFFECT_THRESHOLD_Q))


def classify_trend_row(row, effect_thresh, min_logpval=MIN_LOGPVAL):
    e = row["early_effect"]
    l = row["late_effect"]
    sig = row["max_logpval"]

    if not np.isfinite(e) or not np.isfinite(l):
        return "stable"

    if (max(abs(e), abs(l)) < effect_thresh) or (not np.isfinite(sig)) or (sig < min_logpval):
        return "stable"

    if e > 0 and l > 0:
        return "increase"
    elif e < 0 and l < 0:
        return "decrease"
    elif e > 0 and l < 0:
        return "peak"
    elif e < 0 and l > 0:
        return "valley"
    else:
        return "stable"


def add_trend_class(trend_df: pd.DataFrame):
    out = trend_df.copy()
    effect_thresh = get_effect_threshold(out)
    out["trend_class"] = out.apply(
        lambda r: classify_trend_row(r, effect_thresh=effect_thresh, min_logpval=MIN_LOGPVAL),
        axis=1,
    )
    out["trajectory_score"] = out["max_logpval"].fillna(0) * out["max_abs_effect"].fillna(0)
    return out, effect_thresh


# ============================================================
# FIGURE OPTION 1: QUADRANT PLOT
# ============================================================
def plot_domain_trend_quadrant(
    trend_df: pd.DataFrame,
    domain: str,
    annotate_top=ANNOTATE_TOP_N,
    save_path=None,
):
    set_style()

    dfm = trend_df[trend_df["domain"] == str(domain)].copy()
    if dfm.empty:
        print(f"[WARN] No trend data for domain {domain}")
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.8))

    size = 22 + 18 * np.clip(dfm["max_logpval"].fillna(0).to_numpy(), 0, 8)
    colors = [CLASS_COLORS[c] for c in dfm["trend_class"]]

    ax.scatter(
        dfm["early_effect"],
        dfm["late_effect"],
        s=size,
        c=colors,
        alpha=0.82,
        linewidths=0.3,
        edgecolors="white",
    )

    ax.axhline(0, color="0.45", lw=0.9)
    ax.axvline(0, color="0.45", lw=0.9)

    # annotate strongest genes
    lab_df = dfm.sort_values("trajectory_score", ascending=False).head(annotate_top)
    for _, r in lab_df.iterrows():
        ax.text(
            r["early_effect"],
            r["late_effect"],
            r["gene"],
            fontsize=9,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("Early change (5.7m − 2.5m)")
    ax.set_ylabel("Late change (17.9m − 5.7m)")
    ax.set_title(f"Temporal trend classes in domain {domain}")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CLASS_COLORS[k],
               markeredgecolor="none", markersize=7, label=k.capitalize())
        for k in ["increase", "decrease", "peak", "valley", "stable"]
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", title="Trend class")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


# ============================================================
# FIGURE OPTION 2: FACETED QUADRANTS ACROSS DOMAINS
# ============================================================
def plot_trend_quadrant_facets(
    trend_df: pd.DataFrame,
    domains: Sequence[str],
    annotate_top_each=4,
    save_path=None,
):
    set_style()

    n = len(domains)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))

    # shared axis limits
    x = trend_df["early_effect"].to_numpy(dtype=np.float32)
    y = trend_df["late_effect"].to_numpy(dtype=np.float32)
    lim = np.nanpercentile(np.abs(np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])), 98)
    lim = max(float(lim), 1e-4)

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.1 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, dom in zip(axes, domains):
        dfm = trend_df[trend_df["domain"] == str(dom)].copy()
        if dfm.empty:
            ax.axis("off")
            continue

        size = 18 + 14 * np.clip(dfm["max_logpval"].fillna(0).to_numpy(), 0, 8)
        colors = [CLASS_COLORS[c] for c in dfm["trend_class"]]

        ax.scatter(
            dfm["early_effect"],
            dfm["late_effect"],
            s=size,
            c=colors,
            alpha=0.82,
            linewidths=0.25,
            edgecolors="white",
        )
        ax.axhline(0, color="0.45", lw=0.85)
        ax.axvline(0, color="0.45", lw=0.85)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"Domain {dom}")

        lab_df = dfm.sort_values("trajectory_score", ascending=False).head(annotate_top_each)
        for _, r in lab_df.iterrows():
            ax.text(r["early_effect"], r["late_effect"], r["gene"], fontsize=8, ha="left", va="bottom")

    for ax in axes[len(domains):]:
        ax.axis("off")

    fig.supxlabel("Early change (5.7m − 2.5m)")
    fig.supylabel("Late change (17.9m − 5.7m)")
    fig.suptitle("Temporal trend classes across selected domains", y=0.995, fontsize=16)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CLASS_COLORS[k],
               markeredgecolor="none", markersize=7, label=k.capitalize())
        for k in ["increase", "decrease", "peak", "valley", "stable"]
    ]
    fig.legend(handles=handles, frameon=False, loc="center right", bbox_to_anchor=(1.02, 0.5), title="Trend class")

    fig.tight_layout(rect=[0, 0, 0.95, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


# ============================================================
# FIGURE OPTION 3: TRAJECTORY CLASS STACKED BARS
# ============================================================
def plot_trend_class_stacked_bars(
    trend_df: pd.DataFrame,
    domains: Sequence[str],
    save_path=None,
):
    set_style()

    rows = []
    order = ["increase", "decrease", "peak", "valley", "stable"]

    for dom in domains:
        dfm = trend_df[trend_df["domain"] == str(dom)].copy()
        if dfm.empty:
            continue
        vc = dfm["trend_class"].value_counts(normalize=True)
        row = {"domain": str(dom)}
        for c in order:
            row[c] = float(vc.get(c, 0.0))
        rows.append(row)

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        print("[WARN] No data for trend-class stacked bars.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(plot_df.shape[0])
    bottom = np.zeros(plot_df.shape[0], dtype=np.float32)

    for c in order:
        vals = plot_df[c].to_numpy(dtype=np.float32)
        ax.bar(
            x, vals, bottom=bottom, width=0.72,
            color=CLASS_COLORS[c], alpha=0.90, label=c.capitalize()
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d}" for d in plot_df["domain"]])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Fraction of selected genes")
    ax.set_title("Temporal trend classes across domains")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", title="Trend class")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


# ============================================================
# FIGURE OPTION 4: PSEUDO-TRAJECTORY SMALL MULTIPLES
# ============================================================
def plot_domain_pseudotrajectory_small_multiples(
    trend_df: pd.DataFrame,
    domain: str,
    top_n=PSEUDOTRAJ_TOP_N,
    normalize_rows=False,
    save_path=None,
):
    set_style()

    dfm = trend_df[trend_df["domain"] == str(domain)].copy()
    if dfm.empty:
        print(f"[WARN] No data for domain {domain}")
        return

    dfm = dfm.sort_values("trajectory_score", ascending=False).head(top_n).copy()

    n = len(dfm)
    ncol = 3
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow), squeeze=False)
    axes = axes.ravel()

    x = np.array([2.5, 5.7, 17.9], dtype=np.float32)

    for ax, (_, row) in zip(axes, dfm.iterrows()):
        y = np.array([
            0.0,
            row["early_effect"],
            row["early_effect"] + row["late_effect"],
        ], dtype=np.float32)

        if normalize_rows:
            mu = np.mean(y)
            sd = np.std(y)
            y = (y - mu) / (sd + 1e-8)

        cls = row["trend_class"]
        ax.plot(x, y, color=CLASS_COLORS[cls], lw=2.0, marker="o", ms=4.5)
        ax.axhline(0, color="0.65", lw=0.8, alpha=0.6)
        ax.set_title(row["gene"], pad=5, fontsize=12)
        ax.set_xticks([2.5, 5.7, 17.9])
        ax.set_xticklabels(["2.5", "5.7", "17.9"])
        ax.set_xlabel("Time (m)")
        ax.set_ylabel("Relative\nchange")
        ax.grid(axis="y", alpha=0.14)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Pseudo-trajectories of top genes in domain {domain}", y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


# ============================================================
# FIGURE OPTION 5: SIGNED TREND HEATMAP
# ============================================================
def _trend_class_order_key(c):
    order = {
        "increase": 0,
        "peak": 1,
        "valley": 2,
        "decrease": 3,
        "stable": 4,
    }
    return order.get(c, 99)


def plot_signed_trend_heatmap(
    trend_df: pd.DataFrame,
    domain: str,
    top_n=HEATMAP_TOP_N,
    include_net=True,
    save_path=None,
):
    set_style()

    dfm = trend_df[trend_df["domain"] == str(domain)].copy()
    if dfm.empty:
        print(f"[WARN] No data for domain {domain}")
        return

    dfm = dfm.sort_values(["trend_class", "trajectory_score"], ascending=[True, False]).copy()
    dfm["class_order"] = dfm["trend_class"].map(_trend_class_order_key)
    dfm = dfm.sort_values(["class_order", "trajectory_score"], ascending=[True, False]).head(top_n)

    cols = ["early_effect", "late_effect"]
    if include_net:
        cols.append("net_effect")

    mat = dfm[cols].to_numpy(dtype=np.float32)
    vmax = np.nanpercentile(np.abs(mat[np.isfinite(mat)]), 98)
    vmax = max(float(vmax), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(5.2, 0.32 * dfm.shape[0] + 2.2))
    sns.heatmap(
        dfm[cols].set_axis(dfm["gene"].tolist(), axis=0),
        cmap="coolwarm",
        norm=norm,
        linewidths=0.08,
        linecolor="white",
        cbar_kws={"label": "Signed effect"},
        ax=ax,
    )

    # color gene labels by class
    for tick, cls in zip(ax.get_yticklabels(), dfm["trend_class"]):
        tick.set_color(CLASS_COLORS[cls])

    pretty_cols = {
        "early_effect": "5.7−2.5",
        "late_effect": "17.9−5.7",
        "net_effect": "17.9−2.5",
    }
    ax.set_xticklabels([pretty_cols[c] for c in cols], rotation=0)
    ax.set_title(f"Signed temporal trend heatmap | domain {domain}")
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    all_data = load_all_contrasts(CONTRASTS)

    trend_df = build_domain_gene_trend_table(
        all_data=all_data,
        domains=SELECTED_DOMAINS,
        summary="median",
    )
    trend_df, effect_thresh = add_trend_class(trend_df)

    trend_df.to_csv(os.path.join(OUTDIR, "domain_gene_trend_table.csv"), index=False)
    print(f"[INFO] trend table saved. effect threshold = {effect_thresh:.4f}")

    # --------------------------------------------------------
    # Option 1: main domain 11 quadrant
    # --------------------------------------------------------
    plot_domain_trend_quadrant(
        trend_df,
        domain=FOCUS_DOMAIN,
        annotate_top=ANNOTATE_TOP_N,
        save_path=os.path.join(OUTDIR, f"quadrant_domain_{FOCUS_DOMAIN}.png"),
    )

    # --------------------------------------------------------
    # Option 2: quadrant facets across selected domains
    # --------------------------------------------------------
    plot_trend_quadrant_facets(
        trend_df,
        domains=SELECTED_DOMAINS,
        annotate_top_each=4,
        save_path=os.path.join(OUTDIR, "quadrant_facets_selected_domains.png"),
    )

    # --------------------------------------------------------
    # Option 3: stacked bar of trend classes across domains
    # --------------------------------------------------------
    plot_trend_class_stacked_bars(
        trend_df,
        domains=SELECTED_DOMAINS,
        save_path=os.path.join(OUTDIR, "trend_class_stacked_bars.png"),
    )

    # --------------------------------------------------------
    # Option 4: domain 11 pseudo-trajectories
    # --------------------------------------------------------
    plot_domain_pseudotrajectory_small_multiples(
        trend_df,
        domain=FOCUS_DOMAIN,
        top_n=PSEUDOTRAJ_TOP_N,
        normalize_rows=False,
        save_path=os.path.join(OUTDIR, f"pseudotrajectory_domain_{FOCUS_DOMAIN}.png"),
    )

    # normalized version as an alternative
    plot_domain_pseudotrajectory_small_multiples(
        trend_df,
        domain=FOCUS_DOMAIN,
        top_n=PSEUDOTRAJ_TOP_N,
        normalize_rows=True,
        save_path=os.path.join(OUTDIR, f"pseudotrajectory_domain_{FOCUS_DOMAIN}_zscore.png"),
    )

    # --------------------------------------------------------
    # Option 5: signed trend heatmap
    # --------------------------------------------------------
    plot_signed_trend_heatmap(
        trend_df,
        domain=FOCUS_DOMAIN,
        top_n=HEATMAP_TOP_N,
        include_net=True,
        save_path=os.path.join(OUTDIR, f"signed_trend_heatmap_domain_{FOCUS_DOMAIN}.png"),
    )

    print(f"[DONE] Saved all outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()