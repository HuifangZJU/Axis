
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu


# ============================================================
# CONFIG
# ============================================================

OUTDIR = "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/dlpfc_z_group1_grid70_spatial0.7/analysis/aggregation_evaluation"
# OUTDIR = "/media/huifang/data/registration/SCC/huifang/virtual_tissue/scc_z_group10_grid32_spatial5/analysis/aggregation_evaluation"
LONG_CSV = os.path.join(OUTDIR, "marker_expression_long.csv")
SUMMARY_CSV = os.path.join(OUTDIR, "marker_slice_summary.csv")
ADJ_CSV = os.path.join(OUTDIR, "marker_adjacent_stats.csv")


# ============================================================
# CONFIG
# ============================================================

GENES_TO_PLOT = ["MOBP","CUX2","RAI14","RORB", "ENC1","PCP4", "KCNIP2","SYT6", "FOXP2","MBP","MFGE8"]
# GENES_TO_PLOT =  ['KRT1','MMP10','PTHLH']
# expected left-to-right panel order in the final figure
PANEL_ORDER = [
    "Single 0",
    "Single 1",
    "Single 2",
    "Single 3",
    "Pool (raw)",
    "Pool (warped)",
    "AXIS",
]

sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)


# ============================================================
# HELPERS
# ============================================================

def canonical_layer_order(layer_values):
    """
    Order as Layer* first, WM last.
    Example:
        Layer3, Layer4, Layer5, Layer6, WM
    """
    vals = pd.Series(layer_values).astype(str).dropna().unique().tolist()

    layer_items = []
    wm_items = []
    other_items = []

    for v in vals:
        vv = str(v)
        if vv.lower().startswith("layer"):
            suffix = vv[len("Layer"):]
            try:
                num = int(suffix)
            except Exception:
                num = 10**9
            layer_items.append((num, vv))
        elif vv.upper() == "WM":
            wm_items.append(vv)
        else:
            other_items.append(vv)

    layer_items = [x[1] for x in sorted(layer_items, key=lambda t: t[0])]
    other_items = sorted(other_items)

    return layer_items + other_items + wm_items


def short_layer_labels(layer_order):
    """
    Layer3 -> L3, WM -> WM
    """
    out = []
    for x in layer_order:
        if str(x).startswith("Layer"):
            out.append("L" + str(x).replace("Layer", ""))
        else:
            out.append(str(x))
    return out


def format_pvalue(p):
    if np.isnan(p):
        return "NA"
    if p < 1e-300:
        return r"$P < 10^{-300}$"
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        coef = p / (10 ** exp)
        coef = int(round(coef))
        if coef == 1:
            return rf"$P \leq 1 \times 10^{{{exp}}}$"
        return rf"$P \leq {coef} \times 10^{{{exp}}}$"
    return rf"$P = {p:.2f}$"


def build_panel_label(row):
    """
    Robustly map saved rows to one of the seven display labels.
    Works with several collection styles.

    Expected source columns:
      - slice_label
      - mode
    """
    slice_label = str(row["slice_label"])
    mode = str(row["mode"]).lower() if "mode" in row.index else ""

    # singles
    if slice_label.startswith("Single"):
        return slice_label

    # common concatenation / aggregation cases
    if slice_label == "Concatenated" and mode == "raw":
        return "Concatenated (raw)"
    if slice_label == "Concatenated" and mode == "warped":
        return "Concatenated (warped)"
    if slice_label == "Aggregated" and mode == "max":
        return "AXIS aggregation (max)"

    # already formatted labels
    if slice_label in PANEL_ORDER:
        return slice_label

    # fallback
    return f"{slice_label} ({mode})" if mode else slice_label

def get_panel_palette(panel_order):
    """
    Make the singles light gray, pooled controls medium blue-gray,
    and AXIS highlighted.
    """
    palette = {}
    for p in panel_order:
        print(p)
        if p.startswith("Single"):
            palette[p] = "#c7c7c7"
        elif p == "Pool (raw)":
            palette[p] = "#9ecae1"
        elif p == "Pool (warped)":
            palette[p] = "#6baed6"
        elif p == "AXIS":
            palette[p] = "#d62728"
        else:
            palette[p] = "#999999"
    return palette


# ============================================================
# MAIN PLOTTING
# ============================================================

def plot_gene_violin_seven_panel_from_csv(
    long_csv,
    gene,
    save_path=None,
    xtick_rotation=35,
    scale="width",   # use "count" if you want width to reflect spot number
):
    df = pd.read_csv(long_csv)

    # basic filtering
    df = df[df["gene"] == gene].copy()
    if df.empty:
        return
        raise ValueError(f"No rows found for gene '{gene}' in {long_csv}")


    # build final panel label
    df["panel_label"] = df.apply(build_panel_label, axis=1)

    # keep only the seven desired panels and in that order
    df = df[df["panel_label"].isin(PANEL_ORDER)].copy()
    if df.empty:
        raise ValueError(
            f"No rows matched the seven-panel design for gene '{gene}'. "
            f"Check slice_label/mode values in the CSV."
        )

    panel_order = [x for x in PANEL_ORDER if x in pd.unique(df["panel_label"])]
    layer_order = canonical_layer_order(df["layer"].values)
    xlabels = short_layer_labels(layer_order)

    # consistent y-range across all panels
    global_ymax = float(np.nanmax(df["value_plot"].values))
    y_top = max(global_ymax * 1.1, global_ymax+0.2 )
    # y_top = global_ymax * 1.1

    fig, axes = plt.subplots(1, len(panel_order), figsize=(3.8 * len(panel_order), 4.8), sharey=True)
    if len(panel_order) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panel_order):
        sub = df[df["panel_label"] == panel].copy()

        sns.violinplot(
            data=sub,
            x="layer",
            y="value_plot",
            order=layer_order,
            ax=ax,
            palette="tab10",
            inner="box",
            cut=0,
            linewidth=1.0,
            scale=scale,
        )

        ax.set_title(panel, fontsize=16, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylim(0, y_top)

        # show x tick labels on every subplot
        ax.set_xticks(range(len(layer_order)))
        ax.set_xticklabels(xlabels, rotation=xtick_rotation, ha="center", fontsize=14)

        if ax is axes[0]:
            y_label = sub["y_label"].iloc[0] if "y_label" in sub.columns else "Log counts"
            ax.set_ylabel(y_label, fontsize=15)
        else:
            ax.set_ylabel("")

        ax.grid(axis="y", alpha=0.22)

    # mode shown in title only if unique
    modes = pd.unique(df["mode"]).tolist() if "mode" in df.columns else []
    if len(modes) == 1:
        mode_txt = modes[0]
        fig.suptitle(
            f"{gene} expression distribution across layers ({mode_txt})",
            fontsize=18,
            fontweight="bold",
            y=1.04,
        )
    else:
        fig.suptitle(
            f"{gene} expression distribution across layers",
            fontsize=18,
            fontweight="bold",
            y=1.04,
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()

    return fig, axes

# ============================================================
# 2) SUMMARY METRIC FIGURE FOR EACH GENE
# ============================================================

def plot_gene_summary_metrics_from_csv(
    summary_csv,
    gene,
    metrics=["adjacent_mean_neglog10p", "dominant_margin"],
    save_path=None,
):
    df = pd.read_csv(summary_csv)
    df = df[df["gene"] == gene].copy()
    if df.empty:
        raise ValueError(f"No summary rows found for gene '{gene}' in {summary_csv}")

    df["panel_label"] = df.apply(build_panel_label, axis=1)
    df = df[df["panel_label"].isin(PANEL_ORDER)].copy()

    if df.empty:
        raise ValueError(
            f"No summary rows matched the seven-panel design for gene '{gene}'."
        )

    panel_order = [x for x in PANEL_ORDER if x in pd.unique(df["panel_label"])]
    palette = get_panel_palette(panel_order)

    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5.8 * n_metrics, 5.5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = df[["panel_label", metric]].copy()
        sub["panel_label"] = pd.Categorical(sub["panel_label"], categories=panel_order, ordered=True)
        sub = sub.sort_values("panel_label")

        colors = [palette[p] for p in sub["panel_label"]]

        ax.bar(
            np.arange(len(sub)),
            sub[metric].values,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
        )

        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["panel_label"], rotation=35, ha="right", fontsize=14)
        ax.set_title("Dominant layer margin for MFGE8", fontsize=14, fontweight="bold")
        ax.set_ylabel("Mean log1p expression", fontsize=14)
        ax.grid(axis="y", alpha=0.22)

    fig.suptitle(
        f"{gene}: summary metrics",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()

    return fig, axes


# ============================================================
# 3) OPTIONAL: ONE COMBINED SUMMARY PANEL FOR ALL GENES
# ============================================================

def plot_summary_metric_by_gene_and_case(
    summary_csv,
    metric="adjacent_mean_neglog10p",
    genes=None,
    save_path=None,
):
    df = pd.read_csv(summary_csv)

    if genes is not None:
        df = df[df["gene"].isin(genes)].copy()

    df["panel_label"] = df.apply(build_panel_label, axis=1)
    df = df[df["panel_label"].isin(PANEL_ORDER)].copy()

    panel_order = [x for x in PANEL_ORDER if x in pd.unique(df["panel_label"])]
    palette = get_panel_palette(panel_order)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    sns.barplot(
        data=df,
        x="gene",
        y=metric,
        hue="panel_label",
        hue_order=panel_order,
        palette=palette,
        ax=ax,
        edgecolor="black",
    )

    ax.set_title(metric.replace("_", " "), fontsize=15, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(metric.replace("_", " "), fontsize=13)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()

    return fig, ax

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    for gene in GENES_TO_PLOT:
        save_path = os.path.join(OUTDIR, f"{gene}_seven_panel_violin.png")
        plot_gene_violin_seven_panel_from_csv(
            long_csv=LONG_CSV,
            gene=gene,
            save_path=save_path,
            xtick_rotation=35,
            scale="width",
        )

        summary_path = os.path.join(OUTDIR, f"{gene}_summary_metrics.png")
        plot_gene_summary_metrics_from_csv(
            summary_csv=SUMMARY_CSV,
            gene=gene,
            metrics=["dominant_margin"],
            save_path=summary_path,
        )
    # plot_summary_metric_by_gene_and_case(
    #     summary_csv=SUMMARY_CSV,
    #     metric="adjacent_mean_neglog10p",
    #     # genes=GENES_TO_PLOT,
    #     # save_path=combined_path,
    # )