import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Config
# ============================================================

METHOD_DISPLAY = {
    "unaligned": "Unaligned",
    "simpleitk": "SimpleITK",
    "voxelmorph": "VoxelMorph",
    "nicetrans": "NICE-Trans",
    "paste": "PASTE",
    "gpsa": "GPSA",
    "santo": "SANTO",
    "axis": "AXIS",
}

METHOD_ORDER = [
    "Unaligned",
    "SimpleITK",
    "VoxelMorph",
    "NICE-Trans",
    "PASTE",
    "GPSA",
    "SANTO",
    "AXIS",
]

NOISE_ORDER = [0, 1, 2, 3, 4, 5]


# ============================================================
# Basic helpers
# ============================================================

def get_metric_label(metric):
    mapping = {
        "spot_topk": "Top-k label consistency",
    }
    return mapping.get(metric, metric)


def save_or_show(fig, save_path=None, dpi=300):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def load_metrics_csv(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    df = df.copy()
    df["method"] = df["method"].astype(str).str.lower()
    df["domain"] = df["domain"].astype(str)

    if "noise_level" not in df.columns:
        df["noise_level"] = 0

    df["noise_level"] = df["noise_level"].astype(int)
    df["method_display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"])

    df["pair_id"] = (
        "g" + df["group_id"].astype(str)
        + ":" + df["fixed_id"].astype(str)
        + "→" + df["moving_id"].astype(str)
    )
    df["pair_type"] = (
        df["fixed_id"].astype(str) + "→" + df["moving_id"].astype(str)
    )

    return df


def get_method_order(df):
    present = set(df["method_display"].unique())
    order = [m for m in METHOD_ORDER if m in present]
    extras = [m for m in df["method_display"].unique() if m not in order]
    return order + sorted(extras)


# ============================================================
# Aggregation helpers
# ============================================================

def aggregate_pair_level_with_noise(df, metric="spot_topk"):
    """
    First average over domains so each pair contributes one value.
    """
    pair_df = (
        df.groupby(
            [
                "method",
                "method_display",
                "noise_level",
                "group_id",
                "fixed_id",
                "moving_id",
                "pair_id",
                "pair_type",
            ],
            as_index=False,
        )[metric]
        .mean()
    )
    return pair_df


def summarize_method_noise(pair_df, metric="spot_topk"):
    """
    Aggregate pair-level data across groups/pairs into mean/sem per method x noise.
    """
    summary = (
        pair_df.groupby(["method", "method_display", "noise_level"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    return summary


def compute_drop_from_level0(pair_df, metric="spot_topk"):
    """
    Compute per-pair change relative to noise_level 0.
    delta = score(level_k) - score(level_0)
    """
    base = pair_df[pair_df["noise_level"] == 0][
        ["method", "group_id", "fixed_id", "moving_id", metric]
    ].copy()
    base = base.rename(columns={metric: f"{metric}_level0"})

    merged = pair_df.merge(
        base,
        on=["method", "group_id", "fixed_id", "moving_id"],
        how="left",
    )

    merged[f"{metric}_drop"] = merged[metric] - merged[f"{metric}_level0"]
    return merged


def summarize_drop_from_level0(pair_df, metric="spot_topk"):
    merged = compute_drop_from_level0(pair_df, metric=metric)

    summary = (
        merged.groupby(["method", "method_display", "noise_level"], as_index=False)[f"{metric}_drop"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    return summary


# ============================================================
# Plot 1: robustness curve
# ============================================================

def plot_noise_robustness_curves(
    df,
    metric="spot_topk",
    save_path=None,
    figsize=(8, 5),
    highlight_method="AXIS",
    show_errorbar=True,
):
    pair_df = aggregate_pair_level_with_noise(df, metric=metric)
    summary = summarize_method_noise(pair_df, metric=metric)
    method_order = get_method_order(df)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    palette = sns.color_palette("tab10", n_colors=len(method_order))
    method_to_color = {m: palette[i] for i, m in enumerate(method_order)}

    for method_display in method_order:
        sub = summary[summary["method_display"] == method_display].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("noise_level")

        is_highlight = (method_display == highlight_method)
        alpha = 1.0 if is_highlight else 0.45
        lw = 3.0 if is_highlight else 1.8
        ms = 8 if is_highlight else 5

        ax.plot(
            sub["noise_level"],
            sub["mean"],
            marker="o",
            linewidth=lw,
            markersize=ms,
            alpha=alpha,
            color=method_to_color[method_display],
            label=method_display,
        )

        if show_errorbar:
            ax.errorbar(
                sub["noise_level"],
                sub["mean"],
                yerr=sub["sem"],
                fmt="none",
                ecolor=method_to_color[method_display],
                alpha=alpha,
                capsize=3,
                linewidth=1.2,
            )

    ax.set_xticks(NOISE_ORDER)
    ax.set_xlabel("Noise level")
    ax.set_ylabel(get_metric_label(metric))
    ax.set_title(f"Noise robustness curve: {get_metric_label(metric)}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")

    fig = ax.figure
    fig.tight_layout()
    save_or_show(fig, save_path=save_path)


# ============================================================
# Plot 2: degradation from level 0
# ============================================================

def plot_noise_drop_from_level0(
    df,
    metric="spot_topk",
    save_path=None,
    figsize=(8, 5),
    highlight_method="AXIS",
    show_errorbar=True,
):
    pair_df = aggregate_pair_level_with_noise(df, metric=metric)
    summary = summarize_drop_from_level0(pair_df, metric=metric)
    method_order = get_method_order(df)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    palette = sns.color_palette("tab10", n_colors=len(method_order))
    method_to_color = {m: palette[i] for i, m in enumerate(method_order)}

    drop_col = f"{metric}_drop"

    for method_display in method_order:
        sub = summary[summary["method_display"] == method_display].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("noise_level")

        is_highlight = (method_display == highlight_method)
        alpha = 1.0 if is_highlight else 0.45
        lw = 3.0 if is_highlight else 1.8
        ms = 8 if is_highlight else 5

        ax.plot(
            sub["noise_level"],
            sub["mean"],
            marker="o",
            linewidth=lw,
            markersize=ms,
            alpha=alpha,
            color=method_to_color[method_display],
            label=method_display,
        )

        if show_errorbar:
            ax.errorbar(
                sub["noise_level"],
                sub["mean"],
                yerr=sub["sem"],
                fmt="none",
                ecolor=method_to_color[method_display],
                alpha=alpha,
                capsize=3,
                linewidth=1.2,
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(NOISE_ORDER)
    ax.set_xlabel("Noise level")
    ax.set_ylabel(f"Δ {get_metric_label(metric)} from level 0")
    ax.set_title(f"Performance drop relative to original data")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")

    fig = ax.figure
    fig.tight_layout()
    save_or_show(fig, save_path=save_path)


# ============================================================
# Plot 3: method x noise heatmap
# ============================================================

def plot_noise_method_heatmap(
    df,
    metric="spot_topk",
    save_path=None,
    figsize=(6, 4.8),
    annotate=True,
):
    pair_df = aggregate_pair_level_with_noise(df, metric=metric)

    agg = (
        pair_df.groupby(["method_display", "noise_level"], as_index=False)[metric]
        .mean()
    )

    pivot = agg.pivot(index="method_display", columns="noise_level", values=metric)
    pivot = pivot.reindex(
        index=[m for m in METHOD_ORDER if m in pivot.index],
        columns=[n for n in NOISE_ORDER if n in pivot.columns],
    )

    plt.figure(figsize=figsize)
    ax = plt.gca()

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="viridis",
        annot=annotate,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": get_metric_label(metric)},
    )

    ax.set_xlabel("Noise level")
    ax.set_ylabel("")
    ax.set_title(f"Method × noise level heatmap: {get_metric_label(metric)}")

    fig = ax.figure
    fig.tight_layout()
    save_or_show(fig, save_path=save_path)


# ============================================================
# Plot 4: optional compact boxplot at each noise level
# ============================================================

def plot_noise_level_boxplot(
    df,
    metric="spot_topk",
    noise_level=0,
    save_path=None,
    figsize=(7, 4.8),
):
    """
    Compare methods at one fixed noise level.
    Each dot = one pair-level average across domains.
    """
    pair_df = aggregate_pair_level_with_noise(df, metric=metric)
    pair_df = pair_df[pair_df["noise_level"] == noise_level].copy()

    order = get_method_order(pair_df)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    sns.boxplot(
        data=pair_df,
        x="method_display",
        y=metric,
        order=order,
        ax=ax,
        showfliers=False,
        width=0.6,
        color="white",
        linewidth=1.2,
    )

    sns.stripplot(
        data=pair_df,
        x="method_display",
        y=metric,
        order=order,
        ax=ax,
        color="0.35",
        size=5,
        alpha=0.7,
        jitter=0.18,
    )

    ax.set_xlabel("")
    ax.set_ylabel(get_metric_label(metric))
    ax.set_title(f"Noise level {noise_level}: method comparison")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

    fig = ax.figure
    fig.tight_layout()
    save_or_show(fig, save_path=save_path)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    CSV_PATH = "/media/huifang/data/registration/SCC/evaluation/scc_molecular_metrics_all_methods.csv"

    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.95,
    )

    df = load_metrics_csv(CSV_PATH)

    # Main figure: all methods across noise
    # plot_noise_robustness_curves(
    #     df,
    #     metric="spot_topk",
    #     save_path=None,
    #     highlight_method="AXIS",
    # )

    # Stronger robustness panel: drop from level 0
    # plot_noise_drop_from_level0(
    #     df,
    #     metric="spot_topk",
    #     save_path=None,
    #     highlight_method="AXIS",
    # )

    # Supporting figure: method x noise heatmap
    plot_noise_method_heatmap(
        df,
        metric="spot_topk",
        save_path=None,
    )

    # Optional: compare methods at one specific noise level
    # plot_noise_level_boxplot(
    #     df,
    #     metric="spot_topk",
    #     noise_level=3,
    #     save_path=None,
    # )