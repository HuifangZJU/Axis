import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Callable, Optional
import numpy as np


# ============================================================
# Config
# ============================================================

REQUIRED_COLUMNS = [
    "group_id",
    "fixed_id",
    "moving_id",
    "distance_um",
    "method",
    "domain",
    "dice",
    "spatial_corr",
    "center_shift",
]

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

METRIC_LABELS = {
    "dice": "Dice score",
    "spatial_corr": "Spatial correlation",
    "center_shift": "Center shift",
    "spot_topk": "Spot topk"
}



# ============================================================
# Basic helpers
# ============================================================

def _identity_label(x):
    return str(x)

def apply_heatmap_tick_format(
    ax,
    pivot,
    format_distance_label: Optional[Callable] = None,
    format_domain_label: Optional[Callable] = None,
    xtick_rotation: float = 0,
    ytick_rotation: float = 0,
    xtick_fontsize: int = 14,
    ytick_fontsize: int = 12,
    xtick_shift=(0.0, 0.0),
    ytick_shift=(0.0, 0.0),
):
    """
    Generic helper for formatting and nudging heatmap tick labels.
    """

    if format_distance_label is None:
        format_distance_label = _identity_label
    if format_domain_label is None:
        format_domain_label = _identity_label

    ax.set_xticklabels(
        [format_distance_label(c) for c in pivot.columns],
        rotation=xtick_rotation,
        fontsize=xtick_fontsize,
    )
    for label in ax.get_xticklabels():
        x, y = label.get_position()
        label.set_position((x + xtick_shift[0], y + xtick_shift[1]))

    ax.set_yticklabels(
        [format_domain_label(r) for r in pivot.index],
        rotation=ytick_rotation,
        fontsize=ytick_fontsize,
    )
    for label in ax.get_yticklabels():
        x, y = label.get_position()
        label.set_position((x + ytick_shift[0], y + ytick_shift[1]))


def subset_one_group(df, group_id, method="axis"):
    sub = df[(df["group_id"] == group_id) & (df["method"] == method)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for group_id={group_id}, method='{method}'")
    return sub

def domain_sort_key(x):
    x = str(x)
    m = re.match(r"Layer(\d+)$", x)
    if m:
        return (0, int(m.group(1)))
    if x.upper() == "WM":
        return (1, 999)
    return (2, x)


def get_domain_order(df):
    domains = sorted(df["domain"].dropna().unique(), key=domain_sort_key)
    return domains


def get_method_order(df):
    present = set(df["method_display"].unique())
    order = [m for m in METHOD_ORDER if m in present]
    extras = [m for m in df["method_display"].unique() if m not in order]
    return order + sorted(extras)


def get_metric_label(metric):
    if metric not in METRIC_LABELS:
        raise ValueError(f"Unsupported metric: {metric}")
    label = METRIC_LABELS[metric]
    # if metric == "center_shift":
    #     label += " (lower is better)"
    return label


def metric_direction(metric):
    if metric in ["dice", "spatial_corr"]:
        return "higher"
    if metric == "center_shift":
        return "lower"
    raise ValueError(metric)


def default_cmap(metric):
    if metric in ["dice", "spatial_corr"]:
        return "viridis"
    if metric == "center_shift":
        return "viridis_r"
    return "viridis"


def save_or_show(fig, save_path=None, dpi=300):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ============================================================
# Aggregation helpers
# ============================================================

def aggregate_pair_level(df,metrics=["dice", "spatial_corr", "center_shift"]):
    """
    Average over domains first, so each pair contributes one observation.
    """
    pair_df = (
        df.groupby(
            [
                "method",
                "method_display",
                "group_id",
                "fixed_id",
                "moving_id",
                "distance_um",
                # "distance_bin",
                "pair_id",
                "pair_type",
            ],
            as_index=False,
        )[metrics]
        .mean()
    )
    return pair_df


def aggregate_domain_distance(
    df,
    method="axis",
    group_id=None,
    metric="dice",
    mode="distance_bin",
    DISTANCE_ORDER=None,
    DISTANCE_BIN_ORDER=None
):
    if group_id is None:
        raise ValueError("group_id must be provided for domain-based analysis")

    sub = subset_one_group(df, group_id=group_id, method=method)

    if mode == "distance_bin":
        agg = (
            sub.groupby(["domain", "distance_bin"], as_index=False)[metric]
            .mean()
        )
        pivot = agg.pivot(index="domain", columns="distance_bin", values=metric)
        pivot = pivot.reindex(columns=DISTANCE_BIN_ORDER)

    elif mode == "distance_um":
        agg = (
            sub.groupby(["domain", "distance_um"], as_index=False)[metric]
            .mean()
        )
        pivot = agg.pivot(index="domain", columns="distance_um", values=metric)
        cols = [d for d in DISTANCE_ORDER if d in pivot.columns]
        pivot = pivot.reindex(columns=cols)

    else:
        raise ValueError("mode must be 'distance_bin' or 'distance_um'")

    pivot = pivot.reindex(index=get_domain_order(sub))
    return pivot


def aggregate_domain_near_far(df, method="axis", group_id=None, metric="dice"):
    if group_id is None:
        raise ValueError("group_id must be provided for domain-based analysis")

    sub = subset_one_group(df, group_id=group_id, method=method)

    out = (
        sub.groupby(["domain", "distance_bin"], as_index=False)[metric]
        .mean()
    )
    return out


def find_best_competing_method(df, metric="dice"):
    pair_df = aggregate_pair_level(df)

    sub = pair_df[~pair_df["method"].isin(["axis", "unaligned"])].copy()
    if sub.empty:
        raise ValueError("No competing methods found")

    summary = sub.groupby(["method", "method_display"], as_index=False)[metric].mean()

    if metric_direction(metric) == "higher":
        best_row = summary.loc[summary[metric].idxmax()]
    else:
        best_row = summary.loc[summary[metric].idxmin()]

    return best_row["method"], best_row["method_display"]


# ============================================================
# Plot 1: overall method comparison
# ============================================================


def plot_overall_metric_across_methods(
    df,
    metric="dice",
    save_path=None,
    figsize=(8, 5),
):
    """
    One figure for one metric.
    Each dot = one pair-level average across domains.
    """
    pair_df = aggregate_pair_level(df,metric)
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
        size=6,
        alpha=0.75,
        jitter=0.18,
    )

    mean_df = (
        pair_df.groupby("method_display", as_index=False)[metric]
        .mean()
        .set_index("method_display")
        .reindex(order)
        .reset_index()
    )

    ax.scatter(
        np.arange(len(order)),
        mean_df[metric].values,
        marker="D",
        s=70,
        color="crimson",
        zorder=5,
        label="Mean across pairs",
    )

    ax.set_xlabel("")
    ax.set_ylabel(get_metric_label(metric))
    ax.set_title(f"Overall method comparison: {get_metric_label(metric)}")
    ax.tick_params(axis="x", rotation=25)
    # ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    fig = ax.figure

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        length=0,
        rotation=25,
    )

    fig.tight_layout()
    save_or_show(fig, save_path=save_path)

def plot_overall_metric_across_methods_vertical_compact(
    df,
    metric="dice",
    save_path=None,
    figsize=(5, 5),
):
    pair_df = aggregate_pair_level(df,metric)
    order = get_method_order(pair_df)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    sns.boxplot(
        data=pair_df,
        y="method_display",
        x=metric,
        order=order,
        ax=ax,
        showfliers=False,
        width=0.55,
        color="white",
        linewidth=1.1,
        orient="h",
    )

    sns.stripplot(
        data=pair_df,
        y="method_display",
        x=metric,
        order=order,
        ax=ax,
        color="0.35",
        size=4.5,
        alpha=0.7,
        jitter=0.16,
        orient="h",
    )

    mean_df = (
        pair_df.groupby("method_display", as_index=False)[metric]
        .mean()
        .set_index("method_display")
        .reindex(order)
        .reset_index()
    )

    ax.scatter(
        mean_df[metric].values,
        np.arange(len(order)),
        marker="D",
        s=45,
        color="crimson",
        zorder=5,
    )

    ax.set_ylabel("")
    ax.set_xlabel(get_metric_label(metric))
    ax.set_title(get_metric_label(metric))
    ax.grid(axis="x", alpha=0.22)

    fig = ax.figure
    fig.tight_layout()
    save_or_show(fig, save_path=save_path)


def plot_single_pair_metric_across_methods_vertical_compact(
    df,
    group_id=None,
    fixed_id=None,
    moving_id=None,
    metric="dice",
    save_path=None,
    figsize=(5, 5),
):
    """
    Plot one selected pair using domain-level values directly.

    Each dot = one domain metric for the selected pair.
    Each box = distribution across domains for one method.

    If group_id/fixed_id/moving_id are provided, the dataframe will be filtered.
    Otherwise, df is assumed to already contain only one pair.
    """
    plot_df = df.copy()

    # optional filtering
    if group_id is not None:
        plot_df = plot_df[plot_df["group_id"] == group_id]
    if fixed_id is not None:
        plot_df = plot_df[plot_df["fixed_id"] == fixed_id]
    if moving_id is not None:
        plot_df = plot_df[plot_df["moving_id"] == moving_id]

    if plot_df.empty:
        raise ValueError("No rows left after filtering.")

    # make sure this is only one pair
    pair_info = (
        plot_df[["group_id", "fixed_id", "moving_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if len(pair_info) != 1:
        raise ValueError(
            f"Expected exactly one pair, but found {len(pair_info)} pairs. "
            f"Unique pairs:\n{pair_info}"
        )

    order = get_method_order(plot_df)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    sns.boxplot(
        data=plot_df,
        y="method_display",
        x=metric,
        order=order,
        ax=ax,
        showfliers=False,
        width=0.55,
        color="white",
        linewidth=1.1,
        orient="h",
    )

    sns.stripplot(
        data=plot_df,
        y="method_display",
        x=metric,
        order=order,
        ax=ax,
        color="0.35",
        size=4.5,
        alpha=0.7,
        jitter=0.16,
        orient="h",
    )

    mean_df = (
        plot_df.groupby("method_display", as_index=False)[metric]
        .mean()
        .set_index("method_display")
        .reindex(order)
        .reset_index()
    )

    ax.scatter(
        mean_df[metric].values,
        np.arange(len(order)),
        marker="D",
        s=45,
        color="crimson",
        zorder=5,
    )

    row = pair_info.iloc[0]
    ax.set_ylabel("")
    ax.set_xlabel(get_metric_label(metric))
    ax.set_title(
        f"{get_metric_label(metric)} | Per Marker"
    )
    ax.grid(axis="x", alpha=0.22)

    fig = ax.figure
    fig.tight_layout()
    save_or_show(fig, save_path=save_path)
# ============================================================
# Plot 2: AXIS domain heatmap, near vs far
# ============================================================

def plot_axis_domain_heatmap(
    df,
    metric="dice",
    method="axis",
    group_id=None,
    mode="distance_bin",
    annotate=True,
    save_path=None,
    figsize=(5.5, 4.5),
):
    if group_id is None:
        raise ValueError("group_id must be provided for AXIS domain-based analysis")

    pivot = aggregate_domain_distance(
        df,
        method=method,
        group_id=group_id,
        metric=metric,
        mode=mode,
    )

    plt.figure(figsize=figsize)
    ax = plt.gca()

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=default_cmap(metric),
        annot=annotate,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": get_metric_label(metric)},
    )

    title_suffix = "Near vs far" if mode == "distance_bin" else "by exact distance"
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"{METHOD_DISPLAY.get(method, method.upper())} | Group {group_id}: "
        f"{get_metric_label(metric)} ({title_suffix})"
    )

    fig = ax.figure
    fig.tight_layout()
    save_or_show(fig, save_path=save_path)


# ============================================================
# Plot 6: optional AXIS domain heatmap by exact pair type
# ============================================================

def plot_axis_domain_pairtype_heatmap(
    df,
    metric="dice",
    method="axis",
    group_id=None,
    save_path=None,
    figsize=(6.5, 4.8),
):
    if group_id is None:
        raise ValueError("group_id must be provided for AXIS domain-based analysis")

    sub = subset_one_group(df, group_id=group_id, method=method)

    agg = (
        sub.groupby(["domain", "pair_type"], as_index=False)[metric]
        .mean()
    )

    pivot = agg.pivot(index="domain", columns="pair_type", values=metric)

    pair_order = ["0→1", "0→2", "0→3", "1→2", "2→3"]
    pair_order = [p for p in pair_order if p in pivot.columns]

    pivot = pivot.reindex(index=get_domain_order(sub), columns=pair_order)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=default_cmap(metric),
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": get_metric_label(metric)},
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"{METHOD_DISPLAY.get(method, method.upper())} | Group {group_id}: "
        f"{get_metric_label(metric)} by pair type"
    )

    fig = ax.figure
    fig.tight_layout()
    save_or_show(fig, save_path=save_path)


def plot_domain_distance_method_lines(
    df,
    metric="dice",
    group_id=None,
    methods=None,
    save_path=None,
    figsize=None,
    DISTANCE_ORDER=None
):
    """
    Compare all methods in one group:
      - one panel per domain
      - x-axis = distance_um
      - y-axis = metric
      - color = method
    """
    if group_id is None:
        raise ValueError("group_id must be provided")

    sub = df[df["group_id"] == group_id].copy()
    if methods is not None:
        methods = [m.lower() for m in methods]
        sub = sub[sub["method"].isin(methods)]

    if sub.empty:
        raise ValueError(f"No rows found for group_id={group_id}")

    agg = (
        sub.groupby(["method_display", "domain", "distance_um"], as_index=False)[metric]
        .mean()
    )

    domain_order = get_domain_order(sub)
    method_order = get_method_order(sub)

    agg["domain"] = pd.Categorical(agg["domain"], categories=domain_order, ordered=True)
    agg["method_display"] = pd.Categorical(
        agg["method_display"], categories=method_order, ordered=True
    )
    agg = agg.sort_values(["domain", "method_display", "distance_um"])

    n_domains = len(domain_order)
    if figsize is None:
        figsize = (5.2 * min(n_domains, 3), 4.0 * int(np.ceil(n_domains / 3)))

    g = sns.FacetGrid(
        agg,
        col="domain",
        col_wrap=3,
        height=3.6,
        sharey=False,
        hue="method_display",
        hue_order=method_order,
        palette="tab10",
    )

    g.map_dataframe(
        sns.lineplot,
        x="distance_um",
        y=metric,
        marker="o",
        linewidth=2.0,
    )

    g.add_legend(title="Method")
    g.set_axis_labels("Distance (μm)", get_metric_label(metric))
    g.set_titles("{col_name}")

    for ax in g.axes.flat:
        ax.set_xticks([d for d in DISTANCE_ORDER if d in agg["distance_um"].unique()])
        ax.grid(axis="y", alpha=0.25)

    g.figure.set_size_inches(*figsize)
    g.figure.suptitle(
        f"Group {group_id}: method comparison across exact distances ({get_metric_label(metric)})",
        y=1.02,
    )
    g.figure.tight_layout()

    save_or_show(g.figure, save_path=save_path)

def plot_domain_distance_method_heatmaps(
    df,
    metric="dice",
    group_id=None,
    methods=None,
    save_path=None,
    figsize=(16, 8),
    annotate=True,
    format_distance_label=None,
    format_domain_label=None,
    xtick_shift=(0.0, 0.0),
    ytick_shift=(0.0, 0.0),
    DISTANCE_ORDER=None
):
    """
    One heatmap per method, all in one figure.
    Each heatmap:
      rows = domains
      cols = distance_um
    """
    if group_id is None:
        raise ValueError("group_id must be provided")

    sub = df[df["group_id"] == group_id].copy()
    if methods is not None:
        methods = [m.lower() for m in methods]
        sub = sub[sub["method"].isin(methods)]

    if sub.empty:
        raise ValueError(f"No rows found for group_id={group_id}")

    domain_order = get_domain_order(sub)
    method_order = get_method_order(sub)

    method_display_to_raw = (
        sub[["method", "method_display"]]
        .drop_duplicates()
        .set_index("method_display")["method"]
        .to_dict()
    )

    methods_present = [m for m in method_order if m in sub["method_display"].unique()]
    n_methods = len(methods_present)

    ncols = 4
    nrows = int(np.ceil(n_methods / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    vmin = sub[metric].min()
    vmax = sub[metric].max()
    cmap = default_cmap(metric)

    for ax, method_display in zip(axes.flat, methods_present):
        method_raw = method_display_to_raw[method_display]
        sub_m = sub[sub["method"] == method_raw]

        agg = (
            sub_m.groupby(["domain", "distance_um"], as_index=False)[metric]
            .mean()
        )

        pivot = agg.pivot(index="domain", columns="distance_um", values=metric)
        # cols = [d for d in DISTANCE_ORDER if d in pivot.columns]
        pivot = pivot.reindex(index=domain_order, columns=pivot.columns)



        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=annotate,
            fmt=".2f",
            linewidths=0.5,
            linecolor="white",
            cbar=False,
        )

        apply_heatmap_tick_format(
            ax=ax,
            pivot=pivot,
            format_distance_label=format_distance_label,
            format_domain_label=format_domain_label,
            xtick_rotation=0,
            ytick_rotation=0,
            xtick_fontsize=14,
            ytick_fontsize=12,
            xtick_shift=xtick_shift,
            ytick_shift=ytick_shift,
        )


        ax.set_title(method_display)
        ax.set_xlabel("")
        ax.set_ylabel("")


        # ax.set_xticklabels(
        #     [format_distance_label(c) for c in pivot.columns],
        #     rotation=0, fontsize=14
        # )
        # for label in ax.get_xticklabels():
        #     x, y = label.get_position()
        #     label.set_position((x, y + 0.05))
        #
        # ax.set_yticklabels(
        #     [format_domain_label(r) for r in pivot.index],
        #     rotation=0
        # )
        # for label in ax.get_yticklabels():
        #     x, y = label.get_position()
        #     label.set_position((x+0.05, y))

    # hide unused panels
    for ax in axes.flat[n_methods:]:
        ax.axis("off")

    # shared colorbar
    mappable = axes.flat[0].collections[0]
    # cbar = fig.colorbar(
    #     mappable,
    #     ax=axes,
    #     shrink=0.8,
    #     location="right",
    #     pad=0.02,
    # )
    # cbar.set_label(get_metric_label(metric))

    fig.suptitle(
        f"Group {group_id}: domain × distance heatmaps across methods ({get_metric_label(metric)})",
        y=0.98,
    )
    fig.tight_layout()

    save_or_show(fig, save_path=save_path)
