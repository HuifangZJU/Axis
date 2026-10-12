from __future__ import annotations

"""Shared statistical plotting utilities for registration evaluation.

The module centralizes method naming, method colors, box/strip-plot geometry,
typography, and aggregation so DLPFC, mouse, breast, SCC, and future datasets
can use the same visual language.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# =============================================================================
# Shared data / naming configuration
# =============================================================================

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
    "spatialz": "SpatialZ",
    "axis": "AXIS",
}

# Fixed order is preferred for cross-dataset figures.  A stable order makes
# panels directly comparable even when the distribution of a metric changes.
METHOD_ORDER = [
    "Unaligned",
    "SimpleITK",
    "VoxelMorph",
    "NICE-Trans",
    "PASTE",
    "GPSA",
    "SANTO",
    "SpatialZ",
    "AXIS",
]

# One palette used by all registration evaluation boxplots.
GLOBAL_METHOD_PALETTE = {
    "Unaligned": "#B8B8B8",
    "SimpleITK": "#C7A0C9",
    "VoxelMorph": "#A8C4A2",
    "NICE-Trans": "#7C90B0",
    "PASTE": "#8FA8C9",
    "GPSA": "#D8B384",
    "SANTO": "#8FC7C2",
    "SpatialZ": "#C98F8F",
    "AXIS": "#D45A45",
}

FALLBACK_METHOD_COLORS = (
    "#9AAE7F",
    "#B79AA8",
    "#C49A6C",
    "#7FA7A6",
    "#A67C73",
    "#6F86A6",
)

METRIC_LABELS = {
    "dice": "Dice coefficient",
    "spatial_corr": "Spatial correlation",
    "center_shift": "Center shift ↓",
    "neighbor_accuracy": "Molecular correspondence",
    "match_fraction": "Spot neighbors",
}


# =============================================================================
# Shared publication boxplot style
# =============================================================================

@dataclass(frozen=True)
class BoxplotStyle:
    """Shared layout/style for method-comparison box + dot plots."""

    # Canvas. Multi-metric rows use panel_width * n_metrics.
    panel_width: float = 4.6
    figure_height: float = 5.6
    wspace: float = 0.58
    dpi: int = 350

    # Box geometry.
    box_width: float = 0.55
    box_linewidth: float = 1.10
    box_saturation: float = 1.0
    show_fliers: bool = False

    # Individual pair/domain dots.
    dot_color: str = "0.25"
    dot_size: float = 3.8
    dot_alpha: float = 0.40
    dot_jitter: float = 0.14

    # Typography.
    title_fontsize: float = 18.0
    method_tick_fontsize: float = 15.0
    metric_tick_fontsize: float = 14.0
    title_pad: float = 8.0

    # Axis appearance.
    grid_alpha: float = 0.20
    grid_axis: str = "x"


SHARED_BOXPLOT_STYLE = BoxplotStyle()

# Centralized seaborn theme. Call this from dataset-level scripts instead of
# defining a separate theme for each dataset.
STATISTIC_THEME = {
    "style": "whitegrid",
    "context": "talk",
    "font_scale": 0.95,
}


def set_statistic_plot_theme() -> None:
    sns.set_theme(**STATISTIC_THEME)


def get_global_method_palette(method_names: Sequence[str]) -> Dict[str, str]:
    """Return stable colors for display names, with deterministic fallbacks."""
    palette: Dict[str, str] = {}
    fallback_idx = 0
    for method in method_names:
        method = str(method)
        if method in GLOBAL_METHOD_PALETTE:
            palette[method] = GLOBAL_METHOD_PALETTE[method]
        else:
            palette[method] = FALLBACK_METHOD_COLORS[
                fallback_idx % len(FALLBACK_METHOD_COLORS)
            ]
            fallback_idx += 1
    return palette


# =============================================================================
# Basic helpers
# =============================================================================

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
        raise ValueError(
            f"No rows found for group_id={group_id}, method='{method}'"
        )
    return sub


def domain_sort_key(x):
    x = str(x)
    match = re.match(r"Layer(\d+)$", x)
    if match:
        return (0, int(match.group(1)))
    if x.upper() == "WM":
        return (1, 999)
    return (2, x)


def get_domain_order(df):
    return sorted(df["domain"].dropna().unique(), key=domain_sort_key)


def get_method_order(df):
    present = set(df["method_display"].dropna().astype(str).unique())
    order = [method for method in METHOD_ORDER if method in present]
    extras = sorted(method for method in present if method not in order)
    return order + extras


def get_metric_label(metric):
    if metric not in METRIC_LABELS:
        raise ValueError(f"Unsupported metric: {metric}")
    return METRIC_LABELS[metric]


def metric_direction(metric):
    if metric in {"dice", "spatial_corr", "neighbor_accuracy", "match_fraction"}:
        return "higher"
    if metric == "center_shift":
        return "lower"
    raise ValueError(metric)


def default_cmap(metric):
    if metric in {"dice", "spatial_corr", "neighbor_accuracy", "match_fraction"}:
        return "viridis"
    if metric == "center_shift":
        return "viridis_r"
    return "viridis"


def save_or_show(
    fig,
    save_path=None,
    dpi: int = 300,
    show: bool = True,
    close: bool = False,
):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    if close:
        plt.close(fig)


def load_metrics_csv(csv_path) -> pd.DataFrame:
    """Load a collected metric table and add stable display/pair columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path).copy()
    required = {"method", "group_id", "fixed_id", "moving_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["method"] = df["method"].astype(str).str.lower()
    if "domain" in df.columns:
        df["domain"] = df["domain"].astype(str)

    df["method_display"] = (
        df["method"].map(METHOD_DISPLAY).fillna(df["method"])
    )
    df["pair_id"] = (
        "g"
        + df["group_id"].astype(str)
        + ":"
        + df["fixed_id"].astype(str)
        + "→"
        + df["moving_id"].astype(str)
    )
    df["pair_type"] = (
        df["fixed_id"].astype(str)
        + "→"
        + df["moving_id"].astype(str)
    )
    return df


# =============================================================================
# Aggregation helpers
# =============================================================================

def aggregate_pair_level(
    df: pd.DataFrame,
    metrics: Sequence[str] | str = ("dice", "spatial_corr", "center_shift"),
) -> pd.DataFrame:
    """Average over domains first so each slice pair contributes one dot."""
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = list(metrics)

    missing = [metric for metric in metrics if metric not in df.columns]
    if missing:
        raise ValueError(f"Metric column(s) not found: {missing}")

    group_cols = [
        "method",
        "method_display",
        "group_id",
        "fixed_id",
        "moving_id",
        "pair_id",
        "pair_type",
    ]
    return (
        df.groupby(group_cols, as_index=False, dropna=False)[metrics]
        .mean()
    )


def aggregate_domain_near_far(df, method="axis", group_id=None, metric="dice"):
    if group_id is None:
        raise ValueError("group_id must be provided for domain-based analysis")
    sub = subset_one_group(df, group_id=group_id, method=method)
    return (
        sub.groupby(["domain", "distance_bin"], as_index=False)[metric]
        .mean()
    )


def find_best_competing_method(df, metric="dice"):
    pair_df = aggregate_pair_level(df, metric)
    sub = pair_df[~pair_df["method"].isin(["axis", "unaligned"])].copy()
    if sub.empty:
        raise ValueError("No competing methods found")

    summary = sub.groupby(
        ["method", "method_display"], as_index=False
    )[metric].mean()
    if metric_direction(metric) == "higher":
        best_row = summary.loc[summary[metric].idxmax()]
    else:
        best_row = summary.loc[summary[metric].idxmin()]
    return best_row["method"], best_row["method_display"]


# =============================================================================
# Shared box + dot reports
# =============================================================================

def _resolve_boxplot_order(
    df: pd.DataFrame,
    metric: str,
    sort_by_median: bool,
) -> List[str]:
    if not sort_by_median:
        return get_method_order(df)
    return (
        df.groupby("method_display")[metric]
        .median()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )


def _draw_horizontal_boxplot(
    ax,
    data: pd.DataFrame,
    metric: str,
    order: Sequence[str],
    palette_map: Dict[str, str],
    style: BoxplotStyle,
) -> None:
    sns.boxplot(
        data=data,
        y="method_display",
        x=metric,
        order=list(order),
        palette=palette_map,
        ax=ax,
        showfliers=style.show_fliers,
        width=style.box_width,
        linewidth=style.box_linewidth,
        orient="h",
        saturation=style.box_saturation,
    )
    sns.stripplot(
        data=data,
        y="method_display",
        x=metric,
        order=list(order),
        ax=ax,
        color=style.dot_color,
        size=style.dot_size,
        alpha=style.dot_alpha,
        jitter=style.dot_jitter,
        orient="h",
    )

    ax.set_xlabel(get_metric_label(metric),x=0.45)
    ax.set_ylabel("")
    ax.grid(axis=style.grid_axis, alpha=style.grid_alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=style.metric_tick_fontsize)
    ax.tick_params(axis="y", labelsize=style.method_tick_fontsize, pad=-2)
    # for tick_label in ax.get_yticklabels():
    #     tick_label.set_fontstyle("italic")


def plot_overall_metrics_across_methods_row(
    df: pd.DataFrame,
    metrics: Sequence[str],
    save_path=None,
    figsize: Optional[Tuple[float, float]] = None,
    share_y: bool = False,
    style: BoxplotStyle = SHARED_BOXPLOT_STYLE,
    sort_methods_by_median: bool = False,
    show: bool = True,
):
    """Overall method report; each dot is one pair-level domain average.

    By default method order and colors are fixed globally, which is the intended
    cross-dataset publication behavior. Set ``sort_methods_by_median=True`` only
    for exploratory plots where metric-specific ranking is desired.
    """
    metrics = list(metrics)
    if not metrics:
        raise ValueError("`metrics` must contain at least one metric.")

    set_statistic_plot_theme()
    if figsize is None:
        figsize = (style.panel_width * len(metrics), style.figure_height)

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=figsize,
        sharey=share_y,
        constrained_layout=False,
        gridspec_kw={"wspace": style.wspace},
    )
    if len(metrics) == 1:
        axes = np.asarray([axes], dtype=object)

    all_methods = get_method_order(df)
    palette_map = get_global_method_palette(all_methods)

    for panel_idx, (ax, metric) in enumerate(zip(axes, metrics)):
        pair_df = aggregate_pair_level(df, metric)
        order = _resolve_boxplot_order(
            pair_df,
            metric=metric,
            sort_by_median=sort_methods_by_median,
        )
        _draw_horizontal_boxplot(
            ax=ax,
            data=pair_df,
            metric=metric,
            order=order,
            palette_map=palette_map,
            style=style,
        )
        # ax.set_title(
        #     get_metric_label(metric),
        #     fontsize=style.title_fontsize,
        #     pad=style.title_pad,
        # )

        if share_y and panel_idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    fig.tight_layout()
    save_or_show(
        fig,
        save_path=save_path,
        dpi=style.dpi,
        show=show,
        close=not show,
    )
    return fig, axes


def plot_single_pair_metrics_across_methods_row(
    df: pd.DataFrame,
    metrics: Sequence[str],
    group_id: Optional[int] = None,
    fixed_id: Optional[int] = None,
    moving_id: Optional[int] = None,
    save_path=None,
    figsize: Optional[Tuple[float, float]] = None,
    share_y: bool = False,
    style: BoxplotStyle = SHARED_BOXPLOT_STYLE,
    sort_methods_by_median: bool = True,
    show: bool = True,
):
    """Single-pair domain-level box + dot report using the shared style."""
    metrics = list(metrics)
    if not metrics:
        raise ValueError("`metrics` must contain at least one metric.")

    plot_df = df.copy()
    if group_id is not None:
        plot_df = plot_df[plot_df["group_id"] == group_id]
    if fixed_id is not None:
        plot_df = plot_df[plot_df["fixed_id"] == fixed_id]
    if moving_id is not None:
        plot_df = plot_df[plot_df["moving_id"] == moving_id]
    if plot_df.empty:
        raise ValueError("No rows left after filtering.")

    pair_info = plot_df[["group_id", "fixed_id", "moving_id"]].drop_duplicates()
    if len(pair_info) != 1:
        raise ValueError(
            f"Expected exactly one pair, but found {len(pair_info)} pairs.\n{pair_info}"
        )

    set_statistic_plot_theme()
    if figsize is None:
        figsize = (style.panel_width * len(metrics), style.figure_height)

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=figsize,
        sharey=share_y,
        constrained_layout=False,
        gridspec_kw={"wspace": style.wspace},
    )
    if len(metrics) == 1:
        axes = np.asarray([axes], dtype=object)

    palette_map = get_global_method_palette(get_method_order(plot_df))
    for panel_idx, (ax, metric) in enumerate(zip(axes, metrics)):
        order = _resolve_boxplot_order(
            plot_df,
            metric=metric,
            sort_by_median=sort_methods_by_median,
        )
        _draw_horizontal_boxplot(
            ax=ax,
            data=plot_df,
            metric=metric,
            order=order,
            palette_map=palette_map,
            style=style,
        )
        # ax.set_title(
        #     f"{get_metric_label(metric)}",
        #     fontsize=style.title_fontsize,
        #     pad=style.title_pad,
        # )
        if share_y and panel_idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    fig.tight_layout()
    save_or_show(
        fig,
        save_path=save_path,
        dpi=style.dpi,
        show=show,
        close=not show,
    )
    return fig, axes


# Backward-compatible one-metric wrappers.
def plot_overall_metric_across_methods_vertical_compact(
    df,
    metric="dice",
    save_path=None,
    figsize=(5, 5),
):
    return plot_overall_metrics_across_methods_row(
        df,
        [metric],
        save_path=save_path,
        figsize=figsize,
        show=True,
    )


def plot_single_pair_metric_across_methods_vertical_compact(
    df,
    group_id=None,
    fixed_id=None,
    moving_id=None,
    metric="dice",
    save_path=None,
    figsize=(5, 5),
):
    return plot_single_pair_metrics_across_methods_row(
        df,
        [metric],
        group_id=group_id,
        fixed_id=fixed_id,
        moving_id=moving_id,
        save_path=save_path,
        figsize=figsize,
        show=True,
    )


# =============================================================================
# Heatmaps / distance plots retained from the original utility
# =============================================================================

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
    agg = sub.groupby(["domain", "pair_type"], as_index=False)[metric].mean()
    pivot = agg.pivot(index="domain", columns="pair_type", values=metric)

    pair_order = ["0→1", "0→2", "0→3", "1→2", "2→3"]
    pair_order = [pair for pair in pair_order if pair in pivot.columns]
    pivot = pivot.reindex(index=get_domain_order(sub), columns=pair_order)

    fig, ax = plt.subplots(figsize=figsize)
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
    fig.tight_layout()
    save_or_show(fig, save_path=save_path)
    return fig, ax


def plot_domain_distance_method_lines(
    df,
    metric="dice",
    group_id=None,
    methods=None,
    save_path=None,
    figsize=None,
    DISTANCE_ORDER=None,
):
    if group_id is None:
        raise ValueError("group_id must be provided")

    sub = df[df["group_id"] == group_id].copy()
    if methods is not None:
        methods = [method.lower() for method in methods]
        sub = sub[sub["method"].isin(methods)]
    if sub.empty:
        raise ValueError(f"No rows found for group_id={group_id}")

    agg = sub.groupby(
        ["method_display", "domain", "distance_um"], as_index=False
    )[metric].mean()

    domain_order = get_domain_order(sub)
    method_order = get_method_order(sub)
    palette_map = get_global_method_palette(method_order)

    agg["domain"] = pd.Categorical(
        agg["domain"], categories=domain_order, ordered=True
    )
    agg["method_display"] = pd.Categorical(
        agg["method_display"], categories=method_order, ordered=True
    )
    agg = agg.sort_values(["domain", "method_display", "distance_um"])

    n_domains = len(domain_order)
    if figsize is None:
        figsize = (
            5.2 * min(n_domains, 3),
            4.0 * int(np.ceil(n_domains / 3)),
        )

    g = sns.FacetGrid(
        agg,
        col="domain",
        col_wrap=3,
        height=3.6,
        sharey=False,
        hue="method_display",
        hue_order=method_order,
        palette=palette_map,
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

    if DISTANCE_ORDER is not None:
        for ax in g.axes.flat:
            ax.set_xticks(
                [
                    distance
                    for distance in DISTANCE_ORDER
                    if distance in agg["distance_um"].unique()
                ]
            )
            ax.grid(axis="y", alpha=0.25)

    g.figure.set_size_inches(*figsize)
    g.figure.suptitle(
        f"Group {group_id}: method comparison across exact distances "
        f"({get_metric_label(metric)})",
        y=1.02,
    )
    g.figure.tight_layout()
    save_or_show(g.figure, save_path=save_path)
    return g
