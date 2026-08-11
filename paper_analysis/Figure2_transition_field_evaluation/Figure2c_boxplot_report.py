from utils_statistic_plot import *
# ============================================================
# Config
# ============================================================


DISTANCE_ORDER = [10, 300, 310, 320]
DISTANCE_BIN_ORDER = ["Near (10 μm)", "Far (300–320 μm)"]
CVS_ROOT = Path("/media/huifang/data/registration/evaluation/transition_field")

# ============================================================
# Basic helpers
# ============================================================

def format_distance_label(x):
    """
    10 -> 10um
    300 -> 300um
    """
    try:
        return f"{int(float(x))}um"
    except Exception:
        return str(x)


def format_domain_label(x):
    """
    1.0 -> L3
    2.0 -> L4
    3.0 -> L5
    4.0 -> L6
    5.0 -> WM
    """
    mapping = {
        "1": "L3",
        "1.0": "L3",
        1: "L3",
        1.0: "L3",
        "2": "L4",
        "2.0": "L4",
        2: "L4",
        2.0: "L4",
        "3": "L5",
        "3.0": "L5",
        3: "L5",
        3.0: "L5",
        "4": "L6",
        "4.0": "L6",
        4: "L6",
        4.0: "L6",
        "5": "WM",
        "5.0": "WM",
        5: "WM",
        5.0: "WM",
    }
    return mapping.get(x, mapping.get(str(x), str(x)))


from typing import List, Optional, Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# Global method palette
# Use this everywhere across datasets
# ------------------------------------------------------------
GLOBAL_METHOD_PALETTE = {
    "Unaligned":   "#B8B8B8",  # soft gray
    "SimpleITK":   "#8FA8C9",  # muted blue
    "VoxelMorph":  "#A8C4A2",  # sage green
    "PASTE":       "#C7A0C9",  # dusty purple
    "GPSA":        "#D8B384",  # muted sand
    "SANTO":       "#8FC7C2",  # soft teal
    "SpatialZ":    "#C98F8F",  # muted rose
    "AXIS":        "#D45A45",  # main highlight red
    "NICE-Trans":   "#7C90B0",  # steel blue
}


def get_global_method_palette(method_names: List[str]) -> Dict[str, str]:
    """
    Return a palette dict for the provided method names, using the global palette.
    Unknown methods get fallback muted colors.
    """
    fallback_colors = [
        "#7C90B0",  # steel blue
        "#9AAE7F",  # olive-sage
        "#B79AA8",  # soft plum
        "#C49A6C",  # muted amber
        "#7FA7A6",  # soft teal
        "#A67C73",  # muted clay
    ]

    palette = {}
    fallback_idx = 0
    for m in method_names:
        # if m in GLOBAL_METHOD_PALETTE:
        palette[m] = GLOBAL_METHOD_PALETTE[m]
        # else:
        #     palette[m] = fallback_colors[fallback_idx % len(fallback_colors)]
        #     fallback_idx += 1
    return palette




def plot_single_pair_metrics_across_methods_row(
    df: pd.DataFrame,
    metrics: List[str],
    group_id: Optional[int] = None,
    fixed_id: Optional[int] = None,
    moving_id: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] =(13, 5.2),
    share_y: bool = False,
):
    """
    Plot one selected pair using domain-level values directly.

    Each dot = one domain metric for the selected pair.
    Each box = distribution across domains for one method.

    Changes from the original version:
    1. accepts a list of metrics (e.g. 3 metrics in one row)
    2. removes average markers
    3. sorts methods within each subplot by median value of that metric
    4. uses the same global method color palette as previous plots

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing domain-level metrics.
    metrics : list of str
        Metrics to plot, e.g. ["dice", "spatial_corr", "center_shift"].
    group_id, fixed_id, moving_id : optional
        If provided, filter the dataframe to the selected pair.
    save_path : str or None
        Output path.
    figsize : tuple
        Figure size.
    share_y : bool
        Whether to share the y-axis across subplots. Usually False here,
        because each metric is sorted independently by median.
    """
    if len(metrics) == 0:
        raise ValueError("`metrics` must contain at least one metric.")

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

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=figsize,
        sharey=share_y,
        gridspec_kw={"wspace": 0.7}
    )
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        dfm = plot_df.copy()

        # sort methods by median value of the current metric
        median_order = (
            dfm.groupby("method_display")[metric]
            .median()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        palette_map = get_global_method_palette(median_order)

        sns.boxplot(
            data=dfm,
            y="method_display",
            x=metric,
            order=median_order,
            palette=palette_map,
            ax=ax,
            showfliers=False,
            width=0.55,
            linewidth=1.1,
            orient="h",
            saturation=1.0,
        )

        sns.stripplot(
            data=dfm,
            y="method_display",
            x=metric,
            order=median_order,
            ax=ax,
            color="0.25",
            size=4.0,
            alpha=0.45,
            jitter=0.14,
            orient="h",
        )

        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(f"{get_metric_label(metric)} | Per Marker", fontsize=16, pad=8)
        ax.grid(axis="x", alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if share_y and ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)

        ax.tick_params(axis="x", labelsize=15)
        # ax.tick_params(axis="y", labelsize=14, rotation=45)
        ax.tick_params(axis="y", labelsize=14)

    fig.tight_layout()
    save_or_show(fig, save_path=save_path)
def plot_overall_metrics_across_methods_row(
    df: pd.DataFrame,
    metrics: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (20, 5.5),
    share_y: bool = False,
):
    if len(metrics) == 0:
        raise ValueError("`metrics` must contain at least one metric.")

    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=figsize,
        sharey=share_y,
        constrained_layout=True,
        gridspec_kw={
            "wspace": 0.65,
        },
    )

    if n_metrics == 1:
        axes = [axes]

    # Build once so method colors remain stable across all panels.
    all_methods = df["method_display"].dropna().unique().tolist()
    palette_map = get_global_method_palette(all_methods)

    for ax, metric in zip(axes, metrics):
        pair_df = aggregate_pair_level(df, metric).copy()

        median_order = (
            pair_df.groupby("method_display")[metric]
            .median()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        sns.boxplot(
            data=pair_df,
            y="method_display",
            x=metric,
            order=median_order,
            palette=palette_map,
            ax=ax,
            showfliers=False,
            width=0.55,
            linewidth=1.1,
            orient="h",
            saturation=1.0,
        )

        sns.stripplot(
            data=pair_df,
            y="method_display",
            x=metric,
            order=median_order,
            ax=ax,
            color="0.25",
            size=3.8,
            alpha=0.40,
            jitter=0.14,
            orient="h",
        )

        ax.set_title(
            get_metric_label(metric),
            fontsize=16,
            pad=8,
        )

        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.grid(
            axis="x",
            alpha=0.20,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if share_y and ax is not axes[0]:
            ax.tick_params(
                axis="y",
                labelleft=False,
            )
        else:
            ax.tick_params(
                axis="y",
                labelsize=13,
                pad=2,
            )

        ax.tick_params(
            axis="x",
            labelsize=14,
        )

    save_or_show(
        fig,
        save_path=save_path,
    )

def plot_overall_metric_across_methods(
    df,
    metric="dice",
    save_path=None,
    figsize=(8, 5),
):
    """
    One figure for one metric.
    Each dot = one pair-level average across domains.

    Revised version:
    1. keeps one metric input
    2. removes the average marker
    3. sorts methods by median value of the metric
    4. uses the global method palette
    5. keeps the original layout:
       methods on x-axis, metric on y-axis
    """
    pair_df = aggregate_pair_level(df, metric).copy()

    # sort methods by median value of current metric
    order = (
        pair_df.groupby("method_display")[metric]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    palette_map = get_global_method_palette(order)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    sns.boxplot(
        data=pair_df,
        x="method_display",
        y=metric,
        order=order,
        palette=palette_map,
        ax=ax,
        showfliers=False,
        width=0.6,
        linewidth=1.2,
        saturation=1.0,
    )

    sns.stripplot(
        data=pair_df,
        x="method_display",
        y=metric,
        order=order,
        ax=ax,
        color="0.25",
        size=5.5,
        alpha=0.45,
        jitter=0.18,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Neighborhood label consistency", fontsize=16,weight='bold')
    # ax.set_title(f"Overall method comparison: {get_metric_label(metric)}", fontsize=13, pad=8)
    ax.grid(axis="y", alpha=0.25)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # keep original top x-axis style
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("bottom")
    ax.tick_params(
        axis="x",
        top=False,
        labeltop=False,
        bottom=True,
        labelbottom=True,
        length=0,
        rotation=25,
        labelsize=15,
        pad=6,
    )
    ax.tick_params(axis="y", labelsize=14)

    fig = ax.figure
    fig.tight_layout()
    save_or_show(fig, save_path=save_path)

def load_metrics_csv(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)


    # missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    # if missing:
    #     raise ValueError(
    #         f"CSV is missing required columns: {missing}\n"
    #         f"Available columns: {list(df.columns)}"
    #     )

    df = df.copy()
    df["method"] = df["method"].astype(str).str.lower()
    df["domain"] = df["domain"].astype(str)

    df["method_display"] = df["method"].map(METHOD_DISPLAY).fillna(df["method"])
    df["pair_id"] = (
        "g" + df["group_id"].astype(str)
        + ":" + df["fixed_id"].astype(str)
        + "→" + df["moving_id"].astype(str)
    )
    df["pair_type"] = (
        df["fixed_id"].astype(str) + "→" + df["moving_id"].astype(str)
    )

    # df["distance_bin"] = np.where(
    #     df["distance_um"] == 10,
    #     "Near (10 μm)",
    #     "Far (300–320 μm)"
    # )
    # print(df.columns)
    # test = input()

    return df
def aggregate_domain_distance(
    df,
    method="axis",
    group_id=None,
    metric="dice",
    mode="distance_bin",
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


from matplotlib.colors import LinearSegmentedColormap
def make_light_distance_cmap():
    colors = [
        "#f7fbff",  # near white
        "#dbe9f6",
        "#b7d1ea",
        "#7fb3d5",
        "#4f8fc4",
        "#2f6690",  # deep blue
    ]
    return LinearSegmentedColormap.from_list("single_blue", colors, N=256)
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
    # cmap = default_cmap(metric)
    cmap = make_light_distance_cmap()

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
            ytick_fontsize=14,
            xtick_shift=xtick_shift,
            ytick_shift=ytick_shift,
        )

        if method_display =="Unaligned":
            method_display = "Standardized initialization"
        ax.set_title(method_display,weight="bold")
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


def run_dlpfc():
    CSV_PATH = CVS_ROOT/"dlpfc/dlpfc_metrics_all_methods.csv"

    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.95,
    )

    df = load_metrics_csv(CSV_PATH)
    plot_overall_metrics_across_methods_row(df,["dice","spatial_corr","center_shift","neighbor_accuracy"],save_path =None)


    # group_id = 1
    #
    # plot_domain_distance_method_heatmaps(
    #     df,
    #     metric="dice",
    #     group_id=group_id,
    #     format_distance_label=format_distance_label,
    #     format_domain_label=format_domain_label,
    #     xtick_shift=(0.0, 0.05),
    #     ytick_shift=(0.05, 0.0),
    #     DISTANCE_ORDER=DISTANCE_ORDER
    #
    # )

def run_xenium_mouse():
    # CSV_PATH = "/media/huifang/data/registration/xenium/mouse_brain/evaluation/mouse_time_domain_metrics_all_methods.csv"
    CSV_PATH = CVS_ROOT / "mouse/mouse_metrics_all_methods.csv"
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.95,
    )
    df = load_metrics_csv(CSV_PATH)
    plot_overall_metrics_across_methods_row(df, ["dice", "spatial_corr", "center_shift","neighbor_accuracy"],
                                            save_path=None)

def run_xenium_breast():
    CSV_PATH = CVS_ROOT / "breast/breast_metrics_all_methods.csv"
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.95,
    )
    df = load_metrics_csv(CSV_PATH)
    plot_single_pair_metrics_across_methods_row(
        df,
        ["dice", "spatial_corr", "center_shift","neighbor_accuracy"],
        group_id=0,
        fixed_id=0,
        moving_id=1,
        save_path=None
    )


def run_scc():
    CSV_PATH = CVS_ROOT / "scc/scc_metrics_all_methods.csv"
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.95,
    )
    df = load_metrics_csv(CSV_PATH)
    plot_overall_metrics_across_methods_row(df, ["neighbor_accuracy"],figsize=(5,5.2),
        save_path="/media/huifang/data/registration/evaluation/SCC.png"
    )


# ============================================================
# Example usage: run one figure at a time
# ============================================================

if __name__ == "__main__":
    # run_dlpfc()



    # run_xenium_breast()
    run_scc()
    # run_xenium_mouse()


    #
    # # CSV_PATH = "/media/huifang/data/registration/DLPFC/evaluation/dlpfc_domain_metrics_all_methods.csv"
    # # CSV_PATH = "/media/huifang/data/registration/xenium/mouse_brain/evaluation/mouse_time_domain_metrics_all_methods.csv"
    # # CSV_PATH = "/media/huifang/data/registration/breast_cancer/evaluation/breast_cancer_domain_metrics_all_methods.csv"
    # CSV_PATH = "/media/huifang/data/registration/SCC/evaluation/scc_molecular_metrics_all_methods.csv"
    #
    # sns.set_theme(
    #     style="whitegrid",
    #     context="talk",
    #     font_scale=0.95,
    # )
    #
    # df = load_metrics_csv(CSV_PATH)
    #
    # # --------------------------------------------------------
    # # Overall comparison across all methods
    # # Uncomment one at a time
    # # --------------------------------------------------------
    #
    # # plot_single_pair_metric_across_methods_vertical_compact(
    # #     df,
    # #     group_id=0,
    # #     fixed_id=0,
    # #     moving_id=1,
    # #     metric="spatial_corr",
    # # )
    #
    #
    # plot_overall_metric_across_methods(df, metric="spot_topk")
    # # plot_overall_metric_across_methods_vertical_compact(df, metric="spatial_corr")
    # # plot_overall_metric_across_methods_vertical_compact(df, metric="center_shift")
    #
    # group_id = 2
    # # plot_domain_distance_method_lines(df, metric="spot_topk", group_id=group_id)
    # plot_domain_distance_method_heatmaps(
    #     df,
    #     metric="spot_topk",
    #     group_id=group_id,
    #     # format_distance_label=format_distance_label,
    #     # format_domain_label=format_domain_label,
    #     xtick_shift=(0.0, 0.05),
    #     ytick_shift=(0.05, 0.0),
    #     DISTANCE_ORDER=DISTANCE_ORDER
    # )
    #
    # # plot_axis_domain_heatmap(df, metric="dice", method="axis", group_id=group_id, mode="distance_bin")
    # # plot_axis_domain_heatmap(df, metric="spatial_corr", method="axis", group_id=group_id, mode="distance_bin")
    # # plot_axis_domain_heatmap(df, metric="center_shift", method="axis", group_id=group_id, mode="distance_bin")
    #
    # # plot_axis_domain_heatmap(df, metric="dice", method="axis", group_id=group_id, mode="distance_um")
    # # plot_axis_domain_heatmap(df, metric="spatial_corr", method="axis", group_id=group_id, mode="distance_um")
    # # plot_axis_domain_heatmap(df, metric="center_shift", method="axis", group_id=group_id, mode="distance_um")
    #
    # # plot_axis_domain_pairtype_heatmap(df, metric="dice", method="gpsa", group_id=group_id)
    # # plot_axis_domain_pairtype_heatmap(df, metric="spatial_corr", method="axis", group_id=group_id)
    # # plot_axis_domain_pairtype_heatmap(df, metric="center_shift", method="axis", group_id=group_id)