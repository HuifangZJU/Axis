#!/usr/bin/env python3
from __future__ import annotations
from axis.settings import data_path

"""Streamlined ablation/noise robustness plotting.

The line-chart appearance is controlled from ``LinePlotStyle`` so axis ticks,
labels, limits, typography, line emphasis, and right-side annotations can be
adjusted without editing plotting logic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter


# =============================================================================
# INPUT DATA
# =============================================================================

DATA = {
    "Unaligned": [0.4406, 0.4406, 0.4406, 0.4406, 0.4406, 0.4406],
    "SimpleITK": [0.4835, 0.4878, 0.4818, 0.3856, 0.3394, 0.3806],
    "VoxelMorph": [0.4737, 0.4518, 0.4578, 0.4340, 0.4296, 0.3965],
    "NICE-Trans": [0.5003, 0.4860, 0.4757, 0.4392, 0.3927, 0.3656],
    "PASTE": [0.4638, 0.4638, 0.4638, 0.4638, 0.4638, 0.4638],
    "GPSA": [0.4641, 0.4641, 0.4641, 0.4641, 0.4641, 0.4641],
    "SANTO": [0.4521, 0.4521, 0.4521, 0.4521, 0.4521, 0.4521],
    "AXIS": [0.5025, 0.4843, 0.4826, 0.4804, 0.4873, 0.4729],
}

NOISE_LEVELS = np.array([0, 1, 2, 3, 4, 5], dtype=float)
METRIC_NAME = "Molecular correspondence"

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

METHOD_ORDER = (
    "Unaligned",
    "PASTE",
    "GPSA",
    "SANTO",
    "SimpleITK",
    "VoxelMorph",
    "NICE-Trans",
    "AXIS",
)

METHOD_LINESTYLE = {
    "Unaligned": "-",
    "SimpleITK": "-",
    "VoxelMorph": "-",
    "NICE-Trans": "-",
    "PASTE": "-",
    "GPSA": "--",
    "SANTO": "-",
    "SpatialZ": "-",
    "AXIS": "-",
}

METHOD_MARKER = {
    "Unaligned": "o",
    "SimpleITK": "o",
    "VoxelMorph": "o",
    "NICE-Trans": "o",
    "PASTE": "s",
    "GPSA": "x",
    "SANTO": "o",
    "SpatialZ": "o",
    "AXIS": "o",
}

OUTPUT_PATH = Path(
    data_path("paper_analysis/"
    "Figure2_transition_evaluation/j_ablation.png")
)


# =============================================================================
# CENTRALIZED LINE-PLOT STYLE
# =============================================================================

@dataclass(frozen=True)
class LinePlotStyle:
    # Figure / export
    figsize: Tuple[float, float] = (12.0, 6.0)
    dpi: int = 400

    # Titles and axis labels
    title: str = "Robustness to input noise"
    title_fontsize: float = 18.0
    title_pad: float = 12.0
    xlabel: str = "Noise level"
    xlabel_fontsize: float = 18.0
    xlabel_pad: float = 8.0
    ylabel: str = METRIC_NAME
    ylabel_fontsize: float = 18.0
    ylabel_pad: float = 8.0

    # ------------------------------------------------------------------
    # Tick controls -- adjust these for x/y tick labels.
    # ------------------------------------------------------------------
    x_ticks: Optional[Tuple[float, ...]] = (0, 1, 2, 3, 4, 5)
    x_tick_labels: Optional[Tuple[str, ...]] = None
    x_tick_fontsize: float = 16.0
    x_tick_rotation: float = 0.0
    x_tick_pad: float = 5.0

    y_ticks: Optional[Tuple[float, ...]] = None
    y_tick_labels: Optional[Tuple[str, ...]] = None
    y_tick_format: str = "%.1f"
    y_tick_fontsize: float = 16.0
    y_tick_rotation: float = 0.0
    y_tick_pad: float = 5.0

    tick_length: float = 4.0
    tick_width: float = 1.0

    # Axis ranges. Set to None for automatic limits.
    xlim: Optional[Tuple[float, float]] = (-0.1, 6.2)
    ylim: Optional[Tuple[float, float]] = None

    # Grid
    y_grid_alpha: float = 0.22
    x_grid_alpha: float = 0.08

    # Methods to emphasize
    highlight_method: str = "AXIS"
    image_baseline: str = "NICE-Trans"

    # Main AXIS style
    highlight_linewidth: float = 3.2
    highlight_markersize: float = 8.5

    # Best non-AXIS baseline
    best_baseline_linewidth: float = 3
    best_baseline_markersize: float = 7.5

    # Image baseline
    image_baseline_linewidth: float = 3
    image_baseline_markersize: float = 7.5

    # Remaining baselines
    other_linewidth: float = 3
    other_markersize: float = 5.0
    other_alpha: float = 1.0

    # Endpoint labels
    show_right_labels: bool = True
    right_label_x_offset: float = 0.15
    right_label_fontsize: float = 30.0
    right_label_min_gap: float = 0.008
    show_delta_in_label: bool = True
    delta_decimals: int = 3

    # Highlight first/last AXIS points
    highlight_endpoint_size: float = 90.0
    highlight_endpoint_edgecolor: str = "white"
    highlight_endpoint_edgewidth: float = 1.0

    # Figure margins. None means use tight_layout().
    left: Optional[float] = None
    right: Optional[float] = None
    bottom: Optional[float] = None
    top: Optional[float] = None


# =============================================================================
# DATA / STYLE HELPERS
# =============================================================================

def build_score_dataframe(
    data: Mapping[str, Sequence[float]],
    noise_levels: Sequence[float],
) -> pd.DataFrame:
    noise_levels = np.asarray(noise_levels, dtype=float)
    df = pd.DataFrame(data, index=noise_levels)
    if len(df) != len(noise_levels):
        raise ValueError("Noise-level count does not match score-vector length.")
    df.index.name = "Noise level"
    return df


def find_best_baseline(df: pd.DataFrame, highlight_method: str) -> str:
    mean_score = df.mean(axis=0)
    candidates = mean_score.drop(labels=[highlight_method], errors="ignore")
    if candidates.empty:
        raise ValueError("No competing baseline is available.")
    return str(candidates.idxmax())


def method_style(
    method: str,
    *,
    best_baseline: str,
    style: LinePlotStyle,
) -> Dict[str, float | str]:
    if method not in GLOBAL_METHOD_PALETTE:
        raise KeyError(
            f"No plotting color is defined for method {method!r} in "
            "GLOBAL_METHOD_PALETTE."
        )

    color = GLOBAL_METHOD_PALETTE[method]

    if method == style.highlight_method:
        return {
            "color": color,
            "lw": style.highlight_linewidth,
            "ms": style.highlight_markersize,
            "alpha": 1.0,
            "zorder": 6,
        }
    if method == best_baseline:
        return {
            "color": color,
            "lw": style.best_baseline_linewidth,
            "ms": style.best_baseline_markersize,
            "alpha": 1,
            "zorder": 5,
        }
    if method == style.image_baseline:
        return {
            "color": color,
            "lw": style.image_baseline_linewidth,
            "ms": style.image_baseline_markersize,
            "alpha": 1,
            "zorder": 5,
        }
    return {
        "color": color,
        "lw": style.other_linewidth,
        "ms": style.other_markersize,
        "alpha": style.other_alpha,
        "zorder": 3,
    }


def _nudge_label_y(y: float, used: list[float], min_gap: float) -> float:
    y_new = float(y)
    while any(abs(y_new - yy) < min_gap for yy in used):
        y_new -= min_gap
    used.append(y_new)
    return y_new


def _apply_tick_style(ax, style: LinePlotStyle) -> None:
    """Apply all x/y tick locations, text, font sizes, rotations and padding."""
    if style.x_ticks is not None:
        ax.set_xticks(style.x_ticks)
        if style.x_tick_labels is not None:
            if len(style.x_tick_labels) != len(style.x_ticks):
                raise ValueError("x_tick_labels must have the same length as x_ticks.")
            ax.set_xticklabels(style.x_tick_labels)

    if style.y_ticks is not None:
        ax.set_yticks(style.y_ticks)
        if style.y_tick_labels is not None:
            if len(style.y_tick_labels) != len(style.y_ticks):
                raise ValueError("y_tick_labels must have the same length as y_ticks.")
            ax.set_yticklabels(style.y_tick_labels)
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter(style.y_tick_format))
    elif style.y_tick_labels is not None:
        raise ValueError("Set y_ticks when supplying y_tick_labels.")
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter(style.y_tick_format))

    ax.tick_params(
        axis="x",
        labelsize=style.x_tick_fontsize,
        labelrotation=style.x_tick_rotation,
        pad=style.x_tick_pad,
        length=style.tick_length,
        width=style.tick_width,
    )
    ax.tick_params(
        axis="y",
        labelsize=style.y_tick_fontsize,
        labelrotation=style.y_tick_rotation,
        pad=style.y_tick_pad,
        length=style.tick_length,
        width=style.tick_width,
    )

    # Explicit horizontal/vertical alignment makes rotated ticks predictable.
    for label in ax.get_xticklabels():
        if style.x_tick_rotation == 0:
            label.set_ha("center")
        elif style.x_tick_rotation > 0:
            label.set_ha("right")
        else:
            label.set_ha("left")
    for label in ax.get_yticklabels():
        label.set_va("center")


# =============================================================================
# MAIN LINE CHART
# =============================================================================

def plot_noise_robustness(
    data: Mapping[str, Sequence[float]],
    noise_levels: Sequence[float],
    *,
    style: Optional[LinePlotStyle] = None,
    save_path: Optional[Path | str] = None,
    show: bool = True,
):
    style = LinePlotStyle() if style is None else style
    df = build_score_dataframe(data, noise_levels)

    if style.highlight_method not in df.columns:
        raise KeyError(f"Highlight method {style.highlight_method!r} is missing from the data.")

    best_baseline = find_best_baseline(df, style.highlight_method)
    x = np.asarray(df.index, dtype=float)

    fig, ax = plt.subplots(figsize=style.figsize)

    # Keep the plotting panel on a clean white background.
    ax.set_facecolor("white")

    # Curves.
    for method in df.columns:
        st = method_style(method, best_baseline=best_baseline, style=style)
        markerfacecolor = "white" if method == "GPSA" else st["color"]

        ax.plot(
            x,
            df[method].to_numpy(dtype=float),
            marker=METHOD_MARKER.get(method, "o"),
            linestyle=METHOD_LINESTYLE.get(method, "-"),
            linewidth=st["lw"],
            markersize=st["ms"],
            color=st["color"],
            markerfacecolor=markerfacecolor,
            markeredgecolor=st["color"],
            markeredgewidth=1.2,
            alpha=st["alpha"],
            zorder=st["zorder"],
        )

    # Right-side direct labels.
    if style.show_right_labels:
        last_x = float(x[-1])
        first_x = float(x[0])
        label_order = df.iloc[-1].sort_values(ascending=False).index.tolist()
        used_y: list[float] = []

        for method in label_order:
            y_end = float(df.iloc[-1][method])
            y_start = float(df.iloc[0][method])
            y_label = _nudge_label_y(y_end, used_y, style.right_label_min_gap)
            st = method_style(method, best_baseline=best_baseline, style=style)

            label_text = method
            if style.show_delta_in_label:
                delta = y_end - y_start
                label_text += f"  ({delta:+.{style.delta_decimals}f})"

            ax.text(
                last_x + style.right_label_x_offset,
                y_label,
                label_text,
                color=st["color"],
                fontsize=style.right_label_fontsize,
                fontweight="bold",
                va="center",
                ha="left",
                clip_on=False,
            )

    # Emphasize first/last points of the highlighted method.
    ax.scatter(
        [x[0], x[-1]],
        [df.iloc[0][style.highlight_method], df.iloc[-1][style.highlight_method]],
        s=style.highlight_endpoint_size,
        color=GLOBAL_METHOD_PALETTE[style.highlight_method],
        edgecolor=style.highlight_endpoint_edgecolor,
        linewidth=style.highlight_endpoint_edgewidth,
        zorder=7,
    )

    # Axis geometry and text.
    if style.xlim is not None:
        ax.set_xlim(*style.xlim)
    if style.ylim is not None:
        ax.set_ylim(*style.ylim)

    ax.set_xlabel(style.xlabel, fontsize=style.xlabel_fontsize, labelpad=style.xlabel_pad)
    ax.set_ylabel(style.ylabel, fontsize=style.ylabel_fontsize, labelpad=style.ylabel_pad)
    ax.set_title(style.title, fontsize=style.title_fontsize, pad=style.title_pad)

    _apply_tick_style(ax, style)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=style.y_grid_alpha)
    ax.grid(axis="x", alpha=style.x_grid_alpha)

    if None not in (style.left, style.right, style.bottom, style.top):
        fig.subplots_adjust(
            left=style.left,
            right=style.right,
            bottom=style.bottom,
            top=style.top,
        )
    else:
        fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=style.dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# =============================================================================
# OPTIONAL HEATMAP (retained from original script)
# =============================================================================

def plot_noise_heatmap_from_summary(
    data,
    metric_label="Top-k label consistency",
    method_order=None,
    save_path=None,
    figsize=(7.2, 4.8),
    cmap="viridis",
    annotate=True,
    highlight_method="AXIS",
):
    df = pd.DataFrame(data).T
    df.columns = [f"L{i}" for i in range(df.shape[1])]
    df.index.name = "Method"

    if method_order is not None:
        method_order = [m for m in method_order if m in df.index]
        df = df.reindex(method_order)

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)
    fig, ax = plt.subplots(figsize=figsize)

    hm = sns.heatmap(
        df,
        ax=ax,
        cmap=cmap,
        annot=annotate,
        fmt=".3f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": metric_label},
    )
    cbar = hm.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel("Noise level")
    ax.set_ylabel("")
    ax.set_title("Noise robustness heatmap", pad=10)
    ax.set_xticklabels([str(i) for i in range(df.shape[1])], rotation=0, fontsize=12)
    ax.tick_params(axis="y", rotation=0, labelsize=12)

    if highlight_method in df.index:
        row_idx = list(df.index).index(highlight_method)
        ax.add_patch(
            Rectangle(
                (0, row_idx),
                width=df.shape[1],
                height=1,
                fill=False,
                edgecolor=GLOBAL_METHOD_PALETTE["AXIS"],
                linewidth=2.5,
                clip_on=False,
            )
        )
        for label in ax.get_yticklabels():
            if label.get_text() == highlight_method:
                label.set_color(GLOBAL_METHOD_PALETTE["AXIS"])
                label.set_fontweight("bold")

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.show()
    return fig, ax


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # All commonly adjusted line-chart parameters are here.
    STYLE = LinePlotStyle(
        figsize=(12.0, 6.0),

        # X tick labels
        x_ticks=(0, 1, 2, 3, 4, 5),
        x_tick_labels=("0", "1", "2", "3", "4", "5"),
        x_tick_fontsize=15,
        x_tick_rotation=0,
        x_tick_pad=6,

        # Y tick labels: set y_ticks explicitly if you want exact positions.
        # Example:
        # y_ticks=(0.35, 0.40, 0.45, 0.50, 0.55),
        y_ticks=None,
        y_tick_labels=None,
        y_tick_format="%.2f",
        y_tick_fontsize=15,
        y_tick_rotation=0,
        y_tick_pad=6,

        # Axis ranges
        xlim=(-0.1, 6.2),
        ylim=None,

        # Labels / title
        xlabel_fontsize=22,
        ylabel_fontsize=22,
        title_fontsize=24,

        # Right-side labels
        right_label_fontsize=16,
        right_label_x_offset=0.15,
        right_label_min_gap=0.008,
    )

    plot_noise_robustness(
        data=DATA,
        noise_levels=NOISE_LEVELS,
        style=STYLE,
        save_path=OUTPUT_PATH,
        show=True,
    )

