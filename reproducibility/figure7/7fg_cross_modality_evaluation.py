#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlined evaluation/visualization of saved XGBoost cross-modality protein predictions.

Outputs written directly to OUTDIR
----------------------------------
- per_protein_metrics_recomputed.csv
- pearson_histogram_kde.png
- <protein>__scatter.png
- <protein>__measured_raw.png
- <protein>__predicted_raw.png
- summary_metrics.json

Case-study selection
--------------------
CASE_STUDY_PROTEINS = None
    -> plot all available protein markers.

CASE_STUDY_PROTEINS = ["IDO1", "CD44", ...]
    -> plot only the requested protein markers.

For each selected protein, exactly three case-study figures are generated:
1. measured-vs-predicted scatter plot;
2. measured raw spatial map with its own colorbar;
3. predicted raw spatial map with its own colorbar.

The measured and predicted spatial maps use the same robust raw-value limits for
direct visual comparison. Their sequential colormap is constructed from only the
BLUE half of seaborn's "vlag" palette, reversed so low values are near white and
high values are blue.
"""

from __future__ import annotations
from axis.settings import data_path

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde, pearsonr


# ============================================================
# USER CONFIGURATION
# ============================================================

PRED_H5AD = (
    data_path("phenocycler/"
    "cross_modality_prediction/xgb_out_search_new/"
    "aligned_pred_test_only.h5ad")
)

OUTDIR = (
    data_path("paper_analysis/"
    "Figure7_modality/fg_cross_modality_prediction")
)

# None -> plot every protein.
CASE_STUDY_PROTEINS: Optional[List[str]] = ['IDO1','PD-L1']
# Example:
# CASE_STUDY_PROTEINS = ["IDO1", "CD44", "Pan-Cytokeratin"]

# Coordinate display.
SWAP_XY = True
INVERT_X = True
INVERT_Y = True

# Scatter sampling.
MAX_SCATTER_POINTS = 30_000
RANDOM_SEED = 0

# Robust raw-value limits shared by measured and predicted maps.
SPATIAL_VLIM_PERCENTILES = (1.0, 99.0)

# Figure output.
FIG_DPI = 450
FONT_FAMILY = "Arial"

# Restrained publication color used for histogram/scatter.
PRIMARY_COLOR = "#3F6B8C"
REFERENCE_COLOR = "#555555"


# ============================================================
# EASY-TO-TUNE FIGURE LAYOUT
# ============================================================

@dataclass(frozen=True)
class PlotLayout:
    # Histogram.
    histogram_figsize: Tuple[float, float] = (4.65, 3.65)
    histogram_title_fontsize: float = 12
    histogram_label_fontsize: float = 10
    histogram_tick_fontsize: float = 10
    histogram_legend_fontsize: float = 10.0

    # Scatter.
    scatter_figsize: Tuple[float, float] = (4.2, 3.9)
    scatter_point_size: float = 1.6
    scatter_alpha: float = 0.20
    scatter_title_fontsize: float = 10.5
    scatter_label_fontsize: float = 9.0
    scatter_tick_fontsize: float = 8.0
    scatter_stats_x: float = 0.04
    scatter_stats_y: float = 0.96

    # Spatial map.
    spatial_figsize: Tuple[float, float] = (4.35, 4.15)
    spatial_point_size: float = 0.40
    spatial_title_fontsize: float = 12
    spatial_colorbar_fraction: float = 0.030
    spatial_colorbar_pad: float = 0.025
    spatial_colorbar_shrink: float = 0.82
    spatial_colorbar_label_fontsize: float = 9.0
    spatial_colorbar_tick_fontsize: float = 8.0

    # Saving.
    dpi: int = FIG_DPI
    pad_inches: float = 0.04


PLOT_LAYOUT = PlotLayout()


# ============================================================
# COLORMAP
# ============================================================

def build_blue_vlag_cmap() -> LinearSegmentedColormap:
    """
    Build a sequential white -> blue map using only the blue half of "vlag".

    seaborn "vlag" runs approximately blue -> white -> red. We take the
    blue-to-white half and reverse it so low abundance is light and high
    abundance is blue.
    """
    vlag = sns.color_palette("vlag", as_cmap=True)
    colors = vlag(np.linspace(0.50, 0.0, 256))
    return LinearSegmentedColormap.from_list("vlag_blue_only", colors)


# SPATIAL_CMAP = build_blue_vlag_cmap()
SPATIAL_CMAP="Blues"

# ============================================================
# GENERAL HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))


def setup_matplotlib(layout: PlotLayout = PLOT_LAYOUT) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": layout.dpi,
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    fig,
    out_path: str,
    layout: PlotLayout = PLOT_LAYOUT,
) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig.savefig(
        out_path,
        dpi=layout.dpi,
        bbox_inches="tight",
        pad_inches=layout.pad_inches,
        facecolor="white",
    )
    plt.close(fig)
    print(f"[SAVE] {out_path}")


def clean_axis(
    ax,
    keep_left: bool = True,
    keep_bottom: bool = True,
) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.spines["bottom"].set_visible(keep_bottom)


def to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def get_coords(adata: ad.AnnData) -> pd.DataFrame:
    if "spatial" in adata.obsm:
        arr = np.asarray(adata.obsm["spatial"])
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                f"obsm['spatial'] has invalid shape: {arr.shape}"
            )
        coords = arr[:, :2].astype(float)

    elif "x" in adata.obs.columns and "y" in adata.obs.columns:
        coords = np.column_stack(
            [
                pd.to_numeric(adata.obs["x"], errors="coerce"),
                pd.to_numeric(adata.obs["y"], errors="coerce"),
            ]
        ).astype(float)

    else:
        raise ValueError(
            "Could not find coordinates in obsm['spatial'] "
            "or obs['x']/obs['y']."
        )

    if SWAP_XY:
        coords = coords[:, [1, 0]]

    return pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1]},
        index=adata.obs_names,
    )


def apply_spatial_orientation(ax) -> None:
    if INVERT_X:
        ax.invert_xaxis()
    if INVERT_Y:
        ax.invert_yaxis()


def robust_vlim(
    *arrays,
    q: Tuple[float, float] = SPATIAL_VLIM_PERCENTILES,
) -> Tuple[float, float]:
    values = []

    for array in arrays:
        array = np.asarray(array, dtype=float).ravel()
        array = array[np.isfinite(array)]
        if array.size:
            values.append(array)

    if not values:
        return 0.0, 1.0

    values = np.concatenate(values)
    vmin, vmax = np.percentile(values, q)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(values))
        vmax = float(np.max(values))

    if vmax <= vmin:
        vmax = vmin + 1e-6

    return float(vmin), float(vmax)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if (
        x.size < 3
        or np.std(x) <= 1e-12
        or np.std(y) <= 1e-12
    ):
        return np.nan

    try:
        return float(pearsonr(x, y)[0])
    except Exception:
        return np.nan


def deterministic_sample_indices(
    n: int,
    max_points: int = MAX_SCATTER_POINTS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    if max_points is None or max_points <= 0 or n <= max_points:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(n, size=max_points, replace=False)
    )


# ============================================================
# METRICS
# ============================================================

def compute_per_protein_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protein_names: Sequence[str],
) -> pd.DataFrame:
    rows = []

    for j, protein in enumerate(protein_names):
        rows.append(
            {
                "protein": str(protein),
                "pearson_r": safe_corr(
                    y_true[:, j],
                    y_pred[:, j],
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("pearson_r", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# PERFORMANCE OVERVIEW: ONLY PEARSON HISTOGRAM + KDE
# ============================================================

def _pearson_histogram_bins(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return 6

    try:
        edges = np.histogram_bin_edges(values, bins="fd")
        n_bins = len(edges) - 1
    except Exception:
        n_bins = 8

    return int(np.clip(n_bins, 6, 12))


def plot_pearson_histogram_kde(
    metrics_df: pd.DataFrame,
    out_path: str,
    layout: PlotLayout = PLOT_LAYOUT,
) -> None:
    values = metrics_df["pearson_r"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        print("[WARN] No finite Pearson values for histogram/KDE.")
        return

    median = float(np.median(values))
    n_bins = _pearson_histogram_bins(values)

    fig, ax = plt.subplots(
        figsize=layout.histogram_figsize,
        facecolor="white",
    )

    counts, _, _ = ax.hist(
        values,
        bins=n_bins,
        density=True,
        color=PRIMARY_COLOR,
        alpha=0.22,
        edgecolor=PRIMARY_COLOR,
        linewidth=0.85,
        label="Empirical\ndistribution",
        zorder=1,
    )

    if values.size >= 5 and np.std(values) > 0:
        xmin = max(-1.0, float(np.min(values)) - 0.08)
        xmax = min(1.0, float(np.max(values)) + 0.08)
        xs = np.linspace(xmin, xmax, 400)
        kde = gaussian_kde(values)

        ax.plot(
            xs,
            kde(xs),
            color=PRIMARY_COLOR,
            linewidth=2.0,
            label="KDE",
            zorder=3,
        )

    top = max(
        float(np.max(counts)) if counts.size else 1.0,
        1e-6,
    )
    ax.scatter(
        values,
        np.full(values.size, -0.018 * top),
        marker="|",
        s=34,
        c=PRIMARY_COLOR,
        alpha=0.65,
        clip_on=False,
        zorder=4,
    )

    ax.axvline(
        median,
        color="0.20",
        linestyle=(0, (3, 2)),
        linewidth=1.0,
        zorder=2,
    )

    ax.text(
        median,
        ax.get_ylim()[1] * 0.97,
        f"  Median $r$ = {median:.2f}",
        ha="left",
        va="top",
        fontsize=layout.histogram_tick_fontsize,
        color="0.25",
    )

    ax.set_xlim(
        max(-1.0, float(np.min(values)) - 0.10),
        min(1.0, float(np.max(values)) + 0.10),
    )
    ax.set_xlabel(
        "Pearson $r$ (measured vs predicted)",
        fontsize=layout.histogram_label_fontsize,
    )
    ax.set_ylabel(
        "Density",
        fontsize=layout.histogram_label_fontsize,
    )
    ax.set_title(
        "Measured-predicted agreement across protein targets",
        loc="left",
        fontsize=layout.histogram_title_fontsize,
        pad=8,
        x=-0.13
    )
    ax.tick_params(
        axis="both",
        labelsize=layout.histogram_tick_fontsize,
    )

    clean_axis(ax)

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=layout.histogram_legend_fontsize,
        handlelength=2.1,
        borderaxespad=0.2,
    )

    fig.tight_layout()
    save_figure(fig, out_path, layout=layout)


# ============================================================
# CASE STUDY 1/3: SCATTER
# ============================================================

def plot_case_scatter(
    truth: np.ndarray,
    pred: np.ndarray,
    protein: str,
    pearson_r: float,
    out_path: str,
    layout: PlotLayout = PLOT_LAYOUT,
) -> None:
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)

    valid = np.isfinite(truth) & np.isfinite(pred)
    truth = truth[valid]
    pred = pred[valid]

    idx = deterministic_sample_indices(
        len(truth),
        max_points=MAX_SCATTER_POINTS,
        seed=RANDOM_SEED,
    )

    lo, hi = robust_vlim(
        truth,
        pred,
        q=(0.5, 99.5),
    )

    fig, ax = plt.subplots(
        figsize=layout.scatter_figsize,
        facecolor="white",
    )

    ax.scatter(
        truth[idx],
        pred[idx],
        s=layout.scatter_point_size,
        alpha=layout.scatter_alpha,
        c=PRIMARY_COLOR,
        linewidths=0,
        rasterized=True,
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        color=REFERENCE_COLOR,
        linewidth=0.9,
        linestyle=(0, (3, 2)),
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(
        "Measured",
        fontsize=layout.scatter_label_fontsize,
    )
    ax.set_ylabel(
        "Predicted",
        fontsize=layout.scatter_label_fontsize,
    )
    ax.set_title(
        protein,
        loc="center",
        fontsize=layout.scatter_title_fontsize,
    )
    ax.tick_params(
        axis="both",
        labelsize=layout.scatter_tick_fontsize,
    )

    ax.text(
        layout.scatter_stats_x,
        layout.scatter_stats_y,
        rf"Pearson $r$ = {pearson_r:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=layout.scatter_tick_fontsize,
        color="0.25",
    )

    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, out_path, layout=layout)


# ============================================================
# CASE STUDY 2/3 + 3/3: RAW SPATIAL MAP
# ============================================================

def plot_raw_spatial_map(
    coords: pd.DataFrame,
    values: np.ndarray,
    protein: str,
    state_label: str,
    vmin: float,
    vmax: float,
    out_path: str,
    layout: PlotLayout = PLOT_LAYOUT,
) -> None:
    values = np.asarray(values, dtype=float).reshape(-1)

    if len(values) != len(coords):
        raise ValueError(
            f"Spatial value length ({len(values)}) does not match "
            f"coordinate length ({len(coords)})."
        )

    finite = (
        np.isfinite(coords["x"].to_numpy())
        & np.isfinite(coords["y"].to_numpy())
        & np.isfinite(values)
    )

    x = coords["x"].to_numpy()[finite]
    y = coords["y"].to_numpy()[finite]
    value = values[finite]

    fig, ax = plt.subplots(
        figsize=layout.spatial_figsize,
        facecolor="white",
    )

    sc = ax.scatter(
        x,
        y,
        c=value,
        s=layout.spatial_point_size,
        cmap=SPATIAL_CMAP,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
    )

    ax.set_aspect("equal", adjustable="box")
    apply_spatial_orientation(ax)
    ax.axis("off")

    # Remove the standard title
    ax.set_title("")

    title_y = 0.95

    # Protein: bold
    ax.text(
        0.5,
        title_y,
        protein,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=layout.spatial_title_fontsize,
        fontweight="bold",
    )

    # State label: normal
    ax.text(
        0.5,
        title_y,
        f" | {state_label}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=layout.spatial_title_fontsize,
        fontweight="normal",
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        fraction=layout.spatial_colorbar_fraction,
        pad=layout.spatial_colorbar_pad,
        shrink=layout.spatial_colorbar_shrink,
    )
    cbar.set_label(
        "log1p protein signal",
        fontsize=layout.spatial_colorbar_label_fontsize,
    )
    cbar.ax.tick_params(
        labelsize=layout.spatial_colorbar_tick_fontsize,
    )

    fig.tight_layout()
    save_figure(fig, out_path, layout=layout)


# ============================================================
# DATA PREPARATION / SELECTION
# ============================================================

def load_evaluation_data(
    pred_h5ad: str,
):
    print(f"[LOAD] {pred_h5ad}")
    adata = ad.read_h5ad(pred_h5ad)

    if "truth" not in adata.layers:
        raise ValueError(
            "Prediction h5ad must contain adata.layers['truth']."
        )

    coords = get_coords(adata)
    y_pred = to_dense(adata.X).astype(np.float32)
    y_true = to_dense(adata.layers["truth"]).astype(np.float32)
    protein_names = [str(x) for x in adata.var_names.tolist()]

    if y_pred.shape != y_true.shape:
        raise ValueError(
            "Prediction and truth shapes differ: "
            f"{y_pred.shape} vs {y_true.shape}"
        )

    split = (
        adata.obs["split"]
        if "split" in adata.obs.columns
        else None
    )

    if (
        split is not None
        and pd.Series(split).nunique() > 1
        and "test" in set(pd.Series(split).astype(str))
    ):
        eval_mask = (
            pd.Series(split).astype(str).to_numpy() == "test"
        )
        eval_coords = coords.loc[eval_mask].reset_index(drop=True)
        eval_true = y_true[eval_mask]
        eval_pred = y_pred[eval_mask]
    else:
        eval_coords = coords.reset_index(drop=True)
        eval_true = y_true
        eval_pred = y_pred

    return (
        adata,
        eval_coords,
        eval_true,
        eval_pred,
        protein_names,
    )


def resolve_case_study_proteins(
    requested: Optional[Sequence[str]],
    protein_names: Sequence[str],
) -> List[str]:
    """
    None or empty -> all proteins, preserving h5ad feature order.
    """
    available = list(protein_names)

    if requested is None or len(requested) == 0:
        return available

    available_lookup = {name: name for name in available}
    selected = [
        name for name in requested
        if name in available_lookup
    ]
    missing = [
        name for name in requested
        if name not in available_lookup
    ]

    if missing:
        print(
            "[WARN] Requested case-study proteins not found and skipped: "
            + ", ".join(missing)
        )

    if not selected:
        raise ValueError(
            "None of the requested case-study proteins were found."
        )

    return selected


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved XGBoost cross-modality protein predictions and "
            "generate a Pearson overview plus per-protein case studies."
        )
    )

    parser.add_argument(
        "--pred_h5ad",
        type=str,
        default=PRED_H5AD,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=OUTDIR,
    )
    parser.add_argument(
        "--case_study_proteins",
        nargs="*",
        default=CASE_STUDY_PROTEINS,
        help=(
            "Protein markers to plot. If omitted or empty, all proteins "
            "in the prediction h5ad are plotted."
        ),
    )

    # A few high-value layout controls are also exposed at CLI level.
    parser.add_argument(
        "--spatial_point_size",
        type=float,
        default=PLOT_LAYOUT.spatial_point_size,
    )
    parser.add_argument(
        "--scatter_point_size",
        type=float,
        default=PLOT_LAYOUT.scatter_point_size,
    )
    parser.add_argument(
        "--colorbar_pad",
        type=float,
        default=PLOT_LAYOUT.spatial_colorbar_pad,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    layout = PlotLayout(
        **{
            **asdict(PLOT_LAYOUT),
            "spatial_point_size": float(args.spatial_point_size),
            "scatter_point_size": float(args.scatter_point_size),
            "spatial_colorbar_pad": float(args.colorbar_pad),
        }
    )

    setup_matplotlib(layout)
    ensure_dir(args.outdir)

    (
        adata,
        eval_coords,
        eval_true,
        eval_pred,
        protein_names,
    ) = load_evaluation_data(args.pred_h5ad)

    # --------------------------------------------------------
    # Per-protein Pearson metrics.
    # --------------------------------------------------------
    metrics_df = compute_per_protein_metrics(
        eval_true,
        eval_pred,
        protein_names,
    )
    metrics_path = os.path.join(
        args.outdir,
        "per_protein_metrics_recomputed.csv",
    )
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[SAVE] {metrics_path}")

    # --------------------------------------------------------
    # Performance overview: ONLY histogram + KDE.
    # Saved directly under output directory.
    # --------------------------------------------------------
    plot_pearson_histogram_kde(
        metrics_df=metrics_df,
        out_path=os.path.join(
            args.outdir,
            "pearson_histogram_kde.png",
        ),
        layout=layout,
    )

    # --------------------------------------------------------
    # Case-study selection.
    # None -> all proteins.
    # --------------------------------------------------------
    case_names = resolve_case_study_proteins(
        args.case_study_proteins,
        protein_names,
    )

    protein_to_index = {
        protein: idx
        for idx, protein in enumerate(protein_names)
    }
    metric_lookup = (
        metrics_df.set_index("protein")["pearson_r"].to_dict()
    )

    # --------------------------------------------------------
    # Exactly 3 direct-output figures per selected protein.
    # --------------------------------------------------------
    for protein in case_names:
        j = protein_to_index[protein]

        truth = np.asarray(eval_true[:, j], dtype=float)
        pred = np.asarray(eval_pred[:, j], dtype=float)

        # Shared raw scale makes measured/predicted maps directly comparable.
        raw_vmin, raw_vmax = robust_vlim(
            truth,
            pred,
            q=SPATIAL_VLIM_PERCENTILES,
        )

        prefix = os.path.join(
            args.outdir,
            safe_name(protein),
        )

        plot_case_scatter(
            truth=truth,
            pred=pred,
            protein=protein,
            pearson_r=float(metric_lookup.get(protein, np.nan)),
            out_path=prefix + "__scatter.png",
            layout=layout,
        )

        plot_raw_spatial_map(
            coords=eval_coords,
            values=truth,
            protein=protein,
            state_label="Measured",
            vmin=raw_vmin,
            vmax=raw_vmax,
            out_path=prefix + "__measured_raw.png",
            layout=layout,
        )

        plot_raw_spatial_map(
            coords=eval_coords,
            values=pred,
            protein=protein,
            state_label="Predicted",
            vmin=raw_vmin,
            vmax=raw_vmax,
            out_path=prefix + "__predicted_raw.png",
            layout=layout,
        )

    # --------------------------------------------------------
    # Compact numerical summary.
    # --------------------------------------------------------
    pearson_values = metrics_df["pearson_r"].to_numpy(dtype=float)
    pearson_values = pearson_values[np.isfinite(pearson_values)]

    summary = {
        "n_cells_in_pred_h5ad": int(adata.n_obs),
        "n_eval_cells": int(eval_true.shape[0]),
        "n_proteins": int(eval_true.shape[1]),
        "case_study_proteins": case_names,
        "median_pearson_r": float(np.nanmedian(pearson_values)),
        "mean_pearson_r": float(np.nanmean(pearson_values)),
        "q1_pearson_r": float(np.nanpercentile(pearson_values, 25)),
        "q3_pearson_r": float(np.nanpercentile(pearson_values, 75)),
        "min_pearson_r": float(np.nanmin(pearson_values)),
        "max_pearson_r": float(np.nanmax(pearson_values)),
    }

    summary_path = os.path.join(
        args.outdir,
        "summary_metrics.json",
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[SAVE] {summary_path}")

    print(
        f"[DONE] Overview + {len(case_names)} protein case studies "
        f"written directly to: {args.outdir}"
    )


if __name__ == "__main__":
    main()
