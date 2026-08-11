#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate saved XGBoost cross-modality protein predictions using raw Pearson
correlation as the single primary quantitative metric.

Main outputs
------------
1. Raw per-protein Pearson boxplot with individual protein points.
2. Raw Pearson empirical histogram with transparent bars plus KDE fit.
3. Raw measured-versus-predicted scatter panels for top proteins.
4. Spatial measured/predicted case-study maps retained for qualitative review.

A separate z-scored Pearson is intentionally omitted because independent
positive affine standardization does not change Pearson correlation.

Figures are saved as high-resolution PNG and vector PDF.
"""

import os
import re
import json
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.stats import pearsonr, gaussian_kde


# ============================================================
# USER CONFIGURATION
# ============================================================

# ---------- Coordinate display ----------
SWAP_XY = True
INVERT_X = True
INVERT_Y = True

# ---------- Output ----------
FIG_DPI = 450
SAVE_PNG = True
SAVE_PDF =False

# ---------- Plot sampling ----------
MAX_SCATTER_POINTS = 30_000
RANDOM_SEED = 0

# ---------- Spatial point sizes ----------
LAYOUT_POINT_SIZE = 0.35
SPATIAL_POINT_SIZE = 1.4

# ---------- Publication colors ----------
RAW_COLOR = "#3F6B8C"         # muted blue
ZSCORE_COLOR = "#C97A40"      # muted warm orange
TRUTH_COLOR = "#30475E"
PRED_COLOR = "#4E79A7"
REFERENCE_COLOR = "#555555"

TRAIN_COLOR = "#4C78A8"
VAL_COLOR = "#E9A23B"
TEST_COLOR = "#D95F59"

# Use one restrained sequential map for protein abundance.
SPATIAL_CMAP = "Blues"

# ---------- Typography ----------
FONT_FAMILY = "Arial"


# ============================================================
# GENERAL HELPERS
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_name(s: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(s),
    )


def setup_matplotlib():
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": FIG_DPI,
            "font.family": "sans-serif",
            "font.sans-serif": [
                FONT_FAMILY,
                "DejaVu Sans",
            ],
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
            "legend.fontsize": 8.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    fig,
    out_base: str,
    tight: bool = True,
):
    """
    Save one figure as publication PNG and optional PDF.

    out_base should not include an extension.
    """
    ensure_dir(
        os.path.dirname(
            os.path.abspath(out_base)
        )
    )

    kwargs = {
        "bbox_inches": "tight"
        if tight
        else None,
        "facecolor": "white",
    }

    if SAVE_PNG:
        png = out_base + ".png"
        fig.savefig(
            png,
            dpi=FIG_DPI,
            **kwargs,
        )

    if SAVE_PDF:
        pdf = out_base + ".pdf"
        fig.savefig(
            pdf,
            **kwargs,
        )


def clean_axis(
    ax,
    keep_left=True,
    keep_bottom=True,
):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(
        keep_left
    )
    ax.spines["bottom"].set_visible(
        keep_bottom
    )


def to_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def get_coords(
    adata: ad.AnnData,
) -> pd.DataFrame:
    if "spatial" in adata.obsm:
        arr = np.asarray(
            adata.obsm["spatial"]
        )

        if (
            arr.ndim != 2
            or arr.shape[1] < 2
        ):
            raise ValueError(
                "obsm['spatial'] has "
                f"invalid shape: {arr.shape}"
            )

        coords = arr[:, :2].astype(
            float
        )

        if SWAP_XY:
            coords = coords[:, [1, 0]]

        return pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
            },
            index=adata.obs_names,
        )

    if (
        "x" in adata.obs.columns
        and "y" in adata.obs.columns
    ):
        coords = np.column_stack(
            [
                pd.to_numeric(
                    adata.obs["x"],
                    errors="coerce",
                ),
                pd.to_numeric(
                    adata.obs["y"],
                    errors="coerce",
                ),
            ]
        )

        if SWAP_XY:
            coords = coords[:, [1, 0]]

        return pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
            },
            index=adata.obs_names,
        )

    raise ValueError(
        "Could not find coordinates in "
        "obsm['spatial'] or obs['x']/obs['y']."
    )


def apply_spatial_orientation(ax):
    if INVERT_X:
        ax.invert_xaxis()
    if INVERT_Y:
        ax.invert_yaxis()


def robust_vlim(
    *arrays,
    q=(1, 99),
):
    values = []

    for array in arrays:
        array = np.asarray(
            array
        ).ravel()

        array = array[
            np.isfinite(array)
        ]

        if array.size:
            values.append(array)

    if not values:
        return 0.0, 1.0

    values = np.concatenate(
        values
    )

    vmin, vmax = np.percentile(
        values,
        q,
    )

    if (
        not np.isfinite(vmin)
        or not np.isfinite(vmax)
        or vmax <= vmin
    ):
        vmin = float(
            np.min(values)
        )
        vmax = float(
            np.max(values)
        )

    if vmax <= vmin:
        vmax = vmin + 1e-6

    return float(vmin), float(vmax)


def safe_corr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Robust Pearson correlation helper."""
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


def zscore_self(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    mu = np.nanmean(
        values
    )
    sigma = np.nanstd(
        values
    )

    if (
        not np.isfinite(sigma)
        or sigma <= 1e-12
    ):
        sigma = 1.0

    return (
        values - mu
    ) / sigma


def deterministic_sample_indices(
    n: int,
    max_points: int = MAX_SCATTER_POINTS,
    seed: int = RANDOM_SEED,
):
    if (
        max_points is None
        or max_points <= 0
        or n <= max_points
    ):
        return np.arange(n)

    rng = np.random.default_rng(
        seed
    )

    return np.sort(
        rng.choice(
            n,
            size=max_points,
            replace=False,
        )
    )


# ============================================================
# METRICS
# ============================================================

def compute_per_protein_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protein_names: Sequence[str],
) -> pd.DataFrame:
    """
    Compute raw Pearson correlation for each protein.

    A separate z-scored Pearson is intentionally omitted because Pearson
    correlation is invariant to independent positive affine standardization.
    """
    rows = []

    for j, protein in enumerate(protein_names):
        yt = np.asarray(y_true[:, j], dtype=float)
        yp = np.asarray(y_pred[:, j], dtype=float)

        rows.append(
            {
                "protein": str(protein),
                "pearson_r": safe_corr(yt, yp),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("pearson_r", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# PLOTTING: LAYOUT
# ============================================================

def plot_layout(
    coords: pd.DataFrame,
    split: Optional[pd.Series],
    out_base: str,
):
    fig, ax = plt.subplots(
        figsize=(4.4, 4.1)
    )

    if (
        split is not None
        and split.nunique() > 1
    ):
        split = pd.Series(
            split,
            index=coords.index,
        ).astype(str)

        color_map = {
            "train": TRAIN_COLOR,
            "val": VAL_COLOR,
            "test": TEST_COLOR,
        }

        for label in pd.unique(
            split
        ):
            mask = (
                split == label
            )

            ax.scatter(
                coords.loc[
                    mask,
                    "x",
                ],
                coords.loc[
                    mask,
                    "y",
                ],
                s=LAYOUT_POINT_SIZE,
                alpha=0.72,
                c=color_map.get(
                    str(label),
                    "0.5",
                ),
                linewidths=0,
                rasterized=True,
                label=str(label),
            )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                1.03,
            ),
            ncol=min(
                3,
                split.nunique(),
            ),
            frameon=False,
            markerscale=4,
            handletextpad=0.35,
            columnspacing=1.0,
        )

    else:
        ax.scatter(
            coords["x"],
            coords["y"],
            s=LAYOUT_POINT_SIZE,
            alpha=0.72,
            c="0.25",
            linewidths=0,
            rasterized=True,
        )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    apply_spatial_orientation(
        ax
    )

    ax.axis("off")

    save_figure(
        fig,
        out_base,
    )
    plt.close(fig)


# ============================================================
# PLOTTING: PEARSON BOXPLOT
# ============================================================

def plot_pearson_boxplot(
    metrics_df: pd.DataFrame,
    out_base: str,
):
    """
    Publication-style raw Pearson summary.

    One dot = one predicted protein.
    The box shows Q1-IQR-Q3 and the center line shows the median.
    """
    values = metrics_df["pearson_r"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        print("[WARN] No finite Pearson values for boxplot.")
        return

    median = float(np.median(values))
    q1, q3 = np.percentile(values, [25, 75])

    fig, ax = plt.subplots(
        figsize=(3.55, 4.25),
        facecolor="white",
    )

    bp = ax.boxplot(
        [values],
        positions=[1.0],
        widths=0.34,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={
            "linewidth": 1.8,
            "color": "0.10",
        },
        whiskerprops={
            "linewidth": 1.0,
            "color": "0.35",
        },
        capprops={
            "linewidth": 1.0,
            "color": "0.35",
        },
        boxprops={
            "linewidth": 1.0,
            "edgecolor": RAW_COLOR,
        },
    )

    bp["boxes"][0].set_facecolor(RAW_COLOR)
    bp["boxes"][0].set_alpha(0.20)

    rng = np.random.default_rng(RANDOM_SEED)
    jitter = rng.normal(
        loc=1.0,
        scale=0.045,
        size=values.size,
    )

    ax.scatter(
        jitter,
        values,
        s=24,
        c=RAW_COLOR,
        alpha=0.72,
        linewidths=0,
        zorder=3,
    )

    # Compact numerical summary.
    ax.text(
        1.21,
        median,
        f"median = {median:.2f}",
        ha="left",
        va="center",
        fontsize=8.3,
        color="0.20",
    )

    ax.text(
        0.04,
        0.98,
        f"n = {values.size} proteins\nIQR = {q1:.2f}–{q3:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="0.35",
    )

    ax.axhline(
        0,
        color="0.84",
        linewidth=0.8,
        zorder=0,
    )

    ymin = min(-0.05, float(np.min(values)) - 0.05)
    ymax = max(1.0, float(np.max(values)) + 0.05)

    ax.set_ylim(ymin, min(1.05, ymax))
    ax.set_xlim(0.68, 1.45)
    ax.set_xticks([])
    ax.set_ylabel("Pearson correlation ($r$)")
    ax.set_title(
        "Cross-modality prediction",
        loc="left",
        fontweight="bold",
        pad=8,
    )

    clean_axis(
        ax,
        keep_left=True,
        keep_bottom=False,
    )

    fig.tight_layout()
    save_figure(fig, out_base)
    plt.close(fig)


# ============================================================
# PLOTTING: PEARSON HISTOGRAM + KDE
# ============================================================

def _pearson_histogram_bins(
    values: np.ndarray,
) -> int:
    """
    Freedman-Diaconis starting point, constrained to a clean 6-12 bins.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return 6

    try:
        edges = np.histogram_bin_edges(
            values,
            bins="fd",
        )
        n_bins = len(edges) - 1
    except Exception:
        n_bins = 8

    return int(np.clip(n_bins, 6, 12))


def plot_pearson_histogram_kde(
    metrics_df: pd.DataFrame,
    out_base: str,
):
    """
    Empirical distribution of raw Pearson correlations.

    Transparent bars show the actual per-protein distribution.
    The KDE curve is overlaid only as a smooth visual summary.
    """
    values = metrics_df["pearson_r"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        print("[WARN] No finite Pearson values for histogram/KDE.")
        return

    median = float(np.median(values))
    n_bins = _pearson_histogram_bins(values)

    fig, ax = plt.subplots(
        figsize=(4.65, 3.65),
        facecolor="white",
    )

    counts, bins, _ = ax.hist(
        values,
        bins=n_bins,
        density=True,
        color=RAW_COLOR,
        alpha=0.22,
        edgecolor=RAW_COLOR,
        linewidth=0.85,
        label="Observed proteins",
        zorder=1,
    )

    if values.size >= 5 and np.std(values) > 0:
        xmin = max(-1.0, float(np.min(values)) - 0.08)
        xmax = min(1.0, float(np.max(values)) + 0.08)
        xs = np.linspace(xmin, xmax, 400)

        kde = gaussian_kde(values)
        ys = kde(xs)

        ax.plot(
            xs,
            ys,
            color=RAW_COLOR,
            linewidth=2.0,
            label="KDE",
            zorder=3,
        )

    # Small rug marks retain visibility of each individual protein.
    top = max(
        float(np.max(counts)) if counts.size else 1.0,
        1e-6,
    )
    rug_y = -0.018 * top

    ax.scatter(
        values,
        np.full(values.size, rug_y),
        marker="|",
        s=34,
        c=RAW_COLOR,
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
        f"  median = {median:.2f}",
        ha="left",
        va="top",
        fontsize=8.0,
        color="0.25",
    )

    ax.set_xlim(
        max(-1.0, float(np.min(values)) - 0.10),
        min(1.0, float(np.max(values)) + 0.10),
    )
    ax.set_xlabel("Pearson correlation ($r$)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Prediction agreement across proteins",
        loc="left",
        fontweight="bold",
        pad=8,
    )

    clean_axis(ax)

    ax.legend(
        loc="upper left",
        frameon=False,
        handlelength=2.1,
        borderaxespad=0.2,
    )

    fig.tight_layout()
    save_figure(fig, out_base)
    plt.close(fig)


# ============================================================
# PLOTTING: GLOBAL SCATTER PANELS
# ============================================================

def plot_global_scatter_panels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protein_names: Sequence[str],
    metrics_df: pd.DataFrame,
    out_base: str,
    n_proteins: int = 9,
):
    """
    Raw measured-versus-predicted scatter panels for top Pearson proteins.
    """
    selected = metrics_df.head(
        min(n_proteins, len(metrics_df))
    )["protein"].tolist()

    protein_to_index = {
        str(protein): idx
        for idx, protein in enumerate(protein_names)
    }

    n = len(selected)
    ncol = 3
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.15 * ncol, 3.05 * nrow),
        squeeze=False,
    )
    axes = axes.ravel()

    for panel_idx, (ax, protein) in enumerate(
        zip(axes, selected)
    ):
        j = protein_to_index[protein]

        yt = np.asarray(y_true[:, j], dtype=float)
        yp = np.asarray(y_pred[:, j], dtype=float)

        idx = deterministic_sample_indices(
            len(yt),
            max_points=MAX_SCATTER_POINTS,
            seed=RANDOM_SEED + panel_idx,
        )

        lo, hi = robust_vlim(
            yt,
            yp,
            q=(0.5, 99.5),
        )

        ax.scatter(
            yt[idx],
            yp[idx],
            s=1.3,
            alpha=0.18,
            c=RAW_COLOR,
            linewidths=0,
            rasterized=True,
        )

        ax.plot(
            [lo, hi],
            [lo, hi],
            linestyle=(0, (3, 2)),
            linewidth=0.9,
            color=REFERENCE_COLOR,
            zorder=0,
        )

        row = metrics_df.loc[
            metrics_df["protein"] == protein
        ].iloc[0]

        ax.text(
            0.04,
            0.96,
            rf"$r$ = {row['pearson_r']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.8,
            color="0.25",
        )

        ax.set_title(
            protein,
            fontweight="bold",
            loc="left",
        )
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        clean_axis(ax)

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    save_figure(fig, out_base)
    plt.close(fig)


# ============================================================
# PLOTTING: CASE-STUDY SPATIAL COMPARISON
# ============================================================

def _scatter_spatial(
    ax,
    coords,
    values,
    cmap,
    vmin,
    vmax,
    point_size,
):
    sc = ax.scatter(
        coords["x"],
        coords["y"],
        c=values,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    apply_spatial_orientation(
        ax
    )

    ax.axis(
        "off"
    )

    return sc


def plot_case_spatial_comparison(
    coords: pd.DataFrame,
    truth: np.ndarray,
    pred: np.ndarray,
    protein: str,
    out_base: str,
):
    """
    One compact 2x2 figure:
        raw truth | raw prediction
        z truth   | z prediction

    Raw panels share one color scale; z-score panels share a second scale.
    """
    truth = np.asarray(
        truth,
        dtype=float,
    )
    pred = np.asarray(
        pred,
        dtype=float,
    )

    truth_z = zscore_self(
        truth
    )
    pred_z = zscore_self(
        pred
    )

    raw_vmin, raw_vmax = robust_vlim(
        truth,
        pred,
        q=(1, 99),
    )

    z_vmin, z_vmax = robust_vlim(
        truth_z,
        pred_z,
        q=(1, 99),
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(6.1, 5.7),
    )

    sc_raw = _scatter_spatial(
        axes[0, 0],
        coords,
        truth,
        cmap=SPATIAL_CMAP,
        vmin=raw_vmin,
        vmax=raw_vmax,
        point_size=SPATIAL_POINT_SIZE,
    )

    _scatter_spatial(
        axes[0, 1],
        coords,
        pred,
        cmap=SPATIAL_CMAP,
        vmin=raw_vmin,
        vmax=raw_vmax,
        point_size=SPATIAL_POINT_SIZE,
    )

    sc_z = _scatter_spatial(
        axes[1, 0],
        coords,
        truth_z,
        cmap=SPATIAL_CMAP,
        vmin=z_vmin,
        vmax=z_vmax,
        point_size=SPATIAL_POINT_SIZE,
    )

    _scatter_spatial(
        axes[1, 1],
        coords,
        pred_z,
        cmap=SPATIAL_CMAP,
        vmin=z_vmin,
        vmax=z_vmax,
        point_size=SPATIAL_POINT_SIZE,
    )

    axes[0, 0].set_title(
        "Measured",
        fontweight="bold",
    )
    axes[0, 1].set_title(
        "Predicted",
        fontweight="bold",
    )

    axes[1, 0].set_title(
        "Measured, z-score",
        fontweight="bold",
    )
    axes[1, 1].set_title(
        "Predicted, z-score",
        fontweight="bold",
    )

    fig.text(
        0.02,
        0.73,
        "Raw",
        rotation=90,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=9,
    )

    fig.text(
        0.02,
        0.29,
        "Standardized",
        rotation=90,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=9,
    )

    # Two compact colorbars, one for each row.
    cbar_raw = fig.colorbar(
        sc_raw,
        ax=axes[0, :],
        fraction=0.024,
        pad=0.02,
    )
    cbar_raw.ax.tick_params(
        labelsize=7
    )

    cbar_z = fig.colorbar(
        sc_z,
        ax=axes[1, :],
        fraction=0.024,
        pad=0.02,
    )
    cbar_z.ax.tick_params(
        labelsize=7
    )

    fig.suptitle(
        protein,
        x=0.07,
        y=0.99,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.91,
        bottom=0.04,
        top=0.92,
        wspace=0.05,
        hspace=0.13,
    )

    save_figure(
        fig,
        out_base,
    )
    plt.close(fig)


# ============================================================
# PLOTTING: CASE-STUDY SCATTER
# ============================================================

def plot_case_scatter_overlay(
    truth: np.ndarray,
    pred: np.ndarray,
    protein: str,
    out_base: str,
    metrics_row: pd.Series,
):
    """
    Raw measured-versus-predicted case-study scatter with Pearson r only.
    """
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)

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
        figsize=(4.2, 3.9)
    )

    ax.scatter(
        truth[idx],
        pred[idx],
        s=1.6,
        alpha=0.20,
        c=RAW_COLOR,
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

    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    ax.set_title(
        protein,
        loc="left",
        fontweight="bold",
    )

    ax.text(
        0.04,
        0.96,
        rf"Pearson $r$ = {metrics_row['pearson_r']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="0.25",
    )

    clean_axis(ax)
    fig.tight_layout()

    save_figure(fig, out_base)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    setup_matplotlib()

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and plot saved "
            "XGBoost cross-modality predictions."
        )
    )

    parser.add_argument(
        "--pred_h5ad",
        type=str,
        default=(
            "/media/huifang/data/registration/phenocycler/"
            "cross_modality_prediction/xgb_out/"
            "xgb_prediction_presta_x_virtual_protein_test0.2/"
            "aligned_pred_test_only.h5ad"
        ),
    )

    parser.add_argument(
        "--metrics_csv",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=(
            "/media/huifang/data/registration/paper_figure/Figure7_modality/cross_modality_prediction"
        ),
    )

    parser.add_argument(
        "--case_study_proteins",
        nargs="*",
        default=None,
        help=(
            "Protein names for detailed plots. "
            "If omitted, top proteins by Pearson correlation are used."
        ),
    )

    parser.add_argument(
        "--n_case_study",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--n_global_scatter",
        type=int,
        default=9,
    )

    args = parser.parse_args()

    ensure_dir(
        args.outdir
    )

    # Dedicated folders keep final figures easier to navigate.
    overview_dir = os.path.join(
        args.outdir,
        "overview",
    )
    case_dir = os.path.join(
        args.outdir,
        "case_studies",
    )

    ensure_dir(
        overview_dir
    )
    ensure_dir(
        case_dir
    )

    adata = ad.read_h5ad(
        args.pred_h5ad
    )

    if "truth" not in adata.layers:
        raise ValueError(
            "Prediction h5ad must contain "
            "adata.layers['truth']."
        )

    coords = get_coords(
        adata
    )

    y_pred = to_dense(
        adata.X
    ).astype(
        np.float32
    )

    y_true = to_dense(
        adata.layers["truth"]
    ).astype(
        np.float32
    )

    protein_names = [
        str(x)
        for x in adata.var_names.tolist()
    ]

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
        and pd.Series(
            split
        ).nunique() > 1
        and "test" in set(
            pd.Series(
                split
            ).astype(str)
        )
    ):
        eval_mask = (
            pd.Series(
                split
            ).astype(str).to_numpy()
            == "test"
        )

        eval_coords = coords.loc[
            eval_mask
        ].reset_index(
            drop=True
        )

        eval_true = y_true[
            eval_mask
        ]

        eval_pred = y_pred[
            eval_mask
        ]

    else:
        eval_coords = coords.reset_index(
            drop=True
        )
        eval_true = y_true
        eval_pred = y_pred

    # --------------------------------------------------------
    # Metrics
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

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Optional previous metric reference
    # --------------------------------------------------------
    if (
        args.metrics_csv is not None
        and os.path.exists(
            args.metrics_csv
        )
    ):
        old_df = pd.read_csv(
            args.metrics_csv
        )

        merge_cols = [
            col
            for col in [
                "protein",
                "pearson_r",
            ]
            if col in old_df.columns
        ]

        if "protein" in merge_cols:
            rename = {
                col: f"old_{col}"
                for col in merge_cols
                if col != "protein"
            }

            merged = metrics_df.merge(
                old_df[
                    merge_cols
                ].rename(
                    columns=rename
                ),
                on="protein",
                how="left",
            )

            merged.to_csv(
                os.path.join(
                    args.outdir,
                    "metrics_with_previous_reference.csv",
                ),
                index=False,
            )

    # --------------------------------------------------------
    # Overview figures
    # --------------------------------------------------------
    plot_layout(
        coords=coords,
        split=(
            None
            if split is None
            else pd.Series(
                split,
                index=coords.index,
            ).astype(str)
        ),
        out_base=os.path.join(
            overview_dir,
            "spatial_split_layout",
        ),
    )

    plot_pearson_boxplot(
        metrics_df=metrics_df,
        out_base=os.path.join(
            overview_dir,
            "pearson_boxplot",
        ),
    )

    plot_pearson_histogram_kde(
        metrics_df=metrics_df,
        out_base=os.path.join(
            overview_dir,
            "pearson_histogram_kde",
        ),
    )

    plot_global_scatter_panels(
        y_true=eval_true,
        y_pred=eval_pred,
        protein_names=protein_names,
        metrics_df=metrics_df,
        out_base=os.path.join(
            overview_dir,
            "top_protein_scatter_panels",
        ),
        n_proteins=args.n_global_scatter,
    )

    # --------------------------------------------------------
    # Case-study selection
    # --------------------------------------------------------
    if (
        args.case_study_proteins is not None
        and len(
            args.case_study_proteins
        ) > 0
    ):
        case_names = [
            protein
            for protein
            in args.case_study_proteins
            if protein in protein_names
        ]

        missing = [
            protein
            for protein
            in args.case_study_proteins
            if protein not in protein_names
        ]

        if missing:
            print(
                "[WARN] Case-study proteins "
                "not found and skipped: "
                + ", ".join(
                    missing
                )
            )

    else:
        case_names = metrics_df.head(
            args.n_case_study
        )["protein"].tolist()

    protein_to_index = {
        protein: idx
        for idx, protein
        in enumerate(
            protein_names
        )
    }

    # --------------------------------------------------------
    # Case studies
    # --------------------------------------------------------
    for protein in case_names:
        j = protein_to_index[
            protein
        ]

        truth = np.asarray(
            eval_true[:, j],
            dtype=float,
        )

        pred = np.asarray(
            eval_pred[:, j],
            dtype=float,
        )

        protein_dir = os.path.join(
            case_dir,
            safe_name(
                protein
            ),
        )

        ensure_dir(
            protein_dir
        )

        plot_case_spatial_comparison(
            coords=eval_coords,
            truth=truth,
            pred=pred,
            protein=protein,
            out_base=os.path.join(
                protein_dir,
                "spatial_comparison",
            ),
        )

        metrics_row = metrics_df.loc[
            metrics_df["protein"]
            == protein
        ].iloc[0]

        plot_case_scatter_overlay(
            truth=truth,
            pred=pred,
            protein=protein,
            out_base=os.path.join(
                protein_dir,
                "scatter_overlay",
            ),
            metrics_row=metrics_row,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    pearson_values = metrics_df[
        "pearson_r"
    ].to_numpy(dtype=float)
    pearson_values = pearson_values[
        np.isfinite(pearson_values)
    ]

    summary = {
        "n_cells_in_pred_h5ad": int(
            adata.n_obs
        ),
        "n_eval_cells": int(
            eval_true.shape[0]
        ),
        "n_proteins": int(
            eval_true.shape[1]
        ),
        "case_study_proteins": case_names,
        "median_pearson_r": float(
            np.nanmedian(pearson_values)
        ),
        "mean_pearson_r": float(
            np.nanmean(pearson_values)
        ),
        "q1_pearson_r": float(
            np.nanpercentile(
                pearson_values,
                25,
            )
        ),
        "q3_pearson_r": float(
            np.nanpercentile(
                pearson_values,
                75,
            )
        ),
        "min_pearson_r": float(
            np.nanmin(pearson_values)
        ),
        "max_pearson_r": float(
            np.nanmax(pearson_values)
        ),
    }

    with open(
        os.path.join(
            args.outdir,
            "summary_metrics.json",
        ),
        "w",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(
        "[DONE] Saved Cell/Nature-style "
        f"evaluation figures to: {args.outdir}"
    )


if __name__ == "__main__":
    main()