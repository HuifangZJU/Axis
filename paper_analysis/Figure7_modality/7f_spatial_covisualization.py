#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cell/Nature-style spatial co-expression maps for gene–protein pairs.

This is a streamlined version of the original batch overlay script.

Main changes
------------
1. Input-data location and output location are configured separately.
2. Only user-selected protein markers are analyzed and plotted.
3. Only the 3-state gene/protein co-expression visualization is retained:
      - protein-dominant
      - both-strong
      - gene-dominant
4. Optional dashed spatial contours can highlight one or more interested states:
      - "gene_dominant"
      - "protein_dominant"
      - "both_strong"
      - "all"
5. Contours are derived from a smoothed local fraction of cells belonging to
   the requested state rather than from a convex hull, which gives a more
   faithful spatial boundary for irregular tissue shapes.

The input gene matrix is normalized per cell to TARGET_SUM and log1p transformed
on-the-fly, matching the original script. Predicted protein values are used as
stored in the RDS matrix.
"""
from typing import Dict, Iterable, Optional, Sequence
import os
import re
import argparse
from typing import Dict, Iterable, List, Sequence, Tuple

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
import scipy.sparse as sp
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


# ============================================================
# USER CONFIGURATION
# ============================================================

# ---------- Input data ----------
DATA_ROOT = "/media/huifang/data/registration/phenocycler/haotian"

DE_RDS = "xenium_DE.rds"
GENE_H5AD = "adata_xenium_cell_level.h5ad"
PROTEIN_RDS = "xenium_protein_predict.rds"
XENIUM_LOCATION_CSV = "LUAD_2_A_xenium_axis.csv"

# ---------- Output ----------
OUTPUT_ROOT = (
    "/media/huifang/data/registration/"
    "paper_figure/Figure7_modality/per_marker_ratio"
)

# ---------- Pair selection ----------
# Only these protein markers will be plotted.
# Set to None or [] to include every available protein marker.
TARGET_PROTEINS = [
    # "IDO1",
    # "NKX2.1",
    # "CD68",
    # "CD20"
    "E.Cadherin",
    "CD8",
    "a.SMA",
    "CXCL13"
]

TOPK_GENES_PER_PROTEIN = 10
MAX_PAIRS = 0                    # 0 = no cap
TARGET_SUM = 1e4

# ---------- Plot sampling ----------
PLOT_POINTS = 200_000
POINT_SIZE = 1.0
RANDOM_SEED = 0

# ---------- Coordinate display ----------
INVERT_X = True
INVERT_Y = True

# ---------- Robust modality scaling ----------
PERCENTILE_LOW = 5.0
PERCENTILE_HIGH = 99.0

# ---------- Point alpha ----------
ALPHA_MIN = 0.05
ALPHA_MAX = 0.95

# ---------- Three-state colors ----------
PROTEIN_RGB = (0.10, 0.35, 0.95)   # muted blue
GENE_RGB = (0.95, 0.25, 0.10)      # muted red-orange
BOTH_RGB = (0.55, 0.05, 0.65)     # muted purple
BOTH_GAMMA = 1.6

# ============================================================
# OPTIONAL REGION CONTOURS
# ============================================================

# Master switch.
PLOT_REGION_CONTOUR = False

# Choose one or more:
#   ("gene_dominant",)
#   ("protein_dominant",)
#   ("both_strong",)
#   ("gene_dominant", "both_strong")
#   ("all",)
CONTOUR_REGIONS = ("both_strong",)

# State definitions are based on robustly scaled gene/protein values in [0, 1].
DOMINANCE_FRACTION_THRESHOLD = 0.65
DOMINANCE_MIN_SIGNAL = 0.25

# "Both strong" means BOTH modalities are high, avoiding a ratio-only
# definition that could label two weak signals as co-expression.
BOTH_STRONG_MIN_SIGNAL = 0.55

# Spatial contour field.
CONTOUR_GRID_SIZE = 220
CONTOUR_GAUSSIAN_SIGMA = 2.5

# A contour is drawn where the smoothed local fraction of selected cells
# exceeds this value.
CONTOUR_FRACTION_LEVEL = 0.45

# Ignore regions with very weak local point support.
CONTOUR_MIN_SUPPORT_FRAC = 0.025

CONTOUR_LINEWIDTH = 1.3
CONTOUR_LINESTYLE = (0, (4, 3))

CONTOUR_COLORS = {
    "protein_dominant": (0,0,0),
    "both_strong": (0,0,0),
    "gene_dominant": (0,0,0),
}

CONTOUR_LABELS = {
    "protein_dominant": "Protein-dominant region",
    "both_strong": "Both-strong region",
    "gene_dominant": "Gene-dominant region",
}

# ---------- Publication figure styling ----------
FIGSIZE = (6.6, 6.2)
FIG_DPI = 350
TITLE_FONTSIZE = 12
LEGEND_FONTSIZE = 8.5


# ============================================================
# STYLE
# ============================================================

def set_publication_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": TITLE_FONTSIZE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ============================================================
# GENERAL I/O HELPERS
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text))


def canonical_name(text: str) -> str:
    text = str(text).strip().lower()
    text = (
        text.replace(".", "")
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
    return re.sub(r"[^a-z0-9]+", "", text)


def read_rds_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    result = pyreadr.read_r(path)
    if len(result) == 0:
        raise ValueError(f"No objects found in RDS: {path}")

    obj = next(iter(result.values()))
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"Expected an R data.frame in {path}, got {type(obj)}"
        )

    return obj


def ensure_numeric_df(
    df: pd.DataFrame,
    fillna: float = 0.0,
) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(fillna)


def read_xenium_location_weird_csv(
    path: str,
) -> pd.DataFrame:
    """
    Read the original coordinate file whose first line is invalid
    (for example ',0,1') and whose remaining rows are row_id,x,y.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        header=None,
        skiprows=1,
    )

    if df.shape[1] < 3:
        raise ValueError(
            f"Xenium location CSV has fewer than 3 columns: {df.shape}"
        )

    df = df.iloc[:, :3].copy()
    df.columns = ["row_id", "x", "y"]
    df[["x", "y"]] = ensure_numeric_df(
        df[["x", "y"]]
    )

    return df


def orient_df_to_match_rows(
    df: pd.DataFrame,
    n_rows: int,
    name: str,
) -> pd.DataFrame:
    n_r, n_c = df.shape

    if n_r == n_rows:
        return df

    if n_c == n_rows:
        print(
            f"[TRANSPOSE] {name}: "
            f"{df.shape} -> {df.T.shape}"
        )
        return df.T

    raise ValueError(
        f"{name} shape {df.shape} does not match "
        f"n_rows={n_rows} in either dimension."
    )


def build_canon_map(names: Iterable[str]) -> Dict[str, str]:
    mapping = {}
    for name in names:
        key = canonical_name(name)
        if key and key not in mapping:
            mapping[key] = name
    return mapping


def resolve_target_proteins(
    requested_proteins: Optional[Sequence[str]],
    available_protein_map: Dict[str, str],
) -> List[str]:
    """
    Resolve user-requested protein names against the actual protein-matrix
    columns using the same canonical-name matching used elsewhere.

    Returns actual column names from the protein matrix.

    If requested_proteins is None or empty, all available proteins are returned.
    """
    if not requested_proteins:
        return list(available_protein_map.values())

    resolved = []
    missing = []

    for protein in requested_proteins:
        key = canonical_name(protein)

        if key in available_protein_map:
            actual = available_protein_map[key]
            if actual not in resolved:
                resolved.append(actual)
        else:
            missing.append(str(protein))

    if missing:
        print(
            "[WARN] Requested proteins not found and will be skipped: "
            + ", ".join(missing)
        )

    if not resolved:
        raise ValueError(
            "None of TARGET_PROTEINS were found in the predicted-protein matrix."
        )

    return resolved


# ============================================================
# GENE NORMALIZATION / EXTRACTION
# ============================================================

def compute_row_scale(
    X_csr,
    target_sum: float,
) -> np.ndarray:
    """
    Per-cell library-size scale for normalize_total.
    """
    row_sum = np.asarray(
        X_csr.sum(axis=1)
    ).ravel().astype(np.float32)

    scale = np.zeros_like(
        row_sum,
        dtype=np.float32,
    )

    nonzero = row_sum > 0
    scale[nonzero] = (
        target_sum / row_sum[nonzero]
    ).astype(np.float32)

    return scale


def extract_gene_log1p_on_sample(
    X,
    X_is_sparse: bool,
    X_csc,
    scale: np.ndarray,
    gene_idx: int,
    sample_idx: np.ndarray,
    pos_map: np.ndarray,
) -> np.ndarray:
    """
    Extract one gene on the plotting sample after normalize_total + log1p,
    without materializing the full normalized matrix.
    """
    n_sample = sample_idx.size
    out = np.zeros(
        n_sample,
        dtype=np.float32,
    )

    if not X_is_sparse:
        raw = X[
            sample_idx,
            gene_idx,
        ].astype(
            np.float32,
            copy=False,
        )

        return np.log1p(
            raw * scale[sample_idx]
        ).astype(
            np.float32,
            copy=False,
        )

    col = X_csc.getcol(
        gene_idx
    ).tocoo()

    sample_positions = pos_map[
        col.row
    ]
    keep = sample_positions >= 0

    if np.any(keep):
        out[
            sample_positions[keep]
        ] = (
            col.data[keep].astype(
                np.float32,
                copy=False,
            )
            * scale[col.row[keep]]
        )

    return np.log1p(out).astype(
        np.float32,
        copy=False,
    )


# ============================================================
# ROBUST CO-EXPRESSION STATE
# ============================================================

def robust_minmax(
    values,
    p_low=PERCENTILE_LOW,
    p_high=PERCENTILE_HIGH,
):
    values = np.asarray(
        values,
        dtype=float,
    )
    finite = values[
        np.isfinite(values)
    ]

    if finite.size == 0:
        return 0.0, 1.0

    low = np.percentile(
        finite,
        p_low,
    )
    high = np.percentile(
        finite,
        p_high,
    )

    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or high <= low
    ):
        low = np.nanmin(finite)
        high = np.nanmax(finite)

        if (
            not np.isfinite(low)
            or not np.isfinite(high)
            or high <= low
        ):
            return 0.0, 1.0

    return float(low), float(high)


def scale_01(
    values,
    p_low=PERCENTILE_LOW,
    p_high=PERCENTILE_HIGH,
):
    low, high = robust_minmax(
        values,
        p_low=p_low,
        p_high=p_high,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    scaled = (
        values - low
    ) / (
        high - low + 1e-12
    )

    return np.clip(
        scaled,
        0.0,
        1.0,
    )


def compute_coexpression_state(
    gene_values,
    protein_values,
):
    """
    Return robustly scaled signals and three spatial-state masks.

    State definitions
    -----------------
    gene_dominant:
        gene contributes >= DOMINANCE_FRACTION_THRESHOLD
        AND gene signal >= DOMINANCE_MIN_SIGNAL

    protein_dominant:
        protein contributes >= DOMINANCE_FRACTION_THRESHOLD
        AND protein signal >= DOMINANCE_MIN_SIGNAL

    both_strong:
        gene >= BOTH_STRONG_MIN_SIGNAL
        AND protein >= BOTH_STRONG_MIN_SIGNAL
    """
    gene_01 = scale_01(
        gene_values
    )
    protein_01 = scale_01(
        protein_values
    )

    denominator = (
        gene_01
        + protein_01
        + 1e-8
    )

    gene_fraction = (
        gene_01 / denominator
    )

    total_01 = np.clip(
        (
            gene_01
            + protein_01
        ) / 2.0,
        0.0,
        1.0,
    )

    gene_dominant = (
        (
            gene_fraction
            >= DOMINANCE_FRACTION_THRESHOLD
        )
        & (
            gene_01
            >= DOMINANCE_MIN_SIGNAL
        )
    )

    protein_dominant = (
        (
            gene_fraction
            <= (
                1.0
                - DOMINANCE_FRACTION_THRESHOLD
            )
        )
        & (
            protein_01
            >= DOMINANCE_MIN_SIGNAL
        )
    )

    both_strong = (
        (
            gene_01
            >= BOTH_STRONG_MIN_SIGNAL
        )
        & (
            protein_01
            >= BOTH_STRONG_MIN_SIGNAL
        )
    )

    masks = {
        "gene_dominant": gene_dominant,
        "protein_dominant": protein_dominant,
        "both_strong": both_strong,
    }

    return (
        gene_01,
        protein_01,
        gene_fraction,
        total_01,
        masks,
    )


# ============================================================
# CONTOUR HELPERS
# ============================================================

def normalize_contour_regions(
    regions: Sequence[str],
) -> Tuple[str, ...]:
    valid = {
        "gene_dominant",
        "protein_dominant",
        "both_strong",
    }

    regions = tuple(
        str(region).strip().lower()
        for region in regions
    )

    if "all" in regions:
        return (
            "protein_dominant",
            "both_strong",
            "gene_dominant",
        )

    unknown = set(regions) - valid
    if unknown:
        raise ValueError(
            f"Unknown CONTOUR_REGIONS: {sorted(unknown)}. "
            f"Allowed: {sorted(valid)} or ('all',)."
        )

    return regions


def build_smoothed_region_fraction(
    x,
    y,
    region_mask,
    grid_size=CONTOUR_GRID_SIZE,
    sigma=CONTOUR_GAUSSIAN_SIGMA,
):
    """
    Build a smoothed local fraction field:

        smoothed selected-cell density
        --------------------------------
        smoothed all-cell density

    This is preferable to a convex hull because it follows spatially coherent
    tissue regions and can retain multiple disconnected components.
    """
    x = np.asarray(
        x,
        dtype=float,
    )
    y = np.asarray(
        y,
        dtype=float,
    )
    region_mask = np.asarray(
        region_mask,
        dtype=bool,
    )

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    dx = max(
        float(xmax - xmin),
        1e-6,
    )
    dy = max(
        float(ymax - ymin),
        1e-6,
    )

    pad_x = 0.01 * dx
    pad_y = 0.01 * dy

    x_edges = np.linspace(
        xmin - pad_x,
        xmax + pad_x,
        grid_size + 1,
    )
    y_edges = np.linspace(
        ymin - pad_y,
        ymax + pad_y,
        grid_size + 1,
    )

    all_hist, _, _ = np.histogram2d(
        x,
        y,
        bins=[
            x_edges,
            y_edges,
        ],
    )

    selected_hist, _, _ = np.histogram2d(
        x[region_mask],
        y[region_mask],
        bins=[
            x_edges,
            y_edges,
        ],
    )

    all_smooth = gaussian_filter(
        all_hist,
        sigma=sigma,
    )
    selected_smooth = gaussian_filter(
        selected_hist,
        sigma=sigma,
    )

    fraction = (
        selected_smooth
        / (
            all_smooth
            + 1e-12
        )
    )

    support_threshold = (
        CONTOUR_MIN_SUPPORT_FRAC
        * np.nanmax(all_smooth)
    )

    fraction[
        all_smooth
        < support_threshold
    ] = np.nan

    x_centers = (
        x_edges[:-1]
        + x_edges[1:]
    ) / 2.0

    y_centers = (
        y_edges[:-1]
        + y_edges[1:]
    ) / 2.0

    return (
        x_centers,
        y_centers,
        fraction,
    )


def draw_region_contours(
    ax,
    x,
    y,
    state_masks,
    requested_regions: Sequence[str],
):
    """
    Draw dashed boundaries for selected spatial states.
    """
    contour_handles = []

    for region in normalize_contour_regions(
        requested_regions
    ):
        region_mask = state_masks[
            region
        ]

        if int(region_mask.sum()) < 10:
            print(
                f"[WARN] Too few points for "
                f"{region} contour; skipped."
            )
            continue

        (
            x_centers,
            y_centers,
            fraction,
        ) = build_smoothed_region_fraction(
            x=x,
            y=y,
            region_mask=region_mask,
        )

        if not np.any(
            np.isfinite(fraction)
        ):
            continue

        finite = fraction[
            np.isfinite(fraction)
        ]

        if (
            finite.size == 0
            or np.nanmax(finite)
            < CONTOUR_FRACTION_LEVEL
        ):
            print(
                f"[WARN] {region} never reaches "
                f"contour level "
                f"{CONTOUR_FRACTION_LEVEL:.2f}; skipped."
            )
            continue

        xx, yy = np.meshgrid(
            x_centers,
            y_centers,
        )

        color = CONTOUR_COLORS[
            region
        ]

        ax.contour(
            xx,
            yy,
            fraction.T,
            levels=[
                CONTOUR_FRACTION_LEVEL
            ],
            colors=[
                color
            ],
            linewidths=CONTOUR_LINEWIDTH,
            linestyles=[
                CONTOUR_LINESTYLE
            ],
            zorder=5,
        )

        contour_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=CONTOUR_LINEWIDTH,
                linestyle=CONTOUR_LINESTYLE,
                label=CONTOUR_LABELS[
                    region
                ],
            )
        )

    return contour_handles


# ============================================================
# PUBLICATION CO-EXPRESSION PLOT
# ============================================================

def plot_coexpression_map(
    x,
    y,
    gene_values,
    protein_values,
    gene_name: str,
    protein_name: str,
    output_png: str,
    logfc=None,
    fdr=None,
    max_points=PLOT_POINTS,
    point_size=POINT_SIZE,
    seed=RANDOM_SEED,
):
    """
    Plot a 3-state co-expression map and optionally dashed region contours.

    Point color:
        blue   -> protein dominant
        purple -> both strong/balanced
        red    -> gene dominant

    Point alpha:
        combined gene/protein intensity

    Dashed contours:
        controlled by PLOT_REGION_CONTOUR and CONTOUR_REGIONS.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    gene_values = np.asarray(
        gene_values
    )
    protein_values = np.asarray(
        protein_values
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(gene_values)
        & np.isfinite(protein_values)
    )

    x = x[valid]
    y = y[valid]
    gene_values = gene_values[valid]
    protein_values = protein_values[valid]

    (
        gene_01,
        protein_01,
        gene_fraction,
        total_01,
        state_masks,
    ) = compute_coexpression_state(
        gene_values,
        protein_values,
    )

    # --------------------------------------------------------
    # Color design
    # --------------------------------------------------------
    protein_rgb = np.asarray(
        PROTEIN_RGB,
        dtype=float,
    )
    gene_rgb = np.asarray(
        GENE_RGB,
        dtype=float,
    )
    both_rgb = np.asarray(
        BOTH_RGB,
        dtype=float,
    )

    base_rgb = (
        (
            1.0 - gene_fraction
        )[:, None]
        * protein_rgb[None, :]
        + gene_fraction[:, None]
        * gene_rgb[None, :]
    )

    balance = (
        1.0
        - np.clip(
            np.abs(
                gene_fraction
                - 0.5
            ) / 0.5,
            0.0,
            1.0,
        )
    )

    both_weight = (
        balance ** BOTH_GAMMA
    ) * total_01

    rgb = (
        (
            1.0 - both_weight
        )[:, None]
        * base_rgb
        + both_weight[:, None]
        * both_rgb[None, :]
    )

    alpha = (
        ALPHA_MIN
        + total_01
        * (
            ALPHA_MAX
            - ALPHA_MIN
        )
    )

    rgba = np.concatenate(
        [
            rgb,
            alpha[:, None],
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Scatter downsampling only.
    # Contours use ALL supplied cells, preserving region shape.
    # --------------------------------------------------------
    scatter_idx = np.arange(
        x.size
    )

    if (
        max_points
        and x.size > max_points
    ):
        rng = np.random.default_rng(
            seed
        )
        scatter_idx = rng.choice(
            x.size,
            size=max_points,
            replace=False,
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    set_publication_style()

    fig, ax = plt.subplots(
        1,
        1,
        figsize=FIGSIZE,
        dpi=FIG_DPI,
        facecolor="white",
    )

    ax.scatter(
        x[scatter_idx],
        y[scatter_idx],
        c=rgba[scatter_idx],
        s=point_size,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    contour_handles = []

    if PLOT_REGION_CONTOUR:
        contour_handles = draw_region_contours(
            ax=ax,
            x=x,
            y=y,
            state_masks=state_masks,
            requested_regions=CONTOUR_REGIONS,
        )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    if INVERT_X:
        ax.invert_xaxis()

    if INVERT_Y:
        ax.invert_yaxis()

    ax.axis("off")

    # Clean two-line title.
    ax.set_title(
        rf"$\it{{{gene_name}}}$ gene  /  {protein_name} protein",
        loc="left",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=10,
    )

    # --------------------------------------------------------
    # Main state legend
    # --------------------------------------------------------
    state_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            label="Protein dominant",
            markerfacecolor=PROTEIN_RGB,
            markeredgecolor="none",
            markersize=6.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            label="Both strong",
            markerfacecolor=BOTH_RGB,
            markeredgecolor="none",
            markersize=6.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            label="Gene dominant",
            markerfacecolor=GENE_RGB,
            markeredgecolor="none",
            markersize=6.5,
        ),
    ]

    legend1 = ax.legend(
        handles=state_handles,
        loc="upper left",
        bbox_to_anchor=(
            0.0,
            1.00,
        ),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handletextpad=0.35,
        columnspacing=1.1,
        borderaxespad=0.0,
    )

    ax.add_artist(
        legend1
    )

    # Optional contour legend appears directly beneath the state legend.
    if contour_handles:
        ax.legend(
            handles=contour_handles,
            loc="upper left",
            bbox_to_anchor=(
                0.0,
                0.955,
            ),
            ncol=min(
                3,
                len(contour_handles),
            ),
            frameon=False,
            fontsize=LEGEND_FONTSIZE - 0.5,
            handlelength=2.4,
            handletextpad=0.5,
            columnspacing=1.0,
            borderaxespad=0.0,
        )

    # Compact statistics annotation; optional but useful for auditability.
    if (
        logfc is not None
        and fdr is not None
    ):
        ax.text(
            0.995,
            0.01,
            (
                rf"$\log FC$={logfc:+.2f}"
                + "\n"
                + rf"FDR={fdr:.1e}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.8,
            color="0.35",
        )

    fig.tight_layout(
        pad=0.5
    )

    ensure_dir(
        os.path.dirname(
            os.path.abspath(
                output_png
            )
        )
    )

    fig.savefig(
        output_png,
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# ============================================================
# DE PAIR SELECTION
# ============================================================

def topk_pairs_per_marker(
    de_df: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    out = de_df.copy()

    out["logFC"] = pd.to_numeric(
        out["logFC"],
        errors="coerce",
    )
    out["AveExpr"] = pd.to_numeric(
        out["AveExpr"],
        errors="coerce",
    )
    out["adj.P.Val"] = pd.to_numeric(
        out["adj.P.Val"],
        errors="coerce",
    )

    out = (
        out.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "marker",
                "gene",
                "logFC",
                "adj.P.Val",
            ]
        )
    )

    out["abs_logFC"] = np.abs(
        out["logFC"]
    )

    if out.columns.duplicated().any():
        out = out.loc[
            :,
            ~out.columns.duplicated(),
        ].copy()

    out = out.sort_values(
        [
            "marker",
            "adj.P.Val",
            "abs_logFC",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    return (
        out.groupby(
            "marker",
            group_keys=False,
        )
        .head(k)
        .copy()
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    data_root: str,
    output_root: str,
    topk: int,
    target_sum: float,
    max_pairs: int,
    plot_points: int,
    point_size: float,
    target_proteins: Optional[Sequence[str]] = TARGET_PROTEINS,
):
    ensure_dir(
        output_root
    )

    manifest_dir = os.path.join(
        output_root,
        "manifest",
    )
    figure_root = os.path.join(
        output_root,
        "per_marker",
    )

    ensure_dir(
        manifest_dir
    )
    ensure_dir(
        figure_root
    )

    # --------------------------------------------------------
    # Load coordinates
    # --------------------------------------------------------
    coord_path = os.path.join(
        data_root,
        XENIUM_LOCATION_CSV,
    )

    print(
        "[LOAD] coordinates:",
        coord_path,
    )

    coord_df = read_xenium_location_weird_csv(
        coord_path
    )

    n_cells = coord_df.shape[0]

    # Preserve the original code's coordinate orientation.
    x_all = coord_df[
        "y"
    ].to_numpy(
        dtype=np.float32
    )

    y_all = coord_df[
        "x"
    ].to_numpy(
        dtype=np.float32
    )

    # One reproducible cell sample shared across all pairs.
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    if (
        plot_points
        and n_cells > plot_points
    ):
        sample_idx = rng.choice(
            n_cells,
            size=plot_points,
            replace=False,
        ).astype(
            np.int32
        )
    else:
        sample_idx = np.arange(
            n_cells,
            dtype=np.int32,
        )

    pos_map = np.full(
        n_cells,
        -1,
        dtype=np.int32,
    )
    pos_map[
        sample_idx
    ] = np.arange(
        sample_idx.size,
        dtype=np.int32,
    )

    x_plot = x_all[
        sample_idx
    ]
    y_plot = y_all[
        sample_idx
    ]

    # --------------------------------------------------------
    # Load predicted protein matrix
    # --------------------------------------------------------
    protein_path = os.path.join(
        data_root,
        PROTEIN_RDS,
    )

    print(
        "[LOAD] predicted protein:",
        protein_path,
    )

    protein_df = ensure_numeric_df(
        read_rds_df(
            protein_path
        )
    )

    protein_df = orient_df_to_match_rows(
        protein_df,
        n_cells,
        "predicted_protein",
    )

    protein_df = ensure_numeric_df(
        protein_df
    )

    protein_map = build_canon_map(
        protein_df.columns
    )

    selected_proteins = resolve_target_proteins(
        requested_proteins=target_proteins,
        available_protein_map=protein_map,
    )

    selected_protein_keys = {
        canonical_name(protein)
        for protein in selected_proteins
    }

    print(
        "[INFO] Selected protein markers: "
        + ", ".join(selected_proteins)
    )

    # --------------------------------------------------------
    # Load gene matrix
    # --------------------------------------------------------
    gene_path = os.path.join(
        data_root,
        GENE_H5AD,
    )

    print(
        "[LOAD] gene h5ad:",
        gene_path,
    )

    adata = ad.read_h5ad(
        gene_path
    )

    gene_names = list(
        adata.var_names
    )

    gene_map = build_canon_map(
        gene_names
    )

    X = adata.X
    X_is_sparse = sp.issparse(
        X
    )

    if X_is_sparse:
        X_csr = X.tocsr()
        X_csc = X.tocsc()
    else:
        X = np.asarray(
            X
        )
        X_csr = None
        X_csc = None

    n_gene_rows = (
        X_csr.shape[0]
        if X_is_sparse
        else X.shape[0]
    )

    n_gene_cols = (
        X_csr.shape[1]
        if X_is_sparse
        else X.shape[1]
    )

    if (
        n_gene_rows != n_cells
        and n_gene_cols == n_cells
    ):
        print(
            "[TRANSPOSE] gene matrix "
            "appears transposed."
        )

        if X_is_sparse:
            X_csr = X_csr.T.tocsr()
            X_csc = X_csc.T.tocsc()
        else:
            X = X.T

    n_gene_rows = (
        X_csr.shape[0]
        if X_is_sparse
        else X.shape[0]
    )

    if n_gene_rows != n_cells:
        raise ValueError(
            "Gene matrix rows do not match "
            f"coordinates: {n_gene_rows} vs {n_cells}"
        )

    if X_is_sparse:
        scale = compute_row_scale(
            X_csr,
            target_sum,
        )
    else:
        row_sum = X.sum(
            axis=1
        ).astype(
            np.float32
        )

        scale = np.zeros_like(
            row_sum,
            dtype=np.float32,
        )

        nonzero = row_sum > 0
        scale[
            nonzero
        ] = (
            target_sum
            / row_sum[nonzero]
        ).astype(
            np.float32
        )

    # --------------------------------------------------------
    # Load + filter DE table
    # --------------------------------------------------------
    de_path = os.path.join(
        data_root,
        DE_RDS,
    )

    print(
        "[LOAD] DE:",
        de_path,
    )

    de_df = read_rds_df(
        de_path
    )

    required = {
        "logFC",
        "AveExpr",
        "adj.P.Val",
        "gene",
        "marker",
    }

    missing = required - set(
        de_df.columns
    )

    if missing:
        raise ValueError(
            f"DE table missing columns: {missing}"
        )

    de_df = de_df.copy()
    de_df["marker_c"] = de_df[
        "marker"
    ].map(
        canonical_name
    )
    de_df["gene_c"] = de_df[
        "gene"
    ].map(
        canonical_name
    )

    de_df = de_df[
        de_df["marker_c"].isin(
            selected_protein_keys
        )
        & de_df["gene_c"].isin(
            gene_map
        )
    ].copy()

    if de_df.empty:
        raise ValueError(
            "No DE rows remain after filtering to the requested "
            "protein markers and available gene names."
        )

    de_df["marker"] = de_df[
        "marker_c"
    ].map(
        protein_map
    )
    de_df["gene"] = de_df[
        "gene_c"
    ].map(
        gene_map
    )

    de_df = de_df.drop(
        columns=[
            "marker_c",
            "gene_c",
        ],
        errors="ignore",
    )

    selected_pairs = topk_pairs_per_marker(
        de_df,
        k=topk,
    )

    if (
        max_pairs
        and selected_pairs.shape[0] > max_pairs
    ):
        selected_pairs = (
            selected_pairs.sort_values(
                [
                    "adj.P.Val",
                    "abs_logFC",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .head(max_pairs)
            .copy()
        )

    manifest_path = os.path.join(
        manifest_dir,
        "selected_gene_protein_pairs.csv",
    )

    protein_manifest_path = os.path.join(
        manifest_dir,
        "selected_protein_markers.txt",
    )

    with open(
        protein_manifest_path,
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n".join(selected_proteins)
            + "\n"
        )

    print(
        "[SAVE] selected protein markers:",
        protein_manifest_path,
    )

    selected_pairs.to_csv(
        manifest_path,
        index=False,
    )

    print(
        "[SAVE] manifest:",
        manifest_path,
    )
    print(
        f"[INFO] plotting "
        f"{selected_pairs.shape[0]} pairs "
        f"using {sample_idx.size} cells"
    )

    # --------------------------------------------------------
    # Pair plotting
    # --------------------------------------------------------
    gene_cache = {}
    protein_cache = {}

    gene_index = {
        gene: idx
        for idx, gene in enumerate(
            gene_names
        )
    }

    selected_pairs = selected_pairs.copy()

    for pair_idx, row in enumerate(
        selected_pairs.itertuples(
            index=False
        ),
        start=1,
    ):
        protein_name = row.marker
        gene_name = row.gene

        logfc = float(
            row.logFC
        )
        fdr = float(
            getattr(
                row,
                "_3",
                row._asdict().get(
                    "adj.P.Val",
                    np.nan,
                ),
            )
        )

        # itertuples sanitizes "adj.P.Val"; safest retrieval:
        fdr = float(
            selected_pairs.iloc[
                pair_idx - 1
            ]["adj.P.Val"]
        )

        if protein_name not in protein_cache:
            protein_cache[
                protein_name
            ] = protein_df[
                protein_name
            ].to_numpy(
                dtype=np.float32,
                copy=False,
            )[sample_idx]

        protein_values = protein_cache[
            protein_name
        ]

        if gene_name not in gene_cache:
            gene_idx = gene_index[
                gene_name
            ]

            gene_cache[
                gene_name
            ] = extract_gene_log1p_on_sample(
                X=X,
                X_is_sparse=X_is_sparse,
                X_csc=X_csc,
                scale=scale,
                gene_idx=gene_idx,
                sample_idx=sample_idx,
                pos_map=pos_map,
            )

        gene_values = gene_cache[
            gene_name
        ]

        marker_dir = os.path.join(
            figure_root,
            safe_name(
                protein_name
            ),
        )
        ensure_dir(
            marker_dir
        )

        output_png = os.path.join(
            marker_dir,
            (
                f"{safe_name(protein_name)}"
                f"__{safe_name(gene_name)}"
                f"__logFC{logfc:+.3f}"
                f"__FDR{fdr:.2e}.png"
            ),
        )

        plot_coexpression_map(
            x=x_plot,
            y=y_plot,
            gene_values=gene_values,
            protein_values=protein_values,
            gene_name=gene_name,
            protein_name=protein_name,
            output_png=output_png,
            logfc=logfc,
            fdr=fdr,
            max_points=plot_points,
            point_size=point_size,
            seed=RANDOM_SEED,
        )

        print(
            f"[SAVE] {pair_idx}/"
            f"{selected_pairs.shape[0]}: "
            f"{output_png}"
        )

    print(
        "[DONE] figures:",
        figure_root,
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Cell/Nature-style "
            "gene-protein spatial co-expression maps."
        )
    )

    parser.add_argument(
        "--data_root",
        default=DATA_ROOT,
        help="Directory containing input RDS/h5ad/coordinate files.",
    )

    parser.add_argument(
        "--output_root",
        default=OUTPUT_ROOT,
        help="Independent directory for all generated outputs.",
    )

    parser.add_argument(
        "--proteins",
        nargs="*",
        default=TARGET_PROTEINS,
        help=(
            "Protein markers to plot, e.g. "
            "--proteins IDO1 HLA-A PanCK. "
            "Pass no names only if you intentionally want all markers."
        ),
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=TOPK_GENES_PER_PROTEIN,
    )

    parser.add_argument(
        "--target_sum",
        type=float,
        default=TARGET_SUM,
    )

    parser.add_argument(
        "--max_pairs",
        type=int,
        default=MAX_PAIRS,
    )

    parser.add_argument(
        "--plot_points",
        type=int,
        default=PLOT_POINTS,
    )

    parser.add_argument(
        "--point_size",
        type=float,
        default=POINT_SIZE,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_pipeline(
        data_root=args.data_root,
        output_root=args.output_root,
        topk=args.topk,
        target_sum=args.target_sum,
        max_pairs=args.max_pairs,
        plot_points=args.plot_points,
        point_size=args.point_size,
        target_proteins=args.proteins,
    )


if __name__ == "__main__":
    main()
