#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-cell validation of virtual protein against paired Xenium gene expression.

Outputs
-------
Tables
- cell_pairing_summary.csv
- protein_gene_marker_metrics.csv
- protein_gene_correlation_matrix_full.csv
- protein_gene_correlation_matrix.csv
- protein_gene_heatmap_order.csv


Figures
- protein_gene_correlation_heatmap.png

- marker_panels/<protein>__<gene>_spatial.png
- marker_panels/<protein>__<gene>_scatter.png

The overall heatmaps report Pearson correlations between generated proteins (rows)
and cognate-gene targets/modules (columns). Displayed protein--gene pairs are first
selected by their cognate Pearson correlation. Their protein correlation profiles are
then hierarchically clustered using correlations against the full available gene/module
universe. Pattern clusters are ranked by cognate correlation strength, and members
within each cluster are likewise ranked so that the strongest matched associations
appear first while related correlation patterns remain grouped. The same paired order
is applied to rows and columns, keeping cognate protein--gene pairs on the diagonal.
Two heatmaps are saved: the raw signed Pearson matrix and the previously requested
column-normalized absolute-correlation view. Marker-specific scatter plots report
Pearson and Spearman correlation.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr


# ============================================================
# CONFIG
# ============================================================
VIRTUAL_PROTEIN_PATH = (
    "/media/huifang/data/registration/phenocycler/huifang/virtual_tissue/virtual_pair/presta/"
    "virtual_protein_1.000_presta_knot30_fitlog1p1_minimal.h5ad"
)
# VIRTUAL_PROTEIN_PATH = ("/media/huifang/data/registration/phenocycler/huifang/virtual_tissue/virtual_pair/virtual_protein_1.000.h5ad")
REAL_GENE_PATH = (
    "/media/huifang/data/registration/phenocycler/huifang/virtual_tissue/virtual_pair/presta/"
    "LUAD_2_A_xenium_0.000_presta_knot30_fitlog1p1_minimal.h5ad"
)
# REAL_GENE_PATH = ("/media/huifang/data/registration/phenocycler/huifang/virtual_tissue/virtual_pair/LUAD_2_A_xenium_0.000.h5ad")



OUTDIR = "/media/huifang/data/registration/paper_analysis/Figure7_modality/bc_evaluation/"

# Saved numerical results / plotting payload.
PAIRING_SUMMARY_CSV = os.path.join(OUTDIR, "cell_pairing_summary.csv")
MARKER_METRICS_CSV = os.path.join(OUTDIR, "protein_gene_marker_metrics.csv")
FULL_CORR_CSV = os.path.join(OUTDIR, "protein_gene_correlation_matrix_full.csv")
DISPLAY_CORR_CSV = os.path.join(OUTDIR, "protein_gene_correlation_matrix.csv")
HEATMAP_ORDER_CSV = os.path.join(OUTDIR, "protein_gene_heatmap_order.csv")
PLOT_PAYLOAD_NPZ = os.path.join(OUTDIR, "protein_gene_plot_payload.npz")
HEATMAP_PNG = os.path.join(OUTDIR, "protein_gene_correlation_heatmap.png")
MARKER_PANEL_DIR = os.path.join(OUTDIR, "marker_panels")

PROTEIN_COORD_KEY = "spatial"
GENE_COORD_KEY = "spatial"
FILTER_VALID = True
VALID_KEY = "valid"
COORD_ATOL = 1e-4
MAX_PAIR_DISTANCE = 1e-2

GENE_LIBRARY_NORMALIZE = True
# None = estimate from the Xenium dataset as the median positive per-cell library size.
# Set a numeric value to override the data-derived target.
GENE_TARGET_SUM: Optional[float] = None
GENE_LOG1P = True
PROTEIN_LOG1P = True
MIN_VALID_CELLS = 20

FIG_DPI = 350
INVERT_Y = True
POINT_SIZE = 0.1
N_MARKERS_TO_PLOT = 8
PLOT_MARKERS: Optional[List[str]] = None  # e.g. ["IDO1", "CD44", "Pan-Cytokeratin"]

# Overall protein x gene correlation heatmaps.
# First select the top N matched protein--gene pairs by their cognate Pearson r.
# Then cluster the displayed proteins by their COMPLETE correlation profile across
# all available cognate-gene targets/modules (not only the displayed columns).
# Pattern clusters and their members are ranked by cognate correlation strength.
# Set N_TOP_HEATMAP_GENES to None or <= 0 to display all available pairs.
N_TOP_HEATMAP_GENES: Optional[int] = 20
N_HEATMAP_PATTERN_CLUSTERS = 8
HEATMAP_CLUSTER_LINKAGE = "average"
HEATMAP_CLUSTER_DISTANCE = "correlation"

# For validation, a strong positive cognate correlation is preferable to a strong
# negative correlation, so ranking uses signed Pearson r by default. Set True only
# if the intended interpretation is association magnitude irrespective of direction.
HEATMAP_RANK_BY_ABS_COGNATE = True

SCATTER_POINT_SIZE = 2.0
SCATTER_ALPHA = 0.10
SCATTER_COLOR = "0.45"
SCATTER_RANGE_PERCENTILES = (0.1, 99.9)
SCATTER_RANGE_PAD_FRAC = 0.035


# ============================================================
# PROTEIN -> COGNATE GENE / GENE MODULE
# ============================================================
PROTEIN_GENE_PAIRS: Dict[str, Tuple[str, ...]] = {
    "CD44": ("CD44",),
    "CD4": ("CD4",),
    "CD31": ("PECAM1",),
    "E-Cadherin": ("CDH1",),
    "LIF": ("LIF",),
    "a-SMA": ("ACTA2",),
    "CD45RO": ("PTPRC",),
    "CD68": ("CD68",),
    "CD20": ("MS4A1",),
    "b-Catenin1": ("CTNNB1",),
    "LAG3": ("LAG3",),
    "b-Actin": ("ACTB",),
    "Podoplanin": ("PDPN",),
    "CD11c": ("ITGAX",),
    "Collagen-IV": ("COL4A1", "COL4A2"),
    "CD8": ("CD8A", "CD8B"),
    "IDO1": ("IDO1",),
    "HLA-A": ("HLA-A",),
    "DC-LAMP": ("LAMP3",),
    "CXCL13": ("CXCL13",),
    "Pan-Cytokeratin": ("KRT8", "KRT18", "KRT19"),
    "Galectin3": ("LGALS3",),
    "CTLA4": ("CTLA4",),
    "HLA-DPB1": ("HLA-DPB1",),
    "PD-L1": ("CD274",),
    "NKX2-1": ("NKX2-1",),
    "TP53": ("TP53",),
    "Ki67": ("MKI67",),
    "CD3e": ("CD3E",),
    "CD163": ("CD163",),
    "FOXP3": ("FOXP3",),
    "PD-1": ("PDCD1",),
    "ICOS": ("ICOS",),
    "CD56": ("NCAM1",),
    "TIGIT": ("TIGIT",),
}


# ============================================================
# I/O + EXTRACTION
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_h5ad(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    adata = sc.read_h5ad(path)
    if FILTER_VALID and VALID_KEY in adata.obs.columns:
        adata = adata[adata.obs[VALID_KEY].astype(bool)].copy()
    return adata


def load_coords(adata, key: str) -> np.ndarray:
    if key not in adata.obsm:
        raise KeyError(f"Coordinate key {key!r} not found in adata.obsm")
    coords = np.asarray(adata.obsm[key], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coordinates under {key!r}: {coords.shape}")
    return coords[:, :2]


def _feature_lookup(adata) -> Dict[str, Tuple[str, str]]:
    lookup: Dict[str, Tuple[str, str]] = {}
    for name in adata.var_names.astype(str):
        lookup.setdefault(name.upper(), ("var", name))
    for name in adata.obs.columns.astype(str):
        if pd.api.types.is_numeric_dtype(adata.obs[name]):
            lookup.setdefault(name.upper(), ("obs", name))
    return lookup


def _extract_one_feature(adata, feature: str) -> np.ndarray:
    hit = _feature_lookup(adata).get(feature.upper())
    if hit is None:
        raise KeyError(feature)
    where, resolved = hit
    if where == "var":
        value = adata.X[:, int(adata.var_names.get_loc(resolved))]
        if sp.issparse(value):
            value = value.toarray()
        value = np.asarray(value).reshape(-1)
    else:
        value = np.asarray(adata.obs[resolved]).reshape(-1)
    return np.nan_to_num(value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extract_protein(adata, protein: str) -> np.ndarray:
    value = _extract_one_feature(adata, protein)
    if np.nanmin(value) < 0:
        raise ValueError(f"Protein marker {protein!r} contains negative values.")
    if PROTEIN_LOG1P:
        value = np.log1p(value)
    return value.astype(np.float32)


def compute_gene_library_scale(adata) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute per-cell Xenium library sizes and normalization factors.

    When ``GENE_TARGET_SUM`` is ``None``, the normalization target is estimated
    directly from the dataset as the median library size among cells with a
    positive total transcript count. This avoids imposing the conventional
    scRNA-seq target of 1e4 on a targeted Xenium panel.
    """
    X = adata.X
    if sp.issparse(X):
        if X.data.size and np.nanmin(X.data) < 0:
            raise ValueError("Real gene matrix contains negative values; expected Xenium counts.")
        library = np.asarray(X.sum(axis=1)).reshape(-1).astype(np.float64)
    else:
        X = np.asarray(X)
        if np.nanmin(X) < 0:
            raise ValueError("Real gene matrix contains negative values; expected Xenium counts.")
        library = X.sum(axis=1, dtype=np.float64)

    library = np.nan_to_num(library, nan=0.0, posinf=0.0, neginf=0.0)
    positive_library = library[library > 0]
    if positive_library.size == 0:
        raise ValueError("No Xenium cells have a positive total transcript count.")

    estimated_target = float(np.median(positive_library))
    target_sum = estimated_target if GENE_TARGET_SUM is None else float(GENE_TARGET_SUM)

    if GENE_LIBRARY_NORMALIZE:
        scale = np.divide(
            target_sum,
            library,
            out=np.zeros_like(library, dtype=np.float64),
            where=library > 0,
        )
    else:
        scale = np.ones_like(library, dtype=np.float64)

    return library.astype(np.float32), scale.astype(np.float32), target_sum


def extract_gene_module(
    adata,
    genes: Sequence[str],
    library_scale: np.ndarray,
) -> Tuple[np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
    vectors: List[np.ndarray] = []
    available: List[str] = []
    missing: List[str] = []
    for gene in genes:
        try:
            vectors.append(_extract_one_feature(adata, gene))
            available.append(gene)
        except KeyError:
            missing.append(gene)

    if not vectors:
        raise KeyError(",".join(genes))

    value = np.sum(np.vstack(vectors), axis=0).astype(np.float32)
    if np.nanmin(value) < 0:
        raise ValueError(f"Gene module {genes!r} contains negative values.")
    value *= np.asarray(library_scale, dtype=np.float32)
    if GENE_LOG1P:
        value = np.log1p(value)
    return value.astype(np.float32), tuple(available), tuple(missing)


# ============================================================
# CELL PAIRING
# ============================================================
def pair_cells(ad_protein, ad_gene):
    p_coords = load_coords(ad_protein, PROTEIN_COORD_KEY)
    g_coords = load_coords(ad_gene, GENE_COORD_KEY)

    # 1) Same row order and matching coordinates.
    if len(p_coords) == len(g_coords):
        distance = np.linalg.norm(p_coords - g_coords, axis=1)
        if np.all(distance <= COORD_ATOL):
            idx = np.arange(len(p_coords), dtype=np.int64)
            return idx, idx.copy(), distance.astype(np.float32), "rowwise_coordinates", g_coords.copy()

    # 2) Shared observation names.
    common = ad_protein.obs_names.intersection(ad_gene.obs_names)
    if len(common):
        p_lookup = pd.Series(np.arange(ad_protein.n_obs), index=ad_protein.obs_names)
        g_lookup = pd.Series(np.arange(ad_gene.n_obs), index=ad_gene.obs_names)
        p_idx = p_lookup.loc[common].to_numpy(dtype=np.int64)
        g_idx = g_lookup.loc[common].to_numpy(dtype=np.int64)
        distance = np.linalg.norm(p_coords[p_idx] - g_coords[g_idx], axis=1)
        return p_idx, g_idx, distance.astype(np.float32), "obs_names", g_coords[g_idx]

    # 3) Mutual nearest coordinates.
    tree_gene = cKDTree(g_coords)
    d_pg, j_pg = tree_gene.query(p_coords, k=1)
    tree_protein = cKDTree(p_coords)
    _, i_gp = tree_protein.query(g_coords, k=1)
    p_all = np.arange(len(p_coords), dtype=np.int64)
    keep = (i_gp[j_pg] == p_all) & (d_pg <= float(MAX_PAIR_DISTANCE))
    p_idx = p_all[keep]
    g_idx = j_pg[keep].astype(np.int64)
    if not len(p_idx):
        raise ValueError("No paired cells found by coordinates.")
    return p_idx, g_idx, d_pg[keep].astype(np.float32), "mutual_nearest_coordinates", g_coords[g_idx]


# ============================================================
# METRICS
# ============================================================
def _valid_xy(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y)
    return x[good], y[good], good


def safe_corr(x: np.ndarray, y: np.ndarray, fn) -> float:
    x, y, _ = _valid_xy(x, y)
    if len(x) < MIN_VALID_CELLS or np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan
    try:
        return float(fn(x, y)[0])
    except Exception:
        return np.nan


def zscore_pairwise(x: np.ndarray, y: np.ndarray):
    x, y, good = _valid_xy(x, y)
    if len(x) < MIN_VALID_CELLS:
        return x, y, good
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12 or not np.isfinite(sx + sy):
        return np.full_like(x, np.nan), np.full_like(y, np.nan), good
    zx = (x - float(np.mean(x))) / sx
    zy = (y - float(np.mean(y))) / sy
    return zx.astype(np.float32), zy.astype(np.float32), good


def compute_metrics(protein: np.ndarray, gene: np.ndarray) -> Dict[str, float]:
    protein_z, gene_z, _ = zscore_pairwise(protein, gene)
    good = np.isfinite(protein_z) & np.isfinite(gene_z)
    protein_z, gene_z = protein_z[good], gene_z[good]
    return {
        "n_cells": int(len(protein_z)),
        "pearson": safe_corr(protein_z, gene_z, pearsonr),
        "spearman": safe_corr(protein_z, gene_z, spearmanr),
    }


# ============================================================
# HEATMAP PATTERN CLUSTERING / ORDERING
# ============================================================
def _fill_nonfinite_profile_matrix(X: np.ndarray) -> np.ndarray:
    """Replace non-finite values column-wise for stable pattern clustering."""
    X = np.asarray(X, dtype=np.float64).copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = np.isfinite(col)
        fill = float(np.median(col[finite])) if np.any(finite) else 0.0
        col[~finite] = fill
        X[:, j] = col
    return X


def cluster_and_rank_matched_pairs(
    full_corr_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    displayed_proteins: Sequence[str],
) -> Tuple[List[str], pd.DataFrame]:
    """Cluster proteins by full gene-correlation patterns, then rank clusters.

    Clustering is performed only on the displayed protein rows, but each row uses
    its correlation profile against *all* available gene/module columns in
    ``full_corr_df``. This mirrors the global-program ordering strategy used in
    the association analysis: the displayed order is not learned from the
    truncated heatmap alone.

    After hierarchical clustering, pattern clusters are ranked by mean cognate
    Pearson correlation. Members within each cluster are ranked by their own
    cognate Pearson correlation. The returned order can therefore be applied to
    both protein rows and their matched gene columns, preserving the cognate
    diagonal while keeping similar patterns together.
    """
    displayed_proteins = [p for p in displayed_proteins if p in full_corr_df.index]
    if not displayed_proteins:
        return [], pd.DataFrame(columns=["protein_marker", "pattern_cluster", "cognate_pearson"])

    metric_lookup = metric_df.set_index("protein_marker")["pearson"].to_dict()
    signed_strength = {p: float(metric_lookup.get(p, np.nan)) for p in displayed_proteins}
    if HEATMAP_RANK_BY_ABS_COGNATE:
        rank_strength = {
            p: abs(v) if np.isfinite(v) else -np.inf
            for p, v in signed_strength.items()
        }
    else:
        rank_strength = {
            p: v if np.isfinite(v) else -np.inf
            for p, v in signed_strength.items()
        }

    # Fallback ranking if clustering is impossible or unnecessary.
    def fallback_order():
        return sorted(displayed_proteins, key=lambda p: rank_strength[p], reverse=True)

    if len(displayed_proteins) <= 1:
        ordered = fallback_order()
        info = pd.DataFrame({
            "protein_marker": ordered,
            "pattern_cluster": [1] * len(ordered),
            "cognate_pearson": [signed_strength[p] for p in ordered],
            "ranking_strength": [rank_strength[p] for p in ordered],
        })
        return ordered, info

    X = full_corr_df.loc[displayed_proteins].to_numpy(dtype=float)
    X = _fill_nonfinite_profile_matrix(X)

    try:
        dist = pdist(X, metric=HEATMAP_CLUSTER_DISTANCE)
        if dist.size == 0 or not np.any(np.isfinite(dist)):
            raise ValueError("degenerate pattern distance")
        dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)
        if np.allclose(dist, 0):
            raise ValueError("all correlation profiles are effectively identical")

        Z = linkage(dist, method=HEATMAP_CLUSTER_LINKAGE)
        try:
            Z = optimal_leaf_ordering(Z, dist)
        except Exception:
            pass

        leaves = leaves_list(Z)
        leaf_order = [displayed_proteins[i] for i in leaves]
        leaf_rank = {p: i for i, p in enumerate(leaf_order)}

        n_clusters = max(1, min(int(N_HEATMAP_PATTERN_CLUSTERS), len(displayed_proteins)))
        raw_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        label_by_protein = {
            displayed_proteins[i]: int(raw_labels[i])
            for i in range(len(displayed_proteins))
        }

        # Rank pattern clusters by mean cognate performance; leaf position is a
        # deterministic secondary key. This keeps coherent pattern modules while
        # moving the strongest modules to the front of the heatmap.
        cluster_members: Dict[int, List[str]] = {}
        for p in displayed_proteins:
            cluster_members.setdefault(label_by_protein[p], []).append(p)

        cluster_summary = []
        for lab, members in cluster_members.items():
            vals = np.array([rank_strength[p] for p in members], dtype=float)
            finite = np.isfinite(vals)
            cluster_score = float(np.mean(vals[finite])) if np.any(finite) else -np.inf
            first_leaf = min(leaf_rank[p] for p in members)
            cluster_summary.append((lab, cluster_score, first_leaf))
        cluster_summary.sort(key=lambda x: (-x[1], x[2]))

        ordered: List[str] = []
        display_cluster_by_protein: Dict[str, int] = {}
        for display_lab, (raw_lab, _, _) in enumerate(cluster_summary, start=1):
            members = cluster_members[raw_lab]
            members.sort(key=lambda p: (-rank_strength[p], leaf_rank[p]))
            ordered.extend(members)
            for p in members:
                display_cluster_by_protein[p] = display_lab

        info = pd.DataFrame({
            "protein_marker": ordered,
            "pattern_cluster": [display_cluster_by_protein[p] for p in ordered],
            "cognate_pearson": [signed_strength[p] for p in ordered],
            "ranking_strength": [rank_strength[p] for p in ordered],
        })
        return ordered, info

    except Exception as exc:
        print(f"[WARN] Heatmap pattern clustering failed ({exc}); using cognate-correlation ranking.")
        ordered = fallback_order()
        info = pd.DataFrame({
            "protein_marker": ordered,
            "pattern_cluster": [1] * len(ordered),
            "cognate_pearson": [signed_strength[p] for p in ordered],
            "ranking_strength": [rank_strength[p] for p in ordered],
        })
        return ordered, info


def _pattern_cluster_boundaries(order_info: Optional[pd.DataFrame]) -> List[int]:
    """Return matrix boundaries between consecutive displayed pattern clusters."""
    if order_info is None or order_info.empty or "pattern_cluster" not in order_info:
        return []
    labels = order_info["pattern_cluster"].to_numpy()
    return [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]


# ============================================================
# PLOTTING
# ============================================================
def set_style() -> None:
    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, path: str) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {path}")


def _heatmap_figsize(df: pd.DataFrame) -> Tuple[float, float]:
    n_rows, n_cols = df.shape
    return max(9.0, 0.34 * n_cols + 3.0), max(7.0, 0.30 * n_rows + 2.5)


def _format_heatmap_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=16,pad=15)
    ax.set_xlabel("Measured gene markers", fontsize=14, weight="bold")
    ax.set_ylabel("Generated protein markers", fontsize=14, weight="bold")
    ax.tick_params(axis="x", rotation=90, labelsize=12)
    ax.tick_params(axis="y", rotation=0, labelsize=12)

    # Gene symbols are italicized; protein-marker names remain roman.
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontstyle("italic")
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontstyle("normal")

def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    path: str,
    order_info: Optional[pd.DataFrame] = None,
) -> None:
    """Raw signed protein x gene Pearson-correlation matrix."""
    set_style()
    fig, ax = plt.subplots(figsize=_heatmap_figsize(corr_df))
    hm=sns.heatmap(
        corr_df,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.12,
        linecolor="white",
        cbar_kws={"label": r"Pearson $r$", "shrink": 0.72, "pad": 0.008},
        ax=ax,
    )

    cbar = hm.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.tick_params(labelsize=11)
    # Because rows and matched columns use the same pair order, a single set of
    # boundaries separates the learned correlation-pattern modules on both axes.
    for b in _pattern_cluster_boundaries(order_info):
        ax.axhline(b, color="0.55", lw=1.25, alpha=0.75)
        ax.axvline(b, color="0.55", lw=1.25, alpha=0.75)
    _format_heatmap_axes(ax, "Gene–protein concordance across matched cells")
    fig.tight_layout()
    save_fig(fig, path)



def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _robust_axis_range(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return -1.0, 1.0
    lo, hi = np.percentile(values, SCATTER_RANGE_PERCENTILES)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    pad = max((hi - lo) * SCATTER_RANGE_PAD_FRAC, 1e-3)
    return float(lo - pad), float(hi + pad)





SCATTER_POINT_SIZE = 1.0
SCATTER_DENSITY_CMAP = "coolwarm"
SCATTER_DENSITY_COLOR_PERCENTILES = (1.0, 99.7)

# Density-dependent transparency for the real scatter points.
# Low-density observations are nearly transparent; high-density observations
# approach full opacity.  ALPHA_POWER > 1 suppresses sparse observations more
# aggressively while preserving dense cores.
SCATTER_DENSITY_ALPHA_MIN = 0.008
SCATTER_DENSITY_ALPHA_MAX = 0.92
SCATTER_DENSITY_ALPHA_POWER = 2.

# Gaussian redistribution dots use the same density-aware transparency trend,
# multiplied by this global scale.  Because many replicas overlap, keep this
# substantially below 1.0.
SCATTER_GAUSSIAN_ALPHA_SCALE = 0.22

# ------------------------------------------------------------------
# Adaptive Gaussian scatter rendering
# ------------------------------------------------------------------
# A moderate kNN radius is used only to estimate local point density
# in the gene-protein correlation space.  It does NOT smooth the
# biological measurements or alter the reported correlations.
SCATTER_PILOT_K = 300

# Gaussian bandwidth is measured in the standardized correlation space.
# For a typical-density point (pilot radius ~= median pilot radius),
# sigma ~= BASE_SCALE * median_radius.  Sparse points have a larger
# pilot radius and therefore receive a much wider Gaussian kernel.
SCATTER_GAUSSIAN_BASE_SCALE = 10
SCATTER_GAUSSIAN_BANDWIDTH_POWER = 1
SCATTER_GAUSSIAN_MIN_SCALE = 10
SCATTER_GAUSSIAN_MAX_SCALE = 10

# Dense points retain almost all of their weight at the measured
# coordinate.  As local sparsity increases, progressively more visual
# weight is redistributed through the Gaussian kernel.
SCATTER_SPREAD_START_RATIO = 1.05
SCATTER_SPREAD_FULL_RATIO = 2.50
SCATTER_SPREAD_POWER = 1.20
SCATTER_MAX_SPREAD_FRACTION = 0.95

# Monte-Carlo Gaussian "splatting" keeps the final panel looking like a
# true scatter plot instead of a raster/KDE image.  The original points
# are always drawn; these replicas are visualization-only samples from
# each sparse point's Gaussian kernel.
SCATTER_GAUSSIAN_REPLICATES = 6
SCATTER_GAUSSIAN_POINT_SIZE = 0.70
SCATTER_MAX_GAUSSIAN_POINTS = 1_500_000
SCATTER_RANDOM_SEED = 1729


def plot_paired_scatter(
    ax,
    gene_z: np.ndarray,
    protein_z: np.ndarray,
    gene_label: str,
    protein_marker: str,
) -> None:
    """
    Adaptive-bandwidth Gaussian scatter visualization.

    Every measured cell is retained at its original gene/protein coordinate.
    In addition, each point can redistribute part of its *visualization weight*
    through a Gaussian kernel in the gene-protein correlation space.

    Local bandwidth is inverse to point density:

        dense region  -> small kNN radius -> narrow Gaussian -> almost no blur
        sparse region -> large kNN radius -> wide Gaussian   -> stronger spread

    The Gaussian redistribution is strictly a rendering operation.  It does
    not alter gene/protein values, cell pairing, spatial maps, heatmaps, or
    Pearson/Spearman correlations.
    """

    # ========================================================
    # 1. ORIGINAL PAIRED OBSERVATIONS
    # ========================================================
    gene_z = np.asarray(gene_z, dtype=np.float32).reshape(-1)
    protein_z = np.asarray(protein_z, dtype=np.float32).reshape(-1)

    good = np.isfinite(gene_z) & np.isfinite(protein_z)
    x = gene_z[good]
    y = protein_z[good]

    if len(x) == 0:
        ax.text(
            0.5, 0.5, "No valid observations",
            transform=ax.transAxes,
            ha="center", va="center", fontsize=12,
        )
        return

    # ========================================================
    # 2. DISPLAY RANGE
    # ========================================================
    xlim = _robust_axis_range(x)
    ylim = _robust_axis_range(y)

    in_view = (
        (x >= xlim[0]) & (x <= xlim[1]) &
        (y >= ylim[0]) & (y <= ylim[1])
    )

    xv = x[in_view]
    yv = y[in_view]

    if len(xv) == 0:
        xv = x
        yv = y

    n_view = len(xv)

    # ========================================================
    # 3. STANDARDIZED CORRELATION SPACE
    #
    # Neighbor searching and Gaussian spreading are both performed in
    # the gene-protein scatter space, NOT in tissue x/y coordinates.
    # Standardization prevents one axis from dominating Euclidean
    # distance if their numerical ranges differ.
    # ========================================================
    x_mean = float(np.mean(xv))
    y_mean = float(np.mean(yv))
    x_scale = max(float(np.std(xv)), 1e-8)
    y_scale = max(float(np.std(yv)), 1e-8)

    xs = (xv - x_mean) / x_scale
    ys = (yv - y_mean) / y_scale
    xy = np.column_stack([xs, ys]).astype(np.float32)

    # Default values for a tiny dataset.
    density_color = np.ones(n_view, dtype=np.float64)
    bandwidth = np.zeros(n_view, dtype=np.float64)
    spread_fraction = np.zeros(n_view, dtype=np.float64)

    if n_view > 2:
        # ====================================================
        # 4. PILOT kNN RADIUS = LOCAL SPARSITY ESTIMATE
        # ====================================================
        pilot_k = min(max(int(SCATTER_PILOT_K), 1), n_view - 1)
        tree = cKDTree(xy)

        pilot_radius, _ = tree.query(
            xy,
            k=[pilot_k + 1],  # first neighbor is the point itself
            workers=-1,
        )
        pilot_radius = np.asarray(pilot_radius, dtype=np.float64).reshape(-1)

        positive_radius = pilot_radius[
            np.isfinite(pilot_radius) & (pilot_radius > 0)
        ]
        if positive_radius.size:
            eps = max(
                float(np.percentile(positive_radius, 0.1)) * 0.1,
                1e-12,
            )
        else:
            eps = 1e-12

        pilot_radius = np.maximum(pilot_radius, eps)
        reference_radius = max(float(np.median(pilot_radius)), eps)
        relative_radius = pilot_radius / reference_radius

        # ====================================================
        # 5. TRUE ADAPTIVE GAUSSIAN BANDWIDTH
        #
        # This is the quantity that controls how far a point's visual
        # weight is spread in the correlation space.
        #
        # power > 1 makes sparse-region bandwidth grow faster.
        # ====================================================
        bandwidth = (
            SCATTER_GAUSSIAN_BASE_SCALE
            * reference_radius
            * np.power(relative_radius, SCATTER_GAUSSIAN_BANDWIDTH_POWER)
        )

        bandwidth = np.clip(
            bandwidth,
            SCATTER_GAUSSIAN_MIN_SCALE * reference_radius,
            SCATTER_GAUSSIAN_MAX_SCALE * reference_radius,
        )

        # ====================================================
        # 6. FRACTION OF VISUAL WEIGHT TO REDISTRIBUTE
        #
        # Typical/dense cells remain almost exactly where measured.
        # Boundary/sparse cells progressively move visual weight into
        # their broader Gaussian kernel.
        # ====================================================
        spread_denominator = max(
            SCATTER_SPREAD_FULL_RATIO - SCATTER_SPREAD_START_RATIO,
            1e-8,
        )
        spread_level = np.clip(
            (relative_radius - SCATTER_SPREAD_START_RATIO)
            / spread_denominator,
            0.0,
            1.0,
        )
        spread_fraction = (
            SCATTER_MAX_SPREAD_FRACTION
            * np.power(spread_level, SCATTER_SPREAD_POWER)
        )

        # ====================================================
        # 7. LOCAL DENSITY USED ONLY FOR POINT COLOR
        # ====================================================
        local_density = (
            float(pilot_k)
            / (np.pi * np.maximum(pilot_radius, eps) ** 2)
        )
        density_color = np.log1p(local_density)

    # ========================================================
    # 8. ROBUST DENSITY COLOR SCALE
    # ========================================================
    finite_density = density_color[np.isfinite(density_color)]
    if finite_density.size:
        color_vmin, color_vmax = np.percentile(
            finite_density,
            SCATTER_DENSITY_COLOR_PERCENTILES,
        )
        if (
            not np.isfinite(color_vmin)
            or not np.isfinite(color_vmax)
            or color_vmax <= color_vmin
        ):
            color_vmin = float(np.min(finite_density))
            color_vmax = float(np.max(finite_density))
            if color_vmax <= color_vmin:
                color_vmax = color_vmin + 1.0
    else:
        color_vmin, color_vmax = 0.0, 1.0

    denom = max(float(color_vmax - color_vmin), 1e-12)
    density_norm = np.clip(
        (density_color - color_vmin) / denom,
        0.0,
        1.0,
    )

    # Density-dependent transparency.  This is independent of the colormap:
    #   density_norm ~= 0 -> nearly transparent
    #   density_norm ~= 1 -> nearly solid
    # Increasing SCATTER_DENSITY_ALPHA_POWER makes sparse points fade faster.
    density_alpha = (
        SCATTER_DENSITY_ALPHA_MIN
        + (SCATTER_DENSITY_ALPHA_MAX - SCATTER_DENSITY_ALPHA_MIN)
        * np.power(density_norm, SCATTER_DENSITY_ALPHA_POWER)
    )
    density_alpha = np.clip(density_alpha, 0.0, 1.0)

    cmap = plt.get_cmap(SCATTER_DENSITY_CMAP)
    base_rgba = cmap(density_norm)

    # ========================================================
    # 9. GAUSSIAN REDISTRIBUTION FOR SPARSE POINTS
    #
    # This is Monte-Carlo Gaussian splatting.  Each replica is a tiny
    # visualization-only dot sampled from N(original_point, bandwidth).
    # The random seed is fixed, so figures are reproducible.
    # ========================================================
    sparse_idx = np.flatnonzero(spread_fraction > 1e-4)

    if sparse_idx.size:
        requested_replicates = max(int(SCATTER_GAUSSIAN_REPLICATES), 1)

        # Keep rendering bounded for very large datasets.
        max_replicates_by_budget = max(
            int(SCATTER_MAX_GAUSSIAN_POINTS // max(len(sparse_idx), 1)),
            1,
        )
        n_replicates = min(requested_replicates, max_replicates_by_budget)

        # If even one replica per sparse source exceeds the rendering
        # budget, keep the most strongly redistributed sources.
        if len(sparse_idx) > int(SCATTER_MAX_GAUSSIAN_POINTS):
            rank = np.argsort(spread_fraction[sparse_idx])[::-1]
            sparse_idx = sparse_idx[rank[: int(SCATTER_MAX_GAUSSIAN_POINTS)]]
            n_replicates = 1

        rng = np.random.default_rng(SCATTER_RANDOM_SEED)

        source_xs = xs[sparse_idx]
        source_ys = ys[sparse_idx]
        source_bw = bandwidth[sparse_idx]
        source_spread = spread_fraction[sparse_idx]
        source_rgba = base_rgba[sparse_idx].copy()
        source_density_alpha = density_alpha[sparse_idx]

        # Gaussian replicas inherit the same density-dependent transparency.
        # Sparse boundary observations therefore remain soft, while regions
        # approaching the dense core become progressively more visible.
        # sqrt(n) keeps total opacity reasonably stable as replica count changes.
        kernel_alpha = (
            SCATTER_GAUSSIAN_ALPHA_SCALE
            * source_density_alpha
            * source_spread
            / np.sqrt(float(n_replicates))
        )

        for _ in range(n_replicates):
            dx = rng.normal(
                loc=0.0,
                scale=source_bw,
                size=len(sparse_idx),
            )
            dy = rng.normal(
                loc=0.0,
                scale=source_bw,
                size=len(sparse_idx),
            )

            # Convert Gaussian samples from standardized correlation
            # space back to the displayed gene/protein coordinates.
            x_kernel = x_mean + (source_xs + dx) * x_scale
            y_kernel = y_mean + (source_ys + dy) * y_scale

            kernel_in_view = (
                (x_kernel >= xlim[0]) & (x_kernel <= xlim[1]) &
                (y_kernel >= ylim[0]) & (y_kernel <= ylim[1])
            )

            if not np.any(kernel_in_view):
                continue

            kernel_rgba = source_rgba[kernel_in_view].copy()
            kernel_rgba[:, 3] = np.clip(
                kernel_alpha[kernel_in_view],
                0.0,
                1.0,
            )

            ax.scatter(
                x_kernel[kernel_in_view],
                y_kernel[kernel_in_view],
                c=kernel_rgba,
                s=SCATTER_GAUSSIAN_POINT_SIZE,
                linewidths=0,
                edgecolors="none",
                rasterized=True,
                zorder=1,
            )

    # ========================================================
    # 10. ORIGINAL REAL POINTS
    #
    # Dense points retain essentially their original opacity.  Sparse
    # boundary points keep a faint real-coordinate core while most of
    # their visual weight has been redistributed through the Gaussian.
    # This suppresses hard artificial envelopes without replacing the
    # plot with a raster density image.
    # ========================================================
    # Original points use density-controlled opacity as well.  Dense points are
    # nearly solid; sparse boundary points are close to transparent.  The
    # spread_fraction term further reduces the real-coordinate core when its
    # visualization weight has been redistributed through the Gaussian kernel.
    core_alpha = density_alpha * (1.0 - spread_fraction)
    core_alpha = np.clip(
        core_alpha,
        SCATTER_DENSITY_ALPHA_MIN * 0.25,
        SCATTER_DENSITY_ALPHA_MAX,
    )

    original_rgba = base_rgba.copy()
    original_rgba[:, 3] = core_alpha

    # Draw sparse cells first and dense cells last.
    order = np.argsort(density_color)

    ax.scatter(
        xv[order],
        yv[order],
        c=original_rgba[order],
        s=SCATTER_POINT_SIZE,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )

    # ========================================================
    # 11. IDENTITY / REFERENCE LINE
    # ========================================================
    diag_lo = max(xlim[0], ylim[0])
    diag_hi = min(xlim[1], ylim[1])

    if diag_hi > diag_lo:
        ax.plot(
            [diag_lo, diag_hi],
            [diag_lo, diag_hi],
            ls="--",
            lw=0.85,
            color="0.35",
            alpha=0.75,
            zorder=4,
        )

    # ========================================================
    # 12. AXES
    # ========================================================
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_box_aspect(1)

    ax.set_xlabel(
        f"Measured {gene_label} gene expression (z-score)",
        fontsize=13,
    )
    ax.set_ylabel(
        f"Generated {protein_marker} protein signal (z-score)",
        fontsize=13,
    )
    # ax.set_title(
    #     f"Gene {gene_label} (real) | Protein {protein_marker} (virtual)"
    # )

    # ========================================================
    # 13. ORIGINAL CELL-LEVEL CORRELATION
    # ========================================================
    stats_text = (
        rf"Pearson $r$ = {safe_corr(x, y, pearsonr):.2f}"
        "\n"
        rf"Spearman $\rho$ = {safe_corr(x, y, spearmanr):.2f}"
    )

    ax.text(
        0.035,
        0.965,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="0.82",
            alpha=0.90,
        ),
        zorder=6,
    )

def plot_marker_panel(
    protein_marker: str,
    gene_label: str,
    protein_vals: np.ndarray,
    gene_vals: np.ndarray,
    coords: np.ndarray,
    spatial_path: str,
    scatter_path: str,
) -> None:
    set_style()
    protein_z, gene_z, valid_pair = zscore_pairwise(protein_vals, gene_vals)
    coords_z = coords[valid_pair]
    finite = np.isfinite(protein_z) & np.isfinite(gene_z)
    protein_z, gene_z, coords_z = protein_z[finite], gene_z[finite], coords_z[finite]

    z_abs = np.concatenate([np.abs(protein_z), np.abs(gene_z)]) if len(protein_z) else np.array([1.0])
    zlim = max(float(np.percentile(z_abs[np.isfinite(z_abs)], 95)), 1e-6)

    # Spatial comparison.
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1))
    sc0 = axes[0].scatter(
        coords_z[:, 0], coords_z[:, 1], c=gene_z, s=POINT_SIZE,
        cmap="vlag", vmin=-zlim, vmax=zlim, linewidths=0, rasterized=True,
    )
    axes[0].set_title(f"Measured gene expression: {gene_label}")
    fig.colorbar(sc0, ax=axes[0], fraction=0.026, pad=0.02, label="Gene expression (z-score)")

    sc1 = axes[1].scatter(
        coords_z[:, 0], coords_z[:, 1], c=protein_z, s=POINT_SIZE,
        cmap="vlag", vmin=-zlim, vmax=zlim, linewidths=0, rasterized=True,
    )
    axes[1].set_title(f"Generated protein signal: {protein_marker}")
    fig.colorbar(sc1, ax=axes[1], fraction=0.026, pad=0.02, label="Protein signal (z-score)")

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        if INVERT_Y:
            ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    save_fig(fig, spatial_path)

    # Paired-cell scatter.
    fig, ax = plt.subplots(figsize=(4.7, 4.6))
    plot_paired_scatter(ax, gene_z, protein_z, gene_label, protein_marker)
    fig.tight_layout()
    save_fig(fig, scatter_path)


# ============================================================
# EVALUATION / SAVED-RESULT WORKFLOW
# ============================================================
def _save_plot_payload(
    pair_cache: Dict[str, Dict[str, object]],
    coords: np.ndarray,
) -> None:
    """Save processed paired-cell arrays required by plot-only mode.

    Plot mode must not reopen the h5ad inputs or recompute preprocessing. The
    payload therefore stores the exact paired protein/gene vectors and paired
    coordinates produced during evaluation.
    """
    proteins = list(pair_cache.keys())
    if not proteins:
        raise ValueError("Cannot save plotting payload: pair cache is empty.")

    protein_values = np.vstack([
        np.asarray(pair_cache[p]["protein_vals"], dtype=np.float32)
        for p in proteins
    ])
    gene_values = np.vstack([
        np.asarray(pair_cache[p]["gene_vals"], dtype=np.float32)
        for p in proteins
    ])
    gene_labels = np.asarray(
        [str(pair_cache[p]["gene_label"]) for p in proteins],
        dtype=str,
    )

    np.savez_compressed(
        PLOT_PAYLOAD_NPZ,
        protein_markers=np.asarray(proteins, dtype=str),
        gene_labels=gene_labels,
        protein_values=protein_values,
        gene_values=gene_values,
        coords=np.asarray(coords, dtype=np.float32),
    )
    print(f"[SAVE] {PLOT_PAYLOAD_NPZ}")


def _load_plot_payload() -> Tuple[Dict[str, Dict[str, object]], np.ndarray]:
    if not os.path.isfile(PLOT_PAYLOAD_NPZ):
        raise FileNotFoundError(
            f"Missing saved plotting payload: {PLOT_PAYLOAD_NPZ}. "
            "Run --mode eval first."
        )

    with np.load(PLOT_PAYLOAD_NPZ, allow_pickle=False) as payload:
        proteins = payload["protein_markers"].astype(str).tolist()
        gene_labels = payload["gene_labels"].astype(str).tolist()
        protein_values = np.asarray(payload["protein_values"], dtype=np.float32)
        gene_values = np.asarray(payload["gene_values"], dtype=np.float32)
        coords = np.asarray(payload["coords"], dtype=np.float32)

    if protein_values.shape != gene_values.shape:
        raise ValueError(
            "Saved protein/gene plotting payload has inconsistent matrix shapes: "
            f"{protein_values.shape} vs {gene_values.shape}."
        )
    if protein_values.shape[0] != len(proteins):
        raise ValueError("Saved plotting payload marker dimension is inconsistent.")
    if protein_values.shape[1] != len(coords):
        raise ValueError("Saved plotting payload coordinate dimension is inconsistent.")

    pair_cache: Dict[str, Dict[str, object]] = {}
    for i, protein_marker in enumerate(proteins):
        pair_cache[protein_marker] = {
            "gene_label": gene_labels[i],
            "protein_vals": protein_values[i],
            "gene_vals": gene_values[i],
        }
    return pair_cache, coords


def evaluate_and_save() -> None:
    """Calculate all validation metrics and save results; create no figures."""
    ensure_dir(OUTDIR)

    print(f"[LOAD] Virtual protein: {VIRTUAL_PROTEIN_PATH}")
    ad_protein = read_h5ad(VIRTUAL_PROTEIN_PATH)
    print(f"[LOAD] Real Xenium gene: {REAL_GENE_PATH}")
    ad_gene = read_h5ad(REAL_GENE_PATH)

    gene_library, gene_library_scale, gene_target_sum = compute_gene_library_scale(ad_gene)
    positive_library = gene_library[gene_library > 0]
    library_median = float(np.median(positive_library))
    library_mean = float(np.mean(positive_library))
    library_q25, library_q75 = np.percentile(positive_library, [25, 75])
    target_source = "dataset median" if GENE_TARGET_SUM is None else "user-specified"

    print(
        "[INFO] Xenium per-cell library size: "
        f"median={library_median:.3f}, mean={library_mean:.3f}, "
        f"Q25={library_q25:.3f}, Q75={library_q75:.3f}, "
        f"n_positive={len(positive_library):,}"
    )
    print(
        "[INFO] Gene preprocessing: "
        f"library_normalize={GENE_LIBRARY_NORMALIZE}, "
        f"target_sum={gene_target_sum:.3f} ({target_source}), "
        f"log1p={GENE_LOG1P}"
    )
    print(
        f"[INFO] Protein preprocessing: log1p={PROTEIN_LOG1P}, "
        "no total-protein normalization"
    )

    p_idx, g_idx, pair_dist, pair_method, coords = pair_cells(ad_protein, ad_gene)

    summary_df = pd.DataFrame([{
        "n_virtual_protein_cells": int(ad_protein.n_obs),
        "n_real_gene_cells": int(ad_gene.n_obs),
        "n_paired_cells": int(len(p_idx)),
        "paired_fraction_vs_protein": float(len(p_idx) / max(ad_protein.n_obs, 1)),
        "paired_fraction_vs_gene": float(len(g_idx) / max(ad_gene.n_obs, 1)),
        "pairing_method": pair_method,
        "median_pair_distance": float(np.median(pair_dist)) if len(pair_dist) else np.nan,
        "max_pair_distance": float(np.max(pair_dist)) if len(pair_dist) else np.nan,
        "protein_coord_key": PROTEIN_COORD_KEY,
        "gene_coord_key": GENE_COORD_KEY,
        "gene_library_normalize": bool(GENE_LIBRARY_NORMALIZE),
        "gene_target_sum": float(gene_target_sum),
        "gene_target_sum_source": target_source,
        "gene_log1p": bool(GENE_LOG1P),
        "protein_log1p": bool(PROTEIN_LOG1P),
        "gene_median_library_size": library_median,
        "gene_mean_library_size": library_mean,
        "gene_library_q25": float(library_q25),
        "gene_library_q75": float(library_q75),
    }])
    summary_df.to_csv(PAIRING_SUMMARY_CSV, index=False)
    print(f"[SAVE] {PAIRING_SUMMARY_CSV}")

    rows = []
    pair_cache: Dict[str, Dict[str, object]] = {}

    # Extract each requested protein and its cognate gene/module once.
    for protein_marker, gene_module in PROTEIN_GENE_PAIRS.items():
        try:
            protein_vals = extract_protein(ad_protein, protein_marker)[p_idx]
        except KeyError:
            print(f"[SKIP] Protein marker not found: {protein_marker}")
            continue

        try:
            gene_all, available_genes, missing_genes = extract_gene_module(
                ad_gene, gene_module, gene_library_scale
            )
        except KeyError:
            print(f"[SKIP] Gene marker(s) not found for {protein_marker}: {gene_module}")
            continue

        gene_vals = gene_all[g_idx]
        gene_label = "+".join(available_genes)
        metrics = compute_metrics(protein_vals, gene_vals)

        rows.append({
            "protein_marker": protein_marker,
            "gene_marker": gene_label,
            "requested_gene_marker": "+".join(gene_module),
            "missing_genes": "+".join(missing_genes),
            "gene_preprocessing": (
                f"library_norm_{gene_target_sum:g}+log1p"
                if GENE_LIBRARY_NORMALIZE and GENE_LOG1P
                else "library_norm" if GENE_LIBRARY_NORMALIZE
                else "log1p" if GENE_LOG1P else "none"
            ),
            "protein_preprocessing": "log1p" if PROTEIN_LOG1P else "none",
            "metric_preprocessing": "independent_marker_zscore",
            **metrics,
        })
        pair_cache[protein_marker] = {
            "gene_label": gene_label,
            "protein_vals": np.asarray(protein_vals, dtype=np.float32),
            "gene_vals": np.asarray(gene_vals, dtype=np.float32),
        }

    if not rows:
        raise ValueError("No valid protein-gene marker pairs were found in the input files.")

    metric_df = pd.DataFrame(rows).sort_values(
        "spearman", ascending=False, na_position="last"
    )
    metric_df.to_csv(MARKER_METRICS_CSV, index=False)
    print(f"[SAVE] {MARKER_METRICS_CSV}")

    # Build the complete protein x gene Pearson-correlation matrix first. It is
    # used for pattern clustering even when only the top matched pairs are shown.
    all_proteins = [
        p for p in metric_df["protein_marker"].tolist()
        if p in pair_cache
    ]
    all_gene_entries = [
        (pair_cache[p]["gene_label"], pair_cache[p]["gene_vals"])
        for p in all_proteins
    ]

    full_corr = np.full(
        (len(all_proteins), len(all_gene_entries)),
        np.nan,
        dtype=np.float32,
    )
    for i, protein_marker in enumerate(all_proteins):
        protein_vals = pair_cache[protein_marker]["protein_vals"]
        for j, (_, gene_vals) in enumerate(all_gene_entries):
            full_corr[i, j] = safe_corr(protein_vals, gene_vals, pearsonr)

    full_corr_df = pd.DataFrame(
        full_corr,
        index=pd.Index(all_proteins, name="Generated protein marker"),
        columns=pd.Index(
            [str(label) for label, _ in all_gene_entries],
            name="Measured gene marker",
        ),
    )
    full_corr_df.to_csv(FULL_CORR_CSV)
    print(f"[SAVE] {FULL_CORR_CSV}")

    # Select displayed matched pairs by cognate performance.
    heatmap_rank_df = metric_df.copy()
    if HEATMAP_RANK_BY_ABS_COGNATE:
        heatmap_rank_df["selection_strength"] = heatmap_rank_df["pearson"].abs()
    else:
        heatmap_rank_df["selection_strength"] = heatmap_rank_df["pearson"]
    heatmap_rank_df = heatmap_rank_df.sort_values(
        "selection_strength", ascending=False, na_position="last"
    ).reset_index(drop=True)
    if N_TOP_HEATMAP_GENES is not None and N_TOP_HEATMAP_GENES > 0:
        heatmap_rank_df = heatmap_rank_df.head(int(N_TOP_HEATMAP_GENES))
    displayed_seed = heatmap_rank_df["protein_marker"].tolist()

    proteins, order_info = cluster_and_rank_matched_pairs(
        full_corr_df=full_corr_df,
        metric_df=metric_df,
        displayed_proteins=displayed_seed,
    )
    if not proteins:
        raise ValueError("No proteins remained for heatmap plotting after clustering.")

    order_info["gene_marker"] = [
        str(pair_cache[p]["gene_label"])
        for p in proteins
    ]
    order_info.to_csv(HEATMAP_ORDER_CSV, index=False)
    print(f"[SAVE] {HEATMAP_ORDER_CSV}")

    displayed_genes = [str(pair_cache[p]["gene_label"]) for p in proteins]
    corr_df = full_corr_df.loc[proteins, displayed_genes].copy()
    corr_df.to_csv(DISPLAY_CORR_CSV)
    print(f"[SAVE] {DISPLAY_CORR_CSV}")

    # Save exact paired arrays needed for marker-panel rendering in plot mode.
    _save_plot_payload(pair_cache=pair_cache, coords=coords)

    print(f"[DONE] Evaluation results saved to: {OUTDIR}")


# ============================================================
# PLOT-ONLY WORKFLOW
# ============================================================
def _require_saved_results() -> None:
    required = [
        PAIRING_SUMMARY_CSV,
        MARKER_METRICS_CSV,
        FULL_CORR_CSV,
        DISPLAY_CORR_CSV,
        HEATMAP_ORDER_CSV,
        PLOT_PAYLOAD_NPZ,
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Missing saved result(s). Run --mode eval first:\n  "
            + "\n  ".join(missing)
        )


def plot_saved_results() -> None:
    """Generate figures exclusively from saved evaluation results."""
    _require_saved_results()
    ensure_dir(OUTDIR)
    ensure_dir(MARKER_PANEL_DIR)

    metric_df = pd.read_csv(MARKER_METRICS_CSV)
    corr_df = pd.read_csv(DISPLAY_CORR_CSV, index_col=0)
    order_info = pd.read_csv(HEATMAP_ORDER_CSV)
    pair_cache, coords = _load_plot_payload()

    # Keep label dtypes stable after CSV round-trip.
    corr_df.index = corr_df.index.astype(str)
    corr_df.columns = corr_df.columns.astype(str)
    if "protein_marker" in order_info.columns:
        order_info["protein_marker"] = order_info["protein_marker"].astype(str)
    if "gene_marker" in order_info.columns:
        order_info["gene_marker"] = order_info["gene_marker"].astype(str)
    if "protein_marker" in metric_df.columns:
        metric_df["protein_marker"] = metric_df["protein_marker"].astype(str)

    # Main correlation heatmap.
    plot_correlation_heatmap(
        corr_df,
        HEATMAP_PNG,
        order_info=order_info,
    )

    # # Marker panels: preserve the exact selection behavior of the attached code.
    if PLOT_MARKERS is None:
        plot_markers = metric_df["protein_marker"].head(N_MARKERS_TO_PLOT).tolist()
    else:
        plot_markers = [m for m in PLOT_MARKERS if m in pair_cache]

    for protein_marker in plot_markers:
        if protein_marker not in pair_cache:
            print(f"[WARN] No saved plotting payload for marker: {protein_marker}")
            continue
        item = pair_cache[protein_marker]
        gene_label = str(item["gene_label"])
        base = _safe_filename(f"{protein_marker}__{gene_label}")
        plot_marker_panel(
            protein_marker=protein_marker,
            gene_label=gene_label,
            protein_vals=np.asarray(item["protein_vals"], dtype=np.float32),
            gene_vals=np.asarray(item["gene_vals"], dtype=np.float32),
            coords=coords,
            spatial_path=os.path.join(MARKER_PANEL_DIR, f"{base}_spatial.png"),
            scatter_path=os.path.join(MARKER_PANEL_DIR, f"{base}_scatter.png"),
        )

    print(f"[DONE] Figures generated from saved results in: {OUTDIR}")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-cell virtual-protein validation with separate evaluation "
            "and plotting workflows."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("eval", "plot", "both"),
        default="plot",
        help=(
            "eval: compute metrics and save numerical/plotting results only; "
            "plot: generate figures only from saved results; "
            "both: run evaluation, then plot from the saved results."
        ),
    )
    args = parser.parse_args()

    if args.mode in {"eval", "both"}:
        evaluate_and_save()

    if args.mode in {"plot", "both"}:
        plot_saved_results()


if __name__ == "__main__":
    main()

