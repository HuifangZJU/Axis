#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spatial gene–protein co-expression maps with clean eval / plot / both modes.

Modes
-----
eval
    Read the original RDS / h5ad / coordinate inputs, select gene–protein pairs,
    preprocess the sampled gene/protein signals, calculate robust co-expression
    states, and save plotting-ready numerical results. No figures are generated.

plot
    Read only the saved evaluation outputs and generate figures. The original
    RDS / h5ad files are not reopened and the biological state calculation is
    not repeated.

both
    Run eval first, then plot from the newly saved evaluation outputs.

Saved structure
---------------
OUTPUT_ROOT/
├── manifest/
│   ├── selected_gene_protein_pairs.csv
│   ├── selected_protein_markers.txt
│   └── evaluation_config.json
├── evaluation/
│   ├── sample_coordinates.npz
│   └── pairs/
│       └── <protein>__<gene>.npz
└── per_marker/
    └── <protein>/
        └── <protein>__<gene>__logFC...__FDR....png

Scientific semantics retained from the source script
----------------------------------------------------
- Xenium gene counts are normalized per cell to TARGET_SUM and log1p transformed.
- Predicted protein values are used as stored.
- Top gene–protein pairs are selected from the DE table.
- Robust 5th–99th percentile scaling defines gene/protein co-expression states.
- Three internal states are retained:
    protein-dominant, both-strong, gene-dominant.
  The figure displays these as Protein-dominant, Both high, and Gene-dominant.
- Optional contours are derived from smoothed local state fractions.

The plotting stage is intentionally layout-only: it receives precomputed
gene_fraction, total_signal, and state masks from eval mode.
"""

from __future__ import annotations
from axis.settings import data_path

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
# INPUT / OUTPUT CONFIG
# ============================================================

DATA_ROOT = data_path("phenocycler/haotian")

DE_RDS = "xenium_DE.rds"
GENE_H5AD = "adata_xenium_cell_level.h5ad"
PROTEIN_RDS = "xenium_protein_predict.rds"
XENIUM_LOCATION_CSV = "LUAD_2_A_xenium_axis.csv"

OUTPUT_ROOT = (
    data_path("paper_analysis/"
    "Figure7_modality/e_per_marker_ratio/per_marker/")
)


# ============================================================
# ANALYSIS CONFIG
# ============================================================

TARGET_PROTEINS: Optional[List[str]] = [
    "IDO1",
    "NKX2.1",
    # "CD68",
    # "CD20",
    # "E.Cadherin",
    # "CD8",
    # "a.SMA",
    # "CXCL13",
]

TOPK_GENES_PER_PROTEIN = 10
MAX_PAIRS = 0                     # 0 = no cap
TARGET_SUM = 1e4
PLOT_POINTS = 200_000             # sampled once during eval
RANDOM_SEED = 0

# Robust scaling used to calculate co-expression states.
PERCENTILE_LOW = 5.0
PERCENTILE_HIGH = 99.0

DOMINANCE_FRACTION_THRESHOLD = 0.75
DOMINANCE_MIN_SIGNAL = 0.25
BOTH_STRONG_MIN_SIGNAL = 0.5


# ============================================================
# VISUAL CONFIG
# ============================================================

INVERT_X = True
INVERT_Y = True

# PROTEIN_RGB = (0.10, 0.35, 0.95)
# GENE_RGB = (0.95, 0.25, 0.10)
# BOTH_RGB = (0.55, 0.05, 0.65)
PROTEIN_RGB = (0.20, 0.45, 0.75)
GENE_RGB    = (0.85, 0.30, 0.20)
BOTH_RGB    = (0.55, 0.30, 0.65)
BOTH_GAMMA = 1.6

ALPHA_MIN = 0.05
ALPHA_MAX = 0.95

PLOT_REGION_CONTOUR =False
CONTOUR_REGIONS = ("both_strong",)

CONTOUR_GRID_SIZE = 220
CONTOUR_GAUSSIAN_SIGMA = 2.5
CONTOUR_FRACTION_LEVEL = 0.45
CONTOUR_MIN_SUPPORT_FRAC = 0.025
CONTOUR_LINEWIDTH = 1.3
CONTOUR_LINESTYLE = (0, (4, 3))

CONTOUR_COLORS = {
    "protein_dominant": (0, 0, 0),
    "both_strong": (0, 0, 0),
    "gene_dominant": (0, 0, 0),
}

CONTOUR_LABELS = {
    "protein_dominant": "Protein-dominant region",
    "both_strong": "Both-high region",
    "gene_dominant": "Gene-dominant region",
}


@dataclass(frozen=True)
class PlotLayout:
    """
    Centralized controls for figure geometry.

    These values can be tuned without touching the evaluation code.
    Coordinates for title/legend/statistics are in axes-relative coordinates.
    """

    # Canvas / main map placement.
    figsize: Tuple[float, float] = (6.6, 6.2)
    dpi: int = 350
    axes_rect: Tuple[float, float, float, float] = (0.04, 0.04, 0.92, 0.88)

    # Scatter.
    point_size: float = 1.0

    # Title.
    title_fontsize: float = 16.0
    title_pad: float = 20.0
    title_loc: str = "left"
    title_x: float = 0.02
    title_ha: str = "left"

    # Main three-state legend.
    legend_fontsize: float = 12
    state_legend_x: float = 0.18
    state_legend_y: float = 1.03
    state_legend_ncol: int = 1
    state_legend_handletextpad: float = 0.40
    state_legend_columnspacing: float = 0.6

    # Optional contour legend.
    contour_legend_x: float = 0.9
    contour_legend_y: float = 1.03
    contour_legend_ncol: int = 3
    contour_legend_handlelength: float = 2.4

    # logFC / FDR audit annotation.
    stats_x: float = 0.995
    stats_y: float = 0.01
    stats_fontsize: float = 7.8

    # Export.
    bbox_inches: str = "tight"
    pad_inches: float = 0.05


PLOT_LAYOUT = PlotLayout()


# ============================================================
# OUTPUT PATHS
# ============================================================

def output_paths(output_root: str) -> Dict[str, str]:
    return {
        "manifest_dir": os.path.join(output_root, "manifest"),
        "evaluation_dir": os.path.join(output_root, "evaluation"),
        "pair_eval_dir": os.path.join(output_root, "evaluation", "pairs"),
        "figure_root": os.path.join(output_root, "per_marker"),
        "pair_manifest": os.path.join(
            output_root, "manifest", "selected_gene_protein_pairs.csv"
        ),
        "protein_manifest": os.path.join(
            output_root, "manifest", "selected_protein_markers.txt"
        ),
        "config_json": os.path.join(
            output_root, "manifest", "evaluation_config.json"
        ),
        "coordinates_npz": os.path.join(
            output_root, "evaluation", "sample_coordinates.npz"
        ),
    }


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
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
    if not result:
        raise ValueError(f"No objects found in RDS: {path}")

    obj = next(iter(result.values()))
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected an R data.frame in {path}, got {type(obj)}")
    return obj


def ensure_numeric_df(df: pd.DataFrame, fillna: float = 0.0) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(fillna)


def read_xenium_location_weird_csv(path: str) -> pd.DataFrame:
    """Read original coordinate CSV with a malformed first line."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path, header=None, skiprows=1)
    if df.shape[1] < 3:
        raise ValueError(
            f"Xenium location CSV has fewer than 3 columns: {df.shape}"
        )

    df = df.iloc[:, :3].copy()
    df.columns = ["row_id", "x", "y"]
    df[["x", "y"]] = ensure_numeric_df(df[["x", "y"]])
    return df


def orient_df_to_match_rows(
    df: pd.DataFrame,
    n_rows: int,
    name: str,
) -> pd.DataFrame:
    if df.shape[0] == n_rows:
        return df
    if df.shape[1] == n_rows:
        print(f"[TRANSPOSE] {name}: {df.shape} -> {df.T.shape}")
        return df.T
    raise ValueError(
        f"{name} shape {df.shape} does not match n_rows={n_rows} "
        "in either dimension."
    )


def build_canon_map(names: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for name in names:
        key = canonical_name(name)
        if key and key not in mapping:
            mapping[key] = str(name)
    return mapping


def resolve_target_proteins(
    requested_proteins: Optional[Sequence[str]],
    available_protein_map: Dict[str, str],
) -> List[str]:
    if not requested_proteins:
        return list(available_protein_map.values())

    resolved: List[str] = []
    missing: List[str] = []

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
            "None of the requested proteins were found in the protein matrix."
        )
    return resolved


# ============================================================
# GENE PREPROCESSING
# ============================================================

def compute_row_scale(X, target_sum: float) -> np.ndarray:
    """Per-cell library-size factor used for normalize_total."""
    if sp.issparse(X):
        row_sum = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    else:
        row_sum = np.asarray(X).sum(axis=1).astype(np.float32)

    scale = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum > 0
    scale[nonzero] = (target_sum / row_sum[nonzero]).astype(np.float32)
    return scale


def prepare_gene_matrix(
    adata: ad.AnnData,
    n_cells: int,
    target_sum: float,
):
    """
    Prepare the gene matrix for sampled normalize_total + log1p extraction.

    Returns
    -------
    X_dense_or_original, X_is_sparse, X_csc, scale, gene_names, gene_index
    """
    gene_names = list(adata.var_names.astype(str))
    X = adata.X
    X_is_sparse = sp.issparse(X)

    if X_is_sparse:
        X_csr = X.tocsr()
        X_csc = X.tocsc()
        n_rows, n_cols = X_csr.shape
    else:
        X = np.asarray(X)
        X_csr = None
        X_csc = None
        n_rows, n_cols = X.shape

    # Retain the source script's defensive transpose behavior.
    if n_rows != n_cells and n_cols == n_cells:
        print("[TRANSPOSE] gene matrix appears transposed.")
        if X_is_sparse:
            X_csr = X_csr.T.tocsr()
            X_csc = X_csc.T.tocsc()
            n_rows = X_csr.shape[0]
        else:
            X = X.T
            n_rows = X.shape[0]

    if n_rows != n_cells:
        raise ValueError(
            f"Gene matrix rows do not match coordinates: {n_rows} vs {n_cells}"
        )

    scale = compute_row_scale(X_csr if X_is_sparse else X, target_sum)
    gene_index = {gene: idx for idx, gene in enumerate(gene_names)}

    return X, X_is_sparse, X_csc, scale, gene_names, gene_index


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
    Extract one normalized/log1p gene vector on the saved cell sample without
    materializing the full normalized matrix.
    """
    if not X_is_sparse:
        raw = X[sample_idx, gene_idx].astype(np.float32, copy=False)
        return np.log1p(raw * scale[sample_idx]).astype(
            np.float32, copy=False
        )

    out = np.zeros(sample_idx.size, dtype=np.float32)
    col = X_csc.getcol(gene_idx).tocoo()
    sample_positions = pos_map[col.row]
    keep = sample_positions >= 0

    if np.any(keep):
        out[sample_positions[keep]] = (
            col.data[keep].astype(np.float32, copy=False)
            * scale[col.row[keep]]
        )

    return np.log1p(out).astype(np.float32, copy=False)


# ============================================================
# PAIR SELECTION
# ============================================================

def topk_pairs_per_marker(de_df: pd.DataFrame, k: int) -> pd.DataFrame:
    out = de_df.copy()

    for column in ["logFC", "AveExpr", "adj.P.Val"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = (
        out.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["marker", "gene", "logFC", "adj.P.Val"])
        .copy()
    )
    out["abs_logFC"] = out["logFC"].abs()

    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()

    out = out.sort_values(
        ["marker", "adj.P.Val", "abs_logFC"],
        ascending=[True, True, False],
    )

    return (
        out.groupby("marker", group_keys=False)
        .head(k)
        .copy()
    )


# ============================================================
# CO-EXPRESSION STATE CALCULATION -- EVAL ONLY
# ============================================================

def robust_minmax(
    values: np.ndarray,
    p_low: float = PERCENTILE_LOW,
    p_high: float = PERCENTILE_HIGH,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return 0.0, 1.0

    low = float(np.percentile(finite, p_low))
    high = float(np.percentile(finite, p_high))

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 0.0, 1.0

    return low, high


def scale_01(
    values: np.ndarray,
    p_low: float = PERCENTILE_LOW,
    p_high: float = PERCENTILE_HIGH,
) -> np.ndarray:
    low, high = robust_minmax(values, p_low=p_low, p_high=p_high)
    scaled = (np.asarray(values, dtype=float) - low) / (high - low + 1e-12)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def compute_coexpression_state(
    gene_values: np.ndarray,
    protein_values: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Calculate the scientific state representation once in eval mode.

    The plotting stage later consumes these arrays directly.
    """
    gene_01 = scale_01(gene_values)
    protein_01 = scale_01(protein_values)

    denominator = gene_01 + protein_01 + 1e-8
    gene_fraction = gene_01 / denominator
    total_01 = np.clip((gene_01 + protein_01) / 2.0, 0.0, 1.0)

    gene_dominant = (
        (gene_fraction >= DOMINANCE_FRACTION_THRESHOLD)
        & (gene_01 >= DOMINANCE_MIN_SIGNAL)
    )
    protein_dominant = (
        (gene_fraction <= (1.0 - DOMINANCE_FRACTION_THRESHOLD))
        & (protein_01 >= DOMINANCE_MIN_SIGNAL)
    )
    both_strong = (
        (gene_01 >= BOTH_STRONG_MIN_SIGNAL)
        & (protein_01 >= BOTH_STRONG_MIN_SIGNAL)
    )

    return {
        "gene_fraction": gene_fraction.astype(np.float32),
        "total_01": total_01.astype(np.float32),
        "gene_dominant": gene_dominant.astype(bool),
        "protein_dominant": protein_dominant.astype(bool),
        "both_strong": both_strong.astype(bool),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_and_save(
    data_root: str,
    output_root: str,
    topk: int,
    target_sum: float,
    max_pairs: int,
    plot_points: int,
    target_proteins: Optional[Sequence[str]],
) -> None:
    paths = output_paths(output_root)
    for key in ["manifest_dir", "evaluation_dir", "pair_eval_dir"]:
        ensure_dir(paths[key])

    # --------------------------------------------------------
    # Coordinates + one reproducible shared sample
    # --------------------------------------------------------
    coord_path = os.path.join(data_root, XENIUM_LOCATION_CSV)
    print("[LOAD] coordinates:", coord_path)
    coord_df = read_xenium_location_weird_csv(coord_path)

    n_cells = coord_df.shape[0]
    x_all = coord_df["y"].to_numpy(dtype=np.float32)
    y_all = coord_df["x"].to_numpy(dtype=np.float32)

    rng = np.random.default_rng(RANDOM_SEED)
    if plot_points and n_cells > plot_points:
        sample_idx = rng.choice(
            n_cells,
            size=plot_points,
            replace=False,
        ).astype(np.int32)
    else:
        sample_idx = np.arange(n_cells, dtype=np.int32)

    pos_map = np.full(n_cells, -1, dtype=np.int32)
    pos_map[sample_idx] = np.arange(sample_idx.size, dtype=np.int32)

    np.savez_compressed(
        paths["coordinates_npz"],
        x=x_all[sample_idx],
        y=y_all[sample_idx],
        sample_idx=sample_idx,
    )
    print("[SAVE]", paths["coordinates_npz"])

    # --------------------------------------------------------
    # Protein matrix
    # --------------------------------------------------------
    protein_path = os.path.join(data_root, PROTEIN_RDS)
    print("[LOAD] predicted protein:", protein_path)
    protein_df = ensure_numeric_df(read_rds_df(protein_path))
    protein_df = orient_df_to_match_rows(
        protein_df, n_cells, "predicted_protein"
    )
    protein_df = ensure_numeric_df(protein_df)

    protein_map = build_canon_map(protein_df.columns)
    selected_proteins = resolve_target_proteins(
        target_proteins,
        protein_map,
    )
    selected_protein_keys = {
        canonical_name(protein) for protein in selected_proteins
    }
    print("[INFO] Selected protein markers:", ", ".join(selected_proteins))

    with open(
        paths["protein_manifest"], "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(selected_proteins) + "\n")

    # --------------------------------------------------------
    # Gene matrix
    # --------------------------------------------------------
    gene_path = os.path.join(data_root, GENE_H5AD)
    print("[LOAD] gene h5ad:", gene_path)
    adata = ad.read_h5ad(gene_path)

    (
        X,
        X_is_sparse,
        X_csc,
        scale,
        gene_names,
        gene_index,
    ) = prepare_gene_matrix(
        adata=adata,
        n_cells=n_cells,
        target_sum=target_sum,
    )
    gene_map = build_canon_map(gene_names)

    # --------------------------------------------------------
    # DE table + pair selection
    # --------------------------------------------------------
    de_path = os.path.join(data_root, DE_RDS)
    print("[LOAD] DE:", de_path)
    de_df = read_rds_df(de_path)

    required = {"logFC", "AveExpr", "adj.P.Val", "gene", "marker"}
    missing = required - set(de_df.columns)
    if missing:
        raise ValueError(f"DE table missing columns: {missing}")

    de_df = de_df.copy()
    de_df["marker_c"] = de_df["marker"].map(canonical_name)
    de_df["gene_c"] = de_df["gene"].map(canonical_name)

    de_df = de_df[
        de_df["marker_c"].isin(selected_protein_keys)
        & de_df["gene_c"].isin(gene_map)
    ].copy()

    if de_df.empty:
        raise ValueError(
            "No DE rows remain after filtering to requested proteins "
            "and available genes."
        )

    de_df["marker"] = de_df["marker_c"].map(protein_map)
    de_df["gene"] = de_df["gene_c"].map(gene_map)
    de_df = de_df.drop(
        columns=["marker_c", "gene_c"],
        errors="ignore",
    )

    selected_pairs = topk_pairs_per_marker(de_df, k=topk)

    if max_pairs and selected_pairs.shape[0] > max_pairs:
        selected_pairs = (
            selected_pairs.sort_values(
                ["adj.P.Val", "abs_logFC"],
                ascending=[True, False],
            )
            .head(max_pairs)
            .copy()
        )

    # --------------------------------------------------------
    # Precompute each pair's plotting-ready state
    # --------------------------------------------------------
    gene_cache: Dict[str, np.ndarray] = {}
    protein_cache: Dict[str, np.ndarray] = {}
    payload_files: List[str] = []
    n_valid_cells: List[int] = []

    x_plot = x_all[sample_idx]
    y_plot = y_all[sample_idx]

    for pair_number, (_, row) in enumerate(
        selected_pairs.iterrows(),
        start=1,
    ):
        protein_name = str(row["marker"])
        gene_name = str(row["gene"])

        if protein_name not in protein_cache:
            protein_cache[protein_name] = (
                protein_df[protein_name]
                .to_numpy(dtype=np.float32, copy=False)[sample_idx]
            )

        if gene_name not in gene_cache:
            gene_cache[gene_name] = extract_gene_log1p_on_sample(
                X=X,
                X_is_sparse=X_is_sparse,
                X_csc=X_csc,
                scale=scale,
                gene_idx=gene_index[gene_name],
                sample_idx=sample_idx,
                pos_map=pos_map,
            )

        protein_values = protein_cache[protein_name]
        gene_values = gene_cache[gene_name]

        valid = (
            np.isfinite(x_plot)
            & np.isfinite(y_plot)
            & np.isfinite(gene_values)
            & np.isfinite(protein_values)
        )
        valid_idx = np.where(valid)[0].astype(np.int32)

        state = compute_coexpression_state(
            gene_values=gene_values[valid],
            protein_values=protein_values[valid],
        )

        payload_name = (
            f"{safe_name(protein_name)}__{safe_name(gene_name)}.npz"
        )
        payload_path = os.path.join(paths["pair_eval_dir"], payload_name)

        np.savez_compressed(
            payload_path,
            valid_idx=valid_idx,
            gene_fraction=state["gene_fraction"],
            total_01=state["total_01"],
            gene_dominant=state["gene_dominant"],
            protein_dominant=state["protein_dominant"],
            both_strong=state["both_strong"],
        )

        payload_rel = os.path.relpath(payload_path, output_root)
        payload_files.append(payload_rel)
        n_valid_cells.append(int(valid_idx.size))

        print(
            f"[EVAL] {pair_number}/{len(selected_pairs)} "
            f"{protein_name} / {gene_name}: {valid_idx.size:,} cells"
        )

    selected_pairs = selected_pairs.reset_index(drop=True)
    selected_pairs["payload_file"] = payload_files
    selected_pairs["n_valid_cells"] = n_valid_cells
    selected_pairs.to_csv(paths["pair_manifest"], index=False)
    print("[SAVE]", paths["pair_manifest"])

    config = {
        "data_root": data_root,
        "target_sum": float(target_sum),
        "topk_genes_per_protein": int(topk),
        "max_pairs": int(max_pairs),
        "plot_points": int(plot_points),
        "random_seed": int(RANDOM_SEED),
        "percentile_low": float(PERCENTILE_LOW),
        "percentile_high": float(PERCENTILE_HIGH),
        "dominance_fraction_threshold": float(
            DOMINANCE_FRACTION_THRESHOLD
        ),
        "dominance_min_signal": float(DOMINANCE_MIN_SIGNAL),
        "both_strong_min_signal": float(BOTH_STRONG_MIN_SIGNAL),
        "selected_proteins": selected_proteins,
    }
    with open(paths["config_json"], "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print("[SAVE]", paths["config_json"])

    print(
        f"[DONE] Evaluation saved for {len(selected_pairs)} pairs "
        f"under: {paths['evaluation_dir']}"
    )


# ============================================================
# CONTOUR HELPERS -- PLOT ONLY
# ============================================================

def normalize_contour_regions(
    regions: Sequence[str],
) -> Tuple[str, ...]:
    valid = {
        "gene_dominant",
        "protein_dominant",
        "both_strong",
    }

    regions = tuple(str(region).strip().lower() for region in regions)

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
    x: np.ndarray,
    y: np.ndarray,
    region_mask: np.ndarray,
    grid_size: int = CONTOUR_GRID_SIZE,
    sigma: float = CONTOUR_GAUSSIAN_SIGMA,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    region_mask = np.asarray(region_mask, dtype=bool)

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    dx = max(float(xmax - xmin), 1e-6)
    dy = max(float(ymax - ymin), 1e-6)

    x_edges = np.linspace(
        xmin - 0.01 * dx,
        xmax + 0.01 * dx,
        grid_size + 1,
    )
    y_edges = np.linspace(
        ymin - 0.01 * dy,
        ymax + 0.01 * dy,
        grid_size + 1,
    )

    all_hist, _, _ = np.histogram2d(
        x, y, bins=[x_edges, y_edges]
    )
    selected_hist, _, _ = np.histogram2d(
        x[region_mask],
        y[region_mask],
        bins=[x_edges, y_edges],
    )

    all_smooth = gaussian_filter(all_hist, sigma=sigma)
    selected_smooth = gaussian_filter(selected_hist, sigma=sigma)

    fraction = selected_smooth / (all_smooth + 1e-12)
    support_threshold = (
        CONTOUR_MIN_SUPPORT_FRAC * np.nanmax(all_smooth)
    )
    fraction[all_smooth < support_threshold] = np.nan

    return (
        (x_edges[:-1] + x_edges[1:]) / 2.0,
        (y_edges[:-1] + y_edges[1:]) / 2.0,
        fraction,
    )


def draw_region_contours(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    state_masks: Dict[str, np.ndarray],
    requested_regions: Sequence[str],
) -> List[Line2D]:
    contour_handles: List[Line2D] = []

    for region in normalize_contour_regions(requested_regions):
        region_mask = state_masks[region]

        if int(region_mask.sum()) < 10:
            print(f"[WARN] Too few points for {region} contour; skipped.")
            continue

        x_centers, y_centers, fraction = build_smoothed_region_fraction(
            x=x,
            y=y,
            region_mask=region_mask,
        )

        finite = fraction[np.isfinite(fraction)]
        if (
            finite.size == 0
            or np.nanmax(finite) < CONTOUR_FRACTION_LEVEL
        ):
            print(
                f"[WARN] {region} never reaches contour level "
                f"{CONTOUR_FRACTION_LEVEL:.2f}; skipped."
            )
            continue

        xx, yy = np.meshgrid(x_centers, y_centers)
        color = CONTOUR_COLORS[region]

        ax.contour(
            xx,
            yy,
            fraction.T,
            levels=[CONTOUR_FRACTION_LEVEL],
            colors=[color],
            linewidths=CONTOUR_LINEWIDTH,
            linestyles=[CONTOUR_LINESTYLE],
            zorder=5,
        )

        contour_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=CONTOUR_LINEWIDTH,
                linestyle=CONTOUR_LINESTYLE,
                label=CONTOUR_LABELS[region],
            )
        )

    return contour_handles


# ============================================================
# PURE PLOTTING HELPERS
# ============================================================

def set_publication_style(layout: PlotLayout = PLOT_LAYOUT) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": layout.title_fontsize,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def coexpression_rgba(
    gene_fraction: np.ndarray,
    total_01: np.ndarray,
) -> np.ndarray:
    """
    Convert precomputed state coordinates into display RGBA values.
    This is a visual mapping only; no biological state is recalculated.
    """
    gene_fraction = np.asarray(gene_fraction, dtype=np.float32)
    total_01 = np.asarray(total_01, dtype=np.float32)

    protein_rgb = np.asarray(PROTEIN_RGB, dtype=float)
    gene_rgb = np.asarray(GENE_RGB, dtype=float)
    both_rgb = np.asarray(BOTH_RGB, dtype=float)

    base_rgb = (
        (1.0 - gene_fraction)[:, None] * protein_rgb[None, :]
        + gene_fraction[:, None] * gene_rgb[None, :]
    )

    balance = 1.0 - np.clip(
        np.abs(gene_fraction - 0.5) / 0.5,
        0.0,
        1.0,
    )
    both_weight = (balance ** BOTH_GAMMA) * total_01

    rgb = (
        (1.0 - both_weight)[:, None] * base_rgb
        + both_weight[:, None] * both_rgb[None, :]
    )
    alpha = (
        ALPHA_MIN
        + total_01 * (ALPHA_MAX - ALPHA_MIN)
    )

    return np.column_stack([rgb, alpha])


def plot_coexpression_map(
    x: np.ndarray,
    y: np.ndarray,
    gene_fraction: np.ndarray,
    total_01: np.ndarray,
    state_masks: Dict[str, np.ndarray],
    gene_name: str,
    protein_name: str,
    output_png: str,
    *,
    logfc: Optional[float] = None,
    fdr: Optional[float] = None,
    layout: PlotLayout = PLOT_LAYOUT,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Render one saved co-expression result.

    Layout tuning is centralized in PlotLayout above. This function does not
    calculate normalization, robust scaling, or state assignments.
    """
    set_publication_style(layout)

    x = np.asarray(x)
    y = np.asarray(y)
    rgba = coexpression_rgba(gene_fraction, total_01)

    # The evaluation payload is already sampled. This optional safety cap is
    # retained only for display scalability if the saved payload is very large.
    scatter_idx = np.arange(x.size)
    if PLOT_POINTS and x.size > PLOT_POINTS:
        rng = np.random.default_rng(seed)
        scatter_idx = rng.choice(
            x.size,
            size=PLOT_POINTS,
            replace=False,
        )

    fig = plt.figure(
        figsize=layout.figsize,
        dpi=layout.dpi,
        facecolor="white",
    )
    ax = fig.add_axes(layout.axes_rect)

    ax.scatter(
        x[scatter_idx],
        y[scatter_idx],
        c=rgba[scatter_idx],
        s=layout.point_size,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    contour_handles: List[Line2D] = []
    if PLOT_REGION_CONTOUR:
        contour_handles = draw_region_contours(
            ax=ax,
            x=x,
            y=y,
            state_masks=state_masks,
            requested_regions=CONTOUR_REGIONS,
        )

    ax.set_aspect("equal", adjustable="box")

    if INVERT_X:
        ax.invert_xaxis()
    if INVERT_Y:
        ax.invert_yaxis()

    ax.axis("off")

    from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

    gene_text = TextArea(
        gene_name,
        textprops=dict(
            fontsize=layout.title_fontsize,
            fontweight="bold",
            fontstyle="italic",
        ),
    )

    gene_label = TextArea(
        " gene | ",
        textprops=dict(
            fontsize=layout.title_fontsize,
            fontweight="normal",
            fontstyle="normal",
        ),
    )

    protein_text = TextArea(
        protein_name,
        textprops=dict(
            fontsize=layout.title_fontsize,
            fontweight="bold",
            fontstyle="normal",
        ),
    )

    protein_label = TextArea(
        " protein",
        textprops=dict(
            fontsize=layout.title_fontsize,
            fontweight="normal",
            fontstyle="normal",
        ),
    )

    title_box = HPacker(
        children=[gene_text, gene_label, protein_text, protein_label],
        align="center",
        pad=0,
        sep=0,
    )

    # Convert title alignment to AnnotationBbox alignment.
    box_alignment = {
        "left": (0.0, 0.0),
        "center": (0.5, 0.0),
        "right": (1.0, 0.0),
    }.get(layout.title_ha, (0.5, 0.0))

    title_artist = AnnotationBbox(
        title_box,
        (layout.title_x, 1.0),
        xycoords=ax.transAxes,
        xybox=(0, layout.title_pad),
        boxcoords="offset points",
        box_alignment=box_alignment,
        frameon=False,
    )

    ax.add_artist(title_artist)
    state_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            label="Protein-dominant",
            markerfacecolor=PROTEIN_RGB,
            markeredgecolor="none",
            markersize=6.5,
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            label="Both high",
            markerfacecolor=BOTH_RGB,
            markeredgecolor="none",
            markersize=6.5,
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            label="Gene-dominant",
            markerfacecolor=GENE_RGB,
            markeredgecolor="none",
            markersize=6.5,
        ),
    ]

    legend1 = ax.legend(
        handles=state_handles,
        loc="upper center",
        bbox_to_anchor=(
            layout.state_legend_x,
            layout.state_legend_y,
        ),
        ncol=layout.state_legend_ncol,
        frameon=False,
        fontsize=layout.legend_fontsize,
        handletextpad=layout.state_legend_handletextpad,
        columnspacing=layout.state_legend_columnspacing,
        borderaxespad=0.0,
    )
    ax.add_artist(legend1)

    if contour_handles:
        ax.legend(
            handles=contour_handles,
            loc="upper right",
            bbox_to_anchor=(
                layout.contour_legend_x,
                layout.contour_legend_y,
            ),
            ncol=min(
                layout.contour_legend_ncol,
                len(contour_handles),
            ),
            frameon=False,
            fontsize=layout.legend_fontsize - 0.5,
            handlelength=layout.contour_legend_handlelength,
            handletextpad=0.5,
            columnspacing=1.0,
            borderaxespad=0.0,
        )

    # if (
    #     logfc is not None
    #     and fdr is not None
    #     and np.isfinite(logfc)
    #     and np.isfinite(fdr)
    # ):
    #     ax.text(
    #         layout.stats_x,
    #         layout.stats_y,
    #         rf"$\log FC$={logfc:+.2f}" + "\n" + rf"FDR={fdr:.1e}",
    #         transform=ax.transAxes,
    #         ha="right",
    #         va="bottom",
    #         fontsize=layout.stats_fontsize,
    #         color="0.35",
    #     )

    ensure_dir(os.path.dirname(os.path.abspath(output_png)))
    fig.savefig(
        output_png,
        dpi=layout.dpi,
        bbox_inches=layout.bbox_inches,
        pad_inches=layout.pad_inches,
        facecolor="white",
    )
    plt.close(fig)


# ============================================================
# PLOT FROM SAVED EVALUATION
# ============================================================

def plot_saved_results(
    output_root: str,
    layout: PlotLayout = PLOT_LAYOUT,
) -> None:
    paths = output_paths(output_root)

    required = [
        paths["pair_manifest"],
        paths["coordinates_npz"],
        paths["config_json"],
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Missing saved evaluation results. Run --mode eval first:\n  "
            + "\n  ".join(missing)
        )

    ensure_dir(paths["figure_root"])

    selected_pairs = pd.read_csv(paths["pair_manifest"])
    with np.load(paths["coordinates_npz"], allow_pickle=False) as coord_data:
        x_sample = np.asarray(coord_data["x"], dtype=np.float32)
        y_sample = np.asarray(coord_data["y"], dtype=np.float32)

    for pair_number, (_, row) in enumerate(
        selected_pairs.iterrows(),
        start=1,
    ):
        protein_name = str(row["marker"])
        gene_name = str(row["gene"])
        logfc = float(row["logFC"])
        fdr = float(row["adj.P.Val"])

        payload_path = os.path.join(
            output_root,
            str(row["payload_file"]),
        )
        if not os.path.isfile(payload_path):
            raise FileNotFoundError(payload_path)

        with np.load(payload_path, allow_pickle=False) as payload:
            valid_idx = np.asarray(payload["valid_idx"], dtype=np.int32)
            gene_fraction = np.asarray(
                payload["gene_fraction"], dtype=np.float32
            )
            total_01 = np.asarray(payload["total_01"], dtype=np.float32)
            state_masks = {
                "gene_dominant": np.asarray(
                    payload["gene_dominant"], dtype=bool
                ),
                "protein_dominant": np.asarray(
                    payload["protein_dominant"], dtype=bool
                ),
                "both_strong": np.asarray(
                    payload["both_strong"], dtype=bool
                ),
            }

        x = x_sample[valid_idx]
        y = y_sample[valid_idx]

        marker_dir = os.path.join(
            paths["figure_root"],
            safe_name(protein_name),
        )
        ensure_dir(marker_dir)

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
            x=x,
            y=y,
            gene_fraction=gene_fraction,
            total_01=total_01,
            state_masks=state_masks,
            gene_name=gene_name,
            protein_name=protein_name,
            output_png=output_png,
            logfc=logfc,
            fdr=fdr,
            layout=layout,
            seed=RANDOM_SEED,
        )

        print(
            f"[PLOT] {pair_number}/{len(selected_pairs)}: {output_png}"
        )

    print("[DONE] figures:", paths["figure_root"])


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and/or plot gene-protein spatial co-expression maps."
        )
    )

    parser.add_argument(
        "--mode",
        choices=("eval", "plot", "both"),
        default="plot",
        help=(
            "eval: calculate/save plotting-ready numerical results only; "
            "plot: plot only from saved results; "
            "both: eval then plot."
        ),
    )
    parser.add_argument(
        "--data_root",
        default=DATA_ROOT,
        help="Directory containing input RDS/h5ad/coordinate files.",
    )
    parser.add_argument(
        "--output_root",
        default=OUTPUT_ROOT,
        help="Directory for evaluation results, manifests, and figures.",
    )
    parser.add_argument(
        "--proteins",
        nargs="*",
        default=TARGET_PROTEINS,
        help=(
            "Protein markers to evaluate. If the configured list is empty/None, "
            "all available protein markers are used."
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
        help=(
            "Number of cells sampled and saved during eval. "
            "The same sample is reused for all pairs."
        ),
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=PLOT_LAYOUT.point_size,
        help="Scatter point size used only during plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Easy command-line point-size tuning without changing the saved evaluation.
    layout = PlotLayout(
        **{
            **asdict(PLOT_LAYOUT),
            "point_size": float(args.point_size),
        }
    )

    if args.mode in {"eval", "both"}:
        evaluate_and_save(
            data_root=args.data_root,
            output_root=args.output_root,
            topk=args.topk,
            target_sum=args.target_sum,
            max_pairs=args.max_pairs,
            plot_points=args.plot_points,
            target_proteins=args.proteins,
        )

    if args.mode in {"plot", "both"}:
        plot_saved_results(
            output_root=args.output_root,
            layout=layout,
        )


if __name__ == "__main__":
    main()

