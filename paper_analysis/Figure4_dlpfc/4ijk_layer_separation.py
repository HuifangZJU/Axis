#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLPFC layer-separation analysis across native sections, pooled observations,
PASTE, and an integrated AXIS atlas.

This streamlined version intentionally avoids scanpy/anndata during evaluation.
Only the H5AD components required by this analysis are read directly with h5py:
X, var index, obs[layer_key], and obsm[spatial_key]. This avoids failures in
anndata's `_clean_uns` when an H5AD contains categorical metadata that is
incompatible with the active pandas/anndata environment, while retaining all
spatial-map content required by the original main figure.

Primary metrics
---------------
1. Kruskal-Wallis rank-based layer-separation effect size:

       eta_H^2 = max(0, (H - k + 1) / (n - k))

2. Mean adjacent-layer AUROC. For each neighboring layer pair, AUROC is made
   direction invariant as max(AUC, 1-AUC), then averaged across available pairs.

Run modes
---------
    RUN_MODE = "eval"   compute and save numerical outputs
    RUN_MODE = "plot"   regenerate figures from saved CSV files
    RUN_MODE = "both"   evaluate, save, then plot
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from scipy.stats import kruskal
from sklearn.metrics import roc_auc_score


# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_MODE = "plot"  # run once to regenerate CSVs; later switch back to "plot"


@dataclass(frozen=True)
class Config:
    run_mode: str = RUN_MODE

    group_id: int = 1
    h5ad_dir: str = "/media/huifang/data/registration/DLPFC/huifang/h5ad"
    layer_key: str = "layer_guess_reordered"
    spatial_key: str = "spatial"
    sample_ids: Tuple[int, ...] = (0, 1, 2, 3)

    paste_h5ad: str = (
        "/media/huifang/data/registration/DLPFC/paste_processed/saved_results/"
        "center1_a0.1_KL_seed2.h5ad"
    )
    axis_h5ad: str = (
        "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/"
        "dlpfc_z_group1_grid80_spatial1/integration/"
        "integrated_dlpfc_z_ref_0_visium_ot.h5ad"
    )
    axis_aggregation_label: str = "median"  # metadata/display only

    marker_genes: Tuple[str, ...] = (
        "MFGE8", "MOBP", "CUX2", "RAI14", "RORB", "ENC1",
        "PCP4", "KCNIP2", "SYT6", "FOXP2", "MBP",
    )
    genes_to_plot: Tuple[str, ...] = (
        "CUX2", "RORB", "PCP4", "KCNIP2", "FOXP2",
        "MBP", "MOBP", "ENC1", "MFGE8",
    )
    main_figure_genes: Tuple[str, ...] = ("PCP4", "MOBP")
    preferred_layer_order: Tuple[str, ...] = (
        "Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"
    )

    # Expression is transformed exactly once, immediately after H5AD loading.
    # All metrics and plots reuse the same log1p-transformed values.
    spatial_cmap: str = "Reds"
    spatial_point_size: float = 4.0
    invert_spatial_y: bool = True

    outdir: str = (
        "/media/huifang/data/registration/paper_analysis/"
        "Figure4_dlfpc/hij_layer_separation/"
    )
    save_formats: Tuple[str, ...] = ("png",)
    png_dpi: int = 450
    plot_main_composite: bool = True
    plot_all_gene_summary: bool = True
    plot_supplementary_violins: bool = False
    annotate_violin_metrics: bool = True

    @property
    def outdir_path(self) -> Path:
        return Path(self.outdir)

    @property
    def real_panels(self) -> Tuple[str, ...]:
        return tuple(f"Single {sid}" for sid in sorted(self.sample_ids))

    @property
    def panel_order(self) -> Tuple[str, ...]:
        return self.real_panels + ("Pool", "PASTE", "Axis")

    @property
    def long_csv(self) -> Path:
        return self.outdir_path / "marker_expression_long.csv"

    @property
    def layer_summary_csv(self) -> Path:
        return self.outdir_path / "marker_layer_summary.csv"

    @property
    def separation_csv(self) -> Path:
        return self.outdir_path / "marker_layer_separation_metrics.csv"

    @property
    def adjacent_auc_csv(self) -> Path:
        return self.outdir_path / "marker_adjacent_layer_auroc.csv"

    @property
    def method_summary_csv(self) -> Path:
        return self.outdir_path / "marker_layer_separation_method_summary.csv"


CFG = Config()


# =============================================================================
# LIGHTWEIGHT H5AD READER
# =============================================================================
def get_panel_color(panel: str) -> str:
    colors = {
        "Pool": "#E69F00",
        "PASTE": "#356E9B",
        "Axis": "#9E2F2A",
    }
    return "#8F8F8F" if str(panel).startswith("Single") else colors.get(panel, "#6F6F6F")


def display_panel_name(panel: str) -> str:
    panel = str(panel)

    if panel.startswith("Single"):
        return "Single section"

    return {
        "Pool": "Pooled",
        "PASTE": "PASTE",
        "Axis": "AXIS",
    }.get(panel, panel)



@dataclass
class MarkerData:
    """Expression, layers, and coordinates required by the evaluation/figures."""

    expression_log1p: np.ndarray
    genes: Tuple[str, ...]
    layers: np.ndarray
    coords: np.ndarray

    @property
    def n_obs(self) -> int:
        return int(self.expression_log1p.shape[0])


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _read_string_dataset(dataset: h5py.Dataset) -> np.ndarray:
    values = dataset[...]
    values = np.asarray(values)
    if values.ndim == 0:
        values = values.reshape(1)
    out = np.empty(values.shape, dtype=object)
    flat_out = out.reshape(-1)
    for i, value in enumerate(values.reshape(-1)):
        if value is None:
            flat_out[i] = np.nan
        else:
            flat_out[i] = _decode_scalar(value)
    return out


def _read_encoded_vector(node: Any) -> np.ndarray:
    """Read a common H5AD dataframe column without invoking anndata.

    Supports plain datasets and the categorical encoding used by AnnData/H5AD.
    Missing categorical values (code < 0) are returned as np.nan.
    """
    if isinstance(node, h5py.Dataset):
        return _read_string_dataset(node)

    if not isinstance(node, h5py.Group):
        raise TypeError(f"Unsupported H5 node type: {type(node)}")

    if "codes" in node and "categories" in node:
        codes = np.asarray(node["codes"][...], dtype=np.int64)
        categories = _read_encoded_vector(node["categories"]).reshape(-1)
        out = np.empty(codes.shape, dtype=object)
        out[:] = np.nan
        valid = (codes >= 0) & (codes < len(categories))
        out[valid] = categories[codes[valid]]
        return out

    for key in ("values", "data"):
        if key in node and isinstance(node[key], h5py.Dataset):
            return _read_string_dataset(node[key])

    encoding = node.attrs.get("encoding-type", "unknown")
    raise ValueError(
        f"Unsupported H5AD vector encoding {encoding!r}; keys={list(node.keys())}"
    )


def _h5ad_var_names(handle: h5py.File) -> np.ndarray:
    if "var" not in handle:
        raise KeyError("H5AD is missing /var")
    var = handle["var"]
    index_key = var.attrs.get("_index", "_index")
    if isinstance(index_key, bytes):
        index_key = index_key.decode("utf-8")
    if index_key not in var:
        # Older/simple H5AD files often use `_index` directly.
        if "_index" in var:
            index_key = "_index"
        else:
            raise KeyError(
                f"Unable to locate var index in H5AD; var keys={list(var.keys())}"
            )
    return _read_encoded_vector(var[index_key]).astype(str).reshape(-1)


def _read_h5_matrix(node: Any) -> Any:
    """Read a dense or CSR/CSC H5AD matrix."""
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[...])

    if not isinstance(node, h5py.Group):
        raise TypeError(f"Unsupported matrix node type: {type(node)}")

    encoding = node.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")

    if {"data", "indices", "indptr"}.issubset(node.keys()):
        data = np.asarray(node["data"][...])
        indices = np.asarray(node["indices"][...], dtype=np.int64)
        indptr = np.asarray(node["indptr"][...], dtype=np.int64)
        shape = tuple(int(x) for x in np.asarray(node.attrs["shape"]).tolist())
        if "csc" in str(encoding).lower():
            return sp.csc_matrix((data, indices, indptr), shape=shape)
        return sp.csr_matrix((data, indices, indptr), shape=shape)

    raise ValueError(
        f"Unsupported H5AD matrix encoding {encoding!r}; keys={list(node.keys())}"
    )


def _read_selected_X(handle: h5py.File, column_indices: Sequence[int]) -> np.ndarray:
    if "X" not in handle:
        raise KeyError("H5AD is missing /X")
    node = handle["X"]
    column_indices = np.asarray(column_indices, dtype=np.int64)

    if isinstance(node, h5py.Dataset):
        # h5py requires monotonically increasing fancy indices.
        order = np.argsort(column_indices)
        sorted_columns = column_indices[order]
        block = np.asarray(node[:, sorted_columns], dtype=np.float32)
        inverse = np.argsort(order)
        return block[:, inverse]

    matrix = _read_h5_matrix(node)
    block = matrix[:, column_indices]
    if sp.issparse(block):
        block = block.toarray()
    return np.asarray(block, dtype=np.float32)


def log1p_expression_once(values: np.ndarray) -> np.ndarray:
    """Apply the only log1p transform used by this script.

    H5AD ``X`` is treated as non-negative raw/count-like expression. This
    transform runs once per loaded H5AD, and downstream code reuses the result.
    """
    values = np.asarray(values, dtype=np.float32)
    if np.any(~np.isfinite(values)):
        raise ValueError("Expression matrix contains non-finite values before log1p.")
    if np.any(values < -1e-6):
        warnings.warn(
            "Expression matrix contains negative values; clipping them to 0 before log1p."
        )
    return np.log1p(np.clip(values, 0.0, None)).astype(np.float32)


def _read_obsm_matrix(
    handle: h5py.File,
    spatial_key: str,
) -> np.ndarray:
    """Read one obsm matrix directly from H5AD without touching /uns."""
    if "obsm" not in handle or spatial_key not in handle["obsm"]:
        keys = list(handle["obsm"].keys()) if "obsm" in handle else []
        raise KeyError(
            f"obsm[{spatial_key!r}] is missing; available obsm keys={keys}"
        )
    node = handle["obsm"][spatial_key]
    matrix = _read_h5_matrix(node)
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError(
            f"obsm[{spatial_key!r}] must have shape (n_obs, >=2); "
            f"received {matrix.shape}."
        )
    return matrix[:, :2]


def read_marker_h5ad(
    path: Path,
    requested_genes: Sequence[str],
    layer_key: str,
    spatial_key: str,
) -> MarkerData:
    """Read marker expression and layer labels directly from an H5AD file.

    The function deliberately ignores /uns and all unrelated metadata. This is
    the key fix for the reported anndata `_clean_uns` categorical runtime error.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as handle:
        var_names = _h5ad_var_names(handle)
        gene_to_index = {gene: i for i, gene in enumerate(var_names)}
        present = [str(g) for g in requested_genes if str(g) in gene_to_index]
        if not present:
            raise ValueError(f"None of the requested marker genes were found in {path}")

        if "obs" not in handle or layer_key not in handle["obs"]:
            keys = list(handle["obs"].keys()) if "obs" in handle else []
            raise KeyError(
                f"obs[{layer_key!r}] is missing from {path}; obs keys={keys}"
            )
        layers = _read_encoded_vector(handle["obs"][layer_key]).reshape(-1)
        coords = _read_obsm_matrix(handle, spatial_key)

        column_indices = [gene_to_index[g] for g in present]
        X_raw = _read_selected_X(handle, column_indices)

    if X_raw.ndim != 2:
        raise ValueError(
            f"Expected a 2D expression matrix in {path}; received {X_raw.shape}"
        )
    if X_raw.shape[0] != len(layers):
        raise ValueError(
            f"Expression rows ({X_raw.shape[0]}) and layer labels ({len(layers)}) "
            f"do not match in {path}."
        )
    if coords.shape[0] != X_raw.shape[0]:
        raise ValueError(
            f"Expression rows ({X_raw.shape[0]}) and spatial rows ({coords.shape[0]}) "
            f"do not match in {path}."
        )
    if not np.all(np.isfinite(coords)):
        raise ValueError(f"Spatial coordinates contain non-finite values in {path}.")

    # Sole expression transform for this file.
    expression_log1p = log1p_expression_once(X_raw)

    return MarkerData(
        expression_log1p=expression_log1p,
        genes=tuple(present),
        layers=layers,
        coords=coords,
    )


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def validate_config(cfg: Config) -> str:
    mode = str(cfg.run_mode).strip().lower()
    if mode not in {"eval", "plot", "both"}:
        raise ValueError("RUN_MODE must be 'eval', 'plot', or 'both'.")
    if not cfg.sample_ids:
        raise ValueError("sample_ids cannot be empty.")
    if not cfg.marker_genes:
        raise ValueError("marker_genes cannot be empty.")
    if not cfg.main_figure_genes:
        raise ValueError("main_figure_genes cannot be empty.")
    if cfg.spatial_point_size <= 0:
        raise ValueError("spatial_point_size must be positive.")
    return mode


def warn_if_mixed_environment() -> None:
    """Warn about cross-environment imports like the one visible in the traceback."""
    pandas_path = Path(pd.__file__).resolve()
    prefix = Path(sys.prefix).resolve()
    try:
        pandas_path.relative_to(prefix)
    except ValueError:
        warnings.warn(
            "pandas is imported from a different environment than sys.prefix: "
            f"pandas={pandas_path}, sys.prefix={prefix}. The streamlined loader "
            "does not use anndata/scanpy, but cleaning PYTHONPATH is still recommended."
        )


def canonical_layer_order(
    layer_values: Iterable[Any],
    preferred_order: Sequence[str],
) -> List[str]:
    values = [
        str(x)
        for x in pd.unique(pd.Series(list(layer_values)).dropna())
        if str(x).lower() not in {"nan", "none", "<na>"}
    ]
    value_set = set(values)
    ordered = [layer for layer in preferred_order if layer in value_set]

    def extra_key(label: str) -> Tuple[int, float, str]:
        low = label.lower()
        if low.startswith("layer"):
            suffix = label[len("Layer"):]
            try:
                return (0, float(suffix), label)
            except ValueError:
                return (0, math.inf, label)
        if label.upper() == "WM":
            return (2, math.inf, label)
        return (1, math.inf, label)

    extras = sorted([x for x in values if x not in ordered], key=extra_key)
    return ordered + extras


def short_layer_label(layer: str) -> str:
    text = str(layer)
    return f"L{text[len('Layer') :]}" if text.startswith("Layer") else text


def scipy_result_value(result: Any, attr: str, index: int) -> float:
    if hasattr(result, attr):
        return float(getattr(result, attr))
    return float(result[index])


def fmt_metric(value: float, digits: int = 2) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


# =============================================================================
# EXPRESSION COLLECTION
# =============================================================================


def marker_data_to_long(
    data: MarkerData,
    panel: str,
    genes: Sequence[str],
) -> pd.DataFrame:
    """Convert the already-log1p-transformed matrix to long format.

    No expression transform occurs here.
    """
    gene_to_index = {gene: i for i, gene in enumerate(data.genes)}
    present = [gene for gene in genes if gene in gene_to_index]
    if not present:
        return pd.DataFrame()

    indices = [gene_to_index[gene] for gene in present]
    X = np.asarray(data.expression_log1p[:, indices], dtype=np.float32)
    n_obs = X.shape[0]

    layers = np.asarray(data.layers, dtype=object)
    valid_layer = np.asarray(
        [
            value is not None
            and not (isinstance(value, float) and np.isnan(value))
            and str(value).lower() not in {"nan", "none", "<na>"}
            for value in layers
        ],
        dtype=bool,
    )

    coords = np.asarray(data.coords, dtype=np.float32)
    expression = X.reshape(-1, order="F")
    long_df = pd.DataFrame(
        {
            "gene": np.repeat(np.asarray(present, dtype=object), n_obs),
            "panel": panel,
            "layer": np.tile(layers, len(present)),
            "x": np.tile(coords[:, 0], len(present)),
            "y": np.tile(coords[:, 1], len(present)),
            "expression_log1p": expression,
        }
    )
    long_df = long_df[np.tile(valid_layer, len(present))].copy()
    long_df["expression_log1p"] = pd.to_numeric(
        long_df["expression_log1p"], errors="coerce"
    ).astype(np.float32)
    long_df = long_df[
        np.isfinite(long_df["expression_log1p"].to_numpy(dtype=float))
    ].copy()

    for col in ("gene", "panel", "layer"):
        long_df[col] = long_df[col].astype(str)

    return long_df[["gene", "panel", "layer", "x", "y", "expression_log1p"]]

def collect_expression_long(cfg: Config) -> pd.DataFrame:
    """Load native sections, pooled observations, PASTE, and AXIS."""
    singles: Dict[int, MarkerData] = {}
    for sid in sorted(cfg.sample_ids):
        path = Path(cfg.h5ad_dir) / f"{cfg.group_id}_{sid}.h5ad"
        print(f"[Load] Single {sid}: {path}")
        singles[sid] = read_marker_h5ad(path, cfg.marker_genes, cfg.layer_key, cfg.spatial_key)

    paste_path = Path(cfg.paste_h5ad).expanduser()
    print(f"[Load] PASTE: {paste_path}")
    paste = read_marker_h5ad(paste_path, cfg.marker_genes, cfg.layer_key, cfg.spatial_key)

    axis_path = Path(cfg.axis_h5ad).expanduser()
    print(f"[Load] Axis ({cfg.axis_aggregation_label}): {axis_path}")
    axis = read_marker_h5ad(axis_path, cfg.marker_genes, cfg.layer_key, cfg.spatial_key)

    all_data = list(singles.values()) + [paste, axis]
    common_genes = [
        gene
        for gene in cfg.marker_genes
        if all(gene in set(data.genes) for data in all_data)
    ]
    missing = [gene for gene in cfg.marker_genes if gene not in common_genes]
    if missing:
        warnings.warn(
            "Genes absent from at least one Single/PASTE/AXIS object were skipped: "
            + ", ".join(missing)
        )
    if not common_genes:
        raise ValueError("No marker gene is shared across all evaluated conditions.")

    single_parts = [
        marker_data_to_long(
            singles[sid],
            panel=f"Single {sid}",
            genes=common_genes,
        )
        for sid in sorted(singles)
    ]
    real_long = pd.concat(single_parts, ignore_index=True)

    pool_long = real_long.copy()
    pool_long["panel"] = "Pool"

    paste_long = marker_data_to_long(
        paste, "PASTE", common_genes
    )
    axis_long = marker_data_to_long(
        axis, "Axis", common_genes
    )

    long_df = pd.concat(
        [real_long, pool_long, paste_long, axis_long],
        ignore_index=True,
    )
    # Keep plain strings rather than pandas Categoricals. Plot/evaluation order is
    # controlled explicitly by cfg.panel_order and cfg.preferred_layer_order.
    long_df["panel"] = long_df["panel"].astype(str)
    long_df["layer"] = long_df["layer"].astype(str)
    return long_df


# =============================================================================
# LAYER-SEPARATION METRICS
# =============================================================================


def compute_layer_summary(
    long_df: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, List[str]]:
    layer_order = canonical_layer_order(
        long_df["layer"].astype(str).values,
        cfg.preferred_layer_order,
    )
    summary = (
        long_df.groupby(["gene", "panel", "layer"], observed=True)["expression_log1p"]
        .agg(
            n="size",
            mean_expression="mean",
            median_expression="median",
            std_expression="std",
            q25_expression=lambda x: x.quantile(0.25),
            q75_expression=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    panel_rank = {panel: i for i, panel in enumerate(cfg.panel_order)}
    layer_rank = {layer: i for i, layer in enumerate(layer_order)}
    gene_rank = {gene: i for i, gene in enumerate(cfg.marker_genes)}
    summary["_panel_rank"] = summary["panel"].map(panel_rank)
    summary["_layer_rank"] = summary["layer"].map(layer_rank)
    summary["_gene_rank"] = summary["gene"].map(gene_rank)
    summary = summary.sort_values(
        ["_gene_rank", "_panel_rank", "_layer_rank"]
    ).drop(columns=["_gene_rank", "_panel_rank", "_layer_rank"])
    return summary.reset_index(drop=True), layer_order


def kruskal_eta_h_squared(
    groups: Sequence[np.ndarray],
) -> Tuple[float, float, float, int, int]:
    valid = [np.asarray(x, dtype=float) for x in groups if len(x) > 0]
    n_total = int(sum(len(x) for x in valid))
    k = len(valid)
    if k < 2 or n_total <= k:
        return np.nan, np.nan, np.nan, n_total, k

    try:
        result = kruskal(*valid)
        h_stat = scipy_result_value(result, "statistic", 0)
        p_value = scipy_result_value(result, "pvalue", 1)
    except ValueError as exc:
        if "identical" not in str(exc).lower():
            raise
        h_stat, p_value = 0.0, 1.0

    eta_h_squared = max(0.0, (h_stat - k + 1.0) / (n_total - k))
    return float(eta_h_squared), float(h_stat), float(p_value), n_total, k


def direction_invariant_auroc(
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> Tuple[float, float, str]:
    left = np.asarray(left_values, dtype=float)
    right = np.asarray(right_values, dtype=float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) == 0 or len(right) == 0:
        return np.nan, np.nan, "NA"

    scores = np.concatenate([left, right])
    labels = np.concatenate(
        [np.zeros(len(left), dtype=np.int8), np.ones(len(right), dtype=np.int8)]
    )
    if np.nanmax(scores) - np.nanmin(scores) < 1e-12:
        raw_auc = 0.5
    else:
        raw_auc = float(roc_auc_score(labels, scores))
    separation_auc = max(raw_auc, 1.0 - raw_auc)
    higher_side = "right" if raw_auc >= 0.5 else "left"
    return float(separation_auc), float(raw_auc), higher_side


def compute_adjacent_layer_aurocs(
    sub: pd.DataFrame,
    layer_order: Sequence[str],
    value_col: str = "expression_log1p",
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    pair_rows: List[Dict[str, Any]] = []
    for left_layer, right_layer in zip(layer_order[:-1], layer_order[1:]):
        left = sub.loc[sub["layer"] == left_layer, value_col].dropna().to_numpy(dtype=float)
        right = sub.loc[sub["layer"] == right_layer, value_col].dropna().to_numpy(dtype=float)
        if len(left) == 0 or len(right) == 0:
            continue

        separation_auc, raw_auc, higher_side = direction_invariant_auroc(left, right)
        pair_rows.append(
            {
                "layer_left": left_layer,
                "layer_right": right_layer,
                "n_left": int(len(left)),
                "n_right": int(len(right)),
                "raw_auc_right_positive": raw_auc,
                "adjacent_separation_auroc": separation_auc,
                "higher_expression_layer": (
                    right_layer if higher_side == "right" else left_layer
                ),
            }
        )

    aucs = np.asarray(
        [row["adjacent_separation_auroc"] for row in pair_rows], dtype=float
    )
    aucs = aucs[np.isfinite(aucs)]
    return (
        {
            "adjacent_auc_mean": float(np.mean(aucs)) if len(aucs) else np.nan,
            "adjacent_auc_median": float(np.median(aucs)) if len(aucs) else np.nan,
            "adjacent_auc_min": float(np.min(aucs)) if len(aucs) else np.nan,
            "adjacent_auc_max": float(np.max(aucs)) if len(aucs) else np.nan,
            "n_adjacent_pairs": int(len(aucs)),
        },
        pair_rows,
    )


def compute_layer_separation_metrics(
    long_df: pd.DataFrame,
    layer_summary: pd.DataFrame,
    layer_order: Sequence[str],
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    present_panels = [p for p in cfg.panel_order if p in set(long_df["panel"])]
    present_genes = [g for g in cfg.marker_genes if g in set(long_df["gene"])]
    mean_lookup = layer_summary.set_index(["gene", "panel", "layer"])[
        "mean_expression"
    ].to_dict()

    rows: List[Dict[str, Any]] = []
    adjacent_rows: List[Dict[str, Any]] = []

    for gene in present_genes:
        for panel in present_panels:
            sub = long_df[(long_df["gene"] == gene) & (long_df["panel"] == panel)]
            if sub.empty:
                continue

            groups: List[np.ndarray] = []
            observed_layers: List[str] = []
            layer_means: List[float] = []
            for layer in layer_order:
                values = sub.loc[sub["layer"] == layer, "expression_log1p"].dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                groups.append(values)
                observed_layers.append(layer)
                layer_means.append(
                    float(mean_lookup.get((gene, panel, layer), np.mean(values)))
                )

            eta_h_sq, h_stat, p_value, n_total, k = kruskal_eta_h_squared(groups)
            auc_summary, pair_rows = compute_adjacent_layer_aurocs(sub, layer_order)
            adjacent_rows.extend(
                {"gene": gene, "panel": panel, **pair} for pair in pair_rows
            )

            if len(layer_means) >= 2:
                order = np.argsort(np.asarray(layer_means))[::-1]
                dominant_layer = observed_layers[int(order[0])]
                second_layer = observed_layers[int(order[1])]
                dominant_margin = float(
                    layer_means[int(order[0])] - layer_means[int(order[1])]
                )
            elif len(layer_means) == 1:
                dominant_layer = observed_layers[0]
                second_layer = np.nan
                dominant_margin = np.nan
            else:
                dominant_layer = second_layer = np.nan
                dominant_margin = np.nan

            rows.append(
                {
                    "gene": gene,
                    "panel": panel,
                    "eta_h_squared": eta_h_sq,
                    "kruskal_h": h_stat,
                    "kruskal_p": p_value,
                    "n_observations": n_total,
                    "n_layers": k,
                    **auc_summary,
                    "dominant_layer": dominant_layer,
                    "second_layer": second_layer,
                    "dominant_margin_log1p": dominant_margin,
                }
            )

    metrics = pd.DataFrame(rows)
    adjacent_df = pd.DataFrame(adjacent_rows)

    # Single-section baselines used to contextualize atlas-level metrics.
    real = metrics[metrics["panel"].isin(cfg.real_panels)]
    eta_baseline = (
        real.groupby("gene", observed=True)["eta_h_squared"]
        .agg(
            single_eta_h_mean="mean",
            single_eta_h_median="median",
            single_eta_h_min="min",
            single_eta_h_max="max",
        )
        .reset_index()
    )
    auc_baseline = (
        real.groupby("gene", observed=True)["adjacent_auc_mean"]
        .agg(
            single_adjacent_auc_mean="mean",
            single_adjacent_auc_median="median",
            single_adjacent_auc_min="min",
            single_adjacent_auc_max="max",
        )
        .reset_index()
    )
    metrics = metrics.merge(eta_baseline, on="gene", how="left", validate="many_to_one")
    metrics = metrics.merge(auc_baseline, on="gene", how="left", validate="many_to_one")
    metrics["delta_eta_h_vs_single_mean"] = (
        metrics["eta_h_squared"] - metrics["single_eta_h_mean"]
    )
    metrics["delta_adjacent_auc_vs_single_mean"] = (
        metrics["adjacent_auc_mean"] - metrics["single_adjacent_auc_mean"]
    )
    metrics["eta_h_exceeds_best_single"] = (
        metrics["eta_h_squared"] > metrics["single_eta_h_max"]
    )
    metrics["adjacent_auc_exceeds_best_single"] = (
        metrics["adjacent_auc_mean"] > metrics["single_adjacent_auc_max"]
    )

    panel_rank = {panel: i for i, panel in enumerate(cfg.panel_order)}
    gene_rank = {gene: i for i, gene in enumerate(cfg.marker_genes)}
    metrics["_panel_rank"] = metrics["panel"].map(panel_rank)
    metrics["_gene_rank"] = metrics["gene"].map(gene_rank)
    metrics = metrics.sort_values(["_gene_rank", "_panel_rank"]).drop(
        columns=["_gene_rank", "_panel_rank"]
    )

    if not adjacent_df.empty:
        layer_rank = {layer: i for i, layer in enumerate(layer_order)}
        adjacent_df["_panel_rank"] = adjacent_df["panel"].map(panel_rank)
        adjacent_df["_gene_rank"] = adjacent_df["gene"].map(gene_rank)
        adjacent_df["_layer_rank"] = adjacent_df["layer_left"].map(layer_rank)
        adjacent_df = adjacent_df.sort_values(
            ["_gene_rank", "_panel_rank", "_layer_rank"]
        ).drop(columns=["_gene_rank", "_panel_rank", "_layer_rank"])

    return metrics.reset_index(drop=True), adjacent_df.reset_index(drop=True)


def summarize_methods(metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for panel in ("Pool", "PASTE", "Axis"):
        panel_df = metrics[metrics["panel"] == panel]
        if panel_df.empty:
            continue
        eta = panel_df["eta_h_squared"].dropna().to_numpy(dtype=float)
        auc = panel_df["adjacent_auc_mean"].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "panel": panel,
                "axis_aggregation_label": (
                    cfg.axis_aggregation_label if panel == "Axis" else ""
                ),
                "n_genes": int(panel_df["gene"].nunique()),
                "mean_eta_h_squared": float(np.mean(eta)) if len(eta) else np.nan,
                "median_eta_h_squared": float(np.median(eta)) if len(eta) else np.nan,
                "mean_adjacent_auc": float(np.mean(auc)) if len(auc) else np.nan,
                "median_adjacent_auc": float(np.median(auc)) if len(auc) else np.nan,
                "mean_delta_eta_h_vs_single": float(
                    panel_df["delta_eta_h_vs_single_mean"].mean()
                ),
                "mean_delta_adjacent_auc_vs_single": float(
                    panel_df["delta_adjacent_auc_vs_single_mean"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def save_metrics(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg.outdir_path.mkdir(parents=True, exist_ok=True)

    print("[1/5] Collect marker expression")
    long_df = collect_expression_long(cfg)

    print("[2/5] Summarize expression by layer")
    layer_summary, layer_order = compute_layer_summary(long_df, cfg)

    print("[3/5] Calculate layer-separation metrics")
    metrics, adjacent_df = compute_layer_separation_metrics(
        long_df, layer_summary, layer_order, cfg
    )

    print("[4/5] Summarize Pool, PASTE, and Axis")
    method_summary = summarize_methods(metrics, cfg)

    print("[5/5] Save numerical outputs")
    long_df.to_csv(cfg.long_csv, index=False)
    layer_summary.to_csv(cfg.layer_summary_csv, index=False)
    metrics.to_csv(cfg.separation_csv, index=False)
    adjacent_df.to_csv(cfg.adjacent_auc_csv, index=False)
    method_summary.to_csv(cfg.method_summary_csv, index=False)

    for path in (
        cfg.long_csv,
        cfg.layer_summary_csv,
        cfg.separation_csv,
        cfg.adjacent_auc_csv,
        cfg.method_summary_csv,
    ):
        print(f"[OK] saved: {path}")
    return long_df, metrics


# =============================================================================
# PLOTTING
# =============================================================================


def set_publication_style() -> None:
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def get_layer_palette(layer_order: Sequence[str]) -> Dict[str, str]:
    base = {
        "Layer1": "#EDF6F0",
        "Layer2": "#DCEDE3",
        "Layer3": "#C7E1D1",
        "Layer4": "#A9D2BB",
        "Layer5": "#83BFA1",
        "Layer6": "#5AA787",
        "WM": "#357A63",
    }
    fallback = sns.color_palette("light:#357A63", n_colors=max(2, len(layer_order) + 2))[2:]
    return {
        str(layer): base.get(
            str(layer), mpl.colors.to_hex(fallback[i % len(fallback)])
        )
        for i, layer in enumerate(layer_order)
    }

# METHOD_PALETTE = {
#     "Real": "#111111",
#     "SpatialZ": "#7C8DA6",
#     "AXIS": "#9E2F2A",
# }
# colors = {"Pool": "#356E9B", "PASTE": "#CC79A7", "Axis": "#E69F00"}
def get_panel_color(panel: str) -> str:
    colors = {"Pool": "#E69F00", "PASTE":  "#356E9B","Axis": "#9E2F2A" }
    return "#8F8F8F" if str(panel).startswith("Single") else colors.get(panel, "#6F6F6F")


def style_axis(ax: mpl.axes.Axes, grid_axis: Optional[str] = "y") -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color="#E8E8E8", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig: mpl.figure.Figure, stem: Path, cfg: Config) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in cfg.save_formats:
        extension = extension.lower().lstrip(".")
        path = stem.with_suffix(f".{extension}")
        kwargs: Dict[str, Any] = {
            "bbox_inches": "tight",
            "facecolor": "white",
            "pad_inches": 0.06,
        }
        if extension == "png":
            kwargs["dpi"] = cfg.png_dpi
        fig.savefig(path, **kwargs)
        print(f"[OK] figure: {path}")


def candidate_panels_present(metrics: pd.DataFrame) -> List[str]:
    observed = set(metrics["panel"].astype(str))
    return [p for p in ("Pool", "PASTE", "Axis") if p in observed]


def ordered_summary_genes(metrics: pd.DataFrame, cfg: Config) -> List[str]:
    available = set(metrics["gene"].astype(str))
    return [gene for gene in cfg.genes_to_plot if gene in available]


def draw_metric_summary_panel(
    ax: mpl.axes.Axes,
    metrics: pd.DataFrame,
    cfg: Config,
    genes: Sequence[str],
    metric_col: str,
    single_min_col: str,
    single_max_col: str,
    single_mean_col: str,
    title: str,
    ylabel: str,
    y_limits: Tuple[float, float],
    show_legend: bool = True,
) -> None:
    candidates = candidate_panels_present(metrics)
    x = np.arange(len(genes), dtype=float)
    baseline = metrics.drop_duplicates("gene").set_index("gene")

    real_low = baseline[single_min_col].reindex(genes).to_numpy(dtype=float)
    real_high = baseline[single_max_col].reindex(genes).to_numpy(dtype=float)
    real_mean = baseline[single_mean_col].reindex(genes).to_numpy(dtype=float)

    ax.vlines(
        x, real_low, real_high,
        color="#C7C7C7", linewidth=4.8, zorder=1,
        label="Single-section range",
    )
    ax.scatter(
        x+0.05, real_mean,
        s=18, color="#7D7D7D", edgecolor="white", linewidth=0.35,
        zorder=3, label="Single-section mean",
    )

    pool_values = (
        metrics[metrics["panel"] == "Pool"]
        .set_index("gene")[metric_col]
        .reindex(genes)
        .to_numpy(dtype=float)
    )

    ax.scatter(
        x-0.05, pool_values,
        s=18, color=get_panel_color("Pool"), edgecolor="white", linewidth=0.35,
        zorder=4, label="Pool",
    )

    offsets = {"Pool": -0.1, "PASTE": 0.0, "Axis": 0.}
    markers = {"Pool":  "s", "PASTE": "o", "Axis":"^"}
    for panel in candidates:
        if panel=="Pool":
            continue
        values = (
            metrics[metrics["panel"] == panel]
            .set_index("gene")[metric_col]
            .reindex(genes)
            .to_numpy(dtype=float)
        )
        ax.scatter(
            x + offsets[panel], values,
            s=80, marker=markers[panel], color=get_panel_color(panel),
            edgecolor="white", linewidth=0.45, zorder=5, label=panel,
        )

    ax.set_title(title, loc="center", fontweight="normal", pad=8,fontsize=18)
    ax.set_ylabel(ylabel,fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=33, ha="center", rotation_mode="anchor",fontsize=12)
    ax.set_ylim(*y_limits)
    style_axis(ax)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        desired = ("Axis","PASTE","Single-section range", "Single-section mean","Pool")
        indices = [
            i for name in desired for i, label in enumerate(labels) if label == name
        ]
        ax.legend(
            [handles[i] for i in indices],
            [
                display_panel_name(labels[i])
                if labels[i] in {"Axis", "Pool", "PASTE"}
                else labels[i]
                for i in indices
            ],
            frameon=False,
            ncol=3,
            loc="upper left",
            bbox_to_anchor=(-0.02, 1.04),
            borderaxespad=0.5,
            handletextpad=0.45,
            columnspacing=0.5,
            fontsize=12,
        )



def plot_all_gene_summary(
    metrics: pd.DataFrame,
    cfg: Config,
) -> None:
    """
    Plot the two all-gene summary metrics as separate standalone figures:

    1. Global cortical-layer separation
    2. Mean adjacent-layer discrimination

    The dimensions and axis layout are chosen to preserve the physical
    subplot geometry of the previous 11.0 x 3.5 two-panel figure.
    """
    genes = ordered_summary_genes(metrics, cfg)
    if not genes:
        warnings.warn("No configured summary genes are available.")
        return

    # ------------------------------------------------------------------
    # Layout
    #
    # Previous combined figure:
    #   figsize = (11.0, 3.5)
    #   two equal-width axes
    #   wspace = 0.20
    #
    # A 5.0 x 3.5 standalone figure approximately preserves the physical
    # width of each axis in the previous two-panel layout.
    # ------------------------------------------------------------------
    FIGSIZE = (5.5, 4)

    LEFT = 0.055
    RIGHT = 0.992
    TOP = 0.86
    BOTTOM = 0.20

    # ==============================================================
    # 1. Global cortical-layer separation
    # ==============================================================
    fig_eta = plt.figure(figsize=FIGSIZE)

    grid_eta = fig_eta.add_gridspec(
        1,
        1,
        left=LEFT,
        right=RIGHT,
        top=TOP,
        bottom=BOTTOM,
    )

    ax_eta = fig_eta.add_subplot(grid_eta[0, 0])

    draw_metric_summary_panel(
        ax_eta,
        metrics,
        cfg,
        genes,
        metric_col="eta_h_squared",
        single_min_col="single_eta_h_min",
        single_max_col="single_eta_h_max",
        single_mean_col="single_eta_h_mean",
        title="Global laminar separation",
        ylabel=r"Kruskal–Wallis effect size",
        y_limits=(-0.05, 1.0),
        show_legend=True,
    )

    save_figure(
        fig_eta,
        cfg.outdir_path / "all_genes_global_layer_separation",
        cfg,
    )
    plt.close(fig_eta)

    # ==============================================================
    # 2. Mean adjacent-layer discrimination
    # ==============================================================
    fig_auc = plt.figure(figsize=FIGSIZE)

    grid_auc = fig_auc.add_gridspec(
        1,
        1,
        left=LEFT,
        right=RIGHT,
        top=TOP,
        bottom=BOTTOM,
    )

    ax_auc = fig_auc.add_subplot(grid_auc[0, 0])

    draw_metric_summary_panel(
        ax_auc,
        metrics,
        cfg,
        genes,
        metric_col="adjacent_auc_mean",
        single_min_col="single_adjacent_auc_min",
        single_max_col="single_adjacent_auc_max",
        single_mean_col="single_adjacent_auc_mean",
        title="Adjacent-layer discrimination",
        ylabel="Mean adjacent-layer AUROC",
        y_limits=(0.48, 1.1),
        show_legend=True,
    )

    # ax_auc.axhline(
    #     0.5,
    #     color="#8A8A8A",
    #     linewidth=0.7,
    #     linestyle="--",
    #     zorder=0,
    # )

    save_figure(
        fig_auc,
        cfg.outdir_path / "all_genes_adjacent_layer_discrimination",
        cfg,
    )
    plt.close(fig_auc)


def violinplot_compat(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    layer_order: Sequence[str],
    palette: Mapping[str, str],
) -> None:
    common = dict(
        data=data,
        x="layer",
        y="expression_log1p",
        order=list(layer_order),
        inner=None,
        cut=0,
        linewidth=0.68,
        saturation=0.95,
        ax=ax,
    )
    try:
        sns.violinplot(
            **common,
            hue="layer",
            hue_order=list(layer_order),
            palette=dict(palette),
            dodge=False,
            legend=False,
            density_norm="width",
            common_norm=False,
        )
    except (TypeError, AttributeError):
        sns.violinplot(
            **common,
            palette=[palette[layer] for layer in layer_order],
            scale="width",
        )
    sns.boxplot(
        data=data,
        x="layer",
        y="expression_log1p",
        order=list(layer_order),
        width=0.14,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "#222222", "linewidth": 0.62},
        medianprops={"color": "#111111", "linewidth": 0.9},
        whiskerprops={"color": "#222222", "linewidth": 0.62},
        capprops={"color": "#222222", "linewidth": 0.62},
        ax=ax,
    )


def representative_panels(cfg: Config) -> List[str]:
    """Panels displayed in each main-figure case study."""
    return list(cfg.real_panels) + ["Pool", "PASTE", "Axis"]


def draw_spatial_expression_map(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    cfg: Config,
) -> mpl.collections.PathCollection:
    """Draw the one-time loaded log1p expression directly.

    No additional log transform or min-max normalization is performed here.
    """
    finite_mask = (
        np.isfinite(data["x"].to_numpy(dtype=float))
        & np.isfinite(data["y"].to_numpy(dtype=float))
        & np.isfinite(data["expression_log1p"].to_numpy(dtype=float))
    )
    sub = data.loc[finite_mask].copy()

    if sub.empty:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax.scatter([], [], c=[], cmap=cfg.spatial_cmap)

    sub = sub.sort_values("expression_log1p", kind="mergesort")
    expression = sub["expression_log1p"].to_numpy(dtype=float)

    vmax = float(np.nanmax(expression)) if expression.size else 1.0
    vmax = max(vmax, 1e-8)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

    scatter = ax.scatter(
        sub["x"].to_numpy(dtype=float),
        sub["y"].to_numpy(dtype=float),
        c=expression,
        cmap=cfg.spatial_cmap,
        norm=norm,
        s=cfg.spatial_point_size,
        alpha=1.0,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal", adjustable="box")
    if cfg.invert_spatial_y:
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return scatter

def draw_representative_case_study(
    fig: mpl.figure.Figure,
    subspec: mpl.gridspec.SubplotSpec,
    long_df: pd.DataFrame,
    metrics: pd.DataFrame,
    gene: str,
    cfg: Config,
    panel_letter: str,
    descriptor: str,
) -> Tuple[List[mpl.axes.Axes], List[mpl.axes.Axes]]:
    """Draw one case study with spatial maps above layer-stratified violins."""
    panels = [
        panel
        for panel in representative_panels(cfg)
        if panel
        in set(
            long_df.loc[
                long_df["gene"].astype(str) == str(gene),
                "panel",
            ].astype(str)
        )
    ]
    if not panels:
        return [], []

    gene_df = long_df[long_df["gene"].astype(str) == str(gene)].copy()
    gene_df = gene_df[gene_df["panel"].astype(str).isin(panels)].copy()

    layer_order = canonical_layer_order(
        gene_df["layer"].astype(str),
        cfg.preferred_layer_order,
    )
    palette = get_layer_palette(layer_order)

    expression_values = gene_df["expression_log1p"].to_numpy(dtype=float)
    y_values = expression_values[np.isfinite(expression_values)]
    y_max = float(np.nanmax(y_values)) if y_values.size else 1.0
    y_top = max(y_max * 1.2, y_max + 0.2)

    nested = subspec.subgridspec(
        2,
        len(panels),
        height_ratios=(1.05, 1.0),
        hspace=0.18,
        wspace=0.10,
    )

    map_axes: List[mpl.axes.Axes] = []
    violin_axes: List[mpl.axes.Axes] = []
    metric_lookup = metrics[
        metrics["gene"].astype(str) == str(gene)
    ].set_index("panel")

    for i, panel in enumerate(panels):
        sub = gene_df[gene_df["panel"].astype(str) == panel].copy()

        map_ax = fig.add_subplot(nested[0, i])
        map_axes.append(map_ax)
        draw_spatial_expression_map(map_ax, sub, cfg)

        panel_color = get_panel_color(panel)
        map_ax.set_title(
            panel,
            color=panel_color if panel in {"Pool", "PASTE", "Axis"} else "#222222",
            fontweight="normal",
            pad=3,
        )

        if i == 0:
            map_ax.text(
                -0.09,
                0.5,
                "Spatial\ngene expression",
                transform=map_ax.transAxes,
                ha="right",
                va="center",
                rotation=90,
                fontsize=7.0,
                color="#444444",
            )
            map_ax.text(
                0.0,
                -0.07,
                "log1p expression",
                transform=map_ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.8,
                color="#666666",
            )

        violin_ax = fig.add_subplot(
            nested[1, i],
            sharey=violin_axes[0] if violin_axes else None,
        )
        violin_axes.append(violin_ax)

        sub["layer"] = pd.Categorical(
            sub["layer"].astype(str),
            categories=layer_order,
            ordered=True,
        )
        violinplot_compat(violin_ax, sub, layer_order, palette)

        violin_ax.set_xlabel("")
        violin_ax.set_ylim(0.0, y_top)
        violin_ax.set_xticks(np.arange(len(layer_order)))
        violin_ax.set_xticklabels(
            [short_layer_label(layer) for layer in layer_order],
            # rotation=33,
            ha="right",
            rotation_mode="anchor",
            fontsize=12,
        )

        if i == 0:
            violin_ax.set_ylabel(
                "log1p (raw expression)"
            )
        else:
            violin_ax.set_ylabel("")
            violin_ax.tick_params(axis="y", labelleft=False)

        style_axis(violin_ax)
        if panel in {"Pool", "PASTE", "Axis"}:
            violin_ax.spines["bottom"].set_color(panel_color)
            violin_ax.spines["bottom"].set_linewidth(1.4)
            violin_ax.set_facecolor(mpl.colors.to_rgba(panel_color, 0.025))

        if panel in metric_lookup.index:
            eta_h = float(metric_lookup.loc[panel, "eta_h_squared"])
            adjacent_auc = float(metric_lookup.loc[panel, "adjacent_auc_mean"])
            violin_ax.text(
                0.035,
                0.965,
                f"$\\eta_H^2$={fmt_metric(eta_h)}\nAdj. AUROC={fmt_metric(adjacent_auc)}",
                transform=violin_ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#333333",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.82,
                ),
            )

    bbox = subspec.get_position(fig)
    fig.text(
        bbox.x0,
        bbox.y1 + 0.017,
        panel_letter,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        bbox.x0 + 0.021,
        bbox.y1 + 0.017,
        f"{gene}: {descriptor}",
        fontsize=9.4,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    return map_axes, violin_axes

def plot_gene_violin(
    long_df: pd.DataFrame,
    metrics: pd.DataFrame,
    gene: str,
    cfg: Config,
) -> None:
    gene_df = long_df[long_df["gene"] == gene].copy()
    if gene_df.empty:
        return
    panel_order = [p for p in cfg.panel_order if p in set(gene_df["panel"])]
    layer_order = canonical_layer_order(gene_df["layer"], cfg.preferred_layer_order)
    palette = get_layer_palette(layer_order)
    ymax = float(np.nanmax(gene_df["expression_log1p"].to_numpy(dtype=float)))
    ytop = max(ymax * 1.06, ymax + 0.10)

    fig, axes = plt.subplots(
        1, len(panel_order),
        figsize=(max(13.0, 1.86 * len(panel_order)), 3.75),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    axes = np.atleast_1d(axes)
    metric_lookup = metrics[metrics["gene"] == gene].set_index("panel")

    for i, (ax, panel) in enumerate(zip(axes, panel_order)):
        sub = gene_df[gene_df["panel"] == panel].copy()
        violinplot_compat(ax, sub, layer_order, palette)
        ax.set_title(panel, fontweight="normal", pad=5)
        ax.set_xlabel("")
        ax.set_ylim(0.0, ytop)
        ax.set_xticks(np.arange(len(layer_order)))
        ax.set_xticklabels(
            [short_layer_label(layer) for layer in layer_order],
            # rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_ylabel("log1p(raw expression)" if i == 0 else "")
        if i > 0:
            ax.tick_params(axis="y", labelleft=False)
        style_axis(ax)

        if cfg.annotate_violin_metrics and panel in metric_lookup.index:
            eta = float(metric_lookup.loc[panel, "eta_h_squared"])
            auc = float(metric_lookup.loc[panel, "adjacent_auc_mean"])
            ax.text(
                0.035, 0.965,
                f"$\\eta_H^2$={fmt_metric(eta)}\nAdj. AUROC={fmt_metric(auc)}",
                transform=ax.transAxes,
                ha="left", va="top", fontsize=6.8,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.84),
            )

    fig.suptitle(f"{gene} layer-stratified expression", y=1.01, fontsize=12, fontweight="bold")
    save_figure(fig, cfg.outdir_path / f"supplement_{gene}_layer_separation_violin", cfg)
    plt.close(fig)


def load_plot_inputs(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = [cfg.long_csv, cfg.separation_csv]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Plot mode requires current evaluation outputs. Missing: "
            + ", ".join(str(path) for path in missing)
        )
    long_df = pd.read_csv(cfg.long_csv)
    metrics = pd.read_csv(cfg.separation_csv)
    required_long = {"gene", "panel", "layer", "x", "y", "expression_log1p"}
    if not required_long.issubset(long_df.columns):
        missing_cols = sorted(required_long.difference(long_df.columns))
        raise ValueError(
            f"{cfg.long_csv} is missing {missing_cols}; rerun with RUN_MODE='eval' "
            "or 'both' because the CSV must contain the one-time log1p expression column."
        )
    for col in ("gene", "panel", "layer"):
        if col in long_df:
            long_df[col] = long_df[col].astype(str)
    for col in ("gene", "panel"):
        if col in metrics:
            metrics[col] = metrics[col].astype(str)
    return long_df, metrics



# =============================================================================
# MERGED REPRESENTATIVE CASE FIGURES
# =============================================================================


@dataclass(frozen=True)
class CaseStudyFigureStyle:
    """Geometry and styling for one gene x method case-study figure.

    All panel positions are explicit figure-coordinate rectangles
    [left, bottom, width, height], making the layout easy to tune without
    changing plotting logic.
    """

    # Entire output canvas.
    figsize: Tuple[float, float] = (5.20, 2.5)

    # (left, bottom, width, height)
    # Left: layer-wise violin plot.
    violin_axes_rect: Tuple[float, float, float, float] = (
        0.12, 0.205, 0.3, 0.65
    )

    # Right: spatial gene map.
    spatial_axes_rect: Tuple[float, float, float, float] = (
        0.45, 0.1, 0.4, 0.785
    )

    # Colorbar belonging to the spatial map.
    cbar_axes_rect: Tuple[float, float, float, float] = (
        0.86, 0.25, 0.020, 0.430
    )

    # Overall title. Set title_y=None to suppress it.
    title_y: Optional[float] = 0.955
    title_fontsize: float = 14.0
    title_weight: str = "normal"

    # Violin settings.
    violin_ylabel: str = "log1p expression"
    violin_ylabel_fontsize: float = 10
    violin_xtick_rotation: float = 0
    violin_xtick_ha: str = "right"
    violin_annotation_fontsize: float = 11
    violin_annotation_x: float = 0.04
    violin_annotation_y: float = 1
    violin_annotation_bbox_alpha: float = 0.0

    # Spatial-map geometry.
    spatial_aspect: str = "equal"
    spatial_anchor: str = "C"
    use_shared_spatial_extent: bool = False
    x_pad_ratio: float = 0.03
    y_pad_ratio: float = 0.03

    # Colorbar settings. Tick locations are chosen by MaxNLocator so they use
    # clean steps such as 0.1, 0.2, 0.5, 1.0 rather than arbitrary linspace values.
    cbar_label: str = "log1p expression"
    cbar_label_fontsize: float = 7.3
    cbar_tick_fontsize: float = 7.0
    cbar_labelpad: float = 4.0
    cbar_tick_pad: float = 2.0
    cbar_nbins: int = 5
    cbar_steps: Tuple[float, ...] = (1.0, 2.0, 2.5, 5.0, 10.0)
    cbar_decimals: int = 1


CASE_STUDY_STYLE = CaseStudyFigureStyle()
CASE_STUDY_PANELS: Tuple[str, ...] = ("Single 0", "Pool", "PASTE", "Axis")


def safe_panel_name(panel: str) -> str:
    return str(panel).lower().replace(" ", "_")


def case_study_panels_present(long_df: pd.DataFrame) -> List[str]:
    observed = set(long_df["panel"].astype(str))
    return [panel for panel in CASE_STUDY_PANELS if panel in observed]


def compute_gene_shared_extent(
    gene_df: pd.DataFrame,
    panels: Sequence[str],
    style: CaseStudyFigureStyle,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Common spatial frame for one gene across the selected methods."""
    sub = gene_df[gene_df["panel"].astype(str).isin(list(panels))].copy()
    x = sub["x"].to_numpy(dtype=float)
    y = sub["y"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return (0.0, 1.0), (0.0, 1.0)

    x = x[finite]
    y = y[finite]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xpad = max((xmax - xmin) * style.x_pad_ratio, 1e-3)
    ypad = max((ymax - ymin) * style.y_pad_ratio, 1e-3)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def representative_y_top(gene_df: pd.DataFrame) -> float:
    """Shared violin y-limit for all methods of one representative gene."""
    values = gene_df["expression_log1p"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    ymax = float(np.nanmax(values))
    return max(ymax * 1.06, ymax + 0.08)


def save_case_figure_fixed_canvas(
    fig: mpl.figure.Figure,
    stem: Path,
    cfg: Config,
) -> None:
    """Save without tight-bbox recropping so all case figures keep one geometry."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in cfg.save_formats:
        extension = extension.lower().lstrip(".")
        path = stem.with_suffix(f".{extension}")
        kwargs: Dict[str, Any] = {
            "facecolor": "white",
            "transparent": False,
        }
        if extension == "png":
            kwargs["dpi"] = cfg.png_dpi
        fig.savefig(path, **kwargs)
        print(f"[OK] figure: {path}")


def format_case_colorbar(
    cbar: mpl.colorbar.Colorbar,
    style: CaseStudyFigureStyle,
) -> None:
    """Use clean automatic tick spacing and fixed one-decimal labels."""
    cbar.locator = mpl.ticker.MaxNLocator(
        nbins=style.cbar_nbins,
        steps=list(style.cbar_steps),
        min_n_ticks=3,
    )
    cbar.formatter = mpl.ticker.FormatStrFormatter(
        f"%.{style.cbar_decimals}f"
    )
    cbar.update_ticks()
    cbar.set_label(
        style.cbar_label,
        fontsize=style.cbar_label_fontsize,
        rotation=90,
        labelpad=style.cbar_labelpad,
    )
    cbar.ax.tick_params(
        labelsize=style.cbar_tick_fontsize,
        pad=style.cbar_tick_pad,
    )


def plot_gene_method_case(
    long_df: pd.DataFrame,
    metrics: pd.DataFrame,
    gene: str,
    panel: str,
    cfg: Config,
    layer_order: Sequence[str],
    palette: Mapping[str, str],
    shared_y_top: float,
    shared_xlim: Tuple[float, float],
    shared_ylim: Tuple[float, float],
    style: CaseStudyFigureStyle= CASE_STUDY_STYLE,
) -> Optional[mpl.figure.Figure]:
    """Plot one case-study image: violin on the left, spatial map on the right."""
    sub = long_df[
        (long_df["gene"].astype(str) == str(gene))
        & (long_df["panel"].astype(str) == str(panel))
    ].copy()
    if sub.empty:
        return None

    fig = plt.figure(figsize=style.figsize, facecolor="white")

    # ------------------------------------------------------------------
    # Left panel: layer-wise expression violin.
    # ------------------------------------------------------------------
    violin_ax = fig.add_axes(style.violin_axes_rect)
    violin_sub = sub.copy()
    violin_sub["layer"] = pd.Categorical(
        violin_sub["layer"].astype(str),
        categories=list(layer_order),
        ordered=True,
    )
    violinplot_compat(violin_ax, violin_sub, layer_order, palette)
    violin_ax.set_ylim(0.0, shared_y_top)
    violin_ax.set_xticks(np.arange(len(layer_order)))
    violin_ax.set_xticklabels(
        [short_layer_label(layer) for layer in layer_order],
        rotation=style.violin_xtick_rotation,
        ha=style.violin_xtick_ha,
        rotation_mode="anchor",
    )
    violin_ax.set_xlabel("")
    violin_ax.set_ylabel(
        style.violin_ylabel,
        fontsize=style.violin_ylabel_fontsize,
    )
    style_axis(violin_ax)

    metric_row = metrics[
        (metrics["gene"].astype(str) == str(gene))
        & (metrics["panel"].astype(str) == str(panel))
    ]

    if cfg.annotate_violin_metrics and not metric_row.empty:
        row = metric_row.iloc[0]
        violin_ax.text(
            style.violin_annotation_x,
            style.violin_annotation_y,
            f"KW $\\eta_H^2$={fmt_metric(float(row['eta_h_squared']))}"
            # f"KW effect size ={fmt_metric(float(row['eta_h_squared']))}"
            f"\nAUROC={fmt_metric(float(row['adjacent_auc_mean']))}",
            transform=violin_ax.transAxes,
            ha="left",
            va="top",
            fontsize=style.violin_annotation_fontsize,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=style.violin_annotation_bbox_alpha,
            ),
        )

    # ------------------------------------------------------------------
    # Right panel: spatial gene map using the already log1p-transformed data.
    # ------------------------------------------------------------------
    spatial_ax = fig.add_axes(style.spatial_axes_rect)
    scatter = draw_spatial_expression_map(
        ax=spatial_ax,
        data=sub,
        cfg=cfg,
    )

    if style.use_shared_spatial_extent:
        spatial_ax.set_xlim(float(shared_xlim[0]), float(shared_xlim[1]))
        if cfg.invert_spatial_y:
            spatial_ax.set_ylim(float(shared_ylim[1]), float(shared_ylim[0]))
        else:
            spatial_ax.set_ylim(float(shared_ylim[0]), float(shared_ylim[1]))

    spatial_ax.set_aspect(
        style.spatial_aspect,
        adjustable="box",
        anchor=style.spatial_anchor,
    )
    spatial_ax.set_xticks([])
    spatial_ax.set_yticks([])
    for spine in spatial_ax.spines.values():
        spine.set_visible(False)

    # ------------------------------------------------------------------
    # Fixed colorbar geometry for the spatial map.
    # ------------------------------------------------------------------
    cax = fig.add_axes(style.cbar_axes_rect)
    cbar = fig.colorbar(scatter, cax=cax)
    format_case_colorbar(cbar, style)

    # One shared case title keeps the left/right panels visually linked.
    if style.title_y is not None:
        fig.suptitle(
            display_panel_name(panel),
            y=style.title_y,
            fontsize=style.title_fontsize,
            fontweight=style.title_weight,
        )

    save_case_figure_fixed_canvas(
        fig,
        cfg.outdir_path / f"main_case_{gene}_{safe_panel_name(panel)}_combined",
        cfg,
    )
    plt.close(fig)
    return fig


def plot_representative_cases(
    long_df: pd.DataFrame,
    metrics: pd.DataFrame,
    cfg: Config,
    style: CaseStudyFigureStyle = CASE_STUDY_STYLE,
) -> None:
    """Generate one merged case figure per representative gene and method."""
    available_genes = set(long_df["gene"].astype(str))
    genes = [gene for gene in cfg.main_figure_genes if gene in available_genes]
    if not genes:
        warnings.warn("No representative genes are available for case plotting.")
        return

    panels = case_study_panels_present(long_df)
    if not panels:
        warnings.warn("No configured case-study panels are available.")
        return

    for gene in genes:
        gene_df = long_df[
            (long_df["gene"].astype(str) == str(gene))
            & (long_df["panel"].astype(str).isin(panels))
        ].copy()
        if gene_df.empty:
            continue

        layer_order = canonical_layer_order(
            gene_df["layer"].astype(str),
            cfg.preferred_layer_order,
        )
        palette = get_layer_palette(layer_order)
        shared_y_top = representative_y_top(gene_df)
        shared_xlim, shared_ylim = compute_gene_shared_extent(
            gene_df,
            panels,
            style,
        )

        for panel in panels:
            if panel == 'PASTE' or panel =='Axis':
                if gene == "PCP4":
                    shared_y_top=2
                else:
                    shared_y_top = 1.
            else:
                if gene == "PCP4":
                    shared_y_top = 4
                else:
                    shared_y_top = 2.5
            plot_gene_method_case(
                long_df=long_df,
                metrics=metrics,
                gene=gene,
                panel=panel,
                cfg=cfg,
                layer_order=layer_order,
                palette=palette,
                shared_y_top=shared_y_top,
                shared_xlim=shared_xlim,
                shared_ylim=shared_ylim,
                style=style,
            )


def plot_all(long_df: pd.DataFrame, metrics: pd.DataFrame, cfg: Config) -> None:
    if cfg.plot_main_composite:
        print("[Plot] Merged representative case figures")
        plot_representative_cases(long_df, metrics, cfg)
    # if cfg.plot_all_gene_summary:
    #     plot_all_gene_summary(metrics, cfg)
    # if cfg.plot_supplementary_violins:
    #     available = set(long_df["gene"].astype(str))
    #     for gene in cfg.genes_to_plot:
    #         if gene in available:
    #             plot_gene_violin(long_df, metrics, gene, cfg)

# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    mode = validate_config(CFG)
    warn_if_mixed_environment()
    CFG.outdir_path.mkdir(parents=True, exist_ok=True)

    if mode in {"eval", "both"}:
        long_df, metrics = save_metrics(CFG)
    else:
        long_df, metrics = load_plot_inputs(CFG)

    if mode in {"plot", "both"}:
        set_publication_style()
        plot_all(long_df, metrics, CFG)


if __name__ == "__main__":
    main()

