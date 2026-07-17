#!/usr/bin/env python3
"""
DLPFC atlas-level gene-enhancement analysis based on global and adjacent-layer separation.

Purpose
-------
This script evaluates whether multi-slice integration reveals a clearer
population/atlas-level laminar expression structure than individual sections or
naive pooling. It is not designed to reproduce the raw expression distribution
of any single slice.

Run modes
---------
    RUN_MODE = "eval"  : build/load AXIS mean, median, and max atlases; collect
                         marker expression; calculate layer-separation metrics.
    RUN_MODE = "plot"  : regenerate figures from saved CSV outputs only.
    RUN_MODE = "both"  : run evaluation followed by plotting.

Primary metrics
---------------
1. Kruskal-Wallis rank-based layer-separation effect size (eta_H^2):

       eta_H^2 = max(0, (H - k + 1) / (n - k))

   where H is the Kruskal-Wallis statistic, k is the number of observed
   cortical layers, and n is the total number of observations. Larger eta_H^2
   indicates stronger global association between expression and layer identity.

2. Mean adjacent-layer AUROC:

   AUROC is computed for every available neighboring layer pair in anatomical
   order (for example L3-L4, L4-L5, L5-L6, and L6-WM). Because marker direction
   can change across the cortical axis, pairwise AUROC is made direction
   invariant as max(AUC, 1-AUC). Thus 0.5 indicates indistinguishable adjacent
   layers and 1.0 indicates perfect local separation. The reported value is the
   unweighted mean across all available adjacent pairs.

The script also saves pair-level AUROCs, dominant-layer summaries, and gains
relative to the mean single-section baseline.

Important design choice
-----------------------
"Pool (raw)" and "Pool (warped)" are intentionally not shown separately here.
Warping changes coordinates but does not change expression values or existing
layer labels, so their expression-by-layer separation is mathematically
identical. Raw and warped pooling should be separated only in spatial analyses
such as domain detection, boundary recovery, Moran's I, or neighborhood tests.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from scipy.stats import kruskal, wilcoxon
from sklearn.metrics import roc_auc_score
from spatial_arithmetic_utils import aggregate_virtual_slices

# =============================================================================
# USER CONFIGURATION
# =============================================================================

RUN_MODE = "plot"  # one of: "eval", "plot", "both"


@dataclass(frozen=True)
class Config:
    run_mode: str = RUN_MODE

    biological_axis_name: str = "dlpfc_z"
    group_id: int = 1
    grid_size: int = 80
    spatial_bandwidth: int = 1
    reference_axis_value: float = 0.0

    root: str = "/media/huifang/data/registration/DLPFC/huifang"
    h5ad_dir: str = "/media/huifang/data/registration/DLPFC/huifang/h5ad"
    layer_key: str = "layer_guess_reordered"

    sample_ids: Tuple[int, ...] = (0, 1, 2, 3)

    marker_genes: Tuple[str, ...] = (
        "MFGE8", "MOBP", "CUX2", "RAI14", "RORB", "ENC1",
        "PCP4", "KCNIP2", "SYT6", "FOXP2", "MBP",
    )
    # Biologically ordered marker panel used in the main summary figure.
    # Superficial/middle cortex -> deep cortex -> white matter -> broader markers.
    genes_to_plot: Tuple[str, ...] = (
        "CUX2", "RORB", "PCP4", "KCNIP2", "FOXP2",
        "MBP", "MOBP", "ENC1", "MFGE8",
    )

    # Two representative markers shown in the main composite figure.
    main_figure_genes: Tuple[str, ...] = ("PCP4", "MOBP")

    # The main representative panels emphasize conservative atlas summaries.
    # AXIS-max remains in the operator-ablation summary and supplementary plots.
    main_axis_methods: Tuple[str, ...] = ("mean", "median")

    # Superficial-to-deep layer order. Missing layers are omitted automatically.
    preferred_layer_order: Tuple[str, ...] = (
        "Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"
    )

    # Atlas aggregation variants to compare.
    axis_aggregation_methods: Tuple[str, ...] = ("mean", "median", "max")
    rebuild_axis_aggregates: bool = False
    aggregation_output_subdir: str = "aggregation"
    aggregation_gene_chunk_size: int = 256

    # Used only when rebuild_axis_aggregates=False.
    # Example:
    axis_aggregated_h5ads = (
        ("mean",
         "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/dlpfc_z_group1_grid80_spatial1/analysis/aggregation_domain/aggregate_mean_ref_0.000.h5ad"),
        ("median",
         "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/dlpfc_z_group1_grid80_spatial1/analysis/aggregation_domain/aggregate_median_ref_0.000.h5ad"),
        ("max",
         "/media/huifang/data/registration/DLPFC/huifang/virtual_tissue/dlpfc_z_group1_grid80_spatial1/analysis/aggregation_domain/aggregate_max_ref_0.000.h5ad"),
    )
    # axis_aggregated_h5ads: Tuple[Tuple[str, str], ...] = ()

    # Expression handling.
    use_log1p: bool = True

    # Plot outputs.
    save_formats: Tuple[str, ...] = ("png", )
    png_dpi: int = 450
    annotate_violin_metrics: bool = True

    # Main Cell/Nature-style composite figure plus optional supplementary panels.
    plot_main_composite: bool = True
    plot_all_gene_summary: bool = True
    plot_supplementary_violins: bool = True

    # Bootstrap confidence intervals and paired marker-level comparisons.
    bootstrap_iterations: int = 5000
    bootstrap_seed: int = 17

    @property
    def analysis_dir(self) -> Path:
        return (
            Path(self.root)
            / "virtual_tissue"
            / (
                f"{self.biological_axis_name}_group{self.group_id}"
                f"_grid{self.grid_size}_spatial{self.spatial_bandwidth}"
            )
            / "analysis"
        )

    @property
    def outdir(self) -> Path:
        # Use a new directory so old fidelity-oriented CSVs cannot be loaded
        # accidentally in plot mode.
        return self.analysis_dir / "aggregation_layer_separation"

    @property
    def long_csv(self) -> Path:
        return self.outdir / "marker_expression_long.csv"

    @property
    def layer_summary_csv(self) -> Path:
        return self.outdir / "marker_layer_summary.csv"

    @property
    def separation_csv(self) -> Path:
        return self.outdir / "marker_layer_separation_metrics.csv"

    @property
    def adjacent_auc_csv(self) -> Path:
        return self.outdir / "marker_adjacent_layer_auroc.csv"

    @property
    def method_summary_csv(self) -> Path:
        return self.outdir / "marker_layer_separation_method_summary.csv"

    @property
    def numerical_report_md(self) -> Path:
        return self.outdir / "marker_layer_separation_report.md"

    @property
    def gain_statistics_csv(self) -> Path:
        return self.outdir / "marker_layer_separation_gain_statistics.csv"

    @property
    def real_panels(self) -> Tuple[str, ...]:
        return tuple(f"Single {sid}" for sid in sorted(self.sample_ids))

    @property
    def axis_panels(self) -> Tuple[str, ...]:
        return tuple(
            f"AXIS-{canonical_aggregation_method(method)}"
            for method in self.axis_aggregation_methods
        )

    @property
    def panel_order(self) -> Tuple[str, ...]:
        return self.real_panels + ("Pool",) + self.axis_panels


CFG = Config()


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def canonical_aggregation_method(method: str) -> str:
    aliases = {
        "mean": "mean",
        "average": "mean",
        "avg": "mean",
        "median": "median",
        "max": "max",
        "maximum": "max",
    }
    key = str(method).strip().lower()
    if key not in aliases:
        raise ValueError(
            f"Unsupported aggregation method {method!r}; "
            f"choose from mean, median, or max."
        )
    return aliases[key]


def validate_config(cfg: Config) -> str:
    mode = cfg.run_mode.lower().strip()
    if mode not in {"eval", "plot", "both"}:
        raise ValueError(
            f"RUN_MODE must be 'eval', 'plot', or 'both'; received {cfg.run_mode!r}."
        )
    if not cfg.sample_ids:
        raise ValueError("sample_ids cannot be empty.")
    if not cfg.marker_genes:
        raise ValueError("marker_genes cannot be empty.")

    methods = tuple(canonical_aggregation_method(x) for x in cfg.axis_aggregation_methods)
    if len(methods) != len(set(methods)):
        raise ValueError("axis_aggregation_methods contains duplicate methods.")
    if cfg.aggregation_gene_chunk_size < 1:
        raise ValueError("aggregation_gene_chunk_size must be positive.")

    return mode


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
    return f"L{text[len('Layer'):]}" if text.startswith("Layer") else text


def scipy_result_value(result: Any, attr: str, index: int) -> float:
    """Support both named and tuple-like SciPy statistical results."""
    if hasattr(result, attr):
        return float(getattr(result, attr))
    return float(result[index])


def fmt_metric(value: float, digits: int = 2) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


# =============================================================================
# DATA LOADING AND AGGREGATION
# =============================================================================


def read_marker_adata(
    path: Path,
    requested_genes: Sequence[str],
    layer_key: str,
) -> Any:
    """Read only requested marker genes into memory using backed mode."""
    import scanpy as sc  # Lazy import: plot-only mode does not require Scanpy.

    if not path.is_file():
        raise FileNotFoundError(path)

    backed = sc.read_h5ad(path, backed="r")
    try:
        if layer_key not in backed.obs:
            raise KeyError(f"{layer_key!r} is missing from {path}")
        present = [gene for gene in requested_genes if gene in backed.var_names]
        if not present:
            raise ValueError(f"None of the requested marker genes were found in {path}")
        adata = backed[:, present].to_memory()
    finally:
        try:
            backed.file.close()
        except Exception:
            pass
    return adata



def resolve_axis_aggregate_paths(cfg: Config) -> Dict[str, Path]:
    methods = [canonical_aggregation_method(x) for x in cfg.axis_aggregation_methods]

    if not cfg.rebuild_axis_aggregates:
        supplied = {
            canonical_aggregation_method(method): Path(path)
            for method, path in cfg.axis_aggregated_h5ads
        }
        missing_methods = [method for method in methods if method not in supplied]
        if missing_methods:
            raise ValueError(
                "rebuild_axis_aggregates=False, but paths are missing for: "
                + ", ".join(missing_methods)
            )
        for method in methods:
            if not supplied[method].is_file():
                raise FileNotFoundError(supplied[method])
        return {method: supplied[method] for method in methods}

    paths: Dict[str, Path] = {}
    for method in methods:
        output_name = f"aggregate_{method}_ref_{float(cfg.reference_axis_value):.3f}.h5ad"
        print(f"[Aggregate] AXIS-{method}")
        output_path = aggregate_virtual_slices(
            input_dir=str(cfg.analysis_dir),
            biological_axis_name=cfg.biological_axis_name,
            reference_axis_value=cfg.reference_axis_value,
            aggregation=method,
            output_subdir=cfg.aggregation_output_subdir,
            output_name=output_name,
            gene_chunk_size=cfg.aggregation_gene_chunk_size,
        )
        path = Path(output_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"aggregate_virtual_slices returned a missing file: {path}"
            )
        paths[method] = path
    return paths


def adata_to_long_df(
    adata: Any,
    panel: str,
    genes: Sequence[str],
    layer_key: str,
    use_log1p: bool,
) -> pd.DataFrame:
    present = [gene for gene in genes if gene in adata.var_names]
    if not present:
        return pd.DataFrame()

    X = adata[:, present].X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    layers = adata.obs[layer_key].astype("string").to_numpy()
    expression_wide = pd.DataFrame(X, columns=present)
    expression_wide["layer"] = layers

    long_df = expression_wide.melt(
        id_vars="layer",
        var_name="gene",
        value_name="value_raw",
    )
    long_df = long_df[
        long_df["layer"].notna()
        & ~long_df["layer"].astype(str).str.lower().isin({"nan", "none", "<na>"})
    ].copy()

    long_df["panel"] = panel
    long_df["value_raw"] = long_df["value_raw"].astype(np.float32)

    if use_log1p:
        long_df["value_plot"] = np.log1p(
            np.clip(long_df["value_raw"].to_numpy(dtype=float), 0.0, None)
        ).astype(np.float32)
        long_df["y_label"] = "log1p(counts)"
    else:
        long_df["value_plot"] = long_df["value_raw"]
        long_df["y_label"] = "counts"

    return long_df[["gene", "panel", "layer", "value_raw", "value_plot", "y_label"]]


def collect_expression_long(cfg: Config) -> pd.DataFrame:
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    single_adatas: Dict[int, Any] = {}
    for sid in sorted(cfg.sample_ids):
        path = Path(cfg.h5ad_dir) / f"{cfg.group_id}_{sid}.h5ad"
        print(f"[Load] real slice {sid}: {path}")
        single_adatas[sid] = read_marker_adata(
            path=path,
            requested_genes=cfg.marker_genes,
            layer_key=cfg.layer_key,
        )

    axis_paths = resolve_axis_aggregate_paths(cfg)
    axis_adatas: Dict[str, Any] = {}
    for method, path in axis_paths.items():
        print(f"[Load] AXIS-{method}: {path}")
        axis_adatas[method] = read_marker_adata(
            path=path,
            requested_genes=cfg.marker_genes,
            layer_key=cfg.layer_key,
        )

    all_adatas = list(single_adatas.values()) + list(axis_adatas.values())
    common_genes = [
        gene
        for gene in cfg.marker_genes
        if all(gene in ad.var_names for ad in all_adatas)
    ]
    missing = [gene for gene in cfg.marker_genes if gene not in common_genes]
    if missing:
        warnings.warn(
            "Genes absent from at least one real/AXIS object were skipped: "
            + ", ".join(missing)
        )
    if not common_genes:
        raise ValueError("No marker gene is shared across all real and AXIS objects.")

    single_parts: List[pd.DataFrame] = []
    for sid in sorted(single_adatas):
        single_parts.append(
            adata_to_long_df(
                single_adatas[sid],
                panel=f"Single {sid}",
                genes=common_genes,
                layer_key=cfg.layer_key,
                use_log1p=cfg.use_log1p,
            )
        )

    real_long = pd.concat(single_parts, ignore_index=True)

    # Pooled expression is simply the union of observations across real slices.
    pool_long = real_long.copy()
    pool_long["panel"] = "Pool"

    parts: List[pd.DataFrame] = [real_long, pool_long]
    for method in [canonical_aggregation_method(x) for x in cfg.axis_aggregation_methods]:
        parts.append(
            adata_to_long_df(
                axis_adatas[method],
                panel=f"AXIS-{method}",
                genes=common_genes,
                layer_key=cfg.layer_key,
                use_log1p=cfg.use_log1p,
            )
        )

    long_df = pd.concat(parts, ignore_index=True)
    panel_categories = [p for p in cfg.panel_order if p in set(long_df["panel"])]
    long_df["panel"] = pd.Categorical(
        long_df["panel"], categories=panel_categories, ordered=True
    )

    long_df.to_csv(cfg.long_csv, index=False)
    print(f"[OK] expression table: {cfg.long_csv}")
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
        long_df.groupby(["gene", "panel", "layer"], observed=True)
        .agg(
            n=("value_plot", "size"),
            mean_expression=("value_plot", "mean"),
            median_expression=("value_plot", "median"),
            std_expression=("value_plot", "std"),
            q25_expression=("value_plot", lambda x: x.quantile(0.25)),
            q75_expression=("value_plot", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )
    summary["layer"] = pd.Categorical(
        summary["layer"], categories=layer_order, ordered=True
    )
    summary = summary.sort_values(["gene", "panel", "layer"]).reset_index(drop=True)
    summary.to_csv(cfg.layer_summary_csv, index=False)
    print(f"[OK] layer summary: {cfg.layer_summary_csv}")
    return summary, layer_order


def kruskal_eta_h_squared(
    groups: Sequence[np.ndarray],
) -> Tuple[float, float, float, int, int]:
    """Kruskal-Wallis eta-squared based on the tie-corrected H statistic."""
    valid = [np.asarray(x, dtype=float) for x in groups if len(x) > 0]
    n_total = int(sum(len(x) for x in valid))
    k = len(valid)

    if k < 2 or n_total <= k:
        return np.nan, np.nan, np.nan, n_total, k

    try:
        result = kruskal(*valid)
        h_stat = scipy_result_value(result, attr="statistic", index=0)
        p_value = scipy_result_value(result, attr="pvalue", index=1)
    except ValueError as exc:
        # SciPy raises when every value in every group is identical. In that
        # case there is no layer separation by definition.
        if "identical" not in str(exc).lower():
            raise
        h_stat, p_value = 0.0, 1.0

    eta_h_squared = max(0.0, (h_stat - k + 1.0) / (n_total - k))
    return float(eta_h_squared), float(h_stat), float(p_value), n_total, k


def direction_invariant_auroc(
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> Tuple[float, float, str]:
    """
    Return direction-invariant AUROC for one adjacent-layer pair.

    ``raw_auc`` treats the right-hand/deeper layer as the positive class.
    ``separation_auc`` is max(raw_auc, 1 - raw_auc), so it measures local
    distinguishability regardless of which layer has higher expression.
    """
    left = np.asarray(left_values, dtype=float)
    right = np.asarray(right_values, dtype=float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]

    if len(left) == 0 or len(right) == 0:
        return np.nan, np.nan, "NA"

    scores = np.concatenate([left, right])
    labels = np.concatenate([
        np.zeros(len(left), dtype=np.int8),
        np.ones(len(right), dtype=np.int8),
    ])

    # All tied values imply chance-level discrimination.
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
    value_col: str = "value_plot",
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Calculate AUROC for every available adjacent pair in layer order."""
    pair_rows: List[Dict[str, Any]] = []

    for left_layer, right_layer in zip(layer_order[:-1], layer_order[1:]):
        left = sub.loc[
            sub["layer"].astype(str) == left_layer,
            value_col,
        ].dropna().to_numpy(dtype=float)
        right = sub.loc[
            sub["layer"].astype(str) == right_layer,
            value_col,
        ].dropna().to_numpy(dtype=float)

        if len(left) == 0 or len(right) == 0:
            continue

        separation_auc, raw_auc, higher_side = direction_invariant_auroc(left, right)
        higher_layer = right_layer if higher_side == "right" else left_layer
        pair_rows.append(
            {
                "layer_left": left_layer,
                "layer_right": right_layer,
                "n_left": int(len(left)),
                "n_right": int(len(right)),
                "raw_auc_right_positive": raw_auc,
                "adjacent_separation_auroc": separation_auc,
                "higher_expression_layer": higher_layer,
            }
        )

    aucs = np.asarray(
        [row["adjacent_separation_auroc"] for row in pair_rows],
        dtype=float,
    )
    aucs = aucs[np.isfinite(aucs)]
    summary = {
        "adjacent_auc_mean": float(np.mean(aucs)) if len(aucs) else np.nan,
        "adjacent_auc_median": float(np.median(aucs)) if len(aucs) else np.nan,
        "adjacent_auc_min": float(np.min(aucs)) if len(aucs) else np.nan,
        "adjacent_auc_max": float(np.max(aucs)) if len(aucs) else np.nan,
        "n_adjacent_pairs": int(len(aucs)),
    }
    return summary, pair_rows


def compute_layer_separation_metrics(
    long_df: pd.DataFrame,
    layer_summary: pd.DataFrame,
    layer_order: Sequence[str],
    cfg: Config,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    adjacent_pair_rows: List[Dict[str, Any]] = []
    present_panels = [p for p in cfg.panel_order if p in set(long_df["panel"].astype(str))]
    present_genes = [g for g in cfg.marker_genes if g in set(long_df["gene"])]

    mean_lookup = layer_summary.set_index(["gene", "panel", "layer"])[
        "mean_expression"
    ].to_dict()

    for gene in present_genes:
        for panel in present_panels:
            sub = long_df[
                (long_df["gene"] == gene)
                & (long_df["panel"].astype(str) == panel)
            ]
            if sub.empty:
                continue

            groups: List[np.ndarray] = []
            observed_layers: List[str] = []
            layer_means: List[float] = []

            for layer in layer_order:
                values = sub.loc[
                    sub["layer"].astype(str) == layer,
                    "value_plot",
                ].dropna().to_numpy(dtype=float)
                if len(values) > 0:
                    groups.append(values)
                    observed_layers.append(layer)
                    layer_means.append(
                        float(mean_lookup.get((gene, panel, layer), np.mean(values)))
                    )

            eta_h_sq, h_stat, p_value, n_total, k = kruskal_eta_h_squared(groups)
            auc_summary, pair_rows = compute_adjacent_layer_aurocs(
                sub=sub,
                layer_order=layer_order,
                value_col="value_plot",
            )
            for pair_row in pair_rows:
                adjacent_pair_rows.append({"gene": gene, "panel": panel, **pair_row})

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
    adjacent_df = pd.DataFrame(adjacent_pair_rows)
    if not adjacent_df.empty:
        panel_rank = {panel: i for i, panel in enumerate(cfg.panel_order)}
        gene_rank = {gene: i for i, gene in enumerate(cfg.marker_genes)}
        layer_rank = {layer: i for i, layer in enumerate(layer_order)}
        adjacent_df["_panel_rank"] = adjacent_df["panel"].map(panel_rank)
        adjacent_df["_gene_rank"] = adjacent_df["gene"].map(gene_rank)
        adjacent_df["_layer_rank"] = adjacent_df["layer_left"].map(layer_rank)
        adjacent_df = adjacent_df.sort_values(
            ["_gene_rank", "_panel_rank", "_layer_rank"]
        ).drop(columns=["_gene_rank", "_panel_rank", "_layer_rank"])
    adjacent_df.to_csv(cfg.adjacent_auc_csv, index=False)
    print(f"[OK] adjacent-layer AUROC table: {cfg.adjacent_auc_csv}")

    # Single-section baselines quantify atlas gain without requiring fidelity to
    # the raw distribution of one section.
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
    metrics["eta_h_ratio_vs_single_mean"] = np.where(
        metrics["single_eta_h_mean"] > 1e-12,
        metrics["eta_h_squared"] / metrics["single_eta_h_mean"],
        np.nan,
    )
    metrics["eta_h_exceeds_best_single"] = (
        metrics["eta_h_squared"] > metrics["single_eta_h_max"]
    )
    metrics["delta_adjacent_auc_vs_single_mean"] = (
        metrics["adjacent_auc_mean"] - metrics["single_adjacent_auc_mean"]
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

    metrics.to_csv(cfg.separation_csv, index=False)
    print(f"[OK] layer-separation metrics: {cfg.separation_csv}")
    return metrics


def summarize_methods(metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    candidate_panels = [
        panel for panel in ("Pool",) + cfg.axis_panels
        if panel in set(metrics["panel"])
    ]
    sub = metrics[metrics["panel"].isin(candidate_panels)].copy()

    rows: List[Dict[str, Any]] = []
    for panel in candidate_panels:
        panel_df = sub[sub["panel"] == panel]
        eta = panel_df["eta_h_squared"].dropna().to_numpy(dtype=float)
        eta_delta = panel_df["delta_eta_h_vs_single_mean"].dropna().to_numpy(dtype=float)
        auc = panel_df["adjacent_auc_mean"].dropna().to_numpy(dtype=float)
        auc_delta = panel_df["delta_adjacent_auc_vs_single_mean"].dropna().to_numpy(dtype=float)
        eta_exceeds = panel_df["eta_h_exceeds_best_single"].dropna().astype(bool).to_numpy()
        auc_exceeds = panel_df["adjacent_auc_exceeds_best_single"].dropna().astype(bool).to_numpy()

        rows.append(
            {
                "panel": panel,
                "n_genes": int(panel_df["gene"].nunique()),
                "mean_eta_h_squared": float(np.mean(eta)) if len(eta) else np.nan,
                "median_eta_h_squared": float(np.median(eta)) if len(eta) else np.nan,
                "mean_delta_eta_h_vs_single": float(np.mean(eta_delta)) if len(eta_delta) else np.nan,
                "median_delta_eta_h_vs_single": float(np.median(eta_delta)) if len(eta_delta) else np.nan,
                "mean_adjacent_auc": float(np.mean(auc)) if len(auc) else np.nan,
                "median_adjacent_auc": float(np.median(auc)) if len(auc) else np.nan,
                "mean_delta_adjacent_auc_vs_single": float(np.mean(auc_delta)) if len(auc_delta) else np.nan,
                "median_delta_adjacent_auc_vs_single": float(np.median(auc_delta)) if len(auc_delta) else np.nan,
                "n_genes_eta_above_best_single": int(np.sum(eta_exceeds)) if len(eta_exceeds) else 0,
                "fraction_genes_eta_above_best_single": float(np.mean(eta_exceeds)) if len(eta_exceeds) else np.nan,
                "n_genes_auc_above_best_single": int(np.sum(auc_exceeds)) if len(auc_exceeds) else 0,
                "fraction_genes_auc_above_best_single": float(np.mean(auc_exceeds)) if len(auc_exceeds) else np.nan,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(cfg.method_summary_csv, index=False)
    print(f"[OK] method summary: {cfg.method_summary_csv}")

    definitions = (
        "# DLPFC atlas-level laminar-enhancement report\n\n"
        "The analysis evaluates population/atlas laminar enhancement rather than "
        "fidelity to the raw distribution of any individual section.\n\n"
        "- **eta_H^2**: Kruskal-Wallis rank-based global layer-separation effect "
        "size, calculated across all observed cortical layers.\n"
        "- **mean adjacent-layer AUROC**: direction-invariant AUROC averaged over "
        "every available neighboring pair in anatomical order. A value of 0.5 "
        "indicates indistinguishable adjacent layers and 1.0 indicates perfect "
        "local separation.\n"
        "- **delta vs single**: candidate metric minus the mean metric across "
        "individual real sections for the same gene.\n"
        "- Pooled raw and warped observations are represented by one `Pool` "
        "condition because coordinate warping cannot change expression-by-layer "
        "metrics when labels and values are unchanged.\n\n"
    )
    try:
        table = summary.to_markdown(index=False, floatfmt=".3f")
    except ImportError:
        table = summary.to_csv(index=False)
    cfg.numerical_report_md.write_text(definitions + table + "\n", encoding="utf-8")
    print(f"[OK] Markdown report: {cfg.numerical_report_md}")
    return summary


def run_evaluation(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_df = collect_expression_long(cfg)
    layer_summary, layer_order = compute_layer_summary(long_df, cfg)
    metrics = compute_layer_separation_metrics(long_df, layer_summary, layer_order, cfg)
    summarize_methods(metrics, cfg)
    return long_df, metrics


# =============================================================================
# PUBLICATION-STYLE PLOTTING
# =============================================================================
# =============================================================================
# PUBLICATION-STYLE PLOTTING
# =============================================================================


def set_publication_style() -> None:
    """Use restrained typography and line weights suitable for journal figures."""
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.titlesize": 9.1,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def get_layer_palette(layer_order: Sequence[str]) -> Dict[str, str]:
    """Muted sequential green palette for cortical layers."""
    base = {
        "Layer1": "#EDF6F0",
        "Layer2": "#DCEDE3",
        "Layer3": "#C7E1D1",
        "Layer4": "#A9D2BB",
        "Layer5": "#83BFA1",
        "Layer6": "#5AA787",
        "WM":     "#357A63",
    }

    # Muted green fallback for any unexpected labels
    fallback = sns.color_palette(
        "light:#357A63",
        n_colors=max(2, len(layer_order) + 2),
    )[2:]

    return {
        layer: base.get(
            str(layer),
            mpl.colors.to_hex(fallback[i % len(fallback)])
        )
        for i, layer in enumerate(layer_order)
    }

def get_panel_color(panel: str) -> str:
    colors = {
        "Pool": "#356E9B",
        "AXIS-mean": "#E69F00",
        "AXIS-median": "#CC79A7",
        "AXIS-max": "#C43C39",
    }
    if panel.startswith("Single"):
        return "#8F8F8F"
    return colors.get(panel, "#6F6F6F")


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


def violinplot_compat(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    layer_order: Sequence[str],
    palette: Mapping[str, str],
) -> None:
    """Version-compatible violin + compact internal box plot."""
    common_kwargs = dict(
        data=data,
        x="layer",
        y="value_plot",
        order=list(layer_order),
        inner=None,
        cut=0,
        linewidth=0.68,
        saturation=0.95,
        ax=ax,
    )
    try:
        sns.violinplot(
            **common_kwargs,
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
            **common_kwargs,
            palette=[palette[layer] for layer in layer_order],
            scale="width",
        )

    sns.boxplot(
        data=data,
        x="layer",
        y="value_plot",
        order=list(layer_order),
        width=0.14,
        showfliers=False,
        showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": "#222222", "linewidth": 0.62},
        medianprops={"color": "#111111", "linewidth": 0.9},
        whiskerprops={"color": "#222222", "linewidth": 0.62},
        capprops={"color": "#222222", "linewidth": 0.62},
        ax=ax,
    )
    for collection in ax.collections:
        collection.set_edgecolor("#4A4A4A")


def bootstrap_median_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Return median and percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    median = float(np.median(values))
    if len(values) == 1 or n_boot < 2:
        return median, median, median

    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    boot = np.median(samples, axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha])
    return median, float(low), float(high)


def paired_wilcoxon_p(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Paired marker-level Wilcoxon test, compatible across SciPy versions."""
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    mask = np.isfinite(candidate) & np.isfinite(reference)
    candidate, reference = candidate[mask], reference[mask]
    if len(candidate) < 2:
        return np.nan
    diff = candidate - reference
    if np.all(np.abs(diff) < 1e-12):
        return 1.0
    try:
        result = wilcoxon(candidate, reference, alternative="greater")
    except TypeError:
        result = wilcoxon(candidate, reference)
    return scipy_result_value(result, attr="pvalue", index=1)


def format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "P = NA"
    if p_value < 1e-3:
        return "P < 0.001"
    return f"P = {p_value:.3f}"


def candidate_panels_present(metrics: pd.DataFrame, cfg: Config) -> List[str]:
    return [
        panel for panel in ("Pool",) + cfg.axis_panels
        if panel in set(metrics["panel"].astype(str))
    ]


def ordered_summary_genes(metrics: pd.DataFrame, cfg: Config) -> List[str]:
    available = set(metrics["gene"].astype(str))
    return [gene for gene in cfg.genes_to_plot if gene in available]


def compute_gain_statistics(
    metrics: pd.DataFrame,
    cfg: Config,
    genes: Sequence[str],
) -> pd.DataFrame:
    """Bootstrap median eta_H^2 gain and paired comparisons against pooling."""
    candidates = candidate_panels_present(metrics, cfg)
    delta = (
        metrics[
            metrics["panel"].isin(candidates)
            & metrics["gene"].isin(genes)
        ]
        .pivot(index="gene", columns="panel", values="delta_eta_h_vs_single_mean")
        .reindex(genes)
    )

    rows: List[Dict[str, Any]] = []
    pool_values = delta["Pool"].to_numpy(dtype=float) if "Pool" in delta else None
    for i, panel in enumerate(candidates):
        values = delta[panel].to_numpy(dtype=float)
        median, low, high = bootstrap_median_ci(
            values,
            n_boot=cfg.bootstrap_iterations,
            seed=cfg.bootstrap_seed + i,
        )
        if panel == "Pool" or pool_values is None:
            p_value = np.nan
        else:
            p_value = paired_wilcoxon_p(values, pool_values)
        rows.append(
            {
                "panel": panel,
                "n_genes": int(np.sum(np.isfinite(values))),
                "median_delta_eta_h": median,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "wilcoxon_greater_vs_pool_p": p_value,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(cfg.gain_statistics_csv, index=False)
    print(f"[OK] gain statistics: {cfg.gain_statistics_csv}")
    return out


def style_axis(ax: mpl.axes.Axes, grid_axis: Optional[str] = "y") -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color="#E8E8E8", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def draw_workflow_panel(ax: mpl.axes.Axes) -> None:
    """Compact conceptual panel: sections -> atlas -> laminar structure."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 1.03, "a", fontsize=10.5, fontweight="bold", va="bottom")
    ax.text(0.08, 1.03, "Population atlas analysis", fontsize=9.2, fontweight="bold", va="bottom")

    # Stacked native sections.
    slice_x, slice_y = 0.04, 0.40
    for i, shift in enumerate((0.08, 0.055, 0.03, 0.005)):
        rect = mpl.patches.FancyBboxPatch(
            (slice_x + shift, slice_y + shift),
            0.20,
            0.30,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=0.8,
            edgecolor="#777777",
            facecolor="#F2F2F2" if i < 3 else "#E5E5E5",
            zorder=4 - i,
        )
        ax.add_patch(rect)
        for yy, color in zip((0.07, 0.14, 0.21), ("#4E79A7", "#F28E2B", "#3E9651")):
            ax.plot(
                [slice_x + shift + 0.02, slice_x + shift + 0.18],
                [slice_y + shift + yy, slice_y + shift + yy],
                color=color,
                linewidth=2.0,
                solid_capstyle="round",
                zorder=6,
            )
    ax.text(0.17, 0.20, "Native sections", ha="center", fontsize=7.8, fontweight="bold")

    ax.annotate(
        "",
        xy=(0.47, 0.57),
        xytext=(0.34, 0.57),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.1),
    )

    atlas = mpl.patches.FancyBboxPatch(
        (0.48, 0.36),
        0.22,
        0.40,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=1.1,
        edgecolor="#CC79A7",
        facecolor="#FBF3F8",
    )
    ax.add_patch(atlas)
    for yy, color, width in (
        (0.44, "#4E79A7", 0.14),
        (0.52, "#F28E2B", 0.16),
        (0.60, "#3E9651", 0.18),
        (0.68, "#D95852", 0.15),
    ):
        ax.plot([0.52, 0.52 + width], [yy, yy], color=color, lw=3.1, solid_capstyle="round")
    ax.text(0.59, 0.20, "AXIS atlas", ha="center", fontsize=7.8, fontweight="bold", color="#B25F91")

    ax.annotate(
        "",
        xy=(0.84, 0.57),
        xytext=(0.72, 0.57),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.1),
    )

    circle = mpl.patches.Circle((0.91, 0.57), 0.105, facecolor="#F4FAF5", edgecolor="#3E9651", linewidth=1.1)
    ax.add_patch(circle)
    ax.text(0.91, 0.61, r"$\eta_H^2$", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#2E7D3C")
    ax.text(0.91, 0.51, "+ adj. AUC", ha="center", va="center", fontsize=6.3, color="#2E7D3C")
    ax.text(0.90, 0.20, "Laminar structure", ha="center", fontsize=7.0, fontweight="bold")

    ax.text(
        0.5,
        0.02,
        "Population structure—not reconstruction of one noisy section",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#555555",
        style="italic",
    )


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
    """Categorical point-and-range summary for one layer-separation metric."""
    candidates = candidate_panels_present(metrics, cfg)
    x = np.arange(len(genes), dtype=float)

    baseline = metrics.drop_duplicates("gene").set_index("gene")
    real_low = baseline[single_min_col].reindex(genes).to_numpy(dtype=float)
    real_high = baseline[single_max_col].reindex(genes).to_numpy(dtype=float)
    real_mean = baseline[single_mean_col].reindex(genes).to_numpy(dtype=float)

    ax.vlines(
        x,
        real_low,
        real_high,
        color="#C7C7C7",
        linewidth=4.8,
        zorder=1,
        label="Single-section range",
    )
    ax.scatter(
        x,
        real_mean,
        s=18,
        color="#7D7D7D",
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
        label="Single-section mean",
    )

    offsets = {
        "Pool": -0.24,
        "AXIS-mean": -0.08,
        "AXIS-median": 0.08,
        "AXIS-max": 0.24,
    }
    markers = {"Pool": "o", "AXIS-mean": "s", "AXIS-median": "^", "AXIS-max": "D"}
    for panel in candidates:
        values = (
            metrics[metrics["panel"] == panel]
            .set_index("gene")[metric_col]
            .reindex(genes)
            .to_numpy(dtype=float)
        )
        ax.scatter(
            x + offsets.get(panel, 0.0),
            values,
            s=28,
            marker=markers.get(panel, "o"),
            color=get_panel_color(panel),
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
            label=panel,
        )

    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=38, ha="right", rotation_mode="anchor")
    ax.set_ylim(*y_limits)
    style_axis(ax)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        order = [
            i
            for name in (
                "Single-section range",
                "Single-section mean",
                "Pool",
                "AXIS-mean",
                "AXIS-median",
                "AXIS-max",
            )
            for i, label in enumerate(labels)
            if label == name
        ]
        ax.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            frameon=False,
            ncol=3,
            loc="upper left",
            borderaxespad=0.2,
            handletextpad=0.45,
            columnspacing=0.9,
        )


def draw_marker_summary_panel(
    ax: mpl.axes.Axes,
    metrics: pd.DataFrame,
    cfg: Config,
    genes: Sequence[str],
    panel_letter: str = "b",
    show_legend: bool = True,
) -> None:
    values = metrics.loc[metrics["gene"].isin(genes), "eta_h_squared"].to_numpy(dtype=float)
    max_value = float(np.nanmax(values)) if np.any(np.isfinite(values)) else 0.7
    draw_metric_summary_panel(
        ax=ax,
        metrics=metrics,
        cfg=cfg,
        genes=genes,
        metric_col="eta_h_squared",
        single_min_col="single_eta_h_min",
        single_max_col="single_eta_h_max",
        single_mean_col="single_eta_h_mean",
        title=f"{panel_letter}  Global cortical-layer separation",
        ylabel=r"Kruskal–Wallis effect size, $\eta_H^2$",
        y_limits=(0.0, min(1.0, max(0.70, max_value * 1.14))),
        show_legend=show_legend,
    )


def draw_adjacent_auc_summary_panel(
    ax: mpl.axes.Axes,
    metrics: pd.DataFrame,
    cfg: Config,
    genes: Sequence[str],
    panel_letter: str = "c",
    show_legend: bool = False,
) -> None:
    draw_metric_summary_panel(
        ax=ax,
        metrics=metrics,
        cfg=cfg,
        genes=genes,
        metric_col="adjacent_auc_mean",
        single_min_col="single_adjacent_auc_min",
        single_max_col="single_adjacent_auc_max",
        single_mean_col="single_adjacent_auc_mean",
        title=f"{panel_letter}  Mean adjacent-layer discrimination",
        ylabel="Mean adjacent-layer AUROC",
        y_limits=(0.48, 1.01),
        show_legend=show_legend,
    )
    ax.axhline(0.5, color="#8A8A8A", linewidth=0.7, linestyle="--", zorder=0)


def draw_gain_panel(
    ax: mpl.axes.Axes,
    metrics: pd.DataFrame,
    gain_stats: pd.DataFrame,
    cfg: Config,
    genes: Sequence[str],
) -> None:
    candidates = candidate_panels_present(metrics, cfg)
    delta = (
        metrics[
            metrics["panel"].isin(candidates)
            & metrics["gene"].isin(genes)
        ]
        .pivot(index="gene", columns="panel", values="delta_eta_h_vs_single_mean")
        .reindex(genes)
    )
    positions = np.arange(len(candidates), dtype=float)

    # Light paired trajectories show that the same marker genes are compared.
    for gene in genes:
        values = delta.loc[gene, candidates].to_numpy(dtype=float)
        ax.plot(positions, values, color="#D4D4D4", linewidth=0.65, alpha=0.72, zorder=1)

    for i, panel in enumerate(candidates):
        values = delta[panel].to_numpy(dtype=float)
        finite = np.isfinite(values)
        rng = np.random.default_rng(cfg.bootstrap_seed + 100 + i)
        jitter = rng.uniform(-0.085, 0.085, size=np.sum(finite))
        ax.scatter(
            np.full(np.sum(finite), positions[i]) + jitter,
            values[finite],
            s=20,
            color=get_panel_color(panel),
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

        row = gain_stats[gain_stats["panel"] == panel]
        if not row.empty:
            median = float(row["median_delta_eta_h"].iloc[0])
            low = float(row["bootstrap_ci_low"].iloc[0])
            high = float(row["bootstrap_ci_high"].iloc[0])
            ax.vlines(positions[i], low, high, color="#111111", linewidth=1.8, zorder=4)
            ax.scatter(
                positions[i],
                median,
                s=36,
                marker="D",
                facecolor="white",
                edgecolor="#111111",
                linewidth=0.9,
                zorder=5,
            )

    ax.axhline(0.0, color="#777777", linewidth=0.75, linestyle="--", zorder=0)
    ax.set_title("c  Gain over single-section analysis", loc="left", fontweight="bold", pad=5)
    ax.set_ylabel(r"$\Delta\eta_H^2$ vs. mean single section")
    display_labels = {
        "Pool": "Pool",
        "AXIS-mean": "AXIS\nmean",
        "AXIS-median": "AXIS\nmedian",
        "AXIS-max": "AXIS\nmax",
    }
    ax.set_xticks(positions)
    ax.set_xticklabels([display_labels.get(x, x) for x in candidates], rotation=0, ha="center")
    style_axis(ax)

    # Paired one-sided Wilcoxon P values are written compactly above AXIS groups.
    y_low, y_high = ax.get_ylim()
    span = y_high - y_low
    ax.set_ylim(y_low, y_high + 0.12 * span)
    y_text = y_high + 0.025 * span
    for i, panel in enumerate(candidates):
        if panel == "Pool":
            continue
        row = gain_stats[gain_stats["panel"] == panel]
        p_value = float(row["wilcoxon_greater_vs_pool_p"].iloc[0]) if not row.empty else np.nan
        ax.text(
            positions[i],
            y_text,
            format_p_value(p_value).replace("P = ", "P="),
            ha="center",
            va="bottom",
            fontsize=6.3,
            color="#444444",
        )
    ax.text(
        0.02,
        0.02,
        "diamond: median; line: 95% bootstrap CI",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#555555",
    )


def representative_panels(cfg: Config) -> List[str]:
    panels = list(cfg.real_panels) + ["Pool"]
    panels.extend(
        f"AXIS-{canonical_aggregation_method(method)}"
        for method in cfg.main_axis_methods
    )
    return panels


def draw_representative_violin_row(
    fig: mpl.figure.Figure,
    subspec: mpl.gridspec.SubplotSpec,
    long_df: pd.DataFrame,
    metrics: pd.DataFrame,
    gene: str,
    cfg: Config,
    panel_letter: str,
    descriptor: str,
) -> List[mpl.axes.Axes]:
    panels = [
        panel for panel in representative_panels(cfg)
        if panel in set(long_df[long_df["gene"] == gene]["panel"].astype(str))
    ]
    if not panels:
        return []

    gene_df = long_df[long_df["gene"] == gene].copy()
    layer_order = canonical_layer_order(gene_df["layer"].astype(str), cfg.preferred_layer_order)
    palette = get_layer_palette(layer_order)
    y_max = float(np.nanmax(gene_df[gene_df["panel"].astype(str).isin(panels)]["value_plot"].to_numpy(dtype=float)))
    y_top = max(y_max * 1.06, y_max + 0.08)

    nested = subspec.subgridspec(1, len(panels), wspace=0.10)
    axes: List[mpl.axes.Axes] = []
    metric_lookup = metrics[metrics["gene"] == gene].set_index("panel")

    for i, panel in enumerate(panels):
        ax = fig.add_subplot(nested[0, i], sharey=axes[0] if axes else None)
        axes.append(ax)
        sub = gene_df[gene_df["panel"].astype(str) == panel].copy()
        sub["layer"] = pd.Categorical(sub["layer"].astype(str), categories=layer_order, ordered=True)
        violinplot_compat(ax, sub, layer_order, palette)

        color = get_panel_color(panel)
        ax.set_title(panel, color=color if not panel.startswith("Single") else "#222222", fontweight="bold", pad=4)
        ax.set_xlabel("")
        ax.set_ylim(0.0, y_top)
        ax.set_xticks(np.arange(len(layer_order)))
        ax.set_xticklabels(
            [short_layer_label(layer) for layer in layer_order],
            rotation=33,
            ha="right",
            rotation_mode="anchor",
        )
        if i == 0:
            ax.set_ylabel("Expression, log1p(counts)" if cfg.use_log1p else "Expression")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        style_axis(ax)

        if not panel.startswith("Single"):
            ax.spines["bottom"].set_color(color)
            ax.spines["bottom"].set_linewidth(1.4)
            ax.set_facecolor(mpl.colors.to_rgba(color, 0.025))

        if panel in metric_lookup.index:
            eta_h = float(metric_lookup.loc[panel, "eta_h_squared"])
            adjacent_auc = float(metric_lookup.loc[panel, "adjacent_auc_mean"])
            ax.text(
                0.035,
                0.965,
                f"$\\eta_H^2$={fmt_metric(eta_h)}\nAdj. AUROC={fmt_metric(adjacent_auc)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.6,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.82),
            )

    bbox = subspec.get_position(fig)
    fig.text(bbox.x0, bbox.y1 + 0.022, panel_letter, fontsize=10.5, fontweight="bold", ha="left", va="bottom")
    fig.text(
        bbox.x0 + 0.021,
        bbox.y1 + 0.022,
        f"{gene}: {descriptor}",
        fontsize=9.4,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    return axes


def plot_main_result_figure(
    long_df: pd.DataFrame,
    metrics: pd.DataFrame,
    cfg: Config,
) -> Optional[mpl.figure.Figure]:
    genes = ordered_summary_genes(metrics, cfg)
    if not genes:
        return None

    main_genes = [gene for gene in cfg.main_figure_genes if gene in set(long_df["gene"])]
    if len(main_genes) < 2:
        warnings.warn("Fewer than two representative genes are available for the main figure.")

    fig = plt.figure(figsize=(16.4, 9.4))
    outer = fig.add_gridspec(
        3,
        14,
        height_ratios=(1.0, 1.22, 1.22),
        left=0.045,
        right=0.995,
        top=0.94,
        bottom=0.078,
        hspace=0.58,
        wspace=0.82,
    )

    ax_workflow = fig.add_subplot(outer[0, 0:3])
    ax_eta = fig.add_subplot(outer[0, 3:9])
    ax_auc = fig.add_subplot(outer[0, 9:14])
    draw_workflow_panel(ax_workflow)
    draw_marker_summary_panel(ax_eta, metrics, cfg, genes, panel_letter="b", show_legend=True)
    draw_adjacent_auc_summary_panel(ax_auc, metrics, cfg, genes, panel_letter="c", show_legend=False)

    descriptors = {
        "PCP4": "L5-associated laminar marker",
        "MBP": "cortex-to-white-matter marker",
    }
    if len(main_genes) >= 1:
        draw_representative_violin_row(
            fig,
            outer[1, :],
            long_df,
            metrics,
            main_genes[0],
            cfg,
            "d",
            descriptors.get(main_genes[0], "representative laminar marker"),
        )
    if len(main_genes) >= 2:
        draw_representative_violin_row(
            fig,
            outer[2, :],
            long_df,
            metrics,
            main_genes[1],
            cfg,
            "e",
            descriptors.get(main_genes[1], "representative laminar marker"),
        )

    fig.suptitle(
        "AXIS integration reveals population-level laminar marker structure",
        fontsize=14.2,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.022,
        (
            r"$\eta_H^2$: global Kruskal–Wallis rank-based layer separation; "
            "adjacent-layer AUROC: mean direction-invariant discrimination across every neighboring layer pair."
        ),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#444444",
    )

    stem = cfg.outdir / "main_figure_laminar_gene_enhancement"
    save_figure(fig, stem, cfg)
    plt.close(fig)
    return fig


def plot_gene_violin(
    long_df: pd.DataFrame,
    metrics: pd.DataFrame,
    gene: str,
    cfg: Config,
) -> Optional[mpl.figure.Figure]:
    """Full operator-ablation violin figure intended for supplementary material."""
    gene_df = long_df[long_df["gene"] == gene].copy()
    if gene_df.empty:
        warnings.warn(f"No expression values found for {gene}; violin skipped.")
        return None

    panel_order = [p for p in cfg.panel_order if p in set(gene_df["panel"].astype(str))]
    layer_order = canonical_layer_order(gene_df["layer"].astype(str).values, cfg.preferred_layer_order)
    palette = get_layer_palette(layer_order)

    global_ymax = float(np.nanmax(gene_df["value_plot"].to_numpy(dtype=float)))
    y_top = max(global_ymax * 1.06, global_ymax + 0.10)

    fig_width = max(13.0, 1.86 * len(panel_order))
    fig, axes = plt.subplots(
        1,
        len(panel_order),
        figsize=(fig_width, 3.75),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    axes = np.atleast_1d(axes)
    metric_lookup = metrics[metrics["gene"] == gene].set_index("panel")

    for panel_index, (ax, panel) in enumerate(zip(axes, panel_order)):
        sub = gene_df[gene_df["panel"].astype(str) == panel].copy()
        sub["layer"] = pd.Categorical(sub["layer"].astype(str), categories=layer_order, ordered=True)
        violinplot_compat(ax, sub, layer_order, palette)

        color = get_panel_color(panel)
        ax.set_title(panel, color=color if panel.startswith("AXIS-") else "#222222", fontweight="bold", pad=5)
        ax.set_xlabel("")
        ax.set_ylim(0.0, y_top)
        ax.set_xticks(np.arange(len(layer_order)))
        ax.set_xticklabels(
            [short_layer_label(layer) for layer in layer_order],
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        if panel_index == 0:
            ax.set_ylabel("Expression, log1p(counts)" if cfg.use_log1p else "Expression")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        style_axis(ax)

        if panel.startswith("AXIS-"):
            ax.spines["bottom"].set_color(color)
            ax.spines["bottom"].set_linewidth(1.5)
            ax.set_facecolor(mpl.colors.to_rgba(color, 0.022))

        if cfg.annotate_violin_metrics and panel in metric_lookup.index:
            eta_h = float(metric_lookup.loc[panel, "eta_h_squared"])
            adjacent_auc = float(metric_lookup.loc[panel, "adjacent_auc_mean"])
            ax.text(
                0.035,
                0.965,
                f"$\\eta_H^2$={fmt_metric(eta_h)}\nAdj. AUROC={fmt_metric(adjacent_auc)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.8,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.84),
            )

    fig.suptitle(
        f"{gene} laminar expression enhancement in the integrated atlas",
        y=1.01,
        fontsize=12.0,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        r"Full operator ablation; $\eta_H^2$ measures global separation and adjacent AUROC measures local neighboring-layer discrimination.",
        ha="center",
        va="top",
        fontsize=7.0,
        color="#444444",
    )

    stem = cfg.outdir / f"supplement_{gene}_layer_separation_violin"
    save_figure(fig, stem, cfg)
    plt.close(fig)
    return fig


def plot_all_gene_summary(metrics: pd.DataFrame, cfg: Config) -> Optional[mpl.figure.Figure]:
    """Standalone summary of global and adjacent-layer separation."""
    genes = ordered_summary_genes(metrics, cfg)
    if not genes:
        return None
    gain_stats = compute_gain_statistics(metrics, cfg, genes)

    fig = plt.figure(figsize=(16.0, 4.4))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.65, 1.65, 0.95),
        left=0.055,
        right=0.992,
        top=0.86,
        bottom=0.20,
        wspace=0.30,
    )
    ax_eta = fig.add_subplot(grid[0, 0])
    ax_auc = fig.add_subplot(grid[0, 1])
    ax_gain = fig.add_subplot(grid[0, 2])
    draw_marker_summary_panel(ax_eta, metrics, cfg, genes, panel_letter="a", show_legend=True)
    draw_adjacent_auc_summary_panel(ax_auc, metrics, cfg, genes, panel_letter="b", show_legend=False)
    draw_gain_panel(ax_gain, metrics, gain_stats, cfg, genes)
    ax_gain.set_title("c  Gain in global separation", loc="left", fontweight="bold")
    fig.suptitle(
        "AXIS integration enhances global and local laminar marker structure",
        fontsize=12.4,
        fontweight="bold",
        y=0.97,
    )
    stem = cfg.outdir / "all_genes_layer_separation_summary"
    save_figure(fig, stem, cfg)
    plt.close(fig)
    return fig


def load_plot_inputs(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = [cfg.long_csv, cfg.separation_csv]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Plot mode requires current layer-separation outputs. Missing: "
            + ", ".join(str(path) for path in missing)
            + ". Run once with RUN_MODE='eval' or 'both'."
        )

    long_df = pd.read_csv(cfg.long_csv)
    metrics = pd.read_csv(cfg.separation_csv)

    required_long = {"gene", "panel", "layer", "value_plot", "value_raw"}
    required_metrics = {
        "gene", "panel", "eta_h_squared", "adjacent_auc_mean",
        "delta_eta_h_vs_single_mean", "single_eta_h_min", "single_eta_h_max",
        "single_adjacent_auc_min", "single_adjacent_auc_max",
    }
    if not required_long.issubset(long_df.columns):
        raise ValueError(f"{cfg.long_csv} is incompatible; rerun evaluation.")
    if not required_metrics.issubset(metrics.columns):
        raise ValueError(f"{cfg.separation_csv} is incompatible; rerun evaluation.")

    return long_df, metrics


def run_plotting(
    cfg: Config,
    long_df: Optional[pd.DataFrame] = None,
    metrics: Optional[pd.DataFrame] = None,
) -> None:
    set_publication_style()
    if long_df is None or metrics is None:
        long_df, metrics = load_plot_inputs(cfg)

    if cfg.plot_main_composite:
        plot_main_result_figure(long_df, metrics, cfg)

    if cfg.plot_all_gene_summary:
        plot_all_gene_summary(metrics, cfg)

    if cfg.plot_supplementary_violins:
        available_genes = set(long_df["gene"])
        for gene in cfg.genes_to_plot:
            if gene not in available_genes:
                warnings.warn(f"{gene} is unavailable and will not be plotted.")
                continue
            plot_gene_violin(long_df, metrics, gene, cfg)


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    mode = validate_config(CFG)
    CFG.outdir.mkdir(parents=True, exist_ok=True)

    long_df: Optional[pd.DataFrame] = None
    metrics: Optional[pd.DataFrame] = None

    if mode in {"eval", "both"}:
        long_df, metrics = run_evaluation(CFG)

    if mode in {"plot", "both"}:
        run_plotting(CFG, long_df=long_df, metrics=metrics)


if __name__ == "__main__":
    main()
