#!/usr/bin/env python3
"""
DLPFC atlas-level layer-separation analysis for single sections, pooled data,
and one selected Axis aggregate.

The selected Axis aggregation operator is controlled by one hyperparameter:

    axis_aggregation_method = "mean"  # "mean", "median", or "max"

Only one integrated result is reported and displayed as ``Axis``. The analysis
uses each object's own ``obs[layer_key]`` annotations.

Run modes
---------
    RUN_MODE = "eval"
        Load marker expression, calculate all metrics, and save the numerical
        outputs.

    RUN_MODE = "plot"
        Load saved numerical outputs and regenerate the main and separate
        figures.

    RUN_MODE = "both"
        Save metrics first, then generate the main figure and separate figures.

Main figure
-----------
The main figure contains only:

1. Overall marker-gene global layer separation.
2. Overall marker-gene adjacent-layer discrimination.
3. Two representative marker-gene case studies.

The former conceptual population-atlas workflow panel is intentionally removed.

Primary metrics
---------------
1. Kruskal-Wallis rank-based layer-separation effect size:

       eta_H^2 = max(0, (H - k + 1) / (n - k))

2. Mean adjacent-layer AUROC:

   AUROC is computed for each available neighboring layer pair and made
   direction invariant as max(AUC, 1 - AUC). Values range from 0.5
   (indistinguishable) to 1.0 (perfectly separable).

Pool (raw) and Pool (warped) are represented by a single ``Pool`` condition
because coordinate warping does not alter expression-by-layer metrics.
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

RUN_MODE = "both"  # one of: "eval", "plot", "both"


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

    # Biologically ordered marker panel used in the overall summary.
    genes_to_plot: Tuple[str, ...] = (
        "CUX2", "RORB", "PCP4", "KCNIP2", "FOXP2",
        "MBP", "MOBP", "ENC1", "MFGE8",
    )

    # Two representative markers shown in the main figure.
    main_figure_genes: Tuple[str, ...] = ("PCP4", "MOBP")

    preferred_layer_order: Tuple[str, ...] = (
        "Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"
    )

    # ------------------------------------------------------------------
    # Select exactly one Axis aggregation operator.
    # Change only this field to "mean", "median", or "max".
    # All tables and plots display the selected result simply as "Axis".
    # ------------------------------------------------------------------
    axis_aggregation_method: str = "median"
    rebuild_axis_aggregate: bool = False
    aggregation_subdir: str = "aggregation"
    aggregation_gene_chunk_size: int = 256

    # Optional explicit path for the selected aggregate. When None, the script
    # expects:
    #   analysis_dir / aggregation_subdir /
    #   aggregate_{method}_ref_{reference_axis_value:.3f}.h5ad
    axis_aggregated_h5ad: Optional[str] = None

    use_log1p: bool = True

    save_formats: Tuple[str, ...] = ("png",)
    png_dpi: int = 450
    annotate_violin_metrics: bool = True

    plot_main_composite: bool = True
    plot_all_gene_summary: bool = True
    plot_supplementary_violins: bool = True

    bootstrap_iterations: int = 5000
    bootstrap_seed: int = 17

    @property
    def axis_method(self) -> str:
        return canonical_aggregation_method(self.axis_aggregation_method)

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
        # Method-specific output directory prevents plot mode from mixing
        # cached results generated with different aggregation operators.
        return self.analysis_dir / f"aggregation_layer_separation_{self.axis_method}"

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
    def panel_order(self) -> Tuple[str, ...]:
        return self.real_panels + ("Pool", "Axis")

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
    if len(cfg.main_figure_genes) < 1:
        raise ValueError("main_figure_genes must contain at least one marker.")
    canonical_aggregation_method(cfg.axis_aggregation_method)
    if cfg.aggregation_gene_chunk_size < 1:
        raise ValueError("aggregation_gene_chunk_size must be positive.")
    if cfg.bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations cannot be negative.")
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



def resolve_axis_aggregate_path(cfg: Config) -> Path:
    """Resolve or build the single selected Axis aggregate."""
    method = cfg.axis_method
    output_name = (
        f"aggregate_{method}_ref_{float(cfg.reference_axis_value):.3f}.h5ad"
    )

    if cfg.rebuild_axis_aggregate:
        print(f"[Aggregate] Axis ({method})")
        output_path = aggregate_virtual_slices(
            input_dir=str(cfg.analysis_dir),
            biological_axis_name=cfg.biological_axis_name,
            reference_axis_value=cfg.reference_axis_value,
            aggregation=method,
            output_subdir=cfg.aggregation_subdir,
            output_name=output_name,
            gene_chunk_size=cfg.aggregation_gene_chunk_size,
        )
        path = Path(output_path)
    elif cfg.axis_aggregated_h5ad is not None:
        path = Path(cfg.axis_aggregated_h5ad)
    else:
        path = cfg.analysis_dir / cfg.aggregation_subdir / output_name

    if not path.is_file():
        raise FileNotFoundError(
            f"Selected Axis aggregate was not found: {path}\n"
            f"Set axis_aggregated_h5ad, or set rebuild_axis_aggregate=True."
        )
    return path
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
    """Load single sections and one selected Axis aggregate into one long table."""
    single_adatas: Dict[int, Any] = {}
    for sid in sorted(cfg.sample_ids):
        path = Path(cfg.h5ad_dir) / f"{cfg.group_id}_{sid}.h5ad"
        print(f"[Load] Single {sid}: {path}")
        single_adatas[sid] = read_marker_adata(
            path=path,
            requested_genes=cfg.marker_genes,
            layer_key=cfg.layer_key,
        )

    axis_path = resolve_axis_aggregate_path(cfg)
    print(f"[Load] Axis ({cfg.axis_method}): {axis_path}")
    axis_adata = read_marker_adata(
        path=axis_path,
        requested_genes=cfg.marker_genes,
        layer_key=cfg.layer_key,
    )

    all_adatas = list(single_adatas.values()) + [axis_adata]
    common_genes = [
        gene
        for gene in cfg.marker_genes
        if all(gene in ad.var_names for ad in all_adatas)
    ]
    missing = [gene for gene in cfg.marker_genes if gene not in common_genes]
    if missing:
        warnings.warn(
            "Genes absent from at least one single/Axis object were skipped: "
            + ", ".join(missing)
        )
    if not common_genes:
        raise ValueError("No marker gene is shared across all single and Axis objects.")

    single_parts = [
        adata_to_long_df(
            single_adatas[sid],
            panel=f"Single {sid}",
            genes=common_genes,
            layer_key=cfg.layer_key,
            use_log1p=cfg.use_log1p,
        )
        for sid in sorted(single_adatas)
    ]
    real_long = pd.concat(single_parts, ignore_index=True)

    # Pool is the union of observations from the native sections.
    pool_long = real_long.copy()
    pool_long["panel"] = "Pool"

    axis_long = adata_to_long_df(
        axis_adata,
        panel="Axis",
        genes=common_genes,
        layer_key=cfg.layer_key,
        use_log1p=cfg.use_log1p,
    )

    long_df = pd.concat([real_long, pool_long, axis_long], ignore_index=True)
    categories = [p for p in cfg.panel_order if p in set(long_df["panel"])]
    long_df["panel"] = pd.Categorical(
        long_df["panel"],
        categories=categories,
        ordered=True,
    )
    return long_df

# =============================================================================
# LAYER-SEPARATION METRICS
# =============================================================================


def compute_layer_summary(
    long_df: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, List[str]]:
    """Summarize marker expression within each panel and cortical layer."""
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
        summary["layer"],
        categories=layer_order,
        ordered=True,
    )
    summary = summary.sort_values(
        ["gene", "panel", "layer"]
    ).reset_index(drop=True)
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    return metrics, adjacent_df


def summarize_methods(metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Summarize marker-level performance for Pool and the selected Axis atlas."""
    candidate_panels = [
        panel
        for panel in ("Pool", "Axis")
        if panel in set(metrics["panel"].astype(str))
    ]

    rows: List[Dict[str, Any]] = []
    for panel in candidate_panels:
        panel_df = metrics[metrics["panel"].astype(str) == panel]
        eta = panel_df["eta_h_squared"].dropna().to_numpy(dtype=float)
        eta_delta = panel_df["delta_eta_h_vs_single_mean"].dropna().to_numpy(dtype=float)
        auc = panel_df["adjacent_auc_mean"].dropna().to_numpy(dtype=float)
        auc_delta = panel_df["delta_adjacent_auc_vs_single_mean"].dropna().to_numpy(dtype=float)
        eta_exceeds = panel_df["eta_h_exceeds_best_single"].dropna().astype(bool).to_numpy()
        auc_exceeds = panel_df["adjacent_auc_exceeds_best_single"].dropna().astype(bool).to_numpy()

        rows.append(
            {
                "panel": panel,
                "axis_aggregation_method": cfg.axis_method if panel == "Axis" else "",
                "n_genes": int(panel_df["gene"].nunique()),
                "mean_eta_h_squared": float(np.mean(eta)) if len(eta) else np.nan,
                "median_eta_h_squared": float(np.median(eta)) if len(eta) else np.nan,
                "mean_delta_eta_h_vs_single": (
                    float(np.mean(eta_delta)) if len(eta_delta) else np.nan
                ),
                "median_delta_eta_h_vs_single": (
                    float(np.median(eta_delta)) if len(eta_delta) else np.nan
                ),
                "mean_adjacent_auc": float(np.mean(auc)) if len(auc) else np.nan,
                "median_adjacent_auc": float(np.median(auc)) if len(auc) else np.nan,
                "mean_delta_adjacent_auc_vs_single": (
                    float(np.mean(auc_delta)) if len(auc_delta) else np.nan
                ),
                "median_delta_adjacent_auc_vs_single": (
                    float(np.median(auc_delta)) if len(auc_delta) else np.nan
                ),
                "n_genes_eta_above_best_single": (
                    int(np.sum(eta_exceeds)) if len(eta_exceeds) else 0
                ),
                "fraction_genes_eta_above_best_single": (
                    float(np.mean(eta_exceeds)) if len(eta_exceeds) else np.nan
                ),
                "n_genes_auc_above_best_single": (
                    int(np.sum(auc_exceeds)) if len(auc_exceeds) else 0
                ),
                "fraction_genes_auc_above_best_single": (
                    float(np.mean(auc_exceeds)) if len(auc_exceeds) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def write_numerical_report(
    method_summary: pd.DataFrame,
    cfg: Config,
) -> None:
    definitions = (
        "# DLPFC atlas-level laminar-enhancement report\n\n"
        f"- Selected Axis aggregation operator: `{cfg.axis_method}`\n"
        "- The selected integrated result is displayed as `Axis` in every table "
        "and figure.\n\n"
        "The analysis evaluates population-level laminar enhancement rather than "
        "fidelity to the raw distribution of any individual section.\n\n"
        "- **eta_H^2**: Kruskal-Wallis rank-based global layer-separation effect "
        "size across observed cortical layers.\n"
        "- **mean adjacent-layer AUROC**: direction-invariant AUROC averaged over "
        "available neighboring pairs in anatomical order.\n"
        "- **delta vs single**: candidate metric minus the mean across individual "
        "real sections for the same gene.\n"
        "- `Pool` represents the union of native observations; raw and warped "
        "pooling are equivalent for expression-by-layer metrics.\n\n"
    )
    try:
        table = method_summary.to_markdown(index=False, floatfmt=".3f")
    except ImportError:
        table = method_summary.to_csv(index=False)
    cfg.numerical_report_md.write_text(
        definitions + table + "\n",
        encoding="utf-8",
    )


def save_metrics(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and save all numerical outputs in one explicit sequence."""
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Collect marker expression")
    long_df = collect_expression_long(cfg)

    print("[2/5] Summarize expression by layer")
    layer_summary, layer_order = compute_layer_summary(long_df, cfg)

    print("[3/5] Calculate layer-separation metrics")
    metrics, adjacent_df = compute_layer_separation_metrics(
        long_df=long_df,
        layer_summary=layer_summary,
        layer_order=layer_order,
        cfg=cfg,
    )

    print("[4/5] Summarize Pool and Axis")
    method_summary = summarize_methods(metrics, cfg)

    print("[5/5] Save numerical outputs")
    long_df.to_csv(cfg.long_csv, index=False)
    layer_summary.to_csv(cfg.layer_summary_csv, index=False)
    metrics.to_csv(cfg.separation_csv, index=False)
    adjacent_df.to_csv(cfg.adjacent_auc_csv, index=False)
    method_summary.to_csv(cfg.method_summary_csv, index=False)
    write_numerical_report(method_summary, cfg)

    for path in (
        cfg.long_csv,
        cfg.layer_summary_csv,
        cfg.separation_csv,
        cfg.adjacent_auc_csv,
        cfg.method_summary_csv,
        cfg.numerical_report_md,
    ):
        print(f"[OK] saved: {path}")

    return long_df, metrics


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
        "Axis": "#E69F00",
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


def candidate_panels_present(
    metrics: pd.DataFrame,
    cfg: Config,
) -> List[str]:
    observed = set(metrics["panel"].astype(str))
    return [panel for panel in ("Pool", "Axis") if panel in observed]
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
    """Plot the single-section range, Pool, and one selected Axis result."""
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

    offsets = {"Pool": -0.12, "Axis": 0.12}
    markers = {"Pool": "o", "Axis": "s"}
    for panel in candidates:
        values = (
            metrics[metrics["panel"].astype(str) == panel]
            .set_index("gene")[metric_col]
            .reindex(genes)
            .to_numpy(dtype=float)
        )
        ax.scatter(
            x + offsets[panel],
            values,
            s=30,
            marker=markers[panel],
            color=get_panel_color(panel),
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
            label=panel,
        )

    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(
        genes,
        rotation=38,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylim(*y_limits)
    style_axis(ax)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        desired = (
            "Single-section range",
            "Single-section mean",
            "Pool",
            "Axis",
        )
        indices = [
            i
            for name in desired
            for i, label in enumerate(labels)
            if label == name
        ]
        ax.legend(
            [handles[i] for i in indices],
            [labels[i] for i in indices],
            frameon=False,
            ncol=2,
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
        "Axis": "Axis",
    }
    ax.set_xticks(positions)
    ax.set_xticklabels([display_labels.get(x, x) for x in candidates], rotation=0, ha="center")
    style_axis(ax)

    # Paired one-sided Wilcoxon P values are written compactly above Axis.
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
    """Panels displayed in each main-figure case-study row."""
    return list(cfg.real_panels) + ["Pool", "Axis"]
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


def plot_main_figure(
    long_df: pd.DataFrame,
    metrics: pd.DataFrame,
    cfg: Config,
) -> Optional[mpl.figure.Figure]:
    """
    Create the main result figure.

    Layout
    ------
    a. Overall marker-gene global layer separation
    b. Overall marker-gene adjacent-layer discrimination
    c. Representative marker-gene case study 1
    d. Representative marker-gene case study 2
    """
    genes = ordered_summary_genes(metrics, cfg)
    if not genes:
        warnings.warn("No configured summary genes are available.")
        return None

    main_genes = [
        gene
        for gene in cfg.main_figure_genes
        if gene in set(long_df["gene"].astype(str))
    ]
    if len(main_genes) < 2:
        warnings.warn(
            "Fewer than two representative genes are available for the main figure."
        )

    fig = plt.figure(figsize=(15.2, 9.2))
    outer = fig.add_gridspec(
        3,
        14,
        height_ratios=(1.0, 1.22, 1.22),
        left=0.050,
        right=0.995,
        top=0.935,
        bottom=0.080,
        hspace=0.58,
        wspace=0.72,
    )

    # Overall marker-gene performance.
    ax_eta = fig.add_subplot(outer[0, 0:7])
    ax_auc = fig.add_subplot(outer[0, 7:14])
    draw_marker_summary_panel(
        ax_eta,
        metrics,
        cfg,
        genes,
        panel_letter="a",
        show_legend=True,
    )
    draw_adjacent_auc_summary_panel(
        ax_auc,
        metrics,
        cfg,
        genes,
        panel_letter="b",
        show_legend=False,
    )

    descriptors = {
        "PCP4": "deep-cortical laminar marker",
        "MOBP": "white-matter-associated myelin marker",
        "MBP": "cortex-to-white-matter marker",
    }

    if len(main_genes) >= 1:
        draw_representative_violin_row(
            fig=fig,
            subspec=outer[1, :],
            long_df=long_df,
            metrics=metrics,
            gene=main_genes[0],
            cfg=cfg,
            panel_letter="c",
            descriptor=descriptors.get(
                main_genes[0],
                "representative laminar marker",
            ),
        )

    if len(main_genes) >= 2:
        draw_representative_violin_row(
            fig=fig,
            subspec=outer[2, :],
            long_df=long_df,
            metrics=metrics,
            gene=main_genes[1],
            cfg=cfg,
            panel_letter="d",
            descriptor=descriptors.get(
                main_genes[1],
                "representative laminar marker",
            ),
        )

    fig.suptitle(
        "Axis integration reveals population-level laminar marker structure",
        fontsize=14.0,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.022,
        (
            r"$\eta_H^2$: global Kruskal–Wallis rank-based layer separation; "
            "adjacent-layer AUROC: mean direction-invariant discrimination "
            "across available neighboring layer pairs."
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
    """Standalone per-gene violin figure for supplementary material."""
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
        ax.set_title(panel, color=color if panel == "Axis" else "#222222", fontweight="bold", pad=5)
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

        if panel == "Axis":
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
        r"$\eta_H^2$ measures global separation and adjacent AUROC measures local neighboring-layer discrimination.",
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
        "Axis integration enhances global and local laminar marker structure",
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


def plot_separate_figures(
    long_df: pd.DataFrame,
    metrics: pd.DataFrame,
    cfg: Config,
) -> None:
    """Create standalone overall and per-gene supplementary figures."""
    if cfg.plot_all_gene_summary:
        plot_all_gene_summary(metrics, cfg)

    if cfg.plot_supplementary_violins:
        available_genes = set(long_df["gene"].astype(str))
        for gene in cfg.genes_to_plot:
            if gene not in available_genes:
                warnings.warn(f"{gene} is unavailable and will not be plotted.")
                continue
            plot_gene_violin(long_df, metrics, gene, cfg)


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Execute the analysis in three explicit stages:

    1. Save metrics.
    2. Plot the main figure.
    3. Plot separate figures.
    """
    mode = validate_config(CFG)
    CFG.outdir.mkdir(parents=True, exist_ok=True)

    if mode in {"eval", "both"}:
        long_df, metrics = save_metrics(CFG)
    else:
        long_df, metrics = load_plot_inputs(CFG)

    if mode in {"plot", "both"}:
        set_publication_style()

        print("[Plot 1/2] Main figure")
        if CFG.plot_main_composite:
            plot_main_figure(long_df, metrics, CFG)

        print("[Plot 2/2] Separate figures")
        plot_separate_figures(long_df, metrics, CFG)


if __name__ == "__main__":
    main()
