#!/usr/bin/env python3
from __future__ import annotations
from axis.settings import data_path

"""Hybrid DLPFC registration evaluation + reporting pipeline.

Modes
-----
eval
    Collect all DLPFC registration metrics and case-study metadata, and save
    them to disk. No figures are generated.

plot
    Read the saved metric/metadata outputs and generate:
      1) the overall box + dot performance report;
      2) one standalone case-study image per registration method;
      3) one standalone fixed/moving annotation legend;
      4) one standalone anatomical-layer legend.
    No registration metric is recomputed.

both
    Run eval first, then plot from the newly saved results.

The shared method palette and boxplot layout are defined in
``utils_statistic_plot.py`` so DLPFC and the other datasets can use the same
publication styling.
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from utils_io import (
    filter_missing_domain_labels,
    load_pair_for_method,
    normalize_pair_by_fixed_frame,
    save_run_config,
    validate_pair,
)
from utils_metrics import (
    PairEvaluationResult,
    collect_evaluation_results,
    evaluate_domain_metrics,
    evaluate_neighborhood_metrics,
    merge_metric_tables,
    case_metadata_path,
    load_case_metadata_npz,
    _decode_contours,
)
from utils_statistic_plot import (
    METHOD_DISPLAY,
    SHARED_BOXPLOT_STYLE,
    load_metrics_csv,
    plot_overall_metrics_across_methods_row,
)


# =============================================================================
# Dataset / output configuration
# =============================================================================

ROOT = Path(data_path("DLPFC"))
AXIS_ROOT = ROOT / "huifang"
H5AD_DIR = AXIS_ROOT / "h5ad"
COMPETING_ROOT = ROOT / "competing_methods"
AXIS_NETWORK_INPUT_DIR = AXIS_ROOT / "network_input"
AXIS_PAIRWISE_DIR = AXIS_ROOT / "pairwise_output"

# Unified DLPFC analysis directory. Eval mode writes metrics/metadata here;
# plot mode reads those files and writes publication figures beside them.
OUTPUT_DIR = Path(
    data_path("paper_analysis/Figure2_transition_evaluation/bc/")
)
METRICS_CSV = OUTPUT_DIR / "dlpfc_metrics_all_methods.csv"
METADATA_DIR = OUTPUT_DIR / "case_metadata"
CONFIG_JSON = OUTPUT_DIR / "dlpfc_evaluation_config.json"
OVERALL_FIGURE = OUTPUT_DIR / "dlpfc_overall_performance.png"

# Case-study outputs are separated so individual method panels can be assembled
# flexibly in the final manuscript figure.
CASE_OUTPUT_DIR = OUTPUT_DIR / "case_study"
CASE_FIXED_MOVING_LEGEND = CASE_OUTPUT_DIR / "fixed_moving_legend.png"
CASE_DOMAIN_LEGEND = CASE_OUTPUT_DIR / "domain_legend.png"

GROUP_IDS = (0, 1, 2)
PAIR_LIST = ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3))
SLICE_POSITION_UM = {0: 0.0, 1: 10.0, 2: 310.0, 3: 320.0}

METHODS = (
    "unaligned",
    "simpleitk",
    "voxelmorph",
    "nicetrans",
    "paste",
    "gpsa",
    "santo",
    "axis",
)
COMPETING_METHODS = tuple(
    method for method in METHODS if method not in {"unaligned", "axis"}
)

# Shared naming with the statistical-report utility.
METHOD_DISPLAY_NAMES = {
    method: METHOD_DISPLAY.get(method, method)
    for method in METHODS
}

H5AD_COORD_KEY = "spatial"
H5AD_LABEL_KEY = "layer_guess_reordered"
AXIS_VALIDATION_COORD_KEY = "coord"
AXIS_VALIDATION_LABEL_KEY = "label"

CORR_GRID_SIZE = 100
CORR_SIGMA = 2.0
DISTANCE_THRESHOLD = 50.0
REMOVE_OUTLIERS = True
NEIGHBOR_RADIUS = None
STRICT = True

# Overall performance report.
OVERALL_METRICS = (
    "dice",
    "spatial_corr",
    "center_shift",
    "neighbor_accuracy",
)

# =============================================================================
# Case-study plotting configuration
# =============================================================================

CASE_GROUP_ID = 1
CASE_FIXED_ID = 1
CASE_MOVING_ID = 2

# Domain colors are intentionally fixed so case-study maps stay consistent with
# the rest of the registration figures.
DOMAIN_COLORS = {
    "0": "#79706E",
    "1": "#4E79A7",
    "2": "#F28E2B",
    "3": "#59A14F",
    "4": "#E15759",
    "5": "#B07AA1",
    "6": "#76B7B2",
    "7": "#EDC948",
    "8": "#FF9DA7",
    "9": "#9C755F",
    "10": "#79706E",
    "11": "#A0CBE8",
    "12": "#FFBE7D",
    "13": "#8CD17D",
    "14": "#FF9D9A",
    "15": "#D4A6C8",
    "16": "#86BCB6",
    "17": "#F1CE63",
    "18": "#FABFD2",
    "19": "#D7B5A6",
    "20": "#BAB0AC",
}
BACKGROUND_COLOR = "#D9D9D9"

# The standalone domain legend intentionally contains only these five DLPFC
# anatomical annotations, as requested.
DOMAIN_LEGEND_LABELS = {
    "1": "Layer3",
    "2": "Layer4",
    "3": "Layer5",
    "4": "Layer6",
    "5": "WM",
}


@dataclass(frozen=True)
class CaseStudyLayout:
    """Centralized controls for all DLPFC case-study graphics."""

    # Individual method map.
    method_figsize: Tuple[float, float] = (4.7, 4.5)
    method_title_fontsize: float = 20
    method_title_pad: float = 3.0
    point_size: float = 10
    point_alpha: float = 0.50
    moving_alpha_add: float = 0.12
    moving_marker_linewidth: float = 1
    contour_width: float = 1.5
    coordinate_padding_ratio: float = 0.03
    coordinate_min_padding: float = 1.0
    invert_y: bool = True

    # Standalone fixed/moving legend.
    slice_legend_figsize: Tuple[float, float] = (2.8, 2.7)
    slice_legend_fontsize: float = 13.0
    slice_legend_marker_size: float = 7.0
    slice_legend_linewidth: float = 1.25
    slice_legend_labelspacing: float = 0.85
    slice_legend_handletextpad: float = 0.55

    # Standalone domain legend.
    domain_legend_figsize: Tuple[float, float] = (2.5, 3.0)
    domain_legend_title: Optional[str] =None
    domain_legend_title_fontsize: float = 13.0
    domain_legend_fontsize: float = 13.0
    domain_legend_marker_size: float = 7.0
    domain_legend_labelspacing: float = 0.78
    domain_legend_handletextpad: float = 0.55

    # Saving / typography.
    font_family: str = "DejaVu Sans"
    dpi: int = 350
    save_pad_inches: float = 0.02


CASE_LAYOUT = CaseStudyLayout()


# =============================================================================
# Evaluation helpers
# =============================================================================

def distance_um(fixed_id: int, moving_id: int) -> float:
    return float(
        abs(
            SLICE_POSITION_UM[moving_id]
            - SLICE_POSITION_UM[fixed_id]
        )
    )


def alpha_for_group(group_id: int) -> float:
    return 0.05 if group_id == 0 else 0.01


def load_pair(
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
):
    source = f"DLPFC/{method}/group_{group_id}/{fixed_id}->{moving_id}"

    coords_fixed, coords_moving, labels_fixed, labels_moving = (
        load_pair_for_method(
            method=method,
            group_id=group_id,
            fixed_id=fixed_id,
            moving_id=moving_id,
            competing_methods=COMPETING_METHODS,
            competing_root=COMPETING_ROOT,
            h5ad_dir=H5AD_DIR,
            h5ad_coord_key=H5AD_COORD_KEY,
            h5ad_label_key=H5AD_LABEL_KEY,
            axis_network_input_dir=AXIS_NETWORK_INPUT_DIR,
            axis_pairwise_dir=AXIS_PAIRWISE_DIR,
            axis_validation_coord_key=AXIS_VALIDATION_COORD_KEY,
            axis_validation_label_key=AXIS_VALIDATION_LABEL_KEY,
        )
    )

    # Most methods share region-label arrays from the AXIS network inputs.
    # GPSA/SANTO retain their method-specific subset labels.
    if method not in {"gpsa", "santo"}:
        labels_fixed = np.load(
            AXIS_NETWORK_INPUT_DIR
            / f"{group_id}_{fixed_id}_region_label.npy"
        )
        labels_moving = np.load(
            AXIS_NETWORK_INPUT_DIR
            / f"{group_id}_{moving_id}_region_label.npy"
        )

    coords_fixed, labels_fixed = filter_missing_domain_labels(
        coords=coords_fixed,
        labels=labels_fixed,
        source=f"{source}: fixed",
        canonicalize_numeric=True,
    )
    coords_moving, labels_moving = filter_missing_domain_labels(
        coords=coords_moving,
        labels=labels_moving,
        source=f"{source}: moving",
        canonicalize_numeric=True,
    )

    return validate_pair(
        coords_fixed,
        labels_fixed,
        coords_moving,
        labels_moving,
        source=source,
    )


def evaluate_one_pair(
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
) -> PairEvaluationResult:
    coords_fixed, coords_moving, labels_fixed, labels_moving = load_pair(
        method,
        group_id,
        fixed_id,
        moving_id,
    )

    (
        coords_fixed,
        coords_moving,
        mask_shape,
        normalization,
    ) = normalize_pair_by_fixed_frame(
        coords_fixed,
        coords_moving,
    )

    domain_metrics, regions_fixed, regions_moving = evaluate_domain_metrics(
        coords_fixed,
        labels_fixed,
        coords_moving,
        labels_moving,
        mask_shape=mask_shape,
        alpha=alpha_for_group(group_id),
        corr_grid_size=CORR_GRID_SIZE,
        corr_sigma=CORR_SIGMA,
        common_labels=None,
        remove_outliers=REMOVE_OUTLIERS,
        neighbor_radius=NEIGHBOR_RADIUS,
    )
    neighborhood_metrics, per_spot = evaluate_neighborhood_metrics(
        coords_fixed,
        labels_fixed,
        coords_moving,
        labels_moving,
        distance_threshold=DISTANCE_THRESHOLD,
    )

    return PairEvaluationResult(
        metrics=merge_metric_tables(
            domain_metrics,
            neighborhood_metrics,
        ),
        per_spot=per_spot,
        coords_fixed=coords_fixed,
        coords_moving=coords_moving,
        labels_fixed=labels_fixed,
        labels_moving=labels_moving,
        regions_fixed=regions_fixed,
        regions_moving=regions_moving,
        normalization=normalization,
    )


# =============================================================================
# Eval mode
# =============================================================================

def evaluate_and_save():
    """Collect/save all DLPFC metrics and case-study metadata; no plotting."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_run_config(
        CONFIG_JSON,
        {
            "dataset": "DLPFC",
            "root": ROOT,
            "methods": METHODS,
            "groups": GROUP_IDS,
            "pairs": PAIR_LIST,
            "slice_position_um": SLICE_POSITION_UM,
            "alpha_by_group": {
                group_id: alpha_for_group(group_id)
                for group_id in GROUP_IDS
            },
            "remove_outliers": REMOVE_OUTLIERS,
            "neighbor_radius": NEIGHBOR_RADIUS,
            "corr_grid_size": CORR_GRID_SIZE,
            "corr_sigma": CORR_SIGMA,
            "distance_threshold": DISTANCE_THRESHOLD,
            "normalization": "isotropic fixed-slice scaling",
        },
    )

    result = collect_evaluation_results(
        methods=METHODS,
        group_ids=GROUP_IDS,
        pair_list=PAIR_LIST,
        evaluate_pair=evaluate_one_pair,
        out_csv=METRICS_CSV,
        metadata_dir=METADATA_DIR,
        distance_name="distance_um",
        distance_fn=distance_um,
        strict=STRICT,
    )
    print(f"[DONE] evaluation results: {METRICS_CSV}")
    print(f"[DONE] case-study metadata: {METADATA_DIR}")
    return result


# =============================================================================
# Plot mode
# =============================================================================

def _require_saved_evaluation() -> None:
    missing = []
    if not METRICS_CSV.is_file():
        missing.append(str(METRICS_CSV))
    if not METADATA_DIR.is_dir():
        missing.append(str(METADATA_DIR))
    if missing:
        raise FileNotFoundError(
            "Missing saved DLPFC evaluation result(s). Run --mode eval first:\n  "
            + "\n  ".join(missing)
        )


def plot_overall_performance():
    """Generate the pair-level DLPFC box + dot performance report."""
    df = load_metrics_csv(METRICS_CSV)

    missing_metrics = [
        metric
        for metric in OVERALL_METRICS
        if metric not in df.columns
    ]
    if missing_metrics:
        raise ValueError(
            f"Saved DLPFC metrics are missing columns: {missing_metrics}"
        )

    return plot_overall_metrics_across_methods_row(
        df=df,
        metrics=OVERALL_METRICS,
        save_path=OVERALL_FIGURE,
        style=SHARED_BOXPLOT_STYLE,
        # Fixed method order + shared colors provide direct visual
        # comparability with other datasets.
        sort_methods_by_median=True,
        share_y=False,
        show=False,
    )


def _canonical_domain(value: Any) -> str:
    """Canonicalize labels such as 1, 1.0, and "1.0" to "1"."""
    text = str(value)
    try:
        numeric = float(text)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return text


def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))


def _load_case_metadata_by_method() -> Dict[str, Mapping[str, np.ndarray]]:
    """Load saved case-study NPZ metadata for the configured pair."""
    metadata_by_method: Dict[str, Mapping[str, np.ndarray]] = {}

    for method in METHODS:
        path = case_metadata_path(
            METADATA_DIR,
            method,
            CASE_GROUP_ID,
            CASE_FIXED_ID,
            CASE_MOVING_ID,
        )
        if not path.exists():
            print(f"[WARN] Missing case-study metadata for {method}: {path}")
            continue

        metadata_by_method[method] = load_case_metadata_npz(path)

    if not metadata_by_method:
        raise FileNotFoundError(
            "No case-study metadata files were found for the selected DLPFC pair."
        )
    return metadata_by_method


def _shared_case_limits(
    metadata_by_method: Mapping[str, Mapping[str, np.ndarray]],
    layout: CaseStudyLayout,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Use one coordinate frame for all method images."""
    finite_arrays = []

    for meta in metadata_by_method.values():
        for key in ("coords_fixed", "coords_moving"):
            coords = np.asarray(meta[key], dtype=float)
            if coords.ndim != 2 or coords.shape[1] < 2:
                continue
            finite = coords[np.all(np.isfinite(coords[:, :2]), axis=1), :2]
            if len(finite):
                finite_arrays.append(finite)

    if not finite_arrays:
        raise ValueError("No finite case-study coordinates are available.")

    all_coords = np.vstack(finite_arrays)
    minimum = np.min(all_coords, axis=0)
    maximum = np.max(all_coords, axis=0)
    padding = np.maximum(
        (maximum - minimum) * layout.coordinate_padding_ratio,
        layout.coordinate_min_padding,
    )

    xlim = (
        float(minimum[0] - padding[0]),
        float(maximum[0] + padding[0]),
    )
    ylim = (
        float(minimum[1] - padding[1]),
        float(maximum[1] + padding[1]),
    )
    return xlim, ylim


def _available_domains(meta: Mapping[str, np.ndarray]) -> Sequence[str]:
    domains = set()
    for key in ("labels_fixed", "labels_moving"):
        labels = np.asarray(meta[key]).reshape(-1)
        domains.update(_canonical_domain(x) for x in labels)
    return sorted(domains, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))


def _style_case_axis(
    ax,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    layout: CaseStudyLayout,
) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    if layout.invert_y:
        ax.set_ylim(ylim[1], ylim[0])
    else:
        ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save_case_figure(fig, path: Path, layout: CaseStudyLayout) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=layout.dpi,
        bbox_inches="tight",
        pad_inches=layout.save_pad_inches,
        facecolor="white",
    )
    plt.close(fig)
    print(f"[OK] case-study figure: {path}")


def _plot_one_case_method(
    *,
    method: str,
    meta: Mapping[str, np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    save_path: Path,
    layout: CaseStudyLayout,
) -> None:
    """Render one registration method as one standalone map with no legend."""
    fixed_coords = np.asarray(meta["coords_fixed"], dtype=float)
    moving_coords = np.asarray(meta["coords_moving"], dtype=float)
    fixed_labels = np.asarray(
        [_canonical_domain(x) for x in meta["labels_fixed"]],
        dtype=str,
    )
    moving_labels = np.asarray(
        [_canonical_domain(x) for x in meta["labels_moving"]],
        dtype=str,
    )

    domains = _available_domains(meta)

    with plt.rc_context(
        {
            "font.family": layout.font_family,
            "font.size": 12,
            "axes.titlesize": layout.method_title_fontsize,
            # "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=layout.method_figsize)
        _style_case_axis(ax, xlim, ylim, layout)

        display = METHOD_DISPLAY_NAMES.get(method, method)
        ax.set_title(
            display,
            fontsize=layout.method_title_fontsize,
            # fontweight="bold",
            pad=layout.method_title_pad,
        )

        # Points: circle = fixed; x = moving. No legend is drawn here.
        for domain in domains:
            color = DOMAIN_COLORS.get(domain, BACKGROUND_COLOR)
            fixed_mask = fixed_labels == domain
            moving_mask = moving_labels == domain

            if np.any(fixed_mask):
                ax.scatter(
                    fixed_coords[fixed_mask, 0],
                    fixed_coords[fixed_mask, 1],
                    s=layout.point_size,
                    c=[color],
                    marker="o",
                    alpha=layout.point_alpha,
                    linewidths=0,
                    rasterized=True,
                )

            if np.any(moving_mask):
                ax.scatter(
                    moving_coords[moving_mask, 0],
                    moving_coords[moving_mask, 1],
                    s=layout.point_size,
                    c=[color],
                    marker="x",
                    alpha=min(1.0, layout.point_alpha + layout.moving_alpha_add),
                    linewidths=layout.moving_marker_linewidth,
                    rasterized=True,
                )

        # Contours: solid = fixed; dashed = moving.
        for domain, contour in _decode_contours(meta, "fixed"):
            domain = _canonical_domain(domain)
            color = DOMAIN_COLORS.get(domain, BACKGROUND_COLOR)
            ax.plot(
                contour[:, 0],
                contour[:, 1],
                color=color,
                lw=layout.contour_width,
            )

        for domain, contour in _decode_contours(meta, "moving"):
            domain = _canonical_domain(domain)
            color = DOMAIN_COLORS.get(domain, BACKGROUND_COLOR)
            ax.plot(
                contour[:, 0],
                contour[:, 1],
                color=color,
                lw=layout.contour_width,
                linestyle=(0, (3, 2)),
            )

        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.92)
        _save_case_figure(fig, save_path, layout)


def _plot_fixed_moving_legend(
    save_path: Path,
    layout: CaseStudyLayout,
) -> None:
    """Standalone vertical legend for fixed/moving point and contour encodings."""
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            color="0.25",
            markersize=layout.slice_legend_marker_size,
            label="Fixed\nslice",
        ),
        Line2D(
            [0], [0],
            marker="x",
            linestyle="none",
            color="0.25",
            markersize=layout.slice_legend_marker_size,
            label="Moving\nslice",
        ),
        Line2D(
            [0], [0],
            color="0.25",
            lw=layout.slice_legend_linewidth,
            label="Fixed\ncontour",
        ),
        Line2D(
            [0], [0],
            color="0.25",
            lw=layout.slice_legend_linewidth,
            linestyle=(0, (3, 2)),
            label="Moving\ncontour",
        ),
    ]

    fig = plt.figure(figsize=layout.slice_legend_figsize, facecolor="white")
    fig.legend(
        handles=handles,
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=layout.slice_legend_fontsize,
        labelspacing=layout.slice_legend_labelspacing,
        handletextpad=layout.slice_legend_handletextpad,
    )
    _save_case_figure(fig, save_path, layout)


def _plot_domain_legend(
    save_path: Path,
    layout: CaseStudyLayout,
) -> None:
    """Standalone DLPFC anatomical-layer legend for domains 1--5."""
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=DOMAIN_COLORS[domain],
            markeredgecolor="none",
            markersize=layout.domain_legend_marker_size,
            label=anatomical_name,
        )
        for domain, anatomical_name in DOMAIN_LEGEND_LABELS.items()
    ]

    fig = plt.figure(figsize=layout.domain_legend_figsize, facecolor="white")
    legend = fig.legend(
        handles=handles,
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=layout.domain_legend_fontsize,
        title=layout.domain_legend_title,
        labelspacing=layout.domain_legend_labelspacing,
        handletextpad=layout.domain_legend_handletextpad,
    )
    if layout.domain_legend_title is not None:
        legend.get_title().set_fontsize(layout.domain_legend_title_fontsize)
        legend.get_title().set_fontweight("bold")

    _save_case_figure(fig, save_path, layout)


def plot_case_study(
    layout: CaseStudyLayout = CASE_LAYOUT,
) -> None:
    """Generate individual method maps plus two standalone legend images."""
    CASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_by_method = _load_case_metadata_by_method()
    xlim, ylim = _shared_case_limits(metadata_by_method, layout)

    for method in METHODS:
        meta = metadata_by_method.get(method)
        if meta is None:
            continue

        save_path = CASE_OUTPUT_DIR / f"{_safe_filename(method)}.png"
        _plot_one_case_method(
            method=method,
            meta=meta,
            xlim=xlim,
            ylim=ylim,
            save_path=save_path,
            layout=layout,
        )

    _plot_fixed_moving_legend(
        save_path=CASE_FIXED_MOVING_LEGEND,
        layout=layout,
    )
    _plot_domain_legend(
        save_path=CASE_DOMAIN_LEGEND,
        layout=layout,
    )


def plot_saved_results():
    """Plot only from saved metric/metadata results."""
    _require_saved_evaluation()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_overall_performance()
    # plot_case_study()

    print(f"[DONE] overall performance figure: {OVERALL_FIGURE}")
    print(f"[DONE] case-study method/legend figures: {CASE_OUTPUT_DIR}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "DLPFC registration analysis with separated evaluation and "
            "publication plotting workflows."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("eval", "plot", "both"),
        default="plot",
        help=(
            "eval: collect/save metrics only; "
            "plot: plot only from saved results; "
            "both: evaluate then plot."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode in {"eval", "both"}:
        evaluate_and_save()

    if args.mode in {"plot", "both"}:
        plot_saved_results()


if __name__ == "__main__":
    main()
