from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from axis.metrics.alignment import (  # noqa: F401  (re-exported for the figure scripts)
    RegionMeta,
    _natural_sort_key,
    _string_labels,
    build_region_metadata,
    evaluate_domain_metrics,
    evaluate_neighborhood_metrics,
    filter_isolated_points,
    merge_metric_tables,
)
from utils_io import (
    NormalizationMeta,
    case_metadata_path,
    load_case_metadata_npz,
    save_case_metadata_npz,
)


@dataclass
class PairEvaluationResult:
    metrics: pd.DataFrame
    per_spot: pd.DataFrame
    coords_fixed: np.ndarray
    coords_moving: np.ndarray
    labels_fixed: np.ndarray
    labels_moving: np.ndarray
    regions_fixed: Dict[str, RegionMeta]
    regions_moving: Dict[str, RegionMeta]
    normalization: NormalizationMeta


# =============================================================================
# NumPy-only metadata transfer
# =============================================================================
def _pack_contours(
    regions: Mapping[str, RegionMeta],
    domains: Sequence[str],
) -> dict[str, np.ndarray]:
    points = []
    offsets = [0]
    domain_indices = []
    for domain_index, domain in enumerate(domains):
        for contour in regions[domain].contours:
            contour = np.asarray(contour, dtype=np.float32)
            if len(contour) < 2:
                continue
            points.append(contour)
            offsets.append(offsets[-1] + len(contour))
            domain_indices.append(domain_index)
    return {
        "contour_points": np.vstack(points).astype(np.float32) if points else np.empty((0, 2), dtype=np.float32),
        "contour_offsets": np.asarray(offsets, dtype=np.int64),
        "contour_domain_indices": np.asarray(domain_indices, dtype=np.int32),
    }


def build_case_metadata_payload(
    result: PairEvaluationResult,
    *,
    method: str,
    group_id: int,
    fixed_id: int,
    moving_id: int,
) -> dict[str, np.ndarray]:
    domains = sorted(
        set(result.regions_fixed).intersection(result.regions_moving),
        key=_natural_sort_key,
    )
    fixed_pack = _pack_contours(result.regions_fixed, domains)
    moving_pack = _pack_contours(result.regions_moving, domains)

    payload = {
        "method": np.asarray([method], dtype="U64"),
        "group_id": np.asarray([group_id], dtype=np.int64),
        "fixed_id": np.asarray([fixed_id], dtype=np.int64),
        "moving_id": np.asarray([moving_id], dtype=np.int64),
        "coords_fixed": np.asarray(result.coords_fixed, dtype=np.float32),
        "coords_moving": np.asarray(result.coords_moving, dtype=np.float32),
        "labels_fixed": _string_labels(result.labels_fixed).astype("U64"),
        "labels_moving": _string_labels(result.labels_moving).astype("U64"),
        "domains": np.asarray(domains, dtype="U64"),
        "centroids_fixed": np.asarray(
            [[result.regions_fixed[d].centroid_x, result.regions_fixed[d].centroid_y] for d in domains],
            dtype=np.float32,
        ),
        "centroids_moving": np.asarray(
            [[result.regions_moving[d].centroid_x, result.regions_moving[d].centroid_y] for d in domains],
            dtype=np.float32,
        ),
        "areas_fixed": np.asarray([result.regions_fixed[d].area for d in domains], dtype=np.float32),
        "areas_moving": np.asarray([result.regions_moving[d].area for d in domains], dtype=np.float32),
        "normalization": result.normalization.as_array(),
        "fixed_contour_points": fixed_pack["contour_points"],
        "fixed_contour_offsets": fixed_pack["contour_offsets"],
        "fixed_contour_domain_indices": fixed_pack["contour_domain_indices"],
        "moving_contour_points": moving_pack["contour_points"],
        "moving_contour_offsets": moving_pack["contour_offsets"],
        "moving_contour_domain_indices": moving_pack["contour_domain_indices"],
    }
    return payload


# =============================================================================
# Collection
# =============================================================================
def collect_evaluation_results(
    *,
    methods: Sequence[str],
    group_ids: Sequence[int],
    pair_list: Sequence[Tuple[int, int]],
    evaluate_pair: Callable[[str, int, int, int], PairEvaluationResult],
    out_csv: Path,
    metadata_dir: Path,
    distance_name: Optional[str] = None,
    distance_fn: Optional[Callable[[int, int], float]] = None,
    strict: bool = True,
) -> pd.DataFrame:
    all_rows = []
    for method in methods:
        for group_id in group_ids:
            for fixed_id, moving_id in pair_list:
                print(f"[INFO] {method} | group={group_id} | pair={fixed_id}->{moving_id}")
                try:
                    result = evaluate_pair(method, group_id, fixed_id, moving_id)
                except Exception as exc:
                    message = f"Failed: {method}, group={group_id}, pair={fixed_id}->{moving_id}"
                    # message = f"Failed: {method}, group={group_id}, pair={fixed_id}->{moving_id}: {exc}"
                    if strict:
                        raise RuntimeError(message) from exc
                    print(f"[WARN] {message}")
                    continue

                metrics = result.metrics.copy()
                metrics.insert(0, "method", method)
                metrics.insert(0, "moving_id", int(moving_id))
                metrics.insert(0, "fixed_id", int(fixed_id))
                metrics.insert(0, "group_id", int(group_id))
                if distance_name is not None and distance_fn is not None:
                    metrics.insert(4, distance_name, float(distance_fn(fixed_id, moving_id)))
                all_rows.append(metrics)

                payload = build_case_metadata_payload(
                    result,
                    method=method,
                    group_id=group_id,
                    fixed_id=fixed_id,
                    moving_id=moving_id,
                )
                save_case_metadata_npz(
                    case_metadata_path(metadata_dir, method, group_id, fixed_id, moving_id),
                    payload,
                )


    if not all_rows:
        raise RuntimeError("No evaluation results were collected")
    output = pd.concat(all_rows, ignore_index=True)

    output["domain"] = output["domain"].astype(str)
    output = output.sort_values(
        ["method", "group_id", "fixed_id", "moving_id", "domain"]
    ).reset_index(drop=True)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_csv, index=False)
    print(f"[OK] metrics: {out_csv}")
    print(f"[OK] metadata: {metadata_dir}")
    return output


# =============================================================================
# Publication-style 2 x 4 case-study figure
# =============================================================================
def _decode_contours(metadata: Mapping[str, np.ndarray], prefix: str):
    points = metadata[f"{prefix}_contour_points"]
    offsets = metadata[f"{prefix}_contour_offsets"]
    domain_indices = metadata[f"{prefix}_contour_domain_indices"]
    domains = metadata["domains"].astype(str)
    for index, domain_index in enumerate(domain_indices):
        start, end = int(offsets[index]), int(offsets[index + 1])
        yield domains[int(domain_index)], points[start:end]


def _aggregate_for_panel(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "dice": float(frame["dice"].mean()),
        "spatial_corr": float(frame["spatial_corr"].mean()),
        "center_shift": float(frame["center_shift"].mean()),
        "neighbor_accuracy": float(frame["neighbor_accuracy"].mean()),
        "n_domains": int(frame["domain"].nunique()),
    }

DOMAIN_COLORS = {
    "0": "#79706E",
    "1":  "#4E79A7",  # blue
    "2":  "#F28E2B",  # orange
    "3":  "#59A14F",  # green
    "4":  "#E15759",  # red
    "5":  "#B07AA1",  # purple
    "6":  "#76B7B2",  # teal
    "7":  "#EDC948",  # yellow
    "8":  "#FF9DA7",  # pink
    "9":  "#9C755F",  # brown
    "10": "#79706E",  # dark gray

    "11": "#A0CBE8",  # light blue
    "12": "#FFBE7D",  # light orange
    "13": "#8CD17D",  # light green
    "14": "#FF9D9A",  # light red
    "15": "#D4A6C8",  # light purple
    "16": "#86BCB6",  # light teal
    "17": "#F1CE63",  # light yellow
    "18": "#FABFD2",  # light pink
    "19": "#D7B5A6",  # light brown
    "20": "#BAB0AC",  # light gray
}

BACKGROUND_COLOR = "#D9D9D9"
def plot_case_study_grid(
    *,
    metrics_csv: Path,
    metadata_dir: Path,
    methods: Sequence[str],
    group_id: int,
    fixed_id: int,
    moving_id: int,
    save_path: Path,
    dataset_title: str,
    method_display_names: Optional[Mapping[str, str]] = None,
    domain_order: Optional[Sequence[Any]] = None,
    domain_cmap: str = "tab10",
    point_size: float = 5.0,
    point_alpha: float = 0.68,
    contour_width: float = 1.0,
    invert_y: bool = True,
    show_domain_legend: bool = True,
    dpi: int = 350,
) -> plt.Figure:
    """Render eight methods together from saved CSV and NPZ metadata only."""

    methods = list(methods)
    if len(methods) != 8:
        raise ValueError(f"The case-study layout requires exactly 8 methods, received {len(methods)}")

    metrics = pd.read_csv(metrics_csv, dtype={"domain": str})
    metrics = metrics[
        (metrics["group_id"] == group_id)
        & (metrics["fixed_id"] == fixed_id)
        & (metrics["moving_id"] == moving_id)
    ].copy()
    if metrics.empty:
        raise ValueError("No matching rows were found in the metrics CSV")

    metadata_by_method = {}
    available_domains = set()
    all_coords = []
    for method in methods:
        path = case_metadata_path(metadata_dir, method, group_id, fixed_id, moving_id)
        if path.exists():
            meta = load_case_metadata_npz(path)
            metadata_by_method[method] = meta
            available_domains.update(meta["domains"].astype(str).tolist())
            all_coords.extend([meta["coords_fixed"], meta["coords_moving"]])

    if domain_order is None:
        domains = sorted(available_domains, key=_natural_sort_key)
    else:
        domains = [str(x) for x in domain_order if str(x) in available_domains]
    if not domains:
        raise ValueError("No domains are available for plotting")

    color_map = {
        domain: DOMAIN_COLORS[str(domain)]
        for domain in domains
    }


    finite = np.vstack([x[np.all(np.isfinite(x), axis=1)] for x in all_coords])
    minimum = finite.min(axis=0)
    maximum = finite.max(axis=0)
    padding = np.maximum((maximum - minimum) * 0.03, 1.0)
    xlim = (minimum[0] - padding[0], maximum[0] + padding[0])
    ylim = (minimum[1] - padding[1], maximum[1] + padding[1])

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(2, 4, figsize=(11.6, 8.5), constrained_layout=True)
        axes = axes.ravel()

        for panel_index, (ax, method) in enumerate(zip(axes, methods)):
            ax.set_aspect("equal")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if invert_y:
                ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            display = method_display_names.get(method, method) if method_display_names else method
            ax.set_title(display, pad=4)
            meta = metadata_by_method.get(method)
            method_metrics = metrics[metrics["method"] == method]
            if meta is None or method_metrics.empty:
                ax.text(0.5, 0.5, "Not available", transform=ax.transAxes, ha="center", va="center")
                continue

            fixed_labels = meta["labels_fixed"].astype(str)
            moving_labels = meta["labels_moving"].astype(str)
            fixed_coords = meta["coords_fixed"]
            moving_coords = meta["coords_moving"]

            for domain in domains:
                color = color_map[domain]
                fixed_mask = fixed_labels == domain
                moving_mask = moving_labels == domain
                if np.any(fixed_mask):
                    ax.scatter(
                        fixed_coords[fixed_mask, 0],
                        fixed_coords[fixed_mask, 1],
                        s=point_size,
                        c=[color],
                        marker="o",
                        alpha=point_alpha,
                        linewidths=0,
                        rasterized=True,
                    )
                if np.any(moving_mask):
                    ax.scatter(
                        moving_coords[moving_mask, 0],
                        moving_coords[moving_mask, 1],
                        s=point_size,
                        c=[color],
                        marker="x",
                        alpha=min(1.0, point_alpha + 0.12),
                        linewidths=0.35,
                        rasterized=True,
                    )

            for domain, contour in _decode_contours(meta, "fixed"):
                if domain in color_map:
                    ax.plot(contour[:, 0], contour[:, 1], color=color_map[domain], lw=contour_width)
            for domain, contour in _decode_contours(meta, "moving"):
                if domain in color_map:
                    ax.plot(
                        contour[:, 0],
                        contour[:, 1],
                        color=color_map[domain],
                        lw=contour_width,
                        linestyle=(0, (3, 2)),
                    )

            summary = _aggregate_for_panel(method_metrics)
            # metric_text = (
            #     f"Dice ↑  {summary['dice']:.3f}\n"
            #     f"Spatial r ↑  {summary['spatial_corr']:.3f}\n"
            #     f"Shift ↓  {summary['center_shift']:.1f}\n"
            #     f"Local ↑  {summary['neighbor_accuracy']:.3f}"
            # )
            # ax.text(
            #     0.02,
            #     0.02,
            #     metric_text,
            #     transform=ax.transAxes,
            #     ha="left",
            #     va="bottom",
            #     fontsize=7.5,
            #     linespacing=1.25,
            #     bbox={"facecolor": "white", "edgecolor": "0.75", "linewidth": 0.5, "alpha": 0.9, "pad": 2.5},
            # )

        slice_handles = [
            Line2D([0], [0], marker="o", linestyle="none", color="0.25", markersize=5, label="Fixed slice"),
            Line2D([0], [0], marker="x", linestyle="none", color="0.25", markersize=5, label="Moving slice"),
            Line2D([0], [0], color="0.25", lw=1.1, label="Fixed contour"),
            Line2D([0], [0], color="0.25", lw=1.1, linestyle=(0, (3, 2)), label="Moving contour"),
        ]
        fig.legend(
            handles=slice_handles,
            loc="lower left",
            bbox_to_anchor=(0., 0.02),
            ncol=4,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.4,
        )

        bottom = 0.08
        if show_domain_legend:
            domain_handles = [
                Line2D([0], [0], marker="o", linestyle="none", color=color_map[d], markersize=4, label=str(d))
                for d in domains
            ]
            ncol = min(10, len(domain_handles))
            fig.legend(
                handles=domain_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=ncol,
                frameon=False,
                title="Domain",
                fontsize=7,
                title_fontsize=8,
                handletextpad=0.25,
                columnspacing=0.8,
            )
            bottom = 0.12 if len(domains) <= 10 else 0.17

        fig.suptitle(
            f"{dataset_title} | group {group_id} | pair {fixed_id}→{moving_id}",
            y=0.99,
            fontsize=13,
            fontweight="bold",
        )
        fig.subplots_adjust(left=0.025, right=0.99, top=0.89, bottom=bottom, wspace=0.05, hspace=0.13)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] case study: {save_path}")
        return fig
