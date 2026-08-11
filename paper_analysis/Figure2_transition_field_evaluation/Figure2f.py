"""Three-panel AXIS alignment overview for a two-sample Xenium breast group.

Coordinate sources
------------------
AXIS pairwise result files:

    root/competing_methods/axis/group_{group_id}/{fixed_id}_{moving_id}.npz

Each result file supplies only coordinates:

    pts1 : original coordinates of the fixed sample
    pts2 : warped coordinates of the moving sample in the fixed frame

Marker-label sources
--------------------
Each sample's marker labels are loaded independently from:

    root/huifang/network_input/{group_id}_{sample_id}_marker_label.npz

For two samples A and B:

- A_B.npz supplies native A as pts1 and warped B -> A as pts2.
- B_A.npz supplies native B as pts1 and warped A -> B as pts2.
- Labels for A always come from A's marker-label NPZ.
- Labels for B always come from B's marker-label NPZ.

The implementation assumes that pts2 preserves the point order of the moving
sample, so the moving sample's marker-label vector can be applied directly to
its warped coordinates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D


@dataclass
class AxisPairCoordinates:
    fixed_id: Any
    moving_id: Any
    pts1: np.ndarray
    pts2: np.ndarray


@dataclass
class XeniumBreastSlice:
    sample_id: Any
    display_name: str
    observed_xy: np.ndarray
    labels: np.ndarray
    aligned_xy: np.ndarray


def npz_get_first(data, keys, source: str):
    """Return the first requested key found in an opened NPZ object."""
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    raise KeyError(
        f"None of the required keys {list(keys)} was found in {source}. "
        f"Available keys: {list(data.files)}"
    )


def _axis_result_path(
    root: str,
    group_id: Any,
    fixed_id: Any,
    moving_id: Any,
    axis_output_subdir: str,
    result_template: str,
) -> str:
    return os.path.join(
        root,
        axis_output_subdir,
        f"group_{group_id}",
        result_template.format(
            group_id=group_id,
            fixed_id=fixed_id,
            moving_id=moving_id,
        ),
    )


def _marker_label_path(
    root: str,
    group_id: Any,
    sample_id: Any,
    marker_label_subdir: str,
    marker_label_template: str,
) -> str:
    return os.path.join(
        root,
        marker_label_subdir,
        marker_label_template.format(
            group_id=group_id,
            sample_id=sample_id,
        ),
    )


def _validate_coords(values: np.ndarray, name: str, source: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(
            f"{name} in {source} must have shape (N, 2+); got {values.shape}."
        )
    return values[:, :2]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.asarray(value != value).item())
    except Exception:
        return False


def _validate_labels(values: np.ndarray, name: str, source: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 2 and 1 in values.shape:
        values = values.reshape(-1)
    elif values.ndim != 1:
        raise ValueError(
            f"{name} in {source} must be one-dimensional or single-column; "
            f"got {values.shape}."
        )

    return np.asarray(
        ["NA" if _is_missing(value) else str(value) for value in values]
    )


def _load_axis_pair_coordinates(
    root: str,
    group_id: Any,
    fixed_id: Any,
    moving_id: Any,
    axis_output_subdir: str = "competing_methods/axis",
    result_template: str = "{fixed_id}_{moving_id}.npz",
) -> AxisPairCoordinates:
    """Load fixed native coordinates and moving warped coordinates."""
    path = _axis_result_path(
        root=root,
        group_id=group_id,
        fixed_id=fixed_id,
        moving_id=moving_id,
        axis_output_subdir=axis_output_subdir,
        result_template=result_template,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing AXIS result file: {path}")

    with np.load(path, allow_pickle=False) as data:
        pts1 = npz_get_first(data, ["pts1"], source=path)
        pts2 = npz_get_first(data, ["pts2"], source=path)

    return AxisPairCoordinates(
        fixed_id=fixed_id,
        moving_id=moving_id,
        pts1=_validate_coords(pts1, "pts1", path),
        pts2=_validate_coords(pts2, "pts2", path),
    )


def _load_sample_marker_labels(
    root: str,
    group_id: Any,
    sample_id: Any,
    marker_label_subdir: str = "huifang/network_input",
    marker_label_template: str = "{group_id}_{sample_id}_marker_label.npz",
    marker_label_key: Optional[str] = None,
) -> np.ndarray:
    """Load one sample's biological marker labels from its marker NPZ."""
    path = _marker_label_path(
        root=root,
        group_id=group_id,
        sample_id=sample_id,
        marker_label_subdir=marker_label_subdir,
        marker_label_template=marker_label_template,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing marker-label file: {path}")

    candidate_keys = []
    if marker_label_key is not None:
        candidate_keys.append(marker_label_key)
    candidate_keys.extend(
        [
            "marker_label",
            "marker_labels",
            "label",
            "labels",
            "arr_0",
        ]
    )

    # Preserve order while removing duplicate candidates.
    candidate_keys = list(dict.fromkeys(candidate_keys))

    with np.load(path, allow_pickle=True) as data:
        labels = npz_get_first(data, candidate_keys, source=path)

    return _validate_labels(labels, "marker labels", path)


def _validate_sample_lengths(
    sample_id: Any,
    labels: np.ndarray,
    native_xy: np.ndarray,
    warped_xy: np.ndarray,
) -> None:
    n_labels = len(labels)
    if len(native_xy) != n_labels:
        raise ValueError(
            f"Native coordinate/marker-label mismatch for sample {sample_id}: "
            f"{len(native_xy)} coordinates versus {n_labels} labels."
        )
    if len(warped_xy) != n_labels:
        raise ValueError(
            f"Warped coordinate/marker-label mismatch for sample {sample_id}: "
            f"{len(warped_xy)} coordinates versus {n_labels} labels. "
            "This function assumes pts2 preserves the moving-sample point order."
        )


def _prepare_two_sample_slices(
    root: str,
    group_id: Any,
    sample_ids: Sequence[Any],
    reference_id: Any,
    axis_output_subdir: str,
    result_template: str,
    marker_label_subdir: str,
    marker_label_template: str,
    marker_label_key: Optional[str],
    slice_names: Optional[Mapping[Any, str]],
) -> Tuple[
    Sequence[XeniumBreastSlice],
    AxisPairCoordinates,
    AxisPairCoordinates,
]:
    sample_ids = tuple(sample_ids)
    if len(sample_ids) != 2:
        raise ValueError(
            "This breast Xenium function is defined for exactly two samples; "
            f"received sample_ids={sample_ids}."
        )
    if reference_id not in sample_ids:
        raise ValueError(
            f"reference_id={reference_id} is not included in sample_ids={sample_ids}."
        )
    if sample_ids[0] == sample_ids[1]:
        raise ValueError("sample_ids must contain two distinct identifiers.")

    sample_a, sample_b = sample_ids

    result_ab = _load_axis_pair_coordinates(
        root=root,
        group_id=group_id,
        fixed_id=sample_a,
        moving_id=sample_b,
        axis_output_subdir=axis_output_subdir,
        result_template=result_template,
    )
    result_ba = _load_axis_pair_coordinates(
        root=root,
        group_id=group_id,
        fixed_id=sample_b,
        moving_id=sample_a,
        axis_output_subdir=axis_output_subdir,
        result_template=result_template,
    )

    labels_a = _load_sample_marker_labels(
        root=root,
        group_id=group_id,
        sample_id=sample_a,
        marker_label_subdir=marker_label_subdir,
        marker_label_template=marker_label_template,
        marker_label_key=marker_label_key,
    )
    labels_b = _load_sample_marker_labels(
        root=root,
        group_id=group_id,
        sample_id=sample_b,
        marker_label_subdir=marker_label_subdir,
        marker_label_template=marker_label_template,
        marker_label_key=marker_label_key,
    )

    # Native sample coordinates are taken from the reciprocal result in which
    # that sample is the fixed sample.
    native_xy = {
        sample_a: result_ab.pts1,
        sample_b: result_ba.pts1,
    }
    labels_by_sample = {
        sample_a: labels_a,
        sample_b: labels_b,
    }

    # Aligned coordinates are taken from the result whose fixed sample is the
    # requested reference. Labels still come from each sample's marker NPZ.
    if reference_id == sample_a:
        aligned_xy = {
            sample_a: result_ab.pts1,
            sample_b: result_ab.pts2,
        }
    else:
        aligned_xy = {
            sample_b: result_ba.pts1,
            sample_a: result_ba.pts2,
        }

    for sample_id in sample_ids:
        _validate_sample_lengths(
            sample_id=sample_id,
            labels=labels_by_sample[sample_id],
            native_xy=native_xy[sample_id],
            warped_xy=aligned_xy[sample_id],
        )

    slices = []
    for sample_id in sample_ids:
        display_name = (
            str(slice_names[sample_id])
            if slice_names is not None and sample_id in slice_names
            else f"Slice {sample_id}"
        )
        slices.append(
            XeniumBreastSlice(
                sample_id=sample_id,
                display_name=display_name,
                observed_xy=native_xy[sample_id],
                labels=labels_by_sample[sample_id],
                aligned_xy=aligned_xy[sample_id],
            )
        )

    return slices, result_ab, result_ba


def _natural_label_sort_key(label: str):
    try:
        return 0, float(label)
    except (TypeError, ValueError):
        return 1, str(label)


def _make_label_color_map(
    labels_by_slice: Sequence[np.ndarray],
    cmap_name: str,
    label_colors: Optional[Mapping[Any, Any]],
    label_order: Optional[Sequence[Any]],
    background_label: Any,
    background_color: Any,
) -> Dict[str, Any]:
    observed = {
        str(label)
        for labels in labels_by_slice
        for label in np.asarray(labels).reshape(-1)
    }

    if label_order is None:
        ordered = sorted(observed, key=_natural_label_sort_key)
    else:
        ordered = [str(value) for value in label_order]
        ordered.extend(
            sorted(observed - set(ordered), key=_natural_label_sort_key)
        )

    background = str(background_label)
    if label_colors is not None:
        supplied = {str(key): value for key, value in label_colors.items()}
        if background in observed and background not in supplied:
            supplied[background] = background_color
        missing = [label for label in ordered if label not in supplied]
        if missing:
            raise KeyError(
                "label_colors is missing entries for: " + ", ".join(missing)
            )
        return {label: supplied[label] for label in ordered}

    foreground = [label for label in ordered if label != background]
    cmap = get_cmap(cmap_name, max(len(foreground), 1))
    result = {label: cmap(index) for index, label in enumerate(foreground)}
    if background in ordered:
        result[background] = background_color
    return {label: result[label] for label in ordered}


def _finite_rows(coords: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(coords), axis=1)


def _scatter_labels(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    color_map: Mapping[str, Any],
    background_label: Any,
    background_point_size: float,
    marker_point_size: float,
    background_alpha: float,
    marker_alpha: float,
    zorder: int,
):
    """Plot background cells faintly and marker cells prominently.

    The background class is drawn first with a smaller point size and lower
    opacity. All non-background marker classes are then drawn on top using the
    larger marker-cell style. The same styling is used in all three panels.
    """
    coords = np.asarray(coords)
    labels = np.asarray(labels).astype(str)
    if coords.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Coordinate/label length mismatch: {coords.shape[0]} versus "
            f"{labels.shape[0]}."
        )

    valid = _finite_rows(coords)
    background = str(background_label)

    # Draw the background first so it does not cover the small marker regions.
    if background in color_map:
        keep = valid & (labels == background)
        if np.any(keep):
            ax.scatter(
                coords[keep, 0],
                coords[keep, 1],
                c=[color_map[background]],
                s=background_point_size,
                alpha=background_alpha,
                linewidths=0,
                edgecolors="none",
                rasterized=True,
                zorder=zorder,
            )

    # Draw marker-positive classes second and above the background.
    for label, color in color_map.items():
        if label == background:
            continue
        keep = valid & (labels == label)
        if not np.any(keep):
            continue
        ax.scatter(
            coords[keep, 0],
            coords[keep, 1],
            c=[color],
            s=marker_point_size,
            alpha=marker_alpha,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
            zorder=zorder + 1,
        )


def _axis_limits(arrays: Sequence[np.ndarray], pad_ratio: float = 0.03):
    finite_arrays = []
    for coords in arrays:
        coords = np.asarray(coords)
        keep = _finite_rows(coords)
        if np.any(keep):
            finite_arrays.append(coords[keep, :2])
    if not finite_arrays:
        raise ValueError("No finite coordinates are available for plotting.")

    merged = np.vstack(finite_arrays)
    low = merged.min(axis=0)
    high = merged.max(axis=0)
    pad = np.maximum((high - low) * pad_ratio, 1e-6)
    return (
        (low[0] - pad[0], high[0] + pad[0]),
        (low[1] - pad[1], high[1] + pad[1]),
    )


def _style_axis(ax, title: str, invert_y: bool):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if invert_y:
        ax.invert_yaxis()


def plot_xenium_breast_axis_alignment_overview(
    group_id: Any,
    sample_ids: Sequence[Any],
    reference_id: Any,
    root: str = "/media/huifang/data/registration/breast_cancer",
    axis_output_subdir: str = "competing_methods/axis",
    result_template: str = "{fixed_id}_{moving_id}_result.npz",
    marker_label_subdir: str = "huifang/network_input",
    marker_label_template: str = "{group_id}_{sample_id}_marker_label_img.npz",
    marker_label_key: Optional[str] = None,
    dataset_title: str = "Xenium breast cancer",
    slice_names: Optional[Mapping[Any, str]] = None,
    label_colors: Optional[Mapping[Any, Any]] = None,
    label_order: Optional[Sequence[Any]] = None,
    cmap_name: str = "tab20",
    background_label: Any = 0,
    background_color: Any = "#BDBDBD",
    background_legend_name: str = "Bg",
    background_point_size: float = 0.8,
    marker_point_size: float = 2.0,
    background_alpha: float = 0.36,
    marker_alpha: float = 0.82,
    stack_dx_fraction: float = 0.17,
    stack_dy_fraction: float = -0.13,
    stack_label_fontsize: float = 9.5,
    show_marker_legend: bool = True,
    legend_title: Optional[str] = "Marker annotation",
    legend_ncol: Optional[int] = 1,
    legend_fontsize: float = 8.5,
    legend_title_fontsize: float = 9.5,
    legend_marker_size: float = 7.0,
    invert_y: bool = True,
    share_overlap_limits: bool = True,
    figsize: Tuple[float, float] = (14.2, 5.4),
    panel_width_ratios: Tuple[float, float, float] = (1.08, 1.0, 1.0),
    legend_width_ratio: float = 0.34,
    panel_wspace: float = 0.015,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 350,
):
    """Plot stacked, native-overlap, and AXIS-aligned views for two samples.

    The three tissue panels occupy adjacent GridSpec columns with very small
    inter-panel spacing. When requested, the marker legend is placed in a
    dedicated fourth column, so it cannot overlap the ``After AXIS`` panel.

    The AXIS result files provide coordinates only. Marker labels are loaded
    from each sample's separate marker-label NPZ. Background and marker cells
    use separate point-size and opacity controls shared by all panels.
    """
    slices, result_forward, result_reverse = _prepare_two_sample_slices(
        root=root,
        group_id=group_id,
        sample_ids=sample_ids,
        reference_id=reference_id,
        axis_output_subdir=axis_output_subdir,
        result_template=result_template,
        marker_label_subdir=marker_label_subdir,
        marker_label_template=marker_label_template,
        marker_label_key=marker_label_key,
        slice_names=slice_names,
    )

    color_map = _make_label_color_map(
        labels_by_slice=[item.labels for item in slices],
        cmap_name=cmap_name,
        label_colors=label_colors,
        label_order=label_order,
        background_label=background_label,
        background_color=background_color,
    )

    widths = []
    heights = []
    for item in slices:
        finite = item.observed_xy[_finite_rows(item.observed_xy)]
        if finite.size:
            span = finite.max(axis=0) - finite.min(axis=0)
            widths.append(float(span[0]))
            heights.append(float(span[1]))
    if not widths or not heights:
        raise ValueError("No finite native coordinates were found.")

    stack_dx = float(np.median(widths) * stack_dx_fraction)
    stack_dy = float(np.median(heights) * stack_dy_fraction)

    # ------------------------------------------------------------------
    # Layout: three adjacent tissue panels plus an optional legend column.
    # A dedicated legend axis is more robust than a figure-level legend
    # positioned over the third panel.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize, constrained_layout=False)

    if show_marker_legend:
        width_ratios = [*panel_width_ratios, legend_width_ratio]
        gs = fig.add_gridspec(
            1,
            4,
            width_ratios=width_ratios,
            left=0.018,
            right=0.985,
            bottom=0.055,
            top=0.91,
            wspace=panel_wspace,
        )
        ax_stack = fig.add_subplot(gs[0, 0])
        ax_before = fig.add_subplot(gs[0, 1])
        ax_after = fig.add_subplot(gs[0, 2])
        ax_legend = fig.add_subplot(gs[0, 3])
        ax_legend.axis("off")
    else:
        gs = fig.add_gridspec(
            1,
            3,
            width_ratios=panel_width_ratios,
            left=0.018,
            right=0.985,
            bottom=0.055,
            top=0.91,
            wspace=panel_wspace,
        )
        ax_stack = fig.add_subplot(gs[0, 0])
        ax_before = fig.add_subplot(gs[0, 1])
        ax_after = fig.add_subplot(gs[0, 2])
        ax_legend = None

    # Panel 1: later slices first; sample_ids[0] remains visually in front.
    stacked_arrays = []
    for index in reversed(range(len(slices))):
        item = slices[index]
        shifted = item.observed_xy + np.asarray(
            [index * stack_dx, index * stack_dy], dtype=float
        )
        stacked_arrays.append(shifted)
        _scatter_labels(
            ax=ax_stack,
            coords=shifted,
            labels=item.labels,
            color_map=color_map,
            background_label=background_label,
            background_point_size=background_point_size,
            marker_point_size=marker_point_size,
            background_alpha=background_alpha,
            marker_alpha=marker_alpha,
            zorder=2 + 2 * (len(slices) - index),
        )

        finite = shifted[_finite_rows(shifted)]
        if finite.size:
            ax_stack.text(
                float(np.min(finite[:, 0])),
                float(np.min(finite[:, 1])),
                item.display_name,
                ha="left",
                va="bottom",
                fontsize=stack_label_fontsize,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                    "pad": 1.4,
                },
                zorder=50,
            )

    stack_xlim, stack_ylim = _axis_limits(stacked_arrays, pad_ratio=0.05)
    ax_stack.set_xlim(*stack_xlim)
    ax_stack.set_ylim(*stack_ylim)
    _style_axis(
        ax_stack,
        f"{dataset_title}, {len(slices)} slices",
        invert_y=invert_y,
    )

    # Panel 2: direct overlap before AXIS.
    for item in slices:
        _scatter_labels(
            ax=ax_before,
            coords=item.observed_xy,
            labels=item.labels,
            color_map=color_map,
            background_label=background_label,
            background_point_size=background_point_size,
            marker_point_size=marker_point_size,
            background_alpha=background_alpha,
            marker_alpha=marker_alpha,
            zorder=2,
        )
    _style_axis(ax_before, "Before AXIS", invert_y=invert_y)

    # Panel 3: reference pts1 plus moving pts2 in the reference frame.
    for item in slices:
        _scatter_labels(
            ax=ax_after,
            coords=item.aligned_xy,
            labels=item.labels,
            color_map=color_map,
            background_label=background_label,
            background_point_size=background_point_size,
            marker_point_size=marker_point_size,
            background_alpha=background_alpha,
            marker_alpha=marker_alpha,
            zorder=2,
        )
    _style_axis(ax_after, "After AXIS", invert_y=invert_y)

    # Equal aspect can leave unused room inside a GridSpec cell. Anchoring the
    # panels toward each other reduces the visual gap without distorting data.
    ax_stack.set_anchor("E")
    ax_before.set_anchor("C")
    ax_after.set_anchor("W")

    if share_overlap_limits:
        xlim, ylim = _axis_limits(
            [item.observed_xy for item in slices]
            + [item.aligned_xy for item in slices],
            pad_ratio=0.03,
        )
        for ax in (ax_before, ax_after):
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if invert_y:
                ax.invert_yaxis()
    else:
        before_xlim, before_ylim = _axis_limits(
            [item.observed_xy for item in slices], pad_ratio=0.03
        )
        after_xlim, after_ylim = _axis_limits(
            [item.aligned_xy for item in slices], pad_ratio=0.03
        )
        ax_before.set_xlim(*before_xlim)
        ax_before.set_ylim(*before_ylim)
        ax_after.set_xlim(*after_xlim)
        ax_after.set_ylim(*after_ylim)
        if invert_y:
            ax_before.invert_yaxis()
            ax_after.invert_yaxis()

    if show_marker_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=legend_marker_size,
                label=(
                    background_legend_name
                    if str(label) == str(background_label)
                    else str(label)
                ),
            )
            for label, color in color_map.items()
        ]

        if legend_ncol is None:
            # Two columns works well for approximately 20 marker classes.
            legend_ncol = 2 if len(handles) > 10 else 1

        legend = ax_legend.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            ncol=legend_ncol,
            frameon=False,
            title=legend_title,
            fontsize=legend_fontsize,
            handletextpad=0.35,
            columnspacing=0.8,
            labelspacing=0.22,
            borderaxespad=0.0,
        )

        if legend.get_title() is not None:
            legend.get_title().set_fontsize(legend_title_fontsize)

    if save_path is not None:
        output_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, slices, color_map


if __name__ == "__main__":
    plot_xenium_breast_axis_alignment_overview(
        root="/media/huifang/data/registration/breast_cancer",
        group_id=0,
        sample_ids=(0, 1),
        reference_id=0,
        background_point_size=0.8,
        marker_point_size=5.0,
        background_alpha=0.16,
        marker_alpha=0.82,
        save_path=None,
        show=True,
    )
