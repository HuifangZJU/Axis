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
from axis.settings import data_path
from axis.config import load_config

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import get_cmap  # matplotlib.cm.get_cmap was removed in matplotlib 3.9
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
    # ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if invert_y:
        ax.invert_yaxis()




from dataclasses import field


@dataclass
class FigureLayout:
    figsize: Tuple[float, float] = (6.2, 5.2)
    left: float = 0.03
    right: float = 0.80
    bottom: float = 0.03
    top: float = 0.92
    legend_anchor: Tuple[float, float] = (1.01, 0.5)
    dpi: int = 350


@dataclass
class BreastAlignmentStyle:
    title_fontsize: float = 18.0
    slice_label_fontsize: float = 16.0
    background_point_size: float = 0.8
    marker_point_size: float = 2.0
    stacked_marker_point_size: Optional[float] = None

    background_alpha: float = 0.36
    marker_alpha: float = 0.82

    stack_dx_fraction: float = 0.17
    stack_dy_fraction: float = -0.13
    stack_pad_ratio: float = 0.05
    overlap_pad_ratio: float = 0.03

    cmap_name: str = "tab20"
    background_label: Any = 0
    background_color: Any = "#BDBDBD"
    background_legend_name: str = "Background"

    show_marker_legend: bool = True
    legend_title: str = "Manual landmarks"
    legend_fontsize: float = 15
    legend_title_fontsize: float = 14.0
    legend_markersize: float = 6.8

    stacked_layout: FigureLayout = field(default_factory=lambda: FigureLayout(figsize=(6.8, 5.5), right=0.80))
    before_layout: FigureLayout = field(default_factory=lambda: FigureLayout(figsize=(5.8, 5.1), right=0.80))
    after_layout: FigureLayout = field(default_factory=lambda: FigureLayout(figsize=(5.8, 5.1), right=0.80))


def _new_figure(layout: FigureLayout):
    fig, ax = plt.subplots(figsize=layout.figsize, constrained_layout=False)
    fig.subplots_adjust(left=layout.left, right=layout.right, bottom=layout.bottom, top=layout.top)
    return fig, ax


def _save_figure(fig, save_path: Optional[str], dpi: int):
    if save_path is None:
        return
    output_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[OK] saved: {save_path}")


def _attach_breast_legend(
    ax,
    color_map,
    style: BreastAlignmentStyle,
    layout: FigureLayout,
):
    """
    Manual landmarks are shown in two columns.
    Background is placed separately as the bottom-most full-width entry.
    """
    if not style.show_marker_legend:
        return None

    background = str(style.background_label)

    # ---------------------------------------------------------
    # Numbered manual landmarks: two columns
    # ---------------------------------------------------------
    marker_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=style.legend_markersize,
            label=str(label),
        )
        for label, color in color_map.items()
        if str(label) != background
    ]

    marker_legend = ax.legend(
        handles=marker_handles,
        loc="center left",
        bbox_to_anchor=(
            layout.legend_anchor[0],
            layout.legend_anchor[1] + 0.055,
        ),
        ncol=2,
        columnspacing=0.70,
        frameon=False,
        title=style.legend_title,
        fontsize=style.legend_fontsize,
        borderaxespad=0.0,
        handletextpad=0.35,
        labelspacing=0.28,
    )

    marker_legend.get_title().set_fontsize(
        style.legend_title_fontsize
    )
    marker_legend.get_title().set_fontweight("bold")

    # Preserve the landmark legend when adding Background.
    ax.add_artist(marker_legend)

    # ---------------------------------------------------------
    # Background: separate bottom row
    # ---------------------------------------------------------
    background_legend = None

    if background in color_map:
        background_handle = Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color_map[background],
            markeredgecolor="none",
            markersize=style.legend_markersize,
            label=style.background_legend_name,
        )

        background_legend = ax.legend(
            handles=[background_handle],
            loc="upper left",
            bbox_to_anchor=(
                layout.legend_anchor[0]+0.04,
                0.23,
            ),
            ncol=1,
            frameon=False,
            fontsize=style.legend_fontsize,
            borderaxespad=0.0,
            handletextpad=0.35,
            labelspacing=0.28,
        )

    return marker_legend, background_legend


def plot_breast_stacked_native(slices, color_map, dataset_title: str, style: BreastAlignmentStyle,
                               invert_y: bool = True, save_path: Optional[str] = None, show: bool = True):
    layout = style.stacked_layout
    fig, ax = _new_figure(layout)
    widths, heights = [], []
    for item in slices:
        finite = item.observed_xy[_finite_rows(item.observed_xy)]
        if finite.size:
            span = finite.max(axis=0) - finite.min(axis=0)
            widths.append(float(span[0])); heights.append(float(span[1]))
    if not widths or not heights:
        raise ValueError("No finite native coordinates were found.")
    stack_dx = float(np.median(widths) * style.stack_dx_fraction)
    stack_dy = float(np.median(heights) * style.stack_dy_fraction)
    marker_ps = style.marker_point_size if style.stacked_marker_point_size is None else style.stacked_marker_point_size

    stacked_arrays = []
    for index in reversed(range(len(slices))):
        item = slices[index]
        shifted = item.observed_xy + np.asarray([index * stack_dx, index * stack_dy], dtype=float)
        stacked_arrays.append(shifted)
        _scatter_labels(ax=ax, coords=shifted, labels=item.labels, color_map=color_map,
                        background_label=style.background_label, background_point_size=style.background_point_size,
                        marker_point_size=marker_ps, background_alpha=style.background_alpha,
                        marker_alpha=style.marker_alpha, zorder=2 + (len(slices) - index))
        finite = shifted[_finite_rows(shifted)]
        if finite.size:
            ax.text(float(np.nanmin(finite[:,0])), float(np.nanmin(finite[:,1])), item.display_name,
                    ha='left', va='bottom', fontsize=style.slice_label_fontsize, fontweight='bold',
                    bbox={'facecolor':'white','edgecolor':'none','alpha':0.84,'pad':1.4}, zorder=50)

    xlim, ylim = _axis_limits(stacked_arrays, pad_ratio=style.stack_pad_ratio)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    _style_axis(ax, f"{dataset_title}, {len(slices)} slices", invert_y=invert_y)
    # _attach_breast_legend(ax, color_map, style, layout)
    _save_figure(fig, save_path, layout.dpi)
    if show: plt.show()
    else: plt.close(fig)
    return fig, ax


def plot_breast_overlap_before(slices, color_map, style: BreastAlignmentStyle, invert_y=True, xlim=None, ylim=None,
                               save_path=None, show=True):
    layout = style.before_layout
    fig, ax = _new_figure(layout)
    for item in slices:
        _scatter_labels(ax=ax, coords=item.observed_xy, labels=item.labels, color_map=color_map,
                        background_label=style.background_label, background_point_size=style.background_point_size,
                        marker_point_size=style.marker_point_size, background_alpha=style.background_alpha,
                        marker_alpha=style.marker_alpha, zorder=2)
    _style_axis(ax, "Before AXIS", invert_y=invert_y)
    if xlim is None or ylim is None:
        xlim, ylim = _axis_limits([item.observed_xy for item in slices], pad_ratio=style.overlap_pad_ratio)
    ax.set_xlim(*xlim)
    if invert_y: ax.set_ylim(ylim[1], ylim[0])
    else: ax.set_ylim(*ylim)
    # _attach_breast_legend(ax, color_map, style, layout)
    _save_figure(fig, save_path, layout.dpi)
    if show: plt.show()
    else: plt.close(fig)
    return fig, ax


def plot_breast_overlap_after(slices, color_map, style: BreastAlignmentStyle, invert_y=True, xlim=None, ylim=None,
                              save_path=None, show=True):
    layout = style.after_layout
    fig, ax = _new_figure(layout)
    for item in slices:
        _scatter_labels(ax=ax, coords=item.aligned_xy, labels=item.labels, color_map=color_map,
                        background_label=style.background_label, background_point_size=style.background_point_size,
                        marker_point_size=style.marker_point_size, background_alpha=style.background_alpha,
                        marker_alpha=style.marker_alpha, zorder=2)
    _style_axis(ax, "After AXIS", invert_y=invert_y)
    if xlim is None or ylim is None:
        xlim, ylim = _axis_limits([item.aligned_xy for item in slices], pad_ratio=style.overlap_pad_ratio)
    ax.set_xlim(*xlim)
    if invert_y: ax.set_ylim(ylim[1], ylim[0])
    else: ax.set_ylim(*ylim)
    _attach_breast_legend(ax, color_map, style, layout)
    _save_figure(fig, save_path, layout.dpi)
    if show: plt.show()
    else: plt.close(fig)
    return fig, ax


def plot_xenium_breast_axis_alignment_figures(
    group_id: Any = 0,
    sample_ids: Sequence[Any] = (0, 1),
    reference_id: Any = 0,
    root: str = load_config('xenium_breast').root,
    axis_output_subdir: str = "competing_methods/axis",
    result_template: str = "{fixed_id}_{moving_id}_result.npz",
    marker_label_subdir: str = "huifang/network_input",
    marker_label_template: str = "{group_id}_{sample_id}_marker_label_img.npz",
    marker_label_key: Optional[str] = None,
    dataset_title: str = "Xenium breast cancer",
    slice_names: Optional[Mapping[Any, str]] = None,
    label_colors: Optional[Mapping[Any, Any]] = None,
    label_order: Optional[Sequence[Any]] = None,
    invert_y: bool = True,
    share_overlap_limits: bool = True,
    style: Optional[BreastAlignmentStyle] = None,
    save_dir: Optional[str] = data_path("paper_analysis/Figure2_transition_evaluation/f/"),
    file_prefix: str = "breast_axis_alignment",
    show: bool = True,
):
    style = BreastAlignmentStyle() if style is None else style
    slices, _, _ = _prepare_two_sample_slices(root=root, group_id=group_id, sample_ids=sample_ids, reference_id=reference_id,
                                              axis_output_subdir=axis_output_subdir, result_template=result_template,
                                              marker_label_subdir=marker_label_subdir, marker_label_template=marker_label_template,
                                              marker_label_key=marker_label_key, slice_names=slice_names)
    label_order_use = [str(x) for x in label_order] if label_order is not None else None
    color_map = _make_label_color_map(labels_by_slice=np.concatenate([item.labels for item in slices]), label_order=label_order_use,
                                      cmap_name=style.cmap_name, label_colors=label_colors,
                                      background_label=style.background_label, background_color=style.background_color)
    if share_overlap_limits:
        shared_xlim, shared_ylim = _axis_limits([item.observed_xy for item in slices] + [item.aligned_xy for item in slices],
                                                pad_ratio=style.overlap_pad_ratio)
    else:
        shared_xlim = shared_ylim = None
    stacked_save = before_save = after_save = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stacked_save = os.path.join(save_dir, f"{file_prefix}_stacked_native.png")
        before_save = os.path.join(save_dir, f"{file_prefix}_before_axis.png")
        after_save = os.path.join(save_dir, f"{file_prefix}_after_axis.png")

    fig1,_ = plot_breast_stacked_native(slices, color_map, dataset_title, style, invert_y, stacked_save, show)
    # fig2,_ = plot_breast_overlap_before(slices, color_map, style, invert_y, shared_xlim, shared_ylim, before_save, show)
    fig3,_ = plot_breast_overlap_after(slices, color_map, style, invert_y, shared_xlim, shared_ylim, after_save, show)
    # return {"slices": slices, "color_map": color_map, "figures": {"stacked": fig1, "before": fig2, "after": fig3}}


if __name__ == "__main__":
    plot_xenium_breast_axis_alignment_figures(show=True)
