from __future__ import annotations
from axis.settings import data_path
from axis.config import load_config

"""Plot DLPFC AXIS alignment results as three separate figures.

This streamlined version separates the original 3-panel overview into three
standalone figures:
    1) stacked native slices,
    2) direct overlap before AXIS,
    3) direct overlap after AXIS.

Key updates
-----------
- Each content type is saved/plotted as an individual figure.
- Discrete domain/layer inputs use an explicit categorical color palette.
- Each figure has its own layer legend on the right side.
- Figure geometry is centralized in dataclass-based layout configs for easy
  tuning.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.lines import Line2D


# =====================================================================
# DATA CONTAINER
# =====================================================================

@dataclass
class DLPFCSlice:
    sample_id: Any
    display_name: str
    observed_xy: np.ndarray
    aligned_xy: np.ndarray
    labels: np.ndarray


# =====================================================================
# LAYOUT / STYLE CONFIG
# =====================================================================

@dataclass
class FigureLayout:
    figsize: Tuple[float, float] = (6.2, 5.3)
    left: float = 0.03
    right: float = 0.78
    bottom: float = 0.03
    top: float = 0.92
    legend_anchor: Tuple[float, float] = (1.01, 0.5)
    dpi: int = 350


@dataclass
class AlignmentPlotStyle:
    # General titles / markers
    title_fontsize: float = 24.0
    slice_label_fontsize: float = 15.0
    point_size: float = 3.2
    stacked_point_size: Optional[float] = None

    # Opacity
    stacked_alpha: float = 0.75
    overlap_alpha: float = 0.40
    aligned_alpha: float = 0.40

    # Stacking geometry
    stack_dx_fraction: float = 0.17
    stack_dy_fraction: float = -0.13

    # Axis limits padding
    stack_pad_ratio: float = 0.05
    overlap_pad_ratio: float = 0.035

    # Discrete categorical palette
    cmap_name: str = "tab20"

    # Legend
    show_domain_legend: bool = True
    legend_title: str = "Layer annotation"
    legend_fontsize: float = 18.0
    legend_title_fontsize: float = 11.0
    legend_markersize: float = 8

    # Layouts for each figure type
    stacked_layout: FigureLayout = field(default_factory=lambda: FigureLayout(figsize=(6.8, 5.6), right=0.79))
    before_layout: FigureLayout = field(default_factory=lambda: FigureLayout(figsize=(5.8, 5.2), right=0.78))
    after_layout: FigureLayout = field(default_factory=lambda: FigureLayout(figsize=(5.8, 5.2), right=0.78))


# =====================================================================
# LOADING HELPERS
# =====================================================================

def _find_group_dir(
    root: str,
    group_id: Any,
    pairwise_subdir: str = "pairwise_output",
) -> str:
    candidates = [
        os.path.join(root, pairwise_subdir, f"group_{group_id}"),
        os.path.join(root, f"group_{group_id}"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "Cannot find the AXIS group directory. Tried:\n  "
        + "\n  ".join(candidates)
    )


def _h5ad_path(root: str, group_id: Any, sample_id: Any) -> str:
    return os.path.join(root, "h5ad", f"{group_id}_{sample_id}.h5ad")


def _network_coord_path(root: str, group_id: Any, sample_id: Any) -> str:
    return os.path.join(root, "network_input", f"{group_id}_{sample_id}_validation.npz")


def _load_observed_slice(
    root: str,
    group_id: Any,
    sample_id: Any,
    obs_key: str,
    coord_key: str = "spatial",
) -> Tuple[np.ndarray, np.ndarray]:
    path = _h5ad_path(root, group_id, sample_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing h5ad file: {path}")

    adata = sc.read_h5ad(path)

    if coord_key in adata.obsm:
        observed_xy = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    else:
        coord_path = _network_coord_path(root, group_id, sample_id)
        if not os.path.exists(coord_path):
            raise KeyError(
                f"obsm['{coord_key}'] is absent and fallback coordinate file does not exist: {coord_path}"
            )
        with np.load(coord_path) as npz:
            if "coord" not in npz:
                raise KeyError(f"'coord' not found in {coord_path}")
            observed_xy = np.asarray(npz["coord"], dtype=np.float32)

    if observed_xy.ndim != 2 or observed_xy.shape[1] < 2:
        raise ValueError(
            f"Coordinates for sample {sample_id} must have shape (N, 2+); received {observed_xy.shape}."
        )

    if obs_key not in adata.obs.columns:
        raise KeyError(
            f"obs['{obs_key}'] is missing from sample {sample_id}. Available columns include: {list(adata.obs.columns[:20])}"
        )

    labels = adata.obs[obs_key].astype("string").fillna("NA").astype(str).to_numpy()
    n = min(observed_xy.shape[0], labels.shape[0])
    return observed_xy[:n, :2], labels[:n]


def _load_axis_aligned_coordinates(
    root: str,
    group_id: Any,
    sample_id: Any,
    reference_id: Any,
    observed_xy: np.ndarray,
    pairwise_subdir: str = "pairwise_output",
    warped_coord_template: str = "{reference_id}_{sample_id}_warped_coor.npy",
) -> np.ndarray:
    if sample_id == reference_id:
        return np.asarray(observed_xy, dtype=np.float32).copy()

    group_dir = _find_group_dir(root, group_id, pairwise_subdir)
    filename = warped_coord_template.format(
        reference_id=reference_id,
        sample_id=sample_id,
        group_id=group_id,
    )
    warped_path = os.path.join(group_dir, filename)
    if not os.path.exists(warped_path):
        raise FileNotFoundError(
            f"Missing AXIS warped coordinate file for sample {sample_id}:\n  {warped_path}"
        )

    aligned_xy = np.asarray(np.load(warped_path), dtype=np.float32)
    if aligned_xy.ndim != 2 or aligned_xy.shape[1] < 2:
        raise ValueError(
            f"Warped coordinates in {warped_path} must have shape (N, 2+); received {aligned_xy.shape}."
        )
    return aligned_xy[:, :2]


def _prepare_dlpfc_slices(
    root: str,
    group_id: Any,
    sample_ids: Sequence[Any],
    reference_id: Any,
    obs_key: str,
    coord_key: str,
    pairwise_subdir: str,
    slice_names: Optional[Mapping[Any, str]],
    warped_coord_template: str,
) -> List[DLPFCSlice]:
    if not sample_ids:
        raise ValueError("sample_ids cannot be empty.")
    if reference_id not in sample_ids:
        raise ValueError(f"reference_id={reference_id} must be included in sample_ids={sample_ids}.")

    slices: List[DLPFCSlice] = []
    for position, sample_id in enumerate(sample_ids, start=1):
        observed_xy, labels = _load_observed_slice(
            root=root,
            group_id=group_id,
            sample_id=sample_id,
            obs_key=obs_key,
            coord_key=coord_key,
        )
        aligned_xy = _load_axis_aligned_coordinates(
            root=root,
            group_id=group_id,
            sample_id=sample_id,
            reference_id=reference_id,
            observed_xy=observed_xy,
            pairwise_subdir=pairwise_subdir,
            warped_coord_template=warped_coord_template,
        )

        n = min(len(observed_xy), len(aligned_xy), len(labels))
        observed_xy = observed_xy[:n, :2]
        aligned_xy = aligned_xy[:n, :2]
        labels = labels[:n]

        display_name = slice_names.get(sample_id, f"Slice {position}") if slice_names else f"Slice {position}"
        slices.append(
            DLPFCSlice(
                sample_id=sample_id,
                display_name=display_name,
                observed_xy=observed_xy,
                aligned_xy=aligned_xy,
                labels=labels,
            )
        )
    return slices


# =====================================================================
# CATEGORY / COLOR HELPERS
# =====================================================================

def _natural_key(value: Any) -> List[Any]:
    text = str(value)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def _shared_category_order(
    label_arrays: Sequence[np.ndarray],
    category_order: Optional[Sequence[Any]] = None,
) -> List[str]:
    if category_order is not None:
        return [str(x) for x in category_order]

    observed: List[str] = []
    seen = set()
    for labels in label_arrays:
        for label in labels:
            label = str(label)
            if label not in seen:
                seen.add(label)
                observed.append(label)
    return sorted(observed, key=_natural_key)


def _make_domain_color_map(
    categories: Sequence[str],
    cmap_name: str = "tab20",
    domain_colors: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    """Create a stable discrete domain color dictionary."""
    if domain_colors is not None:
        missing = [c for c in categories if c not in domain_colors]
        if missing:
            raise KeyError("domain_colors does not define colors for: " + ", ".join(map(str, missing)))
        return {str(c): domain_colors[c] for c in categories}

    cmap = plt.get_cmap(cmap_name)
    if hasattr(cmap, "colors") and cmap.colors is not None:
        palette = list(cmap.colors)
    else:
        palette = [cmap(i / max(len(categories) - 1, 1)) for i in range(max(len(categories), 1))]

    # Cycle if categories exceed native discrete palette length.
    color_map = {}
    for i, category in enumerate(categories):
        color_map[category] = palette[i % len(palette)]
    return color_map


# =====================================================================
# DRAWING HELPERS
# =====================================================================

def _finite_rows(coords: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(coords), axis=1)


def _scatter_domains(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    color_map: Mapping[str, Any],
    point_size: float,
    alpha: float,
    zorder: int = 2,
):
    coords = np.asarray(coords)
    labels = np.asarray(labels).astype(str)
    valid = _finite_rows(coords)
    if not np.any(valid):
        return

    colors = [color_map[label] for label in labels[valid]]
    ax.scatter(
        coords[valid, 0],
        coords[valid, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
        linewidths=0,
        rasterized=True,
        zorder=zorder,
    )


def _axis_limits(
    coordinate_arrays: Iterable[np.ndarray],
    pad_ratio: float = 0.03,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    finite_arrays = []
    for coords in coordinate_arrays:
        coords = np.asarray(coords)
        finite = coords[_finite_rows(coords)]
        if finite.size:
            finite_arrays.append(finite[:, :2])

    if not finite_arrays:
        raise ValueError("No finite coordinates are available for axis limits.")

    all_coords = np.vstack(finite_arrays)
    low = all_coords.min(axis=0)
    high = all_coords.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    pad = span * pad_ratio

    return (
        (float(low[0] - pad[0]), float(high[0] + pad[0])),
        (float(low[1] - pad[1]), float(high[1] + pad[1])),
    )


def _style_spatial_axis(ax, title: str, title_fontsize: float, invert_y: bool = True):
    ax.set_title(title, fontsize=title_fontsize, pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if invert_y:
        ax.invert_yaxis()


def _make_legend_handles(
    categories: Sequence[str],
    color_map: Mapping[str, Any],
    markersize: float,
) -> List[Line2D]:
    return [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color_map[category],
            markeredgecolor="none",
            markersize=markersize,
            label=category,
        )
        for category in categories
    ]


def _attach_right_legend(
    fig,
    ax,
    categories: Sequence[str],
    color_map: Mapping[str, Any],
    style: AlignmentPlotStyle,
    layout: FigureLayout,
):
    if not style.show_domain_legend:
        return None

    handles = _make_legend_handles(
        categories,
        color_map,
        markersize=style.legend_markersize,
    )
    legend = ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=layout.legend_anchor,
        frameon=False,
        fontsize=style.legend_fontsize,
        # title=style.legend_title,
        borderaxespad=0.0,
        handletextpad=0.35,
        labelspacing=0.45,
    )
    if style.legend_title:
        legend.get_title().set_fontsize(style.legend_title_fontsize)
        legend.get_title().set_fontweight("bold")
    return legend


def _new_figure(layout: FigureLayout):
    fig, ax = plt.subplots(figsize=layout.figsize, constrained_layout=False)
    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
    )
    return fig, ax


def _save_figure(fig, save_path: Optional[str], dpi: int):
    if save_path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[OK] saved: {save_path}")


# =====================================================================
# INDIVIDUAL FIGURE PLOTTING
# =====================================================================

def plot_stacked_native_slices(
    slices: Sequence[DLPFCSlice],
    color_map: Mapping[str, Any],
    categories: Sequence[str],
    dataset_title: str,
    style: AlignmentPlotStyle,
    invert_y: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
):
    layout = style.stacked_layout
    fig, ax = _new_figure(layout)

    stacked_point_size = style.point_size if style.stacked_point_size is None else style.stacked_point_size

    native_widths, native_heights = [], []
    for slice_data in slices:
        valid = slice_data.observed_xy[_finite_rows(slice_data.observed_xy)]
        if valid.size:
            span = valid.max(axis=0) - valid.min(axis=0)
            native_widths.append(span[0])
            native_heights.append(span[1])
    if not native_widths or not native_heights:
        raise ValueError("No finite native coordinates were found.")

    stack_dx = float(np.median(native_widths) * style.stack_dx_fraction)
    stack_dy = float(np.median(native_heights) * style.stack_dy_fraction)

    stacked_coordinate_arrays = []
    for index in reversed(range(len(slices))):
        slice_data = slices[index]
        offset = np.array([index * stack_dx, index * stack_dy], dtype=float)
        shifted_xy = slice_data.observed_xy + offset
        stacked_coordinate_arrays.append(shifted_xy)

        _scatter_domains(
            ax,
            shifted_xy,
            slice_data.labels,
            color_map,
            point_size=stacked_point_size,
            alpha=style.stacked_alpha,
            zorder=2 + (len(slices) - index),
        )

        finite = shifted_xy[_finite_rows(shifted_xy)]
        if finite.size:
            x_label = float(np.nanmin(finite[:, 0]))
            y_label = float(np.nanmin(finite[:, 1]))
            ax.text(
                x_label,
                y_label,
                slice_data.display_name,
                ha="left",
                va="bottom",
                fontsize=style.slice_label_fontsize,
                fontweight="bold",
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
                zorder=20,
            )

    xlim, ylim = _axis_limits(stacked_coordinate_arrays, pad_ratio=style.stack_pad_ratio)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _style_spatial_axis(ax, f"{dataset_title}, {len(slices)} slices", style.title_fontsize, invert_y=invert_y)
    # _attach_right_legend(fig, ax, categories, color_map, style, layout)

    _save_figure(fig, save_path, layout.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_overlap_before_axis(
    slices: Sequence[DLPFCSlice],
    color_map: Mapping[str, Any],
    categories: Sequence[str],
    style: AlignmentPlotStyle,
    invert_y: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    layout = style.before_layout
    fig, ax = _new_figure(layout)

    for slice_data in slices:
        _scatter_domains(
            ax,
            slice_data.observed_xy,
            slice_data.labels,
            color_map,
            point_size=style.point_size,
            alpha=style.overlap_alpha,
        )

    _style_spatial_axis(ax, "Before AXIS", style.title_fontsize, invert_y=invert_y)
    if xlim is None or ylim is None:
        xlim, ylim = _axis_limits([slice_data.observed_xy for slice_data in slices], pad_ratio=style.overlap_pad_ratio)
    ax.set_xlim(*xlim)
    if invert_y:
        ax.set_ylim(ylim[1], ylim[0])
    else:
        ax.set_ylim(*ylim)

    # _attach_right_legend(fig, ax, categories, color_map, style, layout)
    _save_figure(fig, save_path, layout.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_overlap_after_axis(
    slices: Sequence[DLPFCSlice],
    color_map: Mapping[str, Any],
    categories: Sequence[str],
    style: AlignmentPlotStyle,
    invert_y: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    layout = style.after_layout
    fig, ax = _new_figure(layout)

    for slice_data in slices:
        _scatter_domains(
            ax,
            slice_data.aligned_xy,
            slice_data.labels,
            color_map,
            point_size=style.point_size,
            alpha=style.aligned_alpha,
        )

    _style_spatial_axis(ax, "After AXIS", style.title_fontsize, invert_y=invert_y)
    if xlim is None or ylim is None:
        xlim, ylim = _axis_limits([slice_data.aligned_xy for slice_data in slices], pad_ratio=style.overlap_pad_ratio)
    ax.set_xlim(*xlim)
    if invert_y:
        ax.set_ylim(ylim[1], ylim[0])
    else:
        ax.set_ylim(*ylim)

    _attach_right_legend(fig, ax, categories, color_map, style, layout)
    _save_figure(fig, save_path, layout.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


# =====================================================================
# WRAPPER
# =====================================================================

def plot_dlpfc_axis_alignment_figures(
    root: str = load_config('dlpfc').root,
    group_id: Any = 1,
    sample_ids: Sequence[Any] = (1, 2, 3, 4),
    reference_id: Any = 1,
    obs_key: str = "layer_guess_reordered",
    coord_key: str = "spatial",
    pairwise_subdir: str = "pairwise_output",
    warped_coord_template: str = "{reference_id}_{sample_id}_warped_coor.npy",
    dataset_title: str = "Human brain",
    slice_names: Optional[Mapping[Any, str]] = None,
    category_order: Optional[Sequence[Any]] = None,
    domain_colors: Optional[Mapping[Any, Any]] = None,
    invert_y: bool = True,
    share_overlap_limits: bool = True,
    style: Optional[AlignmentPlotStyle] = None,
    save_dir: Optional[str] = None,
    file_prefix: str = "dlpfc_axis_alignment",
    show: bool = True,
):
    """Generate three separate figures for DLPFC alignment results.

    Returns
    -------
    output : dict
        {
            "slices": List[DLPFCSlice],
            "categories": List[str],
            "color_map": Dict[str, color],
            "figures": {"stacked": fig, "before": fig, "after": fig},
        }
    """
    style = AlignmentPlotStyle() if style is None else style

    slices = _prepare_dlpfc_slices(
        root=root,
        group_id=group_id,
        sample_ids=list(sample_ids),
        reference_id=reference_id,
        obs_key=obs_key,
        coord_key=coord_key,
        pairwise_subdir=pairwise_subdir,
        slice_names=slice_names,
        warped_coord_template=warped_coord_template,
    )

    categories = _shared_category_order(
        [slice_data.labels for slice_data in slices],
        category_order=category_order,
    )
    color_map = _make_domain_color_map(
        categories,
        cmap_name=style.cmap_name,
        domain_colors=domain_colors,
    )

    stacked_save = before_save = after_save = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stacked_save = os.path.join(save_dir, f"{file_prefix}_stacked_native.png")
        before_save = os.path.join(save_dir, f"{file_prefix}_before_axis.png")
        after_save = os.path.join(save_dir, f"{file_prefix}_after_axis.png")

    if share_overlap_limits:
        shared_xlim, shared_ylim = _axis_limits(
            [
                *(slice_data.observed_xy for slice_data in slices),
                *(slice_data.aligned_xy for slice_data in slices),
            ],
            pad_ratio=style.overlap_pad_ratio,
        )
    else:
        shared_xlim = shared_ylim = None

    fig_stacked, _ = plot_stacked_native_slices(
        slices=slices,
        color_map=color_map,
        categories=categories,
        dataset_title=dataset_title,
        style=style,
        invert_y=invert_y,
        save_path=stacked_save,
        show=show,
    )
    fig_before, _ = plot_overlap_before_axis(
        slices=slices,
        color_map=color_map,
        categories=categories,
        style=style,
        invert_y=invert_y,
        xlim=shared_xlim,
        ylim=shared_ylim,
        save_path=before_save,
        show=show,
    )
    fig_after, _ = plot_overlap_after_axis(
        slices=slices,
        color_map=color_map,
        categories=categories,
        style=style,
        invert_y=invert_y,
        xlim=shared_xlim,
        ylim=shared_ylim,
        save_path=after_save,
        show=show,
    )

    return {
        "slices": slices,
        "categories": categories,
        "color_map": color_map,
        "figures": {
            "stacked": fig_stacked,
            "before": fig_before,
            "after": fig_after,
        },
    }


if __name__ == "__main__":
    plot_dlpfc_axis_alignment_figures(
        root=load_config('dlpfc').root,
        group_id=1,
        sample_ids=(0, 1, 2, 3),
        reference_id=1,
        obs_key="layer_guess_reordered",
        coord_key="spatial",
        dataset_title="Human brain",
        style=AlignmentPlotStyle(
            cmap_name="tab10",
            point_size=10,
            overlap_alpha=0.58,
            aligned_alpha=0.58,
            stack_dx_fraction=0.17,
            stack_dy_fraction=-0.13,
        ),
        save_dir=data_path("paper_analysis/Figure2_transition_evaluation/a/"),
        file_prefix="dlpfc_group1",
        show=False,
    )
