"""Standalone AXIS alignment figures for multi-slice Xenium mouse brain.

Figures
-------
1. Native measurements displayed as spatially offset/stacked slices.
2. Direct overlap of all native slice coordinates before AXIS.
3. Overlap of all AXIS-warped coordinates in a common reference frame.

Only selected spatial domains are assigned categorical colors. All remaining
spatial domains are shown in gray. Domain colors follow the shared mouse-time
convention: domain IDs 0..12 map directly to tab20 indices 0..12.

``swap_xy`` optionally exchanges x/y for both native and aligned coordinates
before plotting, so all three figures use the same display orientation.

Expected file organization
--------------------------
root/
├── h5ads/
│   ├── {group_id}_{sample_id}.h5ad
│   └── ...
├── network_input/
│   ├── {group_id}_{sample_id}_validation.npz
│   └── ...
└── pairwise_output/
    └── group_{group_id}/
        ├── {reference_id}_{sample_id}_warped_coor.npy
        └── ...

The h5ad files provide domain annotations only. Native coordinates are always
loaded from ``network_input/{group_id}_{sample_id}_validation.npz``. The
reference slice keeps these native network-input coordinates, whereas every
nonreference slice uses the corresponding AXIS ``warped_coor.npy`` output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D


# =====================================================================
# SHARED MOUSE-DOMAIN COLOR CONVENTION
# =====================================================================
# Keep this identical to the mouse-time plotting pipeline:
#   ALL_LABELS = list(range(13))
#   BASE_CMAP = plt.get_cmap("tab20")
#   LABEL_TO_COLOR = {lab: BASE_CMAP(i) for i, lab in enumerate(ALL_LABELS)}
MOUSE_DOMAIN_IDS = tuple(range(13))

MOUSE_DOMAIN_CMAP_NAME = "tab20"


@dataclass
class MouseXeniumSlice:
    """Minimal information required to plot one Xenium mouse-brain slice."""

    sample_id: Any
    display_name: str
    observed_xy: np.ndarray
    aligned_xy: np.ndarray
    labels: np.ndarray


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


def _h5ad_path(
    root: str,
    group_id: Any,
    sample_id: Any,
    h5ad_subdir: str = "h5ads",
) -> str:
    return os.path.join(root, h5ad_subdir, f"{group_id}_{sample_id}.h5ad")


def _network_coord_path(
    root: str,
    group_id: Any,
    sample_id: Any,
    network_input_subdir: str = "network_input",
) -> str:
    return os.path.join(
        root,
        network_input_subdir,
        f"{group_id}_{sample_id}_validation.npz",
    )


def _load_network_input_coords(
    root: str,
    group_id: Any,
    sample_id: Any,
    network_input_subdir: str = "network_input",
) -> np.ndarray:
    """Load the native AXIS input coordinates for one slice.

    The coordinate order must match the row order of the corresponding h5ad
    object because the h5ad contributes the per-cell domain annotations.
    """
    path = _network_coord_path(
        root=root,
        group_id=group_id,
        sample_id=sample_id,
        network_input_subdir=network_input_subdir,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing network-input coordinate file: {path}")

    with np.load(path) as data:
        if "coord" not in data:
            raise KeyError(f"'coord' not found in {path}")
        coords = np.asarray(data["coord"], dtype=np.float32)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"Coordinates in {path} must have shape (N, 2+); "
            f"received {coords.shape}."
        )
    return coords[:, :2]


def _load_observed_slice(
    root: str,
    group_id: Any,
    sample_id: Any,
    obs_key: str,
    h5ad_subdir: str = "h5ads",
    network_input_subdir: str = "network_input",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load NPZ native coordinates and h5ad domain labels for one slice."""
    h5ad_path = _h5ad_path(
        root=root,
        group_id=group_id,
        sample_id=sample_id,
        h5ad_subdir=h5ad_subdir,
    )
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"Missing h5ad file: {h5ad_path}")

    # The h5ad is intentionally used only for spatial-domain annotations.
    adata = sc.read_h5ad(h5ad_path)
    if obs_key not in adata.obs.columns:
        raise KeyError(
            f"obs['{obs_key}'] is missing from sample {sample_id}. "
            f"Available columns include: {list(adata.obs.columns[:20])}"
        )

    labels = (
        adata.obs[obs_key]
        .astype("string")
        .fillna("NA")
        .astype(str)
        .to_numpy()
    )
    observed_xy = _load_network_input_coords(
        root=root,
        group_id=group_id,
        sample_id=sample_id,
        network_input_subdir=network_input_subdir,
    )

    if observed_xy.shape[0] != labels.shape[0]:
        raise ValueError(
            "Coordinate/annotation length mismatch for sample "
            f"{sample_id}: network-input coordinates contain "
            f"{observed_xy.shape[0]} rows, but obs['{obs_key}'] contains "
            f"{labels.shape[0]} rows. Their row order must be one-to-one."
        )

    return observed_xy, labels


def _load_axis_aligned_coordinates(
    root: str,
    group_id: Any,
    sample_id: Any,
    reference_id: Any,
    observed_xy: np.ndarray,
    pairwise_subdir: str = "pairwise_output",
    warped_coord_template: str = "{reference_id}_{sample_id}_warped_coor.npy",
) -> np.ndarray:
    """Load one slice's AXIS coordinates in the reference coordinate frame."""
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
            f"Missing AXIS warped coordinate file for sample {sample_id}:\n"
            f"  {warped_path}"
        )

    aligned_xy = np.asarray(np.load(warped_path), dtype=np.float32)
    if aligned_xy.ndim != 2 or aligned_xy.shape[1] < 2:
        raise ValueError(
            f"Warped coordinates in {warped_path} must have shape (N, 2+); "
            f"received {aligned_xy.shape}."
        )
    return aligned_xy[:, :2]


def _finite_rows(coords: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(coords), axis=1)


def _make_selected_domain_color_map(
    selected_domain_ids: Sequence[Any],
    cmap_name: str = MOUSE_DOMAIN_CMAP_NAME,
    domain_colors: Optional[Mapping[Any, Any]] = None,
    all_domain_ids: Sequence[int] = MOUSE_DOMAIN_IDS,
) -> Dict[str, Any]:
    """Return selected-domain colors using the shared mouse-time palette.

    The critical point is that the color of a domain depends on its *global*
    domain ID, not on its position in ``selected_domain_ids``. For example,
    domain 8 always receives ``tab20(8)`` even if domains 2--7 are omitted.
    This matches the mouse-time pipeline's ``LABEL_TO_COLOR`` definition.
    """
    selected = [str(x) for x in selected_domain_ids]

    if len(set(selected)) != len(selected):
        raise ValueError("selected_domain_ids contains duplicate values.")

    if domain_colors is not None:
        normalized = {str(key): value for key, value in domain_colors.items()}
        missing = [domain for domain in selected if domain not in normalized]
        if missing:
            raise KeyError(
                "domain_colors does not define colors for selected domains: "
                + ", ".join(missing)
            )
        return {domain: normalized[domain] for domain in selected}

    all_ids = [int(x) for x in all_domain_ids]
    full_cmap = plt.get_cmap(cmap_name)
    full_color_map = {
        str(domain_id): full_cmap(index)
        for index, domain_id in enumerate(all_ids)
    }

    missing = [domain for domain in selected if domain not in full_color_map]
    if missing:
        raise KeyError(
            "Selected domain IDs are outside the shared mouse palette "
            f"{all_ids}: " + ", ".join(missing)
        )

    return {domain: full_color_map[domain] for domain in selected}


def _scatter_selected_domains(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    selected_color_map: Mapping[str, Any],
    gray_color: Any,
    point_size: float,
    alpha: float,
    gray_alpha: Optional[float] = None,
    marker: str = "o",
    zorder: int = 2,
):
    """Plot nonselected domains in gray and selected domains above them."""
    coords = np.asarray(coords)
    labels = np.asarray(labels).astype(str)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must have shape (N, 2+), got {coords.shape}")
    if len(coords) != len(labels):
        raise ValueError(
            f"Coordinate/label length mismatch: {len(coords)} versus {len(labels)}"
        )

    valid = _finite_rows(coords)
    if not np.any(valid):
        return

    selected_set = set(selected_color_map)
    selected_mask = valid & np.isin(labels, list(selected_set))
    gray_mask = valid & ~selected_mask

    gray_alpha = alpha if gray_alpha is None else gray_alpha

    # Draw gray domains first, allowing selected domains to remain visible.
    if np.any(gray_mask):
        ax.scatter(
            coords[gray_mask, 0],
            coords[gray_mask, 1],
            c=[gray_color],
            s=point_size,
            marker=marker,
            alpha=gray_alpha,
            edgecolors="none",
            linewidths=0,
            rasterized=True,
            zorder=zorder,
        )

    # Draw one selected domain at a time. This preserves explicit colors and
    # makes the rendering independent of categorical code ordering.
    for domain, color in selected_color_map.items():
        keep = valid & (labels == domain)
        if not np.any(keep):
            continue
        ax.scatter(
            coords[keep, 0],
            coords[keep, 1],
            c=[color],
            s=point_size,
            marker=marker,
            alpha=alpha,
            edgecolors="none",
            linewidths=0,
            rasterized=True,
            zorder=zorder + 1,
        )


def _axis_limits(
    coordinate_arrays: Iterable[np.ndarray],
    pad_ratio: float = 0.03,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    finite_arrays: List[np.ndarray] = []
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


def _style_spatial_axis(ax, title: str, invert_y: bool = True):
    ax.set_title(title, fontsize=17, pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if invert_y:
        ax.invert_xaxis()


def _prepare_mouse_xenium_slices(
    root: str,
    group_id: Any,
    sample_ids: Sequence[Any],
    reference_id: Any,
    obs_key: str,
    h5ad_subdir: str,
    network_input_subdir: str,
    pairwise_subdir: str,
    slice_names: Optional[Mapping[Any, str]],
    warped_coord_template: str,
) -> List[MouseXeniumSlice]:
    if not sample_ids:
        raise ValueError("sample_ids cannot be empty.")
    if reference_id not in sample_ids:
        raise ValueError(
            f"reference_id={reference_id} must be included in sample_ids={sample_ids}."
        )

    slices: List[MouseXeniumSlice] = []
    for position, sample_id in enumerate(sample_ids, start=1):
        observed_xy, labels = _load_observed_slice(
            root=root,
            group_id=group_id,
            sample_id=sample_id,
            obs_key=obs_key,
            h5ad_subdir=h5ad_subdir,
            network_input_subdir=network_input_subdir,
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

        if aligned_xy.shape[0] != observed_xy.shape[0]:
            raise ValueError(
                f"AXIS/native coordinate length mismatch for sample {sample_id}: "
                f"warped coordinates contain {aligned_xy.shape[0]} rows, but "
                f"network-input coordinates contain {observed_xy.shape[0]} rows."
            )

        observed_xy = observed_xy[:, :2]
        aligned_xy = aligned_xy[:, :2]

        default_name = f"Slice {position}"
        display_name = (
            slice_names.get(sample_id, default_name)
            if slice_names is not None
            else default_name
        )

        slices.append(
            MouseXeniumSlice(
                sample_id=sample_id,
                display_name=display_name,
                observed_xy=observed_xy,
                aligned_xy=aligned_xy,
                labels=labels,
            )
        )

    return slices




def _apply_coordinate_orientation(
    slices: Sequence[MouseXeniumSlice],
    swap_xy: bool = False,
) -> List[MouseXeniumSlice]:
    """Apply display-only coordinate orientation consistently to all slices.

    ``swap_xy=True`` exchanges the first two coordinate columns for both native
    and AXIS-aligned coordinates. This changes plotting orientation only; labels
    and registration outputs are otherwise untouched.
    """
    if not swap_xy:
        return list(slices)

    oriented: List[MouseXeniumSlice] = []
    for item in slices:
        observed_xy = np.asarray(item.observed_xy, dtype=float)[:, [1, 0]].copy()
        aligned_xy = np.asarray(item.aligned_xy, dtype=float)[:, [1, 0]].copy()
        oriented.append(
            MouseXeniumSlice(
                sample_id=item.sample_id,
                display_name=item.display_name,
                observed_xy=observed_xy,
                aligned_xy=aligned_xy,
                labels=item.labels,
            )
        )
    return oriented


@dataclass
class FigureLayout:
    figsize: Tuple[float, float] = (6.4, 5.3)
    left: float = 0.03
    right: float = 0.80
    bottom: float = 0.03
    top: float = 0.92
    legend_anchor: Tuple[float, float] = (0.9, 0.5)
    dpi: int = 350


@dataclass
class MouseAlignmentStyle:
    title_fontsize: float = 18.0
    slice_label_fontsize: float = 14.0
    point_size: float = 1
    stacked_point_size: Optional[float] = None

    stacked_alpha: float = 0.95
    stacked_gray_alpha: float = 0.42
    overlap_alpha: float = 0.42
    overlap_gray_alpha: float = 0.18
    aligned_alpha: float = 0.42
    aligned_gray_alpha: float = 0.18

    stack_dx_fraction: float = 0.01
    stack_dy_fraction: float = -0.01
    stack_pad_ratio: float = 0.01
    overlap_pad_ratio: float = 0.01

    cmap_name: str = MOUSE_DOMAIN_CMAP_NAME
    gray_color: Any = "#C8C8C8"
    gray_legend_label: str = "Other domains"
    show_other_in_legend: bool = True

    show_domain_legend: bool = True
    legend_title: str = "Selected domain annotation"
    legend_fontsize: float = 14
    legend_title_fontsize: float = 11.0
    legend_markersize: float = 6.5

    stacked_layout: FigureLayout = field(default_factory=lambda: FigureLayout(figsize=(6.9, 5.6), right=0.80))
    before_layout: FigureLayout = field(default_factory=lambda: FigureLayout(figsize=(5.9, 5.2), right=0.80))
    after_layout: FigureLayout = field(default_factory=lambda: FigureLayout(figsize=(5.9, 5.2), right=0.80))


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


def _make_mouse_legend_handles(selected_color_map, gray_color, style: MouseAlignmentStyle):
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color, markeredgecolor="none",
               markersize=style.legend_markersize, label=f"Domain {domain}")
        for domain, color in selected_color_map.items()
    ]
    if style.show_other_in_legend:
        handles.append(
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=gray_color, markeredgecolor="none",
                   markersize=style.legend_markersize, label=style.gray_legend_label)
        )
    return handles


def _attach_mouse_legend(ax, selected_color_map, style: MouseAlignmentStyle, layout: FigureLayout):
    if not style.show_domain_legend:
        return None
    handles = _make_mouse_legend_handles(selected_color_map, style.gray_color, style)
    legend = ax.legend(handles=handles, loc="center left", bbox_to_anchor=layout.legend_anchor,
                       frameon=False, fontsize=style.legend_fontsize,
                       borderaxespad=0.0, handletextpad=0.35, labelspacing=0.5)
    legend.get_title().set_fontsize(style.legend_title_fontsize)
    legend.get_title().set_fontweight("bold")
    return legend


def plot_mouse_stacked_native(
    slices: Sequence[MouseXeniumSlice], selected_color_map: Mapping[str, Any], dataset_title: str,
    style: MouseAlignmentStyle, invert_y: bool = True, save_path: Optional[str] = None, show: bool = True,
):
    layout = style.stacked_layout
    fig, ax = _new_figure(layout)
    stacked_point_size = style.point_size if style.stacked_point_size is None else style.stacked_point_size

    native_widths, native_heights = [], []
    for slice_data in slices:
        valid = slice_data.observed_xy[_finite_rows(slice_data.observed_xy)]
        if valid.size:
            span = valid.max(axis=0) - valid.min(axis=0)
            native_widths.append(float(span[0]))
            native_heights.append(float(span[1]))
    if not native_widths or not native_heights:
        raise ValueError("No finite native coordinates were found.")

    stack_dx = float(np.median(native_widths) * style.stack_dx_fraction)
    stack_dy = float(np.median(native_heights) * style.stack_dy_fraction)


    stacked_arrays = []
    for index in reversed(range(len(slices))):
        slice_data = slices[index]
        shifted_xy = slice_data.observed_xy + np.array([index * stack_dx, index * stack_dy], dtype=float)
        stacked_arrays.append(shifted_xy)
        _scatter_selected_domains(
            ax=ax, coords=shifted_xy, labels=slice_data.labels,
            selected_color_map=selected_color_map, gray_color=style.gray_color,
            point_size=stacked_point_size, alpha=style.stacked_alpha,
            gray_alpha=style.stacked_gray_alpha, zorder=2 + (len(slices) - index),
        )
        finite = shifted_xy[_finite_rows(shifted_xy)]
        if finite.size:
            ax.text(float(np.nanmin(finite[:, 0]))+250, float(np.nanmax(finite[:, 1]))-50, slice_data.display_name,
                    ha="left", va="bottom", fontsize=style.slice_label_fontsize, fontweight="bold", color="black",
                    bbox={"facecolor": "none", "edgecolor": "none", "alpha": 0.84, "pad": 0.5}, zorder=50)

    xlim, ylim = _axis_limits(stacked_arrays, pad_ratio=style.stack_pad_ratio)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _style_spatial_axis(ax, f"{dataset_title}, {len(slices)} slices", invert_y=invert_y)
    # _attach_mouse_legend(ax, selected_color_map, style, layout)

    _save_figure(fig, save_path, layout.dpi)
    if show: plt.show()
    else: plt.close(fig)
    return fig, ax


def plot_mouse_overlap_before(
    slices: Sequence[MouseXeniumSlice], selected_color_map: Mapping[str, Any], style: MouseAlignmentStyle,
    invert_y: bool = True, xlim=None, ylim=None, save_path: Optional[str] = None, show: bool = True,
):
    layout = style.before_layout
    fig, ax = _new_figure(layout)
    for slice_data in slices:
        _scatter_selected_domains(ax=ax, coords=slice_data.observed_xy, labels=slice_data.labels,
                                  selected_color_map=selected_color_map, gray_color=style.gray_color,
                                  point_size=2, alpha=style.overlap_alpha,
                                  gray_alpha=style.overlap_gray_alpha)
    _style_spatial_axis(ax, "Before AXIS", invert_y=invert_y)
    if xlim is None or ylim is None:
        xlim, ylim = _axis_limits([s.observed_xy for s in slices], pad_ratio=style.overlap_pad_ratio)
    ax.set_xlim(*xlim)
    if invert_y: ax.set_xlim(xlim[1], xlim[0])
    else: ax.set_ylim(*ylim)
    # _attach_mouse_legend(ax, selected_color_map, style, layout)
    _save_figure(fig, save_path, layout.dpi)
    if show: plt.show()
    else: plt.close(fig)
    return fig, ax


def plot_mouse_overlap_after(
    slices: Sequence[MouseXeniumSlice], selected_color_map: Mapping[str, Any], style: MouseAlignmentStyle,
    invert_y: bool = True, xlim=None, ylim=None, save_path: Optional[str] = None, show: bool = True,
):
    layout = style.after_layout
    fig, ax = _new_figure(layout)
    for slice_data in slices:
        _scatter_selected_domains(ax=ax, coords=slice_data.aligned_xy, labels=slice_data.labels,
                                  selected_color_map=selected_color_map, gray_color=style.gray_color,
                                  point_size=2, alpha=style.aligned_alpha,
                                  gray_alpha=style.aligned_gray_alpha)
    _style_spatial_axis(ax, "After AXIS", invert_y=invert_y)
    if xlim is None or ylim is None:
        xlim, ylim = _axis_limits([s.aligned_xy for s in slices], pad_ratio=style.overlap_pad_ratio)
    ax.set_xlim(*xlim)
    if invert_y: ax.set_xlim(xlim[1], xlim[0])
    else: ax.set_ylim(*ylim)
    _attach_mouse_legend(ax, selected_color_map, style, layout)
    _save_figure(fig, save_path, layout.dpi)
    if show: plt.show()
    else: plt.close(fig)
    return fig, ax


def plot_mouse_xenium_axis_alignment_figures(
    root: str = "/media/huifang/data/registration/mouse_brain",
    group_id: Any = 0,
    sample_ids: Sequence[Any] = (0, 1, 2, 3, 4, 5),
    reference_id: Any = 0,
    obs_key: str = "domain_id_shared",
    h5ad_subdir: str = "h5ads",
    network_input_subdir: str = "network_input",
    pairwise_subdir: str = "pairwise_output",
    warped_coord_template: str = "{reference_id}_{sample_id}_warped_coor.npy",
    selected_domain_ids: Sequence[Any] = (0, 1, 2, 5, 6, 7, 9, 10, 11),
    dataset_title: str = "Xenium mouse brain",
    slice_names: Optional[Mapping[Any, str]] = None,
    domain_colors: Optional[Mapping[Any, Any]] = None,
    swap_xy: bool = True,
    invert_y: bool = True,
    share_overlap_limits: bool = True,
    style: Optional[MouseAlignmentStyle] = None,
    save_dir: Optional[str] = "/media/huifang/data/registration/paper_analysis/Figure2_transition_evaluation/d/",
    file_prefix: str = "mouse_axis_alignment",
    show: bool = True,
):
    style = MouseAlignmentStyle() if style is None else style
    slices = _prepare_mouse_xenium_slices(
        root=root,
        group_id=group_id,
        sample_ids=list(sample_ids),
        reference_id=reference_id,
        obs_key=obs_key,
        h5ad_subdir=h5ad_subdir,
        network_input_subdir=network_input_subdir,
        pairwise_subdir=pairwise_subdir,
        slice_names=slice_names,
        warped_coord_template=warped_coord_template,
    )

    # Plot-only orientation transform, applied identically to native and aligned
    # coordinates before any stack offsets or axis limits are computed.
    slices = _apply_coordinate_orientation(slices, swap_xy=swap_xy)

    selected_color_map = _make_selected_domain_color_map(
        selected_domain_ids=selected_domain_ids,
        cmap_name=style.cmap_name,
        domain_colors=domain_colors,
        all_domain_ids=MOUSE_DOMAIN_IDS,
    )

    if share_overlap_limits:
        shared_xlim, shared_ylim = _axis_limits([
            *(s.observed_xy for s in slices), *(s.aligned_xy for s in slices)
        ], pad_ratio=style.overlap_pad_ratio)
    else:
        shared_xlim = shared_ylim = None

    stacked_save = before_save = after_save = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        stacked_save = os.path.join(save_dir, f"{file_prefix}_stacked_native.png")
        before_save = os.path.join(save_dir, f"{file_prefix}_before_axis.png")
        after_save = os.path.join(save_dir, f"{file_prefix}_after_axis.png")

    fig1, _ = plot_mouse_stacked_native(slices=slices, selected_color_map=selected_color_map,
                                        dataset_title=dataset_title, style=style, invert_y=invert_y,
                                        save_path=stacked_save, show=show)
    fig2, _ = plot_mouse_overlap_before(slices=slices, selected_color_map=selected_color_map,
                                        style=style, invert_y=invert_y, xlim=shared_xlim, ylim=shared_ylim,
                                        save_path=before_save, show=show)
    fig3, _ = plot_mouse_overlap_after(slices=slices, selected_color_map=selected_color_map,
                                       style=style, invert_y=invert_y, xlim=shared_xlim, ylim=shared_ylim,
                                       save_path=after_save, show=show)
    return {
        "slices": slices,
        "selected_color_map": selected_color_map,
        "swap_xy": swap_xy,
        "figures": {"stacked": fig1, "before": fig2, "after": fig3},
    }


if __name__ == "__main__":
    plot_mouse_xenium_axis_alignment_figures(
        root="/media/huifang/data/registration/mouse_brain",
        group_id=0,
        sample_ids=(0, 1, 2, 3, 4, 5),
        reference_id=4,
        obs_key="domain_id_shared",
        selected_domain_ids=(0, 1, 2, 5, 6, 7, 9, 10, 11),
        swap_xy=True,
        invert_y=True,
        style=MouseAlignmentStyle(
            cmap_name="tab20",
            point_size=0.5,
            overlap_alpha=0.30,
            aligned_alpha=0.30,
            stack_dx_fraction=0.20,
            stack_dy_fraction=0.1,
        ),
        save_dir="/media/huifang/data/registration/paper_analysis/Figure2_transition_evaluation/d/",
        show=True,
    )
