"""Three-panel overview of multi-slice Xenium mouse-brain alignment with AXIS.

Panels
------
1. Native measurements displayed as spatially offset/stacked slices.
2. Direct overlap of all native slice coordinates before AXIS.
3. Overlap of all AXIS-warped coordinates in a common reference frame.

Only selected spatial domains are assigned categorical colors. All remaining
spatial domains are shown in gray.

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
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D


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
    cmap_name: str = "tab10",
    domain_colors: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    """Create stable colors only for the selected domain IDs."""
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

    cmap = get_cmap(cmap_name, max(len(selected), 1))
    return {domain: cmap(i) for i, domain in enumerate(selected)}


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
        ax.invert_yaxis()


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


def plot_mouse_xenium_axis_alignment_overview(
    root: str = "/media/huifang/data/registration/xenium/mouse_brain",
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
    cmap_name: str = "tab10",
    gray_color: Any = "#C8C8C8",
    gray_legend_label: str = "Other domains",
    point_size: float = 2.2,
    stacked_point_size: Optional[float] = None,
    stacked_alpha: float = 0.95,
    stacked_gray_alpha: float = 0.42,
    overlap_alpha: float = 0.42,
    overlap_gray_alpha: float = 0.18,
    aligned_alpha: float = 0.42,
    aligned_gray_alpha: float = 0.18,
    stack_dx_fraction: float = 0.17,
    stack_dy_fraction: float = -0.13,
    stack_label_fontsize: float = 9.5,
    show_domain_legend: bool = True,
    show_other_in_legend: bool = True,
    legend_ncol: Optional[int] = None,
    legend_title: Optional[str] = "Selected domain annotation",
    legend_fontsize: float = 9.0,
    legend_title_fontsize: float = 10.0,
    invert_y: bool = True,
    share_overlap_limits: bool = True,
    figsize: Tuple[float, float] = (16.2, 5.4),
    panel_width_ratios: Tuple[float, float, float] = (1.2, 1.0, 1.0),
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 350,
):
    """Plot stacked, native-overlap, and AXIS-aligned Xenium mouse slices.

    Only ``selected_domain_ids`` are assigned categorical colors. Every other
    value in ``obs[obs_key]`` is rendered in ``gray_color``.

    Parameters
    ----------
    root
        Mouse-brain dataset root.
    h5ad_subdir
        Subdirectory containing ``{group_id}_{sample_id}.h5ad`` files. These
        files are used only for ``obs[obs_key]`` domain annotations.
    network_input_subdir
        Subdirectory containing ``{group_id}_{sample_id}_validation.npz``.
        Native coordinates are always loaded from the ``coord`` array here.
    sample_ids
        Slice identifiers ordered from the visually front slice to the back
        slice in the stacked panel.
    reference_id
        Slice defining the AXIS common coordinate frame.
    selected_domain_ids
        Domain IDs shown with distinct colors. IDs not in this list are gray.
    domain_colors
        Optional explicit mapping for selected domain IDs only, for example
        ``{0: '#...', 1: '#...', ...}``.
    share_overlap_limits
        Use identical axis limits in the before- and after-AXIS panels.

    Returns
    -------
    fig
        Matplotlib figure.
    slices
        Loaded :class:`MouseXeniumSlice` records.
    selected_color_map
        String-normalized selected-domain color mapping.
    """
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

    selected_color_map = _make_selected_domain_color_map(
        selected_domain_ids=selected_domain_ids,
        cmap_name=cmap_name,
        domain_colors=domain_colors,
    )

    stacked_point_size = (
        point_size if stacked_point_size is None else stacked_point_size
    )

    native_widths: List[float] = []
    native_heights: List[float] = []
    for slice_data in slices:
        valid = slice_data.observed_xy[_finite_rows(slice_data.observed_xy)]
        if valid.size:
            span = valid.max(axis=0) - valid.min(axis=0)
            native_widths.append(float(span[0]))
            native_heights.append(float(span[1]))

    if not native_widths or not native_heights:
        raise ValueError("No finite native coordinates were found.")

    stack_dx = float(np.median(native_widths) * stack_dx_fraction)
    stack_dy = float(np.median(native_heights) * stack_dy_fraction)

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=panel_width_ratios,
        left=0.025,
        right=0.985,
        bottom=0.08 if not show_domain_legend else 0.17,
        top=0.90,
        wspace=0.02,
    )
    ax_stack = fig.add_subplot(gs[0, 0])
    ax_before = fig.add_subplot(gs[0, 1])
    ax_after = fig.add_subplot(gs[0, 2])

    # ------------------------------------------------------------------
    # Panel 1: offset native measurements. This follows the same ordering
    # convention as the DLPFC figure: draw the back slice first so the first
    # slice in sample_ids remains visually in front.
    # ------------------------------------------------------------------
    stacked_coordinate_arrays: List[np.ndarray] = []
    for index in reversed(range(len(slices))):
        slice_data = slices[index]
        offset = np.array([index * stack_dx, index * stack_dy], dtype=float)
        shifted_xy = slice_data.observed_xy + offset
        stacked_coordinate_arrays.append(shifted_xy)

        _scatter_selected_domains(
            ax=ax_stack,
            coords=shifted_xy,
            labels=slice_data.labels,
            selected_color_map=selected_color_map,
            gray_color=gray_color,
            point_size=stacked_point_size,
            alpha=stacked_alpha,
            gray_alpha=stacked_gray_alpha,
            zorder=2 + (len(slices) - index),
        )

        finite = shifted_xy[_finite_rows(shifted_xy)]
        if finite.size:
            ax_stack.text(
                float(np.nanmin(finite[:, 0])),
                float(np.nanmax(finite[:, 1])),
                slice_data.display_name,
                ha="left",
                va="bottom",
                fontsize=stack_label_fontsize,
                fontweight="bold",
                color="black",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                    "pad": 1.5,
                },
                zorder=50,
            )

    stack_xlim, stack_ylim = _axis_limits(
        stacked_coordinate_arrays,
        pad_ratio=0.05,
    )
    ax_stack.set_xlim(*stack_xlim)
    ax_stack.set_ylim(*stack_ylim)
    _style_spatial_axis(
        ax_stack,
        f"{dataset_title}, {len(slices)} slices",
        invert_y=invert_y,
    )

    # ------------------------------------------------------------------
    # Panel 2: direct native-coordinate overlap.
    # ------------------------------------------------------------------
    for slice_data in slices:
        _scatter_selected_domains(
            ax=ax_before,
            coords=slice_data.observed_xy,
            labels=slice_data.labels,
            selected_color_map=selected_color_map,
            gray_color=gray_color,
            point_size=point_size,
            alpha=overlap_alpha,
            gray_alpha=overlap_gray_alpha,
        )
    _style_spatial_axis(ax_before, "Before AXIS", invert_y=invert_y)

    # ------------------------------------------------------------------
    # Panel 3: AXIS common-frame overlap.
    # ------------------------------------------------------------------
    for slice_data in slices:
        _scatter_selected_domains(
            ax=ax_after,
            coords=slice_data.aligned_xy,
            labels=slice_data.labels,
            selected_color_map=selected_color_map,
            gray_color=gray_color,
            point_size=point_size,
            alpha=aligned_alpha,
            gray_alpha=aligned_gray_alpha,
        )
    _style_spatial_axis(ax_after, "After AXIS", invert_y=invert_y)

    # ------------------------------------------------------------------
    # Axis limits.
    # ------------------------------------------------------------------
    if share_overlap_limits:
        common_xlim, common_ylim = _axis_limits(
            [
                *(slice_data.observed_xy for slice_data in slices),
                *(slice_data.aligned_xy for slice_data in slices),
            ],
            pad_ratio=0.035,
        )
        for ax in (ax_before, ax_after):
            ax.set_xlim(*common_xlim)
            if invert_y:
                ax.set_ylim(common_ylim[1], common_ylim[0])
            else:
                ax.set_ylim(*common_ylim)
    else:
        before_xlim, before_ylim = _axis_limits(
            [slice_data.observed_xy for slice_data in slices],
            pad_ratio=0.035,
        )
        after_xlim, after_ylim = _axis_limits(
            [slice_data.aligned_xy for slice_data in slices],
            pad_ratio=0.035,
        )
        ax_before.set_xlim(*before_xlim)
        ax_after.set_xlim(*after_xlim)
        if invert_y:
            ax_before.set_ylim(before_ylim[1], before_ylim[0])
            ax_after.set_ylim(after_ylim[1], after_ylim[0])
        else:
            ax_before.set_ylim(*before_ylim)
            ax_after.set_ylim(*after_ylim)

    # ------------------------------------------------------------------
    # Shared selected-domain legend.
    # ------------------------------------------------------------------
    if show_domain_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=6,
                label=f"Domain {domain}",
            )
            for domain, color in selected_color_map.items()
        ]

        if show_other_in_legend:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=gray_color,
                    markeredgecolor="none",
                    markersize=6,
                    label=gray_legend_label,
                )
            )

        if legend_ncol is None:
            legend_ncol = min(len(handles), 5)

        legend = fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=legend_ncol,
            frameon=False,
            fontsize=legend_fontsize,
            title=legend_title,
            columnspacing=1.05,
            handletextpad=0.35,
        )
        if legend_title is not None:
            legend.get_title().set_fontsize(legend_title_fontsize)
            legend.get_title().set_fontweight("bold")

    if save_path is not None:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"[OK] saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, slices, selected_color_map


if __name__ == "__main__":
    plot_mouse_xenium_axis_alignment_overview(
        root="/media/huifang/data/registration/xenium/mouse_brain",
        group_id=0,
        sample_ids=(0, 1, 2,3,4,5),
        reference_id=4,
        obs_key="domain_id_shared",
        h5ad_subdir="h5ads",
        selected_domain_ids=(0, 1, 2, 5, 6, 7, 9, 10, 11),
        point_size=1.,
        overlap_alpha=0.3,
        aligned_alpha=0.3,
        stack_dx_fraction=0.3,
        stack_dy_fraction=0.13,
        show_domain_legend=True,
        legend_title="Selected domain annotation",
        invert_y=False,
        share_overlap_limits=True,
        figsize=(16.2, 5.4),
        save_path=None,
        show=True,
    )
