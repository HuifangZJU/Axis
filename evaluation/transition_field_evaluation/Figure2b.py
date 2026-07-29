"""Three-panel overview of multi-slice DLPFC alignment with AXIS.

Panels
------
1. Native measurements displayed as spatially offset/stacked slices.
2. Direct overlap of all native slice coordinates before AXIS.
3. Overlap of all AXIS-warped coordinates in a common reference frame.

Expected file organization
--------------------------
root/
├── h5ad/
│   ├── {group_id}_{sample_id}.h5ad
│   └── ...
└── pairwise_output/
    └── group_{group_id}/
        ├── {reference_id}_{sample_id}_warped_coor.npy
        └── ...

The reference slice uses its native coordinates as its common-frame
coordinates. Every other slice is loaded from the corresponding AXIS
``warped_coor.npy`` output.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D


@dataclass
class DLPFCSlice:
    """Data needed to draw one DLPFC slice."""

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
    """Locate the AXIS output folder for one group."""
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
    return os.path.join(
        root,
        "network_input",
        f"{group_id}_{sample_id}_validation.npz",
    )


def _load_observed_slice(
    root: str,
    group_id: Any,
    sample_id: Any,
    obs_key: str,
    coord_key: str = "spatial",
):
    """Load AnnData, native coordinates, and domain labels for one slice."""
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
                f"obsm['{coord_key}'] is absent and the fallback coordinate "
                f"file does not exist: {coord_path}"
            )
        with np.load(coord_path) as npz:
            if "coord" not in npz:
                raise KeyError(f"'coord' not found in {coord_path}")
            observed_xy = np.asarray(npz["coord"], dtype=np.float32)

    if observed_xy.ndim != 2 or observed_xy.shape[1] < 2:
        raise ValueError(
            f"Coordinates for sample {sample_id} must have shape (N, 2+); "
            f"received {observed_xy.shape}."
        )
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

    n = min(adata.n_obs, observed_xy.shape[0], labels.shape[0])
    return adata[:n].copy(), observed_xy[:n, :2], labels[:n]


def _load_axis_aligned_coordinates(
    root: str,
    group_id: Any,
    sample_id: Any,
    reference_id: Any,
    observed_xy: np.ndarray,
    pairwise_subdir: str = "pairwise_output",
    warped_coord_template: str = "{reference_id}_{sample_id}_warped_coor.npy",
) -> np.ndarray:
    """Load one slice's AXIS coordinate output in the reference frame."""
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


def _natural_key(value: Any) -> List[Any]:
    """Natural ordering: Layer2 precedes Layer10."""
    text = str(value)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def _shared_category_order(
    label_arrays: Sequence[np.ndarray],
    category_order: Optional[Sequence[Any]] = None,
) -> List[str]:
    if category_order is not None:
        return [str(x) for x in category_order]

    observed = []
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
    cmap_name: str = "tab10",
    domain_colors: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    """Create one stable domain color dictionary shared by all panels."""
    if domain_colors is not None:
        missing = [c for c in categories if c not in domain_colors]
        if missing:
            raise KeyError(
                "domain_colors does not define colors for: "
                + ", ".join(map(str, missing))
            )
        return {str(c): domain_colors[c] for c in categories}

    cmap = get_cmap(cmap_name, max(len(categories), 1))
    return {category: cmap(i) for i, category in enumerate(categories)}


def _finite_rows(coords: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(coords), axis=1)


def _scatter_domains(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    color_map: Mapping[str, Any],
    point_size: float,
    alpha: float,
    marker: str = "o",
    zorder: int = 2,
):
    """Draw categorical domains with explicit colors and no marker borders."""
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
        marker=marker,
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


def _style_spatial_axis(ax, title: str, invert_y: bool = True):
    ax.set_title(title, fontsize=17, pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if invert_y:
        ax.invert_yaxis()


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
        raise ValueError(
            f"reference_id={reference_id} must be included in sample_ids={sample_ids}."
        )

    slices: List[DLPFCSlice] = []
    for position, sample_id in enumerate(sample_ids, start=1):
        _, observed_xy, labels = _load_observed_slice(
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

        default_name = f"Slice {position}"
        display_name = (
            slice_names.get(sample_id, default_name)
            if slice_names is not None
            else default_name
        )

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


def plot_dlpfc_axis_alignment_overview(
    root: str = "/media/huifang/data/registration/DLPFC/huifang",
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
    cmap_name: str = "tab10",
    point_size: float = 3.2,
    stacked_point_size: Optional[float] = None,
    stacked_alpha: float = 0.95,
    overlap_alpha: float = 0.38,
    aligned_alpha: float = 0.38,
    stack_dx_fraction: float = 0.17,
    stack_dy_fraction: float = -0.13,
    stack_label_fontsize: float = 9.5,
    show_domain_legend: bool = True,
    legend_ncol: Optional[int] = None,
    legend_title: Optional[str] = "Layer annotation",
    legend_fontsize: float = 14.0,
    legend_title_fontsize: float = 10.0,
    invert_y: bool = True,
    share_overlap_limits: bool = True,
    figsize: Tuple[float, float] = (15.5, 5.2),
    panel_width_ratios: Tuple[float, float, float] = (1.05, 1.0, 1.0),
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 350,
):
    """Plot stacked, unregistered, and AXIS-aligned DLPFC slices.

    Parameters
    ----------
    root
        Dataset root containing ``h5ad`` and AXIS output folders.
    group_id
        DLPFC group identifier used in input and output file names.
    sample_ids
        Slice IDs, ordered from the front slice to the back slice in panel 1.
    reference_id
        Slice defining the AXIS common coordinate frame.
    obs_key
        AnnData ``obs`` column containing spatial domain/layer labels.
    slice_names
        Optional mapping such as ``{1: "10 µm", 2: "110 µm", ...}``.
        When omitted, panel 1 uses ``Slice 1``, ``Slice 2``, etc.
    stack_dx_fraction, stack_dy_fraction
        Offsets between consecutive stacked slices, expressed as fractions of
        the median native tissue width and height.
    share_overlap_limits
        Use identical axis limits for panels 2 and 3 for a direct comparison.

    Returns
    -------
    fig
        Matplotlib figure.
    slices
        List of loaded :class:`DLPFCSlice` objects.
    color_map
        Domain-to-color mapping used across all panels.
    """
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
        cmap_name=cmap_name,
        domain_colors=domain_colors,
    )

    stacked_point_size = (
        point_size if stacked_point_size is None else stacked_point_size
    )

    native_widths = []
    native_heights = []
    for slice_data in slices:
        valid = slice_data.observed_xy[_finite_rows(slice_data.observed_xy)]
        if valid.size:
            span = valid.max(axis=0) - valid.min(axis=0)
            native_widths.append(span[0])
            native_heights.append(span[1])

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
        bottom=0.08 if not show_domain_legend else 0.16,
        top=0.90,
        wspace=0.10,
    )
    ax_stack = fig.add_subplot(gs[0, 0])
    ax_before = fig.add_subplot(gs[0, 1])
    ax_after = fig.add_subplot(gs[0, 2])

    # ------------------------------------------------------------------
    # Panel 1: offset native measurements. Plot the back slice first so
    # the first slice in sample_ids remains visually in front.
    # ------------------------------------------------------------------
    stacked_coordinate_arrays = []
    for index in reversed(range(len(slices))):
        slice_data = slices[index]
        offset = np.array([index * stack_dx, index * stack_dy], dtype=float)
        shifted_xy = slice_data.observed_xy + offset
        stacked_coordinate_arrays.append(shifted_xy)

        _scatter_domains(
            ax_stack,
            shifted_xy,
            slice_data.labels,
            color_map,
            point_size=stacked_point_size,
            alpha=stacked_alpha,
            zorder=2 + (len(slices) - index),
        )

        finite = shifted_xy[_finite_rows(shifted_xy)]
        if finite.size:
            x_label = float(np.nanmin(finite[:, 0]))
            y_label = float(np.nanmin(finite[:, 1]))
            ax_stack.text(
                x_label,
                y_label,
                slice_data.display_name,
                ha="left",
                va="bottom",
                fontsize=stack_label_fontsize,
                fontweight="bold",
                color="black",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 1.5,
                },
                zorder=20,
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
    # Panel 2: native coordinates directly overlaid.
    # ------------------------------------------------------------------
    for slice_data in slices:
        _scatter_domains(
            ax_before,
            slice_data.observed_xy,
            slice_data.labels,
            color_map,
            point_size=point_size,
            alpha=overlap_alpha,
        )
    _style_spatial_axis(ax_before, "Before AXIS", invert_y=invert_y)

    # ------------------------------------------------------------------
    # Panel 3: AXIS common-frame coordinates directly overlaid.
    # ------------------------------------------------------------------
    for slice_data in slices:
        _scatter_domains(
            ax_after,
            slice_data.aligned_xy,
            slice_data.labels,
            color_map,
            point_size=point_size,
            alpha=aligned_alpha,
        )
    _style_spatial_axis(ax_after, "After AXIS", invert_y=invert_y)

    if share_overlap_limits:
        common_xlim, common_ylim = _axis_limits(
            [
                *(slice_data.observed_xy for slice_data in slices),
                *(slice_data.aligned_xy for slice_data in slices),
            ],
            pad_ratio=0.035,
        )
        for ax in (ax_before, ax_after):
            # Set limits after styling because invert_yaxis changes direction.
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

    if show_domain_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color_map[category],
                markeredgecolor="none",
                markersize=6,
                label=category,
            )
            for category in categories
        ]
        if legend_ncol is None:
            legend_ncol = min(len(categories), 7)
        legend = fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=legend_ncol,
            frameon=False,
            fontsize=legend_fontsize,
            title=legend_title,
            columnspacing=1.1,
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

    return fig, slices, color_map


if __name__ == "__main__":
    plot_dlpfc_axis_alignment_overview(
        root="/media/huifang/data/registration/DLPFC/huifang",
        group_id=1,
        sample_ids=(0, 1, 2, 3),
        reference_id=1,
        obs_key="layer_guess_reordered",
        coord_key="spatial",
        # Optional explicit names, for example:
        # slice_names={1: "Slice 1", 2: "Slice 2", 3: "Slice 3", 4: "Slice 4"},
        cmap_name="tab20",
        point_size=10,
        overlap_alpha=0.58,
        aligned_alpha=0.58,
        stack_dx_fraction=0.17,
        stack_dy_fraction=-0.13,
        show_domain_legend=True,
        legend_title="Layer annotation",
        invert_y=True,
        share_overlap_limits=True,
        figsize=(15.5, 5.2),
        save_path=None,
        show=True,
    )
