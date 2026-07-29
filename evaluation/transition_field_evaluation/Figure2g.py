"""Three-panel SCC multi-slice alignment overview for AXIS.

Panels
------
1. Native SCC slices displayed with spatial offsets to show the number of slices.
2. Direct overlap of native slice coordinates before AXIS.
3. Overlap of AXIS-warped coordinates in one reference frame.

Expected inputs
---------------
root/
├── h5ad/
│   ├── {patient_id}_{sample_id}.h5ad
│   └── ...
└── pairwise_output/
    ├── group_{patient_id}/
    │   ├── {reference_id}_{sample_id}_warped_coor.npy
    │   └── ...
    └── ...

The code also checks for warped-coordinate files placed directly under
``root/pairwise_output``.
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
class SCCSlice:
    """Information needed to draw one SCC tissue slice."""

    sample_id: Any
    display_name: str
    observed_xy: np.ndarray
    aligned_xy: np.ndarray
    labels: np.ndarray


def _h5ad_path(
    root: str,
    patient_id: Any,
    sample_id: Any,
    h5ad_subdir: str,
    h5ad_template: str,
) -> str:
    return os.path.join(
        root,
        h5ad_subdir,
        h5ad_template.format(
            patient_id=patient_id,
            group_id=patient_id,
            sample_id=sample_id,
        ),
    )


def _candidate_axis_dirs(
    root: str,
    patient_id: Any,
    pairwise_subdir: str,
) -> List[str]:
    """Return possible SCC AXIS output directories, most specific first."""
    base = os.path.join(root, pairwise_subdir)
    return [
        os.path.join(base, f"group_{patient_id}"),
        os.path.join(base, str(patient_id)),
        base,
    ]


def _load_np_array(path: str, preferred_keys: Sequence[str]) -> np.ndarray:
    """Load a coordinate array from either NPY or NPZ output."""
    if path.endswith(".npy"):
        return np.asarray(np.load(path), dtype=np.float32)

    if not path.endswith(".npz"):
        raise ValueError(f"Unsupported AXIS coordinate format: {path}")

    with np.load(path) as data:
        for key in preferred_keys:
            if key in data:
                return np.asarray(data[key], dtype=np.float32)

        keys = list(data.files)
        if len(keys) == 1:
            return np.asarray(data[keys[0]], dtype=np.float32)

        raise KeyError(
            f"Could not identify a warped-coordinate array in {path}. "
            f"Available keys: {keys}"
        )


def _load_observed_slice(
    root: str,
    patient_id: Any,
    sample_id: Any,
    obs_key: str,
    coord_key: str,
    h5ad_subdir: str,
    h5ad_template: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load native SCC coordinates and cluster labels from one h5ad file."""
    path = _h5ad_path(
        root=root,
        patient_id=patient_id,
        sample_id=sample_id,
        h5ad_subdir=h5ad_subdir,
        h5ad_template=h5ad_template,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing SCC h5ad file: {path}")

    adata = sc.read_h5ad(path)

    if coord_key not in adata.obsm:
        raise KeyError(
            f"obsm['{coord_key}'] is missing from {path}. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    if obs_key not in adata.obs.columns:
        raise KeyError(
            f"obs['{obs_key}'] is missing from {path}. "
            f"Available columns include: {list(adata.obs.columns[:30])}"
        )

    coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"obsm['{coord_key}'] in {path} must have shape (N, 2+); "
            f"received {coords.shape}."
        )

    labels = (
        adata.obs[obs_key]
        .astype("string")
        .fillna("NA")
        .astype(str)
        .to_numpy()
    )

    if coords.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Coordinate/label mismatch for patient {patient_id}, "
            f"sample {sample_id}: {coords.shape[0]} coordinates versus "
            f"{labels.shape[0]} labels."
        )

    return coords[:, :2], labels


def _load_axis_aligned_coordinates(
    root: str,
    patient_id: Any,
    sample_id: Any,
    reference_id: Any,
    observed_xy: np.ndarray,
    pairwise_subdir: str,
    warped_coord_template: str,
) -> np.ndarray:
    """Load one SCC slice in the AXIS reference frame."""
    if sample_id == reference_id:
        return np.asarray(observed_xy, dtype=np.float32).copy()

    filenames = [
        warped_coord_template.format(
            patient_id=patient_id,
            group_id=patient_id,
            reference_id=reference_id,
            fixed_id=reference_id,
            sample_id=sample_id,
            moving_id=sample_id,
        )
    ]

    # Also tolerate common NPZ naming variants.
    stem = os.path.splitext(filenames[0])[0]
    for suffix in (".npz", ".npy"):
        candidate = stem + suffix
        if candidate not in filenames:
            filenames.append(candidate)

    attempted: List[str] = []
    for directory in _candidate_axis_dirs(root, patient_id, pairwise_subdir):
        for filename in filenames:
            path = os.path.join(directory, filename)
            attempted.append(path)
            if not os.path.exists(path):
                continue

            coords = _load_np_array(
                path,
                preferred_keys=(
                    "warped_coor",
                    "warped_coord",
                    "aligned_coord",
                    "coord",
                    "coords",
                    "pts2",
                    "arr_0",
                ),
            )
            if coords.ndim != 2 or coords.shape[1] < 2:
                raise ValueError(
                    f"AXIS coordinates in {path} must have shape (N, 2+); "
                    f"received {coords.shape}."
                )
            return coords[:, :2]

    raise FileNotFoundError(
        "Could not find the AXIS warped coordinates for "
        f"patient={patient_id}, reference={reference_id}, "
        f"sample={sample_id}. Tried:\n  " + "\n  ".join(attempted)
    )


def _natural_key(value: Any) -> List[Any]:
    text = str(value)
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", text)
    ]


def _shared_category_order(
    label_arrays: Sequence[np.ndarray],
    category_order: Optional[Sequence[Any]],
) -> List[str]:
    observed: List[str] = []
    seen = set()
    for labels in label_arrays:
        for label in np.asarray(labels).astype(str):
            if label not in seen:
                seen.add(label)
                observed.append(label)

    if category_order is None:
        return sorted(observed, key=_natural_key)

    ordered = [str(x) for x in category_order]
    ordered.extend(
        sorted(set(observed) - set(ordered), key=_natural_key)
    )
    return ordered


def _make_cluster_color_map(
    categories: Sequence[str],
    cmap_name: str,
    cluster_colors: Optional[Mapping[Any, Any]],
) -> Dict[str, Any]:
    if cluster_colors is not None:
        normalized = {str(key): value for key, value in cluster_colors.items()}
        missing = [category for category in categories if category not in normalized]
        if missing:
            raise KeyError(
                "cluster_colors is missing entries for: " + ", ".join(missing)
            )
        return {category: normalized[category] for category in categories}

    cmap = get_cmap(cmap_name, max(len(categories), 1))
    return {category: cmap(index) for index, category in enumerate(categories)}


def _finite_rows(coords: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(coords), axis=1)


def _scatter_clusters(
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

    if coords.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Coordinate/label length mismatch: {coords.shape[0]} versus "
            f"{labels.shape[0]}."
        )

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
        linewidths=0,
        edgecolors="none",
        rasterized=True,
        zorder=zorder,
    )


def _axis_limits(
    coordinate_arrays: Iterable[np.ndarray],
    pad_ratio: float,
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


def _style_spatial_axis(ax, title: str, invert_y: bool):
    ax.set_title(title, fontsize=17, fontweight="bold", pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if invert_y:
        ax.invert_yaxis()


def _prepare_scc_slices(
    root: str,
    patient_id: Any,
    sample_ids: Sequence[Any],
    reference_id: Any,
    obs_key: str,
    coord_key: str,
    h5ad_subdir: str,
    h5ad_template: str,
    pairwise_subdir: str,
    warped_coord_template: str,
    slice_names: Optional[Mapping[Any, str]],
) -> List[SCCSlice]:
    sample_ids = list(sample_ids)
    if not sample_ids:
        raise ValueError("sample_ids cannot be empty.")
    if reference_id not in sample_ids:
        raise ValueError(
            f"reference_id={reference_id} must be included in "
            f"sample_ids={sample_ids}."
        )

    slices: List[SCCSlice] = []
    for position, sample_id in enumerate(sample_ids, start=1):
        observed_xy, labels = _load_observed_slice(
            root=root,
            patient_id=patient_id,
            sample_id=sample_id,
            obs_key=obs_key,
            coord_key=coord_key,
            h5ad_subdir=h5ad_subdir,
            h5ad_template=h5ad_template,
        )
        aligned_xy = _load_axis_aligned_coordinates(
            root=root,
            patient_id=patient_id,
            sample_id=sample_id,
            reference_id=reference_id,
            observed_xy=observed_xy,
            pairwise_subdir=pairwise_subdir,
            warped_coord_template=warped_coord_template,
        )

        if aligned_xy.shape[0] != labels.shape[0]:
            raise ValueError(
                f"AXIS coordinate/label mismatch for patient {patient_id}, "
                f"sample {sample_id}: {aligned_xy.shape[0]} aligned "
                f"coordinates versus {labels.shape[0]} labels."
            )

        default_name = f"Slice {position}"
        display_name = (
            slice_names.get(sample_id, default_name)
            if slice_names is not None
            else default_name
        )

        slices.append(
            SCCSlice(
                sample_id=sample_id,
                display_name=display_name,
                observed_xy=observed_xy,
                aligned_xy=aligned_xy,
                labels=labels,
            )
        )

    return slices


def plot_scc_axis_alignment_overview(
    patient_id: Any,
    sample_ids: Sequence[Any],
    reference_id: Any,
    root: str = "/media/huifang/data/registration/SCC/huifang",
    h5ad_subdir: str = "h5ad",
    h5ad_template: str = "{patient_id}_{sample_id}.h5ad",
    obs_key: str = "original_clusters",
    coord_key: str = "spatial_image_coor",
    pairwise_subdir: str = "pairwise_output",
    warped_coord_template: str = "{reference_id}_{sample_id}_warped_coor.npy",
    dataset_title: str = "SCC",
    slice_names: Optional[Mapping[Any, str]] = None,
    category_order: Optional[Sequence[Any]] = None,
    cluster_colors: Optional[Mapping[Any, Any]] = None,
    cmap_name: str = "tab20",
    point_size: float = 30.0,
    stacked_point_size: Optional[float] = None,
    stacked_alpha: float = 0.85,
    overlap_alpha: float = 0.50,
    aligned_alpha: float = 0.50,
    stack_dx_fraction: float = 0.17,
    stack_dy_fraction: float = -0.13,
    stack_label_fontsize: float = 9.5,
    show_cluster_legend: bool = True,
    legend_title: Optional[str] = "Original clusters",
    legend_ncol: Optional[int] = None,
    legend_fontsize: float = 8.5,
    legend_title_fontsize: float = 9.5,
    legend_marker_size: float = 6.0,
    legend_position: str = "right",
    invert_y: bool = True,
    share_overlap_limits: bool = True,
    figsize: Tuple[float, float] = (15.5, 5.2),
    panel_width_ratios: Tuple[float, float, float] = (1.05, 1.0, 1.0),
    legend_width_ratio: float = 0.38,
    panel_wspace: float = 0.04,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 350,
):
    """Plot stacked, native-overlap, and AXIS-aligned SCC slices.

    Parameters
    ----------
    patient_id
        SCC patient identifier used in h5ad names and AXIS group folders.
    sample_ids
        Slice identifiers ordered from the visually front slice to the back
        slice in the stacked panel.
    reference_id
        Slice defining the AXIS common coordinate frame.
    obs_key
        AnnData ``obs`` column containing the original SCC cluster labels.
    coord_key
        AnnData ``obsm`` key containing image-space coordinates.
    legend_position
        ``"right"`` (default) reserves a dedicated fourth GridSpec column
        for the cluster legend. ``"bottom"`` places it below the panels.

    Returns
    -------
    fig, slices, color_map
    """
    slices = _prepare_scc_slices(
        root=root,
        patient_id=patient_id,
        sample_ids=sample_ids,
        reference_id=reference_id,
        obs_key=obs_key,
        coord_key=coord_key,
        h5ad_subdir=h5ad_subdir,
        h5ad_template=h5ad_template,
        pairwise_subdir=pairwise_subdir,
        warped_coord_template=warped_coord_template,
        slice_names=slice_names,
    )

    categories = _shared_category_order(
        [item.labels for item in slices],
        category_order=category_order,
    )
    color_map = _make_cluster_color_map(
        categories=categories,
        cmap_name=cmap_name,
        cluster_colors=cluster_colors,
    )

    stacked_point_size = (
        point_size if stacked_point_size is None else stacked_point_size
    )

    widths: List[float] = []
    heights: List[float] = []
    for item in slices:
        finite = item.observed_xy[_finite_rows(item.observed_xy)]
        if finite.size:
            span = finite.max(axis=0) - finite.min(axis=0)
            widths.append(float(span[0]))
            heights.append(float(span[1]))
    if not widths or not heights:
        raise ValueError("No finite native SCC coordinates were found.")

    stack_dx = float(np.median(widths) * stack_dx_fraction)
    stack_dy = float(np.median(heights) * stack_dy_fraction)

    use_right_legend = show_cluster_legend and legend_position.lower() == "right"

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    if use_right_legend:
        gs = fig.add_gridspec(
            1,
            4,
            width_ratios=(*panel_width_ratios, legend_width_ratio),
            left=0.02,
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
            left=0.025,
            right=0.985,
            bottom=0.08 if not show_cluster_legend else 0.17,
            top=0.90,
            wspace=panel_wspace,
        )
        ax_stack = fig.add_subplot(gs[0, 0])
        ax_before = fig.add_subplot(gs[0, 1])
        ax_after = fig.add_subplot(gs[0, 2])
        ax_legend = None

    # Panel 1: native slices, with sample_ids[0] visually in front.
    stacked_arrays: List[np.ndarray] = []
    for index in reversed(range(len(slices))):
        item = slices[index]
        shifted = item.observed_xy + np.asarray(
            [index * stack_dx, index * stack_dy], dtype=float
        )
        stacked_arrays.append(shifted)

        _scatter_clusters(
            ax=ax_stack,
            coords=shifted,
            labels=item.labels,
            color_map=color_map,
            point_size=stacked_point_size,
            alpha=stacked_alpha,
            zorder=2 + (len(slices) - index),
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
    _style_spatial_axis(
        ax_stack,
        f"{dataset_title} patient {patient_id}, {len(slices)} slices",
        invert_y=invert_y,
    )

    # Panel 2: direct overlap before registration.
    for item in slices:
        _scatter_clusters(
            ax=ax_before,
            coords=item.observed_xy,
            labels=item.labels,
            color_map=color_map,
            point_size=point_size,
            alpha=overlap_alpha,
            zorder=2,
        )
    _style_spatial_axis(ax_before, "Before AXIS", invert_y=invert_y)

    # Panel 3: all slices in the AXIS reference frame.
    for item in slices:
        _scatter_clusters(
            ax=ax_after,
            coords=item.aligned_xy,
            labels=item.labels,
            color_map=color_map,
            point_size=point_size,
            alpha=aligned_alpha,
            zorder=2,
        )
    _style_spatial_axis(ax_after, "After AXIS", invert_y=invert_y)

    if share_overlap_limits:
        common_xlim, common_ylim = _axis_limits(
            [item.observed_xy for item in slices]
            + [item.aligned_xy for item in slices],
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
            [item.observed_xy for item in slices], pad_ratio=0.035
        )
        after_xlim, after_ylim = _axis_limits(
            [item.aligned_xy for item in slices], pad_ratio=0.035
        )
        ax_before.set_xlim(*before_xlim)
        ax_after.set_xlim(*after_xlim)
        if invert_y:
            ax_before.set_ylim(before_ylim[1], before_ylim[0])
            ax_after.set_ylim(after_ylim[1], after_ylim[0])
        else:
            ax_before.set_ylim(*before_ylim)
            ax_after.set_ylim(*after_ylim)

    if show_cluster_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color_map[category],
                markeredgecolor="none",
                markersize=legend_marker_size,
                label=category,
            )
            for category in categories
        ]

        if use_right_legend:
            # A single vertical column is the safest default for a legend
            # occupying a dedicated axis to the right of the whole figure.
            right_ncol = 1 if legend_ncol is None else int(legend_ncol)
            legend = ax_legend.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(0.0, 0.5),
                ncol=right_ncol,
                frameon=False,
                title=legend_title,
                fontsize=legend_fontsize,
                handletextpad=0.35,
                columnspacing=0.8,
                labelspacing=0.25,
                borderaxespad=0.0,
            )
        else:
            bottom_ncol = (
                min(max(len(handles), 1), 8)
                if legend_ncol is None
                else int(legend_ncol)
            )
            legend = fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.005),
                ncol=bottom_ncol,
                frameon=False,
                title=legend_title,
                fontsize=legend_fontsize,
                handletextpad=0.35,
                columnspacing=1.0,
            )

        if legend.get_title() is not None:
            legend.get_title().set_fontsize(legend_title_fontsize)
            legend.get_title().set_fontweight("bold")

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
    # Example: replace sample IDs with the SCC slices available for one patient.
    plot_scc_axis_alignment_overview(
        patient_id=2,
        sample_ids=(0, 1, 2),
        reference_id=0,
        save_path=None,
        show=True,
    )
