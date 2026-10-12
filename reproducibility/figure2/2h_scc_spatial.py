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
from axis.settings import data_path
from axis.config import load_config

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.lines import Line2D


def set_publication_style() -> None:
    """Match the SCC spatial-map styling used in the cluster-fidelity script."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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
    """Stable categorical mapping matching the SCC cluster-fidelity script.

    Numeric cluster IDs are mapped directly to the corresponding discrete
    colormap index, so a cluster keeps the same color even when a subset of
    clusters is present. Non-numeric labels fall back to category order.
    """
    if cluster_colors is not None:
        normalized = {str(key): value for key, value in cluster_colors.items()}
        missing = [category for category in categories if category not in normalized]
        if missing:
            raise KeyError(
                "cluster_colors is missing entries for: " + ", ".join(missing)
            )
        return {category: normalized[category] for category in categories}

    cmap = plt.get_cmap(cmap_name)
    color_map: Dict[str, Any] = {}
    for fallback_index, category in enumerate(categories):
        category = str(category)
        try:
            color_index = int(float(category)) % cmap.N
        except Exception:
            color_index = fallback_index % cmap.N
        color_map[category] = cmap(color_index)
    return color_map


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
    """Draw clusters with the same per-cluster rendering used in script 2."""
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

    coords = coords[valid, :2]
    labels = labels[valid]

    # Draw one cluster at a time, in the stable color-map order.
    for category in color_map:
        keep = labels == str(category)
        if not np.any(keep):
            continue
        ax.scatter(
            coords[keep, 0],
            coords[keep, 1],
            s=point_size,
            color=[color_map[str(category)]],
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
    # ax.set_title(title, fontsize=15, fontweight="normal", pad=6)
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




from dataclasses import field


@dataclass
class FigureLayout:
    # Match the standalone SCC spatial-map geometry in script 2.
    figsize: Tuple[float, float] = (11.2, 3.8)
    left: float = 0.02
    right: float = 0.84
    bottom: float = 0.04
    top: float = 0.88
    legend_anchor: Tuple[float, float] = (0.98, 0.50)
    dpi: int = 400


@dataclass
class SCCAlignmentStyle:
    title_fontsize: float = 15.0
    slice_label_fontsize: float = 15.0
    point_size: float = 50.0
    stacked_point_size: Optional[float] = None

    # Match script 2 categorical spatial-map opacity.
    stacked_alpha: float = 0.85
    overlap_alpha: float = 0.70
    aligned_alpha: float = 0.70

    stack_dx_fraction: float = -0.17
    stack_dy_fraction: float = 0.13
    stack_pad_ratio: float = 0.05
    overlap_pad_ratio: float = 0.03

    # Script 2 uses tab20, with cluster ID -> colormap index mapping.
    cmap_name: str = "tab20"
    show_cluster_legend: bool = True
    legend_title: str = "Cluster"
    legend_fontsize: float = 13.0
    legend_title_fontsize: float = 14
    legend_markersize: float = 6.0

    # Keep the stacked overview somewhat taller, but match the standalone
    # before/after panels to script 2.
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


def _attach_scc_legend(ax, categories, color_map, style: SCCAlignmentStyle, layout: FigureLayout):
    if not style.show_cluster_legend:
        return None
    handles = [
        Line2D([0],[0], marker="o", linestyle="none", markerfacecolor=color_map[c], markeredgecolor="none",
               markersize=style.legend_markersize, label=str(c))
        for c in categories
    ]
    legend = ax.legend(handles=handles, loc="center left", bbox_to_anchor=layout.legend_anchor,
                       frameon=False, title=style.legend_title, fontsize=style.legend_fontsize,
                       borderaxespad=0.0, handletextpad=0.35, labelspacing=0.35)
    legend.get_title().set_fontsize(style.legend_title_fontsize)
    legend.get_title().set_fontweight("bold")
    return legend


def plot_scc_stacked_native(slices, categories, color_map, dataset_title: str, patient_id: Any,
                            style: SCCAlignmentStyle, invert_y=True, save_path=None, show=True):
    layout = style.stacked_layout
    fig, ax = _new_figure(layout)
    stacked_point_size = style.point_size if style.stacked_point_size is None else style.stacked_point_size
    widths, heights = [], []
    for item in slices:
        finite = item.observed_xy[_finite_rows(item.observed_xy)]
        if finite.size:
            span = finite.max(axis=0) - finite.min(axis=0)
            widths.append(float(span[0])); heights.append(float(span[1]))
    if not widths or not heights:
        raise ValueError("No finite native SCC coordinates were found.")
    stack_dx = float(np.median(widths) * style.stack_dx_fraction)
    stack_dy = float(np.median(heights) * style.stack_dy_fraction)
    stacked_arrays = []
    for index in reversed(range(len(slices))):
        item = slices[index]
        shifted = item.observed_xy + np.asarray([index * stack_dx, index * stack_dy], dtype=float)
        stacked_arrays.append(shifted)
        _scatter_clusters(ax=ax, coords=shifted, labels=item.labels, color_map=color_map,
                          point_size=stacked_point_size, alpha=style.stacked_alpha,
                          zorder=2 + (len(slices) - index))
        finite = shifted[_finite_rows(shifted)]
        if finite.size:
            ax.text(float(np.min(finite[:,0])), float(np.min(finite[:,1])), item.display_name,
                    ha='left', va='bottom', fontsize=style.slice_label_fontsize, fontweight='bold',
                    bbox={'facecolor':'white','edgecolor':'none','alpha':0.84,'pad':1.4}, zorder=50)
    xlim, ylim = _axis_limits(stacked_arrays, pad_ratio=style.stack_pad_ratio)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    _style_spatial_axis(ax, f"{dataset_title} patient {patient_id}, {len(slices)} slices", invert_y=invert_y)
    # _attach_scc_legend(ax, categories, color_map, style, layout)
    _save_figure(fig, save_path, layout.dpi)
    if show: plt.show()
    else: plt.close(fig)
    return fig, ax


def plot_scc_overlap_before(slices, categories, color_map, style: SCCAlignmentStyle, invert_y=True, xlim=None, ylim=None, save_path=None, show=True):
    layout = style.before_layout
    fig, ax = _new_figure(layout)
    for item in slices:
        _scatter_clusters(ax=ax, coords=item.observed_xy, labels=item.labels, color_map=color_map,
                          point_size=style.point_size, alpha=style.overlap_alpha, zorder=2)
    _style_spatial_axis(ax, "Before AXIS", invert_y=invert_y)
    if xlim is None or ylim is None:
        xlim, ylim = _axis_limits([item.observed_xy for item in slices], pad_ratio=style.overlap_pad_ratio)
    ax.set_xlim(*xlim)
    if invert_y: ax.set_ylim(ylim[1], ylim[0])
    else: ax.set_ylim(*ylim)
    # _attach_scc_legend(ax, categories, color_map, style, layout)
    _save_figure(fig, save_path, layout.dpi)
    if show: plt.show()
    else: plt.close(fig)
    return fig, ax


def plot_scc_overlap_after(slices, categories, color_map, style: SCCAlignmentStyle, invert_y=True, xlim=None, ylim=None, save_path=None, show=True):
    layout = style.after_layout
    fig, ax = _new_figure(layout)
    for item in slices:
        _scatter_clusters(ax=ax, coords=item.aligned_xy, labels=item.labels, color_map=color_map,
                          point_size=style.point_size, alpha=style.aligned_alpha, zorder=2)
    _style_spatial_axis(ax, "After AXIS", invert_y=invert_y)
    if xlim is None or ylim is None:
        xlim, ylim = _axis_limits([item.aligned_xy for item in slices], pad_ratio=style.overlap_pad_ratio)
    ax.set_xlim(*xlim)
    if invert_y: ax.set_ylim(ylim[1], ylim[0])
    else: ax.set_ylim(*ylim)
    _attach_scc_legend(ax, categories, color_map, style, layout)
    _save_figure(fig, save_path, layout.dpi)
    if show: plt.show()
    else: plt.close(fig)
    return fig, ax


def plot_scc_axis_alignment_figures(
    patient_id: Any = 10,
    sample_ids: Sequence[Any] = (0, 1, 2),
    reference_id: Any = 0,
    root: str = load_config('scc').root,
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
    invert_y: bool = True,
    share_overlap_limits: bool = True,
    style: Optional[SCCAlignmentStyle] = None,
    save_dir: Optional[str] = data_path("paper_analysis/Figure2_transition_evaluation/h/"),
    file_prefix: str = "scc_axis_alignment",
    show: bool = True,
):
    set_publication_style()
    style = SCCAlignmentStyle() if style is None else style
    slices = _prepare_scc_slices(root=root, patient_id=patient_id, sample_ids=sample_ids, reference_id=reference_id,
                                 obs_key=obs_key, coord_key=coord_key, h5ad_subdir=h5ad_subdir,
                                 h5ad_template=h5ad_template, pairwise_subdir=pairwise_subdir,
                                 warped_coord_template=warped_coord_template, slice_names=slice_names)
    categories = _shared_category_order([item.labels for item in slices], category_order=category_order)
    color_map = _make_cluster_color_map(categories=categories, cmap_name=style.cmap_name, cluster_colors=cluster_colors)
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
    fig1,_ = plot_scc_stacked_native(slices, categories, color_map, dataset_title, patient_id, style, invert_y, stacked_save, show)
    # fig2,_ = plot_scc_overlap_before(slices, categories, color_map, style, invert_y, shared_xlim, shared_ylim, before_save, show)
    fig3,_ = plot_scc_overlap_after(slices, categories, color_map, style, invert_y, shared_xlim, shared_ylim, after_save, show)
    # return {"slices": slices, "categories": categories, "color_map": color_map, "figures": {"stacked": fig1, "before": fig2, "after": fig3}}


if __name__ == "__main__":
    plot_scc_axis_alignment_figures(show=True)