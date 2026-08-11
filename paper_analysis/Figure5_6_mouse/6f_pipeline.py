#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot disease-delta mirrored grids for mouse-brain analysis.

All disease-minus-normal deltas are computed on the fly with
spatial_delta_utils.compute_spatial_delta_from_files() after the shared
library-size-normalization -> optional-log1p preprocessing.

Figure 1
--------
All domains shown in the mirrored layout.

Figure 2
--------
Same mirrored layout, but highlight selected anatomical groups only.
Selected domains keep their original colors; all other domains are shown in gray.
The gene-delta half also uses the same mask rule: selected domains are shown with
true delta colors, and other domains are shown in light gray.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from spatial_delta_utils import compute_spatial_delta_from_files
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# ============================================================
# CONFIG
# ============================================================

DISEASE_NORMAL_SPECS = [
    {
        "age": 2.5,
        "label": "2.5m",
        "disease_path": (
            "/media/huifang/data/registration/xenium/mouse_brain/"
            "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
            "virtual_Disease_time_2.500.h5ad"
        ),
        "normal_path": (
            "/media/huifang/data/registration/xenium/mouse_brain/"
            "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
            "virtual_Disease_time_102.500.h5ad"
        ),
    },
    {
        "age": 5.7,
        "label": "5.7m",
        "disease_path": (
            "/media/huifang/data/registration/xenium/mouse_brain/"
            "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
            "virtual_Disease_time_5.700.h5ad"
        ),
        "normal_path": (
            "/media/huifang/data/registration/xenium/mouse_brain/"
            "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
            "virtual_Disease_time_105.700.h5ad"
        ),
    },
    {
        "age": 13.4,
        "label": "13.4m",
        "disease_path": (
            "/media/huifang/data/registration/xenium/mouse_brain/"
            "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
            "virtual_Disease_time_13.400.h5ad"
        ),
        "normal_path": (
            "/media/huifang/data/registration/xenium/mouse_brain/"
            "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
            "virtual_Disease_time_113.400.h5ad"
        ),
    },
]

# Unified count preprocessing used by every disease-delta script.
# raw counts -> per-location library-size normalization -> optional log1p
#            -> disease - normal
LIBRARY_SIZE_NORMALIZE = True
NORMALIZATION_TARGET_SUM = 253
LOG1P_BEFORE_ANALYSIS = True

# Spatial delta is always computed on matched locations. When True, only
# locations valid in BOTH disease and normal are retained in the delta AnnData.
DELTA_USE_COMMON_VALID = True
COORDINATE_DECIMALS = 5

GENE_NAME = "Ntng1"
# GENE_NAME = "Slc17a6"
# GENE_NAME = "Gfap"
SPATIAL_KEY = "spatial"
DOMAIN_KEY = "axis_joint_domain"
VALID_KEY = "valid"

OUTDIR = (
    "/media/huifang/data/registration/paper_figure/"
    "Figure6_disease/delta_mirrored_grid"
)

FIGURE_MODE = "show"  # save / show / both
SHOW_BLOCK = True
FIG_DPI = 400

FIXED_BOX = (0.0, 1024.0, 0.0, 1024.0)
SWAP_XY = True
INVERT_X = True
INVERT_Y = False

MIRROR_RIGHT_TRIM = 70.0
DRAW_MIRROR_DIVIDER = True

DELTA_CMAP = "coolwarm"
DELTA_VMAX_PERCENTILE = 99.0

GENE_POINT_SIZE = 5.0
DOMAIN_POINT_SIZE = 5.0
GENE_ALPHA_MIN = 0.08
GENE_ALPHA_MAX = 0.95
DOMAIN_ALPHA = 0.85
OTHER_GENE_ALPHA = 0.18
OTHER_DOMAIN_ALPHA = 0.55

BACKGROUND_GRAY = "#CFCFCF"
FRAME_COLOR = "#A9A9A9"
TEXT_COLOR = "#222222"
FRAME_WIDTH = 1.5
ANNOTATION_FONTSIZE = 11.5

SELECTED_GROUPS = [
    {"name": "Cortex", "labels": [1, 4, 6, 9, 10]},
    {"name": "Hippocampus", "labels": [8, 11]},
    {"name": "Thalamus", "labels": [0]},
    {"name": "Hypothalamus", "labels": [3]},
]
SELECTED_LABELS = sorted({lab for group in SELECTED_GROUPS for lab in group["labels"]})


# ============================================================
# DATA
# ============================================================

@dataclass
class DeltaRecord:
    age: float
    label: str
    disease_path: str
    normal_path: str
    coords: np.ndarray
    delta_values: np.ndarray
    domains: np.ndarray


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )





def finalize_figure(fig, save_path: str, mode: str) -> None:
    mode = str(mode).lower()
    if mode not in {"save", "show", "both"}:
        raise ValueError("mode must be 'save', 'show', or 'both'")

    if mode in {"save", "both"}:
        ensure_dir(os.path.dirname(save_path) or ".")
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"[SAVE] {save_path}")

    if mode in {"show", "both"}:
        plt.show(block=SHOW_BLOCK)

    plt.close(fig)



def to_dense_1d(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=float).ravel()



def normalize_domains(values: Sequence) -> np.ndarray:
    out = []
    for value in np.asarray(values):
        try:
            out.append(int(float(value)))
        except Exception:
            out.append(-1)
    return np.asarray(out, dtype=int)



def transform_loaded_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    coords = coords[:, :2].copy()

    if SWAP_XY:
        coords = coords[:, [1, 0]]

    xmin, xmax, ymin, ymax = FIXED_BOX
    if INVERT_X:
        coords[:, 0] = xmin + xmax - coords[:, 0]
    if INVERT_Y:
        coords[:, 1] = ymin + ymax - coords[:, 1]
    return coords



def load_delta_record(spec: Dict, gene_name: str) -> DeltaRecord:
    """Compute disease-minus-normal delta on the fly using the shared utility."""
    delta_adata = compute_spatial_delta_from_files(
        disease_path=spec["disease_path"],
        normal_path=spec["normal_path"],
        normalize_total=LIBRARY_SIZE_NORMALIZE,
        target_sum=NORMALIZATION_TARGET_SUM,
        use_log1p=LOG1P_BEFORE_ANALYSIS,
        valid_key=VALID_KEY,
        spatial_key=SPATIAL_KEY,
        domain_key=DOMAIN_KEY,
        genes=[gene_name],
        use_common_valid=DELTA_USE_COMMON_VALID,
        coordinate_decimals=COORDINATE_DECIMALS,
        return_result=False,
    )

    coords = transform_loaded_coords(delta_adata.obsm[SPATIAL_KEY])
    delta_values = to_dense_1d(delta_adata[:, [gene_name]].X)
    domains = normalize_domains(delta_adata.obs[DOMAIN_KEY].to_numpy())

    return DeltaRecord(
        age=float(spec["age"]),
        label=str(spec["label"]),
        disease_path=str(spec["disease_path"]),
        normal_path=str(spec["normal_path"]),
        coords=coords,
        delta_values=delta_values,
        domains=domains,
    )



def load_records(gene_name: str) -> List[DeltaRecord]:
    records = [
        load_delta_record(spec, gene_name)
        for spec in DISEASE_NORMAL_SPECS
    ]
    return sorted(records, key=lambda x: x.age)


# ============================================================
# GEOMETRY
# ============================================================

def mirrored_parameters() -> Tuple[float, float, float, float, float]:
    xmin, xmax, ymin, ymax = FIXED_BOX
    usable_xmax = xmax - float(MIRROR_RIGHT_TRIM)
    if usable_xmax <= xmin:
        raise ValueError("MIRROR_RIGHT_TRIM too large for FIXED_BOX")
    seam_x = usable_xmax
    return xmin, usable_xmax, ymin, ymax, seam_x



def mirrored_extent() -> Tuple[float, float, float, float]:
    xmin, usable_xmax, ymin, ymax, seam_x = mirrored_parameters()
    width = usable_xmax - xmin
    return xmin, seam_x + width, ymin, ymax



def mirrored_half_coords(record: DeltaRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, usable_xmax, ymin, ymax, seam_x = mirrored_parameters()
    x = np.asarray(record.coords[:, 0], dtype=float)
    y = np.asarray(record.coords[:, 1], dtype=float)

    keep = (
        np.isfinite(x) & np.isfinite(y) &
        (x >= xmin) & (x <= usable_xmax) &
        (y >= ymin) & (y <= ymax)
    )

    x = x[keep]
    y = y[keep]
    gene_xy = np.column_stack([x, y])
    domain_xy = np.column_stack([seam_x + (usable_xmax - x), y])
    return gene_xy, domain_xy, keep


# ============================================================
# COLOR HELPERS
# ============================================================

def shared_delta_vmax(records: Sequence[DeltaRecord]) -> float:
    values = np.concatenate([record.delta_values for record in records])
    values = values[np.isfinite(values)]
    vmax = np.percentile(np.abs(values), DELTA_VMAX_PERCENTILE) if values.size else 1.0
    return max(float(vmax), 1e-8)



def build_domain_palette(records: Sequence[DeltaRecord]) -> Dict[int, tuple]:
    labels = sorted(
        {
            int(domain)
            for record in records
            for domain in np.unique(record.domains)
            if int(domain) >= 0
        }
    )
    cmap = plt.get_cmap("tab20")
    return {label: cmap(i % 20) for i, label in enumerate(labels)}



def delta_facecolors(values: np.ndarray, vmax: float) -> np.ndarray:
    norm = Normalize(vmin=-vmax, vmax=vmax, clip=True)
    colors = plt.get_cmap(DELTA_CMAP)(norm(np.asarray(values, dtype=float)))
    strength = np.clip(np.abs(values) / vmax, 0.0, 1.0)
    colors[:, 3] = GENE_ALPHA_MIN + (GENE_ALPHA_MAX - GENE_ALPHA_MIN) * strength
    return colors



def gene_facecolors_selected(
    values: np.ndarray,
    domains: np.ndarray,
    vmax: float,
    selected_domains: Optional[Sequence[int]] = None,
) -> np.ndarray:
    colors = delta_facecolors(values, vmax=vmax)
    if selected_domains is None:
        return colors

    selected_set = {int(v) for v in selected_domains}
    out = colors.copy()
    mask_other = ~np.isin(domains.astype(int), list(selected_set))
    out[mask_other, :3] = np.array([0.80, 0.80, 0.80])
    out[mask_other, 3] = OTHER_GENE_ALPHA
    return out



def domain_facecolors_selected(
    domains: np.ndarray,
    palette: Dict[int, tuple],
    selected_domains: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if selected_domains is None:
        return np.asarray([palette.get(int(d), BACKGROUND_GRAY) for d in domains], dtype=object)

    selected_set = {int(v) for v in selected_domains}
    out = []
    for d in domains.astype(int):
        if d in selected_set:
            out.append(palette.get(d, BACKGROUND_GRAY))
        else:
            out.append(BACKGROUND_GRAY)
    return np.asarray(out, dtype=object)


# ============================================================
# DRAW HELPERS
# ============================================================

def draw_whole_bounding_box(ax, extent: Tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = extent
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=FRAME_COLOR,
            linewidth=FRAME_WIDTH,
            zorder=20,
        )
    )



def style_panel_axis(ax) -> None:
    xmin, xmax, ymin, ymax = mirrored_extent()
    padx = 0.02 * (xmax - xmin)
    pady = 0.04 * (ymax - ymin)
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)



def add_top_labels(ax, gene_name: str) -> None:
    ax.text(
        0.23, 1.015, f"{gene_name} Δ",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=ANNOTATION_FONTSIZE,
        fontweight="bold", color=TEXT_COLOR,
        clip_on=False,
    )
    ax.text(
        0.77, 1.015, "Spatial domains",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=ANNOTATION_FONTSIZE,
        fontweight="bold", color=TEXT_COLOR,
        clip_on=False,
    )



def plot_mirrored_record(
    ax,
    record: DeltaRecord,
    gene_name: str,
    vmax: float,
    domain_palette: Dict[int, tuple],
    selected_domains: Optional[Sequence[int]] = None,
) -> None:
    gene_xy, domain_xy, keep = mirrored_half_coords(record)
    values = record.delta_values[keep]
    domains = record.domains[keep]

    ax.scatter(
        gene_xy[:, 0], gene_xy[:, 1],
        c=gene_facecolors_selected(values, domains, vmax=vmax, selected_domains=selected_domains),
        s=GENE_POINT_SIZE, linewidths=0, rasterized=True, zorder=2,
    )
    ax.scatter(
        domain_xy[:, 0], domain_xy[:, 1],
        c=domain_facecolors_selected(domains, domain_palette, selected_domains=selected_domains),
        s=DOMAIN_POINT_SIZE,
        alpha=DOMAIN_ALPHA if selected_domains is None else 1.0,
        linewidths=0, rasterized=True, zorder=2,
    )

    extent = mirrored_extent()
    draw_whole_bounding_box(ax, extent)

    if DRAW_MIRROR_DIVIDER:
        seam_x = mirrored_parameters()[4]
        ax.plot([seam_x, seam_x], [extent[2], extent[3]], color=FRAME_COLOR, linewidth=0.9, zorder=19)

    add_top_labels(ax, gene_name)
    ax.set_title(record.label, fontweight="bold", pad=12)
    style_panel_axis(ax)


# ============================================================
# LEGENDS
# ============================================================

def make_domain_legend_handles(labels: Sequence[int], include_other: bool = False, other_label: str = "Other domains") -> List[Line2D]:
    palette = plt.get_cmap("tab20")
    handles = []
    for label in labels:
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=palette(int(label) % 20), markeredgecolor="none", markersize=7, label=f"Domain {int(label)}")
        )
    if include_other:
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=BACKGROUND_GRAY, markeredgecolor="none", markersize=7, label=other_label)
        )
    return handles



def make_grouped_domain_legend_handles(groups: Sequence[Dict], include_other: bool = False, other_label: str = "Other domains", palette: Optional[Dict[int, tuple]] = None) -> Tuple[List[Line2D], set]:
    if palette is None:
        palette = {}
    handles: List[Line2D] = []
    section_headers = set()

    for group_idx, group in enumerate(groups):
        group_name = str(group["name"])
        labels = [int(v) for v in group["labels"]]

        handles.append(Line2D([], [], linestyle="", marker="", color="none", label=group_name))
        section_headers.add(group_name)

        for label in labels:
            handles.append(
                Line2D([0], [0], marker="o", linestyle="", markerfacecolor=palette.get(label, BACKGROUND_GRAY), markeredgecolor="none", markersize=7, label=f"Domain {label}")
            )

        if group_idx < len(groups) - 1:
            handles.append(Line2D([], [], linestyle="", marker="", color="none", label=" "))

    if include_other:
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=BACKGROUND_GRAY, markeredgecolor="none", markersize=7, label=other_label)
        )
    return handles, section_headers


# ============================================================
# FIGURES
# ============================================================

def add_shared_colorbar(fig, axes, gene_name: str, vmax: float) -> None:
    cax = fig.add_axes([0.935, 0.30, 0.009, 0.42])
    sm = ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap=DELTA_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(f"{gene_name} disease − control delta", fontsize=10.5)
    cb.ax.tick_params(labelsize=9)



def plot_figure1_all_domains(records: Sequence[DeltaRecord], gene_name: str, palette: Dict[int, tuple], save_path: str, mode: str) -> None:
    set_style()
    vmax = shared_delta_vmax(records)
    fig, axes = plt.subplots(1, len(records), figsize=(16.0, 5.4), squeeze=False)
    axes = axes.ravel()

    for ax, record in zip(axes, records):
        plot_mirrored_record(ax, record, gene_name, vmax, palette, selected_domains=None)

    fig.subplots_adjust(left=0.03, right=0.92, top=0.82, bottom=0.20, wspace=0.06)
    add_shared_colorbar(fig, axes, gene_name, vmax)

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markeredgecolor="none", markersize=6.5, label=f"Domain {label}")
            for label, color in sorted(palette.items())
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=min(7, max(1, len(palette))),
        frameon=False,
        title="Spatial domains",
        fontsize=8.5,
        title_fontsize=9.5,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    fig.suptitle(f"Figure 1 | Disease-associated spatial differences across time | {gene_name}", y=0.965, fontsize=16, fontweight="bold")
    finalize_figure(fig, save_path, mode)



def plot_figure2_selected_domains(records: Sequence[DeltaRecord], gene_name: str, palette: Dict[int, tuple], save_path: str, mode: str) -> None:
    set_style()
    vmax = shared_delta_vmax(records)
    fig, axes = plt.subplots(1, len(records), figsize=(16.0, 5.4), squeeze=False)
    axes = axes.ravel()

    for ax, record in zip(axes, records):
        plot_mirrored_record(ax, record, gene_name, vmax, palette, selected_domains=SELECTED_LABELS)

    fig.subplots_adjust(left=0.03, right=0.92, top=0.82, bottom=0.24, wspace=0.06)
    add_shared_colorbar(fig, axes, gene_name, vmax)

    handles, section_headers = make_grouped_domain_legend_handles(
        groups=SELECTED_GROUPS,
        include_other=True,
        other_label="Other domains",
        palette=palette,
    )
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=False,
        title=None,
        fontsize=8.5,
        handlelength=1.0,
        handletextpad=0.55,
        labelspacing=0.35,
        columnspacing=1.25,
    )
    for text in legend.get_texts():
        label = text.get_text()
        if label in section_headers:
            text.set_fontweight("bold")
            text.set_fontsize(9.5)
        elif label == " ":
            text.set_text("")

    fig.suptitle(f"Figure 2 | Selected anatomical groups | {gene_name}", y=0.965, fontsize=16, fontweight="bold")
    finalize_figure(fig, save_path, mode)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mirrored disease-delta figures.")
    parser.add_argument("--gene", default=GENE_NAME)
    parser.add_argument("--figure-mode", choices=["save", "show", "both"], default=FIGURE_MODE)
    args = parser.parse_args()

    records = load_records(gene_name=args.gene)
    palette = build_domain_palette(records)
    #
    # plot_figure1_all_domains(
    #     records=records,
    #     gene_name=args.gene,
    #     palette=palette,
    #     save_path=os.path.join(OUTDIR, f"figure1_disease_delta_mirrored_all_domains_{args.gene}.png"),
    #     mode=args.figure_mode,
    # )
    plot_figure2_selected_domains(
        records=records,
        gene_name=args.gene,
        palette=palette,
        save_path=os.path.join(OUTDIR, f"figure2_disease_delta_mirrored_selected_domains_{args.gene}.png"),
        mode=args.figure_mode,
    )


if __name__ == "__main__":
    main()
