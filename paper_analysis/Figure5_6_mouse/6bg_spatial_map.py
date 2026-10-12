#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case-study plotting for disease-versus-normal spatial delta analysis.

The script contains two independent visualization components:
1. A held-out Arc AD-WT contrast map at 5.7 months for validation.
2. Flexible three-stage spatial progression maps for user-defined marker genes.

Progression markers and their reader-facing labels are configured through
PROGRESSION_CASES. By default:
    Gfap | Progressive increase
    Mobp | Spatially mixed response

Add, remove, or replace cases in PROGRESSION_CASES without changing the
plotting functions.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from spatial_delta_utils import compute_spatial_delta_from_files


# ============================================================
# CONFIG
# ============================================================

DISEASE_NORMAL_SPECS = [
    {
        "age": 2.5,
        "label": "2.5m",
        "disease_path": (
            "/media/huifang/data/registration/mouse_brain/"
            "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
            "virtual_Disease_time_2.500.h5ad"
        ),
        "normal_path": (
            "/media/huifang/data/registration/mouse_brain/"
            "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
            "virtual_Disease_time_102.500.h5ad"
        ),
    },
    {
        "age": 5.7,
        "label": "5.7m",
        "disease_path": (
            "/media/huifang/data/registration/mouse_brain/"
            "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
            "virtual_Disease_time_5.700.h5ad"
        ),
        "normal_path": (
            "/media/huifang/data/registration/mouse_brain/"
            "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
            "virtual_Disease_time_105.700.h5ad"
        ),
    },
    {
        "age": 13.4,
        "label": "13.4m",
        "disease_path": (
            "/media/huifang/data/registration/mouse_brain/"
            "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
            "virtual_Disease_time_13.400.h5ad"
        ),
        "normal_path": (
            "/media/huifang/data/registration/mouse_brain/"
            "virtual_tissue/Disease_time_grid256_spatial1.0/virtual_cf/"
            "virtual_Disease_time_113.400.h5ad"
        ),
    },
]

LIBRARY_SIZE_NORMALIZE = True
NORMALIZATION_TARGET_SUM = 253
LOG1P_BEFORE_ANALYSIS = True

DELTA_USE_COMMON_VALID = True
COORDINATE_DECIMALS = 5

SPATIAL_KEY = "spatial"
DOMAIN_KEY = "axis_joint_domain"
VALID_KEY = "valid"

OUTDIR = (
    "/media/huifang/data/registration/paper_analysis/Figure56_mouse/6_case_spatial_map/"
)

FIGURE_MODE = "save"  # save / show / both
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

GENE_POINT_SIZE = 1.0
DOMAIN_POINT_SIZE = 1.0
GENE_ALPHA_MIN = 0.38
GENE_ALPHA_MAX = 0.95
DOMAIN_ALPHA = 0.85

BACKGROUND_GRAY = "#CFCFCF"
FRAME_COLOR = "#A9A9A9"
TEXT_COLOR = "#222222"
FRAME_WIDTH = 1.5
ANNOTATION_FONTSIZE = 11.5
DOMAIN_NUMBER_FONTSIZE = 13

SELECTED_GROUPS = [
    {"name": "Cortex", "labels": [1, 4, 6, 9, 10]},
    {"name": "Hippocampus", "labels": [8, 11]},
    {"name": "Thalamus", "labels": [0]},
    {"name": "Hypothalamus", "labels": [3]},
]
SELECTED_LABELS = sorted({lab for group in SELECTED_GROUPS for lab in group["labels"]})

# Flexible progression case studies.
# Each entry controls:
#   gene      : gene symbol loaded from the h5ad files
#   label     : reader-facing text displayed after the gene symbol
#   file_tag  : short string used in output filenames
#
# Add/remove cases here without modifying the plotting code.
PROGRESSION_CASES = [
    {
        "gene": "Gfap",
        "label": "Progressive increase",
        "file_tag": "gfap",
    },
    {
        "gene": "Mobp",
        "label": "Spatially mixed response",
        "file_tag": "mobp",
    },
]


FIG6H_AGE_LABELS = {
    "2.5m": "2.5 months",
    "5.7m": "5.7 months",
    "13.4m": "13.4 months",
}

FIG6H_COLORBAR_LABEL = "AD-WT expression change (log1p)"


# ============================================================
# DATA
# ============================================================

@dataclass
class DeltaRecord:
    age: float
    label: str
    gene_name: str
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
        gene_name=str(gene_name),
        disease_path=str(spec["disease_path"]),
        normal_path=str(spec["normal_path"]),
        coords=coords,
        delta_values=delta_values,
        domains=domains,
    )


def load_records(gene_name: str) -> List[DeltaRecord]:
    records = [load_delta_record(spec, gene_name) for spec in DISEASE_NORMAL_SPECS]
    return sorted(records, key=lambda x: x.age)


# ============================================================
# GEOMETRY AND COLOR
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
    keep = np.isfinite(x) & np.isfinite(y) & (x >= xmin) & (x <= usable_xmax) & (y >= ymin) & (y <= ymax)
    x = x[keep]
    y = y[keep]
    gene_xy = np.column_stack([x, y])
    domain_xy = np.column_stack([seam_x + (usable_xmax - x), y])
    return gene_xy, domain_xy, keep


def shared_delta_vmax(records: Sequence[DeltaRecord]) -> float:
    values = np.concatenate([record.delta_values for record in records])
    values = values[np.isfinite(values)]
    vmax = np.percentile(np.abs(values), DELTA_VMAX_PERCENTILE) if values.size else 1.0
    return max(float(vmax), 1e-8)


def build_domain_palette(records: Sequence[DeltaRecord]) -> Dict[int, tuple]:
    labels = sorted({int(domain) for record in records for domain in np.unique(record.domains) if int(domain) >= 0})
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
    """
    Keep true delta colors on selected domains and render all other domains gray.

    This reproduces the selected-domain gene-map behavior from the original
    mirrored plotting script.
    """
    colors = delta_facecolors(values, vmax=vmax)
    if selected_domains is None:
        return colors

    selected_set = {int(v) for v in selected_domains}
    out = colors.copy()
    mask_other = ~np.isin(domains.astype(int), list(selected_set))
    out[mask_other, :3] = np.array([0.80, 0.80, 0.80])
    out[mask_other, 3] = 0.18
    return out


# ============================================================
# AXIS/PANEL HELPERS
# ============================================================

def draw_whole_bounding_box(ax, extent: Tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = extent
    ax.add_patch(
        Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=FRAME_COLOR, linewidth=FRAME_WIDTH, zorder=20)
    )


def style_panel_axis(ax, extent: Tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = extent
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


def add_mirrored_top_labels(ax, gene_name: str) -> None:
    ax.text(0.23, 1.015, f"{gene_name} Δ", transform=ax.transAxes, ha="center", va="bottom", fontsize=ANNOTATION_FONTSIZE, fontweight="bold", color=TEXT_COLOR, clip_on=False)
    ax.text(0.77, 1.015, "Spatial domains", transform=ax.transAxes, ha="center", va="bottom", fontsize=ANNOTATION_FONTSIZE, fontweight="bold", color=TEXT_COLOR, clip_on=False)


def annotate_domain_numbers_on_mirrored_panel(ax, domain_xy: np.ndarray, domains: np.ndarray, labels: Optional[Sequence[int]] = None) -> None:
    if labels is None:
        labels = sorted(np.unique(domains[domains >= 0]).tolist())
    label_set = set(int(x) for x in labels)
    for domain_id in sorted(label_set):
        mask = domains == int(domain_id)
        if not np.any(mask):
            continue
        x = float(np.median(domain_xy[mask, 0]))
        y = float(np.median(domain_xy[mask, 1]))
        ax.text(
            x, y, str(domain_id),
            ha="center", va="center",
            fontsize=DOMAIN_NUMBER_FONTSIZE,
            fontweight="bold",
            color="black",
            zorder=25,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.75),
        )


# ============================================================
# FIGURES
# ============================================================

def add_shared_colorbar(fig, cmap_name: str, vmax: float, label: str, rect: Sequence[float]) -> None:
    cax = fig.add_axes(rect)
    sm = ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap=cmap_name)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, fontsize=12)
    cb.ax.tick_params(labelsize=10)


def plot_arc_mirrored_5p7m(record: DeltaRecord, palette: Dict[int, tuple], save_path: str, mode: str) -> None:
    set_style()
    vmax = shared_delta_vmax([record])
    fig, ax = plt.subplots(figsize=(8.2, 5.5))

    gene_xy, domain_xy, keep = mirrored_half_coords(record)
    values = record.delta_values[keep]
    domains = record.domains[keep]

    ax.scatter(gene_xy[:, 0], gene_xy[:, 1], c=delta_facecolors(values, vmax=vmax), s=GENE_POINT_SIZE, linewidths=0, rasterized=True, zorder=2)
    ax.scatter(domain_xy[:, 0], domain_xy[:, 1], c=np.asarray([palette.get(int(d), BACKGROUND_GRAY) for d in domains], dtype=object), s=DOMAIN_POINT_SIZE, alpha=DOMAIN_ALPHA, linewidths=0, rasterized=True, zorder=2)

    extent = mirrored_extent()
    # draw_whole_bounding_box(ax, extent)
    # if DRAW_MIRROR_DIVIDER:
    #     seam_x = mirrored_parameters()[4]
    #     ax.plot([seam_x, seam_x], [extent[2], extent[3]], color=FRAME_COLOR, linewidth=0.9, zorder=19)

    # add_mirrored_top_labels(ax, record.gene_name)
    annotate_domain_numbers_on_mirrored_panel(ax, domain_xy=domain_xy, domains=domains)
    style_panel_axis(ax, extent)
    # ax.set_title(f"5.7m disease−WT delta | {record.gene_name}", fontweight="bold", pad=12)
    ax.set_aspect("equal")

    fig.subplots_adjust(left=0.03, right=0.89, top=0.84, bottom=0.08)
    add_shared_colorbar(fig, DELTA_CMAP, vmax, "expression change (log1p)", rect=[0.905, 0.22, 0.018, 0.52])
    finalize_figure(fig, save_path, mode)


def plot_gene_stage_selected_domains_individual(
    records: Sequence[DeltaRecord],
    gene_name: str,
    out_prefix: str,
    mode: str,
    response_label: Optional[str] = None,
) -> None:
    """
    Plot one flexible spatial-response progression case study.

    Outputs
    -------
    1. One stage-specific map for each biological age.
       The map title is the actual age rather than "Stage 1/2/3" or "gene Δ".

    2. One combined three-stage progression row:
           2.5 months  ->  5.7 months  ->  13.4 months
       with a single shared AD-WT expression-change colorbar.

    The spatial-delta values, selected-domain masking, coordinate transform,
    and shared gene-specific vmax are unchanged.
    """
    set_style()

    # Prevent accidental relabeling of spatial data from another gene.
    mismatched = [
        str(record.gene_name)
        for record in records
        if str(record.gene_name) != str(gene_name)
    ]
    if mismatched:
        raise ValueError(
            f"Gene/record mismatch for {gene_name}: "
            f"records contain {sorted(set(mismatched))}"
        )

    records = sorted(records, key=lambda x: x.age)
    vmax = shared_delta_vmax(records)
    extent = (
        mirrored_parameters()[0],
        mirrored_parameters()[1],
        FIXED_BOX[2],
        FIXED_BOX[3],
    )

    if response_label is None:
        response_label = "Spatial AD-WT response"
    response_label = str(response_label)

    # # ============================================================
    # # A. Save the individual age panels
    # # ============================================================
    # for record in records:
    #     fig, ax = plt.subplots(figsize=(5.1, 4.9))
    #
    #     gene_xy, _, keep = mirrored_half_coords(record)
    #     values = record.delta_values[keep]
    #     domains = record.domains[keep]
    #
    #     ax.scatter(
    #         gene_xy[:, 0],
    #         gene_xy[:, 1],
    #         c=gene_facecolors_selected(
    #             values=values,
    #             domains=domains,
    #             vmax=vmax,
    #             selected_domains=SELECTED_LABELS,
    #         ),
    #         s=GENE_POINT_SIZE,
    #         linewidths=0,
    #         rasterized=True,
    #         zorder=2,
    #     )
    #
    #     style_panel_axis(ax, extent)
    #
    #     # Actual biological age is the panel title.
    #     ax.set_title(
    #         FIG6H_AGE_LABELS.get(record.label, record.label),
    #         fontsize=15,
    #         pad=3,
    #         # fontweight="bold",
    #     )
    #
    #     # Gene + response-pattern annotation replaces the computational "gene Δ".
    #     fig.text(
    #         0.45,
    #         0.955,
    #         rf"$\it{{{gene_name}}}$ | {response_label}",
    #         ha="center",
    #         va="top",
    #         fontsize=13,
    #     )
    #
    #     fig.subplots_adjust(
    #         left=0.03,
    #         right=0.87,
    #         top=0.86,
    #         bottom=0.06,
    #     )
    #
    #     add_shared_colorbar(
    #         fig,
    #         DELTA_CMAP,
    #         vmax,
    #         FIG6H_COLORBAR_LABEL,
    #         rect=[0.89, 0.22, 0.02, 0.52],
    #     )
    #
    #     age_tag = record.label.replace(".", "p")
    #     save_path = f"{out_prefix}_{age_tag}.png"
    #     finalize_figure(fig, save_path, mode)

    # ============================================================
    # B. Save a combined three-stage progression row
    # ============================================================
    n_stages = len(records)
    if n_stages == 0:
        return

    fig, axes = plt.subplots(
        1,
        n_stages,
        figsize=(4.1 * n_stages, 4.7),
        squeeze=True,
    )
    axes = axes.ravel()

    for ax, record in zip(axes, records):
        gene_xy, _, keep = mirrored_half_coords(record)
        values = record.delta_values[keep]
        domains = record.domains[keep]

        ax.scatter(
            gene_xy[:, 0],
            gene_xy[:, 1],
            c=gene_facecolors_selected(
                values=values,
                domains=domains,
                vmax=vmax,
                selected_domains=SELECTED_LABELS,
            ),
            s=GENE_POINT_SIZE,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

        style_panel_axis(ax, extent)
        ax.set_title(
            FIG6H_AGE_LABELS.get(record.label, record.label),
            fontsize=14,
            pad=1,
            # fontweight="bold",
            y=0.92
        )

    fig.suptitle(
        rf"$\it{{{gene_name}}}$ | {response_label}",
        fontsize=16,
        # fontweight="bold",
        y=0.93,
    )

    # A single directional annotation communicates temporal ordering without
    # introducing "Stage 1/2/3" terminology.
    fig.text(
        0.46,
        0.09,
        "AD progression  →",
        ha="center",
        va="center",
        fontsize=16,
        color=TEXT_COLOR,
    )

    fig.subplots_adjust(
        left=0.025,
        right=0.90,
        top=0.84,
        bottom=0.13,
        wspace=0.08,
    )

    add_shared_colorbar(
        fig,
        DELTA_CMAP,
        vmax,
        FIG6H_COLORBAR_LABEL,
        rect=[0.915, 0.23, 0.014, 0.49],
    )

    combined_path = f"{out_prefix}_progression.png"
    finalize_figure(fig, combined_path, mode)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot disease-versus-WT spatial case-study maps."
    )
    parser.add_argument(
        "--figure-mode",
        choices=["save", "show", "both"],
        default=FIGURE_MODE,
    )
    args = parser.parse_args()

    ensure_dir(OUTDIR)

    # ------------------------------------------------------------
    # Validation panel: held-out Arc contrast at 5.7 months.
    # This remains independent from the progression-marker list.
    # ------------------------------------------------------------
    arc_records = load_records("Arc")
    palette = build_domain_palette(arc_records)

    record_57 = [
        record
        for record in arc_records
        if np.isclose(record.age, 5.7)
    ][0]

    plot_arc_mirrored_5p7m(
        record=record_57,
        palette=palette,
        save_path=os.path.join(
            OUTDIR,
            "figure1_arc_mirrored_5p7m_all_domains.png",
        ),
        mode=args.figure_mode,
    )

    # ------------------------------------------------------------
    # Flexible progression case studies.
    # Add/remove/replace markers only in PROGRESSION_CASES.
    # ------------------------------------------------------------
    for case_index, case in enumerate(
        PROGRESSION_CASES,
        start=1,
    ):
        gene_name = str(case["gene"])
        response_label = str(case.get("label", "Spatial AD-WT response"))
        file_tag = str(
            case.get(
                "file_tag",
                gene_name.lower(),
            )
        )

        print(
            f"[INFO] Progression case {case_index}: "
            f"{gene_name} | {response_label}"
        )

        records = load_records(gene_name)

        plot_gene_stage_selected_domains_individual(
            records=records,
            gene_name=gene_name,
            response_label=response_label,
            out_prefix=os.path.join(
                OUTDIR,
                f"figure_progression_{case_index}_{file_tag}",
            ),
            mode=args.figure_mode,
        )


if __name__ == "__main__":
    main()


