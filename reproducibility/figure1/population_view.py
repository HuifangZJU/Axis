#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DLPFC population-level 2D visualization.

This script visualizes the integration-generation process in three figures:

Figure 1
    Four ORIGINAL measured DLPFC gene maps loaded directly from the native
    per-slice H5AD files in a 2 x 2 grid. Each slice uses its own high-contrast
    sequential colormap so that slice identity is visually distinct.

Figure 2
    A single evidence plane containing all source-slice observations overlaid
    together in the common coordinate system. Each observation inherits the
    colormap of its source slice.

Figure 3
    The final generated virtual gene map, with its own independent color scale.

Notes
-----
1. No CLI arguments. Edit the CONFIG block only.
2. Gene expression is visualized as log1p(raw counts).
3. The four source slices are inferred from the generation metadata through the
   anchor-identifying obs column.
4. The generated virtual tissue is filtered by obs['valid'] when available.
"""

from __future__ import annotations
from axis.settings import data_path
from axis.config import load_config

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from scipy import sparse


# ============================================================
# CONFIG
# ============================================================

VIRTUAL_PATH = Path(
    data_path("DLPFC/huifang/virtual_tissue/"
    "dlpfc_z_group1_grid80_spatial1/integration/"
    "integrated_dlpfc_z_ref_0_visium_ot.h5ad")
)

META_PATH = Path(
    data_path("DLPFC/huifang/virtual_tissue/"
    "dlpfc_z_group1_grid80_spatial1/integration/"
    "integrated_dlpfc_z_ref_0_visium_ot_generation_meta.h5ad")
)

# Original measured DLPFC sections used ONLY by Figure 1.
ROOT = Path(load_config('dlpfc').root)
GROUP_ID = 1
SAMPLE_IDS: Tuple[int, ...] = (0, 1, 2, 3)
COORD_KEY = "spatial"

MARKER_GENE = "MOBP"
OUT_DIR: Optional[Path] = None

ANCHOR_KEY: Optional[str] = "source_anchor_id"
ANCHOR_KEY_CANDIDATES = (
    "source_anchor_id",
    "anchor_id",
    "anchor",
    "anchor_idx",
    "source_id",
    "source",
    "slice_id",
    "slice",
    "sample_id",
    "sample",
    "section_id",
    "section",
    "source_slice",
    "source_sample",
)
EXPECTED_N_ANCHORS = 4
ANCHOR_ORDER: Optional[List[str]] = None

INVERT_Y = True
# Gene-map scaling follows the separate spatial-map workflow:
# each map uses the full observed log1p range [0, max] with opaque points.
# No percentile clipping and no expression-dependent transparency.
UPPER_PERCENTILE = 99.5  # retained only for backward compatibility; not used for gene maps

FIG1_POINT_SIZE = 24.0
FIG2_POINT_SIZE = 24.0
FIG3_POINT_SIZE = 26.0

FIG1_SIZE = (7.0, 6.2)
FIG2_SIZE = (4.4, 4.1)
FIG3_SIZE = (4.1, 4.1)

# Distinct slice identities are preserved, but the sequential ramps are
# deliberately high-contrast so low/mid expression remains visible.
SLICE_BASE_COLORS = (
    "#4C78A8",  # blue
    "#59A89C",  # teal
    "#D08445",  # warm orange-brown
    "#B279A2",  # mauve
)
GENERATED_BASE_COLOR = "#C4473A"


# ============================================================
# STYLE / IO
# ============================================================

def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "legend.title_fontsize": 7.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.dpi": 120,
        }
    )


def save_or_show(
    fig: plt.Figure,
    outdir: Optional[Path],
    stem: str,
    *,
    dpi: int = 400,
    close: bool = True,
) -> Optional[Dict[str, Path]]:
    if outdir is None:
        plt.show()
        if close:
            plt.close(fig)
        return None

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)

    if close:
        plt.close(fig)

    print(f"[OK] saved PNG: {png_path}")
    print(f"[OK] saved PDF: {pdf_path}")
    return {"png": png_path, "pdf": pdf_path}


# ============================================================
# ROBUST ANNDATA READER
# ============================================================

def safe_read_h5ad(path: Path) -> ad.AnnData:
    """Read H5AD while handling the known older anndata/pandas `_clean_uns` issue."""
    path = Path(path)
    try:
        return ad.read_h5ad(str(path))
    except TypeError as exc:
        if "Unordered Categoricals can only compare equality or not" not in str(exc):
            raise

        read_globals = getattr(ad.read_h5ad, "__globals__", None)
        if not isinstance(read_globals, dict) or "_clean_uns" not in read_globals:
            raise

        old_clean_uns = read_globals["_clean_uns"]
        print(
            f"[WARN] {path.name}: legacy anndata _clean_uns failed; "
            "retrying with that cleanup step disabled."
        )
        try:
            read_globals["_clean_uns"] = lambda d: None
            return ad.read_h5ad(str(path))
        finally:
            read_globals["_clean_uns"] = old_clean_uns


# ============================================================
# DATA LOADING
# ============================================================

def read_inputs(virtual_path: Path, meta_path: Path) -> Tuple[ad.AnnData, ad.AnnData]:
    if not virtual_path.exists():
        raise FileNotFoundError(f"Virtual tissue file not found: {virtual_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Generation metadata file not found: {meta_path}")

    virtual = safe_read_h5ad(virtual_path)
    meta = safe_read_h5ad(meta_path)

    if "valid" in virtual.obs.columns:
        valid = virtual.obs["valid"]
        if valid.dtype == bool:
            mask = valid.to_numpy(dtype=bool)
        elif np.issubdtype(valid.dtype, np.number):
            mask = valid.to_numpy(dtype=float) != 0
        else:
            text = valid.astype(str).str.strip().str.lower()
            mask = text.isin(["true", "1", "yes", "y", "t"]).to_numpy()
        virtual = virtual[mask].copy()

    return virtual, meta


def original_h5ad_path(sample_id: int) -> Path:
    return ROOT / "h5ad" / f"{GROUP_ID}_{sample_id}.h5ad"


def load_original_gene_maps(gene: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Load native coordinates + original measured reads for Figure 1.

    Figure 1 intentionally bypasses the generation metadata. Each panel comes
    directly from the original per-section H5AD file and uses obsm['spatial'].
    """
    records: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for sid in SAMPLE_IDS:
        path = original_h5ad_path(sid)
        if not path.exists():
            raise FileNotFoundError(f"Original slice H5AD not found: {path}")

        adata = safe_read_h5ad(path)
        coords = get_spatial_coords(
            adata,
            source=f"original slice {sid}",
            spatial_key=COORD_KEY,
        )
        values = extract_gene_values(
            adata,
            gene,
            source=f"original slice {sid}",
            apply_log1p=True,
        )
        if len(coords) != len(values):
            raise ValueError(
                f"Original slice {sid}: coordinates ({len(coords)}) and gene values "
                f"({len(values)}) have inconsistent lengths."
            )
        records[int(sid)] = (coords, values)
        print(f"[LOAD] original slice {sid}: {path} | n={len(coords)}")

    return records


def get_spatial_coords(
    adata: ad.AnnData,
    *,
    source: str,
    spatial_key: str = "spatial",
) -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"{source}: obsm[{spatial_key!r}] not found. "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"{source}: invalid spatial shape {coords.shape}")
    coords = coords[:, :2]
    if coords.shape[0] != adata.n_obs:
        raise ValueError(
            f"{source}: coordinate rows ({coords.shape[0]}) do not match n_obs ({adata.n_obs})."
        )
    if not np.isfinite(coords).all():
        raise ValueError(f"{source}: coordinates contain NaN or inf")
    return coords


def extract_gene_values(
    adata: ad.AnnData,
    gene: str,
    *,
    source: str,
    apply_log1p: bool = True,
) -> np.ndarray:
    var_names = pd.Index(adata.var_names.astype(str))
    if gene not in var_names:
        matches = var_names[var_names.str.upper() == gene.upper()].tolist()
        if len(matches) == 1:
            gene = matches[0]
        else:
            raise KeyError(f"{source}: gene {gene!r} not found.")

    idx = int(var_names.get_loc(gene))
    column = adata.X[:, idx]
    if sparse.issparse(column):
        values = column.toarray().reshape(-1)
    else:
        values = np.asarray(column).reshape(-1)

    values = values.astype(float, copy=False)
    values[~np.isfinite(values)] = 0.0
    if apply_log1p:
        values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
    return values


# ============================================================
# ANCHOR DETECTION
# ============================================================

def _decode_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if values.dtype.kind == "S":
        return values.astype(str)
    if values.dtype == object:
        out = []
        for value in values:
            out.append(value.decode("utf-8") if isinstance(value, bytes) else value)
        return np.asarray(out, dtype=object)
    return values


def canonical_category(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return "NA"
    try:
        numeric = float(text)
        if np.isfinite(numeric):
            return str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"
    except Exception:
        pass
    return text


def natural_sort_key(value: str):
    parts = re.split(r"(\d+(?:\.\d+)?)", str(value))
    key = []
    for part in parts:
        if not part:
            continue
        try:
            key.append((0, float(part)))
        except ValueError:
            key.append((1, part.lower()))
    return key


def resolve_anchor_key(
    meta: ad.AnnData,
    anchor_key: Optional[str] = ANCHOR_KEY,
    expected_n_anchors: int = EXPECTED_N_ANCHORS,
) -> str:
    if anchor_key is not None:
        if anchor_key not in meta.obs.columns:
            raise KeyError(
                f"Requested ANCHOR_KEY={anchor_key!r} not found. "
                f"Available: {list(meta.obs.columns)}"
            )
        n_unique = meta.obs[anchor_key].dropna().astype(str).nunique()
        if n_unique != expected_n_anchors:
            raise ValueError(
                f"ANCHOR_KEY={anchor_key!r} has {n_unique} unique values; "
                f"expected {expected_n_anchors}."
            )
        return anchor_key

    for key in ANCHOR_KEY_CANDIDATES:
        if key not in meta.obs.columns:
            continue
        n_unique = meta.obs[key].dropna().astype(str).nunique()
        if n_unique == expected_n_anchors:
            print(f"[INFO] Auto-detected anchor key: {key}")
            return key

    # Robust fallback: if a non-candidate obs column has exactly the expected
    # number of levels, use it only when the choice is unambiguous.
    fallback = []
    for key in meta.obs.columns:
        try:
            n_unique = meta.obs[key].dropna().astype(str).nunique()
        except Exception:
            continue
        if n_unique == expected_n_anchors:
            fallback.append(str(key))

    if len(fallback) == 1:
        print(f"[INFO] Auto-detected four-level anchor key: {fallback[0]}")
        return fallback[0]

    if len(fallback) > 1:
        raise ValueError(
            "Multiple metadata obs columns contain exactly four levels: "
            f"{fallback}. Set ANCHOR_KEY explicitly."
        )

    raise KeyError(
        "Could not detect a metadata obs column identifying four anchors. "
        f"Available meta.obs columns: {list(meta.obs.columns)}"
    )


def get_anchor_labels(
    meta: ad.AnnData,
    anchor_key: str,
    *,
    manual_order: Optional[Sequence[str]] = ANCHOR_ORDER,
) -> Tuple[np.ndarray, List[str]]:
    raw = _decode_array(meta.obs[anchor_key].to_numpy())
    labels = np.asarray([canonical_category(x) for x in raw], dtype=str)
    unique = sorted([x for x in sorted(set(labels.tolist()), key=natural_sort_key) if x != "NA"], key=natural_sort_key)
    if manual_order is not None:
        manual = [canonical_category(x) for x in manual_order]
        if set(manual) != set(unique):
            raise ValueError(f"ANCHOR_ORDER mismatch. Detected: {unique}; requested: {manual}")
        unique = list(manual)
    if len(unique) != EXPECTED_N_ANCHORS:
        raise ValueError(f"Expected {EXPECTED_N_ANCHORS} anchors, found {len(unique)}: {unique}")
    print(f"[INFO] Anchor order ({anchor_key}): {unique}")
    return labels, unique


# ============================================================
# PLOTTING HELPERS
# ============================================================

def clean_spatial_axis(ax: plt.Axes, *, invert_y: bool = True) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if invert_y:
        ax.invert_yaxis()


def combined_limits(
    arrays: Iterable[np.ndarray],
    *,
    pad_fraction: float = 0.035,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    coords = np.vstack([np.asarray(x, dtype=float)[:, :2] for x in arrays if x is not None and len(x) > 0])
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0)
    pad = spans * pad_fraction
    return (
        (float(mins[0] - pad[0]), float(maxs[0] + pad[0])),
        (float(mins[1] - pad[1]), float(maxs[1] + pad[1])),
    )


def apply_limits(
    ax: plt.Axes,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    invert_y: bool = True,
) -> None:
    ax.set_xlim(*xlim)
    if invert_y:
        ax.set_ylim(ylim[1], ylim[0])
    else:
        ax.set_ylim(*ylim)


def make_monochrome_cmap(base_color: str, name: str) -> LinearSegmentedColormap:
    """High-contrast sequential ramp that preserves the current slice hue.

    The previous custom ramps stayed near white for too much of the numerical
    range. Here the hue enters rapidly after zero, while the upper range becomes
    substantially darker. This keeps the same slice identity but makes spatial
    expression structure much easier to see.
    """
    rgb = np.asarray(to_rgb(base_color), dtype=float)
    low = 1.0 - (1.0 - rgb) * 0.72
    dark = np.clip(rgb * 0.62, 0.0, 1.0)
    deep = np.clip(rgb * 0.42, 0.0, 1.0)

    return LinearSegmentedColormap.from_list(
        name,
        [
            (0.00, (0.985, 0.985, 0.985)),
            (0.035, tuple(low)),
            (0.18, tuple(rgb)),
            (0.58, tuple(dark)),
            (1.00, tuple(deep)),
        ],
        N=256,
    )


def full_range_norm(values: np.ndarray) -> Normalize:
    """Normalize one spatial gene map over its full observed log1p range.

    This follows the separate DLPFC spatial-map workflow: vmin is fixed at 0
    for non-negative log1p expression and vmax is the actual maximum value.
    No percentile clipping or min-max remapping is applied.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0, clip=True)

    vmin = 0.0 if np.nanmin(finite) >= 0 else float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if (not np.isfinite(vmax)) or vmax <= vmin:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax, clip=True)


def plot_expression_map(
    ax: plt.Axes,
    coords: np.ndarray,
    values: np.ndarray,
    *,
    norm: Normalize,
    cmap: mpl.colors.Colormap,
    point_size: float,
) -> None:
    """Plot log1p expression directly with fully opaque points.

    Low-expression observations are not faded by alpha. Points are sorted by
    expression so higher-expression observations are drawn last and remain
    visible, matching the separate spatial-gene-map plotting logic.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1]) & np.isfinite(values)
    coords = coords[finite]
    values = values[finite]
    order = np.argsort(values, kind="mergesort")

    ax.scatter(
        coords[order, 0],
        coords[order, 1],
        s=point_size,
        c=values[order],
        cmap=cmap,
        norm=norm,
        alpha=1.0,
        linewidths=0,
        rasterized=True,
    )


# ============================================================
# FIGURE BUILDERS
# ============================================================

def plot_figure1_four_slices(
    *,
    original_maps: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    outdir: Optional[Path],
    stem: str,
    invert_y: bool = INVERT_Y,
) -> Optional[Dict[str, Path]]:
    """Plot the four ORIGINAL measured sections in a 2 x 2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=FIG1_SIZE)
    axes = axes.ravel()

    all_coords = [original_maps[int(sid)][0] for sid in SAMPLE_IDS]
    xlim, ylim = combined_limits(all_coords, pad_fraction=0.035)
    slice_cmaps = [
        make_monochrome_cmap(color, f"slice_{sid}_cmap")
        for sid, color in zip(SAMPLE_IDS, SLICE_BASE_COLORS)
    ]

    for ax, sid, cmap in zip(axes, SAMPLE_IDS, slice_cmaps):
        slice_coords, slice_values = original_maps[int(sid)]
        # Same logic as the separate DLPFC spatial maps: each native section
        # owns its own [0, observed maximum] scale.
        norm = full_range_norm(slice_values)

        plot_expression_map(
            ax,
            slice_coords,
            slice_values,
            norm=norm,
            cmap=cmap,
            point_size=FIG1_POINT_SIZE,
        )
        ax.set_title(f"slice{sid}", pad=4)
        clean_spatial_axis(ax, invert_y=invert_y)
        apply_limits(ax, xlim, ylim, invert_y=invert_y)

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.018)
        cbar.set_label("log1p expression", fontsize=7.2)
        cbar.ax.tick_params(labelsize=6.8, length=2)

    fig.tight_layout()
    return save_or_show(fig, outdir=outdir, stem=stem)


def plot_figure2_combined_evidence_plane(
    *,
    coords: np.ndarray,
    values: np.ndarray,
    anchor_labels: np.ndarray,
    anchor_order: Sequence[str],
    outdir: Optional[Path],
    stem: str,
    invert_y: bool = INVERT_Y,
) -> Optional[Dict[str, Path]]:
    fig, ax = plt.subplots(1, 1, figsize=FIG2_SIZE)
    xlim, ylim = combined_limits([coords], pad_fraction=0.035)
    slice_cmaps = [make_monochrome_cmap(color, f"slice_{i}_cmap_overlay") for i, color in enumerate(SLICE_BASE_COLORS)]

    legend_handles: List[Line2D] = []
    for i, (anchor_label, cmap, base_color) in enumerate(zip(anchor_order, slice_cmaps, SLICE_BASE_COLORS)):
        mask = anchor_labels == str(anchor_label)
        slice_coords = coords[mask]
        slice_values = values[mask]
        norm = full_range_norm(slice_values)
        plot_expression_map(
            ax,
            slice_coords,
            slice_values,
            norm=norm,
            cmap=cmap,
            point_size=FIG2_POINT_SIZE,
        )
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=base_color, markeredgecolor="none",
                markersize=5.8, label=f"slice{anchor_label}"
            )
        )

    clean_spatial_axis(ax, invert_y=invert_y)
    apply_limits(ax, xlim, ylim, invert_y=invert_y)
    ax.legend(
        handles=legend_handles,
        title="Source slices",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.35,
        labelspacing=0.35,
    )
    fig.tight_layout()
    return save_or_show(fig, outdir=outdir, stem=stem)


def plot_figure3_generated_gene_map(
    *,
    coords: np.ndarray,
    values: np.ndarray,
    outdir: Optional[Path],
    stem: str,
    invert_y: bool = INVERT_Y,
) -> Optional[Dict[str, Path]]:
    fig, ax = plt.subplots(1, 1, figsize=FIG3_SIZE)
    xlim, ylim = combined_limits([coords], pad_fraction=0.035)
    cmap = make_monochrome_cmap(GENERATED_BASE_COLOR, "generated_cmap")
    norm = full_range_norm(values)

    plot_expression_map(
        ax,
        coords,
        values,
        norm=norm,
        cmap=cmap,
        point_size=FIG3_POINT_SIZE,
    )
    clean_spatial_axis(ax, invert_y=invert_y)
    apply_limits(ax, xlim, ylim, invert_y=invert_y)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.018)
    cbar.set_label("log1p expression", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.8, length=2)

    fig.tight_layout()
    return save_or_show(fig, outdir=outdir, stem=stem)


# ============================================================
# MAIN WORKFLOW
# ============================================================

def run_population_2d_visualization(
    *,
    virtual_path: Path = VIRTUAL_PATH,
    meta_path: Path = META_PATH,
    gene: str = MARKER_GENE,
    outdir: Optional[Path] = OUT_DIR,
    invert_y: bool = INVERT_Y,
) -> Dict[str, Optional[Dict[str, Path]]]:
    set_publication_style()

    virtual, meta = read_inputs(virtual_path=virtual_path, meta_path=meta_path)
    original_maps = load_original_gene_maps(gene)

    evidence_coords = get_spatial_coords(meta, source="generation metadata")
    generated_coords = get_spatial_coords(virtual, source="generated virtual tissue")

    evidence_values = extract_gene_values(meta, gene, source="generation metadata", apply_log1p=True)
    generated_values = extract_gene_values(virtual, gene, source="generated virtual tissue", apply_log1p=True)

    if len(evidence_coords) != len(evidence_values):
        raise ValueError("Evidence coordinates and gene values have inconsistent lengths.")
    if len(generated_coords) != len(generated_values):
        raise ValueError("Generated coordinates and gene values have inconsistent lengths.")

    anchor_key = resolve_anchor_key(meta, anchor_key=ANCHOR_KEY)
    anchor_labels, anchor_order = get_anchor_labels(meta, anchor_key, manual_order=ANCHOR_ORDER)

    stem_base = Path(virtual_path).stem

    fig1 = plot_figure1_four_slices(
        original_maps=original_maps,
        outdir=outdir,
        stem=f"{stem_base}_figure1_original_four_slices",
        invert_y=invert_y,
    )

    fig2 = plot_figure2_combined_evidence_plane(
        coords=evidence_coords,
        values=evidence_values,
        anchor_labels=anchor_labels,
        anchor_order=anchor_order,
        outdir=outdir,
        stem=f"{stem_base}_figure2_combined_evidence_plane",
        invert_y=invert_y,
    )

    fig3 = plot_figure3_generated_gene_map(
        coords=generated_coords,
        values=generated_values,
        outdir=outdir,
        stem=f"{stem_base}_figure3_generated_gene_map",
        invert_y=invert_y,
    )

    print("\n[SUMMARY]")
    print(f"Virtual tissue: {virtual_path}")
    print(f"Generation metadata: {meta_path}")
    print(f"Marker gene: {gene}")
    print(f"Anchor key: {anchor_key}")
    print(f"Anchor order: {anchor_order}")
    print(f"Original slices loaded: {list(SAMPLE_IDS)}")
    print(f"Total evidence points: {len(evidence_coords)}")
    print(f"Generated points: {len(generated_coords)}")

    return {"figure1": fig1, "figure2": fig2, "figure3": fig3}


def main() -> None:
    run_population_2d_visualization()


if __name__ == "__main__":
    main()
