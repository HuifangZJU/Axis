#!/usr/bin/env python3
"""
Streamlined visualization of the virtual-tissue generation process.

This script generates two figures:

Figure 1
--------
A. User-defined Visium template/support
B. Collected gene-expression evidence overlaid on the template
C. Collected categorical attribute evidence overlaid on the template

Figure 2
--------
A. Generated virtual gene-expression map
B. Generated virtual categorical attribute map

Design notes
------------
1. No CLI arguments are used. Edit the default values below when needed.
2. Each figure is handled by `save_or_show(...)`.
3. By default, `OUT_DIR = None`, so figures are shown interactively.
   Set `OUT_DIR` manually to save PNG/PDF outputs.
4. For the virtual tissue, only obs['valid'] points are kept when available.
5. Gene maps use log1p-transformed expression for visualization.
"""

from __future__ import annotations
from axis.settings import data_path

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy import sparse

VIRTUAL_PATH = Path(
    data_path("SCC/huifang/virtual_tissue/"
    "scc_z_group2_grid40_spatial3_gene_weight/query/"
    "virtual_scc_z_10.000.h5ad")
)

META_PATH = Path(
    data_path("SCC/huifang/virtual_tissue/"
    "scc_z_group2_grid40_spatial3_gene_weight/query/"
    "virtual_scc_z_10.000_generation_meta.h5ad")
)

MARKER_GENE = "KRT1"
ATTRIBUTE_NAME = "original_clusters"
OUT_DIR: Optional[Path] = None
TEMPLATE_POINT_SIZE = 60.0
EVIDENCE_POINT_SIZE = 65.0
GENERATED_POINT_SIZE = 60.0
UPPER_PERCENTILE = 99.5
INVERT_Y = True


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
    outdir: Optional[Path] = None,
    stem: Optional[str] = None,
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

    if stem is None:
        raise ValueError("`stem` must be provided when `outdir` is not None.")

    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)

    if close:
        plt.close(fig)

    print(f"[OK] saved PNG: {png_path}")
    print(f"[OK] saved PDF: {pdf_path}")

    return {"png": png_path, "pdf": pdf_path}


def read_inputs(
    virtual_path: Path,
    meta_path: Path,
) -> Tuple[ad.AnnData, ad.AnnData]:
    if not Path(virtual_path).exists():
        raise FileNotFoundError(f"Virtual tissue file not found: {virtual_path}")
    if not Path(meta_path).exists():
        raise FileNotFoundError(f"Generation metadata file not found: {meta_path}")

    virtual = ad.read_h5ad(virtual_path)
    meta = ad.read_h5ad(meta_path)

    if "valid" in virtual.obs.columns:
        virtual = virtual[virtual.obs["valid"]].copy()

    return virtual, meta


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
        raise ValueError(
            f"{source}: expected spatial coordinates with shape [N, 2+], "
            f"received {coords.shape}."
        )

    coords = coords[:, :2]

    if coords.shape[0] != adata.n_obs:
        raise ValueError(
            f"{source}: coordinate rows ({coords.shape[0]}) do not match "
            f"n_obs ({adata.n_obs})."
        )

    if not np.isfinite(coords).all():
        raise ValueError(f"{source}: spatial coordinates contain NaN or infinity.")

    return coords


def get_grid_points(meta: ad.AnnData) -> np.ndarray:
    if "grid_pts" not in meta.uns:
        raise KeyError(
            "Generation metadata does not contain uns['grid_pts']. "
            f"Available uns keys: {list(meta.uns.keys())}"
        )

    grid_pts = np.asarray(meta.uns["grid_pts"], dtype=float)
    if grid_pts.ndim != 2 or grid_pts.shape[1] < 2:
        raise ValueError(
            f"uns['grid_pts'] must have shape [M, 2+], received {grid_pts.shape}."
        )
    return grid_pts[:, :2]


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
            raise KeyError(
                f"{source}: gene {gene!r} not found. "
                f"Number of available genes: {adata.n_vars}."
            )

    gene_index = int(var_names.get_loc(gene))
    column = adata.X[:, gene_index]

    if sparse.issparse(column):
        values = column.toarray().reshape(-1)
    else:
        values = np.asarray(column).reshape(-1)

    values = values.astype(float, copy=False)
    values[~np.isfinite(values)] = 0.0

    if apply_log1p:
        values = np.log1p(np.clip(values, a_min=0.0, a_max=None))

    return values


def _decode_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape(-1)

    if values.dtype.kind == "S":
        return values.astype(str)

    if values.dtype == object:
        decoded = []
        for value in values:
            if isinstance(value, bytes):
                decoded.append(value.decode("utf-8"))
            else:
                decoded.append(value)
        return np.asarray(decoded, dtype=object)

    return values


def _read_attribute_names(adata: ad.AnnData) -> Optional[List[str]]:
    for key in ("attribute_names", "attrs_names", "attr_names"):
        if key in adata.uns:
            return [str(x) for x in np.asarray(adata.uns[key]).reshape(-1)]

    for key in ("attribute_schema_json", "attr_schema_json"):
        if key in adata.uns:
            raw = adata.uns[key]
            try:
                schema = json.loads(str(raw))
            except Exception:
                continue

            if isinstance(schema, dict):
                return [str(x) for x in schema.keys()]

            if isinstance(schema, list):
                names: List[str] = []
                for i, item in enumerate(schema):
                    if isinstance(item, str):
                        names.append(item)
                    elif isinstance(item, dict) and "name" in item:
                        names.append(str(item["name"]))
                    else:
                        names.append(f"attribute_{i}")
                return names

    return None


def extract_attribute_values(
    adata: ad.AnnData,
    attribute: str,
    *,
    source: str,
) -> np.ndarray:
    candidate_obs_columns = (
        attribute,
        f"attr__{attribute}",
        f"attribute__{attribute}",
    )

    for column in candidate_obs_columns:
        if column in adata.obs:
            return _decode_array(adata.obs[column].to_numpy())

    if attribute in adata.obsm:
        values = np.asarray(adata.obsm[attribute])

        if values.ndim == 1:
            return _decode_array(values)

        if values.ndim == 2 and values.shape[1] == 1:
            return _decode_array(values[:, 0])

        raise ValueError(
            f"{source}: obsm[{attribute!r}] has shape {values.shape}; "
            "a categorical attribute must be one-dimensional or N x 1."
        )

    for matrix_key in ("attributes", "attrs"):
        if matrix_key not in adata.obsm:
            continue

        matrix = np.asarray(adata.obsm[matrix_key])
        names = _read_attribute_names(adata)

        if names is None:
            raise KeyError(
                f"{source}: obsm[{matrix_key!r}] exists, but no attribute names "
                "were found in adata.uns."
            )

        if attribute not in names:
            continue

        column_index = names.index(attribute)

        if matrix.ndim != 2 or column_index >= matrix.shape[1]:
            raise ValueError(
                f"{source}: obsm[{matrix_key!r}] has shape {matrix.shape}, "
                f"but attribute {attribute!r} maps to column {column_index}."
            )

        return _decode_array(matrix[:, column_index])

    raise KeyError(
        f"{source}: attribute {attribute!r} was not found.\n"
        f"obs columns: {list(adata.obs.columns)}\n"
        f"obsm keys: {list(adata.obsm.keys())}\n"
        f"uns keys related to attributes: "
        f"{[k for k in adata.uns.keys() if 'attr' in str(k).lower()]}"
    )


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
            if numeric.is_integer():
                return str(int(numeric))
            return f"{numeric:g}"
    except (TypeError, ValueError):
        pass

    return text


def category_sort_key(value: str) -> Tuple[int, float, str]:
    try:
        return (0, float(value), value)
    except (TypeError, ValueError):
        return (1, math.inf, value)


def build_category_color_map(
    evidence_values: Sequence[Any],
    generated_values: Sequence[Any],
    *,
    cmap_name: str = "tab20",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[float, float, float, float]]]:
    evidence = np.asarray([canonical_category(x) for x in evidence_values], dtype=str)
    generated = np.asarray([canonical_category(x) for x in generated_values], dtype=str)

    categories = sorted(
        set(evidence.tolist()) | set(generated.tolist()),
        key=category_sort_key,
    )

    if len(categories) > 20 and cmap_name == "tab20":
        raise ValueError(
            f"The attribute contains {len(categories)} categories, but tab20 "
            "provides only 20 distinct colors."
        )

    cmap = plt.get_cmap(cmap_name, max(len(categories), 1))
    color_map = {category: cmap(i) for i, category in enumerate(categories)}
    return evidence, generated, color_map


def make_shared_gene_norm(
    evidence_values: np.ndarray,
    generated_values: np.ndarray,
    *,
    upper_percentile: float = 99.5,
) -> Normalize:
    combined = np.concatenate(
        [
            np.asarray(evidence_values, dtype=float).reshape(-1),
            np.asarray(generated_values, dtype=float).reshape(-1),
        ]
    )

    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)

    vmin = 0.0 if np.nanmin(finite) >= 0 else float(np.nanmin(finite))
    vmax = float(np.nanpercentile(finite, upper_percentile))

    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    return Normalize(vmin=vmin, vmax=vmax, clip=True)


def expression_rgba(
    values: np.ndarray,
    *,
    norm: Normalize,
    cmap_name: str = "Reds",
    alpha_min: float = 0.04,
    alpha_max: float = 0.95,
    alpha_gamma: float = 0.75,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scaled = np.clip(norm(values), 0.0, 1.0)

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(scaled)

    alpha = alpha_min + (alpha_max - alpha_min) * np.power(scaled, alpha_gamma)
    rgba[:, 3] = alpha
    return rgba


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.08,
    y: float = 1.03,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def clean_spatial_axis(
    ax: plt.Axes,
    *,
    invert_y: bool = True,
) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if invert_y:
        ax.invert_yaxis()


def combined_limits(
    arrays: Iterable[np.ndarray],
    *,
    pad_fraction: float = 0.035,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    coords = np.vstack(
        [
            np.asarray(x, dtype=float)[:, :2]
            for x in arrays
            if x is not None and len(x) > 0
        ]
    )

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


def plot_template_grid(
    ax: plt.Axes,
    grid_pts: np.ndarray,
    *,
    point_size: float,
) -> None:
    ax.scatter(
        grid_pts[:, 0],
        grid_pts[:, 1],
        s=point_size,
        facecolors="none",
        edgecolors="#A5A5A5",
        linewidths=0.55,
        rasterized=True,
    )


def plot_expression_map(
    ax: plt.Axes,
    coords: np.ndarray,
    values: np.ndarray,
    *,
    norm: Normalize,
    point_size: float,
    cmap_name: str = "Reds",
    alpha_min: float = 0.04,
    alpha_max: float = 0.75,
    alpha_gamma: float = 0.75,
) -> None:
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)

    order = np.argsort(values)
    rgba = expression_rgba(
        values,
        norm=norm,
        cmap_name=cmap_name,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_gamma=alpha_gamma,
    )

    ax.scatter(
        coords[order, 0],
        coords[order, 1],
        s=point_size,
        c=rgba[order],
        linewidths=0,
        rasterized=True,
    )


def plot_attribute_map(
    ax: plt.Axes,
    coords: np.ndarray,
    categories: np.ndarray,
    *,
    color_map: Mapping[str, Tuple[float, float, float, float]],
    point_size: float,
) -> None:
    colors = [color_map[str(category)] for category in categories]

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=point_size,
        c=colors,
        alpha=0.6,
        linewidths=0,
        rasterized=True,
    )


def make_attribute_legend(
    color_map: Mapping[str, Tuple[float, float, float, float]],
    *,
    marker_size: float = 5.0,
) -> List[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=marker_size,
            label=str(category),
        )
        for category, color in color_map.items()
    ]


def plot_figure1_generation_inputs(
    *,
    grid_pts: np.ndarray,
    evidence_coords: np.ndarray,
    evidence_gene_values: np.ndarray,
    evidence_categories: np.ndarray,
    gene_norm: Normalize,
    category_color_map: Mapping[str, Tuple[float, float, float, float]],
    gene: str,
    attribute: str,
    outdir: Optional[Path] = None,
    stem: str = "figure1_generation_inputs",
    template_point_size: float = 34.0,
    evidence_point_size: float = 8.0,
    invert_y: bool = True,
) -> Optional[Dict[str, Path]]:
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.75), constrained_layout=True)

    xlim, ylim = combined_limits([grid_pts, evidence_coords], pad_fraction=0.035)

    ax = axes[0]
    plot_template_grid(ax, grid_pts, point_size=template_point_size)
    ax.set_title("User-defined Visium support", pad=5)
    panel_label(ax, "A")
    clean_spatial_axis(ax, invert_y=invert_y)
    apply_limits(ax, xlim, ylim, invert_y=invert_y)

    ax = axes[1]
    plot_template_grid(ax, grid_pts, point_size=template_point_size)
    plot_expression_map(
        ax,
        evidence_coords,
        evidence_gene_values,
        norm=gene_norm,
        point_size=evidence_point_size,
    )
    ax.set_title(f"Collected gene evidence: log1p({gene})", pad=5)
    panel_label(ax, "B")
    clean_spatial_axis(ax, invert_y=invert_y)
    apply_limits(ax, xlim, ylim, invert_y=invert_y)

    sm = mpl.cm.ScalarMappable(norm=gene_norm, cmap="Reds")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.015)
    cbar.set_label("log1p expression", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7, length=2)

    ax = axes[2]
    plot_template_grid(ax, grid_pts, point_size=template_point_size)
    plot_attribute_map(
        ax,
        evidence_coords,
        evidence_categories,
        color_map=category_color_map,
        point_size=evidence_point_size,
    )
    ax.set_title(f"Collected attribute evidence: {attribute}", pad=5)
    panel_label(ax, "C")
    clean_spatial_axis(ax, invert_y=invert_y)
    apply_limits(ax, xlim, ylim, invert_y=invert_y)

    legend_handles = make_attribute_legend(category_color_map)
    ax.legend(
        handles=legend_handles,
        title=attribute,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        borderaxespad=0,
        handletextpad=0.35,
        labelspacing=0.35,
        ncol=1 ,
    )

    return save_or_show(fig, outdir=outdir, stem=stem)


def plot_figure2_generated_outputs(
    *,
    generated_coords: np.ndarray,
    generated_gene_values: np.ndarray,
    generated_categories: np.ndarray,
    gene_norm: Normalize,
    category_color_map: Mapping[str, Tuple[float, float, float, float]],
    gene: str,
    attribute: str,
    outdir: Optional[Path] = None,
    stem: str = "figure2_generated_outputs",
    generated_point_size: float = 18.0,
    invert_y: bool = True,
) -> Optional[Dict[str, Path]]:
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.85), constrained_layout=True)

    xlim, ylim = combined_limits([generated_coords], pad_fraction=0.035)

    ax = axes[0]
    plot_expression_map(
        ax,
        generated_coords,
        generated_gene_values,
        norm=gene_norm,
        point_size=generated_point_size,
    )
    ax.set_title(f"Generated virtual gene map: log1p({gene})", pad=5)
    panel_label(ax, "A")
    clean_spatial_axis(ax, invert_y=invert_y)
    apply_limits(ax, xlim, ylim, invert_y=invert_y)

    sm = mpl.cm.ScalarMappable(norm=gene_norm, cmap="Reds")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.015)
    cbar.set_label("log1p expression", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7, length=2)

    ax = axes[1]
    plot_attribute_map(
        ax,
        generated_coords,
        generated_categories,
        color_map=category_color_map,
        point_size=generated_point_size,
    )
    ax.set_title(f"Generated virtual attribute map: {attribute}", pad=5)
    panel_label(ax, "B")
    clean_spatial_axis(ax, invert_y=invert_y)
    apply_limits(ax, xlim, ylim, invert_y=invert_y)

    legend_handles = make_attribute_legend(category_color_map)
    ax.legend(
        handles=legend_handles,
        title=attribute,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        borderaxespad=0,
        handletextpad=0.35,
        labelspacing=0.35,
        ncol=1 if len(legend_handles) <= 10 else 2,
    )

    return save_or_show(fig, outdir=outdir, stem=stem)


def run_generation_process_visualization(
    *,
    virtual_path: Path = VIRTUAL_PATH,
    meta_path: Path = META_PATH,
    gene: str = MARKER_GENE,
    attribute: str = ATTRIBUTE_NAME,
    outdir: Optional[Path] = OUT_DIR,
    template_point_size: float = TEMPLATE_POINT_SIZE,
    evidence_point_size: float = EVIDENCE_POINT_SIZE,
    generated_point_size: float = GENERATED_POINT_SIZE,
    upper_percentile: float = UPPER_PERCENTILE,
    invert_y: bool = INVERT_Y,
) -> Dict[str, Optional[Dict[str, Path]]]:
    set_publication_style()

    virtual, meta = read_inputs(virtual_path=virtual_path, meta_path=meta_path)

    grid_pts = get_grid_points(meta)
    evidence_coords = get_spatial_coords(meta, source="generation metadata")
    generated_coords = get_spatial_coords(virtual, source="generated virtual tissue")

    evidence_gene_values = extract_gene_values(meta, gene, source="generation metadata", apply_log1p=True)
    generated_gene_values = extract_gene_values(virtual, gene, source="generated virtual tissue", apply_log1p=True)

    evidence_attribute_raw = extract_attribute_values(meta, attribute, source="generation metadata")
    generated_attribute_raw = extract_attribute_values(virtual, attribute, source="generated virtual tissue")

    if len(evidence_attribute_raw) != len(evidence_coords):
        raise ValueError(
            "Evidence attribute length does not match evidence coordinates: "
            f"{len(evidence_attribute_raw)} versus {len(evidence_coords)}."
        )

    if len(generated_attribute_raw) != len(generated_coords):
        raise ValueError(
            "Generated attribute length does not match generated coordinates: "
            f"{len(generated_attribute_raw)} versus {len(generated_coords)}."
        )

    evidence_categories, generated_categories, category_color_map = build_category_color_map(
        evidence_attribute_raw,
        generated_attribute_raw,
        cmap_name="tab20",
    )

    gene_norm = make_shared_gene_norm(
        evidence_gene_values,
        generated_gene_values,
        upper_percentile=upper_percentile,
    )

    stem_base = Path(virtual_path).stem

    fig1_paths = plot_figure1_generation_inputs(
        grid_pts=grid_pts,
        evidence_coords=evidence_coords,
        evidence_gene_values=evidence_gene_values,
        evidence_categories=evidence_categories,
        gene_norm=gene_norm,
        category_color_map=category_color_map,
        gene=gene,
        attribute=attribute,
        outdir=outdir,
        stem=f"{stem_base}_figure1_generation_inputs",
        template_point_size=template_point_size,
        evidence_point_size=evidence_point_size,
        invert_y=invert_y,
    )

    fig2_paths = plot_figure2_generated_outputs(
        generated_coords=generated_coords,
        generated_gene_values=generated_gene_values,
        generated_categories=generated_categories,
        gene_norm=gene_norm,
        category_color_map=category_color_map,
        gene=gene,
        attribute=attribute,
        outdir=outdir,
        stem=f"{stem_base}_figure2_generated_outputs",
        generated_point_size=generated_point_size,
        invert_y=invert_y,
    )

    return {"figure1": fig1_paths, "figure2": fig2_paths}


def main() -> None:
    run_generation_process_visualization()


if __name__ == "__main__":
    main()
