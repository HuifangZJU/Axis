#!/usr/bin/env python3
"""
3D visualization of SCC virtual-tissue generation on a user-defined spatial support.

This script produces four figures while preserving the plotting style used in the
previous SCC 3D generation/transport script.

Figure 1
--------
Stack 3 real SCC slices. Supports:
    - gene mode  : KRT1 in Reds colormap
    - domain mode: original_clusters in seaborn tab10 palette

Figure 2
--------
Stack:
    - real anchor slice A
    - middle virtual support plate (Visium grid only; no transported evidence)
    - real anchor slice B

Figure 3
--------
Evidence-transport field:
    - real anchor slice A
    - middle virtual support plate + projected evidence
    - real anchor slice B
    - lifted smooth curves from each anchor to the middle support

Figure 4
--------
Stack:
    - real anchor slice A
    - final generated virtual tissue slice
    - real anchor slice B

Notes
-----
1. No CLI arguments are used. Edit the constants below if needed.
2. By default all figures are shown (`OUT_DIR = None`).
3. Real anchors and query-axis information are read from the virtual slice
   and generation metadata when available.
4. The generation metadata is expected to store:
      - projected evidence coordinates in obsm["spatial"]
      - the Visium query support in uns["grid_pts"]
5. The final generated virtual tissue is read from VIRTUAL_PATH.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import sparse


# =============================================================================
# Defaults
# =============================================================================

ROOT = Path("/media/huifang/data/registration/SCC/huifang")

VIRTUAL_PATH = ROOT / "virtual_tissue" / "scc_z_group2_grid40_spatial3_gene_weight" / "query" / "virtual_scc_z_10.000.h5ad"
META_PATH = ROOT / "virtual_tissue" / "scc_z_group2_grid40_spatial3_gene_weight" / "query" / "virtual_scc_z_10.000_generation_meta.h5ad"

GROUP_ID = 2
H5AD_SUBDIR = "h5ad"
H5AD_TEMPLATE = "{group_id}_{sample_id}.h5ad"
REAL_COORD_KEY = "spatial_image_coor"

DISPLAY_MODE = "domain"        # "gene" or "domain"
MARKER_GENE = "KRT1"
ATTRIBUTE_NAME = "original_clusters"

# Optional override. If None, detect the available SCC sample ids for the group
# and use the first three sorted slices.
REAL_STACK_SAMPLE_IDS: Optional[Sequence[Any]] = None

OUT_DIR: Optional[Path] = None

# 3D appearance
DEPTH_SCALE = 35.0
ELEV = 15
AZIM = -145
FIGSIZE = (8, 7.9)

# Planes and points
GRID_POINT_SIZE = 15.0
EVIDENCE_POINT_SIZE = 25.0
REAL_POINT_SIZE = 15
VIRTUAL_POINT_SIZE = 15

REAL_POINT_ALPHA = 1
EVIDENCE_POINT_ALPHA = 0.65
VIRTUAL_POINT_ALPHA = 1

VIRTUAL_FRAME_COLOR = "#555555"
VIRTUAL_FRAME_LINEWIDTH = 1.15
VIRTUAL_FRAME_ALPHA = 0.88
VIRTUAL_FRAME_DASH_PATTERN = (0, (4.0, 3.0))

REAL_FRAME_COLOR = "#3D3D3D"
REAL_FRAME_LINEWIDTH = 1.05
REAL_FRAME_ALPHA = 0.72

# Curves
MAX_CURVES_PER_ANCHOR = 550
RANDOM_SEED = 0
CONTROL_SCALE = 0.34
CURVE_LINEWIDTH = 2
CURVE_ALPHA = 0.1
CURVE_COLOR_A = "#4E79A7"
CURVE_COLOR_B = "#D3725B"
ARROW_HEAD_LENGTH_SCALE = 0.5
ARROW_HEAD_WIDTH_SCALE = 0.5
# Color scale
UPPER_PERCENTILE = 99.5
GENE_CMAP = "Reds"
ATTRIBUTE_PALETTE = "tab10"

INVERT_X = True
INVERT_Y = True


# =============================================================================
# Style
# =============================================================================

def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 15,
            "axes.titlesize": 15,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 15.0,
            "ytick.labelsize": 15.0,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 350,
            "figure.dpi": 120,
        }
    )


# =============================================================================
# IO helpers
# =============================================================================

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
    if not virtual_path.exists():
        raise FileNotFoundError(f"Virtual tissue file not found: {virtual_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Generation metadata file not found: {meta_path}")

    virtual = ad.read_h5ad(virtual_path)
    meta = ad.read_h5ad(meta_path)
    return virtual, meta


def parse_h5ad_path(
    root: Path,
    group_id: Any,
    sample_id: Any,
    h5ad_subdir: str = H5AD_SUBDIR,
    h5ad_template: str = H5AD_TEMPLATE,
) -> Path:
    return root / h5ad_subdir / h5ad_template.format(group_id=group_id, sample_id=sample_id)


def load_real_anchor_slice(
    root: Path,
    group_id: Any,
    sample_id: Any,
    *,
    coord_key: str = REAL_COORD_KEY,
) -> Tuple[ad.AnnData, np.ndarray]:
    path = parse_h5ad_path(root, group_id, sample_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing SCC h5ad file: {path}")

    adata = sc.read_h5ad(path)

    if coord_key not in adata.obsm:
        raise KeyError(
            f"obsm[{coord_key!r}] is missing from {path}. "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )

    coords = np.asarray(adata.obsm[coord_key], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"{path}: expected coordinates with shape [N,2+], got {coords.shape}")

    coords = coords[:, :2]
    if coords.shape[0] != adata.n_obs:
        n = min(coords.shape[0], adata.n_obs)
        adata = adata[:n].copy()
        coords = coords[:n]
        print(f"[WARN] {path.name}: trimmed to first {n} observations to match coordinates.")

    return adata, coords


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
            f"{source}: expected spatial coordinates with shape [N,2+], got {coords.shape}."
        )
    coords = coords[:, :2]

    if coords.shape[0] != adata.n_obs:
        raise ValueError(
            f"{source}: coordinate rows ({coords.shape[0]}) do not match n_obs ({adata.n_obs})."
        )

    return coords


def get_grid_points(meta: ad.AnnData) -> np.ndarray:
    if "grid_pts" not in meta.uns:
        raise KeyError(
            "Generation metadata does not contain uns['grid_pts']. "
            f"Available uns keys: {list(meta.uns.keys())}"
        )

    grid_pts = np.asarray(meta.uns["grid_pts"], dtype=float)
    if grid_pts.ndim != 2 or grid_pts.shape[1] < 2:
        raise ValueError(f"uns['grid_pts'] must have shape [M,2+], received {grid_pts.shape}.")
    return grid_pts[:, :2]


# =============================================================================
# Metadata parsing
# =============================================================================

def _tolist(values: Any) -> List[Any]:
    arr = np.asarray(values, dtype=object).reshape(-1)
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        elif isinstance(x, np.generic):
            out.append(x.item())
        else:
            out.append(x)
    return out


def canonical_value(value: Any) -> str:
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
    except Exception:
        pass
    return text


def sort_key(value: Any):
    text = canonical_value(value)
    try:
        return (0, float(text), text)
    except Exception:
        return (1, text)


def get_source_anchor_ids(
    virtual: ad.AnnData,
    meta: ad.AnnData,
    *,
    fallback: Tuple[int, int] = (0, 2),
) -> Tuple[Any, Any]:
    for holder in (meta.uns, virtual.uns):
        if "source_anchor_ids" in holder:
            vals = _tolist(holder["source_anchor_ids"])
            if len(vals) >= 2:
                return vals[0], vals[1]
    return fallback


def get_axis_positions(
    virtual: ad.AnnData,
    meta: ad.AnnData,
    *,
    fallback_query: float = 10.0,
    fallback_sources: Tuple[float, float] = (0.0, 20.0),
) -> Tuple[float, float, float]:
    query = None
    src = None

    for holder in (meta.uns, virtual.uns):
        if query is None and "query_axis_value" in holder:
            try:
                query = float(np.asarray(holder["query_axis_value"]).reshape(-1)[0])
            except Exception:
                pass
        if src is None and "source_axis_positions" in holder:
            try:
                vals = [float(x) for x in _tolist(holder["source_axis_positions"])]
                if len(vals) >= 2:
                    src = (vals[0], vals[1])
            except Exception:
                pass

    if src is None:
        src = fallback_sources
    if query is None:
        query = fallback_query

    return float(src[0]), float(query), float(src[1])


def detect_group_sample_ids(
    root: Path,
    group_id: Any,
    *,
    h5ad_subdir: str = H5AD_SUBDIR,
) -> List[Any]:
    folder = root / h5ad_subdir
    if not folder.exists():
        raise FileNotFoundError(f"H5AD folder not found: {folder}")

    paths = sorted(folder.glob(f"{group_id}_*.h5ad"))
    sample_ids = []
    prefix = f"{group_id}_"
    for path in paths:
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix):]
        try:
            sample_ids.append(int(suffix))
        except Exception:
            sample_ids.append(suffix)

    if len(sample_ids) == 0:
        raise RuntimeError(f"No h5ad files found for group {group_id} in {folder}")
    return sorted(sample_ids, key=sort_key)


def infer_three_real_samples(
    root: Path,
    group_id: Any,
    anchor_a_id: Any,
    anchor_b_id: Any,
    requested: Optional[Sequence[Any]] = REAL_STACK_SAMPLE_IDS,
) -> List[Any]:
    if requested is not None:
        sample_ids = list(requested)
    else:
        sample_ids = detect_group_sample_ids(root, group_id)

    sample_ids = sorted(sample_ids, key=sort_key)

    if len(sample_ids) < 3:
        raise ValueError(
            f"At least three real slices are required for Figure 1, but only "
            f"{len(sample_ids)} were found: {sample_ids}"
        )

    if len(sample_ids) > 3:
        can = [canonical_value(x) for x in sample_ids]
        a = canonical_value(anchor_a_id)
        b = canonical_value(anchor_b_id)
        if a in can and b in can:
            ia = can.index(a)
            ib = can.index(b)
            lo, hi = sorted([ia, ib])
            if hi - lo >= 2:
                mid = sample_ids[(lo + hi) // 2]
                return [sample_ids[lo], mid, sample_ids[hi]]
        return sample_ids[:3]

    return sample_ids[:3]


def get_three_real_depths(
    sample_ids: Sequence[Any],
    *,
    anchor_a_id: Any,
    anchor_b_id: Any,
    axis_a: float,
    axis_query: float,
    axis_b: float,
) -> List[float]:
    can = [canonical_value(x) for x in sample_ids]
    a = canonical_value(anchor_a_id)
    b = canonical_value(anchor_b_id)

    if len(sample_ids) == 3 and a in can and b in can:
        ia = can.index(a)
        ib = can.index(b)
        other_indices = [i for i in range(3) if i not in {ia, ib}]
        if len(other_indices) == 1:
            depths = [None, None, None]
            depths[ia] = axis_a * DEPTH_SCALE
            depths[ib] = axis_b * DEPTH_SCALE
            depths[other_indices[0]] = axis_query * DEPTH_SCALE
            return [float(x) for x in depths]

    values = np.linspace(axis_a, axis_b, len(sample_ids), dtype=float) * DEPTH_SCALE
    return [float(x) for x in values]


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
                names = []
                for i, item in enumerate(schema):
                    if isinstance(item, str):
                        names.append(item)
                    elif isinstance(item, dict) and "name" in item:
                        names.append(str(item["name"]))
                    else:
                        names.append(f"attribute_{i}")
                return names
    return None


# =============================================================================
# Value extraction
# =============================================================================

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
                f"{source}: obsm[{matrix_key!r}] exists, but no attribute names were found in adata.uns."
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
        f"obsm keys: {list(adata.obsm.keys())}"
    )


def build_domain_color_map(
    *arrays: Sequence[Any],
    palette_name: str = ATTRIBUTE_PALETTE,
) -> Tuple[List[np.ndarray], Dict[str, Tuple[float, float, float, float]]]:
    canonical_arrays = [np.asarray([canonical_value(x) for x in arr], dtype=str) for arr in arrays]
    categories = sorted(set(np.concatenate(canonical_arrays).tolist()), key=sort_key)

    colors = sns.color_palette(palette_name, n_colors=max(len(categories), 1))
    color_map = {category: colors[i % len(colors)] for i, category in enumerate(categories)}
    return canonical_arrays, color_map


def make_shared_gene_norm(
    *arrays: np.ndarray,
    upper_percentile: float = UPPER_PERCENTILE,
) -> Normalize:
    combined = np.concatenate([np.asarray(x, dtype=float).reshape(-1) for x in arrays])
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
    cmap_name: str = GENE_CMAP,
    alpha_min: float = 0.04,
    alpha_max: float = 0.96,
    alpha_gamma: float = 0.75,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scaled = np.clip(norm(values), 0.0, 1.0)

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(scaled)

    alpha = alpha_min + (alpha_max - alpha_min) * np.power(scaled, alpha_gamma)
    rgba[:, 3] = alpha
    return rgba


# =============================================================================
# Geometry and plotting helpers
# =============================================================================

def finite_rows(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    return np.all(np.isfinite(coords[:, :2]), axis=1)


def combined_xy_limits(
    arrays: Iterable[np.ndarray],
    *,
    pad_fraction: float = 0.05,
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


def draw_slice_plane(
    ax,
    depth: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    *,
    edgecolor: str,
    linewidth: float,
    alpha: float,
    face_alpha: float = 0.04,
    linestyle: str = "-",
    dashed: bool = False,
) -> None:
    x0, x1 = x_bounds
    y0, y1 = y_bounds

    verts = [[
        (depth, x0, y0),
        (depth, x1, y0),
        (depth, x1, y1),
        (depth, x0, y1),
    ]]
    poly = Poly3DCollection(
        verts,
        facecolors=(0.92, 0.92, 0.92, face_alpha),
        edgecolors=(0, 0, 0, 0),
        linewidths=0,
    )
    ax.add_collection3d(poly)

    frame = np.array(
        [
            [depth, x0, y0],
            [depth, x1, y0],
            [depth, x1, y1],
            [depth, x0, y1],
            [depth, x0, y0],
        ],
        dtype=float,
    )
    line, = ax.plot(
        frame[:, 0],
        frame[:, 1],
        frame[:, 2],
        color=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        zorder=0,
    )
    if dashed:
        line.set_dashes(VIRTUAL_FRAME_DASH_PATTERN[1])


def add_plane_label(
    ax,
    depth: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    text: str,
    *,
    color: str = "black",
    fontsize: float = 10.0,
) -> None:
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    x_text = x0 + 0.04 * (x1 - x0)
    y_text = y0 - 0.09 * (y1 - y0)

    ax.text(
        depth,
        x_text,
        y_text,
        text,
        color=color,
        fontsize=fontsize,
        ha="left",
        va="top",
        zorder=10,
    )


def style_3d_axis(
    ax,
    *,
    depth_positions: Sequence[float],
    depth_labels: Sequence[str],
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    invert_x: bool = INVERT_X,
    invert_y: bool = INVERT_Y,
) -> None:
    x_pad = 0.05 * max(x_bounds[1] - x_bounds[0], 1.0)
    y_pad = 0.05 * max(y_bounds[1] - y_bounds[0], 1.0)
    d_pad = 0.08 * max(depth_positions[-1] - depth_positions[0], 1.0)

    ax.set_xlim(depth_positions[0] - d_pad, depth_positions[-1] + d_pad)

    if invert_x:
        ax.set_ylim(x_bounds[1] + x_pad, x_bounds[0] - x_pad)
    else:
        ax.set_ylim(x_bounds[0] - x_pad, x_bounds[1] + x_pad)

    if invert_y:
        ax.set_zlim(y_bounds[1] + y_pad, y_bounds[0] - y_pad)
    else:
        ax.set_zlim(y_bounds[0] - y_pad, y_bounds[1] + y_pad)

    ax.set_xticks(depth_positions)
    ax.set_xticklabels(depth_labels, fontsize=8.2)
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.set_xlabel("Biological depth", labelpad=10)
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    ax.set_proj_type("ortho")
    ax.view_init(elev=ELEV, azim=AZIM)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0.82, 0.82, 0.82, 0.25))

    depth_span = max(depth_positions[-1] - depth_positions[0], 1.0)
    x_span = max(x_bounds[1] - x_bounds[0], 1.0)
    y_span = max(y_bounds[1] - y_bounds[0], 1.0)
    ax.set_box_aspect((depth_span * 1.35, x_span, y_span))


def plot_template_grid_3d(
    ax,
    grid_pts: np.ndarray,
    *,
    depth: float,
    point_size: float,
) -> None:
    ax.scatter(
        np.full(len(grid_pts), depth),
        grid_pts[:, 0],
        grid_pts[:, 1],
        s=point_size,
        facecolors="none",
        edgecolors="#A8A8A8",
        linewidths=0.55,
        depthshade=False,
        rasterized=True,
    )


def plot_expression_points_3d(
    ax,
    coords: np.ndarray,
    values: np.ndarray,
    *,
    depth: float,
    norm: Normalize,
    point_size: float,
    cmap_name: str = GENE_CMAP,
    alpha_max: float = 0.92,
) -> None:
    order = np.argsort(np.asarray(values, dtype=float))
    rgba = expression_rgba(values, norm=norm, cmap_name=cmap_name, alpha_max=alpha_max)

    ax.scatter(
        np.full(len(coords), depth)[order],
        coords[order, 0],
        coords[order, 1],
        c=rgba[order],
        s=point_size,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )


def plot_category_points_3d(
    ax,
    coords: np.ndarray,
    categories: np.ndarray,
    *,
    depth: float,
    color_map: Mapping[str, Tuple[float, float, float, float]],
    point_size: float,
    alpha: float = 0.90,
) -> None:
    colors = [color_map[str(category)] for category in categories]
    ax.scatter(
        np.full(len(coords), depth),
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )


# =============================================================================
# Transport-curve helpers
# =============================================================================

def choose_subset_indices(
    n_total: int,
    n_keep: int,
    *,
    seed: int,
) -> np.ndarray:
    if n_total <= n_keep:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_keep, replace=False)
    return np.sort(idx)


def cubic_bezier_2d(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(n_steps), dtype=float)[:, None]
    omt = 1.0 - t
    curve = (
        omt**3 * p0[None, :]
        + 3.0 * omt**2 * t * p1[None, :]
        + 3.0 * omt * t**2 * p2[None, :]
        + t**3 * p3[None, :]
    )
    return curve.astype(np.float32)


def build_smooth_curve_to_plate(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    *,
    control_scale: float = CONTROL_SCALE,
    n_steps: int = 44,
) -> np.ndarray:
    start_xy = np.asarray(start_xy, dtype=float)
    end_xy = np.asarray(end_xy, dtype=float)

    disp = end_xy - start_xy
    dist = float(np.linalg.norm(disp))
    if not np.isfinite(dist) or dist <= 1e-8:
        return np.repeat(start_xy[None, :], n_steps, axis=0).astype(np.float32)

    unit = disp / dist
    perp = np.array([-unit[1], unit[0]], dtype=float)
    sign = -1.0 if start_xy[0] < end_xy[0] else 1.0
    bend = 0.18 * dist * sign * perp

    handle = control_scale * dist
    c1 = start_xy + handle * unit + 0.55 * bend
    c2 = end_xy - handle * unit + 0.25 * bend

    return cubic_bezier_2d(start_xy, c1, c2, end_xy, n_steps)


def lift_curve_to_3d(
    curve_xy: np.ndarray,
    *,
    depth_start: float,
    depth_end: float,
) -> np.ndarray:
    depths = np.linspace(depth_start, depth_end, len(curve_xy), dtype=float)
    return np.column_stack([depths, curve_xy[:, 0], curve_xy[:, 1]]).astype(np.float32)


def make_arrowhead_segments_3d(
    tip: np.ndarray,
    previous: np.ndarray,
    *,
    head_length_scale: float = ARROW_HEAD_LENGTH_SCALE,
    head_width_scale: float = ARROW_HEAD_WIDTH_SCALE,
) -> Optional[np.ndarray]:
    tip = np.asarray(tip, dtype=float)
    previous = np.asarray(previous, dtype=float)

    tangent = tip - previous
    step_len = float(np.linalg.norm(tangent))
    if not np.isfinite(step_len) or step_len <= 1e-8:
        return None

    tangent = tangent / step_len
    head_len = head_length_scale * step_len
    head_width = head_width_scale * head_len

    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(tangent, ref)) > 0.90:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)

    side = np.cross(tangent, ref)
    side_norm = float(np.linalg.norm(side))
    if not np.isfinite(side_norm) or side_norm <= 1e-8:
        return None
    side = side / side_norm

    base = tip - head_len * tangent
    wing1 = base + head_width * side
    wing2 = base - head_width * side

    return np.array([[tip, wing1], [tip, wing2]], dtype=float)


def build_transport_curves(
    start_coords: np.ndarray,
    end_coords: np.ndarray,
    *,
    depth_start: float,
    depth_end: float,
    max_curves: int,
    seed: int,
) -> List[np.ndarray]:
    n = min(len(start_coords), len(end_coords))
    if n == 0:
        return []

    start_coords = np.asarray(start_coords[:n], dtype=float)
    end_coords = np.asarray(end_coords[:n], dtype=float)

    valid = finite_rows(start_coords) & finite_rows(end_coords)
    start_coords = start_coords[valid]
    end_coords = end_coords[valid]

    if len(start_coords) == 0:
        return []

    keep = choose_subset_indices(len(start_coords), max_curves, seed=seed)
    start_coords = start_coords[keep]
    end_coords = end_coords[keep]

    curves = []
    for p0, p1 in zip(start_coords, end_coords):
        curve_xy = build_smooth_curve_to_plate(p0, p1, control_scale=CONTROL_SCALE, n_steps=44)
        curve_3d = lift_curve_to_3d(curve_xy, depth_start=depth_start, depth_end=depth_end)
        curves.append(curve_3d)
    return curves


def add_curve_collection(
    ax,
    curves: Sequence[np.ndarray],
    *,
    color: str,
    linewidth: float,
    alpha: float,
    add_arrowheads: bool = True,
) -> None:
    for curve in curves:
        ax.plot(
            curve[:, 0], curve[:, 1], curve[:, 2],
            color=color, linewidth=linewidth, alpha=alpha, zorder=3
        )

        if add_arrowheads and len(curve) >= 3:
            arrow_segments = make_arrowhead_segments_3d(
                tip=curve[-1],
                previous=curve[-3],
            )
            if arrow_segments is not None:
                for seg in arrow_segments:
                    ax.plot(
                        seg[:, 0], seg[:, 1], seg[:, 2],
                        color=color, linewidth=linewidth, alpha=alpha, zorder=4
                    )


# =============================================================================
# Context
# =============================================================================

class Context:
    pass


def infer_evidence_group_indices(
    meta: ad.AnnData,
    n_a: int,
    n_b: int,
    anchor_a_id: Any,
    anchor_b_id: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    source_columns = [
        "source_anchor_id",
        "source_id",
        "anchor_id",
        "anchor_sample_id",
        "source_sample_id",
    ]

    for column in source_columns:
        if column in meta.obs.columns:
            raw = meta.obs[column].to_numpy()
            decoded = np.asarray([canonical_value(x) for x in raw], dtype=str)
            idx_a = np.where(decoded == canonical_value(anchor_a_id))[0]
            idx_b = np.where(decoded == canonical_value(anchor_b_id))[0]
            if len(idx_a) > 0 and len(idx_b) > 0:
                return idx_a, idx_b

    total_expected = int(n_a + n_b)
    if meta.n_obs < total_expected:
        raise ValueError(
            f"Generation metadata has {meta.n_obs} evidence points, "
            f"but {total_expected} are expected from the two anchors."
        )

    idx_a = np.arange(n_a)
    idx_b = np.arange(n_a, n_a + n_b)
    return idx_a, idx_b


def prepare_context(
    *,
    root: Path = ROOT,
    virtual_path: Path = VIRTUAL_PATH,
    meta_path: Path = META_PATH,
    group_id: Any = GROUP_ID,
) -> Context:
    virtual, meta = read_inputs(virtual_path, meta_path)

    anchor_a_id, anchor_b_id = get_source_anchor_ids(virtual, meta)
    axis_a, axis_query, axis_b = get_axis_positions(virtual, meta)

    real_a, coords_a = load_real_anchor_slice(root, group_id, anchor_a_id)
    real_b, coords_b = load_real_anchor_slice(root, group_id, anchor_b_id)

    grid_pts = get_grid_points(meta)
    evidence_coords = get_spatial_coords(meta, source="generation metadata", spatial_key="spatial")
    virtual_coords = get_spatial_coords(virtual, source="generated virtual tissue", spatial_key="spatial")

    if "valid" in virtual.obs.columns:
        valid_mask = virtual.obs["valid"].to_numpy().astype(bool)
        virtual = virtual[valid_mask].copy()
        virtual_coords = virtual_coords[valid_mask]

    idx_a, idx_b = infer_evidence_group_indices(meta, len(coords_a), len(coords_b), anchor_a_id, anchor_b_id)

    three_real_ids = infer_three_real_samples(root, group_id, anchor_a_id, anchor_b_id, REAL_STACK_SAMPLE_IDS)
    three_real_depths = get_three_real_depths(
        three_real_ids,
        anchor_a_id=anchor_a_id,
        anchor_b_id=anchor_b_id,
        axis_a=axis_a,
        axis_query=axis_query,
        axis_b=axis_b,
    )
    three_real_adatas = []
    three_real_coords = []
    for sid in three_real_ids:
        adata_i, coords_i = load_real_anchor_slice(root, group_id, sid)
        three_real_adatas.append(adata_i)
        three_real_coords.append(coords_i)

    x_bounds, y_bounds = combined_xy_limits(
        [coords_a, coords_b, grid_pts, evidence_coords, virtual_coords] + three_real_coords
    )

    ctx = Context()
    ctx.virtual = virtual
    ctx.meta = meta

    ctx.anchor_a_id = anchor_a_id
    ctx.anchor_b_id = anchor_b_id

    ctx.axis_a = axis_a
    ctx.axis_query = axis_query
    ctx.axis_b = axis_b

    ctx.depth_a = axis_a * DEPTH_SCALE
    ctx.depth_query = axis_query * DEPTH_SCALE
    ctx.depth_b = axis_b * DEPTH_SCALE

    ctx.real_a = real_a
    ctx.real_b = real_b
    ctx.coords_a = coords_a
    ctx.coords_b = coords_b

    ctx.grid_pts = grid_pts
    ctx.evidence_coords = evidence_coords
    ctx.virtual_coords = virtual_coords

    ctx.evidence_idx_a = idx_a
    ctx.evidence_idx_b = idx_b

    ctx.three_real_ids = three_real_ids
    ctx.three_real_depths = three_real_depths
    ctx.three_real_adatas = three_real_adatas
    ctx.three_real_coords = three_real_coords

    ctx.x_bounds = x_bounds
    ctx.y_bounds = y_bounds
    return ctx


# =============================================================================
# Reusable panel plotting
# =============================================================================

def plot_entities_in_mode(
    ax,
    items: Sequence[Tuple[np.ndarray, Any, float]],
    *,
    display_mode: str,
    gene: str,
    attribute: str,
    point_size: float,
    point_alpha: float,
):
    display_mode = str(display_mode).lower()

    if display_mode == "gene":
        all_values = []
        values_per_item = []
        for _, adata_i, _ in items:
            vals = extract_gene_values(
                adata_i,
                gene,
                source="gene plotting source",
                apply_log1p=True,
            )
            values_per_item.append(vals)
            all_values.append(vals)

        norm = make_shared_gene_norm(*all_values)

        for (coords, _, depth), vals in zip(items, values_per_item):
            plot_expression_points_3d(
                ax,
                coords,
                vals,
                depth=depth,
                norm=norm,
                point_size=point_size,
                alpha_max=point_alpha,
            )
        return {"norm": norm, "cmap_name": GENE_CMAP}

    else:
        all_attrs = []
        attrs_per_item = []
        for _, adata_i, _ in items:
            vals = extract_attribute_values(
                adata_i,
                attribute,
                source="attribute plotting source",
            )
            attrs_per_item.append(vals)
            all_attrs.append(vals)

        arrays, color_map = build_domain_color_map(*all_attrs, palette_name=ATTRIBUTE_PALETTE)

        for (coords, _, depth), arr in zip(items, arrays):
            plot_category_points_3d(
                ax,
                coords,
                arr,
                depth=depth,
                color_map=color_map,
                point_size=point_size,
                alpha=point_alpha,
            )
        return {"color_map": color_map}


def add_domain_legend(
    fig: plt.Figure,
    color_map: Mapping[str, Tuple[float, float, float, float]],
    *,
    title: str="clusters",
    anchor: Tuple[float, float] = (0.86, 0.50),
) -> None:
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=5.0,
            label=str(cat),
        )
        for cat, color in color_map.items()
    ]
    fig.legend(
        handles=handles,
        title=title,
        loc="center left",
        bbox_to_anchor=anchor,
        frameon=False,
        ncol=1,
        handletextpad=0.35,
        columnspacing=0.8,
        labelspacing=0.28,
    )


def add_gene_colorbar(
    fig: plt.Figure,
    ax,
    norm: Normalize,
    *,
    label: str,
) -> None:
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=GENE_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.05, shrink=0.72)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


# =============================================================================
# Figure 1: stack 3 real slices
# =============================================================================

def plot_figure1_three_real_slices(
    *,
    display_mode: str = DISPLAY_MODE,
    gene: str = MARKER_GENE,
    attribute: str = ATTRIBUTE_NAME,
    outdir: Optional[Path] = OUT_DIR,
    stem: str = "figure1_three_real_slices_3d",
):
    set_publication_style()
    ctx = prepare_context()

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    for sid, depth in zip(ctx.three_real_ids, ctx.three_real_depths):
        draw_slice_plane(
            ax, depth, ctx.x_bounds, ctx.y_bounds,
            edgecolor=REAL_FRAME_COLOR, linewidth=REAL_FRAME_LINEWIDTH,
            alpha=REAL_FRAME_ALPHA, face_alpha=0.04
        )

        add_plane_label(
            ax, depth, ctx.x_bounds, ctx.y_bounds,
            f"Slice {sid} ({sid*10} um)", color="black", fontsize=16
        )

    items = [
        (coords_i, adata_i, depth_i)
        for coords_i, adata_i, depth_i in zip(
            ctx.three_real_coords, ctx.three_real_adatas, ctx.three_real_depths
        )
    ]
    result = plot_entities_in_mode(
        ax, items,
        display_mode=display_mode,
        gene=gene,
        attribute=attribute,
        point_size=REAL_POINT_SIZE,
        point_alpha=REAL_POINT_ALPHA,
    )

    style_3d_axis(
        ax,
        depth_positions=ctx.three_real_depths,
        depth_labels=[str(x) for x in ctx.three_real_ids],
        x_bounds=ctx.x_bounds,
        y_bounds=ctx.y_bounds,
    )

    if str(display_mode).lower() == "gene":
        add_gene_colorbar(fig, ax, result["norm"], label=f"log1p({gene})")
        title_detail = f"{gene} across three real SCC slices"
    else:
        add_domain_legend(fig, result["color_map"], title="clusters")
        title_detail = f"{attribute} across three real SCC slices"

    ax.set_title(
        "Three real SCC slices\n"
        f"{title_detail}",
        pad=15,
        fontsize=12.2,
        fontweight="bold",
    )
    fig.tight_layout()
    return save_or_show(fig, outdir=outdir, stem=f"{stem}_{display_mode}")


# =============================================================================
# Figure 2: anchors + empty virtual support
# =============================================================================

def plot_figure2_anchor_support_stack(
    *,
    display_mode: str = DISPLAY_MODE,
    gene: str = MARKER_GENE,
    attribute: str = ATTRIBUTE_NAME,
    outdir: Optional[Path] = OUT_DIR,
    stem: str = "figure2_anchor_support_stack_3d",
):
    set_publication_style()
    ctx = prepare_context()

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    for depth, label, is_virtual in [
        (ctx.depth_a, f"Anchor ({ctx.anchor_a_id*10} um)", False),
        (ctx.depth_query, "Visium spatial support", True),
        (ctx.depth_b, f"Anchor ({ctx.anchor_b_id*10} um)", False),
    ]:
        draw_slice_plane(
            ax, depth, ctx.x_bounds, ctx.y_bounds,
            edgecolor=VIRTUAL_FRAME_COLOR if is_virtual else REAL_FRAME_COLOR,
            linewidth=VIRTUAL_FRAME_LINEWIDTH if is_virtual else REAL_FRAME_LINEWIDTH,
            alpha=VIRTUAL_FRAME_ALPHA if is_virtual else REAL_FRAME_ALPHA,
            face_alpha=0.02 if is_virtual else 0.04,
            linestyle="--" if is_virtual else "-",
            dashed=is_virtual,
        )
        add_plane_label(ax, depth, ctx.x_bounds, ctx.y_bounds, label, color=VIRTUAL_FRAME_COLOR if is_virtual else "black",fontsize=16)

    plot_template_grid_3d(ax, ctx.grid_pts, depth=ctx.depth_query, point_size=GRID_POINT_SIZE)

    items = [
        (ctx.coords_a, ctx.real_a, ctx.depth_a),
        (ctx.coords_b, ctx.real_b, ctx.depth_b),
    ]
    result = plot_entities_in_mode(
        ax, items,
        display_mode=display_mode,
        gene=gene,
        attribute=attribute,
        point_size=REAL_POINT_SIZE,
        point_alpha=REAL_POINT_ALPHA,
    )

    style_3d_axis(
        ax,
        depth_positions=[ctx.depth_a, ctx.depth_query, ctx.depth_b],
        depth_labels=[f"{ctx.axis_a:g}", f"{ctx.axis_query:g}", f"{ctx.axis_b:g}"],
        x_bounds=ctx.x_bounds,
        y_bounds=ctx.y_bounds,
    )

    aux_handles = [
        Line2D([0], [0], color=VIRTUAL_FRAME_COLOR, linestyle="--", linewidth=VIRTUAL_FRAME_LINEWIDTH, label="Visium support"),
    ]
    leg = ax.legend(handles=aux_handles, loc="upper left", bbox_to_anchor=(0.01, 0.96), frameon=False)
    ax.add_artist(leg)

    if str(display_mode).lower() == "gene":
        add_gene_colorbar(fig, ax, result["norm"], label=f"log1p({gene})")
        title_detail = f"{gene} on anchors with the query support in the middle"
    else:
        add_domain_legend(fig, result["color_map"])
        title_detail = f"{attribute} on anchors with the query support in the middle"

    ax.set_title(
        "Real anchors and user-defined spatial support\n"
        f"{title_detail}",
        pad=15,
        fontsize=12.2,
        fontweight="bold",
    )
    fig.tight_layout()
    return save_or_show(fig, outdir=outdir, stem=f"{stem}_{display_mode}")


# =============================================================================
# Figure 3: transport field (current old figure 2)
# =============================================================================

def plot_figure3_transport_field(
    *,
    display_mode: str = DISPLAY_MODE,
    gene: str = MARKER_GENE,
    attribute: str = ATTRIBUTE_NAME,
    outdir: Optional[Path] = OUT_DIR,
    stem: str = "figure3_transport_field_3d",
):
    set_publication_style()
    ctx = prepare_context()

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    for depth, label, is_virtual in [
        (ctx.depth_a, f"Anchor ({ctx.anchor_a_id*10} um)", False),
        (ctx.depth_query, "Transported evidence", True),
        (ctx.depth_b, f"Anchor ({ctx.anchor_b_id*10} um)", False),
    ]:
        draw_slice_plane(
            ax, depth, ctx.x_bounds, ctx.y_bounds,
            edgecolor=VIRTUAL_FRAME_COLOR if is_virtual else REAL_FRAME_COLOR,
            linewidth=VIRTUAL_FRAME_LINEWIDTH if is_virtual else REAL_FRAME_LINEWIDTH,
            alpha=VIRTUAL_FRAME_ALPHA if is_virtual else REAL_FRAME_ALPHA,
            face_alpha=0.015 if is_virtual else 0.03,
            linestyle="--" if is_virtual else "-",
            dashed=is_virtual,
        )
        add_plane_label(ax, depth, ctx.x_bounds, ctx.y_bounds, label, color=VIRTUAL_FRAME_COLOR if is_virtual else "black",fontsize=16)

    plot_template_grid_3d(ax, ctx.grid_pts, depth=ctx.depth_query, point_size=GRID_POINT_SIZE)

    if str(display_mode).lower() == "gene":
        vals_a = extract_gene_values(ctx.real_a, gene, source=f"anchor {ctx.anchor_a_id}", apply_log1p=True)
        vals_b = extract_gene_values(ctx.real_b, gene, source=f"anchor {ctx.anchor_b_id}", apply_log1p=True)
        vals_ev = extract_gene_values(ctx.meta, gene, source="generation metadata", apply_log1p=True)
        norm = make_shared_gene_norm(vals_a, vals_b, vals_ev)

        plot_expression_points_3d(ax, ctx.coords_a, vals_a, depth=ctx.depth_a, norm=norm, point_size=REAL_POINT_SIZE * 0.95, alpha_max=0.40)
        plot_expression_points_3d(ax, ctx.coords_b, vals_b, depth=ctx.depth_b, norm=norm, point_size=REAL_POINT_SIZE * 0.95, alpha_max=0.40)
        plot_expression_points_3d(ax, ctx.evidence_coords, vals_ev, depth=ctx.depth_query, norm=norm, point_size=EVIDENCE_POINT_SIZE, alpha_max=0.92)

        add_gene_colorbar(fig, ax, norm, label=f"log1p({gene})")
        title_detail = f"{gene} transport from the anchors to the query support"

    else:
        attr_a = extract_attribute_values(ctx.real_a, attribute, source=f"anchor {ctx.anchor_a_id}")
        attr_b = extract_attribute_values(ctx.real_b, attribute, source=f"anchor {ctx.anchor_b_id}")
        attr_ev = extract_attribute_values(ctx.meta, attribute, source="generation metadata")

        (cat_a, cat_b, cat_ev), color_map = build_domain_color_map(attr_a, attr_b, attr_ev, palette_name=ATTRIBUTE_PALETTE)

        plot_category_points_3d(ax, ctx.coords_a, cat_a, depth=ctx.depth_a, color_map=color_map, point_size=REAL_POINT_SIZE * 0.95, alpha=0.38)
        plot_category_points_3d(ax, ctx.coords_b, cat_b, depth=ctx.depth_b, color_map=color_map, point_size=REAL_POINT_SIZE * 0.95, alpha=0.38)
        plot_category_points_3d(ax, ctx.evidence_coords, cat_ev, depth=ctx.depth_query, color_map=color_map, point_size=EVIDENCE_POINT_SIZE, alpha=0.90)

        add_domain_legend(fig, color_map)
        title_detail = f"{attribute} transport from the anchors to the query support"

    evidence_a = ctx.evidence_coords[ctx.evidence_idx_a]
    evidence_b = ctx.evidence_coords[ctx.evidence_idx_b]

    curves_a = build_transport_curves(
        ctx.coords_a, evidence_a,
        depth_start=ctx.depth_a, depth_end=ctx.depth_query,
        max_curves=MAX_CURVES_PER_ANCHOR, seed=RANDOM_SEED
    )
    curves_b = build_transport_curves(
        ctx.coords_b, evidence_b,
        depth_start=ctx.depth_b, depth_end=ctx.depth_query,
        max_curves=MAX_CURVES_PER_ANCHOR, seed=RANDOM_SEED + 1
    )

    add_curve_collection(ax, curves_a, color=CURVE_COLOR_A, linewidth=CURVE_LINEWIDTH, alpha=CURVE_ALPHA, add_arrowheads=True)
    add_curve_collection(ax, curves_b, color=CURVE_COLOR_B, linewidth=CURVE_LINEWIDTH, alpha=CURVE_ALPHA, add_arrowheads=True)

    aux_handles = [
        Line2D([0], [0], color=CURVE_COLOR_A, linewidth=CURVE_LINEWIDTH, label=f"Anchor {ctx.anchor_a_id} → query"),
        Line2D([0], [0], color=CURVE_COLOR_B, linewidth=CURVE_LINEWIDTH, label=f"Anchor {ctx.anchor_b_id} → query"),
        Line2D([0], [0], color=VIRTUAL_FRAME_COLOR, linestyle="--", linewidth=VIRTUAL_FRAME_LINEWIDTH, label="Query support"),
    ]
    leg = ax.legend(handles=aux_handles, loc="upper left", bbox_to_anchor=(0.05, 0.9), frameon=False)
    ax.add_artist(leg)

    style_3d_axis(
        ax,
        depth_positions=[ctx.depth_a, ctx.depth_query, ctx.depth_b],
        depth_labels=[f"{ctx.axis_a:g}", f"{ctx.axis_query:g}", f"{ctx.axis_b:g}"],
        x_bounds=ctx.x_bounds,
        y_bounds=ctx.y_bounds,
    )

    ax.set_title(
        "Evidence transport to the common spatial support\n"
        f"{title_detail}",
        pad=15,
        fontsize=12.2,
        fontweight="bold",
    )
    fig.tight_layout()
    return save_or_show(fig, outdir=outdir, stem=f"{stem}_{display_mode}")


# =============================================================================
# Figure 4: anchors + final generated virtual slice
# =============================================================================

def plot_figure4_generated_virtual_stack(
    *,
    display_mode: str = DISPLAY_MODE,
    gene: str = MARKER_GENE,
    attribute: str = ATTRIBUTE_NAME,
    outdir: Optional[Path] = OUT_DIR,
    stem: str = "figure4_generated_virtual_stack_3d",
):
    set_publication_style()
    ctx = prepare_context()

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    for depth, label, is_virtual in [
        # (ctx.depth_a, f"Anchor ({ctx.anchor_a_id*10} um)", False),
        (ctx.depth_query, "Virtual (10 um)", True),
        # (ctx.depth_b, f"Anchor ({ctx.anchor_b_id*10} um)", False),
    ]:
        draw_slice_plane(
            ax, depth, ctx.x_bounds, ctx.y_bounds,
            edgecolor=VIRTUAL_FRAME_COLOR if is_virtual else REAL_FRAME_COLOR,
            linewidth=VIRTUAL_FRAME_LINEWIDTH if is_virtual else REAL_FRAME_LINEWIDTH,
            alpha=VIRTUAL_FRAME_ALPHA if is_virtual else REAL_FRAME_ALPHA,
            face_alpha=0.02 if is_virtual else 0.04,
            linestyle="--" if is_virtual else "-",
            dashed=is_virtual,
        )
        add_plane_label(ax, depth, ctx.x_bounds, ctx.y_bounds, label, color=VIRTUAL_FRAME_COLOR if is_virtual else "black",fontsize=16)

    items = [
        # (ctx.coords_a, ctx.real_a, ctx.depth_a),
        (ctx.virtual_coords, ctx.virtual, ctx.depth_query),
        # (ctx.coords_b, ctx.real_b, ctx.depth_b),
    ]
    result = plot_entities_in_mode(
        ax, items,
        display_mode=display_mode,
        gene=gene,
        attribute=attribute,
        point_size=VIRTUAL_POINT_SIZE,
        point_alpha=VIRTUAL_POINT_ALPHA,
    )

    # style_3d_axis(
    #     ax,
    #     depth_positions=[ctx.depth_a, ctx.depth_query, ctx.depth_b],
    #     depth_labels=[f"{ctx.axis_a:g}", f"{ctx.axis_query:g}", f"{ctx.axis_b:g}"],
    #     x_bounds=ctx.x_bounds,
    #     y_bounds=ctx.y_bounds,
    # )
    style_3d_axis(
        ax,
        depth_positions=[ ctx.depth_query],
        depth_labels=[f"{ctx.axis_query:g}"],
        x_bounds=ctx.x_bounds,
        y_bounds=ctx.y_bounds,
    )

    aux_handles = [
        Line2D([0], [0], color=VIRTUAL_FRAME_COLOR, linestyle="--", linewidth=VIRTUAL_FRAME_LINEWIDTH, label="Generated virtual slice"),
    ]
    leg = ax.legend(handles=aux_handles, loc="upper left", bbox_to_anchor=(0.01, 0.96), frameon=False)
    ax.add_artist(leg)

    if str(display_mode).lower() == "gene":
        add_gene_colorbar(fig, ax, result["norm"], label=f"log1p({gene})")
        title_detail = f"{gene} on the anchors and the final generated virtual slice"
    else:
        add_domain_legend(fig, result["color_map"])
        title_detail = f"{attribute} on the anchors and the final generated virtual slice"

    ax.set_title(
        "Final generated virtual tissue in 3D context\n"
        f"{title_detail}",
        pad=15,
        fontsize=12.2,
        fontweight="bold",
    )
    fig.tight_layout()
    return save_or_show(fig, outdir=outdir, stem=f"{stem}_{display_mode}")


# =============================================================================
# Run
# =============================================================================

def main() -> None:
    # plot_figure1_three_real_slices(display_mode=DISPLAY_MODE)
    # plot_figure2_anchor_support_stack(display_mode=DISPLAY_MODE)
    plot_figure3_transport_field(display_mode=DISPLAY_MODE)
    # plot_figure4_generated_virtual_stack(display_mode=DISPLAY_MODE)


if __name__ == "__main__":
    main()
