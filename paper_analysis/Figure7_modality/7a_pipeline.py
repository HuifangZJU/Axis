
import os
import argparse
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


# ============================================================
# Recycled data I/O from the previous Figure-7a script
# ============================================================
REAL_GENE = "/media/huifang/data/registration/phenocycler/huifang/h5ad/LUAD_2_A_xenium.h5ad"
REAL_PROTEIN = "/media/huifang/data/registration/phenocycler/huifang/h5ad/LUAD_2_A_protein.h5ad"
VIRTUAL_PROTEIN = "/media/huifang/data/registration/phenocycler/huifang/virtual_tissue/modality_LUAD_2_A/analysis/virtual_protein_1.000.h5ad"
OUTDIR = "/media/huifang/data/registration/paper_analysis/Figure7_modality/pipeline/"


# ============================================================
# Recycled overall orientation style
# ============================================================
SWAP_XY = True
INVERT_X = True
INVERT_Y = False

CANVAS_SIZE = 1024.0
CANVAS_PAD = 36.0
MAX_PLOT_CELLS = 50000
RANDOM_SEED = 0
POINT_SIZE = 1.0
FIG_DPI = 300

# top-down/slanted view, aligned with the style used in the other pipeline figures
VIEW_ELEV = 30.0
VIEW_AZIM = -64.0

# plane / stack controls
FRAME_LINEWIDTH = 1.0
FRAME_COLOR_REAL = "0.55"
FRAME_COLOR_MATCHED = "0.55"
FRAME_DASHED = (0, (4.0, 2.4))

FIG1_GAP = 1024.0

# Figure 2 has two independent inter-slice gap controls.
FIG2_GENE_TO_MATCHED_GAP = 520.0
FIG2_MATCHED_TO_PROJECTION_GAP = 780.0

FIG3_GAP = 520.0
FIG4_GAP = 520.0

# transparency controls
FIG1_GENE_ALPHA = 0.88
FIG1_PROTEIN_ALPHA = 0.88

FIG2_REAL_ALPHA = 0.88
FIG2_MATCHED_ALPHA = 0.72
FIG2_PROJECTION_ALPHA = 0.26

FIG3_REAL_ALPHA = 0.88
FIG3_MATCHED_ALPHA = 0.72

FIG4_REAL_ALPHA = 0.88
FIG4_MATCHED_ALPHA = 0.72

# matched-slice marker edge control
FIG2_MATCHED_MARKER_EDGE_COLOR = "0.55"
FIG2_MATCHED_MARKER_EDGE_WIDTH = 0.02

FIG4_MATCHED_MARKER_EDGE_COLOR = "0.55"
FIG4_MATCHED_MARKER_EDGE_WIDTH = 0.02

# default feature names
DEFAULT_GENE_FEATURE = "HLA-A"
DEFAULT_PROTEIN_FEATURE = "HLA-A"


# ============================================================
# Color palette (recycled style, regenerated for the new layout)
# ============================================================
# Spatial gene maps use the standard Matplotlib Reds colormap by default.
GENE_CMAP = plt.get_cmap("Reds")

# Figure 4 only: reuse the original orange gene palette.
FIG4_GENE_CMAP = LinearSegmentedColormap.from_list(
    "axis_gene_orange",
    ["#ffffff", "#fef0d9", "#fdcc8a", "#fc8d59", "#d7301f"],
)

# Keep the blue protein palette from the regenerated pipeline.
PROTEIN_CMAP = LinearSegmentedColormap.from_list(
    "axis_protein_blue",
    ["#ffffff", "#eff3ff", "#bdd7e7", "#6baed6", "#2171b5"],
)


@dataclass
class SliceLayer:
    name: str
    coords: np.ndarray
    values: np.ndarray
    cmap_kind: str   # "gene" or "protein"
    alpha: float
    z: float
    frame_style: str  # "solid", "dashed", "none"
    edge_color: Optional[str] = None
    edge_width: float = 0.0
    cmap_override: Optional[object] = None


# ============================================================
# Utilities
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_or_show(fig, save_path: Optional[str], show: bool, dpi: int = FIG_DPI):
    if save_path is not None:
        outdir = os.path.dirname(os.path.abspath(save_path))
        if outdir:
            ensure_dir(outdir)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"[OK] saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def read_h5ad(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return sc.read_h5ad(path)


def get_valid_mask(adata):
    if "valid" in adata.obs.columns:
        return np.asarray(adata.obs["valid"]).astype(bool)
    return np.ones(adata.n_obs, dtype=bool)


def maybe_downsample_indices(n: int, max_cells: int = MAX_PLOT_CELLS, seed: int = RANDOM_SEED):
    if n <= max_cells:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_cells, replace=False))


def prepare_coords(
    adata,
    coord_key: str = "spatial",
    valid_only: bool = True,
    swap_xy: bool = SWAP_XY,
    max_cells: int = MAX_PLOT_CELLS,
    seed: int = RANDOM_SEED,
):
    if coord_key not in adata.obsm:
        raise KeyError(f"Coordinate key '{coord_key}' not found in adata.obsm")

    coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    mask = get_valid_mask(adata) if valid_only else np.ones(adata.n_obs, dtype=bool)
    coords = coords[mask]

    if swap_xy:
        coords = coords[:, [1, 0]]

    keep_idx = maybe_downsample_indices(coords.shape[0], max_cells=max_cells, seed=seed)
    return coords[keep_idx], mask, keep_idx


def normalize_coords_to_canvas(coords: np.ndarray, canvas_size: float = CANVAS_SIZE, pad: float = CANVAS_PAD):
    coords = np.asarray(coords, dtype=np.float32)
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    span_x = max(float(xmax - xmin), 1e-6)
    span_y = max(float(ymax - ymin), 1e-6)
    scale = (canvas_size - 2 * pad) / max(span_x, span_y)

    out = coords.copy()
    out[:, 0] = (coords[:, 0] - xmin) * scale + pad
    out[:, 1] = (coords[:, 1] - ymin) * scale + pad

    extra_x = canvas_size - (out[:, 0].max() - out[:, 0].min()) - 2 * pad
    extra_y = canvas_size - (out[:, 1].max() - out[:, 1].min()) - 2 * pad
    out[:, 0] += 0.5 * max(extra_x, 0.0)
    out[:, 1] += 0.5 * max(extra_y, 0.0)
    return out


def extract_feature_vals(
    adata,
    feature: str,
    valid_mask=None,
    keep_idx=None,
    use_log1p: bool = True,
):
    if feature in adata.var_names:
        idx = int(np.where(adata.var_names == feature)[0][0])
        vals = adata.X[:, idx]
        if sp.issparse(vals):
            vals = vals.toarray().ravel()
        else:
            vals = np.asarray(vals).ravel()
        vals = vals.astype(np.float32)
        if use_log1p and np.nanmin(vals) >= 0:
            vals = np.log1p(vals)
    elif feature in adata.obs.columns:
        vals = np.asarray(adata.obs[feature]).ravel().astype(np.float32)
    else:
        raise ValueError(f"Feature '{feature}' not found in var_names or obs")

    if valid_mask is not None:
        vals = vals[valid_mask]
    if keep_idx is not None:
        vals = vals[keep_idx]
    return np.asarray(vals, dtype=np.float32)


def robust_vrange(vals_list: Sequence[np.ndarray], low_pct: float = 1.0, high_pct: float = 99.0):
    arrs = []
    for vals in vals_list:
        vals = np.asarray(vals).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            arrs.append(vals)
    if not arrs:
        return 0.0, 1.0
    arr = np.concatenate(arrs)
    vmin = float(np.percentile(arr, low_pct))
    vmax = float(np.percentile(arr, high_pct))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def choose_cmap(kind: str):
    if kind == "gene":
        return GENE_CMAP
    if kind == "protein":
        return PROTEIN_CMAP
    raise ValueError(f"Unknown cmap kind: {kind}")


def values_to_rgba(
    values: np.ndarray,
    cmap_kind: str,
    vmin: float,
    vmax: float,
    alpha: float,
    cmap_override=None,
):
    cmap = choose_cmap(cmap_kind) if cmap_override is None else cmap_override
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    scaled = np.clip(norm(np.asarray(values, dtype=float)), 0.0, 1.0)
    rgba = np.asarray(cmap(scaled), dtype=float)
    rgba[:, 3] = alpha * scaled
    return rgba


def load_spatial_feature(
    h5ad_path: str,
    feature: str,
    *,
    coord_key: str = "spatial",
    seed: int = RANDOM_SEED,
    max_cells: int = MAX_PLOT_CELLS,
):
    adata = read_h5ad(h5ad_path)
    coords, valid_mask, keep_idx = prepare_coords(
        adata,
        coord_key=coord_key,
        valid_only=True,
        swap_xy=SWAP_XY,
        max_cells=max_cells,
        seed=seed,
    )
    coords = normalize_coords_to_canvas(coords, canvas_size=CANVAS_SIZE, pad=CANVAS_PAD)
    values = extract_feature_vals(
        adata,
        feature,
        valid_mask=valid_mask,
        keep_idx=keep_idx,
        use_log1p=True,
    )
    return coords, values


# ============================================================
# 3D drawing helpers
# ============================================================
def draw_plane_frame(ax, z: float, frame_style: str):
    if frame_style == "none":
        return

    xs = [0.0, CANVAS_SIZE, CANVAS_SIZE, 0.0, 0.0]
    ys = [0.0, 0.0, CANVAS_SIZE, CANVAS_SIZE, 0.0]
    kwargs = dict(color=FRAME_COLOR_REAL, linewidth=FRAME_LINEWIDTH, alpha=0.85, zorder=1)
    if frame_style == "dashed":
        kwargs["linestyle"] = FRAME_DASHED
        kwargs["color"] = FRAME_COLOR_MATCHED
    ax.plot(xs, ys, [z] * len(xs), **kwargs)


def style_3d_axis(ax, z_positions: Sequence[float], elev: float = VIEW_ELEV, azim: float = VIEW_AZIM):
    x_pad = 18.0
    y_pad = 18.0
    z_positions = np.asarray(z_positions, dtype=float)
    zmin = float(np.min(z_positions))
    zmax = float(np.max(z_positions))
    zspan = max(zmax - zmin, 1.0)

    ax.set_xlim(-x_pad, CANVAS_SIZE + x_pad)
    ax.set_ylim(-y_pad, CANVAS_SIZE + y_pad)
    ax.set_zlim(zmin - 0.18 * zspan, zmax + 0.12 * zspan)

    if INVERT_X:
        ax.invert_xaxis()
    if INVERT_Y:
        ax.invert_yaxis()

    ax.view_init(elev=float(elev), azim=float(azim))
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    try:
        ax.set_box_aspect((1.0, 1.0, max(0.55, zspan / CANVAS_SIZE * 1.1)))
    except Exception:
        pass

    ax.set_axis_off()
    ax.grid(False)


def draw_slice(ax, layer: SliceLayer, vmin: float, vmax: float):
    rgba = values_to_rgba(
        layer.values,
        layer.cmap_kind,
        vmin,
        vmax,
        layer.alpha,
        cmap_override=layer.cmap_override,
    )
    edgecolors = layer.edge_color if layer.edge_color is not None else "none"
    linewidths = layer.edge_width if layer.edge_color is not None else 0.0

    ax.scatter(
        layer.coords[:, 0],
        layer.coords[:, 1],
        np.full(layer.coords.shape[0], float(layer.z)),
        c=rgba,
        s=POINT_SIZE,
        edgecolors=edgecolors,
        linewidths=linewidths,
        depthshade=False,
        rasterized=True,
        zorder=3,
    )
    draw_plane_frame(ax, layer.z, layer.frame_style)


def compute_ranges(layers: Sequence[SliceLayer]):
    gene_vals = [x.values for x in layers if x.cmap_kind == "gene"]
    prot_vals = [x.values for x in layers if x.cmap_kind == "protein"]
    vr = {}
    if gene_vals:
        vr["gene"] = robust_vrange(gene_vals, low_pct=1.0, high_pct=99.0)
    if prot_vals:
        vr["protein"] = robust_vrange(prot_vals, low_pct=1.0, high_pct=99.0)
    return vr


def render_stack(
    layers: Sequence[SliceLayer],
    title: str,
    save_path: Optional[str],
    show: bool,
    figsize: Tuple[float, float] = (5.0, 5.4),
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    vranges = compute_ranges(layers)
    for layer in layers:
        vmin, vmax = vranges[layer.cmap_kind]
        draw_slice(ax, layer, vmin=vmin, vmax=vmax)

    style_3d_axis(ax, [x.z for x in layers], elev=VIEW_ELEV, azim=VIEW_AZIM)
    # ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    save_or_show(fig, save_path=save_path, show=show, dpi=FIG_DPI)


# ============================================================
# Figure builders
# ============================================================
def build_figure1(gene_feature: str, protein_feature: str, outdir: str, show: bool):
    gene_coords, gene_vals = load_spatial_feature(REAL_GENE, gene_feature, seed=1)
    prot_coords, prot_vals = load_spatial_feature(REAL_PROTEIN, protein_feature, seed=2)

    layers = [
        SliceLayer(
            name="Measured gene",
            coords=gene_coords,
            values=gene_vals,
            cmap_kind="gene",
            alpha=FIG1_GENE_ALPHA,
            z=FIG1_GAP,
            frame_style="solid",
        ),
        SliceLayer(
            name="Measured protein",
            coords=prot_coords,
            values=prot_vals,
            cmap_kind="protein",
            alpha=FIG1_PROTEIN_ALPHA,
            z=0.0,
            frame_style="solid",
        ),
    ]

    render_stack(
        layers,
        title="Measured spatial gene and protein",
        save_path=os.path.join(outdir, f"figure1_measured_gene_protein_{gene_feature}_{protein_feature}.png"),
        show=show,
        figsize=(3, 6),
    )


def build_figure2(gene_feature: str, protein_feature: str, outdir: str, show: bool):
    gene_coords, gene_vals = load_spatial_feature(REAL_GENE, gene_feature, seed=3)
    virt_coords, virt_vals = load_spatial_feature(VIRTUAL_PROTEIN, protein_feature, seed=4)
    prot_coords, prot_vals = load_spatial_feature(REAL_PROTEIN, protein_feature, seed=5)

    # Figure-2 layout has two independently controlled gaps:
    # measured gene -> matched virtual protein -> measured protein projection.
    z_projection = 0.0
    z_matched = z_projection + float(FIG2_MATCHED_TO_PROJECTION_GAP)
    z_gene = z_matched + float(FIG2_GENE_TO_MATCHED_GAP)

    layers = [
        SliceLayer(
            name="Measured gene",
            coords=gene_coords,
            values=gene_vals,
            cmap_kind="gene",
            alpha=FIG2_REAL_ALPHA,
            z=z_gene,
            frame_style="solid",
        ),
        SliceLayer(
            name="Matched virtual protein",
            coords=virt_coords,
            values=virt_vals,
            cmap_kind="protein",
            alpha=FIG2_MATCHED_ALPHA,
            z=z_matched,
            frame_style="dashed",
            edge_color=FIG2_MATCHED_MARKER_EDGE_COLOR,
            edge_width=FIG2_MATCHED_MARKER_EDGE_WIDTH,
        ),
        SliceLayer(
            name="Measured protein",
            coords=prot_coords,
            values=prot_vals,
            cmap_kind="protein",
            alpha=FIG2_PROJECTION_ALPHA,
            z=z_projection,
            frame_style="none",
        ),
    ]

    render_stack(
        layers,
        title="Gene → matched virtual protein → measured protein",
        save_path=os.path.join(outdir, f"figure2_gene_virtualprotein_measuredprotein_{gene_feature}_{protein_feature}.png"),
        show=show,
        figsize=(3, 6),
    )


def build_figure3(gene_feature: str, protein_feature: str, outdir: str, show: bool):
    gene_coords, gene_vals = load_spatial_feature(REAL_GENE, gene_feature, seed=6)
    virt_coords, virt_vals = load_spatial_feature(VIRTUAL_PROTEIN, protein_feature, seed=7)

    layers = [
        SliceLayer(
            name="Measured gene",
            coords=gene_coords,
            values=gene_vals,
            cmap_kind="gene",
            alpha=FIG3_REAL_ALPHA,
            z=FIG3_GAP,
            frame_style="solid",
        ),
        SliceLayer(
            name="Matched virtual protein",
            coords=virt_coords,
            values=virt_vals,
            cmap_kind="protein",
            alpha=FIG3_MATCHED_ALPHA,
            z=0.0,
            frame_style="dashed",
            edge_color=None,
            edge_width=0.0,
        ),
    ]

    render_stack(
        layers,
        title="Measured gene and matched virtual protein",
        save_path=os.path.join(outdir, f"figure3_gene_virtualprotein_noedge_{gene_feature}_{protein_feature}.png"),
        show=show,
        figsize=(3, 6),
    )


def build_figure4(gene_feature: str, protein_feature: str, outdir: str, show: bool):
    gene_coords, gene_vals = load_spatial_feature(REAL_GENE, gene_feature, seed=8)
    virt_coords, virt_vals = load_spatial_feature(VIRTUAL_PROTEIN, protein_feature, seed=9)

    layers = [
        SliceLayer(
            name="Measured gene",
            coords=gene_coords,
            values=gene_vals,
            cmap_kind="gene",
            alpha=FIG4_REAL_ALPHA,
            z=FIG4_GAP,
            frame_style="solid",
            cmap_override=FIG4_GENE_CMAP,
        ),
        SliceLayer(
            name="Matched virtual protein",
            coords=virt_coords,
            values=virt_vals,
            cmap_kind="protein",
            alpha=FIG4_MATCHED_ALPHA,
            z=0.0,
            frame_style="dashed",
            edge_color=FIG4_MATCHED_MARKER_EDGE_COLOR,
            edge_width=FIG4_MATCHED_MARKER_EDGE_WIDTH,
        ),
    ]

    render_stack(
        layers,
        title="Measured gene and matched virtual protein",
        save_path=os.path.join(outdir, f"figure4_gene_virtualprotein_edge_{gene_feature}_{protein_feature}.png"),
        show=show,
        figsize=(3.0, 6),
    )


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Regenerated Figure-7a cross-modality pipeline."
    )
    parser.add_argument("--real-gene", type=str, default=REAL_GENE)
    parser.add_argument("--real-protein", type=str, default=REAL_PROTEIN)
    parser.add_argument("--virtual-protein", type=str, default=VIRTUAL_PROTEIN)
    parser.add_argument("--outdir", type=str, default=OUTDIR)
    parser.add_argument("--gene-feature", type=str, default=DEFAULT_GENE_FEATURE)
    parser.add_argument("--protein-feature", type=str, default=DEFAULT_PROTEIN_FEATURE)
    parser.add_argument("--which", type=str, default="all", choices=["all", "1", "2", "3", "4"])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--fig1-gap", type=float, default=FIG1_GAP)
    parser.add_argument(
        "--fig2-gene-to-matched-gap",
        type=float,
        default=FIG2_GENE_TO_MATCHED_GAP,
    )
    parser.add_argument(
        "--fig2-matched-to-projection-gap",
        type=float,
        default=FIG2_MATCHED_TO_PROJECTION_GAP,
    )
    parser.add_argument("--fig3-gap", type=float, default=FIG3_GAP)
    parser.add_argument("--fig4-gap", type=float, default=FIG4_GAP)
    parser.add_argument("--view-elev", type=float, default=VIEW_ELEV)
    parser.add_argument("--view-azim", type=float, default=VIEW_AZIM)
    parser.add_argument("--max-plot-cells", type=int, default=MAX_PLOT_CELLS)
    parser.add_argument("--point-size", type=float, default=POINT_SIZE)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    # keep the original dir names but allow CLI override
    globals()["REAL_GENE"] = args.real_gene
    globals()["REAL_PROTEIN"] = args.real_protein
    globals()["VIRTUAL_PROTEIN"] = args.virtual_protein
    globals()["OUTDIR"] = args.outdir

    globals()["FIG1_GAP"] = float(args.fig1_gap)
    globals()["FIG2_GENE_TO_MATCHED_GAP"] = float(args.fig2_gene_to_matched_gap)
    globals()["FIG2_MATCHED_TO_PROJECTION_GAP"] = float(args.fig2_matched_to_projection_gap)
    globals()["FIG3_GAP"] = float(args.fig3_gap)
    globals()["FIG4_GAP"] = float(args.fig4_gap)
    globals()["VIEW_ELEV"] = float(args.view_elev)
    globals()["VIEW_AZIM"] = float(args.view_azim)
    globals()["MAX_PLOT_CELLS"] = int(args.max_plot_cells)
    globals()["POINT_SIZE"] = float(args.point_size)
    globals()["RANDOM_SEED"] = int(args.random_seed)

    ensure_dir(OUTDIR)

    print(f"[INFO] REAL_GENE = {REAL_GENE}")
    print(f"[INFO] REAL_PROTEIN = {REAL_PROTEIN}")
    print(f"[INFO] VIRTUAL_PROTEIN = {VIRTUAL_PROTEIN}")
    print(f"[INFO] OUTDIR = {OUTDIR}")
    print(f"[INFO] gene feature = {args.gene_feature}")
    print(f"[INFO] protein feature = {args.protein_feature}")

    if args.which in ("all", "1"):
        build_figure1(args.gene_feature, args.protein_feature, OUTDIR, show=args.show)
    if args.which in ("all", "2"):
        build_figure2(args.gene_feature, args.protein_feature, OUTDIR, show=args.show)
    if args.which in ("all", "3"):
        build_figure3(args.gene_feature, args.protein_feature, OUTDIR, show=args.show)
    if args.which in ("all", "4"):
        build_figure4(args.gene_feature, args.protein_feature, OUTDIR, show=args.show)

    print(f"[DONE] Figure-7a regenerated pipeline written/displayed from: {OUTDIR}")


if __name__ == "__main__":
    main()
