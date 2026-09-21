import os
from typing import Sequence, Tuple, Optional

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.cm import ScalarMappable


# ============================================================
# Global plotting controls
# ============================================================
SWAP_XY = True
INVERT_X = True
INVERT_Y = False
CANVAS_SIZE = 1024.0
MAX_PLOT_CELLS = 50000
RANDOM_SEED = 0
STACK_OFFSET_X = 100.0
STACK_OFFSET_Y = -100.0
BASE_POINT_SIZE = 1.0
ZOOM_POINT_SIZE = 100.0
ROI_LINEWIDTH = 1.8
FRAME_LINEWIDTH = 2
FIG_DPI = 300


# ============================================================
# Basic helpers
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_or_show(fig, save_path: Optional[str] = None, dpi: int = FIG_DPI, show: bool = True):
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


def get_valid_mask(adata):
    if "valid" in adata.obs.columns:
        return np.asarray(adata.obs["valid"]).astype(bool)
    return np.ones(adata.n_obs, dtype=bool)


def maybe_downsample_indices(n: int, max_cells: int = MAX_PLOT_CELLS, seed: int = RANDOM_SEED):
    if n <= max_cells:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_cells, replace=False))
    return idx


def prepare_coords(
    adata,
    coord_key: str = "spatial",
    valid_only: bool = True,
    swap_xy: bool = SWAP_XY,
    max_cells: int = MAX_PLOT_CELLS,
):
    if coord_key not in adata.obsm:
        raise KeyError(f"Coordinate key '{coord_key}' not found in adata.obsm")

    coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    mask = get_valid_mask(adata) if valid_only else np.ones(adata.n_obs, dtype=bool)
    coords = coords[mask]

    if swap_xy:
        coords = coords[:, [1, 0]]

    keep_idx = maybe_downsample_indices(coords.shape[0], max_cells=max_cells)
    return coords[keep_idx], mask, keep_idx


def normalize_coords_to_canvas(coords: np.ndarray, canvas_size: float = CANVAS_SIZE, pad: float = 36.0):
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
        raise ValueError(f"Feature '{feature}' not found in var_names or obs.")

    if valid_mask is not None:
        vals = vals[valid_mask]
    if keep_idx is not None:
        vals = vals[keep_idx]
    return np.asarray(vals, dtype=np.float32)


def robust_vrange(vals_list, low_pct: float = 1.0, high_pct: float = 99.0):
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


def apply_axis_style(ax, xlim, ylim, invert_x: bool = INVERT_X, invert_y: bool = INVERT_Y):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    for s in ax.spines.values():
        s.set_visible(False)


# ============================================================
# Stack plotting
# ============================================================

def _make_stack_extent(n_panels: int):
    xmax = CANVAS_SIZE + STACK_OFFSET_X * (n_panels - 1) + 45
    ymin = STACK_OFFSET_Y * (n_panels - 1) - 45
    return (-45, xmax), (CANVAS_SIZE + 45, ymin)


def _build_marker_stack_data(adata, markers, coord_key, use_log1p=True, max_cells=MAX_PLOT_CELLS):
    coords, valid_mask, keep_idx = prepare_coords(
        adata,
        coord_key=coord_key,
        valid_only=True,
        swap_xy=SWAP_XY,
        max_cells=max_cells,
    )
    coords = normalize_coords_to_canvas(coords, canvas_size=CANVAS_SIZE)
    values = [
        extract_feature_vals(adata, m, valid_mask=valid_mask, keep_idx=keep_idx, use_log1p=use_log1p)
        for m in markers
    ]
    return coords, values


def plot_stacked_signal_panels(
    adata,
    markers: Sequence[str],
    title: str,
    cmap: str,
    colorbar_label: str,
    save_path: Optional[str] = None,
    coord_key: str = "spatial",
    dashed_frame: bool = False,
    show: bool = True,
    point_size: float = BASE_POINT_SIZE,
    max_cells: int = MAX_PLOT_CELLS,
):
    coords, values = _build_marker_stack_data(
        adata=adata,
        markers=markers,
        coord_key=coord_key,
        use_log1p=True,
        max_cells=max_cells,
    )
    vmin, vmax = robust_vrange(values, 1, 99)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(6.9, 6.3), facecolor="white")
    xlim, ylim = _make_stack_extent(len(markers))

    # draw from back to front so the front panels fully cover the overlap region
    for draw_rank, i in enumerate(range(len(markers) - 1, -1, -1)):
        marker = markers[i]
        vals = values[i]
        dx = STACK_OFFSET_X * i
        dy = STACK_OFFSET_Y * i
        shifted = coords + np.array([dx, dy], dtype=np.float32)
        zbase = 10 + draw_rank * 3

        # opaque white panel background: prevents seeing the rear panel through overlap
        ax.add_patch(
            Rectangle(
                (dx, dy), CANVAS_SIZE, CANVAS_SIZE,
                facecolor="white",
                edgecolor="none",
                linewidth=0,
                zorder=zbase,
            )
        )

        rgba = plt.get_cmap(cmap)(norm(vals))
        vmax_panel = max(float(np.nanmax(vals)), 1e-6)
        rgba[:, -1] = np.clip(vals / vmax_panel, 0.05, 1.0)

        ax.scatter(
            shifted[:, 0], shifted[:, 1],
            c=rgba, s=point_size, linewidths=0, rasterized=True, zorder=zbase + 1,
        )
        ax.add_patch(
            Rectangle(
                (dx, dy), CANVAS_SIZE, CANVAS_SIZE,
                fill=False,
                edgecolor="0.25",
                linewidth=FRAME_LINEWIDTH,
                linestyle="--" if dashed_frame else "-",
                zorder=zbase + 2,
            )
        )
        # ax.text(
        #     dx + 18, dy + 32,
        #     marker,
        #     fontsize=10.5,
        #     fontweight="bold",
        #     ha="left", va="bottom",
        #     bbox=dict(facecolor="white", edgecolor="none", alpha=0.96, pad=0.5),
        #     zorder=zbase + 3,
        # )

    apply_axis_style(ax, xlim=xlim, ylim=ylim)
    # ax.set_title(title, fontsize=14, pad=10)

    # sm = ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    # cbar.set_label(colorbar_label, fontsize=9)
    # cbar.ax.tick_params(labelsize=8)

    save_or_show(fig, save_path=save_path, show=show)


# ============================================================
# Overlay callout plotting
# ============================================================

def plot_location_overlay_with_side_callout(
    ad_gene,
    ad_other,
    title: str,
    save_path: Optional[str],
    gene_coord_key: str = "spatial",
    other_coord_key: str = "spatial",
    gene_label: str = "Gene cells",
    other_label: str = "Protein cells",
    gene_color: str = "#B2182B",
    other_color: str = "#2166AC",
    gene_marker: str = "o",
    other_marker: str = "^",
    roi_bbox: Optional[Tuple[float, float, float, float]] = None,
    show: bool = True,
    max_cells: int = MAX_PLOT_CELLS,
):
    coords_gene, _, _ = prepare_coords(
        ad_gene, coord_key=gene_coord_key, valid_only=True, swap_xy=SWAP_XY, max_cells=max_cells
    )
    coords_other, _, _ = prepare_coords(
        ad_other, coord_key=other_coord_key, valid_only=True, swap_xy=SWAP_XY, max_cells=max_cells
    )

    coords_gene = normalize_coords_to_canvas(coords_gene, canvas_size=CANVAS_SIZE)
    coords_other = normalize_coords_to_canvas(coords_other, canvas_size=CANVAS_SIZE)

    if roi_bbox is None:
        roi_bbox = (430, 610, 430, 610)
    rx0, rx1, ry0, ry1 = roi_bbox

    fig = plt.figure(figsize=(13, 7.6), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1], wspace=-0.01)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])

    # main panel
    ax_main.scatter(
        coords_gene[:, 0], coords_gene[:, 1], s=1.5, c=gene_color,
        marker=gene_marker, alpha=0.66, linewidths=0, rasterized=True
    )
    ax_main.scatter(
        coords_other[:, 0], coords_other[:, 1], s=2, c=other_color,
        marker=other_marker, alpha=0.66, linewidths=0, rasterized=True
    )
    ax_main.add_patch(
        Rectangle((0, 0), CANVAS_SIZE, CANVAS_SIZE, fill=False, edgecolor="0.25", linewidth=1.4)
    )
    ax_main.add_patch(
        Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False,
                  edgecolor="black", linewidth=ROI_LINEWIDTH, linestyle="--")
    )
    apply_axis_style(ax_main, xlim=(-20, CANVAS_SIZE + 20), ylim=(CANVAS_SIZE + 20, -20))
    # ax_main.set_title(title, fontsize=13.5, pad=10)
    # ax_main.text(
    #     0.98, 0.985, "Full layout",
    #     transform=ax_main.transAxes,
    #     ha="right", va="top",
    #     fontsize=11.5, fontweight="bold",
    #     bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.4),
    # )

    # zoom panel on right
    gmask = (
        (coords_gene[:, 0] >= rx0) & (coords_gene[:, 0] <= rx1) &
        (coords_gene[:, 1] >= ry0) & (coords_gene[:, 1] <= ry1)
    )
    pmask = (
        (coords_other[:, 0] >= rx0) & (coords_other[:, 0] <= rx1) &
        (coords_other[:, 1] >= ry0) & (coords_other[:, 1] <= ry1)
    )
    ax_zoom.scatter(
        coords_gene[gmask, 0], coords_gene[gmask, 1], s=ZOOM_POINT_SIZE, c=gene_color,
        marker=gene_marker, alpha=0.82, linewidths=0, label=gene_label, rasterized=True
    )
    ax_zoom.scatter(
        coords_other[pmask, 0], coords_other[pmask, 1], s=ZOOM_POINT_SIZE + 4, c=other_color,
        marker=other_marker, alpha=0.82, linewidths=0, label=other_label, rasterized=True
    )
    ax_zoom.add_patch(
        Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False, edgecolor="0.25", linewidth=1.4)
    )
    apply_axis_style(ax_zoom, xlim=(rx0 - 10, rx1 + 10), ylim=(ry1 + 10, ry0 - 10))
    # ax_zoom.set_title("Callout: same-cell spatial correspondence", fontsize=12, pad=28)
    ax_zoom.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        fontsize=16,
        markerscale=1.0,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    # connectors: connect the right two ROI corners in the main panel
    # to the left two corners of the zoom-panel bounding box
    # X-axis is inverted:
    # visual right corners of ROI -> visual left corners of zoom panel

    fig.add_artist(
        ConnectionPatch(
            xyA=(rx0, ry0), coordsA=ax_main.transData,
            xyB=(rx1, ry0), coordsB=ax_zoom.transData,
            color="0.35", lw=1.1,
        )
    )

    fig.add_artist(
        ConnectionPatch(
            xyA=(rx0, ry1), coordsA=ax_main.transData,
            xyB=(rx1, ry1), coordsB=ax_zoom.transData,
            color="0.35", lw=1.1,
        )
    )
    save_or_show(fig, save_path=save_path, show=show)


# ============================================================
# Main wrapper
# ============================================================

def generate_cross_modality_pipeline_figures(
    real_gene_path: str,
    real_protein_path: str,
    virtual_protein_path: str,
    gene_markers: Sequence[str],
    protein_markers: Sequence[str],
    outdir: str,
    roi_bbox: Optional[Tuple[float, float, float, float]] = None,
    gene_coord_key: str = "spatial",
    real_protein_coord_key: str = "spatial",
    virtual_protein_coord_key: str = "spatial",
    show: bool = True,
):
    ensure_dir(outdir)

    ad_gene = sc.read_h5ad(real_gene_path)
    ad_real_protein = sc.read_h5ad(real_protein_path)
    ad_virtual_protein = sc.read_h5ad(virtual_protein_path)
    #
    plot_stacked_signal_panels(
        adata=ad_gene,
        markers=gene_markers,
        title="Figure 1. Measured gene tissue",
        cmap="Reds",
        colorbar_label="log1p gene signal",
        save_path=os.path.join(outdir, "figure1_real_gene_stack.png"),
        coord_key=gene_coord_key,
        dashed_frame=False,
        show=show,
    )

    plot_stacked_signal_panels(
        adata=ad_real_protein,
        markers=protein_markers,
        title="Figure 2. Measured protein tissue",
        cmap="Blues",
        colorbar_label="log1p protein signal",
        save_path=os.path.join(outdir, "figure2_real_protein_stack.png"),
        coord_key=real_protein_coord_key,
        dashed_frame=False,
        show=show,
    )

    plot_location_overlay_with_side_callout(
        ad_gene=ad_gene,
        ad_other=ad_real_protein,
        title="Figure 3. Measured gene and protein cell layouts",
        save_path=os.path.join(outdir, "figure3_real_gene_real_protein_overlay.png"),
        gene_coord_key=gene_coord_key,
        other_coord_key=real_protein_coord_key,
        gene_label="Gene cells",
        other_label="Protein cells",
        gene_color="#B2182B",
        other_color="#2166AC",
        gene_marker="o",
        other_marker="^",
        roi_bbox=roi_bbox,
        show=show,
    )

    plot_stacked_signal_panels(
        adata=ad_gene,
        markers=gene_markers,
        title="Figure 4. Measured gene tissue",
        cmap="Reds",
        colorbar_label="log1p gene signal",
        save_path=os.path.join(outdir, "figure4_real_gene_stack_for_virtual.png"),
        coord_key=gene_coord_key,
        dashed_frame=False,
        show=show,
    )

    plot_stacked_signal_panels(
        adata=ad_virtual_protein,
        markers=protein_markers,
        title="Figure 5. Gene-aligned virtual protein tissue",
        cmap="Blues",
        colorbar_label="log1p protein signal",
        save_path=os.path.join(outdir, "figure5_virtual_protein_stack.png"),
        coord_key=virtual_protein_coord_key,
        dashed_frame=True,
        show=show,
    )

    plot_location_overlay_with_side_callout(
        ad_gene=ad_gene,
        ad_other=ad_virtual_protein,
        title="Figure 6. Gene and virtual-protein cell layouts",
        save_path=os.path.join(outdir, "figure6_real_gene_virtual_protein_overlay.png"),
        gene_coord_key=gene_coord_key,
        other_coord_key=virtual_protein_coord_key,
        gene_label="Gene cells",
        other_label="Virtual protein cells",
        gene_color="#B2182B",
        other_color="#2166AC",
        gene_marker="o",
        other_marker="^",
        roi_bbox=roi_bbox,
        show=show,
    )

    return {
        "figure1": os.path.join(outdir, "figure1_real_gene_stack.png"),
        "figure2": os.path.join(outdir, "figure2_real_protein_stack.png"),
        "figure3": os.path.join(outdir, "figure3_real_gene_real_protein_overlay.png"),
        "figure4": os.path.join(outdir, "figure4_real_gene_stack_for_virtual.png"),
        "figure5": os.path.join(outdir, "figure5_virtual_protein_stack.png"),
        "figure6": os.path.join(outdir, "figure6_real_gene_virtual_protein_overlay.png"),
    }


if __name__ == "__main__":
    REAL_GENE = "/media/huifang/data/registration/phenocycler/huifang/h5ad/LUAD_2_A_xenium.h5ad"
    REAL_PROTEIN = "/media/huifang/data/registration/phenocycler/huifang/h5ad/LUAD_2_A_protein.h5ad"
    VIRTUAL_PROTEIN = "/media/huifang/data/registration/phenocycler/huifang/virtual_tissue/modality_LUAD_2_A/analysis/virtual_protein_1.000.h5ad"
    OUTDIR = "/media/huifang/data/registration/paper_analysis/Figure7_modality/pipeline/"

    GENE_MARKERS = ["HLA-A", "MKI67","HLA-DPB1"]
    PROTEIN_MARKERS = ["HLA-A", "Ki67","HLA-DPB1"]
    ROI_BBOX = (600, 630, 600, 630)

    generate_cross_modality_pipeline_figures(
        real_gene_path=REAL_GENE,
        real_protein_path=REAL_PROTEIN,
        virtual_protein_path=VIRTUAL_PROTEIN,
        gene_markers=GENE_MARKERS,
        protein_markers=PROTEIN_MARKERS,
        outdir=OUTDIR,
        roi_bbox=ROI_BBOX,
        gene_coord_key="spatial",
        real_protein_coord_key="spatial",
        virtual_protein_coord_key="spatial",
        show=True,
    )
