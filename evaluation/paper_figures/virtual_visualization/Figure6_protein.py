import os
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


# ============================================================
# Basic helpers
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def robust_vrange(vals_list, low_pct=1.0, high_pct=99.0):
    arrs = []
    for v in vals_list:
        if v is None:
            continue
        v = np.asarray(v).ravel()
        v = v[np.isfinite(v)]
        if len(v) > 0:
            arrs.append(v)

    if len(arrs) == 0:
        return 0.0, 1.0

    arr = np.concatenate(arrs)
    vmin = np.percentile(arr, low_pct)
    vmax = np.percentile(arr, high_pct)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def get_valid_mask(adata):
    if "valid" in adata.obs.columns:
        return np.asarray(adata.obs["valid"]).astype(bool)
    return np.ones(adata.n_obs, dtype=bool)


def load_coords(adata, coord_key="spatial", valid_only=True):
    if coord_key not in adata.obsm:
        raise KeyError(f"Coordinate key '{coord_key}' not found in adata.obsm")
    coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    if valid_only:
        mask = get_valid_mask(adata)
        return coords[mask], mask
    return coords, np.ones(adata.n_obs, dtype=bool)


def extract_feature_vals(adata, feature, use_log1p=True, valid_mask=None):
    if feature in adata.var_names:
        j = int(np.where(adata.var_names == feature)[0][0])
        v = adata.X[:, j]
        if sp.issparse(v):
            v = np.asarray(v.toarray()).ravel().astype(np.float32)
        else:
            v = np.asarray(v).ravel().astype(np.float32)

        if use_log1p and np.nanmin(v) >= 0:
            v = np.log1p(v)

    elif feature in adata.obs.columns:
        v = np.asarray(adata.obs[feature]).ravel()
        v = v.astype(np.float32)

    else:
        raise ValueError(f"Feature '{feature}' not found in var_names or obs.")

    if valid_mask is not None:
        v = v[valid_mask]

    return np.asarray(v, dtype=np.float32)


def compute_bbox_from_coords(coords, pad_ratio=0.05):
    coords = np.asarray(coords, dtype=np.float32)
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    return (
        xmin - dx * pad_ratio,
        xmax + dx * pad_ratio,
        ymin - dy * pad_ratio,
        ymax + dy * pad_ratio,
    )


def union_bbox(bboxes):
    return (
        min([b[0] for b in bboxes]),
        max([b[1] for b in bboxes]),
        min([b[2] for b in bboxes]),
        max([b[3] for b in bboxes]),
    )


def apply_bbox(ax, bbox):
    xmin, xmax, ymin, ymax = bbox
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def add_bbox_rect(ax, bbox, color="black", linestyle="--", linewidth=1.5):
    xmin, xmax, ymin, ymax = bbox
    rect = Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        edgecolor=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    ax.add_patch(rect)


def style_panel_frame(ax, dashed=False):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("0.35")
        if dashed:
            spine.set_linestyle((0, (4, 4)))


# ============================================================
# Panel plotting
# ============================================================

def plot_signal_panel(
    ax,
    coords,
    vals,
    title,
    norm,
    cmap="viridis",
    s=8,
    invert_y=True,
    bbox=None,
    dashedline=False,
):
    sca = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=vals,
        s=s,
        cmap=cmap,
        norm=norm,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    if bbox is None:
        bbox = compute_bbox_from_coords(coords, pad_ratio=0.05)
    apply_bbox(ax, bbox)

    if invert_y:
        ax.invert_yaxis()

    style_panel_frame(ax, dashed=dashedline)
    return sca


def plot_location_panel(
    ax,
    coords,
    title,
    full_bbox,
    color="0.35",
    s=5,
    alpha=0.65,
    invert_y=True,
    dashedline=False,
):
    ax.scatter(
        coords[::100, 0],
        coords[::100, 1],
        s=s,
        c=color,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    apply_bbox(ax, full_bbox)

    if invert_y:
        ax.invert_yaxis()

    style_panel_frame(ax, dashed=dashedline)


def plot_overlay_panel(
    ax,
    coords_gene,
    coords_other,
    title,
    bbox,
    roi_bbox=None,
    invert_y=True,
    gene_color="#4d4d4d",
    other_color="#d95f02",
    gene_label="Xenium cells",
    other_label="Protein cells",
    gene_size=24,
    other_size=24,
    gene_alpha=0.7,
    other_alpha=0.7,
    dashedline=True,
    draw_legend=True,
):
    ax.scatter(
        coords_gene[:, 0],
        coords_gene[:, 1],
        s=gene_size,
        facecolors="none",
        edgecolors=gene_color,
        linewidths=2,
        alpha=gene_alpha,
        marker="o",
        label=gene_label,
        rasterized=True,
    )

    ax.scatter(
        coords_other[:, 0],
        coords_other[:, 1],
        s=other_size,
        c=other_color,
        linewidths=2,
        alpha=other_alpha,
        marker="x",
        label=other_label,
        rasterized=True,
    )

    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    apply_bbox(ax, bbox)

    if roi_bbox is not None:
        add_bbox_rect(ax, roi_bbox, color="black", linestyle="--", linewidth=1.4)

    if invert_y:
        ax.invert_yaxis()

    style_panel_frame(ax, dashed=dashedline)

    if draw_legend:
        ax.legend(loc="upper right", frameon=False, fontsize=9, handletextpad=0.4)


# ============================================================
# Figure 1
# ============================================================

def plot_cross_modality_maps(
    ad_gene,
    ad_real_protein,
    ad_virtual_protein,
    gene_names,
    protein_names,
    out_png=None,
    gene_coord_key="spatial",
    real_protein_coord_key="spatial",
    virtual_protein_coord_key="spatial",
    gene_use_log1p=True,
    protein_use_log1p=True,
    gene_cmap="Reds",
    protein_cmap="viridis",
    gene_point_size=8,
    protein_point_size=10,
    invert_y=True,
    gene_bbox=None,
    real_protein_bbox=None,
    virtual_protein_bbox=None,
):
    coords_gene, mask_gene = load_coords(ad_gene, gene_coord_key, valid_only=True)
    coords_real_p, mask_real_p = load_coords(ad_real_protein, real_protein_coord_key, valid_only=True)
    coords_virtual_p, mask_virtual_p = load_coords(ad_virtual_protein, virtual_protein_coord_key, valid_only=True)

    if gene_bbox is None:
        gene_bbox = compute_bbox_from_coords(coords_gene, pad_ratio=0.05)
    if real_protein_bbox is None:
        real_protein_bbox = compute_bbox_from_coords(coords_real_p, pad_ratio=0.05)
    if virtual_protein_bbox is None:
        virtual_protein_bbox = compute_bbox_from_coords(coords_virtual_p, pad_ratio=0.05)

    ncols = max(len(gene_names), len(protein_names))
    fig, axes = plt.subplots(3, ncols, figsize=(4.0 * ncols, 10.5), squeeze=False)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.04, wspace=0.30, hspace=0.18)

    # row 1: gene
    for c in range(ncols):
        ax = axes[0, c]
        if c < len(gene_names):
            g = gene_names[c]
            vals = extract_feature_vals(ad_gene, g, use_log1p=gene_use_log1p, valid_mask=mask_gene)
            vmin, vmax = robust_vrange([vals], 1, 99)
            norm = Normalize(vmin=vmin, vmax=vmax)
            sca = plot_signal_panel(
                ax=ax,
                coords=coords_gene,
                vals=vals,
                title=f"Real Xenium\n{g}",
                norm=norm,
                cmap=gene_cmap,
                s=gene_point_size,
                invert_y=invert_y,
                bbox=gene_bbox,
                dashedline=False,
            )
            cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(g, fontsize=9)
            cbar.ax.tick_params(labelsize=8)
        else:
            ax.axis("off")

    # row 2: real protein
    for c in range(ncols):
        ax = axes[1, c]
        if c < len(protein_names):
            p = protein_names[c]
            vals_real = extract_feature_vals(ad_real_protein, p, use_log1p=protein_use_log1p, valid_mask=mask_real_p)
            vals_virtual = extract_feature_vals(ad_virtual_protein, p, use_log1p=protein_use_log1p, valid_mask=mask_virtual_p)
            vmin, vmax = robust_vrange([vals_real, vals_virtual], 1, 99)
            norm = Normalize(vmin=vmin, vmax=vmax)
            sca = plot_signal_panel(
                ax=ax,
                coords=coords_real_p,
                vals=vals_real,
                title=f"Real protein\n{p}",
                norm=norm,
                cmap=protein_cmap,
                s=protein_point_size,
                invert_y=invert_y,
                bbox=real_protein_bbox,
                dashedline=False,
            )
            cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(p, fontsize=9)
            cbar.ax.tick_params(labelsize=8)
        else:
            ax.axis("off")

    # row 3: virtual protein
    for c in range(ncols):
        ax = axes[2, c]
        if c < len(protein_names):
            p = protein_names[c]
            vals_real = extract_feature_vals(ad_real_protein, p, use_log1p=protein_use_log1p, valid_mask=mask_real_p)
            vals_virtual = extract_feature_vals(ad_virtual_protein, p, use_log1p=protein_use_log1p, valid_mask=mask_virtual_p)
            vmin, vmax = robust_vrange([vals_real, vals_virtual], 1, 99)
            norm = Normalize(vmin=vmin, vmax=vmax)
            sca = plot_signal_panel(
                ax=ax,
                coords=coords_virtual_p,
                vals=vals_virtual,
                title=f"Virtual protein\n{p}",
                norm=norm,
                cmap=protein_cmap,
                s=protein_point_size,
                invert_y=invert_y,
                bbox=virtual_protein_bbox,
                dashedline=True,
            )
            cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(p, fontsize=9)
            cbar.ax.tick_params(labelsize=8)
        else:
            ax.axis("off")

    axes[0, 0].set_ylabel("Xenium", fontsize=12)
    axes[1, 0].set_ylabel("Real protein", fontsize=12)
    axes[2, 0].set_ylabel("Virtual protein", fontsize=12)

    fig.suptitle("Figure 1. Cross-modality signal maps", fontsize=14, y=0.995)

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved: {out_png}")
    else:
        plt.show()


# ============================================================
# Figure 2
# ============================================================

def plot_query_region_on_xenium(
    ad_gene,
    out_png=None,
    gene_coord_key="spatial",
    invert_y=True,
    point_size=5,
    full_bbox=None,
):
    coords_gene, _ = load_coords(ad_gene, gene_coord_key, valid_only=True)

    if full_bbox is None:
        full_bbox = compute_bbox_from_coords(coords_gene, pad_ratio=0.05)

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))
    plot_location_panel(
        ax=ax,
        coords=coords_gene,
        title="Figure 2. Xenium cell locations\nDashed box = manually specified ROI",
        full_bbox=full_bbox,
        color="0.35",
        s=point_size,
        alpha=0.65,
        invert_y=invert_y,
        dashedline=False,
    )

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved: {out_png}")
    else:
        plt.show()


# ============================================================
# Figure 3
# ============================================================

def plot_location_overlays(
    ad_gene,
    ad_real_protein,
    ad_virtual_protein,
    roi_bbox,
    out_png=None,
    gene_coord_key="spatial",
    real_protein_coord_key="spatial",
    virtual_protein_coord_key="spatial",
    invert_y=True,
    full_bbox=None,
    gene_color="#4d4d4d",
    real_protein_color="#2878B5",
    virtual_protein_color="#d95f02",
    gene_size=24,
    protein_size=24,
):
    coords_gene, _ = load_coords(ad_gene, gene_coord_key, valid_only=True)
    coords_real_p, _ = load_coords(ad_real_protein, real_protein_coord_key, valid_only=True)
    coords_virtual_p, _ = load_coords(ad_virtual_protein, virtual_protein_coord_key, valid_only=True)

    if full_bbox is None:
        full_bbox = union_bbox([
            compute_bbox_from_coords(coords_gene, 0.05),
            compute_bbox_from_coords(coords_real_p, 0.05),
            compute_bbox_from_coords(coords_virtual_p, 0.05),
        ])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), squeeze=False)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.05, wspace=0.15, hspace=0.18)

    # plot_overlay_panel(
    #     ax=axes[0, 0],
    #     coords_gene=coords_gene,
    #     coords_other=coords_real_p,
    #     title="Real gene + real protein (full view)",
    #     bbox=full_bbox,
    #     roi_bbox=roi_bbox,
    #     invert_y=invert_y,
    #     gene_color=gene_color,
    #     other_color=real_protein_color,
    #     gene_label="Xenium cells",
    #     other_label="Real protein cells",
    #     gene_size=gene_size,
    #     other_size=protein_size,
    #     draw_legend=True,
    # )

    plot_overlay_panel(
        ax=axes[0, 1],
        coords_gene=coords_gene,
        coords_other=coords_real_p,
        title="ROI zoom: real gene + real protein",
        bbox=roi_bbox,
        roi_bbox=None,
        invert_y=invert_y,
        gene_color=gene_color,
        other_color=real_protein_color,
        gene_label="Xenium cells",
        other_label="Real protein cells",
        gene_size=gene_size,
        other_size=protein_size,
        draw_legend=True,
    )

    # plot_overlay_panel(
    #     ax=axes[1, 0],
    #     coords_gene=coords_gene,
    #     coords_other=coords_virtual_p,
    #     title="Real gene + virtual protein (full view)",
    #     bbox=full_bbox,
    #     roi_bbox=roi_bbox,
    #     invert_y=invert_y,
    #     gene_color=gene_color,
    #     other_color=virtual_protein_color,
    #     gene_label="Xenium cells",
    #     other_label="Virtual protein cells",
    #     gene_size=gene_size,
    #     other_size=protein_size,
    #     draw_legend=True,
    # )

    plot_overlay_panel(
        ax=axes[1, 1],
        coords_gene=coords_gene,
        coords_other=coords_virtual_p,
        title="ROI zoom: real gene + virtual protein",
        bbox=roi_bbox,
        roi_bbox=None,
        invert_y=invert_y,
        gene_color=gene_color,
        other_color=virtual_protein_color,
        gene_label="Xenium cells",
        other_label="Virtual protein cells",
        gene_size=gene_size,
        other_size=protein_size,
        draw_legend=True,
    )

    fig.suptitle("Figure 3. Cell-location overlays across modalities", fontsize=14, y=0.99)

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved: {out_png}")
    else:
        plt.show()


# ============================================================
# Wrapper
# ============================================================

def generate_cross_modality_figures(
    real_xenium_path,
    real_protein_path,
    virtual_protein_path,
    gene_names,
    protein_names,
    outdir,
    roi_bbox,   # (xmin, xmax, ymin, ymax)
    gene_coord_key="spatial",
    real_protein_coord_key="spatial",
    virtual_protein_coord_key="spatial",
    gene_use_log1p=True,
    protein_use_log1p=True,
    gene_cmap="Reds",
    protein_cmap="viridis",
    invert_y=True,
    gene_bbox=None,
    real_protein_bbox=None,
    virtual_protein_bbox=None,
    overlay_full_bbox=None,
):
    ensure_dir(outdir)

    ad_gene = sc.read_h5ad(real_xenium_path)
    ad_real_protein = sc.read_h5ad(real_protein_path)
    ad_virtual_protein = sc.read_h5ad(virtual_protein_path)

    coords_gene, _ = load_coords(ad_gene, gene_coord_key, valid_only=True)
    coords_real_p, _ = load_coords(ad_real_protein, real_protein_coord_key, valid_only=True)
    coords_virtual_p, _ = load_coords(ad_virtual_protein, virtual_protein_coord_key, valid_only=True)

    if gene_bbox is None:
        gene_bbox = compute_bbox_from_coords(coords_gene, pad_ratio=0.05)
    if real_protein_bbox is None:
        real_protein_bbox = compute_bbox_from_coords(coords_real_p, pad_ratio=0.05)
    if virtual_protein_bbox is None:
        virtual_protein_bbox = compute_bbox_from_coords(coords_virtual_p, pad_ratio=0.05)
    if overlay_full_bbox is None:
        overlay_full_bbox = union_bbox([
            compute_bbox_from_coords(coords_gene, 0.05),
            compute_bbox_from_coords(coords_real_p, 0.05),
            compute_bbox_from_coords(coords_virtual_p, 0.05),
        ])

    fig1_path = os.path.join(outdir, "figure1_cross_modality_maps.png")
    # plot_cross_modality_maps(
    #     ad_gene=ad_gene,
    #     ad_real_protein=ad_real_protein,
    #     ad_virtual_protein=ad_virtual_protein,
    #     gene_names=gene_names,
    #     protein_names=protein_names,
    #     out_png=fig1_path,
    #     gene_coord_key=gene_coord_key,
    #     real_protein_coord_key=real_protein_coord_key,
    #     virtual_protein_coord_key=virtual_protein_coord_key,
    #     gene_use_log1p=gene_use_log1p,
    #     protein_use_log1p=protein_use_log1p,
    #     gene_cmap=gene_cmap,
    #     protein_cmap=protein_cmap,
    #     gene_point_size=0.1,
    #     protein_point_size=0.05,
    #     invert_y=invert_y,
    #     gene_bbox=gene_bbox,
    #     real_protein_bbox=real_protein_bbox,
    #     virtual_protein_bbox=virtual_protein_bbox,
    # )

    fig2_path = os.path.join(outdir, "figure2_xenium_query_region.png")
    plot_query_region_on_xenium(
        ad_gene=ad_gene,
        out_png=fig2_path,
        gene_coord_key=gene_coord_key,
        invert_y=invert_y,
        point_size=5,
        full_bbox=gene_bbox,
    )

    fig3_path = os.path.join(outdir, "figure3_location_overlays.png")
    plot_location_overlays(
        ad_gene=ad_gene,
        ad_real_protein=ad_real_protein,
        ad_virtual_protein=ad_virtual_protein,
        roi_bbox=roi_bbox,
        out_png=fig3_path,
        gene_coord_key=gene_coord_key,
        real_protein_coord_key=real_protein_coord_key,
        virtual_protein_coord_key=virtual_protein_coord_key,
        invert_y=invert_y,
        full_bbox=overlay_full_bbox,
        gene_color="red",
        real_protein_color="blue",
        virtual_protein_color="blue",
        gene_size=50,
        protein_size=50,
    )

    print("[DONE]")
    print("Figure1:", fig1_path)
    print("Figure2:", fig2_path)
    print("Figure3:", fig3_path)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    REAL_XENIUM = "/media/huifang/data/registration/phenocycler/huifang/h5ad/LUAD_2_A_xenium.h5ad"
    REAL_PROTEIN = "/media/huifang/data/registration/phenocycler/huifang/h5ad/LUAD_2_A_protein.h5ad"
    VIRTUAL_PROTEIN = "/media/huifang/data/registration/phenocycler/huifang/virtual_tissue/modality_LUAD_2_A/analysis/virtual_protein_1.000.h5ad"

    OUTDIR = "/media/huifang/data/registration/phenocycler/huifang/virtual_tissue/modality_LUAD_2_A/analysis/figuers"


    GENE_NAMES = ["ACTA2", "CD44"]
    PROTEIN_NAMES = ["IDO1","HLA-A"]

    # manually specified ROI
    ROI_BBOX = (695, 705, 390,400)

    generate_cross_modality_figures(
        real_xenium_path=REAL_XENIUM,
        real_protein_path=REAL_PROTEIN,
        virtual_protein_path=VIRTUAL_PROTEIN,
        gene_names=GENE_NAMES,
        protein_names=PROTEIN_NAMES,
        outdir=OUTDIR,
        roi_bbox=ROI_BBOX,
        gene_coord_key="spatial",
        real_protein_coord_key="spatial",
        virtual_protein_coord_key="spatial",
        gene_use_log1p=True,
        protein_use_log1p=True,
        gene_cmap="Reds",
        protein_cmap="viridis",
        invert_y=True,
    )