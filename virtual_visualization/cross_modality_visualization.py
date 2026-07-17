import os
import math
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle


# ============================================================
# Basic helpers
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_coords_npz(NET_IN, group_id, sample_id):
    path = os.path.join(NET_IN, "{}_{}_validation.npz".format(group_id, sample_id))
    d = np.load(path)
    return d["coord"].astype(np.float32)


def robust_vrange(vals_list, low_pct=1.0, high_pct=99.0):
    """
    Robust value range from a list of arrays.
    """
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
    """
    Use adata.obs['valid'] if available, otherwise all True.
    """
    if "valid" in adata.obs.columns:
        mask = np.asarray(adata.obs["valid"]).astype(bool)
        return mask
    return np.ones(adata.n_obs, dtype=bool)


def load_coords(adata, coord_key="spatial_axis", valid_only=True):
    if coord_key not in adata.obsm:
        raise KeyError(f"Coordinate key '{coord_key}' not found in adata.obsm")
    coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    if valid_only:
        mask = get_valid_mask(adata)
        coords = coords[mask]
        return coords, mask
    return coords, np.ones(adata.n_obs, dtype=bool)


def extract_feature_vals(adata, feature, use_log1p=True, valid_mask=None):
    """
    Extract feature values from:
    1) adata.var_names / adata.X
    2) adata.obs columns

    Returns 1D np.float32 array.
    """
    # from X / var_names
    if feature in adata.var_names:
        j = int(np.where(adata.var_names == feature)[0][0])
        v = adata.X[:, j]
        if sp.issparse(v):
            v = np.asarray(v.toarray()).ravel().astype(np.float32)
        else:
            v = np.asarray(v).ravel().astype(np.float32)

        if use_log1p:
            # only apply log1p when values are nonnegative
            if np.nanmin(v) >= 0:
                v = np.log1p(v)

    # from obs
    elif feature in adata.obs.columns:
        v = np.asarray(adata.obs[feature]).ravel()
        try:
            v = v.astype(np.float32)
        except Exception:
            raise ValueError(
                f"Feature '{feature}' exists in obs, but is not numeric."
            )

    else:
        raise ValueError(
            f"Feature '{feature}' not found in either adata.var_names or adata.obs.columns."
        )

    if valid_mask is not None:
        v = v[valid_mask]

    return np.asarray(v, dtype=np.float32)


def compute_bbox_from_coords(coords, pad_ratio=0.08):
    """
    Bounding box from coords: (xmin, xmax, ymin, ymax)
    """
    coords = np.asarray(coords, dtype=np.float32)
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)

    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= dx * pad_ratio
    xmax += dx * pad_ratio
    ymin -= dy * pad_ratio
    ymax += dy * pad_ratio
    return (xmin, xmax, ymin, ymax)


def union_bbox(bboxes):
    xs0 = [b[0] for b in bboxes]
    xs1 = [b[1] for b in bboxes]
    ys0 = [b[2] for b in bboxes]
    ys1 = [b[3] for b in bboxes]
    return (min(xs0), max(xs1), min(ys0), max(ys1))


def apply_bbox(ax, bbox):
    xmin, xmax, ymin, ymax = bbox
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def add_bbox_rect(ax, bbox, color="black", linestyle="--", linewidth=1.6):
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


def plot_signal_panel(
    ax,
    coords,
    vals,
    title,
    norm,
    cmap="viridis",
    s=8,
    invert_y=True,
    xlim=None,
    ylim=None,
    dashedline=False,
    add_colorbar=False,
    fig=None,
    cbar_label=None,
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

    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        bbox = compute_bbox_from_coords(coords, pad_ratio=0.05)
        apply_bbox(ax, bbox)

    if invert_y:
        ax.invert_yaxis()

    style_panel_frame(ax, dashed=dashedline)

    if add_colorbar and fig is not None:
        cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    return sca


def plot_location_panel(
    ax,
    coords,
    title,
    bbox=None,
    color="0.35",
    s=5,
    alpha=0.65,
    invert_y=True,
    xlim=None,
    ylim=None,
    dashedline=False,
):
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
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

    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        bb = compute_bbox_from_coords(coords, pad_ratio=0.05)
        apply_bbox(ax, bb)

    if bbox is not None:
        add_bbox_rect(ax, bbox, color="black", linestyle="--", linewidth=1.5)

    if invert_y:
        ax.invert_yaxis()

    style_panel_frame(ax, dashed=dashedline)


def plot_overlay_panel(
    ax,
    coords_gene,
    coords_other,
    title,
    bbox=None,
    invert_y=True,
    xlim=None,
    ylim=None,
    gene_color="#4d4d4d",
    other_color="#d95f02",
    gene_label="Xenium cells",
    other_label="Protein cells",
    gene_size=22,
    other_size=22,
    gene_alpha=0.70,
    other_alpha=0.85,
    dashedline=False,
    draw_legend=False,
):
    # gene: bigger hollow circle
    ax.scatter(
        coords_gene[:, 0],
        coords_gene[:, 1],
        s=gene_size,
        facecolors="none",
        edgecolors=gene_color,
        linewidths=0.7,
        alpha=gene_alpha,
        marker="o",
        label=gene_label,
        rasterized=True,
    )

    # protein: x marker
    ax.scatter(
        coords_other[:, 0],
        coords_other[:, 1],
        s=other_size,
        c=other_color,
        linewidths=0.8,
        alpha=other_alpha,
        marker="x",
        label=other_label,
        rasterized=True,
    )

    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        bb = union_bbox([
            compute_bbox_from_coords(coords_gene, pad_ratio=0.05),
            compute_bbox_from_coords(coords_other, pad_ratio=0.05),
        ])
        apply_bbox(ax, bb)

    if bbox is not None:
        add_bbox_rect(ax, bbox, color="black", linestyle="--", linewidth=1.5)

    if invert_y:
        ax.invert_yaxis()

    style_panel_frame(ax, dashed=dashedline)

    if draw_legend:
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=9,
            handletextpad=0.4,
            borderpad=0.2,
        )


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
    protein_point_size=8,
    invert_y=True,
    add_panel_colorbars=True,
):
    coords_gene, mask_gene = load_coords(ad_gene, gene_coord_key)

    coords_real_p, mask_real_p = load_coords(ad_real_protein, real_protein_coord_key)


    coords_virtual_p, mask_virtual_p = load_coords(ad_virtual_protein, virtual_protein_coord_key, valid_only=True)

    ncols = max(len(gene_names), len(protein_names))
    fig, axes = plt.subplots(3, ncols, figsize=(4.0 * ncols, 10.5), squeeze=False)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.04, wspace=0.30, hspace=0.18)

    # global extents for each modality
    gene_bbox = compute_bbox_from_coords(coords_gene, pad_ratio=0.05)
    realp_bbox = compute_bbox_from_coords(coords_real_p, pad_ratio=0.05)
    virtp_bbox = compute_bbox_from_coords(coords_virtual_p, pad_ratio=0.05)

    # row 1: real gene maps
    for c in range(ncols):
        ax = axes[0, c]
        if c < len(gene_names):
            g = gene_names[c]
            vals_gene = extract_feature_vals(ad_gene, g, use_log1p=gene_use_log1p, valid_mask=mask_gene)
            vmin, vmax = robust_vrange([vals_gene], low_pct=1, high_pct=99)
            norm = Normalize(vmin=vmin, vmax=vmax)


            plot_signal_panel(
                ax=ax,
                coords=coords_gene,
                vals=vals_gene,
                title=f"Real gene\n{g}",
                norm=norm,
                cmap=gene_cmap,
                s=gene_point_size,
                invert_y=invert_y,
                xlim=(gene_bbox[0], gene_bbox[1]),
                ylim=(gene_bbox[2], gene_bbox[3]),
                dashedline=False,
                add_colorbar=add_panel_colorbars,
                fig=fig,
                cbar_label=g,
            )
        else:
            ax.axis("off")

    # row 2: real protein maps
    for c in range(ncols):
        ax = axes[1, c]
        if c < len(protein_names):
            p = protein_names[c]
            vals_real = extract_feature_vals(ad_real_protein, p, use_log1p=protein_use_log1p, valid_mask=mask_real_p)
            vals_virtual = extract_feature_vals(ad_virtual_protein, p, use_log1p=protein_use_log1p, valid_mask=mask_virtual_p)
            vmin, vmax = robust_vrange([vals_real, vals_virtual], low_pct=1, high_pct=99)
            norm = Normalize(vmin=vmin, vmax=vmax)

            plot_signal_panel(
                ax=ax,
                coords=coords_real_p,
                vals=vals_real,
                title=f"Real protein\n{p}",
                norm=norm,
                cmap=protein_cmap,
                s=protein_point_size,
                invert_y=invert_y,
                xlim=(realp_bbox[0], realp_bbox[1]),
                ylim=(realp_bbox[2], realp_bbox[3]),
                dashedline=False,
                add_colorbar=add_panel_colorbars,
                fig=fig,
                cbar_label=p,
            )
        else:
            ax.axis("off")

    # row 3: virtual protein maps
    for c in range(ncols):
        ax = axes[2, c]
        if c < len(protein_names):
            p = protein_names[c]
            vals_real = extract_feature_vals(ad_real_protein, p, use_log1p=protein_use_log1p, valid_mask=mask_real_p)
            vals_virtual = extract_feature_vals(ad_virtual_protein, p, use_log1p=protein_use_log1p, valid_mask=mask_virtual_p)
            vmin, vmax = robust_vrange([vals_real, vals_virtual], low_pct=1, high_pct=99)
            norm = Normalize(vmin=vmin, vmax=vmax)

            plot_signal_panel(
                ax=ax,
                coords=coords_virtual_p,
                vals=vals_virtual,
                title=f"Virtual protein\n{p}",
                norm=norm,
                cmap=protein_cmap,
                s=protein_point_size,
                invert_y=invert_y,
                xlim=(virtp_bbox[0], virtp_bbox[1]),
                ylim=(virtp_bbox[2], virtp_bbox[3]),
                dashedline=True,
                add_colorbar=add_panel_colorbars,
                fig=fig,
                cbar_label=p,
            )
        else:
            ax.axis("off")

    # row labels
    axes[0, 0].set_ylabel("Xenium", fontsize=12)
    axes[1, 0].set_ylabel("Real protein", fontsize=12)
    axes[2, 0].set_ylabel("Virtual protein", fontsize=12)

    fig.suptitle("Figure 1. Cross-modality signal maps", fontsize=14, y=0.995)

    if out_png is not None:
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
    ad_real_protein,
    ad_virtual_protein=None,
    out_png=None,
    gene_coord_key="spatial",
    real_protein_coord_key="spatial",
    virtual_protein_coord_key="spatial",
    invert_y=True,
    roi_source="union_protein",
    point_size=5,
):
    coords_gene, _ = load_coords(ad_gene, gene_coord_key, valid_only=True)
    coords_real_p, _ = load_coords(ad_real_protein, real_protein_coord_key, valid_only=True)

    if ad_virtual_protein is not None:
        coords_virtual_p, _ = load_coords(ad_virtual_protein, virtual_protein_coord_key, valid_only=True)
    else:
        coords_virtual_p = None

    gene_bbox = compute_bbox_from_coords(coords_gene, pad_ratio=0.05)

    if roi_source == "real_protein":
        roi_bbox = compute_bbox_from_coords(coords_real_p, pad_ratio=0.08)
    elif roi_source == "virtual_protein" and coords_virtual_p is not None:
        roi_bbox = compute_bbox_from_coords(coords_virtual_p, pad_ratio=0.08)
    else:
        bbs = [compute_bbox_from_coords(coords_real_p, pad_ratio=0.08)]
        if coords_virtual_p is not None:
            bbs.append(compute_bbox_from_coords(coords_virtual_p, pad_ratio=0.08))
        roi_bbox = union_bbox(bbs)

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))
    plot_location_panel(
        ax=ax,
        coords=coords_gene,
        title="Figure 2. Xenium cell locations\nDashed box = queried protein region",
        bbox=roi_bbox,
        color="0.35",
        s=point_size,
        alpha=0.65,
        invert_y=invert_y,
        xlim=(gene_bbox[0], gene_bbox[1]),
        ylim=(gene_bbox[2], gene_bbox[3]),
        dashedline=False,
    )

    if out_png is not None:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved: {out_png}")
    else:
        plt.show()

    return roi_bbox


# ============================================================
# Figure 3
# ============================================================

def plot_location_overlays(
    ad_gene,
    ad_real_protein,
    ad_virtual_protein,
    out_png=None,
    gene_coord_key="spatial",
    real_protein_coord_key="spatial",
    virtual_protein_coord_key="spatial",
    invert_y=True,
    roi_bbox=None,
    gene_color="#4d4d4d",
    real_protein_color="#2878B5",
    virtual_protein_color="#d95f02",
    gene_size=22,
    protein_size=22,
):
    coords_gene, _ = load_coords(ad_gene, gene_coord_key, valid_only=True)
    coords_real_p, _ = load_coords(ad_real_protein, real_protein_coord_key, valid_only=True)
    coords_virtual_p, _ = load_coords(ad_virtual_protein, virtual_protein_coord_key, valid_only=True)

    full_bbox = union_bbox([
        compute_bbox_from_coords(coords_gene, pad_ratio=0.05),
        compute_bbox_from_coords(coords_real_p, pad_ratio=0.05),
        compute_bbox_from_coords(coords_virtual_p, pad_ratio=0.05),
    ])

    if roi_bbox is None:
        roi_bbox = union_bbox([
            compute_bbox_from_coords(coords_real_p, pad_ratio=0.08),
            compute_bbox_from_coords(coords_virtual_p, pad_ratio=0.08),
        ])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), squeeze=False)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.05, wspace=0.15, hspace=0.18)

    # full: gene + real protein
    plot_overlay_panel(
        ax=axes[0, 0],
        coords_gene=coords_gene,
        coords_other=coords_real_p,
        title="Real gene + real protein (full view)",
        bbox=roi_bbox,
        invert_y=invert_y,
        xlim=(full_bbox[0], full_bbox[1]),
        ylim=(full_bbox[2], full_bbox[3]),
        gene_color=gene_color,
        other_color=real_protein_color,
        gene_label="Xenium cells",
        other_label="Real protein cells",
        gene_size=gene_size,
        other_size=protein_size,
        draw_legend=True,
    )

    # zoom: gene + real protein
    plot_overlay_panel(
        ax=axes[0, 1],
        coords_gene=coords_gene,
        coords_other=coords_real_p,
        title="ROI zoom: real gene + real protein",
        bbox=None,
        invert_y=invert_y,
        xlim=(roi_bbox[0], roi_bbox[1]),
        ylim=(roi_bbox[2], roi_bbox[3]),
        gene_color=gene_color,
        other_color=real_protein_color,
        gene_label="Xenium cells",
        other_label="Real protein cells",
        gene_size=gene_size,
        other_size=protein_size,
        draw_legend=False,
    )

    # full: gene + virtual protein
    plot_overlay_panel(
        ax=axes[1, 0],
        coords_gene=coords_gene,
        coords_other=coords_virtual_p,
        title="Real gene + virtual protein (full view)",
        bbox=roi_bbox,
        invert_y=invert_y,
        xlim=(full_bbox[0], full_bbox[1]),
        ylim=(full_bbox[2], full_bbox[3]),
        gene_color=gene_color,
        other_color=virtual_protein_color,
        gene_label="Xenium cells",
        other_label="Virtual protein cells",
        gene_size=gene_size,
        other_size=protein_size,
        draw_legend=True,
    )

    # zoom: gene + virtual protein
    plot_overlay_panel(
        ax=axes[1, 1],
        coords_gene=coords_gene,
        coords_other=coords_virtual_p,
        title="ROI zoom: real gene + virtual protein",
        bbox=None,
        invert_y=invert_y,
        xlim=(roi_bbox[0], roi_bbox[1]),
        ylim=(roi_bbox[2], roi_bbox[3]),
        gene_color=gene_color,
        other_color=virtual_protein_color,
        gene_label="Xenium cells",
        other_label="Virtual protein cells",
        gene_size=gene_size,
        other_size=protein_size,
        draw_legend=False,
    )

    fig.suptitle("Figure 3. Cell-location overlays across modalities", fontsize=14, y=0.99)

    if out_png is not None:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved: {out_png}")
    else:
        plt.show()


# ============================================================
# Main wrapper
# ============================================================

def generate_cross_modality_figures(
    real_xenium_path,
    real_protein_path,
    virtual_protein_path,
    gene_names,
    protein_names,
    outdir,
    gene_coord_key="spatial",
    real_protein_coord_key="spatial",
    virtual_protein_coord_key="spatial",
    gene_use_log1p=True,
    protein_use_log1p=True,
    gene_cmap="Reds",
    protein_cmap="viridis",
    invert_y=True,
):
    ensure_dir(outdir)

    ad_gene = sc.read_h5ad(real_xenium_path)
    ad_real_protein = sc.read_h5ad(real_protein_path)
    ad_virtual_protein = sc.read_h5ad(virtual_protein_path)

    # Figure 1
    fig1_path = os.path.join(outdir, "figure1_cross_modality_maps.png")
    plot_cross_modality_maps(
        ad_gene=ad_gene,
        ad_real_protein=ad_real_protein,
        ad_virtual_protein=ad_virtual_protein,
        gene_names=gene_names,
        protein_names=protein_names,
        out_png=fig1_path,
        gene_coord_key=gene_coord_key,
        real_protein_coord_key=real_protein_coord_key,
        virtual_protein_coord_key=virtual_protein_coord_key,
        gene_use_log1p=gene_use_log1p,
        protein_use_log1p=protein_use_log1p,
        gene_cmap=gene_cmap,
        protein_cmap=protein_cmap,
        gene_point_size=0.1,
        protein_point_size=0.05,
        invert_y=invert_y,
        add_panel_colorbars=True,
    )

    # Figure 2
    fig2_path = os.path.join(outdir, "figure2_xenium_query_region.png")
    roi_bbox = plot_query_region_on_xenium(
        ad_gene=ad_gene,
        ad_real_protein=ad_real_protein,
        ad_virtual_protein=ad_virtual_protein,
        out_png=fig2_path,
        gene_coord_key=gene_coord_key,
        real_protein_coord_key=real_protein_coord_key,
        virtual_protein_coord_key=virtual_protein_coord_key,
        invert_y=invert_y,
        roi_source="union_protein",
        point_size=0.1,
    )

    # Figure 3
    fig3_path = os.path.join(outdir, "figure3_location_overlays.png")
    plot_location_overlays(
        ad_gene=ad_gene,
        ad_real_protein=ad_real_protein,
        ad_virtual_protein=ad_virtual_protein,
        out_png=fig3_path,
        gene_coord_key=gene_coord_key,
        real_protein_coord_key=real_protein_coord_key,
        virtual_protein_coord_key=virtual_protein_coord_key,
        invert_y=invert_y,
        roi_bbox=roi_bbox,
        gene_color="#4d4d4d",
        real_protein_color="#2878B5",
        virtual_protein_color="#d95f02",
        gene_size=1,
        protein_size=1,
    )

    print("[DONE]")
    print(" Figure1:", fig1_path)
    print(" Figure2:", fig2_path)
    print(" Figure3:", fig3_path)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    REAL_XENIUM = "/media/huifang/data/registration/phenocycler/huifang/h5ad/LUAD_2_A_xenium.h5ad"
    REAL_PROTEIN = "/media/huifang/data/registration/phenocycler/huifang/h5ad/LUAD_2_A_protein.h5ad"
    VIRTUAL_PROTEIN = "/media/huifang/data/registration/phenocycler/huifang/virtual_tissue/modality_LUAD_2_A/analysis/virtual_protein_1.000.h5ad"

    OUTDIR = "/media/huifang/data/registration/phenocycler/huifang/virtual_tissue/modality_LUAD_2_A/analysis/figuers"

    ROOT = "/media/huifang/data/registration/phenocycler/huifang"
    NET_IN = os.path.join(ROOT, "network_input")
    H5AD_DIR = os.path.join(ROOT, "h5ad")

    # Example:
    GENE_NAMES = ["ACTA2", "CD44"]
    PROTEIN_NAMES = ["IDO1","HLA-A"]

    generate_cross_modality_figures(
        real_xenium_path=REAL_XENIUM,
        real_protein_path=REAL_PROTEIN,
        virtual_protein_path=VIRTUAL_PROTEIN,
        gene_names=GENE_NAMES,
        protein_names=PROTEIN_NAMES,
        outdir=OUTDIR,
        gene_coord_key="spatial",
        real_protein_coord_key="spatial",
        virtual_protein_coord_key="spatial",
        gene_use_log1p=True,
        protein_use_log1p=True,
        gene_cmap="Reds",
        protein_cmap="Blues",
        invert_y=True,
    )


