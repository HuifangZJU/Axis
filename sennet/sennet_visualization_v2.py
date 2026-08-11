import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import sparse
from matplotlib.lines import Line2D

def get_feature_values(adata, feature_name):
    """Extract one feature from AnnData as a 1D NumPy array."""
    values = adata[:, feature_name].X
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values).reshape(-1)


def normalize_values(values):
    """Normalize values to [0, 1] for plotting, using log1p for visualization."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values[~np.isfinite(values)] = 0

    # Shift if negative exists
    if values.min() < 0:
        values = values - values.min()

    # visualization scale
    values = np.log1p(values)

    vmin = values.min()
    vmax = values.max()
    vrange = vmax - vmin

    if vrange == 0:
        normalized = np.zeros_like(values)
    else:
        normalized = (values - vmin) / vrange

    return normalized, values


def make_colors(normalized_values, cmap_name):
    """Convert normalized values to RGBA colors with nonzero alpha."""
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(normalized_values)
    colors[:, 3] = 0.15 + 0.85 * normalized_values
    return colors


from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

def visualize_marker_overlay(
        xenium_coordinate,
        virtual_protein_coordinate,
        codex_coordinate,
        xenium_gene_expr,
        virtual_protein_expr,
        codex_protein_expr1,
        codex_protein_expr2,
        gene_key="CDKN1A",
        protein_key1="p16",
        protein_key2="HSP47",
):
    """
    2 × 3 visualization:

    [0, 0] CDKN1A expression in Xenium
    [0, 1] HSP47 expression in CODEX
    [0, 2] Aligned/generated HSP47 expression in Xenium cells

    [1, 0] CODEX p16 and HSP47 co-visualization
    [1, 1] Xenium CDKN1A and aligned HSP47 co-visualization
    [1, 2] Unused
    """

    # ============================================================
    # Prepare log1p expression for visualization
    # ============================================================
    _, log_gene = normalize_values(xenium_gene_expr)
    _, log_virtual = normalize_values(virtual_protein_expr)
    _, log_codex1 = normalize_values(codex_protein_expr1)
    _, log_codex2 = normalize_values(codex_protein_expr2)

    def normalize_from_range(values, vmin, vmax):
        """Normalize an array using a supplied numerical range."""
        values = np.asarray(values, dtype=float)

        if np.isclose(vmax, vmin):
            return np.zeros_like(values)

        normalized = (values - vmin) / (vmax - vmin)
        return np.clip(normalized, 0.0, 1.0)

    # Gene and p16 use their own ranges.
    gene_vmin = np.nanmin(log_gene)
    gene_vmax = np.nanmax(log_gene)

    p16_vmin = np.nanmin(log_codex1)
    p16_vmax = np.nanmax(log_codex1)

    norm_gene = normalize_from_range(
        log_gene,
        gene_vmin,
        gene_vmax,
    )

    norm_codex1 = normalize_from_range(
        log_codex1,
        p16_vmin,
        p16_vmax,
    )

    # Use one shared range for original and aligned HSP47.
    # This makes their intensities directly comparable.
    hsp47_vmin = min(
        np.nanmin(log_codex2),
        np.nanmin(log_virtual),
    )
    hsp47_vmax = max(
        np.nanmax(log_codex2),
        np.nanmax(log_virtual),
    )

    norm_codex2 = normalize_from_range(
        log_codex2,
        hsp47_vmin,
        hsp47_vmax,
    )

    norm_virtual = normalize_from_range(
        log_virtual,
        hsp47_vmin,
        hsp47_vmax,
    )

    # ============================================================
    # Colors
    # ============================================================
    # CDKN1A: red
    # p16: purple
    # Original CODEX HSP47: blue
    # Aligned HSP47: green
    colors_gene = make_colors(norm_gene, "Reds")
    colors_codex1 = make_colors(norm_codex1, "Oranges")
    colors_codex2 = make_colors(norm_codex2, "Blues")
    colors_virtual = make_colors(norm_virtual, "Greens")

    # ============================================================
    # Figure
    # ============================================================
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(24, 12),
    )

    fig.suptitle(
        (
            f"Pair: Xenium {xenium_sampleid}–{xenium_regionid} | "
            f"CODEX {codex_sampleid}–{codex_regionid}"
        ),
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )

    # ============================================================
    # [0, 0] Original Xenium gene
    # ============================================================
    axes[0, 0].scatter(
        xenium_coordinate[:, 0],
        xenium_coordinate[:, 1],
        color=colors_gene,
        s=5,
        linewidths=0,
        rasterized=True,
    )

    axes[0, 0].set_title(
        f"{gene_key} Expression in Xenium (reference)",
        fontsize=20,
    )

    # ============================================================
    # [0, 1] Original CODEX HSP47
    # ============================================================
    axes[0, 1].scatter(
        codex_coordinate[:, 0],
        codex_coordinate[:, 1],
        color=colors_codex2,
        s=5,
        linewidths=0,
        rasterized=True,
    )

    axes[0, 1].set_title(
        f"{protein_key2} Expression in CODEX",
        fontsize=20,
    )

    # ============================================================
    # [0, 2] Aligned/generated HSP47 in Xenium cells
    # ============================================================
    axes[0, 2].scatter(
        virtual_protein_coordinate[:, 0],
        virtual_protein_coordinate[:, 1],
        color=colors_virtual,
        s=5,
        linewidths=0,
        rasterized=True,
    )

    axes[0, 2].set_title(
        f"Aligned {protein_key2} Expression in Xenium Cells",
        fontsize=20,
    )

    # ============================================================
    # [1, 0] CODEX p16 + HSP47 overlay
    # ============================================================
    # Draw HSP47 first
    axes[1, 0].scatter(
        codex_coordinate[:, 0],
        codex_coordinate[:, 1],
        color=colors_codex2,
        marker="x",
        s=5,
        linewidths=1,
        zorder=1,
        rasterized=True,
    )

    # Draw p16 second
    axes[1, 0].scatter(
        codex_coordinate[:, 0],
        codex_coordinate[:, 1],
        color=colors_codex1,
        marker="o",
        s=5,
        linewidths=0,
        zorder=2,
        rasterized=True,
    )

    axes[1, 0].set_title(
        f"{protein_key1} + {protein_key2} in CODEX Cells",
        fontsize=20,
    )

    codex_legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=cm.get_cmap("Oranges")(0.9),
            markeredgecolor="none",
            markersize=9,
            label=protein_key1,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color=cm.get_cmap("Blues")(0.75),
            markeredgewidth=1.5,
            markersize=9,
            label=protein_key2,
        ),
    ]

    axes[1, 0].legend(
        handles=codex_legend_handles,
        fontsize=12,
        loc="best",
        frameon=True,
        framealpha=0.9,
    )

    # ============================================================
    # [1, 1] CDKN1A + aligned HSP47 overlay
    # ============================================================
    # Draw aligned HSP47 first
    axes[1, 1].scatter(
        virtual_protein_coordinate[:, 0],
        virtual_protein_coordinate[:, 1],
        color=colors_virtual,
        marker="x",
        s=5,
        linewidths=2,
        zorder=1,
        rasterized=True,
    )

    # Draw CDKN1A second
    axes[1, 1].scatter(
        xenium_coordinate[:, 0],
        xenium_coordinate[:, 1],
        color=colors_gene,
        marker="o",
        s=5,
        linewidths=0,
        zorder=2,
        rasterized=True,
    )

    axes[1, 1].set_title(
        f"{gene_key} + Aligned {protein_key2} in Xenium Cells",
        fontsize=20,
    )

    xenium_legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=cm.get_cmap("Reds")(0.75),
            markeredgecolor="none",
            markersize=9,
            label=gene_key,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            color=cm.get_cmap("Greens")(0.75),
            markeredgewidth=1.5,
            markersize=9,
            label=f"Aligned {protein_key2}",
        ),
    ]

    axes[1, 1].legend(
        handles=xenium_legend_handles,
        fontsize=12,
        loc="best",
        frameon=True,
        framealpha=0.9,
    )

    # The sixth panel is currently unused.
    axes[1, 2].axis("off")

    # ============================================================
    # Common spatial range
    # ============================================================
    all_coordinates = np.vstack(
        [
            xenium_coordinate,
            virtual_protein_coordinate,
            codex_coordinate,
        ]
    )

    xmin = np.nanmin(all_coordinates[:, 0])
    xmax = np.nanmax(all_coordinates[:, 0])
    ymin = np.nanmin(all_coordinates[:, 1])
    ymax = np.nanmax(all_coordinates[:, 1])

    x_padding = 0.02 * max(xmax - xmin, 1e-8)
    y_padding = 0.02 * max(ymax - ymin, 1e-8)

    for ax in axes.flat:
        if ax is axes[1, 2]:
            continue

        ax.set_xlim(
            xmin - x_padding,
            xmax + x_padding,
        )
        ax.set_ylim(
            ymin - y_padding,
            ymax + y_padding,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Keep the subplot border visible
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")

    # ============================================================
    # Colorbars
    # One colorbar for each individual-signal panel
    # ============================================================
    gene_sm = cm.ScalarMappable(
        norm=mcolors.Normalize(
            vmin=gene_vmin,
            vmax=gene_vmax,
        ),
        cmap="Reds",
    )
    gene_sm.set_array([])

    codex_hsp47_sm = cm.ScalarMappable(
        norm=mcolors.Normalize(
            vmin=hsp47_vmin,
            vmax=hsp47_vmax,
        ),
        cmap="Blues",
    )
    codex_hsp47_sm.set_array([])

    aligned_hsp47_sm = cm.ScalarMappable(
        norm=mcolors.Normalize(
            vmin=hsp47_vmin,
            vmax=hsp47_vmax,
        ),
        cmap="Greens",
    )
    aligned_hsp47_sm.set_array([])

    # Correctly attached to [0, 0].
    cbar_gene = fig.colorbar(
        gene_sm,
        ax=axes[0, 0],
        fraction=0.046,
        pad=0.02,
        shrink=0.5,
        aspect=25,
    )
    cbar_gene.set_label(
        f"log1p {gene_key}",
        fontsize=13,
    )
    cbar_gene.ax.tick_params(labelsize=10)

    # Correctly attached to [0, 1].
    cbar_codex = fig.colorbar(
        codex_hsp47_sm,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.02,
        shrink=0.5,
        aspect=25,
    )
    cbar_codex.set_label(
        f"log1p CODEX {protein_key2}",
        fontsize=13,
    )
    cbar_codex.ax.tick_params(labelsize=10)

    # Correctly attached to [0, 2], rather than [1, 0].
    cbar_virtual = fig.colorbar(
        aligned_hsp47_sm,
        ax=axes[0, 2],
        fraction=0.046,
        pad=0.02,
        shrink=0.5,
        aspect=25,
    )
    cbar_virtual.set_label(
        f"log1p aligned {protein_key2}",
        fontsize=13,
    )
    cbar_virtual.ax.tick_params(labelsize=10)

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    plt.show()
    # fig.savefig(
    #     f"/media/huifang/data/sennet/Axis_generated_proteins/figures/{i}.png",
    #     dpi=450,
    #     bbox_inches="tight",
    # )
    # print('saved')
    # test = input()



# =============================================================================
# Load and visualize all pairs
# =============================================================================

file_path = "/media/huifang/data/sennet/xenium_codex_pairs.txt"

with open(file_path, "r") as file:
    sennet_pairs = file.readlines()

# for i in range(0,len(sennet_pairs)):
for i in range(0, len(sennet_pairs)):
    # bad cases
    if i in [8, 25, 26]:
        continue

    print(f"Showing pair {i}")

    line = sennet_pairs[i].strip()
    if not line:
        continue

    xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = line.split()

    xenium_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/Axis_generated_proteins/"
        f"virtual_sennet_{i}_xenium.h5ad"
    )

    virtual_protein_data = sc.read_h5ad(
        "/media/huifang/data/sennet/Axis_generated_proteins/"
        f"virtual_sennet_{i}_protein.h5ad"
    )

    #

    codex_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/Axis_registered_data/"
        f"codex_{codex_sampleid}_{codex_regionid}.h5ad"
    )




    gene_key = "CDKN1A"
    protein_key1 = "p16"
    protein_key2 = "HSP47"
    coor_key = "aligned_combined"

    xenium_coor = np.asarray(xenium_gene_data.obsm["spatial"])
    codex_coor = np.stack([
        codex_gene_data.obs['x_trans'].values,
        codex_gene_data.obs['y_trans'].values
    ], axis=1)

    # Virtual protein uses its own spatial coordinate
    virtual_protein_coor = np.asarray(virtual_protein_data.obsm["spatial"])

    # expression
    xenium_gene_expr = get_feature_values(xenium_gene_data, gene_key)

    virtual_protein_expr = get_feature_values(virtual_protein_data, protein_key2)

    codex_protein_expr1 = codex_gene_data.obs[protein_key1].to_numpy(dtype=float)
    codex_protein_expr2 = codex_gene_data.obs[protein_key2].to_numpy(dtype=float)

    visualize_marker_overlay(
        xenium_coordinate=xenium_coor,
        virtual_protein_coordinate=virtual_protein_coor,
        codex_coordinate=codex_coor,
        xenium_gene_expr=xenium_gene_expr,
        virtual_protein_expr=virtual_protein_expr,
        codex_protein_expr1=codex_protein_expr1,
        codex_protein_expr2=codex_protein_expr2,
        gene_key=gene_key,
        protein_key1=protein_key1,
        protein_key2=protein_key2,
    )