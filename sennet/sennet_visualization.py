import numpy as np
from matplotlib import pyplot as plt
import scanpy as sc
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def visualize_marker_overlay():
    f, a = plt.subplots(1, 2, figsize=(16, 8))
    # Normalize gene expression to [0, 1]
    norm_xenium = (xenium_gene_expr - xenium_gene_expr.min()) / (xenium_gene_expr.max() - xenium_gene_expr.min())
    norm_codex = (codex_gene_expr - codex_gene_expr.min()) / (codex_gene_expr.max() - codex_gene_expr.min())

    # Convert cmap to RGBA with alpha = normalized expression
    cmap_red = cm.get_cmap('Reds')
    # cmap_red = plt.colormaps['Reds']
    colors_xenium = cmap_red(norm_xenium)
    colors_xenium[:, -1] = norm_xenium  # set alpha channel

    cmap_blue = cm.get_cmap('Blues')
    # cmap_blue = plt.colormaps['Blues']
    colors_codex = cmap_blue(norm_codex)
    colors_codex[:, -1] = norm_codex  # set alpha channel

    a[0].scatter(xenium_coordinate[:, 0], xenium_coordinate[:, 1],
                 color=colors_xenium, s=10, label='Xenium CDKN1A')
    a[0].scatter(codex_coordinate[:, 0], codex_coordinate[:, 1],
                 color=colors_codex, s=10, label='CODEX p16')

    a[0].set_title("Unregistered Marker Levels", fontsize=22)

    # Scatter with RGBA color array
    a[1].scatter(warped_xenium_coor[:, 0], warped_xenium_coor[:, 1],
                 color=colors_xenium, s=10, label='Xenium CDKN1A')

    a[1].scatter(warped_codex_coor[:, 0], warped_codex_coor[:, 1],
                 color=colors_codex, s=10, label='CODEX p16')

    a[1].set_title("Registered CODEX to Xenium with Marker Levels", fontsize=22)

    # Create normalization
    norm_xenium = mcolors.Normalize(vmin=xenium_gene_expr.min(), vmax=xenium_gene_expr.max())
    norm_codex = mcolors.Normalize(vmin=codex_gene_expr.min(), vmax=codex_gene_expr.max())

    # Create ScalarMappables (used for colorbars)
    sm_xenium = cm.ScalarMappable(cmap='Reds', norm=norm_xenium)
    sm_xenium.set_array([])  # required for colorbar

    sm_codex = cm.ScalarMappable(cmap='Blues', norm=norm_codex)
    sm_codex.set_array([])

    # Add colorbars to the subplot
    # Colorbar 1 (e.g., CDKN1A)
    cbar_ax1 = f.add_axes([0.91, 0.30, 0.01, 0.3])  # [left, bottom, width, height]
    cbar = f.colorbar(sm_xenium, cax=cbar_ax1)
    cbar.set_label('CDKN1A', fontsize=14)  # Set label font size
    cbar.ax.tick_params(labelsize=10)  # Set
    # Colorbar 2 (e.g., p16) placed lower
    cbar_ax2 = f.add_axes([0.95, 0.30, 0.01, 0.3])
    cbar = f.colorbar(sm_codex, cax=cbar_ax2)
    cbar.set_label('p16', fontsize=14)  # Set label font size
    cbar.ax.tick_params(labelsize=10)  # Set
    for ax in a.flat:
        ax.set_aspect('equal')
    plt.show()


def formatting_xenium_data():
    xenium_gene_data.obsm["spatial"] = np.column_stack(
        [
            xenium_gene_data.obs["x_trans"].to_numpy(),
            xenium_gene_data.obs["y_trans"].to_numpy(),
        ]
    ).astype(np.float32)

    # ---------------------------------------------------------------------
    # Remove redundant coordinate columns from obs
    # ---------------------------------------------------------------------
    coordinate_columns = [
        "x_centroid",
        "y_centroid",
        "x_trans",
        "y_trans",
        "x_aligned",
        "y_aligned",
        "x_aligned_combined",
        "y_aligned_combined",
    ]

    columns_to_remove = [
        column
        for column in coordinate_columns
        if column in xenium_gene_data.obs.columns
    ]

    xenium_gene_data.obs.drop(
        columns=columns_to_remove,
        inplace=True,
    )

    output_path = f"/media/huifang/data/sennet/Axis_generated_proteins/virtual_sennet_{i}_xenium.h5ad"

    xenium_gene_data.write_h5ad(output_path)
    print(f"Saved simplified h5ad to:\n{output_path}")





file_path = '/media/huifang/data/sennet/xenium_codex_pairs.txt'
file = open(file_path)
sennet_pairs = file.readlines()
r_values = []
p_values = []


for i in range(0,len(sennet_pairs)):
    # bad ccases
    if i == 8 or i == 25 or i == 26:
        continue
    print(i)

    line = sennet_pairs[i]
    xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = line.rstrip().split(' ')
    xenium_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/Axis_registered_data/xenium" + f"_{xenium_sampleid}_{xenium_regionid}_with_ct.h5ad")


    # formatting_xenium_data()
    # continue


    codex_gene_data = sc.read_h5ad(
        "/media/huifang/data/sennet/Axis_registered_data/codex" + f"_{codex_sampleid}_{codex_regionid}.h5ad")


    virtual_protein_data =sc.read_h5ad(f"/media/huifang/data/sennet/Axis_generated_proteins/virtual_sennet_{i}_protein.h5ad")



    gene_key = 'CDKN1A'
    protein_key = 'p16'
    # coor_key = 'aligned'
    coor_key = "aligned_combined"

    xenium_coordinate = np.stack([
        xenium_gene_data.obs['x_trans'].values,
        xenium_gene_data.obs['y_trans'].values
    ], axis=1)
    print(xenium_gene_data)
    print(xenium_coordinate)
    test = input()



    codex_coordinate = np.stack([
        codex_gene_data.obs['x_trans'].values,
        codex_gene_data.obs['y_trans'].values
    ], axis=1)

    warped_xenium_coor = np.stack([
        xenium_gene_data.obs[f'x_{coor_key}'].values,
        xenium_gene_data.obs[f'y_{coor_key}'].values
    ], axis=1)

    warped_codex_coor= np.stack([
        codex_gene_data.obs[f'x_{coor_key}'].values,
        codex_gene_data.obs[f'y_{coor_key}'].values
    ], axis=1)

    virtual_protein_coor = np.asarray(virtual_protein_data.obsm['spatial'])


    xenium_gene_expr = xenium_gene_data[:, gene_key].to_df()[gene_key]
    virtual_protein_expr = virtual_protein_data[:,protein_key].to_df()[protein_key]
    virtual_protein_expr = virtual_protein_expr - virtual_protein_expr.min()



    # CODEX: use p16 intensity from .obs
    codex_gene_expr = codex_gene_data.obs[protein_key].values
    codex_gene_expr = codex_gene_expr - codex_gene_expr.min()



    xenium_gene_expr = np.log1p(xenium_gene_expr)
    codex_gene_expr = np.log1p(codex_gene_expr)
    visualize_marker_overlay()