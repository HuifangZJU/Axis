"""Diagnostic plots used by the preprocessing scripts."""

import numpy as np
from matplotlib import pyplot as plt


def visualize_spatial_features(
    coords,
    features,
    normalize=False,
    cmap="viridis",
    point_size=1,
    figsize=(18, 7),
    share_color_scale=False,
    save_path=None,
    X_LIM=None,
    Y_LIM=None
):
    """
    Visualize 10 spatial feature maps as scatter plots.

    Parameters
    ----------
    coords : np.ndarray
        Shape (n, 2), spatial coordinates.
    features : np.ndarray
        Shape (n, 10), feature matrix.
    normalize : bool
        If True, min-max normalize each feature independently.
    cmap : str
        Matplotlib colormap.
    point_size : int
        Scatter point size.
    figsize : tuple
        Figure size.
    share_color_scale : bool
        If True, use global min/max across all features.
    save_path : str or None
        If provided, save figure to this path.
    """

    assert coords.shape[1] == 2, "coords must be (n, 2)"
    assert features.shape[0] == coords.shape[0], "coords and features must match in n"
    assert features.shape[1] == 10, "features must be (n, 10)"

    coords = np.asarray(coords)
    features = np.asarray(features)

    # Optional normalization
    if normalize:
        f_min = features.min(axis=0, keepdims=True)
        f_max = features.max(axis=0, keepdims=True)
        features = (features - f_min) / (f_max - f_min + 1e-8)

    # Global color scale if needed
    if share_color_scale:
        global_min = features.min()
        global_max = features.max()
    else:
        global_min = None
        global_max = None

    fig, axes = plt.subplots(2, 5, figsize=figsize)
    axes = axes.flatten()

    for i in range(10):
        ax = axes[i]

        vmin = global_min if share_color_scale else features[:, i].min()
        vmax = global_max if share_color_scale else features[:, i].max()

        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=features[:, i],
            cmap=cmap,
            s=point_size,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(f"Feature {i}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(0,X_LIM)
        ax.set_ylim(0,Y_LIM)
        ax.invert_yaxis()
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_dimensional_images_side_by_side(patch_matrix: np.ndarray,ncols=5):
    # 3) Plot each dimension in a subplot
    n_dims = patch_matrix.shape[-1]
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)

    # Flatten axes so we can index easily
    axes_flat = axes.flatten()

    for dim_idx in range(n_dims):
        ax = axes_flat[dim_idx]

        # Extract the 2D patch grid for this dimension
        patch_image = patch_matrix[:, :, dim_idx]

        im = ax.imshow(
            patch_image,           # shape (out_height, out_width)
            origin='upper',        # row=0 at the top
            cmap='viridis',        # or 'gray', etc.
            aspect='auto'
        )
        ax.set_title(f"Dimension {dim_idx}")
        ax.set_xlabel("Patch (x)")
        ax.set_ylabel("Patch (y)")
        ax.axis('equal')
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots if n_dims < nrows*ncols
    for dim_idx in range(n_dims, nrows*ncols):
        axes_flat[dim_idx].axis("off")

    plt.tight_layout()
    plt.show()


def load_gene_expression(adata, genes):
    gene_expr = adata[:, genes].X
    if hasattr(gene_expr, 'toarray'):
        gene_expr = gene_expr.toarray()

    # Double log1p transform
    gene_expr = np.log1p(np.log1p(gene_expr))

    # Min-max normalization per gene
    gene_expr_norm = (gene_expr - gene_expr.min(axis=0)) / (gene_expr.max(axis=0) - gene_expr.min(axis=0) + 1e-6)
    return gene_expr_norm


def plot_marker_genes(adata, genes):
    n = len(genes)
    gene_expr_norm = load_gene_expression(adata, genes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)

    for i, gene in enumerate(genes):
        ax = axes[i] if n > 1 else axes
        expr = gene_expr_norm[:, i]
        cmap = plt.cm.Reds
        colors = cmap(expr)
        colors[:, -1] = expr  # set alpha as expression level

        ax.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1],
                   color=colors, s=10)
        ax.set_title(gene,fontsize=26)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    plt.show()

