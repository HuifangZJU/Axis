# pretsa_spatial_preprocess_scale_safe.py

from axis.settings import data_path
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import anndata as ad
from scipy import sparse
from patsy import dmatrix

def make_minimal_output_adata(
    source_adata,
    X_final,
    use_log1p_for_fit,
    knot,
    saved_scale_name="normalized_original_scale",
):
    """
    Create a minimal AnnData for saving.

    Keeps only:
        .X
        .obs index
        .var index
        .obsm["spatial"]
        minimal .uns metadata

    Does not keep intermediate layers.
    """
    X_final = np.asarray(X_final, dtype=np.float32)

    # Minimal obs: preserve cell IDs/index only
    obs_min = pd.DataFrame(index=source_adata.obs_names.copy())

    # Minimal var: preserve feature names only
    var_min = pd.DataFrame(index=source_adata.var_names.copy())

    adata_min = ad.AnnData(
        X=X_final,
        obs=obs_min,
        var=var_min,
    )

    # Keep spatial locations
    if "spatial" not in source_adata.obsm:
        raise KeyError("source_adata does not contain .obsm['spatial'].")

    adata_min.obsm["spatial"] = np.asarray(
        source_adata.obsm["spatial"],
        dtype=np.float32,
    )

    # Minimal provenance
    adata_min.uns["presta"] = {
        "method": "spatialFit",
        "knot": int(knot),
        "use_log1p_for_fit": bool(use_log1p_for_fit),
        "saved_X_scale": saved_scale_name,
        "note": (
            "X contains PreTSA-smoothed expression returned to normalized original scale. "
            "Spatial coordinates are stored in obsm['spatial']."
        ),
    }

    return adata_min
# ============================================================
# Configuration
# ============================================================

gene_h5ad_path = (
    data_path("phenocycler/huifang/"
    "virtual_tissue/virtual_pair/LUAD_2_A_xenium_0.000.h5ad")
)

protein_h5ad_path = (
    data_path("phenocycler/huifang/"
    "virtual_tissue/virtual_pair/virtual_protein_1.000.h5ad")
)

output_dir = Path(
    data_path("phenocycler/huifang/"
    "virtual_tissue/virtual_pair/presta")
)
output_dir.mkdir(parents=True, exist_ok=True)

knot = 50

gene_marker = "ACTA2"
protein_marker = "IDO1"

# ------------------------------------------------------------
# Important scale-control hyperparameters
# ------------------------------------------------------------

# If False:
#     fit spatialFit(normalized expression)
# If True:
#     fit spatialFit(log1p(normalized expression))
#     then convert back by expm1 before saving.
USE_LOG1P_FOR_FIT = True

# The final saved .X is always on normalized original scale.
SAVE_X_AS_ORIGINAL_SCALE = True

# Visualization always uses log1p values.
VISUALIZE_WITH_LOG1P = True

# If your .X is already normalized on the desired original scale,
# set this to True to avoid double library-size normalization.
GENE_X_ALREADY_NORMALIZED = False
PROTEIN_X_ALREADY_NORMALIZED = False

# Library-size normalization target if .X is not already normalized.
target_sum = 1e4

# Chunk size over features during smoothing.
feature_chunk_size = 500

# Output dtype
output_dtype = np.float32


# ============================================================
# PreTSA basis and spatialFit
# ============================================================

def _bs(x, df):
    B = dmatrix(
        f"bs(x, df={df}, include_intercept=False) - 1",
        {"x": np.asarray(x, dtype=np.float64)},
        return_type="dataframe",
    ).to_numpy()

    return np.ascontiguousarray(B, dtype=np.float64)


def _tp_basis(row_vals, col_vals, df):
    X = _bs(row_vals, df)
    Y = _bs(col_vals, df)

    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    Y = np.concatenate([np.ones((Y.shape[0], 1)), Y], axis=1)

    cols = [
        X[:, i] * Y[:, j]
        for i in range(X.shape[1])
        for j in range(Y.shape[1])
    ]

    B = np.stack(cols, axis=1)

    keep = B.std(axis=0, ddof=1) > 0
    B = B[:, keep]

    B = np.concatenate([np.ones((B.shape[0], 1)), B], axis=1)

    return np.ascontiguousarray(B, dtype=np.float64)


def build_spatial_basis(coord, knot=20):
    row_vals = np.asarray(coord["row"], dtype=np.float64)
    col_vals = np.asarray(coord["col"], dtype=np.float64)

    df = int(knot) + 3
    B = _tp_basis(row_vals, col_vals, df)

    print(f"Spatial basis shape: cells x basis = {B.shape}")

    if B.shape[0] <= B.shape[1]:
        print(
            "Warning: number of cells is smaller than or close to number of basis terms. "
            "The fit may be unstable. Consider reducing knot."
        )

    return B


def spatialFit(expr, coord, knot=20, chunk_size=500, ridge=1e-6, dtype=np.float32):
    """
    Spatial smoothing function.

    Parameters
    ----------
    expr:
        Normalized expression matrix in features x cells format.
        This can be either:
            normalized original scale
        or:
            log1p(normalized original scale)
    coord:
        Dictionary with coord["row"] and coord["col"], length = n_cells.
    knot:
        Spline knot parameter.
    chunk_size:
        Number of features to process per chunk.

    Returns
    -------
    pred:
        Smoothed expression matrix in features x cells format.
        Same scale as input expr.
    """
    expr = np.asarray(expr, dtype=np.float64)

    row_vals = np.asarray(coord["row"], dtype=np.float64)
    col_vals = np.asarray(coord["col"], dtype=np.float64)

    if expr.ndim != 2:
        raise ValueError(f"expr must be 2D features x cells, got shape {expr.shape}")

    n_features, n_cells = expr.shape

    if n_cells != len(row_vals):
        raise ValueError(
            f"expr is features x cells, but expr.shape[1]={n_cells} "
            f"does not match coord length={len(row_vals)}"
        )

    B = build_spatial_basis(coord, knot=knot)

    tBB = B.T @ B
    if ridge is not None and ridge > 0:
        tBB = tBB + ridge * np.eye(tBB.shape[0], dtype=np.float64)

    Bt = B.T

    pred = np.zeros((n_features, n_cells), dtype=dtype)

    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        print(f"  smoothing features {start}:{end} / {n_features}")

        # expr_chunk_GN: features_chunk x cells
        expr_chunk_GN = expr[start:end, :]

        # Convert to cells x features_chunk for linear solve
        X_chunk_NG = expr_chunk_GN.T

        # coef: basis x features_chunk
        coef = np.linalg.solve(tBB, Bt @ X_chunk_NG)

        # pred_chunk_NG: cells x features_chunk
        pred_chunk_NG = B @ coef

        # Back to features_chunk x cells
        pred[start:end, :] = pred_chunk_NG.T.astype(dtype)

    return pred


# ============================================================
# AnnData and preprocessing utilities
# ============================================================

def get_X(adata, layer=None):
    X = adata.layers[layer] if layer is not None else adata.X

    if sparse.issparse(X):
        return X.tocsr()

    return np.asarray(X)


def clean_invalid_values(X):
    """
    Clean NaN/inf and clip negative values.
    """
    if sparse.issparse(X):
        X = X.copy().tocsr()
        X.data = np.nan_to_num(X.data, nan=0.0, posinf=0.0, neginf=0.0)
        X.data[X.data < 0] = 0
        X.eliminate_zeros()
        return X

    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[X < 0] = 0
    return X


def normalize_total(X, target_sum=1e4):
    """
    Cell-wise normalization.

    Input:
        cells x features

    Output:
        cells x features on normalized original scale
    """
    if sparse.issparse(X):
        X = X.copy().tocsr()
        libsize = np.asarray(X.sum(axis=1)).ravel()

        scale = np.zeros_like(libsize, dtype=np.float64)
        nonzero = libsize > 0
        scale[nonzero] = target_sum / libsize[nonzero]

        X_norm = sparse.diags(scale) @ X
        return X_norm.tocsr()

    X = np.asarray(X, dtype=np.float32)
    libsize = X.sum(axis=1)

    scale = np.zeros_like(libsize, dtype=np.float32)
    nonzero = libsize > 0
    scale[nonzero] = target_sum / libsize[nonzero]

    return X * scale[:, None]


def to_dense_float32(X):
    if sparse.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def get_spatial_coord_for_presta(adata):
    """
    AnnData spatial convention:
        obsm['spatial'][:, 0] = x / col
        obsm['spatial'][:, 1] = y / row
    """
    if "spatial" not in adata.obsm:
        raise KeyError("Cannot find adata.obsm['spatial'].")

    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)

    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['spatial'] should have shape (n_cells, 2), got {spatial.shape}"
        )

    return {
        "row": spatial[:, 1],
        "col": spatial[:, 0],
    }


def make_fit_scale_matrix(X_original_scale, use_log1p_for_fit=True):
    """
    Convert normalized original-scale expression into fitting scale.
    """
    X_original_scale = np.asarray(X_original_scale, dtype=np.float32)
    X_original_scale = np.nan_to_num(X_original_scale, nan=0.0, posinf=0.0, neginf=0.0)
    X_original_scale[X_original_scale < 0] = 0

    if use_log1p_for_fit:
        return np.log1p(X_original_scale).astype(np.float32)

    return X_original_scale.astype(np.float32)


def inverse_fit_scale_matrix(X_smoothed_fit_scale, use_log1p_for_fit=True):
    """
    Convert smoothed expression from fitting scale back to normalized original scale.
    """
    X_smoothed_fit_scale = np.asarray(X_smoothed_fit_scale, dtype=np.float32)
    X_smoothed_fit_scale = np.nan_to_num(
        X_smoothed_fit_scale,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Fitted values may become slightly negative because of least-squares smoothing.
    # Expression should not be negative on either log or original scale.
    X_smoothed_fit_scale[X_smoothed_fit_scale < 0] = 0

    if use_log1p_for_fit:
        X_original_scale = np.expm1(X_smoothed_fit_scale)
    else:
        X_original_scale = X_smoothed_fit_scale

    X_original_scale = np.asarray(X_original_scale, dtype=np.float32)
    X_original_scale = np.nan_to_num(
        X_original_scale,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    X_original_scale[X_original_scale < 0] = 0

    return X_original_scale


def preprocess_to_normalized_original_scale(
    adata,
    modality,
    already_normalized=False,
    input_layer=None,
    target_sum=1e4,
):
    """
    Produce normalized original-scale expression matrix.

    This is the scale that will be saved after smoothing.

    If already_normalized=False:
        raw/read-like .X -> normalize_total(target_sum)

    If already_normalized=True:
        use .X as the original scale, after invalid-value cleanup.
    """
    adata_out = adata.copy()
    adata_out.var_names_make_unique()

    X = get_X(adata_out, layer=input_layer)
    X = clean_invalid_values(X)

    if already_normalized:
        print(f"{modality}: .X is treated as already normalized original-scale expression.")
        X_original_scale = X
        preprocessing_used = "existing_X_cleaned_only"
    else:
        print(f"{modality}: normalize_total(target_sum={target_sum})")
        X_original_scale = normalize_total(X, target_sum=target_sum)
        preprocessing_used = f"normalize_total_{target_sum}"

    X_original_scale = to_dense_float32(X_original_scale)

    adata_out.layers["normalized_original_scale"] = X_original_scale.copy()

    adata_out.uns["presta_preprocessing"] = {
        "modality": modality,
        "input_layer": input_layer if input_layer is not None else "X",
        "already_normalized": bool(already_normalized),
        "target_sum": float(target_sum),
        "preprocessing_used": preprocessing_used,
        "note": (
            "normalized_original_scale is the scale used for final saving. "
            "Optional log1p may be used only for spline fitting."
        ),
    }

    print(
        f"{modality} normalized original-scale matrix: "
        f"shape={X_original_scale.shape}, dtype={X_original_scale.dtype}, "
        f"min={np.nanmin(X_original_scale):.4f}, "
        f"max={np.nanmax(X_original_scale):.4f}"
    )

    return adata_out, X_original_scale


def run_presta_scale_safe(
    adata,
    X_original_scale_CN,
    knot=20,
    use_log1p_for_fit=True,
    chunk_size=500,
):
    """
    Run PreTSA/spatialFit while preserving final output on normalized original scale.

    Returns
    -------
    X_smoothed_original_CN:
        Final smoothed matrix on normalized original scale.
        This is what should be saved in output .X.

    X_fit_CN:
        Input to spatialFit, used for QC only.

    X_smoothed_fit_CN:
        Smoothed matrix on fitting scale, used for QC only.
    """
    coord = get_spatial_coord_for_presta(adata)

    # Convert to fitting scale
    X_fit_CN = make_fit_scale_matrix(
        X_original_scale_CN,
        use_log1p_for_fit=use_log1p_for_fit,
    )

    print(
        f"Fit-scale matrix: shape={X_fit_CN.shape}, "
        f"min={np.nanmin(X_fit_CN):.4f}, max={np.nanmax(X_fit_CN):.4f}, "
        f"use_log1p_for_fit={use_log1p_for_fit}"
    )

    # spatialFit expects features x cells
    expr_fit_GN = X_fit_CN.T

    pred_fit_GN = spatialFit(
        expr=expr_fit_GN,
        coord=coord,
        knot=knot,
        chunk_size=chunk_size,
        ridge=1e-6,
        dtype=np.float32,
    )

    X_smoothed_fit_CN = pred_fit_GN.T.astype(np.float32)
    X_smoothed_fit_CN[X_smoothed_fit_CN < 0] = 0

    # Convert back to normalized original scale for saving
    X_smoothed_original_CN = inverse_fit_scale_matrix(
        X_smoothed_fit_CN,
        use_log1p_for_fit=use_log1p_for_fit,
    ).astype(np.float32)

    print(
        f"Smoothed original-scale matrix: shape={X_smoothed_original_CN.shape}, "
        f"dtype={X_smoothed_original_CN.dtype}, "
        f"min={np.nanmin(X_smoothed_original_CN):.4f}, "
        f"max={np.nanmax(X_smoothed_original_CN):.4f}"
    )

    return X_smoothed_original_CN, X_fit_CN, X_smoothed_fit_CN


# ============================================================
# Marker plotting
# ============================================================

def find_marker_index(adata, marker_name):
    var_names = pd.Index(adata.var_names.astype(str))

    if marker_name in var_names:
        return marker_name, int(var_names.get_loc(marker_name))

    marker_upper = marker_name.upper()
    upper_to_name = {v.upper(): v for v in var_names}

    if marker_upper in upper_to_name:
        matched = upper_to_name[marker_upper]
        return matched, int(var_names.get_loc(matched))

    raise KeyError(
        f"Cannot find marker '{marker_name}' in var_names. "
        f"First 10 var_names: {list(var_names[:10])}"
    )


def get_marker_values(adata, marker_name, layer):
    marker_key, marker_idx = find_marker_index(adata, marker_name)

    X = adata.layers[layer] if layer is not None else adata.X

    if sparse.issparse(X):
        vals = X[:, marker_idx].toarray().ravel()
    else:
        vals = np.asarray(X[:, marker_idx]).ravel()

    return marker_key, vals


def values_for_visualization(vals, use_log1p=True):
    vals = np.asarray(vals, dtype=np.float32)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    vals[vals < 0] = 0

    if use_log1p:
        vals = np.log1p(vals)

    return vals


def robust_vmin_vmax(vals, low=1, high=99):
    vals = np.asarray(vals, dtype=float)

    if np.all(vals == vals[0]):
        return float(vals[0]), float(vals[0] + 1e-6)

    vmin, vmax = np.nanpercentile(vals, [low, high])

    if vmin == vmax:
        vmax = vmin + 1e-6

    return float(vmin), float(vmax)





def get_marker_values_from_matrix(adata, X, marker_name):
    marker_key, marker_idx = find_marker_index(adata, marker_name)
    vals = np.asarray(X[:, marker_idx]).ravel()
    return marker_key, vals


def plot_presta_marker_qc_from_arrays(
    gene_adata,
    gene_X_original,
    gene_X_smoothed_original,
    protein_adata,
    protein_X_original,
    protein_X_smoothed_original,
    gene_marker="ACTA2",
    protein_marker="IDO1",
    out_path=None,
):
    """
    Four panels:
        original gene, log1p visualization
        PreTSA-smoothed gene, log1p visualization
        original protein, log1p visualization
        PreTSA-smoothed protein, log1p visualization

    All input matrices are on normalized original scale.
    Visualization always uses log1p.
    """
    gene_marker_key, gene_orig = get_marker_values_from_matrix(
        gene_adata,
        gene_X_original,
        gene_marker,
    )

    _, gene_smooth = get_marker_values_from_matrix(
        gene_adata,
        gene_X_smoothed_original,
        gene_marker_key,
    )

    protein_marker_key, protein_orig = get_marker_values_from_matrix(
        protein_adata,
        protein_X_original,
        protein_marker,
    )

    _, protein_smooth = get_marker_values_from_matrix(
        protein_adata,
        protein_X_smoothed_original,
        protein_marker_key,
    )

    # Always log1p for visualization
    gene_orig_vis = values_for_visualization(gene_orig, use_log1p=True)
    gene_smooth_vis = values_for_visualization(gene_smooth, use_log1p=True)

    protein_orig_vis = values_for_visualization(protein_orig, use_log1p=True)
    protein_smooth_vis = values_for_visualization(protein_smooth, use_log1p=True)

    gene_scale = robust_vmin_vmax(
        np.concatenate([gene_orig_vis, gene_smooth_vis])
    )
    protein_scale = robust_vmin_vmax(
        np.concatenate([protein_orig_vis, protein_smooth_vis])
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sc0 = plot_marker_spatial(
        axes[0, 0],
        gene_adata,
        gene_orig_vis,
        title=f"Original gene log1p: {gene_marker_key}",
        scale=gene_scale,
    )
    plt.colorbar(sc0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    sc1 = plot_marker_spatial(
        axes[0, 1],
        gene_adata,
        gene_smooth_vis,
        title=f"PreTSA-smoothed gene log1p: {gene_marker_key}",
        scale=gene_scale,
    )
    plt.colorbar(sc1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    sc2 = plot_marker_spatial(
        axes[1, 0],
        protein_adata,
        protein_orig_vis,
        title=f"Original protein log1p: {protein_marker_key}",
        scale=protein_scale,
    )
    plt.colorbar(sc2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    sc3 = plot_marker_spatial(
        axes[1, 1],
        protein_adata,
        protein_smooth_vis,
        title=f"PreTSA-smoothed protein log1p: {protein_marker_key}",
        scale=protein_scale,
    )
    plt.colorbar(sc3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"PreTSA QC, knot={knot}, fit_log1p={USE_LOG1P_FOR_FIT}; visualization log1p",
        fontsize=14,
    )
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved QC figure: {out_path}")
        plt.close(fig)
    else:
        plt.show()



def plot_marker_spatial(
    ax,
    adata,
    values,
    title,
    point_size=0.2,
    cmap="magma",
    scale=None,
):
    spatial = np.asarray(adata.obsm["spatial"])

    x = spatial[:, 0]
    y = spatial[:, 1]

    if scale is None:
        vmin, vmax = robust_vmin_vmax(values)
    else:
        vmin, vmax = scale

    sc = ax.scatter(
        x,
        y,
        c=values,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )

    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.axis("off")

    return sc


def plot_presta_marker_qc(
    gene_adata,
    protein_adata,
    gene_marker="ACTA2",
    protein_marker="IDO1",
    original_layer="normalized_original_scale",
    smoothed_layer="presta_smoothed_original_scale",
    out_path=None,
):
    """
    Four panels:
        original normalized ACTA2 gene, visualized as log1p
        PreTSA-smoothed ACTA2 gene, saved scale but visualized as log1p
        original normalized IDO1 protein, visualized as log1p
        PreTSA-smoothed IDO1 protein, saved scale but visualized as log1p
    """
    gene_marker_key, gene_orig = get_marker_values(
        gene_adata,
        gene_marker,
        layer=original_layer,
    )
    _, gene_smooth = get_marker_values(
        gene_adata,
        gene_marker_key,
        layer=smoothed_layer,
    )

    protein_marker_key, protein_orig = get_marker_values(
        protein_adata,
        protein_marker,
        layer=original_layer,
    )
    _, protein_smooth = get_marker_values(
        protein_adata,
        protein_marker_key,
        layer=smoothed_layer,
    )

    # Always log1p for visualization
    gene_orig_vis = values_for_visualization(gene_orig, use_log1p=True)
    gene_smooth_vis = values_for_visualization(gene_smooth, use_log1p=True)

    protein_orig_vis = values_for_visualization(protein_orig, use_log1p=True)
    protein_smooth_vis = values_for_visualization(protein_smooth, use_log1p=True)

    gene_scale = robust_vmin_vmax(np.concatenate([gene_orig_vis, gene_smooth_vis]))
    protein_scale = robust_vmin_vmax(np.concatenate([protein_orig_vis, protein_smooth_vis]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sc0 = plot_marker_spatial(
        axes[0, 0],
        gene_adata,
        gene_orig_vis,
        title=f"Original gene log1p: {gene_marker_key}",
        scale=gene_scale,
    )
    plt.colorbar(sc0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    sc1 = plot_marker_spatial(
        axes[0, 1],
        gene_adata,
        gene_smooth_vis,
        title=f"PreTSA-smoothed gene log1p: {gene_marker_key}",
        scale=gene_scale,
    )
    plt.colorbar(sc1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    sc2 = plot_marker_spatial(
        axes[1, 0],
        protein_adata,
        protein_orig_vis,
        title=f"Original protein log1p: {protein_marker_key}",
        scale=protein_scale,
    )
    plt.colorbar(sc2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    sc3 = plot_marker_spatial(
        axes[1, 1],
        protein_adata,
        protein_smooth_vis,
        title=f"PreTSA-smoothed protein log1p: {protein_marker_key}",
        scale=protein_scale,
    )
    plt.colorbar(sc3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"PreTSA QC, knot={knot}, fit_log1p={USE_LOG1P_FOR_FIT}; visualization always log1p",
        fontsize=14,
    )
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved QC figure: {out_path}")
        plt.close(fig)
    else:
        plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("Reading gene AnnData...")
    gene_adata = ad.read_h5ad(gene_h5ad_path)
    gene_adata.var_names_make_unique()
    print(gene_adata)

    print("Reading protein AnnData...")
    protein_adata = ad.read_h5ad(protein_h5ad_path)
    protein_adata.var_names_make_unique()
    print(protein_adata)

    # ------------------------------------------------------------
    # Preprocess to normalized original scale
    # ------------------------------------------------------------
    print("\nPreprocessing gene data to normalized original scale...")
    gene_adata_norm, gene_X_original = preprocess_to_normalized_original_scale(
        gene_adata,
        modality="gene",
        already_normalized=GENE_X_ALREADY_NORMALIZED,
        input_layer=None,
        target_sum=target_sum,
    )

    print("\nPreprocessing protein data to normalized original scale...")
    protein_adata_norm, protein_X_original = preprocess_to_normalized_original_scale(
        protein_adata,
        modality="protein",
        already_normalized=PROTEIN_X_ALREADY_NORMALIZED,
        input_layer=None,
        target_sum=target_sum,
    )

    # ------------------------------------------------------------
    # Run PreTSA
    # ------------------------------------------------------------
    print("\nRunning PreTSA on gene data...")
    gene_X_smoothed_original, gene_X_fit, gene_X_smoothed_fit = run_presta_scale_safe(
        gene_adata_norm,
        X_original_scale_CN=gene_X_original,
        knot=knot,
        use_log1p_for_fit=USE_LOG1P_FOR_FIT,
        chunk_size=feature_chunk_size,
    )

    print("\nRunning PreTSA on protein data...")
    protein_X_smoothed_original, protein_X_fit, protein_X_smoothed_fit = run_presta_scale_safe(
        protein_adata_norm,
        X_original_scale_CN=protein_X_original,
        knot=knot,
        use_log1p_for_fit=USE_LOG1P_FOR_FIT,
        chunk_size=feature_chunk_size,
    )

    # ------------------------------------------------------------
    # QC visualization
    # ------------------------------------------------------------
    qc_fig_path = output_dir / (
        f"presta_qc_ACTA2_gene_IDO1_protein_"
        f"knot{knot}_fitlog1p{int(USE_LOG1P_FOR_FIT)}.png"
    )

    plot_presta_marker_qc_from_arrays(
        gene_adata=gene_adata_norm,
        gene_X_original=gene_X_original,
        gene_X_smoothed_original=gene_X_smoothed_original,
        protein_adata=protein_adata_norm,
        protein_X_original=protein_X_original,
        protein_X_smoothed_original=protein_X_smoothed_original,
        gene_marker=gene_marker,
        protein_marker=protein_marker,
        out_path=qc_fig_path,
    )

    # ------------------------------------------------------------
    # Create minimal output AnnData objects
    # ------------------------------------------------------------
    gene_adata_min = make_minimal_output_adata(
        source_adata=gene_adata_norm,
        X_final=gene_X_smoothed_original,
        use_log1p_for_fit=USE_LOG1P_FOR_FIT,
        knot=knot,
        saved_scale_name="normalized_original_scale",
    )

    protein_adata_min = make_minimal_output_adata(
        source_adata=protein_adata_norm,
        X_final=protein_X_smoothed_original,
        use_log1p_for_fit=USE_LOG1P_FOR_FIT,
        knot=knot,
        saved_scale_name="normalized_original_scale",
    )

    # ------------------------------------------------------------
    # Save compact h5ad outputs
    # ------------------------------------------------------------
    gene_out_path = output_dir / (
        f"LUAD_2_A_xenium_0.000_presta_knot{knot}_"
        f"fitlog1p{int(USE_LOG1P_FOR_FIT)}_minimal.h5ad"
    )

    protein_out_path = output_dir / (
        f"virtual_protein_1.000_presta_knot{knot}_"
        f"fitlog1p{int(USE_LOG1P_FOR_FIT)}_minimal.h5ad"
    )

    print(f"\nSaving minimal gene output to: {gene_out_path}")
    gene_adata_min.write_h5ad(gene_out_path)

    print(f"Saving minimal protein output to: {protein_out_path}")
    protein_adata_min.write_h5ad(protein_out_path)

    print("\nDone.")