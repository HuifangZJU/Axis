"""Molecular pseudo-images (Methods, "Pseudo-images for spatial molecular profiles").

The molecular matrices of all samples are restricted to shared features,
concatenated and projected to ``C_p`` principal components. Each component is
then rasterised onto a regular grid with a Gaussian kernel (Eq. (1)),

``P_k^(c)(u) = sum_i K_sigma(|u - x_ki|) z_ki^(c) / (sum_i K_sigma(|u - x_ki|) + eps)``,

giving a ``C_p x H x W`` pseudo-image per sample (Eq. (2)) plus a binary mask of
grid cells with molecular support.
"""

from __future__ import annotations

from functools import reduce
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter

__all__ = [
    "reduce_gene_reads",
    "channelwise_min_max_normalize",
    "get_gene_feature_matrix",
    "get_gene_feature_matrix_soft",
    "remove_salt_pepper",
    "bilateral_smoothing",
    "resize_channels",
    "rasterize_pseudoimage",
    "PSEUDOIMAGE_PRESETS",
    "build_pseudoimages",
]


# =============================================================================
# Building blocks (identical numerics to the original preprocessing utilities)
# =============================================================================
def reduce_gene_reads(
    gene_reads: np.ndarray, method: str = "pca", n_components: int = 10, random_state: Optional[int] = None
) -> np.ndarray:
    """Project an ``(n, m)`` molecular matrix to ``n_components`` dimensions."""
    if not isinstance(gene_reads, np.ndarray):
        raise ValueError("gene_reads must be a NumPy array of shape (n, m).")
    if method.lower() == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components, random_state=random_state).fit_transform(gene_reads)
    if method.lower() == "umap":
        import umap.umap_ as umap

        return umap.UMAP(n_components=n_components, random_state=random_state).fit_transform(gene_reads)
    raise ValueError("method must be one of ['pca', 'umap'].")


def channelwise_min_max_normalize(data: np.ndarray) -> np.ndarray:
    """Map every column of an ``(N, C)`` array to ``[0, 1]``."""
    mins = data.min(axis=0)
    ranges = data.max(axis=0) - mins
    ranges[ranges == 0] = 1e-8
    return (data - mins) / ranges


def get_gene_feature_matrix(coords: np.ndarray, reduced_data: np.ndarray, image_size=(2016, 2016), patch_size: int = 32):
    """Hard binning: average the features of all spots falling in each grid cell."""
    if coords.shape[0] != reduced_data.shape[0]:
        raise ValueError("coords and reduced_data must have the same number of rows.")
    _, n_dims = reduced_data.shape
    height, width = image_size
    out_h, out_w = height // patch_size, width // patch_size

    px = np.asarray(coords[:, 0]).astype(int) // patch_size
    py = np.asarray(coords[:, 1]).astype(int) // patch_size
    inside = (px >= 0) & (px < out_w) & (py >= 0) & (py < out_h)
    sum_array = np.zeros((out_h, out_w, n_dims), dtype=float)
    count_array = np.zeros((out_h, out_w), dtype=int)
    np.add.at(sum_array, (py[inside], px[inside]), reduced_data[inside])
    np.add.at(count_array, (py[inside], px[inside]), 1)

    patch_matrix = np.zeros_like(sum_array)
    valid = count_array > 0
    patch_matrix[valid, :] = sum_array[valid, :] / count_array[valid, None]
    return patch_matrix, valid.astype(np.uint8)


_NEIGHBOUR_OFFSETS = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)])


def get_gene_feature_matrix_soft(
    coords: np.ndarray,
    reduced_data: np.ndarray,
    image_size=(2016, 2016),
    patch_size: int = 32,
    kernel: str = "gaussian",
    sigma: Optional[float] = None,
    eps: float = 1e-6,
    normalize_per_cell: bool = True,
):
    """Soft kernel rasterisation, Eq. (1).

    Every spot distributes its feature vector over the 3x3 neighbourhood of grid
    cells around it, with Gaussian weights ``exp(-d^2 / 2 sigma^2)`` of its
    distance to each cell centre (``sigma = patch_size / 2`` by default).

    Returns the ``(H/ps, W/ps, C)`` pseudo-image and the ``(H/ps, W/ps)`` support mask.
    """
    if coords.shape[0] != reduced_data.shape[0]:
        raise ValueError("coords and reduced_data must have the same number of rows.")
    if kernel not in {"gaussian", "inverse_distance"}:
        raise ValueError("kernel must be 'gaussian' or 'inverse_distance'.")
    if sigma is None and kernel == "gaussian":
        sigma = patch_size / 2.0

    _, n_dims = reduced_data.shape
    height, width = image_size
    out_h, out_w = height // patch_size, width // patch_size

    coords = np.asarray(coords, dtype=np.float64)
    x, y = coords[:, 0], coords[:, 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y = x[inside], y[inside]
    values = np.asarray(reduced_data, dtype=np.float64)[inside]
    px = x.astype(int) // patch_size
    py = y.astype(int) // patch_size

    ny = py[:, None] + _NEIGHBOUR_OFFSETS[None, :, 0]
    nx = px[:, None] + _NEIGHBOUR_OFFSETS[None, :, 1]
    valid = (nx >= 0) & (nx < out_w) & (ny >= 0) & (ny < out_h)
    dists = np.hypot(x[:, None] - (nx + 0.5) * patch_size, y[:, None] - (ny + 0.5) * patch_size)

    if kernel == "gaussian":
        w = np.exp(-(dists**2) / (2.0 * sigma**2 + 1e-12))
    else:
        w = 1.0 / (dists + eps)
    w = np.where(valid, w, 0.0)
    if normalize_per_cell:
        s = w.sum(axis=1, keepdims=True)
        w = np.divide(w, s, out=np.zeros_like(w), where=s > 0)

    keep = valid & (w > 0)
    rows, slots = np.nonzero(keep)
    sum_array = np.zeros((out_h, out_w, n_dims), dtype=np.float64)
    weight_sum = np.zeros((out_h, out_w), dtype=np.float64)
    ww = w[rows, slots]
    np.add.at(sum_array, (ny[rows, slots], nx[rows, slots]), ww[:, None] * values[rows])
    np.add.at(weight_sum, (ny[rows, slots], nx[rows, slots]), ww)

    patch_matrix = np.zeros_like(sum_array)
    support = weight_sum > 0
    patch_matrix[support, :] = sum_array[support, :] / weight_sum[support, None]
    return patch_matrix.astype(np.float32), support.astype(np.uint8)


def remove_salt_pepper(feature_matrix: np.ndarray, size: int = 2) -> np.ndarray:
    """Channel-wise median filter."""
    filtered = np.zeros_like(feature_matrix)
    for c in range(feature_matrix.shape[2]):
        filtered[..., c] = median_filter(feature_matrix[..., c], size=size)
    return filtered


def bilateral_smoothing(feature_matrix: np.ndarray, weight: float = 0.01) -> np.ndarray:
    """Edge-preserving smoothing (bilateral filter followed by TV denoising)."""
    import cv2
    from skimage.restoration import denoise_tv_chambolle

    if feature_matrix.max() > 1:
        feature_matrix = (feature_matrix - feature_matrix.min()) / feature_matrix.max()
    out = np.zeros_like(feature_matrix)
    for i in range(feature_matrix.shape[2]):
        channel = cv2.bilateralFilter(feature_matrix[..., i].astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=25)
        out[..., i] = denoise_tv_chambolle(channel.astype(np.float64), weight=weight, eps=1e-4)
    return out


def resize_channels(feature_matrix: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize of every channel to ``size x size``."""
    import cv2

    return np.stack(
        [
            cv2.resize(feature_matrix[:, :, i], (size, size), interpolation=cv2.INTER_NEAREST)
            for i in range(feature_matrix.shape[2])
        ],
        axis=-1,
    )


# =============================================================================
# Presets reproducing the dataset-specific preprocessing of the manuscript
# =============================================================================
#: ``patch_size``: grid cell size in canvas pixels, or ``patch_divisor``: number
#: of cells along the canvas. ``median``: median-filter size (``None`` = off).
#: ``resize_divisor``: nearest-neighbour resize to ``canvas / resize_divisor``.
#: ``min_value``: values below are zeroed.
PSEUDOIMAGE_PRESETS: Dict[str, Dict[str, object]] = {
    # SCC / Visium: 32 x 32 cells (16 px on a 512 canvas), median filter, nearest resize to canvas / 8.
    "visium": {"patch_divisor": 32, "median": 2, "resize_divisor": 8, "min_value": None, "soft": True},
    # Xenium / PhenoCycler: 8-px cells (canvas / 8, i.e. 128 x 128 on a 1024 canvas), small values zeroed.
    "xenium": {"patch_size": 8, "median": None, "resize_divisor": None, "min_value": 0.01, "soft": True},
}


def rasterize_pseudoimage(
    coords: np.ndarray,
    embedding: np.ndarray,
    canvas_size: int,
    preset: str = "visium",
    **overrides,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterise a per-spot embedding into a pseudo-image at 1/8 canvas resolution.

    Returns ``(pseudoimage (h, w, C) float32, mask (h, w) uint8)``.
    """
    params = dict(PSEUDOIMAGE_PRESETS[preset]) if preset else {}
    params.update(overrides)
    patch_size = int(params.get("patch_size") or canvas_size // int(params["patch_divisor"]))
    if params.get("soft", True):
        feat, _ = get_gene_feature_matrix_soft(
            coords, embedding, (canvas_size, canvas_size), patch_size=patch_size, sigma=params.get("sigma")
        )
    else:
        feat, _ = get_gene_feature_matrix(coords, embedding, (canvas_size, canvas_size), patch_size=patch_size)
    if params.get("median"):
        feat = remove_salt_pepper(feat, size=int(params["median"]))
    if params.get("resize_divisor"):
        feat = resize_channels(feat, canvas_size // int(params["resize_divisor"]))
    if params.get("min_value") is not None:
        feat[feat < float(params["min_value"])] = 0
    mask = np.any(feat > 0, axis=-1).astype(np.uint8)
    expected = canvas_size // 8
    if feat.shape[0] != expected:
        raise ValueError(
            f"Pseudo-image is {feat.shape[0]}x{feat.shape[1]} but the network fuses molecular features "
            f"at 1/8 of the canvas ({expected}x{expected}); adjust patch_divisor/resize_divisor."
        )
    return feat.astype(np.float32), mask


def _dense(X) -> np.ndarray:
    return np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)


def build_pseudoimages(
    adatas: Sequence,
    *,
    n_comps: int = 10,
    spatial_key: str = "spatial_canvas",
    preset: str = "visium",
    layer: Optional[str] = None,
    normalize: bool = True,
    target_sum: float = 1e4,
    min_cells: Optional[int] = None,
    min_genes: Optional[int] = None,
    random_state: Optional[int] = 0,
    key_added: str = "axis",
    **raster_kwargs,
) -> List:
    """Build molecular pseudo-images for a set of samples (scanpy-style, in place).

    Parameters
    ----------
    adatas
        Samples to register. Each must have been passed through
        :func:`axis.pp.prepare_canvas` (or provide canvas coordinates in
        ``obsm[spatial_key]`` and the canvas image in ``uns[key_added]["image"]``).
    n_comps
        Number of principal components ``C_p``.
    preset
        ``"visium"`` or ``"xenium"``, see :data:`PSEUDOIMAGE_PRESETS`.
    layer
        Use ``adata.layers[layer]`` instead of ``adata.X``.
    normalize
        Normalise counts to ``target_sum`` per spot and ``log1p``-transform
        (set ``False`` if ``X`` is already normalised).
    min_cells, min_genes
        Optional per-sample filtering applied before feature harmonisation
        (SCC defaults: 15 and 100).
    random_state
        Seed of the PCA solver.

    Returns
    -------
    The (possibly filtered) AnnData objects. For each sample,
    ``uns[key_added]`` gains ``"pseudoimage"`` ``(h, w, C)``, ``"mask"`` ``(h, w)``
    and ``"pseudoimage_params"``, and ``obsm["X_axis_pca"]`` stores the
    normalised spot embedding.
    """
    import scanpy as sc

    work = []
    for adata in adatas:
        if key_added not in adata.uns or "image" not in adata.uns[key_added]:
            raise KeyError(f"uns[{key_added!r}]['image'] missing; run axis.pp.prepare_canvas first.")
        if spatial_key not in adata.obsm:
            raise KeyError(f"obsm[{spatial_key!r}] missing; run axis.pp.prepare_canvas first.")
        if min_cells is not None or min_genes is not None:
            if min_cells is not None:
                sc.pp.filter_genes(adata, min_cells=min_cells)
            if min_genes is not None:
                sc.pp.filter_cells(adata, min_genes=min_genes)
        a = adata.copy()
        if layer is not None:
            a.X = a.layers[layer].copy()
        if normalize:
            sc.pp.normalize_total(a, target_sum=target_sum)
            sc.pp.log1p(a)
        work.append(a)

    common = reduce(np.intersect1d, [np.asarray(a.var_names) for a in work])
    if len(common) == 0:
        raise ValueError("The samples share no molecular features.")
    blocks = [_dense(a[:, common].X) for a in work]
    reduced = channelwise_min_max_normalize(reduce_gene_reads(np.vstack(blocks), "pca", n_comps, random_state=random_state))

    start = 0
    for adata, block in zip(adatas, blocks):
        emb = reduced[start : start + block.shape[0]]
        start += block.shape[0]
        canvas = int(adata.uns[key_added]["image"].shape[0])
        pseudo, mask = rasterize_pseudoimage(np.asarray(adata.obsm[spatial_key]), emb, canvas, preset=preset, **raster_kwargs)
        adata.obsm["X_axis_pca"] = emb.astype(np.float32)
        adata.uns[key_added]["pseudoimage"] = pseudo
        adata.uns[key_added]["mask"] = mask
        adata.uns[key_added]["pseudoimage_params"] = {
            "n_comps": int(n_comps),
            "preset": str(preset),
            "n_shared_features": int(len(common)),
            "normalize": bool(normalize),
            **{k: v for k, v in raster_kwargs.items() if isinstance(v, (int, float, str, bool))},
        }
    return list(adatas)
