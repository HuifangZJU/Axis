"""Morphology-image preprocessing.

Each sample is represented on a square canvas (512 px for Visium, 1024 px for
Xenium / PhenoCycler), with the same geometric transform applied to the image
and to the spot/cell coordinates (Methods, "AXIS analysis of real datasets").
"""

from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

__all__ = [
    "prepare_canvas",
    "crop_square_then_resize_square",
    "to_grayscale",
    "clahe",
    "resize_and_flip_image_coords",
    "degrade_he_for_ablation",
    "random_deform_field",
]

Image.MAX_IMAGE_PIXELS = None


def crop_square_then_resize_square(img: Image.Image, original_uv: np.ndarray, crop_para, final_size: int):
    """Crop a square ``(left, top, side)`` and resize it to ``final_size``.

    Returns ``(standard_img, final_img, standard_uv, final_uv)`` where the
    "standard" outputs are resized to a 10,000-px reference canvas.
    """
    left, top, side_length = crop_para
    cropped_img = img.crop((left, top, left + side_length, top + side_length))
    cropped_uv = original_uv - np.array([left, top])

    standard_img = cropped_img.resize((10000, 10000), Image.LANCZOS)
    standard_uv = cropped_uv * (10000 / side_length)
    final_img = cropped_img.resize((final_size, final_size), Image.LANCZOS)
    final_uv = cropped_uv * (final_size / side_length)
    return standard_img, final_img, standard_uv, final_uv


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Standard RGB -> grayscale conversion (ITU-R 601-2 luma), uint8."""
    img = np.asarray(image)
    if img.ndim == 2:
        return img
    return np.asarray(Image.fromarray(img.astype(np.uint8)[..., :3]).convert("L"))


def clahe(
    dapi_img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8), dilate_iterations: int = 2
) -> np.ndarray:
    """DAPI enhancement used for Xenium and PhenoCycler.

    Contrast-limited adaptive histogram equalisation (clip limit 2.0, 8x8 tiles)
    followed by two iterations of 3x3 elliptical dilation.
    """
    import cv2

    img_uint8 = (dapi_img * 255).astype("uint8") if dapi_img.max() <= 1.0 else dapi_img.astype("uint8")
    enhanced = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(img_uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(enhanced, kernel, iterations=dilate_iterations)


def resize_and_flip_image_coords(image, coords, new_size, flip_paras, group_idx=0, image_idx=0):
    """Resize a DAPI image to a square canvas, flip it, and apply CLAHE (Xenium)."""
    import cv2

    flip_lr, flip_ud = flip_paras[group_idx][image_idx]
    h, w = image.shape[:2]
    new_img = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
    new_coords = coords.copy().astype(np.float32)
    new_coords[:, 0] *= new_size / w
    new_coords[:, 1] *= new_size / h
    if flip_lr:
        new_img = cv2.flip(new_img, 1)
        new_coords[:, 0] = new_size - 1 - new_coords[:, 0]
    if flip_ud:
        new_img = cv2.flip(new_img, 0)
        new_coords[:, 1] = new_size - 1 - new_coords[:, 1]
    return clahe(new_img), new_coords


def prepare_canvas(
    adata,
    image: Optional[np.ndarray] = None,
    *,
    spatial_key: str = "spatial",
    canvas_size: int = 512,
    scale: float = 1.0,
    crop: Optional[Tuple[float, float, float]] = None,
    grayscale: bool = False,
    enhance_dapi: bool = False,
    key_added: str = "spatial_canvas",
    uns_key: str = "axis",
):
    """Place a sample's morphology image and coordinates on a square canvas (in place).

    Parameters
    ----------
    adata
        AnnData with spot/cell coordinates in ``obsm[spatial_key]`` (``x, y``).
    image
        Morphology image (``H x W``, ``H x W x 3`` or ``H x W x 4``; an alpha
        channel is dropped). ``uint8`` images and floats in ``[0, 1]`` or
        ``[0, 255]`` are used as is; images with larger values (e.g. 12/16-bit
        DAPI) are rescaled linearly by their maximum to 8 bits. Defaults to
        ``adata.uns["image_array"]``.
    spatial_key
        Coordinates in the pixel frame of ``image`` (after multiplying by ``scale``).
    canvas_size
        Output canvas size; must be divisible by 16 (512 or 1024 in the manuscript).
    scale
        Factor mapping ``obsm[spatial_key]`` to image pixels (e.g. a Visium
        ``tissue_hires_scalef``).
    crop
        Optional square crop ``(left, top, side)`` in image pixels. Without a
        crop, non-square images are padded to a square.
    grayscale
        Convert RGB to grayscale (used when samples of different modalities are
        registered; Methods "Input data").
    enhance_dapi
        Apply [`clahe`][axis.pp.clahe] (Xenium / PhenoCycler DAPI).

    Returns
    -------
    anndata.AnnData
        ``adata``, with the canvas image in ``uns[uns_key]["image"]`` and the
        canvas coordinates in ``obsm[key_added]``.
    """
    if canvas_size % 16:
        raise ValueError("canvas_size must be divisible by 16")
    if image is None:
        if "image_array" not in adata.uns:
            raise KeyError("No image given and adata.uns['image_array'] is missing.")
        image = adata.uns["image_array"]
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[2] == 4:  # RGBA export: drop the alpha channel
        img = img[..., :3]
    if img.dtype != np.uint8:
        peak = float(img.max()) if img.size else 0.0
        if peak <= 1.0:
            img = np.clip(img, 0, 1) * 255
        elif peak > 255:  # 12/16-bit images would otherwise saturate
            img = img.astype(np.float64) / peak * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2] * float(scale)

    if crop is None:
        h, w = img.shape[:2]
        side = max(h, w)
        if h != w:
            pad = [(0, side - h), (0, side - w)] + ([(0, 0)] if img.ndim == 3 else [])
            img = np.pad(img, pad, mode="constant", constant_values=255 if not enhance_dapi else 0)
        crop = (0, 0, side)

    left, top, side = crop
    pil = Image.fromarray(img).crop((left, top, left + side, top + side))
    out = np.asarray(pil.resize((canvas_size, canvas_size), Image.LANCZOS))
    final_uv = (coords - np.array([left, top])) * (canvas_size / side)
    if grayscale or enhance_dapi:
        out = to_grayscale(out)
    if enhance_dapi:
        out = clahe(out)

    adata.uns.setdefault(uns_key, {})
    adata.uns[uns_key]["image"] = out
    adata.uns[uns_key]["canvas_size"] = int(canvas_size)
    adata.obsm[key_added] = np.asarray(final_uv, dtype=np.float32)
    return adata


# =============================================================================
# Controlled morphology degradation (Fig. 2j, "Morphology ablation")
# =============================================================================
def _ensure_rng(seed):
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    return np.random.default_rng(seed), seed


def _to_float(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0, True
    return img.astype(np.float32), False


def _to_dtype(img_f, was_uint8):
    img_f = np.clip(img_f, 0.0, 1.0)
    return (img_f * 255.0).astype(np.uint8) if was_uint8 else img_f


def _lowfreq_noise(h, w, scale=48, sigma=6.0, seed=0):
    rng = np.random.default_rng(seed)
    hh, ww = max(4, h // scale), max(4, w // scale)
    x = gaussian_filter(rng.random((hh, ww)).astype(np.float32), sigma=1.0)
    up = np.kron(x, np.ones((int(np.ceil(h / hh)), int(np.ceil(w / ww))), dtype=np.float32))[:h, :w]
    up = gaussian_filter(up, sigma=sigma)
    return (up - up.min()) / (up.max() - up.min() + 1e-8)


def _jpeg_compress(img_f, quality):
    arr = (np.clip(img_f, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality), subsampling=2, optimize=False)
    buf.seek(0)
    out = np.asarray(Image.open(buf)).astype(np.float32) / 255.0
    return np.mean(out, axis=-1) if img_f.ndim == 2 else out


def _down_up(img_f, factor):
    H, W = img_f.shape[:2]
    newH, newW = max(1, int(H * factor)), max(1, int(W * factor))
    pil = Image.fromarray((np.clip(img_f, 0, 1) * 255).astype(np.uint8))
    back = np.asarray(pil.resize((newW, newH), Image.BILINEAR).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
    if img_f.ndim == 2 and back.ndim == 3:
        back = np.mean(back, axis=-1)
    return back


def _vignette(h, w, strength=0.5):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    r = r / (r.max() + 1e-8)
    return np.clip(1.0 - strength * (r**2), 0.2, 1.0)


def _soft_occlusion_mask(h, w, n_blobs=6, min_r=0.07, max_r=0.20, feather=0.15, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.zeros((h, w), np.float32)
    for _ in range(n_blobs):
        cy, cx = rng.uniform(0.1 * h, 0.9 * h), rng.uniform(0.1 * w, 0.9 * w)
        ry, rx = rng.uniform(min_r * h, max_r * h), rng.uniform(min_r * w, max_r * w)
        d = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        edge = 1.0 - np.clip((d - 1.0) / (feather + 1e-8), 0, 1)
        mask = np.maximum(mask, np.clip(edge, 0, 1) * (d <= 1.0 + feather))
    mask = gaussian_filter(mask, sigma=3.0)
    return (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)


def degrade_he_for_ablation(image: np.ndarray, level: float = 0.7, seed: Optional[int] = None) -> np.ndarray:
    """Spatially heterogeneous H&E degradation with strength ``level`` in [0, 1].

    Combines spatially varying defocus (Gaussian sigma 2/5/9 px), feathered
    occlusions, local resolution loss, stain/gamma drift, non-uniform
    illumination and JPEG compression at quality ``60 - 40 * level``.
    """
    rng, seed = _ensure_rng(seed)
    img_f, was_uint8 = _to_float(np.asarray(image))
    H, W = img_f.shape[:2]
    rng = np.random.default_rng(seed)

    focus_map = (_lowfreq_noise(H, W, scale=48, sigma=5.0, seed=seed) > 0.45 + 0.3 * level).astype(np.float32)
    focus_map = gaussian_filter(focus_map, sigma=8.0)
    focus_map = (focus_map - focus_map.min()) / (focus_map.max() - focus_map.min() + 1e-8)

    sigmas = [2, 5, 9]
    bins = len(sigmas)
    fm_scaled = np.clip(focus_map * (bins - 1), 0, bins - 1)
    weight_maps = [np.clip(1.0 - np.abs(fm_scaled - k), 0.0, 1.0) for k in range(bins)]
    weight_sum = np.maximum(1e-8, np.sum(weight_maps, axis=0))
    weight_maps = [w / weight_sum for w in weight_maps]

    if img_f.ndim == 2:
        blurs = [gaussian_filter(img_f, sigma=s) for s in sigmas]
        defocus = sum(w * b for w, b in zip(weight_maps, blurs))
    else:
        blurs = [np.stack([gaussian_filter(img_f[..., c], sigma=s) for c in range(img_f.shape[2])], axis=-1) for s in sigmas]
        defocus = sum(w[..., None] * b for w, b in zip(weight_maps, blurs))

    occ_mask = _soft_occlusion_mask(H, W, n_blobs=int(4 + 8 * level), min_r=0.06, max_r=0.22, feather=0.20, seed=seed + 1)
    occ_mask = occ_mask ** (0.2 + 0.8 * level)
    blend = np.clip(0.35 * focus_map + 0.65 * occ_mask, 0, 1)
    strength = 0.5 + 0.5 * level
    expand = (lambda m: m[..., None]) if img_f.ndim == 3 else (lambda m: m)
    comp = (1 - strength * expand(blend)) * img_f + strength * expand(blend) * defocus

    res_mask = _lowfreq_noise(H, W, scale=64, sigma=7.0, seed=seed + 2) ** 1.5
    low = _down_up(comp, factor=0.35 + 0.25 * level)
    comp = (1 - expand(res_mask)) * comp + expand(res_mask) * low

    if img_f.ndim == 3:
        gains = 1.0 + rng.normal(0, 0.15 * level, size=(3,))
        gamma = 1.0 + rng.normal(0, 0.25 * level)
        comp = np.power(np.clip(comp * gains[None, None, :], 0, 1), gamma)

    comp = comp * expand(_vignette(H, W, strength=0.3 * level))
    comp = _jpeg_compress(comp, quality=max(5, int(60 - 40 * level)))
    return _to_dtype(comp, was_uint8)


def random_deform_field(
    H,
    W,
    grid_res=32,
    smooth_sigma=8.0,
    amplitude=20.0,
    seed=None,
    zero_translation="image",
    coords=None,
    mask=None,
    match_amplitude=True,
):
    """Smooth random displacement field ``(dy, dx)`` with RMS ``amplitude`` pixels."""
    rng, seed = _ensure_rng(seed)
    hC, wC = max(4, H // grid_res) + 3, max(4, W // grid_res) + 3
    dyc = gaussian_filter(rng.normal(0, 1, (hC, wC)).astype(np.float32), sigma=smooth_sigma)
    dxc = gaussian_filter(rng.normal(0, 1, (hC, wC)).astype(np.float32), sigma=smooth_sigma)
    dyc -= dyc.mean()
    dxc -= dxc.mean()
    dyc *= amplitude / (np.std(dyc) + 1e-8)
    dxc *= amplitude / (np.std(dxc) + 1e-8)

    yy, xx = np.meshgrid(np.linspace(0, hC - 1, H, dtype=np.float32), np.linspace(0, wC - 1, W, dtype=np.float32), indexing="ij")
    dy = map_coordinates(dyc, [yy, xx], order=1, mode="reflect")
    dx = map_coordinates(dxc, [yy, xx], order=1, mode="reflect")

    if zero_translation == "image":
        dy -= dy.mean()
        dx -= dx.mean()
    elif zero_translation == "coords" and coords is not None:
        xq = np.clip(coords[:, 0].astype(np.float32), 0, W - 1)
        yq = np.clip(coords[:, 1].astype(np.float32), 0, H - 1)
        dx -= map_coordinates(dx, [yq, xq], order=1, mode="reflect").mean()
        dy -= map_coordinates(dy, [yq, xq], order=1, mode="reflect").mean()
    elif zero_translation == "mask" and mask is not None:
        M = mask.astype(np.float32)
        M /= M.sum() + 1e-8
        dx -= (dx * M).sum()
        dy -= (dy * M).sum()
    elif zero_translation == "center":
        dx -= map_coordinates(dx, [[H / 2.0], [W / 2.0]], order=1, mode="reflect")[0]
        dy -= map_coordinates(dy, [[H / 2.0], [W / 2.0]], order=1, mode="reflect")[0]

    if match_amplitude:
        scale = amplitude / (np.sqrt((dx**2 + dy**2).mean()) + 1e-8)
        dx *= scale
        dy *= scale
    return dy, dx
