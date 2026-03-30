from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib as mpl
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
from PIL import Image
import io
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# --- helper: make RNG (auto-seed if None) ---
def _ensure_rng(seed):
    if seed is None:  # new random seed every call
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    return np.random.default_rng(seed), seed
# ----------------- helpers -----------------
def _to_float(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0, True
    return img.astype(np.float32), False

def _to_dtype(img_f, was_uint8):
    img_f = np.clip(img_f, 0.0, 1.0)
    return (img_f * 255.0).astype(np.uint8) if was_uint8 else img_f

def _lowfreq_noise(h, w, scale=48, sigma=6.0, seed=0):
    rng = np.random.default_rng(seed)
    hh = max(4, h // scale); ww = max(4, w // scale)
    x = rng.random((hh, ww)).astype(np.float32)
    x = gaussian_filter(x, sigma=1.0)
    up = np.kron(x, np.ones((int(np.ceil(h/hh)), int(np.ceil(w/ww))), dtype=np.float32))[:h, :w]
    up = gaussian_filter(up, sigma=sigma)
    up = (up - up.min()) / (up.max() - up.min() + 1e-8)
    return up

def _jpeg_compress(img_f, quality):
    # expects float [0,1]
    arr = (np.clip(img_f,0,1)*255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality), subsampling=2, optimize=False)
    buf.seek(0)
    out = np.asarray(Image.open(buf)).astype(np.float32)/255.0
    if img_f.ndim == 2:
        out = np.mean(out, axis=-1)  # keep grayscale if input was gray
    return out

def _down_up(img_f, factor):
    H, W = img_f.shape[:2]
    newH, newW = max(1,int(H*factor)), max(1,int(W*factor))
    pil = Image.fromarray((np.clip(img_f,0,1)*255).astype(np.uint8) if img_f.ndim==2
                          else (np.clip(img_f,0,1)*255).astype(np.uint8))
    small = pil.resize((newW, newH), Image.BILINEAR)
    back = small.resize((W, H), Image.BILINEAR)
    back = np.asarray(back).astype(np.float32)/255.0
    if img_f.ndim == 2 and back.ndim == 3:
        back = np.mean(back, axis=-1)
    return back

def _vignette(h, w, strength=0.5):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h/2.0, w/2.0
    r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    r = r / (r.max() + 1e-8)
    v = 1.0 - strength * (r**2)
    return np.clip(v, 0.2, 1.0)

def _soft_occlusion_mask(h, w, n_blobs=6, min_r=0.07, max_r=0.20, feather=0.15, seed=0):
    """Smooth ‘blob’ mask in [0,1] where 1=occlude/blur more."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.zeros((h,w), np.float32)
    for _ in range(n_blobs):
        cy = rng.uniform(0.1*h, 0.9*h)
        cx = rng.uniform(0.1*w, 0.9*w)
        ry = rng.uniform(min_r*h, max_r*h)
        rx = rng.uniform(min_r*w, max_r*w)
        # ellipse distance
        d = ((yy-cy)/ry)**2 + ((xx-cx)/rx)**2
        blob = (d <= 1.0).astype(np.float32)
        # feather edges by distance transform
        edge = 1.0 - np.clip((d - 1.0) / (feather + 1e-8), 0, 1)
        blob = np.clip(edge, 0, 1) * (d <= 1.0 + feather)
        mask = np.maximum(mask, blob)
    mask = gaussian_filter(mask, sigma=3.0)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask

def degrade_he_for_ablation(image, level=0.7, seed=None):
    """
    Spatially non-uniform, realistic degradation for H&E ablation.
    level in [0,1]: higher => stronger degradation.
    """
    rng, seed = _ensure_rng(seed)
    img_f, was_uint8 = _to_float(image)
    H, W = img_f.shape[:2]
    rng = np.random.default_rng(seed)

    # 1) Spatially varying defocus mask (low-frequency) to blend multi-blurs
    focus_map = _lowfreq_noise(H, W, scale=48, sigma=5.0, seed=seed)
    # make it more binary-ish to create larger out-of-focus regions
    t = 0.45 + 0.3*level
    focus_map = (focus_map > t).astype(np.float32)
    focus_map = gaussian_filter(focus_map, sigma=8.0)
    focus_map = (focus_map - focus_map.min()) / (focus_map.max() - focus_map.min() + 1e-8)

    # multi-level blurs to blend
    sigmas = [2, 5, 9]  # mix of mild to heavy blur
    # pick weights by mapping focus_map to {0..len(sigmas)-1} bins softly
    bins = len(sigmas)
    fm_scaled = np.clip(focus_map * (bins-1), 0, bins-1)
    # soft assignment via triangular weights
    Wmaps = []
    for k in range(bins):
        w = np.clip(1.0 - np.abs(fm_scaled - k), 0.0, 1.0)
        Wmaps.append(w)
    Wsum = np.maximum(1e-8, np.sum(Wmaps, axis=0))
    Wmaps = [w/Wsum for w in Wmaps]

    if img_f.ndim == 2:
        blurs = [gaussian_filter(img_f, sigma=s) for s in sigmas]
        defocus = sum(w[...,None] * b[...,None] if b.ndim==2 else w * b for w, b in zip(Wmaps, blurs))
        defocus = defocus.squeeze() if defocus.ndim==3 else defocus
    else:
        blurs = [np.stack([gaussian_filter(img_f[...,c], sigma=s) for c in range(img_f.shape[2])], axis=-1) for s in sigmas]
        defocus = sum(w[...,None]*b for w,b in zip(Wmaps, blurs))

    # 2) Soft occlusions (hide regions smoothly)
    occ_mask = _soft_occlusion_mask(H, W,
                                    n_blobs=int(4 + 8*level),
                                    min_r=0.06, max_r=0.22,
                                    feather=0.20, seed=seed+1)
    # raised power to sharpen blob interiors with smooth edges
    occ_mask = occ_mask ** (0.2 + 0.8*level)

    # 3) Blend original with defocus using a composite mask (focus + occlusion)
    # more weight on defocus in occluded regions
    blend_mask = np.clip(0.35*focus_map + 0.65*occ_mask, 0, 1)
    # increase strength with level
    strength = 0.5 + 0.5*level
    comp = (1 - strength*blend_mask[...,None]) * img_f + (strength*blend_mask[...,None]) * defocus

    # 4) Downsample+upsample (resolution loss) — spatially varying factor
    # create a bias field to decide where resolution is worse
    res_mask = _lowfreq_noise(H, W, scale=64, sigma=7.0, seed=seed+2)
    res_mask = res_mask**(1.5)  # push to 0 or 1
    low = _down_up(comp, factor=0.35 + 0.25*level)  # global LR
    comp = (1 - res_mask[...,None]) * comp + res_mask[...,None] * low

    # 5) Stain drift / desaturation (simple but effective)
    # subtle color channel scaling + gamma drift
    if img_f.ndim == 3:
        gains = 1.0 + rng.normal(0, 0.15*level, size=(3,))
        gamma = 1.0 + rng.normal(0, 0.25*level)
        comp = np.power(np.clip(comp * gains[None,None,:], 0, 1), gamma)

    # 6) Vignette / illumination non-uniformity
    vig = _vignette(H, W, strength=0.3*level)
    comp = comp * (vig[...,None] if comp.ndim==3 else vig)

    # 7) JPEG artifacts (strong at higher levels)
    q = int(60 - 40*level)  # 60→20
    comp = _jpeg_compress(comp, quality=max(5, q))

    return _to_dtype(comp, was_uint8)

# ----------------- deformation (same as before, works with this) -----------------

def random_deform_field(
    H, W,
    grid_res=32,
    smooth_sigma=8.0,
    amplitude=20.0,
    seed=None,
    zero_translation="image",  # "image" | "coords" | "mask" | "center"
    coords=None,
    mask=None,
    match_amplitude=True
):
    rng, seed = _ensure_rng(seed)

    # --- coarse noise, smoothed ---
    hC = max(4, H // grid_res) + 3
    wC = max(4, W // grid_res) + 3
    dyc = rng.normal(0, 1, (hC, wC)).astype(np.float32)
    dxc = rng.normal(0, 1, (hC, wC)).astype(np.float32)
    dyc = gaussian_filter(dyc, sigma=smooth_sigma)
    dxc = gaussian_filter(dxc, sigma=smooth_sigma)
    dyc -= dyc.mean(); dxc -= dxc.mean()
    dyc *= (amplitude / (np.std(dyc) + 1e-8))
    dxc *= (amplitude / (np.std(dxc) + 1e-8))

    # --- upsample ---
    yy, xx = np.meshgrid(
        np.linspace(0, hC-1, H, dtype=np.float32),
        np.linspace(0, wC-1, W, dtype=np.float32),
        indexing="ij"
    )
    dy = map_coordinates(dyc, [yy, xx], order=1, mode="reflect")
    dx = map_coordinates(dxc, [yy, xx], order=1, mode="reflect")

    # --- remove translation as you configured ---
    if zero_translation == "image":
        dy -= dy.mean(); dx -= dx.mean()
    elif zero_translation == "coords" and coords is not None:
        xq = np.clip(coords[:,0].astype(np.float32), 0, W-1)
        yq = np.clip(coords[:,1].astype(np.float32), 0, H-1)
        dx -= map_coordinates(dx, [yq, xq], order=1, mode="reflect").mean()
        dy -= map_coordinates(dy, [yq, xq], order=1, mode="reflect").mean()
    elif zero_translation == "mask" and mask is not None:
        M = (mask.astype(np.float32)); M /= (M.sum() + 1e-8)
        dx -= (dx*M).sum(); dy -= (dy*M).sum()
    elif zero_translation == "center":
        cy, cx = H/2.0, W/2.0
        dx -= map_coordinates(dx, [[cy],[cx]], order=1, mode="reflect")[0]
        dy -= map_coordinates(dy, [[cy],[cx]], order=1, mode="reflect")[0]

    if match_amplitude:
        rms = np.sqrt((dx**2 + dy**2).mean()) + 1e-8
        scale = amplitude / rms
        dx *= scale; dy *= scale

    return dy, dx

