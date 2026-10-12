# Your own data

AXIS needs, for every sample, an AnnData with spot/cell coordinates and a
morphology image in the same pixel frame.

```python
import anndata as ad
import numpy as np
import axis
from PIL import Image

adatas = []
for path, img in [("s0.h5ad", "s0.png"), ("s1.h5ad", "s1.png")]:
    a = ad.read_h5ad(path)                       # counts in X, coordinates in obsm["spatial"]
    image = np.asarray(Image.open(img))
    axis.pp.prepare_canvas(a, image, spatial_key="spatial", canvas_size=512,
                           scale=1.0,             # image px per obsm["spatial"] unit, e.g. tissue_hires_scalef for a Visium hires PNG
                           crop=None)             # (left, top, side) to crop around the tissue
    adatas.append(a)
axis.pp.build_pseudoimages(adatas, n_comps=10, preset="visium")
```

`obsm["spatial"]` holds `(x, y)` = (column, row) coordinates. Spots that fall
outside the canvas are silently left out of the pseudo-image, so check that
`a.obsm["spatial_canvas"]` lies within `[0, canvas_size)` and that
`a.uns["axis"]["mask"].sum() > 0` (or look at `axis.pl.pseudoimage(a)`).

Guidelines

* **Canvas size** must be divisible by 16; use 512 for Visium-like data and
  1024 for single-cell platforms (`preset="xenium"`, `fusion="linear"`).
* **Image formats**: 8-bit images and floats in [0, 1] or [0, 255] are used as
  they are; images with larger values (12/16-bit DAPI) are rescaled linearly by
  their maximum, and an alpha channel is dropped. If a few saturated pixels
  dominate a fluorescence image, clip it at a high percentile first.
* **Different image modalities** (H&E vs DAPI): pass `grayscale=True` so all
  samples share the lowest channel dimensionality; `enhance_dapi=True` applies
  the CLAHE enhancement used for Xenium.
* **Different molecular modalities**: give the shared features identical
  `var_names` in every sample (e.g. rename protein CD45RO to PTPRC).
  `build_pseudoimages` uses only the intersection of `var_names`, and `n_comps`
  cannot exceed the number of shared features (`n_comps=3` for the three LUAD
  marker pairs in `configs/phenocycler_luad.yaml`).
* **Coordinates of the results** (`obsm["spatial_canvas"]`,
  `obsm["spatial_axis"]`, `vt.obsm["spatial"]`) are canvas pixels.
  `prepare_canvas` maps `xy` to `(xy * scale - (left, top)) * canvas_size / side`
  (without `crop`: `left = top = 0`, `side` = longer image side) and does not
  store this transform, so keep `scale` and `crop` to map results back with
  `xy = (canvas_xy * side / canvas_size + (left, top)) / scale`.
* **Store parameters** in a YAML config (see `configs/`) and load it with
  `axis.config.load_config(path)`.
