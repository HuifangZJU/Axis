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
                           scale=1.0,             # e.g. Visium tissue_hires_scalef
                           crop=None)             # (left, top, side) to crop around the tissue
    adatas.append(a)
axis.pp.build_pseudoimages(adatas, n_comps=10, preset="visium")
```

Guidelines

* **Canvas size** must be divisible by 16; use 512 for Visium-like data and
  1024 for single-cell platforms (`preset="xenium"`, `fusion="linear"`).
* **Different image modalities** (H&E vs DAPI): pass `grayscale=True` so all
  samples share the lowest channel dimensionality; `enhance_dapi=True` applies
  the CLAHE enhancement used for Xenium.
* **Different molecular modalities**: build pseudo-images from features shared
  across modalities (e.g. cognate gene–protein pairs).
* **Store parameters** in a YAML config (see `configs/`) and load it with
  `axis.config.load_config(path)`.
