# Quickstart: SCC patient 10

This example uses three serial 10x Visium sections of a cutaneous squamous
cell carcinoma collected at 0, 10 and 20 µm (Ji *et al.*, *Cell* 2020). It
learns transition fields, aligns the sections, and generates the 10-µm section
from its two neighbours while the measured 10-µm section is held out.

## Data

```bash
pip install gdown
gdown 1JEzGLEK23lgcn60QVaQANiKG3BpiX1BK -O scc_patient10.zip
unzip scc_patient10.zip -d scc_patient10      # 10_0.h5ad, 10_1.h5ad, 10_2.h5ad
```

Each file holds raw counts (`X`), the 512-px H&E canvas
(`uns["image_array"]`), canvas coordinates (`obsm["spatial_image_coor"]`), spot
clusters (`obs["original_clusters"]`) and the published AXIS alignment to
section 0 (`obsm["spatial_axis"]`).

## 1 · Canvases and pseudo-images

```python
import axis

adatas = axis.datasets.load_scc_sample("scc_patient10", patient=10)
for a in adatas:
    axis.pp.prepare_canvas(a, spatial_key="spatial_image_coor", canvas_size=512)
axis.pp.build_pseudoimages(adatas, n_comps=10, preset="visium")
axis.pl.pseudoimage(adatas[0])
```

`build_pseudoimages` normalises counts (10⁴ per spot, `log1p`), projects the
shared genes of all sections onto 10 principal components and rasterises every
component with a Gaussian kernel (Methods Eq. 1) on a 1/8-resolution grid.

## 2 · Transition fields

```python
fields = axis.tl.fit_transition(
    adatas, morph_win=9, mol_win=5, lambda_mol=0.5,   # SCC settings (configs/scc.yaml)
    max_steps=3000,                                   # 500 epochs over the 6 ordered pairs
)
fields.save("scc_fields")                             # {fixed}_{moving}_flow.npy
axis.pl.loss(fields.history)
axis.pl.registration(adatas[0].uns["axis"]["image"], adatas[2].uns["axis"]["image"], fields[(0, 2)])
```

!!! tip "Hardware"
    Training at 512 px is best done on a GPU (on CPU it needs more than 14 GB of RAM). On a CPU use
    `canvas_size=256` in `prepare_canvas` and fewer steps: 400 steps took
    55 min on 4 CPU cores. Inference at 512 px takes ~12 s per pair on the same
    CPU. A trained checkpoint can be supplied with `checkpoint=` (and `epochs=0`
    to skip training); the whole quickstart then runs in about 1.5 min.

## 3 · Spatially matched analysis

```python
axis.tl.match(adatas, fields, reference=0)            # writes obsm["spatial_axis"]
axis.pl.matched(adatas, color="original_clusters")

matched = axis.tl.match(adatas, fields, reference=0, reconstruct=True,
                        attr_keys=["original_clusters"])   # all sections on section 0's spots
delta = axis.tl.contrast(matched[2], matched[0], genes=["KRT1", "COL1A1"])
```

## 4 · Virtual tissue at 10 µm

```python
vt = axis.tl.generate(
    adatas, fields, positions=[0, 10, 20], query=10, anchors=[0, 2],
    grid="visium", grid_size=40,
    attr_keys=["original_clusters"], spatial_bandwidth=3, attr_spatial_bandwidth=4, max_dist=10,
)
axis.pl.spatial(vt, "original_clusters")
axis.pl.spatial(vt, "KRT1")
vt.write_h5ad("virtual_10um.h5ad")
```

`vt.uns["axis_generation"]` records the anchors, relative position α = 0.5 and
anchor weights β = (0.5, 0.5).

## Script version

```bash
python examples/quickstart_scc.py --data-dir scc_patient10 --out quickstart_out
```

writes the figures shown in the [gallery](gallery.md) and a `report.json`
comparing the learned alignment with the published one and the generated
section with the held-out measurement.
