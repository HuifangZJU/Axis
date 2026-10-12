# Transition fields

A transition field \(D_{a\to b}\) is a dense displacement field on the canvas of
sample *b* that registers sample *a* onto it (pull-back convention:
\(\tilde M_a(p) = M_a(p + D_{a\to b}(p))\)). AXIS learns one network for all
ordered pairs of a sample series.

```python
import axis

adatas = axis.datasets.toy_series(n_samples=3, canvas_size=128)   # or your data
axis.pp.build_pseudoimages(adatas, n_comps=6)

fields = axis.tl.fit_transition(
    adatas,
    fusion="transformer",      # "linear" for large Xenium / PhenoCycler canvases
    morph_win=21, mol_win=15,  # local-NCC windows (Eqs. 21-22)
    lambda_morph=1.0, lambda_mol=1.0, lambda_smooth=1.0,
    epochs=1000,               # 1000 epochs x 6 ordered pairs = 6000 steps: hours on CPU; use a GPU or cap with max_steps=...
)
```

## Using the fields

```python
coords = adatas[2].obsm["spatial_canvas"]        # canvas coordinates (x, y) of sample 2
moving_image = adatas[2].uns["axis"]["image"]     # canvas image of sample 2

fields[(0, 2)]                                    # (2, H, W) (dy, dx) field in canvas px: D_{2→0}
fields.transport(coords, fixed=0, moving=2)             # spots of 2 in the frame of 0
fields.transport(coords, fixed=0, moving=2, alpha=0.5)  # half-way (Eq. 25)

from axis.registration import warp_image, upsample_flow
warped = warp_image(moving_image, fields[(0, 2)])        # canvas-frame image at any resolution
hires_field = upsample_flow(fields[(0, 2)], (2048, 2048))  # resolution transfer (Fig. 2k-m)
```

Fields are indexed by `(fixed, moving)`, so `fields[(b, a)]` is
\(D_{a\to b}\). Channel 0 is the displacement along *y* (rows) and channel 1
along *x* (columns), in canvas pixels; keep this order when plotting or
transporting points yourself.

`warp_image` resizes the field to the image's height and width, so the image
must cover the canvas frame (the square crop or padding made by
`prepare_canvas`) — at any resolution. A non-square original image would be
warped with a stretched field.

`fields.save(dir)` writes `{fixed}_{moving}_flow.npy`, the layout used by the
reproducibility pipelines; `axis.tl.TransitionFields.load(dir)` reads it back.

## Starting from a checkpoint

```python
fields = axis.tl.fit_transition(adatas, checkpoint="model_1000.pth", epochs=0)
```

Checkpoints of the original training scripts use the same parameter names
(checked against the released architecture in `tests/test_registration.py`).
With `checkpoint=`, the network is built from the current inputs and arguments
— morphology channels (3 for RGB, 1 for grayscale), `n_comps` of
`build_pseudoimages` and `fusion` (default `"transformer"`; a configuration
stored in the checkpoint is not used) — and the parameters with matching names
and shapes are copied. These settings must therefore match the checkpoint
(`fusion="linear"` and grayscale input for the Xenium and PhenoCycler models).
With `epochs=0`, a checkpoint that leaves any network parameter untrained raises
a `ValueError`. To build the network from an AXIS checkpoint's stored
configuration, or from explicit arguments for an original `state_dict`, load it
strictly and pass it as `model=`:

```python
model = axis.registration.load_model("model_1000.pth", in_channels=3, in_gene_channels=10, fusion_module="transformer")
fields = axis.tl.fit_transition(adatas, model=model, epochs=0)
```

An image-only checkpoint can warm start a multimodal network that is then
trained further (`epochs > 0`; the fusion layers start untrained, which the log
reports), as in the original training scripts
(`reproducibility/pipelines/training/train_multimodal*.py`). `out_dir=` with
`save_every=` writes `model_{epoch}.pth` and `model_final.pth`.

## Evaluating alignment

The domain metrics need the `metrics` extra (alphashape, shapely,
scikit-image; see [Installation](../installation.md)).

```python
from axis.metrics import evaluate_domain_metrics, evaluate_neighborhood_metrics

ref_xy, ref_labels = adatas[0].obsm["spatial_canvas"], adatas[0].obs["domain"]   # "original_clusters" for SCC
warped_xy = fields.transport(adatas[2].obsm["spatial_canvas"], fixed=0, moving=2)
labels = adatas[2].obs["domain"]
size = fields.canvas_size

dom, _, _ = evaluate_domain_metrics(ref_xy, ref_labels, warped_xy, labels, mask_shape=(size, size), alpha=0.05)
nbh, _ = evaluate_neighborhood_metrics(ref_xy, ref_labels, warped_xy, labels, distance_threshold=20)
```

`dom` has one row per domain with `dice`, `spatial_corr` (Pearson correlation
of the Gaussian-smoothed domain density maps; the column the Fig. 2 scripts
report), `density_cosine` (the normalised inner product \(S_c\) written in
Methods) and `center_shift` (canvas px); `nbh` holds the molecular-neighbourhood
correspondence (Methods, "Evaluation").
