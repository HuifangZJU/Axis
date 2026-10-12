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
    epochs=1000,               # or max_steps=...
)
```

## Using the fields

```python
fields[(0, 2)]                       # (2, H, W) field registering sample 2 onto sample 0
fields.transport(coords, fixed=0, moving=2)             # spots of 2 in the frame of 0
fields.transport(coords, fixed=0, moving=2, alpha=0.5)  # half-way (Eq. 25)

from axis.registration import warp_image, upsample_flow
warped = warp_image(moving_image, fields[(0, 2)])        # any image resolution
hires_field = upsample_flow(fields[(0, 2)], (2048, 2048))  # resolution transfer (Fig. 2k-m)
```

`fields.save(dir)` writes `{fixed}_{moving}_flow.npy`, the layout used by the
reproducibility pipelines; `axis.tl.TransitionFields.load(dir)` reads it back.

## Starting from a checkpoint

```python
fields = axis.tl.fit_transition(adatas, checkpoint="model_1000.pth", epochs=0)
```

Checkpoints written by the original training scripts load unchanged: the
network keeps the original parameter names (tested against the released
architecture in `tests/test_registration.py`). Image-only checkpoints can warm
start a multimodal network — matching parameters are copied, as in the paper.

## Evaluating alignment

```python
from axis.metrics import evaluate_domain_metrics, evaluate_neighborhood_metrics

dom, _, _ = evaluate_domain_metrics(ref_xy, ref_labels, warped_xy, labels, mask_shape=(512, 512), alpha=0.05)
nbh, _ = evaluate_neighborhood_metrics(ref_xy, ref_labels, warped_xy, labels, distance_threshold=20)
```

`dom` contains Dice, spatial correlation and centre shift per domain;
`nbh` the molecular-neighbourhood correspondence (Methods, "Evaluation").
