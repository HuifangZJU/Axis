# Virtual tissue generation

For a query \(\tau_q\) between anchors \(\tau_a < \tau_q < \tau_b\), AXIS

1. scales the bidirectional fields by \(\alpha_q = (\tau_q-\tau_a)/(\tau_b-\tau_a)\)
   and transports both anchors into the query anatomy (Eqs. 24–26). Because the
   fields follow the pull-back convention of
   [Transition fields](transition_fields.md), Eq. 26 is applied through the
   nearest-neighbour inverse used by `fields.transport`: a spot of anchor *a* at
   \(x\) is placed at the canvas pixel \(p\) whose position
   \(p + \alpha_q D_{a\to b}(p)\) is nearest to \(x\), and anchor *b* is handled
   likewise with \((1-\alpha_q) D_{b\to a}\). Points therefore move by about
   \(-\alpha_q D_{a\to b}(x)\), not by \(+\alpha_q D_{a\to b}(x)\) as Eq. 26 is
   written;
2. pools the transported measurements into an *evidence plane* (Eq. 27);
3. keeps the output locations supported by the transported evidence (within
   `max_dist` canvas px of an evidence point, after a neighbourhood smoothing of
   this mask on the output grid, restricted to its largest connected region) and
   generates each of them as a weighted average of nearby evidence, with weights
   combining the biological-axis distance of the source anchor, the local
   spatial distance and the local molecular quality (Eqs. 28–35).

The first example continues the [Quickstart](../quickstart.md) (SCC patient 10,
512-px canvas):

```python
vt = axis.tl.generate(
    adatas, fields,
    positions=[0, 10, 20],      # one position per sample along the biological axis
    query=10, anchors=[0, 2],   # candidate anchors (default: all samples); the two bracketing the query are used
    grid="visium", grid_size=40,             # or grid="pixel", or an (M, 2) array (e.g. cells)
    attr_keys=["original_clusters"],          # annotations generated on the same support
    spatial_bandwidth=3,                      # clusters use the same bandwidth (Fig. 3); attr_spatial_bandwidth=... sets their own
    max_dist=10,                              # tissue-support threshold (canvas px)
)
```

`vt.obs` contains generated annotations plus `valid`, `confidence` and
`nearest_dist`; `vt.uns["axis_generation"]` stores α and the anchor weights β.
Unsupported locations stay in `vt` with `valid=False`, zero expression, missing
annotations and `nearest_dist = inf`; use `vt[vt.obs["valid"]]` for analyses.
Because only the largest connected supported region is kept, separate tissue
pieces are dropped.

## Dense series and 3D volumes

This example uses the four DLPFC sections of one donor (Fig. 4; see
`configs/dlpfc.yaml`):

```python
depths = range(40, 310, 30)
series = {z: axis.tl.generate(adatas, fields, positions=[0, 10, 310, 320], query=z, anchors=[1, 2], genes=["PCP4"],
                              grid="visium", grid_size=80, attr_keys=["layer_guess_reordered"],
                              spatial_bandwidth=1, attr_spatial_bandwidth=6, max_dist=5)
          for z in depths}
```

Each output stores a dense `X` over the selected genes for every grid location,
so restrict `genes` for dense series. The molecular-quality weight is computed
from the selected genes, as in `reproducibility/pipelines/generation/dlpfc_depth.py`,
which generated `PCP4` alone. Stacking the outputs with their depth gives the
virtual 3D tissue of Fig. 4.

## Weights

```python
axis.pl.axis_weights([0, 20], query=10)   # biological-axis weights (σ_τ = spacing / 3)
```

The low-level pieces (`axis.generation.build_query_evidence`,
`prepare_local_query_support`, `generate_virtual_tissue_fixed_grid`) can be
combined directly for custom supports or weighting.
