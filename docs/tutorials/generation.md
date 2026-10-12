# Virtual tissue generation

For a query \(\tau_q\) between anchors \(\tau_a < \tau_q < \tau_b\), AXIS

1. scales the bidirectional fields by \(\alpha_q = (\tau_q-\tau_a)/(\tau_b-\tau_a)\)
   and transports both anchors into the query anatomy (Eqs. 24–26);
2. pools the transported measurements into an *evidence plane* (Eq. 27);
3. generates every output location as a weighted average of nearby evidence,
   with weights combining the biological-axis distance of the source anchor,
   the local spatial distance and the local molecular quality (Eqs. 28–35).

```python
vt = axis.tl.generate(
    adatas, fields,
    positions=[0, 10, 20],      # one position per sample along the biological axis
    query=10, anchors=[0, 2],   # bracketing anchors are chosen automatically
    grid="visium", grid_size=40,             # or grid="pixel", or an (M, 2) array (e.g. cells)
    attr_keys=["original_clusters"],          # annotations generated on the same support
    spatial_bandwidth=3, attr_spatial_bandwidth=4,
    max_dist=10,                              # tissue-support threshold (canvas px)
)
```

`vt.obs` contains generated annotations plus `valid`, `confidence` and
`nearest_dist`; `vt.uns["axis_generation"]` stores α and the anchor weights β.

## Dense series and 3D volumes

```python
depths = range(40, 310, 30)
series = {z: axis.tl.generate(adatas, fields, positions=[0, 10, 310, 320], query=z, anchors=[1, 2],
                              grid="visium", grid_size=80, attr_keys=["layer_guess_reordered"],
                              spatial_bandwidth=1, attr_spatial_bandwidth=6, max_dist=5)
          for z in depths}
```

Stacking the outputs with their depth gives the virtual 3D tissue of Fig. 4.

## Weights

```python
axis.pl.axis_weights([0, 20], query=10)   # biological-axis weights (σ_τ = spacing / 3)
```

The low-level pieces (`axis.generation.build_query_evidence`,
`prepare_local_query_support`, `generate_virtual_tissue_fixed_grid`) can be
combined directly for custom supports or weighting.
