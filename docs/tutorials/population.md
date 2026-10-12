# Population-level integration

All samples are transported into the reference frame, pooled with equal total
mass per sample and coupled by sparse Sinkhorn transport with a mixed spatial /
molecular cost (α = 0.6) to a lattice in that frame (`grid`, about
`grid_size**2` points over the canvas), restricted to the largest connected
region of lattice points within `max_dist` (5 canvas px by default) of a
transported location of any sample. The coupled profiles are denoised by
adaptive singular-value thresholding and edge-preserving anisotropic diffusion,
iterating coupling and low-rank reconstruction (Eqs. 36–39).

The example uses the DLPFC sections of Fig. 4h–k (expert layer labels in
`obs["layer_guess_reordered"]`, `grid_size=80` as in
`reproducibility/pipelines/generation/dlpfc_population.py`). On the quickstart
sample use `attr_keys=["original_clusters"]` and `grid_size=40`, the SCC spot
density used in the quickstart (the lattice scales with the canvas, so this
holds on the 512- and the 256-px canvas).

```python
population = axis.tl.integrate(
    adatas, fields, reference=0,
    grid="visium", grid_size=80, attr_keys=["layer_guess_reordered"],
    alpha=0.6, keep_nuc_frac=0.6, fuse_radius=10, pca_dim=32,
)
```

Laminar organisation can be scored with
`axis.metrics.kruskal_eta_h_squared` and `axis.metrics.adjacent_layer_aurocs`
(Fig. 4i).
