# Population-level integration

All samples are matched to a reference, pooled with equal total mass per
sample and coupled to the reference support by sparse Sinkhorn transport with
a mixed spatial / molecular cost (α = 0.6). The coupled profiles are denoised
by adaptive singular-value thresholding and edge-preserving anisotropic
diffusion, iterating coupling and low-rank reconstruction (Eqs. 36–39).

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
