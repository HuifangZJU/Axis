# Spatially matched analysis

Every sample is transported into a common reference with its own transition
field and reconstructed on the reference support, independently of the other
samples (\(\beta_r = 1\)). Matched locations can then be compared directly.

```python
matched = axis.tl.match(
    adatas, fields, reference=0, reconstruct=True,
    attr_keys=["axis_joint_domain"], spatial_bandwidth=1.0, max_dist=8,
)
```

## Location-wise contrasts

```python
delta = axis.tl.contrast(matched_ad, matched_wt, genes=["Gfap", "Arc", "Mobp"])
# delta.X = log1p(normalised AD) - log1p(normalised WT) at every matched location
```

Library sizes are always computed from all genes of each location
(`axis.matching.contrast`), so selecting marker genes does not change the
normalisation.

## Temporal trajectories

With three matched states \(\tau_1<\tau_2<\tau_3\), consecutive changes
\(\Delta_{12}, \Delta_{23}\) at each location classify it as increase,
decrease, mid-stage peak, mid-stage trough or stable (Fig. 5g–i); see
`reproducibility/figure5/5ghi_pixel_delta_analysis.py` for the full analysis.
