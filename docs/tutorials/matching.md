# Spatially matched analysis

Every sample is transported into a common reference with its own transition
field and reconstructed on the reference support, independently of the other
samples (\(\beta_r = 1\)). Matched locations can then be compared directly.
The examples use the Xenium mouse-brain AD/WT sections of Figs 5–6 (not
distributed with AXIS), whose `obs["axis_joint_domain"]` holds the joint domain
labels of Methods; on the quickstart SCC sample use
`attr_keys=["original_clusters"]` and human genes such as `["KRT1", "COL1A1"]`.

```python
matched = axis.tl.match(
    adatas, fields, reference=0, reconstruct=True,
    attr_keys=["axis_joint_domain"], spatial_bandwidth=1.0, attr_spatial_bandwidth=4.0, max_dist=8,
)
```

This reconstructs every sample on the reference sample's own locations. The
Fig. 5–6 pipelines (`reproducibility/pipelines/generation/mouse_ad_*.py`)
reconstruct on a 256 × 256 pixel grid over the 1024-px canvas with at least
three contributing neighbours instead: add `grid="pixel", grid_size=256,
min_neighbors=3`.

## Location-wise contrasts

```python
import numpy as np

matched_ad, matched_wt = matched[0], matched[1]   # e.g. adatas = [AD, WT] at one age; AD is the reference
# Methods normalise all states to one target: the median per-location library size of the reconstructions
lib = np.concatenate([np.asarray(m.X[m.obs["valid"].to_numpy()].sum(axis=1)).ravel() for m in matched])
delta = axis.tl.contrast(matched_ad, matched_wt, genes=["Gfap", "Arc", "Mobp"], target_sum=float(np.median(lib)))
# delta.X = log1p(normalised AD) - log1p(normalised WT) at the locations valid in both (obs["valid"]);
# normalised = counts / library size * target_sum (1e4 if target_sum is not given)
```

Library sizes are computed from all genes of each matched location
(`axis.matching.contrast`), so selecting genes in `contrast` does not change the
normalisation. The matched objects contain only the genes shared by all samples,
and only the genes passed to `match(genes=...)` if given, so reconstruct all genes
when the contrasts will be normalised.

## Temporal trajectories

With three matched states \(\tau_1<\tau_2<\tau_3\), consecutive changes
\(\Delta_{12}, \Delta_{23}\) at each location classify it as increase,
decrease, mid-stage peak, mid-stage trough or stable (Fig. 5g–i); see
`reproducibility/figure5/5ghi_pixel_delta_analysis.py` for the full analysis.
