# Method summary

This page maps the Methods of the manuscript to the implementation.

## Pseudo-images (`axis.pp`)

`axis.pp.build_pseudoimages` processes the molecular matrices
\(G_k \in \mathbb{R}^{P\times N_k}\) of all samples in four steps, after the
optional per-sample filters (`min_cells`, `min_genes`):

1. Normalise each sample to `target_sum` (default \(10^4\)) counts per entity,
   with library sizes over all of its features, and \(\log(1+x)\)-transform
   (`normalize=True`).
2. Restrict to the features shared by all samples.
3. Concatenate and project to \(C_p\) principal components (`n_comps`,
   default 10).
4. Min-max scale each component to [0, 1] over all samples.

Methods Eq. (1) then rasterises each component with a Gaussian kernel,

\[
P_k^{(c)}(u)=\frac{\sum_i K_\sigma(\lVert u-x_{k,i}\rVert)\, z^{(c)}_{k,i}}{\sum_i K_\sigma(\lVert u-x_{k,i}\rVert)+\varepsilon},
\qquad K_\sigma(r)=e^{-r^2/2\sigma^2},
\]

with \(\sigma\) half a grid cell.

The implementation (`axis.pp.get_gene_feature_matrix_soft`) evaluates the
kernel at the centres of cells of `patch_size` canvas pixels, with
\(\sigma\) = `patch_size` / 2 by default (`sigma`), but deviates from Eq. (1)
in two ways:

- Each entity contributes only to the 3 × 3 cells around the cell that
  contains it, and with `normalize_per_cell=True` (the default, used by
  `build_pseudoimages`) its weights over these cells are first normalised to
  sum to one.
- Cells without any contribution are set to 0 and excluded from the support
  mask instead of using \(\varepsilon\).

`axis.pp.build_pseudoimages` then applies the preset (`preset`, see
`axis.pp.PSEUDOIMAGE_PRESETS`) through `axis.pp.rasterize_pseudoimage`:

- `"visium"`: cells of canvas // 32 pixels (a 32 × 32 grid on 512- and
  1024-px canvases), a size-2 median filter and nearest-neighbour resizing to
  canvas / 8.
- `"xenium"`: 8-px cells, with values below 0.01 set to 0.

The pseudo-image must be at 1/8 of the canvas, the resolution of the level-4
encoder features; `rasterize_pseudoimage` raises a `ValueError` otherwise. The
stored mask marks the cells in which any component is positive after these
steps.

## Transition-field network (`axis.registration`)

| Component | Equation | Code |
|---|---|---|
| five-level convolutional encoder (\(c_e\) … \(16c_e\)) | — | `blocks.Conv_encoder` |
| transformer fusion at level 4 | (3)–(5) | `blocks.TransformerFusion` |
| convolution fusion at level 4 | (6) | `blocks.FusionBlock` |
| Swin decoder, affine initialisation | (8)–(11) | `blocks.Trans_decoder`, `blocks.AffineHead_block` |
| coarse-to-fine residual refinement | (12)–(17) | `blocks.Trans_decoder`, `blocks.DeformHead_block` |
| shifted-window attention blocks | (18)–(19) | `blocks.SwinTrans_Block` |
| image / pseudo-image warping | — | `blocks.SpatialTransformer_block` |
| objective | (20) | `losses.RegistrationLoss` |
| squared local NCC (morphology) | (21) | `losses.NCC` |
| signed masked NCC (molecules, ≥ 95 % overlap) | (22) | `losses.MaskedNCC` |
| smoothness | (23) | `losses.smoothness_loss` |

## Virtual-tissue generation (`axis.generation`)

| Step | Equation | Code |
|---|---|---|
| relative position \(\alpha_q\), scaled fields | (24)–(25) | `relative_position_on_axis`, `build_query_evidence` |
| transport into the query anatomy, evidence plane | (26)–(27) | `axis.registration.find_inverse_warp_coords`, `stack_weighted_evidence` |
| biological-axis weights, \(\sigma_\tau\) = spacing / 3 | (28)–(29) | `compute_axis_kernel_weights` |
| spatial kernel | (30)–(32) | `prepare_local_query_support` |
| molecular-quality weight (\(\gamma_C=1\), \(\gamma_G=0.7\), clip [0.5, 2]) | — | `compute_local_relative_quality_weights` |
| weighted generation / categorical support | (33)–(35) | `generate_virtual_tissue_fixed_grid` |
| tissue-support filter (`max_dist`), smoothing, largest connected region | — | `compute_dense_tissue_mask_on_any_grid` |

The tissue-support filter excludes output locations farther than `max_dist`
canvas pixels from every transported evidence point (Methods `max_distance`;
default 8 in `axis.tl.generate` and `axis.tl.match`, 5 in `axis.tl.integrate`).
Beyond the Methods, the mask is then smoothed on the grid, which can add or
remove locations: a location is kept if at least 2 locations within 1.25 grid
spacings (median nearest-neighbour distance), itself included, pass the
distance threshold. Only the largest connected supported region is kept, so
smaller, separate tissue pieces are not generated: `axis.tl.generate` and
`axis.tl.match(..., reconstruct=True)` keep their locations with
`obs["valid"] = False` and zero molecular values, and `axis.tl.integrate`
leaves them out of its output.

## Spatially matched analysis (`axis.matching`)

| Step | Equation | Code |
|---|---|---|
| reconstruction on a reference support (\(\beta_r=1\)) | — | `reconstruct_on_reference`, `axis.tl.match` |
| location-wise contrasts (library-size normalisation, \(\log(1+x)\)) | — | `contrast.compute_spatial_delta`, `axis.tl.contrast` |
| Sinkhorn coupling with mixed cost | (36)–(37) | `population.integrate_population` |
| adaptive low rank and anisotropic diffusion | (38)–(39) | `population._svt_adaptive_local`, `population._latent_aniso_local` |
| XGBoost cross-modality prediction | — | `cross_modality.fit_cross_modality` |

By default, contrasts normalise every location of both states to `target_sum`
counts (library size over all features) and \(\log(1+x)\)-transform them
before subtraction. `target_sum` defaults to \(10^4\); the Methods instead use
the median per-location library size across all reconstructed states (the
mouse-brain figure scripts use 253). The package does not compute this median,
so pass it as `axis.tl.contrast(a, b, target_sum=...)`.

## Evaluation (`axis.metrics`)

Dice, spatial correlation and centre shift of matched domains and
molecular-neighbourhood correspondence (`alignment`; the spatial correlation
\(S_c\) of Methods, the normalised inner product of the domain density maps,
is the column `density_cosine` of `evaluate_domain_metrics`, while
`spatial_corr` is their Pearson correlation, the value the Fig. 2 scripts
report); NCC, gradient-orientation
profiles, cluster composition, nearest-neighbour Wasserstein distance,
neighbourhood connectivity, Kruskal–Wallis effect size, adjacent-layer AUROC
and contrast agreement (`generation`).
