# Method summary

This page maps the Methods of the manuscript to the implementation.

## Pseudo-images (`axis.pp`)

The molecular matrices \(G_k \in \mathbb{R}^{P\times N_k}\) of all samples share
\(P\) features and are concatenated and projected to \(C_p\) principal
components. Each component is rasterised with a Gaussian kernel,

\[
P_k^{(c)}(u)=\frac{\sum_i K_\sigma(\lVert u-x_{k,i}\rVert)\, z^{(c)}_{k,i}}{\sum_i K_\sigma(\lVert u-x_{k,i}\rVert)+\varepsilon},
\qquad K_\sigma(r)=e^{-r^2/2\sigma^2},
\]

with \(\sigma\) half a grid cell — `axis.pp.get_gene_feature_matrix_soft`,
`axis.pp.build_pseudoimages`.

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
| tissue-support filter (`max_distance`) | — | `compute_dense_tissue_mask_on_any_grid` |

## Spatially matched analysis (`axis.matching`)

| Step | Equation | Code |
|---|---|---|
| reconstruction on a reference support (\(\beta_r=1\)) | — | `reconstruct_on_reference`, `axis.tl.match` |
| location-wise contrasts | — | `contrast.compute_spatial_delta`, `axis.tl.contrast` |
| Sinkhorn coupling with mixed cost | (36)–(37) | `population.integrate_population` |
| adaptive low rank and anisotropic diffusion | (38)–(39) | `population._svt_adaptive_local`, `population._latent_aniso_local` |
| XGBoost cross-modality prediction | — | `cross_modality.fit_cross_modality` |

## Evaluation (`axis.metrics`)

Dice, spatial correlation and centre shift of matched domains and
molecular-neighbourhood correspondence (`alignment`); NCC, gradient-orientation
profiles, cluster composition, nearest-neighbour Wasserstein distance,
neighbourhood connectivity, Kruskal–Wallis effect size, adjacent-layer AUROC
and contrast agreement (`generation`).
