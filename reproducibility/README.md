# Reproducing the manuscript

This folder contains the exact analysis code behind every figure of
*Virtual tissue generation and spatially matched analysis from spatial omics*
(Ma, Zhuang, Hou, Zhang & Ji). The reusable method lives in the `axis`
package (`src/axis`); the scripts here are the dataset-specific drivers and the
plotting code used for the paper.

> The scripts are kept as close as possible to the versions that produced the
> published figures. Compared with the original submission snapshot
> (tag `v0-submission-snapshot`) they only differ in (i) machine-specific paths,
> which now resolve through `AXIS_DATA_DIR` / `AXIS_EXPERIMENT_DIR` and
> `configs/*.yaml`, (ii) imports of code that moved into the `axis` package,
> and (iii) a handful of fixes listed in the release notes.

## 1. Setup

```bash
pip install -e ".[paper]"          # axis + all figure dependencies
export AXIS_DATA_DIR=/path/to/axis_data           # processed datasets
export AXIS_EXPERIMENT_DIR=/path/to/axis_runs     # checkpoints and logs (default: $AXIS_DATA_DIR/experiments)
```

Every script resolves its inputs below these two roots. The expected layout of
each dataset (relative to `AXIS_DATA_DIR`) and its analysis parameters are
recorded in [`configs/`](https://github.com/HuifangZJU/Axis/tree/main/configs):

| Dataset | Config | Root | Used in |
|---|---|---|---|
| Human SCC, 10x Visium (GSE144240) | `scc.yaml` | `SCC/huifang` | Figs 2h–m, 3 |
| Human DLPFC, 10x Visium (HumanPilot) | `dlpfc.yaml` | `DLPFC/huifang` | Figs 2a–c, 4 |
| TgCRND8 / WT mouse brain, 10x Xenium | `xenium_mouse_brain.yaml` | `xenium/mouse_brain`, `mouse_brain` | Figs 1, 2d–e, 5, 6 |
| Human breast cancer, 10x Xenium | `xenium_breast.yaml` | `breast_cancer` | Fig. 2f–g |
| LUAD PhenoCycler + Xenium | `phenocycler_luad.yaml` | `phenocycler/huifang` | Fig. 7 |

`paper_analysis/` below `AXIS_DATA_DIR` receives the figure outputs.

Scripts that import sibling helpers (`utils_io`, `legacy_models`, …) must be
run from their own folder, e.g. `cd reproducibility/figure2 && python 2bc_dlpfc_metric.py`.

## 2. Pipeline

The figures consume the outputs of four pipeline stages
([`pipelines/`](https://github.com/HuifangZJU/Axis/tree/main/reproducibility/pipelines)), run per dataset in this order:

| Stage | Scripts | Output (below the dataset root) |
|---|---|---|
| 1. Canvases & pseudo-images | `pipelines/preprocessing/{scc,scc_ablation,scc_istar,dlpfc,xenium_mouse_brain}.py` | `h5ad/`, `network_input/` |
| 2. Transition-field learning | `pipelines/training/train_*.py` (SLURM templates in `training/slurm/`) | `$AXIS_EXPERIMENT_DIR/{image,combined}/<run>/saved_models/` |
| 3. Pairwise fields | `pipelines/inference/run_{scc,dlpfc,xenium_mouse,phenocycler}.py` | `pairwise_output/group_<g>/{F}_{M}_flow.npy`, `obsm["spatial_axis"]` |
| 4. Generation & matching | `pipelines/generation/*.py` | `virtual_tissue/` |

Training follows Methods: an image-only network (`train_image*.py`) is used
to warm-start the multimodal network (`train_multimodal*.py`), which is then
trained for 1,000 epochs on all ordered section pairs of a group.
`training/eval_*.py` reproduce the per-pair evaluations used for Fig. 2;
`training/exploratory/` holds SenNet experiments that are **not** part of the
manuscript.

| Generation script | Analysis | Figure |
|---|---|---|
| `generation/scc_depth.py` | held-out 10-µm SCC section from the 0/20-µm anchors | 3 |
| `generation/dlpfc_depth.py` | dense virtual DLPFC sections between 10 and 310 µm | 4a–g |
| `generation/dlpfc_population.py` | population-level DLPFC integration (Sinkhorn + low rank) | 4h–k |
| `generation/dlpfc_depth_integration.py` | alternative integration with the local evidence-weighted generator instead of Sinkhorn coupling | – |
| `generation/mouse_ad_time.py` | virtual AD mouse brains at 4.1–14.85 months; matched analysis | 5 |
| `generation/mouse_ad_disease.py` | AD and WT matched tissues, 13.4-month AD and held-out 5.7-month WT | 6 |
| `generation/lung_cross_modality.py` | PhenoCycler protein reconstructed on Xenium cells | 7 |

## 3. Figure → script map

| Panel | Script | What it does |
|---|---|---|
| **Fig. 1** | `figure1/simu_data_generator.py`, `simu_elements.py`, `simu_method_view.py` | simulated tissue states and schematic elements (1a, c–d) |
| 1b | `figure1/figure1b.py` | Visium-like coarse measurements simulated from Xenium mouse brain |
| 1d | `figure1/spatial_layout_symbols.py` | pixel / spot / single-cell layout symbols |
| 1e | `figure1/population_view.py` | population-level integration schematic (DLPFC) |
| **Fig. 2a** | `figure2/2a_dlpfc_spatial.py` | DLPFC alignment, case study |
| 2b–c | `figure2/2bc_dlpfc_metric.py` | DLPFC Dice / spatial correlation / centre shift / neighbourhood |
| 2d | `figure2/2d_mouse_spatial.py` | Xenium mouse-brain alignment |
| 2e | `figure2/2e_mouse_metric.py` | mouse-brain metrics |
| 2f | `figure2/2f_breast_spatial.py` | Xenium breast alignment |
| 2g | `figure2/2g_breast_metric.py` | breast metrics |
| 2h | `figure2/2h_scc_spatial.py` | SCC alignment |
| 2i | `figure2/2i_scc_metric.py` | SCC metrics |
| 2j | `figure2/2j_ablation_report.py` | morphology-ablation robustness |
| 2k–l | `figure2/2kl_istar_plots.py` | field transfer to iSTAR super-resolution maps |
| 2m | `figure2/2m_istar_metric.py` | resolution-transferability correlations |
| **Fig. 3a** | `figure3/3a1_generation_pipeline.py`, `3a2_generation_pipeline.py` | SCC generation schematic (evidence plane → virtual section) |
| 3b–c | `figure3/3bc_cluster_evaluation.py` | cluster composition, within-cluster structure, connectivity |
| 3d–g | `figure3/3defg_gene_evaluation.py` | KRT1 / COL1A1 maps, NCC and gradient-orientation fidelity |
| **Fig. 4a** | `figure4/4a_reconstruction_pipeline.py` | DLPFC dense-reconstruction schematic |
| 4b, e | `figure4/4be_resection_views.py` | 3D volumes and virtual re-sections (domains, PCP4) |
| 4c–d | `figure4/4cd_3ddomain_evaluation.py` | laminar continuity and domain-neighbourhood agreement |
| 4f–g | `figure4/4fg_gene_evaluation.py` | shift-tolerant NCC to the anchor sections |
| 4h | `figure4/4h_population_pipeline.py` | population-level integration schematic |
| 4i–k | `figure4/4ijk_layer_separation.py` | Kruskal–Wallis effect size, adjacent-layer AUROC, PCP4/MOBP maps |
| **Fig. 5a** | `figure5/5a_pipeline.py` | AD time-axis generation schematic |
| 5b | `figure5/5b.py` | Allen Mouse Brain Atlas annotations |
| 5c | `figure5/5c_visual_comparison.py` | AXIS vs SpatialZ at 4.1 / 11.8 months |
| 5d–e | `figure5/5de_domain_evaluation.py` | local domain-neighbourhood agreement and matrices |
| 5f | `figure5/5f_pipeline.py` | spatially matched temporal analysis schematic |
| 5g–i | `figure5/5ghi_pixel_delta_analysis.py` | location-wise temporal trajectories (Gfap, Arc, Nrep, Bhlhe22) |
| (5–6 inputs) | `figure5/analysis/mouse_joint_domain_detection.py` | joint spatial domains (`axis_joint_domain`) |
| (5–6 inputs) | `figure5/analysis/replace_mouse_brain_spatial_coords.py`, `calculate_global_library_size_6_virtual_slices.py`, `diagnose_log1p_before_subtraction_arc.py` | coordinate update, global library size, preprocessing diagnostics |
| **Fig. 6a** | `figure6/6a_pipeline.py` | AD–WT matched analysis schematic |
| 6b, g | `figure6/6bg_spatial_map.py` | location-wise AD–WT contrasts (Arc, Gfap, Mobp) |
| 6c–f | `figure6/6cdef_delta_evaluation.py` | virtual vs measured contrasts at 5.7 months; regional genes |
| **Fig. 7a** | `figure7/7a_pipeline.py`, `7a_pipeline_cartoon.py` | cross-modality schematic |
| 7b–c | `figure7/7bc_protein_evaluation.py` | cognate gene–protein concordance |
| 7d | `figure7/7d_de_dotplot.py`, `analysis/R/run_limma.R`, `analysis/R/plot_dotplot.R` | limma gene–protein association slopes |
| 7e | `figure7/7e_spatial_covisualization.py` | NKX2.1 / NAPSA / MUC5B / IDO1 co-maps |
| 7f–g | `figure7/7fg_cross_modality_evaluation.py`, `analysis/xenium_to_protein_xgboost_h5ad.py`, `analysis/pretsa_spatial.py` | XGBoost protein prediction (PreTSA-smoothed Xenium predictors) |

The R scripts in `figure7/analysis/R/` source two external helpers:
`spatialPred.R` from PreTSA (set `PRETSA_DIR`) and a plotting helper
`spatialplot.R` (set `AXIS_R_HELPERS`).

## 4. Checking the scripts

```bash
python tools/check_script_imports.py reproducibility examples --strict
```

statically resolves every import (including names) of every script without
running it; it is part of CI.
