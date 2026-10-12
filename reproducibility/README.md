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
> and (iii) the fixes listed in the changelog (`docs/changelog.md`).

## 1. Setup

```bash
pip install -e ".[paper]"          # axis + all figure dependencies
export AXIS_DATA_DIR=/path/to/axis_data           # processed datasets
export AXIS_EXPERIMENT_DIR=/path/to/axis_runs     # checkpoints and logs (default: $AXIS_DATA_DIR/experiments)
```

Every Python script resolves its inputs below these two roots, except the
default arguments of `figure7/analysis/xenium_to_protein_xgboost_h5ad.py`; the
R scripts in `figure7/analysis/R/` use paths relative to the working directory.
The expected layout of each dataset (relative to `AXIS_DATA_DIR`) and its
analysis parameters are recorded in [`configs/`](https://github.com/HuifangZJU/Axis/tree/main/configs):

| Dataset | Config | Root | Used in |
|---|---|---|---|
| Human SCC, 10x Visium (GSE144240) | `scc.yaml` | `SCC/huifang` | Figs 2h–m, 3 |
| Human DLPFC, 10x Visium (HumanPilot) | `dlpfc.yaml` | `DLPFC/huifang` | Figs 2a–c, 4 |
| TgCRND8 / WT mouse brain, 10x Xenium | `xenium_mouse_brain.yaml` | `xenium/mouse_brain`, `mouse_brain` | Figs 1, 2d–e, 5, 6 |
| Human breast cancer, 10x Xenium | `xenium_breast.yaml` | `breast_cancer` | Fig. 2f–g |
| LUAD PhenoCycler + Xenium | `phenocycler_luad.yaml` | `phenocycler/huifang` | Fig. 7 |

Most figure outputs go below `paper_analysis/` in `AXIS_DATA_DIR`. The
exceptions are:

- `figure5/analysis/calculate_global_library_size_6_virtual_slices.py`, which
  writes `paper_figure/Figure6_disease/library_size_diagnosis/`;
- `figure2/2m_istar_metric.py`, which writes its metric tables to
  `SCC/istar_alignment_eval/`;
- `pipelines/training/eval_*xenium*.py`, which write `xenium/result/`.

Besides the dataset roots above, some scripts also read other directories below
`AXIS_DATA_DIR`, e.g. `SCC/paste`, `SCC/istar`, `SCC/competing_methods`,
`DLPFC/paste_processed`, `humanpilot/`, `xenium/raw_data`,
`phenocycler/haotian/` and `phenocycler/cross_modality_prediction/`.

Scripts that import sibling helpers (`utils_io`, `legacy_models`, …) must be
run from their own folder, e.g. `cd reproducibility/figure2 && python 2bc_dlpfc_metric.py`.

## 2. Pipeline

The figures consume the outputs of four pipeline stages
([`pipelines/`](https://github.com/HuifangZJU/Axis/tree/main/reproducibility/pipelines)), run per dataset in this order:

| Stage | Scripts | Output (below the dataset root) |
|---|---|---|
| 1. Canvases & pseudo-images | `pipelines/preprocessing/{scc,scc_ablation,scc_istar,dlpfc,xenium_mouse_brain}.py` | `h5ad/`, `network_input/` |
| 2. Transition-field learning | `pipelines/training/train_*.py` (SLURM templates in `training/slurm/`) | `$AXIS_EXPERIMENT_DIR/{image,combined,phenocycler}/<run>/saved_models/` |
| 3. Pairwise fields | `pipelines/inference/run_{scc,dlpfc,xenium_mouse,phenocycler}.py` | `pairwise_output/group_<g>/{F}_{M}_flow.npy`, `obsm["spatial_axis"]` |
| 4. Generation & matching | `pipelines/generation/*.py` | `virtual_tissue/` |

**The write steps of stages 1 and 3 are switched off in the distributed
scripts**, which are kept as they were used for the paper.

- **Stage 1.** The preprocessing scripts display the pseudo-images but do not
  write the network inputs. To write them:
    - in `scc.py` and `dlpfc.py`, uncomment the `prefix` line(s) and the
      `save_network_input(out_dir, prefix, ...)` call at the end of the
      per-sample loop, and add `from axis.io import save_network_input`; this
      writes `network_input/`;
    - in `scc_ablation.py` and `xenium_mouse_brain.py`, uncomment the
      `plt.imsave` / `np.save` / `np.savez` lines and replace the literal
      `"$AXIS_DATA_DIR/..."` strings with `data_path("...")`. In
      `scc_ablation.py`, also import `data_path` from `axis.settings` and leave
      the `_field.npz` line commented, because `dx` / `dy` are not defined.
      They write to `SCC/huifang/ablation/image/<level>/` (the files
      `scc_ablation_pair` reads) and to `xenium/` (with the `_<size>` suffix
      that `Image_svg_pair` expects);
    - `scc_istar.py` only displays the iSTAR pseudo-images. Its commented save
      call cannot be re-enabled as is, because `image`, `labels`, `gene_mask`
      and `out_dir` are not defined.
- **Stage 3.** The inference scripts do not save new fields. `run_scc.py`
  and `run_dlpfc.py` only copy the reference-frame coordinates from
  `network_input/` and `pairwise_output/` into `obsm["spatial_axis"]` of the
  `h5ad/<group>_<sample>.h5ad` files (a group without such files is skipped
  with a log line), and `run_xenium_mouse.py` only maps the existing
  `pairwise_output/` results onto the reference sample. They stop with an
  error (e.g. `FileNotFoundError`) when a `network_input/` or
  `pairwise_output/` file they need is missing. `run_phenocycler.py` computes
  and plots the fields without saving them. To compute and save the fields,
  uncomment:
    - `generate_pairwise_result()` in `run_scc.py` and `run_dlpfc.py`, plus the
      `save_results_per_pair` loop in `run_dlpfc.py`;
    - `run_network()` in `run_xenium_mouse.py`;
    - the `save_results_per_pair` loop in `run_phenocycler.py`.

The training scripts are the versions used for the paper. Each is set up for
one dataset; edit its dataset block and `--model_dir` for the others
(`train_image.py` derives the folder name from `--patient` and
`--ablation_level`).

- `train_image.py` trains the image-only network (`NICE_Trans_img`) on the SCC
  morphology-ablation pairs (`scc_ablation_pair`: all six ordered pairs of the
  sections of `--patient` at `--ablation_level`).
- `train_image_xenium.py` trains it on the Xenium pairs listed in
  `xenium_imglist/group1_{train,test}.txt`.
- `train_image_phenocycler.py` trains it on the morphology pairs listed in
  `$AXIS_DATA_DIR/phenocycler/<data_name>.txt` (lines
  `data_name fixed_modality moving_modality`) and saves under
  `$AXIS_EXPERIMENT_DIR/phenocycler/`.
- `train_multimodal.py` trains the multimodal network on the DLPFC pairs listed
  in `dlpfc_imglist/dlpfc_group_2_{train,test}.txt`. By default
  (`--load_model`) it warm-starts from the image-only checkpoint
  `$AXIS_EXPERIMENT_DIR/image/img_3data_model/saved_models/model_950.pth` and
  saves `model_<epoch>.pth` every 50 epochs.
- `train_multimodal_xenium.py` trains the multimodal network, by default from
  scratch, on the LUAD_3_A protein/Xenium pairs (`phenocycler_omics_pair`) and
  saves every 100 epochs.

Both multimodal scripts default to `--epochs 1001` and write to
`$AXIS_EXPERIMENT_DIR/combined/<model_dir>/saved_models/`.

`Image_svg_pair` (used by `train_image_xenium.py` and `train_multimodal.py`)
reads its list file below `AXIS_DATA_DIR`. Each line holds a `fixed moving`
pair of sample prefixes relative to `AXIS_DATA_DIR`, and
`<prefix>_image_<size>.png`, `_pca_out_<size>.npy`, `_pca_mask_<size>.npy` and
`_validation_<size>.npz` must exist (`<size>` is the `base_size` of `Image_svg_pair`: 512 in `train_multimodal.py`, `--base_size` (default 1024) in `train_image_xenium.py`;
`save_network_input` writes the files without the `_<size>` suffix). The list
files are not distributed.

The stage-3 inference scripts load `trained_models/.../model_1000.pth` below
the dataset root (see each `run_*.py`), so copy the stage-2 checkpoint there.

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
| **Fig. 1** | `figure1/simu_data_generator.py`, `simu_elements.py`, `simu_method_view.py` | simulated tissue states and schematic elements (1a, d) |
| 1b–c | `figure1/figure1b.py` | Visium-like coarse measurements simulated from Xenium mouse brain (1b) and the mouse-brain states, shared-frame layout and Arc AD–WT location-wise comparison (1c) |
| 1d | `figure1/spatial_layout_symbols.py` | pixel / spot / single-cell layout symbols |
| 1e | `figure1/population_view.py` | population-level integration schematic (DLPFC) |
| **Fig. 2a** | `figure2/2a_dlpfc_spatial.py` | DLPFC sections stacked and overlaid before / after AXIS |
| 2b–c | `figure2/2bc_dlpfc_metric.py` | representative section pair for every method (2b); Dice / spatial correlation / centre shift / molecular-neighbourhood box plots (2c) |
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
| 7b–c | `figure7/analysis/pretsa_spatial.py`, `figure7/7bc_protein_evaluation.py` | cognate gene–protein concordance on PreTSA-smoothed virtual protein and Xenium genes |
| 7d | `figure7/analysis/R/normalize.R` → `plot_xenium_predict.R` → `run_limma.R`, then `analysis/R/plot_dotplot.R` / `figure7/7d_de_dotplot.py` | limma gene–protein association slopes |
| 7e | `figure7/analysis/R/remove_outlier.R`, `figure7/7e_spatial_covisualization.py` | NKX2.1 / NAPSA / MUC5B / IDO1 co-maps |
| 7f–g | `figure7/7fg_cross_modality_evaluation.py`, `analysis/xenium_to_protein_xgboost_h5ad.py`, `analysis/pretsa_spatial.py` | XGBoost protein prediction (PreTSA-smoothed Xenium predictors) |

Notes for Fig. 7:

- **7b–c.** `7bc_protein_evaluation.py` reads the `..._presta_knot30_...` files
  that `pretsa_spatial.py` writes to
  `phenocycler/huifang/virtual_tissue/virtual_pair/presta/`; set `knot = 30` in
  `pretsa_spatial.py`, whose default is 50.
- **R scripts.** The scripts in `figure7/analysis/R/` read and write
  `multi_human_LUAD/...` relative to the working directory. They need R with
  Matrix, limma, ggplot2, dplyr, tidyr and RColorBrewer.
- **plot_xenium_predict.R** writes the `xenium_protein_pretsa_50.rds` used by
  `run_limma.R` and `remove_outlier.R`. It sources `spatialPred.R` from PreTSA
  (set `PRETSA_DIR`) and the plotting helper `spatialplot.R`, which is not
  distributed (set `AXIS_R_HELPERS`).
- **7d / 7e plotters.** `7d_de_dotplot.py` reads `xenium_DE.rds`, and
  `7e_spatial_covisualization.py --mode eval` (or `both`) reads
  `xenium_DE.rds`, `xenium_protein_predict.rds`, `LUAD_2_A_xenium_axis.csv`
  and `adata_xenium_cell_level.h5ad`, from
  `$AXIS_DATA_DIR/phenocycler/haotian/` (`--input_folder` / `--data_root`);
  its default `--mode plot` only redraws the results saved by `eval`. Copy
  `multi_human_LUAD/xenium_DE.rds`,
  `multi_human_LUAD/expr/xenium_protein_predict.rds` and
  `multi_human_LUAD/spatial_axis/LUAD_2_A_xenium_axis.csv` there.
- **7f–g.** `7fg_cross_modality_evaluation.py` reads
  `$AXIS_DATA_DIR/phenocycler/cross_modality_prediction/xgb_out_search_new/aligned_pred_test_only.h5ad`.
  The defaults of `xenium_to_protein_xgboost_h5ad.py` are paths relative to
  the working directory, so run it with:
    - `--gene_h5ad`: the PreTSA-smoothed Xenium h5ad written by
      `pretsa_spatial.py`;
    - `--protein_h5ad`: the unsmoothed
      `$AXIS_DATA_DIR/phenocycler/huifang/virtual_tissue/virtual_pair/virtual_protein_1.000.h5ad`;
    - `--outdir $AXIS_DATA_DIR/phenocycler/cross_modality_prediction/xgb_out_search_new/`.

## 4. Checking the scripts

```bash
python tools/check_script_imports.py reproducibility examples --strict
```

statically resolves every import (including names) of every script without
running it; it is part of CI.
