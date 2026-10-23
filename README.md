<h1 align="center">AXIS</h1>

<p align="center"><b>Virtual tissue generation and spatially matched analysis from spatial omics</b></p>

<p align="center">
  <a href="https://github.com/HuifangZJU/Axis/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/HuifangZJU/Axis/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://huifangzju.github.io/Axis/"><img alt="Docs" src="https://github.com/HuifangZJU/Axis/actions/workflows/docs.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%E2%80%933.13-blue">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>

AXIS learns dense, bidirectional **transition fields** between spatial-omics
tissue sections by combining molecular measurements (as *pseudo-images*) with
H&E or DAPI morphology. The same fields are used to

* **generate unmeasured tissue states** – intermediate anatomical depths, time
  points or disease states – on any output support (pixel grid, Visium-like
  spots or single cells), and
* **place independently measured tissues in a shared anatomical frame** for
  location-matched comparison, population-level integration and
  cross-modality analysis.

<p align="center"><img src="https://raw.githubusercontent.com/HuifangZJU/Axis/main/docs/assets/axis_overview.jpg" width="900" alt="AXIS overview (manuscript Fig. 1)"></p>

<p align="center"><sub><b>Fig. 1</b> | (a–c) Virtual tissue generation and spatially matched analysis along anatomical, temporal, disease and modality axes. (d) AXIS encodes molecular measurements as pseudo-images, learns pixel-level transition fields with a multimodal coarse-to-fine network, and infers tissue states with biological, spatial and molecular weights. (e) Applications.</sub></p>

**Documentation:** <https://huifangzju.github.io/Axis/> ·
**Paper reproduction:** [`reproducibility/`](reproducibility/README.md)

---

## Installation

AXIS requires Python ≥ 3.10 and PyTorch ≥ 2.0 (a CUDA GPU is recommended for training).

```bash
pip install "axis-spatial @ git+https://github.com/HuifangZJU/Axis.git"
```

or, for development and to reproduce the paper,

```bash
git clone https://github.com/HuifangZJU/Axis.git && cd Axis
conda create -n axis python=3.11 -y && conda activate axis
pip install -e ".[paper,dev]"          # extras: metrics, crossmodal, train, paper, docs, dev
```

Install the PyTorch build matching your CUDA version first if needed
(<https://pytorch.org/get-started/locally/>). The distribution is called
`axis-spatial`; the import name is `axis`.

## Quickstart: SCC, patient 10

Three serial 10x Visium sections of a cutaneous squamous cell carcinoma
(0, 10 and 20 µm; Ji *et al.*, *Cell* 2020). We learn transition fields, place
all sections in one anatomical frame, and **generate the 10-µm section from the
0- and 20-µm sections**, holding the measured 10-µm section out for comparison.

```bash
pip install gdown
gdown 1JEzGLEK23lgcn60QVaQANiKG3BpiX1BK -O scc_patient10.zip && unzip scc_patient10.zip -d scc_patient10
```

```python
import axis

# 1. load the sections and build canvases + molecular pseudo-images
adatas = axis.datasets.load_scc_sample("scc_patient10", patient=10)       # 10_0/1/2.h5ad
for a in adatas:
    axis.pp.prepare_canvas(a, spatial_key="spatial_image_coor", canvas_size=512)
axis.pp.build_pseudoimages(adatas, n_comps=10, preset="visium")

# 2. learn transition fields between all section pairs
fields = axis.tl.fit_transition(adatas, morph_win=9, mol_win=5, lambda_mol=0.5,
                                max_steps=3000)          # 500 epochs over the 6 ordered pairs
fields.save("scc_fields")

# 3. spatially matched analysis: express every section in the frame of section 0
axis.tl.match(adatas, fields, reference=0, key_added="spatial_axis_new")  # keeps the published obsm["spatial_axis"]
axis.pl.matched(adatas, color="original_clusters", key="spatial_axis_new")

# 4. virtual tissue generation at the unmeasured depth of 10 µm
vt = axis.tl.generate(adatas, fields, positions=[0, 10, 20], query=10, anchors=[0, 2],
                      grid="visium", grid_size=40, attr_keys=["original_clusters"],
                      spatial_bandwidth=3, max_dist=10)                    # Fig. 3 settings
axis.pl.spatial(vt, "original_clusters")
axis.pl.spatial(vt, "KRT1")
```

The same steps, with a comparison against the held-out section and the
published AXIS alignment, are packaged as a script:

```bash
python examples/quickstart_scc.py --data-dir scc_patient10 --out quickstart_out             # GPU
python examples/quickstart_scc.py --data-dir scc_patient10 --canvas-size 256 --max-steps 400  # CPU
```

Training dominates the run time: 400 steps at 256 px took 55 min on 4 CPU
cores (training at 512 px needs a GPU or more than 14 GB of RAM). Pass
`checkpoint=` to `axis.tl.fit_transition` (or `--checkpoint` to the script) to
start from a trained model; with `epochs=0` the whole quickstart then runs in
about 1.5 min on the same CPU. Results of the CPU run are shown in the
[gallery](https://huifangzju.github.io/Axis/gallery/).

## Usage by application

| Application (manuscript) | API |
|---|---|
| Transition fields / alignment (Fig. 2) | `axis.tl.fit_transition` → `TransitionFields`; `fields.transport(coords, fixed, moving, alpha)` |
| Virtual tissue at an unmeasured state (Fig. 3) | `axis.tl.generate(adatas, fields, positions, query, anchors=...)` |
| Dense 3D reconstruction along depth (Fig. 4a–g) | loop `axis.tl.generate` over query depths (`grid="visium"` or `"pixel"`) |
| Population-level tissue (Fig. 4h–k) | `axis.tl.integrate(adatas, fields, reference=0)` |
| Virtual time series, matched temporal analysis (Fig. 5) | `axis.tl.generate` along time; `axis.tl.match(..., reconstruct=True)` |
| Location-wise disease contrasts (Fig. 6) | `axis.tl.match(..., reconstruct=True)` then `axis.tl.contrast(ad_disease, ad_control)` |
| Cross-modality co-measurement and prediction (Fig. 7) | `axis.tl.match` across modalities; `axis.tl.predict_modality(X, Y, targets)` |
| Evaluation | `axis.metrics` (Dice, spatial correlation, centre shift, neighbourhood correspondence, NCC, cluster fidelity, laminar AUROC, contrast agreement) |

Lower-level building blocks are available in `axis.registration`
(networks, losses, training loop, warping), `axis.generation` (evidence
planes, weights, generator) and `axis.matching` (reference reconstruction,
contrasts, Sinkhorn integration, XGBoost prediction). See the
[tutorials](https://huifangzju.github.io/Axis/tutorials/) and
[API reference](https://huifangzju.github.io/Axis/api/).

### Data locations

No paths are hard-coded. Datasets are resolved below `AXIS_DATA_DIR`
(default `./data`) and trained models below `AXIS_EXPERIMENT_DIR`
(default `$AXIS_DATA_DIR/experiments`); dataset layouts and the parameters used
in the paper are recorded in [`configs/`](configs):

```python
import axis
axis.settings.data_dir = "/path/to/axis_data"   # or: export AXIS_DATA_DIR=...
cfg = axis.config.load_config("scc")            # configs/scc.yaml
cfg.root, cfg.generation.grid_size              # '/path/to/axis_data/SCC/huifang', 40
```

## Datasets

| Dataset | Technology | Source |
|---|---|---|
| Human cutaneous SCC (4 patients × 3 sections) | 10x Visium | GEO [GSE144240](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144240) |
| Human DLPFC (12 sections) | 10x Visium | [LieberInstitute/HumanPilot](https://github.com/LieberInstitute/HumanPilot) |
| TgCRND8 AD / WT mouse brain (2.5–17.9 months) | 10x Xenium | [10x Genomics datasets](https://www.10xgenomics.com/datasets) |
| Human breast cancer (paired sections) | 10x Xenium | [10x Genomics datasets](https://www.10xgenomics.com/datasets) |
| Lung adenocarcinoma LUAD_2_A | PhenoCycler + 10x Xenium | [kero.hgc.jp](https://kero.hgc.jp/Ad-SpatialAnalysis_2024.html) |
| Quickstart sample (SCC patient 10, processed) | 10x Visium | [Google Drive](https://drive.google.com/file/d/1JEzGLEK23lgcn60QVaQANiKG3BpiX1BK/view) |

## Reproducing the paper

[`reproducibility/`](reproducibility/README.md) contains the preprocessing,
training, inference and generation pipelines and one folder per figure
(`figure1` … `figure7`), with a figure-panel → script table.

## Citation

If you use AXIS, please cite

> Ma H., Zhuang H., Hou W., Zhang A., Ji Z. *Virtual tissue generation and
> spatially matched analysis from spatial omics.* (manuscript under review)

Machine-readable metadata are in [`CITATION.cff`](CITATION.cff).

## License

AXIS is released under the [MIT License](LICENSE). Questions and bug reports:
[GitHub issues](https://github.com/HuifangZJU/Axis/issues) or
zhicheng.ji@duke.edu.
