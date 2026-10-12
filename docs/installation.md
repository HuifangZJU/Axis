# Installation

AXIS needs Python ≥ 3.10 and PyTorch ≥ 2.0. Training benefits greatly from a
CUDA GPU; inference, generation and matching run on CPU.

=== "pip"

    ```bash
    pip install "axis-spatial @ git+https://github.com/HuifangZJU/Axis.git"
    ```

=== "development / paper"

    ```bash
    git clone https://github.com/HuifangZJU/Axis.git && cd Axis
    conda create -n axis python=3.11 -y && conda activate axis
    pip install -e ".[paper,dev]"
    ```

Install the PyTorch wheel that matches your CUDA version first if pip would
otherwise pick an unsuitable build (<https://pytorch.org/get-started/locally/>).

## Optional dependencies

| Extra | Contents | Needed for |
|---|---|---|
| `metrics` | alphashape, shapely, scikit-image | alignment metrics (Dice on alpha-shape masks), training monitors |
| `crossmodal` | xgboost | `axis.tl.predict_modality` |
| `train` | tensorboardX, imgaug | the original training scripts in `reproducibility/pipelines/training` |
| `paper` | all of the above + seaborn, h5py, pyreadr, statsmodels, paste-bio, POT, igraph, leidenalg | every figure script |
| `docs` | mkdocs-material, mkdocstrings | building this site |
| `dev` | pytest, ruff, build, twine | tests and packaging |

## Configuring data locations

```bash
export AXIS_DATA_DIR=/path/to/axis_data          # processed datasets (default ./data)
export AXIS_EXPERIMENT_DIR=/path/to/axis_runs    # checkpoints, logs (default $AXIS_DATA_DIR/experiments)
export AXIS_DEVICE=cuda                          # optional; "auto" picks CUDA when available
```

The same settings can be changed at run time through `axis.settings`.

## Checking the installation

```bash
python -c "import axis; print(axis.__version__, axis.settings)"
pytest -q            # from a source checkout
```
