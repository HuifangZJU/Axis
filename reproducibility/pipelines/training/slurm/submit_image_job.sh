#!/bin/bash
# SLURM template used for the manuscript experiments (single GPU).
# Adjust the partition / resources for your cluster and set:
#   AXIS_DATA_DIR        processed datasets (see configs/*.yaml)
#   AXIS_EXPERIMENT_DIR  checkpoints and training logs
#   AXIS_CONDA_ENV       conda environment with axis installed (default: axis)
#SBATCH --job-name=axis_image
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate "${AXIS_CONDA_ENV:-axis}"

# Submit from reproducibility/pipelines/training:  sbatch slurm/<this file>
cd "${SLURM_SUBMIT_DIR:-.}"
python train_image.py
# python train_image_xenium.py
# python train_image_phenocycler.py
