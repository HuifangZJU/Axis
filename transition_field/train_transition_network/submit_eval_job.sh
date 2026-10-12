#!/bin/bash
#SBATCH --job-name=deep_learning_job     # Job name
#SBATCH --partition=scavenger-gpu     # Job name
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=8       #cpu number
#SBATCH --mem=64G

source /hpc/group/jilab/huifang/miniconda3/etc/profile.d/conda.sh
# Activate the "cellpose" conda environment
conda activate cellpose

# Change to your working directory where main.py is located
cd /hpc/group/jilab/huifang/code/Nice-trans
# Run your deep learning model
#python hf_img_train.py
python hf_eval_combined_xenium_mouse.py
#python hf_eval_combined_xenium_breast.py
#python hf_eval_image_xenium.py
conda deactivate



