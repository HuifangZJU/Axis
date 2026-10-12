#!/bin/bash
#SBATCH --job-name=deep_learning_job     # Job name
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8       #cpu number
#SBATCH --mem=64G

source /hpc/group/jilab/huifang/miniconda3/etc/profile.d/conda.sh
# Activate the "cellpose" conda environment
conda activate cellpose

# Change to your working directory where main.py is located
cd /hpc/group/jilab/huifang/code/Nice-trans
# Run your deep learning model
#python hf_combined_train.py --patient 10 --ablation_level 1
#python hf_eval_combined_ablation.py
python hf_combined_train_xenium.py
#python hf_combined_train.py
conda deactivate



