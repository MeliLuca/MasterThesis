#!/bin/bash
#SBATCH --job-name="fcn"
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=500:00:00
echo "home directory is $HOME"
pwd
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate new_env
echo "conda version from echo $(conda --version)"
export WANDB_API_KEY=1c9bad0d162960a36cc8fecc456d0dc024cb834c
python3 $HOME/storage/MasterThesis/main.py \
 --epochs 150 \
 --logger wandb \
 --stats_path fcn_ade20k_224_ADAMW \
 --dataset ade20k \
 --batch_size 64 \
 --optimizer adamw \
 --backbone fcn \
 --use_segmentation \