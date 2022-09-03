#!/bin/bash
#SBATCH --job-name="vit_rad"
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=500:00:00
echo "directory"
pwd
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate new_env
echo "conda version from echo $(conda --version)"
export WANDB_API_KEY=1c9bad0d162960a36cc8fecc456d0dc024cb834c
python3 $HOME/storage/MasterThesis/main.py \
 --epochs 60 \
 --logger wandb \
 --stats_path vit_ade20k_224_16_ADAMW_REL_ANG_DIST\
 --dataset ade20k \
 --batch_size 64 \
 --patch_size 16 \
 --optimizer adamw \
 --backbone vit \
 --use_relations \
 --use_angles \
 --use_distances \
 --exp_id up