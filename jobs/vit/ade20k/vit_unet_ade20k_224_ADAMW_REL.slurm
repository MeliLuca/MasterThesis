#!/bin/bash
#SBATCH --job-name="unet_rel"
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
 --epochs 150 \
 --logger wandb \
 --stats_path vit_unet_ade20k_224_ADAMW_REL \
 --dataset ade20k \
 --batch_size 32 \
 --optimizer adamw \
 --backbone vit \
 --use_segmentation \
 --model_version unet \
 --use_relations