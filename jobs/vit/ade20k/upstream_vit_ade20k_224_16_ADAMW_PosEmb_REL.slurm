#!/bin/bash
#SBATCH --job-name="up_EPosRel"
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
 --stats_path upstream_vit_ade20k_224_16_ADAMW_PosEmb_REL\
 --dataset ade20k \
 --batch_size 64 \
 --patch_size 16 \
 --optimizer adamw \
 --backbone vit \
 --task upstream \
 --use_relations \
 --use_positional_embeddings \
 --exp_id up