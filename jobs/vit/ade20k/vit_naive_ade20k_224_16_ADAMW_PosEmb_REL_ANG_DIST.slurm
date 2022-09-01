#!/bin/bash
#SBATCH --job-name="PeRAD"
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=500:00:00
echo "directory"
pwd
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate new_env
echo "conda version from echo $(conda --version)"
##export WANDB_API_KEY=0f74986f3bceaa9427d8e972a76dbe0c049ab86c
python3 $HOME/storage/MasterThesis/main.py \
 --epochs 150 \
 --logger wandb \
 --stats_path vit_naive_ade20k_224_16_ADAMW_PosEmb_REL_ANG_DIST\
 --dataset ade20k \
 --batch_size 64 \
 --optimizer adamw \
 --backbone vit \
 --use_segmentation \
 --model_version naive \
 --patch_size 16 \
 --use_positional_embeddings \
 --use_relations \
 --use_angles \
 --use_distances \
 --metric_monitor val_miou \