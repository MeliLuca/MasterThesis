#!/bin/bash
#SBATCH --job-name="pup"
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
 --stats_path vit_pup_ade20k_224_SGD \
 --dataset ade20k \
 --batch_size 64 \
 --optimizer sgd \
 --weight_decay 0.9 \
 --lr 0.001 \
 --backbone vit \
 --use_segmentation \
 --model_version pup \
 --exp_id sgd \