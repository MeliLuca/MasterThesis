#!/bin/sh
#SBATCH --job-name=Naive
#SBATCH --ntask=1
#SBATCH --time=500:00:00
#SBATCH --gres=gpu:1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate new_env
cd $HOME/storage/MasterThesis
export WANDB_API_KEY=1c9bad0d162960a36cc8fecc456d0dc024cb834c
python3 $HOME/storage/MasterThesis/main.py \
 --epochs 100 \
 --logger wandb \
 --stats_path try_vit_naive-dec_ade20k_224_ADAMW\
 --dataset ade20k \
 --batch_size 128 \
 --optimizer adamw \
 --backbone vit \
 --use_segmentation \
 --model_version naive 