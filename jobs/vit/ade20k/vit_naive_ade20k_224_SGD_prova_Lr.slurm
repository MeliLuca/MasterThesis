#!/bin/bash
#SBATCH --job-name="naive"
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
 --stats_path vit_naive_ade20k_224_SGD_prova_lr\
 --dataset ade20k \
 --batch_size 16 \
 --optimizer sgd \
 --weight_decay 0.0 \
 --lr 0.001 \
 --decay_lr_sched=polynomial \
 --backbone vit \
 --use_segmentation \
 --model_version naive \
 --exp_id test_lr