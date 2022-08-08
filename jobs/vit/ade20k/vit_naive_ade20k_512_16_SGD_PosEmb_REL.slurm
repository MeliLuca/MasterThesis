#!/bin/bash
#SBATCH --job-name="nvRel512"
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
 --epochs 64 \
 --logger wandb \
 --stats_path vit_naive_ade20k_512_16_SGD_PosEmb_REL\
 --dataset ade20k_512 \
 --batch_size 8 \
 --optimizer sgd \
 --decay_lr_sched polynomial \
 --weight_decay 0 \
 --lr 0.001 \
 --backbone vit \
 --use_segmentation \
 --use_relations \
 --model_version naive \
 --patch_size 16 \
 --use_positional_embeddings \
 --metric_monitor val_miou \
 --exp_id sgd_polyDecay