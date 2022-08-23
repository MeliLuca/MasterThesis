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
export WANDB_API_KEY=ea6aee9ecbca9ca1ea2625da6ebe714375849129
python3 $HOME/storage/MasterThesis/main.py \
 --epochs 150 \
 --logger wandb \
 --stats_path vit_pup_ade20k_224_16_ADAMW \
 --dataset ade20k \
 --batch_size 32 \
 --optimizer adamw \
 --backbone vit \
 --use_segmentation \
 --model_version pup \
 --patch_size 16 \
 --metric_monitor val_miou 
