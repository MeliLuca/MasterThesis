#!/bin/bash
#SBATCH --job-name="sgmtPoly512"
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=500:00:00
echo "directory"
pwd
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate new_env
echo -n 'Questo job viene eseguito sui seguenti nodi: '
echo ${SLURM_NODELIST}
nvidia-smi
which nvcc
echo "conda version from echo $(conda --version)"
export WANDB_API_KEY=1c9bad0d162960a36cc8fecc456d0dc024cb834c
### Esecuzione del nostro programma
### Without singularity...
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate new_env
export WANDB_API_KEY=1c9bad0d162960a36cc8fecc456d0dc024cb834c
python3 $HOME/storage/MasterThesis/main.py \
 --epochs 100 \
 --logger wandb \
 --stats_path resnet18_cifar10_224_SGD_WD_DecayLR_posEmb_REL \
 --dataset cifar10_224 \
 --batch_size 128 \
 --optimizer sgd \
 --weight_decay 0.0001 \
 --lr 0.1 \
 --backbone resnet18 \
 --use_supervision \
 --use_relations \
 --use_positional_embeddings \
 --task upstream \
 --exp_id down_ONLY_shuff
## added more layer that usual !!!