#!/bin/sh
### job name
#SBATCH --job-name=Up_R_Shuf32


### Scelta della coda "testing" per tesisti

### Numero di host (nodes) occupati dal Job e numero di processi da allocare per nodo (ppn)
# SBATCH --ntasks=1

### Memoria che ci si aspetta che il programma utilizzi

### Limite di tempo che vogliamo assegnare al nostro job (qui un'ora: HH:MM:SS)
# SBATCH --time=500:00:00

### Richiesta della percentuale di GPU
# SBATCH --gres=gpu:1

### Alcuni comandi di shell che stampano lo stato
### del sistema prima dell'esecuzione
echo -n 'Date: '
date
echo -n 'Directory: '
pwd
echo -n 'Questo job viene eseguito sui seguenti nodi: '
echo ${SLURM_NODELIST}
nvidia-smi
which nvcc
### Esecuzione del nostro programma


### Without singularity...
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate new_env
export WANDB_API_KEY=1c9bad0d162960a36cc8fecc456d0dc024cb834c
python3 $HOME/storage/MasterThesis/main.py \
 --epochs 100 \
 --logger wandb \
 --stats_path upstream_SHUFF_resnet50_FirstLayer_cifar10_SGD_WD_DecayLR_REL\
 --dataset cifar10 \
 --batch_size 256 \
 --optimizer sgd \
 --weight_decay 0.0001 \
 --lr 0.1 \
 --backbone cifar10_model \
 --task upstream \
 --shuffle_patches \
 --use_relations \
 --exp_id up_SHUFF
## added more layer that usual !!!