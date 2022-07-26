#!/bin/sh
### job name
#SBATCH --job-name=naiveDec


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
 --stats_path try_vit_naive-dec_ade20k_224_ADAMW.slurm\
 --dataset ade20k \
 --batch_size 128 \
 --optimizer adamw \
 --backbone vit \
 --use_segmentation \
 --model_version naive \
## added more layer that usual !!!