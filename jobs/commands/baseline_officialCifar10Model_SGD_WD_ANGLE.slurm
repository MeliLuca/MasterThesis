#!/bin/sh 
### job name
#SBATCH --job-name=resnetWithAttention
### Redirezione dello standard output e standard error del programma
### Scelta della coda "testing" per tesisti
### Numero di host (nodes) occupati dal Job e numero di processi da allocare per nodo (ppn)
#SBATCH --ntasks=1
### Memoria che ci si aspetta che il programma utilizzi
#SBATCH --mem=50G
### Limite di tempo che vogliamo assegnare al nostro job (qui un'ora: HH:MM:SS)
#SBATCH --time=500:00:00
### Richiesta della percentuale di GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

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
python3 $HOME/storage/MasterThesis/main.py --stats_path=resnetOfficial_SGD_WD_ANGLE --dataset=cifar10 --batch_size=256 --optimizer=sgd --weight_decay=0.0001 --lr=0.1 --backbone=cifar10_model --use_angles --logger=csv
