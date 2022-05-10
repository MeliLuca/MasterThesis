#!/bin/sh
### job name
#SBATCH --job-name=MRes18-0023

#SBATCH --mail-user=alex.dametto@studenti.unipd.it

### Redirezione dello standard output e standard error del programma
#SBATCH --error=outputs/MRes18-0023.err
#SBATCH --output=outputs/MRes18-0023.out


### Scelta della coda "testing" per tesisti
#SBATCH --partition=testing

### Numero di host (nodes) occupati dal Job e numero di processi da allocare per nodo (ppn)
#SBATCH --ntasks=1

### Memoria che ci si aspetta che il programma utilizzi
#SBATCH --mem=50G

### Limite di tempo che vogliamo assegnare al nostro job (qui un'ora: HH:MM:SS)
#SBATCH --time=500:00:00

### Richiesta della macchina
#SBATCH --constraint=vgpu2-0

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
conda activate torch
bash ${HOME}/storage/TinyActionChallenge/commands/MRes18-0023.sh