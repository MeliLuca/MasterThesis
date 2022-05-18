#!/bin/sh
### job name
SBATCH --job-name=Res50small-baseline

#SBATCH --mail-user=alex.dametto@studenti.unipd.it

### Redirezione dello standard output e standard error del programma
SBATCH --error=outputs/resnet50baseline_226imgSize.err
SBATCH --output=outputs/resnet50baseline_226imgSize.out


### Scelta della coda "testing" per tesisti
SBATCH --partition=testing

### Numero di host (nodes) occupati dal Job e numero di processi da allocare per nodo (ppn)
SBATCH --ntasks=1

### Memoria che ci si aspetta che il programma utilizzi
SBATCH --mem=50G

### Limite di tempo che vogliamo assegnare al nostro job (qui un'ora: HH:MM:SS)
SBATCH --time=500:00:00

### Richiesta della percentuale di GPU
SBATCH --gres=gpu:1

SBATCH --cpus-per-task=4

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
python3 $HOME/storage/MasterThesis/main.py --epochs=300 --stats_path=baseline_226_imgSize --dataset=cifar10_226

## 258741