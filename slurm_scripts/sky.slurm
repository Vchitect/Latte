#!/usr/bin/env bash
#SBATCH --job-name=LAVITA-ffs # nom du job
#SBATCH --partition group-name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8 # nombre de GPU �|  réserver (un unique GPU ici)
#SBATCH --cpus-per-task=16
#SBATCH --time=500:00:00 # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=slurm_log/%j.out # nom du fichier de sortie
#SBATCH --error=slurm_log/%j.err # nom du fichier d'erreur (ici commun avec la sortie)

source ~/.bashrc

conda activate lavita

srun python train.py --config ./configs/sky/sky_train.yaml