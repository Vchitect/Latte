#!/usr/bin/env bash
#SBATCH --job-name=Latte-ffs #To give your job a name, replace "Latte-ffs" with an appropriate name
#SBATCH --partition group-name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=500:00:00
#SBATCH --output=slurm_log/%j.out 
#SBATCH --error=slurm_log/%j.err

source ~/.bashrc

conda activate latte

srun python train.py --config ./configs/ffs/ffs_train.yaml