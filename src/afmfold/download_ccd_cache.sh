#!/bin/bash
#SBATCH -J build-anm
#SBATCH -A gts-yke8
#SBATCH -N1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

source ~/scratch/venv_afmfold/bin/activate
cd ~/scratch/afmfold/src/afmfold
srun python scripts/gen_ccd_cache.py -n 8
