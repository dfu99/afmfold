#!/bin/bash
#SBATCH -J afmfold_gen
#SBATCH -A gts-yke8
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

source ~/scratch/venv_afmfold/bin/activate
cd ~/scratch/afmfold/
srun python scripts/generate_candidates.py \
	--native-pdb storage/4G1E.pdb \
	--name 4G1E \
	--json-file storage/4G1E-ea24b86d99cd41569c9f93d49184e362-add-msa.json \
	--out-dir data/candidates2 \
	--max-deviation 50.0 \
	--grid-size 5.0 \
	--max-trial 3
