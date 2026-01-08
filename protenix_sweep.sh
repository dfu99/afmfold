#!/bin/bash
#SBATCH -J afmfold_sweep
#SBATCH -A gts-yke8
#SBATCH -N1 --gres=gpu:A100:4
#SBATCH -C A100-80GB
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

source ~/scratch/venv_afmfold/bin/activate
cd ~/scratch/afmfold/

export OMP_NUM_THREADS=6

srun python3 scripts/launch_protenix_shards.py \
  --gpus 0,1,2,3 \
  --command "python3 scripts/protenix_unit_sweep.py \
    --json-file storage/4G1E-ea24b86d99cd41569c9f93d49184e362-add-msa.json \
    --out-dir data/test_guided_sweep \
    --bc-pdb storage/4G1E.pdb \
    --name 4G1E \
    --eo-offsets 20,30,40 \
    --ec-offsets -5,-10 \
    --y-max-list 1.0,2.0,3.0 \
    --t-start-list 0.1,0.2,0.4 \
    --seeds 101,102,103,104 \
    --n-sample 2"
