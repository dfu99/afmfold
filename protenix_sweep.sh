#!/bin/bash
#SBATCH -J afmfold_sweep
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

export OMP_NUM_THREADS=6

JSON_FILE=$1
OUT_DIR=$2
BC_PDB=$3
SEEDS=$4
EO_OFFSETS=$5
EC_OFFSETS=$6
Y_MAX_LIST=$7
T_START_LIST=$8
N_SAMPLE=${9}

srun python3 scripts/protenix_unit_sweep.py \
  --json-file "$JSON_FILE" \
  --out-dir "$OUT_DIR" \
  --bc-pdb "$BC_PDB" \
  --name 4G1E \
  --eo-offsets="$EO_OFFSETS" \
  --ec-offsets="$EC_OFFSETS" \
  --y-max-list="$Y_MAX_LIST" \
  --t-start-list="$T_START_LIST" \
  --seeds="$SEEDS" \
  --n-sample "$N_SAMPLE"
