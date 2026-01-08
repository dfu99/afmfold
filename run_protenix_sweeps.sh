#!/bin/bash

JSON_FILE=storage/4G1E-ea24b86d99cd41569c9f93d49184e362-add-msa.json
OUT_DIR=data/test_guided_sweep
BC_PDB=storage/4G1E.pdb

EO_OFFSETS="20,30,40"
EC_OFFSETS="-5,-10"
Y_MAX_LIST="1.0,2.0,3.0"
T_START_LIST="0.1,0.2,0.4"
N_SAMPLE=2

SEEDS=(101 102 103 104 105 106 107 108)
NUM_SHARDS=4

CHUNK_SIZE=$(( ( ${#SEEDS[@]} + NUM_SHARDS - 1 ) / NUM_SHARDS ))

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
  START=$((SHARD_ID * CHUNK_SIZE))
  END=$((START + CHUNK_SIZE))
  CHUNK=("${SEEDS[@]:$START:$CHUNK_SIZE}")
  if [ ${#CHUNK[@]} -eq 0 ]; then
    continue
  fi
  SEED_LIST=$(IFS=,; echo "${CHUNK[*]}")
  sbatch -- protenix_sweep.sh \
    "$JSON_FILE" \
    "$OUT_DIR" \
    "$BC_PDB" \
    "$SEED_LIST" \
    "$EO_OFFSETS" \
    "$EC_OFFSETS" \
    "$Y_MAX_LIST" \
    "$T_START_LIST" \
    "$N_SAMPLE"
done
