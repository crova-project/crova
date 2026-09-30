#!/bin/bash
# Reference GPU (e.g. NVIDIA): responses and logits for one model, split over 4 GPUs.
#SBATCH --job-name=crova-reference
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
set -euo pipefail
MODEL=${MODEL:-jais}
SHARD="--set shard=${SLURM_ARRAY_TASK_ID} --set num_shards=4"
uv run crova generate --config configs/$MODEL/generate.yaml $SHARD
uv run crova forward --config configs/$MODEL/forward-reference.yaml $SHARD
