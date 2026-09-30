#!/bin/bash
# Target GPU (e.g. AMD): one precision setting against the copied reference outputs.
#SBATCH --job-name=crova-target
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
MODEL=${MODEL:-jais}
PRECISION=${PRECISION:-bf16}
OUT=runs/$MODEL/amd/$PRECISION
uv run crova forward --config configs/$MODEL/forward-target.yaml \
  --set precision=$PRECISION --set output=$OUT/stats
uv run crova generate --config configs/$MODEL/generate.yaml \
  --set precision=$PRECISION --set splits=[development] --set output=$OUT/responses
uv run crova compare --config configs/$MODEL/compare.yaml \
  --set stats=$OUT/stats --set target_responses=$OUT/responses --set output=$OUT/summary.json
uv run crova accuracy --config configs/$MODEL/accuracy.yaml \
  --set precision=$PRECISION --set output=$OUT/accuracy.json
