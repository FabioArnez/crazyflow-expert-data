#!/usr/bin/env bash
# Evaluate trained expert runs on the fixed test conditions and write one JSON per condition.
# Usage: scripts/eval_seeds.sh <out_prefix> <run_dir> [<run_dir> ...]
# Conditions: 5-gate chains (training gate-normal noise), 5-gate circles (training turn range),
# both with random target speeds, 1,024 episodes each; plus the label/chatter analysis on the
# training distribution (512 episodes). Deterministic policy, DR on.
set -euo pipefail
prefix=$1
shift
for run in "$@"; do
  seed=$(grep -E "^  seed:" "$run/.hydra/config.yaml" | awk '{print $2}')
  ckpt="$run/checkpoint.pkl"
  uv run expert-eval --ckpt "$ckpt" --n-gates 5 --track-type chain --n-episodes 1024 \
    --json "${prefix}_seed${seed}_chain5.json" > /dev/null
  uv run expert-eval --ckpt "$ckpt" --n-gates 5 --track-type circle --n-episodes 1024 \
    --json "${prefix}_seed${seed}_circle5.json" > /dev/null
  uv run python scripts/analyse_label_consistency.py --ckpt "$ckpt" --n-episodes 512 \
    --out "${prefix}_seed${seed}_labels.json" > /dev/null
  echo "seed ${seed}: done"
done
