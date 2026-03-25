#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-6011}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export WORLD_SIZE="$((GPUS_PER_NODE * NNODES))"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ALGO="${NCCL_ALGO:-^NVLS}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CPU_OFFLOAD=true

DISTRIBUTED_ARGS="--nnodes=${NNODES} --node_rank=${NODE_RANK} --nproc_per_node=${GPUS_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT}"

torchrun ${DISTRIBUTED_ARGS} inference/pipeline/entry.py \
  --config-load-path example/sr_540p/config.json \
  --prompt "$(<example/assets/prompt.txt)" \
  --image_path example/assets/image.png \
  --seconds 10 \
  --br_width 448 \
  --br_height 256 \
  --output_path "output_example_sr_540p_$(date '+%Y%m%d_%H%M%S')" \
  --sr_width 896 \
  --sr_height 512 \
  2>&1 | tee "log_example_sr_540p_$(date '+%Y%m%d_%H%M%S').log"
