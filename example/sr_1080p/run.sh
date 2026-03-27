#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-6012}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export WORLD_SIZE="$((GPUS_PER_NODE * NNODES))"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ALGO="${NCCL_ALGO:-^NVLS}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export SR2_1080="${SR2_1080:-true}"
export CPU_OFFLOAD="${CPU_OFFLOAD:-true}"

DISTRIBUTED_ARGS="--nnodes=${NNODES} --node_rank=${NODE_RANK} --nproc_per_node=${GPUS_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT}"

# ==============================================================================================
# RUNNING ON CONSUMER GPUs (e.g., RTX 5090)
# ==============================================================================================
# If you want to run this script on a consumer GPU, please follow these steps to avoid OOM errors:
# 
# 1. Define MAGI_COMPILER_OFFLOAD_ARGS and append it to the `torchrun` command below.
# 2. Update `engine_config.cp_size` in `config.json` to exactly match the number of GPUs on your machine.
# 3. Depending on your NUMA node configuration, use `numactl` as a prefix to optimize memory bandwidth:
#    - If spanning multiple NUMA nodes: `numactl --interleave=all`
#    - If on a single NUMA node:        `numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE`
#
# --- Example Usage ---
# MAGI_COMPILER_OFFLOAD_ARGS="--offload_config.model_cpu_offload --offload_config.gpu_resident_weight_ratio 0.35 --offload_config.offload_policy HEURISTIC"
# numactl --interleave=all torchrun ${DISTRIBUTED_ARGS} inference/pipeline/entry.py ... $MAGI_COMPILER_OFFLOAD_ARGS
# ==============================================================================================

torchrun ${DISTRIBUTED_ARGS} inference/pipeline/entry.py \
  --config-load-path example/sr_1080p/config.json \
  --prompt "$(<example/assets/prompt.txt)" \
  --image_path example/assets/image.png \
  --seconds 4 \
  --br_width 448 \
  --br_height 256 \
  --sr_width 1920 \
  --sr_height 1088 \
  --output_path "output_example_sr_1080p_$(date '+%Y%m%d_%H%M%S')" \
  2>&1 | tee "log_example_sr_1080p_$(date '+%Y%m%d_%H%M%S').log"
