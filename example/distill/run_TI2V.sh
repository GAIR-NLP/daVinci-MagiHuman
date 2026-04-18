#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-6016}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export WORLD_SIZE="$((GPUS_PER_NODE * NNODES))"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ALGO="${NCCL_ALGO:-^NVLS}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Optional runtime knobs. Edit these defaults in the script when needed.
CPU_OFFLOAD="${CPU_OFFLOAD:-false}"
ENABLE_MAGI_COMPILER_OFFLOAD="${ENABLE_MAGI_COMPILER_OFFLOAD:-false}"
GPU_RESIDENT_WEIGHT_RATIO="${GPU_RESIDENT_WEIGHT_RATIO:-0.35}"
OFFLOAD_POLICY="${OFFLOAD_POLICY:-HEURISTIC}"
CP_SIZE="${CP_SIZE:-${GPUS_PER_NODE}}"
LAUNCH_PREFIX="${LAUNCH_PREFIX:-}"
export CPU_OFFLOAD
MAGI_COMPILER_OFFLOAD_ARGS=""
if [[ "${ENABLE_MAGI_COMPILER_OFFLOAD}" == "true" ]]; then
  MAGI_COMPILER_OFFLOAD_ARGS="--offload_config.model_cpu_offload --offload_config.gpu_resident_weight_ratio ${GPU_RESIDENT_WEIGHT_RATIO} --offload_config.offload_policy ${OFFLOAD_POLICY}"
fi
DISTRIBUTED_ARGS="--nnodes=${NNODES} --node_rank=${NODE_RANK} --nproc_per_node=${GPUS_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT}"

# ==============================================================================================
# RUNNING ON CONSUMER GPUs (e.g., RTX 5090)
# ==============================================================================================
# If you want to run this script on a consumer GPU, please follow these steps to avoid OOM errors:
#
# 1. Define MAGI_COMPILER_OFFLOAD_ARGS and append it to the `torchrun` command below.
# 2. By default, `CP_SIZE` follows `GPUS_PER_NODE`. Override `CP_SIZE` only if you need a different context parallel size.
# 3. Depending on your NUMA node configuration, use `numactl` as a prefix to optimize memory bandwidth:
#    - If spanning multiple NUMA nodes: `numactl --interleave=all`
#    - If on a single NUMA node:        `numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE`
#
# --- Example Usage ---
# MAGI_COMPILER_OFFLOAD_ARGS="--offload_config.model_cpu_offload --offload_config.gpu_resident_weight_ratio 0.35 --offload_config.offload_policy HEURISTIC"
# numactl --interleave=all torchrun ${DISTRIBUTED_ARGS} inference/pipeline/entry.py ... $MAGI_COMPILER_OFFLOAD_ARGS
# ==============================================================================================

$LAUNCH_PREFIX torchrun ${DISTRIBUTED_ARGS} inference/pipeline/entry.py ${MAGI_COMPILER_OFFLOAD_ARGS} \
  --config-load-path example/distill/config.json \
  --engine_config.cp_size "${CP_SIZE}" \
  --prompt "$(<example/assets/prompt.txt)" \
  --image_path example/assets/image.png \
  --seconds 4 \
  --br_width 448 \
  --br_height 256 \
  --output_path "output_example_distill_ti2v_$(date '+%Y%m%d_%H%M%S')" \
  2>&1 | tee "log_example_distill_ti2v_$(date '+%Y%m%d_%H%M%S').log"
