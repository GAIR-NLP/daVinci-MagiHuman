#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-6021}"
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
# Example:
# CPU_OFFLOAD="${CPU_OFFLOAD:-true}"
# ENABLE_MAGI_COMPILER_OFFLOAD="${ENABLE_MAGI_COMPILER_OFFLOAD:-true}"
export CPU_OFFLOAD
MAGI_COMPILER_OFFLOAD_ARGS=""
if [[ "${ENABLE_MAGI_COMPILER_OFFLOAD}" == "true" ]]; then
  MAGI_COMPILER_OFFLOAD_ARGS="--offload_config.model_cpu_offload --offload_config.gpu_resident_weight_ratio ${GPU_RESIDENT_WEIGHT_RATIO} --offload_config.offload_policy ${OFFLOAD_POLICY}"
fi
DISTRIBUTED_ARGS="--nnodes=${NNODES} --node_rank=${NODE_RANK} --nproc_per_node=${GPUS_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT}"
PROMPT_PATH="${PROMPT_PATH:-example/assets/video10.txt}"
IMAGE_PATH="${IMAGE_PATH:-example/assets/video10.jpeg}"
AUDIO_PATH="${AUDIO_PATH:-example/assets/video10.ogg}"
BR_WIDTH="${BR_WIDTH:-448}"
BR_HEIGHT="${BR_HEIGHT:-256}"

if [[ ! -f "${PROMPT_PATH}" ]]; then
  echo "Error: PROMPT_PATH does not exist: ${PROMPT_PATH}" >&2
  exit 1
fi

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Error: IMAGE_PATH does not exist: ${IMAGE_PATH}" >&2
  exit 1
fi

if [[ ! -f "${AUDIO_PATH}" ]]; then
  echo "Error: AUDIO_PATH does not exist: ${AUDIO_PATH}" >&2
  echo "Edit AUDIO_PATH near the top of this script before running run_TIA2V.sh." >&2
  exit 1
fi

$LAUNCH_PREFIX torchrun ${DISTRIBUTED_ARGS} inference/pipeline/entry.py ${MAGI_COMPILER_OFFLOAD_ARGS} \
  --config-load-path example/base/config.json \
  --engine_config.cp_size "${CP_SIZE}" \
  --prompt "$(<"${PROMPT_PATH}")" \
  --image_path "${IMAGE_PATH}" \
  --audio_path "${AUDIO_PATH}" \
  --seconds 4 \
  --br_width "${BR_WIDTH}" \
  --br_height "${BR_HEIGHT}" \
  --output_path "output_example_base_tia2v_${BR_WIDTH}x${BR_HEIGHT}_$(date '+%Y%m%d_%H%M%S')" \
  2>&1 | tee "log_example_base_tia2v_${BR_WIDTH}x${BR_HEIGHT}_$(date '+%Y%m%d_%H%M%S').log"
