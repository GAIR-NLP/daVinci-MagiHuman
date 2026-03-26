#!/usr/bin/env bash
# Persistent inference server — loads models once and keeps them in GPU memory.
#
# Usage:
#   bash run_server.sh                        # foreground (Ctrl-C to stop)
#   bash run_server.sh &                      # background
#   SERVER_PORT=9000 bash run_server.sh       # custom port (default: 8765)
#   CONFIG_PATH=example/distill/config.json bash run_server.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate davinci

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ALGO="${NCCL_ALGO:-^NVLS}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PORT="${SERVER_PORT:-8765}"

echo "[run_server] Starting server on port ${PORT} ..."
echo "[run_server] Config: ${CONFIG_PATH:-example/base/config.json}"
echo "[run_server] Stop with Ctrl-C or kill the process."

torchrun \
  --nnodes=1 --node_rank=0 --nproc_per_node=1 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 \
  inference/pipeline/entry.py \
  --config-load-path "${CONFIG_PATH:-example/base/config.json}" \
  --serve \
  --port "${PORT}" \
  2>&1 | tee "log_server_$(date '+%Y%m%d_%H%M%S').log"
