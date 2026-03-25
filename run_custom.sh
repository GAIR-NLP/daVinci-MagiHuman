#!/usr/bin/env bash
cd ~/daVinci-MagiHuman
export CUDA_VISIBLE_DEVICES=0
source ~/miniforge3/etc/profile.d/conda.sh
conda activate davinci
export MASTER_ADDR=localhost
export MASTER_PORT=6009
export NNODES=1
export NODE_RANK=0
export GPUS_PER_NODE=1
export WORLD_SIZE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_ALGO=^NVLS
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:6009 \
  inference/pipeline/entry.py \
  --config-load-path example/base/config.json \
  --prompt "$1" \
  --image_path "$2" \
  --br_width "${3:-448}" \
  --br_height "${4:-256}" \
  --seconds "${5:-5}" \
  --output_path "output_$(date '+%Y%m%d_%H%M%S')"
