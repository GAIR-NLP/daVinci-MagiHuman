#!/usr/bin/env bash
# Apple Silicon (MPS) inference script — no torchrun, no CUDA required.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ---- MPS Environment ----
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---- Run inference (single process, no torchrun) ----
python inference/pipeline/entry_mps.py \
  --config-load-path example/distill/config_mps.json \
  --prompt "$(<example/assets/prompt.txt)" \
  --image_path example/assets/image.png \
  --seconds 4 \
  --br_width 448 \
  --br_height 256 \
  --output_path "output_example_distill_mps_$(date '+%Y%m%d_%H%M%S')" \
  2>&1 | tee "log_example_distill_mps_$(date '+%Y%m%d_%H%M%S').log"
