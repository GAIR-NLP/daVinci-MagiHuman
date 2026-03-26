#!/usr/bin/env bash
# Usage: run_custom.sh <prompt> <image_path> [br_width] [br_height] [seconds]
#
# If run_server.sh is already running, requests are forwarded to it (no model reload).
# Otherwise falls back to direct torchrun invocation (slow, reloads models).
#
# Examples:
#   bash run_server.sh &          # start server once
#   ~/run_custom.sh "smile" image.png
#   ~/run_custom.sh "smile" image.png 448 256 10

cd ~/daVinci-MagiHuman

PROMPT="${1:?Usage: run_custom.sh <prompt> <image_path> [br_width] [br_height] [seconds]}"
IMAGE="${2:?image_path required}"
BR_W="${3:-448}"
BR_H="${4:-256}"
SECS="${5:-5}"
PORT="${SERVER_PORT:-8765}"
OUT="output_$(date '+%Y%m%d_%H%M%S')"

# ---------------------------------------------------------------------------
# Try the persistent server first (no model reload)
# ---------------------------------------------------------------------------
if python3 -c "
import urllib.request, sys
try:
    urllib.request.urlopen('http://localhost:${PORT}/healthz', timeout=2)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then

  echo "[client] Server on port ${PORT} detected — forwarding request …"

  PROMPT="$PROMPT" IMAGE="$IMAGE" \
  BR_W="$BR_W" BR_H="$BR_H" SECS="$SECS" \
  OUT="$OUT" PORT="$PORT" \
  python3 <<'PYEOF'
import json, os, sys, time, urllib.request

req = {
    "prompt":     os.environ["PROMPT"],
    "image_path": os.environ["IMAGE"],
    "br_width":   int(os.environ["BR_W"]),
    "br_height":  int(os.environ["BR_H"]),
    "seconds":    int(os.environ["SECS"]),
    "output_path": os.environ["OUT"],
}
data = json.dumps(req).encode()

t0 = time.time()
try:
    resp = urllib.request.urlopen(
        urllib.request.Request(
            f"http://localhost:{os.environ['PORT']}/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        ),
        timeout=7200,
    )
    result = json.loads(resp.read())
except urllib.error.HTTPError as e:
    result = json.loads(e.read())

if "error" in result:
    print(f"[client] ERROR: {result['error']}", file=sys.stderr)
    sys.exit(1)

print(f"[client] Done in {time.time() - t0:.1f}s  —  {result['output_path']}")
PYEOF
  exit $?
fi

# ---------------------------------------------------------------------------
# Fallback: direct torchrun (models are reloaded each time)
# ---------------------------------------------------------------------------
echo "[client] Server not found on port ${PORT} — launching directly (models will be reloaded) …"
echo "[client] Tip: start the server once with:  bash run_server.sh &"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate davinci

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_ADDR=localhost
export MASTER_PORT=6009
export NNODES=1 NODE_RANK=0 GPUS_PER_NODE=1 WORLD_SIZE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ALGO="${NCCL_ALGO:-^NVLS}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:6009 \
  inference/pipeline/entry.py \
  --config-load-path example/base/config.json \
  --prompt "$PROMPT" \
  --image_path "$IMAGE" \
  --br_width "$BR_W" \
  --br_height "$BR_H" \
  --seconds "$SECS" \
  --output_path "$OUT" \
  2>&1 | tee "log_$(date '+%Y%m%d_%H%M%S').log"
