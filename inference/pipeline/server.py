# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persistent inference server.

Loads all models once at startup, then serves generation requests over HTTP.
Start with run_server.sh; send requests via run_custom.sh or curl.

POST /generate  — JSON body, returns JSON with output_path / error
GET  /healthz   — liveness probe
"""

import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# Ensure project root is importable when launched via torchrun
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.common import parse_config
from inference.infra import initialize_infra
from inference.model.dit import get_dit
from inference.utils import print_rank_0

try:
    from .pipeline import MagiPipeline
except ImportError:
    from inference.pipeline.pipeline import MagiPipeline

# ---------------------------------------------------------------------------
# Shared state between HTTP handler thread and inference main thread
# ---------------------------------------------------------------------------
_request_queue: queue.Queue = queue.Queue()
_result_store: dict = {}

_VALID_KWARGS = frozenset({
    "seed", "seconds",
    "br_width", "br_height",
    "sr_width", "sr_height",
    "output_width", "output_height",
    "upsample_mode",
})


# ---------------------------------------------------------------------------
# HTTP handler (runs in a background daemon thread)
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # suppress per-request access logs
        pass

    def do_GET(self):
        if self.path == "/healthz":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length))
        except Exception as exc:
            self._json(400, {"error": f"invalid JSON: {exc}"})
            return

        if "prompt" not in req or "image_path" not in req:
            self._json(400, {"error": "prompt and image_path are required"})
            return

        req_id = f"{time.time():.6f}"
        done = threading.Event()
        _request_queue.put((req_id, req, done))
        # Wait up to 2 hours for inference to complete
        done.wait(timeout=7200)

        result = _result_store.pop(req_id, {"error": "timeout"})
        self._json(200 if "output_path" in result else 500, result)

    def _json(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)


def _start_http(port: int) -> HTTPServer:
    srv = HTTPServer(("localhost", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    port = int(os.environ.get("SERVER_PORT", "8765"))

    initialize_infra()
    config = parse_config()

    print_rank_0("[Server] Loading models …")
    t0 = time.time()
    model = get_dit(config.arch_config, config.engine_config)
    pipeline = MagiPipeline(model, config.evaluation_config)
    print_rank_0(f"[Server] Ready in {time.time() - t0:.1f}s  —  http://localhost:{port}")

    _start_http(port)

    # Inference loop runs on the main thread to keep all CUDA ops on one thread.
    while True:
        try:
            req_id, req, done = _request_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            t1 = time.time()
            out = pipeline.run_offline(
                prompt=req["prompt"],
                image=req["image_path"],
                audio=req.get("audio_path"),
                save_path_prefix=req.get("output_path", f"output_{req_id}"),
                **{k: v for k, v in req.items() if k in _VALID_KWARGS},
            )
            _result_store[req_id] = {
                "output_path": out,
                "elapsed": round(time.time() - t1, 1),
            }
        except Exception as exc:
            print_rank_0(f"[Server] inference error: {exc}")
            _result_store[req_id] = {"error": str(exc)}
        finally:
            done.set()


if __name__ == "__main__":
    main()
