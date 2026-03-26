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

import argparse
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# When launched via `torchrun path/to/entry.py`, only the script's directory is
# prepended to sys.path.  Explicitly insert the project root so that the
# `inference` package is always importable regardless of how PYTHONPATH is set.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.common import parse_config
from inference.infra import initialize_infra
from inference.model.dit import get_dit
from inference.utils import print_rank_0

try:
    from .pipeline import MagiPipeline
except ImportError:
    # Keep compatibility when entry.py is executed as a script path.
    from inference.pipeline import MagiPipeline

# ---------------------------------------------------------------------------
# Persistent server (used only when --serve is passed)
# ---------------------------------------------------------------------------
_request_queue: queue.Queue = queue.Queue()
_result_store: dict = {}

_SERVE_KWARGS = frozenset({
    "seed", "seconds",
    "br_width", "br_height",
    "sr_width", "sr_height",
    "output_width", "output_height",
    "upsample_mode",
})


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
        done.wait(timeout=7200)  # wait up to 2 hours for inference
        result = _result_store.pop(req_id, {"error": "timeout"})
        self._json(200 if "output_path" in result else 500, result)

    def _json(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)


def _run_server_loop(pipeline: "MagiPipeline", port: int):
    """Start HTTP listener in a daemon thread, then process requests on the main thread."""
    srv = HTTPServer(("localhost", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print_rank_0(f"[Server] Listening on http://localhost:{port}  (Ctrl-C to stop)")

    while True:
        try:
            req_id, req, done = _request_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            t0 = time.time()
            out = pipeline.run_offline(
                prompt=req["prompt"],
                image=req["image_path"],
                audio=req.get("audio_path"),
                save_path_prefix=req.get("output_path", f"output_{req_id}"),
                **{k: v for k, v in req.items() if k in _SERVE_KWARGS},
            )
            _result_store[req_id] = {"output_path": out, "elapsed": round(time.time() - t0, 1)}
        except Exception as exc:
            print_rank_0(f"[Server] inference error: {exc}")
            _result_store[req_id] = {"error": str(exc)}
        finally:
            done.set()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run DiT pipeline with unified offline entry.")
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--save_path_prefix", type=str, help="Path prefix for saving outputs.")
    parser.add_argument("--output_path", type=str, help="Alias of --save_path_prefix for MAGI-style CLI.")

    parser.add_argument("--image_path", type=str, help="Path to image for i2v mode.")
    parser.add_argument(
        "--audio_path", type=str, default=None, help="Path to optional audio for lipsync mode; omit to use i2v or t2v"
    )

    # Optional runtime controls; forwarded to pipeline methods when provided.
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seconds", type=int)
    parser.add_argument("--br_width", type=int)
    parser.add_argument("--br_height", type=int)
    parser.add_argument("--sr_width", type=int)
    parser.add_argument("--sr_height", type=int)
    parser.add_argument("--output_width", type=int)
    parser.add_argument("--output_height", type=int)
    parser.add_argument("--upsample_mode", type=str)
    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_arguments()
    config = parse_config()
    model = get_dit(config.arch_config, config.engine_config)
    pipeline = MagiPipeline(model, config.evaluation_config)

    # Server mode: activated via env var MAGI_SERVE=1 to avoid CLI arg
    # conflicts with pydantic-settings' own argument parser.
    if os.environ.get("MAGI_SERVE") == "1":
        port = int(os.environ.get("SERVER_PORT", "8765"))
        _run_server_loop(pipeline, port)
        return  # unreachable; loop runs until Ctrl-C

    save_path_prefix = args.save_path_prefix or args.output_path
    if not save_path_prefix:
        print_rank_0("Error: --save_path_prefix (or --output_path) is required.")
        sys.exit(1)

    optional_kwargs = {
        "seed": args.seed,
        "seconds": args.seconds,
        "br_width": args.br_width,
        "br_height": args.br_height,
        "sr_width": args.sr_width,
        "sr_height": args.sr_height,
        "output_width": args.output_width,
        "output_height": args.output_height,
        "upsample_mode": args.upsample_mode,
    }
    optional_kwargs = {k: v for k, v in optional_kwargs.items() if v is not None and v is not False}

    prompt = args.prompt
    image_path = args.image_path
    audio_path = args.audio_path

    if not prompt:
        print_rank_0("Error: --prompt is required.")
        sys.exit(1)
    if not image_path:
        print_rank_0("Error: --image_path is required.")
        sys.exit(1)

    pipeline.run_offline(
        prompt=prompt, image=image_path, audio=audio_path, save_path_prefix=save_path_prefix, **optional_kwargs
    )


if __name__ == "__main__":
    initialize_infra()
    main()
