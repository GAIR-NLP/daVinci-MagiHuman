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
import sys

from inference.common import parse_config
from inference.infra import initialize_infra
from inference.model.dit import get_dit
from inference.utils import print_rank_0

try:
    from .pipeline import MagiPipeline
except ImportError:
    # Keep compatibility when entry.py is executed as a script path.
    from inference.pipeline import MagiPipeline


def infer_generation_mode(image_path: str | None, audio_path: str | None) -> str:
    if image_path and audio_path:
        return "tia2v"
    if image_path:
        return "ti2v"
    if audio_path:
        return "ta2v"
    return "t2v"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run DiT pipeline with unified offline entry.")
    parser.add_argument("--prompt", type=str, help="Prompt text for generation.")
    parser.add_argument("--save_path_prefix", type=str, help="Path prefix for saving outputs.")
    parser.add_argument("--output_path", type=str, help="Alias of --save_path_prefix for MAGI-style CLI.")

    parser.add_argument(
        "--image_path",
        type=str,
        help="Optional reference image path for ti2v or tia2v mode; omit for text-only generation.",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Optional reference audio path for ta2v or tia2v mode.",
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


def main():
    args = parse_arguments()
    config = parse_config()
    model = get_dit(config.arch_config, config.engine_config)
    pipeline = MagiPipeline(model, config.evaluation_config)
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

    print_rank_0(f"Running inference mode: {infer_generation_mode(image_path, audio_path)}")

    pipeline.run_offline(
        prompt=prompt, image=image_path, audio=audio_path, save_path_prefix=save_path_prefix, **optional_kwargs
    )


if __name__ == "__main__":
    initialize_infra()
    main()
