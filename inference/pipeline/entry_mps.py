# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apple Silicon (MPS) port — single-process entry point.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Single-process inference entry point for Apple Silicon (MPS) and CPU.

Unlike the original ``entry.py`` which requires ``torchrun`` and CUDA,
this entry point:
- Runs as a plain ``python`` script (no torchrun)
- Does NOT require distributed init (single GPU on Apple Silicon)
- Sets up the MPS environment variables automatically
"""

import argparse
import os
import sys

# ---- Environment setup (must come before torch import) ----
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from inference.device_utils import get_device, get_dtype, manual_seed_all
from inference.common import parse_config
from inference.utils import print_rank_0, set_random_seed


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run DiT pipeline (MPS / single-process mode)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--save_path_prefix", type=str, help="Path prefix for saving outputs.")
    parser.add_argument("--output_path", type=str, help="Alias of --save_path_prefix.")

    parser.add_argument("--image_path", type=str, help="Path to image for i2v mode.")
    parser.add_argument("--audio_path", type=str, default=None, help="Path to audio for lipsync mode.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=int, default=4)
    parser.add_argument("--br_width", type=int, default=448)
    parser.add_argument("--br_height", type=int, default=256)
    parser.add_argument("--sr_width", type=int)
    parser.add_argument("--sr_height", type=int)
    parser.add_argument("--output_width", type=int)
    parser.add_argument("--output_height", type=int)
    parser.add_argument("--upsample_mode", type=str)
    args, _ = parser.parse_known_args()
    return args


def main():
    device = get_device()
    dtype = get_dtype(device)
    print_rank_0(f"Device: {device}, Dtype: {dtype}")

    args = parse_arguments()
    config = parse_config()

    # Set random seed
    seed = args.seed or config.engine_config.seed
    set_random_seed(seed)

    # Build model (single-process, no distributed)
    from inference.model.dit.dit_module import DiTModel
    from inference.model.dit.dit_model import get_dit
    from inference.pipeline.pipeline import MagiPipeline

    model = get_dit(config.arch_config, config.engine_config)
    pipeline = MagiPipeline(model, config.evaluation_config, device=device)

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
    optional_kwargs = {k: v for k, v in optional_kwargs.items() if v is not None}

    if not args.image_path:
        print_rank_0("Error: --image_path is required.")
        sys.exit(1)

    pipeline.run_offline(
        prompt=args.prompt,
        image=args.image_path,
        audio=args.audio_path,
        save_path_prefix=save_path_prefix,
        **optional_kwargs,
    )


if __name__ == "__main__":
    main()
