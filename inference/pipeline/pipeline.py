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

import os
import random
from typing import Optional, Union

import imageio
import soundfile as sf
import torch
from PIL import Image

from inference.common import EvaluationConfig, parse_config
from inference.device_utils import get_device, fork_rng_devices
from inference.model.dit import get_dit
from inference.model.dit import DiTModel
from .video_generate import MagiEvaluator
from .video_process import merge_video_and_audio, upsample_video


class MagiPipeline:
    """Pipeline facade for inference."""

    def __init__(self, model: DiTModel, evaluation_config: EvaluationConfig, device: str = None):
        device = device or get_device()
        self.model = model
        self.evaluation_config = evaluation_config
        config = parse_config()
        if evaluation_config.use_sr_model:
            config.engine_config.load = evaluation_config.sr_model_path
            sr_model = get_dit(config.sr_arch_config, config.engine_config)
        else:
            sr_model = None
        self.evaluator = MagiEvaluator(model, sr_model, evaluation_config, device)

    def _validate_offline_request(
        self,
        prompt: str,
        save_path_prefix: str,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("`prompt` must be a non-empty string.")
        if not save_path_prefix or not save_path_prefix.strip():
            raise ValueError("`save_path_prefix` must be a non-empty string.")

    def run_offline(
        self,
        prompt: str,
        image: Union[str, Image.Image, None],
        audio: Optional[str],
        save_path_prefix: str,
        seed: int = 42,
        seconds: int = 4,
        br_width: int = 480,
        br_height: int = 272,
        sr_width: Optional[int] = None,
        sr_height: Optional[int] = None,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
        upsample_mode: Optional[str] = None,
    ):
        self._validate_offline_request(prompt=prompt, save_path_prefix=save_path_prefix)

        if self.evaluator.sr_model is not None:
            save_path = f"{save_path_prefix}_{seconds}s_{br_width}x{br_height}_{sr_width}x{sr_height}.mp4"
        else:
            save_path = f"{save_path_prefix}_{seconds}s_{br_width}x{br_height}.mp4"

        with torch.random.fork_rng(devices=fork_rng_devices()):
            torch.random.manual_seed(seed)
            video_np, audio_np = self.evaluator.evaluate(
                prompt,
                image,
                audio,
                seconds=seconds,
                br_width=br_width,
                br_height=br_height,
                sr_width=sr_width,
                sr_height=sr_height,
                br_num_inference_steps=self.evaluation_config.num_inference_steps,
                sr_num_inference_steps=self.evaluation_config.sr_num_inference_steps,
            )

        if output_width is not None and output_height is not None:
            video_np = upsample_video(video_np, output_width, output_height, upsample_mode)

        # Save output (on rank 0 or single-process mode)
        is_last_rank = True
        if torch.distributed.is_initialized():
            is_last_rank = torch.distributed.get_rank() == torch.distributed.get_world_size() - 1

        if is_last_rank:
            saving_name = f"{prompt.replace(' ', '_')[:10]}"
            audio_path = saving_name + str(random.randint(0, 1000000)) + ".wav"
            video_path = saving_name + str(random.randint(0, 1000000)) + ".mp4"
            sf.write(audio_path, audio_np, self.evaluator.audio_vae.sample_rate)
            imageio.mimwrite(video_path, video_np, fps=self.evaluation_config.fps, quality=8, output_params=["-loglevel", "error"])
            assert os.path.exists(video_path)
            merge_video_and_audio(video_path, audio_path, save_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return save_path
