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

"""
Diffusers-compatible pipeline for MagiHuman audio-video generation.

Usage:
    from pipeline_magihuman import MagiHumanPipeline

    pipe = MagiHumanPipeline.from_pretrained(
        "Sand-AI/MagiHuman",
        dit_path="/path/to/checkpoints/base",
        audio_model_path="/path/to/stable-audio-open-1.0",
        txt_model_path="/path/to/t5gemma-9b-9b-ul2",
        vae_model_path="/path/to/Wan2.2-TI2V-5B",
        turbo_vae_config_path="/path/to/TurboV3-Wan22-TinyShallow_7_7.json",
        turbo_vae_ckpt_path="/path/to/checkpoint-340000.ckpt",
    )

    # Text-to-video
    result = pipe("A woman talking naturally", seconds=5, height=272, width=480)

    # Image-to-video
    from PIL import Image
    img = Image.open("reference.png")
    result = pipe("A woman talking", image=img, seconds=5)

    # Audio-driven (lip-sync)
    result = pipe("A woman talking", image=img, audio_path="speech.wav", seconds=5)

    # Access outputs
    result.video   # np.ndarray (T, H, W, C) uint8
    result.audio   # np.ndarray (S, C) float32
"""

import gc
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput, logging
from PIL import Image

logger = logging.get_logger(__name__)


@dataclass
class MagiHumanPipelineOutput(BaseOutput):
    """Output of :class:`MagiHumanPipeline`.

    Attributes:
        video: Video frames as ``np.ndarray`` with shape ``(T, H, W, C)`` and dtype ``uint8``.
        audio: Audio samples as ``np.ndarray`` with shape ``(S, C)`` and dtype ``float32``.
    """

    video: np.ndarray
    audio: np.ndarray


class MagiHumanPipeline(DiffusionPipeline):
    """Diffusers-compatible pipeline for MagiHuman audio-video generation.

    This wraps the native MagiHuman inference stack (DiT + VAEs + T5-Gemma +
    FlowUniPC scheduler) so it can be used via the standard diffusers API.
    """

    def __init__(self, evaluator):
        super().__init__()
        # Store the evaluator; we do NOT register sub-modules via
        # register_modules() because the MagiHuman components don't follow
        # standard diffusers ModelMixin conventions for all parts.
        self._evaluator = evaluator

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        *,
        dit_path: Optional[str] = None,
        sr_dit_path: Optional[str] = None,
        audio_model_path: str = "",
        txt_model_path: str = "",
        vae_model_path: str = "",
        turbo_vae_config_path: str = "",
        turbo_vae_ckpt_path: str = "",
        use_turbo_vae: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        # Scheduler / generation defaults
        num_inference_steps: int = 32,
        cfg_number: int = 2,
        shift: float = 5.0,
        video_txt_guidance_scale: float = 5.0,
        audio_txt_guidance_scale: float = 5.0,
        sr_num_inference_steps: int = 5,
        sr_video_txt_guidance_scale: float = 3.5,
        **kwargs,
    ):
        """Build a :class:`MagiHumanPipeline` from local checkpoint paths.

        Parameters
        ----------
        pretrained_model_name_or_path:
            Optional base directory.  When provided, relative component paths
            are resolved against this directory.
        dit_path:
            Path to the base DiT checkpoint directory (contains
            ``model.safetensors`` or sharded index).
        sr_dit_path:
            Path to the super-resolution DiT checkpoint (optional).
        audio_model_path:
            Path to ``stabilityai/stable-audio-open-1.0``.
        txt_model_path:
            Path to ``google/t5gemma-9b-9b-ul2``.
        vae_model_path:
            Path to ``Wan-AI/Wan2.2-TI2V-5B`` (must contain ``Wan2.2_VAE.pth``).
        turbo_vae_config_path:
            JSON config for TurboVAED.
        turbo_vae_ckpt_path:
            Checkpoint for TurboVAED.
        use_turbo_vae:
            Whether to use TurboVAED for faster decoding (default True).
        torch_dtype:
            Weight dtype (default ``bfloat16``).
        device:
            Target device (default ``"cuda"``).
        num_inference_steps:
            Denoising steps for base resolution (default 32).
        cfg_number:
            Classifier-free guidance passes (1 = distilled/no-CFG, 2 = full CFG).
        """
        # Lazy imports so that users who only inspect the file don't need the
        # full MagiHuman dependency tree installed.
        from inference.common import (
            EvaluationConfig,
            ModelConfig,
            EngineConfig,
            ensure_distributed,
        )
        from inference.model.dit import get_dit, DiTModel
        from inference.pipeline.video_generate import MagiEvaluator

        # 1. Ensure torch.distributed is initialised (single-GPU safe).
        ensure_distributed()

        # 2. Resolve paths -------------------------------------------------
        base = pretrained_model_name_or_path or ""

        def _resolve(p: str) -> str:
            if not p:
                return p
            if os.path.isabs(p):
                return p
            joined = os.path.join(base, p)
            return joined if os.path.exists(joined) else p

        dit_path = _resolve(dit_path or "")
        sr_dit_path = _resolve(sr_dit_path or "") if sr_dit_path else None
        audio_model_path = _resolve(audio_model_path)
        txt_model_path = _resolve(txt_model_path)
        vae_model_path = _resolve(vae_model_path)
        turbo_vae_config_path = _resolve(turbo_vae_config_path)
        turbo_vae_ckpt_path = _resolve(turbo_vae_ckpt_path)

        # 3. Build configs --------------------------------------------------
        model_config = ModelConfig()
        model_config.num_heads_q = model_config.hidden_size // model_config.head_dim
        model_config.num_heads_kv = model_config.num_query_groups

        engine_config = EngineConfig(load=dit_path, cp_size=1)

        eval_config = EvaluationConfig(
            num_inference_steps=num_inference_steps,
            cfg_number=cfg_number,
            shift=shift,
            video_txt_guidance_scale=video_txt_guidance_scale,
            audio_txt_guidance_scale=audio_txt_guidance_scale,
            audio_model_path=audio_model_path,
            txt_model_path=txt_model_path,
            vae_model_path=vae_model_path,
            use_turbo_vae=use_turbo_vae,
            student_config_path=turbo_vae_config_path,
            student_ckpt_path=turbo_vae_ckpt_path,
            sr_num_inference_steps=sr_num_inference_steps,
            sr_video_txt_guidance_scale=sr_video_txt_guidance_scale,
            use_sr_model=sr_dit_path is not None and sr_dit_path != "",
            sr_model_path=sr_dit_path or "",
        )

        # 4. Load DiT model ------------------------------------------------
        logger.info("Loading base DiT model from %s", dit_path)
        dit_model = get_dit(model_config, engine_config)

        # 5. Optionally load SR DiT ----------------------------------------
        sr_model = None
        if sr_dit_path:
            logger.info("Loading SR DiT model from %s", sr_dit_path)
            sr_engine_config = EngineConfig(load=sr_dit_path, cp_size=1)
            sr_model = get_dit(model_config, sr_engine_config)

        # 6. Build evaluator ------------------------------------------------
        logger.info("Building MagiEvaluator …")
        evaluator = MagiEvaluator(
            model=dit_model,
            sr_model=sr_model,
            config=eval_config,
            device=device,
            weight_dtype=torch_dtype,
        )

        gc.collect()
        torch.cuda.empty_cache()

        pipe = cls(evaluator=evaluator)
        pipe._eval_config = eval_config
        return pipe

    @classmethod
    def from_config_json(
        cls,
        config_path: str,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """Build a pipeline from an existing MagiHuman ``config.json``.

        This mirrors the native launch flow (``--config-load-path config.json``)
        but returns a diffusers pipeline instead.

        Parameters
        ----------
        config_path:
            Path to a MagiHuman config JSON (e.g. ``example/base/config.json``).
        """
        import json

        with open(config_path) as f:
            cfg = json.load(f)

        engine = cfg.get("engine_config", {})
        evaluation = cfg.get("evaluation_config", {})

        return cls.from_pretrained(
            pretrained_model_name_or_path=None,
            dit_path=engine.get("load", ""),
            audio_model_path=evaluation.get("audio_model_path", ""),
            txt_model_path=evaluation.get("txt_model_path", ""),
            vae_model_path=evaluation.get("vae_model_path", ""),
            turbo_vae_config_path=evaluation.get("student_config_path", ""),
            turbo_vae_ckpt_path=evaluation.get("student_ckpt_path", ""),
            use_turbo_vae=evaluation.get("use_turbo_vae", True),
            num_inference_steps=evaluation.get("num_inference_steps", 32),
            cfg_number=evaluation.get("cfg_number", 2),
            torch_dtype=torch_dtype,
            device=device,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        *,
        image: Optional[Union[str, Image.Image]] = None,
        audio_path: Optional[str] = None,
        seconds: int = 5,
        height: int = 272,
        width: int = 480,
        sr_height: Optional[int] = None,
        sr_width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        sr_num_inference_steps: Optional[int] = None,
        seed: int = 42,
        return_dict: bool = True,
    ) -> Union[MagiHumanPipelineOutput, Tuple[np.ndarray, np.ndarray]]:
        """Generate an audio-video clip.

        Parameters
        ----------
        prompt:
            Text description of the desired video.
        image:
            Reference image for image-to-video mode (PIL Image or file path).
        audio_path:
            Path to a ``.wav`` file for audio-driven lip-sync mode.
        seconds:
            Duration of the generated video in seconds (default 5).
        height / width:
            Base resolution (default 272x480 for 256p).
        sr_height / sr_width:
            Super-resolution target (e.g. 544x960 for 540p, or 1088x1920 for
            1080p).  Requires a SR model loaded at init time.
        num_inference_steps:
            Override the default denoising steps for this call.
        sr_num_inference_steps:
            Override SR denoising steps for this call.
        seed:
            Random seed for reproducibility.
        return_dict:
            If ``True`` (default), return a :class:`MagiHumanPipelineOutput`.
            Otherwise return a ``(video, audio)`` tuple.

        Returns
        -------
        :class:`MagiHumanPipelineOutput` or ``(video, audio)`` tuple.
        """
        evaluator = self._evaluator

        # Resolve image path -> PIL
        if isinstance(image, str):
            from diffusers.utils import load_image
            image = load_image(image)

        br_steps = num_inference_steps or self._eval_config.num_inference_steps
        sr_steps = sr_num_inference_steps or self._eval_config.sr_num_inference_steps

        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.random.manual_seed(seed)
            video_np, audio_np = evaluator.evaluate(
                prompt=prompt,
                image=image,
                audio_path=audio_path,
                seconds=seconds,
                br_width=width,
                br_height=height,
                sr_width=sr_width,
                sr_height=sr_height,
                br_num_inference_steps=br_steps,
                sr_num_inference_steps=sr_steps,
            )

        if not return_dict:
            return video_np, audio_np
        return MagiHumanPipelineOutput(video=video_np, audio=audio_np)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def save_output(
        self,
        output: MagiHumanPipelineOutput,
        save_path: str,
        fps: int = 25,
    ) -> str:
        """Save pipeline output as an MP4 with merged audio.

        Parameters
        ----------
        output:
            A :class:`MagiHumanPipelineOutput` returned by ``__call__``.
        save_path:
            Destination ``.mp4`` path.
        fps:
            Frames per second (default 25).

        Returns
        -------
        The ``save_path`` string.
        """
        import random as _random
        import imageio
        import soundfile as sf
        from inference.pipeline.video_process import merge_video_and_audio

        sample_rate = self._evaluator.audio_vae.sample_rate
        tag = _random.randint(0, 999999)
        tmp_video = f"/tmp/_magi_tmp_video_{tag}.mp4"
        tmp_audio = f"/tmp/_magi_tmp_audio_{tag}.wav"

        sf.write(tmp_audio, output.audio, sample_rate)
        imageio.mimwrite(
            tmp_video,
            output.video,
            fps=fps,
            quality=8,
            output_params=["-loglevel", "error"],
        )
        merge_video_and_audio(tmp_video, tmp_audio, save_path)
        return save_path
