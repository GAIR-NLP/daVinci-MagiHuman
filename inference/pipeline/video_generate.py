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

import copy
import math
import os
import shutil
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor
from PIL import Image
from tqdm import tqdm

from inference.common import CPUOffloadWrapper, EvaluationConfig, get_arch_memory
from inference.device_utils import get_device, get_dtype, empty_cache, is_cuda
from inference.infra.distributed import get_cp_group
from inference.model.dit import DiTModel
from inference.model.sa_audio import SAAudioFeatureExtractor
from inference.model.turbo_vaed import TurboVAED, get_turbo_vaed
from inference.model.vae2_2 import Wan2_2_VAE, get_vae2_2
from inference.utils import env_is_true, event_path_timer, print_mem_info_rank_0, print_rank_0, print_rank_last
from .prompt_process import get_padded_t5_gemma_embedding, pad_or_trim
from .scheduler_unipc import FlowUniPCMultistepScheduler
from .data_proxy import MagiDataProxy
from .video_process import load_audio_and_encode, resample_audio_sinc, resizecrop


def schedule_latent_step(
    *,
    video_scheduler,
    audio_scheduler,
    latent_video: torch.Tensor,
    latent_audio: torch.Tensor,
    t,
    idx: int,
    steps: int,
    v_cfg_video: torch.Tensor,
    v_cfg_audio: torch.Tensor,
    is_a2v: bool,
    cfg_number: int,
    use_sr_model: bool,
    using_sde_flag: bool,
):
    if cfg_number == 1 and (not use_sr_model):
        latent_video = video_scheduler.step_ddim(v_cfg_video, idx, latent_video)
        latent_audio = audio_scheduler.step_ddim(v_cfg_audio, idx, latent_audio)
        return latent_video, latent_audio

    if using_sde_flag:
        print_rank_0("Using sde scheduler")
        if use_sr_model:
            latent_video = video_scheduler.step(v_cfg_video, t, latent_video, return_dict=False)[0]
            return latent_video, latent_audio

        if idx < int(steps * (3 / 4)):
            noise_theta = 1.0 if (idx + 1) % 2 == 0 else 0.0
        else:
            noise_theta = 1.0 if idx % 3 == 0 else 0.0

        latent_video = video_scheduler.step_sde(v_cfg_video, idx, latent_video, noise_theta=noise_theta)
        if not is_a2v:
            latent_audio = audio_scheduler.step_sde(v_cfg_audio, idx, latent_audio, noise_theta=noise_theta)
        return latent_video, latent_audio

    latent_video = video_scheduler.step(v_cfg_video, t, latent_video, return_dict=False)[0]
    if not is_a2v and not use_sr_model:
        latent_audio = audio_scheduler.step(v_cfg_audio, t, latent_audio, return_dict=False)[0]
    return latent_video, latent_audio


@dataclass
class EvalInput:
    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: torch.Tensor | list[int]
    txt_feat: torch.Tensor
    txt_feat_len: torch.Tensor | list[int]


class ZeroSNRDDPMDiscretization:
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
        shift_scale=1.0,  # noise schedule t_n -> t_m: logSNR(t_m) = logSNR(t_n) - log(shift_scale)
        keep_start=False,
        post_shift=False,
    ):
        if keep_start and not post_shift:
            linear_start = linear_start / (shift_scale + (1 - shift_scale) * linear_start)
        self.num_timesteps = num_timesteps
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64) ** 2
        betas = betas.numpy()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

        # SNR shift
        if not post_shift:
            self.alphas_cumprod = self.alphas_cumprod / (shift_scale + (1 - shift_scale) * self.alphas_cumprod)

        self.post_shift = post_shift
        self.shift_scale = shift_scale

    def __call__(self, n, do_append_zero=True, device="cpu", flip=False, return_idx=False):
        if return_idx:
            sigmas, idx = self.get_sigmas(n, device=device, return_idx=return_idx)
        else:
            sigmas = self.get_sigmas(n, device=device, return_idx=return_idx)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])]) if do_append_zero else sigmas
        if return_idx:
            return sigmas if not flip else torch.flip(sigmas, (0,)), idx
        else:
            return sigmas if not flip else torch.flip(sigmas, (0,))

    def get_sigmas(self, n, device="cpu", return_idx=False):
        if n < self.num_timesteps:
            timesteps = np.linspace(self.num_timesteps - 1, 0, n, endpoint=False).astype(int)[::-1]
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        alphas_cumprod = to_torch(alphas_cumprod)
        alphas_cumprod_sqrt = alphas_cumprod.sqrt()
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()

        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)

        if self.post_shift:
            alphas_cumprod_sqrt = (
                alphas_cumprod_sqrt**2 / (self.shift_scale + (1 - self.shift_scale) * alphas_cumprod_sqrt**2)
            ) ** 0.5

        if return_idx:
            return torch.flip(alphas_cumprod_sqrt, (0,)), timesteps
        else:
            return torch.flip(alphas_cumprod_sqrt, (0,))


class MagiEvaluator:
    def __init__(
        self,
        model: DiTModel,
        sr_model: Optional[DiTModel],
        config: EvaluationConfig,
        device: str = None,
        weight_dtype: torch.dtype = None,
    ):
        device = device or get_device()
        weight_dtype = weight_dtype or get_dtype(device)

        self._dit_on_cpu = False  # reserved for future MPS hybrid mode

        self.model = model
        self.model.eval()
        self.sr_model = sr_model
        if self.sr_model is not None:
            self.sr_model.eval()
            if env_is_true("CPU_OFFLOAD") and env_is_true("SR2_1080"):
                self.model = self.model.to(torch.device("cpu"))
                self.sr_model = self.sr_model.to(torch.device("cpu"))
        self.device = device
        self.config = config
        self.dtype = weight_dtype
        self.data_proxy = MagiDataProxy(config.data_proxy_config)
        sr_data_proxy_config = copy.deepcopy(config.data_proxy_config)
        sr_data_proxy_config.coords_style = "v1"
        self.sr_data_proxy = MagiDataProxy(sr_data_proxy_config)
        self.vae_stride = config.vae_stride
        self.z_dim = config.z_dim
        self.patch_size = config.patch_size

        self.sr_video_txt_guidance_scale = config.sr_video_txt_guidance_scale
        self.video_txt_guidance_scale = config.video_txt_guidance_scale
        self.audio_txt_guidance_scale = config.audio_txt_guidance_scale
        self.noise_value = config.noise_value
        self.shift = config.shift
        self.fps = config.fps
        self.use_cfg_trick = config.use_cfg_trick
        self.cfg_trick_start_frame = config.cfg_trick_start_frame
        self.cfg_trick_value = config.cfg_trick_value
        self.using_sde_flag = config.using_sde_flag

        print_mem_info_rank_0("Begin init MagiEvaluator")

        vae_model_path = os.path.join(config.vae_model_path, "Wan2.2_VAE.pth")
        self.vae: Wan2_2_VAE = CPUOffloadWrapper(
            get_vae2_2(vae_model_path, self.device, weight_dtype=weight_dtype), is_cpu_offload=get_arch_memory() <= 48
        )
        if config.use_turbo_vae:
            self.turbo_vae: TurboVAED = CPUOffloadWrapper(
                get_turbo_vaed(config.student_config_path, config.student_ckpt_path, self.device, weight_dtype=weight_dtype),
                is_cpu_offload=get_arch_memory() <= 48,
            )

        print_mem_info_rank_0("After init video vae")
        print_rank_0(f"vae loaded from {vae_model_path}")
        self.video_processor = VideoProcessor(vae_scale_factor=16)
        self.audio_vae = SAAudioFeatureExtractor(device=self.device, model_path=config.audio_model_path)
        self.sigmas = ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)
        print_mem_info_rank_0("After init audio vae")

        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"  # noqa: E501
        negative_prompt += ", low quality, worst quality, poor quality, noise, background noise, hiss, hum, buzz, crackle, static, compression artifacts, MP3 artifacts, digital clipping, distortion, muffled, muddy, unclear, echo, reverb, room echo, over-reverberated, hollow sound, distant, washed out, harsh, shrill, piercing, grating, tinny, thin sound, boomy, bass-heavy, flat EQ, over-compressed, abrupt cut, jarring transition, sudden silence, looping artifact, music, instrumental, sirens, alarms, crowd noise, unrelated sound effects, chaotic, disorganized, messy, cheap sound"
        negative_prompt += ", emotionless, flat delivery, deadpan, lifeless, apathetic, robotic, mechanical, monotone, flat intonation, undynamic, boring, reading from a script, AI voice, synthetic, text-to-speech, TTS, insincere, fake emotion, exaggerated, overly dramatic, melodramatic, cheesy, cringey, hesitant, unconfident, tired, weak voice, stuttering, stammering, mumbling, slurred speech, mispronounced, bad articulation, lisp, vocal fry, creaky voice, mouth clicks, lip smacks, wet mouth sounds, heavy breathing, audible inhales, plosives, p-pops, coughing, clearing throat, sneezing, speaking too fast, rushed, speaking too slow, dragged out, unnatural pauses, awkward silence, choppy, disjointed, multiple speakers, two voices, background talking, out of tune, off-key, autotune artifacts"

        txt_encoder_device = "cpu" if get_arch_memory() <= 48 else self.device
        self.txt_model_path = config.txt_model_path

        self.context_null, self.original_context_null_len = get_padded_t5_gemma_embedding(
            negative_prompt,
            self.txt_model_path,
            txt_encoder_device,
            self.dtype,
            config.t5_gemma_target_length,
        )
        print_mem_info_rank_0("After init t5 gamma")

    def forward(self, eval_input: EvalInput, use_sr_model: bool = False):
        if use_sr_model:
            eval_input = self.sr_data_proxy.process_input(eval_input)
            noise_pred = self.sr_model(*eval_input)
            noise_pred = self.sr_data_proxy.process_output(noise_pred)
        else:
            eval_input = self.data_proxy.process_input(eval_input)
            noise_pred = self.model(*eval_input)
            noise_pred = self.data_proxy.process_output(noise_pred)
        return noise_pred

    @torch.inference_mode()
    def evaluate(
        self,
        prompt: str,
        image: Optional[Image.Image],
        audio_path: Optional[str],
        seconds: int,
        br_width: int,
        br_height: int,
        sr_width: Optional[int],
        sr_height: Optional[int],
        br_num_inference_steps: int,
        sr_num_inference_steps: int,
    ):
        event_path_timer().reset()
        event_path_timer().synced_record("Step1: Prepare Latent Features")
        br_latent_height = br_height // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
        br_latent_width = br_width // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
        br_height = br_latent_height * self.vae_stride[1]
        br_width = br_latent_width * self.vae_stride[2]

        # init latent
        if audio_path is not None:
            latent_audio = load_audio_and_encode(self.audio_vae, audio_path, seconds)
            latent_audio = latent_audio.permute(0, 2, 1)
            num_frames = latent_audio.shape[1]
            is_a2v = True
            print_rank_0(f"Using provided audio, latent_audio: {latent_audio.shape}")
        else:
            num_frames = seconds * self.fps + 1
            latent_audio = torch.randn(1, num_frames, 64, dtype=torch.float32, device="cpu").to(self.device)
            is_a2v = False
            print_rank_0(f"Using random audio, latent_audio: {latent_audio.shape}")
        latent_length = (num_frames - 1) // 4 + 1
        latent_video = torch.randn(
            1, self.z_dim, latent_length, br_latent_height, br_latent_width, dtype=torch.float32, device="cpu"
        ).to(self.device)

        context, original_context_len = get_padded_t5_gemma_embedding(
            prompt, self.txt_model_path, self.device, self.dtype, self.config.t5_gemma_target_length
        )

        event_path_timer().synced_record("Step2: Encode Image for Basic Resolution")
        if image is not None:
            br_image = self.encode_image(image, br_height, br_width)
        else:
            br_image = None
        event_path_timer().synced_record("Step3: Basic Resolution Evaluation")

        if env_is_true("CPU_OFFLOAD") and env_is_true("SR2_1080"):
            self.model = self.model.to(self.device)
        br_latent_video, br_latent_audio = self.evaluate_with_latent(
            context,
            original_context_len,
            br_image,
            latent_video.clone(),
            latent_audio.clone(),
            br_num_inference_steps,
            is_a2v,
            use_sr_model=False,
        )
        if env_is_true("CPU_OFFLOAD") and env_is_true("SR2_1080"):
            self.model = self.model.to(torch.device("cpu"))

        if sr_width is not None and sr_height is not None and self.sr_model is not None:
            event_path_timer().synced_record("Step4: Encode Image for Super Resolution")
            sr_latent_height = sr_height // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
            sr_latent_width = sr_width // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
            sr_height = sr_latent_height * self.vae_stride[1]
            sr_width = sr_latent_width * self.vae_stride[2]
            if image is not None:
                sr_image = self.encode_image(image, sr_height, sr_width)
            else:
                sr_image = None
            latent_video = torch.nn.functional.interpolate(
                br_latent_video, size=(latent_length, sr_latent_height, sr_latent_width), mode="trilinear", align_corners=True
            )
            if self.noise_value != 0:
                noise = torch.randn_like(latent_video, device="cpu").to(latent_video.device)
                sigmas = self.sigmas.to(latent_video.device)
                sigma = sigmas[self.noise_value]
                latent_video = latent_video * sigma + noise * (1 - sigma**2) ** 0.5
            event_path_timer().synced_record("Step5: Super Resolution Evaluation")
            print_mem_info_rank_0("Before super resolution evaluation")
            latent_audio = br_latent_audio.clone()
            br_latent_audio = torch.randn_like(
                br_latent_audio, device="cpu"
            ).to(br_latent_audio.device) * self.config.sr_audio_noise_scale + br_latent_audio * (1 - self.config.sr_audio_noise_scale)

            if env_is_true("CPU_OFFLOAD") and env_is_true("SR2_1080"):
                self.sr_model = self.sr_model.to(self.device)
            latent_video, _ = self.evaluate_with_latent(
                context,
                original_context_len,
                sr_image,
                latent_video.clone(),
                br_latent_audio.clone(),
                sr_num_inference_steps,
                is_a2v,
                use_sr_model=True,
            )
            if env_is_true("CPU_OFFLOAD") and env_is_true("SR2_1080"):
                self.sr_model = self.sr_model.to(torch.device("cpu"))
        else:
            latent_video = br_latent_video
            latent_audio = br_latent_audio

        event_path_timer().synced_record("Step6: Decode Video", print_fn=print_rank_last)
        result = self.post_process(latent_video, latent_audio)
        event_path_timer().synced_record("Step8: Post Process", print_fn=print_rank_last)
        return result

    def schedule(
        self,
        video_scheduler,
        audio_scheduler,
        latent_video,
        latent_audio,
        t,
        idx,
        steps,
        v_cfg_video,
        v_cfg_audio,
        is_a2v,
        cfg_number,
        use_sr_model=False,
    ):
        return schedule_latent_step(
            video_scheduler=video_scheduler,
            audio_scheduler=audio_scheduler,
            latent_video=latent_video,
            latent_audio=latent_audio,
            t=t,
            idx=idx,
            steps=steps,
            v_cfg_video=v_cfg_video,
            v_cfg_audio=v_cfg_audio,
            is_a2v=is_a2v,
            cfg_number=cfg_number,
            use_sr_model=use_sr_model,
            using_sde_flag=self.config.using_sde_flag,
        )

    @torch.inference_mode()
    def evaluate_with_latent(
        self,
        context: torch.Tensor,
        original_context_len: int,
        latent_image: Optional[torch.Tensor],
        latent_video: torch.Tensor,
        latent_audio: torch.Tensor,
        num_inference_steps: int,
        is_a2v: bool = False,
        use_sr_model: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_scheduler = FlowUniPCMultistepScheduler()
        audio_scheduler = FlowUniPCMultistepScheduler()
        video_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=self.shift)
        audio_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=self.shift)
        timesteps = video_scheduler.timesteps

        # a inference trick to aviod over exposure in the I2V evaluation
        latent_length = latent_video.shape[2]
        sr_video_txt_guidance_scale = (
            torch.tensor(self.sr_video_txt_guidance_scale, device=self.device).expand(1, 1, latent_length, 1, 1).clone()
        )
        if self.use_cfg_trick:
            sr_video_txt_guidance_scale[:, :, : self.cfg_trick_start_frame] = min(
                self.cfg_trick_value, self.sr_video_txt_guidance_scale
            )

        # forward
        for idx, t in enumerate(
            tqdm(timesteps, disable=False if not torch.distributed.is_initialized() else torch.distributed.get_rank() != torch.distributed.get_world_size() - 1)
        ):
            if latent_image is not None:
                latent_video[:, :, :1] = latent_image[:, :, :1]
            video_txt_guidance_scale = self.video_txt_guidance_scale if t > 500 else 2.0
            eval_input_cond = EvalInput(
                x_t=latent_video,
                audio_x_t=latent_audio,
                audio_feat_len=[latent_audio.shape[1]],
                txt_feat=context,
                txt_feat_len=[original_context_len],
            )   # txt + audio
            v_output = self.forward(eval_input_cond, use_sr_model=use_sr_model)
            v_cond_video = v_output[0]
            v_cond_audio = v_output[1]

            cfg_number = self.config.sr_cfg_number if use_sr_model else self.config.cfg_number
            if cfg_number == 1:
                v_cfg_video = v_cond_video
                v_cfg_audio = v_cond_audio
            elif cfg_number == 2:
                eval_input_uncond = EvalInput(
                    x_t=latent_video,
                    audio_x_t=latent_audio,
                    audio_feat_len=[latent_audio.shape[1]],
                    txt_feat=self.context_null,
                    txt_feat_len=[self.original_context_null_len],
                )
                v_output_uncond = self.forward(eval_input_uncond, use_sr_model=use_sr_model)
                v_uncond_video = v_output_uncond[0]
                v_uncond_audio = v_output_uncond[1]
                if use_sr_model:
                    v_cfg_video = v_uncond_video + sr_video_txt_guidance_scale * (v_cond_video - v_uncond_video)
                else:
                    v_cfg_video = v_uncond_video + video_txt_guidance_scale * (v_cond_video - v_uncond_video)
                v_cfg_audio = v_uncond_audio + self.audio_txt_guidance_scale * (v_cond_audio - v_uncond_audio)
            else:
                raise ValueError(f"Invalid cfg_number: {cfg_number}")

            latent_video, latent_audio = self.schedule(
                video_scheduler,
                audio_scheduler,
                latent_video,
                latent_audio,
                t,
                idx,
                timesteps,
                v_cfg_video,
                v_cfg_audio,
                is_a2v,
                cfg_number,
                use_sr_model,
            )

        print_rank_0(f"latent_video: {latent_video.shape}, latent_audio: {latent_audio.shape}")
        if latent_image is not None:
            latent_video[:, :, :1] = latent_image[:, :, :1]
        return latent_video, latent_audio

    def encode_image(self, image: Image.Image, height: int, width: int):
        image = load_image(image)
        image = resizecrop(image, height, width)
        image = self.video_processor.preprocess(image, height=height, width=width)
        image = image.to(device=self.device, dtype=self.dtype).unsqueeze(2)
        image = self.vae.encode(image).to(torch.float32)
        return image

    def decode_video(self, latent: torch.Tensor, group: torch.distributed.ProcessGroup = None):
        if self.config.use_turbo_vae:
            is_memory_limited = env_is_true("CPU_OFFLOAD") and env_is_true("SR2_1080")
            videos = self.turbo_vae.decode(latent.to(self.dtype), output_offload=is_memory_limited).float()
        else:
            videos = self.vae.decode(latent.squeeze(0).to(self.dtype), group=group)
        if videos is None:
            return None
        videos.mul_(0.5).add_(0.5).clamp_(0, 1)
        videos = [video.cpu() for video in videos]
        videos = [video.permute(1, 2, 3, 0) * 255 for video in videos]
        videos = [video.numpy().astype(np.uint8) for video in videos]
        return videos

    def post_process(self, latent_video: torch.Tensor, latent_audio: torch.Tensor):
        empty_cache()
        # CTHW -> THWC
        videos_np = self.decode_video(latent_video, group=get_cp_group())
        empty_cache()
        event_path_timer().synced_record("Step7: Decode Audio", print_fn=print_rank_last)

        is_last_rank = True
        if torch.distributed.is_initialized():
            is_last_rank = torch.distributed.get_rank() == torch.distributed.get_world_size() - 1

        if is_last_rank:
            video_np = videos_np[0]

            latent_audio = latent_audio.squeeze(0)
            audio_output = self.audio_vae.decode(latent_audio.T)
            audio_output_np = audio_output.squeeze(0).T.cpu().numpy()
            audio_output_np = resample_audio_sinc(audio_output_np, 441 / 512)

            return video_np, audio_output_np
        else:
            return None, None


