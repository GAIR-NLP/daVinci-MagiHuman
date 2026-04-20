"""
Training dataset for MagiHuman DiT model.

Handles loading pre-encoded latents (video, audio, text) and preparing
them in the format expected by the DiT model's forward pass.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """A single training sample with pre-encoded latents."""
    video_latent: torch.Tensor      # [C, T, H, W] - from video VAE
    audio_latent: torch.Tensor      # [T_audio, C_audio] - from audio VAE
    text_embedding: torch.Tensor    # [seq_len, C_text] - from T5-Gemma
    text_len: int                   # original text embedding length (before padding)


class LatentPreprocessor:
    """Pre-encode raw videos/audio/text into latents offline.

    Usage:
        preprocessor = LatentPreprocessor(config, device)
        preprocessor.process_directory("raw_data/", "latent_data/")
    """

    def __init__(self, config, device="cuda"):
        from inference.common import EvaluationConfig
        from inference.model.vae2_2 import get_vae2_2
        from inference.model.sa_audio import SAAudioFeatureExtractor
        from inference.pipeline.prompt_process import get_padded_t5_gemma_embedding

        self.config = config
        self.device = device
        self.dtype = torch.bfloat16

        # Load VAEs
        vae_path = os.path.join(config.vae_model_path, "Wan2.2_VAE.pth")
        self.video_vae = get_vae2_2(vae_path, device, weight_dtype=self.dtype)
        self.video_vae.eval()

        self.audio_vae = SAAudioFeatureExtractor(device=device, model_path=config.audio_model_path)

        self.txt_model_path = config.txt_model_path
        self.t5_gemma_target_length = config.t5_gemma_target_length
        self._get_text_embedding = get_padded_t5_gemma_embedding

    @torch.no_grad()
    def encode_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Encode video frames to latent space.

        Args:
            video_tensor: [B, C, T, H, W] in [-1, 1] range, float32

        Returns:
            [B, z_dim, T_latent, H_latent, W_latent]
        """
        return self.video_vae.encode(video_tensor.to(self.device, self.dtype))

    @torch.no_grad()
    def encode_audio(self, audio_path: str, seconds: float) -> torch.Tensor:
        """Encode audio file to latent space.

        Returns:
            [1, T_audio, 64]
        """
        from inference.pipeline.video_process import load_audio_and_encode
        latent_audio = load_audio_and_encode(self.audio_vae, audio_path, seconds)
        return latent_audio.permute(0, 2, 1)  # [1, T_audio, 64]

    @torch.no_grad()
    def encode_text(self, prompt: str):
        """Encode text prompt.

        Returns:
            (embedding, original_len) - padded T5-Gemma embedding and its length
        """
        return self._get_text_embedding(
            prompt, self.txt_model_path, self.device, self.dtype,
            self.t5_gemma_target_length,
        )

    def process_sample(
        self,
        video_tensor: torch.Tensor,
        audio_path: str,
        prompt: str,
        seconds: float,
        output_dir: str,
        sample_id: str,
    ):
        """Process a single sample and save latents to disk.

        Args:
            video_tensor: [1, C, T, H, W] normalized to [-1, 1]
            audio_path: Path to audio file
            prompt: Text prompt
            seconds: Duration in seconds
            output_dir: Directory to save latents
            sample_id: Unique identifier for this sample
        """
        os.makedirs(output_dir, exist_ok=True)

        video_latent = self.encode_video(video_tensor).cpu()
        audio_latent = self.encode_audio(audio_path, seconds).cpu()
        text_embedding, text_len = self.encode_text(prompt)
        text_embedding = text_embedding.cpu()

        sample_path = os.path.join(output_dir, f"{sample_id}.pt")
        torch.save({
            "video_latent": video_latent.squeeze(0),  # [C, T, H, W]
            "audio_latent": audio_latent.squeeze(0),   # [T_audio, 64]
            "text_embedding": text_embedding.squeeze(0),  # [seq_len, C_text]
            "text_len": text_len,
        }, sample_path)

        logger.info(f"Saved latents: {sample_path}")

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        manifest_path: str,
    ):
        """Batch process samples from a manifest file.

        Manifest JSON format:
        [
            {
                "id": "sample_001",
                "video": "path/to/video.mp4",
                "audio": "path/to/audio.wav",
                "prompt": "A person speaking...",
                "seconds": 5
            },
            ...
        ]
        """
        from diffusers.video_processor import VideoProcessor
        import torchvision.io as io

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        video_processor = VideoProcessor(vae_scale_factor=16)

        for entry in manifest:
            sample_id = entry["id"]
            output_path = os.path.join(output_dir, f"{sample_id}.pt")
            if os.path.exists(output_path):
                logger.info(f"Skipping {sample_id}, already processed")
                continue

            video_path = os.path.join(input_dir, entry["video"])
            audio_path = os.path.join(input_dir, entry["audio"])

            # Load video: [T, H, W, C] -> [1, C, T, H, W]
            video_frames, _, _ = io.read_video(video_path, pts_unit="sec")
            video_frames = video_frames.permute(3, 0, 1, 2).unsqueeze(0).float()
            video_frames = video_frames / 127.5 - 1.0  # normalize to [-1, 1]

            self.process_sample(
                video_tensor=video_frames,
                audio_path=audio_path,
                prompt=entry["prompt"],
                seconds=entry["seconds"],
                output_dir=output_dir,
                sample_id=sample_id,
            )


class MagiTrainingDataset(Dataset):
    """Dataset that loads pre-encoded latents for DiT training.

    Expected directory structure:
        latent_dir/
            sample_001.pt
            sample_002.pt
            ...

    Each .pt file contains:
        - video_latent: [C, T, H, W]
        - audio_latent: [T_audio, C_audio]
        - text_embedding: [seq_len, C_text]
        - text_len: int
    """

    def __init__(
        self,
        latent_dir: str,
        patch_size: int = 2,
        t_patch_size: int = 1,
        spatial_rope_interpolation: str = "extra",
        ref_audio_offset: int = 1000,
        text_offset: int = 0,
        coords_style: str = "v2",
    ):
        self.latent_dir = latent_dir
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        self.spatial_rope_interpolation = spatial_rope_interpolation
        self.ref_audio_offset = ref_audio_offset
        self.text_offset = text_offset
        self.coords_style = coords_style

        # Discover all .pt files
        self.samples = sorted([
            f for f in os.listdir(latent_dir) if f.endswith(".pt")
        ])
        if not self.samples:
            raise ValueError(f"No .pt files found in {latent_dir}")
        logger.info(f"Found {len(self.samples)} training samples in {latent_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path = os.path.join(self.latent_dir, self.samples[idx])
        data = torch.load(path, map_location="cpu", weights_only=True)

        video_latent = data["video_latent"]    # [C, T, H, W]
        audio_latent = data["audio_latent"]    # [T_audio, C_audio]
        text_embedding = data["text_embedding"]  # [seq_len, C_text]
        text_len = data["text_len"]

        return {
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "text_embedding": text_embedding,
            "text_len": text_len,
        }
