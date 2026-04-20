"""
Simple training loop for MagiHuman DiT model with LoRA support.

Implements flow-matching diffusion training with the same noise schedule
used during inference (ZeroSNRDDPMDiscretization).
"""

import logging
import math
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference.common import Modality, VarlenHandler
from inference.common.config import MagiPipelineConfig, ModelConfig, EngineConfig
from inference.model.dit.dit_module import DiTModel, FFAHandler
from inference.pipeline.data_proxy import (
    MagiDataProxy,
    SingleData,
    SimplePackedData,
    EvalInput,
)

from .lora import inject_lora, save_lora_weights, load_lora_weights

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for MagiTrainer."""

    # LoRA
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: [
        r"block\.layers\.\d+\.attention\.linear_qkv",
        r"block\.layers\.\d+\.attention\.linear_proj",
    ])

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Training
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "bf16"  # "bf16", "fp32"
    gradient_checkpointing: bool = False

    # Schedule
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"  # "cosine", "constant", "linear"

    # Noise schedule (matches inference ZeroSNRDDPMDiscretization)
    num_train_timesteps: int = 1000
    shift_scale: float = 1.0
    linear_start: float = 0.00085
    linear_end: float = 0.012

    # Logging & checkpointing
    output_dir: str = "checkpoints"
    log_every: int = 10
    save_every: int = 500
    save_total_limit: int = 5

    # Loss weighting
    video_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0

    # Data proxy (mirrors inference config)
    patch_size: int = 2
    t_patch_size: int = 1
    coords_style: str = "v2"


class NoiseSchedule:
    """Flow-matching noise schedule matching ZeroSNRDDPMDiscretization.

    For flow matching, the forward process is:
        x_t = sigma * x_0 + (1 - sigma^2)^0.5 * noise
    where sigma = alphas_cumprod_sqrt, following the zero-SNR schedule.
    """

    def __init__(
        self,
        linear_start: float = 0.00085,
        linear_end: float = 0.012,
        num_timesteps: int = 1000,
        shift_scale: float = 1.0,
    ):
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # SNR shift
        alphas_cumprod = alphas_cumprod / (shift_scale + (1 - shift_scale) * alphas_cumprod)

        # Compute signal and noise rates
        self.alphas_cumprod_sqrt = alphas_cumprod.sqrt().float()  # signal rate
        self.one_minus_alphas_cumprod_sqrt = (1.0 - alphas_cumprod).sqrt().float()  # noise rate
        self.num_timesteps = num_timesteps

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to clean samples at given timesteps.

        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        signal_rate = self.alphas_cumprod_sqrt[timesteps].to(clean.device)
        noise_rate = self.one_minus_alphas_cumprod_sqrt[timesteps].to(clean.device)

        # Reshape for broadcasting
        while signal_rate.dim() < clean.dim():
            signal_rate = signal_rate.unsqueeze(-1)
            noise_rate = noise_rate.unsqueeze(-1)

        return signal_rate * clean + noise_rate * noise

    def get_velocity(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity target for v-prediction.

        v = sqrt(alpha_cumprod_t) * noise - sqrt(1 - alpha_cumprod_t) * x_0
        """
        signal_rate = self.alphas_cumprod_sqrt[timesteps].to(clean.device)
        noise_rate = self.one_minus_alphas_cumprod_sqrt[timesteps].to(clean.device)

        while signal_rate.dim() < clean.dim():
            signal_rate = signal_rate.unsqueeze(-1)
            noise_rate = noise_rate.unsqueeze(-1)

        return signal_rate * noise - noise_rate * clean

    def get_flow_target(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flow-matching target: f = x_0 - noise (direction from noise to data)."""
        return clean - noise


class MagiTrainer:
    """Training loop for MagiHuman DiT with LoRA or full fine-tuning."""

    def __init__(
        self,
        model: DiTModel,
        trainer_config: TrainerConfig,
        model_config: ModelConfig,
    ):
        self.config = trainer_config
        self.model_config = model_config
        self.device = f"cuda:{torch.cuda.current_device()}"
        self.output_dir = Path(trainer_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model setup
        self.model = model

        if trainer_config.use_lora:
            injected = inject_lora(
                self.model,
                trainer_config.lora_target_modules,
                rank=trainer_config.lora_rank,
                alpha=trainer_config.lora_alpha,
                dropout=trainer_config.lora_dropout,
            )
            logger.info(f"LoRA injected into: {injected}")
        else:
            # Full fine-tune: unfreeze everything
            for param in self.model.parameters():
                param.requires_grad = True

        if trainer_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.train()

        # Noise schedule
        self.noise_schedule = NoiseSchedule(
            linear_start=trainer_config.linear_start,
            linear_end=trainer_config.linear_end,
            num_timesteps=trainer_config.num_train_timesteps,
            shift_scale=trainer_config.shift_scale,
        )

        # Data proxy for tokenization (convert latents -> packed token sequence)
        from inference.common.config import DataProxyConfig
        data_proxy_config = DataProxyConfig(
            patch_size=trainer_config.patch_size,
            t_patch_size=trainer_config.t_patch_size,
            coords_style=trainer_config.coords_style,
        )
        self.data_proxy = MagiDataProxy(data_proxy_config)

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=trainer_config.lr,
            weight_decay=trainer_config.weight_decay,
            betas=trainer_config.betas,
            eps=trainer_config.eps,
        )

        # Track state
        self.global_step = 0
        self.best_loss = float("inf")

    def _build_lr_scheduler(self, total_steps: int):
        """Create learning rate scheduler."""
        warmup = self.config.warmup_steps

        if self.config.lr_scheduler == "constant":
            def lr_lambda(step):
                if step < warmup:
                    return step / max(1, warmup)
                return 1.0
        elif self.config.lr_scheduler == "cosine":
            def lr_lambda(step):
                if step < warmup:
                    return step / max(1, warmup)
                progress = (step - warmup) / max(1, total_steps - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        elif self.config.lr_scheduler == "linear":
            def lr_lambda(step):
                if step < warmup:
                    return step / max(1, warmup)
                progress = (step - warmup) / max(1, total_steps - warmup)
                return max(0.0, 1.0 - progress)
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.config.lr_scheduler}")

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _prepare_model_input(
        self,
        noisy_video: torch.Tensor,
        audio_latent: torch.Tensor,
        text_embedding: torch.Tensor,
        text_len: int,
    ):
        """Convert latents into the packed token format expected by DiTModel.forward().

        This mirrors the inference pipeline's data_proxy.process_input() but for training.

        Args:
            noisy_video: [1, C, T, H, W] - noisy video latents
            audio_latent: [1, T_audio, C_audio] - noisy audio latents
            text_embedding: [1, seq_len, C_text] - text condition
            text_len: original text length

        Returns:
            Tuple of (x, coords_mapping, modality_mapping, varlen_handler, local_attn_handler)
        """
        eval_input = EvalInput(
            x_t=noisy_video,
            audio_x_t=audio_latent,
            audio_feat_len=[audio_latent.shape[1]],
            txt_feat=text_embedding,
            txt_feat_len=[text_len],
        )
        return self.data_proxy.process_input(eval_input)

    def _unpack_model_output(self, model_output: torch.Tensor):
        """Convert packed model output back to video and audio tensors.

        Returns:
            (video_pred, audio_pred) - [1, C, T, H, W] and [1, T_audio, C_audio]
        """
        return self.data_proxy.process_output(model_output)

    def training_step(self, batch: dict) -> dict:
        """Execute a single training step.

        Args:
            batch: dict with keys:
                - video_latent: [C, T, H, W]
                - audio_latent: [T_audio, C_audio]
                - text_embedding: [seq_len, C_text]
                - text_len: int

        Returns:
            dict with loss values
        """
        # Add batch dimension
        video_latent = batch["video_latent"].unsqueeze(0).to(self.device)  # [1, C, T, H, W]
        audio_latent = batch["audio_latent"].unsqueeze(0).to(self.device)  # [1, T_audio, C_audio]
        text_embedding = batch["text_embedding"].unsqueeze(0).to(self.device)  # [1, seq_len, C_text]
        text_len = batch["text_len"]
        if isinstance(text_len, torch.Tensor):
            text_len = text_len.item()

        # Sample random timestep
        timestep = torch.randint(0, self.config.num_train_timesteps, (1,), device="cpu")

        # Sample noise for video and audio
        video_noise = torch.randn_like(video_latent)
        audio_noise = torch.randn_like(audio_latent)

        # Forward diffusion: add noise
        noisy_video = self.noise_schedule.add_noise(video_latent, video_noise, timestep)
        noisy_audio = self.noise_schedule.add_noise(audio_latent, audio_noise, timestep)

        # Prepare model input (pack into unified token sequence)
        model_input = self._prepare_model_input(
            noisy_video, noisy_audio, text_embedding, text_len,
        )

        # Forward pass through DiT
        if self.config.mixed_precision == "bf16":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model_output = self.model(*model_input)
        else:
            model_output = self.model(*model_input)

        # Unpack output
        pred_video, pred_audio = self._unpack_model_output(model_output)

        # Compute flow-matching target
        # The model predicts the velocity field: v = x_0 - noise
        # (flow prediction mode matching the FlowUniPCMultistepScheduler)
        video_target = self.noise_schedule.get_flow_target(video_latent, video_noise)
        audio_target = self.noise_schedule.get_flow_target(
            audio_latent.permute(0, 2, 1),  # match pred_audio shape [1, C, T]
            audio_noise.permute(0, 2, 1),
        )

        # MSE loss
        video_loss = F.mse_loss(pred_video.float(), video_target.float())
        audio_loss = F.mse_loss(pred_audio.float(), audio_target.float())

        total_loss = (
            self.config.video_loss_weight * video_loss
            + self.config.audio_loss_weight * audio_loss
        )

        return {
            "loss": total_loss,
            "video_loss": video_loss.detach(),
            "audio_loss": audio_loss.detach(),
        }

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1,
        resume_from: Optional[str] = None,
    ):
        """Main training loop.

        Args:
            dataloader: DataLoader yielding training samples.
            num_epochs: Number of epochs to train.
            resume_from: Path to a checkpoint to resume from.
        """
        if resume_from:
            self._load_checkpoint(resume_from)

        total_steps = len(dataloader) * num_epochs // self.config.gradient_accumulation_steps
        lr_scheduler = self._build_lr_scheduler(total_steps)

        logger.info(f"Starting training for {num_epochs} epochs, {total_steps} optimizer steps")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  LoRA: {self.config.use_lora}")
        logger.info(f"  Learning rate: {self.config.lr}")
        logger.info(f"  Output dir: {self.output_dir}")

        self.model.train()
        accum_loss = 0.0
        accum_video_loss = 0.0
        accum_audio_loss = 0.0

        for epoch in range(num_epochs):
            for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
                losses = self.training_step(batch)
                loss = losses["loss"] / self.config.gradient_accumulation_steps
                loss.backward()

                accum_loss += losses["loss"].detach().item()
                accum_video_loss += losses["video_loss"].item()
                accum_audio_loss += losses["audio_loss"].item()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.config.max_grad_norm,
                        )

                    self.optimizer.step()
                    lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.log_every == 0:
                        avg_loss = accum_loss / self.config.log_every / self.config.gradient_accumulation_steps
                        avg_vloss = accum_video_loss / self.config.log_every / self.config.gradient_accumulation_steps
                        avg_aloss = accum_audio_loss / self.config.log_every / self.config.gradient_accumulation_steps
                        lr = lr_scheduler.get_last_lr()[0]
                        logger.info(
                            f"Step {self.global_step} | "
                            f"Loss: {avg_loss:.5f} | "
                            f"Video: {avg_vloss:.5f} | "
                            f"Audio: {avg_aloss:.5f} | "
                            f"LR: {lr:.2e}"
                        )
                        accum_loss = 0.0
                        accum_video_loss = 0.0
                        accum_audio_loss = 0.0

                    # Save checkpoint
                    if self.global_step % self.config.save_every == 0:
                        self._save_checkpoint()

        # Final save
        self._save_checkpoint()
        logger.info("Training complete!")

    def _save_checkpoint(self):
        """Save training checkpoint."""
        step = self.global_step

        if self.config.use_lora:
            save_path = str(self.output_dir / f"lora_step_{step}.pt")
            save_lora_weights(self.model, save_path)
        else:
            save_path = str(self.output_dir / f"model_step_{step}.pt")
            torch.save(self.model.state_dict(), save_path)

        # Save optimizer state
        opt_path = str(self.output_dir / f"optimizer_step_{step}.pt")
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, opt_path)

        logger.info(f"Checkpoint saved at step {step}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def _load_checkpoint(self, path: str):
        """Resume training from checkpoint."""
        if self.config.use_lora:
            load_lora_weights(self.model, path, device=self.device)
        else:
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)

        # Try loading optimizer state
        opt_path = path.replace("lora_step_", "optimizer_step_").replace("model_step_", "optimizer_step_")
        if os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location=self.device, weights_only=True)
            self.optimizer.load_state_dict(opt_state["optimizer"])
            self.global_step = opt_state["global_step"]
            logger.info(f"Resumed from step {self.global_step}")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        if self.config.save_total_limit <= 0:
            return

        prefix = "lora_step_" if self.config.use_lora else "model_step_"
        checkpoints = sorted(
            [f for f in os.listdir(self.output_dir) if f.startswith(prefix) and f.endswith(".pt")],
            key=lambda f: int(f.replace(prefix, "").replace(".pt", "")),
        )

        while len(checkpoints) > self.config.save_total_limit:
            old = checkpoints.pop(0)
            old_path = self.output_dir / old
            old_path.unlink(missing_ok=True)
            # Also remove corresponding optimizer state
            opt_name = old.replace(prefix, "optimizer_step_")
            (self.output_dir / opt_name).unlink(missing_ok=True)
