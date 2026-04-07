#!/usr/bin/env python3
"""End-to-end video generation: PyTorch text/VAE + MLX DiT.

Usage:
    python mlx_inference/generate.py \
        --config-load-path example/distill/config_mps.json \
        --prompt "A woman says hello and smiles" \
        --image_path example/assets/image.png \
        --output_path output_mlx_test
"""
import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

import mlx.core as mx
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output_mlx")
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--seconds", type=int, default=2)
    parser.add_argument("--br_width", type=int, default=448)
    parser.add_argument("--br_height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--fp16", action="store_true", help="Use float16 weights (saves memory, same speed)")
    args, _ = parser.parse_known_args()
    return args


def torch_to_mlx(t: torch.Tensor) -> mx.array:
    """Convert PyTorch tensor to MLX array."""
    return mx.array(t.detach().cpu().float().numpy())


def mlx_to_torch(a: mx.array) -> torch.Tensor:
    """Convert MLX array to PyTorch tensor."""
    mx.eval(a)
    return torch.from_numpy(np.array(a, copy=False).copy())


def main():
    args = parse_args()
    print(f"MLX Video Generation")
    print(f"  Prompt: {args.prompt}")
    print(f"  Image: {args.image_path}")
    print(f"  Size: {args.br_width}x{args.br_height}, {args.seconds}s, {args.steps} steps")

    # ====================================================================
    # Step 1: Load PyTorch pipeline (VAE + text encoder) — no DiT needed
    # ====================================================================
    print("\n[1/6] Loading PyTorch VAE + text encoder...")
    t0 = time.time()

    # Force PyTorch to CPU
    import inference.device_utils as du
    du.get_device = lambda force=None: force if force else "cpu"

    from inference.common import parse_config
    from inference.utils import set_random_seed

    config = parse_config()
    set_random_seed(args.seed)

    # Load ONLY VAE + audio — skip the DiT model entirely
    from inference.pipeline.video_generate import MagiEvaluator

    # Pass None as model — MagiEvaluator only stores it, doesn't use it during init
    class DummyModel:
        def eval(self): return self
        def cpu(self): return self
        def to(self, *a, **k): return self
    evaluator = MagiEvaluator(DummyModel(), None, config.evaluation_config)
    print(f"  PyTorch VAE/text loaded in {time.time()-t0:.1f}s")

    # ====================================================================
    # Step 2: Load MLX DiT model
    # ====================================================================
    print("\n[2/6] Loading MLX DiT model...")
    t0 = time.time()

    class ModelConfig:
        num_layers = 40; hidden_size = 5120; head_dim = 128
        num_heads_q = 40; num_heads_kv = 8
        video_in_channels = 192; audio_in_channels = 64; text_in_channels = 3584
        enable_attn_gating = True
        mm_layers = [0, 1, 2, 3, 36, 37, 38, 39]
        local_attn_layers = []; gelu7_layers = [0, 1, 2, 3]; post_norm_layers = []

    from mlx_inference.model.dit_module import DiTModel
    from mlx_inference.loader.weight_converter import load_dit_weights

    mlx_model = DiTModel(ModelConfig())
    # fp32 is same speed as fp16 on Apple Silicon (unified memory) and avoids
    # precision artifacts. Use --fp16 to save memory if needed.
    dit_dtype = mx.float16 if getattr(args, 'fp16', False) else mx.float32
    flat = load_dit_weights("checkpoints/model_repo/distill", dtype=dit_dtype, verbose=True)
    mlx_model.load_weights(list(flat.items()))
    mx.eval(mlx_model.parameters())
    del flat
    import gc; gc.collect()
    print(f"  MLX DiT loaded in {time.time()-t0:.1f}s")

    # ====================================================================
    # Step 3: Encode text + image
    # ====================================================================
    print("\n[3/6] Encoding text and image...")
    t0 = time.time()

    from inference.pipeline.prompt_process import get_padded_t5_gemma_embedding
    context, original_context_len = get_padded_t5_gemma_embedding(
        args.prompt,
        evaluator.txt_model_path,
        "cpu",
        torch.float32,
        config.evaluation_config.t5_gemma_target_length,
    )

    from diffusers.utils import load_image
    from inference.pipeline.video_process import resizecrop
    from diffusers.video_processor import VideoProcessor

    vae_stride = config.evaluation_config.vae_stride
    patch_size = config.evaluation_config.patch_size
    br_latent_height = args.br_height // vae_stride[1] // patch_size[1] * patch_size[1]
    br_latent_width = args.br_width // vae_stride[2] // patch_size[2] * patch_size[2]
    br_height = br_latent_height * vae_stride[1]
    br_width = br_latent_width * vae_stride[2]

    image = load_image(args.image_path)
    image = resizecrop(image, br_height, br_width)
    video_processor = VideoProcessor(vae_scale_factor=16)
    image_tensor = video_processor.preprocess(image, height=br_height, width=br_width)
    image_tensor = image_tensor.to(dtype=torch.float32).unsqueeze(2)
    br_image = evaluator.vae.encode(image_tensor).to(torch.float32)

    fps = config.evaluation_config.fps
    num_frames = args.seconds * fps + 1
    latent_length = (num_frames - 1) // 4 + 1

    torch.manual_seed(args.seed)
    latent_video = torch.randn(1, 48, latent_length, br_latent_height, br_latent_width, dtype=torch.float32)
    latent_audio = torch.randn(1, num_frames, 64, dtype=torch.float32)

    print(f"  Text encoded, image encoded in {time.time()-t0:.1f}s")
    print(f"  latent_video: {latent_video.shape}, latent_audio: {latent_audio.shape}")

    # ====================================================================
    # Step 4: Construct data proxy inputs + run denoising with MLX
    # ====================================================================
    print(f"\n[4/6] Denoising ({args.steps} steps with MLX)...")

    from inference.pipeline.scheduler_unipc import FlowUniPCMultistepScheduler
    from inference.pipeline.data_proxy import MagiDataProxy
    from inference.pipeline.video_generate import EvalInput

    data_proxy = MagiDataProxy(config.evaluation_config.data_proxy_config)
    scheduler = FlowUniPCMultistepScheduler()
    scheduler.set_timesteps(args.steps, device="cpu", shift=config.evaluation_config.shift)
    timesteps = scheduler.timesteps

    # Set first frame to encoded image
    latent_video[:, :, :1] = br_image[:, :, :1]

    t_denoise_start = time.time()
    for idx, t in enumerate(timesteps):
        step_t0 = time.time()

        # Construct eval input
        latent_video[:, :, :1] = br_image[:, :, :1]  # Anchor first frame
        eval_input = EvalInput(
            x_t=latent_video,
            audio_x_t=latent_audio,
            audio_feat_len=[latent_audio.shape[1]],
            txt_feat=context,
            txt_feat_len=[original_context_len],
        )

        # Process input through data proxy (PyTorch)
        processed = data_proxy.process_input(eval_input)
        # processed is a tuple of tensors — the model's actual input args

        # Convert to MLX
        mlx_inputs = []
        for tensor in processed:
            if isinstance(tensor, torch.Tensor):
                mlx_inputs.append(torch_to_mlx(tensor))
            else:
                mlx_inputs.append(tensor)

        # Run MLX DiT forward
        # The model expects (x, coords_mapping, modality_mapping, ...)
        # The data_proxy output format matches the PyTorch DiTModel.__call__ signature
        mlx_out = mlx_model(*mlx_inputs[:3])  # x, coords, modality
        mx.eval(mlx_out)

        # Convert back to PyTorch
        pt_out = mlx_to_torch(mlx_out)

        # Process output through data proxy (PyTorch)
        noise_pred = data_proxy.process_output(pt_out)
        v_video = noise_pred[0]
        v_audio = noise_pred[1]

        # Scheduler step
        from diffusers.utils.torch_utils import randn_tensor
        latent_video = scheduler.step_ddim(v_video, idx, latent_video)
        latent_audio = scheduler.step_ddim(v_audio, idx, latent_audio)

        step_time = time.time() - step_t0
        print(f"  Step {idx+1}/{args.steps}: {step_time:.1f}s")

    # Final re-anchor: overwrite first frame with clean image latent
    latent_video[:, :, :1] = br_image[:, :, :1]

    t_denoise = time.time() - t_denoise_start
    print(f"  Total denoising: {t_denoise:.1f}s ({t_denoise/args.steps:.1f}s/step)")

    # ====================================================================
    # Step 5: Decode video with PyTorch VAE
    # ====================================================================
    print("\n[5/6] Decoding video...")
    t0 = time.time()

    if config.evaluation_config.use_turbo_vae:
        videos = evaluator.turbo_vae.decode(latent_video.to(torch.float32)).float()
    else:
        videos = evaluator.vae.decode(latent_video.squeeze(0).to(torch.float32))

    videos.mul_(0.5).add_(0.5).clamp_(0, 1)
    video_np = videos[0].cpu().permute(1, 2, 3, 0).numpy() * 255
    video_np = video_np.astype(np.uint8)
    print(f"  Video frames: {video_np.shape}")  # Should be (T, H, W, 3)

    # Audio decode
    latent_audio_out = latent_audio.squeeze(0)
    audio_output = evaluator.audio_vae.decode(latent_audio_out.T)
    audio_np = audio_output.squeeze(0).T.cpu().numpy()
    from inference.pipeline.video_process import resample_audio_sinc
    audio_np = resample_audio_sinc(audio_np, 441 / 512)

    print(f"  Decoded in {time.time()-t0:.1f}s")

    # ====================================================================
    # Step 6: Save output
    # ====================================================================
    print("\n[6/6] Saving...")
    import imageio
    import soundfile as sf
    import random

    save_path = f"{args.output_path}_{args.seconds}s_{args.br_width}x{args.br_height}.mp4"
    import uuid
    uid = str(uuid.uuid4())[:8]
    tmp_video = f"tmp_video_{uid}.mp4"
    tmp_audio = f"tmp_audio_{uid}.wav"

    # Save video directly (without audio merge to avoid ffmpeg issues)
    imageio.mimwrite(save_path, video_np, fps=fps)

    print(f"\nDone! Output: {save_path}")
    print(f"Total time: denoising={t_denoise:.0f}s")


if __name__ == "__main__":
    main()
