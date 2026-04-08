#!/usr/bin/env python3
"""End-to-end video generation: PyTorch text/VAE + MLX DiT.

Memory-optimized: loads each model sequentially and frees it before the next.

Usage:
    python mlx_inference/generate.py \
        --config-load-path example/distill/config_mps.json \
        --prompt "A woman says hello and smiles" \
        --image_path example/assets/image.png \
        --output_path output_mlx_test
"""
import argparse
import gc
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
    parser.add_argument("--sr_width", type=int, default=None, help="Super-resolution width (e.g. 960 for 540p)")
    parser.add_argument("--sr_height", type=int, default=None, help="Super-resolution height (e.g. 544 for 540p)")
    parser.add_argument("--sr_steps", type=int, default=5, help="SR denoising steps")
    parser.add_argument("--sr_checkpoint", type=str, default="checkpoints/model_repo/540p_sr", help="SR model checkpoint dir")
    parser.add_argument("--fp16", action="store_true", help="Use float16 DiT weights (saves ~30GB)")
    args, _ = parser.parse_known_args()
    return args


def torch_to_mlx(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().cpu().float().numpy())


def mlx_to_torch(a: mx.array) -> torch.Tensor:
    mx.eval(a)
    return torch.from_numpy(np.array(a, copy=False).copy())


def free_memory():
    """Aggressively free all unused memory."""
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    try:
        mx.clear_cache()
    except:
        pass


def main():
    args = parse_args()
    total_t0 = time.time()
    # Force unbuffered output
    import builtins
    _print = builtins.print
    def print(*a, **kw):
        kw.setdefault('flush', True)
        _print(*a, **kw)

    print(f"MLX Video Generation")
    print(f"  Prompt: {args.prompt}")
    print(f"  Image: {args.image_path}")
    print(f"  Size: {args.br_width}x{args.br_height}, {args.seconds}s, {args.steps} steps")

    import inference.device_utils as du
    du.get_device = lambda force=None: force if force else "cpu"

    from inference.common import parse_config
    from inference.utils import set_random_seed

    config = parse_config()
    set_random_seed(args.seed)

    vae_stride = config.evaluation_config.vae_stride
    patch_size = config.evaluation_config.patch_size
    br_latent_height = args.br_height // vae_stride[1] // patch_size[1] * patch_size[1]
    br_latent_width = args.br_width // vae_stride[2] // patch_size[2] * patch_size[2]
    br_height = br_latent_height * vae_stride[1]
    br_width = br_latent_width * vae_stride[2]
    fps = config.evaluation_config.fps
    num_frames = args.seconds * fps + 1
    latent_length = (num_frames - 1) // 4 + 1

    # ====================================================================
    # Step 1: Encode text in a SUBPROCESS (guarantees 38GB is freed after)
    # ====================================================================
    print("\n[1/6] Encoding text...")
    t0 = time.time()

    # Run text encoding in subprocess to guarantee memory cleanup
    import subprocess as sp
    import tempfile
    embed_path = os.path.join(tempfile.gettempdir(), f"_magi_text_embed_{os.getpid()}.npy")
    encode_script = f"""
import sys, os, torch, numpy as np
sys.argv = {sys.argv}
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['WORLD_SIZE'] = '1'; os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'; os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '29500'
import inference.device_utils as du
du.get_device = lambda force=None: force if force else 'cpu'
from inference.pipeline.prompt_process import get_padded_t5_gemma_embedding
ctx, clen = get_padded_t5_gemma_embedding(
    {repr(args.prompt)},
    {repr(config.evaluation_config.txt_model_path)},
    'cpu', torch.float32,
    {config.evaluation_config.t5_gemma_target_length},
)
np.savez({repr(embed_path)}, context=ctx.numpy(), context_len=clen)
print(f'DONE shape={{ctx.shape}} len={{clen}}')
"""
    result = sp.run(
        [sys.executable, "-c", encode_script],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    if result.returncode != 0:
        print(f"  Text encoding failed (rc={result.returncode}):")
        print(f"  stdout: {result.stdout[-300:]}")
        print(f"  stderr: {result.stderr[-300:]}")
        sys.exit(1)

    # Load the embedding from disk (tiny: ~9MB)
    npz_path = embed_path if embed_path.endswith(".npz") else embed_path + ".npz"
    if not os.path.exists(npz_path):
        print(f"  Error: embedding file not found at {npz_path}")
        print(f"  Subprocess stdout: {result.stdout[-200:]}")
        sys.exit(1)
    data = np.load(npz_path)
    context = torch.from_numpy(data["context"])
    original_context_len = int(data["context_len"])
    os.remove(npz_path)
    print(f"  Text encoded in {time.time()-t0:.1f}s, shape={context.shape} (subprocess, 0 residual memory)", flush=True)

    # ====================================================================
    # Step 2: Encode image (load VAE encoder, encode, then FREE it)
    # ====================================================================
    print("\n[2/6] Encoding image...")
    t0 = time.time()

    from inference.model.vae2_2.vae2_2_model import get_vae2_2
    vae_path = os.path.join(config.evaluation_config.vae_model_path, "Wan2.2_VAE.pth")
    vae_encode = get_vae2_2(vae_path, device="cpu")

    from diffusers.utils import load_image
    from inference.pipeline.video_process import resizecrop
    from diffusers.video_processor import VideoProcessor

    image = load_image(args.image_path)
    image = resizecrop(image, br_height, br_width)
    video_processor = VideoProcessor(vae_scale_factor=16)
    image_tensor = video_processor.preprocess(image, height=br_height, width=br_width)
    image_tensor = image_tensor.to(dtype=torch.float32).unsqueeze(2)
    br_image = vae_encode.encode(image_tensor).to(torch.float32)

    # Free the VAE encoder (~2.6GB)
    del vae_encode
    free_memory()
    print(f"  Image encoded in {time.time()-t0:.1f}s, VAE freed")

    # Create initial noise
    torch.manual_seed(args.seed)
    latent_video = torch.randn(1, 48, latent_length, br_latent_height, br_latent_width, dtype=torch.float32)
    latent_audio = torch.randn(1, num_frames, 64, dtype=torch.float32)
    latent_video[:, :, :1] = br_image[:, :, :1]
    print(f"  latent_video: {latent_video.shape}, latent_audio: {latent_audio.shape}")

    # ====================================================================
    # Step 3: Load MLX DiT model (main memory consumer: ~30-61GB)
    # ====================================================================
    print(f"\n[3/6] Loading MLX DiT model...")
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

    # Auto-select dtype: fp16 for large resolutions to save ~30GB memory
    total_pixels = br_width * br_height * latent_length
    use_fp16 = args.fp16 or (total_pixels > 256 * 448 * 13 * 2)  # > 2x baseline
    dit_dtype = mx.float16 if use_fp16 else mx.float32
    if use_fp16 and not args.fp16:
        print(f"  Auto-selecting fp16 for {br_width}x{br_height} (saves ~30GB memory)")
    mlx_model = DiTModel(ModelConfig())
    flat = load_dit_weights("checkpoints/model_repo/distill", dtype=dit_dtype, verbose=True)
    mlx_model.load_weights(list(flat.items()))
    mx.eval(mlx_model.parameters())
    del flat; free_memory()
    print(f"  MLX DiT loaded in {time.time()-t0:.1f}s ({dit_dtype})")

    # ====================================================================
    # Step 4: Denoising loop (MLX DiT)
    # ====================================================================
    print(f"\n[4/6] Denoising ({args.steps} steps)...")

    from inference.pipeline.scheduler_unipc import FlowUniPCMultistepScheduler
    from inference.pipeline.data_proxy import MagiDataProxy
    from inference.pipeline.video_generate import EvalInput

    data_proxy = MagiDataProxy(config.evaluation_config.data_proxy_config)
    scheduler = FlowUniPCMultistepScheduler()
    scheduler.set_timesteps(args.steps, device="cpu", shift=config.evaluation_config.shift)
    timesteps = scheduler.timesteps

    t_denoise_start = time.time()
    for idx, t in enumerate(timesteps):
        step_t0 = time.time()

        latent_video[:, :, :1] = br_image[:, :, :1]
        eval_input = EvalInput(
            x_t=latent_video, audio_x_t=latent_audio,
            audio_feat_len=[latent_audio.shape[1]],
            txt_feat=context, txt_feat_len=[original_context_len],
        )

        processed = data_proxy.process_input(eval_input)
        mlx_inputs = [torch_to_mlx(t) if isinstance(t, torch.Tensor) else t for t in processed]

        mlx_out = mlx_model(*mlx_inputs[:3])
        mx.eval(mlx_out)
        pt_out = mlx_to_torch(mlx_out)

        noise_pred = data_proxy.process_output(pt_out)
        latent_video = scheduler.step_ddim(noise_pred[0], idx, latent_video)
        latent_audio = scheduler.step_ddim(noise_pred[1], idx, latent_audio)

        print(f"  Step {idx+1}/{args.steps}: {time.time()-step_t0:.1f}s")

    latent_video[:, :, :1] = br_image[:, :, :1]
    t_denoise = time.time() - t_denoise_start
    print(f"  Total denoising: {t_denoise:.1f}s ({t_denoise/args.steps:.1f}s/step)")

    # Free base MLX DiT model before SR or VAE
    del mlx_model; free_memory()
    print("  Base DiT freed from memory")

    # Save base resolution results
    br_latent_video = latent_video
    br_latent_audio = latent_audio

    # ====================================================================
    # Step 5: Super-Resolution (optional)
    # ====================================================================
    if args.sr_width and args.sr_height and os.path.exists(args.sr_checkpoint):
        print(f"\n[5/7] Super-Resolution ({args.sr_width}x{args.sr_height}, {args.sr_steps} steps, CFG=3.5)...")
        t_sr_start = time.time()

        # Compute SR latent dimensions
        sr_latent_height = args.sr_height // vae_stride[1] // patch_size[1] * patch_size[1]
        sr_latent_width = args.sr_width // vae_stride[2] // patch_size[2] * patch_size[2]
        sr_height = sr_latent_height * vae_stride[1]
        sr_width = sr_latent_width * vae_stride[2]
        print(f"  SR latent: {latent_length}x{sr_latent_height}x{sr_latent_width} -> {sr_width}x{sr_height}")

        # Trilinear interpolate base latent to SR resolution
        latent_video = torch.nn.functional.interpolate(
            br_latent_video,
            size=(latent_length, sr_latent_height, sr_latent_width),
            mode="trilinear", align_corners=True,
        )

        # Add noise (noise_value=220 from config)
        noise_value = config.evaluation_config.noise_value  # 220
        from inference.pipeline.video_generate import ZeroSNRDDPMDiscretization
        sigmas = ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)
        if noise_value != 0:
            noise = torch.randn_like(latent_video, device="cpu")
            sigma = sigmas[noise_value]
            latent_video = latent_video * sigma + noise * (1 - sigma**2) ** 0.5

        # Encode reference image at SR resolution
        vae_sr = get_vae2_2(vae_path, device="cpu")
        sr_image = load_image(args.image_path)
        sr_image = resizecrop(sr_image, sr_height, sr_width)
        sr_img_t = video_processor.preprocess(sr_image, height=sr_height, width=sr_width)
        sr_img_t = sr_img_t.to(dtype=torch.float32).unsqueeze(2)
        sr_br_image = vae_sr.encode(sr_img_t).to(torch.float32)
        del vae_sr; free_memory()

        # Audio for SR: save clean audio for final decode, create noisy for SR model
        clean_latent_audio = br_latent_audio.clone()
        sr_audio_noise_scale = config.evaluation_config.sr_audio_noise_scale
        sr_latent_audio = torch.randn_like(br_latent_audio, device="cpu") * sr_audio_noise_scale + br_latent_audio * (1 - sr_audio_noise_scale)
        latent_audio = sr_latent_audio

        # Encode negative prompt for CFG (in subprocess to save memory)
        print("  Encoding negative prompt for CFG...")
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, low quality, worst quality, poor quality, noise, background noise, hiss, hum, buzz, crackle, static, compression artifacts"
        neg_embed_path = os.path.join(tempfile.gettempdir(), f"_magi_neg_embed_{os.getpid()}.npz")
        neg_script = f"""
import sys, os, torch, numpy as np
sys.argv = {sys.argv}
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['WORLD_SIZE'] = '1'; os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'; os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '29500'
import inference.device_utils as du
du.get_device = lambda force=None: force if force else 'cpu'
from inference.pipeline.prompt_process import get_padded_t5_gemma_embedding
ctx, clen = get_padded_t5_gemma_embedding(
    {repr(negative_prompt)},
    {repr(config.evaluation_config.txt_model_path)},
    'cpu', torch.float32,
    {config.evaluation_config.t5_gemma_target_length},
)
np.savez({repr(neg_embed_path)}, context=ctx.numpy(), context_len=clen)
"""
        result = sp.run([sys.executable, "-c", neg_script],
                       capture_output=True, text=True, timeout=300,
                       env={**os.environ, "PYTHONPATH": os.getcwd()})
        if result.returncode != 0:
            print(f"  Negative prompt encoding failed, skipping CFG")
            context_null = context  # fallback: same as positive
            context_null_len = original_context_len
        else:
            neg_data = np.load(neg_embed_path)
            context_null = torch.from_numpy(neg_data["context"])
            context_null_len = int(neg_data["context_len"])
            os.remove(neg_embed_path)
        print(f"  Negative prompt encoded")

        # Load SR MLX DiT model (fp32 for CFG precision — subtraction needs it)
        print(f"  Loading SR model from {args.sr_checkpoint} (fp32 for CFG precision)...")
        sr_mlx_model = DiTModel(ModelConfig())
        sr_flat = load_dit_weights(args.sr_checkpoint, dtype=mx.float32, verbose=False)
        sr_mlx_model.load_weights(list(sr_flat.items()))
        mx.eval(sr_mlx_model.parameters())
        del sr_flat; free_memory()
        print(f"  SR model loaded")

        # SR denoising loop with classifier-free guidance
        import copy
        sr_data_proxy_config = copy.deepcopy(config.evaluation_config.data_proxy_config)
        sr_data_proxy_config.coords_style = "v1"
        sr_data_proxy = MagiDataProxy(sr_data_proxy_config)
        sr_data_proxy_uncond = MagiDataProxy(sr_data_proxy_config)

        sr_scheduler = FlowUniPCMultistepScheduler()
        sr_scheduler.set_timesteps(args.sr_steps, device="cpu", shift=config.evaluation_config.shift)
        sr_timesteps = sr_scheduler.timesteps

        # CFG parameters
        sr_guidance_scale = config.evaluation_config.sr_video_txt_guidance_scale  # 3.5
        audio_guidance_scale = config.evaluation_config.audio_txt_guidance_scale  # 5.0
        use_cfg_trick = config.evaluation_config.use_cfg_trick
        cfg_trick_start_frame = config.evaluation_config.cfg_trick_start_frame  # 13
        cfg_trick_value = config.evaluation_config.cfg_trick_value  # 2.0

        # Build per-frame guidance scale tensor with cfg_trick
        guidance_tensor = torch.full((1, 1, latent_length, 1, 1), sr_guidance_scale)
        if use_cfg_trick:
            guidance_tensor[:, :, :cfg_trick_start_frame] = min(cfg_trick_value, sr_guidance_scale)

        for idx, t in enumerate(sr_timesteps):
            step_t0 = time.time()
            if sr_br_image is not None:
                latent_video[:, :, :1] = sr_br_image[:, :, :1]

            # === Conditioned forward (real prompt) ===
            eval_input_cond = EvalInput(
                x_t=latent_video, audio_x_t=latent_audio,
                audio_feat_len=[latent_audio.shape[1]],
                txt_feat=context, txt_feat_len=[original_context_len],
            )
            proc_cond = sr_data_proxy.process_input(eval_input_cond)
            mlx_cond = [torch_to_mlx(t_) if isinstance(t_, torch.Tensor) else t_ for t_ in proc_cond]
            out_cond = sr_mlx_model(*mlx_cond[:3])
            mx.eval(out_cond)
            pt_cond = mlx_to_torch(out_cond)
            pred_cond = sr_data_proxy.process_output(pt_cond)
            v_cond_video = pred_cond[0]
            v_cond_audio = pred_cond[1]

            # === Unconditioned forward (negative prompt) ===
            eval_input_uncond = EvalInput(
                x_t=latent_video, audio_x_t=latent_audio,
                audio_feat_len=[latent_audio.shape[1]],
                txt_feat=context_null, txt_feat_len=[context_null_len],
            )
            proc_uncond = sr_data_proxy_uncond.process_input(eval_input_uncond)
            mlx_uncond = [torch_to_mlx(t_) if isinstance(t_, torch.Tensor) else t_ for t_ in proc_uncond]
            out_uncond = sr_mlx_model(*mlx_uncond[:3])
            mx.eval(out_uncond)
            pt_uncond = mlx_to_torch(out_uncond)
            pred_uncond = sr_data_proxy_uncond.process_output(pt_uncond)
            v_uncond_video = pred_uncond[0]
            v_uncond_audio = pred_uncond[1]

            # === CFG interpolation (SR always uses fixed guidance, no t>500 check) ===
            v_cfg_video = v_uncond_video + guidance_tensor * (v_cond_video - v_uncond_video)
            v_cfg_audio = v_uncond_audio + audio_guidance_scale * (v_cond_audio - v_uncond_audio)

            # === Scheduler step (video only — SR doesn't update audio) ===
            latent_video = sr_scheduler.step(v_cfg_video, t, latent_video, return_dict=False)[0]

            print(f"  SR Step {idx+1}/{args.sr_steps}: {time.time()-step_t0:.1f}s (CFG)")

        if sr_br_image is not None:
            latent_video[:, :, :1] = sr_br_image[:, :, :1]

        # Use clean base audio for final decode
        latent_audio = clean_latent_audio

        del sr_mlx_model; free_memory()
        t_sr = time.time() - t_sr_start
        print(f"  SR total: {t_sr:.1f}s")

        # Update dimensions for VAE decode
        br_width = sr_width
        br_height = sr_height
        n_steps_label = f"{args.steps}+{args.sr_steps}"
    else:
        if args.sr_width and args.sr_height:
            print(f"\n  SR skipped (checkpoint not found at {args.sr_checkpoint})")
        n_steps_label = str(args.steps)

    # ====================================================================
    # Step 6: Decode video (Wan2.2 VAE)
    # ====================================================================
    decode_step = "6/7" if args.sr_width else "5/6"
    print(f"\n[{decode_step}] Decoding video (Wan2.2 VAE)...")
    t0 = time.time()

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    vae_pixels = br_width * br_height
    vae_device = "cpu" if vae_pixels > 480 * 640 else ("mps" if mps_available else "cpu")
    if vae_device == "cpu" and mps_available:
        print(f"  Using CPU for VAE decode ({br_width}x{br_height} too large for MPS)")
    vae_decode = get_vae2_2(vae_path, device=vae_device)
    videos = vae_decode.decode(latent_video.to(vae_device, dtype=torch.float32))
    videos = videos.float().cpu()
    del vae_decode; free_memory()

    videos.mul_(0.5).add_(0.5).clamp_(0, 1)
    video_np = videos[0].permute(1, 2, 3, 0).numpy() * 255
    video_np = video_np.astype(np.uint8)
    del videos; free_memory()
    print(f"  Video frames: {video_np.shape}, decoded in {time.time()-t0:.1f}s")

    # Audio decode (load Stable Audio VAE, decode, free)
    from inference.model.sa_audio import SAAudioFeatureExtractor
    audio_vae = SAAudioFeatureExtractor(device="cpu", model_path=config.evaluation_config.audio_model_path)
    audio_sample_rate = audio_vae.sample_rate
    latent_audio_out = latent_audio.squeeze(0)
    audio_output = audio_vae.decode(latent_audio_out.T)
    audio_np = audio_output.squeeze(0).T.cpu().numpy()
    from inference.pipeline.video_process import resample_audio_sinc
    audio_np = resample_audio_sinc(audio_np, 441 / 512)
    del audio_vae; free_memory()

    # ====================================================================
    # Step 6: Save output with audio
    # ====================================================================
    print("\n[6/6] Saving with audio...")
    import imageio
    import soundfile as sf
    import subprocess
    import shutil
    import uuid

    if args.sr_width and args.sr_height:
        save_path = f"{args.output_path}_{args.seconds}s_{br_width}x{br_height}_sr.mp4"
    else:
        save_path = f"{args.output_path}_{args.seconds}s_{args.br_width}x{args.br_height}.mp4"
    uid = str(uuid.uuid4())[:8]
    tmp_video = os.path.join(os.getcwd(), f"_tmp_vid_{uid}.mp4")
    tmp_audio = os.path.join(os.getcwd(), f"_tmp_aud_{uid}.wav")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        for path in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
            if os.path.exists(path):
                ffmpeg = path
                break

    imageio.mimwrite(tmp_video, video_np, fps=fps, quality=8,
                     output_params=["-loglevel", "error"] if ffmpeg else [])
    sf.write(tmp_audio, audio_np, audio_sample_rate)

    if ffmpeg and os.path.exists(tmp_video) and os.path.exists(tmp_audio):
        cmd = [ffmpeg, "-y", "-i", tmp_video, "-i", tmp_audio,
               "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "copy", "-c:a", "aac", "-shortest",
               save_path, "-loglevel", "error"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Merged video + audio")
        else:
            print(f"  ffmpeg failed: {result.stderr}")
            os.rename(tmp_video, save_path)
        for f in [tmp_video, tmp_audio]:
            if os.path.exists(f):
                os.remove(f)
    else:
        if os.path.exists(tmp_video):
            os.rename(tmp_video, save_path)
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)
        print(f"  Saved video without audio")

    total_time = time.time() - total_t0
    print(f"\nDone! Output: {save_path}")
    print(f"  Denoising: {t_denoise:.0f}s, Total: {total_time:.0f}s")


if __name__ == "__main__":
    main()
