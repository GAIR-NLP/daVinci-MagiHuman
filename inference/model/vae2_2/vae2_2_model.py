import gc

import torch

from .vae2_2_module import Wan2_2_VAE


def get_vae2_2(model_path, device="cuda", weight_dtype=torch.float32) -> Wan2_2_VAE:
    # ghitza-patch 2026-11-04: forward `device` into the constructor. Without it, __init__ moved
    # the inner VAE to its default ("cuda" = the full cuda:0 DiT card) and OOM'd ("device not
    # ready") before the outer .to(device) could place it on the aux GPU. Build it on `device`.
    vae = Wan2_2_VAE(vae_pth=model_path, device=device).to(device).to(weight_dtype)
    vae.vae.requires_grad_(False)
    vae.vae.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return vae


__all__ = ["Wan2_2_VAE", "get_vae2_2"]
