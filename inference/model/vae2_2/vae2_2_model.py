import gc

import torch

from inference.device_utils import get_device, empty_cache
from .vae2_2_module import Wan2_2_VAE


def get_vae2_2(model_path, device=None, weight_dtype=torch.float32) -> Wan2_2_VAE:
    device = device or get_device()
    vae = Wan2_2_VAE(vae_pth=model_path).to(device).to(weight_dtype)
    vae.vae.requires_grad_(False)
    vae.vae.eval()
    gc.collect()
    empty_cache()
    return vae


__all__ = ["Wan2_2_VAE", "get_vae2_2"]
