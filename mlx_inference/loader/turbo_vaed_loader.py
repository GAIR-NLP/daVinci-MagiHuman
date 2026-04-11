"""Load PyTorch Turbo VAE checkpoint into MLX TurboVAED model.

Handles:
- Extracting ema_state_dict from .ckpt
- Stripping 'module.' prefix
- Transposing Conv3d/Conv2d weights from PyTorch NCDHW -> MLX NDHWC layout
- Key remapping from PyTorch module names to MLX module names
"""

import json
import gc

import mlx.core as mx
import numpy as np
import torch


def load_turbo_vaed_weights(
    config_path: str,
    ckpt_path: str,
    dtype: mx.Dtype = mx.float32,
    verbose: bool = True,
):
    """Load Turbo VAE decoder weights, returning flat dict + config + mean/std.

    Returns:
        (flat_weights, config, mean, inv_std)
    """
    with open(config_path) as f:
        config = json.load(f)

    if verbose:
        print(f"  Loading Turbo VAE checkpoint from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "ema_state_dict" in ckpt, "ckpt must contain ema_state_dict"

    state_dict = ckpt["ema_state_dict"]
    del ckpt
    gc.collect()

    # Strip 'module.' prefix
    cleaned = {}
    for key, value in state_dict.items():
        k = key[7:] if key.startswith("module.") else key
        cleaned[k] = value
    state_dict = cleaned

    # Check if keys have 'decoder.' prefix or not
    sample_key = next(iter(state_dict.keys()))
    if not sample_key.startswith("decoder."):
        # Weights are for decoder only, add prefix
        state_dict = {f"decoder.{k}": v for k, v in state_dict.items()}

    # Extract mean/std before processing
    # These are stored on the TurboVAED class, not in the checkpoint
    # We compute them from the hardcoded values
    mean_values = [
        -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
        -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
        -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
        -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
        -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
        0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
    ]
    std_values = [
        0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
        0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
        0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
        0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
        0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
        0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
    ]
    mean = mx.array(mean_values, dtype=dtype)
    inv_std = mx.array([1.0 / s for s in std_values], dtype=dtype)

    # Convert weights with key remapping and transposition
    flat_weights = {}
    skipped = 0
    for key, tensor in state_dict.items():
        # Skip non-decoder weights (aligned_feature_projection_heads, etc.)
        if not key.startswith("decoder."):
            skipped += 1
            continue

        np_arr = tensor.detach().cpu().float().numpy()

        # Remap PyTorch key structure to MLX structure:
        # PyTorch: upsamplers.0.resample.1.{weight,bias} (Conv2d in Sequential[Upsample, Conv2d])
        # MLX:     upsamplers.0.spatial_conv.{weight,bias}
        mlx_key = key.replace(".resample.1.", ".spatial_conv.")

        # Transpose convolution weights from PyTorch -> MLX layout
        if "weight" in mlx_key and "norm" not in mlx_key:
            if np_arr.ndim == 5:
                # Conv3d: [O, I, D, H, W] -> [O, D, H, W, I]
                np_arr = np_arr.transpose(0, 2, 3, 4, 1)
            elif np_arr.ndim == 4:
                # Conv2d: [O, I, H, W] -> [O, H, W, I]
                np_arr = np_arr.transpose(0, 2, 3, 1)

        mx_arr = mx.array(np_arr)
        if dtype != mx.float32:
            mx_arr = mx_arr.astype(dtype)
        flat_weights[mlx_key] = mx_arr

    del state_dict
    gc.collect()

    if skipped and verbose:
        print(f"  Skipped {skipped} non-decoder weights")

    if verbose:
        total_bytes = sum(w.nbytes for w in flat_weights.values())
        print(f"  Loaded {len(flat_weights)} weights ({total_bytes / 1e9:.1f}GB {dtype})")

    return flat_weights, config, mean, inv_std
