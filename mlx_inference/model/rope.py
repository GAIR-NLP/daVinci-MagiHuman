"""Rotary Position Embeddings and Fourier frequency generation for MLX."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def freq_bands(
    num_bands: int,
    temperature: float = 10000.0,
    step: int = 2,
) -> mx.array:
    """Compute inverse frequency bands for RoPE.

    Returns: [num_bands / step]
    """
    exp = mx.arange(0, num_bands, step).astype(mx.float32) / num_bands
    return 1.0 / (temperature ** exp)


def rotate_half(x: mx.array) -> mx.array:
    """Rotate half of the last dimension: [x1, x2] -> [-x2, x1]."""
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_emb(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> mx.array:
    """Apply rotary position embedding.

    Args:
        x: (batch, seqlen, nheads, headdim)
        cos, sin: (..., rotary_dim / 2)

    Returns: same shape as x
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]

    # Broadcast cos/sin: (..., d) -> (..., 1, 2*d)
    cos = mx.repeat(cos[..., None, :], repeats=2, axis=-1)  # (..., 1, 2d)
    sin = mx.repeat(sin[..., None, :], repeats=2, axis=-1)

    x_rot = x[..., :ro_dim]
    x_pass = x[..., ro_dim:]

    rotated = x_rot * cos + rotate_half(x_rot) * sin
    return mx.concatenate([rotated, x_pass], axis=-1)


class ElementWiseFourierEmbed(nn.Module):
    """Adaptive spatio-temporal Fourier embedding for RoPE.

    Input: [L, 9] coords (time, row, col, T, H, W, ref_T, ref_H, ref_W)
    Output: [L, dim] -- concatenated sin/cos embeddings
    """

    def __init__(
        self,
        dim: int,
        temperature: float = 10000.0,
        learnable: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

        # Store as regular attribute — MLX will pick it up for weight loading
        self.bands = freq_bands(dim // 8, temperature=temperature, step=1)

    def __call__(self, coords: mx.array) -> mx.array:
        """
        Args:
            coords: [L, 9] - (time, row, col, T, H, W, ref_T, ref_H, ref_W)
        Returns:
            emb: [L, dim] - Fourier embedding (sin + cos concatenated)
        """
        coords_xyz = coords[:, :3]   # [L, 3] - (t, h, w)
        sizes = coords[:, 3:6]       # [L, 3] - (T, H, W)
        refs = coords[:, 6:9]        # [L, 3] - (ref_T, ref_H, ref_W)

        # Adaptive scales for aspect ratio
        scales = (refs - 1) / (sizes - 1)  # [L, 3]
        # Handle edge case: both ref and size are 1
        mask = (refs == 1) & (sizes == 1)
        scales = mx.where(mask, mx.ones_like(scales), scales)

        # Center alignment (spatial only, not temporal)
        centers = (sizes - 1) / 2  # [L, 3]
        centers = mx.concatenate([mx.zeros((centers.shape[0], 1)), centers[:, 1:]], axis=1)
        coords_xyz = coords_xyz - centers

        # Project to frequency bands: [L, 3, B]
        bands = self.bands  # [B]
        proj = coords_xyz[..., None] * scales[..., None] * bands  # [L, 3, B]

        # Sin/cos
        sin_proj = mx.sin(proj)  # [L, 3, B]
        cos_proj = mx.cos(proj)  # [L, 3, B]

        # Concatenate and flatten: [L, 3, B] + [L, 3, B] -> [L, 6, B] -> [L, 6*B]
        emb = mx.concatenate([sin_proj, cos_proj], axis=1)  # [L, 6, B]
        return emb.reshape(emb.shape[0], -1)  # [L, dim]
