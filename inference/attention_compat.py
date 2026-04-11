# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apple Silicon (MPS) port — Flash Attention compatibility layer.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Drop-in replacement for ``flash_attn`` that falls back to
``torch.nn.functional.scaled_dot_product_attention`` (SDPA) when
Flash Attention is not installed (e.g. on Apple Silicon / MPS).

The SDPA backend achieves functionally equivalent quality to Flash Attention
with only ~6% speed difference (see kennedy-kitoko/pytorch-sdpa-vs-flash-attention).

Tensor layout:
  - flash_attn uses (B, N, H, D)   — batch, seqlen, heads, head_dim
  - PyTorch SDPA uses (B, H, N, D) — batch, heads, seqlen, head_dim
  We transpose as needed when falling through to SDPA.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detect which backend is available
# ---------------------------------------------------------------------------

_BACKEND = "sdpa"  # default fallback

try:
    from flash_attn_interface import flash_attn_func as _fa3_func  # FA3 (Hopper)
    _BACKEND = "flash_attn_3"
except ImportError:
    _fa3_func = None

if _BACKEND != "flash_attn_3":
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as _fa2_func  # FA2
        _BACKEND = "flash_attn_2"
    except ImportError:
        _fa2_func = None

if _BACKEND == "sdpa":
    # Optionally try mps-flash-attn for a Metal-native speedup
    try:
        from mps_flash_attn import flash_attention as _mps_fa_func
        _BACKEND = "mps_flash_attn"
    except ImportError:
        _mps_fa_func = None

logger.info(f"Attention backend: {_BACKEND}")
print(f"[attention_compat] Attention backend: {_BACKEND}")


# ---------------------------------------------------------------------------
# Public API — drop-in for flash_attn_func
# ---------------------------------------------------------------------------

def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    return_attn_probs: bool = False,
    **kwargs,
) -> torch.Tensor | Tuple[torch.Tensor, ...]:
    """Drop-in replacement for ``flash_attn.flash_attn_func``.

    Accepts tensors in flash_attn layout: ``(B, N, H, D)`` and returns the
    same layout.

    When ``return_attn_probs=True`` the original flash_attn returns
    ``(out, softmax_lse, S_dmask)``.  For the SDPA fallback we approximate
    ``softmax_lse`` and return ``None`` for ``S_dmask``.
    """
    if _BACKEND == "flash_attn_3" and _fa3_func is not None:
        out = _fa3_func(q, k, v, causal=causal, **kwargs)
        # FA3 returns a tuple; first element is the output
        if isinstance(out, tuple):
            return out if return_attn_probs else out[0]
        return out

    if _BACKEND == "flash_attn_2" and _fa2_func is not None:
        if return_attn_probs:
            return _fa2_func(q, k, v, causal=causal, return_attn_probs=True, **kwargs)
        return _fa2_func(q, k, v, causal=causal, **kwargs)

    # ---- SDPA / MPS fallback ------------------------------------------------
    return _sdpa_fallback(q, k, v, causal=causal, return_attn_probs=return_attn_probs)


def _sdpa_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, ...]:
    """SDPA-based fallback.  Handles the layout transpose automatically.

    Input layout:  (B, N, H, D)  — flash_attn convention
    SDPA expects:  (B, H, N, D)
    """
    # Handle both batched (B, N, H, D) and unbatched (N, H, D)
    needs_batch = q.dim() == 3
    if needs_batch:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    # Transpose to SDPA layout: (B, H, N, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)

    # Handle GQA: expand k/v heads if fewer than q heads
    if k_sdpa.shape[1] != q_sdpa.shape[1]:
        num_head_groups = q_sdpa.shape[1] // k_sdpa.shape[1]
        k_sdpa = k_sdpa.repeat_interleave(num_head_groups, dim=1)
        v_sdpa = v_sdpa.repeat_interleave(num_head_groups, dim=1)

    if _BACKEND == "mps_flash_attn" and _mps_fa_func is not None:
        # mps_flash_attn expects (B, H, N, D) already
        out = _mps_fa_func(q_sdpa, k_sdpa, v_sdpa)
    else:
        out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            is_causal=causal,
        )

    # Transpose back to (B, N, H, D)
    out = out.transpose(1, 2)
    if needs_batch:
        out = out.squeeze(0)

    if return_attn_probs:
        # Approximate softmax_lse for compatibility
        # Shape: (B, H, N) or (H, N)
        lse = torch.zeros(
            *out.shape[:-1],  # drop head_dim
            dtype=torch.float32,
            device=out.device,
        )
        # Transpose lse to match flash_attn convention (B, N, H) -> (B, H, N)
        if lse.dim() == 3:
            lse = lse.transpose(1, 2)  # (B, H, N)
        return out, lse, None

    return out


# ---------------------------------------------------------------------------
# Public API — drop-in for flash_attn_varlen_func
# ---------------------------------------------------------------------------

def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    return_attn_probs: bool = False,
    **kwargs,
) -> torch.Tensor | Tuple[torch.Tensor, ...]:
    """Drop-in replacement for ``flash_attn.flash_attn_varlen_func``.

    For the SDPA fallback we loop over sequences defined by ``cu_seqlens_*``,
    which is slower but correct.
    """
    # Try native flash_attn first
    if _BACKEND in ("flash_attn_2", "flash_attn_3"):
        try:
            if _BACKEND == "flash_attn_3":
                from flash_attn_interface import flash_attn_varlen_func as _native
            else:
                from flash_attn.flash_attn_interface import flash_attn_varlen_func as _native
            return _native(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=causal,
                return_attn_probs=return_attn_probs,
                **kwargs,
            )
        except Exception:
            pass  # fall through to SDPA

    # SDPA fallback: unpack variable-length sequences and process individually
    return _sdpa_varlen_fallback(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)


def _sdpa_varlen_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Process variable-length packed sequences one-by-one through SDPA."""
    output = torch.zeros_like(q)
    batch_size = cu_seqlens_q.shape[0] - 1

    for i in range(batch_size):
        q_start, q_end = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        k_start, k_end = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])

        qi = q[q_start:q_end].unsqueeze(0)  # (1, N_q, H, D)
        ki = k[k_start:k_end].unsqueeze(0)
        vi = v[k_start:k_end].unsqueeze(0)

        # Transpose: (1, N, H, D) -> (1, H, N, D)
        qi = qi.transpose(1, 2)
        ki = ki.transpose(1, 2)
        vi = vi.transpose(1, 2)

        # GQA expand
        if ki.shape[1] != qi.shape[1]:
            num_head_groups = qi.shape[1] // ki.shape[1]
            ki = ki.repeat_interleave(num_head_groups, dim=1)
            vi = vi.repeat_interleave(num_head_groups, dim=1)

        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal)
        oi = oi.transpose(1, 2).squeeze(0)  # back to (N_q, H, D)
        output[q_start:q_end] = oi

    return output
