# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apple Silicon (MPS) port — device abstraction layer.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Device abstraction utilities for running daVinci-MagiHuman on CUDA, MPS, or CPU.

Follows the pattern established by the VibeVoice Apple Silicon port
(github.com/rcarmo/VibeVoice/commit/04f1116).
"""

import os
from contextlib import contextmanager, nullcontext

import torch

# ---------------------------------------------------------------------------
# Environment setup — must be called before any MPS tensor operations
# ---------------------------------------------------------------------------
# Allow unsupported MPS ops to fall back to CPU transparently.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Let Metal use as much unified memory as it wants (256 GB Mac Studio).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device(force: str | None = None) -> str:
    """Return the best available device string.

    Priority: cuda > mps > cpu.  Pass *force* to override (e.g. ``"cpu"``).
    """
    if force is not None:
        return force
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_mps(device: str | torch.device | None = None) -> bool:
    """Check whether *device* is an MPS device."""
    if device is None:
        device = get_device()
    return str(device).startswith("mps")


def is_cuda(device: str | torch.device | None = None) -> bool:
    if device is None:
        device = get_device()
    return str(device).startswith("cuda")


def is_cpu(device: str | torch.device | None = None) -> bool:
    if device is None:
        device = get_device()
    return str(device) == "cpu"


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

def get_dtype(device: str | torch.device | None = None) -> torch.dtype:
    """Return the recommended compute dtype for *device*.

    - CUDA  -> bfloat16  (native on Ampere+)
    - MPS   -> float32   (float16 causes precision artifacts in diffusion models;
                          256GB unified memory makes fp32 feasible)
    - CPU   -> float32
    """
    device = device or get_device()
    if is_cuda(device):
        return torch.bfloat16
    # MPS: use float32 to avoid color/shape artifacts from fp16 precision loss
    # in multi-step diffusion denoising. With 256GB unified memory this is fine
    # (~60GB model vs ~30GB in fp16).
    return torch.float32


def safe_dtype(dtype: torch.dtype, device: str | torch.device | None = None) -> torch.dtype:
    """Ensure *dtype* is supported on *device*.  Falls back gracefully."""
    device = device or get_device()
    if is_mps(device):
        # MPS: use float32 for best precision; fall back from unsupported types
        if dtype in (torch.bfloat16,):
            return torch.float32
        # float8 types (torch.float8_e4m3fn etc.) are CUDA-only
        dtype_name = str(dtype)
        if "float8" in dtype_name:
            return torch.float16
    return dtype


# ---------------------------------------------------------------------------
# Attention implementation selector
# ---------------------------------------------------------------------------

def get_attention_impl(device: str | torch.device | None = None) -> str:
    """Return the recommended attention implementation name.

    - CUDA  -> ``"flash_attention_2"``
    - MPS   -> ``"sdpa"``
    - CPU   -> ``"sdpa"``
    """
    device = device or get_device()
    if is_cuda(device):
        return "flash_attention_2"
    return "sdpa"


# ---------------------------------------------------------------------------
# Memory / synchronization helpers
# ---------------------------------------------------------------------------

def empty_cache(device: str | torch.device | None = None) -> None:
    device = device or get_device()
    if is_cuda(device):
        torch.cuda.empty_cache()
    elif is_mps(device):
        torch.mps.empty_cache()


def synchronize(device: str | torch.device | None = None) -> None:
    device = device or get_device()
    if is_cuda(device):
        torch.cuda.synchronize()
    elif is_mps(device):
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------

def manual_seed_all(seed: int, device: str | torch.device | None = None) -> None:
    """Seed all relevant RNGs for *device*."""
    torch.manual_seed(seed)
    device = device or get_device()
    if is_cuda(device):
        torch.cuda.manual_seed_all(seed)
    elif is_mps(device):
        # torch.mps.manual_seed is available in recent PyTorch builds
        if hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(seed)


# ---------------------------------------------------------------------------
# Autocast context
# ---------------------------------------------------------------------------

@contextmanager
def autocast_context(device: str | torch.device | None = None, dtype: torch.dtype | None = None):
    """Yield an autocast context appropriate for *device*."""
    device = device or get_device()
    dtype = dtype or get_dtype(device)
    if is_cuda(device):
        with torch.cuda.amp.autocast(dtype=dtype):
            yield
    elif is_mps(device):
        # torch.autocast("mps", ...) was added in PyTorch 2.3+
        try:
            with torch.autocast("mps", dtype=dtype):
                yield
        except Exception:
            yield  # graceful no-op fallback
    else:
        with torch.autocast("cpu", dtype=dtype):
            yield


# ---------------------------------------------------------------------------
# Random tensor generation (MPS-safe)
# ---------------------------------------------------------------------------

def randn_like(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """``torch.randn_like`` that works reliably on MPS.

    MPS has known issues with some random-generation kernels.  The safe
    pattern is to generate on CPU and move to the target device.
    """
    device = tensor.device
    if is_mps(device):
        return torch.randn_like(tensor, device="cpu", **kwargs).to(device)
    return torch.randn_like(tensor, **kwargs)


def randn(*shape, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu") -> torch.Tensor:
    """``torch.randn`` that works reliably on MPS."""
    if is_mps(device):
        return torch.randn(*shape, dtype=dtype, device="cpu").to(device)
    return torch.randn(*shape, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# torch.compile wrapper
# ---------------------------------------------------------------------------

def maybe_compile(model_or_fn, device: str | torch.device | None = None, **kwargs):
    """Apply ``torch.compile`` only when the backend supports it well.

    On MPS we skip compilation (limited support, often errors).
    """
    device = device or get_device()
    if is_mps(device) or is_cpu(device):
        return model_or_fn  # no-op
    return torch.compile(model_or_fn, **kwargs)


# ---------------------------------------------------------------------------
# fork_rng helper
# ---------------------------------------------------------------------------

def fork_rng_devices(device: str | torch.device | None = None) -> list:
    """Return the device list for ``torch.random.fork_rng(devices=...)``."""
    device = device or get_device()
    if is_cuda(device):
        return [torch.cuda.current_device()]
    # MPS and CPU: fork_rng with empty device list works fine
    return []


# ---------------------------------------------------------------------------
# GPU memory info (for logging)
# ---------------------------------------------------------------------------

def get_memory_info(device: str | torch.device | None = None) -> dict:
    """Return a dict of memory stats (in GB) for logging."""
    device = device or get_device()
    if is_cuda(device):
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        }
    if is_mps(device):
        # MPS memory reporting is limited; return what we can
        try:
            allocated = torch.mps.current_allocated_memory()
            return {
                "allocated_gb": round(allocated / 1024**3, 2),
                "max_allocated_gb": 0,
                "reserved_gb": 0,
                "max_reserved_gb": 0,
            }
        except Exception:
            pass
    return {"allocated_gb": 0, "max_allocated_gb": 0, "reserved_gb": 0, "max_reserved_gb": 0}


# ---------------------------------------------------------------------------
# Architecture detection (replaces is_hopper_arch / get_arch_memory)
# ---------------------------------------------------------------------------

def is_hopper_arch() -> bool:
    """True only on CUDA Hopper (SM 9.x) GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()[0] == 9
    return False


def get_arch_memory(unit: str = "GB") -> float:
    """Return total device memory in *unit*.  Works on CUDA and MPS."""
    if torch.cuda.is_available():
        total_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    elif is_mps():
        # Apple Silicon unified memory — os.sysconf is the best proxy
        import platform
        if platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            total_bytes = int(result.stdout.strip())
        else:
            total_bytes = 0
    else:
        total_bytes = 0

    divisors = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    if unit not in divisors:
        raise ValueError(f"Invalid unit: {unit}")
    return float(total_bytes) / divisors[unit]


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def get_device_count() -> int:
    """Number of accelerators visible to the process."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if is_mps():
        return 1  # Apple Silicon has one unified GPU
    return 0


def set_device(device_idx: int) -> None:
    """Set the current device index (only meaningful on CUDA)."""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_idx)
    # MPS: no-op (single device)
