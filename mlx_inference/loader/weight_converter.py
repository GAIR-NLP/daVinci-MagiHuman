"""Load daVinci-MagiHuman safetensors checkpoints into MLX models."""

import gc
import json
import os
from pathlib import Path
from typing import Dict

import mlx.core as mx
import numpy as np


def _flatten_to_nested(flat: Dict[str, mx.array]) -> dict:
    """Convert dot-separated keys to nested dicts.

    ``{"a.b.c": v}`` -> ``{"a": {"b": {"c": v}}}``
    """
    nested: dict = {}
    for key, value in flat.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return nested


def _convert_numeric_dicts_to_lists(d: dict) -> dict | list:
    """Convert ``{"0": {...}, "1": {...}}`` -> ``[{...}, {...}]``."""
    if isinstance(d, dict):
        # Check if all keys are numeric
        keys = list(d.keys())
        if keys and all(k.isdigit() for k in keys):
            max_idx = max(int(k) for k in keys)
            result = [None] * (max_idx + 1)
            for k, v in d.items():
                result[int(k)] = _convert_numeric_dicts_to_lists(v) if isinstance(v, dict) else v
            return result
        else:
            return {k: _convert_numeric_dicts_to_lists(v) if isinstance(v, dict) else v for k, v in d.items()}
    return d


def load_dit_weights(
    checkpoint_dir: str,
    dtype: mx.Dtype = mx.float16,
    verbose: bool = True,
) -> Dict[str, mx.array]:
    """Load sharded safetensors checkpoint, convert to MLX arrays.

    Args:
        checkpoint_dir: Directory containing model-*.safetensors and index.json
        dtype: Target dtype (default float16 for ~28GB)
        verbose: Print progress

    Returns:
        Flat dict of ``{key: mx.array}``
    """
    from safetensors import safe_open

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    total_size = index["metadata"]["total_size"]

    # Group keys by shard
    shards: Dict[str, list] = {}
    for key, shard_file in weight_map.items():
        shards.setdefault(shard_file, []).append(key)

    if verbose:
        print(f"Loading {len(weight_map)} weights from {len(shards)} shards "
              f"({total_size / 1e9:.1f}GB fp32 -> {dtype})")

    flat_weights: Dict[str, mx.array] = {}
    loaded_bytes = 0

    for shard_idx, (shard_file, keys) in enumerate(sorted(shards.items())):
        shard_path = os.path.join(checkpoint_dir, shard_file)
        if verbose:
            print(f"  [{shard_idx + 1}/{len(shards)}] {shard_file} ({len(keys)} keys)")

        with safe_open(shard_path, framework="numpy") as f:
            for key in keys:
                np_array = f.get_tensor(key)
                loaded_bytes += np_array.nbytes

                # Convert to MLX array with target dtype
                mx_array = mx.array(np_array)
                if dtype != mx.float32:
                    mx_array = mx_array.astype(dtype)
                flat_weights[key] = mx_array

        # Free numpy memory between shards
        gc.collect()

    if verbose:
        total_mx_bytes = sum(w.nbytes for w in flat_weights.values())
        print(f"Loaded: {total_mx_bytes / 1e9:.1f}GB in {dtype}")

    return flat_weights


def weights_to_nested(flat_weights: Dict[str, mx.array]) -> dict:
    """Convert flat weight dict to nested structure for ``model.update()``."""
    nested = _flatten_to_nested(flat_weights)
    return _convert_numeric_dicts_to_lists(nested)


def load_and_apply(model, checkpoint_dir: str, dtype: mx.Dtype = mx.float16, verbose: bool = True):
    """Load weights and apply to an MLX model in one step."""
    flat = load_dit_weights(checkpoint_dir, dtype=dtype, verbose=verbose)
    nested = weights_to_nested(flat)
    model.load_weights(list(flat.items()))
    mx.eval(model.parameters())
    if verbose:
        n_params = sum(p.size for _, p in model.parameters() if isinstance(p, mx.array))
        print(f"Model loaded: {n_params / 1e9:.1f}B parameters")
    return model
