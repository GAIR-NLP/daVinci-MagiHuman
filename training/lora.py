"""
LoRA (Low-Rank Adaptation) implementation for MagiHuman DiT model.

Supports both BaseLinear (single-expert) and NativeMoELinear (multi-expert)
layers used by the model's Attention and MLP modules.
"""

import logging
import re
from typing import Optional

import torch
import torch.nn as nn

from inference.model.dit.dit_module import BaseLinear, NativeMoELinear, ModalityDispatcher

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """LoRA adapter that wraps a BaseLinear or NativeMoELinear layer.

    For single-expert (BaseLinear) layers, a single LoRA pair (A, B) is used.
    For multi-expert (NativeMoELinear) layers, per-expert LoRA pairs are used
    so that modality-specific adaptations are learned independently.
    """

    def __init__(
        self,
        base_layer: BaseLinear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        num_experts = base_layer.num_experts

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        if num_experts == 1:
            # Single LoRA pair for BaseLinear
            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)
        else:
            # Per-expert LoRA pairs for NativeMoELinear
            self.lora_A = nn.ModuleList([
                nn.Linear(in_features, rank, bias=False) for _ in range(num_experts)
            ])
            self.lora_B = nn.ModuleList([
                nn.Linear(rank, out_features, bias=False) for _ in range(num_experts)
            ])
            for i in range(num_experts):
                nn.init.kaiming_uniform_(self.lora_A[i].weight, a=5**0.5)
                nn.init.zeros_(self.lora_B[i].weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        input: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        modality_dispatcher: Optional[ModalityDispatcher] = None,
    ) -> torch.Tensor:
        # Base layer forward (handles both BaseLinear and NativeMoELinear)
        base_out = self.base_layer(input, output_dtype=output_dtype, modality_dispatcher=modality_dispatcher)

        if self.num_experts == 1:
            # Single LoRA path
            lora_input = self.dropout(input.to(torch.float32))
            lora_out = self.lora_B(self.lora_A(lora_input))
            return base_out + lora_out.to(base_out.dtype) * self.scaling
        else:
            # Per-expert LoRA path
            if modality_dispatcher is None:
                raise ValueError("NativeMoELinear LoRA requires modality_dispatcher")

            input_list = modality_dispatcher.dispatch(input)
            lora_outputs = []
            for i in range(self.num_experts):
                expert_input = self.dropout(input_list[i].to(torch.float32))
                expert_lora = self.lora_B[i](self.lora_A[i](expert_input))
                lora_outputs.append(expert_lora.to(base_out.dtype) * self.scaling)

            lora_combined = modality_dispatcher.undispatch(*lora_outputs)
            return base_out + lora_combined


def _get_module_by_name(model: nn.Module, name: str):
    """Get a submodule by dotted name, handling integer indices for ModuleList."""
    parts = name.split(".")
    module = model
    for p in parts:
        if p.isdigit():
            module = module[int(p)]
        else:
            module = getattr(module, p)
    return module


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Set a submodule by dotted name, handling integer indices for ModuleList."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
) -> list[str]:
    """Inject LoRA adapters into target modules of the DiT model.

    Args:
        model: The DiTModel instance.
        target_modules: List of regex patterns matching module names to wrap.
            Typical targets:
                - r"block\\.layers\\.\\d+\\.attention\\.linear_qkv"
                - r"block\\.layers\\.\\d+\\.attention\\.linear_proj"
                - r"block\\.layers\\.\\d+\\.mlp\\.up_gate_proj"
                - r"block\\.layers\\.\\d+\\.mlp\\.down_proj"
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.
        dropout: Dropout probability applied to LoRA input.

    Returns:
        List of module names that were wrapped with LoRA.
    """
    injected = []

    # Collect targets first to avoid modifying dict during iteration
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, (BaseLinear, NativeMoELinear)):
            continue
        for pattern in target_modules:
            if re.fullmatch(pattern, name):
                targets.append((name, module))
                break

    for name, module in targets:
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        _set_module_by_name(model, name, lora_layer)
        injected.append(name)

    # Freeze all base parameters, only train LoRA weights
    for param_name, param in model.named_parameters():
        if "lora_A" in param_name or "lora_B" in param_name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA injected into {len(injected)} modules")
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    return injected


def save_lora_weights(model: nn.Module, path: str):
    """Save only LoRA parameters to a file."""
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.detach().cpu()
    torch.save(lora_state, path)
    logger.info(f"Saved {len(lora_state)} LoRA tensors to {path}")


def load_lora_weights(model: nn.Module, path: str, device: str = "cpu"):
    """Load LoRA parameters into an already-injected model."""
    lora_state = torch.load(path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    # Filter out expected missing keys (base model params)
    real_missing = [k for k in missing if "lora_A" in k or "lora_B" in k]
    if real_missing:
        logger.warning(f"Missing LoRA keys: {real_missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    logger.info(f"Loaded {len(lora_state)} LoRA tensors from {path}")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into base layers for deployment (no extra latency).

    After merging, LoRALinear wrappers are replaced with the original base layers
    whose weights have been updated in-place.
    """
    merged = []
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            targets.append((name, module))

    for name, lora_module in targets:
        base = lora_module.base_layer
        scaling = lora_module.scaling

        if lora_module.num_experts == 1:
            # W' = W + scaling * B @ A
            delta = (lora_module.lora_B.weight @ lora_module.lora_A.weight) * scaling
            base.weight.data += delta.to(base.weight.dtype)
        else:
            # Per-expert merge
            weight_chunks = base.weight.chunk(lora_module.num_experts, dim=0)
            merged_chunks = []
            for i in range(lora_module.num_experts):
                delta = (lora_module.lora_B[i].weight @ lora_module.lora_A[i].weight) * scaling
                merged_chunks.append(weight_chunks[i] + delta.to(weight_chunks[i].dtype))
            base.weight.data = torch.cat(merged_chunks, dim=0)

        _set_module_by_name(model, name, base)
        merged.append(name)

    logger.info(f"Merged LoRA into {len(merged)} base layers")
    return model
