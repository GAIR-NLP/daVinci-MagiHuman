"""daVinci-MagiHuman DiT model ported to MLX.

15B parameter, 40-layer single-stream Transformer for video+audio generation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .rope import ElementWiseFourierEmbed, apply_rotary_emb


# ---------------------------------------------------------------------------
# Modality dispatcher — groups tokens by modality for MoE routing
# ---------------------------------------------------------------------------

class ModalityDispatcher:
    """Sort tokens by modality (video=0, audio=1, text=2) for per-expert processing."""

    def __init__(self, modality_mapping: mx.array, num_modalities: int):
        self.modality_mapping = modality_mapping
        self.num_modalities = num_modalities

        # Precompute stable sort permutation
        self.permute_mapping = mx.argsort(modality_mapping)
        self.inv_permute_mapping = mx.argsort(self.permute_mapping)

        # Sorted modality map + group sizes
        self.permuted_modality_mapping = modality_mapping[self.permute_mapping]
        # bincount: count tokens per modality
        self.group_size_cpu = []
        for i in range(num_modalities):
            count = int(mx.sum(modality_mapping == i).item())
            self.group_size_cpu.append(count)

    def dispatch(self, x: mx.array) -> List[mx.array]:
        """Split x (already in permuted order) into per-modality groups."""
        splits = []
        offset = 0
        for size in self.group_size_cpu:
            splits.append(x[offset : offset + size])
            offset += size
        return splits

    def undispatch(self, *groups: mx.array) -> mx.array:
        """Concatenate per-modality groups back together."""
        return mx.concatenate(list(groups), axis=0)

    @staticmethod
    def permute(x: mx.array, perm: mx.array) -> mx.array:
        return x[perm]

    @staticmethod
    def inv_permute(x: mx.array, inv_perm: mx.array) -> mx.array:
        return x[inv_perm]


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class MLPActivationType(Enum):
    SWIGLU7 = "swiglu7"
    GELU7 = "gelu7"


def swiglu7(x: mx.array, alpha: float = 1.702, limit: float = 7.0) -> mx.array:
    x = x.astype(mx.float32)
    x_glu = x[..., ::2]
    x_linear = x[..., 1::2]
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
    out_glu = x_glu * mx.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


def gelu7(x: mx.array, alpha: float = 1.702, limit: float = 7.0) -> mx.array:
    x = x.astype(mx.float32)
    x_glu = mx.clip(x, a_min=None, a_max=limit)
    return x_glu * mx.sigmoid(alpha * x_glu)


def create_activation_func(activation_type: MLPActivationType) -> Callable:
    if activation_type == MLPActivationType.SWIGLU7:
        return swiglu7
    elif activation_type == MLPActivationType.GELU7:
        return gelu7
    raise ValueError(f"Unknown activation type: {activation_type}")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class MultiModalityRMSNorm(nn.Module):
    """RMSNorm with optional per-modality scaling."""

    def __init__(self, dim: int, eps: float = 1e-6, num_modality: int = 1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality
        # Zero-initialized: (weight+1) = identity at init
        self.weight = mx.zeros((dim * num_modality,))

    def _rms_norm(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: mx.array, modality_dispatcher: Optional[ModalityDispatcher] = None) -> mx.array:
        if self.num_modality == 1:
            return self._forward_single(x)
        else:
            return self._forward_multi(x, modality_dispatcher)

    def _forward_single(self, x: mx.array) -> mx.array:
        t = self._rms_norm(x)
        return (t * (self.weight + 1)).astype(x.dtype)

    def _forward_multi(self, x: mx.array, md: ModalityDispatcher) -> mx.array:
        t = self._rms_norm(x)
        weight_chunks = mx.split(self.weight, self.num_modality, axis=0)
        t_list = md.dispatch(t)
        for i in range(self.num_modality):
            t_list[i] = t_list[i] * (weight_chunks[i] + 1)
        return md.undispatch(*t_list).astype(x.dtype)


# ---------------------------------------------------------------------------
# Linear layers
# ---------------------------------------------------------------------------

class BaseLinear(nn.Module):
    """Linear layer with weight shape [out_features * num_experts, in_features]."""

    def __init__(self, in_features: int, out_features: int, num_experts: int = 1, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.use_bias = bias
        self.weight = mx.zeros((out_features * num_experts, in_features))
        if bias:
            self.bias = mx.zeros((out_features * num_experts,))

    def __call__(self, x: mx.array, modality_dispatcher: Optional[ModalityDispatcher] = None) -> mx.array:
        out = x @ self.weight.T
        if self.use_bias:
            out = out + self.bias
        return out


class NativeMoELinear(nn.Module):
    """Multi-expert linear: dispatch → per-expert matmul → undispatch."""

    def __init__(self, in_features: int, out_features: int, num_experts: int = 3, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.use_bias = bias
        self.weight = mx.zeros((out_features * num_experts, in_features))
        if bias:
            self.bias = mx.zeros((out_features * num_experts,))

    def __call__(self, x: mx.array, modality_dispatcher: Optional[ModalityDispatcher] = None) -> mx.array:
        input_list = modality_dispatcher.dispatch(x)
        weight_chunks = mx.split(self.weight, self.num_experts, axis=0)
        if self.use_bias:
            bias_chunks = mx.split(self.bias, self.num_experts, axis=0)
        else:
            bias_chunks = [None] * self.num_experts

        for i in range(self.num_experts):
            out = input_list[i] @ weight_chunks[i].T
            if bias_chunks[i] is not None:
                out = out + bias_chunks[i]
            input_list[i] = out

        return modality_dispatcher.undispatch(*input_list)


def create_linear(in_features, out_features, num_experts=1, bias=True, **kwargs):
    if num_experts == 1:
        return BaseLinear(in_features, out_features, num_experts=1, bias=bias)
    else:
        return NativeMoELinear(in_features, out_features, num_experts=num_experts, bias=bias)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

@dataclass
class AttentionConfig:
    hidden_size: int
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    num_modality: int
    num_layers: int
    use_local_attn: bool = False
    enable_attn_gating: bool = False


class Attention(nn.Module):
    """Multi-head GQA attention with RoPE and per-modality experts."""

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config

        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=config.num_modality)
        self.gating_size = config.num_heads_q if config.enable_attn_gating else 0

        self.linear_qkv = create_linear(
            config.hidden_size,
            config.num_heads_q * config.head_dim + config.num_heads_kv * config.head_dim * 2 + self.gating_size,
            num_experts=config.num_modality,
            bias=False,
        )
        self.linear_proj = create_linear(
            config.num_heads_q * config.head_dim,
            config.hidden_size,
            num_experts=config.num_modality,
            bias=False,
        )
        self.q_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)
        self.k_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)

        self.q_size = config.num_heads_q * config.head_dim
        self.kv_size = config.num_heads_kv * config.head_dim

    def __call__(
        self,
        hidden_states: mx.array,
        rope: mx.array,
        permute_mapping: mx.array,
        inv_permute_mapping: mx.array,
        modality_dispatcher: ModalityDispatcher,
    ) -> mx.array:
        # 1. Pre-norm
        hidden_states = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher)

        # 2. QKV projection
        qkv = self.linear_qkv(hidden_states, modality_dispatcher=modality_dispatcher).astype(mx.float32)

        # 3. Split Q, K, V, G
        splits = [self.q_size, self.kv_size, self.kv_size, self.gating_size]
        cumsum = [0]
        for s in splits:
            cumsum.append(cumsum[-1] + s)
        q = qkv[:, cumsum[0]:cumsum[1]]
        k = qkv[:, cumsum[1]:cumsum[2]]
        v = qkv[:, cumsum[2]:cumsum[3]]
        g = qkv[:, cumsum[3]:cumsum[4]] if self.gating_size > 0 else None

        # 4. Reshape to heads
        N = q.shape[0]
        q = q.reshape(N, self.config.num_heads_q, self.config.head_dim)
        k = k.reshape(N, self.config.num_heads_kv, self.config.head_dim)
        v = v.reshape(N, self.config.num_heads_kv, self.config.head_dim)
        if g is not None:
            g = g.reshape(N, self.config.num_heads_q, -1)

        # 5. Per-head Q/K norms
        q = self.q_norm(q, modality_dispatcher=modality_dispatcher)
        k = self.k_norm(k, modality_dispatcher=modality_dispatcher)

        # 6. Inv-permute to restore original token order (for RoPE)
        q = ModalityDispatcher.inv_permute(q, inv_permute_mapping)[None, ...]  # [1, N, H, D]
        k = ModalityDispatcher.inv_permute(k, inv_permute_mapping)[None, ...]
        v = ModalityDispatcher.inv_permute(v, inv_permute_mapping)[None, ...]

        # 7. Apply RoPE
        sin_emb, cos_emb = mx.split(rope, 2, axis=-1)
        q = apply_rotary_emb(q, cos_emb, sin_emb)
        k = apply_rotary_emb(k, cos_emb, sin_emb)

        # 8. Transpose to (B, H, N, D) for SDPA
        q = q.transpose(0, 2, 1, 3)  # [1, num_heads_q, N, D]
        k = k.transpose(0, 2, 1, 3)  # [1, num_heads_kv, N, D]
        v = v.transpose(0, 2, 1, 3)

        # 9. SDPA (handles GQA natively when num_q = multiple of num_kv)
        scale = 1.0 / (self.config.head_dim ** 0.5)
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        # attn_out: [1, num_heads_q, N, D]

        # 10. Transpose back and squeeze batch
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(N, -1)  # [N, num_heads_q * D]

        # 11. Re-permute to modality-grouped order
        attn_out = ModalityDispatcher.permute(attn_out, permute_mapping)

        # 12. Gating
        if self.config.enable_attn_gating and g is not None:
            attn_out = attn_out.reshape(N, self.config.num_heads_q, self.config.head_dim)
            attn_out = attn_out * mx.sigmoid(g)
            attn_out = attn_out.reshape(N, -1)

        # 13. Output projection
        out = self.linear_proj(attn_out, modality_dispatcher=modality_dispatcher)
        return out


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

@dataclass
class MLPConfig:
    hidden_size: int
    intermediate_size: int
    activation_type: MLPActivationType
    num_modality: int = 1
    num_layers: int = 1
    gated_act: bool = False


class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        num_experts = config.num_modality
        intermediate_size_up = config.intermediate_size * 2 if config.gated_act else config.intermediate_size

        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=config.num_modality)
        self.up_gate_proj = create_linear(
            config.hidden_size, intermediate_size_up, num_experts=num_experts, bias=False,
        )
        self.down_proj = create_linear(
            config.intermediate_size, config.hidden_size, num_experts=num_experts, bias=False,
        )
        self.activation_func = create_activation_func(config.activation_type)

    def __call__(self, x: mx.array, modality_dispatcher: ModalityDispatcher) -> mx.array:
        x = self.pre_norm(x, modality_dispatcher=modality_dispatcher)
        x = self.up_gate_proj(x, modality_dispatcher=modality_dispatcher).astype(mx.float32)
        x = self.activation_func(x)
        x = self.down_proj(x, modality_dispatcher=modality_dispatcher).astype(mx.float32)
        return x


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------

class TransFormerLayer(nn.Module):
    def __init__(self, model_config: Any, layer_idx: int):
        super().__init__()
        num_modality = 3 if layer_idx in model_config.mm_layers else 1
        use_local_attn = layer_idx in model_config.local_attn_layers
        self.post_norm = layer_idx in model_config.post_norm_layers

        attn_config = AttentionConfig(
            hidden_size=model_config.hidden_size,
            num_heads_q=model_config.num_heads_q,
            num_heads_kv=model_config.num_heads_kv,
            head_dim=model_config.head_dim,
            num_modality=num_modality,
            num_layers=model_config.num_layers,
            use_local_attn=use_local_attn,
            enable_attn_gating=model_config.enable_attn_gating,
        )
        self.attention = Attention(attn_config)

        activation_type = MLPActivationType.GELU7 if layer_idx in model_config.gelu7_layers else MLPActivationType.SWIGLU7
        if activation_type == MLPActivationType.SWIGLU7:
            gated_act = True
            intermediate_size = int(model_config.hidden_size * 4 * 2 / 3) // 4 * 4
        else:
            gated_act = False
            intermediate_size = model_config.hidden_size * 4

        mlp_config = MLPConfig(
            hidden_size=model_config.hidden_size,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            num_modality=num_modality,
            num_layers=model_config.num_layers,
            gated_act=gated_act,
        )
        self.mlp = MLP(mlp_config)

        if self.post_norm:
            self.attn_post_norm = MultiModalityRMSNorm(model_config.hidden_size, num_modality=num_modality)
            self.mlp_post_norm = MultiModalityRMSNorm(model_config.hidden_size, num_modality=num_modality)

    def __call__(
        self,
        hidden_states: mx.array,
        rope: mx.array,
        permute_mapping: mx.array,
        inv_permute_mapping: mx.array,
        modality_dispatcher: ModalityDispatcher,
    ) -> mx.array:
        # Attention + residual
        attn_out = self.attention(
            hidden_states, rope, permute_mapping, inv_permute_mapping, modality_dispatcher,
        )
        if self.post_norm:
            attn_out = self.attn_post_norm(attn_out, modality_dispatcher=modality_dispatcher)
        hidden_states = hidden_states + attn_out
        mx.eval(hidden_states)  # Force eval to free attention intermediates

        # MLP + residual
        mlp_out = self.mlp(hidden_states, modality_dispatcher)
        if self.post_norm:
            mlp_out = self.mlp_post_norm(mlp_out, modality_dispatcher=modality_dispatcher)
        hidden_states = hidden_states + mlp_out

        return hidden_states


# ---------------------------------------------------------------------------
# Transformer block (40 layers)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, model_config: Any):
        super().__init__()
        self.layers = [TransFormerLayer(model_config, i) for i in range(model_config.num_layers)]

    def __call__(
        self,
        x: mx.array,
        rope: mx.array,
        permute_mapping: mx.array,
        inv_permute_mapping: mx.array,
        modality_dispatcher: ModalityDispatcher,
    ) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x, rope, permute_mapping, inv_permute_mapping, modality_dispatcher)
            # Memory management: evaluate after each layer
            mx.eval(x)
        return x


# ---------------------------------------------------------------------------
# Adapter (input projection)
# ---------------------------------------------------------------------------

class Adapter(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, video_in: int, audio_in: int, text_in: int):
        super().__init__()
        self.video_embedder = nn.Linear(video_in, hidden_size)
        self.text_embedder = nn.Linear(text_in, hidden_size)
        self.audio_embedder = nn.Linear(audio_in, hidden_size)
        head_dim = hidden_size // num_heads
        self.rope = ElementWiseFourierEmbed(head_dim, learnable=False)

    def __call__(
        self,
        x: mx.array,
        coords_mapping: mx.array,
        video_mask: mx.array,
        audio_mask: mx.array,
        text_mask: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        rope = self.rope(coords_mapping)
        hidden_size = self.video_embedder.weight.shape[0]

        # Compute all three embeddings for ALL tokens, then select via mask
        # This avoids boolean indexing (unsupported in MLX) at the cost of
        # extra compute, but it's only 3 linear layers on the input.
        text_in = self.text_embedder.weight.shape[1]
        audio_in = self.audio_embedder.weight.shape[1]
        video_in = self.video_embedder.weight.shape[1]

        text_emb = self.text_embedder(x[:, :text_in])    # [N, hidden]
        audio_emb = self.audio_embedder(x[:, :audio_in])
        video_emb = self.video_embedder(x[:, :video_in])

        # Select per-token using masks
        output = mx.where(text_mask[:, None], text_emb, mx.zeros_like(text_emb))
        output = mx.where(audio_mask[:, None], audio_emb, output)
        output = mx.where(video_mask[:, None], video_emb, output)

        return output, rope


# ---------------------------------------------------------------------------
# DiTModel — full model
# ---------------------------------------------------------------------------

class Modality:
    VIDEO = 0
    AUDIO = 1
    TEXT = 2


class DiTModel(nn.Module):
    """15B parameter video+audio diffusion transformer."""

    def __init__(self, model_config: Any):
        super().__init__()
        self.hidden_size = model_config.hidden_size
        self.video_in_channels = model_config.video_in_channels
        self.audio_in_channels = model_config.audio_in_channels

        self.adapter = Adapter(
            hidden_size=model_config.hidden_size,
            num_heads=model_config.num_heads_q,
            video_in=model_config.video_in_channels,
            audio_in=model_config.audio_in_channels,
            text_in=model_config.text_in_channels,
        )
        self.block = TransformerBlock(model_config)

        self.final_norm_video = MultiModalityRMSNorm(model_config.hidden_size)
        self.final_norm_audio = MultiModalityRMSNorm(model_config.hidden_size)
        self.final_linear_video = nn.Linear(model_config.hidden_size, model_config.video_in_channels, bias=False)
        self.final_linear_audio = nn.Linear(model_config.hidden_size, model_config.audio_in_channels, bias=False)

    def __call__(
        self,
        x: mx.array,
        coords_mapping: mx.array,
        modality_mapping: mx.array,
    ) -> mx.array:
        # 1. Build modality dispatcher
        md = ModalityDispatcher(modality_mapping, num_modalities=3)

        # 2. Create masks
        video_mask = modality_mapping == Modality.VIDEO
        audio_mask = modality_mapping == Modality.AUDIO
        text_mask = modality_mapping == Modality.TEXT

        # 3. Adapter: per-modality embedding + RoPE
        x, rope = self.adapter(x, coords_mapping, video_mask, audio_mask, text_mask)

        # 4. Permute to modality-grouped order
        x = ModalityDispatcher.permute(x, md.permute_mapping)

        # 5. Transformer (40 layers)
        x = self.block(x, rope, md.permute_mapping, md.inv_permute_mapping, md)

        # 6. Inv-permute to original order
        x = ModalityDispatcher.inv_permute(x, md.inv_permute_mapping)

        # 7. Per-modality output heads
        # Compute both output heads for all tokens, select via mask
        max_out = max(self.video_in_channels, self.audio_in_channels)

        x_f32 = x.astype(mx.float32)
        x_video = self.final_linear_video(self.final_norm_video(x_f32))
        x_audio = self.final_linear_audio(self.final_norm_audio(x_f32))

        # Pad to common width
        if self.video_in_channels < max_out:
            x_video = mx.pad(x_video, [(0, 0), (0, max_out - self.video_in_channels)])
        if self.audio_in_channels < max_out:
            x_audio = mx.pad(x_audio, [(0, 0), (0, max_out - self.audio_in_channels)])

        # Select: video tokens get video output, audio tokens get audio output
        x_out = mx.where(video_mask[:, None], x_video, mx.zeros_like(x_video))
        x_out = mx.where(audio_mask[:, None], x_audio, x_out)

        return x_out
