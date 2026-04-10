#!/usr/bin/env python3
"""Bisect which PyTorch operation produces wrong results on MPS vs CPU.

Loads the real DiT model, runs each layer's sub-operations on both MPS
and CPU with identical inputs, and reports any divergences.
"""
import os, sys, torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"

sys.argv = ["test", "--config-load-path", "example/distill/config_mps.json"]

# Force CPU for model loading
import inference.device_utils as du
_original_get_device = du.get_device
du.get_device = lambda force=None: force if force else "cpu"

from inference.common import parse_config
from inference.model.dit.dit_model import get_dit
from inference.model.dit.dit_module import (
    ModalityDispatcher, MultiModalityRMSNorm, apply_rotary_emb_torch,
    swiglu7, gelu7, ElementWiseFourierEmbed,
)
from inference.utils import set_random_seed

THRESHOLD = 0.01  # max acceptable absolute difference


def cmp(name, cpu_out, mps_out, detail=False):
    if cpu_out is None or mps_out is None:
        print(f"  {name}: SKIPPED")
        return True
    a, b = cpu_out.float(), mps_out.cpu().float()
    diff = (b - a).abs()
    mx = diff.max().item()
    mn = diff.mean().item()
    ok = mx < THRESHOLD
    status = "OK" if ok else "*** MISMATCH ***"
    print(f"  {name}: max_diff={mx:.6f} mean_diff={mn:.8f} [{status}]")
    if not ok and detail:
        # Find where the worst differences are
        flat = diff.flatten()
        worst_idx = flat.topk(5).indices
        for idx in worst_idx:
            print(f"    idx={idx.item()}: cpu={a.flatten()[idx]:.6f} mps={b.flatten()[idx]:.6f}")
    return ok


print("Loading model on CPU...")
set_random_seed(42)
config = parse_config()
model = get_dit(config.arch_config, config.engine_config)
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params\n")

# Create realistic input tensors (matching what the pipeline produces)
# For 256p 2s: video has ~5824 tokens, audio ~51, text ~640
torch.manual_seed(42)
N_video = 5824
N_audio = 51
N_text = 640
N_total = N_video + N_audio + N_text
hidden_size = 5120

print("=" * 60)
print("PHASE 1: Individual operations")
print("=" * 60)

# --- Test 1: torch.matmul (large) ---
print("\n[1] torch.matmul (5120x5120)")
x = torch.randn(N_total, hidden_size)
w = torch.randn(hidden_size, hidden_size)
cmp("matmul", x @ w.t(), x.to("mps") @ w.to("mps").t())

# --- Test 2: RMSNorm (single modality) ---
print("\n[2] RMSNorm (single modality)")
norm = model.block.layers[4].attention.pre_norm  # single-expert layer
x = torch.randn(N_total, hidden_size)
mapping = torch.zeros(N_total, dtype=torch.long)
md_cpu = ModalityDispatcher(mapping, num_modalities=1)
out_cpu = norm(x, modality_dispatcher=md_cpu)
norm_mps = MultiModalityRMSNorm(hidden_size, num_modality=1).to("mps")
norm_mps.load_state_dict(norm.state_dict())
md_mps = ModalityDispatcher(mapping.to("mps"), num_modalities=1)
out_mps = norm_mps(x.to("mps"), modality_dispatcher=md_mps)
cmp("RMSNorm_single", out_cpu, out_mps)

# --- Test 3: RMSNorm (multi modality - 3 experts) ---
print("\n[3] RMSNorm (3 modalities)")
norm3 = model.block.layers[0].attention.pre_norm  # multi-expert layer
mapping3 = torch.cat([
    torch.zeros(N_video, dtype=torch.long),
    torch.ones(N_audio, dtype=torch.long),
    torch.full((N_text,), 2, dtype=torch.long),
])
md3_cpu = ModalityDispatcher(mapping3, num_modalities=3)
out_cpu = norm3(x[:N_total], modality_dispatcher=md3_cpu)
norm3_mps = MultiModalityRMSNorm(hidden_size, num_modality=3).to("mps")
norm3_mps.load_state_dict(norm3.state_dict())
md3_mps = ModalityDispatcher(mapping3.to("mps"), num_modalities=3)
out_mps = norm3_mps(x[:N_total].to("mps"), modality_dispatcher=md3_mps)
cmp("RMSNorm_multi", out_cpu, out_mps)

# --- Test 4: ModalityDispatcher operations ---
print("\n[4] ModalityDispatcher")
x = torch.randn(N_total, hidden_size)
# permute
perm_cpu = ModalityDispatcher.permute(x, md3_cpu.permuted_modality_mapping)
perm_mps = ModalityDispatcher.permute(x.to("mps"), md3_mps.permuted_modality_mapping)
cmp("permute", perm_cpu, perm_mps)
# inv_permute
inv_perm = md3_cpu.modality_mapping.argsort()
inv_perm_mps = md3_mps.modality_mapping.argsort()
unperm_cpu = ModalityDispatcher.inv_permute(perm_cpu, inv_perm)
unperm_mps = ModalityDispatcher.inv_permute(perm_mps, inv_perm_mps)
cmp("inv_permute", unperm_cpu, unperm_mps)
# dispatch + undispatch
dispatched_cpu = md3_cpu.dispatch(x)
dispatched_mps = md3_mps.dispatch(x.to("mps"))
for i in range(3):
    cmp(f"dispatch[{i}]", dispatched_cpu[i], dispatched_mps[i])

# --- Test 5: swiglu7 / gelu7 ---
print("\n[5] Activations")
x = torch.randn(N_total, 10240)
cmp("swiglu7", swiglu7(x), swiglu7(x.to("mps")))
x2 = torch.randn(N_total, 5120)
cmp("gelu7", gelu7(x2), gelu7(x2.to("mps")))

# --- Test 6: Rotary Embeddings ---
print("\n[6] Rotary Position Embeddings")
q = torch.randn(1, N_total, 40, 128)
rope = model.adapter.rope
coords = torch.randn(N_total, 5)
rope_out_cpu = rope(coords)
rope_mps = ElementWiseFourierEmbed(128, in_pixels=False, learnable=False).to("mps")
rope_out_mps = rope_mps(coords.to("mps"))
cmp("FourierEmbed", rope_out_cpu, rope_out_mps)

sin_cpu, cos_cpu = rope_out_cpu.tensor_split(2, -1)
sin_mps, cos_mps = rope_out_mps.tensor_split(2, -1)
q_cpu = apply_rotary_emb_torch(q, cos_cpu.unsqueeze(0), sin_cpu.unsqueeze(0))
q_mps = apply_rotary_emb_torch(q.to("mps"), cos_mps.unsqueeze(0), sin_mps.unsqueeze(0))
cmp("apply_rotary_emb", q_cpu, q_mps)

# --- Test 7: BaseLinear (single-expert, layer 4) ---
print("\n[7] BaseLinear (single expert)")
bl = model.block.layers[4].attention.linear_qkv
x = torch.randn(N_total, hidden_size)
with torch.no_grad():
    out_cpu = bl(x)
bl_mps = type(bl)(bl.in_features, bl.out_features, bl.num_layers_for_initialization,
                   bl.num_experts, bl.use_bias).to("mps")
bl_mps.load_state_dict(bl.state_dict())
with torch.no_grad():
    out_mps = bl_mps(x.to("mps"))
cmp("BaseLinear", out_cpu, out_mps, detail=True)

# --- Test 8: NativeMoELinear (multi-expert, layer 0) ---
print("\n[8] NativeMoELinear (3 experts)")
ml = model.block.layers[0].attention.linear_qkv
x = torch.randn(N_total, hidden_size)
with torch.no_grad():
    out_cpu = ml(x, modality_dispatcher=md3_cpu)
ml_mps = type(ml)(ml.in_features, ml.out_features, ml.num_layers_for_initialization,
                   ml.num_experts, ml.use_bias).to("mps")
ml_mps.load_state_dict(ml.state_dict())
with torch.no_grad():
    out_mps = ml_mps(x.to("mps"), modality_dispatcher=md3_mps)
cmp("NativeMoELinear", out_cpu, out_mps, detail=True)

# --- Test 9: SDPA attention with GQA ---
print("\n[9] SDPA attention (GQA: 40 q-heads, 8 kv-heads)")
import torch.nn.functional as F
q = torch.randn(1, 40, N_total, 128)
k = torch.randn(1, 8, N_total, 128)
v = torch.randn(1, 8, N_total, 128)
k_exp = k.repeat_interleave(5, dim=1)
v_exp = v.repeat_interleave(5, dim=1)
out_cpu = F.scaled_dot_product_attention(q, k_exp, v_exp)
out_mps = F.scaled_dot_product_attention(q.to("mps"), k_exp.to("mps"), v_exp.to("mps"))
cmp("SDPA_GQA", out_cpu, out_mps)

print("\n" + "=" * 60)
print("PHASE 2: Full transformer layer (layer 4, single-expert)")
print("=" * 60)

# Run a full single-expert layer on CPU vs MPS
layer = model.block.layers[4]
x = torch.randn(N_total, hidden_size)
# Create all required inputs
mapping1 = torch.zeros(N_total, dtype=torch.long)
md1_cpu = ModalityDispatcher(mapping1, num_modalities=1)
md1_mps = ModalityDispatcher(mapping1.to("mps"), num_modalities=1)

rope_embed = rope(coords)
perm_map = md1_cpu.permuted_modality_mapping
inv_perm_map = mapping1.argsort()

from inference.model.dit.dit_module import VarlenHandler, FFAHandler
vh = VarlenHandler(cu_seqlen=torch.tensor([0, N_total]), max_seqlen=torch.tensor(N_total))
ffa = FFAHandler(
    q_ranges=torch.tensor([[0, N_total]]),
    k_ranges=torch.tensor([[0, N_total]]),
    max_seqlen_q=N_total, max_seqlen_k=N_total,
    attn_type_map=torch.zeros(1, dtype=torch.int32),
    softmax_scale=None,
)

with torch.no_grad():
    out_cpu = layer(
        x, rope_embed, perm_map, inv_perm_map,
        vh, ffa, md1_cpu, [N_total]
    )

# Move layer to MPS
layer_mps = layer.to("mps")
perm_map_mps = md1_mps.permuted_modality_mapping
inv_perm_map_mps = mapping1.to("mps").argsort()
rope_embed_mps = rope_mps(coords.to("mps"))
ffa_mps = FFAHandler(
    q_ranges=ffa.q_ranges.to("mps"),
    k_ranges=ffa.k_ranges.to("mps"),
    max_seqlen_q=N_total, max_seqlen_k=N_total,
    attn_type_map=ffa.attn_type_map.to("mps"),
    softmax_scale=None,
)
vh_mps = VarlenHandler(
    cu_seqlen=vh.cu_seqlen.to("mps"),
    max_seqlen=vh.max_seqlen.to("mps") if isinstance(vh.max_seqlen, torch.Tensor) else vh.max_seqlen
)

with torch.no_grad():
    out_mps = layer_mps(
        x.to("mps"), rope_embed_mps, perm_map_mps, inv_perm_map_mps,
        vh_mps, ffa_mps, md1_mps, [N_total]
    )

cmp("full_layer_4", out_cpu, out_mps, detail=True)

# Move layer back to CPU for next test
layer.cpu()

print("\n" + "=" * 60)
print("PHASE 3: Full multi-modal layer (layer 0, 3-expert)")
print("=" * 60)

layer0 = model.block.layers[0]
x = torch.randn(N_total, hidden_size)

with torch.no_grad():
    out_cpu = layer0(
        x, rope_embed, md3_cpu.permuted_modality_mapping,
        mapping3.argsort(), vh, ffa, md3_cpu, [N_total]
    )

layer0_mps = layer0.to("mps")
with torch.no_grad():
    out_mps = layer0_mps(
        x.to("mps"), rope_embed_mps,
        md3_mps.permuted_modality_mapping,
        mapping3.to("mps").argsort(),
        vh_mps, ffa_mps, md3_mps, [N_total]
    )

cmp("full_layer_0_multimodal", out_cpu, out_mps, detail=True)

print("\nBisection complete!")
