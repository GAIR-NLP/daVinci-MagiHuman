# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc

import torch
from inference.device_utils import get_device, empty_cache, is_cuda
from inference.infra.checkpoint import load_model_checkpoint
from inference.infra.distributed import get_cp_rank, get_pp_rank, get_tp_rank
from inference.utils import print_mem_info_rank_0, print_model_size, print_rank_0

from .dit_module import DiTModel


def get_dit(model_config, engine_config):
    """Build and load DiT model."""
    model = DiTModel(model_config=model_config)

    print_rank_0("Build dit model successfully")
    print_rank_0(model)

    tp_rank = get_tp_rank() if torch.distributed.is_initialized() else 0
    cp_rank = get_cp_rank() if torch.distributed.is_initialized() else 0
    pp_rank = get_pp_rank() if torch.distributed.is_initialized() else 0
    print_model_size(
        model, prefix=f"(tp, cp, pp) rank ({tp_rank}, {cp_rank}, {pp_rank}): ", print_func=print_rank_0
    )

    model = load_model_checkpoint(model, engine_config)

    device = get_device()
    model.to(device)
    model.eval()

    # MPS hybrid optimization: move single-expert MLP + linear layers to MPS
    # for ~4-5x speedup on those ops, while keeping the rest on CPU to avoid
    # the MPS chained-operation bug.
    from inference.device_utils import get_mps_device
    mps = get_mps_device()
    if mps and device == "cpu":
        mps_full = 0  # single-expert: MLP + linears on MPS
        mps_linear = 0  # multi-expert: only linears on MPS (MLP stays CPU)
        for i, layer in enumerate(model.block.layers):
            is_single_expert = layer.mlp.up_gate_proj.num_experts == 1
            # Attention linears: safe on MPS for ALL layers
            layer.attention.linear_qkv.to(mps)
            layer.attention.linear_proj.to(mps)
            if is_single_expert:
                # Single-expert MLP: fully safe on MPS
                layer.mlp.to(mps)
                mps_full += 1
            else:
                # Multi-expert MLP: move weights to MPS, dispatch/undispatch on CPU
                # NativeMoELinear.forward handles cross-device matmul automatically
                layer.mlp.up_gate_proj.weight = torch.nn.Parameter(layer.mlp.up_gate_proj.weight.to(mps), requires_grad=False)
                if layer.mlp.up_gate_proj.bias is not None:
                    layer.mlp.up_gate_proj.bias = torch.nn.Parameter(layer.mlp.up_gate_proj.bias.to(mps), requires_grad=False)
                layer.mlp.down_proj.weight = torch.nn.Parameter(layer.mlp.down_proj.weight.to(mps), requires_grad=False)
                if layer.mlp.down_proj.bias is not None:
                    layer.mlp.down_proj.bias = torch.nn.Parameter(layer.mlp.down_proj.bias.to(mps), requires_grad=False)
                mps_linear += 1
        print_rank_0(f"MPS acceleration: {mps_full} layers full MPS (MLP+linears), "
                    f"{mps_linear} layers partial MPS (linears only). "
                    f"Norms + attention prep on CPU.")

    print_mem_info_rank_0("Load model successfully")

    gc.collect()
    empty_cache()
    return model
