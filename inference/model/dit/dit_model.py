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
from inference.infra.checkpoint import load_model_checkpoint
from inference.infra.distributed import get_cp_rank, get_pp_rank, get_tp_rank
from inference.utils import print_mem_info_rank_0, print_model_size, print_rank_0

from .dit_module import DiTModel


def get_dit(model_config, engine_config):
    """Build and load DiT model."""
    model = DiTModel(model_config=model_config)

    print_rank_0("Build dit model successfully")
    print_rank_0(model)
    print_model_size(
        model, prefix=f"(tp, cp, pp) rank ({get_tp_rank()}, {get_cp_rank()}, {get_pp_rank()}): ", print_func=print_rank_0
    )

    model = load_model_checkpoint(model, engine_config)
    # ghitza-patch 2026-11-04: Blackwell/WSL + fp8 trips "CUDA driver error: device not
    # ready" on a single bulk model.cuda(). Warm the context, then move params/buffers
    # one at a time with a sync between each, which clears the bulk-transfer failure.
    import os

    def _move(mod, dev):
        # Chunked per-param move with a sync between each: a bulk .to() trips "device not ready"
        # on Blackwell/WSL + fp8; the chunked move clears it.
        idx = torch.device(dev).index
        torch.zeros(1, device=dev)
        torch.cuda.synchronize(idx)
        for _p in mod.parameters(recurse=True):
            _p.data = _p.data.to(dev, non_blocking=False)
            torch.cuda.synchronize(idx)
        for _b in mod.buffers(recurse=True):
            _b.data = _b.data.to(dev, non_blocking=False)
        torch.cuda.synchronize(idx)

    if os.environ.get("MAGI_DIT_SPLIT") == "1" and torch.cuda.device_count() > 1:
        # ghitza-patch 2026-11-04: pipeline-split the 40-layer block across both GPUs so each card
        # holds ~half the weights AND ~half the activations (full resolution does not fit one 16GB
        # card). adapter + layers[0:k] on cuda:0 (5070 Ti), layers[k:] + final heads on cuda:1
        # (4070). k = MAGI_DIT_CUT. The boundary handoff lives in TransformerBlock.forward.
        k = int(os.environ.get("MAGI_DIT_CUT", "26"))
        _move(model.adapter, "cuda:0")
        for _i, _layer in enumerate(model.block.layers):
            _move(_layer, "cuda:0" if _i < k else "cuda:1")
        for _m in (model.final_norm_video, model.final_norm_audio, model.final_linear_video, model.final_linear_audio):
            _move(_m, "cuda:1")
        print_rank_0(f"[ghitza] DiT pipeline-split across 2 GPUs at layer {k}")
    else:
        _move(model, f"cuda:{torch.cuda.current_device()}")
    model.eval()
    print_mem_info_rank_0("Load model successfully")

    gc.collect()
    torch.cuda.empty_cache()
    return model
