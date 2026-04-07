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
    print_mem_info_rank_0("Load model successfully")

    gc.collect()
    empty_cache()
    return model
