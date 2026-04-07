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

import torch

from inference.common import parse_config
from inference.device_utils import get_device, is_cuda, get_device_count
from inference.infra.distributed import get_dp_rank, initialize_distributed
from inference.utils import print_rank_0, set_random_seed


def initialize_infra():
    device = get_device()
    if not (torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())):
        print_rank_0("WARNING: No GPU accelerator found. Running on CPU (very slow).")

    # Initialize distributed environment (only when CUDA multi-GPU is present)
    if is_cuda(device) and get_device_count() > 0:
        initialize_distributed()

    # Initialize config
    config = parse_config(verbose=True)

    # Initialize random seed
    dp_rank = get_dp_rank() if torch.distributed.is_initialized() else 0
    set_random_seed(config.engine_config.seed + 10 * dp_rank)

    print_rank_0(f"Infra successfully initialized (device={device})")


__all__ = ["initialize_infra"]
