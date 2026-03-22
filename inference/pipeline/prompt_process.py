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

from typing import Tuple

import torch
from torch.nn import functional as F

from inference.model.t5_gemma import get_t5_gemma_embedding


def pad_or_trim(tensor: torch.Tensor, target_size: int, dim: int, pad_value: float = 0.0) -> Tuple[torch.Tensor, int]:
    """
    Pads or trims a tensor along a specified dimension to reach a target size.

    Args:
        tensor (torch.Tensor): The input tensor to be processed.
        target_size (int): The desired size for the specified dimension.
        dim (int): The dimension along which to pad or trim.
        pad_value (float, optional): The value used for padding. Defaults to 0.0.

    Returns:
        torch.Tensor: The resulting tensor with the target size in the specified dimension.
    """
    current_size = tensor.size(dim)
    if current_size < target_size:
        padding_amount = target_size - current_size
        padding_tuple = [0] * (2 * tensor.dim())
        padding_dim_index = tensor.dim() - 1 - dim
        padding_tuple[2 * padding_dim_index + 1] = padding_amount
        return F.pad(tensor, tuple(padding_tuple), "constant", pad_value), current_size

    slicing = [slice(None)] * tensor.dim()
    slicing[dim] = slice(0, target_size)
    return tensor[tuple(slicing)], target_size


def get_padded_t5_gemma_embedding(
    prompt: str,
    model_path: str,
    device: str,
    weight_dtype: torch.dtype,
    target_length: int,
) -> Tuple[torch.Tensor, int]:
    txt_feat = get_t5_gemma_embedding(prompt, model_path, device, weight_dtype)
    txt_feat, original_len = pad_or_trim(txt_feat, target_size=target_length, dim=1)
    return txt_feat.to(torch.float32), original_len


