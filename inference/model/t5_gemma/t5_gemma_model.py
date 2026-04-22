from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer
from transformers.models.t5gemma import T5GemmaEncoderModel

from inference.common import CPUOffloadWrapper, get_arch_memory
from inference.utils import env_is_true


class T5GemmaEncoder:
    def __init__(self, model_path: str, device: str, weight_dtype: torch.dtype):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5GemmaEncoderModel.from_pretrained(
            model_path,
            is_encoder_decoder=False,
            dtype=weight_dtype,
        ).to(device)
        self.model = CPUOffloadWrapper(model, is_cpu_offload=env_is_true("CPU_OFFLOAD") or get_arch_memory() <= 48)

    def encode(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs["last_hidden_state"].half()


_t5_gemma_cache: Dict[Tuple[str, str, torch.dtype], T5GemmaEncoder] = {}


def get_t5_gemma_encoder(model_path: str, device: str, weight_dtype: torch.dtype) -> T5GemmaEncoder:
    cache_key = (model_path, device, weight_dtype)
    if cache_key not in _t5_gemma_cache:
        _t5_gemma_cache[cache_key] = T5GemmaEncoder(model_path=model_path, device=device, weight_dtype=weight_dtype)
    return _t5_gemma_cache[cache_key]


@torch.inference_mode()
def get_t5_gemma_embedding(prompt: str, model_path: str, device: str, weight_dtype: torch.dtype) -> torch.Tensor:
    encoder = get_t5_gemma_encoder(model_path=model_path, device=device, weight_dtype=weight_dtype)
    return encoder.encode(prompt)
