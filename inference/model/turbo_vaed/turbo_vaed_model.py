import json
import torch

from inference.device_utils import get_device, get_dtype
from .turbo_vaed_module import TurboVAED


def get_turbo_vaed(config_path, ckpt_path, device=None, weight_dtype=None) -> TurboVAED:
    device = device or get_device()
    weight_dtype = weight_dtype or torch.float32
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    student = TurboVAED.from_config(config)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "ema_state_dict" in ckpt, "ckpt must contain ema_state_dict"

    state_dict = ckpt["ema_state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    state_dict = new_state_dict

    missing, _ = student.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        sample_key = next(iter(state_dict.keys()))
        if not sample_key.startswith("decoder.") and not sample_key.startswith("encoder."):
            student.decoder.load_state_dict(state_dict, strict=False)

    student = student.to(device, dtype=weight_dtype)
    student.eval()
    student.requires_grad_(False)
    return student
