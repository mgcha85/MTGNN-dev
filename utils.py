# core/utils.py
import torch
import random
import numpy as np
from dataclasses import dataclass

def resolve_devices(device_str: str = "auto"):
    """
    Return (ckpt_device_str, torch_device)
    - ckpt_device_str: "cuda:0" | "cpu"  (safetensors.load_file 용)
    - torch_device: torch.device          (모델 .to(...) 용)
    """
    if device_str == "auto":
        ckpt = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device_str.startswith("cuda"):
        # 허용: "cuda", "cuda:0", "cuda:1"
        ckpt = "cuda:0" if device_str == "cuda" else device_str
    else:
        ckpt = "cpu"
    return ckpt, torch.device(ckpt)

def set_random_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
