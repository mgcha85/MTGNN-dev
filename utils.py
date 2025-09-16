# core/utils.py
import torch
import random
import numpy as np
from dataclasses import dataclass
import yaml

def open_yaml(path: str) -> dict:
    """
    YAML 파일을 읽어서 Python dict로 반환하는 함수.
    
    Args:
        path (str): YAML 파일 경로
    
    Returns:
        dict: YAML 내용 (예: {'solution_columns': [...], 'attack_columns': [...]})
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data

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
