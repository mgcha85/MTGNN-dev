# core/metrics.py
import math
import torch

def rrse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    # Lai et al. 정의
    num = torch.sum((y_true - y_pred) ** 2)
    mean_all = torch.mean(y_true)
    denom = torch.sum((y_true - mean_all) ** 2)
    return (math.sqrt(num) / math.sqrt(denom)).item() if denom > 0 else float("nan")

def rae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    num = torch.sum(torch.abs(y_true - y_pred))
    mean_all = torch.mean(y_true)
    denom = torch.sum(torch.abs(y_true - mean_all))
    return (num / denom).item() if denom > 0 else float("nan")

def pearson_corr(y_pred_np, y_true_np) -> float:
    # y: (T, M) or (B,T,M) 평균 Pearson over columns
    import numpy as np
    pred = y_pred_np
    true = y_true_np
    if pred.ndim == 3:
        pred = pred.reshape(-1, pred.shape[-1])
        true = true.reshape(-1, true.shape[-1])
    sigma_p = pred.std(axis=0)
    sigma_g = true.std(axis=0)
    mean_p = pred.mean(axis=0)
    mean_g = true.mean(axis=0)
    idx = (sigma_g != 0)
    corr = ((pred - mean_p) * (true - mean_g)).mean(axis=0) / (sigma_p * sigma_g + 1e-12)
    corr = corr[idx].mean() if idx.any() else np.nan
    return float(corr)

def s_mape(y_true_np, y_pred_np) -> float:
    import numpy as np
    a = np.abs(y_true_np - y_pred_np)
    b = np.abs(y_true_np) + np.abs(y_pred_np) + 1e-12
    return float((a / b).mean())
