# scripts/forecast.py
import argparse
import numpy as np
import torch
from net import gtnet
from core.config import Cfg
from utils import resolve_devices, set_random_seed
from core.dataset import DataModule
from core.evaluator import Evaluator
from core.checkpoint import load_ckpt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--ckpt", type=str, default="model/Bayesian/model.safetensors")
    p.add_argument("--mc_runs", type=int, default=10)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Cfg()
    cfg.save.ckpt_path = args.ckpt
    ckpt_device_str, torch_device = resolve_devices(args.device)
    set_random_seed(123)

    data = DataModule(
        path=cfg.data.path,
        split_train=cfg.data.split_train,
        split_val=cfg.data.split_val,
        device=torch_device,
        horizon=1,
        seq_in_len=cfg.data.seq_in,
        normalize=cfg.data.normalize,
        seq_out_len=cfg.data.seq_out
    )
    cfg.data.num_nodes = data.m

    model = gtnet(
        cfg.model.gcn_true, cfg.model.buildA_true, cfg.model.gcn_depth,
        cfg.data.num_nodes, torch_device, data.adj,
        dropout=getattr(cfg.model, "dropout", 0.3),
        subgraph_size=cfg.model.subgraph_size, node_dim=cfg.model.node_dim,
        dilation_exponential=cfg.model.dilation_exponential, conv_channels=cfg.model.conv_channels,
        residual_channels=cfg.model.residual_channels, skip_channels=cfg.model.skip_channels,
        end_channels=cfg.model.end_channels, seq_length=cfg.data.seq_in,
        in_dim=cfg.model.in_dim, out_dim=cfg.model.out_dim, layers=cfg.model.layers,
        propalpha=cfg.model.propalpha, tanhalpha=cfg.model.tanhalpha,
        layer_norm_affline=cfg.model.layer_norm_affline
    ).to(torch_device)

    load_ckpt(model, cfg.save.ckpt_path, ckpt_device_str)
    evaluator = Evaluator(torch_device)

    # 최근 P=seq_in 길이만큼의 입력으로 포캐스트
    P = cfg.data.seq_in
    X_last = torch.from_numpy(data.ds.dat[-P:, :]).to(dtype=torch.float, device=torch_device)  # (P, M)
    X_last = X_last.unsqueeze(0).unsqueeze(1).transpose(2, 3)  # (1,1,M,P)

    mean, var, conf = evaluator.mc_dropout_forecast(model, X_last, out_len=cfg.data.seq_out, num_runs=args.mc_runs)

    # 역스케일: scale은 (m,)
    scale_t = data.scale.to(device=torch_device, dtype=mean.dtype).expand_as(mean)
    mean = mean * scale_t
    var = var * (scale_t ** 2)
    conf = conf * scale_t

    print("Forecast shapes:", mean.shape, var.shape, conf.shape)  # (T_out, M)

if __name__ == "__main__":
    main()
