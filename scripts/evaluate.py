# scripts/evaluate.py
import argparse
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
    evalL2 = torch.nn.MSELoss(reduction='sum').to(torch_device)
    evalL1 = torch.nn.L1Loss(reduction='sum').to(torch_device)

    rse, rae, corr, smape = evaluator.evaluate(
        data, data.valid[0], data.valid[1], model, evalL2, evalL1, batch_size=cfg.train.batch_size, is_plot=False
    )
    print(f"Validation: rse={rse:.4f} rae={rae:.4f} corr={corr:.4f} smape={smape:.4f}")

if __name__ == "__main__":
    main()
