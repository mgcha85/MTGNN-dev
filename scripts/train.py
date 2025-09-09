# scripts/train.py
import argparse
import torch
from net import gtnet                 # 기존 net.py의 gtnet 사용
from core.config import Cfg
from core.utils import resolve_devices, set_random_seed
from core.dataset import DataModule
from core.trainer import Trainer
from core.evaluator import Evaluator
from core.checkpoint import save_ckpt

def parse_args():
    p = argparse.ArgumentParser()
    # 데이터/모델/학습 설정은 기존 인자와 맵핑하거나 기본 Cfg를 그대로 써도 됨
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Cfg()  # 필요시 argparse 값으로 override 가능
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
        dropout=cfg.model.dropout if hasattr(cfg.model, "dropout") else 0.3,
        subgraph_size=cfg.model.subgraph_size,
        node_dim=cfg.model.node_dim,
        dilation_exponential=cfg.model.dilation_exponential,
        conv_channels=cfg.model.conv_channels,
        residual_channels=cfg.model.residual_channels,
        skip_channels=cfg.model.skip_channels,
        end_channels=cfg.model.end_channels,
        seq_length=cfg.data.seq_in,
        in_dim=cfg.model.in_dim,
        out_dim=cfg.model.out_dim,
        layers=cfg.model.layers,
        propalpha=cfg.model.propalpha,
        tanhalpha=cfg.model.tanhalpha,
        layer_norm_affline=cfg.model.layer_norm_affline
    ).to(torch_device)

    trainer = Trainer(cfg, torch_device)
    evaluator = Evaluator(torch_device)
    crit, evalL2, evalL1, optim = trainer.fit(data, model)

    best_val = float("inf")
    best_tuple = None

    for epoch in range(cfg.train.epochs):
        train_loss = trainer.fit_one_epoch(
            data, model, crit, optim,
            cfg.train.batch_size, cfg.train.step_size,
            cfg.train.num_split, cfg.train.clip
        )
        val_rse, val_rae, val_corr, val_smape = evaluator.evaluate(
            data, data.valid[0], data.valid[1], model, evalL2, evalL1, cfg.train.batch_size, is_plot=False
        )
        score = val_rse  # 기준
        print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f} | val_rse={val_rse:.4f} rae={val_rae:.4f} corr={val_corr:.4f} smape={val_smape:.4f}")

        if score < best_val:
            best_val = score
            best_tuple = (val_rse, val_rae, val_corr, val_smape)
            save_ckpt(model, cfg.save.ckpt_path)
            # (선택) hp 기록
            with open(cfg.save.hp_path, "w") as f:
                hp = [
                    cfg.model.gcn_depth, cfg.train.lr, cfg.model.conv_channels,
                    cfg.model.residual_channels, cfg.model.skip_channels, cfg.model.end_channels,
                    cfg.model.subgraph_size, getattr(cfg.model, "dropout", 0.3),
                    cfg.model.dilation_exponential, cfg.model.node_dim,
                    cfg.model.propalpha, cfg.model.tanhalpha,
                    cfg.model.layers, epoch+1
                ]
                f.write(str(hp))

    print("Best (val):", best_tuple)

if __name__ == "__main__":
    main()
