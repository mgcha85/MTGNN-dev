# core/trainer.py
import math
import torch
import torch.nn as nn
from trainer import Optim  # 기존 프로젝트의 Optim 사용

class Trainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def _criterion(self, L1=True):
        return (nn.L1Loss(reduction='sum') if L1 else nn.MSELoss(reduction='sum')).to(self.device)

    def fit_one_epoch(self, data, model, criterion, optim, batch_size, step_size, num_split, clip):
        model.train()
        total_loss = 0.0
        n_samples = 0
        it = 0

        for X, Y in data.get_batches(data.train[0], data.train[1], batch_size, True):
            model.zero_grad()
            X = X.unsqueeze(1).transpose(2, 3)  # (B,1,M,T)
            if it % step_size == 0:
                perm = torch.randperm(self.cfg.data.num_nodes)
            num_sub = int(self.cfg.data.num_nodes / num_split) if num_split > 0 else self.cfg.data.num_nodes

            id = perm  # 원 코드에서 id는 쓰지 않는 형태
            tx = X[:, :, :, :]
            ty = Y[:, :, :]      # (B,T,M)
            output = model(tx)   # (B, out_dim, M, ?)
            output = torch.squeeze(output, 3)  # -> (B, out_dim, M)

            scale = data.scale.expand(output.size(0), output.size(1), data.m)
            output = output * scale
            ty = ty * scale

            loss = criterion(output, ty)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * output.size(1) * data.m)

            # grad clip은 Optim 내부에서 처리(기존 코드) / 필요시 여기서도 가능
            optim.step()

            it += 1

        return total_loss / max(n_samples, 1)

    def fit(self, data, model):
        crit = self._criterion(self.cfg.train.L1Loss)
        evalL2 = nn.MSELoss(reduction='sum').to(self.device)
        evalL1 = nn.L1Loss(reduction='sum').to(self.device)

        optim = Optim(
            model.parameters(),
            self.cfg.train.optim,
            self.cfg.train.lr,
            self.cfg.train.clip,
            lr_decay=self.cfg.train.weight_decay
        )

        return crit, evalL2, evalL1, optim
