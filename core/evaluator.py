# core/evaluator.py
import math
import torch
import numpy as np
from core.metrics import rrse as rrse_fn, rae as rae_fn, pearson_corr, s_mape

class Evaluator:
    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def evaluate(self, data, X, Y, model, evalL2, evalL1, batch_size, is_plot=False):
        model.eval()
        total_loss = 0.0
        total_loss_l1 = 0.0
        n_samples = 0

        predict = None
        test = None
        variance = None
        conf95 = None
        r = 0

        for Xb, Yb in data.get_batches(X, Y, batch_size, False):
            Xb = Xb.unsqueeze(1).transpose(2, 3)

            # MC 반복 대신 1회 추론(평가에서는 deterministic)
            output = model(Xb)         # (B, out_len, M, 1 or ?)
            output = torch.squeeze(output)
            if output.ndim < 3:
                output = output.unsqueeze(0)  # (B, T, M)

            scale = data.scale.expand(Yb.size(0), Yb.size(1), data.m)
            output = output * scale
            Yb = Yb * scale

            if predict is None:
                predict = output
                test = Yb
                variance = torch.zeros_like(output)
                conf95 = torch.zeros_like(output)
            else:
                predict = torch.cat((predict, output))
                test = torch.cat((test, Yb))
                variance = torch.cat((variance, torch.zeros_like(output)))
                conf95 = torch.cat((conf95, torch.zeros_like(output)))

            total_loss += evalL2(output, Yb).item()
            total_loss_l1 += evalL1(output, Yb).item()
            n_samples += (output.size(0) * output.size(1) * data.m)

        rse = math.sqrt(total_loss / max(n_samples, 1)) / getattr(data.ds, "rse", 1.0)
        rae = (total_loss_l1 / max(n_samples, 1)) / getattr(data.ds, "rae", 1.0)

        pred_np = predict.detach().cpu().numpy()
        test_np = test.detach().cpu().numpy()
        corr = pearson_corr(pred_np, test_np)

        # sMAPE
        smape = 0.0
        B, T, M = pred_np.shape
        for x in range(B):
            smape += s_mape(test_np[x], pred_np[x])
        smape /= max(B, 1)

        return rse, rae, corr, smape

    @torch.no_grad()
    def mc_dropout_forecast(self, model, X, out_len, num_runs=10):
        """
        X: (1,1,M,P) 형태. 모델 아키텍처에 맞춰 처리.
        out: 평균, 분산, 95% CI
        """
        model.train()  # Dropout 활성화
        outs = []
        for _ in range(num_runs):
            out = model(X)  # (B, T_out, M, 1 or ?)
            out = out[-1, :, :, -1] if out.ndim == 4 else out.squeeze()  # (T_out, M)
            outs.append(out)
        outs = torch.stack(outs, dim=0)  # (N, T, M)
        mean = outs.mean(dim=0)
        var = outs.var(dim=0)
        std = outs.std(dim=0)
        z = 1.96
        conf = z * std / math.sqrt(num_runs)
        return mean, var, conf
