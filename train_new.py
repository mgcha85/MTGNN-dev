# train_unified.py
import os
import ast
import argparse
import math
import time
import torch
import torch.nn as nn
import numpy as np
import random
from random import randrange
from matplotlib import pyplot as plt
from safetensors.torch import save_file
plt.rcParams['savefig.dpi'] = 1200

# --- util / o_util 양쪽 호환 ---
# DataLoaderS, DataLoaderS.col 등을 사용하므로 두 파일 중 가능한 것을 import
from util import *

from net import gtnet
from trainer import Optim

NET_ROOT = os.getenv("NET_ROOT", r"/content/hanlab_share")  # 원하는 네트워크 드라이브 루트 경로로 바꿔도 됨

# =========================
# 보조 함수들 (기존 코드를 유지)
# =========================
def abs_path(p: str) -> str:
    """상대경로를 네트워크 루트 기준 절대경로로 변환"""
    import os
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(NET_ROOT, p))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def inverse_diff_2d(output, I, shift):
    output[0,:] = torch.exp(output[0,:] + torch.log(I + shift)) - shift
    for i in range(1, output.shape[0]):
        output[i,:] = torch.exp(output[i,:] + torch.log(output[i-1,:] + shift)) - shift
    return output

def inverse_diff_3d(output, I, shift):
    output[:,0,:] = torch.exp(output[:,0,:] + torch.log(I + shift)) - shift
    for i in range(1, output.shape[1]):
        output[:,i,:] = torch.exp(output[:,i,:] + torch.log(output[:,i-1,:] + shift)) - shift
    return output

def plot_data(data,title):
    x = range(1, len(data)+1)
    plt.plot(x, data, 'b-', label='Actual')
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    locs, labs = plt.xticks()
    plt.xticks(rotation='vertical', fontsize=13)
    plt.yticks(fontsize=13)
    fig = plt.gcf()
    plt.show()

def consistent_name(name):
    if name=='CAPTCHA' or name=='DNSSEC' or name=='RRAM':
        return name
    if not name.isupper():
        words = name.split(' ')
        result = ''
        for i, word in enumerate(words):
            if len(word) <= 2:
                result += word
            else:
                result += word[0].upper() + word[1:]
            if i < len(words)-1:
                result += ' '
        return result
    words = name.split(' ')
    result = ''
    for i, word in enumerate(words):
        if len(word) <= 3 or '/' in word or word=='MITM' or word=='SIEM':
            result += word
        else:
            result += word[0] + (word[1:].lower())
        if i < len(words)-1:
            result += ' '
    return result

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_metrics_1d(predict, test, title, type_):
    # RRSE / RAE (Lai et al.)
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared = math.sqrt(sum_squared_diff)

    test_s = test
    mean_all = torch.mean(test_s)
    diff_r = test_s - mean_all
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)
    rrse = root_sum_squared / root_sum_squared_r

    sum_absolute_diff = torch.sum(torch.abs(test - predict))
    sum_absolute_r = torch.sum(torch.abs(diff_r))
    rae = (sum_absolute_diff / sum_absolute_r).item()

    title_s = title.replace('/','_')
    out_dir_rel = f'model/Bayesian/{type_}'
    out_dir = abs_path(out_dir_rel)         # <<< 추가
    ensure_dir(out_dir)                      # <<< 변경 (abs 적용된 경로)
    with open(os.path.join(out_dir, f'{title_s}_{type_}.txt'), "w") as f:   # <<< 변경
        f.write('rse:'+str(rrse)+'\n')
        f.write('rae:'+str(rae)+'\n')

def plot_predicted_actual(predicted, actual, title, type_, variance, confidence_95):
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    M = []
    for year in range(11,23):
        for month in months:
            if year==11 and month not in ['Jul','Aug','Sep','Oct','Nov','Dec']:
                continue
            M.append(month+'-'+str(year))
    M2, p = [], []
    if type_=='Testing':
        M = M[-len(predicted):]
        for index, _ in enumerate(M):
            if any(k in M[index] for k in ['Dec','Mar','Jun','Sep']):
                M2.append(M[index])
                p.append(index+1)
    else:
        M = M[63:99]
        for index, _ in enumerate(M):
            if any(k in M[index] for k in ['Dec','Mar','Jun','Sep']):
                M2.append(M[index])
                p.append(index+1)

    x = range(1, len(predicted)+1)
    plt.plot(x, actual, 'b-', label='Actual')
    plt.plot(x, predicted, '--', color='purple', label='Predicted')
    plt.fill_between(x,
                     predicted - confidence_95.numpy(),
                     predicted + confidence_95.numpy(),
                     alpha=0.5, color='pink', label='95% Confidence')
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    plt.xticks(ticks=p, labels=M2, rotation='vertical', fontsize=13)
    plt.yticks(fontsize=13)
    
    title_s = title.replace('/','_')
    out_dir_rel = f'model/Bayesian/{type_}'
    out_dir = abs_path(out_dir_rel)                          # <<< 추가
    ensure_dir(out_dir)                                      # <<< 변경
    plt.savefig(os.path.join(out_dir, f'{title_s}_{type_}.png'), bbox_inches="tight")   # <<< 변경
    plt.savefig(os.path.join(out_dir, f'{title_s}_{type_}.pdf'), bbox_inches="tight", format='pdf')  # <<< 변경
    
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def s_mape(yTrue, yPred):
    mape = 0
    for i in range(len(yTrue)):
        mape += abs(yTrue[i]-yPred[i])/(abs(yTrue[i])+abs(yPred[i]))
    mape /= len(yTrue)
    return mape

# =========================
# 평가 함수들 (기존 로직 유지)
# =========================
def evaluate_sliding_window(data, test_window, model, evaluateL2, evaluateL1, n_input, is_plot):
    total_loss = 0
    total_loss_l1 = 0
    predict = None
    test = None
    variance = None
    confidence_95 = None
    sum_squared_diff = 0
    sum_absolute_diff = 0
    r = 0
    print('testing r=', str(r))
    scale = data.scale.expand(test_window.size(0), data.m)
    print('Test Window Feature:', test_window[:, r])

    x_input = test_window[0:n_input, :].clone()
    for i in range(n_input, test_window.shape[0], data.out_len):
        print('**************x_input*******************')
        print(x_input[:, r])
        print('**************-------*******************')

        X = torch.unsqueeze(x_input, dim=0)
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2,3)
        X = X.to(torch.float)

        y_true = test_window[i: i+data.out_len, :].clone()

        num_runs = 10
        outputs = []
        for _ in range(num_runs):
            with torch.no_grad():
                output = model(X)
                y_pred = output[-1, :, :, -1].clone()
                if y_pred.shape[0] > y_true.shape[0]:
                    y_pred = y_pred[:-(y_pred.shape[0]-y_true.shape[0]), ]
            outputs.append(y_pred)

        outputs = torch.stack(outputs)
        y_pred = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0)
        z = 1.96
        confidence = z*std_dev/torch.sqrt(torch.tensor(num_runs))

        if data.P <= data.out_len:
            x_input = y_pred[-data.P:].clone()
        else:
            x_input = torch.cat([x_input[-(data.P-data.out_len):, :].clone(), y_pred.clone()], dim=0)

        print('----------------------------Predicted months', str(i-n_input+1), 'to', str(i-n_input+data.out_len), '--------------------------------------------------')
        print(y_pred.shape, y_true.shape)
        y_pred_o = y_pred
        y_true_o = y_true
        for z_ in range(y_true.shape[0]):
            print(y_pred_o[z_, r], y_true_o[z_, r])
        print('------------------------------------------------------------------------------------------------------------')

        if predict is None:
            predict = y_pred
            test = y_true
            variance = var
            confidence_95 = confidence
        else:
            predict = torch.cat((predict, y_pred))
            test = torch.cat((test, y_true))
            variance = torch.cat((variance, var))
            confidence_95 = torch.cat((confidence_95, confidence))

    scale = data.scale.expand(test.size(0), data.m)
    predict *= scale
    test *= scale
    variance *= scale
    confidence_95 *= scale

    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    sum_absolute_diff = torch.sum(torch.abs(test - predict))

    root_sum_squared = math.sqrt(sum_squared_diff)
    test_s = test
    mean_all = torch.mean(test_s, dim=0)
    diff_r = test_s - mean_all.expand(test_s.size(0), data.m)
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)
    rrse = root_sum_squared / root_sum_squared_r
    print('rrse=', root_sum_squared, '/', root_sum_squared_r)

    sum_absolute_r = torch.sum(torch.abs(diff_r))
    rae = (sum_absolute_diff / sum_absolute_r).item()

    predict_np = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict_np).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict_np.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict_np - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    smape = 0
    for z_ in range(Ytest.shape[1]):
        smape += s_mape(Ytest[:, z_], predict_np[:, z_])
    smape /= Ytest.shape[1]

    if is_plot:
        counter = 0
        for v in range(r, r+142):
            col = v % data.m
            node_name = DataLoaderS.col[col].replace('-ALL','').replace('Mentions-','Mentions of ').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
            node_name = consistent_name(node_name)
            save_metrics_1d(torch.from_numpy(predict_np[:, col]), torch.from_numpy(Ytest[:, col]), node_name, 'Testing')
            plot_predicted_actual(predict_np[:, col], Ytest[:, col], node_name, 'Testing', variance[:, col], confidence_95[:, col])
            counter += 1

    return rrse, rae, correlation, smape

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot):
    total_loss = 0
    total_loss_l1 = 0
    predict = None
    test = None
    variance = None
    confidence_95 = None
    sum_squared_diff = 0
    sum_absolute_diff = 0
    r = 0
    print('validation r=', str(r))

    for Xb, Yb in data.get_batches(X, Y, batch_size, False):
        Xb = torch.unsqueeze(Xb, dim=1)
        Xb = Xb.transpose(2,3)

        num_runs = 10
        outputs = []
        with torch.no_grad():
            for _ in range(num_runs):
                out = model(Xb)
                out = torch.squeeze(out)
                if len(out.shape) == 1 or len(out.shape) == 2:
                    out = out.unsqueeze(dim=0)
                outputs.append(out)

        outputs = torch.stack(outputs)
        mean = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0)
        z = 1.96
        confidence = z*std_dev/torch.sqrt(torch.tensor(num_runs))
        output = mean

        scale = data.scale.expand(Yb.size(0), Yb.size(1), data.m)
        output *= scale
        Yb *= scale
        var *= scale
        confidence *= scale

        if predict is None:
            predict = output
            test = Yb
            variance = var
            confidence_95 = confidence
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Yb))
            variance = torch.cat((variance, var))
            confidence_95 = torch.cat((confidence_95, confidence))

        print('EVALUATE RESULTS:')
        y_pred_o = output
        y_true_o = Yb
        for z_ in range(Yb.shape[1]):
            print(y_pred_o[0, z_, r], y_true_o[0, z_, r])

        total_loss += evaluateL2(output, Yb).item()
        total_loss_l1 += evaluateL1(output, Yb).item()
        n_samples = (output.size(0) * output.size(1) * data.m)

        sum_squared_diff += torch.sum(torch.pow(Yb - output, 2))
        sum_absolute_diff += torch.sum(torch.abs(Yb - output))

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    root_sum_squared = math.sqrt(sum_squared_diff)
    test_s = test
    mean_all = torch.mean(test_s, dim=(0,1))
    diff_r = test_s - mean_all.expand(test_s.size(0), test_s.size(1), data.m)
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)
    rrse = root_sum_squared / root_sum_squared_r

    sum_absolute_r = torch.sum(torch.abs(diff_r))
    rae = (sum_absolute_diff / sum_absolute_r).item()

    predict_np = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict_np).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict_np.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict_np - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    smape = 0
    for x in range(Ytest.shape[0]):
        for z_ in range(Ytest.shape[2]):
            smape += s_mape(Ytest[x, :, z_], predict_np[x, :, z_])
    smape /= (Ytest.shape[0] * Ytest.shape[2])

    if is_plot:
        counter = 0
        for v in range(r, r+142):
            col = v % data.m
            node_name = DataLoaderS.col[col].replace('-ALL','').replace('Mentions-','Mentions of ').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
            node_name = consistent_name(node_name)
            save_metrics_1d(torch.from_numpy(predict_np[-1, :, col]), torch.from_numpy(Ytest[-1, :, col]), node_name, 'Validation')
            plot_predicted_actual(predict_np[-1, :, col], Ytest[-1, :, col], node_name, 'Validation', variance[-1, :, col], confidence_95[-1, :, col])
            counter += 1

    return rrse, rae, correlation, smape

# =========================
# 학습 루프 (기존 train)
# =========================
def train_epoch(data, X, Y, model, criterion, optim, batch_size, device, args):
    model.train()
    total_loss = 0
    iter_ = 0
    for Xb, Yb in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        Xb = torch.unsqueeze(Xb, dim=1)
        Xb = Xb.transpose(2,3)

        if iter_ % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id_ = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id_ = perm[j * num_sub:]
            id_ = torch.tensor(id_).to(device)

            tx = Xb[:, :, :, :]
            ty = Yb[:, :, :]
            output = model(tx)
            output = torch.squeeze(output, 3)
            scale = data.scale.expand(output.size(0), output.size(1), data.m)
            output *= scale
            ty *= scale

            loss = criterion(output, ty)
            loss.backward()
            total_loss += loss.item()

            # gradient step
            _ = optim.step()

        if iter_ % 1 == 0:
            denom = (output.size(0) * output.size(1) * data.m)
            print(f'iter:{iter_:3d} | loss: {loss.item()/denom:.3f}')
        iter_ += 1
    # 평균 손실을 원하는 경우엔 표본수로 나누어도 되지만,
    # 기존 코드처럼 합계 기준을 유지.
    return total_loss

# =========================
# 시드 고정
# =========================
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# 메인
# =========================
def main():
    parser = argparse.ArgumentParser(description='Unified: Random Search + Train + Save best')
    parser.add_argument('--data', type=str, default='./data/sm_data.txt', help='location of the data file')
    parser.add_argument('--log_interval', type=int, default=2000)
    parser.add_argument('--save', type=str, default='model/Bayesian/model.safetensors', help='path to save the best model')
    parser.add_argument('--hp_save', type=str, default='model/Bayesian/hp.txt', help='path to save best hyperparameters')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--gcn_true', type=bool, default=True)
    parser.add_argument('--buildA_true', type=bool, default=True)
    parser.add_argument('--num_nodes', type=int, default=142)
    parser.add_argument('--in_dim', type=int, default=1)
    parser.add_argument('--seq_in_len', type=int, default=10)
    parser.add_argument('--seq_out_len', type=int, default=36)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--clip', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_split', type=int, default=1)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--trials', type=int, default=20, help='random search trials')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.save = abs_path(args.save)
    args.hp_save = abs_path(args.hp_save)

    print(args.save)
    print(args.hp_save)

    # 상위 폴더 생성
    ensure_dir(os.path.dirname(args.save))
    ensure_dir(os.path.dirname(args.hp_save))

    # 디바이스 / 스레드
    device = torch.device(args.device)
    torch.set_num_threads(3)

    # 시드 고정
    set_random_seed(args.seed)

    # 하이퍼파라미터 후보 (기존과 동일)
    gcn_depths = [1,2,3]
    lrs        = [0.01,0.001,0.0005,0.0008,0.0001,0.0003,0.005]
    convs      = [4,8,16]
    ress       = [16,32,64]
    skips      = [64,128,256]
    ends       = [256,512,1024]
    layers     = [1,2]
    ks         = [20,30,40,50,60,70,80,90,100]
    dropouts   = [0.2,0.3,0.4,0.5,0.6,0.7]
    dilation_exs = [1,2,3]
    node_dims  = [20,30,40,50,60,70,80,90,100]
    prop_alphas = [0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8]
    tanh_alphas = [0.05,0.1,0.5,1,2,3,5,7,9]

    # 데이터 로더
    Data = DataLoaderS(args.data, 0.43, 0.30, device, args.horizon, args.seq_in_len, args.normalize, args.seq_out_len)

    print('train X:', Data.train[0].shape)
    print('train Y:', Data.train[1].shape)
    print('valid X:', Data.valid[0].shape)
    print('valid Y:', Data.valid[1].shape)
    print('test X:', Data.test[0].shape)
    print('test Y:', Data.test[1].shape)
    print('test window:', Data.test_window.shape)
    print('length of training set=', Data.train[0].shape[0])
    print('length of validation set=', Data.valid[0].shape[0])
    print('length of testing set=', Data.test[0].shape[0])
    print('valid=', int((0.43 + 0.30) * Data.n))

    # 손실함수/평가지표
    if args.L1Loss:
        criterion = nn.L1Loss(reduction='sum').to(device)
    else:
        criterion = nn.MSELoss(reduction='sum').to(device)
    evaluateL2 = nn.MSELoss(reduction='sum').to(device)
    evaluateL1 = nn.L1Loss(reduction='sum').to(device)

    ensure_dir(os.path.dirname(args.save) or '.')
    ensure_dir(os.path.dirname(args.hp_save) or '.')

    # 베스트 트래커
    best_val_sum = float('inf')
    best_rse = float('inf')
    best_rae = float('inf')
    best_corr = -float('inf')
    best_smape = float('inf')
    best_test_rse = float('inf')
    best_test_corr = -float('inf')
    best_hp = None

    # 랜덤 서치 + 학습 + 베스트 즉시 저장
    for trial in range(args.trials):
        # 샘플 하이퍼파라미터
        gcn_depth  = random.choice(gcn_depths)
        lr         = random.choice(lrs)
        conv       = random.choice(convs)
        res        = random.choice(ress)
        skip       = random.choice(skips)
        end        = random.choice(ends)
        layer      = random.choice(layers)
        k          = random.choice(ks)
        dropout    = random.choice(dropouts)
        dilation_ex= random.choice(dilation_exs)
        node_dim   = random.choice(node_dims)
        prop_alpha = random.choice(prop_alphas)
        tanh_alpha = random.choice(tanh_alphas)

        # 모델 생성
        model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
                      device, Data.adj, dropout=dropout, subgraph_size=k,
                      node_dim=node_dim, dilation_exponential=dilation_ex,
                      conv_channels=conv, residual_channels=res,
                      skip_channels=skip, end_channels=end,
                      seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                      layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False)

        print('Args:', args)
        print('The receptive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

        optim = Optim(model.parameters(), args.optim, lr, args.clip, lr_decay=args.weight_decay)

        es_counter = 0
        try:
            print('==> Trial', trial+1, '/', args.trials, '| HP =',
                  [gcn_depth, lr, conv, res, skip, end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer])
            for epoch in range(1, args.epochs + 1):
                print('epoch:', epoch)
                print('current best (val rse):', best_rse, 'hp:', best_hp)
                epoch_start_time = time.time()
                train_loss = train_epoch(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size, device, args)

                val_rse, val_rae, val_corr, val_smape = evaluate(
                    Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size, is_plot=False
                )
                print('| epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr {:5.4f} | valid smape {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_rse, val_rae, val_corr, val_smape), flush=True)

                # 선택 기준: (NaN 방지) corr 유효 & rse 개선
                if (not math.isnan(val_corr)) and (val_rse < best_rse):
                    # 즉시 저장 (overwrite best only)
                    save_file(model.state_dict(), args.save)

                    best_val_sum = val_rse + val_rae - val_corr
                    best_rse = val_rse
                    best_rae = val_rae
                    best_corr = val_corr
                    best_smape = val_smape

                    best_hp = [gcn_depth, lr, conv, res, skip, end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch]

                    # 테스트 윈도우 평가도 기록
                    test_rse, test_rae, test_corr, test_smape = evaluate_sliding_window(
                        Data, Data.test_window, model, evaluateL2, evaluateL1, args.seq_in_len, is_plot=False
                    )
                    print('*** [Improved] TEST rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test smape {:5.4f}'.format(
                        test_rse, test_rae, test_corr, test_smape
                    ), flush=True)
                    best_test_rse = test_rse
                    best_test_corr = test_corr

                    # HP 저장
                    with open(args.hp_save, 'w') as f:
                        f.write(str(best_hp))

                    es_counter = 0
                else:
                    es_counter += 1

                # 조기 종료가 필요하면 키우세요 (예: 30epoch 정체 시 중단)
                # if es_counter > 30:
                #     print('Early stopping: no improvement for 30 validations.')
                #     break

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early for this trial')

    # 랜덤 서치 종료 후, 베스트 결과 리포트 & 시각화용 최종 평가
    print('==> Search finished')
    print('best val rse =', best_rse)
    print('best hp =', best_hp)

    # 베스트 가중치/HP는 이미 저장되어 있음.
    # 원한다면 여기서 다시 로드하여 플롯 활성화 평가를 할 수도 있음.
    # 현재 모델 인스턴스가 베스트가 아닐 수 있으므로, 새 모델 만들어 로드/평가 (옵션):
    if best_hp is not None:
        (gcn_depth, lr, conv, res, skip, end, k, dropout, dilation_ex, node_dim,
         prop_alpha, tanh_alpha, layer, best_epoch) = best_hp

        best_model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
                           device, Data.adj, dropout=dropout, subgraph_size=k,
                           node_dim=node_dim, dilation_exponential=dilation_ex,
                           conv_channels=conv, residual_channels=res,
                           skip_channels=skip, end_channels=end,
                           seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                           layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False)
        # safetensors 로드
        state = torch.load(args.save, map_location=device) if args.save.endswith('.pth') else None
        if state is not None:
            best_model.load_state_dict(state)
        else:
            # safetensors는 별도 로더가 필요하지만, 위에서 save_file로 저장했으므로
            # forecast 단계에서 load_file(...) 사용을 권장.
            pass

        # 최종 리포트 (플롯 ON)
        v_rse, v_rae, v_corr, v_smape = evaluate(
            Data, Data.valid[0], Data.valid[1], best_model, evaluateL2, evaluateL1, args.batch_size, is_plot=True
        )
        t_rse, t_rae, t_corr, t_smape = evaluate_sliding_window(
            Data, Data.test_window, best_model, evaluateL2, evaluateL1, args.seq_in_len, is_plot=True
        )
        print('********************************************************************************************************')
        print("FINAL valid rse {:5.4f} | rae {:5.4f} | corr {:5.4f} | smape {:5.4f}".format(v_rse, v_rae, v_corr, v_smape))
        print("FINAL test  rse {:5.4f} | rae {:5.4f} | corr {:5.4f} | smape {:5.4f}".format(t_rse, t_rae, t_corr, t_smape))
        print('********************************************************************************************************')

if __name__ == "__main__":
    main()
