# util_unified.py
import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import csv
from collections import defaultdict

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


class DataLoaderS(object):
    def __init__(
        self,
        file_name,
        train, valid,
        device,
        horizon, window,
        normalize=2, out=1,
        split_mode: str = "strict",
        graph_csv_path: str = "/content/data/graph.csv",
        columns_csv_path: str | None = None,   # None이면 자동 결정
    ):
        assert split_mode in ("strict", "legacy")
        self.P = window
        self.h = horizon
        self.device = device
        self.normalize = 2
        self.out_len = out

        with open(file_name) as fin:
            self.rawdat = np.loadtxt(fin, delimiter='\t')

        # shift for log-diff (현재는 사용 안함)
        self.shift = 0
        self.min_data = np.min(self.rawdat)
        if self.min_data < 0:
            self.shift = (-self.min_data) + 1
        elif self.min_data == 0:
            self.shift = 1

        self.dat = np.zeros(self.rawdat.shape)
        self.diff_dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape

        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n, split_mode)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.test[1].size(1), self.m)
        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        # Graph & columns
        self.adj = self.build_predefined_adj(graph_csv_path, columns_csv_path)
        DataLoaderS.col = self.create_columns(columns_csv_path)

    def _normalized(self, normalize):
        if normalize == 0:
            self.dat = self.rawdat
        elif normalize == 1:
            self.dat = self.rawdat / np.max(self.rawdat)
        elif normalize == 2:
            for i in range(self.m):
                mx = np.max(np.abs(self.rawdat[:, i]))
                self.scale[i] = mx if mx != 0 else 1.0
                self.dat[:, i] = self.rawdat[:, i] / (self.scale[i])
        else:
            self.dat = self.rawdat

    def _difference(self):
        for i in range(1, self.n):
            self.diff_dat[i, :] = np.log(self.rawdat[i, :] + self.shift) - np.log(self.rawdat[i-1, :] + self.shift)
        self.diff_dat[0, :] = self.diff_dat[1, :]

    def _split(self, train, valid, test, mode: str):
        """
        train, valid, test는 인덱스 상한(절댓값)이 아니라 분할 경계 인덱스.
        - strict:  train=[P+h-1, train), valid=[train, valid), test=[valid, n)
        - legacy:  train=[P+h-1, n)     , valid=[train, valid), test=[valid, n)
                    (o_util.py 재현: train이 거의 전체를 포함 → 데이터 누수 위험)
        """
        start_idx = self.P + self.h - 1
        if mode == "strict":
            train_set = range(start_idx, train)
        else:  # legacy
            train_set = range(start_idx, self.n)

        valid_set = range(train, valid)
        test_set  = range(valid, self.n)

        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test  = self._batchify(test_set,  self.h)

        # 테스트 슬라이딩 윈도우(고정 36)
        self.test_window = torch.from_numpy(self.dat[-(36 + self.P):, :])

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n - self.out_len, self.P, self.m))
        Y = torch.zeros((n - self.out_len, self.out_len, self.m))
        for i in range(n - self.out_len):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :, :] = torch.from_numpy(self.dat[idx_set[i]:idx_set[i] + self.out_len, :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        index = torch.randperm(length) if shuffle else torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

    def build_predefined_adj(self, graph_csv_path: str, columns_csv_path: str | None):
        graph = defaultdict(list)
        with open(graph_csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                key_node = row[0]
                adjacent_nodes = [node for node in row[1:] if node]
                graph[key_node].extend(adjacent_nodes)
        print('Graph loaded with', len(graph), 'attacks...')

        # 컬럼 CSV 결정
        if columns_csv_path is None:
            columns_csv_path = '/content/data/sm_data_g.csv'
        with open(columns_csv_path, 'r') as f:
            reader = csv.reader(f)
            col = [c for c in next(reader)]
        print(len(col), 'columns loaded...')

        adj = torch.zeros((len(col), len(col)))
        for i in range(adj.shape[0]):
            if col[i] in graph:
                for j in range(adj.shape[1]):
                    if col[j] in graph[col[i]]:
                        adj[i][j] = 1
                        adj[j][i] = 1
        print('Adjacency created...')
        return adj

    def create_columns(self, columns_csv_path: str | None):
        # 기본 파일명
        if columns_csv_path is None:
            file_name = '/content/data/sm_data_g.csv' if self.m == 123 else '/content/data/data.csv'
        else:
            file_name = columns_csv_path

        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            col = [c for c in next(reader)]
            if 'Date' in col[0]:
                return col[1:]
            return col

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def sym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj

def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader']   = DataLoaderM(data['x_val'],   data['y_val'],   valid_batch_size or batch_size)
    data['test_loader']  = DataLoaderM(data['x_test'],  data['y_test'],  test_batch_size or batch_size)
    data['scaler'] = scaler
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse

def load_node_feature(path):
    with open(path) as fi:
        x = []
        for li in fi:
            li = li.strip().split(",")
            e = [float(t) for t in li[1:]]
            x.append(e)
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    z = torch.tensor((x-mean)/std, dtype=torch.float)
    return z
