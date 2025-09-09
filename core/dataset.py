# core/dataset.py
import torch
from util import DataLoaderS  # 기존 코드 재사용

class DataModule:
    """
    util.DataLoaderS 를 감싼 간단 래퍼.
    학습/검증/테스트 텐서, scale 등을 제공한다.
    """
    def __init__(self, path, split_train, split_val, device, horizon, seq_in_len, normalize, seq_out_len):
        self.ds = DataLoaderS(path, split_train, split_val, device, horizon, seq_in_len, normalize, seq_out_len)
        self.scale = self.ds.scale                # (m,)
        self.m = self.ds.m
        self.n = self.ds.n
        self.P = self.ds.P
        self.out_len = self.ds.out_len
        self.adj = self.ds.adj
        self.train = self.ds.train
        self.valid = self.ds.valid
        self.test = self.ds.test
        self.test_window = self.ds.test_window

    def get_batches(self, X, Y, batch_size, shuffle):
        yield from self.ds.get_batches(X, Y, batch_size, shuffle)
