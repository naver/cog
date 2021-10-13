# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

import logging
import math

import numpy as np
import torch

logger = logging.getLogger()


class TorchIterator:
    """
    Iterator for list of tensors whose first dimension match.
    """

    def __init__(self, tensors, batch_size, shuffle=True):
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = tensors[0].device

        # number of elements in each tensor should be equal and bigger than batch size
        n_elems = [len(t) for t in tensors]
        assert np.all(np.equal(n_elems, n_elems[0]))
        self.n_sample = n_elems[0]
        if self.n_sample < batch_size:
            logger.info(
                "Length of tensors ({}) given to TorchIterator is less than batch size ({}). "
                "Reducing the batch size to {}".format(
                    self.n_sample, batch_size, self.n_sample
                )
            )
            self.batch_size = self.n_sample

        self._s_ix = (
            0  # index of sample that will be fetched as the first sample in next_batch
        )
        self._order = torch.zeros(
            self.n_sample, dtype=torch.long, device=self.device
        )  # order of samples fetched in an epoch
        self.reset_batch_order()

    def __len__(self):
        return math.ceil(self.n_sample / self.batch_size)

    def __iter__(self):
        return self

    def _check_new_epoch(self):
        # check whether there is no not-fetched sample left
        return self._s_ix >= self.n_sample

    def reset_batch_order(self):
        self._s_ix = 0
        if self.shuffle:
            torch.randperm(self.n_sample, out=self._order)
        else:
            torch.arange(self.n_sample, out=self._order)

    def __next__(self):
        if self._check_new_epoch():
            self.reset_batch_order()
            raise StopIteration

        inds = self._order[self._s_ix : self._s_ix + self.batch_size]
        self._s_ix += len(inds)
        batch = [t[inds] for t in self.tensors]
        return batch
