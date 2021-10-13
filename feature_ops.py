# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

import logging
from typing import List

import numpy as np
import torch as th

logger = logging.getLogger()


class FeatureSet():
    """
    Store (image feature, label) pairs.
    """
    def __init__(self, x, y, name):
        assert x.shape[0] == y.shape[0]
        self.name = name
        self.x = x
        self.y = y

    def print_info(self):
        logger.info("FeatureSet:{} | x:{}, y:{} | x.norm:{:.3f} +- {:.3f}, x.non_zero:{:.3f} | unique(y):{}".format(
            self.name,
            list(self.x.shape),
            list(self.y.shape),
            self.x.norm(dim=1, p=2).mean(),
            self.x.norm(dim=1, p=2).std(),
            (self.x != 0).float().sum(dim=1).mean(),
            th.unique(self.y).shape[0]
        ))

    def to_gpu(self):
        """
        Move the tensors into GPU.
        """
        self.x = self.x.to("cuda")
        self.y = self.y.to("cuda")


def move_data_to_cuda(data_sets: List[FeatureSet]):
    """
    Moves the data to GPU if we have enough GPU memory.
    """
    mem_1gb = 1024 * 1024 * 1024
    mem_required = sum([np.prod(set.x.shape) for set in data_sets]) * 4 + 2 * mem_1gb  # a float is 4 bytes
    mem_available = th.cuda.get_device_properties(0).total_memory
    mem_available -= th.cuda.memory_cached(0) + th.cuda.memory_allocated(0)

    def in_gb(x):
        return x / mem_1gb

    logger.info("{:.1f}GB GPU memory available, {:.1f}GB required.".format(in_gb(mem_available), in_gb(mem_required)))
    if mem_available > mem_required:
        logger.info("Moving all the data to GPU.")
        for set in data_sets:
            set.to_gpu()
    else:
        logger.info("Not enough space in GPU. Data will stay in CPU memory.")


def load_feature_set(path, name="", normalize=False):
    """
    Loads features from the file (path).
    The file is expected to be saved by torch.save and contain torch.tensors named X and Y.
    normalize: whether to apply l2 normalization or not.
    """
    # load features
    pkl = th.load(path, "cpu")
    X = pkl["X"]
    Y = pkl["Y"]
    name = pkl.get("name", name)
    assert X.shape[0] == Y.shape[0]
    logger.info(f"Features of {name} are loaded.")

    if normalize:
        logger.info("Applying l2-normalization to the features.")
        X = normalize_features(X)

    fset = FeatureSet(X, Y, name)
    return fset


def normalize_features(X):
    """
    L2-Normalizes the features.
    """
    if isinstance(X, np.ndarray):
        norm = np.linalg.norm(X, axis=1)
        X = X / (norm + 1e-5)[:, np.newaxis]
    elif isinstance(X, th.Tensor):
        X = th.nn.functional.normalize(X, p=2, dim=1)
    else:
        raise NotImplementedError("Unknown type:{}".format(type(X)))
    return X


def split_trainset(trainset, p_val=0.2):
    """
    Randomly split the trainin set into train and val.
    Args:
        p_val: percentage of the validation set.
    """

    train_inds = []
    val_inds = []

    for cix in th.unique(trainset.y):
        # samples belonging to this class cix
        inds = th.where(trainset.y == cix)[0]
        # random ordering of these samples
        order = th.randperm(inds.shape[0])
        inds = inds[order]
        # number of validation samples
        n_val = int(inds.shape[0] * p_val)

        # split the indices into train and val
        train_inds.append(inds[:-n_val])
        val_inds.append(inds[-n_val:])

    train_inds = th.cat(train_inds, dim=0)
    val_inds = th.cat(val_inds, dim=0)
    assert len(trainset.y) == len(train_inds) + len(val_inds), \
        "While splitting the training set into (train, val), some samples are ignored."
    assert len(np.intersect1d(train_inds.numpy(), val_inds.numpy())) == 0, \
        "Training and validation sets overlap!"

    x_train = trainset.x[train_inds]
    y_train = trainset.y[train_inds]
    x_val = trainset.x[val_inds]
    y_val = trainset.y[val_inds]

    return (
        FeatureSet(x_train, y_train, "train"),
        FeatureSet(x_val, y_val, "val"),
    )


def make_fewshot_dataset(feature_set, n_shot):
    """
    Randomly select n_shot sample per class from the feature set.
    """
    assert n_shot > 0, f"n_shot ({n_shot}) must be > 0"

    x = []
    y = []
    for cix in th.unique(feature_set.y):
        # samples whose label is cix
        inds = th.where(feature_set.y == cix)[0]
        # random order of samples
        inds = inds[th.randperm(inds.shape[0])]

        # split the validation set
        x.append(feature_set.x[inds[:n_shot]])
        y.append(feature_set.y[inds[:n_shot]])

    x = th.cat(x, dim=0)
    y = th.cat(y, dim=0)
    assert len(x) == len(y), f"Interesting, the length of x ({x.shape}) and y ({y.shape}) do not match."
    assert len(x) == n_shot * len(th.unique(feature_set.y)), "It seems that we didn't sample the same number of samples per class."

    return FeatureSet(x, y, feature_set.name + "_{}-shot".format(n_shot))
