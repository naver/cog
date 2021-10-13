# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

import os
from yacs.config import CfgNode as CN

# global config
_C = CN()

# for feature extraction
_C.FT_EXTRACT = CN()
# Spatial size of images.
_C.FT_EXTRACT.IMAGE_RESIZE_SIZE = 224
_C.FT_EXTRACT.IMAGE_CROP_SIZE = 224
# Image normalization prior to extracting features
_C.FT_EXTRACT.IMAGE_NORM = CN()
_C.FT_EXTRACT.IMAGE_NORM.MEAN = [0.485, 0.456, 0.406]
_C.FT_EXTRACT.IMAGE_NORM.STD = [0.229, 0.224, 0.225]
# options for data loaders
_C.FT_EXTRACT.BATCH_SIZE = 128
_C.FT_EXTRACT.N_WORKERS = 8

# For the evaluation protocol
_C.EVAL = CN()
_C.EVAL.NAME = "logreg"  # name of the evaluation, used to create output dirs
_C.EVAL.SEED = 22  # seed set while performing the evaluation

_C.CLF = CN()
_C.CLF.BATCH_SIZE = 1024
_C.CLF.N_EPOCHS = 100  # Number of training epochs
_C.CLF.N_TRIALS = 30  # Number of trials with Optuna
_C.CLF.N_SHOT = 0  # Number of training samples to use
_C.CLF.VAL_PERC = 0.2  # Percentage of the training samples used as a validation set
_C.CLF.NORM_FTS = True  # Whether to normalize image features or not
# training the class projection layer
# with high learning rates and lower weight decays often yields better results
_C.CLF.LR_INTV = (1e-1, 1e2)
_C.CLF.WD_INTV = (1e-12, 1e-4)


def get_default_cfg(model=None, from_args=None):
    """
    Returns the default config.
    """

    # clone the default config
    cfg = _C.clone()

    # check if a model specific config file exists
    if model:
        config_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "{}.yaml".format(model)
        )
        if config_file and os.path.exists(config_file):
            cfg.merge_from_file(config_file)

    # check if additional config options are provided as argument
    if from_args:
        cfg.merge_from_list(from_args)

    # freeze the config not to overwrite any option
    cfg.freeze()

    return cfg
