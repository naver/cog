# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

import sys
import os
import logging
from termcolor import colored

from model_loader_dict import MODEL_LOADER_DICT

logger = logging.getLogger()

# Root directory for the pretrained models benchmarked in the paper
# We will re-set this variable through argparse in utils.py/init_program.
MODELS_ROOT_DIR = None


def get_model_dir(model_name):
    """
    Returns the absolute path of model's directory.
    We split model_name into model_title + architecture_name then return MODELS_ROOT_DIR/model_title/architecture_name.
    """

    if MODELS_ROOT_DIR is None:
        print(colored("You wanted to get the model directory for the model {}".format(model_name), "red"))
        print(colored("But MODELS_ROOT_DIR ({}) is not set properly".format(MODELS_ROOT_DIR)))
        print(colored("Please make sure you pass the --models_root_dir argument correctly."))
        sys.exit(-1)

    model_title = model_name.split("_")[0]
    architecture_name = "_".join(model_name.split("_")[1:])

    model_dir = os.path.join(MODELS_ROOT_DIR, model_title, architecture_name)
    return model_dir


def get_model_ckpt(model_name):
    """
    Returns the absolute path of the model ckpt file.
    """
    ckpt_path = os.path.join(get_model_dir(model_name), "model.ckpt")
    return ckpt_path


def get_model_fts_dir(model_name, dataset, split):
    """
    Returns the directory where features for this particular dataset is extracted.
    """
    features_root = os.path.join(
        get_model_dir(model_name),
        dataset,
        "features_{}".format(split)
    )
    return features_root


def get_fts_path(fts_dir):
    """Returns the absolute path of the features file under fts_dir."""
    return os.path.join(fts_dir, "X_Y.pth")


def get_model_fts_paths(model_name, dataset,):
    """
    Returns the paths of training and test features extracted by a model for a particular dataset.
    """

    paths = []
    for split in ("train", "test"):
        paths.append(
            get_fts_path(get_model_fts_dir(model_name, dataset, split)))

    return paths


def get_model_eval_dir(model_name, dataset, cfg):
    """
    Returns the directory where outputs of a particular evaluation are saved.
    """
    # logistic regression classifier
    # - with all available data or
    # - that simulates few-shot setting
    eval_dir = os.path.join(
        get_model_dir(model_name),
        dataset,
        "eval_logreg" if cfg.CLF.N_SHOT == 0 else "eval_logreg-N{}".format(cfg.CLF.N_SHOT),
        "seed_" + str(cfg.EVAL.SEED)
    )

    return eval_dir


def load_pretrained_backbone(model_name, ckpt_file=""):
    """
    Initializes the backbone of the pretrained model, to be used for feature extraction.
    """

    # create a dict of initialization args
    init_args = {
        "model_name": model_name,
        "ckpt_file": ckpt_file,
        # To initialize some of the models we support, we need to load their source codes.
        # And we downloaded codes under <model_dir>s using the prepare_*.sh scripts.
        "model_dir": get_model_dir(model_name) if MODELS_ROOT_DIR else "",
    }

    logger.info("Loading a pretrained backbone...")
    logger.info("model:{}".format(init_args["model_name"]))
    logger.info("checkpoint:{}".format(init_args["ckpt_file"]))
    logger.info("model_dir:{}".format(init_args["model_dir"]))
    backbone, forward = MODEL_LOADER_DICT[model_name](init_args)

    logging.info("Backbone created:")
    logger.info("{}".format(backbone))

    param_count = sum([m.numel() for m in backbone.parameters()])
    logging.info("Parameter count: {:,}".format(param_count))

    logger.info("Moving the backbone to GPU.")
    if model_name != "clip":  # Clip models have their own way of initialization
        backbone = backbone.cuda()
        backbone.eval()

    return backbone, forward
