# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

import sys
import os
import json
import logging
import pickle
import shutil
import random

import numpy as np
import torch as th
from termcolor import colored

import model_utils
import model_loader_dict
import program_logger
from configs.default import get_default_cfg

logger = logging.getLogger()


def none_or_string_flag(argument_value):
    """Checks if the argument value should be None."""
    possible_none_arguments = ("none", "null")

    # try to infer if the string is None or Null
    if argument_value.strip().lower() in possible_none_arguments:
        return None

    # otherwise just return the string as it is
    return argument_value


def is_file(path):
    """Checks whether the path exists and it is a file."""
    return (path is not None) and (path != "") and os.path.isfile(path)


def is_dir(path):
    """Checks whether the path exists and it is a folder."""
    return (path is not None) and (path != "") and os.path.isdir(path)


def save_args(args, file_path=""):
    """
    Prints arguments by logger, and optionally saves them in a JSON file.
    """
    _dict = vars(args)
    _list = sorted(_dict.keys())

    logger.info("Program Arguments:")
    for _k in _list:
        val = _dict[_k]
        if val is not None:
            logger.info("\t{}:{}".format(_k, val))

    if file_path:
        with open(file_path, "w") as fjson:
            json.dump(_dict, fjson, sort_keys=True, indent=4)

    return _dict


def save_cfg(cfg, file_path):
    """
    Prints the config by logger, and saves them into a text file.
    """

    # convert the config into str
    _str = cfg.dump(indent=4)

    logger.info("Program config:")
    logger.info("{}".format(_str))

    with open(file_path, "w") as fid:
        fid.write("{}\n".format(_str))


def save_program_config(args, cfg):
    """
    Dumps all the information regarding the experiment under args.output_dir.
    """
    # save all the arguments into a file
    save_args(args, os.path.join(args.output_dir, "arguments.json"))
    # save all the config into a file
    save_cfg(cfg, os.path.join(args.output_dir, "config.txt"))


def check_model_name(args):
    """Checks if args.model is set properly."""
    if args.model not in model_loader_dict.MODEL_LOADER_DICT:
        print(colored(
            "** --model ({}) is not recognized, i.e., it is not listed in MODEL_LOADER_DICT".format(args.model),
            "red"))
        sys.exit(-1)

    if len(args.model.split("_")) < 2:
        print(colored("** --model ({}) is not recognized".format(args.model), "red"))
        print(colored("   It cannot be split into 2 (or more) pieces separated by underscore(s).", "red"))
        print(colored("   I expect model names to have this form: ", "red"))
        print(colored("   --model=<model_title>_<architecture_name> (notice the underscore!)."))
        sys.exit(-1)


def check_dataset(args, for_feature_extraction=False):
    """Checks if the dataset arguments are set properly."""
    possible_datasets = ("in1k", "cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")
    if args.dataset not in possible_datasets:
        print(colored("** Invalid --dataset ({}).".format(args.dataset), "red"))
        print(colored("   I expect --dataset to take one of {}.".format(possible_datasets), "red"))
        sys.exit(-1)

    # For feature extraction only, we require more arguments to be set
    if for_feature_extraction:
        possible_splits = ("train", "test")
        if args.split not in possible_splits:
            print(colored("** Invalid --split ({}).".format(args.split), "red"))
            print(colored("   I expect --split to be one of {}.".format(possible_splits), "red"))
            sys.exit(-1)

        if args.dataset == "in1k":
            if not is_dir(args.in1k_images_root) or \
               not is_dir(os.path.join(args.in1k_images_root, "train")) or \
               not is_dir(os.path.join(args.in1k_images_root, "val")):
                print(colored("** Invalid path for --in1k_images_root ({})".format(args.in1k_images_root), "red"))
                print(colored("   Either it does not exist or it does not contain 'train' and 'val' directories under.", "red"))
                sys.exit(-1)

        else:
            for cog_path in ("imagenet_images_root", "cog_levels_mapping_file", "cog_concepts_split_file"):
                arg_val = vars(args)[cog_path]
                check_fun = is_file if cog_path.endswith("file") else is_dir
                if not check_fun(arg_val):
                    print(colored("** Invalid path for --{} ({})".format(cog_path, arg_val), "red"))
                    sys.exit(-1)


def check_output_dir(args, cfg=None, for_feature_extraction=False):
    """
    Checks if args.output_dir is set properly.
    If args.output_dir is not provided, but args.models_root_dir is,
    then it sets args.output_dir accordingly.
    """
    # args.output_dir does not have to exist, we will create it anyway.
    if (args.output_dir is None) or (args.output_dir == ""):
        _for_str = "feature extraction" if for_feature_extraction else "logreg"
        if not is_dir(args.models_root_dir):
            print(colored("** Invalid --output_dir ({}) for {}.".format(args.output_dir, _for_str), "red"))
            print(colored("   This is fine, I was going use --models_root_dir to determine the output directory.", "red"))
            print(colored("   But --models_root_dir ({}) is also invalid.".format(args.models_root_dir), "red"))
            print(colored("   So, please read the instructions in the README and the bash script.", "red"))
            print(colored("   Either provide --output_dir or --models_root_dir.", "red"))
            sys.exit(-1)

        print("** Invalid --output_dir ({}) for {}.".format(args.output_dir, _for_str))
        print("   This is fine, I will use --models_root_dir ({}) to determine the output directory.".format(
            args.models_root_dir))
        if for_feature_extraction:
            args.output_dir = model_utils.get_model_fts_dir(args.model, args.dataset, args.split)
        else:
            args.output_dir = model_utils.get_model_eval_dir(args.model, args.dataset, cfg)

    # Set path to where to save features
    if for_feature_extraction:
        args.features_save_path = model_utils.get_fts_path(args.output_dir)


def check_ckpt_file(args):
    """
    Checks if args.ckpt_file is set properly.
    If args.ckpt_file is not provided, but args.models_root_dir is,
    then it sets args.ckpt_file accordingly.
    """
    if not is_file(args.ckpt_file):
        if not is_dir(args.models_root_dir):
            print(colored("** Invalid --ckpt_file ({}) for feature extraction.".format(args.ckpt_file), "red"))
            print(colored("   This is fine, I was going use --models_root_dir to determine the checkpoint file for the model {}.".format(args.model), "red"))
            print(colored("   But --models_root_dir ({}) is also invalid.".format(args.models_root_dir), "red"))
            print(colored("   So, please read the instructions in the README and the bash script.", "red"))
            print(colored("   Either provide --ckpt_file or --models_root_dir.", "red"))
            sys.exit(-1)

        ckpt_file = model_utils.get_model_ckpt(args.model)
        print("** Invalid --ckpt_file ({}) for feature extraction".format(args.ckpt_file))
        print("   This is fine, I will use the checkpoint file {} for the model {} under --models_root_dir".format(
            ckpt_file, args.model))
        args.ckpt_file = ckpt_file


def check_features_dirs(args):
    """Checks if the variables for specifying training and test set features are set properly."""

    # we will set these two paths
    args.train_features_path = None
    args.test_features_path = None

    args_dict = vars(args)

    # check if args.train/test_features_dir are set
    for arg_name in ("train_features_dir", "test_features_dir"):
        arg_val = args_dict[arg_name]
        if is_dir(arg_val):
            fts_path = model_utils.get_fts_path(arg_val)
            if is_file(fts_path):
                args_dict[arg_name.replace("dir", "path")] = fts_path
            else:
                print(colored("** It seems that --{} is set ({})".format(arg_name, arg_val), "red"))
                print(colored("   But there is no file named 'X_Y.pth' under this path.", "red"))
                print(colored("   Make sure that you set --{} correctly.".format(arg_name), "red"))
                sys.exit(-1)

    # otherwise try to locate the features under args.models_root_dir
    if not is_file(args.train_features_path) or not is_file(args.test_features_path):
        print("** Invalid --train_features_dir or --test_features_dir.")
        print("   This is fine, I will use --models_root_dir to determine the features paths.")

        # make sure models_root_dir is set
        if not is_dir(args.models_root_dir):
            print(colored("   But --models_root_dir ({}) is also invalid.".format(args.models_root_dir), "red"))
            sys.exit(-1)

        # make sure model_name is set
        check_model_name(args)

        # make sure dataset is set
        check_dataset(args, for_feature_extraction=False)

        train_test_fts_paths = model_utils.get_model_fts_paths(args.model, args.dataset)
        for fts_path, split in zip(train_test_fts_paths, ("train", "test")):
            if is_file(fts_path):
                args_dict["{}_features_path".format(split)] = fts_path
            else:
                print(colored("** You wanted me to load <{}> set features from {}".format(split, fts_path), "red"))
                print(colored("   But this file does not exist", "red"))
                print(colored("   Make sure that you set --{}_features_dir correctly.".format(split), "red"))
                sys.exit(-1)


def init_program(args, _for="logreg"):
    """
    Initialize the program either for feature extraction or for evaluation.
    Also checks if all the args are valid.
    """
    assert _for in ["ft-extract", "logreg", "results"]

    # load the config file
    cfg = get_default_cfg(args.model, args.opts)

    # set the global models_root_dir
    model_utils.MODELS_ROOT_DIR = args.models_root_dir

    # check if all the remaining arguments set properly
    if _for == "logreg":
        # Check features directory
        # and set args.train_features_path, and args.test_features_path
        check_features_dirs(args)

        # Check or set output directory
        check_output_dir(args, cfg=cfg, for_feature_extraction=False)

    elif _for == "ft-extract":
        # check the model_name argument
        # for evaluation, model_name is optional
        check_model_name(args)

        # Check all data paths.
        check_dataset(args, for_feature_extraction=True)

        # Check or set output directory
        # also set features_save_path
        check_output_dir(args, cfg=None, for_feature_extraction=True)

        # Check model checkpoint file
        check_ckpt_file(args)

    else:  # for "results"
        return args, cfg

    # create the output directory
    make_exp_dir(args.output_dir)

    # init logger
    program_logger.init(os.path.join(args.output_dir, "program.log"))

    # save all the program setup
    save_program_config(args, cfg)

    # fix seed
    random.seed(cfg.EVAL.SEED)
    np.random.seed(cfg.EVAL.SEED)
    th.manual_seed(cfg.EVAL.SEED)

    return args, cfg


def save_pickle(obj, path):
    """
    Save the obj as a pickle file.
    """
    with open(path, "wb") as fid:
        pickle.dump(obj, fid)


def load_pickle(path):
    """
    Loads the obj under path.
    """
    with open(path, "rb") as fid:
        obj = pickle.load(fid)
    return obj


def make_exp_dir(path, purge_first=True):
    """
    Creates an experiment directory.
    If the directory already exists, then purges it first.
    """
    assert path != "", "make_exp_dir received an empty path."

    if purge_first and os.path.exists(path):
        print(colored("The experiment folder {} exists, removing it...".format(path), "red"))
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)

    return path


def write_to_file(str, path):
    """
    Appends the given string into the file pointed by path.
    If the file does not exists, then it creates one.
    """
    with open(path, "a+") as fid:
        fid.write(str)
