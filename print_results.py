# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

import sys
import os
import argparse
import glob
from pathlib import Path

import numpy as np
from termcolor import colored

import utils
import model_utils


parser = argparse.ArgumentParser()
# Specify the logreg experiment folder directly
parser.add_argument('--logreg_root_dir', type=utils.none_or_string_flag,
                    help='Root directory for logistic regression experiments with different seeds.'
                         'Under this folder should be seed-* folders for each experiment')
# Determine the logreg experiment folder based on model name, the evaluation type (many-shot vs few-shot) and dataset.
parser.add_argument('--model', type=utils.none_or_string_flag,
                    help='Name of the model in the <model_title>_<architecture_name> form.'
                         'See the table of models in ./prepare_models/README.md for all the model names we support.'
                         'This is an optional argument that needs to be set along with --models_root_dir and --dataset.'
                         'When these three arguments are set, the script will load the results of the experiments under:'
                         '<models_root_dir>/<model_title>/<architecture_name>/<dataset>/eval_logreg*/.'
                         'If you would like to load the results from a folder directly'
                         'then provide --logreg_root_dir.')
parser.add_argument('--dataset', type=utils.none_or_string_flag,
                    help='Name of the dataset.'
                         'Possible values are ("in1k", "cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")'
                         'This is an optional argument that needs to be set along with --models_root_dir and --model.'
                         'Please see the help message for the --model argument as well.')
parser.add_argument('--models_root_dir', type=utils.none_or_string_flag,
                    help='Root directory for all models, see prepare_models/README.md for a detailed explanation.'
                         'This is an optional argument that needs to be set along with --model and --dataset.'
                         'Please see the help message for the --model argument as well.')
parser.add_argument('opts', default=None,
                    help='see configs/default.py for all options',
                    nargs=argparse.REMAINDER)


def main():
    args = parser.parse_args()

    # check which argument is set
    if utils.is_dir(args.logreg_root_dir):
        logreg_root_dir = args.logreg_root_dir

    elif utils.is_dir(args.models_root_dir):
        args, cfg = utils.init_program(args, _for="results")
        logreg_root_dir = Path(
            model_utils.get_model_eval_dir(args.model, args.dataset, cfg)).parent.absolute()

    else:
        print(colored("** Please provide either --logreg_root_dir or --models_root_dir.", "red"))
        print(colored("   Both of them are invalid.", "red"))
        sys.exit(-1)

    # find seed dirs
    seed_dirs = sorted(glob.glob(
        os.path.join(
            logreg_root_dir,
            "seed*"
        )))

    if len(seed_dirs) == 0:
        print(colored("I was expecting folders for experiments with different seeds under:", "red"))
        print(colored(" ==> <logreg_root_dir>: {}".format(logreg_root_dir), "red"))
        print(colored("But I couldn't find any :/", "red"))
        print(colored("Please make sure the path is correct.", "red"))
        sys.exit(-1)

    # which logs to load
    score_keys = ("test/top1", "test/top5")

    # we will keep 2D array of scores
    scores = None

    for sdir in sorted(seed_dirs):
        print("Seed directory: {}".format(sdir))
        _scores = load_logreg_logs(sdir, score_keys)

        # if we couldn't load logs
        if _scores is None:
            print(colored(" ==> I couldn't load the scores for the experiment under {}".format(sdir), "red"))
            continue

        if scores is None:
            scores = np.copy(_scores)
        else:
            scores = np.vstack([scores, _scores])

    print("")
    print("Combined scores:")
    print(scores)
    if scores is None:
        print(colored("Sorry, I couldn't load any results. :/", "red"))
        print(colored("Please check if the experiments finished peacefully.", "red"))
        sys.exit(-1)

    print("")
    print("We managed to load the logs of {} experiments".format(scores.shape[0]))
    mean = scores.mean(axis=0)
    std = scores.std(axis=0)
    for kix, key in enumerate(score_keys):
        print(" ==> {:s}: {:.1f} +- {:.1f}".format(key, mean[kix], std[kix]))


def load_logreg_logs(exp_dir, score_keys, arrlen_check=100):
    """Returns top-1 and top-5 accuracies obtained at the end of logistic regression training"""

    # we load the results of the final classifier trained on training+val set
    log_file = glob.glob(
        os.path.join(
            exp_dir,
            "final*/logs.pkl"
        )
    )
    # we expect only one log file to be found
    if len(log_file) >= 2:
        print(colored("Something is off: Found {} final*/logs.pkl files under {}".format(
            len(log_file), exp_dir), "red"))
        print(colored("You probably forgot to delete old experiments.", "red"))
        print(colored("I'd suggest you to delete everything under {},".format(exp_dir), "red"))
        print(colored("then re-train the classifier for this seed.", "red"))
        return None
    elif len(log_file) == 0:
        print(colored("No final*/logs.pkl file is found under {}".format(exp_dir), "red"))
        print(colored("Probably the experiment hasn't finished yet (or it crashed, sorry :/)", "red"))
        return None

    logs = utils.load_pickle(log_file[0])
    scores = []

    for key in score_keys:
        score_array = logs[key]

        # make sure that we have the correct number of logs in the array
        if arrlen_check > 0:
            if len(score_array) != arrlen_check:
                print(colored("Something is off: The length of the score vector for {} is not {} ({})".format(
                    key, arrlen_check, len(score_array)), "red"))
                print(colored("I'd say you either trained the classifier for more epochs (default was 100),"))
                print(colored("or the experiment crashed (sorry :/)"))
                continue

        scores.append(score_array[-1])

    return np.array(scores)


if __name__ == "__main__":
    main()