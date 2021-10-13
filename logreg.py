# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

import argparse
import copy
import logging
import math
import os
import shutil
import time

import optuna
import torch as th

import feature_ops
import metrics
import utils
from iterators import TorchIterator
from meters import AverageMeter, ProgressMeter

logger = logging.getLogger()


class LogReg:
    """
    Logistic regression classifier with mini-batch SGD.
    """

    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        # load the training set features
        trainset = feature_ops.load_feature_set(
            args.train_features_path, "train", cfg.CLF.NORM_FTS
        )

        if args.val:
            # randomly split the training set into train + val
            logger.info("Splitting the training set into train and val")
            trainset, testset = feature_ops.split_trainset(trainset, cfg.CLF.VAL_PERC)
        else:
            # load the test set
            testset = feature_ops.load_feature_set(args.test_features_path, "test", cfg.CLF.NORM_FTS)

        if cfg.CLF.N_SHOT > 0:
            logger.info(
                "Simulating few-shot learning setting, {} images per class.".format(
                    cfg.CLF.N_SHOT
                )
            )
            trainset = feature_ops.make_fewshot_dataset(trainset, cfg.CLF.N_SHOT)

        self.trainset = trainset
        self.testset = testset
        self.trainset.print_info()
        self.testset.print_info()

        # determine number of cases
        if len(list(self.trainset.y.shape)) == 1:
            classes = th.unique(self.trainset.y)
            assert th.all(classes == th.unique(self.testset.y))
        args.n_classes = classes.size(0)

        # move all features to the device
        if args.device == "cuda":
            feature_ops.move_data_to_cuda([self.trainset, self.testset])

    def __call__(self, trial=None):
        """
        The function called by Optuna.
        """
        # empty the cache allocated in the previous call
        th.cuda.empty_cache()

        args = copy.deepcopy(self.args)
        cfg = self.cfg

        x_train = self.trainset.x
        y_train = self.trainset.y
        x_test = self.testset.x
        y_test = self.testset.y

        # create training and test set iterators
        train_iter = TorchIterator((x_train, y_train), cfg.CLF.BATCH_SIZE, shuffle=True)
        test_iter = TorchIterator((x_test, y_test), cfg.CLF.BATCH_SIZE, shuffle=False)

        # define logistic classifier
        model = th.nn.Linear(x_train.size(1), args.n_classes).to(args.device)
        crit = th.nn.CrossEntropyLoss().to(args.device)

        # sample a learning rate and weight decay
        if trial is not None:
            lr_intv = cfg.CLF.LR_INTV
            wd_intv = cfg.CLF.WD_INTV
            args.lr = trial.suggest_loguniform("lr", lr_intv[0], lr_intv[1])
            args.wd = trial.suggest_loguniform("wd", wd_intv[0], wd_intv[1])
        optim = th.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd
        )

        args.exp_dir = os.path.join(
            args.output_dir,
            "{}-lr-{}_wd-{}".format("val" if args.val else "final", args.lr, args.wd),
        )
        os.makedirs(args.exp_dir, exist_ok=True)

        # write the model definition into exp_dir
        utils.write_to_file(str(model), os.path.join(args.exp_dir, "model.txt"))

        # logs computed during training / evaluation
        args.logs = {
            "train/loss": [],
            "train/top1": [],
            "train/top5": [],
            "test/loss": [],
            "test/top1": [],
            "test/top5": [],
            "lr": [],
        }

        # predictions over the evaluation sets
        args.preds = []

        for epoch in range(cfg.CLF.N_EPOCHS):
            if not args.val:
                logger.info(f"**Epoch:{epoch}**")
            args.epoch = epoch
            train_stat = train(train_iter, model, crit, optim, epoch, args)
            validate(test_iter, model, crit, args)
            adjust_learning_rate(optim, args, cfg)

            # if something went wrong during training
            # e.g. SGD diverged
            if train_stat == -1:
                break

        # save the logs
        utils.save_pickle(args.logs, f"{args.exp_dir}/logs.pkl")

        # save the predictions
        utils.save_pickle(args.preds, f"{args.exp_dir}/preds.pkl")

        # save the whole args, for ease of access
        utils.save_pickle(vars(args), f"{args.exp_dir}/args.pkl")

        # save also the final model
        th.save(
            {
                "model": model.state_dict(),
            },
            f"{args.exp_dir}/model.pth",
        )

        # return the last test accuracy
        return args.logs["test/top1"][-1]


def train(train_loader, model, criterion, optimizer, epoch, args):
    """
    Train the classifier for one epoch.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (fts, lbls) in enumerate(train_loader):
        fts = fts.to(args.device)
        lbls = lbls.to(args.device)

        # compute output
        output = model(fts)
        loss = criterion(output, lbls)

        if not th.isfinite(loss):
            logger.info("Loss ({}) is not finite, terminating".format(loss.item()))
            optimizer.zero_grad()
            return -1

        # measure accuracy and record loss
        acc1, acc5 = metrics.accuracy(output, lbls, topk=(1, 5))
        losses.update(loss.item(), fts.size(0))
        top1.update(acc1.item(), fts.size(0))
        top5.update(acc5.item(), fts.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (not args.val) and (i % args.print_freq == 0):
            progress.display(i)

    args.logs["train/loss"].append(losses.avg)
    args.logs["train/top1"].append(top1.avg)
    args.logs["train/top5"].append(top5.avg)
    return 0


def validate(val_loader, model, criterion, args):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to evaluate mode
    model.eval()

    # keep predictions per class
    preds = th.ones(len(val_loader.tensors[0]), dtype=th.int32, device=args.device) * -1.
    six = 0

    with th.no_grad():
        for i, (fts, lbls) in enumerate(val_loader):
            fts = fts.to(args.device)
            lbls = lbls.to(args.device)
            bs = fts.size(0)

            # compute output
            output = model(fts)
            loss = criterion(output, lbls)

            # store the predicted classes
            preds[six:six + bs] = th.argmax(output, dim=1)
            six += bs

            # measure accuracy and record loss
            acc1, acc5 = metrics.accuracy(output, lbls, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(acc1[0].item(), bs)
            top5.update(acc5[0].item(), bs)

    # make sure that there is no invalid prediction
    assert th.all(preds >= 0).item()
    args.preds.append(preds.detach().cpu())

    args.logs["test/loss"].append(losses.avg)
    args.logs["test/top1"].append(top1.avg)
    args.logs["test/top5"].append(top5.avg)

    if not args.val:
        logger.info(
            " * Acc@1:{top1.avg:.3f} - Acc@5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
        )


def adjust_learning_rate(optimizer, args, cfg):
    """Decay the learning rate based on cosine schedule"""
    lr = args.lr
    lr *= 0.5 * (1.0 + math.cos(math.pi * args.epoch / cfg.CLF.N_EPOCHS))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    args.logs["lr"].append(lr)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    th.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=utils.none_or_string_flag,
                        help='Name of the model in the <model_title>_<architecture_name> form.'
                             'See the table of models in ./prepare_models/README.md for all the model names we support.'
                             'This is an optional argument that needs to be set along with --models_root_dir and --dataset.'
                             'When these three arguments are set, the script will load features from:'
                             '<models_root_dir>/<model_title>/<architecture_name>/<dataset>/features_*/X_Y.pth.'
                             'If you would like to load pre-extracted features from somewhere else'
                             'then ignore this argument and provide the --train_features_dir and --test_features_dir arguments accordingly')
    parser.add_argument('--models_root_dir', type=utils.none_or_string_flag,
                        help='Root directory for all models, see prepare_models/README.md for a detailed explanation.'
                             'This is an optional argument that needs to be set along with --model and --dataset.'
                             'Please see the help message for the --model argument as well.')
    parser.add_argument("--dataset", type=utils.none_or_string_flag,
                        help="On which dataset to learn classifiers"
                             'Possible values are ("in1k", "cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")'
                             'This is an optional argument that needs to be set along with --models_root_dir and --model.'
                             'Please see the help message for the --model argument as well.')
    parser.add_argument('--train_features_dir', type=utils.none_or_string_flag,
                        help='Path to the directory containing pre-extracted training set features.'
                             'We expect a features file "X_Y.pth" under <train_features_dir>.'
                             'This is an optional argument that needs to be set if --models_root_dir, --model and --dataset are not set.')
    parser.add_argument('--test_features_dir', type=utils.none_or_string_flag,
                        help='Path to the directory containing pre-extracted test set features.'
                             'We expect a features file "X_Y.pth" under <test_features_dir>.'
                             'This is an optional argument that needs to be set if --models_root_dir, --model and --dataset are not set.')
    parser.add_argument('--output_dir', type=utils.none_or_string_flag,
                        help='Where to log program logs.'
                             'This is an optional argument that needs to be set if --models_root_dir is not set.'
                             'If not provided, we try to save the logs under'
                             '<models_root_dir>/<model_title>/<architecture_name>/<dataset>/eval_logreg/seed*')
    # learning rate and momentum are tuned in this program, do not manually set.
    parser.add_argument("--lr", type=float, default=0.0, help="initial learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--mom", type=float, default=0.9, help="momentum")
    # program-related options
    parser.add_argument("--print_freq", default=100, type=int, help="print frequency (default: 10)")
    parser.add_argument("--device", type=str, default="cuda")
    # optionally to overwrite the default config
    parser.add_argument("opts", default=None,
                        help="see configs/default.py for all options",
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.device == "cuda" and not th.cuda.is_available():
        print("CUDA is not available, I will run on CPU.")
        args.device = "cpu"

    # load the config file
    # create output directory,
    # locate pre-extracted features,
    # initialize program logger,
    # save args and cfg
    # this function sets the following arg variables:
    # - train_features_path, type=str
    # - test_features_path, type=str
    # - output_dir, type=str
    args, cfg = utils.init_program(args, _for="logreg")

    # tune hyper-parameters with optuna
    logger.info("Running Optuna...")
    hps_sampler = optuna.samplers.TPESampler(multivariate=True, seed=cfg.EVAL.SEED)
    study = optuna.create_study(sampler=hps_sampler, direction="maximize")

    args.val = True
    logreg = LogReg(args, cfg)
    study.optimize(logreg, n_trials=cfg.CLF.N_TRIALS, n_jobs=1, show_progress_bar=False)
    utils.save_pickle(study, os.path.join(args.output_dir, "study.pkl"))

    logger.info("")
    logger.info("*" * 50)
    logger.info("Hyper-parameter search ended")
    logger.info("best_trial:")
    logger.info(str(study.best_trial))
    logger.info("best_params:")
    logger.info(str(study.best_params))
    logger.info("*" * 50)
    logger.info("")

    # train the final classifier with the tuned hyper-parameters
    del logreg
    th.cuda.empty_cache()
    args.lr = study.best_params["lr"]
    args.wd = study.best_params["wd"]
    args.val = False
    logreg = LogReg(args, cfg)
    logreg()
