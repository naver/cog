# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

import argparse
import logging

import torch as th
from torch.utils.data import DataLoader

import data
import model_utils
import utils

logger = logging.getLogger()


def main(args, cfg):
    """
    Main routine to extract features.
    """

    # initialize the network
    net, forward = model_utils.load_pretrained_backbone(args.model, args.ckpt_file)

    # initialize the dataset loader
    dataset = data.load_dataset(
        args.dataset,
        args.split,
        image_resize_size=cfg.FT_EXTRACT.IMAGE_RESIZE_SIZE,
        image_crop_size=cfg.FT_EXTRACT.IMAGE_CROP_SIZE,
        image_norm_mean=cfg.FT_EXTRACT.IMAGE_NORM.MEAN,
        image_norm_std=cfg.FT_EXTRACT.IMAGE_NORM.STD,
        cog_levels_mapping_file=args.cog_levels_mapping_file,
        cog_concepts_split_file=args.cog_concepts_split_file,
        imagenet_images_root=args.imagenet_images_root,
        in1k_images_root=args.in1k_images_root,
    )
    logger.info("Dataset size:{}".format(len(dataset)))

    loader = DataLoader(dataset,
                        batch_size=cfg.FT_EXTRACT.BATCH_SIZE,
                        shuffle=False,
                        collate_fn=data.my_collate,
                        num_workers=cfg.FT_EXTRACT.N_WORKERS,
                        pin_memory=True)

    # (image, label) pairs to be stored under args.output_dir
    X = None
    Y = None
    six = 0  # sample index

    def _print():
        logger.info("{}/{} extracted.".format(six, len(dataset)))

    with th.no_grad():
        for bix, batch in enumerate(loader):
            assert len(batch) == 2
            image, label = batch

            if forward is None:
                feature = net(image.cuda())
            else:
                feature = forward(image.cuda())

            if X is None:
                logger.info("Size of the first batch: {} and features {}".format(list(image.shape), list(feature.shape)))
                X = th.zeros(len(dataset), feature.size(1), dtype=th.float32, device="cpu")
                if label.ndim == 2:
                    Y = th.zeros(len(dataset), label.size(1), dtype=th.long, device="cpu")
                else:
                    Y = th.zeros(len(dataset), dtype=th.long, device="cpu")

            bs = feature.size(0)
            X[six:six + bs] = feature.cpu()
            Y[six:six + bs] = label
            six += bs
            if bix % 10 == 0:
                _print()

    # remove any zero features
    # in case some of the images couldn't be read from disk
    if six < len(X):
        logger.info("Couldn't extracted features from {} images.".format(len(X) - six))
        X = X[:six]
        Y = Y[:six]

    _print()
    logger.info("DONE")
    logger.info("We have X:{}, and Y:{}".format(
        list(X.shape), list(Y.shape)))

    th.save(
        {
            "X": X,
            "Y": Y,
            "name": "{}_{}_{}".format(args.model, args.dataset, args.split)
        },
        args.features_save_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=utils.none_or_string_flag, required=True,
                        help='Name of the model in the <model_title>_<architecture_name> form.'
                             'See the table of models in ./prepare_models/README.md for all the model names we support.'
                             'To extract features from your custom models, follow the instructions in README.md first.')
    parser.add_argument('--dataset', type=utils.none_or_string_flag, required=True,
                        help='From which dataset to extract representations.'
                             'Possible values are ("in1k", "cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")')
    parser.add_argument('--split', type=utils.none_or_string_flag, required=True,
                        help='Split of the dataset'
                             'Possible values are ("train", "test).'
                             'When --dataset="in1k" and --split="test", we use the official validation set of IN-1K as test set.')
    parser.add_argument('--models_root_dir', type=utils.none_or_string_flag,
                        help='Root directory for all models, see prepare_models/README.md for a detailed explanation.'
                             'We need this argument to load model checkpoint files from'
                             '"<models_root_dir>/<model_title>/<architecture_name>/model.ckpt"'
                             'and save output features to'
                             '<models_root_dir>/<model_title>/<architecture_name>/<dataset>/features_<split>'
                             'This is an optional argument that needs to be set if --ckpt_file and --output_dir are not set.')
    parser.add_argument('--ckpt_file', type=utils.none_or_string_flag,
                        help='Path to the pretrained weights file for the model.'
                             'This is an optional argument that needs to be set if --models_root_dir is not set.')
    parser.add_argument('--output_dir', type=utils.none_or_string_flag,
                        help='Where to extract features.'
                             'This is an optional argument that needs to be set if --models_root_dir is not set.')
    parser.add_argument('--imagenet_images_root', type=utils.none_or_string_flag,
                        help='Root data directory for the images of the concepts selected in the CoG benchmark.'
                             'Images for each concept must be grouped under their synset id, under this root directory'
                             'This is an optional argument that needs to be set if --dataset is one of CoG levels, i.e., one of ("cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")')
    parser.add_argument('--cog_levels_mapping_file', type=utils.none_or_string_flag,
                        help='Pickle file containing a list of concepts in each level (5 lists in total).'
                             'This is an optional argument that needs to be set if --dataset is one of CoG levels, i.e., one of ("cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")')
    parser.add_argument('--cog_concepts_split_file', type=utils.none_or_string_flag,
                        help='Pickle file containing training and test splits for each concept in ImageNet-CoG.'
                             'This is an optional argument that needs to be set if --dataset is one of CoG levels, i.e., one of ("cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")')
    parser.add_argument('--in1k_images_root', type=utils.none_or_string_flag,
                        help='Root directory for the IN-1K dataset.'
                             'There must be "train" and "val" folders under this root, as usual.'
                             'This is an optional argument that needs to be set if --dataset="in1k"')

    # optionally to overwrite config
    parser.add_argument('opts', default=None,
                        help='see configs/default.py for all options', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # load the config file
    # create output directory,
    # initialize program logger,
    # save args and cfg
    # this function sets the following arg variables:
    # - output_dir, type=str
    # - features_save_path, type=str
    args, cfg = utils.init_program(args, _for="ft-extract")

    main(args, cfg)
