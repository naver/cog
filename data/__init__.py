# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

import logging
import os

from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import default_collate
from utils import load_pickle

from data.datasets import CoGLevelDataset

logger = logging.getLogger()


def center_crop(image_resize_size=224, image_crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Image transformation pipeline often used for evaluating image classification models.
    Typically it contains:
        Resize(size of shortest side)
        Crop(size)
        ToTensor()
        Normalize()
    """
    tfm_list = [
        transforms.Resize(image_resize_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(image_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    return transforms.Compose(tfm_list)


def load_dataset(
    dataset,
    split,
    image_resize_size=224,
    image_crop_size=224,
    image_norm_mean=[0.485, 0.456, 0.406],
    image_norm_std=[0.229, 0.224, 0.225],
    transform=None,
    cog_levels_mapping_file="",
    cog_concepts_split_file="",
    imagenet_images_root="",
    in1k_images_root="",
):
    """
    Loads a dataset according to the config.
    """

    # image transformation
    if transform is None:
        transform = center_crop(image_resize_size, image_crop_size, image_norm_mean, image_norm_std)
    logger.info("Image transform:")
    logger.info("{}".format(transform))

    if dataset.startswith("in1k"):
        # load ImageNet-1K challange dataset
        # the test split is the official validation set split
        if split == "test":
            split = "val"
        dset = ImageFolder(os.path.join(in1k_images_root, split), transform=transform)

        # make sure we load the correct number of images
        n_samples = {
            "in1k": {
                "train": 1281167,
                "val": 50000
            },
        }[dataset][split]
        assert len(dset) == n_samples

    elif dataset.startswith("cog"):
        # load a concept generalization level
        # example cog_l1
        _, level = dataset.split("_")
        logger.info(
            "Initializing CoG Dataset level: {}, split: {}".format(
                level, split
            )
        )

        # level is an index of the level between 1 and 5
        assert level.startswith("l"), \
            "level should start with l ({})".format(level)
        assert len(level) == 2, \
            "level should be 2 chars, e.g., l1, ..., l5 ({})".format(level)
        level = int(level[1])
        assert level in list(range(1, 6))

        # load the concepts in this level
        selected_wnids = load_pickle(cog_levels_mapping_file)[level - 1]
        assert len(selected_wnids) == 1000, \
            "There should be 1000 concepts in level {}, but found {} concepts".format(level, len(selected_wnids))

        # initialize the dataset
        dset = CoGLevelDataset(
            imagenet_images_root,
            cog_concepts_split_file,
            selected_wnids,
            split,
            transform,
        )

    else:
        raise NotImplementedError("Ups")

    return dset


def my_collate(batch):
    """
    Extend the default collate function to ignore None samples.
    """
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)
