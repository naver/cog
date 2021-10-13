# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

import os
import logging
import warnings

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from data import load_pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
logger = logging.getLogger()


class CoGLevelDataset(Dataset):
    """Custom PyTorch dataset to load (image, label) pairs of the concepts in a concept generalization level."""

    # for some sanity checks
    n_min_images = 732
    n_max_images = 1300
    n_test_images = 50

    def __init__(
        self,
        images_dir,
        concepts_split_file,
        concepts,
        split,
        transform,
    ):
        assert split == "train" or split == "test"
        super().__init__()
        self.images_dir = images_dir
        self.split = split
        self.transform = transform
        self.concepts = concepts

        # Dictinary that keeps training and test splits for each concept
        images_per_synset = load_pickle(concepts_split_file)

        # go through all the images of the concepts
        image_files = []
        labels = []
        six = 0
        for wnid in sorted(concepts):

            _files = images_per_synset[wnid][split].tolist()
            _files.sort()
            if split == "train":
                assert len(_files) >= self.n_min_images
                assert len(_files) <= self.n_max_images
            elif split == "test":
                assert len(_files) == self.n_test_images

            image_files += _files
            labels += [six] * len(_files)
            six += 1
            logger.info(
                f"CoGLevelDataset found {len(_files)} images for the concept {wnid}"
            )

        self.image_files = image_files
        self.labels = labels
        logger.info(
            f"CoGLevelDataset loaded {len(self.image_files)} images for {len(concepts)} concepts"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Returns the index-th sample in the dataset.
        Note that this getter returns None if an error occurs while loading an image.
        """

        try:
            img_file = self.image_files[index]
            # if img_file starts with a backslash, remove it
            # otherwise, os.path.join does not work
            if img_file[0] == "/" or img_file == "\\":
                img_file = img_file[1:]
            img_file = os.path.join(self.images_dir, img_file)
            img = Image.open(img_file).convert("RGB")

        except Exception as e:
            logger.error("ERROR while loading image {}".format(img_file))
            logger.error("{}".format(e))
            return None

        # peacefully return the pair
        img = self.transform(img)
        label = self.labels[index]
        return img, label
