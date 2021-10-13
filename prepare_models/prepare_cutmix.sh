# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates output folders for the Cutmix, Mixup and Manifold-Mixup models.
# https://github.com/clovaai/CutMix-PyTorch#experiments

# Note that the checkpoints are hosted on a cloud storage platform
# You need to manually download them
# URLs for:
# Cutmix ResNet-50
# https://www.dropbox.com/sh/w8dvfgdc3eirivf/AABnGcTO9wao9xVGWwqsXRala?dl=0
# Mixup ResNet-50
# https://www.dropbox.com/sh/g64c8bda61n12if/AACyaTZnku_Sgibc9UvOSblNa?dl=0
# Manifold Mixup
# https://www.dropbox.com/sh/bjardjje11pti0g/AABFGW0gNrNE8o8TqUf4-SYSa?dl=0

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

mkdir -p ${root_dir}/cutmix/resnet50
mkdir -p ${root_dir}/mixup/resnet50
mkdir -p ${root_dir}/manifold_mixup/resnet50

echo "************************************************************"
echo "Please manually download the checkpoints for Cutmix, Mixup and Manifold-Mixup"
echo "See the URL of the checkpoints in prepare_cutmix.sh, and download them under ${root_dir}/{cutmix/mixup/manifold_mixup}/resnet50/model.ckpt"