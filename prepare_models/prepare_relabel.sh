# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder for ReLabel.
# https://github.com/naver-ai/relabel_imagenet

# Note that the authors host their models on a cloud storage platform
# You need to manually download the model with ResNet50 backbone
# URL of the checkpoint
# https://www.dropbox.com/s/a0jzq3933s1wcig/rn50_relabel_78.9.pth?dl=0

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

mkdir -p ${root_dir}/relabel/resnet50

echo "************************************************************"
echo "Please manually download the checkpoint for ReLabel"
echo "See the URL of the checkpoint in prepare_relabel.sh, and download it under ${root_dir}/relabel/resnet50/model.ckpt"