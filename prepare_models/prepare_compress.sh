# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder for CompReSS.
# https://github.com/UMBCvision/CompReSS

# Note that the authors host their models on a cloud storage platform
# You need to manually download the model with
# teacher: SimCLR ResNet50x4
# student: ResNet50
# networks
# URL of the checkpoint
# https://drive.google.com/file/d/15rzzSkcedEuCE7Cm8yLXopA5PqHUQscb/view?usp=sharing

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

mkdir -p ${root_dir}/compress/resnet50

echo "************************************************************"
echo "Please manually download the checkpoint for CompReSS"
echo "See the URL of the checkpoint in prepare_compress.sh, and download it under ${root_dir}/compress/resnet50/model.ckpt"