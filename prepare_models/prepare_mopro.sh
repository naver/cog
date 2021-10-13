# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder and downloads the checkpoint for the MoPro model trained on WebVision - v1
# See https://github.com/salesforce/MoPro
# URL of the checkpoint
# https://storage.googleapis.com/sfr-pcl-data-research/MoPro_checkpoint/MoPro_V1_epoch90.tar

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

mkdir -p ${root_dir}/mopro/resnet50
cd ${root_dir}/mopro/resnet50

echo "************************************************************"
echo "Downloading the ResNet-50 backbone of MoPro under ${PWD}"

wget "https://storage.googleapis.com/sfr-pcl-data-research/MoPro_checkpoint/MoPro_V1_epoch90.tar"
mv "MoPro_V1_epoch90.tar" "model.ckpt"