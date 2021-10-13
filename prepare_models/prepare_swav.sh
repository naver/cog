# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder and downloads the checkpoint for the best SwAV model with ResNet-50 backbone.
# https://github.com/facebookresearch/swav

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

out_dir=${root_dir}/swav/resnet50
mkdir -p ${out_dir}
cd ${out_dir}

echo "************************************************************"
echo "Downloading the ResNet-50 backbone of SwAV under ${PWD}"

wget https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar
mv "swav_800ep_pretrain.pth.tar" "model.ckpt"

echo "Done"
