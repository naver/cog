# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder and downloads the checkpoint for the best BarlowTwins model with ResNet-50 backbone.
# https://github.com/facebookresearch/barlowtwins


# Arguments:
# root_dir: path where all models are saved
root_dir=$1

model_dir=${root_dir}/barlow/resnet50
mkdir -p ${model_dir}
cd ${model_dir}

echo "************************************************************"
echo "Downloading the ResNet-50 backbone of Barlow-Twins under ${PWD}"

wget -O model.ckpt https://dl.fbaipublicfiles.com/barlowtwins/epochs1000_bs2048_lr0.2_lambd0.0051_proj_8192_8192_8192_scale0.024/resnet50.pth

echo "Done"