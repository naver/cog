# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder and downloads the checkpoint for the best DINO model with ResNet-50 backbone.
# https://github.com/facebookresearch/dino

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

out_dir=${root_dir}/dino/resnet50
mkdir -p ${out_dir}
cd ${out_dir}

echo "************************************************************"
echo "Downloading the ResNet-50 backbone of DINO under ${PWD}"

wget -O model.ckpt https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth

echo "Done"
