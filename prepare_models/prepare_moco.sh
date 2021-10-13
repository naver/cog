# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder and downloads the checkpoint for the best MoCo-v2 model with ResNet-50 backbone.
# https://github.com/facebookresearch/moco/

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

out_dir=${root_dir}/mocov2/resnet50
mkdir -p ${out_dir}
cd ${out_dir}

echo "************************************************************"
echo "Downloading the ResNet-50 backbone of MoCo-v2 under ${PWD}"

wget https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar
mv "moco_v2_800ep_pretrain.pth.tar" "model.ckpt"

echo "Done"
