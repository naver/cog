# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder and downloads the checkpoint for the best MoCHi model with ResNet-50 backbone.
# https://europe.naverlabs.com/research/computer-vision/mochi/#pretrainedmodels

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

out_dir=${root_dir}/mochi/resnet50
mkdir -p ${out_dir}
cd ${out_dir}

echo "************************************************************"
echo "Downloading the ResNet-50 backbone of MoCHi under ${PWD}"

wget https://download.europe.naverlabs.com/ComputerVision/MOCHI_models/mochi_512_1024_512_e1000.pth.tar
mv "mochi_512_1024_512_e1000.pth.tar" "model.ckpt"

echo "Done"
