# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder and downloads the checkpoint for the best SimCLR-v2 model with ResNet-50 backbone.
# Moreover, it downloads a 3rd party TensorFlow-to-PyTorch model converter for SimCLR-v2 and converts the official TensorFlow model into PyTorch.
# https://github.com/google-research/simclr

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

out_dir=${root_dir}/simclrv2/resnet50
mkdir -p ${out_dir}
cd ${out_dir}

echo "************************************************************"
echo "Cloning the PyTorch converter..."
git clone https://github.com/Separius/SimCLRv2-Pytorch
cd SimCLRv2-Pytorch

echo "************************************************************"
echo "Downloading the ResNet-50 backbone of SimCLR-v2 under ${PWD}"
python download.py "r50_1x_sk0"

echo "Converting the TensorFlow model into PyTorch"
python convert.py r50_1x_sk0/model.ckpt-250228

echo "Moving the output checkpoint file to the parent directory"
mv r50_1x_sk0.pth ../model.ckpt

echo "Done"