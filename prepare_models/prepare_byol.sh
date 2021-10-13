# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder and downloads the checkpoint for the best BYOL model with ResNet-50 backbone.
# Moreover, it downloads a 3rd party TensorFlow-to-PyTorch model converter for BYOL and converts the official TensorFlow model into PyTorch.
# https://github.com/deepmind/deepmind-research/tree/master/byol

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

out_dir=${root_dir}/byol/resnet50
mkdir -p ${out_dir}
cd ${out_dir}

echo "************************************************************"
echo "Cloning the PyTorch converter for BYOL under ${PWD}"
git clone https://github.com/chigur/byol-convert
cd byol-convert

echo "************************************************************"
echo "Downloading the ResNet-50 backbone of BYOL under ${PWD}"

wget https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl

echo "Converting the TensorFlow model into PyTorch"
python convert.py pretrain_res50x1.pkl pretrain_res50x1.pth.tar

echo "Moving the output checkpoint file to the parent directory"
mv pretrain_res50x1.pth.tar ../model.ckpt

echo "Done"
