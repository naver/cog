# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates output folders and downloads the checkpoints for three DeiT models: small, distilled small, and distilled base models
# Moreover, it clones the official code repository for the model definition files.
# https://github.com/facebookresearch/deit

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

small_dir=${root_dir}/deit/small
small_distilled_dir=${root_dir}/deit/small_distilled
base_distilled_dir=${root_dir}/deit/base_distilled_384

mkdir -p ${small_dir} ${small_distilled_dir} ${base_distilled_dir}

echo "************************************************************"
echo "Downloading the DeiT backbones"
wget https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth -O ${small_dir}/model.ckpt
wget https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth -O ${small_distilled_dir}/model.ckpt
wget https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth -O ${base_distilled_dir}/model.ckpt

cd ${root_dir}/deit
echo "************************************************************"
echo "Cloning the code repository of DeiT under ${PWD}"
git clone https://github.com/facebookresearch/deit src
cd src
git checkout ab5715372db8c6cad5740714b2216d55aeae052e  # version from 2021/01/18

echo "Done"