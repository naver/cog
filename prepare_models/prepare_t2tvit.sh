# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder and downloads the checkpoint for the T2T_ViT_t-14 model by Yuan et al. 2021.
# Moreover, it clones the official code repository for the model definition file.
# https://github.com/yitu-opensource/T2T-ViT

# Note that the authors host their models on a cloud storage platform
# You need to manually download the checkpoint for T2T_ViT_t-14
# URL of the checkpoint:
# https://drive.google.com/file/d/1WdUT-3qq3duhECKk1CabXGktvd24p3Ti/view?usp=sharing

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

out_dir=${root_dir}/t2tvit/t14
mkdir -p ${out_dir}
cd ${out_dir}

echo "************************************************************"
echo "Please manually download the checkpoint for T2T_ViT_t-14"
echo "See the URL of the checkpoint in prepare_t2tvit.sh, and download it under ${PWD}/model.ckpt"

# We also need the models folder in their repo
# to be able to load pretrained t2t-vit models correctly

cd ..
echo "************************************************************"
echo "Cloning the code repository of T2T_ViT under ${PWD}"
git clone https://github.com/yitu-opensource/T2T-ViT src
cd src
git checkout 9dafdef17f6740fc600dad754e0f40bd246463e0  # version from 2021/03/07
