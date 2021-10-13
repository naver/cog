# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates the output folder for CLIP.
# We will not download any pretrained model for CLIP.
# See the notice below.

echo "************************************************************"
echo "Initializing CLIP models is a bit involved."
echo "So, we will use their API to download & initialize pretrained CLIP models"
echo "Please install CLIP API."
echo "For more information, see their code repository: https://github.com/openai/CLIP"
echo ""

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

model_dir=${root_dir}/clip/resnet50
mkdir -p ${model_dir}