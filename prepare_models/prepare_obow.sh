# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder and downloads the checkpoint for the best OBOW model with ResNet-50 backbone.
# https://github.com/valeoai/obow

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

mkdir -p ${root_dir}/obow/resnet50
cd ${root_dir}/obow/resnet50

echo "************************************************************"
echo "Downloading the ResNet-50 backbone of OBOW under ${PWD}"

# we will download the checkpoint containing only the encoder
wget https://github.com/valeoai/obow/releases/download/v0.1.0/ImageNetFull_ResNet50_OBoW_full_feature_extractor.zip

unzip ImageNetFull_ResNet50_OBoW_full_feature_extractor.zip
mv ImageNetFull_ResNet50_OBoW_full_feature_extractor/tochvision_resnet50_student_K8192_epoch200.pth.tar model.ckpt
rm -rf ImageNetFull_ResNet50_OBoW_full_feature_extractor*

echo "Done"