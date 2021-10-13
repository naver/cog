# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates output folders and downloads the checkpoints for four supervised models in the Torchvision repository:
# the ResNet-50/152, VGG19, Inception-v3 networks

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

change_dir () {
    out_dir=$1
    mkdir -p ${out_dir}
    cd ${out_dir}
    echo "out_dir:${out_dir}"
}

echo "************************************************************"
echo "Downloading the ResNet-50 backbone of SUP"
change_dir "${root_dir}/sup/resnet50"
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
mv "resnet50-19c8e357.pth" "model.ckpt"


echo "************************************************************"
echo "Downloading the ResNet-152 backbone of SUP"
change_dir "${root_dir}/sup/resnet152"
wget https://download.pytorch.org/models/resnet152-b121ed2d.pth
mv "resnet152-b121ed2d.pth" "model.ckpt"


echo "************************************************************"
echo "Downloading the VGG19-with-BN backbone of SUP"
change_dir "${root_dir}/sup/vgg19"
wget https://download.pytorch.org/models/vgg19_bn-c79401a0.pth
mv "vgg19_bn-c79401a0.pth" "model.ckpt"


echo "************************************************************"
echo "Downloading the Inception-v3 backbone of SUP"
change_dir "${root_dir}/sup/inception-v3"
wget https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth
mv "inception_v3_google-1a9a5a14.pth" "model.ckpt"


echo "Done"
