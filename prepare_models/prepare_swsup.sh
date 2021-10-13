# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates output folders and downloads pretrained checkpoints for semi- and semi-weakly supervised models from Yalniz et al. 2019
# https://github.com/facebookresearch/semi-supervised-ImageNet1K-models

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

ssup_dir=${root_dir}/ssup/resnet50
swsup_dir=${root_dir}/swsup/resnet50
mkdir -p ${ssup_dir} ${swsup_dir}

cd ${swsup_dir}
echo "************************************************************"
echo "Downloading the ResNet-50 backbone of S-W-SUP under ${PWD}"
wget https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth
mv "semi_weakly_supervised_resnet50-16a12f1b.pth" "model.ckpt"

cd ${ssup_dir}
echo "************************************************************"
echo "Downloading the ResNet-50 backbone of S-SUP under ${PWD}"
wget https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth
mv "semi_supervised_resnet50-08389792.pth" "model.ckpt"

echo "Done"
