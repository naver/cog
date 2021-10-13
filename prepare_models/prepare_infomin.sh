# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder and downloads the checkpoint for the best InfoMin model with ResNet-50 backbone.
# https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/docs/MODEL_ZOO.md

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

out_dir=${root_dir}/infomin/resnet50
mkdir -p ${out_dir}
cd ${out_dir}

echo "************************************************************"
echo "Downloading the ResNet-50 backbone of InfoMin under ${PWD}"

wget https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAAzMTynP3Qc8mIE4XWkgILUa/InfoMin_800.pth
mv "InfoMin_800.pth" "model.ckpt"

echo "Done"
