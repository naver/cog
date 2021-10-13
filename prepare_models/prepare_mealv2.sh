# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder and downloads the checkpoint for the best MEAL-v2 with ResNet-50 backbone.
# https://github.com/szq0214/MEAL-V2
# URL of the checkpoint for MEALV2_ResNet50_224.pth
# https://1drv.ms/u/s!AtMVZxJ8MfxCi0NGENlMK0pYVDQM?e=GkwZ93

# Note that the authors host their models on OneDrive
# Refer to this stackoverflow post to see how one can download OneDrive files from bash.
# https://unix.stackexchange.com/questions/223734/how-to-download-files-and-folders-from-onedrive-using-wget

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

out_dir=${root_dir}/mealv2/resnet50
mkdir -p ${out_dir}
cd ${out_dir}

echo "************************************************************"
echo "Downloading the ResNet-50 backbone of MEAL-v2 under ${PWD}"

wget -O model.ckpt "https://public.bn.files.1drv.com/y4mgHbcBspmHnUAVB8icP5TM-GNq8xsI3LWLP33H-lIUuB1Jjp_vdd5Y9lhwSlb7Tw8qiMZb3FtAwh4T4fBg7ZGjpuxtkYuxG4BG27jzf4csaQf_jpxSWob2qeR7kTwdLXUb9kZYsgN-btDQSGV50hILMXbTVq8-BYYZAjSj4MKNFrLDTaHoDthy5Oo1zhvOqrAm3sF9lJvUYHx7CxAyciTjGwkj5kScBOLkXKo5rRug3k"

echo "Done"
