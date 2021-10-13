# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder for the adversarially robust model by Salman et al. 2020.
# https://github.com/microsoft/robust-models-transfer

# Note that the authors host their models on a cloud storage platform
# You need to manually download the model with ResNet50 backbone trained with epsilon: 0.1 (the best performing one on CoG)
# URL of the checkpoint
# ==> https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.1.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

mkdir -p ${root_dir}/advrobust/resnet50

echo "************************************************************"
echo "Please manually download the checkpoint for Adv-Robust"
echo "See the URL of the checkpoint in prepare_advrobust.sh, and download it under ${root_dir}/advrobust/resnet50/model.ckpt"