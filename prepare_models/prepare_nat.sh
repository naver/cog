# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates an output folder and downloads the checkpoint for the NAT-M4 model.
# Moreover, it clones the official code repository for the model definition file.
# https://github.com/human-analysis/neural-architecture-transfer
# M4 model is defined here
# https://github.com/human-analysis/neural-architecture-transfer/blob/master/subnets/imagenet/NAT-M4/net.config


# Arguments:
# root_dir: path where all models are saved
root_dir=$1

out_dir=${root_dir}/nat/m4
mkdir -p ${out_dir}
cd ${out_dir}

echo "************************************************************"
echo "Downloading the NAT-M4 backbone ${PWD}"
wget -O model.ckpt https://www.zhichaolu.com/assets/neural-architecture-transfer/pretrained/imagenet/net-img@240-flops@600-top1@80.5/net.weights

cd ..
echo "************************************************************"
echo "Cloning the code repository of NAT under ${PWD}"
git clone https://github.com/human-analysis/neural-architecture-transfer src
cd src
git checkout 5aca00a11b22902a9e7b2f9c33b58bc4ba686828  # version from 2021/03/07

echo "Done"