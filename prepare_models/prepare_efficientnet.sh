# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Creates output folders for EfficientNet-B1/B4 models and also install pycls package from a FAIR team
# The official EfficientNet implementation is in TensorFlow
# https://github.com/qubvel/efficientnet
# But we will use the PyTorch package pycls
# https://github.com/facebookresearch/pycls
# to manage downloading and loading pretrained checkpoints for EfficientNet models.
# Note that this package will download pretrained checkpoints under "/tmp/pycls-download-cache" as denoted here
# https://github.com/facebookresearch/pycls/blob/main/pycls/models/model_zoo.py (_DOWNLOAD_CACHE variable)

# Arguments:
# root_dir: path where all models are saved
root_dir=$1

b1_dir=${root_dir}/efficientnet/b1
b4_dir=${root_dir}/efficientnet/b4
mkdir -p ${b1_dir} ${b4_dir}

cd ..
echo "************************************************************"
echo "Downloading the pycls package under ${PWD}"
git clone https://github.com/facebookresearch/pycls src
cd src

pip install -r requirements.txt
pip install .

echo "Done"