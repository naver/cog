# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Downloads all pretrained model checkpoints

# Root directory for all the models
root_dir=$1

# Set this directory accordingly
if [[ ${root_dir} == "" ]]; then
    echo "Please provide a valid <root_dir> (${root_dir})."
    exit -1
fi

mkdir -p ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "Supervised"
bash prepare_models/prepare_sup.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "Data efficient visual transformer "
bash prepare_models/prepare_deit.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "Neural architecture transfer"
bash prepare_models/prepare_nat.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "EfficientNet"
bash prepare_models/prepare_efficientnet.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "T2T visual transformer"
bash prepare_models/prepare_t2tvit.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "MoCo-v2"
bash prepare_models/prepare_moco.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "MoCHi"
bash prepare_models/prepare_mochi.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "InfoMin"
bash prepare_models/prepare_infomin.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "SwAV"
bash prepare_models/prepare_swav.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "DINO"
bash prepare_models/prepare_dino.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "OBOW"
bash prepare_models/prepare_obow.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "Barlow-Twins"
bash prepare_models/prepare_barlow.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "CompReSS"
bash prepare_models/prepare_compress.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "Cutmix, Mixup and Manifold mixup"
bash prepare_models/prepare_cutmix.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "ReLabel, supervised classifier trained on the relabeled ImageNet-1K"
bash prepare_models/prepare_relabel.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "Adversarially robust models from Salman et al."
bash prepare_models/prepare_advrobust.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "MEAL-V2"
bash prepare_models/prepare_mealv2.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "Semi- and Semi-weakly supervised"
bash prepare_models/prepare_swsup.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "MoPro model pretrained on WebVision dataset"
bash prepare_models/prepare_mopro.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "CLIP models"
bash prepare_models/prepare_clip.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "Installing an environment for PyTorch converters..."
# For SimCLR-v2 and BYOL, we need to convert pretrained tensorflow models into PyTorch ones
conda create -y --name pytorch_converter python=3.6
conda activate pytorch_converter
pip install tensorflow tensorflow-datasets tensorflow-hub absl-py dm-haiku dm-tree jax jaxlib torch

echo "----------------------------------------------------------------------------------------------------"
echo "SimCLR-v2"
bash prepare_models/prepare_simclr.sh ${root_dir}

echo "----------------------------------------------------------------------------------------------------"
echo "BYOL"
bash prepare_models/prepare_byol.sh ${root_dir}
