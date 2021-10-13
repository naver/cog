# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Bash script to extract features from IN-1K and the 5 CoG levels using a pretrained model.
# (For advanced users: Experiments in the for loop are independent, meaning that you can
# parallelize them if you have a resourceful machine)
#
# Before using this script, please follow the instructions under the "Data" section of README.md,
# and download the ImageNet-21K (IN-21K) and ILSVRC-2012 (IN-1K) datasets,
# as well as the two ImageNet CoG level files named cog_levels_mapping_file.pkl and cog_concepts_split_file.pkl.
#
# Also, if you want to extract features from your custom model,
# please see the "Preparing the model" section of README.md first.

# For demonstration purposes, we set model_name to one of the models we support in this repo.
# For the complete list of models, please see the table in prepare_models/README.md.
# Remember that <model_name> is composed of two parts, separated by "_", i.e., <model_name>="<model_title>_<architecture_name>"
model_name="sup_resnet50"

# **************************************************
# User specified parameters
in1k_images_root="None"  # The full path to the IN-1K dataset
imagenet_images_root="None"  # The full path to the IN-21K dataset
cog_levels_mapping_file="None"  # The full path to cog_levels_mapping_file.pkl
cog_concepts_split_file="None"  # The full path to cog_concepts_split_file.pkl

# If you followed our instructions in prepare_models/README.md to download pretrained models under <models_root_dir>,
# then you need to set it here as well.
# If this variable is set, extract_features.py will look for model checkpoint files under
# <models_root_dir>/<model_title>/<architecture_name>/model.ckpt, and
# will save features under
# <models_root_dir>/<model_title>/<architecture_name>/<dataset>/features_<split>/X_Y.pth
# (Optional if <ckpt_file> and <output_dir> are set.)
models_root_dir="None"  # The full path to the root directory of all models.

# If you want to load a checkpoint file of <model_name> from somewhere else other than
# <models_root_dir>/<model_title>/<architecture_name>/model.ckpt,
# then you need to set the <ckpt_file> variable as the full path to this checkpoint file.
# Note that this variable is not needed when <models_root_dir> is set.
# (Optional if <models_root_dir> is set)
ckpt_file="None"  # The full path to the model checkpoint file.

# If you want to output features from somewhere else other than
# <models_root_dir>/<model_title>/<architecture_name>/<dataset>/features_<split>/
# then you need to set the <output_dir> variable as the full path to the output folder.
# Note that this variable should point to a folder, not a file.
# Because features will be saved under <output_dir>/X_Y.pth.
# (Optional if <models_root_dir> is set)
output_dir="None"  # The full path the output folder for features.
# **************************************************

# Nested for loop to extract features of all datasets and splits
for dataset in "in1k" "cog_l1" "cog_l2" "cog_l3" "cog_l4" "cog_l5"; do
    for split in "test" "train" ; do
        echo "Feature extraction - model: ${model_name}, dataset: ${dataset}, split: ${split}"
        python extract_features.py \
            --model=${model_name} \
            --ckpt_file=${ckpt_file} \
            --dataset=${dataset} \
            --split=${split} \
            --models_root_dir=${models_root_dir} \
            --cog_levels_mapping_file=${cog_levels_mapping_file} \
            --cog_concepts_split_file=${cog_concepts_split_file} \
            --imagenet_images_root=${imagenet_images_root} \
            --in1k_images_root=${in1k_images_root} \
            --output_dir=${output_dir}
    done
done
