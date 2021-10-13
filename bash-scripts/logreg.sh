# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

# Bash script to train linear logistic regression classifiers with different seeds
# on the pre-extracted features of IN-1K or any of the CoG levels.
# So, you can the set the arguments below then let the script to launch
# 5 classification experiments on the pre-extracted features.
# (For advanced users: Experiments in the for loop are independent, meaning that you can
# parallelize them if you have a resourceful machine)
#
# At this stage, we assume that you already extracted features from a dataset using a pretrained model.
# If you haven't done it already, please see README.md and bash-scripts/extract_features.sh.

# **************************************************
# User specified parameters

# logreg.py expects paths to two files storing training and test set features.
# We provide two ways to specify these paths:
# 1) set the <models_root_dir>, <model_name> and <dataset> variables to load features from:
#    <models_root_dir>/<model_title>/<architecture_name>/<dataset>/features_{train and test}/X_Y.pth
#    (note that <model_title> and <architecture_name> comes from <model_name>, see below.)
# 2) set the <train_features_dir> and <test_features_dir> variables to load features from:
#    <train_features_dir>/X_Y.pth and <test_features_dir>/X_Y.pth

# ----------
# Variables for the option 1) to load features:

# The full path to the root directory of all models.
# If you followed our instructions in prepare_models/README.md to download pretrained models under <models_root_dir>,
# then you need to set it here as well.
# The same variable you might have used in extract_features.sh.
models_root_dir="None"

# For demonstration purposes, we set model_name to one of the models we support in this repo.
# For the complete list of models, please see the table in prepare_models/README.md.
# Remember that <model_name> is composed of two parts, separated by "_", i.e., <model_name>="<model_title>_<architecture_name>"
model_name="sup_resnet50"

# Which dataset's features to use to train the classifier
# Possible options are ("in1k", "cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")
dataset="None"

# Reminder:
# For this option to work, you must have already extracted features under
# <models_root_dir>/<model_title>/<architecture_name>/<dataset>/features_{train and test}/

# ----------
# Variables for the option 2):

# Full paths to the folders containing training and test set features.
# If these variables are set, we will ignore the <models_root_dir>, <model_name> and <dataset> variables set above.
train_features_dir="None"
test_features_dir="None"

# Reminder:
# For this option to work, you must have already extracted features under
# <train_features_dir>/ and <test_features_dir>/

# ----------
# Variable for the output directory for logs:

# If you chose option-1 for specifying the paths for features
# then logreg.py will save ouput logs under
# <models_root_dir>/<model_title>/<architecture_name>/<dataset>/eval_logreg/
# If you want to save the output logs to somewhere else
# then set <output_dir> accordingly.
# Notice that in the logreg.py callers below, we append the <seed> and <n_shot> variables to the <output_dir> variable.
# (Optional if <models_root_dir>, <model_name> and <dataset> are set)
output_dir="None"  # The full path to the root output folder for logs.

# **************************************************

# For loop to train 5 classifiers with different seeds
for seed in 22 37 45 77 93; do

    # Lets put experiments with different seeds under the same roof.
    # If you provided <models_root_dir> and left <output_dir>=None
    # don't worry, logreg.py will handle the folder name
    output_dir_="None"
    [[ "${output_dir}" != "None" ]] && output_dir_=${output_dir}/seed-${seed}

    # Training a classifier with all available training set
    echo "logreg | seed: ${seed}"
    python logreg.py \
        --models_root_dir=${models_root_dir} \
        --model=${model_name} \
        --dataset=${dataset} \
        --train_features_dir=${train_features_dir} \
        --test_features_dir=${test_features_dir} \
        --output_dir=${output_dir_} \
        EVAL.SEED ${seed}

done

# Finally print the average of 5 experiments
python print_results.py \
    --logreg_root_dir=${output_dir} \
    --models_root_dir=${models_root_dir} \
    --model=${model_name} \
    --dataset=${dataset}