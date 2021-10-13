<!--
ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​
-->

<!-- omit in toc -->
# The ImageNet-CoG Benchmark

| [Project Website](https://europe.naverlabs.com/cog-benchmark) | [Paper (arXiv)](https://arxiv.org/abs/2012.05649) |
| :-: | :-: |

Code repository for the ImageNet-CoG Benchmark introduced in the paper *"Concept Generalization in Visual Representation Learning"* (ICCV 2021). It contains code for reproducing all the experiments reported in the paper, as well as instructions on how to evaluate any custom model on the ImageNet-CoG Benchmark.



```
@InProceedings{sariyildiz2021conceptgeneralization,
    title={Concept Generalization in Visual Representation Learning},
    author={Sariyildiz, Mert Bulent and Kalantidis, Yannis and Larlus, Diane and Alahari, Karteek},
    booktitle={International Conference on Computer Vision},
    year={2021}
}
```

**Contents of the Readme file:**
- [Prerequisites: Benchmark data and code](#prerequisites-benchmark-data-and-code)
  - [Installation](#installation)
  - [Data](#data)
- [Evaluating a model on the ImageNet-CoG benchmark](#evaluating-a-model-on-the-imagenet-cog-benchmark)
  - [CoG step-1: Model preparation](#cog-step-1-model-preparation)
  - [CoG step-2: Feature extraction](#cog-step-2-feature-extraction)
  - [CoG step-3: Learning linear classifiers and testing](#cog-step-3-learning-linear-classifiers-and-testing)

# Prerequisites: Benchmark data and code

## Installation

We developed the benchmark using:
- GCC 7.5.0
- Python 3.7.9
- PyTorch 1.7.1
- Torchvision 0.8.2
- CUDA 10.2.89
- [Optuna](https://optuna.org/) 2.3.0
- [YACS](https://github.com/rbgirshick/yacs) 0.1.8
- termcolor 1.1.0

We recommend creating a separate conda environment for the benchmark:

```
conda create -n cog python=3.7.9
conda activate cog
conda install -c pytorch pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2
conda install -c conda-forge optuna=2.3
conda install termcolor
pip install yacs
```

**Note**: To reproduce the results for SimCLR-v2 and BYOL, you will also need [TensorFlow](https://www.tensorflow.org/), as well as the repos that contain code for converting the [SimCLR-v2](https://github.com/Separius/SimCLRv2-Pytorch) and [BYOL](https://github.com/chigur/byol-convert) pre-trained models into PyTorch. More info can be found in the corresponding scripts under [prepare_models/](./prepare_models).

## Data

To evaluate a model on the ImageNet-CoG benchmark you will need the following data:
- [The full ImageNet dataset (IN-21K)](#the-full-imagenet-dataset-in-21k)
- [The ILSVRC-2012 dataset (IN-1K)](#the-ilsvrc-2012-dataset-in-1k)
- [The CoG level files (for L<sub>1</sub>, L<sub>2</sub>, L<sub>3</sub>, L<sub>4</sub>, L<sub>5</sub>)](#the-cog-level-files)

<!-- omit in toc -->
#### The full ImageNet dataset (IN-21K)
The levels of ImageNet-CoG consist of a selection of 5K synsets in the full ImageNet.
To download the full ImageNet, you need to create an account on [the ImageNet website](https://image-net.org/index.php).
Then, under the "Download" section, you will find the "Winter 2021" release (a tar file of size 1.1TB).
Once you download it, extract it to folder `<imagenet_images_root>` (you will need this path later when extracting features).
After extracting you should see the following folder structure, i.e., a separate folder of images per synset:
```
<imagenet_images_root>
...
|--- n07696728
|   |--- *.JPEG
|--- n11944954
|   |--- *.JPEG
...
```
<!-- omit in toc -->
#### The ILSVRC-2012 dataset (IN-1K)
To evaluate models on the pretraining dataset, you will need the ubiquitous ILSVRC-2012 subset of ImageNet, which we also refer to as _IN-1K_.
It is also available on [the ImageNet website](https://image-net.org/download-images.php).
You will download two tar files for training (138GB) and validation (6.3GB) images, and extract them under `<in1k_images_root>` (again, you will need this path later).
We expect the following folder structure for IN-1K:
```
<in1k_images_root>
...
|--- train
|   |--- n11939491
|       |--- *.JPEG
|   |--- n07836838
|       |--- *.JPEG
|--- val
|   |--- n11939491
|       |--- *.JPEG
|   |--- n07836838
|       |--- *.JPEG
...
```

<!-- omit in toc -->
#### The CoG level files
 These are files that contain the concepts and data splits for ImageNet-CoG, and can be directly downloaded from the links below:
* [cog_levels_mapping_file.pkl](http://download.europe.naverlabs.com/ComputerVision/cog_levels_mapping_file.pkl): List of ImageNet concept names for each ImageNet-CoG level (~100KB).
* [cog_concepts_split_file.pkl](http://download.europe.naverlabs.com/ComputerVision/cog_concepts_split_file.pkl): List of image filenames in the train and test splits for all 5000 ImageNet concepts in the CoG levels (~678MB).

(Note: If clicking on the file names does not open a pop-up window for download, try 1) entering the file URLs directly on the address bar of your browser, or 2) using `wget` by giving the file URLs as arguments.)


# Evaluating a model on the ImageNet-CoG benchmark

After [installing the required packages and downloading the ImageNet data](#prerequisites-benchmark-data-and-code), you can follow the steps below:

1. [CoG step-1](#cog-step-1-model-preparation): Prepare the model you want to evaluate.
1. [CoG step-2](#cog-step-2-feature-extraction): Extract image features for IN-1K and the CoG levels using the frozen backbone of the model.
1. [CoG step-3](#cog-step-3-learning-linear-classifiers-and-testing): Train linear classifiers on the pre-extracted features and measure accuracy.


## CoG step-1: Model preparation

The ImageNet-CoG benchmark is designed to evaluate models that are _pre-trained on IN-1K (the ILSVRC-2012 dataset)_.


<!-- omit in toc -->
### Models evaluated in the paper

To reproduce the results for any of the models evaluated in our [paper](https://arxiv.org/abs/2012.05649) you can follow [the instructions](./prepare_models/README.md) for downloading the their checkpoints.
You will end up with all the checkpoints under the folder `<models_root_dir>`.
Note that every model has a unique name that should be passed with the `--model` argument in all the scripts below.
You can find the model names we used for all the models evaluated in our paper in [this table](./prepare_models/README.md#table-of-models).

<!-- omit in toc -->
### Custom models

Follow these three steps to prepare your model for evaluation on the CoG benchmark.

1. **Give a name to your model:**
Every model has a unique name that is passed with the `--model=<model_name>` argument in all the scripts below.
The model name consists of two parts `<model_name>="<model_title>_<architecture_name>"`.
You need to give such a name to your model, e.g., `<myModel_myArchitecture>`.

1. **Write a model loader function:**
To be able to extract features from your custom model, the [load_pretrained_backbone()](./model_utils.py#L93) function  must be able to load your pretrained model correctly.
To do so, you will need to add a custom model loader function in [model_loaders.py](./model_loaders.py), that takes as input an arguments dict [`init_args`](./model_utils.py#L99) and returns the backbone module and its forward function, e.g., `backbone, forward = load_my_model(init_args)`

1. **Register your model:**
Add a new element in `MODEL_LOADER_DICT` in the [model_loader_dict.py](./model_loader_dict.py) script specifying the name and loader function for your model, e.g.:
```
[...]
"myModel_myArchitecture": load_my_model,
[...]
```

## CoG step-2: Feature extraction

In this step, you will extract image features for the 6 datasets (IN-1K and our CoG levels L<sub>1</sub>, L<sub>2</sub>, L<sub>3</sub>, L<sub>4</sub>, L<sub>5</sub>).
We provide for you a bash script to automatize feature extraction [./bash-scripts/extract_features.sh](./bash-scripts/extract_features.sh) for a given model.
As you notice, *there are variables in this script that you need to set properly*;
Moreover, this script calls [extract_features.py](./extract_features.py) with the provided arguments for each of 6 datasets.
In our GPU cluster, each feature extraction experiment takes ~60min using a V100 GPU and 8 CPU cores.

Please see below for the documentation of the arguments and several examples for [extract_features.py](./extract_features.py).

<details>
<summary>(Click to see the details) Documentation of the arguments for extract_features.py:</summary>

* **Model arguments:**
    * `--model=<model_name>`: The name of the model using which you would like to extract features. It can be one of the models we report in the paper (see [the table with model names for those](./prepare_models/README.md#table-of-models)) or your custom model. This argument will point the code to the proper checkpoint loader function for your model.
    * For loading checkpoint files you have two options (it is enough to set one of them):
        * `--models_root_dir=<models_root_dir>`: If set, the script will look for a model checkpoint at path `<models_root_dir>/<model_title>/<architecture_name>/model.ckpt`,
        * `--ckpt_file=<ckpt_file>`: The full path of any valid checkpoint of the model `<model_name>`.

* **Dataset and split arguments:** You can specify the dataset and split by using the following arguments:
    * `--dataset=<dataset>`: The dataset to extract features from (`"in1k", "cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5"`)
    * `--split=<split>`: The split (`"train"` or `"test"`). Note that for the IN-1K dataset, "test" will extract features from the official validation set.

* **Arguments for the dataset (IN-1K and IN-21K) folders, and the CoG level files:**
    * `--in1k_images_root=<in1k_images_root>`: The full path to the folder you downloaded IN-1K. This argument is required only when extracting features from IN-1K.
    * For the CoG levels (required only when extracting features from any of the CoG levels):
        * `--imagenet_images_root=<imagenet_images_root>`: The full path to the folder with IN-21K.
        * `--cog_levels_mapping_file=cog_levels_mapping_file.pkl`: The full path to the CoG concept-to-level mapping file that you can download from [the section above](#the-cog-level-files).
        * `--cog_concepts_split_file=cog_concepts_split_file.pkl`: The full path to the CoG concepts split file that you can download from [the section above](#the-cog-level-files).

* **Arguments for an output folder for features:** For specifying where to save features you have two options (it is enough to set one of them):
    * `--models_root_dir=<models_root_dir>`: If set, features will be saved to the path `<models_root_dir>/<model_title>/<architecture_name>/<dataset>/features_<split>/X_Y.pth`.
    * `--output_dir=<output_dir>`: The full path to the *folder* where features will be stored (i.e., `<output_dir>/X_Y.pth`)

</details>

<details>
<summary>(Click to see the details) Examples for running extract_features.py:</summary>

To extract features for the *training* set of *IN-1K* from the "ResNet-50" model (which is the supervised baseline we use in the paper), run this command:
```
python extract_features.py \
    --model="sup_resnet50" \
    --dataset="in1k" \
    --split="train" \
    --models_root_dir=<models_root_dir> \
    --in1k_images_root=<in1k_images_root>
```

Note that `--models_root_dir` is set.
Therefore, this command will load the pretrained weights of "ResNet50" from `<models_root_dir>/sup/resnet50/model.ckpt` and save the features into the file `<models_root_dir>/sup/resnet50/in1k/features_train/X_Y.pth`.

To extract the features for the *test* set of *L<sub>5</sub>* from the "SwAV" model, run this command:
```
python extract_features.py \
    --model="swav_resnet50" \
    --dataset="cog_l5" \
    --split="test" \
    --models_root_dir=<models_root_dir> \
    --imagenet_images_root=<imagenet_images_root> \
    --cog_levels_mapping_file=cog_levels_mapping_file.pkl \
    --cog_concepts_split_file=cog_concepts_split_file.pkl
```
This script will load the pretrained weights of "SwAV" from `<models_root_dir>/swav/resnet50/model.ckpt` and save the features into the file `<models_root_dir>/swav/resnet50/cog_l5/features_test/X_Y.pth`

Also don't forget to check the bash script [./bash-scripts/extract_features.sh](./bash-scripts/extract_features.sh).
</details>


## CoG step-3: Learning linear classifiers and testing

After extracting features for the 6 datasets (for their both training and test sets), we train linear classifiers on them.
We divide the classification experiments into two settings:

1. **Learning with all available  images**: Training classifiers with all available training data for the concepts.
(This setting is to reproduce the scores we report in *Section 4.2.1 - Generalization to unseen concepts* of [our paper](https://arxiv.org/abs/2012.05649).)
We train 5 classifiers with different seeds on each CoG level and IN-1K separately, then report their average score.
So, for this setting, you will need to train 30 (6 datasets, 5 seeds) logistic regression classifiers using all available training data for the concepts.
We provide for you a bash script [./bash-scripts/logreg.sh](./bash-scripts/logreg.sh) to automatize running 5 classifiers on a specific dataset (or on a specific training and test set features).
*There are variables in this script that you need to set properly*.
Then, this script calls [logreg.py](./logreg.py) with the provided arguments for 5 seeds.
Finally, it prints the average score of these 5 experiments.
In our GPU cluster, each experiment takes ~75min using a V100 GPU and 8 CPU cores.

2. **Few-shot**: Training classifiers with N in {1, 2, 4, 8, 16, 32, 64 or 128} training samples per concept.
(This setting is to reproduce the scores we report in *Section 4.2.2 - How fast can models adapt to unseen concepts?* of [our paper](https://arxiv.org/abs/2012.05649).)
We again train 5 classifiers with different seeds on each dataset separately, then report their average score.
But this time, we also change the number of training samples per concept.
So, for this setting, you will need to train 240 (8 x 6 x 5) logistic regression classifiers.
We provide for you a bash script [./bash-scripts/fewshot.sh](./bash-scripts/fewshot.sh) to automatize running 5 classifiers on a specific dataset for N in {1, 2, 4, 8, 16, 32, 64 or 128}.
*There are variables in this script that you need to set properly*.
Then this script calls [logreg.py](./logreg.py) with the provided arguments for 5 seeds and each N value.
Finally, it prints the average score for each N.

**Note on hyper-parameter tuning.**
To minimize performance differences due to sub-optimal hyper-parameters, we use the [Optuna](https://optuna.org/) hyperparameter optimization framework to tune the learning rate and weight decay hyper-parameters when training a classifier.
We sample 30 learning rate and weight decay pairs and perform hyper-parameter tuning by partitioning _the training set_ into two temporary subsets used for hyper-parameter training and validation.
Once the optimal hyper-parameters are found, then we train the final classifier with these hyper-parameters on the complete training set and report the accuracy on the test set.
We combine hyper-parameter tuning, classifier training and testing in a single script; one just needs to run the [logreg.py](./logreg.py) script.

Please see below for the documentation of the arguments and several examples for [logreg.py](./logreg.py).

<details>
<summary>(Click to see the details) Documentation of the arguments for logreg.py:</summary>

* **Arguments for training and test set features**:
To train a classifier, you need to load training and test set features extracted for a particular dataset (i.e., `"X_Y.pth"` files you obtained in the feature extraction step above).
You have two options for specifying from where to load pre-extracted features (it is enough to set the arguments for one option):
    * Option-1: Loading pre-extracted features for a particular dataset extracted from a particular model located under `<models_root_dir>`.
    By setting the three arguments below, the script will look for features files `<models_root_dir>/<model_name>/<architecture_name>/<dataset>/features_{train and test}/X_Y.pth`.
    Please refer to [the feature extraction part above](#cog-step-2-feature-extraction) to see how to set these arguments correctly.
        * `--model=<model_name>`: The name of the model.
        * `--dataset=<dataset>`: The name of the dataset.
        * `--models_root_dir=<models_root_dir>`: The full path to the models directory.
    * Option-2: Loading pre-extracted features from arbitrary paths.
    With these two arguments set, the script will look for training and test set features files `<train_set_features_dir>/X_Y.pth` and `<test_set_features_dir>/X_Y.pth`, respectively.
        * `--train_features_dir=<train_features_dir>`: The full path to the folder containing training set features.
        * `--test_features_dir=<test_features_dir>`: The full path to the folder containing test set features.

* **Arguments for an output folder for logs:**
[logreg.py](./logreg.py) produces output logs that can be read by the [print_results.py](./print_results.py) script to print the average of the classification results obtained with multiple seeds.
For specifying where to save logs you have two options (it is enough to set one of them):
    * `--models_root_dir=<models_root_dir>`: If set, the output logs will be saved under the folder `<models_root_dir>/<model_name>/<architecture_name>/<dataset>/eval_logreg/seed-<seed>/`.
    * `--output_dir=<output_dir>`: The full path to the output folder where logs will be saved.

* **Seed for reproducibility**:
    * `EVAL.SEED <seed_number>`: Seed for Python's random module, NumPy and PyTorch.
    Its default value is 22.
    Note that this is not an argument but a [config](./configs/default.py) entry.
    So you need to append `EVAL.SEED <seed_number>` to the arguments list to overwrite the default config value (see the example below).
    We run all classification experiments 5 times, and report the mean and variance of these runs and _strongly recommend you to do the same_.

* **Number of training sample per concept (for few-shot learning):**
    * `CLF.N_SHOT <img_per_class>`: The number of training samples per concept to use for training a few-shot classifier.
    Note that this is not an argument but a [config](./configs/default.py) entry.
    So you need to append `CLF.N_SHOT <img_per_class>` to the arguments list to overwrite the default config value (see the example below).

**Averaging results over multiple seeds.**
The [logreg.py](./logreg.py) script stores top-1 and top-5 accuracies per epoch in a Python list and saves them into a pickle file named `final-*/logs.pkl` under the output logs folder.
For getting the final mean and variance over multiple runs, i.e., after running the logreg.py script with multiple seeds, one can run the [print_results.py](./print_results.py) script by providing the output logs directory with the `--logreg_root_dir` argument.

</details>

<details>
<summary>(Click to see the details) Examples for running logreg.py:</summary>

To train a linear classifier (including its hyper-parameter tuning phase) on top of L<sub>5</sub> features extracted by SwAV, run:

```
python logreg.py \
    --model="swav_resnet50" \
    --dataset="cog_l5" \
    --models_root_dir=<models_root_dir> \
    EVAL.SEED 55  # --> training the classifier with seed 55
```

This script will load pre-extracted features from `<models_root_dir>/swav/resnet50/cog_l5/features_{train and test}/X_Y.pth` and save output logs under `<models_root_dir>/swav/resnet50/cog_l5/eval_logreg/seed-55`.
It will also print the final top1/top5 accuracies for this run.
To print the averaged scores over 5 seeds for SwAV for L<sub>5</sub>, and assuming one used the default output paths, the command is:

```
python print_results.py \
    --logreg_root_dir="<models_root_dir>/swav/resnet50/cog_l5/eval_logreg"
```

To simulate few-shot learning scenarios, you can overwrite the config entry:
```
python logreg.py \
    --model="swav_resnet50" \
    --dataset="cog_l5" \
    --models_root_dir=<models_root_dir> \
    CLF.N_SHOT 8 # --> training the classifier with 8 random training samples per concept
```
This script will use 8 random training samples per concept to train linear classifiers and save output logs under `<models_root_dir>/swav/resnet50/cog_l5/eval_logreg_N8/seed-22`.

Also don't forget to check the bash scripts [./bash-scripts/logreg.sh](./bash-scripts/logreg.sh) and [./bash-scripts/fewshot.sh](./bash-scripts/fewshot.sh).

</details>
