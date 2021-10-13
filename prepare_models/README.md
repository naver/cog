<!--
ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​
-->

# Preparing pre-trained models benchmarked in the paper

In our [paper](https://arxiv.org/abs/2012.05649), we benchmark a large number of visual representation learning models on ImageNet-CoG.
See the [table of models](#table-of-models) below for a complete list of models.
In this folder, you can find scripts with instructions for downloading all the models we evaluate in the paper.
We generally evaluated pre-trained models provided by the corresponding paper authors.

To keep things tidy, we give a name to each of the pretrained models `<model_name>` that is composed into two parts with an underscore between: `<model_name>=<model_title>_<architecture_name>`.
In the [table of models](#table-of-models) below, you will see model names in `<model_name>` form.
Moreover, we download each model under `<models_root_dir>` and define a folder structure like this:
```bash
<models_root_dir>
|--- <model_title>
|   |--- <architecture_name>
|       |--- <model.ckpt>
...
```

Each model comes with a bash script (`prepare_*.sh`) that either directly downloads or contains instructions for how to download (and possibly convert) each of the pretrained models released by its authors.
The bash scripts we provide here expect a single argument, `<models_root_dir>` that specifies the root directory of all models.
Then each model is downloaded under `<models_root_dir>/<model_title>/<architecture_name>`.

E.g. to download the pretrained Barlow-Twins model with ResNet50 architecture one should execute the command:
```bash
> bash prepare_barlow.sh <models_root_dir>
```

We further provide a script to download _all_ the models evaluated in the paper:
```bash
> bash all.sh <models_root_dir>
```
*But we strongly encourage you to read each `prepare_*.sh` script*, because some of them need you to manually download model checkpoints.


**Notes**:
- Some of the pretrained models are hosted on cloud storage platforms, which need to be downloaded manually under `<models_root_dir>/<model_title>/<architecture_name>/model.ckpt`, e.g., for [`prepare_t2tvit.sh`](./prepare_t2tvit.sh). Please take a look at each `prepare_*.sh` before calling them.
- For models originally published in TensorFlow (i.e., SimCLR-v2 and BYOL), you need to first convert the pretrained TensorFlow weights of the models into PyTorch.
Instruction on how to do that are detailed in files [prepare_simclr.sh](./prepare_models/prepare_simclr.sh) and [prepare_byol.sh](./prepare_models/prepare_byol.sh) for SimCLR-v2 and BYOL, respectively.
- To initialize pretrained CLIP models, we use the API provided by its authors. See [`prepare_clip.sh`](./prepare_clip.sh) for details.
- To initialize pretrained EfficientNet models, we use a 3rd-part API (named pycls) provided by a FAIR team. See [`prepare_efficientnet.sh`](./prepare_efficientnet.sh) for details.

# Table of models

In the table of models below, we list the models that we benchmarked in our paper. For each model, we list its model name (to be provided in all our scripts using the ``--model`` argument), the corresponding publication title, and a link to the bash script we provide in this repo that downloads the exact pretrained weights we used.

| Model name              | Publication title                                                                     | Bash script to download                              |
|-------------------------|---------------------------------------------------------------------------------------|------------------------------------------------------|
| sup_resnet50            | Deep Residual Learning for Image Recognition                                          | [prepare_sup.sh](./prepare_sup.sh)                   |
| sup_resnet152           | Deep Residual Learning for Image Recognition                                          | [prepare_sup.sh](./prepare_sup.sh)                   |
| sup_inception-v3        | Rethinking the Inception Architecture for Computer Vision                             | [prepare_sup.sh](./prepare_sup.sh)                   |
| sup_vgg19               | Very Deep Convolutional Networks for Large-Scale Image Recognition                    | [prepare_sup.sh](./prepare_sup.sh)                   |
| deit_small              | Training data-efficient image transformers & distillation through attention           | [prepare_deit.sh](./prepare_deit.sh)                 |
| deit_small_distilled    | Training data-efficient image transformers & distillation through attention           | [prepare_deit.sh](./prepare_deit.sh)                 |
| deit_base_distilled_384 | Training data-efficient image transformers & distillation through attention           | [prepare_deit.sh](./prepare_deit.sh)                 |
| nat_m4                  | Neural Architecture Transfer                                                          | [prepare_nat.sh](./prepare_nat.sh)                   |
| efficientnet_b1         | EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks              | [prepare_efficientnet.sh](./prepare_efficientnet.sh) |
| efficientnet_b4         | EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks              | [prepare_efficientnet.sh](./prepare_efficientnet.sh) |
| t2tvit_t14              | Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet            | [prepare_t2tvit.sh](./prepare_t2tvit.sh)             |
| simclrv2_resnet50       | Big Self-Supervised Models are Strong Semi-Supervised Learners                        | [prepare_simclr.sh](./prepare_simclr.sh)             |
| byol_resnet50           | Bootstrap your own latent: A new approach to self-supervised Learning                 | [prepare_byol.sh](./prepare_byol.sh)                 |
| mocov2_resnet50         | Momentum Contrast for Unsupervised Visual Representation Learning                     | [prepare_moco.sh](./prepare_moco.sh)                 |
| mochi_resnet50          | Hard Negative Mixing for Contrastive Learning                                         | [prepare_mochi.sh](./prepare_mochi.sh)               |
| infomin_resnet50        | What Makes for Good Views for Contrastive Learning?                                   | [prepare_infomin.sh](./prepare_infomin.sh)           |
| swav_resnet50           | Unsupervised Learning of Visual Features by Contrasting Cluster Assignments           | [prepare_swav.sh](./prepare_swav.sh)                 |
| dino_resnet50           | Emerging Properties in Self-Supervised Vision Transformers                            | [prepare_dino.sh](./prepare_dino.sh)                 |
| obow_resnet50           | Online Bag-of-Visual-Words Generation for Unsupervised Representation Learning        | [prepare_obow.sh](./prepare_obow.sh)                 |
| barlow_resnet50         | Barlow Twins: Self-Supervised Learning via Redundancy Reduction                       | [prepare_barlow.sh](./prepare_barlow.sh)             |
| compress_resnet50       | CompRess: Self-Supervised Learning by Compressing Representations                     | [prepare_compress.sh](./prepare_compress.sh)         |
| mixup_resnet50          | mixup: Beyond Empirical Risk Minimization                                             | [prepare_cutmix.sh](./prepare_cutmix.sh)             |
| manifold-mixup_resnet50 | Manifold Mixup: Better Representations by Interpolating Hidden States                 | [prepare_cutmix.sh](./prepare_cutmix.sh)             |
| cutmix_resnet50         | CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features | [prepare_cutmix.sh](./prepare_cutmix.sh)             |
| relabel_resnet50        | Re-labeling ImageNet: from Single to Multi-Labels, from Global to Localized Labels    | [prepare_relabel.sh](./prepare_relabel.sh)           |
| advrobust_resnet50      | Do Adversarially Robust ImageNet Models Transfer Better?                              | [prepare_advrobust.sh](./prepare_advrobust.sh)       |
| mealv2_resnet50         | MEAL V2: Boosting Vanilla ResNet-50 to 80%+ Top-1 Accuracy on ImageNet without Tricks | [prepare_mealv2.sh](./prepare_mealv2.sh)             |
| mopro_resnet50          | MoPro: Webly Supervised Learning with Momentum Prototypes                             | [prepare_mopro.sh](./prepare_mopro.sh)               |
| ssup_resnet50           | Billion-scale semi-supervised learning for image classification                       | [prepare_swsup.sh](./prepare_swsup.sh)               |
| swsup_resnet50          | Billion-scale semi-supervised learning for image classification                       | [prepare_swsup.sh](./prepare_swsup.sh)               |
| clip_resnet50           | Learning Transferable Visual Models From Natural Language Supervision                 | [prepare_clip.sh](./prepare_clip.sh)                 |

