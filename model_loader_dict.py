# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

from model_loaders import *

##################################################
# Dictionary that maps model_name into its loader.
##################################################

MODEL_LOADER_DICT = {
    "sup_resnet50": load_default,
    "sup_resnet152": load_sup_resnet152,
    "sup_inception-v3": load_sup_inception_v3,
    "sup_vgg19": load_sup_vgg19,
    "deit_small": load_deit,
    "deit_small_distilled": load_deit,
    "deit_base_distilled_384": load_deit,
    "nat_m4": load_nat,
    "efficientnet_b1": load_efficientnet,
    "efficientnet_b4": load_efficientnet,
    "t2tvit_t14": load_t2tvit,
    "simclrv2_resnet50": load_simclrv2,
    "byol_resnet50": load_byol,
    "mocov2_resnet50": load_moco,
    "mochi_resnet50": load_moco,
    "infomin_resnet50": load_infomin,
    "swav_resnet50": load_swav,
    "dino_resnet50": load_dino,
    "obow_resnet50": load_obow,
    "barlow_resnet50": load_barlow,
    "compress_resnet50": load_compress,
    "mixup_resnet50": load_cutmix,
    "manifold-mixup_resnet50": load_cutmix,
    "cutmix_resnet50": load_cutmix,
    "relabel_resnet50": load_cutmix,
    "advrobust_resnet50": load_advrobust,
    "mealv2_resnet50": load_mealv2,
    "mopro_resnet50": load_mopro,
    "ssup_resnet50": load_default,
    "swsup_resnet50": load_default,
    "clip_resnet50": load_clip,
    ###########
    # Add the loader of your model here
    # "myModel_myArchitecture": load_my_model  # --> you provide this function in model_loaders.py
    ###########
}