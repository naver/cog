# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

import torch
import sys
import os
import json
from pathlib import Path

from model_loaders_deit import load_deit

##################################################
# Pretrained model loaders
#
# For your custom model, you can write its loader below.
# Example:
# def load_my_model(init_args):
#     ...
#     return backbone, forward
##################################################


def load_sup_vgg19(init_args):
    """
    Load pretrained Supervised VGG19-BN Classifier.
    """
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import vgg19_bn
    backbone = vgg19_bn(pretrained=False)
    backbone.load_state_dict(torch.load(ckpt_file, "cpu"))

    # remove the last FC layer in vgg19
    # nn.Linear(4096, num_classes)
    del backbone.classifier[-1]

    return backbone, None


def load_sup_inception_v3(init_args):
    """
    Load pretrained Supervised Inception-v3 Classifier.
    """
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import inception_v3
    backbone = inception_v3(pretrained=False)
    backbone.load_state_dict(torch.load(ckpt_file, "cpu"))

    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    def forward(x):
        return inception_v3_forward(backbone, x)

    return backbone, forward


def load_sup_resnet152(init_args):
    """
    Load pretrained Supervised ResNet152 Classifier.
    """
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import resnet152
    backbone = resnet152(pretrained=False)
    backbone.load_state_dict(torch.load(ckpt_file, "cpu"))

    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    return backbone, None


def load_mealv2(init_args):
    """
    Load pretrained MEAL-v2 model.
    """
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=False)
    ckpt = torch.load(ckpt_file, "cpu")

    # rename meal_v2 pretrained keys
    # note that state_dict is ckpt itself
    state_dict = ckpt
    for k in list(state_dict.keys()):
        # remove prefix
        state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # make sure we loaded all CNN parameters
    backbone.load_state_dict(state_dict, strict=True)

    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    return backbone, None


def load_moco(init_args):
    """
    Load pretrained MoCo models.
    Note that any model that is saved as MoCo pretrained models are saved (e.g., MoCHi) can also be loaded with this function.
    """
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=False)
    ckpt = torch.load(ckpt_file, "cpu")

    # rename moco pre-trained keys
    # from https://github.com/facebookresearch/moco/blob/master/main_lincls.py
    state_dict = ckpt["state_dict"]
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # make sure we loaded all CNN parameters except the last fc
    msg = backbone.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    return backbone, None


def load_swav(init_args):
    """
    Load pretrained SwAV models.
    """
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=False)

    # load pretrained weights
    # see https://github.com/facebookresearch/swav/blob/master/hubconf.py
    state_dict = torch.load(ckpt_file, "cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = backbone.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    return backbone, None


def load_dino(init_args):
    """
    Load pretrained DINO models with ResNet50 backbone
    """
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=False)
    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    # load pretrained weights
    state_dict = torch.load(ckpt_file, "cpu")
    backbone.load_state_dict(state_dict, strict=True)

    return backbone, None


def load_barlow(init_args):
    """
    Load pretrained Barlow Twins models.
    """
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=False)
    state_dict = torch.load(ckpt_file, "cpu")

    # load weights
    msg = backbone.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    return backbone, None


def load_infomin(init_args):
    """
    Load pretrained InfoMin Aug. model.
    """
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=False)
    ckpt = torch.load(ckpt_file, "cpu")

    # rename pre-trained keys
    state_dict = ckpt["model"]
    for k in list(state_dict.keys()):
        # ignore module.linear and module.jigsaw
        # retain only the encoder layers
        if k.startswith("module.encoder."):
            state_dict[k[len("module.encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # make sure we loaded all CNN parameters except the last fc
    msg = backbone.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    return backbone, None


def load_simclrv2(init_args):
    """
    Load pretrained SimCLR-v2 model.
    """
    ckpt_file = init_args["ckpt_file"]
    model_dir = init_args["model_dir"]

    # Load the resnet.py that comes with the SimCLR-v2's PyTorch converter
    sys.path.insert(
        0,
        os.path.join(
            model_dir,
            "SimCLRv2-Pytorch",
        ),
    )
    import resnet

    backbone, _ = resnet.get_resnet(depth=50, width_multiplier=1, sk_ratio=0)
    backbone.load_state_dict(torch.load(ckpt_file, "cpu")["resnet"])

    def forward(x):
        # return the tensor obtained at the end of the network
        # prior to global average pooling
        return backbone(x, apply_fc=False)

    return backbone, forward


def load_byol(init_args):
    """
    Load pretrained BYOL model.
    """
    ckpt_file = init_args["ckpt_file"]
    model_dir = init_args["model_dir"]

    # Load the resnet.py that comes with the BYOL's PyTorch converter
    sys.path.insert(
        0,
        os.path.join(
            model_dir,
            "byol-convert",
        ),
    )
    import resnet

    # init the backbone
    backbone = resnet.resnet50()
    backbone.load_state_dict(torch.load(ckpt_file, "cpu"))

    def forward(x):
        """
        the custom forward pass to extract the GAP features
        """
        x = resnet.pad_same(x, backbone.conv1)
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = resnet.pad_same(x, backbone.maxpool)
        x = backbone.maxpool(x)

        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)

        x = backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    return backbone, forward


def load_cutmix(init_args):
    """
    Load pretrained models by NAVER & Clova AI teams, e.g. CutMix, ReLabel.
    """
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=False)

    # load the pretrained checkpoint
    state_dict = torch.load(ckpt_file, "cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    else:
        state_dict = state_dict

    # remove "module."
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # since this is a classifier trained on ImageNet-1K,
    # we can enforce all weights to match
    backbone.load_state_dict(state_dict, strict=True)

    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    return backbone, None


def load_compress(init_args):
    """
    Load the pretrained CompReSS models.
    """
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=False)

    # load the pretrained checkpoint
    state_dict = torch.load(ckpt_file, "cpu")
    state_dict = state_dict["model"]

    # remove "module."
    # also, resnet50 model of compress outputs 8192-D logits, so remove the last FC layer
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "fc" not in k}

    msg = backbone.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    return backbone, None


def load_mopro(init_args):
    """Load pretrained MoPro models"""
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=False)

    # load the pretrained checkpoint
    state_dict = torch.load(ckpt_file, "cpu")
    state_dict = state_dict['state_dict']

    # remove "module."
    # also take only the encoder network
    for k in list(state_dict.keys()):
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    state_dict['fc.weight'] = state_dict['classifier.weight']
    state_dict['fc.bias'] = state_dict['classifier.bias']
    del state_dict['classifier.weight']
    del state_dict['classifier.bias']

    backbone.load_state_dict(state_dict, strict=True)

    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    return backbone, None


def load_advrobust(init_args):
    """Loads adversarially robust models from Salman et al. 2020"""
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=False)

    # load the pretrained checkpoint
    state_dict = torch.load(ckpt_file, "cpu")
    state_dict = state_dict["model"]

    # remove "module."
    # also take only the classifier (ignore the attacker network)
    for k in list(state_dict.keys()):
        if k.startswith('module.model'):
            # remove prefix
            state_dict[k[len("module.model."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    backbone.load_state_dict(state_dict, strict=True)

    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    return backbone, None


def load_obow(init_args):
    """Loads unsupervised BoW prediction model from Gidaris et al. 2020"""
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=False)

    # load the pretrained checkpoint
    state_dict = torch.load(ckpt_file, "cpu")
    state_dict = state_dict["network"]

    # the last FC layer is a mapping from 8K to 2K
    # so we ignore it
    del state_dict['fc.weight']
    del state_dict['fc.bias']
    msg = backbone.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    return backbone, None


def load_t2tvit(init_args):
    """Init pretrained T2T visual transformers"""
    model_name = init_args["model_name"]
    model_dir = init_args["model_dir"]
    ckpt_file = init_args["ckpt_file"]

    # the config files for the T2T-ViT models are not defined in timm
    # therefore we also need models.py file provided in the repository
    # we must have downloaded it under the model directory
    sys.path.insert(
        0,
        str(Path(model_dir).parent),
    )
    from src import models as t2tvit_models

    # We support only this particular model for now
    # populate the dict below to support more t2t-vit models
    assert model_name == "t2tvit_t14"

    from timm.models import create_model, load_checkpoint
    backbone = create_model(
        {"t2tvit_t14": "T2t_vit_t_14"}[model_name],
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
    )

    # Load the exponential moving average of the model
    # (raw one is not provided)
    load_checkpoint(backbone, ckpt_file, use_ema=True, strict=True)

    # note that the backbone is a T2T_ViT
    # this backbone provides a function to extract features
    # from the 0-th token, which corresponds to the class token
    forward = backbone.forward_features

    return backbone, forward


def load_nat(init_args):
    """Load pretrained Neural Architecture Transfer models."""
    model_name = init_args["model_name"]
    model_dir = init_args["model_dir"]
    ckpt_file = init_args["ckpt_file"]

    # we need the model definition and configuration
    # provided by the authors, on their github repo
    # we must have downloaded the repo under the model directory
    src_dir = os.path.join(
        str(Path(model_dir).parent),
        "src"
    )
    sys.path.insert(0, src_dir)
    from codebase.networks.natnet import NATNet

    # we support only NAT-M4 models for now
    assert model_name == "nat_m4"

    # Load the config file for the model
    config_file = os.path.join(
        src_dir,
        "subnets",
        "imagenet",
        "NAT-M4",
        "net.config"
    )
    net_config = json.load(open(config_file))

    # create the backbone
    backbone = NATNet.build_from_config(net_config, pretrained=False)

    # load pretrained checkpoint
    state_dict = torch.load(ckpt_file, "cpu")
    backbone.load_state_dict(state_dict, strict=True)

    # replace the classifier with identity to get the features
    backbone.classifier = torch.nn.Identity()

    return backbone, None


def load_efficientnet(init_args):
    """Load pretrained EfficientNet models released by FAIR."""
    model_name = init_args["model_name"]

    # We will use pycls from FAIR to load EfficientNet models.
    # See prepare_efficientnet.sh for instructions to install this library.
    from pycls.models import effnet

    model_name = model_name.split("_")[1].upper()
    assert model_name in ["B{}".format(i) for i in range(8)]

    backbone = effnet(model_name, pretrained=True)

    # replace the classifier with identity to get the features
    backbone.head.fc = torch.nn.Identity()

    def forward(image):
        # EfficientNet models from pycls are trained in BGR order
        # https://github.com/facebookresearch/pycls/blob/b314d43e918ad8763c5882730ad55a1eef5e8347/pycls/datasets/imagenet.py#L87
        image = torch.flip(image, [1])
        return backbone(image)

    return backbone, forward


def load_clip(init_args):
    """
    Loads the pretrained CLIP model.

    You need to install CLIP first as described on their code repository
    https://github.com/openai/CLIP

    We use the CLIP api to initialize the whole model
    and return the image feature extraction function as forward.
    """
    import clip
    clip_model, _ = clip.load("RN50", device="cuda")
    forward = clip_model.encode_image

    return clip_model, forward


def load_default(init_args):
    """
    Load any pretrained model that can be seemlessly loaded into torchvision.models.resnet50().
    Including:
    - sup (obviously)
    - ssup
    - swsup
    """
    ckpt_file = init_args["ckpt_file"]

    # Load the model available in torchvision
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=False)
    backbone.load_state_dict(torch.load(ckpt_file, "cpu"))

    # remove the fully-connected layer
    backbone.fc = torch.nn.Identity()

    return backbone, None


def inception_v3_forward(net, x):
    """
    Forward pass for the Inception-v3 network to extracts features from the global average pooling layer ignoring the last FC layer.

    Args:
        x: (th.Tensor of size [bs, 3, W, H]) batch of images
    """
    # N x 3 x 299 x 299
    x = net.Conv2d_1a_3x3(x)
    # N x 32 x 149 x 149
    x = net.Conv2d_2a_3x3(x)
    # N x 32 x 147 x 147
    x = net.Conv2d_2b_3x3(x)
    # N x 64 x 147 x 147
    x = net.maxpool1(x)
    # N x 64 x 73 x 73
    x = net.Conv2d_3b_1x1(x)
    # N x 80 x 73 x 73
    x = net.Conv2d_4a_3x3(x)
    # N x 192 x 71 x 71
    x = net.maxpool2(x)
    # N x 192 x 35 x 35
    x = net.Mixed_5b(x)
    # N x 256 x 35 x 35
    x = net.Mixed_5c(x)
    # N x 288 x 35 x 35
    x = net.Mixed_5d(x)
    # N x 288 x 35 x 35
    x = net.Mixed_6a(x)
    # N x 768 x 17 x 17
    x = net.Mixed_6b(x)
    # N x 768 x 17 x 17
    x = net.Mixed_6c(x)
    # N x 768 x 17 x 17
    x = net.Mixed_6d(x)
    # N x 768 x 17 x 17
    x = net.Mixed_6e(x)
    # N x 768 x 17 x 17

    # we ignore the aux branch
    # aux = torch.jit.annotate(Optional[Tensor], None)
    # if self.AuxLogits is not None:
    #     if self.training:
    #         aux = self.AuxLogits(x)

    # N x 768 x 17 x 17
    x = net.Mixed_7a(x)
    # N x 1280 x 8 x 8
    x = net.Mixed_7b(x)
    # N x 2048 x 8 x 8
    x = net.Mixed_7c(x)
    # N x 2048 x 8 x 8
    # Adaptive average pooling
    x = net.avgpool(x)
    # N x 2048 x 1 x 1
    x = net.dropout(x)
    # N x 2048 x 1 x 1
    x = torch.flatten(x, 1)
    # N x 2048

    # we also ignore the last FC layer
    # x = self.fc(x)
    # # N x 1000 (num_classes)

    return x