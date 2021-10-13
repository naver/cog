# ImageNet-CoG Benchmark
# Copyright 2021-present NAVER Corp.
# 3-Clause BSD License​

import sys
import torch
from pathlib import Path


def load_deit(init_args):
    """
    Load pretrained Deit models.
    """
    model_name = init_args["model_name"]
    model_dir = init_args["model_dir"]
    ckpt_file = init_args["ckpt_file"]

    # the config files for the deit models are not defined in timm
    # therefore we also need models.py file provided in the deit repository
    # we must have downloaded it under the model directory
    sys.path.insert(
        0,
        str(Path(model_dir).parent),
    )
    from src import models as deit_models

    from timm.models import create_model
    backbone = create_model(
        {
            "deit_small": "deit_small_patch16_224",
            "deit_small_distilled": "deit_small_distilled_patch16_224",
            "deit_base_distilled_384": "deit_base_distilled_patch16_384",
        }[model_name],
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
    )
    state_dict = backbone.state_dict()

    # load model state dict
    ckpt = torch.load(ckpt_file, "cpu")
    checkpoint_model = ckpt['model']

    # remove from the checkpoint
    # the classification layer
    for key_to_remove in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if (key_to_remove in checkpoint_model) and \
           (checkpoint_model[key_to_remove].shape != state_dict[key_to_remove].shape):
            del checkpoint_model[key_to_remove]

    ##################################################
    # To properly setup deit models, we use code from the official deit repo https://github.com/facebookresearch/deit.
    # Lines between 59 and 77 in this file are from https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/main.py
    # Copyright (c) 2015-present, Facebook, Inc.
    # All rights reserved.

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = backbone.patch_embed.num_patches
    num_extra_tokens = backbone.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed
    ##################################################

    backbone.load_state_dict(checkpoint_model, strict=False)

    # note that the backbone is a VisionTransformer
    # defined in timm.models
    # this backbone provides a function to extract features
    # from the 0-th token, which corresponds to the class token
    forward = backbone.forward_features
    # but the distilled version actually outputs two tensors (including distillation token)
    if "distilled" in model_name:
        def forward(x):
            return backbone.forward_features(x)[0]

    return backbone, forward
