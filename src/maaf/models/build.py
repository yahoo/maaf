# Copyright 2022 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.
"""Functions to put everything together and build the model."""

import sys
import os
import torch
from . import composition_models
from .heads import get_task_head
from .image_model import build_image_model
from .text_model import build_text_model
from ..config.compat import MAAF_ALIASES
from ..config import get_config
from ..datasets.datasets import load_dataset


def load_model(path, modifications=[], strict=True):
    config_file = os.path.join(path, "config.yaml")
    model, task, cfg = build_from_config_file(config_file, modifications,
                                              strict_loading=strict)
    checkpoint = os.path.join(path, "latest_checkpoint.pth")
    state_dict = torch.load(checkpoint, map_location=model.device)["model_state_dict"]
    model.load_state_dict(state_dict, strict=strict)
    return model


def build_from_config_file(path, modifications=[], strict_loading=True):
    cfg = get_config()
    cfg.merge_from_file(path)
    cfg.merge_from_list(modifications)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    model, task = build_model(cfg, strict_loading=strict_loading)
    return model, task, cfg


def build_model(cfg, texts=None, strict_loading=True):
    print('Building model', cfg.MODEL.COMPOSITION)

    tokenizer_needs_texts = cfg.MODEL.TEXT_MODEL.TOKENIZER == "simple" and \
        cfg.MODEL.TEXT_MODEL.ARCHITECTURE is not None
    if texts is None and tokenizer_needs_texts:
        texts = load_dataset(cfg)["train"].get_all_texts()

    if "clip" in cfg.MODEL.COMPOSITION:
        from .clip import get_clip_class
        ModelClass, kwargs = get_clip_class(cfg)
    else:
        kwargs = {"image_model": build_image_model(cfg),
                  "text_model": build_text_model(texts, cfg)}
    if cfg.MODEL.COMPOSITION == 'imgonly':
        ModelClass = composition_models.SimpleModelImageOnly
    elif cfg.MODEL.COMPOSITION == 'textonly':
        ModelClass = composition_models.SimpleModelTextOnly
    elif cfg.MODEL.COMPOSITION in MAAF_ALIASES \
            or cfg.MODEL.COMPOSITION in ["residualMAAF", "resmaaf"]:
        if cfg.MODEL.COMPOSITION in ["residualMAAF", "resmaaf"]:
            print("Setting up Residual MAAF")
            ModelClass = composition_models.ResidualMAAF
        else:
            print("Setting up MAAF")
            ModelClass = composition_models.MAAF
        kwargs.update({
            "model_dim": cfg.MODEL.EMBED_DIM,
            "num_heads": cfg.MODEL.MAAF.ATTENTION_HEADS,
            "ff_width": cfg.MODEL.MAAF.BLOCK_WIDTH,
            "dropout": cfg.MODEL.DROPOUT_RATE,
            "num_blocks": cfg.MODEL.MAAF.NUM_BLOCKS,
            "position_encodings": cfg.MODEL.MAAF.POSITION_ENCODING,
            "softmax_replacement": cfg.MODEL.MAAF.ATTN_SOFTMAX_REPLACEMENT,
            "output": cfg.MODEL.MAAF.OUTPUT,
        })
    elif cfg.MODEL.COMPOSITION == 'concat':
        ModelClass = composition_models.Concat
        kwargs.update({"embed_dim": cfg.MODEL.EMBED_DIM,
                       "dropout": cfg.MODEL.DROPOUT_RATE})
    elif cfg.MODEL.COMPOSITION == 'add':
        ModelClass = composition_models.Addition
    elif "clip" in cfg.MODEL.COMPOSITION:
        pass
    elif cfg.MODEL.COMPOSITION == "random":
        ModelClass = composition_models.RandomComposition
    else:
        print('Invalid model', cfg.MODEL.COMPOSITION)
        sys.exit()

    head, task = get_task_head(cfg)

    model = ModelClass(head, **kwargs)

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if cfg.MODEL.WEIGHTS is not None:
        print("Loading model weights from", cfg.MODEL.WEIGHTS)
        loaded_dict = torch.load(cfg.MODEL.WEIGHTS, map_location=model.device)
        model.load_state_dict(loaded_dict["model_state_dict"],
                              strict=strict_loading)

    return model, task


def get_optimizer(cfg, model):
    # create optimizer
    param_dicts = []
    gathered_params = set()
    # apply learning rate adjustments for model components
    image_fc = [p for p in model.image_model_fc_parameters()]
    gathered_params.update(image_fc)
    param_dicts.append({
        'params': image_fc,
        'lr': cfg.SOLVER.LEARNING_RATE
    })
    image_params = model.image_model_parameters(
        include_scratch=cfg.SOLVER.PROJECTION_LR_TIED_TO_PRETRAINED)
    other_img = [p for p in image_params if p not in gathered_params]
    gathered_params.update(other_img)
    param_dicts.append({
        'params': other_img,
        'lr': cfg.SOLVER.PRETRAINED_WEIGHT_LR_FACTOR_IMAGE * cfg.SOLVER.LEARNING_RATE
    })

    text_params = model.text_model_parameters(
        include_scratch=cfg.SOLVER.PROJECTION_LR_TIED_TO_PRETRAINED)
    text_params = [p for p in text_params]
    gathered_params.update(text_params)
    param_dicts.append({
        'params': text_params,
        'lr': cfg.SOLVER.PRETRAINED_WEIGHT_LR_FACTOR_TEXT * cfg.SOLVER.LEARNING_RATE
    })
    param_dicts.append(
        {'params': [p for p in model.parameters() if p not in gathered_params]})

    if cfg.SOLVER.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(
            param_dicts,
            lr=cfg.SOLVER.LEARNING_RATE,
            eps=1e-4,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(
            param_dicts, lr=cfg.SOLVER.LEARNING_RATE, momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    return optimizer
