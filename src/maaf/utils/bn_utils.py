# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


import torch


def apply_bn_mode(model, bn_mode):
    if bn_mode == "freeze_bn":
        freeze_bn(model)
    elif bn_mode == "freeze_except_bn":
        freeze_except_bn(model)
    elif bn_mode == "freeze_bn_averages":
        change_bn_mode(model, bn_mode="eval")
    elif bn_mode == "freeze_except_bn_averages":
        freeze_except_bn(model)
        change_bn_mode(model, bn_mode="train")
    elif bn_mode == "ordinary":
        pass
    else:
        raise ValueError(f"Invalid batch norm mode {bn_mode}")


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()


def freeze_except_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(True)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(True)
            module.train()
        else:
            for param in module.parameters():
                param.requires_grad_(False)
            module.eval()


def change_bn_mode(model, bn_mode="eval"):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if bn_mode == "eval":
                module.eval()
            elif bn_mode == "train":
                module.train()
            else:
                raise ValueError(f"Invalid bn_mode {bn_mode}")
