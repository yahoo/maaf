# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

import torch
import torch.utils.data
from torchvision import transforms as tvt


def get_image_normalizer():
    return tvt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_default_image_transform(clip=False):
    if clip:
        from ..models.clip import get_image_transform_for_clip
        return get_image_transform_for_clip()
    normalizer = get_image_normalizer()
    transform = tvt.Compose([
        tvt.Resize(224),
        tvt.CenterCrop(224),
        tvt.ToTensor(),
        normalizer])
    return transform


def get_augmenting_image_transform(clip=False):
    if clip:
        from ..models.clip import get_augmenting_image_transform_for_clip
        return get_augmenting_image_transform_for_clip()
    normalizer = get_image_normalizer()
    train_transform = tvt.Compose([
        tvt.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.3)),
        tvt.RandomHorizontalFlip(),
        tvt.ToTensor(),
        tvt.Lambda(lambda xx: xx + 0.01 * torch.randn(xx.shape, device="cpu")),
        normalizer
    ])

    return train_transform


def load_dataset(cfg, calibration=None):
    """Loads the input datasets."""
    print('Reading dataset ', cfg.DATASET.NAME)

    if cfg.MODEL.LOSS == "multilabel_soft_margin":
        labels = "one_hot"
    elif cfg.MODEL.LOSS == "mse":
        labels = "index_normed"
    else:
        labels = "index"

    if cfg.MODEL.INCLUDES_IMAGE_TRANSFORM:
        transform = None
    else:
        transform = get_default_image_transform(clip="clip" in cfg.MODEL.COMPOSITION)

    if cfg.DATASET.AUGMENTATION.IMAGE_AUGMENTATION is not None:
        if cfg.DATASET.NAME != "fashioniq" or cfg.MODEL.INCLUDES_IMAGE_TRANSFORM:
            raise NotImplementedError()
        train_transform = get_augmenting_image_transform(
            clip="clip" in cfg.MODEL.COMPOSITION)
    else:
        train_transform = transform

    if cfg.DATASET.NAME == 'fashioniq':
        from .fashioniq import FashionIQDataset as DatasetClass

        trainset = DatasetClass(
            path=cfg.DATASET.PATH,
            split='train',
            transform=train_transform)
        valset = DatasetClass(
            path=cfg.DATASET.PATH,
            split='val',
            transform=transform)
        testset = DatasetClass(
            path=cfg.DATASET.PATH,
            split='test',
            transform=transform)
        dataset_dict = {"train": trainset, "val": valset, "test": testset}
    else:
        import importlib
        datamod = importlib.import_module(f"maaf.datasets.{cfg.DATASET.NAME}")
        DatasetClass = getattr(datamod, datamod.DATASET_CLASS_NAME)
        dataset_dict = {split: DatasetClass(path=cfg.DATASET.PATH, split=split,
                                            transform=transform)
                        for split in ["train", "val", "test"]}

    for name, data in dataset_dict.items():
        if data is not None:
            print(name, 'size', len(data))

    if "test" not in dataset_dict:
        dataset_dict["test"] = None

    return dataset_dict
