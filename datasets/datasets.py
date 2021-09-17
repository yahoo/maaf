# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

import torch
import torch.utils.data
import torchvision
import sys


def load_dataset(opt):
    """Loads the input datasets."""
    print('Reading dataset ', opt.dataset)
    normalizer = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalizer])
    if opt.dataset == 'css3d':
        from .css3d import CSSDataset
        trainset = CSSDataset(
            path=opt.dataset_path,
            split='train',
            transform=transform)
        testset = CSSDataset(
            path=opt.dataset_path,
            split='test',
            transform=transform)
        dataset_dict = {"train": trainset, "test": testset}
    elif opt.dataset == 'fashion200k':
        from .fashion200k import Fashion200k
        trainset = Fashion200k(
            path=opt.dataset_path,
            split='train',
            transform=transform
            )
        testset = Fashion200k(
            path=opt.dataset_path,
            split='test',
            transform=transform
            )
        dataset_dict = {"train": trainset, "test": testset}
    elif opt.dataset == 'mitstates':
        from .mitstates import MITStates
        trainset = MITStates(
            path=opt.dataset_path,
            split='train',
            transform=transform
            )
        testset = MITStates(
            path=opt.dataset_path,
            split='test',
            transform=transform
            )
        dataset_dict = {"train": trainset, "test": testset}
    elif opt.dataset == 'fashioniq':
        from .fashioniq import FashionIQ
        trainset = FashionIQ(
            path=opt.dataset_path,
            split='joint' if opt.train_on_validation_set else 'train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224, scale=(0.8, 1.0),
                                                         ratio=(0.75, 1.3)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda xx: xx + 0.01*torch.randn(xx.shape)),
                normalizer
            ]),
            batch_size=opt.batch_size)
        valset = FashionIQ(
            path=opt.dataset_path,
            split='val',
            transform=transform,
            batch_size=opt.batch_size)
        testset = FashionIQ(
            path=opt.dataset_path,
            split='test',
            transform=transform,
            batch_size=opt.batch_size)
        dataset_dict = {"train": trainset, "val": valset, "test": testset}
    elif opt.dataset == 'birds':
        from .birdstowords import BirdsToWords
        trainset = BirdsToWords(
            path=opt.dataset_path,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224, scale=(0.8, 1.0),
                                                         ratio=(0.75, 1.3)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda xx: xx + 0.01*torch.randn(xx.shape)),
                normalizer
            ]),
            batch_size=opt.batch_size)
        valset = BirdsToWords(
            path=opt.dataset_path,
            split='val',
            transform=transform,
            batch_size=opt.batch_size)
        testset = BirdsToWords(
            path=opt.dataset_path,
            split='test',
            transform=transform,
            batch_size=opt.batch_size)
        dataset_dict = {"train": trainset, "val": valset, "test": testset}
    elif opt.dataset == 'spotthediff':
        from .spotthediff import SpotTheDiff
        trainset = SpotTheDiff(
            path=opt.dataset_path,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224, scale=(0.8, 1.0),
                                                         ratio=(0.75, 1.3)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda xx: xx + 0.01*torch.randn(xx.shape)),
                normalizer
            ]),
            batch_size=opt.batch_size)
        valset = SpotTheDiff(
            path=opt.dataset_path,
            split='val',
            transform=transform,
            batch_size=opt.batch_size)
        testset = SpotTheDiff(
            path=opt.dataset_path,
            split='test',
            transform=transform,
            batch_size=opt.batch_size)
        dataset_dict = {"train": trainset, "val": valset, "test": testset}
    else:
        print('Invalid dataset', opt.dataset)
        sys.exit()

    for name, data in dataset_dict.items():
        print(name, 'size', len(data))
    return dataset_dict
