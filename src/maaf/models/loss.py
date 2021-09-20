# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


import sys
import torch
import torch.nn as nn
import numpy as np


def build_loss(cfg):
    loss_name = cfg.MODEL.LOSS

    learning_type = "metric"
    if loss_name == "softmax_cross_entropy":
        class_weights = torch.Tensor(cfg.DATASET.CLASS_WEIGHTS)
        loss_obj = torch.nn.CrossEntropyLoss(weight=class_weights)
        learning_type = "classification"
    elif loss_name == "multilabel_soft_margin":
        class_weights = torch.Tensor(cfg.DATASET.CLASS_WEIGHTS)
        loss_obj = torch.nn.MultiLabelSoftMarginLoss(weight=class_weights)
        learning_type = "classification"
    elif loss_name == "mse":
        loss_obj = torch.nn.MSELoss()
        learning_type = "regression"
    elif loss_name == "soft_triplet":
        loss_obj = SoftTripletLoss()
    elif loss_name == "batch_based_classification":
        loss_obj = BatchSoftmaxLoss()
    elif loss_name == "double_softmax":
        loss_obj = DoubleBatchSoftmaxLoss()
    elif loss_name == "logistic":
        loss_obj = Logistic()
    elif loss_name == "logistic_cumulative":
        loss_obj = LogisticCumulativeLink(cfg.DATASET.NUM_CLASSES)
    else:
        print('Invalid loss function', loss_name)
        sys.exit()

    return loss_obj, learning_type


class MetricLossBase:

    def __call__(self, sources, targets, judgments):
        raise NotImplementedError


class SoftTripletLoss(MetricLossBase):

    def __call__(self, sources, targets, judgments=None):
        triplets = []
        labels = list(range(sources.shape[0])) + list(range(targets.shape[0]))
        for i in range(len(labels)):
            triplets_i = []
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    for k in range(len(labels)):
                        if labels[i] != labels[k]:
                            triplets_i.append([i, j, k])
            np.random.shuffle(triplets_i)
            triplets += triplets_i[:3]  # WHY?
        assert (triplets and len(triplets) < 2000)
        return self.soft_triplet_loss(torch.cat([sources, targets]), triplets)


class BatchSoftmaxLoss(MetricLossBase):
    """
    Implements batch-wise softmax cross-entropy loss.
    Source-target pairs are assumed to be positive matches.
    """

    def __init__(self, softmax_margin=0, drop_worst_rate=0):
        self.softmax_margin = softmax_margin
        self.drop_worst_rate = drop_worst_rate

    def __call__(self, sources, targets, judgments=None):
        dots = torch.mm(sources, targets.transpose(0, 1))
        if self.softmax_margin > 0:
            dots = dots - (torch.Tensor(np.eye(dots.shape[0])).to(sources.device)
                           * self.softmax_margin)
        labels = torch.tensor(range(dots.shape[0])).long().to(dots.device)
        losses = nn.functional.cross_entropy(dots, labels, reduction='none')
        if self.drop_worst_rate > 0:
            losses, idx = torch.topk(
                losses, k=int(losses.shape[0] * (1 - self.drop_worst_rate)),
                largest=False)
        final_loss = losses.mean()

        return final_loss


class DoubleBatchSoftmaxLoss(MetricLossBase):

    def __init__(self):
        self.loss_first = nn.CrossEntropyLoss()
        self.loss_second = nn.CrossEntropyLoss()

    def __call__(self, sources, targets, judgments=None):
        dots = torch.mm(sources, targets.transpose(0, 1))

        labels = torch.tensor(range(dots.shape[0])).long().to(dots.device)
        losses_a = self.loss_first(dots, labels)
        losses_b = self.loss_second(dots.transpose(0, 1), labels)

        final_loss = losses_a + losses_b
        return final_loss / 2


class Logistic(MetricLossBase, nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.criterion = nn.SoftMarginLoss()
        # self.criterion = nn.BCELoss()  for this need sigmoid in __call__

    def __call__(self, sources, targets, judgments=None):
        dots = torch.sum(sources * targets, dim=1)
        judgments = torch.Tensor(judgments)
        return self.criterion(dots, judgments)


class LogisticCumulativeLink(MetricLossBase, nn.Module):
    """Adapted from https://github.com/EthanRosenthal/spacecutter/...
    blob/master/spacecutter/losses.py"""
    def __init__(self, num_classes):
        MetricLossBase.__init__(self)
        nn.Module.__init__(self)

        num_thresh = num_classes - 1
        self.thresholds = torch.arange(num_thresh).float() - num_thresh / 2
        self.thresholds = nn.Parameter(self.thresholds)


    def __call__(self, sources, targets, judgments=None):
        dots = torch.sum(sources * targets, dim=1).unsqueeze(-1)
        sigmoids = torch.sigmoid(self.thresholds - dots)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )  # batch, num_classes

        judgments = torch.Tensor(judgments).long()

        likelihoods = link_mat[judgments]
        eps = 1e-15
        likelihoods = torch.clamp(likelihoods, eps, 1 - eps)
        neg_log_likelihood = -torch.log(likelihoods)

        loss = torch.mean(neg_log_likelihood)
        return loss
