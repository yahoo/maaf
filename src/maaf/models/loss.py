# Copyright 2022 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
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
    elif loss_name == "logratio":
        loss_obj = LogRatioLoss()
    elif loss_name == "softlabel_softmax":
        loss_obj = SoftLabelSoftmaxLoss()
    else:
        print('Invalid loss function', loss_name)
        sys.exit()

    return loss_obj, learning_type


class MetricLossBase:

    def __call__(self, sources, targets, labels):
        raise NotImplementedError


class SoftTripletLoss(MetricLossBase):

    def __call__(self, sources, targets, labels=None):
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

    def __call__(self, sources, targets, labels=None):
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

    def __call__(self, sources, targets, labels=None):
        dots = torch.mm(sources, targets.transpose(0, 1))

        labels = torch.tensor(range(dots.shape[0])).long().to(dots.device)
        losses_a = self.loss_first(dots, labels)
        losses_b = self.loss_second(dots.transpose(0, 1), labels)

        final_loss = losses_a + losses_b
        return final_loss / 2


class SoftLabelSoftmaxLoss(MetricLossBase):

    def __init__(self):
        self.celoss = nn.CrossEntropyLoss()

    def __call__(self, sources, targets, labels):
        dots = torch.mm(sources, targets.transpose(0, 1))
        iou = labels_from_attributes(labels, dots.shape).to(dots.device)
        # NOTE: using CrossEntropyLoss with probabilities requires
        # a recent PyTorch version. Installing this naively may cause problems
        # e.g. a protobuf conflict...but downgrading protobuf may solve this
        return self.celoss(dots, iou)


def intersection_over_union(first, other):
    first_set = set(first)
    other_set = set(other)
    union = len(first_set.union(other_set))
    intersection = len(first_set.intersection(other_set))
    return intersection / union


def labels_from_attributes(labels, shape):
    """Compute IoU labels given attribute lists"""
    iou = torch.zeros(shape)
    for ii in range(len(labels)):
        source_att = labels[ii][0]
        for jj in range(len(labels)):
            target_att = labels[jj][1]
            iou[ii, jj] = intersection_over_union(source_att, target_att)
    return iou


class LogRatioLoss(MetricLossBase):
    """
    Adapted from
    Kim et al. 'Deep metric learning beyond binary supervision' CVPR 2019.
    See
    https://github.com/tjddus9597/Beyond-Binary-Supervision-CVPR19/blob/master/code/LogRatioLoss.py
    """
    epsilon = 1e-6

    def __init__(self):
        pass

    def __call__(self, sources, targets, labels):
        # get all pairwise distances by broadcasting
        distances = torch.linalg.vector_norm(
            sources[:, None, :] - targets[None, :, :], dim=2)
        log_dist = torch.log(distances + self.epsilon)

        iou = labels_from_attributes(labels, log_dist.shape).to(log_dist.device)
        log_iou = torch.log(iou + self.epsilon)

        # get a loss term for each triple (a, i, j) for i and j in target
        # note that the i=j terms are 0
        loss_terms = (log_dist[:, :, None] - log_dist[:, None, :]) - \
                     (log_iou[:, :, None] - log_iou[:, None, :])
        loss_terms = loss_terms * loss_terms

        loss = torch.mean(loss_terms)

        return loss


class Logistic(MetricLossBase, nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.criterion = nn.SoftMarginLoss()
        # self.criterion = nn.BCELoss()  for this need sigmoid in __call__

    def __call__(self, sources, targets, labels=None):
        dots = torch.sum(sources * targets, dim=1)
        labels = torch.Tensor(labels)
        return self.criterion(dots, labels)


class LogisticCumulativeLink(MetricLossBase, nn.Module):
    """Adapted from https://github.com/EthanRosenthal/spacecutter/...
    blob/master/spacecutter/losses.py"""
    def __init__(self, num_classes):
        MetricLossBase.__init__(self)
        nn.Module.__init__(self)

        num_thresh = num_classes - 1
        self.thresholds = torch.arange(num_thresh).float() - num_thresh / 2
        self.thresholds = nn.Parameter(self.thresholds)


    def __call__(self, sources, targets, labels=None):
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

        labels = torch.Tensor(labels).long()

        likelihoods = link_mat[labels]
        eps = 1e-15
        likelihoods = torch.clamp(likelihoods, eps, 1 - eps)
        neg_log_likelihood = -torch.log(likelihoods)

        loss = torch.mean(neg_log_likelihood)
        return loss
