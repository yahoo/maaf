# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

import time
import numpy as np
from .actions import eval_retrieval
import torch
import torch.utils.data
from .utils.misc_utils import tqdm  # with dynamic_ncols=True
from collections import defaultdict
from .utils.bn_utils import apply_bn_mode


class Trainer:

    def __init__(self, cfg, logger, dataset_dict, model, optimizer, initial_it):
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.logger = logger
        self.optimizer = optimizer
        self.step = initial_it
        self.epoch = initial_it // len(dataset_dict["train"])
        self.scheduled_lr = False
        if len(cfg.SOLVER.SCHEDULE_RATES) > 0:
            self.scheduled_rates = \
                [float(rate) for rate in cfg.SOLVER.SCHEDULE_RATES]
            if cfg.SOLVER.SCHEDULE_ITERS != "":
                self.schedule_iters = \
                    [int(itr) for itr in cfg.SOLVER.SCHEDULE_ITERS]
                self.scheduled_lr = True
                self.current_lr = cfg.SOLVER.LEARNING_RATE

        self.losses_tracking = defaultdict(list)

        self.dataset_dict = dataset_dict
        self.trainset = dataset_dict["train"]
        self.model = model

    def parse_batch(self, batch):
        images = [dd['image'] for dd in batch]
        if hasattr(self.model, "image_transform"):
            images = [self.model.image_transform(im) for im in images]
        if images[0] is not None:
            images = torch.stack(images).float().to(self.device)
        texts = [dd["text"] for dd in batch]
        labels = [dd["label"] for dd in batch]
        labels = torch.Tensor(labels).long().to(self.device)
        return (images, texts, labels)

    def train_step(self, batch):
        self.model.train()
        apply_bn_mode(self.model, self.cfg.SOLVER.BATCH_NORM_MODE)

        parsed_batch = self.parse_batch(batch)

        if self.step == 0 and parsed_batch[0][0] is not None:
            for ii, im in enumerate(parsed_batch[0]):
                self.logger.add_image(f"image_inputs_{ii}", im, self.step)

        losses = []
        loss_value, metrics = self.model.compute_loss(*parsed_batch)

        loss_name = self.cfg.MODEL.LOSS
        loss_weight = 1.0
        losses += [(loss_name, loss_weight, loss_value)]
        total_loss = sum([
            l_weight * l_value
            for _, l_weight, l_value in losses
        ])
        assert not torch.isnan(total_loss)
        losses += [('total training loss', None, total_loss)]

        for key, val in metrics.items():
            losses += [("train_" + key, None, val)]

        # track losses
        for l_name, l_weight, l_value in losses:
            self.losses_tracking[loss_name].append(loss_value.item())
        for key, val in metrics.items():
            self.losses_tracking[key].append(val)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return metrics

    def train(self):
        self.model.to(self.device)

        tic = time.time()
        while self.step < self.cfg.SOLVER.NUM_ITERS:
            cat = "batchwise" if self.cfg.DATASET.SINGLE_CLASS_BATCHES else None
            trainloader = self.trainset.get_loader(
                batch_size=self.cfg.SOLVER.BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=self.cfg.DATA_LOADER.LOADER_NUM_WORKERS,
                category=cat)

            # show/log stats
            print("It {} epoch {} Elapse time {:.4f}".format(
                self.step, self.epoch, time.time() - tic
            ))
            tic = time.time()
            for loss_name in self.losses_tracking:
                the_loss = self.losses_tracking[loss_name]
                avg_loss = np.mean(the_loss[-len(trainloader):])
                print('    ', loss_name, round(avg_loss, 4))
                self.logger.add_scalar(loss_name, avg_loss, self.step)
            self.logger.add_scalar(
                'learning_rate', self.optimizer.param_groups[0]['lr'],
                self.step)

            # test
            evalstep = self.epoch % self.cfg.SOLVER.EVAL_EVERY == 1 or \
                self.cfg.SOLVER.EVAL_EVERY == 1
            if evalstep and self.epoch > 0:
                self.run_eval(eval_on_test=self.cfg.SOLVER.ALWAYS_EVAL_TEST)

            # save checkpoint
            torch.save({
                'it': self.step,
                'model_state_dict': self.model.state_dict(),
            },
                self.logger.file_writer.get_logdir() + '/latest_checkpoint.pth')

            if self.epoch % self.cfg.SOLVER.SAVE_EVERY == 0 and self.epoch > 0:
                torch.save({
                    'it': self.step,
                    'model_state_dict': self.model.state_dict()},
                    self.logger.file_writer.get_logdir()
                        + '/ckpt_epoch{}.pth'.format(self.epoch))

            for batch in tqdm(trainloader, desc='Training for epoch ' + str(self.epoch)):
                self.train_step(batch)
                self.step += 1
                self.update_learning_rate()
            self.epoch += 1

        torch.save({
            'it': self.step,
            'model_state_dict': self.model.state_dict(),
        },
            self.logger.file_writer.get_logdir() + '/latest_checkpoint.pth')
        return self.step

    def simple_test(self, testset, name="val"):
        self.model.eval()
        loader = testset.get_loader(
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.DATA_LOADER.LOADER_NUM_WORKERS)
        metrics = defaultdict(list)
        for batch in tqdm(loader):
            with torch.no_grad():
                loss_value, met_dict = self.model.compute_loss(*self.parse_batch(batch))
                for key, val in met_dict.items():
                    metrics[key].append(val * len(batch))
        metrics = {key: sum(val) / len(testset) for key, val in metrics.items()}
        output = [(f"{name}_{key}", val) for key, val in metrics.items()]
        return output

    def run_eval(self, eval_on_test=False):
        self.model.eval()
        # trainset = self.dataset_dict["train"]
        if eval_on_test:
            testset = self.dataset_dict["test"]
        else:
            testset = self.dataset_dict.get("val", self.dataset_dict["test"])

        tests = []

        tests = self.simple_test(testset, "val")

        try:
            special_subset, name = testset.special_subset()
            special_results = self.simple_test(special_subset, name=name)
            tests += [(metric_name, metric_value)
                      for metric_name, metric_value in special_results]
        except AttributeError:
            pass

        for metric_name, metric_value in tests:
            self.logger.add_scalar(metric_name, metric_value, self.step)
            print(f'    {metric_name}: {metric_value:.4f}')

        return tests

    def update_learning_rate(self):
        if self.scheduled_lr:
            if len(self.schedule_iters) > 0:
                if self.step > self.schedule_iters[0]:
                    lr_factor = self.scheduled_rates[0] / self.current_lr
                    for g in self.optimizer.param_groups:
                        g['lr'] *= lr_factor
                    self.current_lr = self.scheduled_rates[0]

                    del self.schedule_iters[0]
                    del self.scheduled_rates[0]
        else:
            # decay learing rate by old method
            decay = False
            if self.step >= self.cfg.SOLVER.LEARNING_RATE_DECAY_FREQUENCY:
                if self.step == self.cfg.SOLVER.LEARNING_RATE_DECAY_FREQUENCY:
                    decay = True
                elif self.step % self.cfg.SOLVER.LEARNING_RATE_DECAY_FREQUENCY == 0:
                    decay = not self.cfg.SOLVER.LR_DECAY_ONLY_ONCE
            if decay:
                for g in self.optimizer.param_groups:
                    g['lr'] *= self.cfg.SOLVER.LEARNING_RATE_DECAY


class MetricTrainer(Trainer):

    def parse_batch(self, batch):
        source_img = [dd['source_image'] for dd in batch]
        target_img = [dd['target_image'] for dd in batch]
        if source_img[0] is not None:
            if hasattr(self.model, "image_transform"):
                source_img = [self.model.image_transform(im) for im in source_img]
            source_img = torch.stack(source_img).to(self.model.device).float()
        if target_img[0] is not None:
            if hasattr(self.model, "image_transform"):
                target_img = [self.model.image_transform(im) for im in target_img]
            target_img = torch.stack(target_img).to(self.model.device).float()

        source_text = [dd["source_text"] for dd in batch]
        target_text = [dd["target_text"] for dd in batch]

        if "judgment" in batch[0]:
            judgments = [self.trainset.parse_judgment(
                dd["judgment"], loss=self.cfg.MODEL.LOSS) for dd in batch]
        else:
            judgments = [None for dd in batch]

        return (source_img, source_text, target_img, target_text, judgments)

    def run_eval(self, eval_on_test=False):
        self.model.eval()
        if eval_on_test:
            testset = self.dataset_dict["test"]
        else:
            testset = self.dataset_dict.get("val", self.dataset_dict["test"])

        try:
            with torch.no_grad():
                test_results = testset.evaluate(self.model, self.cfg)
        except AttributeError:
            with torch.no_grad():
                test_results = self.metric_eval(testset, eval_on_test=eval_on_test)

        for metric_name, metric_value in test_results:
            self.logger.add_scalar(metric_name, metric_value, self.step)
            print('    ', metric_name, round(metric_value, 4))

        return test_results


    def metric_eval(self, testset, eval_on_test=False):
        if self.cfg.DATASET.NAME in ["fashioniq", "fashion200k"]:
            categ = self.cfg.DATASET.NAME == "fashioniq"
            test_results = eval_retrieval.test(
                self.cfg, self.model, testset, filter_categories=categ)
        else:
                print(f"No special validation for {self.cfg.DATASET.NAME};"
                      "computing average validation loss")
                test_results = self.simple_test(testset, "val")

        return test_results
