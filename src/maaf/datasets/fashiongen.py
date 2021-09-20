# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import h5py
from ..utils.misc_utils import tqdm
import torch
import numpy as np
from sklearn.metrics import average_precision_score

DATASET_CLASS_NAME = "FashionGen"


class FashionGen(Dataset):

    def __init__(self,
                 path="/home/default/ephemeral_drive/Data/fashiongen/",
                 split="train",
                 transform=None,
                 default_image_size=224):
        super().__init__()

        self.data_path = path
        self.default_image_size = default_image_size

        self.transform = transform

        if split == "val":
            split = "validation"
        if split == "test":
            self.data = {"input_description": []}
            return
        self.data = h5py.File(os.path.join(
            self.data_path, f"fashiongen_256_256_{split}.h5"), mode="r")
        self.split = split

    def __len__(self):
        return len(self.data["input_description"])

    def __getitem__(self, idx):
        item = {}
        item["image"] = Image.fromarray(self.data["input_image"][idx])
        if self.transform is not None:
            item["image"] = self.transform(item["image"])
        item["text"] = self.data["input_description"][idx][0].decode("latin-1")

        item["target_text"] = item["text"]
        item["target_image"] = None
        item["source_text"] = None
        item["source_image"] = item["image"]

        return item

    def get_all_texts(self):
        return [cap[0].decode("utf-8") for cap in self.data["input_description"]]

    def get_loader(self,
                   batch_size,
                   shuffle=False,
                   drop_last=False,
                   num_workers=0,
                   category=None):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)

    def evaluate(self, model, cfg=None):
        model.eval()

        text_embed = {}
        image_embed = []
        product_ids = []
        for idx in tqdm(range(len(self))):
            img = Image.fromarray(self.data["input_image"][idx])
            if self.transform is not None:
                img = self.transform(img)
            img = torch.stack([img]).float().to(model.device)
            text = self.data["input_description"][idx][0].decode("latin-1")
            pid = self.data["input_productID"][idx][0]
            img_emb = model(img, [None]).cpu().numpy()
            image_embed.append(img_emb / np.linalg.norm(img_emb))
            product_ids.append(pid)
            if pid not in text_embed:
                text_emb = model([None], [text]).cpu().numpy()
                text_embed[pid] = text_emb / np.linalg.norm(text_emb)

        text_emb_array = np.concatenate(list(text_embed.values()))
        image_emb_array = np.concatenate(image_embed)
        text_pids = list(text_embed.keys())

        sims = image_emb_array @ text_emb_array.T

        correct = np.zeros(sims.shape, dtype=bool)
        for ii, pid in enumerate(product_ids):
            for jj, text_pid in enumerate(text_pids):
                correct[ii, jj] = pid == text_pid

        img_ap_scores = []
        for ii in range(len(correct)):
            img_ap_scores.append(average_precision_score(correct[ii], sims[ii]))
        img_map = np.mean(img_ap_scores)
        txt_ap_scores = []
        for jj in range(correct.shape[1]):
            txt_ap_scores.append(average_precision_score(correct[jj], sims[jj]))
        txt_map = np.mean(txt_ap_scores)
        maps = [("image_to_text_mAP", img_map), ("text_to_image_mAP", txt_map)]

        sorter_per_img = np.argsort(sims, axis=1)[:, ::-1]
        sorter_per_text = np.argsort(sims, axis=0)[::-1]

        correct_sorted_per_img = np.take_along_axis(
            correct, sorter_per_img, axis=1)
        correct_sorted_per_text = np.take_along_axis(
            correct, sorter_per_text, axis=0)

        indic_per_img = np.cumsum(correct_sorted_per_img, axis=1) > 0
        indic_per_txt = np.cumsum(correct_sorted_per_text, axis=0) > 0
        ks = [1, 5, 10]
        recalls_img = [(f"image_to_text_recall{kk}",
                        np.mean(indic_per_img[:, kk])) for kk in ks]
        recalls_txt = [(f"text_to_image_recall{kk}",
                        np.mean(indic_per_txt[kk])) for kk in ks]

        recall_sum = sum([thing[1] for thing in recalls_img]) + \
            sum([thing[1] for thing in recalls_txt])
        recall_sum = [("recall_sum", recall_sum)]

        return maps + recalls_img + recalls_txt + recall_sum
