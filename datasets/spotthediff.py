# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

import random
import numpy as np
import os
import json
import torch.utils.data
import PIL
from .base_dataset import BaseDataset


class SpotTheDiff(BaseDataset):

    def __init__(self, path, split='train', transform=None,
                 batch_size=None, normalize=False):

        super().__init__()
        self.normalize = normalize
        self.batch_size = batch_size

        self.split = split
        self.transform = transform
        self.img_path = path + '/resized_images/'

        annotations_file = os.path.join(path, "annotations",
                                        "{}.json".format(split))

        with open(annotations_file, "r") as fh:
            annotations = json.load(fh)

        # ordering given by json, don't sort
        all_images = []
        queries = []
        for datum in annotations:
            spot_id = datum["img_id"]
            full_path = self.img_path + "{}.png".format(spot_id)

            first_id = len(all_images)
            entry = [{
                'spot_id': spot_id,
                'file_path': full_path,
                'captions': [first_id], # not really a caption!
                'image_id': first_id,
            }]
            all_images += entry

            second_id = len(all_images)
            second_path = self.img_path + "{}_2.png".format(spot_id)
            entry = [{
                'spot_id': spot_id,
                'file_path': second_path,
                'captions': [second_id], # not really a caption!
                'image_id': second_id,
            }]
            all_images += entry

            sentences = datum["sentences"]
            query = {
                "source_id": first_id,
                "target_id": second_id,
                "captions": [sent for sent in sentences if sent != ""]
            }
            # if there are only empty-string captions, this is a useless query
            if len(query["captions"]) > 0:
                queries.append(query)

        self.imgs = all_images
        self.queries = queries

        if split in ["val", "test"]:
            self.test_queries = []
            for query in queries:
                self.test_queries += [{
                      'source_img_id': query['source_id'],
                      'target_img_id': query['target_id'],
                      'target_caption': query['target_id'],
                      'mod': {'str': query['captions'][ii]}
                  } for ii in range(len(query['captions']))]

    def get_all_texts(self):
        texts = []
        for query in self.queries:
            texts += query['captions']
        return texts

    def __len__(self):
        return len(self.imgs)

    def generate_random_query_target(self):
        query = random.choice(self.queries)
        mod_str = random.choice(query['captions'])

        return {
          'source_img_id': query['source_id'],
          'source_img_data': self.get_img(query['source_id']),
          'target_img_id': query['target_id'],
          'target_caption': query['target_id'],
          'target_img_data': self.get_img(query['target_id']),
          'mod': {'str': mod_str}
        }

    def get_img(self, idx, raw_img=False):
        """Retrieve image by global index."""
        img_path = self.imgs[idx]['file_path']
        try:
            with open(img_path, 'rb') as f:
                img = PIL.Image.open(f)
                img = img.convert('RGB')
        except EnvironmentError as ee:
            print("WARNING: EnvironmentError, defaulting to image 0", ee)
            img = self.get_img(0, raw_img=True)
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img
