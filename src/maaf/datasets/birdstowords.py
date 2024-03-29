# Copyright 2022 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

import random
import os
from torch.utils.data import Dataset, DataLoader
import PIL


DATASET_CLASS_NAME = "BirdsToWords"


class BirdsToWordsGallery(Dataset):

    def __init__(self, gallery, transform=None):
        super().__init__()
        self.gallery = gallery
        self.transform = transform

    def __getitem__(self, ind):
        example = self.gallery[ind]
        image = self.get_img(ind)
        example["target_image"] = image
        example["target_text"] = None

        return example

    def __len__(self):
        return len(self.gallery)

    def get_img(self, idx, raw_img=False):
        """Retrieve image by global index."""
        img_path = self.gallery[idx]['file_path']
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


class BirdsToWords(Dataset):

    def __init__(self, path, split='train', transform=None,
                 batch_size=None, normalize=False):

        super().__init__()
        self.normalize = normalize
        self.batch_size = batch_size

        self.split = split
        self.transform = transform
        self.img_path = path + '/images/'

        failures = set()

        tsv = os.path.join(path, "birds-to-words-v1.0.tsv")
        raw_data = []

        with open(tsv, "r") as fh:
            for line in fh:
                entry = line.strip().split("\t")
                if entry[-3] == self.split:
                    raw_data.append(entry)

        #######
        # get image data
        image_fpaths = set()
        for entry in raw_data:
            first_image_fpath = entry[1].split("/")[-1]
            second_image_fpath = entry[5].split("/")[-1]
            image_fpaths.add(first_image_fpath)
            image_fpaths.add(second_image_fpath)

        image_fpaths = sorted(list(image_fpaths))
        img_fpath_to_id = {}
        all_images = []
        for fpath in image_fpaths:
            full_path = self.img_path + fpath
            if os.path.exists(full_path):
                image_id = len(all_images)
                entry = [{
                    'photo_number': fpath.split("?")[-1],
                    'file_path': full_path,
                    'captions': [image_id], # not really a caption!
                    'image_id': image_id,
                }]
                all_images += entry
                img_fpath_to_id[fpath] = image_id
            else:
                failures.add(fpath)

        print(len(failures), " files not found in ", split)
        assert len(all_images) > 0, "no data found"


        #######
        # get pairs and descriptions
        queries = {}
        for entry in raw_data:
            first_image_fpath = entry[1].split("/")[-1]
            second_image_fpath = entry[5].split("/")[-1]

            if first_image_fpath in failures or second_image_fpath in failures:
                continue

            query_dict_key = first_image_fpath + "_" + second_image_fpath
            descrip = entry[-1]
            assert isinstance(descrip, str)
            if query_dict_key in queries:
                query = queries[query_dict_key]
                query["captions"] += [descrip]
            else:
                query = {}
                query["source_id"] = img_fpath_to_id[first_image_fpath]
                query["target_id"] = img_fpath_to_id[second_image_fpath]
                query["captions"] = [descrip]

            queries[query_dict_key] = query

            # during training also use triplet with images swapped
            if self.split == "train":
                word_by_word = entry[-1].split(" ")
                for ii in range(len(word_by_word)):
                    if word_by_word[ii] == "animal1":
                        word_by_word[ii] = "animal2"
                    elif word_by_word[ii] == "animal2":
                        word_by_word[ii] = "animal1"
                flipped_descrip = " ".join(word_by_word)
                assert len(flipped_descrip) == len(entry[-1])

                flipped_key = second_image_fpath + "_" + first_image_fpath
                if flipped_key in queries:
                    flipped_query = queries[flipped_key]
                    flipped_query["captions"] += [flipped_descrip]
                else:
                    flipped_query = {}
                    flipped_query["source_id"] = query["target_id"]
                    flipped_query["target_id"] = query["source_id"]
                    flipped_query["captions"] = [flipped_descrip]

                queries[flipped_key] = flipped_query

        query_keys = sorted(list(queries.keys()))
        queries = [queries[key] for key in query_keys]

        self.data = all_images
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

        self.gallery = BirdsToWordsGallery(self.data, transform=self.transform)

    def get_all_texts(self):
        texts = []
        for query in self.queries:
            texts += query['captions']
        return texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.split in ["val", "test"]:
            query = self.test_queries[idx]
            return {
                'source_image': self.get_img(query["source_img_id"]),
                'source_text': query["mod"]["str"]
            }
        return self.generate_random_query_target()

    def generate_random_query_target(self):
        query = random.choice(self.queries)
        mod_str = random.choice(query['captions'])

        return {
            'source_img_id': query['source_id'],
            'source_image': self.get_img(query['source_id']),
            'target_img_id': query['target_id'],
            'target_caption': query['target_id'],
            'target_image': self.get_img(query['target_id']),
            'source_text': {'str': mod_str},
            "target_text": None
        }

    def get_img(self, idx, raw_img=False):
        """Retrieve image by global index."""
        img_path = self.data[idx]['file_path']
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

    def get_gallery_loader(self, batch_size, num_workers=0):
        return DataLoader(
            self.gallery,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=lambda i: i)
