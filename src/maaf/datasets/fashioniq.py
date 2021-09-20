# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

import random
import os
import json
from torch.utils.data import Dataset, DataLoader, Subset
import PIL
import numpy as np

CATEGORIES = ["dress", "shirt", "toptee"]


class FashionIQGallery(Dataset):

    def __init__(self, gallery, img_by_cat=None, transform=None):
        super().__init__()
        self.gallery = gallery
        self.transform = transform
        if img_by_cat is not None:
            self.gallery_by_cat = \
                {cat: FashionIQGallery(img_by_cat[cat], transform=transform)
                 for cat in img_by_cat}

    def __getitem__(self, ind):
        example = self.gallery[ind]
        image = self.get_img(ind)
        example["target_image"] = image
        example["target_text"] = None

        return example

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
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.gallery)


class FashionIQTrainLoader:
    """
    Each batch is drawn from one of the loaders at random.
    Iteration stops when a loader is queried and raises StopIteration,
    even if the other loaders have more data left.
    """

    def __init__(self, loaders, random_seed=5222020):
        self.loaders = {key: iter(loaders[key]) for key in loaders}
        self.rng = np.random.RandomState(random_seed)

    def __iter__(self):
        return self

    def __next__(self):
        category = self.rng.choice(list(self.loaders.keys()))
        return next(self.loaders[category])


    def __len__(self):
        return sum([len(ld) for ld in self.loaders.values()])


class FashionIQDataset(Dataset):

    def __init__(self, path, split='train', transform=None, normalize=False):
        super().__init__()
        self.categories = CATEGORIES
        self.normalize = normalize

        self.split = split
        self.transform = transform
        self.img_path = path + '/'

        failures = []

        data = {
            'image_splits': {},
            'captions': {}
        }

        def wanted_captions(filename):
            """Select normalized/original caption files."""
            if normalize and "cap." in filename:
                return "normcap." in filename
            else:
                return "normcap." not in filename
        for data_type in data:
            for datafile in os.listdir(path + '/' + data_type):
                if split in datafile and wanted_captions(datafile):
                    data[data_type][datafile] = \
                        json.load(open(path + '/' + data_type + '/' + datafile))

        split_labels = sorted(list(data["image_splits"].keys()))

        global_imgs = []
        img_by_cat = {cat: [] for cat in CATEGORIES}
        self.asin2id = {}
        for splabel in split_labels:
            for asin in data['image_splits'][splabel]:
                # if asin in failures:
                #     continue
                category = splabel.split(".")[1]
                file_path = path + '/img/' + category + '/' + asin
                if os.path.exists(file_path) or split == "test":
                    global_id = len(global_imgs)
                    category_id = len(img_by_cat[category])
                    entry = [{
                        'asin': asin,
                        'file_path': file_path,
                        'captions': [global_id],
                        "image_id": global_id,
                        "category": {category: category_id}
                    }]
                    if asin in self.asin2id:
                        # handle duplicates
                        oldglobal = self.asin2id[asin]
                        subentry = global_imgs[oldglobal]
                        assert category not in subentry["category"], \
                            "{} duplicated in {}".format(asin, category)

                        # update entry to include additional category and id
                        subentry["category"][category] = category_id
                        img_by_cat[category] += [subentry]
                    else:
                        # just add the entry
                        global_imgs += entry
                        img_by_cat[category] += entry
                        self.asin2id[asin] = global_id
                else:
                    failures.append(asin)

        print(len(failures), " files not found in ", split)
        assert len(global_imgs) > 0, "no data found"

        queries = []
        captions = sorted(list(data["captions"].keys()))
        for cap in captions:
            for query in data['captions'][cap]:
                if split != "test" and (query['candidate'] in failures
                                        or query.get('target') in failures):
                    continue
                query['source_id'] = self.asin2id[query['candidate']]
                query["category"] = cap.split(".")[1]
                if split != "test":
                    query['target_id'] = self.asin2id[query['target']]
                    tarcat = global_imgs[query['target_id']]["category"]
                    if query["category"] not in tarcat:
                        print("WARNING: a {} found with a target in {}".format(
                            query["category"], tarcat
                        ))
                soucat = global_imgs[query['source_id']]["category"]
                assert query["category"] in soucat

                queries += [query]

        self.img_by_cat = img_by_cat
        self.data = queries
        self.gallery = FashionIQGallery(global_imgs, img_by_cat,
                                        transform=transform)
        self.data_by_category = \
            {cat: Subset(self, [ii for ii in range(len(self))
                                if self.data[ii]["category"] == cat])
             for cat in self.categories}


        self.id2asin = {val: key for key, val in self.asin2id.items()}

    def get_all_texts(self):
        texts = [' inadditiontothat ']
        for query in self.data:
            texts += query['captions']
        return texts

    def __len__(self):
        return len(self.data)

    def get_loader(self,
                   batch_size,
                   shuffle=False,
                   drop_last=False,
                   num_workers=0,
                   category=None):
        if category == "batchwise":
            loaders = {
                cat: self.get_loader(
                    batch_size, shuffle=shuffle,
                    drop_last=drop_last, num_workers=num_workers,
                    category=cat)
                for cat in self.categories}
            return FashionIQTrainLoader(loaders)
        elif category is None:
            ds = self
        else:
            ds = self.data_by_category[category]
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)

    def get_gallery_loader(self, batch_size, num_workers=0, category=None):
        if category is None:
            gallery = self.gallery
        else:
            gallery = self.gallery.gallery_by_cat[category]
        return DataLoader(
            gallery,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=lambda i: i)

    def __getitem__(self, idx):
        example = self.data[idx]

        if self.split == "train":
            mod_str = random.choice([
                example['captions'][0] + ' inadditiontothat ' + example['captions'][1],
                example['captions'][1] + ' inadditiontothat ' + example['captions'][0],
            ])
        else:
            mod_str = example['captions'][0] + ' inadditiontothat ' + example['captions'][1]

        if len(mod_str) < 2:
            # can happen during training if a caption is tiny
            mod_str = example['captions'][0] + ' inadditiontothat ' + example['captions'][1]

        item = {key: val for key, val in example.items()}

        item["source_image"] = self.gallery.get_img(example['source_id'])
        item["source_text"] = mod_str

        if self.split != "test":
            item["target_image"] = self.gallery.get_img(example['target_id'])
            item["target_text"] = None

        item["judgment"] = 1

        return item

    def get_test_queries(self, category=None):
        if category is not None:
            return [que for que in self.data if que["category"] == category]
        return self.data

    def parse_judgment(self, judgment, loss=None):
        return judgment
