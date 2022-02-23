# Copyright 2022 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

from torch.utils.data import Dataset, DataLoader
import random
import PIL
from os import listdir

DATASET_CLASS_NAME = "MITStates"

TEST_NOUNS = [
    u'armor', u'bracelet', u'bush', u'camera', u'candy', u'castle',
    u'ceramic', u'cheese', u'clock', u'clothes', u'coffee', u'fan', u'fig',
    u'fish', u'foam', u'forest', u'fruit', u'furniture', u'garden', u'gate',
    u'glass', u'horse', u'island', u'laptop', u'lead', u'lightning',
    u'mirror', u'orange', u'paint', u'persimmon', u'plastic', u'plate',
    u'potato', u'road', u'rubber', u'sand', u'shell', u'sky', u'smoke',
    u'steel', u'stream', u'table', u'tea', u'tomato', u'vacuum', u'wax',
    u'wheel', u'window', u'wool'
]


class MITStatesGallery(Dataset):

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
        img_path = self.gallery[idx]['file_path']
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img


class MITStates(Dataset):

    def __init__(self, path, split="train", transform=None):
        super().__init__()

        self.path = path
        self.transform = transform
        self.split = split

        self.data = []

        for f in listdir(path + '/images'):
            if ' ' not in f:
                continue
            adj, noun = f.split()
            if adj == 'adj':
                continue
            if split == 'train' and noun in TEST_NOUNS:
                continue
            if split == 'test' and noun not in TEST_NOUNS:
                continue

            for file_path in listdir(path + '/images/' + f):
                assert (file_path.endswith('jpg'))
                this_index = len(self.data)
                self.data += [{
                    'file_path': path + '/images/' + f + '/' + file_path,
                    'captions': [f],
                    'adj': adj,
                    'noun': noun,
                    "image_id": this_index
                }]

        self.gallery = MITStatesGallery(self.data, self.transform)
        self.caption_index_init_()
        if split == 'test':
            self.generate_test_queries_()
        else:
            self.test_queries = None

        self.saved_item = None

    def get_all_texts(self):
        texts = []
        for img in self.data:
            texts += img['captions']
        return texts

    def __getitem__(self, idx):
        if self.split == "test":
            query = self.test_queries[idx]
            return {
                'source_image': self.get_img(query["source_img_id"]),
                'source_text': query["mod"]["str"]
            }
        else:
            idx, target_idx = self.get_random_pair(idx)
            mod_str = self.data[target_idx]['adj']

            return {
                'source_img_id': idx,
                'source_image': self.get_img(idx),
                'source_cap': self.data[idx]['captions'][0],
                'target_img_id': target_idx,
                'target_image': self.get_img(target_idx),
                'target_cap': self.data[target_idx]['captions'][0],
                'source_text': mod_str,
                "target_text": None
            }

    def get_random_pair(self, idx):
        """
        Eric doesn't know why this pairing thing was in the TIRG code
        but we're keeping it for consistency.
        """
        if self.saved_item is None:
            while True:
                idx, target_idx1 = self.caption_index_sample_(idx)
                idx, target_idx2 = self.caption_index_sample_(idx)
                if self.data[target_idx1]['adj'] != self.data[target_idx2]['adj']:
                    break
            idx, target_idx = [idx, target_idx1]
            self.saved_item = [idx, target_idx2]
        else:
            idx, target_idx = self.saved_item
            self.saved_item = None
        return idx, target_idx

    def caption_index_init_(self):
        self.caption2imgids = {}
        self.noun2adjs = {}
        for i, img in enumerate(self.data):
            cap = img['captions'][0]
            adj = img['adj']
            noun = img['noun']
            if cap not in self.caption2imgids.keys():
                self.caption2imgids[cap] = []
            if noun not in self.noun2adjs.keys():
                self.noun2adjs[noun] = []
            self.caption2imgids[cap].append(i)
            if adj not in self.noun2adjs[noun]:
                self.noun2adjs[noun].append(adj)
        for noun, adjs in self.noun2adjs.items():
            assert len(adjs) >= 2

    def caption_index_sample_(self, idx):
        noun = self.data[idx]['noun']
        # adj = self.data[idx]['adj']
        target_adj = random.choice(self.noun2adjs[noun])
        target_caption = target_adj + ' ' + noun
        target_idx = random.choice(self.caption2imgids[target_caption])
        return idx, target_idx

    def generate_test_queries_(self):
        self.test_queries = []
        for idx, img in enumerate(self.data):
            adj = img['adj']
            noun = img['noun']
            for target_adj in self.noun2adjs[noun]:
                if target_adj != adj:
                    mod_str = target_adj
                    self.test_queries += [{
                        'source_img_id': idx,
                        'source_caption': adj + ' ' + noun,
                        'target_caption': target_adj + ' ' + noun,
                        'mod': {
                            'str': mod_str
                        }
                    }]
        print(len(self.test_queries), 'test queries')

    def __len__(self):
        if self.split == "test":
            return len(self.test_queries)
        return len(self.data)

    def get_img(self, idx, raw_img=False):
        img_path = self.data[idx]['file_path']
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
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
