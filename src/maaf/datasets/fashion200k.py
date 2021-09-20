# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import PIL

DATASET_CLASS_NAME = "Fashion200k"


class Fashion200kGallery(Dataset):

    def __init__(self, gallery, transform=None, img_path=""):
        super().__init__()
        self.gallery = gallery
        self.transform = transform
        self.img_path = img_path

    def __getitem__(self, ind):
        example = self.gallery[ind]
        image = self.get_img(ind)
        example["target_image"] = image
        example["target_text"] = None

        return example

    def __len__(self):
        return len(self.gallery)

    def get_img(self, idx, raw_img=False):
        img_path = self.img_path + self.gallery[idx]['file_path']
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img


class Fashion200k(Dataset):

    def __init__(self, path, split='train', transform=None):
        raise NotImplementedError("This dataset may not be implemented correctly")
        super().__init__()

        if split != "train":
            split = "test"
        self.split = split
        self.transform = transform
        self.img_path = path + '/'

        # get label files for the split
        label_path = path + '/labels/'
        from os import listdir
        from os.path import isfile
        from os.path import join
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        label_files = [f for f in label_files if split in f]

        # read image info from label files
        self.data = []

        def caption_post_process(s):
            return s.strip().replace(
                '.',
                'dotmark').replace('?', 'questionmark').replace(
                    '&', 'andmark').replace('*', 'starmark')

        for filename in label_files:
            print('read ' + filename)
            with open(label_path + '/' + filename) as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('	')
                this_index = len(self.data)
                img = {
                    'file_path': line[0],
                    'detection_score': line[1],
                    'captions': [caption_post_process(line[2])],
                    'split': split,
                    'modifiable': False,
                    "target_id": this_index
                }
                self.data += [img]
        print('Fashion200k:', len(self.data), 'images')

        # generate query for training or testing
        if split == 'train':
            self.caption_index_init_()
        else:
            self.generate_test_queries_()

        self.gallery = Fashion200kGallery(self.data, self.transform, self.img_path)

    def get_gallery_loader(self, batch_size, num_workers=0):
        return DataLoader(
            self.gallery,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=lambda i: i)

    def get_different_word(self, source_caption, target_caption):
        source_words = source_caption.split()
        target_words = target_caption.split()
        for source_word in source_words:
            if source_word not in target_words:
                break
        for target_word in target_words:
            if target_word not in source_words:
                break
        mod_str = 'replace ' + source_word + ' with ' + target_word
        return source_word, target_word, mod_str

    def generate_test_queries_(self):
        file2imgid = {}
        for i, img in enumerate(self.data):
            file2imgid[img['file_path']] = i
        with open(self.img_path + '/test_queries.txt') as f:
            lines = f.readlines()
        self.test_queries = []
        for line in lines:
            source_file, target_file = line.split()
            idx = file2imgid[source_file]
            target_idx = file2imgid[target_file]
            source_caption = self.data[idx]['captions'][0]
            target_caption = self.data[target_idx]['captions'][0]
            source_word, target_word, mod_str = self.get_different_word(
                source_caption, target_caption)
            self.test_queries += [{
                'source_img_id': idx,
                'source_caption': source_caption,
                'target_caption': target_caption,
                'target_id': target_idx,
                'mod': {'str': mod_str}
            }]

    def caption_index_init_(self):
        """ index caption to generate training query-target example on the fly later"""

        # index caption 2 caption_id and caption 2 target_ids
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        for i, img in enumerate(self.data):
            for c in img['captions']:
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        self.caption2imgids = caption2imgids
        print(len(caption2imgids), 'unique cations')

        # parent captions are 1-word shorter than their children
        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                p = p.replace('  ', ' ').strip()
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        self.parent2children_captions = parent2children_captions

        # identify parent captions for each image
        for img in self.data:
            img['modifiable'] = False
            img['parent_captions'] = []
        for p in parent2children_captions:
            if len(parent2children_captions[p]) >= 2:
                for c in parent2children_captions[p]:
                    for imgid in caption2imgids[c]:
                        self.data[imgid]['modifiable'] = True
                        self.data[imgid]['parent_captions'] += [p]
        num_modifiable_imgs = 0
        for img in self.data:
            if img['modifiable']:
                num_modifiable_imgs += 1
        print('Modifiable images', num_modifiable_imgs)

    def caption_index_sample_(self, idx):
        while not self.data[idx]['modifiable']:
            idx = np.random.randint(0, len(self.data))

        # find random target image (same parent)
        img = self.data[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        # find the word difference between query and target (not in parent caption)
        source_caption = self.data[idx]['captions'][0]
        target_caption = self.data[target_idx]['captions'][0]
        source_word, target_word, mod_str = self.get_different_word(
            source_caption, target_caption)
        return idx, target_idx, source_word, target_word, mod_str

    def get_all_texts(self):
        texts = []
        for img in self.data:
            for c in img['captions']:
                texts.append(c)
        return texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.split == "train":
            idx, target_idx, source_word, target_word, mod_str = \
                self.caption_index_sample_(idx)
        else:
            query = self.test_queries[idx]
            idx = query["source_img_id"]
            target_idx = query["target_id"]
            mod_str = query["mod"]["str"]

        out = {}
        out['source_id'] = idx
        # out['source_caption'] = self.data[idx]['captions'][0]
        out['target_id'] = target_idx
        # out['target_caption'] = self.data[target_idx]['captions'][0]

        out["source_image"] = self.get_img(idx)
        out["source_text"] = mod_str
        out["target_image"] = self.get_img(target_idx)
        out["target_text"] = None

        return out

    def get_img(self, idx, raw_img=False):
        img_path = self.img_path + self.data[idx]['file_path']
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

    def parse_judgment(self, judgment, loss=None):
        return judgment
