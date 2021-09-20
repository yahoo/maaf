# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from ..utils.misc_utils import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

DATASET_CLASS_NAME = "QueryiMatFashion"


class GalleryiMatFashion(Dataset):

    def __init__(self, data, path, transform):
        self.img_path = path
        self.transform = transform
        self.gallery = data

    def __getitem__(self, ind):
        return {"target_image": self.get_img(ind + 1),
                "target_text": None}

    def get_img(self, image_id):
        id_string = f"{image_id:07}"
        image_path = os.path.join(
            self.img_path, id_string[:3], id_string + ".jpeg")
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
        except EnvironmentError as ee:
            # print("WARNING: EnvironmentError, defaulting to image 1", ee)
            return self.get_img(1)

        return self.transform(img)

    def __len__(self):
        return len(self.gallery)


class QueryiMatFashion(Dataset):

    def __init__(self, path, split='train', transform=None, rng=None):
        """Test split doesn't have labels, so we don't use it."""
        super().__init__()

        self.split = split
        if split in ["val", "test"]:
            split = "validation"
        self.transform = transform
        self.img_path = os.path.join(path, f"images_{split}/")

        self.data = pd.read_json(os.path.join(path, f"{split}_with_neighbors.json"))
        self.data["labelId"] = self.data["labelId"].map(tuple)
        self.data["neighbors"] = self.data["neighbors"].map(tuple)

        # only keep examples with neighbors
        self.data = self.data[self.data["neighbors"] != ()]

        label_map = pd.read_csv(
            os.path.join(path, "imat2018_label_map.tsv"),
            delimiter="\t"
        )
        self.label_dict = {row["labelId"]: row["labelName"]
                           for _, row in label_map.iterrows()}

        if rng is None:
            rng = np.random.RandomState(314159)
        self.rng = rng

        self.gallery = GalleryiMatFashion(self.data, self.img_path,
                                          self.transform)

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        target_choices = example["neighbors"]

        target = self.rng.choice(target_choices)

        target = self.data.loc[target]  # by imageId index, 1 or more off .iloc
        caption_choices = self.get_caption_choices(example["labelId"],
                                                   target["labelId"])
        caption = self.rng.choice(caption_choices)

        item = {}
        item["source_image"] = self.gallery.get_img(example["imageId"])
        item["source_text"] = caption

        item["target_image"] = self.gallery.get_img(target["imageId"])
        item["target_text"] = None

        item["judgment"] = 1

        return item

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

    def get_gallery_loader(self, batch_size, num_workers=0, category=None):
        return DataLoader(
            self.gallery,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=lambda i: i)

    def get_test_queries(self, category=None):
        return self.data

    def parse_judgment(self, judgment, loss=None):
        return judgment

    def get_label_name(self, label_id):
        return self.label_dict[int(label_id)]

    def get_caption_choices(self, reference_labels, target_labels):
        ref_not_tar, tar_not_ref, total = \
            annotation_difference(reference_labels, target_labels)

        ref_not_tar = [self.get_label_name(rr) for rr in ref_not_tar]
        tar_not_ref = [self.get_label_name(tt) for tt in tar_not_ref]

        assert total > 0
        if len(ref_not_tar) == 0:
            return [f"{tar_not_ref[0]}"]
        if len(tar_not_ref) == 0:
            return [f"not {ref_not_tar[0]}"]
        if len(ref_not_tar) == len(tar_not_ref) == 1:
            ref = ref_not_tar[0]
            tar = tar_not_ref[0]
            if (ref, tar) in attr_diff2caption:
                return attr_diff2caption[(ref, tar)]
            elif tar in target_att2caption:
                return target_att2caption[tar]
            else:
                return generic_captions(ref, tar)
        else:
            return generic_from_multi(ref_not_tar, tar_not_ref)

    def get_all_texts(self):
        # untested
        texts = []
        for first in self.label_dict.keys():
            for second in self.label_dict.keys():
                texts += self.get_caption_choices([first], [second])
        return texts

    def __len__(self):
        return len(self.data)

    def evaluate(self, model, cfg=None):
        model.eval()
        rng = np.random.RandomState(271828)

        if self.split == "train":
            queries = self.data.iloc[:10000]
        else:
            queries = self.data

        query_embs = []
        extras = []
        correct_labels = []
        for _, que in tqdm(queries.iterrows()):
            que_img = self.gallery.get_img(que["imageId"])
            que_img = torch.stack([que_img]).float().to(model.device)
            target_id = rng.choice(que["neighbors"])
            target = self.data.loc[target_id]

            caption_choices = self.get_caption_choices(que["labelId"],
                                                       target["labelId"])

            caption = self.rng.choice(caption_choices)

            embedding = model(que_img, [caption]).cpu().numpy()
            query_embs.append(embedding)

            if target_id not in queries["imageId"]:
                extras.append(target_id)

            correct_labels.append(target["labelId"])

        if len(extras) > 0:
            extras = pd.DataFrame(extras)
            combined = pd.concat([queries, self.data.loc[extras]])
        else:
            combined = queries

        target_embs = []
        for _, tar in tqdm(combined.iterrows()):
            tar_img = self.gallery.get_img(tar["imageId"])
            tar_img = torch.stack([tar_img]).float().to(model.device)
            embedding = model(tar_img, [None]).cpu().numpy()
            target_embs.append(embedding)

        query_embs = np.concatenate(query_embs, axis=0)
        target_embs = np.concatenate(target_embs, axis=0)

        query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
        target_embs = target_embs / np.linalg.norm(target_embs, axis=1, keepdims=True)

        preds = query_embs @ target_embs.T

        correct = np.zeros(preds.shape, dtype=bool)
        for ii, labels in enumerate(correct_labels):
            correct[ii] = (combined["labelId"] == labels).to_numpy()

        roc = roc_auc_score(correct.flatten(), preds.flatten())
        ap_scores = []
        for ii in range(len(correct)):
            ap_scores.append(average_precision_score(correct[ii], preds[ii]))
        map_score = np.mean(ap_scores)

        return [("roc_auc_score", roc), ("mAP", map_score)]


attr_diff2caption_sleeves = {
    ("Long Sleeved", "Short Sleeves"): ["shorter sleeves", "not as long sleeves", "less long sleeves"],
    ("Short Sleeves", "Long Sleeved"): ["longer sleeves", "longer sleeved"],
    ("Sleeveless", "Short Sleeves"): ["with short sleeves", "with sleeves but not long", "not sleeveless", "with sleeves"],
    ("Sleeveless", "Long Sleeved"): ["with long sleeves", "with sleeves", "not sleeveless", "long sleeved"],
    ("Strapless", "Sleeveless"): ["sleeveless but not strapless", "sleeveless"],
}

target_att2caption_sleeves = {
    "Sleeveless": ["sleeveless", "no sleeves", "without sleeves"],
    "Strapless": ["strapless", "no straps", "without straps"],
    "Puff Sleeves": ["puff sleeves", "puffed sleeves", "with puff sleeves"],
    "Long Sleeved": ["with long sleeves", "long sleeved", "long sleeves"],
    "Short Sleeved": ["with short sleeves", "short sleeved", "short sleeves"]
}

target_att2cap_pattern = {
    "Argyle": ["with an argyle pattern", "argyle"],
    "Camouflage": ["with a camouflage pattern", "camouflage", "camo", "with a camo pattern"],
    "Checkered": ["checkered", "with a checkered pattern", "with a checker pattern"],
    "Chevron": ["with a chevron pattern"],
    "Crochet": ["with a crochet pattern", "crochet"],
    "Floral": ["floral", "with a floral pattern", "with flowers"],
    "Fringe": ["with a fringe pattern", "with fringe"],
    "Galaxy": ["with a galaxy pattern"],
    "Geometric": ["with a geometric pattern", "geometric"],
    "Hearts": ["with hearts", "with a hearts pattern"],
    "Herringbone": ["with a herringbone pattern", "herringbone", "with a twill weave pattern"],
    "Houndstooth": ["with a houndstooth pattern", "with a hound's tooth pattern", "dogstooth", "dog's tooth", "pied-de-poule", "hounds tooth check"],
    "Leopard And Cheetah": ["with a leopard or cheetah pattern", "cheetah print", "leopard print"],
    "Marbled": ["with a marbled pattern", "marbled"],
    "Mesh": ["with a mesh pattern"],
    "Paisley": ["with a paisley pattern", "paisley"],
    "Pin Stripes": ["with pin stripes", "with a pin stripe pattern"],
    "Plaid": ["in plaid", "plaid", "with a plaid pattern"],
    "Polka Dot": ["with polka dots", "with a polka dot pattern"],
    "Quilted": ["with a quilted pattern"],
    "Ripped": ["ripped"],
    "Ruched": ["ruched", "with ruching"],
    "Ruffles": ["with ruffles", "ruffled"],
    "Sequins": ["with sequins", "sequined"],
    "Snakeskin": ["with a snakeskin pattern", "snakeskin"],
    "Stripes": ["with stripes", "striped", "with a striped pattern"],
    "Tie Dye": ["with a tie dye pattern", "tie dye"],
    "Zebra": ["with a zebra pattern", "zebra print"]
}

ref_att2cap_pattern = {
    "Argyle": ["without an argyle pattern", "not argyle"],
    "Camouflage": ["without a camouflage pattern", "not camouflage", "not camo", "without a camo pattern"],
    "Checkered": ["not checkered", "without a checkered pattern", "without a checker pattern"],
    "Chevron": ["without a chevron pattern", "no chevrons"],
    "Crochet": ["without a crochet pattern", "not crochet"],
    "Floral": ["not floral", "without a floral pattern", "without flowers", "no flowers"],
    "Fringe": ["without a fringe pattern", "without fringe", "no fringe"],
    "Galaxy": ["without a galaxy pattern", "no galaxy pattern"],
    "Geometric": ["without a geometric pattern"],
    "Hearts": ["without hearts", "without a hearts pattern", "no hearts"],
    "Herringbone": ["without a herringbone pattern", "not herringbone", "without a twill weave pattern"],
    "Houndstooth": ["without a houndstooth pattern", "without a hound's tooth pattern", "not dogstooth", "not dog's tooth", "not pied-de-poule", "no hounds tooth check"],
    "Leopard And Cheetah": ["without a leopard or cheetah pattern", "no cheetah print", "no leopard print"],
    "Marbled": ["without a marbled pattern", "not marbled"],
    "Mesh": ["without a mesh pattern"],
    "Paisley": ["without a paisley pattern", "not paisley"],
    "Pin Stripes": ["without pin stripes", "without a pin stripe pattern", "no pin stripes"],
    "Plaid": ["not plaid", "no plaid", "without a plaid pattern"],
    "Polka Dot": ["without polka dots", "without a polka dot pattern", "no polka dots"],
    "Quilted": ["without a quilted pattern", "not quilted"],
    "Ripped": ["not ripped"],
    "Ruched": ["not ruched", "without ruching"],
    "Ruffles": ["without ruffles", "not ruffled", "no ruffles"],
    "Sequins": ["without sequins", "not sequined", "no sequins"],
    "Snakeskin": ["without a snakeskin pattern", "not snakeskin"],
    "Stripes": ["without stripes", "not striped", "without a striped pattern"],
    "Tie Dye": ["without a tie dye pattern", "not tie dye"],
    "Zebra": ["without a zebra pattern", "not zebra print"]
}

synonyms_style = {
    "Hi-Lo": ["high-low", "high low"],
    "Rhinestone Studded": ["with rhinestones"],
    "Vintage Retro": ["retro", "vintage"]
}

synonyms_color = {
    "Multi Color": ["multicolor", "multicolored"]
}

target_att2caption_neckline = {
    "Backless Dresses": ["backless"],
    "Collared": ["collared"],
    "Shoulder Drapes": ["shoulder drape", "drape shoulder"],
    "Round Neck": ["round necked", "round neck", "round neckline"],
    "Square Necked": ["square necked", "square neck", "square neckline"],
    "Sweetheart Neckline": ["with a sweetheart neckline"],
    "Turtlenecks": ["turtleneck"],
    "U-Necks": ["u-neck", "u neck"],
    "V-Necks": ["v-neck", "v neck"]
}

attr_diff2caption = attr_diff2caption_sleeves
target_att2caption = target_att2caption_sleeves
target_att2caption.update(target_att2cap_pattern)
ref_att2cap = ref_att2cap_pattern
synonyms = synonyms_style
synonyms.update(synonyms_color)


def get_pattern_caption_choices(ref, tar):
    choices = [
        f"{ref}",
        f"{tar}",
        f"{ref} and {tar}",
        f"{tar} and {ref}"
    ]
    return choices


def get_material_caption_choices(ref, tar):
    ref = ref_att2cap_pattern[ref]
    tar = target_att2cap_pattern[tar]
    choices = [
        f"not {ref}",
        f"{tar}",
        f"{tar} material instead of {ref}",
        f"with {tar} material",
        f"not {ref} but {tar}"
    ]
    return choices


def annotation_difference(first, second):
    union = set(first).union(set(second))
    first_not_second = union.difference(set(second))
    second_not_first = union.difference(set(first))
    total = len(first_not_second) + len(second_not_first)
    return first_not_second, second_not_first, total


def generic_captions(ref, tar):
    if ref in synonyms:
        ref = synonyms[ref]
    else:
        ref = [ref]
    if tar in synonyms:
        tar = synonyms[tar]
    else:
        tar = [tar]

    return [f"{tt} not {rr}" for tt in tar for rr in ref]


def generic_from_multi(refs, tars):
    return [f"{tt} not {rr}" for tt in tars for rr in refs]
