
# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from .base_dataset import BaseDataset
import numpy as np
import random
import PIL


class CSSDataset(BaseDataset):
  """CSS dataset."""

  def __init__(self, path, split='train', transform=None):
    super(CSSDataset, self).__init__()

    self.img_path = path + '/images/'
    self.transform = transform
    self.split = split
    self.data = np.load(path + '/css_toy_dataset_novel2_small.dup.npy',
                        allow_pickle=True, encoding="latin1").item()
    self.mods = self.data[self.split]['mods']
    self.imgs = []
    for objects in self.data[self.split]['objects_img']:
      label = len(self.imgs)
      if 'labels' in self.data[self.split]:
        label = self.data[self.split]['labels'][label]
      self.imgs += [{
          'objects': objects,
          'label': label,
          'captions': [str(label)],
          "image_id": len(self.imgs)
      }]

    self.imgid2modtarget = {}
    for i in range(len(self.imgs)):
      self.imgid2modtarget[i] = []
    for i, mod in enumerate(self.mods):
      for k in range(len(mod['from'])):
        f = mod['from'][k]
        t = mod['to'][k]
        self.imgid2modtarget[f] += [(i, t)]

    self.generate_test_queries_()

  def generate_test_queries_(self):
    test_queries = []
    for mod in self.mods:
      for i, j in zip(mod['from'], mod['to']):
        test_queries += [{
            'source_img_id': i,
            'target_caption': self.imgs[j]['captions'][0],
            'mod': {
                'str': mod['to_str']
            }
        }]
    self.test_queries = test_queries

  def get_1st_training_query(self):
    i = np.random.randint(0, len(self.mods))
    mod = self.mods[i]
    j = np.random.randint(0, len(mod['from']))
    self.last_from = mod['from'][j]
    self.last_mod = [i]
    return mod['from'][j], i, mod['to'][j]

  def get_2nd_training_query(self):
    modid, new_to = random.choice(self.imgid2modtarget[self.last_from])
    while modid in self.last_mod:
      modid, new_to = random.choice(self.imgid2modtarget[self.last_from])
    self.last_mod += [modid]
    # mod = self.mods[modid]
    return self.last_from, modid, new_to

  def generate_random_query_target(self):
    try:
      if len(self.last_mod) < 2:
        img1id, modid, img2id = self.get_2nd_training_query()
      else:
        img1id, modid, img2id = self.get_1st_training_query()
    except:
      img1id, modid, img2id = self.get_1st_training_query()

    out = {}
    out['source_img_id'] = img1id
    out['source_img_data'] = self.get_img(img1id)
    out['target_img_id'] = img2id
    out['target_img_data'] = self.get_img(img2id)
    out['mod'] = {'id': modid, 'str': self.mods[modid]['to_str']}
    return out

  def __len__(self):
    return len(self.imgs)

  def get_all_texts(self):
    return [mod['to_str'] for mod in self.mods]

  def get_img(self, idx, raw_img=False, get_2d=False):
    """Gets CSS images."""
    def generate_2d_image(objects):
      img = np.ones((64, 64, 3))
      colortext2values = {
          'gray': [87, 87, 87],
          'red': [244, 35, 35],
          'blue': [42, 75, 215],
          'green': [29, 205, 20],
          'brown': [129, 74, 25],
          'purple': [129, 38, 192],
          'cyan': [41, 208, 208],
          'yellow': [255, 238, 51]
      }
      for obj in objects:
        s = 4.0
        if obj['size'] == 'large':
          s *= 2
        c = [0, 0, 0]
        for j in range(3):
          c[j] = 1.0 * colortext2values[obj['color']][j] / 255.0
        y = obj['pos'][0] * img.shape[0]
        x = obj['pos'][1] * img.shape[1]
        if obj['shape'] == 'rectangle':
          img[int(y - s):int(y + s), int(x - s):int(x + s), :] = c
        if obj['shape'] == 'circle':
          for y0 in range(int(y - s), int(y + s) + 1):
            x0 = x + (abs(y0 - y) - s)
            x1 = 2 * x - x0
            img[y0, int(x0):int(x1), :] = c
        if obj['shape'] == 'triangle':
          for y0 in range(int(y - s), int(y + s)):
            x0 = x + (y0 - y + s) / 2
            x1 = 2 * x - x0
            x0, x1 = min(x0, x1), max(x0, x1)
            img[y0, int(x0):int(x1), :] = c
      return img

    if self.img_path is None or get_2d:
      img = generate_2d_image(self.imgs[idx]['objects'])
    else:
      img_path = self.img_path + ('/css_%s_%06d.png' % (self.split, int(idx)))
      with open(img_path, 'rb') as f:
        img = PIL.Image.open(f)
        img = img.convert('RGB')

    if raw_img:
      return img
    if self.transform:
      img = self.transform(img)
    return img
