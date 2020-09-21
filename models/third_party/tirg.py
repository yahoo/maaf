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

import numpy as np
import torch
import torch.nn.functional as F
from .. import image_model as image_model
from .. import text_model as text_model
from . import torch_functions


class ImgTextCompositionBase(torch.nn.Module):
    """Base class for image + text composition."""

    def __init__(self, opt, texts, text_model_sequence_output=False):
        super(ImgTextCompositionBase, self).__init__()
        self.normalization_layer = torch_functions.NormalizationLayer(
            normalize_scale=4.0, learn_scale=True)
        self.soft_triplet_loss = torch_functions.TripletLoss()
        self.opt = opt

        self.img_model = self.load_image_model(self.opt)
        self.text_model = self.load_text_model(
            texts, self.opt, text_model_sequence_output=text_model_sequence_output)

    def load_image_model(self, opt):
        if opt.image_model_arch is None:
            opt.image_model_arch = 'resnet18'
        if opt.image_model_arch in ['resnet50', 'resnet18']:
            img_model = image_model.ResNetBase(opt)
        else:
            raise Exception('Network {} not available!'.format(
                opt.image_model_arch))
        return img_model

    def load_text_model(self, texts, opt, text_model_sequence_output):
        # text model
        if opt.text_model_arch == 'BERT':
            txtmod = text_model.BERTModel()
            print("BERT successfully loaded")
        elif opt.text_model_arch == 'embeddings':
            txtmod = text_model.EmbeddingModel(texts_to_build_vocab=texts,
                                               word_embed_dim=opt.embed_dim)
            print("Using bare embeddings for text")
            print("vocab size", txtmod.vocab.get_size())
        elif opt.text_model_arch == 'transformer':
            # TODO: pass in args from opt
            txtmod = text_model.SimpleTransformerEncoderModel(
                texts_to_build_vocab=texts,
                word_embed_dim=opt.embed_dim,
                d_model=opt.embed_dim,
                d_ff=opt.embed_dim,
                num_layers=opt.text_model_layers,
                dropout=opt.dropout_rate)
            print("Using transformer model for text")
            print("vocab size", txtmod.vocab.get_size())
        else:
            txtmod = text_model.TextLSTMModel(
                texts_to_build_vocab=texts,
                word_embed_dim=opt.embed_dim,
                lstm_hidden_dim=opt.embed_dim,
                text_model_sequence_output=text_model_sequence_output,
                num_layers=opt.text_model_layers,
                dropout=opt.dropout_rate)
            print("vocab size", txtmod.vocab.get_size())

        if opt.freeze_text_model:
            print("Freezing Text model weights")
            for param in txtmod.parameters():
                param.requires_grad = False
        return txtmod

    def compose_img_text(self, imgs, texts):
        raise NotImplementedError

    def compute_loss(self,
                     imgs_query,
                     modification_texts,
                     imgs_target,
                     soft_triplet_loss=True):
        mod_img1 = self.compose_img_text(imgs_query, modification_texts)
        mod_img1 = self.normalization_layer(mod_img1)

        if self.opt.image_with_unk_string:
            empty_texts = ["<UNK>" for _ in range(len(imgs_target))]
            img2 = self.compose_img_text(imgs_target, empty_texts)
        else:
            img2 = self.extract_img_feature(imgs_target)
        img2 = self.normalization_layer(img2)

        assert mod_img1.shape[1] == img2.shape[1]
        if soft_triplet_loss:
            return self.compute_soft_triplet_loss_(mod_img1, img2)
        else:
            return self.compute_batch_based_classification_loss_(mod_img1, img2)

    def compute_soft_triplet_loss_(self, mod_img1, img2):
        triplets = []
        labels = list(range(mod_img1.shape[0])) + list(range(img2.shape[0]))
        for i in range(len(labels)):
            triplets_i = []
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    for k in range(len(labels)):
                        if labels[i] != labels[k]:
                            triplets_i.append([i, j, k])
            np.random.shuffle(triplets_i)
            triplets += triplets_i[:3]
        assert (triplets and len(triplets) < 2000)
        return self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)

    def compute_batch_based_classification_loss_(self, mod_img1, img2):
        dots = torch.mm(mod_img1, img2.transpose(0, 1))
        labels = torch.tensor(range(dots.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        losses = F.cross_entropy(dots, labels, reduction='none')
        if self.opt.drop_worst_flag:
            losses, idx = torch.topk(
                losses, k=int(losses.shape[0] * (1-self.opt.drop_worst_rate)),
                largest=False)
        final_loss = losses.mean()
        return final_loss

    def extract_img_feature(self, imgs):
        return self.img_model(imgs)

    def extract_text_feature(self, texts):
        if self.opt.text_model_arch == 'BERT':
            text_features, text_masks = self.text_model(texts)
            text_features = self.text_projection_layer(text_features)
            return ((text_features, text_masks))
        else:
            return self.text_model(texts)

class ConCatModule(torch.nn.Module):

    def __init__(self):
        super(ConCatModule, self).__init__()

    def forward(self, xx):
        xx = torch.cat(xx, dim=1)
        return xx


class Concat(ImgTextCompositionBase):
    """Concatenation model."""

    def __init__(self, opt, texts):
        super(Concat, self).__init__(opt, texts)
        embed_dim = opt.embed_dim

        class Composer(torch.nn.Module):
            """Inner composer class."""

            def __init__(self, opt):
                super(Composer, self).__init__()
                self.opt = opt
                self.m = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
                    torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
                    torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
                    torch.nn.Dropout(self.opt.dropout_rate),
                    torch.nn.Linear(2 * embed_dim, embed_dim))

            def forward(self, x):
                f = torch.cat(x, dim=1)
                f = self.m(f)
                return f

        self.composer = Composer(opt)

    def compose_img_text(self, imgs, texts):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(texts)
        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features):
        return self.composer((img_features, text_features))


class TIRG(ImgTextCompositionBase):
    """The TIRG model.

    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, opt, texts):
        super(TIRG, self).__init__(opt, texts,
                                   text_model_sequence_output=False)
        embed_dim = opt.embed_dim
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.gated_feature_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))
        self.res_info_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))

    def compose_img_text(self, imgs, texts):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(texts)
        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features):
        f1 = self.gated_feature_composer((img_features, text_features))
        f2 = self.res_info_composer((img_features, text_features))
        f = F.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1]
        return f


class TIRGLastConv(ImgTextCompositionBase):
    """The TIRG model with spatial modification over the last conv layer.

    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119

    To get behavior from the Google code, use:
    att_layer_spec=none
    image_model_arch=resnet18
    """

    def __init__(self, opt, texts):
        super(TIRGLastConv, self).__init__(opt, texts,
                                           text_model_sequence_output=False)
        assert not any([num in opt.att_layer_spec for num in ["2", "3", "4"]])
        embed_dim = opt.embed_dim
        img_embed_dim = self.img_model.base_network_embedding_size
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.mod2d = torch.nn.Sequential(
            torch.nn.BatchNorm2d(img_embed_dim + embed_dim),
            torch.nn.Conv2d(img_embed_dim + embed_dim,
                            img_embed_dim + embed_dim, [3, 3],
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(img_embed_dim + embed_dim,
                            img_embed_dim, [3, 3], padding=1),
        )

        self.mod2d_gate = torch.nn.Sequential(
            torch.nn.BatchNorm2d(img_embed_dim + embed_dim),
            torch.nn.Conv2d(img_embed_dim + embed_dim,
                            img_embed_dim + embed_dim, [3, 3],
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(img_embed_dim + embed_dim,
                            img_embed_dim, [3, 3], padding=1),
        )

    def load_image_model(self, opt):
        return image_model.ResNetSpatial(opt)

    def compose_img_text(self, imgs, texts):
        text_features = self.extract_text_feature(texts)
        xx = self.img_model(imgs).transpose(-2, -1)
        xx = xx.reshape(xx.shape[0], xx.shape[1], 7, 7)

        # mod
        y = text_features
        y = y.reshape((y.shape[0], y.shape[1], 1, 1)).repeat(
            1, 1, xx.shape[2], xx.shape[3])
        z = torch.cat((xx, y), dim=1)
        t = self.mod2d(z)
        tgate = self.mod2d_gate(z)
        xx = self.a[0] * F.sigmoid(tgate) * xx + self.a[1] * t

        xx = self.img_model.final_layers(xx)
        return xx

    def extract_img_feature(self, imgs):
        xx = self.img_model(imgs)
        return self.img_model.final_layers(xx)
