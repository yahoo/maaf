# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


"""Models for Text and Image Composition."""
import torch
import math
import numpy as np

from .transformer import MultiHeadedAttention, \
    PositionwiseFeedForward, \
    PositionalEncoding, PositionalEncoder, \
    EncoderLayer, Encoder


class NormalizationLayer(torch.nn.Module):
    """Class for normalization layer."""

    def __init__(self, normalize_scale=1.0, learn_scale=True):
        super().__init__()
        self.norm_s = torch.log(torch.FloatTensor([normalize_scale]))
        if learn_scale:
            self.norm_s = torch.nn.Parameter(self.norm_s)
        self.epsilon = 1e-9

    def forward(self, x):
        norm = torch.norm(x, dim=1, keepdim=True).expand_as(x)
        factor = torch.exp(self.norm_s)
        features = factor * x / (norm + self.epsilon)
        return features


class ImgTextCompositionBase(torch.nn.Module):
    """Base class for image + text composition."""

    def __init__(self, loss, image_model=None, text_model=None):
        super().__init__()
        if image_model is None:
            self.image_model = lambda xx: None
        else:
            self.image_model = image_model
        if text_model is None:
            self.text_model = lambda xx: None
        else:
            self.text_model = text_model
        self.loss = loss

    def compose(self, img_emb, text_emb):
        raise NotImplementedError

    def extract_img_feature(self, imgs):
        if all([im is None for im in imgs]):
            return None
        return list(self.image_model(imgs).values())[-1]

    def extract_text_feature(self, texts):
        if all([tt is None for tt in texts]):
            return None
        return self.text_model(texts)

    def forward(self, images, texts):
        image_emb = self.extract_img_feature(images)
        text_emb = self.extract_text_feature(texts)

        return self.compose(image_emb, text_emb)

    def image_model_parameters(self, include_scratch=True):
        if not include_scratch:
            return self.image_model.pretrained_parameters()
        try:
            return self.image_model.parameters()
        except AttributeError:
            return []

    def image_model_fc_parameters(self):
        try:
            return self.image_model.fc.parameters()
        except AttributeError:
            return []

    def text_model_parameters(self, include_scratch=True):
        if not include_scratch:
            return self.text_model.pretrained_parameters()
        try:
            return self.text_model.parameters()
        except AttributeError:
            return []

    @property
    def device(self):
        """Only makes sense if all parameters on same device."""
        return next(self.parameters()).device


class Guess:

    def __init__(self, model_dim=512):
        self.model_dim = model_dim

    @property
    def device(self):
        return torch.device("cpu")

    def __call__(self, *args):
        return torch.tensor(np.random.rand(len(args[0]), self.model_dim))


class RandomComposition(ImgTextCompositionBase):

    def __init__(self, loss, image_model=None, text_model=None):
        super().__init__(loss, image_model=Guess(), text_model=Guess())

    def compose(self, img_emb, text_emb):
        if img_emb is not None:
            return self.image_model(img_emb)
        else:
            return self.text_model(text_emb)

    def forward(self, images, texts):
        image_emb = self.extract_img_feature(images)
        text_emb = self.extract_text_feature(texts)

        return self.compose(image_emb, text_emb)

    def extract_img_feature(self, imgs):
        if all([im is None for im in imgs]):
            return None
        return self.image_model(imgs)

    @property
    def device(self):
        return "cpu"


def get_classifier_class(input_class):

    class ImgTextCompositionClassifier(input_class):

        def __init__(self, loss, embed_dim=512, num_classes=3,
                     image_model=None, text_model=None, **kwargs):
            super().__init__(
                loss, image_model=image_model, text_model=text_model, **kwargs)

            self.classification_head = torch.nn.Linear(embed_dim, num_classes)
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, images, texts):
            image_emb = self.extract_img_feature(images)
            text_emb = self.extract_text_feature(texts)

            composed = self.compose(image_emb, text_emb)
            return self.classification_head(composed)

        def probabilities(self, images, texts):
            logits = self(images, texts)
            return self.softmax(logits)

        def compute_loss(self, images, texts, labels):
            logits = self(images, texts)
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).sum().item() / len(labels)
            loss_value = self.loss(logits, labels)
            metrics = {"loss": loss_value.item(),
                       "accuracy": accuracy}
            return loss_value, metrics

    return ImgTextCompositionClassifier


def get_regression_class(input_class):

    class ImgTextCompositionRegression(input_class):

        def __init__(self, loss, embed_dim=512,
                     image_model=None, text_model=None, **kwargs):
            super().__init__(
                loss, image_model=image_model, text_model=text_model, **kwargs)
            self.regression_head = torch.nn.Linear(embed_dim, 1)

        def forward(self, images, texts):
            image_emb = self.extract_img_feature(images)
            text_emb = self.extract_text_feature(texts)
            composed = self.compose(image_emb, text_emb)
            return torch.sigmoid(self.regression_head(composed))

        def compute_loss(self, images, texts, labels):
            output = self(images, texts)
            loss_value = self.loss(output, labels.float())
            metrics = {"loss": loss_value.item()}
            return loss_value, metrics

    return ImgTextCompositionRegression


def get_metric_learning_class(input_class):

    class ImgTextCompositionMetricModel(input_class):

        def __init__(self, loss, image_model=None, text_model=None,
                     initial_normalization_factor=4.0, **kwargs):
            super().__init__(
                loss, image_model=image_model, text_model=text_model, **kwargs)
            self.normalization_layer = NormalizationLayer(
                normalize_scale=initial_normalization_factor, learn_scale=True)

        def forward(self, images, texts):
            image_emb = self.extract_img_feature(images)
            text_emb = self.extract_text_feature(texts)
            embedding = self.compose(image_emb, text_emb)
            return self.normalization_layer(embedding)

        def compute_loss(self, source_images, source_texts,
                         target_images, target_texts,
                         judgments=None):

            source_emb = self.forward(source_images, source_texts)
            target_emb = self.forward(target_images, target_texts)

            assert source_emb.shape[1] == target_emb.shape[1]
            loss_value = self.loss(source_emb, target_emb, judgments=judgments)
            metrics = {"loss": loss_value.item()}
            if torch.isnan(loss_value):
                import IPython; IPython.embed()
            return loss_value, metrics

    return ImgTextCompositionMetricModel


class SimpleModelImageOnly(ImgTextCompositionBase):

    def compose(self, img_emb, text_emb):
        return img_emb


class SimpleModelTextOnly(ImgTextCompositionBase):

    def compose(self, img_emb, text_emb):
        if len(text_emb[0].shape) > 2:
            return self.pool_text(text_emb[0], text_emb[1])
        return text_emb

    def pool_text(self, xx, text_mask):
        # average pool ignoring the masked entries
        xx = xx.transpose(-2, -1)
        embed_dim = xx.shape[1]
        # xx has shape: batch_size x feat_dim x num_tokens
        # to correctly broadcast, we must add a dimension to text_mask with unsqueeze
        xx = xx.masked_fill(text_mask.unsqueeze(1) == 0, 0.)
        denominators = (text_mask != 0).sum(1)
        denominators = denominators.float().unsqueeze(1)
        return torch.sum(xx, -1).view(-1, embed_dim) / denominators


class Addition(ImgTextCompositionBase):
    """Vector addition model."""

    def compose(self, img_emb, text_emb):
        if img_emb is None:
            assert text_emb is not None, "No images or text available"
            return text_emb
        elif text_emb is None:
            return img_emb

        return img_emb + text_emb


class MixedPositionalEncoder(torch.nn.Module):
    """
    Add positioning information and encode. The first learned_positions tokens
    have learned positional embeddings; any remaining tokens have fixed sinusoidal
    encodings as in Vaswani et al.
    """

    def __init__(self, encoder, learned_positions, embed_dim,
                 dropout, max_len=5000):
        super().__init__()
        self.encoder = encoder
        device = next(self.encoder.parameters()).device
        self.dropout = torch.nn.Dropout(p=dropout)

        self.learned_encodings = torch.nn.Parameter(
            torch.Tensor(learned_positions, embed_dim).to(device))
        # initialization copied from nn.Linear
        torch.nn.init.kaiming_uniform_(self.learned_encodings,
                                       a=math.sqrt(5))

        # fixed encodings for latter part of sequence
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.to(device)
        self.register_buffer('pe', pe)

        self.pos_embed = torch.cat([self.learned_encodings, pe], 0).unsqueeze(0)

    def forward(self, xx, mask):
        "Take in and process masked sequence."
        xx = self.dropout(xx + self.pos_embed[:, :xx.size(1)])
        return self.encoder(xx, mask)


class MAAF(ImgTextCompositionBase):

    def __init__(self, loss, model_dim=512, num_heads=8, ff_width=256,
                 dropout=0.1, num_blocks=1, position_encodings=None,
                 softmax_replacement=None, output="rwpool",
                 image_model=None, text_model=None,
                 ):
        super().__init__(loss, image_model=image_model, text_model=text_model)

        if softmax_replacement == "identity":
            softmax_replacement = torch.nn.Identity()
        attn = MultiHeadedAttention(num_heads, model_dim,
                                    softmax_replacement=softmax_replacement)

        ff = PositionwiseFeedForward(model_dim, ff_width, dropout)

        enclayer = EncoderLayer(model_dim, attn, ff, dropout)
        self.encoder = Encoder(enclayer, num_blocks)

        if position_encodings == "sinusoidal":
            self.position = PositionalEncoding(model_dim, dropout)
            self.model = PositionalEncoder(self.encoder, self.position)
        elif position_encodings == "mixed":
            print("Using mixed positional encodings")
            self.model = MixedPositionalEncoder(
                self.encoder, self.image_model.get_num_tokens(),
                model_dim, dropout)
        else:
            self.model = self.encoder

        # Initialize parameters (in MAAF layers only) with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.output = output
        if self.output == "token":
            self.holistic = \
                torch.randn(model_dim, requires_grad=True)
            self.holistic = torch.nn.Parameter(self.holistic)
        else:
            self.holistic = None

    @classmethod
    def from_config(cls, loss, cfg, image_model=None, text_model=None):
        return cls(
            loss,
            embed_dim=cfg.MODEL.EMBED_DIM,
            num_heads=cfg.MODEL.MAAF.ATTENTION_HEADS,
            ff_width=cfg.MODEL.MAAF.BLOCK_WIDTH,
            dropout=cfg.MODEL.DROPOUT_RATE,
            num_blocks=cfg.MODEL.MAAF.NUM_BLOCKS,
            position_encodings=cfg.MODEL.MAAF.POSITION_ENCODING,
            softmax_replacement=cfg.MODEL.MAAF.ATTN_SOFTMAX_REPLACEMENT,
            output=cfg.MODEL.MAAF.OUTPUT,
            image_model=image_model,
            text_model=text_model
        )

    def extract_img_feature(self, imgs):
        if all([im is None for im in imgs]):
            return None
        return self.image_model(imgs)["projections"]

    def compose(self, img_emb, text_emb):
        embeddings = []
        masks = []

        if img_emb is not None:
            img_mask = torch.ones(img_emb.shape[0], img_emb.shape[1],
                                  dtype=torch.long, device=self.device)
            embeddings += [img_emb]
            masks += [img_mask]

        if text_emb is not None:
            text_emb, text_mask = text_emb
            # text_emb = text_emb.transpose(1, 0)
            embeddings += [text_emb]
            masks += [text_mask]

        concat_features = torch.cat(embeddings, 1)
        concat_mask = torch.cat(masks, 1)
        composed = self.model(concat_features, concat_mask)

        final_embs = []
        if img_emb is not None:
            mod_im_feat = composed[:, :img_emb.shape[1]]
            pooled_im_feat = self.pool_image(mod_im_feat)
            final_embs += [pooled_im_feat]

        if text_emb is not None:
            mod_text_feat = composed[:, -text_emb.shape[1]:]
            pooled_text_feat = self.pool_text(mod_text_feat, text_mask)
            final_embs += [pooled_text_feat]

        return torch.mean(torch.stack(final_embs), 0)

    def pool_image(self, xx):
        if self.output == "rwpool":
            return self.image_model.resolutionwise_pool(xx)
        else:
            return torch.mean(xx, 1)

    def pool_text(self, xx, text_mask):
        # average pool ignoring the masked entries
        xx = xx.transpose(-2, -1)
        embed_dim = xx.shape[1]
        # xx has shape: batch_size x feat_dim x num_tokens
        # to correctly broadcast, we must add a dimension to text_mask with unsqueeze
        xx = xx.masked_fill(text_mask.unsqueeze(1) == 0, 0.)
        denominators = (text_mask != 0).sum(1)
        denominators = denominators.float().unsqueeze(1)
        return torch.sum(xx, -1).view(-1, embed_dim) / denominators


class ResidualMAAF(MAAF):

    def extract_img_feature(self, imgs):
        if all([im is None for im in imgs]):
            return None
        return self.image_model(imgs)

    def compose(self, img_feat, text_emb):
        if img_feat is None:
            img_maaf_inputs = None
        else:
            img_maaf_inputs = img_feat["projections"]

        maaf_feat = MAAF.compose(self, img_maaf_inputs, text_emb)

        addition_feat = 0
        if img_feat is not None:
            addition_feat = addition_feat + img_feat["fc"]
        if text_emb is not None:
            text_emb, text_mask = text_emb
            pooled_text_feat = self.pool_text(text_emb, text_mask)
            addition_feat = addition_feat + pooled_text_feat

        return maaf_feat + addition_feat


class Concat(ImgTextCompositionBase):
    """Concatenation model."""

    def __init__(self, texts, loss, embed_dim=512, dropout=0.1):
        super().__init__(texts, loss, image_model=None, text_model=None)

        self.model = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(2 * embed_dim, embed_dim))

    def compose(self, img_emb, text_emb):
        if img_emb is None:
            return text_emb
        if text_emb is None:
            return img_emb
        concat = torch.cat((img_emb, text_emb), dim=1)
        return self.model(concat)
