# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

import torch
import copy
import math

from . import image_model
from .TransformerModel import MultiHeadedAttention, \
    PositionwiseFeedForward, \
    PositionalEncoding, PositionalEncoder, PositionalDecoder, \
    EncoderLayer, Encoder
from .TransformerExtensions import SymmetricDecoder, FlexibleDecoderLayer
from .third_party.tirg import ImgTextCompositionBase

def get_multiheaded_attn(opt):
    heads = opt.number_attention_heads
    d_model = opt.embed_dim
    if opt.attn_softmax_replacement == "none":
        softmax_replacement = None
    elif opt.attn_softmax_replacement == "identity":
        softmax_replacement = lambda xx: xx
    else:
        raise ValueError(opt.attn_softmax_replacement + " not implemented")

    return MultiHeadedAttention(heads, d_model,
                                softmax_replacement=softmax_replacement)


class SimpleModelImageOnly(ImgTextCompositionBase):

    def compose_img_text(self, imgs, texts):
        return self.extract_img_feature(imgs)


class SimpleModelTextOnly(ImgTextCompositionBase):

    def compose_img_text(self, imgs, texts):
        return self.extract_text_feature(texts)


class Addition(ImgTextCompositionBase):
    """Vector addition model"""

    def compose_img_text(self, imgs, texts):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(texts)
        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features):
        return img_features + text_features


class AttentionComposer(torch.nn.Module):
    """Inner composer class for AttentionComposition."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        N = self.opt.number_attention_blocks
        d_model = opt.embed_dim
        d_ff = self.opt.width_per_attention_block

        dropout = self.opt.dropout_rate

        c = copy.deepcopy
        attn = get_multiheaded_attn(self.opt)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        declayer = FlexibleDecoderLayer(
            d_model, c(attn), c(attn), c(ff), dropout,
            mode=self.opt.attn_2stream_mode)
        self.decoder = SymmetricDecoder(declayer, N)

        if self.opt.attn_positional_encoding == "sinusoidal":
            if self.opt.attn_2stream_mode != "xxx_xmm_xff":
                raise ValueError("Positional encodings not implemented properly for this mode")
            self.pos_embed = PositionalEncoding(d_model, dropout)
            self.m = PositionalDecoder(self.decoder, self.pos_embed)
        elif self.opt.attn_positional_encoding is None:
            self.m = self.decoder
        else:
            raise ValueError("{} not supported for this model".format(
                self.opt.attn_positional_encoding))

        # Initialize parameters with Glorot / fan_avg.
        for p in self.m.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, xx):
        img_features_proj, text_features, text_mask = xx
        if self.opt.attn_positional_encoding is not None:
            xout, mout = self.m(img_features_proj, None, text_features, text_mask)
        else:
            xout, mout = self.m(img_features_proj, text_features, text_mask, None)
        return torch.cat([xout, mout], 1)


class AttentionComposition(ImgTextCompositionBase):

    def __init__(self, opt, texts):
        super(AttentionComposition, self).__init__(opt, texts,
                                                   text_model_sequence_output=True)
        self._create_composer(opt)

    def _create_composer(self, opt):
        self.composer = AttentionComposer(opt)

    def load_image_model(self, opt):
        return image_model.ResNetSpatial(opt)

    def compose_img_text(self, imgs, texts):
        img_features_proj = self.img_model(imgs)
        text_features, text_mask = self.extract_text_feature(texts)
        return self.compose_img_text_features(img_features_proj,
                                              text_features, text_mask)

    def compose_img_text_features(self, img_features,
                                  text_features, text_mask):
        composed = self.composer((img_features, text_features, text_mask))
        mod_im_feat = composed[:, :img_features.shape[1]]
        pooled_im_feat = self.pool_image(mod_im_feat)
        if self.opt.sequence_concat_include_text:
            mod_text_feat = composed[:, -text_features.shape[1]:]
            pooled_text_feat = self.pool_text(mod_text_feat, text_mask)
            return torch.mean(torch.stack(
                [pooled_im_feat, pooled_text_feat]), 0)
        else:
            return pooled_im_feat

    def extract_img_feature(self, imgs, for_composition=False):
        img_features = self.img_model(imgs)
        if self.opt.sequence_concat_img_through_attn:
            # pass through a dummy single-token text, masked out
            mask = torch.zeros((img_features.shape[0], 1))
            mask = mask.long().cuda()
            dummy_text_features = \
                torch.zeros([img_features.shape[0], 1,
                             img_features.shape[2]]).cuda()

            img_features = \
                self.composer((img_features, dummy_text_features, mask))
        return self.pool_image(img_features)

    def pool_image(self, xx):
        if self.opt.resolutionwise_pool:
            return self.img_model.resolutionwise_pool(xx)
        else:
            return torch.mean(xx, 1)

    def pool_text(self, xx, text_mask):
        # average pool ignoring the masked entries
        xx = xx.transpose(-2, -1)
        # xx has shape: batch_size x feat_dim x num_tokens
        # to correctly broadcast, we must add a dimension to text_mask with unsqueeze
        xx = xx.masked_fill(text_mask.unsqueeze(1) == 0, 0.)
        denominators = (text_mask != 0).sum(1)
        denominators = denominators.float().unsqueeze(1)
        return torch.sum(xx, -1).view(-1, self.opt.embed_dim) / denominators


class MixedPositionalEncoder(torch.nn.Module):
    """
    Add positioning information and encode.
    The first learned_positions tokens
    have learned positional embeddings; any remaining tokens have fixed sinusoidal
    encodings as in Vaswani et al.
    """

    def __init__(self, encoder, learned_positions, d_model,
                 dropout, max_len=5000):
        super().__init__()
        self.encoder = encoder
        self.dropout = torch.nn.Dropout(p=dropout)

        self.learned_encodings = torch.nn.Parameter(
            torch.Tensor(learned_positions, d_model).cuda())
        # initialization copied from nn.Linear
        torch.nn.init.kaiming_uniform_(self.learned_encodings,
                                       a=math.sqrt(5))

        # fixed encodings for latter part of sequence
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.cuda()
        self.register_buffer('pe', pe)

        self.pos_embed = torch.cat([self.learned_encodings, pe], 0).unsqueeze(0)

    def forward(self, xx, mask):
        "Take in and process masked sequence."
        xx = self.dropout(xx + self.pos_embed[:,:xx.size(1)])
        return self.encoder(xx, mask)


class SequenceConcatComposer(torch.nn.Module):

    def __init__(self, opt, model):
        super().__init__()
        self.opt = opt

        N = self.opt.number_attention_blocks
        d_model = opt.embed_dim
        d_ff = self.opt.width_per_attention_block
        dropout = self.opt.dropout_rate

        attn = get_multiheaded_attn(self.opt)
        ff_layer = PositionwiseFeedForward(d_model, d_ff, dropout)

        enclayer = EncoderLayer(d_model, attn, ff_layer, dropout)
        self.encoder = Encoder(enclayer, N)

        if self.opt.attn_positional_encoding == "sinusoidal":
            self.position = PositionalEncoding(d_model, dropout)
            self.m = PositionalEncoder(self.encoder, self.position)
        elif self.opt.attn_positional_encoding == "mixed":
            print("Using mixed positional encodings")
            self.m = MixedPositionalEncoder(
                self.encoder, model.img_model.get_num_tokens(),
                d_model, dropout)
        else:
            self.m = self.encoder

        # Initialize parameters with Glorot / fan_avg.
        for p in self.m.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, xx):
        img_features_proj, text_features, text_mask = xx
        concat_features = torch.cat([img_features_proj, text_features], 1)
        im_mask = torch.ones(img_features_proj.shape[:2]).long().cuda()
        concat_mask = torch.cat([im_mask, text_mask], 1)
        return self.m(concat_features, concat_mask)


class SequenceConcatAttention(AttentionComposition):

    def _create_composer(self, opt):
        self.composer = SequenceConcatComposer(opt, self)

    def extract_img_feature(self, imgs, for_composition=False):
        img_features_proj = self.img_model(imgs)
        if self.opt.sequence_concat_img_through_attn:
            mask = torch.ones(img_features_proj.shape[:2]).long().cuda()
            img_features_proj = self.composer.m(img_features_proj, mask)
        return self.pool_image(img_features_proj)


class SeqCatWithOutputToken(AttentionComposition):

    def __init__(self, opt, texts):
        super().__init__(opt, texts)
        self.composer = SequenceConcatComposer(opt, self)
        # output token
        self.holistic = torch.randn(opt.embed_dim, requires_grad=True).cuda()

    def compose_img_text_features(self, img_features,
                                  text_features, text_mask):
        holistic = self.holistic.repeat(img_features.shape[0], 1, 1)
        holistic_plus_img = torch.cat([holistic, img_features], 1)
        composed = self.composer((holistic_plus_img, text_features, text_mask))
        return composed[:, 0]

    def extract_img_feature(self, imgs, for_composition=False):
        img_features = self.img_model(imgs)
        holistic = self.holistic.repeat(img_features.shape[0], 1, 1)
        holistic_plus_img = torch.cat([holistic, img_features], 1)
        mask = torch.ones(holistic_plus_img.shape[:2]).long().cuda()
        result = self.composer.m(holistic_plus_img, mask)
        return result[:, 0]
