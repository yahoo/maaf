# Copyright 2022 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import numpy as np


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # import IPython; IPython.embed()
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None,
              softmax_replacement=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if softmax_replacement is not None:
        scores = scores.masked_fill(mask == 0, 0.)
        p_attn = softmax_replacement(scores)
    else:
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, softmax_replacement=None):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.softmax_replacement = softmax_replacement

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            # Same mask applied to all 49 visual tokens
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout,
                                 softmax_replacement=self.softmax_replacement)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionalDecoder(nn.Module):
    """
    Add positioning infomation and decode, without an encoder.
    """
    def __init__(self, decoder, pos_embed):
        super(PositionalDecoder, self).__init__()
        self.decoder = decoder
        self.pos_embed = pos_embed

    def forward(self, src, src_mask, tgt, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(src, src_mask, tgt, tgt_mask)

    def decode(self, src, src_mask, tgt, tgt_mask):
        return self.decoder(self.pos_embed(src), tgt, tgt_mask, src_mask)


class PositionalEncoder(nn.Module):
    """
    Add positioning infomation and encode, without an decoder.
    """
    def __init__(self, encoder, pos_embed):
        super().__init__()
        self.encoder = encoder
        self.pos_embed = pos_embed

    def forward(self, xx, mask):
        "Take in and process masked sequence."
        return self.encoder(self.pos_embed(xx), mask)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class SymmetricDecoder(nn.Module):
    """Like Decoder but allows 'memory' to be modified and passed around."""
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm_x = LayerNorm(layer.size)
        self.norm_m = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x, memory = layer(x, memory, src_mask, tgt_mask)
        return self.norm_x(x), self.norm_m(memory)


class FlexibleDecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed-forward
    mode format:
    xxx : self attn on x
    mmm : self attn on m
    xmm : x is query, m is keys and values
    xff: feed_forward on x
    xmm.mxx: cross-attn 'in parallel' with x, m inputs the same
        (i.e., mxx does not use output of xmm)

    Separate by _ for a sequence of operations.


    DecoderLayer is equivalent to xxx_xmm_xff,
    except that this layer returns (x, m) rather than just x
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout,
                 mode="xxx_xmm_xff"):
        super().__init__()
        self.mode = mode
        self.size = size
        self.self_attn_x = self_attn
        self.src_attn_x = src_attn
        self.feed_forward_x = feed_forward
        self.self_attn_m = copy.deepcopy(self_attn)
        self.src_attn_m = copy.deepcopy(src_attn)
        self.feed_forward_m = copy.deepcopy(feed_forward)
        self.sublayer = clones(SublayerConnection(size, dropout),
                               len(self.mode.split("_")))

    def get_sublayer(self, x, m, src_mask, tgt_mask, index, spec):
        sublayer = self.sublayer[index]
        if spec == "xxx":
            x = sublayer(x, lambda y: self.self_attn_x(y, y, y, tgt_mask))
        elif spec == "xmm":
            x = sublayer(x, lambda y: self.src_attn_x(y, m, m, src_mask))
        elif spec == "mmm":
            m = sublayer(m, lambda y: self.self_attn_m(y, y, y, src_mask))
        elif spec == "mxx":
            m = sublayer(m, lambda y: self.src_attn_m(y, x, x, tgt_mask))
        elif spec == "xff":
            x = sublayer(x, self.feed_forward_x)
        elif spec == "mff":
            m = sublayer(m, self.feed_forward_m)
        elif spec == "xmm.mxx":
            x_temp = sublayer(x, lambda y: self.src_attn_x(y, m, m, src_mask))
            m = sublayer(m, lambda y: self.src_attn_m(y, x, x, tgt_mask))
            x = x_temp
        else:
            raise ValueError("Invalid attn_2stream_mode")
        return x, m

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        for ii, spec in enumerate(self.mode.split("_")):
            x, m = self.get_sublayer(x, m, src_mask, tgt_mask, ii, spec)
        return x, m
