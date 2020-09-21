# Copyright 2020 Verizon Media, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

import torch
import torch.nn as nn
import copy
from .TransformerModel import SublayerConnection, LayerNorm, clones


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
    """Decoder is made of self-attn, src-attn, and feed forward
    mode format:
    xxx : self attn on x
    mmm : self attn on m
    xmm : x is query, m is keys and values
    xff: feed_forward on x

    DecoderLayer is equivalent to xxx_xmm_xff,
    except that this layer return (x, m) rather than just x
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
        "Follow Figure 1 (right) for connections."
        m = memory
        for ii, spec in enumerate(self.mode.split("_")):
            x, m = self.get_sublayer(x, m, src_mask, tgt_mask, ii, spec)
        return x, m
