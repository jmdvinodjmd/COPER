#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from einops import rearrange, repeat


class MultiheadAttention(nn.Module):
    def __init__(self, latent_dim, data_dim=None, num_heads=1, head_dim=50, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        if data_dim == None:
            data_dim = latent_dim

        self.w_q = nn.Linear(latent_dim, inner_dim)
        self.w_k = nn.Linear(data_dim, inner_dim)
        self.w_v = nn.Linear(data_dim, inner_dim)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, latent_dim)

    def forward(self, latent, data, mask=None):
        # x (batch, num_latents=1, latents),  context(batch, features, channels=1)
        # d_dim = latent.shape[-1]
        # print('x:', x.shape)

        Q = self.w_q(latent) # (batch, num_latents, inner_dim)
        K = self.w_k(data) # (batch, features, inner_dim)
        V = self.w_v(data) # (batch, features, inner_dim)
        Q, K, V = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.num_heads), (Q, K, V))
        QK = torch.einsum('b i k, b j k -> b i j', Q, K) * self.scale # (batch*heads, num_latents, features)
        # print('QK:', QK.shape)

        if mask is not None:
            QK = QK.masked_fill(mask==1, value=-1e9)

        att = QK.softmax(-1)  # (batch*heads, num_latents, features)
        att = self.dropout(att)
        # print('att:', att.shape)
        out = torch.einsum('b i j, b j d -> b i d', att, V) # (batch*heads, num_latents, inner_dim)
        # print('out:', out.shape)
        out = rearrange(out, '(b h) i d -> b i (h d)', h=self.num_heads)
        # print('out:', out.shape)
        out = self.to_out(out)

        return out


class Transformer(nn.Module):
    def __init__(self, latent_dim, data_dim, num_heads=1, head_dim=32, mult = 4, att_dropout = 0.2, ff_dropout = 0.2):
        super(Transformer, self).__init__()

        self.mha = MultiheadAttention(latent_dim=latent_dim, data_dim=data_dim,
                            num_heads=num_heads, head_dim=head_dim, dropout=att_dropout)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
                                nn.Linear(latent_dim, latent_dim * mult),
                                nn.ReLU(),
                                nn.Dropout(ff_dropout),
                                nn.Linear(latent_dim * mult, latent_dim)
                                )
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, latent, data, mask=None):
        out = self.mha(latent, data, mask) + latent
        out = self.norm1(out)
        out = self.mlp(out) + out
        out = self.norm2(out)

        return out
