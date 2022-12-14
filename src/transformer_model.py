import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import numpy as np

import src.utils as utils

from src.ode_cell import ODECell
from src.attention import Transformer

torch.autograd.set_detect_anomaly(True)


class TRANSFORMER(nn.Module):
    '''
        Continuous input Transformer.
	'''
    def __init__(self, config, n_labels, input_dim, num_latents, latent_dim, rec_layers, units, nonlinear,
                 cont_in, cont_out, emb_dim=None, device = torch.device("cpu")):

        super(TRANSFORMER, self).__init__()
        
        self.config = config
        self.device = device
        self.n_labels = n_labels
        self.cont_in = cont_in
        self.cont_out = cont_out
        self.emb_dim = emb_dim
        
        # self.mask = True # True for causal masking
        input = input_dim
        if emb_dim is not None:
            self.embed = nn.Linear(input_dim, emb_dim)
            input = emb_dim

        if cont_in:
            self.ode_in = ODE(input, input, rec_layers, units, nonlinear, config.ode_dropout, device)
            
        self.net = Transformer_Multiple(cont_out, num_latents=num_latents, latent_dim=latent_dim, input_channels=input, att_dropout=config.att_dropout, ff_dropout = config.ff_dropout, self_per_cross_attn=config.self_per_cross_attn,
                            latent_heads = config.latent_heads, cross_heads=config.cross_heads, cross_dim_head=config.cross_dim_head, latent_dim_head=config.latent_dim_head)

        self.norm = nn.LayerNorm(input, eps=1e-06)
        self.classifier = nn.Sequential(
                nn.Linear(input, n_labels),
                # nn.ReLU(),
                # # nn.Linear(300, 300),
                # # nn.ReLU(),
                # nn.Linear(300, n_labels)
            )
        self.act = nn.Sigmoid()

    def forward(self, data, time_steps, obj_t, pred_t):
        setting = self.config.setting
        time_steps = time_steps[0]
        pred_t = pred_t[0]
        h = data   # data (batch, seq_len, features)

        if len(time_steps) == len(pred_t):
            pred_t = []

        if self.emb_dim is not None:
            h = self.embed(h) # (batch, seq_len, emb_dim)
        # print(h.shape)

        if self.cont_in:
            h = self.ode_in(h, time_steps, pred_t, setting, self.config.drop) # (batch, seq_len, emb_dim)

        # print(h.shape)
        h = self.net(h) # (batch, seq_len, emb_dim)
        
        h = self.norm(h)
        # print(out.shape)
        h = h[:,-1,:]
        
        h = self.classifier(h)
        h = self.act(h)

        return h


class Transformer_Multiple(nn.Module):
    def __init__(self, cont_out, num_latents, latent_dim, input_channels = 1, cross_heads = 1,
            latent_heads = 1, cross_dim_head = 128, latent_dim_head = 128,
            att_dropout = 0.2, ff_dropout = 0.2, self_per_cross_attn = 1, device=torch.device("cpu")):

        super(Transformer_Multiple, self).__init__()

        self.pos_encoder = PositionalEncoding(input_channels)#, max_len=num_latents)

        self.cont_out = cont_out

        self.self_per_cross_attn = self_per_cross_attn

        self.cross_attn = Transformer(latent_dim=input_channels, data_dim=input_channels,
                            num_heads=cross_heads, head_dim=cross_dim_head, att_dropout = att_dropout, ff_dropout = ff_dropout)

        self.self_attns = nn.ModuleList()
        for i in range(self_per_cross_attn):
            self.self_attns.append(Transformer(latent_dim=input_channels, data_dim=input_channels, num_heads=latent_heads,
                                        head_dim=latent_dim_head, att_dropout = att_dropout, ff_dropout = ff_dropout))

    def forward(self, data, time_steps=None):
        data = self.pos_encoder(data)

        # if self.cont_out:
        # mask = self._causal_mask(data.shape[1]).to(data.device) ##### commenting this for changing #num_latents
        # else:
        mask = None
        b = data.shape[0]

        latent = self.cross_attn(data, data, mask)

        for i in range(self.self_per_cross_attn):
            latent = self.self_attns[i](latent, latent, mask)

        return latent

    def _causal_mask(self, size):
        mask = torch.ones((1, size, size))
        mask = torch.triu(mask, diagonal=1).type(torch.int16)
        # trg_mask = (target == pad).type(torch.int16).unsqueeze(-2)
        # mask = mask | trg_mask

        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=48):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1 * max_len * d_model)
        self.register_buffer('pe', pe)

    def forward(self, X):
        # X is (batch, seq_len, d_model)
        # self.pe[:, :, :X.size(2)] is (1 * max_len * d_model) but is broadcast to B when added
        X = X + self.pe[:, :X.size(1), :]  # batch * seq_len * d_model
        return X  # batch * seq_len * d_model


class ODE(nn.Module):
    def __init__(self, input_dim, latent_dim, rec_layers, units, nonlinear, ode_dropout, device = torch.device("cpu"), setting='Train'):
        super(ODE, self).__init__()
        self.device = device

        self.ode_cell = ODECell(input_dim, latent_dim, rec_layers, units, nonlinear, ode_dropout, device)

    def forward(self, latents, time_steps, time_pred, setting='Train', drop = 0.0):
        # latent (batch, seq_len, latent_dim)
        prev_t, t_i = time_steps[0],  time_steps[1]
        prev_y = latents[:,0,:].unsqueeze(0)

        minimum_step = (time_steps[-1] - time_steps[0]) / 50.0

        assert(not torch.isnan(latents).any())
        assert(not torch.isnan(time_steps).any())

        new_latents = torch.zeros(latents.shape[0], latents.shape[1] + len(time_pred), latents.shape[2]).to(self.device)
        new_latents[:,0,:] = latents[:,0,:]

        j = 1
        for i in range(1, len(time_steps)):
            yi_ode = self.ode_cell(prev_y, prev_t, t_i, minimum_step)

            prev_t = t_i
            t_i = time_steps[i]

            if t_i in time_pred:
                prev_y = yi_ode
            else:
                prev_y = latents[:,j,:].unsqueeze(0)
                j = j + 1

            # at test time, it will use the observed data as such and
            # ode will be used to generate only the missing time steps
            if setting == 'Test':
                new_latents[:,i,:] = prev_y.squeeze(0)
            else:
                new_latents[:,i,:] = yi_ode.squeeze(0)

        assert(not torch.isnan(new_latents).any())

        return new_latents
