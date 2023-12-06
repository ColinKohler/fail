import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn
from escnn import group

from fail.model.layers import MLP, CNN
from fail.model.transformer import Transformer
from fail.model.so2_transformer import SO2Transformer
from fail.utils.normalizer import LinearNormalizer


class PoseForceEncoder(nn.Module):
    def __init__(self, z_dim=64, seq_len=20, dropout=0.1):
        super().__init__()
        trans_out_dim = 32
        self.transformer = Transformer(
            input_dim=9,
            model_dim=256,
            out_dim=trans_out_dim,
            num_heads=8,
            num_layers=4,
            dropout=dropout,
            input_dropout=dropout,
        )
        self.out = nn.Linear(seq_len * trans_out_dim, z_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.transformer(x)
        return self.out(x.view(batch_size, -1))


class SO2PoseForceEncoder(nn.Module):
    def __init__(self, z_dim=64, seq_len=20, dropout=0.1):
        super().__init__()
        trans_out_dim = 32
        self.transformer = Transformer(
            input_dim=9,
            model_dim=256,
            out_dim=trans_out_dim,
            num_heads=8,
            num_layers=4,
            dropout=dropout,
            input_dropout=dropout,
        )
        self.out = nn.Linear(seq_len * trans_out_dim, z_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.transformer(x)
        return self.out(x.view(batch_size, -1))


class EnergyPolicy(nn.Module):
    def __init__(self, action_dim=3, z_dim=256):
        super().__init__()

        self.encoder = Encoder(z_dim)
        self.mlp = nn.Linear(z_dim + action_dim, 1)

    def forward(self, x, y):
        batch_size = x.size(0)
        z = self.encoder(x)

        z_y = torch.cat([z, y], dim=-1)
        out = self.policy(z_y)

        return out
