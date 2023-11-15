import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from escnn import gspaces
from escnn import nn as enn
from escnn import group

from fail.model.layers import MLP, CNN
from fail.model.transformer import Transformer


def initWeights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Encoder(nn.Module):
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


class StochasticPolicy(nn.Module):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -20
    EPS = 1e-6

    def __init__(self, action_dim=3, seq_len=20, z_dim=256, dropout=0.1):
        super().__init__()

        self.action_dim = action_dim
        self.encoder = Encoder(z_dim, seq_len, dropout)
        self.policy = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(z_dim // 2, action_dim * 2),
        )

        self.apply(initWeights)

    def forward(self, x):
        batch_size = x.size(0)
        z = self.encoder(x)
        out = self.policy(z)

        mean = out[:, : self.action_dim]
        log_std = out[:, self.action_dim :]

        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - action.pow(2)) + self.EPS)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)

        return action, log_prob, mean


class DeterministicPolicy(nn.Module):
    def __init__(self, action_dim=3, z_dim=256):
        super().__init__()

        self.encoder = Encoder(z_dim)
        self.policy = nn.Linear(z_dim, action_dim)

        self.apply(initWeights)

    def forward(self, x):
        batch_size = x.size(0)
        z = self.encoder(x)
        a = self.policy(z)

        return torch.tanh(a)


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
