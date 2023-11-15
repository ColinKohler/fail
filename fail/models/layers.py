import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn
from escnn import group

class MLP(nn.Module):
  def __init__(self, hiddens):
    super().__init__()

    layers = list()
    for h, h_ in zip(hiddens, hiddens[1:]):
      layers.append(nn.Linear(h, h_))
      layers.append(nn.LeakyReLU(0.01, inplace=True))
    self.mlp = nn.Sequential(*layers)

  def forward(self, x):
    return self.mlp(x)

class CNN(nn.Module):
  def __init__(self, hiddens):
    super().__init__()

    layers = list()
    for h, h_ in zip(hiddens, hiddens[1:]):
      layers.append(nn.Conv2d(h, h_, kernel_size=3, padding=1, stride=2))
      layers.append(nn.ReLU(inplace=True))
    self.cnn = nn.Sequential(*layers)

  def forward(self, x):
    return self.cnn(x)

class SO2MLP(nn.Module):
  def __init__(self):
    super().__init__()

    self.G = group.so2_group()
    self.gspace = gspaces.no_base_space(self.G)
    self.in_type = self.gspace.type(
      self.G.standard_representation() + self.G.irrep(0) + self.G.standard_representation() + self.G.irrep(0) + \
      self.G.standard_representation() + self.G.irrep(0) + self.G.standard_representation() + self.G.irrep(0)
    )

    act1 = enn.FourierELU(
      self.gspace,
      channels=64,
      irreps=self.G.bl_regular_representation(L=3).irreps,
      inplace=True,
      type='regular',
      N=8
    )
    self.block1 = enn.SequentialModule(
      enn.Linear(self.in_type, act1.in_type),
      act1
    )

    act2 = enn.FourierELU(
      self.gspace,
      channels=64,
      irreps=self.G.bl_regular_representation(L=2).irreps,
      inplace=True,
      type='regular',
      N=8
    )
    self.block2 = enn.SequentialModule(
      enn.Linear(self.block1.out_type, act2.in_type),
      act2
    )

    self.out_type = self.gspace.type(self.G.standard_representation() + self.G.irrep(0))
    self.policy = enn.Linear(self.block2.out_type, self.out_type)

  def forward(self, x):
    x = self.in_type(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.policy(x)

    return torch.tanh(x.tensor)

class DihedralMLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.G = gspaces.flipRot2dOnR2(N=8)
    self.in_type = enn.FieldType(self.G, [self.G.irrep(1,1) + self.G.irrep(0,0) + self.G.irrep(1,1) + self.G.irrep(0,0) + self.G.irrep(1,1) + self.G.irrep(0,0) + self.G.irrep(1,1) + self.G.irrep(0,0)])

    out_type = enn.FieldType(self.G, 64 * [self.G.regular_repr])
    self.block1 = enn.SequentialModule(
      enn.R2Conv(self.in_type, out_type, kernel_size=1),
      enn.ReLU(out_type, inplace=True)
    )

    in_type = self.block1.out_type
    out_type = enn.FieldType(self.G, 64 * [self.G.regular_repr])
    self.block2 = enn.SequentialModule(
      enn.R2Conv(in_type, out_type, kernel_size=1),
      enn.ReLU(out_type, inplace=True)
    )

    in_type = self.block2.out_type
    self.out_type = enn.FieldType(self.G, [self.G.irrep(1,1) + self.G.irrep(0,0)])
    self.policy = enn.R2Conv(in_type, self.out_type, kernel_size=1)

  def forward(self, x):
    batch_size = x.size(0)

    x = self.in_type(x.view(batch_size, -1, 1, 1))
    x = self.block1(x)
    x = self.block2(x)
    x = self.policy(x)

    return torch.tanh(x.tensor.squeeze())
