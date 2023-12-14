import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn
from escnn import group

from fail.model.layers import SO2MLP, MLP, ResNet
#from fail.model.transformer import Transformer
from fail.model.general_transformer import Transformer
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


class RobotStateObjectPoseEncoder(nn.Module):
    def __init__(
            self,
            robot_state_dim: int,
            object_state_dim: int,
            model_dim: int=256,
            trans_out_dim: int=32,
            z_dim: int=64,
            seq_len: int=20,
            dropout: float=0.1
    ):
        super().__init__()
        self.robot_state_embedding = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(robot_state_dim, model_dim)
        )
        self.object_state_embedding = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(object_state_dim, model_dim)
        )
        self.transformer = Transformer(
            model_dim=model_dim,
            out_dim=trans_out_dim,
            num_heads=8,
            num_layers=4,
            dropout=dropout
        )
        self.out = nn.Linear(seq_len * trans_out_dim, z_dim)

    def forward(self, robot_state, object_state) -> torch.Tensor:
        batch_size = robot_state.size(0)
        robot_embed = self.robot_state_embedding(robot_state)
        object_embed = self.object_state_embedding(object_state)

        x = self.transformer(robot_embed, key=object_embed)
        return self.out(x.view(batch_size, -1))


class RobotStateVisionEncoder(nn.Module):
    def __init__(
            self,
            robot_state_dim: int,
            vision_dim: int,
            model_dim: int=256,
            trans_out_dim: int=32,
            z_dim: int=64,
            seq_len: int=20,
            dropout: float=0.1
    ):
        super().__init__()
        self.robot_state_embedding = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(robot_state_dim, model_dim)
        )
        self.vision_embedding = nn.Sequential(
            nn.Dropout(dropout), ResNet([vision_dim, 8, 16, 32, 64, model_dim])
        )
        self.transformer = Transformer(
            model_dim=model_dim,
            out_dim=trans_out_dim,
            num_heads=8,
            num_layers=4,
            dropout=dropout
        )
        self.out = nn.Linear(seq_len * trans_out_dim, z_dim)

    def forward(self, robot_state: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        batch_size = robot_state.size(0)
        robot_embed = self.robot_state_embedding(robot_state)
        vision_embed = self.vision_embedding(vision).view(batch_size, -1)

        x = self.transformer(robot_embed, key=vision_embed)
        return self.out(x.view(batch_size, -1))


class SO2PoseForceEncoder(nn.Module):
    def __init__(self, in_type, L, z_dim=64, seq_len=20, dropout=0.1):
        super().__init__()
        trans_out_dim = 16

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.L = L

        t = self.G.bl_regular_representation(L=self.L)
        self.trans_type = enn.FieldType(self.gspace, [t] * trans_out_dim * seq_len)

        self.in_type = in_type

        self.transformer = SO2Transformer(
            in_type=in_type,
            L=L,
            model_dim=64,
            out_dim=trans_out_dim,
            num_heads=8,
            num_layers=4,
            dropout=dropout,
            input_dropout=dropout,
        )
        self.out_type = enn.FieldType(self.gspace, [t] * z_dim)
        self.out = SO2MLP(self.trans_type, self.out_type, [z_dim], [self.L], act_out=False)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.transformer(x)
        x = self.trans_type(x.tensor.view(batch_size, -1))
        return self.out(x)
