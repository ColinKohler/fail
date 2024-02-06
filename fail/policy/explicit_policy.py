import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from fail.model.layers import MLP
from fail.model.modules import PoseForceEncoder
from fail.utils import torch_utils
from fail.policy.base_policy import BasePolicy


class ExplicitPolicy(BasePolicy):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -20
    EPS = 1e-6

    def __init__(self, robot_state_dim, world_state_dim, action_dim, num_robot_state, num_world_state, num_action_steps, encoder):
        super().__init__(robot_state_dim, world_state_dim, action_dim, num_robot_state, num_world_state, num_action_steps)

        self.encoder = encoder
        z_dim = self.encoder.z_dim
        self.policy = MLP([z_dim, z_dim // 2, action_dim * 2], act_out=False)

        self.apply(torch_utils.init_weights)

    def forward(self, robot_state, world_state):
        z = self.encoder(robot_state, world_state)
        out = self.policy(z)

        mean = out[:, : self.action_dim]
        log_std = out[:, self.action_dim :]

        return mean, log_std

    def sample(self, robot_state, world_state):
        mean, log_std = self.forward(robot_state, world_state)
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

    def get_action(self, obs, device):
        B = obs['robot_state'].shape[0]

        nrobot_state = self.normalizer["robot_state"].normalize(obs['robot_state'])
        nworld_state = self.normalizer["world_state"].normalize(obs['world_state'])

        Dr = self.robot_state_dim
        Dw = self.world_state_dim
        Tr = self.num_robot_state
        Tw = self.num_world_state
        Ta = self.num_action_steps

        with torch.no_grad():
            _, _, action = self.sample(nrobot_state, nworld_state)
        actions = self.normalizer["action"].unnormalize(actions)

        return {'action': actions}

    def compute_loss(self, batch):
        # Load batch
        nrobot_state = batch["robot_state"].float()
        nworld_state = batch['world_state'].float()
        naction = batch["action"].float()

        Dr = self.robot_state_dim
        Dw = self.world_state_dim
        Tr = self.num_robot_state
        Tw = self.num_world_state
        Ta = self.num_action_steps
        B = naction.shape[0]

        mean, log_prob, _ = self.sample(nrobot_state, nworld_state)
        loss = F.mse_loss(mean, naction)

        return loss

    def get_action_stats(self):
        action_stats = self.normalizer['action'].get_output_stats()

        repeated_stats = dict()
        for key, valye in action_stas.items():
            n_repeats = self.action_dim // value.shape[0]
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats
