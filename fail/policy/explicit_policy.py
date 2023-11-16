import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from fail.model.layers import MLP
from fail.model.modules import PoseForceEncoder
from fail.utils.normalizer import LinearNormalizer
from fail.utils import torch_utils
from fail.policy.base_policy import BasePolicy


class ExplicitPolicy(BasePolicy):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -20
    EPS = 1e-6

    def __init__(self, action_dim, seq_len, z_dim, dropout):
        super().__init__(action_dim, seq_len, z_dim)

        self.normalizer = LinearNormalizer()
        self.encoder = PoseForceEncoder(z_dim, seq_len, dropout)
        self.policy = MLP([z_dim, z_dim // 2, action_dim * 2], act_out=False)

        self.apply(torch_utils.init_weights)

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

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_action(self, obs, goal, device):
        ngoal = self.normalizer["goal"].normalize(goal)
        nobs = self.normalizer["obs"].normalize(np.stack(obs))
        hole_noise = npr.uniform([-0.010, -0.010, 0.0], [0.010, 0.010, 0])

        policy_obs = nobs.unsqueeze(0).flatten(1, 2)
        # policy_obs = torch.concat((ngoal.view(1,1,3).repeat(1,20,1), policy_obs), dim=-1)
        policy_obs[:, :, :3] = ngoal.view(1, 1, 3).repeat(
            1, self.seq_len, 1
        ) - (policy_obs[:, :, :3] + hole_noise)
        policy_obs = policy_obs.to(device)

        with torch.no_grad():
            _, _, policy_action = self.sample(policy_obs)
            policy_action = policy_action.cpu().squeeze().numpy()
        action = self.normalizer["action"].unnormalize(policy_action).cpu()

        return action

    def compute_loss(self, batch):
        # Load batch
        nobs = batch["obs"].float()
        naction = batch["action"].float()
        ngoal = batch["goal"].float()

        B = nobs.shape[0]
        obs = nobs.flatten(1, 2)
        # obs = torch.concat((ngoal[:,0,:].unsqueeze(1).repeat(1,20,1), obs), dim=-1)
        obs[:, :, :3] = (
            ngoal[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
            - obs[:, :, :3]
        )

        mean, log_prob, pred_action = self.sample(obs)
        loss = F.mse_loss(mean, naction[:, -1])

        return loss
