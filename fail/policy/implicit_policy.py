import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from fail.model.layers import MLP
from fail.utils.normalizer import LinearNormalizer
from fail.utils import torch_utils
from fail.policy.base_policy import BasePolicy
from fail.utils import mcmc


class ImplicitPolicy(BasePolicy):
    def __init__(
        self,
        robot_state_dim: int,
        world_state_dim: int,
        action_dim: int,
        num_action_steps: int,
        num_robot_state: int,
        num_world_state: int,
        num_neg_act_samples: int,
        pred_n_iter: int,
        pred_n_samples: int,
        z_dim: int,
        dropout: float,
        encoder: nn.Module,
    ):
        super().__init__(
            robot_state_dim,
            world_state_dim,
            action_dim,
            num_robot_state,
            num_world_state,
            num_action_steps,
        )
        self.num_neg_act_samples = num_neg_act_samples
        self.pred_n_iter = pred_n_iter
        self.pred_n_samples = pred_n_samples

        self.encoder = encoder
        m_dim = z_dim * 4
        self.energy_mlp = MLP(
            [z_dim + action_dim, m_dim, m_dim, m_dim, 1], dropout=dropout, act_out=False
        )

        self.apply(torch_utils.init_weights)

    def forward(self, robot_state, world_state, action):
        B, N, Ta, Da = action.shape
        z = self.encoder(robot_state, world_state)

        z_a = torch.cat(
            [z.unsqueeze(1).expand(-1, N, -1), action.reshape(B, N, -1)], dim=-1
        )
        z_a.reshape(B * N, -1)

        out = self.energy_mlp(z_a)

        return out.view(B, N)

    def get_action(self, robot_state, world_state, device):
        nrobot_state = self.normalizer["robot_state"].normalize(np.stack(robot_state))
        nworld_state = self.normalizer["world_state"].normalize(world_state)

        B = nrobot_state.size(0)
        Ta = self.num_action_steps
        Tr = self.num_robot_state
        Tw = self.num_world_state

        nrobot_state = nrobot_state.view(B, 20, 9)
        nworld_state = nworld_state.view(B, 2, 3)
        #nrobot_state = nrobot_state.unsqueeze(0).flatten(1, 2)
        #nworld_state = nworld_state.unsqueeze(0).flatten(1,2)
        nrobot_state = nrobot_state.to(device).float()
        nworld_state = nworld_state.to(device).float()

        # Sample actions: (1, num_samples, Da)
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )
        actions = action_dist.sample((B, self.pred_n_samples, Ta)).to(
            dtype=robot_state.dtype
        )

        action_probs, actions = mcmc.iterative_dfo(
            self,
            nrobot_state,
            nworld_state,
            actions,
            [action_stats["min"], action_stats["max"]],
        )

        actions = self.normalizer["action"].unnormalize(actions)
        return {"action": actions}

    def compute_loss(self, batch):
        # Load batch
        nrobot_state = batch["robot_state"].float()
        nworld_state = batch["world_state"].float()
        naction = batch["action"].float()

        Da = self.action_dim
        Tr = self.num_robot_state
        Tw = self.num_world_state
        Ta = self.num_action_steps
        B = nrobot_state.shape[0]

        nrobot_state[:, :Tr]
        nworld_state[:, :Tw]
        start = 1
        end = start + Ta
        naction = naction[:, start:end]

        robot_state = nrobot_state.flatten(1, 2)
        world_state = nworld_state.view(B, 2, -1)

        # Add noise to positive samples
        action_noise = torch.normal(
            mean=0,
            std=1e-4,
            size=naction.shape,
            dtype=naction.dtype,
            device=naction.device,
        )
        noisy_actions = naction + action_noise

        # Sample negatives: (B, train_n_neg, Da)
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats["min"], high=action_stats["max"]
        )
        negatives = action_dist.sample((B, self.num_neg_act_samples, Ta)).to(
            dtype=naction.dtype
        )

        # Combine pos and neg samples: (B, train_n_neg+1, Ta, Da)
        targets = torch.cat([noisy_actions.unsqueeze(1), negatives], dim=1)

        # Randomly permute the positive and negative samples
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:, 1].to(naction.device)

        energy = self.forward(robot_state, world_state, targets)
        loss = F.cross_entropy(energy, ground_truth)

        return loss

    def get_action_stats(self):
        action_stats = self.normalizer["action"].get_output_stats()

        repeated_stats = dict()
        for key, value in action_stats.items():
            n_repeats = self.action_dim // value.shape[0]
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats
