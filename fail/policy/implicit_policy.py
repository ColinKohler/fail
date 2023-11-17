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


class ImplicitPolicy(BasePolicy):
    def __init__(self, action_dim, num_neg_act_samples, seq_len, z_dim, dropout):
        super().__init__(action_dim, seq_len, z_dim)
        self.num_neg_act_samples = num_neg_act_samples

        self.encoder = PoseForceEncoder(z_dim, seq_len, dropout)
        self.energy_mlp = MLP([z_dim+action_dim, z_dim // 2, 1], act_out=False)

        self.apply(torch_utils.init_weights)

    def forward(self, x, a):
        batch_size = x.size(0)
        z = self.encoder(x)

        z_a =  torch.cat([z.unsqueeze(1).expand(-1, a.size(1), -1), a], dim=-1)
        B, N, D = z_a.shape
        z_a.reshape(B * N, D)

        out = self.energy_mlp(z_a)

        return out.view(B, N)

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

        # Sample actions: (1, num_samples, Da)
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats['min'],
            high=action_stats['max']
        )
        self.num_neg_act_samples=1024
        samples = action_dist.sample((1, self.num_neg_act_samples)).to(dtype=policy_obs.dtype)

        zero = torch.tensor(0, device=device)
        resample_std = torch.tensor(3e-2, device=device)
        self.pred_n_iter = 5
        for i in range(self.pred_n_iter):
            logits = self.forward(policy_obs, samples)
            prob = torch.softmax(logits, dim=-1)

            if i < (self.pred_n_iter - 1):
                idxs = torch.multinomial(prob, self.num_neg_act_samples, replacement=True)
                samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]
                samples += torch.normal(zero, resample_std, size=samples.shape, device=device)

            idxs = torch.multinomial(prob, num_samples=1, replacement=True)
            acts_n = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs].squeeze(1)

        action = self.normalizer["action"].unnormalize(acts_n).cpu().squeeze()

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

        # Add noise to positive samples
        batch_size = naction.size(0)
        action_noise = torch.normal(mean=0, std=1e-4, size=naction[:,-1].shape, dtype=naction.dtype, device=naction.device)
        noisy_actions = naction[:,-1] + action_noise

        # Sample negatives: (B, train_n_neg, Da)
        action_stats = self.get_action_stats()
        action_dist = torch.distributions.Uniform(
            low=action_stats['min'],
            high=action_stats['max']
        )
        negatives = action_dist.sample((batch_size, self.num_neg_act_samples)).to(dtype=naction.dtype)

        # Combine pos and neg samples: (B, train_n_neg+1, Da)
        targets = torch.cat([noisy_actions.unsqueeze(1), negatives], dim=1)

        # Randomly permute the positive and negative samples
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        ground_truth = (permutation == 0).nonzero()[:,1].to(naction.device)

        energy = self.forward(obs, targets)
        logits = -1.0 * energy
        loss = F.cross_entropy(logits, ground_truth)

        return loss

    def get_action_stats(self):
        action_stats = self.normalizer['action'].get_output_stats()

        repeated_stats = dict()
        for key, value in action_stats.items():
            n_repeats = self.action_dim // value.shape[0]
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats
