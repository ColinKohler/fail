import copy
from typing import Dict
import torch
import numpy as np
import pickle

from fail.dataset.replay_buffer import ReplayBuffer
from fail.utils import torch_utils
from fail.utils.normalizer import LinearNormalizer
from fail.utils.sampler import SequenceSampler, get_val_mask


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_key="state",
        action_key="action",
        seed=0,
        val_ratio=0.0,
    ):
        super().__init__()

        self.replay_buffer = ReplayBuffer()
        self.replay_buffer.load_from_path(path)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        self.train_mask = ~val_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=self.train_mask,
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.obs_key = obs_key
        self.action_key = action_key

        self.normalizer = self.get_normalizer()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = sample[self.obs_key]
        data = {
            "obs": obs,  # T, D_o
            "goal": sample["goal"],  # 1, D_g
            "action": sample[self.action_key],  # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = torch_utils.dict_apply(data, torch.from_numpy)
        return self.normalizer.normalize(torch_data)
