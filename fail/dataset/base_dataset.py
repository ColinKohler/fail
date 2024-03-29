import copy
from typing import Dict
import torch
import numpy as np
import pickle

from fail.dataset.replay_buffer import ReplayBuffer
from fail.utils import torch_utils
from fail.utils.normalizer import LinearNormalizer
from fail.utils.sampler import SequenceSampler, get_val_mask, downsample_mask


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        harmonic_action: bool = False,
        seed: int = 0,
        val_ratio: float = 0.0,
        max_train_episodes: int = None,
    ):
        super().__init__()

        self.replay_buffer = ReplayBuffer()
        self.replay_buffer.load_from_path(path)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        self.train_mask = ~val_mask
        self.train_mask = downsample_mask(
            mask=self.train_mask, max_n=max_train_episodes, seed=seed
        )

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
        self.harmonic_action = harmonic_action

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
        if self.harmonic_action:
            data["action"] = harmonics.convert_action_to_harmonics(data["action"])
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        if self.harmonic_action:
            data["action"] = harmonics.convert_action_to_harmonics(data["action"])

        torch_data = torch_utils.dict_apply(data, torch.from_numpy)
        return self.normalizer.normalize(torch_data)

    def _sample_to_data(self, sample):
        raise NotImplementedError()
