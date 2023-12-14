import numpy as np
from fail.dataset.base_dataset import BaseDataset


class VisionDataset(BaseDataset):
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
        super().__init__(
            path
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            obs_key=obs_key,
            action_key=action_key,
            seed=seed,
            val_ratio=val_ratio
        )

    def _sample_to_data(self, sample):
        obs = sample[self.obs_key]
        action = sample[self.action_key]
        data = {
            "obs": obs,  # T, D_o
            "vision": sample["vision"],  # T, D_v, W, H
            "action": sample[self.action_key],  # T, D_a
        }
        return data