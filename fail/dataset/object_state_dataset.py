import numpy as np
import numpy.random as npr
from fail.dataset.base_dataset import BaseDataset


class ObjectStateDataset(BaseDataset):
    def __init__(
        self,
        path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        harmonic_action: bool = False,
        seed=0,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__(
            path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            harmonic_action=harmonic_action,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes
        )

    def _sample_to_data(self, sample):
        #noise = npr.uniform([-0.010, -0.010, 0.0], [0.010, 0.010, 0])
        data = {
            "robot_state": sample["robot_state"],  # T, D_r
            "world_state": sample["object_state"],  # 1, D_o
            "action": sample["action"],  # T, D_a
        }
        return data
