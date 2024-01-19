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
        seed=0,
        val_ratio=0.0,
    ):
        super().__init__(
            path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=seed,
            val_ratio=val_ratio,
        )

    def _sample_to_data(self, sample):
        noise = npr.uniform([-0.010, -0.010, 0.0], [0.010, 0.010, 0])
        data = {
            "robot_state": sample["robot_state"],  # T, D_r
            "world_state": sample["object_state"] + noise,  # 1, D_o
            "action": sample["action"],  # T, D_a
        }
        return data
