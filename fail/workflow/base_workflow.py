from typing import Optional
import os
import pathlib
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


class BaseWorkflow(object):
    def __init__(self, config: OmegaConf, output_dir: Optional[str] = None):
        super().__init__()

        self.config = config
        self._output_dir = output_dir

    @property
    def output_dir(self):
        output_dir = self._outpur_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def run(self):
        pass
