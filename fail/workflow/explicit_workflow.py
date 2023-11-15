import pathlib
from tqdm import tqdm
import hydra
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from typing import Optional
from omegaconf import OmegaConf

from fail.dataset.base_dataset import BaseDataset
from fail.workflow.base_workflow import BaseWorkflow
from fail.model.policy import StochasticPolicy
from fail.utils import torch_utils

OmegaConf.register_new_resolver("eval", eval, replace=True)


class ExplicitWorkflow(BaseWorkflow):
    def __init__(self, config: OmegaConf, output_dir: Optional[str] = None):
        super().__init__(config, output_dir=output_dir)

        # Set random seed
        seed = config.training.seed
        torch.manual_seed(seed)
        npr.seed(seed)
        random.seed(seed)

        self.model: StochasticPolicy
        self.model = hydra.utils.instantiate(config.policy)
        self.optimizer = hydra.utils.instantiate(
            config.optimizer, params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0

    def run(self):
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(self.config.task.dataset)
        train_dataloader = DataLoader(dataset, **self.config.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **self.config.val_dataloader)

        #self.model.set_normalizer(normalizer)

        lr_scheduler = torch_utils.CosineWarmupScheduler(
            self.optimizer,
            self.config.training.lr_warmup_steps,
            len(train_dataloader) * self.config.training.num_epochs,
        )
        criterion = nn.MSELoss()

        device = torch.device(self.config.training.device)
        self.model.to(device)

        for epoch in range(self.config.training.num_epochs):
            train_losses = list()
            self.model.train()
            with tqdm(
                train_dataloader,
                desc=f"Training Epoch {self.epoch}",
                leave=False,
                mininterval=self.config.training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # Load batch
                    nobs, naction = batch["obs"].float().to(device), batch[
                        "action"
                    ].float().to(device)
                    ngoal = batch["goal"].float().to(device)
                    B = nobs.shape[0]
                    obs = nobs.flatten(1, 2)
                    # obs = torch.concat((ngoal[:,0,:].unsqueeze(1).repeat(1,20,1), obs), dim=-1)
                    obs[:, :, :3] = (
                        ngoal[:, 0, :].unsqueeze(1).repeat(1, self.config.policy.seq_len, 1)
                        - obs[:, :, :3]
                    )

                    # Compute loss
                    mean, log_prob, pred_action = self.model.sample(obs)
                    loss = criterion(mean, naction[:, -1])
                    loss.backward()

                    # Optimization
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()

                    # Logging
                    loss_cpu = loss.item()
                    tepoch.set_postfix(loss=loss_cpu, refresh=False)
                    train_losses.append(loss_cpu)
                tepoch.set_postfix(loss=np.mean(train_losses))

            train_loss = np.mean(train_losses)

            # Validation
            self.model.eval()
            if self.epoch % self.config.training.val_every == 0:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm(
                        val_dataloader,
                        desc=f"Validation epoch {self.epoch}",
                        leave=False,
                        mininterval=self.config.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            # Load batch
                            nobs, naction = batch["obs"].float().to(device), batch[
                                "action"
                            ].float().to(device)
                            ngoal = batch["goal"].float().to(device)
                            B = nobs.shape[0]
                            obs = nobs.flatten(1, 2)
                            # obs = torch.concat((ngoal[:,0,:].unsqueeze(1).repeat(1,20,1), obs), dim=-1)
                            obs[:, :, :3] = (
                                ngoal[:, 0, :].unsqueeze(1).repeat(1, self.config.policy.seq_len, 1)
                                - obs[:, :, :3]
                            )

                            # Compute loss
                            mean, log_prob, pred_action = self.model.sample(obs)
                            loss = criterion(mean, naction[:, -1]).item()
                            val_losses.append(loss)

            # Bookkeeping
            self.global_step += 1
            self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem
)
def main(config):
    workflow = ExplicitWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main()
