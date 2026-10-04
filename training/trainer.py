from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from lutils.constants import MAIN_PROCESS
from lutils.dict_wrapper import DictWrapper
from lutils.logger import Logger
from lutils.running_average import RunningMean


class Trainer:
    """Single-process or DDP trainer for flow matching."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        config: DictWrapper,
        dataset: Dataset,
        device: torch.device,
    ):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.device = device
        self.is_main_process = rank == MAIN_PROCESS

        self.sampler = None
        if world_size > 1:
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )

        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config["batching"]["batch_size"],
            shuffle=self.sampler is None,
            sampler=self.sampler,
            num_workers=self.config["batching"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        self.flow_matching_loss = nn.MSELoss(reduction="none")
        self.running_means = RunningMean()

    def _extract_observations(self, batch: torch.Tensor) -> torch.Tensor:
        observations = batch.to(self.device, non_blocking=True)
        num_observations = self.config["num_observations"]
        if observations.size(1) < num_observations:
            raise ValueError(
                f"Batch contains {observations.size(1)} frames but config requires {num_observations}"
            )
        return observations[:, :num_observations]

    def calculate_loss(self, results: DictWrapper[str, Any]) -> DictWrapper[str, Any]:
        per_pixel_mse = self.flow_matching_loss(
            results.reconstructed_vectors,
            results.target_vectors,
        ).mean(dim=1, keepdim=True)

        smoke_mask = results.get("smoke_mask", None)
        smoke_weight = float(self.config.get("smoke_weight", 1.0))
        background_weight = float(self.config.get("background_weight", 1.0))

        if smoke_mask is not None:
            weight_map = smoke_mask * smoke_weight + (1.0 - smoke_mask) * background_weight
            flow_matching_loss = (per_pixel_mse * weight_map).mean()
            smoke_ratio = smoke_mask.mean()
        else:
            flow_matching_loss = per_pixel_mse.mean()
            smoke_ratio = torch.tensor(0.0, device=flow_matching_loss.device)

        return DictWrapper(
            flow_matching_loss=flow_matching_loss,
            smoke_ratio=smoke_ratio,
        )

    def train_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        logger: Logger,
        initial_step: int,
        total_steps: int,
        eval_fn,
        save_fn,
        eval_every: int,
        checkpoint_every: int,
    ) -> int:
        dmodel = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        dmodel.ae.eval()
        for param in dmodel.ae.parameters():
            param.requires_grad = False

        global_step = initial_step
        model.train()

        progress = tqdm(
            total=total_steps,
            initial=initial_step,
            disable=not self.is_main_process,
            desc="Training",
        )

        while global_step < total_steps:
            if self.sampler is not None:
                self.sampler.set_epoch(global_step)

            for batch in self.dataloader:
                observations = self._extract_observations(batch)
                outputs = model(observations)
                loss_output = self.calculate_loss(outputs)
                loss = loss_output.flow_matching_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dmodel.vector_field_regressor.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                global_step += 1
                self.running_means.update(
                    {
                        "flow_matching_loss": float(loss.detach().item()),
                        "smoke_ratio": float(loss_output.smoke_ratio.detach().item()),
                    }
                )

                if self.is_main_process and global_step % 50 == 0:
                    means = self.running_means.get_values()
                    logger.log("Training/Loss/flow_matching", means["flow_matching_loss"])
                    logger.log("Training/Stats/smoke_ratio", means["smoke_ratio"])
                    logger.log("Training/LR", scheduler.get_last_lr()[0])
                    logger.finalize_logs(step=global_step)

                if eval_every > 0 and global_step % eval_every == 0:
                    eval_fn(global_step)
                    model.train()

                if checkpoint_every > 0 and global_step % checkpoint_every == 0:
                    save_fn(global_step)

                progress.update(1)
                if global_step >= total_steps:
                    break

        progress.close()
        return global_step
