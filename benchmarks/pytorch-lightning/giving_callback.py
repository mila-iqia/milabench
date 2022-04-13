""" PyTorch-Lightning callback that uses giving.give. Can be used on basically any model. """
from __future__ import annotations

import datetime
from collections import abc as collections_abc
from typing import Any, ContextManager

import torch
from giving import give
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.trainer.states import RunningStage
from torch import Tensor


def get_x_from_batch(batch: Any) -> Tensor:
    """Get the x from a batch."""
    if isinstance(batch, Tensor):
        return batch
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, collections_abc.Mapping) and "x" in batch.keys():
        return batch["x"]
    raise NotImplementedError(
        f"Don't know how to extract 'x' from batch of type: {type(batch)}"
    )


class GivingCallback(Callback):
    """PyTorch-Lightning callback that uses giving.give."""

    def __init__(self) -> None:
        super().__init__()
        self._ctx: ContextManager
        self._start_time: datetime.datetime
        self._end_time: datetime.datetime

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        device = pl_module.device
        if isinstance(device, str):
            device = torch.device(device)
        use_cuda = device.type == "cuda"
        trainer.train_dataloader
        # NOTE: request_dataloader is marked as deprecated in PL 1.6, will be removed in pl 1.8.
        # Don't know what the replacement will be atm. Perhaps just `trainer.train_dataloader`.
        loader_or_loaders = trainer.request_dataloader(
            RunningStage.TRAINING, model=pl_module
        )
        if isinstance(loader_or_loaders, list):
            raise NotImplementedError(
                f"There are more than one training dataloaders.. dont know which one to give!"
            )
        loader = loader_or_loaders
        self._start_time = datetime.datetime.now()
        give(loader=loader)
        give(use_cuda=use_cuda)
        give(model=pl_module)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._end_time = datetime.datetime.now()
        delta = self._end_time - self._start_time
        give(walltime=delta)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        give(compute_start=True)
        x = get_x_from_batch(batch)
        self._ctx = give.wrap("compute_start", batch=x)
        self._ctx.__enter__()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | dict[str, Any],
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        x = get_x_from_batch(batch)
        x_batch = x.detach() if x.requires_grad else x
        step = trainer.global_step
        loss = (
            (outputs["loss"] if isinstance(outputs, dict) else outputs)
            .detach()
            .mean()
            .item()
        )
        give(step=step, batch=x_batch)
        give(loss=loss)
        self._ctx.__exit__(None, None, None)
