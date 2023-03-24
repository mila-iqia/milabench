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
        loader_or_loaders = trainer.train_dataloader.loaders
        if isinstance(loader_or_loaders, list):
            raise NotImplementedError(
                f"There are more than one training dataloaders.. dont know which one to give!"
            )
        loader = loader_or_loaders
        self._start_time = datetime.datetime.now()
        give(loader=loader)
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
        x = get_x_from_batch(batch)
        self._ctx = give.wrap("step", batch=x, task="train")
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
        loss = (
            (outputs["loss"] if isinstance(outputs, dict) else outputs)
            .detach()
            .mean()
            .item()
        )
        give(loss=loss)
        self._ctx.__exit__(None, None, None)
