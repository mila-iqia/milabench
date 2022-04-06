""" IDEA: Create a 'giving' pytorch-lightning callback that can be used to instrument any model. """
from __future__ import annotations

from torch import Tensor
from typing import Any, ContextManager
from pytorch_lightning import Callback, Trainer, LightningModule
from giving import give
import typing
import torch
from pytorch_lightning.trainer.states import RunningStage

if typing.TYPE_CHECKING:
    from .model import Model


class GivingCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._ctx: ContextManager

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        device = pl_module.device
        if isinstance(device, str):
            device = torch.device(device)
        use_cuda = device.type == "cuda"

        loader_or_loaders = trainer.request_dataloader(
            RunningStage.TRAINING, model=pl_module
        )
        if isinstance(loader_or_loaders, list):
            raise NotImplementedError(
                f"There are more than one training dataloaders.. dont know which one to give!"
            )
        loader = loader_or_loaders

        give(loader=loader)
        give(use_cuda=use_cuda)
        give(model=pl_module)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        give(compute_start=True)
        x, y = batch
        self._ctx = give.wrap("compute_start", batch=x)
        self._ctx.__enter__()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | dict[str, Any],
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        x, _ = batch
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
