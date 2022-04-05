""" IDEA: Create a 'giving' pytorch-lightning callback that can be used to instrument any model. """
from __future__ import annotations

from torch import Tensor
from typing import Any
from pytorch_lightning import Callback, Trainer, LightningModule
from giving import give
from pytorch_lightning.utilities.types import STEP_OUTPUT
import typing

if typing.TYPE_CHECKING:
    from .model import Model


class GivingCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_batch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        step = trainer.global_step
        give(step)
        return super().on_batch_start(trainer, pl_module)

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | dict[str, Any],
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        # NOTE: need to not report all the stuff, otherwise we'll run out of memory!
        batch_size = batch[0].size(0)
        give(batch_size)
        return super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, unused
        )

