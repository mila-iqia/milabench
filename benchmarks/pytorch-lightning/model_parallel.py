""" IDEA: Inherit from Model, but do model parallel training instead of data parallel. """
from __future__ import annotations


from dataclasses import dataclass, field

import torch
from fairscale.nn import auto_wrap, checkpoint_wrapper, wrap
from model import BACKBONES, Model, C, H, W
from simple_parsing.helpers import choice, field
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torchvision import models
import functools


class ModelParallel(Model):
    @dataclass
    class HParams(Model.HParams):
        # TODO: Use a large-enough model, with more that 100M parameters.
        backbone: type[nn.Module] = choice(
            BACKBONES, default=models.vit_l_32,
        )
        lr: float = field(default=3e-4)

        batch_size: int = 512
        """ Batch size. """

    def __init__(
        self,
        image_dims: tuple[C, H, W],
        n_classes: int,
        hp: ModelParallel.HParams | None = None,
    ):
        super().__init__(image_dims=image_dims, n_classes=n_classes, hp=hp)
        self._model_are_wrapped = False

    def configure_sharded_model(self) -> None:
        # NOTE: From https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#fully-sharded-training
        # TODO: Difference between `wrap` and `auto_wrap`?
        if not self._model_are_wrapped:
            self.backbone = auto_wrap(self.backbone)
            self.output = auto_wrap(self.output)
            self._model_are_wrapped = True

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)

    def configure_optimizers(self) -> Optimizer:
        return super().configure_optimizers()
