""" IDEA: Inherit from Model, but do model parallel training instead of data parallel. """
from __future__ import annotations


from dataclasses import dataclass, field

from fairscale.nn import auto_wrap, checkpoint_wrapper, wrap
from model import BACKBONES, Model
from simple_parsing.helpers import choice, field
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchvision import models


class ModelParallel(Model):
    @dataclass
    class HParams(Model.HParams):
        backbone: type[nn.Module] = choice(
            BACKBONES, default=models.resnet152,
        )
        lr: float = field(default=3e-4)

        batch_size: int = 512
        """ Batch size. """

    def __init__(self, n_classes: int, hp: ModelParallel.HParams | None = None):
        super().__init__(n_classes=n_classes, hp=hp)
        # TODO: Use a large-enough model, with more that 100M parameters.

    def configure_sharded_model(self) -> None:
        # modules are sharded across processes
        # as soon as they are wrapped with ``wrap`` or ``auto_wrap``.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # # Wraps the layer in a Fully Sharded Wrapper automatically
        # linear_layer = wrap(self.linear_layer)

        # Wraps the module recursively
        # based on a minimum number of parameters (default 100M parameters)
        # block = auto_wrap(self.block)

        # For best memory efficiency,
        # add FairScale activation checkpointing
        # final_block = auto_wrap(checkpoint_wrapper(self.final_block))
        # self.model = nn.Sequential(linear_layer, nn.ReLU(), block, final_block)

        self.backbone = auto_wrap(self.backbone)
        self.output = auto_wrap(self.output)
        self.model = nn.Sequential(self.backbone, self.output)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.model.parameters(), lr=self.hp.lr)

