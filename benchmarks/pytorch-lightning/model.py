from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, cast

import torch
from fairscale.nn import auto_wrap
from fairscale.nn.wrap.auto_wrap import ConfigAutoWrap
from pytorch_lightning import LightningModule, Trainer
from simple_parsing import field
from simple_parsing.helpers.serialization.serializable import Serializable
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric
from torchmetrics.classification.accuracy import Accuracy
from torchvision import models

from utils import C, H, W, backbone_choice, get_backbone_network


class Model(LightningModule):
    @dataclass
    class HParams(Serializable):
        backbone: Callable[..., nn.Module] = backbone_choice(default=models.resnet18)
        lr: float = field(default=3e-4)

        batch_size: int = 512
        """ Batch size (in total). Gets divided evenly among the devices when using DP. """

    def __init__(
        self, image_dims: tuple[C, H, W], num_classes: int, hp: HParams | None = None
    ) -> None:
        super().__init__()
        self.hp: Model.HParams = hp or self.HParams()
        in_features, backbone = get_backbone_network(
            network_type=self.hp.backbone, image_dims=image_dims, pretrained=False,
        )
        self.backbone = backbone
        self.output = nn.Linear(in_features, num_classes)
        self.loss = nn.CrossEntropyLoss(reduction="none")

        metrics = nn.ModuleDict({"accuracy": Accuracy(num_classes=num_classes)})
        self.metrics: Mapping[str, Metric] = cast(Mapping[str, Metric], metrics)
        self.save_hyperparameters({"hp": self.hp.to_dict()})

        self.trainer: Trainer
        self._model_are_wrapped = False
        print("Model Hyper-Parameters:", self.hparams)

        self.example_input_array = torch.rand([self.hp.batch_size, *image_dims])

    def configure_sharded_model(self) -> None:
        # NOTE: From https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#fully-sharded-training
        # NOTE: This gets called during train / val / test, so we need to check that we don't wrap
        # the model twice.
        if not self._model_are_wrapped:
            # NOTE: Could probably use any of the cool things from fairscale here, like
            # mixture-of-experts sharding, etc!
            if ConfigAutoWrap.in_autowrap_context:
                print(f"Wrapping models for model-parallel training using fairscale")
                print(f"Trainer state: {self.trainer.state}")
            self.backbone = auto_wrap(self.backbone)
            self.output = auto_wrap(self.output)
            self._model_are_wrapped = True

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.hp.lr)

    def forward(self, x: Tensor) -> Tensor:
        h_x = self.backbone(x)
        logits = self.output(h_x)
        return logits

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        return self.shared_step(batch, batch_idx, phase="val")

    def shared_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, phase: str
    ) -> dict:
        x, y = batch
        y = y.to(self.device)
        logits = self.forward(x)
        loss: Tensor = self.loss(
            logits, y
        )  # partial loss (worker_batch_size, n_classes)
        return {
            "loss": loss,
            "logits": logits,
            "y": y,
        }

    def training_step_end(self, step_output: Tensor | dict[str, Tensor]) -> Tensor:
        loss = self.shared_step_end(step_output, phase="train")
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step_end(self, step_output: Tensor | dict[str, Tensor]) -> Tensor:
        loss = self.shared_step_end(step_output, phase="val")
        self.log("val/loss", loss, on_epoch=True)
        return loss

    def shared_step_end(
        self, step_output: Tensor | dict[str, Tensor], phase: str
    ) -> Tensor:
        assert isinstance(step_output, dict)
        loss = step_output["loss"]  # un-reduced loss (batch_size, n_classes)
        y = step_output["y"]
        logits = step_output["logits"]
        # Log the metrics in `shared_step_end` when they are fused from all workers.
        for name, metric in self.metrics.items():
            metric(logits, y)
            self.log(f"{phase}/{name}", metric)
        return loss.mean()

    # NOTE: Adding these properties in case we are using the auto_find_lr or auto_find_batch_size
    # features of the Trainer, since it modifies these attributes.

    @property
    def batch_size(self) -> int:
        return self.hp.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        print(f"Changing batch size from {self.hp.batch_size} to {value}")
        self.hp.batch_size = value

    @property
    def lr(self) -> float:
        return self.hp.lr

    @lr.setter
    def lr(self, lr: float) -> None:
        print(f"Changing lr from {self.hp.lr} to {lr}")
        self.hp.lr = lr
