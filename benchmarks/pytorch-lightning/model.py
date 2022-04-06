from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from turtle import back
from typing import Mapping
from typing import cast

import torch
from torch import Tensor
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningModule, Trainer
from simple_parsing import Serializable, choice, field
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from torch import Tensor, nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
)
from torchvision import models
import pl_bolts.datamodules
from giving_callback import GivingCallback
from typing import NewType
from utils import BACKBONES, C, H, W, get_backbone_network


class Model(LightningModule):
    @dataclass
    class HParams(Serializable):
        backbone: type[nn.Module] = choice(
            BACKBONES, default=models.resnet18,
        )
        lr: float = field(default=3e-4)

        batch_size: int = 512
        """ Batch size (in total). Gets divided evenly among the devices when using DP. """

    def __init__(
        self, image_dims: tuple[C, H, W], n_classes: int, hp: HParams | None = None
    ) -> None:
        super().__init__()
        self.hp: Model.HParams = hp or self.HParams()
        in_features, backbone = get_backbone_network(
            image_dims=image_dims, network_type=self.hp.backbone, pretrained=False,
        )
        self.backbone = backbone
        self.output = nn.Linear(in_features, n_classes)
        self.loss = nn.CrossEntropyLoss(reduction="none")

        metrics = nn.ModuleDict(
            {
                metric_class.__name__.lower(): metric_class(num_classes=n_classes)
                for metric_class in [
                    Accuracy,
                    # NOTE: These seem to not be working that well (giving the same exact value?)
                    # Precision,
                    # Recall,
                    # F1Score,
                    # ConfusionMatrix,
                ]
            }
        )
        self.metrics: Mapping[str, Metric] = cast(Mapping[str, Metric], metrics)
        self.save_hyperparameters({"hp": self.hp.to_dict()})

        self.trainer: Trainer

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.hp.lr)

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
