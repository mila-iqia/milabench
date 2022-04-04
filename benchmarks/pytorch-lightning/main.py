""" This is the script run by milabench run (by default)

It is possible to use a script from a GitHub repo if it is cloned using
clone_subtree in the benchfile.py, in which case this file can simply
be deleted.
"""
from __future__ import annotations

import inspect
import os
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeVar

# import pl_bolts.datamodules
import torch
from torch import Tensor
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from simple_parsing import ArgumentParser, Serializable, choice, field
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

# TODO maybe use this so we don't have to download the datasets, or use milatools/milavision.
torchvision_dir: Path | None = Path("/network/datasets/torchvision")
if not torchvision_dir.exists():
    torchvision_dir = None

DEFAULT_DATA_DIR: Path = Path(
    os.environ.get("SLURM_TMPDIR", os.environ.get("SCRATCH", "data"))
)

VISION_DATAMODULES: dict[str, type[VisionDataModule]] = {
    # "cifar10": CIFAR10DataModule,
    name.replace("DataModule", ""): cls
    for name, cls in vars(pl_bolts.datamodules).items()
    if inspect.isclass(cls) and issubclass(cls, VisionDataModule)
}

BACKBONES: dict[str, type[nn.Module]] = {
    name: cls
    for name, cls in vars(models).items()
    if inspect.isclass(cls) and issubclass(cls, nn.Module)
}


class MyModel(LightningModule):
    @dataclass
    class HParams(Serializable):
        n_classes: int = field(default=10)
        backbone: type[nn.Module] = choice(
            BACKBONES,
            default=models.resnet18,
            encoding_fn=lambda x: x.__name__,
            decoding_fn=BACKBONES.get,
        )
        lr: float = field(default=3e-4)

    def __init__(self, n_classes: int, hp: HParams | None = None) -> None:
        super().__init__()
        self.hp: MyModel.HParams = hp or self.HParams()
        self.backbone = self.hp.backbone()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.output = nn.Linear(in_features, n_classes)
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.metrics: Mapping[str, Metric] = nn.ModuleDict(
            {
                metric_class.__name__.lower(): metric_class(num_classes=n_classes)
                for metric_class in [
                    Accuracy,
                    Precision,
                    Recall,
                    F1Score,
                    # ConfusionMatrix,
                ]
            }
        )
        self.save_hyperparameters({"hp": self.hp.to_dict()})

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
        loss: Tensor = self.loss(logits, y)  # partial loss (batch_size, n_classes)
        return {
            "loss": loss,
            "logits": logits,
            "y": y,
        }

    def training_step_end(self, step_output: Tensor | dict[str, Tensor]) -> Tensor:
        return self.shared_step_end(step_output, phase="train")

    def validation_step_end(self, step_output: Tensor | dict[str, Tensor]) -> Tensor:
        return self.shared_step_end(step_output, phase="val")

    def shared_step_end(
        self, step_output: Tensor | dict[str, Tensor], phase: str
    ) -> Tensor:
        assert isinstance(step_output, dict)
        loss = step_output["loss"]  # un-reduced loss (batch_size, n_classes)
        y = step_output["y"]
        logits = step_output["logits"]
        for name, metric in self.metrics.items():
            metric(logits, y)
            self.log(f"{phase}/{name}", metric, on_epoch=True, prog_bar=True)
        return loss.mean()


@dataclass
class Options(Serializable):
    datamodule: type[VisionDataModule] = choice(
        VISION_DATAMODULES, default=CIFAR10DataModule
    )
    """ The datamodule to use. Can be any `VisionDataModule` from `pl_bolts.datamodules`. """

    data_dir: Path = DEFAULT_DATA_DIR
    """ The directory to use for load / download datasets. """

    train_epochs: int = 10
    """ Number of training epochs to perform. """

    batch_size: int = 256
    """ Batch size (per GPU). """

    n_gpus: int = torch.cuda.device_count()
    """ Number of GPUs to use. """

    n_workers: int = 4
    """ number of workers to use for data loading. """

    accelerator: str = choice("cpu", "gpu", "tpu", "hpu", default="gpu")
    """ Accelerator to use. """

    enable_checkpointing: bool = False
    """ Wether to enable checkpointing or not. """


def main():
    # os.environ["MILABENCH_DIR_DATA"]
    parser = ArgumentParser()
    parser.add_arguments(Options, "options")
    parser.add_arguments(MyModel.HParams, "hparams")

    args = parser.parse_args()
    options: Options = args.options
    hparams: MyModel.HParams = args.hparams

    print(f"Options: \n{options.dumps_yaml()}")
    print(f"HParams: \n{hparams.dumps_yaml()}")

    datamodule = options.datamodule(
        str(options.data_dir),
        num_workers=options.n_workers,
        batch_size=options.batch_size,
    )

    trainer = Trainer(
        accelerator="gpu",
        strategy="dp",
        max_epochs=options.train_epochs,
        devices=options.n_gpus,
        enable_checkpointing=options.enable_checkpointing,
        callbacks=[RichProgressBar()],
    )
    model = MyModel(n_classes=datamodule.num_classes, hp=hparams)
    trainer.fit(model, datamodule=datamodule)

    validation_results = trainer.validate(model, datamodule=datamodule)
    print(validation_results)


if __name__ == "__main__":
    # Note: The line `if __name__ == "__main__"` is necessary for milabench
    # to recognize the entry point (it does some funky stuff to it).
    main()
