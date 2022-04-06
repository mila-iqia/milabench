""" Utility functions / classes / stuffs """
from __future__ import annotations

import contextlib
import inspect

import inspect
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar, NewType
from typing_extensions import ParamSpec
import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import Trainer
from simple_parsing import ArgumentParser, choice
from simple_parsing.helpers.serialization.serializable import Serializable
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
import pl_bolts.datamodules
import typing
from torchvision import models
from torch import nn


C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)

# TODO maybe use this so we don't have to download the datasets, or use milatools/milavision.
torchvision_dir: Path | None = Path("/network/datasets/torchvision")
if not torchvision_dir.exists():
    torchvision_dir = None

DEFAULT_DATA_DIR: Path = Path(
    os.environ.get("SLURM_TMPDIR", os.environ.get("SCRATCH", "data"))
)

VISION_DATAMODULES: dict[str, type[VisionDataModule]] = {
    name.replace("DataModule", ""): cls
    for name, cls in vars(pl_bolts.datamodules).items()
    if inspect.isclass(cls) and issubclass(cls, VisionDataModule)
}

BACKBONES: dict[str, type[nn.Module]] = {
    name: cls
    for name, cls in vars(models).items()
    if inspect.isclass(cls) and issubclass(cls, nn.Module)
}
ModuleType = TypeVar("ModuleType", bound=nn.Module)


def get_backbone_network(
    network_type: Callable[..., ModuleType],
    *,
    image_dims: tuple[C, H, W],
    pretrained: bool = False,
) -> tuple[int, ModuleType]:
    """Construct a backbone network using the given image dimensions and network type.
    
    Replaces the last fully-connected layer with a `nn.Identity`.
    """
    # TODO: This doesn't work will all model types:
    # - Some of them need more arguments
    # - Some of them don't have a `fc` attribute.
    backbone_signature = inspect.signature(network_type)
    if (
        "image_size" in backbone_signature.parameters
        or backbone_signature.return_annotation is models.VisionTransformer
    ):
        backbone = network_type(image_size=image_dims[-1], pretrained=pretrained)
    else:
        backbone = network_type(pretrained=pretrained)

    # Replace the output layer with a no-op, we'll create our own instead.
    if hasattr(backbone, "fc"):
        in_features: int = self.backbone.fc.in_features  # type: ignore
        backbone.fc = nn.Identity()
    elif isinstance(backbone, models.VisionTransformer):
        # heads_layers: dict[str, nn.Module] = OrderedDict()
        # if representation_size is None:
        #     heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        # else:
        #     heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
        #     heads_layers["act"] = nn.Tanh()
        #     heads_layers["head"] = nn.Linear(representation_size, num_classes)

        head_layers = backbone.heads
        fc = head_layers.get_submodule("head")
        fc_index = list(head_layers).index(fc)
        assert isinstance(fc, nn.Linear)
        in_features = fc.in_features
        head_layers[fc_index] = nn.Identity()
    else:
        raise NotImplementedError(
            f"TODO: Don't yet know how to remove last layer(s) of networks of type "
            f"{type(backbone)}!\n"
        )

    return in_features, backbone


def recommended_n_workers() -> int:
    """ try to compute a suggested max number of worker based on system's resource s"""
    # NOTE: Not quite sure what this does, was taken from the code of DataLoader class.
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass

    # os.cpu_count() could return Optional[int]
    # get cpu count first and check None in order to satify mypy check
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        return cpu_count
    return torch.multiprocessing.cpu_count()


@dataclass
class DataOptions(Serializable):
    datamodule: type[VisionDataModule] = choice(
        VISION_DATAMODULES, default=CIFAR10DataModule
    )
    """ The datamodule to use. Can be any `VisionDataModule` from `pl_bolts.datamodules`. """

    data_dir: Path = DEFAULT_DATA_DIR
    """ The directory to use for load / download datasets. """

    n_workers: int = 16
    """ number of workers to use for data loading. """

