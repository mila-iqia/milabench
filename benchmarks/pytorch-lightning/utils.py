""" Utility functions / classes / stuffs """
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NewType, TypeVar

import pl_bolts.datamodules
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from simple_parsing import choice
from simple_parsing.helpers.serialization.serializable import Serializable
from torch import nn
from torchvision import models

C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)
ModuleType = TypeVar("ModuleType", bound=nn.Module)

DEFAULT_DATA_DIR: Path = Path(os.environ["MILABENCH_DIR_DATA"])

VISION_DATAMODULES: dict[str, type[VisionDataModule]] = {
    name.replace("DataModule", ""): cls
    for name, cls in vars(pl_bolts.datamodules).items()
    if inspect.isclass(cls) and issubclass(cls, VisionDataModule)
}

BACKBONES: list[str] = models.list_models(module=models)


def backbone_choice(default: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
    """
    A field that will automatically download the torchvision model if it's not already downloaded.
    """
    return choice(
        BACKBONES,
        default=default,
    )


def get_backbone_network(
    network_type: str,
    *,
    image_dims: tuple[C, H, W],
    pretrained: bool = False,
) -> tuple[int, ModuleType]:
    """Construct a backbone network using the given image dimensions and network type.

    Replaces the last fully-connected layer with a `nn.Identity`.

    TODO: Add support for more types of models.
    """

    try:
        backbone = models.get_model(network_type, image_size=image_dims[-1], weights="DEFAULT" if pretrained else None)
    except TypeError:
        # For non-vision models
        backbone = models.get_model(network_type, weights="DEFAULT" if pretrained else None)

    # Replace the output layer with a no-op, we'll create our own instead.
    if hasattr(backbone, "fc"):
        in_features: int = backbone.fc.in_features  # type: ignore
        backbone.fc = nn.Identity()
    elif isinstance(backbone, models.VisionTransformer):
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
