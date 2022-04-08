""" Utility functions / classes / stuffs """
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NewType, TypeVar

import pl_bolts.datamodules
import torch
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


BACKBONES: dict[str, type[nn.Module]] = {
    name: cls_or_fn
    for name, cls_or_fn in vars(models).items()
    if (callable(cls_or_fn) and "pretrained" in inspect.signature(cls_or_fn).parameters)
}


def backbone_choice(default: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
    """
    A field that will automatically download the torchvision model if it's not already downloaded.
    """
    return choice(BACKBONES, default=default,)


def get_backbone_network(
    network_type: Callable[..., ModuleType],
    *,
    image_dims: tuple[C, H, W],
    pretrained: bool = False,
) -> tuple[int, ModuleType]:
    """Construct a backbone network using the given image dimensions and network type.

    Replaces the last fully-connected layer with a `nn.Identity`.

    TODO: Add support for more types of models.
    """

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
        in_features: int = backbone.fc.in_features  # type: ignore
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
