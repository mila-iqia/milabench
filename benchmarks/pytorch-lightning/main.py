""" This is the script run by milabench run (by default)

It is possible to use a script from a GitHub repo if it is cloned using
clone_subtree in the benchfile.py, in which case this file can simply
be deleted.
"""
from __future__ import annotations
import contextlib
import inspect

import inspect
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing_extensions import ParamSpec
import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import Trainer
from simple_parsing import ArgumentParser, choice
from simple_parsing.helpers.serialization.serializable import Serializable
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
import pl_bolts.datamodules

# "Relative" imports.
from giving_callback import GivingCallback
from model import Model
from model_parallel import ModelParallel


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


P = ParamSpec("P")


def run(
    model_type: type[Model] = Model,
    data_options_type: type[DataOptions] = DataOptions,
    trainer_type: type[Callable[P, Trainer]] = Trainer,
    *trainer_args: P.args,
    **trainer_defaults: P.kwargs,
):
    """
    Runs a PyTorch-Lightning benchmark, using the command-line arguments.
    
    The benchmark consists of training a model of type `model_type`, using a `Trainer`, on the 
    `LightningDataModule` that is created from the options of type `data_options_type`.
    
    Default values for the keyword arguments of the `Trainer` class can be overridden by
    datamodule to train a model of type `model_type`

    NOTE: trainer_type is just used so we can get nice type-checks for the trainer kwargs. We
    don't actually expect a different subclass of Trainer to be passed.
    """
    # Create a parser so we can extract all the args of the Trainer class, and add it to our own,
    # (better) ArgumentParser.
    trainer_parser = ArgumentParser(add_help=False)
    trainer_parser = Trainer.add_argparse_args(trainer_parser)

    # Copy the arguments from the Trainer parser, and then add our own.
    parser = ArgumentParser(parents=[trainer_parser], add_help=True)

    trainer_parser.set_defaults(enable_checkpointing=False)

    trainer_signature = inspect.signature(Trainer)
    if trainer_defaults:
        # Overwrite the default values with those from the `trainer_defaults` dict.
        sig = trainer_signature.bind_partial(**trainer_defaults)
        print(f"Overwriting default values for the Trainer: {sig.arguments}")
        trainer_parser.set_defaults(**sig.arguments)

    # Add the arguments for the Model:
    parser.add_arguments(model_type.HParams, "hparams")

    # Add arguments for the dataset choice / setup:
    parser.add_arguments(data_options_type, "options")

    args = parser.parse_args()
    args_dict = vars(args)

    # NOTE: Instead of creating the Trainer from the args directly, we instead create it from the
    # kwargs, so we can have better control over some args like `callbacks`, `logger`, etc that
    # accept objects.
    # trainer = Trainer.from_argparse_args(args)

    hparams: Model.HParams = args_dict.pop("hparams")
    options: DataOptions = args_dict.pop("options")

    trainer_kwargs = trainer_defaults.copy()
    trainer_kwargs.update(**args_dict)
    print(f"Trainer kwargs: \n {trainer_kwargs}")
    trainer = trainer_type(*trainer_args, **trainer_kwargs)

    # options: Options = args.options
    # print(f"Options: \n{options.dumps_yaml()}")
    print(f"HParams: \n{hparams.dumps_yaml()}")

    datamodule = options.datamodule(
        str(options.data_dir), num_workers=options.n_workers, pin_memory=True,
    )
    assert hasattr(datamodule, "num_classes")
    n_classes = getattr(datamodule, "num_classes")
    assert isinstance(n_classes, int)

    model = model_type(n_classes=n_classes, hp=hparams)

    # NOTE: Haven't used this new method much yet. Seems to be useful when doing profiling / auto-lr
    # auto batch-size stuff, but those don't work well anyway. Leaving it here for now.
    trainer.tune(model, datamodule=datamodule)

    # Train the model on the provided datamodule.
    trainer.fit(model, datamodule=datamodule)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        trainer.profiler.describe()
    output.seek(0)
    print("---Profiler output:")
    print(output.read())
    print("---End of profiler output")
    validation_results = trainer.validate(model, datamodule=datamodule)
    # TODO: Figure out how to get the profiler output as an object, instead of a string on stdout.
    print(validation_results)


def main():
    # Data-Parallel benchmark:
    # run(
    #     model_type=Model,
    #     gpus=torch.cuda.device_count(),
    #     accelerator="gpu",
    #     strategy="dp",
    #     callbacks=[GivingCallback()],
    #     max_epochs=1,
    #     profiler="simple",
    # )
    # TODO: Extract the profiler output as an object, not a string.
    run(
        model_type=ModelParallel,
        gpus=torch.cuda.device_count(),
        accelerator="gpu",
        strategy="fsdp",
        devices=-1,
        callbacks=[GivingCallback()],
        max_epochs=1,
        # profiler="simple",
    )


if __name__ == "__main__":
    # Note: The line `if __name__ == "__main__"` is necessary for milabench
    # to recognize the entry point (it does some funky stuff to it).
    main()
