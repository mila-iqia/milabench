""" This is the script run by milabench run (by default)

It is possible to use a script from a GitHub repo if it is cloned using
clone_subtree in the benchfile.py, in which case this file can simply
be deleted.
"""
from __future__ import annotations
import torch

# "Relative" imports.
from giving_callback import GivingCallback
from model import Model
from model_parallel import ModelParallel
import contextlib
import inspect

import inspect
import io
import typing
from typing import Callable
from typing_extensions import ParamSpec
import torch
from pytorch_lightning import Trainer
from simple_parsing import ArgumentParser
from utils import DataOptions
from model import Model, C, H, W
from model_parallel import ModelParallel


def data_parallel_benchmark():
    """ Data-Parallel benchmark. """
    run(
        model_type=Model,
        gpus=torch.cuda.device_count(),
        accelerator="gpu",
        strategy="dp",
        callbacks=[GivingCallback()],
        max_epochs=1,
        limit_train_batches=100,
        limit_val_batches=100,
        profiler="simple",
    )


def model_parallel_benchmark():
    """ Model-Parallel benchmark. """
    run(
        model_type=ModelParallel,
        gpus=torch.cuda.device_count(),
        accelerator="gpu",
        strategy="fsdp",
        devices=torch.cuda.device_count(),
        callbacks=[GivingCallback()],
        max_epochs=1,
        limit_train_batches=100,
        limit_val_batches=100,
        profiler="simple",
    )


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

    print(f"HParams: \n{hparams.dumps_yaml()}")
    print(f"Options: \n{options.dumps_yaml()}")

    # Rest of `args_dict` is only for the Trainer.
    trainer_kwargs = trainer_defaults.copy()
    trainer_kwargs.update(**args_dict)
    print(f"Trainer kwargs: \n{trainer_kwargs}")
    trainer = trainer_type(*trainer_args, **trainer_kwargs)

    datamodule = options.datamodule(
        str(options.data_dir),
        num_workers=options.n_workers,
        pin_memory=torch.cuda.is_available(),
    )
    assert hasattr(datamodule, "num_classes")
    n_classes = getattr(datamodule, "num_classes")
    image_dims = datamodule.dims
    assert isinstance(n_classes, int)
    assert len(image_dims) == 3
    image_dims = (C(image_dims[0]), H(image_dims[1]), W(image_dims[2]))

    model = model_type(image_dims=image_dims, n_classes=n_classes, hp=hparams)

    # NOTE: Haven't used this new method much yet. Seems to be useful when doing profiling / auto-lr
    # auto batch-size stuff, but those don't work well anyway. Leaving it here for now.
    trainer.tune(model, datamodule=datamodule)

    # Train the model on the provided datamodule.
    trainer.fit(model, datamodule=datamodule)

    # NOTE: Profiler output is a big string here. We could inspect and report it if needed.
    print("---Profiler output:")
    profiler_summary = trainer.profiler.summary()
    print(profiler_summary)
    print("---End of profiler output")

    validation_results = trainer.validate(model, datamodule=datamodule)
    # TODO: Figure out how to get the profiler output as an object, instead of a string on stdout.
    print(validation_results)


def main():
    # TODO: Extract the profiler output as an object, not a string.
    model_parallel_benchmark()


if __name__ == "__main__":
    # Note: The line `if __name__ == "__main__"` is necessary for milabench
    # to recognize the entry point (it does some funky stuff to it).
    main()
