""" This is the script run by milabench run (by default)

It is possible to use a script from a GitHub repo if it is cloned using
clone_subtree in the benchfile.py, in which case this file can simply
be deleted.
"""
from __future__ import annotations

import pprint
from dataclasses import asdict
from typing import Callable

import torch
import yaml
from giving_callback import GivingCallback
from model import C, H, Model, W
from pytorch_lightning import Trainer
from simple_parsing import ArgumentParser
from typing_extensions import ParamSpec
from utils import DataOptions

P = ParamSpec("P")


def main(
    trainer_type: type[Callable[P, Trainer]] = Trainer,
    *trainer_default_args: P.args,
    **trainer_default_kwargs: P.kwargs,
):
    """
    Runs a PyTorch-Lightning benchmark.

    The benchmark consists of training a model of type `model_type`, using a `Trainer`, on the
    `LightningDataModule` that is created from a `DataOptions` that is also parsed from the
    command-line.

    To create the `Trainer`, the values from `trainer_default_args` and `trainer_default_kwargs`
    are used as the defaults. The values parsed from the command-line then overwrite these values.

    NOTE: trainer_type is just used so we can get nice type-checks for the trainer kwargs. We
    don't actually expect a different subclass of Trainer to be passed.

    Examples:

    Data-Parallel benchmark:
    ```python
    main(
        gpus=torch.cuda.device_count(),
        accelerator="auto",
        strategy="dp",
        callbacks=[GivingCallback()],
        max_epochs=1,
        limit_train_batches=100,
        limit_val_batches=100,
        profiler="simple",
    )
    ```


    Model-Parallel benchmark:
    ```python
    main(
        gpus=torch.cuda.device_count(),
        accelerator="auto",
        strategy="fsdp",
        devices=torch.cuda.device_count(),
        callbacks=[GivingCallback()],
        max_epochs=1,
        limit_train_batches=100,
        limit_val_batches=100,
        profiler="simple",
    )
    ```
    """
    # Create an argument parser.
    parser = ArgumentParser(description=__doc__)

    # Add the arguments for the Model:
    parser.add_arguments(Model.HParams, "hparams")

    # Add arguments for the Trainer of PL:
    Trainer.add_argparse_args(parent_parser=parser, use_argument_group=True)
    if trainer_default_kwargs:
        # NOTE: Uncomment this to turn off checkpointing by default.
        trainer_default_kwargs.setdefault("enable_checkpointing", False)

        # Add the given kwargs as defaults for the parser.
        parser.set_defaults(**trainer_default_kwargs)
        print(f"Overwriting default values for the Trainer: {trainer_default_kwargs}")

    # Add arguments for the dataset choice / setup:
    parser.add_arguments(DataOptions, "options")

    args = parser.parse_args()
    args_dict = vars(args)

    # NOTE: Instead of creating the Trainer from the args directly, we instead create it from the
    # kwargs, so we can have better control over some args like `callbacks`, `logger`, etc that
    # accept objects.
    # trainer = Trainer.from_argparse_args(args)

    hparams: Model.HParams = args_dict.pop("hparams")
    options: DataOptions = args_dict.pop("options")

    print("HParams:")
    _print_indented_yaml(asdict(hparams))
    print("Options:")
    _print_indented_yaml(asdict(options))

    # Rest of `args_dict` is only for the Trainer.
    trainer_kwargs = trainer_default_kwargs.copy()
    trainer_kwargs.update(**args_dict)
    callbacks = trainer_kwargs.setdefault("callbacks", [])
    assert isinstance(callbacks, list)
    if not any(isinstance(c, GivingCallback) for c in callbacks):
        callbacks.append(GivingCallback())

    print(f"Trainer kwargs:")
    pprint.pprint(trainer_kwargs)

    trainer = trainer_type(*trainer_default_args, **trainer_kwargs)

    datamodule = options.datamodule(
        str(options.data_dir),
        num_workers=options.n_workers,
        pin_memory=torch.cuda.is_available(),
        batch_size=hparams.batch_size,
    )
    assert hasattr(datamodule, "num_classes")
    n_classes = getattr(datamodule, "num_classes")
    image_dims = datamodule.dims
    assert isinstance(n_classes, int)
    assert len(image_dims) == 3
    image_dims = (C(image_dims[0]), H(image_dims[1]), W(image_dims[2]))

    model = Model(image_dims=image_dims, num_classes=n_classes, hp=hparams)

    # NOTE: Haven't used this new method much yet. Seems to be useful when doing profiling / auto-lr
    # auto batch-size stuff, but those don't work well anyway. Leaving it here for now.
    trainer.tune(model, datamodule=datamodule)

    # Train the model on the provided datamodule.
    trainer.fit(model, datamodule=datamodule)

    # NOTE: Profiler output is a big string here. We could inspect and report it if needed.
    print("---Profiler output:")
    profiler = trainer.profiler
    profiler_summary = profiler.summary()
    print(profiler_summary)
    print("---End of profiler output")

    # NOTE: Uncomment this to evaluate the model.
    # validation_results = trainer.validate(model, datamodule=datamodule, verbose=False)
    # print(validation_results)


def _print_indented_yaml(stuff):
    import textwrap
    from io import StringIO

    with StringIO() as f:
        yaml.dump(stuff, f)
        f.seek(0)
        print(textwrap.indent(f.read(), prefix="  "))


if __name__ == "__main__":
    # Note: The line `if __name__ == "__main__"` is necessary for milabench
    # to recognize the entry point (it does some funky stuff to it).
    main()
