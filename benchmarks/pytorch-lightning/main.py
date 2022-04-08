""" This is the script run by milabench run (by default)

It is possible to use a script from a GitHub repo if it is cloned using
clone_subtree in the benchfile.py, in which case this file can simply
be deleted.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import pprint
import typing
from typing import Callable

import torch

# "Relative" imports.
from giving_callback import GivingCallback
from model import C, H, Model, W
from pytorch_lightning import Callback, LightningModule, Trainer
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
    """
    # Create a parser so we can extract all the args of the Trainer class, and add it to our own,
    # (better) ArgumentParser.
    trainer_parser = ArgumentParser(add_help=False)
    trainer_parser = Trainer.add_argparse_args(trainer_parser)

    # Copy the arguments from the Trainer parser, and then add our own.
    parser = ArgumentParser(parents=[trainer_parser], add_help=True)

    trainer_parser.set_defaults(enable_checkpointing=False)

    trainer_signature = inspect.signature(Trainer)
    if trainer_default_kwargs:
        # Overwrite the default values with those from the `trainer_defaults` dict.
        sig = trainer_signature.bind_partial(**trainer_default_kwargs)
        print(f"Overwriting default values for the Trainer: {sig.arguments}")
        trainer_parser.set_defaults(**sig.arguments)

    # Add the arguments for the Model:
    parser.add_arguments(Model.HParams, "hparams")

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

    print(f"HParams: \n{hparams.dumps_yaml()}")
    print(f"Options: \n{options.dumps_yaml()}")

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

    # TODO: Is is relevant to evaluate the model?
    # BUG: Sometimes the validation loop doesn't end properly!
    # NOTE: torchscript doesn't seem to be compatible with fully shareded data parallel.
    # eval_model = model.to_torchscript()
    # assert isinstance(eval_model, LightningModule)
    # validation_kwargs = trainer_kwargs.copy()
    # validation_kwargs.update(devices=1, strategy="dp", accelerator="gpu")
    # validation_trainer = Trainer(**validation_kwargs)
    # validation_results = validation_trainer.validate(
    #     model, datamodule=datamodule, verbose=False
    # )
    # print(validation_results)


# def data_parallel_benchmark():
#     """ Data-Parallel benchmark. """
#     # TODO: Maybe we should just pass everything from the command-line instead of this hybrid?
#     main(
#         gpus=torch.cuda.device_count(),
#         accelerator="gpu",
#         strategy="dp",
#         callbacks=[GivingCallback()],
#         max_epochs=1,
#         limit_train_batches=100,
#         limit_val_batches=100,
#         profiler="simple",
#     )


# def model_parallel_benchmark():
#     """ Model-Parallel benchmark. """
#     main(
#         gpus=torch.cuda.device_count(),
#         accelerator="gpu",
#         strategy="fsdp",
#         devices=torch.cuda.device_count(),
#         callbacks=[GivingCallback()],
#         max_epochs=1,
#         limit_train_batches=100,
#         limit_val_batches=100,
#         profiler="simple",
#     )


if __name__ == "__main__":
    # Note: The line `if __name__ == "__main__"` is necessary for milabench
    # to recognize the entry point (it does some funky stuff to it).
    main()
