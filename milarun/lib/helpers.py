
from types import SimpleNamespace as NS
from coleo import Argument, auto_cli, default, tooled
import pkg_resources
import torch
import numpy as np


def resolve(spec):
    entry = pkg_resources.EntryPoint.parse(f"__={spec}")
    return entry.resolve()


def cycle(it):
    while True:
        yield from it


# def dataloop(it, wrapper):
#     for data in cycle(it):
#         with wrapper() as w:
#             yield w, data
#         if wrapper.done():
#             break


def dataloop(it, wrapper):
    iterator = iter(cycle(it))
    while True:
        try:
            with wrapper() as w:
                data = next(iterator)
                yield w, data
        except StopIteration:
            break
        if wrapper.done():
            break


def coleo_main(fn):
    entry_point = tooled(fn)
    def main(experiment, argv):
        args, thunk = auto_cli(fn, [experiment], argv=argv, return_split=True)
        argdict = {k: v for k, v in vars(args).items() if not k.startswith("#")}
        experiment["call"]["arguments"] = argdict
        return thunk()
    return main


@tooled
def init_torch(
    # Seed to use for random numbers
    seed: Argument & int = default(1234),
    # Use CUDA for this model
    cuda: Argument & bool = default(None),
    # Number of threads for PyTorch
    workers: Argument & int = default(None),
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(workers or 1)

    if cuda is None:
        cuda = torch.cuda.is_available()

    if cuda:
        torch.cuda.manual_seed_all(seed)

    return NS(
        device=torch.device("cuda" if cuda else "cpu"),
        cuda=cuda,
        sync=torch.cuda.synchronize if cuda else None,
        workers=workers,
        seed=seed,
    )


@tooled
def iteration_wrapper(
    experiment,
    sync = None,
    # Maximum count before stopping
    max_count: Argument & int = default(1000),
    # Number of seconds for sampling items/second
    sample_duration: Argument & float = default(0.5),
):
    return experiment.chronos.create(
        "train",
        type="rate",
        sync=sync,
        sample_duration=sample_duration,
        max_count=max_count,
    )
