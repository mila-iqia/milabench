from dataclasses import dataclass

from voir.phase import StopProgram
from voir import configurable
from benchmate.monitor import voirfile_monitor
from benchmate.observer import BenchObserver


@dataclass
class Config:
    """voir configuration"""

    # Whether to display the dash or not
    dash: bool = False

    # How often to log the rates
    interval: str = "1s"

    # Number of rates to skip before logging
    skip: int = 5

    # Number of rates to log before stopping
    stop: int = 20

    # Number of seconds between each gpu poll
    gpu_poll: int = 1


@configurable
def instrument_main(ov, options: Config):
    yield ov.phases.init


    yield ov.phases.load_script

    voirfile_monitor(ov, options)

    import ptera
    from operator_learning.data import getDataLoaders
    from operator_learning.utils.misc import get_loss_fn

    rfstr = ptera.refstring(getDataLoaders)

    def get_batch(x):
        return x[0].shape[0]
    #
    # Insert milabench tools
    #
    observer = BenchObserver(
        earlystop=options.stop + options.skip,
        batch_size_fn=get_batch
    )

    probe = ov.probe(f"{rfstr}() as loader", overridable=True)
    probe['loader'].override(lambda x: (observer.loader(x[0]), x[1], x[2]))

    probe = ov.probe(f"{ptera.refstring(get_loss_fn)}() as criterion", overridable=True)
    probe['criterion'].override(observer.criterion)

    #probe = ov.probe("//my_optimizer_creator() as optimizer", overridable=True)
    #probe['optimizer'].override(observer.optimizer)

    #
    # Run the benchmark
    #
    try:
        yield ov.phases.run_script
    except StopProgram:
        print("early stopped")