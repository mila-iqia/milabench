from dataclasses import dataclass

from voir.phase import StopProgram
from voir import configurable
from voir.instruments import dash, early_stop, log
from benchmate.monitor import monitor_monogpu
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
    gpu_poll: int = 3


@configurable
def instrument_main(ov, options: Config):
    yield ov.phases.init


    yield ov.phases.load_script

    if options.dash:
        ov.require(dash)

    ov.require(
        log("value", "progress", "rate", "units", "loss", "gpudata", context="task"),
        early_stop(n=options.stop, key="rate", task="train"),
        monitor_monogpu(poll_interval=options.gpu_poll),
    )

    #
    # Insert milabench tools
    #
    observer = BenchObserver(
        earlystop=options.stop + options.skip,
        batch_size_fn=lambda x: 1
    )

    probe = ov.probe("//my_dataloader_creator() as loader", overridable=True)
    probe['loader'].override(observer.loader)

    probe = ov.probe("//my_criterion_creator() as criterion", overridable=True)
    probe['criterion'].override(observer.criterion)

    probe = ov.probe("//my_optimizer_creator() as optimizer", overridable=True)
    probe['optimizer'].override(observer.optimizer)
    
    #
    # Run the benchmark
    #
    try:
        yield ov.phases.run_script
    except StopProgram:
        print("early stopped")