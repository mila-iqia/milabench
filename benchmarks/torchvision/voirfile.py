from dataclasses import dataclass

from voir import configurable
from voir.instruments import dash, early_stop, gpu_monitor, log, rate
from voir.wrapper import DataloaderWrapper, Wrapper


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
    import torchcompat.core as accelerator

    yield ov.phases.init

    if options.dash:
        ov.require(dash)


    overhead_metrics = [] # "__iter__", "overhead", "process_time"

    ov.require(
        log("value", "progress", "rate", "units", "loss", "gpudata", *overhead_metrics, context="task"),
        early_stop(n=options.stop, key="rate", task="train"),
        gpu_monitor(poll_interval=options.gpu_poll),
    )

    yield ov.phases.load_script

    # Note: the wrapper can also do early stopping, if raise_stop_program=True
    wrapper = Wrapper(
        accelerator.Event, 
        earlystop=options.stop + options.skip
    )

    probe = ov.probe("//dataloader() as loader", overridable=True)
    probe['loader'].override(wrapper.loader)

    probe = ov.probe("//train_epoch > criterion", overridable=True)
    probe['criterion'].override(wrapper.criterion)
    


