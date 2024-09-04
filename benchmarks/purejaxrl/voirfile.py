from dataclasses import dataclass

from voir import configurable
from voir.instruments import dash, early_stop, log, rate
from benchmate.monitor import monitor_monogpu

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
    gpu_poll: int = 0.5


@configurable
def instrument_main(ov, options: Config):
    yield ov.phases.init

    if options.dash:
        ov.require(dash)

    ov.require(
        log("value", "progress", "rate", "units", "loss", "gpudata", context="task"),
        # early_stop(n=options.stop, key="rate", task="train"),
        monitor_monogpu(poll_interval=options.gpu_poll),
    )
