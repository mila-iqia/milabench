from dataclasses import dataclass
import os

from voir import configurable
from voir.phase import StopProgram
from voir.instruments import dash, early_stop, gpu_monitor, log, rate
from benchmate.monitor import monitor_monogpu, log_patterns


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
    # import torchcompat.core as accelerator
    # from benchmate.observer import BenchObserver

    yield ov.phases.init

    if options.dash:
        ov.require(dash)

    ov.require(
        log(*log_patterns(), context="task"),
        early_stop(n=options.stop, key="rate", task="train"),
        monitor_monogpu(poll_interval=options.gpu_poll),
    )

    os.environ["VOIR_EARLYSTOP_COUNT"] = str(options.stop)
    os.environ["VOIR_EARLYSTOP_SKIP"] = str(options.skip)

    try:
        yield ov.phases.run_script
    except StopProgram:
        print("early stopped")