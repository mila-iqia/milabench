from dataclasses import dataclass

from voir import configurable
from voir.instruments import dash, early_stop, log, rate
from benchmate.monitor import log_patterns


@dataclass
class Config:
    """voir configuration"""

    # Whether to display the dash or not
    dash: bool = False

    # How often to log the rates
    interval: str = "1"

    # Number of rates to skip before logging
    skip: int = 5

    # Number of rates to log before stopping
    stop: int = 20


@configurable
def instrument_main(ov, options: Config):
    yield ov.phases.init

    if options.dash:
        ov.require(dash)

    ov.require(
        log(*log_patterns(), context="task"),
        rate(
            interval=options.interval,
            skip=options.skip,
            sync=None,
        ),
        early_stop(n=options.stop, key="rate", task="train"),
    )
