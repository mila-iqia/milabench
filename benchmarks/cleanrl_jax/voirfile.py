from dataclasses import dataclass

from voir import configurable
from benchmate.monitor import voirfile_monitor

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
    gpu_poll: float = 1


@configurable
def instrument_main(ov, options: Config):
    yield ov.phases.init

    voirfile_monitor(ov, options)
