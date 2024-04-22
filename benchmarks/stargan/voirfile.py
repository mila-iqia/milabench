from dataclasses import dataclass

from voir import configurable
from voir.instruments import dash, early_stop, gpu_monitor, log, rate


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


def get_sync():
    import torch

    def has_xpu():
        try:
            import intel_extension_for_pytorch as ipex
            return torch.xpu.is_available()
        except ImportError as err:
            return True

    synchronize = None    
    if has_xpu():
        synchronize = torch.xpu.synchronize
    elif torch.cuda.is_available():
        synchronize = torch.cuda.synchronize

    return synchronize


@configurable
def instrument_main(ov, options: Config):
    sync = get_sync()

    yield ov.phases.init

    if options.dash:
        ov.require(dash)

    ov.require(
        log("value", "progress", "rate", "units", "loss", "gpudata", context="task"),
        rate(
            interval=options.interval,
            skip=options.skip,
            sync=sync,
        ),
        early_stop(n=options.stop, key="rate", task="train"),
        gpu_monitor(poll_interval=options.gpu_poll),
    )
