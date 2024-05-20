from dataclasses import dataclass

from voir import configurable
from voir.instruments import dash, early_stop, gpu_monitor, log, rate

import torchcompat.core as accelerator

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

    yield ov.phases.load_script

    import os
    import torchcompat.core as accelerator
    from voir.wrapper import DataloaderWrapper, Wrapper

    from timm.utils.distributed import is_global_primary
    from timm.data import create_loader

    wrapper = Wrapper(
        accelerator.Event, 
        earlystop=options.stop + 20,
        rank=int(os.getenv("RANK", 0)),
        device=accelerator.fetch_device(int(os.getenv("RANK", 0))),
        backward_callback=accelerator.mark_step,
        step_callback=accelerator.mark_step
    )

    probe = ov.probe("/timm.data.loader/create_loader() as loader", overridable=True)
    probe['loader'].override(wrapper.loader)

    probe = ov.probe("//train_one_epoch > loss_fn", overridable=True)
    probe['loss_fn'].override(wrapper.criterion)

    probe = ov.probe("//train_one_epoch > optimizer", overridable=True)
    probe['optimizer'].override(wrapper.optimizer)

    # Do not save checkpoints
    probe = ov.probe("//main > saver", overridable=True)
    probe['saver'].override(lambda save: None)

    instruments = [
        log(
            "value", "progress", "rate", "units", "loss", "gpudata", context="task"
        ),
        gpu_monitor(poll_interval=options.gpu_poll),
    ] 

    if is_global_primary:
        instruments.append(early_stop(n=options.stop, key="rate", task="train", signal="stop"))

    ov.require(*instruments)
