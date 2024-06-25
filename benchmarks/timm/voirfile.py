from dataclasses import dataclass

from voir import configurable
from voir.phase import StopProgram
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
    from benchmate.observer import BenchObserver

    from timm.utils.distributed import is_global_primary

    observer = BenchObserver(
        accelerator.Event, 
        earlystop=options.stop + options.skip,
        rank=int(os.getenv("RANK", 0)),
        device=accelerator.fetch_device(int(os.getenv("RANK", 0))),
        backward_callback=accelerator.mark_step,
        step_callback=accelerator.mark_step,
        batch_size_fn=lambda x: len(x[0])
    )

    probe = ov.probe("/timm.data.loader/create_loader() as loader", overridable=True)
    probe['loader'].override(observer.loader)

    probe = ov.probe("//train_one_epoch > loss_fn", overridable=True)
    probe['loss_fn'].override(observer.criterion)

    probe = ov.probe("//train_one_epoch > optimizer", overridable=True)
    probe['optimizer'].override(observer.optimizer)

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

    try:
        yield ov.phases.run_script
    except StopProgram:
        print("early stopped")