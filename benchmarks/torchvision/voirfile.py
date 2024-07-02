from dataclasses import dataclass

from voir import configurable
from voir.phase import StopProgram
from voir.instruments import dash, early_stop, log
from benchmate.observer import BenchObserver
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
    gpu_poll: int = 3


@configurable
def instrument_main(ov, options: Config):
    yield ov.phases.init

    if options.dash:
        ov.require(dash)

    overhead_metrics = [] # "__iter__", "overhead", "process_time"

    metrics = [
        "value", 
        "progress", 
        "rate", 
        "units", 
        "loss", 
        "gpudata",
        "iodata",
        "cpudata",
        "process"
    ]

    ov.require(
        log(*metrics, *overhead_metrics, context="task"),
        early_stop(n=options.stop, key="rate", task="train"),
        monitor_monogpu(poll_interval=options.gpu_poll),
    )

    yield ov.phases.load_script

    from benchmate.dataloader import imagenet_dataloader
    import torchcompat.core as accelerator
    from ptera import refstring
    
    # Note: the wrapper can also do early stopping, if raise_stop_program=True
    observer = BenchObserver(
        accelerator.Event, 
        earlystop=options.stop + options.skip,
        batch_size_fn=lambda x: len(x[0])
    )

    probe = ov.probe(f"{refstring(imagenet_dataloader)}() as loader", overridable=True)
    probe['loader'].override(observer.loader)

    probe = ov.probe("//train_epoch > criterion", overridable=True)
    probe['criterion'].override(observer.criterion)
    
    try:
        yield ov.phases.run_script
    except StopProgram:
        print("early stopped")
