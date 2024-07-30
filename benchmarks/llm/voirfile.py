from dataclasses import dataclass

from voir.phase import StopProgram
from voir import configurable
from benchmate.observer import BenchObserver
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
    gpu_poll: int = 3


@configurable
def instrument_main(ov, options: Config):
    yield ov.phases.init

    yield ov.phases.load_script

    voirfile_monitor(ov, options)

    def batch_size(x):
        return x["tokens"].shape[0]

    observer = BenchObserver(
        earlystop=options.stop + options.skip,
        batch_size_fn=batch_size,
    )

    def wrap_dataloader(args):
        sampler, loader = args
        wrapped = observer.loader(loader, custom_step=True)
        return sampler, wrapped
    
    def wrap_lr_scheduler(scheduler):
        original = scheduler.step

        def newstep(*args, **kwargs):
            original(*args, **kwargs)
            observer.step()

        scheduler.step = newstep
        return scheduler

    probe = ov.probe("//LoRAFinetuneRecipeSingleDevice/_setup_data() as loader", overridable=True)
    probe['loader'].override(wrap_dataloader)

    probe = ov.probe("//LoRAFinetuneRecipeSingleDevice/_setup_lr_scheduler() as scheduler", overridable=True)
    probe['scheduler'].override(wrap_lr_scheduler)

    try:
        yield ov.phases.run_script
    except StopProgram:
        print("early stopped")