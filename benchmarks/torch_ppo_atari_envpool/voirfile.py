from dataclasses import dataclass

from voir import configurable
from voir.phase import StopProgram
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

    # GPU monitor, rate, loss etc...
    voirfile_monitor(ov, options)

    yield ov.phases.load_script
    
    step_per_iteration = 0
    
    def fetch_args(args):
        nonlocal step_per_iteration
        step_per_iteration = args.num_envs * args.num_steps
        return args
        
    def batch_size(x):
        return step_per_iteration

    observer = BenchObserver(
        earlystop=options.stop + options.skip,
        batch_size_fn=batch_size,
    )
    
    probe = ov.probe("//main > args", overridable=True)
    probe['args'].override(fetch_args)
    
    # measure the time it took to execute the body
    probe = ov.probe("//main > iterations", overridable=True)
    probe['iterations'].override(observer.loader)

    probe = ov.probe("//main > loss", overridable=True)
    probe["loss"].override(observer.record_loss)

    probe = ov.probe("//main > optimizer", overridable=True)
    probe['optimizer'].override(observer.optimizer)
    
    #
    # Run the benchmark
    #
    try:
        yield ov.phases.run_script
    except StopProgram:
        print("early stopped")