from dataclasses import dataclass

from voir import configurable
from voir.phase import StopProgram
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
    gpu_poll: int = 1


@configurable
def instrument_main(ov, options: Config):
    import torchcompat.core as accelerator

    yield ov.phases.init

    voirfile_monitor(ov, options)

    yield ov.phases.load_script

    # Loss
    (ov.probe("//run > L").throttle(1)["L"].map(float).give("loss"))

    # Compute Start & End + Batch
    ov.probe(
        "//run(inputBatch as batch, !#loop_inputBatch as step, !!#endloop_inputBatch as step_end)"
    ).augment(task=lambda: "train").give()

    try:
        yield ov.phases.run_script
    except StopProgram:
        print("early stopped")