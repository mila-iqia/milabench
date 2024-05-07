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
    import torchcompat.core as accelerator
    return accelerator.synchronize


@configurable
def instrument_main(ov, options: Config):
    def setup(args):
        sync = get_sync()

        if options.dash:
            ov.require(dash)

        ov.require(
            log(
                "value", "progress", "rate", "units", "loss", "gpudata", context="task"
            ),
            rate(
                interval=options.interval,
                skip=options.skip,
                sync=sync,
                batch_size_calc=lambda b: len(b) * args.world_size,
            ),
            early_stop(n=options.stop, key="rate", task="train", signal="stop"),
            gpu_monitor(poll_interval=options.gpu_poll),
        )

        # Loss
        (
            loss_probe.throttle(1)["loss"]
            .map(lambda loss: {"task": "train", "loss": float(loss)})
            .give()
        )

        # Compute Start & End + Batch
        batch_probe.augment(task=lambda: "train").give()

    yield ov.phases.load_script

    from timm.utils.distributed import is_global_primary

    # Loss
    loss_probe = ov.probe("//train_one_epoch > loss")

    # Compute Start & End + Batch
    batch_probe = ov.probe(
        "//train_one_epoch(input as batch, !#loop_batch_idx as step, !!#endloop_batch_idx as step_end)"
    )

    ov.probe("//main(args) > device")["args"].filter(is_global_primary).subscribe(setup)
