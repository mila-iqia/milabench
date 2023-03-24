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

    # Number of rates to log before stopping
    stop: int = 20


@configurable
def instrument_main(ov, options: Config):
    import torch

    yield ov.phases.init

    if options.dash:
        ov.require(dash)

    ov.require(
        log("value", "progress", "rate", "units", "loss", "gpudata", context="task"),
        rate(
            interval=options.interval,
            sync=torch.cuda.synchronize if torch.cuda.is_available() else None,
        ),
        early_stop(n=options.stop, key="rate", task="train"),
        gpu_monitor(poll_interval=3),
    )

    yield ov.phases.load_script

    # PPO

    # Loss
    (
        ov.probe("/stable_baselines3.ppo.ppo/PPO/train > loss")
        .throttle(1)["loss"]
        .map(float)
        .give("loss")
    )

    # Compute Start & End + Batch
    ov.probe(
        "/stable_baselines3.ppo.ppo/PPO/train(rollout_data as batch, !#loop_rollout_data as step, !!#endloop_rollout_data as step_end)"
    ).augment(task=lambda: "train").give()

    # TD3

    # Loss
    (
        ov.probe("/stable_baselines3.td3.td3/TD3/train > actor_loss")
        .throttle(1)["actor_loss"]
        .map(float)
        .give("loss")
    )

    # Compute Start & End + Batch
    ov.probe(
        "/stable_baselines3.td3.td3/TD3/train(_ as batch, !#loop__ as step, !!#endloop__ as step_end)"
    ).augment(task=lambda: "train").give()
