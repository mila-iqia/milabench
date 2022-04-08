import torch

# need to import it because it is not yet loaded once the main script is loaded
from stable_baselines3.ppo.ppo import PPO


def instrument_probes(ov):
    # Probe for the necessary data. More information here:
    # https://breuleux.github.io/milabench/instrument.html#probing-for-milabench

    yield ov.phases.load_script

    # Step + Batch Size + Device
    (
        ov.probe(
            "/stable_baselines3.ppo.ppo/PPO/train(self) > #endloop__ as step"
        )
        .augment(batch_size=lambda self: self.batch_size)
        .augment(use_cuda=lambda self: self.device.type == torch.device("cuda").type)
        .give()
    )

    # Loss
    (
        ov.probe(
            "/stable_baselines3.ppo.ppo/PPO/train > loss"
        )
        .throttle(1)["loss"]
        .map(float)
        .give("loss")
    )

    # Compute Start & End + Batch
    ov.probe(
        "/stable_baselines3.ppo.ppo/PPO/train(rollout_data as batch, !#loop_rollout_data as compute_start, !!#endloop_rollout_data as compute_end)"
    ).give()

