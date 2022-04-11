import torch

# need to import it because it is not yet loaded once the main script is loaded
from stable_baselines3.ppo.ppo import PPO


def instrument_probes(ov):
    # Probe for the necessary data. More information here:
    # https://breuleux.github.io/milabench/instrument.html#probing-for-milabench

    yield ov.phases.load_script

    # PPO

    # Get device
    ov.probe(
        "/stable_baselines3.ppo.ppo/PPO/train > self"
    ).kmap(use_cuda=lambda self:self.device.type == torch.device("cuda").type)

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
        "/stable_baselines3.ppo.ppo/PPO/train(rollout_data as batch, !#loop_rollout_data as compute_start, !!#endloop_rollout_data as compute_end, #endloop_rollout_data as step)"
    ).give()


    # TD3
    # Loss
    (ov.probe("/stable_baselines3.td3.td3/TD3/train > actor_loss").throttle(1)["actor_loss"]
    .map(float)
    .give("loss"))

    # Compute Start & End + Batch
    ov.probe(
        "/stable_baselines3.td3.td3/TD3/train(_ as batch, !#loop__ as compute_start, !!#endloop__ as compute_end, #endloop__ as step)"
    ).give()
