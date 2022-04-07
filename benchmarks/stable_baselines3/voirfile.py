# Import this to instrument the ArgumentParser, remove if no such thing
from milabench.opt import instrument_argparse


def instrument_probes(ov):
    # Probe for the necessary data. More information here:
    # https://breuleux.github.io/milabench/instrument.html#probing-for-milabench

    yield ov.phases.load_script

    #
    (
        ov.probe(
            "/stable_baselines3.ppo.ppo/PPO/train(self) > #endloop__ as step"
        )
        .augment(batch_size=lambda self: self.batch_size)
        .give()
    )

    # loss
    (
        ov.probe(
            "/stable_baselines3.ppo.ppo/PPO/train(self) > loss.item() as loss"
        )
        .give()
    )
