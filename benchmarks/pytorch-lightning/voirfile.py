# Import this to instrument the ArgumentParser, remove if no such thing
from voir.overseer import Overseer

from milabench.opt import instrument_argparse


def instrument_probes(ov: Overseer):
    # Probe for the necessary data. More information here:
    # https://breuleux.github.io/milabench/instrument.html#probing-for-milabench

    yield ov.phases.load_script

    # loss
    ...
    # ov.probe("//Model/training_step_end > loss").give()

    # batch + step
    ...

    # use_cuda
    ...

    # model
    ...

    # loader
    ...

    # batch + compute_start + compute_end
    ...


# def instrument_display_min(ov):
#     yield ov.phases.init
#     ov.given["?loss"].min().print("Minimum loss: {}")
