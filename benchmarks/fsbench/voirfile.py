# Import this to instrument the ArgumentParser, remove if no such thing
from milabench.opt import instrument_argparse


def instrument_probes(ov):
    # Probe for the necessary data. More information here:
    # https://breuleux.github.io/milabench/instrument.html#probing-for-milabench

    yield ov.phases.load_script

    # loss
    ...

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
