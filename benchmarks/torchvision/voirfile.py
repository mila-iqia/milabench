# Import this to instrument the ArgumentParser, remove if no such thing
from milabench.opt import instrument_argparse


def instrument_probes(ov):
    yield ov.phases.load_script
    ov.probe("//main > use_cuda").give()
    ov.probe("//main > model").give()
    ov.probe("//train_epoch > loss").throttle(1)["loss"].map(float).give("loss")
    ov.probe("//train_epoch > loader").give()
    ov.probe("//train_epoch(inp as batch) > #endloop_inp as step").give()
    ov.probe(
        "//train_epoch(inp as batch, !#loop_inp as compute_start, !!#endloop_inp as compute_end)"
    ).give()
