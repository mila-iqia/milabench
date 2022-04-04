from milabench.opt import instrument_argparse


def instrument_probes(ov):
    yield ov.phases.load_script
    ov.probe("//train > loss").kmap(loss=lambda loss: float(loss)).give()
    ov.probe("//train(data as batch) > #endloop_data as step").give()
    ov.probe("//main > train_loader as loader").first().give()
    ov.probe(
        "//train(data as batch, !#loop_data as compute_start, !!#endloop_data as compute_end)"
    ).give()
