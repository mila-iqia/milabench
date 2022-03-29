
def instrument_probes(ov):
    yield ov.phases.load_script
    ov.probe("//train > loss").kmap(loss=lambda loss: float(loss)).give()
    ov.probe("//train(data as batch) > #endloop_data as step").give()
    ov.probe("//main > train_loader as loader").first().give()
