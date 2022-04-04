def instrument_argparse(ov):
    yield ov.phases.load_script
    ov.probe(
        "/argparse/ArgumentParser/parse_args() as options"
    ).first().allow_empty().give()
