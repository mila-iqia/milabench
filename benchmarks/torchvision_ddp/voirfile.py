from voir.phase import StopProgram


@configurable
def instrument_main(ov):
    try:
        yield ov.phases.run_script
    except StopProgram:
        print("early stopped")