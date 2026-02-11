from voir.phase import StopProgram
from voir import configurable


@configurable
def instrument_main(ov):
    try:
        yield ov.phases.run_script
    except StopProgram:
        print("early stopped")