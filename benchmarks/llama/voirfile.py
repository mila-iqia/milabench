from voir.phase import StopProgram
from voir import configurable
from voir.overseer import Overseer


@configurable
def instrument_main(ov: Overseer):
    yield ov.phases.load_script

    try:
        yield ov.phases.run_script
    except StopProgram:
        print("early stopped")