# Import this to instrument the ArgumentParser, remove if no such thing
from milabench.opt import instrument_argparse
from deepspeed.profiling.flops_profiler import FlopsProfiler


class ModelCallWrapper:
    """
    Wrapper to replace model.__call__ method with a wrapper
    which must profile each call.
    """

    def __init__(self, model):
        cls = type(model)
        self.model = model
        self.original_call = cls.__call__
        cls.__call__ = self.call
        self.prof = FlopsProfiler(model)
        self.measures = []

    def call(self, *args, **kwargs):
        # Inspired from deepspeed get_model_profile() source code:
        # https://deepspeed.readthedocs.io/en/latest/_modules/deepspeed/profiling/flops_profiler/profiler.html#get_model_profile
        self.prof.start_profile()

        ret = self.original_call(self.model, *args, **kwargs)

        flops = self.prof.get_total_flops()
        macs = self.prof.get_total_macs()
        params = self.prof.get_total_params()
        self.measures.append((flops, macs, params))
        self.prof.end_profile()
        return ret

    def __str__(self):
        return f"COUNT {len(self.measures)}"


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

    def check(model):
        wrapper = ModelCallWrapper(model)
        ov.give(wrapper=wrapper)

    given_model = ov.given["?model"]
    given_model.subscribe(check)
    ov.given["?wrapper"].print()


def instrument_display_min(ov):
    yield ov.phases.init
    ov.given["?loss"].min().print("Minimum __ loss: {}")


def instrument_print_profiles(ov):
    yield ov.phases.run_script
    ov.given["?wrapper"].print("Wrapper: {}")


def instrument_print_profiles_2(ov):
    yield ov.phases.finalize
    ov.given["?wrapper"].print("Wrapper: {}")
