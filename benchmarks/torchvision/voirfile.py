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
        self.prof = FlopsProfiler(self.model)
        self.measures = []
        cls.__call__ = self.call

    def call(self, *args, **kwargs):
        # Inspired from deepspeed get_model_profile() source code:
        # https://www.deepspeed.ai/tutorials/flops-profiler/#example-training-workflow
        self.prof.start_profile()
        ret = self.original_call(self.model, *args, **kwargs)
        self.prof.stop_profile()

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
    wrapper_cell = ov.given["?wrapper"].accum()
    yield ov.phases.run_script
    wrapper, = wrapper_cell
    print("Wrapper", wrapper)
