# Import this to instrument the ArgumentParser, remove if no such thing
from milabench.opt import instrument_argparse
from deepspeed.profiling.flops_profiler import FlopsProfiler
import torch.profiler as profiler
import torch


class ModelCallWrapper:
    """
    Wrapper to replace model.__call__ method with a wrapper
    which must profile each call.
    """

    def __init__(self, model):
        cls = type(model)
        self.model = model
        self.original_call = cls.__call__
        self.measures = []
        cls.__call__ = self.call_torch_profiler

    def call_deepspeed(self, *args, **kwargs):
        """
        Use deepspeed profiler to profile model() call.
        Fast, but raises a RecursionError
        ("maximum recursion depth exceeded while calling a Python object")
        for big data sizes (--data-size >= 976)
        """
        # Reference:
        # https://www.deepspeed.ai/tutorials/flops-profiler/#example-training-workflow
        prof = FlopsProfiler(self.model)

        prof.start_profile()
        ret = self.original_call(self.model, *args, **kwargs)
        prof.stop_profile()

        flops = prof.get_total_flops()
        macs = prof.get_total_macs()
        params = prof.get_total_params()
        self.measures.append((flops, macs, params))
        prof.end_profile()
        print(len(self.measures), flops, macs, params)

        return ret

    def call_torch_profiler(self, *args, **kwargs):
        """
        Use torch.profiler to profile model() call.
        Slow, but no bugs.
        """
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            # schedule=profiler.schedule(wait=0, warmup=0, active=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) as prof:
            ret = self.original_call(self.model, *args, **kwargs)
            torch.cuda.synchronize()
        evts = prof.key_averages()
        # evts is a torch event list:
        # https://github.com/pytorch/pytorch/blob/9905b1f29a5d23d30cfb570679845766983260c4/torch/autograd/profiler_util.py#L14
        # Each event is an instance of torch.autograd.profiler_util.FunctionEventAvg:
        # https://github.com/pytorch/pytorch/blob/9905b1f29a5d23d30cfb570679845766983260c4/torch/autograd/profiler_util.py#L521
        self.measures.append(evts)
        print(len(self.measures), sum(evt.flops for evt in evts), len(evts), type(evts), type(evts[0]))
        return ret


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


def instrument_print_profiles(ov):
    wrapper_cell = ov.given["?wrapper"].accum()
    yield ov.phases.run_script
    wrapper: ModelCallWrapper = wrapper_cell[0]
    print("Wrapper nb. calls", len(wrapper.measures))
