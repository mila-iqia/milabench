# Import this to instrument the ArgumentParser, remove if no such thing
from milabench.opt import instrument_argparse
from deepspeed.profiling.flops_profiler import FlopsProfiler
import torch.profiler as profiler
import torch


def _profile_with_deepspeed(**kwargs):
    """
    Use deepspeed profiler to profile model() call.
    Fast, but raises a RecursionError
    ("maximum recursion depth exceeded while calling a Python object")
    for big data sizes (--data-size >= 976)
    """
    # Reference:
    # https://www.deepspeed.ai/tutorials/flops-profiler/#example-training-workflow
    model = kwargs["model"]
    prof = FlopsProfiler(model)

    prof.start_profile()
    yield
    prof.stop_profile()

    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    prof.end_profile()
    print(flops, macs, params)

    return flops, macs, params


def _profile_with_torch_profiler():
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
        yield
        torch.cuda.synchronize()
    evts = prof.key_averages()
    # evts is a torch event list:
    # https://github.com/pytorch/pytorch/blob/9905b1f29a5d23d30cfb570679845766983260c4/torch/autograd/profiler_util.py#L14
    # Each event is an instance of torch.autograd.profiler_util.FunctionEventAvg:
    # https://github.com/pytorch/pytorch/blob/9905b1f29a5d23d30cfb570679845766983260c4/torch/autograd/profiler_util.py#L521
    print(sum(evt.flops for evt in evts), len(evts), type(evts), type(evts[0]))
    return evts


def instrument_probes(ov):
    yield ov.phases.load_script
    ov.probe("//main > use_cuda").give()
    ov.probe("//main > model").give()
    ov.probe("//train_epoch > loss").throttle(1)["loss"].map(float).give("loss")
    ov.probe("//train_epoch > loader").give()
    ov.probe("//train_epoch(inp as batch) > #endloop_inp as step").give()
    ov.probe(
        "//train_epoch(inp as batch, model, !#loop_inp as compute_start, !!#endloop_inp as compute_end)"
    ).give()

    ov.given.wmap("compute_start", _profile_with_deepspeed).give("profiler_stats")


def instrument_print_profiles(ov):
    cells = ov.given["?profiler_stats"].accum()
    yield ov.phases.run_script
    print(len(cells), type(cells[0]))
