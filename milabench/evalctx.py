import re
import os
import sys
from argparse import Namespace

from .gpu import get_gpu_info

EXPR_PATTERN = re.compile(r"\$\((?P<expr>.*)\)")

# Memory model config key
MMK = "mem"


def bs(gpu, mem, default=None, multi_gpu=False):
    """Assume memory consumption follow a linear relationship with batch size.
    The resulting batch size is always a multiple of 8 (by default).

    Arguments
    ---------
    gpu: Namespace(mem, count)
        GPU configuration, memory capacity and number of detected GPUs

    mem: Namespace(intercept, slope, multiple)
        Memory model parameters derived from a linear regression

    multi_gpu: bool
        if true the batch size is multiplied by the number of GPUs

    Notes
    -----
    Use ``MILABENCH_FIXED_BATCH`` to disable auto finding the batch size.
    
    """
    fixed_batch = os.getenv("MILABENCH_FIXED_BATCH", "False") in ("True", "1")
    
    assert default is not None, "Auto batch size is experimental, populate default as fallback"
    
    if fixed_batch:
        return default
    
    batch_size = (
        int((gpu.mem - mem.intercept) / mem.slope / mem.multiple) * mem.multiple
    )

    if multi_gpu:
        batch_size *= gpu.count

    return max(batch_size, 1)


def fetch_gpu_configuration(memory_override=None):
    """Retrieve the number of devices and minimum amount of
    available memory across all the devices.

    Notes
    -----

    The amount of available memory can be tweaked using
    the ``MILABENCH_GPU_MEM_LIMIT`` environment variable
    to mock a device with a smaller memory capacity.

    """
    infos = get_gpu_info()
    gpu = dict(mem=float("+inf"), count=len(infos))

    memory_override = float(os.getenv("MILABENCH_GPU_MEM_LIMIT", "+inf"))

    for value in infos.values():
        total = min(value["memory"]["total"], memory_override)
        
        avail = total - value["memory"]["used"]
        gpu["mem"] = min(gpu["mem"], avail)

    return gpu


class ArgumentResolver:
    """Evaluate an expression.

    This is used to derivate arguments from the hardware and configuration.
    """

    def __init__(self, run):
        self.run = run
        self.globals = dict()
        self.locals = dict()

    def _init_context(self):
        """Only initialize the context if it is required."""
        if len(self.locals) != 0:
            return

        gpu = fetch_gpu_configuration()

        mem_model = self.run.get(MMK)
        if mem_model is None:
            raise AttributeError(
                f"{self.run['name']} configuration does not provide a `{MMK}` configuration"
            )

        self.globals = dict()
        self.locals = {"bs": bs, MMK: Namespace(**mem_model), "gpu": Namespace(**gpu)}

    def resolve_argument(self, arg, context=None):
        """Extract the python expression from the argument and evaluate it"""
        if not isinstance(arg, str):
            return arg

        mt = EXPR_PATTERN.search(arg)

        if mt:
            self._init_context()
            expression = mt.groupdict()["expr"]
            arg = eval(expression, self.globals, self.locals)
            
            with open("batch.txt", 'a') as file:
                file.write(f"{self.run['name']}: {arg}\n")

        return str(arg)

    def resolve_arguments(self, args):
        """Modify the arguments inplace"""

        for key, arg in args.items():
            args[key] = self.resolve_argument(arg)

        return args
