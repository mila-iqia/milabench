import re
import os
from math import log2, isinf
from argparse import Namespace

from voir.instruments.gpu import get_gpu_info

EXPR_PATTERN = re.compile(r"\$\((?P<expr>.*)\)")

# Memory model config key
MMK = "mem"


def deduce_good_batch_size(estimate: float, method: str, multiplier: int = 8):
    """Find the closest batch size that match the target batch size"""

    if method == "multiple":
        return int(estimate / multiplier) * multiplier

    elif method == "power2":
        return 2 ** int(log2(estimate))

    return int(estimate)



def default_batch_size(gpu, mem, default=None, multi_gpu=False):
    return default_batch_size


def bs(gpu, mem, default=None, multi_gpu=False):
    """Assume memory consumption follow a linear relationship with batch size.
    The resulting batch size is always a multiple of 8 (by default).

    Arguments
    ---------
    gpu: Namespace(mem, count)
        GPU configuration, memory capacity and number of detected GPUs

    mem: Namespace(intercept, slope, multiple, method)
        Memory model parameters derived from a linear regression
        method is used to

    multi_gpu: bool
        if true the batch size is multiplied by the number of GPUs

    Notes
    -----
    Use ``MILABENCH_FIXED_BATCH`` to disable auto finding the batch size.

    """
    fixed_batch = os.getenv("MILABENCH_FIXED_BATCH", "False") in ("True", "1")

    assert (
        default is not None
    ), "Auto batch size is experimental, populate default as fallback"

    if fixed_batch:
        return default

    estimate = (gpu.mem - mem.intercept) / mem.slope

    batch_size = deduce_good_batch_size(estimate, "multiple", mem.multiple)

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
    infos = get_gpu_info()["gpus"]
    gpu = dict(mem=float("+inf"), count=len(infos))

    memory_override = float(os.getenv("MILABENCH_GPU_MEM_LIMIT", "+inf"))

    for value in infos.values():
        total = min(value["memory"]["total"], memory_override)

        avail = total - value["memory"]["used"]
        gpu["mem"] = min(gpu["mem"], avail)

    return gpu


def fetch_cpu_configuration():
    """Returns the CPU configuration"""
    return {
        "count": os.cpu_count(),
    }


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
        cpu = fetch_cpu_configuration()
        resolver = bs
        
        if isinf(gpu['mem']):
            print("Could not configure batch size")
            resolver = default_batch_size

        mem_model = self.run.get(MMK)
        if mem_model is None:
            raise AttributeError(
                f"{self.run['name']} configuration does not provide a `{MMK}` configuration"
            )

        self.globals = dict()
        self.locals = {
            "bs": resolver,
            MMK: Namespace(**mem_model),
            "gpu": Namespace(**gpu),
            "cpu": Namespace(**cpu),
        }

    def resolve_argument(self, arg, context=None):
        """Extract the python expression from the argument and evaluate it"""
        if not isinstance(arg, str):
            return arg

        mt = EXPR_PATTERN.search(arg)

        if mt:
            self._init_context()
            expression = mt.groupdict()["expr"]
            arg = eval(expression, self.globals, self.locals)

            # dump resolved batch sizes
            # with open("batch.txt", 'a') as file:
            #    file.write(f"{self.run['name']}: {arg}\n")

        return str(arg)

    def resolve_arguments(self, args):
        """Modify the arguments inplace"""

        for key, arg in args.items():
            args[key] = self.resolve_argument(arg)

        return args
