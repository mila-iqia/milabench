import contextvars
import os
from copy import deepcopy
from dataclasses import dataclass
import multiprocessing

import numpy as np
import yaml

from .config import is_autoscale_enabled, system_global
from .validation.validation import ValidationLayer

ROOT = os.path.dirname(__file__)

default_scaling_config = os.path.join(ROOT, "..", "config", "scaling.yaml")


def getenv(name, type):
    value = os.getenv(name)

    if value is not None:
        return type(value)

    return value


@dataclass
class SizerOptions:
    size: int = getenv("MILABENCH_SIZER_BATCH_SIZE", int)
    autoscale: bool = is_autoscale_enabled()
    multiple: int = getenv("MILABENCH_SIZER_MULTIPLE", int)
    optimized: bool = getenv("MILABENCH_SIZER_OPTIMIZED", int)
    capacity: str = getenv("MILABENCH_SIZER_CAPACITY", str)


metric_prefixes = {
    "T": (12, 4),
    "G": (9, 3),
    "M": (6, 2),
    "k": (3, 1),
    "h": (2, None),
    "da": (1, None),
    "d": (-1, None),
    "c": (-2, None),
    "m": (-3, None),
    "u": (-6, None),
    "n": (-9, None),
    "p": (-12, None),
}


def to_octet(value: str) -> float:
    for p, (vm, vb) in metric_prefixes.items():
        if f"{p}iB" in value or f"{p}io" in value:
            return float(value[: -(len(p) + 2)]) * 1024**vb

        if f"{p}B" in value or f"{p}o" in value:
            return float(value[: -(len(p) + 1)]) * 10**vm

    if "io" in value:
        return float(value.replace("io", ""))

    if "o" in value:
        return float(value.replace("o", ""))

    return float(value)


class Sizer:
    """Automatically scale the batch size to match GPU spec"""

    def __init__(self, options=SizerOptions(), scaling_config=None):
        self.options = options
        self.path = scaling_config

        if scaling_config is None:
            scaling_config = default_scaling_config

        with open(scaling_config, "r") as sconf:
            self.scaling_config = yaml.safe_load(sconf)

    def benchscaling(self, benchmark):
        # key
        if isinstance(benchmark, str):
            return self.scaling_config.get(benchmark)

        # benchmark config
        if isinstance(benchmark, dict) and "name" in benchmark:
            return benchmark

        # pack
        return self.scaling_config.get(benchmark.config["name"])

    def get_capacity(self, capacity):
        if self.options.capacity is not None:
            capacity = self.options.capacity

        if isinstance(capacity, str):
            capacity = to_octet(capacity)

        return capacity

    def auto_size(self, benchmark, capacity):
        capacity = self.get_capacity(capacity)

        if capacity is None:
            return None

        config = self.benchscaling(benchmark)

        data = list(sorted(config["model"].items(), key=lambda x: x[0]))
        mem = [to_octet(v[1]) for v in data]
        size = [float(v[0]) for v in data]

        # This does not extrapolate
        # int(np.interp(capacity, mem, size))

        # Use polynomial of degree 1 so it is essentially linear interpolation
        model = np.poly1d(np.polyfit(mem, size, deg=1))

        newsize_f = model(capacity)
        newsize_i = int(newsize_f)

        if (newsize_f - newsize_i) > 0.5:
            newsize_i += 1

        if self.options.multiple is not None:
            newsize_i = (newsize_i // self.options.multiple) * self.options.multiple

        return max(newsize_i, 1)

    def size(self, benchmark, capacity):
        config = self.benchscaling(benchmark)

        if self.options.size is not None:
            return self.options.size

        if self.options.optimized:
            return config["optimized"]

        if self.options.autoscale:
            return self.auto_size(benchmark, capacity)

        return None

    def argv(self, benchmark, capacity, argv):
        """Find the batch size and override it with a new value"""

        config = self.benchscaling(benchmark)
        if config is None:
            return argv

        newsize = self.size(benchmark, capacity)

        if newsize is None:
            return argv

        # <param> <value>
        argv = list(argv)
        argname = config.get("arg")
        if argname is None:
            return argv

        for i, arg in enumerate(argv):
            if arg.endswith(argname):
                break
        else:
            # add the new argument
            return argv + [argname, str(newsize)]

        argv[i + 1] = str(newsize)
        return argv


sizer_global = contextvars.ContextVar("sizer_global", default=Sizer())


def scale_argv(pack, argv):
    sizer = sizer_global.get()
    system = system_global.get()

    capacity = system.get("gpu", dict()).get("capacity")

    return sizer.argv(pack, capacity, argv)


class MemoryUsageExtractor(ValidationLayer):
    """Extract max memory usage per benchmark to populate the memory model"""

    def __init__(self):
        self.filepath = getenv("MILABENCH_SIZER_SAVE", str)

        self.memory = deepcopy(sizer_global.get().scaling_config)
        self.scaling = None
        self.benchname = None
        self.batch_size = 0
        self.max_usage = float("-inf")
        self.early_stopped = False

    def on_start(self, entry):
        if self.filepath is None:
            return

        argv = entry.data["command"]
        self.benchname = entry.pack.config["name"]
        self.batch_size = None
        self.max_usage = float("-inf")

        config = self.memory.get(self.benchname, dict())
        scalingarg = config.get("arg", None)

        if scalingarg is None:
            self.benchname = None
            return

        found = None
        for i, arg in enumerate(argv):
            if arg.endswith(scalingarg):
                found = i
                break

        if found:
            self.batch_size = int(argv[found + 1])

    def on_data(self, entry):
        if self.filepath is None:
            return

        if entry.data is None:
            return

        gpudata = entry.data.get("gpudata")
        if gpudata is not None:
            current_usage = []
            for device, data in gpudata.items():
                usage, total = data.get("memory", [0, 1])
                current_usage.append(usage)

            self.max_usage = max(*current_usage, self.max_usage)

    def on_stop(self, entry):
        self.early_stopped = True

    def on_end(self, entry):
        if self.filepath is None:
            return

        if (
            self.benchname is None
            or self.batch_size is None
            or self.max_usage == float("-inf")
        ):
            return

        # Only update is successful
        rc = entry.data["return_code"]
        if rc == 0 or self.early_stopped:
            config = self.memory.setdefault(self.benchname, dict())
            model = config.setdefault("model", dict())
            model[self.batch_size] = f"{self.max_usage} MiB"
            config["model"] = dict(sorted(model.items(), key=lambda x: x[0]))

        self.benchname = None
        self.batch_size = None
        self.max_usage = float("-inf")

    def report(self, *args):
        if self.filepath is not None:
            with open(self.filepath, "w") as file:
                yaml.dump(self.memory, file)


def resolve_argv(pack, argv):
    context = system_global.get()
    arch = context.get("arch", "cpu")

    device_count = len(pack.config.get("devices", [0]))

    ccl = {
        "hpu": "hccl",
        "cuda": "nccl",
        "rocm": "rccl",
        "xpu": "ccl",
        "cpu": "gloo"
    }

    context["arch"] = arch
    context["ccl"] = ccl.get(arch, "gloo")
    context["cpu_count"] = multiprocessing.cpu_count()
    context["cpu_per_gpu"] = (multiprocessing.cpu_count() - 1) // device_count

    argv = list(argv)
    for i, arg in enumerate(argv):
        argv[i] = str(arg).format(**context)

    return argv